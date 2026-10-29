"""Dataset download, EDF reading, channel standardization, and label mapping.

Dataset: PhysioNet EEG Motor Movement/Imagery Dataset v1.0.0 (EEGMMIDB),
accessed through MNE's :mod:`mne.datasets.eegbci` loader.

Task contract (guide section 4):
- LEFT vs RIGHT *imagined* fists use runs 4, 8, 12 only.
- Runs 6, 10, 14 are HANDS vs FEET imagery and must never be relabeled.
- Annotation descriptions are mapped explicitly: T1 -> left (0), T2 -> right (1).
  T0 is rest and is excluded from classifier training. Numeric event IDs that
  MNE assigns automatically are never used as labels.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from mne.datasets import eegbci

from .config import Config

LR_IMAGERY_RUNS = (4, 8, 12)
HANDS_FEET_IMAGERY_RUNS = (6, 10, 14)
EXECUTION_RUNS = (3, 5, 7, 9, 11, 13)
BASELINE_RUNS = (1, 2)


class DataError(RuntimeError):
    """Raised for missing/incorrect data that requires a human decision."""


@dataclass(frozen=True)
class RunEvents:
    """Annotation-derived events for one EDF run, with explicit labels."""

    subject_id: int
    run_id: int
    samples: np.ndarray  # onset sample index within the run, int64
    trial_ids: np.ndarray  # 1-based index within the run, in onset order
    labels: np.ndarray  # 0 = left, 1 = right
    descriptions: np.ndarray  # original annotation descriptions (T1/T2)
    sfreq_hz: float
    n_samples_total: int
    excluded_rest_count: int  # number of T0 annotations skipped


def check_run_supported(run_id: int) -> None:
    """Reject runs outside the left/right motor-imagery task."""
    if run_id not in LR_IMAGERY_RUNS:
        if run_id in HANDS_FEET_IMAGERY_RUNS:
            raise DataError(
                f"run {run_id} is a HANDS vs FEET imagery run (6/10/14) and "
                "must not be relabeled left/right; use runs 4, 8, 12"
            )
        raise DataError(
            f"run {run_id} is not a left/right motor-imagery run; "
            "supported runs are 4, 8, 12"
        )


def download_data(
    subjects: list[int], runs: list[int], path: str | Path, seed: int | None = None
) -> list[Path]:
    """Download (or reuse local copies of) EEGMMIDB EDF files via MNE."""
    for run in runs:
        check_run_supported(run)
    paths = eegbci.load_data(
        subjects=subjects, runs=runs, path=str(path), update_path=False
    )
    return [Path(p) for p in paths]


def _label_from_description(description: str, label_map: dict[str, int]) -> int:
    if description not in label_map:
        raise DataError(
            f"annotation description {description!r} has no explicit mapping "
            f"in data.label_map ({sorted(label_map)}); refusing to guess labels"
        )
    return label_map[description]


def load_run_raw(
    edf_path: str | Path, channels: tuple[str, ...], sfreq_hz: float
) -> mne.io.Raw:
    """Load one EDF run, standardize channel names, validate the contract.

    Returns the raw object in volts (MNE convention). Channel selection and
    filtering happen later, in :mod:`eeg_controller.preprocess`.
    """
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="WARNING")
    eegbci.standardize(raw)  # in-place: e.g. 'Fc5.' -> 'FC5'

    missing = [ch for ch in channels if ch not in raw.ch_names]
    if missing:
        raise DataError(
            f"{Path(edf_path).name}: required channel(s) {missing} not found in "
            f"EDF (available: {raw.ch_names}); refusing to substitute channels"
        )
    actual = raw.info["sfreq"]
    if not np.isclose(actual, sfreq_hz, rtol=0, atol=1e-9):
        raise DataError(
            f"{Path(edf_path).name}: sampling rate {actual} Hz != expected "
            f"{sfreq_hz} Hz; this contract does not resample"
        )
    return raw


def extract_events(raw: mne.io.Raw, cfg: Config, run_id: int) -> RunEvents:
    """Map annotation *descriptions* to left/right labels for one run.

    T0 (rest) annotations are counted and skipped, never relabeled.
    """
    check_run_supported(run_id)
    descriptions = raw.annotations.description
    onsets = raw.annotations.onset
    sfreq = float(raw.info["sfreq"])

    keep_samples, keep_labels, keep_desc = [], [], []
    rest_count = 0
    for onset_s, desc in zip(onsets, descriptions):
        desc = str(desc).strip()
        if desc == cfg.data.rest_description:
            rest_count += 1
            continue
        keep_labels.append(_label_from_description(desc, cfg.data.label_map))
        # Annotation onset includes Raw.first_time, including on cropped
        # recordings; convert to an index relative to get_data() sample 0.
        keep_samples.append(int(round((onset_s - raw.first_time) * sfreq)))
        keep_desc.append(desc)

    if not keep_labels:
        raise DataError(
            f"run {run_id} contains no T1/T2 annotations; inspect the EDF "
            "annotations before proceeding"
        )
    samples = np.asarray(keep_samples, dtype=np.int64)
    order = np.argsort(samples, kind="stable")
    samples = samples[order]
    labels = np.asarray(keep_labels, dtype=np.int64)[order]
    keep_desc_arr = np.asarray(keep_desc, dtype=object)[order]
    trial_ids = np.arange(1, len(samples) + 1, dtype=np.int64)

    counts = {0: int((labels == 0).sum()), 1: int((labels == 1).sum())}
    print(
        f"  run {run_id}: {len(samples)} left/right events "
        f"(left={counts[0]}, right={counts[1]}), {rest_count} rest skipped"
    )
    return RunEvents(
        subject_id=-1,  # filled by caller
        run_id=run_id,
        samples=samples,
        trial_ids=trial_ids,
        labels=labels,
        descriptions=keep_desc_arr,
        sfreq_hz=sfreq,
        n_samples_total=int(raw.n_times),
        excluded_rest_count=rest_count,
    )


def hash_file(path: str | Path, algorithm: str = "sha256") -> str:
    """SHA-256 (or other) hash of a file, for provenance metadata."""
    h = hashlib.new(algorithm)
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_trial_table(
    cfg: Config,
    events_by_run: dict[int, RunEvents],
    edf_paths: dict[int, Path],
    subject_id: int,
) -> pd.DataFrame:
    """Persist subject_id, run_id, event sample, trial_id, label per trial."""
    rows = []
    for run_id in sorted(events_by_run):
        ev = events_by_run[run_id]
        for i in range(len(ev.samples)):
            rows.append(
                {
                    "subject_id": subject_id,
                    "run_id": run_id,
                    "event_sample": int(ev.samples[i]),
                    "trial_id": int(ev.trial_ids[i]),
                    "label": int(ev.labels[i]),
                    "description": str(ev.descriptions[i]),
                    "edf_file": Path(edf_paths[run_id]).name,
                    "edf_sha256": hash_file(edf_paths[run_id]),
                }
            )
    return pd.DataFrame(rows)
