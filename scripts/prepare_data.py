"""Download EEGMMIDB runs 4/8/12, preprocess per run, persist trial table.

Usage:
    python scripts/prepare_data.py --subjects 1 2 3 --config configs/default.json

Outputs (under --data-dir, default ./data):
    data/raw/                        EDF files (via MNE eegbci loader)
    data/trials_subj<id>.csv         trial table with hashes + run/label
    data/exclusions_subj<id>.csv     exclusion counts by run and class
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from eeg_controller.config import load_config
from eeg_controller.data import LR_IMAGERY_RUNS, download_data, hash_file
from eeg_controller.preprocess import load_and_preprocess_run

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subjects", type=int, nargs="+", required=True)
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default.json"))
    ap.add_argument("--data-dir", default=str(PROJECT_ROOT / "data"))
    args = ap.parse_args()

    cfg = load_config(args.config)
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    subjects = sorted(set(args.subjects))
    print(f"Subjects: {subjects}; runs: {LR_IMAGERY_RUNS}")

    paths = download_data(subjects, list(LR_IMAGERY_RUNS), data_dir / "raw")
    print(f"{len(paths)} EDF file(s) available locally")

    trial_frames, exclusion_frames = [], []
    for subject_id in subjects:
        print(f"Subject {subject_id}:")
        run_paths = {
            run: next(p for p in paths if p.name.startswith(f"S{subject_id:03d}R{run:02d}"))
            for run in LR_IMAGERY_RUNS
        }
        for run in LR_IMAGERY_RUNS:
            print(f"  run {run}: {Path(run_paths[run]).name}")
        epochs_by_run: dict[int, object] = {}
        excl_parts = []
        trial_rows = []
        for run in LR_IMAGERY_RUNS:
            epochs, excl = load_and_preprocess_run(run_paths[run], cfg, run, subject_id)
            epochs_by_run[run] = epochs
            trial_rows.append(epochs.audit_metadata)
            if len(excl):
                excl_parts.append(excl)
        trials = pd.concat(trial_rows, ignore_index=True)

        # Recompute EDF hashes onto the persisted trial table; store the
        # resolved relative path so later scripts don't guess MNE's layout.
        name_to_relpath = {
            Path(run_paths[r]).name: Path(run_paths[r]).resolve().relative_to(
                data_dir.resolve()
            ).as_posix()
            for r in LR_IMAGERY_RUNS
        }
        trials["edf_file"] = trials["run_id"].map(
            {r: Path(run_paths[r]).name for r in LR_IMAGERY_RUNS}
        )
        trials["edf_relpath"] = trials["edf_file"].map(name_to_relpath)
        trials["edf_sha256"] = trials["edf_file"].map(
            {Path(run_paths[r]).name: hash_file(run_paths[r]) for r in LR_IMAGERY_RUNS}
        )
        cols = [
            "subject_id",
            "run_id",
            "trial_id",
            "event_sample",
            "onset_s",
            "label",
            "description",
            "exclusion_reason",
            "edf_file",
            "edf_relpath",
            "edf_sha256",
        ]
        trials = trials[cols].sort_values(["run_id", "trial_id"]).reset_index(drop=True)

        trials_path = data_dir / f"trials_subj{subject_id}.csv"
        trials.to_csv(trials_path, index=False)
        print(
            f"  wrote {trials_path} ({len(trials)} audited trials; "
            f"left={int((trials['label'] == 0).sum())}, "
            f"right={int((trials['label'] == 1).sum())})"
        )

        if excl_parts:
            excl = (
                pd.concat(excl_parts, ignore_index=True)
                .groupby(["run_id", "label", "exclusion_reason"], as_index=False)[
                    "count"
                ]
                .sum()
            )
        else:
            excl = pd.DataFrame(columns=["run_id", "label", "exclusion_reason", "count"])
        excl_path = data_dir / f"exclusions_subj{subject_id}.csv"
        excl.to_csv(excl_path, index=False)
        print(f"  wrote {excl_path}")
        trial_frames.append(trials)
        exclusion_frames.append(excl)

    combined = pd.concat(trial_frames, ignore_index=True)
    combined_path = data_dir / "trials_all.csv"
    combined.to_csv(combined_path, index=False)
    print(f"wrote {combined_path} ({len(combined)} trials across all subjects)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
