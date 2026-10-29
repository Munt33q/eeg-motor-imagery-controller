"""Causal preprocessing: channel contract, referencing, filtering, epochs.

Fixed contract (guide section 5):
- Nine channels: C3, Cz, C4, FC3, FCz, FC4, CP3, CPz, CP4 (this exact order).
- Referencing: per-sample mean across those nine channels (common average).
- Filter: 4th-order Butterworth bandpass 8-30 Hz as SOS, applied with
  :func:`scipy.signal.sosfilt` (forward-only, causal). Never filtfilt.
- Filter state resets at each run; state persists across chunks within a run.
- Epochs: [1.0, 3.0) s after T1/T2 onset -> exactly 320 samples at 160 Hz.
- No baseline correction. Units: volts (MNE convention); convert to µV only
  for display, never for classifier input.

The :class:`CausalBandpass` class keeps filter state between calls so the
same code path serves both batch preprocessing and chunked replay.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfilt

from .config import Config
from .data import DataError, extract_events, load_run_raw

# Reason codes recorded per excluded trial (column: exclusion_reason).
REASON_NONFINITE = "nonfinite"
REASON_FLAT = "flat"
REASON_OUT_OF_BOUNDS = "out_of_bounds"
REASON_INCOMPLETE = "incomplete"  # epoch extends past end of run
REASON_SETTLING = "filter_settling"  # epoch intersects documented settling interval


@dataclass
class CausalBandpass:
    """Forward-only Butterworth bandpass with carry-over state.

    :func:`scipy.signal.sosfilt` is used directly (no zero-phase filtfilt),
    so the output at sample n depends only on samples <= n. Calling with a
    second chunk continues from the state left by the first chunk; call
    :meth:`reset` at every run boundary.
    """

    sfreq_hz: float
    bandpass_hz: tuple[float, float]
    order: int = 4
    _sos: np.ndarray = field(init=False, repr=False)
    _zi: np.ndarray | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._sos = butter(
            self.order,
            self.bandpass_hz,
            btype="bandpass",
            output="sos",
            fs=self.sfreq_hz,
        )

    def reset(self) -> None:
        """Reset filter state (called at every run boundary)."""
        self._zi = None

    def process_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """Filter one chunk [channels, samples], preserving state."""
        chunk = np.asarray(chunk, dtype=np.float64)
        if chunk.ndim != 2:
            raise ValueError("chunk must be 2-D [channels, samples]")
        if self._zi is None:
            n_sections = self._sos.shape[0]
            # zi shape: (n_sections, n_channels, 2) for axis=-1 filtering
            self._zi = np.zeros((n_sections, chunk.shape[0], 2), dtype=np.float64)
        out, self._zi = sosfilt(self._sos, chunk, axis=-1, zi=self._zi)
        return out


@dataclass
class EpochSet:
    """Epoch container: X [trials, channels, samples], y, metadata, contract."""

    X: np.ndarray  # float64 [trials, channels, samples], volts
    y: np.ndarray  # int64 [trials], 0=left, 1=right
    metadata: pd.DataFrame  # one row per trial, incl. exclusion reasons
    channel_names: tuple[str, ...]
    sfreq_hz: float
    epoch_start_s: float
    epoch_end_s: float

    def __post_init__(self) -> None:
        n_trials, n_channels, n_samples = self.X.shape
        if self.y.shape != (n_trials,):
            raise ValueError("y must have shape [trials]")
        if len(self.metadata) != n_trials:
            raise ValueError("metadata must have one row per trial")
        if tuple(self.channel_names) != tuple(self.metadata.attrs["channels"]):
            raise ValueError("metadata channel order does not match channel_names")
        expected = int(round((self.epoch_end_s - self.epoch_start_s) * self.sfreq_hz))
        if n_samples != expected:
            raise ValueError(
                f"epochs have {n_samples} samples; contract requires {expected} "
                f"for [{self.epoch_start_s}, {self.epoch_end_s}) s at "
                f"{self.sfreq_hz} Hz"
            )

    @property
    def samples_per_epoch(self) -> int:
        return self.X.shape[2]

    def eligible_mask(self) -> np.ndarray:
        """Boolean mask of trials that passed all quality checks."""
        return (self.metadata["exclusion_reason"] == "").to_numpy()


def reference_average(
    data: np.ndarray, channel_names: tuple[str, ...], channels: tuple[str, ...]
) -> np.ndarray:
    """Per-sample mean across the fixed nine channels, subtracted per channel.

    ``data`` is [channels, samples] in volts, already in ``channels`` order.
    """
    if tuple(channel_names) != tuple(channels):
        raise ValueError(
            f"referencing requires data in contract order {list(channels)}, "
            f"got {list(channel_names)}"
        )
    mean = data.mean(axis=0, keepdims=True)
    return data - mean


def _check_epoch_quality(epoch: np.ndarray) -> str:
    """Return an exclusion reason code, or '' if the epoch is acceptable."""
    if not np.isfinite(epoch).all():
        return REASON_NONFINITE
    # Flat: a channel with (near-)zero variance across the epoch window.
    stds = epoch.std(axis=1)
    if np.any(stds <= 1e-12):
        return REASON_FLAT
    # Out-of-bounds: |amplitude| beyond 1 mV after bandpassing (in volts).
    if np.any(np.abs(epoch) > 1e-3):
        return REASON_OUT_OF_BOUNDS
    return ""


def build_epochs_for_run(
    raw_data: np.ndarray,
    events: "object",
    cfg: Config,
    channel_names: tuple[str, ...],
    run_id: int,
    subject_id: int,
) -> tuple[EpochSet, pd.DataFrame]:
    """Extract cue-aligned epochs from one preprocessed continuous run.

    ``raw_data`` is [channels, samples] in volts for the whole run, already
    referenced and causally filtered. ``events`` is a
    :class:`~eeg_controller.data.RunEvents`.

    Returns the full EpochSet (kept + excluded rows, so exclusions are
    auditable) and an exclusion summary DataFrame.
    """
    n_channels, n_samples_total = raw_data.shape
    p = cfg.preprocess
    fs = p.sfreq_hz
    start_samp = int(round(p.epoch_start_s * fs))
    end_samp = int(round(p.epoch_end_s * fs))  # half-open: [start, end)
    n_per = end_samp - start_samp
    expected = cfg.samples_per_epoch
    if n_per != expected:
        raise DataError(
            f"epoch window resolves to {n_per} samples; contract requires "
            f"{expected}"
        )
    settling_samples = int(round(p.settling_s * fs))

    X_list, y_list, rows = [], [], []
    for i in range(len(events.samples)):
        onset = int(events.samples[i])
        s0, s1 = onset + start_samp, onset + end_samp
        trial_id = int(events.trial_ids[i])
        reason = ""
        if s1 > n_samples_total:
            reason = REASON_INCOMPLETE
            epoch = None
        else:
            epoch = raw_data[:, s0:s1].astype(np.float64, copy=True)
            reason = _check_epoch_quality(epoch)
        # Settling: exclude startup events whose *epoch window start* lies
        # before the documented settling interval has fully elapsed.
        if reason == "" and onset + start_samp < settling_samples:
            reason = REASON_SETTLING

        row = {
            "subject_id": subject_id,
            "run_id": run_id,
            "trial_id": trial_id,
            "event_sample": onset,
            "onset_s": onset / fs,
            "label": int(events.labels[i]),
            "description": str(events.descriptions[i]),
            "exclusion_reason": reason,
        }
        rows.append(row)
        if reason == "":
            X_list.append(epoch)
            y_list.append(int(events.labels[i]))

    # EpochSet contract: metadata rows correspond 1:1 with X/y rows and
    # contain *eligible* trials only. Excluded trials are recorded in the
    # returned summary (counts by run/class/reason), never silently dropped.
    full = pd.DataFrame(rows)
    kept_mask = (full["exclusion_reason"] == "").to_numpy()
    metadata = full.loc[kept_mask].reset_index(drop=True)
    metadata.attrs["channels"] = tuple(channel_names)
    metadata.attrs["sfreq_hz"] = fs

    if X_list:
        X = np.stack(X_list).astype(np.float64)
        y = np.asarray(y_list, dtype=np.int64)
    else:  # keep the container well-formed with zero eligible trials
        X = np.empty((0, n_channels, n_per), dtype=np.float64)
        y = np.empty((0,), dtype=np.int64)

    epochs = EpochSet(
        X=X,
        y=y,
        metadata=metadata,
        channel_names=tuple(channel_names),
        sfreq_hz=fs,
        epoch_start_s=p.epoch_start_s,
        epoch_end_s=p.epoch_end_s,
    )

    summary = (
        full[~kept_mask]
        .groupby(["run_id", "label", "exclusion_reason"])
        .size()
        .reset_index(name="count")
    )
    return epochs, summary


def load_and_preprocess_run(
    edf_path: str | Path, cfg: Config, run_id: int, subject_id: int
) -> tuple[EpochSet, pd.DataFrame]:
    """Full causal preprocessing for one EDF run, kept run-local.

    Steps: load + standardize -> select the nine contract channels *in
    contract order* -> common-average referencing -> forward-only SOS
    bandpass (fresh filter state; never filters across run boundaries) ->
    cue-aligned epochs with quality exclusions.

    Returns the EpochSet (eligible trials only; exclusions in the summary)
    and the exclusion summary for the run.
    """
    from .data import check_run_supported

    check_run_supported(run_id)
    p = cfg.preprocess
    raw = load_run_raw(edf_path, p.channels, p.sfreq_hz)

    # Select in the exact stored contract order (never rely on EDF order).
    idx = [raw.ch_names.index(ch) for ch in p.channels]
    data = raw.get_data()  # volts, [all_channels, samples]
    data = data[idx, :]

    data = reference_average(data, p.channels, p.channels)

    filt = CausalBandpass(  # fresh instance == state reset at each run
        sfreq_hz=p.sfreq_hz,
        bandpass_hz=p.bandpass_hz,
        order=p.filter_order,
    )
    data = filt.process_chunk(data)

    events = extract_events(raw, cfg, run_id)
    return build_epochs_for_run(
        data, events, cfg, p.channels, run_id, subject_id
    )
