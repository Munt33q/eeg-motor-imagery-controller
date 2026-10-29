"""Shared synthetic fixtures: tiny EEG-like runs, epochs, and models.

Synthetic tests verify mechanics (timing, isolation, bookkeeping), not EEG
classification effectiveness. Real-data coverage lives in the marked
integration test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eeg_controller.config import parse_config
from eeg_controller.preprocess import EpochSet

FS = 160.0
N_CHANNELS = 9
N_SAMPLES = 320  # 2.0 s half-open window at 160 Hz

CHANNELS = ("C3", "Cz", "C4", "FC3", "FCz", "FC4", "CP3", "CPz", "CP4")


@pytest.fixture()
def cfg():
    """The frozen default contract."""
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "configs" / "default.json"
    return parse_config(json.loads(path.read_text(encoding="utf-8")))


def synth_raw_run(
    n_trials: int = 6,
    fs: float = FS,
    n_channels: int = N_CHANNELS,
    seed: int = 0,
    trial_labels: list[int] | None = None,
    trial_spacing_s: float = 4.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a synthetic continuous run [channels, samples] + events.

    Left trials (0) get 10 Hz mu-alpha on channel 0; right trials (1) get it
    on channel 2. This is a mechanical fixture — it does not model real EEG.
    """
    rng = np.random.default_rng(seed)
    spacing = int(trial_spacing_s * fs)
    onsets = np.arange(n_trials, dtype=np.int64) * spacing + int(2.0 * fs)
    labels = (
        np.asarray(trial_labels, dtype=np.int64)
        if trial_labels is not None
        else (np.arange(n_trials) % 2).astype(np.int64)
    )
    total = onsets[-1] + int(5.0 * fs)
    t = np.arange(total) / fs
    data = 0.3 * rng.standard_normal((n_channels, total))  # volts-ish noise
    for onset, lbl in zip(onsets, labels):
        s0 = onset + int(1.0 * fs)
        s1 = onset + int(3.0 * fs)
        freq = 10.0
        wave = 2e-5 * np.sin(2 * np.pi * freq * t[s0:s1])
        ch = 0 if lbl == 0 else 2
        data[ch, s0:s1] += wave
    return data, onsets, labels


def make_epochs(
    n_trials: int = 8, seed: int = 0, labels: list[int] | None = None
) -> EpochSet:
    """Synthetic EpochSet with the same embedded sinusoid trick."""
    data, onsets, lab = synth_raw_run(
        n_trials=n_trials, seed=seed, trial_labels=labels
    )
    start, end = 160, 480  # [1.0, 3.0) s at 160 Hz
    X = np.stack([data[:, o + start : o + end] for o in onsets])
    y = lab.astype(np.int64)
    meta = pd.DataFrame(
        {
            "subject_id": [1] * n_trials,
            "run_id": [4] * n_trials,
            "trial_id": np.arange(1, n_trials + 1),
            "event_sample": onsets,
            "onset_s": onsets / FS,
            "label": y,
            "exclusion_reason": [""] * n_trials,
        }
    )
    meta.attrs["channels"] = CHANNELS
    return EpochSet(
        X=X,
        y=y,
        metadata=meta,
        channel_names=CHANNELS,
        sfreq_hz=FS,
        epoch_start_s=1.0,
        epoch_end_s=3.0,
    )


@pytest.fixture()
def fitted_model(cfg):
    """A CSP+LDA pipeline fitted on a deterministic synthetic fixture."""
    from eeg_controller.model import build_pipeline

    epochs = make_epochs(n_trials=12, seed=1)
    model = build_pipeline(cfg)
    model.fit(epochs.X, epochs.y)
    return model
