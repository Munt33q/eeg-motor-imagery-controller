"""Epoch boundary tests: half-open window, sample counts, quality flags."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eeg_controller.config import Config
from eeg_controller.data import RunEvents
from eeg_controller.preprocess import (
    REASON_FLAT,
    REASON_INCOMPLETE,
    REASON_NONFINITE,
    build_epochs_for_run,
)


def _events_from(labels, onsets):
    return RunEvents(
        subject_id=1,
        run_id=4,
        samples=np.asarray(onsets, dtype=np.int64),
        trial_ids=np.arange(1, len(onsets) + 1, dtype=np.int64),
        labels=np.asarray(labels, dtype=np.int64),
        descriptions=np.array(["T1" if l == 0 else "T2" for l in labels], dtype=object),
        sfreq_hz=160.0,
        n_samples_total=10**9,
        excluded_rest_count=0,
    )


def test_epoch_window_is_320_samples(cfg):
    data, onsets, labels = _simple_run()
    epochs, _ = build_epochs_for_run(data, _events_from(labels, onsets), cfg, _CH, 4, 1)
    assert epochs.samples_per_epoch == 320
    assert epochs.X.shape == (len(onsets), 9, 320)


def test_half_open_interval_first_sample_is_onset_plus_1s(cfg):
    data, onsets, labels = _simple_run()
    epochs, _ = build_epochs_for_run(data, _events_from(labels, onsets), cfg, _CH, 4, 1)
    # First sample of trial 0 must equal data[:, onset + 160]
    onset = int(onsets[0])
    np.testing.assert_array_equal(epochs.X[0, 0], data[0, onset + 160 : onset + 480])


def test_epoch_at_run_end_is_incomplete(cfg):
    data, onsets, labels = _simple_run(total_samples=600)
    # Trial at onset 100 needs samples up to 100+480=580 <= 600 -> complete.
    # Onset 200 needs 200+480=680 > 600 -> incomplete.
    onsets2 = [100, 200]
    labels2 = [0, 1]
    ev = RunEvents(
        subject_id=1,
        run_id=4,
        samples=np.array(onsets2, dtype=np.int64),
        trial_ids=np.array([1, 2], dtype=np.int64),
        labels=np.array(labels2, dtype=np.int64),
        descriptions=np.array(["T1", "T2"], dtype=object),
        sfreq_hz=160.0,
        n_samples_total=600,
        excluded_rest_count=0,
    )
    epochs, summary = build_epochs_for_run(data, ev, cfg, _CH, 4, 1)
    # Only the complete trial is kept; metadata aligns 1:1 with X.
    assert epochs.X.shape[0] == 1
    assert len(epochs.metadata) == 1
    assert int(epochs.metadata["trial_id"].iloc[0]) == 1
    # The excluded trial is recorded with its reason, run, and class.
    assert (summary["exclusion_reason"] == REASON_INCOMPLETE).any()
    assert int(summary["count"].sum()) == 1


def test_flat_and_nonfinite_epochs_excluded(cfg):
    data, onsets, labels = _simple_run()
    # Trial 1: flatten channel 0 across its epoch window.
    o = int(onsets[1])
    data2 = data.copy()
    data2[0, o + 160 : o + 480] = 0.0
    # Trial 2: inject NaN.
    o2 = int(onsets[2])
    data3 = data2.copy()
    data3[1, o2 + 200] = np.nan
    ev = _events_from(labels, onsets)
    epochs, summary = build_epochs_for_run(data3, ev, cfg, _CH, 4, 1)
    reasons = set(summary["exclusion_reason"])
    assert REASON_FLAT in reasons
    assert REASON_NONFINITE in reasons
    # Metadata rows align 1:1 with X rows (eligible only).
    assert epochs.X.shape[0] == len(epochs.metadata) == 2


def _simple_run(n_trials=4, total_samples=None):
    rng = np.random.default_rng(3)
    onsets = [200, 800, 1400, 2000]
    labels = [0, 1, 0, 1]
    total = total_samples or (onsets[-1] + 600)
    data = rng.standard_normal((9, total)) * 1e-5
    return data, onsets, labels


_CH = ("C3", "Cz", "C4", "FC3", "FCz", "FC4", "CP3", "CPz", "CP4")
