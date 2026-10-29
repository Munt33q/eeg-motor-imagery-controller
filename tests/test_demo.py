"""Behavior tests for the demo app's pure logic (no Streamlit)."""

from __future__ import annotations

import numpy as np
import pytest

from eeg_controller.demo import (
    build_replay_events,
    next_trial_time,
    trial_prediction,
    waveform_uv_window,
)
from eeg_controller.replay import ReplayState


def test_build_replay_events_no_labels():
    """Events carry timestamps only — no labels can leak into replay."""
    events, duration = build_replay_events([1.0, 5.5], decision_delay_s=3.0)
    assert [e["decision_s"] for e in events] == [4.0, 8.5]
    assert duration == pytest.approx(9.5)
    for e in events:
        assert set(e) == {"onset_s", "decision_s", "trial_index"}


def test_build_replay_events_rejects_bad_input():
    with pytest.raises(ValueError):
        build_replay_events([], 3.0)
    with pytest.raises(ValueError):
        build_replay_events([5.0, 1.0], 3.0)
    with pytest.raises(ValueError):
        build_replay_events([1.0, 1.0], 3.0)
    with pytest.raises(ValueError):
        build_replay_events([1.0], 0.0)


def test_next_trial_time_advances_one_at_a_time():
    events, _ = build_replay_events([1.0, 5.5, 10.0], 3.0)
    st = ReplayState(duration_s=20.0, events=events)
    assert next_trial_time(st, events) == 4.0
    st.time_s = 4.0
    st.process_due()  # command-less events: no issued_commands entry
    assert 0 not in st.issued_commands
    assert next_trial_time(st, events) == 8.5
    st.time_s = 8.5
    st.process_due()
    assert next_trial_time(st, events) == 13.0
    st.time_s = 13.0
    st.process_due()
    assert next_trial_time(st, events) is None


def test_next_trial_time_ignores_already_processed():
    events, _ = build_replay_events([1.0, 5.5], 3.0)
    st = ReplayState(duration_s=20.0, events=events)
    st.time_s = 8.5
    st.process_due()  # processes both
    assert next_trial_time(st, events) is None


def test_waveform_window_causality_and_units(cfg):
    """Only samples with t <= now; volts -> µV conversion is display-only."""
    fs = cfg.preprocess.sfreq_hz
    rng = np.random.default_rng(0)
    data = rng.standard_normal((9, int(10 * fs))) * 1e-5
    names = cfg.preprocess.channels
    t = 5.0
    times, uv, chans = waveform_uv_window(data, names, t, fs, window_s=2.0)
    assert chans == ("C3", "Cz", "C4")
    assert times[-1] <= t  # no future samples
    assert len(times) == uv.shape[1]
    # volts -> µV: values must be 1e6 times the raw slice
    expected_s0 = int((t - 2.0) * fs)
    expected_s1 = int(np.ceil(t * fs))
    expected = data[[0, 1, 2], expected_s0:expected_s1] * 1e6
    np.testing.assert_allclose(uv, expected)


def test_waveform_window_near_zero_time():
    fs = 160.0
    data = np.zeros((9, 160))
    times, uv, _ = waveform_uv_window(data, ("C3", "Cz", "C4", *["C4"] * 6), 0.5, fs)
    assert len(times) == 80  # samples 0..79 (window is [0, ceil(t*fs)))
    assert uv.shape == (3, 80)


def test_trial_prediction_matches_offline_records(fitted_model, cfg):
    """Live single-trial inference reproduces offline pipeline values."""
    from tests.conftest import make_epochs

    epochs = make_epochs(n_trials=6, seed=3)
    for i in range(len(epochs.X)):
        command, p_left, p_right, pred, inference_ms = trial_prediction(
            fitted_model, epochs.X[i], cfg.controller.threshold
        )
        assert command in ("LEFT", "RIGHT", "HOLD")
        assert 0.0 <= p_left <= 1.0 and 0.0 <= p_right <= 1.0
        assert pred in (0, 1)
        assert inference_ms >= 0.0
        # consistency: command agrees with argmax + threshold
        if command == "LEFT":
            assert p_left > p_right and p_left >= cfg.controller.threshold
        if command == "RIGHT":
            assert p_right > p_left and p_right >= cfg.controller.threshold
        if command == "HOLD":
            assert max(p_left, p_right) < cfg.controller.threshold or p_left == p_right


def test_trial_prediction_rejects_bad_epoch(fitted_model):
    with pytest.raises(ValueError):
        trial_prediction(fitted_model, np.zeros((320,)), 0.7)
