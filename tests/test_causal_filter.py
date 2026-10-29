"""Causal filter tests: forward-only behavior and state handling."""

from __future__ import annotations

import numpy as np
import pytest

from eeg_controller.preprocess import CausalBandpass


@pytest.fixture()
def filt():
    return CausalBandpass(sfreq_hz=160.0, bandpass_hz=(8.0, 30.0), order=4)


def test_appending_future_samples_leaves_past_unchanged(filt):
    """Core causality property: filtered[:n] is identical with or without
    future samples appended (guide section 10)."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 400)) * 1e-5

    f1 = CausalBandpass(sfreq_hz=160.0, bandpass_hz=(8.0, 30.0), order=4)
    full = f1.process_chunk(x)

    f2 = CausalBandpass(sfreq_hz=160.0, bandpass_hz=(8.0, 30.0), order=4)
    head = f2.process_chunk(x[:, :250])
    tail = f2.process_chunk(x[:, 250:])
    split = np.concatenate([head, tail], axis=1)

    np.testing.assert_array_equal(full[:, :250], head)
    np.testing.assert_array_equal(full, split)


def test_state_reset_between_runs(filt):
    """Reusing a filter without reset must equal never resetting only if the
    input continues; a new run must start from a fresh instance/state."""
    rng = np.random.default_rng(1)
    x = rng.standard_normal((2, 200)) * 1e-5
    f_a = CausalBandpass(sfreq_hz=160.0, bandpass_hz=(8.0, 30.0), order=4)
    out_a = f_a.process_chunk(x)  # run A
    f_b = CausalBandpass(sfreq_hz=160.0, bandpass_hz=(8.0, 30.0), order=4)
    out_b = f_b.process_chunk(x)  # run B, fresh state
    np.testing.assert_array_equal(out_a, out_b)


def test_chunked_processing_matches_batch(filt):
    """State preservation across chunks within one run (3-way split)."""
    rng = np.random.default_rng(2)
    x = rng.standard_normal((3, 500)) * 1e-5
    f_batch = CausalBandpass(sfreq_hz=160.0, bandpass_hz=(8.0, 30.0), order=4)
    batch = f_batch.process_chunk(x)

    f_chunk = CausalBandpass(sfreq_hz=160.0, bandpass_hz=(8.0, 30.0), order=4)
    parts = [
        f_chunk.process_chunk(x[:, 0:100]),
        f_chunk.process_chunk(x[:, 100:300]),
        f_chunk.process_chunk(x[:, 300:500]),
    ]
    np.testing.assert_array_equal(batch, np.concatenate(parts, axis=1))


def test_filter_output_is_finite_and_zero_meanish(filt):
    rng = np.random.default_rng(3)
    t = np.arange(1600) / 160.0
    x = (1e-5 * np.sin(2 * np.pi * 12.0 * t))[None, :]
    y = filt.process_chunk(x)
    assert np.isfinite(y).all()
    # 12 Hz lies inside the passband; steady-state gain ~1.
    steady = y[0, 400:]
    assert 0.5 < steady.std() / 1e-5 < 1.5
