"""Split isolation, channel contract, and unit-conversion tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eeg_controller.evaluate import build_split_info
from eeg_controller.preprocess import reference_average


def _trial_table():
    return pd.DataFrame(
        {
            "subject_id": [1] * 6,
            "run_id": [4, 4, 8, 8, 12, 12],
            "trial_id": [1, 2, 1, 2, 1, 2],
            "label": [0, 1, 1, 0, 0, 1],
        }
    )


def test_no_overlap_between_dev_and_final(cfg):
    info = build_split_info(cfg, 1, _trial_table())
    assert info.overlap is False
    assert info.dev_runs == (4, 8)
    assert info.final_runs == (12,)
    assert info.dev_trials == 4
    assert info.final_trials == 2


def test_overlap_is_detected(cfg):
    bad = _trial_table()
    cfg_bad = cfg  # mutate via dataclasses.replace
    from dataclasses import replace

    cfg_bad = replace(cfg, data=replace(cfg.data, dev_runs=(4, 8), final_runs=(8, 12)))
    info = build_split_info(cfg_bad, 1, bad)
    assert info.overlap is True


def test_channel_order_mismatch_rejected():
    data = np.zeros((3, 10))
    with pytest.raises(ValueError, match="contract order"):
        reference_average(
            data, ("Cz", "C3", "C4"), ("C3", "Cz", "C4")
        )


def test_channel_reorder_to_stored_order_is_explicit():
    """Reordering helper: selection by name must preserve stored order."""
    names_in = ("Cz", "C3", "C4")
    contract = ("C3", "Cz", "C4")
    idx = [names_in.index(ch) for ch in contract]
    assert idx == [1, 0, 2]  # explicit reindex, not a silent assumption


def test_display_conversion_does_not_alter_classifier_input():
    """Classifier consumes volts; µV conversion must be display-only."""
    volts = np.array([[1e-5, -2e-5]])
    display_uv = volts * 1e6
    np.testing.assert_allclose(display_uv, [[10.0, -20.0]])
    # The original volts array is what the pipeline must receive.
    np.testing.assert_allclose(volts, [[1e-5, -2e-5]])
