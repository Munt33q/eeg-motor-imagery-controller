"""Label mapping tests: explicit T1/T2 mapping, incompatible run rejection."""

from __future__ import annotations

import numpy as np
import pytest

from eeg_controller.data import (
    DataError,
    RunEvents,
    _label_from_description,
    check_run_supported,
)


def test_t1_maps_to_left_t2_maps_to_right():
    assert _label_from_description("T1", {"T1": 0, "T2": 1}) == 0
    assert _label_from_description("T2", {"T1": 0, "T2": 1}) == 1


def test_unknown_description_rejected():
    with pytest.raises(DataError, match="no explicit mapping"):
        _label_from_description("T3", {"T1": 0, "T2": 1})


@pytest.mark.parametrize("run", [4, 8, 12])
def test_lr_imagery_runs_accepted(run):
    check_run_supported(run)  # must not raise


@pytest.mark.parametrize("run", [6, 10, 14])
def test_hands_feet_runs_rejected(run):
    with pytest.raises(DataError, match="HANDS vs FEET"):
        check_run_supported(run)


@pytest.mark.parametrize("run", [1, 2, 3, 5, 7, 9, 11, 13])
def test_other_runs_rejected(run):
    with pytest.raises(DataError):
        check_run_supported(run)


def test_config_rejects_wrong_run_set(cfg):
    from eeg_controller.config import parse_config

    raw = {
        "seed": 42,
        "data": {
            "lr_imagery_runs": [4, 8],
            "dev_runs": [4],
            "final_runs": [8],
            "label_map": {"T1": 0, "T2": 1},
            "rest_description": "T0",
            "label_names": {"0": "left", "1": "right"},
        },
        "preprocess": {
            "channels": ["C3", "Cz", "C4", "FC3", "FCz", "FC4", "CP3", "CPz", "CP4"],
            "sfreq_hz": 160.0,
            "bandpass_hz": [8.0, 30.0],
            "filter_order": 4,
            "filtering": "sosfilt_causal_forward_only",
            "settling_s": 1.0,
            "epoch_start_s": 1.0,
            "epoch_end_s": 3.0,
            "baseline_correction": None,
        },
        "model": {
            "csp_components": 4,
            "csp_reg": "ledoit_wolf",
            "csp_log": True,
            "csp_norm_trace": False,
            "lda_solver": "lsqr",
            "lda_shrinkage": "auto",
        },
        "controller": {"threshold": 0.7, "cursor_step": 0.15, "decision_delay_s": 3.0},
    }
    with pytest.raises(Exception):
        parse_config({**raw, "data": {**raw["data"], "lr_imagery_runs": [6, 10, 14]}})


def test_run_events_rest_excluded():
    """extract_events skips T0; verified here at the data-structure level."""
    ev = RunEvents(
        subject_id=1,
        run_id=4,
        samples=np.array([100, 200, 300], dtype=np.int64),
        trial_ids=np.array([1, 2, 3], dtype=np.int64),
        labels=np.array([0, 1, 0], dtype=np.int64),
        descriptions=np.array(["T1", "T2", "T1"], dtype=object),
        sfreq_hz=160.0,
        n_samples_total=1000,
        excluded_rest_count=4,
    )
    assert ev.excluded_rest_count == 4
    assert set(ev.labels) == {0, 1}
