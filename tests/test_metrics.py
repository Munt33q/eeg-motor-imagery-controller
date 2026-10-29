"""Metrics tests: coverage, accepted accuracy, nulls, exclusion counting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eeg_controller.evaluate import (
    build_predictions_frame,
    compute_metrics,
    exclusion_summary_table,
    majority_baseline_label,
)
from eeg_controller.model import predict_with_scores


def _df(rows):
    return pd.DataFrame(rows)


def test_accuracy_and_confusion():
    df = _df(
        {
            "true_label": [0, 0, 1, 1, 0],
            "predicted_label": [0, 1, 1, 1, 0],
            "accepted": [True, True, True, False, False],
            "command": ["LEFT", "RIGHT", "RIGHT", "HOLD", "HOLD"],
            "inference_ms": [0.1, 0.2, 0.3, 0.2, 0.1],
        }
    )
    m = compute_metrics(df, {0: "left", 1: "right"}, majority_label=0)
    assert m["n_trials"] == 5
    assert m["accuracy"] == pytest.approx(4 / 5)
    c = m["confusion"]["matrix"]
    assert c["true_left_pred_left"] == 2
    assert c["true_left_pred_right"] == 1
    assert c["true_right_pred_left"] == 0
    assert c["true_right_pred_right"] == 2
    assert m["majority_baseline"]["test_accuracy"] == pytest.approx(3 / 5)


def test_coverage_and_accepted_accuracy():
    df = _df(
        {
            "true_label": [0, 1, 1, 0],
            "predicted_label": [0, 1, 0, 0],
            "accepted": [True, True, True, False],
            "command": ["LEFT", "RIGHT", "LEFT", "HOLD"],
            "inference_ms": [0.1, 0.1, 0.1, 0.1],
        }
    )
    m = compute_metrics(df, {0: "left", 1: "right"})
    assert m["coverage"] == pytest.approx(3 / 4)
    # accepted: 3, correct among them: 2 -> 2/3
    assert m["accepted_accuracy"] == pytest.approx(2 / 3)
    assert m["incorrect_issued_commands"] == 1
    assert m["incorrect_issued_rate"] == pytest.approx(1 / 4)
    assert m["rejected_trials"] == 1


def test_zero_accepted_returns_null_accepted_accuracy():
    df = _df(
        {
            "true_label": [0, 1],
            "predicted_label": [0, 1],
            "accepted": [False, False],
            "command": ["HOLD", "HOLD"],
            "inference_ms": [0.1, 0.1],
        }
    )
    m = compute_metrics(df, {0: "left", 1: "right"})
    assert m["coverage"] == 0.0
    assert m["accepted_accuracy"] is None  # null, not 0
    assert m["incorrect_issued_commands"] == 0


def test_rejected_commands_not_credited_as_correct():
    """A HOLD on a trial the model would have gotten right is not a win."""
    df = _df(
        {
            "true_label": [0, 0],
            "predicted_label": [0, 0],
            "accepted": [True, False],
            "command": ["LEFT", "HOLD"],
            "inference_ms": [0.1, 0.1],
        }
    )
    m = compute_metrics(df, {0: "left", 1: "right"})
    assert m["accepted_accuracy"] == pytest.approx(1.0)
    assert m["coverage"] == pytest.approx(0.5)


def test_majority_baseline_from_dev_labels_only():
    y_dev = np.array([1, 1, 1, 0])
    assert majority_baseline_label(y_dev) == 1
    with pytest.raises(ValueError):
        majority_baseline_label(np.array([], dtype=np.int64))


def test_inference_timing_reported():
    df = _df(
        {
            "true_label": [0, 1, 0],
            "predicted_label": [0, 1, 1],
            "accepted": [True, True, True],
            "command": ["LEFT", "RIGHT", "LEFT"],
            "inference_ms": [1.0, 2.0, 3.0],
        }
    )
    m = compute_metrics(df, {0: "left", 1: "right"})
    assert m["inference_ms"]["median"] == pytest.approx(2.0)
    assert m["inference_ms"]["p95"] == pytest.approx(np.percentile([1, 2, 3], 95))


def test_exclusion_summary_counts_by_run_and_class():
    parts = [
        pd.DataFrame(
            {
                "run_id": [4, 4],
                "label": [0, 1],
                "exclusion_reason": ["flat", "flat"],
                "count": [1, 2],
            }
        ),
        pd.DataFrame(
            {
                "run_id": [12],
                "label": [1],
                "exclusion_reason": ["incomplete"],
                "count": [1],
            }
        ),
    ]
    out = exclusion_summary_table(parts)
    assert len(out) == 3
    assert out["count"].sum() == 4
    row = out[(out["run_id"] == 4) & (out["label"] == 0)]
    assert row["count"].iloc[0] == 1


def test_predictions_frame_labels_never_touch_controller(cfg, fitted_model):
    """End-to-end guard: commands derive from probabilities; a label swap
    must leave every command unchanged."""
    from tests.conftest import make_epochs

    epochs = make_epochs(n_trials=8, seed=5)
    preds_a, _ = build_predictions_frame(
        cfg, fitted_model, epochs, run_id=4, subject_id=1, threshold=0.7
    )
    # Flip every true label in the epoch metadata and recompute.
    epochs_flipped = make_epochs(n_trials=8, seed=5)
    epochs_flipped.y = 1 - epochs_flipped.y
    epochs_flipped.metadata["label"] = epochs_flipped.y
    preds_b, _ = build_predictions_frame(
        cfg, fitted_model, epochs_flipped, run_id=4, subject_id=1, threshold=0.7
    )
    assert list(preds_a["command"]) == list(preds_b["command"])
    np.testing.assert_allclose(preds_a["p_left"], preds_b["p_left"])


def test_predictions_frame_schema(cfg, fitted_model):
    from tests.conftest import make_epochs

    epochs = make_epochs(n_trials=6, seed=9)
    preds, _ = build_predictions_frame(
        cfg, fitted_model, epochs, run_id=4, subject_id=1, threshold=0.7
    )
    expected_cols = {
        "trial_id", "subject_id", "run_id", "onset_s", "decision_s",
        "true_label", "predicted_label", "p_left", "p_right",
        "inference_ms", "command", "accepted", "threshold",
    }
    assert expected_cols <= set(preds.columns)
    # decision_s == onset_s + 3.0
    np.testing.assert_allclose(
        preds["decision_s"], preds["onset_s"] + cfg.controller.decision_delay_s
    )
    # threshold recorded per row
    assert (preds["threshold"] == 0.7).all()
    # accepted iff command != HOLD
    assert (preds["accepted"] == (preds["command"] != "HOLD")).all()
