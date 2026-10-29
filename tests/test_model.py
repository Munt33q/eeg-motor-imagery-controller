"""Pipeline round-trip: saving/loading preserves predictions on a fixture."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.pipeline import Pipeline

from eeg_controller.model import (
    build_pipeline,
    load_model,
    predict_with_scores,
    save_model,
)


def test_save_load_preserves_predictions(fitted_model, tmp_path):
    from tests.conftest import make_epochs

    X = make_epochs(n_trials=6, seed=7).X
    preds_before, p_left_b, p_right_b = predict_with_scores(fitted_model, X)

    model_path = tmp_path / "model.joblib"
    meta_path = tmp_path / "meta.json"
    save_model(
        fitted_model,
        {"subject_id": 1, "fit_runs": [4, 8]},
        model_path,
        meta_path,
    )
    loaded = load_model(model_path)
    assert isinstance(loaded, Pipeline)

    preds_after, p_left_a, p_right_a = predict_with_scores(loaded, X)
    np.testing.assert_array_equal(preds_before, preds_after)
    np.testing.assert_allclose(p_left_b, p_left_a)
    np.testing.assert_allclose(p_right_b, p_right_a)


def test_probability_columns_mapped_via_classes(fitted_model):
    """p_left must follow model.classes_, not column position."""
    from tests.conftest import make_epochs

    X = make_epochs(n_trials=6, seed=8).X
    proba = fitted_model.predict_proba(X)
    classes = list(fitted_model.classes_)
    preds, p_left, p_right = predict_with_scores(fitted_model, X)
    if classes == [0, 1]:
        np.testing.assert_allclose(p_left, proba[:, 0])
        np.testing.assert_allclose(p_right, proba[:, 1])
    elif classes == [1, 0]:
        np.testing.assert_allclose(p_left, proba[:, 1])
        np.testing.assert_allclose(p_right, proba[:, 0])
    else:
        pytest.fail(f"unexpected classes: {classes}")
    # Consistency: argmax of (p_left, p_right) matches predicted label.
    np.testing.assert_array_equal((p_right > p_left).astype(int), preds)


def test_load_missing_model_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="train.py"):
        load_model(tmp_path / "nope.joblib")
