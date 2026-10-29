"""End-to-end synthetic test: prepare -> train -> evaluate mechanics.

Runs the full pipeline on a synthetic in-memory "dataset" (no downloads):
preprocess -> fit -> predict -> controller -> metrics, verifying the data
contract holds throughout. Real-data behavior is covered separately by the
marked integration test.
"""

from __future__ import annotations

import numpy as np
import pytest

from eeg_controller.evaluate import build_predictions_frame
from eeg_controller.model import build_pipeline


def test_full_pipeline_mechanics(cfg, tmp_path, monkeypatch):
    """prepare -> train -> evaluate equivalent on synthetic data."""
    from tests.conftest import make_epochs

    # "prepare": epochs from the synthetic fixture
    train = make_epochs(n_trials=12, seed=11)
    test = make_epochs(n_trials=8, seed=12)
    assert train.X.shape[2] == 320 and test.X.shape[2] == 320

    # "train": CSP fitted on training epochs only
    model = build_pipeline(cfg)
    model.fit(train.X, train.y)
    assert list(model.classes_) in ([0, 1], [1, 0])

    # "evaluate": predict on held-out synthetic epochs, then controller+metrics
    preds, metrics = build_predictions_frame(
        cfg, model, test, run_id=12, subject_id=1, threshold=cfg.controller.threshold
    )
    assert len(preds) == 8
    assert metrics["n_trials"] == 8
    # probability sanity: rows sum to 1
    np.testing.assert_allclose(
        preds["p_left"] + preds["p_right"], np.ones(len(preds)), atol=1e-9
    )
    # Each trial starts at center, independently of previous commands.
    expected_positions = preds["command"].map({"LEFT": -0.15, "RIGHT": 0.15, "HOLD": 0.0})
    np.testing.assert_allclose(preds["cursor_after"], expected_positions)
    # metrics are self-consistent
    n_accepted = int(preds["accepted"].sum())
    assert metrics["rejected_trials"] == 8 - n_accepted
    if n_accepted:
        expected = (
            (preds["true_label"] == preds["predicted_label"]) & preds["accepted"]
        ).sum() / n_accepted
        assert metrics["accepted_accuracy"] == pytest.approx(expected)
