"""Regression coverage for review findings; no classification-effectiveness claims."""
import json
from pathlib import Path

import numpy as np
import pytest

from eeg_controller.config import ConfigError, parse_config
from eeg_controller.controller import choose_command
from eeg_controller.evaluate import build_predictions_frame
from eeg_controller.model import validate_artifact_metadata
from eeg_controller.preprocess import CausalBandpass


def test_replay_hides_labels_and_predictions_until_decision():
    from eeg_controller.replay import ReplayState
    state = ReplayState(10, [{"onset_s": 1, "decision_s": 4,
                             "command": "LEFT", "true_label": 0, "p_left": 0.9}])
    state.tick(2)
    assert state.visible_events() == [{"onset_s": 1, "decision_s": 4}]
    assert not state.future_labels_visible()
    state.tick(2)
    assert state.process_due()[0]["command"] == "LEFT"


def test_cropped_raw_events_use_relative_sample_indices(cfg):
    import mne
    from eeg_controller.data import extract_events
    raw = mne.io.RawArray(np.zeros((1, 1600)), mne.create_info(["C3"], 160, "eeg"), verbose=False)
    raw.set_annotations(mne.Annotations([4.0, 6.0], [0, 0], ["T1", "T2"]))
    raw.crop(tmin=2.0)
    events = extract_events(raw, cfg, 4)
    np.testing.assert_array_equal(events.samples, [320, 640])


def test_latency_wraps_actual_model_inference(cfg, monkeypatch):
    from tests.conftest import make_epochs
    import eeg_controller.evaluate as evaluation

    clock = [0.0]
    monkeypatch.setattr(evaluation.time, "perf_counter", lambda: clock[0])

    def inference(model, X):
        clock[0] += 0.012
        assert len(X) == 1
        return np.array([0]), np.array([0.9]), np.array([0.1])

    monkeypatch.setattr(evaluation, "predict_with_scores", inference)
    predictions, _ = build_predictions_frame(cfg, None, make_epochs(4), 12, 1, 0.7)
    np.testing.assert_allclose(predictions.inference_ms, 12.0)
    np.testing.assert_allclose(predictions.cursor_after, -0.15)


@pytest.mark.parametrize("probs,classes,threshold", [
    ([0.9, 0.1], [0, 1], float("nan")),
    ([0.9, 0.1], [0, 1], 1.1),
    ([1.1, -0.1], [0, 1], 0.7),
    ([0.9, 0.9], [0, 1], 0.7),
    ([0.9, 0.1], [0, 2], 0.7),
])
def test_invalid_controller_inputs_rejected(probs, classes, threshold):
    with pytest.raises(ValueError):
        choose_command(probs, classes, threshold)


def test_filter_step_settles_within_documented_interval(cfg):
    filt = CausalBandpass(160.0, (8.0, 30.0))
    response = filt.process_chunk(np.ones((1, 1600)))
    start = int(cfg.preprocess.settling_s * 160)
    assert np.max(np.abs(response[:, start:])) < 0.001


def test_artifact_contract_mismatch_rejected(cfg):
    metadata = {"subject_id": 1, "fit_runs": [4, 8],
                "config": __import__("dataclasses").asdict(cfg)}
    # JSON serialization turns tuples into lists, as on disk.
    metadata = json.loads(json.dumps(metadata))
    validate_artifact_metadata(metadata, cfg, 1)
    metadata["config"]["preprocess"]["channels"].reverse()
    with pytest.raises(ValueError, match="contract"):
        validate_artifact_metadata(metadata, cfg, 1)


@pytest.mark.parametrize("section,key,value", [
    ("controller", "decision_delay_s", 1.0),
    ("preprocess", "baseline_correction", [0, 1]),
    ("preprocess", "settling_s", -1),
    ("data", "final_runs", [8]),
    ("data", "label_names", {"0": "right", "1": "left"}),
])
def test_invalid_configuration_rejected(section, key, value):
    raw = json.loads((Path(__file__).parents[1] / "configs/default.json").read_text())
    raw[section][key] = value
    with pytest.raises(ConfigError):
        parse_config(raw)
