"""Real-data integration test (marked; excluded from default runs).

Requires the dataset and a trained artifact:

    python scripts/prepare_data.py --subjects 1 --config configs/default.json
    python scripts/train.py --subject 1 --config configs/default.json

Run explicitly with:
    pytest -q -m integration --basetemp=.pytest_tmp

Verifies the real EDF contract: standardized channel names, 160 Hz,
explicit T1/T2 mapping, 320-sample epochs, and label isolation in the
controller path.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import subprocess
import sys

from eeg_controller.config import load_config
from eeg_controller.preprocess import load_and_preprocess_run

pytestmark = pytest.mark.integration

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _require_data(subject: int):
    data_dir = PROJECT_ROOT / "data"
    trials_path = data_dir / f"trials_subj{subject}.csv"
    if not trials_path.exists():
        pytest.skip(
            f"{trials_path.name} not found; run scripts/prepare_data.py first"
        )
    return data_dir, pd.read_csv(trials_path)


@pytest.mark.parametrize("subject", [1])
def test_real_run_contract(subject):
    cfg = load_config(PROJECT_ROOT / "configs" / "default.json")
    data_dir, trials = _require_data(subject)

    edf_relpath = str(trials["edf_relpath"].iloc[0])
    edf_path = data_dir / edf_relpath
    if not edf_path.exists():
        pytest.skip(f"EDF not downloaded: {edf_path}")
    run_id = int(trials["run_id"].iloc[0])
    epochs, exclusions = load_and_preprocess_run(edf_path, cfg, run_id, subject)

    # Contract: 9 channels in stored order, 160 Hz, exactly 320 samples.
    assert epochs.channel_names == cfg.preprocess.channels
    assert epochs.sfreq_hz == 160.0
    assert epochs.samples_per_epoch == 320
    assert epochs.X.dtype == np.float64

    # Labels come from explicit T1/T2 mapping only.
    assert set(np.unique(epochs.y)) <= {0, 1}
    assert set(epochs.metadata["description"]) <= {"T1", "T2"}
    assert (epochs.metadata["label"] == epochs.y).all()

    # Eligible metadata aligns 1:1 with X rows; no excluded rows leak in.
    assert len(epochs.metadata) == epochs.X.shape[0]
    assert (epochs.metadata["exclusion_reason"] == "").all()
    # Kept + excluded trials reproduce the event count for this run.
    n_kept = len(epochs.metadata)
    n_excluded = int(exclusions["count"].sum()) if len(exclusions) else 0
    n_in_table = int((trials["run_id"] == run_id).sum())
    assert n_kept + n_excluded == n_in_table


def test_real_cli_training_evaluation_and_exploratory_isolation(tmp_path):
    data_dir, _ = _require_data(1)
    if not list((data_dir / "raw").rglob("S001R04.edf")):
        pytest.skip("real EDF downloads not available")
    for stage in ("train", "evaluate"):
        subprocess.run([
            sys.executable, str(PROJECT_ROOT / f"scripts/{stage}.py"),
            "--subject", "1", "--data-dir", str(data_dir),
            "--artifacts-dir", str(tmp_path),
        ], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True)
    final = (tmp_path / "predictions_sub1.csv").read_bytes()
    subprocess.run([
        sys.executable, str(PROJECT_ROOT / "scripts/evaluate.py"),
        "--subject", "1", "--data-dir", str(data_dir),
        "--artifacts-dir", str(tmp_path), "--threshold", "0.8",
    ], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True)
    assert (tmp_path / "predictions_sub1.csv").read_bytes() == final
    assert (tmp_path / "predictions_sub1_exploratory.csv").exists()
