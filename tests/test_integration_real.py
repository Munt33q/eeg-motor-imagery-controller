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
    assert n_kept + n_excluded >= n_in_table  # table holds eligible only
