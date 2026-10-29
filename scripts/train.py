"""Fit the CSP+LDA pipeline on development runs and save a trusted artifact.

Usage:
    python scripts/train.py --subject 1 --config configs/default.json

Split contract (guide section 6): fit on dev runs (4+8); run 12 is never
touched here. The artifact plus JSON metadata land in artifacts/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from eeg_controller.config import load_config
from eeg_controller.data import DataError, check_run_supported
from eeg_controller.model import build_pipeline, provenance_metadata, save_model
from eeg_controller.preprocess import load_and_preprocess_run

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subject", type=int, required=True)
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default.json"))
    ap.add_argument("--data-dir", default=str(PROJECT_ROOT / "data"))
    ap.add_argument("--artifacts-dir", default=str(PROJECT_ROOT / "artifacts"))
    args = ap.parse_args()

    cfg = load_config(args.config)
    data_dir = Path(args.data_dir)
    trials = pd.read_csv(data_dir / f"trials_subj{args.subject}.csv")

    X_parts, y_parts = [], []
    for run in cfg.data.dev_runs:
        check_run_supported(run)
        trials_run = trials[trials["run_id"] == run]
        edf_relpath = str(trials_run["edf_relpath"].iloc[0]) if "edf_relpath" in trials_run else None
        if edf_relpath:
            edf_path = data_dir / edf_relpath
        else:
            edf_path = data_dir / "raw" / str(trials_run["edf_file"].iloc[0])
        if not edf_path.exists():
            raise SystemExit(
                f"EDF not found: {edf_path}; re-run scripts/prepare_data.py"
            )
        print(f"Loading dev run {run}: {edf_path.name}")
        epochs, _ = load_and_preprocess_run(edf_path, cfg, run, args.subject)
        X_parts.append(epochs.X)
        y_parts.append(epochs.y)
    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    if len(np.unique(y)) < 2:
        raise DataError(
            "development epochs contain a single class; cannot fit CSP+LDA"
        )
    print(f"Dev set: {X.shape[0]} trials, {X.shape[1]} ch x {X.shape[2]} samples")
    print(f"  left={int((y == 0).sum())}, right={int((y == 1).sum())}")

    model = build_pipeline(cfg)
    model.fit(X, y)
    print(f"Fitted pipeline: {model}")
    print(f"CSP classes_: {model.classes_}")

    artifacts = Path(args.artifacts_dir)
    model_path = artifacts / f"model_sub{args.subject}.joblib"
    meta_path = artifacts / f"model_sub{args.subject}_metadata.json"
    metadata = provenance_metadata(
        cfg, args.subject, list(cfg.data.dev_runs), data_dir / f"trials_subj{args.subject}.csv", cfg.seed
    )
    metadata["n_train_trials"] = int(X.shape[0])
    metadata["n_train_left"] = int((y == 0).sum())
    metadata["n_train_right"] = int((y == 1).sum())
    save_model(model, metadata, model_path, meta_path)
    print(f"Saved {model_path}")
    print(f"Saved {meta_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
