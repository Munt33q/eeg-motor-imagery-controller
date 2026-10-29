"""Evaluate the saved model once on the untouched final run (12).

Usage:
    python scripts/evaluate.py --subject 1 --config configs/default.json \
        [--threshold 0.7] [--exploratory]

Writes artifacts/predictions_sub<id>.csv, artifacts/metrics_sub<id>.json,
and (if --threshold differs from the config default) marks the run
exploratory in the metrics JSON. Final metrics use the frozen threshold.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from eeg_controller.config import load_config
from eeg_controller.data import check_run_supported
from eeg_controller.evaluate import (
    build_split_info,
    build_predictions_frame,
    exclusion_summary_table,
    majority_baseline_label,
)
from eeg_controller.model import load_model
from eeg_controller.preprocess import load_and_preprocess_run

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subject", type=int, required=True)
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default.json"))
    ap.add_argument("--data-dir", default=str(PROJECT_ROOT / "data"))
    ap.add_argument("--artifacts-dir", default=str(PROJECT_ROOT / "artifacts"))
    ap.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="decision threshold; omit to use the frozen config default",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    data_dir = Path(args.data_dir)
    artifacts = Path(args.artifacts_dir)

    threshold = cfg.controller.threshold if args.threshold is None else float(args.threshold)
    exploratory = args.threshold is not None and not np.isclose(
        threshold, cfg.controller.threshold
    )

    split = build_split_info(cfg, args.subject, pd.read_csv(data_dir / f"trials_subj{args.subject}.csv"))
    if split.overlap:
        raise SystemExit("dev/final run overlap detected; fix config before evaluating")
    print(
        f"Split: dev runs {split.dev_runs} ({split.dev_trials} trials) | "
        f"final runs {split.final_runs} ({split.final_trials} trials)"
    )

    model = load_model(artifacts / f"model_sub{args.subject}.joblib")
    print("Model artifact loaded.")

    # Majority baseline from development labels only.
    y_dev = []
    for run in cfg.data.dev_runs:
        t = pd.read_csv(data_dir / f"trials_subj{args.subject}.csv")
        y_dev.extend(t.loc[t["run_id"] == run, "label"].tolist())
    majority_label = majority_baseline_label(np.asarray(y_dev, dtype=np.int64))
    print(f"Majority baseline label (dev only): {majority_label}")

    pred_frames, exclusion_parts = [], []
    metrics_by_run = {}
    for run in cfg.data.final_runs:
        check_run_supported(run)
        trials = pd.read_csv(data_dir / f"trials_subj{args.subject}.csv")
        trials_run = trials[trials["run_id"] == run]
        if "edf_relpath" in trials_run.columns:
            edf_path = data_dir / str(trials_run["edf_relpath"].iloc[0])
        else:
            edf_path = data_dir / "raw" / str(trials_run["edf_file"].iloc[0])
        if not edf_path.exists():
            raise SystemExit(
                f"EDF not found: {edf_path}; re-run scripts/prepare_data.py"
            )
        print(f"Evaluating final run {run}: {edf_path.name}")
        epochs, excl = load_and_preprocess_run(edf_path, cfg, run, args.subject)
        preds, metrics = build_predictions_frame(
            cfg, model, epochs, run, args.subject, threshold
        )
        pred_frames.append(preds)
        metrics_by_run[str(run)] = metrics
        if len(excl):
            exclusion_parts.append(excl)

    predictions = pd.concat(pred_frames, ignore_index=True)
    combined_metrics = compute_combined_metrics(
        predictions, cfg.data.label_names, majority_label
    )
    combined_metrics["subject_id"] = args.subject
    combined_metrics["final_runs"] = list(cfg.data.final_runs)
    combined_metrics["dev_runs"] = list(cfg.data.dev_runs)
    combined_metrics["threshold"] = float(threshold)
    combined_metrics["threshold_is_frozen_default"] = not exploratory
    combined_metrics["per_run"] = metrics_by_run
    combined_metrics["majority_baseline"] = {
        "label": majority_label,
        "dev_source_runs": list(cfg.data.dev_runs),
        "test_accuracy": combined_metrics["majority_baseline"]["test_accuracy"],
    }

    if exclusion_parts:
        exclusions = exclusion_summary_table(exclusion_parts)
    else:
        exclusions = pd.DataFrame(columns=["run_id", "label", "exclusion_reason", "count"])

    pred_path = artifacts / f"predictions_sub{args.subject}.csv"
    predictions.to_csv(pred_path, index=False)
    metrics_path = artifacts / f"metrics_sub{args.subject}.json"
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(combined_metrics, fh, indent=2)
    excl_path = artifacts / f"exclusions_final_sub{args.subject}.csv"
    exclusions.to_csv(excl_path, index=False)

    print(f"\nWrote {pred_path} ({len(predictions)} trials)")
    print(f"Wrote {metrics_path}")
    print(f"Wrote {excl_path}")
    print_metrics_summary(combined_metrics)
    return 0


def compute_combined_metrics(predictions, label_names, majority_label):
    from eeg_controller.evaluate import compute_metrics

    metrics = compute_metrics(predictions, label_names, majority_label)
    return metrics


def print_metrics_summary(m: dict) -> None:
    c = m["confusion"]
    print("\n=== Final-run metrics (all eligible trials) ===")
    print(f"n trials:            {m['n_trials']}")
    acc = m["accuracy"]
    bacc = m["balanced_accuracy"]
    print(f"accuracy:            {acc:.3f}" if acc is not None else "accuracy:            n/a")
    print(
        f"balanced accuracy:   {bacc:.3f}" if bacc is not None else "balanced accuracy:   n/a"
    )
    print(f"confusion (rows=true, cols=pred):")
    print(f"  left : left={c['matrix']['true_left_pred_left']}, right={c['matrix']['true_left_pred_right']}")
    print(f"  right: left={c['matrix']['true_right_pred_left']}, right={c['matrix']['true_right_pred_right']}")
    mb = m["majority_baseline"]
    mb_acc = mb["test_accuracy"]
    print(
        f"majority baseline:   {mb_acc:.3f} (label={mb['label']})"
        if mb_acc is not None
        else "majority baseline:   n/a"
    )
    print(f"coverage:            {m['coverage']:.3f}")
    aa = m["accepted_accuracy"]
    print(
        f"accepted accuracy:   {aa:.3f}"
        if aa is not None
        else "accepted accuracy:   null (no accepted commands)"
    )
    print(f"incorrect issued:    {m['incorrect_issued_commands']} / {m['n_trials']}")
    print(f"rejected trials:     {m['rejected_trials']}")
    inf = m["inference_ms"]
    print(f"inference ms median: {inf['median']:.3f}, p95: {inf['p95']:.3f}")


if __name__ == "__main__":
    sys.exit(main())
