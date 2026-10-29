"""Held-out-run evaluation: predictions, metrics, prediction records.

Contract (guide sections 6-7, 9):
- Refit the fixed pipeline on development runs (4+8); evaluate exactly once
  on the untouched final run (12).
- Labels feed metrics and reporting only — never command generation.
- Metrics distinguish all-trial accuracy from accepted-command accuracy and
  always report coverage alongside accepted accuracy.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .controller import COMMAND_HOLD, choose_command, advance_cursor
from .model import Pipeline, predict_with_scores

COMMANDS = ("LEFT", "RIGHT", "HOLD")


@dataclass
class SplitInfo:
    """Identity audit for the dev/final run split."""

    subject_id: int
    dev_runs: tuple[int, ...]
    final_runs: tuple[int, ...]
    dev_trials: int
    final_trials: int
    overlap: bool


def build_split_info(cfg: Config, subject_id: int, trial_table: pd.DataFrame) -> SplitInfo:
    """Verify no run identity is shared between dev and final sets."""
    dev, final = cfg.data.dev_runs, cfg.data.final_runs
    overlap = bool(set(dev) & set(final))
    sub = trial_table[trial_table["subject_id"] == subject_id]
    n_dev = int(sub["run_id"].isin(dev).sum())
    n_final = int(sub["run_id"].isin(final).sum())
    return SplitInfo(
        subject_id=subject_id,
        dev_runs=tuple(dev),
        final_runs=tuple(final),
        dev_trials=n_dev,
        final_trials=n_final,
        overlap=overlap,
    )


def build_predictions_frame(
    cfg: Config,
    model: Pipeline,
    epochs,
    run_id: int,
    subject_id: int,
    threshold: float,
) -> tuple[pd.DataFrame, dict]:
    """Predict on every eligible epoch and attach controller decisions.

    Produces one row per *eligible* trial with inference timing measured
    around actual model inference via ``perf_counter``. Controller commands
    are computed from model probabilities only — labels are never passed to
    the controller.
    """
    X = epochs.X
    if len(X) == 0:
        raise ValueError(f"run {run_id}: no eligible trials to predict on")
    preds, p_left, p_right = predict_with_scores(model, X)
    labels = epochs.y

    records = []
    inference_ms_all: list[float] = []
    commands = []
    accepted_flags = []
    for i in range(len(X)):
        probs = np.array([p_left[i], p_right[i]])
        t0 = time.perf_counter()
        command = choose_command(probs, classes=(0, 1), threshold=threshold)
        inference_ms = (time.perf_counter() - t0) * 1e3
        inference_ms_all.append(inference_ms)
        accepted = command != COMMAND_HOLD
        commands.append(command)
        accepted_flags.append(accepted)
        records.append(
            {
                "trial_id": int(epochs.metadata.iloc[i]["trial_id"]),
                "subject_id": subject_id,
                "run_id": run_id,
                "onset_s": float(epochs.metadata.iloc[i]["onset_s"]),
                "decision_s": float(
                    epochs.metadata.iloc[i]["onset_s"]
                    + cfg.controller.decision_delay_s
                ),
                "true_label": int(labels[i]),
                "predicted_label": int(preds[i]),
                "p_left": float(p_left[i]),
                "p_right": float(p_right[i]),
                "inference_ms": float(inference_ms),
                "command": command,
                "accepted": bool(accepted),
                "threshold": float(threshold),
            }
        )

    predictions = pd.DataFrame(records)

    # Replay simulation: one move per trial, cursor resets to center per trial.
    cursor_positions = []
    x = 0.0
    for command in commands:
        x = advance_cursor(x, command, step=cfg.controller.cursor_step)
        cursor_positions.append(x)
    predictions["cursor_after"] = cursor_positions

    metrics = compute_metrics(
        predictions, label_names=cfg.data.label_names, majority_label=int(pd.Series(labels).mode()[0])
    )
    return predictions, metrics


def majority_baseline_label(y_dev: np.ndarray) -> int:
    """Majority class from *development* labels only (no test peeking)."""
    if len(y_dev) == 0:
        raise ValueError("development labels are empty")
    return int(pd.Series(y_dev).mode()[0])


def compute_metrics(
    predictions: pd.DataFrame, label_names: dict[int, str], majority_label: int | None = None
) -> dict:
    """Compute the guide's metric set from prediction records.

    All metrics use every eligible trial (rejected commands are never
    counted as correct classifications).
    """
    n = len(predictions)
    y_true = predictions["true_label"].to_numpy()
    y_pred = predictions["predicted_label"].to_numpy()
    accepted = predictions["accepted"].to_numpy(dtype=bool)
    commands = predictions["command"].to_numpy()

    correct = y_true == y_pred
    accuracy = float(correct.mean()) if n else None
    per_class_recall = {}
    for lbl in (0, 1):
        mask = y_true == lbl
        per_class_recall[lbl] = float(correct[mask].mean()) if mask.any() else None
    recalls = [r for r in per_class_recall.values() if r is not None]
    balanced_accuracy = float(np.mean(recalls)) if recalls else None

    tp_l = int(((y_true == 0) & (y_pred == 0)).sum())
    tp_r = int(((y_true == 1) & (y_pred == 1)).sum())
    fp_l = int(((y_true == 1) & (y_pred == 0)).sum())
    fp_r = int(((y_true == 0) & (y_pred == 1)).sum())
    confusion = {
        "n": n,
        "order": ["left", "right"],
        "matrix": {
            "true_left_pred_left": tp_l,
            "true_left_pred_right": fp_r,
            "true_right_pred_left": fp_l,
            "true_right_pred_right": tp_r,
},
        "per_class_recall": {
            "left": per_class_recall[0],
            "right": per_class_recall[1],
        },
    }

    coverage = float(accepted.mean()) if n else None
    accepted_correct = int((correct & accepted).sum())
    accepted_accuracy = (
        float(accepted_correct / accepted.sum()) if accepted.sum() else None
    )
    incorrect_issued = int((~correct & accepted).sum())
    incorrect_issued_rate = float(incorrect_issued / n) if n else None

    majority_accuracy = None
    if majority_label is not None and n:
        majority_accuracy = float((y_true == majority_label).mean())

    inf_ms = predictions["inference_ms"].to_numpy(dtype=float)
    median_inf = float(np.median(inf_ms)) if n else None
    p95_inf = float(np.percentile(inf_ms, 95)) if n else None

    return {
        "n_trials": n,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "confusion": confusion,
        "majority_baseline": {
            "label": majority_label,
            "test_accuracy": majority_accuracy,
        },
        "coverage": coverage,
        "accepted_accuracy": accepted_accuracy,
        "incorrect_issued_commands": incorrect_issued,
        "incorrect_issued_rate": incorrect_issued_rate,
        "rejected_trials": int((~accepted).sum()),
        "inference_ms": {"median": median_inf, "p95": p95_inf},
        "note_inference_vs_delay": (
            "inference_ms is model compute time only; the cue-to-decision "
            "delay is the fixed 3.0 s epoch window and is not comparable"
        ),
    }


def exclusion_summary_table(exclusion_summaries: list[pd.DataFrame]) -> pd.DataFrame:
    """Aggregate per-run exclusion summaries; count exclusions by run/class."""
    if not exclusion_summaries:
        return pd.DataFrame(columns=["run_id", "label", "exclusion_reason", "count"])
    df = pd.concat(exclusion_summaries, ignore_index=True)
    return (
        df.groupby(["run_id", "label", "exclusion_reason"], as_index=False)["count"]
        .sum()
        .sort_values(["run_id", "label", "exclusion_reason"])
        .reset_index(drop=True)
    )
