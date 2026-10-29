"""Recorded EEG replay — subject-specific motor-imagery classifier demo.

Run with:
    .\\.venv\\Scripts\\python.exe -m streamlit run app.py

What this app is:
- A cue-timed **replay of recorded EEG** (PhysioNet EEGMMIDB run 12), not
  live control. Predictions are issued only after each trial's 3-second
  analysis window is fully replayed (onset + 3 s), never earlier.
- Threshold is **fixed at the frozen default** (0.70); there is no tuning
  control. Label-driven scoring appears only in the Results tab.

Label isolation: replay events carry timestamps only. Commands come from
model probabilities at decision time; true labels appear exclusively in
the Results tab as aggregate metrics.
"""

from __future__ import annotations

import json
import time as time_module
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from eeg_controller.config import load_config
from eeg_controller.controller import advance_cursor
from eeg_controller.demo import (
    build_replay_events,
    next_trial_time,
    trial_prediction,
    waveform_uv_window,
)
from eeg_controller.model import load_model, validate_artifact_metadata
from eeg_controller.preprocess import load_and_preprocess_run_data
from eeg_controller.replay import ReplayState

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "default.json"
TICK_S = 0.25  # fragment period; wall-clock dt is measured per tick

st.set_page_config(
    page_title="EEG Motor Imagery Controller",
    page_icon="🧠",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Cached resource loading (keyed by subject so switching never mixes state)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading subject data...")
def load_subject_resources(subject_id: int, data_dir: str, artifacts_dir: str):
    """Load one subject's model, metadata, recording, and prediction records.

    Raises a descriptive error if artifacts are missing or fail the
    contract check (wrong subject, split, channels, or preprocessing).
    """
    cfg = load_config(CONFIG_PATH)
    data_dir, artifacts_dir = Path(data_dir), Path(artifacts_dir)

    model_path = artifacts_dir / f"model_sub{subject_id}.joblib"
    meta_path = artifacts_dir / f"model_sub{subject_id}_metadata.json"
    if not model_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"Missing model artifact for subject {subject_id}. "
            "Run scripts/train.py first."
        )
    with open(meta_path, encoding="utf-8") as fh:
        metadata = json.load(fh)
    validate_artifact_metadata(metadata, cfg, subject_id)  # contract check
    model = load_model(model_path)

    trials = pd.read_csv(data_dir / f"trials_subj{subject_id}.csv")
    final_run = cfg.data.final_runs[0]
    trials_run = trials[trials["run_id"] == final_run]
    if trials_run.empty:
        raise FileNotFoundError(
            f"No trial table rows for subject {subject_id} final run "
            f"{final_run}; run scripts/prepare_data.py first."
        )
    edf_path = data_dir / str(trials_run["edf_relpath"].iloc[0])
    if not edf_path.exists():
        raise FileNotFoundError(
            f"EDF not found: {edf_path}. Run scripts/prepare_data.py first."
        )
    data, events, epochs, _ = load_and_preprocess_run_data(
        edf_path, cfg, final_run, subject_id
    )
    if len(epochs.X) == 0:
        raise ValueError(f"Subject {subject_id} final run has no eligible trials.")

    # Eligible-trial table for this run (chronological), joined with labels
    # for the Results tab ONLY. Never passed to the replay path.
    eligible = trials_run[trials_run["trial_id"].isin(epochs.metadata["trial_id"])]
    eligible = eligible.sort_values("trial_id").reset_index(drop=True)

    metrics_path = artifacts_dir / f"metrics_sub{subject_id}.json"
    metrics = (
        json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics_path.exists()
        else None
    )
    return {
        "cfg": cfg,
        "model": model,
        "metadata": metadata,
        "data": data,               # volts, [9, n_samples] — display only
        "epochs": epochs,           # classifier input (volts)
        "eligible": eligible,       # trial table incl. labels (Results tab only)
        "metrics": metrics,
        "threshold": float(metadata["config"]["controller"]["threshold"]),
        "final_run": final_run,
    }


def reset_session_state() -> None:
    """Clear all replay state (used on subject switch and Reset)."""
    for key in (
        "replay", "cursor_x", "history", "play", "speed",
        "last_tick_s", "decisions",
    ):
        st.session_state.pop(key, None)


# ---------------------------------------------------------------------------
# Sidebar: subject selection and disclosure
# ---------------------------------------------------------------------------
st.sidebar.header("Recorded EEG replay")
st.sidebar.caption("Subject-specific motor imagery classifier")

subject = st.sidebar.selectbox(
    "Subject",
    [1, 2, 3, 4, 5],
    index=0,
    help="Each subject has its own model trained on runs 4+8; run 12 is replayed.",
)
if st.session_state.get("loaded_subject") != subject:
    reset_session_state()
    st.session_state.loaded_subject = subject

try:
    res = load_subject_resources(
        subject, str(PROJECT_ROOT / "data"), str(PROJECT_ROOT / "artifacts")
    )
except (FileNotFoundError, ValueError) as exc:
    st.error(f"**Cannot load subject {subject}:** {exc}")
    st.info(
        "Run these from the repository root, then reload:\n\n"
        "```\n"
        ".\\.venv\\Scripts\\python.exe scripts/prepare_data.py --subjects 1 2 3 4 5\n"
        ".\\.venv\\Scripts\\python.exe scripts/train.py --subject 1\n"
        ".\\.venv\\Scripts\\python.exe scripts/evaluate.py --subject 1\n"
        "```\n"
        "(repeat training/evaluation for each subject)"
    )
    st.stop()

cfg = res["cfg"]
threshold = res["threshold"]
epochs = res["epochs"]

st.sidebar.success(
    f"Model loaded: subject {subject} (runs 4+8 → run {res['final_run']})"
)
st.sidebar.metric("Frozen threshold", f"{threshold:.2f}")
st.sidebar.caption(
    "Threshold is the frozen demo default — no tuning in this app. "
    "Scores are model probabilities, not calibrated confidence."
)
st.sidebar.caption(
    "⚠️ Recorded, cue-gated EEG replay. Predictions fire only after each "
    "3-second analysis window completes. Not live control."
)

# ---------------------------------------------------------------------------
# Session state initialization (fresh ReplayState after reset/subject switch)
# ---------------------------------------------------------------------------
events, duration_s = build_replay_events(
    epochs.metadata["onset_s"], cfg.controller.decision_delay_s
)
fs = cfg.preprocess.sfreq_hz

if "replay" not in st.session_state:
    st.session_state.replay = ReplayState(duration_s=duration_s, events=events, speed=1.0)
    st.session_state.cursor_x = 0.0
    st.session_state.history = []      # issued commands (trial, time, command)
    st.session_state.decisions = {}    # trial_index -> prediction dict
    st.session_state.play = False
    st.session_state.speed = 1.0
    st.session_state.last_tick_s = None

replay: ReplayState = st.session_state.replay


def process_due_decisions() -> None:
    """Issue commands for due trials exactly once (timestamp-gated)."""
    for event in replay.process_due():
        ti = event["trial_index"]
        if ti in st.session_state.decisions:
            continue  # double protection against duplicate moves
        command, p_left, p_right, pred, inference_ms = trial_prediction(
            res["model"], epochs.X[ti], threshold
        )
        st.session_state.decisions[ti] = {
            "command": command, "p_left": p_left, "p_right": p_right,
            "predicted_label": pred, "inference_ms": inference_ms,
        }
        # One move per trial; cursor resets to center for each trial.
        if command != "HOLD":
            st.session_state.cursor_x = advance_cursor(
                0.0, command, step=cfg.controller.cursor_step
            )
        st.session_state.history.append(
            {"trial": ti + 1, "decision_s": event["decision_s"], "command": command}
        )


# ---------------------------------------------------------------------------
# Main area: title, controls
# ---------------------------------------------------------------------------
st.title("Recorded EEG replay — subject-specific motor imagery classifier")
st.caption(
    "Held-out run 12 replay. LEFT/RIGHT commands move the cursor one step "
    "from center; HOLD keeps it centered. One move per trial."
)

c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1.2, 1.6])
if c1.button("▶ Play", width="stretch", disabled=st.session_state.play):
    st.session_state.play = True
    st.session_state.last_tick_s = None
    replay.resume(st.session_state.speed)
    st.rerun()
if c2.button("⏸ Pause", width="stretch", disabled=not st.session_state.play):
    st.session_state.play = False
    replay.pause()
    st.rerun()
if c3.button("⏮ Reset", width="stretch"):
    reset_session_state()
    st.rerun()
speed = c4.select_slider(
    "Speed", options=[1.0, 2.0, 4.0], value=float(st.session_state.speed),
    format_func=lambda v: f"{v:g}×",
)
if speed != st.session_state.speed:
    st.session_state.speed = float(speed)
    replay.set_speed(float(speed)) if st.session_state.play else setattr(replay, "speed", float(speed))
if c5.button("⏭ Next trial", width="stretch"):
    target = next_trial_time(replay, events)
    if target is None:
        st.toast("All trials already processed.")
    else:
        replay.time_s = target
        process_due_decisions()
    st.rerun()


# ---------------------------------------------------------------------------
# Timed fragment: advances the clock AND renders everything time-dependent.
# Elements inside a fragment are redrawn on each fragment rerun while the
# rest of the app (controls above, results below) stays put.
# ---------------------------------------------------------------------------
@st.fragment(run_every=TICK_S if st.session_state.play else None)
def replay_area():
    if st.session_state.play:
        now = time_module.perf_counter()
        last = st.session_state.last_tick_s
        dt_wall = TICK_S if last is None else min(now - last, 1.0)
        st.session_state.last_tick_s = now
        replay.tick(dt_wall)
        process_due_decisions()

    # --- Waveform (display-only µV; only samples with t <= replay time) ---
    times, uv, chans = waveform_uv_window(
        res["data"], cfg.preprocess.channels, replay.time_s, fs
    )
    fig = go.Figure()
    colors = {"C3": "#636EFA", "Cz": "#EF553B", "C4": "#00CC96"}
    offsets = {"C3": 40, "Cz": 0, "C4": -40}
    for i, ch in enumerate(chans):
        fig.add_trace(
            go.Scatter(
                x=times, y=uv[i] + offsets[ch], mode="lines", name=ch,
                line=dict(width=1, color=colors[ch]),
                hovertemplate="%{x:.2f} s, %{y:.1f} µV<extra>" + ch + "</extra>",
            )
        )
    # Cue markers: only those inside the displayed window (timestamps only),
    # and a fixed x-range so old markers don't compress the trace.
    win = 2.0
    for e in replay.visible_events():
        if replay.time_s - win <= e["onset_s"] <= replay.time_s:
            fig.add_vline(x=e["onset_s"], line_width=1, line_dash="dot",
                          line_color="gray", opacity=0.6)
    fig.update_layout(
        height=300, margin=dict(l=10, r=10, t=30, b=10),
        title=f"C3 / Cz / C4 — last 2 s (µV), replay time {replay.time_s:.2f} s",
        xaxis_title="Recording time (s)",
        yaxis_title="Amplitude (µV, offset per channel)",
        xaxis_range=[max(0.0, replay.time_s - win - 0.3), replay.time_s + 0.2],
        legend=dict(orientation="h", y=1.02, x=1),
    )
    st.plotly_chart(fig, width="stretch")

    # --- Cursor + current prediction -------------------------------------
    col_cur, col_pred = st.columns([2, 3])
    with col_cur:
        st.subheader("Cursor")
        x = st.session_state.cursor_x
        cfig = go.Figure()
        cfig.add_hline(y=0, line_width=1, line_color="#DDD")
        cfig.add_vline(x=0, line_width=1, line_color="#DDD")
        cfig.add_trace(go.Scatter(
            x=[x], y=[0], mode="markers",
            marker=dict(size=26, color="#636EFA"),
        ))
        cfig.update_layout(
            height=150, margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(range=[-1.15, 1.15], visible=False),
            yaxis=dict(range=[-0.6, 0.6], visible=False),
        )
        st.plotly_chart(cfig, width="stretch")
        lc, rc = st.columns(2)
        lc.caption("◀ LEFT target (−1)")
        rc.caption("RIGHT target (+1) ▶")

    with col_pred:
        st.subheader("Current prediction")
        if st.session_state.decisions:
            d = st.session_state.decisions[max(st.session_state.decisions)]
            color = {"LEFT": "#EF553B", "RIGHT": "#00CC96", "HOLD": "#888"}[d["command"]]
            st.markdown(
                f"<div style='font-size:2.2em;font-weight:700;color:{color}'>"
                f"{d['command']}</div>",
                unsafe_allow_html=True,
            )
            m1, m2, m3 = st.columns(3)
            m1.metric("p(left)", f"{d['p_left']:.3f}")
            m2.metric("p(right)", f"{d['p_right']:.3f}")
            m3.metric("inference", f"{d['inference_ms']:.2f} ms")
            st.progress(
                float(max(d["p_left"], d["p_right"])),
                text=f"max probability vs. threshold {threshold:.2f}",
            )
        else:
            st.info(
                "Press ▶ Play or ⏭ Next trial — the first decision arrives "
                "3 s after the first cue."
            )
        st.caption(f"Replay clock: {replay.time_s:.2f} s / {duration_s:.1f} s")

    if st.session_state.history:
        st.dataframe(
            pd.DataFrame(st.session_state.history),
            width="stretch", hide_index=True, height=170,
        )


replay_area()

st.caption(
    "Recorded-data replay — predictions are computed live from the model at "
    "each decision time but are fully determined by the recording. Labels "
    "never influence commands. Not a medical device; not live control."
)

# ============================================================================
# Results tab: saved held-out metrics (labels appear only here)
# ============================================================================
st.divider()
st.header("Evaluation results — held-out run 12 (labels appear only here)")
metrics = res["metrics"]
if metrics is None:
    st.warning("No metrics file found; run scripts/evaluate.py to generate results.")
else:
    tab_overview, tab_confusion, tab_trials = st.tabs(
        ["Overview", "Confusion matrix", "Trial-level predictions"]
    )

    with tab_overview:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Accuracy (all 15 trials)", f"{metrics['accuracy']:.3f}")
        k2.metric("Balanced accuracy", f"{metrics['balanced_accuracy']:.3f}")
        k3.metric("Coverage", f"{metrics['coverage']:.3f}")
        acc_acc = metrics["accepted_accuracy"]
        k4.metric(
            "Accepted-command accuracy",
            f"{acc_acc:.3f}" if acc_acc is not None else "null",
        )
        k5, k6, k7, k8 = st.columns(4)
        k5.metric("Incorrect issued", f"{metrics['incorrect_issued_commands']} / {metrics['n_trials']}")
        k6.metric("Rejected (HOLD)", metrics["rejected_trials"])
        k7.metric("Majority baseline", f"{metrics['majority_baseline']['test_accuracy']:.3f}")
        k8.metric("Median inference", f"{metrics['inference_ms']['median']:.2f} ms")
        st.caption(
            "Accepted accuracy covers issued commands only; coverage is always "
            "reported alongside it. Neither alone is adequate."
        )

    with tab_confusion:
        cm = metrics["confusion"]["matrix"]
        cdf = pd.DataFrame(
            {
                "pred LEFT": [cm["true_left_pred_left"], cm["true_right_pred_left"]],
                "pred RIGHT": [cm["true_left_pred_right"], cm["true_right_pred_right"]],
            },
            index=["true LEFT", "true RIGHT"],
        )
        st.table(cdf)
        st.caption(f"n = {metrics['n_trials']} eligible trials; fixed left/right order.")

    with tab_trials:
        preds_path = PROJECT_ROOT / "artifacts" / f"predictions_sub{subject}.csv"
        if preds_path.exists():
            preds = pd.read_csv(preds_path)
            preds["correct"] = preds["true_label"] == preds["predicted_label"]
            st.dataframe(preds, width="stretch", hide_index=True, height=360)
            d1, d2 = st.columns(2)
            d1.download_button(
                "Download predictions CSV",
                data=preds_path.read_bytes(),
                file_name=preds_path.name,
                mime="text/csv",
            )
        else:
            st.info("No predictions CSV found for this subject.")
        d2.download_button(
            "Download metrics JSON",
            data=json.dumps(metrics, indent=2),
            file_name=f"metrics_sub{subject}.json",
            mime="application/json",
        )

st.caption(
    "These metrics come from the offline evaluation (scripts/evaluate.py) on "
    "the untouched final run. The replay panel above never uses labels; this "
    "tab does, for reporting only."
)
