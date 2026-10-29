"""Pure logic for the recorded-replay demo app (no Streamlit imports).

Everything here is deterministic and unit-testable. The Streamlit layer
(``app.py``) only handles widgets, rendering, and session state.

Label isolation: replay events carry timestamps only. Predictions come
from the model at decision time; true labels never enter this module and
cannot select or correct a command.
"""

from __future__ import annotations

import time

import numpy as np

from .controller import choose_command
from .model import Pipeline, predict_with_scores

DISPLAY_CHANNELS = ("C3", "Cz", "C4")


def build_replay_events(
    onsets_s, decision_delay_s: float, duration_buffer_s: float = 1.0
) -> tuple[list[dict], float]:
    """Build the timestamp-only event list for :class:`ReplayState`.

    Returns (events, duration_s). Each event carries its onset, its decision
    time (onset + decision_delay), and a ``trial_index`` into the *eligible*
    epoch array. No labels, no commands.
    """
    onsets = [float(o) for o in onsets_s]
    if not onsets:
        raise ValueError("no trial onsets; nothing to replay")
    if any(b <= a for a, b in zip(onsets, onsets[1:])):
        raise ValueError("onsets must be strictly increasing")
    if decision_delay_s <= 0:
        raise ValueError("decision_delay_s must be positive")
    events = [
        {
            "onset_s": onset,
            "decision_s": onset + float(decision_delay_s),
            "trial_index": i,
        }
        for i, onset in enumerate(onsets)
    ]
    duration = events[-1]["decision_s"] + float(duration_buffer_s)
    return events, float(duration)


def trial_prediction(
    model: Pipeline, epoch: np.ndarray, threshold: float
) -> tuple[str, float, float, int, float]:
    """Run live inference on one epoch and map probabilities to a command.

    Returns (command, p_left, p_right, predicted_label, inference_ms) where
    inference_ms measures actual model inference (predict + probability
    mapping) via ``perf_counter``. The caller is responsible for timestamp
    gating: this function must only be invoked once the trial's analysis
    window is fully available (onset + decision_delay).
    """
    if epoch.ndim != 2:
        raise ValueError("epoch must be 2-D [channels, samples]")
    t0 = time.perf_counter()
    preds, p_left, p_right = predict_with_scores(model, epoch[np.newaxis, ...])
    inference_ms = (time.perf_counter() - t0) * 1e3
    command = choose_command(
        [float(p_left[0]), float(p_right[0])], classes=(0, 1), threshold=threshold
    )
    return command, float(p_left[0]), float(p_right[0]), int(preds[0]), float(inference_ms)


def next_trial_time(state, events: list[dict]) -> float | None:
    """Earliest unprocessed decision time strictly after the replay clock.

    Returns None when every event has been processed. Used by the Next-trial
    control; the caller then processes exactly this one event.
    """
    remaining = [
        e["decision_s"]
        for i, e in enumerate(events)
        if i not in state.processed_event_ids and e["decision_s"] > state.time_s
    ]
    return min(remaining) if remaining else None


def waveform_uv_window(
    data: np.ndarray,
    channel_names: tuple[str, ...],
    t_s: float,
    fs: float,
    window_s: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Display-only extraction: C3/Cz/C4 in µV for samples with t <= now.

    Classifier input is never touched: ``data`` stays in volts internally
    and is multiplied by 1e6 only here, for plotting.
    """
    indices = [channel_names.index(ch) for ch in DISPLAY_CHANNELS]
    fs = float(fs)
    s1 = min(data.shape[1], int(np.ceil(t_s * fs)))  # include sample at t
    s0 = max(0, s1 - int(round(window_s * fs)))
    times = np.arange(s0, s1, dtype=np.float64) / fs
    uv = data[indices, s0:s1] * 1e6
    return times, uv, DISPLAY_CHANNELS
