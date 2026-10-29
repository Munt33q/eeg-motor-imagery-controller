"""Prediction-to-command mapping and cursor game state.

Controller contract (guide section 7):
- :func:`choose_command` takes ONLY model probabilities; true labels are
  never an input, so labels cannot select or correct commands.
- Default threshold 0.70 is an adjustable demo parameter, not a validated
  operating point. Abstention (HOLD) means "below threshold", not "rest".
"""

from __future__ import annotations

import numpy as np

from . import COMMAND_HOLD, COMMAND_LEFT, COMMAND_RIGHT

DEFAULT_THRESHOLD = 0.70
DEFAULT_STEP = 0.15
CURSOR_MIN = -1.0
CURSOR_MAX = 1.0


def choose_command(
    probabilities, classes, threshold: float = DEFAULT_THRESHOLD
) -> str:
    """Map model probabilities to LEFT / RIGHT / HOLD.

    ``probabilities`` is aligned with ``classes`` (e.g. ``model.classes_``);
    column meaning is never assumed. A command is accepted only if the max
    probability reaches ``threshold``; otherwise the controller abstains
    with HOLD. At threshold 0.50 an exact tie produces HOLD.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)
    classes = list(classes)
    if probabilities.shape != (len(classes),):
        raise ValueError(
            f"probabilities shape {probabilities.shape} does not match "
            f"{len(classes)} classes"
        )
    if not np.isfinite(probabilities).all():
        raise ValueError("probabilities must be finite")
    if len(set(classes)) != len(classes):
        raise ValueError("classes must be unique")

    best_idx = int(np.argmax(probabilities))  # ties -> first index
    best_p = float(probabilities[best_idx])
    if best_p < threshold:
        return COMMAND_HOLD
    if best_p == 0.5 and threshold <= 0.5:
        return COMMAND_HOLD  # exact tie at or below 0.5 threshold abstains
    best_class = classes[best_idx]
    if best_class == 0:
        return COMMAND_LEFT
    if best_class == 1:
        return COMMAND_RIGHT
    raise ValueError(f"class {best_class!r} is not a known left/right class")


def advance_cursor(
    x: float, command: str, step: float = DEFAULT_STEP
) -> float:
    """Move the cursor one step; clamp to [-1.0, 1.0]."""
    if command == COMMAND_LEFT:
        x = x - step
    elif command == COMMAND_RIGHT:
        x = x + step
    elif command == COMMAND_HOLD:
        pass
    else:
        raise ValueError(f"unknown command {command!r}")
    return float(min(max(x, CURSOR_MIN), CURSOR_MAX))
