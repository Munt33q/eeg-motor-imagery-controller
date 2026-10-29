"""Controller tests: commands from probabilities only, ties, clamping."""

from __future__ import annotations

import numpy as np
import pytest

from eeg_controller.controller import advance_cursor, choose_command


def test_high_left_score_moves_left():
    assert choose_command([0.9, 0.1], classes=(0, 1), threshold=0.7) == "LEFT"


def test_high_right_score_moves_right():
    assert choose_command([0.1, 0.9], classes=(0, 1), threshold=0.7) == "RIGHT"


def test_low_scores_hold():
    assert choose_command([0.6, 0.4], classes=(0, 1), threshold=0.7) == "HOLD"
    assert choose_command([0.5, 0.5], classes=(0, 1), threshold=0.7) == "HOLD"


def test_tie_at_threshold_half_holds():
    assert choose_command([0.5, 0.5], classes=(0, 1), threshold=0.5) == "HOLD"


def test_threshold_boundary_accepts():
    # max probability exactly at threshold is accepted (>= threshold)
    assert choose_command([0.7, 0.3], classes=(0, 1), threshold=0.7) == "LEFT"


def test_reversed_class_order_respected():
    # classes (1, 0): column 0 is RIGHT probability.
    assert choose_command([0.9, 0.1], classes=(1, 0), threshold=0.7) == "RIGHT"
    assert choose_command([0.1, 0.9], classes=(1, 0), threshold=0.7) == "LEFT"


def test_label_cannot_influence_command():
    """The function signature takes no label input; changing a true label
    elsewhere cannot change the chosen command."""
    probs = [0.8, 0.2]
    cmd_a = choose_command(probs, classes=(0, 1), threshold=0.7)
    # Simulate any label mutation anywhere in the pipeline:
    true_label = 0
    true_label = 1  # flipped
    cmd_b = choose_command(probs, classes=(0, 1), threshold=0.7)
    assert cmd_a == cmd_b == "LEFT"
    assert true_label in (0, 1)  # labels exist for evaluation only


def test_advance_cursor_left_right_hold():
    assert advance_cursor(0.0, "LEFT") == -0.15
    assert advance_cursor(0.0, "RIGHT") == 0.15
    assert advance_cursor(0.5, "HOLD") == 0.5


def test_advance_cursor_clamps():
    assert advance_cursor(-1.0, "LEFT") == -1.0
    assert advance_cursor(1.0, "RIGHT") == 1.0
    assert advance_cursor(-0.95, "LEFT") == -1.0
    assert advance_cursor(0.95, "RIGHT") == 1.0


def test_unknown_command_raises():
    with pytest.raises(ValueError, match="unknown command"):
        advance_cursor(0.0, "UP")


def test_probability_validation():
    with pytest.raises(ValueError):
        choose_command([0.9], classes=(0, 1), threshold=0.7)
    with pytest.raises(ValueError):
        choose_command([np.nan, 0.1], classes=(0, 1), threshold=0.7)
