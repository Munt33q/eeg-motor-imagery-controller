"""Recorded-replay EEG motor-imagery controller (PhysioNet EEGMMIDB, CSP+LDA).

This package implements a cue-timed, recorded-data BCI demonstration:
it replays recorded motor-imagery EEG, classifies left/right imagery with
CSP + LDA, and converts predictions into cursor commands. Labels are used
for training and evaluation only, never for command generation.
"""

__version__ = "0.1.0"

LEFT = 0
RIGHT = 1
LABEL_NAMES = {0: "left", 1: "right"}
COMMAND_LEFT = "LEFT"
COMMAND_RIGHT = "RIGHT"
COMMAND_HOLD = "HOLD"
