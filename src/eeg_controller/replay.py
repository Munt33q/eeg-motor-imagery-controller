"""Deterministic replay timing and event progression (guide section 8).

The replay clock advances in fixed wall-clock ticks. A trial's prediction
becomes available only once its full analysis interval is recorded
(``decision_time >= trial onset + decision_delay``), and each trial issues
at most one command. "Processed" means *issuance handled* — the decision
was shown and the cursor move (if any) applied exactly once.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite


class ReplayError(RuntimeError):
    """Raised when replay invariants are violated."""


@dataclass
class ReplayState:
    """Full replay state; deterministic given (events, speed, time)."""

    duration_s: float
    events: list[dict]  # sorted by onset_s; keys: onset_s, decision_s, command, ...
    speed: float = 1.0
    time_s: float = 0.0
    processed_event_ids: set[int] = field(default_factory=set)
    issued_commands: dict[int, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isfinite(self.speed) or self.speed <= 0:
            raise ReplayError("speed must be positive")
        if not isfinite(self.duration_s) or self.duration_s < 0:
            raise ReplayError("duration must be non-negative")
        onsets = [e["onset_s"] for e in self.events]
        if onsets != sorted(onsets):
            raise ReplayError("events must be sorted by onset_s")
        for e in self.events:
            if not all(isfinite(e[k]) for k in ("onset_s", "decision_s")):
                raise ReplayError("event timestamps must be finite")
            if e["decision_s"] < e["onset_s"] + 3.0:
                raise ReplayError("decision_s must not precede onset_s")

    # -- clock ---------------------------------------------------------------
    def tick(self, dt_wall_s: float) -> None:
        """Advance replay time by dt_wall_s * speed (never beyond duration).

        ``dt_wall_s`` is wall-clock seconds; playback speed scales it. A
        paused state (speed == 0) does not advance regardless of input.
        """
        if not isfinite(dt_wall_s) or dt_wall_s < 0:
            raise ReplayError("dt must be non-negative")
        self.time_s = min(self.time_s + dt_wall_s * self.speed, self.duration_s)

    def set_speed(self, speed: float) -> None:
        if not isfinite(speed) or speed <= 0:
            raise ReplayError("speed must be positive")
        self.speed = speed

    def pause(self) -> None:
        self.speed = 0.0

    def resume(self, speed: float = 1.0) -> None:
        self.set_speed(speed)

    # -- events --------------------------------------------------------------
    def visible_events(self) -> list[dict]:
        """Events whose onset has been reached at current replay time."""
        return [
            dict(e) if e["decision_s"] <= self.time_s else {"onset_s": e["onset_s"], "decision_s": e["decision_s"]}
            for e in self.events if e["onset_s"] <= self.time_s
        ]

    def due_events(self) -> list[dict]:
        """Events whose analysis window is complete but not yet processed."""
        return [
            e
            for i, e in enumerate(self.events)
            if e["decision_s"] <= self.time_s and i not in self.processed_event_ids
        ]

    def process_due(self) -> list[dict]:
        """Mark all due events processed; return them with their commands.

        Each event is returned exactly once per replay lifetime unless
        :meth:`reset` clears it. Commands are immutable once issued: a
        repeated call cannot re-issue a move (the returned list is empty).

        Events without a ``command`` key are supported (e.g. demo replay,
        where the command is produced by live inference at decision time);
        no entry is recorded in ``issued_commands`` for them.
        """
        due = self.due_events()
        out = []
        for i, e in enumerate(self.events):
            if e["decision_s"] <= self.time_s and i not in self.processed_event_ids:
                self.processed_event_ids.add(i)
                if "command" in e:
                    self.issued_commands[i] = e["command"]
                out.append(e)
        if len(out) != len(due):
            raise ReplayError("due-event bookkeeping inconsistent")
        return out

    def command_for_event(self, index: int) -> str:
        if index not in self.issued_commands:
            raise ReplayError(f"event {index} has no issued command")
        return self.issued_commands[index]

    # -- lifecycle -------------------------------------------------------------
    def reset(self) -> None:
        """Clear processed events and issued commands; rewind clock."""
        self.processed_event_ids.clear()
        self.issued_commands.clear()
        self.time_s = 0.0

    # -- queries -----------------------------------------------------------------
    @property
    def is_finished(self) -> bool:
        return self.time_s >= self.duration_s

    def future_labels_visible(self) -> bool:
        """Sanity check: True if any un-reached event's label was exposed."""
        # Events with onset_s > current time must not be exposed by the UI.
        # The state itself never reveals them; visible_events() filters by
        # onset. This helper exists for tests and UI assertions.
        return any(
            e["decision_s"] > self.time_s and any(k in e for k in ("label", "true_label"))
            for e in self.visible_events()
        )
