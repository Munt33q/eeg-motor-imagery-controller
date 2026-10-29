"""Replay timing tests: no early decisions, no duplicate moves, pause/reset."""

from __future__ import annotations

import pytest

from eeg_controller.replay import ReplayError, ReplayState


def _events():
    return [
        {"onset_s": 1.0, "decision_s": 4.0, "command": "LEFT", "label": 0},
        {"onset_s": 5.5, "decision_s": 8.5, "command": "RIGHT", "label": 1},
        {"onset_s": 10.0, "decision_s": 13.0, "command": "HOLD", "label": 1},
    ]


def _state(**kw):
    defaults = dict(duration_s=15.0, events=_events(), speed=1.0)
    defaults.update(kw)
    return ReplayState(**defaults)


def test_no_decision_before_window_completes():
    st = _state()
    st.tick(3.9)  # just before first decision_s (4.0)
    assert st.due_events() == []
    assert st.process_due() == []


def test_decision_available_exactly_at_window_end():
    st = _state()
    st.tick(4.0)
    due = st.process_due()
    assert len(due) == 1
    assert due[0]["command"] == "LEFT"


def test_no_duplicate_moves():
    st = _state()
    st.tick(4.0)
    first = st.process_due()
    assert len(first) == 1
    st.tick(1.0)  # time moves on; same window must not re-fire
    assert st.process_due() == []
    assert st.command_for_event(0) == "LEFT"


def test_all_events_fire_in_order():
    st = _state()
    st.tick(13.5)
    fired = st.process_due()
    assert [e["command"] for e in fired] == ["LEFT", "RIGHT", "HOLD"]
    assert st.process_due() == []


def test_pause_freezes_time():
    st = _state()
    st.tick(2.0)
    st.pause()
    t = st.time_s
    st.tick(5.0)  # ticks while paused do not advance replay time (speed=0)
    assert st.time_s == t
    assert st.speed == 0.0
    st.resume(2.0)
    st.tick(2.0)  # 2 s wall-clock at 2x -> 4 s replay
    assert st.time_s == t + 4.0


def test_reset_clears_processed_events_and_clock():
    st = _state()
    st.tick(4.0)
    st.process_due()
    st.reset()
    assert st.processed_event_ids == set()
    assert st.issued_commands == {}
    assert st.time_s == 0.0
    # After reset, the same decision can be issued again exactly once.
    st.tick(4.0)
    fired = st.process_due()
    assert len(fired) == 1


def test_speed_multiplies_wall_clock():
    st = _state(speed=4.0)
    st.tick(1.0)
    assert st.time_s == 4.0


def test_time_never_exceeds_duration():
    st = _state()
    st.tick(100.0)
    assert st.time_s == 15.0
    assert st.is_finished


def test_events_must_be_sorted():
    events = [
        {"onset_s": 5.0, "decision_s": 8.0, "command": "LEFT"},
        {"onset_s": 1.0, "decision_s": 4.0, "command": "RIGHT"},
    ]
    with pytest.raises(ReplayError):
        ReplayState(duration_s=10.0, events=events)


def test_decision_cannot_precede_onset():
    events = [{"onset_s": 5.0, "decision_s": 4.0, "command": "LEFT"}]
    with pytest.raises(ReplayError):
        ReplayState(duration_s=10.0, events=events)
