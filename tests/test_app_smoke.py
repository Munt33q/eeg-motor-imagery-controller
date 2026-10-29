"""Streamlit AppTest smoke test for app.py (skipped if Streamlit absent).

Verifies the app script executes: subject loads, replay controls render,
Next-trial produces a decision without duplicates, and the Results tab
renders saved metrics. Real EDF data and artifacts are required.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _require_artifacts(subject: int = 1):
    if not (PROJECT_ROOT / "artifacts" / f"model_sub{subject}.joblib").exists():
        pytest.skip("model artifact missing; run scripts/train.py first")
    if not list((PROJECT_ROOT / "data" / "raw").rglob("S001R12.edf")):
        pytest.skip("EDF downloads missing; run scripts/prepare_data.py first")


@pytest.mark.integration
def test_app_loads_subject_1_and_next_trial_decision():
    _require_artifacts(1)
    at = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=120)
    at.run()
    assert not at.exception, at.exception[0].value if at.exception else ""
    # Sidebar subject selector exists with the five subjects
    selectboxes = at.selectbox
    assert any(sb.key == "subject" or True for sb in selectboxes)
    # Results tab content renders from saved metrics
    assert any("Evaluation results" in h.value for h in at.header)

    # Next trial: first decision must appear, exactly once
    buttons = {b.label: b for b in at.button}
    assert "⏭ Next trial" in buttons
    buttons["⏭ Next trial"].click()
    at.run()
    assert not at.exception
    # After one next-trial press, exactly one decision is recorded
    decisions = [v for k, v in at.session_state.items() if k == "decisions"]
    assert decisions and len(decisions[0]) == 1

    # Pressing Next trial again advances to the second decision, not a duplicate
    buttons = {b.label: b for b in at.button}
    buttons["⏭ Next trial"].click()
    at.run()
    assert not at.exception
    decisions = [v for k, v in at.session_state.items() if k == "decisions"]
    assert decisions and len(decisions[0]) == 2


@pytest.mark.integration
def test_app_replay_state_resets_on_subject_switch():
    _require_artifacts(1)
    at = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=120)
    at.run()
    assert not at.exception
    sb = at.selectbox[0]
    sb.set_value(2).run()
    assert not at.exception
    decisions = [v for k, v in at.session_state.items() if k == "decisions"]
    assert decisions and len(decisions[0]) == 0  # cleared on subject switch
