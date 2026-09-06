"""Schema tests: the SQLite table and the Pydantic state objects.

These guard the properties that make the rows trainable -- the action space is
closed, human labels are constrained, one row per (session, turn), and the
judge columns start empty and get filled in later.
"""

from __future__ import annotations

import json
import sqlite3

import pytest
from pydantic import ValidationError

from coach_rl.actions import ACTIONS
from coach_rl.policy import RulePolicy
from coach_rl.prompts import PROMPT_VERSION
from coach_rl.schemas import ClassifiedState, JudgeScores, TurnState
from coach_rl.storage import (
    TURN_COLUMNS,
    TurnRecord,
    attach_judge,
    end_session,
    fetch_turns,
    insert_turn,
    list_sessions,
    set_human_label,
    start_session,
)

from .conftest import make_state


def make_policy_record(session_id: str, turn_index: int = 0, **overrides) -> TurnRecord:
    """A row built the way the session builds one: real state, real policy output."""
    state = make_state(turn_index=turn_index)
    decision = RulePolicy().decide(state)
    fields = dict(
        session_id=session_id,
        turn_index=turn_index,
        state_json=json.dumps(state.model_dump()),
        action=decision.action,
        action_probs_json=json.dumps(decision.as_json_payload()),
        policy_id=decision.policy_id,
        eps=decision.eps,
        response_text="What are you actually deciding here?",
        user_message="I keep going back and forth on the reorg.",
        model_name="claude-opus-5",
        prompt_version=PROMPT_VERSION,
    )
    fields.update(overrides)
    return TurnRecord(**fields)


# --- the turns table -------------------------------------------------------


def test_schema_has_exactly_the_required_columns(conn):
    columns = [row[1] for row in conn.execute("PRAGMA table_info(turns)")]
    assert columns == list(TURN_COLUMNS)


def test_sessions_table_columns(conn):
    columns = [row[1] for row in conn.execute("PRAGMA table_info(sessions)")]
    assert columns == ["session_id", "started_at", "ended_at", "session_notes"]


def test_roundtrip_preserves_every_field(conn):
    session_id = start_session(conn, notes="first session")
    record = make_policy_record(session_id)
    insert_turn(conn, record)

    (loaded,) = fetch_turns(conn, session_id)
    for column in TURN_COLUMNS:
        assert getattr(loaded, column) == getattr(record, column)
    assert loaded.state["user_intent"] == "explore"
    assert loaded.action_probs["order"] == list(ACTIONS)


def test_judge_columns_start_null_and_are_filled_in_later(conn):
    session_id = start_session(conn)
    insert_turn(conn, make_policy_record(session_id))

    (before,) = fetch_turns(conn, session_id)
    assert before.judge_scores_json is None
    assert before.judge_scores is None
    assert before.judge_rationale is None

    attach_judge(
        conn,
        session_id,
        0,
        json.dumps({"insight": 4, "specificity": 3, "forward_movement": 2}),
        "she named the tradeoff she had been avoiding",
    )
    (after,) = fetch_turns(conn, session_id)
    assert after.judge_scores == {"insight": 4, "specificity": 3, "forward_movement": 2}
    assert "tradeoff" in after.judge_rationale


def test_human_label_is_nullable_and_constrained(conn):
    session_id = start_session(conn)
    insert_turn(conn, make_policy_record(session_id))
    assert fetch_turns(conn, session_id)[0].human_label is None

    for label in (-1, 0, 1, None):
        set_human_label(conn, session_id, 0, label)
        assert fetch_turns(conn, session_id)[0].human_label == label

    with pytest.raises(ValueError):
        set_human_label(conn, session_id, 0, 2)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("UPDATE turns SET human_label = 7 WHERE session_id = ?", (session_id,))


def test_action_column_rejects_actions_outside_the_fixed_set(conn):
    session_id = start_session(conn)
    with pytest.raises(sqlite3.IntegrityError):
        insert_turn(conn, make_policy_record(session_id, action="empathize"))


@pytest.mark.parametrize("action", ACTIONS)
def test_every_declared_action_is_insertable(conn, action):
    session_id = start_session(conn)
    insert_turn(conn, make_policy_record(session_id, action=action))
    assert fetch_turns(conn, session_id)[0].action == action


def test_turn_index_is_unique_per_session(conn):
    session_id = start_session(conn)
    insert_turn(conn, make_policy_record(session_id, turn_index=0))
    with pytest.raises(sqlite3.IntegrityError):
        insert_turn(conn, make_policy_record(session_id, turn_index=0))
    # Same index in a different session is fine.
    other = start_session(conn)
    insert_turn(conn, make_policy_record(other, turn_index=0))


def test_turn_requires_an_existing_session(conn):
    with pytest.raises(sqlite3.IntegrityError):
        insert_turn(conn, make_policy_record("does-not-exist"))


def test_prompt_version_and_model_are_recorded_on_every_row(conn):
    session_id = start_session(conn)
    insert_turn(conn, make_policy_record(session_id))
    (row,) = fetch_turns(conn, session_id)
    assert row.prompt_version == PROMPT_VERSION
    assert row.model_name


@pytest.mark.parametrize("column", ["state_json", "action_probs_json", "policy_id", "prompt_version", "model_name"])
def test_required_columns_reject_null(conn, column):
    session_id = start_session(conn)
    with pytest.raises(sqlite3.IntegrityError):
        insert_turn(conn, make_policy_record(session_id, **{column: None}))


def test_session_lifecycle_and_listing(conn):
    session_id = start_session(conn)
    insert_turn(conn, make_policy_record(session_id))
    (row,) = list_sessions(conn)
    assert row["turn_count"] == 1 and row["ended_at"] is None

    end_session(conn, session_id, notes="wrapped up")
    (row,) = list_sessions(conn)
    assert row["ended_at"] is not None
    assert row["session_notes"] == "wrapped up"


# --- the state and judge objects ------------------------------------------


def test_turn_state_rejects_unknown_fields_and_out_of_range_values():
    with pytest.raises(ValidationError):
        make_state(user_intent="ruminate")
    with pytest.raises(ValidationError):
        make_state(emotional_valence=-3.0)
    with pytest.raises(ValidationError):
        ClassifiedState(
            user_intent="explore",
            emotional_valence=0.0,
            specificity_level="vague",
            repeated_theme_flag=False,
            session_phase="opening",
            confidence=0.9,  # hallucinated extra field
        )
    with pytest.raises(ValidationError):
        TurnState.from_classified(
            ClassifiedState(
                user_intent="explore",
                emotional_valence=0.0,
                specificity_level="vague",
                repeated_theme_flag=False,
                session_phase="opening",
            ),
            turn_index=-1,
        )


@pytest.mark.parametrize(
    "valence,expected", [(0.0, False), (0.49, False), (-0.5, True), (0.8, True)]
)
def test_is_emotional_uses_magnitude_not_sign(valence, expected):
    assert make_state(emotional_valence=valence).is_emotional() is expected


def test_judge_scores_bounds_and_total():
    scores = JudgeScores(insight=4, specificity=2, forward_movement=5, rationale="ok")
    assert scores.total == 11
    with pytest.raises(ValidationError):
        JudgeScores(insight=6, specificity=3, forward_movement=3, rationale="ok")
    with pytest.raises(ValidationError):
        JudgeScores(insight=0, specificity=3, forward_movement=3, rationale="ok")
    with pytest.raises(ValidationError):
        JudgeScores(insight=3, specificity=3, forward_movement=3, rationale="")


def test_state_json_schema_is_closed_for_structured_output():
    schema = ClassifiedState.model_json_schema()
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"])
    assert "turn_index" not in schema["properties"], "turn_index must come from the runtime"
