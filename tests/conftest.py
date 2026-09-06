"""Shared fixtures and builders."""

from __future__ import annotations

import json

import pytest

from coach_rl.actions import ACTIONS
from coach_rl.schemas import ClassifiedState, TurnState
from coach_rl.storage import TurnRecord, connect


@pytest.fixture()
def conn(tmp_path):
    """A connection to a throwaway database with the schema applied."""
    connection = connect(tmp_path / "test.db")
    yield connection
    connection.close()


def make_state(**overrides) -> TurnState:
    """A neutral state; override only the field the test is about."""
    fields = dict(
        user_intent="explore",
        emotional_valence=0.0,
        specificity_level="specific",
        repeated_theme_flag=False,
        session_phase="exploration",
    )
    fields.update(overrides)
    turn_index = fields.pop("turn_index", 0)
    return TurnState.from_classified(ClassifiedState(**fields), turn_index)


def make_record(
    turn_index: int,
    action: str = "ask",
    scores: dict | None = None,
    human_label: int | None = None,
    explored: bool = False,
    eps: float = 0.2,
    prompt_version: str = "v1",
    model_name: str = "claude-opus-5",
) -> TurnRecord:
    """A synthetic logged turn, for statistics that do not care how it was produced."""
    return TurnRecord(
        session_id="s1",
        turn_index=turn_index,
        state_json=json.dumps({"user_intent": "explore", "turn_index": turn_index}),
        action=action,
        action_probs_json=json.dumps(
            {
                "order": list(ACTIONS),
                "probs": [0.05, 0.05, 0.8, 0.05, 0.05],
                "base_probs": [0, 0, 1, 0, 0],
                "explored": explored,
            }
        ),
        policy_id="rule/v1+eps0.2",
        eps=eps,
        response_text="reply",
        user_message="message",
        model_name=model_name,
        prompt_version=prompt_version,
        judge_scores_json=json.dumps(scores) if scores else None,
        judge_rationale="rationale" if scores else None,
        human_label=human_label,
    )
