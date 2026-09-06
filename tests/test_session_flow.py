"""Wiring test for the turn loop, with a stub LLM.

This does not test the LLM calls -- it tests the ordering around them, which is
where the data-correctness bugs live: the row is written on the turn it belongs
to, the judge scores the *previous* turn using the next user message, and the
final turn of a session stays unjudged.
"""

from __future__ import annotations

import asyncio

from coach_rl.policy import RulePolicy
from coach_rl.prompts import PROMPT_VERSION
from coach_rl.schemas import ClassifiedState, JudgeScores
from coach_rl.session import CoachSession
from coach_rl.storage import fetch_turns


class StubLLM:
    """Stands in for `LLM`: same three-method surface, no network."""

    model = "stub-model"

    def __init__(self):
        self.judge_inputs: list[str] = []

    async def structured(self, *, system, user, schema, effort="low"):
        if schema is ClassifiedState:
            return ClassifiedState(
                user_intent="seek_advice",
                emotional_valence=0.0,
                specificity_level="specific",
                repeated_theme_flag=False,
                session_phase="exploration",
            )
        if schema is JudgeScores:
            self.judge_inputs.append(user)
            return JudgeScores(
                insight=4, specificity=3, forward_movement=2, rationale="stub"
            )
        raise AssertionError(f"unexpected schema {schema}")

    async def text(self, *, system, messages, max_tokens=1024, effort="medium"):
        return "Say the thing you are avoiding saying."


def test_turns_are_logged_and_the_judge_runs_one_turn_behind(conn):
    llm = StubLLM()
    session = CoachSession.start(llm=llm, policy=RulePolicy(), conn=conn)

    async def scenario():
        await session.turn("Should I take the reorg or fight it?")
        await session.turn("I hadn't thought of it that way. I'll talk to Priya.")
        await session.close()

    asyncio.run(scenario())

    turns = fetch_turns(conn, session.session_id)
    assert [t.turn_index for t in turns] == [0, 1]
    assert all(t.action == "advise" for t in turns)  # seek_advice -> advise
    assert all(t.prompt_version == PROMPT_VERSION for t in turns)
    assert all(t.model_name == "stub-model" for t in turns)
    assert all(t.action_probs["base_probs"] == [0, 0, 0, 1, 0] for t in turns)

    # Turn 0 is judged; the judge saw the user's *next* message as evidence.
    assert turns[0].judge_scores == {
        "insight": 4,
        "specificity": 3,
        "forward_movement": 2,
    }
    assert "I'll talk to Priya" in llm.judge_inputs[0]

    # The last turn has no reaction to be judged against, and stays NULL.
    assert turns[1].judge_scores_json is None


def test_a_failing_judge_does_not_lose_the_turn(conn):
    class BrokenJudge(StubLLM):
        async def structured(self, *, system, user, schema, effort="low"):
            if schema is JudgeScores:
                raise RuntimeError("judge exploded")
            return await super().structured(
                system=system, user=user, schema=schema, effort=effort
            )

    llm = BrokenJudge()
    session = CoachSession.start(llm=llm, policy=RulePolicy(), conn=conn)

    async def scenario():
        await session.turn("first")
        await session.turn("second")
        await session.close()

    asyncio.run(scenario())

    turns = fetch_turns(conn, session.session_id)
    assert len(turns) == 2
    assert turns[0].judge_scores_json is None
    assert session.judge.failures and "judge exploded" in session.judge.failures[0]
