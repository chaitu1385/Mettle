"""The judge: a separate LLM call that scores the previous turn.

It runs off the reply path for two reasons. The obvious one is latency -- the
coachee should not wait for scoring. The structural one is that forward movement
is only visible in the reaction, so a turn cannot be scored until the coachee's
*next* message exists. By then the coach is already answering that message, so
the judge necessarily runs concurrently with the next turn.

Consequence worth knowing: the last turn of a session is never scored, because
nothing follows it.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from dataclasses import dataclass

from .llm import LLM
from .prompts import JUDGE_SYSTEM, JUDGE_USER_TEMPLATE
from .schemas import JudgeScores
from .storage import attach_judge


@dataclass(frozen=True)
class PendingTurn:
    """A written turn waiting for the coachee's reaction before it can be scored."""

    session_id: str
    turn_index: int
    user_message: str
    action: str
    response_text: str


class JudgeRunner:
    """Fire-and-track judge tasks; write scores back to the row when they land."""

    def __init__(self, llm: LLM, conn: sqlite3.Connection):
        self.llm = llm
        self.conn = conn
        self._tasks: set[asyncio.Task] = set()
        self.failures: list[str] = []

    def schedule(self, turn: PendingTurn, next_user_message: str) -> asyncio.Task:
        task = asyncio.create_task(self._run(turn, next_user_message))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def _run(self, turn: PendingTurn, next_user_message: str) -> None:
        try:
            scores = await self.llm.structured(
                system=JUDGE_SYSTEM,
                user=JUDGE_USER_TEMPLATE.format(
                    user_message=turn.user_message,
                    action=turn.action,
                    response_text=turn.response_text,
                    next_user_message=next_user_message,
                ),
                schema=JudgeScores,
                effort="low",
            )
        except Exception as exc:  # a failed judge must never kill the session
            self.failures.append(f"turn {turn.turn_index}: {exc}")
            return

        attach_judge(
            self.conn,
            turn.session_id,
            turn.turn_index,
            json.dumps(scores.model_dump(exclude={"rationale"})),
            scores.rationale,
        )

    async def drain(self) -> None:
        """Wait for outstanding judge calls -- call before closing the session."""
        if self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)
