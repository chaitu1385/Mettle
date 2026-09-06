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

from .llm import LLMClient
from .prompts import JUDGE_SYSTEM, JUDGE_USER_TEMPLATE
from .schemas import JudgeScores
from .storage import attach_judge


async def judge_turn(
    llm: LLMClient,
    *,
    user_message: str,
    action: str,
    response_text: str,
    next_user_message: str,
) -> JudgeScores:
    """Score one completed turn, given the coachee's reaction to it."""
    return await llm.structured(
        system=JUDGE_SYSTEM,
        user=JUDGE_USER_TEMPLATE.format(
            user_message=user_message,
            action=action,
            response_text=response_text,
            next_user_message=next_user_message,
        ),
        schema=JudgeScores,
        effort="low",
    )


class JudgeRunner:
    """Fire-and-track judge tasks; write scores back to the row when they land."""

    def __init__(self, llm: LLMClient, conn: sqlite3.Connection, verbose: bool = False):
        self.llm = llm
        self.conn = conn
        self.verbose = verbose
        self._tasks: set[asyncio.Task] = set()
        self.failures: list[str] = []

    def schedule(
        self,
        *,
        session_id: str,
        turn_index: int,
        user_message: str,
        action: str,
        response_text: str,
        next_user_message: str,
    ) -> asyncio.Task:
        task = asyncio.create_task(
            self._run(
                session_id=session_id,
                turn_index=turn_index,
                user_message=user_message,
                action=action,
                response_text=response_text,
                next_user_message=next_user_message,
            )
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def _run(
        self,
        *,
        session_id: str,
        turn_index: int,
        user_message: str,
        action: str,
        response_text: str,
        next_user_message: str,
    ) -> None:
        try:
            scores = await judge_turn(
                self.llm,
                user_message=user_message,
                action=action,
                response_text=response_text,
                next_user_message=next_user_message,
            )
        except Exception as exc:  # a failed judge must never kill the session
            self.failures.append(f"turn {turn_index}: {exc}")
            return

        payload = scores.model_dump(exclude={"rationale"})
        attach_judge(
            self.conn,
            session_id,
            turn_index,
            json.dumps(payload),
            scores.rationale,
        )
        if self.verbose:
            print(
                f"[judge] turn {turn_index}: insight={scores.insight} "
                f"specificity={scores.specificity} forward={scores.forward_movement}"
            )

    async def drain(self) -> None:
        """Wait for outstanding judge calls -- call before closing the session."""
        if self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)
