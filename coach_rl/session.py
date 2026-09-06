"""Session orchestration: run the graph per turn, log the row, schedule the judge.

The ordering here is the whole trick. When the coachee sends message N:

  1. the judge for turn N-1 is scheduled (message N is the reaction it needs),
  2. the graph runs for turn N and the reply is printed,
  3. the row for turn N is written immediately -- before the judge for it exists.

So a row is complete except for `judge_scores_json` at write time, and the judge
UPDATEs it a few seconds later. A crash therefore loses at most one turn's
reward, never the turn itself.
"""

from __future__ import annotations

import json
import sqlite3

from .graph import build_graph
from .judge import JudgeRunner, PendingTurn
from .llm import LLM
from .policy import Policy
from .prompts import PROMPT_VERSION
from .storage import TurnRecord, insert_turn, start_session, end_session


class CoachSession:
    """One coaching session, writing one row per turn."""

    def __init__(
        self,
        llm: LLM,
        policy: Policy,
        conn: sqlite3.Connection,
        session_id: str,
    ):
        self.llm = llm
        self.policy = policy
        self.conn = conn
        self.session_id = session_id
        self.graph = build_graph(llm, policy)
        self.judge = JudgeRunner(llm, conn)
        self.history: list[dict[str, str]] = []
        self.turn_index = 0
        self.pending: PendingTurn | None = None

    @classmethod
    def start(
        cls,
        llm: LLM,
        policy: Policy,
        conn: sqlite3.Connection,
        notes: str | None = None,
    ) -> "CoachSession":
        """Open the session row, then bind a runner to it.

        The row is created here rather than in `__init__` so constructing the
        object has no side effects: the write is explicit and visible.
        """
        return cls(llm, policy, conn, start_session(conn, notes=notes))

    async def turn(self, user_message: str) -> tuple[str, TurnRecord]:
        """Handle one coachee message: judge the previous turn, then answer this one."""
        if self.pending is not None:
            self.judge.schedule(self.pending, next_user_message=user_message)
            self.pending = None

        result = await self.graph.ainvoke(
            {
                "session_id": self.session_id,
                "turn_index": self.turn_index,
                "user_message": user_message,
                "history": list(self.history),
            }
        )

        decision = result["decision"]
        state = result["state"]
        response_text = result["response_text"]

        record = TurnRecord(
            session_id=self.session_id,
            turn_index=self.turn_index,
            state_json=json.dumps(state.model_dump()),
            action=decision.action,
            action_probs_json=json.dumps(decision.as_json_payload()),
            policy_id=decision.policy_id,
            eps=decision.eps,
            response_text=response_text,
            user_message=user_message,
            model_name=self.llm.model,
            prompt_version=PROMPT_VERSION,
        )
        insert_turn(self.conn, record)

        self.history.append({"role": "user", "content": user_message})
        self.history.append({"role": "assistant", "content": response_text})
        self.pending = PendingTurn(
            session_id=self.session_id,
            turn_index=self.turn_index,
            user_message=user_message,
            action=decision.action,
            response_text=response_text,
        )
        self.turn_index += 1
        return response_text, record

    async def close(self, notes: str | None = None) -> None:
        """Drain outstanding judge calls and close the session row.

        The final turn keeps a NULL judge score on purpose: there is no reaction
        to score it against, and inventing one would poison the reward column.
        """
        await self.judge.drain()
        end_session(self.conn, self.session_id, notes)
