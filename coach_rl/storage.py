"""SQLite persistence: one row per coaching turn, plus a sessions table.

This file is the actual deliverable of the milestone. Everything else exists to
fill these rows correctly. Two constraints are enforced in SQL rather than in
Python, because the point of the exercise is that bad rows never reach the
dataset:

- `action` must be one of the five fixed actions.
- `human_label` is nullable but must be -1, 0, or 1 when present.

`judge_scores_json` / `judge_rationale` start NULL and are filled in later: the
judge cannot score a turn until the user's next message exists.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

from .actions import ACTIONS

_ACTION_LIST_SQL = ", ".join(f"'{a}'" for a in ACTIONS)

SCHEMA = f"""
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS sessions (
    session_id    TEXT PRIMARY KEY,
    started_at    TEXT NOT NULL,
    ended_at      TEXT,
    session_notes TEXT
);

CREATE TABLE IF NOT EXISTS turns (
    session_id        TEXT    NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    turn_index        INTEGER NOT NULL,
    timestamp         TEXT    NOT NULL,
    state_json        TEXT    NOT NULL,
    action            TEXT    NOT NULL CHECK (action IN ({_ACTION_LIST_SQL})),
    action_probs_json TEXT    NOT NULL,
    policy_id         TEXT    NOT NULL,
    eps               REAL    NOT NULL,
    response_text     TEXT    NOT NULL,
    user_message      TEXT    NOT NULL,
    judge_scores_json TEXT,
    judge_rationale   TEXT,
    human_label       INTEGER CHECK (human_label IN (-1, 0, 1)),
    model_name        TEXT    NOT NULL,
    prompt_version    TEXT    NOT NULL,
    PRIMARY KEY (session_id, turn_index)
);

CREATE INDEX IF NOT EXISTS idx_turns_action ON turns(action);
CREATE INDEX IF NOT EXISTS idx_turns_label  ON turns(human_label);
"""


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(db_path: str | Path) -> sqlite3.Connection:
    """Open a connection with the schema applied and rows as dict-likes."""
    path = Path(db_path)
    if path.parent and str(path.parent) not in ("", "."):
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


@contextmanager
def open_db(db_path: str | Path) -> Iterator[sqlite3.Connection]:
    conn = connect(db_path)
    try:
        yield conn
    finally:
        conn.close()


@dataclass
class TurnRecord:
    """One row of training data."""

    session_id: str
    turn_index: int
    state_json: str
    action: str
    action_probs_json: str
    policy_id: str
    eps: float
    response_text: str
    user_message: str
    model_name: str
    prompt_version: str
    timestamp: str = field(default_factory=utcnow)
    judge_scores_json: str | None = None
    judge_rationale: str | None = None
    human_label: int | None = None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "TurnRecord":
        return cls(**{k: row[k] for k in row.keys()})

    @property
    def state(self) -> dict[str, Any]:
        return json.loads(self.state_json)

    @property
    def action_probs(self) -> dict[str, Any]:
        return json.loads(self.action_probs_json)

    @property
    def judge_scores(self) -> dict[str, Any] | None:
        return json.loads(self.judge_scores_json) if self.judge_scores_json else None

    @property
    def explored(self) -> bool:
        return bool(self.action_probs.get("explored", False))


TURN_COLUMNS: tuple[str, ...] = (
    "session_id",
    "turn_index",
    "timestamp",
    "state_json",
    "action",
    "action_probs_json",
    "policy_id",
    "eps",
    "response_text",
    "user_message",
    "judge_scores_json",
    "judge_rationale",
    "human_label",
    "model_name",
    "prompt_version",
)


def start_session(
    conn: sqlite3.Connection,
    session_id: str | None = None,
    notes: str | None = None,
) -> str:
    session_id = session_id or uuid.uuid4().hex[:12]
    conn.execute(
        "INSERT INTO sessions(session_id, started_at, ended_at, session_notes) "
        "VALUES (?, ?, NULL, ?)",
        (session_id, utcnow(), notes),
    )
    conn.commit()
    return session_id


def end_session(
    conn: sqlite3.Connection, session_id: str, notes: str | None = None
) -> None:
    if notes is None:
        conn.execute(
            "UPDATE sessions SET ended_at = ? WHERE session_id = ?",
            (utcnow(), session_id),
        )
    else:
        conn.execute(
            "UPDATE sessions SET ended_at = ?, session_notes = ? WHERE session_id = ?",
            (utcnow(), notes, session_id),
        )
    conn.commit()


def insert_turn(conn: sqlite3.Connection, record: TurnRecord) -> None:
    placeholders = ", ".join("?" for _ in TURN_COLUMNS)
    conn.execute(
        f"INSERT INTO turns({', '.join(TURN_COLUMNS)}) VALUES ({placeholders})",
        tuple(getattr(record, col) for col in TURN_COLUMNS),
    )
    conn.commit()


def attach_judge(
    conn: sqlite3.Connection,
    session_id: str,
    turn_index: int,
    scores_json: str,
    rationale: str,
) -> None:
    """Fill in the reward for a turn once the user's reaction has been seen."""
    conn.execute(
        "UPDATE turns SET judge_scores_json = ?, judge_rationale = ? "
        "WHERE session_id = ? AND turn_index = ?",
        (scores_json, rationale, session_id, turn_index),
    )
    conn.commit()


def set_human_label(
    conn: sqlite3.Connection, session_id: str, turn_index: int, label: int | None
) -> None:
    if label not in (-1, 0, 1, None):
        raise ValueError(f"human_label must be -1, 0, 1 or None, got {label!r}")
    conn.execute(
        "UPDATE turns SET human_label = ? WHERE session_id = ? AND turn_index = ?",
        (label, session_id, turn_index),
    )
    conn.commit()


def fetch_turns(
    conn: sqlite3.Connection, session_id: str | None = None
) -> list[TurnRecord]:
    sql = f"SELECT {', '.join(TURN_COLUMNS)} FROM turns"
    params: Sequence[Any] = ()
    if session_id:
        sql += " WHERE session_id = ?"
        params = (session_id,)
    sql += " ORDER BY session_id, turn_index"
    return [TurnRecord.from_row(row) for row in conn.execute(sql, params)]


def list_sessions(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return list(
        conn.execute(
            "SELECT s.session_id, s.started_at, s.ended_at, s.session_notes, "
            "       COUNT(t.turn_index) AS turn_count "
            "FROM sessions s LEFT JOIN turns t ON t.session_id = s.session_id "
            "GROUP BY s.session_id ORDER BY s.started_at"
        )
    )
