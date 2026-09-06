"""Runtime configuration.

Kept deliberately small and dependency-free: a hand-rolled .env reader instead of
python-dotenv, so the dependency list stays at langgraph / pydantic / scipy /
anthropic.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MODEL = "claude-opus-5"
DEFAULT_DB = "coach_rl.db"
DEFAULT_EPS = 0.2

#: Turns needed before judge-vs-human validation is worth running.
MIN_LABELS_FOR_VALIDATION = 40

#: Correlation below this means the judge is not measuring what the human sees.
CORRELATION_WARN_THRESHOLD = 0.3


def load_dotenv(path: str | Path = ".env") -> None:
    """Load KEY=VALUE lines into os.environ without overwriting what is set."""
    file = Path(path)
    if not file.exists():
        return
    for line in file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


@dataclass(frozen=True)
class Settings:
    model: str = DEFAULT_MODEL
    db_path: str = DEFAULT_DB
    eps: float = DEFAULT_EPS

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        return cls(
            model=os.environ.get("COACH_RL_MODEL", DEFAULT_MODEL),
            db_path=os.environ.get("COACH_RL_DB", DEFAULT_DB),
            eps=float(os.environ.get("COACH_RL_EPS", DEFAULT_EPS)),
        )
