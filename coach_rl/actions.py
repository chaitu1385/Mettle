"""The fixed action space.

This tuple is the contract for the whole system: the policy chooses from it,
the database constrains to it, and the probability vectors logged on every turn
are indexed by it. Nothing else may introduce an action -- in particular the LLM
never picks an action, it only receives one as an instruction.

Changing this list invalidates every previously logged action_probs vector.
If it ever changes, treat old rows as a different dataset.
"""

from __future__ import annotations

from typing import Literal, Sequence

Action = Literal["ask", "reflect", "challenge", "advise", "summarize"]

ACTIONS: tuple[Action, ...] = ("ask", "reflect", "challenge", "advise", "summarize")

N_ACTIONS = len(ACTIONS)

ACTION_INDEX: dict[str, int] = {a: i for i, a in enumerate(ACTIONS)}

PROB_TOLERANCE = 1e-9


def require_action(value: object) -> Action:
    """Return `value` if it is a known action, else raise."""
    if value not in ACTION_INDEX:
        raise ValueError(f"unknown action {value!r}; must be one of {list(ACTIONS)}")
    return value  # type: ignore[return-value]


def one_hot(action: str) -> list[float]:
    """Deterministic distribution over ACTIONS, e.g. 'challenge' -> [0,0,1,0,0]."""
    probs = [0.0] * N_ACTIONS
    probs[ACTION_INDEX[require_action(action)]] = 1.0
    return probs


def validate_probs(probs: Sequence[float]) -> list[float]:
    """Check a full distribution over ACTIONS and return it as a list.

    Partial vectors are rejected on purpose: off-policy evaluation needs the
    probability the behaviour policy assigned to *every* action, and that cannot
    be reconstructed after the fact.
    """
    values = list(probs)
    if len(values) != N_ACTIONS:
        raise ValueError(f"expected {N_ACTIONS} probabilities, got {len(values)}")
    if any(p < -PROB_TOLERANCE or p > 1 + PROB_TOLERANCE for p in values):
        raise ValueError(f"probabilities out of range: {values}")
    total = sum(values)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"probabilities must sum to 1, got {total}")
    return values
