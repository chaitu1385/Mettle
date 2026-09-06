"""The policy interface: state in, action + full distribution out.

The interface is `select(state) -> (action, action_probs, policy_id)`. Two
implementations ship here (a hand-written rule policy and an epsilon-greedy
wrapper); a learned policy later only has to satisfy the same signature.

Why the full distribution: off-policy evaluation needs pi_b(a | s) for every
action, and it cannot be recovered from a logged (state, action) pair after the
fact. A deterministic rule therefore reports [0, 0, 1, 0, 0] -- its *base*
distribution -- and the epsilon wrapper reports the mixed distribution that
actually generated the sample. Both are logged.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass

from .actions import ACTIONS, Action, N_ACTIONS, one_hot, require_action, validate_probs
from .schemas import TurnState


@dataclass(frozen=True)
class Decision:
    """Everything worth logging about one action selection.

    `action_probs` is the behaviour distribution -- the one the sample was drawn
    from, and the one an off-policy estimator divides by. `base_probs` is what
    the wrapped policy wanted before exploration mixed it.
    """

    action: Action
    action_probs: list[float]
    policy_id: str
    base_probs: list[float]
    explored: bool
    eps: float

    def as_json_payload(self) -> dict:
        return {
            "order": list(ACTIONS),
            "probs": self.action_probs,
            "base_probs": self.base_probs,
            "explored": self.explored,
        }


class Policy(ABC):
    """Swappable policy. Subclasses implement `decide`; `select` is the contract."""

    @property
    @abstractmethod
    def policy_id(self) -> str:
        """Stable identifier logged on every row, e.g. 'rule/v1+eps0.2'."""

    @abstractmethod
    def decide(self, state: TurnState) -> Decision:
        ...

    def select(self, state: TurnState) -> tuple[Action, list[float], str]:
        """The interface every policy must satisfy."""
        decision = self.decide(state)
        return decision.action, decision.action_probs, decision.policy_id


class RulePolicy(Policy):
    """Hand-written coaching heuristics. Deterministic, so probs are one-hot.

    Rules are checked in a fixed priority order -- several can fire at once and
    the order is the actual policy, so it is written down rather than implied:

    1. closing phase        -> summarize  (a session that never lands is wasted)
    2. explicit request     -> advise     (honour what was asked for)
    3. emotional            -> reflect    (before challenge: do not push someone
                                           who is still in the feeling)
    4. repeated theme       -> challenge  (a loop is a request for friction)
    5. uncertain or vague   -> ask
    6. otherwise            -> ask        (cheapest way to gather state)
    """

    VERSION = "v1"

    @property
    def policy_id(self) -> str:
        return f"rule/{self.VERSION}"

    def rule_action(self, state: TurnState) -> Action:
        if state.session_phase == "closing":
            return "summarize"
        if state.user_intent == "seek_advice":
            return "advise"
        if state.is_emotional():
            return "reflect"
        if state.repeated_theme_flag:
            return "challenge"
        if state.user_intent == "uncertain" or state.specificity_level == "vague":
            return "ask"
        return "ask"

    def decide(self, state: TurnState) -> Decision:
        action = require_action(self.rule_action(state))
        probs = one_hot(action)
        return Decision(
            action=action,
            action_probs=probs,
            policy_id=self.policy_id,
            base_probs=list(probs),
            explored=False,
            eps=0.0,
        )


class EpsilonWrapper(Policy):
    """Wraps any policy: with probability eps, pick uniformly among the *other* actions.

    The mixed distribution is exact rather than approximated, so it is usable as
    pi_b in an importance-weighted estimator:

        p(a) = (1 - eps) * base(a) + (eps / (N - 1)) * (1 - base(a))

    which is the marginal of "sample a ~ base, then with probability eps replace
    it with a uniform draw from the four actions that are not a". For a
    deterministic base and eps=0.2 that is 0.8 on the rule's choice and 0.05 on
    each of the rest.
    """

    def __init__(self, inner: Policy, eps: float = 0.2, rng: random.Random | None = None):
        if not 0.0 <= eps <= 1.0:
            raise ValueError(f"eps must be in [0, 1], got {eps}")
        self.inner = inner
        self.eps = eps
        self.rng = rng or random.Random()

    @property
    def policy_id(self) -> str:
        return f"{self.inner.policy_id}+eps{self.eps:g}"

    def mixed_probs(self, base_probs: list[float]) -> list[float]:
        spread = self.eps / (N_ACTIONS - 1)
        return validate_probs(
            [(1.0 - self.eps) * p + spread * (1.0 - p) for p in base_probs]
        )

    def decide(self, state: TurnState) -> Decision:
        inner = self.inner.decide(state)
        base_probs = validate_probs(inner.base_probs)
        probs = self.mixed_probs(base_probs)

        action = inner.action
        explored = False
        if self.eps > 0.0 and self.rng.random() < self.eps:
            alternatives = [a for a in ACTIONS if a != inner.action]
            action = self.rng.choice(alternatives)
            explored = True

        return Decision(
            action=require_action(action),
            action_probs=probs,
            policy_id=self.policy_id,
            base_probs=base_probs,
            explored=explored,
            eps=self.eps,
        )


def build_policy(eps: float = 0.2, seed: int | None = None) -> Policy:
    """Default policy stack: rules under epsilon-greedy exploration."""
    rng = random.Random(seed) if seed is not None else None
    inner = RulePolicy()
    if eps <= 0.0:
        return inner
    return EpsilonWrapper(inner, eps=eps, rng=rng)
