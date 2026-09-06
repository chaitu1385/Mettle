"""Policy interface tests.

The contract under test: `select(state) -> (action, action_probs, policy_id)`,
where `action_probs` is a full distribution over all five actions -- one-hot for
a deterministic rule, and the exact epsilon-mixed distribution once wrapped.
That vector is what makes the logged data usable for off-policy evaluation, so
it is tested harder than the rules themselves.
"""

from __future__ import annotations

import random
from collections import Counter

import pytest

from coach_rl.actions import ACTIONS, ACTION_INDEX, N_ACTIONS, one_hot, validate_probs
from coach_rl.policy import Decision, EpsilonWrapper, Policy, RulePolicy, build_policy
from coach_rl.schemas import TurnState

from .conftest import make_state


# --- the interface ---------------------------------------------------------


@pytest.mark.parametrize(
    "policy",
    [RulePolicy(), EpsilonWrapper(RulePolicy(), eps=0.2, rng=random.Random(1))],
    ids=["rule", "epsilon"],
)
def test_select_returns_action_probs_policy_id(policy: Policy):
    action, probs, policy_id = policy.select(make_state())
    assert action in ACTIONS
    assert isinstance(policy_id, str) and policy_id
    assert len(probs) == N_ACTIONS
    assert validate_probs(probs) == probs


def test_probs_cover_every_action_not_just_the_chosen_one():
    """The whole distribution is logged; a scalar for the chosen action is not enough."""
    _, probs, _ = EpsilonWrapper(RulePolicy(), eps=0.2).select(make_state())
    assert all(p > 0 for p in probs), "epsilon-mixed distribution must have full support"


def test_deterministic_rule_reports_one_hot_base_probs():
    state = make_state(repeated_theme_flag=True)
    decision = RulePolicy().decide(state)
    assert decision.action == "challenge"
    assert decision.base_probs == [0.0, 0.0, 1.0, 0.0, 0.0]
    assert decision.action_probs == decision.base_probs
    assert decision.explored is False
    assert decision.eps == 0.0


def test_wrapper_preserves_the_pre_mix_distribution():
    """base_probs stays one-hot after wrapping -- the rule's intent is not lost."""
    state = make_state(repeated_theme_flag=True)
    decision = EpsilonWrapper(RulePolicy(), eps=0.2, rng=random.Random(0)).decide(state)
    assert decision.base_probs == [0.0, 0.0, 1.0, 0.0, 0.0]
    assert decision.action_probs == pytest.approx([0.05, 0.05, 0.8, 0.05, 0.05])


def test_policy_id_records_the_epsilon():
    assert RulePolicy().policy_id == "rule/v1"
    assert EpsilonWrapper(RulePolicy(), eps=0.2).policy_id == "rule/v1+eps0.2"
    assert EpsilonWrapper(RulePolicy(), eps=0.05).policy_id == "rule/v1+eps0.05"


def test_decision_json_payload_is_self_describing():
    payload = EpsilonWrapper(RulePolicy(), eps=0.2).decide(make_state()).as_json_payload()
    assert payload["order"] == list(ACTIONS)
    assert set(payload) == {"order", "probs", "base_probs", "explored"}


# --- the rules -------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides,expected",
    [
        ({"user_intent": "uncertain"}, "ask"),
        ({"specificity_level": "vague"}, "ask"),
        ({"emotional_valence": -0.9}, "reflect"),
        ({"emotional_valence": 0.7}, "reflect"),
        ({"repeated_theme_flag": True}, "challenge"),
        ({"user_intent": "seek_advice"}, "advise"),
        ({"session_phase": "closing"}, "summarize"),
        ({}, "ask"),
    ],
)
def test_rule_policy_mapping(overrides, expected):
    assert RulePolicy().rule_action(make_state(**overrides)) == expected


def test_rule_priority_is_the_documented_order():
    """Several rules fire at once here; priority decides, and it is not accidental."""
    everything = make_state(
        user_intent="seek_advice",
        emotional_valence=-0.9,
        specificity_level="vague",
        repeated_theme_flag=True,
        session_phase="closing",
    )
    assert RulePolicy().rule_action(everything) == "summarize"

    without_closing = make_state(
        user_intent="seek_advice",
        emotional_valence=-0.9,
        repeated_theme_flag=True,
        session_phase="deepening",
    )
    assert RulePolicy().rule_action(without_closing) == "advise"

    # Emotion outranks a repeated theme: do not challenge someone mid-feeling.
    emotional_loop = make_state(emotional_valence=-0.8, repeated_theme_flag=True)
    assert RulePolicy().rule_action(emotional_loop) == "reflect"


def test_rule_policy_only_emits_known_actions():
    rng = random.Random(7)
    policy = RulePolicy()
    for _ in range(200):
        state = make_state(
            user_intent=rng.choice(
                ["uncertain", "explore", "vent", "seek_advice", "decide", "update"]
            ),
            emotional_valence=rng.uniform(-1, 1),
            specificity_level=rng.choice(["vague", "moderate", "specific"]),
            repeated_theme_flag=rng.random() < 0.5,
            session_phase=rng.choice(["opening", "exploration", "deepening", "closing"]),
        )
        assert policy.decide(state).action in ACTIONS


# --- the epsilon wrapper ---------------------------------------------------


def test_mixed_probs_formula_and_normalisation():
    wrapper = EpsilonWrapper(RulePolicy(), eps=0.4)
    mixed = wrapper.mixed_probs(one_hot("advise"))
    assert mixed[ACTION_INDEX["advise"]] == pytest.approx(0.6)
    for action in ACTIONS:
        if action != "advise":
            assert mixed[ACTION_INDEX[action]] == pytest.approx(0.4 / 4)
    assert sum(mixed) == pytest.approx(1.0)


def test_eps_zero_is_the_identity():
    state = make_state(session_phase="closing")
    wrapper = EpsilonWrapper(RulePolicy(), eps=0.0)
    decision = wrapper.decide(state)
    assert decision.action == "summarize"
    assert decision.action_probs == one_hot("summarize")
    assert decision.explored is False


def test_eps_one_always_explores_away_from_the_rule():
    state = make_state(session_phase="closing")
    wrapper = EpsilonWrapper(RulePolicy(), eps=1.0, rng=random.Random(3))
    for _ in range(50):
        decision = wrapper.decide(state)
        assert decision.explored is True
        assert decision.action != "summarize", "explore picks from the OTHER actions"


def test_exploration_fires_at_roughly_eps_and_is_uniform_over_alternatives():
    state = make_state(repeated_theme_flag=True)  # rule says challenge
    wrapper = EpsilonWrapper(RulePolicy(), eps=0.2, rng=random.Random(42))
    decisions = [wrapper.decide(state) for _ in range(4000)]

    explored = sum(d.explored for d in decisions)
    assert 0.17 < explored / len(decisions) < 0.23

    counts = Counter(d.action for d in decisions)
    empirical = counts["challenge"] / len(decisions)
    assert empirical == pytest.approx(0.8, abs=0.03)
    for action in ACTIONS:
        if action != "challenge":
            assert counts[action] / len(decisions) == pytest.approx(0.05, abs=0.02)


def test_logged_probs_match_the_sampling_distribution():
    """The logged vector must be the distribution the sample actually came from."""
    state = make_state()
    wrapper = EpsilonWrapper(RulePolicy(), eps=0.3, rng=random.Random(11))
    decisions = [wrapper.decide(state) for _ in range(6000)]
    logged = decisions[0].action_probs
    counts = Counter(d.action for d in decisions)
    for action in ACTIONS:
        assert counts[action] / len(decisions) == pytest.approx(
            logged[ACTION_INDEX[action]], abs=0.03
        )


def test_wrapper_rejects_an_invalid_eps():
    for bad in (-0.1, 1.5):
        with pytest.raises(ValueError):
            EpsilonWrapper(RulePolicy(), eps=bad)


def test_wrapper_is_reproducible_with_a_seed():
    state = make_state()
    wrapper_a = EpsilonWrapper(RulePolicy(), 0.5, random.Random(9))
    wrapper_b = EpsilonWrapper(RulePolicy(), 0.5, random.Random(9))
    assert [wrapper_a.decide(state).action for _ in range(30)] == [
        wrapper_b.decide(state).action for _ in range(30)
    ]


def test_build_policy_wraps_only_when_eps_is_positive():
    assert isinstance(build_policy(eps=0.0), RulePolicy)
    wrapped = build_policy(eps=0.2, seed=5)
    assert isinstance(wrapped, EpsilonWrapper) and wrapped.eps == 0.2


def test_wrapper_accepts_any_inner_policy():
    """A learned policy only has to satisfy the same interface."""

    class AlwaysAdvise(Policy):
        @property
        def policy_id(self) -> str:
            return "stub/v0"

        def decide(self, state: TurnState) -> Decision:
            probs = one_hot("advise")
            return Decision("advise", probs, self.policy_id, list(probs), False, 0.0)

    wrapper = EpsilonWrapper(AlwaysAdvise(), eps=0.2, rng=random.Random(0))
    action, probs, policy_id = wrapper.select(make_state())
    assert policy_id == "stub/v0+eps0.2"
    assert probs[ACTION_INDEX["advise"]] == pytest.approx(0.8)
    assert action in ACTIONS


def test_stochastic_inner_policy_mixes_correctly():
    """The mixing formula is exact for a non-deterministic base policy too."""

    class Coin(Policy):
        @property
        def policy_id(self) -> str:
            return "coin/v0"

        def decide(self, state: TurnState) -> Decision:
            probs = [0.5, 0.5, 0.0, 0.0, 0.0]
            return Decision("ask", probs, self.policy_id, probs, False, 0.0)

    wrapper = EpsilonWrapper(Coin(), eps=0.2)
    mixed = wrapper.mixed_probs([0.5, 0.5, 0.0, 0.0, 0.0])
    assert mixed[0] == pytest.approx(0.5 * 0.8 + 0.05 * 0.5)
    assert mixed[2] == pytest.approx(0.05)
    assert sum(mixed) == pytest.approx(1.0)


# --- the action space ------------------------------------------------------


def test_action_space_is_the_five_fixed_actions():
    assert ACTIONS == ("ask", "reflect", "challenge", "advise", "summarize")


def test_validate_probs_rejects_partial_or_unnormalised_vectors():
    with pytest.raises(ValueError):
        validate_probs([1.0])
    with pytest.raises(ValueError):
        validate_probs([0.5, 0.5, 0.5, 0.0, 0.0])
    with pytest.raises(ValueError):
        validate_probs([-0.2, 0.4, 0.4, 0.4, 0.0])
