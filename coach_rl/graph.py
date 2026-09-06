"""The LangGraph decision loop: classify_state -> select_action -> generate_response.

The judge is deliberately not a node in this graph. It scores the *previous*
turn and needs the user's next message to see forward movement, so it runs
asynchronously alongside the next turn rather than in the reply path (see
`judge.py`).
"""

from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from .actions import Action
from .llm import LLMClient
from .policy import Decision, Policy
from .prompts import (
    CLASSIFY_SYSTEM,
    CLASSIFY_USER_TEMPLATE,
    RESPONSE_SYSTEM,
    RESPONSE_USER_TEMPLATE,
    action_instruction,
)
from .schemas import ClassifiedState, TurnState

HISTORY_TURNS_IN_PROMPT = 12


class GraphState(TypedDict, total=False):
    """What flows through the graph for a single turn."""

    session_id: str
    turn_index: int
    user_message: str
    history: list[dict[str, str]]
    state: TurnState
    decision: Decision
    action: Action
    response_text: str


def format_history(history: list[dict[str, str]], limit: int = HISTORY_TURNS_IN_PROMPT) -> str:
    if not history:
        return "(this is the first message of the session)"
    recent = history[-limit:]
    speaker = {"user": "Coachee", "assistant": "Coach"}
    return "\n".join(f"{speaker.get(m['role'], m['role'])}: {m['content']}" for m in recent)


def state_summary(state: TurnState) -> str:
    """Compact, human-readable state for the response prompt."""
    return (
        f"intent={state.user_intent}, valence={state.emotional_valence:+.2f}, "
        f"specificity={state.specificity_level}, "
        f"repeated_theme={'yes' if state.repeated_theme_flag else 'no'}, "
        f"phase={state.session_phase}, turn={state.turn_index}"
    )


def build_graph(llm: LLMClient, policy: Policy) -> Any:
    """Compile the three-node graph against a given LLM and policy."""

    async def classify_state(state: GraphState) -> dict[str, Any]:
        """LLM call returning a typed, Pydantic-validated state object."""
        classified = await llm.structured(
            system=CLASSIFY_SYSTEM,
            user=CLASSIFY_USER_TEMPLATE.format(
                history=format_history(state.get("history", [])),
                turn_index=state["turn_index"],
                user_message=state["user_message"],
            ),
            schema=ClassifiedState,
            effort="low",
        )
        # turn_index comes from the runtime, never from the model.
        return {"state": TurnState.from_classified(classified, state["turn_index"])}

    def select_action(state: GraphState) -> dict[str, Any]:
        """The policy interface. No LLM involved: the model never picks the action."""
        decision = policy.decide(state["state"])
        return {"decision": decision, "action": decision.action}

    async def generate_response(state: GraphState) -> dict[str, Any]:
        """Write the coach's reply, with the chosen action as a hard instruction."""
        turn_state: TurnState = state["state"]
        user_block = RESPONSE_USER_TEMPLATE.format(
            history=format_history(state.get("history", [])),
            user_message=state["user_message"],
            state_summary=state_summary(turn_state),
            action_instruction=action_instruction(state["action"]),
        )
        reply = await llm.text(
            system=RESPONSE_SYSTEM,
            messages=[{"role": "user", "content": user_block}],
            effort="medium",
        )
        return {"response_text": reply}

    graph = StateGraph(GraphState)
    graph.add_node("classify_state", classify_state)
    graph.add_node("select_action", select_action)
    graph.add_node("generate_response", generate_response)
    graph.add_edge(START, "classify_state")
    graph.add_edge("classify_state", "select_action")
    graph.add_edge("select_action", "generate_response")
    graph.add_edge("generate_response", END)
    return graph.compile()
