"""The parsing and retry logic around the model call -- not the model call itself.

`structured()` is the only place where an unpredictable string becomes typed
data. Its contract is: validate, retry exactly once with the error fed back,
then fail loudly rather than let a bad row through. That is this codebase's
logic, so it is tested here with a fake transport. No network, no assertions
about what a model would say.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from coach_rl.llm import LLM, LLMError
from coach_rl.schemas import ClassifiedState

VALID = {
    "user_intent": "explore",
    "emotional_valence": 0.1,
    "specificity_level": "specific",
    "repeated_theme_flag": False,
    "session_phase": "opening",
}


def message(text: str, stop_reason: str = "end_turn"):
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        stop_reason=stop_reason,
        stop_details=None,
    )


class FakeClient:
    """Returns canned replies in order and records what it was sent."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls: list[dict] = []
        self.messages = SimpleNamespace(create=self._create)

    async def _create(self, **kwargs):
        self.calls.append(kwargs)
        return self.replies.pop(0)


def structured(client, schema=ClassifiedState):
    return asyncio.run(
        LLM("test-model", client=client).structured(system="s", user="u", schema=schema)
    )


def test_valid_json_is_parsed_on_the_first_call():
    client = FakeClient(message(json.dumps(VALID)))
    assert structured(client).user_intent == "explore"
    assert len(client.calls) == 1


def test_malformed_json_is_retried_once_with_the_error_fed_back():
    client = FakeClient(message("not json at all"), message(json.dumps(VALID)))
    assert structured(client).session_phase == "opening"

    assert len(client.calls) == 2, "exactly one retry"
    followup = client.calls[1]["messages"]
    assert followup[1]["content"] == "not json at all", "the bad reply is echoed back"
    assert "did not validate" in followup[2]["content"]


def test_a_schema_violation_is_retried_the_same_way():
    """Well-formed JSON that breaks the contract must not reach the dataset."""
    bad = json.dumps({**VALID, "emotional_valence": 42})
    client = FakeClient(message(bad), message(json.dumps(VALID)))
    assert structured(client).emotional_valence == 0.1
    assert len(client.calls) == 2
    assert "emotional_valence" in client.calls[1]["messages"][2]["content"]


def test_failing_twice_raises_rather_than_returning_something_wrong():
    client = FakeClient(message("junk"), message("still junk"))
    with pytest.raises(LLMError, match="ClassifiedState failed validation twice"):
        structured(client)
    assert len(client.calls) == 2, "one retry, not an unbounded loop"


def test_a_refusal_is_surfaced_not_retried():
    client = FakeClient(message("", stop_reason="refusal"))
    with pytest.raises(LLMError, match="refused"):
        structured(client)
    assert len(client.calls) == 1


def test_the_schema_is_sent_as_a_structured_output_constraint():
    client = FakeClient(message(json.dumps(VALID)))
    structured(client)
    fmt = client.calls[0]["output_config"]["format"]
    assert fmt["type"] == "json_schema"
    assert fmt["schema"]["additionalProperties"] is False
    assert "title" not in fmt["schema"]


def test_text_rejects_an_empty_reply():
    llm = LLM("test-model", client=FakeClient(message("   ")))
    with pytest.raises(LLMError, match="no text"):
        asyncio.run(llm.text(system="s", messages=[{"role": "user", "content": "u"}]))
