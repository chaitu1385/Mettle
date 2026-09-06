"""Thin async wrapper over the Anthropic Messages API.

Two call shapes are needed by the graph:

- `structured()` -- JSON constrained to a Pydantic model's schema, validated on
  the way back, retried exactly once with the validation error fed back in.
  Structured outputs make a malformed reply rare; the retry is here because
  "rare" is not "never" and a turn that fails validation must not be silently
  dropped from the dataset.
- `text()` -- the coach's reply.

Deliberately not enabled: server-side refusal fallbacks. A silent switch to a
different model mid-session would put two models' behaviour under one
`model_name` value, which is exactly the kind of quiet contamination this
project is built to avoid. A refusal here should be visible, not routed around.
"""

from __future__ import annotations

import json
from typing import Any, Sequence, TypeVar

import anthropic
from pydantic import BaseModel, ValidationError

from .schemas import json_schema_for

T = TypeVar("T", bound=BaseModel)

MAX_TOKENS_STRUCTURED = 1024
MAX_TOKENS_RESPONSE = 1024


class LLMError(RuntimeError):
    """Raised when a call cannot produce a valid result after its one retry."""


def _text_of(message: Any) -> str:
    return "".join(
        block.text for block in message.content if getattr(block, "type", None) == "text"
    ).strip()


class LLM:
    """Async client bound to one model. The model name is logged on every row."""

    def __init__(self, model: str, client: anthropic.AsyncAnthropic | None = None):
        self.model = model
        self.client = client or anthropic.AsyncAnthropic()

    async def structured(
        self,
        *,
        system: str,
        user: str,
        schema: type[T],
        effort: str = "low",
    ) -> T:
        """Return a validated instance of `schema`, retrying once on a bad parse."""
        messages: list[dict[str, Any]] = [{"role": "user", "content": user}]
        last_error: Exception | None = None

        for attempt in range(2):
            message = await self.client.messages.create(
                model=self.model,
                max_tokens=MAX_TOKENS_STRUCTURED,
                system=system,
                messages=messages,
                output_config={
                    "effort": effort,
                    "format": {"type": "json_schema", "schema": json_schema_for(schema)},
                },
            )
            if message.stop_reason == "refusal":
                raise LLMError(f"model refused the request: {message.stop_details}")

            raw = _text_of(message)
            try:
                return schema.model_validate(json.loads(raw))
            except (json.JSONDecodeError, ValidationError) as exc:
                last_error = exc
                if attempt == 0:
                    # Feed the failure back rather than re-rolling blind.
                    messages = messages + [
                        {"role": "assistant", "content": raw or "(empty)"},
                        {
                            "role": "user",
                            "content": (
                                "That did not validate against the required schema:\n"
                                f"{exc}\n\nReturn the corrected object only."
                            ),
                        },
                    ]

        raise LLMError(f"structured output failed twice for {schema.__name__}: {last_error}")

    async def text(
        self,
        *,
        system: str,
        messages: Sequence[dict[str, Any]],
        max_tokens: int = MAX_TOKENS_RESPONSE,
        effort: str = "medium",
    ) -> str:
        message = await self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=list(messages),
            output_config={"effort": effort},
        )
        if message.stop_reason == "refusal":
            raise LLMError(f"model refused the request: {message.stop_details}")
        reply = _text_of(message)
        if not reply:
            raise LLMError("model returned no text")
        return reply
