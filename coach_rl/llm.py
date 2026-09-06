"""The model boundary: the only module that knows the vendor.

`graph.py`, `judge.py` and `session.py` take an `LLM` and call two methods on
it. Anything with those two methods works -- the tests pass a stub and run the
whole decision loop with no network and no API key.

Two call shapes are all the loop needs:

- `structured()` -- JSON constrained to a Pydantic model's schema, validated on
  the way back, retried exactly once with the validation error fed back in.
  Structured outputs make a malformed reply rare; the retry is here because
  "rare" is not "never", and a turn that fails validation must not be silently
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

T = TypeVar("T", bound=BaseModel)

MAX_TOKENS = 1024


class LLMError(RuntimeError):
    """A model call that cannot produce a usable result. Never silently swallowed."""


def _json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Pydantic schema in the shape the Messages API structured-output format wants."""
    schema = model.model_json_schema()
    schema.pop("title", None)
    return schema


def _text_of(message: Any) -> str:
    return "".join(
        block.text for block in message.content if getattr(block, "type", None) == "text"
    ).strip()


class LLM:
    """Async Anthropic client bound to one model, which is logged on every row."""

    def __init__(self, model: str, client: anthropic.AsyncAnthropic | None = None):
        self.model = model
        self.client = client or anthropic.AsyncAnthropic()

    async def _create(self, **kwargs: Any) -> Any:
        message = await self.client.messages.create(
            model=self.model, max_tokens=MAX_TOKENS, **kwargs
        )
        if message.stop_reason == "refusal":
            raise LLMError(f"model refused the request: {message.stop_details}")
        return message

    async def structured(
        self, *, system: str, user: str, schema: type[T], effort: str = "low"
    ) -> T:
        """Return a validated instance of `schema`, retrying once on a bad parse."""
        messages: list[dict[str, Any]] = [{"role": "user", "content": user}]
        last_error: Exception | None = None

        for attempt in range(2):
            message = await self._create(
                system=system,
                messages=messages,
                output_config={
                    "effort": effort,
                    "format": {"type": "json_schema", "schema": _json_schema(schema)},
                },
            )
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

        raise LLMError(f"{schema.__name__} failed validation twice: {last_error}")

    async def text(
        self, *, system: str, messages: Sequence[dict[str, Any]], effort: str = "medium"
    ) -> str:
        message = await self._create(
            system=system, messages=list(messages), output_config={"effort": effort}
        )
        reply = _text_of(message)
        if not reply:
            raise LLMError("model returned no text")
        return reply
