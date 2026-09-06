"""Typed state and judge objects.

These are the RL feature vector and the reward signal. They are Pydantic models
because they cross an LLM boundary: the classifier and the judge return JSON that
must be validated before it is allowed anywhere near the training data.

`extra="forbid"` matters twice over: it makes Pydantic emit
`additionalProperties: false` in the JSON schema handed to the API, and it makes
a hallucinated extra field a hard failure rather than a silently dropped one.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

UserIntent = Literal["uncertain", "explore", "vent", "seek_advice", "decide", "update"]
Specificity = Literal["vague", "moderate", "specific"]
SessionPhase = Literal["opening", "exploration", "deepening", "closing"]

#: |emotional_valence| at or above this counts as "emotional" for the rule policy.
EMOTIONAL_THRESHOLD = 0.5


class ClassifiedState(BaseModel):
    """The part of the state the classifier LLM produces."""

    model_config = ConfigDict(extra="forbid")

    user_intent: UserIntent = Field(
        description="What the user is doing with this message."
    )
    emotional_valence: float = Field(
        ge=-1.0,
        le=1.0,
        description="-1 strongly negative, 0 neutral, +1 strongly positive. "
        "Magnitude is intensity.",
    )
    specificity_level: Specificity = Field(
        description="How concrete the message is: named people/decisions/numbers "
        "are specific; abstractions are vague."
    )
    repeated_theme_flag: bool = Field(
        description="True if this message returns to a theme already raised in "
        "this session without new information."
    )
    session_phase: SessionPhase = Field(
        description="Where the conversation is: opening, exploration, deepening, "
        "or closing (the user is wrapping up or asking for a wrap-up)."
    )


class TurnState(ClassifiedState):
    """The full state the policy sees: classifier output plus runtime facts.

    `turn_index` is authoritative from the runtime, never from the model -- an
    LLM-supplied turn counter would drift and corrupt the episode ordering.
    """

    turn_index: int = Field(ge=0)

    @classmethod
    def from_classified(cls, classified: ClassifiedState, turn_index: int) -> "TurnState":
        return cls(**classified.model_dump(), turn_index=turn_index)

    def is_emotional(self) -> bool:
        return abs(self.emotional_valence) >= EMOTIONAL_THRESHOLD


class JudgeScores(BaseModel):
    """Reward signal for one turn, scored after the user's reaction is visible."""

    model_config = ConfigDict(extra="forbid")

    insight: int = Field(ge=1, le=5, description="Did the turn surface something the user had not said?")
    specificity: int = Field(ge=1, le=5, description="Was it concrete rather than generic coaching filler?")
    forward_movement: int = Field(
        ge=1,
        le=5,
        description="Did the user's next message move -- new information, a decision, "
        "a commitment -- rather than restate?",
    )
    rationale: str = Field(
        min_length=1,
        max_length=400,
        description="One line explaining the scores, naming the evidence in the user's reply.",
    )

    @property
    def total(self) -> int:
        return self.insight + self.specificity + self.forward_movement


# A module constant, not a model field: it must not leak into the JSON schema.
JUDGE_DIMENSIONS: tuple[str, ...] = ("insight", "specificity", "forward_movement")
