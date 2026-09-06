"""Guard on the prompt version.

Prompt text and PROMPT_VERSION must move together. If they do not, one dataset
ends up containing two behaviours under one version label, and nothing later can
separate them -- the rows look fine and the conclusions are wrong.

If this test fails you edited a prompt. Bump PROMPT_VERSION and paste the new
fingerprint into prompts.py.
"""

from __future__ import annotations

import pytest

from coach_rl.actions import ACTIONS
from coach_rl.prompts import (
    ACTION_INSTRUCTIONS,
    PROMPT_FINGERPRINT,
    PROMPT_VERSION,
    action_instruction,
    compute_fingerprint,
)


def test_prompt_fingerprint_matches_the_recorded_version():
    assert compute_fingerprint() == PROMPT_FINGERPRINT, (
        "Prompt text changed. Bump PROMPT_VERSION (currently "
        f"{PROMPT_VERSION!r}) and set PROMPT_FINGERPRINT to "
        f"{compute_fingerprint()!r}."
    )


def test_every_action_has_an_instruction():
    assert set(ACTION_INSTRUCTIONS) == set(ACTIONS)
    for action in ACTIONS:
        assert action_instruction(action).strip()


def test_unknown_action_has_no_instruction():
    with pytest.raises(ValueError):
        action_instruction("empathize")
