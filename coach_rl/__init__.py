"""coach-rl: an RL-instrumented executive coaching agent.

The coaching is the wrapper; the point is the decision loop. See README.md.
"""

from .actions import ACTIONS
from .prompts import PROMPT_VERSION

__version__ = "0.1.0"

__all__ = ["ACTIONS", "PROMPT_VERSION", "__version__"]
