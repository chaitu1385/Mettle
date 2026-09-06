"""Dataset statistics and judge validation -- computation only, no formatting.

Two jobs:

1. Describe the logged data: turn count, action distribution, mean judge score
   per action, and how often the epsilon branch actually fired. If exploration
   is not firing at roughly eps, the "exploration" in the dataset is fictional.

2. Validate the judge against human labels. An LLM judge is a proxy reward, and
   an unchecked proxy is worse than no reward: it looks plausible, ranks turns
   wrongly, and every downstream number inherits the error.

Everything here is pure -- records in, numbers out -- so the maths is testable
without a database, an API key, or a terminal. Rendering lives in `report.py`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

from scipy import stats as scipy_stats

from .actions import ACTIONS
from .config import CORRELATION_WARN_THRESHOLD
from .schemas import JUDGE_DIMENSIONS
from .storage import TurnRecord

#: Total judge score (3-15) at or above which a turn counts as judge-high.
JUDGE_HIGH_TOTAL = 12
#: Total judge score at or below which a turn counts as judge-low.
JUDGE_LOW_TOTAL = 6

HUMAN_LABELS = (-1, 0, 1)
BANDS = ("high", "mid", "low")


@dataclass(frozen=True)
class LabeledTurn:
    """A turn carrying both a judge score and a human label."""

    session_id: str
    turn_index: int
    action: str
    scores: dict[str, int]
    human_label: int
    rationale: str

    @property
    def total(self) -> int:
        return sum(self.scores[d] for d in JUDGE_DIMENSIONS)

    @property
    def band(self) -> str:
        if self.total >= JUDGE_HIGH_TOTAL:
            return "high"
        if self.total <= JUDGE_LOW_TOTAL:
            return "low"
        return "mid"


@dataclass(frozen=True)
class ActionScores:
    """Mean judge score per dimension for one action, over its judged turns."""

    n: int
    means: dict[str, float]

    @property
    def total(self) -> float:
        return sum(self.means.values())


@dataclass(frozen=True)
class Correlation:
    """Spearman result for one judge dimension against the human label."""

    name: str
    rho: float
    p_value: float
    n: int

    @property
    def defined(self) -> bool:
        return not math.isnan(self.rho)

    @property
    def below_threshold(self) -> bool:
        """The single definition of "this dimension fails validation".

        Signed on purpose: a judge that anti-correlates with you is broken too,
        and abs() would hide the worst case.
        """
        return not self.defined or self.rho < CORRELATION_WARN_THRESHOLD


@dataclass(frozen=True)
class ConfusionTable:
    """Judge band against human label, plus the turns that disagree outright."""

    counts: dict[tuple[str, int], int]
    high_but_negative: list[LabeledTurn]
    low_but_positive: list[LabeledTurn]

    def count(self, band: str, label: int) -> int:
        return self.counts.get((band, label), 0)

    @property
    def disagreements(self) -> list[LabeledTurn]:
        return [*self.high_but_negative, *self.low_but_positive]


def to_labeled_turns(records: Iterable[TurnRecord]) -> list[LabeledTurn]:
    """The validation set: turns carrying both a judge score and a human label."""
    out: list[LabeledTurn] = []
    for record in records:
        scores = record.judge_scores
        if record.human_label is None or not scores:
            continue
        if not all(d in scores for d in JUDGE_DIMENSIONS):
            continue
        out.append(
            LabeledTurn(
                session_id=record.session_id,
                turn_index=record.turn_index,
                action=record.action,
                scores={d: int(scores[d]) for d in JUDGE_DIMENSIONS},
                human_label=int(record.human_label),
                rationale=record.judge_rationale or "",
            )
        )
    return out


# --- descriptive statistics ------------------------------------------------


def action_distribution(records: Sequence[TurnRecord]) -> dict[str, int]:
    """Counts for every action, including the ones that never fired."""
    counts = {a: 0 for a in ACTIONS}
    for record in records:
        counts[record.action] = counts.get(record.action, 0) + 1
    return counts


def mean_judge_score_per_action(
    records: Sequence[TurnRecord],
) -> dict[str, ActionScores | None]:
    """Mean judge score per action; None where no turn of that action was judged."""
    buckets: dict[str, list[dict[str, int]]] = {a: [] for a in ACTIONS}
    for record in records:
        scores = record.judge_scores
        if scores and all(d in scores for d in JUDGE_DIMENSIONS):
            buckets.setdefault(record.action, []).append(scores)

    return {
        action: (
            ActionScores(
                n=len(rows),
                means={d: sum(r[d] for r in rows) / len(rows) for d in JUDGE_DIMENSIONS},
            )
            if rows
            else None
        )
        for action, rows in buckets.items()
    }


def explore_count(records: Sequence[TurnRecord]) -> int:
    """Turns where the epsilon branch actually replaced the policy's choice."""
    return sum(1 for record in records if record.explored)


def judged_count(records: Sequence[TurnRecord]) -> int:
    return sum(1 for r in records if r.judge_scores_json)


def labeled_count(records: Sequence[TurnRecord]) -> int:
    return sum(1 for r in records if r.human_label is not None)


# --- judge validation ------------------------------------------------------


def spearman(x: Sequence[float], y: Sequence[float]) -> tuple[float, float]:
    """Spearman rho and p-value; (nan, nan) when it is undefined.

    Undefined means fewer than 3 pairs, or no variance in one input -- e.g.
    every human label is +1. Returning nan explicitly keeps the caller honest
    about it rather than handing back a number that means nothing.
    """
    if len(x) != len(y):
        raise ValueError("x and y must be the same length")
    if len(x) < 3 or len(set(x)) < 2 or len(set(y)) < 2:
        return (math.nan, math.nan)
    result = scipy_stats.spearmanr(list(x), list(y))
    return (float(result.statistic), float(result.pvalue))


def judge_correlations(turns: Sequence[LabeledTurn]) -> list[Correlation]:
    """Correlate each judge dimension, and the summed score, with the human label."""
    labels = [t.human_label for t in turns]
    correlations = [
        Correlation(dim, *spearman([t.scores[dim] for t in turns], labels), len(turns))
        for dim in JUDGE_DIMENSIONS
    ]
    correlations.append(
        Correlation("total", *spearman([t.total for t in turns], labels), len(turns))
    )
    return correlations


def confusion_table(turns: Sequence[LabeledTurn]) -> ConfusionTable:
    counts = {(band, label): 0 for band in BANDS for label in HUMAN_LABELS}
    high_but_negative: list[LabeledTurn] = []
    low_but_positive: list[LabeledTurn] = []
    for turn in turns:
        counts[(turn.band, turn.human_label)] += 1
        if turn.band == "high" and turn.human_label == -1:
            high_but_negative.append(turn)
        elif turn.band == "low" and turn.human_label == 1:
            low_but_positive.append(turn)
    return ConfusionTable(counts, high_but_negative, low_but_positive)
