"""Dataset statistics and judge validation.

Two jobs:

1. Describe the logged data -- turn count, action distribution, mean judge score
   per action, and how often the epsilon branch actually fired. If exploration
   is not firing at roughly eps, the "exploration" in the dataset is fictional.

2. Validate the judge against human labels. An LLM judge is a proxy reward, and
   a proxy nobody has checked is worse than no reward at all: it will look
   plausible and rank turns wrongly, and every downstream number inherits the
   error. So once there are enough human labels, correlate each judge dimension
   with the human label and print the turns where they disagree most sharply.

The functions here are pure (lists in, numbers out) so the correlation maths can
be tested without a database or an API key.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

from scipy import stats as scipy_stats

from .actions import ACTIONS
from .config import CORRELATION_WARN_THRESHOLD, MIN_LABELS_FOR_VALIDATION
from .schemas import JUDGE_DIMENSIONS, JUDGE_MAX_TOTAL, JUDGE_MIN_TOTAL
from .storage import TurnRecord

#: Total judge score (3-15) at or above which a turn counts as judge-high.
JUDGE_HIGH_TOTAL = 12
#: Total judge score at or below which a turn counts as judge-low.
JUDGE_LOW_TOTAL = 6

HUMAN_LABELS = (-1, 0, 1)


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
        # Signed on purpose: a judge that anti-correlates with you is broken too.
        return not self.defined or self.rho < CORRELATION_WARN_THRESHOLD


def to_labeled_turns(records: Iterable[TurnRecord]) -> list[LabeledTurn]:
    out: list[LabeledTurn] = []
    for record in records:
        scores = record.judge_scores
        if scores is None or record.human_label is None:
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


# --------------------------------------------------------------------------
# Descriptive statistics
# --------------------------------------------------------------------------


def action_distribution(records: Sequence[TurnRecord]) -> dict[str, int]:
    """Counts for every action, including the ones that never fired."""
    counts = {a: 0 for a in ACTIONS}
    for record in records:
        counts[record.action] = counts.get(record.action, 0) + 1
    return counts


def mean_judge_score_per_action(
    records: Sequence[TurnRecord],
) -> dict[str, dict[str, float] | None]:
    """Mean of each judge dimension (and the total) per action; None if unscored."""
    buckets: dict[str, list[dict[str, int]]] = {a: [] for a in ACTIONS}
    for record in records:
        scores = record.judge_scores
        if scores and all(d in scores for d in JUDGE_DIMENSIONS):
            buckets.setdefault(record.action, []).append(scores)

    means: dict[str, dict[str, float] | None] = {}
    for action, rows in buckets.items():
        if not rows:
            means[action] = None
            continue
        per_dim = {d: sum(r[d] for r in rows) / len(rows) for d in JUDGE_DIMENSIONS}
        per_dim["total"] = sum(per_dim[d] for d in JUDGE_DIMENSIONS)
        per_dim["n"] = float(len(rows))
        means[action] = per_dim
    return means


def explore_count(records: Sequence[TurnRecord]) -> int:
    """Turns where the epsilon branch actually replaced the policy's choice."""
    return sum(1 for record in records if record.explored)


def judged_count(records: Sequence[TurnRecord]) -> int:
    return sum(1 for r in records if r.judge_scores_json)


def labeled_count(records: Sequence[TurnRecord]) -> int:
    return sum(1 for r in records if r.human_label is not None)


# --------------------------------------------------------------------------
# Judge validation
# --------------------------------------------------------------------------


def spearman(x: Sequence[float], y: Sequence[float]) -> tuple[float, float]:
    """Spearman rho and p-value; (nan, nan) when it is undefined.

    Undefined means fewer than 3 pairs, or no variance in one of the inputs --
    e.g. every human label is +1. scipy warns and returns nan for that case;
    returning nan explicitly keeps the caller honest about it.
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
    correlations: list[Correlation] = []
    for dim in JUDGE_DIMENSIONS:
        rho, p = spearman([t.scores[dim] for t in turns], labels)
        correlations.append(Correlation(dim, rho, p, len(turns)))
    rho, p = spearman([t.total for t in turns], labels)
    correlations.append(Correlation("total", rho, p, len(turns)))
    return correlations


@dataclass(frozen=True)
class ConfusionTable:
    """Judge band against human label, plus the turns that disagree outright."""

    counts: dict[tuple[str, int], int]
    high_but_negative: list[LabeledTurn]
    low_but_positive: list[LabeledTurn]
    n: int

    def count(self, band: str, label: int) -> int:
        return self.counts.get((band, label), 0)

    @property
    def disagreements(self) -> list[LabeledTurn]:
        return [*self.high_but_negative, *self.low_but_positive]


def confusion_table(turns: Sequence[LabeledTurn]) -> ConfusionTable:
    counts: dict[tuple[str, int], int] = {
        (band, label): 0 for band in ("high", "mid", "low") for label in HUMAN_LABELS
    }
    high_but_negative: list[LabeledTurn] = []
    low_but_positive: list[LabeledTurn] = []
    for turn in turns:
        counts[(turn.band, turn.human_label)] = counts.get((turn.band, turn.human_label), 0) + 1
        if turn.band == "high" and turn.human_label == -1:
            high_but_negative.append(turn)
        elif turn.band == "low" and turn.human_label == 1:
            low_but_positive.append(turn)
    return ConfusionTable(counts, high_but_negative, low_but_positive, len(turns))


def validation_warnings(correlations: Sequence[Correlation]) -> list[str]:
    """One warning line per dimension that fails the correlation threshold."""
    warnings: list[str] = []
    for corr in correlations:
        if not corr.defined:
            warnings.append(
                f"{corr.name}: correlation undefined (no variance in the labels "
                f"or too few pairs, n={corr.n})"
            )
        elif corr.rho < CORRELATION_WARN_THRESHOLD:
            warnings.append(
                f"{corr.name}: rho={corr.rho:+.2f} is below "
                f"{CORRELATION_WARN_THRESHOLD} -- the judge is not tracking your "
                f"labels on this dimension; do not train on it yet"
            )
    return warnings


# --------------------------------------------------------------------------
# Report rendering
# --------------------------------------------------------------------------


def format_report(records: Sequence[TurnRecord]) -> str:
    lines: list[str] = []
    total = len(records)
    lines.append("=" * 68)
    lines.append("coach-rl dataset")
    lines.append("=" * 68)
    lines.append(f"turns:            {total}")
    lines.append(f"sessions:         {len({r.session_id for r in records})}")
    lines.append(f"judged turns:     {judged_count(records)}")
    lines.append(f"human-labeled:    {labeled_count(records)}")

    if total == 0:
        lines.append("\nNo turns logged yet. Run a session first.")
        return "\n".join(lines)

    explored = explore_count(records)
    eps_values = sorted({round(r.eps, 4) for r in records})
    lines.append(
        f"epsilon-explored: {explored} ({explored / total:.1%} of turns; "
        f"eps in use: {', '.join(str(e) for e in eps_values)})"
    )
    versions = sorted({r.prompt_version for r in records})
    models = sorted({r.model_name for r in records})
    lines.append(f"prompt versions:  {', '.join(versions)}")
    lines.append(f"models:           {', '.join(models)}")
    if len(versions) > 1 or len(models) > 1:
        lines.append(
            "  ! mixed prompt versions or models in one dataset -- split before training"
        )

    lines.append("")
    lines.append("-- action distribution ------------------------------------------")
    dist = action_distribution(records)
    means = mean_judge_score_per_action(records)
    lines.append(f"{'action':<12}{'n':>5}{'share':>9}{'insight':>9}{'specif.':>9}{'forward':>9}{'total':>8}")
    for action in ACTIONS:
        n = dist[action]
        row = f"{action:<12}{n:>5}{n / total:>9.1%}"
        m = means.get(action)
        if m is None:
            row += f"{'-':>9}{'-':>9}{'-':>9}{'-':>8}"
        else:
            row += (
                f"{m['insight']:>9.2f}{m['specificity']:>9.2f}"
                f"{m['forward_movement']:>9.2f}{m['total']:>8.2f}"
            )
        lines.append(row)
    lines.append("(mean judge scores cover judged turns only; the last turn of each")
    lines.append(" session is never judged -- nothing follows it to react.)")

    lines.append("")
    lines.append(format_validation(records))
    return "\n".join(lines)


def format_validation(records: Sequence[TurnRecord]) -> str:
    turns = to_labeled_turns(records)
    lines = ["-- judge validation ---------------------------------------------"]
    if len(turns) < MIN_LABELS_FOR_VALIDATION:
        lines.append(
            f"{len(turns)}/{MIN_LABELS_FOR_VALIDATION} turns have both a judge score "
            "and a human label."
        )
        lines.append("Validation runs once the threshold is reached. Label more turns:")
        lines.append("    coach-label --session <session_id>")
        return "\n".join(lines)

    correlations = judge_correlations(turns)
    lines.append(f"n = {len(turns)} turns with judge scores and human labels")
    lines.append("")
    lines.append(f"{'dimension':<20}{'spearman rho':>14}{'p':>12}")
    for corr in correlations:
        rho = "undefined" if not corr.defined else f"{corr.rho:+.3f}"
        p = "-" if math.isnan(corr.p_value) else f"{corr.p_value:.4f}"
        lines.append(f"{corr.name:<20}{rho:>14}{p:>12}")

    table = confusion_table(turns)
    lines.append("")
    lines.append("judge band x human label")
    lines.append(f"{'':<8}{'label -1':>10}{'label 0':>10}{'label +1':>10}")
    for band in ("high", "mid", "low"):
        lines.append(
            f"{band:<8}"
            + "".join(f"{table.count(band, label):>10}" for label in HUMAN_LABELS)
        )
    lines.append(
        f"(judge total ranges {JUDGE_MIN_TOTAL}-{JUDGE_MAX_TOTAL}; "
        f"high = >= {JUDGE_HIGH_TOTAL}, low = <= {JUDGE_LOW_TOTAL})"
    )

    if table.disagreements:
        lines.append("")
        lines.append(f"outright disagreements ({len(table.disagreements)}):")
        for turn in table.high_but_negative:
            lines.append(
                f"  judge HIGH ({turn.total}/15), you said BAD  "
                f"[{turn.session_id}#{turn.turn_index}, {turn.action}]"
            )
            lines.append(f"      judge: {turn.rationale}")
        for turn in table.low_but_positive:
            lines.append(
                f"  judge LOW  ({turn.total}/15), you said GOOD "
                f"[{turn.session_id}#{turn.turn_index}, {turn.action}]"
            )
            lines.append(f"      judge: {turn.rationale}")

    warnings = validation_warnings(correlations)
    if warnings:
        lines.append("")
        lines.append("WARNINGS")
        for warning in warnings:
            lines.append(f"  ! {warning}")
    else:
        lines.append("")
        lines.append(
            f"All correlations are at or above {CORRELATION_WARN_THRESHOLD}."
        )
    return "\n".join(lines)
