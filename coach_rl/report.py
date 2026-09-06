"""Terminal rendering of the dataset report. All numbers come from `stats.py`.

Presentation is separated from computation so the statistics stay testable
without asserting on formatted strings, and so a second consumer -- a notebook,
a CSV export -- can use the same numbers without going through a text layer.
"""

from __future__ import annotations

import math
from typing import Sequence

from .actions import ACTIONS
from .stats import (
    BANDS,
    CORRELATION_WARN_THRESHOLD,
    HUMAN_LABELS,
    JUDGE_HIGH_TOTAL,
    JUDGE_LOW_TOTAL,
    JUDGE_MAX_TOTAL,
    JUDGE_MIN_TOTAL,
    MIN_LABELS_FOR_VALIDATION,
    Correlation,
    LabeledTurn,
    action_distribution,
    confusion_table,
    explore_count,
    judge_correlations,
    judged_count,
    labeled_count,
    mean_judge_score_per_action,
    to_labeled_turns,
)
from .storage import TurnRecord


def warning_lines(correlations: Sequence[Correlation]) -> list[str]:
    """One line per dimension that fails validation, per `Correlation.below_threshold`."""
    lines = []
    for corr in correlations:
        if not corr.defined:
            lines.append(
                f"{corr.name}: correlation undefined (no variance in the labels "
                f"or too few pairs, n={corr.n})"
            )
        elif corr.below_threshold:
            lines.append(
                f"{corr.name}: rho={corr.rho:+.2f} is below "
                f"{CORRELATION_WARN_THRESHOLD} -- the judge is not tracking your "
                f"labels on this dimension; do not train on it yet"
            )
    return lines


def format_report(records: Sequence[TurnRecord]) -> str:
    lines = [
        "=" * 66,
        "coach-rl dataset",
        "=" * 66,
        f"turns:            {len(records)}",
        f"sessions:         {len({r.session_id for r in records})}",
        f"judged turns:     {judged_count(records)}",
        f"human-labeled:    {labeled_count(records)}",
    ]
    if not records:
        lines.append("\nNo turns logged yet. Run a session first.")
        return "\n".join(lines)

    lines.append(_provenance(records))
    lines.append("")
    lines.append(_action_table(records))
    lines.append("")
    lines.append(format_validation(records))
    return "\n".join(lines)


def _provenance(records: Sequence[TurnRecord]) -> str:
    """Exploration rate and the prompt/model mix -- the dataset's integrity check."""
    explored = explore_count(records)
    eps_values = sorted({round(r.eps, 4) for r in records})
    versions = sorted({r.prompt_version for r in records})
    models = sorted({r.model_name for r in records})
    lines = [
        f"epsilon-explored: {explored} ({explored / len(records):.1%} of turns; "
        f"eps in use: {', '.join(str(e) for e in eps_values)})",
        f"prompt versions:  {', '.join(versions)}",
        f"models:           {', '.join(models)}",
    ]
    if len(versions) > 1 or len(models) > 1:
        lines.append(
            "  ! mixed prompt versions or models in one dataset -- split before training"
        )
    return "\n".join(lines)


def _action_table(records: Sequence[TurnRecord]) -> str:
    distribution = action_distribution(records)
    means = mean_judge_score_per_action(records)
    lines = [
        "-- action distribution " + "-" * 43,
        f"{'action':<12}{'n':>5}{'share':>9}"
        f"{'insight':>9}{'specif.':>9}{'forward':>9}{'total':>8}",
    ]
    for action in ACTIONS:
        n = distribution[action]
        row = f"{action:<12}{n:>5}{n / len(records):>9.1%}"
        scores = means[action]
        if scores is None:
            row += f"{'-':>9}{'-':>9}{'-':>9}{'-':>8}"
        else:
            row += "".join(f"{v:>9.2f}" for v in scores.means.values())
            row += f"{scores.total:>8.2f}"
        lines.append(row)
    lines.append("(mean judge scores cover judged turns only; the last turn of each")
    lines.append(" session is never judged -- nothing follows it to react.)")
    return "\n".join(lines)


def format_validation(records: Sequence[TurnRecord]) -> str:
    turns = to_labeled_turns(records)
    lines = ["-- judge validation " + "-" * 46]
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

    lines.append("")
    lines.append(_confusion_block(turns))

    warnings = warning_lines(correlations)
    lines.append("")
    if warnings:
        lines.append("WARNINGS")
        lines.extend(f"  ! {w}" for w in warnings)
    else:
        lines.append(f"All correlations are at or above {CORRELATION_WARN_THRESHOLD}.")
    return "\n".join(lines)


def _confusion_block(turns: Sequence[LabeledTurn]) -> str:
    table = confusion_table(turns)
    lines = [
        "judge band x human label",
        f"{'':<8}{'label -1':>10}{'label 0':>10}{'label +1':>10}",
    ]
    for band in BANDS:
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
                f"  judge HIGH ({turn.total}/{JUDGE_MAX_TOTAL}), you said BAD  "
                f"[{turn.session_id}#{turn.turn_index}, {turn.action}]"
            )
            lines.append(f"      judge: {turn.rationale}")
        for turn in table.low_but_positive:
            lines.append(
                f"  judge LOW  ({turn.total}/{JUDGE_MAX_TOTAL}), you said GOOD "
                f"[{turn.session_id}#{turn.turn_index}, {turn.action}]"
            )
            lines.append(f"      judge: {turn.rationale}")
    return "\n".join(lines)
