"""Rendering tests: the report says what the numbers mean.

Kept separate from test_stats.py so the statistics can be asserted as numbers
rather than as substrings of formatted text. What matters here is that the
report gates validation, surfaces disagreements, and refuses to hide a mixed
dataset -- not the exact column widths.
"""

from __future__ import annotations

from coach_rl.config import MIN_LABELS_FOR_VALIDATION
from coach_rl.report import format_report, warning_lines
from coach_rl.stats import judge_correlations

from .conftest import make_record
from .test_stats import make_turn



def test_report_holds_back_validation_until_the_threshold():
    scores = {"insight": 3, "specificity": 3, "forward_movement": 3}
    records = [make_record(i, scores=scores, human_label=1) for i in range(5)]
    report = format_report(records)
    assert f"5/{MIN_LABELS_FOR_VALIDATION}" in report
    assert "spearman" not in report.lower()


def test_report_runs_validation_once_there_are_enough_labels():
    records = []
    for i in range(MIN_LABELS_FOR_VALIDATION):
        score = 1 + (i % 5)
        records.append(
            make_record(
                i,
                action="ask" if i % 2 else "challenge",
                scores={
                    "insight": score,
                    "specificity": score,
                    "forward_movement": score,
                },
                human_label=1 if score > 3 else -1,
                explored=i % 5 == 0,
            )
        )
    report = format_report(records)
    assert "spearman rho" in report
    assert "judge band x human label" in report
    assert f"turns:            {MIN_LABELS_FOR_VALIDATION}" in report
    assert "epsilon-explored: 8" in report


def test_report_flags_mixed_prompt_versions():
    records = [
        make_record(0, prompt_version="v1"),
        make_record(1, prompt_version="v2"),
    ]
    assert "mixed prompt versions" in format_report(records)


def test_report_on_an_empty_database():
    assert "No turns logged yet" in format_report([])


def test_warning_names_the_failing_dimension_and_the_threshold():
    turns = [make_turn((s, 3, s), 1 if s > 3 else -1, i) for i, s in enumerate([1, 2, 4, 5])]
    warnings = warning_lines(judge_correlations(turns))
    assert len(warnings) == 1
    assert warnings[0].startswith("specificity:")
    assert "undefined" in warnings[0]


def test_disagreements_are_printed_with_their_rationales():
    records = [
        make_record(
            i,
            scores={"insight": 5, "specificity": 5, "forward_movement": 5},
            human_label=-1,
        )
        for i in range(MIN_LABELS_FOR_VALIDATION)
    ]
    records.append(
        make_record(
            MIN_LABELS_FOR_VALIDATION,
            scores={"insight": 1, "specificity": 1, "forward_movement": 1},
            human_label=1,
        )
    )
    report = format_report(records)
    assert "outright disagreements" in report
    assert "you said BAD" in report and "you said GOOD" in report
    assert "judge: rationale" in report
