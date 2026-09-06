"""Statistics tests, with the correlation maths checked against known values.

The judge-validation numbers are the ones most likely to be believed without
being checked, so they are pinned against hand-computable cases: perfect
agreement, perfect disagreement, a tie-heavy case, and the degenerate cases
where Spearman is simply undefined.
"""

from __future__ import annotations

import json
import math

import pytest

from coach_rl.config import CORRELATION_WARN_THRESHOLD, MIN_LABELS_FOR_VALIDATION
from coach_rl.stats import (
    JUDGE_HIGH_TOTAL,
    JUDGE_LOW_TOTAL,
    LabeledTurn,
    action_distribution,
    confusion_table,
    explore_count,
    format_report,
    judge_correlations,
    mean_judge_score_per_action,
    spearman,
    to_labeled_turns,
    validation_warnings,
)
from coach_rl.storage import TurnRecord


def make_turn(
    scores: tuple[int, int, int],
    human_label: int,
    turn_index: int = 0,
    action: str = "ask",
    rationale: str = "because",
) -> LabeledTurn:
    insight, specificity, forward = scores
    return LabeledTurn(
        session_id="s1",
        turn_index=turn_index,
        action=action,
        scores={
            "insight": insight,
            "specificity": specificity,
            "forward_movement": forward,
        },
        human_label=human_label,
        rationale=rationale,
    )


def make_record(
    turn_index: int,
    action: str = "ask",
    scores: dict | None = None,
    human_label: int | None = None,
    explored: bool = False,
    eps: float = 0.2,
    prompt_version: str = "v1",
    model_name: str = "claude-opus-5",
) -> TurnRecord:
    probs = [0.05, 0.05, 0.8, 0.05, 0.05]
    return TurnRecord(
        session_id="s1",
        turn_index=turn_index,
        state_json=json.dumps({"user_intent": "explore", "turn_index": turn_index}),
        action=action,
        action_probs_json=json.dumps(
            {
                "order": ["ask", "reflect", "challenge", "advise", "summarize"],
                "probs": probs,
                "base_probs": [0, 0, 1, 0, 0],
                "explored": explored,
            }
        ),
        policy_id="rule/v1+eps0.2",
        eps=eps,
        response_text="reply",
        user_message="message",
        model_name=model_name,
        prompt_version=prompt_version,
        judge_scores_json=json.dumps(scores) if scores else None,
        judge_rationale="rationale" if scores else None,
        human_label=human_label,
    )


# --- spearman --------------------------------------------------------------


def test_spearman_perfect_positive_and_negative():
    x = [1, 2, 3, 4, 5]
    rho, p = spearman(x, [1, 2, 3, 4, 5])
    assert rho == pytest.approx(1.0)
    assert p < 0.05

    rho, _ = spearman(x, [5, 4, 3, 2, 1])
    assert rho == pytest.approx(-1.0)


def test_spearman_is_rank_based_not_linear():
    """Monotone but very non-linear still gives rho = 1; Pearson would not."""
    rho, _ = spearman([1, 2, 3, 4, 5], [1, 2, 4, 8, 1000])
    assert rho == pytest.approx(1.0)


def test_spearman_known_value_with_a_single_swap():
    """Ranks [1,2,3,4,5] vs [2,1,3,4,5]: sum d^2 = 2, rho = 1 - 6*2/(5*24) = 0.9."""
    rho, _ = spearman([1, 2, 3, 4, 5], [2, 1, 3, 4, 5])
    assert rho == pytest.approx(0.9)


def test_spearman_handles_ties():
    """Tied judge scores are the common case: 1-5 integers across many turns."""
    rho, _ = spearman([3, 3, 4, 4, 5, 5], [0, 0, 1, 1, 1, 1])
    assert 0.7 < rho < 0.9


def test_spearman_is_undefined_rather_than_wrong():
    assert all(math.isnan(v) for v in spearman([1, 2], [1, 2]))  # too few pairs
    assert all(math.isnan(v) for v in spearman([1, 1, 1], [1, 2, 3]))  # no variance in x
    assert all(math.isnan(v) for v in spearman([1, 2, 3], [2, 2, 2]))  # no variance in y


def test_spearman_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        spearman([1, 2, 3], [1, 2])


# --- judge correlations ----------------------------------------------------


def test_judge_correlations_cover_each_dimension_and_the_total():
    turns = [
        make_turn((1, 1, 1), -1, 0),
        make_turn((2, 2, 2), -1, 1),
        make_turn((3, 3, 3), 0, 2),
        make_turn((4, 4, 4), 1, 3),
        make_turn((5, 5, 5), 1, 4),
    ]
    # Human labels only take three values, so the label ranks tie: score ranks
    # [1,2,3,4,5] against label ranks [1.5,1.5,3,4.5,4.5] gives
    # rho = 9 / sqrt(10 * 9) = 0.9487, not 1.0.
    expected = 9 / math.sqrt(90)
    correlations = {c.name: c for c in judge_correlations(turns)}
    assert set(correlations) == {"insight", "specificity", "forward_movement", "total"}
    for corr in correlations.values():
        assert corr.rho == pytest.approx(expected, abs=1e-4)
        assert corr.n == 5


def test_a_judge_that_disagrees_correlates_negatively():
    turns = [
        make_turn((5, 5, 5), -1, 0),
        make_turn((4, 4, 4), -1, 1),
        make_turn((3, 3, 3), 0, 2),
        make_turn((1, 1, 1), 1, 3),
    ]
    # Score ranks [4,3,2,1] against label ranks [1.5,1.5,3,4]:
    # rho = -4.5 / sqrt(5 * 4.5) = -0.9487.
    total = {c.name: c for c in judge_correlations(turns)}["total"]
    assert total.rho == pytest.approx(-4.5 / math.sqrt(22.5), abs=1e-4)
    assert total.below_threshold


def test_one_dimension_can_be_uncorrelated_while_the_others_track():
    """The per-dimension breakdown exists to catch exactly this."""
    turns = [
        make_turn((1, 3, 1), -1, 0),
        make_turn((2, 3, 2), -1, 1),
        make_turn((4, 3, 4), 1, 2),
        make_turn((5, 3, 5), 1, 3),
    ]
    # Score ranks [1,2,3,4] against label ranks [1.5,1.5,3.5,3.5]:
    # rho = 4 / sqrt(5 * 4) = 0.8944.
    correlations = {c.name: c for c in judge_correlations(turns)}
    assert correlations["insight"].rho == pytest.approx(4 / math.sqrt(20), abs=1e-4)
    assert not correlations["specificity"].defined  # constant column
    assert correlations["specificity"].below_threshold


# --- warnings --------------------------------------------------------------


def test_warns_only_for_the_dimension_that_fails():
    """insight and forward_movement track the labels; specificity is scrambled to
    rho = 0 exactly (its rank deviations cancel against the label ranks)."""
    turns = [
        make_turn((1, 5, 1), -1, 0),
        make_turn((2, 1, 2), -1, 1),
        make_turn((3, 3, 3), 0, 2),
        make_turn((4, 4, 4), 1, 3),
        make_turn((5, 2, 5), 1, 4),
    ]
    correlations = {c.name: c for c in judge_correlations(turns)}
    assert correlations["specificity"].rho == pytest.approx(0.0, abs=1e-9)
    assert correlations["insight"].rho == pytest.approx(9 / math.sqrt(90), abs=1e-4)

    warnings = validation_warnings(list(correlations.values()))
    assert len(warnings) == 1
    assert warnings[0].startswith("specificity:")
    assert str(CORRELATION_WARN_THRESHOLD) in warnings[0]


def test_a_strongly_negative_correlation_also_warns():
    """abs() would hide the worst case: a judge that is reliably backwards."""
    turns = [make_turn((s, s, s), -1 if s > 3 else 1, i) for i, s in enumerate([1, 2, 4, 5])]
    warnings = validation_warnings(judge_correlations(turns))
    assert len(warnings) == 4


def test_no_warnings_when_every_dimension_tracks():
    turns = [make_turn((s, s, s), 1 if s > 3 else -1, i) for i, s in enumerate([1, 2, 4, 5])]
    assert validation_warnings(judge_correlations(turns)) == []


def test_undefined_correlation_is_warned_not_hidden():
    turns = [make_turn((3, 3, 3), 1, i) for i in range(5)]
    warnings = validation_warnings(judge_correlations(turns))
    assert len(warnings) == 4
    assert all("undefined" in w for w in warnings)


# --- confusion table -------------------------------------------------------


def test_confusion_table_counts_and_disagreements():
    turns = [
        make_turn((5, 5, 5), -1, 0, rationale="judge loved it, human did not"),
        make_turn((5, 4, 4), 1, 1),
        make_turn((1, 1, 1), 1, 2, rationale="judge hated it, human did not"),
        make_turn((1, 2, 1), -1, 3),
        make_turn((3, 3, 3), 0, 4),
    ]
    table = confusion_table(turns)
    assert table.n == 5
    assert table.count("high", -1) == 1
    assert table.count("high", 1) == 1
    assert table.count("low", 1) == 1
    assert table.count("mid", 0) == 1

    assert [t.turn_index for t in table.high_but_negative] == [0]
    assert [t.turn_index for t in table.low_but_positive] == [2]
    rationales = [t.rationale for t in table.disagreements]
    assert "judge loved it, human did not" in rationales
    assert "judge hated it, human did not" in rationales


def test_band_boundaries_are_inclusive():
    assert make_turn((4, 4, 4), 0).total == JUDGE_HIGH_TOTAL
    assert make_turn((4, 4, 4), 0).band == "high"
    assert make_turn((2, 2, 2), 0).total == JUDGE_LOW_TOTAL
    assert make_turn((2, 2, 2), 0).band == "low"
    assert make_turn((3, 2, 2), 0).band == "mid"


# --- descriptive statistics ------------------------------------------------


def test_action_distribution_includes_actions_that_never_fired():
    records = [make_record(0, "ask"), make_record(1, "ask"), make_record(2, "challenge")]
    dist = action_distribution(records)
    assert dist == {"ask": 2, "reflect": 0, "challenge": 1, "advise": 0, "summarize": 0}


def test_explore_count_reads_the_logged_flag():
    records = [
        make_record(0, explored=False),
        make_record(1, explored=True),
        make_record(2, explored=True),
    ]
    assert explore_count(records) == 2


def test_mean_judge_score_per_action_skips_unjudged_turns():
    records = [
        make_record(0, "ask", scores={"insight": 4, "specificity": 2, "forward_movement": 3}),
        make_record(1, "ask", scores={"insight": 2, "specificity": 4, "forward_movement": 1}),
        make_record(2, "ask"),  # unjudged: must not drag the mean toward zero
        make_record(3, "advise", scores={"insight": 5, "specificity": 5, "forward_movement": 5}),
    ]
    means = mean_judge_score_per_action(records)
    assert means["ask"]["insight"] == pytest.approx(3.0)
    assert means["ask"]["specificity"] == pytest.approx(3.0)
    assert means["ask"]["forward_movement"] == pytest.approx(2.0)
    assert means["ask"]["total"] == pytest.approx(8.0)
    assert means["ask"]["n"] == 2
    assert means["advise"]["total"] == pytest.approx(15.0)
    assert means["reflect"] is None


def test_to_labeled_turns_requires_both_signals():
    scores = {"insight": 3, "specificity": 3, "forward_movement": 3}
    records = [
        make_record(0, scores=scores, human_label=1),
        make_record(1, scores=scores),  # judged, unlabeled
        make_record(2, human_label=-1),  # labeled, unjudged
        make_record(3, scores={"insight": 3}, human_label=1),  # partial judge payload
    ]
    labeled = to_labeled_turns(records)
    assert [t.turn_index for t in labeled] == [0]


# --- report ----------------------------------------------------------------


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
