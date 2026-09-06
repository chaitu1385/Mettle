"""Statistics tests: the correlation maths, pinned to hand-computed values.

These numbers are the ones most likely to be believed without being checked, so
they are asserted against cases worked out by hand -- perfect agreement, perfect
disagreement, ties, and the degenerate cases where Spearman is undefined.
Rendering is tested separately in test_report.py.
"""

from __future__ import annotations

import math

import pytest

from coach_rl.stats import (
    JUDGE_HIGH_TOTAL,
    JUDGE_LOW_TOTAL,
    LabeledTurn,
    action_distribution,
    confusion_table,
    explore_count,
    judge_correlations,
    mean_judge_score_per_action,
    spearman,
    to_labeled_turns,
)

from .conftest import make_record


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

    failing = [c.name for c in correlations.values() if c.below_threshold]
    assert failing == ["specificity"]


def test_a_strongly_negative_correlation_also_fails():
    """abs() would hide the worst case: a judge that is reliably backwards."""
    turns = [make_turn((s, s, s), -1 if s > 3 else 1, i) for i, s in enumerate([1, 2, 4, 5])]
    assert all(c.below_threshold for c in judge_correlations(turns))


def test_nothing_fails_when_every_dimension_tracks():
    turns = [make_turn((s, s, s), 1 if s > 3 else -1, i) for i, s in enumerate([1, 2, 4, 5])]
    assert not any(c.below_threshold for c in judge_correlations(turns))


def test_undefined_correlation_counts_as_failing():
    """No variance in the labels is not a pass -- it is "you cannot tell yet"."""
    turns = [make_turn((3, 3, 3), 1, i) for i in range(5)]
    correlations = judge_correlations(turns)
    assert not any(c.defined for c in correlations)
    assert all(c.below_threshold for c in correlations)


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
    assert means["ask"].means == pytest.approx(
        {"insight": 3.0, "specificity": 3.0, "forward_movement": 2.0}
    )
    assert means["ask"].total == pytest.approx(8.0)
    assert means["ask"].n == 2
    assert means["advise"].total == pytest.approx(15.0)
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
