"""대국 성적에서 Elo 를 맞추기."""

import math

import pytest

from eval.elo import PairScore, fit_elo


def expected_score(difference: float) -> float:
    """점수가 `difference` 만큼 높은 쪽이 한 판에서 기대하는 성적 (승 1, 무 0.5)."""
    return 1 / (1 + 10 ** (-difference / 400))


def test_the_rating_gap_reproduces_the_score_without_a_prior() -> None:
    ratings = fit_elo(
        [PairScore("b", "a", wins=70, draws=10, losses=20)], anchor="a", prior_draws=0
    )

    assert ratings["a"] == 0
    # 75% 는 400 log10(3) 차이다.
    assert ratings["b"] == pytest.approx(400 * math.log10(3))


def test_exact_expected_scores_give_back_the_ratings_of_a_long_chain() -> None:
    true = {index: 15.0 * index for index in range(101)}
    scores = []
    for gap in (1, 5):
        for low in range(101 - gap):
            wins = 1000 * expected_score(true[low + gap] - true[low])
            scores.append(PairScore(low + gap, low, wins, 0, 1000 - wins))

    ratings = fit_elo(scores, anchor=0, prior_draws=0)

    assert ratings == pytest.approx(true, abs=1e-3)


def test_the_prior_keeps_a_player_who_won_everything_finite() -> None:
    sweep = [PairScore("b", "a", wins=20, draws=0, losses=0)]

    ratings = fit_elo(sweep, anchor="a", prior_draws=1)

    # 가상의 무승부 하나를 더해 20.5 / 21 이다.
    assert ratings["b"] == pytest.approx(400 * math.log10(20.5 / 0.5))


def test_the_anchor_can_be_given_another_rating() -> None:
    even = [PairScore("a", "b", wins=0, draws=4, losses=0)]

    assert fit_elo(even, anchor="b", anchor_rating=1000) == pytest.approx({"a": 1000, "b": 1000})


def test_a_player_without_a_path_to_the_anchor_is_refused() -> None:
    scores = [PairScore("a", "b", 1, 0, 1), PairScore("c", "d", 1, 0, 1)]

    with pytest.raises(ValueError, match="기준"):
        fit_elo(scores, anchor="a")


def test_a_sweep_without_a_prior_has_no_rating() -> None:
    with pytest.raises(ValueError):
        fit_elo([PairScore("b", "a", 5, 0, 0)], anchor="a", prior_draws=0)
