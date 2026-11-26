"""측정 보고. 랜덤 대국의 통계가 맞게 세어지고 시드에서 재현되는지 본다."""

from dataclasses import replace

import pytest

from games.base import Outcome
from games.mini_gungi import MARSHAL_ONLY, MINI_GUNGI, EndReason
from scripts.measurement_report import Measurement, main, measure, render_summary


def test_every_one_move_game_is_counted_as_a_move_limit_draw() -> None:
    # 마주 본 帥 는 첫 수에 서로 닿지 않고, 첫 수는 다섯 개다.
    measured = measure(replace(MARSHAL_ONLY, move_limit=1), episodes=7, seed=0)
    assert measured.outcomes == {Outcome.DRAW: 7}
    assert measured.reasons == {EndReason.MOVE_LIMIT: 7}
    assert measured.lengths == (1,) * 7
    assert measured.mean_branching_factor == 5


def test_the_branching_factor_averages_over_every_position_moved_from() -> None:
    # 첫 국면의 합법수는 12 개이고, 2 수째 국면은 12 개나 13 개다.
    measured = measure(replace(MINI_GUNGI, move_limit=2), episodes=40, seed=0)
    assert 12 < measured.mean_branching_factor < 12.5


def test_each_episode_has_one_outcome_one_reason_and_one_length() -> None:
    measured = measure(MINI_GUNGI, episodes=30, seed=1)
    assert measured.episodes == 30
    assert sum(measured.outcomes.values()) == 30
    assert sum(measured.reasons.values()) == 30
    assert len(measured.lengths) == 30
    draws = measured.reasons.get(EndReason.REPETITION, 0) + measured.reasons.get(
        EndReason.MOVE_LIMIT, 0
    )
    assert measured.outcomes.get(Outcome.DRAW, 0) == draws


def test_the_same_seed_reproduces_the_measurement() -> None:
    first = measure(MINI_GUNGI, episodes=20, seed=5)
    assert measure(MINI_GUNGI, episodes=20, seed=5) == first
    assert measure(MINI_GUNGI, episodes=20, seed=25) != first


def test_neighbouring_episodes_use_neighbouring_seeds() -> None:
    whole = measure(MINI_GUNGI, episodes=6, seed=10)
    halves = measure(MINI_GUNGI, 3, seed=10).lengths + measure(MINI_GUNGI, 3, seed=13).lengths
    assert whole.lengths == halves


def test_a_measurement_needs_an_episode() -> None:
    with pytest.raises(ValueError):
        measure(MINI_GUNGI, episodes=0, seed=0)


def test_the_summary_reports_every_required_figure() -> None:
    measured = Measurement(
        config=MINI_GUNGI,
        seed=0,
        outcomes={Outcome.FIRST_WINS: 5, Outcome.SECOND_WINS: 3, Outcome.DRAW: 2},
        reasons={EndReason.MARSHAL_CAPTURE: 8, EndReason.REPETITION: 2},
        lengths=(10, 20, 30, 40, 50, 60, 70, 80, 90, 200),
        mean_branching_factor=12.345,
    )
    summary = render_summary(measured)
    for line in (
        "| 선수 승률 | 50.0% |",
        "| 후수 승률 | 30.0% |",
        "| 무승부율 | 20.0% |",
        "| 평균 대국 길이 | 65.0 |",
        "| 최대 대국 길이 | 200 |",
        "| 평균 분기 계수 | 12.35 |",
        "| 帥 잡기 | 8 | 80.0% |",
        "| 수 제한 | 0 | 0.0% |",
        "| 반복 | 2 | 20.0% |",
        "| 합법수 없음 | 0 | 0.0% |",
    ):
        assert line in summary


def test_the_report_says_what_was_measured(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--episodes", "3", "--seed", "8"]) == 0
    printed = capsys.readouterr().out
    assert "랜덤 대국 3판, 시드 8~10" in printed
    assert "수 제한 200, 반복 한도 3" in printed
    assert " 1   .      ▲大     ▲帥     ▲槍      ." in printed
    assert "| 평균 분기 계수 |" in printed
