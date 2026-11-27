"""시험 판 뽑기. 층의 조건, 개수, 겹침, 시드만 본다."""

import pytest

from eval.solver import Solution, Value, Verdict, solve
from eval.trial_positions import Stratum, Trial, select_trials
from games.base import Player
from games.mini_gungi import Gungi
from tests.solver_helpers import SMALL_NANO

STRATA = (
    Stratum("draw", 20, lambda position, value: value.verdict is Verdict.DRAW),
    Stratum("win in 3", 5, lambda position, value: value == Value(Verdict.WIN, 3)),
)


@pytest.fixture(scope="module")
def solution() -> Solution:
    return solve(Gungi(SMALL_NANO), move_limit=SMALL_NANO.move_limit)


def network_boards(solution: Solution, trials: tuple[Trial, ...]) -> list:
    game = solution.game
    return [game.board_key(game.canonical(trial.position)) for trial in trials]


def test_each_stratum_gets_its_count_of_unfinished_positions_it_accepts(
    solution: Solution,
) -> None:
    trials = select_trials(solution, STRATA, seed=0)

    assert [trial.stratum for trial in trials] == ["draw"] * 20 + ["win in 3"] * 5
    for trial in trials:
        assert solution.game.outcome(trial.position) is None
        assert trial.value == solution.value(trial.position)
    assert {trial.value for trial in trials[20:]} == {Value(Verdict.WIN, 3)}


def test_no_two_trials_look_the_same_to_the_network(solution: Solution) -> None:
    """판을 180° 돌리고 색을 바꾼 국면은 신경망에 같은 입력이다."""
    everything = (Stratum("any", 200, lambda position, value: True),)

    trials = select_trials(solution, everything, seed=0)

    assert len(set(network_boards(solution, trials))) == 200
    # 한쪽 차례에 치우치지 않는다.
    assert {trial.position.player for trial in trials} == set(Player)


def test_overlapping_strata_do_not_share_a_board(solution: Solution) -> None:
    twice = tuple(Stratum(name, 30, lambda position, value: True) for name in ("one", "other"))

    trials = select_trials(solution, twice, seed=0)

    assert len(set(network_boards(solution, trials))) == 60


def test_the_seed_fixes_the_selection(solution: Solution) -> None:
    def boards(seed: int) -> list:
        trials = select_trials(solution, STRATA, seed=seed)
        return [solution.game.render(trial.position) for trial in trials]

    assert boards(3) == boards(3)
    assert boards(3) != boards(4)


def test_a_stratum_without_enough_positions_is_refused(solution: Solution) -> None:
    too_many = (Stratum("draw", 10_000, lambda position, value: value.verdict is Verdict.DRAW),)
    with pytest.raises(ValueError, match="draw"):
        select_trials(solution, too_many, seed=0)
