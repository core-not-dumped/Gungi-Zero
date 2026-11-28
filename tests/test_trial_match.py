"""시험 판 대국과 최선수 비율. 판정과 센 수만 본다."""

import pytest

from ai.batched import EvaluationCache
from ai.mcts import MCTS, MCTSAI
from ai.random_play import RandomAI
from eval.best_move_rate import best_move_rate
from eval.perfect import PerfectAI
from eval.solver import Solution, Verdict, solve
from eval.trial_match import play_trials, play_trials_together
from eval.trial_positions import Stratum, Trial, select_trials
from games.mini_gungi import Gungi
from tests.solver_helpers import SMALL_NANO, shortest_path_positions
from tests.test_batched import PerBoardNetwork

STRATA = (
    Stratum("draw", 10, lambda position, value: value.verdict is Verdict.DRAW),
    Stratum("win", 10, lambda position, value: value.verdict is Verdict.WIN),
    Stratum("loss", 5, lambda position, value: value.verdict is Verdict.LOSS),
)


@pytest.fixture(scope="module")
def solution() -> Solution:
    return solve(Gungi(SMALL_NANO), move_limit=SMALL_NANO.move_limit)


@pytest.fixture(scope="module")
def trials(solution: Solution) -> tuple[Trial, ...]:
    return select_trials(solution, STRATA, seed=0)


def test_the_perfect_challenger_keeps_every_value(
    solution: Solution, trials: tuple[Trial, ...]
) -> None:
    results = play_trials(
        solution.game,
        trials[:20],
        lambda index: PerfectAI(solution, 100 + index),
        lambda index: PerfectAI(solution, index),
    )

    assert [result.trial for result in results] == list(trials[:20])
    assert all(result.passed for result in results)
    for result in results:
        # 시험 판에서 시작해, 둘 차례였던 쪽의 성적으로 본다.
        winner = result.episode.outcome.winner
        if result.trial.value.verdict is Verdict.WIN:
            assert winner is result.trial.position.player
            assert len(result.episode.moves) == result.trial.value.dtm
        else:
            assert winner is None


def test_the_ais_are_made_from_the_index_of_the_trial(
    solution: Solution, trials: tuple[Trial, ...]
) -> None:
    challengers: list[int] = []
    opponents: list[int] = []

    def challenger(index: int) -> PerfectAI:
        challengers.append(index)
        return PerfectAI(solution, index)

    def opponent(index: int) -> PerfectAI:
        opponents.append(index)
        return PerfectAI(solution, index)

    play_trials(solution.game, trials[:3], challenger, opponent)

    assert challengers == opponents == [0, 1, 2]


def test_a_missed_win_and_a_lost_draw_fail(solution: Solution, trials: tuple[Trial, ...]) -> None:
    results = play_trials(
        solution.game,
        trials[:20],
        lambda index: RandomAI(solution.game, index),
        lambda index: PerfectAI(solution, index),
    )

    for result in results:
        winner = result.episode.outcome.winner
        player = result.trial.position.player
        if result.trial.value.verdict is Verdict.WIN:
            assert result.passed is (winner is player)
        else:
            assert result.passed is (winner is not player.opponent)
    assert not all(result.passed for result in results)


def test_a_lost_position_is_not_a_trial(solution: Solution, trials: tuple[Trial, ...]) -> None:
    assert trials[20].value.verdict is Verdict.LOSS
    with pytest.raises(ValueError, match="loss"):
        play_trials(
            solution.game,
            trials[20:21],
            lambda index: PerfectAI(solution, index),
            lambda index: PerfectAI(solution, index),
        )


def test_the_best_move_rate_is_counted_overall_and_in_the_group(solution: Solution) -> None:
    game = solution.game
    positions = [p for p in shortest_path_positions(game) if game.outcome(p) is None][:200]
    odd = sum(p.ply % 2 for p in positions)

    perfect = best_move_rate(
        solution, positions, PerfectAI(solution, 0), in_group=lambda p: p.ply % 2 == 1
    )
    random_ai = best_move_rate(
        solution, positions, RandomAI(game, 0), in_group=lambda p: p.ply % 2 == 1
    )

    assert (perfect.all.matches, perfect.all.positions) == (200, 200)
    assert (perfect.group.matches, perfect.group.positions) == (odd, odd)
    assert (perfect.rest.matches, perfect.rest.positions) == (200 - odd, 200 - odd)
    assert random_ai.all.matches < 200
    assert random_ai.all.matches == random_ai.group.matches + random_ai.rest.matches


@pytest.mark.parametrize(("parallel", "remember"), [(1, False), (7, False), (7, True)])
def test_trials_played_together_equal_trials_played_one_by_one(
    solution: Solution, trials: tuple[Trial, ...], parallel: int, remember: bool
) -> None:
    game = solution.game
    chosen = trials[:20]
    network = PerBoardNetwork(game.action_space_size)
    search = MCTS(game, network, simulations=12)
    alone = play_trials(
        game,
        chosen,
        lambda index: MCTSAI(search, temperature=0.0, seed=index),
        lambda index: PerfectAI(solution, index),
    )
    asked = sum(network.sizes)
    network = PerBoardNetwork(game.action_space_size)
    cache = EvaluationCache(10_000) if remember else None

    together = play_trials_together(
        game,
        chosen,
        MCTS(game, network, simulations=12),
        lambda index: PerfectAI(solution, index),
        network=network,
        parallel=parallel,
        cache=cache,
    )

    assert together == alone
    if remember:
        assert sum(network.sizes) < asked
    else:
        assert sum(network.sizes) == asked
    assert max(network.sizes) <= parallel


def test_losing_trials_are_refused_when_played_together_too(
    solution: Solution, trials: tuple[Trial, ...]
) -> None:
    game = solution.game
    network = PerBoardNetwork(game.action_space_size)

    with pytest.raises(ValueError):
        play_trials_together(
            game,
            trials,
            MCTS(game, network, simulations=2),
            lambda index: RandomAI(game, index),
            network=network,
            parallel=4,
        )
