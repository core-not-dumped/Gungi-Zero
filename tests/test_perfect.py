"""완벽한 상대. 어떤 수를 골랐는지와 대국 결과만 본다."""

import pytest

from ai.episode import play_episode
from ai.random_play import RandomAI
from eval.perfect import PerfectAI, best_moves
from eval.solver import Solution, Value, Verdict, solve
from games.base import Game, Outcome, Player
from games.mini_gungi import Gungi
from games.tictactoe import TicTacToe
from tests.solver_helpers import SMALL_NANO, shortest_path_positions, winner_of, with_ply

SEEDS = range(100)


@pytest.fixture(scope="module")
def tictactoe() -> Solution:
    return solve(TicTacToe(), move_limit=None)


@pytest.fixture(scope="module")
def small_nano() -> Solution:
    return solve(Gungi(SMALL_NANO), move_limit=SMALL_NANO.move_limit)


def expected_after_a_best_move(value: Value) -> Value:
    """최선의 수를 둔 다음 국면의 값, 상대 시점."""
    if value.dtm is None:
        return value
    opposite = Verdict.LOSS if value.verdict is Verdict.WIN else Verdict.WIN
    return Value(opposite, value.dtm - 1)


def assert_plays_best(solution: Solution, positions: list, seeds: range) -> set[Verdict]:
    """모든 진행 중인 국면에서 최선의 수를 두는지 보고, 지나간 이론값의 종류를 돌려준다."""
    game: Game = solution.game
    ais = [PerfectAI(solution, seed) for seed in seeds]
    seen: set[Verdict] = set()
    for position in positions:
        if game.outcome(position) is not None:
            continue
        value = solution.value(position)
        seen.add(value.verdict)
        for ai in ais:
            child = game.apply_move(position, ai.choose(position))
            assert solution.value(child) == expected_after_a_best_move(value), game.render(position)
    return seen


def test_it_wins_fastest_holds_the_draw_and_loses_slowest(
    tictactoe: Solution, small_nano: Solution
) -> None:
    seen = assert_plays_best(tictactoe, shortest_path_positions(tictactoe.game), range(3))
    assert seen == set(Verdict)
    assert_plays_best(small_nano, shortest_path_positions(small_nano.game), range(3))


# 한 수가 남으면 진행 중인 국면의 패배는 이루어질 수 없다 — 지는 데에는 두 수가 든다.
@pytest.mark.parametrize(
    ("plies_left", "verdicts"),
    [(1, {Verdict.WIN, Verdict.DRAW}), (3, set(Verdict)), (6, set(Verdict))],
)
def test_it_plays_by_the_values_the_move_limit_leaves(
    small_nano: Solution, plies_left: int, verdicts: set[Verdict]
) -> None:
    game = small_nano.game
    assert isinstance(game, Gungi)
    ply = SMALL_NANO.move_limit - plies_left
    positions = [with_ply(game, p, ply) for p in shortest_path_positions(game)]
    assert assert_plays_best(small_nano, positions, range(3)) == verdicts


def test_the_seed_picks_among_the_best_moves(tictactoe: Solution) -> None:
    # 틱택토의 첫 수는 아홉 개가 모두 비긴다.
    start = tictactoe.game.initial_position()
    assert len({PerfectAI(tictactoe, seed).choose(start) for seed in SEEDS}) == 9
    one, other = PerfectAI(tictactoe, seed=42), PerfectAI(tictactoe, seed=42)
    assert [one.choose(start) for _ in range(5)] == [other.choose(start) for _ in range(5)]


def test_it_refuses_a_finished_position(tictactoe: Solution) -> None:
    game = tictactoe.game
    finished = play_episode(game, RandomAI(game, 0), RandomAI(game, 1)).final_position
    with pytest.raises(ValueError):
        PerfectAI(tictactoe, seed=0).choose(finished)


def test_perfect_tictactoe_is_always_a_draw(tictactoe: Solution) -> None:
    game = tictactoe.game
    episodes = set()
    for seed in SEEDS:
        ai = PerfectAI(tictactoe, seed)
        episode = play_episode(game, ai, ai)
        assert episode.outcome is Outcome.DRAW
        episodes.add(episode.moves)
    assert len(episodes) > 1


@pytest.mark.parametrize("seat", Player)
def test_it_never_loses_to_the_random_ai(tictactoe: Solution, seat: Player) -> None:
    game = tictactoe.game
    for seed in SEEDS:
        perfect, other = PerfectAI(tictactoe, seed), RandomAI(game, seed)
        first, second = (perfect, other) if seat is Player.FIRST else (other, perfect)
        assert play_episode(game, first, second).outcome.winner is not seat.opponent


def test_perfect_play_ends_a_won_position_in_exactly_dtm_plies(small_nano: Solution) -> None:
    """반복을 보지 않고 푼 승패가 실제 대국에서 그대로 난다."""
    game = small_nano.game
    ai = PerfectAI(small_nano, seed=0)
    for start in shortest_path_positions(game):
        value = small_nano.value(start)
        if value.dtm is None:
            continue
        position = start
        for _ in range(value.dtm):
            assert game.outcome(position) is None
            position = game.apply_move(position, ai.choose(position))
        winner = winner_of(start.player, value.verdict)
        outcome = game.outcome(position)
        assert outcome is not None and outcome.winner is winner, game.render(start)


def test_the_best_moves_are_the_ones_that_keep_the_value_by_the_shortest_way(
    tictactoe: Solution,
) -> None:
    game = tictactoe.game
    start = game.initial_position()
    # 틱택토의 첫 수는 모두 비긴다.
    assert set(best_moves(tictactoe, start)) == set(game.legal_moves(start))
    for position in shortest_path_positions(game)[:300]:
        if game.outcome(position) is not None:
            continue
        expected = expected_after_a_best_move(tictactoe.value(position))
        assert set(best_moves(tictactoe, position)) == {
            move
            for move in game.legal_moves(position)
            if tictactoe.value(game.apply_move(position, move)) == expected
        }
