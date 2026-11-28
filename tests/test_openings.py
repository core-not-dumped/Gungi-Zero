"""풀지 못하는 게임의 시작 판."""

import pytest

from eval.openings import balanced_openings, forced_result
from games.tictactoe import TicTacToe

GAME = TicTacToe()


def after(*moves: str):
    position = GAME.initial_position()
    for move in moves:
        position = GAME.apply_move(position, GAME.parse_move(move))
    return position


def move_names() -> list[str]:
    return [GAME.render_move(move) for move in GAME.legal_moves(GAME.initial_position())]


def test_a_win_in_one_is_found_at_depth_one() -> None:
    a1, a2, _, b1, b2, *_ = move_names()
    position = after(a1, b1, a2, b2)

    assert forced_result(GAME, position, 1) == 1


def test_a_loss_that_cannot_be_avoided_needs_depth_two() -> None:
    names = move_names()
    # 선수가 두 줄을 동시에 위협한다. 후수는 하나만 막는다.
    position = after(names[0], names[1], names[4], names[8], names[3])

    assert forced_result(GAME, position, 1) == 0
    assert forced_result(GAME, position, 2) == -1


def test_the_start_has_no_forced_result() -> None:
    assert forced_result(GAME, GAME.initial_position(), 3) == 0


def test_openings_are_distinct_unfinished_and_not_decided_within_the_depth() -> None:
    openings = balanced_openings(GAME, 10, plies=(1, 4), depth=3, seed=0)

    assert len({GAME.board_key(position) for position in openings}) == 10
    assert all(GAME.outcome(position) is None for position in openings)
    assert all(1 <= position.ply <= 4 for position in openings)
    assert all(forced_result(GAME, position, 3) == 0 for position in openings)


def test_the_same_seed_gives_the_same_openings() -> None:
    first = balanced_openings(GAME, 5, plies=(2, 4), depth=2, seed=3)

    assert first == balanced_openings(GAME, 5, plies=(2, 4), depth=2, seed=3)


def test_more_openings_than_the_game_has_is_refused() -> None:
    # 첫 수는 아홉 가지뿐이다.
    with pytest.raises(ValueError, match="시작 판"):
        balanced_openings(GAME, 10, plies=(1, 1), depth=1, seed=0)
