"""나노 군의 시작 국면. 지지 않는 첫 수는 하나뿐이다. solver 없이 실제 규칙의 탐색으로 확인한다."""

from games.base import Player
from games.mini_gungi import Gungi
from games.nano_gungi import NANO_GUNGI
from tests.solver_helpers import forces_a_win

GAME = Gungi(NANO_GUNGI)
# 첫 수 → 후수가 이기는 데 드는 수 (docs/nano-solution.md). 표에 없는 첫 수는 `c2-c1^` 하나다.
LOSING_FIRST_MOVES = {"b1-b2": 1, "b1-a1": 1, "b1-a2": 1, "c2-b2": 3, "c2-a2": 3, "c2-b3": 5}


def after(move: str):
    return GAME.apply_move(GAME.initial_position(), GAME.parse_move(move))


def test_there_are_ten_first_moves() -> None:
    names = [GAME.render_move(m) for m in GAME.legal_moves(GAME.initial_position())]

    slower = {"c1-c2^", "c2-c3", "c2-c4"}
    assert set(names) == set(LOSING_FIRST_MOVES) | slower | {"c2-c1^"}


def test_the_short_losses_are_forced_in_exactly_that_many_plies() -> None:
    for move, plies in LOSING_FIRST_MOVES.items():
        position = after(move)
        assert forces_a_win(GAME, position, Player.SECOND, plies), move
        assert not forces_a_win(GAME, position, Player.SECOND, plies - 1), move


def test_the_drawing_move_is_not_lost_within_five_plies() -> None:
    assert not forces_a_win(GAME, after("c2-c1^"), Player.SECOND, 5)
