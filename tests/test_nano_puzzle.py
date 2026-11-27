"""나노 군의 시작 국면의 3수 강제승. solver 없이 실제 규칙의 탐색으로 확인한다."""

from games.base import Player
from games.mini_gungi import Gungi
from games.nano_gungi import NANO_GUNGI
from tests.solver_helpers import forces_a_win

GAME = Gungi(NANO_GUNGI)


def test_the_first_player_forces_a_win_in_three_plies_and_not_in_two() -> None:
    start = GAME.initial_position()
    assert forces_a_win(GAME, start, Player.FIRST, 3)
    assert not forces_a_win(GAME, start, Player.FIRST, 2)


def test_only_stacking_the_general_on_the_pawn_wins_in_three() -> None:
    start = GAME.initial_position()
    winning = [
        GAME.render_move(move)
        for move in GAME.legal_moves(start)
        if forces_a_win(GAME, GAME.apply_move(start, move), Player.FIRST, 2)
    ]
    assert winning == ["a1-a3^"]


def test_every_reply_to_it_leaves_the_marshal_to_be_taken() -> None:
    position = GAME.apply_move(GAME.initial_position(), GAME.parse_move("a1-a3^"))
    replies = GAME.legal_moves(position)
    assert len(replies) == 7
    for reply in replies:
        assert forces_a_win(GAME, GAME.apply_move(position, reply), Player.FIRST, 1)
