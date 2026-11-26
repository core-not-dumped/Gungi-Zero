"""틱택토 fixture 테스트. 판은 항상 텍스트 렌더로 읽는다 (스펙의 테스트 결정)."""

import numpy as np
import pytest

from games.base import Outcome, Player
from games.tictactoe import TicTacToe, TicTacToePosition

INITIAL = """\
     a   b   c
 3   .   .   .
 2   .   .   .
 1   .   .   .
차례: ▲   ply: 0
"""


def test_initial_position_is_an_empty_board_with_the_first_player_to_move() -> None:
    game = TicTacToe()
    position = game.initial_position()
    assert game.render(position) == INITIAL
    assert position.player is Player.FIRST
    assert position.ply == 0


def test_parse_reads_the_rendered_format_back() -> None:
    game = TicTacToe()
    text = """\
     a   b   c
 3   .   ▽   .
 2   .   ▲   .
 1   ▲   .   ▽
차례: ▲   ply: 4
"""
    position = game.parse(text)
    assert game.render(position) == text
    assert position.player is Player.FIRST
    assert position.ply == 4


def notation(game: TicTacToe, position: TicTacToePosition) -> set[str]:
    return {game.render_move(move) for move in game.legal_moves(position)}


def test_every_empty_square_is_legal_on_the_initial_position() -> None:
    game = TicTacToe()
    assert notation(game, game.initial_position()) == {
        "a1",
        "b1",
        "c1",
        "a2",
        "b2",
        "c2",
        "a3",
        "b3",
        "c3",
    }


def test_applying_a_move_marks_the_square_and_passes_the_turn() -> None:
    game = TicTacToe()
    after = game.apply_move(game.initial_position(), game.parse_move("b2"))
    assert (
        game.render(after)
        == """\
     a   b   c
 3   .   .   .
 2   .   ▲   .
 1   .   .   .
차례: ▽   ply: 1
"""
    )


def test_an_occupied_square_is_not_a_legal_move() -> None:
    game = TicTacToe()
    after = game.apply_move(game.initial_position(), game.parse_move("b2"))
    assert "b2" not in notation(game, after)
    with pytest.raises(ValueError):
        game.apply_move(after, game.parse_move("b2"))


def test_three_in_a_row_wins_and_leaves_no_legal_moves() -> None:
    game = TicTacToe()
    position = game.parse("""\
     a   b   c
 3   ▽   ▽   .
 2   .   .   .
 1   ▲   ▲   ▲
차례: ▽   ply: 5
""")
    assert game.outcome(position) is Outcome.FIRST_WINS
    assert game.legal_moves(position) == ()


def test_a_full_board_without_a_line_is_a_draw() -> None:
    game = TicTacToe()
    position = game.parse("""\
     a   b   c
 3   ▲   ▽   ▲
 2   ▲   ▽   ▽
 1   ▽   ▲   ▲
차례: ▽   ply: 9
""")
    assert game.outcome(position) is Outcome.DRAW
    assert game.legal_moves(position) == ()


def test_an_unfinished_position_has_no_outcome() -> None:
    game = TicTacToe()
    assert game.outcome(game.initial_position()) is None


def test_the_second_player_can_win_too() -> None:
    game = TicTacToe()
    position = game.parse("""\
     a   b   c
 3   ▲   ▲   ▽
 2   .   ▽   ▲
 1   ▽   .   .
차례: ▲   ply: 6
""")
    assert game.outcome(position) is Outcome.SECOND_WINS


def test_the_canonical_form_swaps_the_colours_when_the_second_player_moves() -> None:
    game = TicTacToe()
    position = game.parse("""\
     a   b   c
 3   .   .   .
 2   .   ▲   .
 1   .   .   .
차례: ▽   ply: 1
""")
    # 정규형의 판은 실제 대국에 나올 수 없다 — 둘 차례인 쪽이 말이 하나 적다. 신경망에게는
    # "내 말 / 상대 말" 만 보이므로 문제가 되지 않는다.
    assert (
        game.render(game.canonical(position))
        == """\
     a   b   c
 3   .   .   .
 2   .   ▽   .
 1   .   .   .
차례: ▲   ply: 1
"""
    )


def test_the_canonical_form_leaves_a_first_player_position_alone() -> None:
    game = TicTacToe()
    position = game.parse("""\
     a   b   c
 3   .   .   .
 2   .   ▲   .
 1   .   .   .
차례: ▲   ply: 1
""")
    assert game.canonical(position) == position


def test_the_tensor_puts_the_movers_marks_on_the_first_plane() -> None:
    game = TicTacToe()
    position = game.parse("""\
     a   b   c
 3   .   .   .
 2   .   ▲   .
 1   ▽   .   .
차례: ▲   ply: 2
""")
    # 평면의 행 0 이 1 행(선수 진영)이다. 렌더는 1 행을 맨 아래에 찍으므로 위아래가 뒤집혀 보인다.
    assert np.array_equal(
        game.encode(position),
        np.array(
            [
                [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                [[1, 0, 0], [0, 0, 0], [0, 0, 0]],
            ],
            dtype=np.float32,
        ),
    )


def test_the_planes_hold_exactly_the_marks_on_the_board() -> None:
    game = TicTacToe()
    position = game.parse("""\
     a   b   c
 3   ▲   ▽   .
 2   .   ▲   .
 1   ▽   .   ▲
차례: ▽   ply: 5
""")
    assert game.encode(position).sum() == 5


def test_different_move_orders_to_the_same_board_share_a_board_key() -> None:
    game = TicTacToe()
    one = game.initial_position()
    other = game.initial_position()
    for move in ("a1", "b2", "c3"):
        one = game.apply_move(one, game.parse_move(move))
    for move in ("c3", "b2", "a1"):
        other = game.apply_move(other, game.parse_move(move))
    assert game.render(one) == game.render(other)
    assert game.board_key(one) == game.board_key(other)
