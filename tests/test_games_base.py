"""`games.base` 자체의 작은 규칙들과, 틱택토가 프로토콜에 구조적으로 맞는지의 타입 수준 확인."""

import sys

import pytest

from games.base import Game, Outcome, Player
from games.tictactoe import TicTacToe, TicTacToeMove, TicTacToePosition


def test_tictactoe_satisfies_the_game_protocol() -> None:
    """pyright 가 검사하는 쪽이 본문이다 — `Game` 은 구조적 프로토콜이라 상속이 없다."""
    game: Game[TicTacToePosition, TicTacToeMove] = TicTacToe()
    assert game.action_space_size == 9


@pytest.mark.parametrize("player", list(Player))
def test_the_opponent_of_the_opponent_is_the_player(player: Player) -> None:
    assert player.opponent.opponent is player
    assert player.opponent is not player


@pytest.mark.parametrize("player", list(Player))
def test_the_winner_of_a_win_is_the_player_it_was_made_for(player: Player) -> None:
    assert Outcome.win_for(player).winner is player
    assert Outcome.win_for(player) is not Outcome.win_for(player.opponent)


def test_a_draw_has_no_winner() -> None:
    assert Outcome.DRAW.winner is None


def test_the_game_protocol_has_no_undo() -> None:
    """`Position` 이 불변이라 되돌리기가 필요 없다. 가변 구현은 별개 클래스로 온다 (CLAUDE.md)."""
    assert not hasattr(Game, "undo")


def test_the_game_holds_no_state() -> None:
    game = TicTacToe()
    game.legal_moves(game.initial_position())
    assert vars(game) == {}


def test_nothing_in_games_or_ai_pulls_in_torch() -> None:
    """경계를 넘는 표현은 ndarray 다 (ADR-0001). import 그래프 전체 검사는 05 번 티켓이다."""
    import ai.random_play  # noqa: F401
    import games.tictactoe  # noqa: F401

    assert "torch" not in sys.modules
