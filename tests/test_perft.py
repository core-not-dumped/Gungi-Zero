"""perft. 초기 국면에서 깊이별 노드 수를 고정값으로 박는다.

군의의 값은 정확성 증명이 아니라 회귀 테스트다. 합법수 계산을 바꾸지 않았다면 그대로여야 한다.
"""

import pytest

from games.mini_gungi import MINI_GUNGI, Gungi
from games.tictactoe import TicTacToe
from scripts.perft import perft


@pytest.mark.parametrize(
    ("depth", "nodes"), [(0, 1), (1, 9), (2, 72), (3, 504), (4, 3024), (5, 15120), (6, 54720)]
)
def test_perft_matches_the_known_tictactoe_counts(depth: int, nodes: int) -> None:
    """남이 아는 값으로 perft 자체를 확인한다. 깊이 6 부터는 끝난 판이 가지를 줄인다."""
    game = TicTacToe()
    assert perft(game, game.initial_position(), depth) == nodes


@pytest.mark.parametrize(("depth", "nodes"), [(0, 1), (1, 12), (2, 146), (3, 2118), (4, 30478)])
def test_perft_of_the_initial_mini_gungi_position(depth: int, nodes: int) -> None:
    game = Gungi(MINI_GUNGI)
    assert perft(game, game.initial_position(), depth) == nodes


def test_a_negative_depth_is_refused() -> None:
    game = TicTacToe()
    with pytest.raises(ValueError):
        perft(game, game.initial_position(), -1)
