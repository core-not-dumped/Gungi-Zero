"""Elo 를 매길 대진과 대국."""

from ai.random_play import RandomAI
from eval.elo import PairScore
from eval.ladder import MemoizedAI, ladder_pairs, play_pair
from games.base import Player
from games.tictactoe import TicTacToe

GAME = TicTacToe()


def openings_after(count: int, plies: int) -> list:
    """초기 국면에서 서로 다른 첫 수로 시작해 `plies` 수를 둔 판들."""
    openings = []
    for first in GAME.legal_moves(GAME.initial_position())[:count]:
        position = GAME.apply_move(GAME.initial_position(), first)
        for _ in range(plies - 1):
            position = GAME.apply_move(position, GAME.legal_moves(position)[0])
        openings.append(position)
    return openings


def test_the_ladder_joins_neighbours_and_ties_some_levels_to_the_anchor() -> None:
    pairs = ladder_pairs(6, gaps=(1, 4), anchored=(1, 5, 6, 9))

    assert pairs == [
        (1, 0),
        (5, 0),
        (6, 0),
        (2, 1),
        (3, 2),
        (4, 3),
        (5, 4),
        (6, 5),
        (5, 1),
        (6, 2),
    ]


class Recorder(RandomAI):
    def __init__(self, seed: int, name: str, seen: list) -> None:
        super().__init__(GAME, seed)
        self._name, self._seen = name, seen

    def choose(self, position):
        self._seen.append((self._name, position.ply, position.player))
        return super().choose(position)


def test_each_opening_is_played_twice_with_the_sides_swapped() -> None:
    openings = openings_after(3, 1)
    seen: list = []
    asked: list[int] = []

    def player(index: int):
        asked.append(index)
        return Recorder(index, "player", seen)

    score = play_pair(
        GAME, ("p", player), ("o", lambda index: Recorder(index, "opponent", seen)), openings
    )

    assert isinstance(score, PairScore)
    assert (score.player, score.opponent) == ("p", "o")
    assert score.wins + score.draws + score.losses == 6
    assert asked == list(range(6))
    # 시작 판(1수 뒤, 후수 차례)에서 먼저 두는 쪽이 판마다 바뀐다.
    firsts = [name for name, ply, _ in seen if ply == 1]
    assert firsts == ["player", "opponent"] * 3
    assert {side for _, ply, side in seen if ply == 1} == {Player.SECOND}


def test_the_score_is_counted_for_the_player_whichever_side_it_takes() -> None:
    openings = openings_after(4, 2)

    score = play_pair(
        GAME,
        ("p", lambda index: RandomAI(GAME, 7)),
        ("o", lambda index: RandomAI(GAME, 7)),
        openings,
    )

    # 같은 AI 둘이 선후를 바꿔 같은 판을 두므로 승과 패가 같다.
    assert score.wins == score.losses


class Counting:
    def __init__(self) -> None:
        self.asked = 0
        self._ai = RandomAI(GAME, 0)

    def choose(self, position):
        self.asked += 1
        return self._ai.choose(position)


def test_a_memoized_ai_asks_once_per_position() -> None:
    inner = Counting()
    ai = MemoizedAI(inner)
    start = GAME.initial_position()

    moves = {ai.choose(start) for _ in range(5)}

    assert len(moves) == 1
    assert inner.asked == 1
