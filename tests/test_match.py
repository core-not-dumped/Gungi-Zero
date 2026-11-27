"""대국 harness. 넘긴 두 AI 가 어느 자리에 앉았고 결과를 어떻게 셌는지만 본다."""

import pytest

from ai.episode import play_episode
from ai.random_play import RandomAI
from eval.match import Record, play_match
from games.base import Outcome
from games.tictactoe import TicTacToe, TicTacToeMove, TicTacToePosition


class ScriptedAI:
    """남은 칸 중 `order` 에서 가장 앞선 칸에 둔다."""

    def __init__(self, order: str) -> None:
        game = TicTacToe()
        self._moves = [game.parse_move(text) for text in order.split()]

    def choose(self, position: TicTacToePosition) -> TicTacToeMove:
        return next(move for move in self._moves if position.board[move.square] is None)


# 아래 줄부터 채우는 AI 와 위 줄부터 채우는 AI. 둘이 두면 선수가 먼저 줄을 완성한다.
def bottom_up(seed: int) -> ScriptedAI:
    return ScriptedAI("a1 b1 c1 a2 b2 c2 a3 b3 c3")


def top_down(seed: int) -> ScriptedAI:
    return ScriptedAI("a3 b3 c3 a2 b2 c2 a1 b1 c1")


def test_the_challenger_takes_each_seat_for_every_seed() -> None:
    game = TicTacToe()

    result = play_match(game, bottom_up, top_down, seeds=range(3))

    # 선후를 바꾸면 이기는 쪽도 바뀌지만, 먼저 두는 쪽이 이기므로 재는 AI 의 성적은 갈린다.
    assert result.as_first == Record(wins=3, draws=0, losses=0)
    assert result.as_second == Record(wins=0, draws=0, losses=3)
    # 결정적인 두 AI 는 시드가 달라도 같은 판을 둔다.
    assert (result.distinct_as_first, result.distinct_as_second) == (1, 1)
    assert [e.outcome for e in result.first_episodes] == [Outcome.FIRST_WINS] * 3
    assert [e.outcome for e in result.second_episodes] == [Outcome.FIRST_WINS] * 3
    assert game.render_move(result.first_episodes[0].moves[0]) == "a1"
    assert game.render_move(result.second_episodes[0].moves[0]) == "a3"


def test_each_game_is_built_from_its_seed() -> None:
    game = TicTacToe()

    def random_ai(seed: int) -> RandomAI:
        return RandomAI(game, seed)

    def other_random_ai(seed: int) -> RandomAI:
        return RandomAI(game, seed + 1000)

    seeds = range(5, 45)
    result = play_match(game, random_ai, other_random_ai, seeds=seeds)

    for seed, first, second in zip(
        seeds, result.first_episodes, result.second_episodes, strict=True
    ):
        assert first == play_episode(game, random_ai(seed), other_random_ai(seed))
        assert second == play_episode(game, other_random_ai(seed), random_ai(seed))
    for record, episodes, win in (
        (result.as_first, result.first_episodes, Outcome.FIRST_WINS),
        (result.as_second, result.second_episodes, Outcome.SECOND_WINS),
    ):
        outcomes = [episode.outcome for episode in episodes]
        assert record.wins == outcomes.count(win)
        assert record.draws == outcomes.count(Outcome.DRAW)
        assert record.wins + record.draws + record.losses == len(seeds)
        assert record.wins and record.draws and record.losses
    assert result.distinct_as_first == len({e.moves for e in result.first_episodes}) > 1
    assert result.distinct_as_second == len({e.moves for e in result.second_episodes}) > 1


def test_a_match_without_games_is_rejected() -> None:
    with pytest.raises(ValueError):
        play_match(TicTacToe(), bottom_up, top_down, seeds=range(0))
