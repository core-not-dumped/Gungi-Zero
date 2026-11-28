"""랜덤 AI 와 랜덤 대국. 랜덤 AI 는 Elo 0 의 기준점이다 (`CONTEXT.md`)."""

from collections import Counter

import pytest

from ai.random_play import MixedAI, RandomAI, play_random_episode
from games.base import Outcome
from games.tictactoe import TicTacToe


def test_the_random_ai_only_ever_picks_a_legal_move() -> None:
    game = TicTacToe()
    player = RandomAI(game, seed=7)
    position = game.initial_position()
    while game.outcome(position) is None:
        move = player.choose(position)
        assert move in game.legal_moves(position)
        position = game.apply_move(position, move)


def test_the_random_ai_refuses_a_finished_position() -> None:
    game = TicTacToe()
    finished = play_random_episode(game, seed=3).final_position
    with pytest.raises(ValueError):
        RandomAI(game, seed=0).choose(finished)


def test_the_same_seed_picks_the_same_stream_of_moves() -> None:
    game = TicTacToe()
    position = game.initial_position()
    one, other = RandomAI(game, seed=42), RandomAI(game, seed=42)
    assert [one.choose(position) for _ in range(5)] == [other.choose(position) for _ in range(5)]


def test_the_random_ai_spreads_uniformly_over_the_legal_moves() -> None:
    game = TicTacToe()
    position = game.initial_position()
    player = RandomAI(game, seed=1)
    counts = Counter(game.render_move(player.choose(position)) for _ in range(9000))
    assert len(counts) == 9
    assert all(800 <= count <= 1200 for count in counts.values()), counts


def test_a_random_episode_runs_to_a_result() -> None:
    game = TicTacToe()
    episode = play_random_episode(game, seed=3)
    assert isinstance(episode.outcome, Outcome)
    assert game.outcome(episode.final_position) is episode.outcome
    assert len(episode.moves) == episode.final_position.ply


def test_replaying_an_episodes_moves_reaches_its_final_position() -> None:
    game = TicTacToe()
    episode = play_random_episode(game, seed=5)
    position = game.initial_position()
    for move in episode.moves:
        assert move in game.legal_moves(position)
        position = game.apply_move(position, move)
    assert position == episode.final_position


def test_the_same_seed_replays_the_same_episode() -> None:
    game = TicTacToe()
    assert play_random_episode(game, seed=11) == play_random_episode(game, seed=11)


class First:
    """합법수 가운데 첫 번째만 둔다."""

    def choose(self, position):
        return TicTacToe().legal_moves(position)[0]


@pytest.mark.parametrize(("probability", "low", "high"), [(0.0, 9000, 9000), (0.5, 4800, 5200)])
def test_a_mixed_ai_plays_a_random_move_with_the_given_probability(
    probability: float, low: int, high: int
) -> None:
    game = TicTacToe()
    position = game.initial_position()
    ai = MixedAI(game, First(), probability, seed=2)

    counts = Counter(ai.choose(position) for _ in range(9000))

    # 섞은 무작위 수도 1/9 은 첫 수다: 0.5 면 첫 수가 9분의 5.
    assert low <= counts[game.legal_moves(position)[0]] <= high


def test_a_mixed_ai_with_probability_one_never_asks_the_inner_ai() -> None:
    class Never:
        def choose(self, position):
            raise AssertionError("묻지 않는다")

    game = TicTacToe()
    ai = MixedAI(game, Never(), 1.0, seed=0)

    assert ai.choose(game.initial_position()) in game.legal_moves(game.initial_position())


@pytest.mark.parametrize("probability", [-0.1, 1.1])
def test_a_probability_outside_zero_to_one_is_refused(probability: float) -> None:
    with pytest.raises(ValueError):
        MixedAI(TicTacToe(), First(), probability, seed=0)
