"""자가 대국. 돌려준 `Episode` 와 학습 샘플만 본다."""

from collections.abc import Iterator
from dataclasses import replace
from functools import cache
from typing import Any

import numpy as np
import pytest

from ai.mcts import RootNoise
from ai.self_play import (
    SelfPlayConfig,
    SelfPlayResult,
    TrainingSample,
    play_self_play_episode,
)
from games.base import Game, Outcome, Player
from games.mini_gungi import EndReason, Gungi
from games.nano_gungi import NANO_GUNGI
from games.tictactoe import TicTacToe
from scripts.hyperparameters import NANO_GUNGI_SELF_PLAY, TICTACTOE_SELF_PLAY
from tests.stub_networks import UniformNetwork

# 탐색 횟수만 줄여 빠르게 돈다.
TICTACTOE = replace(TICTACTOE_SELF_PLAY, simulations=16)
NANO = replace(NANO_GUNGI_SELF_PLAY, simulations=16)


def play(game: Game, config: SelfPlayConfig, seed: int) -> SelfPlayResult:
    network = UniformNetwork(game.tensor_shape, game.action_space_size)
    return play_self_play_episode(game, network, config, seed=seed)


@cache
def nano_results() -> tuple[tuple[Gungi, SelfPlayResult], ...]:
    """시드 0~29 의 나노 자가 대국. 여러 테스트가 같은 대국을 읽는다."""
    game = Gungi(NANO_GUNGI)
    return tuple((game, play(game, NANO, seed)) for seed in range(30))


def steps_of(game: Game, result: SelfPlayResult) -> Iterator[tuple[Any, Any, TrainingSample]]:
    """대국을 다시 두며 수를 둔 국면, 그 수, 그 학습 샘플을 낸다."""
    position = game.initial_position()
    for move, sample in zip(result.episode.moves, result.samples, strict=True):
        yield position, move, sample
        position = game.apply_move(position, move)
    assert position == result.episode.final_position


def canonical_index(game: Game, position: Any, move: Any) -> int:
    return game.to_canonical_action_index(position, game.action_index(move))


def test_every_move_leaves_a_sample_of_the_position_it_was_played_from() -> None:
    for game, result in nano_results():
        assert game.outcome(result.episode.final_position) is result.episode.outcome
        # 한 수에 학습 샘플 하나다. 거울 증강은 없다 (ADR-0005).
        for position, _, sample in steps_of(game, result):
            canonical = game.canonical(position)
            assert sample.board.dtype == np.float32
            np.testing.assert_array_equal(sample.board, game.encode(canonical))
            assert sample.legal_mask.dtype == bool
            np.testing.assert_array_equal(sample.legal_mask, game.legal_action_mask(canonical))
            assert sample.policy.dtype == np.float32
            assert sample.policy.shape == (game.action_space_size,)
            assert sample.policy.sum() == pytest.approx(1.0)


def test_the_policy_is_in_the_canonical_frame() -> None:
    would_catch_the_original_frame = 0
    for game, result in nano_results():
        for position, _, sample in steps_of(game, result):
            if position.player is Player.FIRST:
                continue
            assert not sample.policy[~sample.legal_mask].any()
            # 같은 분포를 원래 프레임의 인덱스로 담았다면 위의 검사에 걸리는 국면인지 본다.
            visited = [
                game.action_index(move)
                for move in game.legal_moves(position)
                if sample.policy[canonical_index(game, position, move)] > 0
            ]
            would_catch_the_original_frame += not sample.legal_mask[visited].all()
    assert would_catch_the_original_frame > 0


def check_value_targets(game: Game, result: SelfPlayResult) -> None:
    winner = result.episode.outcome.winner
    for position, _, sample in steps_of(game, result):
        if winner is None:
            assert sample.value == 0.0
        else:
            assert sample.value == (1.0 if position.player is winner else -1.0)


@pytest.mark.parametrize("outcome", Outcome, ids=lambda o: o.name)
def test_the_value_target_is_the_result_seen_by_the_side_to_move(outcome: Outcome) -> None:
    game = TicTacToe()
    results = (play(game, TICTACTOE, seed) for seed in range(100))
    result = next(r for r in results if r.episode.outcome is outcome)

    if outcome is Outcome.FIRST_WINS:
        assert [s.value for s in result.samples[:2]] == [1.0, -1.0]
    check_value_targets(game, result)


@pytest.mark.parametrize(
    ("move_limit", "simulations", "reason"),
    [
        # 탐색이 얕으면 반복에 이르기 전에 帥 가 잡힌다.
        (NANO_GUNGI.move_limit, 50, EndReason.REPETITION),
        (2, NANO.simulations, EndReason.MOVE_LIMIT),
    ],
    ids=["repetition", "move-limit"],
)
def test_draws_by_the_path_give_zero_targets(
    move_limit: int, simulations: int, reason: EndReason
) -> None:
    game = Gungi(replace(NANO_GUNGI, move_limit=move_limit))
    config = replace(NANO, simulations=simulations)
    results = (play(game, config, seed) for seed in range(100))
    result = next(r for r in results if game.end_reason(r.episode.final_position) is reason)

    assert result.samples
    check_value_targets(game, result)


def played_a_most_visited_move(exploration: bool) -> list[bool]:
    """탐색 구간 안(또는 밖)의 수마다, 가장 많이 방문한 수를 두었는지."""
    answers = []
    for game, result in nano_results():
        for number, (position, move, sample) in enumerate(steps_of(game, result)):
            if (number < NANO.exploration_moves) is exploration:
                chosen = sample.policy[canonical_index(game, position, move)]
                answers.append(bool(chosen == sample.policy.max()))
    return answers


def test_the_first_moves_are_drawn_in_proportion_to_the_visits() -> None:
    assert not all(played_a_most_visited_move(exploration=True))


def test_the_later_moves_are_the_most_visited_ones() -> None:
    answers = played_a_most_visited_move(exploration=False)
    assert answers
    assert all(answers)


def test_the_exploration_ends_after_the_configured_number_of_moves() -> None:
    """마지막 탐색 수까지 비례로 고른다. 하나라도 일찍 끊으면 그 수는 늘 최다 방문이다."""
    last = NANO.exploration_moves - 1
    off_the_top = 0
    for game, result in nano_results():
        steps = list(steps_of(game, result))
        if len(steps) > last:
            position, move, sample = steps[last]
            chosen = sample.policy[canonical_index(game, position, move)]
            off_the_top += bool(chosen < sample.policy.max())
    assert off_the_top > 0


def first_policies(config: SelfPlayConfig) -> list[bytes]:
    """시드 0~9 의 첫 방문 분포."""
    game = TicTacToe()
    return [play(game, config, seed).samples[0].policy.tobytes() for seed in range(10)]


def with_noise(alpha: float = 1.8, epsilon: float = 0.25) -> SelfPlayConfig:
    return replace(TICTACTOE, root_noise=RootNoise(alpha, epsilon))


def test_the_search_uses_root_noise() -> None:
    # 균등 신경망의 탐색은 노이즈가 없으면 시드와 무관하다.
    assert len(set(first_policies(TICTACTOE))) > 1
    assert len(set(first_policies(with_noise(epsilon=0.0)))) == 1


def all_policies(config: SelfPlayConfig) -> list[bytes]:
    """시드 0~9 의 대국에 나온 모든 방문 분포."""
    game = TicTacToe()
    return [s.policy.tobytes() for seed in range(10) for s in play(game, config, seed).samples]


@pytest.mark.parametrize(
    "changed",
    [with_noise(alpha=0.03), replace(TICTACTOE, c_puct=0.01)],
    ids=["dirichlet-alpha", "c-puct"],
)
def test_the_search_takes_its_settings_from_the_config(changed: SelfPlayConfig) -> None:
    # c_puct 는 0 이 아닌 가치(끝난 국면)에 닿아야 분포를 바꾼다. 그래서 대국 전체를 본다.
    assert all_policies(changed) != all_policies(TICTACTOE)


def test_the_same_seed_gives_the_same_episode_and_samples() -> None:
    game = Gungi(NANO_GUNGI)
    first, second = play(game, NANO, seed=7), play(game, NANO, seed=7)

    assert first.episode == second.episode
    for one, other in zip(first.samples, second.samples, strict=True):
        np.testing.assert_array_equal(one.board, other.board)
        np.testing.assert_array_equal(one.legal_mask, other.legal_mask)
        np.testing.assert_array_equal(one.policy, other.policy)
        assert one.value == other.value
