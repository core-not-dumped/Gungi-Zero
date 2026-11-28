"""여러 판을 번갈아 진행하며 신경망 평가를 묶는 구동기와 그것을 쓰는 자가 대국."""

from collections.abc import Generator
from dataclasses import replace

import numpy as np
import pytest
import torch

from ai.batched import EvaluationCache, run_batched
from ai.self_play import play_self_play_episode, play_self_play_episodes
from games.mini_gungi import Gungi
from games.nano_gungi import NANO_GUNGI
from games.tictactoe import TicTacToe
from scripts.shared.hyperparameters import NANO_GUNGI_SELF_PLAY, TICTACTOE_SELF_PLAY


class PerBoardNetwork:
    """판마다 따로 정해지는 출력. 묶는 크기와 상관없이 같은 판에는 정확히 같은 답이다."""

    def __init__(self, action_space_size: int) -> None:
        self._ramp = torch.arange(action_space_size) % 7
        self.sizes: list[int] = []

    def __call__(self, boards: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.sizes.append(len(boards))
        weights = torch.arange(boards[0].numel()) % 11
        mark = (boards.flatten(1) * weights).sum(dim=1)
        logits = ((mark[:, None] + self._ramp) % 5).float()
        return logits, ((mark % 9) - 4) / 8


def echo(count: int, first: float) -> Generator[np.ndarray, tuple[np.ndarray, float], list[float]]:
    """`first` 부터 차례로 `count` 번 묻고 받은 가치를 모아 돌려준다."""
    answers = []
    for offset in range(count):
        _, value = yield np.full((1, 1, 1), first + offset, dtype=np.float32)
        answers.append(value)
    return answers


class EchoNetwork:
    def __init__(self) -> None:
        self.sizes: list[int] = []

    def __call__(self, boards: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.sizes.append(len(boards))
        return torch.zeros(len(boards), 1), boards.flatten()


@pytest.mark.parametrize("width", [1, 2, 3, 10])
def test_each_task_gets_its_own_answers_and_results_keep_the_order(width: int) -> None:
    network = EchoNetwork()
    lengths = [3, 1, 0, 5, 2]

    results = run_batched(
        (echo(length, 100 * index) for index, length in enumerate(lengths)),
        network,
        width=width,
    )

    assert results == [[100 * i + o for o in range(n)] for i, n in enumerate(lengths)]
    assert sum(network.sizes) == sum(lengths)
    assert max(network.sizes) <= width


def test_a_finished_task_is_replaced_before_the_next_evaluation() -> None:
    network = EchoNetwork()

    run_batched((echo(n, 0) for n in (1, 4, 0, 2, 2)), network, width=2)

    # 평가 9번을 두 자리에서 빈칸 없이 돌리면 마지막 하나만 혼자다.
    assert network.sizes == [2, 2, 2, 2, 1]


def test_a_task_is_made_only_when_a_seat_opens() -> None:
    made: list[int] = []

    def tasks():
        for index in range(3):
            made.append(index)
            yield echo(2, 0)

    class Watching(EchoNetwork):
        def __call__(self, boards):
            self.seen = [*getattr(self, "seen", []), len(made)]
            return super().__call__(boards)

    network = Watching()
    run_batched(tasks(), network, width=1)

    assert network.seen == [1, 1, 2, 2, 3, 3]


def test_a_width_below_one_is_refused() -> None:
    with pytest.raises(ValueError):
        run_batched(iter(()), EchoNetwork(), width=0)


def assert_same(one, other) -> None:
    assert one.episode == other.episode
    for a, b in zip(one.samples, other.samples, strict=True):
        np.testing.assert_array_equal(a.board, b.board)
        np.testing.assert_array_equal(a.legal_mask, b.legal_mask)
        np.testing.assert_array_equal(a.policy, b.policy)
        assert a.value == b.value


@pytest.mark.parametrize(
    ("game", "config"),
    [
        (TicTacToe(), replace(TICTACTOE_SELF_PLAY, simulations=12)),
        (Gungi(NANO_GUNGI), replace(NANO_GUNGI_SELF_PLAY, simulations=12)),
    ],
    ids=["tictactoe", "nano"],
)
@pytest.mark.parametrize("parallel", [1, 4])
def test_games_played_together_equal_games_played_one_by_one(game, config, parallel) -> None:
    start = game.apply_move(game.initial_position(), game.legal_moves(game.initial_position())[0])
    starts = [None, start, None, start, None, None]
    seeds = [11, 12, 13, 14, 15, 16]
    network = PerBoardNetwork(game.action_space_size)
    alone = [
        play_self_play_episode(game, network, config, seed=seed, start=position)
        for seed, position in zip(seeds, starts, strict=True)
    ]
    asked = sum(network.sizes)

    network = PerBoardNetwork(game.action_space_size)
    together = play_self_play_episodes(
        game, network, config, list(zip(seeds, starts, strict=True)), parallel=parallel
    )

    for one, other in zip(alone, together, strict=True):
        assert_same(one, other)
    # 묻는 횟수는 같고, 부르는 횟수만 줄어든다.
    assert sum(network.sizes) == asked
    assert max(network.sizes) == parallel


@pytest.mark.parametrize("width", [1, 3])
def test_a_remembered_board_is_not_evaluated_again(width: int) -> None:
    network = EchoNetwork()
    cache = EvaluationCache(100)
    # 세 작업이 0, 1, 2, 3 을 똑같이 묻는다.
    results = run_batched((echo(4, 0) for _ in range(3)), network, width=width, cache=cache)

    assert results == [[0, 1, 2, 3]] * 3
    assert sum(network.sizes) == 4
    assert (cache.asked, cache.evaluated) == (12, 4)


def test_the_cache_forgets_the_board_used_longest_ago() -> None:
    network = EchoNetwork()
    cache = EvaluationCache(2)
    boards = [5, 6, 5, 7, 5, 6]

    run_batched((echo(1, board) for board in boards), network, width=1, cache=cache)

    # 7 이 들어올 때 6 을 잊는다. 5 는 방금 썼다.
    assert sum(network.sizes) == 4


def test_a_cache_that_holds_nothing_is_refused() -> None:
    with pytest.raises(ValueError):
        EvaluationCache(0)


@pytest.mark.parametrize(
    ("game", "config"),
    [
        (TicTacToe(), replace(TICTACTOE_SELF_PLAY, simulations=12)),
        (Gungi(NANO_GUNGI), replace(NANO_GUNGI_SELF_PLAY, simulations=12)),
    ],
    ids=["tictactoe", "nano"],
)
def test_remembering_does_not_change_the_games(game, config) -> None:
    games = [(seed, None) for seed in range(8)]
    plain = PerBoardNetwork(game.action_space_size)
    without = play_self_play_episodes(game, plain, config, games, parallel=4)
    remembering = PerBoardNetwork(game.action_space_size)
    cache = EvaluationCache(10_000)

    remembered = play_self_play_episodes(game, remembering, config, games, parallel=4, cache=cache)

    for one, other in zip(without, remembered, strict=True):
        assert_same(one, other)
    assert cache.asked == sum(plain.sizes)
    assert cache.evaluated == sum(remembering.sizes) < cache.asked
