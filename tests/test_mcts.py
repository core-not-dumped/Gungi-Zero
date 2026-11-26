"""MCTS. 어떤 국면에서 무슨 수를 골랐는지와 낸 방문 분포만 본다."""

import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
import torch

from ai.mcts import MCTS, MCTSAI, RootNoise, SearchResult
from games.base import Game, Player
from games.mini_gungi import Gungi
from games.nano_gungi import NANO_GUNGI
from games.tictactoe import TicTacToe
from tests.stub_networks import UniformNetwork


@dataclass(frozen=True)
class Puzzle:
    """사람 검토를 받은 시험 국면 (`.scratch/alphazero-verification/spec.md`)."""

    name: str
    game: Callable[[], Game]
    board: str
    # `board` 에서 읽은 국면에 이어 두면 시험 국면이 된다.
    played: str
    legal_moves: int
    answer: str
    simulations: int


TICTACTOE_WIN_IN_ONE = Puzzle(
    name="tictactoe-win-in-one",
    game=TicTacToe,
    board="""
     a   b   c
 3   .   ▽   ▲
 2   .   .   .
 1   ▲   ▽   .
차례: ▲   ply: 4
""",
    played="",
    legal_moves=5,
    answer="b2",
    simulations=50,
)

TICTACTOE_BLOCK_THE_LOSS = Puzzle(
    name="tictactoe-block-the-loss",
    game=TicTacToe,
    board="""
     a   b   c
 3   ▲   ▽   ▲
 2   ▲   .   ▽
 1   .   .   .
차례: ▽   ply: 5
""",
    played="",
    legal_moves=4,
    answer="a1",
    simulations=50,
)

# 시험 국면은 판의 두 번째 출현이다. `b4-c4` 를 두면 세 번째가 되어 반복 무승부다.
NANO_TAKE_THE_REPETITION = Puzzle(
    name="nano-take-the-repetition",
    game=lambda: Gungi(NANO_GUNGI),
    board="""
     a       b       c
 4   .       .      ▽帥
 3 ▲大/▽兵  ▽大      .
 2   .       .      ▲兵
 1  ▲帥      .       .
차례: ▲   ply: 4
""",
    played="a1-a2 c4-b4 a2-a1 b4-c4 a1-a2 c4-b4 a2-a1",
    legal_moves=11,
    answer="b4-c4",
    # 지는 수 하나에는 상대의 응수가 14개까지 있고, 帥 를 잡는 응수는 균등 정책에서 뒤쪽 순번이다.
    # 100회에서는 정답이 다른 6수와 최다 방문 동률이다. 114회부터 정답이 홀로 앞선다.
    simulations=400,
)

# `a3-a2` 는 帥 잡기, `a3-a4` 는 반복 무승부다.
NANO_CAPTURE_OVER_REPETITION = Puzzle(
    name="nano-capture-over-repetition",
    game=lambda: Gungi(NANO_GUNGI),
    board="""
     a       b       c
 4  ▽兵     ▽帥     ▽大
 3   .       .       .
 2  ▲帥      .      ▲兵
 1  ▲大      .       .
차례: ▲   ply: 2
""",
    played="a1-b1 a4-a3 b1-a1 a3-a4 a1-b1 a4-a3 b1-a1",
    legal_moves=9,
    answer="a3-a2",
    simulations=100,
)

PUZZLES = (
    TICTACTOE_WIN_IN_ONE,
    TICTACTOE_BLOCK_THE_LOSS,
    NANO_TAKE_THE_REPETITION,
    NANO_CAPTURE_OVER_REPETITION,
)


def position_of(puzzle: Puzzle) -> tuple[Game, Any]:
    game = puzzle.game()
    position = game.parse(puzzle.board)
    for text in puzzle.played.split():
        position = game.apply_move(position, game.parse_move(text))
    return game, position


def mcts_for(game: Game, simulations: int, mcts_class: type[MCTS] = MCTS) -> MCTS:
    network = UniformNetwork(game.tensor_shape, game.action_space_size)
    return mcts_class(game, network, simulations=simulations)


@pytest.mark.parametrize("puzzle", PUZZLES, ids=lambda p: p.name)
def test_it_picks_the_right_move(puzzle: Puzzle) -> None:
    game, position = position_of(puzzle)
    assert len(game.legal_moves(position)) == puzzle.legal_moves
    ai = MCTSAI(mcts_for(game, puzzle.simulations), seed=0)
    assert game.render_move(ai.choose(position)) == puzzle.answer


class FlippedBackupMCTS(MCTS):
    """백업 부호가 뒤집힌 탐색."""

    def _backup(self, path, value: float) -> None:
        super()._backup(path, -value)


@pytest.mark.parametrize("puzzle", PUZZLES, ids=lambda p: p.name)
def test_a_flipped_backup_sign_fails_the_puzzle(puzzle: Puzzle) -> None:
    """뒤집어도 맞히는 시험은 부호를 시험하지 못한다."""
    game, position = position_of(puzzle)
    ai = MCTSAI(mcts_for(game, puzzle.simulations, FlippedBackupMCTS), seed=0)
    assert game.render_move(ai.choose(position)) != puzzle.answer


def second_to_move() -> tuple[Game, Any]:
    """후수 차례의 군의. 원래 프레임과 정규형 프레임의 행동 인덱스가 다르다."""
    return position_of(NANO_TAKE_THE_REPETITION)


def test_the_visit_distribution_is_in_the_canonical_frame() -> None:
    game, position = second_to_move()
    assert position.player is Player.SECOND
    result = mcts_for(game, simulations=100).search(position)

    assert result.moves == game.legal_moves(position)
    assert sum(result.visits) == 100
    assert result.policy.shape == (game.action_space_size,)
    assert result.policy.sum() == pytest.approx(1.0)
    for move, visits in zip(result.moves, result.visits, strict=True):
        index = game.to_canonical_action_index(position, game.action_index(move))
        assert result.policy[index] == pytest.approx(visits / 100)
    # 원래 프레임의 인덱스로 담았다면 이 마스크 밖에 질량이 남는다.
    assert not result.policy[~game.legal_action_mask(game.canonical(position))].any()


class FavouringNetwork(UniformNetwork):
    """정규형 행동 인덱스 하나만 크게 선호한다."""

    def __init__(self, game: Game, favourite: int) -> None:
        super().__init__(game.tensor_shape, game.action_space_size)
        self._favourite = favourite

    def __call__(self, boards: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits, values = super().__call__(boards)
        logits[:, self._favourite] = 20.0
        return logits, values


def test_the_policy_is_read_in_the_canonical_frame() -> None:
    game, position = second_to_move()
    for move in game.legal_moves(position):
        favourite = game.to_canonical_action_index(position, game.action_index(move))
        search = MCTS(game, FavouringNetwork(game, favourite), simulations=1)
        assert MCTSAI(search).choose(position) == move, game.render_move(move)


def test_it_never_builds_the_canonical_position(monkeypatch: pytest.MonkeyPatch) -> None:
    game, position = second_to_move()

    def refuse(_: object) -> object:
        raise AssertionError("canonical 을 불렀다")

    monkeypatch.setattr(game, "canonical", refuse)
    mcts_for(game, simulations=100).search(position)


def test_finished_positions_are_judged_by_the_rules_not_the_network() -> None:
    game = TicTacToe()
    # 빈 칸이 하나라 모든 수순이 한 수 만에 끝난다. 평가할 것은 루트뿐이다.
    position = game.parse("""
     a   b   c
 3   ▲   ▽   ▲
 2   ▲   ▽   ▽
 1   ▽   ▲   .
차례: ▲   ply: 8
""")
    network = UniformNetwork(game.tensor_shape, game.action_space_size)
    result = MCTS(game, network, simulations=20).search(position)
    assert result.visits == (20,)
    assert network.calls == 1


def test_it_refuses_a_finished_position() -> None:
    game = TicTacToe()
    finished = game.parse("""
     a   b   c
 3   .   .   .
 2   ▽   ▽   .
 1   ▲   ▲   ▲
차례: ▽   ply: 5
""")
    with pytest.raises(ValueError):
        mcts_for(game, simulations=10).search(finished)


def test_it_refuses_a_canonical_position() -> None:
    game, position = second_to_move()
    with pytest.raises(ValueError):
        mcts_for(game, simulations=10).search(game.canonical(position))


def test_it_refuses_to_search_without_simulations() -> None:
    with pytest.raises(ValueError):
        mcts_for(TicTacToe(), simulations=0)


def noisy_search(seed: int, noise: RootNoise | None) -> SearchResult:
    game = TicTacToe()
    network = UniformNetwork(game.tensor_shape, game.action_space_size)
    search = MCTS(game, network, simulations=50, root_noise=noise, seed=seed)
    return search.search(game.initial_position())


def test_root_noise_changes_the_visits_with_the_seed() -> None:
    noise = RootNoise(alpha=1.8)
    visits = {noisy_search(seed, noise).visits for seed in range(10)}
    assert len(visits) > 1
    assert noisy_search(3, noise).visits == noisy_search(3, noise).visits


def test_without_root_noise_the_seed_changes_nothing() -> None:
    assert len({noisy_search(seed, noise=None).visits for seed in range(10)}) == 1


def result_with_visits(*visits: int) -> SearchResult:
    game = TicTacToe()
    moves = game.legal_moves(game.initial_position())[: len(visits)]
    return SearchResult(moves, visits, np.zeros(game.action_space_size, dtype=np.float32))


def chosen_counts(result: SearchResult, temperature: float, draws: int = 4000) -> list[int]:
    rng = random.Random(0)
    chosen = [result.select_move(temperature, rng) for _ in range(draws)]
    return [chosen.count(move) for move in result.moves]


def test_temperature_zero_picks_among_the_most_visited_moves() -> None:
    counts = chosen_counts(result_with_visits(5, 20, 20, 0), temperature=0)
    assert counts[0] == counts[3] == 0
    assert counts[1] > 1500 and counts[2] > 1500


def test_temperature_one_picks_in_proportion_to_the_visits() -> None:
    counts = chosen_counts(result_with_visits(10, 30, 0), temperature=1)
    assert 800 <= counts[0] <= 1200
    assert counts[2] == 0


def test_a_low_temperature_sharpens_without_overflowing() -> None:
    counts = chosen_counts(result_with_visits(400, 800), temperature=0.001)
    assert counts == [0, 4000]


def test_a_negative_temperature_is_refused() -> None:
    with pytest.raises(ValueError):
        result_with_visits(1, 2).select_move(-1.0, random.Random(0))
