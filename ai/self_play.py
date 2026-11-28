"""자가 대국. 한 AI 가 자기 자신과 한 판을 두고 수마다 학습 샘플을 남긴다 (`CONTEXT.md`).

방문 분포를 남겨야 해서 `play_episode` 를 쓰지 않고 자기 loop 를 둔다. 거울 증강은 하지 않는다
(ADR-0005).
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Generic

import numpy as np
import numpy.typing as npt
import torch

from ai.batched import EvaluationCache, Steps, run_alone, run_batched
from ai.episode import Episode, value_for
from ai.mcts import DEFAULT_C_PUCT, MCTS, RootNoise, canonical_action_indices
from ai.network import Network
from games.base import Game, MoveT, Player, PositionT


@dataclass(frozen=True, slots=True)
class SelfPlayConfig:
    """자가 대국의 설정. 값은 게임마다 다르고 합성 루트가 고른다."""

    simulations: int
    # 자가 대국은 루트 노이즈를 늘 켠다.
    root_noise: RootNoise
    # 대국의 첫 몇 수(양쪽을 합쳐 센다)를 temperature 1 로 고른다. 그 뒤는 0 이다.
    exploration_moves: int
    c_puct: float = DEFAULT_C_PUCT


@dataclass(frozen=True, eq=False)
class TrainingSample:
    """한 수에서 남기는 학습 데이터. 모두 그 국면의 **정규형** 프레임이다."""

    # `(C, H, W)` 정규형 입력.
    board: npt.NDArray[np.float32]
    # `(action_space_size,)` 합법수 마스크.
    legal_mask: npt.NDArray[np.bool_]
    # `(action_space_size,)` 탐색이 낸 방문 분포. 루트 노이즈의 영향이 든 그대로다.
    policy: npt.NDArray[np.float32]
    # 그 국면에서 둘 차례인 쪽 시점의 최종 결과. 승 +1, 무 0, 패 -1.
    value: float


@dataclass(frozen=True, eq=False)
class SelfPlayResult(Generic[PositionT, MoveT]):
    """자가 대국 한 판. `samples[i]` 는 `episode.moves[i]` 를 둔 국면의 것이다."""

    episode: Episode[PositionT, MoveT]
    samples: tuple[TrainingSample, ...]


@dataclass(frozen=True, eq=False)
class _Turn:
    """가치 타깃만 빠진 학습 샘플. 타깃은 대국이 끝나야 안다."""

    player: Player
    board: npt.NDArray[np.float32]
    legal_mask: npt.NDArray[np.bool_]
    policy: npt.NDArray[np.float32]


def play_self_play_episode(
    game: Game[PositionT, MoveT],
    network: Network,
    config: SelfPlayConfig,
    *,
    seed: int | None = None,
    device: torch.device | str = "cpu",
    start: PositionT | None = None,
) -> SelfPlayResult[PositionT, MoveT]:
    """`start`(없으면 초기 국면)부터 종료까지 둔다.

    같은 시드, 같은 시작 국면, 같은 신경망은 같은 결과를 낸다. `exploration_moves` 는 `ply` 가
    아니라 이 대국에서 둔 수를 센다.
    """
    steps = _episode_steps(game, network, config, seed=seed, device=device, start=start)
    return run_alone(steps, network, device)


def play_self_play_episodes(
    game: Game[PositionT, MoveT],
    network: Network,
    config: SelfPlayConfig,
    games: Sequence[tuple[int, PositionT | None]],
    *,
    parallel: int,
    device: torch.device | str = "cpu",
    cache: EvaluationCache | None = None,
) -> list[SelfPlayResult[PositionT, MoveT]]:
    """`games` 의 `(시드, 시작 국면)` 마다 한 판씩, `parallel` 판까지 함께 두며 신경망 평가를
    묶는다. 결과는 `games` 의 순서다.

    대국마다는 `play_self_play_episode` 와 같다. 판을 따로따로 평가하는 신경망이면 `parallel` 에
    상관없이 같은 결과다. 실제 신경망은 묶는 크기에 따라 부동소수점 끝자리가 달라질 수 있다.
    `cache` 를 주면 같은 판의 평가를 되풀이하지 않는다.
    """
    return run_batched(
        (
            _episode_steps(game, network, config, seed=seed, device=device, start=start)
            for seed, start in games
        ),
        network,
        width=parallel,
        device=device,
        cache=cache,
    )


def _episode_steps(
    game: Game[PositionT, MoveT],
    network: Network,
    config: SelfPlayConfig,
    *,
    seed: int | None,
    device: torch.device | str,
    start: PositionT | None,
) -> Steps[SelfPlayResult[PositionT, MoveT]]:
    """자가 대국 한 판을 평가마다 멈추는 작업으로 (`ai.batched`)."""
    search = MCTS(
        game,
        network,
        simulations=config.simulations,
        c_puct=config.c_puct,
        root_noise=config.root_noise,
        seed=seed,
        device=device,
    )
    rng = random.Random(seed)
    position = game.initial_position() if start is None else start
    moves: list[MoveT] = []
    turns: list[_Turn] = []
    while (outcome := game.outcome(position)) is None:
        result = yield from search.search_steps(position)
        # `canonical` 은 후수 차례에서 비싸서 부르지 않는다. `encode` 는 원래 국면에 바로 부르고
        # 마스크는 합법수의 인덱스를 옮겨 만든다.
        legal_mask = np.zeros(game.action_space_size, dtype=bool)
        legal_mask[canonical_action_indices(game, position, result.moves)] = True
        turns.append(_Turn(position.player, game.encode(position), legal_mask, result.policy))
        temperature = 1.0 if len(moves) < config.exploration_moves else 0.0
        move = result.select_move(temperature, rng)
        moves.append(move)
        position = game.apply_move(position, move)

    samples = tuple(
        TrainingSample(turn.board, turn.legal_mask, turn.policy, value_for(outcome, turn.player))
        for turn in turns
    )
    return SelfPlayResult(Episode(tuple(moves), position, outcome), samples)
