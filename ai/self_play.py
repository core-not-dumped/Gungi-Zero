"""자가 대국. 한 AI 가 자기 자신과 한 판을 두고 수마다 학습 샘플을 남긴다 (`CONTEXT.md`).

방문 분포를 남겨야 해서 `play_episode` 를 쓰지 않고 자기 loop 를 둔다. 거울 증강은 하지 않는다
(ADR-0005).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Generic

import numpy as np
import numpy.typing as npt
import torch

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
        result = search.search(position)
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
