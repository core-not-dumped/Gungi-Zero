"""세대 루프 (AlphaZero 의 동기 진행, gating 없음).

한 세대는 현재 신경망으로 N판 자가 대국 → 최근 K세대의 리플레이 버퍼(메모리) → M스텝 학습이다.
손실은 정책 교차 엔트로피 + 가치 평균제곱오차이고 옵티마이저는 Adam 이다.
"""

from __future__ import annotations

import random
from collections import Counter, deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ai.logger import Logger
from ai.network import to_tensor
from ai.self_play import SelfPlayConfig, SelfPlayResult, TrainingSample, play_self_play_episode
from games.base import Game, MoveT, Outcome, PositionT


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """학습의 설정. 값은 게임마다 다르고 합성 루트가 고른다."""

    self_play: SelfPlayConfig
    # N: 한 세대의 자가 대국 수.
    episodes_per_generation: int
    # K: 리플레이 버퍼에 남기는 최근 세대 수. 지금 세대를 포함한다.
    buffer_generations: int
    # M: 한 세대의 학습 스텝 수.
    steps_per_generation: int
    # 스텝마다 버퍼에서 복원 추출하는 학습 샘플 수.
    batch_size: int
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4

    def __post_init__(self) -> None:
        counts = (
            "episodes_per_generation",
            "buffer_generations",
            "steps_per_generation",
            "batch_size",
        )
        for name in counts:
            if getattr(self, name) < 1:
                raise ValueError(f"{name} 은 1 이상이다: {getattr(self, name)}")


def seed_everything(seed: int) -> None:
    """Python·numpy·torch 의 전역 난수를 고정한다. 신경망을 만들기 **전에** 부른다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


CHECKPOINT_PATTERN = "generation-{}.pt"


def checkpoint_path(directory: Path, generation: int) -> Path:
    return directory / CHECKPOINT_PATTERN.format(f"{generation:04d}")


def train(
    game: Game[PositionT, MoveT],
    network: torch.nn.Module,
    config: TrainingConfig,
    *,
    generations: int,
    logger: Logger,
    end_reason: Callable[[PositionT], str],
    checkpoint_dir: Path,
    seed: int,
    device: torch.device | str = "cpu",
    start_position: Callable[[random.Random], PositionT] | None = None,
) -> None:
    """`network`(`Network` 프로토콜을 따르는 모듈)를 `generations` 세대 학습시킨다.

    `network` 를 `device` 로 옮겨 그 자리에서 바꾸고, 평가 모드로 남긴다. 세대마다 그 세대의
    `state_dict` 를 `checkpoint_dir` 에 저장하고, 그곳에 이미 체크포인트가 있으면 시작하기 전에
    `FileExistsError` 다. `end_reason` 은 끝난 국면의 종료 사유 이름이다 — `Game` 은 답하지
    않아 게임을 아는 쪽이 넘긴다.

    `start_position` 은 자가 대국 한 판을 시작할 국면을 고른다. 대국마다 한 번, 학습의 난수를
    받아 부른다. 없으면 모든 대국이 초기 국면에서 시작한다.

    같은 시드, 같은 처음 가중치는 CPU 에서 같은 세대 가중치를 만든다.
    """
    # 다른 학습의 세대가 섞이면 세대 사이의 비교가 조용히 틀린다.
    if any(checkpoint_dir.glob(CHECKPOINT_PATTERN.format("*"))):
        raise FileExistsError(f"이미 체크포인트가 있는 곳이다: {checkpoint_dir}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    network.to(device)
    optimizer = torch.optim.Adam(
        network.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    # 대국의 시드와 배치 추출이 모두 여기서 나온다.
    rng = random.Random(seed)
    buffer: deque[list[TrainingSample]] = deque(maxlen=config.buffer_generations)

    for generation in range(1, generations + 1):
        network.eval()
        results = []
        for _ in range(config.episodes_per_generation):
            # 시작 국면을 고르지 않으면 난수를 쓰지 않는다. 같은 시드의 결과가 그대로다.
            start = None if start_position is None else start_position(rng)
            results.append(
                play_self_play_episode(
                    game,
                    network,
                    config.self_play,
                    seed=rng.getrandbits(63),
                    device=device,
                    start=start,
                )
            )
        buffer.append([sample for result in results for sample in result.samples])
        samples = [sample for generation_samples in buffer for sample in generation_samples]

        network.train()
        losses = [
            _step(network, optimizer, rng.choices(samples, k=config.batch_size), device)
            for _ in range(config.steps_per_generation)
        ]
        network.eval()

        policy, value = (sum(column) / len(losses) for column in zip(*losses, strict=True))
        metrics = {"loss/total": policy + value, "loss/policy": policy, "loss/value": value}
        metrics |= _self_play_metrics(results, end_reason)
        metrics["replay_buffer/samples"] = len(samples)
        # 기록한 세대는 체크포인트가 있도록 먼저 저장한다.
        torch.save(network.state_dict(), checkpoint_path(checkpoint_dir, generation))
        logger.log(metrics, step=generation)


def _step(
    network: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: Sequence[TrainingSample],
    device: torch.device | str,
) -> tuple[float, float]:
    """배치 하나로 한 번 학습하고, 바꾸기 전 가중치의 (정책 손실, 가치 손실)을 돌려준다."""
    boards = to_tensor([sample.board for sample in batch], device)
    masks = to_tensor([sample.legal_mask for sample in batch], device)
    policies = to_tensor([sample.policy for sample in batch], device)
    targets = to_tensor([np.float32(sample.value) for sample in batch], device)

    logits, values = network(boards)
    # MCTS 처럼 합법수만의 softmax 다. 불법수는 -inf 대신 가장 작은 유한값으로 가려
    # `0 * log 0` 이 NaN 이 되지 않게 한다.
    masked = logits.masked_fill(~masks, torch.finfo(logits.dtype).min)
    policy_loss = -(policies * torch.log_softmax(masked, dim=1)).sum(dim=1).mean()
    value_loss = torch.nn.functional.mse_loss(values, targets)

    optimizer.zero_grad()
    (policy_loss + value_loss).backward()
    optimizer.step()
    return policy_loss.item(), value_loss.item()


def _self_play_metrics(
    results: Sequence[SelfPlayResult[Any, Any]], end_reason: Callable[[Any], str]
) -> dict[str, float]:
    """한 세대의 대국 길이, 결과 분포(판 수), 종료 사유(판 수). 없던 종료 사유는 빠진다."""
    lengths = [len(result.episode.moves) for result in results]
    metrics: dict[str, float] = {
        "self_play/length_mean": sum(lengths) / len(lengths),
        "self_play/length_min": min(lengths),
        "self_play/length_max": max(lengths),
    }
    outcomes = Counter(result.episode.outcome for result in results)
    for outcome in Outcome:
        metrics[f"self_play/outcome/{outcome.name.lower()}"] = outcomes[outcome]
    reasons = Counter(end_reason(result.episode.final_position) for result in results)
    for reason, count in reasons.items():
        metrics[f"self_play/end_reason/{reason}"] = count
    return metrics
