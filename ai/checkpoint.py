"""체크포인트: 학습이 세대마다 저장하는 가중치 파일.

이름은 `generation-0042.pt` 꼴이고 폴더 하나가 학습 한 번이다. 파일에는 `state_dict` 만 있어
신경망 구조는 가중치에서 읽어 낸다. 그래서 구조를 바꿔 학습한 체크포인트도 같은 방법으로 읽힌다.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import torch

from ai.resnet import ResNet, ResNetConfig
from games.base import Game

_NAME = re.compile(r"generation-(\d+)\.pt")


def checkpoint_path(directory: Path, generation: int) -> Path:
    """`directory` 에서 `generation` 세대의 체크포인트."""
    return directory / f"generation-{generation:04d}.pt"


def generation_of(path: Path) -> int | None:
    """`generation-0042.pt` 의 42. 체크포인트 이름이 아니면 `None` 이다."""
    match = _NAME.fullmatch(path.name)
    return int(match.group(1)) if match else None


def has_checkpoints(directory: Path) -> bool:
    return any(directory.glob("generation-*.pt"))


def generation_files(directory: Path) -> dict[int, Path]:
    """세대 번호 → 체크포인트, 번호 순. 하나도 없으면 `ValueError` 다."""
    files = {}
    for file in directory.glob("generation-*.pt"):
        generation = generation_of(file)
        if generation is not None:
            files[generation] = file
    if not files:
        raise ValueError(f"{directory} 에 체크포인트가 없다")
    return dict(sorted(files.items()))


def save_network(network: torch.nn.Module, directory: Path, generation: int) -> Path:
    path = checkpoint_path(directory, generation)
    torch.save(network.state_dict(), path)
    return path


def load_network(game: Game[Any, Any], path: Path) -> ResNet:
    """`game` 의 신경망으로 읽는다. 블록 수와 채널 수는 가중치에서 읽는다. CPU 에 학습 모드로
    올라오므로 쓰는 쪽이 장치와 평가 모드를 정한다."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    network = ResNet(game.tensor_shape, game.action_space_size, ResNetConfig.of_weights(state))
    network.load_state_dict(state)
    return network
