"""작은 ResNet. `ai.network.Network` 프로토콜의 구현이다."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import nn

# 머리의 1x1 합성곱이 남기는 평면 수. 게임마다 바꾸지 않는다. 정책 머리는 칸마다 남긴 값으로
# 행동 공간(군의는 칸 수의 제곱의 두 배) 전체를 만들어야 해서 가치 머리보다 두껍다.
POLICY_PLANES = 32
VALUE_PLANES = 8


@dataclass(frozen=True, slots=True)
class ResNetConfig:
    """게임마다 다른 구조 설정."""

    blocks: int
    channels: int

    def __post_init__(self) -> None:
        if self.blocks < 1 or self.channels < 1:
            raise ValueError(f"블록 수와 채널 수는 1 이상이다: {self}")

    @classmethod
    def of_weights(cls, state: Mapping[str, torch.Tensor]) -> ResNetConfig:
        """`ResNet` 의 `state_dict` 에서 읽은 구조. 키 이름은 아래 `ResNet` 의 모양을 따른다."""
        return cls(
            blocks=sum(name.endswith(".first.0.weight") for name in state),
            channels=state["body.0.0.weight"].shape[0],
        )


def _convolution(inputs: int, outputs: int, kernel: int) -> nn.Sequential:
    """판 크기를 지키는 합성곱과 BatchNorm."""
    return nn.Sequential(
        nn.Conv2d(inputs, outputs, kernel, padding=kernel // 2, bias=False),
        nn.BatchNorm2d(outputs),
    )


class _Block(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.first = _convolution(channels, channels, 3)
        self.second = _convolution(channels, channels, 3)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        residual = self.second(torch.relu(self.first(features)))
        return torch.relu(features + residual)


class ResNet(nn.Module):
    """공통 몸통(합성곱 + 잔차 블록) 위의 정책 머리와 tanh 가치 머리.

    BatchNorm 이 있어 평가할 때는 평가 모드로 넘긴다. 가중치 초기화는 torch 의 전역 난수를 쓴다.
    """

    def __init__(
        self,
        tensor_shape: tuple[int, int, int],
        action_space_size: int,
        config: ResNetConfig,
    ) -> None:
        super().__init__()
        self.config = config
        planes, height, width = tensor_shape
        squares = height * width
        self.body = nn.Sequential(
            _convolution(planes, config.channels, 3),
            nn.ReLU(),
            *(_Block(config.channels) for _ in range(config.blocks)),
        )
        self.policy_head = nn.Sequential(
            _convolution(config.channels, POLICY_PLANES, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(POLICY_PLANES * squares, action_space_size),
        )
        self.value_head = nn.Sequential(
            _convolution(config.channels, VALUE_PLANES, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(VALUE_PLANES * squares, config.channels),
            nn.ReLU(),
            nn.Linear(config.channels, 1),
            nn.Tanh(),
        )

    def forward(self, boards: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.body(boards)
        return self.policy_head(features), self.value_head(features).squeeze(1)
