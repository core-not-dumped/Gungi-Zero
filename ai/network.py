"""신경망의 자리. numpy → torch 변환, 배치, 디바이스는 여기서 맡는다."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt
import torch


class Network(Protocol):
    """정규형 입력 → 정책 로짓, 가치. 구현은 갈아끼울 수 있다.

    생성자는 게임의 `tensor_shape`·`action_space_size` 와 구조 설정만 받는다. 합법수 마스크는
    씌우지 않는다 — 부르는 쪽이 씌운다.
    """

    def __call__(self, boards: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """`(B, C, H, W)` → 정책 로짓 `(B, A)`, 가치 `(B,)`.

        정책은 정규형 프레임의 행동 인덱스로 읽고, 가치는 `[-1, 1]` 이며 둘 차례인 쪽 시점이다.
        평가가 신경망의 상태를 바꾸면 안 된다 — BatchNorm 같은 층이 있으면 평가 모드로 넘긴다.
        """
        ...


def to_tensor(
    arrays: Sequence[npt.NDArray[Any] | np.generic], device: torch.device | str = "cpu"
) -> torch.Tensor:
    """모양이 같은 배열(또는 numpy 스칼라)들을 배치 축을 앞에 둔 텐서로. 자료형은 그대로다."""
    return torch.from_numpy(np.stack(arrays)).to(device)


def evaluate(
    network: Network, board: npt.NDArray[np.float32], device: torch.device | str = "cpu"
) -> tuple[npt.NDArray[np.float32], float]:
    """국면 하나의 `encode` 텐서를 평가해 정책 로짓 `(A,)` 와 가치를 돌려준다."""
    with torch.inference_mode():
        logits, values = network(to_tensor([board], device))
    return logits[0].cpu().numpy(), float(values[0])
