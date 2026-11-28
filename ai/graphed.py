"""CUDA 에서 신경망 호출의 비용을 줄이는 평가기.

층마다 GPU 에 명령을 넣는 일을 한 번 기록해 두고(CUDA graph) 호출마다 재생한다. 기록은 묶음 크기가
고정이라 작은 묶음은 남는 자리를 둔 채 계산한다 — GPU 에서는 크기에 따른 시간 차이가 거의 없다.
"""

from __future__ import annotations

import torch

from ai.network import Network


class GraphedNetwork:
    """`network` 를 `size` 판짜리 기록으로 감싼 `Network`. `size` 보다 큰 묶음은 `network` 가 그대로
    평가한다.

    평가 모드에서만 만든다. 기록은 가중치의 자리를 가리키므로, 학습한 뒤에는 새로 만들어 쓴다.
    """

    def __init__(
        self,
        network: Network,
        tensor_shape: tuple[int, int, int],
        size: int,
        device: torch.device | str,
    ) -> None:
        if getattr(network, "training", False):
            raise ValueError("평가 모드의 신경망만 기록한다")
        self._network = network
        self._size = size
        self._inputs = torch.zeros((size, *tensor_shape), device=device)
        with torch.no_grad():
            # 처음 몇 번의 준비 작업은 기록에 넣지 않는다.
            with torch.cuda.stream(torch.cuda.Stream(device=device)):
                for _ in range(3):
                    network(self._inputs)
            torch.cuda.synchronize(device)
            self._graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self._graph):
                self._outputs = network(self._inputs)

    def __call__(self, boards: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        count = len(boards)
        if count > self._size:
            return self._network(boards)
        self._inputs[:count].copy_(boards)
        self._graph.replay()
        logits, values = self._outputs
        # 기록의 출력 자리는 다음 호출이 덮어쓴다.
        return logits[:count].clone(), values[:count].clone()
