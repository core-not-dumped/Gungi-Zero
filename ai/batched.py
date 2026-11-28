"""신경망 평가를 기다리며 멈추는 작업들을 번갈아 진행한다.

작업은 평가할 판(`encode` 텐서)을 `yield` 하고 `(정책 로짓, 가치)` 를 받아 이어 가다가 결과를
`return` 하는 제너레이터다. 탐색과 자가 대국이 이 모양이라, 하나만 돌리면 판을 하나씩 평가하고
여럿을 함께 돌리면 평가가 묶인다. 한 프로세스에서 번갈아 진행할 뿐 CPU 에서 병렬로 돌지 않는다.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Generator, Iterable
from typing import TypeVar

import numpy as np
import numpy.typing as npt
import torch

from ai.network import Network, evaluate, evaluate_batch

ResultT = TypeVar("ResultT")

# 판을 내놓고, 그 판의 정책 로짓과 가치를 받고, 끝나면 결과를 돌려준다.
Steps = Generator[npt.NDArray[np.float32], tuple[npt.NDArray[np.float32], float], ResultT]


def run_alone(
    steps: Steps[ResultT], network: Network, device: torch.device | str = "cpu"
) -> ResultT:
    """작업 하나를 판을 하나씩 평가하며 끝까지 돌린다."""
    try:
        board = next(steps)
        while True:
            board = steps.send(evaluate(network, board, device))
    except StopIteration as finished:
        return finished.value


class EvaluationCache:
    """평가한 판의 답을 기억한다. 같은 `encode` 텐서를 다시 물으면 신경망 없이 답한다.

    **신경망의 가중치가 그대로인 동안에만** 쓴다. 학습하면 새로 만든다. `capacity` 개를 넘으면
    가장 오래 쓰지 않은 판부터 잊는다.
    """

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError(f"기억하는 판 수는 1 이상이다: {capacity}")
        self._capacity = capacity
        self._answers: OrderedDict[bytes, tuple[npt.NDArray[np.float32], float]] = OrderedDict()
        # 물은 횟수와, 그중 신경망이 실제로 평가한 판 수.
        self.asked = 0
        self.evaluated = 0

    def recall(self, key: bytes) -> tuple[npt.NDArray[np.float32], float] | None:
        answer = self._answers.get(key)
        if answer is not None:
            self.asked += 1
            self._answers.move_to_end(key)
        return answer

    def remember(self, key: bytes, answer: tuple[npt.NDArray[np.float32], float]) -> None:
        self.evaluated += 1
        self._answers[key] = answer
        if len(self._answers) > self._capacity:
            self._answers.popitem(last=False)


def run_batched(
    tasks: Iterable[Steps[ResultT]],
    network: Network,
    *,
    width: int,
    device: torch.device | str = "cpu",
    cache: EvaluationCache | None = None,
) -> list[ResultT]:
    """작업을 `width` 개까지 함께 돌려, 내놓은 판들을 한 번에 평가한다. 결과는 `tasks` 의 순서다.

    끝난 작업의 자리는 다음 평가 전에 새 작업으로 채운다. 남은 작업이 `width` 보다 적어지면 묶음도
    작아진다. `width` 가 1 이고 `cache` 가 없으면 `run_alone` 을 차례로 부른 것과 같은 평가를 한다.

    `cache` 를 주면 기억하는 판은 평가를 기다리지 않고 바로 답하고, 한 묶음에서 겹친 판은 한 번만
    평가한다.
    """
    if width < 1:
        raise ValueError(f"함께 돌리는 작업 수는 1 이상이다: {width}")
    waiting = enumerate(tasks)
    results: dict[int, ResultT] = {}
    # 평가를 기다리는 작업: (순번, 작업, 내놓은 판).
    asking: list[tuple[int, Steps[ResultT], npt.NDArray[np.float32]]] = []

    def advance(
        number: int, steps: Steps[ResultT], answer: tuple[npt.NDArray[np.float32], float] | None
    ) -> None:
        try:
            board = next(steps) if answer is None else steps.send(answer)
            while cache is not None and (known := cache.recall(board.tobytes())) is not None:
                board = steps.send(known)
        except StopIteration as finished:
            results[number] = finished.value
        else:
            asking.append((number, steps, board))

    while True:
        while len(asking) < width and (task := next(waiting, None)) is not None:
            advance(*task, None)
        if not asking:
            return [results[number] for number in range(len(results))]
        batch, asking = asking, []
        if cache is None:
            answers = _answers(network, [board for _, _, board in batch], device)
        else:
            keys = [board.tobytes() for _, _, board in batch]
            distinct = {key: board for key, (_, _, board) in zip(keys, batch, strict=True)}
            found = dict(
                zip(distinct, _answers(network, list(distinct.values()), device), strict=True)
            )
            for key, answer in found.items():
                cache.remember(key, answer)
            cache.asked += len(keys)
            answers = [found[key] for key in keys]
        for (number, steps, _), answer in zip(batch, answers, strict=True):
            advance(number, steps, answer)


def _answers(
    network: Network, boards: list[npt.NDArray[np.float32]], device: torch.device | str
) -> list[tuple[npt.NDArray[np.float32], float]]:
    """판마다의 `(정책 로짓, 가치)`. 로짓은 묶음 전체를 붙들지 않게 복사한다."""
    logits, values = evaluate_batch(network, boards, device)
    return [(logit.copy(), float(value)) for logit, value in zip(logits, values, strict=True)]
