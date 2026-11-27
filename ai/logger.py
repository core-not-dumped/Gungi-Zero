"""학습 기록의 자리. wandb 같은 바깥 도구를 쓰는 구현은 합성 루트가 만들어 주입한다."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol


class Logger(Protocol):
    def config(self, values: Mapping[str, Any]) -> None:
        """한 학습의 설정. 바뀌지 않는 값이다. 여러 번 부르면 키를 합친다."""
        ...

    def log(self, metrics: Mapping[str, float], step: int) -> None:
        """`step` 에서 잰 값들. 학습 루프의 `step` 은 1 부터 세는 세대 번호다."""
        ...


class MemoryLogger:
    """받은 것을 메모리에 남긴다. 테스트와 CI 가 쓴다."""

    def __init__(self) -> None:
        self.configuration: dict[str, Any] = {}
        # 부른 순서대로 (step, 값들).
        self.records: list[tuple[int, dict[str, float]]] = []

    def config(self, values: Mapping[str, Any]) -> None:
        self.configuration.update(values)

    def log(self, metrics: Mapping[str, float], step: int) -> None:
        self.records.append((step, dict(metrics)))
