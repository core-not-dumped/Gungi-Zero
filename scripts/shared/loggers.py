"""합성 루트가 주입하는 로거 (`ai.logger.Logger`)."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from ai.logger import Logger

WANDB_PROJECT = "gungi-zero"


class WandbLogger:
    """wandb 의 run 에 남긴다. run 은 부르는 쪽이 만들고 끝낸다."""

    def __init__(self, run: Any) -> None:
        self._run = run

    def config(self, values: Mapping[str, Any]) -> None:
        self._run.config.update(dict(values))

    def log(self, metrics: Mapping[str, float], step: int) -> None:
        self._run.log(dict(metrics), step=step)


class PrintingLogger:
    """받는 대로 JSON 한 줄씩 찍는다. 긴 학습의 진행을 터미널과 로그 파일에서 본다."""

    def config(self, values: Mapping[str, Any]) -> None:
        print(json.dumps(dict(values), ensure_ascii=False), flush=True)

    def log(self, metrics: Mapping[str, float], step: int) -> None:
        print(json.dumps({"step": step, **metrics}, ensure_ascii=False), flush=True)


class TeeLogger:
    """받은 것을 여러 로거에 그대로 넘긴다."""

    def __init__(self, *loggers: Logger) -> None:
        self._loggers = loggers

    def config(self, values: Mapping[str, Any]) -> None:
        for logger in self._loggers:
            logger.config(values)

    def log(self, metrics: Mapping[str, float], step: int) -> None:
        for logger in self._loggers:
            logger.log(metrics, step)
