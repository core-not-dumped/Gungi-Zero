"""합성 루트의 로거. wandb 의 run 자리에는 가짜를 꽂아 네트워크 없이 본다."""

import json
from typing import Any

import pytest

from ai.logger import MemoryLogger
from scripts.shared.loggers import PrintingLogger, TeeLogger, WandbLogger


class FakeConfig(dict):
    def update(self, values: Any) -> None:
        super().update(values)


class FakeRun:
    def __init__(self) -> None:
        self.config = FakeConfig()
        self.logged: list[tuple[dict, int]] = []

    def log(self, metrics: dict, step: int) -> None:
        self.logged.append((metrics, step))


def test_the_wandb_logger_passes_the_config_and_the_steps_to_the_run() -> None:
    run = FakeRun()
    logger = WandbLogger(run)

    logger.config({"game": "nano-gungi", "variant": {"move_limit": 200}})
    logger.config({"evaluation": {"simulations": 100}})
    logger.log({"loss/total": 1.5}, step=3)

    assert run.config == {
        "game": "nano-gungi",
        "variant": {"move_limit": 200},
        "evaluation": {"simulations": 100},
    }
    assert run.logged == [({"loss/total": 1.5}, 3)]


def test_the_printing_logger_writes_a_json_line_for_each_call(
    capsys: pytest.CaptureFixture[str],
) -> None:
    logger = PrintingLogger()

    logger.config({"seed": 0})
    logger.log({"loss/total": 1.5}, step=3)

    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert lines == [{"seed": 0}, {"step": 3, "loss/total": 1.5}]


def test_the_tee_logger_gives_every_logger_the_same_calls() -> None:
    first, second = MemoryLogger(), MemoryLogger()
    logger = TeeLogger(first, second)

    logger.config({"seed": 0})
    logger.log({"loss/total": 1.5}, step=1)

    for each in (first, second):
        assert each.configuration == {"seed": 0}
        assert each.records == [(1, {"loss/total": 1.5})]
