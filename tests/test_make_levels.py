"""Elo 를 잰 기록에서 난이도 단계를 만드는 합성 루트."""

from pathlib import Path

import pytest
import torch

from ai.resnet import ResNet
from scripts import make_levels, rate
from scripts.shared.levels import load_levels
from scripts.shared.setups import SETUPS
from tests.test_rate import some_openings

SETUP = SETUPS["nano-gungi"]


class FakeRun:
    def __init__(self) -> None:
        self.config: dict = {}

    def log(self, metrics: dict, step: int) -> None:
        pass

    def finish(self) -> None:
        pass


def test_it_builds_levels_from_the_log_of_a_rating_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    checkpoints = tmp_path / "run"
    checkpoints.mkdir()
    for number in (1, 2, 3):
        torch.manual_seed(number)
        network = ResNet(SETUP.game.tensor_shape, SETUP.game.action_space_size, SETUP.network)
        torch.save(network.state_dict(), checkpoints / f"generation-{number:04d}.pt")
    monkeypatch.setattr(rate, "start_wandb", lambda mode: FakeRun())
    monkeypatch.setattr(rate, "openings_for", some_openings)
    monkeypatch.setattr(make_levels, "openings_for", some_openings)
    rating = ["--checkpoint-dir", str(checkpoints), "--device", "cpu"]
    rate.main([*rating, "--openings", "2", "--simulations", "2"])
    log = tmp_path / "elo.log"
    # wandb 가 섞어 찍는 줄은 건너뛴다.
    log.write_text("wandb: 안내\n" + capsys.readouterr().out + "wandb: 끝\n", encoding="utf-8")
    output = tmp_path / "levels.json"

    arguments = ["--elo-log", str(log), "--device", "cpu", "--output", str(output)]
    # 작은 대진에서도 단계가 나오게 간격을 좁힌다.
    assert make_levels.main([*arguments, "--step", "20"]) == 0

    levels = load_levels(output)
    assert levels.game == "nano-gungi"
    assert levels.simulations == 2
    assert levels.levels
    assert [level.rating for level in levels.levels] == sorted(
        level.rating for level in levels.levels
    )
    assert all(Path(level.checkpoint).is_file() for level in levels.levels)
    assert all(0 <= level.random_probability < 1 for level in levels.levels)


def test_a_log_without_a_finished_rating_is_refused(tmp_path: Path) -> None:
    log = tmp_path / "elo.log"
    log.write_text("대진 10/287\n", encoding="utf-8")

    with pytest.raises(SystemExit):
        make_levels.main(["--elo-log", str(log)])
