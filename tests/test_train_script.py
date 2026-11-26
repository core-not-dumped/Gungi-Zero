"""학습의 합성 루트. 남기는 설정, 기록, 체크포인트를 본다."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from ai.logger import MemoryLogger
from ai.resnet import ResNetConfig
from scripts.train import SETUPS, GameSetup, main, run


def small(name: str) -> GameSetup:
    """실제 설정에서 크기만 줄인다."""
    setup = SETUPS[name]
    training = replace(
        setup.training,
        self_play=replace(setup.training.self_play, simulations=4),
        episodes_per_generation=2,
        steps_per_generation=2,
        batch_size=4,
    )
    return replace(setup, network=ResNetConfig(blocks=1, channels=4), training=training)


def train_one_generation(name: str, directory: Path) -> MemoryLogger:
    logger = MemoryLogger()
    run(small(name), logger=logger, generations=1, checkpoint_dir=directory, seed=3)
    return logger


def test_the_nano_variant_is_recorded_as_it_is(tmp_path: Path) -> None:
    logger = train_one_generation("nano-gungi", tmp_path)

    assert logger.configuration["variant"] == {
        "width": 3,
        "height": 4,
        "max_tier": 2,
        "pieces": [
            {"kind": "GENERAL", "column": 0, "row": 0},
            {"kind": "MARSHAL", "column": 1, "row": 0},
            {"kind": "PAWN", "column": 2, "row": 1},
        ],
        "move_limit": 200,
        "repetition_limit": 3,
    }


def test_tictactoe_has_no_termination_limits(tmp_path: Path) -> None:
    logger = train_one_generation("tictactoe", tmp_path)

    assert logger.configuration["variant"] == {
        "width": 3,
        "height": 3,
        "max_tier": 1,
        "pieces": [],
        "move_limit": None,
        "repetition_limit": None,
    }


def test_the_hyperparameters_are_recorded(tmp_path: Path) -> None:
    logger = train_one_generation("tictactoe", tmp_path)

    config = logger.configuration
    assert (config["game"], config["seed"], config["generations"]) == ("tictactoe", 3, 1)
    assert config["device"] == "cpu"
    assert config["network"] == {"blocks": 1, "channels": 4}
    assert config["training"]["batch_size"] == 4
    assert config["training"]["self_play"]["root_noise"]["alpha"] == 1.8
    # wandb 가 받을 수 있는 값들이다.
    json.dumps(config)


@pytest.mark.parametrize(
    ("name", "reasons"),
    [
        ("tictactoe", {"LINE", "BOARD_FULL"}),
        ("nano-gungi", {"MARSHAL_CAPTURE", "NO_LEGAL_MOVES", "REPETITION", "MOVE_LIMIT"}),
    ],
)
def test_every_episode_is_counted_under_an_end_reason(
    name: str, reasons: set[str], tmp_path: Path
) -> None:
    [(_, metrics)] = train_one_generation(name, tmp_path).records

    prefix = "self_play/end_reason/"
    counted = {k.removeprefix(prefix): v for k, v in metrics.items() if k.startswith(prefix)}
    assert counted.keys() <= reasons
    assert sum(counted.values()) == 2


def test_the_command_runs_a_tictactoe_generation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(SETUPS, "tictactoe", small("tictactoe"))

    assert main(["--checkpoint-dir", str(tmp_path)]) == 0

    assert [path.name for path in tmp_path.iterdir()] == ["generation-0001.pt"]
    config, generation = (json.loads(line) for line in capsys.readouterr().out.splitlines())
    assert config["game"] == "tictactoe"
    assert generation["step"] == 1
    assert "loss/total" in generation


def test_the_default_checkpoint_directory_is_new_for_each_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(SETUPS, "tictactoe", small("tictactoe"))
    monkeypatch.chdir(tmp_path)

    assert main([]) == 0
    assert main([]) == 0

    runs = list((tmp_path / "checkpoints").iterdir())
    assert len(runs) == 2
