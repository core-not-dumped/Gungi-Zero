"""학습의 합성 루트. 남기는 설정, 기록, 체크포인트를 본다."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from ai.logger import MemoryLogger
from ai.resnet import ResNetConfig
from games.mini_gungi import Gungi
from scripts.train import SETUPS, GameSetup, gungi_variant, main, run
from tests.solver_helpers import SMALL_NANO


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
    # 시작 국면을 고르려면 도달 가능한 상태를 모두 돌아야 해서 여기서는 끈다.
    return replace(
        setup,
        network=ResNetConfig(blocks=1, channels=4),
        training=training,
        initial_start_fraction=1.0,
    )


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
            {"kind": "GENERAL", "column": 2, "row": 1},
            {"kind": "MARSHAL", "column": 1, "row": 0},
            {"kind": "PAWN", "column": 2, "row": 0},
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


def test_the_nano_self_play_starts_from_a_quarter_initial_positions() -> None:
    assert SETUPS["nano-gungi"].initial_start_fraction == 0.25
    assert SETUPS["tictactoe"].initial_start_fraction == 1.0


@pytest.mark.parametrize("fraction", [1.0, 0.0])
def test_the_start_rule_is_recorded_and_injected(fraction: float, tmp_path: Path) -> None:
    """초기 국면에서 두 수 만에 지는 일이 없는 변형이라, 다른 국면에서 시작하면 길이가 달라진다."""
    setup = replace(
        small("nano-gungi"),
        game=Gungi(replace(SMALL_NANO, move_limit=2)),
        variant=gungi_variant(SMALL_NANO),
        # 설정의 `end_reason` 은 원래 게임의 것이다.
        end_reason=lambda position: "end",
        initial_start_fraction=fraction,
    )
    logger = MemoryLogger()

    run(setup, logger=logger, generations=1, checkpoint_dir=tmp_path, seed=3)

    assert logger.configuration["self_play_start"] == {"initial_fraction": fraction}
    [(_, metrics)] = logger.records
    # 수 제한이 2 라 초기 국면의 대국은 1~2 수이고, `ply` 1 에서 시작한 대국은 1 수다.
    if fraction == 1.0:
        assert metrics["self_play/length_max"] == 2
    else:
        assert metrics["self_play/length_max"] == 1
