"""저장한 세대들의 Elo 를 매기는 합성 루트."""

import json
from pathlib import Path
from typing import Any

import pytest
import torch

from ai.resnet import ResNet
from scripts import rate
from scripts.openings import Openings, spread
from scripts.train import SETUPS

SETUP = SETUPS["nano-gungi"]


def save_generations(directory: Path, numbers: tuple[int, ...]) -> None:
    for number in numbers:
        torch.manual_seed(number)
        network = ResNet(SETUP.game.tensor_shape, SETUP.game.action_space_size, SETUP.network)
        torch.save(network.state_dict(), directory / f"generation-{number:04d}.pt")


def some_openings(setup: Any, count: int, solution: Path | None = None) -> Openings:
    """풀이 대신 초기 국면에서 한 수를 둔 판들."""
    game = setup.game
    start = game.initial_position()
    moves = game.legal_moves(start)[:count]
    return Openings([game.apply_move(start, move) for move in moves], "trial_positions", "abc")


@pytest.fixture(autouse=True)
def without_a_solution(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rate, "openings_for", some_openings)


def test_spread_takes_evenly_spaced_items() -> None:
    assert spread(range(200), 4) == [0, 50, 100, 150]
    assert spread("abcde", 5) == list("abcde")


@pytest.mark.parametrize("count", [0, 6])
def test_spread_refuses_a_count_it_cannot_take(count: int) -> None:
    with pytest.raises(ValueError):
        spread("abcde", count)


def test_another_game_is_rated_with_its_own_setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mini = SETUPS["mini-gungi"]
    torch.manual_seed(0)
    network = ResNet(mini.game.tensor_shape, mini.game.action_space_size, mini.network)
    torch.save(network.state_dict(), tmp_path / "generation-0005.pt")
    run = FakeRun()
    monkeypatch.setattr(rate, "start_wandb", lambda mode: run)

    arguments = ["--game", "mini-gungi", "--checkpoint-dir", str(tmp_path), "--device", "cpu"]
    assert rate.main([*arguments, "--openings", "1", "--simulations", "2"]) == 0

    assert run.config["game"] == "mini-gungi"
    assert run.config["variant"] == mini.variant
    assert [step for _, step in run.logged] == [5]


class FakeRun:
    def __init__(self) -> None:
        self.config: dict = {}
        self.logged: list = []
        self.finished = False

    def log(self, metrics: dict, step: int) -> None:
        self.logged.append((metrics, step))

    def finish(self) -> None:
        self.finished = True


def test_it_logs_one_rating_per_generation_in_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    save_generations(tmp_path, (1, 2, 3))
    run = FakeRun()
    monkeypatch.setattr(rate, "start_wandb", lambda mode: run)

    arguments = ["--checkpoint-dir", str(tmp_path), "--device", "cpu"]
    arguments += ["--openings", "2", "--simulations", "2"]
    assert rate.main(arguments) == 0

    assert [step for _, step in run.logged] == [1, 2, 3]
    assert all("elo/rating" in metrics for metrics, _ in run.logged)
    assert run.finished
    # 종료 규칙이 다른 학습의 Elo 와 섞이지 않게 남긴다 (ADR-0004).
    assert run.config["variant"] == SETUP.variant
    assert run.config["elo"]["anchor"] == "random"
    assert run.config["elo"]["openings"] == 2
    assert run.config["elo"]["openings_from"] == "trial_positions"
    assert run.config["elo"]["openings_fingerprint"] == "abc"
    assert run.config["elo"]["simulations"] == 2
    assert run.config["checkpoint_dir"] == str(tmp_path)
    summary = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert [generation for generation, _ in summary["ratings"]] == [1, 2, 3]
    assert [rating for _, rating in summary["ratings"]] == [
        metrics["elo/rating"] for metrics, _ in run.logged
    ]
    # 대진마다 시작 판의 두 배를 둔다.
    assert all(pair["wins"] + pair["draws"] + pair["losses"] == 4 for pair in summary["pairs"])


def test_checkpoints_kept_every_few_generations_are_rated_at_their_numbers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    save_generations(tmp_path, (5, 10, 12))
    run = FakeRun()
    monkeypatch.setattr(rate, "start_wandb", lambda mode: run)

    arguments = ["--checkpoint-dir", str(tmp_path), "--device", "cpu"]
    assert rate.main([*arguments, "--openings", "1", "--simulations", "2"]) == 0

    assert [step for _, step in run.logged] == [5, 10, 12]
    summary = json.loads(capsys.readouterr().out.splitlines()[-1])
    played = {pair[side] for pair in summary["pairs"] for side in ("player", "opponent")}
    assert played == {0, 5, 10, 12}


def test_a_folder_without_checkpoints_is_refused_before_wandb(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:

    def never(mode: Any) -> None:
        raise AssertionError("run 을 열지 않는다")

    monkeypatch.setattr(rate, "start_wandb", never)

    with pytest.raises(SystemExit):
        rate.main(["--checkpoint-dir", str(tmp_path), "--device", "cpu"])


@pytest.mark.parametrize("workers", [1, 2])
def test_only_the_chosen_generations_are_rated_by_any_number_of_workers(
    workers: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    save_generations(tmp_path, (5, 10, 15, 20))
    run = FakeRun()
    monkeypatch.setattr(rate, "start_wandb", lambda mode: run)

    arguments = ["--checkpoint-dir", str(tmp_path), "--device", "cpu", "--openings", "1"]
    arguments += ["--simulations", "2", "--generations", "20", "5", "--workers", str(workers)]
    assert rate.main(arguments) == 0

    assert [step for _, step in run.logged] == [5, 20]
    summary = json.loads(capsys.readouterr().out.splitlines()[-1])
    # 성적은 끝난 순서가 아니라 대진 순서다.
    assert [(p["player"], p["opponent"]) for p in summary["pairs"]] == [(5, 0), (20, 0), (20, 5)]


def test_a_generation_without_a_checkpoint_is_refused(tmp_path: Path) -> None:
    save_generations(tmp_path, (5,))

    with pytest.raises(SystemExit):
        rate.main(["--checkpoint-dir", str(tmp_path), "--generations", "6"])
