"""저장한 세대를 빠르게 채점하는 합성 루트."""

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch

from ai.resnet import ResNet
from eval.solver import Solution, Verdict, solve
from eval.trial_positions import Stratum, select_trials
from games.mini_gungi import Gungi
from scripts import grade
from scripts.openings import Openings
from scripts.train import SETUPS
from tests.solver_helpers import SMALL_NANO


def save_network(game_name: str, path: Path, seed: int = 0) -> Path:
    setup = SETUPS[game_name]
    torch.manual_seed(seed)
    network = ResNet(setup.game.tensor_shape, setup.game.action_space_size, setup.network)
    torch.save(network.state_dict(), path)
    return path


def two_openings(setup: Any, count: int, solution: Path | None = None) -> Openings:
    game = setup.game
    start = game.initial_position()
    moves = game.legal_moves(start)[:count]
    return Openings([game.apply_move(start, move) for move in moves], "balanced", "abc")


def lines(capsys: pytest.CaptureFixture[str]) -> list[dict]:
    return [json.loads(line) for line in capsys.readouterr().out.splitlines()]


def test_an_unsolved_game_is_graded_by_matches_against_random_and_a_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    first = save_network("mini-gungi", tmp_path / "generation-0001.pt", 1)
    second = save_network("mini-gungi", tmp_path / "generation-0002.pt", 2)
    monkeypatch.setattr(grade, "openings_for", two_openings)

    arguments = ["--game", "mini-gungi", "--checkpoint", str(first), str(second)]
    arguments += ["--opponent", str(first), "--openings", "2", "--simulations", "2"]
    assert grade.main([*arguments, "--device", "cpu"]) == 0

    results = lines(capsys)
    assert [result["checkpoint"] for result in results] == [str(first), str(second)]
    for result in results:
        assert result["game"] == "mini-gungi"
        assert result["openings"] == {"source": "balanced", "count": 2, "fingerprint": "abc"}
        assert [match["opponent"] for match in result["matches"]] == ["random", str(first)]
        for match in result["matches"]:
            assert match["wins"] + match["draws"] + match["losses"] == 4
            assert match["score"] == (match["wins"] + match["draws"] / 2) / 4


@pytest.fixture(scope="module")
def small() -> Solution:
    return solve(Gungi(SMALL_NANO), move_limit=SMALL_NANO.move_limit)


def test_a_solved_game_is_graded_on_its_trial_positions(
    small: Solution,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    strata = (
        Stratum("draw", 3, lambda position, value: value.verdict is Verdict.DRAW),
        Stratum("win", 2, lambda position, value: value.verdict is Verdict.WIN),
    )
    trials = select_trials(small, strata, seed=0)
    monkeypatch.setattr(grade, "load_solution_of", lambda setup, path: small)
    monkeypatch.setattr(grade, "nano_trials", lambda solution: trials)
    monkeypatch.setattr(grade, "fingerprint", lambda game, trials: "abc")
    game = small.game
    network = ResNet(game.tensor_shape, game.action_space_size, SETUPS["nano-gungi"].network)
    path = tmp_path / "generation-0005.pt"
    torch.save(network.state_dict(), path)
    # 작은 변형의 신경망을 읽도록 게임을 바꿔 끼운다.
    setup = SETUPS["nano-gungi"]
    monkeypatch.setitem(grade.SETUPS, "nano-gungi", replace(setup, game=game))

    arguments = ["--checkpoint", str(path), "--simulations", "2", "--device", "cpu"]
    assert grade.main(arguments) == 0

    [result] = lines(capsys)
    assert result["trials"]["total"] == 5
    assert result["trials"]["passed"] == 5 - len(result["trials"]["failed"])
    assert [(s["name"], s["total"]) for s in result["trials"]["strata"]] == [
        ("draw", 3),
        ("win", 2),
    ]
    assert result["trials"]["fingerprint"] == "abc"


def test_a_missing_checkpoint_is_refused() -> None:
    with pytest.raises(SystemExit):
        grade.main(["--checkpoint", "none.pt"])
