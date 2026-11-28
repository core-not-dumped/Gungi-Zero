"""나노 군의 검증의 합성 루트: 2.5단계 완료 기준 (ADR-0007)."""

import json
from pathlib import Path
from typing import Any

import pytest
import torch

from ai.resnet import ResNet, ResNetConfig
from eval.solver import Solution, Verdict, solve
from eval.trial_positions import Stratum, select_trials
from games.mini_gungi import Gungi
from scripts import verify_nano
from scripts.train import SETUPS
from scripts.verify_nano import NanoVerification, measure, metrics_of
from tests.solver_helpers import SMALL_NANO
from tests.stub_networks import UniformNetwork


@pytest.fixture(scope="module")
def small() -> Solution:
    return solve(Gungi(SMALL_NANO), move_limit=SMALL_NANO.move_limit)


def small_trials(solution: Solution) -> tuple:
    strata = (
        Stratum("draw", 6, lambda position, value: value.verdict is Verdict.DRAW),
        Stratum("win", 4, lambda position, value: value.verdict is Verdict.WIN),
    )
    return select_trials(solution, strata, seed=0)


def test_it_measures_every_stratum_and_the_diagnostics(small: Solution) -> None:
    game = small.game
    torch.manual_seed(0)
    network = ResNet(game.tensor_shape, game.action_space_size, ResNetConfig(1, 4)).eval()

    measured = measure(
        small, small_trials(small), network, simulations=4, device="cpu", best_move_sample=30
    )

    assert [(s.name, s.total) for s in measured.strata] == [("draw", 6), ("win", 4)]
    assert all(0 <= s.passed <= s.total for s in measured.strata)
    assert len(measured.failed_trials) == 10 - sum(s.passed for s in measured.strata)
    assert measured.value_agreement.positions == len(small)
    assert measured.best_move_rates.all.positions == 30


def test_a_sample_larger_than_the_game_takes_every_unfinished_state(small: Solution) -> None:
    network = UniformNetwork(small.game.tensor_shape, small.game.action_space_size)

    measured = measure(
        small,
        small_trials(small)[:1],
        network,
        simulations=1,
        device="cpu",
        best_move_sample=10**9,
    )

    assert 0 < measured.best_move_rates.all.positions < len(small)


def verification(trials_passed: int = 200) -> NanoVerification:
    return NanoVerification(
        strata=(verify_nano.StratumScore("draw/stack", trials_passed, 200),),
        failed_trials=tuple(range(200 - trials_passed)),
        value_agreement=verify_nano.Agreement(1, 2),
        best_move_rates=verify_nano.BestMoveRate(
            verify_nano.Agreement(3, 4), verify_nano.Agreement(1, 4)
        ),
    )


@pytest.mark.parametrize(("trials_passed", "passed"), [(200, True), (199, False)])
def test_the_criterion_needs_every_trial(trials_passed: int, passed: bool) -> None:
    assert verification(trials_passed).passed is passed


def test_the_strata_and_the_diagnostics_are_logged_apart() -> None:
    assert metrics_of(verification(199)) == {
        "trial/draw/stack/passed": 199,
        "trial/draw/stack/total": 200,
        "trial/passed": 199,
        "trial/total": 200,
        "diagnostic/value_agreement": 0.5,
        "diagnostic/best_move/all": 0.5,
        "diagnostic/best_move/stacking": 0.75,
        "diagnostic/best_move/plain": 0.25,
    }


class FakeRun:
    def __init__(self) -> None:
        self.config: dict = {}
        self.logged: list = []
        self.finished = False

    def log(self, metrics: dict, step: int) -> None:
        self.logged.append((metrics, step))

    def finish(self) -> None:
        self.finished = True


@pytest.mark.parametrize(("passed", "status"), [(True, 0), (False, 1)])
def test_the_command_trains_or_loads_then_reports_the_criterion_in_its_status(
    passed: bool,
    status: int,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """풀이·학습·측정은 가짜로 바꾸고, 명령이 그것들을 잇는 방식만 본다."""
    run = FakeRun()
    asked: dict = {}
    setup = SETUPS["nano-gungi"]

    def fake_train(setup: Any, **options: Any) -> torch.nn.Module:
        asked.update(options)
        options["logger"].config({"game": setup.name})
        return torch.nn.Identity()

    result = verification(200 if passed else 199)
    monkeypatch.setattr(verify_nano, "load_solution_of", lambda setup, path: "solution")
    monkeypatch.setattr(verify_nano, "nano_trials", lambda solution: ())
    monkeypatch.setattr(verify_nano, "fingerprint", lambda game, trials: "abc")
    monkeypatch.setattr(verify_nano, "start_wandb", lambda mode: run)
    monkeypatch.setattr(verify_nano, "train_run", fake_train)
    monkeypatch.setattr(verify_nano, "evaluate_network", lambda *args, **kwargs: result)

    arguments = ["--generations", "3", "--seed", "7", "--device", "cpu"]
    arguments += ["--checkpoint-dir", str(tmp_path)]
    assert verify_nano.main(arguments) == status

    assert (asked["generations"], asked["seed"], asked["device"]) == (3, 7, "cpu")
    assert run.finished
    assert run.config["evaluation"]["simulations"] == 100
    assert run.config["evaluation"]["temperature"] == 0
    assert run.config["evaluation"]["trials_fingerprint"] == "abc"
    metrics, step = run.logged[-1]
    assert step == 3
    # 학습 시간은 평가 전에 적는다.
    assert "time/training_seconds" in run.logged[0][0]
    assert metrics["trial/passed"] == (200 if passed else 199)
    summary = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert summary["passed"] is passed
    assert summary["training_seconds"] >= 0
    assert setup.name == run.config["game"]


def test_a_saved_checkpoint_can_be_evaluated_without_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    setup = SETUPS["nano-gungi"]
    game = setup.game
    torch.manual_seed(1)
    saved = ResNet(game.tensor_shape, game.action_space_size, setup.network)
    path = tmp_path / "generation-0042.pt"
    torch.save(saved.state_dict(), path)
    run = FakeRun()
    evaluated: list[torch.nn.Module] = []

    def evaluate(solution: Any, trials: Any, network: torch.nn.Module, **kwargs: Any) -> Any:
        evaluated.append(network)
        return verification()

    def never(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("학습하지 않는다")

    monkeypatch.setattr(verify_nano, "load_solution_of", lambda setup, path: "solution")
    monkeypatch.setattr(verify_nano, "nano_trials", lambda solution: ())
    monkeypatch.setattr(verify_nano, "fingerprint", lambda game, trials: "abc")
    monkeypatch.setattr(verify_nano, "start_wandb", lambda mode: run)
    monkeypatch.setattr(verify_nano, "train_run", never)
    monkeypatch.setattr(verify_nano, "evaluate_network", evaluate)

    assert verify_nano.main(["--checkpoint", str(path), "--device", "cpu"]) == 0

    [network] = evaluated
    assert not network.training
    for name, weights in saved.state_dict().items():
        torch.testing.assert_close(network.state_dict()[name], weights)
    assert run.config["checkpoint"] == str(path)
    assert run.config["variant"] == setup.variant
    assert all("time/training_seconds" not in metrics for metrics, _ in run.logged)
    # 체크포인트의 세대에 적는다.
    assert run.logged[-1][1] == 42
