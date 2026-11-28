"""튜닝 실험의 합성 루트, 설정 덮어쓰기, run 이름."""

from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from scripts import tune
from scripts.overrides import apply_overrides
from scripts.run_names import evaluation_name, rating_name, training_name, tuning_name
from scripts.train import SETUPS

NANO = SETUPS["nano-gungi"]


def test_overrides_reach_the_training_and_the_self_play() -> None:
    changed = apply_overrides(
        NANO, {"steps_per_generation": 50, "c_puct": 2.5, "exploration_moves": 200}
    )

    assert changed.training.steps_per_generation == 50
    assert changed.training.self_play.c_puct == 2.5
    assert changed.training.self_play.exploration_moves == 200
    assert changed.training.batch_size == NANO.training.batch_size
    assert NANO.training.steps_per_generation != 50


def test_overrides_reach_the_network_the_noise_and_the_starts() -> None:
    changed = apply_overrides(
        NANO, {"blocks": 2, "channels": 32, "dirichlet_epsilon": 0.4, "initial_start_fraction": 0.5}
    )

    assert (changed.network.blocks, changed.network.channels) == (2, 32)
    assert changed.training.self_play.root_noise.epsilon == 0.4
    assert changed.training.self_play.root_noise.alpha == NANO.training.self_play.root_noise.alpha
    assert changed.initial_start_fraction == 0.5


def test_an_unknown_name_is_refused() -> None:
    with pytest.raises(ValueError, match="optimizer"):
        apply_overrides(NANO, {"optimizer": 1})


def test_every_experiment_makes_a_setup_that_differs_from_the_base_in_one_value() -> None:
    for name, overrides in tune.EXPERIMENTS.items():
        assert len(overrides) == (0 if name == "base" else 1), name
        assert (apply_overrides(NANO, overrides) == NANO) is (name == "base"), name


def test_run_names_say_what_the_run_did() -> None:
    when = datetime(2026, 11, 28, 7, 5)
    checkpoint = Path("checkpoints/nano-gungi-seed0-night1/generation-1200.pt")

    assert (
        training_name("nano-gungi", seed=0, episodes=200, steps=200, started=when)
        == "train · nano-gungi · 200판/200스텝 · s0 · 11-28 07:05"
    )
    assert (
        evaluation_name("nano-gungi", checkpoint) == "eval · nano-gungi · seed0-night1 · 1200세대"
    )
    assert rating_name("nano-gungi", checkpoint.parent) == "elo · nano-gungi · seed0-night1"
    assert tuning_name("steps50", 2) == "steps50 · s2"


def test_the_sweep_file_lists_the_experiments_of_the_command() -> None:
    # wandb 는 이 파일을 Windows 의 코드 페이지(cp949)로 읽는다.
    text = Path("sweeps/nano-wide.yaml").read_text(encoding="ascii")

    assert f"values: [{', '.join(tune.EXPERIMENTS)}]" in text
    assert "name: dev/passed" in text
    assert "scripts.tune" in text
    # agent 가 띄운 `python` 은 의존성이 없는 인터프리터다.
    command = text.split("command:")[1].split()
    assert command[:6] == ["-", "uv", "-", "run", "-", "python"]


class FakeRun:
    def __init__(self) -> None:
        self.name = ""
        self.config: dict = {}
        self.logged: list = []
        self.finished = False

    def log(self, metrics: dict, step: int) -> None:
        self.logged.append((metrics, step))

    def finish(self) -> None:
        self.finished = True


def test_one_experiment_trains_with_its_values_and_logs_the_mean_dev_score(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """풀이·학습·채점은 가짜로 바꾸고, 명령이 그것들을 잇는 방식만 본다."""
    run = FakeRun()
    trained: dict = {}
    scores = iter([100, 110, 126])

    def fake_train(setup: Any, **options: Any) -> None:
        trained.update(options, setup=setup)
        directory = options["checkpoint_dir"]
        directory.mkdir(parents=True)
        for generation in (5, 10, 15, 20):
            (directory / f"generation-{generation:04d}.pt").touch()

    def fake_grade(player: Any) -> dict:
        passed = next(scores)
        return {
            "trials": {
                "passed": passed,
                "total": 200,
                "strata": [{"name": "draw/stack", "passed": passed // 2, "total": 100}],
            }
        }

    monkeypatch.setattr(tune, "start_wandb", lambda mode: run)
    monkeypatch.setattr(tune, "load_solution_of", lambda setup, path: "solution")
    monkeypatch.setattr(tune, "cached_dev_trials", lambda *args: ("dev",))
    monkeypatch.setattr(tune, "train_run", fake_train)
    monkeypatch.setattr(tune, "grade_trials", lambda *args, **kwargs: fake_grade)
    monkeypatch.setattr(tune, "checkpoint_player", lambda *args, **kwargs: None)
    threads: list[int] = []
    monkeypatch.setattr(tune.torch, "set_num_threads", threads.append)

    arguments = ["--experiment=steps50", "--seed=2", "--generations=20", "--batch_size=64"]
    assert tune.main([*arguments, "--checkpoint-root", str(tmp_path)]) == 0

    training = trained["setup"].training
    assert (training.steps_per_generation, training.batch_size) == (50, 64)
    assert (trained["seed"], trained["generations"]) == (2, 20)
    assert run.name == "steps50 · s2"
    assert run.config["overrides"] == {"steps_per_generation": 50, "batch_size": 64}
    metrics, step = run.logged[-1]
    assert step == 20
    # 마지막 세 체크포인트의 평균이다.
    assert metrics["dev/passed"] == 112
    assert metrics["dev/passed_last"] == 126
    assert metrics["dev/draw/stack"] == (50 + 55 + 63) / 3
    assert run.finished
    # 채점은 스레드 하나로 한다.
    assert threads == [1]
