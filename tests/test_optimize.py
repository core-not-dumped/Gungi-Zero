"""Optuna 튜닝의 합성 루트: 탐색 범위, 계산량 고정, 중간 중단, 함께 쓰는 저장소."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

import optuna
from scripts import optimize
from scripts.optimize import (
    FIRST_TRIALS,
    enqueue_first_trials,
    generations_for,
    open_storage,
    report_schedule,
    suggest,
)
from scripts.shared.setups import SETUPS

NANO = SETUPS["nano-gungi"]
CHOICES = {
    "buffer_generations": 40,
    "episodes_per_generation": 400,
    "simulations": 200,
    "steps_per_generation": 50,
    "learning_rate": 3e-4,
    "c_puct": 1.5,
    "exploration_moves": 6,
    "dirichlet_epsilon": 0.2,
    "dirichlet_alpha": 1.0,
    "network": "6x96",
}


def test_every_suggested_value_reaches_the_setup() -> None:
    overrides = suggest(optuna.trial.FixedTrial(CHOICES))
    setup = optimize.apply_overrides(NANO, overrides)

    assert (setup.network.blocks, setup.network.channels) == (6, 96)
    assert setup.training.buffer_generations == 40
    assert setup.training.episodes_per_generation == 400
    assert setup.training.steps_per_generation == 50
    assert setup.training.learning_rate == 3e-4
    play = setup.training.self_play
    assert (play.simulations, play.c_puct, play.exploration_moves) == (200, 1.5, 6)
    assert (play.root_noise.epsilon, play.root_noise.alpha) == (0.2, 1.0)


def test_the_ranges_are_the_agreed_ones() -> None:
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    seen: dict[str, list] = {}
    for _ in range(300):
        trial = study.ask()
        suggest(trial)
        for name, value in trial.params.items():
            seen.setdefault(name, []).append(value)
        study.tell(trial, 0.0)

    bounds = {
        "buffer_generations": (20, 80),
        "episodes_per_generation": (200, 600),
        "simulations": (100, 300),
        "steps_per_generation": (25, 150),
        "learning_rate": (1e-4, 1e-3),
        "c_puct": (1.0, 3.0),
        "exploration_moves": (0, 10),
        "dirichlet_epsilon": (0.15, 0.35),
        "dirichlet_alpha": (0.5, 2.0),
    }
    for name, (low, high) in bounds.items():
        assert low <= min(seen[name]) <= max(seen[name]) <= high, name
    assert set(seen["network"]) == {"4x64", "6x64", "6x96"}
    assert all(value % 50 == 0 for value in seen["episodes_per_generation"])
    assert set(seen) == set(bounds) | {"network"}


@pytest.mark.parametrize(
    ("episodes", "simulations", "generations"),
    [(200, 100, 200), (400, 100, 100), (600, 300, 22)],
)
def test_the_budget_of_simulations_fixes_the_number_of_generations(
    episodes: int, simulations: int, generations: int
) -> None:
    assert generations_for(4_000_000, episodes, simulations) == generations


def test_a_tiny_budget_still_trains_some_generations() -> None:
    assert generations_for(10, 600, 300) == 4


@pytest.mark.parametrize("generations", [4, 22, 100, 200])
def test_reports_come_at_quarters_and_the_final_score_from_the_end(generations: int) -> None:
    every, reports, final = report_schedule(generations)

    assert [percent for percent, _ in reports] == [25, 50, 75]
    # 보고하는 세대와 채점하는 세대는 모두 저장하는 세대다.
    saved = {g for g in range(1, generations + 1) if g % every == 0 or g == generations}
    assert {g for _, g in reports} <= saved
    assert set(final) <= saved
    assert final == sorted(set(final))
    assert final[-1] == generations
    assert 1 <= len(final) <= 3
    for percent, generation in reports:
        assert abs(generation / generations - percent / 100) <= every / generations


def test_two_workers_share_one_study(tmp_path: Path) -> None:
    path = tmp_path / "study" / "journal.log"
    first = optuna.create_study(study_name="s", storage=open_storage(path), load_if_exists=True)
    second = optuna.create_study(study_name="s", storage=open_storage(path), load_if_exists=True)

    trial = first.ask()
    trial.suggest_int("x", 0, 9)
    first.tell(trial, 1.0)

    assert len(second.trials) == 1
    assert second.ask().number == 1


class FakeRun:
    def __init__(self) -> None:
        self.name = ""
        self.config: dict = {}
        self.logged: list = []
        self.summary: dict = {}
        self.finished = False

    def log(self, metrics: dict, step: int) -> None:
        self.logged.append((metrics, step))

    def finish(self) -> None:
        self.finished = True


def fake_world(monkeypatch: pytest.MonkeyPatch, score) -> dict:
    """풀이·학습·채점·wandb 를 가짜로 바꾼다. `score(trial 번호, 세대)` 가 통과 수다."""
    world: dict[str, Any] = {"runs": [], "setups": [], "graded": []}

    def fake_train(setup: Any, *, logger, generations, checkpoint_dir, **options: Any) -> None:
        world["setups"].append((setup, generations))
        checkpoint_dir.mkdir(parents=True)
        every = setup.training.checkpoint_every
        for generation in range(1, generations + 1):
            if generation % every == 0 or generation == generations:
                (checkpoint_dir / f"generation-{generation:04d}.pt").touch()
            logger.log({"loss/total": 1.0}, step=generation)

    def fake_grade_trials(setup: Any, *args: Any, **kwargs: Any):
        def grade(file: Path) -> dict:
            number = len(world["setups"]) - 1
            generation = int(file.stem.split("-")[1])
            world["graded"].append((number, generation))
            return {"trials": {"passed": score(number, generation), "total": 200}}

        return grade

    def fake_wandb(*args: Any, **kwargs: Any) -> FakeRun:
        run = FakeRun()
        world["runs"].append(run)
        return run

    monkeypatch.setattr(optimize, "train_run", fake_train)
    monkeypatch.setattr(optimize, "grade_trials", fake_grade_trials)
    monkeypatch.setattr(optimize, "start_wandb", fake_wandb)
    monkeypatch.setattr(optimize, "load_solution_of", lambda setup, path: "solution")
    monkeypatch.setattr(optimize, "cached_dev_trials", lambda *args: ("dev",))
    return world


def arguments(tmp_path: Path, *more: str) -> list[str]:
    return [
        "--journal",
        str(tmp_path / "journal.log"),
        "--checkpoint-root",
        str(tmp_path / "checkpoints"),
        "--budget",
        "400000",
        "--device",
        "cpu",
        *more,
    ]


def test_trials_are_scored_by_the_last_checkpoints_and_keep_only_the_final_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = fake_world(monkeypatch, lambda number, generation: 100 + generation)

    assert optimize.main(arguments(tmp_path, "--trials", "2")) == 0

    study = optuna.load_study(
        study_name="nano-optuna", storage=open_storage(tmp_path / "journal.log")
    )
    assert [t.state.name for t in study.trials] == ["COMPLETE", "COMPLETE"]
    for trial, (setup, generations), run in zip(
        study.trials, world["setups"], world["runs"], strict=True
    ):
        episodes = setup.training.episodes_per_generation
        simulations = setup.training.self_play.simulations
        assert generations == generations_for(400_000, episodes, simulations)
        _, _, final = report_schedule(generations)
        assert trial.value == pytest.approx(100 + sum(final) / len(final))
        assert trial.user_attrs["generations"] == generations
        assert run.finished
        assert run.name.startswith(f"t{trial.number:03d}")
        assert run.summary["dev/passed"] == trial.value
        [kept] = Path(trial.user_attrs["checkpoint_dir"]).glob("*.pt")
        assert kept.stem.endswith(f"{generations:04d}")
    # 처음 두 시도는 미리 넣어 둔 설정이다.
    assert study.trials[0].params["buffer_generations"] == 50
    assert study.trials[1].params["buffer_generations"] == 20


def test_a_second_worker_adds_trials_up_to_the_total(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_world(monkeypatch, lambda number, generation: 150)
    optimize.main(arguments(tmp_path, "--trials", "2"))

    optimize.main(arguments(tmp_path, "--trials", "3"))

    study = optuna.load_study(
        study_name="nano-optuna", storage=open_storage(tmp_path / "journal.log")
    )
    assert len(study.trials) == 3


def test_a_trial_far_behind_the_others_is_pruned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 여덟 번의 준비 뒤의 시도만 아주 낮다.
    world = fake_world(monkeypatch, lambda number, generation: 170 if number < 8 else 10)

    optimize.main(arguments(tmp_path, "--trials", "9"))

    study = optuna.load_study(
        study_name="nano-optuna", storage=open_storage(tmp_path / "journal.log")
    )
    pruned = study.trials[8]
    assert pruned.state.name == "PRUNED"
    # 첫 보고에서 끊어 끝까지 학습하지 않는다.
    assert [g for number, g in world["graded"] if number == 8] == [
        report_schedule(world["setups"][8][1])[1][0][1]
    ]
    assert world["runs"][8].finished


def test_a_worker_whose_trials_keep_failing_stops(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(number: int, generation: int) -> int:
        raise RuntimeError("GPU 가 없다")

    fake_world(monkeypatch, broken)

    optimize.main(arguments(tmp_path, "--trials", "20"))

    study = optuna.load_study(
        study_name="nano-optuna", storage=open_storage(tmp_path / "journal.log")
    )
    assert [t.state.name for t in study.trials if t.state.name != "WAITING"] == ["FAIL"] * 3


def test_workers_starting_together_queue_the_first_trials_once(tmp_path: Path) -> None:
    path = tmp_path / "journal.log"
    optuna.create_study(study_name="s", storage=open_storage(path))

    def worker(_: int) -> None:
        study = optuna.load_study(study_name="s", storage=open_storage(path))
        enqueue_first_trials(study, path)

    with ThreadPoolExecutor(6) as pool:
        list(pool.map(worker, range(6)))

    study = optuna.load_study(study_name="s", storage=open_storage(path))
    assert len(study.trials) == len(FIRST_TRIALS)
