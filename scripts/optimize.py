"""나노 군의의 하이퍼파라미터를 Optuna 로 찾는 합성 루트.

uv run python -m scripts.optimize --trials 100      # 작업자 하나. 여러 터미널에서 함께 띄운다
uv run python -m scripts.optimize --report          # 지금까지의 상위 시도

같은 `--journal` 파일을 쓰는 작업자들은 한 study 를 나눠 한다. `--trials` 는 작업자마다가 아니라
study 전체에서 끝낸(완료와 중단) 시도의 수이고, 함께 돌던 작업자들 때문에 조금 넘을 수 있다.

- 탐색 범위는 `suggest` 에 있고, 근거는 시드 하나의 일변수 실험이다
  (`docs/hyperparameter-search.md`).
- 시도마다의 계산량을 맞춘다: 자가 대국 수 x 탐색 횟수 x 세대 수가 `--budget` 이 되게 세대 수를
  정한다. 학습 스텝은 맞추지 않는다.
- 점수는 연습용 판(`scripts.shared.dev_trials`)의 통과 수다. 계산량의 25·50·75% 에서 보고하고,
  뒤처지는 시도는 중단한다. 마지막 점수는 끝의 체크포인트 셋의 평균이다.
- 시도마다 wandb run 하나를 남긴다. 체크포인트는 마지막 것만 둔다.

모든 시도의 학습 시드가 같다. 상위 설정은 시드를 바꿔 다시 확인하고, 시험 판은 고른 설정에만 쓴다.
"""

from __future__ import annotations

import argparse
import gc
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import torch
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock

import optuna
from ai.checkpoint import checkpoint_path
from scripts.shared.dev_trials import DEV_CACHE, cached_dev_trials
from scripts.shared.grading import grade_trials
from scripts.shared.loggers import WANDB_PROJECT, PrintingLogger, TeeLogger, WandbLogger
from scripts.shared.overrides import apply_overrides
from scripts.shared.setups import SETUPS, GameSetup
from scripts.shared.solution import SOLUTION_PATH, load_solution_of
from scripts.shared.training import CHECKPOINT_ROOT
from scripts.shared.training import run as train_run

GAME = "nano-gungi"
# git 에 넣지 않는다.
JOURNAL_PATH = Path("optuna/nano.journal")
STUDY_NAME = "nano-optuna"
# 시도마다의 자가 대국 탐색 횟수의 합. 튜닝의 기준 설정(200판 x 100회 x 100세대)의 두 배다.
BUDGET = 4_000_000
# 세대가 너무 적으면 학습이라 할 수 없다.
MINIMUM_GENERATIONS = 4
NETWORKS = {"4x64": (4, 64), "6x64": (6, 64), "6x96": (6, 96)}
# 중간 중단: 이만큼 끝난 시도가 쌓인 뒤부터, 같은 지점의 중앙값보다 낮으면 중단한다.
PRUNER_STARTUP_TRIALS = 8
# 한 작업자에서 이만큼 잇달아 실패하면 설정이 아니라 실행 환경의 문제로 보고 멈춘다.
FAILURES_IN_A_ROW = 3
# 처음에 시험해 보는 설정. 일변수 실험에서 좋았던 값을 합친 것과, 지금 설정에 가까운 것.
FIRST_TRIALS: tuple[dict[str, Any], ...] = (
    {
        "buffer_generations": 50,
        "episodes_per_generation": 400,
        "simulations": 100,
        "steps_per_generation": 50,
        "learning_rate": 3e-4,
        "c_puct": 1.25,
        "exploration_moves": 4,
        "dirichlet_epsilon": 0.25,
        "dirichlet_alpha": 1.09,
        "network": "4x64",
    },
    {
        "buffer_generations": 20,
        "episodes_per_generation": 200,
        "simulations": 100,
        "steps_per_generation": 150,
        "learning_rate": 1e-3,
        "c_puct": 1.25,
        "exploration_moves": 4,
        "dirichlet_epsilon": 0.25,
        "dirichlet_alpha": 1.09,
        "network": "4x64",
    },
)


def suggest(trial: optuna.trial.BaseTrial) -> dict[str, float]:
    """시도의 값을 고른다. `scripts.shared.overrides` 의 이름으로 돌려준다."""
    values: dict[str, float] = {
        "buffer_generations": trial.suggest_int("buffer_generations", 20, 80),
        "episodes_per_generation": trial.suggest_int("episodes_per_generation", 200, 600, step=50),
        "simulations": trial.suggest_int("simulations", 100, 300, step=25),
        "steps_per_generation": trial.suggest_int("steps_per_generation", 25, 150, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True),
        "c_puct": trial.suggest_float("c_puct", 1.0, 3.0),
        "exploration_moves": trial.suggest_int("exploration_moves", 0, 10),
        "dirichlet_epsilon": trial.suggest_float("dirichlet_epsilon", 0.15, 0.35),
        "dirichlet_alpha": trial.suggest_float("dirichlet_alpha", 0.5, 2.0, log=True),
    }
    blocks, channels = NETWORKS[trial.suggest_categorical("network", list(NETWORKS))]
    return {**values, "blocks": blocks, "channels": channels}


def generations_for(budget: int, episodes: float, simulations: float) -> int:
    """계산량 `budget` 을 쓰는 세대 수."""
    return max(MINIMUM_GENERATIONS, round(budget / (episodes * simulations)))


def report_schedule(generations: int) -> tuple[int, list[tuple[int, int]], list[int]]:
    """`(저장 간격, [(백분율, 보고하는 세대)], 마지막 점수를 내는 세대들)`. 모두 저장하는 세대다."""
    every = max(1, generations // 20)
    reports = [
        (percent, max(every, round(generations * percent / 100 / every) * every))
        for percent in (25, 50, 75)
    ]
    saved = [g for g in range(every, generations + 1, every)]
    if saved[-1] != generations:
        saved.append(generations)
    return every, reports, saved[-3:]


def open_storage(path: Path) -> JournalStorage:
    """작업자들이 함께 쓰는 파일 저장소. Windows 에는 심볼릭 링크 잠금이 안 되어 열기 잠금이다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return JournalStorage(JournalFileBackend(str(path), lock_obj=JournalFileOpenLock(str(path))))


def enqueue_first_trials(study: optuna.Study, journal: Path) -> None:
    """`FIRST_TRIALS` 를 study 에 한 번만 넣는다. 함께 시작한 작업자들이 서로 보기 전에 넣으면 같은
    설정이 두 번 돌아, 잠그고 확인한다.
    """
    lock = JournalFileOpenLock(f"{journal}.first-trials")
    lock.acquire()
    try:
        for params in FIRST_TRIALS:
            study.enqueue_trial(params, skip_if_exists=True)
    finally:
        lock.release()


def start_wandb(mode: Literal["online", "offline", "disabled"], study: str) -> Any:
    """wandb run 을 연다. torch 처럼 무거워 쓸 때 import 한다."""
    import wandb

    return wandb.init(
        project=WANDB_PROJECT, job_type="optuna", group=study, mode=mode, reinit="finish_previous"
    )


def run_name(number: int, params: Mapping[str, Any]) -> str:
    return (
        f"t{number:03d} · 버퍼{params['buffer_generations']}"
        f" 판{params['episodes_per_generation']} 탐색{params['simulations']}"
        f" 스텝{params['steps_per_generation']} · {params['network']}"
    )


class _Reporting:
    """학습 기록을 넘기면서, 보고하는 세대에서 채점해 Optuna 에 알리고 뒤처지면 중단한다."""

    def __init__(
        self,
        logger: TeeLogger,
        trial: optuna.Trial,
        reports: Mapping[int, int],
        score: Callable[[int], float],
    ) -> None:
        self._logger = logger
        self._trial = trial
        # 세대 → 백분율.
        self._reports = {generation: percent for percent, generation in reports.items()}
        self._score = score

    def config(self, values: Mapping[str, Any]) -> None:
        self._logger.config(values)

    def log(self, metrics: Mapping[str, float], step: int) -> None:
        percent = self._reports.get(step)
        if percent is None:
            self._logger.log(metrics, step)
            return
        # 체크포인트는 기록보다 먼저 저장된다.
        passed = self._score(step)
        self._logger.log({**metrics, "dev/passed_so_far": passed}, step)
        self._trial.report(passed, percent)
        if self._trial.should_prune():
            raise optuna.TrialPruned(f"{percent}% 에서 {passed}")


def _objective(
    args: argparse.Namespace, solution: Any, trials: Any
) -> Callable[[optuna.Trial], float]:
    def objective(trial: optuna.Trial) -> float:
        overrides = suggest(trial)
        generations = generations_for(
            args.budget, overrides["episodes_per_generation"], overrides["simulations"]
        )
        every, reports, final = report_schedule(generations)
        setup: GameSetup = apply_overrides(SETUPS[GAME], {**overrides, "checkpoint_every": every})
        started = datetime.now()
        directory = args.checkpoint_root / f"t{trial.number:03d}-{started:%Y%m%d-%H%M%S}"
        trial.set_user_attr("generations", generations)
        trial.set_user_attr("checkpoint_dir", str(directory))

        wandb_run = start_wandb(args.wandb, args.study)
        wandb_run.name = run_name(trial.number, trial.params)
        logger = TeeLogger(PrintingLogger(), WandbLogger(wandb_run))
        try:
            logger.config(
                {
                    "trial": trial.number,
                    "params": dict(trial.params),
                    "budget": args.budget,
                    "checkpoint_dir": str(directory),
                }
            )
            grade = grade_trials(
                setup,
                args.solution,
                simulations=setup.evaluation_simulations,
                device=args.device,
                trials=trials,
                solution=solution,
            )

            def score(generation: int) -> float:
                return grade(checkpoint_path(directory, generation))["trials"]["passed"]

            train_run(
                setup,
                logger=_Reporting(logger, trial, dict(reports), score),
                generations=generations,
                checkpoint_dir=directory,
                seed=args.seed,
                device=args.device,
            )
            scores = [score(generation) for generation in final]
            value = sum(scores) / len(scores)
            logger.log({"dev/passed": value, "dev/passed_last": scores[-1]}, step=generations)
            wandb_run.summary["dev/passed"] = value
            # 시도마다 파일이 쌓이지 않게 마지막 것만 둔다.
            for file in sorted(directory.glob("generation-*.pt"))[:-1]:
                file.unlink()
            return value
        except optuna.TrialPruned:
            wandb_run.summary["pruned"] = True
            for file in sorted(directory.glob("generation-*.pt"))[:-1]:
                file.unlink()
            raise
        finally:
            wandb_run.finish()
            # 다음 시도가 쓸 GPU 메모리를 돌려놓는다.
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return objective


def stop_after_failures() -> Callable[[optuna.Study, optuna.trial.FrozenTrial], None]:
    """실패가 `FAILURES_IN_A_ROW` 번 이어지면 이 작업자를 멈추는 콜백."""
    failures = 0

    def callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        nonlocal failures
        failures = failures + 1 if trial.state is optuna.trial.TrialState.FAIL else 0
        if failures >= FAILURES_IN_A_ROW:
            study.stop()

    return callback


def report(study: optuna.Study, top: int = 10) -> str:
    """상태별 시도 수와 점수가 높은 시도들."""
    counts: dict[str, int] = {}
    for trial in study.trials:
        counts[trial.state.name] = counts.get(trial.state.name, 0) + 1
    lines = [f"시도 {len(study.trials)}개 {counts}"]
    done = [t for t in study.trials if t.state is optuna.trial.TrialState.COMPLETE]
    for trial in sorted(done, key=lambda t: t.value or 0, reverse=True)[:top]:
        values = ", ".join(
            f"{name}={value:.3g}" if isinstance(value, float) else f"{name}={value}"
            for name, value in trial.params.items()
        )
        generations = trial.user_attrs.get("generations")
        lines.append(f"t{trial.number:03d} {trial.value:.1f} (세대 {generations}): {values}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Optuna 로 하이퍼파라미터를 찾는다")
    parser.add_argument("--study", default=STUDY_NAME)
    parser.add_argument("--journal", type=Path, default=JOURNAL_PATH)
    parser.add_argument("--trials", type=int, default=100, help="study 전체에서 끝낼 시도 수")
    parser.add_argument("--budget", type=int, default=BUDGET, help="시도마다의 자가 대국 탐색 횟수")
    parser.add_argument("--seed", type=int, default=0, help="학습의 시드")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--solution", type=Path, default=SOLUTION_PATH)
    parser.add_argument("--checkpoint-root", type=Path, default=CHECKPOINT_ROOT / "optuna")
    parser.add_argument("--wandb", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--report", action="store_true", help="실험하지 않고 상위 시도를 찍는다")
    args = parser.parse_args(argv)
    if min(args.trials, args.budget) < 1:
        parser.error("--trials 와 --budget 은 1 이상이다")

    storage = open_storage(args.journal)
    if args.report:
        print(report(optuna.load_study(study_name=args.study, storage=storage)))
        return 0
    study = optuna.create_study(
        study_name=args.study,
        storage=storage,
        direction="maximize",
        load_if_exists=True,
        # 함께 도는 작업자가 같은 곳을 고르지 않게, 돌고 있는 시도를 나쁜 결과로 친다.
        sampler=optuna.samplers.TPESampler(constant_liar=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=PRUNER_STARTUP_TRIALS),
    )
    # 계산량이 다른 시도는 한 study 에서 견줄 수 없다.
    study_budget = study.user_attrs.get("budget")
    if study_budget is None:
        study.set_user_attr("budget", args.budget)
    elif study_budget != args.budget:
        parser.error(f"이 study 의 --budget 은 {study_budget} 이다")
    enqueue_first_trials(study, args.journal)

    try:
        solution = load_solution_of(SETUPS[GAME], args.solution)
        trials = cached_dev_trials(solution, args.solution, DEV_CACHE)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    torch.set_num_threads(1)
    finished = (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
    if len(study.get_trials(deepcopy=False, states=finished)) < args.trials:
        study.optimize(
            _objective(args, solution, trials),
            callbacks=[
                optuna.study.MaxTrialsCallback(args.trials, states=finished),
                stop_after_failures(),
            ],
            # GPU 메모리 부족 같은 실패는 그 시도만 실패로 두고 이어 간다.
            catch=(RuntimeError,),
        )
    print(report(study))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
