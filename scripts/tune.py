"""튜닝 실험 한 번: 설정을 덮어써 학습하고 연습용 판으로 채점해 wandb 에 적는다.

uv run wandb sweep sweeps/nano-wide.yaml        # 한 번. sweep ID 를 찍는다
uv run wandb agent <sweep ID>                 # 남은 실험을 차례로 돌린다

한 번만 돌려 보려면 `uv run python -m scripts.tune --experiment steps50 --seed 0` 이다.

채점은 시험 판 200개가 아니라 연습용 판(`scripts.shared.dev_trials`)이다. 점수는 마지막 체크포인트
`--graded`개의 통과 수 평균(`dev/passed`)이고, 완료 기준의 판정은 하지 않는다. 실험 계획은
`docs/hyperparameter-search.md` 에 있다.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import torch

from ai.checkpoint import generation_files
from scripts.shared.dev_trials import DEV_CACHE, DEV_TRIAL_SEED, cached_dev_trials
from scripts.shared.grading import grade_trials
from scripts.shared.loggers import WANDB_PROJECT, PrintingLogger, TeeLogger, WandbLogger
from scripts.shared.overrides import OVERRIDABLE, apply_overrides
from scripts.shared.run_names import tuning_name
from scripts.shared.setups import SETUPS
from scripts.shared.solution import SOLUTION_PATH, load_solution_of
from scripts.shared.training import CHECKPOINT_ROOT
from scripts.shared.training import run as train_run

GAME = "nano-gungi"

# 이름 → 지금 설정에서 바꾸는 값. 한 번에 한 가지씩 바꾼다.
EXPERIMENTS: dict[str, dict[str, float]] = {
    "base": {},
    "steps50": {"steps_per_generation": 50},
    "steps400": {"steps_per_generation": 400},
    "buffer30": {"buffer_generations": 30},
    "buffer50": {"buffer_generations": 50},
    "episodes100": {"episodes_per_generation": 100},
    "episodes400": {"episodes_per_generation": 400},
    "sims50": {"simulations": 50},
    "sims200": {"simulations": 200},
    "cpuct0.75": {"c_puct": 0.75},
    "cpuct2.5": {"c_puct": 2.5},
    "explore-all": {"exploration_moves": 200},
    "noise0.4": {"dirichlet_epsilon": 0.4},
    "lr3e-4": {"learning_rate": 3e-4},
    "lr3e-3": {"learning_rate": 3e-3},
    "batch256": {"batch_size": 256},
    "net2x64": {"blocks": 2},
    "net4x32": {"channels": 32},
}


def start_wandb(mode: Literal["online", "offline", "disabled"]) -> Any:
    """wandb run 을 연다. torch 처럼 무거워 쓸 때 import 한다."""
    import wandb

    return wandb.init(project=WANDB_PROJECT, job_type="tune", mode=mode)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="튜닝 실험 한 번")
    parser.add_argument("--experiment", choices=tuple(EXPERIMENTS), default="base")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument(
        "--graded", type=int, default=3, help="채점해 평균하는 마지막 체크포인트 수"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--solution", type=Path, default=SOLUTION_PATH)
    parser.add_argument("--checkpoint-root", type=Path, default=CHECKPOINT_ROOT / "tune")
    parser.add_argument("--wandb", choices=("online", "offline", "disabled"), default="online")
    for name, kind in OVERRIDABLE.items():
        parser.add_argument(f"--{name}", type=kind, help="이 값을 덮어쓴다")
    args = parser.parse_args(argv)
    if min(args.generations, args.graded) < 1:
        parser.error("--generations 와 --graded 는 1 이상이다")

    overrides = dict(EXPERIMENTS[args.experiment])
    overrides |= {n: getattr(args, n) for n in OVERRIDABLE if getattr(args, n) is not None}
    try:
        setup = apply_overrides(SETUPS[GAME], overrides)
        # 풀이가 없거나 낡았으면 학습에 시간을 쓰기 전에 멈춘다.
        solution = load_solution_of(setup, args.solution)
        trials = cached_dev_trials(solution, args.solution, DEV_CACHE)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    wandb_run = start_wandb(args.wandb)
    wandb_run.name = tuning_name(args.experiment, args.seed)
    logger = TeeLogger(PrintingLogger(), WandbLogger(wandb_run))
    started = datetime.now()
    directory = args.checkpoint_root / f"{args.experiment}-seed{args.seed}-{started:%Y%m%d-%H%M%S}"
    logger.config(
        {
            "experiment": args.experiment,
            "overrides": overrides,
            "checkpoint_dir": str(directory),
            "dev_trials": {"seed": DEV_TRIAL_SEED, "simulations": setup.evaluation_simulations},
        }
    )
    train_run(
        setup,
        logger=logger,
        generations=args.generations,
        checkpoint_dir=directory,
        seed=args.seed,
        device=args.device,
    )

    # 함께 도는 다른 실험의 코어를 빼앗지 않는다.
    torch.set_num_threads(1)
    grade = grade_trials(
        setup,
        args.solution,
        simulations=setup.evaluation_simulations,
        device=args.device,
        trials=trials,
        solution=solution,
    )
    files = list(generation_files(directory).items())[-args.graded :]
    graded = []
    for generation, file in files:
        result = grade(file)["trials"]
        print(json.dumps({"generation": generation, **result}, ensure_ascii=False), flush=True)
        graded.append(result)

    count = len(graded)
    metrics = {
        "dev/passed": sum(result["passed"] for result in graded) / count,
        "dev/passed_last": graded[-1]["passed"],
        "dev/total": graded[-1]["total"],
    }
    for position, stratum in enumerate(graded[-1]["strata"]):
        passed = sum(result["strata"][position]["passed"] for result in graded)
        metrics[f"dev/{stratum['name']}"] = passed / count
    logger.log(metrics, step=args.generations)
    wandb_run.finish()
    print(json.dumps({"experiment": args.experiment, "seed": args.seed, **metrics}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
