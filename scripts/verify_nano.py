"""나노 군의 검증의 합성 루트: 2.5단계 완료 기준 (ADR-0007). wandb 에 기록한다.

uv run python -m scripts.solve_nano                    # 풀이가 없으면 먼저 만든다
uv run python -m scripts.verify_nano --device cuda     # 처음부터 학습하고 잰다
uv run python -m scripts.verify_nano --device cuda --checkpoint <파일>   # 저장한 세대만 잰다

학습한 MCTS AI 가 시험 판 200개를 모두 통과하면 종료 코드가 0, 아니면 1 이다.
층마다의 성적과 진단(가치 일치율, 최선수 비율)을 함께 낸다. 학습 없이 wandb 를
거치지 않으려면 `--wandb disabled` 를, 로그인하지 않고 파일로 남기려면 `--wandb offline` 을 쓴다.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import torch

from ai.checkpoint import generation_of, load_network
from ai.mcts import MCTS, MCTSAI
from ai.network import Network
from eval.best_move_rate import BestMoveRate, best_move_rate
from eval.perfect import PerfectAI
from eval.reachable import reachable_positions
from eval.solver import Solution
from eval.trial_match import play_trials
from eval.trial_positions import Trial
from eval.value_agreement import Agreement, value_agreement
from games.mini_gungi import Gungi
from scripts.shared.hyperparameters import (
    NANO_GUNGI_BEST_MOVE_SAMPLE,
    NANO_GUNGI_BEST_MOVE_SEED,
    NANO_GUNGI_EVALUATION_SIMULATIONS,
    NANO_GUNGI_EVALUATION_TEMPERATURE,
    NANO_GUNGI_GENERATIONS,
)
from scripts.shared.loggers import WANDB_PROJECT, PrintingLogger, TeeLogger, WandbLogger
from scripts.shared.run_names import evaluation_name, training_name
from scripts.shared.setups import SETUPS
from scripts.shared.solution import SOLUTION_PATH, load_solution_of
from scripts.shared.training import new_checkpoint_dir
from scripts.shared.training import run as train_run
from scripts.shared.trials import fingerprint, involves_stacking, nano_trials


@dataclass(frozen=True, slots=True)
class StratumScore:
    """시험 판 한 층의 통과한 수와 전체."""

    name: str
    passed: int
    total: int


@dataclass(frozen=True, slots=True)
class Measurement:
    """시험 판의 성적과 진단."""

    strata: tuple[StratumScore, ...]
    # 통과하지 못한 시험 판의 순번.
    failed_trials: tuple[int, ...]
    # 도달 가능한 상태 전부에서 가치 머리의 승·무·패가 이론값과 같은 비율. 진단이다.
    value_agreement: Agreement
    # 뽑은 국면에서 MCTS 가 최선의 수를 고른 비율. 무리는 쌓기가 걸린 국면이다. 진단이다.
    best_move_rates: BestMoveRate


@dataclass(frozen=True, slots=True)
class NanoVerification(Measurement):
    """검증 한 번의 결과."""

    @property
    def passed(self) -> bool:
        """완료 기준: 시험 판 전부."""
        return not self.failed_trials


def _mcts_ai(
    game: Gungi, network: Network, simulations: int, device: str, seed: int
) -> MCTSAI[Any, Any]:
    return MCTSAI(
        MCTS(game, network, simulations=simulations, device=device),
        temperature=NANO_GUNGI_EVALUATION_TEMPERATURE,
        seed=seed,
    )


def measure(
    solution: Solution[Any, Any],
    trials: Sequence[Trial[Any]],
    network: Network,
    *,
    simulations: int,
    device: str,
    best_move_sample: int,
) -> Measurement:
    """시험 판(상대는 완벽한 상대, 시드 = 순번)과 진단 둘을 잰다. `network` 는 `device` 에 있고
    평가 모드여야 한다."""
    game = solution.game
    assert isinstance(game, Gungi), "나노 군의의 풀이를 받는다"
    results = play_trials(
        game,
        trials,
        lambda index: _mcts_ai(game, network, simulations, device, index),
        lambda index: PerfectAI(solution, index),
    )
    names = dict.fromkeys(trial.stratum for trial in trials)
    strata = []
    for name in names:
        passed = [result.passed for result in results if result.trial.stratum == name]
        strata.append(StratumScore(name, sum(passed), len(passed)))

    unfinished = [p for p in reachable_positions(game) if game.outcome(p) is None]
    sample = random.Random(NANO_GUNGI_BEST_MOVE_SEED).sample(
        unfinished, min(best_move_sample, len(unfinished))
    )
    rates = best_move_rate(
        solution,
        sample,
        _mcts_ai(game, network, simulations, device, NANO_GUNGI_BEST_MOVE_SEED),
        in_group=lambda position: involves_stacking(game, position),
    )
    return Measurement(
        tuple(strata),
        tuple(index for index, result in enumerate(results) if not result.passed),
        value_agreement(solution, network, device=device),
        rates,
    )


def evaluate_network(
    solution: Solution[Any, Any],
    trials: Sequence[Trial[Any]],
    network: Network,
    *,
    device: str,
) -> NanoVerification:
    """실제 설정으로 시험 판과 진단을 모두 잰다."""
    game = solution.game
    assert isinstance(game, Gungi)
    simulations = NANO_GUNGI_EVALUATION_SIMULATIONS
    measured = measure(
        solution,
        trials,
        network,
        simulations=simulations,
        device=device,
        best_move_sample=NANO_GUNGI_BEST_MOVE_SAMPLE,
    )
    return NanoVerification(
        measured.strata,
        measured.failed_trials,
        measured.value_agreement,
        measured.best_move_rates,
    )


def metrics_of(result: NanoVerification) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for stratum in result.strata:
        metrics[f"trial/{stratum.name}/passed"] = stratum.passed
        metrics[f"trial/{stratum.name}/total"] = stratum.total
    metrics["trial/passed"] = sum(stratum.passed for stratum in result.strata)
    metrics["trial/total"] = sum(stratum.total for stratum in result.strata)
    metrics["diagnostic/value_agreement"] = result.value_agreement.rate
    rates = result.best_move_rates
    for name, agreement in (("all", rates.all), ("stacking", rates.group), ("plain", rates.rest)):
        # 뽑힌 국면이 없는 무리는 비율이 없다.
        if agreement.positions:
            metrics[f"diagnostic/best_move/{name}"] = agreement.rate
    return metrics


def start_wandb(mode: Literal["online", "offline", "disabled"]) -> Any:
    """wandb run 을 연다. torch 처럼 무거워 쓸 때 import 한다."""
    import wandb

    return wandb.init(project=WANDB_PROJECT, job_type="verify-nano", mode=mode)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="나노 군의로 학습해 시험 판을 잰다")
    parser.add_argument("--seed", type=int, default=0, help="학습의 시드")
    parser.add_argument("--generations", type=int, default=NANO_GUNGI_GENERATIONS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument(
        "--checkpoint", type=Path, default=None, help="학습하지 않고 이 세대를 잰다"
    )
    parser.add_argument("--solution", type=Path, default=SOLUTION_PATH)
    parser.add_argument("--wandb", choices=("online", "offline", "disabled"), default="online")
    args = parser.parse_args(argv)
    if args.generations < 1:
        parser.error("--generations 는 1 이상이다")

    setup = SETUPS["nano-gungi"]
    # 풀이가 없거나 낡았으면 학습에 시간을 쓰기 전에 멈춘다.
    solution = load_solution_of(setup, args.solution)
    wandb_run = start_wandb(args.wandb)
    if args.checkpoint is None:
        wandb_run.name = training_name(
            setup.name,
            seed=args.seed,
            episodes=setup.training.episodes_per_generation,
            steps=setup.training.steps_per_generation,
            started=datetime.now(),
        )
    else:
        wandb_run.name = evaluation_name(setup.name, args.checkpoint)
    logger = TeeLogger(PrintingLogger(), WandbLogger(wandb_run))
    started = time.perf_counter()
    if args.checkpoint is None:
        directory = args.checkpoint_dir or new_checkpoint_dir(setup.name, args.seed)
        logger.config({"checkpoint_dir": str(directory)})
        network: torch.nn.Module = train_run(
            setup,
            logger=logger,
            generations=args.generations,
            checkpoint_dir=directory,
            seed=args.seed,
            device=args.device,
        )
        step = args.generations
    else:
        # 튜닝으로 구조를 바꿔 학습한 것일 수 있어 구조는 가중치에서 읽는다.
        network = load_network(setup.game, args.checkpoint)
        network.to(args.device).eval()
        logger.config(
            {
                "game": setup.name,
                "checkpoint": str(args.checkpoint),
                "device": args.device,
                "variant": setup.variant,
                "network": asdict(network.config),
            }
        )
        # 체크포인트의 세대에 적는다. 체크포인트 이름이 아니면 0 이다.
        step = generation_of(args.checkpoint) or 0
    # 읽기만 했으면 학습 시간이 없다.
    training_seconds = None
    if args.checkpoint is None:
        training_seconds = round(time.perf_counter() - started, 1)
        # 평가가 중간에 죽어도 남도록 먼저 적는다.
        logger.log({"time/training_seconds": training_seconds}, step=step)

    trials = nano_trials(solution)
    assert isinstance(setup.game, Gungi)
    logger.config(
        {
            "evaluation": {
                "simulations": NANO_GUNGI_EVALUATION_SIMULATIONS,
                "temperature": NANO_GUNGI_EVALUATION_TEMPERATURE,
                "trials_fingerprint": fingerprint(setup.game, trials),
                "best_move_sample": NANO_GUNGI_BEST_MOVE_SAMPLE,
                "best_move_seed": NANO_GUNGI_BEST_MOVE_SEED,
            }
        }
    )
    started = time.perf_counter()
    result = evaluate_network(solution, trials, network, device=args.device)
    evaluation_seconds = round(time.perf_counter() - started, 1)
    logger.log({**metrics_of(result), "time/evaluation_seconds": evaluation_seconds}, step=step)
    wandb_run.finish()
    summary = {
        **asdict(result),
        "passed": result.passed,
        "training_seconds": training_seconds,
        "evaluation_seconds": evaluation_seconds,
    }
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
