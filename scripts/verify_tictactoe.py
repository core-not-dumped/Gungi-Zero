"""틱택토 검증의 합성 루트: 학습한 MCTS AI 를 완벽한 상대와 둔다 (2단계 완료 기준).

uv run python -m scripts.verify_tictactoe

처음 가중치부터 학습하고, 선수·후수 각 100판(시드 0~99)을 둔다. 한 판이라도 지면 종료 코드가
1 이다. 로거는 메모리 로거이고, 설정·세대별 기록·결과를 JSON 으로 한 줄씩 찍는다.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import torch

from ai.logger import Logger, MemoryLogger
from ai.mcts import MCTS, MCTSAI
from eval.match import Record, play_match
from eval.perfect import PerfectAI
from eval.solver import solve
from eval.value_agreement import value_agreement
from games.base import BoardOutcomeGame
from scripts.shared.hyperparameters import (
    TICTACTOE_EVALUATION_SIMULATIONS,
    TICTACTOE_EVALUATION_TEMPERATURE,
    TICTACTOE_GENERATIONS,
)
from scripts.shared.setups import SETUPS, GameSetup
from scripts.shared.training import new_checkpoint_dir, print_records, run

# 완료 기준의 대국 시드. 시드마다 선수·후수로 한 판씩 둔다.
MATCH_SEEDS = range(100)


@dataclass(frozen=True, slots=True)
class Verification:
    """검증 한 번의 측정값. 성적은 학습한 AI 시점이다."""

    as_first: Record
    as_second: Record
    # 가치 머리의 승·무·패가 이론값과 같은 상태의 비율. 진단이고 기준이 아니다.
    agreement: float
    # 자리마다 수열이 서로 다른 대국의 수. 진단이다.
    distinct_as_first: int
    distinct_as_second: int

    @property
    def unbeaten(self) -> bool:
        return self.as_first.losses == 0 and self.as_second.losses == 0


def verify(
    setup: GameSetup,
    *,
    logger: Logger,
    generations: int,
    checkpoint_dir: Path,
    seed: int,
    match_seeds: Sequence[int],
) -> Verification:
    """틱택토 설정 `setup` 을 CPU 에서 학습시키고, 마지막 세대를 재서 로거에도 남긴다."""
    # 틱택토는 `board_outcome` 을 답한다. `GameSetup` 은 공통 인터페이스로만 들고 있다.
    game = cast(BoardOutcomeGame[Any, Any], setup.game)
    logger.config(
        {
            "evaluation": {
                "simulations": TICTACTOE_EVALUATION_SIMULATIONS,
                "temperature": TICTACTOE_EVALUATION_TEMPERATURE,
                "seeds": list(match_seeds),
            }
        }
    )
    network = run(
        setup,
        logger=logger,
        generations=generations,
        checkpoint_dir=checkpoint_dir,
        seed=seed,
    )

    solution = solve(game, move_limit=setup.variant["move_limit"])
    match = play_match(
        game,
        lambda match_seed: MCTSAI(
            MCTS(game, network, simulations=TICTACTOE_EVALUATION_SIMULATIONS),
            temperature=TICTACTOE_EVALUATION_TEMPERATURE,
            seed=match_seed,
        ),
        lambda match_seed: PerfectAI(solution, match_seed),
        seeds=match_seeds,
    )
    result = Verification(
        match.as_first,
        match.as_second,
        value_agreement(solution, network).rate,
        match.distinct_as_first,
        match.distinct_as_second,
    )

    metrics: dict[str, float] = {
        "diagnostic/value_agreement": result.agreement,
        "match/first/distinct": result.distinct_as_first,
        "match/second/distinct": result.distinct_as_second,
    }
    for seat, record in (("first", result.as_first), ("second", result.as_second)):
        metrics |= {f"match/{seat}/{name}": count for name, count in asdict(record).items()}
    logger.log(metrics, step=generations)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="틱택토로 학습해 완벽한 상대와 둔다")
    parser.add_argument("--seed", type=int, default=0, help="학습의 시드")
    parser.add_argument("--generations", type=int, default=TICTACTOE_GENERATIONS)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.generations < 1:
        parser.error("--generations 는 1 이상이다")

    logger = MemoryLogger()
    # 스레드 수가 다르면 부동소수점 합의 순서가 달라져 같은 시드의 가중치가 갈릴 수 있다. 이 크기의
    # 신경망은 스레드 하나가 느리지 않다.
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        result = verify(
            SETUPS["tictactoe"],
            logger=logger,
            generations=args.generations,
            checkpoint_dir=args.checkpoint_dir or new_checkpoint_dir("tictactoe", args.seed),
            seed=args.seed,
            match_seeds=MATCH_SEEDS,
        )
    finally:
        torch.set_num_threads(threads)
    print_records(logger)
    print(json.dumps({**asdict(result), "unbeaten": result.unbeaten}))
    return 0 if result.unbeaten else 1


if __name__ == "__main__":
    raise SystemExit(main())
