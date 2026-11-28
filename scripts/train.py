"""학습의 합성 루트. 게임·하이퍼파라미터·로거를 골라 세대 루프에 넣는다.

uv run python -m scripts.train --game tictactoe --generations 1

로거는 메모리 로거이고, 끝나면 설정 한 줄과 세대마다 한 줄을 JSON 으로 찍는다.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from ai.logger import MemoryLogger
from scripts.shared.setups import SETUPS
from scripts.shared.training import CHECKPOINT_ROOT, new_checkpoint_dir, print_records, run


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="자가 대국으로 학습한다")
    parser.add_argument("--game", choices=tuple(SETUPS), default="tictactoe")
    parser.add_argument("--generations", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu", help="cpu 또는 cuda. 재현은 cpu 에서만 보장한다")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help=f"기본은 {CHECKPOINT_ROOT}/ 아래에 학습마다 새로 만든다",
    )
    args = parser.parse_args(argv)
    if args.generations < 1:
        parser.error("--generations 는 1 이상이다")

    directory = args.checkpoint_dir or new_checkpoint_dir(args.game, args.seed)

    logger = MemoryLogger()
    run(
        SETUPS[args.game],
        logger=logger,
        generations=args.generations,
        checkpoint_dir=directory,
        seed=args.seed,
        device=args.device,
    )
    print_records(logger)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
