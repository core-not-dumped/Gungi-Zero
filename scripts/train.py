"""학습의 합성 루트. 게임·하이퍼파라미터·로거를 골라 세대 루프에 넣는다.

uv run python -m scripts.train --game tictactoe --generations 1

로거는 메모리 로거이고, 끝나면 설정 한 줄과 세대마다 한 줄을 JSON 으로 찍는다.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from ai.logger import Logger, MemoryLogger
from ai.resnet import ResNet, ResNetConfig
from ai.training import TrainingConfig, seed_everything, train
from games.base import Game
from games.mini_gungi import Gungi, VariantConfig
from games.nano_gungi import NANO_GUNGI
from games.tictactoe import SIZE, TicTacToe
from scripts.hyperparameters import (
    NANO_GUNGI_NETWORK,
    NANO_GUNGI_TRAINING,
    TICTACTOE_NETWORK,
    TICTACTOE_TRAINING,
)

CHECKPOINT_ROOT = Path("checkpoints")


@dataclass(frozen=True)
class GameSetup:
    """학습할 게임 하나와 그 설정."""

    name: str
    game: Game[Any, Any]
    # 변형 설정. 학습 기록에 그대로 남겨 종료 규칙이 다른 학습을 구분한다 (ADR-0004).
    variant: dict[str, Any]
    # 끝난 국면 → 종료 사유 이름.
    end_reason: Callable[[Any], str]
    network: ResNetConfig
    training: TrainingConfig


def gungi_variant(config: VariantConfig) -> dict[str, Any]:
    """판 크기, 말 구성(선수의 배치. 후수는 180° 회전이다), 종료 규칙."""
    return {
        "width": config.width,
        "height": config.height,
        "max_tier": config.max_tier,
        "pieces": [
            {"kind": piece.kind.name, "column": piece.column, "row": piece.row}
            for piece in config.setup
        ],
        "move_limit": config.move_limit,
        "repetition_limit": config.repetition_limit,
    }


def _nano_gungi() -> GameSetup:
    game = Gungi(NANO_GUNGI)

    def end_reason(position: Any) -> str:
        reason = game.end_reason(position)
        assert reason is not None, "끝난 국면에만 묻는다"
        return reason.name

    return GameSetup(
        "nano-gungi",
        game,
        gungi_variant(NANO_GUNGI),
        end_reason,
        NANO_GUNGI_NETWORK,
        NANO_GUNGI_TRAINING,
    )


def _tictactoe() -> GameSetup:
    game = TicTacToe()

    def end_reason(position: Any) -> str:
        # 비기는 것은 판이 찼을 때뿐이고, 승패는 줄이 정한다.
        outcome = game.outcome(position)
        assert outcome is not None, "끝난 국면에만 묻는다"
        return "BOARD_FULL" if outcome.winner is None else "LINE"

    variant = {
        "width": SIZE,
        "height": SIZE,
        "max_tier": 1,
        # 처음 판은 비어 있고, 수 제한과 반복 규칙이 없다.
        "pieces": [],
        "move_limit": None,
        "repetition_limit": None,
    }
    return GameSetup("tictactoe", game, variant, end_reason, TICTACTOE_NETWORK, TICTACTOE_TRAINING)


SETUPS: dict[str, GameSetup] = {setup.name: setup for setup in (_tictactoe(), _nano_gungi())}


def run(
    setup: GameSetup,
    *,
    logger: Logger,
    generations: int,
    checkpoint_dir: Path,
    seed: int,
    device: str = "cpu",
) -> ResNet:
    """설정을 로거에 남기고 처음 가중치부터 학습한다. 마지막 세대의 신경망을 돌려준다."""
    logger.config(
        {
            "game": setup.name,
            "seed": seed,
            "device": device,
            "generations": generations,
            "variant": setup.variant,
            "network": asdict(setup.network),
            "training": asdict(setup.training),
        }
    )
    seed_everything(seed)
    network = ResNet(setup.game.tensor_shape, setup.game.action_space_size, setup.network)
    train(
        setup.game,
        network,
        setup.training,
        generations=generations,
        logger=logger,
        end_reason=setup.end_reason,
        checkpoint_dir=checkpoint_dir,
        seed=seed,
        device=device,
    )
    return network


def new_checkpoint_dir(game: str, seed: int) -> Path:
    """학습마다 새로 만드는 기본 체크포인트 폴더."""
    started = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    return CHECKPOINT_ROOT / f"{game}-seed{seed}-{started}"


def print_records(logger: MemoryLogger) -> None:
    """설정 한 줄과 기록마다 한 줄을 JSON 으로 찍는다."""
    print(json.dumps(logger.configuration, ensure_ascii=False))
    for step, metrics in logger.records:
        print(json.dumps({"step": step, **metrics}, ensure_ascii=False))


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
