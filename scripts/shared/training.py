"""학습 한 번을 차린다: 설정을 로거에 남기고 처음 가중치부터 세대 루프를 돌린다."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from ai.logger import Logger, MemoryLogger
from ai.resnet import ResNet
from ai.training import seed_everything, train
from scripts.shared.loggers import PrintingLogger
from scripts.shared.self_play_starts import mixed_starts
from scripts.shared.setups import GameSetup

CHECKPOINT_ROOT = Path("checkpoints")


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
            # 나머지 비율은 도달 가능하고 끝나지 않은 상태에서 균등하게 뽑는다.
            "self_play_start": {"initial_fraction": setup.initial_start_fraction},
        }
    )
    seed_everything(seed)
    network = ResNet(setup.game.tensor_shape, setup.game.action_space_size, setup.network)
    fraction = setup.initial_start_fraction
    start_position = None if fraction == 1 else mixed_starts(setup.game, initial_fraction=fraction)
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
        start_position=start_position,
    )
    return network


def new_checkpoint_dir(game: str, seed: int) -> Path:
    """학습마다 새로 만드는 기본 체크포인트 폴더."""
    started = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    return CHECKPOINT_ROOT / f"{game}-seed{seed}-{started}"


def print_records(logger: MemoryLogger) -> None:
    """설정 한 줄과 기록마다 한 줄을 JSON 으로 찍는다."""
    printer = PrintingLogger()
    printer.config(logger.configuration)
    for step, metrics in logger.records:
        printer.log(metrics, step)
