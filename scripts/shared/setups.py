"""학습하고 재는 게임마다의 설정. 명령은 게임 이름으로 여기서 설정을 고른다."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ai.resnet import ResNetConfig
from ai.training import TrainingConfig
from games.base import Game
from games.mini_gungi import MINI_GUNGI, Gungi, VariantConfig
from games.nano_gungi import NANO_GUNGI
from games.tictactoe import SIZE, TicTacToe
from scripts.shared.hyperparameters import (
    MINI_GUNGI_EVALUATION_SIMULATIONS,
    MINI_GUNGI_NETWORK,
    MINI_GUNGI_TRAINING,
    NANO_GUNGI_EVALUATION_SIMULATIONS,
    NANO_GUNGI_INITIAL_START_FRACTION,
    NANO_GUNGI_NETWORK,
    NANO_GUNGI_TRAINING,
    TICTACTOE_EVALUATION_SIMULATIONS,
    TICTACTOE_NETWORK,
    TICTACTOE_TRAINING,
)


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
    # 자가 대국 가운데 초기 국면에서 시작하는 판의 비율. 1 보다 작으면 나머지는 `mixed_starts` 가
    # 고른 국면에서 시작한다.
    initial_start_fraction: float = 1.0
    # 실력을 잴 때의 MCTS 탐색 횟수.
    evaluation_simulations: int = 100


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


def _gungi(
    name: str,
    config: VariantConfig,
    network: ResNetConfig,
    training: TrainingConfig,
    *,
    evaluation_simulations: int,
    initial_start_fraction: float = 1.0,
) -> GameSetup:
    """군의 변형 하나의 설정. 새 변형은 `Gungi` 가 받는 설정이면 여기에 더하면 된다."""
    game = Gungi(config)

    def end_reason(position: Any) -> str:
        reason = game.end_reason(position)
        assert reason is not None, "끝난 국면에만 묻는다"
        return reason.name

    return GameSetup(
        name,
        game,
        gungi_variant(config),
        end_reason,
        network,
        training,
        initial_start_fraction,
        evaluation_simulations,
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
    return GameSetup(
        "tictactoe",
        game,
        variant,
        end_reason,
        TICTACTOE_NETWORK,
        TICTACTOE_TRAINING,
        evaluation_simulations=TICTACTOE_EVALUATION_SIMULATIONS,
    )


SETUPS: dict[str, GameSetup] = {
    setup.name: setup
    for setup in (
        _tictactoe(),
        _gungi(
            "nano-gungi",
            NANO_GUNGI,
            NANO_GUNGI_NETWORK,
            NANO_GUNGI_TRAINING,
            evaluation_simulations=NANO_GUNGI_EVALUATION_SIMULATIONS,
            initial_start_fraction=NANO_GUNGI_INITIAL_START_FRACTION,
        ),
        _gungi(
            "mini-gungi",
            MINI_GUNGI,
            MINI_GUNGI_NETWORK,
            MINI_GUNGI_TRAINING,
            evaluation_simulations=MINI_GUNGI_EVALUATION_SIMULATIONS,
        ),
    )
}
