"""가치 머리 진단: 신경망의 가치가 풀어 둔 이론값과 얼마나 일치하는지.

MCTS 가 틀린 신경망을 덮어 주어 대국 성적만 좋을 수 있어 따로 본다. 완료 기준은 아니다.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import islice

from ai.network import Network, evaluate_batch
from eval.reachable import reachable_positions
from eval.solver import Solution, Verdict
from games.base import MoveT, PositionT


@dataclass(frozen=True, slots=True)
class Agreement:
    """본 국면의 수와 그중 일치한 수."""

    matches: int
    positions: int

    @property
    def rate(self) -> float:
        return self.matches / self.positions


def _verdict_of(value: float) -> Verdict:
    """`[-1, 1]` 의 가치를 가장 가까운 결과(+1 승, 0 무, -1 패)로. 정확히 ±0.5 는 무승부다."""
    if value > 0.5:
        return Verdict.WIN
    if value < -0.5:
        return Verdict.LOSS
    return Verdict.DRAW


def value_agreement(
    solution: Solution[PositionT, MoveT],
    network: Network,
    *,
    batch_size: int = 512,
    device: str = "cpu",
) -> Agreement:
    """도달 가능한 상태 전부(끝난 상태 포함)에서 가치의 승·무·패가 이론값과 같은 비율을 센다.

    상태마다 `reachable_positions` 의 국면 하나를 보고, 이론값은 그 `ply` 의 수 제한을 반영한다.
    `network` 는 `device` 에 있고 평가가 상태를 바꾸지 않아야 한다 (`Network`).
    """
    game = solution.game
    matches = positions = 0
    remaining = reachable_positions(game)
    while batch := list(islice(remaining, batch_size)):
        _, values = evaluate_batch(network, [game.encode(p) for p in batch], device)
        matches += _count_matches(solution, batch, values.tolist())
        positions += len(batch)
    return Agreement(matches, positions)


def _count_matches(
    solution: Solution[PositionT, MoveT], batch: Sequence[PositionT], values: Sequence[float]
) -> int:
    return sum(
        solution.value(position).verdict is _verdict_of(value)
        for position, value in zip(batch, values, strict=True)
    )
