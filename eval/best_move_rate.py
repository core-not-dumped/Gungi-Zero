"""최선수 진단: AI 가 고른 수가 최선의 수인 비율을, 한 무리와 전체로 나눠 센다.

무리(예: 쌓기가 걸린 국면)의 약점이 전체 평균에 묻히지 않게 한다. 완료 기준은 아니다.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from ai.base import AI
from eval.perfect import best_moves
from eval.solver import Solution
from eval.value_agreement import Agreement
from games.base import MoveT, PositionT


@dataclass(frozen=True, slots=True)
class BestMoveRate:
    """본 국면의 수와 그중 최선의 수를 고른 국면의 수."""

    group: Agreement
    # 무리 밖.
    rest: Agreement

    @property
    def all(self) -> Agreement:
        return Agreement(
            self.group.matches + self.rest.matches, self.group.positions + self.rest.positions
        )


def best_move_rate(
    solution: Solution[PositionT, MoveT],
    positions: Iterable[PositionT],
    ai: AI[PositionT, MoveT],
    *,
    in_group: Callable[[PositionT], bool],
) -> BestMoveRate:
    """진행 중인 `positions` 마다 `ai` 에게 한 번 묻는다. 최선의 수는 `best_moves` 다."""
    # [무리 밖, 무리] 의 (고른 수, 본 수).
    counts = [[0, 0], [0, 0]]
    for position in positions:
        count = counts[bool(in_group(position))]
        count[0] += ai.choose(position) in best_moves(solution, position)
        count[1] += 1
    rest, group = (Agreement(matches, seen) for matches, seen in counts)
    return BestMoveRate(group, rest)
