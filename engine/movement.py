"""말의 사거리 표. 공식 룰북의 컬러 이미지를 2차 자료가 재구성한 값이라 바뀔 수 있다 (ADR-0002).

값을 바꿀 때는 이 모듈만 고친다.
"""

from __future__ import annotations

from dataclasses import dataclass

from engine.pieces import PieceType


@dataclass(frozen=True, slots=True)
class Ray:
    """한 방향의 1단 사거리.

    `column_step`·`row_step` 은 둔 쪽에서 본 한 칸이고 `row_step` +1 이 앞이다.
    `reach` 는 그 방향으로 갈 수 있는 최대 칸 수이며 `None` 이면 제한이 없다.
    """

    column_step: int
    row_step: int
    reach: int | None

    def reach_at(self, tier: int) -> int | None:
        """`tier` 단의 사거리. 단이 1 오를 때마다 유한 방향은 한 칸 늘고, 무제한은 그대로다."""
        return None if self.reach is None else self.reach + tier - 1


ORTHOGONALS = ((0, 1), (0, -1), (-1, 0), (1, 0))
DIAGONALS = ((-1, 1), (1, 1), (-1, -1), (1, -1))


def _rays(steps: tuple[tuple[int, int], ...], reach: int | None) -> tuple[Ray, ...]:
    return tuple(Ray(column_step, row_step, reach) for column_step, row_step in steps)


MOVEMENT: dict[PieceType, tuple[Ray, ...]] = {
    # 帥: 8방향 1칸.
    PieceType.MARSHAL: _rays(ORTHOGONALS + DIAGONALS, 1),
    # 大: 직선 무제한, 대각 1칸.
    PieceType.GENERAL: _rays(ORTHOGONALS, None) + _rays(DIAGONALS, 1),
    # 槍: 전방 2칸, 전방대각 1칸, 후방 1칸. 측면과 후방대각은 없다.
    PieceType.LANCE: _rays(((0, 1),), 2) + _rays(((-1, 1), (1, 1), (0, -1)), 1),
    # 兵: 전방 또는 후방 1칸.
    PieceType.PAWN: _rays(((0, 1), (0, -1)), 1),
}
