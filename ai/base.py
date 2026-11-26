"""수를 고르는 쪽의 공통 인터페이스."""

from __future__ import annotations

from typing import Protocol, TypeVar

from games.base import Move, Position

PositionContraT = TypeVar("PositionContraT", bound=Position, contravariant=True)
MoveCoT = TypeVar("MoveCoT", bound=Move, covariant=True)


class AI(Protocol[PositionContraT, MoveCoT]):
    """국면을 받아 수 하나를 고르는 쪽 (`CONTEXT.md`). 선수·후수 어느 쪽에도 앉는다."""

    def choose(self, position: PositionContraT) -> MoveCoT:
        """`position` 에서 둘 차례인 쪽의 합법수 하나. 끝난 국면에는 부르지 않는다."""
        ...
