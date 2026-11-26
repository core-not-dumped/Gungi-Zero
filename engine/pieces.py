"""말과 진영."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class Side(Enum):
    """선수와 후수."""

    FIRST = auto()
    SECOND = auto()

    @property
    def opponent(self) -> Side:
        return Side.SECOND if self is Side.FIRST else Side.FIRST


class PieceType(Enum):
    """말 종류. 선언 순서에는 뜻이 없다."""

    MARSHAL = auto()  # 帥
    GENERAL = auto()  # 大
    LANCE = auto()  # 槍
    PAWN = auto()  # 兵


@dataclass(frozen=True, slots=True)
class Piece:
    kind: PieceType
    owner: Side
