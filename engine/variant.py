"""변형 설정. 미니 군의와 나노 군의는 같은 엔진의 서로 다른 설정이다."""

from __future__ import annotations

from dataclasses import dataclass

from engine.pieces import PieceType


@dataclass(frozen=True, slots=True)
class Placement:
    """선수의 말 하나를 놓는 자리. `column`·`row` 는 0 부터이고 row 0 이 선수 진영 끝줄이다."""

    kind: PieceType
    column: int
    row: int


@dataclass(frozen=True, slots=True)
class VariantConfig:
    """판 크기, 말 구성, 최대 단, 초기 배치, 수 제한, 반복 한도.

    `setup` 은 선수의 배치만 적는다. 후수는 그 180° 회전이라 대칭이 아닌 배치는 적을 수조차
    없다 (ADR-0003). 말 종류와 개수도 `setup` 이 정한다 — 따로 적으면 서로 어긋날 수 있다.

    `repetition_limit` 은 아직 판정에 쓰지 않는다 (04 번 티켓).
    """

    width: int
    height: int
    max_tier: int
    setup: tuple[Placement, ...]
    move_limit: int
    repetition_limit: int

    def __post_init__(self) -> None:
        if self.width < 1 or self.height < 1:
            raise ValueError(f"판 크기는 1 이상이다: {self.width}x{self.height}")
        if self.max_tier < 1:
            raise ValueError(f"최대 단은 1 이상이다: {self.max_tier}")
        if self.move_limit < 1:
            raise ValueError(f"수 제한은 1 이상이다: {self.move_limit}")
        if self.repetition_limit < 2:
            raise ValueError(f"반복 한도는 2 이상이다: {self.repetition_limit}")

        taken: set[tuple[int, int]] = set()
        for placement in self.setup:
            column, row = placement.column, placement.row
            if not (0 <= column < self.width and 0 <= row < self.height):
                raise ValueError(f"판 밖의 배치다: {placement}")
            rotated = (self.width - 1 - column, self.height - 1 - row)
            for square in ((column, row), rotated):
                if square in taken:
                    raise ValueError(f"배치가 (회전한 상대 배치를 포함해) 겹친다: {placement}")
                taken.add(square)

        marshals = sum(1 for placement in self.setup if placement.kind is PieceType.MARSHAL)
        if marshals != 1:
            raise ValueError(f"帥 는 한 편에 정확히 하나다: {marshals} 개")

    @property
    def square_count(self) -> int:
        return self.width * self.height
