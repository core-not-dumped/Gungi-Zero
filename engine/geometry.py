"""판의 좌표와 180° 회전. 칸 번호와 회전을 아는 곳은 여기 하나다.

- 칸 번호는 `row * width + column` 이다. 열·행은 0 부터이고 row 0 이 선수 진영 끝줄이다.
- 칸 이름은 열 글자와 1 부터 세는 행 번호다 (`a1`).
- 180° 회전은 칸 `i` 를 `칸 수 - 1 - i` 로 보낸다. 후수의 배치와 앞뒤, 정규형이 모두 이 회전이다
  (ADR-0003).
"""

from __future__ import annotations

from dataclasses import dataclass

from engine.pieces import Piece, Side

COLUMN_NAMES = "abcdefghijklmnopqrstuvwxyz"


@dataclass(frozen=True, slots=True)
class BoardGeometry:
    """`width` x `height` 판의 좌표."""

    width: int
    height: int

    @property
    def square_count(self) -> int:
        return self.width * self.height

    def on_board(self, square: int) -> bool:
        return 0 <= square < self.square_count

    def contains(self, column: int, row: int) -> bool:
        return 0 <= column < self.width and 0 <= row < self.height

    def square(self, column: int, row: int) -> int:
        """열·행 → 칸 번호. 판 밖이면 `ValueError` 다."""
        if not self.contains(column, row):
            raise ValueError(f"판 밖의 열·행이다: ({column}, {row})")
        return row * self.width + column

    def column_and_row(self, square: int) -> tuple[int, int]:
        """칸 번호 → 열·행. `square` 의 역이다. 판 밖이면 `ValueError` 다."""
        self._check(square)
        row, column = divmod(square, self.width)
        return column, row

    def rotate(self, square: int) -> int:
        """180° 돌린 칸. 두 번 돌리면 제자리다. 판 밖이면 `ValueError` 다."""
        self._check(square)
        return self.square_count - 1 - square

    def column_name(self, column: int) -> str:
        return COLUMN_NAMES[column]

    def row_name(self, row: int) -> str:
        return str(row + 1)

    def square_name(self, square: int) -> str:
        """칸 번호 → 칸 이름. 판 밖이면 `ValueError` 다."""
        column, row = self.column_and_row(square)
        return self.column_name(column) + self.row_name(row)

    def square_named(self, name: str) -> int:
        """칸 이름 → 칸 번호. `square_name` 의 역이다. 판의 칸 이름이 아니면 `ValueError` 다."""
        column_name, row_name = name[:1], name[1:]
        column = COLUMN_NAMES.find(column_name) if column_name else -1
        if column < 0 or not row_name.isdecimal():
            raise ValueError(f"칸 이름이 'a1' 꼴이 아니다: {name!r}")
        return self.square(column, int(row_name) - 1)

    def _check(self, square: int) -> None:
        if not self.on_board(square):
            raise ValueError(f"판 밖의 칸이다: {square}")


def forward(side: Side) -> int:
    """선수의 앞(row +1, 오른쪽 +1)을 기준으로 한 `side` 의 부호. 후수는 180° 돌려 본다."""
    return 1 if side is Side.FIRST else -1


def rotate_board(
    board: tuple[tuple[Piece, ...], ...],
) -> tuple[tuple[Piece, ...], ...]:
    """판을 180° 돌리고 색을 바꾼다. 탑 안의 순서는 그대로다."""
    return tuple(
        tuple(Piece(piece.kind, piece.owner.opponent) for piece in tower)
        for tower in reversed(board)
    )
