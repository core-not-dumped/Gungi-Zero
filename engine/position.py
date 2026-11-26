"""한 국면과 그 판."""

from __future__ import annotations

from dataclasses import dataclass

from engine.pieces import Piece, Side

# 한 칸에 쌓인 말, 아래(1단)부터 위로. 빈칸은 빈 튜플이다.
Tower = tuple[Piece, ...]


@dataclass(frozen=True, slots=True)
class Position:
    """판 + 차례 + `ply`. 경로상 국면들은 반복 규칙과 함께 온다 (04 번 티켓).

    `board` 는 칸 번호 순서다 — `row * width + column`. row 0 이 선수 진영 끝줄이다.
    """

    board: tuple[Tower, ...]
    side: Side
    ply: int

    @property
    def board_key(self) -> tuple[tuple[Tower, ...], Side]:
        """판 + 차례. 반복 횟수는 이 키로 센다."""
        return (self.board, self.side)

    def rotated(self) -> Position:
        """판을 180° 돌리고 색을 바꾼 국면. 칸 `i` 는 칸 `칸 수 - 1 - i` 로 간다."""
        board = tuple(
            tuple(Piece(piece.kind, piece.owner.opponent) for piece in tower)
            for tower in reversed(self.board)
        )
        return Position(board=board, side=self.side.opponent, ply=self.ply)
