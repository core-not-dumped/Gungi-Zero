"""한 국면과 그 판."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

from engine.pieces import Piece, Side

# 한 칸에 쌓인 말, 아래(1단)부터 위로. 빈칸은 빈 튜플이다.
Tower = tuple[Piece, ...]
Board = tuple[Tower, ...]


@dataclass(frozen=True, slots=True, eq=False)
class Position:
    """판 + 차례 + `ply` + 경로상 국면들. 네 개 전부 동등성과 해시에 들어간다.

    `board` 는 칸 번호 순서다 — `row * width + column`. row 0 이 선수 진영 끝줄이다.

    경로는 `previous` 로 이어지는 연결 리스트다. 새 국면은 이전 국면을 가리키기만 하므로
    복사가 없다. 동등성·해시·반복 횟수는 필요할 때 걸어가며 계산한다 — 재귀하지 않는다.
    """

    board: Board
    side: Side
    ply: int
    previous: Position | None = field(default=None, repr=False)

    def path(self) -> Iterator[Position]:
        """이 국면부터 경로의 첫 국면까지, 새것부터."""
        position: Position | None = self
        while position is not None:
            yield position
            position = position.previous

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Position):
            return NotImplemented
        mine: Position | None = self
        theirs: Position | None = other
        # 같은 객체에 닿으면 남은 경로는 공유하는 것이라 더 볼 것이 없다.
        while mine is not theirs:
            if mine is None or theirs is None:
                return False
            if mine._node_key != theirs._node_key:
                return False
            mine, theirs = mine.previous, theirs.previous
        return True

    def __hash__(self) -> int:
        return hash(tuple(position._node_key for position in self.path()))

    @property
    def _node_key(self) -> tuple[int, tuple[Board, Side]]:
        """경로를 뺀 이 국면 하나. 동등성과 해시가 같은 것을 보게 한다."""
        return (self.ply, self.board_key)

    @property
    def board_key(self) -> tuple[Board, Side]:
        """판 + 차례. 반복 횟수는 이 키로 센다."""
        return (self.board, self.side)

    @property
    def earlier_occurrences(self) -> int:
        """이 판 모양이 경로에 앞서 나온 횟수. 처음이면 0 이다."""
        key = self.board_key
        earlier = self.path()
        next(earlier)
        return sum(1 for position in earlier if position.board_key == key)

    def rotated(self) -> Position:
        """판을 180° 돌리고 색을 바꾼 국면. 칸 `i` 는 칸 `칸 수 - 1 - i` 로 간다.

        경로도 함께 돌린다 — 그래야 반복 횟수가 그대로다.
        """
        oldest, *later = reversed(tuple(self.path()))
        result = oldest._rotated_onto(None)
        for position in later:
            result = position._rotated_onto(result)
        return result

    def _rotated_onto(self, previous: Position | None) -> Position:
        board = tuple(
            tuple(Piece(piece.kind, piece.owner.opponent) for piece in tower)
            for tower in reversed(self.board)
        )
        return Position(board=board, side=self.side.opponent, ply=self.ply, previous=previous)
