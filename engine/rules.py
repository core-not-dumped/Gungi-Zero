"""룰 엔진. 변형 설정 하나를 받아 합법수·수 두기·승패를 답한다."""

from __future__ import annotations

from dataclasses import dataclass

from engine.pieces import Piece, PieceType, Side
from engine.position import Position, Tower
from engine.variant import VariantConfig


@dataclass(frozen=True, slots=True)
class Move:
    """출발 칸과 도착 칸. 이 쌍이 수를 모호함 없이 결정한다."""

    source: int
    target: int


@dataclass(frozen=True, slots=True)
class Result:
    """끝난 대국의 결과. `winner` 가 `None` 이면 무승부다."""

    winner: Side | None


# 1단에서 한 번에 갈 수 있는 (열, 행) 이동. 06 번 티켓이 사거리 표로 바꾼다.
STEPS: dict[PieceType, tuple[tuple[int, int], ...]] = {
    PieceType.MARSHAL: (
        (-1, -1),
        (0, -1),
        (1, -1),
        (-1, 0),
        (1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
    ),
}


@dataclass(frozen=True, slots=True)
class RuleEngine:
    config: VariantConfig

    def initial_position(self) -> Position:
        board: list[Tower] = [()] * self.config.square_count
        for placement in self.config.setup:
            square = placement.row * self.config.width + placement.column
            board[square] = (Piece(placement.kind, Side.FIRST),)
            board[self.config.square_count - 1 - square] = (Piece(placement.kind, Side.SECOND),)
        return Position(board=tuple(board), side=Side.FIRST, ply=0)

    def result(self, position: Position) -> Result | None:
        """끝난 국면의 결과, 진행 중이면 `None`."""
        return self._evaluate(position)[0]

    def legal_moves(self, position: Position) -> tuple[Move, ...]:
        """둘 수 있는 모든 수. 끝난 국면에서는 비어 있다."""
        return self._evaluate(position)[1]

    def _evaluate(self, position: Position) -> tuple[Result | None, tuple[Move, ...]]:
        """결과와 합법수를 함께 낸다 — 합법수 없음이 종료 사유라 둘을 따로 정할 수 없다.

        승패를 무승부보다 먼저 본다. 마지막 수로 帥 를 잡으면 승리이고, 수 제한에 닿은
        국면이라도 둘 수가 없으면 패배다.
        """
        alive = {
            piece.owner
            for tower in position.board
            for piece in tower
            if piece.kind is PieceType.MARSHAL
        }
        for side in Side:
            if side not in alive:
                return Result(winner=side.opponent), ()
        moves = self._moves(position)
        if not moves:
            return Result(winner=position.side.opponent), ()
        # 한도는 지금 나온 것까지 센다.
        repeated = position.earlier_occurrences + 1 >= self.config.repetition_limit
        if repeated or position.ply >= self.config.move_limit:
            return Result(winner=None), ()
        return None, moves

    def _moves(self, position: Position) -> tuple[Move, ...]:
        """종료 규칙을 보지 않고, 말의 이동만으로 나오는 수."""
        moves: list[Move] = []
        for source, tower in enumerate(position.board):
            if not tower or tower[-1].owner is not position.side:
                continue
            for target in self._targets(position, source):
                moves.append(Move(source, target))
        return tuple(moves)

    def apply_move(self, position: Position, move: Move) -> Position:
        """수를 둔 새 국면. 합법수가 아니면 `ValueError` 다."""
        if move not in self.legal_moves(position):
            raise ValueError(f"합법수가 아니다: {move}")
        board = list(position.board)
        # 이동과 잡기뿐이다 — 도착 칸에 있던 적 말은 사라진다.
        board[move.target] = board[move.source]
        board[move.source] = ()
        return Position(
            board=tuple(board),
            side=position.side.opponent,
            ply=position.ply + 1,
            previous=position,
        )

    def _targets(self, position: Position, source: int) -> list[int]:
        tower = position.board[source]
        self._require_single_tier(tower)
        kind = tower[-1].kind
        if kind not in STEPS:
            raise NotImplementedError(f"아직 이동 규칙이 없는 말이다: {kind.name} (06 번 티켓)")

        row, column = divmod(source, self.config.width)
        targets: list[int] = []
        for column_step, row_step in STEPS[kind]:
            to_column, to_row = column + column_step, row + row_step
            if not (0 <= to_column < self.config.width and 0 <= to_row < self.config.height):
                continue
            target = to_row * self.config.width + to_column
            occupant = position.board[target]
            if occupant:
                self._require_single_tier(occupant)
                if occupant[-1].owner is position.side:
                    continue
            targets.append(target)
        return targets

    def _require_single_tier(self, tower: Tower) -> None:
        if len(tower) > 1:
            raise NotImplementedError("탑의 규칙은 아직 없다 (07 번 티켓)")
