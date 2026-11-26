"""룰 엔진. 변형 설정 하나를 받아 합법수·수 두기·승패를 답한다."""

from __future__ import annotations

from dataclasses import dataclass

from engine.movement import MOVEMENT
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


def blocks_path(mover_tier: int, occupant: Tower) -> bool:
    """`mover_tier` 단의 말이 `occupant` 가 있는 칸을 지나 더 갈 수 없는지.

    점유된 칸은 단수와 무관하게 막는다 (ADR-0002). 그래서 지금은 `mover_tier` 를 보지 않는다 —
    다른 해석과 A/B 할 때 이 술어만 바꾼다.
    """
    return bool(occupant)


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
        """`source` 의 맨 위 말이 갈 수 있는 칸. 빈칸으로의 이동과 적 말 잡기가 같은 길이다."""
        tower = position.board[source]
        self._require_single_tier(tower)
        mover = tower[-1]
        # 표의 앞은 선수의 앞이다. 후수는 180° 돌려 본다 (ADR-0003).
        facing = 1 if mover.owner is Side.FIRST else -1
        # 판의 긴 변만큼 가면 어느 방향이든 판 밖이다.
        board_span = max(self.config.width, self.config.height)

        row, column = divmod(source, self.config.width)
        targets: list[int] = []
        for ray in MOVEMENT[mover.kind]:
            reach = board_span if ray.reach is None else ray.reach
            for distance in range(1, reach + 1):
                to_column = column + facing * ray.column_step * distance
                to_row = row + facing * ray.row_step * distance
                if not (0 <= to_column < self.config.width and 0 <= to_row < self.config.height):
                    break
                target = to_row * self.config.width + to_column
                occupant = position.board[target]
                if occupant:
                    self._require_single_tier(occupant)
                    if occupant[-1].owner is not position.side:
                        targets.append(target)
                    else:
                        self._fail_if_stackable(mover, occupant[-1])
                else:
                    targets.append(target)
                if blocks_path(len(tower), occupant):
                    break
        return targets

    def _require_single_tier(self, tower: Tower) -> None:
        if len(tower) > 1:
            raise NotImplementedError("탑의 규칙은 아직 없다 (07 번 티켓)")

    def _fail_if_stackable(self, mover: Piece, below: Piece) -> None:
        """아군 위에 쌓을 수 있는 수를 조용히 빠뜨리지 않는다. 帥 가 끼면 쌓기는 없다."""
        if self.config.max_tier > 1 and PieceType.MARSHAL not in (mover.kind, below.kind):
            raise NotImplementedError("쌓기 규칙은 아직 없다 (07 번 티켓)")
