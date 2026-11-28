"""룰 엔진. 변형 설정 하나를 받아 합법수·수 두기·승패를 답한다."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

from engine.geometry import forward
from engine.movement import MOVEMENT
from engine.pieces import Piece, PieceType, Side
from engine.position import Board, Position, Tower
from engine.variant import VariantConfig

# 엔진 하나가 판만 보는 답을 기억하는 판 + 차례의 상한. 넘으면 모두 잊고 다시 모은다.
REMEMBERED_BOARDS = 100_000


@dataclass(frozen=True, slots=True)
class Move:
    """출발 칸, 도착 칸, 쌓기 여부 (ADR-0006).

    맨 위가 적 말인 칸에는 잡기(`stack=False`)와 쌓기(`stack=True`)가 따로 있다.
    """

    source: int
    target: int
    stack: bool = False


class EndReason(Enum):
    """대국이 끝난 사유 (ADR-0004)."""

    MARSHAL_CAPTURE = auto()  # 帥 잡기
    NO_LEGAL_MOVES = auto()  # 합법수 없음
    REPETITION = auto()  # 같은 판 반복
    MOVE_LIMIT = auto()  # 수 제한


@dataclass(frozen=True, slots=True)
class Result:
    """끝난 대국의 결과. `winner` 가 `None` 이면 무승부다."""

    winner: Side | None
    reason: EndReason


def blocks_path(mover_tier: int, occupant: Tower) -> bool:
    """`mover_tier` 단의 말이 `occupant` 가 있는 칸을 지나 더 갈 수 없는지.

    점유된 칸은 단수와 무관하게 막는다 (ADR-0002). 그래서 지금은 `mover_tier` 를 보지 않는다 —
    다른 해석과 A/B 할 때 이 술어만 바꾼다.
    """
    return bool(occupant)


# 한 방향의 길: 가까운 칸부터 (도착 칸, 이동·잡기의 수, 쌓기의 수).
Path = tuple[tuple[int, Move, Move], ...]


@dataclass(frozen=True, slots=True)
class RuleEngine:
    config: VariantConfig
    # 판 + 차례 → 판만 보는 답. 같은 판을 되풀이해 묻는 일이 많다.
    _boards: dict[tuple[Board, Side], tuple[Result | None, tuple[Move, ...]]] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    # (말, 단, 칸) → 그 말이 빈 판에서 뻗는 길들. 처음 쓸 때 만든다.
    _paths: dict[tuple[Piece, int, int], tuple[Path, ...]] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def initial_position(self) -> Position:
        geometry = self.config.geometry
        board: list[Tower] = [()] * geometry.square_count
        for placement in self.config.setup:
            square = geometry.square(placement.column, placement.row)
            board[square] = (Piece(placement.kind, Side.FIRST),)
            board[geometry.rotate(square)] = (Piece(placement.kind, Side.SECOND),)
        return Position(board=tuple(board), side=Side.FIRST, ply=0)

    def result(self, position: Position) -> Result | None:
        """끝난 국면의 결과, 진행 중이면 `None`."""
        return self._evaluate(position)[0]

    def board_result(self, position: Position) -> Result | None:
        """판 + 차례만으로 끝난 국면의 결과 (帥 잡기, 합법수 없음), 아니면 `None`.

        반복과 수 제한은 보지 않는다.
        """
        return self._evaluate_board(position)[0]

    def legal_moves(self, position: Position) -> tuple[Move, ...]:
        """둘 수 있는 모든 수. 끝난 국면에서는 비어 있다."""
        return self._evaluate(position)[1]

    def _evaluate(self, position: Position) -> tuple[Result | None, tuple[Move, ...]]:
        """결과와 합법수를 함께 낸다 — 합법수 없음이 종료 사유라 둘을 따로 정할 수 없다.

        승패를 무승부보다 먼저 본다. 마지막 수로 帥 를 잡으면 승리이고, 수 제한에 닿은
        국면이라도 둘 수가 없으면 패배다. 반복과 수 제한에 함께 닿으면 사유는 반복이다.
        """
        result, moves = self._evaluate_board(position)
        if result is not None:
            return result, ()
        # 한도는 지금 나온 것까지 센다.
        if position.earlier_occurrences + 1 >= self.config.repetition_limit:
            return Result(None, EndReason.REPETITION), ()
        if position.ply >= self.config.move_limit:
            return Result(None, EndReason.MOVE_LIMIT), ()
        return None, moves

    def _evaluate_board(self, position: Position) -> tuple[Result | None, tuple[Move, ...]]:
        """`_evaluate` 에서 판 + 차례만 보는 부분. 끝나지 않았으면 말의 이동도 함께 낸다.

        답은 판과 차례만으로 정해져 기억해 둔다. 경로에 달린 반복과 수 제한은 기억하지 않는다.
        """
        key = (position.board, position.side)
        answer = self._boards.get(key)
        if answer is None:
            if len(self._boards) >= REMEMBERED_BOARDS:
                self._boards.clear()
            answer = self._boards[key] = self._judge_board(position)
        return answer

    def _judge_board(self, position: Position) -> tuple[Result | None, tuple[Move, ...]]:
        alive = {
            piece.owner
            for tower in position.board
            for piece in tower
            if piece.kind is PieceType.MARSHAL
        }
        for side in Side:
            if side not in alive:
                return Result(side.opponent, EndReason.MARSHAL_CAPTURE), ()
        moves = self._moves(position)
        if not moves:
            return Result(position.side.opponent, EndReason.NO_LEGAL_MOVES), ()
        return None, moves

    def _moves(self, position: Position) -> tuple[Move, ...]:
        """종료 규칙을 보지 않고, 말의 이동만으로 나오는 수."""
        moves: list[Move] = []
        for source, tower in enumerate(position.board):
            if not tower or tower[-1].owner is not position.side:
                continue
            moves.extend(self._piece_moves(position, source))
        return tuple(moves)

    def apply_move(self, position: Position, move: Move) -> Position:
        """수를 둔 새 국면. 합법수가 아니면 `ValueError` 다."""
        if move not in self.legal_moves(position):
            raise ValueError(f"합법수가 아니다: {move}")
        board = list(position.board)
        *left_behind, mover = board[move.source]
        below = board[move.target]
        if not move.stack:
            # 이동이나 잡기다. 도착 칸의 적 말은 전부 사라지고 내 말은 남는다.
            below = tuple(piece for piece in below if piece.owner is mover.owner)
        board[move.source] = tuple(left_behind)
        board[move.target] = (*below, mover)
        return Position(
            board=tuple(board),
            side=position.side.opponent,
            ply=position.ply + 1,
            previous=position,
        )

    def _piece_moves(self, position: Position, source: int) -> list[Move]:
        """`source` 의 맨 위 말이 두는 수. 아래에 깔린 말은 움직이지 않는다."""
        board = position.board
        tower = board[source]
        mover = tower[-1]
        tier = len(tower)
        key = (mover, tier, source)
        paths = self._paths.get(key)
        if paths is None:
            paths = self._paths[key] = self._paths_of(mover, tier, source)

        moves: list[Move] = []
        for path in paths:
            for target, plain, stacked in path:
                occupant = board[target]
                if not occupant:
                    moves.append(plain)
                    continue
                takes, stacks = self._choices(mover, tier, occupant)
                if takes:
                    moves.append(plain)
                if stacks:
                    moves.append(stacked)
                if blocks_path(tier, occupant):
                    break
        return moves

    def _paths_of(self, mover: Piece, tier: int, source: int) -> tuple[Path, ...]:
        """`source` 의 `tier` 단에 있는 `mover` 가 빈 판에서 뻗는 길들. 사거리 표의 방향마다
        하나이고 가까운 칸부터다. 판에 놓인 말과 상관없어 한 번만 계산한다.
        """
        # 표의 앞은 선수의 앞이다.
        facing = forward(mover.owner)
        geometry = self.config.geometry
        # 판의 긴 변만큼 가면 어느 방향이든 판 밖이다.
        board_span = max(geometry.width, geometry.height)

        column, row = geometry.column_and_row(source)
        paths: list[Path] = []
        for ray in MOVEMENT[mover.kind]:
            reach = ray.reach_at(tier)
            path: list[tuple[int, Move, Move]] = []
            for distance in range(1, (board_span if reach is None else reach) + 1):
                to_column = column + facing * ray.column_step * distance
                to_row = row + facing * ray.row_step * distance
                if not geometry.contains(to_column, to_row):
                    break
                target = geometry.square(to_column, to_row)
                path.append((target, Move(source, target), Move(source, target, stack=True)))
            if path:
                paths.append(tuple(path))
        return tuple(paths)

    def _choices(self, mover: Piece, tier: int, occupant: Tower) -> tuple[bool, bool]:
        """`tier` 단의 `mover` 가 말이 있는 `occupant` 칸에서 고를 수 있는 수 (ADR-0006):
        `(잡기, 쌓기)`.

        아군은 쌓기뿐이고, 적은 잡기와 쌓기 둘 다다.
        """
        # 단수 제약: 자기보다 높은 단의 말은 잡을 수도 쌓을 수도 없다.
        if len(occupant) > tier:
            return False, False
        top = occupant[-1]
        # 帥 위에는 누구도 못 올라가고, 帥 자신도 어디에도 못 올라간다 (帥ツケ 금지).
        stacks = (
            len(occupant) < self.config.max_tier
            and top.kind is not PieceType.MARSHAL
            and mover.kind is not PieceType.MARSHAL
        )
        return top.owner is not mover.owner, stacks
