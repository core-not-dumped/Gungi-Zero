"""`engine/` 을 `games.base.Game` 으로 감싸는 어댑터. 행동 인덱스, 텐서, 텍스트 표기를 소유한다."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Hashable
from dataclasses import dataclass, replace
from typing import final

import numpy as np
import numpy.typing as npt

from engine.pieces import Piece, PieceType, Side
from engine.position import Position, Tower
from engine.rules import Move as GungiMove
from engine.rules import RuleEngine
from engine.variant import Placement, VariantConfig
from games.base import Outcome, Player

__all__ = [
    "MARSHAL_ONLY",
    "MINI_GUNGI",
    "SINGLE_TIER",
    "Gungi",
    "GungiMove",
    "GungiPosition",
    "PieceType",
    "Placement",
    "VariantConfig",
]

# 帥 하나씩만 올린 변형. 서로 피하면 반복 규칙이나 수 제한이 대국을 끝낸다.
MARSHAL_ONLY = VariantConfig(
    width=5,
    height=5,
    max_tier=2,
    setup=(Placement(PieceType.MARSHAL, column=2, row=0),),
    move_limit=200,
    repetition_limit=3,
)

# 미니 군의의 기본 설정. 배치와 말 개수는 노브다 — 1단계 끝에 측정으로 확정한다.
MINI_GUNGI = VariantConfig(
    width=5,
    height=5,
    max_tier=2,
    setup=(
        Placement(PieceType.GENERAL, column=1, row=0),
        Placement(PieceType.MARSHAL, column=2, row=0),
        Placement(PieceType.LANCE, column=3, row=0),
        Placement(PieceType.PAWN, column=1, row=1),
        Placement(PieceType.PAWN, column=3, row=1),
    ),
    move_limit=200,
    repetition_limit=3,
)

# 같은 배치에서 쌓기만 뺀 변형.
SINGLE_TIER = replace(MINI_GUNGI, max_tier=1)

EMPTY_CELL = "."
TOWER_SEPARATOR = "/"
COLUMNS = "abcdefghijklmnopqrstuvwxyz"
SYMBOLS = {
    PieceType.MARSHAL: "帥",
    PieceType.GENERAL: "大",
    PieceType.LANCE: "槍",
    PieceType.PAWN: "兵",
}
KINDS_BY_SYMBOL = {symbol: kind for kind, symbol in SYMBOLS.items()}
PLAYERS = {Side.FIRST: Player.FIRST, Side.SECOND: Player.SECOND}
SIDES = {player: side for side, player in PLAYERS.items()}

STACK_MARK = "^"
SQUARE_PATTERN = r"([a-z])([1-9][0-9]*)"
MOVE_PATTERN = re.compile(f"{SQUARE_PATTERN}-{SQUARE_PATTERN}({re.escape(STACK_MARK)}?)")

# 텐서 평면 = (내 말, 상대 말) * 말 종류 * 단. 그 뒤에 반복 횟수와 수 진행도가 한 장씩 온다.
# 평면 번호는 신경망 가중치의 의미이므로 순서를 바꾸지 않는다.
PLANE_KINDS = (PieceType.MARSHAL, PieceType.GENERAL, PieceType.LANCE, PieceType.PAWN)
PLANE_TIERS = 2
PLANE_COUNT = 2 * len(PLANE_KINDS) * PLANE_TIERS + 2

# 렌더의 열 간격과 첫 열의 중심 (화면 폭 기준, 한자는 두 칸이다).
CELL_PITCH = 8
FIRST_CELL_CENTER = 5


@dataclass(frozen=True)
class GungiPosition:
    """엔진의 국면을 `games.base.Position` 으로 보여준다."""

    engine_position: Position

    @property
    def player(self) -> Player:
        return PLAYERS[self.engine_position.side]

    @property
    def ply(self) -> int:
        return self.engine_position.ply


def _display_width(text: str) -> int:
    return sum(2 if unicodedata.east_asian_width(char) in "WF" else 1 for char in text)


def _piece_count(position: GungiPosition) -> int:
    return sum(len(tower) for tower in position.engine_position.board)


@final
class Gungi:
    """변형 설정 하나의 군의."""

    def __init__(self, config: VariantConfig) -> None:
        if config.width > len(COLUMNS):
            raise ValueError(f"열 표기는 {len(COLUMNS)} 개까지다: {config.width}")
        if config.max_tier > PLANE_TIERS:
            raise ValueError(f"텐서 평면은 {PLANE_TIERS} 단까지다: {config.max_tier}")
        self._config = config
        self._engine = RuleEngine(config)

    def initial_position(self) -> GungiPosition:
        return GungiPosition(self._engine.initial_position())

    def render(self, position: GungiPosition) -> str:
        width, height = self._config.width, self._config.height
        header = " " * FIRST_CELL_CENTER + (" " * (CELL_PITCH - 1)).join(COLUMNS[:width])
        lines = [header]
        for row in reversed(range(height)):
            line = f"{row + 1:>2}"
            for column in range(width):
                cell = self._render_tower(position.engine_position.board[row * width + column])
                # 칸을 열 중심에 맞추되, 넓은 탑끼리도 빈칸 하나는 떨어뜨린다.
                start = FIRST_CELL_CENTER + column * CELL_PITCH - _display_width(cell) // 2
                line += " " * max(1, start - _display_width(line)) + cell
            lines.append(line)
        lines.append(f"차례: {position.player.value}   ply: {position.ply}")
        return "\n".join(lines) + "\n"

    def parse(self, text: str) -> GungiPosition:
        """`render` 의 출력을 읽는다. 텍스트에는 경로가 없어 반복 횟수는 0 이다."""
        width, height = self._config.width, self._config.height
        lines = [line for line in text.splitlines() if line.strip()]
        if len(lines) != height + 2:
            raise ValueError(f"판 표기는 {height + 2} 줄이어야 한다: {len(lines)} 줄을 받았다")

        header = lines[0].split()
        if header != list(COLUMNS[:width]):
            raise ValueError(f"열 머리글이 {list(COLUMNS[:width])} 가 아니다: {header}")

        board: list[Tower] = [()] * self._config.square_count
        for row, line in zip(reversed(range(height)), lines[1 : height + 1], strict=True):
            tokens = line.split()
            if tokens[0] != str(row + 1):
                raise ValueError(f"{row + 1} 행을 기대했는데 {tokens[0]!r} 행을 받았다")
            cells = tokens[1:]
            if len(cells) != width:
                raise ValueError(f"{row + 1} 행의 칸이 {width} 개가 아니다: {cells}")
            for column, cell in enumerate(cells):
                board[row * width + column] = self._parse_tower(cell)

        for side in Side:
            marshals = sum(
                1 for tower in board for piece in tower if piece == Piece(PieceType.MARSHAL, side)
            )
            if marshals > 1:
                raise ValueError(
                    f"帥 는 한 편에 하나뿐이다: {PLAYERS[side].value} 가 {marshals} 개"
                )

        turn_tokens = lines[height + 1].split()
        if len(turn_tokens) != 4 or turn_tokens[0] != "차례:" or turn_tokens[2] != "ply:":
            raise ValueError(f"차례 줄이 '차례: ▲   ply: 0' 꼴이 아니다: {lines[height + 1]!r}")
        ply = int(turn_tokens[3])
        if ply < 0:
            raise ValueError(f"ply 는 0 이상이다: {ply}")
        side = SIDES[Player(turn_tokens[1])]
        # 둘 차례인 쪽의 帥 는 방금 잡혔을 수 있지만, 방금 둔 쪽의 帥 는 반드시 남아 있다.
        mover = side.opponent
        if not any(Piece(PieceType.MARSHAL, mover) in tower for tower in board):
            raise ValueError(f"방금 둔 {PLAYERS[mover].value} 의 帥 가 판에 없다")
        return GungiPosition(Position(board=tuple(board), side=side, ply=ply))

    def _render_tower(self, tower: Tower) -> str:
        if not tower:
            return EMPTY_CELL
        return TOWER_SEPARATOR.join(
            PLAYERS[piece.owner].value + SYMBOLS[piece.kind] for piece in reversed(tower)
        )

    def _parse_tower(self, text: str) -> Tower:
        if text == EMPTY_CELL:
            return ()
        names = text.split(TOWER_SEPARATOR)
        if len(names) > self._config.max_tier:
            raise ValueError(f"최대 {self._config.max_tier} 단을 넘는 탑이다: {text!r}")
        pieces: list[Piece] = []
        for name in reversed(names):
            if len(name) != 2 or name[1] not in KINDS_BY_SYMBOL:
                raise ValueError(f"말 표기가 '▲帥' 꼴이 아니다: {name!r}")
            pieces.append(Piece(KINDS_BY_SYMBOL[name[1]], SIDES[Player(name[0])]))
        if any(piece.kind is PieceType.MARSHAL for piece in pieces[:-1]):
            raise ValueError(f"帥 위에는 말을 쌓을 수 없다: {text!r}")
        return tuple(pieces)

    def legal_moves(self, position: GungiPosition) -> tuple[GungiMove, ...]:
        return self._engine.legal_moves(position.engine_position)

    def apply_move(self, position: GungiPosition, move: GungiMove) -> GungiPosition:
        return GungiPosition(self._engine.apply_move(position.engine_position, move))

    def outcome(self, position: GungiPosition) -> Outcome | None:
        result = self._engine.result(position.engine_position)
        if result is None:
            return None
        if result.winner is None:
            return Outcome.DRAW
        return Outcome.win_for(PLAYERS[result.winner])

    @property
    def action_space_size(self) -> int:
        """출발 칸 * 도착 칸 * 쌓기 여부 (ADR-0006)."""
        return 2 * self._square_pair_count

    @property
    def _square_pair_count(self) -> int:
        """쌓기 여부 하나에 해당하는 행동 인덱스의 개수. 출발 칸 * 도착 칸이다."""
        return self._config.square_count**2

    def action_index(self, move: GungiMove) -> int:
        squares = move.source * self._config.square_count + move.target
        return move.stack * self._square_pair_count + squares

    def move_from_action_index(self, index: int) -> GungiMove:
        if not 0 <= index < self.action_space_size:
            raise ValueError(f"행동 공간 밖의 인덱스다: {index}")
        stack, squares = divmod(index, self._square_pair_count)
        source, target = divmod(squares, self._config.square_count)
        return GungiMove(source, target, stack=bool(stack))

    def legal_action_mask(self, position: GungiPosition) -> npt.NDArray[np.bool_]:
        mask = np.zeros(self.action_space_size, dtype=bool)
        for move in self.legal_moves(position):
            mask[self.action_index(move)] = True
        return mask

    def canonical(self, position: GungiPosition) -> GungiPosition:
        if position.player is Player.FIRST:
            return position
        return GungiPosition(position.engine_position.rotated())

    def to_canonical_action_index(self, position: GungiPosition, index: int) -> int:
        if position.player is Player.FIRST:
            return index
        # 180° 회전은 칸 i 를 칸 수-1-i 로 보낸다. 출발·도착에 모두 적용하면 평면 안에서
        # 뒤집는 식이 된다. 쌓기 여부는 그대로다.
        stack, squares = divmod(index, self._square_pair_count)
        return stack * self._square_pair_count + self._square_pair_count - 1 - squares

    def from_canonical_action_index(self, position: GungiPosition, index: int) -> int:
        # 180° 회전은 두 번 하면 제자리라 역변환이 같다.
        return self.to_canonical_action_index(position, index)

    @property
    def tensor_shape(self) -> tuple[int, int, int]:
        return (PLANE_COUNT, self._config.height, self._config.width)

    def encode(self, position: GungiPosition) -> npt.NDArray[np.float32]:
        """말 평면 16장 + 반복 횟수 + 수 진행도.

        반복 횟수는 앞선 출현 횟수를 무승부가 되는 값(`repetition_limit - 1`)으로 나눈 것이고,
        수 진행도는 `ply / move_limit` 이다.
        """
        tensor = np.zeros(self.tensor_shape, dtype=np.float32)
        seen = self.canonical(position).engine_position
        for square, tower in enumerate(seen.board):
            row, column = divmod(square, self._config.width)
            for tier, piece in enumerate(tower):
                owner = 0 if piece.owner is seen.side else 1
                kind = PLANE_KINDS.index(piece.kind)
                plane = (owner * len(PLANE_KINDS) + kind) * PLANE_TIERS + tier
                tensor[plane, row, column] = 1.0
        earlier = position.engine_position.earlier_occurrences
        tensor[-2] = earlier / (self._config.repetition_limit - 1)
        tensor[-1] = position.ply / self._config.move_limit
        return tensor

    def board_key(self, position: GungiPosition) -> Hashable:
        return position.engine_position.board_key

    def render_move(self, move: GungiMove) -> str:
        squares = f"{self._render_square(move.source)}-{self._render_square(move.target)}"
        return squares + (STACK_MARK if move.stack else "")

    def render_log_move(self, position: GungiPosition, move: GungiMove) -> str:
        """로그에 쓰는 표기. 수 표기에 잡기면 `x` 를 덧붙인다. 합법수가 아니면 `ValueError` 다.

        `parse_move` 는 이 표기를 읽지 않는다.
        """
        after = self.apply_move(position, move)
        # 말이 줄었으면 잡기다. 쌓기는 말을 옮길 뿐이라 이 판정에 걸리지 않는다.
        captured = _piece_count(after) < _piece_count(position)
        return self.render_move(move) + ("x" if captured else "")

    def parse_move(self, text: str) -> GungiMove:
        match = MOVE_PATTERN.fullmatch(text.strip())
        if match is None:
            raise ValueError(f"수 표기가 'b1-c3' 또는 'b1-c3^' 꼴이 아니다: {text!r}")
        source_column, source_row, target_column, target_row, mark = match.groups()
        return GungiMove(
            self._parse_square(source_column, source_row),
            self._parse_square(target_column, target_row),
            stack=mark == STACK_MARK,
        )

    def _render_square(self, square: int) -> str:
        row, column = divmod(square, self._config.width)
        return f"{COLUMNS[column]}{row + 1}"

    def _parse_square(self, column_name: str, row_name: str) -> int:
        column, row = COLUMNS.index(column_name), int(row_name) - 1
        if column >= self._config.width or row >= self._config.height:
            raise ValueError(f"판 밖의 칸이다: {column_name}{row_name}")
        return row * self._config.width + column
