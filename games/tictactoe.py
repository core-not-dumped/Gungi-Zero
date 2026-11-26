"""틱택토. 룰 엔진 없이 직접 구현한다.

`Game` 프로토콜이 군의 모양으로 편향되지 않았는지 보는 눈이다.
"""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from typing import final

import numpy as np
import numpy.typing as npt

from games.base import Outcome, Player

SIZE = 3
SQUARE_COUNT = SIZE * SIZE

EMPTY_CELL = "."
COLUMNS = "abc"


# 세 칸이 한 줄을 이루는 조합. 가로 3 · 세로 3 · 대각 2. `SIZE` 에서 유도하지 않고 적었다 —
# 사람이 눈으로 검토할 수 있는 표가 룰 코드와 기대 정답이 같은 오해를 공유하는 것을 막는다.
LINES: tuple[tuple[int, int, int], ...] = (
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),
    (0, 4, 8),
    (2, 4, 6),
)


@dataclass(frozen=True, slots=True)
class TicTacToeMove:
    """한 칸에 자기 말을 놓는다. 칸 번호가 수를 모호함 없이 결정한다."""

    square: int


@dataclass(frozen=True, slots=True)
class TicTacToePosition:
    """판 + 차례. 틱택토에는 반복이 없어 경로를 들고 다닐 필요가 없다.

    `board` 는 칸 번호 순서다 — 0 = a1, 1 = b1, ... 8 = c3. 즉 행 1(선수 진영)이 먼저다.
    `player` 는 판에서 유도하지 않는다. 정규형은 색을 바꾸므로 "선수가 먼저 둔다" 가 깨진 판도
    정상적인 국면이다.
    """

    board: tuple[Player | None, ...]
    player: Player
    # `TicTacToe.canonical` 이 돌려준 국면인지. 원래 국면의 자리에 정규형을 넘기는 실수를 막는다.
    canonical_view: bool = False

    @property
    def ply(self) -> int:
        return sum(1 for cell in self.board if cell is not None)


def _square(column: int, row: int) -> int:
    """열·행(0 기준) → 칸 번호."""
    return row * SIZE + column


def _column_and_row(square: int) -> tuple[int, int]:
    """칸 번호 → 열·행(0 기준). `_square` 의 역이다."""
    return square % SIZE, square // SIZE


@final
class TicTacToe:
    """틱택토의 규칙."""

    def initial_position(self) -> TicTacToePosition:
        return TicTacToePosition(board=(None,) * SQUARE_COUNT, player=Player.FIRST)

    def legal_moves(self, position: TicTacToePosition) -> tuple[TicTacToeMove, ...]:
        if self.outcome(position) is not None:
            return ()
        return tuple(
            TicTacToeMove(square) for square, cell in enumerate(position.board) if cell is None
        )

    def apply_move(self, position: TicTacToePosition, move: TicTacToeMove) -> TicTacToePosition:
        if move not in self.legal_moves(position):
            raise ValueError(f"합법수가 아니다: {self.render_move(move)}")
        board = list(position.board)
        board[move.square] = position.player
        return TicTacToePosition(board=tuple(board), player=position.player.opponent)

    def outcome(self, position: TicTacToePosition) -> Outcome | None:
        for line in LINES:
            owner = position.board[line[0]]
            if owner is not None and all(position.board[square] is owner for square in line):
                return Outcome.win_for(owner)
        if all(cell is not None for cell in position.board):
            return Outcome.DRAW
        return None

    def board_outcome(self, position: TicTacToePosition) -> Outcome | None:
        # 틱택토의 종료는 전부 판이 정한다.
        return self.outcome(position)

    @property
    def action_space_size(self) -> int:
        return SQUARE_COUNT

    def action_index(self, move: TicTacToeMove) -> int:
        if not 0 <= move.square < SQUARE_COUNT:
            raise ValueError(f"판 밖의 칸을 가리키는 수다: {move}")
        return move.square

    def move_from_action_index(self, index: int) -> TicTacToeMove:
        self._check_action_index(index)
        return TicTacToeMove(index)

    def _check_action_index(self, index: int) -> None:
        if not 0 <= index < SQUARE_COUNT:
            raise ValueError(f"행동 공간 밖의 인덱스다: {index}")

    def legal_action_mask(self, position: TicTacToePosition) -> npt.NDArray[np.bool_]:
        mask = np.zeros(SQUARE_COUNT, dtype=bool)
        for move in self.legal_moves(position):
            mask[self.action_index(move)] = True
        return mask

    def canonical(self, position: TicTacToePosition) -> TicTacToePosition:
        if position.canonical_view:
            return position
        board = position.board
        if position.player is Player.SECOND:
            board = tuple(None if cell is None else cell.opponent for cell in board)
        return TicTacToePosition(board=board, player=Player.FIRST, canonical_view=True)

    def to_canonical_action_index(self, position: TicTacToePosition, index: int) -> int:
        if position.canonical_view:
            raise ValueError("정규형이 아니라 원래 국면을 넘겨야 한다")
        self._check_action_index(index)
        # 틱택토의 정규형은 색만 바꾼다 — 판을 돌리지 않으므로 칸 번호가 그대로다.
        # 군의는 180° 회전이라 여기서 칸이 옮겨진다.
        return index

    def from_canonical_action_index(self, position: TicTacToePosition, index: int) -> int:
        return self.to_canonical_action_index(position, index)

    @property
    def tensor_shape(self) -> tuple[int, int, int]:
        return (2, SIZE, SIZE)

    def encode(self, position: TicTacToePosition) -> npt.NDArray[np.float32]:
        tensor = np.zeros(self.tensor_shape, dtype=np.float32)
        for square, cell in enumerate(position.board):
            if cell is None:
                continue
            plane = 0 if cell is position.player else 1
            column, row = _column_and_row(square)
            tensor[plane, row, column] = 1.0
        return tensor

    def board_key(self, position: TicTacToePosition) -> Hashable:
        return (position.board, position.player)

    def render_move(self, move: TicTacToeMove) -> str:
        column, row = _column_and_row(move.square)
        return f"{COLUMNS[column]}{row + 1}"

    def parse_move(self, text: str) -> TicTacToeMove:
        stripped = text.strip()
        if len(stripped) != 2 or stripped[0] not in COLUMNS or not stripped[1].isdigit():
            raise ValueError(f"수 표기가 'b2' 꼴이 아니다: {text!r}")
        row = int(stripped[1]) - 1
        if not 0 <= row < SIZE:
            raise ValueError(f"판 밖의 행이다: {text!r}")
        return TicTacToeMove(_square(COLUMNS.index(stripped[0]), row))

    def render(self, position: TicTacToePosition) -> str:
        lines = ["    " + "".join(f" {column}  " for column in COLUMNS).rstrip()]
        for row in reversed(range(SIZE)):
            cells = (position.board[_square(column, row)] for column in range(SIZE))
            rendered = (EMPTY_CELL if cell is None else cell.value for cell in cells)
            lines.append(f" {row + 1}" + "".join(f"   {cell}" for cell in rendered))
        lines.append(f"차례: {position.player.value}   ply: {position.ply}")
        return "\n".join(lines) + "\n"

    def parse(self, text: str) -> TicTacToePosition:
        lines = [line for line in text.splitlines() if line.strip()]
        if len(lines) != SIZE + 2:
            raise ValueError(f"판 표기는 {SIZE + 2} 줄이어야 한다: {len(lines)} 줄을 받았다")

        header = lines[0].split()
        if header != list(COLUMNS):
            raise ValueError(f"열 머리글이 {list(COLUMNS)} 가 아니다: {header}")

        board: list[Player | None] = [None] * SQUARE_COUNT
        for row, line in zip(reversed(range(SIZE)), lines[1 : SIZE + 1], strict=True):
            tokens = line.split()
            if tokens[0] != str(row + 1):
                raise ValueError(f"{row + 1} 행을 기대했는데 {tokens[0]!r} 행을 받았다")
            cells = tokens[1:]
            if len(cells) != SIZE:
                raise ValueError(f"{row + 1} 행의 칸이 {SIZE} 개가 아니다: {cells}")
            for column, cell in enumerate(cells):
                board[_square(column, row)] = None if cell == EMPTY_CELL else Player(cell)

        turn_tokens = lines[SIZE + 1].split()
        if len(turn_tokens) != 4 or turn_tokens[0] != "차례:" or turn_tokens[2] != "ply:":
            raise ValueError(f"차례 줄이 '차례: ▲   ply: 0' 꼴이 아니다: {lines[SIZE + 1]!r}")
        position = TicTacToePosition(board=tuple(board), player=Player(turn_tokens[1]))
        if position.ply != int(turn_tokens[3]):
            raise ValueError(
                f"ply 가 판과 맞지 않는다: 판은 {position.ply}, 표기는 {turn_tokens[3]}"
            )
        return position
