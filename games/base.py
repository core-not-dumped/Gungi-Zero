"""모든 게임이 따르는 공통 인터페이스.

`ai/` 가 게임에 대해 아는 것은 이 모듈뿐이다 (ADR-0001). 용어는 `CONTEXT.md` 를 따른다 —
`Game` 은 게임 종류, `Position` 은 한 국면이다.

**행동 인덱스의 프레임**: 모든 연산은 받은 `Position` 의 프레임에서 답한다. 신경망에 쓸 때는
`canonical(p)` 를 먼저 얻어 그 국면에 `encode`·`legal_action_mask`·`legal_moves` 를 쓰면
전부 정규형 프레임이다. 신경망이 고른 인덱스는 `from_canonical_action_index(p, i)` 로
원래 프레임으로 되돌린 뒤 `move_from_action_index` 로 수가 된다.
"""

from __future__ import annotations

from collections.abc import Hashable
from enum import Enum, auto
from typing import Protocol, TypeVar

import numpy as np
import numpy.typing as npt


class Player(Enum):
    """선수(▲)와 후수(▽). 정규형에서 움직이는 쪽은 항상 `FIRST` 다."""

    FIRST = "▲"
    SECOND = "▽"

    @property
    def opponent(self) -> Player:
        return Player.SECOND if self is Player.FIRST else Player.FIRST


class Outcome(Enum):
    """끝난 대국의 결과. 둘 차례인 쪽이 아니라 선수·후수로 적는다 — 기록이 시점에 안 흔들린다."""

    FIRST_WINS = auto()
    SECOND_WINS = auto()
    DRAW = auto()

    @classmethod
    def win_for(cls, player: Player) -> Outcome:
        return cls.FIRST_WINS if player is Player.FIRST else cls.SECOND_WINS

    @property
    def winner(self) -> Player | None:
        """이긴 쪽, 무승부면 `None`."""
        if self is Outcome.DRAW:
            return None
        return Player.FIRST if self is Outcome.FIRST_WINS else Player.SECOND


class Position(Protocol):
    """한 국면. 불변이고 해시 가능하다. `Game` 프로토콜에 `undo` 가 없는 이유다."""

    @property
    def player(self) -> Player:
        """이 국면에서 둘 차례인 쪽."""
        ...

    @property
    def ply(self) -> int:
        """초기 국면부터 둔 수의 개수. 초기 국면은 0 이다."""
        ...

    def __hash__(self) -> int: ...


class Move(Protocol):
    """한 국면에서 둘 수 있는 하나의 수. 불변이고 해시 가능하다.

    공통으로 요구하는 것은 그것뿐이다 — 칸의 해석과 표기는 게임마다 다르고 `Game` 이 소유한다.
    """

    def __hash__(self) -> int: ...


PositionT = TypeVar("PositionT", bound=Position)
MoveT = TypeVar("MoveT", bound=Move)


class Game(Protocol[PositionT, MoveT]):
    """게임의 규칙 그 자체. 상태를 갖지 않는다."""

    def initial_position(self) -> PositionT:
        """고정된 시작 국면."""
        ...

    def legal_moves(self, position: PositionT) -> tuple[MoveT, ...]:
        """이 국면에서 둘 수 있는 모든 수. 종료 국면에서는 비어 있다."""
        ...

    def apply_move(self, position: PositionT, move: MoveT) -> PositionT:
        """수를 둔 새 `Position`. 합법수가 아니면 `ValueError` 다."""
        ...

    def outcome(self, position: PositionT) -> Outcome | None:
        """끝난 국면의 결과, 아직 진행 중이면 `None`.

        종료 여부와 결과를 한 연산으로 둔다 — 둘로 나누면 서로 어긋날 수 있다.
        """
        ...

    @property
    def action_space_size(self) -> int:
        """정책 머리의 고정 출력 크기. 국면과 무관하다."""
        ...

    def action_index(self, move: MoveT) -> int:
        """`Move` → 행동 인덱스. 국면과 무관한 순수 좌표 변환이다."""
        ...

    def move_from_action_index(self, index: int) -> MoveT:
        """행동 인덱스 → `Move`. 어떤 수도 가리키지 않는 인덱스면 `ValueError` 다."""
        ...

    def legal_action_mask(self, position: PositionT) -> npt.NDArray[np.bool_]:
        """`(action_space_size,)` 불리언 마스크. 합법수의 행동 인덱스만 True 다."""
        ...

    def canonical(self, position: PositionT) -> PositionT:
        """둘 차례인 쪽이 선수인 것처럼 돌려놓은 국면. 신경망은 이것만 본다 (ADR-0003).

        멱등이다 — 이미 정규형인 국면에 적용하면 그대로다. 돌려받은 국면의 좌표계가
        곧 정규형 좌표계이므로, `legal_moves`·`action_index` 를 그 국면에 그대로 쓰면
        정규형 프레임의 행동 인덱스가 나온다.
        """
        ...

    def to_canonical_action_index(self, position: PositionT, index: int) -> int:
        """`position` 프레임의 행동 인덱스를 정규형 프레임으로 옮긴다."""
        ...

    def from_canonical_action_index(self, position: PositionT, index: int) -> int:
        """정규형 프레임의 행동 인덱스를 `position` 프레임으로 되돌린다.

        신경망 출력을 실제 수로 되돌리는 쪽이다. `to_canonical_action_index` 의 역이다.
        """
        ...

    @property
    def tensor_shape(self) -> tuple[int, int, int]:
        """`encode` 가 돌려주는 `(C, H, W)`. 신경망 입력 크기다."""
        ...

    def encode(self, position: PositionT) -> npt.NDArray[np.float32]:
        """판 → `(C, H, W)` 텐서. 둘 차례인 쪽 시점이라 `encode(p) == encode(canonical(p))` 다.

        경계를 넘는 표현은 `numpy.ndarray` 다 (ADR-0001). torch 변환·배치·디바이스는
        `ai/` 가 맡는다.
        """
        ...

    def board_key(self, position: PositionT) -> Hashable:
        """판 + 차례만으로 만드는 키.

        반복 횟수는 반드시 이 키로 센다 — `Position` 동등성은 경로를 포함한다.
        """
        ...

    def render_move(self, move: MoveT) -> str:
        """수 표기법. 로그와 테스트가 같은 표기를 쓴다."""
        ...

    def parse_move(self, text: str) -> MoveT:
        """`render_move` 의 출력을 다시 `Move` 로 읽는다."""
        ...

    def render(self, position: PositionT) -> str:
        """사람이 읽는 판 표기. fixture 포맷이자 사람 검토 포맷이다."""
        ...

    def parse(self, text: str) -> PositionT:
        """`render` 의 출력을 다시 `Position` 으로 읽는다.

        표기에 없는 것은 돌아오지 않는다. 경로를 들고 다니는 게임에서는 읽은 국면의 경로가
        비어 있어 원래 국면과 다를 수 있다. 판·차례·`ply` 는 같다.
        """
        ...
