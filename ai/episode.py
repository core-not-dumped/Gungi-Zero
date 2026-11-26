"""처음부터 끝까지 진행된 한 판."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic

from games.base import MoveT, Outcome, PositionT


@dataclass(frozen=True, slots=True)
class Episode(Generic[PositionT, MoveT]):
    """한 대국의 기록. 자가 대국이 만들어내는 학습 데이터의 단위다 (`CONTEXT.md`).

    지나간 국면을 들고 있지 않다 — 초기 국면에서 `moves` 를 다시 두면 복원된다.
    """

    moves: tuple[MoveT, ...]
    final_position: PositionT
    outcome: Outcome
