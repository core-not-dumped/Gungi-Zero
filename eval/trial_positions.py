"""시험 판 (`CONTEXT.md`) 뽑기. 풀어 둔 답에서 조건에 맞는 국면을 층마다 정한 수만큼 고른다.

게임은 `games.base` 로만 안다. 게임마다 다른 조건은 `Stratum.accepts` 로 주입한다.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass
from typing import Generic

from eval.reachable import reachable_positions
from eval.solver import Solution, Value
from games.base import MoveT, PositionT


@dataclass(frozen=True)
class Stratum(Generic[PositionT]):
    """시험 판의 한 층: 이름, 뽑을 개수, 받아들이는 국면."""

    name: str
    count: int
    # 진행 중인 국면과 그 이론값을 받는다.
    accepts: Callable[[PositionT, Value], bool]


@dataclass(frozen=True)
class Trial(Generic[PositionT]):
    """시험 판 하나. 둘 차례인 쪽이 시험받는다."""

    stratum: str
    position: PositionT
    # 둘 차례인 쪽 시점의 이론값.
    value: Value


def select_trials(
    solution: Solution[PositionT, MoveT],
    strata: Sequence[Stratum[PositionT]],
    *,
    seed: int,
) -> tuple[Trial[PositionT], ...]:
    """층 순서대로, 층마다 `count` 개를 시드로 고른다. 같은 풀이와 시드는 같은 시험 판을 낸다.

    후보는 도달 가능한 상태마다 `reachable_positions` 의 국면 하나이고, 끝난 국면은 뺀다.
    국면에는 거기까지의 경로가 있어 반복 판정이 이어진다. 정규형의 판이 같은 두 국면은 신경망에
    같은 판이라 함께 뽑지 않는다. 겹치는 층에서는 앞의 층이 먼저 가져가므로 후보가 적은 층을
    앞에 둔다. 후보가 모자란 층이 있으면 `ValueError` 다.
    """
    game = solution.game
    candidates: list[list[Trial[PositionT]]] = [[] for _ in strata]
    for position in reachable_positions(game):
        if game.outcome(position) is not None:
            continue
        value = solution.value(position)
        for stratum, found in zip(strata, candidates, strict=True):
            if stratum.accepts(position, value):
                found.append(Trial(stratum.name, position, value))

    rng = random.Random(seed)
    taken: set[Hashable] = set()
    selected: list[Trial[PositionT]] = []
    for stratum, found in zip(strata, candidates, strict=True):
        rng.shuffle(found)
        remaining = stratum.count
        for trial in found:
            if remaining == 0:
                break
            board = game.board_key(game.canonical(trial.position))
            if board not in taken:
                taken.add(board)
                selected.append(trial)
                remaining -= 1
        if remaining:
            short = f"{stratum.count}개 중 {stratum.count - remaining}개뿐이다"
            raise ValueError(f"층 {stratum.name!r} 의 후보가 모자란다: {short}")
    return tuple(selected)
