"""자가 대국의 시작 판을 고르는 규칙 (ADR-0007). 학습의 합성 루트가 만들어 주입한다.

**풀이를 쓰지 않는다.** 이론값이나 DTM 으로 고르면 정답이 학습에 새어 들어가, 정답이 없는
게임의 학습을 대신 시험하지 못한다.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from functools import lru_cache
from typing import cast

from eval.reachable import reachable_positions
from games.base import Game, MoveT, PositionT


# 가장 최근의 게임 하나만 둔다. 상태 목록이 커서 여러 게임의 것을 함께 두지 않는다.
@lru_cache(maxsize=1)
def _unfinished_positions(game: Game[PositionT, MoveT]) -> list[PositionT]:
    return [position for position in reachable_positions(game) if game.outcome(position) is None]


def mixed_starts(
    game: Game[PositionT, MoveT], *, initial_fraction: float
) -> Callable[[random.Random], PositionT]:
    """`initial_fraction` 의 확률로 초기 국면을, 아니면 도달 가능하고 끝나지 않은 상태에서 균등하게
    뽑은 국면을 고르는 함수. 국면은 상태마다 가장 짧은 경로로 닿은 하나라 그 경로와 `ply` 가
    남는다.

    만들 때 도달 가능한 상태를 모두 돌아 메모리에 둔다 (나노 군의는 몇 분). 같은 프로세스에서
    같은 `game` 으로 다시 만들면 앞에서 돈 것을 쓴다.
    """
    initial = game.initial_position()
    unfinished = cast("list[PositionT]", _unfinished_positions(game))

    def choose(rng: random.Random) -> PositionT:
        if rng.random() < initial_fraction:
            return initial
        return rng.choice(unfinished)

    return choose
