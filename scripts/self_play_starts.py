"""자가 대국의 시작 판을 고르는 규칙 (ADR-0007). 학습의 합성 루트가 만들어 주입한다.

**풀이를 쓰지 않는다.** 이론값이나 DTM 으로 고르면 정답이 학습에 새어 들어가, 정답이 없는
게임의 학습을 대신 시험하지 못한다.
"""

from __future__ import annotations

import random
from collections.abc import Callable

from eval.reachable import reachable_positions
from games.base import Game, MoveT, PositionT


def mixed_starts(
    game: Game[PositionT, MoveT], *, initial_fraction: float
) -> Callable[[random.Random], PositionT]:
    """`initial_fraction` 의 확률로 초기 국면을, 아니면 도달 가능하고 끝나지 않은 상태에서 균등하게
    뽑은 국면을 고르는 함수. 국면은 상태마다 가장 짧은 경로로 닿은 하나라 그 경로와 `ply` 가
    남는다.

    만들 때 도달 가능한 상태를 모두 돌아 메모리에 둔다 (나노 군의는 몇 분).
    """
    initial = game.initial_position()
    unfinished = [
        position for position in reachable_positions(game) if game.outcome(position) is None
    ]

    def choose(rng: random.Random) -> PositionT:
        if rng.random() < initial_fraction:
            return initial
        return rng.choice(unfinished)

    return choose
