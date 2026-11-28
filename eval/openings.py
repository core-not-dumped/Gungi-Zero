"""풀이가 없는 게임에서 실력을 재는 대국의 시작 판.

풀 수 있는 게임은 풀이에서 뽑은 시험 판을 쓴다. 여기는 `Game` 만으로, 몇 수 안에 승패가 정해지는
판을 걸러 낸다 — 그런 판은 두는 쪽의 실력을 가르지 못한다.
"""

from __future__ import annotations

import random
from collections.abc import Hashable

from games.base import Game, MoveT, PositionT

# 시작 판 하나를 찾으려고 무작위로 걸어 보는 횟수의 상한 (평균).
_TRIES_PER_OPENING = 200


def forced_result(game: Game[PositionT, MoveT], position: PositionT, depth: int) -> int:
    """`depth` 수 안에 강제되는 승패. 둘 차례인 쪽이 이기면 1, 지면 -1, 정해지지 않거나 비기면 0.

    가지를 모두 보는 얕은 탐색이라 `depth` 는 작게 쓴다.
    """
    outcome = game.outcome(position)
    if outcome is not None:
        if outcome.winner is None:
            return 0
        return 1 if outcome.winner is position.player else -1
    if depth == 0:
        return 0
    best = -1
    for move in game.legal_moves(position):
        best = max(best, -forced_result(game, game.apply_move(position, move), depth - 1))
        if best == 1:
            break
    return best


def balanced_openings(
    game: Game[PositionT, MoveT],
    count: int,
    *,
    plies: tuple[int, int],
    depth: int,
    seed: int,
) -> tuple[PositionT, ...]:
    """초기 국면에서 `plies` 범위의 수만큼 무작위로 두어 닿는, 서로 다른(`board_key`) 시작 판들.

    끝난 국면과 `depth` 수 안에 승패가 강제되는 국면은 버린다. `depth` 보다 깊은 곳에서 정해지는
    판은 남는다. `count` 개를 찾지 못하면 `ValueError` 다.
    """
    rng = random.Random(seed)
    found: dict[Hashable, PositionT] = {}
    seen: set[Hashable] = set()
    for _ in range(count * _TRIES_PER_OPENING):
        if len(found) == count:
            break
        position = game.initial_position()
        for _ in range(rng.randint(*plies)):
            if game.outcome(position) is not None:
                break
            position = game.apply_move(position, rng.choice(game.legal_moves(position)))
        key = game.board_key(position)
        if key in seen or game.outcome(position) is not None:
            continue
        seen.add(key)
        if forced_result(game, position, depth) == 0:
            found[key] = position
    if len(found) < count:
        raise ValueError(f"시작 판 {count}개를 찾지 못했다 ({len(found)}개)")
    return tuple(found.values())
