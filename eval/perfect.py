"""완벽한 상대 (`CONTEXT.md`). 풀어 둔 답으로 둔다."""

from __future__ import annotations

import random
from typing import Generic

from eval.solver import Solution, Value, Verdict
from games.base import MoveT, PositionT


def _preference(reply: Value) -> tuple[int, int]:
    """둔 다음 국면의 값(상대 시점)에 대한 선호. 클수록 좋다.

    상대가 지는 수는 빠를수록, 그다음은 무승부, 상대가 이기는 수는 늦을수록 낫다.
    """
    if reply.dtm is None:
        return (1, 0)
    if reply.verdict is Verdict.LOSS:
        return (2, -reply.dtm)
    return (0, reply.dtm)


def best_moves(solution: Solution[PositionT, MoveT], position: PositionT) -> list[MoveT]:
    """최선의 수들: 이기는 국면은 가장 빠른 승, 비기는 국면은 무승부 유지, 지는 국면은 가장
    늦은 패. 모두 수 제한을 반영한 값(`Solution.value`)으로 정한다. 끝난 국면에는 비어 있다.
    """
    game = solution.game
    moves = game.legal_moves(position)
    if not moves:
        return []
    preferences = [_preference(solution.value(game.apply_move(position, move))) for move in moves]
    best = max(preferences)
    return [move for move, ranked in zip(moves, preferences, strict=True) if ranked == best]


class PerfectAI(Generic[PositionT, MoveT]):
    """`best_moves` 중 하나를 둔다. 여럿이면 시드로 고른다.

    반복은 둔 다음 국면이 이미 끝난 경우만 본다. 상대가 이기는 수를 놓치며 같은 판을 돌아온
    경로에서 지는 쪽에 서면, 반복으로 비길 수 있는 길을 놓칠 수 있다 (`eval.solver` 의 "반복 규칙").
    """

    def __init__(self, solution: Solution[PositionT, MoveT], seed: int | None = None) -> None:
        self._solution = solution
        self._rng = random.Random(seed)

    def choose(self, position: PositionT) -> MoveT:
        best = best_moves(self._solution, position)
        if not best:
            raise ValueError("합법수가 없는 국면에서는 수를 고를 수 없다")
        return self._rng.choice(best)
