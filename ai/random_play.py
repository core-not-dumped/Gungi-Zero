"""랜덤 AI 와 랜덤 대국. 어떤 게임 모듈도 import 하지 않는다 — `games.base` 만 안다 (ADR-0001)."""

from __future__ import annotations

import random
from typing import Generic

from ai.episode import Episode, play_episode
from games.base import Game, MoveT, PositionT


class RandomAI(Generic[PositionT, MoveT]):
    """합법수 중 균등 무작위로 고른다. Elo 0 의 기준점이다."""

    def __init__(self, game: Game[PositionT, MoveT], seed: int | None = None) -> None:
        self._game = game
        self._rng = random.Random(seed)

    def choose(self, position: PositionT) -> MoveT:
        moves = self._game.legal_moves(position)
        if not moves:
            raise ValueError("합법수가 없는 국면에서는 수를 고를 수 없다")
        return self._rng.choice(moves)


def play_random_episode(
    game: Game[PositionT, MoveT], seed: int | None = None
) -> Episode[PositionT, MoveT]:
    """랜덤 AI 둘이 한 판을 끝까지 둔다. 같은 시드는 같은 대국을 만든다."""
    ai = RandomAI(game, seed)
    return play_episode(game, ai, ai)
