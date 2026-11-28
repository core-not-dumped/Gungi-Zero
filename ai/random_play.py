"""랜덤 AI 와 랜덤 대국. 어떤 게임 모듈도 import 하지 않는다 — `games.base` 만 안다 (ADR-0001)."""

from __future__ import annotations

import random
from typing import Generic

from ai.base import AI
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


class MixedAI(Generic[PositionT, MoveT]):
    """`random_probability` 의 확률로 랜덤 AI 의 수를, 아니면 `ai` 의 수를 둔다. 강한 AI 를 원하는
    만큼 약하게 만든다 — 0 이면 `ai` 그대로, 1 이면 랜덤 AI 다."""

    def __init__(
        self,
        game: Game[PositionT, MoveT],
        ai: AI[PositionT, MoveT],
        random_probability: float,
        seed: int | None = None,
    ) -> None:
        if not 0 <= random_probability <= 1:
            raise ValueError(f"확률은 0 과 1 사이다: {random_probability}")
        self._ai = ai
        self._probability = random_probability
        self._rng = random.Random(seed)
        self._random = RandomAI(game, self._rng.getrandbits(63))

    def choose(self, position: PositionT) -> MoveT:
        if self._rng.random() < self._probability:
            return self._random.choose(position)
        return self._ai.choose(position)


def play_random_episode(
    game: Game[PositionT, MoveT], seed: int | None = None
) -> Episode[PositionT, MoveT]:
    """랜덤 AI 둘이 한 판을 끝까지 둔다. 같은 시드는 같은 대국을 만든다."""
    ai = RandomAI(game, seed)
    return play_episode(game, ai, ai)
