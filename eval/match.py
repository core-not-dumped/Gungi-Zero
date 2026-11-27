"""두 AI 의 대국. 선후를 바꿔 가며 두고 결과 분포를 낸다."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Generic

from ai.base import AI
from ai.episode import Episode, play_episode
from games.base import Game, MoveT, Player, PositionT


@dataclass(frozen=True, slots=True)
class Record:
    """한 자리에 앉은 `challenger` 시점의 판 수."""

    wins: int
    draws: int
    losses: int


def _record(episodes: Sequence[Episode[PositionT, MoveT]], seat: Player) -> Record:
    winners = [episode.outcome.winner for episode in episodes]
    return Record(winners.count(seat), winners.count(None), winners.count(seat.opponent))


@dataclass(frozen=True)
class MatchResult(Generic[PositionT, MoveT]):
    """`play_match` 의 결과. 대국은 시드 순서이고, 성적은 `challenger` 시점이다."""

    # `challenger` 가 선수로 둔 대국들과 후수로 둔 대국들.
    first_episodes: tuple[Episode[PositionT, MoveT], ...]
    second_episodes: tuple[Episode[PositionT, MoveT], ...]

    @property
    def as_first(self) -> Record:
        return _record(self.first_episodes, Player.FIRST)

    @property
    def as_second(self) -> Record:
        return _record(self.second_episodes, Player.SECOND)

    # 수열이 서로 다른 대국의 수. 판 수보다 훨씬 적으면 같은 판을 되풀이해 센 것이다.
    @property
    def distinct_as_first(self) -> int:
        return len({episode.moves for episode in self.first_episodes})

    @property
    def distinct_as_second(self) -> int:
        return len({episode.moves for episode in self.second_episodes})


def play_match(
    game: Game[PositionT, MoveT],
    challenger: Callable[[int], AI[PositionT, MoveT]],
    opponent: Callable[[int], AI[PositionT, MoveT]],
    *,
    seeds: Sequence[int],
) -> MatchResult[PositionT, MoveT]:
    """시드마다 `challenger` 가 선수로 한 판, 후수로 한 판을 둔다.

    `challenger`(성적을 재는 AI)와 `opponent` 는 시드를 받아 AI 를 만든다. 대국마다 새로 만들어서
    한 판은 시드만으로 다시 둘 수 있다. 시드를 쓰지 않는 결정적인 AI 둘은 같은 판을 되풀이한다.
    """
    if not seeds:
        raise ValueError("시드가 하나는 있어야 한다")
    return MatchResult(
        tuple(play_episode(game, challenger(seed), opponent(seed)) for seed in seeds),
        tuple(play_episode(game, opponent(seed), challenger(seed)) for seed in seeds),
    )
