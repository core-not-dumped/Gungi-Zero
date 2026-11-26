"""처음부터 끝까지 진행된 한 판."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic

from ai.base import AI
from games.base import Game, MoveT, Outcome, Player, PositionT


@dataclass(frozen=True, slots=True)
class Episode(Generic[PositionT, MoveT]):
    """한 대국의 기록. 자가 대국이 만들어내는 학습 데이터의 단위다 (`CONTEXT.md`).

    지나간 국면을 들고 있지 않다 — 초기 국면에서 `moves` 를 다시 두면 복원된다.
    """

    moves: tuple[MoveT, ...]
    final_position: PositionT
    outcome: Outcome


def value_for(outcome: Outcome, player: Player) -> float:
    """`player` 시점의 결과. 승 +1, 무 0, 패 -1."""
    if outcome.winner is None:
        return 0.0
    return 1.0 if outcome.winner is player else -1.0


def play_episode(
    game: Game[PositionT, MoveT],
    first: AI[PositionT, MoveT],
    second: AI[PositionT, MoveT],
) -> Episode[PositionT, MoveT]:
    """초기 국면부터 종료까지, `first` 가 선수·`second` 가 후수로 둔다.

    같은 AI 를 양쪽에 넘겨도 된다. 합법수가 아닌 수를 고르면 `ValueError` 다.
    """
    position = game.initial_position()
    moves: list[MoveT] = []
    while (outcome := game.outcome(position)) is None:
        ai = first if position.player is Player.FIRST else second
        move = ai.choose(position)
        moves.append(move)
        position = game.apply_move(position, move)
    return Episode(tuple(moves), position, outcome)
