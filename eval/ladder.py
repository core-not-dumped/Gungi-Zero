"""Elo 를 매길 대진과 대국. 참가자는 번호 달린 단계(세대)들과 기준 0번이다."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Sequence
from typing import Generic, TypeVar

from ai.base import AI
from ai.episode import play_episode
from eval.elo import PairScore
from games.base import Game, MoveT, Player, PositionT

PlayerT = TypeVar("PlayerT", bound=Hashable)


def ladder_pairs(
    top: int, *, gaps: Sequence[int], anchored: Sequence[int]
) -> list[tuple[int, int]]:
    """1..`top` 단계의 대진 `(높은 쪽, 낮은 쪽)`. 기준(0)과 `anchored` 의 단계들, 그리고 번호
    차이가 `gaps` 인 단계끼리다. `top` 을 넘는 단계는 빠진다.

    모든 단계가 서로 두면 대국이 단계 수의 제곱으로 늘어난다. 이웃과의 대진이 곡선의 모양을,
    먼 대진과 기준 대진이 그 높이를 잡는다.
    """
    pairs = [(level, 0) for level in anchored if 1 <= level <= top]
    for gap in gaps:
        pairs.extend((level, level - gap) for level in range(gap + 1, top + 1))
    return pairs


def play_pair(
    game: Game[PositionT, MoveT],
    player: tuple[PlayerT, Callable[[int], AI[PositionT, MoveT]]],
    opponent: tuple[PlayerT, Callable[[int], AI[PositionT, MoveT]]],
    openings: Sequence[PositionT],
) -> PairScore[PlayerT]:
    """시작 판마다 두 판을 둔다. 한 번은 `player` 가, 한 번은 `opponent` 가 먼저 둔다.

    참가자는 `(이름, 대국의 순번을 받아 AI 를 만드는 함수)` 다. 시작 판의 유불리는 선후를 바꾼 두
    판에 똑같이 걸린다.
    """
    name, challenger = player
    other_name, other = opponent
    wins = draws = losses = 0
    for number, opening in enumerate(openings):
        for turn, moves_first in enumerate((True, False)):
            index = 2 * number + turn
            # 시작 판에서 둘 차례인 쪽에 앉는 것이 먼저 두는 것이다.
            seat = opening.player if moves_first else opening.player.opponent
            ai, rival = challenger(index), other(index)
            first, second = (ai, rival) if seat is Player.FIRST else (rival, ai)
            winner = play_episode(game, first, second, start=opening).outcome.winner
            wins += winner is seat
            draws += winner is None
            losses += winner is seat.opponent
    return PairScore(name, other_name, wins, draws, losses)


class MemoizedAI(Generic[PositionT, MoveT]):
    """국면마다 한 번만 묻고 그 수를 되풀이한다.

    temperature 0 의 MCTS 처럼 국면만으로 수가 정해지는 AI 에 쓴다. 여러 상대와 같은 시작 판을
    두므로 탐색을 크게 아낀다.
    """

    def __init__(self, ai: AI[PositionT, MoveT]) -> None:
        self._ai = ai
        self._moves: dict[PositionT, MoveT] = {}

    def choose(self, position: PositionT) -> MoveT:
        move = self._moves.get(position)
        if move is None:
            move = self._moves[position] = self._ai.choose(position)
        return move
