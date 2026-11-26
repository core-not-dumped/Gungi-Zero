"""게임을 끝까지 풀어 상태마다 이론값과 DTM 을 낸다. 게임은 `games.base` 로만 안다.

상태는 `board_key`(판 + 차례)다. 초기 국면에서 너비 우선으로 도달 가능 그래프를 만들고,
판만으로 끝난 상태(`board_outcome`)에서 거꾸로 거슬러 올라가며(후퇴 분석) 승패를 정한다.
끝까지 승패가 정해지지 않는 상태는 무승부다. DTM 은 이기는 쪽이 가장 빠르게, 지는 쪽이 가장
늦게 둘 때 승패가 날 때까지의 수의 개수다.

## 반복 규칙

경로를 보지 않고 풀어도 반복 3회 규칙 아래의 값과 같다. 이기는 쪽은 DTM 이 줄어드는 수만 두고,
지는 쪽이 어떻게 두어도 DTM 은 줄어든다. 그래서 이기는 수순에는 같은 판이 두 번 나오지 않는다.
반복은 비기는 쪽에만 이롭고, 아무도 반복을 피하도록 강제되지 않는다. 비기는 상태에는 비기는
상태로 가는 수가 있고 상대의 패로 가는 수가 없으므로, 그 수만 두면 상대는 승리에 닿지 못한다.
반복과 수 제한은 그런 대국을 무승부로 끝낼 뿐이다.

전제가 하나 있다. 이기는 수순이 앞으로 지날 판이 지나온 경로에 이미 두 번 나왔으면 그 판에서
반복으로 끝난다. 이기는 국면에 처음 들어선 뒤로 줄곧 DTM 을 줄여 온 쪽에게는 이런 일이 없다 —
그 전의 판들은 이기는 국면이 아니었고, 뒤의 판들은 서로 다르다. 이기는 수를 놓치며 같은 판을
돌아온 경로에서는 `Solution.value` 의 승패가 실제와 다를 수 있다. 이미 끝난 국면은 게임의
판정을 그대로 낸다.

## 수 제한

그래프의 값은 수 제한이 없을 때의 값이다. 승패는 `DTM ≤ 수 제한 - ply` 일 때만 이루어지고,
아니면 지는 쪽이 한도까지 버티므로 무승부다.
"""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from enum import Enum, auto
from typing import Generic

from games.base import BoardOutcomeGame, MoveT, Outcome, Player, PositionT


class Verdict(Enum):
    """둘 차례인 쪽 시점의 승·무·패."""

    WIN = auto()
    DRAW = auto()
    LOSS = auto()


@dataclass(frozen=True, slots=True)
class Value:
    """한 국면의 이론값 (`CONTEXT.md`) 과, 승패라면 그때까지의 수 (DTM). 무승부는 `None` 이다."""

    verdict: Verdict
    dtm: int | None


_DRAW = Value(Verdict.DRAW, None)


@dataclass(frozen=True)
class _Graph:
    """도달 가능 그래프. 상태는 처음 닿은 순서의 번호로 부른다."""

    keys: list[Hashable]
    parents: list[list[int]]
    # 서로 다른 자식의 수.
    child_counts: list[int]
    # 판만으로 끝난 상태의 값.
    terminals: dict[int, Value]


def _terminal_value(outcome: Outcome, player: Player) -> Value:
    if outcome.winner is None:
        return _DRAW
    return Value(Verdict.WIN if outcome.winner is player else Verdict.LOSS, 0)


class Solution(Generic[PositionT, MoveT]):
    """풀어 둔 답. `solve` 가 만든다."""

    def __init__(
        self,
        game: BoardOutcomeGame[PositionT, MoveT],
        values: dict[Hashable, Value],
        move_limit: int | None,
    ) -> None:
        self.game = game
        self._values = values
        self._move_limit = move_limit

    def __len__(self) -> int:
        """도달 가능한 상태의 수."""
        return len(self._values)

    def value(self, position: PositionT) -> Value:
        """`position` 의 이론값. 그 `ply` 에서의 수 제한을 반영한다.

        끝난 국면은 게임의 판정을 따른다. 풀지 않은 상태면 `ValueError` 다.
        """
        outcome = self.game.outcome(position)
        if outcome is not None:
            return _terminal_value(outcome, position.player)
        value = self._values.get(self.game.board_key(position))
        if value is None:
            raise ValueError("초기 국면에서 도달할 수 없는 상태다")
        if (
            value.dtm is not None
            and self._move_limit is not None
            and value.dtm > self._move_limit - position.ply
        ):
            return _DRAW
        return value


def solve(
    game: BoardOutcomeGame[PositionT, MoveT], *, move_limit: int | None
) -> Solution[PositionT, MoveT]:
    """초기 국면에서 도달 가능한 모든 상태를 푼다. 상태를 모두 메모리에 둔다.

    `move_limit` 은 게임의 수 제한이고 없으면 `None` 이다. `Game` 은 한도를 알려 주지 않으므로
    게임을 고르는 쪽이 같은 값을 넘긴다. 한도가 너무 작아 상태를 다 펼치지 못하면 `ValueError` 다.
    """
    graph = _explore(game)
    values: list[Value | None] = [None] * len(graph.keys)
    # 승패가 난 상태를 DTM 이 작은 것부터 꺼낸다. 처음 것은 모두 0 이고 하나씩 늘며 덧붙는다.
    decided: list[int] = []
    for index, value in graph.terminals.items():
        values[index] = value
        if value.dtm is not None:
            decided.append(index)

    # 상대의 승으로 확인되지 않은 자식의 수.
    unresolved_children = list(graph.child_counts)
    for index in decided:  # 돌면서 뒤에 덧붙인다.
        child = values[index]
        assert child is not None and child.dtm is not None
        for parent in graph.parents[index]:
            if values[parent] is not None:
                continue
            if child.verdict is Verdict.LOSS:
                # 상대를 지는 국면에 보낼 수 있다. 처음 닿은 것이 가장 빠른 승이다.
                values[parent] = Value(Verdict.WIN, child.dtm + 1)
            else:
                unresolved_children[parent] -= 1
                if unresolved_children[parent]:
                    continue
                # 모든 수가 상대의 승이다. 마지막에 닿은 것이 가장 늦은 패다.
                values[parent] = Value(Verdict.LOSS, child.dtm + 1)
            decided.append(parent)

    solved = {key: value or _DRAW for key, value in zip(graph.keys, values, strict=True)}
    return Solution(game, solved, move_limit)


def _explore(game: BoardOutcomeGame[PositionT, MoveT]) -> _Graph:
    """너비 우선으로 도달 가능 그래프를 만든다.

    상태마다 가장 짧은 경로로 처음 닿은 국면 하나를 펼친다. 그 경로에는 같은 판이 두 번 없으므로
    반복 규칙에 막히지 않는다.
    """
    start = game.initial_position()
    keys = [game.board_key(start)]
    index_of = {keys[0]: 0}
    parents: list[list[int]] = [[]]
    child_counts: list[int] = []
    terminals: dict[int, Value] = {}
    # 아직 펼치지 않은 국면. 펼치고 나면 놓아 준다.
    frontier = [start]
    while frontier:
        following: list[PositionT] = []
        for position in frontier:
            index = len(child_counts)
            outcome = game.board_outcome(position)
            if outcome is not None:
                terminals[index] = _terminal_value(outcome, position.player)
                child_counts.append(0)
                continue
            moves = game.legal_moves(position)
            if not moves:
                raise ValueError(
                    f"수 제한에 막혀 상태를 다 펼치지 못한다: ply {position.ply} 에서 끝났다"
                )
            children: set[int] = set()
            for move in moves:
                child = game.apply_move(position, move)
                key = game.board_key(child)
                found = index_of.get(key)
                if found is None:
                    found = index_of[key] = len(keys)
                    keys.append(key)
                    parents.append([])
                    following.append(child)
                children.add(found)
            for child_index in children:
                parents[child_index].append(index)
            child_counts.append(len(children))
        frontier = following
    return _Graph(keys, parents, child_counts, terminals)
