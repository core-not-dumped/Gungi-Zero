"""초기 국면에서 도달 가능한 상태 돌기와 세기."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from games.base import BoardOutcomeGame, Game, MoveT, PositionT


def reachable_positions(game: Game[PositionT, MoveT]) -> Iterator[PositionT]:
    """도달 가능한 `board_key` 마다 가장 짧은 경로로 처음 닿은 국면 하나를 너비 우선으로 낸다.

    끝난 국면도 낸다. 뻗어 나가는 것은 `legal_moves` 라 수 제한에 닿은 국면에서는 더 나아가지
    않는다. 본 상태의 키와 다음 깊이의 국면들을 메모리에 둔다.
    """
    start = game.initial_position()
    seen = {game.board_key(start)}
    frontier = [start]
    while frontier:
        following: list[PositionT] = []
        for position in frontier:
            yield position
            for move in game.legal_moves(position):
                child = game.apply_move(position, move)
                key = game.board_key(child)
                if key not in seen:
                    seen.add(key)
                    following.append(child)
        frontier = following


@dataclass(frozen=True)
class StateCount:
    """상태는 `board_key`(판 + 차례)다. `ply` 와 경로는 세지 않는다."""

    states: int
    # 그중 판만으로 끝난 상태 (`board_outcome`).
    terminal: int
    # 가장 먼 상태까지의 최단 수 개수.
    depth: int
    # 거짓이면 상한에서 멈춘 것이고, 위의 값들은 거기까지 본 것이다.
    complete: bool


def count_reachable(game: BoardOutcomeGame[PositionT, MoveT], limit: int) -> StateCount:
    """너비 우선으로 센다. `limit` 개를 넘는 상태가 있으면 `limit` 개에서 멈춘다.

    끝난 상태는 `board_outcome` 으로 판정한다. 뻗어 나가는 것은 `legal_moves` 라 수 제한은
    규칙 그대로 적용된다 — 한도에 닿은 국면에서는 더 나아가지 않는다. 한도에 닿은 것만으로는
    끝난 상태로 세지 않는다. 각 상태는 가장 짧은 경로로 처음 닿고 그 경로에는 같은
    `board_key` 가 두 번 나오지 않으므로, 반복 한도가 2 이상이면 반복 규칙은 뻗어 나가는
    것을 막지 않는다.
    """
    if limit < 1:
        raise ValueError(f"상한은 1 이상이다: {limit}")
    start = game.initial_position()
    seen = {game.board_key(start)}
    frontier = [start]
    terminal = 0
    depth = 0
    while True:
        following: list[PositionT] = []
        for position in frontier:
            if game.board_outcome(position) is not None:
                terminal += 1
                continue
            for move in game.legal_moves(position):
                child = game.apply_move(position, move)
                key = game.board_key(child)
                if key in seen:
                    continue
                if len(seen) == limit:
                    return StateCount(limit, terminal, depth + bool(following), complete=False)
                seen.add(key)
                following.append(child)
        if not following:
            return StateCount(len(seen), terminal, depth, complete=True)
        frontier = following
        depth += 1
