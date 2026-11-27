"""solver 와 완벽한 상대의 테스트가 함께 쓰는 것."""

from collections.abc import Hashable
from dataclasses import replace

from eval.solver import Verdict
from games.base import Game, MoveT, Player, PositionT
from games.mini_gungi import MINI_GUNGI, Gungi, GungiPosition, PieceType, Placement
from scripts.state_space import CANDIDATES

# 작은 나노 후보. 상태 수는 docs/nano-state-space.md 의 표에 있다.
SMALL_NANO = next(c.config for c in CANDIDATES if c.name == "3x3 帥兵")

# 3x1 판에 帥 둘. 가운데를 사이에 두고 마주 본다. 선수의 유일한 수는 b1 로 가는 것이고,
# 후수는 그 帥 를 잡는 수 하나뿐이다. 그래서 국면은 셋이다.
FACING_MARSHALS = replace(
    MINI_GUNGI, width=3, height=1, setup=(Placement(PieceType.MARSHAL, column=0, row=0),)
)


def shortest_path_positions(game: Game[PositionT, MoveT]) -> list[PositionT]:
    """도달 가능한 `board_key` 마다 가장 짧은 경로로 닿은 국면 하나."""
    positions = [game.initial_position()]
    seen = {game.board_key(positions[0])}
    for position in positions:  # 돌면서 뒤에 덧붙인다.
        for move in game.legal_moves(position):
            child = game.apply_move(position, move)
            if game.board_key(child) not in seen:
                seen.add(game.board_key(child))
                positions.append(child)
    return positions


def with_ply(game: Gungi, position: GungiPosition, ply: int) -> GungiPosition:
    """같은 판·차례에 `ply` 만 바꾼 국면. 경로는 비어 있다."""
    text = game.render(position).replace(f"ply: {position.ply}", f"ply: {ply}")
    return game.parse(text)


def forces_a_win(
    game: Game[PositionT, MoveT], position: PositionT, winner: Player, plies: int
) -> bool:
    """`winner` 가 `plies` 수 안에 이기도록 강제할 수 있는지. 반복·수 제한은 게임이 판정한다."""
    outcome = game.outcome(position)
    if outcome is not None:
        return outcome.winner is winner
    if plies == 0:
        return False
    replies = (
        forces_a_win(game, game.apply_move(position, move), winner, plies - 1)
        for move in game.legal_moves(position)
    )
    return any(replies) if position.player is winner else all(replies)


class ForcedWinSearch:
    """`forces_a_win` 과 같은 것을 묻되 판마다의 답을 기억해 긴 수순까지 닿는다.

    solver 를 쓰지 않는다. 기억한 답을 다른 경로에서 다시 쓰려고 지나온 경로를 지우고 판마다
    새로 본다. 그래도 실제 규칙의 답과 같다.

    - 강제할 수 없다는 답: 반복은 무승부라 이기려는 쪽을 돕지 않는다. 경로를 지워도 못 이기면
      경로가 있어도 못 이긴다.
    - 강제할 수 있다는 답: 이기는 쪽이 남은 필요 수가 가장 적은 수만 두면 그 값이 줄기만 해서
      같은 판이 다시 나오지 않는다. 시작까지의 경로에 판마다 많아야 한 번 있으면 세 번째가 없다.

    수 제한은 보지 않으므로 `ply + plies` 가 한도 안인 국면에만 쓴다.
    """

    def __init__(self, game: Gungi) -> None:
        self._game = game
        # (판, 이기려는 쪽) → 강제할 수 있다고 확인한 가장 적은 수, 없다고 확인한 가장 많은 수.
        self._forced: dict[tuple[Hashable, Player], int] = {}
        self._not_forced: dict[tuple[Hashable, Player], int] = {}

    def forces(self, position: GungiPosition, winner: Player, plies: int) -> bool:
        game = self._game
        outcome = game.board_outcome(position)
        if outcome is not None:
            return outcome.winner is winner
        if plies == 0:
            return False
        key = (game.board_key(position), winner)
        if key in self._forced and self._forced[key] <= plies:
            return True
        if key in self._not_forced and self._not_forced[key] >= plies:
            return False
        # 판 표기에는 경로가 없어, 다시 읽은 국면은 반복으로 끝나 있지 않다.
        fresh = game.parse(game.render(position))
        replies = (
            self.forces(game.apply_move(fresh, move), winner, plies - 1)
            for move in game.legal_moves(fresh)
        )
        result = any(replies) if position.player is winner else all(replies)
        (self._forced if result else self._not_forced)[key] = plies
        return result


def winner_of(player: Player, verdict: Verdict) -> Player:
    """`player` 시점의 승패에서 이기는 쪽."""
    return player if verdict is Verdict.WIN else player.opponent
