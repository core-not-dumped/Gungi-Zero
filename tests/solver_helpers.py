"""solver 와 완벽한 상대의 테스트가 함께 쓰는 것."""

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


def winner_of(player: Player, verdict: Verdict) -> Player:
    """`player` 시점의 승패에서 이기는 쪽."""
    return player if verdict is Verdict.WIN else player.opponent
