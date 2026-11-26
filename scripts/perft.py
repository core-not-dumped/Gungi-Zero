"""perft: 한 국면에서 정확히 `depth` 수를 두는 수순의 개수.

uv run python -m scripts.perft --depth 4
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from games.base import Game, MoveT, PositionT
from games.mini_gungi import MINI_GUNGI, Gungi


def perft(game: Game[PositionT, MoveT], position: PositionT, depth: int) -> int:
    """같은 국면에 닿는 다른 수순은 따로 센다. 그 전에 끝난 대국은 세지 않는다."""
    if depth < 0:
        raise ValueError(f"깊이는 0 이상이다: {depth}")
    if depth == 0:
        return 1
    return sum(
        perft(game, game.apply_move(position, move), depth - 1)
        for move in game.legal_moves(position)
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="미니 군의 초기 국면의 perft")
    parser.add_argument("--depth", type=int, default=4)
    args = parser.parse_args(argv)

    game = Gungi(MINI_GUNGI)
    for depth in range(1, args.depth + 1):
        print(f"{depth}: {perft(game, game.initial_position(), depth)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
