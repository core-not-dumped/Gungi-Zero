"""랜덤 대국 한 판을 끝까지 돌리고 판을 찍는다.

uv run python -m scripts.random_episode --game gungi-marshal-only --seed 4
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from typing import Any

from ai.episode import Episode
from ai.random_play import play_random_episode
from games.base import Game, MoveT, PositionT
from games.mini_gungi import MARSHAL_ONLY, Gungi
from games.tictactoe import TicTacToe


def result_line(episode: Episode[Any, Any]) -> str:
    """대국 끝에 찍는 한 줄. `결과: ▲ 승 (7 수)` 꼴이다."""
    winner = episode.outcome.winner
    result = "무승부" if winner is None else f"{winner.value} 승"
    return f"결과: {result} ({len(episode.moves)} 수)"


def print_random_episode(game: Game[PositionT, MoveT], seed: int | None) -> None:
    episode = play_random_episode(game, seed)

    position = game.initial_position()
    print(game.render(position))
    for number, move in enumerate(episode.moves, start=1):
        print(f"{number:2}. {position.player.value} {game.render_move(move)}")
        position = game.apply_move(position, move)
        print(game.render(position))
    print(result_line(episode))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="랜덤 대국 한 판")
    parser.add_argument("--game", choices=("tictactoe", "gungi-marshal-only"), default="tictactoe")
    parser.add_argument("--seed", type=int, default=None, help="고정하면 같은 대국이 재현된다")
    args = parser.parse_args(argv)

    if args.game == "tictactoe":
        print_random_episode(TicTacToe(), args.seed)
    else:
        print_random_episode(Gungi(MARSHAL_ONLY), args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
