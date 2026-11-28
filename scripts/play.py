"""pygame 대국 화면을 연다.

uv run python -m scripts.play

`levels/<게임>.json`(`scripts.make_levels` 가 만든다)이 있으면 그 판에서 Elo 난이도를 고를 수
있다. 없으면 상대는 랜덤 AI 하나다.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from ai.random_play import RandomAI
from games.mini_gungi import MINI_GUNGI, Gungi
from games.nano_gungi import NANO_GUNGI
from ui.app import App, OpponentFactory, run
from ui.session import Opponent

# 첫 번째가 기본이다.
VARIANTS = (("나노 3x4", Gungi(NANO_GUNGI)), ("미니 5x5", Gungi(MINI_GUNGI)))


# 난이도 단계를 읽는 게임. 나머지 판에는 학습한 상대가 없다.
LEVEL_GAMES = {VARIANTS[0][0]: "nano-gungi", VARIANTS[1][0]: "mini-gungi"}

OPPONENT_NAME = "랜덤 AI"


def random_opponent(game: Gungi, seed: int) -> Opponent:
    return RandomAI(game, seed).choose


def level_options(directory: Path) -> dict[str, list[tuple[str, OpponentFactory]]]:
    """저장한 난이도 단계를 화면의 판마다의 선택지로. 파일이 없는 판은 빠진다."""
    found = {
        variant: path
        for variant, game in LEVEL_GAMES.items()
        if (path := directory / f"{game}.json").is_file()
    }
    if not found:
        return {}
    # torch 는 난이도가 있을 때만 읽는다.
    from scripts.levels import load_levels, opponent_factory
    from scripts.train import SETUPS

    offered = {}
    for variant, path in found.items():
        levels = load_levels(path)
        setup = SETUPS[levels.game]
        options: list[tuple[str, OpponentFactory]] = [("Elo 0", random_opponent)]
        for level in levels.levels:
            factory = opponent_factory(setup, level, simulations=levels.simulations)
            options.append((f"Elo {level.rating:.0f}", factory))
        offered[variant] = options
    return offered


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="pygame 대국 화면")
    parser.add_argument(
        "--levels", type=Path, default=Path("levels"), help="난이도 단계 파일들의 폴더"
    )
    args = parser.parse_args(argv)
    run(App(VARIANTS, OPPONENT_NAME, random_opponent, level_options(args.levels)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
