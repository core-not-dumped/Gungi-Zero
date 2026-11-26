"""pygame 대국 화면을 연다.

uv run python -m scripts.play
"""

from __future__ import annotations

from ai.random_play import RandomAI
from games.mini_gungi import MINI_GUNGI, Gungi
from games.nano_gungi import NANO_GUNGI
from ui.app import App, run
from ui.session import Opponent

# 첫 번째가 기본이다.
VARIANTS = (("나노 3x4", Gungi(NANO_GUNGI)), ("미니 5x5", Gungi(MINI_GUNGI)))


OPPONENT_NAME = "랜덤 AI"


def random_opponent(game: Gungi, seed: int) -> Opponent:
    return RandomAI(game, seed).choose


def main() -> int:
    run(App(VARIANTS, OPPONENT_NAME, random_opponent))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
