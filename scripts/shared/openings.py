"""실력을 재는 대국의 시작 판을 게임에 맞춰 고른다.

풀이가 있는 게임(지금은 나노 군의)은 시험 판(ADR-0007), 그 밖의 게임은 몇 수 안에 승패가 정해지지
않는 판(`eval.openings`)이다.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from eval.openings import balanced_openings
from games.mini_gungi import Gungi
from scripts.shared.hyperparameters import OPENING_DEPTH, OPENING_PLIES, OPENING_SEED
from scripts.shared.setups import GameSetup
from scripts.shared.solution import SOLUTION_PATH, load_solution_of
from scripts.shared.trials import fingerprint, nano_trials

ItemT = TypeVar("ItemT")

# 풀이에서 시험 판을 뽑는 게임.
SOLVED_GAMES = ("nano-gungi",)


@dataclass(frozen=True)
class Openings:
    positions: list[Any]
    # `trial_positions` 나 `balanced`.
    source: str
    # 같은 시작 판인지 가르는 값.
    fingerprint: str


def spread(items: Sequence[ItemT], count: int) -> list[ItemT]:
    """처음부터 끝까지 고른 간격으로 `count` 개. 시험 판은 층끼리 이어져 있어 층의 비율이 남는다."""
    if not 1 <= count <= len(items):
        raise ValueError(f"1개에서 {len(items)}개까지 고를 수 있다: {count}")
    return [items[index * len(items) // count] for index in range(count)]


def openings_for(setup: GameSetup, count: int, solution: Path = SOLUTION_PATH) -> Openings:
    """`setup` 의 게임에서 실력을 재는 시작 판 `count` 개. 고를 수 없으면 `ValueError`, 풀이가
    필요한데 읽지 못하면 `OSError` 나 `ValueError` 다."""
    game = setup.game
    if setup.name in SOLVED_GAMES:
        trials = spread(nano_trials(load_solution_of(setup, solution)), count)
        assert isinstance(game, Gungi)
        return Openings(
            [trial.position for trial in trials], "trial_positions", fingerprint(game, trials)
        )
    positions = balanced_openings(
        game, count, plies=OPENING_PLIES, depth=OPENING_DEPTH, seed=OPENING_SEED
    )
    text = "".join(game.render(position) for position in positions)
    return Openings(list(positions), "balanced", hashlib.sha256(text.encode()).hexdigest())
