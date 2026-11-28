"""Elo 난이도 단계. `scripts.make_levels` 가 만들고 대국 화면(`scripts.play`)이 읽는다.

단계 하나는 저장한 세대와 무작위 수를 섞는 확률이다. 점수는 `scripts.rate` 와 같은 대진
위의 상대값이라(랜덤 AI 0점), 잴 때와 같은 탐색 횟수로 두어야 그 점수의 상대다.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ai.random_play import MixedAI
from scripts.shared.players import checkpoint_player
from scripts.shared.setups import GameSetup

# git 에 넣지 않는다.
LEVELS_DIR = Path("levels")


def levels_path(game: str, directory: Path = LEVELS_DIR) -> Path:
    """게임마다의 난이도 단계 파일."""
    return directory / f"{game}.json"


@dataclass(frozen=True, slots=True)
class Level:
    """`rating` 은 잰 Elo, `checkpoint` 는 가중치 파일 경로다."""

    rating: float
    checkpoint: str
    random_probability: float


@dataclass(frozen=True, slots=True)
class LevelSet:
    """한 게임의 난이도 단계들. 점수가 낮은 것부터다."""

    game: str
    # 점수를 잴 때의 MCTS 탐색 횟수.
    simulations: int
    levels: tuple[Level, ...]


def pick_levels(candidates: Sequence[Level], *, step: float) -> list[Level]:
    """`step`, `2 step`, … 마다 가장 가까운 후보를 고른다. 반 칸 안에 후보가 없는 점수는 비운다.
    0점은 랜덤 AI 라 고르지 않는다."""
    if not candidates:
        return []
    highest = max(candidate.rating for candidate in candidates)
    picked: list[Level] = []
    target = step
    while target <= highest + step / 2:
        closest = min(candidates, key=lambda candidate: abs(candidate.rating - target))
        if abs(closest.rating - target) <= step / 2 and closest not in picked:
            picked.append(closest)
        target += step
    return picked


def save_levels(levels: LevelSet, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(levels), ensure_ascii=False, indent=2), encoding="utf-8")


def load_levels(path: Path) -> LevelSet:
    data = json.loads(path.read_text(encoding="utf-8"))
    return LevelSet(
        data["game"], data["simulations"], tuple(Level(**level) for level in data["levels"])
    )


def opponent_factory(
    setup: GameSetup, level: Level, *, simulations: int, device: str = "cpu"
) -> Callable[[Any, int], Callable[[Any], Any]]:
    """대국 화면이 쓰는 `(게임, 시드) → 상대`. 가중치는 여기서 바로 읽는다."""
    player = checkpoint_player(
        setup, Path(level.checkpoint), simulations=simulations, device=device
    )

    def make(game: Any, seed: int) -> Callable[[Any], Any]:
        return MixedAI(game, player, level.random_probability, seed).choose

    return make
