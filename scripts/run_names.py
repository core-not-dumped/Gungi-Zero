"""wandb run 의 이름. 목록에서 무엇을 한 run 인지 이름만으로 알아본다."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path


def _folder(directory: Path, game: str) -> str:
    """체크포인트 폴더 이름에서 게임 이름을 뗀 것 (`nano-gungi-seed0-night1` → `seed0-night1`)."""
    return directory.name.removeprefix(f"{game}-") or directory.name


def training_name(game: str, *, seed: int, episodes: int, steps: int, started: datetime) -> str:
    return f"train · {game} · {episodes}판/{steps}스텝 · s{seed} · {started:%m-%d %H:%M}"


def evaluation_name(game: str, checkpoint: Path) -> str:
    number = checkpoint.stem.rpartition("-")[2]
    generation = f"{int(number)}세대" if number.isdigit() else checkpoint.stem
    return f"eval · {game} · {_folder(checkpoint.parent, game)} · {generation}"


def rating_name(game: str, directory: Path) -> str:
    return f"elo · {game} · {_folder(directory, game)}"


def tuning_name(experiment: str, seed: int) -> str:
    return f"{experiment} · s{seed}"
