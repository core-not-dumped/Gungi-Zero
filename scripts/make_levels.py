"""Elo 를 잰 기록에서 대국 화면의 난이도 단계를 만든다.

uv run python -m scripts.rate --checkpoint-dir <폴더> > <기록>
uv run python -m scripts.make_levels --elo-log <기록> --device cuda

게임은 기록의 것이고 단계는 `levels/<게임>.json` 에 저장한다.

후보는 세대들 그대로와, 점수가 가장 높은 세대에 무작위 수를 섞은 것들이다. 섞은 후보는 같은 시작
판에서 몇 상대와 더 두어 점수를 맞춘다. `--step` 점마다 가장 가까운 후보를 골라 저장한다.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ai.base import AI
from ai.random_play import MixedAI, RandomAI
from eval.elo import PairScore, fit_elo
from eval.ladder import play_pair
from scripts.hyperparameters import (
    LEVEL_OPPONENTS,
    LEVEL_RANDOM_PROBABILITIES,
    LEVEL_STEP,
)
from scripts.levels import (
    Level,
    LevelSet,
    checkpoint_player,
    levels_path,
    pick_levels,
    save_levels,
)
from scripts.openings import openings_for
from scripts.rate import RANDOM, generation_files
from scripts.train import SETUPS


def read_rating_log(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """`scripts.rate` 의 출력에서 설정 줄과 결과 줄을 꺼낸다. 없으면 `ValueError` 다."""
    settings = summary = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("{"):
            continue
        values = json.loads(line)
        if "elo" in values:
            settings = values
        if "pairs" in values:
            summary = values
    if settings is None or summary is None:
        raise ValueError(f"{path} 에 끝까지 마친 Elo 기록이 없다")
    return settings, summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Elo 난이도 단계를 만든다")
    parser.add_argument("--elo-log", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--step", type=float, default=LEVEL_STEP)
    parser.add_argument("--output", type=Path, help="기본은 levels/<게임>.json")
    args = parser.parse_args(argv)
    if args.step <= 0:
        parser.error("--step 은 0 보다 크다")
    try:
        settings, summary = read_rating_log(args.elo_log)
        if "openings_from" not in settings["elo"]:
            raise ValueError("시작 판을 남기지 않은 옛 Elo 기록이다. Elo 를 다시 잰다")
        files = generation_files(Path(settings["checkpoint_dir"]))
    except (OSError, ValueError) as error:
        parser.error(str(error))

    setup = SETUPS[settings["game"]]
    game = setup.game
    elo = settings["elo"]
    # 점수를 잰 것과 같은 시작 판이어야 점수를 한 축에 맞출 수 있다.
    openings = openings_for(setup, elo["openings"], Path(elo["solution"]))
    if openings.fingerprint != elo["openings_fingerprint"]:
        parser.error("Elo 를 잰 때와 시작 판이 다르다. Elo 를 다시 잰다")
    ratings: dict[int, float] = dict(summary["ratings"])
    # Elo 를 일부 세대만 쟀을 수 있다.
    files = {generation: files[generation] for generation in ratings}
    # 점수가 같으면 나중 세대.
    strongest = max(ratings, key=lambda generation: (ratings[generation], generation))

    players: dict[Any, Callable[[int], AI[Any, Any]]] = {
        RANDOM: lambda index: RandomAI(game, index)
    }
    saved = list(files)
    early = (saved[i - 1] for i in LEVEL_OPPONENTS if i <= len(saved))
    opponents = dict.fromkeys([RANDOM, *early, strongest])
    for generation in opponents:
        if generation != RANDOM:
            ai = checkpoint_player(
                setup, files[generation], simulations=elo["simulations"], device=args.device
            )
            players[generation] = lambda index, ai=ai: ai
    base = players[strongest](0)

    scores: list[PairScore[Any]] = [PairScore(**pair) for pair in summary["pairs"]]
    for number, probability in enumerate(LEVEL_RANDOM_PROBABILITIES, start=1):
        name = ("mixed", probability)

        def mixed(index: int, probability: float = probability) -> AI[Any, Any]:
            return MixedAI(game, base, probability, index)

        scores.extend(
            play_pair(game, (name, mixed), (other, players[other]), openings.positions)
            for other in opponents
        )
        total = len(LEVEL_RANDOM_PROBABILITIES)
        print(f"약화한 후보 {number}/{total}", file=sys.stderr, flush=True)
    fitted = fit_elo(scores, anchor=RANDOM, prior_draws=elo["prior_draws"])

    candidates = [
        Level(round(fitted[generation], 1), str(file), 0.0) for generation, file in files.items()
    ]
    candidates += [
        Level(round(fitted["mixed", p], 1), str(files[strongest]), p)
        for p in LEVEL_RANDOM_PROBABILITIES
    ]
    levels = LevelSet(
        setup.name, elo["simulations"], tuple(pick_levels(candidates, step=args.step))
    )
    output = args.output or levels_path(setup.name)
    save_levels(levels, output)
    print(json.dumps({"output": str(output), **asdict(levels)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
