"""학습의 세대마다 Elo 를 매겨 wandb 에 적는 합성 루트. 랜덤 AI 가 0점이다.

uv run python -m scripts.rate --checkpoint-dir <폴더> --device cpu --workers 8
uv run python -m scripts.rate --game mini-gungi --checkpoint-dir <폴더> --device cpu --workers 8

시작 판은 게임에 맞춰 고른다(`scripts.openings`): 나노 군의는 시험 판이라 `scripts.solve_nano` 가
만든 풀이가 있어야 하고, 풀이가 없는 게임은 몇 수 안에 승패가 정해지지 않는 판이다. 완벽한 상대는
두지 않고, 참가자끼리 선후를 바꿔 둔다.

`--checkpoint-dir` 의 `generation-NNNN.pt` 들을 잰다. 대진의 간격은 저장한 순서로 센다.
학습과 따로 새 run 을 만든다. step 은 세대 번호다.
wandb 를 거치지 않으려면 `--wandb disabled` 를 쓴다.

`--generations` 로 일부 세대만 잰다. `--workers` 가 1 보다 크면 대진을 프로세스 여럿에 나눠
둔다. 프로세스마다 torch 스레드는 하나라 `--device cpu` 와 함께 쓴다.

Elo 는 이 대진 안에서의 상대적인 강함이다. 완료 기준(시험 판)을 대신하지 않고, 게임·종료 규칙·
아래 설정이 다른 run 의 점수와 비교하지 않는다 (ADR-0004).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

import torch

from ai.base import AI
from ai.random_play import RandomAI
from eval.elo import PairScore, fit_elo
from eval.ladder import ladder_pairs, play_pair
from scripts.hyperparameters import (
    ELO_ANCHORED_EVERY,
    ELO_GAPS,
    ELO_OPENINGS,
    ELO_PRIOR_DRAWS,
    EVALUATION_TEMPERATURE,
)
from scripts.levels import checkpoint_player
from scripts.loggers import PrintingLogger, TeeLogger, WandbLogger
from scripts.openings import openings_for
from scripts.run_names import rating_name
from scripts.solve_nano import SOLUTION_PATH
from scripts.train import SETUPS
from scripts.verify_nano import WANDB_PROJECT

# 랜덤 AI 의 번호. 세대는 1 부터다.
RANDOM = 0


def generation_files(directory: Path) -> dict[int, Path]:
    """세대 번호 → 체크포인트, 번호 순. 하나도 없으면 `ValueError` 다."""
    files = {}
    for file in directory.glob("generation-*.pt"):
        number = file.stem.rpartition("-")[2]
        if number.isdigit():
            files[int(number)] = file
    if not files:
        raise ValueError(f"{directory} 에 체크포인트가 없다")
    return dict(sorted(files.items()))


def anchored_generations(top: int) -> list[int]:
    """랜덤 AI 와 두는 것의 순번(1부터): 처음 것들과 일정한 간격, 마지막."""
    early = {1, 2, 5}
    regular = set(range(ELO_ANCHORED_EVERY, top + 1, ELO_ANCHORED_EVERY))
    return sorted(generation for generation in early | regular | {top} if generation <= top)


def start_wandb(mode: Literal["online", "offline", "disabled"]) -> Any:
    """wandb run 을 연다. torch 처럼 무거워 쓸 때 import 한다."""
    import wandb

    return wandb.init(project=WANDB_PROJECT, job_type="rate", mode=mode)


class Table:
    """대진을 두는 데 필요한 것. 참가자(AI)는 처음 쓸 때 만들고, 프로세스마다 하나씩 둔다."""

    def __init__(
        self,
        game: str,
        files: dict[int, Path],
        openings: Sequence[Any],
        simulations: int,
        device: str,
    ) -> None:
        self._setup = SETUPS[game]
        self._arguments = (files, openings, simulations, device)
        self._players: dict[int, AI[Any, Any]] = {}

    def play(self, pair: tuple[int, int]) -> PairScore[int]:
        high, low = pair
        openings = self._arguments[1]
        game = self._setup.game
        return play_pair(game, (high, self._player(high)), (low, self._player(low)), openings)

    def _player(self, generation: int) -> Callable[[int], AI[Any, Any]]:
        files, _, simulations, device = self._arguments
        setup = self._setup
        if generation == RANDOM:
            return lambda index: RandomAI(setup.game, index)
        if generation not in self._players:
            self._players[generation] = checkpoint_player(
                setup, files[generation], simulations=simulations, device=device
            )
        return lambda index: self._players[generation]


# 일하는 프로세스의 `Table`. 부모 프로세스에서는 쓰지 않는다.
_table: Table | None = None


def _open_table(*arguments: Any) -> None:
    global _table
    # 프로세스 여럿이 코어를 나눠 쓴다.
    torch.set_num_threads(1)
    _table = Table(*arguments)


def _play_at_table(pair: tuple[int, int]) -> PairScore[int]:
    assert _table is not None
    return _table.play(pair)


def play_pairs(
    pairs: Sequence[tuple[int, int]], arguments: tuple[Any, ...], *, workers: int
) -> Iterator[tuple[int, PairScore[int]]]:
    """대진마다 `(순번, 성적)` 을 끝나는 대로 낸다. `arguments` 는 `Table` 의 것이다."""
    if workers == 1:
        table = Table(*arguments)
        yield from ((number, table.play(pair)) for number, pair in enumerate(pairs))
        return
    with ProcessPoolExecutor(workers, initializer=_open_table, initargs=arguments) as pool:
        numbers = {pool.submit(_play_at_table, pair): number for number, pair in enumerate(pairs)}
        for done in as_completed(numbers):
            yield numbers[done], done.result()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="세대마다의 Elo 를 매긴다")
    parser.add_argument("--game", choices=tuple(SETUPS), default="nano-gungi")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--openings", type=int, default=ELO_OPENINGS, help="대진마다의 시작 판 수")
    parser.add_argument("--simulations", type=int, help="기본은 그 게임의 평가 탐색 횟수")
    parser.add_argument("--solution", type=Path, default=SOLUTION_PATH)
    parser.add_argument("--generations", type=int, nargs="+", help="이 세대들만 잰다. 기본은 전부")
    parser.add_argument("--workers", type=int, default=1, help="대진을 두는 프로세스 수")
    parser.add_argument("--wandb", choices=("online", "offline", "disabled"), default="online")
    args = parser.parse_args(argv)
    setup = SETUPS[args.game]
    simulations = setup.evaluation_simulations if args.simulations is None else args.simulations
    if min(args.openings, simulations, args.workers) < 1:
        parser.error("--openings, --simulations, --workers 는 1 이상이다")
    try:
        files = generation_files(args.checkpoint_dir)
    except ValueError as error:
        parser.error(str(error))
    if args.generations:
        missing = sorted(set(args.generations) - files.keys())
        if missing:
            parser.error(f"체크포인트가 없는 세대다: {missing}")
        files = {generation: files[generation] for generation in sorted(set(args.generations))}

    try:
        openings = openings_for(setup, args.openings, args.solution)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    # 대진은 저장한 순번(1부터, 0 은 랜덤 AI)으로 짜고 세대 번호로 바꾼다.
    generations = [RANDOM, *files]
    pairs = [
        (generations[high], generations[low])
        for high, low in ladder_pairs(
            len(files), gaps=ELO_GAPS, anchored=anchored_generations(len(files))
        )
    ]
    wandb_run = start_wandb(args.wandb)
    wandb_run.name = rating_name(setup.name, args.checkpoint_dir)
    logger = TeeLogger(PrintingLogger(), WandbLogger(wandb_run))
    logger.config(
        {
            "game": setup.name,
            "variant": setup.variant,
            "network": asdict(setup.network),
            "checkpoint_dir": str(args.checkpoint_dir),
            "device": args.device,
            "workers": args.workers,
            "elo": {
                "anchor": "random",
                "openings": args.openings,
                "openings_from": openings.source,
                "openings_fingerprint": openings.fingerprint,
                "solution": str(args.solution),
                "gaps": list(ELO_GAPS),
                "anchored": [generations[i] for i in anchored_generations(len(files))],
                "prior_draws": ELO_PRIOR_DRAWS,
                "simulations": simulations,
                "temperature": EVALUATION_TEMPERATURE,
            },
        }
    )

    started = time.perf_counter()
    arguments = (setup.name, files, openings.positions, simulations, args.device)
    finished: dict[int, PairScore[int]] = {}
    for number, score in play_pairs(pairs, arguments, workers=args.workers):
        finished[number] = score
        seconds = time.perf_counter() - started
        print(
            f"대진 {len(finished)}/{len(pairs)}, {seconds:.0f}초: {score.player} 대 "
            f"{score.opponent} {score.wins}승 {score.draws}무 {score.losses}패",
            file=sys.stderr,
            flush=True,
        )
    scores = [finished[number] for number in range(len(pairs))]
    ratings = fit_elo(scores, anchor=RANDOM, prior_draws=ELO_PRIOR_DRAWS)

    curve = [(generation, round(ratings[generation], 1)) for generation in files]
    for generation, rating in curve:
        logger.log({"elo/rating": rating}, step=generation)
    wandb_run.finish()
    summary = {
        "ratings": curve,
        "pairs": [asdict(score) for score in scores],
        "seconds": round(time.perf_counter() - started, 1),
    }
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
