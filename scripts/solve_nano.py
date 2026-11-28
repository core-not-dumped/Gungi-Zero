"""나노 군의를 풀어 파일로 저장하는 합성 루트.

uv run python -m scripts.solve_nano

저장한 파일을 다시 읽어 완벽한 상대끼리 시드 0~9 를 두고, 수치를 JSON 한 줄로 찍는다. 시작
국면의 이론값과 다르게 끝난 대국이 있으면 종료 코드가 1 이다. 수치와 해석은
`docs/nano-solution.md` 에 있다.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ai.episode import play_episode
from eval.perfect import PerfectAI
from eval.solver import Value, Verdict, solve
from games.base import Outcome, Player
from scripts.shared.setups import SETUPS, GameSetup
from scripts.shared.solution import SOLUTION_PATH, _board_outcome_game, load_solution_of

SEEDS = range(10)


@dataclass(frozen=True, slots=True)
class PerfectEpisode:
    """완벽한 상대끼리 둔 한 판."""

    seed: int
    outcome: Outcome
    plies: int


@dataclass(frozen=True)
class Report:
    """풀이 한 번의 수치. 승·무·패와 DTM 은 수 제한을 보지 않은 값이다."""

    states: int
    # 시작 국면의 이론값. 수 제한을 반영한다.
    start: Value
    counts: dict[Verdict, int]
    max_dtm: int | None
    move_limit: int | None
    # 푸는 데 걸린 시간. 저장과 대국은 뺀다.
    solve_seconds: float
    episodes: tuple[PerfectEpisode, ...]

    @property
    def holds(self) -> bool:
        """모든 대국이 시작 국면의 이론값으로 끝났는지."""
        # 시작 국면은 선수 차례다.
        expected = {
            Verdict.WIN: Outcome.win_for(Player.FIRST),
            Verdict.DRAW: Outcome.DRAW,
            Verdict.LOSS: Outcome.win_for(Player.SECOND),
        }[self.start.verdict]
        return all(episode.outcome is expected for episode in self.episodes)


def solve_to_file(setup: GameSetup, path: Path, *, seeds: Sequence[int]) -> Report:
    """`setup` 의 게임을 풀어 `path` 에 저장하고, 다시 읽은 풀이로 시드마다 한 판씩 둔다."""
    started = time.perf_counter()
    solved = solve(_board_outcome_game(setup), move_limit=setup.variant["move_limit"])
    seconds = time.perf_counter() - started
    solved.save(path, variant=setup.variant)
    # 두 벌을 함께 들고 있지 않는다.
    del solved

    solution = load_solution_of(setup, path)
    game = solution.game
    episodes = []
    for seed in seeds:
        ai = PerfectAI(solution, seed)
        episode = play_episode(game, ai, ai)
        episodes.append(PerfectEpisode(seed, episode.outcome, len(episode.moves)))
    return Report(
        states=len(solution),
        start=solution.value(game.initial_position()),
        counts=solution.verdict_counts(),
        max_dtm=solution.max_dtm(),
        move_limit=solution.move_limit,
        solve_seconds=seconds,
        episodes=tuple(episodes),
    )


def _summary(report: Report) -> dict[str, Any]:
    return {
        **asdict(report),
        "start": {"verdict": report.start.verdict.name, "dtm": report.start.dtm},
        "counts": {verdict.name: count for verdict, count in report.counts.items()},
        "solve_seconds": round(report.solve_seconds, 1),
        "episodes": [
            {**asdict(episode), "outcome": episode.outcome.name} for episode in report.episodes
        ],
        "holds": report.holds,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="나노 군의를 풀어 파일로 저장한다")
    parser.add_argument("--output", type=Path, default=SOLUTION_PATH)
    args = parser.parse_args(argv)
    report = solve_to_file(SETUPS["nano-gungi"], args.output, seeds=SEEDS)
    print(json.dumps(_summary(report)))
    return 0 if report.holds else 1


if __name__ == "__main__":
    raise SystemExit(main())
