"""나노 군의의 시험 판을 정하는 합성 루트 (ADR-0007).

uv run python -m scripts.nano_trials

`scripts.solve_nano` 가 만든 풀이에서 시험 판 200개를 뽑아, 층마다의 개수와 지문을 JSON 한 줄로
찍는다. 층·개수·시드를 바꾸면 이전의 검증 결과와 비교할 수 없다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from eval.solver import Solution, Value, Verdict
from eval.trial_positions import Stratum, Trial, select_trials
from games.mini_gungi import Gungi, GungiPosition
from scripts.solve_nano import SOLUTION_PATH, load_solution_of
from scripts.train import SETUPS

TRIAL_SEED = 0


def involves_stacking(game: Gungi, position: GungiPosition) -> bool:
    """2단 탑이 있거나 쌓기가 합법인 판."""
    squares = range(game.geometry.square_count)
    if any(len(game.tower(position, square)) > 1 for square in squares):
        return True
    return any(move.stack for move in game.legal_moves(position))


def _draw_with_a_losing_move(
    solution: Solution[Any, Any], position: GungiPosition, value: Value
) -> bool:
    """비기는 판 가운데 지는 수가 있는 것. 어느 수나 비기는 판은 첫 수를 시험하지 못한다."""
    if value.verdict is not Verdict.DRAW:
        return False
    game = solution.game
    return any(
        solution.value(game.apply_move(position, move)).verdict is Verdict.WIN
        for move in game.legal_moves(position)
    )


def _win_within(position_value: Value, shortest: int, longest: int | None) -> bool:
    dtm = position_value.dtm
    if position_value.verdict is not Verdict.WIN or dtm is None:
        return False
    return dtm >= shortest and (longest is None or dtm <= longest)


def strata_of(solution: Solution[Any, Any]) -> tuple[Stratum[GungiPosition], ...]:
    """승 100(DTM 13 이상 40, 9~11 40, 5~7 20)과 무 100. 층마다 4분의 3 이 쌓기가 걸린 판이다."""
    game = solution.game
    assert isinstance(game, Gungi), "나노 군의의 풀이를 받는다"

    def stratum(name: str, count: int, stacking: bool, wanted: Any) -> Stratum[GungiPosition]:
        return Stratum(
            f"{name}/{'stack' if stacking else 'plain'}",
            count,
            lambda position, value: (
                wanted(position, value) and involves_stacking(game, position) is stacking
            ),
        )

    # (이름, 쌓기 판 수, 그 밖의 판 수, 조건). 후보가 적은 층이 앞이다.
    bands = (
        ("win-13+", 30, 10, lambda position, value: _win_within(value, 13, None)),
        ("win-9-11", 30, 10, lambda position, value: _win_within(value, 9, 11)),
        ("win-5-7", 15, 5, lambda position, value: _win_within(value, 5, 7)),
        (
            "draw",
            75,
            25,
            lambda position, value: _draw_with_a_losing_move(solution, position, value),
        ),
    )
    return tuple(
        stratum(name, count, stacking, wanted)
        for name, stacked, plain, wanted in bands
        for stacking, count in ((True, stacked), (False, plain))
    )


def nano_trials(solution: Solution[Any, Any]) -> tuple[Trial[GungiPosition], ...]:
    """2.5단계 검증의 시험 판."""
    return select_trials(solution, strata_of(solution), seed=TRIAL_SEED)


def fingerprint(game: Gungi, trials: Sequence[Trial[GungiPosition]]) -> str:
    """시험 판의 층·판·차례·`ply` 와 순서를 요약한 값. 시험 판이 달라졌는지 본다."""
    text = "".join(f"{trial.stratum}\n{game.render(trial.position)}" for trial in trials)
    return hashlib.sha256(text.encode()).hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="나노 군의의 시험 판을 뽑는다")
    parser.add_argument("--solution", type=Path, default=SOLUTION_PATH)
    args = parser.parse_args(argv)
    solution = load_solution_of(SETUPS["nano-gungi"], args.solution)
    game = solution.game
    assert isinstance(game, Gungi)
    trials = nano_trials(solution)
    print(
        json.dumps(
            {
                "seed": TRIAL_SEED,
                "strata": Counter(trial.stratum for trial in trials),
                "players": Counter(trial.position.player.name for trial in trials),
                "fingerprint": fingerprint(game, trials),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
