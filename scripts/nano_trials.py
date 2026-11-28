"""나노 군의의 시험 판을 정하는 합성 루트 (ADR-0007).

uv run python -m scripts.nano_trials

`scripts.solve_nano` 가 만든 풀이에서 시험 판 200개를 뽑아, 층마다의 개수와 지문을 JSON 한 줄로
찍는다. 층·개수·시드를 바꾸면 이전의 검증 결과와 비교할 수 없다.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from games.mini_gungi import Gungi
from scripts.shared.setups import SETUPS
from scripts.shared.solution import SOLUTION_PATH, load_solution_of
from scripts.shared.trials import (
    TRIAL_SEED,
    fingerprint,
    nano_trials,
)


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
