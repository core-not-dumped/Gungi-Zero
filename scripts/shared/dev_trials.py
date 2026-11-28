"""튜닝에 쓰는 연습용 시험 판. 2.5단계의 시험 판 200개와 겹치지 않는다.

시험 판 점수를 보며 설정을 고르면 그 200판에 맞춰진다. 층과 개수는 같고 시드만 다르게 뽑아,
시험 판과 신경망에 같은 판(정규형의 `board_key`)은 뺀다.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from eval.solver import Solution
from eval.trial_positions import Trial, select_trials
from scripts.shared.trials import nano_trials, strata_of

DEV_CACHE = Path("solutions/nano-gungi-dev-trials.pickle")


DEV_TRIAL_SEED = 1


def dev_trials(solution: Solution[Any, Any]) -> tuple[Trial[Any], ...]:
    return select_trials(
        solution,
        strata_of(solution),
        seed=DEV_TRIAL_SEED,
        exclude=[trial.position for trial in nano_trials(solution)],
    )


def cached_dev_trials(
    solution: Solution[Any, Any], solution_path: Path, cache: Path
) -> tuple[Trial[Any], ...]:
    """뽑으려면 도달 가능한 상태를 두 번 돌아야 해서 파일에 남긴다. 풀이 파일이 더 새로우면 다시
    뽑는다. 파일은 이 함수가 쓴 것만 읽는다."""
    if cache.is_file() and cache.stat().st_mtime >= solution_path.stat().st_mtime:
        with cache.open("rb") as file:
            return pickle.load(file)
    trials = dev_trials(solution)
    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open("wb") as file:
        pickle.dump(trials, file)
    return trials
