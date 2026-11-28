"""체크포인트를 시험 판에서 채점한다. 튜닝(`scripts.tune`, `scripts.optimize`)과
`scripts.grade` 가 쓴다."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch

from ai.batched import EvaluationCache
from ai.checkpoint import load_network
from ai.graphed import GraphedNetwork
from ai.mcts import MCTS
from ai.network import Network
from eval.perfect import PerfectAI
from eval.solver import Solution
from eval.trial_match import play_trials_together
from eval.trial_positions import Trial
from games.mini_gungi import Gungi
from scripts.shared.hyperparameters import GRADE_EVALUATION_CACHE, GRADE_PARALLEL_TRIALS
from scripts.shared.setups import GameSetup
from scripts.shared.solution import load_solution_of
from scripts.shared.trials import fingerprint, nano_trials


def grade_trials(
    setup: GameSetup,
    solution_path: Path,
    *,
    simulations: int,
    device: str,
    trials: Sequence[Trial[Any]] | None = None,
    solution: Solution[Any, Any] | None = None,
) -> Callable[[Path], dict[str, Any]]:
    """체크포인트를 시험 판에서 채점하는 함수. 풀이는 여기서 한 번 읽는다. `trials` 를 주면
    2.5단계의 시험 판 대신 그것을, `solution` 을 주면 읽어 둔 풀이를 쓴다.

    시험 판을 함께 두며 평가를 묶고 답을 기억한다. CUDA 에서는 평가를 기록해 재생한다.
    """
    if solution is None:
        solution = load_solution_of(setup, solution_path)
    if trials is None:
        trials = nano_trials(solution)
    game = solution.game
    assert isinstance(game, Gungi)
    mark = fingerprint(game, trials)

    def grade(file: Path) -> dict[str, Any]:
        network = load_network(setup.game, file).to(device).eval()
        parallel = GRADE_PARALLEL_TRIALS
        evaluator: Network = network
        if torch.device(device).type == "cuda":
            evaluator = GraphedNetwork(network, game.tensor_shape, parallel, device)
        results = play_trials_together(
            game,
            trials,
            MCTS(game, evaluator, simulations=simulations, device=device),
            lambda index: PerfectAI(solution, index),
            network=evaluator,
            parallel=parallel,
            device=device,
            cache=EvaluationCache(GRADE_EVALUATION_CACHE),
        )
        names = dict.fromkeys(trial.stratum for trial in trials)
        strata = []
        for name in names:
            passed = [result.passed for result in results if result.trial.stratum == name]
            strata.append({"name": name, "passed": sum(passed), "total": len(passed)})
        return {
            "trials": {
                "passed": sum(result.passed for result in results),
                "total": len(results),
                "failed": [index for index, result in enumerate(results) if not result.passed],
                "strata": strata,
                "fingerprint": mark,
            }
        }

    return grade
