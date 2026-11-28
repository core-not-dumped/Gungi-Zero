"""저장한 세대를 빠르게 채점하는 합성 루트. 진단(가치 일치율, 최선수 비율)은 하지 않는다.

uv run python -m scripts.grade --checkpoint <파일> [<파일> …]
uv run python -m scripts.grade --game mini-gungi --checkpoint <파일> --opponent <기준 파일>

풀이가 있는 게임(나노 군의)은 시험 판 200개의 통과 수와 실패한 판의 순번을 낸다. 그 밖의 게임은
시작 판마다 선후를 바꿔 랜덤 AI, 그리고 `--opponent` 의 세대와 둔 성적을 낸다. 체크포인트마다
JSON 한 줄이다. 완료 기준의 판정은 `scripts.verify_nano` 가 한다.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from ai.base import AI
from ai.random_play import RandomAI
from eval.ladder import play_pair
from eval.perfect import PerfectAI
from eval.solver import Solution
from eval.trial_match import play_trials
from eval.trial_positions import Trial
from games.mini_gungi import Gungi
from scripts.levels import checkpoint_player
from scripts.nano_trials import fingerprint, nano_trials
from scripts.openings import SOLVED_GAMES, openings_for
from scripts.solve_nano import SOLUTION_PATH, load_solution_of
from scripts.train import SETUPS, GameSetup

Player = Callable[[int], AI[Any, Any]]


def grade_trials(
    setup: GameSetup,
    solution_path: Path,
    *,
    trials: Sequence[Trial[Any]] | None = None,
    solution: Solution[Any, Any] | None = None,
) -> Callable[[Player], dict[str, Any]]:
    """AI 를 시험 판에서 채점하는 함수. 풀이는 여기서 한 번 읽는다. `trials` 를 주면 2.5단계의
    시험 판 대신 그것을, `solution` 을 주면 읽어 둔 풀이를 쓴다."""
    if solution is None:
        solution = load_solution_of(setup, solution_path)
    if trials is None:
        trials = nano_trials(solution)
    game = solution.game
    assert isinstance(game, Gungi)
    mark = fingerprint(game, trials)

    def grade(player: Player) -> dict[str, Any]:
        results = play_trials(game, trials, player, lambda index: PerfectAI(solution, index))
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


def grade_matches(
    setup: GameSetup, count: int, opponents: Sequence[tuple[str, Player]]
) -> Callable[[Player], dict[str, Any]]:
    """AI 를 상대들과 두게 해 채점하는 함수."""
    openings = openings_for(setup, count)

    def grade(player: Player) -> dict[str, Any]:
        matches = []
        for name, opponent in opponents:
            score = play_pair(setup.game, ("graded", player), (name, opponent), openings.positions)
            games = score.wins + score.draws + score.losses
            matches.append(
                {
                    "opponent": name,
                    "wins": score.wins,
                    "draws": score.draws,
                    "losses": score.losses,
                    "score": (score.wins + score.draws / 2) / games,
                }
            )
        return {
            "openings": {
                "source": openings.source,
                "count": len(openings.positions),
                "fingerprint": openings.fingerprint,
            },
            "matches": matches,
        }

    return grade


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="저장한 세대를 채점한다")
    parser.add_argument("--game", choices=tuple(SETUPS), default="nano-gungi")
    parser.add_argument("--checkpoint", type=Path, nargs="+", required=True)
    parser.add_argument("--simulations", type=int, help="기본은 그 게임의 평가 탐색 횟수")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--solution", type=Path, default=SOLUTION_PATH)
    parser.add_argument("--openings", type=int, default=20, help="풀이가 없는 게임의 시작 판 수")
    parser.add_argument("--opponent", type=Path, help="풀이가 없는 게임에서 함께 둘 기준 세대")
    args = parser.parse_args(argv)
    setup = SETUPS[args.game]
    simulations = args.simulations or setup.evaluation_simulations
    solved = setup.name in SOLVED_GAMES
    if solved and args.opponent:
        parser.error("풀이가 있는 게임은 시험 판으로 채점한다. --opponent 를 쓰지 않는다")
    files = [*args.checkpoint, *([args.opponent] if args.opponent else [])]
    missing = [str(file) for file in files if not file.is_file()]
    if missing:
        parser.error(f"체크포인트가 없다: {missing}")

    def player_of(file: Path) -> Player:
        ai = checkpoint_player(setup, file, simulations=simulations, device=args.device)
        return lambda index: ai

    try:
        if solved:
            grade = grade_trials(setup, args.solution)
        else:
            opponents: list[tuple[str, Player]] = [
                ("random", lambda index: RandomAI(setup.game, index))
            ]
            if args.opponent:
                opponents.append((str(args.opponent), player_of(args.opponent)))
            grade = grade_matches(setup, args.openings, opponents)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    for file in args.checkpoint:
        result = {
            "game": setup.name,
            "checkpoint": str(file),
            "simulations": simulations,
            **grade(player_of(file)),
        }
        print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
