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

import torch

from ai.base import AI
from ai.random_play import RandomAI
from eval.ladder import play_pair
from scripts.shared.grading import grade_trials
from scripts.shared.openings import SOLVED_GAMES, openings_for
from scripts.shared.players import checkpoint_player
from scripts.shared.setups import SETUPS, GameSetup
from scripts.shared.solution import SOLUTION_PATH

Player = Callable[[int], AI[Any, Any]]


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
    parser.add_argument("--device", help="기본은 CUDA 가 있으면 cuda, 없으면 cpu")
    parser.add_argument("--solution", type=Path, default=SOLUTION_PATH)
    parser.add_argument("--openings", type=int, default=20, help="풀이가 없는 게임의 시작 판 수")
    parser.add_argument("--opponent", type=Path, help="풀이가 없는 게임에서 함께 둘 기준 세대")
    args = parser.parse_args(argv)
    setup = SETUPS[args.game]
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    simulations = args.simulations or setup.evaluation_simulations
    solved = setup.name in SOLVED_GAMES
    if solved and args.opponent:
        parser.error("풀이가 있는 게임은 시험 판으로 채점한다. --opponent 를 쓰지 않는다")
    files = [*args.checkpoint, *([args.opponent] if args.opponent else [])]
    missing = [str(file) for file in files if not file.is_file()]
    if missing:
        parser.error(f"체크포인트가 없다: {missing}")

    def player_of(file: Path) -> Player:
        ai = checkpoint_player(setup, file, simulations=simulations, device=device)
        return lambda index: ai

    try:
        if solved:
            grade_trial_file = grade_trials(
                setup, args.solution, simulations=simulations, device=device
            )
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
            **(grade_trial_file(file) if solved else grade(player_of(file))),
        }
        print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
