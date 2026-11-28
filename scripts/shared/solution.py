"""나노 군의의 풀이 파일. `scripts.solve_nano` 가 만들고 시험 판·채점·Elo 가 읽는다."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from eval.solver import Solution, load_solution
from games.base import BoardOutcomeGame
from scripts.shared.setups import GameSetup

# git 에 넣지 않는다.
SOLUTION_PATH = Path("solutions/nano-gungi.pickle")


def _board_outcome_game(setup: GameSetup) -> BoardOutcomeGame[Any, Any]:
    # 군의는 `board_outcome` 을 답한다. `GameSetup` 은 공통 인터페이스로만 들고 있다.
    return cast(BoardOutcomeGame[Any, Any], setup.game)


def load_solution_of(setup: GameSetup, path: Path) -> Solution[Any, Any]:
    """`path` 의 풀이를 읽는다. `setup` 의 변형을 푼 것이 아니면 `ValueError` 다."""
    return load_solution(path, _board_outcome_game(setup), variant=setup.variant)
