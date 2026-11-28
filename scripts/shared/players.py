"""평가 대국의 AI: 저장한 세대로 두는 쪽.

Elo(`scripts.rate`, `scripts.make_levels`), 채점(`scripts.grade`), 대국 화면의 난이도
(`scripts.shared.levels`)가 같은 AI 로 둔다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ai.checkpoint import load_network
from ai.mcts import MCTS, MCTSAI
from eval.ladder import MemoizedAI
from scripts.shared.hyperparameters import EVALUATION_TEMPERATURE
from scripts.shared.setups import GameSetup

# 랜덤 AI 의 번호. 세대는 1 부터다.
RANDOM = 0


def checkpoint_player(
    setup: GameSetup, path: Path, *, simulations: int, device: str
) -> MemoizedAI[Any, Any]:
    """저장한 세대로 두는 평가용 MCTS AI. 국면마다 한 번만 탐색한다."""
    network = load_network(setup.game, path).to(device).eval()
    search = MCTS(setup.game, network, simulations=simulations, device=device)
    return MemoizedAI(MCTSAI(search, temperature=EVALUATION_TEMPERATURE, seed=0))
