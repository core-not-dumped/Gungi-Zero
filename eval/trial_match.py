"""시험 판 대국: 시험 판마다 둘 차례인 쪽을 `challenger` 가 잡고 한 판을 둔다.

신경망으로 두는 MCTS 는 `play_trials_together` 로 여러 판을 함께 두며 평가를 묶는다.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Generic

from ai.base import AI
from ai.batched import EvaluationCache, Steps, run_batched
from ai.episode import Episode, play_episode
from ai.mcts import MCTS
from ai.network import Network
from eval.solver import Verdict
from eval.trial_positions import Trial
from games.base import Game, MoveT, Player, PositionT


@dataclass(frozen=True)
class TrialResult(Generic[PositionT, MoveT]):
    """시험 판 하나의 대국과 판정."""

    trial: Trial[PositionT]
    # 시험 판에서 시작한 대국.
    episode: Episode[PositionT, MoveT]

    @property
    def passed(self) -> bool:
        """이론값을 지켰는지. 승리 판은 이겨야 하고 무승부 판은 지지 않아야 한다."""
        player = self.trial.position.player
        winner = self.episode.outcome.winner
        if self.trial.value.verdict is Verdict.WIN:
            return winner is player
        return winner is not player.opponent


def play_trials(
    game: Game[PositionT, MoveT],
    trials: Sequence[Trial[PositionT]],
    challenger: Callable[[int], AI[PositionT, MoveT]],
    opponent: Callable[[int], AI[PositionT, MoveT]],
) -> tuple[TrialResult[PositionT, MoveT], ...]:
    """시험 판 순서대로 둔다. `challenger` 와 `opponent` 는 시험 판의 순번(0 부터)을 받아 AI 를
    만든다. 대국은 수 제한을 포함한 게임의 규칙이 끝낸다.

    지는 시험 판은 지킬 것이 없어 `ValueError` 다.
    """
    _refuse_lost(trials)
    results = []
    for index, trial in enumerate(trials):
        tested, other = challenger(index), opponent(index)
        first = trial.position.player is Player.FIRST
        episode = play_episode(
            game,
            tested if first else other,
            other if first else tested,
            start=trial.position,
        )
        results.append(TrialResult(trial, episode))
    return tuple(results)


def _refuse_lost(trials: Sequence[Trial[PositionT]]) -> None:
    for trial in trials:
        if trial.value.verdict is Verdict.LOSS:
            raise ValueError(f"지는 국면은 시험 판이 아니다: 층 {trial.stratum!r}")


def play_trials_together(
    game: Game[PositionT, MoveT],
    trials: Sequence[Trial[PositionT]],
    search: MCTS[PositionT, MoveT],
    opponent: Callable[[int], AI[PositionT, MoveT]],
    *,
    network: Network,
    parallel: int,
    device: str = "cpu",
    cache: EvaluationCache | None = None,
) -> tuple[TrialResult[PositionT, MoveT], ...]:
    """`play_trials` 를 `parallel` 판까지 함께 두며 신경망 평가를 묶어 한다. 시험받는 쪽은
    temperature 0 의 `search` 이고, 가장 많이 방문한 수가 여럿이면 시험 판의 순번을 시드로 고른다.

    판을 따로따로 평가하는 `network` 라면, `challenger` 가 순번을 시드로 한 `MCTSAI` 인
    `play_trials` 와 같은 대국이다. `search` 의 신경망은 쓰지 않고 `network` 가 평가한다. 지는 시험
    판은 `ValueError` 다.
    """
    _refuse_lost(trials)

    def one(index: int) -> Steps[TrialResult[PositionT, MoveT]]:
        trial = trials[index]
        tie_break, other = random.Random(index), opponent(index)
        position = trial.position
        moves: list[MoveT] = []
        while (outcome := game.outcome(position)) is None:
            if position.player is trial.position.player:
                result = yield from search.search_steps(position)
                move = result.select_move(0.0, tie_break)
            else:
                move = other.choose(position)
            moves.append(move)
            position = game.apply_move(position, move)
        return TrialResult(trial, Episode(tuple(moves), position, outcome))

    return tuple(
        run_batched(
            (one(index) for index in range(len(trials))),
            network,
            width=parallel,
            device=device,
            cache=cache,
        )
    )
