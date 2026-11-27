"""시험 판 대국: 시험 판마다 둘 차례인 쪽을 `challenger` 가 잡고 한 판을 둔다."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Generic

from ai.base import AI
from ai.episode import Episode, play_episode
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
    for trial in trials:
        if trial.value.verdict is Verdict.LOSS:
            raise ValueError(f"지는 국면은 시험 판이 아니다: 층 {trial.stratum!r}")
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
