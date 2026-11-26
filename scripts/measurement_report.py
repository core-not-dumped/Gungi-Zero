"""랜덤 대국을 돌려 한 변형의 통계를 낸다. 배치·말 개수와 종료 규칙을 정하는 근거다.

uv run python -m scripts.measurement_report

출력은 `docs/measurement-report.md` 의 "측정" 절에 그대로 싣는다.
"""

from __future__ import annotations

import argparse
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from ai.random_play import play_random_episode
from games.base import Outcome
from games.mini_gungi import MINI_GUNGI, EndReason, Gungi, VariantConfig

OUTCOME_NAMES = {
    Outcome.FIRST_WINS: "선수 승률",
    Outcome.SECOND_WINS: "후수 승률",
    Outcome.DRAW: "무승부율",
}
REASON_NAMES = {
    EndReason.MARSHAL_CAPTURE: "帥 잡기",
    EndReason.MOVE_LIMIT: "수 제한",
    EndReason.REPETITION: "반복",
    EndReason.NO_LEGAL_MOVES: "합법수 없음",
}
PERCENTILES = (50, 90, 99)


@dataclass(frozen=True)
class Measurement:
    """대국 `i` 는 시드 `seed + i` 의 랜덤 대국이다. 길이는 둔 수의 개수다."""

    config: VariantConfig
    seed: int
    outcomes: dict[Outcome, int]
    reasons: dict[EndReason, int]
    lengths: tuple[int, ...]
    # 수를 둔 모든 국면의 합법수 개수의 평균.
    mean_branching_factor: float

    @property
    def episodes(self) -> int:
        return len(self.lengths)

    def percentile(self, percent: int) -> int:
        """대국의 `percent`% 이상이 이 길이 안에 끝난다 (nearest-rank)."""
        return sorted(self.lengths)[math.ceil(percent * self.episodes / 100) - 1]


def measure(config: VariantConfig, episodes: int, seed: int) -> Measurement:
    if episodes < 1:
        raise ValueError(f"대국 수는 1 이상이다: {episodes}")
    game = Gungi(config)
    outcomes: Counter[Outcome] = Counter()
    reasons: Counter[EndReason] = Counter()
    lengths: list[int] = []
    legal_moves = 0
    for number in range(episodes):
        episode = play_random_episode(game, seed + number)
        # 분기 계수는 `Episode` 에 없어 수를 다시 두며 센다.
        position = game.initial_position()
        for move in episode.moves:
            legal_moves += len(game.legal_moves(position))
            position = game.apply_move(position, move)
        reason = game.end_reason(episode.final_position)
        assert reason is not None, "랜덤 대국은 끝난 국면에서 멈춘다"
        outcomes[episode.outcome] += 1
        reasons[reason] += 1
        lengths.append(len(episode.moves))
    return Measurement(
        config=config,
        seed=seed,
        outcomes=dict(outcomes),
        reasons=dict(reasons),
        lengths=tuple(lengths),
        mean_branching_factor=legal_moves / sum(lengths),
    )


def render_summary(measured: Measurement) -> str:
    """마크다운 표 두 개: 수치와 종료 사유 분포."""
    total = measured.episodes

    def share(count: int) -> str:
        return f"{100 * count / total:.1f}%"

    lines = ["| 항목 | 값 |", "| --- | --- |"]
    lines += [
        f"| {name} | {share(measured.outcomes.get(outcome, 0))} |"
        for outcome, name in OUTCOME_NAMES.items()
    ]
    lines.append(f"| 평균 대국 길이 | {sum(measured.lengths) / total:.1f} |")
    lines += [f"| 길이 {p}% 지점 | {measured.percentile(p)} |" for p in PERCENTILES]
    lines.append(f"| 최대 대국 길이 | {max(measured.lengths)} |")
    lines.append(f"| 평균 분기 계수 | {measured.mean_branching_factor:.2f} |")
    lines += ["", "| 종료 사유 | 대국 | 비율 |", "| --- | --- | --- |"]
    for reason, name in REASON_NAMES.items():
        count = measured.reasons.get(reason, 0)
        lines.append(f"| {name} | {count} | {share(count)} |")
    return "\n".join(lines) + "\n"


def render_report(measured: Measurement) -> str:
    """측정한 설정과 초기 배치를 앞에 붙인 보고."""
    config = measured.config
    game = Gungi(config)
    last_seed = measured.seed + measured.episodes - 1
    return f"""\
랜덤 대국 {measured.episodes}판, 시드 {measured.seed}~{last_seed}. 모두 예외 없이 끝났다.
판 {config.width}x{config.height}, 최대 {config.max_tier}단, \
수 제한 {config.move_limit}, 반복 한도 {config.repetition_limit}.

```
{game.render(game.initial_position())}```

{render_summary(measured)}"""


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="미니 군의 랜덤 대국 측정")
    parser.add_argument("--episodes", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    measured = measure(MINI_GUNGI, args.episodes, args.seed)
    print(render_report(measured), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
