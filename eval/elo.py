"""대국 성적에서 Elo 를 맞춘다.

세대마다 차례로 갱신하지 않고 모든 대진의 성적을 한 번에 맞춘다 (Bradley-Terry 의 최대 우도).
그래서 대진을 늘어놓는 순서가 점수를 바꾸지 않는다. 무승부는 반 승이다.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np

PlayerT = TypeVar("PlayerT", bound=Hashable)

# 점수 차 400 이 승산 10배다.
_SCALE = math.log(10) / 400


@dataclass(frozen=True, slots=True)
class PairScore(Generic[PlayerT]):
    """`player` 시점에서 본, `opponent` 와 둔 대국의 승·무·패 수."""

    player: PlayerT
    opponent: PlayerT
    wins: float
    draws: float
    losses: float


def expected_score(difference: float) -> float:
    """점수가 `difference` 만큼 높은 쪽이 한 판에서 기대하는 성적 (승 1, 무 0.5)."""
    return 1 / (1 + 10 ** (-difference / 400))


def fit_elo(
    scores: Sequence[PairScore[PlayerT]],
    *,
    anchor: PlayerT,
    anchor_rating: float = 0.0,
    prior_draws: float = 1.0,
) -> dict[PlayerT, float]:
    """`anchor` 를 `anchor_rating` 에 고정하고 나머지의 점수를 맞춘다.

    `prior_draws` 는 대진마다 더하는 가상의 무승부 수다. 한쪽이 전승한 대진이 있으면 점수 차가
    끝없이 벌어지므로 이것이 막는다. 그만큼 점수 차는 조금 좁게 나온다. 점수가 정해지지 않는
    성적(기준과 이어지지 않거나, 가상의 무승부 없이 전승·전패로만 이어진 참가자)은 `ValueError` 다.
    """
    players = list(dict.fromkeys([anchor, *(p for s in scores for p in (s.player, s.opponent))]))
    index = {player: number for number, player in enumerate(players)}
    first = np.array([index[s.player] for s in scores], dtype=int)
    second = np.array([index[s.opponent] for s in scores], dtype=int)
    games = np.array([s.wins + s.draws + s.losses + prior_draws for s in scores], dtype=float)
    points = np.array([s.wins + 0.5 * (s.draws + prior_draws) for s in scores], dtype=float)
    _require_ratings(len(players), first, second, points, games - points)

    def log_likelihood(x: np.ndarray) -> float:
        gap = x[first] - x[second]
        return float(
            -np.sum(points * np.logaddexp(0, -gap) + (games - points) * np.logaddexp(0, gap))
        )

    # 기준(0번)은 0 에 두고 나머지만 뉴턴법으로 푼다.
    x = np.zeros(len(players))
    best = log_likelihood(x)
    for _ in range(200):
        win = 1 / (1 + np.exp(x[second] - x[first]))
        residual = points - games * win
        gradient = np.zeros(len(players))
        np.add.at(gradient, first, residual)
        np.add.at(gradient, second, -residual)
        if np.max(np.abs(gradient[1:]), initial=0.0) < 1e-9:
            break
        weight = games * win * (1 - win)
        curvature = np.zeros((len(players), len(players)))
        np.add.at(curvature, (first, first), weight)
        np.add.at(curvature, (second, second), weight)
        np.add.at(curvature, (first, second), -weight)
        np.add.at(curvature, (second, first), -weight)
        step = np.zeros(len(players))
        step[1:] = np.linalg.solve(curvature[1:, 1:], gradient[1:])
        # 지나쳐 우도가 내려가면 걸음을 줄인다.
        size = 1.0
        while (reached := log_likelihood(x + size * step)) < best and size > 1e-8:
            size /= 2
        x, best = x + size * step, reached
    return {player: anchor_rating + float(x[index[player]]) / _SCALE for player in players}


def _require_ratings(
    count: int, first: np.ndarray, second: np.ndarray, won: np.ndarray, lost: np.ndarray
) -> None:
    """점수를 딴 방향으로 이은 그래프가 강하게 연결되어야 최대 우도가 유한하게 하나다."""
    beats: list[set[int]] = [set() for _ in range(count)]
    beaten: list[set[int]] = [set() for _ in range(count)]
    for a, b, a_points, b_points in zip(first.tolist(), second.tolist(), won, lost, strict=True):
        for winner, loser, scored in ((a, b, a_points), (b, a, b_points)):
            if scored > 0:
                beats[winner].add(loser)
                beaten[loser].add(winner)
    for graph in (beats, beaten):
        seen = {0}
        stack = [0]
        while stack:
            for following in graph[stack.pop()] - seen:
                seen.add(following)
                stack.append(following)
        if len(seen) < count:
            raise ValueError(
                "기준과 승패 양쪽으로 이어지지 않은 참가자가 있어 점수가 정해지지 않는다"
            )
