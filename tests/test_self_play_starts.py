"""자가 대국의 시작 국면을 고르는 규칙 (ADR-0007)."""

import random
from collections import Counter

import pytest

from games.mini_gungi import Gungi
from scripts.shared import self_play_starts
from scripts.shared.self_play_starts import mixed_starts
from tests.solver_helpers import SMALL_NANO, shortest_path_positions

GAME = Gungi(SMALL_NANO)


def unfinished() -> list:
    return [p for p in shortest_path_positions(GAME) if GAME.outcome(p) is None]


def draw(initial_fraction: float, times: int, seed: int = 0) -> list:
    choose = mixed_starts(GAME, initial_fraction=initial_fraction)
    rng = random.Random(seed)
    return [choose(rng) for _ in range(times)]


def test_the_initial_position_gets_its_share() -> None:
    chosen = draw(0.25, 4000)

    share = sum(position == GAME.initial_position() for position in chosen) / len(chosen)
    assert 0.23 < share < 0.27


def test_the_others_are_unfinished_states_with_their_shortest_paths() -> None:
    candidates = set(unfinished())

    chosen = draw(0.0, 2000)

    # 경로와 `ply` 까지 같은 국면이다. 반복과 수 제한이 이어진다.
    assert set(chosen) <= candidates
    assert any(position.ply > 1 for position in chosen)


def test_every_state_is_about_as_likely() -> None:
    candidates = unfinished()
    times = 50 * len(candidates)

    counts = Counter(GAME.board_key(position) for position in draw(0.0, times))

    assert len(counts) == len(candidates)
    assert max(counts.values()) < 100


def test_the_same_random_state_gives_the_same_starts() -> None:
    assert draw(0.25, 50, seed=3) == draw(0.25, 50, seed=3)
    assert draw(0.25, 50, seed=3) != draw(0.25, 50, seed=4)


def test_the_states_are_walked_once_for_the_same_game(monkeypatch: pytest.MonkeyPatch) -> None:
    game = Gungi(SMALL_NANO)
    walks = []
    original = self_play_starts.reachable_positions

    def counting(walked):
        walks.append(walked)
        return original(walked)

    monkeypatch.setattr(self_play_starts, "reachable_positions", counting)

    first = mixed_starts(game, initial_fraction=0.0)
    second = mixed_starts(game, initial_fraction=0.5)

    assert len(walks) == 1
    assert first(random.Random(1)) == mixed_starts(game, initial_fraction=0.0)(random.Random(1))
    assert second(random.Random(1)) is not None
