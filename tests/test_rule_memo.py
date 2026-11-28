"""판+차례마다의 규칙 결과를 기억해도 답이 같은지. 반복과 수 제한은 경로를 따른다."""

import random

import pytest

import engine.rules
from engine.rules import EndReason, Move, RuleEngine
from games.mini_gungi import MINI_GUNGI
from games.nano_gungi import NANO_GUNGI

# 두 帥 가 한 칸 나갔다가 돌아온다. 네 수마다 같은 판, 같은 차례가 된다.
ROUND_TRIP = ((1, 4), (10, 7), (4, 1), (7, 10))


def test_the_same_board_ends_differently_depending_on_the_path() -> None:
    rules = RuleEngine(NANO_GUNGI)
    positions = [rules.initial_position()]
    for source, target in ROUND_TRIP * 2:
        positions.append(rules.apply_move(positions[-1], Move(source, target)))

    first, last = positions[0], positions[-1]

    assert first.board_key == last.board_key
    ended = rules.result(last)
    assert ended is not None
    assert ended.reason is EndReason.REPETITION
    assert rules.legal_moves(last) == ()
    # 같은 판의 처음 모습은 여전히 진행 중이다.
    assert rules.result(first) is None
    assert rules.legal_moves(first)
    # 판만 보는 답은 둘이 같다.
    assert rules.board_result(first) is None
    assert rules.board_result(last) is None


def walk(rules: RuleEngine, seed: int) -> list:
    rng = random.Random(seed)
    position = rules.initial_position()
    seen = []
    while True:
        result, moves = rules.result(position), rules.legal_moves(position)
        seen.append((result, moves, rules.board_result(position)))
        if result is not None:
            return seen
        position = rules.apply_move(position, rng.choice(moves))


@pytest.mark.parametrize("capacity", [1, 7])
def test_forgetting_does_not_change_any_answer(
    capacity: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = [walk(RuleEngine(MINI_GUNGI), seed) for seed in range(10)]
    monkeypatch.setattr(engine.rules, "REMEMBERED_BOARDS", capacity)
    rules = RuleEngine(MINI_GUNGI)

    assert [walk(rules, seed) for seed in range(10)] == expected
    # 다시 물어도 같다.
    assert [walk(rules, seed) for seed in range(10)] == expected


def test_two_variants_do_not_share_what_they_remember() -> None:
    mini, nano = RuleEngine(MINI_GUNGI), RuleEngine(NANO_GUNGI)

    assert mini.legal_moves(mini.initial_position()) != nano.legal_moves(nano.initial_position())
    assert RuleEngine(NANO_GUNGI) == nano
