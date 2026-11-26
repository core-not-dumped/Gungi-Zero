"""상태 공간 세기. 남이 아는 값(틱택토)과 손으로 셀 수 있는 작은 변형으로 확인한다."""

from dataclasses import replace

import pytest

from eval.reachable import StateCount, count_reachable
from games.mini_gungi import MINI_GUNGI, Gungi
from games.tictactoe import TicTacToe
from scripts.state_space import CANDIDATES, main
from tests.solver_helpers import FACING_MARSHALS


def test_the_tictactoe_count_matches_the_known_number_of_reachable_positions() -> None:
    # 빈 판을 포함해 5478 국면이고, 그중 끝난 국면은 958 개다. 마지막 국면은 9 수째다.
    counted = count_reachable(TicTacToe(), limit=10_000)
    assert counted == StateCount(states=5478, terminal=958, depth=9, complete=True)


def test_a_hand_counted_gungi_variant() -> None:
    counted = count_reachable(Gungi(FACING_MARSHALS), limit=100)
    assert counted == StateCount(states=3, terminal=1, depth=2, complete=True)


def test_the_move_limit_bounds_what_is_reachable() -> None:
    counted = count_reachable(Gungi(replace(FACING_MARSHALS, move_limit=1)), limit=100)
    # 수 제한에 닿은 국면은 판만으로 끝난 것이 아니다.
    assert counted == StateCount(states=2, terminal=0, depth=1, complete=True)


def test_counting_stops_when_there_are_more_states_than_the_limit() -> None:
    counted = count_reachable(TicTacToe(), limit=100)
    assert not counted.complete
    assert counted.states == 100


def test_a_space_exactly_as_large_as_the_limit_is_complete() -> None:
    assert count_reachable(TicTacToe(), limit=5478).complete
    assert not count_reachable(TicTacToe(), limit=5477).complete


def test_the_limit_counts_at_least_the_initial_position() -> None:
    with pytest.raises(ValueError):
        count_reachable(TicTacToe(), limit=0)


def test_candidates_differ_only_in_board_and_pieces() -> None:
    assert len({candidate.name for candidate in CANDIDATES}) == len(CANDIDATES)
    for candidate in CANDIDATES:
        config = candidate.config
        assert config.max_tier == MINI_GUNGI.max_tier
        assert config.move_limit == MINI_GUNGI.move_limit
        assert config.repetition_limit == MINI_GUNGI.repetition_limit


def test_the_report_has_a_row_for_every_candidate_and_marks_the_capped_ones(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["--limit", "2000"]) == 0
    printed = capsys.readouterr().out
    rows = {line.split("|")[1].strip(): line for line in printed.splitlines() if "|" in line}
    for candidate in CANDIDATES:
        assert candidate.name in rows
    # 가장 작은 후보는 2000 안에서 다 세어지고, 그보다 큰 후보는 상한에서 멈춘다.
    assert "| 1,812 | 240 | 11 | 가능 |" in rows["3x3 帥兵"]
    assert "| 2,000 초과 | - | - | 완전 탐색 불가 |" in rows["3x3 帥大"]


def test_an_unmeasured_candidate_is_reported_as_such_without_counting(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["--limit", "10"]) == 0
    printed = capsys.readouterr().out
    unmeasured = [line.split("|")[1].strip() for line in printed.splitlines() if "미측정" in line]
    assert unmeasured == ["3x4 帥大槍", "4x4 帥大兵", "3x4 帥大槍兵", "5x5 帥大槍兵兵"]
