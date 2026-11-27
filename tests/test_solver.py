"""solver. 남이 아는 값(틱택토), 손으로 풀리는 작은 변형, 실제 규칙 위의 깊이 탐색에 댄다."""

import pickle
import random
from dataclasses import replace
from pathlib import Path

import pytest

from eval.solver import Solution, Value, Verdict, load_solution, solve
from games.base import Player
from games.mini_gungi import Gungi, GungiPosition
from games.tictactoe import TicTacToe
from tests.solver_helpers import (
    FACING_MARSHALS,
    SMALL_NANO,
    ForcedWinSearch,
    forces_a_win,
    shortest_path_positions,
    winner_of,
    with_ply,
)


@pytest.fixture(scope="module")
def small_nano() -> Gungi:
    return Gungi(SMALL_NANO)


@pytest.fixture(scope="module")
def small_nano_solution(small_nano: Gungi) -> Solution:
    return solve(small_nano, move_limit=SMALL_NANO.move_limit)


def test_tictactoe_is_a_draw_over_the_known_number_of_positions() -> None:
    game = TicTacToe()
    solution = solve(game, move_limit=None)
    assert solution.value(game.initial_position()) == Value(Verdict.DRAW, None)
    assert len(solution) == 5478


def test_a_hand_solved_gungi_variant() -> None:
    game = Gungi(FACING_MARSHALS)
    solution = solve(game, move_limit=FACING_MARSHALS.move_limit)
    position = game.initial_position()
    assert solution.value(position) == Value(Verdict.LOSS, 2)
    position = game.apply_move(position, game.parse_move("a1-b1"))
    assert solution.value(position) == Value(Verdict.WIN, 1)
    position = game.apply_move(position, game.parse_move("c1-b1"))
    assert solution.value(position) == Value(Verdict.LOSS, 0)
    assert len(solution) == 3


def test_the_small_nano_candidate_has_the_documented_number_of_states(
    small_nano_solution: Solution,
) -> None:
    assert len(small_nano_solution) == 1812


def test_decided_values_hold_in_a_search_over_the_real_rules(
    small_nano: Gungi, small_nano_solution: Solution
) -> None:
    """승패는 DTM 안에 강제되고, 한 수 적게는 강제되지 않는다."""
    decided: dict[Verdict, list[tuple[GungiPosition, int]]] = {Verdict.WIN: [], Verdict.LOSS: []}
    for position in shortest_path_positions(small_nano):
        value = small_nano_solution.value(position)
        if value.dtm is not None:
            decided[value.verdict].append((position, value.dtm))

    rng = random.Random(0)
    for verdict, found in decided.items():
        # 무작위 표본에 가장 긴 수순을 더한다.
        sample = [*rng.sample(found, 60), max(found, key=lambda item: item[1])]
        for position, dtm in sample:
            winner = winner_of(position.player, verdict)
            assert forces_a_win(small_nano, position, winner, dtm), small_nano.render(position)
            if dtm > 0:
                assert not forces_a_win(small_nano, position, winner, dtm - 1)


@pytest.mark.parametrize("verdict", [Verdict.WIN, Verdict.LOSS])
def test_a_result_that_does_not_fit_in_the_move_limit_is_a_draw(
    small_nano: Gungi, small_nano_solution: Solution, verdict: Verdict
) -> None:
    for position in shortest_path_positions(small_nano):
        value = small_nano_solution.value(position)
        if value.verdict is verdict and value.dtm is not None and value.dtm >= 3:
            break
    else:
        pytest.fail("DTM 3 이상인 국면이 없다")
    winner = winner_of(position.player, verdict)
    last_ply = SMALL_NANO.move_limit - value.dtm

    in_time = with_ply(small_nano, position, last_ply)
    assert small_nano_solution.value(in_time) == value
    assert forces_a_win(small_nano, in_time, winner, value.dtm)

    too_late = with_ply(small_nano, position, last_ply + 1)
    assert small_nano_solution.value(too_late) == Value(Verdict.DRAW, None)
    assert not forces_a_win(small_nano, too_late, winner, value.dtm)


def test_a_position_the_path_has_ended_takes_the_games_result() -> None:
    """판만 보면 진행 중인 국면이라도 반복으로 끝났으면 무승부다."""
    game = Gungi(SMALL_NANO)
    solution = solve(game, move_limit=SMALL_NANO.move_limit)
    position = game.initial_position()
    for text in ("b1-c1", "b3-a3", "c1-b1", "a3-b3") * 2:
        position = game.apply_move(position, game.parse_move(text))
    assert game.outcome(position) is not None and game.board_outcome(position) is None
    assert solution.value(position) == Value(Verdict.DRAW, None)


def test_a_move_limit_that_cuts_the_graph_short_is_refused() -> None:
    config = replace(SMALL_NANO, move_limit=2)
    with pytest.raises(ValueError):
        solve(Gungi(config), move_limit=config.move_limit)


def test_a_position_outside_the_solved_graph_is_refused() -> None:
    game = Gungi(FACING_MARSHALS)
    solution = solve(game, move_limit=FACING_MARSHALS.move_limit)
    unreachable = game.parse("     a       b       c\n 1  ▲帥      .      ▽帥\n차례: ▽   ply: 0\n")
    with pytest.raises(ValueError):
        solution.value(unreachable)


def test_it_counts_the_states_by_value_and_finds_the_longest_dtm() -> None:
    solution = solve(Gungi(FACING_MARSHALS), move_limit=FACING_MARSHALS.move_limit)
    assert solution.verdict_counts() == {Verdict.WIN: 1, Verdict.DRAW: 0, Verdict.LOSS: 2}
    assert solution.max_dtm() == 2


def test_the_counts_do_not_depend_on_the_move_limit(
    small_nano: Gungi, small_nano_solution: Solution
) -> None:
    unlimited = solve(small_nano, move_limit=None)
    assert small_nano_solution.verdict_counts() == unlimited.verdict_counts()
    assert sum(unlimited.verdict_counts().values()) == len(unlimited)
    assert small_nano_solution.max_dtm() == 5


def test_a_saved_solution_answers_the_same_with_its_own_move_limit(
    small_nano: Gungi, small_nano_solution: Solution, tmp_path: Path
) -> None:
    path = tmp_path / "made" / "small.pickle"
    small_nano_solution.save(path, variant={"name": "small"})

    loaded = load_solution(path, small_nano, variant={"name": "small"})

    assert loaded.move_limit == SMALL_NANO.move_limit
    assert len(loaded) == len(small_nano_solution)
    # 수 제한에 걸리는 `ply` 까지 같아야 한다.
    for position in shortest_path_positions(small_nano):
        for ply in (position.ply, SMALL_NANO.move_limit - 3):
            late = with_ply(small_nano, position, ply)
            assert loaded.value(late) == small_nano_solution.value(late)


def test_a_solution_of_another_variant_is_refused(
    small_nano: Gungi, small_nano_solution: Solution, tmp_path: Path
) -> None:
    path = tmp_path / "small.pickle"
    small_nano_solution.save(path, variant={"move_limit": 200})
    with pytest.raises(ValueError, match="변형"):
        load_solution(path, small_nano, variant={"move_limit": 100})


def test_a_solution_in_another_format_is_refused(small_nano: Gungi, tmp_path: Path) -> None:
    path = tmp_path / "other.pickle"
    path.write_bytes(pickle.dumps({"format": 0}))
    with pytest.raises(ValueError, match="형식"):
        load_solution(path, small_nano, variant={})


def test_the_remembering_search_agrees_with_the_search_over_the_real_rules(
    small_nano: Gungi,
) -> None:
    """DTM 이 긴 판의 확인(`test_solve_nano`)이 기대는 탐색을, 가정이 없는 탐색에 댄다."""
    search = ForcedWinSearch(small_nano)
    # 승·무·패가 섞여 있어 두 답이 모두 나온다.
    for position in shortest_path_positions(small_nano):
        for winner in Player:
            for plies in range(6):
                expected = forces_a_win(small_nano, position, winner, plies)
                assert search.forces(position, winner, plies) is expected
