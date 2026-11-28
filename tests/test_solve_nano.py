"""나노 군의 풀이의 합성 루트: 만든 파일, 낸 수치, 완벽한 상대끼리의 대국."""

import json
import random
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest

from eval.reachable import reachable_positions
from eval.solver import Solution, Value, Verdict
from games.base import Outcome
from games.mini_gungi import Gungi, GungiPosition, VariantConfig
from scripts import solve_nano
from scripts.nano_trials import fingerprint, involves_stacking, nano_trials
from scripts.solve_nano import PerfectEpisode, Report, load_solution_of, main, solve_to_file
from scripts.train import SETUPS, GameSetup, gungi_variant
from tests.solver_helpers import (
    FACING_MARSHALS,
    SMALL_NANO,
    ForcedWinSearch,
    forces_a_win,
    winner_of,
)


def setup_of(config: VariantConfig) -> GameSetup:
    """나노 군의 설정에서 게임만 작은 것으로 바꾼다."""
    return replace(SETUPS["nano-gungi"], game=Gungi(config), variant=gungi_variant(config))


def test_it_saves_a_solution_the_perfect_opponent_plays_from(tmp_path: Path) -> None:
    setup = setup_of(SMALL_NANO)
    path = tmp_path / "small.pickle"

    report = solve_to_file(setup, path, seeds=range(3))

    assert report.states == 1812
    assert report.start == Value(Verdict.DRAW, None)
    assert sum(report.counts.values()) == 1812
    assert report.max_dtm == 5
    assert report.move_limit == SMALL_NANO.move_limit
    assert [episode.seed for episode in report.episodes] == [0, 1, 2]
    assert all(episode.outcome is Outcome.DRAW for episode in report.episodes)
    assert report.holds
    assert len(load_solution_of(setup, path)) == 1812


def test_a_decided_start_is_expected_to_end_in_that_result(tmp_path: Path) -> None:
    # 선수가 두 수 만에 지는 변형이다.
    report = solve_to_file(setup_of(FACING_MARSHALS), tmp_path / "facing.pickle", seeds=range(2))

    assert report.start == Value(Verdict.LOSS, 2)
    assert report.episodes == (
        PerfectEpisode(0, Outcome.SECOND_WINS, 2),
        PerfectEpisode(1, Outcome.SECOND_WINS, 2),
    )
    assert report.holds


@pytest.mark.parametrize(
    ("start", "outcome", "holds"),
    [
        (Value(Verdict.DRAW, None), Outcome.DRAW, True),
        (Value(Verdict.DRAW, None), Outcome.FIRST_WINS, False),
        (Value(Verdict.WIN, 3), Outcome.FIRST_WINS, True),
        (Value(Verdict.WIN, 3), Outcome.DRAW, False),
        (Value(Verdict.LOSS, 2), Outcome.SECOND_WINS, True),
        (Value(Verdict.LOSS, 2), Outcome.FIRST_WINS, False),
    ],
)
def test_a_game_off_the_start_value_fails(start: Value, outcome: Outcome, holds: bool) -> None:
    report = Report(
        states=1,
        start=start,
        counts={},
        max_dtm=None,
        move_limit=200,
        solve_seconds=0.0,
        episodes=(PerfectEpisode(0, outcome, 3),),
    )
    assert report.holds is holds


def test_the_command_prints_the_figures(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(solve_nano.SETUPS, "nano-gungi", setup_of(SMALL_NANO))
    path = tmp_path / "nano.pickle"

    assert main(["--output", str(path)]) == 0

    summary = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert path.is_file()
    assert summary["holds"] is True
    assert set(summary["counts"]) == {"WIN", "DRAW", "LOSS"}
    assert [episode["seed"] for episode in summary["episodes"]] == list(range(10))


@pytest.mark.from_scratch
def test_the_nano_gungi_is_solved_over_the_documented_number_of_states(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """실제 나노 군의. 상태 수는 docs/nano-state-space.md 의 것이다."""
    path = tmp_path / "nano.pickle"

    status = main(["--output", str(path)])

    summary = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert status == 0, summary
    assert summary["states"] == 881_080
    solution = load_solution_of(SETUPS["nano-gungi"], path)
    check_long_results_by_search(solution)
    check_trial_positions(solution)


# ADR-0007 의 시험 판. 층·개수·시드나 풀이가 달라지면 바뀌고, 이전의 검증과 비교할 수 없게 된다.
TRIALS_FINGERPRINT = "cff3d8961bdf9d3ea0869220d041a0ad5fcc9601a72b25a99f7f5559c33b57c5"


def check_trial_positions(solution: Solution) -> None:
    game = solution.game
    assert isinstance(game, Gungi)

    trials = nano_trials(solution)

    assert Counter(trial.stratum for trial in trials) == {
        "win-13+/stack": 30,
        "win-13+/plain": 10,
        "win-9-11/stack": 30,
        "win-9-11/plain": 10,
        "win-5-7/stack": 15,
        "win-5-7/plain": 5,
        "draw/stack": 75,
        "draw/plain": 25,
    }
    for trial in trials:
        assert involves_stacking(game, trial.position) is trial.stratum.endswith("/stack")
        replies = {
            solution.value(game.apply_move(trial.position, move)).verdict
            for move in game.legal_moves(trial.position)
        }
        if trial.stratum.startswith("draw"):
            # 비기는 수와 지는 수가 함께 있다.
            assert trial.value.verdict is Verdict.DRAW
            assert Verdict.WIN in replies
        else:
            assert trial.value.verdict is Verdict.WIN
            assert trial.value.dtm is not None and trial.value.dtm >= 5
    assert fingerprint(game, trials) == TRIALS_FINGERPRINT


# 기억하지 않고 실제 규칙을 그대로 걷는 탐색은 이 DTM 까지만 몇 초 안에 끝난다.
LONGEST_PLAIN_SEARCH = 9
SAMPLES_PER_DTM = 3


def check_long_results_by_search(solution: Solution) -> None:
    """DTM 5~20 의 표본: 승패가 DTM 수 안에는 강제되고 한 수 적게는 강제되지 않는다."""
    game = solution.game
    assert isinstance(game, Gungi)
    by_dtm: dict[int, list[GungiPosition]] = {}
    for position in reachable_positions(game):
        if game.board_outcome(position) is None:
            dtm = solution.value(position).dtm
            if dtm is not None and dtm >= 5:
                by_dtm.setdefault(dtm, []).append(position)
    # 가장 긴 것까지 빠짐없이 뽑는다.
    assert sorted(by_dtm) == list(range(5, 21))

    search = ForcedWinSearch(game)
    rng = random.Random(0)
    for dtm, found in sorted(by_dtm.items()):
        for position in rng.sample(found, SAMPLES_PER_DTM):
            winner = winner_of(position.player, solution.value(position).verdict)
            where = f"DTM {dtm}\n{game.render(position)}"
            assert search.forces(position, winner, dtm), where
            assert not search.forces(position, winner, dtm - 1), where
            if dtm <= LONGEST_PLAIN_SEARCH:
                assert forces_a_win(game, position, winner, dtm), where
                assert not forces_a_win(game, position, winner, dtm - 1), where
