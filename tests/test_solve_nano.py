"""나노 군의 풀이의 합성 루트: 만든 파일, 낸 수치, 완벽한 상대끼리의 대국."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from eval.solver import Value, Verdict
from games.base import Outcome
from games.mini_gungi import Gungi, VariantConfig
from scripts import solve_nano
from scripts.solve_nano import PerfectEpisode, Report, load_solution_of, main, solve_to_file
from scripts.train import SETUPS, GameSetup, gungi_variant
from tests.solver_helpers import FACING_MARSHALS, SMALL_NANO


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


@pytest.mark.slow
def test_the_nano_gungi_is_solved_over_the_documented_number_of_states(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """실제 나노 군의. 상태 수는 docs/nano-state-space.md 의 것이다."""
    status = main(["--output", str(tmp_path / "nano.pickle")])

    summary = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert status == 0, summary
    assert summary["states"] == 881_080
