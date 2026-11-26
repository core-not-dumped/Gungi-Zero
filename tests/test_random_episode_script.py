"""랜덤 대국 스크립트. 한 판이 끝까지 돌아가는지 보는 눈이다."""

import re

import pytest

from scripts.random_episode import main


def test_the_script_plays_one_episode_and_reports_a_result(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["--seed", "4"]) == 0
    printed = capsys.readouterr().out
    assert printed.startswith("     a   b   c\n")
    assert printed.rstrip().splitlines()[-1].startswith("결과: ")


def test_the_script_logs_gungi_moves_in_the_move_notation(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["--game", "gungi-marshal-only", "--seed", "4"]) == 0
    printed = capsys.readouterr().out
    assert printed.startswith("     a       b       c       d       e\n")
    assert re.search(r"^ 1\. ▲ c1-[b-d][12]$", printed, re.MULTILINE)
    assert printed.rstrip().splitlines()[-1].startswith("결과: ")


def test_the_script_marks_the_capture_that_ends_a_gungi_episode(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # 시드 4 는 帥 를 잡아 끝나는 대국이다. 帥 만 있으므로 잡기는 마지막 수 하나뿐이다.
    main(["--game", "gungi-marshal-only", "--seed", "4"])
    moves = re.findall(r"^ *\d+\. [▲▽] (\S+)$", capsys.readouterr().out, re.MULTILINE)
    assert moves[-1].endswith("x")
    assert not any(move.endswith("x") for move in moves[:-1])


def test_the_script_is_reproducible_from_its_seed(
    capsys: pytest.CaptureFixture[str],
) -> None:
    main(["--seed", "4"])
    first = capsys.readouterr().out
    main(["--seed", "4"])
    assert capsys.readouterr().out == first
