"""틱택토 검증의 합성 루트: 2단계 완료 기준 (`.scratch/alphazero-verification/spec.md`)."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from ai.logger import MemoryLogger
from ai.resnet import ResNetConfig
from eval.match import Record
from scripts import verify_tictactoe
from scripts.train import SETUPS, GameSetup
from scripts.verify_tictactoe import Verification, main, verify


def barely_trained() -> GameSetup:
    """실제 설정에서 크기만 줄인다. 이만큼 학습해서는 완벽한 상대를 버티지 못한다."""
    setup = SETUPS["tictactoe"]
    training = replace(
        setup.training,
        self_play=replace(setup.training.self_play, simulations=4),
        episodes_per_generation=2,
        steps_per_generation=2,
        batch_size=4,
    )
    return replace(setup, network=ResNetConfig(blocks=1, channels=4), training=training)


@pytest.mark.parametrize(
    ("first", "second", "unbeaten"),
    [
        (Record(0, 100, 0), Record(0, 100, 0), True),
        (Record(1, 99, 0), Record(0, 100, 0), True),
        (Record(0, 99, 1), Record(0, 100, 0), False),
        (Record(0, 100, 0), Record(0, 99, 1), False),
    ],
)
def test_one_loss_in_either_seat_fails(first: Record, second: Record, unbeaten: bool) -> None:
    assert (
        Verification(
            first, second, agreement=0.5, distinct_as_first=9, distinct_as_second=9
        ).unbeaten
        is unbeaten
    )


def test_it_trains_measures_both_seats_and_records_them(tmp_path: Path) -> None:
    logger = MemoryLogger()

    result = verify(
        barely_trained(),
        logger=logger,
        generations=2,
        checkpoint_dir=tmp_path,
        seed=0,
        match_seeds=range(3),
    )

    for record in (result.as_first, result.as_second):
        assert record.wins + record.draws + record.losses == 3
    assert 0 <= result.agreement <= 1
    assert logger.configuration["variant"]["width"] == 3
    assert logger.configuration["evaluation"] == {
        "simulations": 50,
        "temperature": 0.0,
        "seeds": [0, 1, 2],
    }
    assert (tmp_path / "generation-0002.pt").is_file()
    # 학습의 세대 기록 뒤에, 마지막 세대에 잰 값이 붙는다.
    assert [step for step, _ in logger.records] == [1, 2, 2]
    measured = logger.records[-1][1]
    assert measured == {
        "match/first/wins": result.as_first.wins,
        "match/first/draws": result.as_first.draws,
        "match/first/losses": result.as_first.losses,
        "match/second/wins": result.as_second.wins,
        "match/second/draws": result.as_second.draws,
        "match/second/losses": result.as_second.losses,
        "diagnostic/value_agreement": result.agreement,
        "match/first/distinct": result.distinct_as_first,
        "match/second/distinct": result.distinct_as_second,
    }
    assert 1 <= result.distinct_as_first <= 3


@pytest.mark.parametrize(("unbeaten", "status"), [(True, 0), (False, 1)])
def test_the_command_reports_the_criterion_in_its_status(
    unbeaten: bool,
    status: int,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    losses = 0 if unbeaten else 1
    result = Verification(
        Record(0, 100 - losses, losses),
        Record(0, 100, 0),
        agreement=0.5,
        distinct_as_first=9,
        distinct_as_second=9,
    )
    asked: dict = {}

    def fake(setup: GameSetup, **options: object) -> Verification:
        asked.update(options, setup=setup)
        return result

    monkeypatch.setattr(verify_tictactoe, "verify", fake)

    assert main(["--checkpoint-dir", str(tmp_path), "--seed", "7"]) == status

    assert asked["setup"] is SETUPS["tictactoe"]
    assert asked["seed"] == 7
    assert asked["checkpoint_dir"] == tmp_path
    # 완료 기준은 시드 0~99 다.
    assert asked["match_seeds"] == range(100)
    summary = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert summary["unbeaten"] is unbeaten
    assert summary["as_first"]["losses"] == losses


def test_one_generation_of_the_real_settings_reaches_the_measurement(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """CI 의 스모크 테스트. 한 세대로는 기준을 바라지 않고, 200판과 진단까지 도는지만 본다."""
    status = main(["--generations", "1", "--checkpoint-dir", str(tmp_path)])

    summary = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert status == (0 if summary["unbeaten"] else 1)
    for seat in ("as_first", "as_second"):
        assert sum(summary[seat].values()) == 100


@pytest.mark.slow
def test_the_trained_ai_never_loses_to_the_perfect_opponent(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """2단계 완료 기준: 선수·후수 각 100판(시드 0~99) 무패."""
    status = main(["--checkpoint-dir", str(tmp_path)])

    assert status == 0, capsys.readouterr().out.splitlines()[-1]
