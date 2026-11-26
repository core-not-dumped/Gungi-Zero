"""규칙별 fixture 를 룰 엔진에 대어 본다. 판과 기대 정답은 `tests.gungi_fixtures` 가 갖는다."""

import pytest

from games.base import Outcome
from games.mini_gungi import MINI_GUNGI, Gungi, GungiPosition
from tests.gungi_fixtures import SECTIONS, Fixture

FIXTURES = [fixture for section in SECTIONS for fixture in section.fixtures]


def check(game: Gungi, position: GungiPosition, legal: str | None, outcome: Outcome | None) -> None:
    assert game.outcome(position) is outcome
    moves = [game.render_move(move) for move in game.legal_moves(position)]
    if outcome is not None:
        assert moves == []
    if legal is not None:
        expected = legal.split()
        assert len(expected) == len(set(expected)), "기대 합법수에 같은 수가 두 번 적혔다"
        assert set(moves) == set(expected)


@pytest.mark.parametrize("fixture", FIXTURES, ids=[fixture.name for fixture in FIXTURES])
def test_the_engine_agrees_with_the_fixture(fixture: Fixture) -> None:
    game = Gungi(MINI_GUNGI)
    start = game.parse(fixture.board)
    # 검토 문서에 실리는 그림이 텍스트 렌더 형식 그대로여야 한다.
    assert game.render(start) == fixture.board
    check(game, start, fixture.legal, fixture.outcome)

    for then in fixture.then:
        position = start
        for move in then.moves.split():
            position = game.apply_move(position, game.parse_move(move))
        assert game.render(position) == then.board
        check(game, position, then.legal, then.outcome)


def test_every_rule_has_a_fixture_and_the_names_are_unique() -> None:
    assert all(section.fixtures for section in SECTIONS)
    names = [fixture.name for fixture in FIXTURES]
    assert len(names) == len(set(names))
