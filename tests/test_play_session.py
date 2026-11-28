"""칸 클릭을 수로 바꾸는 부분. 창 없이 돈다."""

import pytest

from games.base import Outcome, Player
from games.mini_gungi import EndReason, Gungi, GungiMove, GungiPosition
from games.nano_gungi import NANO_GUNGI
from ui.session import PlaySession

GAME = Gungi(NANO_GUNGI)

# ▲大 앞에 ▲兵 이 있다.
OWN_PIECE_AHEAD = """\
     a       b       c
 4   .      ▽帥     ▽大
 3   .       .       .
 2  ▲兵      .       .
 1  ▲大     ▲帥      .
차례: ▲   ply: 0
"""

# ▲兵 c2 앞에 ▽兵 이 있다. c2-c3 은 잡기와 쌓기가 모두 합법이다.
FACE_TO_FACE = """\
     a       b       c
 4   .      ▽帥     ▽大
 3   .       .      ▽兵
 2   .       .      ▲兵
 1  ▲大     ▲帥      .
차례: ▲   ply: 0
"""

# ▲大 이 a3 에서 b4 의 ▽帥 를 잡을 수 있다.
MARSHAL_IN_REACH = """\
     a       b       c
 4   .      ▽帥      .
 3  ▲大      .       .
 2   .       .       .
 1   .      ▲帥      .
차례: ▲   ply: 4
"""


def square(name: str) -> int:
    return GAME.parse_move(f"{name}-{name}").source


def session_at(text: str) -> PlaySession:
    return PlaySession(GAME, position=GAME.parse(text))


def click(session: PlaySession, *names: str) -> None:
    for name in names:
        session.click(square(name))


def targets(session: PlaySession) -> set[str]:
    return {GAME.render_move(GungiMove(target, target))[:2] for target in session.targets}


def test_a_new_session_starts_at_the_initial_position() -> None:
    session = PlaySession(GAME)
    assert session.position == GAME.initial_position()
    assert session.selected is None
    assert session.log == []
    assert session.outcome is None


def test_clicking_a_movable_piece_shows_where_it_can_go() -> None:
    session = PlaySession(GAME)
    click(session, "b1")
    assert session.selected == square("b1")
    assert targets(session) == {"a1", "a2", "b2"}


def test_clicking_a_target_plays_the_move_and_logs_it() -> None:
    session = PlaySession(GAME)
    click(session, "b1", "b2")
    assert session.position == GAME.apply_move(GAME.initial_position(), GAME.parse_move("b1-b2"))
    assert session.log == ["1. ▲ b1-b2"]
    assert session.selected is None
    assert session.targets == frozenset()


@pytest.mark.parametrize("elsewhere", ["a4", "c4", "b1"])
def test_clicking_elsewhere_releases_the_selection(elsewhere: str) -> None:
    # a4 는 상대 말, c4 는 갈 수 없는 빈칸, b1 은 고른 말 자신이다.
    session = PlaySession(GAME)
    click(session, "b1", elsewhere)
    assert session.selected is None
    assert session.position == GAME.initial_position()


def test_clicking_outside_the_board_releases_the_selection() -> None:
    session = PlaySession(GAME)
    click(session, "b1")
    session.click(None)
    assert session.selected is None


def test_clicking_another_movable_piece_selects_it_instead() -> None:
    session = PlaySession(GAME)
    click(session, "b1", "c2")
    assert session.selected == square("c2")
    assert targets(session) == {"a2", "b2", "b3", "c1", "c3", "c4"}


@pytest.mark.parametrize("name", ["b3", "a4"])
def test_a_square_without_a_movable_piece_cannot_be_selected(name: str) -> None:
    session = PlaySession(GAME)
    click(session, name)
    assert session.selected is None


@pytest.mark.parametrize(("stack", "notation"), [(False, "c2-c3x"), (True, "c2-c3^")])
def test_a_square_open_to_capture_and_stack_asks_which(stack: bool, notation: str) -> None:
    session = session_at(FACE_TO_FACE)
    click(session, "c2", "c3")
    assert session.choosing == square("c3")
    assert session.log == []

    session.choose(stack=stack)
    assert session.log == [f"1. ▲ {notation}"]
    assert session.choosing is None
    assert session.selected is None


@pytest.mark.parametrize("elsewhere", ["c3", "a1", None])
def test_clicking_the_board_while_asked_to_choose_cancels(elsewhere: str | None) -> None:
    session = session_at(FACE_TO_FACE)
    click(session, "c2", "c3")
    session.click(None if elsewhere is None else square(elsewhere))
    assert session.choosing is None
    assert session.selected is None
    assert session.position == GAME.parse(FACE_TO_FACE)


def test_choosing_without_being_asked_does_nothing() -> None:
    session = session_at(FACE_TO_FACE)
    session.choose(stack=False)
    assert session.position == GAME.parse(FACE_TO_FACE)


def test_stacking_on_an_own_piece_needs_no_choice() -> None:
    session = session_at(OWN_PIECE_AHEAD)
    click(session, "a1", "a2")
    assert session.log == ["1. ▲ a1-a2^"]
    assert session.choosing is None


def test_a_move_that_is_not_legal_is_not_played() -> None:
    session = session_at(OWN_PIECE_AHEAD)
    click(session, "b1", "a2")  # 帥 는 어디에도 못 올라간다.
    assert session.log == []
    assert session.position == GAME.parse(OWN_PIECE_AHEAD)


def test_the_game_ends_with_a_result_and_ignores_further_clicks() -> None:
    session = session_at(MARSHAL_IN_REACH)
    click(session, "a3", "b4")
    if session.choosing is not None:
        session.choose(stack=False)
    assert session.outcome is Outcome.FIRST_WINS
    assert session.end_reason is EndReason.MARSHAL_CAPTURE
    assert session.log == ["5. ▲ a3-b4x"]

    click(session, "b1")
    assert session.selected is None


def first_legal(position: GungiPosition) -> GungiMove:
    return GAME.legal_moves(position)[0]


def test_the_opponent_moves_only_on_its_own_turn() -> None:
    session = PlaySession(GAME, opponent=first_legal, opponent_player=Player.SECOND)
    assert not session.opponent_to_move
    session.play_opponent()
    assert session.log == []

    click(session, "b1", "b2")
    assert session.opponent_to_move
    after_human = session.position
    session.play_opponent()
    assert session.position == GAME.apply_move(after_human, first_legal(after_human))
    assert len(session.log) == 2
    assert not session.opponent_to_move


def test_the_human_cannot_move_for_the_opponent() -> None:
    session = PlaySession(GAME, opponent=first_legal, opponent_player=Player.FIRST)
    click(session, "a1")
    assert session.selected is None
    assert session.opponent_to_move


def test_the_opponent_does_not_move_after_the_game_ends() -> None:
    session = PlaySession(
        GAME,
        opponent=first_legal,
        opponent_player=Player.SECOND,
        position=GAME.parse(MARSHAL_IN_REACH),
    )
    click(session, "a3", "b4")
    if session.choosing is not None:
        session.choose(stack=False)
    assert not session.opponent_to_move
