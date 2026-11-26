"""군의 fixture 테스트. `games.base.Game` 으로만 게임을 보고 판은 텍스트 렌더로 읽는다."""

from dataclasses import replace

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from ai.random_play import play_random_episode
from games.base import Outcome
from games.mini_gungi import (
    MARSHAL_ONLY,
    Gungi,
    GungiPosition,
    PieceType,
    Placement,
    VariantConfig,
)

# 스펙 "텍스트 판 표기" 절의 그림 그대로다.
SPEC_BOARD = """\
     a       b       c       d       e
 5   .      ▽槍     ▽帥     ▽大      .
 4   .      ▽兵      .      ▽兵      .
 3   .       .       .       .       .
 2   .      ▲兵      .      ▲兵      .
 1   .      ▲大     ▲帥     ▲槍      .
차례: ▲   ply: 1
"""

# 帥 말고는 아직 움직일 수 없지만(06 번 티켓), 판을 놓고 읽고 쓰는 데는 쓸 수 있다.
FOUR_KINDS = VariantConfig(
    width=5,
    height=5,
    max_tier=2,
    setup=(
        Placement(PieceType.GENERAL, column=1, row=0),
        Placement(PieceType.MARSHAL, column=2, row=0),
        Placement(PieceType.LANCE, column=3, row=0),
        Placement(PieceType.PAWN, column=1, row=1),
        Placement(PieceType.PAWN, column=3, row=1),
    ),
    move_limit=200,
    repetition_limit=3,
)


def notation(game: Gungi, position: GungiPosition) -> set[str]:
    return {game.render_move(move) for move in game.legal_moves(position)}


def play(game: Gungi, position: GungiPosition, *moves: str) -> GungiPosition:
    for move in moves:
        position = game.apply_move(position, game.parse_move(move))
    return position


def test_the_initial_position_puts_the_second_player_on_the_rotated_setup() -> None:
    game = Gungi(FOUR_KINDS)
    assert game.render(game.initial_position()) == SPEC_BOARD.replace("ply: 1", "ply: 0")


def test_parse_reads_the_rendered_format_back() -> None:
    game = Gungi(FOUR_KINDS)
    position = game.parse(SPEC_BOARD)
    assert game.render(position) == SPEC_BOARD
    assert position.ply == 1


def test_a_tower_is_written_top_first_and_round_trips() -> None:
    game = Gungi(FOUR_KINDS)
    text = """\
     a       b       c       d       e
 5 ▽兵/▲大 ▽兵/▲槍  ▽帥      .       .
 4   .       .       .       .       .
 3   .       .    ▲兵/▽大    .       .
 2   .       .       .       .       .
 1   .       .      ▲帥      .       .
차례: ▽   ply: 7
"""
    position = game.parse(text)
    assert game.parse(game.render(position)) == position
    assert "▲兵/▽大" in game.render(position)
    # 맨 왼쪽에 탑이 붙어 있어도 행 번호·이웃 칸과 빈칸 하나는 떨어진다.
    assert " 5 ▽兵/▲大 ▽兵/▲槍" in game.render(position)


@pytest.mark.parametrize(
    "row_one",
    [
        " 1   .       .      ▲王      .       .",
        " 1   .       .   ▲兵/▲兵/▲兵   .       .",
        " 1  ▲帥      .      ▲帥      .       .",
        " 1   .       .      ▲帥      .",
    ],
)
def test_a_board_that_cannot_exist_is_refused(row_one: str) -> None:
    game = Gungi(MARSHAL_ONLY)
    with pytest.raises(ValueError):
        game.parse(f"""\
     a       b       c       d       e
 5   .       .      ▽帥      .       .
 4   .       .       .       .       .
 3   .       .       .       .       .
 2   .       .       .       .       .
{row_one}
차례: ▲   ply: 0
""")


def test_the_marshal_only_variant_starts_with_two_marshals_facing_each_other() -> None:
    game = Gungi(MARSHAL_ONLY)
    assert (
        game.render(game.initial_position())
        == """\
     a       b       c       d       e
 5   .       .      ▽帥      .       .
 4   .       .       .       .       .
 3   .       .       .       .       .
 2   .       .       .       .       .
 1   .       .      ▲帥      .       .
차례: ▲   ply: 0
"""
    )


def test_a_marshal_steps_one_square_in_all_eight_directions() -> None:
    game = Gungi(MARSHAL_ONLY)
    position = game.parse("""\
     a       b       c       d       e
 5   .       .       .       .      ▽帥
 4   .       .       .       .       .
 3   .       .      ▲帥      .       .
 2   .       .       .       .       .
 1   .       .       .       .       .
차례: ▲   ply: 0
""")
    assert notation(game, position) == {
        "c3-b2",
        "c3-c2",
        "c3-d2",
        "c3-b3",
        "c3-d3",
        "c3-b4",
        "c3-c4",
        "c3-d4",
    }


def test_a_marshal_on_the_edge_does_not_wrap_around_the_board() -> None:
    game = Gungi(MARSHAL_ONLY)
    position = game.parse("""\
     a       b       c       d       e
 5   .       .       .       .       .
 4   .       .       .       .       .
 3   .       .       .       .      ▽帥
 2   .       .       .       .       .
 1  ▲帥      .       .       .       .
차례: ▽   ply: 1
""")
    assert notation(game, position) == {"e3-d2", "e3-e2", "e3-d3", "e3-d4", "e3-e4"}


def test_moving_leaves_the_square_empty_and_passes_the_turn() -> None:
    game = Gungi(MARSHAL_ONLY)
    after = play(game, game.initial_position(), "c1-b2")
    assert (
        game.render(after)
        == """\
     a       b       c       d       e
 5   .       .      ▽帥      .       .
 4   .       .       .       .       .
 3   .       .       .       .       .
 2   .      ▲帥      .       .       .
 1   .       .       .       .       .
차례: ▽   ply: 1
"""
    )


@pytest.mark.parametrize("move", ["c1-c3", "c1-c1", "c5-c4", "a1-a2"])
def test_an_illegal_move_is_refused(move: str) -> None:
    game = Gungi(MARSHAL_ONLY)
    with pytest.raises(ValueError):
        play(game, game.initial_position(), move)


@pytest.mark.parametrize(
    ("mover", "move", "outcome"),
    [("▲", "c2-c3", Outcome.FIRST_WINS), ("▽", "c3-c2", Outcome.SECOND_WINS)],
)
def test_capturing_the_enemy_marshal_by_moving_onto_it_wins(
    mover: str, move: str, outcome: Outcome
) -> None:
    game = Gungi(MARSHAL_ONLY)
    position = game.parse(f"""\
     a       b       c       d       e
 5   .       .       .       .       .
 4   .       .       .       .       .
 3   .       .      ▽帥      .       .
 2   .       .      ▲帥      .       .
 1   .       .       .       .       .
차례: {mover}   ply: 4
""")
    assert game.outcome(position) is None
    after = play(game, position, move)
    assert game.outcome(after) is outcome
    assert game.legal_moves(after) == ()
    assert game.render(after).count("帥") == 1


def test_reaching_the_move_limit_is_a_draw() -> None:
    game = Gungi(replace(MARSHAL_ONLY, move_limit=4))
    position = play(game, game.initial_position(), "c1-b1", "c5-b5", "b1-c1")
    assert game.outcome(position) is None
    position = play(game, position, "b5-c5")
    assert game.outcome(position) is Outcome.DRAW
    assert game.legal_moves(position) == ()
    with pytest.raises(ValueError):
        play(game, position, "c1-b1")


def test_a_capture_on_the_last_allowed_move_still_wins() -> None:
    game = Gungi(MARSHAL_ONLY)
    position = game.parse("""\
     a       b       c       d       e
 5   .       .       .       .       .
 4   .       .       .       .       .
 3   .       .      ▽帥      .       .
 2   .       .      ▲帥      .       .
 1   .       .       .       .       .
차례: ▽   ply: 199
""")
    assert game.outcome(play(game, position, "c3-c2")) is Outcome.SECOND_WINS


@pytest.mark.parametrize("seed", range(50))
def test_a_random_episode_ends(seed: int) -> None:
    game = Gungi(MARSHAL_ONLY)
    episode = play_random_episode(game, seed)
    assert len(episode.moves) <= MARSHAL_ONLY.move_limit
    if episode.outcome is Outcome.DRAW:
        assert len(episode.moves) == MARSHAL_ONLY.move_limit


def test_moves_are_written_from_square_to_square() -> None:
    game = Gungi(MARSHAL_ONLY)
    assert game.render_move(game.parse_move("b1-c3")) == "b1-c3"
    for text in ("b1c3", "f1-c3", "b6-c3", "b0-c3", "b1-c3x", "b01-c3", "b1-c3-d4", ""):
        with pytest.raises(ValueError):
            game.parse_move(text)


def test_the_action_space_is_source_square_times_target_square() -> None:
    game = Gungi(MARSHAL_ONLY)
    assert game.action_space_size == 625
    # 칸 번호는 a1=0, b1=1, ... e5=24 이고 인덱스는 출발 * 25 + 도착이다. b1-c3 = 1*25+12.
    assert game.action_index(game.parse_move("b1-c3")) == 37
    assert game.render_move(game.move_from_action_index(624)) == "e5-e5"
    for index in (-1, 625):
        with pytest.raises(ValueError):
            game.move_from_action_index(index)


def test_the_canonical_form_rotates_the_board_and_swaps_the_colours() -> None:
    game = Gungi(FOUR_KINDS)
    position = game.parse("""\
     a       b       c       d       e
 5   .       .      ▽帥      .       .
 4   .       .       .       .       .
 3   .       .       .       .       .
 2   .    ▲兵/▽大    .       .       .
 1  ▲槍      .      ▲帥      .       .
차례: ▽   ply: 3
""")
    assert (
        game.render(game.canonical(position))
        == """\
     a       b       c       d       e
 5   .       .      ▽帥      .      ▽槍
 4   .       .       .    ▽兵/▲大    .
 3   .       .       .       .       .
 2   .       .       .       .       .
 1   .       .      ▲帥      .       .
차례: ▲   ply: 3
"""
    )
    # b2-a3 을 180° 돌리면 d4-e3 이다.
    index = game.action_index(game.parse_move("b2-a3"))
    carried = game.to_canonical_action_index(position, index)
    assert game.render_move(game.move_from_action_index(carried)) == "d4-e3"


def test_the_tensor_has_one_plane_per_owner_kind_and_tier() -> None:
    game = Gungi(FOUR_KINDS)
    position = game.parse("""\
     a       b       c       d       e
 5   .       .      ▽帥      .       .
 4   .       .       .       .       .
 3   .       .       .       .       .
 2   .    ▲兵/▽大    .       .       .
 1  ▲槍      .      ▲帥      .       .
차례: ▽   ply: 3
""")
    tensor = game.encode(position)
    assert game.tensor_shape == tensor.shape == (18, 5, 5)
    # 평면 = (내 말 0 / 상대 말 1) * 8 + 말 종류(帥 大 槍 兵) * 2 + (단 - 1).
    # 둘 차례가 ▽ 이므로 판은 180° 돌아가고 좌표는 (평면, 행, 열) 이다.
    ones = {
        (int(plane), int(row), int(column)) for plane, row, column in np.argwhere(tensor[:16] == 1)
    }
    assert ones == {
        (0, 0, 2),  # 내 帥 1단
        (2, 3, 3),  # 내 大 1단 (탑의 아래)
        (8, 4, 2),  # 상대 帥 1단
        (12, 4, 4),  # 상대 槍 1단
        (15, 3, 3),  # 상대 兵 2단 (탑의 위)
    }
    assert tensor[:16].sum() == 5


@given(picks=st.lists(st.integers(min_value=0, max_value=2**20), max_size=60))
def test_the_planes_add_up_to_the_pieces_on_the_board(picks: list[int]) -> None:
    game = Gungi(MARSHAL_ONLY)
    position = game.initial_position()
    for pick in picks:
        moves = game.legal_moves(position)
        if not moves:
            break
        position = game.apply_move(position, moves[pick % len(moves)])
        board = game.render(position).split("차례:")[0]
        assert game.encode(position)[:16].sum() == board.count("▲") + board.count("▽")


@pytest.mark.parametrize(
    ("change", "value"),
    [
        ("width", 0),
        ("move_limit", 0),
        ("repetition_limit", 1),
        ("setup", ()),
        ("setup", (Placement(PieceType.MARSHAL, column=5, row=0),)),
        # 가운데 칸은 180° 회전하면 자기 자리라 상대 帥 와 겹친다.
        ("setup", (Placement(PieceType.MARSHAL, column=2, row=2),)),
    ],
)
def test_a_broken_variant_is_refused(change: str, value: object) -> None:
    with pytest.raises(ValueError):
        replace(MARSHAL_ONLY, **{change: value})


def test_the_board_size_comes_from_the_variant() -> None:
    game = Gungi(
        VariantConfig(
            width=3,
            height=4,
            max_tier=1,
            setup=(Placement(PieceType.MARSHAL, column=0, row=0),),
            move_limit=10,
            repetition_limit=3,
        )
    )
    assert game.action_space_size == 144
    assert game.tensor_shape == (18, 4, 3)
    assert notation(game, game.initial_position()) == {"a1-b1", "a1-a2", "a1-b2"}


def test_a_variant_with_more_tiers_than_the_planes_hold_is_refused() -> None:
    with pytest.raises(ValueError):
        Gungi(replace(MARSHAL_ONLY, max_tier=3))


@pytest.mark.parametrize(
    "row_two",
    [
        # 06 번 티켓 전에는 帥 말고는 움직일 줄 모른다. 07 번 전에는 탑이 있으면 둘 수 없다.
        " 2   .      ▲兵      .       .       .",
        " 2   .    ▲兵/▽大    .       .       .",
        " 2   .    ▽兵/▽大    .       .       .",
    ],
)
def test_rules_that_do_not_exist_yet_fail_loudly(row_two: str) -> None:
    game = Gungi(FOUR_KINDS)
    position = game.parse(f"""\
     a       b       c       d       e
 5   .       .      ▽帥      .       .
 4   .       .       .       .       .
 3   .       .       .       .       .
{row_two}
 1   .       .      ▲帥      .       .
차례: ▲   ply: 0
""")
    with pytest.raises(NotImplementedError):
        game.legal_moves(position)
