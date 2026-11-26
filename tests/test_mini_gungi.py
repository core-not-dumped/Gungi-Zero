"""군의 fixture 테스트. `games.base.Game` 으로만 게임을 보고 판은 텍스트 렌더로 읽는다."""

from collections import Counter
from dataclasses import replace

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from ai.random_play import play_random_episode
from games.base import Outcome
from games.mini_gungi import (
    MARSHAL_ONLY,
    SINGLE_TIER,
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

# 판을 놓고 읽고 쓰는 데 쓴다. 쌓기가 아직 없어(07 번 티켓) 수를 낼 때는 `SINGLE_TIER` 를 쓴다.
FOUR_KINDS = replace(SINGLE_TIER, max_tier=2)


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


@pytest.mark.parametrize(
    ("row_five", "row_one"),
    [
        # 帥 가 하나도 없다.
        (" 5   .       .       .       .       .", " 1   .       .       .       .       ."),
        # 방금 둔 ▽ 의 帥 가 없다. 자기 수로 자기 帥 를 잃을 수는 없다.
        (" 5   .       .       .       .       .", " 1   .       .      ▲帥      .       ."),
        # 帥 위에는 아군도 적군도 올라갈 수 없다.
        (" 5   .       .      ▽帥      .       .", " 1   .       .    ▽兵/▲帥    .       ."),
        (" 5   .       .    ▽兵/▽帥    .       .", " 1   .       .      ▲帥      .       ."),
    ],
)
def test_a_board_that_no_game_can_reach_is_refused(row_five: str, row_one: str) -> None:
    game = Gungi(FOUR_KINDS)
    with pytest.raises(ValueError):
        game.parse(f"""\
     a       b       c       d       e
{row_five}
 4   .       .       .       .       .
 3   .       .       .       .       .
 2   .       .       .       .       .
{row_one}
차례: ▲   ply: 4
""")


def test_a_marshal_on_top_of_a_tower_and_a_finished_board_are_accepted() -> None:
    game = Gungi(FOUR_KINDS)
    # 둘 차례인 ▲ 의 帥 가 잡힌 판은 끝난 대국의 정상적인 모습이다.
    position = game.parse("""\
     a       b       c       d       e
 5   .       .    ▽帥/▲兵    .       .
 4   .       .       .       .       .
 3   .       .       .       .       .
 2   .       .       .       .       .
 1   .       .       .       .       .
차례: ▲   ply: 4
""")
    assert game.outcome(position) is Outcome.SECOND_WINS


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


# 두 帥 가 한 칸 갔다가 돌아온다. 네 수마다 같은 판, 같은 차례가 된다.
ROUND_TRIP = ("c1-b1", "c5-b5", "b1-c1", "b5-c5")


def repetition_plane(game: Gungi, position: GungiPosition) -> float:
    plane = game.encode(position)[16]
    assert np.all(plane == plane[0, 0])
    return float(plane[0, 0])


def test_the_third_appearance_of_a_board_is_a_draw() -> None:
    game = Gungi(MARSHAL_ONLY)
    second = play(game, game.initial_position(), *ROUND_TRIP)
    assert game.outcome(second) is None
    almost = play(game, second, *ROUND_TRIP[:3])
    assert game.outcome(almost) is None
    third = play(game, almost, ROUND_TRIP[3])
    assert game.outcome(third) is Outcome.DRAW
    assert game.legal_moves(third) == ()
    with pytest.raises(ValueError):
        play(game, third, "c1-b1")


def test_the_repetition_limit_comes_from_the_variant() -> None:
    game = Gungi(replace(MARSHAL_ONLY, repetition_limit=2))
    position = play(game, game.initial_position(), *ROUND_TRIP)
    assert game.outcome(position) is Outcome.DRAW


def test_the_same_board_with_the_other_side_to_move_is_not_a_repetition() -> None:
    game = Gungi(replace(MARSHAL_ONLY, repetition_limit=2))
    initial = game.initial_position()
    # ▲ 는 세 수, ▽ 는 두 수에 제자리로 온다. 판은 처음과 같지만 차례가 ▽ 다.
    position = play(game, initial, "c1-b1", "c5-b5", "b1-b2", "b5-c5", "b2-c1")
    assert game.render(position).split("차례:")[0] == game.render(initial).split("차례:")[0]
    assert game.board_key(position) != game.board_key(initial)
    assert game.outcome(position) is None


def test_a_parsed_position_has_no_repetitions() -> None:
    game = Gungi(MARSHAL_ONLY)
    played = play(game, game.initial_position(), *ROUND_TRIP, *ROUND_TRIP[:3])
    parsed = game.parse(game.render(played))
    assert repetition_plane(game, parsed) == 0.0
    # 같은 수를 두어도 파싱한 쪽은 앞선 판을 모르므로 끝나지 않는다.
    assert game.outcome(play(game, played, ROUND_TRIP[3])) is Outcome.DRAW
    assert game.outcome(play(game, parsed, ROUND_TRIP[3])) is None


def test_different_routes_to_the_same_board_share_the_key_but_not_the_position() -> None:
    game = Gungi(MARSHAL_ONLY)
    one = play(game, game.initial_position(), "c1-b1", "c5-b5", "b1-b2", "b5-b4")
    other = play(game, game.initial_position(), "c1-c2", "c5-c4", "c2-b2", "c4-b4")
    assert game.render(one) == game.render(other)
    assert game.board_key(one) == game.board_key(other)
    assert one != other
    assert game.canonical(one) != game.canonical(other)


def test_the_same_route_makes_equal_positions() -> None:
    game = Gungi(MARSHAL_ONLY)
    one = play(game, game.initial_position(), *ROUND_TRIP)
    other = play(game, game.initial_position(), *ROUND_TRIP)
    assert one == other
    assert hash(one) == hash(other)


def test_a_path_far_longer_than_a_game_can_be_compared_hashed_and_rotated() -> None:
    game = Gungi(replace(MARSHAL_ONLY, move_limit=2000, repetition_limit=1000))
    moves = ROUND_TRIP * 400
    one = play(game, game.initial_position(), *moves)
    other = play(game, game.initial_position(), *moves)
    assert one == other
    assert hash(one) == hash(other)
    after = play(game, one, "c1-b1")
    assert game.canonical(after) == game.canonical(play(game, other, "c1-b1"))


@given(picks=st.lists(st.integers(min_value=0, max_value=2**20), max_size=40))
def test_the_board_key_ignores_the_path_but_the_position_does_not(picks: list[int]) -> None:
    game = Gungi(MARSHAL_ONLY)
    position = game.initial_position()
    for pick in picks:
        moves = game.legal_moves(position)
        if not moves:
            break
        position = game.apply_move(position, moves[pick % len(moves)])
        # 판·차례·ply 는 같고 경로만 비어 있는 국면이다.
        pathless = game.parse(game.render(position))
        assert game.board_key(pathless) == game.board_key(position)
        assert pathless != position


@pytest.mark.parametrize(
    ("turn", "rows", "outcome"),
    [
        # a·b 열이 ▲ 로 꽉 차 있다. 兵 은 위아래로만 가고, 帥 의 이웃도 전부 아군이다.
        (
            "▲",
            (
                " 5  ▲兵     ▲兵      .       .      ▽帥",
                " 4  ▲兵     ▲兵      .       .       .",
                " 3  ▲兵     ▲兵      .       .       .",
                " 2  ▲兵     ▲兵      .       .       .",
                " 1  ▲帥     ▲兵      .       .       .",
            ),
            Outcome.SECOND_WINS,
        ),
        # 같은 판을 180° 돌린 것.
        (
            "▽",
            (
                " 5   .       .       .      ▽兵     ▽帥",
                " 4   .       .       .      ▽兵     ▽兵",
                " 3   .       .       .      ▽兵     ▽兵",
                " 2   .       .       .      ▽兵     ▽兵",
                " 1  ▲帥      .       .      ▽兵     ▽兵",
            ),
            Outcome.FIRST_WINS,
        ),
    ],
    ids=["first", "second"],
)
# 수 제한에 닿은 국면이어도 둘 수가 없으면 패배다.
@pytest.mark.parametrize("ply", [0, 200])
def test_a_player_without_a_legal_move_loses(
    turn: str, rows: tuple[str, ...], outcome: Outcome, ply: int
) -> None:
    game = Gungi(SINGLE_TIER)
    board = "\n".join(rows)
    position = game.parse(f"""\
     a       b       c       d       e
{board}
차례: {turn}   ply: {ply}
""")
    assert game.legal_moves(position) == ()
    assert game.outcome(position) is outcome


def test_the_tensor_counts_earlier_appearances_of_the_board() -> None:
    game = Gungi(MARSHAL_ONLY)
    position = game.initial_position()
    assert repetition_plane(game, position) == 0.0
    once = play(game, position, "c1-b1")
    assert repetition_plane(game, once) == 0.0
    # ▽ 차례인 국면도 같게 센다 — 정규형으로 돌려도 경로가 따라온다.
    again = play(game, once, "c5-b5", "b1-c1", "b5-c5", "c1-b1")
    assert repetition_plane(game, again) == 0.5
    assert repetition_plane(game, game.canonical(again)) == 0.5
    third = play(game, position, *ROUND_TRIP, *ROUND_TRIP)
    assert repetition_plane(game, third) == 1.0


def test_the_tensor_shows_how_far_the_game_has_gone() -> None:
    game = Gungi(MARSHAL_ONLY)
    position = game.initial_position()
    assert np.all(game.encode(position)[17] == 0.0)
    assert np.all(game.encode(play(game, position, "c1-b1"))[17] == np.float32(1 / 200))
    late = game.parse(game.render(position).replace("ply: 0", "ply: 150"))
    assert np.all(game.encode(late)[17] == 0.75)


@pytest.mark.parametrize("seed", range(50))
def test_a_random_episode_ends(seed: int) -> None:
    game = Gungi(MARSHAL_ONLY)
    episode = play_random_episode(game, seed)
    assert len(episode.moves) <= MARSHAL_ONLY.move_limit
    if episode.outcome is Outcome.DRAW and len(episode.moves) < MARSHAL_ONLY.move_limit:
        position = game.initial_position()
        keys = [game.board_key(position)]
        for move in episode.moves:
            position = game.apply_move(position, move)
            keys.append(game.board_key(position))
        assert keys.count(keys[-1]) == MARSHAL_ONLY.repetition_limit


def test_moves_are_written_from_square_to_square() -> None:
    game = Gungi(MARSHAL_ONLY)
    assert game.render_move(game.parse_move("b1-c3")) == "b1-c3"
    for text in ("b1c3", "f1-c3", "b6-c3", "b0-c3", "b1-c3x", "b01-c3", "b1-c3-d4", ""):
        with pytest.raises(ValueError):
            game.parse_move(text)


def test_the_log_notation_marks_only_captures() -> None:
    game = Gungi(MARSHAL_ONLY)
    position = game.parse("""\
     a       b       c       d       e
 5   .       .       .       .       .
 4   .       .       .       .       .
 3   .       .      ▽帥      .       .
 2   .       .      ▲帥      .       .
 1   .       .       .       .       .
차례: ▲   ply: 4
""")
    capture = game.parse_move("c2-c3")
    step = game.parse_move("c2-b2")
    assert game.render_log_move(position, capture) == "c2-c3x"
    assert game.render_log_move(position, step) == "c2-b2"
    # x 는 로그에만 붙는다. 수 표기 자체는 그대로다.
    assert game.render_move(capture) == "c2-c3"


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
        # 07 번 티켓 전에는 탑이 있거나 아군 위에 쌓을 수 있는 판에서 수를 낼 수 없다.
        " 2   .    ▲兵/▽大    .       .       .",
        " 2   .    ▽兵/▽大    .       .       .",
        " 2  ▲兵     ▲大      .       .       .",
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


# ── 말 4종의 1단 이동표와 막힘 ──
# 帥 의 이동표 fixture 는 위의 `test_a_marshal_steps_one_square_in_all_eight_directions` 다.


def test_a_general_slides_in_straight_lines_and_steps_one_diagonally() -> None:
    game = Gungi(SINGLE_TIER)
    position = game.parse("""\
     a       b       c       d       e
 5   .       .       .       .      ▽帥
 4   .       .       .       .       .
 3   .       .      ▲大      .       .
 2   .       .       .       .       .
 1  ▲帥      .       .       .       .
차례: ▲   ply: 0
""")
    assert notation(game, position) == {
        # 大 의 직선은 끝까지.
        *("c3-c4", "c3-c5", "c3-c2", "c3-c1", "c3-b3", "c3-a3", "c3-d3", "c3-e3"),
        # 大 의 대각은 한 칸.
        *("c3-b4", "c3-d4", "c3-b2", "c3-d2"),
        # 帥
        *("a1-a2", "a1-b1", "a1-b2"),
    }


@pytest.mark.parametrize(
    ("board", "expected"),
    [
        (
            """\
     a       b       c       d       e
 5   .       .       .       .      ▽帥
 4   .       .       .       .       .
 3   .       .       .       .       .
 2   .       .      ▲槍      .       .
 1  ▲帥      .       .       .       .
차례: ▲   ply: 0
""",
            {
                # 槍 은 전방 둘, 전방대각 하나, 후방 하나. 측면·후방대각은 없다.
                *("c2-c3", "c2-c4", "c2-b3", "c2-d3", "c2-c1"),
                # 帥
                *("a1-a2", "a1-b1", "a1-b2"),
            },
        ),
        # ▽ 의 앞은 1 행 쪽이다.
        (
            """\
     a       b       c       d       e
 5   .       .       .       .      ▽帥
 4   .       .      ▽槍      .       .
 3   .       .       .       .       .
 2   .       .       .       .       .
 1  ▲帥      .       .       .       .
차례: ▽   ply: 1
""",
            {
                *("c4-c3", "c4-c2", "c4-d3", "c4-b3", "c4-c5"),
                *("e5-e4", "e5-d5", "e5-d4"),
            },
        ),
    ],
    ids=["first", "second"],
)
def test_a_lance_moves_mostly_forward(board: str, expected: set[str]) -> None:
    game = Gungi(SINGLE_TIER)
    assert notation(game, game.parse(board)) == expected


@pytest.mark.parametrize(
    ("turn", "expected"),
    [
        ("▲", {"c2-c3", "c2-c1", "a1-a2", "a1-b1", "a1-b2"}),
        ("▽", {"c4-c3", "c4-c5", "e5-e4", "e5-d5", "e5-d4"}),
    ],
    ids=["first", "second"],
)
def test_a_pawn_steps_one_square_forward_or_backward(turn: str, expected: set[str]) -> None:
    game = Gungi(SINGLE_TIER)
    position = game.parse(f"""\
     a       b       c       d       e
 5   .       .       .       .      ▽帥
 4   .       .      ▽兵      .       .
 3   .       .       .       .       .
 2   .       .      ▲兵      .       .
 1  ▲帥      .       .       .       .
차례: {turn}   ply: 0
""")
    assert notation(game, position) == expected


def test_an_occupied_square_blocks_the_way_and_only_an_enemy_can_be_taken() -> None:
    game = Gungi(SINGLE_TIER)
    position = game.parse("""\
     a       b       c       d       e
 5   .       .       .       .      ▽帥
 4   .       .      ▽兵      .       .
 3   .      ▲兵     ▲大     ▽槍      .
 2   .      ▲兵      .      ▽兵      .
 1  ▲帥      .      ▲兵      .       .
차례: ▲   ply: 0
""")
    assert notation(game, position) == {
        # 大 은 위·오른쪽의 적을 잡고 거기서 멈춘다 (c5, e3 에는 못 간다).
        *("c3-c4", "c3-d3"),
        # 아래는 아군 c1 앞까지다. 왼쪽은 바로 옆이 아군이라 못 간다 (a3 도 못 간다).
        "c3-c2",
        # 대각. b2 는 아군이다.
        *("c3-b4", "c3-d4", "c3-d2"),
        # 兵 b3, b2, c1
        *("b3-b4", "b2-b1", "c1-c2"),
        # 帥
        *("a1-a2", "a1-b1"),
    }


@pytest.mark.parametrize(
    ("board", "expected"),
    [
        (
            """\
     a       b       c       d       e
 5   .       .       .       .      ▽帥
 4   .       .       .       .       .
 3   .       .       .       .       .
 2   .       .      ▲槍      .       .
 1  ▲帥      .       .       .       .
차례: ▲   ply: 0
""",
            {
                # 槍 전방
                *(
                    "c2-c3",
                    "c2-c4",
                ),
                # 槍 전방대각과 후방
                *("c2-b3", "c2-d3", "c2-c1"),
                # 帥
                *("a1-a2", "a1-b1", "a1-b2"),
            },
        ),
        # 바로 앞의 아군은 뛰어넘지 못한다.
        (
            """\
     a       b       c       d       e
 5   .       .       .       .      ▽帥
 4   .       .       .       .       .
 3   .       .      ▲兵      .       .
 2   .       .      ▲槍      .       .
 1  ▲帥      .       .       .       .
차례: ▲   ply: 0
""",
            {
                # 槍 전방대각과 후방
                *("c2-b3", "c2-d3", "c2-c1"),
                # 兵
                "c3-c4",
                # 帥
                *("a1-a2", "a1-b1", "a1-b2"),
            },
        ),
        # 바로 앞의 적은 잡고 거기서 멈춘다.
        (
            """\
     a       b       c       d       e
 5   .       .       .       .      ▽帥
 4   .       .       .       .       .
 3   .       .      ▽兵      .       .
 2   .       .      ▲槍      .       .
 1  ▲帥      .       .       .       .
차례: ▲   ply: 0
""",
            {
                # 槍 전방
                "c2-c3",
                # 槍 전방대각과 후방
                *("c2-b3", "c2-d3", "c2-c1"),
                # 帥
                *("a1-a2", "a1-b1", "a1-b2"),
            },
        ),
        # 두 칸 앞의 아군 앞까지만 간다.
        (
            """\
     a       b       c       d       e
 5   .       .       .       .      ▽帥
 4   .       .      ▲兵      .       .
 3   .       .       .       .       .
 2   .       .      ▲槍      .       .
 1  ▲帥      .       .       .       .
차례: ▲   ply: 0
""",
            {
                # 槍 전방
                "c2-c3",
                # 槍 전방대각과 후방
                *("c2-b3", "c2-d3", "c2-c1"),
                # 兵
                *("c4-c3", "c4-c5"),
                # 帥
                *("a1-a2", "a1-b1", "a1-b2"),
            },
        ),
        # 두 칸 앞의 적은 잡는다.
        (
            """\
     a       b       c       d       e
 5   .       .       .       .      ▽帥
 4   .       .      ▽兵      .       .
 3   .       .       .       .       .
 2   .       .      ▲槍      .       .
 1  ▲帥      .       .       .       .
차례: ▲   ply: 0
""",
            {
                # 槍 전방
                *(
                    "c2-c3",
                    "c2-c4",
                ),
                # 槍 전방대각과 후방
                *("c2-b3", "c2-d3", "c2-c1"),
                # 帥
                *("a1-a2", "a1-b1", "a1-b2"),
            },
        ),
    ],
    ids=["open", "friend-near", "enemy-near", "friend-far", "enemy-far"],
)
def test_a_lance_slides_forward_and_does_not_jump(board: str, expected: set[str]) -> None:
    game = Gungi(SINGLE_TIER)
    assert notation(game, game.parse(board)) == expected


def pieces_by_square(game: Gungi, position: GungiPosition) -> dict[str, str]:
    """렌더에서 읽은 `칸 → 말`. 빈칸은 빠진다."""
    header, *rows, _turn = game.render(position).splitlines()
    columns = header.split()
    squares: dict[str, str] = {}
    for line in rows:
        row, *cells = line.split()
        squares.update(
            {column + row: cell for column, cell in zip(columns, cells, strict=True) if cell != "."}
        )
    return squares


@given(picks=st.lists(st.integers(min_value=0, max_value=2**20), max_size=60))
def test_pieces_only_leave_the_board_by_being_captured(picks: list[int]) -> None:
    game = Gungi(SINGLE_TIER)
    position = game.initial_position()
    for pick in picks:
        moves = game.legal_moves(position)
        if not moves:
            break
        move = moves[pick % len(moves)]
        before = pieces_by_square(game, position)
        captured = before.get(game.render_move(move).split("-")[1])
        position = game.apply_move(position, move)
        after = pieces_by_square(game, position)
        assert Counter(after.values()) == Counter(before.values()) - Counter(
            [captured] if captured else []
        )
        # 잡히는 것은 언제나 상대 말이다. 수를 둔 뒤의 차례가 그 상대다.
        assert captured is None or captured.startswith(position.player.value)


@pytest.mark.parametrize("seed", range(50))
def test_a_random_episode_with_all_four_kinds_ends(seed: int) -> None:
    episode = play_random_episode(Gungi(SINGLE_TIER), seed)
    assert len(episode.moves) <= SINGLE_TIER.move_limit
