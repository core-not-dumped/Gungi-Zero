"""군의 fixture 테스트. `games.base.Game` 으로만 게임을 보고 판은 텍스트 렌더로 읽는다."""

from collections import Counter
from collections.abc import Iterator
from dataclasses import replace

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from ai.random_play import play_random_episode
from games.base import Outcome
from games.mini_gungi import (
    MARSHAL_ONLY,
    MINI_GUNGI,
    SINGLE_TIER,
    Gungi,
    GungiMove,
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


def notation(game: Gungi, position: GungiPosition) -> set[str]:
    return {game.render_move(move) for move in game.legal_moves(position)}


def play(game: Gungi, position: GungiPosition, *moves: str) -> GungiPosition:
    for move in moves:
        position = game.apply_move(position, game.parse_move(move))
    return position


def test_the_initial_position_puts_the_second_player_on_the_rotated_setup() -> None:
    game = Gungi(MINI_GUNGI)
    assert game.render(game.initial_position()) == SPEC_BOARD.replace("ply: 1", "ply: 0")


def test_parse_reads_the_rendered_format_back() -> None:
    game = Gungi(MINI_GUNGI)
    position = game.parse(SPEC_BOARD)
    assert game.render(position) == SPEC_BOARD
    assert position.ply == 1


def test_a_tower_is_written_top_first_and_round_trips() -> None:
    game = Gungi(MINI_GUNGI)
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
    game = Gungi(MINI_GUNGI)
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
    game = Gungi(MINI_GUNGI)
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
    assert game.render_move(game.parse_move("b1-c3^")) == "b1-c3^"
    bad = ("b1c3", "f1-c3", "b6-c3", "b0-c3", "b1-c3x", "b01-c3", "b1-c3-d4", "")
    for text in (*bad, "b1^-c3", "b1-c3^^", "b1-c3^x"):
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


def test_the_action_space_is_stacking_times_source_square_times_target_square() -> None:
    game = Gungi(MARSHAL_ONLY)
    assert game.action_space_size == 1250
    # 칸 번호는 a1=0, b1=1, ... e5=24 이고 인덱스는 쌓기 * 625 + 출발 * 25 + 도착이다.
    # b1-c3 = 1*25+12.
    assert game.action_index(game.parse_move("b1-c3")) == 37
    assert game.action_index(game.parse_move("b1-c3^")) == 625 + 37
    assert game.render_move(game.move_from_action_index(624)) == "e5-e5"
    assert game.render_move(game.move_from_action_index(1249)) == "e5-e5^"
    for index in (-1, 1250):
        with pytest.raises(ValueError):
            game.move_from_action_index(index)


def test_every_action_index_round_trips_through_the_move_notation() -> None:
    game = Gungi(MINI_GUNGI)
    for index in range(game.action_space_size):
        text = game.render_move(game.move_from_action_index(index))
        assert game.action_index(game.parse_move(text)) == index


def test_the_canonical_form_keeps_the_stacking_choice() -> None:
    game = Gungi(MINI_GUNGI)
    position = game.parse(SPEC_BOARD.replace("차례: ▲", "차례: ▽"))
    for text, rotated in (("b4-b3", "d2-d3"), ("b4-b3^", "d2-d3^")):
        index = game.action_index(game.parse_move(text))
        carried = game.to_canonical_action_index(position, index)
        assert game.render_move(game.move_from_action_index(carried)) == rotated
        assert game.from_canonical_action_index(position, carried) == index


def test_the_canonical_form_rotates_the_board_and_swaps_the_colours() -> None:
    game = Gungi(MINI_GUNGI)
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
    game = Gungi(MINI_GUNGI)
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
    assert game.action_space_size == 288
    assert game.tensor_shape == (18, 4, 3)
    assert notation(game, game.initial_position()) == {"a1-b1", "a1-a2", "a1-b2"}


def test_a_variant_with_more_tiers_than_the_planes_hold_is_refused() -> None:
    with pytest.raises(ValueError):
        Gungi(replace(MARSHAL_ONLY, max_tier=3))


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


def towers_by_square(game: Gungi, position: GungiPosition) -> dict[str, list[str]]:
    """렌더에서 읽은 `칸 → 탑`. 탑은 맨 위부터이고 빈칸은 빠진다."""
    header, *rows, _turn = game.render(position).splitlines()
    columns = header.split()
    squares: dict[str, list[str]] = {}
    for line in rows:
        row, *cells = line.split()
        for column, cell in zip(columns, cells, strict=True):
            if cell != ".":
                squares[column + row] = cell.split("/")
    return squares


def pieces(towers: dict[str, list[str]]) -> Counter[str]:
    return Counter(piece for tower in towers.values() for piece in tower)


def random_walk(
    game: Gungi, picks: list[int]
) -> Iterator[tuple[GungiPosition, GungiMove, GungiPosition]]:
    """`picks` 를 따라 두며 `(두기 전 국면, 수, 둔 뒤 국면)` 을 낸다."""
    position = game.initial_position()
    for pick in picks:
        moves = game.legal_moves(position)
        if not moves:
            return
        move = moves[pick % len(moves)]
        after = game.apply_move(position, move)
        yield position, move, after
        position = after


@pytest.mark.parametrize("config", [SINGLE_TIER, MINI_GUNGI], ids=["single-tier", "mini"])
@given(picks=st.lists(st.integers(min_value=0, max_value=2**20), max_size=60))
def test_pieces_only_leave_the_board_by_being_captured(
    config: VariantConfig, picks: list[int]
) -> None:
    game = Gungi(config)
    for position, move, after in random_walk(game, picks):
        before = towers_by_square(game, position)
        text = game.render_move(move)
        target = before.get(text.removesuffix("^").split("-")[1], [])
        # 수를 둔 뒤의 차례가 상대다. 잡으면 도착 칸의 상대 말이 전부 사라지고, 쌓으면 그대로다.
        opponent = after.player.value
        captured = [] if text.endswith("^") else [p for p in target if p.startswith(opponent)]
        assert pieces(towers_by_square(game, after)) == pieces(before) - Counter(captured)


@pytest.mark.parametrize("config", [SINGLE_TIER, MINI_GUNGI], ids=["single-tier", "mini"])
@given(picks=st.lists(st.integers(min_value=0, max_value=2**20), max_size=60))
def test_towers_stay_within_the_tier_limit_with_any_marshal_on_top(
    config: VariantConfig, picks: list[int]
) -> None:
    game = Gungi(config)
    for _, _, after in random_walk(game, picks):
        for tower in towers_by_square(game, after).values():
            assert len(tower) <= config.max_tier
            assert not any(piece.endswith("帥") for piece in tower[1:])


@pytest.mark.parametrize("config", [SINGLE_TIER, MINI_GUNGI], ids=["single-tier", "mini"])
@pytest.mark.parametrize("seed", range(50))
def test_a_random_episode_with_all_four_kinds_ends(config: VariantConfig, seed: int) -> None:
    episode = play_random_episode(Gungi(config), seed)
    assert len(episode.moves) <= config.move_limit


# ── 쌓기와 단 ──
# 판 그림의 탑은 `상단/하단` 이다.


def board(turn: str, *rows: str) -> str:
    """5 행부터 1 행까지의 칸을 받아 텍스트 판 표기를 만든다."""
    lines = ["     a       b       c       d       e"]
    lines += [f" {5 - index}  {row}" for index, row in enumerate(rows)]
    return "\n".join([*lines, f"차례: {turn}   ply: 4"]) + "\n"


# 帥 a1 의 세 수. 아래 fixture 대부분에서 a1 의 ▲帥 는 판의 일에 끼지 않는다.
CORNER_MARSHAL = ("a1-a2", "a1-b1", "a1-b2")


def moves_on(text: str) -> set[str]:
    game = Gungi(MINI_GUNGI)
    return notation(game, game.parse(text))


def towers_after(text: str, move: str) -> dict[str, list[str]]:
    game = Gungi(MINI_GUNGI)
    return towers_by_square(game, play(game, game.parse(text), move))


# 쌓기: 적 칸에서는 잡기와 쌓기 중에 고른다.
ENEMY_IN_FRONT = board(
    "▲",
    " .       .       .       .      ▽帥",
    " .       .       .       .       .",
    " .       .      ▽兵      .       .",
    " .       .      ▲兵      .       .",
    "▲帥      .       .       .       .",
)


def test_an_enemy_topped_square_can_be_captured_or_stacked_on() -> None:
    assert moves_on(ENEMY_IN_FRONT) == {"c2-c3", "c2-c3^", "c2-c1", *CORNER_MARSHAL}


def test_stacking_on_an_enemy_keeps_it_underneath_and_capturing_removes_it() -> None:
    assert towers_after(ENEMY_IN_FRONT, "c2-c3^")["c3"] == ["▲兵", "▽兵"]
    assert towers_after(ENEMY_IN_FRONT, "c2-c3")["c3"] == ["▲兵"]
    game = Gungi(MINI_GUNGI)
    position = game.parse(ENEMY_IN_FRONT)
    # x 는 잡기에만 붙는다.
    assert game.render_log_move(position, game.parse_move("c2-c3")) == "c2-c3x"
    assert game.render_log_move(position, game.parse_move("c2-c3^")) == "c2-c3^"


def test_a_friendly_square_can_only_be_stacked_on() -> None:
    text = board(
        "▲",
        " .       .       .       .      ▽帥",
        " .       .       .       .       .",
        " .       .      ▲兵      .       .",
        " .       .      ▲兵      .       .",
        "▲帥      .       .       .       .",
    )
    assert moves_on(text) == {"c2-c3^", "c2-c1", "c3-c4", "c3-c2^", *CORNER_MARSHAL}
    assert towers_after(text, "c2-c3^")["c3"] == ["▲兵", "▲兵"]


def test_a_higher_piece_can_stack_down_onto_a_friend() -> None:
    text = board(
        "▲",
        " .       .       .       .      ▽帥",
        " .       .       .       .       .",
        " .       .      ▲兵      .       .",
        " .       .    ▲兵/▲大   .       .",
        "▲帥      .       .       .       .",
    )
    # c3 의 兵 (1단) 은 뒤의 2단 탑에 못 올라간다.
    assert moves_on(text) == {"c2-c3^", "c2-c1", "c3-c4", *CORNER_MARSHAL}


# 단수 제약
@pytest.mark.parametrize("tower", ["▽兵/▽槍", "▽兵/▲槍"])
def test_a_higher_tower_can_be_neither_captured_nor_stacked_on(tower: str) -> None:
    text = board(
        "▲",
        " .       .       .       .      ▽帥",
        " .       .       .       .       .",
        f" .       .    {tower}   .       .",
        " .       .      ▲兵      .       .",
        "▲帥      .       .       .       .",
    )
    assert moves_on(text) == {"c2-c1", *CORNER_MARSHAL}


def test_a_full_tower_can_be_captured_from_the_same_tier_but_not_stacked_on() -> None:
    text = board(
        "▲",
        " .       .       .       .      ▽帥",
        " .       .       .       .       .",
        " .       .    ▽兵/▽槍   .       .",
        " .       .    ▲兵/▲大   .       .",
        "▲帥      .       .       .       .",
    )
    assert moves_on(text) == {"c2-c3", "c2-c1", *CORNER_MARSHAL}


def test_a_tower_with_a_friend_on_top_cannot_be_captured_for_the_enemy_underneath() -> None:
    text = board(
        "▲",
        " .       .       .       .      ▽帥",
        " .       .       .       .       .",
        " .       .    ▲兵/▽兵   .       .",
        " .       .    ▲兵/▲大   .       .",
        "▲帥      .       .       .       .",
    )
    # 두 ▲兵 사이는 서로 막힌다 — 상대 탑이 2단이라 쌓을 자리가 없다.
    assert moves_on(text) == {"c3-c4", "c3-c5", "c2-c1", *CORNER_MARSHAL}


# 사거리 증가: 2단은 유한 방향이 한 칸 늘고, 무제한 방향은 그대로다.
@pytest.mark.parametrize(
    ("rows", "expected"),
    [
        pytest.param(
            (
                " .      ▽帥      .       .       .",
                " .       .       .       .       .",
                " .       .    ▲兵/▲大   .       .",
                " .       .       .       .       .",
                "▲帥      .       .       .       .",
            ),
            {"c3-c4", "c3-c5", "c3-c2", "c3-c1", *CORNER_MARSHAL},
            id="pawn",
        ),
        pytest.param(
            (
                " .      ▽帥      .       .       .",
                " .       .       .       .       .",
                " .       .    ▲槍/▲兵   .       .",
                " .       .       .       .       .",
                "▲帥      .       .       .       .",
            ),
            {
                # 전방 3 칸 중 판 안의 2 칸
                *("c3-c4", "c3-c5"),
                # 전방대각 2
                *("c3-b4", "c3-a5", "c3-d4", "c3-e5"),
                # 후방 2
                *("c3-c2", "c3-c1"),
                *CORNER_MARSHAL,
            },
            id="lance-back-and-diagonal",
        ),
        pytest.param(
            (
                " .      ▽帥      .       .       .",
                " .       .       .       .       .",
                " .       .       .       .       .",
                " .       .    ▲槍/▲兵   .       .",
                "▲帥      .       .       .       .",
            ),
            {
                # 전방 3
                *("c2-c3", "c2-c4", "c2-c5"),
                # 전방대각 2
                *("c2-b3", "c2-a4", "c2-d3", "c2-e4"),
                # 후방 2 칸 중 판 안의 1 칸
                "c2-c1",
                *CORNER_MARSHAL,
            },
            id="lance-forward",
        ),
        pytest.param(
            (
                " .      ▽帥      .       .       .",
                " .       .       .       .       .",
                " .       .    ▲大/▲兵   .       .",
                " .       .       .       .       .",
                " .      ▲帥      .       .       .",
            ),
            {
                # 직선은 그대로 무제한
                *("c3-c4", "c3-c5", "c3-c2", "c3-c1", "c3-b3", "c3-a3", "c3-d3", "c3-e3"),
                # 대각 2
                *("c3-b4", "c3-a5", "c3-d4", "c3-e5", "c3-b2", "c3-a1", "c3-d2", "c3-e1"),
                # 帥
                *("b1-a1", "b1-c1", "b1-a2", "b1-b2", "b1-c2"),
            },
            id="general",
        ),
        # 帥 는 帥ツケ 금지 때문에 실제 대국에서는 2단에 오르지 못한다. 이동표만 확인한다.
        pytest.param(
            (
                " .      ▽帥      .       .       .",
                " .       .       .       .       .",
                " .       .    ▲帥/▲兵   .       .",
                " .       .       .       .       .",
                " .       .       .       .       .",
            ),
            {
                *("c3-c4", "c3-c5", "c3-c2", "c3-c1", "c3-b3", "c3-a3", "c3-d3", "c3-e3"),
                *("c3-b4", "c3-a5", "c3-d4", "c3-e5", "c3-b2", "c3-a1", "c3-d2", "c3-e1"),
            },
            id="marshal",
        ),
    ],
)
def test_the_second_tier_adds_one_square_to_limited_directions(
    rows: tuple[str, ...], expected: set[str]
) -> None:
    assert moves_on(board("▲", *rows)) == expected


def test_a_second_tier_piece_is_still_blocked_by_a_first_tier_piece() -> None:
    text = board(
        "▲",
        " .       .       .       .      ▽帥",
        " .       .       .       .       .",
        " .       .      ▽兵      .       .",
        " .       .    ▲兵/▲大   .       .",
        "▲帥      .       .       .       .",
    )
    assert moves_on(text) == {"c2-c3", "c2-c3^", "c2-c1", *CORNER_MARSHAL}


def test_the_longer_reach_of_a_second_tier_lance_does_not_jump() -> None:
    text = board(
        "▲",
        " .      ▽帥      .       .       .",
        " .       .      ▽兵      .       .",
        " .       .       .       .       .",
        " .       .    ▲槍/▲兵   .       .",
        "▲帥      .       .       .       .",
    )
    assert moves_on(text) == {
        # 전방 3 칸 중 c4 에서 멈춘다.
        *("c2-c3", "c2-c4", "c2-c4^"),
        *("c2-b3", "c2-a4", "c2-d3", "c2-e4", "c2-c1"),
        *CORNER_MARSHAL,
    }


# 맨 위만 이동
def test_only_the_top_piece_moves_and_the_one_below_is_freed_when_it_leaves() -> None:
    game = Gungi(MINI_GUNGI)
    rows = (
        " .       .       .       .      ▽帥",
        " .       .       .       .       .",
        " .       .    ▽兵/▲大   .       .",
        " .       .       .       .       .",
        "▲帥      .       .       .       .",
    )
    # ▲ 의 차례여도 탑 아래의 ▲大 는 얼어 있어 帥 만 움직인다.
    assert moves_on(board("▲", *rows)) == set(CORNER_MARSHAL)

    freed = play(game, game.parse(board("▽", *rows)), "c3-c4")
    assert towers_by_square(game, freed)["c3"] == ["▲大"]
    assert notation(game, freed) == {
        # 위가 비킨 大 (1단). c4 의 적은 잡거나 쌓는다.
        *("c3-c4", "c3-c4^", "c3-c2", "c3-c1", "c3-b3", "c3-a3", "c3-d3", "c3-e3"),
        *("c3-b4", "c3-d4", "c3-b2", "c3-d2"),
        *CORNER_MARSHAL,
    }


# 탑 잡기
@pytest.mark.parametrize(
    ("tower", "left"),
    [
        pytest.param("▽兵/▽槍", ["▲兵"], id="all-enemy"),
        pytest.param("▽兵/▲槍", ["▲兵", "▲槍"], id="friend-underneath"),
    ],
)
def test_capturing_a_tower_removes_every_enemy_piece_and_keeps_mine(
    tower: str, left: list[str]
) -> None:
    text = board(
        "▲",
        " .       .       .       .      ▽帥",
        " .       .       .       .       .",
        f" .       .    {tower}   .       .",
        " .       .    ▲兵/▲大   .       .",
        "▲帥      .       .       .       .",
    )
    after = towers_after(text, "c2-c3")
    assert after["c3"] == left
    assert after["c2"] == ["▲大"]


# 帥 위 금지: 누구도 帥 위에 올라가지 못한다.
def test_nothing_can_be_stacked_on_either_marshal() -> None:
    text = board(
        "▲",
        " .       .       .       .       .",
        " .       .       .       .       .",
        " .       .      ▽帥      .       .",
        "▲兵      .      ▲兵      .       .",
        "▲帥      .       .       .       .",
    )
    # c3 의 적 帥 는 잡기만 된다. a2 의 兵 은 뒤의 아군 帥 에 올라가지 못한다.
    assert moves_on(text) == {"c2-c3", "c2-c1", "a2-a3", "a1-b1", "a1-b2"}


# 帥ツケ 금지: 帥 자신은 아군 위에도 적 위에도 올라가지 못한다. 위의 규칙과 다른 규칙이다.
def test_a_marshal_cannot_stack_on_anything() -> None:
    text = board(
        "▲",
        " .       .       .       .      ▽帥",
        " .       .       .       .       .",
        " .       .       .       .       .",
        "▲兵     ▽兵      .       .       .",
        "▲帥      .       .       .       .",
    )
    # b2 의 적은 잡기만 되고, a2 의 아군으로는 갈 수 없다.
    assert moves_on(text) == {"a1-b2", "a1-b1", "a2-a3"}
