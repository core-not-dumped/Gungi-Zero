"""군의 어댑터 테스트. `games.base.Game` 으로만 게임을 보고 판은 텍스트 렌더로 읽는다.

규칙별 fixture 는 `tests/gungi_fixtures.py` 에 있고 `tests/test_gungi_rules.py` 가 돌린다.
"""

from collections import Counter
from collections.abc import Iterator
from dataclasses import replace

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from ai.random_play import play_random_episode
from games.base import Outcome, Player
from games.mini_gungi import (
    MARSHAL_ONLY,
    MINI_GUNGI,
    SINGLE_TIER,
    EndReason,
    Gungi,
    GungiMove,
    GungiPosition,
    PieceType,
    Placement,
    VariantConfig,
)
from games.nano_gungi import NANO_GUNGI

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


def test_reaching_the_move_limit_is_a_draw() -> None:
    game = Gungi(replace(MARSHAL_ONLY, move_limit=4))
    position = play(game, game.initial_position(), "c1-b1", "c5-b5", "b1-c1")
    assert game.outcome(position) is None
    position = play(game, position, "b5-c5")
    assert game.outcome(position) is Outcome.DRAW
    assert game.legal_moves(position) == ()
    with pytest.raises(ValueError):
        play(game, position, "c1-b1")


# 두 帥 가 한 칸 갔다가 돌아온다. 네 수마다 같은 판, 같은 차례가 된다.
ROUND_TRIP = ("c1-b1", "c5-b5", "b1-c1", "b5-c5")


def repetition_plane(game: Gungi, position: GungiPosition) -> float:
    plane = game.encode(position)[16]
    assert np.all(plane == plane[0, 0])
    return float(plane[0, 0])


def test_the_repetition_limit_comes_from_the_variant() -> None:
    game = Gungi(replace(MARSHAL_ONLY, repetition_limit=2))
    position = play(game, game.initial_position(), *ROUND_TRIP)
    assert game.outcome(position) is Outcome.DRAW
    with pytest.raises(ValueError):
        play(game, position, "c1-b1")


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


def test_the_log_notation_does_not_mark_stacking_on_an_enemy() -> None:
    game = Gungi(MINI_GUNGI)
    position = game.parse("""     a       b       c       d       e
 5   .       .       .       .      ▽帥
 4   .       .       .       .       .
 3   .       .      ▽兵      .       .
 2   .       .      ▲兵      .       .
 1  ▲帥      .       .       .       .
차례: ▲   ply: 4
""")
    assert game.render_log_move(position, game.parse_move("c2-c3^")) == "c2-c3^"


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


@pytest.mark.parametrize(
    "config", [SINGLE_TIER, MINI_GUNGI, NANO_GUNGI], ids=["single-tier", "mini", "nano"]
)
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


@pytest.mark.parametrize(
    "config", [SINGLE_TIER, MINI_GUNGI, NANO_GUNGI], ids=["single-tier", "mini", "nano"]
)
@given(picks=st.lists(st.integers(min_value=0, max_value=2**20), max_size=60))
def test_towers_stay_within_the_tier_limit_with_any_marshal_on_top(
    config: VariantConfig, picks: list[int]
) -> None:
    game = Gungi(config)
    for _, _, after in random_walk(game, picks):
        for tower in towers_by_square(game, after).values():
            assert len(tower) <= config.max_tier
            assert not any(piece.endswith("帥") for piece in tower[1:])


@pytest.mark.parametrize(
    "config",
    [MARSHAL_ONLY, SINGLE_TIER, MINI_GUNGI, NANO_GUNGI],
    ids=["marshal-only", "single-tier", "mini", "nano"],
)
@given(picks=st.lists(st.integers(min_value=0, max_value=2**20), max_size=60))
def test_the_planes_add_up_to_the_pieces_on_the_board(
    config: VariantConfig, picks: list[int]
) -> None:
    game = Gungi(config)
    for _, _, after in random_walk(game, picks):
        on_board = pieces(towers_by_square(game, after)).total()
        assert game.encode(after)[:16].sum() == on_board


@pytest.mark.parametrize(
    "config", [SINGLE_TIER, MINI_GUNGI, NANO_GUNGI], ids=["single-tier", "mini", "nano"]
)
@pytest.mark.parametrize("seed", range(50))
def test_a_random_episode_with_pieces_besides_marshals_ends(
    config: VariantConfig, seed: int
) -> None:
    episode = play_random_episode(Gungi(config), seed)
    assert len(episode.moves) <= config.move_limit


@pytest.mark.parametrize(
    ("config", "moves", "reason"),
    [
        (replace(MARSHAL_ONLY, move_limit=4), ROUND_TRIP, EndReason.MOVE_LIMIT),
        (MARSHAL_ONLY, ROUND_TRIP * 2, EndReason.REPETITION),
        # 둘 다 걸리면 반복으로 센다.
        (replace(MARSHAL_ONLY, move_limit=8), ROUND_TRIP * 2, EndReason.REPETITION),
        (MARSHAL_ONLY, ("c1-c2", "c5-c4", "c2-c3", "c4-b4", "c3-b4"), EndReason.MARSHAL_CAPTURE),
    ],
)
def test_a_finished_game_says_why_it_ended(
    config: VariantConfig, moves: tuple[str, ...], reason: EndReason
) -> None:
    game = Gungi(config)
    before = play(game, game.initial_position(), *moves[:-1])
    assert game.end_reason(before) is None
    assert game.end_reason(play(game, before, moves[-1])) is reason


def test_a_tower_is_listed_from_the_bottom_with_each_owner() -> None:
    game = Gungi(NANO_GUNGI)
    position = game.parse("""     a       b       c
 4   .      ▽帥      .
 3   .       .       .
 2 ▽兵/▲大   .       .
 1   .      ▲帥      .
차례: ▲   ply: 0
""")
    a2 = game.parse_move("a2-a2").source
    assert game.tower(position, a2) == ((Player.FIRST, "大"), (Player.SECOND, "兵"))
    assert game.tower(position, 0) == ()
