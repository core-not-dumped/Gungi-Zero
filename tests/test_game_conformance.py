"""게임에 대해 파라미터화된 적합성 스위트.

이 스위트가 "`games.base.Game` 을 구현했다" 의 정의다. 새 게임을 붙일 때 `GAMES` 에 한 줄
더하고 통과시킨다. 테스트는 `games.base.Game` 을 통해서만 게임을 보고, 판을 읽을 때는
텍스트 렌더를 쓴다 — 텐서로 읽으면 인코딩 버그와 룰 버그가 섞인다.

게임마다 타입이 달라 여기서는 `Game[Any, Any]` 로 받는다. 프로토콜에 구조적으로 맞는지는
pyright 가 `tests/test_games_base.py` 에서 검사한다.
"""

from typing import Any

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from games.base import Game, Outcome, Player
from games.mini_gungi import MARSHAL_ONLY, Gungi
from games.tictactoe import TicTacToe

# 새 게임은 여기에 더한다. 적합성 스위트 전체가 그 게임에도 돈다.
GAMES: list[Any] = [
    pytest.param(TicTacToe(), id="tictactoe"),
    pytest.param(Gungi(MARSHAL_ONLY), id="gungi-marshal-only"),
]

# 합법수 목록에서 몇 번째를 고를지의 나열. 게임마다 분기 수가 달라 나머지로 접는다.
PICKS = st.lists(st.integers(min_value=0, max_value=2**20), max_size=40)


def walk(game: Game[Any, Any], picks: list[int]) -> list[Any]:
    """`picks` 를 따라 수를 두며 지나간 국면 전부를 모은다. 초기 국면을 포함한다."""
    position = game.initial_position()
    reached = [position]
    for pick in picks:
        moves = game.legal_moves(position)
        if not moves:
            break
        position = game.apply_move(position, moves[pick % len(moves)])
        reached.append(position)
    return reached


@pytest.mark.parametrize("game", GAMES)
@given(picks=PICKS)
def test_the_legal_action_mask_says_exactly_what_apply_move_accepts(
    game: Game[Any, Any], picks: list[int]
) -> None:
    """마스크를 `apply_move` 에 묶는다.

    합법수 목록으로 기대값을 만들면 마스크를 만든 식을 그대로 다시 적는 셈이어서 틀릴 수가
    없다. 그래서 행동 공간을 전부 쓸어 `apply_move` 가 받아들이는지로 판정한다. 쓸기는
    마지막 국면에만 한다 — 625 칸짜리 군의에서는 국면마다 쓸면 너무 느리다.
    """
    reached = walk(game, picks)
    for position in reached:
        mask = game.legal_action_mask(position)
        assert mask.shape == (game.action_space_size,)
        assert mask.dtype == bool
        for move in game.legal_moves(position):
            assert mask[game.action_index(move)]

    position = reached[-1]
    mask = game.legal_action_mask(position)
    for index in range(game.action_space_size):
        try:
            game.apply_move(position, game.move_from_action_index(index))
        except ValueError:
            accepted = False
        else:
            accepted = True
        assert bool(mask[index]) is accepted, index


@pytest.mark.parametrize("game", GAMES)
@given(picks=PICKS)
def test_every_legal_move_round_trips_through_its_action_index(
    game: Game[Any, Any], picks: list[int]
) -> None:
    for position in walk(game, picks):
        for move in game.legal_moves(position):
            index = game.action_index(move)
            assert 0 <= index < game.action_space_size
            assert game.move_from_action_index(index) == move


@pytest.mark.parametrize("game", GAMES)
@given(picks=PICKS)
def test_parse_round_trips_the_render(game: Game[Any, Any], picks: list[int]) -> None:
    for position in walk(game, picks):
        assert game.parse(game.render(position)) == position


@pytest.mark.parametrize("game", GAMES)
@given(picks=PICKS)
def test_a_position_is_either_terminal_with_a_result_or_has_a_legal_move(
    game: Game[Any, Any], picks: list[int]
) -> None:
    for position in walk(game, picks):
        outcome = game.outcome(position)
        if outcome is None:
            assert game.legal_moves(position) != ()
        else:
            assert isinstance(outcome, Outcome)
            assert game.legal_moves(position) == ()


@pytest.mark.parametrize("game", GAMES)
@given(picks=PICKS)
def test_the_board_key_is_hashable_and_equal_for_equal_positions(
    game: Game[Any, Any], picks: list[int]
) -> None:
    """여기서 확인할 수 있는 것은 이만큼이다.

    정작 중요한 성질 — "경로가 달라도 판과 차례가 같으면 같은 키" — 는 경로를 들고 다니는
    게임에서만 시험할 수 있다. 틱택토에는 경로가 없어 그 성질이 공짜로 참이 된다. 반복
    규칙을 가진 게임이 붙는 티켓(04)에서 그 성질을 시험하는 것이 이 테스트의 몫이 아니다.
    """
    for position in walk(game, picks):
        key = game.board_key(position)
        assert hash(key) == hash(game.board_key(game.parse(game.render(position))))
        assert key == game.board_key(game.parse(game.render(position)))


@pytest.mark.parametrize("game", GAMES)
@given(picks=PICKS)
def test_the_canonical_form_puts_the_mover_first_and_is_idempotent(
    game: Game[Any, Any], picks: list[int]
) -> None:
    for position in walk(game, picks):
        canonical = game.canonical(position)
        assert canonical.player is Player.FIRST
        assert canonical.ply == position.ply
        # 티켓은 `정규형(정규형(p)) == p` 로 적지만, 정규형은 멱등이다 — 이미 정규형인 국면에
        # 적용하면 그대로다. 그 특수한 경우가 티켓이 적은 등식이다.
        assert game.canonical(canonical) == canonical


@pytest.mark.parametrize("game", GAMES)
@given(picks=PICKS)
def test_the_canonical_form_carries_the_legal_move_set_across(
    game: Game[Any, Any], picks: list[int]
) -> None:
    for position in walk(game, picks):
        carried = {
            game.to_canonical_action_index(position, game.action_index(move))
            for move in game.legal_moves(position)
        }
        canonical = {game.action_index(move) for move in game.legal_moves(game.canonical(position))}
        assert carried == canonical


@pytest.mark.parametrize("game", GAMES)
@given(picks=PICKS)
def test_the_canonical_action_index_mapping_is_invertible(
    game: Game[Any, Any], picks: list[int]
) -> None:
    for position in walk(game, picks):
        for move in game.legal_moves(position):
            index = game.action_index(move)
            carried = game.to_canonical_action_index(position, index)
            assert 0 <= carried < game.action_space_size
            assert game.from_canonical_action_index(position, carried) == index


@pytest.mark.parametrize("game", GAMES)
@given(picks=PICKS)
def test_the_tensor_has_the_declared_shape_and_is_float32(
    game: Game[Any, Any], picks: list[int]
) -> None:
    for position in walk(game, picks):
        tensor = game.encode(position)
        # 경계를 넘는 표현은 ndarray 다 (ADR-0001). torch 는 의존성에 아예 없다.
        assert isinstance(tensor, np.ndarray)
        assert tensor.shape == game.tensor_shape
        assert tensor.dtype == np.float32


@pytest.mark.parametrize("game", GAMES)
@given(picks=PICKS)
def test_the_tensor_is_the_movers_view_so_the_canonical_form_encodes_the_same(
    game: Game[Any, Any], picks: list[int]
) -> None:
    for position in walk(game, picks):
        assert np.array_equal(game.encode(position), game.encode(game.canonical(position)))


@pytest.mark.parametrize("game", GAMES)
@given(picks=PICKS)
def test_positions_are_immutable_and_hashable(game: Game[Any, Any], picks: list[int]) -> None:
    for position in walk(game, picks):
        twin = game.parse(game.render(position))
        assert twin == position
        assert hash(twin) == hash(position)
        assert len({position, twin}) == 1
        # 속성 이름을 변수로 둔다 — 직접 대입하면 타입 검사기가 먼저 막아 테스트가 돌지 않는다.
        attribute = "player"
        with pytest.raises(AttributeError):
            setattr(position, attribute, position.player)
