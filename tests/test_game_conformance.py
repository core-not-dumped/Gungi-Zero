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

from ai.random_play import play_random_episode
from games.base import Game, Outcome, Player
from games.mini_gungi import MARSHAL_ONLY, MINI_GUNGI, SINGLE_TIER, Gungi
from games.nano_gungi import NANO_GUNGI
from games.tictactoe import TicTacToe

# 새 게임은 여기에 더한다. 적합성 스위트 전체가 그 게임에도 돈다.
GAMES: list[Any] = [
    pytest.param(TicTacToe(), id="tictactoe"),
    pytest.param(Gungi(MARSHAL_ONLY), id="gungi-marshal-only"),
    pytest.param(Gungi(SINGLE_TIER), id="gungi-single-tier"),
    pytest.param(Gungi(MINI_GUNGI), id="gungi-mini"),
    pytest.param(Gungi(NANO_GUNGI), id="gungi-nano"),
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
    마지막 국면에만 한다 — 1250 칸짜리 군의에서는 국면마다 쓸면 너무 느리다.
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
    """텍스트에는 경로가 없어 파싱한 국면은 원래 국면과 다를 수 있다. 보이는 것은 모두 같다."""
    for position in walk(game, picks):
        parsed = game.parse(game.render(position))
        assert game.render(parsed) == game.render(position)
        assert parsed.player is position.player
        assert parsed.ply == position.ply
        assert game.board_key(parsed) == game.board_key(position)
        assert game.parse(game.render(parsed)) == parsed
    initial = game.initial_position()
    assert game.parse(game.render(initial)) == initial


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
def test_the_board_key_is_hashable_and_survives_losing_the_path(
    game: Game[Any, Any], picks: list[int]
) -> None:
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


def swapped(outcome: Outcome | None) -> Outcome | None:
    """선수 승과 후수 승을 맞바꾼 결과. 무승부와 진행 중은 그대로다."""
    if outcome is None or outcome.winner is None:
        return outcome
    return Outcome.win_for(outcome.winner.opponent)


def check_the_frames_agree_on_the_result(game: Game[Any, Any], position: Any) -> None:
    outcome = game.outcome(position)
    expected = outcome if position.player is Player.FIRST else swapped(outcome)
    assert game.outcome(game.canonical(position)) is expected


@pytest.mark.parametrize("game", GAMES)
@given(picks=PICKS)
def test_the_canonical_form_swaps_the_winner_when_the_second_player_is_to_move(
    game: Game[Any, Any], picks: list[int]
) -> None:
    for position in walk(game, picks):
        check_the_frames_agree_on_the_result(game, position)


@pytest.mark.parametrize("game", GAMES)
def test_finished_episodes_keep_their_winner_across_frames(game: Game[Any, Any]) -> None:
    """끝난 국면은 짧은 무작위 걸음에 잘 안 나온다. 끝까지 둔 대국으로 두 차례 모두를 본다."""
    decided: set[Player] = set()
    for seed in range(100):
        final = play_random_episode(game, seed).final_position
        check_the_frames_agree_on_the_result(game, final)
        if game.outcome(final) is not Outcome.DRAW:
            decided.add(final.player)
    assert Player.SECOND in decided, "승패가 뒤바뀌는 경우가 한 번도 나오지 않았다"


@pytest.mark.parametrize("game", GAMES)
@given(picks=PICKS)
def test_a_move_played_in_either_frame_reaches_the_same_position(
    game: Game[Any, Any], picks: list[int]
) -> None:
    position = walk(game, picks)[-1]
    canonical = game.canonical(position)
    for move in game.legal_moves(position):
        index = game.to_canonical_action_index(position, game.action_index(move))
        carried = game.move_from_action_index(index)
        original = game.canonical(game.apply_move(position, move))
        assert game.canonical(game.apply_move(canonical, carried)) == original


@pytest.mark.parametrize("game", GAMES)
@given(picks=PICKS)
def test_a_canonical_position_is_refused_where_the_original_is_wanted(
    game: Game[Any, Any], picks: list[int]
) -> None:
    for position in walk(game, picks):
        canonical = game.canonical(position)
        assert canonical != position
        for convert in (game.to_canonical_action_index, game.from_canonical_action_index):
            with pytest.raises(ValueError):
                convert(canonical, 0)


@pytest.mark.parametrize("game", GAMES)
@pytest.mark.parametrize("player_to_move", list(Player))
def test_an_index_outside_the_action_space_is_refused(
    game: Game[Any, Any], player_to_move: Player
) -> None:
    position = game.initial_position()
    if player_to_move is Player.SECOND:
        position = game.apply_move(position, game.legal_moves(position)[0])
    assert position.player is player_to_move
    size = game.action_space_size
    for index in (-1, -size, size, 2 * size):
        with pytest.raises(ValueError):
            game.move_from_action_index(index)
        for convert in (game.to_canonical_action_index, game.from_canonical_action_index):
            with pytest.raises(ValueError):
                convert(position, index)


@pytest.mark.parametrize("game", GAMES)
@given(picks=PICKS)
def test_the_legal_moves_come_in_the_same_order_every_time(
    game: Game[Any, Any], picks: list[int]
) -> None:
    for position, replayed in zip(walk(game, picks), walk(game, picks), strict=True):
        assert game.legal_moves(replayed) == game.legal_moves(position)


@pytest.mark.parametrize("game", GAMES)
@given(picks=PICKS)
def test_parse_move_reads_back_what_render_move_writes(
    game: Game[Any, Any], picks: list[int]
) -> None:
    for position in walk(game, picks):
        for move in game.legal_moves(position):
            assert game.parse_move(game.render_move(move)) == move


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
    # 같은 수를 다시 두면 경로까지 같은 국면이 나온다.
    for position, replayed in zip(walk(game, picks), walk(game, picks), strict=True):
        assert replayed == position
        assert hash(replayed) == hash(position)
        text = game.render(position)
        one, twin = game.parse(text), game.parse(text)
        assert twin == one
        assert hash(twin) == hash(one)
        assert len({one, twin}) == 1
        # 속성 이름을 변수로 둔다 — 직접 대입하면 타입 검사기가 먼저 막아 테스트가 돌지 않는다.
        attribute = "player"
        with pytest.raises(AttributeError):
            setattr(position, attribute, position.player)
