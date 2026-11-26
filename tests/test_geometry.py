"""판 좌표와 회전. 군의 어댑터가 `geometry` 로 내보내는 것을 본다."""

import pytest

from games.mini_gungi import MINI_GUNGI, Gungi
from games.nano_gungi import NANO_GUNGI

GAMES = [pytest.param(Gungi(MINI_GUNGI), id="mini"), pytest.param(Gungi(NANO_GUNGI), id="nano")]


@pytest.mark.parametrize("game", GAMES)
def test_square_numbers_names_and_columns_and_rows_round_trip(game: Gungi) -> None:
    geometry = game.geometry
    names = []
    for square in range(geometry.square_count):
        assert geometry.square(*geometry.column_and_row(square)) == square
        assert geometry.square_named(geometry.square_name(square)) == square
        names.append(geometry.square_name(square))
    assert len(set(names)) == geometry.square_count


def test_squares_are_numbered_along_each_row_from_the_first_players_end() -> None:
    geometry = Gungi(NANO_GUNGI).geometry
    assert [geometry.square_name(square) for square in range(4)] == ["a1", "b1", "c1", "a2"]


@pytest.mark.parametrize(("name", "rotated"), [("a1", "c4"), ("b1", "b4"), ("c2", "a3")])
def test_rotation_turns_the_board_half_way_round(name: str, rotated: str) -> None:
    geometry = Gungi(NANO_GUNGI).geometry
    assert geometry.square_name(geometry.rotate(geometry.square_named(name))) == rotated
    assert geometry.square_name(geometry.rotate(geometry.square_named(rotated))) == name


@pytest.mark.parametrize("square", [-1, 12])
def test_a_square_off_the_board_is_refused(square: int) -> None:
    game = Gungi(NANO_GUNGI)
    for operation in (game.geometry.column_and_row, game.geometry.rotate, game.render_square):
        with pytest.raises(ValueError):
            operation(square)


@pytest.mark.parametrize(("column", "row"), [(-1, 0), (3, 0), (0, -1), (0, 4)])
def test_a_column_or_row_off_the_board_is_refused(column: int, row: int) -> None:
    with pytest.raises(ValueError):
        Gungi(NANO_GUNGI).geometry.square(column, row)


@pytest.mark.parametrize("name", ["", "a", "1", "d1", "a5", "a0", "A1", "a-1", "ab1"])
def test_a_name_that_is_not_a_square_of_the_board_is_refused(name: str) -> None:
    with pytest.raises(ValueError):
        Gungi(NANO_GUNGI).geometry.square_named(name)
