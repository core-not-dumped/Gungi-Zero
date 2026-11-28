"""대국 화면의 조립. 창을 열지 않는다 — 클릭은 좌표로 넣고 그리기는 메모리 표면에 한다."""

from collections.abc import Iterator

import pygame
import pytest

from games.base import Player
from games.mini_gungi import Gungi
from scripts.play import OPPONENT_NAME, VARIANTS, random_opponent
from ui.app import LOG_LINES, OPPONENT_DELAY_MS, WINDOW_SIZE, App


@pytest.fixture(autouse=True)
def fonts() -> Iterator[None]:
    pygame.font.init()
    yield
    pygame.font.quit()


def new_app() -> App:
    return App(VARIANTS, OPPONENT_NAME, random_opponent)


def press(app: App, label: str) -> None:
    [button] = [button for button in app.buttons() if button.label == label]
    app.click(button.rect.center)


def labels(app: App) -> list[str]:
    return [button.label for button in app.buttons()]


def click_square(app: App, name: str) -> None:
    assert app.session is not None
    square = app.session.game.parse_move(f"{name}-{name}").source
    app.click(app.square_rect(square).center)


def draw(app: App) -> None:
    app.draw(pygame.Surface(WINDOW_SIZE))


def wait_for_opponent(app: App) -> None:
    app.tick(1000)
    app.tick(1000 + OPPONENT_DELAY_MS)


def test_the_default_variant_is_nano() -> None:
    app = new_app()
    press(app, "대국 시작")
    assert app.session is not None
    assert app.session.game.tensor_shape[1:] == (4, 3)


def test_the_mini_variant_can_be_chosen() -> None:
    app = new_app()
    press(app, VARIANTS[1][0])
    press(app, "대국 시작")
    assert app.session is not None
    assert app.session.game.tensor_shape[1:] == (5, 5)


def test_the_nano_board_starts_from_the_play_setup() -> None:
    app = new_app()
    press(app, "대국 시작")
    assert app.session is not None
    game, position = app.session.game, app.session.position
    assert game.tensor_shape[1:] == (4, 3)
    placed = {
        name: game.tower(position, game.parse_move(f"{name}-{name}").source)
        for name in ("b1", "c1", "c2")
    }
    assert [tower[0][1] for tower in placed.values()] == ["帥", "兵", "大"]
    assert {tower[0][0] for tower in placed.values()} == {Player.FIRST}


def test_two_people_take_turns_on_the_same_screen() -> None:
    app = new_app()
    press(app, "사람")
    press(app, "대국 시작")
    click_square(app, "b1")
    click_square(app, "a1")
    wait_for_opponent(app)
    assert app.session is not None
    assert len(app.session.log) == 1
    click_square(app, "b4")
    click_square(app, "c4")
    assert len(app.session.log) == 2


def test_the_random_opponent_moves_after_a_pause() -> None:
    app = new_app()
    press(app, "▽ 후수")
    press(app, "대국 시작")
    assert app.session is not None
    assert app.session.opponent_to_move
    app.tick(1000)
    app.tick(1000 + OPPONENT_DELAY_MS - 1)
    assert app.session.log == []
    app.tick(1000 + OPPONENT_DELAY_MS)
    assert len(app.session.log) == 1
    assert app.session.position.player is Player.SECOND


def log_after_reply(digits: str) -> list[str]:
    app = new_app()
    press(app, "▽ 후수")
    for digit in digits:
        app.key(getattr(pygame, f"K_{digit}"), digit)
    press(app, "대국 시작")
    wait_for_opponent(app)
    assert app.session is not None
    return app.session.log


def test_the_seed_fixes_the_random_opponent() -> None:
    assert log_after_reply("7") == log_after_reply("7")
    assert len({log_after_reply(str(seed))[0] for seed in range(1, 10)}) > 1


def test_the_seed_is_edited_with_buttons_and_the_keyboard() -> None:
    app = new_app()
    press(app, "+")
    press(app, "+")
    press(app, "-")
    assert app.seed == 1
    app.key(pygame.K_2, "2")
    assert app.seed == 12
    app.key(pygame.K_BACKSPACE, "\b")
    app.key(pygame.K_BACKSPACE, "\b")
    app.key(pygame.K_BACKSPACE, "\b")
    assert app.seed == 0
    press(app, "-")
    assert app.seed == 0


def test_side_and_seed_are_offered_only_against_the_random_opponent() -> None:
    app = new_app()
    assert "▽ 후수" in labels(app)
    press(app, "사람")
    assert "▽ 후수" not in labels(app)
    assert "+" not in labels(app)


def test_capture_or_stack_is_chosen_with_a_button() -> None:
    app = new_app()
    press(app, "사람")
    press(app, "대국 시작")
    assert app.session is not None
    game = app.session.game
    app.session.position = game.parse("""\
     a       b       c
 4   .      ▽帥     ▽大
 3   .       .      ▽兵
 2   .       .      ▲兵
 1  ▲大     ▲帥      .
차례: ▲   ply: 0
""")
    assert "쌓기" not in labels(app)
    click_square(app, "c2")
    click_square(app, "c3")
    draw(app)
    press(app, "쌓기")
    assert app.session.log == ["1. ▲ c2-c3^"]


def test_a_new_game_can_be_started_from_the_result() -> None:
    app = new_app()
    press(app, "사람")
    press(app, "대국 시작")
    assert app.session is not None
    game: Gungi = app.session.game
    app.session.position = game.parse("""\
     a       b       c
 4   .      ▽帥      .
 3  ▲大      .       .
 2   .       .       .
 1   .      ▲帥      .
차례: ▲   ply: 4
""")
    assert app.result_text() is None
    click_square(app, "a3")
    click_square(app, "b4")
    if app.session.choosing is not None:
        press(app, "잡기")
    assert app.result_text() == "▲ 선수 승 (帥 잡기)"
    draw(app)
    press(app, "새 대국")
    assert app.session is None
    assert "대국 시작" in labels(app)


def test_earlier_moves_can_be_scrolled_back_into_view() -> None:
    app = new_app()
    press(app, "사람")
    press(app, "대국 시작")
    assert app.session is not None
    app.session.log = [str(number) for number in range(1, LOG_LINES + 6)]
    assert app.visible_log() == app.session.log[5:]
    app.scroll(2)
    assert app.visible_log() == app.session.log[3:-2]
    app.scroll(100)
    assert app.visible_log() == app.session.log[:LOG_LINES]
    app.scroll(-100)
    assert app.visible_log() == app.session.log[5:]

    app.scroll(2)
    press(app, "새 대국")
    press(app, "대국 시작")
    assert app.log_scroll == 0


@pytest.mark.parametrize("variant", [name for name, _ in VARIANTS])
def test_every_screen_draws(variant: str) -> None:
    app = new_app()
    draw(app)
    press(app, variant)
    press(app, "대국 시작")
    click_square(app, "b1")
    draw(app)
    for button in app.buttons():
        assert pygame.Rect((0, 0), WINDOW_SIZE).contains(button.rect)


def levelled_app(chosen: list[str]) -> App:
    def factory(name: str):
        def make(game: Gungi, seed: int):
            chosen.append(name)
            return random_opponent(game, seed)

        return make

    names = ("랜덤 AI", "Elo 210", "Elo 405")
    levels = {VARIANTS[0][0]: [(name, factory(name)) for name in names]}
    return App(VARIANTS, OPPONENT_NAME, random_opponent, levels)


def test_a_variant_with_levels_starts_at_the_weakest_and_can_be_made_stronger() -> None:
    chosen: list[str] = []
    app = levelled_app(chosen)
    press(app, "대국 시작")
    press(app, "새 대국")
    press(app, "강하게")
    press(app, "강하게")
    press(app, "강하게")
    press(app, "대국 시작")
    press(app, "새 대국")
    press(app, "약하게")
    press(app, "대국 시작")

    assert chosen == ["랜덤 AI", "Elo 405", "Elo 210"]
    assert app.opponent_name == "Elo 210"
    draw(app)


def test_the_levelled_opponent_button_is_just_ai() -> None:
    assert "AI" in labels(levelled_app([]))


def test_a_variant_without_levels_keeps_the_single_opponent() -> None:
    chosen: list[str] = []
    app = levelled_app(chosen)
    press(app, "강하게")
    press(app, VARIANTS[1][0])

    assert "강하게" not in labels(app)
    assert OPPONENT_NAME in labels(app)
    press(app, "대국 시작")
    assert app.opponent_name == OPPONENT_NAME
    assert chosen == []


def test_levels_are_not_offered_between_two_people() -> None:
    app = levelled_app([])
    press(app, "사람")
    assert "강하게" not in labels(app)
    draw(app)
