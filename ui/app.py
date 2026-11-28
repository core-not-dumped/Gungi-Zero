"""pygame 대국 화면. 설정 화면과 대국 화면을 그리고, 입력을 `PlaySession` 에 넘긴다.

게임 변형과 사람이 아닌 상대는 밖에서 받는다 (`scripts/play.py`).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial

import pygame

from games.base import Player
from games.mini_gungi import EndReason, Gungi, TowerView
from ui.session import Opponent, PlaySession

# (게임, 시드) 를 받아 새 대국의 상대를 만든다.
OpponentFactory = Callable[[Gungi, int], Opponent]

WINDOW_SIZE = (880, 640)
# 상대가 두기 전에 쉬는 시간. 사람이 자기가 둔 수를 볼 틈이다.
OPPONENT_DELAY_MS = 600
FRAMES_PER_SECOND = 30
MAX_SEED_DIGITS = 9

BOARD_LEFT, BOARD_TOP = 50, 60
BOARD_SPAN = 500
MAX_CELL = 110
PANEL_LEFT = 590
LOG_TOP = 270
LOG_LINE = 24
LOG_LINES = (WINDOW_SIZE[1] - LOG_TOP - 10) // LOG_LINE

# 한자와 ▲▽ 를 가진 글꼴. 앞에서부터 설치된 것을 쓴다.
FONT_NAMES = "malgungothic,applesdgothicneo,notosanscjkkr,notosanscjk,nanumgothic,microsoftyahei"

BACKGROUND = (44, 40, 36)
TEXT = (240, 234, 222)
CELL = (214, 178, 122)
CELL_BORDER = (96, 70, 40)
SELECTED = (250, 222, 90)
CHOOSING = (240, 140, 60)
TARGET = (40, 130, 70)
BUTTON = (84, 76, 66)
BUTTON_SELECTED = (166, 112, 48)
# 주인별 (말 바탕, 말 글자).
PIECE_COLORS = {
    Player.FIRST: ((30, 28, 26), (250, 246, 238)),
    Player.SECOND: ((252, 250, 244), (150, 30, 30)),
}

PLAYER_NAMES = {Player.FIRST: "▲ 선수", Player.SECOND: "▽ 후수"}
END_REASON_NAMES = {
    EndReason.MARSHAL_CAPTURE: "帥 잡기",
    EndReason.NO_LEGAL_MOVES: "합법수 없음",
    EndReason.REPETITION: "같은 판 반복",
    EndReason.MOVE_LIMIT: "수 제한",
}


@dataclass(frozen=True)
class Button:
    label: str
    rect: pygame.Rect
    action: Callable[[], None]
    selected: bool = False


class App:
    """화면 하나의 상태. `session` 이 없으면 설정 화면이다."""

    def __init__(
        self,
        variants: Sequence[tuple[str, Gungi]],
        opponent_name: str,
        make_opponent: OpponentFactory,
        levels: Mapping[str, Sequence[tuple[str, OpponentFactory]]] | None = None,
    ) -> None:
        """`variants` 는 (이름, 게임) 목록이고 첫 번째가 기본이다.

        `levels` 는 변형 이름마다의 난이도 목록 (이름, 상대)이고 약한 것부터다. 난이도가 있는
        변형에서는 `opponent_name`·`make_opponent` 대신 고른 난이도의 상대가 둔다.
        """
        self._variants = tuple(variants)
        self._default_opponent = (opponent_name, make_opponent)
        self._levels = {name: tuple(options) for name, options in (levels or {}).items()}
        self._level = 0
        self._variant = 0
        self._vs_opponent = True
        self._human = Player.FIRST
        self.seed = 0
        self.session: PlaySession | None = None
        self._opponent_due: int | None = None
        self.log_scroll = 0
        """기보에서 마지막 수부터 위로 올려 본 줄 수."""
        self._fonts: dict[int, pygame.font.Font] = {}

    # 상대

    def _variant_levels(self) -> tuple[tuple[str, OpponentFactory], ...]:
        return self._levels.get(self._variants[self._variant][0], ())

    def _opponent(self) -> tuple[str, OpponentFactory]:
        levels = self._variant_levels()
        return levels[self._level] if levels else self._default_opponent

    @property
    def opponent_name(self) -> str:
        """지금 고른 변형과 난이도에서 두는 상대의 이름."""
        return self._opponent()[0]

    # 입력

    def click(self, point: tuple[int, int]) -> None:
        for button in self.buttons():
            if button.rect.collidepoint(point):
                button.action()
                return
        if self.session is not None:
            self.session.click(self._square_at(self.session, point))

    def key(self, key: int, text: str) -> None:
        """설정 화면에서 시드를 숫자와 백스페이스로 고친다."""
        if self.session is not None:
            return
        if key == pygame.K_BACKSPACE:
            self.seed //= 10
        elif len(text) == 1 and text in "0123456789":
            typed = self.seed * 10 + int(text)
            if len(str(typed)) <= MAX_SEED_DIGITS:
                self.seed = typed

    def scroll(self, lines: int) -> None:
        """기보를 `lines` 줄 위(지난 수)로 넘긴다. 음수면 아래로 간다."""
        if self.session is None:
            return
        hidden = max(len(self.session.log) - LOG_LINES, 0)
        self.log_scroll = min(max(self.log_scroll + lines, 0), hidden)

    def visible_log(self) -> list[str]:
        """화면에 보이는 기보."""
        if self.session is None:
            return []
        end = len(self.session.log) - self.log_scroll
        return self.session.log[max(end - LOG_LINES, 0) : end]

    def tick(self, now_ms: int) -> None:
        """상대의 차례가 되고 `OPPONENT_DELAY_MS` 가 지나면 상대가 둔다."""
        if self.session is None or not self.session.opponent_to_move:
            self._opponent_due = None
        elif self._opponent_due is None:
            self._opponent_due = now_ms + OPPONENT_DELAY_MS
        elif now_ms >= self._opponent_due:
            self._opponent_due = None
            self.session.play_opponent()

    def _start(self) -> None:
        game = self._variants[self._variant][1]
        if self._vs_opponent:
            opponent = self._opponent()[1](game, self.seed)
            self.session = PlaySession(game, opponent, self._human.opponent)
        else:
            self.session = PlaySession(game)
        self._opponent_due = None
        self.log_scroll = 0

    def _back_to_setup(self) -> None:
        self.session = None

    def _set_variant(self, variant: int) -> None:
        if variant != self._variant:
            self._level = 0
        self._variant = variant

    def _add_level(self, amount: int) -> None:
        self._level = min(max(self._level + amount, 0), len(self._variant_levels()) - 1)

    def _set_vs_opponent(self, vs_opponent: bool) -> None:
        self._vs_opponent = vs_opponent

    def _set_human(self, human: Player) -> None:
        self._human = human

    def _add_seed(self, amount: int) -> None:
        self.seed = min(max(self.seed + amount, 0), 10**MAX_SEED_DIGITS - 1)

    # 배치

    def buttons(self) -> list[Button]:
        """지금 화면에 있는 버튼. 그리기와 클릭이 같은 목록을 본다."""
        if self.session is None:
            return self._setup_buttons()
        buttons = [Button("새 대국", pygame.Rect(PANEL_LEFT, 20, 140, 40), self._back_to_setup)]
        if self.session.choosing is not None:
            for index, (label, stack) in enumerate((("잡기", False), ("쌓기", True))):
                rect = pygame.Rect(PANEL_LEFT + index * 130, 190, 120, 44)
                buttons.append(Button(label, rect, partial(self.session.choose, stack=stack)))
        return buttons

    def _setup_buttons(self) -> list[Button]:
        def option(
            label: str, column: int, top: int, action: Callable[[], None], on: bool
        ) -> Button:
            return Button(label, pygame.Rect(260 + column * 190, top, 170, 44), action, on)

        buttons = [
            option(name, index, 130, partial(self._set_variant, index), index == self._variant)
            for index, (name, _) in enumerate(self._variants)
        ]
        ai_label = "AI" if self._variant_levels() else self.opponent_name
        buttons += [
            option(label, index, 210, partial(self._set_vs_opponent, vs), vs == self._vs_opponent)
            for index, (label, vs) in enumerate((("사람", False), (ai_label, True)))
        ]
        if self._vs_opponent:
            buttons += [
                option(
                    PLAYER_NAMES[player],
                    index,
                    290,
                    partial(self._set_human, player),
                    player is self._human,
                )
                for index, player in enumerate(Player)
            ]
            buttons += [
                Button("-", pygame.Rect(260, 370, 50, 44), partial(self._add_seed, -1)),
                Button("+", pygame.Rect(500, 370, 50, 44), partial(self._add_seed, 1)),
            ]
        if self._vs_opponent and self._variant_levels():
            buttons += [
                Button("약하게", pygame.Rect(260, 430, 90, 44), partial(self._add_level, -1)),
                Button("강하게", pygame.Rect(530, 430, 90, 44), partial(self._add_level, 1)),
            ]
        buttons.append(Button("대국 시작", pygame.Rect(260, 500, 360, 56), self._start))
        return buttons

    def square_rect(self, square: int) -> pygame.Rect:
        """칸이 화면에서 차지하는 자리. 선수의 끝줄(row 0)이 아래다."""
        assert self.session is not None
        geometry = self.session.game.geometry
        size = min(MAX_CELL, BOARD_SPAN // max(geometry.width, geometry.height))
        column, row = geometry.column_and_row(square)
        return pygame.Rect(
            BOARD_LEFT + column * size,
            BOARD_TOP + (geometry.height - 1 - row) * size,
            size,
            size,
        )

    def _square_at(self, session: PlaySession, point: tuple[int, int]) -> int | None:
        squares = range(session.game.geometry.square_count)
        return next((s for s in squares if self.square_rect(s).collidepoint(point)), None)

    def result_text(self) -> str | None:
        """끝난 대국의 결과와 사유. 진행 중이면 `None` 이다."""
        if self.session is None:
            return None
        outcome, reason = self.session.outcome, self.session.end_reason
        if outcome is None or reason is None:
            return None
        result = "무승부" if outcome.winner is None else f"{PLAYER_NAMES[outcome.winner]} 승"
        return f"{result} ({END_REASON_NAMES[reason]})"

    # 그리기

    def draw(self, surface: pygame.Surface) -> None:
        surface.fill(BACKGROUND)
        if self.session is None:
            self._draw_setup(surface)
        else:
            self._draw_board(surface, self.session)
            self._draw_panel(surface, self.session)
        for button in self.buttons():
            color = BUTTON_SELECTED if button.selected else BUTTON
            pygame.draw.rect(surface, color, button.rect, border_radius=6)
            self._write(surface, button.label, 20, TEXT, center=button.rect.center)

    def _draw_setup(self, surface: pygame.Surface) -> None:
        self._write(surface, "군의 대국", 36, TEXT, topleft=(80, 50))
        rows = [("변형", 130), ("상대", 210)]
        if self._vs_opponent:
            rows += [("내 선후", 290), ("AI 시드", 370)]
            self._write(surface, str(self.seed), 22, TEXT, center=(405, 392))
            if self._variant_levels():
                rows.append(("난이도", 430))
                self._write(surface, self.opponent_name, 20, TEXT, center=(440, 452))
        for label, top in rows:
            self._write(surface, label, 22, TEXT, midleft=(80, top + 22))

    def _draw_board(self, surface: pygame.Surface, session: PlaySession) -> None:
        geometry = session.game.geometry
        targets = session.targets
        for square in range(geometry.square_count):
            rect = self.square_rect(square)
            color = CELL
            if square == session.choosing:
                color = CHOOSING
            elif square == session.selected:
                color = SELECTED
            pygame.draw.rect(surface, color, rect)
            pygame.draw.rect(surface, CELL_BORDER, rect, 1)
            tower = session.game.tower(session.position, square)
            self._draw_tower(surface, tower, rect)
            if square in targets:
                pygame.draw.rect(surface, TARGET, rect.inflate(-6, -6), 4, border_radius=8)
                if not tower:
                    pygame.draw.circle(surface, TARGET, rect.center, rect.width // 8)

        # 열·행 이름은 수 표기와 같은 곳에서 얻는다 — 로그의 표기와 어긋날 수 없다.
        for column in range(geometry.width):
            name = geometry.column_name(column)
            x = self.square_rect(geometry.square(column, 0)).centerx
            self._write(surface, name, 20, TEXT, center=(x, BOARD_TOP - 20))
        for row in range(geometry.height):
            name = geometry.row_name(row)
            y = self.square_rect(geometry.square(0, row)).centery
            self._write(surface, name, 20, TEXT, center=(BOARD_LEFT - 22, y))

    def _draw_tower(self, surface: pygame.Surface, tower: TowerView, cell: pygame.Rect) -> None:
        """위 말을 크게 위에, 아래 말을 작게 그 밑에 그린다."""
        if not tower:
            return
        inner = cell.inflate(-18, -18)
        if len(tower) == 1:
            plates = [(tower[0], pygame.Rect(inner.x, inner.y + inner.h // 5, inner.w, 0), 0.6)]
        else:
            plates = [
                (tower[1], pygame.Rect(inner.x, inner.y, inner.w, 0), 0.56),
                (tower[0], pygame.Rect(inner.x + inner.w // 8, 0, inner.w * 3 // 4, 0), 0.38),
            ]
        for index, ((owner, symbol), plate, share) in enumerate(plates):
            plate.height = int(inner.h * share)
            if index == 1:
                plate.bottom = inner.bottom
            fill, ink = PIECE_COLORS[owner]
            pygame.draw.rect(surface, fill, plate, border_radius=8)
            pygame.draw.rect(surface, CELL_BORDER, plate, 2, border_radius=8)
            size = max(14, plate.height * 3 // 5)
            self._write(surface, owner.value + symbol, size, ink, center=plate.center)

    def _draw_panel(self, surface: pygame.Surface, session: PlaySession) -> None:
        def line(text: str, top: int, size: int = 22) -> None:
            self._write(surface, text, size, TEXT, topleft=(PANEL_LEFT, top))

        result = self.result_text()
        if result is None:
            who = ""
            if self._vs_opponent:
                who = f" ({self.opponent_name})" if session.opponent_to_move else " (나)"
            line(f"차례: {PLAYER_NAMES[session.position.player]}{who}", 80)
            line(f"{session.position.ply + 1} 수째", 112)
        else:
            line("대국 종료", 80)
            line(result, 112)
            line(f"{session.position.ply} 수", 144)
        if session.choosing is not None:
            line("잡을까, 쌓을까?", 156)
        for index, entry in enumerate(self.visible_log()):
            line(entry, LOG_TOP + index * LOG_LINE, 18)

    def _write(
        self,
        surface: pygame.Surface,
        text: str,
        size: int,
        color: tuple[int, int, int],
        **anchor: tuple[int, int],
    ) -> None:
        if size not in self._fonts:
            self._fonts[size] = pygame.font.SysFont(FONT_NAMES, size)
        image = self._fonts[size].render(text, True, color)
        surface.blit(image, image.get_rect(**anchor))


def run(app: App) -> None:
    """창을 열고 닫힐 때까지 돈다."""
    pygame.init()
    try:
        window = pygame.display.set_mode(WINDOW_SIZE)
        pygame.display.set_caption("gungi-zero")
        clock = pygame.time.Clock()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    app.click(event.pos)
                elif event.type == pygame.MOUSEWHEEL:
                    app.scroll(event.y)
                elif event.type == pygame.KEYDOWN:
                    app.key(event.key, event.unicode)
            app.tick(pygame.time.get_ticks())
            app.draw(window)
            pygame.display.flip()
            clock.tick(FRAMES_PER_SECOND)
    finally:
        pygame.quit()
