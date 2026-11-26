"""대국 화면의 상태. 칸 클릭을 수로 바꾼다. pygame 을 모른다.

합법수·수 두기·결과는 전부 `Gungi` 에 묻는다. 여기는 고른 칸만 기억한다.
"""

from __future__ import annotations

from collections.abc import Callable

from games.base import Outcome, Player
from games.mini_gungi import EndReason, Gungi, GungiMove, GungiPosition

# 사람이 아닌 상대. 자기 차례의 국면을 받아 수를 고른다.
Opponent = Callable[[GungiPosition], GungiMove]


class PlaySession:
    """대국 한 판. `opponent` 가 없으면 사람끼리 둔다."""

    def __init__(
        self,
        game: Gungi,
        opponent: Opponent | None = None,
        opponent_player: Player | None = None,
        position: GungiPosition | None = None,
    ) -> None:
        if (opponent is None) != (opponent_player is None):
            raise ValueError("상대와 상대의 선후는 함께 정한다")
        self.game = game
        self.position = game.initial_position() if position is None else position
        self.log: list[str] = []
        self.selected: int | None = None
        self.choosing: int | None = None
        """잡기와 쌓기 중에 골라야 하는 도착 칸. 고를 것이 없으면 `None` 이다."""
        self._opponent = opponent
        self._opponent_player = opponent_player

    @property
    def outcome(self) -> Outcome | None:
        return self.game.outcome(self.position)

    @property
    def end_reason(self) -> EndReason | None:
        return self.game.end_reason(self.position)

    @property
    def opponent_to_move(self) -> bool:
        return self.outcome is None and self.position.player is self._opponent_player

    @property
    def targets(self) -> frozenset[int]:
        """고른 말이 갈 수 있는 칸."""
        return frozenset(
            move.target
            for move in self.game.legal_moves(self.position)
            if move.source == self.selected
        )

    def click(self, square: int | None) -> None:
        """사람이 칸을 눌렀다. 판 밖이면 `None` 이다."""
        if self.opponent_to_move:
            return
        if self.choosing is not None:
            self._release()
            return
        moves = self.game.legal_moves(self.position)
        candidates = [m for m in moves if m.source == self.selected and m.target == square]
        if len(candidates) > 1:
            self.choosing = square
        elif candidates:
            self._play(candidates[0])
        elif square != self.selected and any(move.source == square for move in moves):
            self.selected = square
        else:
            self._release()

    def choose(self, *, stack: bool) -> None:
        """`choosing` 칸에서 잡기(`stack=False`)나 쌓기를 고른다."""
        if self.selected is None or self.choosing is None:
            return
        self._play(GungiMove(self.selected, self.choosing, stack))

    def play_opponent(self) -> None:
        """상대의 차례면 상대가 한 수 둔다."""
        if self._opponent is not None and self.opponent_to_move:
            self._play(self._opponent(self.position))

    def _play(self, move: GungiMove) -> None:
        text = self.game.render_log_move(self.position, move)
        entry = f"{self.position.ply + 1}. {self.position.player.value} {text}"
        self.position = self.game.apply_move(self.position, move)
        self.log.append(entry)
        self._release()

    def _release(self) -> None:
        self.selected = None
        self.choosing = None
