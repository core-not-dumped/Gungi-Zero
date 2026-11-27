"""나노 군의 시험 판의 조건. 실제 풀이에서 뽑는 것은 `test_solve_nano` 의 slow 테스트가 본다."""

from games.mini_gungi import Gungi
from games.nano_gungi import NANO_GUNGI
from scripts.nano_trials import involves_stacking

GAME = Gungi(NANO_GUNGI)

# 帥 둘과 서로 닿지 않는 말들.
PLAIN = """
     a       b       c
 4   .       .      ▽帥
 3   .       .       .
 2   .       .       .
 1  ▲帥      .       .
차례: ▲   ply: 8
"""
# 같은 판에 쌓인 탑. 둘 차례인 ▲ 에게는 쌓는 수가 없다.
WITH_A_TOWER = """
     a       b       c
 4   .       .      ▽帥
 3  ▽大/▽兵   .       .
 2   .       .       .
 1  ▲帥      .       .
차례: ▲   ply: 8
"""


def test_a_board_with_a_tower_involves_stacking() -> None:
    assert not any(move.stack for move in GAME.legal_moves(GAME.parse(WITH_A_TOWER)))
    assert involves_stacking(GAME, GAME.parse(WITH_A_TOWER))


def test_a_board_where_stacking_is_legal_involves_stacking() -> None:
    # 시작 국면에는 탑이 없고 ▲大 가 ▽兵 위에 쌓을 수 있다.
    assert involves_stacking(GAME, GAME.initial_position())


def test_a_board_without_either_does_not() -> None:
    assert not involves_stacking(GAME, GAME.parse(PLAIN))
