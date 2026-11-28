"""나노 군의. 2.5단계가 군의의 메커니즘을 검증하려면 그 메커니즘이 대국에 나와야 한다."""

from dataclasses import replace

from ai.random_play import play_random_episode
from games.base import Outcome
from games.mini_gungi import MINI_GUNGI, Gungi
from games.nano_gungi import NANO_GUNGI


def test_nano_differs_from_mini_only_in_board_and_pieces() -> None:
    same_board_and_pieces = replace(
        MINI_GUNGI, width=NANO_GUNGI.width, height=NANO_GUNGI.height, setup=NANO_GUNGI.setup
    )
    assert same_board_and_pieces == NANO_GUNGI


def test_the_action_space_and_planes_follow_the_mini_gungi_encoding() -> None:
    game = Gungi(NANO_GUNGI)
    assert game.action_space_size == 12 * 12 * 2
    assert game.tensor_shape == (18, 4, 3)


def test_the_initial_position() -> None:
    game = Gungi(NANO_GUNGI)
    assert game.render(game.initial_position()) == (
        "     a       b       c\n"
        " 4  ▽兵     ▽帥      .\n"
        " 3  ▽大      .       .\n"
        " 2   .       .      ▲大\n"
        " 1   .      ▲帥     ▲兵\n"
        "차례: ▲   ply: 0\n"
    )


# 두 帥 가 한 칸 나갔다가 돌아온다. 네 수마다 같은 판, 같은 차례가 된다.
ROUND_TRIP = ("b1-b2", "b4-b3", "b2-b1", "b3-b4")


def test_a_repetition_draw_is_not_decided_by_the_board() -> None:
    game = Gungi(NANO_GUNGI)
    position = game.initial_position()
    for text in ROUND_TRIP * 2:
        assert game.outcome(position) is None
        position = game.apply_move(position, game.parse_move(text))
    assert game.outcome(position) is Outcome.DRAW
    assert game.board_outcome(position) is None


def test_a_move_limit_draw_is_not_decided_by_the_board() -> None:
    game = Gungi(replace(NANO_GUNGI, move_limit=len(ROUND_TRIP)))
    position = game.initial_position()
    for text in ROUND_TRIP:
        position = game.apply_move(position, game.parse_move(text))
    assert game.outcome(position) is Outcome.DRAW
    assert game.board_outcome(position) is None


def test_random_episodes_end_and_use_every_tower_mechanic() -> None:
    game = Gungi(NANO_GUNGI)
    seen: set[str] = set()
    wanted = {"2단의 말이 움직인다", "아군 위에 쌓기", "적 위에 쌓기", "2단 잡기"}
    # 2단 잡기는 드물다 (docs/nano-state-space.md). 다 볼 때까지 두되 시드 1000 에서 포기한다.
    for seed in range(1000):
        episode = play_random_episode(game, seed)
        assert game.outcome(episode.final_position) is episode.outcome
        position = game.initial_position()
        for move in episode.moves:
            text = game.render_move(move)
            source, target = (text[:2], text[3:5])
            grid = [line.split()[1:] for line in game.render(position).splitlines()[1:-1]]
            cells = {
                f"{'abc'[column]}{len(grid) - line}": cell
                for line, row in enumerate(grid)
                for column, cell in enumerate(row)
            }
            if "/" in cells[source]:
                seen.add("2단의 말이 움직인다")
            if move.stack:
                friendly = cells[target].startswith(position.player.value)
                seen.add("아군 위에 쌓기" if friendly else "적 위에 쌓기")
            if game.render_log_move(position, move).endswith("x") and "/" in cells[target]:
                seen.add("2단 잡기")
            position = game.apply_move(position, move)
        if seen == wanted:
            break
    assert seen == wanted
