"""`encode` 와 `canonical` 의 호출당 시간을 경로 길이별로 잰다.

uv run python -m scripts.encode_benchmark

시간은 컴퓨터마다 다르다. 전후 비교는 같은 컴퓨터에서 한다 (`docs/encode-speed.md`).
"""

from __future__ import annotations

import timeit
from collections.abc import Callable
from dataclasses import replace

from games.mini_gungi import MARSHAL_ONLY, MINI_GUNGI, Gungi, GungiPosition, VariantConfig

# 짝수 ply 는 선수 차례, 홀수 ply 는 후수 차례다.
PLIES = (0, 1, 20, 21, 80, 81, 198, 199)
REPEATS = 5

# 긴 경로를 만들려고 같은 네 수를 되풀이한다. 아무도 잡히지 않는 수들이다.
CASES: tuple[tuple[str, VariantConfig, tuple[str, ...]], ...] = (
    ("帥만 (말 2개)", MARSHAL_ONLY, ("c1-c2", "c5-c4", "c2-c1", "c4-c5")),
    ("미니 (말 10개)", MINI_GUNGI, ("b2-b3", "d4-d3", "b3-b2", "d3-d4")),
)


def positions(config: VariantConfig, cycle: tuple[str, ...]) -> list[tuple[Gungi, GungiPosition]]:
    """`PLIES` 의 국면들. 반복과 수 제한에 닿지 않게 한도만 늘린다."""
    game = Gungi(replace(config, move_limit=1000, repetition_limit=1000))
    position = game.initial_position()
    reached = [position]
    for number in range(max(PLIES)):
        position = game.apply_move(position, game.parse_move(cycle[number % len(cycle)]))
        reached.append(position)
    return [(game, reached[ply]) for ply in PLIES]


def microseconds(call: Callable[[], object]) -> float:
    """`REPEATS` 번 잰 것 중 가장 빠른 호출당 시간."""
    timer = timeit.Timer(call)
    number, _ = timer.autorange()
    return min(timer.repeat(REPEATS, number)) / number * 1e6


def main() -> int:
    print("| 판 | ply | 차례 | encode (µs) | canonical (µs) |")
    print("| --- | ---: | --- | ---: | ---: |", flush=True)
    for name, config, cycle in CASES:
        for game, position in positions(config, cycle):
            encode = microseconds(lambda g=game, p=position: g.encode(p))
            canonical = microseconds(lambda g=game, p=position: g.canonical(p))
            print(
                f"| {name} | {position.ply} | {position.player.value} "
                f"| {encode:.1f} | {canonical:.1f} |",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
