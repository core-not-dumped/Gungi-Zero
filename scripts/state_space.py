"""나노 군의 후보마다 초기 국면에서 도달 가능한 상태 수를 센다.

uv run python -m scripts.state_space

`docs/nano-state-space.md` 의 "측정" 표가 이 출력에서 나온다.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace

from eval.reachable import count_reachable
from games.mini_gungi import (
    MINI_GUNGI,
    SYMBOLS,
    Gungi,
    PieceType,
    Placement,
    VariantConfig,
)
from games.nano_gungi import NANO_GUNGI

# 이보다 상태가 많으면 "완전 탐색 불가" 로 기록한다.
DEFAULT_LIMIT = 2_000_000


@dataclass(frozen=True)
class Candidate:
    """나노 군의 후보 하나. 판 크기와 말 구성만 미니 군의와 다르다."""

    config: VariantConfig
    # 거짓이면 세지 않는다. 상한을 넘을 것으로 추정할 뿐 확인하지 않은 후보다.
    measured: bool = True

    @property
    def name(self) -> str:
        pieces = "".join(
            SYMBOLS[placement.kind]
            for placement in sorted(self.config.setup, key=lambda p: p.kind.value)
        )
        return f"{self.config.width}x{self.config.height} {pieces}"

    @property
    def setup_text(self) -> str:
        """선수 배치. 후수는 그 180° 회전이다."""
        geometry = self.config.geometry
        return ", ".join(
            f"{SYMBOLS[p.kind]} {geometry.square_name(geometry.square(p.column, p.row))}"
            for p in self.config.setup
        )


def _candidate(
    width: int, height: int, *setup: tuple[PieceType, int, int], measured: bool = True
) -> Candidate:
    placements = tuple(Placement(kind, column=column, row=row) for kind, column, row in setup)
    config = replace(MINI_GUNGI, width=width, height=height, setup=placements)
    return Candidate(config, measured)


_MARSHAL, _GENERAL = PieceType.MARSHAL, PieceType.GENERAL
_LANCE, _PAWN = PieceType.LANCE, PieceType.PAWN

CANDIDATES = (
    _candidate(3, 3, (_MARSHAL, 1, 0), (_PAWN, 0, 0)),
    _candidate(3, 3, (_MARSHAL, 1, 0), (_GENERAL, 0, 0)),
    _candidate(3, 3, (_MARSHAL, 1, 0), (_PAWN, 0, 0), (_PAWN, 2, 0)),
    _candidate(3, 3, (_MARSHAL, 1, 0), (_GENERAL, 0, 0), (_PAWN, 2, 0)),
    _candidate(3, 4, (_MARSHAL, 1, 0), (_GENERAL, 0, 0)),
    _candidate(3, 4, (_MARSHAL, 1, 0), (_PAWN, 0, 1), (_PAWN, 2, 1)),
    Candidate(NANO_GUNGI),
    _candidate(3, 4, (_MARSHAL, 1, 0), (_LANCE, 0, 0), (_PAWN, 2, 1)),
    _candidate(4, 4, (_MARSHAL, 1, 0), (_GENERAL, 0, 0)),
    _candidate(3, 4, (_MARSHAL, 1, 0), (_GENERAL, 0, 0), (_LANCE, 2, 0), measured=False),
    _candidate(4, 4, (_MARSHAL, 1, 0), (_GENERAL, 0, 0), (_PAWN, 2, 1), measured=False),
    _candidate(
        3, 4, (_MARSHAL, 1, 0), (_GENERAL, 0, 0), (_LANCE, 2, 0), (_PAWN, 0, 1), measured=False
    ),
    # 비교 기준: 미니 군의 그대로.
    Candidate(MINI_GUNGI, measured=False),
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="나노 군의 후보의 상태 수 세기")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    args = parser.parse_args(argv)
    print(f"상한 {args.limit:,} 상태. 최대 2단, 배치는 선수 것이고 후수는 그 180° 회전이다.")
    print()
    print("| 후보 | 선수 배치 | 상태 수 | 끝난 상태 | 최대 깊이 | 완전 탐색 | 시간(초) |")
    print("| --- | --- | ---: | ---: | ---: | --- | ---: |", flush=True)
    for candidate in CANDIDATES:
        row = f"| {candidate.name} | {candidate.setup_text} |"
        if not candidate.measured:
            print(f"{row} - | - | - | 미측정(초과 추정) | - |", flush=True)
            continue
        started = time.perf_counter()
        counted = count_reachable(Gungi(candidate.config), args.limit)
        seconds = time.perf_counter() - started
        if counted.complete:
            figures = f"{counted.states:,} | {counted.terminal:,} | {counted.depth} | 가능"
        else:
            # 상한에서 멈췄으면 끝난 상태와 깊이는 일부만 본 값이라 싣지 않는다.
            figures = f"{counted.states:,} 초과 | - | - | 완전 탐색 불가"
        print(f"{row} {figures} | {seconds:.0f} |", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
