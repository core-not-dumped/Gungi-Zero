"""나노 군의. 미니 군의와 같은 엔진의 다른 설정이다."""

from __future__ import annotations

from dataclasses import replace

from games.mini_gungi import MINI_GUNGI, PieceType, Placement

# 완전 탐색할 수 있는 크기다. 상태 수와 고른 근거는 docs/nano-state-space.md.
NANO_GUNGI = replace(
    MINI_GUNGI,
    width=3,
    height=4,
    setup=(
        Placement(PieceType.GENERAL, column=2, row=1),
        Placement(PieceType.MARSHAL, column=1, row=0),
        Placement(PieceType.PAWN, column=2, row=0),
    ),
)
