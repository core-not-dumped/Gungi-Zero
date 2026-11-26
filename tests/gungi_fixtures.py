"""미니 군의의 규칙별 fixture. 테스트와 사람 검토 문서가 함께 쓰는 하나의 출처다.

`tests/test_gungi_rules.py` 가 룰 엔진에 대어 보고, `scripts/fixture_review.py` 가
`docs/fixture-review.md` 를 만든다. 고친 뒤에는 문서를 다시 만든다.

모든 fixture 는 `games.mini_gungi.MINI_GUNGI` 설정이다. 판의 탑은 `상단/하단` 이다.
판·기대 합법수·결과는 사람이 읽고 검토하는 값이므로 엔진의 출력을 옮겨 적어 맞추지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass

from games.base import Outcome


@dataclass(frozen=True)
class Then:
    """fixture 의 판에서 `moves` 를 차례로 둔 뒤의 기대."""

    moves: str
    board: str
    outcome: Outcome | None = None
    # 둔 뒤의 합법수. `None` 이면 확인하지 않는다.
    legal: str | None = None
    note: str = ""


@dataclass(frozen=True)
class Fixture:
    name: str
    # 이 fixture 가 검증하는 것, 한 줄.
    checks: str
    board: str
    # 차례인 쪽의 합법수 전부. 공백으로 나눈다.
    legal: str
    outcome: Outcome | None = None
    then: tuple[Then, ...] = ()
    # 검토자가 눈여겨볼 점. 합법수 아래에 실린다.
    note: str = ""


@dataclass(frozen=True)
class Section:
    """채택한 규칙 하나와 그 fixture 들."""

    title: str
    rule: str
    # 근거의 강도. 검토할 때 어디를 의심해야 하는지 알려준다.
    source: str
    fixtures: tuple[Fixture, ...]


# ── 초기 배치 ──

INITIAL = Section(
    title="초기 배치",
    rule="5x5 판에 편마다 帥·大·槍 하나와 兵 둘. 후수의 배치는 선수 배치의 180° 회전이다.",
    source="발명 (ADR-0003). 배치와 말 개수는 측정으로 확정 (docs/measurement-report.md)",
    fixtures=(
        Fixture(
            name="initial-position",
            checks="시작 판과 선수의 첫 수 전부.",
            board="""\
     a       b       c       d       e
 5   .      ▽槍     ▽帥     ▽大      .
 4   .      ▽兵      .      ▽兵      .
 3   .       .       .       .       .
 2   .      ▲兵      .      ▲兵      .
 1   .      ▲大     ▲帥     ▲槍      .
차례: ▲   ply: 0
""",
            legal=("b1-a1 b1-a2 b1-c2 b1-b2^ c1-c2 d1-c2 d1-e2 d1-d2^ b2-b3 b2-b1^ d2-d3 d2-d1^"),
        ),
    ),
)

# ── 1단 이동표 ──

MARSHAL_MOVES = Section(
    title="帥 의 이동",
    rule="8방향으로 1칸. 판 밖으로는 못 간다.",
    source="2차 자료만",
    fixtures=(
        Fixture(
            name="marshal-center",
            checks="판 가운데의 帥 는 여덟 칸에 간다.",
            board="""\
     a       b       c       d       e
 5   .       .       .       .      ▽帥
 4   .       .       .       .       .
 3   .       .      ▲帥      .       .
 2   .       .       .       .       .
 1   .       .       .       .       .
차례: ▲   ply: 0
""",
            legal="c3-b2 c3-c2 c3-d2 c3-b3 c3-d3 c3-b4 c3-c4 c3-d4",
        ),
        Fixture(
            name="marshal-edge",
            checks="가장자리의 帥 는 반대편으로 넘어가지 않는다.",
            board="""\
     a       b       c       d       e
 5   .       .       .       .       .
 4   .       .       .       .       .
 3   .       .       .       .      ▽帥
 2   .       .       .       .       .
 1  ▲帥      .       .       .       .
차례: ▽   ply: 1
""",
            legal="e3-d2 e3-e2 e3-d3 e3-d4 e3-e4",
        ),
    ),
)

GENERAL_MOVES = Section(
    title="大 의 이동",
    rule="직선(상하좌우)은 무제한, 대각은 1칸.",
    source="2차 자료만",
    fixtures=(
        Fixture(
            name="general-center",
            checks="직선은 판 끝까지, 대각은 한 칸.",
            board="""\
     a       b       c       d       e
 5   .       .       .       .      ▽帥
 4   .       .       .       .       .
 3   .       .      ▲大      .       .
 2   .       .       .       .       .
 1  ▲帥      .       .       .       .
차례: ▲   ply: 0
""",
            legal=(
                "c3-c4 c3-c5 c3-c2 c3-c1 c3-b3 c3-a3 c3-d3 c3-e3 "
                "c3-b4 c3-d4 c3-b2 c3-d2 "
                "a1-a2 a1-b1 a1-b2"
            ),
        ),
    ),
)

LANCE_MOVES = Section(
    title="槍 의 이동",
    rule="전방 2칸, 전방대각 1칸, 후방 1칸. 측면과 후방대각은 없다. 전방은 상대 진영 쪽이다.",
    source="2차 자료만",
    fixtures=(
        Fixture(
            name="lance-first",
            checks="선수(▲)의 전방은 5 행 쪽이다.",
            board="""\
     a       b       c       d       e
 5   .       .       .       .      ▽帥
 4   .       .       .       .       .
 3   .       .       .       .       .
 2   .       .      ▲槍      .       .
 1  ▲帥      .       .       .       .
차례: ▲   ply: 0
""",
            legal="c2-c3 c2-c4 c2-b3 c2-d3 c2-c1 a1-a2 a1-b1 a1-b2",
        ),
        Fixture(
            name="lance-second",
            checks="후수(▽)의 전방은 1 행 쪽이다.",
            board="""\
     a       b       c       d       e
 5   .       .       .       .      ▽帥
 4   .       .      ▽槍      .       .
 3   .       .       .       .       .
 2   .       .       .       .       .
 1  ▲帥      .       .       .       .
차례: ▽   ply: 1
""",
            legal="c4-c3 c4-c2 c4-d3 c4-b3 c4-c5 e5-e4 e5-d5 e5-d4",
        ),
    ),
)

PAWN_BOARD = """\
     a       b       c       d       e
 5   .       .       .       .      ▽帥
 4   .       .      ▽兵      .       .
 3   .       .       .       .       .
 2   .       .      ▲兵      .       .
 1  ▲帥      .       .       .       .
"""

PAWN_MOVES = Section(
    title="兵 의 이동",
    rule="전방 또는 후방으로 1칸.",
    source="2차 자료만",
    fixtures=(
        Fixture(
            name="pawn-first",
            checks="선수의 兵 은 앞뒤 한 칸씩만 간다.",
            board=PAWN_BOARD + "차례: ▲   ply: 0\n",
            legal="c2-c3 c2-c1 a1-a2 a1-b1 a1-b2",
        ),
        Fixture(
            name="pawn-second",
            checks="후수의 兵 도 앞뒤 한 칸씩만 간다.",
            board=PAWN_BOARD + "차례: ▽   ply: 1\n",
            legal="c4-c3 c4-c5 e5-e4 e5-d5 e5-d4",
        ),
    ),
)

# ── 막힘 ──

BLOCKING = Section(
    title="막힘",
    rule=(
        "말이 있는 칸은 단수와 무관하게 길을 막는다. 그 칸까지는 갈 수 있으면 가고"
        "(잡기·쌓기), 그 너머로는 못 간다. 槍 의 전방 2칸도 뛰어넘지 않는다."
    ),
    source=(
        "뛰어넘기는 砲·筒·弓 만 된다는 원칙은 공식 Q&A. 槍 에의 적용은 거기서 유도. "
        "일부 2차 자료는 '더 높은 단만 막는다' 로 요약하며 채택하지 않았다 (ADR-0002)"
    ),
    fixtures=(
        Fixture(
            name="blocking-all-around",
            checks="아군·적에게 둘러싸인 大 와 槍 은 첫 번째 말에서 멈춘다.",
            board="""\
     a       b       c       d       e
 5   .       .       .       .      ▽帥
 4   .       .      ▽兵      .       .
 3   .      ▲兵     ▲大     ▽槍      .
 2   .      ▲兵      .      ▽兵      .
 1  ▲帥      .      ▲槍      .       .
차례: ▲   ply: 0
""",
            legal=(
                "c3-c4 c3-c4^ c3-d3 c3-d3^ c3-c2 c3-c1^ c3-b3^ "
                "c3-b4 c3-d4 c3-b2^ c3-d2 c3-d2^ "
                "c1-c2 c1-c3^ c1-b2^ c1-d2 c1-d2^ "
                "b3-b4 b3-b2^ "
                "b2-b3^ b2-b1 "
                "a1-a2 a1-b1"
            ),
            note="c5, e3, a3 처럼 말 너머의 칸은 없다.",
        ),
        Fixture(
            name="lance-blocked-by-near-friend",
            checks="槍 은 바로 앞의 아군을 뛰어넘지 못한다. 그 위에 쌓기만 된다.",
            board="""\
     a       b       c       d       e
 5   .       .       .       .      ▽帥
 4   .       .       .       .       .
 3   .       .      ▲兵      .       .
 2   .       .      ▲槍      .       .
 1  ▲帥      .       .       .       .
차례: ▲   ply: 0
""",
            legal="c2-c3^ c2-b3 c2-d3 c2-c1 c3-c4 c3-c2^ a1-a2 a1-b1 a1-b2",
        ),
        Fixture(
            name="lance-blocked-by-near-enemy",
            checks="槍 은 바로 앞의 적을 잡거나 그 위에 쌓고 거기서 멈춘다.",
            board="""\
     a       b       c       d       e
 5   .       .       .       .      ▽帥
 4   .       .       .       .       .
 3   .       .      ▽兵      .       .
 2   .       .      ▲槍      .       .
 1  ▲帥      .       .       .       .
차례: ▲   ply: 0
""",
            legal="c2-c3 c2-c3^ c2-b3 c2-d3 c2-c1 a1-a2 a1-b1 a1-b2",
        ),
        Fixture(
            name="lance-reaches-far-friend",
            checks="槍 은 두 칸 앞의 아군 위에 쌓을 수 있다.",
            board="""\
     a       b       c       d       e
 5   .       .       .       .      ▽帥
 4   .       .      ▲兵      .       .
 3   .       .       .       .       .
 2   .       .      ▲槍      .       .
 1  ▲帥      .       .       .       .
차례: ▲   ply: 0
""",
            legal="c2-c3 c2-c4^ c2-b3 c2-d3 c2-c1 c4-c3 c4-c5 a1-a2 a1-b1 a1-b2",
        ),
        Fixture(
            name="lance-reaches-far-enemy",
            checks="槍 은 두 칸 앞의 적을 잡거나 그 위에 쌓을 수 있다.",
            board="""\
     a       b       c       d       e
 5   .       .       .       .      ▽帥
 4   .       .      ▽兵      .       .
 3   .       .       .       .       .
 2   .       .      ▲槍      .       .
 1  ▲帥      .       .       .       .
차례: ▲   ply: 0
""",
            legal="c2-c3 c2-c4 c2-c4^ c2-b3 c2-d3 c2-c1 a1-a2 a1-b1 a1-b2",
        ),
        Fixture(
            name="second-tier-blocked-by-first-tier",
            checks="2단의 大 도 1단 말을 넘어가지 못한다.",
            board="""\
     a       b       c       d       e
 5   .       .       .       .      ▽帥
 4   .       .       .       .       .
 3   .       .      ▽兵      .       .
 2   .       .    ▲大/▲兵    .       .
 1  ▲帥      .       .       .       .
차례: ▲   ply: 4
""",
            legal=(
                "c2-c3 c2-c3^ c2-c1 c2-b2 c2-a2 c2-d2 c2-e2 "
                "c2-b3 c2-a4 c2-d3 c2-e4 c2-b1 c2-d1 "
                "a1-a2 a1-b1 a1-b2"
            ),
            note="大 의 전방은 무제한이지만 c4, c5 에는 못 간다.",
        ),
        Fixture(
            name="second-tier-lance-does-not-jump",
            checks="사거리가 3칸으로 늘어난 2단 槍 도 중간의 말에서 멈춘다.",
            board="""\
     a       b       c       d       e
 5   .      ▽帥      .       .       .
 4   .       .      ▽兵      .       .
 3   .       .       .       .       .
 2   .       .    ▲槍/▲兵    .       .
 1  ▲帥      .       .       .       .
차례: ▲   ply: 4
""",
            legal="c2-c3 c2-c4 c2-c4^ c2-b3 c2-a4 c2-d3 c2-e4 c2-c1 a1-a2 a1-b1 a1-b2",
            note="c5 는 사거리 안이지만 c4 에 막힌다.",
        ),
    ),
)

# ── 쌓기 ──

ENEMY_IN_FRONT = """\
     a       b       c       d       e
 5   .       .       .       .      ▽帥
 4   .       .       .       .       .
 3   .       .      ▽兵      .       .
 2   .       .      ▲兵      .       .
 1  ▲帥      .       .       .       .
차례: ▲   ply: 4
"""

STACKING = Section(
    title="쌓기와 잡기의 선택",
    rule=(
        "맨 위가 적 말인 칸에서는 잡기(`c2-c3`)와 쌓기(`c2-c3^`) 중에서 고른다. "
        "아군 칸에는 쌓기만 된다. 탑은 최대 2단이다."
    ),
    source="선택 가능은 공식 Q&A (ADR-0006). 2단 상한은 주로 2차 자료",
    fixtures=(
        Fixture(
            name="enemy-capture-or-stack",
            checks="적 위에 쌓으면 적은 아래에 남고, 잡으면 사라진다.",
            board=ENEMY_IN_FRONT,
            legal="c2-c3 c2-c3^ c2-c1 a1-a2 a1-b1 a1-b2",
            then=(
                Then(
                    moves="c2-c3^",
                    board="""\
     a       b       c       d       e
 5   .       .       .       .      ▽帥
 4   .       .       .       .       .
 3   .       .    ▲兵/▽兵    .       .
 2   .       .       .       .       .
 1  ▲帥      .       .       .       .
차례: ▽   ply: 5
""",
                ),
                Then(
                    moves="c2-c3",
                    board="""\
     a       b       c       d       e
 5   .       .       .       .      ▽帥
 4   .       .       .       .       .
 3   .       .      ▲兵      .       .
 2   .       .       .       .       .
 1  ▲帥      .       .       .       .
차례: ▽   ply: 5
""",
                ),
            ),
        ),
        Fixture(
            name="friend-stack-only",
            checks="아군 칸에는 쌓기만 있다.",
            board="""\
     a       b       c       d       e
 5   .       .       .       .      ▽帥
 4   .       .       .       .       .
 3   .       .      ▲兵      .       .
 2   .       .      ▲兵      .       .
 1  ▲帥      .       .       .       .
차례: ▲   ply: 4
""",
            legal="c2-c3^ c2-c1 c3-c4 c3-c2^ a1-a2 a1-b1 a1-b2",
            then=(
                Then(
                    moves="c2-c3^",
                    board="""\
     a       b       c       d       e
 5   .       .       .       .      ▽帥
 4   .       .       .       .       .
 3   .       .    ▲兵/▲兵    .       .
 2   .       .       .       .       .
 1  ▲帥      .       .       .       .
차례: ▽   ply: 5
""",
                ),
            ),
        ),
    ),
)

TIER_LIMIT = Section(
    title="단수 제약",
    rule=(
        "자기보다 높은 단의 말은 잡을 수도 쌓을 수도 없다. 낮은 곳에는 내려가며 쌓을 수 있다. "
        "맨 위가 아군인 탑은 아래에 적이 있어도 잡을 수 없다."
    ),
    source=(
        "잡기 제한은 발매사 특집 기사, 쌓기 제한은 2차 자료, 맨 위가 아군이면 못 잡는 것은 공식 Q&A"
    ),
    fixtures=(
        Fixture(
            name="higher-piece-stacks-down",
            checks="2단 말은 1단 아군 위에 쌓을 수 있고, 1단 말은 2단 탑에 못 올라간다.",
            board="""\
     a       b       c       d       e
 5   .       .       .       .      ▽帥
 4   .       .       .       .       .
 3   .       .      ▲兵      .       .
 2   .       .    ▲兵/▲大    .       .
 1  ▲帥      .       .       .       .
차례: ▲   ply: 4
""",
            legal="c2-c3^ c2-c1 c3-c4 a1-a2 a1-b1 a1-b2",
            note="c3-c2^ 가 없다.",
        ),
        Fixture(
            name="higher-enemy-tower",
            checks="1단 말은 2단의 적 탑을 잡지도 쌓지도 못한다.",
            board="""\
     a       b       c       d       e
 5   .       .       .       .      ▽帥
 4   .       .       .       .       .
 3   .       .    ▽兵/▽槍    .       .
 2   .       .      ▲兵      .       .
 1  ▲帥      .       .       .       .
차례: ▲   ply: 4
""",
            legal="c2-c1 a1-a2 a1-b1 a1-b2",
        ),
        Fixture(
            name="higher-mixed-tower",
            checks="2단 탑의 아래가 아군이어도 1단 말은 손댈 수 없다.",
            board="""\
     a       b       c       d       e
 5   .       .       .       .      ▽帥
 4   .       .       .       .       .
 3   .       .    ▽兵/▲槍    .       .
 2   .       .      ▲兵      .       .
 1  ▲帥      .       .       .       .
차례: ▲   ply: 4
""",
            legal="c2-c1 a1-a2 a1-b1 a1-b2",
        ),
        Fixture(
            name="full-tower-capture-only",
            checks="같은 2단이면 꽉 찬 적 탑을 잡을 수 있지만 3단으로 쌓을 수는 없다.",
            board="""\
     a       b       c       d       e
 5   .       .       .       .      ▽帥
 4   .       .       .       .       .
 3   .       .    ▽兵/▽槍    .       .
 2   .       .    ▲兵/▲大    .       .
 1  ▲帥      .       .       .       .
차례: ▲   ply: 4
""",
            legal="c2-c3 c2-c1 a1-a2 a1-b1 a1-b2",
        ),
        Fixture(
            name="friend-on-top",
            checks="맨 위가 아군인 탑은 아래의 적을 잡을 수 없다. 꽉 차서 쌓을 수도 없다.",
            board="""\
     a       b       c       d       e
 5   .       .       .       .      ▽帥
 4   .       .       .       .       .
 3   .       .    ▲兵/▽兵    .       .
 2   .       .    ▲兵/▲大    .       .
 1  ▲帥      .       .       .       .
차례: ▲   ply: 4
""",
            legal="c3-c4 c3-c5 c2-c1 a1-a2 a1-b1 a1-b2",
            note="두 ▲兵 사이에는 어느 쪽으로도 수가 없다.",
        ),
    ),
)

REACH = Section(
    title="2단의 사거리",
    rule="2단의 말은 유한 방향 사거리가 1칸씩 늘어난다. 무제한 방향은 그대로다.",
    source="2차 자료만",
    fixtures=(
        Fixture(
            name="second-tier-pawn",
            checks="2단 兵: 앞뒤 2칸.",
            board="""\
     a       b       c       d       e
 5   .      ▽帥      .       .       .
 4   .       .       .       .       .
 3   .       .    ▲兵/▲大    .       .
 2   .       .       .       .       .
 1  ▲帥      .       .       .       .
차례: ▲   ply: 4
""",
            legal="c3-c4 c3-c5 c3-c2 c3-c1 a1-a2 a1-b1 a1-b2",
        ),
        Fixture(
            name="second-tier-lance-back",
            checks="2단 槍: 전방대각 2칸, 후방 2칸. 전방 3칸 중 판 안은 2칸이다.",
            board="""\
     a       b       c       d       e
 5   .      ▽帥      .       .       .
 4   .       .       .       .       .
 3   .       .    ▲槍/▲兵    .       .
 2   .       .       .       .       .
 1  ▲帥      .       .       .       .
차례: ▲   ply: 4
""",
            legal="c3-c4 c3-c5 c3-b4 c3-a5 c3-d4 c3-e5 c3-c2 c3-c1 a1-a2 a1-b1 a1-b2",
        ),
        Fixture(
            name="second-tier-lance-forward",
            checks="2단 槍: 전방 3칸. 후방 2칸 중 판 안은 1칸이다.",
            board="""\
     a       b       c       d       e
 5   .      ▽帥      .       .       .
 4   .       .       .       .       .
 3   .       .       .       .       .
 2   .       .    ▲槍/▲兵    .       .
 1  ▲帥      .       .       .       .
차례: ▲   ply: 4
""",
            legal="c2-c3 c2-c4 c2-c5 c2-b3 c2-a4 c2-d3 c2-e4 c2-c1 a1-a2 a1-b1 a1-b2",
        ),
        Fixture(
            name="second-tier-general",
            checks="2단 大: 직선은 그대로 무제한, 대각 2칸.",
            board="""\
     a       b       c       d       e
 5   .      ▽帥      .       .       .
 4   .       .       .       .       .
 3   .       .    ▲大/▲兵    .       .
 2   .       .       .       .       .
 1   .      ▲帥      .       .       .
차례: ▲   ply: 4
""",
            legal=(
                "c3-c4 c3-c5 c3-c2 c3-c1 c3-b3 c3-a3 c3-d3 c3-e3 "
                "c3-b4 c3-a5 c3-d4 c3-e5 c3-b2 c3-a1 c3-d2 c3-e1 "
                "b1-a1 b1-c1 b1-a2 b1-b2 b1-c2"
            ),
        ),
        Fixture(
            name="second-tier-marshal",
            checks=(
                "2단 帥: 8방향 2칸. 帥ツケ 금지 때문에 실제 대국에서는 나올 수 없는 판이다 "
                "(검토 질문 1)."
            ),
            board="""\
     a       b       c       d       e
 5   .      ▽帥      .       .       .
 4   .       .       .       .       .
 3   .       .    ▲帥/▲兵    .       .
 2   .       .       .       .       .
 1   .       .       .       .       .
차례: ▲   ply: 4
""",
            legal=(
                "c3-c4 c3-c5 c3-c2 c3-c1 c3-b3 c3-a3 c3-d3 c3-e3 "
                "c3-b4 c3-a5 c3-d4 c3-e5 c3-b2 c3-a1 c3-d2 c3-e1"
            ),
        ),
    ),
)

FROZEN_BOARD = """\
     a       b       c       d       e
 5   .       .       .       .      ▽帥
 4   .       .       .       .       .
 3   .       .    ▽兵/▲大    .       .
 2   .       .       .       .       .
 1  ▲帥      .       .       .       .
"""

TOP_ONLY = Section(
    title="맨 위만 이동",
    rule="탑의 맨 위 말만 움직인다. 아래 말은 얼어 있고, 위가 비키면 다시 움직인다.",
    source="공식 Q&A. 다시 움직이는 것은 거기서 유도",
    fixtures=(
        Fixture(
            name="buried-piece-is-frozen",
            checks="▽兵 아래의 ▲大 는 움직일 수 없다.",
            board=FROZEN_BOARD + "차례: ▲   ply: 4\n",
            legal="a1-a2 a1-b1 a1-b2",
        ),
        Fixture(
            name="buried-piece-is-freed",
            checks="위의 ▽兵 이 비키면 ▲大 가 1단 말로 다시 움직인다.",
            board=FROZEN_BOARD + "차례: ▽   ply: 5\n",
            legal="c3-c2 c3-c1 c3-c4 c3-c5 e5-e4 e5-d5 e5-d4",
            note="c3 의 ▽兵 은 2단이라 앞뒤 2칸이다.",
            then=(
                Then(
                    moves="c3-c4",
                    board="""\
     a       b       c       d       e
 5   .       .       .       .      ▽帥
 4   .       .      ▽兵      .       .
 3   .       .      ▲大      .       .
 2   .       .       .       .       .
 1  ▲帥      .       .       .       .
차례: ▲   ply: 6
""",
                    legal=(
                        "c3-c4 c3-c4^ c3-c2 c3-c1 c3-b3 c3-a3 c3-d3 c3-e3 "
                        "c3-b4 c3-d4 c3-b2 c3-d2 "
                        "a1-a2 a1-b1 a1-b2"
                    ),
                ),
            ),
        ),
    ),
)

TOWER_CAPTURE = Section(
    title="탑 잡기",
    rule="탑을 잡으면 그 칸의 적 말이 전부 잡히고 내 말은 남는다. 잡힌 말은 다시 쓰지 않는다.",
    source="공식 Q&A. 재사용 없음은 2차 자료",
    fixtures=(
        Fixture(
            name="capture-enemy-tower",
            checks="적 말 둘로 된 탑을 잡으면 둘 다 사라진다.",
            board="""\
     a       b       c       d       e
 5   .       .       .       .      ▽帥
 4   .       .       .       .       .
 3   .       .    ▽兵/▽槍    .       .
 2   .       .    ▲兵/▲大    .       .
 1  ▲帥      .       .       .       .
차례: ▲   ply: 4
""",
            legal="c2-c3 c2-c1 a1-a2 a1-b1 a1-b2",
            then=(
                Then(
                    moves="c2-c3",
                    board="""\
     a       b       c       d       e
 5   .       .       .       .      ▽帥
 4   .       .       .       .       .
 3   .       .      ▲兵      .       .
 2   .       .      ▲大      .       .
 1  ▲帥      .       .       .       .
차례: ▽   ply: 5
""",
                ),
            ),
        ),
        Fixture(
            name="capture-keeps-my-piece",
            checks="탑 아래의 내 말은 남고, 잡은 말이 그 위에 선다.",
            board="""\
     a       b       c       d       e
 5   .       .       .       .      ▽帥
 4   .       .       .       .       .
 3   .       .    ▽兵/▲槍    .       .
 2   .       .    ▲兵/▲大    .       .
 1  ▲帥      .       .       .       .
차례: ▲   ply: 4
""",
            legal="c2-c3 c2-c1 a1-a2 a1-b1 a1-b2",
            then=(
                Then(
                    moves="c2-c3",
                    board="""\
     a       b       c       d       e
 5   .       .       .       .      ▽帥
 4   .       .       .       .       .
 3   .       .    ▲兵/▲槍    .       .
 2   .       .      ▲大      .       .
 1  ▲帥      .       .       .       .
차례: ▽   ply: 5
""",
                ),
            ),
        ),
    ),
)

NOTHING_ON_MARSHAL = Section(
    title="帥 위 쌓기 금지",
    rule="帥 위에는 아군이든 적이든 아무것도 쌓을 수 없다. 적 帥 는 잡기만 된다.",
    source="공식 Q&A",
    fixtures=(
        Fixture(
            name="nothing-on-marshals",
            checks="어느 편의 帥 위에도 올라갈 수 없다.",
            board="""\
     a       b       c       d       e
 5   .       .       .       .       .
 4   .       .       .       .       .
 3   .       .      ▽帥      .       .
 2  ▲兵      .      ▲兵      .       .
 1  ▲帥      .       .       .       .
차례: ▲   ply: 4
""",
            legal="c2-c3 c2-c1 a2-a3 a1-b1 a1-b2",
            note="c2-c3^ 와 a2-a1^ 가 없다.",
        ),
    ),
)

MARSHAL_NEVER_STACKS = Section(
    title="帥ツケ 금지",
    rule=("帥 자신은 아군 위에도 적 위에도 올라가지 않는다. '帥 위 쌓기 금지' 와 다른 규칙이다."),
    source="공식 Q&A. 입문·초급 규칙이며 중급·상급은 허용한다",
    fixtures=(
        Fixture(
            name="marshal-never-stacks",
            checks="帥 는 적 칸에서 잡기만 하고, 아군 칸으로는 갈 수 없다.",
            board="""\
     a       b       c       d       e
 5   .       .       .       .      ▽帥
 4   .       .       .       .       .
 3   .       .       .       .       .
 2  ▲兵     ▽兵      .       .       .
 1  ▲帥      .       .       .       .
차례: ▲   ply: 4
""",
            legal="a1-b2 a1-b1 a2-a3",
        ),
    ),
)

# ── 종료 규칙 ──

FACING_MARSHALS = """\
     a       b       c       d       e
 5   .       .       .       .       .
 4   .       .       .       .       .
 3   .       .      ▽帥      .       .
 2   .       .      ▲帥      .       .
 1   .       .       .       .       .
"""

MARSHAL_CAPTURE = Section(
    title="帥 를 잡으면 승리",
    rule="상대의 帥 를 잡은 쪽이 이긴다. 수 제한에 닿는 마지막 수로 잡아도 승리다.",
    source="승리 조건은 공식 Q&A. 수 제한보다 먼저 보는 순서는 발명 (ADR-0004)",
    fixtures=(
        Fixture(
            name="first-captures-marshal",
            checks="선수가 帥 를 잡으면 선수 승.",
            board=FACING_MARSHALS + "차례: ▲   ply: 4\n",
            legal="c2-b1 c2-c1 c2-d1 c2-b2 c2-d2 c2-b3 c2-c3 c2-d3",
            then=(
                Then(
                    moves="c2-c3",
                    board="""\
     a       b       c       d       e
 5   .       .       .       .       .
 4   .       .       .       .       .
 3   .       .      ▲帥      .       .
 2   .       .       .       .       .
 1   .       .       .       .       .
차례: ▽   ply: 5
""",
                    outcome=Outcome.FIRST_WINS,
                ),
            ),
        ),
        Fixture(
            name="second-captures-marshal-on-last-move",
            checks="200 번째 수로 帥 를 잡으면 무승부가 아니라 후수 승.",
            board=FACING_MARSHALS + "차례: ▽   ply: 199\n",
            legal="c3-b2 c3-c2 c3-d2 c3-b3 c3-d3 c3-b4 c3-c4 c3-d4",
            then=(
                Then(
                    moves="c3-c2",
                    board="""\
     a       b       c       d       e
 5   .       .       .       .       .
 4   .       .       .       .       .
 3   .       .       .       .       .
 2   .       .      ▽帥      .       .
 1   .       .       .       .       .
차례: ▲   ply: 200
""",
                    outcome=Outcome.SECOND_WINS,
                ),
            ),
        ),
    ),
)

MOVE_LIMIT = Section(
    title="200수 무승부",
    rule="200 번째 수까지 승부가 나지 않으면 무승부다.",
    source="발명 (ADR-0004)",
    fixtures=(
        Fixture(
            name="move-limit",
            checks="199 수까지는 진행 중이고 200 번째 수를 두면 끝난다.",
            board="""\
     a       b       c       d       e
 5   .       .      ▽帥      .       .
 4   .       .       .       .       .
 3   .       .       .       .       .
 2   .       .       .       .       .
 1   .       .      ▲帥      .       .
차례: ▽   ply: 199
""",
            legal="c5-b5 c5-b4 c5-c4 c5-d4 c5-d5",
            then=(
                Then(
                    moves="c5-b5",
                    board="""\
     a       b       c       d       e
 5   .      ▽帥      .       .       .
 4   .       .       .       .       .
 3   .       .       .       .       .
 2   .       .       .       .       .
 1   .       .      ▲帥      .       .
차례: ▲   ply: 200
""",
                    outcome=Outcome.DRAW,
                ),
            ),
        ),
    ),
)

TWO_MARSHALS = """\
     a       b       c       d       e
 5   .       .      ▽帥      .       .
 4   .       .       .       .       .
 3   .       .       .       .       .
 2   .       .       .       .       .
 1   .       .      ▲帥      .       .
"""

# 두 帥 가 한 칸 갔다가 돌아온다. 네 수마다 같은 판, 같은 차례가 된다.
ROUND_TRIP = "c1-b1 c5-b5 b1-c1 b5-c5"

REPETITION = Section(
    title="같은 판 3회 반복 무승부",
    rule="판 배치와 차례가 모두 같은 국면이 세 번째 나타나면 무승부다. 처음 판도 한 번으로 센다.",
    source="발명 (ADR-0004). 2차 자료의 공식 규칙은 4회 뒤 재대국이다",
    fixtures=(
        Fixture(
            name="third-appearance",
            checks="7 수까지는 진행 중이고, 8 번째 수로 시작 판이 세 번째가 되면 끝난다.",
            board=TWO_MARSHALS + "차례: ▲   ply: 0\n",
            legal="c1-b1 c1-b2 c1-c2 c1-d2 c1-d1",
            then=(
                Then(
                    moves=f"{ROUND_TRIP} c1-b1 c5-b5 b1-c1",
                    board="""\
     a       b       c       d       e
 5   .      ▽帥      .       .       .
 4   .       .       .       .       .
 3   .       .       .       .       .
 2   .       .       .       .       .
 1   .       .      ▲帥      .       .
차례: ▽   ply: 7
""",
                    legal="b5-a5 b5-a4 b5-b4 b5-c4 b5-c5",
                ),
                Then(
                    moves=f"{ROUND_TRIP} {ROUND_TRIP}",
                    board=TWO_MARSHALS + "차례: ▲   ply: 8\n",
                    outcome=Outcome.DRAW,
                ),
            ),
        ),
        Fixture(
            name="other-side-to-move",
            checks="같은 배치가 세 번 나와도 차례가 다른 것은 따로 센다.",
            board=TWO_MARSHALS + "차례: ▲   ply: 0\n",
            legal="c1-b1 c1-b2 c1-c2 c1-d2 c1-d1",
            then=(
                Then(
                    moves=f"{ROUND_TRIP} c1-b1 c5-b5 b1-b2 b5-c5 b2-c1",
                    board=TWO_MARSHALS + "차례: ▽   ply: 9\n",
                    legal="c5-b5 c5-b4 c5-c4 c5-d4 c5-d5",
                    note=(
                        "▲ 는 세 수, ▽ 는 두 수에 제자리로 와서 같은 배치가 ▽ 차례로 나타난다. "
                        "시작 배치는 0·4 수 뒤(▲ 차례)와 9 수 뒤(▽ 차례)에 나왔다."
                    ),
                ),
            ),
        ),
    ),
)

# 帥 의 이웃 세 칸이 모두 2단 탑이고 맨 위가 적이다. 1단의 帥 는 잡을 수 없다.
FIRST_IS_STUCK = """\
     a       b       c       d       e
 5   .       .       .       .      ▽帥
 4   .       .       .       .       .
 3   .       .       .       .       .
 2 ▽兵/▲兵 ▽槍/▲槍   .       .       .
 1  ▲帥   ▽兵/▲兵    .       .       .
"""

NO_LEGAL_MOVE = Section(
    title="합법수가 없으면 패배",
    rule="둘 차례인 쪽에 합법수가 없으면 진다. 수 제한에 닿은 국면이어도 패배가 먼저다.",
    source="발명 (ADR-0004). 공식 규칙은 이 상황을 정의하지 않는다",
    fixtures=(
        Fixture(
            name="first-has-no-move",
            checks="선수의 말이 帥 만 남기고 모두 깔렸고, 帥 는 높은 탑에 갇혔다.",
            board=FIRST_IS_STUCK + "차례: ▲   ply: 30\n",
            legal="",
            outcome=Outcome.SECOND_WINS,
        ),
        Fixture(
            name="second-has-no-move",
            checks="위의 판을 180° 돌리면 후수의 패배다.",
            board="""\
     a       b       c       d       e
 5   .       .       .    ▲兵/▽兵   ▽帥
 4   .       .       .    ▲槍/▽槍 ▲兵/▽兵
 3   .       .       .       .       .
 2   .       .       .       .       .
 1  ▲帥      .       .       .       .
차례: ▽   ply: 31
""",
            legal="",
            outcome=Outcome.FIRST_WINS,
        ),
        Fixture(
            name="no-move-at-move-limit",
            checks="200 수에 닿은 국면이어도 둘 수가 없으면 무승부가 아니라 패배다.",
            board=FIRST_IS_STUCK + "차례: ▲   ply: 200\n",
            legal="",
            outcome=Outcome.SECOND_WINS,
        ),
    ),
)

SECTIONS = (
    INITIAL,
    MARSHAL_MOVES,
    GENERAL_MOVES,
    LANCE_MOVES,
    PAWN_MOVES,
    BLOCKING,
    STACKING,
    TIER_LIMIT,
    REACH,
    TOP_ONLY,
    TOWER_CAPTURE,
    NOTHING_ON_MARSHAL,
    MARSHAL_NEVER_STACKS,
    MARSHAL_CAPTURE,
    MOVE_LIMIT,
    REPETITION,
    NO_LEGAL_MOVE,
)
