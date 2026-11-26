"""규칙별 fixture 를 사람이 검토할 문서로 만든다.

uv run python -m scripts.fixture_review

문서의 판·합법수·결과는 `tests.gungi_fixtures` 의 값 그대로다. 엔진을 부르지 않는다 —
엔진과 맞는지는 테스트가 본다.
"""

from __future__ import annotations

from pathlib import Path

from games.base import Outcome
from tests.gungi_fixtures import SECTIONS, Fixture, Then

DOCUMENT = Path(__file__).resolve().parent.parent / "docs" / "fixture-review.md"

OUTCOMES = {
    Outcome.FIRST_WINS: "▲ 승",
    Outcome.SECOND_WINS: "▽ 승",
    Outcome.DRAW: "무승부",
}

INTRODUCTION = """\
# 미니 군의 규칙 검토

<!-- 생성 파일이다. 직접 고치지 말고 tests/gungi_fixtures.py 를 고친 뒤
     `uv run python -m scripts.fixture_review` 를 실행한다. -->

룰 엔진과 기대 정답을 같은 에이전트가 썼다. 같은 규칙 오해가 양쪽에 들어갔다면 테스트는 전부
통과한다. 그래서 사람이 판과 기대 합법수를 읽고 **규칙 해석이 맞는지** 확인한다. 아래의 모든
판·합법수·결과는 테스트가 엔진에 대어 확인하는 값이다.

## 읽는 법

- ▲ 선수(아래, 1 행 쪽), ▽ 후수(위). ▽ 의 "앞" 은 1 행 쪽이다.
- 열은 `a`~`e`, 행은 `1`~`5`. `b1-c3` 은 이동 또는 잡기, `b1-c3^` 은 쌓기다.
- 탑은 `상단/하단` 이다. `▲兵/▽大` 는 2단에 ▲兵, 그 아래 1단에 ▽大.
- 합법수는 차례인 쪽의 **전부**다. 적혀 있지 않은 수는 둘 수 없다는 뜻이다.
- fixture 의 판은 규칙 하나를 보려고 직접 놓은 것이라 실제 대국의 말 개수와 다를 수 있다.

## 검토할 때 볼 것

fixture 마다 **빠진 수**와 **있으면 안 되는 수**를 찾는다. 각 규칙에 적은 "근거" 가 약한 곳
(2차 자료, 발명)을 특히 본다. 지적은 이 파일이 아니라
`.scratch/mini-gungi-engine/issues/08-fixture-review-document.md` 의 `## Comments` 에
fixture 이름과 함께 적는다.

## 검토 질문

규칙을 읽는 것과 별개로 결정이 필요한 것들이다.

1. **2단의 帥.** 帥ツケ 금지와 "帥 위 쌓기 금지" 때문에 帥 는 실제 대국에서 2단에 오를 수 없다.
   그런데 텍스트 판을 읽는 `parse` 는 `▲帥/▲兵` 을 받고, `second-tier-marshal` 은 그런 판으로
   2칸 이동을 확인한다. 지금대로 둘지, `parse` 가 거절하게 하고 fixture 를 지울지.
2. **사거리 표.** 말의 이동과 2단의 +1 은 공식 룰북의 그림을 2차 자료가 옮긴 값이다
   (ADR-0002). 룰북이 있다면 帥·大·槍·兵 의 1단·2단 이동을 대조한다.
3. **종료 규칙.** 200수·3회 반복·합법수 없음은 발명이다 (ADR-0004). 해석의 옳고 그름이 아니라
   이대로 학습할지를 본다. 값은 09 번 측정 뒤에 다시 정한다.
"""


def render_document() -> str:
    parts = [INTRODUCTION, _contents()]
    for number, section in enumerate(SECTIONS, start=1):
        parts.append(
            f"## {number}. {section.title}\n\n"
            f"**규칙**: {section.rule}\n\n"
            f"**근거**: {section.source}\n"
        )
        parts.extend(
            _fixture(f"{number}.{index}", fixture)
            for index, fixture in enumerate(section.fixtures, start=1)
        )
    return "\n".join(parts)


def _contents() -> str:
    lines = ["## 규칙 목록", "", "| # | 규칙 | fixture |", "|---|---|---|"]
    for number, section in enumerate(SECTIONS, start=1):
        names = ", ".join(f"`{fixture.name}`" for fixture in section.fixtures)
        lines.append(f"| {number} | {section.title} | {names} |")
    return "\n".join(lines) + "\n"


def _fixture(number: str, fixture: Fixture) -> str:
    parts = [
        f"### {number} `{fixture.name}`\n",
        f"{fixture.checks}\n",
        _board(fixture.board),
        _expectation(fixture.board, fixture.legal, fixture.outcome),
    ]
    parts.extend(_then(then) for then in fixture.then)
    return "\n".join(parts)


def _then(then: Then) -> str:
    moves = " ".join(f"`{move}`" for move in then.moves.split())
    return "\n".join(
        [
            f"**{moves} 를 두면**\n",
            _board(then.board),
            _expectation(then.board, then.legal, then.outcome),
        ]
    )


def _board(board: str) -> str:
    return f"```text\n{board}```\n"


def _expectation(board: str, legal: str | None, outcome: Outcome | None) -> str:
    """판 아래에 붙는 결과와 합법수. 합법수는 움직이는 말별로 묶는다."""
    if outcome is not None:
        return f"결과: **{OUTCOMES[outcome]}**. 합법수 없음.\n"
    if legal is None:
        return "결과: 진행 중.\n"

    tops = _top_pieces(board)
    by_source: dict[str, list[str]] = {}
    for move in legal.split():
        by_source.setdefault(move.split("-")[0], []).append(move)
    lines = [f"합법수 {len(legal.split())} 개:", ""]
    for source, moves in by_source.items():
        listed = " ".join(f"`{move}`" for move in moves)
        lines.append(f"- {source} {tops[source]}: {listed}")
    return "\n".join(lines) + "\n"


def _top_pieces(board: str) -> dict[str, str]:
    """텍스트 판 표기에서 읽은 `칸 → 맨 위 말`."""
    header, *rows, _turn = board.splitlines()
    columns = header.split()
    tops: dict[str, str] = {}
    for line in rows:
        row, *cells = line.split()
        for column, cell in zip(columns, cells, strict=True):
            tops[column + row] = cell.split("/")[0]
    return tops


def main() -> int:
    DOCUMENT.write_text(render_document(), encoding="utf-8", newline="\n")
    print(DOCUMENT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
