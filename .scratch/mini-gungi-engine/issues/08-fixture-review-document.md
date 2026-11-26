# 08: fixture 사람 검토 문서

**What to build:** 사람이 코드를 읽지 않고 규칙 해석이 맞는지 판단할 수 있는 문서가 나온다.

에이전트가 룰 엔진과 기대 정답을 둘 다 썼으므로 같은 규칙 오해가 양쪽에 들어갔을 수 있다. 상관된 오류라 테스트를 몇 겹 쌓아도 잡히지 않는다. 이 검토가 유일한 방어선이다.

**Blocked by:** 07

**Status:** ready-for-human

- [x] 채택한 규칙마다 fixture 가 하나 이상 모여 있다
- [x] fixture 마다 판 그림(텍스트 렌더 형식)과 기대 합법수(`b1-c3` 표기)가 나란히 있다
- [x] 각 fixture 가 어떤 규칙을 검증하는지 한 줄로 적혀 있다
- [x] 문서는 테스트에서 생성되거나 테스트와 같은 출처를 쓴다. 손으로 옮겨 적어 갈라지지 않는다
- [ ] 사람 검토를 받는다. 지적된 오해가 있으면 룰 엔진과 fixture 양쪽을 고친다

## Comments

- 검토 문서는 `docs/fixture-review.md` 다. 출처는 `tests/gungi_fixtures.py` 하나이고 `uv run python -m scripts.fixture_review` 로 다시 만든다. 커밋된 문서가 fixture 와 어긋나면 테스트가 실패한다.
- 남은 것은 사람 검토다. 문서의 "검토 질문" 세 가지(2단의 帥, 사거리 표, 종료 규칙)를 포함해 지적을 여기에 fixture 이름과 함께 적는다.
