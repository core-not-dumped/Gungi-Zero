# 02: 이력과 무관한 종료 `board_outcome`

**What to build:** solver 가 반복·수 제한과 무관한 종료(帥 잡기·합법수 없음)를 명시적인 연산으로 묻는다. 지금 `count_reachable` 이 기대는 "최단 경로 대표 국면" 논증이 코드로 드러난다 (후속 메모 5번).

**Blocked by:** -

**Status:** ready-for-agent

- [ ] `games.base` 에 선택적 프로토콜(`board_outcome(position) -> Outcome | None`)이 있다. `Game` 본체는 바뀌지 않는다
- [ ] 틱택토와 군의(미니·나노)가 구현한다
- [ ] 성질 테스트: `board_outcome(p)` 가 `None` 이 아니면 `outcome(p)` 와 같다. 반복 국면과 수 제한 국면에서는 `board_outcome` 이 `None` 이고 `outcome` 이 무승부다
- [ ] 정규형에서의 부호가 `outcome` 과 같은 규칙으로 뒤바뀐다 (`games/base.py` "프레임")
- [ ] `count_reachable` 을 `eval/` 로 옮기고 종료 판정을 `board_outcome` 으로 바꾼다. `docs/nano-state-space.md` 의 상태 수가 그대로다 (`tests/test_state_space.py`)

설계: spec "이력과 무관한 종료".
