# 02: 이력과 무관한 종료 `board_outcome`

**What to build:** 帥 잡기·합법수 없음처럼 판 + 차례만으로 끝난 것을 반복·수 제한과 따로 묻는 연산이 세 게임에 있다. 도달 가능 상태 세기가 "최단 경로로 만난 국면에는 반복이 없다"는 암묵적 논증 대신 이 연산에 기댄다. solver 를 위한 선행 정리다.

**Blocked by:** None (can start immediately)

**Status:** ready-for-agent

- [x] 공통 게임 인터페이스에 선택적 프로토콜 `board_outcome(position) -> Outcome | None` 이 있다. `Game` 본체는 바뀌지 않는다
- [x] 틱택토·미니 군의·나노 군의가 구현한다
- [x] 성질 테스트: 값이 `None` 이 아니면 `outcome` 과 같다. 반복 국면과 수 제한 국면에서는 `None` 이고 `outcome` 만 무승부다
- [x] 정규형에서 승패 부호가 `outcome` 과 같은 규칙으로 뒤바뀐다
- [x] 도달 가능 상태 세기를 재는 쪽으로 옮기고 종료 판정을 `board_outcome` 으로 바꾼다. 기존 상태 수 고정값이 그대로 통과한다

spec: "이력과 무관한 종료".

## Comments

- 결과(2026-11-27): 선택적 프로토콜은 `games.base.BoardOutcomeGame`, 세기는 `eval/reachable.py`, 후보 표를 내는 스크립트는 `scripts/state_space.py` 에 남았다.
- 상태 수 고정값은 그대로다. `terminal` 고정값 하나는 바뀌었다: `move_limit=1` 변형에서 1 → 0. 수 제한에 닿은 국면은 판만으로 끝난 것이 아니다.
- 세기가 뻗어 나가는 것은 여전히 `legal_moves` 라 수 제한이 도달 범위를 자른다. 반복 3회가 이것을 막지 않는 근거는 `count_reachable` docstring 에 있다.
- 바꾼 세기로 `scripts.state_space` 를 다시 돌려 `docs/nano-state-space.md` 표의 상태 수·끝난 상태·최대 깊이가 모두 같음을 확인했다(상한 200만).
