# 02: 이력과 무관한 종료 `board_outcome`

**What to build:** 帥 잡기·합법수 없음처럼 판 + 차례만으로 끝난 것을 반복·수 제한과 따로 묻는 연산이 세 게임에 있다. 도달 가능 상태 세기가 "최단 경로로 만난 국면에는 반복이 없다"는 암묵적 논증 대신 이 연산에 기댄다. solver 를 위한 선행 정리다.

**Blocked by:** None (can start immediately)

**Status:** ready-for-agent

- [ ] 공통 게임 인터페이스에 선택적 프로토콜 `board_outcome(position) -> Outcome | None` 이 있다. `Game` 본체는 바뀌지 않는다
- [ ] 틱택토·미니 군의·나노 군의가 구현한다
- [ ] 성질 테스트: 값이 `None` 이 아니면 `outcome` 과 같다. 반복 국면과 수 제한 국면에서는 `None` 이고 `outcome` 만 무승부다
- [ ] 정규형에서 승패 부호가 `outcome` 과 같은 규칙으로 뒤바뀐다
- [ ] 도달 가능 상태 세기를 재는 쪽으로 옮기고 종료 판정을 `board_outcome` 으로 바꾼다. 기존 상태 수 고정값이 그대로 통과한다

spec: "이력과 무관한 종료".
