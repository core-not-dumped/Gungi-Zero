# 아키텍처 점검의 남은 후보

출처: `docs/reviews/architecture-review-20261127.html` (`fbb477b` 기준). 번호는 보고서의 것이다.
1번(좌표계 약속)과 3번(판 좌표·회전)은 2026-11-27 에 했다. 소유자 결정은 아래와 같다.

## 2. `ai/` 의 대국 진행 module — 2단계 설계에서

- 대국 loop 이 랜덤 AI 에 묶여 있고 `Episode` 에 ply 별 기록이 없어, 쓰는 쪽이 대국을 다시 둔다.
- 설계에서 함께 정할 것: 수를 고르는 쪽의 seam, 정책 타깃의 기록, 가치 타깃의 부호
  (`games/base.py` 의 "프레임" 절 위에서), `scripts/` 에 있는 `perft`·`count_reachable` 의 자리,
  ADR-0004 의 종료 규칙 버전 기록.
- MCTS 의 `canonical` 비용도 여기서 본다 (`docs/encode-speed.md`).

## 4. 국면 하나는 한 번만 평가하기 — 메모

- `outcome`, `legal_action_mask`, `apply_move` 가 각각 합법수 생성과 O(ply) 반복 검사를 다시 한다.
- CLAUDE.md 대로 MCTS 를 만든 뒤 측정하고 정한다. `Game` interface 는 바꾸지 않고 안에서 감춘다.
- 결과가 같은지는 perft 로 확인한다.

## 5. 판만으로 끝난 것과 이력으로 끝난 것 — 메모, 2.5단계 전에

- `Game.outcome` 은 帥 잡기·합법수 없음(판 + 차례)과 반복·수 제한(경로 + ply)을 합쳐 답한다.
- solver 를 쓰기 전에 정할 것: solver 상태가 무엇인지, 이력과 무관한 종료를 누가 답하는지,
  그 답과 반복 3회·200수 규칙의 관계 (`docs/nano-state-space.md` 가 미룬 결정).
- 지금 `scripts/state_space.py` 의 `count_reachable` 은 docstring 의 논증에 기댄다.

## 6, 7 — 하지 않는다

말 종류별 표의 확인 지점(6)과 화면용 interface 이름 붙이기(7)는 하지 않기로 했다.

## 2·5 의 결정 (2026-11-27)

2번과 5번은 2단계 설계에서 정했다. `.scratch/alphazero-verification/spec.md` 참조.
수를 고르는 seam 은 `AI.choose`, 이력과 무관한 종료는 `games.base` 의 선택적 프로토콜
`board_outcome`, `count_reachable` 은 `eval/` 로 옮기고, `perft` 는 `scripts/` 에 남긴다.

## 8. 행동 구조의 노출 — 미룬다

출발-도착 어텐션 정책 머리(Transformer)는 행동 인덱스가 (출발, 도착, 쌓기)로 이루어졌다는
구조를 알아야 한다. `games.base` 는 지금 그 구조를 내놓지 않는다. 2단계의 `Network` 는
`tensor_shape`·`action_space_size` 만 받는다. 이 구조를 어떻게 내놓을지는 Transformer 를 할 때 정한다.
