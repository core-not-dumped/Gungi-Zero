# 01: AI seam 과 eval 경계

**What to build:** 수를 고르는 쪽이 모두 `AI` 프로토콜(`choose(position) -> Move`)을 따르고, 어떤 두 AI 든 `play_episode` 로 한 판을 둔다. 재는 쪽(`eval` 패키지)의 의존 방향이 import 경계 테스트로 강제된다. 뒤의 MCTS AI·완벽한 상대가 이 자리에 꽂히는 선행 정리다.

**Blocked by:** None (can start immediately)

**Status:** ready-for-agent

- [x] `AI` 프로토콜이 학습하는 쪽(`ai`)에 있고 랜덤 AI 가 따른다
- [x] `play_episode(game, first, second) -> Episode` 가 있고 랜덤 대국이 이것을 쓴다. 같은 시드의 대국이 바뀌지 않는다(기존 랜덤 대국·측정 보고 테스트가 그대로 통과)
- [x] 두 AI 의 선후를 바꿔 두어도 `Episode` 의 결과가 선수·후수 기준으로 적힌다
- [x] import 경계: `eval` 은 `ai`·`games.base`·표준 라이브러리·numpy 만 쓰고, `ai` 는 `eval` 을 import 하지 않는다. 위반을 심었을 때 잡히는지도 테스트한다

spec: "AI 프로토콜과 대국 loop", "모듈과 의존 방향".

## Comments

- 결과(2026-11-27): `AI` 는 `ai/base.py`, `play_episode` 는 `ai/episode.py`. 변형 넷의 시드 0~99 랜덤 대국이 바뀌기 전과 같음을 일회성 스크립트로 확인했다.
- `eval` 의 경계 검사는 `ai` 안으로 따라 들어가지 않는다. `ai` 가 torch 를 쓰는 것은 `eval` 의 위반이 아니다. `ai` 의 규칙은 금지 목록이라 `ai` 가 `scripts`·`ui`·`wandb` 를 import 하는 것은 아직 잡지 않는다.
