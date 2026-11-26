# 01: AI seam 과 eval 경계

**What to build:** 수를 고르는 쪽이 모두 `AI` 프로토콜(`choose(position) -> Move`)을 따르고, 어떤 두 AI 든 `play_episode` 로 한 판을 둔다. 재는 쪽(`eval` 패키지)의 의존 방향이 import 경계 테스트로 강제된다. 뒤의 MCTS AI·완벽한 상대가 이 자리에 꽂히는 선행 정리다.

**Blocked by:** None (can start immediately)

**Status:** ready-for-agent

- [ ] `AI` 프로토콜이 학습하는 쪽(`ai`)에 있고 랜덤 AI 가 따른다
- [ ] `play_episode(game, first, second) -> Episode` 가 있고 랜덤 대국이 이것을 쓴다. 같은 시드의 대국이 바뀌지 않는다(기존 랜덤 대국·측정 보고 테스트가 그대로 통과)
- [ ] 두 AI 의 선후를 바꿔 두어도 `Episode` 의 결과가 선수·후수 기준으로 적힌다
- [ ] import 경계: `eval` 은 `ai`·`games.base`·표준 라이브러리·numpy 만 쓰고, `ai` 는 `eval` 을 import 하지 않는다. 위반을 심었을 때 잡히는지도 테스트한다

spec: "AI 프로토콜과 대국 loop", "모듈과 의존 방향".
