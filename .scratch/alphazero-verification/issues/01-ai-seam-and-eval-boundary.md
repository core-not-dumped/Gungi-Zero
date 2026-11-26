# 01: AI seam 과 eval/ 경계

**What to build:** 수를 고르는 쪽이 `AI` 프로토콜 하나로 모이고, 어떤 두 AI 든 `play_episode` 로 한 판을 둔다. `eval/` 패키지의 의존 방향이 import 경계 테스트로 강제된다. 이후 MCTS AI·완벽한 상대가 이 seam 에 꽂힌다.

**Blocked by:** -

**Status:** ready-for-agent

- [ ] `ai/` 에 `AI` 프로토콜(`choose(position) -> Move`)이 있고 `RandomAI` 가 따른다
- [ ] `play_episode(game, first, second) -> Episode` 가 있고 `play_random_episode` 가 이것을 쓴다. 같은 시드의 대국이 바뀌지 않는다 (기존 테스트·측정 보고가 그대로 통과)
- [ ] import 경계: `eval/` 은 `ai/`·`games.base`·표준 라이브러리·numpy 만 쓴다. `ai/` 는 `eval/` 을 import 하지 않는다. 위반을 심어 잡히는지도 테스트한다
- [ ] 두 AI 의 선후를 바꿔 둘 때 `Episode` 의 결과가 선수·후수 기준으로 적힌다

설계: `.scratch/alphazero-verification/spec.md` "AI seam", "패키지와 의존 방향".
