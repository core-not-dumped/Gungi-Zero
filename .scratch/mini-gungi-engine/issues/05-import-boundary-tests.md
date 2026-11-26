# 05: import 경계 테스트

**What to build:** 의존 방향 원칙이 문서가 아니라 테스트로 지켜진다. 위반하면 CI 가 깨진다. `ai/` 코드는 아직 없지만 지금 만들어 둔다 — 2단계에 가서 위반을 발견하면 늦다.

**Blocked by:** 03

**Status:** ready-for-agent

- [ ] `ai/` 의 어떤 모듈도 `games.base` 외의 게임 모듈을 import 하지 않음을 검사한다. `ai/` 가 비어 있어도 통과하며 앞으로를 막는다
- [ ] `engine/` 의 어떤 모듈도 numpy·torch·wandb·`games/`·`ai/` 를 import 하지 않음을 검사한다
- [ ] 검사는 import 그래프를 보며, 모듈을 실제로 import 해 보는 데 의존하지 않는다
- [ ] 일부러 위반하는 모듈을 넣으면 테스트가 실패함을 확인한다
