# 07: 틱택토 검증 — 2단계 완료 기준

**What to build:** 틱택토로 학습한 MCTS AI 가 완벽한 상대에게 선수·후수 각 100판 무패하고, 이 검증이 CI 에서 wandb 없이 돈다.

**Blocked by:** 03 (solver 와 완벽한 상대), 06 (ResNet, 학습 루프, 로거)

**Status:** ready-for-agent

- [ ] 재는 쪽의 대국 harness: 두 AI 로 선수·후수를 바꿔 각 n판 두고 결과 분포를 낸다
- [ ] 완료 기준: MCTS AI(temperature 0, 탐색 50) 대 완벽한 상대, 시드 0~99, 선수·후수 각 100판 무패
- [ ] 진단: 도달 가능 5,478 국면에서 가치 머리와 이론값의 일치율을 기록한다(기준 아님)
- [ ] CI 에서 고정 시드로 위 기준을 테스트한다. CPU 로 5분을 넘으면 1세대 스모크로 내리고 전체 검증은 `slow` 표시로 로컬에서 돌린다. 내리면 측정 시간과 결정을 CLAUDE.md 에 적는다
- [ ] 통과하면 CLAUDE.md 의 진행 단계를 갱신한다

spec: "Testing Decisions / 완료 기준", "torch 와 CI".
