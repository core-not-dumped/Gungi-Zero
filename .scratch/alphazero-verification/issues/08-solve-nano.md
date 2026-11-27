# 08: 나노 군의 풀기

**What to build:** 3x4 帥·大·兵 나노 군의의 이론값이 파일로 저장되어 있고, 시작 국면의 이론값과 최대 DTM 이 문서에 있다. 완벽한 상대가 그 파일로 나노 군의를 둔다.

**Blocked by:** 03 (solver 와 완벽한 상대)

**Status:** ready-for-agent

- [x] 스크립트가 나노 군의를 풀어 파일로 저장한다(git 제외). 다시 만드는 명령을 문서에 적는다
- [x] 상태 수가 881,080 으로 나노 상태 공간 문서와 같다
- [x] 시작 국면의 이론값, 최대 DTM, 승·무·패 국면 수, 걸린 시간을 문서에 적는다
- [x] 최대 DTM 과 수 제한 200 의 관계를 적는다. DTM 이 남은 수보다 길어 무승부가 되는 국면이 있는지 본다
- [x] 완벽한 상대끼리의 나노 대국이 시작 국면의 이론값으로 끝난다(시드 몇 개)

spec: "Solver 와 완벽한 상대".

## Comments

- 결과(2026-11-27): 풀이 문서는 `docs/nano-solution.md`, 합성 루트는 `scripts/solve_nano.py` 다. 파일은 `solutions/nano-gungi.pickle` 이고 수 제한과 변형 설정을 함께 적는다. `Solution` 에 `verdict_counts()`·`max_dtm()`·`save()` 를 더했고, 읽기는 `eval.solver.load_solution` (나노는 `scripts.solve_nano.load_solution_of`) 다.
- 시작 국면은 **선수 승(DTM 3)**, 최대 DTM 20, 승 682,032 · 무 22,392 · 패 176,656.
- 09 에서 정해야 할 것: 이론값이 선수 승이라 완료 기준의 판정은 선수 100판에만 걸리고, 시험하는 국면이 좁다. 풀이 문서의 "2.5단계 검증에 주는 뜻" 에 있다. 학습하지 않은 MCTS(탐색 100)는 선수·후수 100패였다.
