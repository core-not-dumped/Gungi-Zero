# gungi-zero

헌터헌터 군의(Gungi)를 AlphaZero 계열 자가 대국 강화학습으로 학습시키는 프로젝트.
MCTS·자가 대국·학습 루프는 프레임워크 없이 직접 구현한다. 신경망은 PyTorch, 기록은 wandb 다.

코드를 만지기 전에 `CONTEXT.md`(용어)와 건드리는 영역의 `docs/adr/`(되돌리기 어려운 결정)를 읽는다.

## 현재: 2.5단계 — 나노 군의 검증

완료 기준: 완벽한 상대와 **풀이에서 뽑은 시험 판 200개에서 이론값**을 지키고 시작 국면의 3수 퍼즐을 푼다 (`.scratch/alphazero-verification/spec.md`, ADR-0007).

설계하기 전에 읽는다.

- `games/base.py` 의 "프레임"·"잘못된 입력" 절 — 정규형에서 승패 부호와 행동 인덱스가 옮겨지는 법
- `.scratch/mini-gungi-engine/notes-architecture-followups.md` — 2단계 설계에서 정한 것과 미룬 것
- `docs/encode-speed.md` — `canonical` 은 후수 차례에서 경로 길이에 비례해 비싸다

틱택토에는 경로·탑·쌓기가 없다. 반복, 수 진행도, 군의 메커니즘은 2단계 검증을 통과했어도 확인되지 않았다.

### 나노 군의 풀이

- `uv run python -m scripts.solve_nano` — 나노 군의를 풀어 `solutions/` 에 저장한다(git 제외). 룰 엔진이나 solver 를 바꾸면 다시 만든다.
- `uv run python -m scripts.verify_nano --device cuda` — 나노 군의로 학습해 시험 판과 3수 퍼즐을 잰다(wandb 기록). 1차 시도(2026-11-27)는 149/200 으로 통과하지 못했다 (`docs/nano-verification-report.md`).
- 시작 국면의 이론값은 **선수 승(DTM 3)** 이다. 그래서 검증은 시작 국면이 아니라 시험 판으로 하고, 자가 대국도 도달 가능한 여러 판에서 시작한다.
- `uv run python -m scripts.nano_trials` — 풀이에서 시험 판을 뽑아 층마다의 개수와 지문을 찍는다. 층·개수·시드를 바꾸면 ADR 을 고친다.
- `engine/`, `games/mini_gungi.py`·`nano_gungi.py`, `eval/`(solver·완벽한 상대·시험 판), `scripts/solve_nano.py`·`nano_trials.py` 를 바꾸면 `-m slow` 를 다시 돌린다. CI 는 나노 군의를 풀지 않는다.

### 틱택토 검증 (2단계, 2026-11-27 통과)

- `uv run python -m scripts.verify_tictactoe` — 처음부터 학습해 완벽한 상대와 둔다. 학습 시드 0 은 선수·후수 각 100판 무패, 가치 일치율 0.79 였다. 100판 중 서로 다른 대국은 선수 5가지, 후수 60가지라 "100판 무패" 가 시험하는 국면은 판 수보다 훨씬 좁다.
- 로컬 CPU(Windows, torch 스레드 1)에서 420초가 걸려 CI 의 5분 기준을 넘는다. CI 는 1세대 스모크 테스트만 돌고, 완료 기준은 `uv run pytest -m slow` 로 로컬에서 확인한다. `-m slow` 는 나노 군의 풀이와 그 확인(약 16분)도 함께 돌린다. CI 환경(리눅스)에서는 학습 전체를 재지도 통과시키지도 않았다.
- `ai/`, `games/base.py`, `games/tictactoe.py`, `eval/`(solver·완벽한 상대·harness), `scripts/train.py`·`verify_tictactoe.py`, 틱택토 하이퍼파라미터를 바꾸면 `-m slow` 를 다시 돌린다. CI 는 학습이 약해지는 변경을 잡지 못한다.

## 원칙

의존 방향은 `ai/` → `games/` ← `engine/` 이다 (ADR-0001).

- `ai/` 가 아는 게임 모듈은 `games.base` 뿐이다. 게임을 고르는 합성 루트는 `scripts/` 다.
- `eval/`(재는 쪽: 완벽한 상대, 대국 승률, 진단)은 `ai/` 와 `games.base` 만 쓴다. `ai/` 는 `eval/` 을 모른다.
- `engine/` 은 표준 라이브러리와 자기 자신만 쓰는 순수 Python 이다.
- `ui/` 는 합법수와 결과를 게임 어댑터에 묻는다.
- `games/` 와 `ai/` 사이의 판 표현은 `numpy.ndarray` `(C, H, W)` 다. torch 변환·배치·디바이스는 `ai/` 가 맡는다.
- 행동 공간과 텐서 크기는 `Game` 에서 읽는다. 게임과 변형마다 다르다.
- 학습 루프는 로거를 주입받는다. 틱택토 검증은 wandb 없이 돈다.
- `Position` 은 불변이고 해시 가능하며 `apply_move` 가 새 `Position` 을 반환한다. `Game` 에 `undo` 를 두지 않는다. 가변 구현은 측정 후 프로토콜을 만족하는 별개 클래스로 추가하고, 단순 구현과 결과가 같은지 테스트한다.
- 반복 횟수는 `board_key`(판 + 차례)로 센다. `Position` 동등성은 경로를 포함한다 — 빼면 "같은" 두 국면의 반복 횟수가 달라진다. 그래서 MCTS 는 전치표 없이 트리로 만든다.
- 신경망에 보이는 판은 현재 플레이어 시점의 정규형이다 (ADR-0003).
- 신경망 입력은 현재 국면 하나다. 과거 국면 스택은 쌓지 않는다 — 군의의 이력 의존 규칙은 반복뿐이고 텐서의 평면 한 장이 요약한다.
- 학습 데이터에 거울 증강을 쓰지 않는다. 룰이 대칭이어도 초기 배치가 비대칭이다 (ADR-0005).
- 처음엔 단순하고 정확하게. 속도 최적화는 검증 후 측정하고 진행한다.

## 군의 규칙

규칙을 확인하거나 바꾸려면 `docs/adr/0002`·`0004`·`0006` 과 `docs/fixture-review.md`(생성 문서, 출처는 `tests/gungi_fixtures.py`)를 읽는다.

룰 엔진과 기대 정답을 에이전트가 함께 쓰면 같은 오해가 양쪽에 들어간다. 그래서 새로 쓰거나 바꾼 fixture 는 사람의 검토를 받는다.

1단계 검증의 남은 한계: 사람은 한 화면 요약만 검토해 개별 판의 기대 합법수를 확인하지 않았고, 사거리 표는 룰북과 대조하지 않았다 (`.scratch/mini-gungi-engine/issues/08-fixture-review-document.md`).

## 진행 단계

1. 미니 군의 룰 엔진 — 완료
- 1.5. pygame 대국 화면 — 완료 (사람이 직접 두어 규칙 해석 확인)
2. AlphaZero 구현 + 틱택토 검증 — 완료
- 2.5. **나노 군의 검증** ← 현재 — 나노 군의(`docs/nano-state-space.md`)를 완전 탐색으로 풀고 그 답에 대어, 5×5 에 연산을 쓰기 전에 탑·단수 제약·쌓기 인코딩의 버그를 잡는다. 미니 군의와 같은 엔진·같은 인코딩 방식이며 크기는 판을 따른다. 槍 은 여기서 검증되지 않는다.
3. **미니 군의 학습** — 랜덤 AI 상대 승률 90% 이상, 세대별 Elo 상승, 탐색 AI 상대 Elo 측정
4. **속도 개선** — 여러 판 동시 진행 + 신경망 배치 추론. 반복 검사가 느리면 증분 Zobrist 해시
5. **Gumbel AlphaZero** — 적은 탐색 횟수에서 AlphaZero 와 비교
6. (선택) **9×9 입문 룰** 구현 및 학습

## 평가

- Elo 기준점: 랜덤 AI = 0, **탐색 AI**(알파베타), 이전 세대 체크포인트들과의 대국
- 랜덤 AI 90% 는 버그가 있어도 통과하는 약한 기준이다. 탐색 AI 상대 Elo 를 같이 본다.
- 로거로 학습 손실, 자가 대국 통계, Elo 를 기록한다.
- Elo 비교는 같은 종료 규칙 안에서만 유효하다. 종료 규칙을 바꾸면 기록에 버전을 남긴다 (ADR-0004).

## Agent skills

### Issue tracker

이슈는 저장소 안 `.scratch/<feature-slug>/` 의 마크다운 파일로 산다. `docs/agents/issue-tracker.md` 참조.

### Triage labels

다섯 개 표준 triage 역할을 그대로 쓴다. `docs/agents/triage-labels.md` 참조.

### Domain docs

단일 컨텍스트: 저장소 루트의 `CONTEXT.md` 와 `docs/adr/`. `docs/agents/domain.md` 참조.
