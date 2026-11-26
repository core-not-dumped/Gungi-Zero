# `games/` 패키지를 게임과 AI 사이의 유일한 경계로 둔다

"engine/ 은 ai/ 에 의존하지 않는다" 와 "ai/ 는 특정 게임에 묶이지 않는다" 를 동시에 지키면 공통 게임 인터페이스(합법수·수 두기·승패·판→텐서)가 둘 중 어디에도 놓일 수 없다. 그래서 최상위 `games/` 를 세 번째 자리로 만든다. `games/base.py` 가 `Game`·`Position`·`Move` 프로토콜을, `games/mini_gungi.py` 가 `engine/` 을 감싸는 어댑터를, `games/tictactoe.py` 가 룰 엔진 없는 직접 구현을 담는다. 의존 방향은 `ai/` → `games/` ← `engine/` 이고, `engine/` 과 `ai/` 는 서로를 모른다.

## Considered Options

- **`engine/` 이 인터페이스를 직접 구현**: engine 이 텐서 인코딩과 AI 용 추상화를 알게 되어 "순수 룰 엔진" 이라는 성질을 잃는다. 룰 엔진 테스트에 PyTorch 가 끌려 들어온다.
- **`ai/` 안에 게임별 어댑터**: ai/ 가 군의와 틱택토를 import 하게 되어 게임 독립성을 잃는다. 2단계의 틱택토 검증이 "ai/ 가 게임에 묶이지 않는다" 를 증명하지 못한다.

## Consequences

"ai/ 는 `games.base` 외의 어떤 게임 모듈도 import 하지 않는다" 를 import 그래프 테스트로 강제할 수 있다. 원칙이 문서가 아니라 CI 에 산다.

대가는 간접층 하나와, 군의 쪽 변경이 `engine/` 과 `games/mini_gungi.py` 두 군데에 걸친다는 점이다. 경계를 넘는 판 표현은 `numpy.ndarray` `(C, H, W)` 로 고정하고, torch 변환·배치·디바이스는 전부 `ai/` 가 맡는다.
