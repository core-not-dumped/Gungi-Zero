# 01: 프로젝트 스캐폴딩

**What to build:** 저장소에서 테스트·린트·타입 체크가 돌아간다. 아직 게임 코드는 없고, 빈 패키지 구조와 통과하는 사소한 테스트 하나만 있다.

**Blocked by:** None (can start immediately)

**Status:** ready-for-agent

- [x] `pyproject.toml` 이 Python 3.11 을 요구하고, uv 로 의존성이 잠긴다
- [x] numpy·pytest·hypothesis 가 의존성에 들어간다
- [x] `uv run pytest` 가 돌고 통과한다
- [x] `uv run ruff check` 와 `uv run ruff format --check` 가 통과한다
- [x] `uv run pyright` 가 basic 모드로 통과한다
- [x] flat 레이아웃으로 `games/`·`engine/`·`ai/`·`eval/`·`ui/`·`tests/` 패키지가 존재한다 (내용은 비어 있어도 된다)
- [x] PyTorch·wandb·pygame 는 넣지 않는다. 1단계에 필요 없다

## Comments

**2026-11-26** — 구현 완료. flat 레이아웃(`games/`·`engine/`·`ai/`·`eval/`·`ui/`·`tests/`)에
패키지 `__init__.py` 만 두고, `tests/test_scaffolding.py` 하나가 여섯 패키지가 import 되고
저장소 루트 바로 아래에 있는지 확인한다. `uv.lock` 에 torch·wandb·pygame 은 없다.

네 게이트 전부 통과: `uv run pytest`(6 passed) · `uv run ruff check` · `uv run ruff format --check` ·
`uv run pyright`(basic, 0 errors). 게이트가 공허하지 않은지 일부러 위반하는 모듈로 확인했다 —
ruff 는 F401·I001 을, pyright 는 반환 타입 불일치를 잡았다.

`[tool.uv] package = false` 로 설치 없는 가상 프로젝트로 두고, `pythonpath = ["."]` 로
flat 패키지를 테스트에서 import 한다. import 경계 검사는 이 티켓 범위가 아니다 — 05 번.
