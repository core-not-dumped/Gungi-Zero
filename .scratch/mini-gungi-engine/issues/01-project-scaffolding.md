# 01: 프로젝트 스캐폴딩

**What to build:** 저장소에서 테스트·린트·타입 체크가 돌아간다. 아직 게임 코드는 없고, 빈 패키지 구조와 통과하는 사소한 테스트 하나만 있다.

**Blocked by:** None (can start immediately)

**Status:** ready-for-agent

- [ ] `pyproject.toml` 이 Python 3.11 을 요구하고, uv 로 의존성이 잠긴다
- [ ] numpy·pytest·hypothesis 가 의존성에 들어간다
- [ ] `uv run pytest` 가 돌고 통과한다
- [ ] `uv run ruff check` 와 `uv run ruff format --check` 가 통과한다
- [ ] `uv run pyright` 가 basic 모드로 통과한다
- [ ] flat 레이아웃으로 `games/`·`engine/`·`ai/`·`eval/`·`ui/`·`tests/` 패키지가 존재한다 (내용은 비어 있어도 된다)
- [ ] PyTorch·wandb·pygame 는 넣지 않는다. 1단계에 필요 없다
