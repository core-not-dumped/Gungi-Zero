# import 경계 검사의 남은 빈틈

**언제 볼 것:** 07 번 티켓이 끝난 뒤 아키텍처 점검 때. 지금은 고치지 않는다.

05 번 티켓과 CI 추가분(`d1c5447..2792d54`)의 코드 리뷰에서 나온 사소한 지적 세 건이다.
재현해 보지 않고 코드를 읽고 판단했다. 대상은 `tests/test_import_boundaries.py`.

1. **`games/` 의 역방향 import 를 검사하지 않는다.** `FORBIDDEN` 에 `ai`·`engine` 규칙만 있다.
   `ai/` 에서 닿지 않는 `games/` 모듈이 `ai`·`eval`·`ui`·`scripts` 를 import 해도 통과한다
   (예: `games/mini_gungi.py` 의 `import ai.episode`). ADR-0001 의 `ai/ → games/ ← engine/` 에
   어긋난다. 05 번 티켓 범위 밖이었다. 결정할 것: `games` 규칙을 추가할지.

2. **동적 import 를 보지 못한다.** `importlib.import_module("games.mini_gungi")` 나
   `__import__(...)` 는 통과한다. 모듈 docstring 에 적어 둔 한계다. 선택지: 문자열 리터럴 인자를
   import 로 취급하거나, `ai/`·`engine/` 에서 동적 import 를 금지한다.

3. **`from games import *` 가 하위 모듈 검사를 빠져나간다.** `games/__init__.py` 에
   `__all__ = ["mini_gungi"]` 가 있으면 실행 시 import 되지만 검사는 `games` 만 기록한다.
   ruff F403 이 star import 를 막으므로 `noqa` 를 달아야 실제로 샌다. 선택지: 별칭이 `*` 이면
   대상 패키지의 직접 하위 모듈을 모두 import 한 것으로 본다.
