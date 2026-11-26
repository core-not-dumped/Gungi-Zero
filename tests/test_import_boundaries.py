"""`ai/`·`engine/`·`eval/` 의 의존 경계를 import 그래프로 강제한다 (ADR-0001).

모듈을 import 하지 않고 소스를 `ast` 로 읽는다. 그래서 설치되지 않은 의존성(torch·wandb)의
위반도 잡히고, 함수 안이나 `TYPE_CHECKING` 아래의 import 도 빠지지 않는다.
`importlib.import_module` 같은 동적 import 는 보지 못한다.
"""

import ast
import sys
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def forbidden_for_ai(name: str) -> bool:
    """`ai/` 가 아는 게임 모듈은 `games.base` 뿐이고, 재는 쪽(`eval/`)은 모른다.

    `engine/` 도 게임 모듈이다.
    """
    top = name.partition(".")[0]
    return top in {"engine", "eval"} or (top == "games" and name not in {"games", "games.base"})


def forbidden_for_engine(name: str) -> bool:
    """`engine/` 은 순수 Python 이다 — 표준 라이브러리와 자기 자신만 쓴다.

    금지 목록(numpy·torch·wandb·`games/`·`ai/` …) 대신 허용 목록이라 새 의존성도 막힌다.
    """
    top = name.partition(".")[0]
    return top != "engine" and top not in sys.stdlib_module_names


def forbidden_for_eval(name: str) -> bool:
    """`eval/` 은 `ai/`·`games.base`·표준 라이브러리·numpy 와 자기 자신만 쓴다."""
    top = name.partition(".")[0]
    if top in {"eval", "ai", "numpy"} or top in sys.stdlib_module_names:
        return False
    return name not in {"games", "games.base"}


FORBIDDEN: dict[str, Callable[[str], bool]] = {
    "ai": forbidden_for_ai,
    "engine": forbidden_for_engine,
    "eval": forbidden_for_eval,
}

# 따라 들어가지 않는 패키지. 그 안의 import 는 그 패키지의 경계가 본다 —
# `ai/` 가 torch 를 쓴다고 그것을 부르는 `eval/` 이 위반은 아니다.
NOT_FOLLOWED: dict[str, set[str]] = {"eval": {"ai"}}


def source_files(root: Path) -> dict[str, Path]:
    """저장소 안의 모듈 이름 → 파일. 점으로 시작하는 폴더(`.venv` 등)는 저장소 코드가 아니다.

    `__init__.py` 가 없는 폴더도 import 되므로 패키지로 넣는다. 그 파일 경로는 존재하지 않는다.
    """
    files: dict[str, Path] = {}
    for entry in sorted(root.iterdir()):
        if entry.name.startswith("."):
            continue
        for path in sorted(entry.rglob("*.py")) if entry.is_dir() else [entry]:
            if path.suffix != ".py":
                continue
            parts = path.relative_to(root).with_suffix("").parts
            for end in range(1, len(parts)):
                files.setdefault(".".join(parts[:end]), root.joinpath(*parts[:end], "__init__.py"))
            if parts[-1] == "__init__":
                parts = parts[:-1]
            files[".".join(parts)] = path
    return files


def imported_names(module: str, files: dict[str, Path]) -> Iterator[str]:
    """`module` 의 import 문이 가리키는 모듈 이름들. 상위 패키지도 포함한다."""
    path = files[module]
    package = module.split(".")
    if path.name != "__init__.py":
        package = package[:-1]
    if not path.is_file():  # `__init__.py` 없는 패키지는 실행할 코드가 없다.
        return

    for node in ast.walk(ast.parse(path.read_bytes(), filename=str(path))):
        if isinstance(node, ast.Import):
            targets = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            prefix = package[: len(package) - node.level + 1] if node.level else []
            base = ".".join([*prefix, *([node.module] if node.module else [])])
            # `from games import mini_gungi` 의 이름은 하위 모듈일 수도, 속성일 수도 있다.
            submodules = [f"{base}.{alias.name}" for alias in node.names]
            targets = [base, *(name for name in submodules if name in files)]
        else:
            continue
        for target in targets:
            parts = target.split(".")
            yield from (".".join(parts[: end + 1]) for end in range(len(parts)))


def violations(root: Path, package: str) -> list[str]:
    """`package` 에서 금지된 모듈에 닿는 import 경로들. 저장소 안의 모듈을 거쳐 가도 잡는다."""
    is_forbidden = FORBIDDEN[package]
    not_followed = NOT_FOLLOWED.get(package, set())
    files = source_files(root)
    queue: list[tuple[str, ...]] = [(name,) for name in files if name.split(".")[0] == package]
    seen = {chain[0] for chain in queue}
    found = set()
    for chain in queue:  # 돌면서 뒤에 덧붙인다 — 너비 우선이라 가장 짧은 경로가 보고된다.
        for name in imported_names(chain[-1], files):
            if is_forbidden(name):
                found.add(" -> ".join([*chain, name]))
            elif name in files and name not in seen and name.split(".")[0] not in not_followed:
                seen.add(name)
                queue.append((*chain, name))
    return sorted(found)


@pytest.mark.parametrize("package", FORBIDDEN)
def test_the_repository_respects_the_import_boundary(package: str) -> None:
    assert violations(REPO_ROOT, package) == []


def test_the_check_reads_the_real_source_tree() -> None:
    """경로가 틀려 아무것도 읽지 않으면 위의 검사는 공허하게 통과한다."""
    files = source_files(REPO_ROOT)
    assert {"ai", "engine", "engine.rules", "eval", "games.base"} <= files.keys()
    assert "games.base" in set(imported_names("ai.random_play", files))


# 아래는 검사 자체의 검사다: 일부러 위반하는 모듈을 심은 가짜 저장소에 돌린다.


def plant(root: Path, sources: dict[str, str]) -> Path:
    """실제 저장소를 닮은 최소 트리에 `sources` 를 덮어쓴다."""
    skeleton = {
        "ai/__init__.py": "",
        "engine/__init__.py": "",
        "engine/rules.py": "",
        "eval/__init__.py": "",
        "games/__init__.py": "",
        "games/base.py": "",
        "games/mini_gungi.py": "",
    }
    for relative, source in (skeleton | sources).items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
    return root


@pytest.mark.parametrize(
    ("source", "reached"),
    [
        ("import games.mini_gungi", "games.mini_gungi"),
        ("import games.mini_gungi as gungi", "games.mini_gungi"),
        ("from games import mini_gungi", "games.mini_gungi"),
        ("from games.mini_gungi import Gungi", "games.mini_gungi"),
        # 아직 없는 게임 모듈도 막는다.
        ("from games.chess import Chess", "games.chess"),
        ("import engine.rules", "engine"),
        # 학습하는 쪽은 재는 쪽을 모른다.
        ("import eval", "eval"),
        ("from eval import solver", "eval"),
        ("def f():\n    import games.mini_gungi", "games.mini_gungi"),
        (
            "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    import games.mini_gungi",
            "games.mini_gungi",
        ),
    ],
)
def test_ai_importing_a_game_module_is_caught(tmp_path: Path, source: str, reached: str) -> None:
    root = plant(tmp_path, {"ai/mcts.py": source})
    assert f"ai.mcts -> {reached}" in violations(root, "ai")


@pytest.mark.parametrize(
    ("sources", "chain"),
    [
        (
            {"ai/mcts.py": "import scripts.play", "scripts/play.py": "import games.mini_gungi"},
            "ai.mcts -> scripts.play -> games.mini_gungi",
        ),
        (
            {"ai/mcts.py": "import games.base", "games/base.py": "from . import mini_gungi"},
            "ai.mcts -> games.base -> games.mini_gungi",
        ),
        (
            {"ai/mcts.py": "import games.base", "games/__init__.py": "from .mini_gungi import x"},
            "ai.mcts -> games -> games.mini_gungi",
        ),
        (
            {"ai/search/tree.py": "from games.mini_gungi import Gungi"},
            "ai.search.tree -> games.mini_gungi",
        ),
        (
            {"ai/mcts.py": "from games import variants", "games/variants/chess.py": ""},
            "ai.mcts -> games.variants",
        ),
        (
            {"ai/mcts.py": "import helpers", "helpers.py": "import eval"},
            "ai.mcts -> helpers -> eval",
        ),
        (
            {"ai/mcts.py": "import helpers", "helpers.py": "import games.mini_gungi"},
            "ai.mcts -> helpers -> games.mini_gungi",
        ),
        (
            {"ai/mcts.py": "from common import x", "common/x.py": "import games.mini_gungi"},
            "ai.mcts -> common.x -> games.mini_gungi",
        ),
    ],
)
def test_ai_reaching_a_game_module_indirectly_is_caught(
    tmp_path: Path, sources: dict[str, str], chain: str
) -> None:
    assert chain in violations(plant(tmp_path, sources), "ai")


@pytest.mark.parametrize(
    ("source", "reached"),
    [
        ("import numpy", "numpy"),
        ("from numpy import typing", "numpy"),
        ("import torch.nn", "torch"),
        ("import wandb", "wandb"),
        ("from games.base import Outcome", "games"),
        ("from ai.episode import Episode", "ai"),
        ("import eval", "eval"),
        ("import ui", "ui"),
        ("def f():\n    import numpy", "numpy"),
    ],
)
def test_engine_importing_anything_impure_is_caught(
    tmp_path: Path, source: str, reached: str
) -> None:
    root = plant(tmp_path, {"engine/rules.py": source})
    assert f"engine.rules -> {reached}" in violations(root, "engine")


@pytest.mark.parametrize(
    ("source", "reached"),
    [
        ("import games.mini_gungi", "games.mini_gungi"),
        ("from games import mini_gungi", "games.mini_gungi"),
        ("import engine.rules", "engine"),
        ("import torch", "torch"),
        ("import wandb", "wandb"),
        ("import ui", "ui"),
        ("from scripts import perft", "scripts"),
        ("def f():\n    import games.mini_gungi", "games.mini_gungi"),
    ],
)
def test_eval_importing_anything_else_is_caught(tmp_path: Path, source: str, reached: str) -> None:
    root = plant(tmp_path, {"eval/solver.py": source})
    assert f"eval.solver -> {reached}" in violations(root, "eval")


@pytest.mark.parametrize(
    ("sources", "chain"),
    [
        (
            {"eval/solver.py": "import games.base", "games/base.py": "from . import mini_gungi"},
            "eval.solver -> games.base -> games.mini_gungi",
        ),
        (
            {"eval/solver.py": "import games.base", "games/__init__.py": "import engine"},
            "eval.solver -> games -> engine",
        ),
        (
            {"eval/solver.py": "from . import perfect", "eval/perfect.py": "import torch"},
            "eval.perfect -> torch",
        ),
    ],
)
def test_eval_reaching_a_forbidden_module_indirectly_is_caught(
    tmp_path: Path, sources: dict[str, str], chain: str
) -> None:
    assert chain in violations(plant(tmp_path, sources), "eval")


def test_eval_may_use_the_ai_whatever_the_ai_itself_imports(tmp_path: Path) -> None:
    """`ai/` 의 import 는 `ai/` 의 경계가 본다. 재는 쪽이 torch 를 쓰는 AI 를 불러도 된다."""
    root = plant(
        tmp_path,
        {
            "eval/solver.py": (
                "from __future__ import annotations\nimport dataclasses\nimport numpy.typing\n"
                "from ai import network\nfrom ai.episode import play_episode\n"
                "from games import base\nfrom games.base import Game\nfrom . import perfect"
            ),
            "eval/perfect.py": "",
            "ai/episode.py": "",
            "ai/network.py": "import torch",
        },
    )
    assert violations(root, "eval") == []


def test_allowed_imports_pass(tmp_path: Path) -> None:
    root = plant(
        tmp_path,
        {
            "ai/mcts.py": (
                "import numpy\nimport torch\nfrom games import base\n"
                "from games.base import Game\nfrom . import episode"
            ),
            "ai/episode.py": "",
            "engine/rules.py": (
                "from __future__ import annotations\nimport dataclasses\n"
                "from engine import pieces\nfrom .pieces import Piece"
            ),
            "engine/pieces.py": "",
        },
    )
    assert violations(root, "ai") == []
    assert violations(root, "engine") == []


def test_an_empty_ai_package_passes(tmp_path: Path) -> None:
    assert violations(plant(tmp_path, {}), "ai") == []


@pytest.mark.parametrize("forbidden", ["engine", "ai"])
def test_ui_does_not_import_the_engine_or_the_ai_directly(forbidden: str) -> None:
    """화면은 규칙을 `Game` 에 묻고(그 어댑터가 `engine/` 을 쓰는 것은 괜찮다) AI 는 주입받는다."""
    files = source_files(REPO_ROOT)
    modules = [name for name in files if name.split(".")[0] == "ui"]
    assert "ui.app" in modules
    # `imported_names` 는 직접 import 만 낸다.
    assert [m for m in modules if forbidden in set(imported_names(m, files))] == []
