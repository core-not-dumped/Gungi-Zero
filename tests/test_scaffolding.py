"""스캐폴딩 스모크 테스트: flat 레이아웃의 패키지들이 실제로 import 된다."""

import importlib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# 있어야 하는 패키지를 손으로 적은 목록이다. 디렉터리를 훑어 만들면 동어반복이 된다.
PACKAGES = ["games", "engine", "ai", "eval", "ui", "scripts", "tests"]


@pytest.mark.parametrize("name", PACKAGES)
def test_package_imports_from_the_repository_root(name: str) -> None:
    """flat 레이아웃이다 — src/ 아래가 아니라 저장소 루트 바로 아래에 있고, 그대로 import 된다."""
    assert (REPO_ROOT / name / "__init__.py").is_file()
    assert importlib.import_module(name).__file__ is not None
