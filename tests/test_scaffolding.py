"""스캐폴딩 스모크 테스트: flat 레이아웃의 패키지들이 실제로 import 된다."""

import importlib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

PACKAGES = ["games", "engine", "ai", "eval", "ui", "tests"]


@pytest.mark.parametrize("name", PACKAGES)
def test_package_is_importable(name: str) -> None:
    module = importlib.import_module(name)
    assert module.__file__ is not None


@pytest.mark.parametrize("name", PACKAGES)
def test_package_sits_in_the_repository_root(name: str) -> None:
    """flat 레이아웃이다 — src/ 아래가 아니라 저장소 루트 바로 아래에 있다."""
    assert (REPO_ROOT / name / "__init__.py").is_file()
