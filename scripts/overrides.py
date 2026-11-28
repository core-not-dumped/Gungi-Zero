"""학습 설정의 값을 이름으로 덮어쓴다. 튜닝 실험이 설정 파일을 고치지 않고 값을 바꾼다."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, replace
from typing import Any

from ai.resnet import ResNetConfig
from ai.self_play import SelfPlayConfig
from ai.training import TrainingConfig
from scripts.train import GameSetup

_INTEGERS = (
    "episodes_per_generation",
    "buffer_generations",
    "steps_per_generation",
    "batch_size",
    "simulations",
    "exploration_moves",
    "parallel_games",
    "evaluation_cache",
    "blocks",
    "channels",
)
_REALS = (
    "learning_rate",
    "weight_decay",
    "c_puct",
    "dirichlet_alpha",
    "dirichlet_epsilon",
    "initial_start_fraction",
)
# 덮어쓸 수 있는 이름 → 값의 자료형.
OVERRIDABLE: dict[str, type] = {
    **{name: int for name in _INTEGERS},
    **{name: float for name in _REALS},
}
# 덮어쓰는 이름과 설정의 필드 이름이 다른 것.
_NOISE = {"dirichlet_alpha": "alpha", "dirichlet_epsilon": "epsilon"}


def apply_overrides(setup: GameSetup, overrides: Mapping[str, float]) -> GameSetup:
    """`setup` 에서 `overrides` 의 값만 바꾼 설정. 모르는 이름은 `ValueError` 다."""
    unknown = sorted(set(overrides) - set(OVERRIDABLE))
    if unknown:
        raise ValueError(f"덮어쓸 수 없는 이름이다: {unknown}")

    def chosen(config: type) -> dict[str, Any]:
        names = {field.name for field in fields(config)}
        return {name: value for name, value in overrides.items() if name in names}

    noise = replace(
        setup.training.self_play.root_noise,
        **{_NOISE[name]: value for name, value in overrides.items() if name in _NOISE},
    )
    self_play = replace(setup.training.self_play, root_noise=noise, **chosen(SelfPlayConfig))
    training = replace(setup.training, self_play=self_play, **chosen(TrainingConfig))
    return replace(
        setup,
        training=training,
        network=replace(setup.network, **chosen(ResNetConfig)),
        initial_start_fraction=overrides.get(
            "initial_start_fraction", setup.initial_start_fraction
        ),
    )
