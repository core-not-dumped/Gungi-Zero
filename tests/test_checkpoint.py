"""체크포인트: 세대 이름, 폴더 나열, 구조를 가중치에서 읽기."""

from pathlib import Path

import pytest
import torch

from ai.checkpoint import (
    checkpoint_path,
    generation_files,
    generation_of,
    has_checkpoints,
    load_network,
    save_network,
)
from ai.resnet import ResNet, ResNetConfig
from scripts.shared.setups import SETUPS

GAME = SETUPS["nano-gungi"].game


def network(config: ResNetConfig) -> ResNet:
    return ResNet(GAME.tensor_shape, GAME.action_space_size, config)


def test_a_generation_round_trips_through_its_file_name(tmp_path: Path) -> None:
    path = checkpoint_path(tmp_path, 42)

    assert path.name == "generation-0042.pt"
    assert generation_of(path) == 42


@pytest.mark.parametrize("name", ["generation-.pt", "generation-0042.pth", "best-0042.pt", "a.pt"])
def test_other_file_names_have_no_generation(name: str) -> None:
    assert generation_of(Path(name)) is None


def test_a_folder_lists_its_generations_in_order(tmp_path: Path) -> None:
    saved = network(ResNetConfig(1, 4))
    for generation in (10, 2, 1):
        save_network(saved, tmp_path, generation)
    (tmp_path / "generation-notes.pt").touch()

    assert list(generation_files(tmp_path)) == [1, 2, 10]
    assert has_checkpoints(tmp_path)


def test_an_empty_folder_has_no_generations(tmp_path: Path) -> None:
    assert not has_checkpoints(tmp_path)
    with pytest.raises(ValueError, match="체크포인트가 없다"):
        generation_files(tmp_path)


@pytest.mark.parametrize(("blocks", "channels"), [(1, 4), (3, 8)])
def test_a_checkpoint_of_another_size_is_read_from_its_weights(
    blocks: int, channels: int, tmp_path: Path
) -> None:
    saved = network(ResNetConfig(blocks, channels))
    path = save_network(saved, tmp_path, 1)

    loaded = load_network(GAME, path)

    assert loaded.config == ResNetConfig(blocks, channels)
    for name, weights in saved.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name], weights)
