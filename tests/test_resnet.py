"""ResNet. 모양(값의 범위 포함)과 시드 재현성을 본다 — 성능 테스트는 두지 않는다."""

import pytest
import torch

from ai.network import Network
from ai.resnet import ResNet, ResNetConfig
from games.base import Game
from games.mini_gungi import Gungi
from games.nano_gungi import NANO_GUNGI
from games.tictactoe import TicTacToe

SMALL = ResNetConfig(blocks=2, channels=8)


@pytest.mark.parametrize(
    ("game", "tensor_shape", "actions"),
    [(TicTacToe(), (2, 3, 3), 9), (Gungi(NANO_GUNGI), (18, 4, 3), 288)],
    ids=["tictactoe", "nano-gungi"],
)
def test_the_resnet_follows_the_network_protocol(
    game: Game, tensor_shape: tuple[int, int, int], actions: int
) -> None:
    assert (game.tensor_shape, game.action_space_size) == (tensor_shape, actions)
    network: Network = ResNet(game.tensor_shape, game.action_space_size, SMALL).eval()
    # 가치가 범위를 벗어나는지 보려고 큰 값을 섞는다.
    boards = torch.randn(5, *tensor_shape) * 100

    logits, values = network(boards)

    assert logits.shape == (5, actions)
    assert values.shape == (5,)
    assert values.abs().max() <= 1


def build(seed: int) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    return ResNet((2, 3, 3), 9, SMALL).state_dict()


def test_the_same_seed_builds_the_same_weights() -> None:
    first, second, other = build(0), build(0), build(1)

    assert all(torch.equal(first[name], second[name]) for name in first)
    assert not all(torch.equal(first[name], other[name]) for name in first)


@pytest.mark.parametrize("config", [(0, 8), (1, 0)])
def test_an_empty_structure_is_rejected(config: tuple[int, int]) -> None:
    with pytest.raises(ValueError):
        ResNetConfig(*config)
