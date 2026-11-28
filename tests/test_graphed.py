"""GPU 명령을 한 번 기록해 재생하는 평가기. CUDA 가 없으면 건너뛴다."""

import pytest
import torch

from ai.graphed import GraphedNetwork
from ai.resnet import ResNet, ResNetConfig

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA 가 없다")

SHAPE = (18, 4, 3)
ACTIONS = 288


def network() -> ResNet:
    torch.manual_seed(0)
    return ResNet(SHAPE, ACTIONS, ResNetConfig(2, 16)).to("cuda").eval()


def boards(count: int, seed: int = 1) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.rand(count, *SHAPE).to("cuda")


def assert_close(graphed: GraphedNetwork, plain: ResNet, batch: torch.Tensor) -> None:
    with torch.inference_mode():
        expected = plain(batch)
        answered = graphed(batch)
    for one, other in zip(answered, expected, strict=True):
        torch.testing.assert_close(one, other, rtol=0, atol=1e-3)


@pytest.mark.parametrize("count", [1, 3, 8, 11])
def test_it_answers_like_the_network_for_any_batch_size(count: int) -> None:
    plain = network()
    assert_close(GraphedNetwork(plain, SHAPE, 8, "cuda"), plain, boards(count))


def test_earlier_answers_survive_the_next_call() -> None:
    plain = network()
    graphed = GraphedNetwork(plain, SHAPE, 8, "cuda")
    with torch.inference_mode():
        logits, values = graphed(boards(8, seed=1))
        kept = logits.clone(), values.clone()
        graphed(boards(8, seed=2))

    torch.testing.assert_close(logits, kept[0])
    torch.testing.assert_close(values, kept[1])


def test_a_new_recording_follows_weights_changed_in_place() -> None:
    plain = network()
    GraphedNetwork(plain, SHAPE, 8, "cuda")
    plain.train()
    optimizer = torch.optim.Adam(plain.parameters(), lr=0.1)
    logits, values = plain(boards(8))
    (logits.sum() + values.sum()).backward()
    optimizer.step()
    plain.eval()

    assert_close(GraphedNetwork(plain, SHAPE, 8, "cuda"), plain, boards(5))


def test_a_network_in_training_mode_is_refused() -> None:
    with pytest.raises(ValueError):
        GraphedNetwork(network().train(), SHAPE, 8, "cuda")
