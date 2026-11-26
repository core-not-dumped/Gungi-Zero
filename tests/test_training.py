"""세대 루프. 주입 seam(신경망, 로거)과 체크포인트에서 보이는 것만 본다."""

import math
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch

from ai.logger import MemoryLogger
from ai.resnet import ResNet, ResNetConfig
from ai.self_play import SelfPlayConfig
from ai.training import TrainingConfig, seed_everything, train
from games.base import Game
from games.mini_gungi import Gungi, PieceType, Placement
from games.nano_gungi import NANO_GUNGI
from games.tictactoe import TicTacToe
from scripts.hyperparameters import NANO_GUNGI_SELF_PLAY, TICTACTOE_SELF_PLAY

# 한 수 만에 수 제한 무승부로 끝나는 나노. 학습 샘플이 모두 초기 국면이고 가치 타깃이 0 이다.
ONE_MOVE_NANO = Gungi(replace(NANO_GUNGI, move_limit=1))


def small_config(self_play: SelfPlayConfig = TICTACTOE_SELF_PLAY, **changes: Any) -> TrainingConfig:
    config = TrainingConfig(
        self_play=replace(self_play, simulations=4),
        episodes_per_generation=3,
        buffer_generations=2,
        steps_per_generation=4,
        batch_size=8,
    )
    return replace(config, **changes)


class SpyNetwork(torch.nn.Module):
    """균등 정책과 상수 가치를 내고, 어떤 모드로 무엇을 받았는지 남긴다."""

    def __init__(self, action_space_size: int, value: float = 0.0) -> None:
        super().__init__()
        self._action_space_size = action_space_size
        self._value = value
        self.weight = torch.nn.Parameter(torch.ones(()))
        # 부른 순서대로 (학습 모드인지, 입력).
        self.calls: list[tuple[bool, torch.Tensor]] = []

    def forward(self, boards: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.calls.append((self.training, boards.clone()))
        # 출력은 가중치와 무관하지만 기울기는 흐를 수 있어야 한다.
        zero = self.weight * 0
        batch = boards.shape[0]
        return (
            zero + torch.zeros(batch, self._action_space_size),
            zero + torch.full((batch,), self._value),
        )


def run(
    game: Game,
    tmp_path: Path,
    config: TrainingConfig,
    *,
    network: torch.nn.Module | None = None,
    generations: int = 1,
    seed: int = 0,
    end_reason: Callable[[Any], str] = lambda position: "end",
) -> MemoryLogger:
    logger = MemoryLogger()
    train(
        game,
        network or SpyNetwork(game.action_space_size),
        config,
        generations=generations,
        logger=logger,
        end_reason=end_reason,
        checkpoint_dir=tmp_path,
        seed=seed,
    )
    return logger


def test_a_generation_plays_in_evaluation_mode_and_then_trains(tmp_path: Path) -> None:
    network = SpyNetwork(TicTacToe().action_space_size)

    run(TicTacToe(), tmp_path, small_config(), network=network, generations=2)

    modes = [training for training, _ in network.calls]
    # 자가 대국(평가) → M 스텝(학습) 이 세대마다 되풀이된다.
    switches = [mode for i, mode in enumerate(modes) if i == 0 or modes[i - 1] != mode]
    assert switches == [False, True, False, True]
    assert sum(modes) == 2 * 4
    assert all(boards.shape == (8, 2, 3, 3) for training, boards in network.calls if training)
    assert not network.training


def test_each_episode_gets_its_own_seed(tmp_path: Path) -> None:
    game = TicTacToe()
    network = SpyNetwork(game.action_space_size)

    logger = run(
        game,
        tmp_path,
        small_config(episodes_per_generation=20),
        network=network,
        generations=2,
    )

    assert any(m["self_play/length_min"] < m["self_play/length_max"] for _, m in logger.records)
    # 신경망의 출력이 같아도 두 세대의 대국은 다르다.
    plays: list[list[bytes]] = [[]]
    for training, boards in network.calls:
        if not training:
            plays[-1].append(boards.numpy().tobytes())
        elif plays[-1]:
            plays.append([])
    assert plays[0] != plays[1]


def test_the_loss_is_the_masked_cross_entropy_plus_the_value_error(tmp_path: Path) -> None:
    game = ONE_MOVE_NANO
    legal = len(game.legal_moves(game.initial_position()))
    config = small_config(NANO_GUNGI_SELF_PLAY)

    logger = run(game, tmp_path, config, network=SpyNetwork(game.action_space_size, value=0.5))

    [(step, metrics)] = logger.records
    assert step == 1
    # 균등 정책의 교차 엔트로피는 합법수 위에서 재면 타깃과 무관하게 log(합법수) 다.
    assert metrics["loss/policy"] == pytest.approx(math.log(legal))
    assert metrics["loss/value"] == pytest.approx(0.25)
    assert metrics["loss/total"] == pytest.approx(math.log(legal) + 0.25)


# 帥 둘이 대각으로 붙어 있어 첫 수에 잡을 수 있는 군의. 다른 수는 그다음에 帥 를 잡힌다.
CAPTURE_AT_ONCE = Gungi(
    replace(NANO_GUNGI, width=2, height=2, setup=(Placement(PieceType.MARSHAL, 0, 0),))
)


class TableNetwork(torch.nn.Module):
    """국면과 무관한 정책 로짓을 학습한다. 가치는 학습 모드에서만 0.5 다."""

    def __init__(self, action_space_size: int) -> None:
        super().__init__()
        self.logits = torch.nn.Parameter(torch.zeros(action_space_size))

    def forward(self, boards: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = boards.shape[0]
        # 평가 때는 0 이라야 탐색이 가치에 끌리지 않는다.
        value = 0.5 if self.training else 0.0
        return self.logits.repeat(batch, 1), self.logits.sum() * 0 + torch.full((batch,), value)


def test_training_moves_toward_the_search_policy_and_the_result(tmp_path: Path) -> None:
    game = CAPTURE_AT_ONCE
    start = game.initial_position()
    [capture] = [
        move
        for move in game.legal_moves(start)
        if game.outcome(game.apply_move(start, move)) is not None
    ]
    network = TableNetwork(game.action_space_size)
    config = small_config(NANO_GUNGI_SELF_PLAY, steps_per_generation=50, learning_rate=0.1)
    config = replace(
        config, self_play=replace(config.self_play, simulations=64, exploration_moves=0)
    )

    logger = run(game, tmp_path, config, network=network)

    # 모든 대국이 한 수의 선수 승이라 가치 타깃은 +1 이다. 부호가 뒤집히면 2.25 다.
    assert logger.records[0][1]["self_play/length_max"] == 1
    assert logger.records[0][1]["loss/value"] == pytest.approx(0.25)
    legal = [game.action_index(move) for move in game.legal_moves(start)]
    assert len(legal) > 1
    best = max(legal, key=lambda index: network.logits[index].item())
    assert best == game.action_index(capture)


def test_the_self_play_statistics_are_logged(tmp_path: Path) -> None:
    game = ONE_MOVE_NANO

    def end_reason(position: Any) -> str:
        reason = game.end_reason(position)
        assert reason is not None
        return reason.name

    logger = run(game, tmp_path, small_config(NANO_GUNGI_SELF_PLAY), end_reason=end_reason)

    [(_, metrics)] = logger.records
    assert metrics["self_play/length_mean"] == 1
    assert metrics["self_play/length_min"] == 1
    assert metrics["self_play/length_max"] == 1
    assert metrics["self_play/outcome/first_wins"] == 0
    assert metrics["self_play/outcome/second_wins"] == 0
    assert metrics["self_play/outcome/draw"] == 3
    assert metrics["self_play/end_reason/MOVE_LIMIT"] == 3


def test_the_replay_buffer_keeps_the_last_generations(tmp_path: Path) -> None:
    logger = run(ONE_MOVE_NANO, tmp_path, small_config(NANO_GUNGI_SELF_PLAY), generations=3)

    assert [step for step, _ in logger.records] == [1, 2, 3]
    assert [m["replay_buffer/samples"] for _, m in logger.records] == [3, 6, 6]


def train_resnet(directory: Path, seed: int) -> torch.nn.Module:
    game = TicTacToe()
    seed_everything(seed)
    network = ResNet(game.tensor_shape, game.action_space_size, ResNetConfig(1, 8))
    run(game, directory, small_config(), network=network, generations=2, seed=seed)
    return network


def load(path: Path) -> dict[str, torch.Tensor]:
    return torch.load(path, weights_only=True)


def same(one: dict[str, torch.Tensor], other: dict[str, torch.Tensor]) -> bool:
    return one.keys() == other.keys() and all(torch.equal(one[name], other[name]) for name in one)


def test_every_generation_leaves_a_checkpoint(tmp_path: Path) -> None:
    network = train_resnet(tmp_path, seed=0)

    paths = sorted(tmp_path.iterdir())
    assert [path.name for path in paths] == ["generation-0001.pt", "generation-0002.pt"]
    first, second = (load(path) for path in paths)
    assert not same(first, second)
    assert same(second, network.state_dict())


def test_the_same_seed_gives_the_same_generations_on_the_cpu(tmp_path: Path) -> None:
    train_resnet(tmp_path / "first", seed=0)
    train_resnet(tmp_path / "again", seed=0)
    train_resnet(tmp_path / "other", seed=1)

    for name in ("generation-0001.pt", "generation-0002.pt"):
        first = load(tmp_path / "first" / name)
        assert same(first, load(tmp_path / "again" / name))
        assert not same(first, load(tmp_path / "other" / name))


def test_an_existing_checkpoint_is_not_overwritten(tmp_path: Path) -> None:
    # 이번에 쓸 번호가 아니어도 다른 학습의 것이다.
    existing = tmp_path / "generation-0009.pt"
    existing.write_bytes(b"kept")
    network = SpyNetwork(TicTacToe().action_space_size)

    with pytest.raises(FileExistsError):
        run(TicTacToe(), tmp_path, small_config(), network=network, generations=2)

    assert existing.read_bytes() == b"kept"
    # 대국에 시간을 쓰기 전에 멈춘다.
    assert not network.calls


@pytest.mark.parametrize(
    "field",
    ["episodes_per_generation", "buffer_generations", "steps_per_generation", "batch_size"],
)
def test_a_config_that_would_do_nothing_is_rejected(field: str) -> None:
    with pytest.raises(ValueError):
        replace(small_config(), **{field: 0})


def test_the_optimizer_defaults_follow_the_spec() -> None:
    config = small_config()

    assert (config.learning_rate, config.weight_decay) == (1e-3, 1e-4)
