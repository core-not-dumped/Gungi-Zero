"""Elo 난이도 단계: 고르기, 저장하기, 상대 만들기."""

from pathlib import Path

import pytest
import torch

from ai.resnet import ResNet
from scripts.play import level_options
from scripts.shared.levels import (
    Level,
    LevelSet,
    load_levels,
    opponent_factory,
    pick_levels,
    save_levels,
)
from scripts.shared.setups import SETUPS

SETUP = SETUPS["nano-gungi"]


def level(rating: float, probability: float = 0.0, checkpoint: str = "a.pt") -> Level:
    return Level(rating, checkpoint, probability)


def test_the_closest_candidate_is_picked_for_every_step() -> None:
    candidates = [level(70), level(190), level(240), level(385), level(409)]

    assert pick_levels(candidates, step=200) == [level(190), level(409)]


def test_a_step_without_a_near_candidate_is_left_empty() -> None:
    candidates = [level(40), level(420), level(610)]

    # 200 에서 반 칸(100) 안의 후보가 없다.
    assert pick_levels(candidates, step=200) == [level(420), level(610)]


def test_one_candidate_serves_only_one_step() -> None:
    assert pick_levels([level(300)], step=200) == [level(300)]


def test_levels_survive_saving_and_loading(tmp_path: Path) -> None:
    levels = LevelSet("nano-gungi", 100, (level(201.5, 0.4), level(409)))
    path = tmp_path / "sub" / "levels.json"

    save_levels(levels, path)

    assert load_levels(path) == levels


def test_an_opponent_plays_legal_moves_from_a_saved_checkpoint(tmp_path: Path) -> None:
    game = SETUP.game
    torch.manual_seed(0)
    network = ResNet(game.tensor_shape, game.action_space_size, SETUP.network)
    path = tmp_path / "generation-0001.pt"
    torch.save(network.state_dict(), path)

    make = opponent_factory(SETUP, level(100, 0.5, str(path)), simulations=2)
    choose = make(game, 3)
    position = game.initial_position()
    while game.outcome(position) is None:
        move = choose(position)
        assert move in game.legal_moves(position)
        position = game.apply_move(position, move)


def test_a_missing_checkpoint_is_reported_when_the_factory_is_built(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        opponent_factory(SETUP, level(100, 0.0, str(tmp_path / "none.pt")), simulations=2)


def test_the_play_screen_offers_random_first_then_the_saved_levels(tmp_path: Path) -> None:
    game = SETUP.game
    network = ResNet(game.tensor_shape, game.action_space_size, SETUP.network)
    checkpoint = tmp_path / "generation-0001.pt"
    torch.save(network.state_dict(), checkpoint)
    path = tmp_path / "nano-gungi.json"
    save_levels(LevelSet("nano-gungi", 2, (level(201.4, 0.4, str(checkpoint)),)), path)

    offered = level_options(tmp_path)

    # 단계 파일이 있는 게임의 판에만 난이도가 있다.
    assert list(offered) == ["나노 3x4"]
    options = offered["나노 3x4"]

    assert [name for name, _ in options] == ["Elo 0", "Elo 201"]


def test_without_a_level_file_the_play_screen_has_no_levels(tmp_path: Path) -> None:
    assert level_options(tmp_path / "none") == {}
