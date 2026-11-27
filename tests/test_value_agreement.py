"""가치 머리 진단. 시험용 신경망을 꽂아 일치로 센 국면 수를 본다."""

from collections.abc import Callable

import pytest
import torch

from eval.solver import Solution, Verdict, solve
from eval.value_agreement import Agreement, value_agreement
from games.tictactoe import TicTacToe
from tests.solver_helpers import shortest_path_positions


@pytest.fixture(scope="module")
def tictactoe() -> Solution:
    return solve(TicTacToe(), move_limit=None)


class ConstantValue:
    def __init__(self, value: float) -> None:
        self._value = value

    def __call__(self, boards: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = boards.shape[0]
        return torch.zeros(batch, 9), torch.full((batch,), self._value)


def count_verdicts(solution: Solution) -> Callable[[Verdict], int]:
    verdicts = [solution.value(p).verdict for p in shortest_path_positions(solution.game)]
    return verdicts.count


@pytest.mark.parametrize(
    ("value", "verdict"),
    [
        (1.0, Verdict.WIN),
        (0.51, Verdict.WIN),
        (0.49, Verdict.DRAW),
        (-0.49, Verdict.DRAW),
        (-0.51, Verdict.LOSS),
        (-1.0, Verdict.LOSS),
    ],
)
def test_a_value_counts_as_the_nearest_result(
    tictactoe: Solution, value: float, verdict: Verdict
) -> None:
    agreement = value_agreement(tictactoe, ConstantValue(value))

    # 도달 가능한 5,478 국면 전부를 본다.
    assert agreement == Agreement(matches=count_verdicts(tictactoe)(verdict), positions=5478)
    assert agreement.rate == agreement.matches / 5478


class CenterOwner:
    """둘 차례인 쪽의 말이 가운데에 있으면 +1, 상대의 말이면 -1, 비었으면 0."""

    def __call__(self, boards: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.zeros(boards.shape[0], 9), boards[:, 0, 1, 1] - boards[:, 1, 1, 1]


def test_each_position_is_judged_by_its_own_value_whatever_the_batch(tictactoe: Solution) -> None:
    game = tictactoe.game
    expected = 0
    for position in shortest_path_positions(game):
        center = game.encode(position)[:, 1, 1]
        guess = {1: Verdict.WIN, 0: Verdict.DRAW, -1: Verdict.LOSS}[int(center[0] - center[1])]
        expected += tictactoe.value(position).verdict is guess

    for batch_size in (1, 7, 10_000):
        agreement = value_agreement(tictactoe, CenterOwner(), batch_size=batch_size)
        assert agreement == Agreement(expected, 5478)
