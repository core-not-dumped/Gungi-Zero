"""신경망 자리에 꽂는 시험용 신경망."""

import torch


class UniformNetwork:
    """균등 정책, 가치 0. 맞는 수는 탐색이 종료 국면을 정확히 평가하는 데에서만 나온다."""

    def __init__(self, tensor_shape: tuple[int, int, int], action_space_size: int) -> None:
        self._tensor_shape = tensor_shape
        self._action_space_size = action_space_size
        # 몇 번 평가했는지. 탐색이 신경망을 부르지 않아야 하는 곳을 시험한다.
        self.calls = 0

    def __call__(self, boards: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assert boards.shape[1:] == self._tensor_shape
        self.calls += 1
        batch = boards.shape[0]
        return torch.zeros(batch, self._action_space_size), torch.zeros(batch)
