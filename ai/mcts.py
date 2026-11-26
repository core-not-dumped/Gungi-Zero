"""신경망이 이끄는 몬테카를로 트리 탐색 (AlphaZero 의 PUCT).

트리는 받은 국면에서 수를 두어 만든 원래 국면들로만 이루어진다. `canonical` 은 후수 차례에서
경로 길이에 비례해 비싸서 부르지 않는다 — `encode` 는 원래 국면에 바로 부르고, 정책은 합법수의
행동 인덱스를 정규형 프레임으로 옮겨 읽는다. 전치표는 없다. `Position` 동등성은 경로를 포함하므로
판이 같은 두 국면도 반복 횟수가 다를 수 있다.

가치는 둘 차례인 쪽 시점이고 한 수를 거슬러 올라갈 때마다 부호가 뒤집힌다.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Generic

import numpy as np
import numpy.typing as npt
import torch

from ai.episode import value_for
from ai.network import Network, evaluate
from games.base import Game, MoveT, PositionT

DEFAULT_C_PUCT = 1.25


def canonical_action_indices(
    game: Game[PositionT, MoveT], position: PositionT, moves: Sequence[MoveT]
) -> list[int]:
    """원래 국면 `position` 의 `moves` 를 정규형 프레임의 행동 인덱스로. 같은 순서다."""
    return [game.to_canonical_action_index(position, game.action_index(move)) for move in moves]


@dataclass(frozen=True, slots=True)
class RootNoise:
    """루트의 사전 확률에 섞는 Dirichlet 노이즈. 자가 대국에서만 켠다."""

    alpha: float
    epsilon: float = 0.25


@dataclass(frozen=True, eq=False)
class SearchResult(Generic[MoveT]):
    """탐색 한 번의 결과."""

    # 루트의 합법수와 각 수를 방문한 횟수. 같은 순서다.
    moves: tuple[MoveT, ...]
    visits: tuple[int, ...]
    # `(action_space_size,)` 방문 분포. **정규형** 행동 인덱스이고 합이 1 이다.
    policy: npt.NDArray[np.float32]

    def select_move(self, temperature: float, rng: random.Random) -> MoveT:
        """방문 수의 `1 / temperature` 제곱에 비례해 고른다.

        0 이면 가장 많이 방문한 수를 고르고, 여럿이면 그중에서 무작위다.
        """
        if temperature < 0:
            raise ValueError(f"temperature 는 0 이상이다: {temperature}")
        most = max(self.visits)
        if temperature == 0:
            weights = [float(count == most) for count in self.visits]
        else:
            # 가장 큰 값으로 먼저 나눠 낮은 temperature 에서도 넘치지 않게 한다.
            weights = [(count / most) ** (1 / temperature) for count in self.visits]
        return rng.choices(self.moves, weights=weights)[0]


class _Node(Generic[PositionT, MoveT]):
    """트리의 한 국면. 처음 방문할 때 만들고 평가한다."""

    def __init__(self, position: PositionT) -> None:
        self.position = position
        self.visits = 0
        # 이 국면으로 오는 수를 **둔 쪽** 시점의 가치 합. 부모가 그대로 비교한다.
        self.value_sum = 0.0
        # 끝난 국면이면 둘 차례인 쪽 시점의 결과. 진행 중이면 `None` 이다.
        self.terminal_value: float | None = None
        self.moves: tuple[MoveT, ...] = ()
        self.priors: npt.NDArray[np.float64] = np.zeros(0)
        self.children: list[_Node[PositionT, MoveT] | None] = []


class MCTS(Generic[PositionT, MoveT]):
    """한 게임과 한 신경망의 탐색. 탐색 사이에 트리를 남기지 않는다."""

    def __init__(
        self,
        game: Game[PositionT, MoveT],
        network: Network,
        *,
        simulations: int,
        c_puct: float = DEFAULT_C_PUCT,
        root_noise: RootNoise | None = None,
        seed: int | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        if simulations < 1:
            raise ValueError(f"탐색 횟수는 1 이상이다: {simulations}")
        self._game = game
        self._network = network
        self._simulations = simulations
        self._c_puct = c_puct
        self._root_noise = root_noise
        self._rng = np.random.default_rng(seed)
        self._device = device

    def search(self, position: PositionT) -> SearchResult[MoveT]:
        """`position` 에서 탐색한다. 끝난 국면이거나 `canonical` 이 돌려준 국면이면 `ValueError`."""
        root: _Node[PositionT, MoveT] = _Node(position)
        self._expand(root)
        if root.terminal_value is not None:
            raise ValueError("끝난 국면은 탐색할 수 없다")
        # 루트를 평가한 것도 방문이다. 0 이면 첫 선택이 사전 확률을 보지 못한다.
        root.visits = 1
        if self._root_noise is not None:
            self._add_noise(root, self._root_noise)
        for _ in range(self._simulations):
            self._simulate(root)
        return self._result(root)

    def _simulate(self, root: _Node[PositionT, MoveT]) -> None:
        """루트에서 PUCT 로 내려가 새 국면 하나(또는 끝난 국면)에 닿고 그 가치를 거슬러 올린다."""
        path = [root]
        node = root
        while node.terminal_value is None:
            index = self._select(node)
            child = node.children[index]
            if child is None:
                child = _Node(self._game.apply_move(node.position, node.moves[index]))
                node.children[index] = child
                value = self._expand(child)
                path.append(child)
                break
            node = child
            path.append(node)
        else:
            value = node.terminal_value
        self._backup(path, value)

    def _expand(self, node: _Node[PositionT, MoveT]) -> float:
        """국면을 평가해 둘 차례인 쪽 시점의 가치를 돌려준다.

        끝난 국면은 신경망이 아니라 규칙으로 평가한다. 진행 중이면 합법수와 사전 확률을 채운다.
        """
        game = self._game
        outcome = game.outcome(node.position)
        if outcome is not None:
            node.terminal_value = value_for(outcome, node.position.player)
            return node.terminal_value

        node.moves = game.legal_moves(node.position)
        node.children = [None] * len(node.moves)
        logits, value = evaluate(self._network, game.encode(node.position), self._device)
        indices = canonical_action_indices(game, node.position, node.moves)
        legal = logits[indices].astype(np.float64)
        # 합법수만의 softmax. 가장 큰 로짓을 빼 넘치지 않게 한다.
        weights = np.exp(legal - legal.max())
        node.priors = weights / weights.sum()
        return value

    def _add_noise(self, root: _Node[PositionT, MoveT], noise: RootNoise) -> None:
        sample = self._rng.dirichlet([noise.alpha] * len(root.moves))
        root.priors = (1 - noise.epsilon) * root.priors + noise.epsilon * sample

    def _select(self, node: _Node[PositionT, MoveT]) -> int:
        """PUCT 점수가 가장 큰 수의 번호. 방문하지 않은 수의 가치는 0 으로 본다."""
        exploration = self._c_puct * math.sqrt(node.visits)

        def score(index: int) -> float:
            child = node.children[index]
            visits = 0 if child is None else child.visits
            value = 0.0 if child is None else child.value_sum / child.visits
            return value + exploration * node.priors[index] / (1 + visits)

        return max(range(len(node.moves)), key=score)

    def _backup(self, path: list[_Node[PositionT, MoveT]], value: float) -> None:
        """`value` 는 `path` 마지막 국면에서 둘 차례인 쪽 시점이다."""
        for node in reversed(path):
            # 이 국면으로 오는 수를 둔 쪽은 여기서 둘 차례인 쪽의 상대다.
            value = -value
            node.value_sum += value
            node.visits += 1

    def _result(self, root: _Node[PositionT, MoveT]) -> SearchResult[MoveT]:
        visits = tuple(0 if child is None else child.visits for child in root.children)
        policy = np.zeros(self._game.action_space_size, dtype=np.float32)
        indices = canonical_action_indices(self._game, root.position, root.moves)
        policy[indices] = np.asarray(visits, dtype=np.float32) / sum(visits)
        return SearchResult(root.moves, visits, policy)


class MCTSAI(Generic[PositionT, MoveT]):
    """탐색해 방문 수로 수를 고르는 AI (`CONTEXT.md`)."""

    def __init__(
        self,
        mcts: MCTS[PositionT, MoveT],
        *,
        temperature: float = 0.0,
        seed: int | None = None,
    ) -> None:
        self._mcts = mcts
        self._temperature = temperature
        self._rng = random.Random(seed)

    def choose(self, position: PositionT) -> MoveT:
        return self._mcts.search(position).select_move(self._temperature, self._rng)
