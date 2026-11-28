# Stable-Baselines3 는 여러 환경을 어떻게 묶어 처리하는가

조사일 2026-11-28. 1차 자료(소스 코드, 공식 문서, 저장소 이슈)만 근거로 했다. 코드는 실행하지 않고 읽기만 했다.

| 대상 | 태그 | 커밋 |
| --- | --- | --- |
| [`DLR-RM/stable-baselines3`](https://github.com/DLR-RM/stable-baselines3) (SB3) | `v2.9.2` | `0cdcd200deef06b9ad0af72f6ddfb07c06c6df39` |
| [`Stable-Baselines-Team/stable-baselines3-contrib`](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib) | `v2.9.2` | `a8af63682070df0bda9707400b2e971055600910` |
| [`Farama-Foundation/Gymnasium`](https://github.com/Farama-Foundation/Gymnasium) | `v1.4.0` | `8ba8c3a63bb4e000b7f33923f43d292acac07be1` |
| [`hill-a/stable-baselines`](https://github.com/hill-a/stable-baselines) (옛 TensorFlow 판) | `v2.10.1` | `e54554d872e87bdf6068c9a508d1328c84330b68` |

출처의 `[n]` 은 문서 끝의 목록이다. `L` 뒤의 수는 그 태그에서의 줄 번호다. **[추정]** 은 자료에 없는 이 문서의 판단이고, **[계산]** 은 인용한 사실에서 따라 나오는 것이다.

## 1. `VecEnv` 의 구조

### 인터페이스

`VecEnv` 는 환경 `num_envs` 개를 환경 하나처럼 다루는 추상 클래스다. `step(actions)` 는 길이 `n_envs` 의 행동 배열을 받아 `obs, rewards, dones, infos` 를 돌려준다. `obs`·`rewards`·`dones` 는 첫 축이 환경 번호인 NumPy 배열이고 `infos` 는 dict 의 목록이다. Gymnasium `Env` 의 5개 반환값과 달리 `dones = terminated or truncated` 하나이고, 시간 제한으로 끊긴 것은 `infos[i]["TimeLimit.truncated"]` 로 구분한다 [1] [2].

`step` 은 두 단계의 합이다 [1] `VecEnv.step` (L214-222).

```python
def step(self, actions):
    self.step_async(actions)
    return self.step_wait()
```

`step_async` 는 모든 환경에 한 걸음을 시작하라고 알리고, `step_wait` 는 결과를 기다린다. 앞의 것이 끝나지 않았는데 `step_async` 를 다시 부르면 안 된다 [1] (L122-141). 읽은 두 수집 함수는 `env.step()` 만 부르므로 `step_async` 와 `step_wait` 사이에 다른 일을 끼워 넣지 않는다 [5] (L218), [6] (L580).

### `DummyVecEnv` 와 `SubprocVecEnv`

| | `DummyVecEnv` [3] | `SubprocVecEnv` [4] |
| --- | --- | --- |
| 실행 | 현재 프로세스에서 환경을 차례로 부른다 | 환경마다 프로세스 하나 |
| `step_async` | 행동을 저장만 한다 (L53-54) | 각 파이프에 `("step", action)` 을 보낸다 (L131-134) |
| `step_wait` | `for env_idx in range(num_envs)` 로 `env.step` (L56-73) | `[remote.recv() for remote in self.remotes]` 로 모두 받아 `np.stack` (L136-140) |
| 쓸 곳(docstring) | `CartPole-v1` 처럼 가벼운 환경. 프로세스 비용이 계산보다 크다 | 계산이 무거운 환경. IO 위주가 아니면 환경 수를 논리 코어 수 이하로 |

`make_vec_env` 의 기본값은 `DummyVecEnv` 이고 docstring 은 이것이 보통 더 빠르다고 적는다 [7] `make_vec_env` (L54-56, L121-126). PPO 문서는 CNN 이 아닐 때 CPU 에서 돌리고 CPU 활용을 높이려면 `SubprocVecEnv` 를 써 보라고 한다 [8].

### 프로세스 간 통신

[4] 에서 확인한 것.

- 환경마다 `ctx.Pipe()` 하나와 `daemon=True` 프로세스 하나를 만든다 (L116-124). 환경 생성 함수는 `CloudpickleWrapper` 로 감싸 넘긴다.
- 자식의 `_worker` 는 `remote.recv()` 로 `(cmd, data)` 를 받아 처리하는 무한 루프다. `step`, `reset`, `render`, `close`, `get_spaces`, `env_method`, `get_attr`, `has_attr`, `set_attr`, `is_wrapped` 를 받는다 (L31-72).
- 부모는 전부 보낸 다음 **번호 순서대로** 받는다. 준비된 것부터 받는 코드는 없다 (L132-137).
- 시작 방식의 기본값은 `forkserver` 가 있으면 `forkserver`, 없으면 `spawn` 이다 (L108-113). 같은 태그의 문서 [2] 는 "Linux 의 기본 시작 방식은 `fork`" 라고 적어 코드와 다르게 읽힌다. 이 문서는 코드를 따른다. `forkserver`·`spawn` 에서는 `if __name__ == "__main__":` 이 필요하다 [2].

### 끝난 환경의 자동 reset

두 구현 모두 `step` 안에서, 끝난 환경을 **그 자리에서** reset 한다 [3] (L63-72), [4] (L37-43).

```python
done = terminated or truncated
info["TimeLimit.truncated"] = truncated and not terminated
if done:
    # save final observation where user can get it, then reset
    info["terminal_observation"] = observation
    observation, reset_info = env.reset()
```

- 반환하는 `obs[i]` 는 **다음 대국의 첫 관측**이고, 끝난 대국의 마지막 관측은 `infos[i]["terminal_observation"]` 에 있다 [2].
- reset 정보는 `vec_env.reset_infos` 에 남는다. 사용자는 처음 한 번만 `reset()` 을 부른다 [2].
- `SubprocVecEnv` 에서는 reset 이 자식 안에서 일어난 뒤에 응답을 보낸다. **[계산]** 느린 reset 도 그 `step_wait` 의 대기에 들어간다.
- `seed()` 는 환경 `idx` 에 `seed + idx` 를 정해 두고 다음 `reset()` 에 한 번만 넘긴다 [1] (L292-309). 자동 reset 은 `env.reset()` 을 인자 없이 부른다.

## 2. 신경망 추론이 묶이는 지점

핵심은 하나다. `VecEnv` 가 돌려주는 `(n_envs, *obs_shape)` 배열이 **그대로 배치 축**이 되어 forward 한 번에 들어간다.

### on-policy (PPO, A2C): `OnPolicyAlgorithm.collect_rollouts` [5]

```python
while n_steps < n_rollout_steps:
    with th.no_grad():
        obs_tensor = obs_as_tensor(self._last_obs, self.device)  # (n_envs, ...)
        actions, values, log_probs = self.policy(obs_tensor)  # forward 1회
    actions = actions.cpu().numpy()
    ...
    new_obs, rewards, dones, infos = env.step(clipped_actions)  # 환경 n_envs 개
    self.num_timesteps += env.num_envs
```

(L194-220 에서 간추림.) `ActorCriticPolicy.forward` 는 행동·가치·로그 확률을 함께 낸다 [9] (L636-658).

배치가 아닌 forward 가 두 곳 더 있다.

- 시간 제한으로 끊긴 환경은 `terminal_observation` 의 가치를 **환경마다 하나씩** `predict_values` 로 구해 보상에 더한다 (L236-245).
- 수집이 끝나면 마지막 관측 배치의 가치를 한 번 더 계산해 GAE 의 끝값으로 쓴다 (L258-262).

행동에 마스크가 필요하면 contrib 의 `MaskablePPO` 가 `env.env_method("action_masks")` 로 환경마다 마스크를 받아 `np.stack` 하고, `self.policy(obs_tensor, action_masks=action_masks)` 로 같은 배치에 넘긴다 [12] (L227-233), [13] `get_action_masks` (L16-17).

### rollout buffer 의 모양

[10] `RolloutBuffer` 에서 확인한 것.

| 단계 | 모양 |
| --- | --- |
| `reset` (L391-401) | `observations` `(n_steps, n_envs, *obs_shape)`, `actions` `(n_steps, n_envs, action_dim)`, `rewards`·`returns`·`episode_starts`·`values`·`log_probs`·`advantages` `(n_steps, n_envs)` |
| `add` (L440-479) | 호출 한 번이 한 시각의 `n_envs` 줄을 채운다 |
| `compute_returns_and_advantage` (L403-438) | 시간 축을 거꾸로 돌며 환경별로 벡터 계산. 대국 경계는 `episode_starts` 로 끊는다 |
| `get` (L481-506) | `swap_and_flatten` 으로 `(n_steps * n_envs, ...)` 로 편 뒤(환경별 순서 유지, L63-75) 섞어 미니배치로 낸다 |

한 번의 갱신에 쓰는 데이터는 `n_steps * n_envs` 개의 **걸음**이다 [11]. PPO 는 이것을 `n_epochs` 번 돌며 `batch_size` 씩 학습한다 [14] (L204-213). 수집의 배치 크기(`n_envs`)와 학습의 배치 크기(`batch_size`)는 별개다.

### off-policy (DQN, SAC, TD3 등): `OffPolicyAlgorithm.collect_rollouts` [6]

- 반복마다 `_sample_action(learning_starts, action_noise, env.num_envs)` → `env.step(actions)` → `_store_transition(...)` 이다 (L571-595).
- `_sample_action` 은 `self.predict(self._last_obs, deterministic=False)` 를 부른다 (L409-410). `BasePolicy.predict` 는 `obs_to_tensor` 로 `(-1, *obs_shape)` 모양을 만들어 `th.no_grad()` 에서 `_predict` 를 한 번 부른다 [9] (L270-277, L365-370).
- `learning_starts` 전의 준비 구간에서는 신경망 없이 환경 수만큼 무작위 행동을 뽑는다 (L401-403).
- DQN 의 epsilon-greedy 는 난수를 **배치 전체에 한 번** 뽑는다. 걸리면 모든 환경이 무작위 행동이고, 아니면 모두 신경망의 greedy 행동이다 [15] `DQN.predict` (L253-263), [16] `_predict` (L68-72).
- 환경이 둘 이상이면 `train_freq` 의 단위는 `step` 이어야 한다. 대국 단위 수집은 환경 하나일 때만 된다 (L563-564).
- `_store_transition` 은 `done` 인 환경의 `next_obs` 를 `terminal_observation` 으로 바꿔 저장하고, 다음 추론에는 새 대국의 관측(`new_obs`)을 쓴다 (L492-521).
- 환경이 여럿이면 `gradient_steps` 도 맞추라고 문서가 경고한다. `-1` 이면 모은 전이 수만큼 학습한다 [17].

`ReplayBuffer` 는 `(buffer_size, n_envs, ...)` 로 저장하고 `buffer_size = max(buffer_size // n_envs, 1)` 로 줄인다. 표본은 시각 번호와 환경 번호를 각각 무작위로 뽑는다 [10] (L198, L213-224, L307-309).

## 3. 대국 길이가 다를 때와 동기 방식의 병목

### 묶음 크기

끝난 환경이 같은 `step` 안에서 다시 시작하므로 `step` 의 결과는 항상 `n_envs` 줄이다. 수집은 대국이 아니라 걸음 수(`n_steps`, 또는 `train_freq`)로 끝난다 [5] (L194), [11]. 그래서 **[계산]** 긴 대국만 남아 묶음이 줄어드는 구간이 없다. 대가는 대국 중간에서 수집이 잘리는 것이다. on-policy 는 잘린 곳을 가치 추정으로 잇고(L258-262) 다음 수집이 `_last_obs` 에서 이어 간다. `learn()` 을 이어 부를 때도 환경을 reset 하지 않을 수 있다 [18] (L420-424).

### 가장 느린 환경 기다리기

- `SubprocVecEnv.step_wait` 는 모든 파이프의 응답을 받아야 반환한다 [4] (L137). **[계산]** 한 걸음의 환경 시간은 가장 느린 환경(자동 reset 포함)이 정하고, 그동안 끝난 프로세스와 신경망은 논다.
- `DummyVecEnv` 는 순차 실행이라 [3] (L58) **[계산]** 한 걸음의 환경 시간이 모든 환경 시간의 합이다.
- 읽은 `SubprocVecEnv`, `DummyVecEnv`, 두 `collect_rollouts` 에는 일부 환경만 먼저 진행하는 경로가 없다. `step_async` 는 `strict=True` 로 환경 수와 행동 수가 같기를 요구한다 [4] (L132).

### 참고: Gymnasium 의 벡터 환경

SB3 의 `VecEnv` 는 `gymnasium.vector.VectorEnv` 와 다른 자체 클래스다 [1] [19]. Gymnasium 도 `step_async`/`step_wait` 를 가진 `AsyncVectorEnv` 와 순차의 `SyncVectorEnv` 를 두지만 [21] (L482-528), **자동 reset 의 기본값이 다르다.**

| `AutoresetMode` | 동작 [20] (L292-334) |
| --- | --- |
| `NEXT_STEP` (Gymnasium 기본 [20] L84, [21] L138) | 끝난 **다음** `step` 에서 reset 한다. 그 걸음의 행동은 쓰이지 않고 보상은 0 이다 |
| `SAME_STEP` | 같은 걸음에서 reset 하고 마지막 관측을 `info["final_obs"]` 에 둔다. SB3 의 방식에 해당한다 |
| `DISABLED` | 사용자가 reset 한다 |

Gymnasium 은 모든 하위 환경이 끝나기를 기다리지 않으려고 자동 reset 을 둘 수 있다고 설명한다 [19] (L65-70).

## 4. 옛 `hill-a/stable-baselines` 와 같은 점

묶는 구조는 같다. `v2.10.1` 에도 `SubprocVecEnv` 의 파이프·`_worker`·`step_async`/`step_wait` 와 `terminal_observation` 을 남기는 자동 reset 이 있고 [22] (L11-23, L102-111), `DummyVecEnv` 도 순차 실행에 같은 처리를 한다 [23] (L41-51). PPO2 의 `Runner` 도 `model.step(self.obs, ...)` 로 관측 배치를 한 번에 평가한 뒤 `env.step` 한다 [24] (L471-482). 달라진 점은 백엔드(TensorFlow → PyTorch)와, 옛 4개 반환 `Env` 대신 Gymnasium 의 5개 반환을 SB3 형식으로 바꾸는 부분이다. SB3 는 상위 API 를 대체로 유지했다고 적는다 [25].

## 5. MCTS / AlphaZero 를 제공하는가

**제공하지 않는다.**

- SB3 `v2.9.2` 가 공개하는 알고리즘은 `A2C`, `DDPG`, `DQN`, `PPO`, `SAC`, `TD3`(과 `HerReplayBuffer`)다 [26]. contrib 는 `ARS`, `QRDQN`, `TQC`, `TRPO`, `CrossQ`, `MaskablePPO`, `RecurrentPPO` 다 [27]. 알고리즘 표에도 MCTS·AlphaZero 가 없다 [11].
- AlphaZero 요청 이슈 [#1464](https://github.com/DLR-RM/stable-baselines3/issues/1464) 는 `not_planned` 로 닫혔다. 유지 관리자 araffin 은 SB3 가 model-free 단일 에이전트에 집중하므로 범위 밖이라고 답했다. [#1755](https://github.com/DLR-RM/stable-baselines3/issues/1755) 의 "어느 알고리즘이 AlphaZero 인가" 에는 "없다" 고 답했다.
- 확인 범위: 두 저장소의 Python·문서 파일 목록에서 이름에 mcts·alpha·zero 가 든 것을 찾지 못했고, SB3 changelog 에도 그 낱말이 없다. 소스 전체의 본문 검색은 하지 못했다.

## 6. 우리 설계와의 비교

### 지금 코드

- `ai/mcts.py`: 신경망을 부르는 곳은 `MCTS._expand` 의 `evaluate(...)` 하나이고, `evaluate` 는 길이 1 의 `evaluate_batch` 다 (`ai/network.py`). `search` 는 루트를 한 번 펼친 뒤 `_simulate` 를 `simulations` 번 돈다. 끝난 국면에 닿은 시뮬레이션은 평가하지 않는다. 그래서 한 수의 forward 는 **최대 `1 + simulations` 번**이다(나노 군의는 `simulations=100`, `scripts/hyperparameters.py`).
- `ai/self_play.py`: `play_self_play_episode` 가 대국마다 `MCTS` 와 `random.Random(seed)` 를 따로 만든다. 가치 타깃은 대국이 끝난 뒤에야 채운다.
- `ai/training.py`: `train` 이 `episodes_per_generation` 판(나노 200, 미니 100)을 차례로 끝내고 버퍼에 넣은 뒤 학습한다. 대국마다 `start_position(rng)`, `rng.getrandbits(63)` 순서로 난수를 쓰고, 학습 배치도 같은 `rng` 로 뽑는다.

계획 중인 설계(이 저장소에는 아직 없다): 탐색을 Python 제너레이터로 바꿔 평가가 필요하면 yield 하고, 여러 대국의 요청을 모아 forward 한 번으로 계산한 뒤 답을 send 한다. 한 프로세스, 동시 판 수 `parallel_games`(기본 1).

### 대응 관계

| SB3 | 우리 설계 |
| --- | --- |
| 환경 `i` | 대국 제너레이터 자리 `i` |
| `obs` `(n_envs, ...)` | 이번 라운드에 yield 된 `encode` 텐서들 `(B, C, H, W)` |
| `policy(obs_tensor)` | `evaluate_batch(network, boards, device)` |
| `step(actions)` | 각 제너레이터에 `(logits, value)` 를 `send` |
| `done` + `terminal_observation` | `StopIteration.value` 의 `SelfPlayResult` |
| 자동 reset | 빈 자리에 다음 대국을 만들어 첫 yield 까지 진행 |
| `n_envs` 고정 | `B <= parallel_games`. 대국 할당량이 끝나 가면 준다 |

### 가져올 만한 것

모두 **[추정]** 인 설계 제안이다.

1. **구동기와 대국의 분리.** `VecEnv` 가 "환경들을 진행하는 쪽" 과 "신경망" 을 배열 경계로 나눈 것처럼, 제너레이터는 `(C, H, W)` 배열을 내고 `(logits, value)` 를 받기만 하게 한다. 배치·torch·디바이스는 구동기 한 곳(`evaluate_batch`)에 남는다. 합법수만의 softmax 는 지금처럼 받은 쪽(탐색)에서 하면 마스크를 묶을 필요가 없다. `search()` 는 같은 제너레이터를 길이 1 배치로 돌리는 얇은 함수로 남기면 `MCTSAI` 가 그대로 쓴다.
2. **`SAME_STEP` 식 채우기.** 대국이 끝나면 같은 라운드 안에서 다음 대국을 시작해 첫 요청을 받는다. 다음 forward 에 빈칸이 없다. Gymnasium 의 `NEXT_STEP` 처럼 한 라운드를 비울 이유가 없다. 시작하자마자 끝나는 대국은 yield 없이 반환하므로 요청이 나올 때까지 반복해서 채워야 한다.
3. **자리와 대국의 식별자 분리.** SB3 의 시드는 자리(`seed + idx`)에 붙고 처음 reset 에만 쓰인다. 우리는 지금처럼 **대국 번호 순서로** 시작 국면과 시드를 뽑고, 끝난 순서가 아니라 번호 순서로 결과를 모아야 한다. 그러지 않으면 버퍼의 샘플 순서가 `parallel_games` 에 따라 달라져 `rng.choices` 의 학습 배치가 바뀐다.
4. **`parallel_games=1` 을 `DummyVecEnv` 처럼 기준 구현으로.** 길이 1 배치는 지금의 `evaluate` 와 같은 호출이라 같은 시드의 결과가 지금과 같은지 테스트할 수 있다. 둘 이상에서는 배치 크기에 따라 부동소수점 결과가 미세하게 달라질 수 있어 비트 단위 일치를 기대하지 않는 편이 안전하다.
5. **한 프로세스가 먼저.** SB3 도 기본은 `DummyVecEnv` 다. 환경이 가벼우면 프로세스 간 비용이 더 크다는 것이 그 근거다 [3] [7].

### 맞지 않는 것

| SB3 의 가정 | 우리 상황 |
| --- | --- |
| 환경 한 걸음에 추론 한 번 | 수 하나에 평가 최대 101번. 라운드의 단위는 "수" 가 아니라 "평가 요청" 이다. 대국마다 같은 라운드에 다른 수의 다른 시뮬레이션에 있고, 끝난 잎은 yield 없이 지나간다 |
| 환경과 정책이 `step` 경계에서 번갈아 일한다 | 평가가 탐색의 한가운데에서 필요하다. `step(actions)` 인터페이스에는 중간에 멈출 자리가 없어 제너레이터(재개 가능한 탐색)가 필요하다 |
| 걸음 수로 수집을 끊고 가치 추정으로 잇는다 | 가치 타깃이 최종 결과라 끝나지 않은 대국은 학습 샘플이 되지 못한다 |
| `n_envs` 가 항상 같다 | 세대마다 대국 수가 정해져 있어 끝에서 묶음이 준다. 신경망은 `(B, ...)` 의 어떤 `B` 도 받으므로 빈칸을 덧대지 않아도 된다 |
| `SubprocVecEnv`: 프로세스마다 환경, 부모에 정책 | 같은 모양이면 시뮬레이션마다 왕복해 메시지가 수당 약 100배가 된다 **[추정]**. 프로세스를 나눈다면 대국 전체를 자식에 두고 평가 요청만 묶는 다른 구조여야 한다 **[추정]** |
| 난수가 배치 전체에 한 번(DQN) | 대국마다 독립된 난수가 필요하고 지금 코드가 이미 그렇다 |

`SubprocVecEnv` 의 "가장 느린 환경 기다리기" 는 우리의 한 프로세스 구동기에서는 모양이 다르다. **[계산]** 제너레이터를 차례로 진행하므로 한 라운드의 시간은 모든 대국의 Python 작업의 합 + forward 한 번이다.

### 꼬리의 작은 묶음

알려진 값(작업 요청에서 받은 것으로, 이 저장소의 `ai/`·`docs/`·`scripts/`·`eval/`·`.scratch/` 에서 근거 기록을 찾지 못했다): 옛 배치 시험에서 64판을 열어도 평균 묶음이 약 9였다.

- **[계산]** 대국 하나는 라운드마다 요청을 많아야 하나 낸다. 대국 `i` 의 총 평가 수를 `E_i` 라 하면 라운드 수 `R >= max(E_i)` 이고, 평균 묶음은 `sum(E_i) / R` 이다. 빈자리를 바로 채워도 마지막에 남은 긴 대국의 길이만큼은 작은 묶음으로 돈다.
- 저장소의 기록(`docs/hyperparameter-search.md`, "다시 잰 값", 부하가 섞인 측정)은 CUDA 에서 배치 1·8·32·128 의 처리량이 836·6,767·26,348·102,354 국면/초다. **[계산]** forward 한 번의 시간이 약 1.2ms 로 거의 같으므로, 이 범위에서 GPU 평가 시간은 국면 수보다 **라운드 수**에 좌우된다. CPU 는 398·808·871·907 이라 배치 8 이상의 이득이 거의 없다.
- 같은 기록의 CPU 프로파일(스레드 1)은 신경망 85%, 룰 9%, MCTS 트리 2% 다. **[추정]** GPU 에서 묶으면 Python 쪽(룰, 트리, `encode`)의 비중이 커져, `parallel_games` 를 늘려 얻는 이득이 그 지점에서 평평해진다.

SB3 식으로 꼬리를 없애는 길과 그 대가.

| 방법 | 대가 **[추정]** |
| --- | --- |
| 대국을 계속 채우고 200판이 끝나면 학습, 진행 중인 대국은 다음 세대로 넘긴다 | SB3 의 이어 두기에 가장 가깝다. 한 대국에 두 신경망의 정책이 섞이고, 동기 세대 진행이 바뀌며, 재현을 위해 진행 중인 상태를 관리해야 한다 |
| 200판이 끝나면 남은 대국을 버린다 | 짧은 대국이 먼저 끝나므로 긴 대국이 빠지는 편향이 생긴다. 권하지 않는다 |
| 수 제한 전에 잘라 가치 추정으로 타깃을 채운다 | SB3 의 bootstrap 에 해당한다. 학습 타깃의 정의가 바뀐다 |
| 지금의 동기 방식을 유지하고 꼬리를 받아들인다 | 가장 단순하다. 할당량이 `parallel_games` 보다 충분히 클 때만 이득이 크다 |

한 트리 안에서 요청을 여러 개 만드는 방법(virtual loss 등)은 SB3 에 대응하는 것이 없고 탐색 결과를 바꾼다. 이 문서에서는 조사하지 않았다.

**[추정]** 처음에는 마지막 방법으로 구현하고, 라운드마다의 묶음 크기를 기록해 꼬리가 실제로 차지하는 시간을 잰 뒤 정하는 것이 저장소의 원칙("속도 최적화는 검증 후 측정하고 진행")에 맞는다. 첫 번째 방법은 학습 동작을 바꾸므로 하이퍼파라미터처럼 비교해야 한다.

## 7. 확인하지 못한 것

- 아무 코드도 실행하지 않았다. 처리량, `DummyVecEnv` 와 `SubprocVecEnv` 의 속도 차이, 우리 설계의 묶음 크기는 재지 않았다.
- 파이프로 보내는 객체의 직렬화 방식은 Python `multiprocessing` 문서를 읽어 확인하지 않았다.
- `RecurrentPPO`, `ARS`(와 contrib 의 `async_eval`), `HerReplayBuffer`, `VecEnvWrapper` 들의 묶음 처리는 읽지 않았다.
- 두 저장소의 소스 본문 전체에서 MCTS 를 검색하지 못했다(5절의 확인 범위).
- Gymnasium 이 링크한 자동 reset 설명 글(`farama.org/Vector-Autoreset-Mode`)은 읽지 않았고 소스의 docstring 과 코드만 읽었다.
- "평균 묶음 약 9" 의 측정 조건과 원인(긴 꼬리인지)은 확인하지 못했다.

## 결론

1. SB3 의 묶음은 `VecEnv` 에서 나온다. 환경 `n_envs` 개의 관측을 첫 축으로 쌓아 forward 한 번에 넣고, 끝난 환경은 같은 `step` 안에서 reset 해 크기를 지킨다.
2. 동기 방식이라 `SubprocVecEnv` 는 가장 느린 환경을 기다리고, 기본값은 순차 실행의 `DummyVecEnv` 다. MCTS·AlphaZero 는 범위 밖이라 제공하지 않는다.
3. 우리가 가져올 것은 배열 경계의 구동기, 같은 라운드의 빈자리 채우기, 크기 1 기준 구현이다. 시드와 결과 순서는 자리가 아니라 대국 번호에 묶는다.
4. 맞지 않는 것은 "걸음당 추론 한 번" 과 "아무 데서나 끊고 bootstrap" 이다. 우리는 수당 평가가 최대 101번이고 타깃이 최종 결과다.
5. 따라서 자동 reset 식 채우기만으로는 세대 끝의 작은 묶음이 없어지지 않는다. **[추정]** 먼저 동기 방식으로 만들고 꼬리를 측정한 뒤 이어 두기를 비교한다.

## 출처

SB3 의 경로는 `https://github.com/DLR-RM/stable-baselines3/blob/v2.9.2/` 아래다.

1. [`stable_baselines3/common/vec_env/base_vec_env.py`](https://github.com/DLR-RM/stable-baselines3/blob/v2.9.2/stable_baselines3/common/vec_env/base_vec_env.py) — `VecEnv`
2. [`docs/guide/vec_envs.md`](https://github.com/DLR-RM/stable-baselines3/blob/v2.9.2/docs/guide/vec_envs.md)
3. [`stable_baselines3/common/vec_env/dummy_vec_env.py`](https://github.com/DLR-RM/stable-baselines3/blob/v2.9.2/stable_baselines3/common/vec_env/dummy_vec_env.py) — `DummyVecEnv`
4. [`stable_baselines3/common/vec_env/subproc_vec_env.py`](https://github.com/DLR-RM/stable-baselines3/blob/v2.9.2/stable_baselines3/common/vec_env/subproc_vec_env.py) — `_worker`, `SubprocVecEnv`
5. [`stable_baselines3/common/on_policy_algorithm.py`](https://github.com/DLR-RM/stable-baselines3/blob/v2.9.2/stable_baselines3/common/on_policy_algorithm.py) — `OnPolicyAlgorithm.collect_rollouts`
6. [`stable_baselines3/common/off_policy_algorithm.py`](https://github.com/DLR-RM/stable-baselines3/blob/v2.9.2/stable_baselines3/common/off_policy_algorithm.py) — `collect_rollouts`, `_sample_action`, `_store_transition`
7. [`stable_baselines3/common/env_util.py`](https://github.com/DLR-RM/stable-baselines3/blob/v2.9.2/stable_baselines3/common/env_util.py) — `make_vec_env`
8. [`docs/modules/ppo.md`](https://github.com/DLR-RM/stable-baselines3/blob/v2.9.2/docs/modules/ppo.md)
9. [`stable_baselines3/common/policies.py`](https://github.com/DLR-RM/stable-baselines3/blob/v2.9.2/stable_baselines3/common/policies.py) — `obs_to_tensor`, `BasePolicy.predict`, `ActorCriticPolicy.forward`
10. [`stable_baselines3/common/buffers.py`](https://github.com/DLR-RM/stable-baselines3/blob/v2.9.2/stable_baselines3/common/buffers.py) — `BaseBuffer`, `ReplayBuffer`, `RolloutBuffer`
11. [`docs/guide/algos.md`](https://github.com/DLR-RM/stable-baselines3/blob/v2.9.2/docs/guide/algos.md)
12. [contrib `sb3_contrib/ppo_mask/ppo_mask.py`](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/blob/v2.9.2/sb3_contrib/ppo_mask/ppo_mask.py) — `MaskablePPO.collect_rollouts`
13. [contrib `sb3_contrib/common/maskable/utils.py`](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/blob/v2.9.2/sb3_contrib/common/maskable/utils.py)
14. [`stable_baselines3/ppo/ppo.py`](https://github.com/DLR-RM/stable-baselines3/blob/v2.9.2/stable_baselines3/ppo/ppo.py) — `PPO.train`
15. [`stable_baselines3/dqn/dqn.py`](https://github.com/DLR-RM/stable-baselines3/blob/v2.9.2/stable_baselines3/dqn/dqn.py) — `DQN.predict`
16. [`stable_baselines3/dqn/policies.py`](https://github.com/DLR-RM/stable-baselines3/blob/v2.9.2/stable_baselines3/dqn/policies.py)
17. [`docs/guide/examples.md`](https://github.com/DLR-RM/stable-baselines3/blob/v2.9.2/docs/guide/examples.md) — "Multiprocessing with off-policy algorithms"
18. [`stable_baselines3/common/base_class.py`](https://github.com/DLR-RM/stable-baselines3/blob/v2.9.2/stable_baselines3/common/base_class.py) — `_setup_learn`
19. [Gymnasium `gymnasium/vector/vector_env.py`](https://github.com/Farama-Foundation/Gymnasium/blob/v1.4.0/gymnasium/vector/vector_env.py) — `AutoresetMode`, `VectorEnv`
20. [Gymnasium `gymnasium/vector/sync_vector_env.py`](https://github.com/Farama-Foundation/Gymnasium/blob/v1.4.0/gymnasium/vector/sync_vector_env.py) — `SyncVectorEnv.step`
21. [Gymnasium `gymnasium/vector/async_vector_env.py`](https://github.com/Farama-Foundation/Gymnasium/blob/v1.4.0/gymnasium/vector/async_vector_env.py) — `AsyncVectorEnv`
22. [SB2 `stable_baselines/common/vec_env/subproc_vec_env.py`](https://github.com/hill-a/stable-baselines/blob/v2.10.1/stable_baselines/common/vec_env/subproc_vec_env.py)
23. [SB2 `stable_baselines/common/vec_env/dummy_vec_env.py`](https://github.com/hill-a/stable-baselines/blob/v2.10.1/stable_baselines/common/vec_env/dummy_vec_env.py)
24. [SB2 `stable_baselines/ppo2/ppo2.py`](https://github.com/hill-a/stable-baselines/blob/v2.10.1/stable_baselines/ppo2/ppo2.py) — `Runner._run`
25. [`docs/guide/migration.md`](https://github.com/DLR-RM/stable-baselines3/blob/v2.9.2/docs/guide/migration.md)
26. [`stable_baselines3/__init__.py`](https://github.com/DLR-RM/stable-baselines3/blob/v2.9.2/stable_baselines3/__init__.py)
27. [contrib `sb3_contrib/__init__.py`](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/blob/v2.9.2/sb3_contrib/__init__.py)
