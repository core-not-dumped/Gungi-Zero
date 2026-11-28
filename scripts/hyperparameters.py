"""게임별 하이퍼파라미터. 처음 값이고, 바꾸면 기록한다.

Dirichlet alpha 는 10 / 평균 합법수다. 평균 합법수는 랜덤 AI 둘의 1만 판(시드 0~9999)에서 각 수의
합법수 개수를 더해 둔 수의 개수로 나눈 값이고, 2026-11-27 에 쟀다.

## 바꾼 기록

2026-11-27, 틱택토 (티켓 07). 처음 값은 temperature 1 인 첫 2수, 블록 2·채널 32, 세대마다 50판,
버퍼 5세대였고, 20세대 학습 뒤 후수 100판 중 14~22판을 졌다(시드 0~2). 지금 값(첫 9수 = 모든 수,
채널 64, 100판, 버퍼 10세대, 50세대)은 학습 시드 0~9 가 45·50세대에서 모두 선수·후수 무패였다
(Windows CPU, torch 스레드 1). 시드와 설정 비교는 저장소에 남기지 않은 일회성 스크립트로 쟀다.

- 첫 9수: 처음 설정의 후수 패배 국면에서 신경망이 지는 수를 비기는 수로 보고 있었다. 모든 수를
  temperature 1 로 두자 가치 일치율이 0.5 안팎에서 0.76 으로 올랐다. 드문 국면을 겪지 못한 것이
  원인이라는 것은 추정이다.
- 채널 64, 100판, 버퍼 10세대: 첫 9수만 바꾸고 50판을 두면 시드 0~2 에서 후수 패배가 남았다.
  판 수를 늘리면 시드 0~2 중 둘이, 채널까지 늘리면 셋 다 45세대부터 무패였다. 항목 하나씩의
  효과를 모두 가른 것은 아니다.

2026-11-27, 나노 군의 (티켓 09). 세대마다의 자가 대국을 100판에서 200판으로 늘렸다. 100판의 1차
학습은 시험 판 149/200 이었고, 100세대의 학습 샘플이 약 26,500개뿐이라 데이터 부족으로
추정했다 (`docs/nano-verification-report.md`). 200판의 결과는 아직 없다. 체크포인트는 5세대마다
남긴다 (1차는 세대마다). 학습의 결과에는 영향이 없다.

2026-11-28, 나노 군의의 시작 배치를 바꿨다 (ADR-0007). 새 배치의 평균 합법수에 맞춰 Dirichlet
alpha 를 1.08 에서 1.09 로 고쳤다. 위의 기록과 그때까지의 성적은 이전 배치의 것이다.

2026-11-28, 군의의 자가 대국을 64판씩 함께 두며 신경망 평가를 묶는다 (`parallel_games`). 탐색과
학습의 방식은 같다. 틱택토는 한 판씩이다.

2026-11-28, 군의의 자가 대국에서 평가한 판의 답을 세대 안에서 기억하고(`evaluation_cache`, 나노
10만·미니 5만 판), CUDA 에서는 평가를 기록해 재생한다(`cuda_graph`). 방식은 같고,
부동소수점 끝자리의
차이로 드물게 수가 달라질 수 있다.
"""

from ai.mcts import RootNoise
from ai.resnet import ResNetConfig
from ai.self_play import SelfPlayConfig
from ai.training import TrainingConfig

TICTACTOE_SELF_PLAY = SelfPlayConfig(
    simulations=50,
    # 평균 합법수 5.569 = 425,793 / 76,451 수 (일회성 스크립트).
    root_noise=RootNoise(alpha=1.80, epsilon=0.25),
    # 틱택토의 가장 긴 대국이라 모든 수를 temperature 1 로 고른다.
    exploration_moves=9,
)

NANO_GUNGI_SELF_PLAY = SelfPlayConfig(
    simulations=100,
    # 평균 합법수 9.206. 측정 보고 도구(`scripts.measurement_report`)의 평균 분기 계수.
    root_noise=RootNoise(alpha=1.09, epsilon=0.25),
    exploration_moves=4,
)

# 구조와 학습 양은 spec 에 없는 값이다. 틱택토는 07 에서 골랐고, 나노는 처음 값이라 나노 검증(09)
# 에서 다시 본다.
TICTACTOE_NETWORK = ResNetConfig(blocks=2, channels=64)
TICTACTOE_TRAINING = TrainingConfig(
    self_play=TICTACTOE_SELF_PLAY,
    episodes_per_generation=100,
    buffer_generations=10,
    steps_per_generation=100,
    batch_size=64,
)
# 틱택토 검증(`scripts.verify_tictactoe`)이 학습하는 세대 수.
TICTACTOE_GENERATIONS = 50
# 평가 대국의 탐색 횟수와 temperature.
TICTACTOE_EVALUATION_SIMULATIONS = 50
TICTACTOE_EVALUATION_TEMPERATURE = 0.0

# 자가 대국 가운데 시작 국면에서 시작하는 판의 비율. 나머지는 도달 가능하고 끝나지 않은 상태에서
# 균등하게 뽑는다 (ADR-0007).
NANO_GUNGI_INITIAL_START_FRACTION = 0.25
# 나노 검증(`scripts.verify_nano`)이 학습하는 세대 수.
NANO_GUNGI_GENERATIONS = 100
# 평가 대국의 탐색 횟수와 temperature (ADR-0007).
NANO_GUNGI_EVALUATION_SIMULATIONS = 100
NANO_GUNGI_EVALUATION_TEMPERATURE = 0.0
# 최선수 비율 진단이 보는 국면 수와 뽑는 시드.
NANO_GUNGI_BEST_MOVE_SAMPLE = 5000
NANO_GUNGI_BEST_MOVE_SEED = 0

# 아래의 실력 측정(`scripts.grade`·`rate`·`make_levels`)은 모든 게임이 함께 쓴다.
# 평가 AI 의 temperature. 탐색 횟수는 게임마다의 평가 값을 쓴다.
EVALUATION_TEMPERATURE = 0.0
# 풀이가 없는 게임의 시작 판: 초기 국면에서 무작위로 두는 수의 범위, 이 깊이 안에 승패가
# 강제되는 판은 버린다, 시드.
OPENING_PLIES = (2, 8)
OPENING_DEPTH = 3
OPENING_SEED = 0

# 세대마다의 Elo(`scripts.rate`). 대진마다의 시작 판 수(선후를 바꿔 두 판씩).
ELO_OPENINGS = 50
# 서로 두는 체크포인트의 저장 순번 차이.
ELO_GAPS = (1, 5, 20)
# 랜덤 AI 와는 1·2·5번째, 이 간격의 순번, 마지막 체크포인트가 둔다.
ELO_ANCHORED_EVERY = 10
# 대진마다 더하는 가상의 무승부 수.
ELO_PRIOR_DRAWS = 1.0

# 대국 화면의 난이도 단계(`scripts.make_levels`). 단계 사이의 점수 간격.
LEVEL_STEP = 200.0
# 가장 강한 세대에 섞어 보는 무작위 수의 확률.
LEVEL_RANDOM_PROBABILITIES = tuple(percent / 100 for percent in range(5, 100, 5))
# 섞은 후보의 점수를 재는 상대의 저장 순번. 랜덤 AI 와 가장 강한 세대도 함께 둔다.
LEVEL_OPPONENTS = (1, 5)

NANO_GUNGI_NETWORK = ResNetConfig(blocks=4, channels=64)
NANO_GUNGI_TRAINING = TrainingConfig(
    self_play=NANO_GUNGI_SELF_PLAY,
    episodes_per_generation=200,
    buffer_generations=10,
    steps_per_generation=200,
    batch_size=128,
    checkpoint_every=5,
    parallel_games=64,
    evaluation_cache=100_000,
    cuda_graph=True,
)

# 미니 군의. **처음 값이고 학습해 본 적이 없다** (3단계에서 다시 본다). 구조와 학습 양은 나노 군의의
# 것을 그대로 두었다. 초기 국면에서만 시작한다 — 상태를 모두 돌 수 없다.
MINI_GUNGI_SELF_PLAY = SelfPlayConfig(
    simulations=100,
    # 평균 합법수 14.47. 측정 보고(`docs/measurement-report.md`)의 평균 분기 계수.
    root_noise=RootNoise(alpha=0.69, epsilon=0.25),
    # 랜덤 대국 길이의 중앙값 30수의 3분의 1.
    exploration_moves=10,
)
MINI_GUNGI_NETWORK = ResNetConfig(blocks=4, channels=64)
MINI_GUNGI_TRAINING = TrainingConfig(
    self_play=MINI_GUNGI_SELF_PLAY,
    episodes_per_generation=100,
    buffer_generations=10,
    steps_per_generation=200,
    batch_size=128,
    checkpoint_every=5,
    parallel_games=64,
    evaluation_cache=50_000,
    cuda_graph=True,
)
MINI_GUNGI_EVALUATION_SIMULATIONS = 100
