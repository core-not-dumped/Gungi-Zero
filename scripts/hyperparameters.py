"""게임별 하이퍼파라미터. 처음 값이고, 바꾸면 기록한다.

Dirichlet alpha 는 10 / 평균 합법수다. 평균 합법수는 랜덤 AI 둘의 1만 판(시드 0~9999)에서 각 수의
합법수 개수를 더해 둔 수의 개수로 나눈 값이고, 2026-11-27 에 쟀다.
"""

from ai.mcts import RootNoise
from ai.resnet import ResNetConfig
from ai.self_play import SelfPlayConfig
from ai.training import TrainingConfig

TICTACTOE_SELF_PLAY = SelfPlayConfig(
    simulations=50,
    # 평균 합법수 5.569 = 425,793 / 76,451 수 (일회성 스크립트).
    root_noise=RootNoise(alpha=1.80, epsilon=0.25),
    exploration_moves=2,
)

NANO_GUNGI_SELF_PLAY = SelfPlayConfig(
    simulations=100,
    # 평균 합법수 9.218. 측정 보고 도구(`scripts.measurement_report`)의 평균 분기 계수.
    root_noise=RootNoise(alpha=1.08, epsilon=0.25),
    exploration_moves=4,
)

# 아래 구조와 학습 양은 spec 에 없는 처음 값이다. 틱택토 검증(07)과 나노 검증(09)에서 다시 본다.
TICTACTOE_NETWORK = ResNetConfig(blocks=2, channels=32)
TICTACTOE_TRAINING = TrainingConfig(
    self_play=TICTACTOE_SELF_PLAY,
    episodes_per_generation=50,
    buffer_generations=5,
    steps_per_generation=100,
    batch_size=64,
)

NANO_GUNGI_NETWORK = ResNetConfig(blocks=4, channels=64)
NANO_GUNGI_TRAINING = TrainingConfig(
    self_play=NANO_GUNGI_SELF_PLAY,
    episodes_per_generation=100,
    buffer_generations=10,
    steps_per_generation=200,
    batch_size=128,
)
