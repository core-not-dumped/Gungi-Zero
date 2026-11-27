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
    # 평균 합법수 9.218. 측정 보고 도구(`scripts.measurement_report`)의 평균 분기 계수.
    root_noise=RootNoise(alpha=1.08, epsilon=0.25),
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

NANO_GUNGI_NETWORK = ResNetConfig(blocks=4, channels=64)
NANO_GUNGI_TRAINING = TrainingConfig(
    self_play=NANO_GUNGI_SELF_PLAY,
    episodes_per_generation=100,
    buffer_generations=10,
    steps_per_generation=200,
    batch_size=128,
)
