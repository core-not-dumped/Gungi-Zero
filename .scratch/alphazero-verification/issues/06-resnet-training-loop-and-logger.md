# 06: ResNet, 학습 루프, 로거

**What to build:** 작은 ResNet 으로 틱택토의 세대 하나(자가 대국 → 리플레이 버퍼 → 학습)를 끝까지 돌리는 스크립트가 메모리 로거로 돌고, 체크포인트를 남긴다.

**Blocked by:** 05 (자가 대국과 학습 샘플)

**Status:** ready-for-agent

- [x] 작은 ResNet(정책 머리 + tanh 가치 머리, 블록 수·채널 수 설정)이 `Network` 프로토콜을 따른다. 틱택토 `(2, 3, 3)`·9 와 나노 `(18, 4, 3)`·288 의 모양으로 돈다. 신경망 단독 테스트는 모양과 시드 재현성만 본다
- [x] numpy → torch 변환, 배치, 디바이스 처리가 학습하는 쪽 한 곳에 있다
- [x] 동기 세대 루프(N판 → 최근 K세대 리플레이 버퍼 → M스텝), gating 없음. 손실은 정책 교차 엔트로피 + 가치 평균제곱오차, Adam lr 1e-3, weight decay 1e-4
- [x] 로거 프로토콜(`config`, `log`)과 메모리 로거. 손실과 자가 대국 통계(길이, 결과 분포, 종료 사유)를 기록한다
- [x] 합성 루트가 변형 설정(수 제한, 반복 한도, 판 크기, 말 구성)을 config 로 남긴다 (ADR-0004)
- [x] 세대마다 체크포인트를 로컬에 저장한다(git 제외)
- [x] 같은 시드는 CPU 에서 같은 세대 가중치를 만든다

spec: "신경망 프로토콜", "자가 대국과 학습 루프", "로거".

## Comments

- 결과(2026-11-27): ResNet 은 `ai/resnet.py`, 세대 루프는 `ai/training.py` (`train`, `TrainingConfig`, `seed_everything`), 로거는 `ai/logger.py`, 합성 루트는 `scripts/train.py` 다. 실행은 `uv run python -m scripts.train --game tictactoe --generations 1`.
- numpy → torch 변환과 배치는 `ai/network.py` 의 `to_tensor` 다. 학습 손실은 MCTS 처럼 합법수 마스크를 씌운 softmax 로 잰다.
- 종료 사유는 `Game` 이 답하지 않아 `train` 이 `end_reason` 함수를 주입받는다. 틱택토의 사유 이름은 `LINE`, `BOARD_FULL` 이다.
- 구조와 학습 양(`scripts/hyperparameters.py` 의 `*_NETWORK`, `*_TRAINING`)은 spec 에 없는 처음 값이고 조정하지 않았다.
- 재현은 CPU 에서 틱택토 2세대로 확인했다. CUDA 는 나노 1세대가 끝까지 도는 것만 한 번 확인했다(자동 테스트 없음).
- 07 에서 쓸 것: 로컬 CPU 에서 틱택토 1세대(50판, 100스텝)가 프로세스 시작을 포함해 약 10초다. `train` 은 넘긴 신경망을 바꾸고 평가 모드로 남긴다. 체크포인트는 `state_dict` 만이라 읽는 쪽이 같은 구조의 `ResNet` 을 만들어야 한다. 체크포인트가 이미 있는 폴더에서는 시작하지 않고, 이어서 학습하는 기능은 없다.
- 09 에서 할 것: 메모리 로거는 학습이 끝나야 설정과 지표를 찍는다. 중간에 죽으면 가중치만 남으므로, wandb 로거는 시작할 때 설정을 보내야 한다.

