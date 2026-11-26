# 04: torch 의존성과 `Network` 프로토콜 + 작은 ResNet

**What to build:** 신경망이 `Network` 프로토콜 뒤에 있고, 작은 ResNet 하나가 그것을 구현한다. CI 는 CPU torch, 로컬은 CUDA torch 를 받는다.

**Blocked by:** -

**Status:** ready-for-agent

- [ ] torch 를 의존성에 넣는다. uv 플랫폼 마커로 리눅스는 CPU 인덱스, Windows 는 CUDA 인덱스. `uv sync --locked` 가 CI 에서 통과한다
- [ ] `ai/` 의 `Network` 프로토콜: `(B, C, H, W)` → `(정책 로짓 (B, A), 가치 (B,) ∈ [−1, 1])`. 생성자는 `tensor_shape`·`action_space_size`·구조 설정만 받는다. 마스크는 밖에서 씌운다
- [ ] 작은 ResNet(블록 수·채널 수 설정). 틱택토 `(2, 3, 3)`·9 와 나노 `(18, 4, 3)`·288 의 모양으로 돈다
- [ ] numpy `(C, H, W)` → torch 배치 변환과 디바이스 처리가 `ai/` 한 곳에 있다
- [ ] 같은 시드는 CPU 에서 같은 출력을 낸다
- [ ] 신경망 단독 테스트는 모양과 시드 재현성만 본다. 과적합 같은 내부 테스트는 두지 않는다 (spec "Seam")

설계: spec "`Network` 프로토콜", "CI 와 torch". 행동 구조 노출은 하지 않는다 (후속 메모 8번).
