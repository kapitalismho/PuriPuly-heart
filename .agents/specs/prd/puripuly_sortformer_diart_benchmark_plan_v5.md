# PuriPuly Peer Channel 화자 전환 Teacher / Reference Characterization Benchmark Plan v5

## TC — Teacher / Reference Characterization — Streaming Sortformer v2.1 F32/Vulkan vs diart + pyannote

**Status:** revised experiment plan  
**Supersedes:** `puripuly_sortformer_diart_benchmark_plan_v4.md`  
**Primary purpose:** PSEM의 reference 역할과 optional KD teacher 역할을 분리해 characterization하고, teacher-dependent 학습 결정을 위한 재현 가능한 artifact를 만든다.  
**PSEM dependency:** PSEM Dataset/GT-only 단계와 병렬 진행 가능하며, teacher-dependent KD 단계 전에만 완료되면 된다.  
**Public-data corpus selection:** intentionally deferred to a separate design session

---

# 1. 프로젝트 목표

PuriPuly peer channel의 입력은 화자별 분리 채널이 아니라 **하나의 16 kHz mono mixed-audio timeline**이다.

최종 제품이 필요한 것은 full diarization UI나 영구 speaker identity가 아니다. 필요한 것은 다음에 더 가깝다.

```text
현재 logical turn 안에서

- 새로운 화자가 들어왔는가?
- 기존 화자에서 다른 화자로 handoff 되었는가?
- overlap / interruption이 시작되었는가?
- speaker-change 근거가 충분히 강해 logical turn boundary 후보로 쓸 수 있는가?
```

최종 student는 다음 Pareto frontier를 목표로 한다.

```text
낮은 harmful false change
낮은 detection latency
낮은 compute / memory
충분한 speaker-change recall
충분한 overlap awareness
```

이번 benchmark는 student를 학습하거나 최종 배포 모델을 선택하는 단계가 아니다.

이번 단계의 질문은 다음이다.

> **어떤 teacher / reference가 PuriPuly가 실제로 필요한 speaker-event 정보를 가장 유용하게 제공하는가?**

이 문서는 PuriPuly 구조, Peer Channel, VAD의 목적, A/B 평가의 의미에 대한 배경이 필요할 때 참고할 수 있도록 다음 맥락을 보존한다.

---

## 1.1 PuriPuly에서 실제로 들어오는 것은 "화자별 채널"이 아니라 하나의 mono mix다

PuriPuly Heart는 VRChat을 중심으로 사용하는 LLM 기반 양방향 음성 번역 애플리케이션이다.

Peer Channel은 사용자의 마이크(`self`)가 아니라 **상대방과 주변 사람들의 음성(`peer`)** 을 캡처하여 STT와 번역에 사용하는 경로다.

여기서 중요한 전제는, PuriPuly가 상대 화자마다 분리된 오디오 스트림을 받는 것이 아니라는 점이다. 실제 문제는 대략 다음과 같다.

```text
하나의 mono mixed audio channel

[A가 말함] [A가 계속 말함] [B가 끼어듦] [B가 말함] [B+C overlap] [C가 말함]
──────────────────────────────────────────────────────────────────────────→ time
```

즉 입력은:

```text
speaker A channel
speaker B channel
speaker C channel
```

처럼 이미 분리되어 있지 않다.

우리가 받는 것은 기본적으로:

```text
A + B + C + room/noise가 섞인 하나의 waveform
```

이다.

따라서 최종적으로 풀고 싶은 문제는 매우 구체적이다.

> **"하나의 mono mixed audio stream에서, 현재 말하던 사람과 다음 사람이 달라졌는가?"**

이를 온라인으로, 가능한 한 빨리, 그리고 오탐을 적게 내면서 판단하고 싶다.

---

## 1.2 왜 이 문제가 PuriPuly에서 중요한가: 여러 화자가 섞이면 자막이 난잡해진다

Peer Channel에는 여러 사람이 연속해서 말할 수 있다.

예를 들어 실제 대화가:

```text
A: 오늘 어디 갈 거야?
B: 나는 카페 갈 것 같아.
C: 나도 같이 가도 돼?
```

인데 시스템이 speaker boundary를 전혀 모르면, STT / 번역 / 자막 파이프라인 입장에서는 이것이 하나의 긴 발화처럼 섞일 수 있다.

```text
"오늘 어디 갈 거야 나는 카페 갈 것 같아 나도 같이 가도 돼"
```

그 결과 다음 문제가 생길 수 있다.

- 서로 다른 사람의 문장이 하나의 자막 turn으로 붙음
- 번역 문맥이 서로 다른 화자의 발화를 같은 사람의 연속 발화처럼 취급함
- 짧은 맞장구나 끼어들기가 이전 화자의 문장에 붙음
- overlap 이후 실제로 누가 말을 이어갔는지 불명확해짐
- 여러 사람이 있는 VRChat 환경에서 자막이 빠르게 누적되어 읽기 어려워짐

따라서 speaker-change detection의 목적은 단순한 diarization benchmark 점수를 높이는 것이 아니다.

**PuriPuly의 Peer 자막과 대화 문맥을 더 깨끗한 turn 단위로 유지하기 위한 제품 기능**이다.

---

## 1.3 최종 모델이 풀 문제

최종 모델의 primary objective는:

```text
mono mixed audio
      ↓
small streaming speaker-event model
      ↓
P(speaker_change)
```

이다.

Overlap 정보는 speaker-change 판단과 상태 오염 방지에 유용하므로 secondary output으로 유지한다.

```text
mono mixed audio
      ↓
shared small encoder
      ├── P(speaker_change)   ← primary
      └── P(overlap)          ← secondary
```

제품에서 우선 필요한 event는:

```text
speaker_change

overlap_start
overlap_end
```

이다.

반대로 v0에서 반드시 필요하지 않은 것은:

```text
stable global speaker ID
정확한 전체 speaker count
"이 사람은 10분 전에 나온 A와 동일 인물"이라는 장기 re-identification
3명 이상 overlap의 정확한 participant attribution
full diarization output
```

이다.

즉 목표는 "작은 diarization system"을 그대로 만드는 것이 아니라 **PuriPuly에 필요한 local speaker transition problem만 떼어서 푸는 것**이다.

---

## 1.4 엔지니어링 목표: 정확도 하나가 아니라 Pareto frontier

최종 배포 모델에서는 네 가지가 동시에 중요하다.

### 1. 낮은 화자 전환 오탐

PuriPuly에서는 false speaker change가 많으면 실제로 같은 사람이 계속 말하는데도 자막 / context turn을 불필요하게 쪼갤 수 있다.

예:

```text
실제:
A ───────────────── A

오탐 모델:
A ── change ── change ── change ── A
```

따라서 단순 recall뿐 아니라:

```text
false changes / minute
```

가 핵심 제품 지표다.

### 2. 낮은 detection latency

화자가 바뀐 뒤 너무 늦게 알면 이미 STT와 자막이 다음 화자의 내용을 이전 turn에 붙인 뒤일 수 있다.

따라서 accuracy가 높더라도 수 초 뒤에야 change를 알려주는 모델은 제품 목적에 적합하지 않다.

### 3. 낮은 compute cost

speaker model은 단독으로 실행되지 않는다.

PuriPuly 로컬 환경에서는 동시에 다음이 실행될 수 있다.

```text
VAD
ASR
speaker-change model
UI / overlay
번역 관련 처리
기타 audio processing
```

따라서 diarization 정확도를 조금 높이기 위해 매우 큰 GPU 모델을 상시 실행하는 것은 목표와 맞지 않는다.

최종 학생 모델은 가능한 한 작은 parameter / memory / RTF로 동작해야 한다.

### 4. 충분히 높은 change recall / overlap 품질

물론 작고 빠르기만 해서는 의미가 없다.

중요한 실제 speaker transition을 놓치지 않아야 하고, overlap 때문에 speaker state가 쉽게 오염되지 않아야 한다.

따라서 우리가 찾는 것은 단순한 최고 F1 지점이 아니라:

```text
높은 change accuracy
+ 낮은 false-change rate
+ 낮은 observed detection latency
+ 낮은 compute cost
```

의 **Pareto-optimal 또는 Pareto frontier에 가까운 지점**이다.

예를 들어 F1이 0.2% 높지만 compute가 10배이고 latency도 훨씬 크다면 최종 제품에서는 더 나쁜 선택일 수 있다.

---

## 1.5 왜 기존 diarization 모델을 그대로 쓰지 않고 작은 전용 모델을 만들려는가

기존 공개 모델과 접근들을 조사한 결과, PuriPuly 목표와 완전히 일치하는 선택지가 쉽게 나오지 않았다.

대표적인 문제는 다음과 같다.

- full speaker diarization을 목표로 하여 우리가 필요하지 않은 global speaker tracking까지 수행함
- 모델 또는 전체 pipeline이 로컬 상시 실행용으로 너무 무거움
- 충분한 정확도를 얻으려면 look-ahead / latency가 너무 커짐
- diarization DER 최적화와 PuriPuly의 `speaker_change + low false positive` 목표가 다름
- streaming은 가능하지만 전체 compute / memory 부담이 작지 않음
- 반대로 매우 작은 방법은 원하는 change / overlap 성능이 부족할 수 있음

PuriPuly가 실제로 원하는 것은 더 좁다.

```text
"누구인지 영구적으로 기억"
        ↓ 필요 없음

"지금 화자가 바뀌었는지 빠르게 판단"
        ↓ 필요함
```

따라서 full diarization을 그대로 축소하기보다는, **speaker-change와 overlap에 초점을 둔 작은 streaming model을 별도로 학습**하는 방향을 선택한다.

---

## 1.6 최종 개발 전략: 공개 데이터 + 직접 supervision + knowledge distillation

최종 학생 모델은 가능한 한 공개적으로 사용할 수 있는 데이터와 재현 가능한 학습 파이프라인을 기반으로 만든다.

큰 흐름은 다음과 같다.

```text
공개 Ground Truth speaker data
          │
          ├── 직접 speaker-change supervision
          ├── overlap supervision
          └── speaker representation supervision

강한 teacher model / pipeline
          │
          └── soft change / overlap targets

                    ↓
             small student model
                    ↓
       low-latency / low-FP / low-compute
```

중요한 원칙은 teacher에 전적으로 의존하지 않는 것이다.

먼저 GT-only student baseline을 만들 수 있어야 하고, 그 다음 teacher distillation이 실제로 도움이 되는지 비교한다.

---

## 1.7 이번 단계에서 teacher를 고를 때 보는 것

Teacher는 최종 학생처럼 작을 필요는 없다. 학습 시 offline으로 사용할 수 있기 때문이다.

하지만 아무리 강한 teacher라도 PuriPuly가 필요로 하는 문제에서 좋은 supervision을 주지 못하면 의미가 없다.

따라서 teacher / reference 후보를 볼 때 가장 중요하게 보는 것은:

```text
1. PuriPuly B500 정책에서 speaker-change 성능이 높은가?
2. false changes / minute가 낮은가?
3. overlap을 충분히 안정적으로 잡는가?
4. low-latency streaming 조건에서도 성능이 유지되는가?
5. soft target으로 변환하기 쉬운 출력을 제공하는가?
```

Compute cost도 기록하지만, teacher 자체의 compute는 최종 student deployment cost와 동일한 제약은 아니다.

반면 **teacher가 미래 오디오를 지나치게 많이 보고 만든 정답을 훨씬 짧은 latency의 student에게 그대로 요구하는 것**은 좋지 않으므로 streaming / latency 특성은 중요하다.

---

## 1.8 왜 지금 Sortformer와 diart + pyannote를 비교하는가

현재 primary teacher 후보는:

```text
NVIDIA Streaming Sortformer v2.1
```

이다.

Sortformer는 streaming diarization을 직접 수행하고, low-latency configuration이 있으며, speaker activity를 frame 단위로 얻을 수 있어 `speaker_change / overlap` soft target으로 변환하기 좋다.

동시에 비교 대상으로:

```text
diart + pyannote/segmentation-3.0
```

을 평가한다.

diart는 Sortformer와 내부 구조가 매우 다르다.

```text
Sortformer:
streaming neural diarization + speaker cache

diart:
rolling segmentation + speaker embedding + online clustering
```

따라서 둘을 비교하면 단순히 "어느 diarizer가 DER가 낮은가"가 아니라 다음을 알 수 있다.

```text
PuriPuly가 실제 요구하는 local speaker-change 문제에서
어느 방식이 더 좋은 upper bound를 제공하는가?
```

또한 diart를 A와 B 두 방식으로 보면:

```text
A = VAD 도움 없이 diarization backend 자체 성능
B = PuriPuly VAD 정책이 문제를 줄여준 뒤의 제품 성능
```

을 분리할 수 있다.

이 차이는 최종 학생 모델이 정말 장기 speaker tracking까지 배워야 하는지, 아니면 훨씬 작은 local event model이면 되는지를 판단하는 근거가 된다.

---

## 1.9 이번 benchmark의 역할

이번 benchmark의 목적은 학생 모델을 최종 평가하는 것이 아니다.

이번 단계에서 결정할 것은:

```text
Teacher / upper-bound reference selection
```

이다.

구체적으로 비교한다.

```text
Sortformer v2.1 low-latency
    → PuriPuly B500

diart + pyannote segmentation-3.0
    → A
    → B500
    → latency 0.5 / 1.0 sweep
```

그리고 주로 본다.

```text
Change F1 @ ±250 / ±500 ms
False changes / minute
Missed changes / minute
Overlap F1
Observed detection latency p50 / p95
RTF / compute / memory
```

이 결과를 바탕으로:

```text
Sortformer를 primary distillation teacher로 확정할지

diart 계열을 보조 teacher / reference로 사용할지

teacher를 쓰더라도 어떤 target(change / overlap)에 집중할지

최종 student가 어느 수준의 성능과 latency를 목표로 해야 할지
```

를 결정한다.

---

# 2. 개발 전략에서 이번 benchmark의 위치

TC와 PSEM은 직렬 한 줄이 아니라 일부 병렬로 진행한다.

```text
                     ┌──────────────────────────────┐
Public GT data ─────>| TC — Teacher Characterization|
     |               | reference / KD suitability  |
     |               └──────────────┬───────────────┘
     |                              |
     |                              | selected canonical teacher targets
     v                              v
PSEM Phase 0 ──> PSEM Phase 1/2 ──> PSEM Phase 3 KD
Dataset/labels      GT-only            optional teacher-dependent
```

중요한 원칙:

- PSEM의 GT-only baseline은 TC 완료를 기다리지 않는다.
- teacher distillation은 GT supervision을 대체하지 않는다.
- TC는 `가장 강한 reference`와 `가장 유용한 KD teacher`를 별도로 판단한다.
- teacher가 최종 product threshold에서 직접 성공해야만 KD source가 되는 것은 아니다.
- 반대로 raw teacher evidence가 label noise를 만들 정도로 불안정하다면 높은 naive recall만으로 좋은 KD teacher라고 보지 않는다.
- TC가 선택에 사용한 evaluation data는 이후 PSEM final test로 재사용하지 않는다.

TC의 downstream deliverable은 backend-specific slot/cluster 형식이 아니라 **model-neutral target contract**여야 한다.

# 3. R6 / R8 / R9의 위치

기존 실험은 폐기되는 것이 아니라 이번 benchmark의 prior evidence가 된다.

## 3.1 R6

R6는 explicit CURRENT-speaker memory + candidate OTHER + persistence / VAD / overlap controller + handoff라는 제품형 접근을 시험했다.

그 결과는 향후 실험에 중요한 경고를 준다.

```text
aggregate speaker information이 존재하는 것
!=
극저 false-event 영역에서 유용한 product handoff가 되는 것
```

따라서 이번 benchmark는 단일 hard-event F1만으로 teacher를 고르지 않는다.

## 3.2 R8

R8은 `transcribe.cpp` Streaming Sortformer Q8_0를 사용한 direct pretrained feasibility probe였다.

R8의 Q8 결과는 다음을 보여주는 historical evidence다.

- raw probability에 speaker information이 존재한다.
- naive single-threshold onset decoder는 low-FE 영역에서 매우 약할 수 있다.
- native segment-start candidate stream은 높은 candidate recall을 가지지만 매우 많은 false candidates를 낼 수 있다.
- compute feasibility와 event quality는 분리해서 봐야 한다.

## 3.3 R9

R9은 **Q8 Sortformer candidate verification upper-bound probe**다.

R9은 다음을 탐색한다.

```text
native 0.5 segment-start candidates
        ↓
probability-only verification
        ↓
optional speaker-cache embedding verification
        ↓
recall-vs-false-event ceiling
```

R9의 false-event reference points는 **절대 pass/fail 기준이 아니다.**

또한 R9의 모델/runtime은 이번 새 benchmark의 primary Sortformer와 다르다.

```text
R9:
  Q8_0
  CPU probability dumps
  development-known AMI/AliMeeting panel

이번 benchmark:
  F32 GGUF
  Vulkan
  expanded public-data characterization
```

따라서:

> **R9은 이번 benchmark의 continuation gate가 아니다.**

R9 결과는 어떤 failure mode와 어떤 verifier/evidence를 집중 관찰할지 알려주는 prior / hypothesis generator로 사용한다.

---

# 4. Primary systems

## 4.1 Sortformer characterization candidate

이번 benchmark의 Sortformer는 다음으로 고정한다.

```text
runtime:
  handy-computer/transcribe.cpp

model family:
  NVIDIA Streaming Sortformer 4spk v2.1

artifact:
  F32 GGUF

backend:
  Vulkan

streaming preset:
  TRANSCRIBE_SORTFORMER_PRESET_LOW_LATENCY
```

low-latency geometry:

```text
frame unit          = 80 ms
chunk               = 6 × 80 ms
right context       = 7 × 80 ms
FIFO                = 188 frames
update period       = 144 frames
speaker cache       = 188 frames

input-buffer / algorithmic lookahead
≈ (6 + 7) × 80 ms
≈ 1.04 s
```

이 1.04초는 compute time과 분리해서 기록한다.

이번 characterization은 F32/Vulkan/Low-Latency **한 profile만** 사용한다. ULR / 0.32 s track은 이 문서와 PSEM v0 계획에서 제외하고, F32↔Q8 parity arm은 이번 실험에 포함하지 않는다.

### Sortformer에서 반드시 저장할 것

```text
raw T × 4 speaker-activity probabilities
native speaker segments
canonical audio-consumed frontier
per-update / per-chunk processing timing
Vulkan device identity
runtime / model revision receipt
```

speaker-cache embedding/state는 telemetry-only extraction으로 안전하게 얻을 수 있으면 저장한다.

단:

> embedding dump가 어렵다는 이유만으로 primary benchmark를 막지 않는다.

### Sortformer tuning 정책

Sortformer는 이번 benchmark에서 F32/Vulkan/low-latency 단일 profile로 고정한다.

기존 R8/R9 Q8 계열 실험의 threshold / post-processing을 B 결과를 보고 다시 맞추지 않는다.

만약 기존 실험에서 TEST 데이터를 보면서 threshold를 조정했다면:

```text
DEV / TEST split을 새로 확정
↓
DEV에서 threshold freeze
↓
TEST를 다시 한 번만 평가
```

해야 공식 수치로 사용할 수 있다.

## 4.2 diart comparison/reference candidate

primary diart 구조는 native rolling-window streaming을 유지한다.

```text
segmentation:
  pyannote/segmentation-3.0 계열

embedding:
  pyannote speaker embedding

pipeline:
  rolling segmentation
  + embedding
  + incremental clustering
```

primary latency sweep:

```text
D-L050:
  duration = 5.0 s
  step     = 0.5 s
  latency  = 0.5 s

D-L100:
  duration = 5.0 s
  step     = 0.5 s
  latency  = 1.0 s
```

Sortformer의 480 ms chunk geometry를 diart에 강제로 맞추지 않는다.

### diart 환경 고정

첫 기준 환경:

```text
diart 0.9.2 계열
pyannote.audio >= 3.0, < 3.1
segmentation = pyannote/segmentation-3.0
embedding    = pyannote/embedding
```

실험 시작 시 반드시 실제 설치 버전을 lock file에 기록한다.

```text
environment/
  pip_freeze.txt
  python_version.txt
  torch_version.txt
  cuda_version.txt
  gpu.txt
```

pyannote models는 Hugging Face에서 gated access가 필요할 수 있으므로 실행 전 user conditions 동의와 token authentication을 완료한다.

### diart parameter tuning 정책

이번 benchmark의 목적이 성능 상한 비교이므로, 최종 공식 diart score는 가능하면 DEV(TC-DEV) set에서 parameter tuning을 한 뒤 frozen evaluation에서 측정한다.

튜닝 가능한 핵심:

```text
tau_active
rho_update
delta_new
```

단, 다음 규칙은 절대 지킨다.

```text
DEV  → tuning 가능
TEST → tuning 금지
```

frozen evaluation 결과를 본 뒤 threshold를 바꾸면 해당 score는 공식 비교에서 제외한다.

### diart tuning objective

DER 하나만 최적화하지 않는다.

PuriPuly 목적은 full diarization이 아니라 speaker-change / overlap event이기 때문이다.

추천 tuning 순서:

```text
1. mean B Change F1이 가장 높은 설정
2. 거의 같은 Change F1이라면 false event/min이 더 낮은 설정 선택
3. 다시 비슷하면 overlap frame F1이 높은 설정 선택
```

즉 임의의 복잡한 가중합보다 lexicographic rule을 사용한다.

예:

```text
Config X
mean B Change F1 = 0.932
false/min        = 0.41

Config Y
mean B Change F1 = 0.931
false/min        = 0.18
```

차이가 사실상 미미한 범위라고 판단하면 Y를 선호한다.

> Tuning 기준과 "미미한 차이" tolerance는 실험 시작 전에 고정한다.  
> 예: Change F1 절대값 0.005 이내.

---

# 5. Backend를 공정하게 비교하는 지점

내부 구조를 같게 만들지 않는다.

```text
Sortformer
  native streaming model
  + AOSC speaker cache
  + F32/Vulkan

diart
  native rolling segmentation
  + embedding
  + online clustering
```

공정성은 다음 공통 계약에서 확보한다.

```text
동일 source audio
동일 16 kHz canonical timeline
동일 GT normalization
동일 canonical event taxonomy
동일 A / B500 evaluation projections
동일 matching
동일 metric definitions
동일 latency accounting principles
```

---

# 6. 세 층 평가 계약

이번 benchmark의 핵심 contract다.

## Layer 1 — Native teacher evidence

각 backend가 실제로 가진 정보를 최대한 보존한다.

Sortformer 예:

```text
T×4 probability
native segment
slot activation / reactivation pattern
optional cache embedding evidence
```

diart 예:

```text
incremental Annotation
segmentation / local activity evidence, if exposed
speaker embedding evidence, if exposed
clustering similarity / assignment evidence, if exposed
```

Layer 1의 목적은:

> **teacher 내부에 학습 가능한 speaker-change / overlap / short-term continuity 정보가 얼마나 남아 있는가?**

를 보는 것이다.

여기서 관찰되는 richer speaker evidence는 PSEM v0가 그 정보를 모두 출력해야 한다는 뜻이 아니다.

PSEM v0의 필수 model outputs는 별도 문서에서 정의한 `P(speaker_change)`, `P(overlap)`, `speaker_embedding`이다.

## Layer 2 — Canonical speaker events

backend별 native 형식을 model-neutral event vocabulary로 투영한다.

```text
new_speaker_candidate
new_speaker_onset
handoff_confirmed
overlap_start
overlap_end
same_speaker_resume
```

Layer 2는 teacher/reference characterization vocabulary다. 특히 overlap 안에서 `new_speaker_onset`을 관찰할 수 있어도 그것을 PSEM v0의 `speaker_change`와 동일시하지 않는다.

PSEM v0와 직접 대응하는 핵심 의미는:

```text
PSEM speaker_change  <-> handoff_confirmed semantics
PSEM overlap         <-> overlap state / start / end semantics
speaker embedding    <-> GT-trained short-term representation; direct teacher embedding regression은 요구하지 않음
```

## Layer 3 — Evaluation / policy projection

canonical evidence를 다음 primary views로 투영한다.

```text
A
B500
```

secondary replay:

```text
B500 + current 5~7 s bounded-turn policy
B500 + legacy hard-7 diagnostic
speaker-change-assisted cap policy
```

이 layer는 benchmark diagnostic을 위한 projection이다. overlap 표시 UI, transcript/translation 처리, speaker embedding의 제품 소비 방식은 이 문서에서 정하지 않는다.

필요한 hard-boundary diagnostic은 기존 provider-neutral 개념인:

```text
logical_finalize(boundary_source_sample)
```

까지 기록할 수 있지만, 실제 STT/LLM/UI implementation policy는 별도 integration 문서의 책임이다.

이 분리를 통해:

```text
teacher evidence는 좋음 + hard decoder는 나쁨
```

과

```text
teacher evidence 자체가 약함
```

을 구분한다.

# 7. Canonical source timeline과 기존 harness 재사용

새 독립 canonical stack을 만들지 않는다.

기준은 기존:

```text
experiments/speaker_turn_boundary/
```

의 contract를 재사용한다.

재사용 대상:

```text
16 kHz mono canonical source timeline
audio_epoch
SourcePosition
SpeakerBoundaryEvent
DetectorProgress
boundary_source_sample
observed_source_sample_at_emit
one-to-one matching semantics
GT active-speaker transition semantics
VAD / detector coalescing primitives
result identity / reproducibility concepts
```

특히 timestamp는 다음을 분리한다.

```text
boundary_source_sample
  = event가 실제로 속하는 source 위치

observed_source_sample_at_emit
  = 그 결정을 낼 때까지 소비한 source audio frontier

processing / wall time
  = 실제 compute / scheduling cost
```

event location accuracy와 causal availability를 한 timestamp로 합치지 않는다.

---

# 8. Ground Truth normalization

GT speaker annotation을 source-contiguous active-speaker intervals로 정규화한다.

예:

```text
[start, end)  active_speakers

0.00–0.80     {A}
0.80–1.00     {A,B}
1.00–1.70     {B}
1.70–2.20     {}
2.20–3.00     {B}
```

원본 timestamp를 보존한다.

80 ms grid는 Sortformer 분석용 projection으로 사용할 수 있지만:

> **GT boundary의 canonical 위치 자체를 80 ms로 강제 양자화하지 않는다.**

---

# 9. Canonical reference taxonomy

## 9.1 Clean direct handoff

```text
{A} → {B}
```

reference:

```text
new_speaker_onset
handoff_confirmed
```

clean/gap hard-turn headline target에 포함한다.

## 9.2 Gap speaker change

```text
{A} → {} → {B}
```

A view에서는 speaker change다.

B500 view에서는 oracle VAD turn이 gap 안에서 이미 닫혔는지에 따라 speaker-change 필요 여부가 달라진다.

## 9.3 Same-speaker gap / resume

```text
{A} → {} → {A}
```

speaker-change positive가 아니다.

중요한 hard-negative다.

## 9.4 Interruption / overlap onset

```text
{A} → {A,B}
```

reference:

```text
new_speaker_onset
overlap_start
```

단:

> **이 event를 곧바로 hard logical turn cut 정답으로 취급하지 않는다.**

teacher characterization에서는 중요한 positive evidence지만 product hard-boundary headline에서는 clean/gap handoff와 분리한다.

## 9.5 Overlap return

```text
{A} → {A,B} → {A}
```

reference:

```text
overlap_start
overlap_end
NO hard speaker handoff
```

## 9.6 Overlap takeover

```text
{A} → {A,B} → {B}
```

reference:

```text
new_speaker_onset at B entry
overlap_start
overlap_end
handoff_confirmed / exclusive-new-speaker state after takeover
```

teacher event와 product hard action의 timestamp를 반드시 같은 것으로 가정하지 않는다.

## 9.7 Complex overlap

예:

```text
{A,B} → {B,C}
```

v0 headline에서는 full participant attribution을 요구하지 않는다.

별도 diagnostic bucket으로 보존한다.

---


## 9.8 PSEM v0와의 canonical mapping

TC의 event taxonomy가 PSEM output보다 더 세밀하다는 점을 명시적으로 유지한다.

```text
TC: new_speaker_onset
  = overlap 안에서 새로운 화자 관련 evidence가 처음 나타나는 시점까지 분석 가능
  = PSEM v0의 필수 output 아님

TC: handoff_confirmed
  = reliable single-speaker state 전후의 화자가 달라졌다고 볼 수 있는 전환
  = PSEM v0 speaker_change의 label/evaluation 의미와 직접 대응

TC: overlap_start / overlap_end
  = PSEM v0 overlap state와 직접 대응
```

따라서 `A -> A+B -> A`는 TC에서 new-speaker/overlap evidence를 분석할 수 있어도 PSEM speaker-change positive가 아니다.

`A -> A+B -> B`에서는 B 진입 시점의 richer evidence와 overlap 종료 뒤 handoff confirmation을 분리해서 저장한다. PSEM v0의 `speaker_change` target은 후자에 맞춘다.

# 10. A evaluation view

A는 backend의 session-level speaker continuity 능력을 본다.

silence가 길어도 평가 memory를 reset하지 않는다.

예:

```text
A
↓
5 s silence
↓
B
```

A:

```text
different speaker
→ speaker-change reference 유지
```

A는 제품 action 그 자체가 아니라 teacher/reference capability diagnostic이다.

---

# 11. VAD와 speaker model의 관계, 그리고 B500 정의

## 11.1 VAD와 화자 모델은 서로 다른 문제를 푼다

VAD(Voice Activity Detection)는 기본적으로 다음을 판단한다.

```text
지금 사람이 말하고 있는가?
말이 시작되었는가?
말이 끝났는가?
```

화자 전환 모델은 다음을 판단한다.

```text
계속 말이 이어지고 있는데 사람이 바뀌었는가?
동시에 두 명 이상이 말하는가?
```

예를 들어:

```text
A: 안녕하세요
B: 네 반갑습니다
```

두 화자 사이에 거의 침묵이 없다면 VAD 입장에서는 이것을 하나의 연속된 speech region으로 볼 수 있다.

```text
VAD:
speech_start ───────────────── speech_end
             A        B
```

하지만 PuriPuly는 내부적으로:

```text
A
↓ speaker_change
B
```

를 알고 싶을 수 있다.

따라서 VAD와 speaker model은 경쟁 관계가 아니라 역할 분담 관계다.

---

## 11.2 최종 런타임에서의 관계: late fusion

중요: **새 speaker model이 VAD 출력을 neural input으로 받는 구조가 아니다.**

둘은 같은 오디오를 병렬로 처리한다.

```text
                         ┌── PuriPuly VAD
16 kHz mono audio ───────┤
                         └── Speaker Model
                                  │
                                  ├── speaker_change probability
                                  └── overlap probability

VAD events + speaker events
          ↓
     state/event manager
```

즉 모델 결합이 아니라 **late fusion** 이다.

---

## 11.3 PuriPuly VAD가 speaker-change 문제를 쉽게 만드는 이유

이 실험에서 VAD를 고려해야 하는 가장 중요한 이유다.

다음 상황을 생각한다.

```text
A
↓
300 ms silence
↓
B
```

PuriPuly의 VAD hangover가 500 ms라면 300 ms 침묵 동안에도 아직 같은 speech turn으로 간주된다.

따라서 speaker model이 다음을 알아내야 한다.

```text
A → B
speaker_change = YES
```

반대로:

```text
A
↓
800 ms silence
↓
B
```

hangover가 500 ms라면 A 이후 이미 VAD가 `speech_end`를 발생시킨다.

```text
A
↓
speech_end
↓
B의 새로운 speech_start
```

이 경우 **speaker model이 A→B change를 검출할 필요가 없다.**

이미 VAD가 기존 turn을 끝냈기 때문이다.

즉 같은 A→B라도 중간 침묵 길이에 따라 speaker model에게 요구되는 문제가 달라진다.

이것이 A 평가와 B 평가를 분리하는 이유다.

---

## 11.4 B500 정의 — primary product-context view

`B500`은 다음으로 고정한다.

```text
oracle GT speech activity
+
configured silence hangover = 500 ms
+
product-equivalent chunk quantization
+
NO max-turn cap
```

16 kHz, 512-sample chunk 기준:

```text
chunk = 32 ms
ceil(500 / 32) = 16 chunks
effective hangover = 512 ms
```

metadata:

```text
configured_hangover_ms = 500
effective_hangover_ms  = 512
max_turn_ms            = null
```

## 11.5 B500이 하는 일

GT speaker union으로 perfect speech/non-speech timeline을 만든다.

```text
어떤 speaker라도 active → speech
아무도 active 아님     → silence
```

그 위에 silence hangover만 적용한다.

B500은 다음을 흉내내지 않는다.

```text
Silero speech probability error
threshold error
start debounce error
noise false positive
```

즉:

```text
perfect speech/non-speech
+
500 ms configured product hangover
```

만 본다.

## 11.6 B500이 하지 않는 일

B500에는 다음이 없다.

```text
5 s soft cap
7 s hard cap
legacy PEER_MAX_SEGMENT_MS
speaker-change forced cut
actual Silero errors
```

따라서 primary B500 characterization 결과에 max-duration split이 섞이지 않는다.

---

## 11.7 왜 B에서 실제 PuriPuly VAD를 사용하지 않는가

이번 실험은 speaker backend 성능 비교다.

실제 VAD를 사용하면 결과에 두 오류가 섞인다.

```text
speaker backend 오류
+
Silero VAD 오류
```

예를 들어 실제 VAD가 너무 일찍 speech_end를 내면 원래 speaker model이 맞혀야 할 A→B change가 평가 대상에서 사라질 수도 있다.

그러면 speaker model이 실제보다 좋아 보인다.

반대로 VAD가 speech_end를 너무 늦게 내면 불필요하게 더 어려운 speaker-change 문제를 backend에게 요구할 수도 있다.

따라서 이번 단계에서는:

```text
Ground Truth speech activity
       ↓
PuriPuly turn policy
       ↓
oracle-policy VAD
```

를 사용한다.

실제 PuriPuly VAD까지 포함한 end-to-end 평가는 나중에 별도의 **C benchmark**로 추가한다.

이번 문서의 범위는 C가 아니다.

---

# 12. B500 oracle-policy pseudo algorithm

continuous GT boundary를 먼저 계산하고 product-equivalent chunk semantics로 투영한다.

```python
CONFIGURED_HANGOVER_MS = 500
EFFECTIVE_HANGOVER_MS = 512
MAX_TURN_MS = None

turn_open = False
silence_start = None

for source_position in canonical_timeline:
    speech = any_gt_speaker_active(source_position)

    if not turn_open:
        if speech:
            emit_oracle_turn_start(source_position)
            turn_open = True
            silence_start = None
        continue

    if speech:
        silence_start = None
        continue

    if silence_start is None:
        silence_start = source_position

    if product_equivalent_elapsed(silence_start, source_position) >= 512 ms:
        emit_oracle_turn_end(reason="silence")
        turn_open = False
        silence_start = None
```

`max_duration` branch는 존재하지 않는다.

---

# 13. Max-turn / bounded-turn interaction study

max-turn은 teacher evidence와 다른 종류의 정책이다.

speaker-change evidence:

```text
"이 source 위치가 의미적으로 speaker boundary일 가능성이 높다"
```

max-turn cap:

```text
"의미적 boundary를 못 찾았더라도 turn을 더 이상 늘리지 않는다"
```

따라서 둘을 동일한 event origin으로 합치지 않는다.

## 13.1 Secondary policy replay arms

TC의 reference/KD role characterization과 분리하여 같은 raw outputs에서 별도 replay한다.

```text
CAP-NONE
  = B500 only

CAP-CURRENT
  = 다른 제품 브랜치의 현재 5~7 s bounded-turn semantics

CAP-LEGACY-7
  = legacy hard 7 s diagnostic only

CAP-SCD-ASSISTED
  = accepted speaker-change를 primary logical boundary로 사용하고
    bounded-turn cap은 fallback safety policy로 사용
```

## 13.2 Current 5~7 s 정책 import rule

이 문서는 current policy의 세부 동작을 추측하지 않는다.

실행 전에 반드시:

```text
owning branch
commit SHA
source path
config
exact soft/hard cap semantics
timer reset semantics
```

를 receipt로 고정한다.

다른 브랜치에서 실제 구현된 정책을 그대로 policy adapter로 가져온다.

## 13.3 Cap study의 제품 질문

```text
speaker-change가 있으면 forced cap cut이 얼마나 줄어드는가?

speaker-change 후 cap age를 reset하면
불필요한 연속 강제 cut이 얼마나 줄어드는가?

no-cap에서 pathological long turn이 얼마나 늘어나는가?

current 5~7 policy가 의미적 speaker boundary 직전/직후에
불필요한 fragmentation을 만드는가?

speaker-change를 primary로 쓰고 cap을 fallback으로 축소할 수 있는가?

최종적으로 cap을 유지 / 완화 / 제거할 근거가 있는가?
```

이 결과는 teacher selection headline과 분리해서 보고한다.

---

# 14. Public data — 이번 문서에서 고정하는 방향만

정확한 공개 corpus 선정은 **별도 세션으로 보류한다.**

이 문서에서는 데이터의 역할과 필수 coverage만 고정한다.

## 14.1 데이터 pool의 역할

최소 세 그룹으로 분리한다.

```text
1. development-known regression pool
   - 기존 R8/R9 exposure 포함 가능
   - regression / failure reproduction / adapter sanity

2. new development pool
   - parameter tuning
   - diart tuning
   - event policy development
   - threshold / verifier analysis

3. frozen evaluation pool
   - tuning 전에 freeze
   - final teacher/reference characterization
```

## 14.2 필수 event coverage 방향

새 데이터는 단순 총 시간보다 다음 strata의 충분한 coverage를 우선한다.

```text
clean direct handoff
silence-gap different-speaker change
same-speaker pause / resume
overlap / interruption onset
overlap takeover
overlap return
short backchannel
speaker return
long stable same-speaker speech
hard-negative non-change regions
```

## 14.3 환경 다양성 방향

가능한 범위에서 다음 축을 넓힌다.

```text
KO / EN / JA / ZH
same-language conversation
cross-language conversation
same-speaker code-switch

2-speaker conversation
multi-speaker meeting

near-field
far-field
noise
reverb
different microphones / distances
```

## 14.4 Training-overlap metadata

pretrained teacher가 사용했을 가능성이 있는 corpus는 무조건 버리는 대신 다음처럼 명시한다.

```text
known_train_overlap
possible_train_overlap
unknown
best-effort-disjoint
```

training-overlap 가능성이 큰 데이터는 confirmatory claim보다 regression / development-known evidence로 취급한다.

## 14.5 이번 문서가 정하지 않는 것

다음은 별도 데이터 설계 세션에서 결정한다.

```text
정확한 corpus 목록
corpus별 시간
language별 quota
event별 minimum count
license / download workflow
frozen panel의 최종 구성
```

## 14.6 최소 annotation 요구사항

모든 backend는 **동일한 audio 파일**을 사용한다.

최소 필요 annotation:

```text
speaker ID
start timestamp
end timestamp
```

overlap은 두 speaker segment가 겹치도록 annotation되어 있어야 한다.

권장 format:

```text
RTTM
```

또는 RTTM으로 무손실 변환 가능한 내부 format.

## 14.7 언어 구성

가능하다면 다음 조합을 포함한다.

```text
KO ↔ KO
EN ↔ EN
JA ↔ JA
ZH ↔ ZH

KO ↔ EN
KO ↔ JA
KO ↔ ZH
EN ↔ JA
EN ↔ ZH
JA ↔ ZH
```

또한:

```text
같은 언어 / 다른 화자
다른 언어 / 다른 화자
같은 화자의 code-switch
```

를 구분해 기록할 수 있으면 좋다.

단, language label 자체는 이번 speaker backend의 입력이 아니다.

## 14.8 pyannote segmentation-3.0 데이터 중복 주의

`pyannote/segmentation-3.0` model card에는 학습 데이터 조합으로 다음 계열이 명시되어 있다.

```text
AISHELL
AliMeeting
AMI
AVA-AVD
DIHARD
Ego4D
MSDWild
REPERE
VoxConverse
```

따라서 이들 benchmark를 그대로 frozen evaluation으로 사용할 경우 완전히 새로운 product-like data보다 유리할 가능성이 있다.

가능하면 최종 결론은:

```text
PuriPuly 실제 환경과 유사하지만
모델 학습에 사용되지 않은 별도 held-out data
```

에서 내리는 것을 권장한다.

공개 benchmark 결과는 참고값으로 별도 보고한다.

---

# 15. DEV / frozen evaluation discipline

최소 세 역할을 분리한다.

```text
TC-REGRESSION
  기존 R8/R9-compatible regression / smoke / implementation parity 용도

TC-DEV
  eventizer, threshold, verifier, adapter 선택에 사용 가능

TC-EVAL
  TC의 reference suitability / KD teacher suitability 결론을 내기 위한 frozen evaluation
```

## 15.1 TC-DEV에서 허용되는 변경

TC-DEV는 탐색과 adapter/eventizer 개발을 위한 구간이다. 다음 변경은 TC-DEV 안에서 허용한다.

```text
diart tau_active / rho_update / delta_new tuning
Eventizer / policy sanity 및 tuning
optional verifier development
diagnostic threshold selection
adapter instrumentation / raw-artifact sanity fix
```

단, 이 변경들은 TC-EVAL을 보기 전에 freeze되어야 한다.

## 15.2 TC-EVAL에서 금지되는 변경

TC-EVAL을 확인한 뒤에는 다음을 하지 않는다.

```text
결과를 본 뒤 parameter 변경
결과를 보고 corpus 교체
stratum 결과를 본 뒤 threshold 재조정
backend별로 서로 다른 GT / matching / collar 적용
불리한 session만 사후 제외
```

이 규칙은 Sortformer와 diart에 동일하게 적용한다. backend마다 다른 평가 규칙을 적용해 frontier를 유리하게 만들지 않는다.

## 15.3 downstream data lifecycle

중요한 lifecycle 규칙:

> **TC-EVAL을 teacher/reference 선택에 사용한 순간, 그 데이터는 downstream PSEM 관점에서 development-known이 된다.**

따라서 PSEM의 최종 confirmatory test는 별도의 `PSEM-FINAL`을 사용한다.

```text
TC-EVAL
  -> teacher/reference 선택에 사용
  -> 이후 PSEM-DEV 성격으로만 취급 가능

PSEM-FINAL
  -> teacher 선택, target 설계, architecture/KD/compression 결정에 사용하지 않음
  -> 최종 student 확인용으로 별도 유지
```

frozen evaluation 결과를 본 뒤 TC eventizer/threshold를 바꾸면 새 evaluation cycle로 취급한다.

# 16. Raw artifact contract

최종 RTTM이나 score만 저장하지 않는다.

## 16.1 모든 backend mandatory

```text
source file / session identity
audio hash or immutable source receipt
sample rate
canonical source duration

native output
canonical audio position
audio consumed frontier
processing timestamps
runtime metadata
model/runtime revision

normalized active-speaker representation
canonical events
```

## 16.2 Sortformer mandatory

```text
T×4 speaker probabilities
native segments
Vulkan backend/device identity
low-latency preset receipt
per-update compute timing
```

## 16.3 diart mandatory

```text
incrementally emitted Annotation
speaker labels / active sets
audio consumed frontier
processing timing
```

## 16.4 diart optional-but-desired soft evidence

API / instrumentation으로 안정적으로 얻을 수 있다면 저장한다.

```text
segmentation posterior / local activity score
speaker embeddings
clustering similarity / assignment score
```

없다고 해서 diart arm이 invalid가 되는 것은 아니다.

단 soft-target teacher로서의 capability가 제한된다는 사실은 report에 명시한다.

---

# 17. Native evidence → normalized evidence adapter

backend별 adapter는 native output을 canonical source timeline에 맞춘다.

예시 normalized record:

```json
{
  "backend": "sortformer_v2.1_f32_vulkan",
  "file_id": "sample_001",
  "source_sample": 199680,
  "observed_source_sample": 216320,
  "active_speakers": ["slot1"],
  "overlap": false,
  "processing_ms": 24.8,
  "native_ref": "frame_156"
}
```

raw probability / embedding처럼 큰 tensor는 별도 artifact에 저장하고 record에는 immutable reference를 둔다.

---

# 18. Common canonical Eventizer

backend마다 서로 다른 speaker-change 정의를 만들지 않는다.

```text
native backend output
        ↓
backend adapter
        ↓
normalized active-speaker evidence
        ↓
COMMON CANONICAL EVENTIZER
```

Eventizer는 최소 다음을 출력할 수 있어야 한다.

```text
new_speaker_candidate
new_speaker_onset
handoff_confirmed
overlap_start
overlap_end
same_speaker_resume
```

backend-specific native event는 별도 diagnostic으로 보존할 수 있지만 headline comparison은 common canonical events를 사용한다.

## 18.1 Eventizer 상태

최소 상태:

```text
last_reliable_solo_speaker
in_overlap
vad_turn_open
```

A 모드에서는 `vad_turn_open`을 항상 true로 취급한다.

B 모드에서는 oracle-policy VAD가 제어한다.

## 18.2 Direct speaker change 정책

입력:

```text
{A}
{A}
{B}
{B}
```

이전 reliable solo speaker:

```text
A
```

새 solo:

```text
B
```

둘이 다르면:

```text
speaker_change
```

를 발생시킨다.

Change timestamp는 새 solo speaker가 시작된 시점으로 한다.

## 18.3 Silence 정책 — A

A에서는 silence가 speaker state를 reset하지 않는다.

```text
{A}
{}
{}
{}
{B}
```

결과:

```text
A != B
→ speaker_change
```

silence가 300 ms든 10초든 동일하다.

A는 full-session speaker continuity 능력을 보는 diagnostic이기 때문이다.

## 18.4 Overlap 동안 speaker state는 freeze

```text
last_reliable_solo_speaker = A
```

상태에서 overlap이 시작되면:

```text
A+B
```

동안 `last_reliable_solo_speaker`를 갱신하지 않는다.

overlap이 끝난 뒤 solo speaker를 A와 비교한다.

## 18.5 3명 이상 overlap

PuriPuly v0 평가에서는 정확한 speaker count를 맞히는 문제가 아니다.

따라서:

```text
2 speakers active
3 speakers active
4 speakers active
```

모두:

```text
overlap = true
```

로 collapse한다.

정확한 participant identity / count는 headline metric에 포함하지 않는다.

## 18.6 복잡한 overlap identity transition

예:

```text
A+B → B+C
```

이런 경우 정확히:

```text
A left
C entered
```

를 알아내는 것은 full diarization에 가까운 문제다.

현재 학생 모델 v0 범위가 아니므로 speaker_change headline score에 강제로 identity event를 만들지 않는다.

가능하면 별도 `hard_overlap_transition` bucket으로 보존한다.

---

# 19. B500 projection semantics

B500은 backend inference를 다시 실행하지 않는다.

동일 raw inference에서:

```text
canonical events
+
oracle B500 turn state
        ↓
B500-required-event view
```

를 만든다.

예:

```text
A
↓ 300 ms GT silence
B
```

```text
300 ms < effective 512 ms
→ same oracle VAD turn
→ different speaker change 필요
```

반면:

```text
A
↓ 800 ms GT silence
B
```

```text
800 ms >= effective 512 ms
→ oracle VAD turn closed
→ B는 새 turn 첫 speaker
→ cross-turn speaker-change hard requirement 없음
```

backend가 예측한 silence 자체로 B500 memory를 reset하지 않는다.

---

# 20. Teacher information-content evaluation

teacher viability는 최종 hard event 한 줄로만 판단하지 않는다.

가능한 경우 다음을 본다.

```text
candidate-stream recall ceiling
score / margin separation around positive vs hard-negative regions
recall-vs-false-event frontier
overlap evidence quality
same-speaker resume rejection
short-backchannel retention
latency of evidence availability
short-term speaker continuity / relational evidence
```

특히 overlap에서는 teacher가 다음 정보를 어느 정도 보존하는지 별도 diagnostic으로 볼 수 있다.

```text
overlap 자체의 존재
new-speaker-related evidence의 등장
pre/post-overlap speaker continuity
handoff confirmation에 필요한 evidence
```

다만 이 richer information-content 결과를 곧바로 PSEM head 요구사항으로 변환하지 않는다. PSEM v0는 overlap 내부 participant attribution을 목표로 하지 않는다.

continuous soft score가 없는 backend는 가능한 native evidence 수준까지만 평가한다. 시스템마다 없는 tensor를 억지로 만들어 비교하지 않는다.

# 21. Canonical event metrics

speaker-change / new-speaker event:

```text
Precision
Recall
F1 @ ±250 ms
F1 @ ±500 ms

False events / source hour
False events / source minute

Missed events / source hour
```

필요하면 active-speech denominator도 병행 보고한다.

특히 false-event rate는 반드시 headline metric으로 둔다.

PuriPuly에서 false change는 context/turn을 불필요하게 분리할 수 있기 때문이다.

matching은 deterministic one-to-one matching을 사용한다.

한 GT event에 여러 prediction이 collar 안에 들어오면 하나만 TP이고 나머지는 FP다.

## 21.1 Change event matching

Reference change:

```text
10.000 s
```

prediction:

```text
10.180 s
```

±250 ms metric에서는 TP.

prediction:

```text
10.430 s
```

이면:

```text
±250 ms → miss
±500 ms → TP
```

따라서 두 collar를 모두 보고한다.

```text
Change F1 @ ±250 ms
Change F1 @ ±500 ms
```

## 21.2 One-to-one matching

한 GT event에 prediction 하나만 match한다.

예:

```text
GT:
10.000

Pred:
9.900
10.100
10.200
```

collar 안에 세 개가 있어도:

```text
TP = 1
FP = 2
```

다.

matching은 같은 collar 안에서 시간 차이가 가장 작은 pair를 우선하는 one-to-one matching을 사용한다.

dataset 전체에서 동일 알고리즘을 사용한다.

---

# 22. False-event frontier — 절대 기준 없음

primary deliverable은 다음 곡선이다.

```text
recall / F1
    vs
false-event rate
    vs
observed availability latency
```

특정 false-event rate를 합격선으로 고정하지 않는다.

예:

```text
1
5
10
20
50
100 false events/hour
```

를 표에 넣을 수 있지만:

> **reference operating points일 뿐 pass/fail gate가 아니다.**

reference/KD role 판단은 전체 frontier와 error composition을 함께 보고 수행한다.

---

# 23. Overlap metrics

overlap은 hard handoff와 분리한다.

보고:

```text
Overlap frame Precision / Recall / F1

Overlap-start event F1 @ ±250 / ±500 ms
Overlap-end event F1 @ ±250 / ±500 ms

overlap takeover diagnostics
overlap return diagnostics
```

정확한 overlap participant count / identity는 v0 headline이 아니다.

---

# 24. Product hard-action diagnostics

teacher event와 별도로 clean/gap product target을 본다.

headline hard-action target:

```text
clean direct handoff
gap speaker change that remains inside the active B500 turn
```

severe harm:

```text
stable same-speaker active speech 안의 hard logical boundary
```

separate fragmentation cost:

```text
same-speaker pause/resume 주변 불필요한 split
```

overlap onset은 hard-action benefit score에 강제로 넣지 않는다.

---

# 25. Latency contract

각 event에 다음을 저장한다.

```text
reference_source_sample
predicted_boundary_source_sample
observed_source_sample_at_emit
processing_start / stop
runtime emission time, if measured
```

## 25.1 Boundary location error

```text
predicted_boundary - reference_boundary
```

## 25.2 Evidence availability delay

```text
observed_source_sample_at_emit
-
reference_source_sample
```

예:

```text
GT speaker change:
10.000 s

그 change를 포함한 prediction이
audio 10.820 s까지 소비한 뒤 처음 확정됨
```

그러면:

```text
algorithmic detection delay
= 10.820 - 10.000
= 820 ms
```

이를 모든 event에 대해 계산한다.

## 25.3 Compute time

algorithmic buffering과 별도 기록한다.

최종 보고:

```text
availability delay p50 / p90 / p95
compute p50 / p95 / p99
RTF
peak memory
```

## 25.4 Processing latency

모델 계산 시간은 별도로 측정한다.

Sortformer 공식 1.04 s는 input buffer latency이며 compute time은 포함하지 않는다.

따라서 실제 제품 관점에서는:

```text
input / algorithmic waiting
+
model compute
+
optional event stabilization
```

을 함께 봐야 한다.

## 25.5 Timing 측정 방법

Accuracy benchmark를 빠르게 offline으로 돌려도 된다.

하지만 timing benchmark는 별도 측정한다.

권장:

```text
batch_size = 1
warmup = 충분히 수행
동일 GPU / 동일 precision
각 파일 3회 이상
첫 warmup run 제외
```

기록:

```text
mean update compute ms
p50 update compute ms
p95 update compute ms
RTF
peak GPU memory
```

가능하면 Sortformer와 diart는 같은 hardware에서 측정한다.

---

# 26. Sortformer timing / Vulkan rules

이번 환경에는 CUDA를 요구하지 않는다.

Sortformer compute benchmark:

```text
backend = Vulkan
batch / stream semantics = transcribe.cpp native path
preset = LOW_LATENCY
model = F32 GGUF
```

반드시 기록:

```text
Vulkan device
driver/runtime
resolved backend identity
CPU fallback 여부
model load time
RTF
per-update compute timing
peak process memory
GPU memory, observable하면 기록
```

Vulkan이 material graph를 CPU로 silent fallback하면 그 compute result를 Vulkan 결과로 보고하지 않는다.

---

# 27. diart timing rules

diart는 실제 실행 가능한 backend를 그대로 기록한다.

CUDA가 없다는 이유로 Sortformer와 같은 Vulkan execution path를 강제하지 않는다.

teacher compute는 deployment gate가 아니므로:

```text
accuracy / teacher information
```

과

```text
runtime cost
```

를 분리해서 해석한다.

hardware/runtime이 다르면 raw RTF를 완전한 architecture fairness 수치로 과도하게 해석하지 않는다.

---

# 28. Mandatory experiment matrix

## 28.1 Inference runs

| Run | Backend | Native setting | Device |
|---|---|---|---|
| `SF-F32-VK-LL` | Sortformer v2.1 F32 GGUF / transcribe.cpp | low-latency preset | Vulkan |
| `D-L050` | diart + segmentation + embedding | latency 0.5 s | actual recorded backend |
| `D-L100` | diart + segmentation + embedding | latency 1.0 s | actual recorded backend |

mandatory inference = **3**

## 28.2 Evaluation views

| ID | Raw inference | Projection |
|---|---|---|
| `SF-F32-VK-A` | SF-F32-VK-LL | A |
| `SF-F32-VK-B500` | SF-F32-VK-LL | B500 |
| `D-A-L050` | D-L050 | A |
| `D-B500-L050` | D-L050 | B500 |
| `D-A-L100` | D-L100 | A |
| `D-B500-L100` | D-L100 | B500 |

mandatory evaluation views = **6**

기존 v3의 “diart 2 inference에서 evaluation view 6개”라는 표현은 삭제한다.

diart 자체는 2 inference → 4 views다.

Sortformer 1 inference → 2 views를 합쳐 전체 mandatory view가 6개다.

---

# 29. Optional secondary analyses

primary 결과를 오염시키지 않는 범위에서 다음을 허용한다.

## 29.1 diart tuned

DEV에서만:

```text
tau_active
rho_update
delta_new
```

를 tuning할 수 있다.

stock과 tuned는 반드시 별도 report한다.

## 29.2 R9-informed verification

R9이 probability-only 또는 embedding evidence에서 유용한 verifier family를 찾으면 F32 output에 대한 secondary analysis 설계에 참고할 수 있다.

단:

- R9 threshold를 그대로 복사하지 않는다.
- R9 결과를 F32 benchmark의 pass/fail gate로 쓰지 않는다.
- secondary 결과를 native/raw primary result와 섞지 않는다.

## 29.3 Product stabilization

예:

```text
candidate persistence
temporal debounce
duplicate suppression
```

추가 latency를 포함해서 별도 표시한다.

---

# 30. Scenario buckets

전체 평균 외에 반드시 다음 strata를 별도로 본다.

```text
clean_direct_switch

same_speaker_short_gap
different_speaker_short_gap

same_speaker_medium_gap
different_speaker_medium_gap

long_gap_same_speaker
long_gap_different_speaker

overlap_onset
overlap_return
overlap_takeover

short_backchannel
speaker_return

long_stable_same_speaker_negative

complex_overlap_transition
```

multilingual metadata가 있으면:

```text
same-language switch
cross-language switch
same-speaker code-switch
```

를 별도 aggregate한다.

## 30.1 bucket 정의

### clean_direct_switch

```text
A → B
```

중간 silence 거의 없음.

### same_speaker_short_gap / different_speaker_short_gap

```text
A → 100~400 ms silence → A
A → 100~400 ms silence → B
```

short gap에서 false change 억제 능력과 change recall을 각각 확인한다.

### same_speaker_medium_gap / different_speaker_medium_gap

```text
A → 500~1000 ms silence → A
A → 500~1000 ms silence → B
```

예:

```text
800 ms gap

B500:
VAD turn closed
→ change 불필요

same VAD turn이 유지되는 경우
→ change 필요
```

### long_gap_same_speaker / long_gap_different_speaker

```text
A → >1.1 s silence → A
A → >1.1 s silence → B
```

A와 B 성능 차이를 만드는 대표 bucket.

### overlap_return / overlap_takeover

```text
A → A+B → A     목표: overlap_start, overlap_end, NO speaker_change
A → A+B → B     목표: overlap_start, overlap_end, speaker_change
```

### short_backchannel

```text
A → B("네"/"응"/"yeah") → A
```

짧은 B를 놓치는지 확인.

## 30.2 Multilingual bucket

annotation이 가능하다면:

```text
same-language speaker switch
cross-language speaker switch
same-speaker code switch
```

를 별도 집계한다.

예:

```text
A(KO) → B(KO)
A(KO) → B(EN)
A(KO) → A(EN)
```

현재 Sortformer와 diart에게 language ID를 입력하지 않는다.

이는 향후 학생 모델의 multilingual auxiliary training 효과와 비교할 baseline으로 활용한다.

---

# 31. Error taxonomy와 hard-negative dump

단순 FP/FN 숫자만 남기지 않는다.

최소 taxonomy:

```text
same_speaker_resume
slot_flicker / identity_flicker
silence-gap confusion
overlap artifact
backchannel miss
return confusion
cache / clustering discontinuity
boundary timing late
boundary timing early
unknown
```

각 error 주변 source context reference를 저장한다.

향후 student hard-negative mining에 직접 사용할 수 있어야 한다.

---

# 32. Product cap replay metrics

cap study에서는 teacher headline metric과 다른 값을 본다.

```text
forced_cap_cut_count

speaker_change_cut_count

share_of_cap_cuts_preempted_by_valid_speaker_change

same-speaker fragmentation

stable-speech harmful split

mixed-speaker turn contamination, measurable하면

turn-duration distribution

p95 / p99 logical turn duration

pathological long-turn count
```

boundary reason은 최소 다음을 구분한다.

```text
silence
speaker_change
max_duration
```

speaker-change logical finalize가 발생하면 product-policy replay에서 **새 logical turn의 cap age를 boundary에서 다시 시작하는 정책**을 별도 명시한다.

current product branch의 실제 semantics가 다르면 owning branch semantics를 우선하고 차이를 기록한다.

---

# 33. Cap 정책 결정은 이번 benchmark에서 자동화하지 않는다

결과는 다음 선택지를 비교할 근거를 제공한다.

```text
cap 유지
cap을 hard safety fallback으로 축소
soft cap 완화
speaker-change 이후 timer reset
특정 조건에서 cap 제거
전체 cap 제거
```

어느 방향도 사전 결론으로 고정하지 않는다.

---

# 34. Result tables

## 34.1 Teacher / event headline

| Backend | View | Change F1 ±250 | Change F1 ±500 | False/h | Overlap F1 | avail p50 | avail p95 | RTF |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Sortformer F32 Vulkan LL | A | | | | | | | |
| Sortformer F32 Vulkan LL | B500 | | | | | | | |
| diart L0.5 | A | | | | | | | |
| diart L0.5 | B500 | | | | | | | |
| diart L1.0 | A | | | | | | | |
| diart L1.0 | B500 | | | | | | | |

## 34.2 Teacher information capability

| Backend | Raw probability/posterior | Embedding evidence | Candidate ceiling | Re-eventizable without inference |
|---|---|---|---:|---|
| Sortformer F32 | yes | optional if dumpable | | yes |
| diart L0.5 | capability-dependent | capability-dependent | | yes where artifacts permit |
| diart L1.0 | capability-dependent | capability-dependent | | yes where artifacts permit |

## 34.3 Cap replay

별도 표로 보고한다.

primary reference/KD characterization table에 섞지 않는다.

---

# 35. 가장 중요한 그래프

하나의 single score ranking보다 frontier를 우선한다.

권장:

```text
Recall@250
   ↑
   │       curves per backend/view
   │
   └────────────────────────→ false events / hour
```

각 operating point에:

```text
observed p50/p95 availability delay
```

를 annotation하거나 별도 latency-frontier plot으로 연결한다.

추가로:

```text
teacher evidence / candidate ceiling
vs
final canonical event curve
```

를 보여주면 decoder/filtering information loss를 확인할 수 있다.

## 35.1 Accuracy vs observed detection delay

가능하면 최종 report에 다음 그래프도 하나 만든다.

```text
Change F1
   ↑
   │                           ● Sortformer B
   │
   │                  ● diart B L1.0
   │
   │            ● diart B L0.5
   │
   └────────────────────────────────────→ observed p95 detection delay
```

핵심은 configured latency 숫자가 아니라:

```text
실제 accuracy
vs
실제 observed detection delay
```

다.

---

# 36. 구현 구조

새 고립형 canonical benchmark directory를 만들지 않는다.

공통 코드는:

```text
experiments/speaker_turn_boundary/
```

의 canonical timeline / event / GT / matching contract를 import한다.

실험-specific code는 기존 speaker experiment family 안에 두되, 최소한 다음 역할을 분리한다.

```text
adapters/
  sortformer_transcribe_f32
  diart

normalization/
  active_speaker_timeline

eventization/
  common_eventizer

policies/
  A
  B500
  cap_replay

reports/
  teacher_evidence
  canonical_events
  product_policy
```

정확한 최종 directory name은 implementation 시작 시 repository convention에 맞춰 정해도 된다.

중요한 것은:

> `speaker_turn_boundary`와 다른 canonical timestamp/event/matching 정의를 새로 만들지 않는 것이다.

---

# 37. Artifact storage

대형 model/output tensor는 Git에 넣지 않는다.

권장:

```text
external cache
  models/
  raw_predictions/
  probability_tensors/
  optional_embeddings/
  runtime_logs/

Git
  configs/
  manifests/
  hashes/
  compact reports/
  experiment plan/
  reproducibility receipts/
```

각 raw artifact는 manifest에서 hash와 provenance를 가리킨다.

---

# 38. 실행 순서

TC는 PSEM의 Phase 번호와 독립된 workstream으로 운영한다.

## TC-A — Protocol / harness extension

1. 기존 `speaker_turn_boundary` canonical contract를 import한다.
2. GT normalizer를 확장한다.
3. A projection을 구현한다.
4. `B500 = 500 configured / 512 effective / no cap`을 구현한다.
5. canonical event taxonomy와 PSEM mapping을 구현한다.
6. teacher evidence / canonical event / policy projection schema를 분리한다.
7. deterministic synthetic / fixture test로 semantics를 검증한다.

## TC-B — Runtime adapters

1. `transcribe.cpp` commit/revision을 freeze한다.
2. Sortformer v2.1 F32 GGUF identity를 freeze한다.
3. Vulkan backend/device receipt를 만든다.
4. Low-Latency preset을 freeze한다.
5. raw probability / segment extraction을 검증한다.
6. diart runtime 및 model revisions를 freeze한다.
7. diart raw artifact capability를 확인한다.

## TC-C — Data acquisition

공개 데이터 corpus 선정은 별도 세션 결과를 따른다.

그 세션에서 정한:

```text
TC-REGRESSION
TC-DEV
TC-EVAL
```

manifest를 freeze한다. 별도 `PSEM-FINAL`은 TC 선택 과정에서 소비하지 않는다.

## TC-D — Native inference

```text
Sortformer F32 Vulkan LL → 각 파일 1회
diart L0.5              → 각 파일 1회
diart L1.0              → 각 파일 1회
```

raw artifacts를 삭제하지 않는다.

## TC-E — Teacher / event characterization

각 raw output에서:

```text
A
B500
```

을 생성한다.

full frontier, event strata, overlap, latency, error taxonomy, information-content를 계산한다.

결론은 최소 두 role로 분리한다.

```text
reference suitability
KD teacher suitability
```

## TC-F — Secondary cap-policy replay

teacher/reference 선택 결과와 별개로:

```text
CAP-NONE
CAP-CURRENT
CAP-LEGACY-7
CAP-SCD-ASSISTED
```

를 같은 raw evidence에서 replay한다.

current 5~7 policy semantics가 아직 freeze되지 않았다면 TC-F만 보류할 수 있다. TC의 primary reference/KD characterization은 완료 가능하다.

# 39. Reference / KD teacher 선택 원칙

단일 F1로 하나의 “winner”를 고르지 않는다. TC는 두 개의 서로 다른 역할을 판단한다.

## 39.1 Reference suitability

PSEM과 동일한 task projection에서 비교 기준으로 사용할 만한가를 본다.

```text
canonical event frontier
hard-negative behavior
overlap quality
A/B500 robustness
latency / reproducibility
```

reference는 student가 반드시 모방해야 하는 teacher를 뜻하지 않는다.

## 39.2 KD teacher suitability

student supervision source로 유용한지를 별도로 본다.

```text
raw soft evidence information-content
GT와의 정렬
confidence / uncertainty 보존
PSEM change/overlap semantics로 안정적으로 변환 가능한가
student가 재현 가능한 horizon의 정보인가
domain sensitivity
license / provenance / reproducibility
```

따라서 가능한 결론은 다음을 모두 허용한다.

```text
reference = Sortformer, KD teacher = Sortformer
reference = diart,      KD teacher = Sortformer
reference = Sortformer, KD teacher = none
reference = diart,      KD teacher = none
reference = backend A,  KD teacher = backend B / combined target
```

### Outcome A — Sortformer F32가 reference와 KD teacher 모두 강함

raw soft evidence와 PSEM-compatible target이 모두 안정적이면 두 역할을 함께 맡길 수 있다.

### Outcome B — Sortformer hard event는 약하지만 soft evidence는 유용

reference로는 약해도 KD teacher로는 가치가 있을 수 있다.

### Outcome C — diart가 local task reference로 더 적합

A/B500의 local event problem에서 더 안정적이라면 reference 역할을 diart가 맡을 수 있다. 이것이 곧 diart embedding/cluster state를 student가 복제해야 한다는 뜻은 아니다.

특히 **A에서는 Sortformer가 뚜렷하게 우세하지만 B500에서 diart와의 격차가 크게 줄어드는 패턴**이 나오면, 다음 가설을 별도로 기록한다.

> PSEM의 실제 local speaker-event 문제에서는 Sortformer의 long-horizon diarization memory 전체를 복제할 필요가 없을 수 있다.

반대로 A와 B500 모두에서 격차가 유지되면, VAD-bounded local view만으로는 설명되지 않는 continuity information이 teacher 성능에 기여하는지 추가 분석한다.

## 39.4 A와 B의 차이가 의미하는 것

A와 B를 둘 다 재는 이유는 단순히 점수를 많이 만들기 위해서가 아니다.

예를 들어:

```text
DIART A Change F1     = 0.86
DIART B500 Change F1  = 0.94
```

라면 다음 의미가 있다.

> diart가 전체 세션의 장기적인 speaker continuity를 완벽하게 추적하지 못해도,
> PuriPuly VAD가 긴 침묵에서 turn을 미리 끊어주면
> 제품에서 실제 필요한 local speaker-change 문제는 훨씬 잘 풀 수 있다.

이 결과는 향후 tiny student 모델의 scope를 줄일 수 있는 근거가 된다.

## 39.5 각 backend view의 의미

### Sortformer B의 의미

Sortformer-B는 현재 teacher의 "제품 문제에서의 상한"을 보는 실험이다.

Sortformer가 full diarization에서 갖는 능력을 그대로 학생에게 요구하지 않는다.

우리가 실제로 알고 싶은 것은:

```text
PuriPuly VAD policy가 적용된 상태에서
Sortformer가 speaker_change / overlap을 얼마나 잘 제공하는가?
```

이 값이 학생 모델의 realistic target이 된다.

### diart A의 의미

diart-A는:

```text
VAD 도움 없이
diart + segmentation-3.0 + embedding + clustering 자체가
speaker continuity와 overlap을 얼마나 잘 처리하는가
```

를 보여준다.

Sortformer A와 비교 가능한 diagnostic이다.

### diart B의 의미

diart-B는 제품 관점에서 더 중요하다.

질문:

```text
PuriPuly VAD가 긴 gap을 처리해준다면
diart가 남은 local speaker-change 문제를 얼마나 잘 처리하는가?
```

만약 diart-B가 Sortformer-B에 매우 근접한다면:

- 작은 local speaker-event model이 충분할 가능성이 높아짐
- full Sortformer 구조를 학생이 복제할 필요가 없음
- teacher distillation target도 change/overlap 중심으로 더 단순화 가능

## 39.6 성능 패턴 해석 가이드

### Case 1 — Sortformer B 매우 높음, diart B 많이 낮음

- Sortformer의 streaming speaker tracking이 실제 제품 문제에서도 강한 가치가 있음
- 학생 모델 distillation teacher로 Sortformer를 사용하는 이유가 강해짐

### Case 2 — Sortformer A > diart A, Sortformer B ≈ diart B

매우 중요한 결과다.

- long-term/full diarization은 Sortformer가 훨씬 강함
- 하지만 PuriPuly VAD가 문제를 줄여주면 두 시스템 차이가 작아짐
- 학생 모델은 full Sortformer를 흉내낼 필요가 없음
- 작은 local speaker-event model 전략이 강하게 지지됨

### Case 3 — diart가 long gap에서 약한 패턴

- 긴 gap의 speaker continuity가 diart의 약점
- 500 ms VAD 정책이 speaker model 부담을 크게 줄여줌
- PuriPuly low-latency VAD 모드가 speaker-change 안정성에도 유리할 가능성

### Case 4 — 0.5~1초 gap continuity를 backend가 잘 처리하는 패턴

- backend가 0.5~1초 gap continuity를 충분히 잘 처리함
- VAD hangover 선택을 speaker model 성능보다 STT/turn UX 기준으로 결정할 여지가 큼

### Case 5 — A와 B 모두 false change rate가 높은 패턴

- 단순 F1보다 state stabilization이 중요
- 학생 모델 학습에서 false-change hard negative와 temporal smoothing이 핵심이 될 수 있음

### Outcome D — reference는 가능하지만 KD가 불필요

GT-only student가 충분하거나 teacher target이 불안정하면 reference만 유지하고 KD는 제거한다.

### Outcome E — 둘 다 teacher 정보가 빈약

teacher 기반 전략을 재검토하고 GT-only / 다른 representation 후보를 우선한다.

# 40. 이번 benchmark가 PSEM 단계에 넘길 것

최종 deliverable은 단순 leaderboard가 아니다.

최소 다음을 freeze한다.

```text
performance/reference role
KD teacher role (없을 수도 있음)

canonical TeacherTargetBundle:
  soft_change
  soft_overlap
  confidence
  valid_mask / uncertainty metadata
  provenance

PSEM change semantics:
  handoff_confirmed와 정렬

PSEM overlap semantics:
  binary simultaneous-speech overlap과 정렬

student가 필요한 continuity horizon
필수 hard-negative 종류
realistic evidence latency target
어떤 full-diarization 능력을 버려도 되는가
```

중요한 제한:

- `new_speaker_onset` 같은 richer TC diagnostic을 PSEM `P(speaker_change)`에 그대로 넣지 않는다.
- Sortformer의 4 slot, diart의 cluster ID, teacher internal embedding을 PSEM output contract로 강제하지 않는다.
- PSEM speaker embedding은 기본적으로 GT speaker identity supervision으로 학습한다.
- overlap/embedding을 최종 제품에서 어떻게 소비할지는 이 benchmark가 결정하지 않는다.

## 40.1 학생 모델 학습 계획에 이 실험이 주는 정보

이번 benchmark를 끝내면 다음을 결정할 수 있다.

1. Sortformer-B500가 실제 teacher ceiling으로 얼마나 높은가
2. full diarization 능력 중 PuriPuly에 실제 필요한 부분이 얼마나 되는가
3. B500 VAD 정책이 speaker-change task를 얼마나 줄여주는가
4. diart의 A→B500 성능 변화가 얼마나 큰가
5. overlap yes/no가 충분한지
6. 학생 모델이 장기 global speaker tracking을 배워야 하는지
7. distillation에서 어떤 teacher target에 집중해야 하는지
8. 학생 모델 목표 latency와 false-change budget을 어느 수준으로 잡을지
9. 최종 학생 모델에서 어느 정도 compute를 지불할 가치가 있는지

# 41. 이번 benchmark의 최종 질문

### Q1

```text
transcribe.cpp F32/Vulkan Streaming Sortformer v2.1은
A와 cap-free B500에서
어떤 speaker-change / overlap information frontier를 제공하는가?
```

### Q2

```text
diart L0.5 / L1.0은
A와 cap-free B500에서
Sortformer F32와 어떤 tradeoff 차이를 보이는가?
```

### Q3

```text
hard event 결과가 나쁠 때에도
raw soft evidence는 tiny student를 가르칠 만큼 유용한가?
```

### Q4

```text
PuriPuly가 실제 필요로 하는 것은
장기 full speaker tracking인가,
아니면 local change / overlap evidence인가?
```

### Q5

```text
speaker-change evidence를 넣으면
현재 5~7 s bounded-turn cap을
유지 / 완화 / fallback화 / 제거할 근거가 생기는가?
```

### Q6

```text
동일한 diart가 PuriPuly B500 정책의 도움을 받으면(B)
Sortformer-B500에 얼마나 가까워지는가?
```

---

# 42. 실행 전 체크리스트

## Protocol

- [ ] canonical 16 kHz source timeline이 기존 `speaker_turn_boundary` contract와 동일하다.
- [ ] teacher evidence / canonical event / product-policy layer가 분리되어 있다.
- [ ] A semantics가 freeze되었다.
- [ ] B500 configured hangover = 500 ms다.
- [ ] B500 effective product-equivalent hangover = 512 ms다.
- [ ] **B500에 max-turn cap이 없다.**
- [ ] overlap onset을 hard product cut 정답으로 강제하지 않는다.
- [ ] one-to-one event matching이 freeze되었다.
- [ ] false-event reference points가 pass/fail gate가 아님을 config/report에 명시했다.

## Sortformer

- [ ] runtime은 `transcribe.cpp`로 freeze되었다.
- [ ] model은 v2.1 F32 GGUF로 freeze되었다.
- [ ] backend는 Vulkan으로 freeze되었다.
- [ ] CUDA dependency가 없다.
- [ ] low-latency preset identity가 freeze되었다.
- [ ] Vulkan device/backend identity가 기록된다.
- [ ] CPU silent fallback 검사가 있다.
- [ ] T×4 raw probability를 보존한다.
- [ ] native segments를 보존한다.
- [ ] observed audio frontier와 compute timing을 보존한다.

## diart

- [ ] L0.5 / L1.0 stock config가 freeze되었다.
- [ ] model/runtime revisions가 기록된다.
- [ ] incremental Annotation을 보존한다.
- [ ] soft posterior / embedding / clustering evidence의 export capability를 preflight에서 기록한다.
- [ ] DEV tuning과 frozen evaluation이 분리된다.

## Data

- [ ] exact corpus selection은 별도 data-design 결정으로 추적된다.
- [ ] regression / DEV / frozen evaluation role이 분리된다.
- [ ] source/session 단위 split 원칙이 있다.
- [ ] 주요 event strata coverage가 manifest에 기록된다.
- [ ] language/environment metadata가 가능한 범위에서 기록된다.
- [ ] known/possible training overlap을 기록한다.

## Cap replay

- [ ] primary B500 결과와 cap 결과를 섞지 않는다.
- [ ] current 5~7 policy의 owning branch / SHA / semantics receipt가 있다.
- [ ] legacy 7 s hard cap은 diagnostic으로만 표시한다.
- [ ] boundary reason이 silence / speaker_change / max_duration으로 분리된다.

## Reproducibility

- [ ] raw prediction은 삭제하지 않는다.
- [ ] large artifact는 external cache + hash manifest로 관리한다.
- [ ] TEST/frozen evaluation 결과를 보고 parameter를 바꾸지 않는다.
- [ ] report에서 R8/R9 Q8 결과와 새 F32 결과를 같은 model arm으로 오인하지 않는다.

## v3에서 계승한 공통 평가 조건

- [ ] 모든 WAV가 동일한 16 kHz mono 조건으로 normalize되었다.
- [ ] GT speaker annotation이 overlap을 보존한다.
- [ ] pyannote/segmentation-3.0과 pyannote/embedding 접근 권한이 준비되었다.
- [ ] A에서는 silence가 speaker state를 reset하지 않는다.
- [ ] B에서는 oracle VAD만 state를 reset한다.
- [ ] overlap 중에는 last reliable solo speaker를 freeze한다.
- [ ] 3+ speaker overlap은 binary overlap으로 collapse한다.
- [ ] ±250 ms / ±500 ms Change F1을 모두 계산한다.
- [ ] false changes/min을 계산한다.
- [ ] streaming `audio_consumed_until`을 저장한다.
- [ ] compute latency를 input-buffer latency와 별도로 측정한다.

## 실험 중 하지 말아야 할 것

### 금지 1: diart 내부 구조를 Sortformer와 똑같이 만들기

하지 않는다.

```text
diart step = 480 ms because Sortformer chunk=480 ms
```

같은 강제 설정은 primary 실험에서 사용하지 않는다.

### 금지 2: B에서 실제 VAD 사용

이번 B에서는 actual Silero VAD를 사용하지 않는다.

그것은 후속 C benchmark다.

### 금지 3: backend prediction silence로 B state reset

B의 state reset은 oracle-policy VAD만 결정한다.

### 금지 4: TEST tuning

결과를 본 뒤 threshold/tau/rho/delta를 수정하지 않는다.

### 금지 5: DER만 보고 결론

DER은 참고값으로 기록할 수 있지만 headline KPI는 아니다.

우리가 만드는 학생 모델은 full diarization이 아니기 때문이다.

### 금지 6: overlap identity를 과도하게 활용

diart가 A+B의 identity를 더 자세히 제공하더라도 v0 benchmark는:

```text
overlap true/false
overlap 후 solo speaker continuity
```

까지만 사용한다.

### 금지 7: final RTTM만 저장

streaming availability timestamp를 잃으면 detection latency를 제대로 비교할 수 없다.

raw streaming output을 보존한다.

---

# 43. 기존 v3에서 폐기되는 문장 / 가정

다음 가정은 v5에서 더 이상 유효하지 않다.

```text
"B500 includes a legacy 7 s max-turn"
→ 폐기

"현재 branch의 7 s cap이 현재 제품 정책을 대표한다"
→ 폐기

"Sortformer primary run은 NeMo/CUDA를 전제로 한다"
→ 폐기

"R8/R9 Q8 output을 새 distillation teacher output으로 그대로 사용한다"
→ 폐기

"Sortformer는 B500만 보고 A는 기존 결과가 있으면 참고한다"
→ 폐기

"diart 2 inference가 6 evaluation views를 만든다"
→ 폐기

"새 speaker_benchmark canonical stack을 별도로 만든다"
→ 폐기

"20 FE/h 등의 operating point가 절대 continuation gate다"
→ 폐기

"Teacher characterization을 PSEM Phase 0이라고 부른다"
→ 폐기; 별도 TC workstream으로 운영

"가장 좋은 reference와 KD teacher는 반드시 같은 backend다"
→ 폐기

"TC에서 관찰한 richer overlap identity evidence는 PSEM v0가 모두 출력해야 한다"
→ 폐기
```

---

# 44. Source / repository references

## PuriPuly canonical experiment infrastructure

```text
experiments/speaker_turn_boundary/
```

canonical timeline, boundary event, detector progress, GT transition, VAD/coalescing/matching semantics의 기반으로 사용한다.

## Existing Sortformer evidence

```text
experiments/speaker_representation_scd/
  R8_STREAMING_SORTFORMER_FEASIBILITY_EXPERIMENT_PLAN.en.md
  R9_SORTFORMER_CHANGE_VERIFICATION_UPPER_BOUND_EXPERIMENT_PLAN.en.md
```

R8/R9은 Q8 historical/preliminary evidence다.

## Product-level speaker-change semantics

```text
.agents/specs/prd/
  bounded_turn_episode_speaker_change_fusion_experiment_review_gated.md
```

clean/gap hard boundary와 interruption/overlap soft-marker를 구분하는 product semantics를 따른다.

## External systems

```text
handy-computer/transcribe.cpp
NVIDIA Streaming Sortformer 4spk v2.1
diart
pyannote segmentation / embedding
```

실행 시 정확한 source/model revisions와 artifact hashes를 freeze한다.

## PuriPuly repository

- PuriPuly Heart README  
  `https://github.com/kapitalismho/PuriPuly-heart/blob/main/README.md`

- VAD defaults  
  `https://github.com/kapitalismho/PuriPuly-heart/blob/main/src/puripuly_heart/config/vad_defaults.py`

- VAD gating implementation  
  `https://github.com/kapitalismho/PuriPuly-heart/blob/main/src/puripuly_heart/core/vad/gating.py`

- Peer translation channel  
  `https://github.com/kapitalismho/PuriPuly-heart/blob/main/src/puripuly_heart/core/orchestrator/peer_translation_channel.py`

## Sortformer

- NVIDIA Streaming Sortformer 4spk v2.1 model card  
  `https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1`

Current official low-latency table at the time the v3 plan was written:

```text
latency             1.04 s
chunk                6 × 80 ms
right context        7 × 80 ms
FIFO                 188 frames
update period        144 frames
speaker cache        188 frames
```

The model card defines this latency as input-buffer latency and explicitly excludes compute time.

## diart

- Official repository  
  `https://github.com/juanmc2005/diart`

- Stable SpeakerDiarizationConfig documentation  
  `https://diart.readthedocs.io/en/stable/autoapi/diart/blocks/diarization/`

The stable API exposes:

```text
duration
step
latency
tau_active
rho_update
delta_new
```

and the default configuration uses a 5 s duration and 0.5 s step.

The original diart approach explicitly studies adjustable online latency in the 0.5–5 s range.

## pyannote segmentation-3.0

- Model card  
  `https://huggingface.co/pyannote/segmentation-3.0`

The model is an overlap-aware powerset speaker segmentation model.  
The model card describes up to 3 local speakers per chunk and up to 2 simultaneous speakers per frame.  
For this PuriPuly benchmark, exact speaker count is not used; all `>=2` active speakers are collapsed to binary overlap.

---

# 45. 최종 요약

TC — Teacher / Reference Characterization의 구조는 다음과 같다.

```text
                    PUBLIC GT DATA
                          │
             ┌────────────┴────────────┐
             │                         │
             ▼                         ▼
  Sortformer v2.1 F32            diart native
  transcribe.cpp                 L0.5 / L1.0
  Vulkan                         actual backend
  LOW_LATENCY ≈ 1.04 s
             │                         │
             └────────────┬────────────┘
                          ▼
                RAW TEACHER EVIDENCE
                          │
                          ▼
                CANONICAL EVENT LAYER
       richer diagnostic events + PSEM mapping
                          │
                 ┌────────┴────────┐
                 ▼                 ▼
                 A               B500
                                  no cap
                 │                 │
                 └────────┬────────┘
                          ▼
          reference suitability / KD suitability
                          │
                          ▼
              canonical TeacherTargetBundle
                          │
                          ▼
                    PSEM Phase 3 KD
```

핵심 원칙은 다음이다.

1. **TC는 PSEM Phase 0/1과 병렬로 진행한다.** GT-only student 개발은 teacher 선택을 기다리지 않는다.
2. **Sortformer characterization은 F32/Vulkan/LL 약 1.04 s 한 profile만 사용한다.** ULR은 제외한다.
3. **reference 역할과 KD teacher 역할을 분리해서 선택한다.** 둘은 같을 필요가 없다.
4. **TC에서는 teacher가 보존한 richer speaker/overlap 정보를 넓게 측정한다.** 그러나 이것이 PSEM v0 output 요구사항을 늘리지는 않는다.
5. **PSEM v0 `speaker_change`는 handoff-confirmed 의미에 맞춘다.** overlap 안의 new-speaker evidence를 그대로 change target으로 사용하지 않는다.
6. **B500은 500 ms configured / 512 ms effective hangover이며 max-turn cap이 없다.** cap은 secondary replay다.
7. **TC에서 사용한 evaluation set은 teacher 선택 이후 PSEM final test가 아니다.** 별도 `PSEM-FINAL`을 유지한다.
8. **overlap UI, transcript/translation 처리, speaker embedding의 제품 소비 정책은 이 문서의 범위 밖이다.**

이렇게 하면 TC는 “teacher가 무엇을 얼마나 알고 있는가”를 최대한 보존해서 평가하고, PSEM은 그중 실제 v0 task에 필요한 supervision만 안정적으로 받아 사용할 수 있다.

공정성을 확보하는 지점은 backend 내부가 아니라 **공통 evaluation layer**다.

```text
동일 audio
동일 GT
동일 active-speaker representation
동일 Eventizer
동일 oracle PuriPuly VAD policy
동일 event matching
동일 metrics
```

이 결과를 바탕으로 다음 단계의 tiny speaker-event student model이 **full diarization 능력을 얼마나 버려도 되는지**, 그리고 **Sortformer v2.1에서 무엇을 distill해야 하는지**를 결정한다.
