# vcPuriPuly Peer Channel용 초경량 Streaming Speaker Event Model — 설계 및 학습 계획 v2

> **문서 상태:** Final design draft / implementation-oriented research plan  
> **작성 기준일:** 2026-08-13  
> **대상 프로젝트:** PuriPuly-heart  
> **가칭:** **PuriPuly Speaker Event Model (PSEM)**  
> **핵심 목적:** 기존 streaming VAD와 병렬로 동작하면서 full diarization 없이 **speaker change**, **binary overlap**, **short-term speaker representation**을 저지연으로 추정한다.  
> **Upstream dependency:** teacher/reference 관련 결정은 별도 `TC — Teacher / Reference Characterization` 문서의 결과를 import한다.  
> **Downstream scope:** overlap UI, STT/translation turn 처리, embedding 소비 정책 등 product integration은 별도 문서에서 다룬다.

---

## 0. 한 페이지 요약

PSEM v0는 16 kHz mono mixed audio를 입력으로 받아 다음을 출력하는 작은 streaming acoustic model이다.

```text
P(speaker_change)
P(overlap)
64~128d speaker_embedding
(optional) language auxiliary logits
```

PSEM은 VAD나 full diarization을 대체하지 않는다. 기존 VAD와 같은 오디오를 병렬로 보되 neural inference graph는 독립적으로 유지한다.

v0의 overlap 책임은 단순하다.

```text
지금 동시에 2명 이상 말하는가? -> P(overlap)
```

PSEM v0는 overlap 안에서 누가 새로 들어왔는지, 누가 빠졌는지, 몇 번째 speaker인지, A/B/C identity가 무엇인지를 출력하지 않는다. overlap 전후의 reliable single-speaker 상태가 다를 때만 `speaker_change` 의미를 부여한다.

```text
A -> A+B -> A  : overlap yes, speaker change no
A -> A+B -> B  : overlap yes, overlap 종료 뒤 speaker change yes
```

Streaming profile은 **Low-Latency 한 가지로 고정**한다. reference geometry는 80 ms grid에서 `CHUNK=6 + RIGHT_CONTEXT=7`, 약 **1.04 s input buffer**다. v0의 모든 학습·평가는 이 단일 profile로 진행한다.

첫 baseline은 5~10M parameters로 시작한다. 순서는 다음이다.

1. Dataset/label pipeline을 만든다.
2. **GT-only baseline**으로 문제 자체가 풀리는지 확인한다.
3. language auxiliary가 hard-case에 실제 도움이 되는지 실험한다.
4. 별도 TC workstream이 선택한 teacher/target이 준비되면 **optional KD**를 비교한다.
5. KD가 반복적으로 도움이 될 때만 canonical recipe에 남긴다.
6. 성능이 확인된 뒤 5M -&gt; 2M -&gt; 1M 순으로 축소한다.
7. overlap 내부 participant tracking은 v0 이후 연구로 남긴다.

**Canonical training은 teacher 없이도 완전히 재현 가능해야 한다.** Teacher는 optional noisy expert다.

학습 데이터셋 이름은 현재 확정하지 않는다.

- KO: **TBD**
- EN: **TBD**
- JA: **TBD**
- ZH: **TBD**
- multilingual / code-switch: **TBD**

제품에서 overlap을 어떻게 보여줄지, transcript/translation을 어떻게 분할할지, speaker embedding을 어떤 state/context에 사용할지는 이 문서의 모델 성공 조건에 포함하지 않는다.

---

# 1. 프로젝트 맥락

## 1.1 PuriPuly에서 이 모델이 필요한 이유

PuriPuly Peer Channel은 하나의 mixed-audio channel 안에서 여러 사람이 연속하거나 겹쳐 말할 수 있다. 기존 VAD는 speech/non-speech와 utterance lifecycle을 담당하지만, VAD만으로는 다음을 직접 알 수 없다.

```text
같은 VAD-active 구간 안에서 화자가 바뀌었는가?
현재 두 명 이상이 동시에 말하고 있는가?
짧은 gap 전후의 speaker continuity가 같은가?
```

PSEM의 목적은 이 좁은 acoustic speaker-event 문제를 푸는 것이다.

기본 pipeline relation:

```text
16 kHz mono audio
   ├── Existing VAD
   └── PSEM
         ├── P(speaker_change)
         ├── P(overlap)
         └── speaker_embedding
```

PSEM은 full diarization, persistent speaker ID, speaker enrollment를 목표로 하지 않는다.

또한 이 문서는 PSEM output을 실제 UI/STT/translation/context에서 어떻게 소비할지 결정하지 않는다. downstream product integration은 별도 설계 문서의 책임이다.

## 1.2 기존 VAD와의 관계

PuriPuly에는 이미 streaming VAD가 존재한다. PSEM은 speech/non-speech 문제를 다시 학습하는 모델이 아니다.

고정 원칙:

> VAD와 PSEM은 동일 오디오에 대해 **병렬로 독립 실행**한다.  
> PSEM이 VAD implementation을 import하거나 neural graph를 합치지 않는다.

### Implementation snapshot — non-normative context

이 설계를 시작할 때 참조한 PuriPuly VAD/runtime 계열에는 다음 형태의 streaming contract가 존재한다. 이 값들은 **PSEM model contract나 영구 제품 정책이 아니라 implementation grounding**이다. 실제 통합 시점에는 owning branch/revision을 다시 확인한다.

```text
audio sample rate                 16 kHz
reference VAD chunk               512 samples ~= 32 ms
VAD lifecycle events              SpeechStart / SpeechChunk / SpeechEnd
reference start debounce/commit   3 chunks / 3 chunks
known hangover profiles           500 ms / 1000 ms
```

VAD hangover는 application/runtime 설정이며 PSEM training label 자체와 분리한다. 500 ms와 1000 ms 같은 hangover 설정이 존재할 수 있지만, PSEM을 hangover 값별로 다시 학습하지 않는다.

max-turn / bounded-turn 정책 역시 PSEM model contract가 아니다. 해당 정책의 정확한 semantics는 owning implementation에서 다룬다. PSEM primary evaluation의 B500 view에는 max-turn cap을 넣지 않는다.

이 문서에서 필요한 VAD relation은 **model independence와 evaluation projection**까지다. PSEM output으로 실제 STT 세션을 자를지, UI에 overlap을 표시할지 등은 범위 밖이다.

# 2. 최종 요구사항

## 2.1 필수 기능 요구사항

### R1. 화자 전환 검출

다음과 같은 상황에서 `speaker_change`를 검출해야 한다.

```text
A -> B
A -> short silence -> B
A -> very short B backchannel -> A
```

반대로 다음은 false change를 최대한 억제해야 한다.

```text
A -> A
A -> short silence -> A
A(KO) -> A(EN)        # same speaker code-switch
A -> acoustic/channel perturbation -> A
```

### R2. Overlap 검출

v0에서 요구하는 overlap 기능은 **binary overlap**이다.

```text
one speaker     -> overlap = 0
2+ speakers     -> overlap = 1
```

여기서 overlap은 **고정 길이 window 안에 여러 화자가 등장했는가**가 아니라, 시간축의 각 시점에서 **동시에 활성인 화자가 2명 이상인가**를 뜻한다. 2명, 3명, 4명 이상을 서로 다른 class로 구분하지 않는다.

v0에서는 overlap 속 개별 화자 ID attribution은 필수가 아니다.

### R3. Short-term speaker representation

모델은 64~128차원 speaker embedding을 추가로 출력한다.

목적은:

- 같은 speaker의 representation을 가깝게 만들기
- 다른 speaker를 멀게 만들기
- 짧은 silence/hangover 구간을 넘어 short-term speaker continuity를 보조하기

이다.

이 embedding은 **global speaker ID를 제공하는 것이 아니다.**

### R4. Multilingual robustness

최소 다음 언어를 주요 deployment domain으로 다룬다.

- Korean (KO)
- English (EN)
- Japanese (JA)
- Chinese / Mandarin-centered ZH

실사용에서는 **서로 다른 화자가 서로 다른 언어로 대화하는 경우가 흔할 것**을 전제로 한다.

동시에 다음도 반드시 정상 동작해야 한다.

```text
A(KO) -> B(KO)      # 언어가 같지만 speaker change
A(KO) -> B(EN)      # 언어와 speaker 모두 change
A(KO) -> A(EN)      # language change only, speaker change 아님
```

### R5. Existing VAD와 독립

VAD hangover를 500 ms, 700 ms, 1000 ms 등으로 바꾸어도 speaker model을 재학습하지 않아야 한다.

### R6. Local/offline runtime

PuriPuly의 무료 오픈소스 앱에 포함될 수 있도록 local inference가 가능해야 한다.

### R7. Open-source release 가능성

- inference code 공개
- training code 공개
- preprocessing/evaluation 공개
- architecture/config 공개
- checkpoint 공개 가능성을 목표
- canonical recipe가 commercial API나 closed service에 의존하지 않아야 함

---

## 2.2 Latency 요구사항

PSEM v0는 **Low-Latency 단일 streaming profile**로 고정한다. 완전 causal을 목표로 하지 않으며 look-ahead를 허용한다.

reference geometry는 Streaming Sortformer v2.1 Low-Latency characterization과 맞춘다.

```text
reference frame grid = 80 ms
current chunk         = 6 frames = 480 ms
right context         = 7 frames = 560 ms
reference input buffer = 1040 ms
```

PSEM 내부 log-Mel frame은 10/20 ms처럼 더 촘촘할 수 있다. 중요한 것은 **모델이 사용할 수 있는 future context와 output timestamp contract를 LL 조건에 맞추는 것**이다.

최종 latency 평가는 1.04 s라는 buffer 숫자 하나로 끝내지 않는다.

```text
reference boundary
      |
      v
model evidence available
      |
      v
model/event emitted
```

반드시 분리해서 측정한다.

- reference boundary 위치
- consumed/source-audio frontier
- feature extraction / inference compute
- event post-processing latency
- p50 / p90 / p95 evidence/event latency

speaker-change latency의 reference는 label semantics를 따른다.

- non-overlap `A -> B`: B speech onset
- `A -> A+B -> B`: overlap 종료 뒤 B-only state가 확인되는 handoff boundary

Teacher가 더 긴 internal memory/cache를 갖더라도 동일 1.04 s input buffer가 동일 memory capacity를 뜻하지는 않는다. KD target은 student가 물리적으로 재현하기 어려운 장기 identity를 요구하지 않아야 한다.

## 2.3 모델 크기 / compute 요구사항

초기부터 극단적으로 작은 모델을 만들지 않는다.

### Baseline

```text
5~10M parameters
```

목적은 **문제가 실제로 풀리는지 먼저 검증**하는 것이다.

### Compression sweep

```text
5~10M
  |
  v
 ~5M
  |
  v
 ~2M
  |
  v
 ~1M
```

Sortformer v2.1 계열이 약 117M parameters인 것을 기준으로 하면 5~10M만 되어도 이미 parameter count가 매우 크게 줄어든다.

최종 목표 크기는 성능을 본 뒤 정한다.

**v0 release 목표:** `<=5M`이 이상적이나 hard requirement로 고정하지 않는다.  
**stretch goal:** `1~2M + INT8`에서도 성능 유지.

---

# 3. 명시적인 비목표(Non-goals)

이 프로젝트가 실패하지 않으려면 아래 문제를 일부러 풀지 않아야 한다.

## v0에서 하지 않는 것

- full speaker diarization
- 30분/1시간 동안 지속되는 global speaker identity
- arbitrary number of speaker clustering
- 4/8-speaker stable output slot
- permutation-invariant diarization decoding
- 3명 이상 overlap speaker attribution
- overlap 안에서 word/token별 speaker attribution
- source separation
- multi-speaker ASR
- "메인 화자" 자동 판별
- 사전 speaker enrollment

특히 **메인 화자 / 나머지 화자**를 neural model의 class로 만들지 않는다.

PuriPuly의 Peer Channel에는 사전에 등록된 주 화자가 없고, 불특정 다수와 연속적으로 대화한다. 따라서 acoustic model이 "MAIN"의 의미를 알 수 없다.

---

# 4. 왜 full diarization이 아니라 speaker event detection인가

PuriPuly가 필요한 문제는 full diarization보다 훨씬 좁다.

Full diarization은 보통 다음을 함께 해결해야 한다.

- 전체 스트림의 speaker count 추정
- speaker slot/identity의 지속적인 유지
- 장기 speaker re-identification
- overlap 구간의 multi-speaker activity assignment
- permutation / clustering / speaker memory 관리

반면 PuriPuly v0의 질문은 다음 두 가지가 핵심이다.

```text
"지금 말하던 사람이 바뀌었는가?"
"지금 두 명 이상이 겹쳐 말하고 있는가?"
```

따라서 full diarizer를 축소하는 것보다 **문제 정의 자체를 speaker event detection으로 제한**한다.

이렇게 하면:

- global speaker ID가 필요 없고,
- arbitrary speaker clustering이 필요 없고,
- 장시간 speaker cache가 필수가 아니며,
- 기존 VAD와 독립적으로 병렬 실행할 수 있고,
- 실제 제품에서 필요한 change/overlap event에 capacity를 집중할 수 있다.

Speaker embedding 역시 장기적인 speaker fingerprint를 보장하기 위한 것이 아니라, **짧은 gap/hangover와 local continuity에서 같은 사람인지 판단하는 auxiliary representation**으로 먼저 사용한다.

# 5. 최종 v0 모델 아키텍처

## 5.1 전체 구조

```text
16 kHz mono audio
       |
       v
Log-Mel frontend (40~64 bins)
       |
       v
Lightweight streaming temporal encoder
(Causal Conv / TCN + GRU 우선)
       |
       +----------------+----------------+----------------+
       |                |                |                |
       v                v                v                v
 Change Head       Overlap Head     Speaker Head     Language Head
 P(change)         P(overlap)       64~128d emb      KO/EN/JA/ZH/...
```

### Shared encoder 권장 방향

첫 baseline에서는 다음 계열을 우선한다.

```text
log-Mel
  -> causal convolution stem
  -> depthwise/temporal convolution blocks
  -> 1~2 layer GRU
  -> shared hidden state
```

이유:

- streaming state 구현이 단순함
- Transformer/KV-cache보다 runtime 관리가 단순함
- CPU inference 최적화가 비교적 쉬움
- 5~10M 이하 모델 구성에 유리함
- look-ahead가 필요한 경우 입력 framing에서 명시적으로 제공 가능

Conformer/Transformer는 baseline이 안 풀릴 때 두 번째 선택지로 둔다.

---

## 5.2 Change Head

출력:

```text
p_change[t] in [0, 1]
```

의미:

> frame `t` 주변의 **명확한 single-speaker 상태**에서, 현재 추적되는 화자가 이전의 single-speaker 상태와 다른 화자로 전환되었을 확률.

v0에서 `speaker_change`는 **새 화자가 처음 등장한 순간**과 동일한 개념이 아니다. 특히 overlap 중에는 여러 화자가 동시에 활성일 수 있으므로 speaker handoff를 확정하지 않는다.

중요:

- 이 head가 primary product output이다.
- speaker ID를 출력하지 않는다.
- arrival-order speaker slot을 출력하지 않는다.
- global clustering을 하지 않는다.
- overlap 구간에서는 change decision과 change supervision을 기본적으로 defer/mask한다.
- overlap이 끝나 다시 single-speaker 상태가 되었을 때, overlap 전 single speaker와 이후 single speaker가 같은지/다른지를 기준으로 `speaker_change`를 확정한다.

---

## 5.3 Overlap Head

출력:

```text
p_overlap[t] in [0, 1]
```

의미:

> 시간축의 시점 `t`에서 동시에 두 명 이상의 speech activity가 존재할 확률.

이 값은 "최근 window 안에 서로 다른 화자가 몇 명 등장했는가"가 아니다. 예를 들어 A가 말한 뒤 B가 순차적으로 말했지만 동시에 발화하지 않았다면 같은 input buffer 안에 둘이 모두 존재해도 `overlap=0`이다.

v0에서는 `speaker_count=2/3/4`를 예측하지 않는다. 2명 이상이면 모두 동일한 binary overlap 상태로 취급한다.

---

## 5.4 Speaker Embedding Head

출력 예시:

```text
speaker_embedding[t] = 128-dimensional normalized vector
```

학습 목표:

```text
same speaker     -> embedding distance small
different speaker -> embedding distance large
```

speaker embedding을 처음부터 넣는 이유는 단순 `CHANGE/NO_CHANGE` classifier보다 encoder가 speaker identity에 민감한 representation을 형성하도록 유도하기 위해서다.

다만 embedding을 제품의 hard dependency로 두지 않는다.

v0 product event는 우선 `p_change`와 `p_overlap`만으로도 동작 가능해야 한다.

---

## 5.5 Language Auxiliary Head

### 목적

다국어 대화에서는 다음 상황이 흔하다.

```text
A(KO) -> B(EN)
```

여기서는 speaker acoustics뿐 아니라 linguistic characteristics도 크게 변하므로 speaker transition에 유용한 보조 representation을 제공할 수 있다.

하지만 다음 상황도 반드시 존재한다.

```text
A(KO) -> A(EN)
```

즉 language switch와 speaker switch는 동일하지 않다.

### v0 원칙

**Language head는 auxiliary task로만 둔다.**

```text
Shared Encoder
  +-> Change Head
  +-> Overlap Head
  +-> Speaker Head
  +-> Language Head
```

v0에서는 아래처럼 language logits를 Change Head에 직접 concatenate하는 것을 기본안으로 하지 않는다.

```text
# v0에서 기본적으로 피함
language_logits ----+
                    +--> change
speaker_features ---+
```

이유는 dataset shortcut 위험 때문이다.

### 언어 class

최소:

```text
KO
EN
JA
ZH
OTHER / UNKNOWN
```

`MIXED`는 데이터 label 품질이 충분한 경우에만 별도 class로 둔다.

### 출시 판단

language auxiliary head는 **ablation 필수**다.

```text
Model-A: no language head
Model-B: language auxiliary head
```

speaker change generalization에 실질적인 이득이 없거나 code-switch false positive가 증가하면 release model에서 제거한다.

---

# 6. VAD와의 pipeline 관계

## 6.1 두 모델은 병렬로 실행

```text
                      +------------------+
audio frames -------->| Existing VAD     |
      |               +------------------+
      |
      |               +------------------+
      +-------------->| PSEM             |
                      +------------------+
```

둘은 같은 source audio timeline을 보지만 neural inference graph와 owner는 독립이다.

PSEM frame output의 최소 contract:

```text
source position / observed frontier
change_prob
overlap_prob
speaker_embedding
optional language logits
```

VAD event와 PSEM output을 최종 application에서 어떻게 조합할지는 별도 integration 정책이다.

## 6.2 VAD hangover와 학습의 분리

VAD hangover는 application runtime 설정이다. PSEM은 특정 500/1000 ms 값에 종속되어 학습하지 않는다.

학습 데이터에는 다양한 gap을 포함해 representation이 hangover 설정에 과적합되지 않도록 한다.

```text
gap range example: 0 ~ 1.2 s
```

왜 다양한 gap을 넣는지의 예:

```text
configured hangover = 500 ms
A --300 ms silence--> B
→ 하나의 VAD-active turn 안에서 speaker change evidence가 필요할 수 있음

configured hangover = 1000 ms
A --800 ms silence--> B
→ 역시 하나의 VAD-active turn 안에서 speaker change evidence가 필요할 수 있음
```

반대로 silence가 충분히 길어 VAD lifecycle이 이미 `SpeechEnd`를 성립시킨 view에서는, 그 경계를 넘어 이전 speaker와의 short-term continuity를 강제할 필요가 없다. 이 설명은 PSEM을 특정 hangover에 맞춰 학습한다는 뜻이 아니다.

PSEM label은 source timeline에서 직접 만든다. 실제 runtime에서 어떤 gap이 하나의 VAD turn 안에 남는지는 downstream projection에서 결정한다.

Primary comparative evaluation에서는 두 view를 사용한다.

```text
A     : VAD turn reset을 강제하지 않는 capability view
B500  : oracle speech activity + configured 500 ms hangover
        512-sample chunk semantics에서는 effective 512 ms
        max-turn cap 없음
```

### A/B500 continuity-state semantics

두 view의 차이가 재현 가능하도록 evaluation-side state reset도 명시한다.

```text
PSEM-A
  silence/VAD policy 때문에 short-term speaker continuity state를 강제로 reset하지 않음
  → 모델의 raw continuity capability를 봄

PSEM-B500
  oracle speech activity + B500 hangover로 SpeechEnd가 성립하면
  evaluation-side short-term continuity state를 reset
  → 다음 SpeechStart는 새 bounded VAD episode로 취급
```

이 reset은 **evaluation projection의 규칙**이지, 실제 제품 Event Manager가 반드시 같은 방식으로 state를 관리해야 한다는 결정이 아니다.

이 문서는 bounded-turn 정책의 제품 채택 여부를 결정하지 않는다.

# 7. Overlap 정책

## 7.1 v0: binary overlap + conservative change semantics

PSEM v0의 overlap 책임은 **binary simultaneous-speech detection**이다.

```text
single speaker  -> overlap = 0
2+ speakers     -> overlap = 1
```

v0는 overlap 안에서 다음을 풀지 않는다.

```text
누가 새로 들어왔는가
누가 빠졌는가
A+B인지 A+C인지
현재 speaker가 overlap 안에 계속 존재하는가
3명 이상일 때 각 participant identity가 무엇인가
```

speaker change는 overlap 내부에서 즉시 확정하지 않는다. 이해하기 쉬운 규칙은 다음과 같다.

```text
single speaker before overlap
        ↓
overlap = YES
        ↓
single speaker after overlap
        ↓
before/after speaker continuity가 같은가?
```

### A -&gt; A+B -&gt; A

```text
overlap = YES
post-overlap speaker = pre-overlap speaker
speaker_change = NO
```

### A -&gt; A+B -&gt; B

```text
overlap = YES
post-overlap speaker != pre-overlap speaker
speaker_change = YES at post-overlap resolution
```

B가 overlap에 처음 들어온 순간 자체를 PSEM v0의 `speaker_change` 정답으로 사용하지 않는다.

### A -&gt; A+B+C+D -&gt; C

2명 이상은 모두 같은 binary overlap state다. overlap 전 reliable single speaker와 overlap 후 reliable single speaker가 다르면 change positive가 될 수 있지만, overlap 내부 B/C/D attribution은 요구하지 않는다.

### A -&gt; A+B+C -&gt; silence

post-overlap reliable single-speaker state가 없으면 handoff를 억지로 확정하지 않는다. change target은 unresolved로 남기고 short-term continuity는 다음 reliable state 또는 VAD lifecycle에서 다시 시작할 수 있다.

이 정책은 모델의 event semantics다. overlap을 사용자에게 어떻게 보여주거나 transcript/translation에서 어떻게 활용할지는 이 문서에서 정하지 않는다.

## 7.2 v1 이후 연구

v0 성공 후 필요하면 다음을 별도 연구할 수 있다.

```text
P(previous/current speaker still active during overlap)
```

이 확장은 full diarization이 아니라 **직전 reliable speaker가 overlap 안에서도 계속 존재하는지**만 묻는 좁은 target-conditioned tracking 문제로 둔다. 가능한 구조 예는 다음과 같다.

```text
current-speaker prototype
          |
          v
audio -> target-conditioned activity head
          |
          v
P(current speaker active)
```

예를 들어 `A -> A+B`에서는 current A가 계속 active일 수 있고, `A+B -> B`에서는 A가 빠졌다는 evidence를 낼 수 있다. 이 head는 v0 필수 output이 아니며 v0 성공 후 별도 ablation으로만 검토한다.

그 다음 단계에서만 2-speaker local attribution을 고려한다.

```text
A -> A+B -> B
slot0 -> slot0+slot1 -> slot1
```

다음은 계속 scope 밖이다.

```text
long-term global A/B/C identity
arbitrary-speaker diarization
overlap word attribution
source separation
```

# 8. Downstream product integration은 별도 문서에서 다룬다

PSEM v0는 다음 model/runtime evidence를 제공하는 데서 책임을 끝낸다.

```text
P(speaker_change)
P(overlap)
speaker_embedding
(optional) language logits
source-timeline metadata
```

이 문서에서는 다음을 결정하지 않는다.

```text
overlap을 자막/UI에서 어떻게 표시할지
speaker_change를 STT session hard split으로 사용할지
transcript를 사후 semantic segmentation할지
LLM translation에서 overlap span을 어떻게 처리할지
speaker_embedding을 context/history routing에 사용할지
몇 개의 speaker-local context를 유지할지
```

이러한 선택은 acoustic model 품질과 독립적인 product/integration policy이며 후속 문서에서 다룬다.

따라서 PSEM의 성공 여부는 downstream UI/LLM policy의 성공 여부와 분리해서 평가한다.

# 9. 데이터 계획

## 9.1 실제 데이터셋 슬롯 — 현재는 이름을 확정하지 않음

이 문서에서는 실제 데이터셋 이름을 확정하지 않는다.


| 영역                                | Dataset | 필요 annotation                | License 검토 | 상태        |
| --------------------------------- | ------- | ---------------------------- | ---------- | --------- |
| Korean conversational / meeting   | **TBD** | speaker segments, overlap 권장 | TBD        | 미선정       |
| English conversational / meeting  | **TBD** | speaker segments, overlap 권장 | TBD        | 미선정       |
| Japanese conversational / meeting | **TBD** | speaker segments, overlap 권장 | TBD        | 미선정       |
| Chinese conversational / meeting  | **TBD** | speaker segments, overlap 권장 | TBD        | 미선정       |
| multilingual / code-switch        | **TBD** | speaker + language boundary  | TBD        | 미선정       |
| product-like VR/social audio      | **TBD** | speaker event annotation     | TBD        | 별도 평가셋 권장 |


### 데이터 선택 원칙

- 실제 conversation이어야 함
- speaker ID annotation이 있어야 함
- 가능한 경우 overlap annotation 포함
- microphone/channel 다양성 포함
- far-field / near-field 모두 포함 가능
- 반드시 speaker-disjoint train/dev/test split 가능 여부 확인
- 라이선스가 모델 공개 정책과 충돌하지 않는지 검토
- 실제 오디오를 repo에 재배포하지 않아도 download/preprocess script로 재현 가능해야 함

---

## 9.2 다국어 데이터에서 가장 중요한 4개 조합

모델이 language를 speaker shortcut으로 쓰지 않도록 최소 다음 네 조합을 의도적으로 맞춘다.


| Speaker   | Language  | 예                 | Label         |
| --------- | --------- | ----------------- | ------------- |
| same      | same      | A(KO) -&gt; A(KO) | NO CHANGE     |
| same      | different | A(KO) -&gt; A(EN) | **NO CHANGE** |
| different | same      | A(KO) -&gt; B(KO) | **CHANGE**    |
| different | different | A(KO) -&gt; B(EN) | CHANGE        |


가장 확보가 어려울 가능성이 큰 것은:

```text
same speaker + different language
```

다.

이 데이터가 부족하면 language auxiliary task가 오히려 speaker change shortcut을 강화할 수 있다.

따라서 실제 dataset 선정 시 **cross-language same-speaker sample availability**를 별도 체크 항목으로 둔다.

---

## 9.3 Teacher-characterization data와 PSEM final test의 분리

Teacher/reference 선택에 사용된 TC evaluation data는 teacher 선택 이후 PSEM 관점에서 development-known으로 취급한다.

```text
TC-REGRESSION / TC-DEV / TC-EVAL
  -> teacher/reference/target 선택용
  -> 이후 PSEM tuning에 참고 가능

PSEM-TRAIN / PSEM-DEV
  -> student architecture / loss / KD / compression 결정

PSEM-FINAL
  -> teacher 선택과 student tuning에 사용하지 않는 별도 frozen confirmatory set
```

구체 corpus 이름과 시간/비율은 별도 data-design에서 확정한다.

# 10. Synthetic conversation generation

실제 meeting data만으로는 원하는 boundary condition을 균형 있게 만들기 어렵다.

따라서 single-speaker utterance를 이용한 synthetic conversation을 적극적으로 만든다.

## 10.1 필수 synthetic pattern

### Direct turn

```text
A -> B
A -> A
```

### Short gap

```text
A -> 100ms silence -> A
A -> 100ms silence -> B
A -> 300ms silence -> A
A -> 300ms silence -> B
A -> 800ms silence -> A
A -> 800ms silence -> B
A -> 1200ms silence -> A/B
```

### Backchannel

```text
A long speech
 -> B: "네" / "응" / "yeah" / "はい" / short acknowledgment
 -> A continues
```

### Overlap

```text
A -> A+B -> A
A -> A+B -> B
A -> A+B -> silence
```

### Multilingual speaker transition

```text
A(KO) -> B(EN)
A(KO) -> B(JA)
A(JA) -> B(ZH)
...
```

### Same-speaker code switch

가능한 실제 same-speaker multilingual source를 이용해:

```text
A(KO) -> A(EN)
A(EN) -> A(JA)
...
```

를 hard negative로 만든다.

---

## 10.2 매우 중요한 anti-shortcut 처리

절대 다음과 같이 단순 concat해서 끝내지 않는다.

```text
A = clean headset
B = noisy far-field
A -> B
```

이렇게 만들면 모델이 speaker change 대신 **channel/background change**를 배울 수 있다.

권장 순서:

```text
1. speaker utterance를 먼저 조합
2. turn/gap/overlap 구조를 만든다
3. 전체 synthetic conversation에 동일한
   - room impulse response
   - noise bed
   - EQ
   - gain profile
   - codec
   를 적용
```

그리고 hard negative를 별도로 만든다.

```text
same speaker + channel/noise change -> NO CHANGE
different speaker + same channel    -> CHANGE
```

실제 product audio에서는 speaker와 위치/마이크 조건이 함께 변할 수 있으므로 **두 종류를 모두 포함**해야 한다.

---

# 11. Label 정의

## 11.1 Speaker change label

### 기본 의미

`speaker_change`는 **이전의 reliable single-speaker 상태와 현재의 reliable single-speaker 상태가 다른 화자인가**를 나타낸다.

새 화자가 처음 음성을 낸 순간과 speaker handoff는 항상 같지 않다. 특히 overlap에서는 “누군가 새로 들어왔다”와 “화자가 실제로 넘어갔다”를 분리한다.

### 일반 A -&gt; B

겹침이 없는 전환에서는 B speech onset을 change reference로 둔다.

```text
AAAAA|BBBBB
     ^
   change
```

### Gap 포함

```text
AAAAA .... BBBBB
          ^
        change at B onset
```

training label은 다양한 gap 길이에 대해 만들 수 있다. 실제 runtime에서 이 change가 같은 VAD turn 안에서 필요한지는 A/B500 같은 evaluation projection이 결정한다.

### Same speaker resume

```text
AAAAA .... AAAAA
```

NO CHANGE.

### Code-switch

```text
A(KO) -> A(EN)
```

NO CHANGE.

### 다른 화자, 같은 언어

```text
A(KO) -> B(KO)
```

CHANGE.

### Overlap: A -&gt; A+B -&gt; A

```text
A only      A+B overlap      A only
            [overlap]
```

- overlap 구간에는 `speaker_change` 정답을 부여하지 않는다.
- overlap이 끝난 뒤 speaker가 다시 A이므로 **NO CHANGE**다.

### Overlap: A -&gt; A+B -&gt; B

```text
A only      A+B overlap      B only
            [overlap]          ^
                            change reference
```

B가 overlap에 처음 들어온 순간은 PSEM v0의 change reference가 아니다. overlap이 끝나 B-only reliable state가 확인되는 경계를 change reference로 둔다.

### 3명 이상 overlap: A -&gt; A+B+C+D -&gt; C

2명 이상은 동일한 binary overlap으로 취급한다. overlap 내부 speaker attribution은 만들지 않는다. overlap 전후 reliable solo speaker가 다르면 post-overlap resolution에 CHANGE를 둔다.

### Overlap -&gt; silence

```text
A -> A+B+C -> silence
```

post-overlap reliable single-speaker 상태가 없으므로 handoff는 unresolved다. overlap 내부에는 change supervision을 주지 않는다.

> 구현상 loss에서 제외할 필요가 있는 구간은 training mask/validity metadata로 표현할 수 있지만, 의미적으로는 “overlap 안의 새 speaker 진입을 PSEM change positive로 가르치지 않는다”가 핵심이다.

## 11.2 Change boundary fuzzy target

speaker boundary annotation은 sample-exact ground truth라고 보기 어렵다.

따라서 하나의 exact frame만 positive로 두기보다 boundary 주변에 soft/fuzzy target을 사용할 수 있다.

예:

```text
center boundary = 1.0
+/- 80ms        = 0.8
+/- 160ms       = 0.5
+/- 240ms       = 0.2
outside         = 0
```

정확한 shape는 dev tuning 대상이다. 평가에서는 별도로 `+/-250 ms`, `+/-500 ms` collar를 사용한다.

## 11.3 Overlap label

Overlap label은 **각 시간 시점의 동시 speech activity count**에서 만든다.

```text
0 active speaker -> ignored or 0 depending training validity
1 active speaker -> 0
2+ active speaker -> 1
```

하나의 1.04 s input buffer 안에 A 다음 B가 순차적으로 존재하더라도 동시 활성 구간이 없으면 overlap label은 0이다. 100~200 ms처럼 짧아도 실제 동시 speech가 있으면 overlap=1이다.

2명/3명/4명 이상을 별도 class로 나누지 않는다.

speaker embedding loss는 기본적으로 clean single-speaker region에 강하게 적용한다.

# 12. 학습 objective

기본 multi-task loss:

```text
L_total
 = lambda_change  * L_change
 + lambda_overlap * L_overlap
 + lambda_spk     * L_speaker
 + lambda_lang    * L_language
```

초기 실험용 예시값:

```text
lambda_change  = 1.0
lambda_overlap = 0.5
lambda_spk     = 0.5
lambda_lang    = 0.1
```

이 값은 고정 규칙이 아니라 starting point다.

## 12.1 Change loss

후보:

- BCE with soft labels
- focal BCE if class imbalance가 큼

핵심은 boundary 주변 fuzzy target과 negative sampling이다.

## 12.2 Overlap loss

- BCE / focal BCE
- overlap duration imbalance에 따라 positive weighting

## 12.3 Speaker representation loss

우선순위:

1. supervised contrastive loss
2. cosine-margin / ArcFace-style loss
3. GE2E 계열

첫 구현에서는 **supervised contrastive loss**가 개념적으로 단순하다.

같은 speaker의 서로 다른 구간을 positive pair로, 다른 speaker를 negative로 만든다.

### multilingual speaker embedding 목표

이상적인 embedding은:

```text
embedding(A speaking KO)
 ~ embedding(A speaking EN)

embedding(A speaking KO)
 != embedding(B speaking KO)
```

이다.

v0에서는 별도의 adversarial language-invariance loss까지 넣지 않는다.

먼저 cross-language same-speaker positive pair를 충분히 넣는 것으로 시작한다.

## 12.4 Language loss

- frame/window-level cross entropy
- ambiguous/mixed/unlabeled 구간은 mask
- weight를 작게 둔다

language head가 speaker objective를 압도하지 못하게 한다.

---

# 13. Look-ahead / streaming 학습 방식

## 13.1 Low-Latency 단일 profile

학습과 inference에서 동일한 usable future context를 사용한다.

```text
reference frame grid = 80 ms
current chunk         = 6 frames = 480 ms
right context         = 7 frames = 560 ms
reference input buffer = 1040 ms
```

PSEM 내부 feature는 더 촘촘한 10/20 ms log-Mel frame을 사용할 수 있다. 외부 streaming schedule과 source timestamp contract는 LL reference에 맞춘다.

v0 실험은 모두 이 LL profile을 기준으로 진행한다.

## 13.2 왜 look-ahead를 허용하는가

speaker change 직후의 짧은 future evidence는 다음 case에서 유용할 수 있다.

- 짧은 backchannel
- 비슷한 음색
- overlap 종료 직후
- gap 직후 첫 syllable

따라서 완전 causal 자체를 성공 조건으로 삼지 않는다. 대신 1.04 s reference buffer와 실제 evidence/event latency를 함께 보고한다.

## 13.3 학습 방식

v0에서는 하나의 LL streaming configuration을 canonical config로 관리한다.

```text
PSEM-LL
  reference input buffer ≈ 1.04 s
```

architecture/size/loss를 비교할 때 streaming context 자체를 동시에 바꾸지 않아 실패 원인을 분리한다.

# 14. GT-only baseline을 먼저 만드는 이유

최초의 성공 조건은 **어떤 teacher도 없이 architecture가 GT supervision만으로 문제를 풀 수 있는지** 확인하는 것이다.

## Model A — GT-only

```text
Audio + ground-truth labels
          |
          v
        PSEM
```

이 모델로 확인할 것:

- change detection이 충분히 가능한가
- overlap head가 유용한가
- speaker embedding이 gap/code-switch robustness를 높이는가
- language auxiliary head가 실제로 도움되는가

GT-only가 안 되는데 KD부터 넣으면 실패 원인을 분리하기 어렵다.

---

# 15. Selected Teacher Knowledge Distillation

KD는 optional 단계다. PSEM 문서는 teacher를 선결정하지 않는다.

## 15.1 Teacher는 TC 결과에서 import

별도 `TC — Teacher / Reference Characterization` workstream이 최소 두 역할을 결정한다.

```text
performance/reference role
KD teacher role
```

둘은 같은 backend일 수도 있고 다를 수도 있다. 현재 characterization candidate에는 Streaming Sortformer v2.1 F32/Vulkan/LL이 포함되어 있다. 최종 KD teacher 선택은 TC 결과에 따른다.

TC가 `KD teacher = none`이라고 결론내리거나 KD가 student에서 실질적 개선을 만들지 않으면 canonical PSEM은 GT-only로 남을 수 있다.

## 15.2 Canonical TeacherTargetBundle

PSEM은 backend-specific speaker slot이나 cluster ID를 직접 학습하지 않는다.

TC에서 다음과 같은 model-neutral target artifact를 넘긴다.

```text
TeacherTargetBundle
  soft_change
  soft_overlap
  confidence
  valid_mask / uncertainty metadata
  provenance
```

핵심 semantic mapping:

```text
soft_change
  -> PSEM speaker_change 의미와 정렬
  -> non-overlap A->B에서는 B onset
  -> A->A+B->B에서는 post-overlap handoff resolution
  -> overlap 안의 단순 new-speaker entry를 그대로 positive로 사용하지 않음

soft_overlap
  -> simultaneous 2+ speaker activity와 정렬
```

이렇게 해야 teacher의 richer diagnostic evidence와 PSEM v0의 좁은 output semantics가 충돌하지 않는다.

## 15.3 무엇을 증류하지 않을 것인가

v0에서는 기본적으로 다음을 하지 않는다.

- teacher internal embedding -&gt; PSEM speaker embedding direct regression
- teacher speaker cache state imitation
- Sortformer 4-speaker slot imitation
- diart global cluster ID imitation
- permutation loss imitation
- long-term identity memory imitation

PSEM speaker embedding은 **GT speaker ID로 직접 학습**한다.

## 15.4 Latency alignment

Teacher target을 사용할 때 PSEM의 LL future-context condition과 가능한 범위에서 정렬한다.

```text
Student PSEM:
  LL reference input buffer ≈ 1.04 s

Selected teacher artifact:
  TC에서 source boundary와 observed/consumed frontier를 함께 기록
```

teacher가 더 긴 FIFO/cache/history를 사용했다면 그 차이를 provenance에 남긴다. student가 재현할 수 없는 장기 identity를 KD target으로 강제하지 않는다.

## 15.5 Ground truth 우선

Canonical KD loss는 개념적으로:

```text
GT loss + alpha * teacher_soft_loss
```

형태다.

Teacher와 GT가 강하게 충돌하면 GT를 우선한다.

- teacher confidence가 낮으면 KD loss를 제외하거나 낮춘다.
- teacher와 GT speaker activity가 크게 충돌하면 해당 구간을 제외할 수 있다.
- dataset annotation uncertainty는 별도 validity metadata로 관리한다.

Teacher는 oracle이 아니라 **noisy expert**다.

## 15.6 License / provenance

teacher가 선택되면 해당 model/runtime/license/provenance를 정확히 기록한다. release weights의 최종 license는 선택된 teacher와 dataset 조건을 바탕으로 별도 검토한다.

이 이유를 포함해 **GT-only canonical training path를 항상 유지**한다.

# 16. 실제 비교할 세 가지 학습 경로

## A. GT-only from scratch

```text
random init
  -> GT multi-task training
```

## B. GT + KD from scratch

```text
random init
  -> GT + selected canonical teacher target joint training
```

## C. GT baseline -&gt; KD fine-tune

```text
Model A checkpoint
  -> GT + KD fine-tuning
```

최종 선택은 empirical result로 한다.


| Variant | 목적                                                  |
| ------- | --------------------------------------------------- |
| A       | architecture 자체의 순수 성능                              |
| B       | KD가 optimization/representation을 처음부터 돕는지           |
| C       | 안정적인 GT representation 위에서 teacher가 refinement를 주는지 |


KD가 실질적으로 개선하지 않으면 canonical release에서 완전히 제거할 수 있어야 한다.

---

# 17. 전체 학습 단계

TC는 아래 PSEM phase와 별도 workstream이다.

```text
TC Teacher Characterization ───────────────┐
                                           v
PSEM Phase 0 -> Phase 1 -> Phase 2 -> Phase 3 KD -> Phase 4
     data       GT-only    language    optional       compression
```

TC는 PSEM Phase 0/1/2와 병렬 진행 가능하며 **Phase 3 KD 전에만 완료되면 된다.**

## Phase 0 — Dataset/label pipeline

Deliverables:

- dataset adapter interface
- speaker timeline canonical format
- overlap label generator
- change boundary generator
- language label adapter
- synthetic conversation generator
- speaker-disjoint split validator
- license/provenance manifest

Canonical intermediate representation 예:

```json
{
  "audio": "...",
  "sample_rate": 16000,
  "segments": [
    {"start": 0.0, "end": 1.4, "speaker": "spkA", "language": "ko"},
    {"start": 1.2, "end": 2.1, "speaker": "spkB", "language": "en"}
  ]
}
```

이 representation에서 change/overlap/frame labels를 생성한다.

## Phase 1 — 5~10M GT-only baseline

```text
change + overlap + speaker embedding
```

language head 없이 baseline을 먼저 만든다.

목적:

> speaker/acoustic signal만으로 문제가 풀리는지 확인.

## Phase 2 — Multilingual auxiliary experiment

같은 acoustic backbone에서 language auxiliary의 효과만 분리한다. 실험 variant를 고정한다.

```text
Model A0: change + overlap + speaker embedding
Model A1: change + overlap + speaker embedding + language auxiliary
```

다음 slice를 반드시 비교한다.

```text
same speaker / same language
same speaker / code switch
different speaker / same language
different speaker / different language
```

평균 F1만 좋아지고 code-switch가 악화되면 채택하지 않는다. A0/A1 이름은 실험 artifact와 report에서도 유지해 다른 architecture/KD 변화와 섞이지 않게 한다.

## Phase 3 — Selected-teacher KD

TC가 선택한 canonical target artifact가 있을 때만 실행한다.

```text
B: scratch + GT + KD
C: GT checkpoint + GT + KD
```

GT-only A와 같은 train/dev protocol에서 비교한다. KD가 반복적으로 도움이 없으면 이 phase의 결과를 canonical recipe에서 제거한다.

## Phase 4 — Compression

성능이 확인된 recipe를 기준으로:

```text
10M
5M
2M
1M
```

scale sweep한다. precision/quantization은 암묵적으로 처리하지 않고 별도 matrix로 기록한다.

```text
FP32  : 각 scale의 canonical quality/runtime baseline
INT8  : Windows CPU deployment 후보 scale에서 mandatory arm
FP16  : 실제 지원되는 inference environment에서 diagnostic arm
```

최소한 release 후보 scale에서는 FP32 대 INT8의 quality loss, checkpoint/RAM, CPU RTF, per-chunk latency를 같은 evaluation protocol로 비교한다. FP16은 CPU deployment 성공 조건으로 강제하지 않지만 지원 환경에서는 precision sensitivity를 기록한다.

**architecture가 안 풀린 상태에서 quantization을 먼저 하지 않는다.**

## Phase 5 — Overlap v1 research

v0 이후에 필요성이 확인될 때만:

```text
P(previous/current speaker still active during overlap)
```

같은 target-conditioned extension을 별도 연구한다. v0 release prerequisite가 아니다.

# 18. Evaluation 설계

Full diarization model이 아니므로 DER 하나를 primary metric으로 삼지 않는다.

Training label은 VAD-independent source timeline에서 만들되, evaluation은 TC와 비교 가능한 두 view를 사용한다.

```text
PSEM-A
  VAD turn reset을 강제하지 않는 capability view

PSEM-B500
  oracle GT speech activity
  configured silence hangover = 500 ms
  512-sample product chunk semantics에서 effective 512 ms
  max-turn cap = None
```

bounded-turn / max-turn cap 정책은 PSEM model evaluation의 primary definition에 포함하지 않는다.

## 18.1 Speaker Change Detection

반드시 보고:

- Precision
- Recall
- F1 @ +/-250 ms collar
- F1 @ +/-500 ms collar
- false changes / source hour
- false changes / source minute
- missed changes / source hour
- full recall-vs-false-event frontier

특정 false-event operating point를 절대 pass/fail gate로 고정하지 않는다.

### Boundary latency

- p50 detection latency
- p90 detection latency
- p95 detection latency

Latency 기준점은 task semantics에 따른 reference change boundary다.

- non-overlap `A -> B`: B speech onset
- `A -> A+B -> B`: post-overlap single-speaker handoff resolution boundary

추가 diagnostic으로 source boundary와 실제 evidence availability를 분리해 저장한다.

## 18.2 Overlap

- frame precision
- frame recall
- frame F1
- overlap start boundary error
- overlap end boundary error
- overlap evidence latency

v0에서는 overlap identity attribution metric을 사용하지 않는다.

## 18.3 반드시 분리해서 볼 hard-case slices

### Turn topology

```text
A -> B
A -> gap -> A
A -> gap -> B
A -> B -> A short backchannel
A -> A+B -> A
A -> A+B -> B
```

### Gap duration

```text
0~100 ms
100~300 ms
300~500 ms
500~800 ms
800~1200 ms
```

### Language

```text
KO -> KO
EN -> EN
JA -> JA
ZH -> ZH
KO -> EN
EN -> JA
JA -> ZH
...
```

### Speaker/language disentanglement

```text
same speaker + same language
same speaker + different language
different speaker + same language
different speaker + different language
```

### Acoustic

- quiet / loud
- far-field
- reverb
- background music
- TV/other speech leakage
- codec artifacts
- gain changes
- same-gender / similar voice
- very short utterance

## 18.4 Runtime metrics

- model parameters
- checkpoint size
- peak RAM
- steady-state RAM
- CPU RTF
- CPU utilization
- per-chunk inference p50/p95
- warm-up cost
- startup cost

PuriPuly target hardware는 Windows desktop 환경이므로 GPU benchmark만으로 판단하지 않는다.

# 19. Baseline / 비교 대상

## Primary external reference

PSEM 문서가 reference backend를 다시 정의하지 않는다. 별도 TC 결과에서 선택된 **performance/reference role**과 frozen artifacts를 import한다.

현재 TC의 primary characterized systems는 예를 들어:

```text
Streaming Sortformer v2.1 F32 / Vulkan / LL ≈ 1.04 s
diart L0.5
diart L1.0
```

이지만 최종 PSEM reference role은 TC 결과로 결정한다.

비교는 full DER이 아니라 PSEM과 동일한 canonical task projection에서 한다.

```text
selected reference artifact
  -> canonical change/handoff events
  -> canonical overlap evidence
  -> A / B500 metrics
```

## Optional simple baselines

- adjacent-window speaker embedding cosine SCD
- direct binary SCD without speaker auxiliary loss
- overlap-only independent classifier

이 simple baseline들이 PSEM multi-task 구조를 실제로 정당화하는지 확인한다.

# 20. 필수 ablation matrix

아래 실험은 최종 모델 결정 전에 반드시 한다.


| Experiment                     | 질문                                         |
| ------------------------------ | ------------------------------------------ |
| Change only                    | 가장 단순한 SCD가 어디까지 되는가?                      |
| Change + Speaker               | speaker representation이 change를 실제로 개선하는가? |
| Change + Overlap + Speaker     | overlap multi-task가 encoder에 도움이 되는가?      |
| + Language Aux                 | 다국어에서 실질적 이득이 있는가?                         |
| Language direct feature        | 직접 넣으면 이득인가 shortcut인가?                    |
| GT only vs selected-teacher KD | teacher signal이 실제로 필요한가?                  |
| KD scratch vs KD finetune      | KD 적용 순서는 무엇이 좋은가?                         |
| 10M / 5M / 2M / 1M             | capacity knee point는 어디인가?                 |


---

# 21. 초기 success gate

정확한 숫자는 데이터 확보 후 baseline distribution을 보고 확정한다. false-event frontier 자체가 primary이며 임의의 단일 FE/h 숫자를 절대 gate로 쓰지 않는다.

## Gate A — 접근법 성립

5~10M GT-only 모델이:

- simple cosine SCD baseline보다 명확히 우수
- code-switch hard negative에서 catastrophic false-change를 보이지 않음
- overlap head가 유의미한 성능 확보
- target CPU에서 real-time 충분히 가능

해야 한다.

## Gate B — reference 대비 가치

TC가 선택한 performance/reference와 **동일 A/B500 semantics 및 comparable false-event operating region**에서 비교한다.

- change/handoff frontier가 실용적으로 근접하는가
- overlap quality가 충분한가
- model size/compute가 현저히 작은가
- p95 evidence/event latency가 목적에 맞는가

reference backend가 Sortformer인지 diart인지는 이 gate의 정의가 아니다.

## Gate C — KD 채택

KD가 최소 하나 이상의 핵심 metric에서 반복적으로 개선되고 hard-case regression을 만들지 않을 때만 canonical recipe에 포함한다.

## Gate D — Compression

2M 또는 1M으로 내렸을 때 핵심 model metric이 무너지면 더 이상 줄이지 않는다. `1M` 자체는 성공 기준이 아니다.

# 22. 현재 예상 난이도 / 현실적 성공 확률

> 아래 수치는 실험 결과가 아니라 현재 문제 정의와 선행 연구를 바탕으로 한 **engineering estimate**다.


| 목표                                                    | 난이도   | PoC가 제품 가치에 도달할 추정 가능성 |
| ----------------------------------------------------- | -----: | ----------------------: |
| Speaker change v0                                     | 중     | 약 85~95%               |
| Overlap yes/no                                        | 중     | 약 80~90%               |
| 다국어 서로 다른 speaker 전환                                  | 중~낮음  | 단일언어보다 쉬운 case가 될 수 있음 |
| Same-speaker code-switch 오탐 억제                        | 중~높음  | 데이터 quality에 크게 의존     |
| selected external reference의 change 성능에 근접 + 훨씬 작은 모델 | 중~높음  | 실험으로 판단                |
| 1~2M까지 축소하면서 거의 손실 없음                                 | 높음    | 약 40~60%               |
| overlap 속 current-speaker tracking                    | 높음    | v1 연구 항목               |
| full diarization 대체                                   | 매우 높음 | 현재 목표 아님               |


현재 최대 리스크는 모델 architecture보다 **dataset composition / labels / multilingual shortcut**이다.

---

# 23. 주요 실패 모드와 대응

## 23.1 언어가 바뀌면 speaker가 바뀌었다고 착각

증상:

```text
A(KO) -> A(EN)
```

에서 false change 급증.

대응:

- same-speaker cross-language positive pair 확보
- language head weight 축소
- language logits를 change head에 직접 넣지 않기
- language-head 없는 baseline과 지속 비교

---

## 23.2 Channel/background change를 speaker change로 학습

대응:

- synthetic whole-conversation augmentation
- same-speaker/channel-change hard negative
- source dataset와 speaker identity correlation 제거

---

## 23.3 짧은 backchannel miss

대응:

- canonical LL look-ahead를 고정해 streaming context 변화와 모델 실패를 혼동하지 않기
- short utterance oversampling
- fuzzy boundary label
- teacher soft target 활용 가능

---

## 23.4 Gap 뒤 false speaker change

```text
A -> 300ms silence -> A
```

대응:

- 0~1.2 s same-speaker gap hard negatives
- speaker representation auxiliary loss
- runtime에서 silence 중 prototype 업데이트 금지(embedding 기반 continuity를 사용할 경우)

---

## 23.5 Overlap 때문에 embedding 오염

대응:

- speaker embedding loss는 clean solo region 중심
- overlap region에서 runtime prototype update 금지
- v0 speaker change policy는 overlap 동안 defer

---

## 23.6 KD가 teacher 오류까지 복사

대응:

- GT 우선
- low-confidence teacher masking
- teacher와 GT disagreement 분석
- KD 없이도 canonical model이 학습 가능하도록 유지

---

# 24. Runtime API / timeline contract

훈련과 downstream integration 사이의 model contract를 작게 유지한다.

예시:

```python
@dataclass
class SpeakerModelFrame:
    source_sample: int
    observed_source_sample: int
    emitted_monotonic_ns: int | None
    change_prob: float
    overlap_prob: float
    embedding: np.ndarray | None       # shape [64 or 128]
    language_probs: np.ndarray | None  # optional
```

의미:

```text
source_sample
  = 이 frame/evidence가 가리키는 canonical source 위치

observed_source_sample
  = 이 결과를 만들 때까지 모델이 실제로 소비한 source frontier

emitted_monotonic_ns
  = compute/scheduling latency를 분리해서 측정하기 위한 optional wall-clock receipt
```

이 세 값을 분리해야 1.04 s look-ahead, boundary location error, 실제 inference latency를 섞지 않는다.

Optional model-side eventizer는 다음 semantic event를 만들 수 있다.

```text
speaker_change
overlap_start
overlap_end
```

하지만 downstream application이 이 event를 STT/UI/LLM에서 어떻게 소비할지는 이 API가 규정하지 않는다.

PSEM이 VAD implementation을 직접 import하거나 제어하지 않는다는 owner boundary는 유지한다.

# 25. Model-side event semantics

아래는 PSEM probability를 semantic event로 안정화하는 **model-side 개념**이다. 제품에서 event를 어떻게 사용할지는 별도 문서의 책임이다.

```python
on_speaker_frame(spk_result):
    update_overlap_hysteresis(spk_result.overlap_prob)
    update_change_hysteresis(spk_result.change_prob)

    if overlap_just_started:
        emit_model_event("overlap_start")

    if overlap_active:
        # v0: overlap 내부 participant transition을 speaker_change로 commit하지 않음
        freeze_or_avoid_speaker_prototype_update()
        return

    if overlap_just_ended:
        emit_model_event("overlap_end")
        # reliable clean evidence에서 handoff change를 확정할 수 있음

    if sustained_change_confidence:
        emit_model_event("speaker_change")
```

VAD lifecycle이 short-term continuity state를 언제 reset할지는 runtime adapter에서 source timeline과 함께 처리할 수 있다. 중요한 semantic 원칙은 **overlap 내부의 새 speaker 진입 자체를 PSEM speaker_change로 확정하지 않는 것**이다.

threshold/hysteresis는 dev set tuning 대상이며 training target 의미와 분리한다.

# 26. 오픈소스 / 라이선스 전략

## 26.1 PuriPuly

현재 PuriPuly-heart repository는 `AGPL-3.0-or-later`로 표시되어 있다.

PuriPuly integration code는 repository의 기존 라이선스 정책을 따른다.

## 26.2 PSEM code

선택지는 두 가지다.

### 같은 repo에 넣기

- PuriPuly AGPL-3.0-or-later 아래에서 관리

### 별도 model/training repo

- training/inference library의 별도 permissive license를 검토 가능
- PuriPuly integration layer는 AGPL 유지

어느 쪽이든 source, configs, preprocessing, evaluation을 공개하는 것을 목표로 한다.

## 26.3 Selected teacher provenance

KD를 실제로 채택하는 경우 TC가 선택한 teacher/runtime/model/license 정보를 그대로 provenance에 기록한다.

현재 characterization candidate에는 Streaming Sortformer v2.1 F32/Vulkan/LL이 포함되어 있으며, 최종 KD teacher는 TC 결과로 결정한다.

원칙:

- selected teacher model/version/runtime을 명시한다.
- teacher license와 artifact provenance를 보존한다.
- distilled weights의 최종 공개 라이선스는 별도 검토한다.
- GT-only official training path와 checkpoint를 유지한다.
- `KD teacher = none`도 정상적인 최종 결론으로 허용한다.

---

# 27. 권장 repository layout 예시

별도 model repository를 만든다고 가정하면:

```text
psem/
  README.md
  LICENSE
  MODEL_CARD.md
  pyproject.toml

  configs/
    model_10m.yaml
    model_5m.yaml
    model_2m.yaml
    streaming_lowlatency.yaml
    teacher_targets.yaml

  psem/
    audio.py
    features.py
    encoder.py
    heads.py
    streaming.py
    events.py

  training/
    datasets/
    labels/
    synthetic/
    losses/
    train.py
    distill.py

  evaluation/
    change_metrics.py
    overlap_metrics.py
    latency_metrics.py
    slice_metrics.py

  scripts/
    prepare_ko_TBD.py
    prepare_en_TBD.py
    prepare_ja_TBD.py
    prepare_zh_TBD.py
    generate_synthetic.py

  provenance/
    datasets.yaml
    teachers.yaml

  tests/
```

PuriPuly repo에는 inference adapter만 두고 downstream product policy는 별도 integration owner에서 관리하는 형태도 가능하다.

---

# 28. 실험 기록에 반드시 저장할 것

각 run은 다음 정보를 남긴다.

```text
model parameter count
feature config
chunk size
right context
look-ahead
output stride
training data composition by language
same-speaker code-switch ratio
different-speaker same-language ratio
overlap ratio
gap duration distribution
synthetic/real ratio
loss weights
selected teacher identity
teacher target bundle version
teacher streaming/provenance receipt
KD alpha/temperature
```

이 정보가 없으면 나중에 multilingual/KD 효과를 재현하기 어렵다.

---

# 29. 첫 실험의 최소 matrix

처음부터 수십 개 모델을 돌리지 않는다. streaming profile은 LL 하나로 고정한다.

### Experiment 1 — Core viability

```text
5~10M
change + overlap + speaker embedding
GT only
LL ≈ 1.04 s
```

### Experiment 2 — Language auxiliary

```text
Experiment 1 + language auxiliary head
```

### Experiment 3 — Optional KD

```text
best of Exp1/2
+ TC-selected canonical `TeacherTargetBundle`
```

selected KD teacher가 없거나 target quality가 부족하면 이 experiment는 건너뛸 수 있다.

### Experiment 4 — Compression

```text
best recipe -> 5M -> 2M
```

1M은 2M 결과가 충분히 좋을 때만 간다.

# 30. 최종 의사결정 트리

```text
5~10M GT-only가 speaker change + overlap을 충분히 잡나?
  |
  +-- NO --> architecture/data/label 수정
  |          KD/quantization 하지 않음
  |
  +-- YES
       |
       v
language auxiliary가 hard-case까지 개선하나?
       |
       +-- NO --> language head 제거
       |
       +-- YES --> 유지
       |
       v
TC가 usable KD teacher/target을 선택했나?
       |
       +-- NO --> GT-only recipe 유지
       |
       +-- YES
       |     |
       |     v
       |   KD가 반복적으로 개선하나?
       |     |
       |     +-- NO --> canonical recipe에서 KD 제거
       |     +-- YES --> KD recipe 채택 후보
       |
       v
5M으로 줄여도 핵심 frontier가 유지되나?
       |
       +-- NO --> larger model에서 멈춤
       |
       +-- YES
            |
            v
2M / 1M 테스트
            |
            v
v0 release candidate
            |
            v
필요할 경우 overlap current-speaker tracking은 별도 v1 연구
```

LL ≈1.04 s streaming profile은 v0 의사결정 트리에서 고정 조건이다.

# 31. 최종 v0 정의

한 문장으로 정의하면:

> **PSEM v0는 PuriPuly Peer Channel의 16 kHz mono audio에서 기존 streaming VAD와 병렬로 동작하고, Low-Latency 약 1.04 s streaming context 아래에서 `speaker_change`, binary `overlap`, short-term `speaker_embedding`을 추정하는 초경량 multilingual streaming acoustic model이다. GT-only를 canonical 기반으로 유지하고, teacher-dependent KD는 별도 TC 결과가 유용성을 입증할 때만 optional하게 추가한다.**

### v0 입력

```text
16 kHz mono audio stream
```

### v0 출력

```text
P(speaker_change)
P(overlap)
64~128d speaker embedding
(optional) language auxiliary logits
```

### Optional model-side event

```text
speaker_change
overlap_start
overlap_end
```

### 기존 시스템 / downstream owner가 담당

```text
speech_start / speech_end
VAD hangover
STT
translation
UI
logical-turn / transcript handling
speaker embedding consumption policy
```

### v0가 하지 않는 것

```text
full diarization
persistent speaker IDs
speaker enrollment
MAIN/OTHER identification
overlap 내부 new-speaker attribution
3+ overlap attribution
overlap word attribution
product UI/STT/LLM policy 결정
```

# 32. 최종 우선순위

## 반드시 성공시킬 것

1. **Speaker change** — PSEM handoff semantics
2. **Overlap yes/no**
3. **Multilingual robustness / code-switch false-positive 억제**
4. **Low-Latency 약 1.04 s 조건에서 실제 Windows CPU streaming inference**

## 모델 품질을 높이기 위한 보조 목표

5. Speaker embedding
6. Language auxiliary task
7. Selected-teacher soft distillation, **유효할 때만**

## v0 이후

8. Overlap 중 current speaker presence
9. 2-speaker local overlap attribution
10. optional local re-identification research

## 명시적으로 우선순위가 낮거나 별도 product 문서 범위

11. Full diarization
12. Long-term global speaker ID
13. overlap UI / transcript semantic handling
14. speaker embedding의 product context/history 정책

# 33. 남은 TBD

실제 구현/실험에서 확정해야 할 값은 다음과 같다.

- [ ] KO training dataset: **TBD**
- [ ] EN training dataset: **TBD**
- [ ] JA training dataset: **TBD**
- [ ] ZH training dataset: **TBD**
- [ ] multilingual/code-switch dataset: **TBD**
- [ ] `PSEM-FINAL` held-out evaluation set: **TBD**
- [ ] target Windows CPU classes / minimum hardware: **TBD**
- [ ] canonical model size: baseline 결과 후 결정
- [ ] exact loss weights: ablation 후 결정
- [ ] event hysteresis/threshold: PSEM-DEV에서 결정
- [ ] TC-selected performance/reference role: **TC 결과로 결정**
- [ ] TC-selected KD teacher / target bundle: **TC 결과로 결정; none 허용**
- [ ] release weights license/provenance policy: dataset/teacher 확정 후 검토

Latency profile은 이 문서에서 확정한 항목이다.

```text
PSEM v0 canonical streaming profile: LL ≈ 1.04 s
```

제품에서 PSEM output을 UI/STT/translation/context에 어떻게 소비할지는 이 TBD 목록에 넣지 않는다. 별도 integration 문서의 의사결정으로 추적한다.

# 34. 이 문서에서 의도적으로 닫지 않는 downstream 질문

PSEM acoustic model의 성공 조건과 product integration 정책을 분리한다.

다음 질문은 중요하지만 이 설계/학습 계획에서 결정하지 않는다.

```text
speaker_change를 실제 STT hard split으로 사용할 것인가?
overlap을 자막/UI에 어떻게 표현할 것인가?
overlap span의 transcript를 LLM/translation 단계에서 어떻게 다룰 것인가?
speaker_embedding을 runtime context/history에 사용할 것인가?
```

## 34.1 Downstream integration handoff checklist — non-blocking

후속 integration 문서가 잊지 않고 검토해야 할 구체 항목을 handoff checklist로 보존한다. 아래 항목은 **이 문서의 결정사항이나 PSEM 학습 blocker가 아니다.**

- VAD-active segment 내부 `speaker_change`를 STT input/session의 실제 분할 신호로 사용할 것인가?
- hard split을 사용한다면 STT backend가 safe soft-finalize/restart를 지원하는가?
- word/token timestamp가 충분히 안정적이면, streaming session을 즉시 자르기보다 final transcript를 event source position 기준으로 사후 분할하는 편이 나은가?
- streaming partial/final transcript와 look-ahead 때문에 늦게 emit되는 PSEM event를 어떤 source-timeline contract로 정렬할 것인가?
- overlap span에서 speaker attribution이 없는 transcript를 downstream LLM이 과도하게 복원/추측하지 않도록 어떤 contract를 둘 것인가?
- acoustic metric과 별도로 `false logical split`, `missed logical split`, speaker-mixed transcript/translation-turn 비율을 어떤 product metric으로 정의할 것인가?
- overlap UI/preview, semantic segmentation, embedding-based context routing을 도입한다면 각각을 독립 experiment로 분리할 것인가?

후속 integration 문서는 PSEM이 제공하는 source-timeline metadata와 probability/event contract를 입력으로 사용하면 된다.

따라서 이 항목들은 **PSEM v0 acoustic-model 학습의 blocker가 아니다.**

# 35. References

## PuriPuly

1. PuriPuly-heart repository  
 [https://github.com/kapitalismho/PuriPuly-heart](https://github.com/kapitalismho/PuriPuly-heart)
2. PuriPuly README — project/runtime overview  
 [https://github.com/kapitalismho/PuriPuly-heart/blob/main/README.md](https://github.com/kapitalismho/PuriPuly-heart/blob/main/README.md)
3. Peer runtime — `vad_hangover_ms` and Peer capture runtime  
 [https://github.com/kapitalismho/PuriPuly-heart/blob/main/src/puripuly_heart/core/runtime/peer_channel.py](https://github.com/kapitalismho/PuriPuly-heart/blob/main/src/puripuly_heart/core/runtime/peer_channel.py)
4. VAD defaults — 500 ms low-latency / 1000 ms stable  
 [https://github.com/kapitalismho/PuriPuly-heart/blob/main/src/puripuly_heart/config/vad_defaults.py](https://github.com/kapitalismho/PuriPuly-heart/blob/main/src/puripuly_heart/config/vad_defaults.py)
5. VAD gating implementation — streaming VAD event and Peer settings  
 [https://github.com/kapitalismho/PuriPuly-heart/blob/main/src/puripuly_heart/core/vad/gating.py](https://github.com/kapitalismho/PuriPuly-heart/blob/main/src/puripuly_heart/core/vad/gating.py)

## Streaming diarization / teacher

- `puripuly_sortformer_diart_benchmark_plan_v5.md` — TC reference/KD-teacher characterization contract

6. NVIDIA Streaming Sortformer Diarizer 4spk v2.1 model card — TC characterization candidate 및 1.04 s low-latency reference  
 [https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1)
7. NVIDIA Open Model License  
 [https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/)
8. Online Target Speaker Voice Activity Detection for Speaker Diarization  
  [https://arxiv.org/abs/2207.05920](https://arxiv.org/abs/2207.05920)
9. End-to-end Online Speaker Diarization with Target Speaker Tracking  
  [https://arxiv.org/abs/2310.08696](https://arxiv.org/abs/2310.08696)
10. Target Speaker Voice Activity Detection with Transformers and Its Integration with End-to-End Neural Diarization  
  [https://arxiv.org/abs/2208.13085](https://arxiv.org/abs/2208.13085)

## Multilingual / code-switch

11. DISPLACE Challenge: DIarization of SPeaker and LAnguage in Conversational Environments  
  [https://arxiv.org/abs/2303.00830](https://arxiv.org/abs/2303.00830)
12. DISPLACE Challenge site / multilingual code-mixed diarization context  
  [https://displace2024.github.io/](https://displace2024.github.io/)

# 36. 결론

PSEM v0의 문제 정의는 의도적으로 작게 유지한다.

```text
"reliable single-speaker 상태가 바뀌었는가?"
"현재 두 명 이상이 동시에 말하고 있는가?"
"짧은 범위의 speaker continuity를 표현할 수 있는가?"
```

이를 위해 모델은:

```text
P(speaker_change)
P(overlap)
speaker_embedding
```

을 출력한다.

overlap 내부에서는 누가 새로 들어왔는지 추적하지 않는다. `A -> A+B -> A`는 change가 아니고, `A -> A+B -> B`는 overlap이 끝나 B-only reliable state가 확인된 뒤 change로 본다.

Streaming profile은 **LL 약 1.04 s 하나로 고정**한다.

학습 순서는:

```text
Dataset/labels
  -> 5~10M GT-only
  -> optional language auxiliary
  -> optional TC-selected KD
  -> compression
```

이다.

Teacher/reference 선택은 별도 TC 문서가 담당한다. TC는 PSEM Phase 0/1/2와 병렬로 진행할 수 있고, KD 전에만 완료되면 된다. PSEM은 backend-specific slot/cluster를 직접 모방하지 않고 canonical `TeacherTargetBundle`을 통해 change/handoff와 overlap soft evidence만 선택적으로 받는다.

speaker embedding은 모델 output/representation으로 유지하지만 제품에서 어떻게 사용할지는 이 문서에서 정하지 않는다. overlap을 UI에 표시하거나 transcript/translation을 의미론적으로 처리하는 정책 역시 별도 integration 설계의 책임이다.

따라서 이 문서의 성공 기준은 **작은 LL streaming acoustic model이 GT supervision만으로 speaker change + binary overlap 문제를 풀 수 있는지**, 그리고 선택된 teacher가 있을 때 KD가 실제로 추가 가치를 주는지에 집중한다.