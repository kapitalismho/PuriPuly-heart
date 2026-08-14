# Cloud ASR Finalization 정책 교정 계획서

## 1. 문서의 성격

- 작성일: 2026-08-12
- 기준 커밋: `31008444` (`revert local ASR trailing silence trimming`)
- 대상 provider: Deepgram, Soniox, Qwen ASR Flash Realtime
- 대상 채널: 셀프와 피어
- 목적: provider가 반환한 확정 결과를 앱이 유실·삭제·중복·오매핑하지 않고, 정상 종료와 명시적 중단을 일관되게 처리하도록 finalization 정책을 교정한다.

이 문서는 음향 모델의 CER을 개선하기 위한 계획이 아니다. 입력 레벨, gain, AGC, 리샘플링, VAD의 PCM 보존과 스테레오 다운믹스는 [ASR 입력 품질 후퇴 원인 규명 실험 계획서](asr-input-quality-regression-plan.md)에서 별도로 다룬다.

두 계획은 다음 원칙으로 분리한다.

- production 변경, 테스트와 결과 문서를 서로 다른 작업 단위로 유지한다.
- 이 계획은 provider protocol, 결과 보존, 요청 lifecycle과 종료 drain만 다룬다.
- 입력 품질 계획은 provider final 이벤트의 의미나 adapter 상태 머신을 변경하지 않는다.
- 두 작업의 결과를 비교할 때는 사용한 Git revision을 기록하되 어느 한 계획의 실험 하네스를 다른 계획의 선행 조건으로 만들지 않는다.
- 같은 음성 파일을 소규모 확인에 재사용할 수 있지만 공통 benchmark framework를 새로 만들지는 않는다.

## 2. 최종적으로 답해야 할 질문

1. Deepgram의 segment-level `is_final`을 현재 코드가 local utterance final로 잘못 해석하고 있는가?
2. Deepgram의 manual `Finalize` 한 번에 속한 여러 확정 segment를 누락 없이 한 발화로 만들 수 있는가?
3. Soniox의 확정 token을 timestamp 기반 필터나 replacement merge가 삭제·변형할 수 있는가?
4. Deepgram, Soniox, Qwen 모두에서 이미 전송된 실제 tail, adapter가 추가한 무음과 provider 전용 tail option이 서로 일치하는가?
5. 현재 100 ms top-up/trim 정책은 어디서 왔으며 자연 silence boundary와 강제 max-duration boundary에 각각 적합한가?
6. Qwen의 빈 `completed`, transcription failure와 session 종료가 pending utterance를 남기거나 마지막 결과를 잃게 하는가?
7. 정상 drain, provider 교체·reset과 사용자가 의도한 즉시 abort를 어떤 상태로 구분해야 하는가?
8. 현재의 `SpeechEnd 1회 = backend final event 1회` FIFO 가정을 adapter 내부에서 안전하게 지킬 수 있는가, 아니면 공통 STT contract에 correlation이 필요한가?
9. peer의 7초 max-duration 강제 분할을 실제 무음 경계처럼 다루지 않으면서 결과 지연 상한을 유지할 수 있는가?
10. 교정이 중복 transcript, 오래된 결과 발표 또는 과도한 종료 지연을 새로 만들지는 않는가?

## 3. 시스템 경계와 현재 구조

`ARCHITECTURE.md`에 따르면 provider adapter는 streaming, response normalization과 provider error를 소유하고, channel/controller owner는 lifecycle과 normalized STT event를 소유한다. runtime 교체 뒤 늦게 도착한 결과에는 generation, attachment token, request ID 또는 현재 owner identity 검증이 필요하다.

현재 관련 지점은 다음과 같다.

- `src/puripuly_heart/core/stt/backend.py:12-25`
  - `STTBackendTranscriptEvent`에는 `text`, `is_final`, `final_language_runs`만 있다.
  - `on_speech_end(trailing_silence_ms)`에는 boundary 또는 finalize request ID가 없다.
- `src/puripuly_heart/core/stt/controller.py:467-491`
  - `SpeechEnd`마다 local utterance ID를 pending FIFO에 넣고 provider `on_speech_end()`를 호출한다.
- `src/puripuly_heart/core/stt/controller.py:1021-1075`
  - backend의 모든 `is_final` 이벤트가 pending ID 하나를 먼저 소비한다.
  - 빈 final text도 ID를 소비한 뒤 사용자에게 발표되지 않는다.
- `src/puripuly_heart/core/stt/controller.py:884-893`
  - 정상 drain은 기본 200 ms grace를 사용한다.
- `src/puripuly_heart/core/stt/controller.py:225-284`
  - 즉시 toggle-off는 generation을 교체하고 pending 결과를 비우는 abortive 의미다.
- `src/puripuly_heart/core/vad/gating.py:178-219`
  - VAD hangover에 포함된 chunk는 `SpeechEnd` 전에 이미 provider로 전송된다.
- `src/puripuly_heart/core/vad/gating.py:303-330`, `391-415`
  - peer는 7,000 ms에 `reason=max_duration`, `trailing_silence_ms=0`으로 강제 분할된다.
  - 강제 분할 뒤 연속 speech가 다시 시작될 때 일반 pre-roll/debounce 경로를 거쳐 최근 ring audio가 다시 전송될 수 있다.

이 계약은 provider가 local `SpeechEnd`마다 정확히 하나의 ordered terminal event를 보낸다는 가정에 의존한다. segment final이 여러 번 오거나 성공한 빈 결과·실패에서 terminal event가 오지 않으면 이후 utterance 매핑이 밀릴 수 있다.

현재 기본 VAD hangover는 self와 peer 모두 500 ms다. 따라서 자연 silence boundary에서는 약 500 ms의 실제 무음 PCM이 이미 세 provider에 들어간다. 반면 peer 7초 강제 boundary에서는 관측된 실제 tail이 0 ms다. 두 경계를 같은 tail 정책으로 처리하면 안 된다.

## 4. Provider별 현재 사실

### 4.1 Deepgram

현재 adapter:

- `src/puripuly_heart/providers/stt/deepgram.py:124-157`
  - `is_final` 또는 `speech_final` 중 하나가 참이면 즉시 앱 final을 만든다.
  - `from_finalize`를 읽거나 manual-finalize 단위 aggregate를 만들지 않는다.
- `src/puripuly_heart/providers/stt/deepgram.py:198-208`
  - `interim_results=False`, `vad_events=False`, `endpointing=False`다.
- `src/puripuly_heart/providers/stt/deepgram.py:304-310`
  - local speech end에서 `Finalize` control message를 보낸다.
- `src/puripuly_heart/providers/stt/deepgram.py:348-373`
  - 관측된 tail이 100 ms보다 짧으면 부족분만큼 zero PCM을 물리적으로 추가한 뒤 `Finalize`한다.
- `src/puripuly_heart/providers/stt/deepgram.py:375-387`
  - stop은 send loop를 종료하지만 명확한 finalize fence 또는 `CloseStream` 결과를 기다리지 않는다.

공식 계약:

- `is_final=true`는 특정 오디오 구간의 확정이며 전체 utterance 완료와 같지 않을 수 있다.
- 완전한 utterance는 여러 확정 segment를 누적해야 할 수 있다.
- manual `Finalize`는 처리되지 않은 오디오를 flush한다.
- `from_finalize=true`가 응답에 붙을 수 있지만 처리할 오디오가 거의 없으면 응답 자체가 보장되지 않는다.
- 공식 `Finalize` 문서는 finalize 전 무음을 필수 조건으로 지정하지 않는다. 현재 100 ms zero top-up은 provider protocol 요구가 아니라 앱의 휴리스틱이다.

현재 위험:

- 긴 발화의 앞 segment만 local utterance에 연결되고 뒤 segment가 다음 ID를 소비하거나 버려질 수 있다.
- 빈 segment final도 다음 boundary 매핑을 바꿀 수 있다.
- stop/reset 직전의 마지막 확정 결과가 drain되지 않을 수 있다.
- peer 7초 강제 boundary마다 100 ms synthetic zero가 삽입되며, 이후 continuation audio와 사이에 실제로 없던 gap이 생긴다.

### 4.2 Soniox

현재 adapter:

- `src/puripuly_heart/providers/stt/soniox.py:273-315`
  - final token을 모으고 `<fin>` 또는 `<end>`를 flush 경계로 사용한다.
  - `end_ms <= previous_end_ms`인 token을 건너뛴다.
- `src/puripuly_heart/providers/stt/soniox.py:317-409`
  - final batch가 겹치거나 과거 결과를 교정할 수 있다고 가정해 replacement merge를 수행한다.
- `src/puripuly_heart/providers/stt/soniox.py:479-502`
  - 실제 VAD tail 값 대신 설정 기본값 100 ms를 finalize 요청에 사용한다.
  - controller가 정상적으로 정수 tail을 넘기면 adapter가 별도 물리적 silence를 추가하지 않는다.
- repository 기본값과 2026-08-12 현재 로컬 `settings.json` 값은 모두 100 ms다. 현재 실행 경로에서 150 ms를 사용하는 근거는 확인되지 않았다.

공식 계약:

- final token은 한 번만 전송되고 이후 반복되거나 수정되지 않는다.
- manual finalize 완료는 `<fin>`으로 표시된다.
- provider는 finalize 전에 약 200 ms silence 확보를 권장한다.
- SDK는 `trailing_silence_ms`를 이미 들어간 trailing silence를 trim하는 option으로 설명하며 500 ms는 사용 예시일 뿐 권장 기본값이 아니다.
- `trailing_silence_ms`는 wait나 zero PCM 추가가 아니다. 값을 500 ms로 지정해도 client가 500 ms를 더 기다리는 것은 아니지만, 실제로 들어 있지 않은 500 ms를 선언하면 안 된다.

현재 위험:

- 같은 timestamp를 공유하는 합법적인 token이 있다면 현재 필터가 글자, 공백 또는 문장부호를 삭제할 수 있다.
- 공식 계약에 필요 없는 replacement merge가 확정 text를 변형할 수 있다.
- peer의 7초 강제 boundary는 실제 tail 0 ms인데 Soniox에는 100 ms가 전달될 수 있다.
- 일반 self 발화는 VAD hangover가 이미 PCM으로 전달되므로 peer 강제 boundary와 구분해야 한다.

### 4.3 Qwen ASR Flash Realtime

현재 adapter:

- `src/puripuly_heart/providers/stt/qwen_asr.py:159-168`
  - `conversation.item.input_audio_transcription.completed`만 앱 final로 사용한다.
  - transcript가 비어 있으면 terminal event를 만들지 않는다.
- transcription-specific `failed` 이벤트를 처리하지 않는다.
- `src/puripuly_heart/providers/stt/qwen_asr.py:263-269`
  - local speech end마다 `commit()`을 보낸다.
- `src/puripuly_heart/providers/stt/qwen_asr.py:325-347`
  - 관측된 tail이 100 ms보다 짧으면 부족분만큼 zero PCM을 물리적으로 추가한 뒤 `commit()`한다.
- `src/puripuly_heart/providers/stt/qwen_asr.py:349-359`
  - stop은 `end_session()`이 아니라 `conversation.close()`를 사용한다.

설치된 DashScope SDK에서:

- `commit()`은 event ID가 있는 commit 이벤트를 보낸다.
- `end_session()`은 `session.finish`를 보내고 final recognition 완료를 기다린다.
- `close()`는 WebSocket을 닫을 뿐이다.
- 공식 manual-mode 문서는 `commit()`이 buffer 전체를 하나의 utterance로 인식한다고 설명하지만 commit 전 무음 padding을 요구하지 않는다. 현재 100 ms top-up은 앱의 휴리스틱이다.

현재 위험:

- 빈 completed와 transcription failure가 pending utterance를 종료하지 않아 다음 결과가 과거 ID에 연결될 수 있다.
- 정상 drain/reset에서 마지막 completed가 오기 전에 WebSocket을 닫을 수 있다.
- Qwen이 제공하는 event/item ID를 사용하지 않고 여러 commit의 ordered completion을 가정한다.
- peer 7초 강제 boundary마다 100 ms synthetic zero가 삽입된다.

보관된 과거 로그에서는 Qwen commit 1,204회와 completed 1,196회가 기록됐다. 차이 8회는 toggle-off, provider error와 session reset 경계에 집중됐다. 이는 steady-state 문장 내부 오인식의 증거가 아니라 lifecycle final 유실이 실제로 발생했다는 제한된 증거다.

## 5. 범위와 비범위

### 포함

- provider raw event를 normalized terminal outcome으로 변환하는 adapter 정책
- segment aggregate와 manual-finalize fence
- 성공한 빈 결과와 provider transcription failure 처리
- 정상 drain, provider 교체/reset과 즉시 abort
- pending FIFO 오염, 늦은 결과와 중복 결과 방지
- 세 provider의 실제 관측 tail, synthetic zero padding과 provider 전용 tail option의 정합성
- peer 7초 max-duration boundary와 연속 발화 재시작 정책
- 작은 scripted event test와 최소 live wire 확인

### 제외

- gain, AGC, limiter와 noise suppression
- resampling 품질, capture volume과 Windows APO
- 일반적인 VAD threshold 또는 hangover 최적화
- stereo downmix, HRTF와 공간 분리
- cloud model 간 CER benchmark
- 범용 streaming ASR harness, dashboard 또는 대규모 corpus
- final 정책 수정과 무관한 STT·translation architecture refactor

## 6. 교정 불변조건

### 6.1 Boundary outcome

정상 처리 대상으로 수락한 local speech boundary는 내부적으로 정확히 하나의 terminal outcome으로 끝나야 한다.

- `completed_text`
- `completed_empty`
- `failed`
- `canceled`

현재 공통 event가 이 상태를 모두 표현하지 못하더라도 adapter와 controller의 pending 상태는 정확히 한 번 해소되어야 한다. 첫 좁은 패치에서는 빈 `is_final` acknowledgment와 구조화 로그로 표현할 수 있다. 명시적인 상태가 제품 동작이나 관측 가능성에 필요하다고 판명될 때만 공통 contract를 확장한다.

### 6.2 Text preservation

- provider가 확정한 token 또는 segment를 공식 근거 없이 삭제하지 않는다.
- provider 문서가 final token 불변성을 보장하면 append-only로 처리한다.
- provider의 segment final과 local utterance terminal final을 구분한다.
- normalize 단계의 whitespace·punctuation 변경은 별도로 드러나야 하며 단어를 제거하면 안 된다.

### 6.3 Correlation과 ordering

- 하나의 provider terminal fence가 local boundary 하나만 해소해야 한다.
- 빈 결과와 실패도 순서를 전진시켜야 한다.
- reset 또는 provider 교체 전 요청의 늦은 결과는 새 generation에 발표되지 않아야 한다.
- FIFO만으로 안전함을 입증할 수 없으면 local boundary ID와 provider request/item ID를 연결한다.

### 6.4 Drain과 abort

- 정상 drain은 새 ingress를 막은 뒤 이미 수락한 boundary의 terminal outcome을 bounded wait로 회수한다.
- 사용자의 즉시 toggle-off는 cancel을 명시하고 늦은 결과를 발표하지 않는 abort다.
- 고정 sleep만으로 정상 drain 완료를 추정하지 않는다.
- provider SDK의 blocking 종료 호출은 application event loop를 막지 않는다.

### 6.5 Tail accounting과 truthfulness

모든 provider에서 다음 네 값을 분리해 기록한다.

1. `observed_tail_ms`: VAD가 감지했고 이미 provider에 전송한 실제 silence
2. `injected_padding_ms`: adapter가 finalize/commit 직전에 추가한 zero PCM
3. `declared_trim_ms`: Soniox처럼 provider option으로 알린 기존 tail 길이
4. `boundary_wait_ms`: speech가 끝난 뒤 local boundary를 확정하기 위해 실제로 기다린 시간

이 값들은 서로 대체할 수 없다.

- 자연 silence boundary에서는 기본 VAD hangover 때문에 `observed_tail_ms`가 약 500 ms다. 세 provider 모두 추가 padding을 기본적으로 보내지 않는다.
- Deepgram과 Qwen에는 Soniox `trailing_silence_ms`에 대응하는 trim option이 없다. 두 provider의 현재 100 ms 동작은 zero PCM top-up이다.
- Soniox의 현재 100 ms는 zero PCM이나 wait가 아니라 trim option이다. `declared_trim_ms <= observed_tail_ms + injected_padding_ms`를 만족해야 한다.
- speech 중간 max-duration boundary에서는 `observed_tail_ms=0`이다. 이 경계를 자연 silence boundary처럼 기록하지 않는다.
- 500 ms synthetic padding을 기본 후보로 사용하지 않는다. 자연 boundary에는 이미 약 500 ms가 있고, 강제 boundary에 500 ms를 매번 삽입하면 실제 음성 timeline을 크게 왜곡한다.

### 6.6 Boundary reason별 기본 정책

| Boundary | Deepgram | Soniox | Qwen |
|---|---|---|---|
| 자연 silence | 기존 약 500 ms tail 사용, 추가 zero 0 | 기존 약 500 ms tail 사용, trim은 실제 tail 이하 | 기존 약 500 ms tail 사용, 추가 zero 0 |
| 5~7초 soft window의 실제 pause | 관측된 pause 사용 | 관측된 pause 사용, trim은 실제 tail 이하 | 관측된 pause 사용 |
| hard max-duration, 실제 tail 0 | 0/100/200 ms zero 후보 중 최소 유효값 | zero/trim을 0/0, 100/100, 200/200으로 비교 | 0/100/200 ms zero 후보 중 최소 유효값 |

100 ms는 Deepgram/Qwen의 현재 baseline이고 Soniox의 현재 trim 설정이다. 200 ms는 Soniox가 실제 speech end 뒤 권장하는 대략적인 silence에서 가져온 제한적 후보다. 세 provider 어디에도 500 ms synthetic padding을 공통 기본값으로 정하지 않는다.

## 7. 가설과 우선순위

| ID | 가설 | 예상 사용자 증상 | 우선순위 |
|---|---|---|---:|
| F1 | Deepgram segment final을 utterance final로 오해한다 | 긴 발화 중간·후반 누락, 분리 | 높음 |
| F2 | Soniox timestamp drop 또는 replacement merge가 확정 token을 제거한다 | 한 글자·짧은 단어·문장부호 누락 | 높음 |
| F3 | Soniox가 실제 tail보다 큰 trim을 적용한다 | peer 강제 경계의 마지막 음소 손상 | 높음 |
| F4 | Qwen empty/failed가 pending queue를 해소하지 않는다 | 다음 발화 오매핑, 지연 또는 유실 | 높음 |
| F5 | Qwen normal stop이 final recognition을 기다리지 않는다 | 종료·reset 경계 전체 발화 유실 | 높음 |
| F6 | 공통 FIFO 계약이 여러 outstanding boundary와 late event를 구분하지 못한다 | provider 공통 오매핑·stale publish | 중간~높음 |
| F7 | Deepgram/Qwen의 100 ms zero top-up과 Soniox의 100 ms trim을 같은 tail 정책처럼 취급한다 | provider별 불일치, 강제 경계 품질 저하 | 높음 |
| F8 | 7초 boundary 변경 과정에서 현재의 안정적인 500 ms overlap이 손실된다 | 경계 단어·음절 인식 회귀 | 높음 |
| F9 | 정확히 7초에서 말 중간을 자른다 | 경계 단어·음절 손상 | 높음 |

F1~F6는 모델이 반환한 문장 내부 단어를 더 정확하게 만드는 가설이 아니다. 우선 모델 결과가 앱 경로에서 보존되는지를 검증한다.

## 8. 실행 단계

## P0. 기준 revision과 기존 증거 고정

### 실행

- Git commit과 dirty 상태를 기록한다.
- Deepgram, Soniox, DashScope SDK의 실제 설치 버전을 기록한다.
- self와 peer의 VAD tail 및 peer max-duration 설정을 기록한다.
- 과거 로그 집계는 원본을 수정하지 않고 commit/final/error/reset/toggle 수만 별도 결과로 남긴다.
- 사용자 transcript 원문은 필요하지 않으면 기록하지 않는다.

### 목적

향후 provider 응답 변화와 코드 변경 효과를 혼동하지 않도록 현재 동작을 고정한다. 이 단계에서 대규모 cloud 호출이나 ASR 정확도 실험은 하지 않는다.

## P1. 결정적 event-sequence 기준선

실제 음성 없이 provider callback/message를 scripted sequence로 주입한다. 현재 테스트가 잘못된 provider 의미를 고정하고 있다면 먼저 그 사실을 결과에 기록하고 기대값을 공식 계약에 맞춘다.

### 공통 시나리오

- text가 있는 정상 terminal
- 성공한 빈 terminal
- provider failure
- 같은 local boundary에 terminal이 두 번 오는 경우
- 연속된 local boundary 두 개
- 첫 boundary 실패 후 두 번째 boundary 성공
- 정상 drain 중 terminal 도착
- abort 뒤 늦은 terminal 도착
- provider replacement/reset 전후 generation 교차
- 자연 silence boundary, 7초 max-duration boundary와 continuation boundary
- `observed_tail_ms` 0/100/500과 provider별 padding/trim 결과

### 산출물

- 입력 event 순서
- adapter가 만든 normalized event 순서
- pending queue 전후 상태
- 발표된 transcript 수
- 중복·stale·timeout 여부

## P2. Provider별 좁은 correctness 변경

provider별 변경은 가능한 한 독립된 검토 단위로 구현한다. 한 provider 수정이 다른 provider의 동작을 동시에 바꾸지 않는다.

### P2-B. Peer 7초 boundary

현재 정확히 7,000 ms에서 speech를 자르고 VAD를 non-speech로 reset하는 동작을 다음처럼 바꾼다.

1. hard cap은 현재와 같은 7,000 ms로 유지한다.
2. 5,000~7,000 ms window에서 160~200 ms의 실제 low-speech/pause가 관측되면 그 시점에 continuation boundary를 확정한다.
3. 7,000 ms까지 pause가 없을 때만 hard continuation boundary를 만든다.
4. boundary 뒤에는 현재의 VAD reset, speech 재감지, debounce와 ring pre-roll 재시작 흐름을 유지한다.
5. hard continuation에서는 현재 안정적으로 사용 중인 약 500 ms ring pre-roll overlap을 그대로 유지한다. 0/160/320 ms 축소 후보는 실행하지 않는다.
6. 5~7초 window의 실제 pause boundary에서도 현재 설정된 pre-roll 동작을 임의로 바꾸지 않는다.
7. pre-roll은 현재처럼 다음 segment의 새로운 7초 duration budget에 포함하지 않는다.
8. provider session은 유지하고 boundary별 finalize/commit만 순서대로 처리한다.
9. hard boundary의 synthetic padding은 provider 공통 0/100/200 ms probe 결과로 고른다. 500 ms는 후보에서 제외한다.
10. 현재 중복이 사용자에게 거슬리는 수준이 아니므로 새 seam deduplication을 구현하지 않는다. raw와 발표 결과의 중복 정도만 회귀 지표로 기록한다.

hard cap은 7초 그대로이므로 최대 결과 지연을 늘리지 않는다. 5~7초 구간에서 pause를 먼저 사용해 hard cut 빈도만 낮춘다. window에서 선택된 실제 boundary 시점과 발표 latency를 기록한다.

500 ms overlap은 이미 안정적으로 동작하고 중복도 수용 가능한 기존 동작으로 취급한다. 이번 계획은 overlap 길이 최적화를 범위에서 제외하며, 5~7초 soft-window와 provider final 정책을 바꾸면서 이 동작이 회귀하지 않는지만 확인한다.

이 변경은 `VadGating`의 boundary 선택 조건을 건드리므로 provider adapter 수정과 같은 변경 단위로 섞지 않는다. `soft_pause`와 `max_duration` reason 및 관측된 tail을 provider까지 보존하고, self VAD에는 적용하지 않는다.

### P2-D. Deepgram

1. `is_final` segment를 manual `Finalize` 요청 단위 buffer에 누적한다.
2. 개별 segment 수신만으로 local utterance terminal event를 발표하지 않는다.
3. `speech_final`은 현재 `endpointing=False` 구성에서 manual boundary 대체물로 가정하지 않는다.
4. `from_finalize`를 우선 fence 후보로 사용한다.
5. audio byte가 없거나 `from_finalize` 응답이 오지 않는 경우를 위해 bounded fallback을 설계한다.
6. 같은 finalize 요청에서 terminal event가 두 번 발생하지 않도록 상태를 단조롭게 전환한다.
7. 정상 stop에서는 전송 큐의 audio와 finalize control이 처리된 뒤 결과 fence를 기다린다.
8. 즉시 abort에서는 buffer와 pending outcome을 cancel하고 늦은 event를 버린다.
9. 자연 boundary에서는 이미 들어온 약 500 ms tail을 사용하고 추가 zero를 보내지 않는다.
10. max-duration boundary의 현재 100 ms zero top-up은 provider 요구로 간주하지 않고 0/100/200 ms probe 결과로 결정한다.

Deepgram fallback은 공식 문서만으로 임의의 millisecond 값을 확정하지 않는다. P3 live trace에서 `from_finalize`가 오지 않는 조건과 응답 지연을 확인한 뒤 가장 짧은 bounded timeout 또는 명시적인 request serialization을 선택한다.

### P2-S. Soniox

1. final token을 수신 순서대로 append한다.
2. `<fin>`에서 해당 finalize 요청을 정확히 한 번 flush한다.
3. `end_ms <= previous_end_ms`만을 근거로 token을 삭제하지 않는다.
4. final token이 교정·반복된다는 가정의 replacement merge를 제거한다.
5. 빈 `<fin>`도 pending finalize 요청 하나를 해소한다.
6. effective trim을 `min(configured_trim, observed_trailing_silence)`로 제한한다.
7. peer max-duration처럼 observed tail과 injected padding이 모두 0이면 trim도 0으로 전달한다.
8. 예상과 달리 provider가 final token을 반복한다면 destructive merge를 복구하지 말고 raw sequence와 SDK/provider version을 결과에 남긴다.
9. 자연 boundary에서는 현재 설정 100 ms trim을 baseline으로 유지하되 0/100 ms를 비교한다. 200 ms는 실제 tail이 최소 200 ms일 때만 제한적으로 비교한다.
10. max-duration boundary에서는 실제 zero PCM/trim을 `0/0`, `100/100`, `200/200` ms로 비교한다. trim option만 양수로 보내는 조건은 후보에서 제외한다.

일반 self 발화에는 물리적 silence를 추가하지 않는다. 공식 약 200 ms 권장은 finalize 전 실제 silence에 관한 것이며, 현재 자연 boundary에는 이미 약 500 ms가 들어 있다. SDK 예시의 `trailing_silence_ms=500`은 기존 silence trimming 예시이므로 500 ms를 기다리거나 추가하라는 뜻으로 해석하지 않는다.

### P2-Q. Qwen ASR Flash Realtime

1. transcript가 빈 `completed`도 pending boundary 하나를 해소하는 terminal outcome으로 만든다.
2. `conversation.item.input_audio_transcription.failed`를 구조화해 기록하고 pending boundary 하나를 실패 outcome으로 해소한다.
3. commit `event_id`, committed/completed/failed `item_id`를 가능한 범위에서 동일 요청 흐름에 연결한다.
4. 정상 drain에서는 SDK `end_session()`을 사용해 final recognition과 session 종료를 기다린다.
5. `end_session()`의 blocking wait가 event loop를 막지 않도록 기존 thread/executor ownership 안에서 수행한다.
6. 명시적 toggle-off는 즉시 abort 의미를 유지하고 pending 항목을 cancel로 해소한다.
7. abort 또는 provider 교체 후 들어온 callback이 새 generation에 발표되지 않게 한다.
8. 자연 boundary에서는 이미 들어온 약 500 ms tail을 사용하고 추가 zero를 보내지 않는다.
9. max-duration boundary의 현재 100 ms zero top-up은 공식 commit 요구가 아니므로 0/100/200 ms probe 결과로 결정한다.

현재 공통 event 형식을 유지하는 첫 패치에서는 empty/failed를 사용자에게 빈 자막으로 발표하지 않으면서 내부 queue만 전진시킬 수 있다. 실패 상태를 UI나 telemetry에 노출해야 할 제품 요구가 생기면 공통 terminal status 확장을 별도 결정한다.

## P3. 최소 live wire 확인

### 원칙

- ASR 인식률 benchmark가 아니라 실제 provider event 순서를 확인하는 최소 호출만 수행한다.
- provider마다 짧은 발화, 긴 발화, 연속 두 발화와 빈/거의 빈 finalize를 합쳐 10~20회 이내로 제한한다.
- 같은 audio ID와 chunk timing을 수정 전후에 사용한다.
- transcript 내용보다 event metadata, text hash와 길이를 우선 저장한다.

### 공통 tail 확인

두 boundary를 섞지 않는다.

| 조건 | 실제 기다림 | 이미 전송된 tail | 추가 zero 후보 | Soniox trim 후보 |
|---|---:|---:|---:|---:|
| 자연 speech end | 기본 VAD hangover 약 500 ms | 약 500 ms | 0 | 0/100, 필요 시 200 |
| 5~7초 soft window의 pause | 관측 pause 160~200 ms | 160~200 ms | 0 | 0/관측 tail 이하 |
| 7초 hard continuation | 0 ms | 0 ms | 0/100/200 | injected padding 이하 |

`injected_padding_ms`는 zero PCM을 큐에 즉시 넣는 것이므로 같은 길이의 wall-clock sleep과 같지 않다. 다만 provider가 처리할 audio duration과 timeline은 늘어나므로 latency와 인식 결과를 함께 기록한다. `declared_trim_ms`는 기존 tail을 trim하는 metadata이므로 wall-clock wait를 추가하지 않는다. 실제 사용자 대기에서 가장 큰 고정 비용은 자연 boundary의 VAD hangover 약 500 ms다.

hard continuation padding을 비교할 때 overlap은 현재 값인 약 500 ms로 고정한다. overlap과 padding을 동시에 바꾸지 않는다.

### Hard continuation overlap 회귀 확인

말 또는 단어 도중 정확히 7초 hard cap에 닿는 동일 PCM으로 현재 overlap 보존 여부만 확인한다.

1. 다음 segment에 약 500 ms pre-roll이 전달되는지 기록한다.
2. pre-roll이 다음 7초 duration budget에서 제외되는지 확인한다.
3. 현재 수준보다 경계 단어 누락이나 중복 prefix가 증가하지 않는지 확인한다.
4. overlap을 고정한 상태에서만 padding 0/100/200 ms를 비교한다.

overlap 0/160/320 ms 최적화와 새 seam deduplication은 이 계획의 실행 범위에서 제외한다. 500 ms overlap 자체에서 사용자에게 거슬리는 회귀가 새로 관찰될 때만 별도 문제로 다시 연다.

### Deepgram 필수 확인

- 한 manual `Finalize`에 `is_final` segment가 몇 개 오는가
- 각 segment의 `start`, `duration`, `is_final`, `speech_final`, `from_finalize`
- 유효 audio가 있을 때 `from_finalize` 도착률과 지연
- audio가 없거나 이미 처리된 경우 별도 응답 유무
- 연속 finalize 두 개의 fence 순서
- 자연 boundary에서 추가 zero가 0인지 확인하고, hard continuation에서 zero padding 0/100/200 ms의 raw aggregate와 terminal latency 비교

### Soniox 필수 확인

- final token의 수신 순서와 `start_ms`, `end_ms`
- 동일 또는 역행 timestamp 존재 여부
- `<fin>`당 pending finalize 소비 수
- actual tail 0/약 500 ms에서 전달한 trim 값
- 자연 boundary의 같은 PCM에서 trim 0/100 ms가 raw provider text와 latency를 바꾸는지 보는 제한적 probe
- hard continuation의 같은 PCM에서 실제 zero/trim `0/0`, `100/100`, `200/200` ms 비교

마지막 비교는 native trim의 잠재적 인식 영향만 확인한다. 작은 표본으로 CER 개선을 일반화하지 않는다.

### Qwen 필수 확인

- commit `event_id`와 committed/completed/failed `item_id`
- commit당 terminal outcome 수
- 빈 completed와 transcription failed 뒤 pending queue 상태
- `end_session()` 시작부터 completed와 `session.finished`까지의 순서와 시간
- abort 뒤 late callback 발표 여부
- 자연 boundary에서 추가 zero가 0인지 확인하고, hard continuation에서 zero padding 0/100/200 ms의 completed text와 terminal latency 비교

## P4. Lifecycle과 concurrency 검증

다음은 실제 음성 대신 짧은 fixture 또는 scripted callback으로 우선 수행한다.

| 시나리오 | 기대 결과 |
|---|---|
| 말 끝 직후 정상 provider 교체 | 이미 수락한 결과를 bounded drain 후 한 번만 발표 |
| 말 끝 직후 앱 정상 종료 | 마지막 결과 회수 또는 명시적 timeout outcome |
| 말 끝 직후 toggle-off | pending cancel, 늦은 결과 미발표 |
| 0/100/300/1000 ms 뒤 toggle-off | 각 run의 abort 시점과 결과 여부가 일관됨 |
| 180초 session reset 경계 | old generation 결과가 새 session에 매핑되지 않음 |
| 연속 speech boundary 두 개 | 순서가 유지되고 각각 terminal outcome 하나 |
| 첫 요청 empty/failed, 다음 요청 성공 | 두 번째 text가 첫 utterance ID를 소비하지 않음 |
| provider terminal 중복 | 사용자 transcript 한 번만 발표 |
| terminal timeout | queue가 무기한 남지 않고 상태가 관측 가능함 |
| 5~7초에 실제 pause 존재 | pause에서 boundary, 기존 VAD 재시작과 pre-roll 정책 유지 |
| 7초까지 pause 없는 연속 speech | hard continuation 1회, 기존 약 500 ms pre-roll overlap으로 재개 |
| 연속 speech 30초 | unique audio와 기존 pre-roll/padding을 분리 계측하고 현재보다 중복·누락이 증가하지 않음 |

## P5. 공통 contract 결정 gate

### adapter-only로 종료하는 조건

- 각 adapter가 정상 boundary마다 terminal outcome 정확히 하나를 보장한다.
- 연속 boundary, empty, failed, reset과 abort 테스트에서 FIFO 오매핑이 없다.
- provider별 ID를 core에 노출하지 않아도 late result가 generation 검증으로 차단된다.

### 공통 contract 확장이 필요한 조건

- 여러 outstanding boundary의 응답이 provider에서 순서대로 온다는 보장이 없다.
- adapter 내부 serialization이 audio 전송 중단이나 과도한 latency를 만든다.
- terminal failure/cancel을 빈 final로 표현하면서 관측 가능성 또는 제품 동작이 손실된다.
- reset 경계에서 FIFO만으로 old/new 요청을 안전하게 구분할 수 없다.

필요한 경우 별도 변경 단위에서 다음 일반화된 형태를 검토한다.

- controller가 생성한 `boundary_id`를 `on_speech_end()`에 전달
- normalized event에 `boundary_id`와 `terminal_status` 추가
- provider item/event ID는 adapter 내부 metadata로 유지하고 local `boundary_id`에 연결
- core에 Deepgram `from_finalize`, Soniox `<fin>`, Qwen `item_id` 같은 provider 전용 개념을 노출하지 않음

공통 contract 변경은 provider adapter의 책임을 core로 옮기지 않아야 한다. provider별 wire 의미는 adapter가 소유하고 core는 correlation과 generic terminal status만 소유한다.

## 9. Provider별 결정적 테스트 행렬

### Deepgram

| 입력 event | 기대 normalized 결과 |
|---|---|
| `is_final` segment 1개 + finalize fence | 합친 text terminal 1개 |
| `is_final` segment 여러 개 + fence | 순서대로 결합된 text terminal 1개 |
| 빈 segment + fence | empty terminal 1개 |
| segment 뒤 `speech_final=False` | fence 전 사용자 final 없음 |
| `from_finalize` 없는 no-audio finalize | bounded empty terminal 또는 no-request fast path |
| stop과 fence 경합 | completed 또는 canceled 중 하나만 발생 |
| natural tail 약 500 ms | 추가 zero 0 |
| hard continuation tail 0 | 선택된 0/100/200 ms만 추가 |

기존 테스트가 `is_final=True`, `speech_final=False`만으로 앱 final을 기대한다면 공식 의미에 맞게 교체한다.

### Soniox

| 입력 event | 기대 normalized 결과 |
|---|---|
| final token 여러 개 + `<fin>` | 수신 순서 text terminal 1개 |
| 같은 `end_ms` token 두 개 | 두 token 모두 보존 |
| 감소한 `end_ms` token | token 보존 및 진단, 임의 삭제 없음 |
| 빈 `<fin>` | empty terminal 1개 |
| 연속 finalize 두 개 | `<fin>`마다 terminal 하나 |
| actual tail 0/configured 100 | provider trim 0 |
| actual tail 500/configured 100 | provider trim 최대 100 |
| hard continuation padding 100 | 실제 zero 100, trim은 최대 100 |

### Qwen

| 입력 event | 기대 normalized 결과 |
|---|---|
| completed with text | text terminal 1개 |
| completed with empty text | empty terminal 1개 |
| transcription failed | failed terminal 1개 또는 내부 empty ack 1개 |
| committed 뒤 completed | 같은 pending boundary 해소 |
| 두 commit 순차 completed | 각 boundary를 정확히 한 번 해소 |
| end_session 중 completed | 결과 처리 후 session 종료 |
| abort 뒤 completed | 사용자 발표 없음, canceled 상태 유지 |
| natural tail 약 500 ms | 추가 zero 0 |
| hard continuation tail 0 | 선택된 0/100/200 ms만 추가 |

## 10. 측정 지표

### Protocol correctness

- accepted boundary 수
- completed-text, completed-empty, failed, canceled 수
- boundary당 terminal outcome 수
- provider 확정 text hash/길이와 앱 text hash/길이
- 누락·중복·순서 역전·다른 utterance ID 매핑 수
- 종료 후 pending queue 크기
- late/stale event 차단 수
- boundary reason별 `observed_tail_ms`, `injected_padding_ms`, `declared_trim_ms`
- max-duration continuation 전후 중복·누락 PCM sample 수
- continuation의 ring `pre_roll_ms`
- seam에 나타난 중복 token/문자 수
- hard-boundary 마지막·첫 단어 보존율

### Timing

- speech end부터 provider fence까지
- fence부터 normalized terminal event까지
- 정상 drain 시작부터 완료까지
- VAD boundary wait와 synthetic padding 처리 시간을 분리한 latency
- soft target부터 실제 continuation boundary까지의 추가 지연
- timeout 수와 timeout 조건
- steady-state transcript latency 변화

### 제한적 인식 영향

- provider raw text에 이미 없던 글자를 adapter 누락으로 계산하지 않는다.
- raw provider text에는 있지만 앱 text에서 사라진 문자·token만 text preservation 실패로 계산한다.
- Soniox native trim probe에서는 마지막 글자/단어 보존만 보고하며 일반 CER 개선 주장을 하지 않는다.
- Qwen completed raw text와 앱 text가 같다면 문장 내부 오인식은 이 계획의 원인으로 분류하지 않는다.

## 11. 사전 판정 규칙

### 명백한 correctness 실패

- 정상 boundary 하나에 terminal outcome이 0개 또는 2개 이상
- empty/failed 뒤 pending queue가 남아 다음 결과를 소비
- provider 확정 token/segment가 adapter에서 사라짐
- old generation 결과가 새 session에서 발표됨
- normal drain이 마지막 결과를 기다리지 않고 연결을 닫음
- actual tail보다 큰 Soniox trim 전달
- natural boundary에서 Deepgram/Qwen이 이미 충분한 tail에 zero를 중복 추가
- hard continuation의 기존 약 500 ms pre-roll이 사라지거나 예상보다 크게 증가
- hard continuation마다 500 ms synthetic padding을 기본 적용
- bounded timeout 없이 pending 또는 stop이 무기한 대기

### 수정 성공

- scripted 행렬의 모든 terminal-count와 queue 불변조건 통과
- raw provider aggregate와 normalized text가 허용된 whitespace 정규화 외에는 동일
- 정상 drain과 abort 결과가 명시적으로 구분됨
- 연속 boundary와 reset 경계에서 오매핑·중복 발표 없음
- 자연 boundary와 hard continuation의 tail accounting이 구분되고 세 provider 모두 기록됨
- 30초 연속 speech에서 unique PCM, 기존 pre-roll과 padding이 구분되며 현재보다 중복·누락이 증가하지 않음
- 5~7초 soft window가 hard cut 빈도를 낮추고 7초 hard cap을 넘기지 않음
- 기존 약 500 ms overlap에서 경계 단어 보존과 사용자 가시 중복이 회귀하지 않음
- 수정 전보다 종료 latency가 늘면 그 증가가 실제 terminal fence 대기와 연결되며 상한이 있음

### 인식률에 관한 표현 제한

- adapter가 원래 provider text를 보존하게 된 효과는 `result-preservation improvement`로 보고한다.
- provider raw text 자체가 개선된 경우에만 final timing 또는 trim의 `recognition-impact signal`로 보고한다.
- 소규모 live trace만으로 일반 인식률 향상이나 CER 개선을 주장하지 않는다.
- 세 provider 전체의 문장 내부 오인식을 finalization 하나로 설명하지 않는다.

## 12. 구현 변경 단위

권장 변경 순서는 다음과 같다.

1. 현재 provider event sequence를 고정하는 집중 테스트와 최소 metadata 진단
2. 세 provider 공통 tail accounting과 boundary reason 진단
3. Soniox append-only token 처리와 truthful trim
4. Qwen empty/failed outcome과 graceful `end_session()`
5. Deepgram finalize aggregate와 실제 wire 결과에 근거한 fence/fallback
6. peer 5~7초 soft-window와 7초 hard cap을 구현하되 기존 약 500 ms pre-roll overlap 보존
7. provider별 hard-continuation padding 최소 후보 결정
8. lifecycle/reset/toggle 누적 검증
9. adapter-only 불변조건으로 해결되지 않을 때만 공통 STT contract 확장

각 변경 단위는 다음 원칙을 지킨다.

- 다른 provider의 동작을 동시에 바꾸지 않는다.
- input gain, PCM tail, VAD threshold와 downmix를 같이 바꾸지 않는다.
- provider별 tail 후보를 비교할 때 boundary reason과 실제 PCM은 고정한다.
- production UI나 설정을 추가하지 않는다.
- 코드와 무관한 테스트 정비를 섞지 않는다.
- merge, push, publish와 release는 별도 명시적 승인 없이는 수행하지 않는다.

## 13. 최소 산출물

### Run manifest

- run ID와 일시
- Git commit과 dirty 상태
- provider, SDK version, model ID와 endpoint
- self 또는 peer 채널
- local boundary ID 또는 실행 순번
- 종료 원인 `drain`, `reset`, `replacement`, `abort`, `error`
- provider option과 실제 관측 tail

### Event result

- provider event 종류와 순번
- provider request/item metadata
- text 원문 대신 가능한 경우 길이와 hash
- normalized terminal outcome
- pending queue 전후 크기
- 발표 여부와 stale-drop 여부
- latency와 timeout/error

### 최종 `RESULTS.md`

provider와 가설별로 다음 중 하나를 기록한다.

- `confirmed and corrected`
- `not reproduced but contract corrected`
- `not supported`
- `inconclusive`
- `requires shared contract change`

그리고 수정 전후 event sequence, 실패 사례, latency 비용, 남은 위험과 공통 contract 결정 결과를 남긴다.

## 14. 비용과 실행 제한

- 이 계획은 CPU 장시간 추론을 요구하지 않는다.
- scripted event test와 provider별 10~20회 이내의 live trace를 우선한다.
- cloud 호출을 모든 PCM level이나 모델 조합으로 확장하지 않는다.
- recognition-impact probe는 말끝 민감 문장 소수로 제한한다.
- 별도 benchmark harness, dashboard와 corpus download를 만들지 않는다.
- 장시간 CPU ASR가 필요해진다면 범위가 입력 품질 계획으로 넘어간 것이므로 이 계획에서 실행하지 않는다.

## 15. Architecture drift 감시

다음은 drift 위험이다.

- Deepgram, Soniox 또는 Qwen 전용 event 이름이 core normalized contract에 노출됨
- provider adapter가 controller의 private pending queue를 직접 변경함
- controller가 provider WebSocket 또는 SDK lifecycle을 직접 소유함
- blocking provider shutdown이 application event loop에서 실행됨
- runtime replacement 뒤 old adapter callback이 generation 검증 없이 현재 owner를 변경함
- finalization 수정 명목으로 capture/VAD ownership까지 이동함

공통 `boundary_id` 또는 generic `terminal_status` 확장은 provider-specific 의미를 core에 노출하지 않는다면 기존 architecture와 정렬될 수 있다. 실제 구현에서 owner나 port 경계가 바뀌면 결과 문서와 사용자 보고에 suspected architecture drift를 명시한다.

## 16. 참고 자료

- [Deepgram: Configure Endpointing and Interim Results](https://developers.deepgram.com/docs/understand-endpointing-interim-results)
- [Deepgram: Finalize](https://developers.deepgram.com/docs/finalize)
- [Deepgram: Live Streaming API](https://developers.deepgram.com/reference/speech-to-text/listen-streaming)
- [Soniox: Real-time Transcription](https://soniox.com/docs/stt/rt/real-time-transcription)
- [Soniox: Manual Finalization](https://soniox.com/docs/stt/rt/manual-finalization)
- [Soniox: WebSocket API](https://soniox.com/docs/api-reference/stt/websocket-api)
- [Soniox: Real-time Transcription with Web SDK](https://soniox.com/docs/sdk/web-SDK/stt/realtime-transcription)
- [Qwen: Realtime ASR Client Events](https://www.alibabacloud.com/help/en/model-studio/qwen-asr-realtime-client-events)
- [Qwen: Realtime ASR Server Events](https://www.alibabacloud.com/help/en/model-studio/server-events)
- [Qwen: Realtime ASR Python SDK](https://www.alibabacloud.com/help/id/model-studio/qwen-asr-realtime-python-sdk)
- [Qwen: Realtime ASR Interaction Process](https://www.alibabacloud.com/help/en/model-studio/qwen-asr-realtime-interaction-process)

공식 provider 문서는 protocol 의미와 수정 방향의 근거다. 실제 앱에서의 발생 빈도, 종료 지연과 result-preservation 효과는 이 계획의 scripted event와 최소 live trace로만 판정한다.
