# 화자 귀속·조기 경계 정책 대 구현 전수 감사

## 1. 결론

**조기 경계가 전혀 없는 것은 아니다. 그러나 승인된 조기 진행 방식 중 중요한 한 갈래가 구현되지 않았다.**

- 로컬 ASR은 아직 제출하지 않은 PCM을 화자 경계에서 닫고 다음 입력을 보존한다.
- timed streaming은 공급자가 **이미 확정한** 텍스트를 화자 경계에 맞추어 먼저 내보낼 수 있다.
- 반면 **화자 경계는 확인됐지만 필요한 final이 아직 없을 때, 공급자의 합법적인 제어로 그 final을 요청하는 연결이 없다.** Deepgram의 `Finalize`가 대표적인 누락이다. 경계 이벤트가 생성되는 것과 이 제어가 실행되는 것은 다르다.
- Gemini는 조기·복수 authoritative final을 처리하고 번역할 수 있다. 무조건 최종 연결 종료까지 기다리는 구현은 아니다. 하지만 final과 원본 오디오의 대응 관계가 없으면 Nemotron 화자를 붙일 수 없다. 현재 관측된 해당 결과가 회색인 것은 보수 정책에 맞는다.
- 번역을 기다리게 하지 않고 최초 가독 표시 후 최대 1초 동안 한 번 귀속하는 기능은 구현됐다. 이 기능이 없는 final, 없는 시간 대응, 없는 조기 입력 제어를 대신하지는 않는다.

| 분류 | 확인한 차이 | 판단 |
|---|---|---|
| D1 | Nemotron 경계에서 missing-final 진행 제어로 이어지는 연결 부재 | 핵심 구현 누락. 특히 Deepgram의 선택된 정책과 불일치 |
| D2 | Gemini 독립 final의 도착 시각으로 60초 source-idle 만료를 연장 | 명시 정책과 불일치. 현재 엔진 실행으로 확인 |
| D3 | Gemini native VoiceActivity의 종류·오디오 offset 등 진단 사실 소실 | 보존 계약의 일부 누락. 이를 보존해도 final 매핑은 자동 복구되지 않음 |
| D4 | 로컬 결과에 실제 선택된 delegate/model/native provenance가 충분히 남지 않음 | 옵션 고정은 구현. 결과 출처 기록은 불완전 |
| D5 | Parakeet의 raw token/emission/advance 정보 보존 미구현 | 원래 선택된 보존 항목의 잔여 작업. 이번 승인에서 제외한 음향 누락 수리와 구별 |
| D6 | #216 R4 수락 기록이 missing-final 제어 부재를 완료로 취급 | 완료·문서 범위의 불일치. #264 수락으로 원래 프로그램 전체가 완료되지 않음 |
| 정당한 변경 | Qwen Peer의 native sentence별 조기 번역을 막음 | #255/PR #257의 명시 개정. 원래 #212 문구로 되돌리면 오히려 최신 정책 위반 |
| 허용된 제한 | 매핑 없는 Gemini final은 회색, provisional interim은 번역하지 않음 | 현재 finality·소유권 정책 준수 |
| 검증 공백 | 전체 경로의 음향 품질, 최신 프로파일 비교, 일부 실제 렌더·VR 경로 | 미검증이지 자동으로 구현 결함 또는 성공은 아님 |

이 문서는 **감사 보고서**다. 제품 코드를 수정하거나 정책을 새로 승인하지 않는다. 코드·관측·허용된 제한·후속 범위를 분리한다. “전수”는 관련 요구사항을 빠짐없이 판정 대상으로 삼았다는 뜻이지 모든 입력·장치·경쟁 조건에서 정확성을 인증한다는 뜻이 아니다.

## 2. 기준과 범위

### 2.1 고정 후보

| 항목 | 기준 |
|---|---|
| 감사 HEAD | `96929f2644a3c0c0f82a9a8f49f252fcbf9b4007` |
| 현재 제품 코드 핀 | `6c826f420205fd748a65431108cceb845da56eab` |
| 후속 증거 핀 | `b5e59dfdef6dc2ce4b36063368d1b296864e7d44` |
| 이슈 상태 조회 | 2026-10-18 UTC; 최초 메타데이터 스냅샷 `17:18:02Z` |
| 변경 범위 | 이 Markdown 보고서만. 구현·테스트·GitHub 게시·PR·merge 변경 없음 |
| 신규 실행 | 네트워크 없는 통제된 production-class 관측. 새 모델/API/GUI/VR 실행 없음 |

코드 경로와 행 번호는 이 HEAD 기준이다. 아직 공개되지 않은 로컬 커밋을 GitHub에서 조회할 수 있다고 가정하지 않는다. 경로는 저장소 상대 경로이며, 코드 근거의 축약 접두어는 다음과 같다.

- `STT/` = `src/puripuly_heart/core/stt/`
- `ORCH/` = `src/puripuly_heart/core/orchestrator/`
- `PROVIDER/` = `src/puripuly_heart/providers/stt/`
- `SPEAKER/` = `src/puripuly_heart/core/speaker/`
- `OVERLAY/` = `src/puripuly_heart/core/overlay/`
- `EVIDENCE/` = `native/nemotron/`

### 2.2 권한 우선순위

기본은 [#211](https://github.com/kapitalismho/PuriPuly-heart/issues/211)과 하위 이슈 [#212](https://github.com/kapitalismho/PuriPuly-heart/issues/212)–[#219](https://github.com/kapitalismho/PuriPuly-heart/issues/219)다. 단, 나중에 명시적으로 승인된 변경은 옛 문구보다 우선한다.

| 근거 | 현재 해석 |
|---|---|
| [#212 고정 provider matrix](https://github.com/kapitalismho/PuriPuly-heart/blob/c4e48975143c7b131b64bb6ede2801e9f937e2a9/docs/asr-speaker-policy.md) | 경로별 P/T/C, native 사실, 합법 제어의 원래 기준. 현재 로컬 트리에 이 파일이 없어 원격 고정본을 확인 |
| [#194](https://github.com/kapitalismho/PuriPuly-heart/issues/194), B′ 및 #212 | Gemini independent final 수용, final-only, 오디오·텍스트 종료 분리. 오래된 ACK barrier/interim fallback을 대체 |
| [#218 후속 기록](https://github.com/kapitalismho/PuriPuly-heart/issues/218#issuecomment-6243553002) | 귀속 시점·대표 화자·재사용 문맥 정책 및 한정 관측 |
| [#255](https://github.com/kapitalismho/PuriPuly-heart/issues/255), [PR #257](https://github.com/kapitalismho/PuriPuly-heart/pull/257) | QWEN-HANDOVER-2. Peer는 native task terminal에서 한 번 조립; native sentence ending으로 번역하지 않음 |
| [#264](https://github.com/kapitalismho/PuriPuly-heart/issues/264) | 공유 증거 수명, 경계 확인, 무대기 번역·최대 1초 단회 late assignment, 한정 Gemini 조사 |
| [#264 구현 기록](https://github.com/kapitalismho/PuriPuly-heart/issues/264#issuecomment-6252285261), [범위 수락](https://github.com/kapitalismho/PuriPuly-heart/issues/264#issuecomment-6252391329) | 해당 deliverable의 수락. #211/#218 전체 품질·프로파일·VR 성공으로 확대하지 않음 |
| [현재 계획](issue218-speaker-attribution-plan.md), [구조 문서](architecture.md) | 구현 지도와 이력. 이 문서가 더 좁게 쓰였다는 이유만으로 이슈의 미완료 요구가 취소되지는 않음 |

조회 시 #211/#218/#264는 OPEN, #212–#217은 CLOSED/COMPLETED, #219는 OPEN이었다. Project는 각각 #211 Backlog, #218/#264 In progress였다. #255는 CLOSED, PR #257은 MERGED다. 이슈 상태와 구현 완전성은 별개다.

### 2.3 혼동하면 안 되는 개념

| 개념 | 의미 | 보장하지 않는 것 |
|---|---|---|
| 화자 경계 검출 | 캡처 좌표에서 A→B 근거를 확인 | 공급자 입력이 실제 닫혔음 |
| 로컬 PCM 조기 절단 | 미제출 A 입력을 닫고 B를 후속 요청에 보존 | 이미 제출한 파형을 소급 수정 |
| 기존 final 조기 release | 이미 확정된 일부 텍스트를 먼저 소스로 접수 | 공급자가 더 빨리 final을 생성 |
| missing-final progress | 아직 없는 final을 얻으려고 lawful 제어 실행 | 확정 텍스트 내용·단어 생존 보장 |
| 가변 partial/interim | 나중에 철회·수정될 수 있는 가설 | 번역 가능한 authoritative source |
| 확정 prefix | 전체 turn이 끝나지 않았어도 공급자가 확정한 부분 | 그 prefix에 정확한 단어 시간이 있음 |
| 화자 귀속 | 검증된 오디오 대응에 Nemotron 증거 결합 | ASR이 누락한 텍스트 복구 |
| 1초 late assignment | 이미 읽을 수 있는 미확정 카드의 색을 한 번 해결 | 1초 대기, 재번역, A→B 재색칠, 카드 재배치 |

사용자가 말한 “2티어”는 이 보고서에서 **시간 대응 정보가 제한된 ASR 활용군**으로 설명한다. 조사한 관련 이슈와 고정 matrix에서 문자 그대로의 `Tier2`를 공통 계약 명칭으로 찾지는 못했다. 따라서 과거 Stage 2, 무료 서비스 tier, 모델 정밀도와 동일시하지 않는다. 공식 P/T/C 구분과 개별 메타데이터 능력을 함께 봐야 한다.

## 3. 조기 경계: 구현된 부분과 빠진 부분

### 3.1 로컬은 실제로 입력을 일찍 닫는다

`STT/batch_speaker.py`는 보유한 미제출 PCM에 경계를 적용한다. native 요청의 PCM·범위·옵션은 제출 뒤 고정되며 B의 미소비 입력은 다음 요청에 남는다. 이미 디코딩한 부모 전체를 다시 디코딩하는 방식이 아니다.

현재 확인 규칙은 단일 clear frame만으로 자르는 대신 서로 다른 native frame의 연속 지지를 요구한다. 경쟁 슬롯의 근거를 긴 A의 지지량으로 상쇄하지 않는다. 귀속 거부와 경계 commit은 서로 다른 결정이다.

두 clear frame과 10ms native frame 간격을 곱한 20ms는 **관측 근거 길이**이지 20ms 안에 클라우드 final/번역/화면이 나온다는 약속이 아니다. 프로파일 청크, lookahead, 증거 도착, 애플리케이션 처리, ASR이 각각 영향을 준다. `0.64`와 `1.04`는 diarizer geometry이며 Parakeet 디코더의 정확도 설정이 아니다.

### 3.2 streaming에는 두 갈래가 필요하다

#211 R1, #212의 Deepgram S2, #216 R4는 다음을 구분한다.

1. **이미 final이 있음:** 검증된 시간 대응으로 prefix를 접수하고 번역한다.
2. **필요한 final이 없음:** 경로가 허용하는 진행 제어를 현재 전송 frontier에 실행해 final을 요청한다. 뒤의 입력·중복 제어·자연 종료와의 경쟁을 별도로 관리한다.

현재 두 번째 갈래가 연결되지 않았다.

| 실제 경로 | 코드 근거 | 하는 일 |
|---|---|---|
| 화자 증거 수신 | `STT/streaming_speaker.py:210–240` | 경계를 기록하고 `_release_available` 호출 |
| release 판정 | 같은 파일 `1085–1093` | latest가 없거나 mapped/timed가 아니면 release 불가 |
| 엔진 전달 | `STT/scoped_engine.py:1753–1757` | `STTProviderAudioBoundary` 전달 |
| 앱 전달 | `ORCH/translation_channel_callbacks.py:253–258`; `peer_translation_channel.py:485–489` | 공통 source ledger에 경계 전달 |
| ledger | `STT/backend.py:361–374` | 경계를 기록. provider 제어를 요청하지 않음 |
| 실제 seal 경로 | `STT/scoped_engine.py:1175–1197` | 기존 로컬 `SpeechEnd`를 통해 seal |
| 실제 Deepgram 제어 | `PROVIDER/deepgram.py:822–839` | 기존 seal 경로의 `Finalize` 전송 |

`STTSourceCapabilities.optional_progress` 필드가 있다는 사실도 구현 증거가 아니다. 기본값은 false이고 생산 코드에서 경계 진행 actuator가 이를 소비하지 않는다.

새로운 무네트워크 실행에서는 **가변 partial만 있을 때 화자 경계는 생겼지만 source release와 provider seal은 모두 없었다.** 이후 native 확정 prefix가 들어오자 같은 경계에서 `No. `를 release하고 최종 tail `Fifty.`를 보존했다. 따라서 “partial을 전혀 활용하지 않는다”도, “경계로 provider를 조기 확정시킨다”도 현재 코드에 대한 정확한 설명이 아니다.

### 3.3 언제 빠졌는가

“작동하던 Nemotron 클라우드 조기 절단을 최근 삭제했다”는 근거는 찾지 못했다. 확인한 이력은 **첫 streaming 통합에서 기존-final release만 채택하고, missing-final 제어 연결을 만들지 않은 뒤 그 범위를 후속 작업이 이어받은 것**이다.

| 시점 / 커밋 | 확인한 사실 |
|---|---|
| 2026-10-06, 정책 커밋 `5f650fb6` | 기존 final 활용과 필요한 lawful input progress를 구분한 정책 |
| 2026-10-12 22:58:47 KST, `90b33d591cf1d55133caafa81eacbabd46de47a8` | 첫 `StreamingSpeakerSession` 통합. 구조 문서에 wrapper가 추가 physical cut을 하지 않는다고 명시. 현재 rebase 이력의 해시이며 이전 기록은 `0ba21054` |
| #216 수락 기록 | `EVIDENCE/issue216_acceptance.json:46`은 “No new optional physical speaker cuts are advertised. Existing finals and natural/hardcut/drain paths are selected.”를 R4 coverage로 기재 |
| 2026-10-13 16:53:24 KST, `d95e60696e0158d027fbf9068679fb3b7ed1f6b1` | 후속 계획 Outcome B에 “Do not alter physical cloud input control.”. 현재 계획의 역사 구간 `143`행 |
| 2026-10-18, #264 초기 `8c01ad68` 이후 | 공유 증거·경계 확인·late assignment를 보강했지만 이 좁아진 클라우드 제어 범위를 계승 |

이것은 제품 코드의 조기 prefix 성공과 모순되지 않는다. 과거 live Deepgram에서 final이 먼저 도착한 사례와 현재 local GPU의 이른 논리적 출력은 실제 성과다. 하지만 **없는 final을 더 빨리 생성하도록 요청했다는 증거는 아니다.**

현재 로컬 GPU 사례의 논리적 번역 출시는 safety 대비 약 1.108초, 자연 종료 대비 약 2.054초 빨랐다. 해당 사례의 값이지 일반 지연 보장이나 현재 pixel latency가 아니다. #216의 이전 early-source 관측은 translation call이 0이었던 별도 실행이므로 현재 번역 관측과 합치지 않는다.

### 3.4 보완 방향

경계 이벤트를 provider socket에 직접 연결하는 작은 편법으로 끝내면 안 된다. 기존 source/epoch/전송 frontier 소유자가 다음을 관리해야 한다.

- 새 경계가 **기존 final release로 충족됐는지** 먼저 판정한다.
- 충족되지 않았다면 해당 경로의 허용된 제어만 현재 전송 frontier에 요청한다.
- 같은 frontier의 중복 요청과 자연 종료·hardcut·drain 경쟁을 합친다.
- 제어 뒤에도 후속 오디오를 보존하고, 반환 final/tail을 공통 ledger에서 한 번만 접수한다.
- 늦게 확인한 경계를 이미 전송한 파형의 과거 위치에 소급 적용했다고 표시하지 않는다.
- 실제 제어 시점, 적용 frontier, final 도착, 유용한 source 출시를 따로 관측한다.

이는 모든 공급자에 동일한 subsecond 절단을 강제하자는 뜻이 아니다. 경로별 제약과 최신 정책이 우선한다.

## 4. 현재 선택 가능한 ASR 경로별 판정

P는 미제출 입력을 다룰 수 있는 pre-segment 경로, T는 검증된 native timing을 활용하는 경로, C는 더 보수적인 source 소유권 경로다. 시간 정보가 없는 C도 유용한 전사를 먼저 내보낼 수 있다.

| 선택 경로 | 현재 활용 | 남은 차이 / 제한 |
|---|---|---|
| `local_cpu_auto` | 실제 delegate를 골라 로컬 P 실행 | 결과에 resolved delegate/model 기록 부족(D4). Auto를 독립 음향 모델로 취급하지 않음 |
| `local_parakeet_v3` | 미제출 PCM 경계 절단, 결과·tail 처리 | raw points 보존 D5. 기존 lexical 누락은 #264 승인 제외 |
| `local_parakeet_ja` | 같은 로컬 P 소유권 | 모든 언어·짧은 발언 정확도 검증을 의미하지 않음 |
| `local_qwen` | CPU 로컬 P, 제출 옵션 고정 | D4. native 결과 뒤 latest proof와 표시 해결은 별도 |
| `local_qwen_gpu` | GPU 로컬 P, 실제 Nemotron/Gemma 공존 사례 | 현재 positive early/late 사례 있음. 일반 정확도·VR 여유는 미검증 |
| `deepgram` | 적격 `nova-3`/`general-nova-3`의 word/final prefix 활용 | missing-final `Finalize` 연결 D1. 가장 직접적인 정책 불일치 |
| `elevenlabs_scribe` | 적격 timestamped committed text(CWT), plain fallback의 단일 winner | 기존 commit은 있으나 경계→missing-final commit 소유권 연결 없음. 최소 0.3초 uncommitted audio 등 조건을 지켜야 함 |
| `soniox` | 자동 endpoint ON, 확정 토큰과 적격 native clock | `<end>`와 manual `<fin>`은 다름. 첫 manual 이후 clock은 보수 C로 격하. 빠른 speaker flush는 원래 DEFER였으므로 부재 자체를 모두 버그로 판정하지 않음 |
| `qwen_audio` | #255에 따라 Peer task terminal의 조립 source, 최대 2개 연결 handover | native sentence별 Peer 번역 금지는 승인된 변경. Nemotron 제어가 자동 구현된 것은 아님 |
| `gemini_transcribe` | 독립·조기·복수 authoritative final, local final ACK를 기다리지 않는 전송 | final mapping 없음→회색. 추가 경계 EOS 연결·빈도 qualification 없음. D2/D3 별도 |
| `rolling_free` | 실제 선택된 Scribe/Gemini/Deepgram delegate와 epoch를 따라감 | 각 delegate 능력·한계를 그대로 상속. 이름만으로 timing/무료 비용을 보장하지 않음 |
| `custom_offline` | 한 WAV 요청과 응답의 whole-request enclosure 사용 | 임의 서버의 stateless 의미·단어 timing은 미인증. whole-span만으로 P 자격을 부여하지 않음 |
| `custom_realtime` | provisional delta와 matching completed terminal 분리 | commit ACK는 텍스트 final이 아님. 임의 서버의 경계 제어·시간 대응을 추정하지 않는 C 제한 |

적격성 guard도 확인했다. Deepgram은 요청/반환 model, Scribe는 정확한 model과 유효 CWT, Soniox는 model/endpoint 및 manual 이전 clock, Qwen은 model/endpoint를 확인한다. “사용자 모델 이름이 무엇이든 무조건 timed로 믿는다”는 일괄 finding은 성립하지 않는다.

### 4.1 Qwen은 옛 matrix만 보면 잘못 판정한다

원래 #212 S4의 native sentence 조기 release를 현재 Peer에 그대로 요구하면 안 된다. #255 QWEN-HANDOVER-2는 다음을 명시적으로 선택했다.

- 기존 로컬 pause/safety 구간을 유지하고, 로컬 segment마다 native task 하나를 사용한다.
- 채널당 최대 두 연결로 predecessor drain 중 successor를 시작한다.
- Peer는 `task_finished`에서 native final들을 한 번 조립한다.
- native sentence ending, interim, fixed text는 Peer 번역 trigger가 아니다. SELF의 stable/speculative 정책은 별도다.

현재 `PROVIDER/qwen_audio.py:863–877,921–950`이 이 구분을 구현한다. endpoint별 clock 판정은 다음과 같다. task를 하나 더 열었다고 검증되지 않은 단어 시간이 생기지 않는다.

| 지역 / endpoint 종류 | 현재 clock 판정 |
|---|---|
| Beijing / `legacy_shared` | unknown |
| Beijing / `workspace_dedicated` | 정확한 기본 model과 엄격한 endpoint guard를 만족할 때만 qualified |
| Singapore / `legacy_shared` | unknown |
| Singapore / `workspace_dedicated` | unknown |

따라서 후속 수리는 Deepgram의 누락을 고친다는 이유로 Qwen의 승인된 task 계약을 되돌려서는 안 된다. 과거 five-task live 관측도 새로운 최대 두 연결 정책의 전체 서비스 검증을 대체하지 않는다.

## 5. Gemini와 “2티어” ASR: 무엇이 되고 무엇이 안 되는가

### 5.1 두 종류의 낮은 메타데이터 경로를 구분해야 한다

| 상태 | 가능한 활용 | 불가능한 추정 |
|---|---|---|
| 단어 시간은 없지만 final이 대응하는 전체 입력 범위는 검증됨 | 전체 범위의 화자 근거로 한 발언을 귀속. whole-span에 실제 여러 사람이 있으면 혼합/미확정 유지 | 임의 단어별 A/B 분할 |
| final이 어느 입력 범위인지도 검증되지 않음 | authoritative final을 독립 source로 접수·번역·표시 | 도착 시각, 현재 VAD, 현재 화자, FIFO로 입력 범위나 화자 만들기 |

관측된 Gemini는 두 번째다. **“단어 timestamp가 없으니 아무것도 못 한다”가 아니라 “텍스트 활용은 되지만 화자 결합의 전제가 부족하다”**가 정확하다. 1초 동안 색을 늦게 붙일 수 있어도 입력 대응이 없으면 계속 회색이다.

### 5.2 현재 작동하는 경로

1. `PROVIDER/gemini_transcribe.py:634–679`가 authoritative `input_transcription`마다 독립 `STTRecognitionUnit`을 만든다.
2. UUID와 receipt sequence로 delivery를 구분한다. `finished=True`만 기다리는 gate가 아니다.
3. `STT/streaming_speaker.py:1204–1205`는 provider의 speaker identity를 활성 권위로 쓰지 않고 전달한다.
4. `STT/scoped_engine.py:1758–1794`가 현재 stream/receipt를 확인한다.
5. `ORCH/translation_channel_callbacks.py:204–247`, `peer_translation_channel.py:458–472`가 공통 접수·publication sequence를 부여한다.
6. `STT/recognition_units.py:67–91`의 origin은 `source_binding=stream_scoped`, `order_basis=provider_receipt`다. 이후 같은 SourceChild/문맥/번역 경로를 사용한다.

receipt order는 이 C 경로가 허용한 반환 순서이지 검증된 음향 순서가 아니다. final은 로컬 turn 없이도 이미 유효하게 열린 stream에서 수용할 수 있지만, 이것이 아무 demand/readiness 없이 항상 마이크·연결을 켠다는 뜻은 아니다.

`seal`은 정렬된 오디오 전송과 EOS 제출을 처리하고 local input terminal을 내보낸다. **final ACK가 올 때까지 다음 오디오 전송을 막지 않는다.** 같은 연결에서 successor 입력을 계속 보낼 수 있다. 새 오디오 없는 중복 EOS는 억제한다(`gemini_transcribe.py:937–965`). 입력 큐도 유한하며 overflow를 무음 손실로 위장하지 않는다.

통제된 실제 adapter 관측에서는 final `No.`, 별도 final `No.`, 빈 final에 서로 다른 세 ID와 순서가 부여됐다. 둘의 문자열이 같다고 두 번째 발언을 지우지 않았다. 같은 unit의 재접수는 거절했다. `finished=False` final도 수용했다. interim/activity는 authoritative unit을 만들지 않았다. 이는 실제 서비스 음향 정확도나 wire replay의 모든 변형을 검증한 것은 아니다.

### 5.3 partial이 사라진 것이 아니라 finality 계약이 다르다

- Deepgram 등의 **native immutable prefix**는 전체 turn이 끝나기 전에 접수할 수 있다.
- Gemini의 **speculative interim**은 authoritative final이 아니다. 번역하면 나중에 final에서 빠진 내용이나 바뀐 내용을 확정 원문으로 만들게 된다.
- #134/#170 시기의 manual activity/final-ACK barrier, 시간 경과에 따른 degraded interim fallback은 #194 Hybrid/B′ 및 #212 final-only 정책으로 대체됐다.
- #191의 Stage 1/Stage 2와 보편적인 1초 규칙도 현재 #211 정책이 아니다.
- #211의 “B′ 재설계 제외”는 이미 존재하는 독립-final 구조를 제거하라는 뜻이 아니라, 추가 재설계를 이번 범위로 확대하지 말라는 뜻이다.

따라서 옛 interim fallback을 되살리는 것은 빠진 기능의 단순 복원이 아니라 **정책 변경**이다. 이번 감사는 이를 권하지 않는다.

### 5.4 남아 있는 조기 진행 기회

Gemini도 “화자 매핑 불가”와 “입력을 진행시켜 final을 요청할 방법이 전혀 없음”은 다르다. 기존 EOS 제어는 존재한다. 하지만 Nemotron 경계에서 이를 요청하는 소유권 연결과 촘촘한 빈도의 안정성 검증은 없다.

이 갈래를 다룬다면 현재 전송 frontier의 제어, successor 보존, 중복 EOS 합치기, 독립 final 보존을 확인해야 한다. EOS 하나를 특정 발언의 final ACK로 간주하거나, 응답을 그 시점의 화자에게 붙여서는 안 된다. #264의 두 실험이 지원되는 좁은 수리를 찾지 못했다는 결과는 모든 조기 진행 의무를 삭제한 결정이 아니다.

### 5.5 원시 관측이 말하는 것

아래는 **기존 원시 자료를 다시 읽은 결과**다. 이번 감사에서 클라우드를 재호출하지 않았다.

| 자료 | 관측 | 말할 수 없는 것 |
|---|---|---|
| #218 valid baseline3 | raw 13개 메시지, interim 8개, authoritative final은 `Gentlemen of the press.` 하나. interim에는 앞 A+B 내용이 있었음 | A가 이미 authoritative final이었는데 앱이 버렸다는 주장 |
| #264 automatic, 중간 EOS 없음 | 같은 93,404 samples/5.83775초 입력. interim 8개, B final 하나, EOF 전 도착 | 이 방식이 일반적으로 내용 누락을 해결한다는 주장 |
| #264 manual activity | 같은 입력. interim 10개, B final 하나, EOF 뒤 약 0.244초 도착 | manual activity가 누락·매핑을 해결한다는 주장 |

세 자료의 해당 final payload에는 utterance 시작/끝 offset이 없었다. #218은 raw→SDK→adapter→engine에서 동일 B final이 보존됐다. EOF 뒤 약 9.015초까지 추가 final을 관측하지 못했으며, 그 관측 창을 영구 부재로 확대하지 않는다.

#264의 두 valid 호출은 `GOOGLE_API_KEY2`를 사용한 기존 기록이며 총 입력 11.6755초다. credential 값과 비공개 archive 내용은 이 문서에 싣지 않는다. 결론은 **이 사례에서 지원되는 설정 변경만으로 A final이나 매핑을 복구하지 못했다**는 것이다. 모든 EOS가 쓸모없다거나 Gemini 전체가 고장이라는 일반 결론은 아니다.

### 5.6 D3: 실제로 버리는 진단 정보와 애초에 없는 매핑

#212 S5는 native activity timing을 진단 사실로 보존하되 final 대응과 혼동하지 말라고 한다. 현재 `gemini_transcribe.py:634–708`은 final/GoAway 등을 처리하지만 **VoiceActivity의 type/audioOffset을 결과 진단으로 보존하지 않는다.**

원시 자료에는 VoiceActivity offset `4.724s`가 raw/SDK 단계에 있었다. 이후 recognition unit에는 이 사실이 없다. 오디오 전송 loop의 `_audio_since_fence`는 EOS 중복 억제용이며, 결과의 `native_audio_progress`/`control_id`/`control_sent_sample`을 채우는 기록이 아니다.

이는 **전체 엔진 입력 provenance가 없다**는 뜻은 아니다. 소실되는 native 진단·fence 연결의 범위를 특정한 finding이다. activity offset을 보존해도 그 offset이 어느 final 텍스트를 둘러싸는지는 별도로 증명해야 한다.

또 adapter가 가상의 final-level offset을 자동 보존하지 않는 경로도 확인했지만, 이번 raw final에는 그 필드 자체가 없었다. 이를 “SDK가 실제 final timestamp를 버린 원인”으로 보고하면 틀린다.

## 6. 조기 경계 외의 정책 불일치

### 6.1 D2: Gemini final 도착이 source-idle을 갱신한다

#194 R5, B′ 요구 6, #211 R6, #216 R5는 **반환 텍스트·keepalive만으로 source-speech idle 적격성을 갱신하지 않는다**고 명시한다.

현재 `STT/scoped_engine.py:1769–1776`은 independent stream의 비어 있지 않은 `STTRecognitionUnit`을 받으면 `_last_native_speech_at_s`를 현재 receipt clock으로 갱신한다. `2609–2617`, `2711–2724`의 두 idle 계산은 그 시각을 `max(...)`에 넣는다. Gemini가 해당 session flags를 사용한다.

실제 엔진을 통제된 clock으로 실행한 결과:

| 사건 | 관측 |
|---|---|
| 마지막 source activity = 100초 | 원래 source-idle 기준 만료는 160초 |
| final receipt = 154초 | native receipt 시각이 154로 갱신 |
| clock = 161초 | 연결 유지. pending turns=0, source work pending=false |
| clock = 214초 | 연결 종료 |

이는 단순히 pending final을 drain하느라 기다린 사례가 아니다. 기존 `tests/core/test_stt_scoped_engine.py:4310–4373`도 이 연장을 기대한다. 반면 Soniox의 `test_soniox_source_lifecycle.py:430–450`은 반환 텍스트에도 160초 만료를 유지한다.

도입 커밋은 `49ff1619885ef605756565fca76cbcb7339491a2`, 2026-10-05 08:06:07 KST, `fix(stt): improve scoped provider recovery and Gemini diagnostics`다. 필드·receipt 갱신·두 idle 계산이 함께 들어갔고 #264 baseline의 조상에도 있다. **#264가 새로 만든 regression은 아니다.**

[INFERENCE] 늦은 전사가 실제 source activity가 아닌데도 연결·배경 입력 수명을 연장할 수 있다. 별도 9분 최대 수명은 남는다. 이번 실행은 live 전송량이나 요금 측정이 아니다. 수리할 때 source-idle과 pending drain/max age를 함께 제거해서는 안 된다. VAD가 놓친 onset의 authoritative final을 수용하는 것과 receipt를 새 음성 시계로 쓰는 것은 별개다.

### 6.2 D4: 로컬 옵션은 고정되지만 실제 출처 기록은 부족하다

올바르게 구현된 부분:

- `STT/batch_speaker.py:83–90`의 제출 옵션 고정.
- `PROVIDER/local_qwen_sherpa.py:396–405,440–443`의 backend copy에 결합한 decode.
- GPU 경로 `174–184,223–229`의 고정 hint.
- `PROVIDER/local_cpu.py:124–137,178–210`의 실제 Auto delegate 선택.

남은 부분:

- 실제 backend/model/capability는 `batch_speaker.py:284–299` 제출 log에 있으나 `392–430`의 outgoing update/terminal에 native provenance로 충분히 남지 않는다.
- `local_qwen_sherpa.py:520–543` 및 GPU terminal 경로도 관련 native 사실을 생략한다.
- settings snapshot의 선택 provider/signature는 Auto가 실제로 해석한 decoder/model의 명시 기록과 같지 않다.

#212 A1/#215 R8의 “실제로 사용한 backend/model을 결과 출처에 남긴다”는 요구에 비해 불완전하다. **잘못된 decoder를 실행했다거나 옵션이 도중에 바뀌었다는 finding은 아니다.**

### 6.3 D5: Parakeet raw timing/decoder facts는 텍스트만 남긴다

#212의 local 공통 선택은 optional timestamp 보존 항목을 IMPLEMENT로 정했다. token, emission points, TDT advances, decoder/version/단위를 보존하되 이를 word alignment로 속이지 않는 계약이다.

Parakeet은 text-only decode 경로를 상속하며 `local_qwen_sherpa.py:298–312`가 `result.text.strip()`만 읽는다. 따라서 raw points 보존은 미구현이다. 이것이 현 PCM 경계 절단을 불가능하게 만드는 것은 아니며, 해당 점들을 단어 구간으로 승격해서도 안 된다.

사용자는 기존 Parakeet 누락을 수용했고 #264는 추가 Parakeet-specific 수리·runtime/input-window/음향 실험을 제외했다. **이 보존 차이를 기록하는 것과 Parakeet 누락 수리를 다시 완료 조건으로 요구하는 것은 다르다.** 이번 감사는 후자를 하지 않는다.

## 7. 원문·번역·화자 표시의 공통 계약

이 범위에서는 새 결정적 반례를 발견하지 않았다. 아래는 현재 코드와 기존 한정 실행 근거가 지지하는 구현이다. 전체 공급자·전체 음향 입력의 완전성을 인증하지는 않는다.

### 7.1 원문 접수와 번역

- `ORCH/translation_turn.py:250–307,1323–1472`의 SourceChild는 source identity/revision, raw text, 언어, 순서, 출처, 설정, 입증된 group을 고정한다. 번역 target/retry는 같은 source의 attempt이지 새 발언이 아니다.
- `STT/backend.py:408–705`의 ledger는 contribution ID와 범위로 접수한다. 문자열이 같다는 이유로 실제 반복을 삭제하지 않으며, 이미 소비한 prefix 뒤 terminal tail을 남긴다.
- `ORCH/translation_request.py:506–593`은 prior context snapshot을 얻은 뒤 source를 한 번 등록한다. U1은 자기 자신을 문맥으로 보지 않고, U2는 U1 번역 완료 전에도 U1의 원문을 볼 수 있다.
- `translation_turn.py:1640–1792`와 `peer_translation_channel.py:797–912`는 bounded concurrent translation과 source-ordered prefix 출력을 유지한다. 앞 child의 출력이 뒤 sibling 전체의 완료를 기다리지 않는다.
- 불명확한 **source 소유권**은 회색으로 포장해 접수하지 않는다. 입증된 source의 **화자 미상**은 유효 텍스트와 함께 표시할 수 있다.
- 색 변화만으로 문장을 자르지 않는다. 실제 화자 전환과 적격 lexical junction은 분할 근거가 될 수 있다. 번역을 끝낸 뒤 결과를 비례 분할하거나 A-B-A의 두 A를 원문 단계에서 합치지 않는다.
- prefix trimming은 OFF다. `effective_text`는 raw text 그대로이며, renderer에 두 번째 fuzzy deduplicator가 없다.

### 7.2 표시 전 해결과 표시 후 1초 해결

`core/speaker_identity.py`가 source-keyed assignment를 소유하고 presenter가 최초 실제 가독 표시를 알린다. 렌더러가 음성에서 화자를 추론하지 않는다.

1. 제출 시 PCM·request·옵션을 고정한다.
2. own-native-final 당시의 귀속 snapshot은 역사적 출처로 유지한다.
3. 번역·게시 준비 중 새 적격 증거를 사용한다. 번역을 추가로 기다리게 하지 않는다.
4. 원문 또는 번역이 처음 읽을 수 있게 된 시점에 최대 1.000초 창을 시작한다.
5. 그 안에 해결되면 미확정→확정 배정을 한 번 적용한다. 이미 확정한 A를 B로 바꾸지 않는다.
6. 만료·삭제·퇴거·reset·epoch retirement가 먼저 오면 닫는다.

500ms는 좋은 결과를 구분하는 보고 구간이지 대기 시간이나 보장이다. 원문 전용 표시와 번역 실패 대체 출력도 최초 가독 표시가 될 수 있다. hidden/protected/ordered 대기는 아직 읽을 수 없는 상태이므로 번역 완료 시각을 그대로 이 시계로 쓰면 안 된다.

late metadata 갱신은 identity/style/revision만 바꾸며 텍스트·언어·위치·순서·appearance·TTL·문맥·번역 횟수를 바꾸지 않는다. 색상 팔레트는 결과 도착 순서가 아니라 권위 있는 source 순서로 배정한다. 따라서 화자 key를 이미 알지만 앞 source 때문에 색상만 잠시 미정일 수 있다.

### 7.3 A→B→A, 카드, 팔레트

- 처음부터 같은 group/epoch/key가 알려졌고 현재 호환되는 A 카드가 보이면 새 A source를 append하고 최신 위치·전체 카드 TTL을 갱신한다. 원문/문맥 순서는 여전히 A-B-A다.
- 이미 회색으로 읽은 독립 카드가 나중 A로 해결되면 **그 카드에서 색만 바꾼다.** 이전 A 카드로 합치거나 옮기지 않는다. #264의 정당한 표시 안정성 예외다.
- 같은 색만으로 같은 사람이라고 판단하지 않는다. unknown gray들은 한 사람이 아니다. overflow gray라도 실제 key가 같으면 호환 조건에서 append할 수 있다.
- 네 색, source-order 할당, epoch 내 비재활용, SELF의 별도 처리, overflow divider 조건을 유지한다.
- retry/다른 target/동일 source completion/metadata/reconnect replay/텍스트 없는 marker는 새 발언 append나 TTL 갱신이 아니다. 오래된 expiry callback은 새 revision을 지우지 않는다.
- 누적 카드는 caption presentation transform이다. 누적 전체 텍스트를 OSC/chatbox로 재전송하지 않고 내부 화자 label을 프롬프트에 넣지 않는다. 기존 Peer 출력 금지는 유지한다.
- 과거의 인공적인 1초 occupant-replacement gate는 제거됐다. source-order·SELF 보호·두 slot·watchdog에 따른 정상 대기를 같은 gate로 오인하면 안 된다.

근거: `OVERLAY/presenter.py:267–344,493–703,914–1003,1091–1298,1323–1414`, `presenter_sources.py:18–118`, `core/runtime/output.py:432–434,891–897`.

### 7.4 증거 수명·경계·귀속

하나의 capture-scoped `SPEAKER/evidence.py` view를 batch/streaming/presentation이 공유한다. request→contribution→source의 proof handoff가 필요한 onset/source/proven replay 범위를 유지한다. 정상 producer/ASR close만으로 이미 인계된 표시 증거를 없애지 않는다. reset·epoch 변경·hard pressure·명시 retirement는 별도다.

기존의 “own-native-final까지 들어온 정보만 사용”은 최종 표시 제한이 아니다. immutable origin과 갱신 가능한 presentation assignment를 분리한다. 돌아온 A의 origin이 `unavailable`로 남아도 표시 key가 나중 원래 A로 해결될 수 있다.

입증된 짧은 **이미 사용한 replay**의 이물 화자를 제한적으로 허용하는 것은 #218에서 승인된 대표 화자 정책이다. 반대로 새 B의 짧은 `No.`를 긴 A의 비율로 덮거나, 미관측 4ms를 무음이라고 간주하는 것은 허용되지 않는다. 500ms 고유 onset 복구와 300ms hard-rollover replay는 구별한다.

## 8. Runtime·프로파일·자원

### 8.1 구현된 초기 runtime

현재 diarizer factory는 `NVIDIA/NeMo-Speech.cpp` 기반 초기 runtime이다. ASR에 transcribe.cpp가 존재한다는 이유로 #219의 diarizer migration이 끝난 것은 아니다.

기존 기록은 실제 Windows/RX 7900 XTX/Vulkan 실행, 8-channel 모델, 두 profile load, CPU Parakeet 또는 GPU Qwen과의 공존을 포함한다. 선택 Vulkan GPU의 neural compute를 요구하고 숨은 CPU/CUDA diarizer fallback을 허용하지 않는다. CPU capture/frontend/postprocess가 있다는 사실은 neural fallback 허용과 다르다.

F16 artifact는 201,832,928 bytes, 356 tensors이며 125 F16 + 231 F32다. “모든 연산·누산이 FP16”이라고 주장할 수 없다. 모델은 고정 community artifact의 hash/load/geometry를 확인했지만 원 NVIDIA checkpoint부터 publisher conversion까지 독립 재현한 것은 아니다.

| 프로파일 | chunk/right/left/FIFO/cache/update, 80ms coarse frame | 의미 |
|---|---|---|
| 1.04 | `9/4/0/264/264/222` | 720ms + 320ms geometry |
| 0.64 | `6/2/0/264/264/222` | 480ms + 160ms geometry |

둘 다 구현·실행됐지만 production winner/default는 미선택이다. 설정 기본값도 disabled/profile=None/device=None/consent=False다. 짧은 label을 전체 체감 지연이나 정확도로 읽으면 안 된다.

### 8.2 유한 자원과 수명

| 소유자 | 주요 상한 / 동작 | 한계 |
|---|---|---|
| SpeakerRuntime | pending 64,000 samples/128 items, map 2,048 pieces, bounded evidence | 짧은/합성 pressure 관측이지 무한 wall-clock 보장 아님 |
| native/IPC | probability retention 1,024 frames × 8, 한 operation in flight | 전체 세션 확률을 무한 보관하지 않음 |
| 공유 증거 | 2,048 frames, 128 proofs, 4,096 range pieces | hard eviction은 관련 proof를 무효화; identity reset으로 메모리를 숨기지 않음 |
| streaming wrapper | 64 turns, maps/timings 각 4,096, boundary history 256 | successor backlog 전체 서비스 SLA는 별도 |
| presentation resolver | pending source 128, alias/disposition 4,096, late window 1초 | ASR 연결·PCM을 1초 더 유지하기 위한 정책이 아님 |
| remote lifetime | source-idle 60초, provider별 max/drain | Gemini receipt 갱신 D2 예외가 남음 |

VAD short gap/tail을 처리하고 long idle에서는 frontend/inference를 pause하면서 모델/화자 연속성을 유지한다. 새 capture speech는 한 번만 공급하며 ASR용 onset/replay를 새 diarizer capture로 재공급하지 않는다. 기존 causal tail padding은 새로운 synthetic natural-end flush 제안과 다르다. 그 제안을 이번 기본 수리로 새로 도입한 것은 아니다.

실제 현재 #264의 GPU counter sampler 세 번은 실패했다. 과거 adapter-level 점유율은 주변 프로세스를 포함하며 Nemotron 단독 VRAM이나 GPU 포화 증거가 아니다. host elapsed timing은 pure GPU inference timestamp가 아니다. 4ms coverage 부족 때문에 다음 720ms chunk를 기다린 사례도 “4ms 계산에 720ms가 걸렸다”는 뜻이 아니다.

## 9. 요구사항 전수 대조표

표의 **일치**는 해당 코드 계약을 확인했다는 뜻이다. 실행 범위는 10절에 별도로 적는다. **부분**은 구현·관측의 일부만 충족, **누락**은 선택된 의무 부재, **허용 제한**은 승인된 conservative 경로, **후속**은 의도적으로 나중인 작업이다. 이미 승인된 제외·개정을 결함으로 다시 세지 않는다.

### 9.1 #211 상위 요구와 완료 기준

| 조항 | 판정 | 근거 / 남은 경계 |
|---|---|---|
| R1 두 입력 전략·공통 source | 부분 | 로컬 P와 기존 timed-final release 구현. missing-final 제어 D1; 3–5절 |
| R2 초기 F16/Vulkan·후속 migration | 초기 일치 / 후속 미완 | 8절, #213 및 #219 행 |
| R3 두 전체 profile·후속 선택 | 구현 / 선택 미완 | geometry 실행됨. #218의 완성 후보 비교·owner 선택은 미완 |
| R4 source/time/ownership/번역 | 일치, 출처 일부 부족 | 7.1절; 로컬 native 사실 D4 별도 |
| R5 sole authority·identity·표시 | 현 개정에 일치 | 7.2–7.4절. #264의 단회 late 예외 적용 |
| R6 lifecycle/resources | 부분 | pause/bounds/fault isolation 구현. Gemini source-idle D2, VR/자원 주장 한계 |
| R7 artifact 의존·병렬 진행 | 구조 일치 / 프로그램 미완 | #215/#216은 #212/#213/#214, #217은 #214, #218은 조립 결과, #219는 초기 자격 기준에 의존. 닫힌 하위 이슈 수로 완료 판단 불가 |
| R8 실제 경계·pre-roll·반복 | 부분 | onset500/rollover300/text-only0, trimming OFF, timer 유지. 새 cloud continuation은 D1 구현 때 아직 다룰 의무 |
| C1 실제 GPU/model/두 profile | 한정 관측 | 초기 실제 Vulkan 기록. 모든 기기·정밀도 조합 보장 아님 |
| C2 실제 유용한 조기 source/번역 | 부분 | local GPU와 일부 native-final 사례 양성. missing-final actuator는 입증되지 않음 |
| C3 내용·소유권·음향 품질 | 부분 | received-event accounting과 실제 단어 생존/반복/파편화는 다른 기준 |
| C4 문맥 once·순서·동시성 | 한정 일치 | SourceChild/ledger/context contract 및 기존 실행 기록 |
| C5 실제 표시 경로 | 부분 | 기존 native texture/current 일부 replay 있음. 최신 전체 coupled pixels·HMD 없음 |
| C6 자원·상태·공존 | 부분 | bounded/pause/CPU·GPU ASR 공존. distinct 8+/capture loss/VR 자격 미완 |
| C7 경로 정책·최종 절충 | 부분 | 13 routes와 4 Qwen endpoint 판정. D1·미선택 profile 잔존 |
| C8 통합·취득·호환·수명 | 부분 | 기존 acquire/consent/CLI/package 근거. D2와 current native v3 배포 검증 경계 |

### 9.2 #212 경로 결정

| 조항 | 판정 | 현 구현과의 연결 |
|---|---|---|
| R1 실제 revision/model/backend/endpoint inventory | 조사 완료 / 기록 일부 부족 | 13개 선택 경로·4 Qwen endpoint. 결과 출처 D4 |
| R2 versioned finality/timing/control matrix | 기존 고정 matrix 유효 | 후속 #255/#264와 함께 읽어야 함. D3/D5는 선택된 보존의 구현 부족 |
| R3 P/T/C·delegate·queued options | 선택 완료 / 제품 부분 | CPU/GPU 옵션 freeze 구현, cloud guards 존재, D4 |
| R4 missing-final progress 결정 | 결정 존재 / 제품 누락 | Deepgram D1, Scribe 부분, Soniox/Gemini/Custom의 자격 제한 분리 |
| R5 complete-text/short/overlap/continuation | 결정 및 부분 구현 | 단어별 번역 금지·자연 짧은 발언 보존·500/300 구분. 새 제어 continuation은 미구현 |
| R6 계약·관측·unknown 구분 | 유지해야 할 증거 계약 | 고정 보고·후속 결정을 재사용. 별도 zero-repeat 인증이나 profile 선택 이슈가 아님 |

조사 이슈의 지원되는 negative는 정당한 완료 결과다. 하지만 downstream 구현에서 선택된 positive 의무를 빼고 같은 완료 판정을 재사용할 수는 없다.

### 9.3 #213 초기 runtime

| 조항 | 판정 | 코드/기록 |
|---|---|---|
| R1 실제 runtime/model/dependency pin | 일치, lineage 한계 | `composition/application_runtime.py:475–485`; Nemotron artifact manifest; 8.1절 |
| R2 F16·명시 GPU·fallback 금지 | 한정 실제 관측 | `app/adapters/nemotron/backend.py`, `worker.py`, native patches; all-F16 연산 주장은 불가 |
| R3 두 full profile | 구현·실행 | `app/adapters/nemotron/protocol.py:6–9`; winner 없음 |
| R4 resident weights/stream/ASR 분리 | 구현·공존 관측 | worker `74–95,125–130,172–179`; VR 경쟁 미측정 |
| R5 backend-neutral evidence/frontier | 구현 | `SPEAKER/contracts.py:11–42,65–109`, `runtime.py:444–608` |
| R6 VAD pause/tail/state/source map | 구현·한정 관측 | runtime `276–308,376–423,483–608`; long idle은 close/reset과 다름 |
| R7 총체적 bounds | 구현·bounded 관측 | contracts `45–62`; 8.2절. 장시간 wall-clock 전체 메모리 보장은 아님 |
| R8 epoch/stale/overload/unavailable | 구현·fault 관측 | runtime `125–275,425–442,610–621`; 정상 ASR 진행 보존 |
| R9 readiness/취득/consent/packaging | 기존 구현·기록 | `app/services/speaker_application.py`, `app/adapters/nemotron/assets.py`, `docs/cli.md:309–331`; current v3 전체 배포와 구분 |
| R10 original capture once | 구현 | `core/runtime/peer_channel.py:253–270`; replay를 새 capture로 넣지 않음 |

### 9.4 #214 공통 source·문맥

| 조항 | 판정 | 코드/남은 경계 |
|---|---|---|
| R1 versioned SourceChild | 구현, native 사실 일부 부족 | `translation_turn.py:250–275,1350–1398`; D4 |
| R2 audio boundary≠finality | 일치 | ledger `286–325,361–384,529–595`; 불변 prefix와 bounded tail |
| R3 identity/range once·terminal tail | 일치 | ledger `414–598`, normalizer `131–159,231–254` |
| R4 번역 전 composition·A-B-A | 일치 | turn `92–188,1323–1472`; 소유권 불명은 gray success 아님 |
| R5 prior-only·history once·짧은 source | 일치 | request `450–593`, channel runtime `104–159`; target/retry 공유 |
| R6 bounded concurrency·순서·실패 격리 | 일치 | turn `1640–1792`, peer `855–912,1033–1069` |
| R7 admission/recognition/execution 종료 분리 | 일치 | ledger `379–384,600–705`; ordinary seal은 접수된 번역 취소 아님 |
| R8 Soniox end/fin/drain 공통 ledger | 일치 | peer `496–571`; marker-only는 source/context/TTL 없음 |
| R9 optional filter once | 허용된 OFF | effective=raw. 사용하지 않는 filter/편집 archive를 만들 필요 없음 |

완료 기준의 batch/append-replace-final, terminal tail, repeated text, epoch/settings/stop race, Soniox marker sequence, prior context, reverse completion은 기존 `tests/core/test_source_context_contract.py`와 최종 shared-handoff 실행 기록에서 확인했다. 이번 감사에서 재실행한 test suite는 아니다.

### 9.5 #215 로컬 경계

| 조항 | 판정 | 코드/남은 경계 |
|---|---|---|
| R1 retained 미제출 prefix 절단 | 구현 | `batch_speaker.py:125–185`; 관측 시각과 경계 좌표 분리 |
| R2 단일 cursor owner·stale/duplicate | 구현 | 같은 파일 `164–207`; committed watermark |
| R3 selected range 한 번 decode·A-B-A | 구현 | `187–207,245–280`; whole-parent 재decode 없음 |
| R4 overlap/uncertain/짧은 정정 | 구현, 음향 범위 제한 | `evidence.py:272–345`, `speaker_attribution.py:96–195`; short source floor 없음 |
| R5 successor·자연 종료·6초 safety·pressure | 구현 | batch `125–150`; 기존 `listen_delivery.py` timer/lock 유지 |
| R6 onset500/rollover300/disjoint child | 구현 | 기존 VAD 경로 + batch context/source partition; 800ms 합산 아님 |
| R7 display group 유지 | 구현 | batch `225–230,245–251` |
| R8 실제 native options freeze | 실행 일치 / 기록 부분 | CPU backend copy/GPU tuple; D4는 결과 출처 기록 |
| R9 trimming OFF/필요시 국소 대응 | 허용된 OFF | 기존 반복을 숨기지 않음. Parakeet 제외 범위 존중 |

실제 local GPU 조기 출력은 positive 근거다. 모든 절단점의 첫 음절·짧은 발언 정확도나 Parakeet 누락 해결의 증거로 확대하지 않는다.

### 9.6 #216 streaming·untimed

| 조항 | 판정 | 코드/남은 경계 |
|---|---|---|
| R1 native finality/identity/timing/coverage 보존 | 부분 | 핵심 provider facts 보존. Gemini D3, raw optional metadata 한계 |
| R2 경계 독립 기록·ordered prefix·residual | 구현된 갈래 있음 | `streaming_speaker.py:210–240,1085–1203`; 없는 final은 만들어내지 않음 |
| R3 complete-text·짧은 source·raw text | 구현 | qualified lexical partition `431–847`; unknown clock은 tolerance로 위장하지 않음 |
| R4 lawful progress·coalescing·races | 누락/부분 | D1. 기존 natural controls의 race 처리가 새 경계 제어 구현을 대신하지 않음 |
| R5 sent/source/evidence/control·lifetime | 부분 | 성공한 전송 map과 cursor는 구분. D2, 새 actuator successor qualification 필요 |
| R6 실제 경계별 pre-roll | 기존 경로 구현 / 새 제어 미완 | text-only release는 추가 replay/timer reset 없음 |
| R7 untimed whole-span·Gemini C | 허용 제한 및 후속 개정 | 검증된 범위만 귀속. bounded proven replay 예외 외 새 B 흡수 금지 |
| R8 Nemotron sole authority | 구현 | provider speaker runs 제거, Soniox native diarization OFF; raw 보존은 활성 권위 아님 |
| R9 actual route/epoch·보수 fallback | 구현, 일부 기록 부족 | Rolling delegate 유지, cloud model guards. normal text를 버리는 fallback 아님 |

Soniox automatic endpoint ON은 #212의 명시 후속 결정이다. `<end>`는 source progress, `<fin>`은 matching manual operation, close는 drain/retirement다. post-finalize clock의 미해결 counterexample은 그대로 남는다. guessed fixed offset, silent clamp, marker마다 신규 source를 만들지 않는다.

### 9.7 #217 caption

| 조항 | 판정 | 코드/남은 경계 |
|---|---|---|
| R1 enclosing group | 구현 | `recognition_units.py:190–214`, turn `1360–1398`; ambiguous/cross-group 독립 |
| R2 현재 보이는 compatible card만 append | 구현 | presenter `493–542`; color가 아니라 group/epoch/key |
| R3 atomic append/newest/revision/TTL | 구현 | presenter `545–703,1091–1127,1323–1347` |
| R4 source/attempt/card·replay·resurrection | 구현 | presenter sources `18–118`; metadata/target/retry는 새 utterance 아님 |
| R5 인공 1초 gate 제거 | 구현 | presenter `267–344,982–1000`; 보호·순서 대기는 유지 |
| R6 palette/freeze/handoff/divider/cache | 현 개정에 일치 | unconditional freeze만 #264가 대체. 실제 렌더 완료 범위는 부분 |
| R7 destination isolation | 구현 | caption만 누적; Peer chatbox 금지·no internal labels |
| R8 native compatibility | 구현·한정 실행 | wire16/card1/`speaker_identity_presentation` v3. matching rebuild 필요 |
| R9 effective text 한 번·no second dedupe | 구현 | upstream raw/effective 그대로. 같은 텍스트의 별도 source 보존 |

기존 fake-clock/fixture는 A-B-A, group, overflow/unknown, stale expiry, SELF, replay, target, marker, same-text를 다룬다. 실제 renderer 완료 기준은 최신 모든 경로에서 충족됐다고 쓰지 않는다(10.3절).

### 9.8 #218 실제 조립·품질·선택

| 조항 | 판정 | 남은 경계 |
|---|---|---|
| R1 실제 조립 후보 | 과거 비교 및 현재 changed-path 있음 | 현재 완성 후보의 최종 qualification과 다름 |
| R2 전체 geometry·matched 조건 | 기록 있음 | 두 profile 모두 미선택 |
| R3 baseline/1.04/0.64 세 arm | 과거 있음 / 현재 최종 비교 미완 | #264 smoke를 새 three-arm 비교로 부르지 않음 |
| R4 causal 사건·source/translation/표시 시점 | 부분 | 논리 적용·socket ACK·pixel·photon을 분리 |
| R5 대표 음향 사례 | 부분 | clear A-B-A/No/KO/JA/long 등 한정. Chinese, 자연 code-switch, distinct8+/9인, 자연 overlap 자격 미완 |
| R6 품질·coverage·호출·자원 | 부분 | 식별 수≠정확도. 현재 GPU counters/물리 capture loss/실제 billing은 모름 |
| R7 CPU/GPU ASR 공존·환경 | 승인 non-VR 관측 | VRChat/SteamVR/HMD는 승인된 후속 범위. 이번 non-VR 실험의 새 blocker 아님 |
| R8 owner 절충·profile 선택 | 미완 | scoped #264 수락은 production enablement가 아님 |
| R9 설정/휴리스틱/후속 baseline | 기록 있음 / 최종 baseline 미완 | 수치들은 reversible heuristics이지 검증된 acoustic optimum 아님 |

현재 사용한 active/clear/runner-up/competing 기준과 120ms 연속·200ms 총량·20% ambiguity allowance는 최적값으로 승인된 것이 아니다. 단순 비율로 짧은 다른 화자를 흡수하지 않는 slot-aware 제한이 별도로 필요하며 구현돼 있다.

### 9.9 #219 후속 migration

모두 **의도된 후속 범위**다. 초기 구현 버그처럼 세거나 지금 target이 사용 불가능하다고 단정하지 않는다.

| 조항 | 현 상태 / 필요한 후속 |
|---|---|
| R1 pinned transcribe.cpp diarizer/ABI | 초기 NeMo factory 유지. target adapter/build 자격 미실시 |
| R2 checkpoint/precision/두 geometry | 초기 기준 있음. target conversion/load/profile 재확인 필요 |
| R3 cumulative/delta/order/evidence 정규화 | 공통 contract 존재. target normalization 미구현 |
| R4 pause/tail/map/bounds/once feed | 초기 구현 재사용 기준. target parity 미실시 |
| R5 동일 조건 initial-target 비교 | 미실시. 과거 초기 smoke가 migration 비교는 아님 |
| R6 선택 profile target 재검증 | #218 profile 미선택이므로 해당 gate에 아직 도달하지 않음 |
| R7 owned cutover/drain/epoch/stale | 초기 epoch 기제 있음. 실제 runtime cutover 미실시 |
| R8 acquisition/packaging/옛 default 제거 | target qualified 뒤 deliberate cutover 필요. dual fallback은 새로 만들지 않음 |
| R9 practical repetition 유지 | 대표 사례 재사용. 새 zero-repeat 인증·전 공급자 sweep 불필요 |

### 9.10 #264 최신 보완

| 조항 | 판정 | 근거 / 범위 |
|---|---|---|
| R1 공유 좌표·증거 수명·custody·bounds | 구현·한정 검증 | `SPEAKER/evidence.py:32–262,355–417`; request→contribution→source handoff, 정상 close와 reset/pressure 구분 |
| R2 candidate/confirm/commit·slot-aware | 선택된 경로 구현 | 로컬 실제 절단/streaming lexical source 구분. #211의 missing-final actuator D1을 대신하지 않음 |
| R3 무대기 번역·최대1초 단회 귀속 | 구현·positive/negative 경계 근거 | 7.2–7.3절. 일반 500ms/1s 성공률 보장 아님 |
| R4 한정 Gemini 원시 조사 | 지원되는 negative 완료 | 두 KEY2 variant에서 final 누락/매핑 해결 없음. interim 승격·새 aligner·무한 실험 제외 |

#264 수락은 해당 범위에서 유지한다. 옛 custody 결함이 수리됐는데 현재도 남았다고 쓰거나, 승인된 Parakeet 제외를 무효화하거나, 이 수락을 #211/#218 완료로 확대하는 세 가지 오류를 모두 피해야 한다.

## 10. 검증 근거와 한계

### 10.1 이번 감사에서 새로 실행한 관측

실제 production classes를 통제된 provider/증거/clock과 함께 실행했다. pytest suite, 음향 모델, SDK 네트워크, GUI를 실행한 것이 아니다.

| 관측 | 실행 대상과 조건 | 실제 결과 |
|---|---|---|
| 경계와 finality 분리 | `StreamingSpeakerSession` + `STTContributionConsumptionLedger`; scripted provider와 화자 확률, mutable→native stable→terminal | 경계 sample=1600, provisional 접수 없음, stable `No. `만 boundary release, terminal `Fifty.` 잔여, provider seals=0, successor sent_until=4800, source 보존=true |
| source-idle | 실제 `ScopedRecognitionEngine`; controlled stream/clock, source100→final154 | t161에도 open, pending turns=0/source work=false, last source100/native receipt154, t214까지 close |
| Gemini independent finals | 실제 `_GeminiTranscribeLiveSession`/projection/map/ledger; 기록용 local transport | 한 audio write와 한 EOS, 새 오디오 없는 EOS 억제. 세 독립 final ID/receipt 수용, 같은 unit replay 거절, interim/activity는 접수 안 함 |

경계 probe의 첫 시도는 harness가 native event ID/sequence를 재사용해서 dedupe된 잘못된 입력이었다. 서로 다른 receipt/sequence로 수정한 유효 실행이 위 결과다. Gemini probe의 첫 network guard는 asyncio 내부 socketpair까지 막아 harness가 시작하지 못했다. event loop 초기화 뒤 guard를 적용한 유효 실행과 구분한다. 이 invalid 시도들을 제품 실패나 성공으로 세지 않는다.

앞 두 유효 probe는 subprocess exit 0이었다. 결과의 핵심 원문은 다음과 같다.

```json
{
  "boundary_sample": 1600,
  "provisional_source_release": null,
  "provisional_admitted_text": "",
  "stable_source_release": null,
  "boundary_release_reason": "boundary",
  "released_text": "No. ",
  "remaining_terminal_text": "Fifty.",
  "provider_seals": 0,
  "successor_sent_until": 4800,
  "source_exact": true
}
```

여기서 `stable_source_release:null`은 stable update를 반환받은 그 호출에서 직접 release가 나온 것이 아니라, 후속 boundary reevaluation에서 `released_text`가 나왔다는 구분이다. `null`을 native final 자체가 없었다는 뜻으로 읽으면 안 된다.

```json
{
  "at161": {
    "closed": false,
    "pending_turns": 0,
    "source_work_pending": false,
    "last_source_speech_s": 100,
    "last_native_receipt_s": 154
  },
  "closed_by214": true,
  "network_calls": 0
}
```

### 10.2 재사용한 기존 실행 자료

| 저장소 자료 | 사용한 주장 | 사용하지 않은 주장 |
|---|---|---|
| `native/nemotron/verification.json`, `packaging-verification.json` | 초기 실제 GPU/model/profile, bounds/pause/공존/취득 | 현재 모든 기기·v3 배포·VR 완성 |
| `EVIDENCE/issue216_report.json`, `issue216_acceptance.json` | 실제 Deepgram 기존-final 조기 source, optional control 0, R4 수락 문구 | 실제 번역 또는 missing-final actuator 완성 |
| `EVIDENCE/issue218_report.json`, cloud/policy 후속 기록 | 당시 후보의 품질·회색·분할·자원 관측 | 현재도 같은 결과라는 단정 |
| `EVIDENCE/issue264_gemini_findings.json` 및 대응 원시 기록 | 두 valid variant의 supported negative, raw/SDK/app final 보존 | Gemini 전체 정확도·모든 종료 방식에 대한 일반 결론 |
| `EVIDENCE/issue264_shared_handoff_repair.json:132–194` | 마지막 영향 경계의 634 passed/0 failed/0 skipped/211 warnings, custody smokes | 이번 감사의 새 suite 실행·음향 정확도 |
| `EVIDENCE/issue264_validation_streaming_after_report.json:268–917` | 현재 GPU 1개 + native tape 4개 경로, 20 source/실제 Gemma 번역 | fresh live Deepgram 호출·전체 selectable-route qualification |
| `EVIDENCE/issue264_completion_evidence.json:592–899` | 현재 제품/증거 핀, 수리·관측·잔여 한계의 결합 | failed capture를 pass로 바꾸기 |

초기 일부 #218 진단은 5ms evidence pump였다. 후속 policy 실험의 production 설정 50ms drain과 동일 cadence로 간주하지 않는다(`issue218_experiment.py:593–602`, `issue218_policy_validation.py:740–747`). 이 차이를 제거하지 않은 수치끼리 정책 효과나 실제 앱 지연을 직접 비교할 수 없다.

현재 native tape 사례는 실제 `StreamingSpeakerSession` iterator를 직접 소비하고 Peer 처리를 기다렸다. 일반 streaming `ScopedRecognitionEngine`의 deferred queue는 이 tape 실행이 아니라 별도 real-class custody 회귀가 다룬다. GPU 사례는 실제 scoped engine을 사용했다(`issue264_validation_streaming_after_report.json:915`, `issue264_completion_evidence.json:824–827`).

현재 after 기록은 20개 source 중 14개가 최초 가독 표시 전에 식별, 2개가 500ms 이내, 1개가 500ms–1초 사이, 3개가 미확정이었다. **식별된 개수는 화자 정확도가 아니다.** 두 `No.` 사례의 late 변화는 이미 알려진 acoustic key의 source-order palette 해결이었고, 새로 화자를 알아낸 사례와 구분한다.

실제 GPU의 돌아온 A는 논리적 최초 표시 후 약 639.653ms에 원래 key0로 해결됐다. immutable origin은 당시 unavailable로 남았다. 이는 late assignment의 작동 사례이지 일반 성공률·최대 지연 보장·실제 화면 pixel latency가 아니다.

### 10.3 실제 표시의 성공과 실패를 분리한다

- 현재 원 five-pipeline의 automatic 21개 + named 5개 capture는 미적격으로 남아 있다. host ACK/socket write-drain은 native paint ACK가 아니다.
- 이후 같은 GPU snapshot bytes를 display-on replay한 결과는 4 qualified + 1 superseded-invalid이며 strict exit 1이 유지됐다. 전체 성공으로 바꾸거나 원 pipeline의 음성→pixel 지연을 복구했다고 쓰지 않는다.
- 현재 native-short `No.`/`He washed` display-on control은 startup/reveal 단계에서 실패해 snapshot/readback이 0이었다. 이를 음향 누락·화자 코드 regression으로 단정할 근거는 없다.
- corrected hidden control은 별도 유효 관측이다. 관련 경로의 적용성을 확인한 이전 핀 자료이며, 잘못된 초기 hidden fixture와 혼합하지 않는다.
- 과거 #217의 native D3D11 texture readback/Flet fixed-snapshot 관측은 실제 렌더 근거지만 현재 v3/OpenVR/HMD의 새 실행은 아니다.
- v3 `--check-startup-contract`는 graphics/OpenVR 이전 early exit다. 실제 설치·ordinary app startup·VRChat 공존·wearer readability를 증명하지 않는다.

현재 `speaker_identity_presentation` v3는 matching native binary를 요구한다. wire16/card1이 같아도 구 v2 capability binary를 조용히 허용하지 않는다.

## 11. 보완 우선순위와 완료 판단

다음은 감사로 드러난 수정 의무와 권장 순서다. 이번 문서 작성이 새 구현·API 지출·정책 변경 승인까지 포함하지는 않는다.

| 순서 | 작업 | 완료를 보여 줄 관측 |
|---|---|---|
| 1 | Deepgram missing-final 경계→lawful progress owner 연결 | 같은 PCM/조건에서 기존-final이 없는 확인 경계가 실제 Finalize를 한 번 실행하고, 유용한 source를 앞당기며 successor/tail/order를 보존 |
| 2 | 경로별 조기 진행 범위를 명시 | Scribe minima/one-winner, Gemini independent final/EOS, Soniox DEFER, Qwen #255를 각각 유지. 한 provider 성공을 전역 완료로 쓰지 않음 |
| 3 | Gemini source-idle D2 정합성 수리 | source100/final154에서도 idle은 source 기준. pending final drain과 max age는 계속 보존 |
| 4 | 기존 native provenance 보존 D3/D4 보완 | 실제 받은 activity/제어/delegate/model 사실을 기존 결과 출처에 유지. 없던 final 매핑·word intervals는 만들지 않음 |
| 5 | 완료 기록·구조 설명 D6 정합화 | “경계 기록”, “기존-final release”, “missing-final actuator”의 구현/검증 상태를 별도로 표시 |
| 6 | 남은 #218 비교·owner 선택 | 완성 후보의 필요한 non-VR matched 비교에서 정확도/coverage/짧은 정정/의미 분할/유용한 지연/자원 절충을 분리 제시 |

Parakeet 추가 음향 수리를 여기 끼워 넣지 않는다. D5는 원래 보존 의무의 잔여로 남기고 현 제외 범위를 존중한다. aligner, speculative interim 승격, cloud pre-send hold, per-speaker 병렬 연결, 번역 대기, 새로운 고정 최소 발화 길이를 조기 경계 누락의 대체 해법으로 도입하지 않는다.

### 판정이 바뀌어야 하는 반례

- B의 짧은 정정이 A에 흡수되면 경계·귀속 규칙을 다시 봐야 한다.
- confidence 변화만으로 번역이 잘게 나뉘면 source composition 문제다.
- 경계는 로그에 있지만 실제 제어가 없으면 actuator 완료가 아니다.
- 실제 제어는 빨랐지만 successor backlog나 내용 손상이 늘면 단순 latency 성공이 아니다.
- 텍스트를 먼저 보냈어도 매핑 없는 final에 색을 추정하면 보수 정책 위반이다.
- 모든 결과를 회색으로 만들어 오귀속률만 낮추면 제품 목표 달성이 아니다.
- final 원문에 없는 A를 앱 accounting이나 색상 처리로 복원했다고 주장하면 finality 계약 위반이다.

**최우선 문제는 모델 교체가 아니라 빠진 제어 연결이다.** 그와 독립적으로 Gemini의 유용한 독립-final 경로를 유지하고, 입력 대응이 없는 한계와 실제 보존 누락·idle 불일치를 따로 고쳐야 한다. 공통 source/번역/late 표시의 이미 작동하는 부분은 되돌릴 이유가 없다.

## 12. 다이어리제이션 벤치(#266) 실측·시뮬레이션으로 추가 확인한 문제

2026-10-19 추가. [#266](https://github.com/kapitalismho/PuriPuly-heart/issues/266) 벤치 하네스의 실제 앱 실행과 오프라인 정책 시뮬레이션에서 새로 확인한 내용이다. §1–§11의 판정은 수정하지 않는다. 영향은 §12.6에 따로 적는다. 제품 코드는 바꾸지 않았다.

### 12.1 기준

| 항목 | 기준 |
|---|---|
| 제품 코드 핀 | `6c826f420205fd748a65431108cceb845da56eab` (§2.1과 같음) |
| 하네스 | `experiments/diarization_bench/`, 테스트 전용 로컬 커밋 `8a896b66`–`1ceaaecb` |
| 실행 방식 | 실제 headless 조립. Peer 루프백 소스만 실시간 WAV로 대체하고 읽기 전용 관측기를 붙임. 설정은 제어 카탈로그로만 변경 |
| 코퍼스 | 26케이스, 15.2분. AMI 실제 회의 6구간, LibriSpeech·Common Voice 합성 대화, SAPI 합성 짧은 발화 |
| 이 절의 실측 | 로컬 GPU Qwen3-ASR 1.7B, baseline(화자 끔)과 1.04 완료. 0.64와 다른 경로는 진행 중 |
| 조건 | 비VR. VRChat 종료 상태. VRChat 실행 중에 돌린 초기 스모크는 근거에서 제외 |
| 시뮬레이터 | 녹음한 증거 항목·요청 수명·제출 범위를 제품 `CausalBatchSession`·`SpeakerEvidenceView`에 가상 시계로 재생. ASR은 참조 단어로 대체(oracle) |

시뮬레이터 충실도: 정책을 바꾸지 않은 재생에서 `ls_transfer_gap0_mf`(0.64)의 화자 컷 위치(2.964초), 유닛 경계, 유닛별 판정(uncertain → mixed ×3 → 늦은 색)이 라이브와 일치했다. 시각 차이는 0.2초 이내였다. 26케이스 전체의 충실도 비교는 아직 하지 않았다.

### 12.2 D7: 접합부의 확률 교차가 이후 화자 전환을 지운다

**관측.** 간격 없는 A-B-A-B-A 케이스에서 1.04와 0.64 모두 첫 전환(2.96초)만 화자 컷이 났다. 나머지 네 전환은 6초 안전 컷 단위로만 잘려 `mixed` 회색이 됐다.

**모델은 전환을 봤다.** 녹음한 증거 타임라인에서 슬롯은 0→1→0→1→0으로 올바르게 바뀌었다. 명확한 후속 화자 증거는 후속 발화 시작 후 0.47–0.97초 안에, 모두 자연 종료보다 먼저 도착했다.

**원인.** `SPEAKER/evidence.py:291–295`의 `transitions()`는 competing 프레임을 만나면 `previous`를 지운다. competing은 두 번째 슬롯 확률이 0.35 이상인 프레임이다(`STT/speaker_attribution.py:12,39`). 9.0초 전환에서는 8.96–9.04초에 이런 프레임이 3개 있었다. 그 뒤 B가 명확해져도 전환을 보고하지 않으므로 `STT/batch_speaker.py:125`의 컷 선택기가 자를 근거가 없다. 첫 전환은 접합부에 교차 프레임이 없어서 통과했다.

**의도와 충돌.** 이 동작은 의도된 것이다. `tests/core/test_causal_batch_speaker.py:630`은 교차 프레임 하나만 있어도 A→B 컷을 막고 `mixed`로 두는 것을 고정한다. #211 R1/R8과 #215 R4의 "겹침·백채널은 억지로 자르지 않는다"가 근거다. 그러나 화자가 바뀔 때 A 확률이 내려가고 B 확률이 올라가는 교차는 평범한 교대에서도 거의 항상 생긴다. 0.35 기준은 이 교차를 겹침으로 분류한다. #218에서 합의한 초기 모호성 예산(연속 120ms, 총 200ms 이하)은 화자 귀속에만 적용되고 경계 확인에는 적용되지 않는다.

**규모(라이브, GPU Qwen 1.04, 유효 22케이스).** 연속 전환 116개 중 73개가 분리됐다. 분리되지 않은 31개 중 23개는 자연 종료 전에 명확한 증거가 이미 있었다. 7개는 증거가 없었고, 1개는 증거가 자연 종료 뒤에 도착했다. 나머지 12개는 ASR이 경계 단어를 인식하지 못해 판정에서 빠졌다.

**변형 시뮬레이션.** 같은 1.04 녹음, 26케이스, oracle ASR이다. competing 구간에서 직전 화자를 얼마나 유지하느냐에 따른 결과다.

| 직전 화자 유지 | 현재(항상 지움) | 120ms 이하 | 250ms 이하 | 항상 유지 |
|---|---:|---:|---:|---:|
| 연속 전환 분리 (133개 중) | 89 | 94 | 97 | 97 |
| 연속 전환에서 컷 | 74 | 81 | 83 | 85 |
| 백채널 내부에서 컷 | 11 | 16 | 19 | 29 |
| 겹침에서 컷 | 3 | 5 | 8 | 17 |
| 0.5초 안에 전환이 없는 곳에서 컷 | 10 | 12 | 16 | 19 |
| 올바른 색 단어 | 664 | 711 | 725 | 749 |
| 틀린 색 단어 | 5 | 5 | 5 | 5 |

**판단.** 구현은 의도된 보수 규칙을 따른다. 하지만 그 기준이 평범한 교대의 교차까지 잡아서, 로컬 경로 조기 경계 누락의 주원인이 된다. 허용 폭을 늘리면 백채널과 겹침 근처 컷도 함께 늘지만, 틀린 색은 늘지 않았다. 허용 폭은 #264 R2에 따라 maintainer가 고를 절충이며 이 문서는 승인하지 않는다. 선택한 안은 시뮬레이션이 아니라 라이브로 다시 확인해야 한다. 이 표는 녹음 한 개와 이 코퍼스에 맞춘 수치이므로 과적합 위험이 있다.

### 12.3 D8: 화자 기능을 켜면 ASR final 직후 이벤트 루프가 멈춘다 (원인 확인 중)

**관측.** 하네스 WAV 소스가 기록한 프레임 지연이다. baseline 26케이스는 최대 0.08초였다. 1.04는 26케이스 중 22개가 0.5초, 4개가 1초를 넘었고 최대는 1.39초였다. 1초를 넘은 4개는 하네스 유효성 기준으로 무효 처리돼 §12.5 수치에서 빠졌다.

**시점.** 큰 멈춤은 native final → source 승인 → 최초 가독 표시와 겹친다. 예를 들어 `ls_return_after_gap`에서는 12.4초(0.62초), 22.7초(0.65초), 32.3초(0.77초)에 멈췄고, 각각 승인 직후다.

**비용 후보.** 같은 녹음 입력을 시뮬레이터로 재생하며 제품 코드를 프로파일했다. CPU가 놀고 있는 조건이다. 50초 케이스에서 결과는 다음과 같았다.
- `transitions()`: 4,680회, 누적 4.2초.
- `attribute_speaker`(`STT/speaker_attribution.py:96`): 15회, 누적 2.45초. 호출당 약 0.16초이며, `STT/batch_speaker.py:334`의 `resolve`를 통해 불린다.
- 표시 쪽 `core/speaker_identity.py:211`도 미해결 proof마다 `resolve`를 다시 부른다.
- 증거 보관 상한 2,048프레임(`SPEAKER/evidence.py:33`)이 스캔 길이를 묶지만, 스캔은 Python 루프다.

**위험.** 실제 루프백 소스의 큐는 64 × 1,024프레임이다(`core/audio/desktop_source.py:72–73`). 48kHz에서 약 1.37초분이다. 이보다 긴 멈춤은 운영 환경에서 오디오 프레임 유실로 이어질 수 있다.

**추가 관측(CPU Parakeet).** 화자를 켠 1.04·0.64에서는 22케이스 중 6–7개가 1초를 넘었다. 화자를 끈 baseline에서는 16개가 넘었고 최대 1.30초였다. 다만 이 baseline은 D10의 묵은 증거 프레임을 계속 스캔하는 상태였다. 같은 조건에서 묵은 프레임이 없던 GPU Qwen baseline은 최대 0.08초였다. 프레임 스캔 비용이 원인이라는 가설과 맞는다.

**대조 실행.** 화자를 한 번도 켜지 않은 새 프로세스에서 같은 CPU Parakeet baseline 22케이스를 다시 돌렸다. 화자 컷은 0개였고, 최대 멈춤은 0.13초였다. 오염된 baseline에서는 16개 케이스가 1초를 넘었다. 두 실행에서 하네스 기록기의 작업은 같았다. 새 증거가 없었고 입력 기록도 같았다. 따라서 이 차이는 Parakeet CPU 디코딩이나 기록기가 아니라, 제품이 묵은 증거 프레임을 오디오 입력마다 다시 스캔한 비용으로 설명된다. 화자를 켠 조건에서의 멈춤 원인을 완전히 가르려면 기록 A/B가 여전히 필요하다. 증거 항목 직렬화는 화자를 켠 조건에만 있기 때문이다.

**한계.** 하네스 기록기도 같은 루프에서 직렬화한다. 기록을 켜고 끈 A/B로 원인 분리를 마치기 전에는 제품 결함으로 확정하지 않는다. 표시 경로(presenter, allocator)는 직접 프로파일하지 않았다. 실제 장치에서 프레임 유실이 관측된 것도 아니다.

하네스는 이후 멈춘 케이스를 무효로 빼지 않는다. 대신 채점에 포함하고 장치 큐(1.37초)를 넘은 케이스를 따로 표시한다. 멈춤은 측정 대상인 앱 자체의 동작이기 때문이다.

### 12.4 D9 후보: 몰린 캡처 시각에서 범위 병합 예외로 Peer 캡처가 끝난다

하네스 초기 버전의 WAV 소스는 프레임에 실제 방출 시각을 찍었다. 이벤트 루프 지연 뒤 프레임이 몰려 나간 조건에서 Peer 캡처가 `session_failed`로 끝났다. 원인은 `core/audio/ownership.py:491–516`의 범위 병합 중 `core/audio/format.py:48`이 낸 `ValueError: source monotonic range must be increasing`이다.

소스를 샘플 시계 기준으로 바꾼 뒤에는 재현되지 않았다. 실제 WASAPI 콜백에서 같은 조건이 생기는지는 확인하지 않았다. end < start가 되는 정확한 경로도 밝히지 못했다. 후보로만 남긴다.

### 12.4.1 D10: 화자 기능을 끈 뒤에도 묵은 증거가 새 캡처 세션에 유령 컷을 만든다

**관측.** CPU Parakeet 경로는 1.04 → 0.64 → baseline(화자 끔) 순서로 실행됐다. baseline 22케이스 거의 모두에서 캡처 시작 기준 같은 두 위치(232,544·240,864 샘플, 약 13.5초·14.05초)에 `reason=speaker` 컷이 났다. 두 컷의 `evidence_ready`는 모든 케이스에서 같았다. 그 값은 직전 0.64 조건 마지막 케이스의 증거 시각으로, 약 1,000초 전이다. 화자를 한 번도 켜지 않은 상태로 먼저 돈 GPU Qwen baseline에는 화자 컷이 0개였다.

**경로.**
- 로컬 ASR Peer 세션은 화자 기능이 꺼져 있어도 `CausalBatchSession`으로 감싼다(`app/wiring/wiring_local_asr_provider_runtime.py:186`).
- 이 세션은 공유 증거 뷰를 쓴다. 기능을 끈 뒤에도 마지막 프레임이 그대로 남는다.
- 컷 선택기는 묵은 전환이 현재 요청에 속하는지 `capture_epoch`로만 확인한다(`STT/batch_speaker.py:172`).
- 이 값은 새 캡처 소스마다 0에서 다시 시작한다(`core/audio/source.py:445`). 그래서 다른 캡처 세션의 같은 샘플 위치와 구별되지 않는다.

**영향.** 화자 분석을 켰다가 끈 사용자는 이후 모든 Peer 캡처 세션에서 같은 오프셋에 의미 없는 컷을 받는다. 컷 자체는 텍스트를 지우지 않는다. 하지만 발화를 잘라 번역 단위를 쪼갠다. 기능을 켠 상태에서도, 새 캡처 세션의 새 증거가 도착하기 전 짧은 구간에 같은 혼동이 생길 수 있다. 이것은 추론이며 실측하지는 않았다. #211 R4·R6의 "capture/provider/speaker/activation 세대를 구분하고 오래된 작업을 펜스한다"와 어긋난다.

**하네스 영향.** 화자 조건 뒤에 baseline이 돈 경로(CPU Parakeet, Soniox, Qwen Audio, Scribe)의 baseline은 오염됐다. 화자를 켠 적 없는 새 프로세스에서 다시 측정한다.

### 12.4.2 D11: Soniox Peer는 세션의 첫 문장 뒤 모든 텍스트를 조용히 버린다

**관측.** Soniox 경로에서는 화자 설정과 상관없이 케이스마다 첫 턴만 source로 승인됐다. 1.04·0.64·baseline 모두 같았다. 두 번째 턴부터는 native terminal이 빈 결과였다. 예를 들어 `ls_paused_control`은 5턴 중 1턴, `ami_en2002a`는 75초 동안 source 2개였다. 같은 오디오를 Deepgram·Qwen Audio·Scribe·로컬 경로는 턴마다 정상 승인했다.

**경로.** 한 케이스(5턴)로 프로브했다. 네트워크 Soniox 호출은 1회, 입력은 약 21초였다.
1. 첫 수동 finalize(`<fin>`)를 해결하면, `providers/stt/soniox.py:978`이 조건 없이 세션을 `_independent_source_mode = True`로 바꾼다.
2. 그 뒤 토큰은 턴 update로 나가지 않는다. stream 단위 recognition unit으로만 나간다(`soniox.py:768`, `_release_stream_source`).
3. 엔진은 2–5번째 문장 4개를 `STTRecognitionUnitTerminal`로 정상 방출했다.
4. Peer 승인 단계의 currency 검사(`ORCH/translation_channel_callbacks.py` `_recognition_stream_is_current`)가 4개 모두 거부했다. 원인은 캡처 소유자 쪽이다. `core/runtime/peer_channel.py:943–957`은 `_current_stream_capture_epoch == stream.capture_epoch`를 요구한다. 그런데 이 값은 stream-input 경로(`gemini_transcribe`, `rolling_free`)에서만 갱신된다(`peer_channel.py:280–290`). Soniox에서는 `None`으로 남는다.
5. 거부 경로에는 로그나 진단이 없다.

**영향.**
- Peer에서 Soniox를 쓰면 각 provider 세션의 첫 문장 이후 원문·번역이 나오지 않는다.
- 이 경로의 recognition unit은 `StreamingSpeakerSession`이 화자 정보를 지운 채 전달한다(`STT/streaming_speaker.py`, `STTRecognitionUnit` 분기). 그래서 승인되더라도 Nemotron 귀속은 붙지 않는다.
- #216의 Soniox dual-control 구현이 Peer 승인 경로까지는 연결되지 않은 것으로 보인다. 단위 테스트가 캡처 currency를 대체했는지는 확인하지 않았다.

**하네스 영향.** Soniox 경로의 다이어리제이션 지표는 이 결함 때문에 의미가 없다. 측정을 중단했고, 리포트에는 경로 차단으로 표시한다.

### 12.4.3 D12 후보: 공급자 인증 실패가 공급자를 바꾼 뒤에도 Peer 캡처를 막는다

**관측.** 같은 프로세스에서 ElevenLabs Scribe 화자 조건을 돌리던 중(12:19 KST) `stt.scribe_auth_error`가 3회 반복됐다. 이어 `stt.provider_recovery_exhausted`로 Peer 캡처가 `faulted`가 됐다. 같은 키는 1.5시간 전 baseline에서 정상이었고, 직후 조회에서는 401 `invalid_api_key`였다. maintainer 확인으로는 실행 중 ElevenLabs 계정이 정지됐다. 인증 실패 자체는 제품 문제가 아니며, Scribe는 벤치에서 제외했다.

문제는 그다음이다. 하네스가 `provider.apply`로 Gemini로 바꾸자 결과가 `degraded`였다. 이후 `capture.set peer on`은 매번 `application_operation_failed`로 끝났고, `cause_diagnostics`는 이전 Scribe 인증 실패를 가리켰다. 52개 케이스가 모두 1초 안에 차단됐다.

**한계.** 이것이 의도된 "인증 실패는 채널을 멈춘다"(`docs/cli.md`의 Qwen 복구 절과 같은 계열) 동작인지 확인하지 않았다. 의도라면 다른 공급자로 바꿀 때도 재활성화가 막히는 것이 맞는지가 쟁점이다. 사용자 관점에서는 키가 만료된 공급자를 다른 공급자로 바꿔도 재시작 전까지 Peer가 살아나지 않는다. 결정적 재현과 코드 경로 확인은 하지 않았다.

### 12.5 실측 요약 (예비, 로컬 GPU Qwen)

| 지표 | baseline (26케이스) | 1.04 (유효 22케이스) |
|---|---:|---:|
| 연속 전환 분리 / 전체 | 24 / 133 | 73 / 116 |
| 자연 종료 전 방출 | 24 | 72 |
| 선행 화자 표시 지연 p50 | 2.32초 | 1.23초 |
| baseline 대비 앞당김 p50 | – | 0.32초 |
| 화자 컷 탐지 지연 p50 / p90 | – | 0.77 / 1.04초 |
| 경계 오차 ±200ms 이내 | 16 / 37 | 67 / 80 |
| 화자 컷: 연속 / 백채널 / 겹침 / 전환 없음 | – | 64 / 9 / 3 / 10 |
| 올바른 색 / 틀린 색 (단어 가중) | 0% / 0% | 38.2% / 0.0% |
| 회색 상태 (단어) | not_ready 2,533 | mixed 1,063, unavailable 144, uncertain 97 |
| 표시 전 확정 / 500ms 이내 / 1초 이내 / 회색 유지 | 0 / 0 / 0 / 178 | 50 / 9 / 41 / 132 |
| 표시 불변식 위반 | 0 | 0 |
| 1초 이하 짧은 발화 분리 | 6 / 76 | 16 / 65 |

**주의.**
- 분모가 다르다. 1.04에서는 무효 4케이스가 빠졌다.
- 1회 실행이고, 합성 케이스가 섞여 있다.
- 클라우드 경로와 0.64는 진행 중이다.
- 틀린 색이 0%인 것은 정책이 보수적으로 회색을 많이 남긴 결과이지, 귀속 성공의 증거가 아니다(§11의 "모든 결과를 회색으로 만들어 오귀속률만 낮추면" 반례). 회색은 대부분 `mixed`이며, D7로 전환이 합쳐진 유닛과 겹친다.

### 12.6 기존 판정에 미치는 영향

- §3.1의 "로컬은 실제로 입력을 일찍 닫는다"는 여전히 맞다. 다만 범위를 좁혀야 한다. 접합부에 확률 교차가 생기는 연속 교대에서는 D7 때문에 조기 컷이 대부분 일어나지 않는다. 로컬 경로의 조기 경계도 부분 구현으로 본다.
- §11 우선순위에 다음을 더한다.
  - D7: 경계 확인 규칙을 귀속의 모호성 예산과 정합시키는 결정과 라이브 확인.
  - D8: 기록 A/B로 원인을 가른 뒤 순위를 정한다.
  - D9: 실제 장치 재현 전까지 후보로만 둔다.
  - D10: 캡처 세션 정체성으로 증거를 펜스하고, 화자 기능을 끌 때 컷 경로가 묵은 증거를 쓰지 않게 하는 것. 결정적 재현이 가능하므로 높은 우선순위로 본다.
  - D11: Soniox Peer의 independent recognition unit이 승인되도록 캡처 currency를 맞추고, 거부 경로에 진단을 남기는 것. 텍스트 자체가 사라지므로 화자 작업보다 앞선 최우선 결함으로 본다.
  - D12: 공급자 교체가 이전 공급자의 소진된 복구 상태를 끊는지 결정적으로 재현하고 의도를 확인하는 것.
- 이 절의 시뮬레이션 수치는 정책 후보를 거르는 근거일 뿐이다. #218의 프로파일 선택이나 완료 판정을 대신하지 않는다.
