# PuriPuly Agent 기획 초안

> **상태:** Grill Me를 위한 Working Draft  
> **최초 작성:** 2026-08-29 / **제품 방향 갱신:** 2026-09-12  
> **목적:** 기존 PuriPuly의 실시간 번역·전사 시스템을 기반으로, 대화와 화면을 이해하고 외부 정보 및 애플리케이션 기능을 활용하는 Context-aware Agent로 확장할 가능성을 검토한다.
>
> 이 문서는 최종 요구사항 명세가 아니다. 1~3장은 제품 목표·공통 경험·기술 기반을, **4장은 첫 킬러 기능 가설인 "오늘 뭐 하지?"를 독립적으로** 다룬다. 스마트 글래스의 소프트웨어화라는 기반과 개별 기능 가설을 구분하며, 기능의 구체적인 범위는 Grill Me와 사용자 검증으로 좁힌다.
>
> **Runtime 방향:** Codex는 내장형 기본 경험으로 유지하고 **Hermes·OpenClaw 연결을 주력 제품 경로**로 삼는다. 두 연결은 초기 범위이며 Codex를 필수 중계기로 요구하지 않는다. 기존 기술 검토는 별도 표시가 없는 한 2026-08-29 기준이고, 이번 문서 재배치로 외부 API·지원 버전·약관을 재검증한 것은 아니다.

---

## 1. 목표와 맥락

### 1-1. 목표

PuriPuly를 단순한 실시간 번역 도구에서 **현재 사용자가 나누고 있는 대화와 보고 있는 화면을 이해하는 Context-aware Agent**로 확장한다.

이 방향을 **스마트 글래스의 소프트웨어화**로 본다. 별도 하드웨어를 만드는 것이 아니라, 사용자가 듣고 보는 상황을 함께 이해하고 현재 경험 안에서 도움을 주는 동행 경험을 PC·VR 환경의 소프트웨어로 구현한다.

기존 PuriPuly의 번역 기능은 버리지 않는다. 오히려 기존에 확보한 다음 자산을 에이전트의 감각기관으로 재해석한다.

- Self / Peer 오디오 입력을 별도로 다루는 실시간 음성 파이프라인
- ASR을 통한 지속적인 원문 전사
- 실시간 번역 및 다국어 사용 환경에서 축적한 운영 경험
- 자막·OSC·Overlay 등 VR 환경에 결과를 전달하는 출력 경로
- 여러 외부/로컬 모델을 다뤄온 provider 구조

에이전트의 핵심 목표는 사용자가 매번 상황을 다시 설명하지 않아도 되도록 하는 것이다.

예를 들어 사용자가 AI에게 다음처럼 완전한 프롬프트를 입력하게 하는 것이 목표가 아니다.

> "현재 VRChat에서 다섯 명이 같이 있고 한 명은 Quest 사용자야. 친구들은 공포 월드를 싫어하고 우리는 10시에 이벤트가 있어서 약 40분밖에 시간이 없어. 이 조건에 맞는 월드를 찾아줘."

대신 실제 대화가 이미 다음처럼 흘렀다면,

> A: "Quest 되는 데면 좋겠는데."  
> B: "공포는 별로야."  
> C: "우리 지금 다섯 명이잖아."  
> Self: "10시에 이벤트도 가야 하고."

사용자는 단지 다음처럼 말할 수 있어야 한다.

> **"퓨리야 적당한 데 찾아줘."**

에이전트는 직전 대화, 현재 화면, 현재 세션 상태와 필요 시 외부 정보를 조합해서 사용자의 생략된 의도를 복원한다.

이때 "퓨리야"는 제품이 강제하는 고정 Wake Word가 아니다. **Self ASR의 최종 전사문을 대상으로 동작하는 사용자가 직접 설정 가능한 텍스트 Trigger**다. 사용자는 원하는 예약어 또는 별칭을 설정하거나 음성 Trigger를 완전히 끌 수 있다.

#### VRChat을 넘어설 가능성

초기 진입점과 검증 환경은 VRChat이다. PuriPuly가 이미 가장 강한 자산을 가지고 있고, VR 환경에서는 타이핑과 상황 재설명의 비용이 특히 크기 때문이다.

그러나 장기적으로는 제품의 핵심을 다음처럼 더 넓게 정의할 가능성을 열어둔다.

> **Conversation + Screen + Application State를 함께 이해하는 Context-aware Agent**

이 정의가 유효하다면 적용 대상은 VRChat에만 한정되지 않을 수 있다.

- 다른 Social VR 환경
- Discord Voice + 게임
- 온라인 협동 게임 또는 멀티플레이 세션
- 방송·스트리밍 환경
- 실시간 온라인 음성 대화 환경
- 특정 데스크톱 작업 중 화면과 대화를 함께 이해해야 하는 환경

다만 이는 현재 확정된 제품 피벗이 아니다. **VRChat 특화 Agent가 더 강한 제품인지, 더 범용적인 Conversation/Screen Agent가 더 큰 기회인지 Grill Me를 통해 검증해야 하는 상위 제품 가설**이다.

#### 성공에 대한 초기 가설

성공 여부는 "기능 수"보다 다음 경험이 실제로 성립하는지로 판단한다.

1. 사용자가 AI에게 이미 발생한 상황을 다시 설명하지 않아도 되는가.
2. "이거", "그거", "아까 말한 거", "다른 거"처럼 짧고 생략된 표현이 실제로 유용하게 동작하는가.
3. 대화와 화면을 함께 제공했을 때 각각을 따로 제공하는 것보다 명확한 제품 가치가 생기는가.
4. 에이전트가 PuriPuly 및 외부 도구를 사용해 사람이 직접 여러 창을 오가며 해야 할 일을 줄여주는가.
5. 번역 기능을 사용하던 기존 흐름을 망가뜨리지 않고 자연스럽게 에이전트 기능으로 확장되는가.

첫 킬러 기능으로 탐색할 **"오늘 뭐 하지?"와 새로운 자극의 발견·시작**은 4장에서 별도로 다룬다. 특정 기능의 성공 기준을 이 상위 제품 목표 전체와 동일시하지 않는다.

---

### 1-2. 맥락

#### PuriPuly가 이미 가지고 있는 출발점

PuriPuly는 이미 온라인 음성 대화를 실시간으로 처리하는 제품이다. 따라서 일반적인 Agent 제품과 달리, 사용자가 Agent에게 말을 걸기 전부터 현재 세션의 일부를 알고 있을 수 있다.

기존 파이프라인은 대략 다음과 같은 역할을 한다.

```text
Self / Peer Audio
       ↓
Audio segmentation / VAD
       ↓
ASR
       ↓
Original transcript
       ↓
Translation / Subtitle
```

Agent 관점에서는 이 중 `Original transcript`가 중요한 Context Sensor가 된다.

**Agent가 사용하는 대화의 Source of Truth는 번역문이 아니라 원문 전사문 그대로**로 한다. 최신 범용 모델의 다국어 이해 능력을 활용하고, 번역 과정에서 고유명사·말투·뉘앙스·불확실성이 변형되는 것을 Agent context의 기본 전제로 만들지 않는다.

번역 결과는 계속 사용자에게 제공할 수 있고, 필요하다면 검색 질의 생성이나 fallback에 보조적으로 사용할 수 있지만, Agent context의 canonical text로 취급하지 않는다.

#### 기존 VRChat Companion 생태계가 해결한 문제

VRCX, VRCX-0, VRCNext 등은 이미 다음 문제를 상당 부분 해결한다.

- 친구·월드·인스턴스 정보
- 현재 참가자와 친구 상태
- 방문 및 Social History
- Timeline
- Event 및 Group 정보
- Automation
- VR Overlay
- 외부 Integration

따라서 PuriPuly가 이 모든 기능을 다시 구현하는 것은 목표가 아니다.

특히 VRCX-0는 현재 MCP Server, third-party Integration API, Headless Mode, Social AI를 공개적으로 제공하고 있어 Agent의 선택적 Context Provider로 활용할 가능성이 있다.

반면 PuriPuly가 자연스럽게 확보할 수 있는 고유 Context는 **실시간 대화 그 자체**다.

```text
VRCX / VRCX-0
→ VRChat에서 지금/과거에 어떤 일이 있었는가?

PuriPuly
→ 지금 사람들이 무엇을 이야기하고 있는가?

Screen
→ 사용자가 지금 무엇을 보고 있는가?

Agent
→ 이 정보들을 현재 요청과 어떻게 연결할 것인가?
```

이 역할 분리를 기본 가설로 둔다.

#### Screen을 핵심 Sensor로 취급하는 이유

화면은 부가적인 enrichment가 아니라 **Conversation과 동급의 핵심 Sensor**로 본다.

대화만으로는 다음 표현을 해석할 수 없다.

- "이거 뭐야?"
- "저 버튼 누르면 돼?"
- "여기랑 비슷한 데 찾아줘."
- "지금 화면에 나온 사람/문구가 아까 말한 거 맞아?"
- "이 게임에서 지금 뭘 해야 돼?"

반대로 화면만으로는 방금 사람들이 어떤 조건을 합의했거나 무엇을 가리키고 있는지 알기 어렵다.

따라서 제품의 기본 Context 모델은 다음처럼 본다.

```text
Conversation
     +
Screen
     +
Application / Session State
     +
External Knowledge
     ↓
Agent Reasoning
```

특히 화면이 필요하고 캡처가 허용된 Agent Turn에서 요청 시점의 화면을 timestamp와 함께 고정하면, 사용자가 "이거"라고 말했을 때 **그 말을 했던 순간 실제로 무엇을 보고 있었는지**를 재현할 수 있다.

Screen이 핵심 Sensor라는 것과 매 요청에 화면이 필요하다는 것은 다르다. 화면이 필요 없는 요청은 캡처 없이 동작하고, 화면을 확인할 수 없으면 현재 대상을 임의로 추측하지 않는다.

#### 왜 지금 Agent인가

2026년의 강한 범용 모델과 Agent Harness는 단순 응답 생성이 아니라 다음을 안정적으로 처리하기 시작했다.

- 긴 Context
- 멀티모달 입력
- Tool Calling
- Web Research
- 다단계 추론
- 상태를 유지하는 대화
- 실행 중 Progress Streaming
- Human Approval
- 외부 애플리케이션 통합

따라서 PuriPuly가 모든 "AI 기능"을 직접 규칙으로 구현하는 대신, **풍부한 Context와 안전한 Tool Surface를 제공하고 강한 Agent Runtime이 상황에 맞게 조합하도록 하는 방향**을 검토할 수 있다.

---

### 1-3. 방향성

#### 1. 번역기에서 Agent로의 단절이 아니라 연속적인 확장

```text
Real-time Translation
        ↓
Conversation Awareness
        ↓
Context-aware Assistance
        ↓
Agent
```

PuriPuly가 쌓아온 번역/ASR 인프라는 폐기 대상이 아니라 Agent의 Sensor Layer가 된다.

#### 2. Conversation과 Screen을 두 개의 핵심 Sensor로 둔다

- Conversation: Self / Peer의 원문 최종 전사
- Screen: 사용자가 허용한 Agent 요청 시점의 화면 및 필요 시 추가 캡처

VRChat 상태, VRCX-0, Web Search 등은 이 두 Sensor를 보완하는 구조로 본다.

#### 3. Context 자체를 제품 가치로 본다

핵심 차별화 가설은 "LLM이 있다"가 아니다.

> **사용자가 상황을 다시 설명하지 않아도 되는 Agent**

를 목표로 한다.

즉 모델 자체보다 다음의 품질이 제품 경쟁력이 된다.

- 어떤 정보를 관찰하는가
- 어떤 정보를 언제 저장하는가
- 어떤 정보를 Agent Turn에 넣는가
- 무엇을 필요할 때만 Retrieval하는가
- 정보의 출처와 시점을 얼마나 정확하게 보존하는가
- 사용자 명령과 단순 관찰을 얼마나 확실히 구분하는가

#### 4. Codex는 내장하고 Hermes·OpenClaw 연결을 주력으로 삼는다


| 경로          | 제품에서의 역할                                                               | 실행 책임                                            |
| ----------- | ---------------------------------------------------------------------- | ------------------------------------------------ |
| 내장 Codex    | 외부 Agent를 준비하지 않은 사용자도 앱 안에서 시작하는 기본 경험                                | PuriPuly가 Codex App Server를 통합하고 해당 프로세스와 연결을 관리 |
| Hermes 연결   | 사용자가 이미 쓰는 Hermes에 Conversation·Screen·세션 Context·Tool·HUD를 연결하는 주력 경로 | 연결한 Hermes가 추론과 Tool Loop를 실행                    |
| OpenClaw 연결 | 사용자가 이미 쓰는 OpenClaw에 같은 제품 기능을 연결하는 주력 경로                              | 연결한 OpenClaw가 추론과 Tool Loop를 실행                  |


내장 Codex를 제거하거나 단순 데모로 남기지 않는다. 다만 외부 연결을 Codex 제품 완성 후의 부가 기능으로 미루지 않고, **Hermes와 OpenClaw 각각의 실제 Context-aware 요청→도구 활용→결과 반환 경험**을 초기 완료 조건에 포함한다.

외부 연결은 Codex가 다른 Agent를 호출하는 중계 구조가 아니다. 선택한 Agent가 PuriPuly의 허용된 Context와 Tool을 직접 활용하고 결과를 Agent Chat Phase·VR Overlay로 돌려주는 구조다. 외부 경로에서 Codex 실행·ChatGPT 로그인을 필수로 요구하지 않는다.

PuriPuly는 Context·Tool·제품 권한·출력을 소유한다. Runtime별 연결 방식은 adapter로 분리하되, 모든 Agent 기능을 같은 API로 재구현하거나 기존 Agent의 설정·기억·Skill 시스템을 복제하지 않는다. 연결 계약은 2-5를 따른다.

#### 5. Realtime Audio는 두 번째 경로로 고려한다

MVP의 Voice Input은 기존 PuriPuly ASR을 사용한다.

```text
Voice
→ Self ASR Final Transcript
→ User-defined text trigger
→ Agent command
```

Realtime Audio 모델은 이 구조를 당장 대체하지 않는다.

향후 다음 가치가 충분히 크다고 판단되면 추가적인 Agent Input Mode로 검토한다.

- 말하는 도중의 interruption
- 더 자연스러운 turn-taking
- 억양·강조 등 text에서 사라지는 정보
- 매우 낮은 latency의 conversational interaction
- 직접적인 audio-native multimodal reasoning

즉 기본 구조는 `Transcript-driven Agent`이고, Realtime Audio는 같은 Agent Chat/Context Layer로 연결할 수 있는 확장 경로로 둔다.

#### 6. VRChat은 첫 Context Provider이지 Agent Core 자체가 아니다

Agent Core에 VRChat 특수 개념을 과도하게 하드코딩하지 않는다.

초기에는 VRChat adapter가 첫 번째이자 가장 중요한 구현이지만, 장기적으로 다른 환경의 Context Provider가 들어올 수 있는 방향을 고려한다.

---

### 1-4. 범위

#### PuriPuly가 직접 책임지는 것

- Self / Peer의 실시간 원문 전사
- Agent가 사용할 Conversation Context
- 사용자 정의 Voice Trigger
- Text Agent Input
- Screen Capture 및 command-time screenshot
- Agent Chat Phase
- 내장 Codex lifecycle과 Hermes·OpenClaw 연결·해제·요청·결과 반환
- PuriPuly 자체 기능을 조회·조작하는 typed Tool Surface; Read와 Write 권한을 분리하고 필요한 동작부터 연결
- Agent에 필요한 최소 VRChat Context
- 필요한 범위의 VRChat API 호출과 최소 저장
- Context Retrieval
- Agent 실행 기록 및 Audit
- VR에서 확인할 수 있는 간략한 Agent 상태/결과 출력

#### 직접 구현하되 최소 범위로 제한하는 것

VRCX-0가 없어도 Standalone으로 Agent가 동작해야 한다.

따라서 필요하면 PuriPuly가 VRChat API를 직접 사용해 다음과 같은 최소 정보는 확보한다.

- 현재 World / Instance
- 신뢰 가능한 참가 인원; 이름과 raw log는 기존 scene 경계대로 로컬에 유지
- 사용자가 연결·허용하고 실제 조회가 가능한 경우의 필요한 Friend Presence
- Agent 기능에 직접 필요한 World / Event / Group 정보
- 짧은 Session History 또는 Cache

무엇을 저장할지는 "VRCX와 비슷해질 수 있는가?"가 아니라 **Agent의 실제 Context Quality에 필요한가?**를 기준으로 결정한다.

#### 재발명하지 않는 것

- 수년 단위의 Social Timeline
- 완전한 Friend Manager
- Social Graph 전체
- Avatar Database
- Group Management UI
- 완전한 VRChat Launcher
- VRCX 수준의 장기 History 분석
- 범용 Visual Automation Editor

VRCX-0 등의 외부 프로그램이 존재하면 해당 데이터를 optional enrichment로 활용한다.

#### 현재 범위 밖이지만 열어두는 것

- VRChat 외 Context Provider
- Realtime Audio Agent
- 장기 Persistent Personal Memory
- 고위험 또는 광범위한 자동 행동
- Codex·Hermes·OpenClaw 이외의 추가 Runtime과 Runtime 고유 기능의 완전한 동등 지원
- 임의의 Agent를 위한 범용 공개 Context API·SDK·Plugin 플랫폼

이들은 초기 구조가 불필요하게 막지 않도록 하되 MVP에서 동시에 해결하려 하지 않는다.

현재 인스턴스 전체 인원이 함께 이동할 일행의 인원과 같다고 가정하지 않는다. 확보되지 않은 신원·기기·친구 상태를 추측하거나 Peer 음성을 참가자 신원에 임의로 결합하지 않는다. 장기 이력은 사용자가 연결·허용한 경우의 선택적 보강 정보이며 기본 전제가 아니다.

이 절은 Agent 제품의 책임 범위다. 모든 Tool과 응용 기능을 한 번에 구현한다는 뜻은 아니며, 첫 킬러 기능의 구체적인 지원 범위와 자동 행동 제외는 4-3에서 정한다.

---

## 2. 기능

> **주의:** 아래는 동일한 Context + Agent 기반에서 가능한 경험과 응용군이다. 모두 구현한다는 뜻은 아니며, 구체적인 제품 기능은 검증에 따라 수정할 수 있다. 첫 킬러 기능의 목적·상세 흐름·검증 기준은 **4장**에 분리한다.

### 2-1. 대표 기능

현재 가장 강한 대표 기능 가설은 **Contextual Agent Request**다.

핵심 경험은 다음과 같다.

> 사용자가 AI에게 현재 상황을 다시 설명하지 않고도, 짧고 생략된 명령으로 실질적인 작업을 맡길 수 있다.

#### 가설 A — Conversation → Intent → Research

대화:

> Peer A: "Quest 되는 데로 가자."  
> Peer B: "공포는 별로."  
> Peer C: "우리 지금 다섯 명이잖아."  
> Self: "10시 이벤트 전까지만 할 수 있고."

명령:

> "퓨리야 적당한 데 찾아줘."

Agent가 활용하는 것:

- 위의 **원문** 전사
- 현재 화면
- 현재 World / Instance와 신뢰 가능한 참가 인원·대화에서 확인한 일행 조건
- Web / World / Event Search
- 필요하면 VRCX-0 등 외부 Context

결과:

- 대화에서 implicit constraint 추출
- 후보 조사
- 현재 상황에 맞춘 비교
- Agent Chat Phase에서 근거와 함께 제시
- VR에서는 간략한 상태/결과 확인

#### 가설 B — Conversational + Visual Deixis

> "퓨리야 이거 뭐야?"  
> "이거 아까 걔가 말한 거 맞아?"  
> "여기랑 비슷한 곳 찾아줘."  
> "저 버튼 누르면 돼?"

`이거`, `저거`, `여기`는 Screen에서 grounding하고, `아까`, `걔가 말한 것`, `우리 조건`은 Conversation에서 grounding한다.

이 둘을 동시에 해결하는 것을 주요 UX로 본다.

#### 가설 C — Contextual Recall / Recovery

> "퓨리야 나 없는 동안 뭐 결정됐어?"  
> "아까 Quest 조건을 어떻게 정했지?"  
> "그 프로젝트 이름 뭐였어?"  
> "아까 말한 링크 찾아줘."

실시간 음성 대화가 휘발되는 문제를 Agent가 보완한다. 허용된 세션 Context 안에서만 조회하고, 화자 신원은 신뢰 가능한 provenance가 있을 때만 사용한다.

#### 대표 기능이 만족해야 할 조건

- ChatGPT를 별도 창에서 여는 것보다 명확히 편해야 한다.
- 단순 검색창이나 Chatbot과 달리 **Puri가 이미 알고 있는 Context**가 결과를 바꿔야 한다.
- 30초 내 데모만 봐도 차이가 이해되어야 한다.
- VR에서 실제로 반복 사용하고 싶은 행동이어야 한다.
- 특정 World나 특정 Script에만 의존하지 않아야 한다.

이 조건을 만족하지 못한다면 대표 경험의 방식과 기능 가설을 Grill Me를 통해 수정한다. 이 절은 공통 Context-aware 경험을 설명하며, 이를 먼저 체감하게 할 구체적인 킬러 기능은 4장에서 다룬다.

---

### 2-2. 확장 기능

아래는 동일한 Context + Agent primitive에서 파생될 수 있는 응용군이다. 모두 구현한다는 의미는 아니다.

#### Discovery

- 현재 대화 조건에 맞는 World 찾기
- 현재 Party에 맞는 Activity 찾기
- 현재 시간과 이후 일정을 고려한 Event 찾기
- 관심사·활동에 맞는 Group과 참여 경로 찾기
- 사용자가 허용하고 확인 가능한 일행 Context와 결합한 추천

첫 킬러 기능으로 검토하는 Discovery의 목적·상세 흐름·검증 기준은 4장에 모은다.

#### Research / Fact Check

- 방금 대화에 나온 주장 확인
- 고유명사·프로젝트·제품·행사 찾기
- 대화에 등장한 링크나 공식 문서 탐색
- 여러 출처를 비교해 결론 정리

#### Recall

- 최근 대화에서 특정 정보 찾기
- 어떤 조건이 논의됐는지 찾기; 화자 조회는 신뢰 가능한 provenance 범위에 한정
- AFK 동안의 핵심 결정 복구
- 이전 Agent 결과 다시 참조

#### Planning / Group Assistance

- 여러 참가자가 말한 조건을 한 번에 합치기
- 시간 제한을 고려한 다음 행동 추천
- 서로 충돌하는 선호 정리
- 그룹이 결정하지 못한 선택지 비교

#### Current-view Assistance

- 현재 UI / 안내판 / 게임 화면 설명
- 화면의 특정 요소와 대화 맥락 연결
- "지금 뭘 해야 하는가?"에 대한 상황 기반 도움
- 화면 속 항목을 Web Research와 연결

#### PuriPuly Control

예:

> "번역 잠깐 꺼줘."  
> "상대 자막만 켜줘."  
> "지금 어떤 ASR 쓰고 있어?"  
> "이 마이크로 바꿔줘."  
> "방금 오류 왜 났는지 봐줘."

에이전트가 PuriPuly의 내부 상태를 읽고 기능을 조작할 수 있다면 설정 UI를 직접 찾아다니는 비용을 줄일 수 있다.

#### Persistent / Scheduled Behavior

향후에는 자연어 요청을 지속적인 조건이나 일정으로 변환할 가능성도 있다.

> "좋아하는 친구가 들어오면 알려줘."  
> "다음 이벤트 10분 전에 알려줘."

단, 자동 행동은 단순 조회보다 권한과 안전성 문제가 크므로 MVP의 핵심으로 두지 않는다.

#### VRChat 밖으로의 확장 가설

동일한 primitive가 VRChat 밖에서도 성립하는지 검토한다.

예:

```text
Discord Voice conversation
+
Current game / desktop screen
+
Application context
+
Agent tools
```

이 조합이 충분히 강하다면 PuriPuly의 타겟 자체가 확장될 수 있다.

반대로 VRChat 특화 Context가 제품 가치의 대부분이라면 범용화를 시도하지 않는 것이 더 나을 수 있다. 이 역시 Grill Me 대상이다.

---

### 2-3. 상호작용

#### Text Input

별도의 **Agent Chat Phase**에서 직접 텍스트를 입력한다.

Text Input은 항상 명시적인 사용자 Instruction으로 취급한다.

#### Voice Input

Voice Input은 기존 Self ASR의 **최종 전사문**을 사용한다.

```text
Self Audio
   ↓
ASR Final Transcript
   ↓
User-defined Trigger Detection
   ↓
Agent Command
```

예약어는 사용자가 직접 설정한다.

예:

- "퓨리야"
- "Puri"
- "Hey Puri"
- 사용자가 원하는 임의의 구문

예약어는 음향 Wake Word 모델이 아니라 **전사문에 대한 deterministic text trigger**다.

초기 규칙은 다음을 고려한다.

- `SELF` 채널만 Agent 명령을 발생시킬 수 있다.
- `PEER`에서 동일한 문자열이 나와도 절대로 명령으로 실행하지 않는다.
- Partial transcript가 아니라 Final transcript만 Trigger 판단에 사용한다.
- 예약어와 명령이 같은 utterance에 있으면 예약어 이후 문장을 명령으로 사용한다.
- 예약어만 별도 utterance로 끝난 경우 짧은 Armed Window를 두고 다음 Self Final을 명령으로 받을 수 있다.
- Trigger alias, enable/disable, matching policy는 사용자 설정이 가능해야 한다.
- Agent로 소비된 command utterance는 기본적으로 일반 번역/VRChat Chatbox 출력으로 흘리지 않는다.

#### Follow-up Interaction

Agent Chat Phase의 현재 thread와 최근 결과를 유지하여 다음과 같은 짧은 follow-up이 가능해야 한다.

> "2번 자세히."  
> "그건 빼고."  
> "좀 더 늦게 시작하는 걸로."  
> "그거 진짜 맞아?"

#### Realtime Audio 확장

향후 Realtime Audio Mode를 도입하더라도 Text/Transcript 입력과 경쟁하는 별도 제품으로 만들기보다 동일 Agent Session에 들어오는 추가 입력 경로로 본다.

---

### 2-4. 컨텍스트

Agent Context를 크게 다섯 범주로 본다.

#### Conversation

- Self 원문 Final Transcript
- Peer 원문 Final Transcript
- timestamp
- channel / speaker provenance
- source language
- utterance identity
- 세션과의 연관 정보

**번역문이 아니라 원문을 기본 Context로 사용한다.**

#### Screen

Screen은 Conversation과 동급의 핵심 Sensor다. 사용자가 보고 있는 대상과 대화 맥락을 함께 이해하는 데 사용한다. 캡처 허용과 대상 설정을 존중하며, 필요한 요청에서는 **명령 시점의 화면과 timestamp를 묶은 Context Checkpoint**를 사용한다.

화면이 필요 없는 요청도 동작해야 한다. 캡처가 불가능한 시각적 질문은 현재 대상을 임의로 추측하지 않는다.

초기에는 다음 전략을 우선한다.

- 사용자가 지정한 target window 또는 VRChat viewport를 capture
- 전체 데스크톱 무조건 capture는 피함
- Puri Agent UI 자체가 feedback loop로 계속 들어가지 않도록 가능하면 제외
- Agent가 필요하면 turn 중 추가 screenshot을 요청할 수 있도록 확장

내장 Codex 경로에서는 로컬 screenshot을 `localImage`로 제공할 수 있다. Hermes·OpenClaw에는 실제 지원하는 이미지 전달 경로를 확인한 뒤 적용한다. PuriPuly의 로컬 파일 경로가 다른 호스트의 Agent에서도 읽힌다고 가정하지 않는다.

#### Session / Application State

- 현재 World / Instance
- 신뢰 가능한 참가 인원과 대화에서 확인된 실제 일행의 구성·제약
- 현재 활동과 요청에 필요한 세션 내 사건
- PuriPuly translation / ASR / capture / provider 상태
- 현재 Agent Session 상태

#### External Context

- Direct VRChat API
- VRCX-0 optional integration
- Web Search
- World / Event / Group 데이터와 요청에 필요한 외부 지식; 출처·확인 시점 포함
- 기타 앱별 Context는 별도 확장 가설로 검토

#### Agent State

- 사용자가 선택한 Agent 경로·연결과 그 경로에 속한 session / thread / request 식별자
- 최근 Agent 결과
- 이전 요청의 결과·근거와 사용자의 후속 선택
- 현재 진행 중 Tool Call
- 승인 대기 상태

기능별 상태는 공통 세션 위에 구분해서 둔다. Discovery 후보·거절·방문·참여 결과는 4-6에서 다룬다.

후속 요청은 해당 결과를 만든 연결의 세션에 보낸다. 경로를 바꿨다고 이전 Agent의 대화나 외부 기억을 자동 복사하지 않는다.

#### Context Selection 원칙

모든 Context를 매 Turn 전체 dump하지 않는다.

```text
Small always-on context
        +
Turn checkpoint
        +
Context retrieval tools
        ↓
Agent
```

형태를 지향한다.

짧은 최근 대화는 직접 제공할 수 있지만, 긴 세션은 `recent`, `search`, `around timestamp`처럼 필요한 부분을 가져오는 Query Surface가 필요하다.

---

### 2-5. 외부 연동

#### Agent 연결 계약 — 주력 Hermes·OpenClaw, 내장 Codex

세 경로의 공통 제품 기반은 **Conversation·Screen·세션 상태를 활용한 상황 인지형 도움과 PuriPuly HUD 출력**이다. 요청의 의미 복원·필요한 도구 활용·후속 대화로 이를 검증한다. Runtime의 내부 API나 모든 고유 기능을 동일하게 만드는 계약은 아니다.

**PuriPuly가 제공하는 것**

- 명시적인 사용자 요청과 필요한 세션 Context
- 원문·화면·상태·외부 정보의 출처, 시점, 명령/관찰 구분
- 허용된 대화·세션 조회와 화면 입력, 필요한 애플리케이션·외부 Tool
- Agent 상태·결과 표시와 현재 요청에 연결된 후속 대화
- 연결별 접근 권한, 해제, 로컬 보관·전송 제어

**선택한 Agent가 담당하는 것**

- 추론, 검색·Tool 선택과 실행 루프
- 자신의 대화 세션과 내부 실행 상태
- 요청에 대한 결과와 근거·불확실성 반환

#### Hermes·OpenClaw — 주력 연결

사용자가 사용하는 Hermes 또는 OpenClaw 인스턴스를 연결한다. 단순히 Context를 내보내는 것으로 끝내지 않고, PuriPuly의 Text / Voice 요청을 받아 Context와 허용된 Tool을 활용하고 결과를 Agent Chat Phase와 VR Overlay에 반환해야 한다.

- 연결 설정에서 대상과 연결 상태를 확인하고 명시적으로 선택·해제할 수 있다.
- 하나의 요청은 사용자가 선택한 하나의 경로로 보낸다. 같은 대화를 모든 Agent에 동시에 전송하지 않는다.
- 외부 연결은 Codex 실행·인증 없이 동작한다. 연결 실패 시 다른 Runtime으로 몰래 전환하거나 Context를 재전송하지 않는다.
- Context 조회 권한과 HUD 결과 반환 권한을 구분한다. 결과를 반환할 수 있다는 이유로 PuriPuly 설정 변경이나 VRChat 외부 행동 권한을 얻지 않는다.
- 결과는 원래 요청·연결·세션에 대응시킨다. 연결을 해제하거나 요청을 취소한 뒤 도착한 응답을 현재 답변으로 표시하지 않는다.
- 실행 중임·완료·실패를 구분하고 취소 요청을 지원한다. 세부 progress streaming이나 실제 원격 실행 중단은 각 Runtime의 확인된 지원 범위를 표시하며, 로컬 표시 취소를 원격 중단 완료로 표현하지 않는다.
- 연결 해제 후에는 새 Context 전송·조회 권한을 차단한다. 이미 외부에 전달된 데이터까지 삭제됐다고 주장하지 않는다.

Hermes·OpenClaw 자체의 설치·호스팅·모델 계정·장기 기억 관리 전체를 PuriPuly 안에 재구현하지 않는다. 기존 Agent의 개인화가 있더라도 이번 세션의 명시적 제약과 PuriPuly의 권한 경계를 무시할 근거가 되지 않는다.

MCP, 해당 Runtime의 공식 API, Skill / Plugin 연결은 가능한 구현 후보이지 확인된 지원 사실이 아니다. **두 Runtime 각각에서 요청 전달·Context/Tool 접근·결과 반환이 가능한 실제 경로와 지원 버전을 먼저 검증**한다. MCP로 Tool을 노출했다는 사실만으로 Agent 호출과 HUD 응답 경로까지 완성됐다고 보지 않는다.

#### Codex — 내장형 기본 경험

외부 Agent 연결 없이 사용할 수 있는 앱 내 경로다. 사용자별 공식 인증과 Codex App Server의 thread/turn·추론·Tool Loop·streamed events·approval을 통합한다.

내장 경로도 같은 Context·Tool·권한·출력 계약을 사용한다. Hermes·OpenClaw 사용을 위한 필수 중계기나 계정 계층으로 만들지 않으며, 외부 연결이 없어도 검증 대상으로 정한 Context-aware 요청과 후속 도움을 완결한다.

#### Direct VRChat API

기존 앱 상태만으로 부족한 VRChat 정보를 필요한 범위에서 보완한다. VRCX-0 설치를 전제로 하지 않는다.

조회 가능 항목과 인증·이용 조건은 통합 전에 확인한다. API가 모든 일행 조건·월드 활동 안내·이벤트 일정·그룹 참여 정보를 제공한다고 전제하지 않는다. 확보할 수 없는 항목은 확인된 외부 정보원으로 보완하거나 미확인으로 남긴다.

#### VRCX-0

선택적 Context Provider. 사용자가 연결과 활용을 허용한 상태·방문 정보·favorite 등으로 필요한 Context를 보강할 수 있다.

VRCX-0가 없어도 Agent의 기본 Context-aware 요청 경로는 동작해야 한다. 장기 Social Graph나 친구별 취향 수집으로 MVP 범위를 넓히지 않는다.

#### Web / External Search

Research / Fact Check / Discovery 등 요청에 필요한 Web Search와 domain-specific search를 Agent Tool로 제공한다. 출처와 최신성·불확실성을 보존하며, 킬러 기능에 필요한 검색 데이터의 상세 범위는 4-5에서 다룬다.

#### Realtime Audio

향후 low-latency conversational mode 또는 prosody가 실제 사용자 가치를 만든다고 검증되면 추가한다.

---

## 3. 기술적인 면

### 3-1. 필요 조건

#### 세션 한정 Context Logging

현재의 진단 로그만으로는 Agent Context를 만들 수 없다.

Agent를 위해 별도의 **Session / Conversation Event Layer**가 필요하다.

최소한 다음을 안정적으로 기록할 수 있어야 한다.

```text
Conversation
- self final transcript
- peer final transcript
- timestamp
- utterance id
- source language
- provenance / channel

Session
- session start/end
- world / instance change
- relevant participant change
- relevant VRChat state change

Agent
- command detected
- text command
- trigger used
- context checkpoint
- screenshot reference (when captured)
- thread / turn
- tool calls
- approvals
- result / failure
```

이 기록은 Debug Log와 목적이 다르다.

```text
Diagnostic Log
→ 프로그램이 왜 실패했는가?

Session / Context Log
→ 그 순간 무슨 일이 벌어지고 있었는가?

Agent Audit
→ Agent가 무엇을 보고 무엇을 실행했는가?
```

세 개를 개념적으로 분리할 필요가 있다.

원칙:

- Raw Audio는 기본적으로 저장하지 않는다.
- Transcript는 로컬 / session-scoped가 기본이다.
- Peer transcript 저장은 명시적인 Privacy Policy와 사용자 제어가 필요하다.
- 모든 것을 영구 보존하지 않고 bounded session store부터 시작한다.
- timestamp, source timestamp, observed time, ingestion order를 필요에 따라 구분한다.
- Agent가 전체 로그 파일을 직접 grep하게 만들기보다 구조화된 Query Interface를 제공한다.

기능별 결과 기록은 공통 Context·Audit와 구분한다. 첫 킬러 기능의 기록·평가 경계는 4-6과 4-7을 따른다. 이 Event Layer가 영구적인 대화 저장소나 중앙 수집 서버를 뜻하지 않는다.

#### Conversation Context Query

Agent가 다음과 같은 동작을 안정적으로 할 수 있어야 한다.

- 최근 N분 / N개 발화 읽기
- 특정 단어·주제 검색; 화자 식별은 신뢰할 수 있는 provenance가 있을 때만 사용
- 특정 timestamp 주변 대화 읽기
- Self / Peer 구분
- 원문 그대로 반환
- provenance 반환

#### Screen Sensor

- 지정 window / viewport capture
- 캡처가 허용되고 필요한 요청의 command-time screenshot
- timestamp와 capture target 보존
- 추가 on-demand capture
- 민감한 다른 창을 실수로 포함하지 않는 capture policy
- 선택한 Runtime이 접근 가능한 허용된 이미지 전달 경로; 로컬 파일과 외부 전송 구분

#### LLM-friendly Application Interface / Headless Control Plane

Agent가 GUI 요소를 클릭하거나 화면을 scraping해서 PuriPuly를 조작해서는 안 된다.

PuriPuly 기능을 **stable, typed, structured interface**로 제공해야 한다.

큰 범주는 다음처럼 분리할 수 있다.

```text
Query
- runtime status
- provider status
- capture status
- current settings/state
- recent conversation
- current context

Command
- translation on/off
- STT on/off
- peer translation on/off
- capture target
- text submission
- provider-related intentional operations
- overlay / agent presentation operations

Context
- transcript retrieval
- screenshot
- VRChat state
- session events

Lifecycle
- start / stop / recover
```

목표는 "CLI를 많이 만드는 것"이 아니라 **GUI와 독립된 Application Control Plane**을 확보하는 것이다.

구현 형태 후보:

- Local IPC / JSON API
- MCP Server
- Structured CLI JSON mode
- Headless runtime
- 위의 조합

Hermes·OpenClaw와 내장 Codex가 같은 Tool 의미를 사용할 수 있도록, **PuriPuly의 기능을 MCP 또는 동등한 typed surface로 노출**하는 방식을 검토한다. 실제 전송 방식은 각 Runtime의 확인된 연결 경로에 맞춘다.

전체 Control Plane을 첫 연결 전에 완성할 필요는 없다. 필요한 Query·Context·Command부터 연결하며, 공개 SDK·Plugin 플랫폼 구축과 앱 내부의 명시적인 Tool 경계는 구분한다. Context 조회·HUD 결과 반환·Write 권한을 각각 분리한다.

Read tool과 Write tool은 분리하고, 중요한 Write는 Agent Runtime의 Approval과 PuriPuly 자체 policy를 모두 통과하게 한다.

#### Direct VRChat Context Provider

Standalone으로 필요한 최소 VRChat Context를 제공해야 한다.

- 기존 상태에서 신뢰 가능한 현재 World / Instance와 참가 인원
- 접근이 확인된 월드·이벤트·그룹 후보 검색·상세 정보 조회
- API에서 제공하지 않는 필요한 지식을 보완할 외부 정보원과의 연결
- 필요한 최소 cache와 rate limit 처리
- source/freshness와 unavailable 상태 표시

VRCX-0 연결 여부가 core 기능의 전제 조건이 되어서는 안 된다.

#### Agent Runtime Integration

Hermes·OpenClaw는 2-5의 연결 계약을 각각 충족해야 한다. 내장 Codex는 앱이 관리하는 process lifecycle·사용자 인증·thread/turn·streamed events·approval·취소·오류 처리를 연결한다.

PuriPuly는 내장 프로세스와 외부 연결의 수명을 구분한다. 외부 Agent를 연결했다고 그 프로세스나 호스팅 환경의 소유권을 가져오지 않는다.

각 adapter는 요청과 세션의 대응, Context/Tool 접근, 결과 반환, 연결 실패와 지원 기능을 제품 경계에 전달한다. Codex의 thread ID나 event 형식을 다른 Runtime에 그대로 강제하지 않는다. 필요한 입력이나 결과 반환 기능이 없는 경로는 미지원·차단 상태로 남기며 연결 완료로 판정하지 않는다.

#### Agent Chat Phase

기존 번역 화면과 별도의 Agent 전용 UI가 필요하다.

필요 정보:

- 사용자 질문
- Agent 답변
- 진행 상태
- Tool 실행
- 승인 요청
- 검색 결과
- 이미지 Context
- 오류 및 retry
- thread history

VR Overlay는 전체 Chat UI를 복제하기보다 진행 상태와 짧은 결과를 확인하는 glanceable surface로 시작할 수 있다.

#### Authority / Security Model

명령과 관찰을 구조적으로 구분해야 한다.

```text
Direct Agent Chat text
→ AUTHORITY_USER

SELF transcript + configured trigger
→ AUTHORITY_USER

Normal SELF conversation
→ OBSERVATION

PEER transcript
→ OBSERVATION

Screen
→ OBSERVATION

VRChat / VRCX-0 / Web data
→ OBSERVATION
```

Peer가 사용자의 예약어를 말하더라도 실행 권한을 얻지 못해야 한다.

Web page, World description, 이벤트·그룹 소개, screen text 등이 Agent에게 새로운 "명령"을 주는 Prompt Injection 경로가 되지 않도록 provenance가 필요하다.

#### Privacy

Conversation과 Screen을 핵심 Sensor로 삼는 만큼 기존 번역기보다 더 명확한 Privacy Boundary가 필요하다.

- 어떤 Context가 로컬에 저장되는가
- 어떤 Context가 선택한 Codex/OpenAI 또는 Hermes·OpenClaw 연결 대상과 그 모델 provider로 전송되는가
- Screen Capture 범위
- Peer transcript 전송
- Session retention
- User delete/disable controls
- Diagnostic telemetry와 conversation content의 분리

기존 `ARCHITECTURE.md`의 scene 경계에서는 이름과 raw log를 로컬에 유지한다. Agent 연결을 이유로 이를 자동 전송하거나 Peer 음성을 특정 참가자 신원과 임의로 결합하지 않는다. 세션 종료·삭제 시 로컬 Context뿐 아니라 screenshot, 결과 근거 참조, Runtime thread로 이미 전달된 정보의 보관 차이와 삭제 가능 범위를 설명해야 한다.

외부 Agent의 기존 보관·기억·도구 정책은 PuriPuly의 로컬 세션 삭제와 별개다. 연결 전에 데이터 수신 대상과 전송 항목을 알리고, 확인하지 못한 외부 보관 정책은 미확인으로 표시한다. 외부 Agent가 자체적으로 가진 도구까지 PuriPuly가 통제한다고 주장하지 않는다. 연결 방식이 원격 전송을 요구하면 그 접근 제어와 전송 보호를 해당 adapter의 선행 조건으로 검증한다.

---

### 3-2. 현재 구조와의 연결

현재 시스템의 구조적 기준은 `ARCHITECTURE.md`다. 이 절은 해당 시스템 맵과 이번 제품 목표의 연결이며, Agent 구현이 이미 존재한다는 주장이 아니다. 세부 코드 상태와 외부 API는 구현 착수 시 확인한다.


| 연결 경계                                                   | Context-aware Agent를 위한 연결                                             |
| ------------------------------------------------------- | ---------------------------------------------------------------------- |
| Self / Peer 캡처·번역 owner                                 | 번역 흐름을 유지하면서 허용된 원문 Final Transcript를 세션 Context로 연결                   |
| `UiApplicationPort`와 UI-facing application boundary     | 전체 UI Port를 노출하지 않고 Agent용 Query·Context·Command를 typed operation으로 분리 |
| 화면 입력과 Context Checkpoint                               | Conversation과 동급인 Screen Sensor를 새로 연결하되 허용된 캡처 대상·시점·수명을 보존           |
| VRChat scene owner의 process-lifetime immutable snapshot | 신뢰 가능한 상태를 재사용하고 월드 검색 등 없는 정보만 adapter로 보완                            |
| Output runtime과 Overlay owners                          | 번역 출력과 Agent 상태·결과의 표시 책임을 구분                                          |
| Settings / Secrets owner                                | 사용자 설정·인증 정보를 기존 소유권과 저장 경계 안에서 관리                                     |
| ports/adapters와 composition root                        | 외부 Runtime·Context 정보원을 adapter로 연결하고 장기 자원 owner를 명시                  |


기존 scene snapshot은 `ready`일 때만 local user를 포함한 인원을 노출하며 이름과 raw log는 로컬에 둔다. 이 인원은 현재 인스턴스의 값이지 함께 이동할 일행이나 개인별 플랫폼의 증거가 아니다.

새로 필요한 제품 책임은 **bounded session context, Screen Sensor와 Context Checkpoint, typed Application Tool Surface, 내장 Codex lifecycle, Hermes·OpenClaw 연결과 요청/응답 lifecycle**이다. 기능별 데이터와 상태는 이 경계 위에 추가하며 첫 킬러 기능은 4장에서 구체화한다. 구체적인 모듈 분해는 구현 시 정하되, 별도 범용 Framework를 만드는 근거로 삼지 않는다.

Core가 UI·composition에 의존하지 않는 방향을 유지하고, 런타임 교체를 넘어 오래된 owner 참조를 붙잡지 않는다. Agent를 위해 기존 scene 수집기나 번역 파이프라인을 중복 구현하지 않는다.

이번 변경은 계획서의 목표와 향후 책임을 정렬하는 것이며 현재 runtime 구조를 변경하지 않는다.

---

### 3-3. 에이전트 구조

제품 기반은 Conversation·Screen·Application / Session State를 Agent와 HUD에 연결한다. 아래는 공통 Context-aware 요청을 실행하는 구조 초안이며, 분기는 동시 실행이 아니라 사용자 선택을 뜻한다.

```text
Text / Self ASR Final + Trigger
              ↓
PuriPuly Request + Permission-scoped Conversation / Screen / Session Context
              ↓
      Selected Agent Connection
      ├─ Hermes connector     [주력 외부 연결]
      ├─ OpenClaw connector   [주력 외부 연결]
      └─ Codex App Server    [내장형 기본 경험]
              ↕
Puri Tools / VR Context / Web·External / Allowed Screen
              ↓
Agent Result + Evidence / Follow-up
              ↓
PuriPuly Agent Chat Phase / VR Overlay
```

#### Application이 소유하는 것

- 현재 화면
- 현재 대화
- PuriPuly runtime
- VRChat Context
- Product permissions
- Tool definitions
- UI
- PuriPuly가 보관하는 Context와 연결별 접근·전송 권한
- Context와 외부 정보의 출처·최신성
- 기능별 결과와 세션 내 사용자 피드백
- 요청·선택한 연결·외부 세션의 대응과 현재 결과의 표시 수명

#### 선택한 Agent가 소유하는 것

- 자체 session / thread와 실행 상태
- Agent reasoning loop와 Tool selection
- 지원하는 event·approval·중단 기능
- 자체 모델·기억·Skill 설정과 외부 보관 정책

내장 Codex 경로에서는 App Server의 thread/turn/event를 통합한다. Hermes·OpenClaw 경로에서는 각각의 외부 실행 상태를 연결하며, PuriPuly가 그 내부 상태 전체를 복제하지 않는다.

#### 제품 Context는 공유하고 Runtime 연결은 분리한다

세 실제 경로에 필요한 adapter와 공통 제품 경계는 만들되, 미래의 모든 Agent를 위한 완벽한 abstraction은 만들지 않는다.

다음은 Codex나 특정 외부 Runtime 전용 구조 안에 묻히지 않도록 한다.

- ConversationStore
- Screen Sensor
- VR Context Provider
- Puri Application Control Plane / typed Tool Surface
- Authority/Privacy rules

이것들은 제품 자체의 자산이다. 특정 Runtime이나 개별 기능의 내부 구현에 종속시키지 않는다.

---

### 3-4. 입력과 컨텍스트

#### Text Command

Agent Chat Phase의 직접 Text Input.

항상 User Authority.

#### Transcript Command

사용자 설정 Trigger를 Self Final Transcript에서 감지한다.

예:

```text
raw self transcript:
"아 그러고 보니까 퓨리야 아까 얘기한 월드 찾아줘"

trigger:
"퓨리야"

agent command:
"아까 얘기한 월드 찾아줘"
```

Agent에는 필요하면 `raw_transcript`, `trigger`, timestamp, utterance ID를 provenance로 함께 보존한다.

#### Conversation Input

원문을 그대로 사용한다.

예:

```text
[21:14:02][peer][ja]
Questでも行けるところがいいな

[21:14:11][peer][en]
I don't want anything too scary.

[21:14:20][self][ko]
10시에 이벤트도 가야 하고.
```

Agent에게 번역된 단일 언어 transcript를 canonical context로 만들지 않는다.

#### Screen Input

캡처가 허용되고 화면이 필요한 Agent Request에서는 명령 시점의 screenshot을 checkpoint한다. 화면이 필요 없는 요청은 이 입력을 기다리지 않는다.

Codex App Server는 현재 Turn Input으로 다음 형태를 공식 지원한다.

- text
- image URL
- local image path

내장 Codex에서는 로컬 screenshot을 `localImage`로 전달한다. Hermes·OpenClaw에는 지원하는 이미지 입력·접근 범위를 검증하고 명시적으로 허용된 경로로만 전달한다. 이미지 전달이 미지원이면 해당 시각적 요청의 한계를 표시하고 텍스트만으로 가능한 요청과 구분한다.

#### Context Freshness / Provenance

모든 중요한 Context는 최소한 다음 정보를 가질 수 있어야 한다.

```text
value
source
observed_at
source_timestamp (if available)
session_id
authority
freshness / revision (where useful)
```

Agent가 "현재"와 "과거"를 혼동하지 않도록 source time을 보존한다.

#### Long Session

긴 세션 전체를 매 Turn prompt로 반복하지 않는다.

- 최근 일정 구간은 직접 Context
- 오래된 대화는 retrieval
- previous Agent result는 thread state
- 외부 상태는 필요할 때 tool call

방식으로 token/context budget을 관리한다.

---

### 3-5. 실행과 출력

#### Tool Surface

초기 Tool은 Read-heavy로 시작한다. 아래는 구현 후보이지 확정 공개 API나 지원 완료 목록이 아니다.

예:

```text
Conversation
- get_recent_transcript
- search_transcript

Screen
- get_current_view
- capture_current_view

PuriPuly
- get_status
- get_provider_status
- get_capture_status

VRChat
- get_current_room
- get_current_context
- get_relevant_friend_state (연결·허용·조회 가능 범위에 한정)
- get_world_info

External
- web search / research
```

Write Tool은 점진적으로 추가한다. 앱 내부의 명시적인 typed operation부터 검토하고, 외부 Agent의 응답만으로 사용자 승인을 대체하지 않는다.

```text
PuriPuly
- set_translation_enabled
- set_stt_enabled
- set_peer_translation_enabled
- change_capture_target
- submit_text
- control_overlay
...
```

장기적으로는 "Agent가 PuriPuly를 거의 전부 조작할 수 있다"를 목표로 하되, 임의의 GUI 클릭이 아니라 **명시적인 typed operation**을 늘려가는 방식으로 한다.

#### Approval

중요한 상태 변경은 다음 두 층을 고려한다.

1. 선택한 Runtime이 제공하는 approval 기능
2. PuriPuly 자체 command policy

Read-only 조회와 reversible local toggle은 낮은 friction으로, 외부 행동이나 중요한 변경은 더 강한 confirmation으로 분류할 수 있다.

Runtime approval 지원 여부와 무관하게 PuriPuly 자체 접근·행동 정책은 우회할 수 없다. 외부 Agent의 자체 도구까지 PuriPuly가 통제한다고 주장하지 않는다. 첫 킬러 기능에서 제외하는 자동 행동은 4-3의 경계를 따른다.

#### Agent Chat Phase

Agent 기능의 메인 UI.

번역 화면과 역할을 분리한다.

Agent Chat Phase에서:

- 전체 답변
- 긴 Research
- 링크
- 후보 비교
- screenshot context
- tool progress
- approval
- follow-up
- error/retry

를 다룬다.

#### VR Overlay

VR에서는 glanceable output을 우선한다. 현재 경험을 벗어나지 않고 필요한 도움을 받는 출력 경계다.

- 요청을 받았는지, 실행 중인지, 완료·실패했는지 구분
- 현재 질문에 대한 짧은 답과 필요한 다음 행동
- 판단에 중요한 불확실성·승인 필요 상태
- 음성으로 이어갈 수 있는 후속 요청

복잡한 Research, 긴 출처와 비교는 Agent Chat Phase에 남긴다. Overlay는 일반 Agent 결과를 표시하고, 첫 킬러 기능의 추천·시작 안내 표현은 4-6에서 구체화한다.

---

### 3-6. 외부 시스템 통합

#### Hermes·OpenClaw — 주력 연결의 기술 검증

2-5가 제품 연결 계약의 기준이다. 각 Runtime의 공식 통합 수단과 지원 버전을 확인하고, 다음을 실제 연결로 검증한 후 adapter 방식을 정한다.

- PuriPuly의 사용자 요청을 외부 Agent에 전달하고 후속 세션을 유지하는 경로
- 허용된 Context와 Puri / 외부 Tool을 조회하는 경로
- 원래 요청에 대응하는 최종 결과를 PuriPuly HUD로 반환하는 경로
- 인증·접근 범위·연결 해제, 실패와 취소의 실제 의미
- 이미지 입력과 progress streaming 등 기능별 지원 여부

한 Runtime의 성공을 다른 Runtime의 지원 증거로 쓰지 않는다. 구체적인 endpoint·Skill 형식·Plugin API는 이번 문서 수정에서 확인하지 않았으므로 확정 인터페이스로 기재하지 않는다. 필요한 기능이 없으면 차단 항목과 가능한 연결 방식을 기록하고, Codex 경로의 성공으로 주력 연결 완료를 대체하지 않는다.

#### Codex App Server — 내장 Runtime

2026년 8월 현재 OpenAI 공식 문서는 Codex App Server를 다음 용도로 명시한다.

> 제품 내부에 Codex를 깊게 통합하여 authentication, conversation history, approvals, streamed agent events 등을 직접 다루는 인터페이스.

또한 2026-08-19 OpenAI Developer Blog는 "제품 자체에 Agent가 포함되는 경우" App Server를 사용하라고 설명하고, Application이 자체 context와 tools를 소유하며 Codex가 Agent Loop를 담당하는 패턴을 제시한다.

내장 경로의 integration target은 Codex App Server로 유지한다.

내장 MVP에서 검토할 App Server 기능:

```text
account/read
account/login/start

thread/start / resume
turn/start
turn interrupt / cancel

streamed item / turn events
approval requests

text input
localImage input

rate-limit / account state
```

공통 제품 Tool 경계를 Codex가 호출할 MCP 또는 동등한 local tool surface에 연결한다. 이 경계의 의미를 Codex 전용으로 정의하지 않는다.

Experimental API에만 의존하는 기능은 MVP의 핵심 경로에서 가능한 한 피한다.

#### Codex 내장 경로의 인증

공식 App Server 문서상 다음 두 방식이 지원된다.

- ChatGPT managed authentication
- API key authentication

ChatGPT managed mode에서는 Codex가 browser/device-code OAuth flow와 token refresh를 소유한다.

내장 Codex에서 ChatGPT subscription을 활용한다면 원칙은 다음과 같다. 이 인증을 Hermes·OpenClaw 경로의 전제 조건으로 삼지 않는다.

- **각 사용자가 자신의 ChatGPT 계정으로 직접 로그인**
- 사용자의 Codex entitlement와 rate limit을 그대로 사용
- 개발자 계정을 사용자들과 공유하지 않음
- PuriPuly 서버가 사용자 ChatGPT credential을 대신 수집하거나 pooling하지 않음
- 가능하면 Codex가 관리하는 공식 auth flow 그대로 사용
- 로컬 credential storage는 OS credential store/keyring 우선

#### 2026년 8월 기준 약관 적합성 검토

현재 공식 문서만 놓고 보면 **기술적·제품적 의도는 PuriPuly MVP와 높은 정합성이 있다.**

근거:

1. App Server 공식 문서가 명시적으로 "Embed Codex into your product"라고 설명한다.
2. 2026-08-19 공식 Developer Blog는 기존 제품·workflow 내부에 Codex를 넣는 것을 권장한다.
3. App Server는 per-user ChatGPT managed login을 공식 제공한다.
4. 제품이 own context와 MCP tools를 제공하고 Codex가 Agent loop를 담당하는 공식 예시가 존재한다.
5. App Server는 local image input을 지원하므로 screenshot을 비공식 우회 없이 제공할 수 있다.

그러나 **2026-01-01 발효 개인용 OpenAI Terms of Use에는 별도로 주의해야 할 문구가 있다.**

특히:

- 계정 자격 증명을 타인과 공유하거나 계정을 타인에게 제공하면 안 됨
- rate limit / protective measure를 우회하면 안 됨
- 서비스의 data 또는 Output을 자동/프로그래밍 방식으로 extract하면 안 된다는 일반 제한이 있음

App Server 자체가 공식적으로 프로그램 내 통합과 streamed output을 위해 제공되는 인터페이스라는 점에서, 정상적인 App Server protocol consumption까지 금지하려는 취지라고 단정하기는 어렵다. 반대로 일반 Terms 문구만으로 PuriPuly의 **비코딩 목적 + ChatGPT subscription 기반 third-party product embedding**이 무조건 허용된다고 법적 결론을 내리는 것도 안전하지 않다.

따라서 현재 상태를 다음처럼 취급한다.

```text
Technical support by official docs:
HIGH

Product-integration intent:
HIGH

Per-user authentication fit:
HIGH

Shared-account / pooled-access design:
NOT ALLOWED — 사용하지 않음

Rate-limit circumvention:
NOT ALLOWED — 사용하지 않음

Consumer Terms와 App Server embedding의 정확한 계약상 경계:
RELEASE GATE — 공개 배포 전 재확인
```

MVP 개발과 내부/개인 검증은 공식 App Server 사용 방식에 맞춰 진행하되, **공개 배포 또는 사용자를 대상으로 한 정식 Codex 기능 출시 전에는 당시 최신 Terms / Service Terms / Codex 문서를 다시 확인하고, 가능하면 OpenAI Support 또는 Codex 팀의 서면 확인을 받는 것을 Release Gate로 둔다.**

약관 리스크를 줄이는 구현 원칙:

- 공식 App Server protocol만 사용
- 각 사용자 본인의 login/entitlement 사용
- credentials 공유/중계/재판매 금지
- rate limit 상태를 존중
- 보호장치/approval 우회 금지
- App Server가 제공하지 않는 비공식 인증 경로 사용 금지
- 공개 출시 전 약관 재검토

API Key 기반 사용은 API/Business Terms 계열로 계약 구조가 더 명확할 가능성이 있지만 UX와 비용 모델이 달라지므로 별도 옵션으로 본다.

> 이 항목은 법률 자문이 아니라 2026-08-29 현재 공식 문서와 약관에 기반한 제품 설계용 리스크 정리다.

공식 참고 자료:

- Codex App Server: [https://developers.openai.com/codex/app-server](https://developers.openai.com/codex/app-server)
- Codex Authentication: [https://developers.openai.com/codex/auth](https://developers.openai.com/codex/auth)
- Codex as a platform (2026-08-19): [https://developers.openai.com/blog/codex-as-a-platform](https://developers.openai.com/blog/codex-as-a-platform)
- OpenAI Terms of Use (effective 2026-01-01): [https://openai.com/policies/terms-of-use/](https://openai.com/policies/terms-of-use/)

#### Direct VRChat API

PuriPuly Standalone mode에서 기존 로컬 상태만으로 부족한 Context를 필요한 범위에서 보완한다. 통합 역할은 2-5, 첫 킬러 기능의 데이터 요구는 4-5를 따른다.

Agent에 필요하지 않은 장기 Social Data까지 수집하지 않는다.

#### VRCX-0

선택적 enrichment이며 적용 범위는 2-5를 따른다. 장기 Timeline·Social History·친구 관계 기능은 외부 시스템의 영역으로 남기며, 연동할 수 있다는 이유만으로 MVP 수집 범위에 추가하지 않는다.

통합은 가능한 한 API/MCP 경계를 사용하고 내부 DB 포맷에 직접 결합하지 않는다.

참고:

- [https://github.com/Map1en/VRCX-0](https://github.com/Map1en/VRCX-0)

#### Realtime Audio

MVP 이후 별도 실험 축.

핵심 질문은 "기술적으로 붙일 수 있는가?"가 아니라 다음이다.

- Transcript Trigger보다 실제 UX가 좋아지는가?
- 비용/latency 복잡성을 감수할 가치가 있는가?
- Conversation context는 어차피 Puri ASR log를 유지해야 하는가?
- Audio-native 정보가 대표 기능을 실질적으로 개선하는가?

---

### 3-7. 구현 전략

공통 Agent 기반의 구현과 개별 기능의 사용자 가치 검증을 구분한다. 아래는 초기 전략이며 모든 Phase를 순서대로 완성해야 사용자에게 배울 수 있다는 뜻은 아니다. 첫 킬러 기능의 사용자 문제·데이터 probe는 4-8에 따라 초기에 함께 진행한다.

#### Phase A — Agent MVP Vertical Slice

목표:

> **"Puri Agent Chat에서 텍스트 또는 음성으로 요청하면, 현재 Conversation + 허용된 Screen + 필요한 Context를 선택한 Agent가 보고 Puri Tool을 사용한 뒤 결과를 Agent Chat Phase와 VR Overlay에 반환한다."**

Hermes·OpenClaw 각각의 실제 호출·Context/Tool 접근·결과 반환 가능성을 먼저 확인한다. 두 연결을 내장 Codex 완성 뒤로 미루지 않으며, Codex는 독립 내장 경로로 함께 완결한다.

최소 구성:

1. Hermes·OpenClaw 연결 설정·인증·요청·후속 세션·해제
2. Codex App Server local process integration과 사용자별 공식 인증
3. 별도 Agent Chat Phase와 짧은 VR 상태·결과 출력
4. Text command + Self ASR Final Transcript 기반 사용자 정의 Trigger
5. 필요한 요청에서 허용된 command-time screenshot
6. 최근 Self / Peer 원문 transcript와 최소 VRChat / Application Context
7. 필요한 Puri Query / Context tools와 승인 경계가 명확한 local command
8. Agent progress / result / failure / cancellation 처리
9. 기본 approval / authority / privacy와 bounded session context

경로별 근거를 남긴다. Hermes와 OpenClaw는 각각 Codex 실행·로그인 없이 요청과 후속 결과가 실제 HUD로 돌아와야 하며, Codex도 외부 Agent 없이 동작해야 한다. 화면 기반 요청은 각 경로의 실제 이미지 지원 범위를 확인한다. 연결 실패·해제·취소 후 응답 처리와 허용되지 않은 Context 접근을 확인하고, 한 경로의 통과로 다른 경로를 완료 처리하지 않는다.

이 단계는 공통 end-to-end 통합의 검증이다. 스트리밍이나 Tool 호출 성공만으로 사용자 가치나 첫 킬러 기능이 입증됐다고 보지 않는다.

#### Phase B — Context Foundation 강화

기본 보관·전송·삭제 정책은 Phase A부터 적용하고, 여기서는 긴 세션과 Retrieval의 품질·운영을 강화한다.

- Peer transcript의 정식 Session Logging
- recent/search/time-range retrieval
- Session Event 구조
- Agent Audit
- screenshot checkpoint lifecycle
- Context privacy settings
- long-session context budget 관리

#### Phase C — Application Control Plane 확대

- UI와 독립적인 Query/Command ports
- Headless/local service
- Puri MCP tool surface 확대
- 주요 PuriPuly 기능의 typed control
- read/write permission classification
- recover/restart/status diagnostics

#### Phase D — VRChat Context 강화

- Direct VRChat Provider의 필요한 기능 확대
- 필요한 World / Event / Group 정보와 허용된 Friend Context 조회
- 필요한 최소 cache
- VRCX-0 optional enrichment
- source/freshness 충돌 처리

#### Phase E — 대표 기능 검증과 피벗

2-1 / 2-2의 Context-aware 경험을 실제 VR 세션으로 검증한다. 첫 킬러 기능의 구체적인 성공 기준은 4-7을 따른다. 사용자 검증과 피벗은 이 Phase까지 기다리지 않고 초기부터 반복한다.

평가할 항목:

- Context recall
- Context precision
- 잘못된 "이거/그거" 해석
- 불필요한 tool calls
- command latency
- screenshot이 실제로 필요했던 비율
- false trigger
- peer instruction 오인
- 사용자가 다시 설명해야 한 정보량
- 결과의 실제 usefulness
- Agent 기능의 반복 사용률

결과에 따라 공통 경험의 부족한 부분을 개선하고, 첫 킬러 기능 가설을 강화·수정하거나 교체한다.

#### Phase F — Realtime Audio / Beyond VRChat

제품 가설이 검증된 이후 검토한다.

- Realtime Audio Adapter
- 더 자연스러운 conversational turn-taking
- VRChat 외 Context Provider
- Discord / game / desktop session 적용
- Persistent automation
- Codex·Hermes·OpenClaw 이외의 추가 Agent Runtime
- 실제 수요가 생긴 경우의 범용 공개 Context API·SDK·Plugin 플랫폼

이 단계에서야 PuriPuly가 "VRChat 번역기 + Agent"로 남는 것이 최적인지, "대화와 화면을 이해하는 더 범용적인 Agent"로 확장할 것인지 본격적으로 판단한다.

---

## 4. 첫 킬러 기능 가설 — 오늘 뭐 하지?

이 장은 1~3장의 Context-aware Agent 기반 위에서 **먼저 검증할 구체적인 사용자 가치**를 다룬다. Discovery를 제품의 유일한 중심 기능이나 확정된 최종 해법으로 고정하지 않는다. 여기서 익힌 것이 다른 기능으로 이어질 수 있으며, 사용자 문제와 결과에 따라 첫 기능 자체를 수정할 수 있다.

### 4-1. 목적 — 익숙한 범위를 넘어 새로운 자극으로

> **"오늘 뭐 하지?"를 해결하고, 사람들이 세이프 스페이스나 익숙한 곳·활동을 넘어 원하는 새로운 자극을 발견하고 시작하도록 돕는다.**

Discovery의 대상은 **월드·이벤트·그룹**을 포함한다. 월드는 장소와 활동, 이벤트는 시간과 참여할 일, 그룹은 커뮤니티와 지속적인 활동을 발견하는 경로다. 전달하는 가치는 검색 결과 목록이 아니라 **지금 무엇을 해볼지와 어떻게 시작할지**다.

#### 새로움은 안전한 관계를 떠나라는 요구가 아니다

익숙한 범위를 넓히는 것은 사용자의 선택이다. 편안한 관계나 장소를 열등한 상태로 취급하거나, 낯선 사람과의 교류를 강요하지 않는다.

검증할 가능성 중 하나는 다음과 같다.

> **사람들이 원하는 건 새로운 장소 자체가 아니라, 익숙한 친구와도 새로운 이야기를 하고 다른 모습을 볼 수 있는 계기일 수 있다.**

이 경우 **익숙한 사람들과 새로운 일을 하는 것**도 목적을 달성한다. 같은 장소에서도 활동과 대화가 달라질 수 있으며, 월드 이동 자체가 성공 조건은 아니다. 반대로 새로운 사람·커뮤니티와의 접점이 핵심이라면 월드 추천만으로는 부족하고 이벤트·그룹 발견과 참여 안내가 중요해질 수 있다.

어느 쪽이 핵심 욕구인지 아직 확인하지 않았다. 이를 근거로 별도의 대화 소재 생성 기능, 관계 분석이나 사람 매칭을 확정하지 않는다. 실제 경험에서 무엇이 새로웠고 어떤 계기가 필요했는지 확인해 첫 기능을 좁힌다.

### 4-2. 사용자 상황과 경험 가설

#### 해결할 두 상황


| 사용자 상황               | 요청 예시                               | 기대하는 결과                                |
| -------------------- | ----------------------------------- | -------------------------------------- |
| 초심자: 어디부터 시작해야 할지 모름 | "처음인데 신기한 걸 해보고 싶어. 사람 많은 건 부담스러워." | 부담에 맞는 월드·이벤트·그룹과 할 거리, 접근·참여 조건, 첫 행동 |
| 기존 사용자: 익숙한 선택지만 반복  | "우리 평소에 안 하는 거 하나 골라봐."             | 현재 일행의 조건을 지키면서 다른 활동을 제안하고, 왜 새로운지 설명 |


두 상황은 별도 제품이나 필수 프로필 설정으로 나누지 않는다. 현재 요청과 대화에서 원하는 도움의 정도를 파악하며, 새 사용자에게 방문 이력을 먼저 요구하지 않는다.

이 구분만으로 첫 핵심 사용자를 확정하지 않는다. Agent를 이미 쓴다는 사실도 이 문제를 절실하게 겪는다는 증거가 아니다. 실제로 새로운 경험을 찾았던 상황, 사용한 대안, 막힌 지점과 결과를 통해 검증 대상을 좁힌다.

#### 해결할 변화와 수단을 구분한다

- **월드 Discovery:** 장소와 그 안의 활동을 발견하고 시작한다.
- **이벤트 Discovery:** 일정·관심사·참여 조건에 맞는 행사나 공동 활동에 참여할 계기를 찾는다.
- **그룹 Discovery:** 원하는 활동이나 관심사를 공유하는 커뮤니티와 참여 경로를 찾는다. 그룹 가입 자체를 즐거운 교류의 증거로 보지 않는다.
- **검증할 경험 가설:** 익숙한 친구와 새로운 활동·대화를 시작하는 계기가 필요한가, 새로운 사람·커뮤니티와의 접점이 필요한가, 또는 다른 막힘이 있는가.

마지막 항목은 확정된 별도 기능이 아니다. 사용자가 새로운 장소를 원한다고 가정하지 않고, 어떤 경험의 변화가 유용했는지 확인한다.

### 4-3. 첫 검증 범위와 제외 사항

첫 핵심 사용자, 어떤 욕구를 먼저 해결할지, 구체적인 기능 조합과 대상별 MVP 지원 깊이는 아직 확정하지 않는다. 아래는 검증 범위 초안이지 고정된 출시 체크리스트가 아니다.

- 최근 대화·현재 화면·신뢰 가능한 상태에서 현재 요청과 일행의 조건 파악
- 월드·이벤트·그룹 후보 확보와 출처·최신성·접근 및 참여 조건 확인
- 적합성과 새로움을 설명하는 주추천 하나, 필요한 경우 대안 하나
- 대상 링크와 이동·참여·활동의 첫 행동 안내
- "가봤어", "다른 거", "더 쉬운 곳", "어떻게 시작해?" 같은 후속 요청
- Agent Chat Phase / VR Overlay에서 짧은 제안·진행 상태·시작 도움 확인
- 선택과 실제 방문·참여·활동 결과를 구분하는 최소 검증 경로

월드만으로 범위를 미리 고정하거나 이벤트·그룹을 일괄 후순위로 밀지 않는다. 반대로 세 대상의 모든 기능을 첫 MVP에서 동일한 깊이로 구현한다는 뜻도 아니다. 데이터 접근과 사용자 가치에 근거해 지원 범위·검증 순서를 정하고 미지원 항목을 명시한다.

#### 기존 기반에서 가져오는 경계

- 내장 Codex와 Hermes·OpenClaw 각각의 독립 요청·후속 대화·결과 반환 경로
- 원문 중심의 bounded session context와 출처·시점·명령/관찰 구분
- 사용자 제어 아래의 전송·보관·삭제와 화면 캡처
- VRCX-0 없이 동작하는 Standalone 경로와 허용된 경우의 선택적 보강

현재 인스턴스 인원이 실제 일행 인원과 같다고 가정하지 않는다. 신뢰 가능한 상태와 대화에서 확인된 기기·시간·기피 요소를 사용하고, 참가자 이름이나 개인별 기기를 자동으로 알 수 있다고 전제하지 않는다. 장기 방문 이력이 없어도 추천은 동작해야 한다.

#### 이 기능의 초기 검증에서 하지 않는 것

- 자동 Join·초대·포털 생성·그룹 가입·이벤트 참가 신청 등 외부 상태 변경
- 개인·인스턴스 자체를 대상으로 한 매칭이나 자동 관계 분석
- 장기 개인 취향·Group Taste 학습과 Persistent Social Memory의 기본 수집
- 광범위한 PuriPuly 설정 조작이나 범용 플랫폼 전체를 이 기능의 선행 조건으로 만드는 것

사용자가 직접 선택하고 이동·참여한다. 같은 장소에서 새 활동을 시작하는 경우도 포함한다. **Agent 전반의 PuriPuly Control 가능성은 2-2와 3장에서 유지**하되, Discovery의 효과를 검증하기 위해 그 전체를 구현할 필요는 없다.

### 4-4. 대표 흐름과 추천·시작 도움

#### 대표 사용자 흐름

1. 대화에서 "Quest 가능", "다섯 명", "공포 제외", "한 시간", "같이 할 활동" 같은 조건이 나온다.
2. 사용자가 "퓨리야, 우리 평소에 안 하는 거 하나 골라봐"처럼 Text 또는 Self Trigger로 추천을 요청한다. 주변 대화만으로 먼저 끼어들거나 추천을 실행하지 않는다.
3. Agent가 현재 조건과 출처를 묶고, 선택을 바꿀 중요한 누락·충돌만 짧게 확인한다.
4. 실제로 확인 가능한 월드·이벤트·그룹과 활동 후보를 찾고, 대상별 제약을 점검한 뒤 적합성과 새로움을 비교한다.
5. **주추천 하나와 필요할 때 대안 하나**를 제시한다. 사용자가 원하지 않은 긴 결과 목록은 기본 출력으로 삼지 않는다.
6. 사용자는 추천 이유·주의점·첫 행동을 보고 선택하거나 "다른 거", "가봤어"로 조건을 보완한다.
7. 링크나 식별 가능한 진입·참여 정보를 따라 사용자가 직접 이동하거나 참여한다. 같은 장소의 새 활동이라면 이동 없이 시작할 수 있다.
8. "이제 뭐 해?"라고 물으면 해당 대상의 확인 가능한 안내와 필요한 경우 허용된 현재 화면을 바탕으로 시작을 돕는다.

예시 출력의 형태는 다음과 같다. 아래는 월드 활동을 제안하는 한 사례이지 Discovery의 전체 범위가 아니다. 실제 추천에서는 확인한 대상·정보·출처를 사용하며, 이 예시의 활동과 시간은 특정 월드의 사실을 뜻하지 않는다.

> "이번엔 협동 탐험 쪽을 추천해. 방금 말한 기기·인원 조건에 맞고, 오늘 계속하던 수다와 달리 같이 목표를 해결할 수 있어. 먼저 월드 안내에서 시작 위치와 참여 방법을 확인하자."

#### 추천이 설명해야 하는 것

- **무엇을 해볼 수 있는가:** 월드·이벤트·그룹의 이름·링크와 실제 활동 또는 참여 기회
- **왜 지금 우리에게 맞는가:** 현재 요청과 일행의 조건 중 선택에 영향을 준 근거
- **무엇이 새로운가:** 현재 활동·세션 내 방문·사용자 진술과 비교한 차이
- **실행 가능한가:** 플랫폼·인원·접근·가입 또는 참가 조건·일정과 확인되지 않은 항목
- **어떻게 시작하는가:** 이동·참여 방법과 첫 행동, 필요한 준비나 주의점

가이드는 검색 설명문의 재진술이 아니라 다음 행동을 알 수 있게 해야 한다. 다만 확인되지 않은 시작 버튼·게임 규칙·소요 시간을 만들어내지 않는다. 퍼즐 정답이나 공략 전체를 먼저 공개하지 않고 시작 안내를 기본으로 한다.

#### 적합성·새로움·다양성의 경계

추천은 **취향 적합도·현재 일행 적합도·실행 가능성·새로움**을 함께 고려한다. 이 네 축은 제품 판단 기준이지 확정된 점수식이 아니다.

- 명시적인 금지·필수 조건은 새로움보다 우선한다. 후보가 없다고 몰래 완화하지 않는다.
- 시간은 확정 이용 조건과 추정 소요 시간을 구분한다. "약 40분" 같은 값에는 근거와 불확실성을 남긴다.
- 새로움은 단순 신작·비인기 대상이나 이동 여부가 아니라 **이 사용자와 일행에게 지금 다른 경험인가**로 판단한다. 익숙한 친구와 새로운 활동·대화를 시작하는 것도 포함할 수 있다.
- 기본 근거는 현재 세션의 활동·방문·추천/거절과 사용자의 "가봤어", "평소에는 수다만 해" 같은 진술이다.
- 장기 이력이 없으면 "처음 가는 곳"이나 "모두 안 가본 곳"이라고 단정하지 않는다. 활동의 차이를 설명하거나 방문 여부를 모른다고 밝힌다.
- 대안은 같은 후보의 이름만 바꾼 반복이 아니라 다른 선택 이유를 제공한다.
- "가봤어"나 "그건 싫어"는 세션 내 후속 추천에 반영한다. "가봤어"를 그 월드를 싫어한다는 뜻으로 해석하지 않는다.

### 4-5. 후보 데이터와 추천 방식

첫 구현은 **작더라도 출처가 확인되는 월드·이벤트·그룹 후보와 검색 경로**로 시작할 수 있다. 전체 대상을 포괄하는 자체 추천 DB나 학습 모델은 선행 조건이 아니다. 대상별 데이터 확보 가능성과 사용자 가치를 확인해 지원 깊이와 검증 순서를 정하며, 정보가 부족한 유형을 지원 완료로 표시하지 않는다.

후보에는 다음 정보가 필요하다. 확보할 수 없는 값은 비워 두며, 누락을 유리한 값으로 간주하지 않는다.


| 정보                 | 추천·가이드에서의 역할                                    |
| ------------------ | ----------------------------------------------- |
| 대상 유형, ID / 링크, 이름 | 월드·이벤트·그룹을 구분하고 실제 대상을 식별                       |
| 플랫폼, 인원·접근·참여 조건   | 지금 일행이 이용하거나 참여할 수 있는지 판단; 대상에 적용되는 조건만 사용      |
| 활동·분위기·기피 요소       | 하고 싶은 일과 피하고 싶은 경험 비교                           |
| 일정·시간대·소요 시간 정보    | 이벤트 시작·종료와 현재 참여 가능성, 제한 시간 내 가능성 판단; 추정과 확정 구분 |
| 시작 방법·참여 경로·안내 출처  | 이동·참여 후 첫 행동을 설명; 그룹 가입과 실제 활동 참여를 구분           |
| 정보 출처·확인 시점        | 오래되거나 상충하는 설명과 확인된 사실 구분                        |


이벤트의 취소·변경 여부와 그룹의 가입 조건·활동 일정은 해당 정보원에서 확인한 범위만 제시한다. 월드의 존재가 이벤트 개최나 그룹 활동을 증명하지 않으며, 월드 API 하나가 모든 유형의 정보를 제공한다고 가정하지 않는다.

추천 처리의 개념적 순서는 다음과 같다.

```text
현재 요청 + 세션 조건
→ 월드·이벤트·그룹 / 활동 후보 확보
→ 필수 조건 확인 / 불명확한 후보 구분
→ 활동·분위기 적합성 비교
→ 근거 있는 새로움·대안 다양성 고려
→ 주추천 + 필요한 대안 + 시작 가이드
→ 선택 / 거절 / 확인된 방문·참여·활동 결과
```

Semantic search, embedding retrieval, LLM reranking, 별도 novelty 보정은 구현 선택지다. 후보 설명과 단순 검색만으로 충분한지 먼저 보고, 실제 실패를 개선할 필요가 있을 때 추가한다. 특정 추천 라이브러리·벡터 DB·점수식을 제품 요구사항으로 고정하지 않는다.

핵심 불확실성은 **각 대상이 어떤 경험을 제공하고 어떻게 참여할 수 있는지 설명할 데이터를 확보할 수 있는가**다. 대상별 데이터 접근 가능성·사용 조건·최신성·누락 비율을 초기 검증에서 확인한다.

#### 정보가 부족하거나 조건에 맞는 후보가 없을 때

- 중요한 조건이 충돌하면 이를 드러내고 사용자에게 어느 조건을 바꿀지 묻는다.
- 필수 호환성이나 접근 가능성을 확인하지 못한 후보를 "지금 갈 수 있다"고 확정하지 않는다.
- 적합한 후보가 없으면 없다고 말하고, 바꿀 수 있는 조건을 제안하되 실제 완화는 사용자가 선택한다.
- 화면 캡처·외부 연동·방문 이력이 없어도 가능한 정보로 추천한다. 그 부재로 확인할 수 없는 사실은 명시한다.
- 가이드 근거가 부족하면 정확한 시작 방법을 모른다고 밝히고, 해당 월드·행사·그룹의 공식 안내나 사용자가 제공할 수 있는 화면으로 연결한다.

### 4-6. 공통 Agent 기반에 연결할 기능별 상태·Tool·출력

#### 기능별 Context와 결과 기록

공통 Session / Conversation Event Layer 위에 아래 기록을 구분해서 둔다. 영구 대화 저장소나 중앙 수집 서버를 만들기 위한 목록이 아니다.

```text
Discovery
- recommendation and candidate type / identity (world, event, group, activity)
- constraint / novelty evidence references
- selection / rejection and explicit reason
- observed or user-confirmed visit / participation / activity outcome, or unknown
```

추천 당시 조건·새로움의 근거, 세션 내 추천·거절·선택과 확인된 방문·참여·활동을 후속 요청에 활용한다. 월드·이벤트·그룹의 ID와 관찰 종류를 구분하고, 관찰하지 못한 결과는 `unknown`으로 남긴다. 보관·삭제와 검증 동의 범위는 3-1과 4-7을 따른다.

#### Discovery Tool Surface

아래는 구현 후보이지 확정 공개 API가 아니다. 공통 Conversation·Screen·Application Tool에 기능별 조회를 더한다.

```text
Session
- get_session_recommendations

Discovery
- search_candidates (world / event / group)
- get_candidate_info
- get_participation_guidance
```

대상별 검색 방식과 데이터는 정보원에 맞춰 구분한다. 공통 이름이 동일한 외부 API의 존재를 뜻하지 않는다. 상세 조회가 활동·참여 안내까지 제공하지 않으면 출처를 확인할 수 있는 외부 검색으로 보완한다. 없는 데이터를 채워 넣거나 단일 월드 API로 모든 유형이 지원된다고 가정하지 않는다.

#### Agent Chat Phase

공통 Agent Chat Phase에 이 기능의 추천·활동 시작·후속 도움을 다음처럼 표시한다.

- 주추천과 필요한 대안
- 적합성·새로움 근거, 적용 조건과 확인되지 않은 항목
- 월드·이벤트·그룹 링크와 이동·참여·활동의 첫 행동
- 정보 출처와 필요한 screenshot context
- "가봤어", "다른 거", "어떻게 시작해?" 후속 요청
- 진행 상태, 취소, 오류와 재시도

긴 출처·비교는 여기에서 확인할 수 있지만, 기본 경험을 일반 Research 보고서로 만들지 않는다.

#### VR Overlay

VR 안에서도 **무엇을 추천하는지, 왜인지, 다음에 무엇을 할지**를 이해할 수 있어야 한다. 짧게 보되 선택에 중요한 불명확한 조건을 감추지 않는다.

```text
추천 탐색 중: 지금 일행의 조건에 맞는 다른 활동을 찾는 중
추천 결과: 대상명·유형 + 활동 / 참여 기회 + 우리에게 맞는 이유 + 새로운 점
시작 안내: 이동·참여 경로 또는 현 위치의 첫 행동 + 중요한 주의점
후속 요청: "다른 거" / "가봤어" / "어떻게 시작해?"
```

대상명과 행동은 실제 확인한 정보로 채운다. 근거의 상세 내용은 Agent Chat Phase에 남기고, 핵심 추천·도움 확인을 위해 VR을 벗어나 긴 목록을 읽게 하지 않는다. HUD는 스마트 글래스의 소프트웨어화에서 현재 경험을 벗어나지 않고 도움을 받는 출력 경계다.

### 4-7. 사용자 가치와 반복 사용 검증

첫 기능 검증은 **추천 노출 → 선택 → 이동·참여 또는 현 위치에서의 시도 → 활동·교류 시작 → 다음 요청**을 구분해서 본다. 링크 클릭은 Join·참가·가입이 아니며, 같은 월드에 들어갔거나 그룹에 가입했다는 사실만으로 새로운 경험이 생겼다고 단정하지 않는다. 월드 이동은 가능한 경로 중 하나이지 모든 성공의 필수 조건이 아니다.


| 질문                 | 확인할 근거                                                                                                         |
| ------------------ | -------------------------------------------------------------------------------------------------------------- |
| 초심자가 시작할 수 있는가     | 이동·참여 경로와 첫 행동을 이해하고 실제로 활동을 시작했는지 관찰·확인                                                                       |
| 새로운 경험을 만들었는가      | 이전 활동·사용자 진술 대비 무엇이 달라졌는지 확인; 새 장소, 익숙한 친구와의 새 활동·대화, 새 커뮤니티 접점을 구분                                            |
| Context가 결과를 개선하는가 | 같은 후보 정보에서 명시적 요청만 쓴 경우와 대화·세션 조건을 함께 쓴 경우의 조건 위반·적합성·재설명 필요 비교; 시각적 요청은 화면과 대화의 결합이 지칭 해석과 다음 행동을 개선하는지 별도 확인 |
| 의사결정을 좁혔는가         | 선택 여부, 거절 이유, 추가로 물어야 했던 정보 기록                                                                                 |
| 다시 찾는가             | 이후 추천이 필요한 순간에 사용자가 자발적으로 다시 요청했는지 확인                                                                          |
| 기존 경험을 지켰는가        | 번역 방해, 원치 않는 개입, 잘못된 Trigger, 과도한 Context 전송 여부                                                                |


Join·참가·가입·체류·즉시 이탈은 유용한 행동 신호지만 만족도나 교류의 정답은 아니다. AFK·친구 사정·기술 문제·일정 종료가 원인일 수 있으므로 가능한 경우 사용자의 이유와 함께 해석한다. 대화의 새로움이나 관계의 변화를 전사만으로 자동 판정하지 않는다.

기본 피드백은 현재 세션의 선택·거절과 관찰 가능한 방문·참여·활동 결과다. 자동 관찰이 불가능하면 허가된 사용자 검증에서 직접 관찰하거나 짧게 확인하고, 관찰하지 못한 결과는 `unknown`으로 남긴다.

다음 세션의 재요청을 확인하려면 별도 동의를 받은 검증 참여자의 후속 확인 또는 최소한의 추천 사용 기록이 필요하다. 이를 이유로 Peer 원문·참가자 신원·장기 Social History를 기본 수집하지 않는다. 세션 간 저장을 도입한다면 목적·항목·기간·삭제 제어를 먼저 정한다.

후속 논의의 "20~30명에게 약 2주"는 사용자 검증 설계의 후보이지 확정 규모나 성공 임계값이 아니다. 초심자·기존 사용자·Agent 사용 여부 중 어떤 집단과 상황을 먼저 검증할지, 관찰 범위와 판단 기준을 평가 전에 정한다. 현재 성공률·추천 품질·재사용 성과가 입증된 상태는 아니다.

추천을 수락해도 새 활동을 시작하지 못하면 가이드를, 조건에 맞는 후보가 반복해서 없으면 데이터를, 만족했지만 다시 찾지 않으면 사용 빈도와 제품 가설을 재검토한다. 새로운 자극을 막는 이유가 장소 부족이 아니라 합의·참여 부담·대화의 계기라면 Discovery의 방식이나 첫 기능 자체를 재검토한다. Runtime 통합 성공만으로 사용자 가치를 입증했다고 보지 않는다.

### 4-8. 검증 순서와 남은 결정

공통 기반의 구현 전략은 3-7을 따른다. 이 절은 첫 킬러 기능의 검증 경로이며, 전체 Context-aware Agent의 유일한 로드맵이 아니다. 사용자 문제와 데이터에 관한 probe는 공통 기반을 전부 만든 뒤로 미루지 않는다.

#### 기능 검증 A — 사용자 문제·Discovery 데이터와 주력 연결 가능성 검증

사용자가 실제로 "오늘 뭐 하지?"에서 막혔던 상황과 새로운 경험을 즐겼던 상황을 확인한다. 월드·이벤트·그룹을 대상으로 보되, 필요한 것이 새 장소인지, 익숙한 친구와의 새 활동·대화 계기인지, 새로운 커뮤니티 접점인지 먼저 구분한다.

- 실제 행동·기존 대안·막힌 지점에 근거해 첫 핵심 사용자와 검증할 경험을 정함
- 월드·이벤트·그룹 각각의 정보원 접근·이용 조건·참여 조건·활동·시작 안내·최신성 확인
- 대상별 데이터와 사용자 가치에 근거해 첫 구현의 지원 깊이·검증 순서를 정하고, 미지원 범위를 명시
- 대화·화면·상태가 재설명을 줄이고 짧은 요청의 의미를 복원하는지 확인
- 대화에서 주어진 조건을 지키면서 후보를 좁히고, 방문 이력 없이도 정직하게 새로움을 설명할 수 있는지 확인
- 실제 대상에서 이동·참여·첫 활동 안내가 맞는지 확인; 현 위치에서 새 활동을 시작하는 경우도 구분
- Hermes와 OpenClaw 각각의 실제 요청→Context/Tool 접근→결과 반환 경로 확인
- 각 연결의 지원 버전·설정·인증·이미지·취소·권한 해제 제약 기록

이 단계는 제품 가설·데이터와 연결 가능성의 초기 probe이지 통합 MVP가 아니다. 후보와 가이드 근거가 부족하면 데이터 접근을, 주력 연결이 막히면 해당 통합 경로를 수정한다. 핵심 욕구가 다르면 첫 기능 가설을 다시 정한다. 미확인 상태를 성공으로 처리하거나 Codex 통합만으로 주력 연결을 대체하지 않는다.

#### 기능 검증 B — Discovery end-to-end 경험 검증

목표:

> **"사용자가 상황을 다시 설명하지 않고 '오늘 뭐 하지?'라고 요청하면, 대화·필요한 화면·세션 상태를 이해한 Agent의 제안과 시작 도움을 VR에서 확인하고, 원하는 새로운 활동이나 교류를 시도한 뒤 후속 도움을 받을 수 있다."**

공통 Conversation·Screen·세션 Context·Discovery 정보·HUD를 기반으로 Hermes·OpenClaw의 end-to-end 연결을 우선 검증하고, 내장 Codex도 독립 사용 경로로 완결한다. 아래는 기능 검증 A에서 정할 대상별 지원 범위에 적용할 통합 기준이며, 월드·이벤트·그룹의 모든 기능을 동일하게 구현하라는 뜻은 아니다.

1. Hermes·OpenClaw 각각의 연결 계약과 내장 Codex의 사용자 인증·lifecycle
2. Text Input과 Self Final Transcript 기반 사용자 정의 Trigger
3. 최근 Self / Peer 원문과 신뢰 가능한 세션 상태의 bounded context
4. 검증 범위로 정한 월드·이벤트·그룹 후보 검색·상세·참여 안내와 출처; 대상별 지원·미지원 범위 명시
5. 조건 준수, 근거 있는 새로움, 세션 내 거절/선택 반영
6. Agent Chat Phase와 VR Overlay의 추천·첫 행동·후속 안내
7. 필요한 요청에서 허용된 command-time screenshot
8. 명령/관찰 경계, 최소 Tool 권한, 전송·보관·삭제 제어
9. 선택·방문·참여·실제 활동 결과를 구분할 최소 기록 또는 동의받은 관찰 경로

Peer Context 보관 정책과 화면 Privacy는 다음 단계로 미루지 않는다. 화면을 끄거나 VRCX-0를 설치하지 않아도 일반 추천이 가능해야 한다.

통합 완료는 응답 스트리밍이 아니라 **4-4의 요청→제안→선택→이동·참여 또는 현 위치의 활동 시작→후속 도움 흐름이 실제 VR 세션에서 성립함**으로 판단한다. 화면이 필요한 요청에서는 대화와 화면의 연결을 별도로 확인한다. 아직 관찰하지 않은 사용자 만족도나 반복 사용을 구현 완료와 혼동하지 않는다.

**경로별·지원 대상별 완료 근거**를 따로 남긴다. Hermes와 OpenClaw는 각각 Codex 실행·로그인 없이 제안·시작 도움·후속 요청이 실제 HUD에 돌아와야 하고, 내장 Codex는 두 외부 Agent 없이 같은 핵심 경험이 동작해야 한다. 연결 실패·해제·취소 후 응답 처리와 허용되지 않은 Context 접근도 확인한다. 한 Runtime이나 월드 유형의 통과로 다른 Runtime·이벤트·그룹까지 완료 처리하지 않는다.

#### 기능 검증 C — 사용자 가치와 반복 사용 검증

기능 검증 A에서 정한 첫 사용자와 상황을 대상으로 4-7의 평가 기준을 적용하고, 다른 집단에도 성립하는지는 별도로 확인한다. 별도 사용자 검증은 동의·데이터 보관 범위와 필요한 배포 조건을 충족한 환경에서 진행한다.

- 요청만 사용한 답변 대비 Conversation·Screen·세션 Context의 추가 가치와 재설명 감소
- 실제 선택·이동·참여·활동 시작과 기존 패턴 대비 새로운 경험
- 익숙한 친구와의 새 활동·대화 계기와 새로운 커뮤니티 접점 중 어떤 변화가 유용했는지 확인
- 거절·빠른 이탈·가이드 실패의 이유
- 이후 추천이 필요한 순간의 자발적 재요청
- 번역 방해, false trigger, Peer 발화의 명령 오인, Privacy 위반
- Hermes·OpenClaw 각각의 연결 설정부터 첫 유용한 HUD 결과까지의 막힘과 재사용
- 외부 경로와 내장 경로별 결과·미지원 기능·실패 원인의 구분

상황 재설명 감소와 필요한 화면 지칭의 정확성은 제품 기반의 가치 지표다. Latency와 불필요한 Tool 호출은 도움을 받는 흐름이 막힌 원인을 찾는 보조 지표로 본다.

산출물에는 관찰 결과·미관찰 항목·실패 원인과 **유지 / 데이터·시작 도움 개선 / 첫 기능·사용자 가설 수정 또는 교체** 결정을 남긴다. Discovery 한 가설의 성패와 스마트 글래스의 소프트웨어화라는 상위 가설을 구분하며, 후자의 가치도 근거 없이 입증됐다고 보지 않는다. 근거 없는 수치 목표나 긍정적인 결론을 미리 정하지 않는다.

남은 제품 결정은 **첫 핵심 사용자, 가장 중요한 막힘, 대상별 지원 깊이, 데이터 운영 방식, 검증의 성공·실패 판단 기준**이다. 익숙한 친구와의 새 대화·활동 계기와 새로운 커뮤니티 접점 중 어느 쪽이 핵심인지는 실제 경험으로 좁힌다. 기능에 관한 합의는 구현·배포·추가 수집의 승인과 별개다.

---

## 문서 전체를 관통하는 핵심 가설

```text
PuriPuly는 모든 VRChat 기능을 재구현하는 프로그램이 아니다.

PuriPuly는
"지금 무슨 이야기를 하고 있는가"
그리고
"지금 무엇을 보고 있는가"
를 가장 잘 아는 Sensor가 된다.

필요한 애플리케이션 상태와 외부 정보는
직접 최소한으로 확보하거나 다른 시스템에서 가져온다.

선택한 Agent Runtime은 그 Context를 바탕으로
무엇을 알아보고 무엇을 할지 결정하는 Agent Loop를 제공한다.
```

최종적으로 검증해야 하는 제품 경험은 하나다.

> **사용자가 AI에게 상황을 설명하는 대신, 그냥 현재 상황 속에서 짧게 말해도 되는가?**

그 경험이 강하다면 기존 PuriPuly의 번역 인프라는 단순 기능 하나가 아니라 Agent로 넘어가기 위한 가장 중요한 출발점이 된다.

이 기반을 먼저 체감하게 할 킬러 기능으로 "오늘 뭐 하지?"를 탐색한다. 그 기능의 성패와 제품 기반 전체의 가설은 구분하며, 둘 다 실제 사용자 경험으로 검증한다.