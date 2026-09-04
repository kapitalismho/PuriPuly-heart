# PuriPuly Agent 기획 초안

> **상태:** Grill Me를 위한 Working Draft  
> **기준 시점:** 2026-08-29  
> **목적:** 기존 PuriPuly의 실시간 번역·전사 시스템을 기반으로, 대화와 화면을 이해하고 외부 정보 및 애플리케이션 기능을 활용하는 에이전트로 확장할 가능성을 검토한다.
>
> 이 문서는 최종 요구사항 명세가 아니다. 특히 **2-1 대표 기능**과 **2-2 확장 기능**은 현재 가장 불확실한 영역이며, Grill Me를 통해 보강하거나 필요하면 전제 자체를 피벗하는 것을 전제로 한다.

---

## 1. 목표와 맥락

### 1-1. 목표

PuriPuly를 단순한 실시간 번역 도구에서 **현재 사용자가 나누고 있는 대화와 보고 있는 화면을 이해하는 Context-aware Agent**로 확장한다.

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

특히 Agent Turn이 시작되는 시점의 화면을 timestamp와 함께 고정하면, 사용자가 "이거"라고 말했을 때 **그 말을 했던 순간 실제로 무엇을 보고 있었는지**를 재현할 수 있다.

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
- Screen: Agent 요청 시점의 화면 및 필요 시 추가 캡처

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

#### 4. MVP의 Agent Runtime은 Codex로 고정한다

초기 MVP에서는 범용 provider abstraction부터 크게 설계하지 않는다.

**Codex App Server를 첫 Agent Runtime으로 실제 통합하고 end-to-end 경험을 검증하는 것을 MVP의 중심 목표**로 둔다.

그 이유는 다음과 같다.

- App Server가 제품 내부 Agent 통합을 공식적인 사용 방식으로 제공한다.
- Conversation/thread lifecycle, streamed events, approval, tool use를 새로 만들 필요가 없다.
- ChatGPT 로그인과 API Key 로그인을 모두 지원한다.
- Text뿐 아니라 local image 입력을 지원하여 command-time screenshot과 직접 결합할 수 있다.
- PuriPuly가 Context와 Tool을 소유하고 Codex가 Agent Loop를 담당하는 경계가 명확하다.

다만 PuriPuly의 Conversation Store, Screen Sensor, Application Control Surface 자체는 Codex 내부 구현에 종속시키지 않는다. MVP는 Codex-first로 가되, 장기적으로 다른 Agent Runtime을 연결할 여지는 남긴다.

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
- Agent Runtime과의 통신
- PuriPuly 자체 기능을 조회·조작하는 Tool Surface
- Agent에 필요한 최소 VRChat Context
- 필요한 범위의 VRChat API 호출과 최소 저장
- Context Retrieval
- Agent 실행 기록 및 Audit
- VR에서 확인할 수 있는 간략한 Agent 상태/결과 출력

#### 직접 구현하되 최소 범위로 제한하는 것

VRCX-0가 없어도 Standalone으로 Agent가 동작해야 한다.

따라서 필요하면 PuriPuly가 VRChat API를 직접 사용해 다음과 같은 최소 정보는 확보한다.

- 현재 World / Instance
- 현재 참가자
- 필요한 Friend Presence
- Agent 기능에 직접 필요한 World/Event 정보
- 짧은 Session History 또는 Cache

무엇을 저장할지는 "VRCX와 비슷해질 수 있는가?"가 아니라 **Agent의 실제 Context Quality에 필요한가?**를 기준으로 결정한다.

#### 가능한 한 재발명하지 않는 것

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
- 여러 Agent Runtime의 동등한 지원

이들은 초기 구조가 불필요하게 막지 않도록 하되 MVP에서 동시에 해결하려 하지 않는다.

---

## 2. 기능

> **주의:** 2-1과 2-2는 이 문서에서 가장 불확실한 영역이다. 아래 내용은 구현 목록이 아니라 현재의 Product Hypothesis다. Grill Me 결과에 따라 기능을 강화하거나, 대표 기능 자체를 교체하거나, 더 좁거나 넓은 제품으로 피벗할 수 있다.

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
- 현재 World / Instance / Participants
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
> "아까 Quest 얘기한 사람이 누구였지?"  
> "그 프로젝트 이름 뭐였어?"  
> "아까 말한 링크 찾아줘."

실시간 음성 대화가 휘발되는 문제를 Agent가 보완한다.

#### 대표 기능이 만족해야 할 조건

- ChatGPT를 별도 창에서 여는 것보다 명확히 편해야 한다.
- 단순 검색창이나 Chatbot과 달리 **Puri가 이미 알고 있는 Context**가 결과를 바꿔야 한다.
- 30초 내 데모만 봐도 차이가 이해되어야 한다.
- VR에서 실제로 반복 사용하고 싶은 행동이어야 한다.
- 특정 World나 특정 Script에만 의존하지 않아야 한다.

이 조건을 만족하지 못한다면 2-1의 대표 기능 가설은 Grill Me를 통해 교체한다.

---

### 2-2. 확장 기능

아래는 동일한 Context + Agent primitive에서 파생될 수 있는 응용군이다. 모두 구현한다는 의미는 아니다.

#### Discovery

- 현재 대화 조건에 맞는 World 찾기
- 현재 Party에 맞는 Activity 찾기
- 현재 시간과 이후 일정을 고려한 Event 찾기
- 현재 친구 위치 또는 구성과 결합한 추천

#### Research / Fact Check

- 방금 대화에 나온 주장 확인
- 고유명사·프로젝트·제품·행사 찾기
- 대화에 등장한 링크나 공식 문서 탐색
- 여러 출처를 비교해 결론 정리

#### Recall

- 최근 대화에서 특정 정보 찾기
- 누가 어떤 조건을 말했는지 찾기
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

Screen은 핵심 Sensor다.

Agent Turn 시작 시 기본적으로 **현재 사용자가 보고 있는 화면을 timestamp와 함께 capture하는 Context Checkpoint**를 만든다.

초기에는 다음 전략을 우선한다.

- 사용자가 지정한 target window 또는 VRChat viewport를 capture
- 전체 데스크톱 무조건 capture는 피함
- Puri Agent UI 자체가 feedback loop로 계속 들어가지 않도록 가능하면 제외
- Agent가 필요하면 turn 중 추가 screenshot을 요청할 수 있도록 확장

Codex App Server는 `localImage` input을 지원하므로 MVP에서는 로컬 screenshot 파일을 직접 Turn Input으로 제공할 수 있다.

#### Session / Application State

- 현재 World / Instance
- 현재 참가자
- 필요한 Friend Presence
- PuriPuly translation / ASR / capture / provider 상태
- 현재 Agent Session 상태

#### External Context

- Direct VRChat API
- VRCX-0 optional integration
- Web Search
- World / Event 데이터
- 향후 기타 앱별 Context Provider

#### Agent State

- 현재 Codex thread
- 최근 Agent 결과
- 이전 후보와 사용자의 거절/선택
- 현재 진행 중 Tool Call
- 승인 대기 상태

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

#### Codex

MVP의 Agent Runtime.

PuriPuly가 다음을 소유한다.

- Conversation
- Screen
- VRChat / Application Context
- UI
- Product Rules
- Tools

Codex가 다음을 담당한다.

- Agent loop
- thread / turn
- reasoning
- tool selection
- streamed progress
- approval workflow
- multi-step execution

#### Direct VRChat API

VRCX-0 설치 여부와 무관하게 Standalone 기능을 위해 필요한 최소 Context를 직접 수집한다.

PuriPuly는 이를 거대한 Social DB로 확장하는 대신 Agent에 실제 필요한 범위만 저장한다.

#### VRCX-0

Optional richer Context Provider.

가능한 활용:

- 장기 activity/history
- 친구 관계 및 presence
- favorite
- social graph
- 과거 방문 정보

PuriPuly가 VRCX-0를 필수 dependency로 요구하지 않는다.

#### Web / External Search

대표 기능 후보인 Research/Discovery를 위해 Web Search 및 domain-specific search를 Agent Tool로 제공한다.

#### Realtime Audio

향후 low-latency conversational mode 또는 prosody가 실제 사용자 가치를 만든다고 검증되면 추가한다.

---

## 3. 기술적인 면

### 3-1. 필요 조건

#### 풍부한 Context Logging

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
- screenshot reference
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

#### Conversation Context Query

Agent가 다음과 같은 동작을 안정적으로 할 수 있어야 한다.

- 최근 N분 / N개 발화 읽기
- 특정 단어·사람·주제 검색
- 특정 timestamp 주변 대화 읽기
- Self / Peer 구분
- 원문 그대로 반환
- provenance 반환

#### Screen Sensor

- 지정 window / viewport capture
- Agent Turn 시작 시 command-time screenshot
- timestamp와 capture target 보존
- 추가 on-demand capture
- 민감한 다른 창을 실수로 포함하지 않는 capture policy
- local image path를 Agent Runtime에 안전하게 제공

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

Codex MVP와의 적합성을 고려하면 **PuriPuly를 Codex가 호출할 수 있는 MCP 또는 동등한 typed local tool surface로 노출**하는 방식이 우선 검토 대상이다.

Read tool과 Write tool은 분리하고, 중요한 Write는 Agent Runtime의 Approval과 PuriPuly 자체 policy를 모두 통과하게 한다.

#### Direct VRChat Context Provider

Standalone으로 필요한 최소 VRChat Context를 제공해야 한다.

- 현재 location
- World / Instance
- 참가자
- 필요한 Friend / Event / World 조회
- 최소 cache
- rate limit 처리
- source/freshness 표시

VRCX-0 연결 여부가 core 기능의 전제 조건이 되어서는 안 된다.

#### Agent Runtime Integration

MVP에서는 Codex App Server와 다음이 필요하다.

- 프로세스 lifecycle
- account state
- per-user authentication
- thread start / resume
- turn start
- text + local image input
- stream event handling
- approval handling
- rate/usage 상태
- interruption / cancellation
- error / reconnect handling

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

Web page, World description, screen text 등이 Agent에게 새로운 "명령"을 주는 Prompt Injection 경로가 되지 않도록 provenance가 필요하다.

#### Privacy

Conversation과 Screen을 핵심 Sensor로 삼는 만큼 기존 번역기보다 더 명확한 Privacy Boundary가 필요하다.

- 어떤 Context가 로컬에 저장되는가
- 어떤 Context가 Codex/OpenAI로 전송되는가
- Screen Capture 범위
- Peer transcript 전송
- Session retention
- User delete/disable controls
- Diagnostic telemetry와 conversation content의 분리

---

### 3-2. 현재 구조와의 괴리

현재 `dev` 기준으로 이미 활용 가능한 기반과 Agent를 위해 필요한 구조 사이에 다음 괴리가 있다.

| 영역 | 현재 상태 | Agent에 필요한 변화 |
|---|---|---|
| Self/Peer 음성 처리 | 이미 분리된 실시간 파이프라인 존재 | Agent context ingress로 재사용 |
| Conversation Record | `ConversationRecord` 구조 존재 | Peer까지 포함하고 Agent retrieval이 가능한 session store 필요 |
| Peer transcript logging | 현재 `ConversationRecordChannel = Literal["self"]` | `peer` 원문 Final Transcript를 privacy-aware하게 기록할 수 있어야 함 |
| Observability | Runtime/Diagnostic/Provider event 구조 존재 | Diagnostic과 별도의 Session Context / Agent Audit layer 필요 |
| UI Application Port | 이미 STT, translation, peer translation, overlay, capture 등 많은 동작을 제어 | 너무 UI 중심·광범위하므로 Agent용 Query/Command interface로 분리 |
| CLI / Runtime entry | 진단 및 실행 command 기반이 일부 존재 | 실행 중인 PuriPuly를 안정적으로 조회·조작할 local control plane/headless 경로 필요 |
| Screen | 기존 번역의 핵심 Context는 아님 | first-class Screen Sensor와 command-time checkpoint 필요 |
| Agent Runtime | 없음 | Codex App Server process/thread/turn/event integration 필요 |
| Agent UI | 없음 | 별도 Agent Chat Phase 필요 |
| Agent Command | 없음 | Text input + Self Final Transcript의 configurable trigger routing 필요 |
| VRChat Context | OSC/번역 중심 | Agent용 direct VRChat API provider와 최소 cache 필요 |
| External Context | 일부 provider integration 중심 | VRCX-0/Web/MCP 등을 Agent Tool로 연결할 구조 필요 |
| Authority Model | Agent-specific distinction 없음 | user instruction vs observed content를 데이터 레벨에서 구분 |
| Context Retrieval | Agent용 API 없음 | recent/search/time-range 기반 구조화 Query 필요 |

#### 로깅의 가장 직접적인 괴리

현재 코드의 `ConversationRecord`는 transcript, translation, source language 등의 필드를 이미 가지고 있어 출발점은 좋다.

하지만 현재 channel type이 `Literal["self"]`로 제한되어 있다.

Agent가 다음 질문에 답하려면 이 경계가 바뀌어야 한다.

> "아까 상대가 뭐라고 했어?"  
> "누가 Quest 얘기했지?"  
> "나 없는 동안 뭐 결정됐어?"

즉 Peer Final Transcript가 **번역 화면으로 흘러가는 transient data가 아니라 Agent가 조회할 수 있는 Session Context**가 되어야 한다.

#### Control Plane의 가장 직접적인 괴리

현재 `UiApplicationPort`는 이미 다음과 같은 많은 동작을 갖고 있다.

- `submit_text`
- `set_translation_enabled`
- `set_stt_enabled`
- `set_peer_translation_enabled`
- `set_overlay_enabled`
- capture option
- provider application
- diagnostics/logging
- calibration/settings operations

이는 PuriPuly가 이미 "조작 가능한 runtime"임을 의미하지만, 인터페이스가 UI workflow와 설정 관심사를 광범위하게 함께 품고 있다.

Agent에게 이 Port 전체를 그대로 노출하는 대신:

```text
ApplicationQuery
ApplicationCommand
SessionQuery
ContextQuery
Lifecycle
```

처럼 Agent와 Headless가 공유할 수 있는 더 작은 semantic surface를 추출할 필요가 있다.

목표는 **Agent가 PuriPuly의 거의 모든 유의미한 기능을 조작할 수 있게 하되, GUI 구현 세부사항에는 의존하지 않는 것**이다.

---

### 3-3. 에이전트 구조

MVP의 큰 구조는 다음을 목표로 한다.

```text
                 User
          ┌────────┴────────┐
          │                 │
        Text              Voice
          │                 │
  Agent Chat Phase      Self ASR Final
          │                 │
          │          Configurable Trigger
          └────────┬────────┘
                   │
              Agent Request
                   │
          Context Checkpoint
     ┌─────────────┼─────────────┐
     │             │             │
Conversation     Screen      Session/App
     │             │             │
     └─────────────┼─────────────┘
                   │
             Codex App Server
                   │
              Agent Loop
                   │
       ┌───────────┼───────────┐
       │           │           │
   Puri Tools   VR Context   Web/External
       │           │           │
       └───────────┼───────────┘
                   │
                 Result
          ┌────────┴────────┐
          │                 │
   Agent Chat Phase      VR Overlay
```

#### Application이 소유하는 것

- 현재 화면
- 현재 대화
- PuriPuly runtime
- VRChat Context
- Product permissions
- Tool definitions
- UI
- Context retention

#### Codex가 소유하는 것

- thread / turn
- Agent reasoning loop
- Tool selection
- streamed events
- approval interaction
- multi-step execution state

이 경계는 OpenAI가 2026년 8월 공개한 Codex App Server 패턴과도 일치한다. OpenAI의 공식 예시 역시 Application이 business context와 MCP data/actions를 소유하고 Codex App Server가 Agent loop를 제공하는 형태다.

#### Codex-first, Agent-core는 분리

MVP에서는 "미래에 어떤 모델도 꽂을 수 있는 완벽한 abstraction"을 먼저 만들지 않는다.

다만 다음은 Codex 전용 구조 안에 묻히지 않도록 한다.

- ConversationStore
- Screen Sensor
- VR Context Provider
- Puri Control Plane
- Authority/Privacy rules

이것들은 제품 자체의 자산이다.

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

Agent Request 발생 시 screenshot을 함께 checkpoint한다.

Codex App Server는 현재 Turn Input으로 다음 형태를 공식 지원한다.

- text
- image URL
- local image path

따라서 MVP에서는 로컬 screenshot을 `localImage`로 직접 넣는 방식이 자연스럽다.

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

초기 Tool은 Read-heavy로 시작한다.

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
- get_participants
- get_relevant_friend_state
- get_world_info

External
- web search / research
```

Write Tool은 점진적으로 추가한다.

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

1. Codex App Server approval
2. PuriPuly 자체 command policy

Read-only 조회와 reversible local toggle은 낮은 friction으로, 외부 행동이나 중요한 변경은 더 강한 confirmation으로 분류할 수 있다.

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

VR에서는 glanceable output을 우선한다.

```text
PURI
검색 중…

↓

PURI
조건에 맞는 후보 3개를 찾음.
1. ...
2. ...
3. ...
```

복잡한 Research 결과는 Agent Chat Phase에 남기고, 필요하면 사용자가 VR 안에서 짧은 follow-up을 음성으로 이어간다.

---

### 3-6. 외부 시스템 통합

#### Codex App Server — MVP의 핵심

2026년 8월 현재 OpenAI 공식 문서는 Codex App Server를 다음 용도로 명시한다.

> 제품 내부에 Codex를 깊게 통합하여 authentication, conversation history, approvals, streamed agent events 등을 직접 다루는 인터페이스.

또한 2026-08-19 OpenAI Developer Blog는 "제품 자체에 Agent가 포함되는 경우" App Server를 사용하라고 설명하고, Application이 자체 context와 tools를 소유하며 Codex가 Agent Loop를 담당하는 패턴을 제시한다.

따라서 기술적인 integration target은 명확하다.

MVP에서 우선 검토할 App Server 기능:

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

PuriPuly Tool은 Codex가 안정적으로 호출할 수 있는 MCP 또는 동등한 local tool surface로 제공하는 것을 우선 검토한다.

Experimental API에만 의존하는 기능은 MVP의 핵심 경로에서 가능한 한 피한다.

#### 인증

공식 App Server 문서상 다음 두 방식이 지원된다.

- ChatGPT managed authentication
- API key authentication

ChatGPT managed mode에서는 Codex가 browser/device-code OAuth flow와 token refresh를 소유한다.

PuriPuly MVP에서 ChatGPT subscription을 활용한다면 원칙은 다음과 같다.

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

- Codex App Server: https://developers.openai.com/codex/app-server
- Codex Authentication: https://developers.openai.com/codex/auth
- Codex as a platform (2026-08-19): https://developers.openai.com/blog/codex-as-a-platform
- OpenAI Terms of Use (effective 2026-01-01): https://openai.com/policies/terms-of-use/

#### Direct VRChat API

PuriPuly Standalone mode의 기본 Context Source.

단 Agent에 필요하지 않은 장기 Social Data까지 수집하는 방향으로 범위를 확장하지 않는다.

#### VRCX-0

Optional enrichment.

VRCX-0가 제공하는 MCP / Integration API / Headless 기능을 활용할 수 있다면 다음 문제를 재구현하지 않는다.

- 장기 Timeline
- Social History
- Rich Friend/Presence data
- Activity 분석

통합은 가능한 한 API/MCP 경계를 사용하고 내부 DB 포맷에 직접 결합하지 않는다.

참고:

- https://github.com/Map1en/VRCX-0

#### Realtime Audio

MVP 이후 별도 실험 축.

핵심 질문은 "기술적으로 붙일 수 있는가?"가 아니라 다음이다.

- Transcript Trigger보다 실제 UX가 좋아지는가?
- 비용/latency 복잡성을 감수할 가치가 있는가?
- Conversation context는 어차피 Puri ASR log를 유지해야 하는가?
- Audio-native 정보가 대표 기능을 실질적으로 개선하는가?

---

### 3-7. 구현 전략

#### Phase A — Codex MVP Vertical Slice

목표:

> **"Puri Agent Chat에서 텍스트 또는 음성으로 요청하면, 현재 Conversation + Screen + 최소 Context를 Codex가 보고 Puri Tool을 사용한 뒤 결과를 Agent Chat Phase에 반환한다."**

최소 구성:

1. Codex App Server local process integration
2. ChatGPT managed login 또는 API key login
3. 별도 Agent Chat Phase
4. Text command
5. Self ASR Final Transcript 기반 사용자 정의 Trigger
6. command-time screenshot
7. 최근 Self + Peer 원문 transcript
8. 최소 Puri read/control tools
9. 최소 Direct VRChat Context
10. streamed Agent progress/result
11. 기본 approval / authority boundary

이 Phase가 실제 MVP다.

#### Phase B — Context Foundation 강화

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
- World / Event / Friend search
- 필요한 최소 cache
- VRCX-0 optional enrichment
- source/freshness 충돌 처리

#### Phase E — 대표 기능 검증과 피벗

2-1 / 2-2에서 정의한 기능을 실제 VR 세션으로 검증한다.

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

결과에 따라 대표 기능을 강화하거나 2-1 자체를 교체한다.

#### Phase F — Realtime Audio / Beyond VRChat

제품 가설이 검증된 이후 검토한다.

- Realtime Audio Adapter
- 더 자연스러운 conversational turn-taking
- VRChat 외 Context Provider
- Discord / game / desktop session 적용
- Persistent automation
- 다른 Agent Runtime

이 단계에서야 PuriPuly가 "VRChat 번역기 + Agent"로 남는 것이 최적인지, "대화와 화면을 이해하는 더 범용적인 Agent"로 확장할 것인지 본격적으로 판단한다.

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

Codex는 그 Context를 바탕으로
무엇을 알아보고 무엇을 할지 결정하는 Agent Loop를 제공한다.
```

최종적으로 검증해야 하는 제품 경험은 하나다.

> **사용자가 AI에게 상황을 설명하는 대신, 그냥 현재 상황 속에서 짧게 말해도 되는가?**

그 경험이 강하다면 기존 PuriPuly의 번역 인프라는 단순 기능 하나가 아니라 Agent로 넘어가기 위한 가장 중요한 출발점이 된다.
