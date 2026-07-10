<p align="center">
  <img src="src/puripuly_heart/data/icons/icon.png" alt="PuriPuly <3" width="128" />
</p>

<h1 align="center">PuriPuly <3</h1>

<p align="center">
  <img src="https://img.shields.io/badge/version-2.2.2-blue" alt="Version" />
  <img src="https://img.shields.io/badge/license-AGPL--3.0--or--later-blue" alt="License: AGPL-3.0-or-later" />
  <img src="https://img.shields.io/badge/python-3.12-yellow" alt="Python" />
  <img src="https://img.shields.io/badge/platform-Windows-lightgrey" alt="Platform" />
</p>

<p align="center">VRChat용 LLM 기반 양방향 번역기</p>

<h2 align="center">
  <a href="README.md">🇺🇸 English</a> ·
  <a href="README.ar.md">🇸🇦 العربية</a> ·
  <a href="README.bg.md">🇧🇬 Български</a> ·
  <a href="README.ca.md">CA Català</a> ·
  <a href="README.cs.md">🇨🇿 Čeština</a> ·
  <a href="README.da.md">🇩🇰 Dansk</a> ·
  <a href="README.de.md">🇩🇪 Deutsch</a> ·
  <a href="README.el.md">🇬🇷 Ελληνικά</a> ·
  <a href="README.es.md">🇪🇸 Español</a> ·
  <a href="README.et.md">🇪🇪 Eesti</a> ·
  <a href="README.fi.md">🇫🇮 Suomi</a> ·
  <a href="README.fr.md">🇫🇷 Français</a> ·
  <a href="README.hi.md">🇮🇳 हिन्दी</a> ·
  <a href="README.hu.md">🇭🇺 Magyar</a> ·
  <a href="README.id.md">🇮🇩 Bahasa Indonesia</a> ·
  <a href="README.it.md">🇮🇹 Italiano</a> ·
  <a href="README.ja.md">🇯🇵 日本語</a> ·
  🇰🇷 한국어 ·
  <a href="README.lt.md">🇱🇹 Lietuvių</a> ·
  <a href="README.lv.md">🇱🇻 Latviešu</a> ·
  <a href="README.ms.md">🇲🇾 Bahasa Melayu</a> ·
  <a href="README.nl.md">🇳🇱 Nederlands</a> ·
  <a href="README.no.md">🇳🇴 Norsk</a> ·
  <a href="README.pl.md">🇵🇱 Polski</a> ·
  <a href="README.pt.md">🇵🇹 Português</a> ·
  <a href="README.ro.md">🇷🇴 Română</a> ·
  <a href="README.ru.md">🇷🇺 Русский</a> ·
  <a href="README.sk.md">🇸🇰 Slovenčina</a> ·
  <a href="README.sv.md">🇸🇪 Svenska</a> ·
  <a href="README.th.md">🇹🇭 ไทย</a> ·
  <a href="README.tr.md">🇹🇷 Türkçe</a> ·
  <a href="README.uk.md">🇺🇦 Українська</a> ·
  <a href="README.vi.md">🇻🇳 Tiếng Việt</a> ·
  <a href="README.zh-CN.md">🇨🇳 简体中文</a> ·
  <a href="README.zh-TW.md">🇹🇼 繁體中文</a>
</h2>

> ⚠️ **이것은 이식 가능한 포크입니다** [kapitalismho/PuriPuly-heart](https://github.com/kapitalismho/PuriPuly-heart). 간편한 배포 및 수정을 위해 수정되었습니다. [포터블 빌드 다운로드 ←](../../releases)

---

## 데모

![PuriPuly와 VRCT 간 번역 결과 비교.](docs/images/demo/ko-en_screenshot.png)

---

<video src="https://github.com/user-attachments/assets/c667f44d-b91d-42a9-b24a-e6a993b392d3" controls width="100%"></video>

PuriPuly를 통한 실제 외국 친구들과의 소통 더 보기:
- [데모 1](https://www.youtube.com/watch?v=3p0CamYui0o)
- [데모 2](https://youtu.be/DoX36Y7J_lc?si=YjbeVTS8v3jGQB1w)
- [데모 3](https://www.youtube.com/watch?v=D0npvp68xNY)

---

## 드디어, 진짜 친구처럼 말하세요.

당신도 그런 적이 있죠.
친구를 위로하고 싶었지만,
"괜찮아?"밖에 못했던.

'번역기'로는 진심을 전할 수 없다는 걸
이미 알고 계시죠.

그래서 진심을 전할 수 있는 번역기를 만들었습니다.

- **LLM 기반 로컬라이제이션** — 은어, 구어체, 격식/비격식 표현까지 자연스럽게 전달합니다.
- **컨텍스트 메모리** — 이전 대화 맥락을 인식하여 자연스럽게 대화가 이어집니다.
- **양방향 음성 번역** — 상대방의 목소리도 번역하며, VR 자막 오버레이를 지원합니다.
- **Discord로 시작** — 복잡한 설정 없이 바로 시작할 수 있습니다.

## 자주 묻는 질문

- **번역 품질은 어떤가요?**
→ 양쪽 모두 PuriPuly를 사용하면 가장 깊은 대화까지 가능합니다. 정량적으로, Gemma 4로 DeepL보다 6배 우수합니다. 아래 '번역 비교' 섹션에서 확인하세요.

- **말하기부터 번역까지 얼마나 걸리나요?**
→ Gemma 4와 클라우드 STT 서비스를 사용하면 지연 시간은 보통 1초대 중후반입니다.

- **유료인가요?**
→ 네, 하지만 나중에입니다. 신규 사용자에게 무료 사용량이 제공되며, 그 이후에도 가격이 매우 저렴합니다. $1로 수천 번 사용할 수 있습니다.

- **API 키가 필요한가요?**
→ 네, 하지만 역시 나중에입니다. 설치 후 Discord로 인증하면 바로 사용할 수 있습니다.

- **상대방 음성 번역 기능은 얼마나 잘 작동하나요?**
→ 조용한 환경에서 일대일 대화에 가장 적합합니다. 세 명까지는 가능하지만 보장되지는 않습니다. VRChat에서는 Earmuff를 사용하여 환경을 제어하세요.

- **음성 인식이 안 좋거나 느려요.**
→ 로컬 Qwen ASR을 사용 중이라면 클라우드 STT 서비스로 전환하는 것을 권장합니다. Intel CPU인 경우 PuriPuly를 P-cores에만 고정하도록 설정하세요.

- **음성과 대화 내용은 어떻게 처리되나요?**
→ 음성과 대화 내용은 로컬에 저장되며 PuriPuly 서버로 전송되지 않습니다. 다른 사람의 목소리, 전사 결과, 번역 결과는 절대 기록되지 않습니다. 단, STT 서비스와 번역 제공업체는 데이터를 처리할 수 있습니다.

### [📥 다운로드](https://github.com/kapitalismho/PuriPuly-heart/releases/latest)

---

## 번역 비교
![번역 품질 벤치마크 차트. Gemba MQM 프레임워크(심사 모델: Gemini 3.1 Pro Preview)로 216개의 다중 턴 한국어→영어/일본어/중국어 간체 샘플을 평가한 평균 오류 패널티(낮을수록 좋음).](docs/images/performance/1.png)

- Microsoft Gemba MQM 프레임워크를 사용하여 실험을 진행했습니다.
- 실제 대화에 더 가깝게 다중 턴 환경으로 설정했습니다.
- 전체 결과는 [여기](https://github.com/kapitalismho/korean-llm-context-translation-benchmark)에서 확인하세요.

## 비용

### 달러당 사용 횟수

#### 추천 모델

| LLM \ ASR | Qwen ASR (로컬) | Qwen ASR (클라우드) | Soniox | Deepgram |
|---|---|---|---|---|
| **Gemma 4 26B A4B** | 14,380 | 2,920 | 3,710 | 1,180 |
| **DeepSeek V4 Flash** | 19,410 | 3,080 | 3,980 | 1,210 |

#### 기타 모델

| LLM \ ASR | Qwen ASR (로컬) | Qwen ASR (클라우드) | Soniox | Deepgram |
|---|---|---|---|---|
| **Gemma 4 31B (Cerebras)** | 920 | 730 | 770 | 540 |
| **DeepSeek V4 Pro** | 6,400 | 2,330 | 2,810 | 1,070 |
| **Gemini 3 Flash** | 1,710 | 1,170 | 1,280 | 740 |
| **Gemini 3.1 Flash-Lite** | 3,430 | 1,770 | 2,030 | 940 |
| **Qwen 3.5 Plus** | 7,460 | 2,460 | — | — |
| **로컬 LLM** | 무제한 | 3,660 | 5,000 | 1,290 |

### 발화당 비용

#### 추천 모델

| LLM \ ASR | Qwen ASR (로컬) | Qwen ASR (클라우드) | Soniox | Deepgram |
|---|---|---|---|---|
| **Gemma 4 26B A4B** | ~$0.00007 | ~$0.0003 | ~$0.0003 | ~$0.0008 |
| **DeepSeek V4 Flash** | ~$0.00005 | ~$0.0003 | ~$0.0003 | ~$0.0008 |

#### 기타 모델

| LLM \ ASR | Qwen ASR (로컬) | Qwen ASR (클라우드) | Soniox | Deepgram |
|---|---|---|---|---|
| **Gemma 4 31B (Cerebras)** | ~$0.0011 | ~$0.0014 | ~$0.0013 | ~$0.0019 |
| **DeepSeek V4 Pro** | ~$0.0002 | ~$0.0004 | ~$0.0004 | ~$0.0009 |
| **Gemini 3 Flash** | ~$0.0006 | ~$0.0009 | ~$0.0008 | ~$0.0014 |
| **Gemini 3.1 Flash-Lite** | ~$0.0003 | ~$0.0006 | ~$0.0005 | ~$0.0011 |
| **Qwen 3.5 Plus** | ~$0.0001 | ~$0.0004 | — | — |
| **로컬 LLM** | $0 | ~$0.0003 | ~$0.0002 | ~$0.0008 |

*   *계산: (입력 900 토큰 + 출력 12 토큰) × 발화당 평균 1.2회 LLM 호출.*
*   *모든 비용과 사용량은 대략적입니다.*
*   *DeepSeek는 70% 캐시 히트율을 가정합니다.*
*   *Qwen API 비용은 베이징 지역 기준입니다.*
*   *2026년 5월 25일 기준 가격 / 빠른 응답 모드 활성.*

### 무료 크레딧

| 서비스 | 무료 크레딧 | 기간 | 비고 |
|--------|------------|------|------|
| **Deepgram** | $200 | 무제한 | - |
| **Google AI Studio** | $10 | 1년 | Gemini 구독자 매월 제공 |
| **Alibaba Cloud** | 모델당 1M 토큰 | 90일 | 싱가포르 지역 |
| **Alibaba Cloud** | ¥300 | 1년 | 중국 학생 |
| **Cerebras** | 매일 1M 토큰 | 무제한 | 분당 5회 호출 제한 |

---

# 문제가 있거나 무언가 불명확한 점이 있으면 [Twitter/X](https://x.com/kapitalismho)로 DM 주세요.

## 사용법

1. [다운로드 페이지](https://github.com/kapitalismho/PuriPuly-heart/releases/latest)에서 최신 버전을 다운로드하세요.
2. PuriPuly를 설치하세요.
3. **STT** 버튼을 클릭하세요.
4. **TRANS** 버튼을 클릭한 후 Discord로 인증하세요.
5. **Subtitles** 버튼을 클릭하여 VR 자막을 켜세요.
6. (선택) **Peer** 버튼을 클릭하여 상대방 음성 번역을 활성화하세요.

   > 상대방 음성 번역은 조용한 환경에서 가장 잘 작동합니다. VRChat에서는 Earmuff를 사용하여 환경을 제어하세요.

7. VRChat에서 OSC를 활성화하세요: 액션 메뉴 → 설정 → OSC → 활성화.

### 오디오 캡처가 작동하지 않는 경우
오디오 캡처가 작동하지 않으면 **설정 > 일반**을 열고 다음 단계를 따르세요.

1. **Audio Host API**를 **Auto** 또는 **MME**로 변경하세요.
2. 올바른 마이크를 선택하세요.
3. 앱을 다시 시작하세요.

---

### 중국 사용자를 위한 참고사항

지역에서 Soniox/Gemini/Deepgram이 차단된 경우 다음 조합을 사용하세요:

- STT: **Qwen ASR**
- LLM: **DeepSeek V4 Flash**

   > Discord 대신 QQ로 인증할 수 있습니다.

---

### 자체 API 키 사용하기

사용하려는 서비스의 가이드를 따르세요.

번역에는 OpenRouter를 통한 Gemma 4 모델을 추천합니다.

ASR도 설정하세요 — PuriPuly는 클라우드 STT와 함께 최고의 경험을 제공합니다.
같은 Qwen ASR이라도 로컬과 클라우드 인식의 차이가 뚜렷합니다.

Deepgram으로 시작하는 것을 추천합니다.
가입만 해도 $200 무료 크레딧을 받습니다.

<details>
<summary><h3>OpenRouter</h3></summary>

1. 스크린샷의 빨간 원 안의 옵션을 설정하세요.
   ![step0](docs/images/openrouter/0.png)

2. 앱에서 빨간 원 안의 버튼을 클릭하세요.
   ![step1](docs/images/openrouter/1.png)

3. OpenRouter에 로그인하세요.
   ![step2](docs/images/openrouter/2.png)

4. 빨간 원 안의 버튼을 클릭하여 결제 화면을 나가세요.
   ![step3](docs/images/openrouter/3.png)

5. **Authorize** 버튼을 클릭하세요.
   ![step4](docs/images/openrouter/4.png)

6. 사용할 금액만큼 선불 충전하세요.
   ![step5](docs/images/openrouter/5.png)

<details>
<summary><h3>Authorize 버튼이 작동하지 않은 경우</h3></summary>

Authorize를 클릭했지만 여전히 인증되지 않은 경우 다시 시도하거나 API 키를 직접 발급하세요:

6. 오른쪽 상단의 계정을 클릭하고, 왼쪽의 API Keys 탭으로 이동한 후 중앙의 Create 버튼을 클릭하세요.
   ![step6](docs/images/openrouter/6.png)

7. Create 버튼을 클릭하세요.
   ![step7](docs/images/openrouter/7.png)

8. API 키를 복사하는 버튼을 클릭한 후 번역기의 API 탭에 붙여넣으세요.
   ![step8](docs/images/openrouter/8.png)

</details>

</details>

<details>
<summary><h3>DeepSeek</h3></summary>

1. 스크린샷의 빨간 원 안의 옵션을 설정하세요.
   ![step0](docs/images/deepseek/0.png)

2. [DeepSeek 공식 홈페이지](https://www.deepseek.com/en/)에서 **Access API** 버튼을 클릭하세요.
   ![step1](docs/images/deepseek/1.png)

3. 홈페이지에서 로그인하세요.
   ![step2](docs/images/deepseek/2.png)

4. API Keys 탭으로 이동하여 **Create new API Keys**를 클릭하세요.
   ![step3](docs/images/deepseek/3.png)

5. API 키를 복사하는 버튼을 클릭한 후 번역기의 API 탭에 붙여넣으세요.
   ![step4](docs/images/deepseek/4.png)

6. Top Up 탭으로 이동하여 사용할 금액만큼 선불 충전하세요.
   ![step5](docs/images/deepseek/5.png)

</details>

<details>
<summary><h3>Deepgram</h3></summary>

1. [Deepgram Console](https://console.deepgram.com/)에 로그인하세요.
   ![step1](docs/images/deepgram/1.png)

2. 환영 메시지/설문이 보이면 **Skip**을 클릭하세요.
   ![step2](docs/images/deepgram/2.png)

3. 서비스 선택 화면에서 **STT (Speech-to-Text)**를 선택하세요.
   ![step3](docs/images/deepgram/3.png)

4. API Keys 메뉴에서 **Create a New API Key**를 클릭하세요.
   ![step4](docs/images/deepgram/4.png)

5. 키 이름(예: `puripuly`)을 입력하고 생성하세요.
   ![step5](docs/images/deepgram/5.png)

6. 생성된 키를 복사하여 PuriPuly 설정에 붙여넣으세요.
   ![step6](docs/images/deepgram/6.png)

</details>

<details>
<summary><h3>Gemini</h3></summary>

1. [Google AI Studio](https://aistudio.google.com/apikey)에서 **Get API key** 버튼을 클릭하세요.
   ![step1](docs/images/gemini/1.png)

2. 새 프로젝트를 만드세요.
   ![step2](docs/images/gemini/2.png)

3. 프로젝트 이름을 원하는 대로 지정하세요.
   ![step3](docs/images/gemini/3.png)

4. 만든 프로젝트를 선택하고 **Create key**를 클릭하세요.
   ![step4](docs/images/gemini/4.png)

5. 원형 영역을 클릭하세요.
   ![step5](docs/images/gemini/5.png)

6. 원형 영역을 클릭하여 키를 복사하세요.
   ![step6](docs/images/gemini/6.png)

7. (권장) 노란색 **Set Up Billing** 버튼을 클릭하여 유료 티어로 업그레이드하세요.
   ![step7](docs/images/gemini/7.png)

<details>
<summary><h3>Gemini 유료 구독자를 위한</h3></summary>

8. [Google Developer Program](https://developers.google.com/program/my-benefits)에 가입하세요.
   ![step8](docs/images/gemini/8.png)

9. 7단계에서 설정한 유료 티어 프로젝트를 선택하세요.
   ![step9](docs/images/gemini/9.png)

</details>

</details>

<details>
<summary><h3>Qwen</h3></summary>

1. 지역에 따라 Alibaba Cloud Model Studio에 접속하세요:
   - [중국 본토](https://bailian.console.aliyun.com/cn-beijing)
   - [중국 외 지역](https://bailian.console.alibabacloud.com)

2. 위 URL에서 로그인하세요. API 키에 맞는 올바른 지역(예: 베이징)을 선택하세요.
   ![step2](docs/images/qwen/1.png)

3. 오른쪽 상단의 **톱니바퀴 아이콘**을 클릭하세요.
   ![step3](docs/images/qwen/2.png)

4. 워크스페이스를 만들고 **API-KEY** 페이지로 이동하세요.
   ![step4](docs/images/qwen/3.png)

5. **Create API Key**를 클릭하세요.
   ![step5](docs/images/qwen/4.png)

6. 계정과 워크스페이스를 지정하고 확인을 클릭하세요.
   ![step6](docs/images/qwen/5.png)

7. 원형 영역을 클릭하여 키를 복사하세요.
   ![step7](docs/images/qwen/6.png)

</details>

<details>
<summary><h3>Soniox</h3></summary>

1. [Soniox Console](https://console.soniox.com/)에 로그인하세요.
   ![step1](docs/images/soniox/1.png)

2. 원하는 조직 이름을 입력하세요.
   ![step2](docs/images/soniox/2.png)

3. **Add Funds**를 클릭하여 결제 수단을 연결하세요.
   ![step3](docs/images/soniox/3.png)

4. Soniox는 선불 크레딧이 필요합니다. 충전 후 **API Keys** 메뉴로 이동하세요.
   ![step4](docs/images/soniox/4.png)

5. 새 API Key를 생성하세요.
   ![step5](docs/images/soniox/5.png)

6. 생성된 키를 복사하여 PuriPuly 설정에 붙여넣으세요.
   ![step6](docs/images/soniox/6.png)

</details>

<details>
<summary><h3>Cerebras</h3></summary>

1. [Cerebras](https://www.cerebras.ai/)에서 **Get started**를 클릭하세요.
   ![step1](docs/images/cerebras/1.png)

2. 로그인하세요.
   ![step2](docs/images/cerebras/2.png)

3. 원하는 플랜을 선택하세요. 무료 티어부터 시작하는 것을 권장합니다.
   ![step3](docs/images/cerebras/3.png)

4. API 키를 복사하여 PuriPuly에 붙여넣으세요.
   ![step4](docs/images/cerebras/4.png)

<details>
<summary><h3>유료 플랜으로 전환</h3></summary>

5. **Billing** 탭으로 이동하세요.
   ![step5](docs/images/cerebras/5.png)

6. 이름을 입력하세요.
   ![step6](docs/images/cerebras/6.png)

7. 필요한 만큼 크레딧을 충전하세요.
   ![step7](docs/images/cerebras/7.png)

</details>

</details>

---

## 개발

### 환경 요약

| 영역 | 권장 환경 |
|---|---|
| Python 앱 | Windows |
| VR 오버레이 | Windows |
| 브로커 서비스 | Linux / WSL |

### Python 앱

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
```

```bash
# pip
pip install -e '.[dev]'

# 또는 uv
uv sync --dev
```

```bash
pre-commit install
```

### GUI 실행

```bash
# venv 활성화 후
python -m puripuly_heart.main run-gui

# 또는 uv로 실행
uv run python -m puripuly_heart.main run-gui
```

```bash
# 숨겨진 UI를 검사용으로 표시
python -m puripuly_heart.main run-gui --debug-ui-preview
```

### 테스트 및 린팅

```bash
black src tests          # 포맷팅
ruff check src tests     # 린팅
python -m pytest         # 테스트 (venv에서 권장)
```

### VR 오버레이

VR 자막 오버레이는 `native/overlay/`의 Rust 프로젝트에서 빌드됩니다.

```powershell
cargo test --manifest-path native/overlay/Cargo.toml -q

cargo build `
  --manifest-path native/overlay/Cargo.toml `
  --locked `
  --release `
  --bin PuriPulyHeartOverlay `
  --target-dir target

New-Item -ItemType Directory -Force -Path build/overlay | Out-Null
Copy-Item target/release/PuriPulyHeartOverlay.exe build/overlay/PuriPulyHeartOverlay.exe -Force
Copy-Item third_party/openvr/win64/openvr_api.dll build/overlay/openvr_api.dll -Force

.\build\overlay\PuriPulyHeartOverlay.exe --check-startup-contract
```

### 브로커 서비스

자세한 내용은 `broker/README.md`를 참조하세요.

```bash
pnpm install --frozen-lockfile
pnpm run typecheck
pnpm exec vitest run
pnpm --filter @puripuly-heart/broker run verify:config
pnpm --filter @puripuly-heart/broker run dev
```

---

## 개발자

[salee](https://github.com/kapitalismho)

---

## 기여자

[RICHARDwuxiaofei](https://github.com/RICHARDwuxiaofei)

---

## 특별 감사

SUI\_32C, Nagikokoro, motoka96, \_Ykol魚, kascr\_, Just Monika V, FLUVIA, Han โชเล่ย์, EA\_PE, Ephedrine, ~ eri ~

---

## 라이선스

[AGPL-3.0-or-later](LICENSE)

서드파티 라이선스 및 고지: `src/puripuly_heart/data/THIRD_PARTY_NOTICES.txt`
