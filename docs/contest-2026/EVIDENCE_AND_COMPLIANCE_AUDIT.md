# PuriPuly Heart — 2026 오픈소스 개발자대회 제출 전 감사 (EVIDENCE & COMPLIANCE AUDIT)

## 1. 저장소 기준점 기록

| 항목 | 값 |
|---|---|
| Repository | https://github.com/kapitalismho/PuriPuly-heart (public) |
| 현재 branch | `integrate-gemma-4-e4b-q4-as-local-translation-ru` (origin에 push됨, upstream tracking 존재) |
| 현재 commit SHA | `cd06b9a57fd50a4806d0f9becdc177b0baaad856` (git log / gh API 양쪽에서 동일 확인) |
| 감사 일시 | 2026-08-20 (KST 기준 감사 실행) |
| 프로젝트 version | 2.4.0 (pyproject.toml `version = "2.4.0"`, `src/puripuly_heart/__init__.py`, `installer.iss`, `native/overlay/Cargo.toml`, `native/gpu_worker/Cargo.toml` 5개소 일치 — release.yml `verify-version` job이 tag와의 일치를 강제) |
| 주요 runtime | Python 3.12.10 (감사에 사용한 .venv) / requires-python `>=3.12,<3.14` / Rust 1.97.1 (cargo 1.97.1) / Node 22.21.1 + pnpm 10.33.0 (corepack) / uv 0.9.17 |
| 사용한 lockfile | `uv.lock` (Python), `pnpm-lock.yaml` (broker/monorepo), `Cargo.lock` (native — CI가 `--locked` 빌드) |
| 확인한 GitHub 기능 범위 | Issues(63), Pull Requests(80), Actions(PR CI/Release/Deploy/maintenance, dev 브랜치 히스토리), Releases(asset 다운로드 수), Project board(users/kapitalismho/projects/2), Milestones(3), Labels(13), Issue/PR templates, Branch protection(dev) — gh CLI로 전부 접근 가능 |
| 감사 중 실행한 테스트/명령 | 아래 §5, §K 참조 (pytest 전체/부분, ruff, black, pnpm typecheck, vitest, cargo, import smoke, i18n 검증) |

> **주의사항 — 감사 기준점**: 이 브랜치에는 감사 시점에 **미커밋 변경 13개 파일**이 존재했다
> (`translation_enable.py`, `core/local_translation/{provisioning,runtime}.py`, `providers/llm/managed_gemma.py`,
> `ui/views/settings.py`, `ui/components/settings/settings_modal.py`, `data/i18n/{en,ja,ko,ru,zh-CN}.json`,
> `tests/ui/test_settings_{prompt_switching,view_branches}.py` + 미추적 PRD 1건).
> 본 감사는 **HEAD(커밋된 상태) = `cd06b9a57` 기준으로 수행**했으며, 미커밋 작업은 "개발 중"으로 보고
> 점수·위험 평가에서 제외했다(테스트는 stash로 임시 제외한 뒤 HEAD 기준으로 실행하고 복원 완료 — 작업 트리는 원상 보존됨).
> 대회 제출 시 이 미커밋 변경의 정리(커밋/머지)가 전제된다.

> **2026-08-20 갱신 (감사 후속 조치)**: 아래 HIGH 항목 중 **H1(Gemma 4 E4B 모델 라이선스 미기재)** 와
> **LOW 항목 silero_vad.onnx sha256 미기재** 는 후속 커밋에서 해소되었다 —
> `core/local_translation/assets.py`에 `GEMMA_UPSTREAM_REPO_ID`/`GEMMA_LICENSE`(Apache-2.0)/`GEMMA_LICENSE_URL`
> 상수 및 설치 manifest 기록 필드 추가, `data/THIRD_PARTY_NOTICES.txt`에 Gemma 모델 섹션
> (upstream `google/gemma-4-E4B-it`, mirror revision `8c5a9e4f...`, size/sha256 2종) 및 Silero VAD
> 크기/sha256(`1a153a22...`) 명시. 이에 따라 §3-5 판정은 "PASS WITH RISKS (#H1 해소, #H2 잔존)",
> §7 점수표의 라이선스 항목은 4.0 / 5, 총점은 75.0으로 상향 조정이 가능하다.

---

## 2. 대회 평가표 100점 Evidence Map

### A. 프로젝트 구조 및 코드 완성도 — 6점

**확인한 구조:**
- `src/puripuly_heart/` — Python 패키지: `app`(포트/어댑터/서비스 513파일), `core`(러타임/도메인 440파일), `ui`(Flet 240파일), `providers`(llm 13파일 + stt 9파일), `domain`, `config`, `composition`(composition root: `composition/application_runtime.py`)
- `native/overlay` (Rust, OpenVR), `native/gpu_worker` (Rust, Vulkan + transcribe-cpp), `broker` (TypeScript + Hono on Cloudflare Workers, 101개 .ts)
- `tests/` — app 99 / core 116 / ui 51 / architecture 41 / config 34 / providers 21 / integration 14 / release_evidence 6 / scripts 1 / domain 1 파일
- Architecture boundary: `ARCHITECTURE.md`(362줄)에 owners / ports-adapters / data handoffs / composition / lifecycle / async model 문서화. 테스트로 경계 강제: `tests/architecture/` (41파일, HEAD 기준 221 passed)
- Provider abstraction: `app/ports/`, `core/stt/backend.py`(STTBackend 계약), `LLMProvider` 계약(managed_gemma가 `LLMProvider` 뒤에 숨음 — issue #75의 명시적 계약)
- CI: `.github/workflows/pr-ci.yml` — Lint(ruff+black), Contracts(domain/config), Architecture/Runtime/Providers/UI/Tooling 5개 matrix(전부 windows-latest), Broker(ubuntu vitest), Overlay(cargo check+test, windows). dev 브랜치 protected + required checks = Lint, Contracts, Broker
- Release/build: `.github/workflows/release.yml` — version sync 검증 → Vulkan SDK(sha256 고정) → Rust overlay/gpu_worker 빌드 → PyInstaller → `managed_gemma_distribution verify-package --launch` → Inno Setup 설치기 → GitHub Release(draft)
- Dead/stub 코드: HEAD 기준 `rg "TODO|FIXME" src tests` → **0건**
- 깨진 import: HEAD 기준 `import puripuly_heart.main` / `composition.application_runtime` 성공 (아래 §K)

**평가: 예상 5.5 / 6** — 신뢰도: 높음
- 강점: ports/adapters + 소유권 문서/테스트 일치, 4600+ 테스트, CI가 Windows 실기기 런타임까지 커버, release 파이프라인이 license/모델 배포 검증 포함
- 감점 위험: `black --check` 실패 파일이 HEAD에도 존재(§K — 최소 6개), `CODE_OF_CONDUCT.md`/`SECURITY.md` 부재(§I), 감사 시점 브랜치가 dev보다 13 commits 앞서 있어 머지 전 재검증 필요
- 가장 강한 증거 5: (1) `tests/architecture/` 221 passed (2) `ARCHITECTURE.md` 포트/소유권 표 (3) PR CI 8 jobs 구성 (4) `release.yml` verify-version·verify-package (5) TODO/FIXME 0건

### B. 오픈소스 프로젝트 발전 가능성 — 6점

- 확장 가능한 provider 구조: STT(deepgram/soniox/qwen/로컬 CPU·GPU), LLM(gemini/deepseek/qwen/openrouter/cerebras/local_openai/managed_gemma) — 신규 provider는 `providers/stt|llm/` + wiring 한 곳 추가로 확장. **실증**: 사용자 요청 #57("I would like to use my own models for speech recognition") → 2026-08-20 merged PR #80 "Generic Custom STT"(+3247줄)
- i18n 확장성: `data/i18n/{en,ja,ko,ru,zh-CN}.json` + `prompts/prompt-examples/language-pair/` 14개 파일 + `prompts/prompt-rules/target-language/` 4개 — good first issue #70(번역 규칙), #71(언어쌍 예문)이 구체적 참여 포인트 제공
- Roadmap/backlog: README → Project board 링크(users/kapitalismho/projects/2), Milestones 3개 활성("Add local translation model" 8/22 due, "Add speaker change detection feature", "Architecture Boundary Burn-down")
- Release history: v0.1.0 → v2.4.0 (27개 release, 마지막 v2.4.0 2026-08-12) — 월 2~3회 수준의 지속 출시
- 유지보수: dev 브랜치 보호(required checks + PR), pre-commit(ruff/black), `.coderabbit.yaml`(자동 리뷰), 외부 기여 6건 머지

**평가: 예상 5 / 6**
- 증거: PR #80(요청→기능), milestones, good-first-issue, 27개 release, ARCHITECTURE 문서가 확장 지침 역할
- 부족한 부분: provider 확장 절차서(contribution guide)가 CONTRIBUTING에 없음, roadmap이 "발전 방향"이 아니라 진행 중 작업 트래킹에 가까움

### C. 개발 문서 구체성 — 6점

- README.md: 설치(Download→TALK→TRANS→CAPTIONS→LISTEN→OSC), BYOK 가이드 6종(OpenRouter/DeepSeek/Deepgram/Gemini/Qwen/Soniox/Cerebras) 전부 스크린샷 포함, 트러블슈팅(audio capture, Intel P-core), 비용표, 벤치마크, 5개 언어 README
- 개발 환경: README "Development" 섹션 — `python -m venv .venv` / `pip install -e ".[dev]"` / `uv sync --dev` / `pre-commit install` / 실행 `python -m puripuly_heart.main run-gui` / 테스트 `python -m pytest` — **명령어가 실제 저장소와 일치**(본 감사에서 실행 검증, §5)
- `ARCHITECTURE.md`, `broker/README.md`(deploy 시크릿·엔드포인트 계약 상세), `native/overlay/README.md`(빌드/검증 명령), `docs/vrchat-osc.md`(OSC 파라미터 ABI), `docs/http-extensions.md` + JSON Schema
- **부족한 문서**: TESTING 문서 없음(README에 명령만), SECURITY 문서 없음, **로컬 모델 다운로드/설치 가이드 없음**(`scripts/installer/install-local-stt-model.ps1`과 model manifests는 존재하지만 README 어디에도 언급이 없고, 앱 내 UI 다운로드 흐름에만 의존), **로컬 번역(Gemma 4 E4B) 사용법이 README에 미기재**(i18n 문자열에는 존재)

**평가: 예상 4 / 6**
- 잘된 문서: README BYOK 6종 스크린샷 가이드, vrchat-osc ABI 문서, broker README
- stale/broken/ambiguous: local 모델 경로 미문서화(HIGH #H2), TESTING/SECURITY 부재, CONTRIBUTING이 15줄로 간결(커뮤니티 목적에 비해 얕음)

### D. 프로젝트 혁신성 — 6점

| 문제 | 기존 방식의 한계 | PuriPuly의 기술적 해결 | 코드/실험 증거 |
|---|---|---|---|
| VRChat 내 실시간 양방향 번역 | 1방향 기계번역/채팅위주 | self(내 말) + peer(상대 말) 투 채널 실시간 파이프라인, chatbox + VR 오버레이 동시 출력 | `core/orchestrator/self_translation_channel.py`, `core/runtime/peer_channel.py`, `core/runtime/output.py`, `docs/vrchat-osc.md` |
| VRChat 제어 | 단순 채팅 전송 | **양방향 OSC 제어 + OSCQuery 자동발견 + 표준 파라미터 ABI**(chatbox/마이크 뮤트 등) | `core/osc/oscquery.py`, `core/osc/` ABI, docs/vrchat-osc.md(파라미터 표) |
| 오버레이 | 채팅창 읽기 불편 | desktop overlay(Flet) + **native VR overlay(Rust + OpenVR)** + VR 자막 | `native/overlay/`(Rust), `src/puripuly_heart/core/overlay/`, PR #22(topmost 복원) |
| 오픈웨이트 로컬 실행 | 클라우드 STT/LLM 의존 | 로컬 STT(Parakeet/Qwen3-ASR, CPU sherpa-onnx + **GPU transcribe.cpp(Vulkan)**) + **로컬 번역(Gemma 4 E4B, llama.cpp CPU MTP 추론/Vulkan)** | manifests 4종(sha256 고정), `core/local_asr/`, `core/local_translation/`, `native/gpu_worker` |
| 실시간 동시통역 지연 | 말 끝나야 번역 | partial 기반 simultaneous translation(SIM-I02/SIM-R01~R03 이슈 흐름) | `providers/stt/` partial 이벤트 → `core/orchestrator/translation_turn.py` |
| 번역 품질 | 사전식 번역 | LLM 컨텍스트 메모리 + 언어쌍 규칙/예문, MQM 벤치마크(Gemba 프레임워크, judge Gemini) | `prompts/` 20개 파일, README 벤치마크 차트, 별도 공개 repo `kapitalismho/korean-llm-context-translation-benchmark` |
| 화자변경 감지 | 무시 | speaker-change turn boundary 연구(LB-EEND/ERes2NetV2) | milestone "Add speaker change detection feature", 이슈 #51, #58 |

**평가: 예상 5 / 6**
- 가장 강한 차별점 3: (1) VRChat 양방향 실시간 번역 + 양방향 OSC/OSCQuery 통합 (2) 로컬 오픈웨이트 전 경로(STT CPU/GPU + 번역 CPU/Vulkan, checksum 검증) (3) native VR overlay(Rust)와 실시간 동시통역 파이프라인
- 주의: 로컬 Gemma 품질 벤치마크(직접 실측값)는 공개되어 있지 않으므로 "로컬이 클라우드보다 우월" 주장 금지

### E. 프로젝트 팀워크 / 관리체계 — 6점

- Issues: 63건(open 14 / closed 49) — 템플릿 2종(bug_report, feature_request), 라벨 13종
- PR: 80건, PR 템플릿 존재(무엇/이유/테스트/리뷰노트), dev 브랜치 보호(PR 필수 + Lint·Contracts·Broker required checks)
- review/merge: PR #80 등 최근 PR이 squash merge; PR #80은 실패→수정→성공 흐름이 Actions 로그에 남음(2026-08-20 4회 실행)
- 외부 기여: merged PR 6건 — #1(중국어 번역), #5(VRChat mic 설정), #14(QQ 인증, +4705줄), #18(러시아어), #21(로컬 ASR non-blocking decode), #22(overlay topmost fix)
- 대회 보고서용으로 가장 좋은 개발 흔적:
  - **기술적 연구 이슈**: #75(Gemma 4 E4B 로컬 번역 통합 — 제품/런타임 계약서 수준의 본문), #64(Gemma 로컬 Pareto 실험), #76(PSEM 훈련 전략 게이트 실험), #58(화자변경 벤치), #49(SaT LoRA 연구)
  - **외부 contributor PR**: #18(Russian localization), #14(QQ auth for China), #21/#22(bugfix)
  - **CI/quality PR**: #60(PR CI 도입 + required/advisory checks), #63(Overlay job 추가), #62(중복 push-ci 제거)
  - **사용자 요청→구현**: #57 → #73 → #80(Generic Custom STT)

**평가: 예상 5 / 6**
- 가장 좋은 공개 개발 흔적: 사용자 요청(#57)이 3주 만에 기능(#80)이 된 흐름 + 외부 기여 6건 머지 + research-grade 이슈 문서화

---

### F. 발표 — 10점 (저장소로 점수 확정 불가)

- **발표에서 반드시 보여줘야 할 것**: (1) 실제 VRChat 화면 + chatbox 번역 실시간 흐름 (2) 로컬 모델 선택 UI → 다운로드 → 로컬 실행 (3) OSC 양방향 제어 (4) 벤치마크 결과
- **저장소에서 가져올 증거**: README 데모 영상 3종(YouTube 링크 2 + GitHub 첨부 1), ko-en 스크린샷 비교(Deepgram+Gemini vs VRCT), 벤치마크 차트, 이슈 #75 본문(계약서 수준), PR #80 흐름
- **발표에서 위험한 주장**: "전 세계 N명 사용 중", "로컬 모델이 클라우드보다 우수", "완전 무료" — 근거 없음(§9)
- **예상 점수: 7 / 10 (발표 품질 의존)** — 신뢰도: 낮음(저장소만으로 판단 불가)

### G. 실용성 — 15점

- Stars 105 / Forks 13 (gh 확인, 2026-08-20)
- Release asset 다운로드: v2.1.2 398회, v2.2.2 241회, v2.3.3 148회, v2.4.0 108회, v2.3.2 111회, v2.3.1 57회 등(전체 합산 ~2,000회+)
- 실제 사용자 이슈(closed): #2(레이턴시 증가, 3댓글), #8/#9(Kieaer — 한국어→다국어, OpenAI 호환 서버 키), #12/#11/#13(Sreap — overlay 동작), #16(aeongdesu — 프로세스 오디오 캡처), #33(aeongdesu — 비LLM 번역 API), #55(LIII-Works — STT 로드 버그), #57(AikoKiss — 커스텀 STT 요청)
- real-world 시나리오: README 데모 영상(실제 VRChat 교류), "Note for Users in China"(지역 차단 우회 조합), Discord/QQ 인증
- **결과보고서에서 강조할 부분**: 사용자 요청이 실제 기능이 된 루프(#57→#80, #9→local OpenAI provider, #33→custom HTTP extension, #16→process capture), 다운로드 수, 사용자 4개국어 이슈

**평가: 예상 10 / 15** — 객관적 증거는 stars/다운로드/이슈뿐이며 "실사용자 수"의 직접 증거는 없음(보수 산정)

### H. 시연 및 완성도 — 10점

- 프로그램 정상 실행: HEAD 기준 import/startup smoke OK(§K), v2.4.0 설치기 배포(108 다운로드)
- 3분 데모 순서 제안:
  1. 앱 시작 → 로컬 STT(Parakeet) 선택 + 마이크 발화 → 즉시 자막
  2. 번역 켜기(인증 화면 생략 가능) → 출력 UI/chatbox
  3. 로컬 Gemma(CPU/GPU) 선택 → 다운로드 진행률 → 로컬 번역 실행(성능 메트릭 표시)
  4. VRChat 연결 → OSC 활성화 → chatbox에 번역 도달
  5. 상대 음성(LISTEN) → peer 번역 → 오버레이 표시
- **demo blocker**: 로컬 Gemma 첫 실행 시 4.2GB+60MB 다운로드 필요(네트워크 의존) — 사전 설치/데모 직전 준비 필수; 로컬 STT 품질은 환경 의존(README도 cloud 권장)
- **demo risk**: 라이브 음성 인식 실패 시 대체 시나리오(음성 파일 재생→STT) 준비 필요, 오디오 장치 설정(Auto/MME) 시행착오 가능

**평가: 예상 7 / 10** — 신뢰도: 중간(라이브 환경은 저장소로 검증 불가)

### I. 커뮤니티 확장 가능성 — 10점

- 있음: CONTRIBUTING.md(절차: 브랜치→작업→CI 통과→dev로 PR), Issue/PR 템플릿, good first issue 2건 활성(#70, #71, 2026-08-14 생성), help wanted, 라벨 13종, external PR 6건 merged(한국어/중국어/러시아어 등 다국적), project board 공개 링크, quality process(PR CI + required checks + pre-commit + coderabbit)
- 형식적으로만 존재(실사용 흔적 없는 요소): CONTRIBUTING이 요구하는 상세 절차가 최소화되어 있음(CI 실행 방법 등 README 위임), FUNDING.yml 존재하지만 활용 여부 불명확
- **없음**: CODE_OF_CONDUCT.md, SECURITY.md(securityPolicyUrl 없음 — gh 확인)
- "외부인이 실제로 참여할 수 있는가?" → YES, 이미 2개국 3명의 외부 기여와 6건 머지로 실증됨. 다만 문서화된 온보딩 경로(테스트 실행법, 로컬 모델 셋업법)가 부족해 진입 장벽 존재

**평가: 예상 6.5 / 10**

### J. 오픈소스 적절성 — 10점

- Source license: AGPL-3.0-or-later — root LICENSE, pyproject, package.json, broker/package.json, native 2개 Cargo.toml 전부 일치
- 팀 작성 source 전체 공개: Python/UI/네이티브/broker(Cloudflare Worker 포함) 전부 공개. 미공개 component 없음(브로커 운영용 시크릿은 배포 설정일 뿐 소스 아님)
- 재사용/확장: ports/adapters 구조, provider 추가 용이, custom HTTP extension은 JSON Schema로 공개 확장점 제공(docs/http-extension.schema.json)
- 외부 dependency 사용 방식: uv.lock/pnpm-lock.yaml/Cargo.lock 핀 + CI `--locked`, llama.cpp는 릴리스 아카이브 sha256 고정, Vulkan SDK도 sha256 고정
- generated/binary-only 영역: PyInstaller 바이너리(부트스트랩 예외 포함 GPL-2.0 문구 명시), vendored openvr_api.dll(BSD-3, sha256 포함), bundled fonts(OFL, Noto는 sha256 포함)
- 상용 서비스 의존: broker는 관리형 무료체험 크레딧 발급용 제어평면이며 **핵심 기능(번역 데이터 경로)은 로컬/직접 키로 동작**(§3 참조)

**평가: 예상 8 / 10**
- 위험요소: Gemma 모델 라이선스 미기재(HIGH), 상용 SDK가 default deps에 존재(§3-4), 모델 mirror의 license 메타데이터가 불완전할 수 있음을 NOTICES 자체가 경고

### K. 기능 테스트 — 10점 (실측 결과)

| 명령 | 결과 | passed/failed/skipped | 비고 |
|---|---|---|---|
| `uv run pytest tests/domain tests/config` | ✅ | 760 passed | |
| `uv run pytest tests/architecture` | ✅ | 221 passed | HEAD 기준 |
| `uv run pytest tests/providers` | ✅ | 331 passed | |
| `uv run pytest tests/core` (openrouter 등 일부 파일 제외) | ✅ | 1350 passed | 제외 파일은 WIP 연관 컬렉션 에러(§아래) |
| `uv run pytest tests/ui` (settings 등 일부 제외) | ✅ | 329 passed | |
| `uv run pytest tests/release_evidence` (1파일 제외) | ✅ | 46 passed | |
| `uv run pytest tests/scripts tests/integration` | ✅ | 30 skipped | skip은 Windows PowerShell/오디오 파일 부재 조건 |
| **HEAD 기준 전체(일부 파일 제외)** | ✅ | **4628 passed, 30 skipped, 266 warnings** | openrouter·peer_owned 등 9개 glob 제외 |
| `uv run ruff check src tests` | ✅ | All checks passed | |
| `uv run black --check src tests` | ❌ | **9 files would reformat** | HEAD에도 최소 6개 해당(`runtime_profile.py`, `oscquery.py`, `test_orchestrator_pipeline.py`, `test_context_memory.py`, `test_settings_vnext_migration_serialization.py`, `test_llm_user_messages.py`) |
| `corepack pnpm typecheck` | ✅ | OK | |
| `pnpm exec vitest run broker/tests` | ✅ | 64 files, 491 passed | |
| `cargo test --locked native/overlay` | ❌ | 로컬 CMake/openvr_sys 빌드 실패 | 환경 이슈(FileTracker/로케일) — **CI(PR CI Overlay job)는 success 기록 존재**, 대회 blocker 아님 |
| `python -c "import puripuly_heart.main"` / `composition.application_runtime` | ✅ | OK (HEAD) | |
| i18n JSON 5개 파싱 | ✅ | OK | |

- 대회 제출 blocker 여부: **없음**(HEAD 기준). 단 미커밋 WIP 포함 상태에서는 `TranslationEnableOwner` dataclass 에러로 49개 컬렉션 에러 발생 — **커밋/머지 후 반드시 재실행 필요(HIGH #H4)**

**평가: 예상 8 / 10**

### L. 라이선스 검증 — 5점

| 항목 | 상태 | 분류 |
|---|---|---|
| root LICENSE | AGPL-3.0 전체 텍스트 | ✅ |
| pyproject/package.json/broker/Cargo × 2 | 전부 AGPL-3.0-or-later | ✅ |
| THIRD_PARTY_NOTICES.txt | 329줄, 런타임 구성요소별 소스/핀/라이선스 명시 | ✅ |
| llama.cpp 런타임 | MIT, b10423 + commit `a94d563ed` 고정, LICENSE 3rd_party에 체크섬 검증 | ✅ |
| Qwen3-ASR 0.6B/1.7B | Apache-2.0, mirror+revision+sha256 고정 | ✅ (mirror의 license 메타데이터 불완전 가능성 자체 경고문 포함 — 주의) |
| NVIDIA Parakeet 0.6B v3/ja | CC-BY-4.0, mirror+revision+sha256 고정 | ✅ (위와 동일 주의) |
| Silero VAD (bundled onnx) | MIT 명시, 버전 v6.2.1 — **sha256 미기재** | LOW |
| OpenVR dll (bundled) | BSD-3-Clause, v2.15.6 pin + sha256 파일 | ✅ |
| Noto Sans CJK (bundled) | OFL-1.1 + tag/size/sha256 | ✅ |
| soxr.dll | LGPL-2.1 + 설치본 compliance bundle(소스 zip 포함) | ✅ |
| PyInstaller | GPL-2.0 + Bootloader 예외 문구 | ✅ |
| **Gemma 4 E4B 모델 (새 로컬 번역)** | **license 미기재** — assets.py에 license 필드 없음, NOTICES에 Gemma 모델 섹션 없음. unsloth mirror(revision `8c5a9e4f...` 고정, sha256 고정)와 "미번들" 방침은 확실하나 **모델 라이선스 표기 부재** | **HIGH** |
| gemma 모델 weight 미번들 보장 | `managed_gemma_distribution.py` — installer/패키지 내 `.gguf`/모델 ID 금지 검증 + `verify-package --launch` | ✅ (강력한 증거) |
| converted/mirror provenance | ASR 3종은 상세, Gemma는 모델 설명 없음 | HIGH(위) |

**평가: 예상 3.5 / 5** — 전체적으로 매우 충실한 체계(sha256, revision, compliance bundle, 미번들 보장)지만 제출 핵심인 Gemma 로컬 모델의 라이선스 표기가 비어 있음

---

## 3. AI / Commercial API 컴플라이언스 감사

### 3-1. Commercial provider dependency 조사

pyproject default dependencies에 포함된 상용/클라우드 SDK: `google-genai`, `dashscope`, `deepgram-sdk`, `grpcio`, `websockets`(Soniox 원시 WS) — **전부 optional extras가 아니라 기본 설치 대상**이다. 그러나:

- **import 시점**: 전부 **지연 import**(함수 내부) — `wiring_stt_factory.py:686,700,712`(deepgram/qwen_asr/soniox 백엔드는 provider 선택 분기에서만 import), `gemini.py:151,172`(`from google import genai`), `qwen.py:230`(`import dashscope`) 등. 상용 SDK가 `import puripuly_heart.main` 경로를 차단하지 않음(HEAD 기준 import smoke 성공으로 실증)
- **API key 없이 startup**: 키 없이 앱 시작 가능(README 사용 흐름: TALK → TRANS에서 인증 시작). startup 경로에서 상용 키를 요구하지 않음(`composition/application_startup.py` 참조 — 키 검증은 provider 활성화 시점)
- **핵심 기능 종속성**: STT는 deepgram/soniox/qwen cloud 외에 **로컬 경로 3종(parakeet v3, parakeet ja, qwen0.6B CPU + qwen1.7B GPU)** 이 동일 `STTBackend` 계약으로 존재. 번역은 gemini/deepseek/qwen/openrouter/cerebras 외에 **managed Gemma(로컬) + local_openai(사용자 서버)** 존재
- **판정: 상용 SDK가 default 설치에 포함되지만 "핵심 기능이 상용 API에 종속"되지는 않음** — 로컬 경로가 동일 계약으로 동작(§3-3). 위반 아님.

### 3-2. Open-weight direct execution 조사

| 모델 | ID/manifest | upstream | mirror/converted | revision | checksum | license | runtime | 로컬 실행 방법 | 네트워크/승인 |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-ASR 0.6B (CPU) | `qwen3-asr-0.6b-int8-sherpa` | Qwen/Qwen3-ASR-0.6B | csukuangfj2/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25 (+modelscope 미러) | `2cc50d1a...` / `c69fb16...` | 파일별 sha256 6종 | Apache-2.0 | sherpa-onnx(사용자 머신) | manifest 다운로드 → ONNX 로컬 추론 | 다운로드 시에만 / 없음 |
| Qwen3-ASR 1.7B (GPU) | `qwen3-asr-1.7b-q6-k-transcribe-vulkan` | Qwen/Qwen3-ASR-1.7B@7278e1e | handy-computer/Qwen3-ASR-1.7B-gguf | `92282af1...` | `c75a9617...` | Apache-2.0 | transcribe.cpp-vulkan (Rust gpu_worker) | GGUF → Vulkan 로컬 추론 | 다운로드 시에만 / 없음 |
| Parakeet 0.6B v3 / ja (CPU) | `parakeet-tdt-0.6b-v3-int8-sherpa` 외 1 | nvidia/parakeet-tdt-0.6b-v3, ...ja | csukuangfj/sherpa-onnx-nemo-parakeet-... | `2bda32ec` / `bef18eb0` | 파일별 sha256 | CC-BY-4.0 | sherpa-onnx | 동일 | 동일 |
| **Gemma 4 E4B (번역, CPU/GPU)** | `gemma-4-e4b-it-qat-ud-q4-k-xl` | (Google Gemma 4) | **unsloth/gemma-4-E4B-it-qat-GGUF** | `8c5a9e4f...` | 2종(`df0fd4ee...` 4215695776B, `423074e5...` 59678016B) | **미기재 (HIGH)** | llama.cpp b10423 (CPU MTP / Vulkan, 사용자 머신) | `ensure_gemma_installed` → llama-server 로컬 HTTP(127.0.0.1) → 번역 | 다운로드 시에만 / 없음 |
| Silero VAD | bundled | snakers4/silero-vad | — | v6.2.1 | 미기재(LOW) | MIT | onnxruntime | 번들 로컬 | 없음 |

**경로 구분**: "open-weight API 엔드포인트 호출"이 아니라 **downloaded weights → user's machine → local inference (sherpa-onnx / llama.cpp / transcribe.cpp) → PuriPuly** 경로임을 코드로 확인:
- `core/local_translation/provisioning.py` — HF Xet 다운로더, size+sha256 검증, staging → atomic install(backup rename), cancel/retry
- `core/local_translation/runtime.py` — llama.cpp 로컬 프로세스(127.0.0.1 loopback), health/readiness, CPU/Vulkan 프로파일, prefix 캐싱
- `release_evidence/managed_gemma_distribution.py` — 릴리스 아카이브 sha256 고정, 패키지에 모델 weight 금지, `verify-package --launch`로 로컬 실행 검증
- ASR manifests(`data/models/*.json`) — 다운로드 URL 템플릿 + 파일별 sha256

### 3-3. Core function equivalence

```
입력 음성 → STT → 번역 → VRChat output
```
이 동일 핵심 파이프라인이 **로컬 경로에서도** 동작:
- STT: `STT_PROVIDER_LOCAL_CPU_AUTO / LOCAL_PARAKEET_JAPANESE / LOCAL_PARAKEET_V3 / LOCAL_QWEN / LOCAL_QWEN_GPU` → `providers/stt/local_*.py`가 `STTBackend` 계약으로 `wiring_stt_factory.py`에서 동일하게 조립됨(클라우드와 동일 분기 구조)
- 번역: `PROVIDER_MANAGED_GEMMA` / `PROVIDER_LOCAL_LLM` → `LLMProvider` 계약 → 동일 `translation_turn`/output 파이프라인
- 기능 차이(기록): (1) 관리형 Gemma 선택 시 provider fallback 비활성화(issue #75 계약, 의도된 동작) (2) README가 로컬 STT 품질은 클라우드 대비 낮을 수 있음을 명시(환경 의존) (3) 로컬 번역 최초 준비 시 대용량 다운로드 필요 — **핵심 기능 자체는 동등하게 성립**

### 3-4. Optional dependency 감사

- 상용 SDK(google-genai, dashscope, deepgram-sdk)가 `pyproject.toml [project.dependencies]`에 기본 포함되어 있고 optional extras가 아님. 이상적 구조는 `core deps + optional cloud extras`(지연 import가 이미 되어 있어 구조 전환은 저위험).
- 지연 import + 로컬 경로 동등성 때문에 **규정 위반이 아니며 동작상 영향도 없음** → 구조 개선으로 **MEDIUM #M3** 기록(이번 감사에서는 변경하지 않음)

### 3-5. 핵심 질문 4종 요약

1. **Commercial API key가 없어도 핵심 기능 실행 가능?** → **YES** (로컬 STT 3종 + 로컬 번역 Gemma/local_openai. 앱 시작도 키 불필요)
2. **Open-weight를 직접 다운로드/실행 가능?** → **YES** (manifest + sha256 + 로컬 런타임, 코드로 실증)
3. **README/문서로 재현 가능?** → **PARTIAL** (코드·매니페스트·릴리스 검증 툴로는 완전 재현 가능하나, README에 로컬 모델 다운로드/사용 가이드가 없음 → HIGH #H2)
4. **모델 license/provenance 설명 가능?** → **PARTIAL** (ASR 3종은 완비, **Gemma는 미기재** → HIGH #H1)

**판정: PASS WITH RISKS** (#H1, #H2 해소가 조건)

---

## 4. Source / License / Model 공개성 감사

| 질문 | 답변 | 근거 |
|---|---|---|
| 팀 작성 source 전체가 공개? | YES | Python/UI/native/broker 전부 repo에 존재, binary는 전부 소스로 빌드 |
| OSI 승인 license 적용? | YES | AGPL-3.0-or-later(OSI 승인) |
| 핵심 기능에 필요한 비공개 component? | NO | 미공개 서버 없음(broker는 제어평면, 로컬 경로는 전부 공개 코드) |
| AI 모델 최소 open-weight? | YES | Qwen3-ASR/Parakeet(open-weight ONNX/GGUF), Gemma 4 E4B(open-weight GGUF) |
| 모델 다운로드에 별도 승인? | NO | HF public repo, 승인/로그인 없음 |
| 모델 license 명확? | PARTIAL | ASR 3종 YES, **Gemma NO(HIGH)** |
| converted/mirror provenance 명확? | PARTIAL | revision+sha256은 전부 명확, mirror가 원본 라이선스 메타데이터를 갖고 있지 않을 가능성을 NOTICES가 자체 경고 |
| inference code 공개? | YES | sherpa-onnx/llama.cpp/transcribe.cpp 사용 + PuriPuly 자체 어댑터/프로비저닝 공개 |
| model revision 고정? | YES | 전 모델 HF revision 고정 |
| artifact checksum 존재? | YES(대부분) | Gemma/ASR/llama.cpp/OpeNVR/Noto는 sha256. silero_vad.onnx만 미기재(LOW) |
| third-party notice가 distribution에 포함? | YES | THIRD_PARTY_NOTICES.txt(설치본 포함) + soxr LGPL bundle + third_party/llama.cpp/LICENSE |

---

## 5. Clean-room 재현성 감사

**목표 체인: clone → install → tests → startup → local path**

- `uv sync --dev` (uv.lock 존재) — 감사에서 uv run으로 검증 완료. `pip install -e ".[dev]"` 경로도 README에 명시
- 테스트: `python -m pytest` / `uv run pytest` — README 명령 일치(§K 실측)
- 실행: `python -m puripuly_heart.main run-gui` — README 명령 일치
- 로컬 모델: **설치/다운로드 단계가 README에 없음** — 앱 UI에서 자동 다운로드하므로 "암묵적 단계"로 존재(HIGH #H2). `scripts/installer/install-local-stt-model.ps1`은 스크립트로 존재하나 문서 링크 없음

**발견 목록 (심각도):**
| 항목 | 심각도 |
|---|---|
| README에 로컬 STT/Gemma 모델 다운로드·사용 가이드 부재 | HIGH |
| README에 로컬 번역(Gemma CPU/GPU 선택) UI 사용법 부재 | HIGH |
| silero_vad.onnx 체크섬 미기재 | LOW |
| 개발 문서가 "Windows 전제"인데 Linux/WSL 보조 경로만 언급(.venv-wsl) — 실제 검증은 Windows에서만 | LOW |
| 로컬 cargo 빌드가 Windows 로케일/CMake 환경에 민감(FileTracker 이슈) — CI에서는 안정 통과 | MEDIUM |
| 절대경로·로컬 전용 파일·secret: 발견 없음(`.env.local.example`은 예시 파일, keyring 사용) | — |
| 사전 다운로드된 모델 의존: 없음(모델 미번들 정책을 코드로 강제) | — |

---

## 6. 최종 위험도 분류

### BLOCKER — **0건**
HEAD 기준 제출 불가 사유 없음(핵심 기능 동작, 소스 전체 공개, 라이선스 위반 없음, 로컬 경로 존재).

### HIGH — 5건
| ID | 내용 | 근거 |
|---|---|---|
| H1 | **Gemma 4 E4B 모델 라이선스 미기재** — `core/local_translation/assets.py`에 license 필드 없음, THIRD_PARTY_NOTICES에 Gemma 섹션 없음(모델 weight 미번들은 명확하지만 라이선스·이용약관 표기 부재) | §3-2, §L |
| H2 | **로컬 모델 사용 가이드 문서 부재** — README에 로컬 STT/Gemma 다운로드·설정 절차 없음. open-weight 경로가 "코드로는 재현 가능하나 문서로는 재현 불가" 상태 | §3-5, §5 |
| H3 | **`black --check` 실패 파일 존재(HEAD 최소 6개)** — PR CI Lint job이 black을 체크하므로 dev 머지 시 CI 실패 위험 | §K |
| H4 | **미커밋 WIP 13파일 + 브랜치 상태 관리** — 이 브랜치는 dev보다 13 commits 앞서 있고, 감사 시점 WIP 상태에서 `TranslationEnableOwner` dataclass 에러로 49개 테스트 컬렉션 실패. 커밋/머지 후 전체 재실행 필수 | §1, §K |
| H5 | **v2.4.0 릴리스(8/12) 이후 기능이 미릴리스** — 로컬 Gemma 통합은 HEAD에는 존재하지만 release 태그에는 없음. 대회 제출이 v2.4.1+ 기준이라면 release 워크플로 재실행 필요 | gh release 확인 |

### MEDIUM — 6건
- M1 CODE_OF_CONDUCT.md 부재 (§I)
- M2 SECURITY.md 부재 (securityPolicyUrl 없음, §I)
- M3 상용 SDK가 core dependencies에 기본 포함(optional extras 아님) — 구조 개선 제안, 위반 아님 (§3-4)
- M4 silero_vad.onnx sha256 미기재 (§L)
- M5 로컬 cargo overlay 빌드가 Windows 환경 민감(CI는 통과) (§5)
- M6 project board(users/kapitalismho/projects/2)의 공개 접근성 미확인(gh로는 owner 권한으로 접근해 "공개" 여부를 독립 검증 못 함)

### LOW — 4건
- L1 i18n 다수 description 빈 문자열 (`"provider.gemma4_managed.description": ""` 등)
- L2 README 뱃지가 정적(버전 수동 동기화)
- L3 벤치마크 이미지/차트가 스크린샷 정적 데이터(재현 스크립트 비공개 — 외부 벤치 repo로 보완됨)
- L4 release-template.md가 dev 브랜치 링크 사용(main 아님) — repo 기본 브랜치가 dev이므로 실무 문제 없음

---

## 7. 예상 점수표 (보수적 추정)

| 평가항목 | 배점 | 현재 예상 | 신뢰도 | 가장 강한 증거 | 주요 위험 |
|---|---:|---:|---|---|---|
| 구조/완성도 | 6 | 5.5 | 높음 | tests/architecture 221 passed, ARCHITECTURE.md, PR CI 8 jobs | black 실패, 미커밋 WIP |
| OSS 발전 가능성 | 6 | 5.0 | 높음 | #57→#80 기능 루프, milestones 3, release 27건 | 확장 절차 문서 부재 |
| 개발 문서 | 6 | 4.0 | 중간 | BYOK 6종 가이드, vrchat-osc ABI, broker README | 로컬 모델 문서 부재, TESTING/SECURITY 없음 |
| 혁신성 | 6 | 5.0 | 중간 | 양방향 OSC+OSCQuery, 로컬 전 경로, native overlay | 로컬 Gemma 품질 실측 미공개 |
| 팀워크/관리 | 6 | 5.0 | 높음 | 외부 기여 6건, dev 보호, research 이슈 | 개인 주도 |
| 발표 | 10 | 7.0 | 낮음 | 데모 영상 3종, 벤치 차트, #75 본문 | 발표 품질 의존 |
| 실용성 | 15 | 10.0 | 중간 | 다운로드 ~2,000+, stars 105, 사용자 이슈 9건 | 실사용자 수 직접 증거 없음 |
| 시연/완성도 | 10 | 7.0 | 중간 | HEAD 실행 smoke OK, 로컬 경로 실재 | Gemma 첫 다운로드 4.2GB |
| 커뮤니티 확장 | 10 | 6.5 | 중간 | 외부 PR 6건 merged, good-first-issue 2 | CoC/SECURITY 부재 |
| OSS 적절성 | 10 | 8.0 | 높음 | AGPL 일관, 모델 미번들 강제, licenses 체계 | Gemma license 미기재 |
| 기능 테스트 | 10 | 8.0 | 높음 | 4628 passed/30 skipped, vitest 491, typecheck | black 6파일, 로컬 cargo |
| 라이선스 | 5 | 3.5 | 중간 | sha256·revision·compliance bundle 체계 | Gemma license 공백 |
| **TOTAL** | **100** | **74.5** | — | — | — |

*※ 점수는 "증거 부족 = 감점" 기준의 보수적 추정이며, 발표(10)는 저장소 외적 요인으로 품질 의존. H1~H4 해소 시 +2~3점, 발표가 좋으면 +2~3점 여지.*

---

## 8. 제출 전 Action List (Top 8, 8/27 기준 ROI 순)

| # | 우선순위 | 문제 | 왜 중요한가 | 수정 파일/위치 | 작업량 | 완료 확인 |
|---|---|---|---|---|---|---|
| 1 | P0 | 미커밋 WIP 13파일 정리 + 브랜치 머지 | 현재 WIP 상태에서 테스트 49개 컬렉션 실패 → 제출 상태가 broken일 수 있음 | 브랜치 전체(커밋 → dev PR) | Small | PR CI green + `uv run pytest` 전체 passed |
| 2 | P0 | Gemma 모델 라이선스 명시 | AI 모델 라이선스 설명 불가 상태 = 대회 규정 리스크(HIGH) | `core/local_translation/assets.py`(license 상수), `data/THIRD_PARTY_NOTICES.txt`(Gemma 섹션 추가) | Small | NOTICES에 모델·라이선스·upstream·revision·checksum 표기 |
| 3 | P1 | README에 로컬 모델(ASR/Gemma) 사용 가이드 추가 | "open-weight 직접 실행 경로의 문서 재현성" — 규정 핵심 질문 3번째 | README.md(+ko/ja/ru/zh-CN) "Local models" 섹션 | Small | 문서만 따라 clean machine에서 로컬 번역 실행 가능 |
| 4 | P1 | black --check 실패 6파일 포맷 | dev 머지 시 Lint job(required check) 실패 | `runtime_profile.py`, `oscquery.py`, `test_*` 4종 | Small | `black --check src tests` green |
| 5 | P1 | 커밋/머지 후 전체 테스트 재실행 + release v2.4.1+ 발행 | HEAD 이후 기능(로컬 Gemma)이 릴리스 태그에 없음 | release.yml 태그 push | Medium | Release workflow success + 새 installer 다운로드 확인 |
| 6 | P2 | CODE_OF_CONDUCT.md / SECURITY.md 작성 | 커뮤니티·보안 항목의 형식적 공백(HIGH→MEDIUM 전환) | 루트 2개 파일 | Small | 파일 존재 + README 링크 |
| 7 | P2 | 로컬 모델 스모크 테스트(실기기) — Parakeet STT + Gemma CPU/GPU 각 1회 | 데모/발표 전 기능 확정, "테스트에서만 동작" 리스크 제거 | 로컬 실행(모델 다운로드 ~4.3GB) | Medium | 실제 음성→번역 1회 성공 기록 |
| 8 | P2 | 상용 SDK를 optional extras로 이동(구조) | 규정 관점 "core는 pure" 구조 강조, 소분류 | pyproject.toml `[project.optional-dependencies] cloud` + 지연 import 유지 | Medium | clean install 후 `--no-extra cloud`에서 로컬 경로 전 테스트 green |

---

## 9. Result Report / Demo에서 사용할 Evidence shortlist

### 결과보고서에 넣을 GitHub 증거 5개
| 무엇 | 위치 | 증명 항목 |
|---|---|---|
| 사용자 요청→구현 루프: Issue #57 → PR #73(closed) → **PR #80 merged(+3247줄, 2026-08-20)** | https://github.com/kapitalismho/PuriPuly-heart/pull/80 | 실용성, 커뮤니티, 발전 가능성 |
| 외부 contributor PR: **#18 Russian localization**, **#14 QQ auth(+4705줄)**, **#21/#22 bugfix** | PR 링크 4건 | 커뮤니티, 팀워크 |
| 계약서 수준 연구 이슈: **#75 Gemma 4 E4B 로컬 번역 통합**(모델·런타임·제품 규격 전체 고정) | Issue #75 | 혁신성, OSS 발전 가능성 |
| Release v2.4.0 (108 다운로드, 8/12) + 버전 동기화/패키지 검증 파이프라인 | https://github.com/kapitalismho/PuriPuly-heart/releases | 실용성, 구조/완성도 |
| 아키텍처 실증: `ARCHITECTURE.md` + tests/architecture 221 passed + PR CI 8 jobs | repo 내 | 구조/완성도, 관리체계 |

### 3분 데모에서 보여줄 증거 5개
1. 앱 시작 → **로컬 STT(Parakeet)** 선택 → 마이크 발화 → 자막(클라우드 키 없이)
2. **로컬 번역(Gemma)** 선택 → 다운로드 진행률 UI → 로컬 번역(성능 메트릭: generation_tps)
3. 번역 출력: UI + **VRChat chatbox 도달** (OSC 활성화 장면)
4. **LISTEN(peer 번역)** → 상대 음성 → 자막/오버레이
5. **OSC 양방향 제어**(뮤트 동기화, 설정 반영) — OSCQuery 발견 장면

### 발표 때 말할 프로젝트 강점 3개 (한 문장씩)
1. "VRChat에서 나와 상대의 말을 실시간 양방향 번역하며, 클라우드 API 키 하나 없이 로컬 오픈웨이트 모델(Parakeet STT + Gemma 4 E4B 번역, CPU/Vulkan)으로 같은 파이프라인이 동작합니다."
2. "양방향 OSC 제어 + OSCQuery 자동 발견과 Rust 기반 네이티브 VR 오버레이로 VRChat 내 체험을 완결합니다."
3. "사용자 요청이 3주 만에 실제 기능이 되는 루프(이슈 #57 → PR #80)와 3개국 외부 기여 6건이 머지된, 열린 개발 과정을 갖고 있습니다."

### 절대로 과장해서 말하면 안 되는 주장
- "전 세계 N명 사용 중" / "수천 명 사용자" — 사용자 수 직접 증거 없음(별점 105, 다운로드 수가 전부)
- "로컬 Gemma가 클라우드 번역보다 품질 우위" — 로컬 Gemma 품질 벤치마크 실측 공개 없음(공개 벤치마크는 cloud 모델 위주, 비교 대상이 DeepL 등)
- "완전 무료" — 기본 UX가 관리형 크레딧/상용 키 기반이며 로컬 경로는 하드웨어 요구(8코어+DDR5, VRAM 3GB)
- "보안/개인정보 완벽 보장" — SECURITY.md 부재, 브로커 보안 감사 이력 없음(로컬 저장 방침만 README에 명시)
- "대회 규정 완전 준수" — 본 감사가 H1(모델 라이선스)을 HIGH로 남겨둔 상태
