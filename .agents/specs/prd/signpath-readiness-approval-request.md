# SignPath readiness — maintainer approval request

Status: investigation in progress; not an approved policy, Foundation application, or release authorization.

## Authority and baseline

- Contract: https://github.com/kapitalismho/PuriPuly-heart/issues/74 (2026-09-12 revision).
- The maintainer subsequently requested execution, reserving sensitive documents and direct approval requests to the Director.
- Protected remote `dev`, local `dev`, and execution branch `apply-code-signing-by-signpath` started at `6395fd8d0f6b3711529bfcc051618b2911bbc923`; no pre-existing dirty paths; execution branch has no upstream.
- GitHub Project status was changed from Ready to In progress and read back. No issue criteria were marked complete.
- Implementation commencement does not authorize Foundation correspondence/submission, account or repository permission changes, production deletion/deployment, push, or release publication.

## Confirmed evidence

### Behavior at the pinned execution baseline

The canonical telemetry preference defaults to ON. Disabling it clears its anonymous identity and suppresses app-active-day requests. Enabling it again creates a new identity. The telemetry JSON payload contains only `anonymous_id` and `active_date_utc`; ordinary HTTP and network metadata are separate from that payload. Failed sends can be retried, so a daily aggregate must not be described as a hard limit of one HTTP request per day.

The Broker stores an HMAC-derived subject reference and active date in `app_active_days`, with uniqueness per subject/date. Its retention rule deletes dates strictly earlier than UTC today minus 35 days; this can retain 36 calendar dates including today. Do not silently change that behavior to match a simplified policy phrase. At baseline `6395fd8d`, the legacy `telemetry_active_days` and `telemetry_subjects` structures remained in the migration/persistence contract; current runtime aggregation did not use them. The preparation candidate adds forward migration 0025 and removes their current contract definitions. Production contents, external consumers, migration application, and deletion effects remain unverified.

The installer currently has no privacy page or telemetry control. It automatically attempts ASR model downloads for missing models; the application automatically checks GitHub for updates on launch. Neither automatic flow is disabled by the telemetry preference. Thus, “telemetry disabled” must not be described as “no network requests.”

Evidence sources: `src/puripuly_heart/core/telemetry.py`, `app/services/telemetry_reporting.py`, `config/settings_vnext/`, `broker/src/telemetry.ts`, `broker/src/scheduled.ts`, `src/puripuly_heart/core/updater.py`, and `installer.iss` at the baseline.

### Roles and signing mechanics

The live repository collaborators API lists only `kapitalismho`, with admin access. Protected `dev` has required approving review count zero and does not enforce the protection on administrators. These facts do not verify SignPath membership or MFA. The authenticated GitHub user API returned no MFA value; this is unknown, not evidence that MFA is disabled.

An isolated stock Inno Setup 6.6.1 probe produced a generated uninstaller with ProductName `PuriPuly <3` and ProductVersion `2.6.1`. Initial compilation emitted the unsigned candidate and failed without producing an installer. After external signing with an ephemeral file-only test certificate, recompilation reused it successfully. No installer was executed and no certificate store was modified. This proves local generation/reuse mechanics, not Foundation eligibility, trusted signing, installed-byte verification, timestamps, or acceptance. The stock uninstaller FileVersion remains `51.1053.0.0`; product version is the synchronized project identity. Two signing requests remain technically feasible, not yet production-proven.

Current Foundation conditions were read at https://signpath.org/terms.html. They require installation-time policy display and disabling options for functions collecting user data and transmitting it to systems not specified by the user; they do not explicitly mandate affirmative opt-in. Acceptance remains the Foundation's decision.

### Existing public release

The v2.6.1 installer at https://github.com/kapitalismho/PuriPuly-heart/releases/tag/v2.6.1 has SHA-256 `a9c23993b77e4e07aa4f6c9118617bbb3f08ba38a955123499f4a2671218803d` and is unsigned. It matches the installer in https://github.com/kapitalismho/PuriPuly-heart/actions/runs/34189959193 byte for byte. That release was built from `1b44c7705ad056e4cd5c4314337b4fda4a7bf14b`, not the current dev baseline. Supply-chain classification is still under investigation; it is not ready for approval.

## Decision A — automatic update and resource-download interpretation

The Foundation's public privacy condition does not explicitly resolve whether ordinary IP/HTTP metadata exposed by automatic update checks and model downloads requires separate installation-time disabling options. These requests contain no telemetry identifier, but still contact third parties without a fresh explicit choice. Treating them as already accepted would be unsupported; extending product settings without approval would expand the agreed scope.

Deferred, not approved or sent. The maintainer requested concise wording, then questioned the need for a preliminary inquiry. The Director proposed continuing preparation and disclosing these flows in the eventual application instead of treating a separate inquiry as a prerequisite. The short draft is retained below as history; no additional network controls or Foundation interpretation have been approved.

> I'm preparing [PuriPuly <3](https://github.com/kapitalismho/PuriPuly-heart), an open-source Windows app, for Foundation signing. Two questions:
>
> 1. The app automatically checks GitHub for updates, and the installer downloads speech models from Hugging Face/ModelScope. These requests expose IP addresses but send no audio, conversation text, or telemetry ID. Does your privacy policy require installer options to disable these requests, or is disclosure sufficient?
>
> 2. Can we sign our PyInstaller-packaged app and Inno Setup's generated uninstaller? We would leave third-party runtime binaries outside our signing scope.
>
> Separately, we plan an installer-displayed Privacy Policy and a default-on usage-statistics checkbox that users can uncheck before first launch. That setting would not disable the update checks or model downloads above.

Alternatives requiring a new explicit maintainer decision: authorize separate installer/application controls for those automatic functions, or defer this interpretation and all policy finalization pending later Foundation review. No alternative is adopted by this document.

## Decision B — approved public role responsibilities

Approved directly by the maintainer in this conversation on 2026-09-12. Approval covers the responsibilities below, not the full Code Signing Policy, account configuration, or MFA verification:

- Author: `@kapitalismho`, matching the currently verified repository administrator. Responsible for source/build changes and dependency provenance.
- Reviewer: `@kapitalismho`, responsible for reviewing contributions from non-committers, with normal repository review rules preserved. This does not claim independent-person review or create a quorum.
- Sole signing Approver: `@kapitalismho`, already approved in #74. Personally reviews each signing request and originating build. Two serial requests mean two manual approval actions by that person.
- Required GitHub and SignPath MFA must be verified before onboarding/signing. No account permission or protection change is authorized by approving these descriptions.

The full Code Signing Policy and attribution wording will be presented separately. It must not claim that the Foundation already provides this project with signing before acceptance.

## Evidence still required from account/operator access

These are factual prerequisites, not requests for secrets. Redacted configuration/account evidence is sufficient; do not paste credentials or raw user records.

- The maintainer confirmed no SignPath account or application exists. GitHub MFA remains unverified. Future account creation, roles, MFA, application contents, and submission require their applicable checks and approvals.
- Cloudflare Broker request logging/observability, Logpush/security/analytics retention, including treatment of telemetry request IPs and headers.
- Production D1 migration/retention execution and aggregate legacy-table counts; any external consumer of those tables. Production deletion requires a later separate approval.
- Managed OpenRouter privacy/training/prompt-logging configuration and routing behavior; no zero-retention promise is established by the repository.
- Readers and retention of the PuriPuly-operated Discord reporting/alert channels; QQ bot operation and retention not covered by the Broker alone.

## Reserved later approvals

Complete supply-chain classification; full Privacy Policy and all five localized policy texts; non-Korean installer translations; full Code Signing Policy; Foundation application contents and submission; account/integration configuration; any altered signing-stage count; production legacy deletion/deployment; release publication.

No policy document is finalized, no application is submitted, and no production deletion or release is authorized by this draft. Required actual five-language installer and real signed-candidate validation remain outstanding.

## Investigation correction

The initial supply-chain report classified the Flet-bundled `libmpv-2.dll` as LGPL using a build recipe. Implementation research found that the historical release-era recipe instead enables GPL/nonfree FFmpeg options and lacks the cited mpv LGPL build option. The original LGPL classification is not accepted. Exact historical provenance and redistribution obligations are being reconciled before any libmpv notice or dependency change. Independent soxr source-bundle and zeroconf notice repairs may proceed; no library replacement is approved.

Subsequent byte inspection refined that correction: the exact shipped libmpv binary embeds `-Dgpl=false`, FFmpeg `--disable-gpl --disable-nonfree --enable-version3`, and LGPLv3-or-later library license strings. It must not be called nonredistributable based on a historical recipe that does not match its build flags. The exact corresponding-source/build-recipe chain is still unresolved; embedded flags are not proof of complete source provenance.

## Decision C — ASIO-only binary exclusion

Approved directly by the maintainer: “그럼 제외”. Exclude sounddevice's ASIO-specific `libportaudio64bit-asio.dll` from the Windows package while retaining its ordinary PortAudio DLL. PuriPuly does not enable ASIO or expose it in its host-API selector. The upstream `SD_ENABLE_ASIO` environment-variable override would select the missing DLL and can fail; this compatibility limitation was explained before approval. Do not silently suppress that environment variable or alter the ordinary audio paths. Verify actual packaged contents and default audio loading. This approval does not authorize removal or replacement of libmpv or other unrelated runtime components.

## Local preparation checkpoint evidence

These checks establish local code readiness only, not Foundation acceptance or a signed release:

- Forward migration `0025_drop_legacy_translation_telemetry.sql`: populated legacy tables were removed while current `app_active_days` rows and DAU/WAU/MAU results were preserved. Initial Broker suite: 68 files, 616 tests passed; TypeScript check passed. No production operation ran. At candidate `cbbda5c3`, the deploy workflow would apply 0025 on the next ordinary deployment; review rejected that procedural-only boundary. The repair excludes 0025 from ordinary pre/post migration staging and requires both the dedicated deletion option and the exact phrase `delete retired translation telemetry from production D1`. Only after deployment, health/finalization/smoke checks and the same-run backup prerequisite does a dedicated directory apply 0025. Real extracted workflow-shell tests cover missing/default/wrong confirmation, ordinary staging, missing backup, and the approved path. Repaired Broker suite: 69 files, 633 passed, 1 skipped; TypeScript passed. YAML and all 28 shell run blocks parsed; actionlint was unavailable. These are local checks, not a production deletion approval or execution.
- Both real native release EXEs and the real PyInstaller main EXE were built with ProductName `PuriPuly <3`, ProductVersion `2.6.1` (numeric `2.6.1.0`). The generated-uninstaller metadata/reuse proof remains the isolated Inno 6.6.1 probe, not an installed signed uninstaller.
- The pre-review `build.spec` packaging retained the checked upstream license payloads and excluded the ASIO-only DLL. An isolated frozen sounddevice probe loaded ordinary PortAudio and enumerated MME, DirectSound, WASAPI, and WDM-KS without loading ASIO. Subsequent review repairs removed comments and unused metadata-reader code without changing these packaging rules or generated version-resource bytes.
- The LGPL companion ZIP includes the two soxr source archives, the applied python-soxr patch, zeroconf source, and a provenance manifest. SHA-256: `14346f3534be424ce44e2919800dcb04d74794a623b666b5b44961beae7325cb`. Archive hashes and patch reconstruction passed.
- Director integration: `pytest tests/release_evidence tests/app/test_release_dependency_guards.py tests/core/test_soxr_runtime.py tests/config/test_public_compatibility_surfaces.py -q` passed. Ruff and Black checks on the new metadata helper passed.
- Director compiled the complete pre-review installer with Inno Setup 6.6.1, then extracted it without installation using innounp 2.71.1. All 1,137 expected payload files matched that dist/staged input set byte for byte; the ASIO-only DLL was absent and ordinary PortAudio present. The embedded companion ZIP matched the hash above.
- Integrated unsigned installer: `installer_output/PuriPulyHeart-Setup-2.6.1.exe`, 173,766,150 bytes, SHA-256 `0396a9182ff1171b54d76fe818c964d2a3c75338d0717f46c6e2923892b1f408`. This is a local test artifact, not a release candidate approved for publication.
- The release-provenance helper accepted the existing real local artifacts and verified all 26 exact license payloads, source archive integrity, metadata and asset hashes. Negative probes rejected mismatched title/tag/body, altered payload/archive content, and inconsistent local-versus-hosted identity. Local artifact provenance is not GitHub or SignPath trusted-origin proof and does not establish a rebuild from the repaired candidate.
- Post-repair Director integration passed the release-evidence, dependency-guard, process-capture, managed-Gemma and public-compatibility Python suites; the focused Broker migration/persistence/deployment suite passed all 40 tests and TypeScript passed. The actual provenance command interface was exercised. No hosted workflow, deployment or publication ran.

The current checkpoint does not implement installer privacy UI, localized policy text, SignPath workflow requests, signature verification, or a signed release. Full policy approval, infrastructure evidence, libmpv corresponding-source resolution, account/MFA setup, production deletion approval, and end-to-end signed validation remain separate unmet criteria.

## Checkpoint review adjudication

Review range: `6395fd8d0f6b3711529bfcc051618b2911bbc923..cbbda5c37b6b780b8533ae7ed4f481e8e6eca48f`. Independent general and migration-safety reviewers examined the integrated candidate.

- ACCEPT general F1 / migration F1: separate 0025 from ordinary deployment with explicit default-safe deletion confirmation; merely moving it to an automatically applied deferred stage would not satisfy this decision.
- ACCEPT general F2 / migration F2: update the Broker runbook's baseline-versus-migrated schema and operator approval instructions.
- ACCEPT general F3: remove unused PE-reading/verification helpers instead of retaining unconsumed future workflow code.
- ACCEPT general F4: remove newly added explanatory code comments; no comment permission was granted.
- ACCEPT migration F3: clarify that retained legacy structures describe the pinned baseline, not the preparation candidate.
- General G1: retain the executed isolated-uninstaller probe evidence with its stated limits. Throwaway artifacts were intentionally removed; no installed or Foundation-signed uninstaller verification is claimed.
- General G2: a single focused package/source-bundle guard is now called by both the local build script and the official release workflow. The workflow also checks project/installer PE identity, records source/run/artifact provenance, and validates the rendered release surface before draft release creation. This closes the missing code path, subject to independent review; a successful hosted run and live signing integration remain unverified.
- General G3: retain the explicit SignPath metadata/account uncertainty. Stock uninstaller FileVersion is not claimed to be the project's product version.

The maintainer requested that nonessential external inquiries be deferred. Both the preliminary SignPath inquiry and the proposed media-kit source inquiry remain unposted. Libmpv's matching historical build recipe was subsequently located, including FFmpeg patches; exact revisions for several unpinned dependencies remain unavailable. Do not replace those gaps with a guessed source bundle or a blanket compliance claim.

## Integrated checkpoint repair record

The fresh integrated review covered `6395fd8d..2685a1d0`; the retained migration reviewer separately verified the deletion-gate repair at `2685a1d0`.

- Migration result: `repair_verified`; ordinary deployment exclusion, dedicated confirmation, backup prerequisite, step ordering, unchanged SQL and corrected operator documentation were verified. Production remains untouched.
- ACCEPT integrated F1: delete unused `normalize_tag_version`; the live tag/version checks remain unchanged. Existing 13 release-identity behavior tests, Ruff, Black and the real CLI help command passed after removal.
- ACCEPT integrated F2: describe the zeroconf sdist as supplied corresponding source, not the input used by this project to build the upstream wheel's native modules.
- The corrected notice was staged into the existing dist and the real Inno installer rebuilt. All 1,137 extracted application payloads matched the staged dist/overlay, including the corrected notice; ordinary PortAudio remained present and ASIO absent. This is a notice-only restaging of the previously compiled local binaries, not a full rebuild from the repair commit.
- Restaged unsigned installer: 173,764,990 bytes, SHA-256 `ae64ccc38fdb586d35373d2e506a9e5c26ca5847d129bc9db0c2b38e164bbcef`. This supersedes the earlier local installer file; the earlier hash records that earlier check only. The repair probe's extraction directory was removed.

Independent repair verification returned `repair_verified` for `2685a1d0..8f466d23bbb26ef6c1e7e277bb891902906019ea`. The reviewer independently re-extracted the restaged installer, matched all 1,137 payloads and the corrected notice, reran the 26-payload/source-bundle guard and 13 release-identity tests, and found no directly related regression. The Director accepts this local preparation checkpoint, not completion of issue #74 or permission to publish. The obsolete Director-owned historical extraction was removed after review; the unsigned installer, dist, source bundle and upstream audit evidence remain available. No change to the project's runtime architecture was identified by this preparation; the release checks and explicit migration-approval boundary are the intended changes.

## Privacy drafting outcome — resumed

Baseline: `401b659469c6d8a14edba005322024b1d482ebe1`, branch `apply-code-signing-by-signpath`, no upstream and no pre-existing dirty paths. GitHub issue #74 remains open and its Project status is `In progress`.

The maintainer authorized the proportionate sequence discussed in this conversation: reuse the existing data-flow evidence, check only material operator-controlled settings, prepare policy wording for direct approval, then implement and verify the already-approved installer UX and translations. Application submission is last. Provider-internal forensic audits, complete upstream rebuild reconstruction and a separate preliminary inquiry are not new prerequisites. Existing approved signing coverage, five-language installer support, approval gates and legacy-removal direction remain unchanged; no production deletion or remote publication is authorized.

The Director owns policy wording and approval requests. `privacy-draft-facts` and `privacy-operator-access` are read-only research owners. They provide factual citations and bounded access results, not policy approval. The next boundary is a reviewed policy draft with specific unresolved operator questions; it is not an accepted installer implementation or a published Privacy Policy.

### Bounded evidence collected

- `privacy-draft-facts` compared the last canonical Broker deployment source `07a06450151cf2890a3a77bb0daa6b17b4a72c50` with this outcome's baseline. App-active-day ingestion, aggregation and retention are unchanged; `telemetry.ts`, `scheduled.ts` and `app.ts` are identical. This is source/deploy-record evidence, not a live database inspection.
- `privacy-operator-access` ran one read-only `wrangler whoami`: not authenticated. It did not initiate login. GitHub run `33870595961` records the last canonical deployment and version `46089a96-72f9-4b93-8ad7-5852d0691a33`; dashboard changes after that deployment remain unknown.
- No observability/Logpush setting is declared in the repository. This does not establish that account-level logs are disabled. Cloudflare account settings and D1 runtime retention overrides were not readable here.
- The recorded OpenRouter guardrail uses `enforce_zdr=false`. The draft makes no zero-retention or no-training promise and does not need a provider-internal audit to avoid such promises.
- App-active-day cleanup uses a strict date cutoff: dates before UTC today minus 35 days are cleanup targets, leaving 36 dates including today after cleanup. Request/issuance/audit retention can depend on live D1 settings; installation lifecycle constants are not proof of an executing deletion job. Legacy telemetry production removal is not claimed.
- Correction to the earlier proportionality research: a model download failure can continue installation (`installer.iss:993-1003`); insufficient disk space has a separate blocking path. The draft therefore says missing models trigger downloads, not that every download failure prevents installation.
- Provider-policy URLs below were fetched successfully by the fact researcher. They are linked as the providers' policies, not evidence of particular account-level retention or training settings.

### 개인정보처리방침 — 한국어 검토안

**승인·공개 전 초안입니다.** 아래 설치 화면 설명은 승인 후 구현할 버전에 대한 문구입니다. 현재 배포된 설치 프로그램에 해당 기능이 있다는 뜻이 아닙니다. 운영 보관 설정에 관한 아래 확인 사항을 해결하고 설치 동작을 검증한 뒤 공개합니다. 기존 한국어 짧은 안내 문구와 기본 ON 결정은 변경하지 않습니다.

#### 1. 적용 범위

이 방침은 PuriPuly 앱과 프로젝트가 운영하는 인증·사용 통계 서버의 정보 처리에 관한 설명입니다. 선택한 외부 음성 인식·번역 서비스, 계정 서비스와 다운로드 서비스에는 각 서비스의 정책도 적용됩니다.

#### 2. 사용 통계

PuriPuly는 사용자 수를 추정하기 위해 무작위 식별자와 앱 사용 날짜(UTC)를 서버에 보냅니다. 사용 통계 항목에는 대화 내용, 음성, 번역 결과, 언어, 제공자·모델 선택, API 키가 포함되지 않습니다.

서버는 식별자를 비밀키로 변환한 참조값과 사용 날짜를 저장합니다. 동일한 참조값과 날짜는 하나의 활동 기록으로 집계하며, 계정·장치 인증 정보와 연결하지 않습니다. 집계는 실제 사람 수가 아니라 서로 다른 익명 식별자의 활동을 기준으로 한 추정치입니다. 전송 실패나 앱 재시작으로 같은 날짜에 요청을 다시 보낼 수 있습니다.

신규 설치에서는 사용 통계가 기본으로 켜져 있습니다. 설치 중 표시되는 개인정보처리방침 페이지에서 ‘익명 사용 통계 보내기’를 해제하면 첫 실행부터 통계 요청을 보내지 않습니다. 설치 후에도 설정에서 변경할 수 있으며, 업그레이드는 사용자가 바꾸지 않는 한 기존 선택을 유지합니다. 이 설정은 선택 가능한 기능 설정이며, 기본으로 체크되어 있다는 사실을 별도의 동의 표시로 취급하지 않습니다.

통계를 끄면 이후 통계 요청을 중단하고 기기의 통계 식별자를 제거합니다. 다시 켜면 새 식별자를 만듭니다. 이 동작이 이미 서버에 저장된 활동 기록을 즉시 삭제하는 것은 아닙니다. 정기 정리에서는 사용 날짜가 UTC 기준 오늘보다 35일을 초과하여 오래된 기록을 삭제 대상으로 삼습니다. 정리 후에는 오늘을 포함한 36개 날짜의 기록이 남을 수 있습니다.

사용 통계를 꺼도 업데이트 확인, 모델 다운로드, 사용자가 이용하는 인증·음성 인식·번역 기능의 통신은 별도로 발생합니다.

#### 3. 음성 인식·번역과 VRChat 연동

로컬 음성 인식·번역 모델은 기기에서 처리합니다. 외부 음성 인식 서비스를 선택하면 처리할 음성을, 외부 번역 서비스를 선택하면 원문과 번역에 필요한 문맥을 선택한 서비스로 보냅니다. 직접 등록한 API 키는 해당 서비스 인증에 사용됩니다.

VRChat 연동은 기기의 로그에서 참가자 수를 파악하여 번역 요청의 문맥에 포함할 수 있습니다. 이 연동에서 읽은 참가자 이름과 원본 로그 자체는 전송하지 않습니다. 사용자 지정 HTTP 번역 확장에는 이 참가자 수 정보도 전달하지 않습니다.

관리형 OpenRouter 번역은 OpenRouter와 실제 추론 제공자가 처리하며, 요청에 관리형 사용자 참조값이 포함됩니다. 프로젝트의 인증 서버는 통상적인 번역 본문을 중계하는 경로가 아닙니다. 외부 서비스의 내용 보관·학습 이용은 해당 서비스의 정책과 설정에 따라 달라지며, PuriPuly는 모든 요청에 대해 무보관이나 학습 미사용을 보장하지 않습니다.

외부 처리를 피하려면 로컬 제공자를 사용하거나 해당 음성 인식·번역 기능을 끌 수 있습니다. 로컬 모델 사용도 별도의 모델 다운로드나 업데이트 확인까지 없애는 것은 아닙니다. 사용자 지정 음성 인식 주소나 HTTP 확장을 사용하면 설정한 주소로 처리할 음성 또는 텍스트와 설정된 인증정보가 전송되므로 해당 서버의 정책을 확인해야 합니다.

OSC 기능은 설정된 대상에 자막과 제어 정보를 전달합니다. 기본 대상은 기기 내부이며, 자동 연결 모드는 로컬 네트워크에서 서비스 검색·광고를 수행합니다. 설정에서 연결 모드를 변경하거나 끌 수 있습니다.

#### 4. 관리형 계정·이용권한과 보안

관리형 서비스를 이용하면 인증, 이용권한·사용량 확인, 키 전달과 부정 이용 방지를 위해 프로젝트 서버와 통신합니다. 이 과정에서 설치 식별자, 장치 공개키, 하드웨어 정보의 해시, 앱 버전, 인증 서비스의 계정 정보나 인증값을 처리할 수 있습니다.

프로젝트 서버에는 변환된 계정 참조값, 설치·장치 참조정보, 이용권한·키 전달 상태와 보상 기록 등이 저장됩니다. QQ 인증값과 OpenRouter API 키의 원문을 이 데이터베이스에 저장하지 않습니다. 인증·발급 등 보안 대상 요청에는 비밀키로 변환한 IP 참조값, 요청 시각, 국가·네트워크 및 연결 특성 등의 기록을 사용합니다. 이는 사용 통계 테이블에 저장하는 항목과 구분됩니다.

관리형 계정이 설정되어 있으면 앱 시작 시 이용권한·사용량을 확인할 수 있습니다. 관리형 인증은 로컬 기능이나 본인 API 키를 사용하는 기능과 별개입니다. 인증·발급·보안 기록에는 운영 설정과 만료 상태에 따른 정리 규칙이 적용되며, 계정·이용권한·지급된 보상 기록 전체가 동일한 기간 후 자동 삭제되는 것은 아닙니다.

운영용 Discord 채널에는 일별 키 발급 수와 활동 통계 등의 집계를 전달합니다. Discord·QQ 자체의 계정 및 서비스 처리는 해당 서비스 정책을 따릅니다.

#### 5. 업데이트·모델 다운로드와 서버 기반시설

앱은 시작 시 GitHub에서 새 릴리스 정보를 자동으로 확인합니다. 설치 프로그램은 필요한 모델이 없으면 Hugging Face 또는 ModelScope에서 다운로드를 시도하며, 앱에서도 로컬 모델 준비를 위해 다운로드가 발생할 수 있습니다. 이 요청에는 사용 통계 식별자나 대화·음성을 넣지 않지만, 접속 대상은 IP 주소와 일반적인 HTTP·네트워크 정보를 처리합니다. 사용 통계 설정은 이 요청을 끄는 설정이 아닙니다.

프로젝트 서버는 Cloudflare Workers와 D1을 사용합니다. 서버 연결에 필요한 IP·요청 정보 등의 기반시설 처리는 사용 통계 요청 본문의 두 항목과 별개입니다. 데이터베이스의 활동 기록에 IP를 넣지 않는다는 설명이 Cloudflare를 포함한 모든 처리 계층에서 IP나 요청 로그가 전혀 남지 않는다는 뜻은 아닙니다.

데이터베이스 기록을 삭제하더라도 복구 이력이나 배포 시 만든 백업 사본이 즉시 함께 사라지는 것은 아닙니다. 이전 버전의 사용 통계 기록이 운영 데이터베이스에서 모두 삭제되었다고도 보장하지 않습니다.

#### 6. 기기에 남는 정보

설정, 모델 파일과 진단 로그는 주로 `%LOCALAPPDATA%\puripuly-heart`에 저장됩니다. API 키 등의 비밀정보는 Windows 자격 증명 저장소 또는 암호화된 파일 저장소를 사용합니다. 진단 로그에는 실행 상태와 오류 정보가 들어갈 수 있으므로 다른 사람에게 전달하기 전에 내용을 확인해야 합니다.

앱 제거는 앱 데이터 폴더를 정리하지만, Windows 자격 증명 저장소나 외부 서비스에 남은 정보까지 모두 삭제한다는 뜻은 아닙니다.

#### 7. 관련 서비스 정책

- 서버·업데이트: [Cloudflare](https://www.cloudflare.com/privacypolicy/), [GitHub](https://docs.github.com/en/site-policy/privacy-policies/github-general-privacy-statement).
- 인증·운영 채널: [Discord](https://discord.com/privacy), [Tencent QQ](https://privacy.tencent.com/home).
- 모델 다운로드: [Hugging Face](https://huggingface.co/privacy), [ModelScope](https://modelscope.cn/protocol/Privacy-Policy).
- 음성 인식·번역: [OpenRouter](https://openrouter.ai/privacy), [OpenRouter의 추론 제공자별 로깅 안내](https://openrouter.ai/docs/guides/privacy/provider-logging), [Google](https://policies.google.com/privacy) 및 [Gemini API 약관](https://ai.google.dev/gemini-api/terms), [Alibaba Model Studio](https://www.alibabacloud.com/help/en/model-studio/privacy-notice), [DeepSeek](https://cdn.deepseek.com/policies/en-US/deepseek-privacy-policy-2025-02-14.html), [Cerebras](https://www.cerebras.ai/privacy-policy), [Deepgram](https://deepgram.com/privacy), [ElevenLabs](https://elevenlabs.io/privacy-policy), [Soniox](https://soniox.com/policies/privacy-policy).

선택한 기능과 제공자에 해당하는 정책을 확인하세요. OpenRouter의 실제 추론 제공자와 사용자 지정 서버에는 그 제공자의 정책이 추가로 적용됩니다.

#### 8. 문의

프로젝트 운영자는 GitHub의 [kapitalismho](https://github.com/kapitalismho)입니다. 문의 경로는 [프로젝트 저장소](https://github.com/kapitalismho/PuriPuly-heart)에서 확인할 수 있습니다. 공개 이슈에 API 키, 인증값, 음성·대화 원문이나 다른 사람의 개인정보를 올리지 마세요.

### 공개 전 확인 사항 — 정책 본문과 구분

1. Cloudflare의 실제 Workers Logs/Logpush 활성화 여부와 사용 요금제, 현재 배포 버전. 이 환경에는 계정 접근이 없어 읽지 못했습니다. 설정 화면의 비밀정보 없는 확인만 필요하며 원본 요청 로그는 필요 없습니다.
2. 운영 D1의 보관 설정값과 적용된 마이그레이션 상태. 설정·마이그레이션 메타데이터만으로 보관 문구를 구체화하며, 사용자 원본 기록을 읽거나 레거시 데이터를 삭제하지 않습니다. 일반적인 자동 삭제를 보장하는 문구는 사용하지 않습니다.
3. 위 사실을 반영한 한국어 본문에 대한 직접 승인. 이후 영어·일본어·중국어 간체·번체 정책과 UI 번역을 제시하여 승인받고 설치 화면을 구현합니다. 다른 언어 번역을 한국어 의미 검토보다 앞서 확정하지 않습니다.

OpenRouter의 무보관·학습 미사용, Discord·QQ 내부의 고정 보관 기간을 주장하지 않으므로 공급자 내부 감사는 이 초안의 선행 조건이 아닙니다. 자동 업데이트·모델 다운로드에 대한 Foundation 해석과 실제 서명 대상 수락은 신청 자료에서 사실대로 공개할 사항이며, 새 기능이나 이미 받은 수락으로 취급하지 않습니다.
