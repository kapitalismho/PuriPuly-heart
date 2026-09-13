# SignPath readiness — maintainer approval request

Status: preapplication preparation only under the maintainer's latest instruction recorded below. Policy translations and current-user telemetry lifecycle checks are complete at their recorded candidates; elevated-install/uninstall and per-language runtime checks are skipped, not passed. Actual signing and signed-candidate validation belong after acceptance, not to preapplication completion. Code signing policy final approval and corresponding-source evidence remain pending. No public policy publication, Foundation application, or release is authorized by this record.

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

### 개인정보처리방침 — 한국어 승인본

**한국어 본문 승인 완료, 공개·설치 적용 전입니다.** 유지관리자는 문의 절의 공개 이슈 주의 문장 하나만 삭제하는 조건으로 아래 본문을 승인했습니다. 설치 화면 설명은 구현할 버전에 대한 문구이며, 현재 배포된 설치 프로그램에 해당 기능이 있다는 뜻이 아닙니다. 번역문 승인과 설치 동작 검증은 별도입니다. 기존 한국어 짧은 안내 문구와 기본 ON 결정은 변경하지 않습니다.

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

프로젝트 운영자는 GitHub의 [kapitalismho](https://github.com/kapitalismho)입니다. 문의 경로는 [프로젝트 저장소](https://github.com/kapitalismho/PuriPuly-heart)에서 확인할 수 있습니다.

### 공개 전 확인 사항 — 정책 본문과 구분

1. 유지관리자 확인: Workers Logs와 Traces는 모두 꺼져 있습니다. 이 진술을 운영 사실로 반영하며 재확인하지 않습니다. 별도의 Logpush 작업, 요금제, 현재 배포 버전까지 확인한 것으로 확대하지 않습니다. 승인된 본문은 모든 기반시설의 무기록을 주장하지 않습니다.
2. 운영 D1의 보관 설정값과 적용된 마이그레이션 상태. 설정·마이그레이션 메타데이터만으로 보관 문구를 구체화하며, 사용자 원본 기록을 읽거나 레거시 데이터를 삭제하지 않습니다. 일반적인 자동 삭제를 보장하는 문구는 사용하지 않습니다.
3. 한국어 본문은 문의 절의 공개 이슈 주의 문장만 삭제하여 직접 승인받았습니다. 영어·일본어·중국어 간체·번체 정책과 UI 번역은 별도로 승인받은 뒤 설치 화면에 적용합니다.

OpenRouter의 무보관·학습 미사용, Discord·QQ 내부의 고정 보관 기간을 주장하지 않으므로 공급자 내부 감사는 이 초안의 선행 조건이 아닙니다. 자동 업데이트·모델 다운로드에 대한 Foundation 해석과 실제 서명 대상 수락은 신청 자료에서 사실대로 공개할 사항이며, 새 기능이나 이미 받은 수락으로 취급하지 않습니다.

### Draft checkpoint disposition

Independent `FULL_REVIEW` by `privacy-draft-review` covered `401b659469c6d8a14edba005322024b1d482ebe1..417694beae0e87acb59b23f047ecf822361a5225` and returned `no_material_findings`. The reviewer checked all material data flows, current-versus-proposed installer wording, retention limits, local storage/uninstall qualifications, the provider list and the bounded operator questions. All 16 linked service-policy URLs responded successfully; no provider-specific internal retention claims were inferred from that check.

The Director accepts the draft as ready for maintainer review, not as an approved/public policy or completed privacy Outcome. Operator configuration evidence and direct wording approval remain outstanding. No runtime code, installer UI, account configuration, remote publication, production records or architecture was changed in this outcome. No throwaway scripts or new tests were introduced.

### Maintainer approval and language-draft boundary

The maintainer approved the Korean policy with exactly one deletion: the public-issue warning sentence in section 8. All other Korean body wording is unchanged. The maintainer also reported that Workers Logs and Traces are both disabled. This is accepted operator evidence, not a request to repeat the check and not evidence about separate Logpush jobs, backups, D1 configuration or Foundation acceptance.

Language preparation starts from `b3d3f30f7fe0f9591ebfb9fbff44b9daadcaee24` on `apply-code-signing-by-signpath`, with no upstream and no pre-existing dirty paths. The Director owns the policy/UI translations below. They must preserve the approved Korean semantics, omit the deleted warning, and await direct approval. Read-only language review checks the complete four-language candidate before presentation. No installer implementation, publication or production operation is authorized by language-draft preparation.

### Installer copy — four-language approval draft

These are proposed translations of the approved Korean disclosure, not new collection or consent rules. The checkbox is checked on a fresh installation; `[x]` is not part of its label. Existing Inno navigation buttons retain their built-in localized labels. Full policies below are offline page content; external service links are references, not substitutes for displaying that content.

| Element | English | 日本語 | 简体中文 | 繁體中文 |
| --- | --- | --- | --- | --- |
| Policy page title | Privacy Policy | プライバシーポリシー | 隐私政策 | 隱私權政策 |
| Disclosure title | Usage statistics | 利用統計 | 使用统计 | 使用統計 |
| Purpose | PuriPuly collects minimal anonymous data to count users. | PuriPulyはユーザー数を集計するために、最小限の匿名データを収集します。 | PuriPuly收集最少量的匿名数据，用于统计用户数量。 | PuriPuly收集最少量的匿名資料，用於統計使用者人數。 |
| Fields | The statistics contain only a random identifier and the date the app was used (UTC). | 統計に含まれるのは、ランダムな識別子とアプリの利用日（UTC）のみです。 | 统计仅包含随机标识符和应用使用日期（UTC）。 | 統計僅包含隨機識別碼及應用程式使用日期（UTC）。 |
| Exclusions | They do not contain sensitive information such as conversation content or audio. | 会話の内容や音声などの機微な情報は含まれません。 | 不包含对话内容、音频等敏感信息。 | 不包含對話內容、音訊等敏感資訊。 |
| Checkbox label | Send anonymous usage statistics | 匿名の利用統計を送信する | 发送匿名使用统计 | 傳送匿名使用統計 |
| Settings guidance | You can change this at any time in Settings after installation. | インストール後も、設定からいつでも変更できます。 | 安装后也可随时在设置中更改。 | 安裝後也可隨時在設定中變更。 |

### Privacy Policy — English approval draft

**Translation awaiting approval.** This translates the approved Korean body after the section 8 deletion. Installation-related statements describe the version to be implemented and verified, not the current public installer. This note is review metadata, not part of the policy page.

#### 1. Scope

This policy explains how information is handled by the PuriPuly app and the project's authentication and usage-statistics servers. The policies of the external speech recognition, translation, account and download services you select also apply.

#### 2. Usage statistics

PuriPuly sends a random identifier and the date the app was used (UTC) to its server to estimate the number of users. Usage statistics do not include conversation content, audio, translation results, language, provider or model selections, or API keys.

The server stores a reference derived from the identifier using a secret key, together with the usage date. The same reference and date count as one activity record and are not linked to account or device authentication information. These counts are estimates based on activity from distinct anonymous identifiers, not a count of verified individual people. Requests may be retried on the same date after a transmission failure or an app restart.

Usage statistics are enabled by default on new installations. Unchecking “Send anonymous usage statistics” on the Privacy Policy page during installation prevents statistics requests from the first launch onward. You can also change this in Settings after installation. Upgrades preserve your existing choice unless you change it. This is an optional feature setting; a preselected checkbox is not treated as a separate indication of consent.

Turning statistics off stops subsequent statistics requests and removes the statistics identifier from your device. Turning them on again creates a new identifier. This does not immediately delete activity records already stored on the server. Periodic cleanup targets records whose usage date is more than 35 days before the current UTC date. Records for 36 dates, including today, may remain after cleanup.

Turning usage statistics off does not stop separate communications for update checks, model downloads, or the authentication, speech recognition and translation features you use.

#### 3. Speech recognition, translation and VRChat integration

Local speech recognition and translation models process data on your device. Selecting an external speech recognition service sends the audio to be processed to that service. Selecting an external translation service sends the source text and context needed for translation to that service. API keys you supply are used to authenticate with the corresponding service.

VRChat integration can obtain the participant count from logs on your device and include it as context in translation requests. It does not transmit the participant names or original logs read by this integration. Custom HTTP translation extensions do not receive this participant count either.

Managed OpenRouter translation is processed by OpenRouter and the actual inference provider, and requests include a managed user reference. The project's authentication server is not the route through which ordinary translation content is relayed. External services' retention and use of content for training depend on their policies and settings. PuriPuly does not guarantee zero retention or exclusion from training for every request.

To avoid external processing, you can use a local provider or turn off the relevant speech recognition or translation feature. Using local models does not eliminate separate model downloads or update checks. If you use a custom speech recognition address or HTTP extension, the audio or text to be processed and configured authentication information are sent to the configured address, so you should check that server's policy.

OSC sends subtitles and control information to the configured destination. The default destination is on your device, and automatic connection mode discovers and advertises services on the local network. You can change the connection mode or turn it off in Settings.

#### 4. Managed accounts, entitlements and security

Using managed services involves communication with the project's server for authentication, entitlement and usage checks, key delivery and abuse prevention. This may involve processing installation identifiers, device public keys, hashes of hardware information, the app version, and account information or authentication values from authentication services.

The project's server stores information such as derived account references, installation and device references, entitlement and key-delivery status, and reward records. It does not store the original QQ authentication values or OpenRouter API keys in this database. Requests subject to security checks, such as authentication and issuance requests, use records including IP references derived using a secret key, request times, country, network and connection characteristics. These are separate from the fields stored in the usage-statistics table.

If a managed account is configured, the app may check entitlements and usage at startup. Managed authentication is separate from local features and features that use your own API keys. Authentication, issuance and security records are subject to cleanup rules based on operational settings and expiration status. Not all account, entitlement and credited-reward records are automatically deleted after the same period.

Aggregates such as daily key-issuance counts and activity statistics are sent to an operational Discord channel. Discord's and QQ's own account and service processing is governed by their respective policies.

#### 5. Updates, model downloads and server infrastructure

The app automatically checks GitHub for new release information at startup. The installer attempts to download required models from Hugging Face or ModelScope if they are missing, and the app may also download files to prepare local models. These requests do not include the usage-statistics identifier, conversations or audio, but the destination processes IP addresses and ordinary HTTP and network information. The usage-statistics setting does not disable these requests.

The project's server uses Cloudflare Workers and D1. Infrastructure processing of information such as IP addresses and requests needed for server connections is separate from the two fields in the usage-statistics request body. Saying that IP addresses are not added to activity records in the database does not mean that no IP or request logs remain at any processing layer, including Cloudflare.

Deleting database records does not necessarily remove recovery history or backup copies created during deployment at the same time. We also do not guarantee that all usage-statistics records from earlier versions have been deleted from the operational database.

#### 6. Information stored on your device

Settings, model files and diagnostic logs are mainly stored in `%LOCALAPPDATA%\puripuly-heart`. Secrets such as API keys use Windows Credential Manager or an encrypted file store. Diagnostic logs may contain runtime status and error information, so check their contents before sharing them with others.

Uninstalling the app cleans up the app data folder, but this does not mean that all information in Windows Credential Manager or external services is also deleted.

#### 7. Related service policies

- Server and updates: [Cloudflare](https://www.cloudflare.com/privacypolicy/), [GitHub](https://docs.github.com/en/site-policy/privacy-policies/github-general-privacy-statement).
- Authentication and operational channels: [Discord](https://discord.com/privacy), [Tencent QQ](https://privacy.tencent.com/home).
- Model downloads: [Hugging Face](https://huggingface.co/privacy), [ModelScope](https://modelscope.cn/protocol/Privacy-Policy).
- Speech recognition and translation: [OpenRouter](https://openrouter.ai/privacy), [OpenRouter's inference-provider logging guidance](https://openrouter.ai/docs/guides/privacy/provider-logging), [Google](https://policies.google.com/privacy) and [Gemini API terms](https://ai.google.dev/gemini-api/terms), [Alibaba Model Studio](https://www.alibabacloud.com/help/en/model-studio/privacy-notice), [DeepSeek](https://cdn.deepseek.com/policies/en-US/deepseek-privacy-policy-2025-02-14.html), [Cerebras](https://www.cerebras.ai/privacy-policy), [Deepgram](https://deepgram.com/privacy), [ElevenLabs](https://elevenlabs.io/privacy-policy), [Soniox](https://soniox.com/policies/privacy-policy).

Check the policies relevant to the features and providers you select. The policies of OpenRouter's actual inference providers and the providers of custom servers also apply.

#### 8. Contact

The project is operated by [kapitalismho](https://github.com/kapitalismho) on GitHub. Contact information is available in the [project repository](https://github.com/kapitalismho/PuriPuly-heart).

### プライバシーポリシー — 日本語承認用案

**翻訳は承認待ちです。** 第8節の指定文を削除した承認済み韓国語本文の翻訳です。インストールに関する記述は、今後実装・検証するバージョンについてのもので、現在公開中のインストーラーの説明ではありません。この注記はレビュー用であり、ポリシーページの本文には含めません。

#### 1. 適用範囲

このポリシーは、PuriPulyアプリとプロジェクトが運営する認証・利用統計サーバーでの情報の取り扱いを説明します。選択した外部の音声認識・翻訳サービス、アカウントサービス、ダウンロードサービスには、それぞれのサービスのポリシーも適用されます。

#### 2. 利用統計

PuriPulyはユーザー数を推定するために、ランダムな識別子とアプリの利用日（UTC）をサーバーに送信します。利用統計の項目には、会話の内容、音声、翻訳結果、言語、プロバイダー・モデルの選択、APIキーは含まれません。

サーバーは、識別子を秘密鍵で変換した参照値と利用日を保存します。同じ参照値と日付は1件の活動記録として集計し、アカウント・デバイスの認証情報とは関連付けません。集計は、異なる匿名識別子の活動に基づく推定値であり、実際の個人を確認して数えたものではありません。送信の失敗やアプリの再起動により、同じ日にリクエストを再送することがあります。

新規インストールでは、利用統計はデフォルトで有効です。インストール中に表示されるプライバシーポリシーのページで「匿名の利用統計を送信する」のチェックを外すと、初回起動から統計リクエストを送信しません。インストール後も設定で変更でき、アップグレードではユーザーが変更しない限り既存の選択を維持します。これは任意の機能設定であり、あらかじめチェックされていることを別個の同意表示として扱いません。

統計を無効にすると、その後の統計リクエストを停止し、デバイスから統計用識別子を削除します。再び有効にすると、新しい識別子を作成します。これによって、サーバーに保存済みの活動記録が直ちに削除されるわけではありません。定期的な整理では、利用日がUTCの当日から35日を超えて古い記録を削除対象とします。整理後も、当日を含む36日分の日付の記録が残ることがあります。

利用統計を無効にしても、更新の確認、モデルのダウンロード、利用する認証・音声認識・翻訳機能の通信は別途発生します。

#### 3. 音声認識・翻訳とVRChat連携

ローカルの音声認識・翻訳モデルはデバイス上で処理します。外部の音声認識サービスを選択すると処理対象の音声を、外部の翻訳サービスを選択すると原文と翻訳に必要な文脈を、選択したサービスへ送信します。自分で登録したAPIキーは、該当するサービスの認証に使用されます。

VRChat連携では、デバイス上のログから参加者数を取得し、翻訳リクエストの文脈に含めることがあります。この連携で読み取った参加者名や元のログ自体は送信しません。カスタムHTTP翻訳拡張には、この参加者数の情報も渡しません。

管理型OpenRouter翻訳はOpenRouterと実際の推論プロバイダーが処理し、リクエストには管理型ユーザー参照値が含まれます。プロジェクトの認証サーバーは、通常の翻訳本文を中継する経路ではありません。外部サービスによる内容の保存や学習への利用は、各サービスのポリシーと設定によって異なり、PuriPulyはすべてのリクエストについて無保存や学習への不使用を保証しません。

外部での処理を避けるには、ローカルのプロバイダーを使用するか、該当する音声認識・翻訳機能を無効にできます。ローカルモデルを使用しても、別途行われるモデルのダウンロードや更新の確認までなくなるわけではありません。カスタム音声認識アドレスやHTTP拡張を使用する場合、処理対象の音声またはテキストと設定した認証情報が指定先へ送信されるため、そのサーバーのポリシーを確認してください。

OSC機能は、設定した送信先に字幕と制御情報を送ります。デフォルトの送信先はデバイス内であり、自動接続モードではローカルネットワークでサービスの検索・広告を行います。設定で接続モードを変更したり、無効にしたりできます。

#### 4. 管理型アカウント・利用権限とセキュリティ

管理型サービスを利用すると、認証、利用権限・使用量の確認、キーの配信、不正利用の防止のためにプロジェクトのサーバーと通信します。この過程で、インストール識別子、デバイスの公開鍵、ハードウェア情報のハッシュ、アプリのバージョン、認証サービスのアカウント情報や認証値を処理することがあります。

プロジェクトのサーバーには、変換したアカウント参照値、インストール・デバイスの参照情報、利用権限・キー配信の状態、報酬記録などを保存します。このデータベースには、QQの認証値やOpenRouter APIキーの元の値を保存しません。認証・発行などのセキュリティ確認対象リクエストには、秘密鍵で変換したIP参照値、リクエスト時刻、国・ネットワーク・接続の特性などの記録を使用します。これは利用統計テーブルに保存する項目とは別です。

管理型アカウントを設定している場合、アプリ起動時に利用権限・使用量を確認することがあります。管理型認証は、ローカル機能や自分のAPIキーを使用する機能とは別です。認証・発行・セキュリティ記録には、運用設定や有効期限の状態に応じた整理ルールが適用されます。アカウント・利用権限・付与済み報酬の記録がすべて同じ期間の後に自動削除されるわけではありません。

運用用Discordチャンネルには、日別のキー発行数や活動統計などの集計を送信します。Discord・QQ自体のアカウントやサービスの処理には、各サービスのポリシーが適用されます。

#### 5. 更新・モデルのダウンロードとサーバー基盤

アプリは起動時にGitHubで新しいリリース情報を自動確認します。インストーラーは必要なモデルがない場合、Hugging FaceまたはModelScopeからのダウンロードを試みます。アプリでも、ローカルモデルの準備のためにダウンロードが発生することがあります。これらのリクエストには利用統計の識別子や会話・音声を含めませんが、接続先はIPアドレスや一般的なHTTP・ネットワーク情報を処理します。利用統計の設定では、これらのリクエストは無効になりません。

プロジェクトのサーバーはCloudflare WorkersとD1を使用します。サーバー接続に必要なIP・リクエスト情報などの基盤側の処理は、利用統計リクエスト本文の2項目とは別です。データベースの活動記録にIPを加えないという説明は、Cloudflareを含むすべての処理層でIPやリクエストのログが一切残らないという意味ではありません。

データベースの記録を削除しても、復元履歴やデプロイ時に作成したバックアップが直ちに一緒に消えるとは限りません。旧バージョンの利用統計記録が運用データベースからすべて削除されたことも保証しません。

#### 6. デバイスに残る情報

設定、モデルファイル、診断ログは、主に`%LOCALAPPDATA%\puripuly-heart`に保存されます。APIキーなどの秘密情報には、Windows資格情報マネージャーまたは暗号化されたファイルストアを使用します。診断ログには動作状態やエラー情報が含まれることがあるため、他の人に渡す前に内容を確認してください。

アプリをアンインストールするとアプリのデータフォルダーを整理しますが、Windows資格情報マネージャーや外部サービスに残る情報まですべて削除するという意味ではありません。

#### 7. 関連サービスのポリシー

- サーバー・更新：[Cloudflare](https://www.cloudflare.com/privacypolicy/)、[GitHub](https://docs.github.com/en/site-policy/privacy-policies/github-general-privacy-statement)。
- 認証・運用チャンネル：[Discord](https://discord.com/privacy)、[Tencent QQ](https://privacy.tencent.com/home)。
- モデルのダウンロード：[Hugging Face](https://huggingface.co/privacy)、[ModelScope](https://modelscope.cn/protocol/Privacy-Policy)。
- 音声認識・翻訳：[OpenRouter](https://openrouter.ai/privacy)、[OpenRouterの推論プロバイダー別ロギング案内](https://openrouter.ai/docs/guides/privacy/provider-logging)、[Google](https://policies.google.com/privacy)と[Gemini API利用規約](https://ai.google.dev/gemini-api/terms)、[Alibaba Model Studio](https://www.alibabacloud.com/help/en/model-studio/privacy-notice)、[DeepSeek](https://cdn.deepseek.com/policies/en-US/deepseek-privacy-policy-2025-02-14.html)、[Cerebras](https://www.cerebras.ai/privacy-policy)、[Deepgram](https://deepgram.com/privacy)、[ElevenLabs](https://elevenlabs.io/privacy-policy)、[Soniox](https://soniox.com/policies/privacy-policy)。

選択した機能とプロバイダーに該当するポリシーを確認してください。OpenRouterの実際の推論プロバイダーとカスタムサーバーには、その提供者のポリシーも適用されます。

#### 8. お問い合わせ

プロジェクトの運営者はGitHubの[kapitalismho](https://github.com/kapitalismho)です。お問い合わせ先は[プロジェクトのリポジトリ](https://github.com/kapitalismho/PuriPuly-heart)で確認できます。

### 隐私政策 — 简体中文待审批稿

**译文尚待批准。** 本文翻译自删除第8节指定句子后的已批准韩文正文。安装相关描述适用于之后将实现并验证的版本，并非当前公开安装程序的现有功能。本说明仅供审阅，不属于政策页面正文。

#### 1. 适用范围

本政策说明PuriPuly应用及项目运营的身份验证、使用统计服务器如何处理信息。您选择的外部语音识别、翻译、账号及下载服务也适用各自的政策。

#### 2. 使用统计

PuriPuly将随机标识符和应用使用日期（UTC）发送至服务器，以估算用户数量。使用统计不包含对话内容、音频、翻译结果、语言、服务提供商或模型选择、API密钥。

服务器保存通过密钥转换标识符得到的引用值及使用日期。同一引用值和日期计为一条活动记录，不与账号、设备身份验证信息关联。统计是根据不同匿名标识符的活动作出的估算，并非经核实的实际人数。发送失败或应用重启后，可能在同一天重试请求。

新安装默认开启使用统计。在安装过程中显示的隐私政策页面上取消勾选“发送匿名使用统计”，即可从首次启动起不发送统计请求。安装后也可在设置中更改；升级时，除非您主动更改，否则保留原有选择。这是可选功能设置，不将默认勾选视为单独的同意表示。

关闭统计会停止后续统计请求，并移除设备上的统计标识符。再次开启时会创建新的标识符。这不会立即删除服务器上已保存的活动记录。定期清理会将使用日期比当前UTC日期早超过35天的记录列为删除对象。清理后仍可能保留包括当天在内的36个日期的记录。

关闭使用统计后，更新检查、模型下载，以及您使用的身份验证、语音识别、翻译功能仍会分别进行通信。

#### 3. 语音识别、翻译与VRChat集成

本地语音识别和翻译模型在设备上处理数据。选择外部语音识别服务时，会将待处理音频发送至所选服务；选择外部翻译服务时，会将原文和翻译所需上下文发送至所选服务。您自行添加的API密钥用于相应服务的身份验证。

VRChat集成可从设备上的日志获取参与人数，并将其作为上下文包含在翻译请求中。不会发送此集成读取的参与者姓名或原始日志本身。自定义HTTP翻译扩展也不会收到该参与人数信息。

托管OpenRouter翻译由OpenRouter及实际推理服务提供商处理，请求包含托管用户引用值。项目的身份验证服务器不是常规翻译正文的中转路径。外部服务对内容的保留和训练使用取决于其政策及设置，PuriPuly不保证所有请求均为零保留或不用于训练。

要避免外部处理，可使用本地服务提供程序或关闭相应语音识别、翻译功能。使用本地模型并不会消除单独发生的模型下载或更新检查。使用自定义语音识别地址或HTTP扩展时，待处理音频或文本及配置的身份验证信息会发送至配置的地址，因此应查看该服务器的政策。

OSC功能将字幕和控制信息发送至配置的目标。默认目标位于本设备内，自动连接模式会在局域网中发现及通告服务。您可以在设置中更改连接模式或将其关闭。

#### 4. 托管账号、使用权限与安全

使用托管服务时，应用会与项目服务器通信，用于身份验证、使用权限及用量检查、密钥交付和防止滥用。这一过程可能涉及处理安装标识符、设备公钥、硬件信息的哈希值、应用版本，以及身份验证服务的账号信息或认证值。

项目服务器保存转换后的账号引用值、安装及设备引用信息、使用权限和密钥交付状态、奖励记录等。此数据库不保存QQ认证值或OpenRouter API密钥的原始值。对于身份验证、发放等需要安全检查的请求，会使用通过密钥转换的IP引用值、请求时间、国家、网络及连接特征等记录。这些信息与使用统计表保存的字段相互区分。

配置托管账号后，应用可能在启动时检查使用权限及用量。托管身份验证独立于本地功能及使用您自己的API密钥的功能。身份验证、发放、安全记录适用根据运营设置和到期状态执行的清理规则，并非所有账号、使用权限、已发放奖励的记录都会在相同期限后自动删除。

日密钥发放数量、活动统计等汇总会发送至运营用Discord频道。Discord、QQ本身的账号及服务处理适用相应服务的政策。

#### 5. 更新、模型下载与服务器基础设施

应用启动时会自动向GitHub查询新版本信息。安装程序发现所需模型缺失时，会尝试从Hugging Face或ModelScope下载；应用也可能为准备本地模型而进行下载。这些请求不包含使用统计标识符、对话或音频，但接收方会处理IP地址及一般HTTP、网络信息。使用统计设置不会关闭这些请求。

项目服务器使用Cloudflare Workers和D1。服务器连接所需的IP、请求信息等基础设施处理，与使用统计请求正文中的两个字段不同。数据库活动记录不加入IP，并不意味着包括Cloudflare在内的所有处理层都完全不保留IP或请求日志。

删除数据库记录不一定会立即一并清除恢复历史或部署时创建的备份副本。我们也不保证旧版本的使用统计记录已全部从运营数据库中删除。

#### 6. 设备上保留的信息

设置、模型文件和诊断日志主要保存在`%LOCALAPPDATA%\puripuly-heart`。API密钥等机密信息使用Windows凭据管理器或加密文件存储。诊断日志可能包含运行状态和错误信息，因此向他人提供之前应检查其内容。

卸载应用会清理应用数据文件夹，但不意味着Windows凭据管理器或外部服务中保留的信息也会全部删除。

#### 7. 相关服务政策

- 服务器及更新：[Cloudflare](https://www.cloudflare.com/privacypolicy/)、[GitHub](https://docs.github.com/en/site-policy/privacy-policies/github-general-privacy-statement)。
- 身份验证及运营频道：[Discord](https://discord.com/privacy)、[Tencent QQ](https://privacy.tencent.com/home)。
- 模型下载：[Hugging Face](https://huggingface.co/privacy)、[ModelScope](https://modelscope.cn/protocol/Privacy-Policy)。
- 语音识别及翻译：[OpenRouter](https://openrouter.ai/privacy)、[OpenRouter的推理服务提供商日志说明](https://openrouter.ai/docs/guides/privacy/provider-logging)、[Google](https://policies.google.com/privacy)及[Gemini API条款](https://ai.google.dev/gemini-api/terms)、[Alibaba Model Studio](https://www.alibabacloud.com/help/en/model-studio/privacy-notice)、[DeepSeek](https://cdn.deepseek.com/policies/en-US/deepseek-privacy-policy-2025-02-14.html)、[Cerebras](https://www.cerebras.ai/privacy-policy)、[Deepgram](https://deepgram.com/privacy)、[ElevenLabs](https://elevenlabs.io/privacy-policy)、[Soniox](https://soniox.com/policies/privacy-policy)。

请查看与所选功能及服务提供商相关的政策。OpenRouter的实际推理服务提供商和自定义服务器还适用其提供方的政策。

#### 8. 联系方式

项目运营者为GitHub上的[kapitalismho](https://github.com/kapitalismho)。联系方式可在[项目仓库](https://github.com/kapitalismho/PuriPuly-heart)中查看。

### 隱私權政策 — 繁體中文待核准稿

**譯文尚待核准。** 本文譯自刪除第8節指定句子後的已核准韓文正文。安裝相關描述適用於之後將實作並驗證的版本，並非目前公開安裝程式的現有功能。本說明僅供審閱，不屬於政策頁面正文。

#### 1. 適用範圍

本政策說明PuriPuly應用程式及專案營運的驗證、使用統計伺服器如何處理資訊。您選擇的外部語音辨識、翻譯、帳號及下載服務也適用各自的政策。

#### 2. 使用統計

PuriPuly將隨機識別碼及應用程式使用日期（UTC）傳送至伺服器，以估算使用者人數。使用統計不包含對話內容、音訊、翻譯結果、語言、服務供應商或模型選擇、API金鑰。

伺服器儲存以密鑰轉換識別碼所得的參照值及使用日期。同一參照值和日期計為一筆活動紀錄，不與帳號、裝置驗證資訊連結。統計是根據不同匿名識別碼的活動作出的估算，並非經核實的實際人數。傳送失敗或應用程式重新啟動後，可能在同一天重試請求。

新安裝預設開啟使用統計。在安裝過程中顯示的隱私權政策頁面上取消勾選「傳送匿名使用統計」，即可從首次啟動起不傳送統計請求。安裝後也可在設定中變更；升級時，除非您主動變更，否則保留原有選擇。這是選用功能設定，不將預設勾選視為個別的同意表示。

關閉統計會停止後續統計請求，並移除裝置上的統計識別碼。再次開啟時會建立新的識別碼。這不會立即刪除伺服器上已儲存的活動紀錄。定期清理會將使用日期比目前UTC日期早超過35天的紀錄列為刪除對象。清理後仍可能保留包括當天在內的36個日期的紀錄。

關閉使用統計後，更新檢查、模型下載，以及您使用的驗證、語音辨識、翻譯功能仍會分別進行通訊。

#### 3. 語音辨識、翻譯與VRChat整合

本機語音辨識和翻譯模型在裝置上處理資料。選擇外部語音辨識服務時，會將待處理音訊傳送至所選服務；選擇外部翻譯服務時，會將原文及翻譯所需上下文傳送至所選服務。您自行新增的API金鑰用於相應服務的驗證。

VRChat整合可從裝置上的記錄檔取得參與人數，並將其作為上下文納入翻譯請求。不會傳送此整合讀取的參與者姓名或原始記錄檔本身。自訂HTTP翻譯擴充功能也不會收到該參與人數資訊。

託管OpenRouter翻譯由OpenRouter及實際推論服務供應商處理，請求包含託管使用者參照值。專案的驗證伺服器不是一般翻譯正文的轉送路徑。外部服務對內容的保留和訓練使用取決於其政策及設定，PuriPuly不保證所有請求均為零保留或不供訓練使用。

若要避免外部處理，可使用本機提供者或關閉相應語音辨識、翻譯功能。使用本機模型並不會消除另外發生的模型下載或更新檢查。使用自訂語音辨識位址或HTTP擴充功能時，待處理音訊或文字及設定的驗證資訊會傳送至設定的位址，因此應查看該伺服器的政策。

OSC功能將字幕和控制資訊傳送至設定的目標。預設目標位於本裝置內，自動連線模式會在區域網路中探索及公告服務。您可以在設定中變更連線模式或將其關閉。

#### 4. 託管帳號、使用權限與安全

使用託管服務時，應用程式會與專案伺服器通訊，用於驗證、使用權限及用量檢查、金鑰交付和防止濫用。這一過程可能涉及處理安裝識別碼、裝置公鑰、硬體資訊的雜湊值、應用程式版本，以及驗證服務的帳號資訊或驗證值。

專案伺服器儲存轉換後的帳號參照值、安裝及裝置參照資訊、使用權限和金鑰交付狀態、獎勵紀錄等。此資料庫不儲存QQ驗證值或OpenRouter API金鑰的原始值。對於驗證、發放等需要安全檢查的請求，會使用以密鑰轉換的IP參照值、請求時間、國家、網路及連線特徵等紀錄。這些資訊與使用統計資料表儲存的欄位分開處理。

設定託管帳號後，應用程式可能在啟動時檢查使用權限及用量。託管驗證獨立於本機功能及使用您自己的API金鑰的功能。驗證、發放、安全紀錄適用依營運設定和到期狀態執行的清理規則，並非所有帳號、使用權限、已發放獎勵的紀錄都會在相同期限後自動刪除。

每日金鑰發放數量、活動統計等彙總會傳送至營運用Discord頻道。Discord、QQ本身的帳號及服務處理適用相應服務的政策。

#### 5. 更新、模型下載與伺服器基礎設施

應用程式啟動時會自動向GitHub查詢新版本資訊。安裝程式發現所需模型缺少時，會嘗試從Hugging Face或ModelScope下載；應用程式也可能為準備本機模型而進行下載。這些請求不包含使用統計識別碼、對話或音訊，但接收方會處理IP位址及一般HTTP、網路資訊。使用統計設定不會關閉這些請求。

專案伺服器使用Cloudflare Workers和D1。伺服器連線所需的IP、請求資訊等基礎設施處理，與使用統計請求正文中的兩個欄位不同。資料庫活動紀錄不加入IP，並不代表包括Cloudflare在內的所有處理層都完全不保留IP或請求記錄。

刪除資料庫紀錄不一定會立即一併清除復原歷程或部署時建立的備份副本。我們也不保證舊版本的使用統計紀錄已全部從營運資料庫中刪除。

#### 6. 裝置上保留的資訊

設定、模型檔案和診斷記錄檔主要儲存在`%LOCALAPPDATA%\puripuly-heart`。API金鑰等機密資訊使用Windows認證管理員或加密檔案儲存。診斷記錄檔可能包含執行狀態和錯誤資訊，因此提供給他人之前應檢查其內容。

解除安裝應用程式會清理應用程式資料夾中的資料，但不代表Windows認證管理員或外部服務中保留的資訊也會全部刪除。

#### 7. 相關服務政策

- 伺服器及更新：[Cloudflare](https://www.cloudflare.com/privacypolicy/)、[GitHub](https://docs.github.com/en/site-policy/privacy-policies/github-general-privacy-statement)。
- 驗證及營運頻道：[Discord](https://discord.com/privacy)、[Tencent QQ](https://privacy.tencent.com/home)。
- 模型下載：[Hugging Face](https://huggingface.co/privacy)、[ModelScope](https://modelscope.cn/protocol/Privacy-Policy)。
- 語音辨識及翻譯：[OpenRouter](https://openrouter.ai/privacy)、[OpenRouter的推論服務供應商記錄政策說明](https://openrouter.ai/docs/guides/privacy/provider-logging)、[Google](https://policies.google.com/privacy)及[Gemini API條款](https://ai.google.dev/gemini-api/terms)、[Alibaba Model Studio](https://www.alibabacloud.com/help/en/model-studio/privacy-notice)、[DeepSeek](https://cdn.deepseek.com/policies/en-US/deepseek-privacy-policy-2025-02-14.html)、[Cerebras](https://www.cerebras.ai/privacy-policy)、[Deepgram](https://deepgram.com/privacy)、[ElevenLabs](https://elevenlabs.io/privacy-policy)、[Soniox](https://soniox.com/policies/privacy-policy)。

請查看與所選功能及服務供應商相關的政策。OpenRouter的實際推論服務供應商和自訂伺服器還適用其提供方的政策。

#### 8. 聯絡方式

專案營運者為GitHub上的[kapitalismho](https://github.com/kapitalismho)。聯絡方式可在[專案儲存庫](https://github.com/kapitalismho/PuriPuly-heart)中查看。

### Language-draft checkpoint disposition

Independent `privacy-language-review` completed `FULL_REVIEW` of `b3d3f30f7fe0f9591ebfb9fbff44b9daadcaee24..127f5ccd1f108bdd43adbcbd1d7cf505e6347cf2` with `no_material_findings`. It verified the exact one-sentence Korean deletion, the bounded operator Logs/Traces OFF report, semantic equivalence of all four full policies, and the seven translated installer-copy elements. Each of the five policy bodies has eight sections and the same 18 service/contact links. The removed public-issue warning is absent from every translated body.

The Director accepts the translation drafts as ready for direct maintainer approval, not as approved translations or a shipped installer. The earlier Korean factual review remains applicable because the Korean body has no other changes. No runtime code or architecture changed, and no throwaway scripts or permanent tests were added. Remaining D1 configuration evidence is separate from the already granted wording approval.

### Translation approval and installer implementation outcome

The maintainer directly approved the four full policy translations and the installer copy table in response to the preceding approval request. Together with the earlier Korean approval, all five language bodies and the proposed localized installer copy are approved for implementation. Historical draft headings above identify the reviewed proposals; they do not override this approval. Publication, Foundation submission, deployment and production deletion remain separate actions.

Outcome baseline: `adf37b566ece698dbd1d04d30373acd0f54ed7c5`, branch `apply-code-signing-by-signpath`, no upstream, clean pre-existing tree. Issue #74 is open and its Project status was read back as `In progress`.

State: `PLANNED` → `IMPLEMENTING`. One `heavy_worker`, `installer-privacy-implementation`, owns the installer UI, mechanically extracted approved policy assets, canonical settings/entry-point integration, and focused verification. A single owner is used because first-transmission ordering, upgrade preservation and Windows original-user/elevated profile handling share a high-impact persistence boundary. The Director retains approval/progress documentation and public-policy linkage decisions.

Required proof: actual five-language offline policy display and accessible checkbox, fresh ON/OFF, existing ON/OFF upgrades without unrelated settings loss, failure-safe persistence, canonical Settings agreement and no first-start telemetry after installation opt-out. The implementation must reuse supported config lifecycle and smoke isolation, never touch the maintainer's actual app profile. Runtime code changes require a newly built packaged candidate. The integration barrier is all implementation plus local verification, then a committed candidate for independent installer/UI coverage and persistence/first-transmission specialist coverage. Scope B/signing/live deployment are not part of this installer-only Outcome.

Desktop-verification interruption: the maintainer reported repeated installer windows during implementation. The Director stopped further installer/UI verification, including silent installation and uninstall probes, pending explicit permission. The implementation owner reports its probes launched `installer_output/privacy-smoke/PuriPulyHeart-Setup-2.6.1.exe`; the last observed setup PIDs were 27772/35540, and its subsequent CIM query found zero matching setup processes. No screenshots were successfully captured. This is not a visual-validation pass. Noninteractive source and test work may continue; the installer runtime criteria remain blocked, not waived.

Operational failure messages may identify failed preference reads/writes and prevent unsafe installation completion without another policy-approval round. They must not add consent, privacy or retention claims. The five approved policy bodies and seven approved installer-copy elements remain unchanged.

Implementation owner reported source changes complete with noninteractive checks passing: focused installer-preference/main CLI/core telemetry tests, dependency guards/public compatibility tests, settings migration/startup tests, Black/Ruff, and PowerShell parsing. The hidden preference command uses the canonical settings lifecycle before runtime composition. Approved policy assets are in `installer/privacy/`; root `PRIVACY.md` contains the approved English body. All five installer bodies were checked for eight sections and 18 reference URLs.

The exact-source PyInstaller rebuild produced main executable SHA-256 `2a0a86f8cb0119bdc53a2019730ff71e650438b35c7261f2f284b195da3c00bd`. Inno Setup 6.6.1 compiled, but did not execute, `installer_output/privacy-final/PuriPulyHeart-Setup-2.6.1.exe`: 173,280,875 bytes, SHA-256 `c4500fd4dde1a0da4c925e2ba443f4c3838b30b515f725db65ef225431250323`. Before the desktop interruption, an earlier isolated silent fresh install saved ON, and a separate packaged helper probe saved OFF with no anonymous ID; those are not final-candidate or five-language lifecycle proof.

State: `IMPLEMENTED / CLAIMED`, runtime verification `BLOCKED`; not `IMPLEMENTATION_READY` or accepted. Preserve a committed source checkpoint for independent noninteractive persistence and installer-integration scrutiny, with the known runtime gaps retained. This bounded review does not waive the original acceptance gate. No installer execution may resume without explicit permission. The earlier isolated installed fixture is preserved; uninstall cleanup would itself require permission.

Independent source checkpoint: `installer-privacy-source-review` reviewed `adf37b566ece698dbd1d04d30373acd0f54ed7c5..7c755672fd138a560b2b178434a6d4eaf20fe7e9`, including installer integration and specialist persistence/profile/ordering risks. It confirmed approved-copy equivalence, 271 passing focused tests and byte-identical compiler output without running an installer. Visual/lifecycle proof remains blocked, not inapplicable.

Director adjudication of the complete findings wave: **ACCEPT F1**, original-user execution can fall back to the wrong elevated identity; **ACCEPT F2**, unsupported settings can be discovered too late after downloads/file replacement; **ACCEPT F3**, the additional writer exposes deterministic temporary-file collisions; **ACCEPT F4**, local smoke uninstall cleanup still targets the operator's real app-data directory despite isolated preference writes. These findings are not waived by passing tests. State: `REPAIRING` for noninteractive source work, with runtime acceptance still `BLOCKED`. The original owner receives all repairs together, including narrowly scoped atomic persistence changes if required. Preserve supported installation modes; escalate any unavoidable compatibility decision rather than silently dropping a mode. No installer or uninstall execution is authorized.

Repair owner reports F1–F4 addressed without installer execution: unsafe already-elevated profile resolution is refused with restart-without-elevation guidance; normal per-user/in-wizard elevation remains intended supported behavior; the read probe rejects future/invalid telemetry earlier; unique exclusive temporary files replace shared persistence temporaries; and the smoke root now covers app processes plus install/uninstall deletion. Accurate policy-load errors are distinct from preference errors. The Director accepts fail-safe refusal rather than guessing another user's profile. Its Windows lifecycle behavior remains unverified.

Focused privacy/settings/release tests and the settings suite passed; PowerShell parsing and an extracted real probe matrix passed (ON=20, OFF=21, future/null/string-enabled=23). Newly introduced source-text-only tests were removed; behavioral persistence regression checks remain. The rebuilt packaged helper ran against an isolated temporary config and saved OFF with no anonymous ID, exit 0. The repaired compiled installer remains **not executed**: `installer_output/privacy-final/PuriPulyHeart-Setup-2.6.1.exe`, 173,291,757 bytes, SHA-256 `99f017a5f61dd2338afa039e4c96875c115fc8bf7bb9367d74dc9caff64bed24`; packaged executable SHA-256 `c3b46008c772c6966a1737046fa5e012cc9ed2dddb8227607ab01d310c098e9e`.

Residual persistence limitation: unique temporary files prevent shared-temporary collisions, not concurrent read-modify-write transactions. Complete stale snapshots can still replace one another; no cross-process transaction guarantee is claimed. The next review is retained `REPAIR_VERIFY` against the complete committed repair delta, with blocked runtime evidence unchanged.

Retained review at `910a31d7` closed F1–F4 at the source/noninteractive level but found **N1**, a new smoke-destination regression: deriving the installation directory from the overridden temporary LOCALAPPDATA would trigger the production temporary-directory guard and redirect installation onto the real default path. The Director **ACCEPTS N1** and withheld acceptance. The original owner repaired it by capturing original user Local AppData before overrides and retaining the separate non-temporary `Programs\PuriPulyHeart-LocalSTT-Test` destination. Production path guards remain enabled; app-data isolation remains temporary. A `finally` now restores environment and removes the scratch profile on failure as well as success.

N1 repair evidence: a throwaway executable path predicate reproduced the old temporary destination and confirmed the repaired destination is neither temporary nor the production path; forced-failure cleanup restored both environment variables; PowerShell parsing, affected focused tests and lint passed. Obsolete exact-path source pins were removed rather than re-pinned. Only smoke script/test metadata changed, so the `99f017a5...` compiled installer remains applicable, still unexecuted. A further retained repair review is required; all original runtime blockers remain.

Retained `REPAIR_VERIFY` at `0bb2a625` verified N1 with no new findings and 308 passing noninteractive tests. The Director additionally **ACCEPTS R2/R3** from its residual notes for bounded smoke safety: smoke icons must not touch production shortcuts, and suspicious smoke destinations must refuse rather than reset to the production directory. The original owner added smoke-only compile guards that omit every icon and reject empty/default/repository/temporary destinations. Production reset behavior is unchanged. Both compile variants and a safe non-GUI action probe passed; production output stayed byte-identical (`99f017a5...`). These source changes require retained repair verification.

Disposition of other residuals: **R1 accepted limitation**, a failed smoke run may leave its distinct test installation for separately authorized cleanup; no automatic uninstaller was run. **R4 accepted informational risk**, the installer schema-version constant currently equals canonical version 39 but must be maintained with future schema changes. Neither residual permits touching the actual production profile or waives blocked runtime checks.

### Maintainer revision UI-2026-09-13-B

Baseline: `5b03677c5b22234930ea96d07fd426bb7329b759`, branch `apply-code-signing-by-signpath`, no upstream, clean before this revision. The original owner remains responsible for installer layout and configuration-path mechanics; the Director personally owns all policy wording and translations. Earlier source repairs at this baseline were independently verified with 308 focused tests. That evidence does not validate the revised layout or policy.

The maintainer now requests: replace middle-dot separators with commas; show only the full policy on the privacy page, like the license page; place only the single localized anonymous-statistics checkbox on the ASR redownload options page, with no adjacent short explanation, heading or guidance; remove low-importance policy content and propagate the revision across five languages using the Director's judgment. This supersedes the earlier same-page full-policy/short-disclosure/checkbox criterion and the earlier fixed eight-section, 18-link bodies. Those superseded criteria remain historical, not passed or continuing obligations. Default ON, preserved upgrade choice, and opt-out effective before first transmission remain unchanged.

The Director has personally rewritten Korean, English, Japanese, Simplified Chinese and Traditional Chinese into six sections. Repeated implementation detail and the long provider-link list are removed. Material disclosures remain: telemetry fields and purpose, default and upgrade behavior, disable/re-enable identity lifecycle, the 35-day cutoff/36 possible UTC dates, external audio/text and provider retention/training limits, managed-account/security processing, background communications, local credentials/logs, deletion limits and operator contact. No middle-dot separators remain. Root `PRIVACY.md` is the same English body with Markdown headings. Mechanical file checks passed; independent semantic review is still required. This local revision records the new agreement; the older remote issue text has not been silently edited.

Korean validation evidence and correction: the first authorized invocation failed before UI because the supplied log parent was absent; the missing parent is an observed preparation error, not a proven sole cause of exit 1. The separately authorized retry (root PID 36968, child 37508) ran from 21:03:29 to 21:05:10 UTC+09 on 2026-09-13 and exited 0. Its screenshot, `installer_output/korean-stage-5b03677c/korean-stage-welcome.png`, shows the previous Korean full-policy memo and same-page checkbox. A replacement validator launched zero processes; total setup invocations were two, not three.

The maintainer reports that ordinary app launch retained the existing settings and showed telemetry ON. This is ground truth. The smoke compile's path lost backslashes and became a relative config path under the isolated installation; OFF was stored there, not in the default settings consumed by ordinary launch. Therefore the retry proved only display of the previous Korean page and writing OFF into that test fixture, not installed-app Settings agreement or first-launch suppression. The expected absolute fixture root was absent. Preserve the installed test fixture pending separately authorized cleanup. Repair smoke path handling and validate helper/app-loader agreement on one explicitly selected config without changing the maintainer's actual settings or adding a production preference side channel.

No further installer, uninstall, silent-install or app-GUI execution is authorized. Implementation and noninteractive checks proceed; final compilation depends on the Director's updated policy assets. Changed UI needs a new explicitly authorized runtime check. The prior successful screenshot cannot satisfy the revised page-layout criteria.

Revised implementation checkpoint: owner reports memo-only full-height privacy page, one localized checkbox row on the ASR/options page, and complete removal of unused short-disclosure strings/controls. Canonical persistence behavior is unchanged. Smoke roots are passed as absolute forward-slash paths, rejected at compilation if malformed, normalized at runtime and reused across preference/model/deletion paths without a separate installer environment override. No production settings sidecar was introduced.

Noninteractive evidence: 14 focused tests passed; PowerShell parsing/root preflight passed; the frozen hidden helper wrote OFF with no anonymous ID using an explicit isolated config. Both Inno Setup 6.6.1 variants compiled against the Director's six-section policies; a malformed `C:relative` definition was rejected. Production artifact: `installer_output/ui-2026-09-13-b/production/PuriPulyHeart-Setup-2.6.1.exe`, 173,284,718 bytes, SHA-256 `58c8c37504d543bbf6a7a32feece97226ba3b28a6f9dab39d11bc9b3dd05f6b6`. Smoke artifact: `installer_output/ui-2026-09-13-b/smoke/PuriPulyHeart-Setup-2.6.1.exe`, 173,284,780 bytes, SHA-256 `1276e9ca9ee0e39b112bb828a584e989cd7bd4484e336b7c8c2da33045b977de`. Neither was executed.

State: revised source `IMPLEMENTED / CLAIMED`, runtime acceptance `BLOCKED`. The next stable-commit review covers the complete revision in two lanes: retained installer/persistence integration scrutiny under the revised contract and a fresh independent policy factual/translation review. Unchanged original review evidence may be retained, but no earlier UI screenshot establishes the new layout or ordinary first-start agreement.

Complete revised review wave at `06afe548150d14bd6646a40310aa967624cdd895`: fresh `condensed-privacy-policy-review` reported `no_material_findings`, confirming all five bodies, English Markdown equivalence, factual disclosures and directed omissions. Its optional wording observations are **REJECTED as unnecessary cosmetic changes**; no factual or material translation defect was identified. The retained installer lane verified the revised source layout, normalized shared paths, malformed-define rejection, and isolated helper/app-loader agreement, but found **F5**, a stale compatibility-inventory reference to the deliberately deleted source-pin test. Its broader focused suite returned two failures and 268 passes. The Director **ACCEPTS F5** and directly removes that single stale reference rather than restoring the obsolete test. Runtime acceptance remains blocked.

### Maintainer revision UI-2026-09-13-C

Baseline: `d09a60b01c2c9be6d3d98258fbe2e9ef048f957c`, branch `apply-code-signing-by-signpath`, no upstream, clean before revision. The maintainer explicitly confirmed that a separate privacy page should match the license-page layout, with agree mapping to telemetry ON and decline mapping to OFF, while either choice permits installation. The actual software license agreement page and its mandatory gate remain unchanged. This replaces UI-B's memo-only page and additional-options checkbox. The unchanged fresh-install default is ON; upgrade choice preservation and the canonical boolean remain required. No separate consent state machine is introduced.

Preserve the native localized license introduction (Korean: `계속하기 전에 다음 중요한 정보를 읽어보세요.`). The privacy instruction is exactly `약관에 동의하면 최소 데이터를 익명으로 전송해요.`; the Director personally supplied English, Japanese, Simplified Chinese and Traditional Chinese equivalents. Native localized agree/decline captions are reused on the separate privacy page. Remove the additional-options telemetry checkbox and all its obsolete callers/messages. The Director has personally updated the policy choice paragraph across all five installer assets and root English policy; other six-section disclosures remain unchanged.

The retained heavy implementation owner handles layout, canonical probe/persistence control migration and narrowly related checks. The Director owns sensitive policy/i18n wording and this record. Noninteractive checks and compiles may proceed; new UI-C runtime evidence is required. The preceding UI-B Korean isolated install successfully displayed the then-current layout and saved OFF at an absolute scoped config, but actual Settings GUI inspection stopped because the maintainer's pre-existing app window remained active. The maintainer again reports ordinary app settings ON. This unresolved end-to-end observation is not overridden by fixture-file or helper results. Existing app/profile and old fixtures remain untouched.

UI-C state: `PLANNED` → `IMPLEMENTING`; actual GUI/profile-agreement validation remains blocked pending a safe staged execution boundary. Integration waits for both the owner's complete code changes and the Director's updated policy assets. Review must cover the revised two-radio behavior, unchanged real-license gate, five-language instruction/policy meaning, and canonical setting ordering; prior layout screenshots do not satisfy UI-C.

UI-C source checkpoint: owner changed only `installer.iss`, cloning the native license instruction/viewer/radio bounds and fonts onto the separate privacy page, reusing native introduction/radio messages and the Director's translated instruction. The tasks checkbox and its layout/copy are removed. Both privacy choices feed one telemetry boolean without gating Next; the real software-license page remains untouched. The Director's five policy paragraphs describe these choices consistently. Production and isolated variants compiled with Inno Setup 6.6.1; neither was executed. Production SHA-256: `574ad0b439aa67c162926362c9ae0a9497ef0be9719b033fa71903eba9f123a2`; isolated SHA-256: `2929f929c01fa46f8f124e4395709580dd7a094e1fb5dca1ed6e382a026b00c3`, under `installer_output/ui-2026-09-13-c/`.

The owner ran seven installer-preference tests successfully. The Director then ran the installer-preference, release-dependency, public-compatibility and settings-migration suites together: exit 0. Source investigation found no evidence warranting a production default-path change: the installer original-user path and ordinary app default resolve the same profile convention; smoke variants deliberately target another profile. Actual GUI Settings agreement remains unverified. State is `IMPLEMENTED / CLAIMED`, not accepted; a fresh independent source/meaning review covers the revised radio behavior and retained license gate at the next committed boundary, with UI runtime blockers explicit.

Fresh UI-C review (`license-style-privacy-review`, `d09a60b0..ccba612382706526b8d48d966efd7b6073de56ba`) verified source layout, separate radio grouping/state flow, unchanged mandatory license gate, and all five instruction/choice-paragraph meanings. It found one low-severity localization defect: Traditional Chinese's native introduction called the privacy page a license agreement. The Director **ACCEPTS F1** and corrects that locale's native introduction to the already-used neutral wording `在繼續安裝之前請閱讀以下重要資訊。`, matching the requested Korean meaning. This neutral introduction is shared by both pages; the actual license body, mandatory instruction and gate remain unchanged. Runtime dependencies D1–D4 (radio clicks/rendering/silent/elevation/actual app Settings) remain unverified, not source defects or waived checks. Recompile affected artifacts and retain the reviewer for the bounded locale repair.

F1 repair compilation: both variants rebuilt with Inno Setup 6.6.1, without execution. Production: 173,292,449 bytes, SHA-256 `827c9650ba2a07b4465c8a3437d99bba66118125ffbe1bc77706b409414595c1`. Isolated smoke: 173,292,462 bytes, SHA-256 `a861f5d895ff61a6e113d3c505371246b2fa45900beb4f7c8acbb81e6cdbaa21`. Paths remain `installer_output/ui-2026-09-13-c/{production,smoke}/PuriPulyHeart-Setup-2.6.1.exe`. Prior source-test evidence remains applicable to this introduction-only repair; actual UI/profile agreement is still not verified.

### UI-C runtime validation resumption

The maintainer requested continuation after the reviewed UI-C build. Two setup attempts by the original validation owner exited 1 before any window or installation: overriding setup's profile/environment variables changed the runtime temporary-root interpretation and triggered the compiled-root safety guard. No settings or installation were created. Failure logs are retained under `%LOCALAPPDATA%\Temp\PuriPulyHeart-UI-C-CompileOnly`. These are validation-environment failures, not evidence of product consent behavior.

The Director settled that validation owner with zero owned running processes, then assigned fresh `heavy_worker` `ui-c-korean-validation` to the demanding native-UI/profile-boundary check. Setup must inherit the host environment unchanged; compiled smoke paths provide isolation. Only the later installed app receives scoped LOCALAPPDATA matching that compiled settings path. One owned validation window may run at a time; an unrelated existing user app is not itself a technical blocker, but must never be closed, modified or mistaken for the test app. The authorized stage is Korean license/privacy navigation, OFF installation and actual installed Settings OFF, then report before other languages/upgrades. No production-profile mutation, uninstall or remote operation is authorized.

The fresh validator's inherited-environment setup ran once (root PID 37252, wizard PID 33940) and exited 0, installing to the isolated `Programs\PuriPulyHeart-UI-C-Korean-Validation` path. It observed the target-directory page and later Finish, but its sole Back-button lookup failed before any click was sent. The interactive command contained no silent, task-selection or INF-loading flags. The wizard's advancement mechanism is unknown; do not attribute it to user input or a product defect without evidence. The compiled scoped config contains default ON and an identifier. No app was launched, no config was changed afterward, and no screenshot was captured.

State: runtime `BLOCKED / UNVERIFIED`, awaiting input-ownership coordination with the maintainer. This attempt does not prove the privacy controls, decline/OFF installation or installed Settings agreement. Log: `installer_output/ui-2026-09-13-c/smoke/korean-setup-host-env.log`. Preserve this installed fixture and its config; no repeat invocation or uninstall is authorized at this boundary.

The maintainer confirmed operating the preceding setup and explicitly delegated exclusive input to the agent. The subsequent Korean controlled run used the preserved ON fixture, so its valid scope is **upgrade ON → explicit decline → OFF**, not fresh OFF. Setup (root 36908, wizard 24920) exited 0. Actual observations: software-license decline disabled Next, acceptance enabled it; the separate privacy page matched native geometry and exact Korean copy; both privacy choices permitted Next and survived Back; the full six-section policy scrolled correctly; the additional-options page contained no telemetry checkbox. OFF persistence removed the identifier and preserved all non-telemetry settings immediately after installation.

The installed app then launched without arguments or `--config`, with only child LOCALAPPDATA scoped to the parent of the compiled settings root. Its actual Settings screen displayed `익명 데이터 전송 / 꺼짐`; the Director inspected `installer_output/ui-2026-09-13-c/smoke/korean-evidence/15-installed-app-settings-telemetry-off.png`. App root 25456 and owned Flet child 12816 exited cleanly. A firewall prompt from the isolated automatic OSC configuration was denied/closed, with no allow rule created. App startup subsequently normalized two local STT provider fields; telemetry remained OFF with no identifier. No HTTP observer was used, so no packet-level or all-network-suppression claim is made. The maintainer's actual profile remains unchanged.

The same validation owner continues the authorized automatic matrix: Japanese unchanged-OFF upgrade, fresh English OFF install and installed-app check, Simplified/Traditional Chinese visual and navigation checks, and separate isolated silent fresh-ON/unchanged-ON installs without launching the ON app. All setup processes inherit the host environment; app-only scoped environment aligns default config. No production installer, elevation or uninstall is authorized. Evidence for each case remains tied to its source/artifact, fixture and actual observations; remaining cases are not yet passed.

### Validation scope revision UI-C-VALIDATION-2

The maintainer explicitly removed per-language visual checking: `언어별로 다 볼 필요는 없을 거 같아. 빼`. The Director stops language-only runs and drops the combined five-language runtime task as superseded, replacing it with a focused fresh-install and preference-preservation task. Five-language shipped assets and completed semantic/source review remain unchanged. Unperformed language checks are removed requirements, not passes. Existing Korean runtime evidence remains applicable; only functional fresh OFF, retained OFF, fresh silent ON and retained ON checks continue if not already completed. No extra language runs, remote issue edits, production-profile changes, elevation or uninstall are authorized by this revision.

The retained functional matrix completed on reviewed source `a6a6130f1bb97c7effde2df78949d72fa9bc5c65` and packaged app SHA-256 `c3b46008c772c6966a1737046fa5e012cc9ed2dddb8227607ab01d310c098e9e`:

| Functional case | Observed result | Evidence |
| --- | --- | --- |
| Existing ON, explicit decline during upgrade | OFF, identifier removed, unrelated settings retained; actual installed Settings OFF | `smoke/korean-setup-agent-input.log`, `smoke/korean-evidence/15-installed-app-settings-telemetry-off.png` |
| Existing OFF, upgrade without changing final choice | OFF and no identifier retained; unrelated settings retained, exit 0 | `smoke/japanese-upgrade-off.log` |
| Fresh install, decline | Initial default ON observed, decline persisted OFF/no identifier, exit 0; actual installed Settings OFF | `english-fresh/english-fresh-off.log`, `english-fresh/evidence/10-installed-app-settings-off.png` |
| Fresh silent install | ON and identifier created, exit 0; ON app not launched | `silent-default-on/silent-fresh-on.log` |
| Silent reinstall retaining ON | Same identifier and byte-identical settings retained, exit 0 | `silent-default-on/silent-reinstall-preserve-on.log` |

All paths in the table are relative to `installer_output/ui-2026-09-13-c/`. The Director inspected both installed-app OFF screenshots. The fresh-OFF isolated artifact is 173,292,501 bytes, SHA-256 `c3973f8e9a6dccaedede6eb62e6b758f0087badf9021ee4e667769e7cee60209`; the silent-ON isolated artifact is 173,292,493 bytes, SHA-256 `5664971d1e3b0a3f08aa000d6546a6b2a507a0bc843301554002b15094001bde`. Both were compiled from the same reviewed installer source with isolated AppIds/paths. Installer processes inherited the host environment; ordinary installed-app GUI runs used only scoped child LOCALAPPDATA, with no `--config` and no post-install telemetry edits.

The Japanese run had already completed before the scope reduction and is retained for its unchanged-OFF functional result. The already-open Simplified Chinese language-only run was cancelled normally before installation (exit 2); its OFF fixture was unchanged. Traditional Chinese language-only checks were not started. Those removed checks are not reported as passes. All owned validation windows/processes are closed. Test fixtures are preserved because uninstall/fixture deletion has not been authorized. No production profile, production installer, elevation, release, deployment or firewall allow operation was performed.

The Director accepts the above **local, non-elevated functional checks** as evidence at the reviewed source, not full issue/Goal completion. Elevated/original-user runtime cases, uninstall/signed-candidate validation and packet-level telemetry observation remain unperformed; no previous source-review blocker is silently waived by this result. Five-language product support and semantic review remain intact.

### Scope revision PREAPPLICATION-1

The maintainer explicitly instructed: `1번 넘기고 2번은 신청 전만 진행하자.` This supersedes treating elevated/original-user installation and uninstall checks as blocking the current preapplication Outcome: they are skipped, not passed. The earlier removal of locale-by-locale runtime checks remains in force. Account configuration/MFA verification, actual signing integration and signed-candidate validation remain later-stage work, not experiments required before application. The complete historical #74 implementation is not declared finished.

The current authorized Outcome is to reconcile the local Code signing policy draft with the implemented privacy control, resolve available corresponding-source evidence without unsolicited external inquiry, and prepare concise application material for direct maintainer review. This does not approve the final policy or application contents and does not authorize submission, account/permission changes, signing requests, production deletion/deployment, push, publication, elevation, uninstall, or fixture cleanup. No remote issue edit is performed; this local record preserves the later agreement and its relationship to #74.

Baseline `57833163f02f6805b630ccae4b143daccdde3b23`, branch `apply-code-signing-by-signpath`, no upstream and no pre-existing dirty paths. The Director owns sensitive policy and this record. Read-only researcher `preapplication-evidence-research` owns the bounded remaining-source/application-requirements investigation, not implementation or independent review. Existing reviewed code and runtime evidence remain applicable because this Outcome changes documentation only. A fresh independent researcher will review the integrated committed documentation candidate before acceptance of the preparation delta.

#### Application material prepared for maintainer review

This is local draft material, not a submitted form or a declaration that every eligibility condition is met.

| Item | Prepared value or evidence |
| --- | --- |
| Project | PuriPuly <3 — a Windows application for real-time two-way voice translation, including VRChat integration |
| Project source | https://github.com/kapitalismho/PuriPuly-heart |
| Project license | AGPL-3.0-or-later, as identified in the repository; bundled components retain their respective licenses |
| Existing downloads | https://github.com/kapitalismho/PuriPuly-heart/releases — existing official installer is unsigned and predates the prepared privacy UI |
| Proposed signing policy | `CODE_SIGNING.md`, local draft pending direct final approval and separately authorized publication |
| Privacy evidence | `PRIVACY.md`, five offline installer assets, reviewed source `a6a6130f`, current-user runtime evidence recorded at `57833163` |
| Proposed targets | Project main, overlay and GPU worker EXEs, installer, and generated Inno uninstaller; untouched third-party binaries excluded from individual signing |
| Approval responsibility | `@kapitalismho` personally approves each of two planned sequential signing requests; no automatic release publication |
| Reputation | Research snapshot, 2026-09-13: 114 GitHub stars, 12 forks, 25 releases from 2026-01-16 to 2026-09-08, approximately 2,585 cumulative release-asset downloads; refresh before submission |
| Supply-chain evidence | Existing executable inventory, exact license-payload guards and soxr/zeroconf companion source bundle; libmpv matching-source completeness is unresolved |

Proposed application explanation:

> PuriPuly <3 is an AGPL-3.0-or-later Windows application for real-time two-way voice translation. We propose signing our project executables, installer and generated Inno Setup uninstaller, not individually re-signing untouched third-party runtime binaries. The prepared installer displays the complete Privacy Policy offline in five languages and allows usage statistics to be disabled before first launch without preventing installation. Statistics default to enabled, and upgrades preserve the user's choice. Automatic GitHub update checks and model downloads from Hugging Face or ModelScope are separate network activities disclosed in the policy; turning statistics off does not disable them. Our maintainer will personally approve each signing request. Existing published installers are unsigned and predate this prepared privacy implementation.

The final application must disclose the proposed packaged/generated targets and automatic network activities for Foundation assessment rather than claim that their eligibility or privacy interpretation has already been accepted. Do not claim complete corresponding-source coverage while the libmpv gap remains. Personal application/contact details must be entered or expressly supplied by the maintainer at the later submission step; no contact identity, email address or account is fabricated here.

#### Remaining source evidence and disposition

The bounded read-only investigation could not close libmpv matching-source completeness using accessible public evidence. The retained identity chain links the shipped DLL SHA-256 `d5f0694b08c124e785d858d00082f3e3b158dd9138bfc48c0382bf1eb443a5fc` to the media-kit 2023-09-24 asset and historical recipe `af821697a4d779bd6a09f9a5cf2fb6418d7ca2fd`. mpv and FFmpeg revisions and the two FFmpeg patches are known; roughly twenty recipe dependencies are not pinned.

Additional evidence: the [release record](https://api.github.com/repos/media-kit/libmpv-win32-video-build/releases/tags/2023-09-24) provides binaries and an mpv revision, not a full dependency manifest; its tag tree carries only the version file. The retained build workflow at `23doors/libmpv-win32-video-build@8c5e6289ed00e5228be463e479967604939965bd` generated a per-run Packages Version summary, but the investigated original/fork Actions listings returned no accessible runs. The workflow's restored source cache and conditional update behavior make “upstream tip on build date” an unsupported substitute for the missing record. This establishes unavailable evidence, not proof that records never existed.

Disposition: keep matching-source completeness BLOCKED on an actual build dependency record or a separately approved dependency/build change. Neither upstream correspondence nor dependency replacement is authorized. Do not call the binary nonredistributable from this evidence, require byte-identical rebuilding as a new condition, fabricate source revisions, or declare blanket license compliance. The application draft is usable for review, but the complete supply-chain declaration is not ready for approval. Account creation, administrator install/uninstall testing and real signing would not resolve this source gap.

#### Form and publication boundary

The [application form](https://signpath.org/apply.html) requests project/repository/homepage information, a description and reputation evidence, account-holder name/email and consents. Its download field expects mention of Foundation signing, and a Privacy Policy URL is required for data-collecting software. The [terms](https://signpath.org/terms.html) call for a public “Code signing policy” reference on home/download surfaces, roles, attribution and privacy information. Research found no current signing-policy reference on the existing README/release surfaces. Local relative policy links and all five shipped privacy assets were checked; they do not establish public availability of the prepared documents.

Public wording must not claim signing is already provided. `CODE_SIGNING.md` therefore retains draft/pending status and shows the attribution as conditional on acceptance. Whether a pending/application statement satisfies the form's wording before acceptance is not established by those public sources; do not assert that it does. Final policy approval and a separate authorized publication step remain required before representing the public application package as complete. Name/email, discovery source and personal consents are maintainer inputs at submission, not invented engineering prerequisites or permission to create an account now.

Research returned no new helper files to remove. No product/source behavior, runtime fixture, external account or remote state was changed. The remaining preapplication review covers the two documentation files only and reuses the previously recorded source/runtime evidence without extending its claims. There is no architecture change.
