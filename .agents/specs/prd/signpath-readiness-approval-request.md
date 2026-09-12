# SignPath readiness — maintainer approval request

Status: investigation in progress; not an approved policy, Foundation application, or release authorization.

## Authority and baseline

- Contract: https://github.com/kapitalismho/PuriPuly-heart/issues/74 (2026-09-12 revision).
- The maintainer subsequently requested execution, reserving sensitive documents and direct approval requests to the Director.
- Protected remote `dev`, local `dev`, and execution branch `apply-code-signing-by-signpath` started at `6395fd8d0f6b3711529bfcc051618b2911bbc923`; no pre-existing dirty paths; execution branch has no upstream.
- GitHub Project status was changed from Ready to In progress and read back. No issue criteria were marked complete.
- Implementation commencement does not authorize Foundation correspondence/submission, account or repository permission changes, production deletion/deployment, push, or release publication.

## Confirmed evidence

### Current behavior

The canonical telemetry preference defaults to ON. Disabling it clears its anonymous identity and suppresses app-active-day requests. Enabling it again creates a new identity. The telemetry JSON payload contains only `anonymous_id` and `active_date_utc`; ordinary HTTP and network metadata are separate from that payload. Failed sends can be retried, so a daily aggregate must not be described as a hard limit of one HTTP request per day.

The Broker stores an HMAC-derived subject reference and active date in `app_active_days`, with uniqueness per subject/date. Its retention rule deletes dates strictly earlier than UTC today minus 35 days; this can retain 36 calendar dates including today. Do not silently change that behavior to match a simplified policy phrase. The legacy `telemetry_active_days` and `telemetry_subjects` structures remain in the migration/persistence contract; current runtime aggregation does not use them. Production contents, external consumers, and deletion effects remain unverified.

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

- Forward migration `0025_drop_legacy_translation_telemetry.sql`: populated legacy tables were removed while current `app_active_days` rows and DAU/WAU/MAU results were preserved. Broker suite: 68 files, 616 tests passed; TypeScript check passed. No production operation ran. The existing deploy workflow will apply this migration on the next deployment, so deployment remains prohibited without separate deletion approval.
- Both real native release EXEs and the real PyInstaller main EXE were built with ProductName `PuriPuly <3`, ProductVersion `2.6.1` (numeric `2.6.1.0`). The generated-uninstaller metadata/reuse proof remains the isolated Inno 6.6.1 probe, not an installed signed uninstaller.
- Actual final `build.spec` packaging retained the checked upstream license payloads and excluded the ASIO-only DLL. An isolated frozen sounddevice probe loaded ordinary PortAudio and enumerated MME, DirectSound, WASAPI, and WDM-KS without loading ASIO.
- The LGPL companion ZIP includes the two soxr source archives, the applied python-soxr patch, zeroconf source, and a provenance manifest. SHA-256: `14346f3534be424ce44e2919800dcb04d74794a623b666b5b44961beae7325cb`. Archive hashes and patch reconstruction passed.
- Director integration: `pytest tests/release_evidence tests/app/test_release_dependency_guards.py tests/core/test_soxr_runtime.py tests/config/test_public_compatibility_surfaces.py -q` passed. Ruff and Black checks on the new metadata helper passed.
- Director compiled the complete installer with Inno Setup 6.6.1, then extracted it without installation using innounp 2.71.1. All 1,137 expected payload files matched the final dist/staged inputs byte for byte; the ASIO-only DLL was absent and ordinary PortAudio present. The embedded companion ZIP matched the hash above.
- Integrated unsigned installer: `installer_output/PuriPulyHeart-Setup-2.6.1.exe`, 173,766,150 bytes, SHA-256 `0396a9182ff1171b54d76fe818c964d2a3c75338d0717f46c6e2923892b1f408`. This is a local test artifact, not a release candidate approved for publication.

The current checkpoint does not implement installer privacy UI, localized policy text, SignPath workflow requests, signature verification, or a signed release. Full policy approval, infrastructure evidence, libmpv corresponding-source resolution, account/MFA setup, production deletion approval, and end-to-end signed validation remain separate unmet criteria.
