# OVR-COMPAT-1 — Final policy and software evidence

## Authority and scope

Authority: [#152 OVR-R](https://github.com/kapitalismho/PuriPuly-heart/issues/152), under #145, consuming #146/#147 and the #151 compatibility-preserving baseline. The local `ovr-contract-1.md` **r2** amendment supersedes the historical r1 native-lease clauses. Its SHA256 is `e445fdc6621efddba63f683aa1199b3d40575b1a5160c5cf2228739ae6f2efa9`.

**R-OWNER-1:** retain native P05 fresh-render protection and the D3D11/OpenVR backend; remove the reachable Python retry scheduler, ownership handoff, and cadence-only nonce metadata. No retry-count, duration, backend, or physical-fresh-work reduction is selected.

**R-SCOPE-1, maintainer-approved 2026-09-13:** when asked about post-change package installation/paired rollback and actual deployed source/binary linkage, the maintainer answered `이거 151에서 한거 아니야? 이건 내 직권으로 스킵`. [Published amendment](https://github.com/kapitalismho/PuriPuly-heart/issues/152#issuecomment-5650872179). These external parts of OR07/OR08/R8 are **WAIVED, NOT RUN / NOT PROVIDED**, not passed. #151 installation/restoration evidence remains historical baseline evidence, not proof that this Python delta was installed. Software epoch safety, one native owner, protocol correctness, delta verification, independent review, source identity, and retained physical protection remain required.

Publication of the result in #152 and linkage in #145 was separately approved. Push, merge, release, deployment, and issue closure are not authorized by that approval. Final review/Director acceptance is recorded in #152 against an exact committed candidate; this document does not manufacture a future review verdict.

The earlier #151 agreement removed affected/control comparison and deferred HMD/4–6 h use to user prerelease verification. It is not a physical pass and does not waive software correctness here. No new affected-environment support exclusion is introduced.

## Baseline and source/binary identity

| Record | Identity and evidence boundary |
| --- | --- |
| Historical assessment / publication dev | `1d92e5c3301fcaf7619f3a9521781627f310cee4` / `4e967df9d03649106faa8348c3ec611009529ffe`; historical anchors, not the retirement baseline |
| #151 product source | `a38626e4e4c3fea7dbebbfd41afd9dfdef04231f`; Python/native product code was unchanged through #152 work-start |
| #151 reviewed acceptance record | `ebacb547dd6dd0fa84ebe83b14da218f188ce169`; software/current-pipeline/isolated-package scope, not production or universal certification |
| #151 final canonical package source | `2451a58b3998d2dd9860c516af8121664e955905`; V12 full frozen-package lifecycle passed in 8m50s, 120 tests; qualifications remain in `ovr-integrated-acceptance.md` |
| #152 work-start | `5d481cf10c5608178c797d82132385c247beee2f`, branch `ovr-0-vr-overlay-reliability-program-cross-envir`; clean, upstream equal at start |
| Initial cutover candidate | `752b88d13aaed702c3467e9bfdf70a1ac09c65f6`; independent checkpoint reviewed |
| Final product-source repair | `f95297e039732ab8aa8a4cd4ace3f2c0abd6a67f`; adds preserved-caption expiry repair and removes obsolete request metadata |
| Effective pair contract | App 2.6.1; protocol **8**; `execution_contract {version: 1, revision: "r2"}`; `native_presentation_retry {version: 1, ownership: "exclusive"}` |
| Production selection | Windows x64, D3D11 `TextureType_DirectX`, SteamVR/OpenVR, head/spatial modes; **P05**, handoff experiment **off**; desktop has no VR retry schedule |

Keep the binary lanes separate:

- Measurement/live native executable: SHA256 `5c1597a751b230f2214479d05d80568420c6d503fb8e0bed9bc88ebbd5af927a`, native source `a38626e4…`. This is not the final installer executable or a verified production startup pair.
- Intermediate V10 installer overlay: `1718edda…`; final V12 overlay: `f12b807a…`, app `3489c466…`, release installer `9a618850…`. Only these prefixes are available in the committed receipts; no full hashes are invented. Different builds of unchanged native source are not interchangeable binary evidence.
- OpenVR DLL: SHA256 `bab8ac6ef64e68a9ca53315b0014d131088584b2efdfa6db511d67ec03cfcb4a`, vendored OpenVR v2.15.6.
- Final native production source/backend is unchanged by #152. New Python source is identified above; **no newly installed Python/native pair is claimed**. Exact measurement-build rustc flags/build-host identity and a complete final package hash remain unavailable here.

#151's exact-current-final-binary DirectWrite collection and runtime OpenVR font observation were not exercised. Earlier same-native-source/same-TTC corroboration is not upgraded. Its V9 AppData loss remains historical FAILED/unrecovered/user-waived; later safety checks did not restore the lost data. Older-release upgrade and real-AppId downgrade remain unverified baseline limitations.

## Six-mechanism inventory and final disposition

### 1. Fresh rendering work — RETAIN

- **Reachability/support:** the normal matched VR pair renders fresh under experiment `off`; desktop does not use the native GPU path. Sources: `native/overlay/src/runtime.rs` (`PresentationRuntime`), `renderer/backend.rs`, `openvr.rs`.
- **Purpose/failure coverage:** preserve the historical fresh-work protection across producer/interop/dispatch boundaries. Fresh bursts helped in field narratives; `d3cd13a`/`a38add3` also changed allocation/rendering and isolate neither fresh texture identity nor submit-only sufficiency. #147 performed no Flush-ON leg. The exact protected GPU condition remains unresolved.
- **Owner/cost:** native `PresentationRuntime`/`NativePresentationOwner`; Python emits intent only. Retain one incomplete producer/query plus one CPU successor, persistent 4096×1056 BGRA8 target (nominal 16.5 MiB), readiness query/Flush and 50 ms readiness budget. Driver/compositor allocations are not inferred from this bound.
- **Removal prerequisite:** an API/correctness basis and relevant affected-scope evidence for equivalent protection. Neither exists; fewer Python publications are not evidence for removing fresh GPU work.

### 2. Quiet-tail cadence/window/count — RETAIN

- **Reachability/owner:** native `NativeFreshRetryPolicy`/`NativeFreshSchedule`, selected by `DefaultOverlayProcessRunner`; one schedule per channel. Desktop has neither schedule.
- **Policy/cost:** P05 100 ms cadence, at most five final opportunities over 500 ms **per channel/episode**; stream `min(4, profile maximum)`. Same-episode reconciliation preserves consumption/deadline; final transition starts its final episode. Target retirement and preemption can reduce completed attempts; no catch-up storm or content-age renewal.
- **Failure coverage:** retain historical spread protection, not a claim that P05 is optimal. Scheduled, due, completed, cancelled, and compositor-consumed counts are different. Actual affected-session wall-clock distribution/physical consumption was not measured here.
- **Removal prerequisite:** decision-changing same-environment comparison isolating the necessary dimension. P05/P20 change count and spread together; no count/window reduction is bundled with fallback retirement.

### 3. Generation/target/episode/nonce — RETAIN native intent, REMOVE cadence-only nonce

- **Retained live interface:** `native_fresh_render_generations`, `native_fresh_render_targets`, `native_quiet_tail_episodes`, cause/currentness and semantic-retirement identities. Python presenter owns intent emission; native owns reconciliation/scheduling. They distinguish logical turn/appearance from presentation intent without changing caption age or reanchoring for retries.
- **Removed interface:** Python `begin/tick/end_*_presentation_refresh` state and `session_scope` refresh markers. They existed to make each Python cadence tick revision-worthy; their only scheduler is retired.
- **Cost/coverage:** retain bounded per-channel intent metadata, remove marker scans and cadence-only revisions. Fresh epoch seeds exact live targets at generation 1: finalized caption → final episode; eligible nonfinal caption → stream episode, not old consumed credits.
- **Removal prerequisite:** nonce retirement is coupled to verified native ownership-before-attach and valid current-state replay. Further native-intent removal would need replacement coverage; none is selected.

### 4. Python fallback/native ownership handoff — REMOVE

- **Actual baseline reachability:** generation startup attached the bridge and output sink before awaiting process readiness. Ownership `false` enabled Python 100 ms burst tasks; native authenticated before its separate ready event. Preserved-presenter startup and post-exit callbacks could also restart Python tasks. This was executable scheduling, not a dead flag.
- **Why remove:** protocol 8/r2 already rejects old/missing exclusive capability. This fallback was not a supported old-package mode and violated native-exclusive-before-attach. Old matched rollback artifacts do not require a parallel scheduler in current source.
- **Final ownership:** `OverlayGenerationStartOwner` calls `begin_native_retry_epoch(enabled=not desktop)` before bridge construction/attachment. The presenter emits current intent; only native schedules VR retries. `OverlayProcessManager` owns lifecycle/compatibility checks, not scheduler selection.
- **Removed cost/surfaces:** per-channel burst tasks, start/tick/cancel bookkeeping, ownership callbacks, nonce state, obsolete request/export/resource declarations. All callers, including the measurement script, migrated without compatibility aliases.
- **Correctness:** fresh-epoch publication prunes expired state, seeds valid current captions, and rearms expiration under the new runtime using original stored deadlines. It does not renew caption age or leave duplicate live timers. Desktop omits VR intent; inexact capabilities fail `unsupported_binary`.

### 5. Old backend/experimental profiles — RETAIN diagnostic-only options, no new production backend

- **Reachability/owner:** normal Python runner pins P05/off; native environment selection retains diagnostic profiles and opt-in `cached_frame_rehandoff`. Native manifest resolution owns validation. Desktop rejects non-off experiments.
- **Purpose/cost/coverage:** keep bounded comparison controls and policy rollback options, not a profile menu or automatic vendor tuning. Cached rehandoff requires matching current scene/raster/presentation/device identity and completed producer work; it is not fresh-render conformance and cannot refill recovery allowance. Enum/parse/diagnostic code remains; runtime cost is not claimed to be zero when an experiment is enabled.
- **Disposition:** r2 explicitly retains this experiment; #150 was not required. No post-submit Flush, shared-export backend, or alternate production owner is adopted. Removal would need a separate supported-comparison/rollback decision; speculative simplification is insufficient.

### 6. Legacy tests/diagnostics — RETAIN behavioral protections, REMOVE retired cadence fixtures

- **Removed:** `tests/helpers/overlay_refresh_trace.py`, `tests/core/test_overlay_refresh_trace_contract.py`, `native/overlay/tests/fixtures/refresh_traces.json`, eight nonce-fixture native tests/helpers, and their CI byte-check step. They described the removed Python cadence/nonce contract; byte-pinning it would preserve the wrong owner.
- **Retained/new coverage:** native P05 bounds, currentness, cancellation, placement and recovery tests; Python semantic/TTL/epoch/capability behavior; actual-runtime expiry regressions; a bounded Python-produced payload → native parser/reducer/owner smoke. No new persistent fixture generator or product instrumentation was added.
- **Owner/cost:** owning test suites and bounded diagnostic recorders. Retain native 128-record/4 KiB-per-record rings, bounded flush/drop accounting; logs are not an operational success API or guaranteed lossless delivery.
- **Not selected:** unrelated native external-retry test hooks and integration-test helper restructuring. No remaining old Python production retry owner is preserved by this decision.

## Verification and measured benefit

All measurements are software-layer evidence unless explicitly labeled otherwise.

1. **Pre-fix scheduler reproduction:** `test_native_retry_mode_never_schedules_python_refresh_work` failed with sleeps `[8.0, 0.1]`: the 8 s TTL was valid, the 100 ms Python cadence was the defect. Post-cutover it passes with one publication and no Python cadence.
2. **Final shared integration run at product repair:** 14 files spanning presenter, generation/application lifecycle, bridge/runtime, translation/output callers, desktop renderer/settings, process manager and measurement harness: **557 passed, 2 skipped, 266 existing ElevatedButton deprecation warnings**, 24.74 s (`artifact://364`). The skipped real-subprocess transport scenarios are not a live-process pass. Exact command:

   ```text
   uv run pytest -o addopts= tests/core/test_overlay_presenter.py tests/app/test_overlay_generation_start_owner.py tests/app/test_overlay_application_transitions.py tests/core/test_overlay_bridge.py tests/core/runtime/test_overlay_runtime.py tests/app/test_overlay_translation_enabled_sync.py tests/core/test_dual_target_translation_lifecycle.py tests/core/test_output_owner_wiring.py tests/core/test_soniox_multilingual_release_readiness.py tests/core/test_translation_output_streaming.py tests/ui/test_desktop_overlay_renderer.py tests/app/test_overlay_process_manager.py tests/config/test_overlay_desktop_audio_settings.py tests/scripts/test_ovr_hmd_measurement.py
   ```
3. **Retained native checks:** `cargo test --locked --manifest-path native/overlay/Cargo.toml --test runtime`: **94 passed**. Native production source is unchanged. Full process-manager **101 passed** and harness **24 passed** are included in the shared coverage; Ruff passed on changed Python surfaces.
4. **Expiry regression:** actual old-runtime close preserving presenter → new-runtime adoption at t=11 after final at t=10/deadline18. Pre-repair no replacement timer (`[8.0]`); repaired one 7 s timer (`[8.0, 7.0]`) removes the caption at original t=18. A second case restarts at t=41 and produces no caption, target or expiry task. No native lease is implied.
5. **Cross-language smoke:** actual presenter/adapter produced revision3 with SELF final + PEER stream, generations1/1 and exact live targets. Rust serde parsed it; `OverlayRuntime` returned Applied; `NativePresentationOwner` retained revision3, both targets/generations and two current blocks. **1 passed** via temporary `fresh_epoch_cross_language_smoke`. This proves parsing/reduction/owner initialization, **not GPU scheduling, handoff or HMD consumption**. Producer/payload/smoke files were removed afterward.
6. **Python loopback/offline:** actual presenter/bridge websocket replay and full nine-event offline sequence passed with cleanup complete. No native process was negotiated/launched. The offline proof substituted the prepared-stage loader and shortened waits; it did not verify an installed binary or a genuine 30 s idle. Physical observation is `not_observable`.
7. **Harness provenance:** report schema v3 keeps immutable `prepared_stage` provenance separate from the current eight-file Python SHA256 set. The checkpoint aggregate `6573b274…` belongs to the pre-expiry-repair execution, not the final source. Historical v2 reports remain historical; the current observation command requires v3. Old staged identities are not relabeled as new Python proof.

**Measured control-plane benefit at checkpoint `752b88d1` versus baseline `5d481cf1`:** isolated real baseline/current presenters, identical fixed SELF final/head-locked input, controlled clock, native mode before ready, same final-emit-through-scheduler-completion window excluding cleanup. Publications **23 → 1** (22 fewer, 95.7%); Python 100 ms sleeps **21 → 0**; both retain native intent generation1. This replaced an earlier fixture-derived baseline with actual baseline-owner execution. Six specified product owner/state/process/start/application/runtime files had **8,687 → 7,896 source lines (-791)** at that checkpoint, not cyclomatic complexity and not the final post-repair line count. No GPU, OpenVR submission, HMD, latency, CPU utilization, power, or total-memory benefit is asserted.

Failure history remains visible: the first shared run had **6 failures / 549 passed / 2 skipped** because the measurement script still called removed constructor fields. Caller migration repaired all six. A process test raced a 10 ms startup timeout against a synthetic renderer error; test stimulus synchronization separated those behaviors, and the full 101-test file passed. Checkpoint integration then passed555/2 before the two expiry regressions produced the final557/2.

## OR01–OR08 disposition

| ID | Evidence/disposition |
| --- | --- |
| OR01 | Six rows above have retain/remove decisions, rationale, owner, reachability, cost, failure coverage and removal prerequisites |
| OR02 | Sparse/static/preserved generation shown at Python/bridge and native parser/reducer/owner boundaries; no physical-HMD claim |
| OR03 | Evidence-based RETAIN of native P05/fresh work; affected and long-session physical comparison NOT RUN, not an equivalence pass |
| OR04 | Clear/OFF/current-state/epoch/expiry software tests, including original-deadline rearming and expired-at-restart non-revival |
| OR05 | Exact protocol8/r2/exclusive support, non-exact/old pair rejection, desktop separation; no old-binary fallback path |
| OR06 | Measured publication/scheduler and checkpoint source-line reduction above; no shifted native cost hidden as a GPU improvement |
| OR07 | Software epoch safety and paired rollback procedure retained; post-change installation/rollback execution **WAIVED, NOT RUN** under R-SCOPE-1 |
| OR08 | Python production scheduler/handoff removed; final product source identified; actual deployed-pair linkage **WAIVED, NOT PROVIDED** under R-SCOPE-1 |

## Rollback, residual risk and reassessment

Rollback restores a **matched Python/native/resources/protocol/backend/profile tuple**, not just source, wire version or one executable. OFF/stop ingress → retire publication/connection/process/device epochs → bounded terminate and confirm old child exit → restore recorded matched artifacts → fresh epoch/current-state revalidation, honoring OFF throughout. Unconfirmed exit means circuit-open, never overlapping children. Do not replay old ticks, UI/history/chatbox output or old episode credits.

For this delta, the comparison/rollback baseline is the #151 r2/P05 pair at `5d481cf1` lineage, which already contains application isolation/native lifetime improvements. Restoring the scoped #152 delta must not require returning the subsystem to pre-A/N protocol6 code. Native fresh work/count/window were never removed and need no restoration toggle. Actual post-change paired installation/restoration is waived; a newly exercised rollback pair is not fabricated from a source SHA or truncated package hash.

Reassess on a correlated affected trace, SteamVR/driver/API support change, runtime/device-reset change, measured resource regression, stale/lost first or final update, false recovery, or unacceptable VRChat contention. Stop promotion on an attributable regression and independently restore the relevant retained matched policy; no endless experimentation, background telemetry or vendor-wide tuning is authorized.

Important limits:

- Python remains the sole age/TTL authority under r2. Native has **no independent lease/expiry Hide guarantee**. A stalled app/lost clear or failed replacement can leave old pixels visible; health is not freshness. Withdrawn r1 safety is not passed by this work.
- Known affected environment identities, comparative driver/runtime/HMD paths, physical freshness and 4–6 h stability remain unverified. Historical retained short live exposure was 56.657 s on one local Windows22631/RX7900XTX configuration, not an affected/control or long-session trial. APIs, fences, mirrors and producer readback are not physical scanout.
- Review reproduced a **pre-existing** architecture task-allowlist failure (`test_no_new_unmanaged_task_creation_outside_lifecycle_allowlist`): bridge task count5 vs allowlist1, process8 vs3, stale output entry; counts identical at baseline/candidate. Deferred out of scope. The focused integration pass is **not whole-CI green**.
- Pre-existing native generation-without-target fallback is bounded to a valid current row and was not changed; no expired-content revival was demonstrated. Unrelated test-hook cleanup is also out of scope.
- Public evidence must exclude credentials, personal paths, HMD serials and raw conversation; synthetic content was used. No actual installation, deployment, release or HMD run was performed for #152.

## Review and evidence history

- Checkpoint range `5d481cf1..752b88d1`: two fresh read-only reviewers, lifecycle and native compatibility/evidence. Compatibility: no material findings. Lifecycle F1 (missing preserved expiry timer) **ACCEPT**, repaired in `f95297e0`; F2 (missing actual cross-language payload evidence) **ACCEPT**, smoke passed. Obsolete write-only request metadata removed. Other qualifications are dispositioned above.
- Final product-source repair `f95297e039732ab8aa8a4cd4ace3f2c0abd6a67f` passed the shared557/2 barrier. Final acceptance requires targeted repair verification and complete-Goal terminal review. Their authoritative verdict belongs in #152 against the exact committed receipt candidate; this historical checkpoint is not a substitute for that terminal verdict.
- Supporting session evidence: `local://ovr-r-validation.json` (pre-change only), `local://ovr-r-implementation.json` (implementation/repair details), `artifact://198` (checkpoint555/2), `artifact://364` (final557/2). These are supporting local artifacts, not permanent public links; the substantive outcomes above and the #152 publication remain intelligible without them.
- System ownership follows `ARCHITECTURE.md`; no new manager or backend was introduced. The duplicated Python/native scheduling found in the baseline was removed to align with the approved ownership contract. No suspected new architecture drift is introduced.
