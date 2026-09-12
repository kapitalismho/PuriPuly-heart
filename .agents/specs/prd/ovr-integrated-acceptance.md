# OVR-INTEGRATED-ACCEPTANCE

## Status, authority, and decision boundary

**V1–V8 PREREGISTERED; V9 INITIAL INSTALL FAIL HISTORICAL (installer gate superseded); V10 REPAIRED CANONICAL PASS RECORDED 2026-09-12; FINAL #151 ACCEPTANCE REVIEW_PENDING — NOT ACCEPTED, NO PRODUCTION CERT.**

This local receipt implements the validation-design deliverable requested by issue #151. It consumes:

- issue #151 as the integrated/environment/long-session authority;
- `OVR-CONTRACT-1 r2`, approved through the #149 completion record, whole-file SHA256 `e445fdc6621efddba63f683aa1199b3d40575b1a5160c5cf2228739ae6f2efa9`;
- #147 comment `5607995416`, `OVR-HANDOFF-1 r1`: **evidence-limited retain** of the D3D11 DirectX API-only OpenVR handoff and unchanged production P05 fresh-render quiet-tail;
- #149 comment `5644262150`: native lifetime/progress/recovery software acceptance under r2 and its explicitly limited live observation;
- #150 comment `5644273637`: conditional backend work closed **not required**, not implemented or accepted as a new backend;
- `.agents/specs/prd/ovr-application-acceptance.md`, which remains an accepted application-source receipt with an r2 note, not a reissued whole-#148 matched-pair acceptance;
- `.agents/specs/prd/ovr-native-acceptance.md` and `ARCHITECTURE.md`.

The frozen source scope for this receipt is product source `a38626e4e4c3fea7dbebbfd41afd9dfdef04231f`, independently reviewed through `c32adc94dcc1573b11a567d37a09a4ec9730024d`, with only measurement pins/acceptance documentation afterward through the clean receipt baseline `9e775809877f7074205d98911eb99f8db8641d32`. Unrelated newer development, including an origin/dev line reported two commits ahead with #154 desktop-startup and DeepSeek changes, is excluded and is not integrated by this receipt.

No criterion below restores the withdrawn r1 native-expiry guarantee. Under r2, Python is the sole caption-age/TTL authority. The former native maximum-three-second validity guarantee and expiry-triggered Hide are **withdrawn, not passed**. An application/transport failure or failed replacement can therefore leave old captions displayed; health is not proof of freshness.

## Scope amendment 2026-09-12 — local authority (current #151 scope)

Dated local authority: 2026-09-12. User decision for this local agreement: `1번은 환경상 어려우니 스킵. 2번과 3번은 릴리즈 전에 내가 직접 해볼게. 이상 있으면 내가 보고할테니 이것도 일단 스킵` — (1) affected-vs-normal environment comparison; (2) physical HMD observation; (3) 4–6 h continuous use.

| # | Original obligation | Current disposition for #151 completion | Status / rationale |
| --- | --- | --- | --- |
| 1 | Affected/control comparison with matched pair, counterbalanced exposure, and threshold-gated improvement claim | REMOVED from #151 scope | Access constraints. Unmet, not passed. No environmental fix claim; no reproduction reclassified. Creates no user obligation. |
| 2 | Physical HMD observation (through-lens / operator) for field behavior | DEFERRED to user prerelease verification | Removed from #151 agent gates. Not executed / not passed; silence is not evidence; no physical, production, or certification claim follows from #151 completion. |
| 3 | Continuous 4–6 h use with matched load/logging and resource-trend judgment | DEFERRED to user prerelease verification | Same status as #2. The user is not obligated to follow any formal measurement or threshold protocol; V2.2 figures and §V5 are optional guidance only. |

Retained for #151 completion: product protections and contract r2/P05 semantics (V2.1 and unchanged r2/P05 terms); install/font-pair identity and parity; paired rollback authorization, exact prior installed reference tuple, and paired rollback/restart receipt alignment; the #148 receipt-alignment gap with current-pipeline-only scope labeling. No support-platform or product-code change is made or claimed.

Separation: **#151 agent completion is distinct from the user's prerelease verification and from any release approval.** #151 completion does not certify physical behavior, long-session stability, production readiness, or release fitness, and does not authorize release, push, issue closure, or publication. Remote issue text (#145/#146/#147/#148/#149/#150/#151/#152) is unchanged pending publication approval.

History: the original field obligations are preserved unchanged in §V5 (marked HISTORICAL) and the V2.2 table below. No other section retains them as gates, so this document reads correctly alone.


## V1. Reproducible candidate, effective behavior, and scope

| Field | Preregistered value / interpretation |
| --- | --- |
| Initial assessment / publication dev | `1d92e5c3301fcaf7619f3a9521781627f310cee4` / `4e967df9d03649106faa8348c3ec611009529ffe`; historical only |
| Product source | Native and Python `a38626e4e4c3fea7dbebbfd41afd9dfdef04231f` — native/Python product UNCHANGED. Verified canonical repair COMMITTED `a6c4505cc31f812bb23be6cf2de319ba8b6f0d02` (5 files: `build.spec`, `installer.iss`, `scripts/ci/build-release-artifacts.ps1`, `tests/app/test_release_dependency_guards.py`, `tests/config/public_compatibility_surfaces_snapshot.json`; source hashes per `local://v151-installer-repair.json` match this candidate). |
| Independent review | `c32adc94dcc1573b11a567d37a09a4ec9730024d`; complete-Goal accepted for the scoped r2 software candidate |
| Receipt baseline | `9e775809877f7074205d98911eb99f8db8641d32`; documentation only |
| Native executable | Windows x64 release, app 2.6.1, SHA256 `5c1597a751b230f2214479d05d80568420c6d503fb8e0bed9bc88ebbd5af927a`. Exact rustc flags and build-host identity are unknown; no app-package/archive SHA256 is available. |
| OpenVR DLL | SHA256 `bab8ac6ef64e68a9ca53315b0014d131088584b2efdfa6db511d67ec03cfcb4a`; vendored Valve OpenVR v2.15.6 |
| Negotiated pair | Protocol 8; `execution_contract {revision: r2, version: 1}`; `native_presentation_retry {ownership: exclusive, version: 1}` |
| Backend / retry | D3D11 `TextureType_DirectX` OpenVR API-only handoff; P05: 100 ms cadence, at most five final opportunities over 500 ms; Python legacy retry absent on this matched pair |
| Experiment | `PURIPULY_OVERLAY_HANDOFF_EXPERIMENT=off`; cached-frame rehandoff is opt-in experiment-only and does not qualify as r2 fresh-render conformance or refill recovery allowance |
| Calibration/logging in retained stage | SteamVR target, head-locked, x/y `0/-0.45`, distance `1.1`, scale `1.0`, background alpha `0.24`, detailed logging |
| Prepared stage | `20260911T215539Z-6bc9c94b`; measurement script SHA256 `340febb185e23b630d975490888c574788d461e0f9146f87f20ae918c677e20e` |
| Stage/install relationship | Original `20260911T215539Z-6bc9c94b` stage was an isolated out-of-tree test executable (HISTORICAL). Repaired canonical evidence is a full isolated testAppId install/reinstall/uninstall lifecycle (see V10) — still not the production startup pair. |
| Production startup pair | **Production pair still UNKNOWN / absent — no production install claim.** Repaired V10 lifecycle used an isolated testAppId; production sentinel survived and production data/registry/shortcuts are byte-identical to the PRE-REPAIR VALIDATION baseline (post data-loss), not a production install. |
| Current local environment | Windows 11 build 22631; AMD Radeon RX 7900 XTX, driver `32.0.23033.1002`; SteamVR installed build `25216780`, channel unknown, not running at preregistration; operator readiness **not confirmed**; no live launch authorized |
| HMD / driver / runtime path | The retained short run is one local configuration. Affected-session HMD, connection path, renderer/OpenVR adapter identity relationship, and comparable control environment remain unobserved or unrecorded. No vendor-wide inference is permitted. Runtime OpenVR font path still NOT OBSERVED (no HMD) after repair. |
| Font/package parity | Pre-repair live diagnostics `collection_load_failed` with system-font fallback: HISTORICAL. Repaired testAppId install: production-loader DirectWrite collection PASS against installed bytes (earlier evidence) + current hash/resource chain (see V10); runtime OpenVR font diagnostic NOT OBSERVED (no HMD). |
| Consumer scope | Current landed translation/output pipeline only. #134/#135/#143/#144 closure does not mean their Audio acceptance branch or merged recognition seam is present in this frozen HEAD; later Audio/current-dev integration is unverified. |

Public evidence must use session-local aliases for adapters/resources. It must not collect HMD serials, credentials, persistent device identifiers, user paths, raw conversation, or nearby-person audio/video. Synthetic captions are required unless the operator separately consents to a narrower recording. App telemetry consent is not diagnostic/recording consent.

## V2. Frozen criteria, denominators, exclusions, and stop rules

### V2.1 Correctness criteria that are already normative

The following are pass/fail contract requirements, not field thresholds that this receipt may tune:

1. Preserve r2 ownership/currentness: local application receipt, delivery admission/write, native acceptance, producer readiness, runtime API return, observed runtime visibility, and physical observation remain separate stages. Missing evidence is `not_observed` or `not_observable`, never success.
2. Preserve semantic identity/pairing, provisional→final ordering, identical-text new-turn identity, eligible destinations, two-row current-caption behavior, manual/SELF/peer isolation, and desktop parity.
3. Preserve bounded application/native resources and exact overload terminality already approved in r2, including 8 unsent + 1 active per output scope/destination, presenter 64 entries/16 MiB, bridge current+active+successor/3 MiB, one native incomplete producer/query plus one CPU successor, 64 MiB accounted renderer-cache ceiling, bounded diagnostic/control rings, and no unbounded history/retry queue.
4. Preserve existing normative deadlines/budgets: 5 s scene write; 1 s control write/close; 50 ms native readiness; 2 s due-work no-progress; challenged owner freshness through issuance +3 s; 2 s native-acceptance deadline; 15 s startup; shutdown 3 s graceful then 1 s terminate and 2 s kill-confirmation; 1 s reader cleanup; initial child plus at most three replacements in a 60 s recovery episode, with refill only after 60 continuous qualifying seconds.
5. Preserve P05 exactly. Every retry is an attempt, not a new logical-update denominator and not proof of physical compositor consumption.
6. Preserve r2 expiry semantics: Python TTL/send-time pruning/ordered clear and replay remain required; native expiry/lease checking is absent. The withdrawn r1 safety guarantee cannot be used as a failure-free claim.
7. A software run passes only with every scenario guard and expected receipt satisfied, graceful shutdown acknowledged, confirmed child exit 0, no forced termination or terminal cause, reader/owner cleanup complete, and manager off. A manual report cannot upgrade a failed software run.
8. API success is not physical success. `SetOverlayTexture`, producer readiness, `IsOverlayVisible`, mirror/readback, and through-lens/operator observation retain their own labels.

### V2.2 Historical proposed field thresholds (optional guidance only; not #151 gates)

These values are preserved history: preregistered proposals, not approved requirements and not claims about existing evidence. They are optional guidance for the user's manual prerelease verification only. No approval is required before that manual verification, the user is not obligated to follow the measurement protocol, and results must not be used to loosen the figures afterward. Numerical values below are unchanged.

| Measure | Historical proposed gate |
| --- | --- |
| Application acceptance latency | p95 ≤ 50 ms and worst ≤ 250 ms from caller admission to local application receipt, excluding deliberately blocked upstream provider time |
| Application receipt → native current-state acceptance | p95 ≤ 100 ms and worst ≤ 500 ms while connected and healthy; any normative 2 s acceptance-deadline breach is already a correctness failure |
| Native accepted → successful current handoff | p95 ≤ 100 ms and worst ≤ 500 ms for ordinary non-fault updates; injected fault/recovery cases are judged by normative recovery bounds instead |
| Correlated intended update → physical observation | If a defensible clock mapping exists: p95 ≤ 250 ms and worst ≤ 1,000 ms. Without it, record qualitative correctness only and leave latency `not_measured`. |
| Private bytes after a 30-minute warmup | Least-squares trend over the remaining session ≤ 8 MiB/hour and final-hour median ≤ warmup-end median + 64 MiB |
| Observable GPU allocation after warmup | Trend ≤ 8 MiB/hour and final-hour median ≤ warmup-end median + 64 MiB; unavailable counters are `not_observable`, never zero |
| Process handles / owned live texture-query-task-ID-queue counts | No positive post-warmup trend across hourly windows; all contract-owned counts remain at/below their normative caps. Driver-private/compositor objects are not inferred from software counts. |
| CPU/GPU utilization regression | Candidate/session p95 must not exceed the matched control by more than 10 percentage points under the same workload; absolute utilization is descriptive, not a substitute for progress/correctness |

Any accepted correctness failure overrides performance percentiles. Aggregate percentiles must retain the worst incidents with stage/failure-family classification.

The combined latency proposals do not replace the six required stage measurements: application acceptance, delivery through transport-write return, native apply, render/producer completion, handoff API return, and observed HMD. Record each stage delta separately with eligible denominators and worst-incident failure-family classification. Unavailable boundaries or unsupported cross-clock mappings remain `not_measured`; no aggregate may substitute for them.


### V2.3 Denominator and failure definition

The primary denominator is **logical obligations**, never diagnostic records, retry attempts, frames, or elapsed seconds:

- one update obligation for each preregistered logical caption revision expected to become current;
- one invalidation obligation for each clear/OFF/retirement expected to remove authority;
- one lifecycle obligation for each requested start/restart/stop transition;
- one fault-recovery obligation for each preregistered injected fault episode;
- one physical observation window per logical update/invalidation selected for observation.

For every environment/session report: `failures / eligible obligations`, split by scenario and stage, plus logical update count, P05 scheduled/completed/cancelled attempts, actual handoffs, idle ratio, restart count, elapsed exposure, and workload. Coalesced intermediate revisions remain application-semantic evidence but are not fabricated as native/HMD observations. Diagnostic duplicates sharing one revision/attempt/visibility observation count once. In particular, the retained run's seven failure-labelled diagnostic copies represent four distinct visibility-observation events, not seven independent failures.

A correctness failure is any wrong/missing/stale identity or text, prohibited route, unauthorized revival, required clear/OFF not taking effect at its applicable software/API stage, false success, normative bound/deadline breach, unbounded growth, duplicate child/overlay, unconfirmed teardown, or physical stale/missing/wrong/placement/flicker incident in an eligible observation window. Under r2, continued stale display caused solely by unavailable application removal is a known withdrawn-safety gap; record it as such, not as a passed lease criterion.

### V2.4 Exclusions and session validity

Historical paired-comparison method (comparison REMOVED from #151 scope; preserved here, not a gate): exclusion removed a session from a paired comparison only for a preregistered confounder: candidate/source/hash/profile mismatch; production/other overlay already running; runtime/driver auto-update between paired legs; workload/recording/logging mismatch; missing preparation or corrupt artifact; observer unable to complete the stated method; or an unrelated machine failure that prevents the scenario from beginning. A product startup/runtime/cleanup failure after the scenario begins is a failure, not an exclusion. A failed baseline reproduction is not excluded: it makes the physical-improvement comparison inconclusive.
For any agent-run session or evidence receipt, attempted runs — including failed or aborted runs — are retained, not hidden. This recordkeeping rule imposes no formal logging obligation on the user's personal prerelease manual check.

Stop immediately and preserve evidence on wrong binary/capability, mixed pair, duplicate overlay, inability to confirm old-child termination, resource corruption or continuing growth past a normative cap, user OFF ignored, privacy/consent breach, operator discomfort, runtime/driver instability, or any correctness regression. The historical `stop the affected/control pair if matching conditions cannot be restored` rule is moot for #151 completion because the comparison is removed. Do not launch SteamVR, VRChat, or the overlay automatically; do not kill a preexisting overlay.

## V3. OV01–OV12 scenario/check map

`Reusable` means evidence remains applicable because no product code changed after the reviewed `a38626e4` candidate; it does not broaden the evidence layer. `Required` means the listed observation is still needed for #151 completion; clauses marked DEFERRED or REMOVED are not #151 gates (see the amendment table), while unmarked clauses in the same cell — such as the OV03 font-bundle requirement — remain in force.


| OV | Exact scenario/checks | Reusable evidence | Remaining execution / acceptance rule |
| --- | --- | --- | --- |
| OV01 | Clear; first SELF provisional; same-turn final; delayed translation; add PEER; remove SELF while PEER remains; close PEER; TTL empty; ≥30 s genuine input idle; second SELF update; final clear. Marker changes only on a logical revision. | `ov01-short-r2` live off run: 9 injected events, 3 s holds, 30.016 s input idle, software pass; qualitative no-flicker only. | Long-session repetition DEFERRED to user prerelease verification (no formal protocol required); comparison REMOVED. Not a #151 gate. Each intended logical revision is one denominator; no retry counter in pixels. Confirm software stages separately from physical presence. |
| OV02 | SELF and PEER provisional→final, delayed translation, then a distinct new turn with byte-identical text. Check source/translation pairing, turn/appearance identity, current-caption suppression of late older parent, and route eligibility. | Application receipt owner/currentness coverage and r2 software suites. | Current-pipeline field sequence DEFERRED to user prerelease verification; comparison REMOVED. Not a #151 gate. Observed physical wrong/stale text is a failure where observed; unobserved physical behavior is `not_observed`, never success. Later Audio recognition branch remains unverified. |
| OV03 | Replace both displayed rows; toggle source/secondary preference; exercise mixed-language bundled font, alpha, clipping, and minimum/maximum supported text scale. | Reducer/renderer software coverage only; retained live run used system-font fallback. | Installed/package font bundle is required (retained #151 gate). Screenshots/mirror are application/mirror proxies only. HMD sharpness/clipping/layout observation DEFERRED to user prerelease verification; not a #151 gate. Font fallback cannot pass package parity. |
| OV04 | Head-locked and spatial-locked runs; same-turn update/retry, turn reentry, refresh, pose unavailable then recovery/recenter. Check no retry-driven reanchor and preserved semantic pose. | Native state/runtime harness and reviewed ON06/ON07 software evidence. | Physical spatial tracking/reentry observation DEFERRED to user prerelease verification; comparison REMOVED. Not a #151 gate. Runtime/API state and physical pose are separate. |
| OV05 | Caption OFF; TALK OFF; LISTEN OFF; concurrent manual SELF plus PEER/dual-target output. Check speech-origin retirement only, manual/other channel survives, peer never reaches chatbox, and OFF rejects late old generation. | Application OA06/OA08 and native OFF/old-epoch suites. | No new execution prerequisite is created: existing reviewed application/native evidence is reusable within the current-pipeline claim boundary. Do not claim later #134/#135/#143/#144 recognition integration; it is absent from this HEAD. |
| OV06 | Stop websocket consumer; stall native stdout/stderr; flood reverse diagnostics/control; stall logging export. Check translation/UI/chatbox/application progress, current-state coalescing, explicit delivery disposition, bounded bytes/tasks/queues, reserved lifecycle controls, and finite stop. | Actual stopped-reader websocket, actual Windows unread child pipes, diagnostics and application pressure suites. | Reuse for software acceptance if validation confirms applicability. Field session records drop/omission counters; ring/logger zero does not prove lossless terminal delivery. |
| OV07 | Readiness-preemption flood; health/control flood; transient query lateness; permanent query/device/runtime failure. Check newest due work completes or finitely fails, one producer+successor bound, current-generation credit only, preserved first cause, and bounded recovery. | Reviewed ON01–ON05, including real Windows D3D query late completion; OpenVR fault paths used test doubles. | No destructive GPU fault injection during an ordinary field leg. A separately approved safe fault session is required for actual SteamVR/API claims; host/driver-wide recovery is never guaranteed. |
| OV08 | External Hide; dashboard/runtime visibility change; HMD sleep/wake; tracking loss/recovery; runtime exit/restart. Check desired vs observed visibility and no API-success/physical-success conflation. | Software visibility reconciliation and state harness only. | SteamVR + HMD execution DEFERRED to user prerelease verification; comparison REMOVED. Not a #151 gate. Observation timestamps include uncertainty; mirror is not through-lens. |
| OV09 | Child crash; synchronous hang simulation; ready flap; OFF during restart; late old-epoch event. Check initial+3/60 s budget, 60 s qualifying refill, terminate→kill confirmation, no replacement before exit, no duplicate child/overlay/history replay. | Native/Python supervisor suites and actual subprocess shutdown evidence. | Safe controlled field crash may establish API/environment behavior; do not inject a driver-wide hang. Any unconfirmed exit/duplicate is an immediate stop. |
| OV10 | Delayed TTL expiry/clear while transport is stalled, disconnect/reconnect, current replay, removal while another row survives. Check original age, no stale revival/age reset, current-only replay, and no whole-overlay Hide for surviving row. | r2 real-process fake-GPU smoke: >5 s no autonomous expiry Hide, SELF+PEER→PEER no Hide, empty Hide ≈511.588 ms; application expiry/replay suites. | Judge against r2 only. Native expiry protection is absent by design; lost application removal remains a known safety gap and cannot be called passed. Field reconnect/physical behavior DEFERRED to user prerelease verification; not a #151 gate. |
| OV11 | Clean install, upgrade, paired rollback, mixed protocol/capability, desktop target, packaged SDK/font resources, and production startup selection. | Protocol 8/r2 mixed-pair rejection, desktop suites, staged exe/DLL hashes; REPAIRED canonical testAppId full install/reinstall/uninstall PASS with helper max path 167 (vs 273 pre-repair), sentinel-guarded, all checks (see V10). | **TestAppId lifecycle PASS.** Production startup selection, real-AppId installer downgrade, and runtime OpenVR font observation remain out of scope / not claimed. Font parity for installed bytes established via DirectWrite loader + hash/resource chain; HMD sharpness/clipping/layout DEFERRED to user prerelease. |
| OV12 | Continuous affected and control use for 4–6 h with matched content/load/logging; include VRChat/other-overlay load windows, idle windows, update bursts, and at least one controlled normal restart. Sample private bytes, observable GPU allocation/usage, CPU/GPU utilization, handles, textures, queries, tasks, IDs, queues, drops, updates/retries/handoffs, failures. | Deterministic caps/retirement tests only; retained live exposure was 56.657 s and is not long-session evidence. | DEFERRED to user prerelease verification (no formal protocol or threshold approval required); comparison REMOVED. Not a #151 gate. Where a user session is observed, continuing post-warmup drift, correctness regression, or hidden periodic restart fails; unobserved behavior is `not_observed`, never success. |

## V4. Evidence layers, observation, privacy, and clocks

Every result must use exactly one of these layers and may cite lower layers without promoting them:

1. **Software/local:** application acceptance, delivery disposition, native state/progress, cleanup.
2. **Windows/API:** real D3D11/OpenVR calls, query completion, `SetOverlayTexture`, `IsOverlayVisible`, process/runtime lifecycle.
3. **Proxy visual:** application screenshot, producer readback, or SteamVR mirror, named individually.
4. **Physical:** through-lens capture or operator wearing the HMD.

A physical marker changes only on a logical subtitle change. Retry attempts remain in metadata. If through-lens capture is approved, use synthetic text, frame only the HMD/mirror needed for the question, state resolution/frame rate/clock source, and delete or redact unrelated people/conversation before retention.

Use one host monotonic clock for local stage deltas. Map wall clock with a bracketed monotonic/wall sample at session start/end and report bracket width/drift. Native and host monotonic domains must not be subtracted without an explicit mapping. Manual observation records a bounded or qualitative uncertainty in the operator's own terms; if it cannot be bounded, physical latency is `not_measured`. No application screenshot/readback/mirror timestamp becomes an HMD timestamp.

Capture overhead is a condition. Compare detailed/basic/recording modes only when matched or explicitly classify the instrumentation change; never promote a diagnostic build automatically.

## V5. Affected/control field plan — HISTORICAL (superseded; optional guidance only)

> Historical record — superseded for #151 completion by the 2026-09-12 amendment above. Preserved unchanged below as the original field proposal and as optional guidance for the user's manual prerelease verification. The user is not obligated to follow this protocol. The protections that separate software-contract acceptance from physical-freshness claims and forbid promoting API success to physical success remain in force through V2.1.

### V5.1 Environment selection

Run one actually affected configuration and one normal control. Before either session, pin OS build, renderer and OpenVR adapter aliases/relationship, GPU/driver, SteamVR build/channel, HMD model without serial, wired/wireless/runtime path, VRChat build/load, other overlays, package/source/exe/DLL/protocol/profile/calibration/logging, font bundle result, sequence revision, and artifact checksum. The current machine may be a control only if its status is declared; it is not an affected environment merely because it is available.

### V5.2 Counterbalanced 4–6 hour exposure

Target 4–6 hours continuous use **per environment**. Use matched workload blocks and counterbalance their order where practicable:

- 30 min warmup, excluded only from resource-trend fitting, not correctness denominators;
- ordinary current-pipeline caption traffic with OV01/OV02/OV03/OV05/OV10 rotations;
- at least two ≥30 min idle periods followed by a single update;
- VRChat loaded and representative other-overlay load, separately labeled;
- head/spatial and sleep/wake/tracking blocks for OV04/OV08;
- one controlled application/native restart block for OV09, never a driver-wide hang;
- final 60 min without a forced periodic restart for resource-trend judgment.

Sample process/resource counters at 10 s cadence and at scenario boundaries. Record logical update and failure denominators, retry scheduled/completed/cancelled, idle ratio, restart count, workload, worst latency incidents, diagnostic omissions, and observation coverage. A short reproducible failure stops the leg and is retained; it does not require hours of repeated exposure. If the affected baseline does not reproduce, the improvement claim is inconclusive even if the candidate has zero events.

### V5.3 Acceptance interpretation

- Deterministic + actual application integration pass permits only software-contract acceptance for the exercised scope.
- Affected baseline reproduction plus candidate improvement and relevant exposure permits a physical-freshness improvement claim for that environment/revision only.
- No affected reproduction is inconclusive, not a fix claim.
- No actual Windows/API run leaves API execution not run; test doubles remain mock conformance.
- No affected HMD leaves the field gate open; retain protections and allow only separately approved software-limited rollout.
- Any correctness, UX, resource, or latency regression returns the counterexample to the owning contract/child and stops promotion.

## V6. Current evidence ledger

| Evidence | Layer | Result that may be reused | Explicit limit / gap |
| --- | --- | --- | --- |
| `OVR-CONTRACT-1 r2` SHA256 `e445fdc...` | Authority | Current normative contract; protocol 8/r2 and application-owned expiry | #149 publishes this value as the r2 amendment source while describing it as section-only; local inspection identifies it as the whole-file hash. This receipt records that interpretation but does not rewrite the remote authority. Withdrawn native expiry guarantee is not passed. |
| Application receipt | Software/source | Application source behavior accepted; bounds/currentness/destination isolation evidenced | Source-only historical receipt with r2 note; no reissued whole-#148 matched-pair acceptance. |
| Native receipt / #149 completion | Software + limited Windows | Reviewed r2 source; 309 native tests; Python evidence and six-check real-process fake-GPU smoke; actual Windows D3D query/pipes | Fake OpenVR/HMD for integration smoke; known unchanged 10 ms supervisor fixture failure; not universal environment evidence. |
| #147 `OVR-HANDOFF-1 r1` | Decision | Evidence-limited retain; existing API handoff and P05 retained | No causal handoff fix, no affected A/B, no protection removal. |
| #150 not-required comment | Decision | No backend adoption was selected | Not an implemented backend acceptance. |
| Preparation `20260911T215539Z-6bc9c94b` | Artifact identity | Source `a38626e4`, exe/DLL/protocol/profile pins, experiment off | Isolated OS-temporary stage only; not a durable archive, installed/production startup, or font-complete package. The durable historical archive is the #149 completion comment `https://github.com/kapitalismho/PuriPuly-heart/issues/149#issuecomment-5644262150`; use the temp stage only as a corroborating local copy while it exists. |
| `run-live-off-4f70537d.json`, SHA256 `1cf1809857d0d0284867a3d03ac0fc1066364bd0b9262853176d6c851ba17314` | Software + API diagnostics | 56.657 s; fixed 3 s holds; 30.016 s true input idle; presenter/bridge/process/native seam pass; shutdown ACK, graceful completion, exit 0, cleanup complete; arm off/P05 | One short environment; no long exposure/control/VRChat load/per-frame latency. Native retention evicted 59 records; terminal delivery completeness unknown. Ring/logger zero must not be described as lossless. Seven failure-labelled copies are four distinct visibility-observation events. `collection_load_failed` used system-font fallback. |
| `observation-live-off-4f70537d.json` | Physical qualitative | Operator reported no flicker during that single run | Flicker-only, no exact timestamps, no per-caption/text/layout/placement/latency assurance, no affected-device equivalence. |
| Deterministic caps/retirement suites | Software | Contract caps and semantic retirement after warmup inputs | Not multi-hour RSS/VRAM/driver/compositor trend evidence. |
| V8 focused validation | Software/local | Lock check exit 0; harness 24 passed; authoritative arm-off dry run 9/9 applied with cleanup complete, SHA256 `efd57ab0c8415bf0aa1fc0caf13545ab1ea9e7852f560987347a4da467271aa9` | Local presenter→bridge acceptance only. No native process, OpenVR, HMD, install, cached/fresh discrimination, or long exposure. Extra disclosed same-result launcher variant SHA256 `0d488d2f6f60921f84356d763c6b013b4a0182d952c2b01abe31291d700962ed` is not the authoritative V8 run. |
| V10 repaired canonical install (2026-09-12, `local://v151-installer-repair.json`; committed `a6c4505c…`; cleanup `local://v151-installer-cleanup.json`) | Install lifecycle | Canonical `build-release-artifacts.ps1 -AppVersion 2.6.1 -InnoSetupVersion 6.6.1` result `passed`; full isolated testAppId install/reinstall/uninstall with sentinel guard, all checks; helper 1416 files, max installed path 167 (vs 273 pre-repair), `contains_elevenlabs=false` | TestAppId scope only, not production. Production data/registry/shortcuts byte-identical to PRE-REPAIR VALIDATION baseline (post data-loss), not a restore of the original 5151-file tree. Runtime OpenVR font NOT OBSERVED (no HMD). Cleanup complete: both installer exes deleted after hash+byte verification; 9 evidence-only files (2221969 B, zero executables) remain as small verification logs. |

Current acceptance state by layer:

- software correctness for the reviewed r2 candidate: substantial reusable evidence; the preregistered focused V8 local-software validation passed 3/3, without rerunning or relabeling the historical full suites;
- Windows API on the retained short live run: observed but diagnostically incomplete (HISTORICAL);
- physical HMD: one qualitative no-flicker observation only; further HMD observation DEFERRED to user prerelease verification (not a #151 gate; no formal protocol required);
- affected environment/control comparison: REMOVED from #151 scope (not run, not passed; no user obligation; no fix claim);
- 4–6 h long session: not run; DEFERRED to user prerelease verification (not a #151 gate; no formal protocol required);
- installed testAppId pair/font lifecycle: REPAIRED canonical PASS (V10) — install/reinstall/uninstall, DirectWrite installed-bytes parity + hash/resource chain; production pair, real-AppId downgrade, and runtime OpenVR font observation NOT claimed;
- integrated #151 acceptance: **REVIEW_PENDING — repaired evidence recorded, acceptance decision retained by Director; no production cert claimed.**

## V7. Staged stop and rollback

No rollback artifact may be inferred from a source revision alone. A rollback pair is Python package + native executable + OpenVR DLL/resources + protocol/capabilities + configuration, with its own hashes and startup selection.

| Stage | Promotion condition | Stop trigger | Rollback pair / action |
| --- | --- | --- | --- |
| 0 — validation only | Focused software checks pass or are explicitly dispositioned; no live launch | Candidate/hash mismatch, software regression, artifact corruption | No product state changed; discard the new temporary validation output only. Existing stage remains evidence, not an install. |
| 1 — #148/#149 compatibility-preserving candidate | Matched protocol 8/r2 tuple verified in isolated testAppId install (V10 PASS); installation resources complete for that scope; production-startup selection still unverified | Mixed pair, startup mismatch, font/resource mismatch, duplicate process, correctness regression | **Production prior tuple still UNKNOWN — no production promotion claimed.** TestAppId lifecycle verified with external-production sentinel + owned-namespace cleanup (production data/registry/shortcuts byte-identical to PRE-REPAIR VALIDATION baseline). Procedure for any future production step: OFF/stop ingress; retire publication/connection/process/device epochs; request graceful shutdown; terminate/kill within normative bounds; require confirmed old-child exit; restore the whole recorded prior tuple; restart with a fresh epoch and revalidated current state. |
| 2 — #150 handoff adoption | Not applicable: #150 is not required and no handoff change is selected | Any attempt to promote Flush/cached/shared/backend experiment as production without new authority | No selected adoption pair exists; do not fabricate one. Return to Stage 1's unchanged D3D11/P05 pair. |
| 3 — limited affected/control rollout | NOT IN #151 SCOPE: comparison removed; HMD/long-session path deferred to separate user prerelease verification with no Stage-3 protocol obligation. | Not applicable to #151 completion. On the separate user path, any observed eligible physical/correctness regression, ignored OFF, unconfirmed termination, or unmatched runtime update stops promotion. | No #151 rollback pair is created by this stage; preserve any failed candidate artifacts. |
| 4 — #152 retry/compatibility decision | Outside #151; must validate its delta against this baseline | Any unapproved retry reduction/removal or compatibility change | Roll back the entire #152-selected pair to the exact #151 accepted pair. That #151 accepted pair does not yet exist; it must not be named now. |

A rollback/restart must not replay old-epoch scenes, history, validity messages, or compatibility ticks, and must continue to honor user OFF. Source-only revert, wire-only downgrade, or swapping only the executable is prohibited.

## V8. Focused executable validation and completion blockers

The following exact **software-only** commands were preregistered above, then executed with Director authorization after the preregistration was frozen. They did not launch SteamVR or perform a physical experiment. Execution used the frozen repository root and CPython 3.12.10:

```text
uv lock --check --offline
uv run --no-project --python 3.12 python -m pytest -q --tb=short --override-ini="addopts=" tests/scripts/test_ovr_hmd_measurement.py
uv run --no-project --python 3.12 python scripts/bench_ovr_hmd_measurement.py dry-run --stage <stage-root>/20260911T215539Z-6bc9c94b --arm off --hold-seconds 3.0 --idle-seconds 30.0 --run-timeout-seconds 30.0
```
Resolve `<stage-root>` at execution time with Python `tempfile.gettempdir()` joined with `puripuly-heart/ovr-measurement`; do not publish the expanded user-specific path. The exact session suffix is `20260911T215539Z-6bc9c94b`.

Focused result: **3/3 commands passed**.

| Check | Observed result |
| --- | --- |
| Offline lock validation | Exit 0; resolved 128 packages |
| Measurement-harness tests | Exit 0; 24 passed in 0.38 s |
| Arm-off dry run | Exit 0; `offline_dry_run-off-3a1b02de`; elapsed 8.563 s; 9/9 application receipts applied; cleanup complete; physical `not_observable` |
| Primary dry-run artifact | `run-offline_dry_run-off-3a1b02de.json`, SHA256 `efd57ab0c8415bf0aa1fc0caf13545ab1ea9e7852f560987347a4da467271aa9`; matching checksum sidecar |
| Disclosed extra run | A direct-Python launcher variant ran before the exact uv rerun and produced the same local-only pass: `run-offline_dry_run-off-eb47f15d.json`, elapsed 8.547 s, SHA256 `0d488d2f6f60921f84356d763c6b013b4a0182d952c2b01abe31291d700962ed`; matching checksum sidecar. It is retained and is not the authoritative V8 run. |

The harness added uniquely named run artifacts only; pre-existing preparation, live, observation, and earlier offline artifacts were not rewritten. No product file, test, or measurement infrastructure changed.

The dry run validates the pinned staged pair and exercises presenter→bridge local application only; offline waits are intentionally shortened by the harness. It is not native rendering, OpenVR execution, HMD evidence, long-session exposure, installed-pair evidence, or cached/fresh discrimination. Do not rerun the cached experimental arm for #151 r2 acceptance.

Existing full native/Python suites and independent review are reusable because no product code changed after `a38626e4`; a documentation validation must not be mislabeled as a fresh 309/375-case execution. If code applicability changes, rerun the affected native/Python matrices before using those receipts.

### Current blockers for #151 completion (actual remaining only; installer canonical gate PASSED in V10)

1. Final source candidate is now PINNED `a6c4505cc31f812bb23be6cf2de319ba8b6f0d02` (committed; hashes match `local://v151-installer-repair.json`); no further source work in this receipt.
2. The #148 acceptance alignment gap: its local receipt accepts application source and notes r2 but does not reissue whole-Goal matched-pair acceptance; issue closure alone cannot fill that receipt gap.
3. Current-pipeline-only scope labeling. The Audio acceptance branch/recognition seam is not landed in this frozen HEAD, so no integrated Audio claim is permitted; later Audio integration and unrelated current-dev changes require their own validation.
4. Retained scope decisions (not installer gates): runtime OpenVR font observation NOT OBSERVED (no HMD); production startup selection / real-AppId installer downgrade not run and not claimed (testAppId scope only; file-level 1054-manifest rollback PASS only); HMD + long-session DEFERRED to user prerelease and NOT blockers; comparison REMOVED.

Resolved since preregistration (HISTORICAL below, not blockers): user authorized installation/rollback with cleanup; isolated testAppId install/reinstall/uninstall PASS with sentinel guard; installed-bytes DirectWrite font parity PASS + hash/resource chain. Pre-repair `collection_load_failed`, unknown-tuple, and authorization-absent judgments are explicitly HISTORICAL (V9) and do not describe current state.

### Separate user prerelease verification (not #151 gates; no formal protocol required)

- Physical HMD observation and 4–6 h continuous use are deferred to the user before release and are not a #151 completion precondition. Status: not executed / not passed; silence is not evidence; no physical, production, or certification claim follows from #151 completion.
- Affected/control comparison is removed from scope and creates no user obligation. No environmental fix claim is made.
- V2.2 figures and §V5 are optional guidance only.

Current conclusion: **OVR-INTEGRATED-ACCEPTANCE REVIEW_PENDING — repaired installer evidence recorded (V10); acceptance decision retained by Director. No production cert, release, push, issue closure, or publication is claimed.**

## V9. Initial install-validation 2026-09-12 — HISTORICAL FAIL (installer gate SUPERSEDED by V10; incident record RETAINED)

Source: `local://v151-install-validation.json` (`validation_complete_with_fatal_incident`, verdict `FAIL`) against baseline `c971da9d` plus uncommitted source repair in flight. Installer/build-source repair is owned by `v151-installer-repair`; this receipt edits documentation only and asserts no installer fix. Source-independent authority is root `ARCHITECTURE.md` (Python presenter/state own caption age/expiry and current replay; native renders the accepted snapshot under revision/epoch checks with no separate expiry timer; wire contract version 8 / execution-contract r2; bounded lifecycle/shutdown). Prior V1–V8 history is immutable and is not rewritten here.

User authorization scope: the §amendment removals/deferrals stand (comparison removed; HMD/long-session deferred to user prerelease, not agent gates). The user additionally authorized installation/rollback with cleanup and stated no app is currently installed. Preexisting registry/files were found during preflight and were intentionally protected; that protection is recorded without disputing the user's statement.

Preflight (journal + setup evidence artifact; sanitized, no personal paths/secrets): real install 1063 files / 611189911 B / tree `67762ed2…`; real registry full typed snapshot; real AppData 5151 files / 44940179 B / tree `63852111…` (count/bytes/tree only); preexisting temp smoke 1 file / 376422101 B / tree `ea9a981f…`; preexisting logs `073cf6f3…` + `125de0cb…`. Temp-log pointers below are historical evidence references, not live durable artifacts (session root removed at cleanup).

Canonical gate — FAIL, not substituted: `scripts/ci/build-release-artifacts.ps1 -AppVersion 2.6.1 -InnoSetupVersion 6.6.1` failed ISCC exit 3 on the deeply nested release-only process-capture helper (source `scripts/ci/build-release-artifacts.ps1:831-838`, `ProcessCaptureSmokeArtifactRoot` line 835), and the short-drive workaround's alternate-AppId canonical-helper installer (SHA256 `47964a93…`) exited 5 and rolled back on MoveFile Win32-3 at a 273-char destination under `%LOCALAPPDATA%/Programs/PuriPulyHeart-LocalSTT-Test/process-capture-smoke/elevenlabs/…`. Supplementary release-equivalent lifecycle (same payload and alternate AppId, helper omitted; explicitly not a canonical pass): installer `beb3ccee…` install PASS; app/gpu worker 2.6.1; overlay contract 8 / r2 / exclusive; GUI startup PASS; local-Qwen runtime PASS; installed TTC DirectWrite collection PASS (`resolver_path_match=true`); runtime OpenVR font diagnostic NOT OBSERVED (no SteamVR/HMD); repair-reinstall PASS (five mutated key files restored, stale `soxr` copies removed); uninstall PASS for owned files + alternate-AppId registration. Candidate hashes preserved: real candidate installer `30d84ad0…`; `PuriPulyHeart.exe 9b3de1c9…`; `Overlay fe0cc0d4…`; `GpuWorker d5d8b671…`; `openvr_api.dll bab8ac6e…`; `NotoSansCJK-Medium.ttc 197d5e1e…`; innounp 1013-entry archive PASS. Protocol suites PASS (old manifest without quiet-tail profile defaults P05; handoff experiment strict default off).

Prior-release drill (file level only): GitHub asset `549929441` digest `a9c23993b77e4e07aa4f6c9118617bbb3f08ba38a955123499f4a2671218803d`, archive verify/list/extract PASS, 1054 app files, manifest `805aa60b…`, app/gpu 2.6.1, overlay contract 6; candidate→prior file-level restore matched the 1054-file manifest. This was not a real-AppId installer/registry downgrade and is not claimed as one.

Fatal AppData incident — FAILED, unrecovered, user-waived (never relabel, never claim full restore): during the isolated alternate-AppId test, the uninstaller resolved `{localappdata}` via the Windows known folder despite the process environment override and deleted the real `%LOCALAPPDATA%/puripuly-heart` tree (5151 files / 44940179 B, tree `63852111…` → 1 empty log file). Cause on record: `installer.iss:145-147` deletes `{localappdata}\puripuly-heart` for every AppId while `scripts/ci/build-release-artifacts.ps1:845-859` isolates only the provisioned smoke root and `:1034-1042` then runs the alternate-AppId uninstaller. The user was informed immediately and explicitly waived recovery (`아 상관 없어 계속해. 데이터 별거 없으니`); no user-data copy exists in validation artifacts so restoration is impossible. Real install + registry and the approved preexisting temp smoke/logs verified exact preflight match at cleanup; all owned artifacts (processes, drive mappings, session root, repo build outputs, alternate install/registration, generated wheel, start-menu/desktop entries) were removed. Real AppData remains unrestored (1 file / 0 B / tree `bf8cfcd3…`) by explicit waiver.

Superseded installer blockers (HISTORICAL — resolved by V10 repair; retained here only as history): (a) canonical MAX_PATH/helper compile+install failure; (b) pre-repair production-pair/font/rollback-tuple judgments; (c) wait for repaired evidence — now ARRIVED (`local://v151-installer-repair.json`, committed `a6c4505c…`). The fatal AppData incident record above is RETAINED unrecovered user-waived and is not superseded. Conclusion of this section alone: **V9 FAIL historical; current status is V10 + V8 REVIEW_PENDING.**

## V10. Repaired canonical install 2026-09-12 — PASS (testAppId scope; REVIEW_PENDING, no production cert)

Evidence: `local://v151-installer-repair.json` (`result: passed`) on baseline `c971da9d`; verified canonical source now COMMITTED `a6c4505cc31f812bb23be6cf2de319ba8b6f0d02` (5 files: `build.spec`, `installer.iss`, `scripts/ci/build-release-artifacts.ps1`, `tests/app/test_release_dependency_guards.py`, `tests/config/public_compatibility_surfaces_snapshot.json`; evidence source hashes match this candidate; native/Python product `a38626e4` unchanged). Command: canonical `build-release-artifacts.ps1 -AppVersion 2.6.1 -InnoSetupVersion 6.6.1`.

Source pair fingerprints (repaired build): release installer `59a3e552…` (172613906 B); isolated testAppId installer `7b1c23ef…` (203274565 B); app `2d878cce…`; overlay `1718edda…`; gpu worker `dd685509…`; helper `f72d760f…` (1416 files, max installed path 167 vs 273 pre-repair, `contains_elevenlabs=false`). Build sources: `build.spec cc0786c7…`, `installer.iss 26c578e3…`, `build-release-artifacts.ps1 5b0a209c…`.

Lifecycle: full isolated testAppId install/reinstall/uninstall PASSED with safety sentinel guard, all checks. Safety: external-production sentinel survived; production data/registry/shortcuts byte-identical to the PRE-REPAIR VALIDATION baseline (post data-loss: data `0a3a28f7…`, registry `180752dc…`) — explicitly NOT a restore of the original 5151-file tree; the V9 data-loss incident remains FAILED unrecovered user-waived. Test registry/install/final test-user-state namespace removed. Cleanup COMPLETE per `local://v151-installer-cleanup.json`: retained `release-installer.exe` (172613906 B, `59a3e552…`) and `isolated-installer.exe` (203274565 B, `7b1c23ef…`) deleted after hash+byte verification (~376 MB); remaining `retained-final` holds 9 evidence-only files (2221969 B, zero executables — install/reinstall logs by hash+size plus small runtime-check receipts and installer digest); no owned processes/maps/junction touched or left; test install absent with no uninstaller run; preexisting stages/install/data/registry/shortcuts and unrelated drives untouched. Former temp paths are historical pointers, not live durable artifacts.

Font/resources: installed-bytes DirectWrite production-loader collection PASS (earlier evidence) + current hash/resource chain (`openvr_api.dll bab8ac6e…`, `NotoSansCJK-Medium.ttc 197d5e1e…`, protocol 8/r2). Runtime OpenVR font diagnostic still NOT OBSERVED (no HMD) — retained scope decision, not a blocker. Prior 1054-file FILE-level rollback PASS stands; real-AppId installer/registry downgrade NOT RUN and not claimed.

Status: installer gate CLOSED for the testAppId scope. Final #151 acceptance is **REVIEW_PENDING** under the V8 remaining items; this receipt does not self-claim acceptance and does not assert unqualified production acceptance — the software-only isolation limitation applies (no HMD, no production startup pair, no real-AppId downgrade). No release, push, issue closure, or publication is claimed. No further work in this receipt; complete and ready for checkpoint integration.
