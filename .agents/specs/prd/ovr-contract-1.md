# OVR-CONTRACT-1 — Output acceptance, presentation and recovery

## 0. Revision, authority and evidence baseline (C0, C9)

**Current revision: r2. Status: APPROVED scoped amendment in the current maintainer conversation: remove native caption validity leases and expiry-triggered Hide entirely. The r1 approval and original text below remain history for the superseded clauses and continue to govern unaffected requirements.**

Authority: [OVR-C #146](https://github.com/kapitalismho/PuriPuly-heart/issues/146), under [OVR-0 #145](https://github.com/kapitalismho/PuriPuly-heart/issues/145). The maintainer explicitly approved `OVR-CONTRACT-1 r1` and its support scope after reviewing the approval request, then requested publication to both issues. Approval covers the normative target below, including selected bounds, lease, protocol floor and OC01–OC12; it is not evidence that those changes have landed or that HMD freshness is solved. G-C is open for #148/#149 implementation under this contract; G-H and later acceptance gates remain independent. The pre-approval draft SHA256 was `dbce885550e665a2506290f5bed0addde1912f23ce7261bb9bf416beae9876a3`. Approval-record edits do not change its selected runtime policy. References below to a proposal or proposed engineering ceiling describe selection/evidence provenance, not a remaining design decision.

| Record | Evidence / disposition |
| --- | --- |
| Initial assessment | `1d92e5c3301fcaf7619f3a9521781627f310cee4` |
| Work-start Python/native source | `4e967df9d03649106faa8348c3ec611009529ffe`, identical to issue publication's observed dev |
| Branch / upstream / initial dirty paths | `ovr-0-vr-overlay-reliability-program-cross-envir`; no configured upstream; no dirty paths |
| Relevant intervening delta | `git diff --stat initial..HEAD`: 15 files, +1668/-94; Gemini transcription lifecycle, VRChat scene handling, settings migration and associated tests/context. No overlay/output/translation-turn/native source changes in that range. This is source evidence, not Windows execution evidence. |
| Package declarations | Python and native Cargo package both declare 2.6.1. A declaration is not a deployed-build receipt. |
| Actually deployed Python/native pair | **UNKNOWN / not verified.** No deployed executable pair in inspected issue evidence; no native overlay executable or dist tree found in this worktree. Do not substitute HEAD for an installed build. |
| Vendored client library | `third_party/openvr/README.md`: OpenVR v2.15.6, declared DLL SHA256 `bab8ac6ef64e68a9ca53315b0014d131088584b2efdfa6db511d67ec03cfcb4a`; native dependency `openvr_sys = 2.1.3`. Neither identifies the user's running SteamVR version. |
| Current production support baseline | Windows x64, D3D11 hardware on OpenVR-selected adapter, SteamVR/OpenVR overlay alongside VRChat; head-locked and spatial-locked; desktop behavior preserved. Current manifest version 6; native exclusive-retry capability version 1. |
| Environment certification | No affected-session GPU/driver/SteamVR/HMD/connection-path inventory or physical observation obtained. Source inspection does not establish the cause of the field staleness reports. |

Before a deployed acceptance receipt, record Python artifact/version/source, native executable SHA256/source, OpenVR DLL hash, resolved retry profile and settings, actual GPU/driver/SteamVR/HMD/connection path, observation method and exposure. Missing deployed evidence does not prevent drafting or software design approval; it blocks claims about the deployed pair and HMD conformance.

Non-goals: backend adoption or API selection for #147/#150; OpenXR/standalone/OBS/browser expansion; exactly-once physical display; Audio endpoint/model/provider retry changes; project-wide Failure redesign (#78); compatibility protection removal (#152); refactoring owners merely to rename or split files. This contract approval opens #148/#149's implementation prerequisite, not blanket merge, push, release or deployment permission. This task publishes the approved contract and updates #146/#145; no production cutover is performed.

### 0.1 r2 amendment: application-owned caption expiry

The maintainer explicitly chose removal of native expiry checks and expiry-triggered Hide after reviewing the tradeoff. The implementation baseline is `396a6eadf0aa7edc8e83cd8e75a2caebdd461f6f`. A direct owner comparison against pre-N `80b15cef49ad47417f6a22abfe27c2a5fabba387` found that initial N `49541c90f3809f4bfad47a6025722f64abe5c7a8` introduced whole-overlay Hide during caption removal, interrupting a surviving row. This amendment removes the independent expiry policy rather than retaining it behind a bypass.

The following terms take precedence over the original r1 text:

- Python presenter/state remain the sole caption-age and expiry authority. Existing presenter TTL composition, send-time pruning, ordered clear/OFF, current-state replay and semantic retirement remain required.
- Native applies the authenticated current snapshot under existing process-epoch, monotonic revision and semantic-retirement rules. No content-validity challenge/response, block lease, lease admission, lease renewal, independent caption expiry timer or expiry-triggered Hide remains. Neither rendering nor handoff waits for a separate caption-validity response.
- The r1 maximum-three-second content-validity guarantee and immediate native expiry hide/removal requirement in §4 are **withdrawn, not passed**. If the application cannot deliver expiry/removal or a replacement frame cannot be produced, old captions may remain visible. Health monitoring does not certify caption freshness or restore this removed guarantee.
- Explicit OFF, runtime detach, shutdown and their bounded process cleanup remain. The normal successful-empty-frame 500 ms hide grace remains. Receiving or removing one row does not by itself require a whole-overlay Hide while another row remains drawable.
- Health challenges, their anti-replay/current-process checks and time bounds, due-work progress, current-revision handoff accounting, GPU/query ownership, diagnostic bounds and recovery budgets remain required. A health deadline is not a caption lease. Remove `lease_valid` and `lease_scene_revision` from owner status; retain current-revision coverage and real visibility facts without fabricated lease-success fields.
- Matching package protocol is **8** and execution capability is `execution_contract {version: 1, revision: "r2"}`. Exclusive native retry capability remains version 1. Protocol 7/r1 is historical rollback material, not a supported mixed or degraded mode. Removed validity messages are unsupported protocol input, not silently ignored compatibility paths. Migrate Python, native, desktop schema consumers and measurement tools together.
- P05, the existing D3D11/OpenVR backend and the opt-in cached-frame experiment remain. Cached rehandoff still requires identical current scene/raster/presentation/device identity and completed producer work, cannot count as fresh rendering and cannot refill recovery allowance; only its obsolete lease requirement is removed.

Affected receipts: #146 contract, #149 native/runtime/supervisor, #148 matching application/bridge integration, and downstream #147/#150/#151/#152 consumers. In OC08, application stall/lost-clear no longer requires independent native caption expiry; ordered invalidation on receipt, startup replay pruning, health/recovery and scoped behavior remain. OC11 still requires separate software/API evidence and physical observation. No earlier r1 lease evidence establishes r2 behavior.

Required r2 verification includes first-frame submission without validity exchange; SELF+PEER to PEER-only replacement without Hide under delayed visibility observation; no autonomous caption-expiry Hide during quiet input; preserved empty-frame/OFF/shutdown behavior; preserved health/recovery/currentness and experiment accounting; and matched protocol/capability acceptance plus obsolete-pair/message rejection. Physical flicker elimination remains unverified until an operator-authorized HMD run.

This local amendment records the maintainer decision. It does not claim the remote r1 approval comments were rewritten or authorize push, publication, deployment, release or issue closure. The remaining sections preserve original r1 wording as the inherited base; only the terms identified above are superseded.

## 1. Production composition and responsibility (C1)

Paths below are relative to `src/puripuly_heart/` unless prefixed `native/`. `ARCHITECTURE.md` remains the implementation system map. This proposal retains its existing owner hierarchy; no new top-level manager or competing scheduler is selected.

### 1.1 Actual call / await / ownership map

```text
app/wiring/wiring_runtime_pipeline.py
  constructs TranslationTurnLifecycleOwner + TranslationOutputProjectionOwner + OutputRuntime
composition/application_runtime.py
  injects projection into OverlayApplicationOwner
TranslationTurnLifecycleOwner
  -> TranslationChannelOwnerCallbacks -> SELF/peer channel
  -> TranslationOutputProjectionOwner.project_translation_result
  -> OutputRuntime.publish_overlay_event
  -> OverlayPresenter.emit -> OverlayPresentationState
  -> OverlayBridge.replace_snapshot -> websocket.send
  -> native BridgeClient -> NativePresentationOwner / PresentationRuntime
  -> renderer -> GPU event query -> OpenVR -> visibility reconciliation
```

| Actual await at work-start | Authority / coupling |
| --- | --- |
| `translation_turn.py::_run_parent`, `_execute_child` | Non-dual next LLM waits `semantic_done_event`. Processor completion releases that event **before** waiting predecessor `closed_event` and submitting output. A semantic execution barrier is not an output-close barrier. Dual-target SELF retains independent child execution. |
| `translation_output_projection.py::project_translation_result` | Single-target SELF: UI, awaited overlay final/close, then chatbox; peer: awaited overlay before UI. Exception isolation does not prevent slow-socket coupling. Dual-target progressive chatbox and primary-only projection have distinct ordering. |
| `self_primary_presentation` | `_self_publish_lock` then `_self_surface_lock` span caller's awaited surface work. Current-caption authorization lives here, not only in the presenter. |
| `output.py::publish_overlay_event` | Validates/records in-flight identity under `_overlay_delivery_lock`, creates a sink task, releases lock, then **awaits the task**. Task creation is not delivery isolation. |
| `presenter.py::emit`, `_publish_if_changed` | `_ownership_transition_lock` can span reduction, scene publication and bridge send. Expiry/preference/calibration mutation paths are not uniformly under this lock. All local mutations need one consistent acceptance boundary. |
| `bridge.py::replace_snapshot`, `_handle_connection` | `_snapshot_lock` covers state replacement and awaited broadcast; authentication replay also holds it while sending. Stored snapshot is only one state, but waiting callers/tasks are not bounded. Heartbeats/control are outside this writer serialization. |
| `bridge.py::_broadcast_json` | Sends sequentially; catches send failures and prunes connections. A successful emit return may follow a caught transport failure or no connected consumer. It is not a transport receipt. |
| Native frame cycle | `runtime.rs` may cancel readiness on a bridge message before checking that message's currentness. Synchronous rendering/driver/OpenVR calls can prevent async timers from running. |
| Existing lifecycle | `OverlayApplicationOwner` policy → `OverlaySessionTransitionOwner` serialized replacement → `OverlayGenerationStartOwner` composition → `OverlayRuntimeHandle` owned tasks/process/bridge → `OverlayProcessManager` child events/termination. |

### 1.2 Frozen target responsibilities

| Boundary / existing owner | Authority | Termination / prohibited authority |
| --- | --- | --- |
| Semantic/output: translation turn + projection + OutputRuntime | Origin/activation/parent/target admission, source/translation pairing, provisional/final, generation/order, routing and publication permission; local submission receipt and parent obligations | Every child releases semantic and output-close obligations once on success/failure/empty/cancel/expiry/rejection. No socket/GPU/HMD await owns those obligations. Does not invent overlay placement or recognition retry. |
| Projection/presenter: projection + OverlayPresenter/State | Authorize current caption; apply ordered events; select two-row scene; original age, logical occupant/appearance, scene revision and freshness intent | Return only after state **and** required intent are committed, or explicit not-applied rejection. No remote I/O in local mutation critical sections, including TTL/settings/reset paths. |
| Delivery: OutputRuntime destination handoff + OverlayBridge | Bounded parent-batch admission, then latest accepted scene/intent mailbox and one transport writer; written/failure receipts; current replay | Parent batch terminates at destination-local application/rejection, not socket completion. Bridge-owned delivery survives caller cancellation only in valid scope. On ambiguous write, retire connection epoch; never retry a history batch. |
| Native: NativePresentationOwner / PresentationRuntime | Apply epoch/revision, due episode, render generation, GPU readiness, handoff, compatibility schedules, visibility and placement | Current work progresses or finitely fails. Cannot renew semantic age, grant publication permission, or equate diagnostics/heartbeat with progress. |
| Backend adapter: existing renderer/OpenVR | CPU object, producer work/query, pixel content, export association and API stage guarantees | Retain actual GPU work after observer cancellation. Prove safe mutation/reclamation at backend boundary; no fictional copy-complete or HMD receipt. |
| Supervisor: existing application/runtime/process owners | Start/stop, stale-progress observation, process epoch, terminate/kill, finite restart episode and circuit-open/user action | Replacement only after confirmed old-child exit. Cannot claim GPU/OS-wide hang recovery or reset recovery allowance on ready alone. |

## 2. Acceptance and truthful receipts (C2)

One local transaction validates scope/order, applies reducer state and required intent, assigns/reuses its publication key, and records a receipt before returning. No await of a socket, logger, GPU or HMD may occur inside it. Waiting for a bounded local predecessor is allowed; an unbounded task-per-event queue is not. Cancellation before application yields `not_applied/cancelled`; cancellation after the commit point cannot turn acceptance into an unknown or cause reapplication. Duplicate calls return the retained disposition or `stale/retired`, not a second event. Local accepted state can coexist with a separately rejected/superseded delivery.

| Stage | Owner and correlation | Success | Failure / non-observation |
| --- | --- | --- | --- |
| `application_accepted` | Projection/presenter; publication scope + parent/target/presentation revision + reducer sequence | State and required intent applied in valid scope | `not_applied` with stale/cancelled/closed/overload/invalid reason; accepted-but-delivery-failed is not rewritten as not-applied |
| `delivery_admitted` | Output/bridge; destination, publication key, scene revision, process/connection epoch | Bounded reservation/current-state responsibility acquired | `delivery_rejected`, explicitly releasing ordering obligations; no ASR/LLM retry |
| `transport_written` | Single bridge writer; exact epoch/revision/control barrier | Actual library send returned successfully | Write error/timeout is failed or ambiguous, never native acceptance; buffered bytes may still have reached peer |
| `native_accepted` | Native reducer; epoch + applied revision + control/scoped invalidation watermarks | Corresponding state and intent actually applied | Pending/not observed is not failure until deadline; mismatched/retired revision rejected; missing receipt is not inferred from logs |
| `producer_ready` | Backend/native; device epoch + render attempt + query association | Completion query for preceding producer work actually met | Pending, cancelled observer, late, query error and device loss distinct |
| `runtime_api_accepted` / `export_copy_completed` | Backend/native; same attempt + scene/intent coverage | Respectively successful selected API return / actual backend-provided copy completion | Current DirectX `SetOverlayTexture` supplies only API return. `export_copy_completed = not_observable` on this adapter, not success |
| `runtime_visibility_observed` | Native/OpenVR; observation sequence and current desired revision | Reported `IsOverlayVisible` state, with desired/observed distinguished | Unknown/query error/mismatch are distinct; `ShowOverlay` success alone is not observation |
| `physical_observation` | Independent HMD observer; correlated scene/build/session and observation method | External observation only, with exposure/error stated | Default `not_observable`; neither logs, producer-ready nor runtime visibility proves pixels |

Receipt fields are a semantic schema, not a parallel global Failure framework: `scope {channel, origin, activation, parent, targets, publication_generation/order}`, `publication_revision/event_seq`, `destination`, `process_epoch`, `scene_revision`, optional `render_attempt`, optional `compatibility {target,generation,episode}`, `stage`, `outcome`, safe `cause`, `superseded_by`, `recovery_disposition`, local observation sequence/time. Absent stage evidence is `not_observed`/`not_observable`, never an implied success. Preserve first cause and attach cleanup failure; do not replace query/device/bridge failure with generic `renderer_init_failed`.

Outcomes: `applied`, `admitted`, `written`, `observed`, `not_applied`, `cancelled_local`, `stale`, `superseded_product`, `delivery_rejected`, `attempt_failed_recoverable`, `terminal_failed`. Supersession is authorized product state replacement, not a euphemism for pressure loss. Historical application receipt and delivery disposition remain separate even after detailed receipt retention expires; bounded counters/watermarks preserve truth without an infinite audit log.

No per-frame ACK. Native sends a coalesced **current health/status API record** at 250 ms while due and 1 s while idle, plus terminal/control changes. Include latest applied revision, latest current-covered handoff, in-flight stage, due-episode elapsed duration, state classification, and invalidation/lease state. Receipts can skip intermediate revisions; an applied high-watermark certifies application of that current full scene, **not application or display of every skipped scene**. Application semantic receipts cover skipped semantic transitions. Detailed render receipts use the existing bounded diagnostic ring only and are not operational authority.

## 3. Identity, currentness and coalescing (C3)

| Identity | Lifetime / retirement |
| --- | --- |
| Publication scope/generation | Existing output/translation generation, refined by origin and activation where speech-only retirement is required. Manual is SELF channel but not speech origin. Generation cannot be reused while callbacks may exist. |
| Parent, child, target, presentation revision | Existing parent UUID, derived target/run child identity and parent presentation revision; preserve dual-target behavior. Parent order is monotonic within existing scope. Retirement closes all owned child/order obligations. |
| Reducer event sequence / adapter event ID | Existing adapter sequence is meaningful only in its publication namespace; `evt-1` after adapter replacement cannot collide with a surviving delivered set. Qualify with existing generation, not a new global UUID. |
| Logical turn / occupant / appearance | Reducer `(channel, utterance_id)`, block ID, `occupant_key`, `appearance_seq`; new identical text has a new turn. Presentation revision or compatibility tick is not a new appearance. |
| Scene revision | Monotonic in presentation epoch; increments on accepted scene/intent/control changes. Reset to zero only in a fresh epoch after retiring old admission. |
| Compatibility generation/target/episode | Reuse existing fresh-render and quiet-tail fields. Bound to valid target, original episode cadence/deadline and process epoch. Not semantic identity. |
| Process/device/connection epoch | Existing runtime instance/token/generation; device loss retires its resource epoch. Old callbacks/receipts cannot advance replacement. |
| Render/submission attempt | Native-local monotonic attempt in device epoch; binds query/work/pixels and stage receipts. Never inferred from texture pointer identity. |

| Input/change | Application rule | Wire / native disposition |
| --- | --- | --- |
| Provisional update | Apply authorized sequence to current turn; reject stale duplicate. Final may replace mutable preview, retaining application receipt | Intermediate full scenes may coalesce after reduction; no provisional FIFO |
| Final transition | Commit its semantic transition once; original committed source/provenance not retroactively rewritten by unrelated late result | No exactly-once display guarantee. A newer scene may product-supersede its display attempt; retain that distinction |
| Late translation | Only declared translation/presentation fields can advance with valid target/config/sequence; finalized does not freeze all translation text. Preserve final pairing, original turn order and newer active priority | Late old history can complete without reoccupying the current row; evicted terminal turns cannot resurrect |
| Same-text new turn | Preserve distinct turn/order/appearance and normal spatial new-turn meaning; no string-only dedup | Render signatures cannot suppress required identity/intent change |
| Same-turn reentry | Preview-only removal allows a newer authorized final; terminal retirement does not. Keep original occupant/appearance where it survives | No new spatial anchor merely for empty→same-turn, retry, translation or metadata change within an epoch |
| Clear / Caption OFF / scoped abort | Ordered local invalidation barrier with monotonic scope watermark; remove invalid target/intent before return. Caption OFF is destination-wide; TALK/LISTEN are origin-scoped | Latest full state may absorb clear only while carrying its watermark. Native applies watermark before render/handoff; older in-flight unauthorized work never handed off after invalidation is known |
| Freshness intent | Carry unmet `(target,generation,episode)` until covered, authorized supersession, expiry or epoch retirement | Newest snapshot joins still-valid intent, not merely its last text. Coalescing cannot reset deadline/count or erase unmet obligation |
| New current scene | Reducer explicitly authorizes supersession | Drop old unsent scene, mark display attempt superseded; retain shared due deadline under continuous churn (§5) |
| Replay | Re-evaluate permission/expiry/scoped watermarks on current state, not cached `snapshot()` alone | Send one valid current scene and newly scoped still-required intent. Never old ticks, FIFO history, expired captions or retired episode credits |

An OFF cannot unsend bytes or erase already delivered HMD pixels instantaneously. On receiving its barrier native rejects old work before new rendering/handoff, requests hide when appropriate, and reports actual observation separately. If control cannot pass a stalled write, retire transport and use native lease/supervisor termination; local invalidation still completes immediately.

### 3.1 Safe semantic retirement

Reuse existing parent generation/order and reducer sequence. Owners retain live/potentially reentrant entries plus finite completed-key rings. Maintain a retired contiguous order frontier and only bounded holes belonging to admitted live parents; a callback older than the frontier is rejected even after its UUID falls out of a ring. Unknown identities cannot create an old turn from a callback: only authorized parent admission can establish a new identity/order. Preview-only tombstones retain their sequence rule until the parent becomes terminal. Retirement waits for all target children and output obligations, not first target or first SELF contribution.

Do not delete existing lifetime terminal/seen sets with LRU alone. Native only needs current live identities, bounded retained same-turn spatial records, and the propagated non-reentry frontier. Once the application declares a turn semantically retired, it cannot reenter via a later snapshot. Same-turn records lacking that proof remain reserved; at the cap reject additional admission explicitly rather than forget identity. Live indefinitely valid captions retain their slot; age alone does not retire them. Process replacement invalidates native pose state: one `recovery_anchor` for the still-valid current spatial occupant is allowed after a valid pose, explicitly not a new semantic turn. Within an epoch, pose-unavailable consumes no new-turn/reanchor identity; retain one current pending anchor, supersede it only with authorized newer occupant, and resolve when pose returns.

## 4. Time, expiry, lease and reset matrix (C4)

Keep presenter policy at work-start: meaningful visible-content anchor + **8 s**; closed content without visible anchor + **5 s**; shown SELF translation minimum **4 s** combined by max; retained hidden-window eviction + **5 s** combined by min. `OverlayPresentationState.entry_expiration_components` is the source of composition rules. Metadata/refresh does not renew age; source/audio time is correlation only and never replaces the overlay clock origin. No-timeout active state remains legitimate while its owner continues authorizing it.

Selected correction: native independently enforces a **3 s maximum validity lease**, renewed only by application owner revalidation, not socket ping, snapshot receive time, reconnect, native-ready or a detached heartbeat task. This bounds stale display when the application loop cannot send expiry/clear. It is a degraded-safety ceiling, not a new successful caption endpoint. Early hide during severe scheduling/transport delay is intentional and observable, preferable to indefinitely stale private text.

Clock model uses durations and a native challenge, not subtraction of unrelated monotonic timestamps:

1. Native issues an epoch-scoped increasing challenge every **1 s** and retains its native send instant `n0`. Keep at most four outstanding challenges; each expires at `n0 + 3 s`.
2. Under the local acceptance owner, application rechecks current scope and original semantic expiration and returns challenge ID, current scene/control revision and each valid block's remaining lifetime `r` in seconds. For unbounded legitimate active lifetime use `r = 3 s`; for finite lifetime clamp to `[0, 3 s]`. Revalidation cannot originate from a stale saved response.
3. Native derives `deadline = n0 + min(r, 3 s)` using its own clock. Never `receive_time + r`. The challenge preceded the application's calculation, so this is conservative under delivery delay. An expired/unknown/old-epoch/replayed challenge cannot extend a lease. Accept a challenge response once, in increasing renewal order, only for its matching current block/scene or an explicitly retained same occupant.
4. A newer content revision needs matching validity before display. A scope-clear has priority and needs no lease to invalidate. A lease-only renewal does not create a scene revision, new turn, render due work or compatibility episode. Native health status can carry the next challenge; no frame ACK is introduced.
5. On expiry, invalidate the affected content locally and request hide/removal immediately (do not add the ordinary 500 ms empty-scene tail). Report `lease_expired/degraded`; reconnect alone cannot revive it. A still-valid application owner may subsequently authorize current content with a fresh challenge, without resetting original age.

Assumption: both monotonic clocks represent elapsed seconds at ordinary OS clock accuracy; suspend/resume invalidates lease/challenges before displaying again. No clock-offset mapping is assumed. This mechanism does not guarantee a hide during a synchronous driver/OS-wide hang; external supervision provides the separate bounded attempt (§5).

| Event | Immediate semantic / admission action | Scene, replay and process action |
| --- | --- | --- |
| Caption OFF | Retire overlay destination permission and its pending delivery for all origins; retain valid translation/UI/history/chatbox work | Empty barrier, discard retry/lease, stop overlay child; no automatic restart until enabled. Do not reinterpret as LISTEN/TALK OFF or cancel their recognition/manual work |
| LISTEN OFF | Abort peer speech activation, recognition and uncommitted peer publication including source-only/cancellation callbacks | Remove peer scene/intent only; SELF speech/manual and their destinations continue |
| TALK OFF | Abort SELF **speech-origin** activation/speculation/uncommitted speech; do not bump a generation shared by valid manual parents | Remove affected speech preview/scene intent only; manual and peer remain valid; committed history not erased |
| Runtime detach / target switch | Detach that destination, invalidate transport epoch; retain only explicitly preserved valid presenter state | Old writer disposed, hide/teardown old child; target-specific controls/intent cannot cross to desktop/native accidentally |
| Startup replay | No new semantic admission or age anchor | Unconditionally prune expired/unauthorized state and rebuild timers; handshake/capability/lease before display; replay current state only |
| Graceful replacement | Named valid scopes survive; old callbacks lose resource authority, not valid manually admitted result authority | Confirm old child exit first; same original expiry; new process/lease and recovery-anchor policy; old compatibility episode retired |
| Crash recovery | Mark failed delivery attempt, preserve valid application state/history and first cause | Budgeted restart, new epoch, fresh lease and current-state revalidation; no old ticks or previously transmitted chatbox/history replay |
| Global shutdown | Close all ingress/publication scopes before awaits; cancel/reject pending work with terminal receipts | Empty/invalidation intent, bounded writer/process teardown, no restart; preserve termination-failure reference/actionable status |

## 5. Due-work progress, GPU lifetime and recovery (C5)

### 5.1 Due episode and fairness

A due episode starts when native accepts valid render/handoff/visibility/placement work, **before** any cancellable wait. Its wall-clock no-progress deadline is **2 s**, retaining the existing duration but correcting its timeout-only origin. Track required current scene/intent coverage and producer stage separately. New snapshots, stale input, preemption, render returning, heartbeats and logs do not reset the publication deadline. Producer-ready is useful stage evidence but not completion of the handoff obligation.

A handoff covering the then-current valid scene and required intent, or observed completion of required hide/placement reconciliation, completes the relevant work. Authorized OFF/expiry can end it as invalidated, not presented. New valid scene supersedes old display work but inherits the unresolved deadline, so continuous churn must present a current eligible scene or finitely fail; repeated submission of an obsolete scene cannot keep it healthy. A completed episode may start a new one for genuinely new due work. Retry opportunities never change content age.

Preserve existing fairness values: at most **8** ingress messages before a readiness/timer/control poll, at most **8** OpenVR events per turn, OpenVR polling **50 ms**. Apply fairness also to protocol Ping/Pong/no-op loops, stale snapshots and reverse diagnostics, not just currently ignored heartbeat types. A render already enqueued is not re-enqueued on every preemption. Revalidate currentness at handoff and before mutating an associated texture.

Health classifications: `healthy_idle`, `intentional_hidden`, `no_drawable_content`, `pose_unavailable`, `runtime_unavailable`, `due`, `recovering`, `terminal_failed`. Idle/hidden have no frame-progress expectation. Pose unavailable is not a driver stall: retain bounded current placement intent, suppress a new unanchored spatial display rather than invent an anchor, and continue control/lease/status; a previously established valid anchor need not be discarded. Returning pose creates due reconciliation under the same occupant. Runtime absence hides/retires resource intent and consumes finite recovery, not an endless spawn loop.

### 5.2 GPU and adapter safety

Current Windows path owns one persistent 4096×1056 BGRA8 target (nominal **17,301,504 bytes / 16.5 MiB**), clones its COM reference for frames, and creates a new `D3D11_QUERY_EVENT` for each readiness call (`End`, `Flush`, `GetData(DONOTFLUSH)`, 50 ms). It has no query-reuse pool. Cloning the reference does not preserve old pixels, and dropping a future/query wrapper does not cancel issued GPU work.

Selected baseline resource contract: **one outstanding producer generation/query and one CPU-only successor scene per device epoch**. Keep the persistent target/update model; do not select double buffering, a shared-handle backend or full-frame copying here. GPU-incomplete superseded work occupies the same reservation until completion/device retirement. This deliberately sacrifices catch-up enqueueing under a stalled GPU for a finite resource bound.

| State | Required ownership and safe transition |
| --- | --- |
| Before GPU enqueue | Discard superseded CPU scene freely with disposition; no GPU token exists |
| After enqueue / CPU wait cancelled | Native retains generation, query, texture/device and completion observation; cancellation ends publication permission for that attempt, not GPU ownership. No successor render until preceding producer work is complete or device/process is retired |
| Readiness late | Preserve reservation; poll existing completion instead of issuing another render/query. Readiness wait budget **50 ms**, recheck at **100 ms** recovery cadence within original **2 s** due deadline |
| Producer ready, before handoff | Query certifies only its bound generation and commands preceding its marker. Recheck scope, lease, scene and intent. If superseded, retire attempt and render bounded successor; no stale handoff |
| Query reuse | Only after the previous query's completion/result is consumed and no previous observer can report against a reassigned token. A reused query must never certify another generation. Per-attempt creation is allowed within the one-outstanding bound |
| Handoff / mutable pixels | Current DirectX adapter reports only successful `SetOverlayTexture`. Persistent texture is a live update surface, **not** retained historical pixels. Serialize producer writes and API calls on its owning context; only current authorized content may update it. Do not label an older receipt as evidence for pixels after mutation |
| Export association / disposal | Retain the associated CPU texture/device resources while OpenVR may reference them. Hide is not release. Use successful API-defined association release/overlay destruction during teardown, preserve producer-work ownership, and retire device/process on fatal failure. No arbitrary N-frame delay or COM refcount is consumer-release proof |
| Device/query/runtime fatal or no completion | Close admission, preserve cause, retire device/process via supervisor; never allocate an unlimited replacement pool or free a live native producer on the assumption CPU cancellation killed it |

Pinned [OpenVR v2.15.6 header](https://github.com/ValveSoftware/openvr/blob/v2.15.6/headers/openvr.h) documents `SetOverlayTexture` and `ClearOverlayTexture` (release association), but does not expose a DirectX export-copy completion fence or HMD observation. `GetOverlayTexture` is an acquired native handle requiring `ReleaseNativeOverlayHandle`, not a consumer-release query for the submitted source. `TextureType_DXGISharedHandle` has a different direct-consumption/atomic-update comment; do not apply its rules to the current `TextureType_DirectX` or silently adopt that backend.

Backend-specific proof for any stronger pixel immutability/copy-complete guarantee belongs to #147/#150. The common safe failure policy is fixed: if a proposed backend cannot prove safe update/reclamation under this bound, do not adopt it; retain the baseline API-only claim or narrowly amend the affected safety/bound clause. This is not a runtime-affecting TBD delegated to N.

### 5.3 Existing supervisor policy

Native status uses a bounded control API path independent of stdout/stderr. The existing Python process manager checks every **250 ms**: due elapsed >= **2 s** without coverage is stalled; no fresh owner-execution proof within **3 s** is unresponsive, even while no frames were expected. A worker-thread heartbeat cannot substitute for owner status. Host application loop stalls are covered by the native lease, not a Python watchdog that cannot run.

Owner-execution freshness uses the reverse of the validity challenge, without cross-clock subtraction: Python issues an epoch-scoped increasing health challenge every **1 s**, recording its own issuance instant `p0` before bounded control admission. Native's presentation owner (not a pipe reader/heartbeat thread) echoes the challenge with its current status after processing it. Keep at most **4** outstanding challenges, each valid through `p0 + 3 s`; accept a response once, in increasing challenge order. A valid response establishes owner freshness only through **`p0 + 3 s`**, never arrival time + 3 s. Old, duplicated, expired or old-epoch responses do not renew health. At connected startup the first health deadline is connection time + 3 s; later only a valid owner's challenge response advances it. Thus staged historical statuses can supply evidence but cannot perpetually extend liveness. Piggyback challenges/responses on current control/status messages using their existing keyed slots; no per-frame ACK or unbounded challenge queue. A blocked forward writer can cause conservative degraded recovery rather than a false healthy verdict.

Written-but-unaccepted current state has a separate **2 s native-acceptance deadline** starting at the Python-local return of the first successful scene/control write whose obligation remains unresolved. Completion requires current full-scene application covering the outstanding valid scene and invalidation/intent watermarks, or an explicit correlated native rejection/terminal disposition. A successor write or old-scene healthy status cannot renew this deadline; successor application may cover the still-valid obligation because this is state delivery, not an event FIFO. Product invalidation can retire the content obligation, but any required native clear/hide barrier remains due until its own correlated disposition. At expiry retire delivery epoch and enter bounded recovery with `native_acceptance_stalled`. After acceptance, the native due deadline independently governs handoff/visibility; neither producer-ready nor application of an older scene satisfies it.

Keep actual production startup readiness budget **15 s**, now covering owned prepare/spawn/handshake work too; any timed-out late spawn remains owned and must be reaped before retry. Shutdown: **3 s total** graceful request+ACK+exit, then **1 s** terminate grace, then kill with **2 s** exit-confirmation budget; pipe-reader cleanup at most **1 s**, independent from proof of process exit. These deadlines include, rather than follow, blocked shutdown sends. Cancellation of a write/close future alone is not closure; abort the owned transport and retain any unresolved resource reference.

Automatic recovery: initial launch plus at most **3 replacement attempts** per episode, delays **50/100/150 ms** (preserve existing cadence), and a **60 s absolute episode window** from first failure. Count an opportunity before spawn, including startup failure, crash, runtime absence and ready-then-crash; neither ready nor a fresh connection refills it. Start no attempt after the window/budget closes; terminate an in-progress failed recovery at window expiry. Circuit opens with actionable retry status. User OFF/shutdown always suppresses restart.

Refill only after **60 continuous seconds** of owner health without failure, with at least one current valid handoff or confirmed requested hide after recovery, and no overdue work. Startup-ready or mere intentional idle without that evidence cannot refill. Explicit user retry opens a new episode only after old child exit is confirmed. Do not immediately refill repeatedly within a failing episode. If kill/exit cannot be confirmed, retain the handle, report `termination_unconfirmed`, open the circuit and **do not spawn a replacement**. No guarantee is made against a driver/OS-wide hang.

## 6. Bound inventory and overload terminality (C6)

Values marked **selected** are proposed engineering ceilings, not measured optima or landed implementation. Numbers owned by Audio are incorporated by reference, not forked here. Byte limits below mean application-controlled retained payload/serialized bytes, not Python/Rust allocator or driver/OS total RSS. Count actual copies separately; aliased immutable allocations once. Reservation transfers retain original age. Report actual allocator/VRAM exposure separately in V.

| Item | Existing value / selected target; owner | Pressure outcome, reset and rationale |
| --- | --- | --- |
| Semantic parent admission | Existing translation owner uncapped. Consume #134 C8 peer **8 waiting / 12 s admission TTL**, existing peer semantic execution; #144 S6 SELF speech **2 running + 8 waiting / 12 s**, all targets share parent slot | Audio owner retires oldest not-started to explicit expired/source-only outcome, releases ordering without another LLM. Scope-local reset. No overlay duplicate parent FIFO |
| Parent-batch output | Consume #134 C8/C12 and #144 S6: **8 unsent + 1 active per speech scope/destination**, original parent order; local acceptance/rejection closes slot | Oldest wholly unsent batch explicitly `output_overload`; already submitted pages/history not replayed. Manual separate: selected **8 unsent + 1 active per destination**, reject incoming manual batch at capacity, do not evict admitted manual with speech pressure |
| Pending output payload reservation | Selected **1 MiB retained UTF-8 text + metadata per parent batch**, including all source/target/progressive revisions retained concurrently; **9 MiB aggregate per origin scope/destination**, covering eight unsent plus one active. Existing OutputRuntime/translation-output handoff owns reservation **before** queue admission; applies to manual, speech and non-overlay destinations alike | Charge actual copies at each owner; shared immutable payload once until final owner releases, while each destination still reserves its quota. Revision replacement cannot allocate an unreserved second full batch. Reject oversized incoming batch as `output_payload_exhausted` before acceptance, without truncation or ASR/LLM retry. Under aggregate pressure, speech follows oldest-wholly-unsent rejection above; manual rejects incoming. Release on application/rejection/terminal disposal, never on mere queue transfer; upstream provider-result buffers retain their separate existing owner bounds |
| Local overlay transaction | Selected **1 active** reducer transaction, no additional event-task queue; input supplied through bounded parent slots. Provisional ingress is one latest mutable update per live parent, with final/control priority | Reject before application if local reservation unavailable; never call lost semantic event accepted. Terminal/clear uses reserved state, not another LLM/task |
| Presenter entries / payload | Existing entry/timer maps unbounded, display 2 rows. Selected **64 live/reentrant entries**, **1 MiB UTF-8 per event**, **16 MiB aggregate retained text/metadata payload**; presenter | Prune eligible retired/expired state first; otherwise reject incoming overlay application with overload, leaving valid manual/other destinations intact. Do not silently truncate text or evict an in-use identity. At most one expiry timer per admitted entry |
| Scene / intent mailbox | Existing one stored snapshot but unbounded waiters. Selected **1 current immutable scene + 1 active writer reference + 1 pending successor**, **2 drawable blocks**, **1 MiB serialized scene each**, **2 compatibility intents** (SELF/peer); bridge | Share immutable references where possible; at most **3 MiB** scene wire copies. Coalesce unsent full state after reduction; retire superseded display attempt. Over-size delivery explicitly rejects and sends reserved invalidation/hide rather than leave unauthorized old content indefinitely; semantic/UI/history acceptance remains distinct |
| Control/lifecycle reserve | Selected **8 keyed slots × 4 KiB** each per direction for handshake/status/lease/invalidation/shutdown/error/settings control; one sticky current invalidation watermark and terminal cause | Priority over data between writes; coalesce newer state of same kind, never bury terminality in diagnostics. Unexpected non-coalescible control overflow fails connection epoch. One slow active write bounded below |
| Active transport writer | Selected **1 per direction/connection**, plus one **1 MiB** serialization scratch; one authenticated runtime; no extra spawned send tasks | Current scene expires/supersedes while pending; writer revalidates before send. Actual write fallback **5 s** from #134 C10; native control/status write selected **1 s**. Timeout/error makes delivery ambiguous and retires connection, not published success |
| SDK / OS staging | Existing Python defaults and native tungstenite default max write buffer unbounded. Selected max message/frame **1 MiB**, library outbound high-water **64 KiB**, hard queued serialized bytes **1 MiB + 64 KiB**, at most one fragmented message; disable compression for predictable bound | Configure supported library controls explicitly, bound producers independently, use close/abort after write deadline. Close wait selected **1 s** before abort. OS socket/driver staging size is not guaranteed or directly measurable here; no total-memory claim from a library watermark. Reject oversized frame before parsing full unbounded payload |
| Native ingress | Existing full unbounded block Vec despite 2 displayed slots. Selected same **1 MiB / 2-block** scene envelope, **1 pending scene**, **1 active parsed scene**, control reserve above, one parse scratch | Validate before allocation growth, stale revisions discarded, newest valid scene coalesces. Invalid/oversized peer is protocol failure, not success. Epoch retirement clears pending input |
| Native GPU/query/export | Existing persistent texture, per-wait query but no outstanding-work ledger. Selected **1 incomplete producer/query + 1 CPU successor**, **1 persistent associated target**, no retired-generation pool | Keep occupied slot after CPU cancellation; at original deadline fail/recover rather than enqueue more. Release only under §5.2; nominal texture 16.5 MiB excludes driver/compositor allocation |
| Native readiness / no-progress | Existing **50 ms**, **100 ms retry**, **2 s** deadline starts after timeout. Selected same values, deadline from first due work | Lateness preserves resource reservation; cancellations/churn never reset deadline. Completion/current coverage or explicit terminal disposition ends episode |
| Compatibility opportunities | Retain p05 default: **100 ms**, final **5 / 500 ms**, stream **min(4, profile max)**. p10 **10/1000 ms**, p15 **15/1500 ms**, p20 **20/2000 ms**, no_retry **0/0**, one_retry **1/2000 ms** | Existing profile deadline inclusive; no catch-up storm. One schedule per channel, retry mailbox **1**. Same episode preserves consumed count/deadline; new process retires old episode. No default tuning/removal in C |
| Python legacy retry | Existing fallback **2 s / 100 ms**, one task per channel | Retain only on the old matched package before cutover. Contract-capable native has exclusive ownership before attach; no simultaneous Python/native retry. Desktop has neither native nor Python VR bursts |
| Lease / health / teardown / recovery | Selected §4/§5 values; supervisor owns recovery, presenter owns validity | Original deadlines survive queue moves; runtime-ready never replenishes budget. OFF retires relevant scope before physical cleanup |
| Native diagnostic ring / flush | Existing **128** presentation records, **8** per flush, **25 ms** presentation flush budget; fresh and successful audit each **128**. Selected max **4 KiB/record**, max **512 KiB/ring** | Hot path only bounded enqueue; dedicated owned diagnostic writer, no await on render/control. Drop oldest with saturating per-ring dropped count, acknowledge only successful writes. **25 ms** covers all diagnostic write/flush work, not only one existing wrapper; pending pipe cannot own shutdown |
| Python reverse events / logs | Existing bridge/process `Queue()` unbounded; desktop renderer queue **64**. Selected diagnostics **128 × 4 KiB** per reverse source, control reserve separate; retain desktop 64 with explicit overflow | Drop/coalesce diagnostics with counters; terminal current status sticky. Max pipe line **4 KiB**, discard oversized diagnostic line with count; invalid/oversized required control is protocol failure. Failure dumps off operational path; selected bounded **1 MiB** dump and **1 s** flush then counted abandonment |
| Existing Python diagnostics | process **256**, stdout/stderr **100 each**, presenter **30**, removals **50**, bridge **30**, translation/chatbox/STT/native **50 each**; OutputRuntime decisions **4096** | Retain count caps, add **4 KiB/record**, drop counters and bounded failure dump. These are diagnostics, never an acceptance/recovery API |
| Renderer caches | Existing format **32**, layout **512**, line **2048**, block **1024** entries | Preserve LRU counts; selected additional **64 MiB accounted retained cache budget** for owned strings/layout/commands/surfaces. Evict only safe unused cache entries; if backend cannot account an entry conservatively, do not cache it. Driver allocations remain separately reported, not certified bounded by entry count |
| Completed identities / correlation | #134 peer completed publication keys **4096** + live keys; existing SELF committed **1024**, projection tombstones **4096**, presenter closed/preview LRU **64 each** | Preserve owning caps, add frontier rules §3.1; OutputRuntime completed ring selected **4096 per origin scope**, pending keys bounded by admission. No lifetime-growing delivered/closed/scene-terminal/spatial/peer-seen sets. Detailed native correlation/audits **128**, current spatial/reentry records **64**, live parent holes bounded by admission |

No renewal of admission TTL on handoff into overlay. Transport scenes are **derived display state**, not another queue of semantic parent batches: a batch releases at local acceptance/rejection and the bridge owns only the latest scene/intent reservation. Controls, diagnostics, identity records and renderer caches have separate declared accounting; increasing any cap to hide failure requires amendment. Full implementation must demonstrate these bounds at cap and cap+1 with terminal receipts, not merely list constants.

## 7. Audio and other consumer alignment (C7)

Consumed authorities:

- [#134 AUDIO-LISTEN-1](https://github.com/kapitalismho/PuriPuly-heart/issues/134), canonical body finalized 2026-09-09, updated 07:04:53Z: design complete / implementation authorized; C2/C3 ownership and source time, C8 admission, C10 fallback bounds, C12 output. Older `scope-r1` comments explicitly not frozen and withdrawn uniform timeout proposals are not authority.
- [#143 AUDIO-SHARED-1](https://github.com/kapitalismho/PuriPuly-heart/issues/143), 2026-09-09, updated 08:02:25Z: shared source/request/provider implementation, not shared SELF/LISTEN product policy; production extraction requires accepted #135; I2 is shared-ready.
- [#144 AUDIO-SELF-1](https://github.com/kapitalismho/PuriPuly-heart/issues/144), 2026-09-09, updated 08:09:16Z, S4–S6: speech-only abort, graceful named-request permits, dual-target publication and separate resource limits; SELF cutover requires #143 I2.

| At work-start actually landed | Planned, not landed |
| --- | --- |
| One translation lifecycle/projection/output composition; channel generation/order, parent/child/target identities; semantic-done versus closed separation; dual-target concurrency/progressive snapshots; late-primary currentness; peer chatbox denial | Audio bounded parent admission/TTL and bounded destination batch handoff are absent in these owners; output tasks/delivered IDs and closed-parent IDs grow |
| TALK OFF abort intent in `SelfCaptureSessionOwner._release_plan`; manual bypasses capture/STT | `self_capture_provider.py` abort/reset → `self_translation_channel.reset_provider_channel` → `cancel_pending(channel='self')` still catches manual; speech-origin correction not landed |
| Existing provider/resource handles and shared local GPU coordinator; SELF merge/preview behavior | Audio common request exchange/cutover and named retired-request graceful permits are not established by this baseline; low-latency retired finals still blanket-rejected |
| Meaningful presenter age/identity fixtures, native owner, visibility reconciliation, bounded caches/diagnostic counts | Slow transport isolation, finite byte/identity bounds, cancellation-only due deadline, native validity lease and external stale-progress recovery not implemented |

Upstream recognition limits (8 unsent sealed segments, 12 s seal TTL, 256 receive events, 1 MiB text/provenance, 256 runs/parent), SELF retained PCM 2,880,000 sample-equivalents, Audio endpoint values and provider setup/final/recognition retry belong to those authorities. They are **not** overlay defaults or reasons to reset content age. Existing finite translation bounds win; only absent bounds take #134's 60 s per-child watchdog. Do not reopen recognition protocol numbers here.

SELF primary owns established UI/overlay route; secondary-first chatbox snapshots remain legal, then combined configured target order. Both targets and later admitted parents can execute independently. Manual is not put behind speech admission or cancelled by STT-only change; LISTEN never cancels SELF; peer never falls back to chatbox. System disclosure stays separate. Desktop shares semantic acceptance/currentness/expiry and destination isolation, but not VR pose, GPU retry or OpenVR health semantics.

Shared seam ownership after approval: #148 is primary owner of OutputRuntime/projection/presenter/bridge acceptance, receipts and any still-missing parent/output admission integration required for its acceptance. #149 owns native and existing process-supervisor health/termination. #135 owns LISTEN's upstream admission policy implementation; #144 owns SELF's speech-scope binding/caller migration. If Audio lands an equivalent shared seam first, #148 consumes its exact revision and validates it instead of duplicating it. Scope cancellation APIs are implemented once; no parallel PRs may independently mutate the same shared owner. A/N do not wait for full Audio convergence or E's HMD access.

## 8. Protocol support floor, migration and rollback (C8)

Selected support floor: **new matched manifest protocol 7** for the contract-capable Python/native package. Current protocol 6 remains only the historical rollback pair, not a hidden degraded mode under the new contract. The version change is needed for mandatory scoped invalidation/current-status/lease semantics; optional ignored fields would make falsely successful mixed deployments possible.

Before connecting/attaching publication, require authenticated runtime instance + generation and capability `execution_contract {version: 1, revision: "r1"}` plus existing exact exclusive native-retry capability `{version: 1, ownership: "exclusive"}`. Both sides reject wrong protocol/instance/generation, missing required lease/status/invalidation support or unsupported capability. No absent-identity compatibility acceptance on the new native path. Capability names are selected wire requirements; changing them requires coordinated Python/native protocol change, not independent guessing. Later compatible editorial amendments need not alter wire version; behavioral/capability changes must version appropriately.

| Combination | Disposition |
| --- | --- |
| New Python + contract-capable native protocol 7 | Supported software contract after OA/ON/V evidence; exclusive native retry from startup, no fallback race |
| New Python + old native protocol 6 or missing capability | Fail fast `unsupported_binary`, no active overlay; other destinations remain usable |
| Old Python + new native protocol 7 | Exact manifest mismatch, no active overlay |
| Old matched Python/native protocol 6 | Rollback support only, with known baseline gaps and no OVR-CONTRACT-1 conformance claim |
| Desktop packaged with new Python | Shared protocol schema/version updated atomically in desktop manifest/runner if consumed there; desktop-specific capability advertises its actual semantic/control support, not native GPU/lease capability. No VR capability requirement or burst on desktop |

Migration order:

1. **G-C approval** of this exact revision/support scope. Pin actual branch/deployed pair and any newer Audio seam revisions; distinguish characterization from conformance.
2. #148 implements local acceptance, scoped receipt/parent handoff and bounded writer under existing scene/backend policy; #149 independently implements native resource/due/status/lease and supervisor recovery. Freeze shared protocol 7 envelope in one integration-owned change before conflicting schema edits. No production cutover of a half-pair.
3. Integrate exact matched Python/native/desktop pair, all local mutation paths, output predecessor closure and bounded replay; remove old awaited-transport path and optional/missing-identity native compatibility only at this matched cutover. Preserve P05/profile defaults and rendering backend. If Audio already implements the seam, reuse it.
4. Run common OC/OA/ON cases with current backend first. #147's approved G-H determines retained handoff or conditional #150. Any backend adopts §5/§6 safety/receipt contracts, not another recovery owner.
5. #151 validates integrated compatibility, package/protocol mismatch behavior and actual environment exposure; #152 alone accepts retaining/reducing/replacing/removing historical protections and validates its delta. No API-success/HMD shortcut.

Rollback: stop ingress and retire output/connection epochs; boundedly terminate old child and confirm exit; replace **both** Python and native packaged artifacts (plus matching desktop/protocol resources) with the recorded previous matched pair; restart with a fresh runtime epoch and revalidated current state. Do not replay UI/history/chatbox or old compatibility ticks. If old child cannot terminate, stop at circuit-open, not overlapping binaries. No persistent settings migration is selected here; keep existing valid settings/profile. Record rollback reason, exact hashes, residual baseline gaps and downstream receipts invalidated by it. Never downgrade only wire format while leaving a mismatched binary alive.

## 9. Common acceptance cases and evidence ledger (C8)

These IDs are normative shared clauses for A/N/B/V/R. Existing fixtures are reusable **ingredients**, not blanket passes for the expanded contract. Deterministic timing/identity/resource assertions must use real owners with controlled clocks/barriers; fake GPU/HMD paths prove only software behavior.

| ID | Required counterexample / pass condition | Existing reusable entrypoint; remaining evidence |
| --- | --- | --- |
| OC01 | Stalled socket through actual projection→output→presenter→bridge: local applied receipt, next translation and other destinations continue; pending count/bytes remain bounded | `test_translation_turn_owner::test_blocked_overlay_does_not_delay_next_single_target_self_llm` only proves early semantic release; new actual-chain slow-socket acceptance required |
| OC02 | Provisional→final→clear under pressure/cancellation: final semantically applied once, clear watermark wins, unmet intent preserved or explicitly retired, bounded wire | `test_overlay_presenter` same-turn preview promotion/stale clear; bridge initial/live revision fixture. Replace every-tick transport assumption with intent coverage, not loss of final transition |
| OC03 | Same-text different turn, preview-only same-turn reentry and old late translation: correct identity/appearance, current caption and spatial meaning | Presenter `test_presenter_pair_state_same_text_different_turn_replacement_still_publishes_and_logs`, `test_presenter_ignores_stale_self_active_update_after_preview_only_retirement_but_allows_newer_final`; dual-target newer-surface fixture |
| OC04 | Preempt every readiness attempt before 50 ms from the **first** due work: current coverage or explicit failure by original 2 s; old submissions/churn cannot renew | Native `production_owner_preemption_preserves_due_and_completes_on_pending_snapshot`; existing no-progress fixture first waits 1500 ms, so cancellation-only starvation needs added coverage |
| OC05 | Cancel after real GPU enqueue, late completion, successor and query reuse: no second outstanding producer, wrong-generation readiness, premature disposal or old-pixel claim | Native owner teardown fixtures + `windows_graphics_real_readiness_reports_ready_and_honours_cancellation`; existing real query test cancels before call, not in-flight retirement. Windows resource proof still required |
| OC06 | Long idle/hidden/no-content/pose-unavailable versus blocked render/API: only true due/unresponsive failure recovers; pose retry preserves identity | Native `production_owner_event_pump_preserves_idle_hide_tail`, `unavailable_spatial_pose_is_consumed_without_blocking_texture_submit`; update pose-safe deferred-anchor semantics deliberately |
| OC07 | Stop stdout/stderr readers and flood reverse diagnostics/control no-ops, then drip buffered old increasing status records after owner hang: publication/control progress or finite failure; no health renewal from arrival, bounded buffers/drop counts | Native diagnostic pending/error helpers and event-flood fixtures; Python process/bridge health-challenge and written-but-unaccepted-current-scene scenarios still required. Log presence is not pass evidence |
| OC08 | OFF during every restart/callback window, lost clear while connected, app-loop stall, expiry before startup replay: no invalid scope revival; finite lease hide attempt, manual/other channel survive | Generation-start stale-boundary, runtime detach-before-broadcast and presenter TTL fixtures; add lease delays/replay and scoped OFF combinations |
| OC09 | Continuous crash→ready→crash, absent runtime, hung child, kill failure and late spawn: initial+3 max, 60 s window, no duplicate child or ready refill | Process-manager kill escalation/application restart fixtures; ready-flap and unconfirmed-exit bounds are new requirements |
| OC10 | Long churn beyond 64/1024/4096 retention capacities and maximum live reservations: bounded state, no stale resurrection after eviction, explicit overload and safe GPU/cache retirement | Presenter tombstone-overflow fixture protects stale semantics but uses unbounded terminal set; output lifetime identity fixture likewise not memory proof. Add combined frontier/cap+1 evidence |
| OC11 | Actual Windows build/backend stage correlation, failed show/hide, ambiguous send/submit and independent HMD observation | Fake OpenVR SetOverlayTexture test is not real API proof; real D3D query is not HMD proof. Current backend copy-complete/HMD default not_observable; record actual runtime visibility separately |
| OC12 | SELF speech/manual/peer and desktop, dual-target secondary-first, destination rejection, STT-only replace and global shutdown | `test_dual_target_translation_lifecycle::test_end_to_end_secondary_first_publishes_progressive_parent_snapshots`, projection peer denial, generation-start desktop/native fixtures; new speech-only cancellation and mixed binary checks required |

### 9.1 Executed during this contract investigation

Setup: work-start source above; existing shared Windows Python environment located by `uv python find`; **Python 3.12.10**, `uv 0.9.17`; `PYTHONPATH=src`; no production source edits or native build. Reused fixtures were sufficient for their existing claims; no new permanent tests were added to a design-only change.

Command (repository root):

```text
uv run --no-project python -m pytest
  tests/core/runtime/test_output_runtime.py::test_output_runtime_delivers_channel_separate_overlay_events_in_order
  tests/core/runtime/test_output_runtime.py::test_output_runtime_accepts_distinct_parent_presentation_revisions_once
  tests/core/runtime/test_output_runtime.py::test_output_runtime_denies_peer_chatbox_without_user_text
  tests/core/test_overlay_bridge.py::test_overlay_bridge_does_not_send_stale_initial_snapshot_after_newer_live_snapshot
  tests/core/test_overlay_presenter.py::test_presenter_does_not_reorder_existing_turn_when_translation_updates
```

Executed as one command: **5 passed in 1.38 s**. This confirms existing ordered channel events, distinct presentation revisions, peer route denial, authenticated replay order and late-translation order only. It does **not** mark OC01–OC12 conformant.

A throwaway in-memory Python probe reused `_BlockingInitialSnapshotConnection` against actual `OverlayBridge._handle_connection` and `replace_snapshot`: hold initial revision-0 send at its event barrier; schedule revision 1; yield one loop turn; observe replacement pending and local snapshot still revision 0; release send; await completion; observe revision 1. Output:

```text
stalled initial send: replacement pending, local snapshot revision=0
send released: replacement complete, local snapshot revision=1
exit=0
```

This reproduces bridge-lock acceptance coupling; it is not a measured end-to-end latency or a field-cause proof. The existing replay fixture's order is valid, but preserving that order must not require holding local acceptance behind socket I/O.

A throwaway arithmetic probe evaluated **48** combinations of native challenge origin (0/10/100000 s), response delay (0/0.1/1/4 s) and remaining validity (0/0.1/2/10000 s): `n0 + min(r, 3)` never exceeded original expiry or `n0 + 3`. This checks the proposed conservative duration formula, **not** a running lease protocol, suspend handling or real clock-rate calibration.

Not run: native build/Rust suite, actual Windows driver cancellation/retirement, OpenVR runtime/HMD observation, log-pipe/ready-flap full conformance, long-session memory campaign, affected-user A/B, deployed mixed-package acceptance. Those are downstream OC evidence, not silently passed prerequisites. No files/scripts generated by the probes require cleanup.

## 10. Decision rationale, amendment and completion gate (C9)

| Selected | Rejected / reason | Amendment trigger |
| --- | --- | --- |
| Existing owners with bounded local acceptance then latest scene/intent writer | Fire-and-forget before reducer application loses truthful acceptance; exhaustive scene FIFO multiplies latency; new manager duplicates authority | Actual composition reveals a missing authority that cannot be expressed by existing owners |
| Semantic receipts separate from display supersession | Exactly-once physical final display is unsupported and conflicts with current-caption UX | Explicit new product requirement with evidence/approval |
| Duration/challenge lease, 3 s degraded ceiling | App-only expiry fails during app-loop stall; receive-time TTL rejuvenates old content; cross-clock subtraction unsupported | Measured legitimate scheduling exposure shows unacceptable early hides, or clock/API contradiction; revise lease only, not Audio endpoints |
| One outstanding producer, persistent baseline target, API-only handoff claim | CPU cancellation as GPU termination; COM clone as old pixels; arbitrary buffering as consumer release | #147 demonstrates API contradiction or #150 selects a proven handoff needing a narrowly changed bound |
| Due-start 2 s deadline and external owner status | Timeout-only start permits cancellation-only starvation; thread heartbeat/log activity can conceal hung owner | Evidence identifies a different finite healthy-work envelope or a status mechanism contradiction |
| Existing 3-retry cadence with finite 60 s window and stable-progress refill | Ready-triggered refill permits infinite flaps; automatic overlap after failed kill risks duplicate child/resources | Measured startup/recovery exposure requires revised budget, with explicit safety/stop semantics |
| Matched protocol 7, fail-fast mixed binaries | Optional fields on protocol 6 cannot prove required lease/receipt safety; fallback dual retry races ownership | Approved compatible negotiation proof; never infer from source version alone |
| Keep P05 and alternate profiles; no HMD cure claim | Tune/remove compatibility protection based on temporal correlation or passing mocks | G-H evidence, V baseline and R explicit delta acceptance |

Independent draft review adjudication: accepted three findings before finalizing r1: pending output batch bytes lacked an upstream reservation; written-but-unaccepted native scenes lacked a concrete deadline; buffered old status could falsely renew health. §6 now reserves batch payloads before handoff; §5.3 selects a non-renewable 2 s acceptance interval and challenge-issuance-relative 3 s owner freshness. These are contract corrections, not claims of implemented runtime fixes.

Both independent reviewers subsequently verified their respective corrections as resolved by targeted document reread; no validation commands or production checks were claimed by those reviews. A further throwaway arithmetic check confirmed that delayed pre-hang status responses cannot extend freshness beyond their last challenge issuance + 3 s, and successive writes leave the first unresolved acceptance deadline unchanged. These are logical checks only; running supervisor/protocol conformance remains OC07/OC09 downstream work.

Maintainer approval record:

```text
G-C / OVR-CONTRACT-1 r1: design-frozen
Support scope: Windows x64 SteamVR/OpenVR D3D11, head/spatial; desktop preserved
Approved clauses: §§1–8 and OC01–OC12, including selected bounds/lease/protocol floor
Consumed authority: #134 AUDIO-LISTEN-1, #143 AUDIO-SHARED-1, #144 AUDIO-SELF-1 (dates above)
Source baseline: 4e967df9d03649106faa8348c3ec611009529ffe
Deployed pair / environment: UNKNOWN / not verified; not certified by this approval
Evidence: §9.1; unexecuted OC/native/HMD cases remain downstream acceptance work
Approver / date: @kapitalismho / 2026-09-09; explicit approval in task conversation
Approved pre-publication draft SHA256: dbce885550e665a2506290f5bed0addde1912f23ce7261bb9bf416beae9876a3
Supersedes: none
```

Completion: **OVR-C contract ACCEPTED / design-frozen; G-C OPEN**. Missing deployed pair remains unverified evidence, not hidden under design approval. No production implementation or HMD result is claimed. The complete contract and approval receipt are published in #146 and indexed in #145; GitHub issue state alone is not the approval authority. A revised approval must name changed clauses, rationale and affected A/N/B/V/R receipts; unaffected receipts do not restart. Architecture drift introduced by this change: none (documentation only). Future implementation extends existing boundaries rather than adding a root manager.
