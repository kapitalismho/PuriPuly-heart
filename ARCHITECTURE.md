# Architecture

Implementation-oriented system map for PuriPuly Heart.

Use this document to locate:

- runtime owners,
- data handoffs,
- ports and adapters,
- composition points,
- lifecycle boundaries,
- relevant source files.

For detailed behavior, read the referenced code and tests.

## Architecture Model

- Python desktop application is the main system.
- Runtime state and resources belong to explicit owners.
- Owners depend on ports, not concrete providers.
- Adapters connect ports to UI, audio, providers, storage, native processes, and external systems.
- Composition selects adapters and connects owners.
- Core code must not depend on UI or composition code.

Dependency direction:

```text
UI / infrastructure adapters
            ↓
application ports and services
            ↓
core runtime and domain contracts
```

## System Boundaries


| Boundary             | Responsibility                                                    |
| -------------------- | ----------------------------------------------------------------- |
| Python application   | UI, channels, providers, translation, output, settings, lifecycle |
| OS audio             | Microphone, output loopback, process capture                      |
| STT backends         | Audio to transcript events                                        |
| Translation backends | Transcript to translated text                                     |
| VRChat OSC           | Bidirectional control, canonical state, chatbox output, mute      |
| Overlay processes    | Desktop and VR subtitle presentation                              |
| GPU worker           | Native local GPU ASR                                              |
| Broker               | Managed identity, entitlement, credentials, telemetry             |
| Local storage        | Settings, secrets, diagnostics, model assets                      |


Broker is a control-plane dependency, not part of the normal utterance data path.

## Runtime Ownership


| Owner                   | Owns                                                       | Key path                                                                  |
| ----------------------- | ---------------------------------------------------------- | ------------------------------------------------------------------------- |
| UI application boundary | UI-facing application operations                           | `app/services/ui_application.py`                 |
| Settings owner          | Canonical settings, persistence, projection, rollback      | `app/services/canonical_settings_persistence.py` |
| Runtime pipeline        | Active runtime component set                               | `app/wiring_runtime_pipeline.py`                    |
| Self capture owner      | Microphone source and capture lifecycle                    | `core/runtime/self_capture.py`                       |
| Self translation owner  | Self STT events, turns, state, output projection           | `core/orchestrator/self_translation_channel.py`      |
| Peer capture owner      | Target, source, VAD, task, provider attachment, generation | `core/runtime/peer_channel.py`                       |
| Local ASR runtime       | Local recognition channels and backend transitions         | `core/runtime/local_asr_provider_runtime.py`         |
| Managed local translation | Gemma provisioning, readiness, backend, prefix, and process lifecycle | `app/services/managed_gemma_translation.py` and `core/local_translation/runtime.py` |
| Translation turn owner  | Request lifecycle, cancellation, stale-result rejection    | `core/orchestrator/translation_turn.py`                 |
| Output runtime          | Routing, delivery tasks, destinations, delivery history    | `core/runtime/output.py`                              |
| Overlay owners          | Overlay selection, process lifecycle, state, calibration   | `app/services/overlay_application.py`            |
| Managed-account runtime | Authentication, entitlement, usage, credential release     | `app/wiring_managed_account.py`                      |
| OSC control runtime   | Receiver lifecycle, routing, state publication, restart    | `app/services/osc/control_runtime.py`                        |
| OSCQuery service      | Zeroconf discovery, receiver advertisement, OSCQuery tree | `core/osc/oscquery.py`                                        |
| Shutdown adapter        | Ordered application teardown                               | `app/adapters/application_runtime_shutdown.py`      |
| VRChat scene owner     | Process-lifetime instance population, immutable snapshots | `core/vrchat_scene_service.py`, `core/vrchat_scene.py` |


Ownership may span several processing stages. Do not assume one owner per pipeline stage.

## Data Handoffs

### Self speech

```text
microphone
→ normalized 16 kHz mono frames
→ generation-owned VAD events
→ source-ordered audio-segment identities
→ scoped recognition updates and terminals
→ self translation turns
→ publication intents
→ output runtime
→ UI / chatbox / overlays
```

Primary coordinator:

```text
SelfTranslationChannelOwner
```

### Manual text

```text
UI intent
→ final self transcript
→ manual translation turn
→ self publication path
```

Manual text bypasses capture and STT.

### Peer speech

```text
loopback or process audio
→ peer capture, acoustic observation, and LISTEN delivery controller
→ scoped provider turn updates and terminal receipts
→ source-ordered peer transcript admission and translation
→ publication intents
→ output runtime
→ UI / overlays
```

Peer output must not reach the VRChat chatbox.

### Peer audio ownership

Physical microphone, loopback, and process callbacks assign capture-epoch, callback-order, source-sample, and monotonic ranges before queue admission. A full callback queue is therefore an explicit known-loss interval on the next admitted frame; native/status failures begin an unknown-loss epoch instead of being reinterpreted as silence or wall-clock time.

Desktop normalization is mono-first and produces one 16 kHz normalized sample coordinate while retaining the corresponding source-sample and monotonic range. Orderly end-of-stream flushes resampler residue and labels any VAD padding as context-only. Abort or discontinuity does not flush the resampler: already emitted PCM remains content, buffered or filter-resident accepted ranges become explicit failed ranges, and the affected segment seals as failed. A process terminal marker never evicts admitted PCM from the capture queue.

`PeerCaptureSessionOwner` owns a `PeerAudioSegmentLedger` for each active capture lifetime. The ledger assigns stable segment IDs and source order, freezes the effective provider attachment, activation generation, delivery profile, hangover, and resolved settings at segment start, and separates content, copied prefix context, synthetic context, and failed ranges. Runtime status reports requested and effective delivery settings separately while a transition is pending. Endpoint-only changes rebind the next segment's policy without changing the capture generation or rewriting an open segment; a physical source, target, sample-rate, or explicit restart change invalidates the generation and terminalizes every unfinished old-ledger slot once before replacement. A segment follows `open → sealed → terminal`; early terminalization of an open segment is rejected. Terminal receipts retire exactly once in source order, with live snapshots pruned on retirement and recent dedupe receipts retained in a bounded 4096-entry window.

Capture iteration and acoustic segmentation begin after target resolution without waiting for provider or Smart Turn setup. Provider readiness is a distinct gate in the bounded serialized dispatch queue: it admits eight wholly unsent sealed segments in addition to the downstream-active and current capture segments, exact prefix or synthetic context from each ownership snapshot, and 32 reserved terminal/control events. Retained PCM is bounded by those segment slots, each segment's six-second source range, and configured prefix; there is no shorter aggregate-content cap inside that count and freshness envelope. One `ListenDeliveryController` owns both OFF and ON acoustic endpoints. Before four seconds OFF and unsupported-language snapshots use persisted hangover; supported fixed-language ON submits one 224 ms-frontier probe to the single `SmartTurnInferenceOwner`, seals at 512 ms only for complete evidence available strictly before that deadline at the pinned language threshold, and otherwise seals at 800 ms. At four seconds every profile immediately uses the preserved 224 ms pause budget, while the independent six-second source-content deadline seals even without later callbacks. Natural endpoints reset the bounded eight-second Smart Turn context; synthetic rollover retains it, and capture gaps invalidate it. Queue pressure above eight eligible recognition segments retires the oldest as `overload`; each wholly unsent sealed segment has its own twelve-second seal-age TTL and then retires as `expired_before_recognition`. Provider latency therefore does not suspend callback-range progression, acoustic segmentation, Smart Turn inference, or source-time delivery deadlines.

A naturally finite capture dispatches every accepted owned event, becomes honestly inactive, and drains the provider instead of aborting it. Stable segment publication remains available during that drain; a scoped authoritative final or empty receipt wins. If the legitimate provider drain returns without a scoped terminal, the unresolved sealed segment retires as failed with `provider_drain_without_scoped_terminal`, never as successful empty. User stop, source failure, and provider failure retain their abort/cancel semantics. An `OFF` transition may keep an eligible local backend loaded only after invalidating its publication generation and aborting peer ingress, so retained resources cannot publish late peer output.

LISTEN and SELF both use `ScopedRecognitionEngine` through the existing local-ASR runtime and provider handle. `OwnedVadEvent` retains segment ownership through this boundary; only a scoped recognition terminal can retire the matching source slot or admit a final transcript. Concrete sessions implement `STTScopedTurnSession` and preserve native request/item/task provenance separately from application identity. Each concrete route maintains one native parser with client-specific product consumers and retention profiles; the former SELF `ManagedSTTProvider` lifecycle has been retired.

SELF capture owns a generation-bound `PeerAudioSegmentLedger` and a nonblocking serialized recognition dispatcher. The dispatcher permits exactly eight wholly unsent recognition segments, ages them from their original source admission time for 12 seconds, and retires capacity/TTL losers with explicit scoped failure terminals. Its required rejection/failure operations cross `SelfCaptureVadSinkAdapter` and `SelfTranslationChannelOwner` into `LocalASRProviderRuntimeOwner`, which resolves the current or retained recognition engine by the segment's frozen provider/settings scope; a missing operation is an error, never a silent discard. Retained PCM and control-event exhaustion fail the affected scoped turn immediately; accepted stable text may close as degraded/incomplete, while a request with no stable authority closes as failed. Every scoped terminal retires its ledger slot, and contiguous capture ranges coalesce so an ongoing segment does not retain one metadata object per callback. Recognition failure faults and closes only the current SELF capture generation; manual text and peer ownership remain independent.

Scoped SELF contributions carry the resolved provider/settings scope. A scope change is a merge barrier, while a same-scope provider epoch rotation remains merge-compatible. Acoustic segment IDs map to the current publication ID until the real `SpeechEnd`, so a successor arriving before that endpoint receives the configured post-end grace rather than awaiting the watchdog. Shared channel callbacks project scoped readiness as the deduplicated SELF `STREAMING` session state (including the one-time PuriPuly ON disclosure), then project failed or failure-bearing terminals as `DISCONNECTED`. User-visible failures use the typed `UserErrorReport`/message-key contract; internal terminal reasons remain sanitized diagnostics and are never published as raw UI strings.

`ProspectiveSpeakerTransitionReceiver` accepts injected source-scoped hypotheses through the peer capture owner and the existing delivery controller. Its receipt distinguishes a prospective seal at the current accepted frontier from an already-separated, too-late, invalid, duplicate, or retracted hypothesis. This does not activate a speaker producer, reset its reference, or partition previously recognized text.

`PretranslationOwnershipOwner` consumes the same hypothesis stream independently of C13 cut receipts. Default composition leaves it disabled. When enabled, a sealed Deepgram Nova-3 scoped terminal's preserved word times are partitioned at translation admission into ordered ownership children of the existing parent; other providers omit timing and remain unsplit. Mixed, unmapped, or straddling tokens stay UNKNOWN. Confirmed OTHER-to-OTHER keeps distinct local groups. Units are immutable once the translation parent is admitted.

`SmartTurnInferenceOwner` constructs an ONNX Runtime CPU session for the pinned v3.2 CPU artifact with two intra-op threads, one inter-op thread, and sequential execution. It owns optional download/loading, one setup attempt, and one actual executing inference with no pending queue or automatic retry. Logical pause, segment, activation, OFF, or shutdown retirement revokes result authority without treating cancellation as native completion; an occupied native setup or inference call is retained until its actual completion and cannot be replaced. Shutdown closes logical admission immediately, while a shielded named cleanup task survives cancellation of its waiter and reclaims a session constructed late. Requests bind activation, segment, pause, context revision, source frontier, configuration snapshot, and input revision. The controller records evidence completion independently from the input frontier and rejects non-finite, failed, stale, or late evidence.

### Managed translation

```text
managed authentication
→ Broker entitlement or credential release
→ provider runtime activation
→ normal translation request path
```

### VRChat scene context

```text
VRChat process lifetime
→ log tailer
→ whitelist parser
→ member-set trust
→ immutable snapshot
→ request prep
→ structured scene to LLM
```

- Owner shared across pipeline rebuilds; no audio, VAD, or OSC dependency.
- Count includes the local user; only `ready` snapshots expose it. Names and raw logs remain local.
- Rendered as sanitized `<scene>` prefix in the translation user message; absent equals no scene.
- Custom HTTP extensions never receive scene data.

## Ports and Adapters


| Boundary        | Contract                                             | Implementations                          |
| --------------- | ---------------------------------------------------- | ---------------------------------------- |
| UI application  | `UiApplicationPort`                                  | Flet application boundary                |
| UI presentation | `UiPresentationPort`, `UIEventBridgePort`            | Flet presentation adapters               |
| Audio capture   | Capture and VAD ports                                | Microphone, loopback, process capture    |
| STT             | Provider and local ASR ports                         | CPU ASR, GPU worker, remote STT          |
| Translation     | `TranslationRequestPort`                             | BYOK, managed, local or remote providers |
| Output          | Publication and destination contracts                | UI bridge, OSC, overlay                  |
| Overlay         | `OverlaySink`, overlay protocol                      | Desktop overlay, native VR overlay       |
| GPU worker      | `GpuWorkerClientPort`, `GpuWorkerProcessFactoryPort` | Native worker process adapter            |
| Secrets         | `SecretStore`                                        | Keyring, encrypted file, memory          |
| Settings secrets | `SettingsSecretsPort`                               | Typed settings projection and mutation owner over the configured secret store |
| Settings UI      | `ProviderSettingsSnapshot`, `GeneralSettingsSnapshot`, `PromptSettingsSnapshot`, `OverlaySettingsSnapshot`, and typed settings intents | Flet settings presentation and `UiApplicationBoundary` |
| Shutdown        | Runtime shutdown ports                               | Application shutdown adapter             |
| OSC control ABI | Stable parameter schema and codec contract           | Control schema and codec                |
| OSC integration | `OscControlApplicationPort`, `OscQueryServicePort`    | OSC control adapter, OSCQuery adapter   |


Multiple adapters on one port are alternatives unless the owner explicitly supports multiple destinations.

Output supports multiple simultaneous destinations.

## Composition

Primary composition root:

```text
src/puripuly_heart/composition/application_runtime.py
```

Responsibilities:

- load settings and secrets,
- construct owners,
- select adapters,
- compose providers,
- compose self and peer runtimes,
- compose output and overlays,
- compose the VRChat OSC control and OSCQuery runtime,
- compose managed-account services,
- install startup and shutdown,
- return `UiApplicationBoundary`.

Composition may construct resources.

Long-lived resource ownership must be transferred to an explicit owner.

## Runtime Pipeline

`RuntimePipelineLauncher` builds and installs the active component set.

Typical components:

- self capture,
- self translation channel,
- peer runtime,
- local ASR runtime,
- STT provider handles,
- translation requests,
- LLM runtime,
- output runtime,
- UI event queue,
- VRChat microphone state.

Provider or settings changes may replace runtime components.

Do not retain references across replacement unless the API explicitly allows it.

## Configuration

### Persisted intent

- Canonical schema: `AppSettingsVNext`
- Owner: canonical settings persistence service
- Persistence: `config/settings_vnext/compat.py` (first-run, current load, recognized vNext-to-vNext migration, R00 archive-then-reset)
- Desktop overlay defaults, limits, presets, ordering, and visual values: `config/desktop_overlay_values.py`
- Provider selection enums and normalization values: `config/provider_values.py`
- Translation model and connection values: `config/translation_values.py`

`SettingsView` consumes only frozen surface snapshots and emits focused typed intents. The settings application owner replays those intents onto the latest canonical settings before persistence and runtime application.

Contains user selections, not active runtime resources.

### Resolved configuration

Converts persisted intent into effective runtime configuration.

Includes:

- provider and model selection,
- local or remote execution,
- capture target,
- overlay target,
- credential source,
- defaults and capability constraints.

Runtime owners should consume resolved configuration.

### Runtime state

Examples:

- active audio source,
- VAD instance,
- capture generation,
- provider attachment,
- translation turns,
- output tasks,
- overlay process,
- provider handles.

Runtime state belongs to its lifecycle owner and is not persisted settings.

When a settings draft exits, the typed intent is persisted and then passed through the provider-apply boundary. For an active Self capture, provider application must converge both the Self capture owner and the Local ASR channel to the requested live runtime signature before the applied signature cache is updated. Idle or disabled Self capture never forces preparation for an unrelated apply, but an explicit STT selection may still prepare the dormant provider without committing a live handoff. A smooth active handoff keeps the current provider until the owning translation channel completes the utterance at `SpeechEnd`; failed, cancelled, or non-converged application leaves the previous cache truth intact.

## Provider Boundaries

### STT

Execution options:

- Python-process local ASR,
- native GPU worker,
- remote provider.

SELF and LISTEN both allocate a provider epoch and open the backend with the explicit scoped projection in `core/stt/backend.py`. `STTSessionEventProjection` allocates exactly that scoped event stream and owns the common turn identity, payload and update sequences, seal state, terminal receipt, retirement, and once-only epoch-end receipt. Provisional or stable provider updates are not application-terminal transcripts.

`ScopedRecognitionEngine` is the sole production recognition owner for both channels. Each channel has separate provider epochs, event buffers, request state, retention profiles, cancellation and consumer policy; physical CPU/GPU resources may still be shared through their existing owners. `STTProviderEventBuffer` bounds native event ingress, and `STTScopedTurnNormalizer` owns text assembly, native-event deduplication, and complete text/language-run conservation.

Stable updates retain the provider's raw cumulative assembly internally and normalize only the complete assembled boundary exposed to consumers. Contribution offsets are then assigned against that normalized whole, so leading/trailing whitespace is excluded without deleting separators between append fragments; a separator that becomes internal on a later fragment belongs to that later contribution. Terminal normalization therefore cannot shift or truncate a contribution, including whitespace-bearing multilingual results. Terminals list the contributions included in their full transcript. The consumption ledger returns every unconsumed contribution slice in source order plus any authoritative terminal tail not represented by a stable contribution, exactly once; consuming an early prefix therefore cannot hide a later terminal-only suffix. Cumulative replacements compare their normalized whole against the already-published normalized prefix; a replacement that contradicts that prefix fails as `provider_stable_prefix_inconsistent`. Protocols that define cumulative sentence joining use that same existing join function for stable projections and terminal text; Qwen Audio projects `_join_sentences` snapshots rather than incompatible raw sentence appends. The assembly byte ceiling accounts prospectively for both the private raw cumulative text/runs and the normalized public representation, so whitespace discarded at the publication boundary cannot accumulate outside the request bound.

Recognition retention limits are supplied by the binding and exposed as current/high-water sample and byte accounting. Streaming routes retain PCM while the concrete adapter writer owns or queues it and release only after actual ordered writer progress; batch/local and custom-offline routes transfer the reservation to their buffered representation and retain it through terminality, charging float32 storage at four bytes per normalized sample only where that is the adapter representation. Custom mode/capability, not its selector alias, determines whether writes release retention. SELF uses a 2,880,000 normalized mono sample-equivalent bound without adding a time endpoint. LISTEN recognition accounting covers the capture owner's complete finite envelope.

Scoped abort invalidates the logical turn and provider-epoch authority before waiting on an in-flight open, write, seal, or final operation. The current session is detached and late native callbacks cannot publish; physically continuing native work remains explicitly owned by bounded cleanup/quarantine and is not reported as stopped. The shared GPU owner admits at most eight pending jobs per channel before its speech-end/sequence FIFO while preserving independent channel cancellation and the single compatible worker. Admission failure keeps the public `pending_capacity` reason through the scoped recognition terminal.

Provider configuration handoff retains an old scoped owner until source-ordered queued segments using that configuration are drained. The shared factory resolves every production SELF and LISTEN selector through scoped recognition; there is no channel-selected legacy fallback. Custom realtime peer configuration rejects an explicit incompatible `turn_detection` before activation.

The configured Deepgram, Gemini Transcribe Live, Soniox, and Scribe message shapes do not provide a reliable per-turn native identifier. Their scoped completion barriers therefore retire the native epoch before another turn, rather than trusting fixture-only IDs or assigning a delayed unkeyed result to a newer segment. Custom realtime retains keyed reuse through the native committed item ID; an unkeyed completion retires its epoch. Qwen ASR requires the documented native item ID. Provider/model configuration and physical resource ownership survive where supported; epoch retirement is not a change to the capture clock or segment identity.

GPU worker split:

- Python adapter: process launch, authentication, requests, heartbeat, cancellation, shutdown.
- Rust worker: device discovery, model activation, native transcription.

### Translation

Provider adapters own:

- authentication,
- endpoint and model mapping,
- request schema,
- provider parameters,
- streaming,
- response normalization,
- provider errors.

The managed local Gemma adapter remains behind `LLMProvider`; its application/runtime owners handle model installation, llama.cpp process health, CPU/Vulkan profile selection, language-pair prefix readiness, and shutdown.

Translation owners retain:

- turn lifecycle,
- cancellation,
- stale-result rejection,
- publication handoff.

Peer final parents enter `TranslationTurnLifecycleOwner` in source order with the existing deterministic language-run/target child identity. Optional pretranslation ownership units further expand those children by unique positional local groups without creating a second parent. Units are admitted only when they reconstruct the accepted terminal text; unaligned or unsupported timing keeps the unsplit parent. Relation labels require covering valid reference evidence before admission, so missing, invalid, overlapping, or UNKNOWN support stays UNKNOWN rather than defaulting to CURRENT. Confirmed local transition hypotheses still split groups and never revise an already admitted parent. The owner bounds waiting peer parents at eight and expires waiting work twelve seconds after admission. Every child terminal path releases the existing semantic predecessor gate; request settings remain admission-time snapshots and scene context remains preparation-time context.

SELF speech parents have a separate finite envelope of two running parents, eight not-started parents, and 12 seconds from speech-parent admission. A dual-target speech turn occupies one parent slot while its child translations share that slot. Manual SELF turns use the same ordered lifecycle but are not cancelled, expired, or evicted by the speech envelope or TALK OFF; they may wait behind earlier admitted work.

## Output

`OutputRuntime` owns:

- route selection,
- chatbox state,
- overlay deliveries,
- UI event bridge,
- delivery tasks,
- duplicate protection,
- destination replacement,
- shutdown cleanup.

Peer UI and the selected overlay destination have independent owned handoff lanes: one active writer and eight waiting parent batches per destination. `TranslationUiMessageQueue` records acceptance, overload, timeout, retirement, and local UI intake submission by parent/publication identity; a one-slot application intake is not an unbounded secondary peer queue. Output handoff releases translation semantics without waiting for physical display. Sink failures are destination receipts, not reasons to replay recognition or translation.

SELF speech chatbox pagination likewise retains at most eight unsent speech messages plus one active writer and expires unsent speech after 12 seconds from output handoff. Overflow retires the oldest unsent speech publication as `output_overload`; expiry is `output_timeout`. Manual SELF messages are not candidates for speech eviction or expiry.

Peer publication carries activation generation and source order through translation, source-only/cancellation fallbacks, and output. Retiring an activation cancels its owned output and rejects late work; completed publication identities have bounded retention, while source-order checks still reject stale callbacks after eviction. Accepted enqueue or a completed sink coroutine is not a remote display acknowledgement.

Caption/overlay enablement gates only destination availability. The application overlay owner no longer disables LISTEN intent or capture when the overlay is disabled; explicit LISTEN OFF remains the capture/publication abort authority. Peer conversation errors carry the same parent, generation, and source-order authority as their source/translation events. Unscoped runtime session-status changes use the separate status path, not the conversation feed.


| Publication       | UI               | Chatbox             | Overlay          |
| ----------------- | ---------------- | ------------------- | ---------------- |
| Self utterance    | Yes              | Yes                 | Yes              |
| Peer subtitle     | Yes              | No                  | Yes              |
| System disclosure | Policy-dependent | Explicit route only | Policy-dependent |


Destination adapters must not bypass routing policy.

## Lifecycle

Every owner of a task, process, source, or provider session must define:

- ingress stop,
- cancellation or draining,
- late-callback rejection,
- resource release,
- restart behavior.

### Stale-work protection

Used mechanisms include:

- generations,
- attachment tokens,
- request IDs,
- current-component checks,
- cancellation,
- stale completion rejection.

Retired work must not mutate current state or publish user-visible output.

### Replacement sequence

1. Stop or freeze ingress.
2. Invalidate previous generation.
3. Cancel or detach active work.
4. Construct replacement.
5. Install replacement.
6. Resume ingress.
7. Release retired resources.

### Shutdown direction

1. Stop application ingress.
2. Stop capture.
3. Cancel translation and provider work.
4. Close output and UI bridges.
5. Terminate child processes.
6. Close managed authentication.
7. Release remaining services.

Use shutdown code and lifecycle tests for exact ordering.

## Async Event Model

- The Python application runs asynchronous runtime work on its `asyncio` event loop.
- Owners create, track, and close their own background tasks.
- Do not create detached tasks without assigning lifecycle ownership.
- Capture, STT, translation, UI, and child-process events cross owner boundaries through ports, callbacks, or owned queues.
- Callbacks must delegate to the receiving owner; they must not mutate another owner's private runtime state.
- Ordering is local to the owning channel or queue. Do not assume global ordering across self, peer, UI, and provider events.
- Runtime replacement may leave old work in flight. Validate generations, attachment tokens, request IDs, or current-owner identity before applying results.
- Late or retired work must not mutate current state or publish user-visible output.
- Blocking model, device, or native work must not block the application event loop; use the established worker, executor, or child-process boundary.
- Shutdown order is: stop ingress, cancel or drain owned work, close external resources, then clear runtime references.
