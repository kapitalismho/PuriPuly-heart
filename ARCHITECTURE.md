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

Capture callbacks assign capture-epoch, callback-order, source-sample, and monotonic ranges before queue admission. A full queue is an explicit known-loss interval on the next admitted frame; native/status failures begin an unknown-loss epoch instead of being reinterpreted as silence.

Normalization is mono-first to one 16 kHz coordinate while retaining the source-sample and monotonic ranges. Orderly end-of-stream flushes resampler residue; abort or discontinuity seals the affected segment as failed without flushing, and a process terminal marker never evicts admitted PCM.

`PeerCaptureSessionOwner` owns a `PeerAudioSegmentLedger` per capture lifetime (`core/audio/ownership.py`). The ledger assigns stable segment IDs and source order, freezes the provider attachment, activation generation, delivery profile, hangover, resolved LISTEN onset/derived exit thresholds, and other settings at segment start, and separates content, prefix/synthetic context, and failed ranges. Segments follow `open → sealed → terminal`, terminalized exactly once in source order. Endpoint-only changes rebind the next segment's policy; physical source, target, sample-rate, or explicit restart changes invalidate the generation and terminalize unfinished old-ledger slots once.

Segmentation starts after target resolution without waiting for provider or Smart Turn setup; provider readiness is a separate bounded dispatch gate. One `ListenDeliveryController` owns both OFF and ON acoustic endpoints (`core/audio/listen_delivery.py`). Under the LISTEN-ENDPOINT-7S-HYST010 amendment, LISTEN onset remains three observations at the configured 0.10–1.00 threshold, while active continuation uses `max(0.10, onset - 0.10)`. Pre-four-second endpoint profiles are retained, qualifying observed pause uses 224 ms from four seconds and 128 ms from six seconds, and an independent timer hard-seals accepted ownership at seven seconds. Rollover preserves acoustic continuation without manufacturing silence or reacquiring onset. A naturally finite capture drains the provider instead of aborting it, with a scoped authoritative final winning; a drain returning no scoped terminal retires the segment as failed. User stop, source, and provider failures keep abort/cancel semantics, and an `OFF` transition invalidates the publication generation before retaining any local backend.

LISTEN and SELF share `ScopedRecognitionEngine` through the local-ASR runtime (`core/stt/scoped_engine.py`, `core/runtime/local_asr_provider_runtime.py`). `OwnedVadEvent` carries segment ownership across that boundary; only a scoped recognition terminal retires the source slot or admits a final transcript. SELF capture adds a generation-bound ledger, a nonblocking serialized recognition dispatcher, and one shared retained-audio budget resolved by frozen provider/settings scope; a missing resolution is an error, never a silent discard. Scope changes are merge barriers; channel callbacks project scoped readiness as `STREAMING` and failed terminals as `DISCONNECTED`, with user-visible failures on the typed `UserErrorReport` contract. LISTEN retention capacities use the seven-second maximum plus the existing prefix/frame and retained-slot accounting; SELF retention is unchanged.

`SmartTurnInferenceOwner` (`core/audio/smart_turn.py`) is the optional ONNX Runtime CPU endpoint probe: one setup attempt and one executing inference, with pause, segment, activation, the four-second step, OFF, or shutdown retirement revoking result authority.

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

SELF and LISTEN both allocate a provider epoch and open the backend with the explicit scoped projection in `core/stt/backend.py`. `STTSessionEventProjection` owns that scoped event stream: turn identity, update sequences, seal state, terminal receipt, and once-only epoch-end. Provisional or stable provider updates are not application-terminal transcripts.

`ScopedRecognitionEngine` (`core/stt/scoped_engine.py`) is the sole production recognition owner for both channels. Channels keep separate epochs, event buffers, request state, retention profiles, cancellation, and consumer policy; physical CPU/GPU resources stay shared through their existing owners. `STTProviderEventBuffer` bounds native ingress; `STTScopedTurnNormalizer` owns text assembly, deduplication, and complete text/language-run conservation, including terminal-only tails. Retention limits come from the binding as sample-equivalent and byte accounting.

The Soniox scoped adapter adds 200 ms of immediate PCM silence only when sealing a LISTEN turn at the fixed delivery boundary: a `delivery_pause` with at least 4 and less than 7 seconds of normalized content, or a `delivery_deadline` hard cut at the seven-second boundary. The synthetic provider input is sent atomically immediately before `finalize`; it has no source capture range and does not enter segment ownership or next-turn accumulation. Natural endpoints, SELF turns, and other providers retain their existing finalize behavior. This selector and the endpoint schedule implement the LISTEN-ENDPOINT-7S-HYST010 amendment.

Scoped abort invalidates the logical turn and provider-epoch authority first, detaches the session, and quarantines continuing native work as owned cleanup that is never reported as stopped. Configuration changes hand off a scoped replacement only after the new provider accepts the immutable settings scope; failure keeps the previous runtime signature, and the old owner keeps serving queued segments from its frozen configuration. Unkeyed provider completions retire the native epoch instead of being assigned to a newer segment.

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

Peer final parents enter `TranslationTurnLifecycleOwner` in source order with the existing deterministic language-run/target child identity. Waiting peer parents are bounded with expiry. Every child terminal path releases the existing semantic predecessor gate; request settings remain admission-time snapshots and scene context remains preparation-time context.

SELF speech parents have a separate bounded envelope. A dual-target speech turn occupies one parent slot while its child translations share that slot. Manual SELF turns use the same ordered lifecycle but are not cancelled, expired, or evicted by the speech envelope or TALK OFF; they may wait behind earlier admitted work.

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
