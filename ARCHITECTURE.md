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
→ VAD events
→ self STT events
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

Capture iteration and acoustic segmentation begin after target resolution without waiting for provider setup or startup. Provider readiness is a distinct gate in the bounded serialized dispatch queue: it admits eight wholly unsent sealed segments in addition to the downstream-active and current capture segments, exact prefix or synthetic context from each ownership snapshot, and 32 reserved terminal/control events. Retained PCM is bounded by those segment slots, each segment's six-second source range, and configured prefix; there is no shorter aggregate-content cap inside that count and freshness envelope. The LISTEN `OFF` delivery controller owns acoustic endpoints independently of provider progress: after four seconds of source content it seals at the first 224 ms persisted non-speech range, a six-second source-content deadline seals even when no later callback arrives, and continued speech rolls directly into the next owned segment without a second onset or repeated prefix. Queue pressure above eight eligible segments retires the oldest as `overload`; each wholly unsent sealed segment has its own twelve-second seal-age TTL and then retires as `expired_before_recognition`. Provider latency therefore does not suspend callback-range progression, acoustic segmentation, or source-time delivery deadlines.

A naturally finite capture dispatches every accepted owned event, becomes honestly inactive, and drains the provider instead of aborting it. Stable segment publication remains available during that drain; a scoped authoritative final or empty receipt wins. If the legitimate provider drain returns without a scoped terminal, the unresolved sealed segment retires as failed with `provider_drain_without_scoped_terminal`, never as successful empty. User stop, source failure, and provider failure retain their abort/cancel semantics. An `OFF` transition may keep an eligible local backend loaded only after invalidating its publication generation and aborting peer ingress, so retained resources cannot publish late peer output.

LISTEN uses `ScopedRecognitionEngine` through the existing local-ASR runtime and provider handle. `OwnedVadEvent` retains segment ownership through this boundary; only a scoped recognition terminal can retire the matching source slot or admit a final peer transcript. Concrete sessions implement `STTScopedTurnSession` and preserve native request/item/task provenance separately from application identity. SELF alone retains `ManagedSTTProvider` and its legacy event correlation until the separately scoped SELF migration; each concrete route maintains one native parser with client-specific event projections.

`ProspectiveSpeakerTransitionReceiver` accepts injected source-scoped hypotheses through the peer capture owner and the existing delivery controller. Its receipt distinguishes a prospective seal at the current accepted frontier from an already-separated, too-late, invalid, duplicate, or retracted hypothesis. This does not activate a speaker producer, reset its reference, or partition previously recognized text.

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

SELF consumes the existing session, partial, final, and failure event projection. LISTEN consumes the scoped update/terminal exchange in `core/stt/backend.py`; provisional or stable provider updates are not application-terminal transcripts.

`ScopedRecognitionEngine` owns ordered begin/payload/seal execution, route-resolved watchdogs, provider epochs, and bounded late-resource cleanup. `STTProviderEventBuffer` bounds native event ingress; `STTScopedTurnNormalizer` owns text assembly, native-event deduplication, and complete text/language-run conservation. Protocol adapters own their native completion barriers and actual writer progress. The existing CPU/GPU runtime retains physical model, device, and process ownership.

Provider configuration handoff retains an old scoped owner until source-ordered queued segments using that configuration are drained. The factory resolves all existing selectors and aliases without a legacy LISTEN fallback. Custom realtime peer configuration rejects an explicit incompatible `turn_detection` before activation; temporary SELF configuration projection remains separate.

The configured Deepgram, Gemini Transcribe Live, Soniox, and Scribe message shapes do not provide a reliable per-turn native identifier. Their scoped LISTEN completion barriers therefore retire the native epoch before another turn, rather than trusting fixture-only IDs or assigning a delayed unkeyed result to a newer segment. Custom realtime retains keyed reuse through the native committed item ID; an unkeyed completion retires its epoch. Qwen ASR requires the documented native item ID. Provider/model configuration and physical resource ownership survive where supported; epoch retirement is not a change to the capture clock or segment identity.

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

Peer final parents enter `TranslationTurnLifecycleOwner` in source order with the existing deterministic language-run/target child identity. The owner bounds waiting peer parents at eight and expires waiting work twelve seconds after admission. Every child terminal path releases the existing semantic predecessor gate; request settings remain admission-time snapshots and scene context remains preparation-time context.

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
