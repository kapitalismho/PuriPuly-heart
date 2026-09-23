# Architecture

Implementation-oriented system map for PuriPuly Heart.

Use this document to locate:

- runtime owners,
- data handoffs,
- ports and adapters,
- composition points,
- lifecycle boundaries,
- relevant source files.

For detailed behavior, runtime policy values, and migration rules, read the referenced code and tests.

Python source paths are relative to `src/puripuly_heart/`. Paths beginning with `src/`, `native/`, or `tests/` are relative to the repository root.

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
| Runtime pipeline        | Active runtime component set                               | `app/wiring/wiring_runtime_pipeline.py`                    |
| Self capture owner      | Microphone source and capture lifecycle                    | `core/runtime/self_capture.py`                       |
| Self translation owner  | Self STT events, turns, state, output projection           | `core/orchestrator/self_translation_channel.py`      |
| Peer capture owner      | Target, source, VAD, task, provider attachment, generation | `core/runtime/peer_channel.py`                       |
| Local ASR runtime       | Local recognition channels and backend transitions         | `core/runtime/local_asr_provider_runtime.py`         |
| Managed local translation | Gemma provisioning, readiness, backend, prefix, and process lifecycle | `app/services/managed_gemma_translation.py` and `core/local_translation/runtime.py` |
| Translation turn owner  | Request lifecycle, cancellation, stale-result rejection    | `core/orchestrator/translation_turn.py`                 |
| Output runtime          | Routing, delivery tasks, destinations, delivery history    | `core/runtime/output.py`                              |
| Overlay owners          | Overlay selection, process lifecycle, state, calibration   | `app/services/overlay/overlay_application.py`            |
| Managed-account runtime | Authentication, entitlement, usage, credential release     | `app/wiring/wiring_managed_account.py`                      |
| OSC control runtime   | Receiver lifecycle, routing, state publication, restart    | `app/services/osc/control_runtime.py`                        |
| OSCQuery service      | Zeroconf discovery, receiver advertisement, OSCQuery tree | `core/osc/oscquery.py`                                        |
| Shutdown adapter        | Ordered application teardown                               | `app/adapters/application_runtime_shutdown.py`      |
| VRChat scene owner     | Process-lifetime instance population, immutable snapshots | `core/vrchat_scene_service.py`, `core/vrchat_scene.py` |


Ownership may span several processing stages. Do not assume one owner per pipeline stage.

## Data Handoffs

### Self speech

```text
microphone
→ normalized audio frames
→ owned VAD events
→ scoped recognition events
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
→ peer capture and segmentation
→ scoped recognition events
→ source-ordered peer translation
→ publication intents
→ output runtime
→ UI / overlays
```

Peer output must not reach the VRChat chatbox.

### Audio ownership

- Capture preserves source order and timing. Audio loss is explicit, not silence.
- Capture owners retain generation-bound segment ledgers and freeze provider and endpoint settings for admitted segments (`core/audio/ownership.py`).
- `OwnedVadEvent` carries segment identity into recognition. Only scoped recognition terminals retire source slots or admit final transcripts.
- `ListenDeliveryController` owns peer segmentation independently of provider readiness (`core/audio/listen_delivery.py`).
- Self and peer share `VadGating` but retain separate onset and endpoint policies. Delivery rollover preserves acoustic continuity.
- `SmartTurnInferenceOwner` owns peer endpoint inference for supported languages and rejects retired results (`core/audio/smart_turn.py`).

Orderly capture completion drains recognition. Stop or discontinuity invalidates affected work.

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
- Only trusted population context crosses into translation requests. Names and raw logs remain local.
- Request preparation projects snapshots into sanitized LLM scene context.
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
- Persistence and migration: `config/settings_vnext/compat.py`
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

Runtime owners should consume resolved configuration (`config/resolved.py`, `config/runtime_resolution.py`).

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

Settings persistence owns user intent; runtime owners own its application to active resources. The provider-apply boundary coordinates capture and Local ASR owners. Failed or incomplete application must not be represented as successfully applied runtime state.

Implementation: `app/services/provider/provider_runtime_apply.py`. Behavior tests: `tests/app/test_stt_provider_apply_vertical.py`.

## Provider Boundaries

### STT

Execution options:

- Python-process local ASR,
- native GPU worker,
- remote provider.

`ScopedRecognitionEngine` owns recognition for both channels (`core/stt/scoped_engine.py`).

- Channels retain separate provider epochs, bounded buffers, cancellation, and retention policies.
- A provider epoch has at most one open audio-input turn. Sessions that explicitly permit sealed-turn overlap may retain bounded, identity-scoped sealed turns while the next turn captures; terminal normalization and publication remain in source order.
- Local CPU decode stays serialized by `LocalDecodeCoordinator`; if an admitted turn's decode fails, its already-queued successor remains serialized and is decoded without retrying the failed turn. The shared GPU runtime retains its bounded work admission and explicitly terminalizes queued work when fatal worker recovery fails. Remote sessions do not overlap turns unless their protocol adapter can unambiguously bind acknowledgements and results.
- Physical CPU/GPU resources remain shared through their runtime owners.
- `STTSessionEventProjection` keeps payload/update sequences and terminal authority per admitted identity (`core/stt/session_projection.py`). Explicit epoch retirement rejects late callbacks; a terminal that retires an overlap-capable epoch retains terminal-only authority for successors admitted before that failure.
- `STTScopedTurnNormalizer` assembles text, language runs, and session-scoped speaker runs per identity. Provider updates are not final application transcripts.

Provider replacement preserves frozen settings for every admitted turn and waits for the epoch boundary. Abort invalidates turn and epoch authority before native cleanup; graceful close drains or boundedly terminalizes admitted work.

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

The managed local Gemma adapter remains behind `LLMProvider`; its application/runtime owners handle model provisioning, backend readiness, and process lifecycle.

Cloud translation may use bounded hedged attempts according to resolved runtime policy, not persisted fallback selections (`config/runtime_resolution.py`, `core/llm/fallback_racing.py`).

Translation owners retain:

- turn lifecycle,
- cancellation,
- stale-result rejection,
- publication handoff.

`TranslationTurnLifecycleOwner` owns bounded Self and Peer admission and the lifecycle of parent turns and child translations. `TranslationRequestOwner` owns request preparation and provider-generation authority.

Peer translations may execute concurrently, but source-context preparation and publication preserve source order. Channel execution limits remain separate from provider-wide admission shared by Self and Peer.

Self speculative selection remains in the Self owner. Once a turn is admitted, the turn lifecycle owns subsequent translation and publication.

Implementation: `core/orchestrator/translation_turn.py`, `core/orchestrator/translation_request.py`. Behavior tests: `tests/core/test_translation_turn_owner.py`, `tests/core/test_translation_request_owner.py`, `tests/core/test_hedged_attempts.py`.

## Output

`OutputRuntime` owns:

- route selection and chatbox state,
- destination-scoped admission and delivery receipts,
- duplicate and retired-publication rejection,
- UI event bridge,
- destination replacement,
- shutdown cleanup.

Delivery boundaries:

- Peer UI and overlay destinations have independent bounded queues and writers.
- Self chatbox delivery owns its bounded admission and expiry policy.
- Output handoff releases translation ordering without waiting for display. Sink failure does not replay recognition or translation.
- Peer publications retain activation generation and source order through output. Retiring an activation cancels its deliveries and rejects late work.
- Destination admission and presenter application receipts are explicit; neither is a remote display acknowledgement.
- Unmanaged Self original-overlay publications and their source-only close events use an admission lane separate from managed Self translation-parent lifetime. Original B can apply and release its projection while A translation is pending; A translation and close still carry A's managed identity and cannot retire, reorder, or clear B.

Caption and overlay settings control destinations, not peer capture. Conversation errors share publication identity; runtime session status uses a separate path.


| Publication       | UI               | Chatbox             | Overlay          |
| ----------------- | ---------------- | ------------------- | ---------------- |
| Self utterance    | Yes              | Yes                 | Yes              |
| Peer subtitle     | Yes              | No                  | Yes              |
| System disclosure | Policy-dependent | Explicit route only | Policy-dependent |


Destination adapters must not bypass routing policy.

Each destination has independent admission and delivery state. Replacing one
destination must not block or retire work for the others.

Implementation: `core/runtime/output.py`. Behavior tests: `tests/core/runtime/test_output_runtime.py`.

### Overlays

| Owner | Responsibility |
| --- | --- |
| Application (`app/services/overlay/`) | Target selection, recovery, and generation replacement |
| Python runtime (`core/overlay/`, `core/runtime/overlay.py`) | Caption state and expiry, scene delivery, and process lifecycle |
| Native runtime (`native/overlay/src/runtime.rs`) | VR rendering, presentation retries, and GPU resources |

Each generation owns its tasks and shutdown. Python owns caption lifetime; native owns presentation retries.

`OverlayPresenter` owns provider-independent Peer subtitle admission and pacing (`core/overlay/presenter.py`); output retains bounded waiting work.

Behavior tests: `tests/core/test_overlay_presenter.py`.

## Runtime Logging

| Owner | Responsibility |
| --- | --- |
| `SessionRuntimeLoggingService` | Shared console, local file, and Logs view delivery |
| Translation owners | Accepted SELF/PEER source and target records |
| Overlay owners | Bounded failure evidence and reliable lifecycle warnings |

- `SessionRuntimeLoggingService` owns bounded asynchronous file delivery. Producers must not block on file I/O.
- Basic-audience records reach the console and Logs view. Selected technical diagnostics are file-only and metadata-only; accepted conversation uses a separate secret-protected path.
- Queue pressure prioritizes warning, error, and terminal evidence. Logging does not guarantee complete persistence.
- The writer retains ownership through stream closure; replacement must not race a retiring writer.

Implementation: `core/runtime_logging.py`, `app/services/application_runtime_logging.py`. Behavior tests: `tests/core/test_runtime_logging.py`, `tests/core/test_file_logging.py`.

## Runtime Layout

- `runtime_layout.py` separates host and interpreter paths, read-only resources, and writable user data.
- Features resolve runtime paths through this boundary, not process flags or the working directory.
- Bootstrap selects shared runtime, UI asset, and framework storage paths before application startup.
- Packaging does not change feature ownership or application logging policy.

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

### Replacement

- The owner controls ingress and decides whether admitted work drains or is cancelled.
- Admitted work retains its provider scope and frozen settings during a graceful handoff.
- Replacement must revoke retired work's authority before it can affect the current runtime.
- Retired resources remain owned until cleanup completes.

Exact handoff and cleanup ordering belongs to each owner's implementation and lifecycle tests.

### Shutdown direction

Stop ingress before draining or cancelling owned work. Close external resources before clearing runtime references.

The application shutdown adapter coordinates teardown across capture, translation, output, child processes, and application services.

Window-close orchestration must survive ordinary UI-task cancellation until ordered shutdown completes.

Implementation: `app/adapters/application_runtime_shutdown.py`. Use shutdown code and lifecycle tests for exact ordering.

Shutdown diagnostics expose bounded lifecycle metadata, not user content or credentials.

Child processes remain owned for the host lifetime. Abrupt-exit containment is a fallback, not a substitute for graceful shutdown.

## Async Event Model

- The Python application runs asynchronous runtime work on its `asyncio` event loop.
- Owners create, track, and close their own background tasks.
- Do not create detached tasks without assigning lifecycle ownership.
- Capture, STT, translation, UI, and child-process events cross owner boundaries through ports, callbacks, or owned queues.
- Callbacks must delegate to the receiving owner; they must not mutate another owner's private runtime state.
- Ordering is local to the owning channel or queue. Do not assume global ordering across self, peer, UI, and provider events.
- Blocking model, device, or native work must not block the application event loop; use the established worker, executor, or child-process boundary.
