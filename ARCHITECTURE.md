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
- Capture owners retain generation-bound segment ledgers (`core/audio/ownership.py`). Segments freeze provider and endpoint settings and follow `open → sealed → terminal`.
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

When a settings draft exits, the typed intent is persisted and then passed through the provider-apply boundary. For an active Self capture, provider application must converge both the Self capture owner and the Local ASR channel to the requested live runtime signature before the applied signature cache is updated. Idle or disabled Self capture never forces preparation for an unrelated apply, but an explicit STT selection may still prepare the dormant provider without committing a live handoff. A smooth active handoff keeps the current provider and frozen endpoint settings until the owning translation channel completes the utterance at `SpeechEnd`; failed, cancelled, or non-converged application leaves the previous cache truth intact.

## Provider Boundaries

### STT

Execution options:

- Python-process local ASR,
- native GPU worker,
- remote provider.

`ScopedRecognitionEngine` owns recognition for both channels (`core/stt/scoped_engine.py`).

- Channels retain separate provider epochs, bounded buffers, cancellation, and retention policies.
- Physical CPU/GPU resources remain shared through their runtime owners.
- `STTSessionEventProjection` defines scoped turn updates and terminal receipts (`core/stt/backend.py`).
- `STTScopedTurnNormalizer` assembles text, language runs, and session-scoped speaker runs. Provider updates are not final application transcripts.

Provider replacement preserves frozen settings for admitted work. Abort invalidates turn and epoch authority before native cleanup.

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

Cloud translation hedging is resolved runtime policy, not persisted user intent. Eligible primaries receive a second attempt with the same model, connection, credential, and routing after 1300 ms or a primary error. OpenRouter primaries retain the additional Gemma 31B ModelRun emergency attempt after 4400 ms. Custom HTTP, managed local Gemma, and local LLM primaries remain single-attempt. Settings schema 45 discards retired fallback selections; neither the settings UI nor OSC exposes a fallback selector.

Translation owners retain:

- turn lifecycle,
- cancellation,
- stale-result rejection,
- publication handoff.

`TranslationTurnLifecycleOwner` admits peer turns in source order. Self and peer speech have separate bounded queues with expiry; child translations share their parent slot.

The turn owner segments LISTEN transcripts by language and speaker; LLM translation batches each transcript while preserving segment identity.

Manual self turns share the ordered lifecycle but are not subject to speech eviction, expiry, or TALK OFF cancellation.

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
- Self chatbox speech has bounded pending delivery and expiry. Manual messages are exempt from speech eviction and expiry.
- Output handoff releases translation ordering without waiting for display. Sink failure does not replay recognition or translation.
- Peer publications retain activation generation and source order through output. Retiring an activation cancels its deliveries and rejects late work.
- Destination admission and presenter application receipts are explicit; neither is a remote display acknowledgement.

Caption and overlay settings control destinations, not peer capture. Explicit LISTEN OFF aborts capture and publication. Conversation errors share publication identity; runtime session status uses a separate path.


| Publication       | UI               | Chatbox             | Overlay          |
| ----------------- | ---------------- | ------------------- | ---------------- |
| Self utterance    | Yes              | Yes                 | Yes              |
| Peer subtitle     | Yes              | No                  | Yes              |
| System disclosure | Policy-dependent | Explicit route only | Policy-dependent |


Destination adapters must not bypass routing policy.

Each destination has independent admission and delivery state. Replacing one
destination must not block or retire work for the others.

### Overlays

| Responsibility | Key path |
| --- | --- |
| Overlay selection and recovery | `app/services/overlay/` |
| Caption state, scene delivery, and process lifecycle | `core/overlay/` |
| Generation tasks and shutdown | `core/runtime/overlay.py` |
| Native VR presentation | `native/overlay/src/runtime.rs` |

Python paths are relative to `src/puripuly_heart/`.

Overlay split:

- Python: overlay selection, caption state and expiry, scene delivery, process lifecycle.
- Native: VR rendering, render retries, and GPU resources.

`OverlayPresenter` owns provider-independent LISTEN caption admission and pacing;
the output writer retains waiting work within its existing bounded batches.

Application recovery coordinates generation replacement. Each generation owns its
tasks and shutdown.

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
