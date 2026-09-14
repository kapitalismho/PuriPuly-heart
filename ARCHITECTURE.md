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
- `SmartTurnInferenceOwner` owns optional endpoint inference and rejects retired results (`core/audio/smart_turn.py`).

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
- `STTScopedTurnNormalizer` assembles text and language runs. Provider updates are not final application transcripts.

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

Translation owners retain:

- turn lifecycle,
- cancellation,
- stale-result rejection,
- publication handoff.

`TranslationTurnLifecycleOwner` admits peer turns in source order. Self and peer speech have separate bounded queues with expiry; child translations share their parent slot.

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
- Destination acceptance is not a remote display acknowledgement.

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

Application recovery coordinates generation replacement. Each generation owns its
tasks and shutdown.

## Runtime Logging

`SessionRuntimeLoggingService` is the single composition owner for application
runtime logs. It attaches the shared console and bounded queued-file sinks,
applies sink redaction, controls Basic/Detailed visibility, and forwards the
same accepted records to the Logs view. Views and feature owners must not attach
parallel root handlers.

| Event family | Owner and diagnostic question | Mode | Sink | Correlation and retention | Disposition |
| --- | --- | --- | --- | --- | --- |
| Session, effective settings, providers, and backends | Application startup, `SettingsApplicationOwner`, and capture/provider state adapters: what configuration became effective or remained previous? | Basic state/change/failed-or-degraded receipt; Detailed adds resolved safe context | Main file and live Logs view | Session, requested/effective state, provider, generation; change-only | Keep/enrich |
| Capture progress | `run_audio_vad_loop`: are there no frames, frames without admitted speech, or resumed/admitted speech? Capture state adapters: is a provider pending, ready, or failed? | Basic transition/10-second audio-window summary; Detailed keeps bounded audio metrics | Main file and live Logs view | Channel and capture/provider generation; transition state only | Add/aggregate |
| Recognition terminals | SELF/PEER channel owners: did the scoped segment finish, produce no text, expire, cancel, or fail, and why? | Basic semantic terminal; Detailed adds provider epoch/turn and named timing | Main file and live Logs view | Segment/utterance, provider epoch/turn, activation generation; terminal once | Keep/enrich |
| Translation terminals and latency | Translation lifecycle/diagnostics owners: was each target translated, skipped, expired, cancelled, superseded, or failed? | Basic target result and end-to-end latency; Detailed adds stage durations, target generation/order, and one target-specific context summary | Main file and live Logs view | Parent/child utterance, target index/language, turn generation/order; bounded timelines | Reduce/fold |
| Destination results | `OutputRuntime`: what did UI, chatbox, or overlay independently accept, skip, deny, replace, coalesce, expire, or fail? | Basic destination result; asynchronous completion remains a separate correlated result | Main file and live Logs view | Publication ID/kind and route; bounded runtime decision history | Add/connect |
| Accepted conversation content | Translation output projection and `SessionRuntimeLoggingService`: which accepted source/target text belongs to the turn? | Same designated content records in Basic and Detailed | Main file and live Logs conversation surface | Source once by semantic parent; translation once by parent/target; bounded service/UI dedupe and field size | Keep/add |
| Provider request/response bodies and third-party HTTP lines | Provider boundary: no diagnostic question requires copied bodies or library request chatter | Neither mode | Dropped before main file/UI | No retention | Remove |
| Managed service, Gemma, and ASR GPU recovery | Managed/runtime owners: what readiness, backend, failure, fallback, and final recovery state occurred? | Basic meaningful transition/result; Detailed adds bounded attempt/progress context | Main file and live Logs view | Operation/provider/channel and effective generation; change-only | Keep/enrich |
| Shutdown and logging delivery | Application shutdown coordinator and runtime logging sink: which owner first failed, what additional cleanup failed, and was queued delivery lost? | Basic callback, first-cause terminal, cleanup, and loss receipts | Main file plus console fallback where file delivery is unavailable | Ordered owner/callback/phase causes; terminal delivery counters | Add/connect |
| Desktop/native overlay | Overlay application/process/diagnostic recorder and native runtime: where did startup, presentation, recovery, cleanup, or export stop? | Basic correlated lifecycle/presentation result; Detailed bounded episode/artifact evidence | Main file/live Logs plus bounded failure artifacts | Target, instance, generation, revision, episode, source/parent sequences; capped history/artifacts | Reduce/enrich |

Overlay logging uses revisioned requested/effective/pending-or-failed mode
receipts across Python, desktop, and native owners. Basic carries generation-
and-target-correlated start, ready, meaningful presentation changes,
first-visible, failure, cleanup, and artifact receipts. Detailed retains bounded
stage history across disable and writes capped JSONL: 4 KiB per line, 1 MiB per
file, a 1-second write deadline, and at most 8 files or 8 MiB. Artifact receipts
separate file-write success, retained-capture completeness, and unknown native
terminal delivery. Physical HMD visibility is not observable by the runtime.

Conversation content is an explicit local diagnostic category. Records carry
channel, utterance identity, turn kind, language, target index, and disposition.
The service deduplicates source records by semantic turn and translations by
target. Secret-shaped substrings are redacted and individual text fields are
bounded before reaching console, file, or UI sinks. Basic and Detailed use the
same content-safety rules.

File delivery is asynchronous through a bounded queue. Producers never wait on
ordinary file I/O; saturation is counted, terminal records can evict an older
queued record, handler exceptions do not stop the listener, and shutdown uses
a bounded drain. Any known delivery loss is appended to a later terminal
receipt.

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
