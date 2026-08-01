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
| VRChat OSC           | Self chatbox output                                               |
| Overlay processes    | Desktop and VR subtitle presentation                              |
| GPU worker           | Native local GPU ASR                                              |
| Broker               | Managed identity, entitlement, credentials, telemetry             |
| Local storage        | Settings, secrets, diagnostics, model assets                      |


Broker is a control-plane dependency, not part of the normal utterance data path.

## Runtime Ownership


| Owner                   | Owns                                                       | Key path                                                                  |
| ----------------------- | ---------------------------------------------------------- | ------------------------------------------------------------------------- |
| UI application boundary | UI-facing application operations                           | `app/services/ui_[application.py](http://application.py)`                 |
| Settings owner          | Canonical settings, persistence, projection, rollback      | `app/services/canonical_settings_[persistence.py](http://persistence.py)` |
| Runtime pipeline        | Active runtime component set                               | `app/wiring_runtime_[pipeline.py](http://pipeline.py)`                    |
| Self capture owner      | Microphone source and capture lifecycle                    | `core/runtime/self_[capture.py](http://capture.py)`                       |
| Self translation owner  | Self STT events, turns, state, output projection           | `core/orchestrator/self_translation_[channel.py](http://channel.py)`      |
| Peer capture owner      | Target, source, VAD, task, provider attachment, generation | `core/runtime/peer_[channel.py](http://channel.py)`                       |
| Local ASR runtime       | Local recognition channels and backend transitions         | `core/runtime/local_asr_provider_[runtime.py](http://runtime.py)`         |
| Translation turn owner  | Request lifecycle, cancellation, stale-result rejection    | `core/orchestrator/translation_[turn.py](http://turn.py)`                 |
| Output runtime          | Routing, delivery tasks, destinations, delivery history    | `core/runtime/[output.py](http://output.py)`                              |
| Overlay owners          | Overlay selection, process lifecycle, state, calibration   | `app/services/overlay_[application.py](http://application.py)`            |
| Managed-account runtime | Authentication, entitlement, usage, credential release     | `app/wiring_managed_[account.py](http://account.py)`                      |
| Shutdown adapter        | Ordered application teardown                               | `app/adapters/application_runtime_[shutdown.py](http://shutdown.py)`      |


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
→ peer capture and VAD
→ peer STT events
→ peer translation
→ publication intents
→ output runtime
→ UI / overlays

```

Peer output must not reach the VRChat chatbox.

### Managed translation

```text
managed authentication
→ Broker entitlement or credential release
→ provider runtime activation
→ normal translation request path

```

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
| Shutdown        | Runtime shutdown ports                               | Application shutdown adapter             |


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
- Compatibility projection: `AppSettings`
- Owner: canonical settings persistence service

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

## Provider Boundaries

### STT

Execution options:

- Python-process local ASR,
- native GPU worker,
- remote provider.

Channel owners consume normalized STT events:

- session state,
- partial transcript,
- final transcript,
- failure.

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

Translation owners retain:

- turn lifecycle,
- cancellation,
- stale-result rejection,
- publication handoff.

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

Key paths:


| Concern                                | Path                                                                                    |
| -------------------------------------- | --------------------------------------------------------------------------------------- |
| Lifecycle task conventions             | `src/puripuly_heart/core/[lifecycle.py](http://lifecycle.py)`                           |
| Self event handling                    | `src/puripuly_heart/core/orchestrator/self_translation_[channel.py](http://channel.py)` |
| Peer capture and generation guards     | `src/puripuly_heart/core/runtime/peer_[channel.py](http://channel.py)`                  |
| Translation requests and stale results | `src/puripuly_heart/core/orchestrator/translation_[request.py](http://request.py)`      |
| Output delivery tasks and UI bridge    | `src/puripuly_heart/core/runtime/[output.py](http://output.py)`                         |
| GPU worker async process protocol      | `src/puripuly_heart/app/adapters/gpu_worker_[process.py](http://process.py)`            |
| Application shutdown                   | `src/puripuly_heart/app/adapters/application_runtime_[shutdown.py](http://shutdown.py)` |


## Code Map

### Entry and UI


| Path                                                                         | Responsibility          |
| ---------------------------------------------------------------------------- | ----------------------- |
| `src/puripuly_heart/[main.py](http://main.py)`                               | Process entry           |
| `src/puripuly_heart/ui/[app.py](http://app.py)`                              | Flet presentation root  |
| `src/puripuly_heart/app/ports/ui_[application.py](http://application.py)`    | UI application contract |
| `src/puripuly_heart/app/services/ui_[application.py](http://application.py)` | Application boundary    |


### Composition


| Path                                                                           | Responsibility              |
| ------------------------------------------------------------------------------ | --------------------------- |
| `src/puripuly_heart/composition/application_[runtime.py](http://runtime.py)`   | Main composition root       |
| `src/puripuly_heart/composition/application_[settings.py](http://settings.py)` | Settings startup            |
| `src/puripuly_heart/composition/application_[startup.py](http://startup.py)`   | Startup ordering            |
| `src/puripuly_heart/app/wiring_provider_[runtime.py](http://runtime.py)`       | Provider composition        |
| `src/puripuly_heart/app/wiring_peer_[application.py](http://application.py)`   | Peer composition            |
| `src/puripuly_heart/app/wiring_managed_[account.py](http://account.py)`        | Managed-account composition |


### Runtime


| Path                                                                                    | Responsibility       |
| --------------------------------------------------------------------------------------- | -------------------- |
| `src/puripuly_heart/core/runtime/self_[capture.py](http://capture.py)`                  | Self capture         |
| `src/puripuly_heart/core/runtime/peer_[channel.py](http://channel.py)`                  | Peer capture         |
| `src/puripuly_heart/core/orchestrator/self_translation_[channel.py](http://channel.py)` | Self translation     |
| `src/puripuly_heart/core/orchestrator/translation_[turn.py](http://turn.py)`            | Turn lifecycle       |
| `src/puripuly_heart/core/orchestrator/translation_[request.py](http://request.py)`      | Translation boundary |
| `src/puripuly_heart/core/runtime/local_asr_provider_[runtime.py](http://runtime.py)`    | Local ASR runtime    |
| `src/puripuly_heart/core/runtime/[output.py](http://output.py)`                         | Output runtime       |


### Infrastructure


| Path                                                                         | Responsibility              |
| ---------------------------------------------------------------------------- | --------------------------- |
| `src/puripuly_heart/core/storage/[secrets.py](http://secrets.py)`            | Secret stores               |
| `src/puripuly_heart/core/overlay/[protocol.py](http://protocol.py)`          | Overlay contract            |
| `src/puripuly_heart/core/overlay/[process.py](http://process.py)`            | Overlay process supervision |
| `src/puripuly_heart/app/adapters/gpu_worker_[process.py](http://process.py)` | GPU worker adapter          |
| `src/puripuly_heart/core/gpu_[worker.py](http://worker.py)`                  | GPU worker contracts        |
| `native/gpu_worker/src/[lib.rs](http://lib.rs)`                              | Native GPU runtime          |
| `native/overlay/src/[lib.rs](http://lib.rs)`                                 | Native VR overlay           |
| `broker/src/app.ts`                                                          | Broker routes               |


## Architecture Evidence


| Path                                                                       | Enforces                     |
| -------------------------------------------------------------------------- | ---------------------------- |
| `tests/architecture/test_dependency_[boundaries.py](http://boundaries.py)` | Dependency direction         |
| `tests/architecture/test_lifecycle_task_[guard.py](http://guard.py)`       | Task ownership               |
| `tests/config/compatibility_surface_inventory.json`                        | Compatibility surfaces       |
| `tests/app/test_gpu_worker_[process.py](http://process.py)`                | GPU process contract         |
| `tests/core/runtime/test_output_[runtime.py](http://runtime.py)`           | Output ownership and routing |
| `tests/core/test_peer_channel_[routing.py](http://routing.py)`             | Peer routing                 |


