# Architecture

System map for PuriPuly Heart: runtime owners, data handoffs, dependency boundaries, and source entry points.

Keep this document focused on stable architectural decisions:

- Update the relevant section when ownership, a boundary, or a lifecycle contract changes; do not append a work log.
- State each invariant once. Keep policy values, retry schedules, UI layout, log field catalogs, and migration procedures in their owning code, tests, or focused guides.
- Link to representative implementations and behavior tests instead of reproducing their acceptance criteria.

Python paths are relative to `src/puripuly_heart/`; `src/`, `native/`, `tests/`, and `scripts/` paths are repository-relative.

## Architecture Model

- The Python application hosts the shared runtime for GUI and headless operation.
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
| UI application boundary | UI-facing application operations | `app/services/ui_application.py` |
| Application control owner | Typed operations, state queries, mutation ordering | `app/services/application_control.py`, `app/services/application_control_events.py` |
| Local control host | Authenticated endpoint, instance identity and lifetime | `cli/host.py`, `cli/transport.py`, `core/control_instance.py` |
| Settings owner          | Canonical settings, persistence, projection, rollback      | `app/services/canonical_settings_persistence.py` |
| Runtime pipeline        | Active runtime component set                               | `app/wiring/wiring_runtime_pipeline.py`                    |
| Self capture owner      | Microphone source and capture lifecycle                    | `core/runtime/self_capture.py`                       |
| Self translation owner  | Self STT events, turns, state, output projection           | `core/orchestrator/self_translation_channel.py`      |
| Peer capture owner      | Peer audio source, segmentation, and capture lifecycle | `core/runtime/peer_channel.py`                       |
| Local ASR runtime       | Local recognition channels and backend transitions         | `core/runtime/local_asr_provider_runtime.py`         |
| Managed local translation | Local model provisioning and runtime lifecycle | `app/services/managed_gemma_translation.py` and `core/local_translation/runtime.py` |
| Translation turn owner  | Request lifecycle, cancellation, stale-result rejection    | `core/orchestrator/translation_turn.py`                 |
| Output runtime          | Routing, delivery tasks, destinations, delivery history    | `core/runtime/output.py`                              |
| Overlay owners          | Overlay selection, process lifecycle, state, calibration   | `app/services/overlay/overlay_application.py`            |
| Managed-account runtime | Authentication, entitlement, usage, credential release     | `app/wiring/wiring_managed_account.py`                      |
| ChatGPT account owner   | OAuth session and credential lifecycle | `app/services/chatgpt_account.py`, `core/chatgpt/session.py` |
| OSC control runtime   | Receiver lifecycle, routing, state publication, restart    | `app/services/osc/control_runtime.py`                        |
| OSCQuery service      | Zeroconf discovery, receiver advertisement, OSCQuery tree | `core/osc/oscquery.py`                                        |
| Shutdown adapter        | Ordered application teardown                               | `app/adapters/application_runtime_shutdown.py`      |
| VRChat scene owner     | Process-lifetime instance population, immutable snapshots | `core/vrchat_scene_service.py`, `core/vrchat_scene.py` |

Ownership may span several processing stages. Do not assume one owner per pipeline stage.

## Data Handoffs

```text
Self:   microphone → capture / VAD → recognition → translation turns
        → output → UI / chatbox / overlays
Manual: text intent → manual translation turn → Self publication path
Peer:   loopback / process audio → capture / segmentation → recognition
        → ordered translation admission → output → UI / overlays
```

`SelfTranslationChannelOwner` coordinates Self speech. Manual text bypasses capture and STT. Peer output must not reach the VRChat chatbox.

### Audio ownership

- Capture owns source order, timing, and audio scope; audio loss is explicit, not silence.
- Segmentation is independent of provider readiness. Self and Peer share audio contracts but retain channel-specific endpoint policies.
- Capture authority determines which audio may enter recognition. Source completion drains admitted input; stop or discontinuity invalidates affected work.

Implementation: `core/audio/ownership.py`, `core/audio/listen_delivery.py`.

### Managed translation

Managed authentication → Broker entitlement or credential release → provider activation → normal translation request path.

### VRChat scene context

VRChat logs → trusted scene snapshot → sanitized translation request context.

The scene owner survives pipeline rebuilds and has no audio, VAD, or OSC dependency. Only trusted population context reaches the LLM; names and raw logs remain local. Custom HTTP extensions never receive scene data.

## Ports and Adapters

| Boundary | Contract | Adapters |
| --- | --- | --- |
| Application control | Typed commands, queries, operations, and events | Local CLI transport and `ApplicationControlOwner` |
| UI application | `UiApplicationPort` | `UiApplicationBoundary` |
| UI presentation | `UiPresentationPort`, `UIEventBridgePort` | Flet and headless presentation |
| Audio and STT | Capture, VAD, provider, and local ASR ports | Microphone, loopback, process capture, local and remote STT |
| Translation | `TranslationRequestPort`, `LLMProvider` | BYOK, managed, local, and remote providers |
| Output | Publication contracts, `OverlaySink`, overlay protocol | UI bridge, OSC, desktop and native VR overlays |
| GPU worker | `GpuWorkerClientPort`, `GpuWorkerProcessFactoryPort` | Native worker process |
| Secrets | `SecretStore`, `SettingsSecretsPort` | Secret storage and typed settings projection/mutation |
| Settings UI | Frozen surface snapshots and typed intents | Flet settings and `UiApplicationBoundary` |
| Shutdown | Runtime shutdown ports | Application shutdown adapter |
| OSC | Stable schema/codec, `OscControlApplicationPort`, `OscQueryServicePort` | OSC control and OSCQuery |

Adapters on one port are alternatives unless the owner supports fanout. Output explicitly supports simultaneous destinations.

Protocol guides: [CLI](cli.md), [VRChat OSC](vrchat-osc.md), [HTTP extensions](http-extensions.md).

## Composition

`composition/application_runtime.py` loads settings and secrets, constructs owners, selects adapters, and connects capture, providers, translation, output, overlays, OSC, and account services. It installs startup/shutdown and returns `UiApplicationBoundary`.

Composition may construct resources, but must transfer long-lived ownership to an explicit runtime owner.

## Local Application Control

GUI and headless hosts share application owners and runtime resources; presentation adapters determine whether there is a main window.

- `ApplicationControlOwner` exposes typed operations and owner-backed queries. CLI, GUI, and OSC mutations share application ordering and resource-conflict boundaries.
- Queries distinguish persisted intent from effective runtime state. Operations belong to the application, not the requesting client connection.
- The local control host owns authenticated same-user access and instance identity. Event subscriptions are bounded and privacy-filtered; slow clients cannot block runtime producers.

Entry points are listed in Runtime Ownership; command and protocol details belong in the [CLI guide](cli.md).

## Runtime Pipeline

`RuntimePipelineLauncher` builds and installs the active capture, STT, translation, output, UI-event, and VRChat microphone-state components.

Provider or settings changes may replace components. Do not retain references across replacement unless the API explicitly permits it.

## Configuration

| Layer | Responsibility | Entry points |
| --- | --- | --- |
| Persisted intent | Canonical settings and persistence; no live resources | `app/services/canonical_settings_persistence.py` |
| Resolved configuration | Effective selections and capability constraints | `config/resolved.py`, `config/runtime_resolution.py` |
| Runtime state | Active resources and work | Respective lifecycle owners |

`SettingsView` consumes frozen snapshots and emits focused typed intents. The settings owner applies them to the latest canonical settings before persistence and runtime application.

Settings persistence owns intent; runtime owners own its application. The provider-apply boundary coordinates capture and Local ASR owners. Failed or incomplete application must not appear as successfully applied runtime state.

Implementation: `app/services/provider/provider_runtime_apply.py`. Behavior: `tests/app/test_stt_provider_apply_vertical.py`.

## External Network Boundaries

`core/external_network.py` resolves external TLS and proxy policy at connection-owner boundaries. `core/network_clients.py` and library-specific adapters apply that policy; existing provider and runtime owners retain client reuse, cancellation, and closure.

- Default Windows cloud connections use native certificate-chain verification. Explicit CA overrides in the cloud policy do not silently gain additional default roots, and startup does not synthesize CA environment overrides.
- External HTTP and WebSocket connections share proxy and bypass selection while preserving intentional protocol-specific overrides. Owners capture policy once and apply it to each destination, including scheme-less proxy addresses and effective default-port bypass rules. Local/custom connections remain direct and retain their existing transport-specific TLS behavior; failures do not authorize direct fallback, disabled verification, or request replay.
- SDK realtime handshakes are configured separately from SDK HTTP clients. Narrow provider adapters contain version-coupled SDK seams without global SSL or SDK monkeypatches. GenAI HTTP clients are app-owned and explicitly closed by the existing LLM and Transcribe owners. GenAI Live retains SDK redirect handling for the same selected proxy route; redirects that change that route fail before a second connection rather than silently bypassing policy.
- Download workers carry the captured environment and system proxy/bypass policy through child startup and the existing native-to-HTTP fallback. Native Xet owns its Windows verification independently; policies it cannot honor must use the existing managed HTTP path rather than silently broaden trust or change routes. The managed HTTP factory preserves Hugging Face's redirect, timeout, and request-hook contract.

## Provider Boundaries

### STT

`ScopedRecognitionEngine` owns channel-scoped recognition. Self and Peer have separate recognition lifecycles while physical CPU/GPU resources remain shared through runtime owners.

- Provider adapters translate native events into shared STT contracts. Local audio completion and transcript delivery are distinct.
- Capture scope and provider lifetime authorize recognition results; provider recovery remains inside the recognition boundary.
- Local execution uses Python-process ASR or a native GPU worker. Python owns the worker connection and lifecycle; Rust owns device discovery, model activation, and transcription.

Implementation: `core/stt/scoped_engine.py`, `core/stt/session_projection.py`. Behavior: `tests/core/test_stt_scoped_engine.py`.

### Translation

- Provider adapters own authentication and provider-specific request, response, streaming, and error handling.
- `TranslationTurnLifecycleOwner` owns admission, cancellation, and publication. `TranslationRequestOwner` owns request preparation and provider authority.
- Self and Peer have separate channel admission while sharing provider capacity. Peer execution may be concurrent, but context preparation and publication preserve admitted order.
- Runtime owners manage provider replacement; managed local Gemma remains behind the same `LLMProvider` boundary.

Implementation: `core/orchestrator/translation_turn.py`, `core/orchestrator/translation_request.py`, `core/runtime/provider_rebuild.py`. Behavior: `tests/core/test_translation_turn_owner.py`, `tests/core/test_translation_request_owner.py`.

### ChatGPT connection

ChatGPT authentication is separate from Broker. The account owner manages OAuth and credential persistence; its session survives translation-provider replacement. Refresh credentials use the secret store, while access tokens remain in memory.

The provider owns connection capacity and in-flight exchanges shared by Self and Peer. Connection readiness is independent of audio capture; caller cancellation does not transfer responsibility for exchange cleanup.

Implementation: `app/services/chatgpt_account.py`, `providers/llm/chatgpt_plan.py`. Behavior: `tests/providers/test_chatgpt_plan_provider.py`.

## Output

`OutputRuntime` owns publication routing, destination delivery state, and cleanup.

| Publication | UI | Chatbox | Overlay |
| --- | --- | --- | --- |
| Self utterance | Yes | Yes | Yes |
| Peer subtitle | Yes | No | Yes |
| System disclosure | Policy-dependent | Explicit route only | Policy-dependent |

- Destinations have independent bounded delivery. A destination failure must not block other destinations or replay recognition and translation.
- Presentation does not gate translation execution. Output receipts distinguish application handoff from presentation, not physical display.
- Caption and overlay preferences control destinations, not capture.
- Runtime errors use the shared dashboard error path; interactive validation stays on its owning surface.

Implementation: `core/runtime/output.py`, `core/orchestrator/translation_output_projection.py`. Behavior: `tests/core/runtime/test_output_runtime.py`.

### Overlays

| Owner | Responsibility |
| --- | --- |
| Application (`app/services/overlay/`) | Target selection, recovery, and generation replacement |
| Python runtime (`core/overlay/`, `core/runtime/overlay.py`) | Caption lifetime, scene delivery, and process lifecycle |
| Native runtime (`native/overlay/src/runtime.rs`) | VR rendering and GPU resources |

`OverlayPresenter` owns caption admission and lifetime independently of providers. Python owns logical caption state; native rendering must not redefine that lifetime.

Behavior: `tests/core/test_overlay_presenter.py`, `native/overlay/tests/runtime.rs`.

## Runtime Logging

`SessionRuntimeLoggingService` owns console, file, and Logs-view delivery. Translation owners produce accepted conversation records; capture/provider/overlay owners produce lifecycle and failure evidence.

- File delivery is asynchronous and bounded; producers do not wait for file I/O.
- Technical diagnostics are metadata-only and exclude credentials, user text, and audio. Accepted conversation records use a separate secret-protected path.
- OpenRouter authentication and normalized translation/recognition transport failures retain available certificate-verification codes and safe policy labels. Raw exception text, endpoint details, and CA file paths are not diagnostic payloads. Other providers' existing boolean credential-verification contracts remain unchanged.
- Timing records describe application-observable stages, not physical display.

Implementation: `core/runtime_logging.py`, `app/services/application_runtime_logging.py`. Behavior: `tests/core/test_runtime_logging.py`.

## Runtime Layout

- `runtime_layout.py` separates host/interpreter paths, read-only resources, and writable user data. Features use this boundary rather than process flags or the working directory.
- Bootstrap selects runtime and resource paths before application startup. Packaging does not change feature ownership.
- Native distributions bundle Python, application code, dependencies, and resources as an immutable payload. Startup must not write into installed directories; code changes require rebuilding the payload.
- Import and resource-loading adapters preserve application contracts across source and packaged hosts. Native libraries and filesystem-backed resources remain accessible through their runtime layout.
- Installer cleanup owns only verified distribution files, never modified or unlisted files or writable user data.

Build policy and validation: `scripts/ci/build-native-experimental.ps1`, `release_evidence/native_distribution.py`. Runtime behavior: `tests/test_runtime_layout.py`, `tests/test_native_python_runtime.py`.

## Lifecycle

Every task, process, source, and provider session has an owner responsible for ingress stop, cancellation/draining, late-callback rejection, resource release, and restart.

### Stale-work protection and replacement

- Generations, attachment tokens, request IDs, and current-owner checks prevent retired work from mutating current state or publishing output.
- The owner decides whether admitted work drains or is cancelled. Gracefully admitted work retains its provider scope and frozen settings.
- Replacement revokes retired authority before it can affect the new runtime. Retired resources remain owned until cleanup completes.

### Shutdown

Stop ingress, drain or cancel owned work, close external resources, then clear runtime references.

The application shutdown adapter coordinates capture, translation, output, child processes, and services. Ordered teardown survives ordinary UI-task cancellation; child processes remain owned for the host lifetime.

Exact handoff and teardown ordering belongs to owner implementations and lifecycle tests. Application entry point: `app/adapters/application_runtime_shutdown.py`.

## Async Event Model

- The Python application runs asynchronous runtime work on its `asyncio` event loop.
- Owners create, track, and close their own background tasks.
- Do not create detached tasks without assigning lifecycle ownership.
- Capture, STT, translation, UI, and child-process events cross owner boundaries through ports, callbacks, or owned queues.
- Callbacks must delegate to the receiving owner; they must not mutate another owner's private runtime state.
- Ordering is local to the owning channel or queue. Do not assume global ordering across self, peer, UI, and provider events.
- Blocking model, device, or native work must not block the application event loop; use the established worker, executor, or child-process boundary.
