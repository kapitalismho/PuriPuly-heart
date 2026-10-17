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

`tests/architecture/test_dependency_boundaries.py` enforces this boundary across all core modules, independently of the generic layer taxonomy, including nested and type-only imports. The exact existing `core.discord.oauth_callback_page` → `ui.i18n` edge is transitional localization debt; no other core-to-UI/composition edge is allowed. Dynamic imports and module names in process-launch strings are outside this import rule.

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

- `SelfTranslationChannelOwner` coordinates Self speech.
- Manual text bypasses capture and STT.
- Peer output must not reach the VRChat chatbox.

### Audio ownership

- Capture owns source order, timing, and audio scope. Audio loss is explicit; it is not silence.
- Segmentation does not depend on provider readiness.
- Self and Peer share audio contracts. Each channel retains its own endpoint policies.
- Capture authority determines which audio may enter recognition.
- Source completion drains admitted input. Stop or discontinuity invalidates affected work.

Implementation: `core/audio/ownership.py`, `core/audio/listen_delivery.py`.

### Managed translation

Managed authentication → Broker entitlement or credential release → provider activation → normal translation request path.

### VRChat scene context

VRChat logs → trusted scene snapshot → sanitized translation request context.

- The scene owner survives pipeline rebuilds. It has no audio, VAD, or OSC dependency.
- Only trusted population context reaches the LLM. Names and raw logs remain local.
- Custom HTTP extensions never receive scene data.

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

`composition/application_runtime.py`:

- Loads settings and secrets.
- Constructs owners and selects adapters.
- Connects capture, providers, translation, output, overlays, OSC, and account services.
- Installs startup and shutdown.
- Returns `UiApplicationBoundary`.

Composition may construct resources, but must transfer long-lived ownership to an explicit runtime owner.

## Local Application Control

GUI and headless hosts share application owners and runtime resources. Presentation adapters determine whether there is a main window.

- **Operations and queries**
  - `ApplicationControlOwner` exposes typed operations and owner-backed queries.
  - CLI, GUI, and OSC mutations share application ordering and resource-conflict boundaries.
  - Queries distinguish persisted intent from effective runtime state.
  - `translation_runtime_ready` reports provider attachment, not successful native-model preparation. Managed readiness and failures belong to `models.status`.
  - Translation enablement publishes preparation demand before warmup. A current-generation warmup failure or cancellation rolls effective enablement off without discarding the selected provider or its actionable failure. Later explicit enablement retries; superseded work cannot overwrite newer intent.
  - Operations belong to the application, not the requesting client connection.
- **Local control host**
  - The host owns authenticated same-user access and instance identity.
  - Event subscriptions are bounded and privacy-filtered. Slow clients cannot block runtime producers.
  - Readiness waits for endpoint publication. Blocking publication runs through the owned-thread boundary; ingress freeze fences startup, and endpoint cleanup waits for publication before releasing the lease.
  - Windows discovery readers permit deletion of the advisory record while finishing their existing snapshots. The separate ownership lock and ACL/authentication checks remain authoritative; ordinary publication and cleanup failures still propagate.
- **GUI settings queues**
  - Queues obtain the current diagnostic instance identity through the data-only `UiApplicationPort.diagnostic_instance_id` property.
  - `UiApplicationBoundary` reads the composed control owner's identity. The GUI does not access operational control directly.
  - Queues carry their existing operation/request identity while executing application intents.
  - Presentation and notifications are not authoritative.
- **Retained diagnostics**
  - Queries expose current-instance metadata and delivery health, not raw log files or a second state authority.
  - Receipts and owner snapshots determine effects.
  - Diagnostic references explain failure, degradation, recovery, and incomplete cleanup.

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

Settings changes follow this path:

1. `SettingsView` consumes frozen snapshots and emits focused typed intents.
2. The settings owner applies those intents to the latest canonical settings.
3. The result proceeds to persistence and runtime application.

- Settings persistence owns intent. Runtime owners own its application.
- The provider-apply boundary coordinates capture and Local ASR owners.
- Failed or incomplete application must not appear as successfully applied runtime state.
- LLM applied-profile signatures and retry state follow actual provider attachment or replacement. Initial attachment uses the exact launch snapshot; an absent provider remains retryable. Replacement records its outcome before usage refresh or STT convergence. Generic settings, restoration and post-transaction cache updates are STT-only and cannot assert LLM application. Provider application advances STT signatures only after STT convergence, so an independent STT failure can leave the overall transaction committed but runtime-degraded without losing the actual LLM binding.

Implementation: `app/services/provider/provider_runtime_apply.py`. Behavior: `tests/app/test_stt_provider_apply_vertical.py`.

## Provider Boundaries

### STT

`ScopedRecognitionEngine` owns channel-scoped recognition. Self and Peer have separate recognition lifecycles. Runtime owners share physical CPU/GPU resources across them.

- **Provider contracts**
  - Provider adapters translate native events into shared STT contracts.
  - Local audio completion and transcript delivery are distinct.
  - Capture scope and provider lifetime authorize recognition results.
  - Provider recovery stays inside the recognition boundary.
  - Direct Qwen Audio overlaps locally bounded tasks using at most two connections and bounded unsent audio per Self/Peer channel, preserving source order and each channel's endpoint and publication policies.
- **Failure identity**
  - Typed turn/input/epoch results and provider-mutation results retain the original safe cause before reduction to a terminal reason.
  - Capture generation, provider generation/epoch, and turn identity remain distinct.
  - Cleanup and exhaustion link to the first cause rather than replacing it.
- **Warmup and readiness**
  - Failed warmup of an attached lazy-retryable engine does not itself stop usable Self capture or establish GPU readiness.
  - The current-generation attached phase, effective capture, and GPU-owner readiness are separate observations.
  - Existing retry/fallback policy is unchanged.
- **Local execution**
  - ASR runs in the Python process or a native GPU worker.
  - Python owns the worker connection and lifecycle.
  - Rust owns device discovery, model activation, and transcription.

Implementation: `core/stt/scoped_engine.py`, `core/stt/session_projection.py`. Behavior: `tests/core/test_stt_scoped_engine.py`.

### Translation

- **Provider and request ownership**
  - Provider adapters own authentication and provider-specific request, response, streaming, and error handling.
  - `TranslationTurnLifecycleOwner` owns admission, cancellation, and publication.
  - `TranslationRequestOwner` owns request preparation and provider authority.
  - Runtime owners manage provider replacement.
  - Managed local Gemma stays behind the same `LLMProvider` boundary.
- **Channel ordering**
  - Self and Peer have separate channel admission while sharing provider capacity.
  - Peer execution may be concurrent. Context preparation and publication preserve admitted order.
- **Manual admission**
  - Admission returns existing parent/child turn identities.
  - The originating operation/request identity is carried explicitly through asynchronous work.
  - Admission, bounded typing-completion management, translation success, and destination delivery are separate facts.
- **Diagnostic classification**
  - Classification uses typed status, transport, and documented protocol cause, not arbitrary exception prose.
  - Existing rolling retry/exclusion heuristics remain operational policy.
  - A heuristic decision is not evidence of a known diagnostic cause.

Implementation: `core/orchestrator/translation_turn.py`, `core/orchestrator/translation_request.py`, `core/runtime/provider_rebuild.py`. Behavior: `tests/core/test_translation_turn_owner.py`, `tests/core/test_translation_request_owner.py`.

### Managed local translation

- **Model and provisioning identity**
  - E4B and 26B A4B QAT are distinct managed selections behind the same `LLMProvider` boundary.
  - Each model owns its pinned assets, installation directory and verified manifest. Provisioning is on demand and atomic; selecting E4B does not provision 26B.
  - The runtime owner validates the executable's pinned build/commit before launch, including supported runtime-root overrides.
- **Execution ownership**
  - One managed runtime owner owns one server and one loaded target model. Vulkan admits at most two active translations; effective CPU execution admits one.
  - Three resident language-pair prefix slots are cache capacity, not authorization for three active translations. Same-pair concurrent requests reserve different mutable slots.
  - The provider factory delegates managed capacity to this owner rather than applying the unrelated remote-provider concurrency preference. Channel owners retain turn ordering, stale-result rejection and destination eligibility.
- **Slots and lifecycle**
  - A request retains its slot through native completion or cancellation settlement. Prefix preparation, restore, save and LRU eviction cannot overwrite a busy slot.
  - Model, device and backend transitions drain active work before replacing the server and reject results from superseded state. A failed request cannot terminate another valid request merely to reclaim its slot.
  - Prefix compatibility includes model/artifact, runtime/executable, effective backend/profile, prompt/template and language pair.
- **Backend policy**
  - E4B preserves serialized CPU MTP and its internal GPU-to-CPU fallback; fallback uses CPU admission and the CPU profile.
  - 26B is Vulkan-only with the fixed owner-selected placement/cache profile. Initialization failure is actionable, not permission to substitute CPU, E4B or a remote provider.

Implementation: `core/local_translation/runtime.py`, `runtime_profile.py`, `assets.py`, `prefix_cache.py`, `providers/llm/managed_gemma.py`, and `app/wiring/wiring_managed_gemma.py`. Behavior: `tests/core/local_translation/test_gemma_runtime.py`, `tests/providers/test_managed_gemma_concurrency.py`. Fixed profiles and selection controls: [CLI guide](cli.md#managed-local-translation).


### ChatGPT connection

- ChatGPT authentication is separate from Broker.
- The account owner manages OAuth and credential persistence. Its session survives translation-provider replacement.
- Refresh credentials use the secret store. Access tokens remain in memory.
- The provider owns connection capacity and in-flight exchanges shared by Self and Peer.
- Connection readiness is independent of audio capture.
- Caller cancellation does not transfer responsibility for exchange cleanup.

Implementation: `app/services/chatgpt_account.py`, `providers/llm/chatgpt_plan.py`. Behavior: `tests/providers/test_chatgpt_plan_provider.py`.

## Output

`OutputRuntime` owns publication routing, destination delivery state, and cleanup.

| Publication | UI | Chatbox | Overlay |
| --- | --- | --- | --- |
| Self utterance | Yes | Yes | Yes |
| Peer subtitle | Yes | No | Yes |
| System disclosure | Policy-dependent | Explicit route only | Policy-dependent |

- **Destination delivery**
  - Each destination has independent bounded delivery.
  - A destination failure must not block other destinations or replay recognition and translation.
  - Presentation does not gate translation execution.
  - Output receipts distinguish application handoff from presentation. They do not prove physical display.
- **Identity and delivery health**
  - Publication/turn identities and explicitly admitted operation/request identities cross each destination boundary unchanged.
  - Optional observer failure is separate delivery-health evidence, not a new translation failure.
- **OSC recovery**
  - OSC owners retain a failed publication obligation until actual local sends complete it.
  - An empty or unrelated delta and old cache equality cannot establish recovery.
  - No local send acknowledges remote receipt.
- **Chatbox typing**
  - Typing reasons are control state, not utterance publications.
  - Repeated input activity creates no publication identity or routing observation.
  - The paginator owns aggregate typing transitions and local-send failure evidence.
- **Preferences and errors**
  - Caption and overlay preferences control destinations, not capture.
  - Runtime errors use the shared dashboard error path.
  - Interactive validation stays on its owning surface.

Implementation: `core/runtime/output.py`, `core/orchestrator/translation_output_projection.py`. Behavior: `tests/core/runtime/test_output_runtime.py`.

### Overlays

| Owner | Responsibility |
| --- | --- |
| Application (`app/services/overlay/`) | Target selection, recovery, and generation replacement |
| Python runtime (`core/overlay/`, `core/runtime/overlay.py`) | Caption lifetime, scene delivery, and process lifecycle |
| Native runtime (`native/overlay/src/runtime.rs`) | VR rendering and GPU resources |

- **Caption ownership**
  - `OverlayPresenter` owns caption admission and lifetime independently of providers.
  - Python owns logical caption state. Native rendering must not redefine that lifetime.
- **Child status and failure ingress**
  - Ingress validates finite safe metadata and rejects stale instances/generations before coalescing.
  - Primary runtime causes and secondary cleanup causes stay separate across the native/Python bridge.
  - Diagnostic-drop loss and reliable-terminal-delivery loss are distinct observations.
  - Reserved lifecycle/control capacity stays independent of ordinary diagnostic traffic.
  - A child acknowledgment or process exit alone does not establish graceful terminal cleanup or physical HMD presentation.
- **Desktop child diagnostics**
  - Ingress reuses the producer's finite event/disposition and bounded-value contract.
  - Invalid declared values are rejected observably before retention or diagnostic artifacts.
  - Ingress does not accept arbitrary strings for later redaction.

Behavior: `tests/core/test_overlay_presenter.py`, `native/overlay/tests/runtime.rs`.

## Runtime Logging

`SessionRuntimeLoggingService` owns console, file, and Logs-view delivery, plus bounded current-instance diagnostic retention.

- Translation owners produce accepted conversation records.
- Capture, provider, and overlay owners produce lifecycle and failure evidence.

Logging contracts:

- **Cause preservation**
  - Owners preserve causes in existing `ErrorDiagnostics` and typed results before converting them to messages or fallback outcomes.
  - `core.diagnostics.emit_failure` packages typed metadata for the existing logging ingress. Domain owners do not import the file writer.
- **Identity and authority**
  - Record identity, occurrence time, and known cause links survive projection.
  - Admitted asynchronous work carries operation/request identity explicitly.
  - Channel, turn, provider epoch, and generation remain under source-owner authority.
  - An expired command context cannot assign unrelated later work to that command.
- **Ingress validation**
  - Ingress validates bounded metadata before memory, CLI, and file projection.
  - Unknown causes stay unknown. Rendered prose is not a classification source.
  - Optional-detail rejection stays observable without erasing permitted code or identity.
- **Audience projection**
  - One retained fact supplies the permitted Basic summary, selected main-file metadata, CLI observation, and receipt reference.
  - Basic and file-only audiences remain distinct. There is no Detailed mode.
- **Log selection**
  - Rendered output logs select meaningful effects rather than mirror every owner decision.
  - Expected policy/no-op decisions stay in bounded owner history without routine log rows.
  - Intermediate overlay handoff/pacing evidence is file-only.
  - Typed output failures are not repeated as generic destination or translation-error summaries.
  - GUI overlay notifications do not duplicate owner lifecycle logs.
- **Delivery and health**
  - File delivery is asynchronous and bounded.
  - Live health observes queue, writer, rejection, and subscriber loss independently of writing to the failed sink.
  - Enqueue, write, and durable persistence are distinct. Current-run history is not a durable audit.
- **Privacy**
  - Technical diagnostics exclude credentials, user text, audio, arbitrary exception/child content, and personal paths.
  - Accepted conversation records keep their separate secret-protected path and explicit CLI content opt-ins.
- **Timing**
  - Timing records describe application-observable stages, not physical display.

References:

- Implementation: `core/diagnostics.py`, `core/runtime_logging.py`, `app/services/application_runtime_logging.py`.
- Logging behavior: `tests/core/test_runtime_logging.py`.
- Output failure correlation, privacy, and noise suppression: `tests/core/test_output_logging.py`.
- Schema, query limits, and loss semantics: [CLI guide](cli.md#retained-diagnostics-and-delivery-health).

## Runtime Layout

- `runtime_layout.py` separates host/interpreter paths, read-only resources, and writable user data. Features use this boundary rather than process flags or the working directory.
- Bootstrap selects runtime and resource paths before application startup. Packaging does not change feature ownership.
- Native distributions bundle Python, application code, dependencies, and resources as an immutable payload.
  - Startup must not write into installed directories.
  - Code changes require rebuilding the payload.
- Import and resource-loading adapters preserve application contracts across source and packaged hosts.
  - Native libraries and filesystem-backed resources remain accessible through their runtime layout.
- Installer cleanup owns only verified distribution files. It does not own modified or unlisted files or writable user data.

Build policy and validation: `scripts/ci/build-native-experimental.ps1`, `release_evidence/native_distribution.py`. Runtime behavior: `tests/release_evidence/test_native_runtime_layout.py`, `tests/release_evidence/test_native_python_runtime.py`, `tests/release_evidence/test_native_bootstrap.py`.

### Model assets

- **Source contracts**
  - Model-specific source registries and packaged file contracts own download endpoints, revisions, sizes, and hashes.
  - Changing the preferred source alone does not invalidate a current installed model.
- **Staging and promotion**
  - Each source attempt writes to owned staging under the model provisioning lease.
  - Transfer or integrity failure prevents promotion of that attempt.
  - A successful attempt is validated before replacing the installed model.
- **Provisioning lifecycle**
  - Setup provisions required CPU ASR assets before app replacement. Provisioning failure blocks installation.
  - Managed Gemma and optional GPU ASR keep on-demand runtime preparation.
  - Adding mirrors does not change these lifecycle owners.

Implementation: `core/local_asr/local_stt_runtime_installer.py`, `core/local_translation/provisioning.py`, `installer.iss`. Behavior: `tests/core/test_local_stt_runtime_installer.py`, `tests/core/local_translation/test_gemma_provisioning.py`, `tests/scripts/test_install_local_stt_model.py`.

## Lifecycle

Every task, process, source, and provider session has an owner. That owner is responsible for:

- Stopping ingress.
- Cancelling or draining work.
- Rejecting late callbacks.
- Releasing resources.
- Restarting.

### Stale-work protection and replacement

- Generations, attachment tokens, request IDs, and current-owner checks prevent retired work from mutating current state or publishing output.
- The owner decides whether admitted work drains or is cancelled.
- Gracefully admitted work retains its provider scope and frozen settings.
- Replacement revokes retired authority before it can affect the new runtime.
- Retired resources remain owned until cleanup completes.
- Source-ordered recognition handoff evaluates cleanup debt after backend shutdown, not after channel-only close.

### Shutdown

Shutdown order:

1. Stop ingress.
2. Drain or cancel owned work.
3. Close external resources.
4. Clear runtime references.

- **Application coordination**
  - The application shutdown adapter coordinates capture, translation, output, child processes, and services.
  - Ordered teardown survives ordinary UI-task cancellation.
  - Child processes remain owned for the host lifetime.
- **Failure ownership**
  - Primary operational failures and bounded secondary cleanup failures remain separate.
  - GPU activation failure preserves its original typed cause.
  - Failed temporary cleanup remains owned and visible to established explicit retry/close paths.
- **GPU process cleanup**
  - Startup resources that fail cleanup stay with the process factory owned by `SharedGpuASRRuntime`.
  - Runtime shutdown invokes that factory's explicit cleanup port.
  - Supplied admission operation/request identities reach process startup before first failure retention.
  - Neither an absent client nor a failed termination attempt establishes completed resource release.
- **Output cleanup**
  - Independent output destinations continue cleanup when one destination fails.

Exact handoff and teardown ordering belongs to owner implementations and lifecycle tests. Application entry point: `app/adapters/application_runtime_shutdown.py`.

## Async Event Model

- The Python application runs asynchronous runtime work on its `asyncio` event loop.
- Owners create, track, and close their own background tasks.
- Assign lifecycle ownership before creating detached tasks.
- Capture, STT, translation, UI, and child-process events cross owner boundaries through ports, callbacks, or owned queues.
- Callbacks delegate to the receiving owner. They must not mutate another owner's private runtime state.
- Ordering is local to the owning channel or queue. There is no assumed global ordering across self, peer, UI, and provider events.
- Run blocking model, device, native, startup credential, TLS-policy, and endpoint-publication work through the established worker, executor, or child-process boundary. Keep application state, async clients, and presentation on the event loop; cancellation drains owned synchronous work before releasing its resources.

### Bounded task-creation enforcement

`tests/architecture/test_lifecycle_task_guard.py` scans all Python files under `src/puripuly_heart`, including evidence runners, using `tests/helpers/task_creation.py`. It recognizes direct asyncio task creation (`create_task`, `Task`, `ensure_future`), ordinary imported/assigned aliases, resolvable loop `create_task`, and identifiable Flet page or injected `run_task` bindings. Nested function and lambda bodies use completed enclosing lexical bindings, including aliases assigned after the callback definition; defaults and decorators retain definition-order analysis. A method name alone does not establish task creation.

Each reviewed identity names its module, enclosing qualified function, callee family, maximum multiplicity, and lifetime rationale. Owning primitives, named owners, and legacy exceptions are separate dispositions. Removal is allowed; a new identity or increased multiplicity requires explicit review. Missing/empty protected roots and source discovery, decoding, or parsing failures fail the check.

`ManagedGemmaRuntimeOwner._await_server_work` is a named-owner boundary: `_server_tasks` retains each shielded native operation until its outcome is consumed. Admission limits translation and prefix work together to two live workers on Vulkan or one on CPU; three resident slots do not increase that bound. Caller cancellation waits for settlement, and lifecycle stop bounds remaining work before transport/process teardown. The registry's single creation-site count is lexical, not a one-request runtime policy.

This is bounded lexical analysis, not dynamic task-factory or whole-program alias resolution. Callback invocation timing, arbitrary untyped receivers, interprocedural aliases, star imports, indirect inheritance, global/nonlocal rebinding, pattern bindings, and conditional-flow joins are not resolved. Structural acceptance does not prove cancellation, draining, cleanup, or generation progress; owner behavior tests retain those responsibilities.

Existing narrow lifetime debt is recorded at the exact creation identities, without changing runtime behavior:

- `ScopedRecognitionEngine._normalization_diagnostic` does not retain its notification task; `_notify_terminal_failure` retains notifications, but `close()` does not drain either notification path.
- `GuiControlBoundary.start` retains its stop watcher, which `stop()` cancels without joining.
- `_AsyncioOverlayProcess.terminate` shields an exit-confirmation waiter without retaining, cancelling, or joining it after termination timeout.
- The evidence runner's `local_asr_production_composition._run_stage` cancels/gathers on timeout, but not on caller cancellation.

These exceptions permit only their reviewed existing identities and caps. The Flet application shutdown driver is separately retained and excluded from cancelling itself; that is not a claim that ordinary page callback cleanup drains the driver.
