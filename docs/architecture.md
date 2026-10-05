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
| Peer capture owner      | Target, source, VAD, task, provider attachment, generation | `core/runtime/peer_channel.py`                       |
| Local ASR runtime       | Local recognition channels and backend transitions         | `core/runtime/local_asr_provider_runtime.py`         |
| Managed local translation | Gemma provisioning, readiness, backend, prefix, and process lifecycle | `app/services/managed_gemma_translation.py` and `core/local_translation/runtime.py` |
| Translation turn owner  | Request lifecycle, cancellation, stale-result rejection    | `core/orchestrator/translation_turn.py`                 |
| Output runtime          | Routing, delivery tasks, destinations, delivery history    | `core/runtime/output.py`                              |
| Overlay owners          | Overlay selection, process lifecycle, state, calibration   | `app/services/overlay/overlay_application.py`            |
| Managed-account runtime | Authentication, entitlement, usage, credential release     | `app/wiring/wiring_managed_account.py`                      |
| ChatGPT account owner   | Sign in with ChatGPT OAuth, refresh-token storage, access-token refresh | `app/services/chatgpt_account.py`, `core/chatgpt/session.py` |
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

- Capture preserves source order and timing; audio loss is explicit, not silence.
- Capture owners retain generation-bound segment ledgers and freeze provider and endpoint settings for admitted work (`core/audio/ownership.py`).
- `OwnedVadEvent` carries local segment identity; `OwnedStreamInput` carries permitted continuous audio and live capture authority. Scoped PCM is deduplicated by source range, including pre-roll.
- `ListenDeliveryController` owns Peer segmentation independently of provider readiness (`core/audio/listen_delivery.py`). Self and Peer share `VadGating` but retain separate onset and endpoint policies.
- `SmartTurnInferenceOwner` owns Peer endpoint inference and rejects retired results (`core/audio/smart_turn.py`).
- Normal source completion settles provider readiness before draining retained input. Stop or discontinuity invalidates affected work; capture faults remain visible even if cleanup also fails.

### Managed translation

Managed authentication → Broker entitlement or credential release → provider activation → normal translation request path. Broker is not an utterance relay.

### VRChat scene context

VRChat log tailing → whitelist parsing and member-set trust → immutable snapshot → sanitized translation request context.

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

The Settings view receives the application-owned HTTP extension registry at construction; it does not create and discard a second default-directory registry.

## Local Application Control

GUI and headless hosts share application owners and runtime resources. Presentation adapters determine whether there is a main window; headless operation retains error state and severity without GUI notifications.

- `ApplicationControlOwner` exposes typed commands and owner-backed queries. CLI commands, ordered GUI intents, and OSC edits use existing owners and shared mutation ordering; resource conflicts have a separate ordering boundary.
- Queries distinguish committed settings from effective runtime state. Owned tasks and bounded operation receipts outlive client connections and distinguish persistence from runtime completion.
- `ControlEvents` provides bounded, privacy-filtered subscriptions. Content requires explicit opt-in; slow clients do not block producers, and event gaps require snapshot resynchronization.
- `HostedApplication` owns the authenticated same-user loopback endpoint and settings-identity lease. Shutdown stops ingress and drains owned operations before releasing runtime resources and the lease.

Entry points are listed in Runtime Ownership; command and protocol details belong in the [CLI guide](cli.md).

## Runtime Pipeline

`RuntimePipelineLauncher` builds and installs the active capture, STT, translation, output, UI-event, and VRChat microphone-state components.

Provider or settings changes may replace components. Do not retain references across replacement unless the API explicitly permits it.

## Configuration

| Layer | Responsibility | Entry points |
| --- | --- | --- |
| Persisted intent | Canonical `AppSettingsVNext` selections, persistence, and migration; no live resources | `app/services/canonical_settings_persistence.py`, `config/settings_vnext/compat.py` |
| Resolved configuration | Provider, model, execution, capture/overlay target, credentials, and capability constraints | `config/resolved.py`, `config/runtime_resolution.py` |
| Runtime state | Active sources, generations, provider attachments, turns, tasks, and processes | Respective lifecycle owners |

`SettingsView` consumes frozen snapshots and emits focused typed intents. The settings owner applies them to the latest canonical settings before persistence and runtime application.

Settings persistence owns intent; runtime owners own its application. The provider-apply boundary coordinates capture and Local ASR owners. Failed or incomplete application must not appear as successfully applied runtime state.

Canonical policy values live in `config/desktop_overlay_values.py`, `config/provider_values.py`, and `config/translation_values.py`, not in this document.

Implementation: `app/services/provider/provider_runtime_apply.py`. Behavior: `tests/app/test_stt_provider_apply_vertical.py`.

## Provider Boundaries

### STT

`ScopedRecognitionEngine` owns channel-scoped recognition (`core/stt/scoped_engine.py`). Self and Peer have separate epochs, bounded buffers, cancellation, and retention; physical CPU/GPU resources remain shared through their runtime owners.

`STTSessionEventProjection` defines scoped turn receipts and independent recognition events (`core/stt/session_projection.py`):

- Turn-bound providers use `STTScopedTurnNormalizer`; provider updates are not final application transcripts.
- Gemini, including its Rolling member, uses automatic server VAD with locally ordered `audio_stream_end` fences. Fences share the audio writer without blocking reception or subsequent input.
- For independent recognition, `STTProviderInputTerminal` retires local audio bookkeeping independently of text delivery; it is not a transcript or server acknowledgement. `STTRecognitionUnitTerminal` carries native finals in provider receipt order, without inventing local segment or speaker correspondence.
- Capture scope and provider epoch jointly authorize results. Input readiness is separate from text authority, so accepted finals can drain without reviving an ended stream.
- Stream recovery is bounded and rechecks live capture authority. It forwards only definitely-unsent retained audio; submitted or uncertain-delivery ranges are not replayed.
- Soniox adapters classify retryable failures; the engine owns recovery budgets. Recoverable failures preserve Self capture intent, while permanent or exhausted failures deactivate capture.

Provider replacement preserves frozen settings for admitted work. Abort revokes turn and epoch authority before native cleanup.

Local execution uses Python-process ASR or a native GPU worker. The Python adapter owns worker launch, authentication, requests, heartbeat, cancellation, and shutdown; Rust owns device discovery, model activation, and transcription.

Behavior: `tests/core/test_stt_scoped_engine.py`, `tests/providers/test_gemini_transcribe_lifecycle.py`, `tests/providers/test_soniox_reuse.py`.

### Translation

- Provider adapters own authentication, endpoint/model mapping, request/response formats, streaming, and provider errors.
- `TranslationTurnLifecycleOwner` owns bounded Self/Peer admission, parent turns, child translations, cancellation, and publication. `TranslationRequestOwner` owns request preparation and provider-generation authority.
- Peer execution may be concurrent, but source-context preparation and publication preserve admitted order: segment order for turn-bound STT, receipt order for independent finals.
- Channel execution limits are separate from provider-wide admission shared by Self and Peer. Self speculative selection stays in the Self owner.
- Managed local Gemma remains behind `LLMProvider`; its application/runtime owners handle provisioning, readiness, and process lifecycle.
- Bounded hedging is resolved runtime policy, not persisted fallback selection (`config/runtime_resolution.py`, `core/llm/fallback_racing.py`).

Implementation: `core/orchestrator/translation_turn.py`, `core/orchestrator/translation_request.py`. Behavior: `tests/core/test_translation_turn_owner.py`, `tests/core/test_translation_request_owner.py`, `tests/core/test_hedged_attempts.py`.

### ChatGPT connection

- `ChatGptAccountOwner` owns loopback OAuth; `ChatGptSession` survives provider rebuilds. Refresh credentials and account metadata use the secret store; access tokens remain in memory. This path does not use Broker.
- `ChatGptPlanLLMProvider` owns the bounded Responses API WebSocket pool and shared FIFO admission across Self and Peer.
- `FallbackRacingLLMProvider` uses `LLMRequestAdmissionPort` / `LLMRequestExecution` to race the attempts granted by available pool capacity, rather than a ChatGPT hedge timer (`core/llm/provider.py`).
- `LlmConnectionReadinessOwner` prepares connections while translation is enabled, independently of Talk and Listen (`app/services/llm_connection_readiness.py`).
- Caller cancellation ends delivery, not an already-running exchange. The provider owns bounded draining and retains pool capacity until release; draining can still consume upstream usage.
- Translation-off retires unused capacity and closes in-flight connections after their responses. Provider close joins exchanges and cleanup; retired reservations cannot revive an old pool generation.

Implementation: `providers/llm/chatgpt_plan.py`. Behavior: `tests/providers/test_chatgpt_plan_provider.py`, `tests/app/test_chatgpt_adaptive_dispatch.py`.

## Output

`OutputRuntime` owns routing, chatbox state, destination admission and receipts, duplicate/stale-publication rejection, destination replacement, and cleanup.

| Publication | UI | Chatbox | Overlay |
| --- | --- | --- | --- |
| Self utterance | Yes | Yes | Yes |
| Peer subtitle | Yes | No | Yes |
| System disclosure | Policy-dependent | Explicit route only | Policy-dependent |

- Each destination has independent bounded delivery state. Failure or replacement in one must not block or retire the others, or replay recognition/translation.
- `TranslationUiMessageQueue` and `OutputRuntime` own separate Self/manual and Peer writer lanes into the shared UI consumer queue. UI consumption does not gate eligible translation or Self source-caption presentation.
- Delivery authority and sequence checks reject stale, replaced, or duplicate callbacks; delayed events cannot overwrite newer dashboard state. Source retirement preserves manual-input isolation.
- Output handoff releases translation ordering without waiting for display. Admission, queue submission, and presenter application are distinct receipts, none a physical-display acknowledgement.
- Peer publications retain activation generation and admitted order. Caption/overlay preferences control destinations, not capture.
- The activation-notice preference is an output policy: apply it after persistence without restarting capture/providers or changing Peer consent.
- Runtime errors use the shared dashboard error path; interactive settings/authentication validation stays on its owning surface. Conversation errors retain publication identity; runtime session status is separate.

Implementation: `core/runtime/output.py`, `core/orchestrator/translation_output_projection.py`, `ui/event_dispatch.py`. Behavior: `tests/core/runtime/test_output_runtime.py`, `tests/core/test_translation_ui_delivery.py`, `tests/core/test_self_ui_isolation.py`.

### Overlays

| Owner | Responsibility |
| --- | --- |
| Application (`app/services/overlay/`) | Target selection, recovery, and generation replacement |
| Python runtime (`core/overlay/`, `core/runtime/overlay.py`) | Caption lifetime, scene delivery, and process lifecycle |
| Native runtime (`native/overlay/src/runtime.rs`) | VR rendering, presentation retries, and GPU resources |

`OverlayPresenter` owns provider-independent Peer admission/pacing and Self source-first captions (`core/overlay/presenter.py`). Translation updates the same logical Self caption; rendering protection must not force early semantic finalization.

Each generation owns its tasks and shutdown. Python projects freshness intent; native alone schedules bounded presentation retries. Retry state must not redefine caption lifetime. Software submission is not proof of physical HMD freshness.

Behavior: `tests/core/test_overlay_presenter.py`, `tests/core/test_overlay_active_freshness.py`, `native/overlay/tests/runtime.rs`.

## Runtime Logging

`SessionRuntimeLoggingService` owns console, file, and Logs-view delivery. Translation owners produce accepted conversation records; capture/provider/overlay owners produce lifecycle and failure evidence.

- File delivery is asynchronous and bounded; producers never wait for file I/O. Queue pressure prioritizes warnings, errors, and terminal evidence, without guaranteeing complete persistence.
- Basic records reach the console and Logs view. Technical diagnostics are metadata-only; accepted conversation has a separate secret-protected path. Diagnostics exclude credentials, user text, raw provider prose, and audio.
- Translation latency uses task-local request/attempt scopes and the existing file writer only (`core/llm/latency.py`); unavailable logging does not fall back to console output.
- E2E timing ends at the Self chatbox send or Peer presenter-application receipt, not physical display. Gemini uses a scoped approximate last-speech origin, not exact utterance attribution; missing origins remain unmeasured.
- Writers retain ownership through stream closure so replacement cannot race cleanup.
- Render-only desktop/preview startup remains independent of domain-model, provider, secrets, and STT imports.

Implementation: `core/runtime_logging.py`, `app/services/application_runtime_logging.py`. Behavior: `tests/core/test_runtime_logging.py`, `tests/core/test_file_logging.py`, `tests/core/test_llm_latency.py`, `tests/app/test_desktop_overlay_runner.py`.

## Runtime Layout

- `runtime_layout.py` separates host/interpreter paths, read-only resources, and writable user data. Features use this boundary rather than process flags or the working directory.
- Bootstrap selects runtime, UI asset, and framework storage paths before application startup. Packaging does not change feature ownership or logging policy.
- Native packaging supplies its Python/VC++ dependencies. The build recompiles application, dependency, and supplied standard-library sources as optimization-0, unchecked-hash bytecode. Compilation uses the staged interpreter with an explicit runtime home and staged dependency search path, disables user-site packages, and restores the caller's Python environment afterward. Sourceless standard-library bytecode from the pinned Python SDK retains its upstream compilation settings and separately recorded provenance. Deployed sources, bytecode, and resources form one immutable payload; code changes require rebuilding it rather than runtime source-hash validation. Startup must not write into installed directories.
- The standard library uses CPython's conventional root-level `python314.zip`, including early interpreter bootstrap modules. Its staged loose `Lib` payload is removed after bundling; the deployed runtime does not depend on that directory.
- Application code and Flet use standard `zipimport` from `app/python.zip`, with bytecode and matching diagnostic sources in the archive. Flet package resources, including `icons.json`, remain available through `importlib.resources`. The standalone `app/product_bootstrap.pyc` stays on disk; its source and parallel loose application/Flet code are absent from the built artifact.
- Other dependency bytecode lives under `_native_dependencies/` in `app/python.zip`, with a versioned module index. A shared archive-backed `SourceFileLoader` preserves physical source filenames, package search paths, source inspection, and filesystem resources. Dependency sources, metadata, native extensions, and DLLs remain on disk, but loose dependency bytecode is removed. Indexed code missing from or corrupt in the archive fails explicitly instead of silently recompiling a source fallback.
- GUI and console bootstrap install the same dependency finder; `sitecustomize` also activates it for configured Python children. Earlier import search locations and native-extension precedence remain authoritative. The GUI host, embedded SeriousPython launcher, console host, and child environment put both code archives before loose application/dependency paths. Bundling reduces distinct code-file accesses without deferring module execution. Build validation and artifact manifests distinguish archive-file identity from member/source hashes, directory entries, and compile provenance.
- Application resources remain filesystem-backed: models, locale bundles, fonts, notices, and images resolve through `RuntimeLayout.package_resource()`, while prompts use the application resource root. Native extensions and DLLs retain their filesystem dependency/runtime layout. Source and PyInstaller hosts preserve their existing resource roots.
- The native Windows GUI host sets `com.salee.PuriPulyHeart` as its process AppUserModelID before creating windows. Installer application shortcuts use the same ID and the packaged product ICO directly, rather than the EXE's cached shell icon. Taskbar grouping can retain an old icon independently of the live window's `WM_GETICON`; verify the actual Explorer taskbar after restarting the installed app. This shell identity is separate from the installer AppId. [Windows AppUserModelID guidance](https://learn.microsoft.com/en-us/windows/win32/shell/appids).
- Main and caption windows pass an absolute filesystem ICO path to Flet's `Window.icon`, not an asset URL. Runtime window icons are separate from the EXE resource and AppUserModelID. The embedded client resolves relative assets under `app/assets`, while product icons reside under `app/puripuly_heart/data/icons`; an unresolved relative ICO can clear the runtime window icons.
- Installer cleanup is limited to manifest-owned obsolete files with identity/hash checks. Modified or unlisted files and writable user data remain outside cleanup ownership.
- Native installers require Inno Setup 7.1.0 or newer for extended-length runtime paths and cleanup of previously installed payloads. The native release workflow pins the official 7.1.0 x64 installer by SHA256, prepares a portable compiler under the runner's temporary directory, and passes its path through `build-native-release-artifacts.ps1 -InnoSetupCompiler`. The release script verifies the compiler's exact version before building; local callers may omit the path to use a standard Inno Setup 7 installation. [Inno Setup 7 release notes](https://github.com/jrsoftware/issrc/releases/tag/is-7_1_0) document the removal of `MAX_PATH` limits.
- Soxr source archives retain their pinned SHA256 checks and are extracted by the prepared Python interpreter's isolated `tarfile` CLI with the `data` extraction filter, not a PATH-selected `tar`. Each extraction has a 60-second deadline and terminates its process tree on timeout. Native release/build command wrappers record the executable, arguments, and elapsed time so compilation, packaging, and validation stalls can be distinguished.
- Font resources are immutable for the process lifetime. The UI caches font-file resolution by asset root and family; the main window registers its locale fonts, while the desktop caption renderer registers only its caption font (`ui/fonts.py`).
- Core audio/network initialization stays on its existing composition paths. Deepgram, ElevenLabs, and QwenAudio verification implementations, encrypted-store cryptography, managed-identity signing, and OAuth verification dependencies import eagerly; startup optimization does not postpone these imports until the first operation.

Build policy and validation: `scripts/ci/build-native-experimental.ps1`, `release_evidence/native_distribution.py`, `tests/release_evidence/test_native_distribution.py`. Runtime-path and archive-loader behavior: `tests/test_runtime_layout.py`, `tests/test_native_python_runtime.py`.

## Lifecycle

Every task, process, source, and provider session has an owner responsible for ingress stop, cancellation/draining, late-callback rejection, resource release, and restart.

### Stale-work protection and replacement

- Generations, attachment tokens, request IDs, and current-owner checks prevent retired work from mutating current state or publishing output.
- The owner decides whether admitted work drains or is cancelled. Gracefully admitted work retains its provider scope and frozen settings.
- Replacement revokes retired authority before it can affect the new runtime. Retired resources remain owned until cleanup completes.

### Shutdown

Stop ingress, drain or cancel owned work, close external resources, then clear runtime references.

The application shutdown adapter coordinates capture, translation, output, child processes, and services. Window-close orchestration survives ordinary UI-task cancellation until ordered shutdown completes. Child processes remain owned for the host lifetime; abrupt-exit containment is only a fallback.

Exact handoff and teardown ordering belongs to owner implementations and lifecycle tests. Application entry point: `app/adapters/application_runtime_shutdown.py`.

## Async Event Model

- The Python application runs asynchronous runtime work on its `asyncio` event loop.
- Owners create, track, and close their own background tasks.
- Do not create detached tasks without assigning lifecycle ownership.
- Capture, STT, translation, UI, and child-process events cross owner boundaries through ports, callbacks, or owned queues.
- Callbacks must delegate to the receiving owner; they must not mutate another owner's private runtime state.
- Ordering is local to the owning channel or queue. Do not assume global ordering across self, peer, UI, and provider events.
- Blocking model, device, or native work must not block the application event loop; use the established worker, executor, or child-process boundary.
