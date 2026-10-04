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

### Self speech

```text
microphone
→ normalized audio frames
→ owned VAD boundaries and permitted stream input
→ scoped turn results or independent recognition units
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
→ scoped turn results or independent recognition units
→ ordered peer translation admission
→ publication intents
→ output runtime
→ UI / overlays
```

Peer output must not reach the VRChat chatbox.

### Audio ownership

- Capture preserves source order and timing. Audio loss is explicit, not silence.
- Capture owners retain generation-bound segment ledgers and freeze provider and endpoint settings for admitted segments (`core/audio/ownership.py`).
- `OwnedVadEvent` carries local segment identity; Gemini and its Rolling member also receive generation-owned `OwnedStreamInput` from the same permitted normalized frames. All scoped PCM, including VAD pre-roll, deduplicates by source range within its recognition stream. Initial unseen context and unseen suffixes are preserved; already-submitted context is not replayed.
- Gemini `STTProviderInputTerminal` retires an audio slot after submission, without claiming a transcript or server completion. Independently received `STTRecognitionUnitTerminal` supplies text with channel, capture/activation generation, provider epoch, settings scope, and receipt identity.
- Independent Gemini finals, including the Rolling member, freeze an approximate last-speech origin from the latest VAD-positive real capture frame observed by the scoped engine. Each channel keeps one timestamp and its capture/settings scope; startup preserves the pending same-scope observation, while stream/provider retirement, capture changes, mute, and discontinuity reset it. Silence and local input submission do not advance or erase it. No per-utterance matching, provider-offset map, or artificial speech-end event is introduced.
- Input terminal metadata goes directly to the bound capture callback, independently of deferred recognition delivery, so a slow text consumer cannot retain audio slots or lose their retirement on text-buffer overflow.
- Input retirement also clears that exact local segment's VAD/timing bookkeeping; native unit identities are never used to guess a local segment. Failed open inputs are failure-sealed, and later VAD events cannot reopen their retired slots.
- A failed independent audio write retires its provider epoch before recovery. The next valid retained source frame can admit a replacement recognition stream through `begin_stream`, without inventing a local speech segment or waiting for a new `SpeechStart`. Recovery excludes the failed write's entire source range because remote delivery is unknown, then forwards only definitely-unsent audio under the existing capture retention budget.
- Recovery opening and stream admission are bounded and recheck live capture authority after awaits. Generation, settings, capture-epoch, mute, stop, and discontinuity invalidation prevent stale recovery; exhausted connection or physical cleanup failure does not trigger a new attempt for every queued frame. Capture owners supply live `OwnedStreamInput.is_current` guards, and Rolling preserves the same stream ownership contract.
- Queued PCM remains capture-epoch guarded. Ordered boundary controls retain generation authority but are not invalidated when a subsequent frame updates the current capture epoch; a discontinuity must retire the previous stream even while its writer was delayed.
- `ListenDeliveryController` owns peer segmentation independently of provider readiness (`core/audio/listen_delivery.py`). Deadline seals serialize with the current frame's VAD processing, continuous-input enqueue, and owned-event dispatch, so a timer cannot seal a segment before its already-produced frame is accounted for and queued.
- Self and peer share `VadGating` but retain separate onset and endpoint policies. Delivery rollover preserves acoustic continuity.
- `SmartTurnInferenceOwner` owns peer endpoint inference for supported languages and rejects retired results (`core/audio/smart_turn.py`).

Orderly capture completion drains recognition when provider ingress is ready. At normal source end, Peer first waits for any in-flight provider setup to settle: successful startup drains the retained finite input, while a returned pending result aborts and releases dispatch work before source/provider teardown, without fabricating readiness. Stop or discontinuity invalidates affected work.
Gemini requests automatic server activity detection with `prefix_padding_ms=500` and `silence_duration_ms=400`. These provider settings do not alter local VAD/SmartTurn policies, capture pre-roll, continuous PCM coverage, or native-final-only text admission; no artificial silence or repeated audio is added.
Gemini local endpoints enqueue `audio_stream_end` through the same ordered writer as audio; they do not stop the receiver or block subsequent input waiting for text. Mute, discontinuity, source change, and explicit stop retire the affected stream. Stream-only queued audio does not count as pending local speech for idle lifetime extension.
Consecutive fences without newly written audio are coalesced by that writer. A server GoAway notice with positive `timeLeft` leaves audio and final reception live until the first bounded deadline or socket closure; duplicate notices cannot extend the deadline. Event pressure preserves already-accepted finals and reserves one bounded retirement-control slot rather than clearing accepted text.
An unchanged effective SELF intent preserves its active capture generation. A new capture generation or capture epoch starts a fresh native stream even when provider settings are unchanged. SELF readiness is tracked separately from text authority: current input failure or the end of either the ready or latest submitted epoch disconnects readiness, including before that epoch's first native final. The ended stream is fenced before publishing the state change, so already-accepted finals may still drain without reviving an ended or superseded stream.

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
| Local application control | Typed commands, queries, operations, and events | `ApplicationControlOwner`, local transport |
| UI application  | `UiApplicationPort`                                  | `UiApplicationBoundary` |
| UI presentation | `UiPresentationPort`, `UIEventBridgePort`            | Flet and headless presentation adapters   |
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

## Local Application Control

GUI and headless hosts share application owners and runtime resources. Presentation adapters select whether the host has a main window. Headless presentation preserves application error state and severity without GUI notifications.

`ApplicationControlOwner` exposes a finite catalog of typed commands and owner-backed queries. Settings projections and runtime dependencies come from existing application owners and composition.

- Settings and provider edits use existing owners, with shared ordering for CLI commands, ordered GUI intents, and OSC edits.
- Canonical mutations and resource conflicts have separate ordering boundaries.
- `settings.current` projects committed settings. Status queries distinguish selected settings from effective runtime state.
- Submitted tasks and bounded operation receipts belong to the control owner, not client connections. Receipts distinguish durable settings commits from runtime completion.
- `ControlEvents` provides bounded, privacy-filtered subscriptions to the shared runtime event stream. Content requires explicit opt-in; slow clients do not block producers, and gaps require snapshot resynchronization.

`HostedApplication` owns the authenticated same-user loopback endpoint and settings-identity lease. Shutdown stops ingress and drains owned operations before releasing runtime resources and the lease.

Implementation: `app/services/application_control.py`, `app/services/application_control_events.py`, `cli/host.py`, `cli/transport.py`, `core/control_instance.py`. Command and protocol details: [CLI guide](cli.md).

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

`intent.osc.activation_notice_enabled` defaults to `true` and controls only the Talk and Listen activation chatbox notices. The General tab's fifth row exposes one direct on/off card and two empty cards; its existing four rows are unchanged. Notice-only edits persist through `ActivationNoticeSettingsIntent`, then synchronously update the active output owner without preparing or restarting capture, providers, or overlays. Failed persistence restores the committed settings projection and leaves the output policy unchanged.


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
- Physical CPU/GPU resources remain shared through their runtime owners.
- `STTSessionEventProjection` defines scoped turn receipts and independent recognition events (`core/stt/session_projection.py`). `STTScopedTurnNormalizer` remains the turn-bound provider path; it does not attach Gemini text to a local turn.
- Gemini uses automatic server VAD with locally led `audio_stream_end` fences, 16-kHz mono PCM16LE, and unchanged 32-ms packetization. Manual activity controls, final-plus-ACK matching, and interim-to-final timeout promotion are absent.
- Gemini finals are consumed exactly once in provider receipt order through the existing translation owners, including Rolling. This is an explicitly approved output policy, not a guarantee of original-audio ordering. Activity offsets and receive time do not establish local segment, word, or speaker correspondence.
- Currentness requires both capture and provider-runtime ownership. Missing text is not empty success; only explicit native finals become independent recognition units. Bounded event/audio buffers and finite EOF observation constrain resource retention without asserting that every source sample has a result.
- Soniox adapters classify retryable failures; the engine owns a shared three-failure recovery budget with 0.8/1.6-second backoff. Successful final or empty results reset the budget. Recovery opens a fresh epoch for the next valid utterance without replaying failed audio. Authentication, configuration, protocol, and unknown faults are not retried.
- Recoverable Soniox terminals preserve self capture intent without publishing a terminal UI error. Permanent or exhausted failures deactivate capture. User abort invalidates pending admission and late results. Retained self capture keeps the source token and ledger generation aligned; already-admitted segments retain their frozen identity.
- Soniox readiness is bounded at 5 seconds. Final wait is 5 seconds for peer and 20 seconds for self; peer sealed-segment TTL remains 12 seconds. The peer deadline reserves time for queued work but does not guarantee delivery through repeated failures, and later final responses lose authority.

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

The managed local Gemma adapter remains behind `LLMProvider`; its application/runtime owners handle model provisioning, backend readiness, and process lifecycle.

GPT 6 Luna over the `chatgpt` connection uses the user's ChatGPT plan through Sign in with ChatGPT:

- `ChatGptAccountOwner` runs the loopback OAuth flow (PKCE, dynamic client registration, ID-token verification) and never routes through the Broker.
- `ChatGptSession` is shared across provider rebuilds. The secret store keeps only the refresh token, issued client ID, host ID, and account label; access tokens stay in memory because they exceed the Windows credential size limit.
- `ChatGptPlanLLMProvider` owns a Responses API WebSocket pool with six prepared connections and a six-connection limit. Opening, reserved, running, draining, and closing connections all retain their pool slots until ownership ends.
- Logical requests share one FIFO admission queue across Self and Peer. At an event-loop boundary, the pool first gives queued requests one ready idle connection each, then gives remaining idle connections to newly admitted requests in FIFO order, at most one extra each. There is no batching timer, mandatory pair, or later upgrade from one attempt to two.
- `FallbackRacingLLMProvider` uses the core `LLMRequestAdmissionPort` and per-request `LLMRequestExecution` contract to start the granted one or two attempts immediately and publish the first complete success. ChatGPT has no 1,700 ms hedge timer. A single granted attempt may re-enter FIFO admission once with a one-attempt recovery after failure; a paired request cannot start a third attempt. Existing authentication retry behavior remains separate. Direct `ChatGptPlanLLMProvider.translate()` stays single-attempt.
- `LlmConnectionReadinessOwner` prepares the pool while translation is on, independently of Talk and Listen. Pipeline installation and provider replacement also synchronize readiness, so initial and replacement providers can prepare before their first translation.
- Cancelling an attempt stops delivery to its caller without closing a running exchange. The provider drains it under the original 30-second response timeout and returns the connection only after successful completion. A cancelled waiter sends no abandoned request, and unused reservations return exactly once. Draining responses retain their pool slots and continue to consume upstream usage. Admission wait remains observable as connection wait, separately from the logical concurrency semaphore queue.
- Errors, response timeouts, and obsolete token generations retire connections. Turning translation off cancels queued admission and opening work, retires idle and unused reserved connections, and lets in-flight exchanges close after their responses instead of returning to the pool. Retired reservations cannot send or reopen the old pool generation. Provider close cancels and joins owned exchanges and cleanup, including background drains.
- Detached-connection cleanup finishes before cancellation propagates. Cancelling preparation closes partially opened connections and releases reserved pool slots; closing the readiness owner cannot advance into queued preparation (`app/services/llm_connection_readiness.py`, `providers/llm/chatgpt_plan.py`).
- ChatGPT login permission failures use a warning snackbar. Inference errors use the dashboard's primary text slot: structured eligibility and usage-limit codes select dedicated subscription and Codex-limit guidance; without a subscription code, HTTP 403 and 429 select the same respective messages. Explicit subscription codes retain precedence, other providers keep their own classification, and unanimous parallel-attempt failures preserve the dedicated message (`core/error_messages.py`, `ui/event_dispatch.py`).

Cloud translation may use bounded hedged attempts according to resolved runtime policy, not persisted fallback selections (`config/runtime_resolution.py`, `core/llm/fallback_racing.py`).

Translation owners retain:

- turn lifecycle,
- cancellation,
- stale-result rejection,
- publication handoff.

`TranslationTurnLifecycleOwner` owns bounded Self and Peer admission and the lifecycle of parent turns and child translations. `TranslationRequestOwner` owns request preparation and provider-generation authority.

Peer translations may execute concurrently, but source-context preparation and publication preserve source order. Channel execution limits remain separate from provider-wide admission shared by Self and Peer.

Self speculative selection remains in the Self owner. Once a turn is admitted, the turn lifecycle owns subsequent translation and publication.

Implementation: `core/orchestrator/translation_turn.py`, `core/orchestrator/translation_request.py`, `core/llm/provider.py`, `core/llm/fallback_racing.py`, `providers/llm/chatgpt_plan.py`. Behavior tests: `tests/core/test_translation_turn_owner.py`, `tests/core/test_translation_request_owner.py`, `tests/core/test_hedged_attempts.py`, `tests/providers/test_chatgpt_plan_provider.py`, `tests/app/test_chatgpt_adaptive_dispatch.py`.

## Output

`OutputRuntime` owns:

- route selection and chatbox state,
- destination-scoped admission and delivery receipts,
- duplicate and retired-publication rejection,
- UI event bridge,
- destination replacement,
- shutdown cleanup.

Delivery boundaries:

- Self/manual and Peer UI publications use independently bounded writer lanes owned by `TranslationUiMessageQueue` and `OutputRuntime`, sharing the production capacity-one consumer queue and destination-sequence authority. UI admission does not wait for consumption or gate Self source Presenter application and otherwise eligible translation execution. Peer overlay delivery remains independently bounded.
- Self chatbox delivery owns its bounded admission and expiry policy.
- `OutputRuntime.activation_notice_enabled` gates the immediate Talk notice and queued Listen disclosure before destination handoff. Disabled notices produce an `activation_notice_disabled` routing outcome; enabling the preference does not replay them. Ordinary Self output, typing, subtitles, errors, and the initial Peer consent requirement are unchanged. Pipeline construction and recreation initialize this policy from canonical settings; Talk's existing activation eligibility and cooldown remain owned by the Self translation channel.
- Output handoff releases translation ordering without waiting for display. Sink failure does not replay recognition or translation.
- Peer publications retain activation generation and `source_order` through output. For turn-bound providers this follows segment order; independent Gemini finals use receipt-ordered admission into the same monotonic publication sequence. Retiring an activation cancels its deliveries and rejects late work.
- Peer text without speaker runs, including independent Gemini finals, is `non_diarized` and uses the existing gold style without a speaker hold or guessed identity. Explicit uncertain or missing speaker attribution keeps the gray fallback; first-readable presentation remains pinned.
- Destination admission and presenter application receipts are explicit; neither is a remote display acknowledgement.
- E2E summaries measure last source speech to the first successful Self chatbox page send or the Peer presenter application receipt. Gemini's frozen approximate origin propagates through the existing latency timeline without waiting for local `SpeechEnd`; its summaries include `estimated=true`. Missing speech observations remain unmeasured. A newer utterance observed before an older native final can underestimate the older result's latency; these estimates are not exact utterance attribution.

Self/manual UI delivery retains 32 waiting events plus one active event. Peer retains eight waiting batches plus an active batch, each with at most 32 outstanding events including its active write. Delivered Peer payloads are released; this is not a limit on lifetime batch emissions or provider segmentation. Each lane owns one writer with a five-second write timeout. Including the queue and active consumer, these boundaries retain at most 323 distinct event payloads. Capacity exhaustion, write failure, retirement, replacement, and shutdown receive explicit destination-local routing dispositions rather than replaying recognition or translation. `accepted_handoff` means admission; `ui_queue_submitted` means local queue submission, not UI application or physical display.

Optional `UIEvent` delivery authority rejects retired, replaced, or duplicate callbacks. A shared sequence prevents delayed older Self/manual/Peer events from replacing newer visible dashboard state while preserving authorized logical history and error handling. Source retirement preserves manual isolation. Destination replacement joins both UI writers without retiring other output destinations. Context preparation, predecessor ordering, execution slots, and speculative reuse remain translation-owner constraints, independent of UI consumption.

Caption and overlay settings control destinations, not peer capture. Conversation errors share publication identity; runtime session status uses a separate path.

Runtime error messages use plain text in the dashboard's upper-right `DisplayCard`, through `UIEventBridge` / `AppDashboardEventDestination` or explicitly error-marked application messages. Provider-specific errors, including OpenRouter and managed-account failures, do not add separate banners, settings actions, or error snackbars. Overlay failures use the existing reason-specific dashboard notice, which yields to conversation content and clears on recovery. Local-ASR feedback distinguishes failures from progress and compatibility notices. Interactive settings/authentication validation and non-error notifications remain local to their owning surfaces.


| Publication       | UI               | Chatbox             | Overlay          |
| ----------------- | ---------------- | ------------------- | ---------------- |
| Self utterance    | Yes              | Yes                 | Yes              |
| Peer subtitle     | Yes              | No                  | Yes              |
| System disclosure | Policy-dependent | Explicit route only | Policy-dependent |


Destination adapters must not bypass routing policy.

Each destination has independent admission and delivery state. Replacing one
destination must not block or retire work for the others.

Implementation: `core/runtime/output.py`, `core/orchestrator/translation_output_projection.py`, `ui/event_dispatch.py`. Behavior tests: `tests/core/runtime/test_output_runtime.py`, `tests/core/test_translation_ui_delivery.py`, `tests/core/test_self_ui_isolation.py`.

### Overlays

| Owner | Responsibility |
| --- | --- |
| Application (`app/services/overlay/`) | Target selection, recovery, and generation replacement |
| Python runtime (`core/overlay/`, `core/runtime/overlay.py`) | Caption state and expiry, scene delivery, and process lifecycle |
| Native runtime (`native/overlay/src/runtime.rs`) | VR rendering, presentation retries, and GPU resources |

Each generation owns its tasks and shutdown. Python owns caption lifetime; native owns presentation retries.

`OverlayPresenter` owns provider-independent Peer subtitle admission and pacing (`core/overlay/presenter.py`); output retains bounded waiting work.

SELF source-first presentation uses normalized stable contributions when available and authoritative terminal or independent results otherwise. Source remains the primary line; translation updates the same logical caption. Active text is already readable and is not prematurely finalized to obtain rendering protection. Merge/speculation, sticky preview translation, active-row protection, and existing late-result/expiry rules remain independent of native retries.

Changed, visible active SELF captions establish stream-phase freshness through `OverlayPresenter` and `NativeRetryIntentProjection`. Same-target updates advance trigger generation without renewing the stream episode's deadline or completed count. Semantic finalization enters the final phase; a changed final translation retains its distinct final episode. Unchanged content does not trigger freshness. Native alone schedules the existing bounded retries; desktop rendering has no retry cadence. Scene coalescing may display source and translation together without an original-only dwell or render acknowledgement.

Behavior tests: `tests/core/test_overlay_presenter.py`, `tests/core/test_overlay_active_freshness.py`, `tests/core/test_overlay_bridge.py`, and `native/overlay/tests/runtime.rs`. Software application/submission evidence is not physical HMD freshness evidence.

## Runtime Logging

| Owner | Responsibility |
| --- | --- |
| `SessionRuntimeLoggingService` | Shared console, local file, and Logs view delivery |
| Translation owners | Accepted SELF/PEER source and target records |
| Overlay owners | Bounded failure evidence and reliable lifecycle warnings |

- `SessionRuntimeLoggingService` owns bounded asynchronous file delivery. Producers must not block on file I/O.
- Basic-audience records reach the console and Logs view. Selected technical diagnostics are file-only and metadata-only; accepted conversation uses a separate secret-protected path.
- Capture basic logs report input stalls/resumption and VAD speech boundaries, not periodic frame/speech-presence summaries. Peer VAD diagnostic windows remain available.
- Queue pressure prioritizes warning, error, and terminal evidence. Logging does not guarantee complete persistence.
- The writer retains ownership through stream closure; replacement must not race a retiring writer.
- Persisted records include calendar date and process ID. Recognition terminals correlate channel, utterance, provider epoch/turn, activation generation, watchdog timing, and recovery decisions. Self capture failures identify the actual active-intent transition; peer expiry records sealed wait and TTL.
- Soniox file-only turn summaries distinguish finalize enqueue/write, final reception/acceptance, server error codes, transport closure, and local cleanup. A completed write is not a server acknowledgement. Diagnostic fields exclude external error prose, credentials, transcript tokens, speaker identities, and PCM.
- Translation latency uses task-local request and attempt scopes (`core/llm/latency.py`). Only `request_end` and `attempt_end` summaries are written under `[Diagnostic][LlmLatency]`; there are no start/per-token events, tokenization, prompt hashes, or separate collectors.
- These summaries go through the existing asynchronous file writer only, not the Logs view, console, control log events, or structured diagnostic fanout. An unavailable or closed logging service drops them without a console fallback. Basic E2E and conversation logging remain unchanged.
- `request_ms` covers backend execution through completion-authority checks and normalization, not preparation, scheduling, capture, or output delivery. `queue_ms` measures permit acquisition. Attempts identify provider, configured/actual model, transport, outcome, and elapsed time; `winner_attempt` is zero-based. Cancelled hedge summaries can follow their correlated request summary.
- `network_ms` runs from the latest send to attempt closure, including response processing and connection release. `ttft_ms` records only the first nonempty streamed text; `first_text_to_done_ms` runs from that text to attempt closure. Nonstreaming responses leave these text timings `none`, never substituting headers or full-response time.
- A cancelled ChatGPT attempt summary ends when its caller is cancelled; the provider-owned background drain is excluded and emits no second attempt summary.
- Auth, connection-pool wait, actual handshake/reuse, service tier, response usage, and local server timings are recorded only where observable. Usage is provider-reported, not estimated; unavailable/ambiguous counts remain `none`. Source, prompt, and context contribute character counts only. Summaries contain no text, headers, URLs, credentials, or external error prose.

Startup logging and latency imports remain safe for render-only desktop and preview dispatch. The LLM provider and racing contracts keep annotation-only domain-model imports behind `TYPE_CHECKING`; configuring logging must not load domain models, provider adapters, secrets, or STT. The real-process dispatch checks in `tests/app/test_desktop_overlay_runner.py` enforce this boundary.


Implementation: `core/runtime_logging.py`, `core/llm/latency.py`, `app/services/application_runtime_logging.py`. Behavior tests: `tests/core/test_runtime_logging.py`, `tests/core/test_file_logging.py`, `tests/core/test_llm_latency.py`.

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
