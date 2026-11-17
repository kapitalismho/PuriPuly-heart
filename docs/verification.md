# Product-area verification

Choose the smallest path through the actual application that establishes the changed behavior. These examples are starting points, not required scenarios, minimum coverage, or an all-area sweep. Adapt inputs, tools, sequence, and additional checks to the engineering decision. Product areas organize this map; they do not limit the scope of a change.

## Shared entrypoints and evidence

- Use the [CLI guide](cli.md) for source/installed entrypoints, instance selection, isolation, shutdown, syntax discovery, operation results, events, and privacy. Read the running instance's `capabilities`, `settings choices`, and command help rather than copying a stale command catalog.
- Use the [architecture map](architecture.md) to locate owners and connections, the [development guide](../.github/CONTRIBUTING.md) for build prerequisites, and [runtime layout](../src/puripuly_heart/runtime_layout.py) for writable/resource roots. Isolate settings, secrets, discovery, providers, telemetry, and outputs; a separate settings file alone is insufficient. Use synthetic content and controlled local resources, then stop owned instances and clean up only probe-created resources.
- Identify the checkout/import or rebuilt executable and the selected instance. Restart source hosts after code changes. Read terminal operation results and the relevant owner's effective state: command success or admission alone does not establish downstream completion, saved settings, restart persistence, or presentation.
- Follow the **same concrete input/result** across affected connections, using turn/channel, operation, photo/attempt, or conversation identities where available. Separate passing ASR, translation, and screenshot checks do not establish a connected speech-to-caption flow. Unchanged upstream stages may use prepared inputs when those inputs reach the affected production path.
- Observe the actual GUI or headset when the claim concerns presentation or interaction. Renderer readiness, producer logs, component probes, offline artifacts, and simulator results have narrower scopes. Use [retained diagnostics and delivery health](cli.md#retained-diagnostics-and-delivery-health) to investigate missing results, not to infer success from absent errors.

## Settings and application management

**Entrypoints:** GUI settings and application controls; CLI `app`, `settings current`, `settings choices`, and `settings apply`; terminal operations and relevant status queries. [Settings/runtime semantics](cli.md#settings-and-runtime-state) distinguish persisted intent from applied resources. Use [account controls](cli.md#credentials-authorization-and-consent) and model/provider readiness only when the changed behavior needs them.

**Example — save and restart:** Start an isolated source or installed host with a known setting and record its instance identity and revision. Choose a supported harmless change, apply/save it through the application, and inspect the terminal result, committed value, and relevant effective state. Stop the host to completion, start a fresh process on the same isolated state, and read the value and effective state again. The changed value in a different instance establishes persistence and reload; an `applied` receipt alone does not. A saved-but-degraded provider selection is not usable-provider evidence.

For startup-timeout or cancellation changes, observe cleanup of any already spawned child before replacement. A failed or cancelled start does not itself prove that the child exited; graceful acknowledgement and forced escalation retain their separate meanings.

**Environment:** A runnable candidate and isolated writable roots; no account or model is needed for a setting unrelated to those resources. GUI editing claims require using the actual settings UI. For a visible caption-setting effect, also use [captions and output](#captions-and-output).

## Speech recognition and translation

**Entrypoints:** Self/Peer capture controls and `capture status`, `asr status`, provider/model status, `translation` controls, and [transcript/translation events](cli.md#events-and-privacy). [Audio tests and diagnosis](cli.md#peer-capture-and-audio-tests) observe input/VAD, not recognition. [Recognition/translation probes](cli.md#independent-component-checks) use isolated synthetic input and do not enter conversation output. [Manual text](cli.md#manual-text) bypasses capture/ASR and enters the Self translation/output path.

**Example — one speech result:** With the selected channel's capture and recognition ready and translation enabled, start observing events, feed a short known phrase through that channel's intended audio source, and follow its transcript and translation result. Stop capture and confirm it settles. This establishes that input-to-result connection, not delivery or display. For cancellation/recovery changes, interrupt the affected operation, observe settlement and absence of stale results, then send a fresh phrase to demonstrate recovery. Compose with [captions and output](#captions-and-output) when the promise includes subtitles.

**Environment:** A controlled microphone/loopback/process-audio source, ready ASR assets or an authorized provider, and a ready translation backend when used. Peer capture requires the [existing explicit consent](cli.md#peer-capture-and-audio-tests). A channel switch must preserve Self/Peer identity; stopping or replacing a source must not publish retired work. Missing devices, assets, or authorization are gaps, not successful speech evidence.

**Measurements:** [ASR benchmarks](../benchmarks/asr/README.md) own preparation, worker prerequisites, frozen inputs, scoring, and timing boundaries. Their resident decode and paced replay results are not end-to-end caption latency. Compare relevant inputs/configurations/environments; select runs for the decision, not a universal corpus, repetition count, or threshold.

## Captions and output

**Entrypoints:** GUI caption/output settings, CLI `overlay` and `osc` controls/status, [manual text](cli.md#manual-text), and the actual desktop overlay, SteamVR overlay, or VRChat chatbox. [Output ownership](architecture.md#output-and-context), [native overlay interfaces](../native/overlay/README.md), and [OSC documentation](vrchat-osc.md) define routing and destination boundaries. Desktop and VR captions use different renderers.

**Example — wrapping with prepared input:** With the intended caption destination ready, submit a known long manual-text input through the application and follow its turn's completed result to the visible caption. Inspect wrapping, language/content, clipping, and visibility at the changed size. If replacement/order is affected, revise or follow with another result and observe replacement and ordering rather than accumulated stale content. This establishes the selected Self output/rendering path; it does not establish microphone/ASR or the other renderer. A disconnected renderer preview establishes only rendering behavior.

When relevant to the change, vary short/long/short content, simultaneous Self/Peer results, or clear/hide followed by a fresh caption. Observe stable layout and channel separation, stale revisions staying superseded, and the new caption returning without resurrecting cleared content.

**Environment:** The actual desktop GUI for desktop claims; built native runtime, SteamVR, and headset for VR presentation; VRChat and configured OSC for chatbox presentation. A controlled OSC receiver can prove packet receipt but not the game's visible chatbox. Use prepared input only at the affected production boundary. Peer content must never reach the chatbox, destinations must remain independent, and expired/cleared captions must not reappear. For settings-triggered layout changes, compose with [settings and application management](#settings-and-application-management).

## Image translation

**Entrypoints:** GUI image-translation settings (also discoverable `photo.*` settings), [capture watcher](../src/puripuly_heart/core/photo/capture.py), [photo application owner](../src/puripuly_heart/app/services/photo_application.py), and the native photo surface's pull/hide interaction. This is a separate photo process, not the caption overlay. The [photo runtime boundary](../native/overlay/README.md#photo-runtime-boundary) owns preparation and rendering prerequisites; [photo lifecycle](../src/puripuly_heart/core/photo/lifecycle.py) owns authorization and requests.

**Example — translate a captured image:** With image translation enabled, a ready route, and a synthetic scene containing known text, take a new VRChat camera photo after startup and observe its wrist offer. Grip and pull it through the [native interaction](../native/overlay/src/photo/interaction.rs), then follow that photo/attempt through recognition/translation and inspect masking, typesetting, and the translated image result card in the headset. Correlate `[Photo]` diagnostic records by `photo_id`/`attempt_id`; `first_display` corroborates rendering but does not replace looking at the card. No upload should precede an authorized pull. This establishes the connected capture-to-result path for that route; a request-complete log alone does not establish readable placement or headset presentation. Keep the result screen in this area even when changing its layout.

Release the result card after translation settles and confirm it stays visible for 4 seconds before its existing 150 ms fade-out. Re-grabbing and placing it restarts the 4-second wait. A card released during preparation or translation must stay visible while that work is pending.

**Environment:** Windows, VRChat camera/log input, the photo model/runtime assets, native photo runtime, SteamVR/headset for interaction and presentation, and authorization for the selected provider. Use controlled image content; do not upload personal photos for routine probes. Missing VR/provider access limits the connected claim.

**Measurements:** [Offline rendering](../native/overlay/README.md#offline-rendering) and its [job schema/output implementation](../native/overlay/src/photo/offline.rs) exercise image processing and composition with supplied translation blocks, without provider requests or OpenVR. Compare the same image, blocks, assets, configuration, and hardware when measuring pixels or timing. Offline artifacts do not establish acquisition, recognition quality, live translation, or physical presentation.

Existing opt-in [model/reference experiments](../native/overlay/tests/photo_processing.rs), [processing comparisons](../native/overlay/src/photo/processing/tests.rs), and [typesetting probes](../native/overlay/src/photo/typeset/tests.rs) own their external fixture/model/ORT prerequisites. Run only relevant experiments and confirm the intended ignored case actually executed; a skipped case is not measurement evidence.

## Chatbot

**Entrypoints:** The actual [Puri chat view](../src/puripuly_heart/ui/views/agent_chat.py), backed by the [conversation owner](../src/puripuly_heart/app/services/agent_chat/owner.py) and its application tools. Submit in the Agent tab; tap a before/after change card to approve, use its Back action to decline, or send `/cancel` to cancel. Observe request/reply, selected application context, tool outcomes, approval/cancellation, and the affected application owner/surface. An assistant's claim of success is not evidence that its tool changed the app; debug-preview cards are not actual tool execution.

**Example — change a caption setting:** In an isolated app with a ready chatbot service, ask Puri for a concrete caption-setting change. Check that its proposed change fits the request and current context, approve it in the actual UI, and observe the terminal tool result and effective setting. Feed a known caption result and inspect the visible effect. This composes chatbot, [settings](#settings-and-application-management), and [captions](#captions-and-output). For approval/cancellation changes, reject or cancel before approval and verify no unapproved mutation; cancellation after a commit must not be mistaken for rollback.

**Environment:** Actual GUI plus authorized account/model and isolated Broker/Relay resources for live requests; the destination environment required by the requested effect. Keep credentials in secure application controls, not conversation text, and do not trigger support uploads without explicit user action.

**Measurements:** Follow the [chatbot benchmark instructions](../benchmarks/chatbot/README.md), [CLI help/results](../benchmarks/chatbot/__main__.py), and [calibration implementation](../benchmarks/chatbot/judging.py). Controlled smoke checks use synthetic responses/state and are not live-model evaluation. Live-model evaluation requires the documented isolated resources and access policy; preserve `blocked_calibration` and other blocked/calibration status rather than treating completed conversations as a calibrated pass. Neither mode proves actual GUI interaction or real application effects. Choose comparisons for the engineering question; do not duplicate the simulator's case inventories, commands, rubrics, or model policy here.

## Compose, maintain, report

Examples of selection, not additional required scenarios:

- **Settings persistence:** settings/application management, save, completed shutdown, and fresh-process reload.
- **Caption wrapping:** captions/output with prepared input that reaches the changed renderer.
- **Speech results missing from captions:** speech/translation plus captions/output, following the same channel and result across the handoff.
- **Puri changing caption settings:** chatbot approval/tool effect plus effective settings plus the visible caption result.

Update the affected guidance when behavior, interfaces, commands, prerequisites, or observation methods change. Keep shared setup and syntax with their existing owners rather than adding a second manual. Distinguish guidance drift, missing tools, product regressions, and unavailable environments; never weaken the expected outcome to hide a regression.

Report briefly: what ran, what was observed, and meaningful remaining gaps. Keep component/offline/simulator evidence separate from connected application and presentation evidence. No new report schema or recurring full sweep is required.
