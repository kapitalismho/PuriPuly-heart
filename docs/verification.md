# Product-area verification

Use observed results in the running application as the main completion evidence for product behavior. Choose and compose the smallest relevant path below; examples are adaptable starting points, not a mandatory checklist or an all-area sweep. Code tests support repeatable regression protection. Narrow internal changes may need only narrower evidence.

## Shared entrypoints and evidence

- The [CLI guide](cli.md) owns source/installed entrypoints, instance selection, lifecycle, syntax, operation results, events, and privacy. Discover the running instance's `capabilities`, `settings choices`, and command help instead of copying a command catalog.
- Use the [architecture map](architecture.md) for owners/connections, the [development guide](../.github/CONTRIBUTING.md) for build prerequisites, and [runtime layout](../src/puripuly_heart/runtime_layout.py) for resource/writable roots. Isolate settings, secrets, discovery, providers, telemetry, and outputs; a separate settings file alone is insufficient. Use synthetic inputs and controlled local resources, stop owned instances, and clean up only probe-created resources.
- Identify the checkout/import or rebuilt executable and selected instance. Restart source hosts after code changes. Read terminal operation results and the relevant owner's effective state: command admission does not establish completed behavior, saved settings, restart persistence, or presentation.
- Follow the **same concrete input/result** across affected connections, using channel/turn, operation, photo/attempt, or conversation identities. Separate passing component checks do not establish a connected flow. Prepared inputs may bypass unchanged upstream stages when they enter the affected production path.
- Presentation and interaction claims require the actual GUI, desktop overlay, game, or headset surface. Producer logs, renderer readiness, component probes, offline artifacts, and simulator results establish only their own scopes. Use [diagnostics](cli.md#retained-diagnostics-and-delivery-health) to investigate missing results, not to infer success from absent errors. Unavailable essential observations remain explicit gaps.

## Settings and application management

**Inputs and observations:** GUI settings/application controls; CLI `app`, `settings current`, `settings choices`, and `settings apply`; terminal operations and owner status. [Settings/runtime semantics](cli.md#settings-and-runtime-state) distinguish committed intent from applied resources. Use [account controls](cli.md#credentials-authorization-and-consent) and model/provider readiness when relevant.

**Example — save and restart:** In an isolated host, record the instance and settings revision. Apply a supported harmless change through the application; inspect its terminal result, committed value, and relevant effective state. Stop to completion, start a fresh process on the same state, and read the value and owner state again. A different instance retaining the change establishes reload; an `applied` receipt alone does not. Saved-but-degraded provider selection is not usable-provider evidence.

**Expectations and limits:** Failed/cancelled startup must settle already spawned children before replacement. Authentication retry must preserve existing connections on failure and prevent cancelled attempts from activating; use the focused [ChatGPT account checks](../tests/app/test_chatgpt_account.py) for detailed retry boundaries rather than repeating an OAuth incident recipe here.

**Prerequisites:** Runnable candidate and isolated writable roots. No account/model is needed for an unrelated setting. GUI editing claims require the actual settings UI; visible caption effects also need [captions/output](#captions-and-output).

## Speech recognition and translation

**Inputs and observations:** Self/Peer capture controls, `capture status`, `asr status`, provider/model status, translation controls, and [transcript/translation events](cli.md#events-and-privacy). [Audio tests](cli.md#peer-capture-and-audio-tests) establish input/VAD, not recognition. [Component probes](cli.md#independent-component-checks) do not enter conversation output. [Manual text](cli.md#manual-text) enters Self translation/output without capture/ASR.

**Example — one speech result:** With capture/recognition ready and translation enabled, observe events and feed a known short phrase through the intended channel's audio source. Follow its transcript and translation, then stop capture and confirm settlement. Compose with [captions/output](#captions-and-output) to establish delivery/display. For interruption changes, cancel or replace the affected operation, check that retired work cannot publish, then send a fresh phrase to demonstrate recovery.

**Prerequisites and limits:** Controlled microphone/loopback/process audio, ready ASR assets or authorized provider, and a ready translation backend when used. Peer capture requires [explicit consent](cli.md#peer-capture-and-audio-tests). Preserve Self/Peer identity through switching and recovery. Missing devices, assets, or authorization are gaps.

**Measurements:** [ASR benchmarks](../benchmarks/asr/README.md) own input preparation, worker prerequisites, scoring, and timing boundaries. Resident decode and paced replay are not end-to-end caption latency. Compare relevant inputs/configurations/environments; no universal corpus, repetition count, or threshold is required.

## Captions and output

**Inputs and observations:** Caption/output settings, CLI `overlay` and `osc` controls/status, [manual text](cli.md#manual-text), and the actual desktop overlay, SteamVR overlay, or VRChat chatbox. [Output ownership](architecture.md#output-and-context), [native interfaces](../native/overlay/README.md), and [OSC documentation](vrchat-osc.md) define destination boundaries.

**Example — wrapping with prepared input:** With the intended destination ready, submit known long manual text through the application. Follow its turn's completed result to the visible caption; inspect content, wrapping, clipping, and visibility at the changed size. If order/replacement is affected, follow with another result or clear and send a fresh caption. Observe the new content without stale revisions or cleared content returning.

**Expectations and limits:** Self/Peer remain separate; Peer never reaches the chatbox; destination failures remain independent. An open Self turn keeps its current caption until source closure starts idle expiry; final content replaces it under the same identity, and retired results cannot revive it. [Caption policy checks](../tests/core/test_self_caption_policy.py) protect detailed timing/order boundaries. For OSC discovery changes, use a controlled OSCQuery endpoint and `osc status` to observe discovery, retargeting, fallback, and recovery; [runtime checks](../tests/core/runtime/test_oscquery_runtime.py) cover interleavings. Local routing or packet receipt does not establish the game's visible chatbox.

**Prerequisites:** Actual desktop GUI for desktop claims; built native runtime plus SteamVR/headset for VR presentation; VRChat and configured OSC for game presentation. Desktop and VR renderers need their own observations. Prepared Self text does not establish microphone/ASR. Compose with [settings](#settings-and-application-management) for settings-triggered layout effects.

## Image translation

**Inputs and observations:** GUI image-translation settings (`photo.*`), [capture watcher](../src/puripuly_heart/core/photo/capture.py), [application owner](../src/puripuly_heart/app/services/photo_application.py), and native pull/hide interaction. The [photo runtime](../native/overlay/README.md#photo-runtime-boundary) is separate from caption overlays; [photo lifecycle](../src/puripuly_heart/core/photo/lifecycle.py) owns authorization and requests.

**Example — translate a captured image:** With image translation enabled and a ready route, take a new VRChat camera photo of synthetic known text after startup. Observe its wrist offer, grip/pull through the [native interaction](../native/overlay/src/photo/interaction.rs), and follow the same photo/attempt through recognition/translation to its headset result card. Inspect masking, typesetting, and readable placement. No upload may precede the authorized pull. `[Photo]` identities and `first_display` corroborate the result but do not replace viewing it.

**Expectations and limits:** A released card remains visible while work is pending; after settlement it waits 4 seconds before the existing 150 ms fade, with re-grab/place restarting that wait. Capture selects the newest eligible photo without duplicate offers or replaying startup/disabled-period photos; notification recovery and log rotation must preserve that selection. Quiet Windows watching uses notifications rather than repeated scans, and failures remain visible. [Capture checks](../tests/core/test_photo_capture.py) and [notification checks](../tests/core/test_photo_directory_notifications.py) cover rotation, duplicate, and recovery variants without requiring an app scenario for each.

**Prerequisites:** Windows, controlled camera/log input, photo model/runtime assets, native photo runtime, SteamVR/headset, and authorization for the selected provider. Never use personal photos for routine probes. Controlled notifications or application offers do not establish physical presentation; missing VR/provider access limits the connected claim.

**Measurements:** [Offline rendering](../native/overlay/README.md#offline-rendering) and its [job schema](../native/overlay/src/photo/offline.rs) exercise supplied images/translation blocks without provider requests or OpenVR. Compare the same inputs, assets, configuration, and hardware. Existing opt-in [model/reference](../native/overlay/tests/photo_processing.rs), [processing](../native/overlay/src/photo/processing/tests.rs), and [typesetting](../native/overlay/src/photo/typeset/tests.rs) probes own external fixture/model/ORT prerequisites; confirm the intended ignored case executed. These artifacts do not establish acquisition, recognition quality, live translation, or headset display.

## Chatbot

**Inputs and observations:** The actual [Puri chat view](../src/puripuly_heart/ui/views/agent_chat.py), [conversation owner](../src/puripuly_heart/app/services/agent_chat/owner.py), and application tools. Submit in the Agent tab; tap a change card to approve, use Back to decline, or send `/cancel`. Observe context use, request/reply, tool results, and the affected owner/surface. An assistant's success claim or debug-preview card is not an application effect.

**Example — change a caption setting:** With an isolated app and ready chatbot service, ask Puri for a concrete caption-setting change. Check the proposal against the request/current context, approve it in the UI, and inspect the terminal tool result and effective setting. Feed known caption content and observe its visible effect. This composes chatbot, [settings](#settings-and-application-management), and [captions/output](#captions-and-output). When approval/cancellation is affected, reject or cancel before approval and verify no unapproved mutation; cancellation after commit is not rollback.

**Prerequisites and limits:** Actual GUI, authorized account/model, isolated Broker/Relay resources, and the requested effect's destination environment. Keep credentials in secure app controls, not chat; support uploads require explicit user action.

**Measurements:** [Chatbot benchmark instructions](../benchmarks/chatbot/README.md), [CLI/results](../benchmarks/chatbot/__main__.py), and [calibration](../benchmarks/chatbot/judging.py) own simulator use. Controlled smoke checks use synthetic responses/state, not live-model evaluation. Live evaluation requires its documented access/resources and preserves `blocked_calibration` and other blocked statuses. Neither mode proves GUI interaction or real app effects. Do not duplicate simulator cases, commands, rubrics, or model policy here.

## Build and packaging checks

Use the [Native Release workflow](../.github/workflows/release-native.yml), [artifact builder](../scripts/ci/build-native-release-artifacts.ps1), and [release-evidence checks](../tests/release_evidence/) for packaging changes. Native Release is explicitly dispatched on `dev` for an existing version tag; it builds the validated tag commit and stages a draft, not a published release. Publication remains a separate maintainer action after staging completes.

Choose affected build/startup/metadata checks in isolated short target directories. Compiler reuse must still assemble fresh final artifacts. Synthetic installer compression/extraction establishes only that boundary; extensive installer lifecycle verification remains opt-in for relevant changes. Hosted cache transfer, complete installer qualification, and speed/size claims require their own observed evidence, not local reuse or a component smoke pass.

## Compose, maintain, report

- Settings persistence: apply/save, completed shutdown, fresh-process reload.
- Caption wrapping: prepared input through the changed production renderer.
- Missing speech captions: follow the same channel/result from speech/translation to visible output.
- Puri changing captions: approval/tool effect, effective setting, visible caption result.
- Prompt/resource packaging: use the same synthetic manual text in source/rebuilt installed layouts, observing the assembled prompt at a controlled provider and the completed result. Existing prompt/resource contracts cover templates, language rules, `PURIPULY_HEART_PROMPTS_DIR`, and user overrides. The [HTTP extension example](http-extensions/examples/mymemory.json) supports controlled request/response checks; installed examples and user discovery remain distinct.

Update reusable paths when their entrypoints, inputs, prerequisites, expected observations, or evidence boundaries change. Restoring documented behavior does not by itself need a new example. Keep setup and syntax with their existing owners.

Report what ran, what was observed, and meaningful gaps. Distinguish product regressions from test/tool or guidance drift and unavailable environments; never weaken expected behavior to hide a failure. Keep component/offline/simulator evidence separate from connected application and presentation evidence.
