# Issue 206: SELF source-first implementation evidence

Issue: <https://github.com/kapitalismho/PuriPuly-heart/issues/206>

## Acceptance scope

The reported latency/staleness problem occurred in **SteamVR**, not the desktop overlay. On 2026-10-03 the maintainer explicitly deferred safety testing to a later SteamVR session wearing the HMD:

> 안전 테스트는 나중에 스팀VR 환경에서 HMD 쓰고 또 하는 걸로 해줘. 지연 버그는 스팀 VR 환경에 있었다는 걸 명심해

At implementation completion, this record covered software/application behavior and sampled desktop presentation; the physical SteamVR criteria were not run and were user-deferred. A later [two-case worn-HMD session](issue-206-hmd-test-plan.md#short-physical-session-2026-10-03-utc) observed baseline and candidate short stable/head-locked captions normally, with no perceived difference. The subsequently authorized [resumed batch](issue-206-hmd-test-plan.md#resumed-batch-2026-10-03-utc) added 18 software/cleanup passes with continuous wearer observation of no anomaly, then stopped on a harness-only reconnect authentication mismatch. That failed case remains failed; unexecuted spatial, application and sustained scope and detailed native-stage/resource measurements remain unverified. Neither software results nor these short observations establishes that the historical SteamVR delay is fixed. No full physical acceptance, issue closure, merge, push or deployment is implied.

## Source and user-visible changes

Baseline: `8666bb57935b7c6da0c3c8aeaec9d116762d9f3d`. The implementation commit containing this record is the candidate for the execution below; subsequent repairs must preserve or refresh affected evidence. Remote `dev` at inspection was `991ef0bdf06d744ebb6e7aa6d8df5d4ee5f8ca46`, an ancestor of the baseline. Its four pre-existing local descendants are outside this implementation's change boundary.

- SELF source captions and otherwise eligible translation requests no longer wait for dashboard consumption. The existing output/UI owner admits bounded delivery work; admission is not a display acknowledgement.
- Meaningful visible active SELF source updates receive existing native stream-phase rendering protection without becoming semantic finals. Same-target updates do not restart exhausted retry budgets or deadlines.
- Source/translation identity, active-row protection, source preferences, sticky preview secondary text, late-result/expiry rules, merge/resume/grace, speculative reuse, PEER presentation, chatbox routing, and desktop priorities remain unchanged.
- UI destination overload/failure is explicit. Required history/error events are not silently coalesced as presentation state. Delivery authority rejects obsolete/duplicate callbacks; an older authorized history/error event cannot overwrite newer accepted dashboard presentation.

Ownership remains with `OutputRuntime`, `TranslationUiMessageQueue`, `OverlayPresenter`, `NativeRetryIntentProjection`, and the native presentation owner. The existing UI lifecycle now covers SELF/manual as well as PEER; no second publication/translation/retry owner, Python retry loop, wire phase, backend, or production profile was introduced. No suspected architecture drift beyond that documented lifecycle expansion was identified.

Bounds: SELF/manual 32 waiting events + one active; PEER eight waiting batches + one active, each at most 32 outstanding events; one writer per lane, five-second write timeout. Delivered PEER payloads are released, so there is no 32-event lifetime limit on a segmented turn. Including one consumer-queue event and one consumer-held event, the bound is 323 distinct delivery payloads. Exhaustion is destination-local `output_overload`; timeout, publication failure, retirement, replacement, and close have explicit receipts. `accepted_handoff` and `ui_queue_submitted` do not claim UI application or physical display.

## Criteria and observed results

| Criterion | Result | Evidence and limits |
| --- | --- | --- |
| C1 UI isolation | Passed software | `test_self_ui_isolation.py`: independent Gemini-shaped, scoped stable, and scoped final-only routes fill the capacity-one queue and barrier-stop the actual UI consumer. Real owners apply source and invoke the gated provider before UI release; translated state also applies before release. One provider call per route. |
| C2 bounded lifecycle | Passed software | `test_translation_ui_delivery.py` and output/UI suites: overload with required error events, timeout/exception, mixed SELF/manual/PEER delayed callbacks, history/error retention, duplicate rejection, source retirement/manual isolation, replacement, cancellation-before-entry, shutdown, and late callbacks. PEER same-parent flow delivers 72 events without a lifetime-count cutoff. |
| C3 provider semantics | Passed software | SELF owner and low-latency suites cover terminals, suffixes, duplicate finals, grace/resume, configuration authority and failures. Scoped pressure probe waits through unmodified grace while speculation runs, with no extra request. Final-only route has no caption before terminal; independent units do not acquire a local VAD identity. |
| C4 native accounting | Passed software | Actual Python projection feeds native reducer/retry accounting and a production native event loop over a local WebSocket. Stream deadline/count persist across revisions and exhaustion; final and translation-final transitions remain distinct. Complete native suite covers readiness, currentness, recovery, clear, resources, and spatial history. Fake OpenVR submission/test-renderer boundaries are not HMD evidence. |
| C5 shared window | Passed software | Active SELF / Peer-1 at t=10.1 / paced Peer-2 at t=11.1 / late SELF translation retains SELF ID, occupant, appearance and slot. Finalized eviction control rejects the displaced SELF translation. Default two slots and one-second PEER gate remain. Native spatial regression suite passed. |
| C6 delivery/expiry | Passed software | Three real-owner probes complete translation while bridge send is independently blocked, then emit monotonic scenes after release. Bridge tests exercise 20 active updates, latest-scene coalescing, send-time expiry, clear/OFF and reconnect without resurrection. No source-only dwell or render-ack gate. |
| C7 output parity | Passed software | Translation turn/output/channel suites cover speculation/reuse, counts, context/configuration, dual targets, partial failure, manual isolation, history and routing. No PEER chatbox output; no external paid provider requests were made. |
| C8 PEER/desktop | Passed scoped software and sampled actual pixels | Eight PEER scenarios per baseline/candidate: independent/scoped translated-first, original option, translation disabled, failure fallback. Actual desktop windows render owner-produced scenes; source/translation arbitration and white SELF/gold PEER styling preserved. No native retry cadence on desktop. |

Baseline reproduction observed an independent-result ingress stalled before source application/provider invocation behind a full UI queue; stable ingress already progressed. Separately, baseline active SELF source rendered with null freshness generation/episode. Those are software gaps, not a causal proof of physical HMD delay.

## Reproducible checks

Integrated Python command, after formatting:

```powershell
uv run --frozen pytest tests/core/test_self_ui_isolation.py tests/core/test_translation_ui_delivery.py tests/core/test_translation_output_projection_owner.py tests/core/test_self_translation_channel_owner.py tests/core/test_self_translation_low_latency.py tests/core/test_peer_translation_channel_owner.py tests/core/test_translation_turn_owner.py tests/core/runtime/test_output_runtime.py tests/ui/test_event_bridge.py tests/domain/test_domain_models.py tests/app/test_runtime_pipeline_composition.py tests/app/test_runtime_pipeline_custom_http.py tests/app/test_application_shutdown.py tests/core/test_overlay_active_freshness.py tests/core/test_overlay_presenter.py tests/core/test_overlay_bridge.py tests/ui/test_desktop_overlay_renderer.py -o addopts=-s -q --tb=short
```

Result: **608 passed**, 33 existing `asyncio.iscoroutinefunction` deprecation warnings, 13.75 s. Printed owner observations for all three routes included `source_application=true provider_calls=1 ui_released=false`, `translation_application=true ui_released=false`, and separately `provider_completed=true translation_application=true bridge_released=false`.

The independent checkpoint review of `8666bb57935b7c6da0c3c8aeaec9d116762d9f3d..16ea76a83cae6ce4319c4f8480eb89d839c69235` found one medium-severity lifecycle defect (F1): replacing a completed/failed UI bridge in the same loop turn as SELF writer creation could cancel the writer before coroutine entry, leaving a stranded writer reference. The finding was accepted and repaired with identity-checked task-completion ownership; cancellation-before-entry now releases the writer and starts retained current work. No task-per-event, capacity, timeout, rendering or provider policy changed.

The production `OutputRuntime.start_ui_event_bridge` regression failed before repair for both completed and failed previous bridges while waiting for actual history/dashboard consumption (`artifact://99`). After repair, both cases passed without using `wait_for_idle` as recovery. A failed-bridge cleanup fixture was corrected to the existing single-error rethrow contract. Repair verification command:

```powershell
uv run --extra dev pytest tests/core/test_translation_ui_delivery.py tests/core/test_self_ui_isolation.py tests/core/test_translation_output_projection_owner.py tests/core/runtime/test_output_runtime.py tests/ui/test_event_bridge.py tests/core/test_overlay_bridge.py -o addopts=-s -q --tb=short
```

Result: **177 passed**, 3.72 s (`artifact://104`), including all six owner pressure smokes. Ruff passed and Black left both changed files unchanged. The 608-test integrated run and raw desktop hashes above describe the initial checkpoint; this bounded repair changes only SELF writer completion cleanup and its regression tests. Native/desktop rendering, protocol, layout, translation scheduling and the deferred physical scope are unchanged; their prior evidence is retained, not relabeled as rerun.

Native commands:

```powershell
cargo test --locked --manifest-path native/overlay/Cargo.toml --target-dir C:/pph206-native -- --test-threads=1
cargo test --locked --manifest-path native/overlay/Cargo.toml --target-dir C:/pph206-native python_active_self_projection_drives_native_stream_and_final_accounting -- --nocapture
cargo test --locked --manifest-path native/overlay/Cargo.toml --target-dir C:/pph206-native production_owner_renders_python_active_self_without_renewing_exhausted_stream -- --nocapture
cargo build --locked --release --manifest-path native/overlay/Cargo.toml --bin PuriPulyHeartOverlay --target-dir C:/pph206-native
C:/pph206-native/release/PuriPulyHeartOverlay.exe --check-startup-contract
```

Native suite: **280 passed, one existing opt-in memory/performance probe ignored**. Both named cross-language scenarios passed. Release build passed. Startup-contract stdout reported app 2.7.0, bridge 15, execution r2/version 1, exclusive native presentation retry/version 1 and speaker identity/version 2; that probe retained stdout rather than the startup process return code.

Native executable identities (SHA256):

| Executed/built artifact | SHA256 |
| --- | --- |
| Release `PuriPulyHeartOverlay.exe` | `5a4c14864dd2c8662bcfdb77c1076812f6a58fa45ad998c56d66e2b3880184ee` |
| Library test executable | `c7992bff7e3479cf884692922f9f42f8366b82974ab47b6720dc5d7e58dcda51` |
| Runtime test executable | `3b0ce1bd6168125229eea25b4aa26864c8f13ecb65529dbfe883dcd8f67e8f66` |

Native product implementation is unchanged; Rust changes add behavior tests. No source-pinned baseline release binary was available, and an unrelated existing binary was not substituted for it. Production P05 remains 100 ms cadence, 500 ms deadline, at most four stream/five final opportunities, separate 2 s no-progress timeout, handoff off, API-only D3D11/OpenVR backend. This does not promise every opportunity occurs or measure physical latency.

## Actual application and desktop observations

Environment: Windows 10.0.22631 x64; Python 3.14.7; Flet/flet-desktop 1.0.0; AMD Radeon RX 7900 XTX, driver 32.0.31041.1004; reported desktop 2560×1440.

An isolated config was used for each baseline/candidate GUI-backed and headless host. Actual module CLI commands exercised `app status`, settings/current/choices/capabilities, translation off, desktop target selection, overlay on/effective state, synthetic manual source submission, overlay off and ordered shutdown. Effective SELF/PEER capture remained off; no consent acceptance, private recording, paid API run, existing-user-GUI operation or VRChat launch occurred.

Owned Win32 `PrintWindow` captures were inspected separately from JSON state. The real GUI dashboard and production desktop overlay showed synthetic manual source. Real `FletDesktopRendererWindow` windows exercised active source, delayed translation/sticky secondary, mixed SELF/PEER, final arbitration, clear/recovery, and 120 synthetic updates requested at 50 ms cadence (actual burst duration ~9.45 s). Twenty-one sampled renderer scenes and the manual source-only overlay had byte-identical baseline/candidate captures.

Exposure: baseline/candidate GUI 178.4/256.418 s; headless 143.8/61.913 s; extended renderer ~24.28/24.32 s; PEER renderer ~33.87/33.36 s; sampled scenes used 2 s dwell. These are sampled layout observations, not continuous flicker, physical pixel freshness, live-provider performance, VR reanchor/resource-cost or long-session certification.

Direct pointer interaction with the unrelated history-tab control was **not successfully exercised**: a targeted message did not change the tab; a later click was withheld when the target was not the owned GUI. No pointer-interaction pass is claimed. Application controls were exercised through the real CLI; this change adds no GUI control.

OpenVR reported runtime installed and HMD present; SteamVR/VRChat processes were initially absent. Detection is not observation, device/firmware/connection identity, or proof of freshness. Physical SteamVR work was not started after the user's deferral.

## Evidence retention and follow-up

Session evidence: integrated Python output `artifact://78`; native suite `artifact://61`; native production-loop scenario `artifact://56`; native release build `artifact://58`; owner reports `agent://ui-isolation`, `local://active-self-freshness-evidence.md`, and `agent://surface-validation`. These are local session references, not public hosting.

Surface command JSON/stdout/stderr, execution hashes, scene snapshots, owned-window PNGs, scope amendment and probe source record were retained in the local archive:

`C:/Users/salee/AppData/Local/Temp/puripuly-206-surface-qfizsrcq-evidence.zip`

SHA256: `2bb59ca44fd6fa44a80ed05a277cea96c576cd1f093ad26ab1392122b5096e8d`.

The local archive is not committed or uploaded. The tables, commands, identities and permanent behavior regressions above retain the portable result; raw images require that archive. Temporary executable probes/configs/baseline export were removed and owned GUI/headless/renderer processes closed.

Later worn-HMD SteamVR acceptance must compare a retained baseline/candidate on the affected setup and record source/native identities, effective protocol/profile, device/driver/connection, exposure and observation method. Cover delayed translation, stable bursts, resumed speech, mixed channels, head/spatial lock, clear/OFF and recovery/reconnect. Observe flicker, reanchors, old source/translation pairs and bounded rendering/resource work. Keep source availability, Presenter application, bridge/native readiness/submission, translation readiness and actual HMD observation separate. SELF E2E still ends at first successful chatbox page send, not overlay display. No universal-device or 4–6 h stability conclusion follows from the short software/desktop runs.

The subsequent maintainer request is to prepare the complete test tooling now
and run the physical session together after the wearer returns. The
[SteamVR worn-HMD plan and operator commands](issue-206-hmd-test-plan.md) record
the pinned source/native kit, offline scenario/stop evidence, explicit
wearer-ready gate, observation procedure and later-session prerequisites.
This preparation does not supersede or pass the physical criteria above.
