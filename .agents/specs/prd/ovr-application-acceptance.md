# OVR-APPLICATION-ACCEPTANCE

## Status and authority

**APPLICATION SOURCE CANDIDATE IMPLEMENTED / FULL OA ACCEPTANCE NOT YET ESTABLISHED / PRODUCTION CUTOVER NOT PERFORMED** for #148.

| Record | Value |
| --- | --- |
| Authority | #148, approved #146, `OVR-CONTRACT-1 r1`, `ARCHITECTURE.md` |
| Starting/characterized source | `97e211047db355b226e54a28790222d00cf95a19` on `ovr-0-vr-overlay-reliability-program-cross-envir`; upstream `origin/ovr-0-vr-overlay-reliability-program-cross-envir`, 0 ahead / 0 behind, clean at characterization start |
| Parent implementation baseline | `4e967df9d03649106faa8348c3ec611009529ffe`; only contract documentation differed from that parent at characterization |
| Contract consumed | `OVR-CONTRACT-1 r1`, DESIGN-FROZEN 2026-09-09, SHA256 `115e15b9c577c421ca6c86980c4c99b956ad4a336595cd864097e8af12e4416e`, G-C open |
| Audio revision consumed | No production Audio seam revision. #135's upstream LISTEN recognition admission and #144's SELF speech-origin caller migration, after #143's shared extraction, were not landed at the characterized HEAD. Under r1 §7, #148 owns the parent-output admission and application receipt seam without waiting for full Audio convergence. If an equivalent shared output seam lands first, #148 must consume its exact revision. No Audio landing is a blanket #148 prerequisite. |
| Current source protocol | Python `OVERLAY_CONTRACT_VERSION = 6`; native `EXPECTED_CONTRACT_VERSION = 6` |
| Runtime used | Python 3.12.10 via `uv run --no-project python`; websockets 16.1.1 |
| Source candidate identity | Current working-tree source only. No Director commit was performed, so there is no candidate commit SHA. Worker policy prevented Git mutation; this is not attributed to a user prohibition. |
| Deployed pair | Not deployed. No Python/native artifact hashes, authenticated runtime-instance receipt, or matched-pair release record exists. |

## Exact changed file list

Read-only `git status --short` at receipt update reported 0 staged, 25 unstaged, and 1 untracked path:

```text
M  ARCHITECTURE.md
M  pyproject.toml
M  src/puripuly_heart/core/orchestrator/peer_translation_channel.py
M  src/puripuly_heart/core/orchestrator/self_translation_channel.py
M  src/puripuly_heart/core/orchestrator/translation_output_projection.py
M  src/puripuly_heart/core/orchestrator/translation_channel_callbacks.py
M  src/puripuly_heart/core/orchestrator/translation_request.py
M  src/puripuly_heart/core/orchestrator/translation_turn.py
M  src/puripuly_heart/core/overlay/bridge.py
M  src/puripuly_heart/core/overlay/presenter.py
M  src/puripuly_heart/core/overlay/process.py
M  src/puripuly_heart/core/overlay/sink.py
M  src/puripuly_heart/core/runtime/output.py
M  tests/app/test_overlay_generation_start_owner.py
M  tests/app/test_overlay_process_manager.py
M  tests/app/test_overlay_translation_enabled_sync.py
M  tests/core/runtime/test_output_runtime.py
M  tests/core/test_dual_target_translation_lifecycle.py
M  tests/core/test_output_owner_wiring.py
M  tests/core/test_overlay_bridge.py
M  tests/core/test_overlay_presenter.py
M  tests/core/test_translation_output_streaming.py
M  tests/core/test_translation_turn_owner.py
M  tests/helpers/overlay_refresh_trace.py
M  uv.lock
?? .agents/specs/prd/ovr-application-acceptance.md
```

## Characterized before-map

```text
TranslationTurnLifecycleOwner
  _run_parent -> predecessor.semantic_done_event       semantic ordering / next LLM
  _execute_child -> mark semantic done
                 -> predecessor.closed_event           output ordering
                 -> TranslationChannelOwnerCallbacks
                 -> TranslationOutputProjectionOwner.project_translation_result
                     SELF: UI -> overlay final -> close -> allowed chatbox
                     peer: overlay final -> close -> UI -> chatbox denial
                 -> OutputRuntime.publish_overlay_event
                     _overlay_delivery_lock: identity + task registration
                     await per-event task
                 -> OverlayPresenter.emit
                     _ownership_transition_lock: reducer apply + scene generation
                     await OverlayBridge.replace_snapshot
                 -> OverlayBridge.replace_snapshot
                     _snapshot_lock: store scene + sequential websocket.send
                 -> native BridgeClient / NativePresentationOwner / renderer / OpenVR
```

At the characterized baseline, semantic completion released before output submission, so a later LLM could progress. Parent `closed_event`, parent completion, projection completion, and later route steps still waited for the socket. `OutputRuntime` created a task but immediately awaited it. `OverlayPresenter` committed state before awaiting the bridge without a local application receipt. `OverlayBridge` stored and remotely sent under the same lock.

## Retained baseline engineering probe

The characterization probe imported `compose_translation_test_harness` and constructed the real `TranslationTurnLifecycleOwner`, `TranslationOutputProjectionOwner`, `OutputRuntime`, `OverlayPresenter`, and `OverlayBridge`. External dependencies were a deterministic LLM, `RecordingOscQueue`, and an in-memory websocket connection passed through `OverlayBridge._handle_connection`. The connection authenticated normally, accepted revision 0, and blocked its first live `send()` behind an `asyncio.Event`. `FakeClock` and a controlled presenter sleep barrier drove expiry. Refresh bursts were disabled.

The load case admitted 12 peer turns through `handle_peer_transcript_final_for_test`. Cancellation was injected while a second event waited for the presenter ownership lock and after the first event reached the bridge send. Expiry advanced the clock from 300.0 to 308.0 while revision 1 was stalled. The removed temporary probe command was:

```text
PYTHONPATH=src;. uv run --no-project python %TEMP%\ovr_acceptance_probe.py
```

Retained observations from that baseline run:

| Stage | Characterized observation |
| --- | --- |
| Local state apply | First live bridge send started at 3.505 ms, after presenter and bridge had stored revision 1. |
| Ingress caller completion | All 12 peer ingress calls returned by 3.028 ms. |
| Semantic done | All 12 provider results returned by 4.293 ms; all 12 active parents had `semantic_done_event` set while the first send remained blocked. |
| Caller/output completion | No first or last parent closed while stalled. Twelve parent tasks and twelve child tasks remained active. |
| Transport completion | Barrier released at 4.348 ms; the injected fake `send()` returned at 4.494 ms. This was socket-boundary return, not native/HMD acceptance. |
| Output terminal | All parents became terminal at 6.779 ms after release; twelve live snapshots were written. |
| Other destinations | UI queue contained 12 entries while stalled and 24 after release. A separate allowed SELF chatbox publication returned `published` and enqueued one message during the stall. |
| Pending growth at load 12 | Baseline pending tasks 1; stalled pending tasks 27, growth +26. State contained 12 active parents, 12 parent tasks, 12 child tasks, one OutputRuntime delivery task/in-flight identity, and one presenter expiry task. |
| Cancel before apply | Event 2 cancelled while waiting for the real presenter lock did not apply and raised `CancelledError`; event 1 remained in flight. |
| Cancel during write | After event 1 applied to presenter and bridge, cancelling its caller cancelled the fake send. State stayed applied, OutputRuntime retained no delivered identity, and retry of the identical event returned `published` without another transport send. The baseline could not distinguish applied from write-ambiguous. |
| Concurrent expiry/age | At age 8.0 s presenter had committed empty revision 2 while bridge replay still held revision 1 with content. After release, revisions 1 and 2 crossed `send()` at age 8.0 s in order. The baseline had no send-time age revalidation. |

These timings are separate deterministic in-process observations, not performance limits or kernel-buffer measurements. The probe did not characterize a real peer that stopped reading, OS/kernel staging bytes, native parsing, OpenVR, GPU work, or physical HMD observation.

## Implemented after-map candidate

```text
translation parent/generation/order/targets
  -> projection supplies parent output scope and conservative retained bytes
  -> OutputRuntime bounded parent admission
       per origin/destination: 1 active + 8 unsent
       per parent: 1 MiB; per scope: 9 MiB
       manual at cap: reject incoming output_overload
       speech at cap: evict oldest unsent output_overload
  -> OverlayPresenter one atomic reducer transaction
       commit reducer state + scene/intent or return explicit not-applied receipt
       64 entries; 1 MiB/event; 16 MiB aggregate; bounded identities/frontiers
  -> parent output obligation closes at local application receipt

OverlayBridge independently owns delivery
  current scene + active write + one successor
  one writer; latest-scene coalescing; bounded keyed controls
  1 MiB scene and at most three retained scene copies
  scene write 5 s; control write 1 s; close 1 s then abort
  cancellation/error/timeout => ambiguous receipt + connection epoch retirement
  reconnect => send-time-revalidated current scene, never FIFO history
```

`OverlayPresenter` does not perform websocket I/O. `OverlayBridge.replace_snapshot` performs bounded serialization and local mailbox admission; its writer owns remote delivery and delivery disposition.

## Selected bounds in the candidate

| Resource | Candidate bound/policy |
| --- | --- |
| Parent output | 8 unsent + 1 active per origin/destination |
| Parent retained payload | 1 MiB |
| Origin/destination retained payload | 9 MiB |
| Local reducer | One atomic active transaction; no presenter event-task queue |
| Presenter state | 64 entries, 1 MiB/event, 16 MiB aggregate |
| Completed application identity | 4096 globally retained publication identities; exact merged retired sequence ranges preserve holes; at most 4096 admitted adapter namespaces; generation retirement removes obsolete managed scopes |
| Bridge scenes | Current + active + successor, each at most 1 MiB, aggregate at most 3 MiB |
| Bridge controls | Eight keyed slots, each at most 4 KiB |
| Reverse diagnostics | 128 entries, overwrite/drop-count policy, 4 KiB per entry |
| Reverse correctness controls | Keyed bounded storage with producer backpressure; diagnostics cannot displace controls |
| Websocket | `compression=None`, `max_size=1 MiB`, `write_limit=64 KiB`, `close_timeout=1 s` |
| Write/teardown | Scene 5 s, control 1 s, close 1 s; ambiguous write retires epoch |

## Exact verification commands and selectors

Final fast application verification command:

```text
uv lock --check --offline && uv run --no-project python -m pytest -vv tests/core/test_overlay_bridge.py tests/core/runtime/test_output_runtime.py tests/core/test_output_owner_wiring.py tests/core/test_translation_turn_owner.py tests/core/test_translation_output_streaming.py tests/core/test_dual_target_translation_lifecycle.py tests/core/test_overlay_presenter.py tests/core/test_self_translation_channel_owner.py tests/core/test_peer_translation_channel_owner.py tests/app/test_overlay_generation_start_owner.py tests/app/test_overlay_translation_enabled_sync.py tests/app/test_overlay_process_manager.py::test_process_reverse_queue_bounds_diagnostics_and_backpressures_controls
```

Observed result: `365 passed in 3.87s`.

Focused repair regression command:

```text
uv run --no-project python -m pytest -vv tests/core/test_overlay_bridge.py::test_overlay_bridge_unresolved_transport_caps_one_epoch_and_rejects_replacement tests/core/test_overlay_bridge.py::test_overlay_bridge_reserves_shutdown_and_coalesces_control_overflow_retirement tests/core/test_overlay_bridge.py::test_overlay_bridge_stop_closes_ingress_and_terminalizes_pending_scene tests/core/test_overlay_bridge.py::test_overlay_bridge_revalidates_successor_age_after_stalled_send tests/core/test_overlay_bridge.py::test_overlay_bridge_real_socket_stopped_reader_stops_bounded_and_truthfully tests/core/runtime/test_output_runtime.py::test_output_runtime_live_parent_hole_survives_newer_overload_terminal tests/core/runtime/test_output_runtime.py::test_output_runtime_retired_publication_ranges_preserve_sequence_holes tests/core/runtime/test_output_runtime.py::test_output_runtime_bounds_global_publication_and_adapter_namespace_identity tests/core/runtime/test_output_runtime.py::test_output_runtime_applies_parent_payload_limit_to_non_overlay_destinations tests/core/test_dual_target_translation_lifecycle.py::test_two_dual_target_parents_complete_secondary_first_through_stalled_bridge tests/core/test_output_owner_wiring.py::test_actual_owner_chain_completes_twelve_parents_while_bridge_socket_is_stalled tests/core/test_translation_turn_owner.py::test_self_speech_reset_cancels_speech_without_cancelling_manual_parent
```

Observed result: `12 passed in 2.02s`.

Exact changed-file Ruff command:

```text
uv run --no-project python -m ruff check src/puripuly_heart/core/overlay/bridge.py src/puripuly_heart/core/runtime/output.py src/puripuly_heart/core/orchestrator/translation_output_projection.py src/puripuly_heart/core/orchestrator/translation_turn.py src/puripuly_heart/core/orchestrator/self_translation_channel.py src/puripuly_heart/core/orchestrator/peer_translation_channel.py src/puripuly_heart/core/orchestrator/translation_channel_callbacks.py tests/core/test_overlay_bridge.py tests/core/runtime/test_output_runtime.py tests/core/test_output_owner_wiring.py tests/core/test_dual_target_translation_lifecycle.py tests/core/test_translation_turn_owner.py
```

Observed result: `All checks passed!`.

Exact final formatting command:

```text
uv run --no-project python -m black src/puripuly_heart/core/orchestrator/peer_translation_channel.py src/puripuly_heart/core/orchestrator/translation_output_projection.py
```

Observed result: two files unchanged. The preceding changed-file formatting pass reformatted ten files and left two unchanged.

### Repair evidence

- Unresolved transport ownership is capped to one connection epoch and one send/close task pair. A replacement connection is rejected while that epoch remains unresolved, and `stop()` reports the outstanding operations once rather than allocating repeated close work.
- Shutdown owns a reserved terminal control path. Correctness controls precede scene replay/data; retirement is coalesced to one task under a 200-attempt overflow burst.
- Ingress closes synchronously when stop begins. Pending scene admission becomes an explicit `delivery_rejected` receipt, and replacements during/after stop are rejected without mutating the accepted scene.
- Send-time age pruning now removes the expired block together with its native fresh-render target and quiet-tail episode.
- Publication retirement uses exact ranges rather than a maximum frontier. The 4100-event hole regression admits missing sequence 2 after sequences 1 and 3–4100 while still rejecting the retired duplicate sequence 1.
- The 4100-namespace regression holds global publication identity at 4096 and rejects identities beyond the namespace capacity instead of growing lifetime sets/maps.
- Parent admission and bytes apply independently to overlay, UI, and chatbox. A 512 KiB admitted non-overlay parent plus a 512 KiB+1 output is terminally rejected and releases all destination reservations.
- Two dual-target parents complete secondary-first through a stalled bridge. Each parent produces the secondary-only chatbox update followed by the complete `(0, 1)` update, then releases every destination reservation while transport remains stalled.
- TALK OFF cancellation is scoped to SELF speech turns. The focused selector proves an admitted manual SELF parent survives and completes.

### Integrated stalled-send and real-socket evidence

`tests/core/test_output_owner_wiring.py::test_actual_owner_chain_completes_twelve_parents_while_bridge_socket_is_stalled` now drives the actual translation owners, projection, `OutputRuntime`, `OverlayPresenter`, and `OverlayBridge`. It alternates twelve manual/peer parents and asserts twelve distinct parent IDs, twelve distinct blocks, UI queue size 24, chatbox size 6, zero output reservations, and zero completed bridge sends while the bridge connection is stalled. The independent validation wave observed completion in 7.118 ms; the retained regression enforces the behavioral counts and non-transport completion rather than a platform-sensitive wall-clock number.

`tests/core/test_overlay_bridge.py::test_overlay_bridge_real_socket_stopped_reader_stops_bounded_and_truthfully` uses an actual loopback websocket, a 4 KiB receive buffer, paused client reads, and repeated 900,000-character scenes until the server write is active. Its exact standalone command passed in 1.42 s. A direct retained smoke observation blocked at revision 6; `stop()` returned success in 1012.717 ms with zero unresolved transport tasks, zero authenticated connections, and `stopped=True`. The resistant-send/close selector separately proves that a genuinely unresolved operation causes a bounded, truthful `ExceptionGroup`.

## OA01–OA10 ledger

No row below establishes full OA acceptance. Full OA acceptance still awaits the remaining exact criteria and independent complete-goal review required by #148.

| OA | Candidate evidence | Disposition |
| --- | --- | --- |
| OA01 | Integrated production-owner stalled-send selector and real stopped-reader loopback websocket pass with bounded truthful stop | **Application subcriterion covered; not full OA acceptance** |
| OA02 | Deterministic cancellation records ambiguity, retires one epoch, rejects replacement while unresolved, and replays current state only after resolution | **Partial candidate evidence; native/HMD acceptance absent** |
| OA03 | Provisional/final/clear coverage, one active plus one successor scene, reserved shutdown control, and 200-overflow coalescing pass | **Application pressure evidence covered; full cross-process invalidation criterion not run** |
| OA04 | Same-text identity, exact sequence-hole retirement, late-currentness, and bounded 4100-namespace regressions pass | **Application subcriterion covered; independent review pending** |
| OA05 | Send-time expiry removes expired blocks and native intent references; reconnect replay remains current | **Application subcriterion covered; native lease/epoch acceptance not implemented** |
| OA06 | Focused TALK OFF regression proves speech-only cancellation preserves an admitted manual SELF parent | **Application origin-scope correction covered; full TALK/LISTEN/Caption matrix not exercised** |
| OA07 | Reverse diagnostics remain bounded; correctness controls are bounded/backpressured; shutdown is reserved and control overflow coalesces retirement | **Application subcriterion covered; full cross-process flood lifecycle not reviewed** |
| OA08 | Destination-local failure isolation and parent terminalization remain covered, including cancellation release before peer fallback output | **Partial candidate evidence; full failure/ASR-rerun criterion not run** |
| OA09 | One-active/eight-waiting admission, 1 MiB parent cap, 9 MiB scope cap, non-overlay destination accounting, and manual/speech cap dispositions pass | **Application subcriterion covered; upstream Audio admission remains external; full OA09 not accepted** |
| OA10 | Protocol-6 application/desktop baseline remains green | **Protocol 7 and matched packaged pair remain blocked** |

## Native-dependent and other remaining gaps

The following are not claimed by this application candidate:

- protocol 7 and `execution_contract {version: 1, revision: "r1"}` negotiation;
- native three-second validity lease and challenge/response renewal;
- native scoped invalidation watermarks and current-status receipts;
- bounded native due/render/GPU/OpenVR attempt progress and #149 supervisor recovery;
- packaged desktop/native resources and mixed-version fail-fast evidence;
- deployed Python/native artifact hashes and authenticated runtime-instance receipts;
- physical native/OpenVR/HMD observation and cross-OS stopped-reader staging measurements;
- the full TALK/LISTEN/Caption scope matrix beyond the repaired TALK OFF/manual SELF boundary;
- independent complete-goal review.

Native, GPU, OpenVR, and HMD observations are valuable downstream/native evidence, but **HMD execution is not a prerequisite for completing the software-only #148 application source outcome**. Lack of HMD evidence must be recorded as `not_observed`/`not_observable`; it must not be used to withhold software source completion once the application criteria and review are satisfied. Production activation of a matched native/application pair remains a separate release/cutover decision.

## Pair/cutover and rollback

The current working tree is an uncommitted source candidate, not a deployed pair. A Director may later create a commit and artifact identities. Production cutover requires the appropriate exact matched Python/native/resources tuple and capability/version evidence; this receipt does not assert those exist.

Rollback is paired: stop ingress; retire output, connection, process, and device epochs; boundedly terminate and confirm the old child exit; then restore the recorded prior Python/native/resources tuple. Restart with a fresh epoch and revalidated current state. Never replay old history scenes or compatibility ticks, never downgrade only the wire format, and circuit-open if old-child exit cannot be confirmed.
