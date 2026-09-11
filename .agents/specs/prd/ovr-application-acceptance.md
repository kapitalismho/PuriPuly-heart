# OVR-APPLICATION-ACCEPTANCE

## Status and authority

**APPLICATION SOURCE ACCEPTED AT `3f71ae548f4f85d2807fac1b033a5fdb3c37fa18` / COMPLETE-GOAL FAST REVIEW `repair_required` SOLELY FOR EXTERNAL #149 / WHOLE #148 GOAL BLOCKED / PRODUCTION CUTOVER NOT PERFORMED**.

**Current integration amendment:** the maintainer subsequently approved OVR-CONTRACT-1 r2 (§0.1 of `ovr-contract-1.md`), removing native caption expiry/lease enforcement entirely. The protocol-7/r1 and native-lease prerequisites in this historical application receipt are superseded by matched protocol 8/r2; native lease acceptance is withdrawn, not an outstanding r2 requirement or a passed criterion. Python presenter TTL, send-time expiry pruning, ordered invalidation and current replay remain required. Current source integration and executable evidence are recorded in `ovr-native-acceptance.md` under r2. This amendment does not turn the original source-only receipt into a deployed-pair or whole-issue acceptance.

| Record | Value |
| --- | --- |
| Authority | #148, approved #146, `OVR-CONTRACT-1 r1`, `ARCHITECTURE.md` |
| Final accepted application source | `3f71ae548f4f85d2807fac1b033a5fdb3c37fa18`; the final FAST complete-Goal review accepted the affected-test repair delta with `no_new_app_defect`. Its `repair_required` verdict is solely for external #149 matched protocol-7 integration, not application source. This is not complete issue #148 acceptance or a matched protocol-7 cutover. |
| Baseline and source lineage | Characterized baseline `97e211047db355b226e54a28790222d00cf95a19`; earlier implementation evidence `4e967df9d03649106faa8348c3ec611009529ffe`; source repair lineage `b7ba7163dbabea570de0a37aa2cf3bd44bfb58b7` -> `2e2aaaa13d753bb027cf1edbb5747dc575b68627` -> prior accepted behavior checkpoint `b0463abf37df21479477f3d3d602fb65ee4630f1` -> documentation baseline `c87110e8f25b1c18e666c7662b17319a51fcc265` -> final accepted application source `3f71ae548f4f85d2807fac1b033a5fdb3c37fa18`. |
| Contract consumed | `OVR-CONTRACT-1 r1`, DESIGN-FROZEN 2026-09-09, SHA256 `115e15b9c577c421ca6c86980c4c99b956ad4a336595cd864097e8af12e4416e`, G-C open |
| Audio revision consumed | No production Audio seam revision. #135's upstream LISTEN recognition admission and #144's SELF speech-origin caller migration, after #143's shared extraction, were not landed at the characterized HEAD. Under r1 §7, #148 owns the parent-output admission and application receipt seam without waiting for full Audio convergence. If an equivalent shared output seam lands first, #148 must consume its exact revision. No Audio landing is a blanket #148 prerequisite. |
| Current source protocol | Python `OVERLAY_CONTRACT_VERSION = 6`; native `EXPECTED_CONTRACT_VERSION = 6` |
| Runtime used | Python 3.12.10 via `uv run --no-project python`; websockets 16.1.1 |
| Application source disposition | **ACCEPTED at `3f71ae548f4f85d2807fac1b033a5fdb3c37fa18`**. The final review found no actionable application defect. The source-only actor changes retain the existing owners and introduce no suspected architecture drift. |
| Deployed pair | Not deployed. No Python/native artifact hashes, authenticated runtime-instance receipt, or matched-pair release record exists. |

## Prior accepted behavior checkpoint boundary

Prior checkpoint `b0463abf37df21479477f3d3d602fb65ee4630f1` contains these 11 paths:

```text
.agents/specs/prd/ovr-application-acceptance.md
ARCHITECTURE.md
src/puripuly_heart/core/orchestrator/peer_translation_channel.py
src/puripuly_heart/core/orchestrator/self_translation_channel.py
src/puripuly_heart/core/orchestrator/translation_output_projection.py
src/puripuly_heart/core/overlay/process.py
src/puripuly_heart/core/runtime/output.py
tests/app/test_overlay_process_manager.py
tests/core/runtime/test_output_runtime.py
tests/core/test_dual_target_translation_lifecycle.py
tests/core/test_output_owner_wiring.py
```

This record retains `b0463abf37df21479477f3d3d602fb65ee4630f1` as the immutable prior behavior checkpoint in the final source lineage.

## Final accepted application source boundary

Final accepted application source `3f71ae548f4f85d2807fac1b033a5fdb3c37fa18`, based on `c87110e8f25b1c18e666c7662b17319a51fcc265`, contains the affected-test repair in:

```text
.agents/specs/prd/ovr-application-acceptance.md
src/puripuly_heart/core/orchestrator/translation_output_projection.py
tests/core/test_translation_output_projection_owner.py
tests/ui/test_desktop_overlay_renderer.py
```

The final production change rejects an older dual-target SELF chatbox snapshot once a newer primary surface has already been presented while preserving later revisions of the same turn. Projection-owner tests use the production parent-admission, destination-readiness, projection, and completion sequence. Desktop malformed-peer tests inject frames through an actual websocket server instead of the removed private `OverlayBridge._broadcast_json` API. No compatibility shim, weakened assertion, timeout increase, desktop failure-code repinning, or retained throwaway probe was introduced. A future documentation-only commit does not change or need to replace the final accepted source identity.

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

## Implemented after-map at final accepted application source

```text
translation parent/generation/order/targets
  -> projection supplies identity-accounted retained payload allocations per parent
  -> OutputRuntime bounded parent admission and destination-local readiness/release
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

## Selected bounds at final accepted application source

| Resource | Final accepted source bound/policy |
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
| Reverse correctness controls | Eight keyed non-lifecycle slots with explicit capacity rejection receipts; reserved priority lifecycle storage keeps ready/shutdown and the first terminal cause observable; non-coalescible overflow fails the process through the existing lifecycle path; reader teardown is finite |
| Websocket | `compression=None`, `max_size=1 MiB`, `write_limit=64 KiB`, `close_timeout=1 s` |
| Write/teardown | Scene 5 s, control 1 s, close 1 s; ambiguous write retires epoch |

## Exact verification commands and selectors

Final fast application matrix command, run at final accepted application source `3f71ae548f4f85d2807fac1b033a5fdb3c37fa18`:

```text
uv lock --check --offline && uv run --no-project python -m pytest -q tests/core/test_overlay_bridge.py tests/core/runtime/test_output_runtime.py tests/core/test_output_owner_wiring.py tests/core/test_translation_turn_owner.py tests/core/test_translation_output_streaming.py tests/core/test_dual_target_translation_lifecycle.py tests/core/test_overlay_presenter.py tests/core/test_self_translation_channel_owner.py tests/core/test_peer_translation_channel_owner.py tests/app/test_overlay_generation_start_owner.py tests/app/test_overlay_translation_enabled_sync.py tests/app/test_overlay_process_manager.py::test_process_reverse_queue_bounds_diagnostics_and_rejects_excess_controls tests/app/test_overlay_process_manager.py::test_owned_process_stop_finishes_with_full_reverse_control_queue tests/app/test_overlay_process_manager.py::test_actual_manager_consumes_reserved_ready_and_runtime_error_after_control_flood tests/app/test_overlay_process_manager.py::test_actual_manager_fails_process_on_noncoalescible_reverse_control_overflow
```

Observed result at prior behavior checkpoint `b0463abf37df21479477f3d3d602fb65ee4630f1`: all `379` collected cases passed in `4.58s`; the command also completed `uv lock --check --offline`. Re-run at final accepted application source `3f71ae548f4f85d2807fac1b033a5fdb3c37fa18`: `uv lock --check --offline` passed, actual collection remained `379`, and the test command passed in `4.52s`.

Final output-focused FAST REPAIR_VERIFY command:

```text
uv run --no-project python -m pytest -q tests/core/runtime/test_output_runtime.py::test_output_runtime_charges_independent_equal_payload_allocations_separately tests/core/test_output_owner_wiring.py::test_production_projection_rejects_independent_equal_payload_copies_above_bound tests/core/test_output_owner_wiring.py::test_production_projection_counts_aliased_source_once_for_legal_large_parent tests/core/test_dual_target_translation_lifecycle.py::test_overlapping_dual_target_parent_projects_ready_surfaces_before_chatbox tests/core/runtime/test_output_runtime.py::test_output_runtime_overlay_replacement_preserves_ui_and_chatbox_parent_admission tests/core/runtime/test_output_runtime.py::test_output_runtime_parent_admission_applies_destination_local_overload_policy tests/core/test_output_owner_wiring.py::test_caption_off_after_parent_admission_preserves_ui_chatbox_and_history
```

Observed result at the prior behavior checkpoint: all `7` selected cases passed in `0.89s`; verdict **VERIFIED**.

Final process-focused FAST REPAIR_VERIFY command:

```text
uv run --no-project python -m pytest -q tests/app/test_overlay_process_manager.py::test_process_reverse_queue_bounds_diagnostics_and_rejects_excess_controls tests/app/test_overlay_process_manager.py::test_owned_process_stop_finishes_with_full_reverse_control_queue tests/app/test_overlay_process_manager.py::test_actual_manager_consumes_reserved_ready_and_runtime_error_after_control_flood tests/app/test_overlay_process_manager.py::test_actual_manager_fails_process_on_noncoalescible_reverse_control_overflow --tb=short
```

Observed result at the prior behavior checkpoint: all `4` selected cases passed in `0.68s`; verdict **VERIFIED**.

Changed-file Ruff command:

```text
uv run --no-project python -m ruff check src/puripuly_heart/core/overlay/process.py src/puripuly_heart/core/runtime/output.py src/puripuly_heart/core/orchestrator/translation_output_projection.py src/puripuly_heart/core/orchestrator/self_translation_channel.py src/puripuly_heart/core/orchestrator/peer_translation_channel.py tests/app/test_overlay_process_manager.py tests/core/runtime/test_output_runtime.py tests/core/test_output_owner_wiring.py tests/core/test_dual_target_translation_lifecycle.py
```

Observed result: `All checks passed!` in `0.17s`.

### Final accepted source affected-test verification

Original broader affected-test reproduction command:

```text
uv run --no-project python -m pytest -q --tb=no --override-ini="addopts=" tests/app/test_overlay_process_manager.py tests/core/runtime/test_overlay_runtime.py tests/core/test_translation_output_projection_owner.py tests/app/test_desktop_overlay_runner.py tests/ui/test_desktop_overlay_renderer.py tests/core/test_overlay_manifest.py tests/core/test_overlay_protocol.py tests/app/test_overlay_session_transition_owner.py tests/app/test_overlay_diagnostics_port_lifecycle.py tests/app/test_overlay_application_transitions.py
```

Observed before this repair: `337` collected, `14` failed, `320` passed, `3` skipped in `5.55s`. Observed at final accepted source `3f71ae548f4f85d2807fac1b033a5fdb3c37fa18` with the exact same command: exit `0`, `334` passed and `3` skipped in `4.47s`. The three skips require `INTEGRATION=1` real subprocesses. The earlier twelve deterministic failures and three order-dependent desktop failures did not recur. Their timeouts and expected window failure code were not changed. No external HMD cause was involved.

Projection-owner command:

```text
uv run --no-project python -m pytest -q --tb=short --override-ini="addopts=" tests/core/test_translation_output_projection_owner.py
```

Observed result: all `22` cases passed in `0.41s`, including the ten previously failing production-admission/currentness/failure-isolation selectors.

Actual-wire desktop selectors:

```text
uv run --no-project python -m pytest -q --tb=short --override-ini="addopts=" tests/ui/test_desktop_overlay_renderer.py::test_desktop_overlay_later_malformed_snapshot_is_ignored_and_controls_dispatch tests/ui/test_desktop_overlay_renderer.py::test_desktop_overlay_invalid_runtime_control_reports_error_without_dispatch
```

Observed result: both cases passed in `0.29s`.

Final-source Ruff command:

```text
uv run --no-project python -m ruff check src/puripuly_heart/core/orchestrator/translation_output_projection.py tests/core/test_translation_output_projection_owner.py tests/ui/test_desktop_overlay_renderer.py
```

Observed result: `All checks passed!` in `0.18s`.

### Repair evidence

- Parent admission is destination-local across overlay, UI, and chatbox. Caption replacement retires only overlay batches, and late results still reach admitted UI, chatbox, and history without rerunning the translation provider.
- Speech pressure evicts the oldest wholly unsent batch before aggregate byte rejection. Manual pressure rejects only the incoming pressured destination, so independently admitted destinations continue.
- Retained payload accounting de-duplicates live immutable aliases by identity while charging independent equal allocations separately. Production projection accepts a 400 KiB source plus 400 KiB translation and an exactly 1 MiB source-only parent plus its matching close, while a 600 KiB source plus an independently allocated equal 600 KiB passthrough translation is terminally rejected before projection.
- Managed overlay events require an already-admitted parent, generation, order, and expected target. Unknown identities are rejected without retiring the live parent; SELF preview uses its own active/latest scope and cannot evict speech.
- TALK reset cancels SELF speech scope without cancelling an in-flight manual parent; LISTEN reset clears Peer state without retiring SELF; Caption OFF plus sink generation replacement prevents a late old callback from applying to the replacement sink.
- Currentness is paired to actual parent admission: the guard preserves later revisions of the same turn while rejecting an older parent snapshot after a newer dual-target SELF primary surface is visible, even while destination-local chatbox admission drains in FIFO order. The accepted projection-owner selectors exercise both revision progression and older-parent suppression.
- Reverse lifecycle controls have reserved priority and preserve the first terminal cause. The actual `_AsyncioOverlayProcess` lifecycle sink and actual `OverlayProcessManager` consume ready plus runtime failure after eight queued renderer controls without an exception; a ninth non-coalescible renderer control produces the explicit `reverse_control_rejected` receipt and fails the process with `reverse_control_capacity`. Owned-process reader teardown remains finite.
- The retained real stopped-reader loopback websocket, the twelve-parent integrated stalled-send chain, and the overlapping dual-target destination-admission chain remain in the passing fast matrix.

### FAST review disposition

- The prior behavior checkpoint's two independent FAST FULL_REVIEW lanes covered output/application behavior and process reverse-control/finite-stop behavior. Every accepted finding from those lanes was repaired in the source lineage above.
- The checkpoint's final output-focused and process-focused FAST REPAIR_VERIFY verdicts were both **VERIFIED** against exact commit `b0463abf37df21479477f3d3d602fb65ee4630f1`. The `379`-case matrix and focused `7`-case output and `4`-case process commands above remain valid for that checkpoint.
- A proposed cancellation finding based on a synthetic yielding bridge was **REJECTED** because the yielding bridge was not the actual production bridge behavior. It was not treated as production evidence and was not revived as a repair requirement.
- Final FAST complete-Goal review at `3f71ae548f4f85d2807fac1b033a5fdb3c37fa18`: **`repair_required` solely for external #149**. The application repair delta was accepted with **`no_new_app_defect`** and no actionable application source finding.
- The broader `337`-case command independently exited `0` with `334` passed and `3` `INTEGRATION!=1` real-subprocess skips. `ElevatedButton` deprecation warnings are outside this application repair scope.
- No throwaway probe file is retained in the prior checkpoint or final accepted source.
- The application source outcome is complete. The whole #148 Goal remains blocked only on external #149 matched protocol-7 native + Python/desktop integration; neither review asserts deployment or cutover.

### Integrated stalled-send and real-socket evidence

`tests/core/test_output_owner_wiring.py::test_actual_owner_chain_completes_twelve_parents_while_bridge_socket_is_stalled` now drives the actual translation owners, projection, `OutputRuntime`, `OverlayPresenter`, and `OverlayBridge`. It alternates twelve manual/peer parents and asserts twelve distinct parent IDs, twelve distinct blocks, UI queue size 24, chatbox size 6, zero output reservations, and zero completed bridge sends while the bridge connection is stalled. The independent validation wave observed completion in 7.118 ms; the retained regression enforces the behavioral counts and non-transport completion rather than a platform-sensitive wall-clock number.

`tests/core/test_overlay_bridge.py::test_overlay_bridge_real_socket_stopped_reader_stops_bounded_and_truthfully` uses an actual loopback websocket, a 4 KiB receive buffer, paused client reads, and repeated 900,000-character scenes until the server write is active. Its exact standalone command passed in 1.42 s. A direct retained smoke observation blocked at revision 6; `stop()` returned success in 1012.717 ms with zero unresolved transport tasks, zero authenticated connections, and `stopped=True`. The resistant-send/close selector separately proves that a genuinely unresolved operation causes a bounded, truthful `ExceptionGroup`.

`tests/core/test_dual_target_translation_lifecycle.py::test_overlapping_dual_target_parent_projects_ready_surfaces_before_chatbox` admits two actual manual dual-target parents, keeps both secondary provider calls behind controlled barriers, and stalls the real bridge writer. After parent one releases its overlay/UI obligations while retaining the active chatbox scope, parent two's primary UI and overlay state apply while its chatbox batch remains the single waiter. No detached projection task or additional FIFO is introduced; the parent-owned child remains bounded until its remaining destination becomes ready.

`tests/core/test_output_owner_wiring.py::test_production_projection_rejects_independent_equal_payload_copies_above_bound` drives the actual owners with a 600 KiB source and independently allocated equal passthrough translation. It confirms one provider call, no translation projection/chatbox publication, terminal reservation release, and no retained output bytes.

## OA01–OA10 ledger

No row below establishes whole-Goal or complete issue #148 acceptance. The application source is finally accepted at `3f71ae548f4f85d2807fac1b033a5fdb3c37fa18` with no actionable application defect. The complete-Goal FAST verdict is `repair_required` solely for external #149 matched protocol-7 native + Python/desktop integration and its integrated binary, scope, status, invalidation, supervisor, and deployment receipts.

| OA | Final application source evidence | Disposition |
| --- | --- | --- |
| OA01 | Integrated production-owner stalled-send selector and real stopped-reader loopback websocket pass with bounded truthful stop | **Application subcriterion covered; not full OA acceptance** |
| OA02 | Deterministic cancellation records ambiguity, retires one epoch, rejects replacement while unresolved, and replays current state only after resolution | **Final application source evidence; native/HMD acceptance remains external and HMD is non-blocking** |
| OA03 | Provisional/final/clear coverage, one active plus one successor scene, reserved shutdown control, and 200-overflow coalescing pass | **Application pressure evidence covered; full cross-process invalidation criterion not run** |
| OA04 | Same-text identity, exact sequence-hole retirement, late-currentness, and bounded 4100-namespace regressions pass | **Application subcriterion independently reviewed and accepted; matched integration evidence is external #149** |
| OA05 | Send-time expiry removes expired blocks and native intent references; reconnect replay remains current | **Application subcriterion covered; native lease/epoch acceptance not implemented** |
| OA06 | Production owner tests exercise TALK speech-only reset with an in-flight manual parent, LISTEN Peer reset with a live SELF parent, and Caption OFF/replacement with a late old result | **Application scope matrix covered; native invalidation remains external** |
| OA07 | Reverse diagnostics remain bounded; reserved lifecycle priority preserves ready/shutdown and the first terminal cause; actual manager/process tests prove explicit overflow failure without consumer exceptions; owned-process reader cancellation and finite shutdown receipts pass | **Application subcriterion covered; native flood behavior remains external** |
| OA08 | Overlay replacement/failure preserves UI, chatbox, and history, and the controlled provider is invoked exactly once | **Application destination/failure isolation covered; native destination evidence absent** |
| OA09 | One-active/eight-waiting admission, speech eviction before rejection, manual destination-local rejection, identity-based actual-copy accounting, exact alias-aware 1 MiB accounting, and overlapping dual-target progress from actual owners through presenter/bridge pass | **Application subcriterion covered; upstream Audio admission remains external; full OA09 not accepted** |
| OA10 | Protocol-6 application/desktop baseline remains green | **Protocol 7 and matched packaged pair remain blocked** |

## Sole whole-Goal repair requirement: external #149

Final complete-Goal FAST review has been performed at `3f71ae548f4f85d2807fac1b033a5fdb3c37fa18`. Its only remaining `repair_required` scope is #149 matched protocol-7 native + Python/desktop integration:

- protocol-7 `execution_contract {version: 1, revision: "r1"}` negotiation;
- native three-second validity lease and challenge/response renewal;
- native current-status receipts and scoped invalidation watermarks;
- bounded native due/render/GPU/OpenVR attempt progress and #149 supervisor recovery;
- integrated application/native binary and scope receipts, packaged desktop/native resources, and mixed-version fail-fast evidence;
- deployed Python/native artifact hashes, authenticated runtime-instance receipts, and a matched-pair release record.

Physical native/OpenVR/HMD observation and cross-OS stopped-reader staging measurements remain valuable non-blocking observations. **No HMD execution prerequisite applies to software acceptance.** Lack of HMD evidence must be recorded as `not_observed`/`not_observable`; it does not withhold the accepted software-only #148 application source outcome. Production activation of a matched native/application pair remains a separate release/cutover decision.

## Pair/cutover and rollback

Final accepted application source `3f71ae548f4f85d2807fac1b033a5fdb3c37fa18` is not a deployed pair. A future documentation-only commit does not alter or need to replace this source identity. No push, deployment, production cutover, or issue closure was performed. Production cutover requires the exact matched Python/native/resources tuple and capability/version evidence; this receipt does not assert those exist.

Rollback is paired: stop ingress; retire output, connection, process, and device epochs; boundedly terminate and confirm the old child exit; then restore the recorded prior Python/native/resources tuple. Restart with a fresh epoch and revalidated current state. Never replay old history scenes or compatibility ticks, never downgrade only the wire format, and circuit-open if old-child exit cannot be confirmed.
