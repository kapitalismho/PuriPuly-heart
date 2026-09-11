# OVR-NATIVE-ACCEPTANCE

## Status and authority

**IMPLEMENTED / CLAIMED FOR OVR-N; pending a committed candidate, independent FAST review, and Director acceptance. PHYSICAL-HMD FRESHNESS IS NOT CERTIFIED.**

This document records implementation claims and local verification evidence. It is not an acceptance decision.

| Record | Value |
| --- | --- |
| Authority | Issue #149 under #145; approved `OVR-CONTRACT-1 r1` from #146 |
| Actual work-start SHA | `80b15cef49ad47417f6a22abfe27c2a5fabba387` |
| Work-start branch | `ovr-0-vr-overlay-reliability-program-cross-envir` |
| Work-start upstream | `origin/ovr-0-vr-overlay-reliability-program-cross-envir` |
| Work-start tree state | Clean baseline |
| Consumed contract | `.agents/specs/prd/ovr-contract-1.md`, SHA256 `115e15b9c577c421ca6c86980c4c99b956ad4a336595cd864097e8af12e4416e` |
| Resolved N1 profile | `p05`: 100 ms cadence, 500 ms scheduling wall, maximum 5 final opportunities |
| Wire contract | Version 7; execution contract `r1` version 1; native retry ownership `exclusive` version 1 |
| Production entrypoints | Python `OverlayProcessManager` / `_AsyncioOverlayProcess`; Rust `run_with_manifest` -> `NativePresentationOwner::run` -> `PresentationRuntime` / `CaptionRenderer` / `OpenVrOverlay` |
| Native build | Windows x64 release profile, Rust 1.97.1, `C:/ovr-target/release/PuriPulyHeartOverlay.exe` |
| Binary SHA256 | `a00ca7edf3aec6c425ca88d34645fdec91af5450526e70b07dc72dc94bf6cda5` |
| Python verification runtime | Python 3.14.0 for direct `python` runs; project tools also resolved Python 3.12.10 through `uv` |
| Physical environment | Windows 11 x64; no SteamVR session and no physical HMD observation |

The implementation extends the existing native owner and Python supervisor. It does not add a second root manager, a backend-specific recovery controller, a compatibility wire path, or a historical retry queue.

## Owner and resource map

| Resource/state | Owner | Lifetime and terminal rule |
| --- | --- | --- |
| Current scene, active write, successor scene | Python `OverlayBridge` | Bounded current/active/successor mailbox; cancellation or timeout retires the connection epoch rather than declaring remote acceptance |
| Authenticated process identity | Python process manager and native `BridgeClient` | Exact overlay instance ID plus runtime generation on lifecycle, health, and validity messages; old-epoch messages are rejected |
| Scene validity | Native owner challenge plus Python bridge response | A non-empty scene cannot be presented until a matching current-revision response supplies finite occupant leases; expiry invalidates eligibility and requests hide reconciliation |
| CPU render objects | `CaptionRenderer` | Reused within the device/process epoch; text-format, layout, line-command, and block-command caches share a fixed 64 MiB accounted-retained-byte budget and retain independent entry caps |
| Render attempt and D3D readiness query | `NativePresentationOwner` / `CaptionRenderer` | One outstanding producer/query is retained after enqueue through readiness or a distinct late/cancel/fatal terminal outcome; it is never reused to certify a successor generation |
| Producer-ready frame | Native presentation owner | Scope, lease, scene generation, and due intent are revalidated before handoff |
| Handoff texture association | OpenVR submitter | Retained according to the OpenVR association and process/device epoch; software submit success is not reported as physical visibility |
| Retry intent | `PresentationRuntime` | One bounded schedule per channel; preemption transfers due intent to the current scene; completion is charged only to an eligible successful submission |
| Spatial seen identity | Native spatial-lock state | Maximum 64 retained identities; semantic identities retire only behind protocol frontiers; capacity refusal is explicit |
| Diagnostics and logs | `OverlayLogger` writer owner | Diagnostics use a bounded nonblocking queue with saturating drop accounting. One dedicated writer thread owns stdout/stderr; one bounded current-write state records generation, actual start, and timeout without a metadata FIFO. A Windows watchdog marks timeout and requests cancellation while holding the same state lock used by writer completion/successor start, so cancellation cannot cross generations, and enforces the 25 ms start-relative deadline. `run_with_manifest` explicitly closes admission and boundedly joins both threads after native/GPU owner teardown; an unresolved writer or watchdog is a process-terminal cleanup failure, never a confirmed join. Reliable lifecycle records retain a small priority queue and bounded acknowledgement. |
| Reverse status/control | Python process reader and reserved lifecycle storage | Diagnostics are bounded/coalesced; ready, runtime error, shutdown acknowledgement, and the first terminal cause cannot be buried by diagnostic flood |
| Child process and readers | Existing process manager | Ingress/restart is suppressed first, shutdown and reader drain are bounded, terminate escalates to kill, and actual exit must be confirmed |

Renderer command-list accounting charges the raster area of each retained visual at four bytes per pixel plus a fixed command-list overhead, together with retained key text. Because driver-private D3D/D2D allocation is opaque, this is an explicit conservative software accounting boundary, not a measurement of driver heap residency.

## Progress and failure timeline

1. Authentication binds the websocket to one overlay instance and runtime generation.
2. A non-empty initial or replacement scene causes the native owner to issue a bounded validity challenge. The bridge answers only for the current locally accepted revision.
3. A matching response installs finite occupant leases. Empty scenes bypass the lease gate so clear and shutdown remain prompt.
4. A presentation attempt records logical revision, scene generation, cause, render generation, and attempt number. Producer progress and publication progress remain separate facts.
5. Websocket ingress, OpenVR events, readiness polling, retry cadence, lease expiry, hide deadlines, and shutdown receive bounded per-turn service. Heartbeats, stale snapshots, control no-ops, and preemption do not reset the original due-work no-progress deadline.
6. Cancellation after D3D query enqueue retains attempt/query ownership. Late completion can retire that producer but cannot authorize an old generation or satisfy a successor query. CPU state may advance to a successor while the sole producer remains incomplete; no second producer is enqueued.
7. One readiness lateness performs bounded retry. Repeated no-progress consumes the fixed episode budget and preserves readiness/query/device/runtime/bridge causes.
8. Health status reports owned stage and last meaningful progress. Only a challenged, current-identity, current-generation status qualifies; the Python supervisor does not equate a live pipe or fresh ready handshake with presentation progress.
9. Restart allowance is bounded to the initial child plus three replacements in the failure window. Exactly 60 seconds of qualifying current-owner progress refills it; cap plus one is rejected before refill.
10. Teardown stops publication first, hides/releases in runtime order, drains or cancels readers, escalates terminate to kill, confirms child exit, and preserves the first failure if cleanup also fails.

## ON01-ON10 implementation claims and local evidence

`New regression` means added or materially strengthened in this implementation. `Existing regression` means an already-present selector that was retained and rerun.

| ID | Claim state | Exact local evidence |
| --- | --- | --- |
| ON01 | Implemented; locally observed | **Strengthened regression:** `native/overlay/tests/runtime.rs::production_owner_readiness_no_progress_escalates_after_legacy_count_without_submit` completed in 2.21 s with more than five readiness timeouts, CPU successor revision progress, and zero submit calls while one producer remained incomplete. **Existing regression:** `production_owner_preemption_preserves_due_and_completes_on_pending_snapshot` completed in the default suite and records current-generation retry completion. |
| ON02 | Implemented; locally observed | **Existing regression:** `native/overlay/tests/runtime.rs::production_owner_openvr_event_flood_does_not_starve_snapshot_submit` completed in the normal default-parallel native run. The default command completed 288 tests across six suites; no serial override is retained. |
| ON03 | Implemented; locally observed | **Existing regressions:** `production_owner_single_readiness_timeout_retries_without_submit_or_exit` and `production_owner_active_schedule_readiness_failure_is_terminal` completed in the 90-test runtime integration binary. The strengthened ON01 selector covers bounded repeated no-progress. |
| ON04 | Implemented at the Windows D3D boundary; OpenVR/HMD conformance not observed | **New real-Windows regression:** `native/overlay/tests/renderer.rs::windows_graphics_real_query_cancelled_after_enqueue_is_retained_until_late_completion` passed, 1 passed / 65 filtered / 0.05 s. It creates a real Windows D3D11 device/query, cancels only after enqueue, observes retained ownership, then observes late readiness. **Strengthened regression:** the ON01 selector proves CPU-only successor progress without a second producer. No SteamVR runtime, real OpenVR compositor, or HMD was involved. |
| ON05 | Implemented; locally observed | **New real-Windows regression:** `native/overlay/src/logging.rs::tests::real_stopped_stdout_and_stderr_pipes_do_not_own_process_shutdown` passed within the focused logging run. It launches disposable native test children with actual piped stdout and stderr, intentionally retains the unread pipe ends, fills each pipe, and observes bounded logger shutdown and child exit. **New transition regressions:** `::stalled_watchdog_tracks_one_current_write_and_preserves_timeout` proves a stalled writer retains exactly one current generation/start/deadline state under queue saturation and reports the elapsed timeout after the sink releases; `::timeout_cancellation_cannot_cross_from_completed_write_to_successor` races A completion/B start against A's timeout cancellation and observes that B cannot start until A's correlated cancellation returns, then completes B without a timeout. **Strengthened regressions:** `::diagnostic_hot_path_is_nonblocking_and_bounded_when_writer_stalls` and `::dropped_record_counter_saturates` passed. **Existing regressions:** `tests/app/test_overlay_process_manager.py::test_process_reverse_queue_bounds_diagnostics_and_rejects_excess_controls` and `::test_actual_manager_consumes_reserved_ready_and_runtime_error_after_control_flood` completed in the unchanged affected Python suite. |
| ON06 | Implemented; locally observed | **Existing regression:** `native/overlay/tests/runtime.rs::production_owner_overlay_hidden_reasserts_show_when_desired_visible` and the idle/empty/pose-unavailable cases completed in the 90-test runtime integration binary. These paths do not consume the due-work no-progress episode unless eligible work is actually stalled. |
| ON07 | Implemented in software/runtime harness; physical visibility not observed | Existing external-hide/show, runtime-event, spatial-reentry, refresh, and translation-update regressions completed in the full native suite. The renderer integration binary completed all 66 tests. These are API/state observations, not headset pixel observations. |
| ON08 | Implemented; locally observed | **New regressions:** `tests/app/test_overlay_process_manager.py::test_owner_health_requires_current_identity_and_sixty_seconds_before_restart_refill` and `tests/app/test_overlay_application_transitions.py::test_terminal_restart_budget_rejects_cap_plus_one_until_qualified_progress`. Together with existing `::test_watch_runtime_restarts_connected_crash_and_keeps_peer_activation`, the exact six-selector Python reliability command completed 6 passed in 0.39 s. Existing terminate/kill escalation selectors completed in the 137-test affected Python run. |
| ON09 | Implemented; locally observed | **New native regressions:** `runtime::tests::validity_challenge_window_rejects_oldest_at_cap_plus_one`, `::current_scene_lease_expires_and_requests_hide_reconciliation`, and `::matching_validity_response_installs_only_current_scene_lease` each passed exactly. **New Python regressions:** `test_overlay_bridge_health_challenge_window_is_bounded_at_cap_plus_one`, `test_overlay_bridge_only_challenged_current_status_clears_acceptance_deadline`, and `test_overlay_bridge_validity_response_uses_current_revision_and_clamps_expired_lease` passed in the exact six-selector command. |
| ON10 | Implemented; locally observed | **New regressions:** `renderer::cache::tests::bounded_lru_cache_enforces_retained_bytes_at_cap_and_cap_plus_one`, `::bounded_lru_cache_rejects_single_unaccountable_oversized_entry`, and `::renderer_cache_partitions_sum_to_selected_64_mib_budget` each passed exactly. Existing spatial/peer identity, bridge mailbox, audit, and process queue bounds completed in the full native and affected Python suites. |

## OVR-CONTRACT-1 outcome coverage claims

| Criterion | Disposition |
| --- | --- |
| OC04 | Claimed covered by cancellation-only readiness, stale/preempted scene transfer, the fixed original deadline, and current-scene completion selectors listed above. |
| OC05 | Claimed covered at the Windows D3D11 query boundary by the real-query after-enqueue selector and at the owner boundary by retained-producer/successor tests. OpenVR submit-path tests in this environment use `FakeOpenVr`/mocks; no SteamVR compositor or physical HMD ran. |
| OC06 | Claimed covered by idle/empty/hidden/pose-unavailable and deferred spatial-anchor regressions; only due unresponsive work consumes recovery. |
| OC07 | Claimed covered by the actual stopped-stdout/stderr disposable native child, bounded single-current-write watchdog state, generation-associated cancellation, start-relative 25 ms timeout preservation, explicit logger admission close/cancel/bounded join, process-terminal cleanup failure on an unresolved writer/watchdog, nonblocking diagnostic admission, saturating oversize/drop accounting, reverse diagnostic floods, reserved controls, sticky first cause, and finite reader teardown. Log presence is not pass evidence. |
| OC08 | Claimed covered by exact epoch authentication, current-scene replay, bounded validity challenges, current-revision leases, expiry, OFF/restart suppression, and stale callback rejection. |
| OC09 | Claimed covered by the initial-plus-three limit, exact cap-plus-one rejection, current-identity 60-second refill, ready-flap without refill, hung process escalation, and duplicate-child prevention. |
| OC10 | Claimed covered by semantic retirement filtering, 64-entry spatial/peer identity limits, the aggregate 64 MiB accounted renderer-cache limit, bounded audits/queues, and overload/drop reporting. |
| OC11 | Windows release build and direct executable startup-contract smoke check completed locally. Software visibility and API outcomes remain distinct from physical-HMD observation. |
| OC12 | The affected Python bridge/process/application matrix completed 137 passed in 4.07 s. The broader protocol/manifest/bridge/process/desktop/shutdown/lifecycle matrix previously completed 173 passed, 2 integration-gated skips in 3.81 s; the two exact gated desktop import selectors then completed 2 passed in 0.74 s with `INTEGRATION=1`. |

## Verification evidence

### Default-parallel native suite

```text
CARGO_TARGET_DIR=C:/ovr-target cargo test --locked --manifest-path native/overlay/Cargo.toml
```

Observed after the bounded diagnostic-writer lifetime, current-write observation, and cancellation-association repairs: `292 passed` across six suites (124 library, 66 renderer integration, 90 runtime integration, 12 state integration); no serial override was used. The real stopped-pipe selector completed with actual unread stdout and stderr pipes. A prior command run amid competing Cargo invocations was cancelled and is not acceptance evidence; the isolated normal-default command above completed. The initial cache-weight implementation exposed four renderer regressions by evicting line visuals needed in the same frame; accounting was corrected to charge visual raster bounds plus overhead, and the 66-test renderer integration binary then completed before the full default run.

Exact focused observations:

- Real Windows query cancellation after enqueue: 1 passed, 65 filtered, 0.05 s.
- Persistent incomplete producer with CPU-only successor and bounded escalation: 1 passed, 89 filtered, 2.21 s.
- Three native validity/lease boundary selectors: each 1 passed, 119 filtered.
- Three cache byte-boundary selectors: each 1 passed, 119 filtered.
- Nonblocking diagnostic writer, bounded current-write watchdog state, generation-associated cancellation race, saturating drop counter, actual stopped stdout/stderr pipes, and routing/mode selectors: 9 passed in the focused logging run.
- Actual stopped stdout/stderr coverage retained disposable native children with unread OS pipes; the focused run's child-exit bound remained 2 s per stream.
- Resolved `p05` unit and production-policy selectors: each 1 passed.
- `cargo fmt --manifest-path native/overlay/Cargo.toml -- --check`: completed with no output.

### Python bridge, process, health, validity, and restart

```text
python -m pytest -q --override-ini=addopts= \
  tests/core/test_overlay_bridge.py \
  tests/app/test_overlay_process_manager.py \
  tests/app/test_overlay_application_transitions.py \
  tests/app/test_application_runtime_lifecycle.py
```

Observed after final formatting: `137 passed in 4.07s`.

The exact six-selector health/validity/restart command observed `6 passed in 0.39s`. `uv run --frozen --extra dev ruff check` reported `All checks passed!`; `uv run --frozen --extra dev black --check` reported all five changed Python files unchanged.

### Release build and executable smoke check

```text
CARGO_TARGET_DIR=C:/ovr-target cargo build --locked --release --manifest-path native/overlay/Cargo.toml --bin PuriPulyHeartOverlay
C:/ovr-target/release/PuriPulyHeartOverlay.exe --check-startup-contract
```

Release build completed in 6.87 s. Observed startup contract:

```json
{"app_version":"2.6.1","contract_version":7,"execution_contract":{"revision":"r1","version":1},"native_presentation_retry":{"ownership":"exclusive","version":1}}
```

Release binary SHA256: `a00ca7edf3aec6c425ca88d34645fdec91af5450526e70b07dc72dc94bf6cda5`.

## Compatibility, limitations, and rollback

Head-locked and spatial-locked behavior, desktop paths, OpenVR-selected D3D11 adapter validation, accounted renderer-cache policy, and exclusive native retry ownership remain supported. Protocol 6 is not retained: Python and native must deploy as a matched protocol-7 pair.

No SteamVR/HMD session was observed. This record does not certify physical pixel freshness, compositor behavior, driver-wide hang recovery, headset sleep/wake on a particular device, or guaranteed user-visible recovery from a host OS/GPU-driver hang. The real Windows cancellation selector proves D3D11 query ownership and late completion only. OpenVR-facing tests without SteamVR use test doubles. Software reports physical visibility as `not_observable` unless independently observed.

Rollback is paired. Stop application ingress; set overlay OFF; retire publication, connection, process, and device epochs; boundedly terminate and confirm the protocol-7 child exit; restore the recorded prior Python/native/resources tuple; then restart with a fresh epoch and revalidated current state. Never downgrade only one wire peer, replay old retry history, reuse an old validity lease, or launch a replacement while old-child termination is unconfirmed.
