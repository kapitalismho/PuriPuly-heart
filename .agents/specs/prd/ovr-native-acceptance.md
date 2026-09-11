# OVR-NATIVE-ACCEPTANCE

## Status and authority

**SOFTWARE OUTCOME ACCEPTED at source commit `9867b819afb2d26d3e8cfbc09f4de83f815f8fde`. Independent FAST checkpoint, repair verification, and complete-Goal review passed. PHYSICAL-HMD FRESHNESS AND DEPLOYED-PAIR CONFORMANCE ARE NOT CERTIFIED.**

The Director accepts the local software implementation and its applicable evidence for issue #149 at the reviewed source commit. This receipt does not authorize or claim push, merge, deployment, release, or issue closure. A later documentation-only receipt commit does not replace the accepted source identity.

### Review and acceptance receipt

| Boundary | Exact candidate and result |
| --- | --- |
| Complete source range | `80b15cef49ad47417f6a22abfe27c2a5fabba387..9867b819afb2d26d3e8cfbc09f4de83f815f8fde` |
| Initial independent FAST checkpoints | `49541c90f3809f4bfad47a6025722f64abe5c7a8`: native lifetime/concurrency and Python supervisor/protocol lanes found N-F1–N-F5 and P-F1–P-F5. All were accepted for correction, not waived. |
| Python FAST repair verification | `e35a4b4d73f1b7b8dc658f1e3f60711d2a48d541`: P-F1–P-F5 verified; unchanged Python evidence retained for the final native-only repair. |
| Native FAST repair verification | `9867b819afb2d26d3e8cfbc09f4de83f815f8fde`: all N-F1–N-F5 verified, including prolonged pose waiting and separately reported cleanup failure. |
| Fresh complete-Goal FAST review | `9867b819afb2d26d3e8cfbc09f4de83f815f8fde`: `accepted`, no findings, complete N1–N9 / ON01–ON10 / applicable OC04–OC12 coverage. Production mixed-version/capability fail-fast and desktop capability handling were additionally probed; release binary hash was independently checked. |
| Applicable validation | Native default-parallel suite: 298 passed. Python integration matrix with `INTEGRATION=1`: 235 passed. Real Windows D3D query and unread stdout/stderr child-process checks passed; OpenVR-facing tests used test doubles. Exact scope and commands remain below. |
| Director disposition | Software Outcome and Goal accepted at the reviewed source SHA; no unresolved accepted findings. No new architecture owner or boundary introduced. SteamVR/HMD, installed Python/native artifacts, and deployment remain unverified. |

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
| Binary SHA256 | `710a5d6ed576ac6821803c45625da2c4169f24480347c7681e90c39a4392d294` |
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

## Consolidated checkpoint repair record

The stable candidate `49541c90f3809f4bfad47a6025722f64abe5c7a8` failed two independent FAST reviews. The Director adjudicated N-F1 through N-F5 and P-F1 through P-F5 as accepted findings. The committed repairs preserve the existing native owner, Python bridge/process supervisor, profile `p05`, and protocol 7 boundaries; they add no manager or Audio authority. The initial failed verdict and invalidated claims remain recorded here as history.

| Finding | Repaired disposition and observable evidence |
| --- | --- |
| N-F1 | Desired runtime visibility now requires a current valid lease. `production_owner_expired_lease_hides_without_reasserting_stale_texture` exercises the full owner pump through the three-second lease expiry, observes hide, and proves the associated stale texture is not shown again. |
| N-F2 | Due deadlines arm only for accepted render, handoff, visibility, placement, hide, or invalidation work; challenged status, timer wakes, stale/no-op messages, lease-only renewal, and a pending pose acquisition do not arm them. Pose waiting retains bounded retry cadence while control, challenged status, and lease traffic remain serviceable. A submitted frame or observed required visibility completion clears a real episode. `production_owner_pose_wait_outlives_no_progress_budget_then_handoffs_same_occupant`, `production_owner_stable_visible_renewals_do_not_arm_due_deadline`, and the strengthened `production_owner_event_pump_preserves_idle_hide_tail` run beyond their controlled no-progress deadlines without false failure; `production_owner_readiness_no_progress_escalates_after_legacy_count_without_submit` retains real GPU-churn failure coverage. |
| N-F3 | Pose-unavailable spatial reanchor keeps the current pending anchor, suppresses unanchored handoff/show, and retries the same occupant at bounded cadence without restarting the owner. `production_owner_pose_wait_outlives_no_progress_budget_then_handoffs_same_occupant` exercises the actual `NativePresentationOwner` loop beyond its no-progress budget with runtime control and health challenges active, then observes one same-occupant handoff after pose recovery. `unavailable_spatial_pose_defers_handoff_and_retries_same_occupant` retains the direct runtime boundary coverage. |
| N-F4 | Runtime bridge, render, OpenVR, disconnect, stop, readiness-late, readiness-cancelled, query, and stalled causes retain distinct control-event reasons. Startup no longer rewrites readiness lateness to `renderer_init_failed`. Terminal owner teardown preserves the primary error while reporting an actionable cleanup-hide failure separately, and promotes cleanup failure when there is no primary; it never fabricates confirmed hide. `production_owner_preserves_primary_failure_and_reports_hide_cleanup_failure`, `production_owner_promotes_hide_cleanup_failure_after_successful_shutdown`, and `runtime_and_startup_failure_reasons_preserve_first_distinct_cause` exercise those paths. |
| N-F5 | Native status now reports the normative classification, latest actual stage, desired visibility, separately observed runtime visibility, lease state, due elapsed time, and current-covered handoff or observed hide. Pose-unavailable and terminal classifications are emitted from actual owner states. Successful Show/Hide API return records a pending request rather than fabricating runtime observation. |
| P-F1 / P-F2 | The bridge marks only outstanding, issuance-relative, unexpired challenges as validated. The manager accepts only increasing validated challenge evidence and requires no overdue work plus a current lease-covered handoff or observed requested hide for a contiguous 60-second refill. Native challenge replies carry the owner’s actual due episode. `test_owner_health_requires_validated_increasing_challenges_for_sixty_second_refill` covers unchallenged, replayed, and increasing valid evidence through the real bridge validation path. |
| P-F3 | The total startup budget begins before preparation. Preparation and manifest writing run off the application loop; a timed-out or cancelled late spawn remains manager-owned and is reaped before replacement. `test_startup_budget_includes_nonblocking_prepare_and_reaps_late_spawn_before_replacement` uses controlled barriers and proves no duplicate spawn. The translation startup harness now emits the exact protocol-7 identity, generation, and capability envelope rather than timing out on an obsolete ready stub. |
| P-F4 | Production generation composition now supplies `bridge.broadcast_shutdown` to native and desktop managers. Runtime close marks shutdown intent without falsely marking the request sent; the manager owns one three-second request/ACK/exit deadline followed by the existing one-second terminate and two-second kill/confirmed-exit bounds. Process readers drain through EOF before bounded cancellation so a terminal ACK cannot be lost behind process exit. Closing a generation suppresses retry-ownership callbacks that would otherwise create fresh presenter retry tasks on the closing runtime; the successor generation re-establishes ownership. `OverlayRuntimeHandle` adds no broadcast-and-sleep budget before `process.stop`. Generation-owner, process-manager, runtime-handle, translation restart-reuse, and real-subprocess ACK selectors exercise the actual boundary. |
| P-F5 | Earlier ON08/OC09 challenged-health evidence and the native graceful-shutdown composition statement were invalidated by FAST review and replaced by the selectors above. Independent repair verification and complete-Goal review subsequently passed at the candidates in the acceptance receipt; the earlier claims are not retroactively treated as passing evidence. |

## ON01-ON10 implementation claims and local evidence

`New regression` means added or materially strengthened in this implementation. `Existing regression` means an already-present selector that was retained and rerun.

| ID | Claim state | Exact local evidence |
| --- | --- | --- |
| ON01 | Implemented; locally observed | **Strengthened regression:** `native/overlay/tests/runtime.rs::production_owner_readiness_no_progress_escalates_after_legacy_count_without_submit` completed in 2.21 s with more than five readiness timeouts, CPU successor revision progress, and zero submit calls while one producer remained incomplete. **Existing regression:** `production_owner_preemption_preserves_due_and_completes_on_pending_snapshot` completed in the default suite and records current-generation retry completion. |
| ON02 | Implemented; locally observed | **Existing regression:** `native/overlay/tests/runtime.rs::production_owner_openvr_event_flood_does_not_starve_snapshot_submit` completed in the normal default-parallel native run. The default command completed 298 tests across six suites; no serial override is retained. |
| ON03 | Implemented; locally observed | **Existing regressions:** `production_owner_single_readiness_timeout_retries_without_submit_or_exit` and `production_owner_active_schedule_readiness_failure_is_terminal` completed in the 95-test runtime integration binary. The strengthened ON01 selector covers bounded repeated no-progress. |
| ON04 | Implemented at the Windows D3D boundary; OpenVR/HMD conformance not observed | **New real-Windows regression:** `native/overlay/tests/renderer.rs::windows_graphics_real_query_cancelled_after_enqueue_is_retained_until_late_completion` passed, 1 passed / 65 filtered / 0.05 s. It creates a real Windows D3D11 device/query, cancels only after enqueue, observes retained ownership, then observes late readiness. **Strengthened regression:** the ON01 selector proves CPU-only successor progress without a second producer. No SteamVR runtime, real OpenVR compositor, or HMD was involved. |
| ON05 | Implemented; locally observed | **New real-Windows regression:** `native/overlay/src/logging.rs::tests::real_stopped_stdout_and_stderr_pipes_do_not_own_process_shutdown` passed within the focused logging run. It launches disposable native test children with actual piped stdout and stderr, intentionally retains the unread pipe ends, fills each pipe, and observes bounded logger shutdown and child exit. **New transition regressions:** `::stalled_watchdog_tracks_one_current_write_and_preserves_timeout` proves a stalled writer retains exactly one current generation/start/deadline state under queue saturation and reports the elapsed timeout after the sink releases; `::timeout_cancellation_cannot_cross_from_completed_write_to_successor` races A completion/B start against A's timeout cancellation and observes that B cannot start until A's correlated cancellation returns, then completes B without a timeout. **Strengthened regressions:** `::diagnostic_hot_path_is_nonblocking_and_bounded_when_writer_stalls` and `::dropped_record_counter_saturates` passed. **Existing regressions:** `tests/app/test_overlay_process_manager.py::test_process_reverse_queue_bounds_diagnostics_and_rejects_excess_controls` and `::test_actual_manager_consumes_reserved_ready_and_runtime_error_after_control_flood` completed in the unchanged affected Python suite. |
| ON06 | Implemented; locally observed | **Regressions:** `native/overlay/tests/runtime.rs::production_owner_overlay_hidden_reasserts_show_when_desired_visible`, `production_owner_stable_visible_renewals_do_not_arm_due_deadline`, `production_owner_expired_lease_hides_without_reasserting_stale_texture`, `production_owner_event_pump_preserves_idle_hide_tail`, `production_owner_pose_wait_outlives_no_progress_budget_then_handoffs_same_occupant`, `production_owner_preserves_primary_failure_and_reports_hide_cleanup_failure`, and `production_owner_promotes_hide_cleanup_failure_after_successful_shutdown` completed in the 95-test runtime integration binary. Idle, stable-valid, hidden, empty, and temporarily pose-unavailable paths do not consume the due-work no-progress episode unless eligible work is actually stalled; primary and teardown-hide failures remain separately observable. |
| ON07 | Implemented in software/runtime harness; physical visibility not observed | Existing external-hide/show, runtime-event, spatial-reentry, refresh, and translation-update regressions completed in the full native suite. The renderer integration binary completed all 66 tests. These are API/state observations, not headset pixel observations. |
| ON08 | Implemented; locally observed | **New regressions:** `tests/app/test_overlay_process_manager.py::test_owner_health_requires_validated_increasing_challenges_for_sixty_second_refill`, `::test_startup_budget_includes_nonblocking_prepare_and_reaps_late_spawn_before_replacement`, and `tests/app/test_overlay_application_transitions.py::test_terminal_restart_budget_rejects_cap_plus_one_until_qualified_progress`. The exact production protocol/manifest/bridge/process/application/generation/diagnostics/translation/runtime/lifecycle/desktop matrix completed 235 passed with `INTEGRATION=1`. Existing terminate/kill escalation and real-subprocess shutdown ACK selectors completed in that run. |
| ON09 | Implemented; locally observed | **Native regressions:** `runtime::tests::validity_challenge_window_rejects_oldest_at_cap_plus_one`, `::current_scene_lease_expires_and_requests_hide_reconciliation`, `::matching_validity_response_installs_only_current_scene_lease`, and the full-owner `production_owner_expired_lease_hides_without_reasserting_stale_texture` completed. **Python regressions:** bounded health challenge, challenged current status, current-revision lease, and strictly increasing validated refill coverage completed in the affected matrix. |
| ON10 | Implemented; locally observed | **New regressions:** `renderer::cache::tests::bounded_lru_cache_enforces_retained_bytes_at_cap_and_cap_plus_one`, `::bounded_lru_cache_rejects_single_unaccountable_oversized_entry`, and `::renderer_cache_partitions_sum_to_selected_64_mib_budget` each passed exactly. Existing spatial/peer identity, bridge mailbox, audit, and process queue bounds completed in the full native and affected Python suites. |

## OVR-CONTRACT-1 outcome coverage claims

| Criterion | Disposition |
| --- | --- |
| OC04 | Claimed covered by cancellation-only readiness, stale/preempted scene transfer, the fixed original deadline, and current-scene completion selectors listed above. |
| OC05 | Claimed covered at the Windows D3D11 query boundary by the real-query after-enqueue selector and at the owner boundary by retained-producer/successor tests. OpenVR submit-path tests in this environment use `FakeOpenVr`/mocks; no SteamVR compositor or physical HMD ran. |
| OC06 | Claimed covered by idle/empty/hidden/pose-unavailable and deferred spatial-anchor regressions; only due unresponsive work consumes recovery. |
| OC07 | Claimed covered by the actual stopped-stdout/stderr disposable native child, bounded single-current-write watchdog state, generation-associated cancellation, start-relative 25 ms timeout preservation, explicit logger admission close/cancel/bounded join, process-terminal cleanup failure on an unresolved writer/watchdog, nonblocking diagnostic admission, saturating oversize/drop accounting, reverse diagnostic floods, reserved controls, sticky first cause, and finite reader teardown. Log presence is not pass evidence. |
| OC08 | Claimed covered by exact epoch authentication, current-scene replay, bounded validity challenges, current-revision leases, expiry, OFF/restart suppression, and stale callback rejection. |
| OC09 | Claimed covered by the initial-plus-three limit, exact cap-plus-one rejection, strictly increasing bridge-validated current-identity challenges, contiguous 60-second qualifying owner progress, ready-flap and challenge-replay without refill, hung process escalation, and duplicate-child prevention. |
| OC10 | Claimed covered by semantic retirement filtering, 64-entry spatial/peer identity limits, the aggregate 64 MiB accounted renderer-cache limit, bounded audits/queues, and overload/drop reporting. |
| OC11 | Windows release build and direct executable startup-contract smoke check completed locally. Software visibility and API outcomes remain distinct from physical-HMD observation. |
| OC12 | The exact production protocol/manifest/bridge/process/application/generation/diagnostics/translation/runtime/lifecycle/desktop matrix completed 235 passed in 8.44 s with `INTEGRATION=1` and `PYTHONPATH=src`. This includes the three translation startup/restart assertions and the real-subprocess shutdown-ACK selector. |

## Verification evidence

### Default-parallel native suite

```text
CARGO_TARGET_DIR=C:/ovr-target cargo test --locked --manifest-path native/overlay/Cargo.toml
```

Observed after the residual native checkpoint repairs: `298 passed` across six suites (125 library, 66 renderer integration, 95 runtime integration, 12 state integration); no serial override was used. The run includes lease expiry without stale re-show, stable renewal without false due work, owner-loop pose waiting beyond the no-progress budget with same-occupant recovery, distinct primary and cleanup failure reporting, real stopped-pipe handling, retained D3D query ownership, and the bounded diagnostic-writer lifetime.

Exact focused observations:

- Real Windows query cancellation after enqueue: 1 passed, 65 filtered, 0.05 s.
- Persistent incomplete producer with CPU-only successor and bounded escalation: 1 passed in the focused selector.
- Lease expiry/no stale re-show, stable renewal/no false due, idle tail beyond the controlled deadline, owner-loop pose wait and same-occupant recovery, and primary-plus-cleanup/cleanup-only teardown failure reporting completed in the 95-test runtime integration binary.
- Three cache byte-boundary selectors: each 1 passed, 119 filtered.
- Nonblocking diagnostic writer, bounded current-write watchdog state, generation-associated cancellation race, saturating drop counter, actual stopped stdout/stderr pipes, and routing/mode selectors: 9 passed in the focused logging run.
- Actual stopped stdout/stderr coverage retained disposable native children with unread OS pipes; the focused run's child-exit bound remained 2 s per stream.
- Resolved `p05` unit and production-policy selectors: each 1 passed.
- `cargo fmt --manifest-path native/overlay/Cargo.toml -- --check`: completed with no output.

### Python bridge, process, health, validity, and restart

```text
INTEGRATION=1 PYTHONPATH=src python -m pytest -q --override-ini=addopts= \
  tests/core/test_overlay_protocol.py \
  tests/core/test_overlay_manifest.py \
  tests/core/test_overlay_bridge.py \
  tests/app/test_overlay_process_manager.py \
  tests/app/test_overlay_application_transitions.py \
  tests/app/test_overlay_generation_start_owner.py \
  tests/app/test_overlay_diagnostics_port_lifecycle.py \
  tests/app/test_overlay_translation_enabled_sync.py \
  tests/core/runtime/test_overlay_runtime.py \
  tests/app/test_application_runtime_lifecycle.py \
  tests/app/test_desktop_overlay_runner.py
```

Observed after repairing the protocol-faithful translation startup seam and closing-runtime retry callback: `235 passed in 8.44s`. The three previously failing translation selectors also completed directly: `3 passed in 0.84s`.

### Release build and executable smoke check

```text
CARGO_TARGET_DIR=C:/ovr-target cargo build --locked --release --manifest-path native/overlay/Cargo.toml --bin PuriPulyHeartOverlay
C:/ovr-target/release/PuriPulyHeartOverlay.exe --check-startup-contract
```

Release build completed in 7.92 s. Observed startup contract:

```json
{"app_version":"2.6.1","contract_version":7,"execution_contract":{"revision":"r1","version":1},"native_presentation_retry":{"ownership":"exclusive","version":1}}
```

Release binary SHA256: `710a5d6ed576ac6821803c45625da2c4169f24480347c7681e90c39a4392d294`.

## Compatibility, limitations, and rollback

Head-locked and spatial-locked behavior, desktop paths, OpenVR-selected D3D11 adapter validation, accounted renderer-cache policy, and exclusive native retry ownership remain supported. Protocol 6 is not retained: Python and native must deploy as a matched protocol-7 pair.

No SteamVR/HMD session was observed. This record does not certify physical pixel freshness, compositor behavior, driver-wide hang recovery, headset sleep/wake on a particular device, or guaranteed user-visible recovery from a host OS/GPU-driver hang. The real Windows cancellation selector proves D3D11 query ownership and late completion only. OpenVR-facing tests without SteamVR use test doubles. Software reports physical visibility as `not_observable` unless independently observed.

Rollback is paired. Stop application ingress; set overlay OFF; retire publication, connection, process, and device epochs; boundedly terminate and confirm the protocol-7 child exit; restore the recorded prior Python/native/resources tuple; then restart with a fresh epoch and revalidated current state. Never downgrade only one wire peer, replay old retry history, reuse an old validity lease, or launch a replacement while old-child termination is unconfirmed.
