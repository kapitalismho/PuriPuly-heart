# OVR-NATIVE-ACCEPTANCE

## Status and authority

**HISTORICAL SOFTWARE ACCEPTANCE: `9867b819afb2d26d3e8cfbc09f4de83f815f8fde`. Subsequent live measurement exposed per-update flicker and a nonzero shutdown exit that the harness incorrectly marked passing. Repair source `afd46cd31edf8115fbab99deb303c7e145c8b3f3` is implemented and locally verified; independent FAST checkpoint findings were adjudicated and repaired, with repair verification and complete-Goal review pending. PHYSICAL-HMD FRESHNESS AND DEPLOYED-PAIR CONFORMANCE ARE NOT CERTIFIED.**

The prior receipt below remains historical evidence, not proof that the later counterexamples passed. The current repair does not authorize push, merge, deployment, release, issue closure, or another live run before operator readiness.

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

## Live counterexample and retained-display/shutdown repair

The work began from clean `5e93eeb734053804675f427b833faafa6d0c8098` on the same branch, 13 commits ahead of its upstream and none behind. Issue #149 remained open and its Project status was confirmed `In progress`. The approved r1 contract and protocol 7 are unchanged.

### Retained historical observations

- `run-live-d1114ceb.json` failed startup before captions. The initial snapshot-before-control handshake repair was independently reviewed through `5e93eeb734053804675f427b833faafa6d0c8098`.
- `run-live-48efb081.json`, under the OS-temporary stage `ovr-measurement/20260911T132538Z-ecede971`, completed nine injected events and 30.005 seconds of true input idle. Its child exited 1, despite the original report's software `pass`. The raw report remains immutable; `assessment-live-48efb081.json` records that shutdown was not accepted.
- The operator reported mild flicker on every caption update: “자막이 갱신될 때 깜빡임을 느꼈어” and “갱신될때마다. 깜빡임이 아주 심한건 아니었고”. This is qualitative HMD evidence, not measured latency.
- Source inspection found revision changes invalidating the displayed lease and causing Hide/Show while matching validation was pending. This is consistent with the observation, not an instrumented proof of its physical cause.
- A controlled loopback/subprocess probe reproduced premature Windows termination at exit 1 when normal child exit was delayed at least 50 ms. Runtime close cancelled the writer before the manager's graceful request. The historical live exit's exact cause remains unknown because the old report retained no terminal diagnostic tail.

### Repair boundaries

| Boundary | Implemented result |
| --- | --- |
| Native displayed authorization | Already displayed pixels may remain only for the same post-frontier-pruning occupant set and their original unexpired leases. A new revision requires its own matching lease before any render/handoff. Retention neither copies the texture nor extends a deadline. |
| Invalidation and status | Actual expiry, OFF/empty, occupant-set changes, frontier invalidation and epoch teardown do not retain unauthorized pixels. Pending revision status distinguishes displayed authorization from current-covered handoff; retained old pixels cannot qualify supervisor refill. |
| Native orderly completion | Successful owner teardown emits the existing identity-bound `shutdown_complete` event before logger shutdown. Failed owner teardown emits no success ACK. Exit codes and wire/status vocabulary are unchanged. |
| Python teardown | Semantic ingress stops first; owned transport/readers remain available through manager shutdown. One inclusive graceful request/ACK/exit deadline precedes terminate/kill escalation. Failed cleanup retains unresolved resource ownership and the first terminal cause. |
| Measurement decision | One runtime-owned close replaces duplicate premature broadcasts. Pass requires confirmed exit 0, ACK, no forced exit, no terminal cause, successful cleanup and manager state off. A bounded public shutdown receipt preserves lifecycle events and hashed stderr diagnostics without raw caption/secret text. |

No new root owner, backend, buffer, profile, retry default or compatibility path was introduced. `ARCHITECTURE.md` clarifies retained pixels versus authorization to write a new revision; no suspected architecture drift was identified.

### Applicable local evidence and matched pair

- Native pre-fix `production_owner_retains_displayed_same_occupant_until_revision_validated` failed because the retained gap reported an invalid lease. Post-fix it passes and observes no hide/show, clear, new submission, current-covered handoff or false due episode before matching validity; replayed older validity does not submit.
- `production_owner_true_expiry_during_retained_revision_gap_hides_without_clearing` passes: expiry during the gap hides without stale re-show or a texture clear. Identity coverage includes reordered two rows, row removal, replacement and semantic-frontier pruning. ACK coverage distinguishes successful resource teardown from cleanup failure.
- Director integration: `CARGO_TARGET_DIR=C:/ovr-target cargo test --locked --manifest-path native/overlay/Cargo.toml` passed **303 tests**. OpenVR-facing scenarios use test doubles; real Windows D3D/logging regressions remain in this suite.
- Director integration: the Python matrix listed below plus `tests/scripts/test_ovr_hmd_measurement.py`, with `PYTHONPATH=src INTEGRATION=1`, passed **251 tests in 21.93 seconds**. It includes real delayed synthetic subprocess ACK/exit 0, request failure without premature termination, nonzero/missing-ACK/forced/unconfirmed outcomes, terminal drain and retained cleanup ownership.
- Native release build and actual `--check-startup-contract` passed: app 2.6.1, protocol 7, r1 version 1, exclusive retry version 1. Native/Python product source is `8231d93d370aff4a081ea1da0d964a0f8155ebb3`; subsequent harness metadata records this source separately from the historical accepted source.
- Rebuilt executable SHA256: `f747db2496a74e48dac4950ab0ff9c369a9d343d9c01816530e8aa2382d47a1d`. Vendored OpenVR 2.15.6 DLL SHA256 remains `bab8ac6ef64e68a9ca53315b0014d131088584b2efdfa6db511d67ec03cfcb4a`.
- Isolated preparation: `ovr-measurement/20260911T141618Z-fdf91bef`. Actual CLI offline dry-run passed and retained `run-offline_dry_run-b36a6c67.json`; it exercises the presenter/bridge sequence, not a live native owner or physical display.
- The rerun preserves `ov01-short-r2`, nine injected events, p05/basic, three-second readable holds and at least 30 seconds of true input idle. As in the prior stage, only the executable and DLL are staged; the font bundle is absent, so this is not installed-resource parity. Prior live diagnostics observed system-font fallback.
- Python Black check and Ruff check passed for all six modified Python surfaces. Throwaway synthetic smoke scaffolding was removed; external evidence and immutable measurement reports are retained.

### Independent checkpoint adjudication and repair boundary

Two independent FAST checkpoint reviewers covered the complete integrated candidate `5e93eeb734053804675f427b833faafa6d0c8098..5b7f859f42e0984462ef02ac1659dff10a1228ea`: native authorization/persistent-surface lifetime and Python transport/shutdown/measurement truth. They completed their coverage and confirmed the principal repairs, including an independent real runtime/bridge/presenter/manager/subprocess shutdown probe: one shutdown control, ACK plus delayed exit 0, no force. The pre-fix copy instead lost the writer and terminated at exit 1. Neither review established the historical live exit's cause or physical flicker elimination.

| Finding / observation | Director disposition |
| --- | --- |
| Python F1: inferred reader completion | ACCEPT. Actual process exit is now separate from positive bounded reader settlement. Cancellation/unresolved readers retain ownership; returncode alone cannot certify cleanup. |
| Python F2: forced exit labelled native nonzero | ACCEPT. Forced escalation retains its own cause, without overwriting an earlier primary runtime/startup cause. |
| Python F3 / native F2: ambiguous worktree source | ACCEPT. The product pin is explicitly `python_source`; `native_source` and historical `accepted_source` remain separate, and the exact executing measurement script SHA256 is recorded. |
| Python F4: real transport regression gap | ACCEPT. The synthetic task-name assertion was replaced by a real bridge/manager/delayed-subprocess shutdown regression that observes one control, ACK and exit 0. |
| Native F1: ACK only observed after the graceful deadline | REJECT as a requirement to accept late observation as normal completion. The three-second inclusive contract is retained. A late ACK is recorded as evidence but cannot set `graceful_completed`; positive reader cleanup is covered by the accepted Python F1 repair. |
| Native F3: Python 3.12 sub-tick test budgets | DEFER_OUT_OF_SCOPE for code changes. Two pre-existing ten-millisecond startup-budget tests fail with the repository Python 3.12.10 clock resolution and pass under Python 3.14.0. Current evidence explicitly uses Python 3.14.0; no whole-environment pass is implied. |
| Native O1: expiry crosses calibration boundary | ACCEPT. Post-calibration expiry hides immediately and returns before any render, readiness wait, handoff or empty-scene tail. A deterministic boundary regression passes. |
| Native O4: new failure reasons normalized to unknown | ACCEPT. Existing application normalization and all five locale bundles preserve the four new shutdown reasons. No new error framework was added. |
| Native O5: avoidable identity lookup allocations | ACCEPT narrowly. The bounded two-row lease checks borrow existing identity strings. |
| Native O2 / O3 and freshness precision | O2 REJECT as a production-unreachable undrawable block under existing presenter admission. O3 is the explicit r1 tradeoff: old same-occupant pixels can remain only until their original lease expires. Freshness episodes resume only before their original deadline; elapsed episodes expire, never renew or count as completed. |

The complete repair source is `afd46cd31edf8115fbab99deb303c7e145c8b3f3`. Director integration passed **304 native tests** and **262 Python tests in 21.36 seconds**, using Python **3.14.0**, `PYTHONPATH=src` and `INTEGRATION=1`. The Python matrix above additionally includes `tests/scripts/test_ovr_hmd_measurement.py`, `tests/ui/test_desktop_overlay_i18n.py` and `tests/ui/test_i18n_key_usage.py`. The final count excludes a newly added provenance field-copy test and pre-existing literal-wording/constant-only assertions removed from the touched localization test; functional translation-key and error-consumer coverage remains.

The rebuilt executable passed the actual startup-contract CLI with unchanged app/protocol/r1/retry values. Its SHA256 is **`aa0b258e816ff810ff3b9816aeeb31bc25d912498885bb1723750b7206b8c2dc`**; the DLL remains `bab8ac6ef64e68a9ca53315b0014d131088584b2efdfa6db511d67ec03cfcb4a`. Latest isolated stage: **`ovr-measurement/20260911T145038Z-1c0ea718`**, actual offline CLI receipt **`run-offline_dry_run-49cfd469.json`**. Earlier stages and raw reports remain retained, not overwritten or relabelled as current evidence.

The public localization runtime resolved all twenty combinations of four new shutdown reasons and five locales in a throwaway smoke call, exit 0. This verifies displayed-text resolution, not visual desktop rendering. The measurement CLI still uses the same sequence/profile and operator-ready live guard. No native/SteamVR/HMD live rerun occurred.

Checkpoint repair verification and fresh complete-Goal review will assess this stable repaired candidate. Physical flicker elimination, an observed SteamVR shutdown of the rebuilt pair and installed font-resource parity remain unverified; the next live run requires operator readiness.

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
