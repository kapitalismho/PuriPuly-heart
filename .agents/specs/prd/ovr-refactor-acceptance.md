# OVR-REF — issue #158 implementation and verification record

## Authority and candidate

- Authority: https://github.com/kapitalismho/PuriPuly-heart/issues/158 (published 2026-09-13; no comments at implementation start), pinned `OVR-CONTRACT-1` section 0.1 r2 and narrowed integrated acceptance scope. Protocol 8, execution r2, P05 and cached-rehandoff opt-in remain unchanged.
- Issue reference baseline: `5d481cf10c5608178c797d82132385c247beee2f`. Actual implementation baseline: `17d0150316123295ecddff7bb27e4558efd61f09`, branch `ovr-0-vr-overlay-reliability-program-cross-envir`, clean tree, three commits ahead of tracked origin. Intervening native-exclusive Python retry retirement and preserved-caption expiry rearming are retained, not reverted.
- One integrated Outcome, four non-overlapping implementation workstreams: native; process; bridge; presenter/output. Director owns composition, architecture, shared validation, commits and review adjudication. No new dependency between workstreams; integration after all owners settled.
- GitHub Project item `PVTI_lAHOBFl7T84BdUS3zg6sWiQ` was changed from Backlog to `In progress` and read back successfully. Issue remains open. No push, PR publication, merge, deployment or production installation authorized/performed.
- Implementation commit: [`d3b4b734894c3d38fa8b137c13aba1362402130d`](https://github.com/kapitalismho/PuriPuly-heart/commit/d3b4b734894c3d38fa8b137c13aba1362402130d); reviewed final source: [`aaabd5c93ef67fce46d92e4ae84d4c5815933c38`](https://github.com/kapitalismho/PuriPuly-heart/commit/aaabd5c93ef67fce46d92e4ae84d4c5815933c38). Commits are local only; links become remotely resolvable only after separately authorized push. Current state: **ACCEPTED**, issue #158 engineering/refactor scope only.

## Responsibility map and constraints

`ARCHITECTURE.md`, “Overlay internal responsibility map”, gives the before/after map for every moved ledger, task/process, frame/query, deadline and cleanup operation. Production uses the extracted components, not test-only alternatives.

- Native: retry-episode state/transitions and audit (`retry_episode.rs`), spatial policy (`spatial_policy.rs`), explicit frame progress (`frame_cycle.rs`), status projection (`runtime_diagnostics.rs`). Existing runtime owns resources, scheduling, deadlines and teardown. Renderer/OpenVR adapters and cohesive presentation reducer are not split.
- Python process: `process_runners.py` owns launch/preparation; `process_adapter.py` keeps parsing, bounded queues, trust-origin events and reader settlement together. Manager retains lifecycle policy. Public facade exports remain; private adapter imports/test fakes migrate without new compatibility shims.
- Bridge: `bridge_mailbox.py`, `bridge_session.py`, `bridge_transport.py` isolate state/admission, authentication/health and bounded execution. Facade retains sole writer/heartbeat, connection epochs, unresolved resources and cleanup. Reverse queue and subprocess queue remain distinct.
- Presenter/output: `presenter_acceptance.py`, `presenter_projection.py`, `runtime/output_batch.py` move bounded bookkeeping with transitions. Reducer/expiry and routing/replacement remain with original owners. Local application commit never waits for remote I/O.
- Development root depth remains unchanged because process runners remain at the same directory depth; regression coverage asserts the real root, nested Rust staleness and DLL preparation. `process-read-*` and bridge task registration/classification remain compatible with runtime staged teardown.
- No new top-level policy owner, cycle, lifetime, protocol, cap, timeout, retry value, expiry authority, backend, provider/Audio policy, or physical-display guarantee is intended.

## Environment and before evidence

Windows 11 x64, repository `.venv` Python 3.12.10 / pytest 9.1.1 / uv 0.9.17. Default system Python 3.14 was not used. `uv run --no-project --python 3.12` was confirmed to select the same repository `.venv`. No dependency/lock changes.

Rust: `rustc 1.97.1 (8bab26f4f 2026-07-14)`, `x86_64-pc-windows-msvc`, LLVM 22.1.6; Cargo 1.97.1, locked dependency resolution. All cargo commands below run in `native/overlay` unless a manifest path is specified.

| Baseline boundary | Exact check / result |
| --- | --- |
| Presenter/output/projection | `uv run --no-project --python 3.12 python -m pytest -q --tb=short --override-ini=addopts= tests/core/test_overlay_presenter.py tests/core/runtime/test_output_runtime.py tests/core/test_output_owner_wiring.py tests/core/test_translation_output_projection_owner.py tests/architecture/test_output_routing_ownership.py`: 194 passed |
| Process | `uv run pytest tests/app/test_overlay_process_manager.py -q`: 101 passed |
| Bridge | `uv run --no-project --python 3.12 python -m pytest -q --tb=short --override-ini=addopts= tests/core/test_overlay_bridge.py`: 33 passed |
| Composition/desktop/diagnostics | 15-file consumer matrix: 296 passed, 4 integration-gated skips on clean baseline; existing Flet ElevatedButton deprecation warnings |
| Native | `cargo check --locked && cargo test --locked --test runtime && cargo test --locked --test state && cargo test --locked --test renderer -- --test-threads=1`, `CARGO_TARGET_DIR=C:/temp/ovr158-native-baseline`: check passed; runtime 94, state 12, renderer 66 passed |

Pre-existing environment failure: native default deep worktree target failed OpenVR CMake/MSBuild FileTracker FTK1011 `.tlog` path creation. Short target directories resolved it without source/dependency changes. No baseline behavioral failure was observed in the exercised suites. A validator rerun during incomplete bridge edits produced missing-private-field failures; that moving-tree run is not baseline or final evidence. Final stable suites below pass.

Additional pre-existing check failure recovered during terminal review: the removed
`test_no_new_unmanaged_task_creation_outside_lifecycle_allowlist` already failed at
baseline `17d01503`. The reviewer replayed that baseline test's scanner and baseline
allowlists against source blobs from the same revision: bridge actual 5 versus
allowlist 1; process actual 8 versus allowlist 3; output actual 1 versus allowlist 2.
Thus unexpected deltas were bridge +4/process +5 and the output allowance was stale
by 1 before this refactor. This is an implementation-inventory failure, not an
observed runtime-lifecycle failure. It was absent from the initial focused baseline
matrix, not introduced by the extraction. Retained allowlist numbers/rationale
assertions are historical/stale and no longer enforce source-call inventory.

## Final-source verification before review

### Python production composition

Exact broad command:

```text
.venv/Scripts/python.exe -m pytest tests/core/runtime/test_overlay_runtime.py tests/app/test_overlay_generation_start_owner.py tests/app/test_overlay_application_transitions.py tests/app/test_overlay_session_transition_owner.py tests/app/test_overlay_diagnostics_port_lifecycle.py tests/core/test_overlay_diagnostics.py tests/core/test_overlay_manifest.py tests/core/test_overlay_protocol.py tests/scripts/test_ovr_hmd_measurement.py tests/architecture/test_overlay_session_transition_ownership.py tests/architecture/test_overlay_generation_start_ownership.py tests/architecture/test_desktop_overlay_surface_boundary.py tests/architecture/test_lifecycle_task_guard.py tests/architecture/test_output_routing_ownership.py tests/architecture/test_runtime_pipeline_direct_ownership.py tests/core/test_overlay_presenter.py tests/core/test_overlay_bridge.py tests/app/test_overlay_process_manager.py tests/core/runtime/test_output_runtime.py tests/app/test_settings_projection.py tests/core/test_translation_output_projection_owner.py tests/config/test_public_compatibility_surfaces.py -p no:cacheprovider --tb=short -q -rA
```

Result: **519 passed, 2 skipped**, 521 collected. Both skips are explicitly exercised with `INTEGRATION=1` below.

With environment `INTEGRATION=1`:

```text
.venv/Scripts/python.exe -m pytest tests/core/runtime/test_overlay_runtime.py::test_overlay_runtime_receives_real_subprocess_shutdown_ack_before_reader_cleanup tests/core/runtime/test_overlay_runtime.py::test_runtime_real_bridge_writer_delivers_one_shutdown_before_delayed_child_exit tests/app/test_desktop_overlay_runner.py::test_import_run_desktop_overlay_dispatch_is_provider_secret_and_stt_free tests/app/test_desktop_overlay_runner.py::test_import_preview_dispatch_is_provider_secret_and_stt_free -p no:cacheprovider --tb=short -q -o addopts=
```

Result: **4 passed**. Actual subprocess shutdown ACK precedes reader cleanup; real bridge writer delivers one shutdown before delayed exit; isolated desktop/preview CLI imports remain provider/secret/STT-free.

`INTEGRATION=1`, `.venv/Scripts/python.exe -m pytest tests/ui/test_desktop_overlay_startup.py tests/ui/test_desktop_overlay_renderer.py`: **133 passed**, 266 pre-existing Flet deprecation warnings. Counts from overlapping focused runs are not summed as unique tests.

Additional final focused checks:

- Presenter/output baseline matrix plus `tests/app/test_overlay_translation_enabled_sync.py tests/app/test_overlay_generation_start_owner.py`: **215 passed**. Production-owner smoke applied SELF final, admitted a manual parent to all three destinations, replaced overlay only, observed UI/chatbox obligations survive, completed them and observed zero retained batches before shutdown.
- `uv run pytest tests/app/test_overlay_process_manager.py -q`: **104 passed**. New moved-boundary coverage covers development root/nested Rust staleness, explicit trust origin, and real pipe pressure.
- `uv run pytest tests/app/test_overlay_process_manager.py::test_real_subprocess_pressure_preserves_lifecycle_and_bounded_cleanup -q`: **1 passed**. Real child emitted 2,048 trace records plus ready/shutdown controls; diagnostic loss >0, controls retained, acknowledged normal exit 0, complete readers, manager off.
- Bridge + `tests/core/test_output_owner_wiring.py tests/core/test_dual_target_translation_lifecycle.py`: **56 passed**. Real WebSocket tests `test_overlay_bridge_real_socket_stopped_reader_stops_bounded_and_truthfully` and `test_overlay_bridge_real_socket_initial_snapshot_precedes_pending_controls` pass. Blocked/cancelled owner-chain checks additionally use controlled connection doubles; do not describe those doubles as real sockets.
- Black check on all 17 changed Python/test files: unchanged. Focused Ruff checks pass. Presenter/output compileall and public facade import smoke pass.
- Source-call-count inventory assertion was removed rather than re-pinned to moved implementation lines; its unused scanning helpers were removed. It did not test lifecycle outcomes. Meaningful bounded-cleanup, task registration/shutdown ordering and no-old-generation assertions remain. Remaining lifecycle guard checks: **11 passed**.

### Native, Windows and artifact identity

`CARGO_TARGET_DIR=C:/temp/ovr158-native-work`:

```text
cargo fmt --all -- --check
cargo check --locked
cargo test --locked -- --skip windows_graphics_
cargo test --locked windows_graphics_ -- --test-threads=1
```

Results: fmt/check pass; **280 deterministic tests passed**, **21 Windows graphics tests passed serially** (5 library/backend-layout, 15 renderer, 1 runtime). These are Windows D3D11/resource checks, not physical HMD or OpenVR-compositor certification. Validation-only owner reconfirmed final source fmt/check, listed 301 tests and spot-reran surviving-row production-loop behavior; retained the applicable complete implementer matrix.

Actual native integration-test processes use real WebSockets and the production owner loop with simulated backend/submitter. The following command shape was executed for each name: `cargo test --locked --test runtime NAME -- --exact`:

- `initial_readiness_processes_new_snapshot_before_submit_and_ready`: pass.
- `production_owner_surviving_peer_transition_never_hides_with_delayed_observation`: pass.
- `production_owner_stable_visible_silence_does_not_arm_due_deadline`: pass.
- `production_owner_event_pump_preserves_idle_hide_tail`: pass.

This proves the modeled first-frame/currentness, surviving-row, no-autonomous-expiry-Hide and empty grace behavior, not physical pixels. Existing matrix covers readiness cancellation/delay, current-scene progress under floods, P05 episode bounds and cached-rehandoff accounting.

Release build with `CARGO_TARGET_DIR=C:/temp/ovr158-native-release`: `cargo build --locked --release --bin PuriPulyHeartOverlay`. Actual subprocess `C:/temp/ovr158-native-release/release/PuriPulyHeartOverlay.exe --check-startup-contract` returned app 2.6.1, protocol 8, execution r2/version 1 and exclusive native retry/version 1.

| Artifact | SHA256 |
| --- | --- |
| `C:/temp/ovr158-native-release/release/PuriPulyHeartOverlay.exe` | `453aeec95091562a20ad066225a8f5ba0c6718c27b19600888ad579ba02acb58` |
| `C:/temp/ovr158-native-work/debug/PuriPulyHeartOverlay.exe` | `c3f87ebc13b3bfa4d306a698faaf7c1a5cdbb2172ec87f509c0666f8255e1f0e` |
| `third_party/openvr/win64/openvr_api.dll` | `bab8ac6ef64e68a9ca53315b0014d131088584b2efdfa6db511d67ec03cfcb4a` |

### Launch/package scope

14/14 facade/internal/desktop/main runtime imports passed; static import-graph reachability found all eight new Python modules. Existing public compatibility and development/frozen command-shape tests pass, and isolated desktop entry points were exercised. The alias registry and `build.spec` remain unchanged: extracted modules use static imports, not dynamic discovery.

Actual PyInstaller Analysis/frozen binary was **not run/built**: build dependencies and complete staging artifacts are absent. Static reachability is supporting evidence, not proof of actual PyInstaller collection. This is proportionate moved-module/launch-path validation, not a new installer certification. No production files or installed application were touched. Pyright is unavailable in the resolved environment; it is not represented as passed.

## Review and residual scope

Checkpoint range: `17d0150316123295ecddff7bb27e4558efd61f09..d3b4b734894c3d38fa8b137c13aba1362402130d`. Both fresh read-only specialists verified the exact committed clean candidate and completed assigned coverage.

- Native specialist: **no material findings**. Independently reran fmt, locked all-target check, 280 deterministic tests, 21 serial Windows graphics checks, four production-owner WebSocket probes, release build/startup contract. Covered GPU/query lifetime, currentness/fairness, retry accounting, spatial transitions, status consumers and teardown. An independent build at a different target path differed from the retained release artifact; bit-for-bit reproducible builds are not claimed. The recorded artifact hash and startup result remain the tested identity.
- Python specialist: complete coverage of presenter/output, bridge, process, application lifecycle, desktop/measurement and launch discovery. Independently reproduced 519/2 broad matrix, all four gated subprocess probes, 194 presenter/output tests, 104 process tests and named real WebSocket/pipe probes. No product behavior regression found; two observations adjudicated below.

| Finding | Director disposition | Evidence and rationale |
| --- | --- | --- |
| F1: removal of exact task-call-count inventory guard | REJECT as a required repair; implementation-inventory retirement explicitly retained | The assertion compared per-file source call counts, not managed lifetime, registration, cancellation, cleanup or old-generation behavior. It already failed at baseline as detailed above; extraction also changed call-site locations. Restoring/re-pinning would preserve an implementation-detail assertion contrary to the applicable test policy. Its helpers became unused after deletion. Behavioral cleanup/registration/real subprocess tests remain and pass; no meaningful behavior assertion was removed. This intentionally retires the source-count protection, not runtime ownership. Retained historical allowlist/rationale assertions are stale and outside this focused change. |
| F2: one cold combined run lost the expected earlier `gpu_query_failed` test cause to `shutdown_forced` | DEFER_OUT_OF_SCOPE as an unproven refactor attribution; record observed timing risk | `test_forced_shutdown_preserves_earlier_runtime_terminal_cause` failed once under its 10 ms test shutdown budget. Six immediate candidate combined replays passed (137 tests each); six read-only baseline replays passed (134 tests each). Reviewer verified unchanged deadline/drain mechanism, apart from explicit event-envelope unwrapping. This is not claimed as a reproduced baseline defect or silently erased from evidence. No timeout widening, expected-outcome change or unrelated policy repair was made. Final targeted and broad suites pass; intermittent test timing remains a residual uncertainty. |

Checkpoint review required no production repair. Terminal review of
`17d01503..99d7a2a73a4f0d9dab0af333c12c99df6cefa077` completed all Goal coverage
and returned two bounded findings, both **ACCEPTED** by the Director:

- **R1 — baseline inventory failure record:** corrected above using the terminal
  reviewer's baseline-blob replay. The earlier description did not identify that
  this source-count guard was already failing.
- **R2 — measurement source identity coverage:** added all eight extracted Python
  modules to `scripts/bench_ovr_hmd_measurement.py::RUNTIME_PYTHON_SOURCE_FILES`.
  Existing identity schema and historical native/package pins remain unchanged.
  Each moved implementation now contributes to the measured Python file-set hash.

Repair evidence: `uv run pytest tests/scripts/test_ovr_hmd_measurement.py`:
**25 passed**; Black and Ruff checks on script/test pass. The new regression
`test_runtime_identity_changes_when_extracted_implementation_changes` changes
isolated source-file bytes and requires the reported aggregate identity to change
for each extracted boundary. An in-memory pre-fix list replay reproduced the missing
identity change; the repaired list passed all eight modifications. Temporary probe
directories were automatically removed, with no production-source mutation.

Native and application runtime/test source are unchanged by this repair; their
checkpoint evidence remains applicable. Measurement consumer evidence is refreshed.
Terminal reviewer additionally ran the 10 ms first-cause test 30/30 times green;
the earlier cold-run uncertainty remains recorded rather than reclassified.
Bounded repair verification at `aaabd5c93ef67fce46d92e4ae84d4c5815933c38`
closed R1 and R2 and returned **accepted** for the complete Goal range
`17d0150316123295ecddff7bb27e4558efd61f09..aaabd5c93ef67fce46d92e4ae84d4c5815933c38`.
The reviewer independently reran all 25 measurement tests and script/test
Black/Ruff checks, retained the applicable complete native/Python coverage and
confirmed a clean candidate. Director accepts the integrated Outcome and Goal at
that reviewed source revision. This final receipt update changes documentation
only; no artifact is rebuilt or promoted by acceptance.

All required structural, preservation, integration, Windows/API software checks
and claim-bounded completion-record criteria are satisfied at the reviewed source.
The proportionate package checks and explicitly unperformed/deferred checks retain
their labels above; none is promoted to full frozen-package or physical proof.
No architecture drift beyond the intended subordinate internal boundaries was
identified. No throwaway probe or temporary scaffolding was added to tracked
product source. Issue remains open/In progress remotely because final publication
and issue closure were not authorized.

Physical HMD observation and 4–6-hour exposure remain deferred to user prerelease work; affected/control comparison remains removed. No field-flicker, physical freshness, universal environment, release or installer acceptance is claimed. Historical #148/#149/#151 receipts supply baseline authority only, not automatic validation of the new executable. No changes to #152 protection-policy decisions or other Audio branches are included.
