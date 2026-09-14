# Issue 160 completion evidence

## Source identity

- Branch: `listen-add-soniox-speaker-segmentation-and-provi`
- Original implementation branch base from integrated `dev`: `150e3980276f8610ed632d7e04a530f34f0bb15c`.
- Integrated #145 overlay ownership was consumed through #162 source `55a2ac3991ba55173e3e09f3da8b8fa9a8d52c2e`.
- Final product source: `da810cfd2e77ae4a6790c7016aa30763761e7b34`, independently terminal-reviewed against the original integrated base after source-language eligibility and paced-wait cancellation repairs. Later documentation-only identity/architecture clarification does not change the exercised product.
- Designated #158 source: `b4ff4a2b90e3b688a7f85bd775ac620ca108e24a`
- Inherited #159 authority: `LISTEN-ENDPOINT-7S-HYST010`, amendment comment 5653348843
- Overlay contract: protocol 8 / execution contract r2
- Final terminal review exercised 514 affected tests and the five repair-closure cases successfully; Ruff and scoped Black checks passed. The complete suite was not rerun after the terminal repairs; the earlier full run of 6,034 passed / 37 skipped remains historical evidence, not a final-tree full-suite claim.

## Implemented result

Soniox final-token speaker attribution now survives independent final speaker-run normalization with provider-session scope. Language/speaker/unknown changes produce conserved ordered LISTEN children; SELF and manual retain their pre-existing single-transcript segmentation and primary identity. LLM-backed peer translation submits one whole-parent logical batch with explicit child identities and maps responses by UUID. Per-segment language eligibility is evaluated before batching: unsupported segments retain explicit `source_only`/`unsupported_source_language` output, all-unsupported parents make no LLM call, and mixed parents make one request for eligible segments while carrying every segment as whole-transcript context. Batch JSON is user data while the batch response contract is appended to the prepared system request, preserving the configured linguistic prompt, context, and scene semantics. A bounded per-call output allowance is 128 tokens per eligible segment, capped at 4,096, and is propagated through every supported LLM adapter without changing the ordinary one-segment/default request. The common OutputRuntime/OverlayPresenter path fills vacancies immediately and gates logical replacements at 1.0 second, using shared SELF/PEER occupant state, existing expiry, authority, and bounded output ownership.

The retained policy remains latest-conversation-first. Pacing can increase explicit overload retirement. There is no new presentation-wait TTL, no larger queue, and no physical display acknowledgement claim.

## Completion matrix

| ID | Production evidence and observed result |
| --- | --- |
| E01 | Source identities are recorded above: original integrated-`dev` base `150e398…`, #145 ownership integrated through #162 at `55a2ac3…`, designated #158 source `b4ff4a2…`, inherited #159 authority, and protocol 8/r2. The stable pre-terminal-repair source is `3bf12d7…`; no later commit, merge, or deployment identity is claimed. |
| E02 | Soniox provider and normalization regressions cover present/missing IDs, provider-session epoch changes, invalid metadata degradation to unknown, and independence from generic confidence. `SonioxRealtimeSTTBackend` enables diarization only for the LISTEN/peer factory instance. |
| E03 | Translation-turn normalization covers A→B, A→unknown→A, repeated unknown, unknown with language transitions, separated unknown, and punctuation association while conserving normalized text. Speaker-run splitting is now gated to `turn_kind == "peer"`; SELF single/dual-target and manual A→B metadata remain one source transcript with the parent as primary ID. |
| E04 | `test_six_segment_batch_serializes_system_contract_and_bounded_openrouter_budget` drives the real OpenRouter adapter through `httpx.MockTransport`: six segments become one HTTP request, `max_tokens=768`, the customizable prompt remains in the system message, the batch contract is system-side, and the user message is only JSON input. Mixed-eligibility and all-unsupported production-owner tests prove unsupported segments remain explicit source-only outputs, full parent context is retained, eligible segments use one provider request, and an all-unsupported parent uses zero calls. Existing tests cover shuffled UUID mapping, mixed runs, one segment, incomplete/unusable output, source-only, failure, cancellation, and no retry/parallel fan-out. |
| E05 | Pacing is owned by `OutputRuntime`/`OverlayPresenter`, with no provider condition. Common-path tests exercise translated and source-only peer results, no-speaker/single-segment input, Soniox factory wiring, and non-Soniox-compatible output events. |
| E06 | `test_output_presenter_five_ready_peers_follow_two_slot_pacing_schedule` exercises the production output/presenter path with a controlled monotonic clock: two free occupants are admitted immediately, then the three replacements at 1.0-second intervals. The desktop smoke reproduced the same two-slot progression. |
| E07 | One-slot/shared-anchor and same-occupant tests preserve SELF immediate behavior and prevent updates from moving the anchor. `test_protected_rows_are_not_evicted_by_elapsed_pacing_interval` advances the clock ten seconds while a peer waits behind a protected SELF row: no peer replacement timer is scheduled, the peer remains pending, and the protected row remains selected. Close/replay/calibration paths retain their existing non-occupant semantics. |
| E08 | Controlled expiry/state-change tests wake a pending peer when expiry frees a slot and exercise simultaneous expiry/deadline/SELF transitions under the presenter ownership lock. The retained tests assert no over-admission, stale-capacity commit, protected eviction, or expired-deadline spin. |
| E09 | See the dedicated result-order frontier evidence below: later readiness is retained behind a nonterminal predecessor; failure, cancellation, and generation retirement release the frontier; late obsolete output is rejected. |
| E10 | Translation lifecycle tests hold a predecessor at output while a successor translation completes, and destination-isolation tests allow UI/other admitted destinations to progress. Presenter/output locks are released across waits. Pending parent ownership remains the existing one-active-plus-eight-unsent envelope rather than a task-per-caption queue. |
| E11 | Byte-pressure coverage holds one active plus eight unsent 1 MiB parents, reporting `reserved_bytes=9 MiB`. The production pressure regression submits 12 parents while the first is active: depth is exactly active=1, unsent=8, batches=9; parents 2–4 receive explicit `output_overload` as the oldest wholly-unsent parents. After release, only parent 1 and parents 5–12 apply in order, with active=0, unsent=0, batches=0, reserved_bytes=0. No catch-up interval or rerun path is added. |
| E12 | LISTEN generation retirement cancels active/pending output, rejects late completion as `publication_generation_retired`, and preserves selected valid occupants. The bounded cancellation regression cancels a peer during the presenter's replacement deadline wait, observes cancellation of the owned deadline, leaves no additional pending wait tasks, and proves no late `application_applied` receipt. Existing OFF, destination replacement, source change, shutdown, and manual/SELF isolation regressions remain green. Speaker-session scope is metadata only and does not revoke publication authority. |
| E13 | Existing presenter/bridge suites cover original-age expiry, send-time pruning, latest full-scene coalescing, current-state replay, reconnect barriers, and retry without age renewal/history replay. The desktop repair additionally recognizes the established `desktop_first_visible` reverse lifecycle message without changing protocol. |
| E14 | See the real desktop/native smoke below. It used production owners and a running Flet surface, observed the logical 0/0/1/2/3-second schedule, authenticated protocol-8 transport, application receipts, screenshot evidence, and graceful native shutdown. It is software evidence, not HMD physical-exposure certification. |
| E15 | Focused translation/provider/pacing/output regressions, the full Python suite, Ruff, Black, native locked tests/release build/startup contract, and the synthetic desktop smoke are recorded below. No paid Soniox call/private audio/HMD test was performed. |

### Output pressure and waiting dispositions

- Logical pacing wait: no new TTL. A candidate behind protected selected rows waits on presenter state change, not on the elapsed one-second replacement timer.
- Count pressure: one active plus eight wholly unsent parent batches per origin/destination. With 12 parents and a blocked active parent, unsent parents 2–4 terminate as `output_overload`; retained parents 5–12 apply after parent 1.
- Byte pressure: 1 MiB per parent and 9 MiB per scope remain unchanged. Admission diagnostics report active, unsent, batch count, and reserved bytes; the retained regression observes 9 MiB at capacity and zero residue after completion.
- Result-order wait: later results may be ready but do not enter output before a predecessor is terminal. Predecessor failure/cancellation advances the frontier; retired late output remains terminally rejected.
- Latestness remains intentional: pacing may increase explicit overload retirement. There is no unbounded reorder lane, lossless historical replay, catch-up burst, or claim of a fixed maximum caption wait.

## E09 result-order frontier

Command:

```text
uv run --no-project python -m pytest -q tests/core/test_translation_turn_owner.py::test_peer_result_availability_waits_for_predecessor_terminal_and_failure_releases_frontier tests/core/test_translation_turn_owner.py::test_peer_cancellation_releases_frontier_for_new_generation tests/core/runtime/test_output_runtime.py::test_retired_peer_generation_cancels_active_and_rejects_late_output --tb=short
```

Result: 3 passed. The production `TranslationTurnLifecycleOwner` scenario makes the second peer result available while the predecessor output is still blocked/nonterminal, observes no second output until predecessor completion, then emits `first`, `second`. A failed predecessor releases the next result. Channel cancellation releases a new generation while cancelling the old waiting frontier. The production `OutputRuntime` rejects a late retired-generation result as `publication_generation_retired`.

This is distinct from E04's shuffled item mapping inside one LLM batch.

## E14 real desktop/native smoke

The smoke used the production `OverlayRuntimeHandle`, `OverlayBridge`, `OverlayProcessManager`, `DesktopFletOverlayRunner`, `OutputRuntime`, and `OverlayPresenter`; the manager wrote the protocol-8 launch manifest with the live Python owner PID and held its auth/lifetime through graceful shutdown. Synthetic captions only were used.

The first instrumented run isolated a product regression: the real desktop emitted `desktop_first_visible`, but `OverlayBridge` omitted that established process-manager event from `_REVERSE_KNOWN_TYPES`, retired the authenticated connection as `unknown_reverse_message_type`, and caused `runtime_disconnected`. `core/overlay/bridge.py` now accepts that lifecycle event; a regression test keeps the connection authenticated after it. Probe-only faults encountered separately were an unavailable Pillow import and an incorrect probe import; neither implicated product code. One later startup attempt reported the existing recoverable `window_reveal_lost`; a clean retry completed.

Successful command:

```text
PYTHONPATH=src uv run --no-project python .tmp_issue160_desktop_smoke.py
```

Observed real-time logical selections on the live Flet desktop surface:

```text
1.484s  captions 1,2 selected immediately
2.484s  captions 2,3 selected
3.484s  captions 3,4 selected
4.484s  captions 4,5 selected
4.984s  application_applied=5, process state=connected
6.078s  graceful stop: exit_code=0, acknowledged=true, forced=false, cleanup_succeeded=true
```

Durable evidence:

- `evidence/issue160-paced-desktop-smoke.jsonl` — production owner/process/bridge events and exact pacing observations.
- Local-only `evidence/issue160-paced-desktop-final.png` — real Windows desktop capture visibly showing synthetic captions 4 and 5 in the final two selected rows. SHA-256: `0eaa9c92b9820eae4f395fe4aac46df09311aa3e6dd2fc8511342b613c701953`. Not committed because the transparent overlay capture includes unrelated desktop content.

The successful smoke proves application-owned logical admission and actual desktop transport/rendering. It does not claim exactly one second of physical exposure, lossless intermediate transport, HMD visibility, or deployment certification.

## Known pre-existing shutdown race

Independent review observed intermittent failure of
`tests/app/test_overlay_process_manager.py::test_forced_shutdown_preserves_earlier_runtime_terminal_cause`.
The reported cause is the existing approximately 10 ms shutdown/startup task-registration race in
`process.py`. This is architecture drift outside issue #160's approved translation/pacing scope and
was deliberately not repaired. It is recorded separately from the valid prior full-suite result and
from the post-repair runs below; a future owner should make the process-manager shutdown ordering
deterministic rather than weaken or delay the test.

## Final automated verification

Prior integrated Python suite:

```text
PYTHONPATH=src uv run --no-project python -m pytest -q --tb=short --junitxml=.agents/specs/prd/evidence/issue160-pytest.xml
```

Result before the review repairs: **6,029 passed, 37 skipped, 0 failed, 0 errors; 6,066 total**, 186.297 seconds in JUnit / 187.59 seconds wall time.

The first post-repair full run failed one affected test because its `_DeterministicLLM` test provider still parsed the former instruction-prefixed batch body and did not implement the new optional output-budget contract. That test provider was migrated to accept `max_output_tokens` and parse the JSON-only user body; its focused production-composition scenario then passed. This was a review-repair integration failure, not the separate pre-existing process-manager race recorded above.

Complete Python suite before the bounded terminal-review repairs:

```text
uv run --no-project python -m pytest -q --junitxml=.agents/specs/prd/evidence/issue160-postrepair-pytest.xml
```

Result: **6,034 passed, 37 skipped, 0 failed, 0 errors; 6,071 total**, 187.237 seconds in JUnit / 189.04 seconds wall time. The intentionally cleaned-up JUnit file is not a retained artifact. The reported process-manager race did not occur in this run; this does not make that independently observed intermittent race a passing architectural guarantee.

Focused review-repair verification ran the translation request/turn owners, every affected supported LLM provider suite, managed Gemma runtime, LLM semaphore, Soniox channel wiring, protected-row pacing, and 12-parent output pressure: all passed in 4.31 seconds. The formerly failing multilingual production-composition scenario passed separately in 1.03 seconds.

Bounded terminal-review repair verification:

```text
uv run --no-project python -m pytest -q tests/core/test_translation_request_owner.py tests/core/test_translation_turn_owner.py tests/core/test_soniox_multilingual_release_readiness.py tests/core/runtime/test_output_runtime.py tests/core/test_overlay_presenter.py --tb=short
```

Result: all full affected regressions passed in the final 1.89-second run. This includes the production parent all-unsupported/mixed-eligibility scenarios and the output/presenter cancellation lifecycle. The production parent eligibility test passed both cases separately in 0.92 seconds, and the focused cancellation cleanup test passed in 0.64 seconds. No paid provider request was made.

Formatting and lint:

```text
uv run --no-project python -m black <21 changed Python source/test paths>
uv run --no-project python -m ruff check .
```

Black reformatted four files and left seventeen unchanged. Ruff reported `All checks passed`.
The bounded terminal-review repair formatting pass covered its five changed Python
source/test files; all five were unchanged after formatting.

Native commands ran inside Visual Studio 2022 Build Tools Developer Command Prompt 17.14.22 with `CARGO_TARGET_DIR=C:\tmp\ph160-target`:

```text
cargo test --locked --manifest-path native\overlay\Cargo.toml
cargo build --locked --release --bin PuriPulyHeartOverlay --manifest-path native\overlay\Cargo.toml
C:\tmp\ph160-target\release\PuriPulyHeartOverlay.exe --check-startup-contract
certutil -hashfile C:\tmp\ph160-target\release\PuriPulyHeartOverlay.exe SHA256
```

Result: **235 passed, 1 ignored, 0 failed** (lib 129/1 ignored, renderer 94, state 12); release build passed. Startup identity: app `2.6.1`, contract `8`, execution `r2` version 1, native presentation retry `exclusive` version 1. Exercised release executable SHA-256: `697fec416a4a54fa94ac56eae6baf2462709938a9fc4aa91682e8e4ef8888f93`.

## Remaining limits

- No HMD was available; physical HMD exposure/freshness is unperformed, not passed.
- No paid Soniox request or private-audio upload was performed.
- The real desktop smoke used authorized synthetic content and validates the changed software/UI path, not diarization accuracy or provider latency.
