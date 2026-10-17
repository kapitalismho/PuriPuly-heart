# SELF-STREAM-2 integrated verification

Authority: [issue #233, SELF-STREAM-2](https://github.com/kapitalismho/PuriPuly-heart/issues/233). Implementation baseline: `f1ab9a47d9caaf67bd7666b7d5f5e380da46b185`. First reviewed checkpoint: `5087173cb3f8fe7b1312c59337e5ec7f41e6b81e`.

## Initial integrated executions — failed, not final acceptance

The first command began before the first source freeze was invalidated. Source/test repairs landed during that execution. Its failures are retained observations; neither its passing count nor its displayed source lines establish an exact final-candidate pass.

```text
.venv/Scripts/python.exe -m pytest tests/core tests/ui tests/architecture tests/app -o addopts="" -q
```

Observed: **6337 passed, 9 failed, 9 skipped, 256 warnings**, 752.68 seconds.

- Two `test_self_ui_isolation.py` cases still required the source buffer to stay open while translation ran. That expectation is explicitly superseded by R4. Replacement evidence must preserve source/overlay/provider progress under blocked UI delivery, source closure, and retention of the same running provider work.
- Three failures in output streaming/branch coverage were the earlier asynchronous-destination observation and overlay-failure accounting defects. Their separate post-repair 183-case execution is recorded in `regression_migration.md`; this initial broad execution is not relabeled as passing.
- `test_first_successful_page_keeps_latency_after_earlier_page_failures` exposed a real diagnostic regression: successful later pages lost the existing first-success summary. The original assertion remains unchanged. See `FIRST_SUCCESS_REPAIR.md`.
- Three native vocabulary cases waited for the retired managed-SELF final-event path. Migration must retain native configuration, scope retirement, stale rejection and fresh-source consumer guarantees.

```text
.venv/Scripts/python.exe -m pytest tests/config tests/providers tests/cli tests/domain tests/scripts tests/release_evidence -o addopts="" -q
```

Observed: **2060 passed, 21 failed, 3 warnings**, 441.13 seconds.

The failures were in `test_qwen_audio_delivery_recovery.py` (12), `test_qwen_audio_peer.py` (2), and `test_qwen_audio_runtime_quarantine.py` (7). Migrating real managed-source observation resolved 20, not all 21. The remaining source-gap case exposed a genuine unadmitted acoustic-onset obligation stranded after discontinuity. Its correct source remained `A+C+tail`, but it could not seal; the original expected source was not weakened. Exact source-cancellation authority, native provenance, and both negative stale-input and positive fresh-source progress require repair verification.

## Independent checkpoint findings

The original independent reports are retained in `review_whole_checkpoint.md` and `review_resource_checkpoint.md`. Both reviewed the immutable baseline-to-`5087173` range; decisive custom probes loaded pinned source rather than silently using concurrent worktree edits.

Accepted material defects:

1. A changed queued chatbox correction received a false superseded receipt and could be erased by successor speech before transmission.
2. A source discontinuity stranded an onset cancelled before native request admission.
3. A late managed speech result could replace a newer manual dashboard turn.
4. Actual accepted secondary-target logging used its child identity and created a duplicate conversation row beside the parent-target snapshot row.
5. Secondary failure could hide an otherwise valid current-source primary dashboard translation.
6. Physically completed failed speculation remained exempt from ordinary waiting admission and expiry.
7. A later first successful chatbox page lost the existing latency summary after earlier page failures.

## Repair checkpoint and follow-on verification

The first repair checkpoint is `8eba7512bf6c50fdb01024fd536bada0351aca14`. Subsequent fixture-only commit `c848fa2fd18d8a0b7c99faeb8c74a9a41d16d4d9` pins `gui/owner_presentation.py`, which the committed UI regressions already imported and the working-tree checks used. That commit did not change working source or test bytes. A clean `8eba` checkout alone lacked this required test dependency; its omission is retained as review finding RV1, not hidden behind successful working-tree runs.

The bounded reviews in `review_whole_8eba_repair.md` and `review_resource_8eba_repair.md` verified the initial queued-correction, source-gap, resource, child-identity, partial-target and original latency failure closures. They returned **repair_failed**, not approval, because two related cases remained:

- A failed preview's later successful page lost its preview provenance and was incorrectly counted as final latency while source was open.
- Older manual completion could overwrite newer managed speech: the opposite direction of the repaired mixed-path dashboard race still lacked shared current-slot fencing.

The complete non-integration execution ran with the frozen `8eba` product bytes and the existing presentation fixture, subsequently pinned by `c848`:

```text
.venv/Scripts/python.exe -m pytest tests --ignore=tests/integration -o addopts="" -q -rs --durations=10
```

Observed: **8455 passed, 2 failed, 9 skipped, 256 warnings**, 1126.89 seconds. Both failures are `test_real_capture_gap_bypasses_blocked_fifo_retires_both_slots_and_new_source_progresses` (`known_loss-self` and `epoch-self`) in `tests/providers/test_qwen_audio_recovery_boundaries.py`: its `BoundaryRuntimeSink` test adapter lacks the new `handle_source_segment_retired` protocol method. The production source path must not gain a compatibility fallback to conceal this missing test-adapter migration.

The nine skips require real subprocesses/process ownership: two desktop-overlay-runner cases, one GPU-worker-process case, two overlay-runtime cases, two Windows-process-ownership cases, and two Flet-desktop-view-process-owner cases. They are not counted as passes or replaced by unrelated native observations. Warnings concern dependency Python/deprecation compatibility and existing test-side deprecations; no suppression was added.

Checkpoint `48c13784b690fd9a9c12dc6149eed10e42a345f4` retains preview phase metadata through an earlier page failure and completes the capture fixture protocol cutover. Checkpoint `b3995754cd7f63e1ca7bf446f0eb46c00bdd1366` adds shared consumption-time/pending-source authority. Relevant executions were 151 paginator/diagnostic cases, 117 native/capture fixture cases, and 323 UI/output cases; their owner reports preserve commands and boundaries.

The resource continuation returned **repair_verified** at `b399`; see `review_resource_b399_repair.md`. The whole continuation returned **repair_failed**; see `review_whole_b399_repair.md`. It closed the missing fixture, preview-phase and original reverse-completion failures, and directly exercised queued cancellation, OFF/reset, expiry and valid successor progress. Its new exact-pin observation exposed a delayed second language segment of an older admitted recognition parent reclaiming the dashboard after newer managed speech completed. That checkpoint therefore still required a source-claim repair; the architecture guarantee was not weakened.

The 22-stage native owner run `gui/evidence_owner_repairs_06/` matches all 565 product files in `b399`, despite its recorded execution HEAD being `c848` plus the then-uncommitted repairs. Original pins are retained. Raw `dashboard_status` values in some manual stages refer to hidden label values, not displayed labels: native screenshots 07 and 18 show no stale status. This is an observation-schema clarification, not an established visible product defect or permission for an incidental renderer change.

Static checks at `b399`: Ruff over `src tests --exclude tests/fixtures/self_grace_233` passed; the real-owner helper passed its separate Ruff check; Black reported all 55 changed Python files unchanged. The complete non-integration suite finished with frozen `b399` product/test bytes: **8464 passed, 0 failed, 9 skipped, 256 warnings in 1091.38 seconds**. See `integration_b399.json` and its retained output. This passing suite does not close the independently discovered source-claim defect.

A separate exact-`b399` probe established a visible legacy-status failure, recorded in `review_whole_b399_status_addendum.md`. With the real four-slot UI queue blocked, a manual source receives `destination_write_timeout` and no accepted handoff. After recovery its valid current translation displays, but the previous managed translating label remains visible because only the skipped source setter hides it. This differs from the correctly hidden values in native stages 07/18. The approved repair resets managed status at both legacy display setters; it does not reject the valid translation, add retries, or invent a manual lifecycle.

The same checkpoint's native partial-target capture also established two related conversation-rendering defects: failed Japanese work omitted its target identity, and a valid current-source English result inherited the aggregate failure label. The target/header and target/status addenda retain those original observations and the narrow repair criteria.

## Final presentation repair candidate

Candidate `27e93797dd9ab8b6a2a7969eac73e409ccaffcda` repairs all three follow-on surfaces without changing source policy, provider ownership, admission, retries or logging keys. Only original unparented sources can establish fresh pending presentation authority; both legacy display setters clear managed status; conversation target-work headers and current-ready partial-success labels use existing per-target facts.

- **2160 passed, 0 failed, 2 skipped, 43 warnings in 95.48 seconds**: every UI and architecture test plus affected SELF, projection, turn/request, dual-target, output, paginator and diagnostic suites. Exact command and observed result excerpt: `integration_27e93797.json` / `.txt`. The two skips still require real process ownership. This is not a new complete non-integration run; the separately attributed `b399` breadth result remains 8464 passed / 9 skipped.
- Ruff passed for maintained `src/tests` excluding historical task fixtures and separately for the three current GUI probes. Black left all 57 maintained/current-probe Python files unchanged.
- The whole reviewer returned **repair_verified**, directly executing 28 asserted cases from exact Git-archive imports, not a pytest collection count. It closed the segmented-source, actual source-timeout/status and target-rendering findings with valid-progress, accepted-history, stale/terminal and manual-isolation counterchecks. See `review_whole_27e93797_repair.md`.
- The resource reviewer independently returned **repair_verified applicability** at the same pin. No resource/source/paginator mechanism changed; its earlier 13/17-case and owner-smoke evidence retains its original attribution. See `review_resource_27e93797_applicability.md`.
- Native `gui/evidence_owner_repairs_07/` completed 32 owner-driven stages and both actual Logs-mode interactions. Importing entry and exit HEAD are exactly `27e93797`; all 565 product Python/JSON files and current helper/runner bytes match. The reviewer independently reconstructed the same product manifest from Git blobs. The Director inspected target-pending/secondary-first/partial-failure, segmented source/history, genuine source-timeout recovery, successor status and Logs-toggle pixels.
- Task-local `.gitattributes` retains raw evidence/probe bytes without changing product newline policy. `evidence_byte_retention.json` proves the 107 historical artifact changes are CRLF/LF-only; the independently reviewed current GUI helper is the sole substantively modified existing fixture in that intermediate delta. No historical measurement was rerun or relabeled.

Final runtime `_04` consolidation also completed on unchanged `27e93797` product bytes: both six-episode local CPU-ASR runs, actual CLI operations and the correlating summary command exited 0. All 556 product Python files matched entry/exit and across runs. Actual loopback packets, running-work handoff, overlay-independent progress and effective CLI state are recorded in `audio/RUNTIME_RESULTS.md`; native ASR is final-only and physical C10 remains unperformed. The independent terminal evidence audit is still separate. Original failed check outputs remain in `integration_initial_core.txt`, `integration_initial_providers.txt` and `integration_8eba.txt`, with export identities and path substitutions in `retained_check_outputs.json`.

## Observation boundaries

- Runtime audio, native GUI, deterministic owners, and integrated suites have separate records and source pins. A successful precursor run remains evidence at its recorded content; it is not automatically promoted across behavioral repairs.
- Local sender acceptance, local packet reception, actual app presentation, and physical VRChat/HMD visibility are distinct observations.
- Physical C10 remains **blocked/unperformed**: no isolated authorized VRChat/HMD session was established. The issue must remain open while that required evidence is absent.
- Live cloud-provider integration tests, paid services, private audio, and user-session OSC destinations were not authorized or used.
