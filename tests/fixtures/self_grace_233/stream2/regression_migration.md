# #233 SELF-STREAM-2 regression migration

## Scope and candidate

- Baseline: `f1ab9a47d9caaf67bd7666b7d5f5e380da46b185`, plus the shared task's in-progress candidate changes.
- Owned changes: `tests/core/test_self_translation_low_latency.py`, `test_translation_output_streaming.py`, `test_translation_owner_branch_coverage.py`, `test_channel_runtime.py`, `test_self_translation_channel_owner.py`, and obsolete helper/import removal in `tests/helpers/translation_owners.py`.
- No product source, domain events, historical experiment/audio/GUI artifacts, maintained documentation, Git index/history, or remote mutations by this worker.
- Controlled boundaries: synthetic text/VAD events, controllable local fake LLM providers and output sinks. Actual SELF/turn/request/output owners perform the work. These are deterministic software behavior checks, not ASR, GUI, UDP-receiver, VRChat, HMD, or real translation-model observations.

## Migration method and meaningful proofs

Retired private `_MergeBuffer.speculative_attempt` construction was replaced with actual `SelfTranslationChannelOwner.handle_stt_event()` / `handle_vad_event()` ingress and provider release/failure control. Waiting observes sink results, snapshots, provider invocation records, or actual owned timers; no mock replaces the changed turn owner.

The drain/retirement test's callback observer calls the real `on_draft_updated` callback before signaling its completion barrier. Destructive cases wait for the controlled provider to return an actual cancellation-suppressed late result and for retirement completion before asserting destination rejection. This avoids turning an unobserved, not-yet-completed result into a false no-output pass.

Preserved/consolidated proofs include:

- Source-first overlay output and same logical caption/publication identity through source revision and final promotion.
- Ready and running exact/approved-boundary reuse, including multilingual punctuation, with one provider invocation and retained result update ID through promotion.
- Sticky preview provenance is retained until a current result arrives; a distinct current result re-emits with its new update identity even when the visible translated text is unchanged.
- Meaning-changing question/number/negation/content revisions invalidate old work and publish only current results.
- Translation-relevant configuration invalidates work; output-only changes preserve compatible calls.
- Generic speculation failure falls back once after grace; secondary prestart failure remains silent and receives its bounded final fallback.
- Pending/confirmed continuation obligations, false starts, watchdogs, manual input, recognition units, and existing Peer behavior remain exercised by the assigned existing suites.
- Distinct requests repeating the same English or Korean text are preserved both within one source unit and across separate equal-text publication identities.
- An ended unresolved predecessor's actual watchdog survives a newer false start; false-start settlement cannot release the predecessor's continuation obligation.
- Two targets start at the preserved eligibility point, allow secondary-first progression while primary remains pending, and publish combined results in configured order without duplicate calls.
- Replacement providers fence old results for ready/running primary/secondary work, including source already closed.
- Source closure does not await provider completion; manual/successor computation is not held behind prior speech execution.
- Sticky overlay translations retain result provenance while a revised source is pending; the actual semantic final clears an incompatible old translation after current translation failure.
- Final provider language metadata survives promotion, including a blank translation's detected source language.
- Late acoustic end after prefix/suffix watchdog publication retains diagnostics provenance and the required older chatbox emission opportunity.
- Owned draft shutdown is exercised with a provider that observes actual cancellation, rather than a fabricated private speculative task.
- Admission close positively preserves valid already-admitted drain while rejecting new STT input; destructive speech reset and turn-channel close fence actual cancellation-suppressed late provider results.
- Drafts do not enter accepted context; canonical context is remembered once for each existing language-pair history (dual-target histories intentionally have one entry per target, not a duplicate entry in the same pair).

## Deleted mechanical cases / obsolete mechanisms

- Removed `_SpeculativeAttempt` / `_SpeculativeAttemptStatus` imports and `make_speculative_attempt()` from the shared helper, and removed obsolete `run_self_spec_translation()` forwarding adapter. All callers on assigned surfaces were migrated or removed. A repository-wide `tests` search for both retired helper names found no remaining callers, including outside the five owned files.
- Deleted branch-only spy tests `test_next_action_evaluator_starts_failed_fallback_once` and `test_maybe_restart_spec_replaces_previous_task_and_state`; these asserted private forwarding/state layout while replacing real owner behavior. Their meaningful failure/restart invariants are covered through providers and output above.
- Deleted the active-overlay mirror-field dataclass-layout test.
- Retired private evaluator spy variants that only checked `_commit_merge` invocation on exact/boundary/excluded punctuation; behavior is now asserted through actual provider reuse/restart and consumer payloads.
- Removed old speculative-state copy/status/sequence assertions and unused fixture constructors/diagnostic stubs. Retained source-timer/runtime cleanup assertions where those owners still own that work.
- Consolidated forged-result commit/diagnostic variants into reachable reuse, result-identity, sticky-caption, and actual semantic-final behavior proofs instead of asserting the new implementation layout.

## Explicit superseded expectations and preserved content policy

1. Director approved separating the historical SELF-GRACE-1 comparison from the maintained suite. The maintained low-latency test file no longer imports/runs `tests/fixtures/self_grace_233/experiment.py`; its 53 schedules × 4 historical policies are retained historical evidence, not current acceptance. The historical-evidence owner is adapting that separate runner against pinned historical source. This worker did not change retained fixtures.
2. Old source-close blanking/semantic-final expectations are superseded by R8. Source closure keeps the pending active caption protected and sticky. A separate reachable failure test preserves clearing at the actual semantic final.
3. Old scoped late-end test expected a newer suffix to suppress the queued earlier prefix and emit only one summary. R7 now requires prefix first-page opportunity before suffix; the migrated test observes both local sends at the same controlled clock and two corresponding latency summaries.
4. The old RUNNING-buffer-open dependency is not retained: closure is observed while provider work remains pending and compatible operations continue without duplicate invocation.
5. Dual-target content formatting is **not** superseded. Read-only `git show f1ab9a47...:src/puripuly_heart/core/orchestrator/translation_output_projection.py` established `_publish_self_snapshot()` uses `transcript_text=""`, `include_source=False` even when configuration `chatbox_include_source=True`. Migrated dual-target fixtures explicitly retain `True` and still require translation-only newline payloads. An interim attempted `False` fixture normalization was reverted before the next run; core repaired the actual regression.
6. Director inspected production shutdown order (`RuntimePipelineResourceOwner.close`: ingress → turns → output → source channel, with application shutdown likewise closing turns first) and approved replacing the old fabricated Self-owned speculative-task assertion with actual composed start/stop cancellation. Direct `Self.close()` no longer owning provider operations is an ownership-contract migration, not a proven product defect. Bare `Self.close_ingress()` stops new admission, not already-admitted drain: historical callbacks also did not use this flag as publication-retirement authority. A separate parameterized test positively preserves accepted drain/reopen/final reuse and requires destructive speech reset/turn-channel close to fence actual cancellation-suppressed late results.


## Executed checks and interim outcomes

All five assigned files were executed together; no unrelated repository-wide validation, formatting, or type-check claims are made.

1. First migration run: **161 passed, 14 failed, 175 executed** (counts from pytest progress markers). Its PowerShell-style environment prefix was rejected by the available shell; pytest nevertheless executed the current editable checkout. This is retained diagnostic evidence, not the explicit-PYTHONPATH acceptance run. Failures included stale private fixture references/UI expectations, dual-format content regression, sealed replacement fallback, blank-result handling, final language metadata, and late-end diagnostics/order.
2. Explicit `PYTHONPATH=src`, after migration/core repairs: **172 passed, 4 failed, 176 executed**. Three failures were a newly added incorrect text-only dual-history assertion (existing contexts are per language pair); repaired to preserve exactly one canonical entry per target. One failure was a test reading nonexistent `SelfTranscriptFinal.secondary_text`; repaired to observe the actual preceding blank-active event.
3. Next explicit run: **175 passed, 1 failed, 176 executed**. Remaining failure exposed missing revised-source-only chatbox fallback after a usable old preview and failures of the current speculative/canonical requests. The actual semantic-final overlay clearing assertions passed. Reported to core and Director without weakening the fallback assertion.
4. Strengthened identity/repeated-speech/shutdown subset: **11 passed, 1 failed, 134 deselected**. The failure required direct `Self.close()` to cancel work now owned by the turn owner. Director approved migration to the actual composed shutdown boundary, rather than adding cross-owner task ownership merely to satisfy the probe.
5. Composed shutdown plus initial closed-ingress callback probe: **1 passed, 1 failed, 24 deselected**. Composed cancellation passed. The late provider callback after `close_ingress()` actually enqueued `pending (late translation)`. Initial classification as a defect was withdrawn after Director compared historical behavior and drain authority; the observed packet is a preserved accepted-drain fact, not a product violation.
6. Corrected drain/destructive-retirement, watchdog, and composed-stop subset: **4 passed, 1 failed, 71 deselected**. Destructive reset and turn-channel close correctly fenced an actual late provider result; older unresolved watchdog and composed shutdown passed. The remaining test expected new STT admission to be silently ignored, but actual preserved ingress rejects it with `RuntimeError("Self translation ingress is closed")`; repaired to assert this explicit rejection while preserving the accepted-drain result.
7. Repeated focused subset after that assertion repair: **5 passed, 0 failed, 71 deselected**. Accepted drain and explicit new-ingress rejection, destructive late-result fencing, older-watchdog preservation, and composed shutdown all executed and passed.
8. Equal-/changed-payload sticky-result identity subset: **2 passed, 0 failed, 72 deselected**. Both current results with changed text and distinct current results with unchanged visible text emitted their new update IDs while preserving the old result ID on the interim sticky caption.
9. Final migrated five-file collection: **183 tests collected**, with no collection errors. Command was the all-five-file command below with `--collect-only` instead of `--tb=short`. Collection is not an execution/pass claim.
10. After separating the physical-result return barrier from logical retirement and explicitly waiting turn-owner idle: **3 passed, 0 failed, 25 deselected** for `test_draft_drain_and_destructive_retirement_are_distinct` in the Self owner file.
11. Meaning-sensitive content/question/number/negation pairs: **4 passed, 0 failed, 44 deselected** in the low-latency file.
12. First core-released product freeze, full five files: **180 passed, 3 failed, 183 executed**. New bounded asynchronous overlay handoff exposed two obsolete observation barriers (overlay preview did not mean chatbox enqueue had completed; source processing must no longer remain blocked behind the sink). Those were migrated to independent destination completion and positive source/STT/provider progress while the sink remains blocked. The third failure concerned missing overlay-failure diagnostic accounting.
13. Full rerun launched just before core invalidated that freeze: **182 passed, 1 failed, 183 executed**. Waiting for the actual overlay failure diagnostic timed out; the meaningful failure-accounting assertion remains intact. Core diagnosed a real missing post-delivery `destination_publish_failed` → `record_overlay_sink_failure` path in the new asynchronous handoff, not merely an observation delay. This run is retained as invalidated/interim evidence, not final stable-candidate verification.
14. Restored core product freeze after failure-accounting repair, full five files: **183 passed, 0 failed, 0 skipped, 183 executed** in 3.60 s. This is the final combined execution for this bounded migration. Candidate content fingerprints are recorded below.


Reproducible command:

```text
PYTHONPATH=src .venv/Scripts/python.exe -m pytest tests/core/test_self_translation_low_latency.py tests/core/test_translation_output_streaming.py tests/core/test_translation_owner_branch_coverage.py tests/core/test_channel_runtime.py tests/core/test_self_translation_channel_owner.py -o addopts= -q --tb=short
```

The final focused drain/retirement/shutdown subset was run with:

```text
PYTHONPATH=src .venv/Scripts/python.exe -m pytest tests/core/test_self_translation_low_latency.py tests/core/test_self_translation_channel_owner.py -o addopts= -q --tb=short -k 'draft_drain_and_destructive_retirement_are_distinct or new_false_start_keeps_older_unresolved_watchdog or composed_shutdown_cancels_and_awaits_draft_provider_work'
```

## Product defects reported to core

- Source-closed running work lost required replacement-generation fallback: repaired by core; subsequent assigned run passed.
- Final overlay used configured language instead of retained translation metadata: repaired by core; subsequent assigned run passed.
- Blank/whitespace result produced source plus blank parentheses: repaired by core; subsequent assigned run passed, including detected source-language metadata.
- Late-end inherited timeline retention: repaired by core; subsequent assigned run passed after R7 emission-order migration.
- Dual-target payload incorrectly included source under `chatbox_include_source=True`: baseline policy established and core repaired; subsequent assigned run passed.
- Current source-only failure fallback after an older usable preview: repaired by core; the retained behavior assertion passed in the final combined run.
- Bare-ingress callback observation: **not a product defect**. The negative assertion was unsupported; corrected tests distinguish accepted drain from destructive retirement. No product change was requested to suppress valid admitted work on admission close alone.
- Bounded overlay handoff failure accounting: repaired by core through the existing actual-delivery failure callback, without replay/retry. The retained completion-aware diagnostic assertion passed in the final combined run.

## Invalidated frozen-candidate fingerprints

SHA-256 captured for the 182-pass/1-fail intermediate candidate before core's overlay-failure-accounting repair. These identify the failed candidate, not the eventual final one:

```text
a24538c35b157d1c68ac035799d551f68dd89c0e8431cbc19701e7c2a479621c  src/puripuly_heart/core/orchestrator/channel_runtime.py
a213d24d0b3c0a966d79e2afa05362fae210c6be603b9fcf6e6a6b6bb172199f  src/puripuly_heart/core/orchestrator/self_translation_channel.py
d411b9fc777d23f7863ad2fc805690625bf87571395daca69fadd732553112e6  src/puripuly_heart/core/orchestrator/translation_turn.py
de6266596fd1abeb59329033c19bafc9762768116947893a9044422643465733  src/puripuly_heart/core/orchestrator/translation_request.py
2d9ae71ce873a4ef9ad1538f5b5dd44eb0d23cf446a0a461e25fb436b6790580  src/puripuly_heart/core/orchestrator/translation_output_projection.py
6c8cb391fdef79f4baac7205f313d26c4c3edaebc47ee4221427674191fc864a  src/puripuly_heart/core/runtime/output.py
db78438aac5de7f65f8368ca42bb237ad78f045421ca10a99274986554fb6b26  src/puripuly_heart/core/runtime/output_batch.py
933515a6142258ea335f53fd482acbd1512afa10f05d52159642781dc94f4b58  src/puripuly_heart/core/osc/chatbox_paginator.py
06cd314014e897658419ea73dc7f9232244cc2074a3b07efa5d05c36c0eed850  src/puripuly_heart/domain/events.py
cb0bc5e269e25d6d98a02305083641b191f92f0ebf8d2c13627f0cf38de86e0e  tests/core/test_self_translation_low_latency.py
1880fbb330edb3c11e4e3d01a5baba25691916110a42ec59364e81c6b7ef1922  tests/core/test_translation_output_streaming.py
f32fce441178699dc5ce4012cf621e66a98bcbcebf0af47984ad6fac7cc6deab  tests/core/test_translation_owner_branch_coverage.py
6e40cd10388c69615ed6271f9785a8eb473505e717ab06fc40cda4d005466fc9  tests/core/test_channel_runtime.py
6248b0652c05b51eebf53bab525b148243fd1b9c7364c6787e80cb6a54065191  tests/core/test_self_translation_channel_owner.py
7e2c57ee13356f1e0435e0a09c61ffc2e6a5203c7187970479c82013861f007a  tests/helpers/translation_owners.py
```

## Final frozen-candidate fingerprints

SHA-256 captured alongside the successful 183-case execution after core explicitly restored its product freeze:

```text
a24538c35b157d1c68ac035799d551f68dd89c0e8431cbc19701e7c2a479621c  src/puripuly_heart/core/orchestrator/channel_runtime.py
a213d24d0b3c0a966d79e2afa05362fae210c6be603b9fcf6e6a6b6bb172199f  src/puripuly_heart/core/orchestrator/self_translation_channel.py
d411b9fc777d23f7863ad2fc805690625bf87571395daca69fadd732553112e6  src/puripuly_heart/core/orchestrator/translation_turn.py
de6266596fd1abeb59329033c19bafc9762768116947893a9044422643465733  src/puripuly_heart/core/orchestrator/translation_request.py
5f55bd185486a4026e1d26de98f281d11df9b855dfdad624656dd9f73ebe24ce  src/puripuly_heart/core/orchestrator/translation_output_projection.py
ba28b169a1aa522715325dbf96b15341ed9342e9d529ad3443d4407073ba9200  src/puripuly_heart/core/runtime/output.py
db78438aac5de7f65f8368ca42bb237ad78f045421ca10a99274986554fb6b26  src/puripuly_heart/core/runtime/output_batch.py
933515a6142258ea335f53fd482acbd1512afa10f05d52159642781dc94f4b58  src/puripuly_heart/core/osc/chatbox_paginator.py
06cd314014e897658419ea73dc7f9232244cc2074a3b07efa5d05c36c0eed850  src/puripuly_heart/domain/events.py
cb0bc5e269e25d6d98a02305083641b191f92f0ebf8d2c13627f0cf38de86e0e  tests/core/test_self_translation_low_latency.py
1880fbb330edb3c11e4e3d01a5baba25691916110a42ec59364e81c6b7ef1922  tests/core/test_translation_output_streaming.py
f32fce441178699dc5ce4012cf621e66a98bcbcebf0af47984ad6fac7cc6deab  tests/core/test_translation_owner_branch_coverage.py
6e40cd10388c69615ed6271f9785a8eb473505e717ab06fc40cda4d005466fc9  tests/core/test_channel_runtime.py
6248b0652c05b51eebf53bab525b148243fd1b9c7364c6787e80cb6a54065191  tests/core/test_self_translation_channel_owner.py
7e2c57ee13356f1e0435e0a09c61ffc2e6a5203c7187970479c82013861f007a  tests/helpers/translation_owners.py
```

## Initial five-file completion status

The initial bounded five-file migration completed with all 183 collected cases passing on the initial frozen fingerprints above. The later ownership extension and newly discovered source-gap defect are recorded separately below; the initial pass is not a final post-follow-up candidate claim. Historical characterization remains separate. Director owns integrated validation, review, and issue acceptance. This worker made test/helper changes only and did not mutate product sources or publish/commit anything.

## Bounded provider/UI/native-vocabulary follow-up

Director extended test-only ownership after checkpoint `5087173cb3f8fe7b1312c59337e5ec7f41e6b81e` to:

- `tests/providers/test_qwen_audio_delivery_recovery.py`
- `tests/providers/test_qwen_audio_peer.py`
- `tests/providers/test_qwen_audio_runtime_quarantine.py`
- `tests/core/test_self_ui_isolation.py`
- `tests/app/test_self_stt_source_language_runtime.py`

Preserved external evidence, not rerun merely to confirm:

- Director's provider/config/CLI/domain/scripts/release suite: **2060 passed, 21 failed, 3 warnings**; full retained failure output `artifact://169`. The 21 failures waited on old managed-source observations in the three provider files.
- Director's core/UI/architecture/app suite: **6337 passed, 9 failed, 9 skipped, 256 warnings**; retained output `artifact://147`. Three failures were the already-repaired pre-freeze asynchronous/diagnostic cases; two UI tests retained the superseded RUNNING-buffer-open expectation; three native vocabulary provider variants waited for old final observations. The separate first-successful-page diagnostics failure belongs to core.

The provider migration uses a test-local call-through `SelfSourceObserver`: it retains the exact `request.transcript`, including recognition origins, from a managed source-closed callback, calls the real presentation callback, and requires an actual queued `SELF_TURN_UPDATED` snapshot with matching identity/text/source closure before reporting that sealed source. Managed closure revisions are deduplicated by UUID, never text. Legacy `TRANSCRIPT_FINAL` observations remain unchanged for actual unmanaged/Peer/manual routes, including the explicit low-latency-disabled retained-Qwen loopback. A source-closed snapshot proves sealed source only; it is not described as every target's successful final or canonical context admission. Translation completion is separately required through actual current-source ready target results after owner idle, not invocation alone.

Preserved quarantine/provenance guarantees include exact origin counts/ranges and capture epochs; retry counts/backoffs and real native task terminals; unadmitted/pre-gap/expired/retired source rejection in both source and LLM observations; independent manual/Peer progress; provider cleanup-debt ownership; fresh positive source/translation progress; socket retirement ordering, physical socket caps, task reuse, exact PCM, and native vocabulary propagation. No transport/resource/config assertion or timeout was weakened.

The UI isolation migration observes an actual call-through source-closed snapshot after the real grace timer while the original provider remains blocked and the UI queue/bridge remain blocked. It retains source-first overlay identity/appearance, pending target status, one invocation, same operation result after release, and actual translation application without releasing UI backpressure. The native vocabulary test observes its final managed translation by result update ID rather than the retired `TRANSLATION_DONE` event; actual native terms, scope retirement, old-source rejection, fresh source, and connection/PCM assertions remain unchanged.

The F2 producer-authority migration explicitly authors `independent_recognition=True` on the four self-owner test producers that model independent native recognition input, including the repeated-unit proof. Their original assertions and typed identities/settings scopes are unchanged. Generic PCM negative receipts remain unqualified/default-false; this does not equate input delivery, native recognition completion, source closure, or canonical context admission. Core owns the single corresponding peer-owner annotation; that file was not transferred to this worker.

Follow-up executions so far:

1. Three provider files plus the original five: **233 passed, 1 failed, 234 executed**. Twenty original observation failures were repaired without weakening their assertions. The remaining actual source-gap/manual-interleaving case still failed its original `Already admitted A. Fresh C. Exact tail.` source expectation.
2. A discriminating source-consumer failure probe, not a confirmation rerun: **0 passed, 1 failed, 22 deselected**. The observed sealed-source consumer list contained only the independent manual source, not a differently grouped SELF source. The original A+C requirement was retained; further authoritative snapshot/source state was requested before classifying a product defect.
3. Authoritative snapshot/source-state probe: **0 passed, 1 failed, 22 deselected**; full bounded state retained at `artifact://231:raw:16-16`. The exact full A+C+tail text progressed through source revisions 1–3 under one UUID but every snapshot remained `source_closed=False`, with one pending continuation and no grace/await-VAD blocker.
4. Acoustic/native identity probe: **0 passed, 1 failed, 22 deselected**; `artifact://246:raw:16-16`. A pre-gap activation-1/order-3/capture-epoch-1 acoustic start remained pending with zero chunks and no end; native terminals cancelled orders 1/2 and completed fresh order 4, with no native event for order 3.
5. Actual release/settlement call-through probe requested by core: **0 passed, 1 failed, 22 deselected**; `artifact://250:raw:16-16`. Actual `release_input_segment` calls were empty; actual settlement attempts touched orders 1/2/4 only, never the stranded order-3 identity. This is a source-gap obligation-release defect, not justified source regrouping or a translation-observation failure. Exact typed keys and native provenance were sent to core and Director; the original combined-source assertion remains unchanged.
6. Newly assigned UI isolation and native vocabulary files: **28 passed, 0 failed, 2 warnings**. All three Deepgram/Soniox/Qwen native vocabulary replacements completed their original positive and negative native/resource/PCM assertions using the managed final-result observation.

The temporary private-ledger/release probes were removed from test code after their observations were retained above. The maintained helper keeps only a concise expected-versus-observed sealed-source failure message and the actual request/snapshot provenance bridge.

Director-requested formatter work: inspected all hunks of `black --diff` for the 11 owned test/helper files (`artifact://247`). Only four original migrated core files required formatting; every hunk was in a task-modified function or task-created spacing, with no unrelated baseline restyling or assertion/config changes. Applied Black to those four files. After removing transient probes, `black --check` on all 11 owned files passed: **11 files would be left unchanged**. The seven other files needed no formatter mutation.


## Final follow-up verification and candidate

Core restored its formal nine-path F2/F5 source freeze after the decisive original real-Qwen source-gap case passed unchanged: **1 passed, 0 failed in 1.51s**. The final combined ten-file run then passed **262 passed, 0 failed, 2 warnings in 21.04s**. This includes the original `Already admitted A. Fresh C. Exact tail.` assertion, its three recognition origins across capture epochs, the exactly-once manual completion, and rejection of unadmitted B from both source and translation observations. The core-owned typed-receipt forwarding/producer-authority repair resolves the previously observed stranded acoustic obligation on this real integration path; the test was not changed to accept a different source boundary or payload.

Exact final test command:

```text
PYTHONPATH=src .venv/Scripts/python.exe -m pytest tests/core/test_self_translation_low_latency.py tests/core/test_translation_output_streaming.py tests/core/test_translation_owner_branch_coverage.py tests/core/test_channel_runtime.py tests/core/test_self_translation_channel_owner.py tests/providers/test_qwen_audio_delivery_recovery.py tests/providers/test_qwen_audio_peer.py tests/providers/test_qwen_audio_runtime_quarantine.py tests/core/test_self_ui_isolation.py tests/app/test_self_stt_source_language_runtime.py -o addopts= -q --tb=short
```

Final formatter check, including the independent-producer fixture annotations: **11 files would be left unchanged**, exit 0. The only test warnings were Deepgram's Pydantic V1/Python 3.14 compatibility warning and the `websockets.legacy` deprecation warning; neither was suppressed. No Ruff, type check, application-wide suite, or paid/live-provider check was run by this worker.

Final fingerprints were captured during the source hold. The paginator SHA-256 independently matches the F1 owner's frozen value. These supersede the earlier five-file fingerprints for this follow-up candidate:

```text
a24538c35b157d1c68ac035799d551f68dd89c0e8431cbc19701e7c2a479621c  src/puripuly_heart/core/orchestrator/channel_runtime.py
14706d489967e4facf9a2c4933fa15a48333475fbc2f3b227de48dcc7ed3a6f7  src/puripuly_heart/core/orchestrator/self_translation_channel.py
036471d468dda40b3ba4a11747de2876f9e02701fe16b8373ab22ee43123096d  src/puripuly_heart/core/orchestrator/translation_turn.py
de6266596fd1abeb59329033c19bafc9762768116947893a9044422643465733  src/puripuly_heart/core/orchestrator/translation_request.py
bf604602eed642fcf8ae76a75228380582437d2f03ab86b46b56b407ca4a6f24  src/puripuly_heart/core/orchestrator/translation_output_projection.py
6a424daa9f8117854d93460ce34f40b01a7af6617e74f4ef7a87dc86c6ddfe97  src/puripuly_heart/core/orchestrator/translation_diagnostics.py
c7a89f97790ac7aeb6effc934fa262630e706aba0fda900d023920077cfe73b7  src/puripuly_heart/core/orchestrator/translation_channel_callbacks.py
ba28b169a1aa522715325dbf96b15341ed9342e9d529ad3443d4407073ba9200  src/puripuly_heart/core/runtime/output.py
db78438aac5de7f65f8368ca42bb237ad78f045421ca10a99274986554fb6b26  src/puripuly_heart/core/runtime/output_batch.py
17db1807ae40e54ecce3d23f7e82f4f1724701cbfd8232f68fadb5d010527dbd  src/puripuly_heart/core/runtime/self_capture.py
ab0ea6141ab3dcab37392bf1f0ee56e9983f5f8cae2234df16f613db90c47d7d  src/puripuly_heart/core/osc/chatbox_paginator.py
06cd314014e897658419ea73dc7f9232244cc2074a3b07efa5d05c36c0eed850  src/puripuly_heart/domain/events.py
efa5d130268fd864573218ff16bae8de6f20c9055f27cda38eb427468c872fa9  src/puripuly_heart/app/adapters/self_capture/self_capture_vad_sink.py
844e81f1981a928404d20614b0179dc25092641c49c7b0517c1ef4c2fd0d9d73  src/puripuly_heart/app/ports/capture_vad_runtime.py
ecca41551b57965efcb9e0946cb591d7a134ac221639c8892a996709cd806001  src/puripuly_heart/core/stt/backend.py
08596f3d3c5b566ce4cd54df61d5a54de6620418746b5e271f917b98672a1b8f  src/puripuly_heart/core/stt/scoped_engine.py
50de0d4d4baa0c06729647b49fd8a76f8b37bb29b915cc6e3129d3832e3917c6  src/puripuly_heart/providers/stt/gemini_transcribe.py
51db6c56e032b67d4accadf5f285466198509f7e2b0287a28682a65888feca94  tests/core/test_self_translation_low_latency.py
016a5baccfbecd54b4f4d84aeec71b9434d6492fe246618e0ac1945193932571  tests/core/test_translation_output_streaming.py
36fe9f31c2a79e8e43d667ec63a606e30b51fe1307d57d7bf1974d7ca4e663f6  tests/core/test_translation_owner_branch_coverage.py
6e40cd10388c69615ed6271f9785a8eb473505e717ab06fc40cda4d005466fc9  tests/core/test_channel_runtime.py
62bd04dd41283f24b6d02d57f256542fd713eb610ae3296e55d8626f5e7b2684  tests/core/test_self_translation_channel_owner.py
7e2c57ee13356f1e0435e0a09c61ffc2e6a5203c7187970479c82013861f007a  tests/helpers/translation_owners.py
f02e546caa7e79bf23cbf0a34ddbaef44e03b0c4680b31e8760008e134e1f0f8  tests/providers/test_qwen_audio_delivery_recovery.py
11bd71267a80160ad101b938fcabbcdadf3e7733522326724ac67c54d9157045  tests/providers/test_qwen_audio_peer.py
e24e537f2c2a800e19c806d9d52b9a1802c46a81b35187cbe7ab823faac2935a  tests/providers/test_qwen_audio_runtime_quarantine.py
ef886a40ddff06c832849f9fe5abd23201b40dbd5c87cbc6df07e9ee6a43331a  tests/core/test_self_ui_isolation.py
6ef15357c8ae43224e13efc725874c31dedc64e4f8247641255b027701fb1ee3  tests/app/test_self_stt_source_language_runtime.py
```

The bounded regression migration is complete across ten test suites and the obsolete helper. Mechanical retired-layout/history comparisons were removed rather than repinned; meaningful consumer-visible proofs and approved ownership/source-closure migrations are described above. No unresolved product defect remains in this exercised scope. Unmanaged Self/Peer and low-latency-disabled legacy observers remain unchanged; source provenance, identity-based deduplication, and negative quarantine checks are preserved. This worker made no product/domain/event, maintained-documentation, historical-fixture/audio/GUI, or Git mutations. Main retains integrated verification, independent review, and issue-completion authority; no issue-wide completion or new Git pin is claimed here.
