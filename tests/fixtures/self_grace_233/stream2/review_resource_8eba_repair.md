# Focused stream resource/source repair verification

RESULT: repair_failed

DOMAIN: REPAIR_VERIFY. Original baseline `f1ab9a47d9caaf67bd7666b7d5f5e380da46b185`; previously reviewed `5087173cb3f8fe7b1312c59337e5ec7f41e6b81e`; repaired pin `8eba7512bf6c50fdb01024fd536bada0351aca14`. Initial HEAD returned that repaired pin. Reviewed actual product/test delta for F1, F2, adjudicated whole F5 and coupled F6. No product/test/Git/remote mutation performed.

## Remaining material defect: F6 failed-preview provenance (medium)

Locations: `src/puripuly_heart/core/osc/chatbox_paginator.py:72-78,491-531`; `core/orchestrator/translation_diagnostics.py:404-413`.

The first preview-page failure notifies `output_send_failed`. `_notify_delivery` removes that revision's stored `final=False` metadata on every status except `first_page_sent`. Normal pagination still continues. F6 now correctly observes the first later successful page, but `self_turn_final` is `None`; diagnostics treats every non-False value as final `self_chatbox_send`. R9 requires preview-send measurement and final acceptance/final-send summaries to remain distinct.

Original isolated observation on the repaired pin, exit 0: actual TranslationOutputProjectionOwner, OutputRuntime, ChatboxPaginator and TranslationLatencyDiagnosticsOwner; only local sender/clock controlled; authoritative source-open presentation snapshot. First `abcd` failed; later `efgh` succeeded:

```text
source_closed=false, status=active
packets=[efgh]
page_facts=[{page_index:1, first_successful_page:true, self_turn_final:null}]
basic_summaries=["[Basic][Latency] channel=self endpoint=chatbox_send last_speech_to_chatbox_send_ms=4000"]
```

This records final-send latency before source closure. It is not a physical-recipient claim. Existing repaired tests cover failed initial pages of a final and later pages of an already-successful promoted preview; both pass but neither covers this failed-preview branch.

Required direction: retain correct preview/final provenance for remaining active pages after a failed first-page opportunity. R7 may still settle that opportunity under existing destination-failure policy. Do not add a guard wait, retries, request replay, or change first-success accounting.

## COVERAGE and closure evidence

- **F1 queued correction — verified within scope.** Rejected/queued/unhandled preparation outcomes are now distinct; only rejected incoming candidates receive false-candidate supersession. Both real-owner complete/multi-page correction regressions executed/passed. They require current A correction before B, prove B computation/UI readiness and three provider invocations, preserve same-turn/source-revision identity/final context, and permit B to supersede A's remaining pages. No execution-order or OutputRuntime change.
- **F2 orphan onset — verified within scope.** Capture abort returns exact newly cancelled ledger receipts. Only current recoverable Qwen discontinuity forwards via the existing required port/adapter; destructive teardown does not forward. SELF settles a matching full acoustic identity and reevaluates existing grace without inventing provider/consumption terminality or releasing native permits. Original real-owned-ingress A+C+exact-tail case executed/passed with unchanged combined source, three origins across epochs, rejection of unadmitted B, original native cancellations, independent manual output and fresh terminal assertions.
- **F2 negative/neighboring ownership — verified bounded cases.** Executed exact retirement with another active obligation; generic PCM submitted non-completion; repeated independent submitted-before-local-end then scoped successor with old-generation input; and recoverable/abort/concurrent-shutdown capture variants. All passed. Logical source retirement, recognition completion and physical cleanup remain separate.
- **Independent-route expansion — inspected all production producers/consumers.** Grep found all STTProviderInputTerminal constructors in ScopedRecognitionEngine and Gemini plus both callback consumers. Engine derives authority from STTIndependentRecognitionSession.independent_recognition_units, stamping synthesized submitted/failure/cancellation/finish receipts. Gemini's established independent session authors the flag in all four paths. Current capture generation gates SELF bookkeeping; full acoustic identity plus settings scope gates independent observation retirement. Generic PCM remains timing-only; Peer policy is unchanged. No provider-ID heuristic for independent route. Supplied native-producer assertion evidence was inspected, not rerun independently.
- **Whole F5 failed completed speculation — verified within scope.** Once current/retiring operations physically settle, failed/incomplete work releases only its speculation reservation and rejoins ordinary eight-parent waiting under original source-close admission timestamp/deadline. Existing successful-ready adoption remains. Live/retiring cleanup retains its exemption/permits. All five new cases executed/passed: failure before/after deadline, original overflow order, fresh speculation plus one admitted fallback, retiring cancellation-suppressing work followed by failed replacement expiry.
- **F6 original failure — original branch verified, new coupled defect remains.** Unchanged first-success-after-earlier-failures test, managed-final equivalent, and successful-preview-promotion negative all executed/passed. The failed-preview probe above shows incomplete repair coverage.
- **Test migration — inspected actual assertions.** SelfSourceObserver captures authoritative request transcripts at source-closed snapshot delivery, preserving recognition origins absent from presentation snapshots. require_final_source additionally requires matching complete/current ready targets. Original Qwen A+C+tail grouping/provenance/negative/fresh-progress assertions remain. Legacy TRANSCRIPT_FINAL/private transcript-method observation was replaced rather than bypassed.
- **Retained review basis.** Unchanged R1/R3/R4/R5 request preparation, context exactly once, compatible operation reuse, Index budgets, semaphore/race physical cleanup and shutdown retain applicable original review coverage. Expanded F2/F5 source/resource paths are inspected above. Coupled UI authority change was inspected; complete UI/log repair adjudication belongs to its separately assigned reviewer. This is not a new whole-checkpoint approval.

## Executed verification

Worktree interpreter, explicit worktree imports, `PYTHONDONTWRITEBYTECODE=1`:

```text
.venv/Scripts/python.exe -B -m pytest -p no:cacheprovider -o addopts= -q --tb=short
  tests/core/test_self_stream_queued_correction_233.py
  tests/providers/test_qwen_audio_delivery_recovery.py::test_real_self_vad_gap_settles_admitted_stable_and_fresh_tail_reaches_source_and_translation
  tests/core/runtime/test_self_capture_session.py::test_qwen_source_gap_forwards_only_exact_new_cancellations_while_capture_survives
  tests/core/test_self_stream_owned_ingress_233.py::test_exact_logical_source_retirement_preserves_other_active_obligations
  tests/core/test_self_stream_owned_ingress_233.py::test_submitted_pcm_input_does_not_complete_scoped_source_obligation
  tests/core/test_self_stream_owned_ingress_233.py::test_independent_submitted_before_end_repeats_then_scoped_source_handoff_fences_old_input
  tests/core/test_self_stream_pressure_233.py::test_completed_failed_speculation_uses_original_wait_deadline_and_releases_speculation
  tests/core/test_self_stream_pressure_233.py::test_completed_failure_rejoins_eight_waiter_budget_using_original_overflow_order
  tests/core/test_self_stream_pressure_233.py::test_completed_failure_frees_speculation_then_existing_final_fallback_runs_once
  tests/core/test_self_stream_pressure_233.py::test_retiring_native_cleanup_retains_live_exception_until_failed_replacement_settles
  tests/core/test_translation_latency_diagnostics_owner.py::test_first_successful_page_keeps_latency_after_earlier_page_failures
  tests/core/test_translation_latency_diagnostics_owner.py::test_managed_failed_first_page_releases_guard_but_retains_first_success_latency
  tests/core/test_translation_latency_diagnostics_owner.py::test_promoted_preview_remaining_pages_do_not_fabricate_first_final_success
```

**17 passed in 2.70 s, exit 0.** Separately, the novel failed-preview probe exited 0 and exposed the incorrect final summary. Original failed-candidate checks were not rerun merely to confirm supplied failures; the authorized repaired candidate was exercised.

## Limits and handoff

All checks use actual owners with controlled local transport/provider/sender boundaries and synthetic in-memory state. No paid/cloud/native-hardware, physical UDP/VRChat/HMD, or intended-GUI pass is claimed. No new whole-suite/build/linter/formatter/type-check claim. Director-supplied 143/1070/262 repair results are prior evidence, not final-pin acceptance instructions. Final broad offline and GUI/native verification were in flight and are not counted completed here. Probes ran with `python -B -c`; pytest cache was disabled; no temporary repository files or long-lived resources were created.

Director: retain F1/F2/F5 closure; adjudicate F6's failed-preview provenance defect and assign bounded repair/regression plus a new pin for verification. This report is not issue closure, merge authority or a physical-delivery claim.
