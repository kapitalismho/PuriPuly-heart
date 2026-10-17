# Focused stream resource/source repair verification — final bounded candidate

RESULT: repair_verified

DOMAIN: REPAIR_VERIFY continuation for RV2 failed-preview provenance and directly related resource/source/queue regressions. Original baseline `f1ab9a47d9caaf67bd7666b7d5f5e380da46b185`; original checkpoint `5087173cb3f8fe7b1312c59337e5ec7f41e6b81e`; previous reviewed repair `8eba7512bf6c50fdb01024fd536bada0351aca14`; final frozen candidate `b3995754cd7f63e1ca7bf446f0eb46c00bdd1366`. HEAD returned that final pin before inspection/checks. Read-only: no product/test/Git/remote edits.

The original independent findings and failed-repair proof remain intact in `local://stream-checkpoint-resource-review.md` and `local://stream-checkpoint-resource-repair-review.md`. This is bounded closure, not a new whole-checkpoint approval or issue-completion judgment.

## RV2 original failure and decisive closure

Original 8eba failure: source-open active preview `abcd` failed; later page `efgh` was the first successful local send. The paginator had discarded revision `final=False` metadata on `output_send_failed`, so the later page reported `self_turn_final=None` and produced a false Basic final `chatbox_send` summary before source closure.

The final repair changes only the paginator metadata-retirement condition: `output_send_failed` now retains preview/final provenance while that active message can still paginate, just as `first_page_sent` does. The existing first-attempt failure receipt remains unchanged. Completion, replacement, supersession and shutdown still remove metadata through existing owners. No guard, pagination, throttle, retry, provider-attempt, source-close or physical-permit policy was changed.

Original fresh-process observation on **b3995754**, exit 0, using actual TranslationOutputProjectionOwner, OutputRuntime, ChatboxPaginator and TranslationLatencyDiagnosticsOwner. Runtime module paths were observed under this worktree's `src`. Only the local sender/clock were controlled; input was a typed authoritative ready presentation snapshot.

```text
PREVIEW:
  source_closed=false, status=active
  packets=[efgh]
  page_facts=[{page_index:1, first_successful_page:true, self_turn_final:false}]
  Basic summaries=[]
  first-opportunity receipts=[output_send_failed]
FINAL (explicit source-closed revision at clock 14):
  attempts=[abcd, efgh, abcd, efgh, ijkl]
  packets=[efgh, abcd, efgh, ijkl]
  later facts=(0,true,true),(1,false,true),(2,false,true)
  Basic summaries=["[Basic][Latency] channel=self endpoint=chatbox_send last_speech_to_chatbox_send_ms=5000"]
  remaining latency timelines=0
  first-opportunity receipts=[output_send_failed, first_page_sent]
```

The 9→14 measurement belongs to an actual new final packet, not the earlier preview or an invented later first-page receipt. R7's permitted failure disposition was retained; later preview success did not regain a send-opportunity barrier. The expected structured local-send-failure warning occurred; the probe exited normally and closed its runtime. This is local transmission evidence only, not recipient/physical-display evidence.

## Actual delta and invalidation assessment

Inspected the complete relevant 8eba→b399 delta, not filenames alone:

- **RV2 paginator:** one condition preserving metadata on first-page send failure. Inspected completion/clear/pending-overload/rejected/supersession/shutdown cleanup callsites. Metadata remains bounded by active/pending message ownership.
- **New diagnostics test:** `test_failed_preview_later_success_stays_preview_until_genuine_final_send` exercises real projection/runtime/paginator/diagnostics owners, both continued preview pages, unchanged failure receipt, a genuine final send, exactly one final summary and timeline retirement. The prior original-final-failure and successful-preview-promotion tests remain unchanged.
- **Four capture fixtures:** the required typed cancellation operation is implemented/forwarded instead of omitted. Original native-slot, FIFO bypass, old-result rejection, retained-byte, fresh-source, PCM and capture-active assertions are unchanged; new exact-cancel/no-Peer-receipt and destructive-retirement assertions strengthen them. No source/F5 product semantics changed.
- **b399 shared presentation authority:** inspected `_SelfPresentationAuthority`, source-pending claim, canonical admission, stream snapshot publication, late legacy translation fences and the one explicit parent UUID forwarded by Self._handle_transcript. These affect UI presentation eligibility, not request preparation, canonical source context, provider permits or chatbox admission. OutputRuntime snapshot publication remains unconditional after the overlay/presentation selection. Therefore prior source/F5 coverage remains applicable; queued correction was re-exercised on b399. Whole RV3/UI authority acceptance remains with its assigned reviewer; no independent whole/UI verdict is inferred.
- **Tracked GUI fixture:** it exercises owners for separately assigned GUI evidence; it does not change product bytes or invalidate source/physical-resource review. This review makes no GUI-proof claim from reading that fixture.

## COVERAGE

- **RV2 / coupled R9:** inspected and independently executed original-failure closure, genuine-final progress, no fabricated preview final-summary and no later-page first-opportunity fabrication. Verified within this scope.
- **F1 / R7 queued correction:** retained prior verified closure; both complete and multi-page real-owner correction cases re-executed on b399. Required A-before-ready-B, concurrent computation, final identity/context and normal remaining-page supersession assertions passed.
- **F2 / R2 exact source retirement:** retained the prior reviewed exact receipt producers/required port/adapter/callback consumers and own previous original A+C+tail/negative-identity/PCM/independent/OFF cases (17 passed on 8eba). Those product paths are unchanged except an unrelated UI parent UUID keyword. Re-executed all four real capture-gap boundary variants on b399; exact blocked-onset cancellation, rejected old source, two native-slot cleanup, fresh progress and no Peer source-retirement assertions passed.
- **Independent input route:** prior complete production-constructor/callback audit remains applicable. No new provider-ID inference, generic PCM-as-recognition terminal, source ledger consumption, physical cleanup or generation-policy change appears in this delta.
- **F5 / R5 completed failure:** prior five executed expiry/overflow/fallback/retiring-cleanup cases and physical accounting inspection retained. TranslationTurnLifecycleOwner, request owner, physical provider/race owners and the F5 tests are unchanged in this delta; no new deadline/permit invalidation found. No duplicate request or early permit release was introduced by RV2.
- **Paginator neighbors:** active/pending replacement and stale-watermark behavior re-executed, plus original first-success-after-earlier-failures, managed-final equivalent and successful-promoted-preview negative. All passed.
- **Broader R1–R5 unchanged basis:** retained original identity/context/reuse/admission/shutdown coverage and the previous bounded source/F5 review. No unsupported new full-review or physical-runtime claim.

## Exact-pin checks exercised by this reviewer

Worktree `.venv/Scripts/python.exe -B -m pytest -p no:cacheprovider -o addopts= -q --tb=short`, with `PYTHONDONTWRITEBYTECODE=1` and explicit worktree imports, selected:

1. `tests/core/test_translation_latency_diagnostics_owner.py::test_failed_preview_later_success_stays_preview_until_genuine_final_send`
2. `::test_first_successful_page_keeps_latency_after_earlier_page_failures`
3. `::test_managed_failed_first_page_releases_guard_but_retains_first_success_latency`
4. `::test_promoted_preview_remaining_pages_do_not_fabricate_first_final_success`
5. `tests/core/test_self_stream_queued_correction_233.py` (both cases)
6. `tests/core/test_chatbox_paginator.py::test_newer_active_revision_replaces_all_unsent_pages`
7. `::test_newer_pending_revision_replaces_in_place_behind_system_message`
8. `::test_latest_self_turn_watermark_rejects_stale_messages_after_queue_drains`
9. `tests/providers/test_qwen_audio_recovery_boundaries.py::test_real_capture_gap_bypasses_blocked_fifo_retires_both_slots_and_new_source_progresses` (four variants)

**13 passed in 1.36 s; exit 0.** The separate original-path preview/final smoke above also exited 0. No tests were skipped in this bounded selection. No failing predecessor was rerun merely to confirm the supplied failure. No repository probe files, bytecode or pytest cache were created; no long-lived owned resources remain.

The repair owner's 151-case result and source owner's 117-case migration result were read as supporting prior receipts, not substituted for this candidate's independent check or relabeled as final global acceptance.

## Material limits and remaining Goal obligations

No build/formatter/linter/type-check/repository-wide pass is claimed by this reviewer. The final full non-integration run was still in flight at assignment; its eventual outcome belongs to the Director/integration evidence record. Director-reported 22-stage native-GUI evidence was not independently executed/audited here.

C10 recipient/HMD/actual VRChat visibility is not established by controlled local sender observations. Any unavailable intended physical access remains explicitly blocked/unperformed in the completion index rather than a pass. Final same-candidate runtime/OSC/GUI evidence applicability, required C9/C10 observations, broad-check outcome and the final evidence index/adoption decision remain Director/evidence-auditor obligations. This bounded repair verification does not close #233 or authorize push/merge/release.

No remaining material defect was found in the assigned RV2 continuation. Retain this exact pin for integration and the separately owned RV3/evidence decisions.
