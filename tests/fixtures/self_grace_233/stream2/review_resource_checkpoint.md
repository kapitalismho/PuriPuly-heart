# Independent SELF-STREAM-2 resource/lifecycle review

RESULT: findings

DOMAIN: FULL_REVIEW, focused R1–R5 and coupled R7–R9 resource/source ownership, cancellation, context, and required chatbox-publication guarantees.

Range: `f1ab9a47d9caaf67bd7666b7d5f5e380da46b185..5087173cb3f8fe7b1312c59337e5ec7f41e6b81e`. Initial HEAD was the declared candidate. Authority read first: issue #233 SELF-STREAM-2 and `docs/architecture.md`. No other reviewer's conclusions were consulted. No product/test/maintained-document/Git/remote mutation was made.

After the Director disclosed uncommitted product/test changes, the final three novel probes used an in-memory import loader reading all six affected modules from `git show 5087173`: paginator, translation diagnostics, output projection, and the three changed Qwen test-helper modules. The combined pinned-definition subprocess exited 0: F1 reproduced; owned-Qwen continuation/repetition and saturated cancellation observations passed. Earlier worktree probes are exploratory only. The disclosed product delta was first-successful-page diagnostic tracking and retention of pending latency on send failure; no Self/turn/request resource semantics changed.

## F1 — High: a queued changed correction is falsely terminalized before transmission

Contract: R7 requires an earlier sealed-source correction to retain its normal first-page transmission opportunity before newer speech establishes the paginator watermark. A queued replacement is not a failed/superseded destination operation.

Locations:
- `core/osc/chatbox_paginator.py:98-103`: every true return from `_prepare_self_turn_enqueue` now notifies `output_superseded` for the incoming message.
- The unchanged helper at `:397-412` also returns true after successfully inserting that incoming replacement into `_pending_messages`; true means handled, not necessarily rejected.
- `core/runtime/output.py:637-643,798-813`: the false receipt is retained; the terminal snapshot becomes settled and is removed from the speech guard without any successful first-page receipt.

Reachable path: unrelated long-caption pagination is active; A's preview queues behind it; A gets a changed current-source translation while queued; A closes; B computes a preview before the paginator reaches A. The incoming A replacement remains queued but is reported superseded. Final promotion releases the guard. B's newer-turn enqueue prunes the unsent A correction.

Original executed observation, exit 0, importing this worktree's real Self/turn/request/projection/output/paginator owners with only controlled provider/sender and clock:

```text
after A seal: packets=[xxxxxxxxxxxx], pending=[en:alpha changed], stream guard records=[]
after B preview: packets=[xxxxxxxxxxxx], pending=[en:bravo]
next page opportunity: packets=[xxxxxxxxxxxx,xxxxxxxxxxxx,en:bravo], pending=[]
```

A never transmitted a first page. B was ordinary successor speech, not an intervening manual publication. The initial long pagination was not a later manual supersession exception.

Counterevidence: real-owner successful-preview dedup, failed-send retry opportunity, and queued-identical promotion tests exist (`tests/core/test_self_stream_pressure_233.py:247-340`), but none replaces a queued preview with a changed revision. They can pass with F1. Computation remains concurrent; this is not an execution-serialization defect.

Required repair direction: distinguish successful queued replacement from incoming rejection in the existing paginator receipt boundary, and retain A's guard until a genuine send/disposition. Add pending-changed-A/ready-B coverage without weakening ordinary pagination supersession/manual bypass. No repair applied here.

## F2 — High: discontinuity leaves an unadmitted onset's continuation permanently unresolved

Contract: R2 requires cancellation/expiry/scoped reset to release the matching acoustic obligation, with bookkeeping bounded by actual source/request ownership. It must not strand earlier usable source.

Reachable conditions are exactly the Director-reported remaining failing Qwen source-gap scenario: A has usable stable source; A and B occupy the two admitted Qwen tasks; a third spoken turn enters the serial dispatcher and waits for request capacity; source discontinuity aborts that ingress.

Conclusive source path:
- `SelfTranslationChannelOwner.handle_vad_event:263-268` records the third onset before awaiting downstream ingress at `:325-327`.
- Unchanged `ScopedRecognitionEngine.handle_owned_vad_event:713-724` waits for capacity before `_handle_start` creates its provider identity at `:1002-1007`. The third onset therefore is not in `_turns`.
- Discontinuity forwards to `ScopedRecognitionEngine.abort` (`:399-405`), which emits terminals only for admitted `_turns` (`:846-868`): A and B, not the third onset.
- Unchanged `_GenerationGuardedVadSink.handle_stream_input:199-206` then aborts the serial worker. Its `abort:312-327` cancels the waiting dispatch, drops queued chunks/end, and terminalizes the audio ledger only.
- The Self observation consequently has no matching provider terminal, usable contribution, delivered end, or watchdog: it remains the initial zero-chunk observation with no timeout. `_continuation_blocker:1767-1769` rejects every later source closure, including legitimate fresh-epoch source. `SelfTranslationChannelOwner.handle_stream_input:241-243` merely forwards the boundary and does not retire that acoustic state.

Decisive external evidence supplied by the Director: after preserving real source-closed observation, 20 of the original 21 failures resolved, but `test_real_self_vad_gap_settles_admitted_stable_and_fresh_tail_reaches_source_and_translation` still failed; the migration worker observed only the manual source marker and no sealed SELF. This review did not rerun their reported failure. The static path explains that outcome without assuming harmless source regrouping. The subsequently assigned repair owner independently reported the exact surviving acoustic identity activation=1/order=3/epoch=1 with zero chunks, no end, no native request/terminal, while A+C+tail remained a draft. This corroborates the orphan-observation diagnosis; it is not an executed repair or a passing acceptance claim.

Impact: admitted A and legitimate fresh C/tail may preview but cannot reach canonical context/final completion. This is a real source-lifecycle defect, not just an old UI-event wait. Required repair direction: settle the interrupted, not-yet-request-admitted onset at the actual ingress cancellation/discontinuity ownership boundary, without clearing unrelated obligations or adding provider cancellation/replay. Preserve negative UNADMITTED_B fencing and positive fresh progress/native-origin assertions. No repair applied here.

## COVERAGE

Paths below are under `src/puripuly_heart/` unless marked tests.

- R1 identity separation: inspected — acoustic identity, scoped publication UUID, one draft parent generation/order, per-target operations/results, and presentation revisions remain distinct. `core/orchestrator/self_translation_channel.py:604-689,1660-1765`; `translation_turn.py:543-790`.
- R2 per-acoustic continuation/bounds: inspected plus novel owned-ingress Qwen execution for matching-terminal/false-start cases; **F2 identifies the reachable cancellation/discontinuity ownership hole**. Other matching contribution/terminal removes only its acoustic key; all unresolved observations gate closure; newer false starts preserve older watchdogs; resets cancel observations. Retained-audio/request ownership otherwise bounds the map, but F2 leaves an observation after that ownership is gone. `self_translation_channel.py:1376-1400,1481-1534,1583-1658,1767-1774`; unchanged `core/stt/scoped_engine.py:1778-1800,2075-2140`; `providers/stt/qwen_audio.py:220-239,292-296`.
- R2 consumed-prefix/suffix, actual terminals, genuine repetitions: inspected — seal does not retire the recognition ledger; suffix publication identity remains separate and matching terminal retires bookkeeping. Novel Qwen observation retained repeated source and two native origins. `self_translation_channel.py:604-689,740-748`; unchanged `core/stt/backend.py:152-231`. Existing suffix test `tests/core/test_self_translation_channel_owner.py:669-723` forcibly commits the prefix and alone is not real-ingress acceptance proof.
- R3 single draft-to-sealed lifecycle: inspected — compatible operations survive source revisions/seal; incompatible meaning/provider/configuration retires old work and fences completions; same parent closes exactly once. `translation_turn.py:543-727,841-902,1951-2012`; `self_translation_channel.py:1817-1876`.
- R3 ready/running primary+secondary: inspected — seal keeps the operations map; secondary eligibility remains post-end-grace/no-blocker rather than every early revision. Meaningful replacement tests: `tests/core/test_self_translation_low_latency.py:1407-1699`, `test_self_stream_233.py:54-109`. Not independently rerun as a suite.
- R4 source close independent of LLM: inspected and observed — source close has no primary RUNNING guard; saturated-resource probe sealed 11 sources with three provider operations physically active. Fixed-schedule comparisons `tests/fixtures/self_grace_233/stream2/run.py:221-251` were inspected, not rerun. Source-configuration barrier/handoff consumers were inspected; exploratory real-handoff probes did not establish a suspected cross-scope-blocker failure, so it is not a finding.
- R5 two canonical / one speculation / eight waiting: inspected and executed — drafts do not enter canonical waiting selection/start TTL; seal sets admission time; occupied speculation is excluded from ordinary waiting expiry/eviction until conversion or valid completion. `translation_turn.py:641-679,792-850,1779-1811`. Full occupancy observation below exercised 2/1/8 without duplicate invocation or premature permit release.
- R5 physical provider/native cleanup: inspected, controlled cancellation-suppression support — current/retiring operations are awaited before occupancy release (`translation_turn.py:889-900`). Unchanged semaphore/racing provider retain permits through tracked cleanup: `core/llm/provider.py:98-118`; `fallback_racing.py:291-309,341-369`. Actual native-engine hardware cancellation was not exercised by this reviewer.
- R5 actual prepared versus canonical context: inspected — draft preparation snapshots actual context/prompt/config/scene without remembering source; ordered seal remembers source once per target; adopted output keeps original result `context_texts`, not seal metadata. `self_translation_channel.py:782-841,977-993,1817-1839`; unchanged `translation_request.py:444-536,659-866`; `translation_turn.py:641-657,853-871`. Existing intervening-manual-context/one-invocation test `test_self_stream_pressure_233.py:405-445` inspected.
- Failure/fallback request budgets: inspected — one generic canonical fallback; identical Index failure carries terminal failure identity to avoid renewed HTTP attempts; changed source/config/provider keeps its own request identity. `translation_turn.py:853-862`; unchanged `translation_request.py:719-729,783-866`; migrated assertions `tests/core/test_index_self_request_budget.py:129-329`. No additional retry/hedge machinery found.
- Generation/OFF/shutdown: inspected and executed for explicit reset/OFF; **F2 separately breaks source-discontinuity cancellation progress** — logical cancellation fences output while physical cleanup remains; speech reset preserves manual/Peer scope. Production teardown closes capture/ingress/turns before output/channel/ASR/LLM. `translation_turn.py:1077-1177`; `self_translation_channel.py:192-219`; unchanged `app/wiring/wiring_runtime_pipeline.py:332-455`; `app/adapters/self_capture/self_capture_provider.py:42-71,108-125`. Existing repeated-cancellation-suppressing regression `test_self_translation_channel_owner.py:175-264` inspected.
- Coupled R7: inspected plus original F1 reproduction — first-page success, queued promotion, target-set fingerprint and failed-send opportunity are otherwise represented separately. No all-target/all-pages execution barrier found. F1 breaks the required earlier correction opportunity.
- Coupled R8: inspected — source-close remains active while applicable target work is pending; terminal overlay closure is separate; older results do not retake newer overlay/dashboard state; retained row updates are identity-aware; accepted logs remain final-result-only. `translation_output_projection.py:1305-1406,2788-2901`; `ui/event_projection.py:63-123`; `ui/event_dispatch.py:452-471`. Actual GUI/HMD was not exercised in this focused code review.
- Coupled R9: inspected — preview first-page/final-send diagnostics are separate; identical promotion records preview reuse rather than a fabricated packet; receipts disclaim physical acknowledgment. `translation_diagnostics.py:369-418`; `core/runtime/output.py:637-664,737-764`. F1 also reports an incorrect destination disposition.
- Removed tests: inspected — private speculative-attempt layout assertions generally have real lifecycle/source/result replacements for exact/boundary reuse, meaningful revision, secondary prestart/fallback, provider/config replacement and cancellation. The broad Qwen failures remain an unmigrated observation boundary, not proof those guarantees passed. F1's pending-changed-revision guarantee was not covered by the old/new positive cases.

## Original observations and limits

### Novel actual Qwen continuation/provenance probe — final exit 0

Real SelfCaptureSessionOwner serial dispatch, owned VAD, LocalASR provider/runtime, ScopedRecognitionEngine, Qwen adapter, callbacks and translation owners. Only websocket transport, synthetic audio, LLM and sinks were controlled. Canonical admission was observed through a forwarding owner callback retaining transcript origins, not displayed text.

1. A terminal during active prebuffer B: source `[echo]`, one B obligation, no canonical admission after grace.
2. B ended unresolved then C false-started: one B obligation and no premature canonical admission.
3. Matching actual terminals: exactly one canonical `echo echo`, two distinct recognition-origin unit IDs, no unresolved obligations/open source. LLM inputs were `echo` then `echo echo` for the meaningful revision.

Retained failed exploratory setup receipts: first probe expected a third websocket instead of Qwen's two-slot reuse (timeout). Second prematurely sent C task-finished before observing its matching finish-task (`qwen_unexpected_terminal`, timeout). The final probe coordinated that control boundary and exited 0. These are probe/observation errors, not product findings or hidden passes.

### Saturated occupancy and cancellation-suppressing provider — exit 0

```text
saturated: active provider calls=3, canonical slots available=0,
           waiting parents=8, speculative parents=1, source open=false,
           provider permits available=0
logical OFF before physical completion: reset done=false, active calls=3,
           available permits=0, total invocations=3, sent packets=[]
after physical release: active calls=0, provider permits=3,
           canonical slots=2, total invocations=3,
           terminals=11 / unique children=11 / all exactly once,
           sent packets=[]
```

The controlled provider replaces native work; this is not native hardware reclamation timing.

### Director-reported integration failures

Director supplied `artifact://169`: 2060 passed, 21 failed, 3 warnings. Those failures were not rerun. Subsequent Director evidence reports 20 resolved through observation migration, with the real source-gap failure remaining and diagnosed as F2. Many Qwen helpers wait for legacy `TRANSCRIPT_FINAL`/`_handle_transcript` source output after low-latency source publication switched to `SELF_TURN_UPDATED`. Canonical child transcripts still carry merged origins; presentation snapshots do not. Migration must retain ordering, contribution ranges, native provenance, exactly-once acceptance and negative/fresh-progress assertions, not substitute any equal-text snapshot.

No whole-suite/build/formatter/linter/type-check, real paid/cloud provider, physical OSC/VRChat receipt, GUI or HMD pass is claimed here. Novel probes ran via the worktree interpreter with explicit worktree imports and `-B -c`, synthetic in-memory state and controlled transports, not temporary repository scripts. Context-manager/harness teardown closed owned work; no probe scripts remain.

Director decision: adjudicate F1 and F2 and assign repairs/regressions plus bounded repair review. This candidate does not satisfy R7's earlier-correction opportunity or R2's matching-cancellation progress guarantee. Integration acceptance and broader completion-evidence auditing remain with their owners.
