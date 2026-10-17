# SELF-STREAM-2 source-gap and completed-failure resource repairs

This task-local record follows the Director-integrated checkpoint `5087173cb3f8fe7b1312c59337e5ec7f41e6b81e` and the separate first-success receipt repair. It does not relabel historical SELF-GRACE-1 output as current acceptance. The preceding records and JSON outputs remain intact. Git operations, maintained documentation, publication, independent review, and final integration belong to the Director.

## Observed failures and preserved guarantees

### Resource F2: a pre-admission source onset survived a recoverable Qwen gap

The original owned-ingress integration retained `Already admitted A. Fresh C. Exact tail.` but could not close its source. The stranded acoustic identity was activation 1 / segment order 3 / capture epoch 1, with zero chunks, no end, and no native request/terminal. Qwen's two admitted native tasks were cancelled and a later epoch successfully delivered C and its exact tail; unrelated manual work progressed. Original evidence remains in `artifact://231`, `artifact://246`, and `artifact://250`. The existing source/output/provenance assertions were retained, not regrouped or weakened.

The capture ledger had already cancelled that unfinished identity, but `_GenerationGuardedVadSink.abort` discarded the returned typed receipts. A queued onset is removed from the unsent map before the source owner awaits native admission, so neither the unsent timeout nor a nonexistent native terminal could retire this particular source obligation.

The repair returns the existing `cancel_unfinished` receipts and forwards only newly cancelled full identities through the existing capture sink/port to SELF after a recoverable Qwen source discontinuity, while the capture generation is still current. SELF settles only the matching acoustic identity and reevaluates the existing grace rule. It does not clear an epoch, invent a native/recognition/consumption terminal, replay input, alter source provenance, or change request/native lifetime. Destructive abort/OFF/shutdown does not forward this recovery handoff; a concurrent shutdown also fails the existing generation-current guard.

### Independent input bookkeeping is not scoped source completion

Readback of baseline `f1ab9a47d9caaf67bd7666b7d5f5e380da46b185` confirmed that the generic `release_input_segment` acoustic-settlement loop was task-added, not baseline PCM policy (`artifact://258`). Removing it without preserving independent-recognition bookkeeping would strand local-VAD records whose later end is intentionally ignored after input release.

`STTProviderInputTerminal.independent_recognition` now carries producer-authored route provenance. Existing independent-only ScopedRecognitionEngine and Gemini producer paths author it; the engine also preserves its own established independent route when finishing an input receipt. The actual callback passes the full event and fences old capture generations before current SELF bookkeeping. Only an explicitly independent receipt with the exact acoustic identity and matching settings scope retires independent local-VAD bookkeeping. An unqualified/generic PCM receipt remains timing-only. This bit does not claim recognition completion, native cleanup, source text coverage, or successful source finalization.

This corrects the earlier report's insufficient lifetime account: an entered, unadmitted onset can have no end/watchdog and no native terminal. Its existing capture-ledger cancellation is the missing retirement authority. The map remains tied to real unresolved ingress/source obligations rather than an arbitrary eviction cap. In the composed bounded-Qwen path, the two admitted native turns and one currently dispatched admission await can enter source bookkeeping; queued-before-dispatch onsets have not entered it. Usable contributions, matching terminals/end-watchdogs, exact recovery cancellation, independent-route bookkeeping bypass, and real reset/shutdown retain their separate authorities. Other active episodes and native cleanup debt are not reclaimed by the new handoff.

### Whole-review F5: physically completed failed speculation stayed exempt forever

The pinned review observed two canonical slots occupied, a third sealed live speculation physically failing, and the failed parent still holding speculative occupancy after its original waiting TTL. It neither entered the ordinary eight-waiter budget nor expired; a neighboring ordinary waiter did expire. Original review evidence belongs to `agent://checkpoint-stream-whole` and the Director's preserved record.

Once all current operation tasks and retiring operations are physically settled, successful complete results still use the existing ready-result adoption path. A failed/incomplete result that needs canonical work releases only its original speculative reservation and joins ordinary bounded waiting. It uses the original source-close admission timestamp/deadline, enforces the existing oldest-first eight-pending overflow policy, and expires immediately if that deadline has already elapsed. Live operations or retiring native cleanup preserve their exemption and physical provider permits. Canonical execution still requires a real permit. Existing generic final fallback runs exactly once when admitted before expiry; no new retry policy, attempt budget, slot, scheduler, or source admission is introduced.

## Exact repair product surfaces

All paths below are under `src/puripuly_heart/`:

- `core/orchestrator/self_translation_channel.py`: exact logical source retirement; typed independent-input bookkeeping; generic PCM/source separation.
- `core/orchestrator/translation_turn.py`: completed-failure occupancy conversion, original-deadline waiting/expiry, and closed-parent execution guard.
- `core/orchestrator/translation_channel_callbacks.py`: full typed input forwarding and capture-generation fence.
- `core/runtime/self_capture.py`: existing ledger cancellation receipts returned by abort and forwarded only for current recoverable source gaps.
- `app/ports/capture_vad_runtime.py`: required typed SELF source-retirement operation.
- `app/adapters/self_capture/self_capture_vad_sink.py`: current-owner forwarding of that operation.
- `core/stt/backend.py`: explicit independent-input route field.
- `core/stt/scoped_engine.py`: established independent producer/finish provenance.
- `providers/stt/gemini_transcribe.py`: existing independent producer provenance.

The paginator F1 correction belongs to `stream-chatbox-repair`; UI projection/queue and realtime-log parent metadata repairs belong to `progressive-ui`. Their transferred surfaces were not edited by this repair. The earlier first-success diagnostic repair remains preserved.

## Changed regression surfaces

- `tests/core/runtime/test_self_capture_session.py`: capture sink contract; real ledger cancellation sequencing; recovery versus destructive abort/shutdown; independent fake producer provenance.
- `tests/app/test_self_capture_vad_sink_adapter.py`: required source-retirement forwarding.
- `tests/core/test_self_stream_owned_ingress_233.py`: exact identity rejection, unrelated active-obligation protection, generic PCM non-completion, repeated independent input-before-local-end, independent unit delivery, scoped successor progress, and late old-generation fencing through actual callbacks.
- `tests/core/test_self_stream_pressure_233.py`: failed-before/after-original-deadline expiry; original eight-waiter overflow order; new speculation and existing final fallback recovery; still-physical retiring cleanup followed by failed replacement expiry.
- `tests/core/test_peer_translation_channel_owner.py`: one independent fake-producer provenance annotation; no assertion change or Peer product-policy change.
- `tests/core/test_stt_scoped_engine.py`: real engine independent input provenance assertion across existing successful/failed stream-retirement cases.
- `tests/providers/test_protocol_a_scoped_sessions.py`: real Gemini submitted/failed/cancelled/repeated-client producer provenance assertions.

Transferred self-owner test fake annotations and the original Qwen integration remain owned by `stream-regression-migration`.

## Executed verification

Environment: current worktree `.venv`, CPython 3.14.7, imports from this worktree's `src`. All providers/transports/senders used here are deterministic local controlled boundaries. No paid service, private input, microphone, GUI, UDP receiver, VRChat, or HMD was used by these checks. No type-checker claim is made.

### Full focused owner/provider/architecture run

```text
.venv/Scripts/python.exe -m pytest -o addopts='' tests/core/runtime/test_self_capture_session.py tests/app/test_self_capture_vad_sink_adapter.py tests/core/test_self_stream_owned_ingress_233.py tests/core/test_self_stream_pressure_233.py tests/core/test_peer_translation_channel_owner.py tests/core/test_translation_turn_owner.py tests/core/test_translation_request_owner.py tests/core/test_translation_latency_diagnostics_owner.py tests/core/test_self_stream_233.py tests/core/test_dual_target_translation_lifecycle.py tests/core/test_index_self_request_budget.py tests/core/test_chatbox_paginator.py tests/core/runtime/test_output_runtime.py tests/core/test_stt_scoped_engine.py tests/providers/test_gemini_transcribe_backend.py tests/providers/test_gemini_transcribe_lifecycle.py tests/providers/test_gemini_transcribe_reuse.py tests/architecture -q --tb=short
```

**1070 passed in 104.46 s**, with 211 existing dependency deprecation warnings (WebSocket close reason and Google GenAI/Python private typing API). All architecture tests actually executed. This run includes the five new F5 cases and all new source/capture cases; it precedes only the additional assertion-only native-producer provenance tightening and test formatting below. No product source changed afterward.

### Final changed-path and native-producer assertion run

```text
.venv/Scripts/python.exe -m pytest -o addopts='' tests/core/runtime/test_self_capture_session.py tests/app/test_self_capture_vad_sink_adapter.py tests/core/test_self_stream_owned_ingress_233.py tests/core/test_self_stream_pressure_233.py tests/core/test_peer_translation_channel_owner.py tests/providers/test_protocol_a_scoped_sessions.py tests/core/test_stt_scoped_engine.py::test_independent_retirement_resumes_unsent_audio_without_new_local_turn -q --tb=short
```

**143 passed in 6.20 s**, with two dependency warnings (Deepgram/Pydantic V1 on Python 3.14 and Google GenAI typing API). The strengthened real-producer assertions executed. Black check and Ruff check passed for all nine product paths plus all seven changed regression paths: **16 files unchanged / all checks passed**.

### Original real owned-ingress Qwen gap proof

The migration owner executed the unchanged original case `tests/providers/test_qwen_audio_delivery_recovery.py::test_real_self_vad_gap_settles_admitted_stable_and_fresh_tail_reaches_source_and_translation`: **1 passed in 1.51 s**. It preserved exact A+C+tail source, three recognition origins across capture epochs, one manual result, rejection of unadmitted B text, the two original native cancellations, and the fresh terminal.

The same owner then executed its ten-file final migration set: **262 passed, 0 failed in 21.04 s**, two dependency warnings. That run includes the original gap case again, with unchanged source/output assertions, and captured the frozen product/test hashes. This is peer-reported executable evidence, not a second run by this worker.

### Standalone current existing-owner route

```text
.venv/Scripts/python.exe -m tests.fixtures.self_grace_233.stream2.run --output tests/fixtures/self_grace_233/stream2/result-source-resource-repair.json
```

Completed successfully; full console output is `artifact://281`. All **eight cases passed**: identical final, changed final, late predecessor, and dual targets, each with overlay absent and disabled. It observed actual local sender packets, preview before source close, compatible running-work preservation without cancellation/reissue, status-only identical final promotion, changed correction, and retained per-target snapshots through the existing source/turn/request/output/paginator route. It imports no pytest test function.

The two fixed-source schedules (`fast`, `late`) both closed exactly `alpha`, `bravo`, made exactly those two provider calls, preserved Bravo's accepted Alpha context, emitted exactly `en:alpha`, `en:bravo`, and observed no cancellation. This is a deterministic source/LLM scheduling observation, not a hardware latency claim. The JSON explicitly records its controlled-destination limits.

## Preserved failed fixture attempts

- Initial focused source/capture run: **89 passed, 4 failed**. Three new capture variants attempted `submitted` while their ledger segment was still open; the real ledger correctly rejected that sequence. The fixture now delivers its real captured end before the input-submitted callback. One new independent-unit case used receipt sequence zero; the real consumption ledger correctly excluded it. The fixture now uses its actual one-based receipt sequence. No product guard or outcome assertion was relaxed.
- Next expanded run: **107 passed, 2 failed**. Two new pressure compositions omitted the harness's required `osc` dependency and failed before exercising product code. Both now compose a real paginator/local sender; no protection was removed.
- Subsequent complete and changed-path results are above. Prior original gap failures and the whole/resource review findings remain preserved rather than silently replaced.

## Frozen repair source hashes

```text
14706d489967e4facf9a2c4933fa15a48333475fbc2f3b227de48dcc7ed3a6f7  core/orchestrator/self_translation_channel.py
036471d468dda40b3ba4a11747de2876f9e02701fe16b8373ab22ee43123096d  core/orchestrator/translation_turn.py
c7a89f97790ac7aeb6effc934fa262630e706aba0fda900d023920077cfe73b7  core/orchestrator/translation_channel_callbacks.py
17db1807ae40e54ecce3d23f7e82f4f1724701cbfd8232f68fadb5d010527dbd  core/runtime/self_capture.py
844e81f1981a928404d20614b0179dc25092641c49c7b0517c1ef4c2fd0d9d73  app/ports/capture_vad_runtime.py
efa5d130268fd864573218ff16bae8de6f20c9055f27cda38eb427468c872fa9  app/adapters/self_capture/self_capture_vad_sink.py
ecca41551b57965efcb9e0946cb591d7a134ac221639c8892a996709cd806001  core/stt/backend.py
08596f3d3c5b566ce4cd54df61d5a54de6620418746b5e271f917b98672a1b8f  core/stt/scoped_engine.py
50de0d4d4baa0c06729647b49fd8a76f8b37bb29b915cc6e3129d3832e3917c6  providers/stt/gemini_transcribe.py
```

## Handoff and limits

Owned F2/F5 source freeze was restored after the original real gap case passed, and explicitly communicated to the Director, migration owner, and runtime-validation owner. No remaining observed product failure or planned product edit remains in this worker's scope. No Git mutation was performed. The Director must integrate expanded source/resource `REPAIR_VERIFY`, final broad candidate validation, the independently repaired UI/paginator evidence, and any authorized native/CLI/GUI run. Runtime `_04` remains Director-gated; earlier `_03` evidence is not relabeled as this source pin. This record does not independently close issue #233 or claim external destination/hardware behavior.

## Subsequent full-suite capture-fixture cutover

The Director's full non-integration run later completed with **8455 passed, 2 failed, 9 skipped, 256 warnings in 1126.89 s**. The two original failures, retained in `artifact://333:120-410`, were `test_real_capture_gap_bypasses_blocked_fifo_retires_both_slots_and_new_source_progresses[known_loss-self]` and `[epoch-self]`: the fixture's `BoundaryRuntimeSink` lacked the newly required `handle_source_segment_retired` operation. They were not rerun to confirm the supplied failure.

The fixture wraps LocalASR rather than SELF VAD ingress, but its composition also owns a real SELF translation owner. `BoundaryRuntimeSink` now receives that explicit owner, records the typed cancellation through its existing native-boundary base, and forwards it to the real source owner. The original FIFO bypass, two native-slot retirement, old-result rejection, retained-byte, fresh-source, PCM-byte and capture-active assertions remain unchanged. Additional assertions require precisely the blocked unadmitted onset's cancelled identity on SELF and no SELF source-retirement receipts on Peer.

The complete callsite audit also migrated the shared native-only `RuntimeSink` in Qwen configuration/quarantine fixtures, the nested native stream sink in the SELF capture suite, and the direct native-engine SELF guard in the scoped-engine suite. Native-only boundaries retain observable typed receipt records rather than inventing native/recognition terminals or silently dropping the operation. The repeated independent-stream/reconfiguration fixture now explicitly asserts that destructive settings retirement does not forward recovery receipts.

Production capture wiring and live app/provider compositions already use the real SELF owner or its required adapter. The remaining `vad_sink=object()` settings fixture and silent Peer fake passed to a SELF quiescence-only fixture intentionally do not dispatch source ingress; they are not live protocol implementations. Release-evidence `VadSink` implementations found by the audit are Peer-only. These unrelated cold/Peer fixtures were intentionally unchanged.

Exactly four test files changed; **no production source was edited by this fixture-only repair**:

```text
40d072b11e5f640aa10fecb77d9bddb19652b5bf957d96bd4f0c92849be22ae0  tests/providers/test_qwen_audio_recovery_boundaries.py
73d7cc16685243ce4c7a6b9a0e34d4ce4bc2c38f8a5db9268bd8482b51ad25b5  tests/providers/test_qwen_audio_configuration.py
50d6f7e02138dc9621a510dcf00e8a5c538f4596968bfd1a936e828e8e0ad8a3  tests/core/runtime/test_self_capture_session.py
4a78e0f311e95128144edfa3d185aed1aff364d0e25f2637b497da580d734cf2  tests/core/test_stt_scoped_engine.py
```

Post-repair executable check:

```text
.venv/Scripts/python.exe -m pytest -o addopts='' tests/providers/test_qwen_audio_recovery_boundaries.py tests/providers/test_qwen_audio_configuration.py tests/providers/test_qwen_audio_runtime_quarantine.py tests/core/runtime/test_self_capture_session.py tests/core/test_stt_scoped_engine.py::test_queued_capture_discontinuity_retires_stream_after_successor_arrives -q --tb=short
```

**117 passed in 19.48 s**. The two original failures executed successfully as part of the full boundary file, alongside both Peer variants and the neighboring configuration/quarantine/native-stream cases. Black left all four files unchanged and Ruff reported all checks passed. No assertion was weakened, no generic product fallback or `getattr` shim was added, and no Git operation was performed. Independently authorized paginator/UI repairs are separate owners' work and are not described as source-byte-unchanged by this fixture-only observation.
