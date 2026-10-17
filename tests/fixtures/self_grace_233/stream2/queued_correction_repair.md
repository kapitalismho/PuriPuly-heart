# SELF-STREAM-2 queued correction repair

## Authority and original failure retained

Director-authorized bounded F1 repair under issue #233 R7/C5. Baseline `f1ab9a47d9caaf67bd7666b7d5f5e380da46b185`; independently reviewed candidate `5087173cb3f8fe7b1312c59337e5ec7f41e6b81e`. Original proof and pinning conditions remain in `local://stream-checkpoint-resource-review.md`, F1 lines 9–34. The supplied failure was not rerun before repair.

Original real-owner controlled-provider/sender/clock observation: unrelated long pagination was active; A's queued `alpha` preview was replaced by `alpha changed`; A sealed; B computed `bravo` before A reached the sender. Paginator falsely reported `output_superseded` for the successfully queued incoming A replacement. OutputRuntime terminalized A; B pruned A. Packets were `[x-page, x-page, en:bravo]`, omitting A's required correction. The unrelated initial long publication was not an intervening manual supersession exception.

## Repair and preserved boundaries

Only `src/puripuly_heart/core/osc/chatbox_paginator.py`, `tests/core/test_chatbox_paginator.py`, and new `tests/core/test_self_stream_queued_correction_233.py` are assigned mutation surfaces. The preparation boundary distinguishes rejected, queued, and unhandled messages. Only rejected incoming candidates receive `output_superseded`; the replaced old pending snapshot keeps its genuine supersession receipt, and the accepted current candidate stays pending until an actual sender/disposition receipt. No OutputRuntime change, retries, throttling/manual rule change, or all-pages barrier.

Existing first-successful-page diagnostic work is preserved. Existing architecture guidance already describes R7 correctly; no maintained docs or Git/remote mutation is authorized here.

## Check selection

Promised result: an earlier required queued changed correction gets its first normal local send opportunity before B, while B's provider work and consumer-visible ready snapshot progress. The new actual Self/turn/request/projection/OutputRuntime/paginator regression drives ordinary VAD/STT owner ingress with controlled provider and sender, not fabricated presentation snapshots. It observes emitted packets, source/result snapshots, invocation counts and final accepted context/history. Two page sizes cover a complete correction and a multi-page correction whose later pages remain supersedable by B. This closes the specific gap left by queued-identical/failed-send/successful-preview tests, which can pass with F1.

The existing paginator behavior test now asserts old superseded, stale rejected, and current first-page-sent receipts separately. Targeted paginator, low-latency, latency-diagnostic and stream2 suites protect identical successful cursor, queued/failed equality, manual isolation, target/resource behavior and existing page supersession. A separate fresh-process changed-path smoke will observe the actual owners at a controlled sender boundary. No UDP/VRChat/GUI/HMD or live provider claim is planned.

## Verification

Source content freeze announced to Director, stream-core, and regression migration owner before checks. No further product changes followed that freeze. Core is separately repairing F2, so this is focused F1 evidence, not final integrated-candidate acceptance.

Interpreter: worktree `.venv/Scripts/python.exe`, Python 3.14.7; pytest 9.1.1 on Windows. Actual standalone module paths were confirmed under this checkout.

1. `.venv/Scripts/python.exe -B -m pytest -o addopts= -v --tb=short tests/core/test_chatbox_paginator.py tests/core/test_self_translation_low_latency.py tests/core/test_translation_latency_diagnostics_owner.py tests/core/test_self_stream_233.py tests/core/test_self_stream_pressure_233.py tests/core/test_self_stream_owned_ingress_233.py tests/core/test_self_stream_queued_correction_233.py`
   - Collected/executed 142: **141 passed, 1 failed**, 4.47 s; exit 1.
   - All 140 existing cases passed. The new 20-character case passed. The new 12-character case wrongly expected fixed slicing `en:alpha cha`; existing paginator word wrapping correctly sent `en:alpha`. No product failure was hidden. Corrected only this expected-page fixture to explicit `(12, en:alpha)` / `(20, en:alpha changed)` cases. Required A-before-B, exactly three invocations, current-source completion/history, and no-all-pages assertions remain.
2. `.venv/Scripts/python.exe -B -m pytest -o addopts= -v --tb=short tests/core/test_self_stream_queued_correction_233.py`
   - Collected/executed both final cases: **2 passed**, 1.05 s; exit 0. The 140 unchanged suite results remain applicable; no full-suite rerun is claimed.
3. Fresh subprocess `.venv/Scripts/python.exe -B -c <in-memory real-owner smoke>` with unrelated 40-character pagination, max_chars=20, controlled provider/sender and clock starting at 10; ordinary Self VAD/STT source progression `alpha` → `alpha changed`, then successor `bravo`.
   - Initial setup exited 1 before creating the harness: unsupported `FakeClock(start=10)` keyword. Read the actual constructor and used `FakeClock(10)`; no product edit or assertion weakening.
   - Corrected smoke **passed, exit 0**. Before page opportunity: one x-page, A sealed, B's ready non-source-closed UI snapshot observed. A received no false supersession or send receipt while queued.
   - At the next opportunity: `[x-page, x-page, en:alpha changed, en:bravo]`. A's actual send receipt identifies presentation revision 4/source revision 2 (preview revision 2; sealed revision 6); B identifies presentation revision 2/source revision 1. Both explicitly `physical_ack=false`.
   - Exactly three provider invocations (`alpha`, `alpha changed`, `bravo`), no cancellation/replay, canonical accepted sources exactly `alpha changed`, `bravo`. B's actual prepared context includes accepted `alpha changed` once.
   - Six actual owner-module hashes were unchanged across the subprocess. Harness teardown and child-process exit completed; no probe files or long-lived processes/services/ports remain. The check used in-memory synthetic source/provider/sink state, a bounded fresh subprocess and normal pytest cache; no external providers, audio devices, persisted settings, secrets or user log files were used.

### Frozen SHA-256

| File | SHA-256 |
| --- | --- |
| `src/puripuly_heart/core/osc/chatbox_paginator.py` | `ab0ea6141ab3dcab37392bf1f0ee56e9983f5f8cae2234df16f613db90c47d7d` |
| `tests/core/test_chatbox_paginator.py` | `ffde6a55626bc9f098dee636b2536f8d6ef5cbe077471eebd88306099f33c37b` |
| `tests/core/test_self_stream_queued_correction_233.py` | `1a4c94886e30e4bd704d68f0af46540a60003fdcbdc5c2b46c6060bcbf082228` |

Standalone loaded-owner dependency hashes:

- `self_translation_channel.py`: `a213d24d0b3c0a966d79e2afa05362fae210c6be603b9fcf6e6a6b6bb172199f`
- `translation_turn.py`: `d411b9fc777d23f7863ad2fc805690625bf87571395daca69fadd732553112e6`
- `translation_request.py`: `de6266596fd1abeb59329033c19bafc9762768116947893a9044422643465733`
- `translation_output_projection.py`: `d273ade39092bb4ed4f2c1763d8f2e261441f06563c7f2a4be772cc69404e66d`
- `core/runtime/output.py`: `ba28b169a1aa522715325dbf96b15341ed9342e9d529ad3443d4407073ba9200`

## Handoff and limits

Original independent pinned F1 failure evidence is reused, not re-created. Existing first-successful-page repair is preserved and its actual diagnostic/guard tests passed in the assigned run. The unchanged architecture already states the preserved contract; Director owns any maintained-guidance integration.

F1's reachable changed-path outcome is repaired at the paginator receipt boundary without OutputRuntime changes. Director owns integrated verification after F2/other work lands, final formatting/check policy, bounded repair review, history/commit, and published evidence integration. No type checker, formatter, repository-wide suite, external provider, native ASR, actual UDP receiver, GUI, VRChat or HMD verification was performed here. This is not a physical delivery, GUI, full-issue completion, or measured latency-improvement claim.

## RV2 follow-on: failed-preview page provenance

Accepted medium finding: `local://stream-checkpoint-resource-repair-review.md`, F6, repaired pin `8eba7512bf6c50fdb01024fd536bada0351aca14`. Director supplied parent `c848fa2f` as fixture-only with the same product bytes. Original proof is retained, not rerun: source-open/active preview `abcd` failed, `efgh` became the first successful page with `self_turn_final=None`, and Basic falsely emitted final `last_speech_to_chatbox_send_ms=4000`.

The Director's source/test mutation hold was honored throughout the full-8eba/offline and native-UI precursor observations. All reads/proposal preparation happened without source/test edits or checks. Only after explicit release were two assigned surfaces changed:

- `src/puripuly_heart/core/osc/chatbox_paginator.py`: one condition retains revision preview/final metadata on `output_send_failed`, just as on `first_page_sent`. Existing active-message completion, replacement, supersession and shutdown paths still retire metadata. The same first-attempt failure receipt is sent; no guard, page, throttle, retry, request or provider-budget change.
- `tests/core/test_translation_latency_diagnostics_owner.py`: one actual projection/runtime/paginator/diagnostics regression fails only the preview's initial send, observes continued preview pages with `final=False` and no Basic final summary, then observes a genuine source-closed final send and exactly one final summary. It requires unchanged first-attempt R7 failure receipts, real continued sender progress, no invented later first-opportunity receipt, and final timeline cleanup. Existing managed-final-failure and successful-promoted-preview tests were preserved unchanged.

Source/test freeze was notified to Director/core/UI before checks; no subsequent mutations were made. Maintained architecture guidance already requires preview/final separation; no maintained docs or Git/remote changes were made.

### RV2 exercised evidence

`.venv/Scripts/python.exe -B -m pytest -p no:cacheprovider -o addopts= -v --tb=short tests/core/test_chatbox_paginator.py tests/core/test_self_translation_low_latency.py tests/core/test_translation_latency_diagnostics_owner.py tests/core/test_self_stream_233.py tests/core/test_self_stream_pressure_233.py tests/core/test_self_stream_owned_ingress_233.py tests/core/test_self_stream_queued_correction_233.py`

**151 collected/executed, 151 passed in 4.53 s**, exit 0; full output `artifact://335`. This includes the new failed-preview/genuine-final regression, both earlier final-failure/promoted-preview tests, F1 correction cases, failure receipts, cursor/equality and relevant stream/resource/source neighbors. No failures/skips were hidden or assertions weakened.

Separate fresh-process `.venv/Scripts/python.exe -B -c <in-memory actual-owner smoke>`: **passed, exit 0**. It instantiated production projection, runtime, paginator and diagnostic owners directly, with controlled sender/clock and a typed authoritative ready snapshot. Preview remained `source_closed=false/status=active`; initial `abcd` failed, later `efgh` sent with `page_index=1/first_successful_page=true/self_turn_final=false`. Basic summaries remained empty, and the sole R7 receipt remained `output_send_failed`. A later explicit revision-2 source-closed/complete final sent `abcd`, then `efgh`; exactly one real final Basic summary measured the controlled 9→14 clock interval as 5000 ms. Attempts were exactly `[abcd, efgh, abcd, efgh]`; final receipts exactly `[output_send_failed, first_page_sent]`. Expected stderr `[Failure] structured_failure` came from the controlled failed first send. Timeline cleanup and runtime close completed.

Four actual owner-module paths were confirmed under this checkout and hashes stayed identical before/after the standalone execution. UI/RV3 work was separately in flight, so these hashes identify the actual smoke bytes rather than claiming a final common integrated/native freeze:

| File | SHA-256 |
| --- | --- |
| `src/puripuly_heart/core/osc/chatbox_paginator.py` | `6f94ff80215f47b5ec24b78455858d21f571bd8ebc2993654f971e5050471291` |
| `tests/core/test_translation_latency_diagnostics_owner.py` | `b3de73ff971a87f0e0915b4a3430380edce86853ba4b97f380b39347eb813e80` |
| `src/puripuly_heart/core/orchestrator/translation_diagnostics.py` | `6a424daa9f8117854d93460ce34f40b01a7af6617e74f4ef7a87dc86c6ddfe97` |
| `src/puripuly_heart/core/runtime/output.py` | `ba28b169a1aa522715325dbf96b15341ed9342e9d529ad3443d4407073ba9200` |
| `src/puripuly_heart/core/orchestrator/translation_output_projection.py` | `5f513984b08dbf9da637584070856cb588cc109a895d3bcf2e72d08911882786` |

Original F1 closure evidence and its original fingerprints remain above; the independent 8eba repair review also verified F1 within scope. RV2's later paginator fingerprint supersedes the earlier product-byte fingerprint without altering the retained original observations. Director owns common-candidate/native freeze, any final integrated rerun, bounded repair verification and history/evidence publication. No native ASR/provider, actual UDP reception, GUI/VRChat/HMD, formatter or type-check pass is claimed here. Only in-memory synthetic state and bounded subprocesses were used; cache writing was disabled for the focused run, no probe files were created and no long-lived owned process/service remains.
