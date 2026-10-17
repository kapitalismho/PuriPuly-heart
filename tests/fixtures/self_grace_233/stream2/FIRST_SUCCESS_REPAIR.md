# First-successful-page latency repair

This record supersedes the three product hashes below from the original `REPORT.md` candidate. It is a narrowly scoped repair of the Director's broad-check failure, not a new retry or pagination policy.

## Failed candidate and observed failure

- Director-reported integrated candidate: `5087173cb3f8fe7b1312c59337e5ec7f41e6b81e`.
- Actual local broad run reported by the Director: 6,337 passed, 9 failed, 9 skipped; failure evidence at `artifact://147:301-340`.
- Failing test: `tests/core/test_translation_latency_diagnostics_owner.py::test_first_successful_page_keeps_latency_after_earlier_page_failures`.
- Preserved assertion: a three-page caption fails its first two local page sends, successfully sends `ijkl` at the third attempt, and emits exactly one existing latency summary, `last_speech_to_chatbox_send_ms=7000`. Actual failure was zero summaries.
- The failure was read directly. It was not rerun merely to confirm it.

## Diagnosis and repair

The preview/final diagnostics cutover had selected `page_index == 0` as though that meant the first successful local transmission. That is only the first-page attempt. Existing pagination can fail earlier pages and still transmit a later page successfully.

- `ChatboxPaginator` now emits an explicit per-message `first_successful_page` fact. It becomes true on the first successful local send, regardless of its page index, and false on later successful pages of that same message.
- `TranslationLatencyDiagnosticsOwner` uses that success fact, retaining the first-success latency contract after earlier page failures.
- R7 still uses the actual first-page attempt success/terminal disposition. No all-pages or first-later-success barrier was introduced. A failed first-page opportunity may release the speech publication guard under existing destination-failure policy.
- `TranslationOutputProjectionOwner` no longer abandons latency merely because that first attempt reports `output_send_failed` while remaining pages can still succeed. Actual `message_terminal`/supersession/exhaustion still retires pending timing.
- An already successful preview promoted to final keeps its pagination cursor and does not manufacture a fresh final send summary from a later page. The added negative test records `(page_index, first_successful_page, self_turn_final)` as `(0, True, False)` then `(1, False, True)` with no fabricated basic final-send summary.

There is no source grouping, grace timer, provider, request-attempt, canonical-capacity, or ASR policy change in this repair.

## Changed surfaces

- `src/puripuly_heart/core/osc/chatbox_paginator.py`
- `src/puripuly_heart/core/orchestrator/translation_diagnostics.py`
- `src/puripuly_heart/core/orchestrator/translation_output_projection.py`
- `tests/core/test_translation_latency_diagnostics_owner.py`

The original failing assertion and fixture were retained unchanged. Two managed-output regressions were added: first-success latency after earlier failed pages while the R7 first-attempt failure remains terminal, and negative protection against turning a promoted preview's remaining page into a fresh final send.

## Executed checks

```text
.venv/Scripts/python.exe -m pytest -o addopts='' tests/core/test_translation_latency_diagnostics_owner.py tests/core/test_chatbox_paginator.py tests/core/runtime/test_output_runtime.py tests/core/test_translation_turn_owner.py tests/core/test_translation_request_owner.py tests/core/test_dual_target_translation_lifecycle.py tests/core/test_index_self_request_budget.py tests/core/test_self_stream_233.py tests/core/test_self_stream_pressure_233.py tests/core/test_self_stream_owned_ingress_233.py -q --tb=short
```

Result: **290 passed in 7.62 seconds**. This includes the unchanged originally failing test, both new positive/negative tests, the 11 owned-VAD/scoped stable-terminal cases, output failure/cancellation/pressure/context cases, Index attempt budgets, and neighboring lifecycle/output/paginator contracts.

Black formatted the four assigned repair files (three already unchanged); Ruff passed on all four. No type-check claim is made.

```text
.venv/Scripts/python.exe -m tests.fixtures.self_grace_233.stream2.run --output tests/fixtures/self_grace_233/stream2/result-first-success-repair.json
```

Result: **passed**, eight standalone current-owner scenarios plus two fixed-source varied-LLM schedules. The original `result.json` remains the preceding pinned result; the new JSON is the repair candidate's current deterministic record. Provider/sink limits remain those documented in `REPORT.md`; no real audio, GUI, recipient or HMD claim is added by this run.

## Repair SHA-256 pins

| Surface | SHA-256 |
| --- | --- |
| `src/puripuly_heart/core/osc/chatbox_paginator.py` | `883a12de4cb09170c914d5a903a7d7a1754bf46fe4f60ed5f4747a29372328ec` |
| `src/puripuly_heart/core/orchestrator/translation_diagnostics.py` | `6a424daa9f8117854d93460ce34f40b01a7af6617e74f4ef7a87dc86c6ddfe97` |
| `src/puripuly_heart/core/orchestrator/translation_output_projection.py` | `d273ade39092bb4ed4f2c1763d8f2e261441f06563c7f2a4be772cc69404e66d` |
| `tests/core/test_translation_latency_diagnostics_owner.py` | `91cc3d959bcce3f35780bc9458d7ba5d7120f36ef585ca329a0ddee7bdf804c5` |

Other product hashes remain those of the original workstream report. This repair is uncommitted by this worker; the Director owns history and final candidate publication.

## Integration status

The source freeze was invalidated before the repair and explicitly restored to the Director, native-validation owner and regression-migration owner after the 290-test result. No further product edits are planned by this worker for this repair. Independent whole/resource review and repair verification must use the updated pins.

The Director holds native `_04` execution until reviews and repair verification settle. Existing `_03` runtime manifests remain precursor/applicability-limited evidence, not the repaired source pin. Broad-check migration and its remaining audio-gap A+C source observation failure are separately owned and not claimed fixed here. If that investigation proves a product defect, it requires its own scoped repair and another pin invalidation.
