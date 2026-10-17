# SELF-STREAM-2 CORE+OUTPUT workstream

This is current implementation evidence, not the archived SELF-GRACE-1 characterization. The historical comparison and its pinned source/harness remain under the historical-evidence owner's control. No elapsed-credit or no-op-rearm timing optimization was selected or implemented.

The subsequent first-success latency repair is recorded in [FIRST_SUCCESS_REPAIR.md](FIRST_SUCCESS_REPAIR.md). That record supersedes three product hashes below, retains the original broad-check failure, and supplies the repaired candidate's 290-test result and `result-first-success-repair.json` standalone evidence. The following sections retain the preceding candidate's evidence and limits rather than silently relabeling its pin.

The later source-gap and completed-failure resource repairs are recorded in [SOURCE_RESOURCE_REPAIR.md](SOURCE_RESOURCE_REPAIR.md). That record supersedes the earlier acoustic lifetime account and affected source hashes, preserves the original failed observations and fixture corrections, and supplies the 1070-test owner/provider/architecture result, final 143-test changed-path result, peer-owned 262-test migration result, and separate `result-source-resource-repair.json` standalone proof. Existing records and JSON outputs are retained at their actual pins.

## Authority and candidate

- Authoritative issue: https://github.com/kapitalismho/PuriPuly-heart/issues/233, SELF-STREAM-2, read in full including its archived authority.
- Supplied synchronized baseline: `f1ab9a47d9caaf67bd7666b7d5f5e380da46b185`.
- Candidate: final product hashes below, subsequently integrated by the Director at `5087173cb3f8fe7b1312c59337e5ec7f41e6b81e` according to the Director's readback. The additional owned-ingress test/report evidence requires Director integration. Git history, publication, maintained documentation, shared domain contracts, and integration acceptance belong to the Director.
- Environment: current worktree `.venv`, CPython 3.14.7; application imports resolved from this worktree's `src`. No paid service, private input, microphone, or user log was used by these focused checks.

## Implemented ownership

- `TranslationTurnLifecycleOwner` reserves identity/order at draft creation, owns result-bearing primary/secondary operations and retired operations, adopts compatible ready/running work at source closure, and publishes immutable `SelfTurnPresentation` snapshots. The old merge-buffer speculative attempt and secondary task-transfer lifecycle are removed.
- `SelfTranslationChannelOwner` retains acoustic reconciliation, consumed-source provenance, continuation/watchdog/grace decisions, and source closure. Source closure no longer depends on primary LLM completion. Grace duration, handler-receipt anchor/rearm rules, awaiting-end policy, and production default were not changed.
- Requests retain the preparation that actually produced their response. Draft preparation does not add accepted context. Ordered source-closed admission remembers canonical source once under the existing per-target context semantics. Existing generic final fallback is preserved once; Index terminal failures preserve their existing request-attempt budget.
- Handover acquires an actual canonical speech permit, never invents one at seal. A sealed live speculation retains original occupancy when both canonical running permits are occupied. Physically ready successful work may retire without starting another provider call or becoming a ninth waiter. Logical cancellation retains physical provider permits until actual cleanup. Speech-specific cancellation no longer holds the shared admission lock throughout physical cleanup, preserving manual admission/bypass.
- `OutputRuntime` receives authoritative snapshots independently of overlay state. Its destination admission retains existing count/byte budgets. Latest unsent revisions replace same-turn obsolete state. A narrow speech publication guard protects an older required correction's normal local first-page opportunity without gating successor translation computation. Manual supersession remains an exception; later pages retain existing supersession behavior.
- `ChatboxPaginator` reports actual successful local first-page transmission and explicit terminal dispositions. Equal final payload/coverage/targets reuse only successful compatible previews; queued equal values keep their send opportunity, failed equal values receive the normal final send opportunity, and unchanged final promotion does not reset pagination. Separate equal-text turns remain separate.
- Dual-target translated chatbox output preserves the previous translation-only newline formatting even when `chatbox_include_source=True`. Single-target output preserves that preference. Current-source final failure still honors existing single-target `fallback_transcript_only` behavior.
- Managed SELF application presentation uses snapshots only, not duplicate legacy transcript/translation UI events. Accepted conversation records remain final-only. Ready target provenance and update identity survive status-only terminal promotion.
- Source-close alone does not emit semantic overlay finalization. Final target settlement preserves actual result language metadata and clears incompatible sticky translation when appropriate. Overlay delivery uses the existing OutputRuntime task and bounded destination-admission ownership, so a blocked overlay or UI destination cannot hold source closure or independent chatbox output. Actual asynchronous sink failure still updates existing diagnostic ownership.
- Preview first-page transmission is separate from existing final-send latency summaries. Identical final promotion records its reused-preview relationship rather than fabricating another final packet. An identical queued payload can be promoted before its first actual transmission without resetting its cursor.

## Acoustic obligation reassessment after upstream overlap

The task-owned `_acoustic_continuations` state remains keyed by actual `AudioSegmentIdentity` (or the legacy acoustic UUID), not publication UUID. Matching usable contribution, actual terminal, false-start release, or the existing per-source watchdog settles only that source. New onsets do not replace older unresolved obligations. Request-ledger ownership of consumed prefixes/suffixes remains separate. OFF/reset/shutdown retire the scoped bookkeeping through the existing owning lifecycle.

The map is bounded by real active/unresolved acoustic ownership and existing watchdog retirement; this report does not reuse the historical statement that only one local recognition request can overlap. PCM release/native logical terminal and actual acoustic obligation settlement remain different facts. No new ASR/native reclamation or request limit was introduced. Current request/terminal/prefix/suffix and continuation regressions belong to the independently migrated five-file suite; the final real-audio/local-ASR and Qwen/provider composition limits must be reported by their execution owners, not inferred from this map.

## Executed final focused checks

### Owner and neighboring contracts

```text
.venv/Scripts/python.exe -m pytest -o addopts='' tests/core/test_translation_turn_owner.py tests/core/test_translation_request_owner.py tests/core/test_chatbox_paginator.py tests/core/runtime/test_output_runtime.py tests/core/test_dual_target_translation_lifecycle.py tests/core/test_index_self_request_budget.py tests/core/test_self_stream_233.py tests/core/test_self_stream_pressure_233.py -q --tb=short
```

Final result: **247 passed in 5.42 s**. The selected set actually executed. New pressure/behavior cases cover preview-before-close, running reuse, target-first/combined output, generic fallback, Index budget, two canonical plus one speculative/eight pending pressure, physical-ready handover, actual provider permits, expiry, cancellation with physical cleanup still outstanding, manual isolation, real prepared context, blocked overlay/UI, failed/queued equal sends, and unchanged pagination cursor/result identity.

### Owned-VAD and scoped stable/terminal ingress

```text
.venv/Scripts/python.exe -m pytest -o addopts='' tests/core/test_self_stream_owned_ingress_233.py -q --tb=short
```

Final result: **11 passed in 6.06 s** on the same frozen product.

- Ten cases use actual `PeerAudioSegmentLedger`-produced `OwnedVadEvent` input through the existing SELF and local-ASR runtime owners, with a controlled scoped-recognition provider accepted under the matching current settings scope. They cover delayed A while B started before any merge buffer, B-ended/C-false-start races, and matching B empty/error/cancel/expiry/watchdog release. Each rejects premature A publication and then observes the earlier usable source's one actual local sender packet after the matching release.
- The eleventh case drives a scoped stable prefix before VAD end, observes its translated preview before watchdog closure, retains the open request's suffix in a successor publication, and reconciles the actual final terminal without losing or duplicating either contribution. Provider calls and local packets are exactly one per retained prefix/suffix.
- Recognition and translation providers are controlled local test boundaries. The production SELF/local-ASR/turn/request/output/paginator ownership and scoped ingress paths execute, but these are not actual ASR decoding or Qwen socket/native-resource measurements.
- An initial fixture attempt without an attached matching scoped provider failed all ten race cases at the local-ASR admission check (`no self provider accepts the segment configuration scope`). The fixture was corrected by supplying a matching controlled provider through normal composition; no product guard or outcome assertion was suppressed.

### Standalone current-owner proof

```text
.venv/Scripts/python.exe -m tests.fixtures.self_grace_233.stream2.run --output tests/fixtures/self_grace_233/stream2/result.json
```

Final result: **passed**, retained in `result.json`.

- Eight standalone cases: identical final, changed final, late predecessor versus successor preview, and secondary-first/combined dual targets, each with an absent or controlled disabled overlay destination.
- Two additional fixed-source schedules vary only controlled LLM timing. Both retain the same `alpha`, `bravo` closed source set, two invocations, no cancellation, and ordered local sender packets.
- The standalone harness invokes existing SELF/turn/request/output/paginator owners through VAD and final-recognition ingress; it does not import or call pytest test functions or inject merge-buffer state.
- Profile: configured 30 ms grace for the standalone test; production 400 ms default is unchanged. This is not an elapsed-credit/timer-tuning result.
- Denominator: 10 synthetic acoustic episodes across the eight standalone cases, plus four episode executions across the two fixed schedules. Changed-source cases contribute two source revisions within one acoustic episode. Provider invocations and packets are recorded per case in the JSON, not normalized to only successful source units.
- Local sender success is not UDP reception or VRChat/HMD visibility. This standalone run has no real audio/ASR, actual translation model, GUI, recipient or HMD claim. Native production composition/loopback and actual UI evidence are separate owners' deliverables.

### Static checks

Ruff checks and Black formatting checks were run on the twelve initial source/test/runner surfaces and separately on the added owned-ingress test. Final result: **Ruff passed; Black reported all 13 files unchanged across the two checks**. No type-check claim is made.

### Independent migrated regressions

The regression-migration owner reported **183 passed, 0 failed** on the frozen candidate, with explicit `PYTHONPATH=src`; its final fingerprints match the projection/output hashes below. That owner's task-local report retains migration rationale, earlier failures, async destination observation barriers, accepted-drain versus destructive-retirement authority, and final evidence. These cases are not counted again in the 247-test command above.

## Failure evidence retained during implementation

- An initial old-mechanics regression run failed on removed `merge_buffer.speculative_attempt` accesses/injected state, old RUNNING-buffer assumptions, and several genuine preserved behaviors. The separate migration owner deleted obsolete incidental-state fixtures and retained meaningful ingress/provider/sink outcomes.
- An early standalone late-predecessor case exposed non-atomic available-capacity conversion: source closure returned before the original speculative reservation had converted to an available canonical permit. Atomic actual-permit handover repaired that gap; final standalone cases pass.
- The first pressure/manual cancellation run exposed a shared admission lock held during physical cancellation cleanup. Manual admission waited until the child watchdog instead of admitting independently. The lock now covers selection/cancellation request, not physical drain; the final actual-permit/manual case passes.
- Regression migration exposed sealed-provider replacement fallback fencing, final provider language metadata, unusable blank preview handling, late endpoint bookkeeping, and current-source final source-only fallback. Repairs preserve their original outcomes and request budgets.
- The first post-handoff frozen regression run had 180 passes and three failures. Two assertions observed one asynchronous destination as though it completed all destinations or expected the source producer to remain blocked on an overlay. They now use independent completion barriers and preserve real outcomes. The third identified missing post-delivery overlay failure accounting; the restored completion callback fixed the actual diagnostic omission. The subsequent frozen 183-case result passed. Earlier pins remain precursor evidence, not final acceptance.
- Bare `SelfTranslationChannelOwner.close_ingress()` stops new admission but intentionally permits accepted work to drain. No broad callback/output suppression was added to satisfy an invented bare-ingress fence. OFF/reset, turn-channel retirement, and actual composed shutdown retain destructive cancellation authority.

## Final SHA-256 fingerprints

| Surface | SHA-256 |
| --- | --- |
| `src/puripuly_heart/core/orchestrator/self_translation_channel.py` | `a213d24d0b3c0a966d79e2afa05362fae210c6be603b9fcf6e6a6b6bb172199f` |
| `src/puripuly_heart/core/orchestrator/translation_turn.py` | `d411b9fc777d23f7863ad2fc805690625bf87571395daca69fadd732553112e6` |
| `src/puripuly_heart/core/orchestrator/channel_runtime.py` | `a24538c35b157d1c68ac035799d551f68dd89c0e8431cbc19701e7c2a479621c` |
| `src/puripuly_heart/core/orchestrator/translation_output_projection.py` | `5f55bd185486a4026e1d26de98f281d11df9b855dfdad624656dd9f73ebe24ce` |
| `src/puripuly_heart/core/orchestrator/translation_diagnostics.py` | `eeebb047fd121d99c640a404d9361347f3744f6d16c270a52ff7f81de892bf66` |
| `src/puripuly_heart/core/runtime/output.py` | `ba28b169a1aa522715325dbf96b15341ed9342e9d529ad3443d4407073ba9200` |
| `src/puripuly_heart/core/runtime/output_batch.py` | `db78438aac5de7f65f8368ca42bb237ad78f045421ca10a99274986554fb6b26` |
| `src/puripuly_heart/core/osc/chatbox_paginator.py` | `933515a6142258ea335f53fd482acbd1512afa10f05d52159642781dc94f4b58` |
| `tests/core/test_self_stream_233.py` | `e1cec01d0a12fcea78733425a627ab44202bdec6a46a291065c1566d6ca9d6bb` |
| `tests/core/test_self_stream_pressure_233.py` | `7794deeda982f96594ab5bfc2d75e9201adb37ed6ef7d5f11c1c3763ef4fe47b` |
| `tests/core/test_self_stream_owned_ingress_233.py` | `75384b7f1d4614c0a46b471f54f49795eb44567ea2f0cd59027fd79a1f9f609e` |
| `tests/core/test_index_self_request_budget.py` | `9af0fbd632b75934d39c155eaaf179dd1af8436fa4333064e4a9bd78b7c68cb4` |
| `tests/fixtures/self_grace_233/stream2/run.py` | `c8bb14cf342049a002c23ea3a73d4bbcf4ac02078483bc47f9d5d5658f53ba93` |

## Integration needs and limits

No known product failure remains in this workstream's exercised surfaces. Product freeze was explicitly restored to native-validation and regression-migration after the observer repair. No further product changes are planned by this worker.

The Director still owns the sustained-final broad integration check, maintained architecture/user guidance, pinned baseline-to-candidate independent review, final evidence audit/index, and any authorized history/publication action. Suggested maintained architecture updates are the draft/source-close/context ownership split, authoritative revision-aware snapshots, destination-local first-page speech guard and successful-payload promotion, and independent bounded overlay handoff described above.

At handoff, the Director separately reported a broad precursor check with 2,060 passes and 21 failures (`artifact://169`), principally around final-source observation barriers in Qwen/SELF ordering/quarantine tests. The Director assigned bounded migration/classification of three test files to the regression owner. Those results are not converted to passes here; genuine product defects must return to the product owner and invalidate the frozen pin before repair. The focused results above do not substitute for that remaining integration check.

Native final-only/stable-contribution real-audio/local-ASR production composition and actual loopback packets belong to stream-runtime-validation; actual GUI presentation belongs to progressive-ui. Their corresponding final candidate fingerprints and observations must be integrated separately. Physical VRChat/HMD access is not established here and must remain explicitly unperformed/blocked unless directly observed by its owner. This workstream does not mark all of issue #233 complete.
