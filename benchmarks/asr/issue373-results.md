# #373 bounded paired pilot

## Decision and checkpoint

**Keep the approved policy.** Six scored cross-speaker joins disappeared without losing recognized source or either correct continuation. Fragmentation and logical MT work did not improve. Higher representative coverage exposed four additional English native-attribution errors, reducing conditional precision. This small probe does not establish general MT quality, full-audio diarization accuracy or physical display latency. No thresholds were tuned; no noise sweep, second recognition pass or third arm was used.

- B0: actual pre-feature `24859f9e6788f4ace4afbdd971193d673441a9c8`, Director-exported to an isolated source directory.
- B1 product: `e7163508bd728cf42043bd65701d10b53eaf3902`. Director confirmed its formatting-only finalization preserved replay semantics; independent product review reported no material findings.
- Separate subprocess import roots shared the existing Python 3.14 environment/dependencies. This artifact and three task-specific scripts contain no product changes.

The product composes normal missing native IDs permissively while keeping unsafe/mixed correspondence independent; preserves a known sentence constraint through merge **and** continuation; keeps continuation attribution source-local; and accepts authoritative text/source/style/divider revisions together. Existing Peer timing, prompts/context owners, concurrency, paging/reading protection, non-target routes and SELF/Peer isolation remain the contract. No speaker IDs enter MT prompts/native rendering.

## Capture and reference scope

Original pinned [EN](inputs/replay-en.json), [JA](inputs/replay-ja.json) and [source/license](inputs/sources.json) manifests supplied one intact 120-second recording each. AMI ES2004d begins at original 32.368 s; IngCrowd dialogue01 at 86.04 s. Japanese audio is noncommercial verification material and is not redistributed.

| Identity / consumption | EN | JA |
|---|---|---|
| Clean WAV SHA-256 | `ee19cafe7382e7f26249b273c0f27ce080963788e35c2ca67034f234d903b6d6` | `8cdbe6868df6e141f80a14ce7e49b501cd371e185e2899a2b1a659dc22d06a2b` |
| Frozen trace SHA-256 | `664799ec630552b90b88acec7da7937690a1d2ba7fed2582e36151e3783159f5` | `6b1df48f0558580e8af036fa9d6ceccb503741387e077180a3b10ca6832a2094` |
| Actual sessions / seals | 1 / 27, one empty final | 1 / 18 |
| Mapped source audio sent / product padding | 97.992 s / 2.000 s | 117.680 s / 3.000 s |

One paced pass used production `PeerCaptureSessionOwner` → audio/VAD loop → Silero ONNX → `ScopedRecognitionEngine` → Soniox. Only device ingress and admission/target/provider-attachment ports were controlled. All 240 input seconds, including gaps/overlap, entered the capture loop; production VAD did not send every silent/input sample. No private/live capture, normal-app startup, keyring, discovery, telemetry or OSC destination was started. Current `require_secret` resolved existing authorized environment values with an isolated memory store; values were never printed.

Actual Soniox request: `stt-rt-v6`, PCM16/16 kHz/mono; hints `[en]` or `[ja]`, strict=false, diarization=true, language identification=false, endpoint detection=false, no context terms. Peer onset=0.4, production three-chunk onset/commit, pre-roll=500 ms, configured hangover=900 ms with production external delivery boundaries. Selective 200 ms finalize padding remained unchanged; no speaker-triggered cuts. Two billable ASR sessions, no reconnect, key-verification call or recognition retry.

The common trace retains original native JSON tokens, cumulative updates, terminal text/runs, native session scope, epochs/turns, actual capture ranges, relative arrivals and write/seal/terminal receipts **before grouping**. All 45 terminal texts matched raw final tokens after existing edge-whitespace normalization. No gap/cut correspondence was invented.

[Pre-arm labels](issue373-references.json) were frozen by the experiment agent from original human reference text, not newly human/audio-adjudicated: six EN and seven JA coherent units, including explicit multi-annotation Japanese continuations. Other ambiguous boundaries remain uncertain. Fixed scoring unit is one uniquely indexed original native final **lexical token**, equal weight in both arms—not human words; punctuation/control tokens are excluded. Actual sent ranges plus exact product padding map native time to original audio. Primary scoring excludes overlap, unavailable references and broken/zero-duration timing. One maximum-duration-overlap, one-to-one native-ID/reference mapping per actual session is reused across arms; unmatched assigned IDs are wrong. EN: 1→FEE013, 2→MEE014; neither can also map to MEO015. JA: 1→speaker_2, 2→speaker_1.

| Primary coverage limits | EN | JA | Pooled |
|---|---|---|---|
| Scorable / lexical native tokens | 81/408 | 498/627 | 579/1035 |
| Excluded unavailable-reference/gap / overlap / timing | 254 / 73 / 0 | 44 / 81 / 4 | 298 / 154 / 4 |
| Annotated different-speaker overlap retained | 12.459 s | 11.210 s | 23.669 s |
| Complete-reference intervals with any scorable recognized token | 12/21 | 37/47 | 49/68 |
| Complete-reference intervals with none | 9/21 | 10/47 | 19/68 |

Even covered intervals are not claimed lexically complete. Full-audio ASR recall/word accuracy and DER are **N/A**. Natural primary tokens all had native IDs, so this sample does not acoustically estimate the unknown-joining/continuity tradeoff.

## Paired results

Each arm received the same updates/terminal receipts once through production Peer translation/caption and `OutputRuntime` owners. Korean target, default product prompt/context policy, layout scale=1.0, request limits and zero-service-time deterministic completion were fixed. Group ranges, real request kinds/prompts/context and accepted revisions were retained. Flow connections derive from each arm's own composition/merge/continuation records, never B1 continuity keys retrofitted into B0. Retained output events were also projected once through that arm's production presenter at recorded delivery times.

Every percentage includes n/N; Δ means B1−B0 percentage points.

| Metric | EN B0 → B1 (Δ) | JA B0 → B1 (Δ) | Pooled B0 → B1 (Δ) |
|---|---|---|---|
| Native assigned coverage, shared | 81/81 = 100% | 498/498 = 100% | 579/579 = 100% |
| Native conditional accuracy, shared | 76/81 = 93.8% | 495/498 = 99.4% | 571/579 = 98.6% |
| Native overall correct, shared | 76/81 = 93.8% | 495/498 = 99.4% | 571/579 = 98.6% |
| Representative coverage | 59/81 = 72.8% → 81/81 = 100% (+27.2) | 465/498 = 93.4% → 498/498 = 100% (+6.6) | 524/579 = 90.5% → 579/579 = 100% (+9.5) |
| Representative conditional accuracy | 58/59 = 98.3% → 76/81 = 93.8% (−4.5) | 462/465 = 99.4% → 495/498 = 99.4% (+0.0 rounded) | 520/524 = 99.2% → 571/579 = 98.6% (−0.6) |
| Representative overall correct | 58/81 = 71.6% → 76/81 = 93.8% (+22.2) | 462/498 = 92.8% → 495/498 = 99.4% (+6.6) | 520/579 = 89.8% → 571/579 = 98.6% (+8.8) |
| Fragmented coherent units | 0/6 = 0% → 0/6 = 0% (0) | 2/7 = 28.6% → 2/7 = 28.6% (0) | 2/13 = 15.4% → 2/13 = 15.4% (0) |
| Mean disconnected flows | 6/6 = 1.00 → 1.00 | 10/7 = 1.43 → 1.43 | 16/13 = 1.23 → 1.23 |
| Joined scored adjacent-final boundaries | 1/3 = 33.3% → 0/3 = 0% (−33.3) | 7/15 = 46.7% → 2/15 = 13.3% (−33.3) | 8/18 = 44.4% → 2/18 = 11.1% (−33.3) |
| Correct-join recall | N/A (0 opportunities), both | 2/3 = 66.7% → 2/3 = 66.7% (0) | 2/3 = 66.7% → 2/3 = 66.7% (0) |
| Join precision | 0/1 = 0% → N/A (0 opportunities) | 2/7 = 28.6% → 2/2 = 100% (+71.4) | 2/8 = 25% → 2/2 = 100% (+75.0) |
| Harmful speaker-change crossings, including within-final | 1/1 = 100% → 0/1 = 0% (−100) | 5/16 = 31.2% → 0/16 = 0% (−31.2) | 6/17 = 35.3% → 0/17 = 0% (−35.3) |
| Lost / duplicate / missing translated-output tokens | 0/408 / 0/408 / 0/408, both | 0/627 / 0/627 / 0/627, both | 0/1035 / 0/1035 / 0/1035, both |
| Colored settled source, arm presenter projection | 81/81 = 100% → 81/81 = 100% (0) | 361/498 = 72.5% → 498/498 = 100% (+27.5) | 442/579 = 76.3% → 579/579 = 100% (+23.7) |
| Gray-vs-colored disagreement on applied final revisions | 8/39 = 20.5% → 0/39 = 0% (−20.5) | 9/35 = 25.7% → 0/36 = 0% (−25.7) | 17/74 = 23.0% → 0/75 = 0% (−23.0) |

Only 18/43 original adjacent-final opportunities have certain small-reference labels; four more are uncertain, the remainder unscorable/empty. Scored joins by kind: EN merge 1→0, continuation 0→0; JA merge 2→0, continuation 5→2; pooled merge 3→0, continuation 5→2. Within-final groups remain EN 39/26 nonempty finals, JA 36/18, pooled 75/44. Within-final scored speaker-change crossings: EN **N/A (0 opportunities)**; JA and pooled 0/4→0/4. No within-final speaker-change coalescing occurred. Completed/unrelated same-speaker joins: EN/pooled 0/2→0/2; JA **N/A (0 opportunities)**. Most other same-speaker boundaries remain uncertain.

Representative scores use actual accepted source attribution, not continuity or RGB. B0's style disagreements describe its genuine old first-readable freeze, not violations of its former contract. B0 JA retains 125/498 scorable tokens gray with `not_ready`; another 12/498 have no final presenter state after one stale/expired repair rejection. B1 has neither. This presenter rejection is **not** translation/history source loss. Natural overflow/handoff/divider-change opportunities were absent: rates are **N/A (0 opportunities)**; separate synthetic GUI/tests cover them. The natural projection checks gray/colored coherence, not a complete RGB/divider certification.

## Work and timing

| Logical work | EN B0 → B1 | JA B0 → B1 | Pooled B0 → B1 |
|---|---|---|---|
| preview / final / repair / continuation | 4/31/6/2 → 4/36/1/2 | 11/27/2/7 → 11/32/0/4 | 15/58/8/9 → 15/68/1/6 |
| Requests per input audio minute | 43/2 = 21.5 → 21.5 | 47/2 = 23.5 → 23.5 | 90/4 = 22.5 → 22.5 |

Structural completions made **zero cloud MT calls**. ASR arrivals/seals are real; service completion is controlled and output/presentation backpressure is not reconstructed. No attribution wait was added. Existing capture-open/seal anchors give these delivery values (seconds, median/p95):

| Timing | EN | JA |
|---|---|---|
| Controlled open→first-readable, both arms | n=26, 3.375/6.006 | n=18, 5.972/6.405 |
| Controlled seal→last accepted revision, B0→B1 | n=26, 0.274/4.267 → 0.237/0.522 | n=18, 0.300/6.986 → 0.300/0.349 |
| Actual seal→provider final, shared | n=26, 0.237/0.301 | n=18, 0.300/0.349 |

Later repairs mechanically extend earlier source settlement, so shorter settlement is not proof of faster MT. The common clock origin is first provider observed timestamp minus its recorded sink-relative arrival; a sub-millisecond sink offset is shared. Physical display-latency distribution: **N/A (not measured)**.

Eight short intervals—four/language, changed boundaries plus a native-error adverse case—produced 16 arm examples: **15/15 actual MT calls completed**, one exact rendered-body cache hit, no failure/retry. Product route `LlmTranslationBackend(OpenRouterLLMProvider)` requested/returned `google/gemma-4-26b-a4b-it`, provider NextBit, Korean, temperature=0.6, reasoning=none, max_tokens=100; default latency preferences allowed Cloudflare/Dekallm/NextBit/Makora and fallbacks. Cache keys require full source/rendered prompt/context/model/parameter identity; same source alone did not hit. Consumption: seven EN/eight JA calls, 14,281 prompt + 245 completion tokens, reported $0.0006773325. Backend-invocation→completed-response wall median/p95: n=15, 0.603/0.807 s. This convenience sample is not a comparative performance claim.

## Five diagnostic examples

Times are recording-relative; annotation IDs link to manifests above. `final:token` indices are zero-based originals.

1. **EN 7.44–9.37 s, `ES2004d.sync.11`/`.326`, `2:18→3:0`:** B0 repaired “Mm-hmm. Standard chip.” across different native keys; B1 kept “Standard chip.” independent. Actual Korean changed from “음, 일반적인 칩이네요.” to “표준 칩이에요.” This is a removed harmful policy join, not repaired recognition.
2. **Candidate-worse precision, the same acknowledgement:** B0's later mixed repair made those contributions unassigned; B1 preserved the native ID. The fixed session map cannot assign one ID to MEE014 and MEO015. Newly assigned EN 22 tokens include 18 correct and four wrong. The identical isolated “Mm-hmm.” request alone used the exact cache (“음, 네.”). Coverage is not free accuracy.
3. **JA ~48.37 s, speaker_2 `a66`, `7:23→8:2`:** B0 appended “はいはいはい。” to speaker_1's children question; B1 used an ordinary independent request, yielding “네네네.” B0's actual Korean included “애가 있다면라든가. 네네네.” The awkward wording is retained, not graded a pass. No recognized short reply was dropped.
4. **JA 53.20–60.19 s, `a67`/`a37`, `8:44→9:1`:** B0 continued the theme-park sentence from an intervening other-speaker acknowledgement; B1 made it independent. Both Korean examples were readable, but source “じじいばば” and paraphrases remain imperfect. The declared theme-park unit still has two flows in both arms; the eating unit still has three, including a missed reference continuation around 100 s. No fragmentation win.
5. **JA 87.84–91.04 s, `a76`/`a177`, `13:43→14:1`:** B0 repaired “うーん。 家族で何かするわけでもないし。” into an old other-speaker caption whose final revision was stale in controlled baseline presentation. B1 independently produced “가족끼리 뭘 같이 하는 것도 아니고.” Independent current-surface display of exported B0 text was readable; that does not negate the baseline timeline rejection.

All six scored B0 harmful boundaries were inspected: EN merge `2:18→3:0`; JA continuation `2:23→3:2`, merge `7:23→8:2`, continuation `8:44→9:1`, continuation `12:31→13:2`, merge `13:43→14:1`. Native IDs differed at every one. B1 has **zero new harmful joins and zero dropped recognized contributions** in this eligible sample. Ten native-key-incompatible joins disappeared overall (five/language); four lack primary references and are not claimed quality wins.

## Actual display, checks and limits

The display owner presented all **16/16 nonblank Korean examples** over eight intervals in one 12.8 s supported current Python Desktop session, using production presenter/event adapter + `FletDesktopRendererWindow`. B1 short replies were separate colored cards versus independently exported B0 merged gray examples. Long Japanese source lines were ellipsized; only initial pages were inspected in this natural subset. Incomplete “It looks—”→“그거—” remained fragmentary. This is real translation/display evidence, not expert MT grading or historical B0 GUI timing reconstruction. Native/HMD physical display was not observed.

Local evidence: `build/issue-373-display/actual-mt-evidence.json`, `actual-mt-01.png`…`16.png`, English/Japanese contact sheets. Display-input SHA-256: `650e40d489e35edacf80a0c2fbf01342cdc646a576d90d1007345f51290de6d9`. Owner reported exit 0, PID removal, backdrop termination/wait. Separate controlled actual-GUI cases observed gray→gold final, gray→cyan repair, overflow divider and paging/reading protection. Apparent lower-cyan disappearance was resolved as an inline-image tool inconsistency: retained PNG cyan masks were identical (4,469 pixels, symmetric difference zero), not a product defect.

Reused Director evidence: corrected integrated selection **414 passed** (native Peer/lifecycle/identity/presentation/turn, Soniox backend/reuse/bounded delivery); fresh-process native-JSON→production Peer/request/output smoke passed conservation/history/no OSC. Initial test invocation named missing `tests/providers/test_soniox.py`, collected zero, then corrected paths passed. Neighboring presentation selection passed 330 with 48 existing asyncio deprecation warnings. Final repository-wide Ruff, Black and Python test-selection checks passed after correcting four benchmark import-order findings; the formatted scorer regenerated the same counts from retained traces. No type-check run is claimed. Initial capture setup failed before service access because environment keys used `export`; loader correction fixed it. A scoring edge-whitespace correspondence error was corrected before final counts; it was not product loss.

Known documentation drift, intentionally unchanged: `native/overlay/README.md:20` promises unconditional first-readable style freeze; #373 amends native Soniox authoritative revisions. Repo rules prohibit modifying that README without explicit request. Architecture ownership remains accurate.

## Reproduction and retained artifacts

Use existing dependencies; export pinned B0 to `.tmp/issue373-baseline` first. Commands are from this worktree. Do not print credentials or redistribute audio. Capture only once if the frozen traces are unavailable:

```text
.venv/Scripts/python.exe benchmarks/asr/issue373.py capture --root . --env ../../../.env.local --language en --audio ../../../benchmarks/asr/data/en/replay-clean.wav --output .tmp/issue373/en-trace.json
```

Repeat capture with `ja`. For **each language**, run these four commands against its same frozen trace, substituting `ja` for `en` as appropriate, then score:

```text
.venv/Scripts/python.exe benchmarks/asr/issue373.py replay --root .tmp/issue373-baseline --trace .tmp/issue373/en-trace.json --arm B0 --output .tmp/issue373/en-B0.json
.venv/Scripts/python.exe benchmarks/asr/issue373.py replay --root . --trace .tmp/issue373/en-trace.json --arm B1 --output .tmp/issue373/en-B1.json
.venv/Scripts/python.exe benchmarks/asr/issue373.py present --root .tmp/issue373-baseline --trace .tmp/issue373/en-B0.json --output .tmp/issue373/en-B0-presentation.json
.venv/Scripts/python.exe benchmarks/asr/issue373.py present --root . --trace .tmp/issue373/en-B1.json --output .tmp/issue373/en-B1-presentation.json
.venv/Scripts/python.exe benchmarks/asr/issue373_score.py --directory .tmp/issue373
```

MT selection is a local JSON array of `{case,file,request_index}` for both arms: EN indices 2,3,30,32; JA 8,21,25,38. Run:

```text
.venv/Scripts/python.exe benchmarks/asr/issue373_mt.py --root . --env ../../../.env.local --selection .tmp/issue373/mt-selection.json --output .tmp/issue373/mt-examples.json
```

`issue373.py`, `issue373_score.py`, `issue373_mt.py` are thin task-specific adapters over production owners/test composition helpers, not a new framework. `scores.json` retains fixed units/exclusions/mapping/groups/boundaries/timing. Provider traces, arm outputs, MT request/response evidence and presenter projections remain under ignored `.tmp/issue373`; GUI evidence remains under ignored `build/issue-373-display`. No audio/bulk results are proposed tracked. Owned processes closed; no personal app state changed. Reuse retained traces for policy reproduction rather than making another paid recognition pass.
