# #373 bounded paired pilot

## Decision and implementation

Keep the approved policy. In this small sample it removed six scored cross-speaker joins without losing recognized source or changing the two correctly joined reference continuations. It did **not** reduce fragmentation or logical MT work. Higher representative coverage exposed four additional English native-attribution errors; conditional precision fell. This is a useful integration probe, not evidence of general MT-quality or full-audio diarization improvement. No threshold tuning, noise sweep, extra ASR pass or third arm was used.

B0 is actual pre-feature `24859f9e6788f4ace4afbdd971193d673441a9c8`, exported by the Director to an isolated source directory. B1 product checkpoint is `e7163508bd728cf42043bd65701d10b53eaf3902`; the Director confirmed its formatting-only finalization did not change replay semantics. Separate subprocess import roots used the same existing Python 3.14 environment and dependencies.

The product change composes normal missing native IDs permissively but keeps malformed/mixed correspondence independent; carries a known sentence-flow constraint through merge **and** continuation; keeps continuation attribution source-local; and accepts authoritative source/text/style/divider revisions together. Existing Peer timing, request/context owners, bounded concurrency, paging/reading protection, non-target routes and SELF/Peer isolation remain the contract. No speaker IDs enter MT prompts or native rendering.

## Inputs, capture and scoring

- Original pinned [English](inputs/replay-en.json), [Japanese](inputs/replay-ja.json) and [source/license](inputs/sources.json) manifests: one intact 120-second recording each. AMI ES2004d starts at original 32.368 s; IngCrowd dialogue01 starts at 86.04 s. Japanese audio is noncommercial verification material and is not redistributed.
- Clean WAV SHA-256: EN `ee19cafe7382e7f26249b273c0f27ce080963788e35c2ca67034f234d903b6d6`; JA `8cdbe6868df6e141f80a14ce7e49b501cd371e185e2899a2b1a659dc22d06a2b`.
- One paced pass each through production `PeerCaptureSessionOwner`, `run_audio_vad_loop`, Silero ONNX, `ScopedRecognitionEngine` and Soniox. Only device ingress/admission/target/provider attachment ports were controlled; no live/private capture, normal app startup, keyring, discovery, telemetry or OSC destination was started. Credentials used current `require_secret`, an isolated memory store and the existing authorized environment values; values were not printed.
- Actual requested Soniox model `stt-rt-v6`; PCM16, 16 kHz, mono; hints `[en]` / `[ja]`, strict hints false, diarization true, language identification false, endpoint detection false, no context terms. Peer VAD onset 0.4, production three-chunk onset/commit, 500 ms pre-roll, 900 ms configured hangover with production external delivery boundaries. Production selective 200 ms finalize padding remained enabled; no speaker-triggered cuts. No reconnect: one actual native session per recording, 27 EN / 18 JA seals; one EN final was empty. Session audio writes were 97.992 / 117.680 s of mapped source plus 2.000 / 3.000 s product padding, respectively. All 240 input seconds entered the production capture loop, including natural gaps/overlap; VAD legitimately did not send every silent/input sample to Soniox. Two billable ASR sessions, no key-verification call or recognition retry.
- Common traces freeze original native JSON tokens, finals/partials, native session scope, provider epochs/turns, language/speaker runs, actual capture ranges, relative arrivals and actual write/seal/terminal receipts **before** grouping. EN trace SHA-256 `664799ec630552b90b88acec7da7937690a1d2ba7fed2582e36151e3783159f5`; JA `6b1df48f0558580e8af036fa9d6ceccb503741387e077180a3b10ca6832a2094`. Raw token text matched all 45 terminal texts after existing edge-whitespace normalization. No gap or cut-to-final pairing was invented.
- [Pre-arm labels](issue373-references.json) were frozen from original human reference text by the experiment agent, not newly human/audio-adjudicated. Six EN and seven JA small coherent units include explicit multi-annotation Japanese continuations; other ambiguous boundaries remain uncertain. These labels are deliberately limited.
- Fixed unit: one uniquely indexed original final native lexical token, equal weight per token in both arms (provider subwords, **not** human words). Punctuation/control tokens are excluded. Actual sent capture spans plus exact product padding map native times back to original audio. Primary speaker scoring requires an unambiguous reference-available interval and excludes different-speaker overlap and broken/zero-duration timing. One one-to-one maximum-duration-overlap native-ID/reference mapping per actual session is reused across arms; unmatched assigned IDs are wrong. EN mapping: native 1→FEE013, 2→MEE014; no native ID can also map to MEO015. JA: 1→speaker_2, 2→speaker_1.
- Of 408 EN / 627 JA lexical tokens, 81 / 498 are primary-scorable. Exclusions: EN 254 unavailable-reference/gap + 73 overlap; JA 44 unavailable-reference/gap + 81 overlap + 4 timing/correspondence. Annotated different-speaker overlap totals 12.459 / 11.210 s, retained in input. Only 12/21 EN and 37/47 JA complete-reference intervals contain any scorable recognized token; the other 9/21 and 10/47 have none. Even those covered intervals are not claimed lexically complete. Full-audio ASR recall/word accuracy and DER are **N/A**, not inferred from these recognized-source scores.
- The same frozen updates and terminal receipts entered each arm's production Peer translation/caption and `OutputRuntime` owners once. Korean target, default product prompt, layout scale 1.0, default request limits/context policy and zero-service-time deterministic completion were held constant. Source-group ranges, actual repair/continuation kinds, request prompts/context and accepted revisions were retained. Connected flows derive from each arm's own records, never B1 continuity metadata retrofitted into B0. Retained events were then projected once through that arm's production `OverlayPresenter` at their recorded delivery times; this controlled projection is not a baseline physical-GUI timing reconstruction.

## Paired counts

Every percentage below includes its numerator/denominator. Δ is B1−B0 in percentage points. Empty opportunities are not zero-quality observations.

| Measure | EN B0 → B1 (Δ) | JA B0 → B1 (Δ) | Pooled B0 → B1 (Δ) |
|---|---|---|---|
| Native assigned coverage, shared | 81/81 = 100% | 498/498 = 100% | 579/579 = 100% |
| Native conditional accuracy, shared | 76/81 = 93.8% | 495/498 = 99.4% | 571/579 = 98.6% |
| Native overall correct, shared | 76/81 = 93.8% | 495/498 = 99.4% | 571/579 = 98.6% |
| Representative assigned coverage | 59/81 = 72.8% → 81/81 = 100% (+27.2) | 465/498 = 93.4% → 498/498 = 100% (+6.6) | 524/579 = 90.5% → 579/579 = 100% (+9.5) |
| Representative conditional accuracy | 58/59 = 98.3% → 76/81 = 93.8% (−4.5) | 462/465 = 99.4% → 495/498 = 99.4% (+0.0 rounded) | 520/524 = 99.2% → 571/579 = 98.6% (−0.6) |
| Representative overall correct | 58/81 = 71.6% → 76/81 = 93.8% (+22.2) | 462/498 = 92.8% → 495/498 = 99.4% (+6.6) | 520/579 = 89.8% → 571/579 = 98.6% (+8.8) |
| Unnecessarily fragmented coherent units | 0/6 = 0% → 0/6 = 0% (0) | 2/7 = 28.6% → 2/7 = 28.6% (0) | 2/13 = 15.4% → 2/13 = 15.4% (0) |
| Mean disconnected flows | 6/6 = 1.00 → 6/6 = 1.00 | 10/7 = 1.43 → 10/7 = 1.43 | 16/13 = 1.23 → 16/13 = 1.23 |
| Joined scored adjacent-final boundaries | 1/3 = 33.3% → 0/3 = 0% (−33.3) | 7/15 = 46.7% → 2/15 = 13.3% (−33.3) | 8/18 = 44.4% → 2/18 = 11.1% (−33.3) |
| Correct-join recall | N/A (0 opportunities), both | 2/3 = 66.7% → 2/3 = 66.7% (0) | 2/3 = 66.7% → 2/3 = 66.7% (0) |
| Join precision | 0/1 = 0% → N/A (0 opportunities) | 2/7 = 28.6% → 2/2 = 100% (+71.4) | 2/8 = 25% → 2/2 = 100% (+75.0) |
| Harmful speaker-change crossings, including within-final | 1/1 = 100% → 0/1 = 0% (−100) | 5/16 = 31.2% → 0/16 = 0% (−31.2) | 6/17 = 35.3% → 0/17 = 0% (−35.3) |
| Lost / duplicate / missing translated-output source tokens | 0/408 / 0/408 / 0/408, both | 0/627 / 0/627 / 0/627, both | 0/1035 / 0/1035 / 0/1035, both |
| Colored settled source, actual arm presenter projection | 81/81 = 100% → 81/81 = 100% (0) | 361/498 = 72.5% → 498/498 = 100% (+27.5) | 442/579 = 76.3% → 579/579 = 100% (+23.7) |
| Attribution/style disagreement on applied final revisions | 8/39 = 20.5% → 0/39 = 0% (−20.5) | 9/35 = 25.7% → 0/36 = 0% (−25.7) | 17/74 = 23.0% → 0/75 = 0% (−23.0) |

Only 18 of 43 original adjacent-final opportunities have certain small-reference labels; four additional boundaries are uncertain, the remainder unscorable/empty. Scored joining kinds: EN merge 1→0, continuation 0→0; JA merge 2→0, continuation 5→2; pooled merge 3→0, continuation 5→2. Within-final groups are unchanged: EN 39/26 nonempty finals, JA 36/18, pooled 75/44. Within-final scored speaker-change crossings are EN **N/A (0 opportunities)**, JA 0/4→0/4, pooled 0/4→0/4; none were coalesced. Completed/unrelated same-speaker joins were 0/2 EN, **N/A (0 opportunities)** JA, 0/2 pooled, both arms; most same-speaker boundaries outside the declared units remain uncertain.

Representative attribution uses accepted source-local attribution, not continuity keys or RGB. The B0 style disagreements describe its real first-readable freeze, not an assertion that B0 violated its old contract. In controlled projection, B0 JA has 125/498 scorable tokens still gray with `not_ready`, plus 12/498 without a final presenter state after one expired/stale repair rejection; B1 has neither. The missing presenter state is **not** lost translation/source history: translation output conserved all tokens. No natural-trace overflow or session-handoff opportunity occurred: overflow/handoff/divider-change error rates are **N/A (0 opportunities)**. Synthetic GUI/test evidence covers those cases separately. Natural input had no missing native-ID tokens in the primary denominator, so the unknown-joining/constraint tradeoff is not acoustically estimated by this pilot.

## Work, timing and actual MT

| Logical work | EN B0 → B1 | JA B0 → B1 | Pooled B0 → B1 |
|---|---|---|---|
| preview / final / repair / continuation | 4 / 31 / 6 / 2 → 4 / 36 / 1 / 2 | 11 / 27 / 2 / 7 → 11 / 32 / 0 / 4 | 15 / 58 / 8 / 9 → 15 / 68 / 1 / 6 |
| requests per input audio minute | 43/2 = 21.5 → 43/2 = 21.5 | 47/2 = 23.5 → 47/2 = 23.5 | 90/4 = 22.5 → 90/4 = 22.5 |

Structural MT requests are controlled completions, **zero actual cloud MT calls**. Product audio cuts, finalize acknowledgements and ASR arrival times are real. No attribution wait was added. Controlled translation/output adds zero service time and does not reconstruct output/presentation backpressure.

For nonempty original finals, controlled segment-open→first-readable delivery median/p95 is EN n=26, 3.375/6.006 s in both arms; JA n=18, 5.972/6.405 s in both. Controlled seal→last accepted revision representing that source is EN n=26, 0.274/4.267 s → 0.237/0.522 s; JA n=18, 0.300/6.986 s → 0.300/0.349 s. A later repair extends the earlier source's settlement anchor, so fewer repairs mechanically shorten this measure; it is not physical display latency or proof of faster MT. Actual seal→provider-final median/p95, shared: EN n=26, 0.237/0.301 s; JA n=18, 0.300/0.349 s. Clock origin is the first provider observed timestamp minus recorded sink-relative arrival; its sub-millisecond sink offset is shared across arms. A meaningful physical display-latency distribution is **N/A (not measured)**.

Eight short natural intervals (four per language) were selected after structural comparison, including changed boundaries and a native-error adverse case. Sixteen logical arm examples made **15/15 completed actual MT calls**, one exact rendered-request-body cache hit, no service failure/retry. The product `LlmTranslationBackend(OpenRouterLLMProvider)` route requested and returned `google/gemma-4-26b-a4b-it`, actual provider NextBit; Korean target, temperature 0.6, reasoning effort none, max_tokens 100, default product provider preferences (Cloudflare/Dekallm/NextBit/Makora, latency sort, fallbacks allowed). Full source, rendered prompt, context, model and parameter body must match for a cache hit; identical source alone did not hit. Seven actual EN calls and eight JA calls; 14,281 prompt / 245 completion tokens, reported provider cost $0.0006773325. Observed service-call wall median/p95 n=15: 0.603/0.807 s, measured backend invocation→completed response; small sequential convenience sample, not a comparative performance claim.

## Diagnostic examples

Times are recording-relative and annotations link to the pinned manifests above. IDs such as `2:18` are original final-index/native-token-index, zero-based.

1. **EN acknowledgement→new speaker, 7.44–9.37 s**, `ES2004d.sync.11` / `.326`, boundary `2:18→3:0`: B0 merged “Mm-hmm. Standard chip.” despite different native keys; B1 kept “Standard chip.” independent. Actual Korean B0 “음, 일반적인 칩이네요.” versus B1 “표준 칩이에요.” shows why a repaired single sentence can blur a reply boundary. This is the scored English harmful join removed by the policy gate, not repaired native recognition.
2. **Candidate-worse attribution precision, same EN acknowledgement**: B0's later mixed repair left these contributions unassigned; B1 retained the known native ID. The fixed session map cannot assign the same native ID to both MEE014 and MEO015, so this acknowledgement stays wrong. Across EN, newly assigned 22 tokens include 18 correct and four wrong; fewer unknowns is not free accuracy. The MT request for isolated “Mm-hmm.” matched exactly across arms and alone used the cache (“음, 네.”).
3. **JA short reply, about 48.37 s**, speaker_2 `a66`, boundary `7:23→8:2`: B0 appended “はいはいはい。” to speaker_1's children question; B1 requested the short reply independently with the ordinary prompt. Actual B0 “아, 근데 그건가, 아직 그렇네. 애가 있다면라든가. 네네네.” versus B1 “네네네.” No recognized short reply was dropped in either arm. The malformed Korean “있다면라든가” is retained evidence, not a quality pass.
4. **JA continuation returning across a reply, 53.20–60.19 s**, `a67` / `a37`, boundary `8:44→9:1`: B0 continued the theme-park sentence from the intervening other-speaker acknowledgement; B1 made it independent. Both real translations were readable, but the provider's source “じじいばば” and context-dependent Korean paraphrases remain imperfect. The declared theme-park unit still has two disconnected flows in both arms; this is not a fragmentation win. The eating unit also remains three flows, including the missed correctly labelled continuation around 100 s.
5. **JA family statement after acknowledgement, 87.84–91.04 s**, `a76` / `a177`, boundary `13:43→14:1`: B0 repaired “うーん。 家族で何かするわけでもないし。” into the earlier other-speaker caption, and that old caption's final repair was stale in the controlled baseline presenter. B1 independently translated “家族で何かするわけでもないし。” as “가족끼리 뭘 같이 하는 것도 아니고.” Current-surface independent display of the exported B0 example was readable; it does not negate the baseline timeline's stale rejection.

All six scored B0 harmful crossings were inspected: EN merge `2:18→3:0`; JA continuation `2:23→3:2`, merge `7:23→8:2`, continuation `8:44→9:1`, continuation `12:31→13:2`, merge `13:43→14:1`. Native IDs differed at every one, so these are policy joins, not a native-ID collision. B1 introduces **zero new harmful joins and zero dropped recognized-source contributions** in the eligible sample. Ten native-key-incompatible joins were removed overall (five EN, five JA); four lack sufficient primary references, so they are not reclassified as proven quality wins. No numerical improvement threshold was imposed.

## Actual surface and verification evidence

The display owner used the current supported Python Desktop caption surface through the production presenter/event adapter and `FletDesktopRendererWindow`. One 12.8 s session displayed all 16/16 nonblank Korean arm examples over eight intervals, with source; short B1 replies were separate colored cards versus independently exported B0 merged gray examples. Long Japanese source lines were ellipsized by the renderer; only initial pages were inspected for this natural subset. The incomplete “It looks—”→“그거—” remained fragmentary. This is actual presentation of real translations, not expert MT grading or recreation of B0's physical timing.

Local retained evidence: `build/issue-373-display/actual-mt-evidence.json`, `actual-mt-01.png` through `actual-mt-16.png`, and English/Japanese contact sheets. Example JSON SHA-256 `650e40d489e35edacf80a0c2fbf01342cdc646a576d90d1007345f51290de6d9` pins the display input. The owner reported renderer exit 0, PID removal and backdrop termination/wait. Separate controlled actual-GUI cases observed gray preview→gold final, gray→cyan repair, overflow divider and paging/reading-protection behavior. An apparent lower-cyan disappearance was a tool inline-image inconsistency: retained before/after PNG cyan masks were identical (4,469 pixels, symmetric difference zero); no product defect was established and no extra GUI campaign was run. Native/HMD physical display was not observed.

Reused Director evidence on the frozen product checkpoint: corrected integrated focused selection **414 passed**, including native Peer/lifecycle/identity/presentation/turn, Soniox backend, session reuse and bounded delivery; fresh-process native-JSON→production Peer/request/output smoke passed conservation/history/no OSC. An initial test command named absent `tests/providers/test_soniox.py` and collected zero tests, then actual paths were corrected. Neighboring presentation selection passed 330 tests with 48 existing asyncio deprecation warnings. Final repository-wide Ruff, Black and Python test-selection checks passed after correcting four benchmark import-order findings; the formatted scorer regenerated the same counts from retained traces. No type-check run is claimed. Initial experiment setup failed before any service call because environment lines used `export`; the narrow environment loader correction resolved it. An edge-whitespace scoring correspondence error was corrected before final counts; it was not product source loss.

Known documentation drift intentionally left unchanged: `native/overlay/README.md:20` describes unconditional first-readable speaker-style freeze; #373 amends this for native Soniox authoritative revisions. Repository rules prohibit modifying that README without an explicit request. The architecture ownership map remains accurate.

## Reproduction and retention

Commands below use the existing environment; do not install/edit product packages or print credentials. Export B0's full pinned source to `.tmp/issue373-baseline` first. Existing clean audio lives in the main checkout (`../../../benchmarks/asr/data/{en,ja}/replay-clean.wav` from this worktree). Do not distribute Japanese audio or commit generated traces/images.

```text
.venv/Scripts/python.exe benchmarks/asr/issue373.py capture --root . --env ../../../.env.local --language en --audio ../../../benchmarks/asr/data/en/replay-clean.wav --output .tmp/issue373/en-trace.json
.venv/Scripts/python.exe benchmarks/asr/issue373.py capture --root . --env ../../../.env.local --language ja --audio ../../../benchmarks/asr/data/ja/replay-clean.wav --output .tmp/issue373/ja-trace.json
```

For each language, replay the **same** saved trace once per arm, then project its retained output events once through the matching presenter:

```text
.venv/Scripts/python.exe benchmarks/asr/issue373.py replay --root .tmp/issue373-baseline --trace .tmp/issue373/en-trace.json --arm B0 --output .tmp/issue373/en-B0.json
.venv/Scripts/python.exe benchmarks/asr/issue373.py replay --root . --trace .tmp/issue373/en-trace.json --arm B1 --output .tmp/issue373/en-B1.json
.venv/Scripts/python.exe benchmarks/asr/issue373.py present --root .tmp/issue373-baseline --trace .tmp/issue373/en-B0.json --output .tmp/issue373/en-B0-presentation.json
.venv/Scripts/python.exe benchmarks/asr/issue373.py present --root . --trace .tmp/issue373/en-B1.json --output .tmp/issue373/en-B1-presentation.json
.venv/Scripts/python.exe benchmarks/asr/issue373_score.py --directory .tmp/issue373
```

Repeat the four replay/present commands with `ja` filenames **before** scoring. For the real-MT subset, create a local JSON array of `{case,file,request_index}` selectors for both arms: EN request indices 2, 3, 30, 32; JA 8, 21, 25, 38. Then:

```text
.venv/Scripts/python.exe benchmarks/asr/issue373_mt.py --root . --env ../../../.env.local --selection .tmp/issue373/mt-selection.json --output .tmp/issue373/mt-examples.json
```

The capture/replay adapter and scorer are task-specific thin glue over production owners/test composition helpers, not a new benchmark framework. `scores.json` retains fixed units, exclusions, mapping, reconciled groups, all scored boundary diagnoses and timing counts. Generated provider traces, arm outputs, request bodies, MT responses and presenter projections stay under ignored `.tmp/issue373`; GUI artifacts stay under ignored `build/issue-373-display`. No audio or bulk output is part of the proposed tracked change. Processes created by this probe were closed; no personal application state was changed. Reusing the frozen trace avoids another billable recognition pass; fresh capture is only needed if deliberately reproducing provider behavior, not for policy replay.
