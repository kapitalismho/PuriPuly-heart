# PSEM Pretranslation Ontology Comparison — Results (frozen diagnostic, no adoption)

Freeze `psem.pretranslation_ontology.freeze.v1` frozen 2026-09-10T07:43:43Z (actual clock, before first probe run).
Ledger `ledger.json` generated 2026-09-10T07:44:54Z, wall 26.5 s local, smoke 9/9 PASS, zero paid API, zero new captures, zero Git mutations, only new dir `experiments/psem_pretranslation_ontology/`.

RICH recompute verifies accepted reference exactly (all NP + guard proxy counts match, F0 rich event lists match). No accepted file edited.

## Shared contract (both ontologies)

Same receiver R2 (accepted, multiple partitions same ASR, same `<fin>` cutoff), same validity/ref epochs/schedule, same tau 0.5, same FRAME 1280, same CONFIRMATION 1600 (100 ms), no param tuning. Same initial CURRENT per object, no hidden out-of-scope GT carry. Same events/raw-new-vs-context strict. Existing accepted engine reused via imports (`decode_state_events`, `r2_partition`, `r2_partition_proxy`, scoring/alignment/conservation helpers); SIMPLE is minimal local frame transform (`ref>=0.5 -> CURRENT_ONLY` incl overlap; else `other>=0.5 -> OTHER_ONLY`; else NONE) decoded through full native frame history BEFORE confirmation coalescing, not a rename of confirmed rich overlap events. GT control mapped identically (rich OVERLAP -> simple CURRENT). Generic optional opaque context preserved; no universal anchor requirement.

## Inputs (same ASR text every arm/ontology)

- NP1 `a4ed5e84…` 892 raw tokens, seal 7.282 / finalize 6.985, pay [19741040,19853040].
- NP2 `a67b4a99…` 360 raw tokens, seal 7.281 / finalize 6.985, pay [33405760,33517760].
- NP3 `d92884ae…` 692 raw tokens, seal 7.297 / finalize 6.984, pay [2553520,2665520].
- OBS `3ec7c03c…` 5 sources 8597 chunks, research FP16 Vulkan profile.
- Old grid masks only, never old scores. Ref map valid-past-first-100ms, tau 0.5.
- Mapping slots identical both ontologies (shared validity): NP1 slot 2 ready 19658240, NP2 slot ready 33463040, NP3 slot ready 2549760.

## F0 events (actual observed; GT control same availability/profile, oracle-marked)

NP actual streams contain no in-pay overlap frames, so SIMPLE and RICH emit identical in-pay events:

- NP1 in-pay: 19808000 OTHER timely avail 5.265 (rich) / 5.265 (simple), margin 2.017. Out-of-scope 19682560 OTHER / 19690240 CURRENT unchanged both.
- NP2 in-pay: 33472000 OTHER timely avail 5.184 both, margin 2.097.
- NP3 in-pay: 2611200 OTHER timely avail 4.857 both, margin 2.440.
- Extended decode payhi+20000: same in-pay sets, zero new distinct boundaries both ontologies (see WAIT_GATE.md).

Guards differ exactly where overlap frames exist (SIMPLE drops UNRESOLVED-bounds, keeps CURRENT path):

- R1: rich F0 3187200 OVERLAP + 3191040 CURRENT; simple F0 none (overlap merged to CURRENT before confirm, no transition).
- R2: rich F0 669440 OVERLAP (out-of-span) + 672000 OTHER applied; simple F0 672000 OTHER only. In-span applicable both 1 event (672000).
- T1: rich F0 669440 OVERLAP + 672000 OTHER + 702720 CURRENT + 712960 OVERLAP + 733440 OTHER; simple F0 672000 OTHER + 702720 CURRENT + 733440 OTHER (overlap bounds removed). In-span applicable rich 3 (702720/712960/733440), simple 2 (702720/733440).
- COMBINED: same as T1 plus in-span 672000: rich applicable 4, simple 3.
- BC1: unmapped, no F0 events both (guarded, not actual wrong zero). Singles: no fires both.

## NP window8 (fixed cohort, same tokens all arms; PAIRED counts PRIMARY)

Conservation exact every case; DROP/DUP/TEXT throwaway detected. GT word oracle upper reuses accepted point (unreachable diagnostic, not target).

- NP1 (8/0/0): R0 4c4w. R2-RICH actualF0 7c0w1u. Gross R0->RICH per-word (ledger `paired_actualF0.rows` read directly, not inferred): 3 fixes A1118/A1119/A1120 wrong->correct, 1 wrong withheld A1117 wrong->UNKNOWN, 0 correct withheld (all 4 B CURRENT stay CURRENT); net +3 correct. R2-SIMPLE actualF0 7c0w1u IDENTICAL (0 delta RICH vs SIMPLE; simpleCURRENT_richUNRESOLVED 0). GT_STATE arms identical 7c0w1u both.
- NP2 (6+1+1/8): R0 4c2w1u1m. R2-RICH actualF0 4c0w3u1m. Gross R0->RICH per-word (read directly): 1 fix C1480 wrong->correct, 1 correct withheld B1653 correct->UNKNOWN, 1 wrong withheld C1481 wrong->UNKNOWN; NET correct +0 (4->4), wrong 2->0. Totals net correct 0 is NOT gross fix 0. The prior parent shorthand '2 wrong both withheld' is superseded here: C1480 was a gross fix, B1653 a correct-withheld. Accepted receiver docs keep their own wording (upstream, owned separately, notified; not edited by this outcome). R2-SIMPLE actualF0 4c0w3u1m IDENTICAL. GT_STATE identical 4c0w3u1m both.
- NP3 (8/0/0): R0 4c4w. R2-RICH actualF0 6c0w2u. Gross R0->RICH per-word (read directly): 2 fixes B323/B324 wrong->correct, 2 wrong withheld B322/B325 wrong->UNKNOWN, 0 correct withheld; net +2 correct. R2-SIMPLE actualF0 6c0w2u IDENTICAL. GT_STATE identical 6c0w2u both.
- Full payload separate (same word sets, not re-scored here): accepted 33GT/28GT/25GT groupings preserved; text/char exact.

NP verdict: SIMPLE == RICH on all 3 NP under shared validity (same events, same partitions, same paired counts). No ontology win, no loss on NP. No wait advantage observed within the available pre-terminal record (see WAIT_GATE.md); not a generic claim about waiting ever.

## Guards (synthetic terminal object-end zero grace shared; actual end-to-end UNKNOWN; proxy excl punc; never aggregated)

Same baseline prefix compared only (no cross-prefix comparison; no R2+T1+COMBINED double count; no broad percent prevalence).

- R1 guard (10 proxy): R0 9c1w. R2-RICH actualF0 7c0w3u. Per-word ownership read directly from ledger (not inferred): B32 wrong->UNKNOWN (0 fixes; not a B32 fix), A492/A493 correct->UNKNOWN; i.e. 0 gross fixes, 1 wrong withheld, 2 correct withheld; net -2 correct, -1 wrong. R2-SIMPLE actualF0 9c1w0u (= R0: no cut, retains 1 wrong, withholds 0). SIMPLE reduces UNKNOWN 3->0 by assigning CURRENT but keeps 1 other-wrong that RICH moved to UNKNOWN. Strict tradeoff, not a win.
- R2 guard (9 proxy): R0 3c6w. R2-RICH actualF0 4c3w2u (accepted). R2-SIMPLE actualF0 4c3w2u IDENTICAL (overlap bound out-of-span, no contrast here).
- T1 guard (21 proxy): R0 10c11w. R2-RICH actualF0 7c0w14u. Gross R0->RICH per-word (ledger ownership read directly, not inferred): 4 fixes B54/55/56/57 wrong->correct, 7 wrong withheld B46/48/49/50/51/52/53 wrong->UNKNOWN, 7 correct withheld A128/132/134/135/136/137/138 correct->UNKNOWN; NET correct 10->7, wrong 11->0 (prior '11 wrong removed, 3 correct withheld' phrasing was net-language on the correct side and is superseded here). R2-SIMPLE actualF0 13c5w3u. SIMPLE vs RICH: +6 correct, +5 new other-wrong, -11 unknown. Net vs R0: +3 correct, -6 wrong, +3 unknown. Reduces UNKNOWN by assigning CURRENT through overlap spans but adds 5 strict other-wrong (all-roles GT count, not frame conditional). GT_STATE arms: RICH 13c7w1u / SIMPLE 13c7w1u identical (accepted GT point reproduced).
- COMBINED (28 proxy, overlaps T1+R2 spans so NOT summed; net triple only, no prevalence claim): R0 12c16w. R2-RICH actualF0 10c2w16u net (R1-projection reference 14c14w history only). Gross R0->RICH per-word (read directly): 7 fixes B43/44/45/54/55/56/57 wrong->correct, 8 wrong withheld B42/46/48/49/50/51/52/53 wrong->UNKNOWN, 8 correct withheld A124/128/132/134/135/136/137/138 correct->UNKNOWN, plus A126 correct->wrong and B41 wrong retained (the 2 RICH wrong). R2-SIMPLE actualF0 16c7w5u. SIMPLE vs RICH: +6 correct, +5 new wrong, -11 unknown (same tradeoff, same double-count scope warning).
- BC1 (4 proxy): all arms 0c4w (unmapped, no F0, guarded not wrong-zero claim). Singles 6c/4c: no fires, zero confident wrong, proxy-only, identical both.

Guard verdict: where overlap bounds exist, SIMPLE trades UNKNOWN for CURRENT and strictly adds other-wrong (R1 +1 retained, T1/COMBINED +5 new vs RICH). Demonstrates the change question (overlap semantics move counts) but neither ontology reaches wrong-zero without withholding cost.

## Four dimensions (same as accepted, both ontologies)

1. Capabilities: R0 no-op; R2 post-terminal X-partition incl return, overlap RICH UNRESOLVED vs SIMPLE CURRENT path, straddle UNKNOWN intact, gap UNRESOLVED. R1 accepted-history reference only (COMBINED 2 seals X672000 Z684896 + X733440 Z746336; NP single-seal).
2. Evidence quality: shared mapping support, valid+mask+speech join, tail EOS invalid, NONE/invalid/gap reset, mask/nonspeech NOOP, confirmed persists, coalesced duplicates.
3. Frontier lateness: R2 uses X with avail<=terminal (seal 7.282/7.281/7.297, NO wait, cutoff before compute). Margins 2.017/2.097/2.440 both ontologies. CPU measured nonzero volatile separate (wall 26.5 s; per-arm decode+partition microseconds, own repartition not zero).
4. Unresolvable ASR loss: NP2 1 missing +1 mixed; straddle/gap unresolved NP1 1, NP2 +2 (mixed incl), NP3 2. Unknown kept in denominators.

## Limits

Research 7 s clips not C56s. Reference/provider unknown. Guards capacity only, never PASS causal/safety, never adoption. ASR-guard absent 0 calls. Overlap RICH UNRESOLVED not enrollment; SIMPLE CURRENT not safe enrollment either (adds wrong). ~24 window + 68 proxy words not representative. P3T/return HISTORY ONLY. #98B validity-diff excluded (cannot assume equivalence). No frozen named reference-free candidate identified in reviewed authorities; not selected/excluded permanently. No training; no wait advantage observed within the available pre-terminal record (scoped, not a generic ever-claim).
