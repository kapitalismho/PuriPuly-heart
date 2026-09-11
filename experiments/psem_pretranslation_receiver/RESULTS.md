# PSEM Pretranslation Receiver — Results (frozen diagnostic, no adoption)

Freeze `psem.pretranslation_receiver.freeze.v1` + revision 2026-09-10T07:36:07Z (actual clock, original C13 contract restoration, not new ontology; prior freeze d060ef86 audited).
Ledger `ledger.json` generated 2026-09-10T07:36:38Z (after revision), wall 26.4 s local, smoke 18/18 PASS, zero paid API, zero new captures, zero Git mutations.

REPAIR: R1 prospective C13 uses actual source ownership ranges, not old per-object first-wins. `r1_project` and `r1_project_guard` maintain seals list with source-position exact-scope check (existing seal Z >= X separates), tooLate if no open content, no arbitrary MAX first cut. Return CONTINUE unsupported allowed does not cancel reference and does not stop future OTHER. Later OTHER is new unknown logical segment, not B identity. Same ordered requests both receivers. PREFIX synthetic 2-distinct-X before-fix 1 applied (second already_separated) fails, after-fix 2 applied passes. Actual COMBINED 2 seals observed: X672000 Z684896 and X733440 Z746336 (684896 < 733440, first does not separate second).

## Inputs (same ASR text every arm)

- NP1 `a4ed5e84…` 892 raw tokens, seal 7.282 / finalize 6.985.
- NP2 `a67b4a99…` 360 raw tokens, seal 7.281 / finalize 6.985.
- NP3 `d92884ae…` 692 raw tokens, seal 7.297 / finalize 6.984.
- OBS `3ec7c03c…` 5 sources 8597 chunks, research FP16 Vulkan profile.
- Old grid masks only, never old scores. Ref map valid-past-first-100ms, tau 0.5.
- Mapping slots: NP1 2 ready 19658240, NP2 2 ready 33463040, NP3 1 ready 2549760.

## F0 events (actual, same for all receivers)

- NP1: 19682560 OTHER (out of scope), 19690240 CURRENT (out of scope), 19808000 OTHER in-pay timely 5.265.
- NP2: 33472000 OTHER in-pay timely 5.184.
- NP3: 2611200 OTHER in-pay timely 4.857.
- GT-STATE control same 100 ms valid support/frontier, same availability, marked oracle not deploy.

## NP window8 (fixed cohort, same tokens all arms; Full NP unchanged single-seal paired exact)

Conservation exact every case; DROP/DUP/TEXT throwaway detected. R1 n_seals 1 each (X/Z as above, segments CURRENT-initial + OTHER-1 new-unknown, reference fixed no-reset, no ASR input change).

- NP1 (8/0/0): R0 4c4w; R1-F0 X19808000 Z19825520 a5.265 4c4w (Z beyond window, frontier lateness); R2-F0 7c0w1u. Paired gross R2 vs R0 ledger-verified: 3 wrong fixed (A1118, A1119, A1120 wrong→correct), 1 wrong withheld (A1117 wrong→UNKNOWN); all 4 B (1152, 1154, 1155, 1157) stay correct; net +3 correct. Oracle upper 8c.
- NP2 (6+1+1/8): R0 4c2w1u1m; R1-F0 X33472000 Z33488704 a5.184 4c2w; R2-F0 4c0w3u1m. Paired gross ledger-verified: net correct 0; gross 1 wrong fixed (C1480 wrong→correct), 1 previous correct withheld (B1653 correct→UNKNOWN), 1 wrong withheld (C1481 wrong→UNKNOWN); B1652/1654/1655 stay correct, C1479 stays unresolved (mixed), C1478 stays missing. Oracle 6c1u1m.
- NP3 (8/0/0): R0 4c4w; R1-F0 X2611200 Z2631344 a4.857 4c4w; R2-F0 6c0w2u. Paired gross ledger-verified: 2 wrong fixed (B323, B324 wrong→correct), 2 wrong withheld (B322, B325 wrong→UNKNOWN); all 4 D (33, 34, 35, 36) stay correct; net +2 correct. Oracle 8c.
- Full payload separate: NP1 33GT 25m5u3x; NP2 28GT 20m6u2x; NP3 25GT 20m4u1x. Unmatched = unrecognized, denominators keep uncertain, no scalar. UNRESOLVED never correct. Other = other-unidentified, not enrollment.

## Four dimensions

1. Capabilities: R0 no-op; R1 prospective multi-seal projection only (CONTINUE noop/unsupported_operation, OVERLAP/NOOP unsupported, duplicates already_separated, old-X already_separated via Z>=X, return does not block future OTHER; requestedX vs sealedZ recorded, segments list ranges/boundaries with chunk-separated Z, reference fixed); R2 post-terminal X-partition incl return, overlap UNRESOLVED, straddle UNKNOWN intact, gap UNRESOLVED.
2. Evidence quality: mapping support, valid+mask+speech join, tail EOS invalid, NONE/invalid/gap reset, mask/nonspeech NOOP, confirmed persists, coalesced duplicates, root-nonspeech mismatch reported.
3. Frontier lateness: R1 Z from flush ledger at avail (NP1 19825520, NP2 33488704, NP3 2631344) beyond window → window equals R0 (conditional, not end-to-end). R2 uses X with avail≤terminal (seal 7.282/7.281/7.297, NO wait, cutoff before compute). Margins 2.017/2.097/2.440. CPU measured nonzero volatile separate.
4. Unresolvable ASR loss: NP2 1 missing +1 mixed; straddle/gap unresolved NP1 1, NP2 +2, NP3 2.

## Guards (synthetic terminal object-end zero grace shared; actual end-to-end UNKNOWN)

Proxy excl punc fixed sets: R1 10, R2 9, T1 21, COMBINED 28 never aggregated, BC1 4, singles 6/4.

- R1 guard: F0 OVERLAP 3187200 + CURRENT 3191040 timely; R1 0 seals (overlap no CUT, continue noop); R0 9c1w; R2-F0 7c0w3u. Paired gross ledger-verified per direction: B32 wrong→unresolved; A492 correct→unresolved; A493 correct→unresolved; A486/487/488/489/490/491/494 stay correct. Wrong→unknown is adjudicated NOT a correct fix, unambiguous; recorded exact with no net-improvement claim.
- R2 guard: F0 OVERLAP 669440 out-of-scope + OTHER 672000 applied Z684896 a42.966 s44.896 n_seals 1; R1-F0 4c5w; R2-F0 4c3w2u (vs R0 3c6w).
- T1 guard: F0 669440/672000 out-of-scope invalid_scope, 702720 CURRENT return unsupported_operation (does not cancel, does not block), 712960 OVERLAP unsupported, 733440 OTHER applied Z746336 a46.832 s48.275 n_seals 1; R1-F0 12c9w; R2-F0 7c0w14u. Paired gross ledger-verified: 4 wrong fixed to correct (B54, B55, B56, B57 wrong→correct), 7 wrong withheld (B46, B48, B49, B50, B51, B52, B53 wrong→UNKNOWN), 7 correct withheld (A128, A132, A134, A135, A136, A137, A138 correct→UNKNOWN); A129/130/131 stay correct; net correct 10→7. GT 2 events (669440 out-scope, 732160 applied) R2-GT 13c7w1u.
- COMBINED [670592,755520] 28: F0 5 events; R1-F0 n_seals 2 applied X672000 Z684896 then X733440 Z746336 (return/overlap in between unsupported but legal, second already checked against 684896<733440 open range). History: invalid_scope, applied, unsupported_operation, unsupported, applied. R1-F0 14c14w; R2-F0 10c2w16u. Binary metric may not change vs single-seal but fragmentation and receipts reflect legal 2 cuts with distinct OTHER-1/OTHER-2 new-unknown segments.
- BC1 [9119360,9125280]: mapping None → actual no events, R0 0c4w only; oracle carried-3 nonbinding noted, no fabrication, no harm claim.
- Singles 6c/4c: no fires, zero confident wrong, proxy-only.

## Limits

Research 7 s clips not C56s. Reference/provider unknown. Guards capacity only never PASS causal/safety never adoption. ASR-guard absent 0 calls. Overlap UNRESOLVED not enrollment. ~24 window + 68 proxy words not representative. P3T/return HISTORY ONLY.
