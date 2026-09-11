# PSEM Ownership Residual Causes — Results (frozen diagnostic, no adoption)

Freeze `psem.ownership_residual_causes.freeze.v1` frozen 2026-09-10T10:19:01.216980+00:00 (actual clock, before first probe run).
Ledger `ledger.json` generated 2026-09-10T10:29:19.160807+00:00, wall 26.9 s local, smoke 8/8 PASS, zero paid API, zero new captures, zero Git mutations, only new dir `experiments/psem_ownership_residual_causes/`.

Baseline recompute verifies accepted reference exactly (all NP R2.actualF0 and R2.GT_STATE rows match receiver ledger 97b2587c; guards R1/R2/T1/COMBINED/BC1/singletons rescoring matches). No accepted file edited.

## Shared contract (all 4 arms)

Same receiver R2 via imported `r2_partition`, same `<fin>` cutoff seals (NP1 7.282 / NP2 7.281 / NP3 7.297, NO wait, cutoff before compute), same causal availability, same valid mask (`valid_native` AND `valid_old`, accepted mask fixed, never relaxed), same CONFIRMATION 1600 (100 ms), same TAU 0.5, same FRAME 1280 (80 ms), same slot anchor (NP1 slot 2 ready 19658240 / NP2 slot 2 ready 33463040 / NP3 slot 1 ready 2549760), same ordered events and reference. Four arms cross two interval conditions with two state conditions:

- AA: ASR `start_src`/`end_src` + actualF0 (baseline, frozen correspondence before substitutions).
- AG: ASR intervals + EXISTING GT_STATE frame-quantized arm (oracle-marked).
- GA: matched GT word intervals + actualF0 (whole-word GT oracle timing diagnostic only, unreachable, no deploy claim).
- GG: matched GT word intervals + GT_STATE (oracle-oracle upper diagnostic only).

Same accepted text/string groups and GT IDs every arm (conservation exact, DROP/DUP/TEXT throwaway detected). Mixed (NP2 C1479) and unmatched (NP2 C1478) keep ASR unchanged, no fabricated word groups. No matched word maps ambiguous multi-group (ambiguous list empty all NP), so no `not_identifiable` rows occur; the check runs and would mark without forcing.

GT_STATE uses the SAME 80 ms frame quantizer and the SAME valid mask as actualF0. It is NOT ideal continuous state. ActualF0 versus GT_STATE equality does NOT prove information or model sufficient. In-pay events are identical both states on all 3 NP (NP1 3 events with 1 in-pay at 19808000; NP2 1 in-pay at 33472000; NP3 1 in-pay at 2611200), so AG equals AA everywhere by construction of the shared decode path, not by proof of sufficiency.

## Inputs (same ASR text every arm)

Full SHA256:

- `ARCHITECTURE.md` `8971048f129d49b48a3abf6b9d546906e155eee5ed3b1e845d60e281e062d068`
- `experiments/psem_pretranslation_receiver/FREEZE.json` `07527fddb22b3bcdde2ad893be80d6804f83c08b8fef85278c34a79df210358b`
- `experiments/psem_pretranslation_receiver/replay.py` `10494780cdee21c77835999859d461239ea5f72abb16076164cfe1b8fa0840c3`
- `experiments/psem_pretranslation_receiver/ledger.json` `97b2587c56dbea6c0d07f11a99293e496e0f30fecd5197e3b0ed12f827878c2f`
- `experiments/psem_pretranslation_receiver/RESULTS.md` `51fa0f680ca4844406c9dd68824f4a7eb007f278daaffa929cd3fca11caf9a4f`
- `experiments/psem_pretranslation_ontology/FREEZE.json` `bede639be8c65ef74f7f1785b89b8804c40dfe3b0937a73abb63b8f5ff4355a4`
- `experiments/psem_pretranslation_ontology/probe.py` `f2703f5cf34b8a633dc90566fde7e3023535895fc7e39895657f40e5b31a660f`
- `experiments/psem_pretranslation_ontology/ledger.json` `501f222867939b5606f1ae98a7757d1ea3100586149cb5c488974fd0fc0ec1fc`
- `experiments/psem_pretranslation_ontology/RESULTS.md` `6a91d3626773e678dfeb0c509a68d50cc60f5c06bc439eacd207cfdb7e70ab14`
- `experiments/psem_repeatability_stage2/captures/NP1.json` `a4ed5e84f9fd866263646660edde1c5400f1269de1f58a98befa044b19da5005`
- `experiments/psem_repeatability_stage2/captures/NP2.json` `a67b4a9953a1a08a55225b221f62af6208c767b67bedff6cdd72feef1dd76177`
- `experiments/psem_repeatability_stage2/captures/NP3.json` `d92884ae50dcbc4e841c5dbc13c805665130dcec9fddc43a4745a32ed329b22d`
- `experiments/psem_phase_a_headroom/observations/OBSERVATIONS.json` `3ec7c03c9c01bff859b9d6ce8d020aec40432c9deb7b951e1cf19be11a90b5aa`
- `experiments/psem_phase_a_headroom/old_grid_cache.json` `8f9474eecbd13c235062bbe8616f93de5b8b6b6316695d9864724b87dbb5814a`
- `experiments/psem_decision_sufficiency/FREEZE.json` `d131883413aa969c0aa6ed5eb32ac134f411ed511af372baa3bf6e5d88d5ac67`
- `experiments/psem_decision_sufficiency/replay.py` `2e08a9ece97c0b8f80730450f5f05124e7b3f97d6268c308af2603720700753d`

Setup: `.venv/Scripts/python.exe experiments/psem_ownership_residual_causes/probe.py smoke` (8/8 PASS) then `.../probe.py run` (wall 26.5 s). Margins seal-minus-avail: NP1 2.017, NP2 2.097, NP3 2.440 (all preterminal about 2 s). Existing plus-0/100/300 extensions already prior-verified with no new distinct boundaries; NOT rerun here; unavailable post-tail absence inferred, not measured.

## NP window8 (fixed 24 cohort; PAIRED gross PRIMARY, net never substitutes gross)

Conservation exact every case and every arm (GT substitution changes only `start_src`/`end_src`, never text or token refs). Baseline AA equals accepted R2.actualF0 row-for-row; AG equals accepted R2.GT_STATE row-for-row.

| Case | R0 | AA (=acc) | AG (=acc GT) | GA (GT oracle) | GG (GT oracle) |
|---|---|---|---|---|---|
| NP1 8/0/0 | 4c4w | 7c0w1u | 7c0w1u | 7c0w1u | 7c0w1u |
| NP2 6+1+1/8 | 4c2w1u1m | 4c0w3u1m | 4c0w3u1m | 2c0w5u1m | 2c0w5u1m |
| NP3 8/0/0 | 4c4w | 6c0w2u | 6c0w2u | 6c0w2u | 6c0w2u |
| Total 24 | — | 17c0w6u1m | 17c0w6u1m | 15c0w8u1m | 15c0w8u1m |

Gross per-word branch changes (ledger `per_word` read directly):

- NP1: A1117 `What` g11 AA straddle-UNRESOLVED → GA state-at-end OTHER (fix by oracle timing). A1120 `cost` g14 AA OTHER correct → GA unsupported-gap UNRESOLVED (correct-withheld by oracle timing overshoot, see validity below). All other 6 stay correct. Gross 1 fix + 1 correct-withheld, net 0. Counts stay 7c0w1u but identity of the unknown rotates.
- NP2: B1652 `only` g7 AA CURRENT correct → GA gap-UNRESOLVED. C1480 `another` g12 AA OTHER correct → GA gap-UNRESOLVED. B1653/C1481/C1479/C1478 unchanged (still 3u1m). Gross 0 fixes + 2 correct-withheld, net −2 correct. Oracle timing strictly harms here.
- NP3: B322 g13 straddle-UNRESOLVED in AA and GA (persists under oracle timing). B325 g16 gap-UNRESOLVED in AA and GA (persists). All other 6 stay correct. Gross 0/0, counts identical.

AG equals AA on every word (no state delta in-pay); GG equals GA on every word. State substitution alone moves nothing.

## Residual-by-residual cause isolation

Frame counts versus duration-weighted overlaps are listed separately and never confused. Frame `valid` means `valid_native` AND `valid_old` (accepted mask). Duration overlap is the sample intersection of the interval with each frame, summed by validity.

### NP1 A1117 What g11 — boundary straddle, no gap

- ASR [19805360,19809200) dur 3840: 4 frames 15472–15475, 4 valid / 0 invalid; dur valid 3840 / invalid 0. Head frames 15472–15474 speech False (silence), tail frame 15475 speech True.
- GT [19808000,19809600) dur 1600: 2 frames 15475 (overlap 1280) + 15476 (overlap 320), 2 valid / 0 invalid; dur valid 1600 / invalid 0.
- Event b 19808000 (OTHER_ONLY, avail 5.265, margin 2.017) strictly inside ASR (19805360 < 19808000 < 19809200) → `straddle-inside-word UNKNOWN intact`. GT start equals b exactly (19808000 < 19808000 false) → no straddle → state-at-end OTHER correct.
- F0 versus GT_STATE candidates on frames 15472–15476: NONE/NONE/NONE/OTHER_ONLY/OTHER_ONLY both states, identical. Equality reflects the shared quantizer path, not proof of sufficiency.
- Quantization diagnostic (independent raw AMI bounds, NOT a boundary-policy probe): previous GT speaker B1157 ends 19802080; silence 5920 samples (370 ms) to A1117 start 19808000; ASR starts 2640 early inside that silence; event b sits exactly on the GT start sample, which is exactly frame 15475 start (15475 × 1280 = 19808000). Other speech right of b, NONE left. The straddle is an ASR onset-early timing cause. No claim of real multi-speaker ambiguity is made: GT word overlap with the quantized boundary is a grid-alignment fact, not proof of two physical speakers inside one word.

### NP3 B322 Well g13 — quantized state boundary, straddle persists under oracle

- ASR [2608240,2614000) dur 5760: 6 frames 2037–2042, 6 valid / 0 invalid; dur valid 5760 / invalid 0.
- GT [2610720,2613120) dur 2400: 3 frames 2039 (overlap 480) + 2040 (overlap 1280) + 2041 (overlap 640), sum 2400; 3 valid / 0 invalid; dur valid 2400 / invalid 0.
- Event b 2611200 (OTHER_ONLY, avail 4.857, margin 2.440) strictly inside BOTH intervals: ASR 2608240 < 2611200 < 2614000 and GT 2610720 < 2611200 < 2613120 (offset 480 samples = 30 ms inside the GT word). Both arms report `straddle-inside-word UNKNOWN intact`.
- 480 samples inside a GT word is NOT real overlap proof. Quantization diagnostic: previous GT speaker D36 ends 2608320; silence 2400 samples (150 ms) to B322 start 2610720; ASR starts 2608240 (80 before D36 end, spanning the silence); event b is exactly frame 2040 start (2040 × 1280 = 2611200). This is a QUANTIZED STATE BOUNDARY inside one GT word, not physical two-speakers-within-same-word. Never claim otherwise.
- F0 versus GT_STATE differ on one pre-boundary frame (f2039: F0 NONE with ref 0.000 versus GT OTHER_ONLY) but decode events are identical (single in-pay OTHER event both states) because the shared valid mask, nonspeech skip, and 100 ms confirmation absorb the single-frame candidate difference. Equality does NOT prove the model sufficient.

### NP2 B1653 song g8 — validity gap (legacy mask), timing cannot fix

- ASR [33452800,33460480) dur 7680: 6 frames 26135–26140, 0 valid / 6 invalid; dur valid 0 / invalid 7680 (100% both metrics).
- GT [33458560,33462400) dur 3840: 4 frames 26139 (overlap 640, invalid) + 26140 (1280, invalid) + 26141 (1280, valid) + 26142 (overlap 640, valid): 2 valid / 2 invalid by count (50%); duration valid 1920 / invalid 1920 (50%). Oracle timing halves the invalid span but 1920 invalid samples remain, so `unsupported-gap UNRESOLVED` persists in GA/GG with prior state-at-end CURRENT.
- F0 versus GT_STATE on frames 26135–26142: F0 CURRENT_ONLY/NONE pattern (ref 0.989/0.745/0.321/0.284/0.448/0.983) versus GT all CURRENT_ONLY; the three NONE frames (26137–26139) sit inside the invalid span where decode issues invalid-reset regardless of candidate, so in-pay events are identical (single OTHER at 33472000) and AG equals AA.
- Provenance (minimum inspection): `old_grid_cache.json` provenance is `derived read-only cache from gpu_export via load_validated_export; old paths unmutated`. Zero old rows overlap [33452800,33460480). Nearest rows end at 33452800 (episode A00270) and resume at 33461184 (episode A00271): an 8384-sample inter-episode coverage gap. Whether scheduling/mapping artifact or true invalid input is NOT established → label `legacy validity mask`, NOT no-acoustic-model-information. Native probs remain present on all 6 frames (`valid_native` True) but invalid blocks adoption; a native-only probe is NOT in this contract.

### NP2 C1481 one g13 — validity gap (legacy mask), oracle timing worse

- ASR [33478720,33485440) dur 6720: 6 frames 26155 (overlap 960, valid) + 26156–26159 full invalid + 26160 (overlap 640, invalid): 1 valid / 5 invalid by count (83.3%); duration valid 960 / invalid 5760 (85.7%).
- GT [33480320,33490400) dur 10080: 9 frames 26156–26164, 0 valid / 9 invalid by count (100%); duration invalid 10080 (100%). The GT tail extends past old-grid max 33479616 into 4 frames with no rows at all. GA/GG stay `unsupported-gap UNRESOLVED` with prior OTHER.
- F0 versus GT_STATE on the span: frame 26155 both OTHER_ONLY; frames 26156–26160 both OTHER_ONLY where F0 has probs (other_max about 0.98) but invalid; frames 26161–26164 F0 NONE versus GT OTHER_ONLY, all invalid. Events identical both states.
- Provenance: head invalid frames 26156–26160 fall in the post-episode tail (episode A00271 ends 33479616; old rows stop there); tail frames 26161–26164 have no old rows at all. NOT established as scheduling versus true-invalid → `legacy validity mask`. Native `valid_native` True throughout; probs present but unusable under the accepted mask.

### NP3 B325 thing g16 — validity gap (legacy mask), single-frame

- ASR [2618800,2622640) dur 3840: 4 frames 2045 (overlap 80, valid) + 2046 (1280, valid) + 2047 (1280, invalid) + 2048 (overlap 1200, valid): 3 valid / 1 invalid by count (25.0%); duration valid 2560 / invalid 1280 (33.3%). Count and duration differ because edge frames are partial — reported separately, never equated.
- GT [2618560,2623360) dur 4800: 5 frames 2045–2049, 4 valid / 1 invalid by count (20.0%); duration valid 3520 / invalid 1280 (26.7%). The single invalid frame 2047 persists under oracle timing, so GA/GG stay `unsupported-gap UNRESOLVED` with prior OTHER.
- F0 versus GT_STATE on frames 2045–2049 identical (OTHER_ONLY throughout, other_max about 0.99), including on invalid frame 2047. Events identical.
- Provenance: zero old rows overlap [2620160,2621440). Nearest rows end 2619136 (episode A00006) and resume 2622400 (episode A00007): a 3264-sample inter-episode coverage gap. NOT established → `legacy validity mask`. Native present, blocked.

### NP2 C1479 mixed + C1478 unmatched — inherent alignment unknown + missing separate

- C1479 `is` GT [33473600,33475360): alignment `mixed` with reason `opcode-replace`, candidate group idxs [4, 14] (both ASR `Is` words far from the GT time, so even though an `Is`-like token exists in the ASR text the match stays uncertain). Strictly the current aligner cannot identify a single corresponding ASR interval; substitution kept unchanged per contract, no word group fabricated. Scored UNRESOLVED all 4 arms by the fixed scoring rule (mixed never correct). Inherent oracle cost 1u under this aligner. GT frames 26151–26152 are both valid with F0 OTHER_ONLY, but there is no owned interval to attach them to. Mixed here does not prove the word is absent from the ASR audio or deleted by the ASR; it proves only that the current aligner cannot place it.
- C1478 `There` GT [33471360,33473600): alignment `unmatched` with reason `deleted`, zero candidate groups under the current aligner (even though ASR g11 `There's` exists nearby, its normalized text and time support do not yield an identified match, so the match stays uncertain). Strictly the current aligner cannot identify any corresponding token; substitution kept unchanged per contract. Scored MISSING all arms. Missing 1 separate: unrecognized under this aligner, denominator kept, no scalar. GT frames 26149–26151 are valid. Unmatched here does not prove ASR deletion or word absence; it proves only that the current aligner identifies no correspondence.
- Together the inherent oracle floor is 1u1m on NP2 regardless of interval or state arm.

## Guards (capacity only, never PASS causal/safety, never adoption)

Exact current-baseline rescoring via imported `run_guards`, `r2_partition_proxy`, `score_proxy`. All match accepted ledger byte-for-byte:

- R1 (10 proxy): R2.actualF0 7c0w3u, all 3 strict straddle (A492, B32, A493). GT_STATE identical counts.
- R2 (9 proxy): R2.actualF0 4c3w2u, 2 strict straddle (A124, B42). GT_STATE 3c6w0u (rescoring nominal, proxy helper has no gap check).
- T1 (21 proxy): R2.actualF0 7c0w14u = 5 strict straddle (B46, A128, A132, B48, B53) + 9 overlap-state CURRENT_PLUS_OTHER. GT_STATE 13c7w1u.
- COMBINED (28 proxy, overlaps T1+R2 spans so NOT summed): R2.actualF0 10c2w16u = 7 strict straddle + 9 overlap-state. GT_STATE 15c12w1u.
- BC1 (4 proxy): unmapped, actual no events, 0c4w all arms (guarded, not an actual wrong-zero claim).
- SINGLE_ES2009c (6 proxy): R2.actualF0 6c0w0u recomputed, matches accepted ledger; GT_STATE 6c0w0u. No fires beyond the no-cut baseline, zero confident wrong.
- SINGLE_ES2009d (4 proxy): R2.actualF0 4c0w0u recomputed, matches accepted ledger; GT_STATE 4c0w0u. No fires beyond the no-cut baseline, zero confident wrong.

Proxy words are already GT times, so the ASR-versus-GT timing control is N/A here: rerunning interval substitution would not be new ASR evidence. Physical overlap versus GT word-span proxy are kept separate; the proxy straddle counts above are word-span proxies, not physical-overlap measurements. The proxy helper performs no gap/validity check, so a cross-guard failure cannot condemn NP gap repair. Soniox data ignored entirely.

## Four dimensions (same as accepted)

1. Capabilities: R0 no-op; R2 post-terminal X-partition including return, overlap UNRESOLVED, straddle UNKNOWN intact, gap UNRESOLVED. R1 accepted-history reference only.
2. Evidence quality: shared mapping support, valid+mask+speech join, tail EOS invalid, NONE/invalid/gap reset, mask/nonspeech skip, confirmed persists, coalesced duplicates.
3. Frontier lateness: R2 uses X with avail ≤ terminal (seals 7.282/7.281/7.297, NO wait, cutoff before compute). Margins 2.017/2.097/2.440. Preterminal about 2 s. CPU measured nonzero volatile, reported separately in ledger (decode microseconds, wall 26.9 s total).
4. Unresolvable ASR loss under accepted mask: NP2 1 missing + 1 mixed; straddle/gap unknowns NP1 1, NP2 +2 (mixed excluded from gap count), NP3 2. Unknown kept in denominators; UNRESOLVED never correct.

## Limits and disposition

Research 7 s clips, not C56s. Reference/provider unknown. Guards capacity only. ASR-guard absent, 0 calls. Overlap UNRESOLVED is not enrollment. 24 window + 68 proxy words not representative. P3T/return history only. Whole-word GT intervals are unreachable oracle diagnostics with no deploy claim. GT_STATE is the same quantizer/mask, not an ideal-state upper bound.

Disposition: HOLD ADOPTION. No policy change is qualified by this control: A1117 isolates to ASR onset timing but its oracle fix is offset by A1120 overshoot into the legacy tail; B322 proves a quantized-boundary straddle that oracle timing cannot remove; three validity gaps persist or worsen under oracle timing because the legacy mask, not the interval, binds; C1479/C1478 are inherent alignment floor. This HOLD reason is classified residual-cause isolation: it is NOT a PSEM stop, and it is no proof that policy cannot help. A native-only probe (using present-but-invalid native probs) is explicitly NOT in this contract.
