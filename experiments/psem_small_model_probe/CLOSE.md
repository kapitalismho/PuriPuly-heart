# PSEM small-model probe — close-out (issue #117, evaluator_revision=2)

> REPAIRED outputs only (`evaluator_revision=2`, `*/results_repaired/`).
> V2 EVAL sessions reused dev-only per program approval — no generalization
> claim; V3 holdout required for any selection claim. Engineering branch
> only: no SOTA, no impossibility, no cross-corpus claims.

## Evaluator repair (why rev 1 is VOID)

- Decoder gate was anchor-only speech: B-only speech starved the 500 ms
  persistence decoder, so clean A->B handoffs could never accumulate.
  Now `speech_gt` := GT any-speech (unmasked, non-empty active_speakers);
  `anchor_speech_gt` kept as diagnostic only. Shared semantics live in
  `cal/eval_semantics.py` (single location; runners import, never duplicate).
- CUT validity is transition-aware: valid iff `source_boundary_time >=
  authoritative_transition - 50 ms`. Tolerance = 50 ms reuses
  `constants_ms.annotation_boundary_jitter` from the V2
  `operational_label_contract.json` (psem-handoff-v1; v0 carries the same
  50 ms). Episode success := first valid CUT; premature-only := missed +
  premature_cut (+ n_premature_cuts). KEEP false-cut usage unchanged.
- Contamination is decoder-dependent: per-episode numerator over
  `[eval_start, first valid CUT source_boundary)` (full window if none),
  using source_boundary_time never decision_time; denominator
  (active-speech hour) unchanged. It now varies across behaviors by design.
- Old tau-0.05-everywhere thresholds are VOID (calibrated under the broken
  gate). `cal/results/thresholds.json`, `main/results/*`,
  `vadreplay/results/*` preserved in place, never overwritten.
- V1 unit tests: `cal/test_eval_semantics.py`, 8 tests / 24 assertions PASS
  (KEEP, B-only 500 ms CUT, silence reset, premature-only scoring,
  contamination divergence, tolerance edge, decoder gate use).

## Gate table 0–6 (repaired numbers)

| Gate | Verdict | Key numbers |
|---|---|---|
| 0 manifest freeze | FROZEN / PASS | 84 rows: CAL12 12 (5 sess), MAIN48 48 (6), EXT24 24 (4), ONTOLOGY16 16 + CONTROL24 24 (MAIN48 subsets). `file_sha256 e5956ab0…6582`, `freeze_sha256 91efb276…6d82`. Session-disjoint (CAL∩MAIN={}, EXT∩(CAL∪MAIN)={}); `episode_id` unique 84/84; `causal_bindable` 84/84. Verified by hash before every repaired run. |
| 1 adapter + decoder | PASS (unchanged, decoder contract untouched) | smoke 6/6; `verify_vendor` 16/16 contracts (+20/20 live on synthetic PCM per LOCALENV). ECAPA receipt sha `0575cb64…e3d0126a`. CPU-only. 15/15 sessions resolved, mono 16 kHz, frame-count match. |
| 2 CAL12 taus (rev2) | FROZEN | firered O tau 0.15 (false 5/8, missed 1/2, median total 560 ms); firered C tau 0.05 (false 4/8, missed 1/2, 610 ms); neovad O+C tau 0.15 (false 0/8, missed 1/2, 2700 ms). Existing TAU_GRID 0.05–0.95 + fixed-priority rule. Raw anchor inference reused (native, stub=false); gates/GT/decisions regenerated frozen. Contam now per-cell: 475.1 / 534.4 / 674.5 s/h. |
| 3 MAIN48 native O | firered SUPPORTED / neovad COLLAPSED | New frozen taus applied as-is. firered O: contam 712.8 s/h, 21/32 false, 6/8 missed (CUT 2/8), src_err p50/p90 1635/1711 ms, dec 500/500 ms, premature 4. neovad O: 867.5 s/h, 4/32 false, 8/8 missed, no delays, premature 0. ≥1 formulation supported → proceeded to Gate 4. |
| 4 MAIN48 causal C + CPU | O→C FLAT; CPU PASS (reused live-run timings) | firered C: 749.4 s/h, 17/32 false, 6/8 missed (CUT 2/8), src 1635/1711 ms, dec 500/500 ms, premature 2. neovad C collapsed (0/8 CUT). CPU all 4 cells `rtf_le_025` + `p99_lt_chunk` (p99 0.7–1.2 ms vs 10 ms chunk; RTF 0.051–0.065; this machine only). |
| 5 VAD replay (firered C only) | INTEGRATION CLEAN | GT any-speech vs prod Silero spans (same profile): missed 6/8→4/8, false 17→21/32, src_p50 1635→1055 ms; retention 2/2 GT CUT hits (+2 more), hit ratio 2. Agreement 0.8336 (24k frames); VAD drops 239/18693 GT-speech frames, adds 3754 gate-on frames, 0/48 zero-coverage. Contam GT 749.4 → prod 720.4 s/h. VAD RTF 0.0029, combined 0.177. |
| 6 ontology | NOT RERUN — prior proxy results stand as marked | Pre-repair proxy: REOPEN ownership 8/16 (rule ≥4); loss-risk increase 0/16; both-poor 8/16. Blind X/Y seed 117. GT-proxy substitution (no real ASR/translation). `ontology/*`, `compare/*` untouched per workstream freeze. |

Topology note: MAIN48 carries zero `A->A+B->A` rows (mandatory KEEP cell
0/0 all cells); KEEP rests on A / overlap_return / A+A+B (n=32). ONTOLOGY16
session concentration: all 16 eps from ES2009a + R1021_M1940 only.

## Q11 answers (repaired numbers, engineering level only)

- A — FireRed O is weak partial signal, NOT promotable: 2/8 valid CUT vs
  21/32 false cuts. SUPPORTED means "proceeded to Gate 4", not "passed".
- B — O≈C flat (missed 6/8 both; CUT 2/8 both; src 1635/1711 both), so the
  5 s→1 s bind tightening is NOT the bottleneck; the observation model is.
- C — NeoVAD unusable as anchor tracker both regimes: 0/8 CUT, 0 sens on
  CUT episodes, no valid source boundaries (delays None) under either gate.
- D — Production VAD clean: retains 2/2 GT-detected CUT episodes, misses
  fall 6→4/8 for +4 false cuts (17→21). Cost is over-triggering on
  non-anchor speech, not under-triggering; nothing in the VAD hides signal.

## Gate 7: EXT24 / CONTROL24 (not opened, per handoff §10)

- 10%-boundary: NOT triggered — no headline near a pass boundary: FireRed
  false cuts 17–21/32 (far above any x1.10-style budget), NeoVAD 8/8 missed
  (collapsed, opposite end).
- corpus-opposite: NOT triggered — no AMI↔AliMeeting reversal behind any
  gate verdict; pooled headlines decide.
- overlap-conflict: NOT triggered — overlap_return sens-vs-primary
  divergence recorded diagnostic-only, no threshold/frontier action.
- CONTROL24: SKIPPED per minimum-validation — ECAPA adapter exists (Gate 1
  live) but FireRed is unpromotable, so the comparison would not change
  action.

## Known limitations

- MAIN48 zero `A->A+B->A` rows (mandatory KEEP cell 0/0).
- ONTOLOGY16 GT-proxy, not real ASR/translation (weights uncached, no LLM
  owner); proxy verdicts stand as marked, not refreshed to rev 2.
- EVAL dev-only; V3 fresh holdout needed before any model-selection claim.
- CPU numbers this-machine only, partly reused live-run timings (Win11,
  Ryzen 7 9800X3D); bind/reset/RSS not re-timed under reuse.
- Calibration diagnostics (frame AUPRC/F1, sens streams) remain
  diagnostic-only, never selection inputs.

## Follow-up candidates (open separately, not implemented)

- FireRed operating-point / policy work (false-cut budget is the blocker).
- NeoVAD foreground-transfer-evidence-only reuse.
- Ownership/transfer primitive (needs real-ASR confirmation).
- Extraction / conditioned-ASR (both-poor 8/16).
