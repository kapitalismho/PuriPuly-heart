# Gate 5 — production VAD replay (firered x C, frozen tau=0.05)

> V2 EVAL sessions reused as dev-only probe per program approval; no unbiased generalization claim; V3 fresh holdout required for selection claims.

> firered regime C only (sole plausible Gate 3+4 candidate); neovad collapsed 8/8 missed under the GT gate and is NOT replayed. Same 48 MAIN48 rows, same causal 1 s bind, same CommonPersistenceDecoder (500 ms confirmer / 300 ms sensitivity); ONLY the speech gate changes (GT anchor speech -> production Silero VAD peer-profile spans).

## GT-gate vs production-VAD (frozen tau=0.05)

| gate | false cuts (KEEP-n) | missed (CUT-n) | src_err p50/p90 (ms) | dec p50/p90 (ms) | CUT events / sens hits |
|---|---|---|---|---|---|
| GT-gate | 13/32 | 5/8 | -670/1098 | 500/500 | 31 / 967 |
| prod-VAD | 21/32 | 1/8 | -1150/1244 | 500/500 | 82 / 2328 |

Contamination (GT-derived, identical both gates): 877.7 s/h over 48 episodes.

Cross-check: the GT-gate row reproduces main firered-C exactly (13/32 false cuts, 5/8 missed, src_err -670/1098 ms, dec 500/500 ms) — lifecycle, binding, and decoder are identical; only the gate differs.

## Gate agreement (per 10 ms frame, GT anchor speech vs prod VAD)

- frames scored: 24000; agreement: 0.6522
- GT-speech frames gated OFF by production VAD: 138/14136
- production-gate-ON frames where GT is off: 8210
- eval windows with ZERO production-gate coverage: 0/48

## Retained-improvement fraction

- CUT successes GT-gate: 3/8; prod-VAD: 7/8; hit-count ratio = 2.333
- episode-level retention: 3/3 GT-detected CUT episodes still detected under prod-VAD
- false cuts GT-gate: 13; prod-VAD: 21
- read: the prod gate is strictly wider (drops only 138/14136 GT-speech frames, adds 8210 gate-on frames) — missed 5/8->1/8, false cuts 13->21. No VAD under-triggering; the cost is over-triggering on non-anchor speech.

## Verdict

Both good: integration clean — production gating preserves the GT-gate detections.

## CPU note (this machine, wall-time)

- production VAD step (512-sample process_chunk, n=326187): median 0.0828 ms, p95 0.1101 ms, max 0.7074 ms; VAD-only RTF 0.002703
- pVAD adapter step (10 ms frames): median 0.5002 ms, p95 0.6374 ms
- combined RTF (VAD + pVAD over eval audio 240 s): 0.1695
