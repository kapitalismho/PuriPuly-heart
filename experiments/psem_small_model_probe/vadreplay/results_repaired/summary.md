# Gate 5 — production VAD replay, repaired evaluator (rev 2) (firered x C, frozen tau=0.05)

> V2 EVAL sessions reused as dev-only probe per program approval; no unbiased generalization claim; V3 fresh holdout required for selection claims.

> Winner-only: firered regime C (sole causal formulation with valid CUTs under repaired MAIN48: 2/8; neovad-C 0/8, nothing to retain). Same 48 MAIN48 rows, same causal bind, same 500 ms confirmer / 300 ms sensitivity; gates compared: GT any-speech vs production Silero VAD spans (thr 0.5, chunk 512, pre-roll/hangover 500 ms).

## GT any-speech gate vs production-VAD (frozen tau=0.05)

| gate | false cuts (KEEP-n) | missed (CUT-n) | src_err p50/p90 (ms) | dec p50/p90 (ms) | CUT events / sens hits | contam s/h |
|---|---|---|---|---|---|---|
| GT-gate | 17/32 | 6/8 | 1635/1711 | 500/500 | 44 / 1294 | 749.4 |
| prod-VAD | 21/32 | 4/8 | 1055/1277 | 500/500 | 82 / 2328 | 720.4 |

Cross-check: the GT-gate row reproduces main/results_repaired firered-C exactly (lifecycle, binding, decoder identical; only the gate differs).

## Gate agreement (per 10 ms frame, GT any-speech vs prod VAD)

- frames scored: 24000; agreement: 0.8336
- GT-speech frames gated OFF by production VAD: 239/18693
- production-gate-ON frames where GT is off: 3754
- eval windows with ZERO production-gate coverage: 0/48

## CUT retention (GT-gate -> prod-VAD)

- CUT successes GT-gate: 2/8; prod-VAD: 4/8; hit-count ratio = 2
- episode-level retention: 2/2 GT-detected CUT episodes still detected under prod-VAD
- false cuts GT-gate: 17; prod-VAD: 21

## Verdict

Both good: integration clean — production gating preserves the GT-gate detections.

## CPU note (this machine, wall-time)

- production VAD step (512-sample process_chunk, n=326187): median 0.0828 ms, p95 0.1339 ms, max 5.05 ms; VAD-only RTF 0.00289
- pVAD adapter step (10 ms frames, reused live-run timings): median 0.4968 ms, p95 0.6063 ms
- combined RTF (VAD + pVAD over eval audio 240 s): 0.1765
