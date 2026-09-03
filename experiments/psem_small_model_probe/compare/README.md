# F0 (frozen Sortformer) vs FireRed/NeoVAD on MAIN48 — head-to-head note

Identical episode set: 48/48 MAIN48 scored for F0 (missing: none).
Small-model rows reused verbatim from `main/results/summary.md` (frozen
taus, no rescoring, no retuning). F0 scored by `run_compare.py` with the
main/ rules: GT speech gate, 500 ms confirmation
(`CommonPersistenceDecoder` via `cal.metrics`), headline metrics via
`cal.metrics.score_episode/aggregate`.

| model | tau | contam s/h | false cuts (KEEP-n) | missed (CUT-n) | src_err p50/p90 (ms) | dec p50/p90 (ms) |
|---|---|---|---|---|---|---|
| F0-sortformer | 0.5 | 877.7 | 5/32 | 7/8 | 2440.0/2440.0 | 500.0/500.0 |
| firered O | 0.05 | 877.7 | 12/32 | 5/8 | -820.0/1068.0 | 500.0/500.0 |
| neovad O | 0.05 | 877.7 | 0/32 | 8/8 | None/None | None/None |
| firered C | 0.05 | 877.7 | 13/32 | 5/8 | -670.0/1098.0 | 500.0/500.0 |
| neovad C | 0.05 | 877.7 | 0/32 | 8/8 | None/None | None/None |

Topology views: A->A+B->A KEEP n/a (MAIN48 carries zero rows, all models).
A->A+B->B CUT: F0 1/8 (sole detection R1021_M1940:A00008, src_err +2440 ms)
vs firered 3/8 (O and C) vs neovad 0/8 (O and C). F0 KEEP by topology:
A 16/16, overlap_return 10/12, A+A+B 1/4. False-cut episodes:
ES2009a:A00002/A00004 (overlap_return), ES2009b:A00003, R1021_M1944:A00003/A00004
(A+A+B).

## #117 fallback promotion check vs F0 (F0/G exact-gap unavailable here)

Rule: contamination AND misses both improve over F0, one >=20% relative,
false cuts <= F0 x 1.10 (= 5.50), dec p90 <= F0+100 ms.

- firered O: FAIL (misses 5/8 vs 7/8 improves 28.6%; contam tied 877.7;
  false 12 > 5.50; dec p90 ok).
- firered C: FAIL (same shape: misses improve 28.6%; contam tied;
  false 13 > 5.50; dec p90 ok).
- neovad O/C: FAIL (misses 8/8 worse than F0 7/8; no CUT detections at all).

Structural note: contamination is GT-window-derived, hence model-independent
on a fixed episode set (all five rows read 877.7 by construction). The
"both improve" leg is therefore unsatisfiable on identical sets — the rule
can only discriminate on misses/false-cuts/delay here. Verdict: no small
model promotes over frozen F0 on MAIN48 under the fallback rule; firered is
the only candidate that beats F0 on any leg (misses), at the cost of 2.2-2.4x
F0's false cuts. (F0 itself misses 7/8 CUTs — neither side owns this subset.)

## Mapping assumptions (explicit)

- Session+time map: `(corpus.lower(), session_id)` -> cached posterior
  session; 16 kHz half-open `[start_sample, end_sample)` cells. No new
  inference; cached posteriors only.
- Anchor slot per episode: oracle `model_decode.py` pattern
  (duration-weighted mean of `probabilities * alive` over trace-valid,
  unmasked, GT anchor-active support cells), restricted to the eval window
  (MAIN48 has no emit spans). Slots landed 44x slot-0 / 4x slot-1. Oracle
  input (GT) favours F0; small models bind from audio instead.
- Operating point tau=0.5: predeclared F0 anchor threshold
  (`config.json` current_anchor_threshold 0.5; probe grid 0.35/0.5/0.65).
  Single point, no sweep. Invalid/dead/gap cells -> UNCERTAIN (HOLD +
  persistence reset), mirroring `relative_probabilities` None.
- Trace gaps: 26/48 episodes have dropped cached cells (~200 ms typical,
  min frame coverage 0.692); gap frames score UNCERTAIN. F0 evidence
  frontiers are recorded in the cache but emit uses source-time accounting
  like main/ (frontier delay not modelled).

## Runtime (no new Sortformer benchmarking)

- F0 config per receipts (`dev/eval_sortformer_model_receipt.json`):
  streaming Sortformer 4spk Q8_0, native frame 80 ms, 480 ms chunks,
  Vulkan `low_latency` backend, 8 threads, recorded algorithmic lookahead
  1040 ms. (Cached eval cells step at 100 ms / 1600 samples.)
- Small-model measured CPU per `main/results/cpu.json` (10 ms frames,
  240 s audio, torch 8+8 threads, 16 CPUs): firered RTF 0.051-0.052,
  step p50 ~0.50 ms; neovad RTF ~0.065, step p50 ~0.61 ms; all gates pass
  (p99 < chunk, RTF <= 0.25), no stub fallback.
- Honest incomparability: receipt-quoted GPU/Vulkan streaming config vs
  measured CPU step times on different hardware/backends — not a
  like-for-like latency comparison, quoted side by side only.
