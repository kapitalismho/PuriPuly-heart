# F0 on MAIN48 (frozen Sortformer, cached posteriors only)

coverage: 48/48 scored, missing=none
tau_f0=0.5 (predeclared), frame 10ms, confirmation 500ms, GT speech gate, oracle anchor-slot per episode

| model | contam s/h | false cuts (KEEP-n) | missed (CUT-n) | src_err p50/p90 (ms) | dec p50/p90 (ms) |
|---|---|---|---|---|---|
| F0-sortformer | 877.7 | 5/32 | 7/8 | 2440.0/2440.0 | 500.0/500.0 |
| firered O (main/) | 877.7 | 12/32 | 5/8 | -820.0/1068.0 | 500.0/500.0 |
| neovad O (main/) | 877.7 | 0/32 | 8/8 | None/None | None/None |
| firered C (main/) | 877.7 | 13/32 | 5/8 | -670.0/1098.0 | 500.0/500.0 |
| neovad C (main/) | 877.7 | 0/32 | 8/8 | None/None | None/None |

A->A+B->A KEEP: n/a (MAIN48 carries zero rows) | A->A+B->B CUT: F0 1/8 (missed 7)
F0 KEEP by topology (same KEEP shape as main/):
- A: 16/16 (false 0)
- overlap_return: 10/12 (false 2)
- A+A+B: 1/4 (false 3)
frame coverage: min 0.692, 26/48 episodes with trace gaps (gap frames UNCERTAIN/HOLD)

## #117 fallback promotion check vs F0 (contam+misses both improve, one >=20%, false <= F0x1.10, dec p90 <= F0+100ms)

- firered O: FAIL — contam 877.7 vs F0 877.7: NO improvement; missed 5/8 vs F0 7/8: improves; >=20% on one: yes (contam n/a, missed 28.6%); false cuts 12/32 vs F0 5 (limit 5.50): EXCEEDS; dec p90 500.0 vs F0 500.0+100: ok
- neovad O: FAIL — contam 877.7 vs F0 877.7: NO improvement; missed 8/8 vs F0 7/8: NO improvement; >=20% on one: NO (contam n/a, missed n/a); false cuts 0/32 vs F0 5 (limit 5.50): ok; p90: neovad-O has no CUT detections (missed all) — cannot pass
- firered C: FAIL — contam 877.7 vs F0 877.7: NO improvement; missed 5/8 vs F0 7/8: improves; >=20% on one: yes (contam n/a, missed 28.6%); false cuts 13/32 vs F0 5 (limit 5.50): EXCEEDS; dec p90 500.0 vs F0 500.0+100: ok
- neovad C: FAIL — contam 877.7 vs F0 877.7: NO improvement; missed 8/8 vs F0 7/8: NO improvement; >=20% on one: NO (contam n/a, missed n/a); false cuts 0/32 vs F0 5 (limit 5.50): ok; p90: neovad-C has no CUT detections (missed all) — cannot pass
