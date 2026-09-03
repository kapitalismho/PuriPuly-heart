# MAIN48 native-ceiling (O) + causal (C) scoring — Gates 3+4

> V2 EVAL sessions reused as dev-only probe per program approval; no unbiased generalization claim; V3 fresh holdout required for selection claims.

> Frozen Gate 2 taus applied as-is (firered/neovad x O/C); no retuning on MAIN48.

| model | regime | tau | contam s/h | false cuts (KEEP-n) | missed (CUT-n) | src_err p50/p90 (ms) | dec p50/p90 (ms) |
|---|---|---|---|---|---|---|---|
| firered | O | 0.15 | 712.8 | 21/32 | 6/8 | 1635.0/1711.0 | 500.0/500.0 |
| neovad | O | 0.15 | 867.5 | 4/32 | 8/8 | None/None | None/None |
| firered | C | 0.05 | 749.4 | 17/32 | 6/8 | 1635.0/1711.0 | 500.0/500.0 |
| neovad | C | 0.15 | 867.5 | 4/32 | 8/8 | None/None | None/None |

## Mandatory topology views (frozen tau)

| model | regime | A->A+B->A KEEP ok/n (false) | A->A+B->B CUT ok/n (missed) | sens hits (KEEP/CUT) |
|---|---|---|---|---|
| firered | O | 0/0 (0) | 2/8 (6) | 0/276 |
| neovad | O | 0/0 (0) | 0/8 (8) | 0/27 |
| firered | C | 0/0 (0) | 2/8 (6) | 0/139 |
| neovad | C | 0/0 (0) | 0/8 (8) | 0/27 |

KEEP breakdown by topology (frozen tau):

| model | regime | topology | KEEP ok/n (false) | sens hits |
|---|---|---|---|---|
| firered | O | A | 7/16 (9) | 536 |
| firered | O | overlap_return | 4/12 (8) | 632 |
| firered | O | A+A+B | 0/4 (4) | 264 |
| neovad | O | A | 14/16 (2) | 98 |
| neovad | O | overlap_return | 10/12 (2) | 60 |
| neovad | O | A+A+B | 4/4 (0) | 30 |
| firered | C | A | 8/16 (8) | 441 |
| firered | C | overlap_return | 6/12 (6) | 363 |
| firered | C | A+A+B | 1/4 (3) | 157 |
| neovad | C | A | 14/16 (2) | 98 |
| neovad | C | overlap_return | 10/12 (2) | 60 |
| neovad | C | A+A+B | 4/4 (0) | 30 |

Diagnostics only: frame AUPRC/F1 (`frame_auprc_diag`, `frame_f1_diag` in `*_calibration.jsonl`), unbound fraction, role-flip agreement.

## Gate 3 verdict (native O)
- firered O: SUPPORTED
- neovad O: COLLAPSED (missed_rate=1.0 at frozen tau)
- Gate 3: at least one formulation supported under native O — proceeded to Gate 4 causal.

## Gate 4 causal gap (O vs C at frozen tau)

See `*_calibration.jsonl` per-tau replay rows and the headline table above; 300 ms sensitivity stream (`sens_hits`) is reversal detection only, no frontier.

## Repaired-evaluator addendum (evaluator_revision=2)

Transition-aware CUT validity (50 ms tolerance); contamination is decoder-dependent (current-segment numerator).

| model | regime | premature-cut CUT episodes |
|---|---|---|
| firered | O | 4 |
| neovad | O | 0 |
| firered | C | 2 |
| neovad | C | 0 |
