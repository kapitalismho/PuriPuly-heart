# MAIN48 native-ceiling (O) + causal (C) scoring — Gates 3+4

> V2 EVAL sessions reused as dev-only probe per program approval; no unbiased generalization claim; V3 fresh holdout required for selection claims.

> Frozen Gate 2 taus applied as-is (firered/neovad x O/C); no retuning on MAIN48.

| model | regime | tau | contam s/h | false cuts (KEEP-n) | missed (CUT-n) | src_err p50/p90 (ms) | dec p50/p90 (ms) |
|---|---|---|---|---|---|---|---|
| firered | O | 0.05 | 877.7 | 12/32 | 5/8 | -820.0/1068.0 | 500.0/500.0 |
| neovad | O | 0.05 | 877.7 | 0/32 | 8/8 | None/None | None/None |
| firered | C | 0.05 | 877.7 | 13/32 | 5/8 | -670.0/1098.0 | 500.0/500.0 |
| neovad | C | 0.05 | 877.7 | 0/32 | 8/8 | None/None | None/None |

## Mandatory topology views (frozen tau)

| model | regime | A->A+B->A KEEP ok/n (false) | A->A+B->B CUT ok/n (missed) | sens hits (KEEP/CUT) |
|---|---|---|---|---|
| firered | O | 0/0 (0) | 3/8 (5) | 0/109 |
| neovad | O | 0/0 (0) | 0/8 (8) | 0/0 |
| firered | C | 0/0 (0) | 3/8 (5) | 0/88 |
| neovad | C | 0/0 (0) | 0/8 (8) | 0/0 |

KEEP breakdown by topology (frozen tau):

| model | regime | topology | KEEP ok/n (false) | sens hits |
|---|---|---|---|---|
| firered | O | A | 11/16 (5) | 221 |
| firered | O | overlap_return | 6/12 (6) | 346 |
| firered | O | A+A+B | 3/4 (1) | 110 |
| neovad | O | A | 16/16 (0) | 0 |
| neovad | O | overlap_return | 12/12 (0) | 0 |
| neovad | O | A+A+B | 4/4 (0) | 0 |
| firered | C | A | 10/16 (6) | 257 |
| firered | C | overlap_return | 6/12 (6) | 340 |
| firered | C | A+A+B | 3/4 (1) | 88 |
| neovad | C | A | 16/16 (0) | 0 |
| neovad | C | overlap_return | 12/12 (0) | 0 |
| neovad | C | A+A+B | 4/4 (0) | 0 |

Diagnostics only: frame AUPRC/F1 (`frame_auprc_diag`, `frame_f1_diag` in `*_calibration.jsonl`), unbound fraction (0.0 — all 48 MAIN48 rows causal-bindable), role-flip agreement (sens-vs-primary divergence on return topologies; see sens hits above).

NOTE: MAIN48 carries zero `A->A+B->A` rows, so the mandatory KEEP cell reads 0/0 for both models x regimes; the KEEP verdict rests on A / overlap_return / A+A+B (n=32).

## Gate 3 verdict (native O)
- firered O: SUPPORTED
- neovad O: COLLAPSED (missed_rate=1.0 at frozen tau)
- Gate 3: at least one formulation supported under native O — proceeded to Gate 4 causal.

## Gate 4 causal gap (O vs C at frozen tau)

- firered O->C is flat: false cuts 12->13/32, missed 5/8 both, CUT success 3/8 both, src_err p50 -820->-670ms / p90 1068->1098ms, decision delay pinned at the 500ms confirmation quantum both regimes. The 1s causal bind costs nothing measurable vs the 5s native bind.
- neovad is collapsed in BOTH regimes (0/8 CUT, 0 false cuts, 0 sens hits): the formulation is unsupported regardless of bind span, so there is no O-C gap to read.
- 300ms sensitivity stream (reversal detection only, no frontier): firered CUT sens 109->88 O->C; KEEP sens concentrates on overlap_return (346/340) where the primary still false-cuts 6/12 — the fast stream sees the return, the 500ms confirmer does not reject it. No threshold or frontier action taken.

