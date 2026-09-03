# CAL12 threshold calibration (Gate 2 scaffold)

> V2 EVAL sessions reused as dev-only probe per program approval; no unbiased generalization claim; V3 fresh holdout required for selection claims.

| model | regime | tau | false_cuts | missed | src_err p50/p90 (ms) | dec p50/p90 (ms) | contam s/h | stub | dry |
|---|---|---|---|---|---|---|---|---|---|
| firered | O | 0.15 | 5/8 | 1/2 | 60.0/60.0 | 500.0/500.0 | 475.08735705209654 | False | False |
| firered | C | 0.05 | 4/8 | 1/2 | 110.0/110.0 | 500.0/500.0 | 534.4107369758576 | False | False |
| neovad | O | 0.15 | 0/8 | 1/2 | 2200.0/2200.0 | 500.0/500.0 | 674.4996823379925 | False | False |
| neovad | C | 0.15 | 0/8 | 1/2 | 2200.0/2200.0 | 500.0/500.0 | 674.4996823379925 | False | False |

Topology views: `A->A+B->A` KEEP vs `A->A+B->B` CUT reported separately in per-tau calibration rows (`*_calibration.jsonl`: `n_keep`, `n_cut`, `false_cuts`, `missed`).
Frame AUPRC/F1 (`frame_*_diag`), unbound fraction, and role-flip agreement are diagnostics only.
