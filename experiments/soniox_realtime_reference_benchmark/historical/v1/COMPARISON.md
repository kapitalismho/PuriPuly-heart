# Soniox reference vs prior PSEM proxy rows — COMPARISON

Prior rows are proxy-vs-actual-provider evidence on frozen PSEM captures,
NOT the same ASR as this benchmark. No head-to-head ranking of rules is
valid here. Rows preserved separately with context, denominator, and
human GT version. No claim of Soniox theoretical upper bound: only the
best observed RT profile within this sample. Async not run;
cost/storage not needed first.

## Prior immutable reference rows (read-only sources)

Return ledger `experiments/psem_return_capacity/ledger.json`
(`d01f07ac…32691`): R2 current all arms error interval [3,5]; R2 oracle
counterfactual [4,7] with 4 definite + 3 straddle; R2 scalar
counterfactual [3,5]. T1 rearm/scalar/oracle/full4 [6,7] (6 definite +
1 straddle); T1 none/orig [11,11]. COMBINED current [12,14] (12 + 2);
COMBINED scalar counterfactual [7,12] (7 + 5); COMBINED oracle
counterfactual [9,13] (9 + 4). GT: AMI words XML all 4 roles, same
annotation archive as stage2 freeze.

P3T ledger `experiments/psem_p3t_rearm/per_case_ledger.json`
(`65b3cff0…527a`): NP1 ownership none/orig wrong 4 → rearm [0,0]
(4 right correct, zero new harm).

## New Soniox rows (this benchmark, human GT denominators)

NP cohort (frozen 8-word sets, pad 32000, same helper):
NP1-MANUAL 8/8, NP2-MANUAL 5/8, NP3-MANUAL 8/8;
NP1-NATURAL 3/8, NP2-NATURAL 2/8, NP3-NATURAL 4/8.
Whole-payload end-to-end (non-punct GT in payload, separate view):
MANUAL 25/29, 13/23, 20/22; NATURAL 9/29, 7/23, 13/22.
EN2009d scopes (non-punct GT, exact prior span IDs):
MANUAL COMBINED 19/28, R2 6/9, T1 13/21;
NATURAL COMBINED 9/28, R2 6/9, T1 3/21.
Speaker: retrospective per-session Hungarian on matched words only
(MANUAL 52/52 mapped-correct across all scopes; NATURAL partly
single-side). Overlap class explicit (R2 9/9, T1 17/4, COMBINED 24/4).

## What the comparison does and does not say

- DOES say: observed Soniox MANUAL recognition on the frozen NP cohorts
  (21/24) and EN2009d scopes sits in the same coarse band as the proxy
  receiver rows above, with full per-word evidence in `ledger.json`.
  Proxy rows measure PSEM ownership assignment on frozen text; Soniox
  rows measure live ASR recognition + server diarization labels. Same
  GT, different systems under test.
- DOES NOT say: which PSEM rule is better; whether Soniox bounds PSEM;
  any DER figure; any cross-system accuracy ranking. NP cuts stress
  isolated context from a different PSEM source-zero, so no
  apples-to-apples claim against full-context behavior either way.
- Human GT version identical (AMI manual words XML via existing
  helpers); no Soniox-as-GT labels were created or used for tuning, and
  PSEM is unchanged.
