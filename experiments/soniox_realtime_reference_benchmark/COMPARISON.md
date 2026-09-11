# Soniox reference vs prior PSEM proxy rows v3 — COMPARISON (active, corrected)

Parallel reference against the same human GT only. Different systems,
different metrics: ranks no rules, states no band. v1 coarse-band
language stays withdrawn.

## Prior immutable proxy rows (read-only, joint receiver pipeline)

Return ledger `experiments/psem_return_capacity/ledger.json`
(`d01f07ac…32691`): R2 current all arms [3,5]; R2 oracle counterfactual
[4,7] (4 definite + 3 straddle); R2 scalar counterfactual [3,5]. T1
rearm/scalar/oracle/full4 [6,7]; T1 none/orig [11,11]. COMBINED current
[12,14]; scalar counterfactual [7,12]; oracle counterfactual [9,13].
Ownership-assignment errors of PSEM text partitions on frozen captures.

P3T ledger `experiments/psem_p3t_rearm/per_case_ledger.json`
(`65b3cff0…527a`): NP1 ownership none/orig wrong 4 → rearm [0,0].

GT version both sides: AMI manual words XML all 4 roles via existing
helpers; no Soniox-as-GT; PSEM unchanged. No identical hyp-region rule
is required where it would exclude required words; GT counts and the
frozen 8 NP IDs are unchanged by the whole-payload hyp correction.

## New live reference rows v3 (joint ASR pipeline, protocol complete)

NP full-payload recall / Levenshtein WER: NP1 0.83–0.86 / 0.14–0.17,
NP2 0.57–0.65 / 0.35–0.48, NP3 0.91 / 0.09 both profiles. Frozen
cohorts (whole-payload hyp, fixed IDs): 8/8, 5/8, 8/8. EN2009d recall /
WER: COMBINED 19/28 / 0.3929, R2 6/9 / 0.4444, T1 13/21 / 0.4286,
identical both profiles. Pre-EOS-final-only attribution reported
separately in RESULTS (NATURAL online vs EOS-assisted gap). Speaker
maps retrospective per-session on matched words only (NP2
minority-role errors counted wrong in FULL A-D; NP1/NP3 pure IDs good).
All sessions finished:true with complete EOS tails; per-word rows in
`ledger.json`.

## Why no ranking follows

Proxy rows score PSEM ownership decisions given frozen text; reference
rows score live ASR recognition plus server diarization labels given
audio. Same GT words, different counted events (mis-assigned vs
mis-recognized/missed), and error-interval semantics have no recall/WER
counterpart. NP cuts stress isolated context from a different
source-zero than full-context behavior. Joint statement only: same
human GT, both records preserved with per-word evidence, neither
bounds the other. Best observed complete profile is not an upper
bound; async not run. Only v3 rows are active; v1/v2 rows live in
`historical/` as withdrawn method records, not results.
