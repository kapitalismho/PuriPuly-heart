# Soniox reference vs prior PSEM proxy rows v2 — COMPARISON

Parallel reference against the same human GT only. The two pipelines
measure different systems under test and different metrics, so this
comparison ranks no rules and states no band. v1 coarse-band language
is withdrawn.

## Prior immutable proxy rows (read-only, joint receiver pipeline)

Return ledger `experiments/psem_return_capacity/ledger.json`
(`d01f07ac…32691`): R2 current all arms [3,5]; R2 oracle counterfactual
[4,7] (4 definite + 3 straddle); R2 scalar counterfactual [3,5]. T1
rearm/scalar/oracle/full4 [6,7]; T1 none/orig [11,11]. COMBINED current
[12,14]; scalar counterfactual [7,12]; oracle counterfactual [9,13].
These count ownership-assignment errors of PSEM text partitions on
frozen captures under the receiver contract, with GT-overlap reported
as a uniform diagnostic.

P3T ledger `experiments/psem_p3t_rearm/per_case_ledger.json`
(`65b3cff0…527a`): NP1 ownership none/orig wrong 4 → rearm [0,0].

GT version both sides: AMI manual words XML all 4 roles via existing
helpers; no Soniox-as-GT anywhere; PSEM unchanged.

## New live reference rows (this benchmark, joint ASR pipeline)

NP full-payload strict recall / WER: NP1-MAN 0.86 / 0.1379, NP2-MAN
0.57 / 0.4783, NP3-MAN 0.91 / 0.0909; NATURAL 0.31 / 0.6897, 0.30 /
0.6957, 0.59 / 0.4091. Frozen 8-word cohorts: MAN 7/8, 4/8, 4/8; NAT
2/8, 1/8, 1/8. EN2009d scopes recall / WER: MAN COMBINED 19/28 /
0.4286, R2 6/9 / 0.4444, T1 13/21 / 0.4762; NAT 9/28 / 0.7143, 6/9 /
0.4444, 3/21 / 0.8571. Speaker maps retrospective per-session on
matched words only. All sessions INCOMPLETE on the finished protocol
with complete token evidence (see RESULTS).

## Why no ranking follows

Proxy rows score PSEM ownership decisions given frozen text; reference
rows score live ASR recognition plus server diarization labels given
audio. Denominators overlap (same GT words) but numerators count
different events (mis-assigned vs mis-recognized/missed words), and the
error-interval semantics (definite + straddle) have no counterpart in
recall/WER. NP cuts additionally stress isolated context from a
different source-zero than full-context behavior. The honest joint
statement is only: both pipelines are measured against the same human
GT, both records are preserved with per-word evidence, and neither
bounds the other. Best observed complete profile is not an upper bound;
async not run.
