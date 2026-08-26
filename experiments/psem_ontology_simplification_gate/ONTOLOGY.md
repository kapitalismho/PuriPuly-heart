# PSEM ontology simplification contract

This experiment introduces two challengers without modifying `psem-relative-occupancy-v0`.

## Candidate A: `psem-simple-anchor-v0`

The inputs are the GT speech gate and `anchor_present`. `speech && !anchor_present` is the only replacement-evidence state. Silence clears pending replacement evidence, masks pause it, and anchor speech clears it. A speaker cut is disabled outside a valid anchored lifecycle.

## Candidate B: `psem-anchor-overlap-v0`

The inputs are the GT speech gate, `anchor_present`, and anchor-conditioned `anchor_overlap_present`. The frozen proxy is exactly `min(p_anchor, p_nonanchor_max)` over alive non-anchor slots.

The primary model mapping treats `speech=1`, anchor below threshold, and overlap proxy above threshold as instantaneous `ANCHOR_UNCERTAIN`. That interval pauses replacement confirmation and cannot emit a cut. The predeclared strict sensitivity maps the same inconsistency to non-anchor speech. No candidate identity, handover memory, promotion logic, or topology-conditioned persistence exists.

## Perfect-state relationship

For a valid anchor, the shared #97 `OTHER_ONLY` product action is equivalent to `speech && !anchor_present`. Therefore Candidate A and Candidate B must reproduce the #97 Gate 0 replacement actions exactly under perfect GT states. Candidate B's overlap state is retained as an explicit robustness guard for frozen posterior inconsistency, not as a requirement for the perfect-state action oracle.

## Causal interpretation

The #97 causal lifecycle is not ontology-independent. Enrollment requires one active slot while every other alive slot remains below `other_low_threshold`, and speaker-cut lifecycle termination uses the old `OTHER_ONLY` state. S2 therefore replays each challenger's decoder against the already-realized #97 anchor episodes and labels every result `fixed-issue-97-lifecycle-counterfactual-ablation`.

## Timing and masks

All boundary and product metrics use exact 16 kHz source samples. Model evidence frontiers determine emission availability. Masked spans preserve pending evidence without accumulating it. No-speech spans clear replacement evidence. Persistence remains fixed at 100, 200, 300, and 500 ms.
