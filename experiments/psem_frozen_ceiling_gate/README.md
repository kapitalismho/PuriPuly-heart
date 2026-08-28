# PSEM FROZEN-CEILING-1

This issue-99 experiment measures the bounded GT causal action frontier and the extractable causal and bounded-noncausal ceilings of the cached frozen Sortformer slot posterior under one oracle mapping per logical anchor episode.

The product target remains Simple Anchor KEEP/CUT/HOLD. Rich slot evidence exists only inside the diagnostic readout. The primary stage reused cached inference. Its scored P-C and P-NC results both remained far from G across both source families, so the authority opened HIDDEN-CEILING-1. No student training, native S2, or fine-tuning is performed.

The predeclared data design uses only old issue-97 DEV sources for probe fitting and contract freeze. Final path-selection scoring uses old issue-97 EVAL sources in two leave-one-corpus-device-family-out folds: AMI Mix-Headset DEV trains the AliMeeting far_ch0 EVAL fold, and AliMeeting far_ch0 DEV trains the AMI Mix-Headset EVAL fold. These are development-known cross-domain results, not a fresh untouched holdout.

Before scoring:

```powershell
uv run python -m experiments.psem_frozen_ceiling_gate.build_ceiling_examples --inventory-only --freeze-mappings
uv run pytest experiments/psem_frozen_ceiling_gate/tests -q
```

After the implementation candidate passes the required pre-experiment review:

```powershell
uv run python -m experiments.psem_frozen_ceiling_gate.gt_action_frontier
uv run python -m experiments.psem_frozen_ceiling_gate.run_posterior_probe
uv run python -m experiments.psem_frozen_ceiling_gate.vad_support_hygiene_replay
uv run python -m experiments.psem_frozen_ceiling_gate.evaluate_ceiling
```

The probe family is fixed to one logistic linear readout and one 8-unit ReLU MLP over the same predeclared temporal taps. Causal taps are current, 1, 3, and 7 frames back. P-NC adds 1, 3, and 5 frames of future frozen posterior evidence, bounded to 500 ms and never labeled causal; GT speech support is not a future-tapped feature. The readout target is live speech without the oracle anchor; its score maps deterministically to Simple Anchor CUT candidates through the fixed threshold and confirmation grid. `action_reference_ledger.jsonl` freezes the issue-98 500 ms fixed-persistence GT episode/event authority for G, S-current, S-probe, P-C, and P-NC; its DEV events exactly match the sealed issue-97 Gate-0 ledger, its DEV/EVAL counts match issue-98, and all S/P arms also share `oracle_mapping_ledger.jsonl`.

The production-VAD hygiene replay reuses the recorded issue-98 VAD spans. It marks the first 500 ms of each recorded support span as pre-roll-only context and excludes that support from replacement confirmation. It performs no speaker-model inference and does not tune VAD.

HIDDEN-CEILING-1 fixes exactly one representation before hidden scoring: the 192-dimensional output of the eighteenth post-LN temporal Transformer block immediately before `diar.spk_head`. `sortformer_hidden_export.patch` adds readback only to the authoritative Q8_0/Vulkan/low-latency runtime. `extract_hidden_features.py` runs the exact padded inputs from the authoritative receipts, requires every instrumented posterior to match its cached trace within an absolute tolerance of 1e-6, and stores the large float32 features outside Git. Hidden causal and bounded-noncausal probes reuse the frozen split, oracle episode mapping, temporal taps, tiny probe classes, target, and product evaluator.

The hidden tiny MLP keeps the predeclared 8-unit architecture and uses the training-only Adam schedule in `hidden_training_config.json`. It stops only when the already frozen train-fit AP and accuracy gates are both reached at a predeclared check interval, or at the fixed maximum epoch. This schedule does not inspect held-out metrics or alter the linear probe, hidden representation, features, target, split, score thresholds, or product evaluator.

Hidden failure attribution uses the frozen neural/acoustic slice checks and the already reported per-source-family improved-metric counts. A result is source-family/domain-localized when at least one frozen family reaches the predeclared family improvement minimum and at least one does not; no new metric threshold is introduced.

After the hidden implementation candidate passes its own pre-experiment review:

```powershell
uv run python -m experiments.psem_frozen_ceiling_gate.extract_hidden_features
uv run python -m experiments.psem_frozen_ceiling_gate.run_hidden_probe
uv run python -m experiments.psem_frozen_ceiling_gate.evaluate_ceiling
```
