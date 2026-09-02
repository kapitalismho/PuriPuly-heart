# Lean adaptation decision

Run: `issue-107-a40-1334720a-01`

Decision: `STOP — no supported adaptation direction`

## Required questions

### 1. GPU and cost

The mandatory arms ran on a single RunPod A40 at a recorded hourly price of USD 0.44.

The phase receipts record USD 5.2035 for H and USD 0.9548 for T2. These values must not be presented as the complete-probe cost: the T2 receipt records 7811.97 GPU seconds while its training receipt records 11994.27 training wall-clock seconds, and environment bring-up/retry time is not coherently aggregated. The complete cost requirement remains unresolved.

### 2. Did H improve over F0?

No. H is worse than F0 on all three pooled primary DEV metrics:

- contamination: 2154.20 versus 1902.41
- false cuts/hour: 75.01 versus 45.85
- missed replacements/hour: 252.41 versus 218.80

### 3. Did T2 justify its added complexity?

No. T2 is better than H on all three pooled metrics, but only by roughly 4-5%. It does not satisfy the Issue 107 requirement that contamination or missed replacements improve by at least 10% relative to H. It also remains worse than F0 on all three pooled metrics.

### 4. Was TA opened?

No. The DEV rule does not authorize TA because T2 does not clearly beat H and neither adapted arm beats F0.

### 5. Did a DEV direction survive EVAL?

Not applicable. No adapted DEV candidate was supported, so EVAL was not opened.

### 6. Next step

Stop the authorized adaptation sequence. Independent post-hoc review may inspect the committed artifacts, but additional training requires a separate amendment and should not be inferred from this result bundle.

## Result interpretation

The negative result is real under the frozen short-budget recipe. H and T2 produced nearly as many total cuts as F0, but placed them less accurately: correct matches fell while false cuts and missed replacements increased. T2 partially recovered H's regression without reaching F0.

AMI was worse for both adapted arms across all three primary metrics. On AliMeeting, H and T2 reduced false cuts but increased contamination and missed replacements, indicating a corpus-dependent tradeoff rather than a general gain.

## Execution and validity limitations

- H was produced under `microbatch=1, accumulation=16`, while T2 used `microbatch=2, accumulation=8`. Both have effective batch 16, but code identity and microbatch-level randomization scope differ. The small T2-over-H delta should not be attributed solely to adaptation depth.
- The declared reset policy is `declared_source_or_reset_boundary_only`, but all 4096 TRAIN crops declare a reset at their window start. DEV inference resets once at the source start. This train/inference recurrent-state mismatch is a plausible confound.
- F0 uses `1 - selected anchor posterior`, whereas H/T2 use a newly trained PSEM replacement logit. A fixed threshold of 0.5 compares different score constructions.
- Replacement BCE uses a TRAIN-derived positive weight of about 9.91 while evaluation applies the raw sigmoid at threshold 0.5. The resulting output is not necessarily calibrated to the natural DEV event prior.
- The objective is frame-level while the primary decision is a confirmed event after 500 ms. Falling total loss does not guarantee better event placement.
- Only one seed and one short 256-step run were authorized. No seed stability or generalization claim is supported.
- The cost receipts do not reconstruct total probe spend consistently.

## Informal post-hoc suggestions

Treat these only as lightweight suggestions for future discussion, not as Issue 107 conclusions or authorization to rerun:

- align TRAIN recurrent-state handling with continuous inference, using valid carried state or an explicit burn-in treatment instead of declaring every random crop a true reset;
- initialize an adapted decision head as a residual correction to the F0 score so the untrained starting point preserves F0 behavior;
- either remove class-prior distortion or explicitly calibrate logits before applying the operating threshold;
- add an event-aligned objective or diagnostic that distinguishes overlap continuation/return from genuine takeover;
- if a new issue authorizes another comparison, run H and T2 under one identical runtime/code identity and repair total-cost accounting first.
