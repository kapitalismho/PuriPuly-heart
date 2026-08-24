# PSEM training-strategy gate

This experiment implements GitHub issue #76 and compares exactly three official arms:
`FROZEN-WAVLM`, `FINETUNE-WAVLM`, and `SCRATCH-PSEM`.

The accepted `PSEM-STRATEGY-DATA-v2` package under `data/v2/` is the immutable natural-data
prerequisite. It contains 93 meetings, leakage-safe TRAIN/DEV/EVAL roles, the independent
`psem-handoff-v1` label generator contract, and a passing 59-check dataset preflight. The
model experiment does not regenerate, weaken, or reinterpret that freeze.

The model input for every arm is a raw 16 kHz three-second window `[t - 2 s, t + 1 s)`. All
arms produce 30 cells at 100 ms resolution, use the same 256-dimensional common head and
losses, and evaluate `handoff_confirmed` at the unsnapped source position. False events per
source hour is a complete curve axis, never a policy ceiling.

Material work is fail-closed behind:

```powershell
$env:SRSCD_CACHE_ROOT = 'C:\Users\salee\AppData\Local\puripuly-heart-research\speaker_representation_scd_v1'
$env:PSEM_CORPUS_ROOT = 'C:\path\to\the\bound\natural-corpus-root'
$env:PSEM_REFERENCE_ROOT = 'C:\path\to\diar-forced-alignment-at-9527b7c'
uv run --project experiments\speaker_representation_scd\environment --frozen python -m experiments.psem_training_strategy_gate.run preflight
```

From a clean committed candidate, derive the fixed TRAIN-only sampling and augmentation
manifests, run the real-batch model audits, and then write the complete preflight receipt:

```powershell
uv run --project experiments\speaker_representation_scd\environment --frozen python -m experiments.psem_training_strategy_gate.run prepare
uv run --project experiments\speaker_representation_scd\environment --frozen python -m experiments.psem_training_strategy_gate.run audit
uv run --project experiments\speaker_representation_scd\environment --frozen python -m experiments.psem_training_strategy_gate.run preflight
```

`prepare` and `audit` refuse a dirty Git candidate. Preflight independently re-hashes every
referenced manifest and audit artifact, writes a machine-readable receipt even when it rejects
a run, and passes only when all eight runtime receipts match the current contract, data,
label generator, source registry, and Git commit. Training commands must consume a current
passing receipt and never infer readiness from file presence alone.

The final report must retain the frozen-data limitation: the common AMI/AliMeeting temporal
activity references are the commit-pinned forced alignments released by Horiguchi et al.
(ASRU 2025); this project does not independently establish their acoustic boundary accuracy.
