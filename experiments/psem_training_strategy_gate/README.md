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

Official fitting uses the frozen manifest order with an effective batch size of four. Every
manifest batch has nonzero handoff, state, and relation supervision. Checkpoint selection uses
DEV event average precision at the middle required matching collar, ±250 ms, with DEV total
loss as the tie-break. EVAL remains sealed throughout fitting.

Run the exact matrix in the required order, or use the guarded all-runs command:

```powershell
uv run --project experiments\speaker_representation_scd\environment --frozen python -m experiments.psem_training_strategy_gate.run train --arm FROZEN-WAVLM --seed 7301
uv run --project experiments\speaker_representation_scd\environment --frozen python -m experiments.psem_training_strategy_gate.run train --arm FROZEN-WAVLM --seed 7302
uv run --project experiments\speaker_representation_scd\environment --frozen python -m experiments.psem_training_strategy_gate.run train --arm FINETUNE-WAVLM --seed 7301
uv run --project experiments\speaker_representation_scd\environment --frozen python -m experiments.psem_training_strategy_gate.run train --arm FINETUNE-WAVLM --seed 7302
uv run --project experiments\speaker_representation_scd\environment --frozen python -m experiments.psem_training_strategy_gate.run train --arm SCRATCH-PSEM --seed 7301
uv run --project experiments\speaker_representation_scd\environment --frozen python -m experiments.psem_training_strategy_gate.run train --arm SCRATCH-PSEM --seed 7302
```

```powershell
uv run --project experiments\speaker_representation_scd\environment --frozen python -m experiments.psem_training_strategy_gate.run train-all
```

Each invocation revalidates the persisted passing preflight before writing. The first run freezes
the exact device, software, host, and thread environment for the complete matrix. Deterministic
two-slot progress and best-model checkpoints preserve a valid resume boundary across interrupted
writes. Every completed prefix is reverified before the next run, and commands refuse to skip an
arm or seed. Re-run the same command after interruption to resume. `training-status` is read-only
and reports the sealed EVAL state and exact six-run progress.

The final report must retain the frozen-data limitation: the common AMI/AliMeeting temporal
activity references are the commit-pinned forced alignments released by Horiguchi et al.
(ASRU 2025); this project does not independently establish their acoustic boundary accuracy.
