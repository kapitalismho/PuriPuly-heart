# PSEM Sortformer adaptation depth

This namespace targets the current issue-107 bounded hobby-engineering probe. The authority is the current body of https://github.com/kapitalismho/PuriPuly-heart/issues/107, not the superseded research-grade body or conflicting historical comments.

## Current execution status

```text
blocked_pending_lean_runner_alignment
```

The authority, config, runtime contract, environment budget, and documentation describe the lean protocol below. Material GPU execution must remain blocked until the Python runner and receipt validators enforce the same 32/256-step, single-seed, single-cell, two-candidate-EVAL, USD-30-hard-stop contract. The existing runner still contains superseded research-grade paths and must not be used for material training yet.

## Local contract verification

```powershell
uv run --project experiments\speaker_representation_scd\environment --frozen python -m experiments.psem_sortformer_adaptation_depth.run preflight --static-only
uv run --project experiments\speaker_representation_scd\environment --frozen pytest experiments\psem_sortformer_adaptation_depth\tests -q
```

Static verification binds the current `contract.json`, `config.json`, `runtime_contract.json`, `runtime_environment.json`, immutable V2 data artifacts, and issue-99 predecessor evidence. Runtime preflight intentionally fails `runtime.material_execution_authorized` while runner alignment is pending.

## Retained invariants

- immutable `PSEM-STRATEGY-DATA-v2` TRAIN/DEV/EVAL assignments;
- exact official `diar_streaming_sortformer_4spk-v2.1.nemo` artifact;
- pinned NeMo revision and immutable NVIDIA container identity;
- raw 16 kHz waveform, four slots, native 80 ms grid, and 1.04 s evidence delay;
- the existing causal GRU-64 PSEM head, composite loss, optimizer groups, learning rates, and 30 s sequence geometry;
- identical TRAIN window and augmentation identities across compared arms;
- EVAL unavailable to fitting and opened once only after DEV selection.

## Lean execution contract

### Arms

```text
F0-FROZEN-FLOAT
H-HEAD
T2-TOP
TA-ALL-TEMPORAL  # conditional only
```

Use seed `7301` only. There is no automatic confirmation seed.

### Cost

```text
target total GPU spend: USD 15
hard stop:               USD 30
```

Record the GPU hourly price and source, actual GPU seconds, accrued cost, and projected remaining cost. A new issue amendment is required before exceeding the hard stop.

### Optional memory-fit estimate

`memory-fit` is an optional, non-authorizing resource estimate. It is not a prerequisite for smoke, training, material authorization, or GPU selection. By default it probes `H-HEAD` and `T2-TOP`; pass `--include-ta` only when a conditional TA estimate is useful.

The command consumes only the first 16 ordered epoch-1 TRAIN rows from the existing sampling manifest and does not run the full manifest validator. Each selected arm runs two optimizer steps on those rows: step 1 initializes optimizer state and warms the graph, and step 2 supplies the rough timing estimate. Peak allocated/reserved VRAM covers both steps. Unit loss weights are sufficient because this command estimates allocation and elapsed time, not scientific quality. No checkpoint or trained model state is persisted.

```bash
python -m experiments.psem_sortformer_adaptation_depth.run memory-fit \
  --checkpoint "$PSEM_SORTFORMER_NEMO_PATH" \
  --nemo-checkout <nemo-checkout> \
  --dependency-lock <cache>/nemo_dependency_lock.json \
  --corpus-root "$PSEM_CORPUS_ROOT" \
  --reference-root "$PSEM_REFERENCE_ROOT" \
  --manifest <cache>/sampling_manifest.jsonl \
  --hourly-price-usd <usd-per-gpu-hour> \
  --hourly-price-source "<provider-price-reference>" \
  --required-inference-gpu-seconds <seconds> \
  --device cuda \
  --output <cache>/optional_resource_estimate.json
```

Add `--include-ta --conditional-ta-inference-gpu-seconds <seconds>` for the conditional TA scenario. The estimate uses no contingency multiplier and enforces no budget threshold. It reports checkpoint I/O outside the command, container startup, retries, and idle provider billing as excluded.

### Smoke and training budget

For `H-HEAD` and `T2-TOP`:

```text
short TRAIN smoke: maximum 32 optimizer steps
official run:      maximum 256 optimizer steps
checkpoint:        final optimizer step
seed:              7301
```

The smoke requires finite forward/backward/update behavior, the exact parameter policy, and a final-eight-step mean loss below the first-eight-step mean loss.

Deterministic resume is not required. An interrupted run is discarded and restarted from step zero with the same seed and manifest. A final model checkpoint is still required for inference.

TA receives the same 32/256-step budget only if DEV shows that T2 clearly justifies deeper adaptation and projected total cost remains below USD 30.

### Evaluation

Use one operating point:

```text
replacement threshold:    0.50
replacement confirmation: 500 ms
```

Required metrics:

- exclusive non-anchor contamination seconds per active-speech hour;
- false/unnecessary cuts;
- missed replacements.

Required views:

- pooled;
- AMI-only;
- AliMeeting-only.

No bootstrap, complete 3x3 frontier, exhaustive topology matrix, or second seed is required. DEV compares F0/H/T2 and selects one direction. EVAL opens once for exactly F0 plus the DEV-selected candidate. EVAL confirms direction only and cannot reopen training or another arm.

## Required artifacts

```text
identity_and_timing_sanity.json
short_smoke_metrics.json
short_training_metrics.json
dev_primary_metrics.json
eval_primary_metrics.json
cost_receipt.json
LEAN_ADAPTATION_DECISION.md
```

## Runner alignment required before material GPU execution

The next implementation step must remove or replace the superseded executable gates:

- eight-epoch sampling/training and DEV early stopping;
- 500-step overfit canary and AP threshold;
- seed 7302 confirmation paths;
- 3x3 frontier, 2,000 bootstrap, equal-corpus/topology acceptance gates;
- freezing every DEV candidate into EVAL;
- final reporting that requires two seeds and all candidates;
- absence of projected/accrued cost enforcement.

Only after those paths and receipts match the lean contract may `material_execution.status` change from `blocked_pending_lean_runner_alignment` to `ready`.

## Claim boundary

The result may guide one internal prototype direction under one seed and a short fixed budget. It is not research-grade evidence, production-readiness evidence, or a target-domain generalization claim. This issue still excludes KD, native causal anchor lifecycle, production VAD execution, acoustic/NEST unfreezing, model-family sweeps, quantization, export, and deployment benchmarking.
