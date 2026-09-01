# PSEM Sortformer adaptation depth

This namespace targets the current issue-107 bounded hobby-engineering probe. The authority is the current body of https://github.com/kapitalismho/PuriPuly-heart/issues/107, not the superseded research-grade body or conflicting historical comments.

## Current execution status

```text
ready
```

The supported CLI now enforces the 32/256-step, seed-7301, singleton DEV/EVAL, F0-plus-winner, and USD-30-hard-stop contract. `ready` authorizes the local material path only after runtime preflight, short smoke, runtime canaries, staged DEV state, and cost validation all pass on a clean committed candidate; it does not authorize GPU provisioning, deployment, or remote execution.

## Local contract verification

```powershell
uv run --project experiments\speaker_representation_scd\environment --frozen python -m experiments.psem_sortformer_adaptation_depth.run preflight --static-only
uv run --project experiments\speaker_representation_scd\environment --frozen pytest experiments\psem_sortformer_adaptation_depth\tests -q
```

Static verification binds the current `contract.json`, `config.json`, `runtime_contract.json`, `runtime_environment.json`, immutable V2 data artifacts, and issue-99 predecessor evidence. Runtime preflight authorizes material work only when those controls, a clean committed candidate, the pinned runtime, and all external paths are present and exact.

## Compatible single-CUDA-GPU runtime preparation

The local derived-image build and validation procedure for a compatible single-CUDA-GPU runtime is documented in [`environment/README.md`](environment/README.md). An A40 remains the preferred deployment target. Detached phase execution, durable heartbeats, operator decision gates, and the external Windows stop watchdog are documented in [`DETACHED_EXECUTION.md`](DETACHED_EXECUTION.md). Neither procedure authorizes registry publication, Pod creation, training, or deployment.

## Retained invariants

- immutable `PSEM-STRATEGY-DATA-v2` TRAIN/DEV/EVAL assignments;
- exact official `diar_streaming_sortformer_4spk-v2.1.nemo` artifact;
- pinned NeMo revision, immutable derived runtime identity, and pinned NVIDIA base provenance;
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

The smoke requires finite forward/backward/update behavior and the exact parameter policy. Its first-eight and final-eight mean losses are recorded as diagnostics rather than used as a blocking trend gate.

Deterministic resume is not required. An interrupted run is discarded and restarted from step zero with the same seed and manifest. A final model checkpoint is still required for inference.

TA receives the same 32/256-step budget only after the trusted operator records an explicit `open_ta` decision and the current cost receipt keeps projected total spend at or below USD 30.

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

The supported operating surface is one fixed 0.50/500 ms cell, one seed, and pooled/AMI/AliMeeting views. Bootstrap intervals, a 3x3 frontier, an exhaustive topology matrix, and a second seed are outside this engineering gate. After F0, H, and T2 DEV evidence is available, the trusted operator records `select_candidate`, `open_ta`, or `stop`. EVAL opens once for exactly F0 plus the selected candidate and cannot reopen training or another arm.

## Required artifacts

```text
identity_and_timing_sanity.json
short_smoke_metrics.json
short_training_metrics.json
dev_primary_metrics.json
eval_primary_metrics.json (selection path only)
cost_receipt.json
ADAPTATION_DECISION.md (all terminal outcomes)
```

## Sequential operator flow

1. Run static and runtime preflight on a clean committed candidate, then materialize and validate the one-epoch sampling manifest and TRAIN-only class weights.
2. Build lineage evidence, run the one-time Pod CUDA canary, then run the arm's 32-step `smoke-arm` and create a `cost-receipt` for the next material action.
3. Initialize staged DEV state from F0. For H and then T2, assemble and validate the material bundle, run the exact 256-step `train-arm`, infer DEV predictions, evaluate the singleton cell, and append the result.
4. Record `dev-decision`. If it is `open_ta`, create `open-ta` authorization and repeat the smoke, cost, training, and DEV sequence for TA before recording the final selection or stop decision.
5. Freeze the candidate set with the final operator decision and cost receipt. `stop` ends with an empty freeze and cannot open EVAL; pass that freeze directly to `final-report` to emit the Outcome-D `ADAPTATION_DECISION.md` without EVAL. Selection freezes exactly F0 plus one winner.
6. For selection only, open EVAL once, infer and evaluate exactly the frozen pair, then pass the EVAL report bundle to `final-report`. No supported command resumes training after EVAL opens.

The canonical decision for every outcome explicitly preserves both boundaries: no student KD is performed or authorized, and no NEST/acoustic-encoder parameter is unfrozen or authorized for unfreezing.

The old eight-epoch, 500-step overfit, seed-7302, automatic Pareto/bootstrap escalation, and all-candidate EVAL paths are not part of the supported material CLI.

## Claim boundary

The result may guide one internal prototype direction under one seed and a short fixed budget. It is not research-grade evidence, production-readiness evidence, or a target-domain generalization claim. This issue still excludes KD, native causal anchor lifecycle, production VAD execution, acoustic/NEST unfreezing, model-family sweeps, quantization, export, and deployment benchmarking.
