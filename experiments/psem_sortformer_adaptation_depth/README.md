# PSEM Sortformer adaptation depth

This namespace implements the issue-107 adaptation-depth experiment without changing the product runtime. It binds the immutable V2 data freeze, issue-99 Simple Anchor evidence, the official float checkpoint, one NVIDIA NeMo revision, and the complete execution protocol before any material metric is inspected.

## Local verification

The read-only local gate needs no model, corpus, GPU, or external service:

```powershell
uv run --project experiments\speaker_representation_scd\environment --frozen python -m experiments.psem_sortformer_adaptation_depth.run preflight --static-only
uv run --project experiments\speaker_representation_scd\environment --frozen pytest experiments\psem_sortformer_adaptation_depth\tests -q
```

The static gate binds `contract.json`, `config.json`, `runtime_contract.json`, `runtime_environment.json`, the V2 data artifacts, and the issue-99 decision. A dirty worktree fails the material gate.

## Engineering validation boundary

Pre-execution acceptance assumes one trusted operator runs the committed CLI sequentially in one clone. The local gate proves ordinary CLI ordering, identity, split, fixed-grid, reporting, and fail-closed behavior; it does not claim protection from fabricated in-package receipts or concurrent protocol commands.

The resulting metrics support an engineering choice about the next experiment only. They are not publication-grade research evidence, production-readiness evidence, or a fresh generalization claim. Missing or non-finite fixed-grid decision evidence is retained as unsupported and forces Outcome D instead of being silently filtered or substituted.

## Authorized runtime

The frozen runtime is Linux x86_64 in `nvcr.io/nvidia/pytorch@sha256:0981807f1a51a156563e28b59dc2e7a9b5c1c7d85d1169d4965c5fd91fa38bcb` (tag `25.01-py3`), Python 3.10 or newer, PyTorch 2.5 or newer, and exactly one CUDA accelerator with at least 80,000,000,000 bytes of device memory. NeMo must be checked out at `1a3c291b3ef0f0e11b72f789b185e1f1bda39bd6` and installed from that checkout with the commands frozen in `runtime_environment.json`.

Set an immutable local image identity, not a mutable tag:

```bash
export PSEM_CONTAINER_IMAGE_IDENTITY=sha256:0981807f1a51a156563e28b59dc2e7a9b5c1c7d85d1169d4965c5fd91fa38bcb
```

The dependency lock records the complete installed distribution inventory, OS identity, immutable container identity, CUDA version, NVIDIA driver, GPU model, and GPU memory. Validation fails if any of them changes.

Runtime preflight also requires:

```text
PSEM_SORTFORMER_NEMO_PATH       exact 471367680-byte .nemo artifact
PSEM_CORPUS_ROOT                all 93 waveforms at their frozen paths and hashes
PSEM_REFERENCE_ROOT             diar-forced-alignment at the frozen commit
PSEM_ADAPTATION_OUTPUT_ROOT     existing absolute cache outside the repository
PSEM_PROTOCOL_REGISTRY_ROOT     existing absolute external directory, distinct from the output root
PSEM_CONTAINER_IMAGE_IDENTITY   immutable sha256 image identity
```

`PSEM_ALLOW_EVAL` must remain absent through lineage, canaries, training, DEV inference, staging, and candidate freeze.

RunPod provisioning, uploads, downloads requiring credentials, corpus materialization, and remote execution require owner approval before use. The commands below describe the approved protocol but do not grant that approval.

## Frozen scientific contract

All arms consume raw 16 kHz mono waveform through the trainable `.nemo` graph at the issue-99 `6/7/188/144/188` chunk, right-context, FIFO, cache-update, and cache-length settings. The native output grid is 80 ms and the charged evidence delay is 1.04 s. Saved rows remain on that complete native grid. The unchanged issue-99 action evaluator samples, for each committed action interval, the latest native frame whose end is not after the action end; it then recomputes the one-slot-per-episode mapping from those action-aligned checkpoint posteriors. A stale serialized slot is fatal.

The shared head is one LayerNorm, one causal GRU with hidden size 64, and two linear logits. The sampling manifest has 4,096 contiguous 30-second TRAIN windows per epoch for eight epochs with a fixed 50/25/25 source-time, replacement-positive, and hard-negative mixture. Training uses micro-batch 1, gradient accumulation 16, 256 optimizer steps per epoch, and the fixed issue-107 optimizer and loss recipe.

DEV checkpoint loss uses every complete source-aligned non-overlapping 30-second sequence. The incomplete tail is excluded from DEV loss but remains in full-source product evaluation. EVAL has no fitting, stopping, escalation, threshold, or configuration path.

## Execution order

Use a clean committed candidate and write all large artifacts under the external experiment cache.

1. Produce runtime prerequisites and immutable receipts.

   ```bash
   python -m experiments.psem_sortformer_adaptation_depth.run preflight --checkpoint "$PSEM_SORTFORMER_NEMO_PATH" --corpus-root "$PSEM_CORPUS_ROOT" --reference-root "$PSEM_REFERENCE_ROOT" --output-root "$PSEM_ADAPTATION_OUTPUT_ROOT" --protocol-registry-root "$PSEM_PROTOCOL_REGISTRY_ROOT" --receipt-output <cache>/preflight.json
   python -m experiments.psem_sortformer_adaptation_depth.run dependency-lock --output <cache>/nemo_dependency_lock.json
   python -m experiments.psem_sortformer_adaptation_depth.run data-split-receipt --output <cache>/data_split_receipt.json
   python -m experiments.psem_sortformer_adaptation_depth.run evaluator-contract --output <cache>/evaluator_contract.json
   python -m experiments.psem_sortformer_adaptation_depth.run model-graph --checkpoint "$PSEM_SORTFORMER_NEMO_PATH" --nemo-checkout <nemo-checkout> --dependency-lock <cache>/nemo_dependency_lock.json --device cuda --output <cache>/model_graph.json
   ```

2. Build the pre-training float/Q8 lineage. `lineage-authorization` permits only the official frozen float checkpoint on the exact issue-99 DEV/EVAL identities; it forbids fitting, checkpoint selection, adapted weights, and EVAL-driven development.

   ```bash
   python -m experiments.psem_sortformer_adaptation_depth.run lineage-authorization --output <cache>/lineage_authorization.json
   python -m experiments.psem_sortformer_adaptation_depth.run build-lineage --checkpoint "$PSEM_SORTFORMER_NEMO_PATH" --nemo-checkout <nemo-checkout> --dependency-lock <cache>/nemo_dependency_lock.json --corpus-root "$PSEM_CORPUS_ROOT" --reference-root "$PSEM_REFERENCE_ROOT" --output-root "$PSEM_ADAPTATION_OUTPUT_ROOT" --authorization <cache>/lineage_authorization.json --device cuda --output <cache>/trainable_checkpoint_lineage.json --runtime-identity-output <cache>/runtime_identity.json
   python -m experiments.psem_sortformer_adaptation_depth.run validate-lineage <cache>/trainable_checkpoint_lineage.json --runtime-identity <cache>/runtime_identity.json --output <cache>/lineage_validation.json
   ```

3. Materialize and validate the one shared TRAIN manifest and TRAIN-only class weights.

   ```bash
   python -m experiments.psem_sortformer_adaptation_depth.run sampling-manifest --corpus-root "$PSEM_CORPUS_ROOT" --reference-root "$PSEM_REFERENCE_ROOT" --output <cache>/sampling_manifest.jsonl
   python -m experiments.psem_sortformer_adaptation_depth.run validate-sampling-manifest --manifest <cache>/sampling_manifest.jsonl --corpus-root "$PSEM_CORPUS_ROOT" --reference-root "$PSEM_REFERENCE_ROOT" --output <cache>/sampling_validation.json
   python -m experiments.psem_sortformer_adaptation_depth.run class-weights --manifest <cache>/sampling_manifest.jsonl --corpus-root "$PSEM_CORPUS_ROOT" --reference-root "$PSEM_REFERENCE_ROOT" --output <cache>/class_weights.json
   ```

4. For `H-HEAD`, `T2-TOP`, and conditionally `TA-ALL-TEMPORAL`, run the raw-waveform graph/gradient/update/timing canary with `canary-arm`, then the fixed 30-minute/500-step overfit canary with `overfit-arm`. The timing canary mutates a future waveform suffix and proves that every output whose charged frontier precedes that suffix is unchanged; it also binds the observed cache/FIFO trace. `TA-ALL-TEMPORAL` additionally requires `--staged-state` and every prior `--staged-dev-result`, and cannot run before the frozen DEV escalation opens it. Assemble and validate the combined overfit receipt with `build-overfit-receipt` and `validate-overfit`. A failure is an implementation or objective failure and does not authorize a recipe change.

5. Build one `material_training_authorization` per arm and authorized seed. `assemble-material-bundle` converts paths into the exact bundle consumed by `validate-material-gate`. The gate revalidates the clean Git head, split, sampling bytes, TRAIN weights, lineage, runtime identity, evaluator, parameter inventory, canaries, overfit result, and current DEV-only staged state.

6. Execute the fixed staged protocol:

   - infer and evaluate DEV for `F0-FROZEN-FLOAT`;
   - initialize state with `stage-init`;
   - train `H-HEAD` seed 7301 with `train-arm`, then `infer`, `evaluate`, and `stage-append`;
   - train `T2-TOP` seed 7301 in the same way;
   - run `TA-ALL-TEMPORAL` only when the staged receipt opens it;
   - run seed 7302 only for arms authorized by the staged receipt.

   Training selects only by complete DEV loss with DEV replacement AP as the frozen tie-break. Each saved checkpoint and prediction row binds its source waveform, base and trained checkpoint, runtime, parameter policy, Git candidate, and code identity. Checkpoint, dependency-lock, prediction, and lineage validation hashes and consumes the same captured bytes. Runtime lineage, canaries, checkpoints, predictions, evaluations, and training results also require content-addressed records in the authority registry.

7. Freeze all DEV results, prediction sets, and selected checkpoint receipts with `freeze-candidates`. This command requires the current clean candidate to match the code used for those artifacts.

8. Open EVAL exactly once:

   ```bash
   python -m experiments.psem_sortformer_adaptation_depth.run open-eval <cache>/candidate_freeze.json --output-root "$PSEM_ADAPTATION_OUTPUT_ROOT"
   ```

   The only authorization is created with exclusive-file semantics under the clone-wide Git common directory at `psem-sortformer-adaptation-depth/<authority-pin>/issue-107-<authority-pin>.json`. This location is shared by every process and worktree in the clone, so changing the caller-provided registry or output directory cannot reopen EVAL. `$PSEM_PROTOCOL_REGISTRY_ROOT` remains an external provenance root recorded in receipts, not the owner of the global seal. EVAL inference requires the persisted authorization, `--protocol-registry-root "$PSEM_PROTOCOL_REGISTRY_ROOT"`, the frozen candidate identity, the exact authorized checkpoint, and the authorized external output root. DEV inference, lineage, canary, overfit, training, staging, and refreeze paths fail after the authority marker exists.

9. Infer and evaluate every frozen candidate on EVAL, then call `final-report`. It validates the complete authorization/result/prediction/checkpoint chain and writes the required metrics, per-source rows, frontiers, topology slices, 2,000-source bootstrap intervals, timing/compute report, decision receipt, and `ADAPTATION_DECISION.md`.

The final report compares issue-99 `G`, issue-99 Q8 `S-current`, `F0-FROZEN-FLOAT`, and each shallower arm; reports pooled, equal-corpus macro, AMI-only, AliMeeting-only, and mandatory topology views; and chooses only outcome A, B, C, or D under the predeclared shallowest-stable-depth rule.

## Scope boundary

This experiment does not perform KD, scratch-student training, native causal anchor lifecycle, production VAD execution, acoustic/NEST unfreezing, LoRA or adapters, architecture or hyperparameter sweeps, new-corpus collection, quantization, export, deployment benchmarking, or production-readiness claims.
