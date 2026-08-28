# PSEM Sortformer adaptation depth

This namespace implements the issue-107 adaptation-depth experiment without changing the product runtime. It binds the immutable V2 data freeze, the issue-99 Simple Anchor evidence, the official float checkpoint, one NVIDIA NeMo revision, and `runtime_contract.json` before any material metric may be produced.

The first local gate is read-only and requires no model or corpus access:

```powershell
uv run --project experiments\speaker_representation_scd\environment --frozen python -m experiments.psem_sortformer_adaptation_depth.run preflight --static-only
uv run --project experiments\speaker_representation_scd\environment --frozen pytest experiments\psem_sortformer_adaptation_depth\tests -q
```

The runtime gate is intentionally fail-closed. It requires Linux, the exact `.nemo` file, byte-for-byte verification of all 93 frozen source waveforms, the exact forced-alignment Git checkout, a clean worktree, and an existing absolute experiment cache outside the repository. It keeps EVAL sealed while preflight and DEV work run.

```text
PSEM_SORTFORMER_NEMO_PATH
PSEM_CORPUS_ROOT
PSEM_REFERENCE_ROOT
PSEM_ADAPTATION_OUTPUT_ROOT
```

RunPod provisioning, corpus or checkpoint upload, credential use, and any other external-service mutation require owner approval before execution. A passing file/path preflight is not authorization for material training; model graph, lineage, gradient, update, evaluator reconstruction, and overfit-canary receipts must also pass.

PSEM feature construction rejects non-binary slot, anchor, or reset indicators, non-one-hot anchors, and any model-evidence delay other than 1.04 seconds. Composite training loss accepts only TRAIN batches and a native Sortformer scalar bound to the frozen NeMo loss kind, adapter origin, and checkpoint identity.

The runtime implementation fixes the #99 low-latency bundle at `chunk/right-context/FIFO/update/cache = 6/7/188/144/188` native 80 ms frames. It captures the 192-dimensional `transformer_encoder` output and the `single_hidden_to_spks` output before sigmoid, then carries both through the same streaming cache update that produces the four arrival-order posteriors. Because #99 exposed stable columns but no slot-validity metadata, all four columns retain the exact #99 alive semantics; posterior thresholds never redefine cache or slot lifecycle.

The shared TRAIN manifest uses 4,096 contiguous 30-second windows per epoch for at most eight epochs. Every epoch is exactly 50% source-time-uniform, 25% replacement-positive, and 25% hard-negative. Window, target-recipe, and four-family label-independent augmentation identities are independent of arm and seed; EVAL cannot enter the loader.

The following local commands inspect committed identities without opening EVAL or starting training:

```powershell
uv run --project experiments\speaker_representation_scd\environment --frozen python -m experiments.psem_sortformer_adaptation_depth.run data-split-receipt
uv run --project experiments\speaker_representation_scd\environment --frozen python -m experiments.psem_sortformer_adaptation_depth.run evaluator-contract
```

Sampling materialization and lineage/overfit validation are runtime-stage commands. Their outputs belong in the external experiment cache until a small receipt is accepted for the repository.

Freeze the complete installed Python distribution inventory on the Linux training runtime before loading the checkpoint:

```powershell
python -m experiments.psem_sortformer_adaptation_depth.run dependency-lock --output <external-cache>/nemo_dependency_lock.json
python -m experiments.psem_sortformer_adaptation_depth.run model-graph --checkpoint <checkpoint> --nemo-checkout <checkout> --dependency-lock <external-cache>/nemo_dependency_lock.json --device cuda
```

The loader rejects a partial package list, a version or platform mismatch, any loaded `nemo.*` module outside the pinned checkout, an executable 18-block/activity/PSEM graph mismatch, an altered context restoration, a non-native final frame, or a PSEM head left on another device. Lineage validation streams each external prediction file, requires exact full-source native-frame coverage with an explicit sub-frame tail, and recomputes Float/Q8 prediction-set and paired-bootstrap hashes.

Official training code accepts only a `material_training_authorization` produced by `validate-material-gate`. That composite gate revalidates the clean preflight Git head, exact split and persisted sampling bytes, recomputed TRAIN class weights, executable runtime/lineage/evaluator identities, parameter inventory, raw-waveform gradient, update, and timing canaries, exact 60-window overfit result, and the ordered DEV-only arm/seed decision. Training rechecks that same clean Git head before the first optimizer step. `TA-ALL-TEMPORAL`, seed 7302, or any development path with an opened EVAL fails closed unless its predeclared DEV rule is proven.

```powershell
python -m experiments.psem_sortformer_adaptation_depth.run validate-material-gate <receipt-bundle.json> --manifest <sampling.jsonl> --corpus-root <corpus-root> --reference-root <reference-root>
```
