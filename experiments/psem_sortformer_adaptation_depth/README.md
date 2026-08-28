# PSEM Sortformer adaptation depth

This namespace implements the issue-107 adaptation-depth experiment without changing the product runtime. It binds the immutable V2 data freeze, the issue-99 Simple Anchor evidence, the official float checkpoint, and one NVIDIA NeMo revision before any material metric may be produced.

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
