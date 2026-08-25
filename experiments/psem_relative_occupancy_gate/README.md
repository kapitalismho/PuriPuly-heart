# PSEM relative-occupancy gate

This namespace implements GitHub issue #97 without neural training. It evaluates the frozen sequence:

```text
Gate 0: GT occupancy + deterministic decoder
Gate 1: frozen posterior + episode-level oracle anchor
Gate 2: frozen posterior + causal anchor
```

The experiment reconstructs an integrity-checked derived view from immutable `PSEM-STRATEGY-DATA-v2`; it does not modify the V2 package.

Required local roots are supplied explicitly or through:

```text
PSEM_CORPUS_ROOT
PSEM_REFERENCE_ROOT
SRSCD_CACHE_ROOT
PSEM_LSEEND_ROOT
```

Typical execution:

```powershell
python -m experiments.psem_relative_occupancy_gate.preflight --output <results>/preflight_receipt.json
python -m experiments.psem_relative_occupancy_gate.derive_relative_occupancy --roles PSEM-STRATEGY-DEV --output <results>/relative_occupancy_manifest.jsonl
python -m experiments.psem_relative_occupancy_gate.run_gate0 --manifest <results>/relative_occupancy_manifest.jsonl --preflight <results>/preflight_receipt.json --output-dir <results>
python -m experiments.psem_relative_occupancy_gate.verify_gate0 --output-dir <results> --receipt <results>/gate0_verification.json
python -m experiments.psem_relative_occupancy_gate.run_sortformer_trace --role PSEM-STRATEGY-DEV --manifest <results>/relative_occupancy_manifest.jsonl --output-dir <cache>/sortformer
python -m experiments.psem_relative_occupancy_gate.run_lseend_trace --role PSEM-STRATEGY-DEV --manifest <results>/relative_occupancy_manifest.jsonl --output-dir <cache>/lseend
python -m experiments.psem_relative_occupancy_gate.run_gate1 --role PSEM-STRATEGY-DEV --manifest <results>/relative_occupancy_manifest.jsonl --trace-dir <cache> --output <results>/gate1_dev_metrics.json
python -m experiments.psem_relative_occupancy_gate.run_gate2 --role PSEM-STRATEGY-DEV --manifest <results>/relative_occupancy_manifest.jsonl --trace-dir <cache> --output <results>/gate2_dev_metrics.json
```

The current Gate 0 implementation rejects EVAL unconditionally. A later frozen DEV-selection checkpoint must bind the ontology, config, evaluator, model pins, trace schema, input manifests, and selected settings before EVAL can be opened once for the final Gate 1/2 comparison.

Preflight revalidates the exact V2 freeze file, its complete artifact set, all 93 source/split/normalization bindings and waveform bytes, both model binaries, and both source checkouts. Git metadata for this experiment worktree is informational because a committed generated receipt cannot self-bind its final commit; load-bearing experiment files are bound directly by content hashes in the Gate results and verification receipt.

Posterior traces stay outside Git. Receipts, metrics, topology slices, latency accounting, product frontiers, and `FINAL_DECISION.md` are the auditable outputs.
