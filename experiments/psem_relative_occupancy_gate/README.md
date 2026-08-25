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
python -m experiments.psem_relative_occupancy_gate.run_sortformer_trace --role PSEM-STRATEGY-DEV --manifest <results>/relative_occupancy_manifest.jsonl --research-root <research> --output <results>/sortformer_model_receipt.json --resume
python -m experiments.psem_relative_occupancy_gate.run_lseend_trace --role PSEM-STRATEGY-DEV --manifest <results>/relative_occupancy_manifest.jsonl --research-root <research> --lseend-root <lseend> --output <results>/lseend_model_receipt.json --resume
python -m experiments.psem_relative_occupancy_gate.run_gate1 --manifest <results>/relative_occupancy_manifest.jsonl --sortformer-receipt <results>/sortformer_model_receipt.json --lseend-receipt <results>/lseend_model_receipt.json --output <results>/gate1_metrics.json --product-output <results>/gate1_product_frontier.json --topology-output <results>/gate1_topology_slices.json --latency-output <results>/gate1_latency_breakdown.json --event-output <results>/gate1_event_ledger.jsonl
python -m experiments.psem_relative_occupancy_gate.run_gate2 --manifest <results>/relative_occupancy_manifest.jsonl --sortformer-receipt <results>/sortformer_model_receipt.json --lseend-receipt <results>/lseend_model_receipt.json --gate0 <results>/gate0_oracle_metrics.json --gate0-verification <results>/gate0_verification.json --gate1 <results>/gate1_metrics.json --gate1-events <results>/gate1_event_ledger.jsonl --gate1-product <results>/gate1_product_frontier.json --gate1-topology <results>/gate1_topology_slices.json --latency <results>/gate1_latency_breakdown.json --output <results>/gate2_metrics.json --product-output <results>/product_frontiers.json --topology-output <results>/topology_slices.json --latency-output <results>/latency_breakdown.json --selection-output <results>/dev_selection_receipt.json --event-output <results>/gate2_event_ledger.jsonl
python -m experiments.psem_relative_occupancy_gate.verify_model_gates --manifest <results>/relative_occupancy_manifest.jsonl --sortformer-receipt <results>/sortformer_model_receipt.json --lseend-receipt <results>/lseend_model_receipt.json --gate0 <results>/gate0_oracle_metrics.json --gate0-verification <results>/gate0_verification.json --gate1 <results>/gate1_metrics.json --gate1-product <results>/gate1_product_frontier.json --gate1-topology <results>/gate1_topology_slices.json --gate1-latency <results>/gate1_latency_breakdown.json --gate1-events <results>/gate1_event_ledger.jsonl --gate2 <results>/gate2_metrics.json --gate2-events <results>/gate2_event_ledger.jsonl --product <results>/product_frontiers.json --topology <results>/topology_slices.json --latency <results>/latency_breakdown.json --selection <results>/dev_selection_receipt.json --output <results>/model_gate_verification.json
```

Gate 0 rejects EVAL unconditionally. Gate 1 and Gate 2 persist source-level event ledgers with boundary, evidence-frontier, emit, lifecycle, and fail-closed exposure fields. The verifier checks those ledgers structurally and also regenerates every DEV artifact. Gate 2 writes a DEV-selection receipt that binds the ontology, config, evaluator, model pins, trace schema, input manifests, selected settings, both event ledgers, Gate 0 verification, and an explicit sealed authorization state. The causal grid contains the 95 declared combinations satisfying `other_low_threshold < active_threshold`; no combination is silently discarded.

Sortformer traces use the pinned Vulkan build and a backend-specific trace root. Earlier CPU traces are non-authoritative diagnostics and cannot be resumed or combined with Vulkan receipts.

Only after the C2 checkpoint commit has passed its single owner review batch, open EVAL once with the accepted C2 head and the frozen DEV selection:

```powershell
python -m experiments.psem_relative_occupancy_gate.authorize_eval --selection <results>/dev_selection_receipt.json --verification <results>/model_gate_verification.json --accepted-c2-head <accepted-c2-head> --manifest-output <eval-results>/relative_occupancy_manifest.jsonl --output <eval-results>/eval_authorization.json
python -m experiments.psem_relative_occupancy_gate.derive_relative_occupancy --roles PSEM-STRATEGY-EVAL --frozen-selection <results>/dev_selection_receipt.json --eval-authorization <eval-results>/eval_authorization.json --output <eval-results>/relative_occupancy_manifest.jsonl
python -m experiments.psem_relative_occupancy_gate.run_eval_traces --manifest <eval-results>/relative_occupancy_manifest.jsonl --access-receipt <eval-results>/relative_occupancy_manifest_access_receipt.json --selection <results>/dev_selection_receipt.json --eval-authorization <eval-results>/eval_authorization.json --research-root <research> --lseend-root <lseend> --sortformer-output <eval-results>/sortformer_model_receipt.json --lseend-output <eval-results>/lseend_model_receipt.json --resume
python -m experiments.psem_relative_occupancy_gate.run_eval --manifest <eval-results>/relative_occupancy_manifest.jsonl --access-receipt <eval-results>/relative_occupancy_manifest_access_receipt.json --selection <results>/dev_selection_receipt.json --eval-authorization <eval-results>/eval_authorization.json --sortformer-receipt <eval-results>/sortformer_model_receipt.json --lseend-receipt <eval-results>/lseend_model_receipt.json --output <eval-results>/eval_metrics.json --product-output <eval-results>/product_frontiers.json --topology-output <eval-results>/topology_slices.json --latency-output <eval-results>/latency_breakdown.json
python -m experiments.psem_relative_occupancy_gate.verify_eval --manifest <eval-results>/relative_occupancy_manifest.jsonl --access-receipt <eval-results>/relative_occupancy_manifest_access_receipt.json --selection <results>/dev_selection_receipt.json --eval-authorization <eval-results>/eval_authorization.json --sortformer-receipt <eval-results>/sortformer_model_receipt.json --lseend-receipt <eval-results>/lseend_model_receipt.json --metrics <eval-results>/eval_metrics.json --product <eval-results>/product_frontiers.json --topology <eval-results>/topology_slices.json --latency <eval-results>/latency_breakdown.json --output <eval-results>/eval_verification.json
```

The EVAL authorization is bound to the exact accepted C2 Git head, canonical DEV verification path and hash, frozen selection, output target, and a single atomic consumption receipt. Every later EVAL entry point revalidates those bindings and the current head. The EVAL trace entry point then runs the same frozen model adapters sequentially and binds both aggregate model receipts to the opened selection and access receipt. The direct adapter CLIs remain limited to TRAIN smoke and sealed DEV execution.

Preflight revalidates the exact V2 freeze file, its complete artifact set, all 93 source/split/normalization bindings and waveform bytes, both model binaries, and both source checkouts. Git metadata for this experiment worktree is informational because a committed generated receipt cannot self-bind its final commit; load-bearing experiment files are bound directly by content hashes in the Gate results and verification receipt.

Posterior traces stay outside Git. Full-source receipts must prove contiguous native-frame coverage, an uninterrupted model epoch, the exact family/backend/role/source path, the configured model/provider or raw Vulkan bench receipt, and one shared external trace root per family. Receipts, event ledgers, metrics, topology slices, latency accounting, product frontiers, and `FINAL_DECISION.md` are the auditable outputs.
