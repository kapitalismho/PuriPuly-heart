# PSEM ontology simplification gate

This issue-98 experiment reuses the completed issue-97 frozen Sortformer and LS-EEND posterior traces. It tests `VAD + anchor_present`, the anchor-conditioned overlap challenger, and the next development path without neural training.

Run unit tests first:

```powershell
uv run pytest experiments/psem_ontology_simplification_gate/tests -q
```

Generate the trace inventory and causal dependency audit:

```powershell
uv run python -m experiments.psem_ontology_simplification_gate.evaluate_simplified_ontologies --inventory-only
```

Run DEV and EVAL offline replay from the cached issue-97 traces:

```powershell
uv run python -m experiments.psem_ontology_simplification_gate.evaluate_simplified_ontologies --role dev
uv run python -m experiments.psem_ontology_simplification_gate.evaluate_simplified_ontologies --role eval
```

No speaker-model inference is performed by these commands. The EVAL evidence is recovery-qualified and development-known, matching issue #97.

The bundled production peer-VAD path is pinned and eligible for deterministic exact-V2 sensitivity replay. Run its CPU inference separately, then derive the sensitivity result:

```powershell
uv run python -m experiments.psem_ontology_simplification_gate.run_production_vad --role dev
uv run python -m experiments.psem_ontology_simplification_gate.evaluate_simplified_ontologies --role dev --production-vad-sensitivity
uv run python -m experiments.psem_ontology_simplification_gate.run_production_vad --role eval
uv run python -m experiments.psem_ontology_simplification_gate.evaluate_simplified_ontologies --role eval --production-vad-sensitivity
```

The VAD replay uses the pinned bundled Silero 6.2.1 ONNX model with the production peer profile and no tuning. Its source-time gate covers pre-roll plus committed chunks through the speech end, excluding trailing hangover. This is a development-known sensitivity arm, not a production-readiness claim.

Verify the complete derived result set:

```powershell
uv run python -m experiments.psem_ontology_simplification_gate.verify_results --role dev --require-production-vad
uv run python -m experiments.psem_ontology_simplification_gate.verify_results --role eval --require-production-vad
```

The final interpretation and ordered issue answers are in `results/eval/PATH_DECISION.md`.

`expected_results_manifest.json` is the independently reviewed byte-level seal for every material DEV/EVAL result, including all frontier cells, diagnostics, paired deltas, bootstrap intervals, production-VAD sensitivity, and the final report. The verifier compares current files with this committed seal and never creates or refreshes it.

Global-overlap event recall uses contiguous unmasked GT overlap runs. Duration buckets are `<500 ms`, `500–1500 ms`, and `>=1500 ms`. A run is recalled when at least one covered evaluation cell meets the predeclared threshold. A short backchannel is an overlap run no longer than 1000 ms with singleton speech on at least one immediate side.

Sustained anchor-dropout probability is duration-weighted: target support contained in contiguous below-threshold runs at least as long as the requested horizon divided by all unmasked target support. Counts and affected episode fractions are reported beside it.
