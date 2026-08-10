# Speaker Representation SCD

This directory contains an experiment-only, public-data study of frozen speaker-change
representations. It is independent from the production runtime and from the legacy
`speaker_turn_boundary` result namespace.

The current authorized sequence is R0, R1, R2, R3, R4, and R6-Z. It covers deterministic
protocol freeze, frozen feature extraction, zero-shot representation analysis, continuous
zero-shot SCD, and a one-time public confirmatory report. It does not authorize model training,
a learned probe or head, fine-tuning, product wiring, or deployment.

The legacy ERes/LS-EEND common-GT manifest is reused only as `development_known`. The public
confirmatory partitions remain sealed until model, split, analysis, code, and run-contract hashes
are locked. LS-EEND is eligible only for event-level contextual comparison and never for
representation AUC/EER or layer ranking.

Validate the R0 contracts from the repository root:

```powershell
uv run python -m experiments.speaker_representation_scd.validate_r0
```

A valid R0 bundle may still report neural execution and confirmatory access as blocked. That is
intentional: weight/corpus acquisition, legacy-run release, a measured smoke forecast, extractor
parity, and a complete confirmatory lock are later gates.

Primary documents:

- `EXPERIMENT_PLAN.en.md`: authority and full scientific plan
- `R0_BASELINE_DECISION_LEDGER.md`: existing-work audit and inheritance boundary
- `R0_DATASET_DECISION.md`: public-only dataset and split decision
- `configs/protocol/`: machine-readable protocol and safety ceilings
- `data/`: source, split, and confirmatory-access contracts
- `models/registry.json`: immutable model identities and unresolved extraction blockers

No model or corpus binary belongs in Git. External material must use an explicit
`SRSCD_CACHE_ROOT` after its acquisition and storage gate is satisfied.
