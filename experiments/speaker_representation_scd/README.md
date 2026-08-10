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

R1 preparation adds an experiment-local CPU research lock, an exact model/source bridge, a
fail-closed acquisition gate, and deterministic extractor smoke tests. Validate that checkpoint
from the repository root with:

```powershell
uv lock --project experiments/speaker_representation_scd/environment --check --python .venv/Scripts/python.exe --no-python-downloads
uv run python -m experiments.speaker_representation_scd.validate_r1_gate
```

The R1 validator is expected to fail until the independent legacy verifier has completed and the
reviewed acquisition gate has been materialized. A passing R1 acquisition gate permits only the
locked environment sync, exact model/source acquisition, and ten-fixture/100-window smoke. It does
not permit corpus download, full extraction, confirmatory access, or training.

All stateful R1 actions use the supervised entrypoint rather than invoking acquisition or smoke
workers directly:

```powershell
$env:SRSCD_CACHE_ROOT = 'C:\Users\salee\AppData\Local\puripuly-heart-research\speaker_representation_scd_v1'
uv run python -m experiments.speaker_representation_scd.r1_execute sync-environment
uv run python -m experiments.speaker_representation_scd.r1_execute models
uv run python -m experiments.speaker_representation_scd.r1_execute smoke --model mhubert-147
```

The entrypoint holds one external lease, creates each worker suspended inside a Windows Job Object,
keeps every descendant under a conservative hard-memory limit below the 24 GiB contract, and
persists the Job Object's authoritative peak in a no-overwrite usage receipt. It also continuously
monitors legacy-process contention and diagnostic process-tree RSS while enforcing the 24-hour
action and 96-hour cumulative ceilings.

Action receipts are lease-bound but non-authoritative on their own. Each downstream phase requires
one unique completed usage attestation containing the same execution ID, receipt path/hash, and
hard-memory accounting. A receipt left by an aborted action is preserved under `control/orphans/`
and retried safely rather than unlocking the next phase or blocking the final receipt path.
