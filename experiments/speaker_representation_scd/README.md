# Speaker Representation SCD

This directory contains an experiment-only study of frozen speaker-change
representations. It is independent from the production runtime and from the legacy
`speaker_turn_boundary` result namespace.

The current authorized sequence is completed R0/R1, R2-L, reduced R3, reduced R4, and candidate
selection. R2-L uses only the already available legacy common-GT panel. R3 uses at most its 810
eligible positive/negative candidate rows at 100/300/500 ms. R4 uses one promoted layer/tap per encoder,
a common 300 ms primary context, a 100 ms primary hop, and at most six source hours; only the top
two encoders may receive a 50 ms sensitivity. It does not authorize model training, a learned probe
or head, confirmatory/test evaluation, fine-tuning, product wiring, or deployment.

The legacy ERes/LS-EEND common-GT manifest is the sole `development_known` experimental dataset.
Zeroth-Korean, JVS, D5, and all other new corpus acquisition are excluded. Preserved external
download attempts are historical only and must not be used or deleted. Future-test selection is
deferred until a separately approved learned-head study. LS-EEND is eligible only through
its existing event-level results and is never rerun or used for representation AUC/EER or layer ranking.

Validate the R0 contracts from the repository root:

```powershell
uv run python -m experiments.speaker_representation_scd.validate_r0
```

A valid R0 bundle may still report neural execution as blocked. R1 model acquisition, four-encoder
smoke, and parity are already complete; R2-L legacy validation and a measured reduced R3/R4
forecast are the next gate.

Primary documents:

- `EXPERIMENT_PLAN.en.md`: authority and full scientific plan
- `R0_BASELINE_DECISION_LEDGER.md`: existing-work audit and inheritance boundary
- `R0_DATASET_DECISION.md`: legacy-common-GT-only dataset decision
- `configs/protocol/`: machine-readable protocol and safety ceilings
- `configs/protocol/reduced_pretraining_screen.json`: owner-amended immediate R2-L/R3/R4 scope
- `NEXT_AGENT_HANDOFF.md`: exact continuation point, approval boundary, and worker instructions
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
