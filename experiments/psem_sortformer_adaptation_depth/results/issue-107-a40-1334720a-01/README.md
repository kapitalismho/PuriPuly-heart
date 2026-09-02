# Issue 107 A40 result bundle

This directory preserves the decision-bearing artifacts and compressed frame predictions exported from RunPod run `issue-107-a40-1334720a-01`.

The run completed `F0-FROZEN-FLOAT`, `H-HEAD`, and `T2-TOP`, then entered `WAITING_FOR_DECISION` after `t2-top-material-and-dev`. The Issue 107 rule resolves the result to `STOP`: neither adapted arm improves at least two pooled metrics over F0, and T2 does not clear the required 10% material-gain condition relative to H. `TA-ALL-TEMPORAL` will not run, and EVAL was not opened because there is no supported adapted candidate.

## Primary DEV result

Lower is better for every metric.

| Arm | Contamination | False cuts/hour | Missed replacements/hour | Matched cuts | False cuts | Missed replacements |
|---|---:|---:|---:|---:|---:|---:|
| F0-FROZEN-FLOAT | 1902.41 | 45.85 | 218.80 | 609 | 206 | 983 |
| H-HEAD | 2154.20 | 75.01 | 252.41 | 458 | 337 | 1134 |
| T2-TOP | 2066.82 | 71.45 | 242.17 | 504 | 321 | 1088 |

T2 improves all three pooled metrics by roughly 4-5% relative to H, but remains worse than F0 on all three metrics. This does not authorize TA under Issue 107.

## Directory map

- `LEAN_ADAPTATION_DECISION.md`: bounded decision and caveats
- `control/`: exported detached-run state and event history
- `logs/`: complete phase stdout/stderr, renamed from `.log` to `.log.txt` to avoid the repository-wide log ignore rule
- `receipts/`: metric results, prediction manifests, training/smoke/cost receipts, sampling manifest, runtime identity, and phase summaries
- `artifacts/issue-107-dev-predictions.tar.gz`: all F0/H/T2 DEV frame predictions
- `artifacts/issue-107-lineage-predictions.tar.gz`: all exported lineage frame predictions

Extract the frame artifacts from this directory with:

```text
tar -xzf artifacts/issue-107-dev-predictions.tar.gz
tar -xzf artifacts/issue-107-lineage-predictions.tar.gz
```

## Large artifacts not committed

The two model checkpoints are intentionally not stored in Git because they total about 943 MiB. Their identities remain bound by the committed checkpoint receipts:

| Arm | Bytes | SHA-256 |
|---|---:|---|
| H-HEAD | 471607147 | `66fe467237330e2be1328de7c120ba4569e5f173558d8298f581eb73392fc6f7` |
| T2-TOP | 471606146 | `f0f7720609eafc564b7a76dcfcef0929861e7c0fbb9bb6661cbd32522be82325` |

The complete local output archive was verified before extraction as 1024079252 bytes with SHA-256 `a55e1ce2207fb1e376e7a3914fc78cf36cc8e518dfebacc841753eb449026a59`.

## Independent review boundary

This bundle is sufficient to audit the recorded DEV decision, per-source/corpus/topology results, raw score behavior, sampling composition, training health, runtime identities, and cost-receipt consistency. It does not contain source audio or the complete normalized reference materialization, so exact rescoring at a new threshold/confirmation point and acoustic annotation review require the frozen external data.

The post-hoc limitations and suggestions in `LEAN_ADAPTATION_DECISION.md` are informal review hypotheses only. They are not amendments to Issue 107, authorization for more compute, or established causal explanations.
