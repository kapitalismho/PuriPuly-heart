# Issue 121 H7301 persistence analysis

This is a posthoc exploratory analysis of the frozen H7301 DEV export. It does not change the Gate 1 decision, open H7302/T2/TA/EVAL, or establish a source-level cause.

## Decision context

The scientific decision remains **STOP / inconclusive; retain F0**. The 100 ms point gains on the declared equal-corpus macro criterion, but only one of three horizons is jointly useful. The 300 ms and 500 ms points fail the joint criterion. No causal conclusion is established.

## DEV ranking AP

AP is computed posthoc over valid-and-mapped DEV frames using the existing `average_precision` implementation. The earlier report called DEV F0 AP unavailable; this bundle computes it now and does not retroactively treat it as preregistered evidence.

| Population | Candidate AP | F0 AP |
| --- | ---: | ---: |
| pooled valid+mapped frames (195375) | 0.487681950383668 | 0.150603849645612 |
| AMI source macro (7) | 0.491683249822344 | 0.144531097981704 |
| AliMeeting source macro (3) | 0.430875816411725 | 0.174723625454627 |
| all ten source macro | 0.473441019799158 | 0.153588856223581 |

The candidate pooled AP is `0.487681950383668`; candidate source-macro AP is AMI `0.491683249822344`, AliMeeting `0.430875816411725`, all ten `0.473441019799158`. Pooled and source-macro quantities are different estimands.

## Global frontier metrics

The following are the canonical equal-corpus macro raw-score metrics. The global threshold is shared across all ten meetings for each envelope; the independent corpus envelope points in the canonical JSON use separate corpus-specific thresholds and are not interchangeable with this table.

| Horizon | Point | Threshold | Contamination | Miss rate | False cuts/hour |
| ---: | --- | ---: | ---: | ---: | ---: |
| 100 ms | F0 reference | 0.5 | 2161.675241 | 0.775487185 | 228.734101 |
| 100 ms | H C-envelope | 0.5887844788775033 | 1561.578295 | 0.583873597 | 108.641572 |
| 100 ms | H M-envelope | 0.3631036176874745 | 1570.013238 | 0.571039961 | 151.139128 |
| 300 ms | F0 reference | 0.5 | 1555.186118 | 0.537088641 | 92.282108 |
| 300 ms | H C-envelope | 0.391021290491487 | 1566.368287 | 0.576840689 | 89.445433 |
| 300 ms | H M-envelope | 0.391021290491487 | 1566.368287 | 0.576840689 | 89.445433 |
| 500 ms | F0 reference | 0.5 | 1978.647650 | 0.687291477 | 30.487115 |
| 500 ms | H C-envelope | 0.7234637539961835 | 2232.236730 | 0.817277070 | 30.451885 |
| 500 ms | H M-envelope | 0.7234637539961835 | 2232.236730 | 0.817277070 | 30.451885 |

Raw and calibrated event metrics are identical at the selected points; calibration only changes score coordinates. Bootstrap intervals in the canonical diagnostics are paired source/meeting-mean intervals (2,000 replicates), not pooled-rate or macro confidence intervals. Timing p90 claims are per meeting, not pooled.

## Positive event-generating runs

A run is high when speech is present and `cand_raw_prob >= threshold`. It must be contiguous in source samples; invalid frames close/reset a run, masked frames are skipped without duration or previous-end updates, and speech-absent or below-threshold frames close a run. Duration is the sum of eligible `(end-start)` intervals divided by 16 ms, measured through continuation after emission. A run is matched only by `monotonic_boundary_matches` with the configured 500 ms product-event tolerance; 500 ms is not the horizon.

| Analysis | Horizon | Threshold | Runs | Matched median/p90/max ms | Unmatched median/p90/max ms |
| --- | ---: | ---: | ---: | --- | --- |
| H100_C | 100 | 0.5887844788775033 | 1262 | 480.0/589.0/960.0 (599) | 300.0/1000.0/4700.0 (663) |
| H100_M | 100 | 0.3631036176874745 | 1490 | 494.0/600.0/995.0 (571) | 300.0/1700.0/8000.0 (919) |
| H300_C | 300 | 0.391021290491487 | 1200 | 500.0/668.2/995.0 (655) | 400.0/1400.0/4700.0 (545) |
| H300_M | 300 | 0.391021290491487 | 1200 | 500.0/668.2/995.0 (655) | 400.0/1400.0/4700.0 (545) |
| H500_C | 500 | 0.7234637539961835 | 540 | 500.0/694.5/972.0 (344) | 892.0/2131.0/4700.0 (196) |
| H500_M | 500 | 0.7234637539961835 | 540 | 500.0/694.5/972.0 (344) | 892.0/2131.0/4700.0 (196) |
| held_H500_C_threshold_H100 | 100 | 0.7234637539961835 | 1216 | 468.5/572.5/972.0 (596) | 300.0/1000.0/4700.0 (620) |
| held_H500_C_threshold_H300 | 300 | 0.7234637539961835 | 1092 | 484.0/581.2/972.0 (655) | 400.0/1300.0/4700.0 (437) |
| held_H500_C_threshold_H500 | 500 | 0.7234637539961835 | 540 | 500.0/694.5/972.0 (344) | 892.0/2131.0/4700.0 (196) |

At the exact selected global C-envelope thresholds, the matched/unmatched run summaries are H100 `599: 480/589/960 ms` and `663: 300/1000/4700 ms`; H300 `655: 500/668.2/995 ms` and `545: 400/1400/4700 ms`; H500 `344: 500/694.5/972 ms` and `196: 892/2131/4700 ms`. Holding the exact H500 C-envelope score threshold while changing only the horizon gives unmatched `300/1000/4700 ms` at H100, `400/1300/4700 ms` at H300, and `892/2131/4700 ms` at H500. This is descriptive persistence/confirmation evidence; unmatched runs can include timing or matching failures and are not categorical ground-truth false positives.

## Reproduction and provenance

Run from the repository root:

```text
uv run python -m experiments.psem_state_corrected_adaptation_gate.results.issue-121-h7301-persistence-v1.persistence_analysis \
  --export-dir experiments/psem_state_corrected_adaptation_gate/results/issue-121-h7301-persistence-v1/export/gpu_export \
  --frontier experiments/psem_state_corrected_adaptation_gate/results/issue-121-h7301-persistence-v1/canonical/dev_frontier.json.gz \
  --diagnostics experiments/psem_state_corrected_adaptation_gate/results/issue-121-h7301-persistence-v1/canonical/gate1_diagnostics.json \
  --out-json experiments/psem_state_corrected_adaptation_gate/results/issue-121-h7301-persistence-v1/persistence_analysis.json \
  --out-md experiments/psem_state_corrected_adaptation_gate/results/issue-121-h7301-persistence-v1/PERSISTENCE_ANALYSIS.md
```

The gzip frontier is deterministic and decompresses byte-for-byte to the canonical frontier SHA recorded in `bundle_manifest.json`. The durable export contains only numeric NPZ arrays, the immutable export manifest, and training metrics; no audio, transcripts, checkpoints, credentials, or process logs are included.

Observed frozen execution: 29m18s CPU postprocess, 53 FIT sources, 142 training steps, 11 CALIB NPZ and 10 DEV NPZ. The GPU/source binding, CPU postprocess source hash, trained-head hash, and export manifest hash are recorded in `bundle_manifest.json`.

Formal commit review remains outstanding. This bundle is prepared for Director review; it has not been committed, pushed, or posted.
