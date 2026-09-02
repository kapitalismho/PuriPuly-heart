# Issue 107 post-hoc recalibration

This is exploratory post-hoc analysis, not a replacement for the official Issue 107 decision.

## Method

- Fit set: first 512 fixed PSEM-STRATEGY-TRAIN manifest crops
- Fit frames: 153,475 valid frames
- Positive prevalence: 0.0887115
- Calibration: positive-slope Platt scaling, fitted independently for H-HEAD and T2-TOP
- DEV grid after freezing calibration: thresholds 0.2, 0.35, 0.5, 0.65, 0.8 and confirmation 100, 300, 500 ms
- Both raw and calibrated DEV frontiers are retained in `recalibration-result.json`.

## Calibration fit

| Arm | Scale | Offset | Calibrated 0.5 as raw probability | Raw NLL | Calibrated NLL | Raw Brier | Calibrated Brier |
|---|---:|---:|---:|---:|---:|---:|---:|
| H-HEAD | 0.849198 | -1.838212 | 0.897029 | 0.303591 | 0.189073 | 0.091540 | 0.052658 |
| T2-TOP | 0.862198 | -1.941521 | 0.904808 | 0.299407 | 0.177121 | 0.090835 | 0.050078 |

The fitted operating shift is consistent with the weighted-BCE concern: a calibrated 0.5 corresponds to a raw model probability near 0.90, not raw 0.5.

## DEV result

F0 frozen at the official 0.5 / 500 ms cell:

| Arm/cell | Contamination s/h | False cuts/h | Misses/h |
|---|---:|---:|---:|
| F0 0.5 / 500 ms | 1902.41 | 45.85 | 218.80 |
| H calibrated 0.65 / 300 ms | 1855.23 | 56.76 | 202.10 |
| T2 calibrated 0.65 / 300 ms | 1773.82 | 56.76 | 194.09 |
| H calibrated 0.8 / 300 ms | 2314.87 | 21.59 | 274.44 |
| T2 calibrated 0.8 / 300 ms | 2241.18 | 22.70 | 269.55 |

No tested H-HEAD or T2-TOP cell dominates F0 on all three metrics. T2 at calibrated 0.65 / 300 ms is the nearest promising tested trade-off: contamination and misses improve over F0, while false cuts remain about 10.91/h higher.

## Interpretation limits

- Platt scaling is monotone, so it changes calibration and operating-point selection but cannot improve the underlying ranking.
- The grid is coarse. An intermediate threshold might occupy a better point, but choosing it from DEV would remain exploratory post-hoc selection.
- This analysis does not correct the TRAIN crop-reset versus continuous DEV-state mismatch.
- The supported conclusion remains that calibration mismatch materially affected the official 0.5 operating point, but calibration alone did not rescue the Issue 107 recipe on the tested grid.
