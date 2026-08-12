# R7-B Fixed-Lag Local Speaker Segmentation Approved Addendum

## 1. Approval and Scope

The owner approved revised scope 2 on 2026-08-12: implement and run the B0 frozen-ERes
joint-segmentation control and the B1 representation-revision experiment. B1 evaluation is allowed
only after the mandatory development gates in the amended R7 plan pass.

This approval does not authorize product integration, production code changes, deployment, a larger
decoder rescue, latency beyond 1,000 ms, or an unbounded representation sweep.

## 2. Evidence Mode

R7-B uses the fast internal-decision mode. All ten meetings previously used by R6 and R7-A are
development-known. They are arranged into five fixed meeting-held folds:

| Fold | Held-out meetings |
| --- | --- |
| 1 | `alimeeting_R8001_M8004`, `ami_IS1009a` |
| 2 | `alimeeting_R8008_M8013`, `ami_EN2001d` |
| 3 | `alimeeting_R8009_M8019`, `ami_TS3006a` |
| 4 | `ami_ES2003a`, `alimeeting_R8007_M8010` |
| 5 | `ami_TS3009b`, `ami_ES2015d` |

Every reported development score is out-of-fold. Each fold uses the following fold as validation
and the remaining six meetings for training. No untouched or promotion-capable claim is permitted.
No separate evaluation panel is opened in this addendum. A passing result stops with a request to
freeze a new untouched natural panel before any confirmatory run.

## 3. Shared Output Semantics

Both arms operate on a 100 ms grid. For a possible boundary at `t`, the input sequence contains
grid cells with feature frontiers from `t - 500 ms` through `t + 1,000 ms`. The source boundary
timestamp remains `t`; the decision becomes available at `t + 1,000 ms` plus measured compute.

The model predicts an identity-invariant local partition:

- a pairwise same-speaker relation for singleton-speaker cells;
- a three-state cell label: silence, singleton speech, or overlap;
- no global speaker name, enrollment identity, or persistent speaker slot.

Boundary scores are derived from the decoded partition. They combine adjacent singleton-speaker
changes, speaker changes across a short silence gap, and predicted overlap onset. The model is not
trained as an independent boundary-pulse classifier. Local maxima and the fixed 200 ms duplicate
suppression rule convert scores into events. Multiple boundaries and short `A → B → A` patterns
remain representable within one rolling window.

Ambiguous annotation cells are excluded from training losses. Event scoring retains the R7-A
`new_speaker_onset` semantics and one-to-one matching at 100, 250, and 500 ms.

## 4. B0 Frozen-ERes Control

B0 reuses the existing 500 ms ERes2NetV2 E-standard final embeddings at 100 ms hops. No candidate
gate is used. A small shared temporal encoder produces local cell states and pairwise relation
embeddings. Training uses pairwise same-speaker loss and cell-state loss only.

B0 is a falsification control. If it fails either aggregate gate, the frozen-ERes segmentation path
stops. No additional B0 seed sweep, larger model, longer context, or evaluation run is authorized.

## 5. B1 Representation Revision

B1 uses the same output semantics and model size as B0, but revises the evidence by concatenating
short-time PCM features to every 100 ms cell:

- mean and standard deviation of 40 log-mel bands from 25 ms frames at 10 ms hops;
- log RMS energy and within-cell speech-frame fraction;
- zero-crossing rate;
- normalized spectral centroid and spectral flatness.

The three-state auxiliary loss is the explicit overlap objective. PCM features are computed with
fixed transforms; feature normalization is fitted on training meetings only inside each fold.
Ground-truth speech, speaker, overlap, or event labels are never inference inputs.

Only this PCM-plus-overlap representation revision is authorized. Encoder fine-tuning, additional
encoders, pitch extractors, augmentation, and architecture search are outside this run.

## 6. Model and Training Boundaries

Each arm uses one input projection, one small bidirectional recurrent temporal layer over the local
fixed-lag window, one pairwise relation head, and one three-state head. Model size and optimization
settings are fixed in the checked-in R7-B configuration before material training.

Training windows consist of a uniform 500 ms background grid plus every 100 ms grid point within
500 ms of a reference change. Two fixed seeds are averaged. Early stopping uses only the fixed
validation fold. Thresholds are selected only from complete out-of-fold development predictions.

## 7. Baseline and Gates

The shared baseline is the frozen adjacent ERes cosine score evaluated continuously without the
R7-A candidate threshold. Its threshold is selected from the same out-of-fold exposure.

B0 improvement means higher per-fold Recall@250 than that baseline at each arm's aggregate
10-false-events/hour operating point. B1 improvement means higher per-fold Recall@250 than B0 at
the corresponding operating point.

An arm passes only if all amended R7 gates hold:

- aggregate Recall@250 is at least 30% at no more than 10 false events/hour;
- aggregate Recall@250 is at least 50% at no more than 20 false events/hour;
- improvement appears in at least four of five held-out folds;
- no held-out fold exceeds twice the selected false-events/hour target;
- overlap-onset and silence-gap-change recall are both non-zero;
- no meeting contributes more than half of matched true positives;
- short valid `A → B → A` events are not collapsed by suppression.

B0 failure does not block the already approved B1 development experiment. B1 failure stops R7-B.
B1 success stops with an internal result and a request to authorize and freeze a new untouched
evaluation panel.

## 8. Execution and Artifacts

The coordinator owns code and configuration. PCM extraction, model training, and full continuous
out-of-fold scoring are CPU-heavy and must run through an OpenCode worker controlled by the Orca
CLI. Long jobs are supervised with approximately 15-minute waits.

Artifacts are stored outside the repository under:

```text
%SRSCD_CACHE_ROOT%/results/r7b/fixed_lag_local_segmentation_v1/
```

Required outputs are the frozen configuration, inventory, cached PCM features, B0 and B1
out-of-fold predictions, metrics, model receipts, curves, representative timelines, and a concise
report. No production module is changed and no architecture drift is expected.
