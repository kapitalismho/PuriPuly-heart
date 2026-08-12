# R5 Frozen-Encoder Causal SCD Head Experiment Plan

## 1. Document Status

This document defines the proposed R5 experiment that follows the completed R3 representation
screen and R4 continuous zero-shot SCD measurement.

It is an experiment plan, not an authorization to start training. R5 execution requires an explicit
owner instruction. Actual experiment runs must be assigned to an OpenCode worker through the Orca
CLI. The coordinator owns documentation and small scoring corrections but does not directly run the
experiment.

The priority is obtaining a useful experimental answer. Harness expansion, broad framework
refactoring, exhaustive validation infrastructure, and production integration are outside scope.

## 2. Decision Summary

R5 asks one question:

> Can a small learned causal temporal head turn the frozen speaker-change information observed in
> R3 into a useful continuous event detector while suppressing the false events that made the R4
> zero-shot cosine detectors impractical?

R5 is a fail-fast learned-probe study. It is technically a training experiment, but it is not encoder
fine-tuning or production-model training.

```text
completed R3/R4 frozen features
             ↓
R5-A: linear probes on all four encoders
             ↓
R5-B0: same small causal TCN, no augmentation
             ↓
R5-B1: top two only, continuous negatives plus targeted augmentation
             ↓
go / stop decision for a later public-data study
```

The four starting representations are:

| Encoder | Frozen representation | R5-A | R5-B0 |
| --- | --- | --- | --- |
| ERes2NetV2 pre-pooling | S3, 300 ms | Yes | Yes |
| mHuBERT-147 | L1, 300 ms | Yes | Yes |
| UniSpeech-SAT Base+ | L1, 300 ms | Yes | Yes by default |
| WavLM Base+ | L3, 300 ms | Yes | Only if R5-A promotes it |

No ERes final-embedding inference and no LS-EEND inference are rerun. Their existing compatible
event-level results remain contextual baselines only.

## 3. Why R5 Exists

R3 showed that the frozen representations contain speaker-discriminative information at short
contexts. ERes S3 was strongest, mHuBERT L1 was the strongest SSL candidate, UniSpeech L1 followed,
and WavLM did not show the expected speaker-aware advantage.

R4 showed that representation quality did not translate into a usable threshold detector. At the
low-false-event operating region, the detectors either emitted nothing or recovered almost none of
the ground-truth changes. More permissive thresholds recovered changes only by producing many false
events.

This supports a narrow next hypothesis:

> The representation contains useful information, but cosine thresholding lacks the temporal memory
> needed to distinguish a persistent speaker transition from phonetic, prosodic, pause, channel, and
> noise variation.

R5 tests this hypothesis with the smallest reasonable learned head while keeping every encoder
frozen.

## 4. Claims R5 May and May Not Support

R5 may support:

- whether a frozen representation is learnable for low-latency SCD;
- whether short causal memory improves the R4 accuracy-versus-false-event trade-off;
- which of the current representations is the best candidate for a later learned-head study;
- whether targeted same-speaker hard negatives improve continuous robustness;
- whether ERes remains attractive after accounting for its much larger input dimension.

R5 may not support:

- production readiness;
- broad multilingual generalization;
- Korean or Japanese performance;
- a confirmed false-event rate at or below one event per hour;
- final handoff classification;
- superiority on a fresh public test set;
- claims about encoder fine-tuning.

The R4 panel is development-known and already influenced candidate selection. It is an internal
continuous evaluation panel, not an untouched test set.

## 5. Non-Goals

R5 does not include:

- encoder fine-tuning, partial unfreezing, or learned layer fusion;
- a Conformer or other large temporal model;
- VAD, OSD, handoff, or speaker-state multi-task learning;
- a large hyperparameter sweep;
- new model acquisition;
- new public-corpus acquisition;
- rerunning legacy ERes final embeddings or LS-EEND;
- product integration, ONNX export, quantization, or distillation;
- broad experiment-harness hardening.

An untouched public-data study is deferred until R5 identifies a candidate worth validating.

## 6. Frozen Inputs

### 6.1 Existing inventory

The legacy common-GT inventory contains:

- 804 total episodes;
- 695 diagnostic episodes;
- 616 source identities;
- 600 unique WAV byte identities;
- 445 R3 sessions;
- 810 R3 anchors: 450 positive and 360 negative;
- 313 existing matched pairs.

The frozen R4 continuous panel contains:

- 80 sessions;
- 2.470229 source hours;
- 88,767 windows at the primary operating resolution;
- 129 ground-truth speaker-change events.

### 6.2 R5 development pool after R4 exclusion

Every R4 session is excluded from R5 train and dev material. The remaining R3 pool contains:

- 365 sessions;
- 588 anchors;
- 321 positive anchors;
- 267 negative anchors.

Its corpus composition is:

- 429 LibriSpeech-synthetic anchors;
- 118 AliMeeting anchors;
- 41 AMI anchors.

Its language coverage is 470 English and 118 Mandarin anchors. No broader language claim is
permitted.

### 6.3 Cached features

R3 already stores pooled features for 100, 300, and 500 ms contexts. R5 uses only the promoted
300 ms representation unless a data-integrity problem makes it unusable. No context sweep is added.

At 300 ms the cached feature tensors contain 10,214 window coordinates across five candidate
layers or taps per encoder. R5 consumes only the promoted layer or tap.

R4 features were extracted from explicit trailing windows:

```text
waveform[frontier - context : frontier]
```

The cached representation therefore does not use audio after its availability frontier. The encoder
may be internally non-causal within the trailing window, but no future audio beyond the frontier is
available to it.

## 7. Split and Leakage Rules

R5 uses three evidence roles:

```text
R3 non-R4 sources
├── train
└── dev

R4 frozen panel
└── internal continuous evaluation
```

Train and dev are deterministically assigned at the strongest available grouping boundary. Each
LibriSpeech synthetic source manifest is indivisible because its speaker graph is connected;
natural meetings are indivisible meeting blocks. Session, waveform, meeting, and synthetic-parent
identity therefore cannot cross the split. Every corpus must be present in both train and dev.

The target is approximately 80% train and 20% dev by grouped source inventory. Exact counts are
reported after grouping, before training starts. The frozen split contains 430 train anchors
(229 positive, 201 negative) and 158 dev anchors (92 positive, 66 negative), across five train groups
and three dev groups. If the strongest grouping metadata reduces either
split to an unusable class count, stop and report rather than weakening the grouping silently.

All normalization statistics, class weights, PCA, dimensionality reduction, and threshold selection
are fitted on train or dev as explicitly defined. R4 data may not determine model weights, early
stopping, augmentation recipes, or thresholds.

Augmented copies remain in the same split as their parent and are never counted as independent
sources.

## 8. Target Definition

R5 predicts new-speaker onset, not conversational handoff.

| Scenario | R5 SCD target |
| --- | --- |
| A continues as A | Negative |
| A pauses and A resumes | Negative |
| A changes loudness, prosody, channel, or noise condition | Negative |
| A becomes B | Positive at first B activity |
| A becomes silence and then B | Positive at first B activity |
| A becomes A+B | Positive at first B activity |
| A+B becomes B-only | Auxiliary handoff-complete event, not a second new-speaker onset |
| A becomes short B backchannel and then A | Positive at B onset and positive when A returns |

Backchannel onset is a valid SCD event even when it is not a handoff. Handoff persistence remains a
future task.

For frame supervision, the primary positive support begins at the first 100 ms frontier after the
GT onset. A trailing feature ending exactly at GT has not observed any new-speaker audio and cannot
be a positive training target:

```text
GT + 100 ms: positive
GT + 200 ms: positive
GT + 300 ms: positive
GT +   0 ms: negative
before GT:   never positive
```

All other valid frames are negative unless excluded by ambiguous or missing annotation. The label
support is fixed before any R5 result is inspected.

## 9. Required Scoring Corrections Before R5

Only two narrow evaluation corrections are required. They are scientific scoring requirements, not
a harness-refactoring project.

1. A causal true positive requires `emit_time >= GT_time`. A pre-boundary emit is a false event even
   when retrospective localization falls inside a symmetric tolerance window.
2. Accuracy-latency ranking must prefer higher F1 or recall under the stated false-event constraint;
   it must not rank a lower F1 as more efficient merely because it is numerically smaller.

Localization and availability remain separate:

```text
localization error  = localized boundary - GT boundary
availability latency = actual emit frontier - GT boundary
```

The cached R3/R4 encoder features are reused. These corrections must not trigger encoder reruns.

## 10. R5-A: Linear Learnability Screen

### 10.1 Purpose

R5-A tests whether speaker-change information can be extracted with a minimal supervised mapping.
It is the cheapest rejection gate and runs on all four encoders.

### 10.2 Model

```text
z_before = frozen feature at GT - 100 ms
z_after  = frozen feature at GT + 300 ms
        ↓
[abs(z_after - z_before), cosine_distance]
        ↓
LayerNorm
        ↓
Linear
        ↓
P(change)
```

No temporal network, augmentation, or layer sweep is used. A single speaker vector is not a valid
change observation by itself, so the probe consumes the fixed before/after change descriptor above.

### 10.3 Training

- weighted binary cross-entropy;
- AdamW;
- initial learning rate `1e-3`;
- maximum 30 epochs;
- early-stopping patience 5;
- deterministic seeds 0, 1, and 2 for promoted candidates;
- no hyperparameter sweep.

One seed may be used as the first fail-fast pass. The two remaining seeds are run only for candidates
that show non-degenerate train and source-held dev behavior.

### 10.4 Measurements

- ROC-AUC;
- PR-AUC;
- EER where defined;
- train-dev gap;
- corpus and condition breakdown;
- calibration-free score distributions.

R5-A does not select a production threshold.

### 10.5 Promotion

ERes S3 and mHuBERT L1 always enter R5-B0 because they are the R3/R4 compact and SSL leaders.
The third default candidate is UniSpeech L1.

WavLM L3 replaces the third candidate only if it ranks in the top two on both source-held dev
ROC-AUC and PR-AUC. Otherwise WavLM stops after R5-A.

If every encoder is near chance, non-finite, or unstable across grouped splits, stop R5 before a
temporal head is trained.

## 11. R5-B0: Unaugmented Causal Temporal Head

### 11.1 Purpose

R5-B0 isolates the value of learned causal memory. It uses existing unaugmented cached features so
that any improvement over R4 cannot be attributed to new audio or augmentation.

### 11.2 Shared head

```text
frozen encoder vector
        ↓
LayerNorm
        ↓
trainable Linear → 256
        ↓
causal residual Conv1d, kernel 3, dilation 1
        ↓
causal residual Conv1d, kernel 3, dilation 2
        ↓
causal residual Conv1d, kernel 3, dilation 4
        ↓
Linear → P(change)
```

Each convolution uses left-only causal padding, 256 hidden units, residual connection, and dropout
0.1. With one convolution per block and a 100 ms hop, the temporal receptive field is 15 frames, or
1.5 seconds of past feature history. No future feature is available.

The architecture, loss, optimizer, scheduler policy, and stopping rule are identical across
encoders. Only the input adapter dimension differs.

### 11.3 Capacity disclosure

The promoted ERes feature is 10,240-dimensional while each SSL feature is 768-dimensional. A direct
trainable projection to 256 therefore gives ERes a much larger adapter:

```text
ERes adapter: approximately 2.62 million weights
SSL adapter:  approximately 0.20 million weights
```

Adapter parameters and temporal-head parameters are reported separately. The primary result is the
practical end-to-end frozen-representation result, but an ERes victory triggers the equal-capacity
control in Section 13.

### 11.4 Training

- encoder remains in `eval()` and receives no gradient;
- weighted binary cross-entropy;
- AdamW, initial learning rate `1e-3`;
- batch size 16 to 32 sequences, selected only by memory fit;
- maximum 30 epochs;
- early-stopping patience 5 on threshold-independent dev PR-AUC;
- dropout 0.1;
- seeds 0, 1, and 2;
- no broad learning-rate, depth, width, or receptive-field sweep.

A single fallback learning rate of `3e-4` is permitted only if the fixed configuration diverges or
produces non-finite values. It is not used merely to improve a weak result.

### 11.5 Sequence sampling

Boundary-centered trajectories are mixed with negative trajectories. The requested support is
GT-1500 ms through GT+1500 ms at a 100 ms hop, but each sequence is clipped to the actual eligible
audio range. This preserves all 588 non-R4 anchors, including 0.9-second synthetic cases, and yields
variable lengths from 6 to 31 frames. Right padding is masked from both loss and metrics; causal
outputs at valid frames cannot observe that padding. State is reset only at declared sequence starts.

R5-B0 is an inexpensive learnability result. It is not allowed to claim a reliable continuous
false-event rate because the short R3 trajectories do not reproduce long free-running exposure.

## 12. R5-B1: Continuous-Negative and Robustness Arm

### 12.1 Entry condition

Only the best two R5-B0 candidates enter R5-B1. Selection uses the dev accuracy-false-event Pareto
relationship and seed stability, not maximum F1 alone.

If no B0 candidate produces useful non-zero continuous recall without a large false-event increase,
R5 stops and B1 is not run.

### 12.2 Existing-data continuous-negative supplement

The most important B1 supplement is not a new corpus. It is longer no-change exposure sampled from
the existing non-R4 training sources.

Sample 10 to 30 second train-only chunks containing:

- long same-speaker speech;
- pause followed by the same speaker;
- stable speech without a change event;
- available gain, noise, codec, and channel variation;
- natural meeting speech where present.

The supplement is capped at approximately two to five source hours across corpora and speakers. It
must remain smaller than a full-corpus extraction project. Sampling is frozen before B1 metrics are
observed.

These chunks teach the free-running head that most time is negative and make false-events-per-hour a
meaningful training concern.

### 12.3 Targeted waveform augmentation

Waveform augmentation is applied only to train data and only for the two B1 candidates.

Safe global transformations include:

- gain;
- additive noise;
- light reverberation;
- band limitation;
- codec simulation.

The primary hard-negative transformations introduce an acoustic discontinuity inside a
same-speaker region:

```text
A clean → A noisy
A normal → A loud
A wideband → A band-limited
A speech → pause → A speech
```

These remain negative SCD examples.

Global transformations cover a whole sequence. Local transformations occur at a random valid time
that is independent of the GT boundary. Positive and negative parents receive the same transformation
families so the transformation itself cannot identify the label.

The following are excluded from the first B1 recipe:

- large pitch shifting;
- voice conversion;
- cross-speaker feature mixup;
- aggressive speed perturbation;
- synthetic overlap without an unambiguous event label.

Each parent contributes at most one global variant and one local hard-negative variant. The total
training inventory may grow by no more than three times. Augmented copies do not increase the stated
independent source count.

### 12.4 Feature extraction for augmentation

Waveform augmentation requires frozen encoder re-extraction only for the selected B1 train material.
It does not rerun R3, R4, ERes final embedding, or LS-EEND measurements.

The preferred environment is native Linux with a separate ROCm environment. WSL2 ROCm is an
acceptable fallback. Large feature caches should live on the Linux filesystem rather than be trained
directly through `/mnt/c`. The current Windows CPU environment is preserved for R3/R4 reproducibility.

## 13. Equal-Capacity Control

If ERes is the R5-B winner, run one control to test whether its advantage depends on the much larger
supervised input adapter.

```text
ERes 10,240-d
        ↓
train-only unsupervised PCA or fixed projection → 768-d
        ↓
same trainable 768 → 256 adapter used by SSL candidates
        ↓
same causal TCN
```

The dimensionality transform is fitted on train only and frozen thereafter. It does not use R4 or
labels. This is a winner-verification control, not a new sweep. If the control reverses the result,
both practical-capacity and equal-capacity conclusions are reported.

## 14. Continuous Evaluation

Every B0 and B1 candidate is run freely over the unchanged R4 panel using the corrected causal
matcher. Model state follows the same reset rule at every declared source start.

Thresholds and debounce policy are selected on dev and frozen before R4 scoring. R4 is never used to
choose a threshold.

Required metrics are:

- Boundary Precision, Recall, and F1 at 100, 250, 500, 750, 1000, and 1500 ms;
- recall within 500 ms as the primary bounded-recall measure;
- recall within 1000 ms as a secondary late-detection diagnostic;
- missed-change rate;
- raw false-event count;
- false events per source hour;
- availability latency median, p90, and p95;
- signed localization error as a separate diagnostic;
- per-corpus and event-type breakdown where counts permit;
- trainable parameter count;
- head compute time and end-to-end RTF where measurable;
- median and individual-seed results.

Because R4 is only 2.47 hours, zero observed false events does not establish a production rate below
one event per hour. Raw counts and exposure hours must accompany every rate.

`Recall@1000ms` must not replace the low-latency criteria. It distinguishes a complete miss from a
detector that eventually reacts too late. A candidate with weak `Recall@500ms` but strong
`Recall@1000ms` is interpreted as delayed confirmation, not successful low-latency SCD.

R5 reports three product-policy views from the same trained head:

| Profile | Recall horizon | Exploratory false-event budget |
| --- | ---: | ---: |
| Fast | 500 ms | 20 per source hour |
| Balanced | 1000 ms | 5 per source hour |
| Stable | 1500 ms | 1 per source hour |

These profiles expose product choices rather than define production acceptance. Threshold and
confirmation are selected on dev and then applied unchanged to R4. An additional R4-only Pareto
surface may be reported as explicitly exploratory, but it cannot be used as confirmatory threshold
selection. The trained positive support remains GT+100/+200/+300 ms for every profile.

### 14.1 Bounded-lookahead sensitivity

After B0, at most the best two viable candidates receive one separate bounded-lookahead diagnostic:

```text
lookahead = 0, 100, or 300 ms
```

The zero-lookahead result remains the primary causal result. The lookahead arm uses the same frozen
representation, head capacity, split, labels, and dev-selected operating-point procedure. It does
not authorize a new layer, context, architecture, or threshold sweep. A 500 or 1000 ms lookahead is
outside R5 because it conflicts with the target latency; 1000 ms is a scoring tolerance only.

Lookahead is charged to actual availability:

```text
availability latency
    = prediction frontier + lookahead + compute completion - GT boundary
```

Retrospectively localizing the boundary near GT does not remove the lookahead cost. Results report
localization error and availability latency separately. A large gain at 100 ms means that immediate
post-boundary evidence is important; little gain by 300 ms suggests that representation quality,
speaker memory, or supervision is the stronger limitation.

This diagnostic is also distinct from a handoff policy delay. Lookahead lets the detector consume
future audio, while a handoff policy observes causal SCD/OSD outputs and waits for persistence before
declaring a conversational handoff. They must not be reported as the same mechanism.

The comparison table includes:

```text
R4 zero-shot ERes pre-pooling
R4 zero-shot mHuBERT
R4 zero-shot UniSpeech-SAT
R4 zero-shot WavLM
R5-A linear probes
R5-B0 causal heads
R5-B1 augmented causal heads
existing compatible ERes-final contextual row
existing compatible LS-EEND contextual row
```

Existing ERes-final and LS-EEND results are included only where audio, GT, timing, and metric
definitions are compatible. Neither model is rerun.

## 15. Fail-Fast Gates

### 15.1 Stop after R5-A

Stop before B0 if every encoder is near chance, non-finite, or dominated by source leakage or corpus
identity. More data or a larger head is not used to rescue a failed linear learnability screen.

### 15.2 Stop after R5-B0

Stop before B1 if no candidate clearly improves the zero-shot R4 Pareto relationship or if apparent
improvement disappears under source-held dev evaluation and repeated seeds.

As a practical guide, a candidate should show useful non-zero recall at low-single-digit or at least
clearly reduced false-events-per-hour, rather than obtaining F1 only by emitting continuously.

### 15.3 R5 go signal

R5 supports a later public-data learned-head study when at least one candidate:

- clearly Pareto-dominates its zero-shot detector;
- provides non-trivial recall within 500 ms;
- reduces false events from the permissive zero-shot regime toward low single digits per hour;
- has median availability latency near or below 300 ms at a useful operating point;
- behaves consistently across at least two of three seeds;
- does not rely entirely on the larger ERes adapter capacity.

These are exploratory decision criteria, not production acceptance thresholds.

### 15.4 Strong stop signal

Recommend reformulating the problem around explicit speaker state, contrastive supervision, or a
different encoder if all candidates still require tens of false events per hour for modest recall,
or if all improvements vanish on continuous evaluation.

## 16. Candidate Interpretation

| Result | Interpretation and next action |
| --- | --- |
| ERes wins practical and equal-capacity controls | Continue with compact ERes plus a small causal head |
| ERes wins only with the large adapter | Report the capacity dependence before choosing a compact path |
| mHuBERT wins | Treat it as the main teacher or learned-head candidate; plan multilingual public validation |
| UniSpeech-SAT wins | Speaker-aware information is learnable even though zero-shot cosine was weaker |
| WavLM revives in R5-A/B0 | Promote it only on measured evidence, not its expected reputation |
| All linear probes fail | Stop before temporal training and reassess labels/features |
| Linear probes work but TCN fails continuously | Add speaker-state or contrastive temporal formulation rather than a larger head immediately |
| B1 helps substantially | Robust negative exposure is a key requirement for the later study |

## 17. Minimal Reproducibility Artifacts

Each run preserves only what is needed to interpret and reproduce the experiment:

- fixed grouped split manifest;
- model and promoted layer/tap identity;
- feature-cache hashes;
- compact training config;
- augmentation recipe and seed when applicable;
- head-only checkpoint;
- epoch-level train/dev metrics;
- raw continuous predictions and event outputs;
- machine-readable metric summary;
- one accuracy-latency plot and representative timelines;
- runtime environment and hardware;
- Git commit and dirty-state disclosure.

Only four mandatory sanity checks are required before measurement:

1. no source, speaker, waveform, or synthetic parent crosses train and dev;
2. no R4 session enters training or threshold selection;
3. labels never mark a pre-GT frame positive;
4. causal event matching requires the actual emit frontier to be at or after GT.

No broader harness-hardening effort is authorized by this plan.

## 18. Execution Order

```text
1. Apply the two narrow scoring corrections using cached results.
2. Freeze grouped train/dev assignments and report exact counts.
3. Run R5-A on all four promoted representations.
4. Promote the fixed B0 candidate set using the R5-A rule.
5. Run one B0 seed per candidate as the first fail-fast pass.
6. Stop failed candidates; complete seeds 1 and 2 for viable candidates.
7. Run the 0/100/300 ms bounded-lookahead diagnostic on at most two viable B0 candidates.
8. Select at most two B1 candidates using the causal result, not lookahead performance alone.
9. Freeze two-to-five hours of train-only continuous negative material.
10. Freeze one targeted augmentation recipe.
11. Extract augmented frozen features for B1 candidates only.
12. Train and evaluate B1 with the same head and seeds.
13. Run the equal-capacity control only if ERes wins.
14. Produce the internal comparison and go/stop recommendation.
15. Stop before public-data acquisition, encoder fine-tuning, or a larger model.
```

## 19. Expected Cost

R5-A and the first B0 pass use existing cached features and should be CPU-feasible. They are expected
to take minutes to, at most, a small number of hours depending on sequence construction and ERes
feature I/O.

B1 is the only stage expected to benefit materially from ROCm because it adds frozen encoder
extraction for selected waveform variants. It is limited to two encoders and a two-to-five-hour
continuous-negative supplement. The cost must be estimated after the exact B1 selection is frozen;
the experiment must not expand to the full legacy inventory automatically.

## 20. R5 Completion Condition

R5 is complete when the report answers all of the following:

1. Can a linear mapping extract speaker-change information from each frozen representation?
2. Does the same small causal TCN improve the zero-shot continuous Pareto relationship?
3. Which representation is best under practical capacity?
4. Does the result survive an equal-capacity control when ERes wins?
5. How much do long continuous negatives and targeted acoustic augmentation reduce false events?
6. What recall and availability latency remain at low false-event operating points?
7. Are failures complete misses or late detections according to `Recall@500ms` and
   `Recall@1000ms`?
8. How much does 100 or 300 ms of explicitly charged lookahead improve the best causal candidates?
9. Is the result stable across seeds and source-held development data?
10. Is there enough evidence to justify an untouched public-data study with the winning frozen
   encoder and small causal head?

R5 ends with a go/stop decision. It does not automatically authorize public-data acquisition,
encoder fine-tuning, a larger temporal model, or production work.
