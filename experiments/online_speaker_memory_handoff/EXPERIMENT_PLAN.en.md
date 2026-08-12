# R6 Online Speaker Memory and Handoff Experiment Plan

## 1. Document Status and Execution Principle

This document defines the R6 experiment that follows the R3 representation screen, the R4
continuous zero-shot evaluation, and the failed R5 frozen causal SCD-head study.

R6 is an experiment for producing a decision-relevant result. It is not a project for building,
hardening, or generalizing an experiment harness.

R6 therefore prioritizes:

- measurements on existing natural meeting audio;
- raw score trajectories and event-level results;
- direct comparison of speaker representations and memory policies;
- analysis of failures on clean handoff, silence gaps, overlap, and backchannels;
- a clear go, conditional-go, or stop decision.

R6 explicitly does not require:

- unit, integration, contract, or harness test suites;
- reusable experiment-framework abstractions;
- broad schema or configuration refactoring;
- exhaustive grid search;
- production integration;
- encoder training or fine-tuning during the initial experiment.

Experiment scripts may contain only the small runtime checks needed to prevent invalid measurements,
such as sample-rate, tensor-shape, finite-score, timestamp-order, and missing-file checks. Test-suite
coverage and harness polish are not completion criteria.

This plan does not authorize a compute run. Code and documentation are implemented by the
coordinator. Material inference, training, or scoring jobs that consume substantial CPU or GPU time
must be approved by the owner and run by an OpenCode worker through the Orca CLI. Long jobs are
supervised through worker messages and approximately 15-minute event-driven waits rather than
continuous polling.

## 2. Project Reframing

The project is no longer defined primarily as:

> Train a neural model to emit a speaker-change pulse.

The R6 definition is:

> Preserve an explicit representation of the current speaker, decide whether incoming speech is
> from that current speaker or from another speaker, and use persistence and speaker activity to
> determine whether the other speaker actually took the conversational turn.

The proposed system is:

```text
mono mixed audio
       ↓
pretrained speaker or speech encoder
       ↓
speaker evidence
       ↓
┌──────────────────┬──────────────────┐
│ CURRENT memory   │ CANDIDATE memory │
│ current speaker  │ possible other   │
└─────────┬────────┴─────────┬────────┘
          ↓                  ↓
       SAME / OTHER evidence
                 ↓
     persistence / VAD / overlap state
                 ↓
          handoff controller
                 ↓
       continue / candidate / handoff
```

The neural encoder supplies speaker evidence. The controller owns the temporal meaning of that
evidence. Speaker change and handoff are related but separate outputs.

## 3. Why R6 Follows from R5

R5 tested the following formulation:

```text
recent absolute feature sequence
            ↓
small causal TCN
            ↓
speaker-change pulse
```

The aligned R5 probe showed that short frozen representations contained some speaker-discriminative
information, but the learned continuous detector did not transfer to natural meetings. At a
representative fixed operating point, the models recovered almost none of the natural meeting
changes while retaining substantially more synthetic-splice detections.

R5 therefore rejected the immediate use of a larger version of the same absolute-feature causal
head. It did not test or reject a detector supplied with an explicit speaker reference.

R6 isolates the unanswered question:

> If the current speaker is represented correctly, do ERes2NetV2 and mHuBERT features separate that
> speaker from an incoming different speaker in continuous natural meetings?

If the answer is no under oracle enrollment, a controller cannot rescue the representation. If the
answer is yes, online memory and handoff policy can be tested without encoder training.

## 4. Primary Research Questions

R6 must answer the following questions with natural-meeting measurements.

1. Does a fixed, clean CURRENT speaker memory separate SAME from OTHER speech during a continuous
   stream without receiving the ground-truth boundary?
2. Does ERes2NetV2 final embedding provide stronger identity evidence than ERes pre-pooling or
   mHuBERT at the latency it requires?
3. Can CURRENT memory survive silence without causing false speaker changes when VAD prevents memory
   updates?
4. Can a CURRENT/CANDIDATE policy suppress short backchannels without missing sustained handoffs?
5. What happens during mono overlap, where one query representation can contain both speakers?
6. How much performance is lost when moving from fixed oracle enrollment to an online memory policy?
7. Does any zero-training memory configuration provide a useful accuracy-latency-false-event trade-off?
8. Does an existing streaming diarizer already meet the same handoff objective closely enough to
   reduce the value of a custom detector?

## 5. Hypotheses

### H1 — Explicit reference hypothesis

A frozen speaker or speech representation will separate CURRENT from OTHER more reliably when it is
compared with an explicit stable CURRENT reference than when a temporal head must infer identity and
emit a boundary from absolute features alone.

### H2 — Silence persistence hypothesis

Holding CURRENT memory unchanged during non-speech will distinguish `A → silence → A` from
`A → silence → B` more reliably than a fixed short receptive field.

### H3 — Evidence/controller decomposition hypothesis

Separating OTHER evidence from handoff confirmation will preserve sensitivity to a new speaker while
reducing false handoffs caused by short backchannels.

### H4 — Representation trade-off hypothesis

ERes final embedding may provide the strongest speaker identity evidence but require longer query
audio, while mHuBERT or ERes pre-pooling may provide weaker but earlier evidence. No representation
is assumed to win before measurement.

### H5 — Training-optional hypothesis

A zero-training CURRENT/CANDIDATE controller may be sufficient. A learned relational head is needed
only if oracle speaker evidence is useful but fixed cosine and policy rules are insufficient online.

## 6. Scope

### 6.1 Initial R6 scope

The initial authorized experiment scope is:

```text
R6-0  Freeze event definitions and evaluation views
R6-A1 Fixed oracle enrollment, zero training
R6-A2 Oracle-gated streaming memory upper bound
R6-B  Online CURRENT/CANDIDATE controller, zero training
R6-S  Streaming diarization reference when environment-feasible
       ↓
go / conditional-go / stop decision
```

R6-C, a tiny learned relational head, is a conditional later stage. It is not executed unless R6-A
shows useful speaker evidence and R6-B shows a remaining policy gap.

### 6.2 Non-goals

R6 does not include:

- rerunning completed ERes final-embedding or LS-EEND baselines without a new measurement need;
- repeating the R3/R4 broad encoder, layer, context, window, or hop sweeps;
- reviving the R5 absolute-feature causal TCN with a larger head;
- encoder fine-tuning, partial unfreezing, or learned layer fusion;
- Conformer training or another large temporal model;
- knowledge distillation;
- source separation;
- a full multi-speaker identity registry;
- production ONNX, quantization, application integration, or UI work;
- new dataset acquisition during the initial R6 fail-fast stage;
- experiment-harness hardening or formal test-suite construction.

## 7. Existing Evidence and Data Reuse

### 7.1 Primary data

R6 reuses the natural meeting data already used by the representation and continuous SCD studies.
The existing R4 panel contains approximately:

- 2.41 hours of natural meeting audio;
- 86 natural speaker-change events;
- AliMeeting and AMI material;
- an additional synthetic subset with 43 speaker-change events.

The natural subset is the primary R6 evidence. The synthetic subset is secondary diagnostic evidence
and must never dominate the headline result.

This amount of natural data is sufficient to kill or promote the oracle-memory hypothesis. It is not
sufficient for a production-readiness or broad multilingual-generalization claim.

### 7.2 Development and evaluation roles

The evidence roles are:

```text
existing non-R4 natural material
└── threshold and policy development

frozen R4 natural continuous panel
└── decision evaluation

existing synthetic material
└── secondary failure analysis only
```

No threshold, persistence rule, query duration, or memory update rule is selected from the frozen R4
evaluation result. If the existing non-R4 material is too small for a fixed dev split, leave-one-
meeting-out development is allowed, but meeting boundaries must remain intact.

### 7.3 Existing model artifacts

R6 reuses:

- existing mHuBERT R4 continuous features where available;
- existing ERes2NetV2 pre-pooling R4 continuous features where available;
- the existing ERes2NetV2 192-dimensional embedding runtime and checkpoint;
- existing common ground truth and event evaluation code;
- existing ERes and LS-EEND event-level baseline artifacts;
- prior R3/R4/R5 result manifests and error examples.

ERes final-embedding inference is run only where full-stream CURRENT/query scores do not already
exist. This is a new memory measurement, not a repetition of the old adjacent-window baseline.

## 8. Representations Under Test

### 8.1 Primary representations

| ID | Representation | CURRENT enrollment | Query context | Priority |
| --- | --- | --- | --- | --- |
| E-FINAL | ERes2NetV2 final 192-d embedding | 1.5 and 2.0 s stable speech | 0.5, 0.75, and 1.0 s | Primary |
| M-L1 | mHuBERT promoted R4 layer | pooled stable memory | 0.3 and 0.5 s where cached | Primary |
| E-S3 | ERes2NetV2 promoted pre-pooling tap | pooled stable memory | 0.3 and 0.5 s where cached | Primary bridge |

The comparison intentionally allows asymmetric enrollment and query durations. A long stable
CURRENT reference and a shorter query reflect the intended product use.

### 8.2 Deferred representations

UniSpeech-SAT and WavLM are not primary R6 candidates because the earlier screen did not justify a
new extraction campaign. They may be included as scoring-only appendix rows only if the exact
continuous features already exist and adding them does not delay the primary decision.

### 8.3 Score definition

The initial score is:

```text
same_score[t] = cosine(normalize(CURRENT), normalize(query[t]))
other_score[t] = 1 - same_score[t]
```

Each representation retains its own calibrated threshold. Raw cosine thresholds are not compared
across encoders as if they were on a common scale.

## 9. Event Definitions and Multiple Policy Views

R6 does not reduce the result to one tolerance such as Recall@500ms. It reports multiple views
because product policies may prefer an early candidate, a balanced handoff decision, or a highly
stable confirmation.

### 9.1 Ground-truth event views

The shared segment annotation derives:

- `new_speaker_onset`: first activity by a speaker different from CURRENT;
- `overlap_start`: first time CURRENT and the new speaker are active together;
- `exclusive_new_onset`: first time the new speaker is active and CURRENT is inactive;
- `current_returns`: CURRENT resumes after a temporary OTHER speaker;
- `speaker_change`: the common event used by the existing baseline evaluation.

### 9.2 Prediction event views

R6 emits or derives:

- `other_candidate_onset`;
- `overlap_or_mixture_suspected` when the available evidence supports it;
- `handoff_confirmed`;
- `candidate_rejected`;
- `current_promoted`.

### 9.3 Handoff policy views

The same raw predictions are evaluated under several sustained-new-speaker policies rather than
declaring one arbitrary persistence duration to be universally correct:

| View | Required sustained OTHER evidence | Intended interpretation |
| --- | ---: | --- |
| Fast | approximately 300 ms | earliest usable handoff candidate |
| Balanced | approximately 500 ms | moderate latency and backchannel rejection |
| Stable | approximately 1,000 ms | conservative handoff confirmation |

These views do not create different speaker identities or different acoustic ground truth. They are
product-policy interpretations of the same activity and prediction timeline.

### 9.4 Backchannel views

Backchannel behavior is reported under multiple return windows:

```text
CURRENT returns within 500 ms
CURRENT returns within 1,000 ms
CURRENT returns within 1,500 ms
```

This exposes the trade-off between rapid handoff confirmation and rejection of short OTHER speech.

## 10. Evaluation Metrics

### 10.1 Candidate speaker-change metrics

For `other_candidate_onset`:

- precision, recall, and F1 at ±100, ±250, ±500, and ±1,000 ms;
- causal Recall@250, Recall@500, Recall@750, Recall@1,000, and Recall@1,500 ms;
- candidate false events per hour;
- early-alert count before the GT onset;
- median, p90, and p95 detection latency;
- unavailable or abstained event count.

Causal Recall@T counts detections in `[GT, GT + T]`. Predictions before GT are not silently credited
as zero-latency successes; they are reported separately as early alerts or false events according to
the matching policy.

### 10.2 Handoff metrics

For `handoff_confirmed`:

- precision, recall, and F1 at ±250, ±500, ±1,000, and ±1,500 ms;
- Recall@500, Recall@1,000, Recall@1,500, and Recall@2,000 ms;
- false handoffs per hour;
- median, p90, and p95 confirmation latency;
- backchannel rejection rate;
- missed clean, silence-gap, overlap, and return-to-current cases.

Candidate recall and handoff precision are not collapsed into one metric. A short OTHER speaker may
correctly cause a candidate and correctly fail to cause a handoff.

### 10.3 Representation diagnostics

R6-A reports:

- SAME-versus-OTHER ROC-AUC;
- EER;
- same/different distribution overlap;
- session-balanced and frame-pooled variants;
- similarity trajectories around each event;
- recovery after a boundary-straddling or overlap region.

Pair discrimination is diagnostic. The R6 decision is based primarily on continuous event behavior.

### 10.4 Operating-point curves

Each representation and policy reports:

- recall versus candidate false events per hour;
- handoff recall versus false handoffs per hour;
- F1 versus median and p95 latency;
- context duration versus performance;
- fast, balanced, and stable policy points;
- a Pareto frontier rather than one best-F1 row.

Useful false-event reference regions include 1, 5, 10, and 20 events per hour where supported by
the small evaluation duration. Raw TP, FP, FN, and evaluated hours are always reported because rate
estimates are coarse on approximately 2.41 hours of natural audio.

### 10.5 Stratification and uncertainty

All headline metrics are accompanied by:

- natural versus synthetic results;
- AliMeeting versus AMI results;
- per-meeting raw counts;
- clean, silence-gap, overlap, backchannel, and same-speaker-change strata where labels permit;
- meeting-level bootstrap intervals or leave-one-meeting-out sensitivity;
- a check that one meeting does not supply most true positives.

No aggregate score is accepted if natural performance has collapsed while synthetic performance
remains high.

## 11. R6-0 — Minimal Protocol Freeze

R6-0 is documentation and configuration work only. It freezes:

- exact data inventories and evidence roles;
- enrollment eligibility;
- prediction and GT timestamp definitions;
- multiple tolerance and persistence views;
- threshold selection rules;
- the small representation/context matrix;
- output paths and run identifiers.

R6-0 must not become a framework, validator, or test-suite project. Its output is one executable
experiment description and the minimum configuration needed to prevent run-to-run ambiguity.

## 12. R6-A1 — Fixed Oracle Enrollment

### 12.1 Purpose

R6-A1 tests representation capacity without online memory-management failures.

### 12.2 Oracle boundary

Ground truth may be used only to select an initial clean, exclusive CURRENT enrollment region and to
score predictions afterward.

Ground truth must not be used to:

- expose the future speaker-change time;
- freeze memory exactly at a boundary;
- place a fixed-length query only around a known boundary;
- select the threshold for an evaluation meeting;
- suppress difficult same-speaker intervals;
- promote CANDIDATE during evaluation.

The cleanest evaluation unit is a first-handoff-after-enrollment stream:

```text
GT-confirmed clean CURRENT enrollment
                ↓
memory fixed
                ↓
stream forward without boundary information
                ↓
first eligible change or non-change horizon
```

Units must include variable-duration pre-event speech and negative no-change streams. They must not
all be boundary-centered clips.

### 12.3 Fixed-memory construction

For ERes final embedding:

```text
one or more stable enrollment embeddings
                ↓
L2-normalized mean or medoid
                ↓
fixed CURRENT prototype
```

For mHuBERT and ERes pre-pooling:

```text
stable pooled enrollment vectors
                ↓
L2-normalized mean or medoid
                ↓
fixed CURRENT prototype
```

Mean and medoid are the only initial aggregation alternatives. No learned fusion is allowed.

### 12.4 R6-A1 outputs

R6-A1 produces:

- full raw score timelines;
- SAME/OTHER distributions;
- event predictions across the threshold sweep;
- all candidate metrics and policy views;
- natural-only and per-meeting tables;
- timeline plots for representative clean, silence-gap, overlap, backchannel, false-positive, and
  missed-change cases.

## 13. R6-A2 — Oracle-Gated Streaming Memory

### 13.1 Purpose

R6-A2 estimates how well a cautious memory policy could work if speech and overlap gating were
reliable. It is an upper bound, not a deployable detector.

### 13.2 Permitted oracle inputs

R6-A2 may use GT-derived:

- speech versus silence;
- exclusive speech versus overlap;
- memory-update eligibility.

It may not use the exact speaker-change boundary to trigger a candidate, freeze, promotion, or event.
Every permitted oracle input is explicitly listed in the result manifest.

### 13.3 Memory rules

The minimal policies are:

1. cautious normalized EMA on high-confidence SAME speech;
2. a small recent stable-vector bank summarized by a medoid.

Both policies:

- preserve CURRENT through silence;
- freeze CURRENT on low similarity;
- freeze CURRENT during overlap or mixture uncertainty;
- reject low-quality query windows;
- avoid adding CANDIDATE evidence to CURRENT.

The A1-to-A2 gap estimates the value and risk of memory adaptation.

## 14. R6-B — Online CURRENT/CANDIDATE Controller

### 14.1 Purpose

R6-B removes GT gating and measures whether the speaker-memory idea works as an online handoff
detector without training.

### 14.2 State model

The minimum state model is:

```text
CURRENT_STABLE
      ↓ low CURRENT similarity
OTHER_SUSPECTED
      ↓ sustained and self-consistent
HANDOFF_PENDING
      ↓ old speaker absent and candidate persists
CURRENT_PROMOTED

OTHER_SUSPECTED
      ↓ candidate disappears or CURRENT returns
CURRENT_STABLE
```

Overlap or mixture uncertainty freezes memory and may delay promotion. It must not manufacture a
clean CANDIDATE embedding from a mixed mono window.

### 14.3 Evidence available to the policy

R6-B may use:

- CURRENT-to-query cosine;
- score trend;
- CANDIDATE self-similarity;
- candidate duration;
- speech confidence;
- available overlap evidence;
- CURRENT memory confidence;
- time since last stable CURRENT speech.

It may not use GT speaker identity or boundary time.

### 14.4 Memory protection

Promotion requires:

- sustained OTHER evidence;
- sufficient active speech;
- internally consistent candidate vectors;
- no strong evidence that CURRENT has resumed;
- a promotion rollback period.

The rollback path is mandatory because a false promotion can otherwise corrupt every later decision.

### 14.5 Policy matrix

R6-B evaluates only three named policy families:

- fast;
- balanced;
- stable.

Each family is evaluated across the shared threshold/false-event curve and multiple recall windows.
There is no unrestricted Cartesian sweep of every threshold, duration, context, aggregation, and
VAD parameter.

## 15. R6-S — Streaming Diarization Reference

### 15.1 Existing baselines

Existing LS-EEND and legacy ERes baseline predictions are reused under the shared GT and metric
definitions. They are not rerun merely to regenerate an already compatible number.

### 15.2 Streaming Sortformer

Streaming Sortformer is a reference system because its arrival-order speaker cache is conceptually
close to explicit speaker memory while its speaker-activity output can support overlap and handoff
logic.

The first action is an environment and inference smoke, not a porting project. If the existing Linux
environment can run the official checkpoint without material engineering work, its activity output
is converted to the shared event format and evaluated with the same metrics. If NeMo, device, or
ROCm compatibility becomes a substantial task, the limitation is recorded and R6-A/R6-B proceed
without waiting for it.

Sortformer is evaluated by handoff metrics, algorithmic latency, compute latency, false handoffs per
hour, and RTF. Published or locally measured DER is not substituted for the R6 objective.

## 16. Optional R6-C — Tiny Relational Head

R6-C is authorized for planning only. Execution requires a separate owner decision after R6-B.

R6-C is entered only when:

- R6-A shows clear oracle SAME/OTHER separation;
- R6-B retains meaningful recall but cannot reach an acceptable false-event region;
- failure analysis indicates that a small combination of relation and state features could resolve
  the errors.

The input is relational rather than an absolute encoder feature sequence:

```text
CURRENT/query cosine
embedding distance or absolute difference summary
score trend
candidate self-consistency
candidate duration
speech and overlap evidence
memory confidence
        ↓
logistic regression or tiny MLP
        ↓
P(NOT_CURRENT)
```

The encoder remains frozen. A new absolute-feature TCN is not allowed under R6-C.

## 17. Decision Gates

The gates are fail-fast engineering decisions, not universal scientific or product thresholds.

### 17.1 R6-A1 oracle gate

At a plausible single-digit candidate false-event rate on natural meetings:

| Natural Recall@1,000 ms | Decision |
| ---: | --- |
| below approximately 30% | Stop the frozen-memory path for that representation |
| approximately 30–60% | Conditional; inspect context, corpus, and error strata |
| above approximately 60% | Promote to A2/B if performance is not dominated by one meeting |

Recall@500 and Recall@1,500 are reported alongside Recall@1,000. A representation is not rejected
solely because it confirms later than 500 ms, and it is not called low-latency solely because it
eventually succeeds by 1,500 ms.

### 17.2 A1-to-A2 gate

- A1 good and A2 good: memory adaptation is viable.
- A1 good and A2 poor: update/freeze policy is contaminating memory.
- A1 poor: do not spend time on controller tuning for that representation.

### 17.3 A2-to-B gate

- B meets a useful handoff Pareto region: no learning is required.
- B loses recall but preserves oracle separation: improve online gating and state policy.
- B has useful evidence but poor calibration: consider R6-C.
- B collapses on natural meetings as R5 did: stop before encoder training and audit the
  representation/domain assumption.

### 17.4 External-baseline gate

If Streaming Sortformer or the reused LS-EEND baseline already meets the desired handoff accuracy,
latency, false-handoff, and compute region, it becomes the product baseline. A custom speaker-memory
system then continues only if it offers a concrete efficiency, latency, deployment, or multilingual
advantage.

## 18. Compute and Run Restraint

R6 is deliberately small.

The initial compute order is:

1. score cached mHuBERT and ERes pre-pooling features;
2. materialize only the targeted ERes final-embedding contexts that are missing;
3. run R6-A1 and stop for analysis;
4. run A2/B only for promoted representations;
5. attempt Sortformer independently when environment-feasible;
6. do not begin R6-C without a new decision.

Failed configurations are retained in raw results. They are not rerun simply to improve a plot.

Minimum run provenance is:

- git commit and dirty-worktree state;
- model/checkpoint identifier;
- representation layer or tap;
- data inventory identifier;
- sample rate;
- enrollment, query, and hop duration;
- threshold and policy parameters;
- timestamp and latency convention;
- hardware/backend;
- Orca/OpenCode worker job identifier where applicable;
- wall-clock duration and approximate RTF.

This provenance is stored with the result, but it does not require a general provenance framework.

## 19. Required Outputs

Each completed stage produces only the artifacts needed to answer the research question:

```text
results/
├── raw_scores.csv or raw_scores.jsonl
├── events.jsonl
├── metrics.json
├── per_meeting.csv
├── configs_used.json
├── plots/
│   ├── timelines/
│   ├── same_other_distributions/
│   ├── recall_false_event_curves/
│   └── accuracy_latency_curves/
└── REPORT.md
```

The exact filenames may follow existing repository conventions. The required meaning is:

- machine-readable raw scores and events;
- multiple tolerance and persistence views;
- natural/synthetic and per-meeting separation;
- representative success and failure timelines;
- one concise decision report.

No dashboard, database, experiment service, or generalized harness is required.

## 20. Final Comparison Table

The report contains, at minimum:

| Method | Context / policy | Candidate R@500 | Candidate R@1000 | Handoff R@1000 | Handoff R@1500 | Candidate FP/h | False handoff/h | Median / p95 latency | RTF |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ERes final fixed memory | best A1 point | | | | | | | | |
| ERes final online memory | fast / balanced / stable | | | | | | | | |
| mHuBERT fixed memory | best A1 point | | | | | | | | |
| mHuBERT online memory | fast / balanced / stable | | | | | | | | |
| ERes pre-pooling memory | promoted points only | | | | | | | | |
| Existing ERes baseline | reused result | | | | | | | | |
| Existing LS-EEND | reused result | | | | | | | | |
| Streaming Sortformer | if feasible | | | | | | | | |

No method is ranked by one F1 value alone. The report discusses the full accuracy-latency-false-
event trade-off.

## 21. Required Plots and Failure Analysis

The most important plot shares one time axis across:

```text
GT speaker activity
VAD / overlap state
CURRENT-to-query similarity
CANDIDATE self-similarity
memory update/freeze/promotion state
OTHER candidate events
handoff confirmation events
existing LS-EEND or Sortformer activity where available
```

At least one timeline is produced for:

- clean direct handoff;
- silence-gap handoff;
- overlap-to-exclusive handoff;
- short backchannel followed by CURRENT;
- same-speaker acoustic or prosodic change;
- false candidate;
- false promotion or rollback;
- missed handoff.

The report must explain whether each failure came from:

- representation overlap;
- insufficient query speech;
- boundary-straddling mixture;
- overlap;
- VAD/OSD gating;
- CURRENT contamination;
- CANDIDATE inconsistency;
- threshold calibration;
- persistence policy;
- corpus/domain mismatch.

## 22. Completion Criteria

R6 is complete when the experiment can answer:

1. Does explicit oracle CURRENT memory reveal usable SAME/OTHER separation on natural meetings?
2. Which of ERes final, mHuBERT, and ERes pre-pooling provides the best useful accuracy-latency
   region?
3. How much performance is lost when moving from fixed enrollment to online memory?
4. Can online CURRENT/CANDIDATE logic suppress backchannels and preserve handoff recall without
   training?
5. How do clean, silence-gap, overlap, and backchannel cases differ?
6. Does the existing LS-EEND result or an environment-feasible Streaming Sortformer baseline already
   provide a better product path?
7. Is the next action no training, a tiny relational head, representation revision, a different
   encoder, or adoption of an existing diarizer?

R6 is not blocked by the absence of a generalized harness or a formal test suite. It is blocked only
if the existing audio, annotations, or model artifacts cannot support a valid measurement.

## 23. Final Decision Outcomes

R6 ends in one of the following decisions.

### Outcome A — Zero-training memory is sufficient

```text
frozen encoder
→ CURRENT/CANDIDATE memory
→ policy controller
→ product-oriented implementation
```

No SCD head, encoder fine-tuning, or distillation is needed.

### Outcome B — Representation is useful but policy is insufficient

```text
frozen encoder
→ memory relation features
→ tiny relational head
→ handoff controller
```

Proceed to the separately authorized R6-C experiment.

### Outcome C — Frozen representation is insufficient

```text
oracle CURRENT still fails
→ inspect layer/context/reference quality
→ consider another encoder or speaker-aware fine-tuning
```

Do not spend time on a larger controller before resolving the representation failure.

### Outcome D — Existing diarizer already satisfies the objective

```text
LS-EEND or Streaming Sortformer
→ shared handoff controller
→ product baseline
```

Continue custom representation research only for a measured latency, compute, deployment, or
language advantage.

### Outcome E — Accurate but too expensive

Distillation becomes relevant only after a system has demonstrated useful natural-meeting accuracy
and is rejected primarily for runtime cost.

