# R7 ERes Candidate-Gated Relation Verifier Experiment Plan and R7-B Approval Gate

## 1. Document Status and Execution Principle

This document originally defined the next experiment after the completed Phase 5 ERes proposal
study, the failed R5 frozen causal-head study, and the completed R6 fixed-speaker-memory study. R7-A
is now complete. Sections 3 through 13 retain the pre-execution rationale and protocol; Sections 14
through 16 record the observed decision and the revised approval boundary for any R7-B work.

R7-A tested:

> Use the high-recall Phase 5 ERes adjacent-change proposals as candidates, then use no more than
> 1,000 ms of total post-boundary evidence to decide whether each candidate is a real speaker
> change or a same-speaker acoustic change.

The purpose is to obtain the result quickly. Experiment-framework quality is secondary. R7 does
not require a generalized harness, exhaustive validation framework, formal test suite, audit
shards, hash chains, independent recomputation, or production-ready abstractions. Only narrow
checks that prevent an invalid measurement are required.

R7-A was implemented by the coordinator. Material feature extraction, training, and full continuous
scoring were run by an OpenCode worker through the Orca CLI. A first evaluation output exposed a
too-coarse development threshold grid that skipped the required low-false-event operating points.
That output was invalidated, the measurement code was corrected narrowly, and only the affected
development selection and evaluation outputs were rerun. Only the corrected final rerun is an R7-A
result.

R7-B remains unauthorized. The original generic proposal to remove ERes candidate points while
otherwise relying on the same frozen ERes evidence is superseded by the revised gate in Section 15.
No R7-B planning expansion, implementation, feature extraction, training, or scoring may begin
until the owner explicitly approves a revised R7-B scope satisfying that gate.

## 2. Decision Summary

R7-A asks:

> Does the ordered local relationship between ERes embeddings contain enough information to reject
> most Phase 5 false candidates while preserving useful speaker-change recall, without ASR,
> persistent speaker identity, or multi-second enrollment?

The experiment is:

```text
16 kHz mono PCM
      ↓
Phase 5 ERes adjacent candidate generator
500 ms left / 500 ms right, 100 ms hop
      ↓
candidate at source time t, initially available at t + 500 ms
      ↓
ordered ERes relation evidence through at most t + 1,000 ms
      ↓
small relation verifier
      ↓
accept / reject candidate, with boundary timestamp retained at t
```

R7-A ended with Outcome C. It did not enter R7-B automatically.

### 2.1 Observed R7-A result

The corrected final evaluation used five natural meetings totaling 2.410 hours and 2,751 reference
speaker changes. The raw adjacent ERes stream emitted 73,833 candidates, approximately 30,638
candidates per source hour or one candidate every 118 ms. It therefore covered about 85% of the
possible 100 ms scan positions.

| Method | Lookahead | Recall@250 | Recall@500 | Evaluation false events/h |
| --- | ---: | ---: | ---: | ---: |
| Raw ERes adjacent candidates | 500 ms | 0.999 | 1.000 | 29,497.394 |
| Linear relation verifier | 500 ms | 0.067 | 0.079 | 95.027 |
| Linear relation verifier | 750 ms | 0.064 | 0.073 | 75.524 |
| Linear relation verifier | 1,000 ms | 0.081 | 0.092 | 86.313 |
| Small relation MLP | 500 ms | 0.052 | 0.059 | 48.966 |
| Small relation MLP | 750 ms | 0.052 | 0.063 | 46.061 |
| Small relation MLP | 1,000 ms | 0.052 | 0.063 | 47.306 |

Verifier rows use the threshold selected at the development target of 10 false events/hour. Those
development operating points realized only 46.061 to 95.027 false events/hour on evaluation while
retaining 5.2% to 8.1% Recall@250. The raw candidate Recall@250 ceiling was 99.927%, so candidate
misses were not the limiting factor. Increasing lookahead from 500 to 750 or 1,000 ms did not
produce a consistent operational gain.

The final artifacts are stored under:

```text
%SRSCD_CACHE_ROOT%/results/r7/eres_candidate_relation_verifier_v1/
```

## 3. Why R7 Exists

The preceding experiments established three relevant facts.

1. Phase 5 ERes adjacent proposals contained real change information. On the development-known
   natural sample, the 100 ms-hop adjacent profile reached 100% transition recall, but its false
   candidates were far too frequent: approximately 46.8 false events per matched transition at the
   representative high-recall point.
2. R6 ERes final embeddings ranked SAME and OTHER frames well in aggregate, reaching evaluation AUC
   0.964 and EER 0.094, but a fixed CURRENT-to-query threshold had no useful extreme-low-false-event
   operating point. Reaching 60% development Recall@1,000 ms cost at least 618 candidate false
   events per hour.
3. Longer pooled query embeddings did not reliably solve the problem because a trailing query near
   a boundary mixes old-speaker and new-speaker speech into one vector. More audio helps only if its
   temporal relationship is preserved rather than averaged away.

R7 therefore does not repeat R5 or R6:

- it is not an absolute-feature causal pulse head like R5;
- it does not require a CURRENT speaker enrollment or persistent memory like R6;
- it does not classify one pooled 1-second query embedding;
- it preserves the ordered relation pattern around a Phase 5 candidate.

## 4. Research Questions and Hypotheses

### 4.1 Primary question

At a fixed Phase 5 candidate ceiling, can a small learned relation verifier move the natural
continuous recall-versus-false-events curve far enough to make ERes useful as a generic acoustic
speaker-change component?

### 4.2 Secondary questions

1. How much does evidence available at 500, 750, and 1,000 ms after the candidate boundary improve
   the trade-off?
2. Are remaining errors caused mainly by candidate misses, candidate-time misalignment, overlap,
   short backchannels, or indistinguishable same-speaker acoustic changes?
3. Does a linear relation model already work, or is a small nonlinear model necessary?
4. If the verifier remains inadequate, is the failure specifically caused by candidate gating in a
   way that justifies testing fixed-lag local segmentation?

### 4.3 Hypotheses

**H1 — Ordered-relation hypothesis**

A real speaker change should more often form a locally coherent pattern in which embeddings on the
left resemble one another, embeddings on the right resemble one another, and cross-boundary
similarity remains lower. A same-speaker phonetic or prosodic excursion should more often be brief
or internally inconsistent.

**H2 — Useful-lookahead hypothesis**

Preserving 750 to 1,000 ms of ordered post-boundary evidence should reject transient false
candidates better than the 500 ms candidate observation alone. The gain is not assumed; it is
measured at identical 100 ms candidate spacing.

**H3 — Candidate-ceiling hypothesis**

If the relation verifier is precise on candidates but total event recall remains poor, the fixed
Phase 5 candidate generator rather than the verifier is the limiting component. This is the main
condition under which R7-B becomes worth proposing.

## 5. Scope

### 5.1 R7-A included work

R7-A includes only:

- the ERes2NetV2 E-standard final embedding already used by Phase 5;
- the Phase 5 adjacent-direct geometry: 500 ms on each side, evaluated every 100 ms, fixed raw
  change threshold `1 - cosine > 0.50`;
- a candidate-centered sequence of ERes embeddings and audio-derived speech/energy evidence;
- total post-boundary evidence deadlines of 500, 750, and 1,000 ms;
- one linear relation baseline and one small nonlinear relation verifier;
- natural continuous development and evaluation;
- direct comparison with the unverified ERes candidate stream and compatible existing baselines;
- a concise error analysis and a recommendation.

The primary input is 16 kHz mono PCM. No ASR output, text, punctuation, language model, provider
speaker label, speaker name, or persistent speaker identity is permitted.

Audio-derived VAD or speech-fraction features are permitted. Ground-truth speaker activity is used
only for training labels, evaluation labels, and error strata; it is never an inference input.

### 5.2 Explicit non-goals

R7-A does not include:

- Deepgram, Soniox, or any other ASR/provider comparison;
- online speaker enrollment, CURRENT/CANDIDATE memory, or global speaker tracking;
- 1-second pooled query embeddings;
- end-to-end encoder fine-tuning;
- a large Transformer, Conformer, EEND, or diarization model;
- stereo or spatial features;
- synthetic-splice-led training;
- broad encoder, layer, window, hop, architecture, or augmentation sweeps;
- product integration, ONNX export, quantization, UI work, or provider wiring;
- a generalized experiment framework;
- formal unit, integration, contract, or audit test suites;
- R7-B implementation or execution.

## 6. Data and Reuse

### 6.1 Primary natural data roles

R7 reuses the ten natural meetings already frozen by R6:

| Role | Sessions | Use |
| --- | --- | --- |
| Development | the five R6 development meetings | candidate generation, training, cross-meeting model selection, calibration, threshold selection |
| Evaluation | the five R6 evaluation meetings | one frozen continuous evaluation after model and thresholds are selected |

The evaluation meetings have already been observed in R6, so R7 is internal decision evidence, not
an untouched confirmatory or product-readiness result. They remain locked against R7 training,
feature normalization, model selection, and threshold selection.

Meeting-held cross-validation inside the five development meetings is sufficient. A new complex
split framework is not required. The final model is trained on all five development meetings only
after model form and threshold-selection rules are fixed.

### 6.2 Phase 5 reuse

The Phase 5 proposal definitions, cached ERes windows, natural false examples, and scoring utilities
may be reused where their timing and source identity match exactly. Phase 5 natural examples may be
used for qualitative error analysis and development-only hard negatives after excluding any R7
evaluation-session overlap.

Missing dense 500 ms ERes windows are extracted only for the ten selected meetings. No broader
corpus-wide inference campaign is authorized.

### 6.3 Target event

The primary event is `new_speaker_onset`: speech begins from a speaker different from the locally
preceding speaker. This keeps R7 independent of conversational handoff policy.

- `A → B` produces one event at B onset.
- `A → silence → B` produces one event at B onset; `A → silence → A` produces no change event.
- `A → A+B` produces one event at B onset.
- `A → B → A`, including a short backchannel, produces two events.
- a later exclusive-new-speaker onset is an error stratum, not a second primary event for the same
  transition.

Stream start without a preceding speaker is not a change event. R7 predicts event times and
confidence only; it does not output a persistent speaker identity.

### 6.4 Training examples

The training universe consists only of candidates emitted by the fixed adjacent-direct generator.

- Positive candidate: candidate boundary lies within 250 ms of an annotated speaker-change event.
- Negative candidate: candidate lies more than 500 ms from every annotated speaker-change event.
- Ambiguous candidate: candidate lies between those regions or has uncertain annotation; exclude it
  from training but retain it for event-level evaluation where applicable.

Multiple candidates around one ground-truth event are weighted so that one event does not dominate
training. All naturally occurring false proposals in the development meetings are retained as hard
negatives. Random easy negatives are not added unless the candidate stream itself lacks enough
negative exposure.

Synthetic material may be shown only as a secondary diagnostic. It cannot select the model,
threshold, or decision.

## 7. Timing Contract

The candidate source timestamp and the decision availability timestamp are separate.

For a candidate boundary at `t`:

```text
candidate boundary timestamp: t
candidate first availability: t + 500 ms
verifier deadlines:           t + 500, t + 750, t + 1,000 ms
maximum algorithmic delay:    1,000 ms
```

The 1,000 ms budget starts at the candidate boundary, not when the candidate is first emitted.
Therefore this plan does not allow `500 ms candidate delay + 1,000 ms verification delay` to be
reported as a 1-second system. Compute time is measured and added separately.

The 500, 750, and 1,000 ms conditions use the same 100 ms hop. They vary future evidence, not scan
frequency. The 750 ms view uses only grid-aligned embeddings whose audio ends no later than the 750
ms deadline. Boundary localization error and availability latency are reported separately.

## 8. Relation Evidence

The verifier receives a candidate-centered ordered sequence rather than one averaged query vector.

For each candidate, construct only evidence available by the selected deadline:

- 500 ms ERes embeddings sampled at 100 ms steps before and after the candidate;
- the local pairwise cosine-similarity matrix;
- within-left and within-right similarity distributions;
- cross-boundary similarity distributions;
- the adjacent change-score trajectory, local peak shape, and persistence;
- embedding-difference magnitudes and short-term variation;
- audio-derived speech fraction, energy, and silence indicators.

The default past context is 1,000 ms and the maximum future context is 1,000 ms. Past context is
already available and is not charged as latency. The future sequence remains temporally ordered; it
must not be pooled into a single 1-second embedding.

No ground-truth speaker identity, overlap label, event type, or boundary is included in inference
features.

## 9. Models and Training

R7-A deliberately tests only two model forms.

### 9.1 Linear relation baseline

Use logistic regression over normalized relation summaries. This establishes whether the relation
features shift the candidate ranking without nonlinear sequence modeling.

### 9.2 Primary verifier

Use one small two-layer MLP over the flattened masked relation matrix plus the compact relation
summaries. Keep the model below approximately 100,000 trainable parameters. Use early stopping and
at most three seeds.

No architecture search is permitted. Hidden width, dropout, optimizer, and class weighting are
chosen once from a very small smoke/development pass. If both models fail to improve development
candidate ranking, do not add a larger head within R7-A.

Model selection and score calibration use meeting-held development predictions only. Evaluation is
opened once after the model form, lookahead views, score calibration, event suppression, and
operating-point selection rules are frozen.

## 10. Event Construction and Metrics

Verifier scores are converted to events by retaining local score maxima and applying one fixed
short duplicate-suppression radius. The radius exists only to collapse repeated reports of the same
boundary; it must not be tuned into a long refractory period that hides short `A → B → A` changes.

Use one-to-one event matching. Report localization tolerances of 100, 250, and 500 ms, with 250 ms
as the primary view. A prediction cannot be available before the audio required by its selected
lookahead condition.

Required measurements are:

- raw candidate ceiling recall before verification;
- event recall, precision, and F1;
- false events per source hour and raw false-event counts;
- recall at development-selected false-event targets of 1, 5, 10, and 20 per hour when reachable;
- false-event reduction at matched recall relative to the raw ERes candidates;
- recall retained at matched false-event rate;
- availability latency median, p90, and p95;
- signed boundary localization error;
- per-meeting results;
- clean change, silence-gap change, overlap onset, short backchannel/return, and same-speaker false
  candidate strata where annotations permit;
- approximate extraction RTF and verifier compute cost.

The primary result is the continuous natural recall-versus-false-events curve, not candidate-level
classification accuracy or AUC alone.

If the best R7-A events can be passed through the existing Phase 5 product scorer without expanding
the harness, contamination and harmful-split metrics may be reported as a secondary view. This
secondary replay must not delay the main R7 report.

## 11. Required Comparisons

The final report contains at least:

| Method | Lookahead | Recall@250 | Recall@500 | False events/h | Median / p95 availability | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Raw ERes adjacent candidates | 500 ms | | | | | fixed Phase 5 generator |
| Linear relation verifier | 500 ms | | | | | |
| Linear relation verifier | 750 ms | | | | | |
| Linear relation verifier | 1,000 ms | | | | | |
| Small relation MLP | 500 ms | | | | | |
| Small relation MLP | 750 ms | | | | | |
| Small relation MLP | 1,000 ms | | | | | |

Existing Phase 5, R5, and R6 numbers may be shown as contextual rows only when their source exposure
and event semantics are clearly labeled. They must not be presented as numerically interchangeable
when the evaluation units differ.

## 12. Minimal Implementation and Run Order

The implementation should be the shortest code path that produces a trustworthy answer.

```text
1. Inventory reusable ERes windows for the ten selected meetings.
2. Implement the fixed Phase 5 adjacent candidate replay and relation-feature extraction.
3. Run one short-session smoke and inspect a few timestamps and relation matrices.
4. Send missing dense ERes extraction to an OpenCode worker through Orca CLI.
5. Build development candidates and labels.
6. Run the linear baseline and the small MLP with meeting-held development selection.
7. Freeze score calibration, event suppression, and operating-point rules.
8. Run the five evaluation meetings once.
9. Produce metrics, a few representative timelines, and a concise report.
10. Stop and ask the owner for the next decision.
```

Only the following validity checks are required:

1. sample rate, embedding shape, and scores are valid and finite;
2. no evaluation session contributes to training, normalization, calibration, or threshold choice;
3. every feature used by a deadline is available by that deadline;
4. event matching is one-to-one and exposure hours are computed from actual scored audio;
5. at least five accepted and five rejected candidates are manually inspected on a timeline.

Fix a discovered measurement bug narrowly and rerun only affected outputs. Do not build a reusable
validation framework around it.

## 13. Minimal Artifacts

Store only what is needed to reproduce and interpret the result:

```text
%SRSCD_CACHE_ROOT%/results/r7/eres_candidate_relation_verifier_v1/
├── config_used.json
├── inventory.json
├── development_metrics.json
├── evaluation_predictions.jsonl
├── evaluation_metrics.json
├── relation_verifier.pt
├── recall_false_event_curve.png
├── representative_timelines.png
└── REPORT.md
```

Also record the Git commit and dirty-state disclosure, ERes checkpoint identity, meeting lists,
feature timing, hardware, worker job identity, wall-clock time, and approximate RTF. A database,
dashboard, shard format, audit ledger, or generalized artifact service is not required.

## 14. R7-A Interpretation

The following outcomes were declared before execution. R7-A selected **Outcome C**.

### Outcome A — Candidate-gated verification is useful

The verifier clearly moves the locked evaluation Pareto curve and retains useful recall at a much
lower false-event rate. Continue with broader validation or a product-shaped replay. Do not build
R7-B merely because it was predeclared.

A practical indicator is at least a fivefold false-event reduction at matched recall, or a clearly
useful recall improvement at one of the fixed 1/5/10/20 false-events-per-hour views, without the
gain being dominated by one meeting. These are decision guides, not claims of production readiness.

### Outcome B — Relation evidence works but candidate gating is the bottleneck

Recommend R7-B when the verifier separates true and false candidates but total performance is
limited mainly by one or more of:

- missed candidates;
- candidate timestamps that are too far from the true boundary;
- multiple nearby boundaries that cannot be represented by independent point decisions;
- short `A → B → A` patterns that require joint local decoding.

This is the strongest evidence for a 1-second fixed-lag local segmentation experiment.

### Outcome C — Frozen ERes local relation is still insufficient

If the relation models do not materially move the locked continuous curve, report the failure and do
not assume that a larger candidate verifier will rescue it. Fixed-lag segmentation may still be
discussed, but encoder adaptation, different short-window features, overlap modeling, or additional
PCM information may be more justified.

### Outcome D — Inconclusive data or measurement

If candidate coverage or natural exposure is too small to distinguish the outcomes, state the
specific missing evidence. Do not substitute harness work for the missing measurement.

### 14.1 Final R7-A disposition

R7-A rules out candidate coverage as the principal bottleneck on this panel. The raw stream found
99.927% of reference changes within 250 ms because it emitted a candidate at approximately 85% of
all 100 ms scan positions. Removing the candidate gate alone therefore exposes little additional
timeline evidence.

The verifier already received ordered embeddings, pairwise relations, speech and energy evidence,
1,000 ms of past context, and up to 1,000 ms of future context. Its failure does not prove that
joint segmentation must fail, but it materially weakens the expectation that a larger decoder over
the same frozen ERes final embeddings will create a useful low-false-event region.

This result is consistent with the failed R5 frozen causal-head and R6 fixed-speaker-memory results:
three different decision forms found aggregate speaker information but failed in the operational
tail. The current evidence points more strongly to frozen-representation, overlap, and short-window
acoustic limitations than to candidate gating.

## 15. Revised Mandatory User Approval Gate Before R7-B

R7-A completion stops execution. The earlier approval question for a generic candidate-free model
over the same frozen ERes evidence is withdrawn and must not be treated as authorization. This
amendment also does not authorize R7-B.

Any future approval request must name a compact revised R7-B plan and distinguish the following two
roles.

### 15.1 B0 frozen-representation control

B0 is a bounded, development-only falsification control. It may use the same frozen ERes
representation only to measure whether true joint segmentation recovers information that the R5,
R6, and R7-A decision forms missed.

B0 must:

- predict an identity-invariant local speaker partition, pairwise same/different relation, or
  permutation-invariant segmentation rather than another independent boundary pulse;
- derive boundary events from the decoded local partition;
- represent silence, one-speaker speech, and overlap explicitly;
- support multiple boundaries, including short `A → B → A`, within one rolling window;
- retain a maximum 1,000 ms fixed lag and report boundary timestamp separately from availability;
- remain development-only until every gate in Section 15.4 passes.

If B0 fails a development gate, the same-frozen-ERes segmentation path stops. A larger decoder,
longer latency, broader seed sweep, or evaluation run is not an allowed rescue within B0.

### 15.2 B1 representation-revision experiment

B1 is the primary scientifically justified direction after Outcome C. It must change the evidence,
not only the decoder. The approved addendum must select and justify at least one bounded revision:

- an adapter or limited fine-tuning of a speaker-aware encoder;
- a different speaker-aware short-window representation;
- additional short-time PCM evidence such as log-mel, pitch, energy, or speech activity;
- an explicit overlap objective or overlap-aware auxiliary head.

B1 must use the same identity-invariant local segmentation semantics required for B0. It may include
the frozen ERes B0 result as a control, but B0 success is not assumed and B1 may not be described as
merely removing candidate gating.

### 15.3 Evidence roles and evaluation boundary

All ten meetings used by R6 and R7-A are development-known for a revised R7-B because their results
have already influenced the experiment design. They may support internal development and
meeting-held fail-fast decisions, but they cannot support an untouched confirmatory claim.

A revised R7-B plan must choose one evidence mode before training:

1. **Fast internal decision:** reuse the existing meetings and label every result internal and
   development-known; or
2. **Promotion-capable evaluation:** freeze new untouched natural meetings before model or threshold
   selection and open them only after the complete development gate passes.

### 15.4 Mandatory development gates

The following minimum gates replace an expectation of incremental improvement:

- aggregate Recall@250 must be at least 30% at no more than 10 false events/hour;
- aggregate Recall@250 must be at least 50% at no more than 20 false events/hour;
- improvement must appear in at least four of five meeting-held development folds;
- no held-out fold may exceed twice the selected 10 or 20 false-events/hour target;
- overlap-onset and silence-gap-change recall must both be non-zero;
- no one meeting may contribute more than half of all matched true positives;
- duplicate suppression must preserve valid short `A → B → A` transitions.

Failure of either aggregate recall gate stops that candidate before evaluation. Passing the gates
authorizes evaluation only when the approved addendum has already named the evaluation panel,
metrics, model form, representation revision, and stopping rule.

### 15.5 Approval wording

The owner must explicitly approve one of the following scopes:

> Approve planning and running revised R7-B0 as a development-only frozen-ERes joint-segmentation
> control under the Section 15 gates.

or:

> Approve a compact R7-B addendum covering B0 and a named representation-revision B1, with B1
> evaluation allowed only after the Section 15 development gates pass.

An affirmative answer must identify the approved scope. Silence, an ambiguous request to continue,
the existence of this amendment, unused compute budget, or an available worker does not authorize
R7-B.

Before any approved R7-B compute run, the compact addendum must define the representation choice,
rolling-window output semantics, permutation or identity handling, overlap state, multiple-boundary
handling, training target, fixed-lag timing, development folds, evidence mode, and comparison against
R7-A. Code remains coordinator-owned; material feature extraction, training, and full continuous
scoring remain OpenCode-worker jobs through the Orca CLI.

## 16. Completion Record

R7-A is complete. Its answers are:

1. The fixed Phase 5 generator reached 99.927% Recall@250 but emitted 29,497.394 false events/hour.
2. Ordered local relation evidence did not retain useful recall while removing false candidates.
3. The 750 and 1,000 ms views provided no consistent operational improvement over 500 ms.
4. Development-selected 10 false-events/hour thresholds retained only 5.2% to 8.1% evaluation
   Recall@250 while realizing 46.061 to 95.027 false events/hour.
5. Candidate coverage was not the bottleneck. Same-speaker false candidates, overlap onsets,
   silence-gap changes, and weak low-tail generalization remained.
6. The original same-frozen-ERes candidate-free R7-B should not be approved. Only the revised B0/B1
   scopes and fail-fast conditions in Section 15 are eligible for a future explicit approval.

R7-A completion does not require a polished harness, comprehensive tests, production integration,
or R7-B preparation. No production module is changed by this experiment plan, and no architecture
drift is expected while implementation and results remain under `experiments/` and the external
research cache.
