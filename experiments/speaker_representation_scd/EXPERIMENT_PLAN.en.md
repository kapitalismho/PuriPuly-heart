# Low-Latency Speaker Representation to SCD Experiment Plan

## Document Status

- Document version: `0.4-r0-protocol-candidate`
- Date: `2026-08-10`
- Experiment ID: `speaker_representation_scd_v1`
- Status: owner-amended R0 protocol candidate; review acceptance is recorded separately
- Core scope: frozen-representation comparison, continuous zero-shot SCD, and a public-data zero-shot confirmatory report
- Conditional follow-up scope: any learned SCD head, partial fine-tuning, handoff extension, and teacher-to-student distillation

This document does not propose modifying the existing `experiments/speaker_turn_boundary/`
experiment. It uses the existing experiment's data and validated evaluation concepts as prior
assets, while defining an independent experiment with its own research questions, model set,
configuration freeze, result directory, test split, and conclusions.

This study uses only existing or public data. It requires no participant recruitment or private
recording. No model downloads, large-scale feature extraction, or confirmatory evaluation shall
be performed before R0 approval. No model or detector training is authorized by the core study.

---

## 1. One-Sentence Objective

> Compare how reliably mHuBERT-147, WavLM Base+, UniSpeech-SAT Base+, and
> ERes2NetV2 pre-pooling representations provide speaker-change information from short,
> causal observations of the same 16 kHz speech using frozen encoders and zero-shot detector
> logic, with development on reused common-GT data and confirmation on sealed public data.

The final result must answer the following questions rather than merely rank encoders.

1. Which representation responds fastest to 100–500 ms of new-speaker evidence?
2. Which representation is most stable under phonetic, prosodic, loudness, language, and
   environmental changes from the same speaker?
3. Does candidate-conditioned zero-shot separation transfer to continuous free-running SCD?
4. Considering accuracy, latency, false switches, and compute together, which encoder warrants
   a separately approved learned-head follow-up?
5. Is there sufficient zero-shot evidence to extend SCD into an actual handoff detector by adding overlap
   and persistence?

---

## 2. Final Decisions Produced by the Experiment

The final decision is not a single highest-scoring model. The experiment shall identify the
following candidates separately.

| Candidate | Selection criteria |
| --- | --- |
| Representation winner | Short context, nuisance robustness, and layer stability |
| Zero-shot event winner | Confirmatory event performance under identical detector logic |
| Multilingual evidence leader | Worst-group and language-shift stability over the public-language coverage actually obtained |
| Efficient backbone winner | Accuracy-latency-compute Pareto frontier |
| Learned-head follow-up candidate | Provides enough zero-shot signal to justify a separately approved training study |
| Research teacher candidate | Provides the most useful research signal regardless of current product eligibility |

Different models may occupy different roles. Product-backbone or deployable-teacher status is not
decided in this pre-training study.

---

## 3. Experiment Scope

### 3.1 Core study

The core study that must be completed under this document is:

```text
R0  Protocol, model, data, split freeze
R1  Model extraction and causal-timing parity
R2  Dataset and ground-truth materialization
R3  Frozen zero-shot representation probe
R4  Frozen continuous zero-shot SCD
R6-Z  Locked public-data zero-shot confirmatory comparison and final report
```

R5 is intentionally omitted from the authorized sequence because it contains learned probes and
heads. The core study is complete only when the R6-Z report and reproducible raw artifacts have been
produced.

### 3.2 Conditional follow-up

The following phases shall be executed only after the R6-Z result passes its gate and receives
separate approval.

```text
R5   Frozen encoder + identical small causal SCD head
R6-T Locked confirmatory comparison of learned heads on a new untouched partition
R7   Top-layer partial fine-tuning
R8   VAD + OSD + SCD multi-task and handoff state
R9   Teacher-to-causal-student distillation and deployment study
```

### 3.3 Excluded from the core study

- Full encoder fine-tuning from the outset
- Any learned linear probe, layer fusion, or SCD head in the current execution
- Applying downstream heads of different sizes to the four encoders
- Reporting full-context features as streaming results
- Selecting thresholds, layers, pooling, or head architecture on the test set
- Treating an SCD event directly as conversational-handoff ground truth
- Converting LS-EEND output into an artificial cosine representation
- Modifying the product runtime or application composition
- Storing model or corpus binaries in Git

---

## 4. Relationship to the Existing Experiment

### 4.1 Reusable prior assets

The following items from `experiments/speaker_turn_boundary/` may be reused after validation.

- The canonical 16 kHz mono source timeline
- Separation of `boundary_source_sample` and `observed_source_sample_at_emit`
- Active-speaker-set annotation and transition-classification concepts
- LibriSpeech synthetic, AMI, and AliMeeting manifests and WAV hashes
- Speaker/session-disjoint validation methods
- Deterministic manifests, canonical JSON, and SHA-256 provenance practices
- One-to-one event-matching concepts
- Separation of localization error and causal availability latency
- Block bootstrap at the source-session and related-synthetic-derivative level
- Calculation of false events using source/session hours as the denominator
- Boundary trajectory, overlap, backchannel, and stress-condition classifications
- The exact audio/GT intersection already used by ERes and LS-EEND, as the primary
  `development-known` paired-comparison set

When an asset is reused, the new experiment manifest shall record the original manifest ID,
canonical content hash, and WAV hash again. Merely referring to an existing path is insufficient
to claim that the data are identical.

### 4.2 Items that shall not be reused

- Existing ERes/LS-EEND thresholds and selected profiles
- Existing Phase 3/4 shortlists and go/stop conclusions
- Existing detector-specific reducers and state machines
- Statistics that directly combine existing Phase 4 raw scores with new encoder scores
- Caches with a different feature-extraction identity
- Any data whose results or labels have already been inspected as a new confirmatory test set

All data observed in the existing experiment shall begin as `development-known` in the new
study. They are valid for paired representation comparison, detector development, error analysis,
and operating-point selection, but not for a new confirmatory claim. New confirmatory claims
require a separately sealed public-corpus speaker/session-disjoint test partition.

### 4.3 Operational separation

The new experiment has its own:

- package: `experiments/speaker_representation_scd/`
- schema namespace: `experiments.speaker_representation_scd.*`
- result root: `experiments/speaker_representation_scd/results/`
- external cache root: an explicit `SRSCD_CACHE_ROOT`
- model registry and run contract
- development/confirmatory split ledger
- held-out access ledger

The new experiment shall not compete for the same CPU/GPU or cache disk while the existing
`speaker_turn_boundary` experiment is running. Its runner shall not modify files, execution
state, caches, or results belonging to the existing experiment.

---

## 5. Research Questions and Falsification Criteria

### H1. Frame-level pretrained-representation hypothesis

A pretrained temporal representation produced from a short trailing window provides a better
speaker-change signal at low observation delay than the ERes2NetV2 final utterance embedding.

Falsification criteria:

- Every SSL representation has recall/latency no better than the ERes final embedding at the
  same false-event budget across the development and public confirmatory panels.

### H2. Speaker-aware SSL hypothesis

WavLM or UniSpeech-SAT representations are more useful than generic multilingual mHuBERT for
same/different-speaker separation or continuous SCD over short intervals.

Falsification criterion:

- With layer, pooling, and detector conditions frozen on development data, the public
  confirmatory paired difference is consistently indistinguishable from zero or favors mHuBERT.

### H3. ERes pre-pooling hypothesis

The features before statistics pooling in ERes2NetV2 contain information useful for frame-local
speaker-change detection.

Falsification criteria:

- The pre-pooling tap can reconstruct the official embedding, yet
- it does not improve on acoustic controls or the ERes final embedding in either the
  representation probe or continuous zero-shot evaluation.

### H4. Small-head sample-efficiency hypothesis — deferred

Given a good frozen representation, a small causal temporal head can learn speaker-change events
with limited labeled data.

This hypothesis is retained for a separately approved follow-up and is not evaluated in the
current pre-training study.

Falsification criteria:

- The nested data-budget learning curve does not meaningfully exceed the acoustic/ERes baseline,
  or
- held-out performance does not emerge without full encoder fine-tuning.

### H5. Handoff-decomposition hypothesis — deferred

Actual handoff decisions require new-speaker onset plus overlap, old-speaker offset, and
new-speaker persistence.

This hypothesis is retained for a separately approved follow-up and is not evaluated in the
current pre-training study.

Falsification criterion:

- An SCD-only event stream reliably separates backchannels from sustained handoffs without any
  additional information. This outcome is considered unlikely but shall be determined by the
  evidence.

---

## 6. Compared Systems

### 6.1 Primary encoders

| Encoder | Initial model ID | Role | Initial layer/tap candidates | Caveat |
| --- | --- | --- | --- | --- |
| mHuBERT-147 | `utter-project/mHuBERT-147` | Multilingual general SSL | L1, L3, L6, L9, L12 | Validate research license and multilingual strata |
| WavLM Base+ | `microsoft/wavlm-base-plus` | Speaker-friendly general SSL | L1, L3, L6, L9, L12 | Do not assume cross-language transfer from English-centric pretraining |
| UniSpeech-SAT Base+ | `microsoft/unispeech-sat-base-plus` | Explicitly speaker-aware SSL | L1, L3, L6, L9, L12 | Test whether the mixture objective exposes or suppresses overlap onset |
| ERes2NetV2 pre-pooling | `iic/speech_eres2netv2_sv_zh-cn_16k-common` | Compact speaker-specialized network | Stage taps + final fused pre-pooling | Prove temporal receptive field and tap parity first |

Before execution, the model registry shall freeze each artifact's repository revision, config,
processor, checkpoint byte SHA-256, parameter count, license, and source URL. `main` or another
mutable tag shall not be used as an execution identity.

The `CC-BY-NC-SA-4.0` terms in the mHuBERT-147 model card shall be considered separately from
commercial product candidacy. Use as a research teacher and the deployability of a derived student
shall not be assumed to be the same legal question; each requires a separate legal/license gate.

The wide ERes2NetV2 checkpoint shall not enter the primary grid. It may be added as a model-size
sensitivity only after the standard checkpoint's pre-pooling hypothesis is supported.

### 6.2 Controls and contextual baselines

| Control | Purpose |
| --- | --- |
| ERes2NetV2 final 192-d embedding | Positive baseline connecting the study to the current sliding-embedding approach |
| Log-mel/MFCC change | Test whether a neural representation is detecting a simple spectral splice |
| RMS energy, pitch, spectral flux | Control for loudness, pitch, codec, and noise artifacts |
| Random or time-shuffled feature | Sanity check for metric or label leakage |
| Legacy LS-EEND event result | Development-known contextual event baseline on the exact common-GT subset; excluded from representation ranking and confirmatory claims |

LS-EEND shall not be included in raw-feature AUC/EER comparisons. Existing results may appear only
in a development-known contextual comparison of Boundary F1, availability latency, and false
events/hour against the same GT events. After one LS-EEND configuration is frozen without D5
access, a new one-shot run of that fixed configuration may enter the R6-Z natural-event
confirmatory table; it remains excluded from representation ranking.

---

## 7. Exact Meaning of the Comparison

This experiment does not establish the causal effect of a pretraining philosophy. The models
differ in training data, objective, frontend, architecture, and license. The permitted conclusion
is therefore:

> Given the selected public checkpoints, their official preprocessing, and a fixed causal
> observation contract, which released representation system was more useful on these data?

The following claims are not permitted:

- Claiming that a speaker-aware objective alone caused a performance increase
- Claiming a fully controlled architectural comparison merely because model sizes are similar
- Claiming KO/JA/ZH multilingual performance from English-corpus results alone
- Reporting offline full-context results as actual streaming latency

### 7.1 Boundary of the research contribution

Prior work already exists on `SSL feature + SCD head`, layer weighting, and SCD distillation.
Therefore, the contribution of this study is not simply attaching a classifier to WavLM or
HuBERT. Meaningful contribution candidates are the following combination:

- Strict control of a 100–500 ms causal observation budget
- Multilingual KO/JA/EN/ZH and code-switch nuisance evaluation
- Identical zero-shot detector comparison across three SSL families and ERes2NetV2 pre-pooling
- Separation of boundary-straddling, overlap, backchannel, and handoff events
- Pareto analysis of availability latency, false events, and compute together
- An evidence-based decision on whether a learned causal-head study is warranted

The final report shall claim only contributions actually supported by the evidence.

---

## 8. Canonical Audio and Timing Contract

### 8.1 Audio domain

- Canonical waveform: 16 kHz, mono, source-timeline PCM
- Source unit: integer sample index
- Milliseconds: `sample_index / 16`
- If resampling is necessary, perform it once and record the resampler identity and output hash
- Do not apply arbitrary per-window peak normalization beyond encoder-specific official normalization
- Do not treat padding samples as observed audio

### 8.2 Event times

Every prediction shall contain at least three time values.

```text
boundary_source_sample
    The model's best estimate of the actual change position

observed_source_sample_at_emit
    The last source sample read when the prediction became available

compute_completed_monotonic_ns
    The wall-clock position at which computation and queueing actually completed
```

Derived quantities are:

```text
localization_error_ms
  = (predicted_boundary - gt_boundary) / 16

availability_latency_ms
  = (observed_frontier - gt_boundary) / 16

compute_latency_ms
  = compute completion - inference submission

end_to_end_latency_ms
  = availability latency + compute/queueing latency
```

Primary causal results enforce `observed_frontier >= gt_boundary`. A retrospective boundary
estimate may produce negative localization error, but availability latency cannot be negative.

### 8.3 Context modes

The following three modes shall never be mixed in reporting.

#### A. `local_trailing_window` — primary low-latency mode

At observation frontier `t`, provide exactly the `[t-P, t)` waveform to the encoder. `P` is one
of 100, 200, 300, 500, 750, and 1000 ms. Mean-pool all valid output frames into one vector.

```text
[ observed past P ms ][frontier t]
          ↓
       encoder
          ↓
   valid frame mean
```

Even for a standard bidirectional Transformer, this invocation is causal on the source timeline
because the input contains no waveform after `t`. The report shall still record that frames
within the window attend bidirectionally to each other and that every hop requires recomputation.

#### B. `left_context_tail_pool` — secondary context study

Provide `[t-C, t)` as input but pool only valid frames from the final `P` ms. After top
configurations are selected, use 2000 ms as the primary `C`, with 1000/4000 ms sensitivities only
if necessary. Past context does not add algorithmic lookahead, but it does add compute and memory.

#### C. `offline_full_context` — diagnostic upper bound

Extract hidden states from an entire recording. Because audio after a boundary can affect earlier
frames, this mode is restricted to representation interpretation and visualization. Its results
cannot enter streaming-latency tables or final winner selection.

### 8.4 Hop and observation schedule

- Candidate-conditioned probe: exact frontier defined by a GT or pseudo-boundary
- Continuous primary hop: 50 ms
- Continuous sensitivity: 20 ms and 100 ms
- Validate model frame timestamps with input-length/output-length experiments and
  source-coordinate fixtures rather than trusting nominal config stride alone
- Record output-frame center and required input frontier in the registry

---

## 9. Ground-Truth Taxonomy

### 9.1 Acoustic active-speaker events

Primary ground truth is generated from the time-varying active-speaker set.

| Event | Active-set example | Meaning | Core SCD positive |
| --- | --- | --- | --- |
| `initial_speech_start` | `{}` → `{A}` | First speaker starts in an epoch | No |
| `same_speaker_resume` | `{A}` → `{}` → `{A}` | Same speaker resumes after a pause | No |
| `new_speaker_onset_clean` | `{A}` → `{B}` | Clean change | Yes |
| `new_speaker_onset_gap` | `{A}` → `{}` → `{B}` | Different speaker after a gap | Yes, at B onset |
| `overlap_start_new_speaker` | `{A}` → `{A,B}` | First appearance of B | Yes |
| `exclusive_new_speaker` | `{A,B}` → `{B}` | Existing A disappears and only B remains | Separate target |
| `backchannel_onset` | `{A}` → `{A,B}` or `{B}` | Brief appearance of B | SCD yes, handoff no |
| `acoustic_nuisance` | `{A}` → `{A}` | Loud/whisper/language/noise change | No |

The primary target for core phases R3–R4 and R6-Z is `new_speaker_onset_*`.
`exclusive_new_speaker` is reported with a separate latency and is not mixed into the primary SCD
positive class.

### 9.2 Conversational handoff

`handoff_complete` is not defined entirely from the active-speaker set. The study distinguishes:

1. `acoustic_exclusive_new_speaker`: start of a B-only interval
2. `conversational_handoff`: an event annotated as B actually taking the conversational floor

No performance claim about the second target is permitted before R8. In R8, an episode annotation
guideline covering overlap, old-speaker offset, B persistence, and backchannel outcome shall be
frozen separately, and a subset shall be independently annotated and adjudicated.

---

## 10. Dataset Plan

### 10.1 Dataset tiers

| Tier | Composition | Role |
| --- | --- | --- |
| D0 deterministic fixtures | Silence, one speaker, clean A→B, gap, overlap, backchannel, gain/noise | Code, timing, and causality validation |
| D1 controlled synthetic speech | Existing LibriSpeech-based same/different/gap/overlap/stress cases | Matched representation probe |
| D2 natural English conversation | AMI and approved meeting/conversation sources | Natural overlap and turn-taking |
| D3 natural Mandarin conversation | AliMeeting and approved sources | Mandarin far-field and overlap |
| D4 public multilingual development | Approved public speech/conversation corpora with usable speaker/session IDs | Publicly obtainable language, language-shift, and nuisance coverage |
| D5 untouched public confirmatory | A separately sealed public corpus or public speaker/session partition | One-time R6-Z evaluation |

All existing D1–D3 data and any previously inspected public data begin as `development-known`.
A legacy held-out name does not guarantee confirmatory status in the new experiment.

### 10.2 Public-only multilingual and confirmatory acquisition

No participant recruitment or private recording is required or authorized. D4 and D5 shall be
obtained only from public corpora whose terms permit the intended research use.

- Reuse the exact ERes/LS-EEND common-GT intersection as development-known data.
- Select D4 public multilingual development sources through a metadata/license/coverage audit.
- Reserve D5 before score extraction as either a never-inspected public corpus or a public
  speaker/session partition whose waveforms, labels, and aggregate outcomes remain unopened.
- Keep all sessions, transformations, and language samples from one known person in one split.
- Record corpus release, download URL, file hashes, license/terms, speaker/session keys, and all
  exclusions in the dataset registry.
- Do not infer sensitive speaker attributes from voice; use only corpus-provided metadata whose
  research use is permitted.

Public availability may not provide balanced KO/JA/EN/ZH coverage or genuine same-speaker
code-switch examples. The report shall restrict multilingual claims to the languages and scenario
strata actually supported; missing KO/JA/EN/ZH strata are limitations, not imputed results.

The corrected dataset decision is frozen in `R0_DATASET_DECISION.md`, SHA-256
`777fea6786e823601f0425b98c7c7fa52a844648b0e39ae1165683150c209308`:

- Development uses the exact legacy common-GT manifest plus 20 hash-selected Zeroth-Korean train
  speakers and 20 fixed JVS development speakers.
- All AMI material is development-known because tracked legacy coverage artifacts already expose
  annotation hashes and derived targets for the proposed meetings.
- Natural in-the-wild D5 uses the complete official VoxConverse v0.3 test partition at repository
  revision `24bf60be297701cd7e4ef18550c6d390c1b87365`; acquisition remains fail-closed until official or
  parity-proven audio bytes are available.
- Natural Mandarin D5 uses the first eight AISHELL-4 official test sessions under the frozen
  `(SHA-256(session_id), session_id)` order.
- Controlled Korean D5 uses every official Zeroth-Korean test speaker.
- Controlled Japanese D5 uses 20 fixed hash-reserved JVS speakers.
- Natural event claims are primary for VoxConverse/AISHELL-4; an English-only VoxConverse claim is
  conditional on a post-lock metadata audit, while Zeroth/JVS support controlled read-speech
  representation and synthetic-boundary claims only.

### 10.3 Required scenarios

Each scenario has an explicit positive or hard-negative role.

```text
A continuous A                  hard negative
A pause A                       hard negative
A → B                           positive clean onset
A → silence → B                 positive gap onset
A → A+B → B                     overlap onset + exclusive B
A → short B → A                 SCD positive, handoff negative
A normal → A loud               nuisance negative
A normal → A whisper            nuisance negative
A language 1 → A language 2     code-switch negative
A clean → A noisy/channel shift nuisance negative
similar-voice A → B              hard positive
```

Synthetic splices shall not be allowed to determine performance by themselves. Results from
synthetic and natural public sources shall always be reported separately. Scenarios unavailable
in public data shall be marked missing rather than synthesized into a confirmatory claim.

### 10.4 Split contract

The following partitions shall be frozen when the data are first materialized.

```text
development
  layer, pooling, prototype, threshold, and hysteresis selection

confirmatory_test
  sealed public speakers/sessions opened once after the zero-shot configuration hash is frozen

future_train
  not materialized or used under the current authorization
```

Required invariants:

- Speaker-disjoint
- Source-session-disjoint
- Transformation-family-disjoint
- All synthetic cases derived from the same original remain in one split
- A recurring participant appearing across corpora/sessions belongs to one connected block
- Confirmatory waveforms, labels, and aggregate statistics are not read before configuration selection;
  metadata needed to prove identity, license, and split eligibility may be audited without scoring
- Test access is recorded in a ledger with command, timestamp, and config hash

### 10.5 Annotation quality

- Canonical annotations store sample-level speaker segments and active sets
- Record boundary-annotator disagreement and preserve ambiguity intervals
- Do not force an ambiguous interval into one point; exclude it from primary exact-boundary
  evaluation or use acceptable-interval matching
- Audit a predefined sample of public annotations and record corpus-specific label uncertainty;
  do not silently treat weak diarization labels as sample-accurate boundaries
- Use a separate guideline and adjudication process for conversational-handoff labels in R8

---

## 11. Pair and Episode Construction

### 11.1 Homogeneous speaker-pair probe

Pairs used to test speaker information itself do not assume a boundary.

- `same_near`: same session, different non-overlapping speech windows
- `same_far`: same speaker, different utterance/session
- `different_matched`: different speakers, with language/channel/SNR/duration matched where possible
- `different_similar_voice`: hard pair using available and consented metadata
- `same_nuisance`: before and after loud/whisper/code-switch/noise variation from the same speaker

Freeze per-speaker and per-session quotas so that one speaker or long session cannot dominate the
pair count. Do not treat millions of frame pairs as independent samples.

### 11.2 Boundary-conditioned probe

Each positive boundary shall be paired one-to-one with a negative pseudo-boundary under the same
corpus/language/stress/context conditions.

Negative candidates:

- Same-speaker pause/resume
- Same-speaker nuisance transition
- A fixed grid point inside a sufficiently stable single-speaker region

Matching shall be frozen from metadata before scores are observed. A negative may be used only
once. Unmatched samples remain in distribution and missingness reports but do not enter the primary
paired comparison.

### 11.3 Boundary trajectory

At minimum, store the following relative-time grid for every event.

```text
-1000, -750, -500, -300, -200, -100,
0, +100, +200, +300, +500, +750, +1000, +1500, +2000 ms
```

Each row records whether its window is:

- Entirely old speaker
- Boundary-straddling
- Overlap
- Entirely new speaker
- Silence-containing
- Ambiguous

These rows support analysis of response onset, minimum similarity, recovery time, and prototype
contamination.

---

## 12. Feature-Extraction Contract

### 12.1 Common rules

- Use `eval()` and gradient-disabled inference
- Use FP32 as the primary numerical mode
- Record deterministic seeds and deterministic-kernel state
- Use the official processor/frontend
- Provide the same canonical waveform range to every encoder
- Batch only windows of identical length
- Declare attention masks and valid-frame masks explicitly
- Do not pool padded values as speech or silence evidence
- Record NaN, zero norm, and too-short input as structured missing reasons rather than hiding them

### 12.2 SSL layers

The R3 single-layer screen uses L1, L3, L6, L9, and L12. Layer numbering shall be defined in the
registry as Transformer-block outputs. The convolutional feature projection is permitted only as
a separate `L0` sensitivity.

R3 does not learn layer weights from labels. Learned weighted layer fusion belongs to the R5
supervised condition.

### 12.3 ERes pre-pooling taps

Before implementation, audit the official source graph and freeze:

- Tap module name and source revision
- Output shape and time stride
- Left and right receptive fields
- Whether the fused feature is exactly the tensor entering statistics pooling
- Official frontend and normalization

Required parity test:

```text
official path final embedding
vs
captured pre-pooling tensor → official pooling/head reconstruction
```

The outputs must agree within an allowed tolerance across multiple durations and stress fixtures.
A tap that does not pass this test cannot be evaluated under the name `ERes pre-pooling`.

### 12.4 Pooling

The R3 primary pooling operation is a simple mean over valid temporal frames.

```text
100, 200, 300, 500, 750, 1000 ms
```

L2-normalize the vector after pooling and before cosine calculation. Standard-deviation
concatenation, attention pooling, and supervised projection are excluded from the R3 primary
condition.

### 12.5 Cache policy

Do not retain full-session frame tensors from every layer indefinitely.

- R1/R3: extract needed layers once and immediately store pooled vectors or bounded shards
- R4: create continuous caches only for layers/contexts promoted on development data
- Cache keys include model hash, processor hash, waveform hash, context mode, window coordinates,
  layer, pooling, dtype, and code hash
- Aggregate JSON does not contain raw tensors
- Large tensor/cache artifacts are stored under the external root
- Cache import is permitted only after sampled recomputation parity

---

## 13. Zero-Shot Score and Detector

### 13.1 Adjacent score

```text
d_adj[t] = 1 - cosine(z[t-1], z[t])
```

Because this score may be sensitive to phonetic change, it serves as a representation sanity
baseline.

### 13.2 Stable-prototype score

```text
d_proto[t] = 1 - cosine(p[t], z[t])
```

Initial prototype candidates:

- Normalized mean of the most recent K=3 or K=5 stable vectors
- EMA alpha selected from a small fixed grid on development data

Common state rule:

1. Update the prototype only in the stable state
2. Enter the candidate state when the change threshold is exceeded
3. Freeze the prototype in the candidate state
4. Emit an event after consecutive confirmation
5. After a confirmed event, promote a new-speaker prototype only after a separate stabilization
   condition is satisfied

### 13.3 Hysteresis/debounce

Apply at least the following common conditions to every representation.

- Single-threshold, one-hop baseline
- Two-hop confirmation
- Three-hop confirmation
- Change/stay dual-threshold hysteresis

Thresholds are selected on development data only. Threshold values may differ by representation,
but the same selection objective and false-event budget shall be used.

### 13.4 Fairness principle

In the first R4 comparison, detector logic, hop, candidate freeze, confirmation, and matching stay
identical. Only the representation changes.

A separate complex reducer for an individual model is permitted only as an ablation after R4.

Results are separated into two panels.

1. `matched-budget panel`: every encoder uses the same pooling/context, hop, detector, and
   observation deadline. This is the one-to-one comparison of the representations themselves.
2. `best-operating-point panel`: each encoder's configuration is selected on development data and
   compared on the accuracy-latency-compute Pareto frontier. This is the practical system
   comparison.

A best configuration using a different pooling duration shall not be described as a fully
controlled one-to-one representation advantage.

### 13.5 Speech-activity conditions

Do not use a different VAD for each encoder. R4 distinguishes three conditions.

1. `common_causal_vad`: use the same frozen causal VAD gate for every encoder. This is the primary
   operational panel.
2. `ungated_full_stream`: run over the entire stream, including silence and noise. This is a
   mandatory robustness panel.
3. `oracle_activity`: use the GT activity mask to select speech windows. This is a diagnostic upper
   bound and cannot support product or streaming claims.

The existing B0 VAD may be reused as the common-gate candidate, but the new protocol shall freeze
its exact checkpoint, threshold, debounce, and availability timing again. If the VAD delays B
onset, that delay shall not be removed from the detector observation frontier.

---

## 14. Core Execution Phases

### R0. Protocol freeze

#### Work

1. Resolve the open decisions in this document.
2. Freeze the primary event, metrics, split, model IDs, and licenses.
3. Record model/corpus artifact revisions and SHA-256 values in the registry.
4. Define the configuration schema and run contract.
5. Implement a confirmatory-test access guard.
6. Obtain approval for compute and storage ceilings.

#### Exit criteria

- Protocol content hash generated
- Model registry complete
- Dataset split ledger complete
- Primary analysis and promotion rule are machine-readable
- Test access fails closed
- License status recorded as one of `research_allowed`, `product_allowed`, `restricted`, or
  `unknown`

### R1. Extraction and causality parity

#### Work

1. Create a locked research environment separate from existing application dependencies.
2. Load the four encoders at exact revisions.
3. Validate feature shape, frame rate, and valid masks on D0 fixtures.
4. Mutation-test that a local trailing window does not read future audio.
5. Verify that changing waveform after frontier `t` cannot alter the prediction at `t`.
6. Validate ERes pre-pooling reconstruction parity.
7. Validate repeated-run determinism and batch/single parity.
8. Measure per-model smoke runtime and peak memory.

#### Exit criteria

- Exact input/output contract exists for every primary extractor
- Causal future-mutation test passes
- Timestamp-mapping fixture passes
- ERes tap parity passes, or the ERes condition is explicitly marked `not_available`
- Forecast is within the approved compute/storage ceiling

### R2. Dataset and GT freeze

#### Work

1. Import existing manifests into the new schema and revalidate source hashes.
2. Audit and materialize approved public D4 development sources; register and seal the public D5
   confirmatory corpus/partition without reading scores or aggregate labels.
3. Generate active-speaker regions and the event taxonomy.
4. Build positive, negative, nuisance, overlap, and backchannel inventories.
5. Check speaker/session/transformation leakage.
6. Freeze pair matching and bootstrap blocks before observing scores.
7. Generate ambiguity and missingness inventories.

#### Exit criteria

- No overlap among development/confirmatory speakers or sources
- Every WAV and annotation hash is frozen
- Event/block counts for every primary stratum are disclosed
- Insufficient language/scenario coverage is predeclared as excluded from confirmatory claims
- Test ledger remains sealed

### R3. Frozen zero-shot representation probe

#### R3-A. Homogeneous pair discrimination

For every encoder/layer/pooling condition, calculate:

- Same/different cosine distributions
- ROC-AUC and EER
- Distribution overlap coefficient
- Same-near, same-far, nuisance, and similar-voice strata
- Per-language and worst-group results

#### R3-B. Boundary-conditioned signal

- Adjacent and prototype distances
- Matched positive/negative AUC/EER
- Paired delta versus acoustic controls
- Clean, gap, overlap, and backchannel trajectories
- Response onset and recovery time

#### Funnel rule

Do not run the entire layer × pooling × hop grid over continuous audio.

1. Evaluate representative layers and pooling settings for every encoder on candidate-conditioned
   examples.
2. Select at most three nondominated configurations per encoder.
3. Preserve at least one sentinel per encoder rather than retaining only the global top model.
4. Perform R4 continuous extraction only for promoted configurations.

Do not eliminate a valid encoder from R4 merely because one layer or pooling condition has weak
zero-shot cosine. Every encoder with a valid extractor retains at least one sentinel configuration
through continuous zero-shot evaluation. The report may recommend a future learned probe because
cosine geometry does not prove supervised extractability, but no such probe is run here.

#### Exit criteria

- Every declared representative layer for all four encoders has a result or missing reason
- Primary matched rows are identical across encoders
- Candidate-conditioned results are not confused with continuous claims
- R4 promotion ledger is generated without test access

### R4. Frozen continuous zero-shot SCD

#### Work

1. Run causal replay at a 50 ms hop.
2. Evaluate adjacent/prototype and hysteresis grids on development data.
3. Build recall-latency-false-event Pareto frontiers across thresholds.
4. Run 20/100 ms hop sensitivities only for top configurations.
5. Decompose errors by clean, gap, overlap, backchannel, and nuisance conditions.
6. Include LS-EEND or legacy ERes results in a contextual table only when they can be mapped to the
   same GT/time contract.

#### Exit criteria

- Free-running false events/hour are available in addition to candidate-conditioned AUC
- Event matching is deterministic and one-to-one
- Localization and availability latency are separated
- A dev-selected zero-shot operating point is locked for each model

### R5. Frozen encoder + small causal SCD head — deferred and not authorized

R5 is the key phase separating whether a representation is directly visible to cosine from whether
a small model can extract it. It is retained as a follow-up protocol sketch only. None of the
linear probes, learned layer weights, temporal heads, checkpoints, or training curves in this
section may be produced under the current execution.

#### R5-A. Linear relational probe

Inputs:

```text
current vector
previous/prototype vector
absolute difference
elementwise product
```

Keep classifier capacity and training budget identical except for the encoder-specific projection.
This probe measures whether speaker-change information is linearly accessible in the representation.

#### R5-B. Primary temporal head

Freeze the primary head in the following form.

```text
encoder vector stream
→ Linear to 256
→ small causal temporal stack
→ Linear
→ P(new_speaker_onset)
```

The primary input stream consists of causal `local_trailing_window` vectors generated on the same
50 ms observation grid as R4. A head using `offline_full_context` hidden states is a separate
upper-bound ablation and cannot be a primary frozen-head result.

Freeze the exact architecture in the protocol before implementation. The initial recommended
primary head is a causal TCN. A causal Conformer is reserved as a top-encoder sensitivity to test
whether head architecture itself changes the encoder ranking.

Fairness conditions:

- Identical hidden dimension
- Identical temporal receptive field
- Identical output grid
- Similar trainable-parameter/FLOP budget
- Identical loss, optimizer, step budget, and early-stop rule
- Identical nested train subsets and seeds
- Frozen encoder

The Linear projection that absorbs encoder-specific input dimensions is counted separately as an
adapter parameter. The temporal head after projection has exactly the same architecture and
parameter count for every encoder.

#### Layer conditions

For each SSL encoder, distinguish:

1. Last layer
2. Best single layer selected in R3/R4
3. Supervised convex weighted layer sum

For ERes, report the best fixed tap and, if needed, a fixed multi-tap projection separately.
Learned fusion shall not be described as a zero-shot result.

#### Training target

- Use a causal, one-sided onset target aligned to the output grid
- Do not mark frames before onset as positive
- Freeze positive width, class weighting, negative sampling, and NMS at protocol freeze
- Calculate weighting from train-label prevalence only

#### Sample-efficiency curve

Use nested subsets defined by speaker/session block.

```text
1%, 5%, 20%, 100%
```

If a very small subset lacks enough positive blocks, raise it to the predefined minimum block unit
and record the actual fraction. Run every condition with a fixed set of multiple seeds and report
optimization variance.

#### Exit criteria

- An identical primary-head condition exists for all four encoders
- Trainable parameter count and receptive field are recorded
- Nested data curves and seed-level results are preserved
- Dev-selected encoder configuration/head checkpoint is locked before test access

### R6-Z. Public-data zero-shot confirmatory test and final report

#### Execution order

1. Record exact code/model/data/config hashes in the frozen evaluation contract.
2. Open the test access ledger.
3. Run the four encoders' locked zero-shot conditions once on the sealed public confirmatory
   partition. Do not train or evaluate a learned head.
4. Independently recompute aggregates from raw rows.
5. Validate missingness, causal frontiers, pair identity, and split leakage.
6. Produce final tables, plots, error analysis, and recommendations.

If a threshold or configuration is changed after viewing test results, reclassify that result as
exploratory. A new confirmatory claim requires a new untouched test set.

---

## 15. Conditional Learned Phases After the Zero-Shot Core

R5 and every phase in this section require a new explicit owner approval. A future learned-head
confirmatory evaluation shall use a new untouched public partition rather than reusing R6-Z as
confirmatory evidence.

### R7. Partial fine-tuning

Limit R7 to at most two or three encoders whose separately approved frozen-head results are
promising in R6-T.

```text
Stage A  encoder frozen
Stage B  top 2 Transformer layers or final ERes stage unfrozen
Stage C  top 4 layers unfrozen
Stage D  full fine-tuning, separate approval required
```

Start each stage from the same data and head, and separate the effect of pretrained representation
from the effect of fine-tuning. Report catastrophic forgetting, language worst-group behavior, and
increased compute together.

### R8. Handoff extension

Extend the input signal to:

```text
SCD probability
VAD/activity
OSD/overlap probability
speaker-state representation
temporal persistence
```

Example states:

```text
CURRENT
→ POSSIBLE_CHANGE
→ OVERLAP_OR_BACKCHANNEL
→ NEW_SPEAKER_PERSISTING
→ HANDOFF_CONFIRMED
```

Store `new_speaker_onset`, `overlap_start`, `exclusive_new_speaker`, and
`conversational_handoff` latencies separately. A backchannel is not an SCD false positive; it is
evaluated separately as a potential handoff false positive.

### R9. Distillation and product study

If a teacher candidate exceeds the product budget, evaluate:

```text
best teacher representation/logits
→ causal student 10–50M
→ INT8 ONNX
→ CPU batch=1 streaming benchmark
```

Candidate distillation targets:

- Selected-layer feature projection
- Pair/prototype geometry
- SCD logits
- VAD/OSD auxiliary logits
- Speaker-state contrastive target

The R9 product gate includes license, model size, peak RAM, one-stream RTF, two-stream backlog,
cold load, and quantization loss in addition to accuracy.

---

## 16. Metrics

### 16.1 Representation metrics

- ROC-AUC
- EER
- Distribution overlap coefficient
- Same/different mean, median, and quantiles
- Paired neural-minus-acoustic AUC difference
- Boundary response onset
- Boundary minimum/maximum distance
- Post-boundary recovery time
- Nuisance false-change score

### 16.2 Event metrics

- Boundary Precision, Recall, and F1 at ±100/250/500 ms
- Recall by availability deadline: 100/200/300/500/750/1000/1500/2000 ms
- Median, p90, and p95 availability latency
- Signed localization error
- False events/minute and false events/source-hour
- Missed-change rate
- Duplicate-event rate
- Backchannel-onset recall
- Backchannel-to-handoff false-conversion rate, added in R8

### 16.3 Compute metrics

- Model parameter count and checkpoint bytes
- Cold model-load time
- Warm batch=1 inference time, p50/p90/p95
- Offline batched throughput
- One-stream RTF
- Algorithmic observation latency
- Compute latency
- End-to-end latency
- Peak RAM/VRAM
- Cache bytes/audio-hour
- CPU/GPU backend, thread count, and precision

GPU batched throughput shall not be reported as real-time batch=1 latency.

---

## 17. Primary Endpoint and Operating Points

### 17.1 Representation stage

The R3 primary diagnostic is the ROC-AUC and EER of prototype distance over matched hard
positive/negative examples. Designate 300 ms and 500 ms as key contexts and report the remaining
contexts as sensitivities forming the latency curve.

### 17.2 Continuous zero-shot

Show the following reference false-event budgets on development data.

```text
0.5, 1, 2, 5 false events / source hour
```

For the primary operating point, freeze a threshold on development data targeting
`<=1 false event/hour`, then prioritize recall within 500 ms under that condition. If the actual
development data are too small to estimate this rate reliably, promote integer false-event
allowances and the complete Pareto frontier to primary status, and record that change in a protocol
amendment before opening the test set.

Key secondary metrics:

- Recall within 300 ms
- F1@250 ms
- Median/p95 availability latency
- Worst-language recall
- Nuisance false events/hour
- One-stream RTF

Do not force the four models into a single ranking through one weighted score.

---

## 18. Matching and Statistical Analysis

### 18.1 Event matching

Match predictions and GT deterministically one-to-one within an epoch.

Match conditions:

1. Compatible event type
2. Localization error within the designated tolerance
3. Availability latency within the designated deadline
4. No causal-frontier violation

First maximize the number of matches, then minimize total availability latency and localization
error. One prediction cannot receive credit for multiple GT events.

### 18.2 Uncertainty unit

Do not use the number of frames, windows, or pairs as the number of independent samples.

- Public conversation: source-session component connected by recurring participants
- Controlled recording: speaker component
- Synthetic data: component connected by common source speaker/utterance/transformation seed

Calculate 95% percentile intervals with 10,000 deterministic whole-block bootstrap replicates.

### 18.3 Paired comparison

- Compare encoders on identical matched examples or identical continuous sessions
- Block-paired bootstrap AUC, recall, false-event, and latency differences
- If missing examples differ across encoders, report both the intersection primary analysis and a
  missingness sensitivity
- Only one locked primary configuration per encoder enters the test set

### 18.4 Multiple comparisons

The layer/pooling grid is a development exploration. On the confirmatory test set, run only the
predeclared pairwise comparisons among the four locked encoder conditions. If a statistical
significance claim is made, freeze the family-wise correction method at protocol freeze. It is not
permitted to select the best test row among many layers and claim superiority without correction.

### 18.5 Missingness

The following are separate reason codes.

- Insufficient input samples
- No valid output frame
- Ambiguous GT
- Missing speaker metadata
- Failed frontend parity
- Nonfinite representation
- Unavailable model artifact
- Compute abort

Do not delete a failed configuration or silently recompute on successful samples only.

---

## 19. Plots and Tables

### 19.1 Required plots

1. Layer × pooling ROC-AUC heatmap
2. Same/different/nuisance score ECDF or violin plot
3. GT and four-encoder boundary trajectories on the same audio timeline
4. Mean clean/gap/overlap/backchannel trajectories with confidence bands
5. Boundary F1 or recall versus observation deadline
6. Recall versus false events/hour Pareto frontier
7. Accuracy versus end-to-end latency
8. Per-language and worst-group comparison
9. Sample-efficiency learning curves
10. Accuracy versus RTF/model-size Pareto plot

### 19.2 Required comparison table

```text
Method / layer / context
ROC-AUC / EER
F1@250 / F1@500
Recall@300 / Recall@500
Median / p95 availability latency
False events/hour
RTF / peak memory
```

Any future trained-head results shall be reported separately from this zero-shot study rather than
retroactively mixed into its row group.

---

## 20. Proposed Experiment Architecture

```text
experiments/speaker_representation_scd/
    EXPERIMENT_PLAN.md
    README.md

    configs/
        protocol/
        extraction/
        detectors/

    data/
        manifests/
        splits/
        annotations/

    models/
        registry.json

    extraction/
        base.py
        ssl.py
        eres_prepooling.py
        pooling.py

    detection/
        adjacent.py
        prototype.py
        hysteresis.py

    evaluation/
        pairs.py
        events.py
        latency.py
        statistics.py

    visualization/
        trajectories.py
        distributions.py
        pareto.py

    scripts/
        preflight.py
        extract_features.py
        run_zero_shot.py
        evaluate_locked.py
        compare_encoders.py

    tests/
    results/
```

The layout may be adjusted to repository naming conventions during implementation. The important
boundary is:

```text
waveform/annotation
→ feature extractor
→ temporal reducer
→ detector or trainable head
→ common event schema
→ model-independent evaluation
```

The new experiment code shall not be imported into production `src/puripuly_heart/` or compose
the application runtime. Future product integration is a separate architecture decision after R9.

---

## 21. Common Schemas

### 21.1 Feature observation

```json
{
  "experiment_id": "speaker_representation_scd_v1",
  "audio_id": "sample001",
  "encoder_id": "wavlm-base-plus",
  "checkpoint_sha256": "...",
  "layer_id": "L6",
  "context_mode": "local_trailing_window",
  "window_start_sample": 32000,
  "window_end_sample": 36800,
  "observed_source_sample": 36800,
  "pooling_ms": 300,
  "vector_cache_ref": "...",
  "valid_frame_count": 15,
  "missing_reason": null
}
```

### 21.2 Raw detector observation

```json
{
  "audio_id": "sample001",
  "method_id": "wavlm-L6-P300-prototype",
  "boundary_source_sample": 55200,
  "observed_source_sample_at_emit": 57600,
  "score": 0.73,
  "state": "candidate",
  "prototype_update": "frozen"
}
```

### 21.3 Normalized event

```json
{
  "audio_id": "sample001",
  "method_id": "wavlm-L6-P300-causal-head",
  "event_type": "new_speaker_onset",
  "boundary_source_sample": 55200,
  "observed_source_sample_at_emit": 57600,
  "confidence": 0.87
}
```

### 21.4 Run contract

Every run records at least:

- Git commit and dirty-worktree state
- Experiment-protocol content hash
- Code/config hash
- Model repository, revision, artifact hash, and license
- Processor/frontend identity
- Dataset manifest and split hash
- Sample rate, context mode, window, pooling, and hop
- Layer/tap, normalization, score, prototype, threshold, and hysteresis
- Training seed, train subset, optimizer, steps, and checkpoint hash when a separately approved
  learned phase is executed; otherwise record `not_applicable`
- Python, PyTorch/Transformers/ORT/CUDA versions
- CPU/GPU, thread count, precision, and batch size
- Start/end UTC, wall time, and peak memory
- Result content hash and child-artifact hashes

---

## 22. Verification

### 22.1 Required automated tests

- Exact 16 kHz sample/time conversion
- Window-coordinate and no-future-read tests
- Future-audio mutation invariance
- Padding/valid-mask exclusion
- Feature-timestamp fixtures
- Model repeated-run determinism
- Batch versus single extraction parity
- ERes tap-to-final reconstruction parity
- L2 normalization and zero-norm behavior
- Prototype update/freeze state transitions
- Event one-to-one matching
- Latency/localization separation
- Split leakage and connected-component blocking
- Run/result self-hash verification
- Aggregate recomputation from raw rows
- Held-out access guard

### 22.2 Independent verification

Before the R6-Z report is accepted, the verifier shall distrust aggregate JSON and recompute the following from raw rows and
the frozen contract.

- Example/pair/session population
- ROC-AUC and EER
- Event matching
- Latency quantiles
- False events/hour
- Bootstrap intervals
- Model/config/data identities

Mutated fixtures changing a score, frontier, pair ID, split, or model hash shall fail verification.

---

## 23. Compute and Environment Plan

The current application environment does not pin PyTorch/Transformers as primary dependencies.
R1 therefore uses a separate locked research environment and does not immediately expand the
production dependency surface for this experiment.

### Preflight order

1. Smoke-test ten fixtures per model
2. Benchmark 100 windows in batch and single modes
3. Measure seconds/window and peak memory per model
4. Calculate job count from the complete coordinate ledger
5. Forecast cache bytes and wall time
6. Compare the forecast with user-approved ceilings

### Compute-reduction funnel

- Screen layers/pooling on R3 candidate windows
- Run R4 continuous evaluation for at most three configurations per encoder
- Cache pooled vectors instead of complete hidden-state tensors
- Restrict offline full-context evaluation to a small diagnostic subset

### Runtime claims

Separate research-GPU batched extraction from target-CPU batch=1 streaming results. A 95M-class
teacher may remain useful as a representation teacher even if it is not real-time. Product
feasibility is judged separately in R9 using a student or a compact ERes candidate.

---

## 24. Promotion and Stop Gates

### Gate G0 — Extraction validity

Pass conditions:

- Exact artifact and license verified
- No-future-read validated
- Frame/tap parity established
- Deterministic cache identity established

An encoder that fails is `not_evaluable`, not a performance loser.

### Gate G1 — Representation signal

Evidence examined:

- Paired AUC delta and interval versus acoustic controls
- Nuisance separation
- 300/500 ms boundary response
- Language consistency

Even if G1 is weak, retain one sentinel configuration per encoder through R4. Failure of zero-shot
cosine does not directly prove the absence of information in the representation and may motivate,
but does not authorize, a learned-probe follow-up.

### Gate G2 — Continuous-detector viability

Positive evidence:

- Candidate AUC transfers to free-running performance
- Useful recall at the false-event budget
- Prototype/hysteresis reduces nuisance false switches
- Availability latency and compute are measurable

If every encoder has high candidate AUC but excessive continuous false events, conclude that the
pair task does not represent actual SCD.

### Gate G3 — Authorization decision for a future frozen-head study

This gate is a recommendation only in the current study. To become a candidate for separately
approved frozen-head work, an encoder should satisfy at least one of the following.

- Higher recall than the ERes final-embedding baseline at the same false cost
- Lower latency or false cost at the same recall
- Meaningful Pareto advantage in multilingual worst-group performance
- Similar accuracy at substantially lower compute/model size

If no encoder exceeds the acoustic/ERes controls, revisit formulation, memory, and annotation
rather than proceeding directly to large-scale fine-tuning.

### Gate G4 — Fine-tuning

R7 is approved only when frozen-head signal exists and there is evidence of remediable
underfitting. Do not perform full fine-tuning merely because the zero-shot ranking is undesirable.

### Gate G5 — Handoff/product

- SCD plus OSD/persistence separates backchannels from sustained turns
- Legal/license gate passes
- Target-device RTF and memory budget pass, or the model has demonstrated value as a distillable
  teacher

---

## 25. Actions for Expected Result Patterns

### A. ERes pre-pooling is strong

- Prioritize an ERes backbone + primary causal head
- Run a wide-ERes-checkpoint sensitivity
- Investigate causalization and ONNX/INT8 feasibility

### B. WavLM/UniSpeech-SAT is strong

- Analyze the best single layer and weighted fusion
- Perform top-layer fine-tuning after the frozen-head stage
- Prioritize teacher-to-small-student distillation

### C. mHuBERT is strong

- Separate language-specific gains from speaker/acoustic nuisance behavior
- Prioritize it as a multilingual teacher
- If its license restricts product use, separate the research teacher from a deployable replacement

### D. Zero-shot is weak but the small head is strong

- Conclude that linearly or temporally extractable information, rather than cosine geometry, is the
  key factor
- Confirm the value of retaining every encoder through the small-head stage rather than selecting
  only the zero-shot winner

### E. All models are weak at 100–300 ms

- Do not conclude immediately that the encoders failed
- Redefine longer left context, stable-speaker memory, and the temporal head as primary variables
- Quantify the 500–1000 ms lower bound and annotation ambiguity

### F. SCD is strong but handoff is weak

- Treat this as an expected and valid result
- Add OSD, old-speaker offset, persistence, and a state machine in R8

---

## 26. Major Risks and Mitigations

| Risk | Incorrect conclusion | Mitigation |
| --- | --- | --- |
| Full-audio bidirectional leakage | Apparent extremely low latency | Primary trailing-window mode and future-mutation test |
| Synthetic splice artifact | Detecting edit artifacts rather than speakers | Acoustic controls and separate natural/controlled strata |
| Pair pseudo-replication | Artificially narrow confidence intervals | Speaker/session-component bootstrap |
| Test tuning | Non-reproducible winner | One-shot test access after config hash freeze |
| Last-layer bias | Missing SSL speaker information | Representative-layer screen |
| Zero-shot cosine bias | Mistaking cosine failure for absence of learnable information | Retain one R4 sentinel and state that supervised extractability remains unknown |
| Incorrect ERes tap | False pre-pooling claim | Final-embedding reconstruction parity |
| Confusing pooling and latency | Reporting 100 ms pooling as 100 ms E2E latency | Separate frontier, compute, and E2E time |
| Language imbalance | Overstated multilingual claim | Language strata and worst-group reporting |
| Confusing overlap and handoff | Treating a backchannel as false SCD | Separate onset, exclusive, and conversational events |
| Model license | Research winner unusable in product | R0 license registry and R9 legal gate |
| Excessive cache/grid | Experiment never completes | Candidate funnel and pooled cache |
| Resource contention with the current legacy run | Delayed or damaged existing results | Independent roots and separated execution schedule |

---

## 27. Deliverables

### Before execution

- Approved protocol Markdown and content hash
- Model registry
- Dataset/split ledger
- GT taxonomy and annotation guide
- Compute forecast
- Pre-execution review

### R1/R2

- Extraction-parity report
- Causality-mutation report
- ERes pre-pooling tap report
- Dataset coverage and leakage audit
- Pair/block ledger

### R3/R4

- Pooled raw feature-score shards
- Representation-metrics table
- Same/different/nuisance distributions
- Boundary-trajectory plots
- Continuous zero-shot event report
- Accuracy-latency-false-event Pareto plots

### R6-Z

- Locked public confirmatory raw scores and events
- Independent verification report
- Final zero-shot comparison report
- Recommendation on whether a separately approved R5 study is warranted

### Conditional R5/R6-T

- Head configs and exact trainable-parameter counts
- Seed-level checkpoints and training curves
- Sample-efficiency report
- New untouched learned-head confirmatory raw events

---

## 28. Final Report Template

The final report follows this order.

### A. Protocol and validity

- Model/checkpoint/license
- Dataset/split/coverage
- Causal context and timing
- Deviations and missingness

### B. Zero-shot representation

- Best single layer and pooling
- Same/different/nuisance AUC/EER
- Boundary-straddling trajectory
- Per-language results

### C. Continuous zero-shot SCD

- Adjacent versus prototype
- Threshold/hysteresis
- Boundary F1, latency, and false events/hour
- Error taxonomy

### D. Learned-head follow-up decision

- Whether zero-shot evidence warrants a separately approved frozen-head experiment
- Encoders and fixed representation conditions eligible for that follow-up
- Data, compute, and license blockers that must be resolved first
- No trained-head result is reported in the current study

### E. Unified comparison

- Accuracy-latency-compute Pareto frontier
- Multilingual worst-group performance
- Contextual ERes-final/LS-EEND baseline

### F. Decision

- Research teacher candidate for follow-up
- Compact-backbone research candidate
- Need for a frozen-head study or fine-tuning
- Need for handoff extension
- Distillation/product recommendation

---

## 29. Execution Checklist

```text
[ ] Protocol approved
[ ] Model revisions/hashes/licenses frozen
[ ] Research environment locked
[ ] D0 causal/parity tests
[ ] ERes pre-pooling tap parity
[ ] Dataset manifest/split/GT frozen
[ ] Pair/block ledger generated
[ ] Compute/storage forecast approved
[ ] R3 candidate representation grid
[ ] Per-encoder configuration funnel
[ ] R4 continuous zero-shot replay
[ ] Zero-shot operating points locked
[ ] Public confirmatory corpus/partition sealed
[ ] Zero-shot test configuration hash locked
[ ] R6-Z confirmatory test executed once
[ ] Independent recomputation/verifier
[ ] Plots and final report
[ ] Separate R5/R7/R8/R9 go/no-go recommendation
```

---

## 30. Open Decisions Before R0 Approval

### 30.1 Owner-resolved constraints

1. The current execution stops after frozen zero-shot representation, continuous zero-shot SCD,
   and a zero-shot confirmatory report. R5 and every learned phase require separate approval.
2. No participant recruitment or private recording is available. D4 and D5 use public data only.
3. The exact legacy ERes/LS-EEND common-GT data may be reused as `development-known` paired
   comparison data, never as untouched confirmatory evidence.
4. D5 shall be a separately sealed public corpus or public speaker/session partition that was not
   used to select layers, pooling, detector logic, thresholds, or operating points.
5. Existing LS-EEND results may enter only a development-known event-level contextual table on
   the exact common-GT/time-contract subset. They do not enter representation ranking or R6-Z
   confirmatory claims.
6. The integer development false-event Pareto frontier is primary. `1/hour` remains a labeled
   reference and becomes an operating point only if the development exposure supports it.
7. The accepted research path is sequential CPU-only execution on the identified local host after
   the legacy run releases resources. The machine-readable contract limits execution to one model
   and one worker, eight CPU threads, 24 GiB resident RAM, 25 GiB source downloads, 20 GiB derived
   cache, 50 GiB external storage, 96 total wall hours, and 24 wall hours per model. Full extraction
   remains disabled until a 10-fixture/100-window smoke forecast passes and is explicitly recorded.
   Any GPU path requires a protocol amendment with a new hardware identity.
8. Restricted or legally unresolved encoders remain eligible only for research comparison. They
   are excluded from product-eligibility claims unless a separate legal/license gate records
   `product_allowed`.

### 30.2 Remaining R0 decisions

No owner-level R0 decision remains open. Artifact acquisition, ERes source/tap parity, environment
locking, smoke forecasting, and confirmatory unlock are downstream gates that must satisfy the
frozen contracts; they do not authorize changing the decisions above.

These decisions cannot be changed after viewing test results. If a change is necessary, increment
the protocol version and content hash and mark existing results exploratory.

---

## 31. Primary References

- [mHuBERT-147 model card](https://huggingface.co/utter-project/mHuBERT-147)
- [mHuBERT-147 paper](https://www.isca-archive.org/interspeech_2024/zanonboito24_interspeech.html)
- [WavLM paper](https://arxiv.org/abs/2110.13900)
- [WavLM Base+ model card](https://huggingface.co/microsoft/wavlm-base-plus)
- [UniSpeech-SAT paper](https://arxiv.org/abs/2110.05752)
- [UniSpeech-SAT Base+ model card](https://huggingface.co/microsoft/unispeech-sat-base-plus)
- [ERes2NetV2 paper](https://arxiv.org/abs/2406.02167)
- [VoxConverse project](https://www.robots.ox.ac.uk/~vgg/data/voxconverse/)
- [VoxConverse v0.3 annotations](https://github.com/joonson/voxconverse)
- [SCDNet: SSL representations and Conformer SCD](https://www.isca-archive.org/interspeech_2024/li24q_interspeech.html)
- [Multi-task VAD/OSD/SCD with wav2vec 2.0](https://arxiv.org/abs/2210.14755)
- [Speaker-change knowledge distillation](https://www.isca-archive.org/interspeech_2024/su24_interspeech.html)

---

## 32. Architecture Boundary

This plan adds an experiment-only artifact and does not change the current application
architecture, runtime owners, providers, UI, audio-capture lifecycle, or production composition.

If implementation adds experiment dependencies to `src/puripuly_heart/` or makes the production
audio path directly own a research model, that change may constitute architecture drift and
requires separate design work and user approval.
