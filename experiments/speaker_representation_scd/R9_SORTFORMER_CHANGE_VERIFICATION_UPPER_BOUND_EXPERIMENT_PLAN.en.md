# R9 Sortformer Change-Verification Upper-Bound Experiment Plan

## 1. Status and Scope

This document defines a bounded follow-up to the completed R8 Streaming Sortformer feasibility
experiment. R8 ended with Outcome C: compute passes, accuracy fails. R8 measured a raw-probability
onset decoder with a single global threshold and reported a recall-versus-false-events curve that
collapsed to 1.7% Recall@250 at 20 false events/hour.

R8 also left a specific, unexplored observation:

> Scored under the model's own fixed post-processing (0.5-threshold speaker segments, the library's
> documented output policy), Sortformer's segment starts reach approximately 85% Recall@250 on the
> same ten meetings, at approximately 1,293 false events/hour.

R9 asks one question about that observation:

> What is the best achievable recall-versus-false-events curve when a change-verification layer
> filters Sortformer's own 0.5 segment-start candidates — separately for (A) features computable
> only from the already-dumped probability tensor, and (B) features that additionally require the
> model's internal speaker-cache embeddings?

**Owner instruction, 2026-08-13: R9 does not target a fixed performance threshold.** The purpose is
to measure the performance upper bound, not to pass the inherited R7-B/R8 usefulness gates. Those
gates are reported as reference lines only and are not continuation criteria. The primary
deliverable is a measured ceiling curve per arm, plus a statement of which information (public vs
internal) carries the ceiling.

R9 planning is complete when this document fixes the candidate stream, features, verifier forms,
scoring, ceiling protocol, comparison rules, artifacts, and outcome rules. Execution requires a
separate explicit owner authorization, exactly as R8 required. Product integration, publication,
and follow-up experiments remain unauthorized.

## 2. Why R9 Is Next

Three measured facts motivate it:

1. R8's single-threshold onset decoder discards the model's own operating semantics. The model's
   natural decision boundary is 0.5 (its shipped `probs_to_speaker_segments` uses a fixed 0.5
   threshold); R8's operating points at 0.988-0.995 convert new-speaker confidence buildup into a
   median +400 to +480 ms decision delay, destroying 250 ms recall. The information loss is a
   property of the R8 decoder, not of the model.
2. The model's own policy emits far more candidates than the product budget allows. The gap between
   the candidate stream (1,293 events/hour) and the product region (1-20 false events/hour) can
   only be closed by verifying each candidate with identity-style evidence. R8 never built that
   layer; a single scalar threshold was its only verifier.
3. Sortformer carries that evidence internally. Its Arrival-Order Speaker Cache (AOSC) maintains
   per-speaker acoustic embeddings — the model's core speaker-identity representation — but the
   public API exposes only speaker segments. R9-A measures the ceiling reachable without those
   embeddings; R9-B measures how much the embeddings add. This split answers the
   "public vs hidden information" question directly.

R9 remains a no-training falsification-first experiment. The only models permitted are small
linear/MLP verifiers trained on frozen candidate features (as in R7-A). No encoder, Sortformer, or
any other model is trained or fine-tuned.

## 3. Claims R9 May and May Not Support

R9 may support:

- a measured recall-versus-false-events ceiling curve for a verification layer over Sortformer's
  own candidate stream, separately for probability-only and embedding-augmented features;
- the maximum Recall@250 reachable at 1, 5, 10, 20, 50, and 100 false events/hour per arm;
- the false-event rate at which each arm reaches 50% and 80% of the candidate-stream ceiling;
- a statement of whether speaker-cache embeddings raise the ceiling relative to probability-only
  features, and by how much;
- an internal, development-known characterization of where the residual false candidates come from
  (same-speaker resume, slot flicker, cache compression, overlap artifacts).

R9 may not support:

- an untouched or confirmatory claim: all ten meetings are development-known, and the model card
  lists AMI and AliMeeting among the training corpora, so training-session overlap with this panel
  is not excluded. These curves are an in-domain, possibly optimistic ceiling;
- a product-readiness or false-event-rate claim below what the measured exposure (4.731 hours) can
  support statistically;
- a claim that the ceiling transfers to meetings with more than four speakers, non-English audio,
  or out-of-domain conditions;
- live push-audio integration claims (the pinned runtime entrypoint is whole-recording);
- a claim about the R7-B/R8 usefulness gates other than as reference lines, because the owner
  explicitly withdrew fixed-threshold pass/fail framing for R9;
- performance claims for presets other than `low_latency` unless a separately approved replay is
  executed.

## 4. Locked Third-Party System

R9 reuses the exact R8 system without change:

- `handy-computer/transcribe.cpp` at commit `d42c3bbdfa2f63c37e5891e27de47a612d62f221` (pinned
  prefix `d42c3bb`), vendored ggml tree `d0c8c9483f6c005599a15195ee24c1c6c6ab1c57`;
- model `handy-computer/diar_streaming_sortformer_4spk-v2.1-gguf`, repository revision
  `7ef0c15dc8f9d717e9d24fac29a6e6551e9c6ddf`, primary file
  `diar_streaming_sortformer_4spk-v2.1-Q8_0.gguf` (SHA-256
  `a5dacdc650790266c7a362e54e6bf51952015487edaa606c4e11632bc32442a9`);
- preset `TRANSCRIBE_SORTFORMER_PRESET_LOW_LATENCY` (chunk 6, right context 7, fifo 188,
  spkcache 188, update period 144; 80 ms frames; approximately 1,040 ms algorithmic lookahead);
- CPU Q8_0 full-panel probability dumps produced by R8 under
  `%SRSCD_CACHE_ROOT%/results/r8/streaming_sortformer_feasibility_v1/probabilities/cpu/*.npz`.

R9-A performs zero new inference. R9-B requires one telemetry-only patch to the external checkout
that dumps the speaker-cache embeddings. The patch must not alter tensor math, cache decisions,
post-processing, or scheduling geometry. On the deterministic Sortformer fixture, patched and
unpatched runs must produce byte-identical `diar.probs` and speaker segments, exactly as R8
validated for its timing patch. Failure of that check invalidates every R9-B result.

## 5. Evidence Mode and Dataset

R9 remains in `fast_internal_development_known` mode and reuses all ten R7-B/R8 meetings and no
other corpus. The exposure is 4.731 source hours with 4,619 `new_speaker_onset` references:

| Fold | Meetings |
| --- | --- |
| 1 | `alimeeting_R8001_M8004`, `ami_IS1009a` |
| 2 | `alimeeting_R8008_M8013`, `ami_EN2001d` |
| 3 | `alimeeting_R8009_M8019`, `ami_TS3006a` |
| 4 | `ami_ES2003a`, `alimeeting_R8007_M8010` |
| 5 | `ami_TS3009b`, `ami_ES2015d` |

Reference stratum composition is dominated by overlap onsets:

| Stratum | Count |
| --- | ---: |
| overlap_onset | 3,420 |
| silence_gap_change | 1,192 |
| short_backchannel_or_return | 5 |
| clean_change | 2 |

Stratum-level analysis is therefore meaningful only for overlap onsets and silence-gap changes.
All folds, waveforms, reference labels, source durations, meeting identifiers, event matcher, and
error-stratum definitions are reused from R7-B/R8 by hash. Audio is 16 kHz mono. No waveform may be
regenerated from a different channel or annotation source.

Every reported score is out-of-fold: each fold is held out in turn, features/verifier parameters
and thresholds are selected on the other eight meetings, and the held-out pair is scored once.
There is no separate untouched evaluation panel. Because AMI and AliMeeting appear in the
upstream Sortformer training corpus list, all R9 curves are explicitly labeled an in-domain upper
bound, and the report must request freezing a new panel before any confirmatory interpretation.

## 6. Candidate Stream

R9 does not use R8's raw-probability onset decoder. The candidate stream is the model's own
output policy: per-slot speaker segments produced by binarizing the R8 probability dumps at the
fixed 0.5 threshold (identical semantics to the library's `probs_to_speaker_segments`, threshold
0.5, no minimum duration).

For each of the four slots, a segment start is the first 80 ms frame of a run of
`probability > 0.5`. The expected candidate count is approximately 10,066 across the panel
(approximately 1,293 candidates per source hour), and the candidate ceiling is approximately 85%
Recall@250 at perfect filtering. The exact counts are recomputed in-protocol and frozen in
`candidates.jsonl` before any verifier score is inspected.

Candidate labels:

- positive: candidate start lies within 250 ms of a reference `new_speaker_onset`;
- negative: candidate start lies more than 500 ms from every reference event;
- ambiguous: between 250 and 500 ms; excluded from verifier training and from threshold selection,
  retained only for event-level evaluation where applicable.

R9-B may optionally add a second candidate source (see Section 8.4): intra-slot embedding-jump
candidates, which address speaker changes that occur inside a continuously active slot and
therefore produce no 0.5 segment start. This extension is secondary, must be fail-fast, and may
not be used to rescue a failed R9-B primary.

## 7. Verification Features

### 7.1 Probability-only features (R9-A)

Computed exclusively from the frozen R8 probability dumps (`T x 4`, 80 ms frames). The base
feature set is computable within the frozen base confirmation window (three frames, 240 ms) after
the candidate crossing; the confirmation-lag diagnostic recomputes the same features with the
frozen diagnostic window (six frames, 480 ms). Availability latency therefore equals the candidate
crossing plus a fixed window, never a data-dependent confidence delay.

1. `argmax_switch`: the dominant (maximum-probability) slot in a fixed window before the candidate
   differs from the dominant slot in a fixed window after it.
2. `gap_ms`: length of the all-slots-inactive gap immediately preceding the candidate crossing, if
   any; encoded as a missing indicator when no gap exists (overlap signature).
3. `pre_depth`: minimum probability of the most-recently-active slot inside the pre-window; a
   deep drop indicates a real speaker exit rather than a flicker.
4. `rise_slope`: probability gain of the candidate slot over the first k frames after its 0.5
   crossing (new-speaker confirmations rise differently from noise flicker).
5. `persistence_ms`: how long the candidate slot stays above 0.5 after the crossing.
6. `same_slot_resume`: the candidate slot was active with high probability in the recent past and
   is re-activating — a pause-resume signature.
7. `co_activity`: whether another slot stays active across the crossing (A -> A+B overlap
   signature) and its probability level.
8. `return_pattern`: whether the previously active slot re-activates shortly after the candidate
   (backchannel/return signature).
9. `cross_probability`: the candidate slot's probability value at the crossing frame and at the
   peak of the following short window.
10. `pre_pattern`: minimum and mean of the previous dominant slot's probability over the
    pre-window.

Exact window sizes and encodings are fixed in the checked-in R9 configuration before any feature
is extracted; the values listed here are the feature family, not the frozen constants.

### 7.2 Embedding features (R9-B)

Requires the Section 4 telemetry-only patch. For each candidate, the patched runtime dumps the
speaker-cache embedding state so the following are computable:

1. `same_slot_similarity`: cosine similarity between the candidate slot's embedding shortly after
   the crossing and its own embedding before the gap (or before the crossing for overlap
   candidates); near 1.0 means the same speaker resumed.
2. `best_other_similarity`: maximum cosine similarity between the candidate slot's embedding and
   every other cached slot embedding; high similarity means the "new" speaker was already known.
3. `embedding_jump`: magnitude of the embedding change of the candidate slot across the crossing.
4. `compression_boundary`: whether the candidate lies at or within one chunk of an AOSC cache
   compression event (identified from the R8 per-chunk telemetry); candidates straddling a
   compression are flagged and handled by the fixed rule defined in the configuration (expected
   default: exclude from embedding comparisons, keep for event-level evaluation).

The embedding dump format, the exact similarity windows, and the compression-handling rule are
fixed before extraction. If dumping the full cache exceeds the 24 GiB experiment ceiling or the
storage budget, a fixed subsampling rule (for example, one representative vector per update
period) is defined in the configuration before extraction begins; a subsampling decision made
after inspecting results is forbidden.

## 8. Arms

### 8.1 R9-A0 — deterministic rule stack

A fixed rule stack over probability-only features: remove same-slot resumes, apply gap and
co-activity rules, apply persistence minimums, then emit remaining candidates. At most five
numeric thresholds are selected on the development folds. This arm is a cheap sanity floor: it
must at least dominate the R8 raw curve at matched false-event rates to justify any learned
verifier work. If it does not, the feature set, not the model form, is the suspected limitation,
and the learned arm may still run as a ceiling probe.

### 8.2 R9-A1 — learned verifier, probability-only

An L2-regularized logistic regression (sklearn lbfgs, deterministic, class-reweighted) over
normalized R9-A features, trained per fold on the eight training meetings; threshold selection
happens on the development folds' out-of-fold scores. lbfgs is deterministic, so three-seed SGD is
unnecessary for the linear model; seeds apply only to the MLP fallback. If the mean out-of-fold
AUROC falls below the frozen trigger in the checked-in R9 configuration, one small MLP (below
approximately 100,000 trainable parameters, the R7-A capacity bound) replaces the logistic for the
ceiling estimate; no architecture search. R9-A1's cross-validated curve is the probability-only
ceiling estimate.

### 8.3 R9-B1 — learned verifier, embedding-augmented

Identical form and discipline as R9-A1, with R9-B features added. The comparison A1 vs B1
isolates the ceiling contribution of the speaker-cache embeddings. R9-B1 runs only after R9-A1
produces a valid curve, unless the owner pre-authorizes running both arms in parallel.

### 8.4 R9-B2 — intra-slot candidate extension (optional, secondary)

If B1 is valid but the candidate ceiling (Section 6) is confirmed to be the binding constraint,
a bounded extension may add intra-slot candidates: points inside continuously active slots where
the slot's embedding similarity to its own recent history drops below a fixed threshold. These
candidates use the same verification machinery. B2 is fail-fast: if the first fold shows no
candidates beyond noise, B2 stops and the R9-B ceiling is reported without it.

## 9. Scoring and Ceiling Protocol

### 9.1 Event construction

Verifier scores are converted to events by local maximum selection with the fixed 200 ms
duplicate-suppression radius inherited from R7-B/R8. Boundary timestamps stay at the candidate
start frame; decision availability is reported separately (Section 9.4). One-to-one event
matching with 100/250/500 ms tolerances; 250 ms is the primary view.

### 9.2 Curves

For each arm, sweep the verifier score threshold with a dense pass followed by exact refinement
over every unique score value bracketing each operating point of interest, as in R8. Report the
continuous recall-versus-false-events curve over all ten meetings, plus per-meeting curves,
stratum recall (overlap onset, silence-gap change), short-return recall, and the maximum
single-meeting share of matched true positives.

### 9.3 Ceiling summary (primary deliverable)

For each arm report, as a single table and figure:

- maximum Recall@250 at 1, 5, 10, 20, 50, and 100 false events/hour;
- the false-event rate at which the arm reaches 50% and 80% of the candidate-stream ceiling;
- the candidate-stream ceiling itself (approximately 85%) and the perfect-filter oracle line
  (recall equal to candidate recall at zero false events) as the absolute upper bound of this
  candidate stream;
- the R8 raw-probability curve as the incumbent lower reference;
- the model's own fixed 0.5 policy point (approximately 85% recall, approximately 1,293 false
  events/hour) as the unfiltered upper reference.

A confirmation-lag diagnostic repeats the best arm with verifier evidence extended to include up
to 500 ms of future audio beyond the candidate crossing. This variant is explicitly labeled
non-causal and answers "how much of the ceiling depends on waiting". It is a diagnostic only and
does not replace the causal curve.

### 9.4 Availability latency

Verification features are causal by construction. Report median/p90 availability latency of
emitted events (candidate crossing plus verification compute) and confirm that the learned arms
do not reintroduce the R8 0.99-threshold confirmation delay (median +400 ms or worse). If any arm
exhibits that signature, it is reported as a defect, not as a valid operating point.

### 9.5 Reference lines

The inherited R7-B/R8 usefulness gates (aggregate Recall@250 of 30% at no more than 10 false
events/hour and 50% at no more than 20 false events/hour; overlap and silence-gap recall non-zero;
maximum meeting share below 50%; held-out false-event rate below twice the transferred target)
are computed and drawn as reference lines on every figure. Per the owner instruction they are
context, not continuation criteria. The threshold-transfer view (threshold selected on the other
eight meetings, applied unchanged to each held-out pair) is mandatory and reported for every arm.

## 10. Compute-Cost Protocol

- R9-A: zero inference. Feature extraction and verifier training run on CPU and are expected to
  take minutes to a small number of hours.
- R9-B: one telemetry-patched CPU Q8_0 full-panel replay (approximately 3 hours at the measured
  R8 RTF of 0.572 on the target machine), embedding dump storage, plus the same feature/training
  cost as R9-A. The same fixed thread count (8), 24 GiB memory ceiling, and 24-hour per-backend
  forecast stop as R8 apply. Vulkan is not re-measured; R8's Vulkan evidence is reused.
- The deterministic-fixture byte-identity validation for the embedding patch is mandatory before
  any full-panel B work.

## 11. Predeclared Outcomes

### Outcome A — probability-only verification raises the ceiling

The R9-A1 curve meaningfully Pareto-dominates the R8 raw curve across the low-false-event region
(the definition of "meaningful" is fixed in the configuration before scoring; a candidate default
is at least a twofold recall increase at two of the 10/20/50 false-events-per-hour points).
Report the A1 ceiling, then decide whether B1 is worth its replay cost.

### Outcome B — embeddings carry the ceiling

A1 does not dominate meaningfully, but B1 does. Conclude that speaker-cache embeddings are the
load-bearing information for change verification, and report the B1 ceiling with the A1 curve as
the public-information contrast.

### Outcome C — neither arm moves the curve

Report that Sortformer's own candidate stream is not separable at low false-event rates with
these features. The measured ceiling is the honest product: state the maximum recall each arm
achieves at each false-event rate, and identify the dominant residual false-candidate classes.
Do not integrate; do not proceed to live push-audio adaptation.

### Outcome D — invalid or inconclusive

Use only for embedding-dump mismatch, backend fallback, corrupted cache continuity, irreconcilable
timestamp mapping, missing source evidence, or an exceeded execution ceiling before one valid
curve completes.

No outcome authorizes follow-up work automatically. R9 always ends with the ceiling table and a
request for the next decision.

## 12. Execution Sequence

1. Freeze the R9 configuration and re-hash the R8 input inventory, probability dumps, telemetry,
   and references.
2. Freeze the candidate stream (`candidates.jsonl`) and report exact candidate and stratum counts.
3. Extract R9-A features from the frozen dumps.
4. Run A0 (rule stack) as the sanity floor; score its curve.
5. Train and cross-validate A1; produce the probability-only ceiling curve and transfer view.
6. If authorized: apply the embedding telemetry patch, validate byte-identity on the fixture,
   and replay the full panel on CPU.
7. Extract R9-B features, train B1, and produce the embedding-augmented ceiling curve.
8. Run the B2 intra-slot extension only under its fail-fast entry condition.
9. Produce the ceiling summary table, curves, stratum and transfer views, latency audit, and
   representative timelines.
10. Select exactly one predeclared outcome and stop with a next-decision request.

## 13. Artifacts

Store all material outputs outside the repository under:

```text
%SRSCD_CACHE_ROOT%/results/r9/sortformer_change_verification_upper_bound_v1/
```

Required artifacts:

```text
config.json
r8_reuse_inventory.json          (hashes of every reused R8 artifact)
candidates.jsonl                 (frozen candidate stream with labels)
features_a.jsonl                 (probability-only features)
features_b.jsonl                 (embedding features, when run)
embedding_patch.diff
embedding_validation.json        (byte-identity fixture validation)
a0_metrics.json
a1_metrics.json
b1_metrics.json
b2_metrics.json                  (when run)
ceiling_summary.json
threshold_transfer_metrics.json
recall_false_event_curves.png    (all arms, reference lines, oracle line)
representative_timelines/
REPORT.md
```

Every receipt and final metric file records its own SHA-256 or is covered by a hashed inventory.
Aborted or invalid runs are preserved and clearly marked.

## 14. Product Architecture Boundary

R9 changes no production module. No architecture drift is expected while implementation and
results remain under `experiments/` and the external research cache. If a later outcome motivates
product use, Sortformer would remain a long-lived native resource as described in the R8 plan
Section 16; R9 provides no integration authorization and the whole-recording API limitation is
unchanged.

## 15. Completion and Approval Boundary

R9 planning is complete when this document fixes the source, model, candidate stream, feature
families, verifier forms, scoring, ceiling protocol, comparison rules, artifacts, and outcome
rules. Execution requires an explicit owner authorization naming the approved arms (for example,
"A only" or "A and B"), exactly as R8 required. Until that authorization, no feature extraction,
verifier training, patch application, or replay may begin.
