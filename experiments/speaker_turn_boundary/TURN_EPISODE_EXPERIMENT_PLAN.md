# Bounded turn-episode speaker-change fusion experiment

## 0. Status, authority, and purpose

Status: normative experiment plan; implementation has not started.

Authority order:

1. Explicit user decisions in the experiment thread, including discarding the long-session design
2. `.agents/specs/prd/speaker_change_turn_boundary_experiment_handoff_en(1).md`
3. GitHub issue #51
4. This plan
5. Verified raw artifacts from committed Phases 0-2 and the uncommitted Phase 3 development sweep
6. Older reports and analyses as historical evidence only

This plan replaces the discarded long-session design. Continuous 15-30 minute replay,
long-term speaker-return memory, session-age drift, and long-session model-state claims
are out of scope.

The experiment answers a product question:

> Can causal speaker-change evidence, after proposal stabilization and VAD fusion,
> reduce speech from different speakers being finalized in one STT/translation turn
> without causing too many harmful splits inside continuous same-speaker speech?

This is not a general diarization benchmark. Production wiring is out of scope. The
result may recommend a detector and policy for a later implementation task, but it may
also conclude that no local detector is ready.

## 1. Decisions fixed by this plan

The following decisions are frozen before implementation.

1. The primary audio condition is one mono mixed-audio source timeline at 16 kHz.
2. The evaluation unit is a bounded turn episode, not a complete meeting.
3. Model state resets at the episode boundary and remains continuous inside the episode.
4. VAD utterance boundaries do not reset LS-EEND neural state inside an episode.
5. A detector proposal is never scored as a product cut before causal clustering and fusion.
6. Clean and gap speaker handoffs are hard-boundary targets.
7. Interruption/overlap onset is a soft-marker target and is excluded from the hard-turn headline.
8. Same-speaker pauses are neutral. They receive neither speaker-change benefit credit nor harmful-active-split cost.
9. A hard logical boundary ends the current logical STT/translation turn while keeping VAD state, detector state, and translation context alive. Provider-specific audio commit mechanics are tested later with oracle traces.
10. Primary benefit is reduction of mixed-speaker turn contamination.
11. Primary harm is a hard boundary inside stable same-speaker active speech.
12. No arbitrary false-split cap removes candidates before the development frontier is constructed.
13. Held-out selection uses a frozen multi-point panel, never one ratio-selected family representative.
14. The previously touched Phase 3 held-out manifests are historical validation inputs, not the sole confirmatory held-out set for this experiment.

## 2. Non-goals

- Persistent real-world speaker identity or enrollment
- Speaker names or diarization UI
- Source separation
- Claiming that overlap-onset detection produces clean mono speaker turns
- Long-session memory, speaker return after minutes, or state-drift evaluation
- Production runtime wiring
- Provider comparison before provider-neutral action semantics are validated
- Treating every unmatched detector reaction as user-visible harm
- Choosing a production threshold from transition-pooled recall

## 3. Product baseline and desired hard action

The existing peer path is:

```text
mono capture
  -> VAD SpeechStart / SpeechChunk / SpeechEnd
  -> STT backend audio stream
  -> SpeechEnd requests backend finalization
  -> peer logical turn
  -> translation turn
```

The current `SpeechEnd` path finalizes audio that has already been transmitted. It does
not provide a source-sample retrospective split primitive. The experiment therefore
separates two layers:

1. provider-neutral logical action at a canonical source sample;
2. provider policy that realizes the action without loss or duplication.

The desired hard action is:

```text
logical_finalize(boundary_source_sample)
```

Its required semantics are:

- audio before the boundary belongs to the old logical turn;
- audio at and after the boundary belongs to the new logical turn;
- every source sample is assigned exactly once unless a provider explicitly requires replay, in which case duplicate provider spans are normalized back to one source span;
- VAD detection continues without a synthetic silence reset;
- LS-EEND state continues inside the episode;
- ERes policy state changes only according to its frozen proposal policy;
- translation context survives the logical turn split;
- stale actions from a prior epoch cannot mutate the current epoch.

## 4. Normative vocabulary

### 4.1 Source session

An original public meeting or original synthetic source group. Episodes derived from
one source session remain in one split and one statistical block.

### 4.2 Turn episode

A bounded, source-contiguous evaluation clip containing warm-up context and one or more
scored transitions or negative-control intervals. Episode extraction never changes the
relative source sample order.

### 4.3 Audio epoch

A monotonic canonical sample-coordinate domain. State and pending actions reset at an
epoch boundary. Every episode starts one new epoch.

### 4.4 Active-speaker interval

A contiguous interval labeled with an active-speaker set such as `{A}`, `{A,B}`, or
the empty set. Ambiguous or insufficiently annotated intervals are explicitly marked.

### 4.5 Reference action

A product target derived from the active-speaker timeline. It has an acceptable source
boundary interval, a detector-evidence onset, an action kind, and a scorable interval.

### 4.6 Detector proposal

A causal speaker-change hypothesis. It may later be clustered, suppressed, routed to a
soft marker, or converted into a hard action.

### 4.7 Logical boundary cluster

A causal group of detector proposals believed to describe one underlying change.

### 4.8 Fusion action

A final product-level outcome after detector clustering and VAD interaction.

### 4.9 Boundary location and availability

- `boundary_source_sample`: where the boundary belongs.
- `observed_source_sample_at_emit`: source frontier observed when the decision became usable.
- `emitted_monotonic_ns`: measured runtime emission time when executing live inference.

Location accuracy and causal availability are never collapsed into one timestamp.

### 4.10 Detector progress and safe frontier

Every stateful detector/policy stack also emits:

```text
DetectorProgress {
    audio_epoch
    observed_source_sample
    safe_boundary_frontier_sample
}
```

`safe_boundary_frontier_sample = s` is a causal guarantee that no future detector,
cluster, or debounce output in the current epoch will refer to a hard boundary at or
before `s`. It includes frontend buffering, neural lookback, confirmation, and cluster
debounce. It is monotonic inside an epoch, never exceeds the observed frontier, and
resets with the epoch. Qwen safe drain relies on this guarantee; a heuristic watermark
is not sufficient.

## 5. Turn-episode construction

### 5.1 Public conversational episodes

Public conversational audio is converted into non-overlapping bounded episodes.

Default construction:

- at least 5 seconds of unscored warm-up before the first scored interval when source context permits;
- 10-20 seconds of scored audio;
- maximum total episode duration of 30 seconds;
- at least 3 seconds after the last scored target when source context permits;
- overlapping candidate windows are merged;
- a merged window longer than 30 seconds is split only at an annotated stable same-speaker or silence interval at least 2 seconds away from a scored target;
- no source sample appears in more than one scored episode within the same pool;
- truncated warm-up or tail coverage is recorded, never silently accepted.

The scoring start must not occur while a proposal cluster is pending. If extraction
cannot provide a stable warm-up frontier, the episode is diagnostic-only.

### 5.2 Synthetic episodes

Existing synthetic cases remain complete episodes. New synthetic cases use the same
canonical schema and include explicit warm-up when needed by the detector frontend.

### 5.3 State contract

- reset LS hidden state, ERes anchor state, VAD state, cluster state, and fusion state at episode start;
- feed warm-up audio through all stateful components;
- exclude warm-up actions and references from headline counts;
- retain state across all VAD events within the scored portion;
- finalize pending proposals causally at episode end and label any tail-dependent action;
- never pad neural windows with future or artificial audio.

### 5.4 Statistical grouping

All episodes from one source session share one uncertainty block. Synthetic episodes
share a block when they reuse the same source speakers, utterances, or transformation
seed family.

## 6. Reference-action taxonomy

### 6.1 Clean handoff

```text
{A} -> {B}
```

- action kind: `hard_boundary`
- target point: B onset
- acceptable boundary interval: target point with the declared localization tolerance
- detector-evidence onset: B onset
- primary product case: yes

### 6.2 Gap handoff

```text
{A} -> {} -> {B}, A != B
```

- action kind: `hard_boundary`
- acceptable boundary interval: `[A speech offset, B speech onset]`
- detector-evidence onset: B onset
- any logical boundary inside the silence separates the speakers correctly;
- a VAD boundary available before B onset is valid product separation;
- a speaker-model proposal cannot receive speaker-change evidence credit before B onset;
- localization error is zero inside the acceptable interval and is distance to the nearest interval edge outside it.

This interval-valued definition prevents a correct VAD silence cut from being called a
miss merely because it is more than 500 ms from B onset.

### 6.3 Interruption onset

```text
{A} -> {A,B}
```

- action kind: `soft_overlap_marker`
- target point: B onset
- no hard-turn benefit credit;
- a hard action at this point is reported as `overlap_hard_action`;
- overlap detection is a model diagnostic, not evidence of source separation.

### 6.4 Speaker departure

```text
{A,B} -> {B}
```

- action kind: `state_update`
- no independent hard-boundary target;
- reported only in overlap diagnostics.

### 6.5 Same-speaker pause

```text
{A} -> {} -> {A}
```

- action kind: `neutral_pause`
- no speaker-change benefit credit;
- a boundary inside the pause is neutral unless downstream evidence shows lexical loss;
- a boundary inside stable active A speech is evaluated as a possible harmful split.

### 6.6 Initial/final speech and structural boundaries

Session starts, episode edges, VAD maximum-duration boundaries, and terminal flushes are
structural actions. They are reported separately and do not receive speaker-change
benefit or detector-harm attribution.

### 6.7 Unscored reference intervals

Ambiguous annotation, missing speaker coverage, channel misalignment, or insufficient
word timing creates an explicit unscored interval. Actions in it are counted as
`unscored_action`, not inferred to be correct or harmful.

## 7. Systems under comparison

### 7.1 B0: current VAD-only replay

B0 reproduces the current peer VAD configuration and causal event timing. It contains
no speaker signal.

### 7.2 B1: structural fusion control

B1 uses the same logical-action engine, silence interval handling, maximum-duration
classification, and provider-neutral segmentation mechanics as neural systems, but
receives no neural proposals.

B1 identifies benefits caused by corrected product mechanics rather than speaker evidence.

### 7.3 Frequency-matched segmentation controls

For each frozen neural policy, create deterministic non-speaker controls with the same
per-episode hard-action count:

- uniformly spaced eligible active-speech cuts;
- energy-change peaks without speaker embeddings;
- shuffled neural boundary positions within the same episode while preserving action count and causal availability distribution where possible.

Seeds and selection rules are frozen. If these controls yield comparable contamination
reduction, the apparent benefit is segment shortening rather than speaker specificity.

### 7.4 LS-EEND family

Reuse all verified LS checkpoints. Neural state continues through VAD boundaries inside
an episode. Candidate reducers include current onset/replacement policies plus a causal
activity-state reducer with hysteresis and overlap-aware proposal kinds.

LS proposal kinds:

- `new_track_onset`
- `dominant_replacement`
- `overlap_onset`
- `track_instability`

Track instability is diagnostic and cannot directly create a hard action.

### 7.5 ERes2NetV2 family

Reuse verified standard and W24 checkpoints. Evaluate adjacent and stable-anchor score
streams as proposals. ERes cannot infer overlap from pairwise speaker similarity alone;
its proposals are `speaker_change_unknown`. Hard actions produced during reference
overlap are therefore reported separately rather than credited as clean separation.

ERes policies include:

- adjacent similarity;
- stable anchor;
- confirmed stable anchor;
- bounded episode-local prototype memory where every update is causal.

No cross-episode speaker identity is retained.

## 8. Proposal event contract

Every raw proposal has:

```text
ProposalEvent {
    proposal_id
    family
    checkpoint
    profile_id
    audio_epoch
    proposal_kind
    boundary_source_sample
    observed_source_sample_at_emit
    emitted_monotonic_ns
    confidence
    state_provenance
    debug_evidence
}
```

Invariants:

- `observed_source_sample_at_emit >= boundary_source_sample`;
- events are deterministic for identical audio, model, frontend, and profile;
- confidence increases with change strength within one profile;
- the proposal records every confirmation sample or posterior frame used;
- an event cannot read samples beyond its observation frontier;
- epoch and source-session identity are mandatory;
- proposal generation and product actionization use separate schema versions.

Progress invariants:

- `safe_boundary_frontier_sample <= observed_source_sample`;
- observed and safe frontiers are monotonic inside one epoch;
- the safe frontier covers every still-possible retrospective boundary from frontend,
  neural, confirmation, and open-cluster state;
- no later proposal may name a boundary at or before an already published safe frontier.

## 9. Causal proposal clustering

Clustering is evaluated before VAD fusion.

### 9.1 Policy parameters

```text
cluster_debounce_ms D in {0, 100, 250}
cluster_boundary_radius_ms W in {250, 500}
refractory_ms R in {0, 250, 500}
representative in {first, max_confidence}
```

The full development grid is retained unless preflight runtime forecasting shows that
it is infeasible. Any reduction must be based on profile-independent sentinel traces
and frozen before full development scoring.

### 9.2 Deterministic causal algorithm

1. Order proposals by observation frontier, boundary position, profile ID, and proposal ID.
2. The first eligible proposal opens a cluster at observation `o0` and boundary `b0`.
3. The cluster closes at source observation `o0 + D`.
4. A proposal joins only if it arrives by cluster close, shares the epoch, and its boundary is within `W` of `b0`.
5. Proposals not eligible for the open cluster remain queued in causal order.
6. `first` chooses the opening proposal.
7. `max_confidence` chooses the greatest confidence; ties prefer earlier observation, smaller absolute distance to the cluster median, earlier boundary, then proposal ID.
8. The cluster boundary equals one observed proposal boundary. No non-observed averaged boundary is invented.
9. Cluster availability is the maximum of cluster close and the representative's observation frontier.
10. After emission, proposals arriving before `availability + R` are suppressed and recorded as refractory proposals.
11. A suppressed proposal never becomes a user-visible action but remains evidence.
12. Episode-end closure uses only audio already observed and is labeled `tail_closed`.

The refractory sweep explicitly measures loss on short B turns and rapid A-B-C
handoffs. Refractory is not assumed to be beneficial.

### 9.3 Cluster evidence

Store member proposal IDs, representative reason, suppression reason, cluster open and
close frontiers, boundary spread, confidence distribution, and refractory ownership.

## 10. Hard/soft actionization

Actionization uses only model and causal audio state, never reference labels.

- LS `dominant_replacement` may request a hard action.
- LS `overlap_onset` requests a soft marker.
- LS unstable-track proposals are diagnostic-only.
- ERes `speaker_change_unknown` requests a hard candidate because ERes does not expose overlap state; reference overlap analysis later measures the cost of that limitation.
- a cluster containing incompatible LS kinds follows the frozen priority `overlap_onset > dominant_replacement > new_track_onset > track_instability`;
- hard and soft outputs from one cluster cannot both create product actions.

## 11. Causal VAD fusion

### 11.1 Inputs

Fusion consumes causally ordered VAD events and post-clustering detector candidates.
Ordering uses observation frontier, with stable source/type/ID tie breaks.

### 11.2 Action types

```text
retain_vad
accelerate_or_replace_vad
add_hard_boundary
emit_soft_marker
suppress_detector_duplicate
suppress_vad_duplicate
structural_max_duration
unscored_action
```

Only `retain_vad`, `accelerate_or_replace_vad`, and `add_hard_boundary` create final
hard logical boundaries.

### 11.3 Association parameters

```text
detector_vad_radius_ms V in {250, 500}
same_silence_interval_association in {false, true}
```

### 11.4 Association rules

1. If a detector candidate is near an already emitted VAD boundary within `V`, retain the VAD action and suppress the detector duplicate.
2. If both boundaries lie in the same causally known silence interval and interval association is enabled, treat them as the same logical boundary even when their point distance exceeds `V`.
3. If no prior VAD action associates, emit the detector hard action immediately at its causal availability.
4. If a later VAD boundary associates with an earlier detector action, suppress the duplicate VAD boundary and relabel the detector action `accelerate_or_replace_vad`.
5. A later VAD boundary that ends speech belonging to the new turn is not associated merely because both actions match nearby reference transitions.
6. Soft markers never suppress hard VAD actions.
7. No association uses future reference annotation.
8. Retrospective reporting may classify product benefit using GT, but it cannot alter actions already emitted by the causal policy.

### 11.5 State effects

- hard action: close logical turn, open successor turn, retain VAD and LS neural state;
- soft marker: record speaker-state evidence, no audio cut;
- VAD silence end: retain normal VAD state transition;
- ERes anchor update: controlled solely by the ERes profile, not by GT attribution;
- epoch end: close pending actions and reset all state.

## 12. Reference matching

### 12.1 Eligibility

A final action matches a reference only if:

1. source session and epoch agree;
2. action kind is compatible;
3. boundary is within the acceptable target interval plus localization tolerance;
4. detector-derived evidence was not available before detector-evidence onset;
5. availability meets the declared deadline;
6. ordered one-to-one matching is preserved.

Primary hard localization tolerance is 500 ms. A 250 ms view is also reported.
Availability deadlines are 250, 500, 1000, 1500, and 2000 ms.

For a gap target, a VAD-owned action may be available before B onset. Its product
separation is valid and its availability is reported as pre-existing rather than
rejected as anticipatory. Detector recovery credit still requires observation of B.

### 12.2 Matching objective

Within each epoch, matching maximizes in order:

1. number of compatible matched references;
2. number of B0-retained hard successes;
3. lower causal availability delay;
4. lower interval localization distance;
5. deterministic lexical IDs.

Contamination is not an input to matching. It is recomputed from final segmentation
after matching, so an action cannot receive a more favorable reference assignment
because it happens to improve the primary outcome.

B0 actions are replayed independently. Neural systems do not receive recovery credit
by reassigning a B0 success, but may receive acceleration credit when the same logical
target becomes usable earlier.

### 12.3 Product attribution

Every final action/reference outcome is one of:

- `retained_b0_success`
- `recovered_b0_hard_miss`
- `accelerated_b0_success`
- `correct_soft_marker`
- `duplicate_hard_boundary`
- `harmful_active_split`
- `lexical_split`
- `neutral_pause_split`
- `structural_split`
- `overlap_hard_action`
- `late_target_action`
- `unscored_action`
- `hard_miss`
- `soft_miss`

These categories are stored per action, not derived only from aggregate subtraction.

## 13. Primary benefit: mixed-speaker turn contamination

### 13.1 Logical segmentation

Sort final hard boundaries by source position. Episode edges and hard boundaries define
logical segments. Actions with identical boundary positions form one boundary.

### 13.2 Contamination algorithm

For each logical segment:

1. intersect the segment with reference active-speaker intervals;
2. exclude silence, ambiguous intervals, and overlap intervals from the primary contamination numerator;
3. find the first substantive singleton speaker in the segment;
4. before any different singleton speaker appears, that speaker owns the turn;
5. once a different singleton speaker appears, all subsequent singleton speech until the next hard boundary is contamination, including a later return of the original speaker;
6. do not double-count source samples;
7. report excluded overlap and unscored duration separately.

This makes a segment containing `A -> B -> C` charge B and C speech as contamination
of A's logical turn. A premature split before the actual handoff does not receive false
benefit: if the successor segment still begins with A and later contains B, B remains
contamination.

### 13.3 Reported benefit metrics

- primary contamination ratio: contaminated singleton-speech source samples divided by
  all scorable singleton-speech source samples;
- primary paired effect: candidate contamination ratio minus B1 contamination ratio
  within each source block, reported as percentage-point change with negative meaning
  improvement;
- contamination milliseconds per episode and source session;
- contamination seconds per five-minute source session;
- contamination milliseconds per active-speech hour;
- absolute and relative reduction from B0 and B1;
- logical turns containing at least 100, 250, and 500 ms of a second singleton speaker;
- primary mixed-turn rate at the predeclared 250 ms substantive threshold;
- clean and gap hard targets remaining merged;
- recovered B0 misses versus accelerated B0 successes;
- contamination avoided per final detector-created hard action.

Transition recall is secondary diagnostic evidence.

## 14. Primary harm: same-speaker fragmentation

Only unmatched final hard actions can receive false-split harm labels.

### 14.1 Harmful active split

An action is `harmful_active_split` when:

- the boundary lies inside a singleton active-speaker interval;
- at least 200 ms of the same singleton speaker is continuously active on both sides;
- no compatible hard reference is matched;
- the action is not a structural maximum-duration boundary;
- the interval is fully scorable.

Report sensitivity at 100 and 300 ms guards without changing the primary 200 ms label.

### 14.2 Lexical split

Where trusted word timing exists, a boundary is a lexical split when it lies inside a
word interval with at least 20 ms of that word on both sides. Word timing source,
revision, and coverage are recorded. Missing word timing produces `not_observable`, not
an inferred negative.

### 14.3 Neutral pause split

An unmatched action inside silence between two singleton spans of the same speaker is
neutral. It is reported separately and cannot improve speaker-change benefit.

### 14.4 Duplicate hard boundary

More than one post-fusion hard boundary attributed to one hard reference is a duplicate.
Clustering should suppress most duplicates; remaining duplicates expose fusion failure.

### 14.5 Overlap hard action

A hard action inside an overlap target or stable overlap interval is reported as an
overlap hard action. It is neither clean-handoff benefit nor same-speaker harm.
Downstream replay determines whether it is useful or harmful.

### 14.6 Fragmentation metrics

- harmful active splits per five-minute source session;
- harmful active splits per active-speech hour;
- lexical splits per hour of word-aligned speech;
- duplicate hard actions per source hour;
- neutral pause splits;
- detector-created fragments with less than 250, 500, and 1000 ms active speech;
- p10/p50/p90 final segment duration and active-speech duration;
- number of consecutive fragments owned by the same speaker;
- legacy unmatched boundary count, labeled historical diagnostic only.

## 15. Timing and runtime metrics

Report separately:

- interval localization error;
- signed point error for clean and overlap targets;
- causal availability delay;
- event lookback;
- cluster debounce delay;
- VAD association delay;
- wall-clock model service time;
- end-to-end scheduling completion delay;
- real-time factor;
- final backlog;
- peak resident memory;
- model load time;
- cache-hit and cache-miss execution time.

No negative detector availability delay is allowed. Pre-existing gap VAD actions are a
separate valid product category rather than negative detector delay.

## 16. Data pools

### 16.1 Historical development evidence

Preserve the completed 1,369 Phase 3 rows and verified raw caches. They are inputs for
corrected replay, not accepted corrected results.

### 16.2 Controlled diagnostic development pool

Use existing synthetic data and add only underrepresented strata after inventory.

Different-speaker gap bins align to the 32 ms product chunk where practical:

```text
0, 32, 64, 96, 160, 256, 384, 800 ms
```

Turn-duration bins:

```text
0.25, 0.50, 0.75, 1.50, 3.00 seconds
```

Negative and stress factors:

- same-speaker concatenation with identical gap/duration recipes;
- gain changes of approximately +/-6 dB and +/-12 dB;
- codec/bandwidth transforms already authorized by the corpus pipeline;
- noise at frozen SNR strata;
- short silence and non-speech spans;
- available laughter/non-speech vocalization annotations;
- prosody or speed transforms only when they preserve speaker and annotation validity.

Positive and same-speaker negative recipes are matched by source duration, gap, stress,
and transform where possible.

### 16.3 Public conversational episode pool

Use authorized AMI and AliMeeting sources already in experiment scope. Do not add ICSI,
AISHELL-4, or another corpus merely to create long-session coverage. Additional corpora
require a separate coverage finding and authorization.

Before audio evaluation, build a metadata-only inventory containing:

- independent source-session count;
- speaker-connected component count;
- source and scored duration;
- hard clean/gap targets;
- overlap soft targets;
- same-speaker pause intervals;
- stable same-speaker active exposure;
- B0-separated and B0-missed hard targets;
- short-turn distribution;
- channel/microphone condition;
- word-alignment coverage;
- language and corpus;
- model-training overlap risk.

Per source session, deterministic hash-stratified sampling prevents meetings with many
transitions from dominating. The default maximum is 12 hard-positive episodes and 12
negative episodes per source session, subject to non-overlap and coverage constraints.
All eligible counts before sampling are preserved.

The inventory freezes the attainable independent-block count before audio execution.
A confirmatory pooled AMI-plus-AliMeeting product claim requires at least eight
independent contributing blocks from each corpus after participant-component grouping.
A corpus with four to seven blocks is corpus-exploratory; fewer than four blocks permits
descriptive rows only. If one corpus misses this gate, the result cannot be generalized
across both corpora even when the equal-corpus pooled point estimate is favorable.

### 16.4 Pool roles

- `diagnostic_dev`: controlled synthetic and public development episodes used for policy construction;
- `frontier_dev`: independently blocked development episodes used for final curves;
- `historical_validation`: previously touched Phase 3 held-out sources, never used for confirmatory claims;
- `confirmatory_heldout`: newly selected unused source sessions and speaker groups, inaccessible until freeze.

## 17. Split and leakage contract

Keep together:

- complete original source session;
- meeting series and related submeetings;
- recurring participant connected components when discoverable;
- all channel views and derivatives of one recording;
- original and transformed synthetic audio;
- all utterances from one synthetic source speaker when speaker-disjointness is claimed;
- every episode sharing any source sample.

Audit checkpoint training provenance. AMI-trained LS results on AMI are reported as
in-domain model evidence, not unseen-corpus generalization. Corpus and model-domain
results remain stratified.

Split artifacts contain group graph hashes and fail closed on cross-split overlap.

Confirmatory held-out audio paths and annotations cannot be opened by the runner until
a self-hashed frozen contract exists. Any earlier access moves the affected session to
historical validation and requires a new held-out group.

## 18. Model-signal diagnostics

These diagnostics determine whether poor product behavior comes from the neural signal
or from proposal/fusion policy. They do not select held-out thresholds alone.

### 18.1 LS diagnostics

- pin the exact 16 kHz-to-8 kHz resampler implementation and configuration;
- measure filter/group delay, internal buffering, chunk-edge behavior, and flush/tail
  behavior and map all LS input/output frames back to canonical 16 kHz samples;
- compare whole-file and chunked-streaming resampler/frontend outputs on a fixed parity
  set, with the streaming path authoritative for product timing;
- include resampler, feature frontend, neural, reducer, confirmation, and cluster delay
  in event availability and the safe frontier;
- posterior trajectories around clean, gap, overlap, and same-speaker controls;
- new-track onset timing;
- dominant-replacement timing;
- active-set hysteresis behavior;
- track flicker per active-speech minute;
- overlap-state precision/recall as a soft diagnostic;
- causal oracle hard-boundary upper bound using only already available posterior frames;
- batch/stream and source-sample timing parity;
- continuous-within-episode versus VAD-reset ablation.

No long-term identity or speaker-return conclusion is allowed.

### 18.2 ERes diagnostics

- pin and parity-check the resampler and feature frontend separately from embedding-model
  parity, including chunked-streaming behavior used by the product-shaped replay;
- same-speaker and different-speaker cosine distributions;
- ROC-AUC and EER by corpus, language, window length, and stress;
- pure-window versus transition-mixed-window behavior;
- anchor drift within an episode;
- consecutive-candidate mutual similarity;
- reaction to gain/noise/codec/prosody controls;
- causal oracle hard-boundary upper bound using available embeddings;
- frontend/export parity for sampled windows.

### 18.3 Signal-level stop condition

For every frozen scalar signal, calculate ROC-AUC for different-speaker hard targets
versus matched same-speaker acoustic negatives. Compare it with the strongest declared
acoustic-only score on the identical examples. LS overlap scores receive a separate
overlap-versus-singleton diagnostic and cannot satisfy the hard-target gate.

For each family, use session-block bootstrap on the paired AUC difference:

- `signal_go`: at least one hard-target score has a 95% lower bound greater than zero;
- `signal_limited`: the point estimate is greater than zero but the interval includes
  zero, or the comparison has fewer than eight independent blocks;
- `signal_stop`: every hard-target score has a 95% upper bound at or below zero.

`signal_go` receives the full predeclared policy grid. `signal_limited` receives the
same-proposal policy ladder and one sentinel profile per policy family, but no expanded
combinatorial sweep. `signal_stop` receives only B0/B1, the raw diagnostic report, and
the no-neural-policy control. This gate controls compute; it is not itself a product
selection claim.

## 19. Required ablations and falsification tests

### F1. Same-proposal policy ladder

Replay identical proposals through:

1. naive proposal-as-cut historical policy;
2. clustering only;
3. clustering plus refractory;
4. plus VAD association;
5. full hard/soft fusion.

Attribute changes only to the policy stage that changed.

### F2. Same-speaker acoustic negatives

Compare proposal/action rates on matched different-speaker and same-speaker gain,
prosody, codec, noise, and pause conditions.

### F3. Frequency-matched segmentation

Compare every finalist to non-speaker controls with matched hard-action counts.

### F4. Boundary-position shuffle

Shuffle boundary positions within eligible regions while preserving action count and
availability delay. Similar benefit after shuffling falsifies speaker-specific timing.

### F5. Recovery versus acceleration

Report contamination reduction caused by B0-miss recovery separately from earlier
availability of a B0-owned gap boundary.

### F6. Overlap-separated result

Remove all overlap references from the hard headline and report overlap soft markers
and overlap hard actions separately.

### F7. Session robustness

Use leave-one-source-session-out analysis and session-block bootstrap. A conclusion
reversed by removing one session is exploratory.

### F8. Oracle logical-action ceiling

Apply exact hard reference actions at controlled location errors and availability
delays. If the provider-neutral turn assembler cannot conserve audio or reduce
contamination, detector evaluation stops.

### F9. Short-turn refractory stress

Measure loss for 250, 500, and 750 ms successor turns under each refractory setting.

### F10. Same-speaker pause neutrality

Verify that treating pause splits as neutral cannot hide lexical or active-speech
fragmentation.

## 20. Development search and frontier

### 20.1 Search order

1. verify raw model/cache parity;
2. generate raw proposals for every existing detector profile;
3. compute signal diagnostics;
4. replay clustering grid;
5. replay hard/soft actionization;
6. replay VAD fusion grid;
7. calculate final product metrics;
8. construct family frontiers.

Neural inference is cached before policy search. No policy changes the model input or
causal observation frontier.

### 20.2 Frontier dimensions

A profile is dominated only if another profile is no worse in all of:

- contamination reduction;
- harmful active splits;
- lexical splits where observable;
- duplicate hard boundaries;
- hard-target recall at every causal deadline;
- causal delay;
- runtime cost;

and is strictly better in at least one. Missing lexical coverage cannot be treated as
zero lexical harm.

The complete integer frontier is preserved. Hourly and five-minute rates are views of
raw counts, not pre-frontier caps.

### 20.3 Cross-family matched-harm comparison

Use the union of achieved integer harmful-active-split counts on the same development
exposure. At each allowance, select the greatest contamination reduction without
exceeding the allowance. Report exact achieved cost rather than pretending continuous
interpolation.

This comparison does not eliminate high-cost profiles; it makes LS and ERes comparable
at observed product harm.

### 20.4 Frozen low/medium/high panel

Freeze up to three unique profiles per family:

- `low_harm`: among profiles with positive contamination reduction, minimize harmful active splits, then maximize contamination reduction;
- `frontier_knee`: normalize harmful-active-split count to `x` where lower is better and
  contamination reduction to `y` where higher is better, draw the chord from the
  low-harm endpoint to the maximum-benefit endpoint, and maximize the signed vertical
  gain `y - y_chord(x)` among frontier points between those endpoints;
- `maximum_benefit`: maximize contamination reduction, then minimize harmful active splits and duplicates.

Knee ties prefer lower `x`, then greater `y`. All remaining ties prefer fewer lexical
splits, lower p95 causal delay, lower runtime cost, and lexical profile ID. If either
axis has zero range, the endpoints coincide, or no interior point has positive signed
gain, omit the knee rather than inventing a medium point. Collapse duplicate profiles
and record the reason.

The panel describes the frontier. It is not a product false-split budget and does not
declare a winner.

## 21. Statistical plan

### 21.1 Primary uncertainty unit

The source session or synthetic source-connected block is primary. Transitions and
episodes are not independent uncertainty units.

### 21.2 Estimation

- aggregate all episodes within each block first;
- compute paired B0/B1/candidate differences within block;
- use 10,000 deterministic block-bootstrap replicates with a frozen seed;
- keep recurring-participant connected components in one resampled block;
- report median paired difference and percentile 95% interval;
- report corpus macro summaries and equal-corpus pooled summaries;
- include raw transition/action/time counts;
- use transition-pooled micro metrics only as descriptive diagnostics.

If fewer than eight independent blocks contribute to a comparison, label its interval
exploratory. If fewer than four contribute, report raw block results without a primary
confidence interval.

No transition-level Wilson interval is used as the primary family evidence.

### 21.3 Multiplicity and interpretation

No uncorrected collection of per-transition p-values is used. The frozen panel is
reported completely. Family claims rely on consistent paired effect direction across
blocks/corpora and bootstrap intervals, not the best isolated profile result.

## 22. Held-out discipline

Before any confirmatory held-out path is opened, freeze a self-hashed artifact binding:

- episode and split manifests;
- source and annotation hashes;
- model/checkpoint/frontend hashes;
- proposal profiles;
- clustering/debounce/refractory;
- hard/soft action mapping;
- VAD fusion;
- scoring and contamination code hashes;
- panel profile IDs;
- bootstrap seed and block graph;
- expected session and episode counts.

Held-out runs are resumable per source session. A pooled summary cannot be written until
all expected sessions complete and verify. Evidence is namespaced by frozen-contract
hash. No threshold, action rule, clustering rule, or panel membership changes after
access.

Failure of a frozen point remains visible.

## 23. Provider-neutral oracle validation

Before provider-specific replay, implement a canonical PCM turn assembler.

Oracle grid:

```text
availability delay: 250, 500, 750, 1000, 1250, 1500, 2000 ms
boundary offset: -200, -100, 0, +100, +200 ms
holdback: 0, 250, 500, 750, 1000, 1500, 2000 ms
```

Verify for every case:

- source sample conservation;
- zero unintended duplication;
- old-turn/new-turn ownership;
- contamination remaining after the action;
- audio that became unrecoverable before action availability;
- fragment duration;
- finalization latency;
- behavior when an action arrives after the boundary has left the holdback ring;
- Qwen-style `SpeechEnd` drain that releases held PCM only after the detector safe
  frontier covers the release region;
- bounded safe-drain timeout and a separately labeled fallback path;
- epoch reset and stale-action rejection.

This phase validates the lifecycle mechanics, not the detector.

## 24. Provider-specific replay

Provider-specific experiments consume exactly the same frozen action traces.

### 24.1 Deepgram family

- maintain explicit source-to-provider span mapping;
- normalize provider timestamps into source time;
- record reconnect epochs and every bridge/resend span;
- allow one source range to map to multiple provider epochs when reconnect bridging
  resends audio, while provider timestamps remain meaningful only inside their epoch;
- deduplicate words after source normalization;
- include a deterministic fake reconnect with a boundary inside or adjacent to the
  bridged region;
- verify finalize behavior and transcript ownership.

### 24.2 Qwen realtime family

- do not invent word timestamps;
- use canonical source-sample PCM holdback;
- split unsent PCM at the action boundary;
- commit the old prefix and retain the suffix for the next turn;
- at VAD `SpeechEnd`, continue detector progress and drain held PCM only when
  `safe_boundary_frontier_sample` proves no later boundary can target it;
- record bounded drain timeout/fallback separately from ordinary completion;
- choose holdback from measured finalist lookback plus scheduling margin.

### 24.3 Soniox family

- evaluate native speaker metadata as its own baseline;
- compare local detector fusion only if it adds value not already supplied natively;
- allow the conclusion that no local detector is appropriate for Soniox.

Provider model IDs, decoding parameters, credentials mode, region, and API/runtime
versions are frozen in the provider-phase contract. Detector/fusion conclusions remain
provider-neutral until this phase succeeds.

Provider comparisons change one factor at a time wherever possible. Audio-input shaping,
detector trace, finalization policy, and transcript/speaker-metadata interpretation are
separate axes. Any arm that changes more than one is labeled factorial and cannot
attribute its effect to a single policy component.

## 25. Downstream STT metrics

For provider/backend configurations with suitable references, report paired:

- WER or CER;
- final transcript latency;
- time from true hard handoff to old-turn final;
- transcript text assigned to the wrong logical turn;
- audio/text loss and duplication;
- number of final transcript fragments;
- fragment active-speech duration;
- lexical split count;
- hallucination/empty-final count;
- reconnect correctness;
- translation-input turn contamination where deterministic replay is available.

Boundary-level improvement that worsens transcript integrity is not product-positive.

## 26. Runtime and feasibility

Measure on the same declared hardware and process configuration:

- CPU model and logical cores;
- ONNX Runtime version and provider;
- thread counts;
- model load and warm-up time;
- mean/p50/p95 service time;
- RTF;
- peak RSS;
- cache size;
- scheduling backlog under one real-time stream;
- two-stream stress as secondary evidence if the product can run self and peer together.

Policy replay cost is separated from neural inference. Large cache or runtime cost may
exclude a product recommendation even when metrics are favorable, but it does not erase
signal-level evidence.

## 27. Cache and provenance contract

### 27.1 Historical preservation

Phase 3 rows, evidence, and partial held-out artifacts are never overwritten. They are
referenced only by direct hash and historical label.

### 27.2 Raw neural caches

LS cache identity binds:

- checkpoint and sidecar hashes;
- frontend and resampler contract;
- source audio hash;
- episode manifest hash;
- model input/output tensor contract;
- capture payload hash.

ERes cache identity additionally binds every window coordinate and embedding payload
hash. Legacy import requires deterministic sampled recomputation with declared numeric
tolerance.

### 27.3 Result artifacts

New results live under `results/turn_episode_v1/`. Every JSON artifact contains a
canonical content hash. Row files also record direct byte SHA-256. Partial runs are
stored per source session and cannot masquerade as complete summaries.

Minimum artifacts:

```text
restart_contract.json
coverage_inventory.json
episode_manifest_dev.json
episode_manifest_heldout.json
proposal_contract.json
fusion_contract.json
development_rows.jsonl
development_summary.json
frozen_panel.json
heldout_session_evidence/<frozen-hash>/<session>.json
heldout_summary.json
oracle_provider_neutral.json
provider_replay/<provider>/<frozen-hash>.json
decision.json
```

## 28. Scientific contract tests

Code-polish tests are not a gate. The following scientific invariants are mandatory.

1. No proposal reads beyond its observation frontier.
2. No cluster contains a proposal arriving after cluster close.
3. Refractory suppression is deterministic and causal.
4. No final action belongs to two clusters.
5. No final hard boundary is duplicated at one source sample.
6. Matching is ordered and one-to-one.
7. Gap interval matching accepts any boundary inside the annotated silence.
8. A detector proposal before B onset receives no gap speaker-change evidence credit.
9. A pre-existing VAD gap boundary remains valid product separation.
10. Overlap soft references cannot raise hard-boundary headline recall.
11. Warm-up actions cannot enter scored counts.
12. Unscored intervals cannot enter benefit or harm numerators.
13. Contamination source samples are never double-counted.
14. A premature boundary that leaves A and B in the successor turn does not receive contamination-reduction credit.
15. Harmful active split requires the same singleton speaker on both guarded sides.
16. Missing word timing is not treated as absence of lexical harm.
17. B0 and B1 use the same episode/reference/fusion infrastructure as neural systems.
18. Frequency-matched controls exactly match declared action counts.
19. Bootstrap resamples blocks, not transitions.
20. Cross-split source, speaker, recording, and transformation overlap fails closed.
21. Held-out cannot open without a valid frozen self-hash.
22. Incomplete held-out sessions cannot produce a decision.
23. Provider-neutral assembly conserves every source sample exactly once.
24. Stale epoch actions cannot mutate the next epoch.
25. Detector safe frontier is monotonic, conservative, and never violated by a later event.
26. Qwen-style safe drain does not release a region before the safe frontier covers it.
27. Deepgram reconnect mapping permits repeated source spans without duplicate normalized words.

## 29. Execution phases and gates

### Phase 0: checkpoint and contract freeze

Deliverables:

- exact restart SHA and dirty-worktree inventory;
- historical artifact hash ledger;
- this plan's self-hash;
- reference/action/fusion schema;
- detector progress and safe-frontier schema;
- B0 logical-finalize replay description.

Gate: no held-out access and no new model execution before the action/scoring contract
passes invariant tests.

### Phase 1: metadata coverage inventory

Deliverables:

- all pool/session/speaker/condition counts;
- split-leak graph;
- B0-separated versus B0-missed hard targets;
- negative exposure inventory;
- word-timing observability;
- exact data-gap list and compute/storage forecast.

Gate: data additions must target an observed gap, not an arbitrary hour count.

### Phase 2: episode/reference implementation

Deliverables:

- bounded episode builder;
- interval-valued hard references;
- hard/soft/neutral/unscored timelines;
- deterministic manifests;
- sampled waveform/annotation audit.

Gate: all reference and split invariants pass, and sampled episodes agree with source
annotations.

### Phase 3: provider-neutral logical-action oracle

Deliverables:

- canonical PCM turn assembler;
- delay/offset/holdback grid;
- source-sample conservation and ownership evidence;
- safe-frontier drain and timeout evidence;
- contamination ceiling and unrecoverable-late curve.

Gate: logical actions must conserve audio and reduce oracle contamination before any
neural family consumes a full policy sweep or confirmatory held-out access.

### Phase 4: raw signal diagnostics

Deliverables:

- LS posterior/reducer/oracle report;
- LS 16-to-8 kHz streaming frontend timing and source-mapping parity;
- ERes calibration/anchor/oracle report;
- acoustic-negative controls;
- signal-level go/limited/stop disposition.

Gate: only signal-positive or diagnostic-limited families enter the full policy sweep.

### Phase 5: corrected proposal and fusion replay

Deliverables:

- complete causal proposal evidence;
- clustering/refractory rows;
- VAD fusion actions;
- policy-ladder ablations;
- corrected rescoring of historical development caches.

Gate: independent recomputation agrees on actions, contamination, harm, and timing.

### Phase 6: development frontier and freeze

Deliverables:

- full product-metric frontiers;
- matched-harm comparisons;
- low/knee/high panels;
- self-hashed frozen contract;
- no-selection explanation when no positive frontier exists.

Gate: every expected development row completes before freeze.

### Phase 7: confirmatory held-out

Deliverables:

- per-session paired evidence;
- session-block uncertainty;
- complete frozen panel including poor points;
- clean/gap headline and separate overlap result;
- runtime evidence.

Gate: no pooled conclusion from partial session completion.

### Phase 8: provider-specific frozen trace replay

Deliverables:

- exact same frozen real-detector traces;
- transcript integrity and latency;
- provider-specific feasibility;
- native Soniox comparison where applicable.

Gate: boundary metrics alone cannot yield a product-positive recommendation.

### Phase 9: independent verification and conclusion

Deliverables:

- coordinator recomputation from per-session evidence;
- provenance/cache/split/timing audit;
- failure-example audit selected by frozen rule;
- explicit model-signal, policy, provider, and product conclusions;
- implementation handoff or no-selection report.

## 30. Failure-example selection

Examples are selected without manual cherry-picking.

For each frozen profile and corpus, retain deterministic examples for:

- five largest contamination regressions;
- five largest contamination improvements;
- five highest-confidence harmful active splits;
- five duplicate clusters with the most proposals;
- five late but accurately localized targets;
- five clean/gap misses with strongest model evidence;
- five overlap hard actions;
- all audio loss/duplication violations.

Ties use source-session ID, episode ID, and source sample. Held-out examples are selected
only after aggregate calculation by this frozen rule.

## 31. Decision framework

The conclusion is layered.

### 31.1 Signal-positive

The family contains causal speaker-specific evidence beyond acoustic/frequency controls.

### 31.2 Policy-positive

A causal clustering/fusion profile reduces contamination relative to B1 at matched
harm on development and preserves effect direction across held-out source blocks.

### 31.3 Provider-feasible

The frozen actions can be realized without unacceptable audio/text loss, duplication,
or finalization delay.

### 31.4 Product candidate

At least one frozen point:

- has positive held-out contamination reduction with session-block evidence;
- is better than frequency-matched controls;
- has an explicitly reported harmful-active-split rate;
- does not depend on overlap being counted as a clean hard success;
- does not materially worsen downstream transcript integrity;
- meets runtime/backpressure requirements;
- binds exact model, frontend, proposal, cluster, fusion, and provider policy.

An exact product tolerance for harmful splits may remain a product-owner decision. The
experiment still reports the full matched-harm frontier and can recommend dominance or
no selection without inventing a cap.

### 31.5 Required negative outcomes

Use explicit outcomes when appropriate:

- `signal_negative`
- `signal_positive_policy_negative`
- `policy_positive_provider_infeasible`
- `provisional_product_candidate`
- `no_local_detector_selected`

No production-ready claim is allowed without representative product-domain audio.

## 32. Independent verification and worker governance

The coordinator owns design, implementation, and acceptance.

Each execution phase uses a fresh paid DeepSeek OpenCode worker. The worker:

- receives exact commands and timeout requirements;
- performs execution only;
- does not edit source, manifests, contracts, or results;
- does not retry or restart a failed full run without a new coordinator instruction;
- reports `worker_done` or escalation;
- is monitored with ten-minute event waits rather than one-minute polling.

Worker reports are not evidence. The coordinator independently verifies:

- file and self hashes;
- expected session/profile counts;
- split completeness;
- cache identities and sampled payload recomputation;
- causal timing;
- cluster membership;
- final action taxonomy;
- contamination sample accounting;
- harmful-split classifications;
- session-block bootstrap inputs;
- aggregate recomputation;
- provider audio conservation.

Experiment-scientific tests are strict. Code style, docstrings, and unrelated polishing
are not acceptance gates.

## 33. Architecture boundary

All work remains under `experiments/speaker_turn_boundary`. Production owners,
composition, settings, provider adapters, and runtime lifecycle code are read as the
behavioral baseline but are not modified during detector/fusion experimentation.

Provider oracle adapters are experiment doubles or replay harnesses. Any later
production integration requires a separate reviewed implementation task. This plan
introduces no intended production architecture change.

## 34. Immediate implementation order

1. Record SHA, dirty state, and historical artifact hashes.
2. Create `turn_episode_v1` schemas and scientific contract tests.
3. Build metadata-only coverage inventory without opening confirmatory held-out audio.
4. Implement interval-valued reference actions and bounded episode extraction.
5. Implement contamination and same-speaker harm scoring against hand-built fixtures.
6. Implement and pass the provider-neutral PCM oracle grid.
7. Implement causal cluster/refractory replay.
8. Implement VAD fusion and complete action evidence.
9. Verify reuse of LS captures and ERes embeddings.
10. Correctly rescore existing development evidence.
11. Decide data additions from inventory findings.
12. Build the full development frontier and frozen panel.
13. For each phase that needs model or replay execution, dispatch a new paid DeepSeek
    OpenCode worker with coordinator-authored commands; independently verify its output
    before accepting the phase gate.
14. Open confirmatory held-out only after the oracle, development, and freeze gates pass.

Until steps 1-11 complete, the accepted experiment conclusion is:

> Useful raw speaker-change signal exists in both families, but no corrected
> product-level detector/fusion selection has been made.
