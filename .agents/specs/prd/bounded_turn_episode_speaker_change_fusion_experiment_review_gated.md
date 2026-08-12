# Bounded turn-episode speaker-change fusion experiment

## 0. Status, authority, and purpose

Status: reviewed normative experiment plan. Phases 0-4 are accepted and closed. Phase 5 is
paused before compact design regeneration and execution; no corrected product-level
detector/fusion selection has been made.

### Owner override for Phase 5 execution (2026-08-11)

On 2026-08-11 the owner explicitly approved the locally verified revision-8 repairs as a
sufficient owner override for Phase 5 execution and, for this Phase 5 only, replaced the
Section 29 accepted pre-execution review requirement with that override. No re-review was
requested or performed. The override covers exactly the two revision-8 repairs:

1. observable empty word timing (`word_intervals=[]`) remains an observable lexical
   negative and only missing timing (`word_intervals=None`) increments
   `lexical_not_observable`;
2. the typed historical aggregate explicitly includes B0 and B1, all 408 baseline case
   identities, the complete contamination/harm metric field set, ordered
   identity/action/score digests, and the exhaustive B0/B1 equivalence receipt; the
   redundant unprojected historical-correction output is removed.

This document's authority SHA-256 is re-pinned below to the hash of this amended
document. The re-pinned hash supersedes `bad637985e6ea2b82b0ac0e233b99ca7364d324dd2c5a38ec446b95a8604fbc4`
for all Phase 5 gate checks. The revision-7 verdict `repair_required` remains recorded
for revision 7; the override does not claim an independent accepted verdict for
revision 8.

Current compact-Phase-5 amendment source:
`.agents/specs/prd/drafts/bounded_turn_episode_speaker_change_fusion_experiment_review_gated.source.r1.md`.
Document review verdict for this amendment: ready.

Authority order:

1. Explicit user decisions in the experiment thread.
2. The approved compact Phase 5 source snapshot named above.
3. `.agents/specs/prd/speaker_change_turn_boundary_experiment_handoff_en(1).md`.
4. This normative experiment plan.
5. GitHub issue #51 as product/history context where it does not conflict with the items above.
6. Verified raw artifacts from accepted Phases 0-4.
7. Older reports and analyses as historical evidence only.

If a lower-authority artifact conflicts with the action, scoring, sampling, or selection
contract in this plan, the lower-authority artifact is not normative for the new run.
Historical artifacts may be reused only as evidence or cached model output after their
identity and causal semantics are verified.
The compact Phase 5 source and corresponding explicit owner decisions supersede every
conflicting exhaustive-sweep, W24-replay, full-verifier, and runtime instruction in the
older handoff or historical artifacts.

The experiment answers a product question:

> Can causal speaker-change evidence, after proposal stabilization and VAD fusion,
> reduce speech from different speakers being finalized in one STT/translation turn
> without causing too many harmful or excessive same-speaker splits?

This is not a general diarization benchmark. Production wiring is out of scope. The
result may recommend a detector and policy for a later implementation task, but it may
also conclude that no local detector is ready.

## 1. Decisions fixed by this plan

The following decisions are frozen before implementation.

1. The primary audio condition is one mono mixed-audio source timeline at 16 kHz.
2. The evaluation unit is a bounded turn episode, not a complete meeting.
3. Episode-local reset plus warm-up is permitted for scored evaluation only after the state-equivalence contract in Section 5.4 passes for the relevant family/profile class. If it fails, source-prefix state snapshots or source-prefix replay are required.
4. VAD utterance boundaries do not reset LS-EEND neural state inside an episode.
5. A detector proposal is never scored as a product cut before causal clustering and fusion.
6. Clean and gap speaker handoffs are hard-boundary targets.
7. Interruption/overlap onset is a soft-marker target and is excluded from the clean/gap hard-turn headline.
8. Same-speaker pauses receive no speaker-change benefit credit and no severe active-speech-split label, but detector-created pause splits remain a separately measured fragmentation cost.
9. A hard logical boundary ends the current logical STT/translation turn while keeping VAD state, detector state, and translation context alive. Provider-specific audio commit mechanics are tested later with oracle traces.
10. Primary benefit is reduction of mixed-speaker turn contamination on the clean/gap headline stratum defined in Section 13.
11. Primary severe harm is a hard boundary inside stable same-speaker active speech. Harm flags are independent of reference-match benefit attribution.
12. No arbitrary false-split cap removes candidates before the development frontier is constructed.
13. Held-out selection uses a frozen multi-point panel, never one ratio-selected family representative.
14. The previously touched Phase 3 held-out manifests are historical validation inputs, not the sole confirmatory held-out set for this experiment.
15. Target-enriched episode pools are used for comparative efficacy and failure analysis, not to infer natural five-minute/session rates. Natural-rate estimates come only from the unbiased natural-exposure pool defined in Section 16.
16. Newly recorded private or product-domain conversational audio is not required by this experiment. Its absence is an external-validity limitation, not an automatic experiment failure.
17. Independent verification proves complete artifact identity, completeness, aggregate
    arithmetic, and outcome-critical invariants without replaying the entire Phase 5
    computation a second time. Raw/derived trace reconstruction uses a frozen
    deterministic stratified audit sample plus all mandatory sentinels and failure
    examples. Broad harness regression and duplicate full-grid recomputation are not
    experiment acceptance gates.
18. Compact Phase 5 is planned as a 2-3 hour execution on the declared Phase 4 CPU
    environment. Including exact design regeneration, mandatory pre-execution review,
    execution, sampled independent audit, and the owner report, the expected elapsed
    time after explicit Goal resume is 3-5 hours. This is a planning envelope rather
    than accepted scientific evidence; a regenerated pre-execution forecast materially
    above three execution hours must be reported before execution.

## 2. Non-goals

- Persistent real-world speaker identity or enrollment
- Speaker names or diarization UI
- Source separation
- Claiming that overlap-onset detection produces clean mono speaker turns
- Production runtime wiring
- Provider comparison before provider-neutral action semantics are validated
- Treating every unmatched detector reaction as user-visible harm
- Treating every same-speaker pause split as zero-cost
- Choosing a production threshold from transition-pooled recall
- Inferring natural five-minute false-split or contamination rates from target-enriched episodes
- Requiring newly recorded private or product-domain conversational audio as a gate to complete the experiment

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

Each scored episode is tagged as one of:

- `hard_only`: the scored region contains clean/gap hard targets and no overlap reference or stable overlap interval;
- `overlap_present`: the scored region contains an overlap reference or stable overlap interval;
- `negative_only`: no different-speaker hard target is present.

The primary clean/gap contamination headline uses only `hard_only` exposure.
`negative_only` episodes are reserved for fragmentation/false-action analysis, and
`overlap_present` episodes are reported separately.

### 5.2 Synthetic episodes

Existing synthetic cases remain complete episodes. New synthetic cases use the same
canonical schema and include explicit warm-up when needed by the detector frontend.

### 5.3 Episode state contract

- reset LS hidden state, ERes anchor state, VAD state, cluster state, and fusion state at episode start only when Section 5.4 permits reset-based evaluation;
- feed warm-up audio through all stateful components;
- exclude warm-up actions and references from headline counts;
- retain state across all VAD events within the scored portion;
- finalize pending proposals causally at episode end and label any tail-dependent action;
- never pad neural windows with future or artificial audio.

### 5.4 State-equivalence contract

Bounded extraction must not silently change the detector state seen at a scored target.
Before reset-plus-warm-up episodes are accepted for a family/profile class, run a fixed
state-parity set in two modes:

1. `source_prefix`: replay the original source from its start through the target region;
2. `episode_reset`: reset at the extracted episode start, replay the declared warm-up,
   then score the same target region.

Compare, at minimum:

- raw LS posteriors or ERes embeddings/similarity scores at aligned source coordinates;
- proposal count and proposal kinds;
- proposal boundary positions and observation frontiers;
- post-clustering actions and safe-frontier progression.

The parity tolerance is frozen before the comparison and is family/output specific.
If the parity gate passes, reset-plus-warm-up episodes may be used for that declared
family/profile class. If it fails materially, scored episodes for that class must start
from deterministic source-prefix state snapshots or source-prefix replay. A failed
parity case remains diagnostic evidence and cannot be hidden by increasing warm-up
post hoc.

### 5.5 Statistical grouping

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

### 7.2 B1: structural-engine equivalence control

B1 routes the exact B0 VAD event stream through the new logical-action, evidence, and
scoring infrastructure but receives no neural proposals.

In the absence of neural proposals, B1 hard segmentation must be action-for-action and
source-sample-for-source-sample identical to B0, including ordinary VAD and any existing
maximum-duration behavior. New bookkeeping, silence-interval attribution, or action
schemas may not create, delete, move, or accelerate a B0 boundary.

Any B0/B1 segmentation difference is an implementation-contract failure, not a product
gain. B1 exists to prove that later neural gains are not artifacts of a rewritten
baseline engine.

### 7.3 Frequency-matched segmentation controls

For each frozen neural policy, create deterministic non-speaker controls with the same
per-episode detector-created hard-action count. Control placement may use only
product-observable causal information available to the tested policy; reference active-
speaker labels, future GT boundaries, and unobserved future audio are forbidden inputs.

Required controls:

- uniformly spaced cuts inside causal VAD-active eligible regions;
- energy-change peaks computed only from causally observed audio, without speaker embeddings;
- shuffled neural boundary positions within the same causal VAD-active regions while preserving action count and the empirical causal-availability distribution as closely as possible.

Every control action must satisfy `boundary_source_sample <= observed_source_sample_at_emit`.
Seeds, eligibility rules, and infeasible-placement behavior are frozen. GT is used only
after action generation for scoring. If these controls yield comparable contamination
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
    confidence_semantics_id
    state_provenance
    debug_evidence
}
```

Invariants:

- `observed_source_sample_at_emit >= boundary_source_sample`;
- events are deterministic for identical audio, model, frontend, and profile;
- `confidence` is interpreted only under its declared `confidence_semantics_id`; no cross-kind or cross-semantics comparison is allowed unless a frozen calibration contract explicitly makes the values comparable;
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
6. Determine the cluster output kind before selecting its representative. For LS mixed-kind clusters, use the frozen semantic priority `overlap_onset > dominant_replacement > new_track_onset > track_instability`. ERes clusters remain `speaker_change_unknown`.
7. Restrict representative selection to proposals compatible with the chosen output kind. `first` chooses the earliest eligible proposal in that subset.
8. `max_confidence` may be used only when all candidate proposals in the representative subset share a comparable frozen `confidence_semantics_id`. Otherwise the deterministic fallback is `first`. Ties prefer earlier observation, smaller absolute distance to the compatible-subset boundary median, earlier boundary, then proposal ID.
9. The cluster boundary equals the chosen representative proposal boundary. No non-observed averaged boundary is invented.
10. Cluster availability is the maximum of cluster close and the representative's observation frontier.
11. After emission, proposals arriving before `availability + R` are suppressed and recorded as refractory proposals.
12. A suppressed proposal never becomes a user-visible action but remains evidence.
13. Episode-end closure uses only audio already observed and is labeled `tail_closed`.

The refractory sweep explicitly measures loss on short B turns and rapid A-B-C
handoffs. Refractory is not assumed to be beneficial.

### 9.3 Cluster evidence

Store member proposal IDs, chosen output kind, compatible representative subset,
representative proposal ID, representative reason, confidence semantics, suppression
reason, cluster open and close frontiers, boundary spread, confidence distribution, and
refractory ownership.

## 10. Hard/soft actionization

Actionization uses only model and causal audio state, never reference labels.

- LS `dominant_replacement` may request a hard action.
- LS `overlap_onset` requests a soft marker.
- LS unstable-track proposals are diagnostic-only.
- ERes `speaker_change_unknown` requests a hard candidate because ERes does not expose overlap state; reference overlap analysis later measures the cost of that limitation.
- cluster output kind and representative boundary always come from semantically compatible proposals under Section 9;
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

### 12.3 Product attribution and orthogonal harm flags

Reference matching assigns benefit/miss attribution. Harm classification is a separate
axis. A boundary may legitimately match a hard reference and still receive a harm flag
when its actual source position fragments preceding same-speaker speech.

Each reference receives one `benefit_attribution` value where applicable:

- `retained_b0_success`
- `recovered_b0_hard_miss`
- `accelerated_b0_success`
- `correct_soft_marker`
- `late_target_action`
- `hard_miss`
- `soft_miss`
- `none`

Each final action independently stores zero or more `harm_or_structure_flags`:

- `harmful_active_split`
- `lexical_split`
- `same_speaker_pause_split`
- `duplicate_hard_boundary`
- `structural_split`
- `overlap_hard_action`
- `unscored_action`

A matched action is not presumed harmless. In particular, an early boundary inside the
500 ms matching tolerance can receive recovery/acceleration credit and simultaneously
receive `harmful_active_split` or `lexical_split` when the corresponding observable
conditions hold. Aggregate reports preserve both axes rather than forcing one exclusive
label.

## 13. Primary benefit: mixed-speaker turn contamination

### 13.1 Logical segmentation

Sort final hard boundaries by source position. Episode edges and hard boundaries define
logical segments. Actions with identical boundary positions form one boundary.

### 13.2 Turn-owner threshold

A singleton speaker becomes the owner of a logical segment only after at least 100 ms
of continuous scorable singleton speech from that speaker inside the segment. Shorter
singleton runs before ownership are treated as annotation/onset jitter for ownership
purposes and are reported separately. Sensitivity views at 50 and 200 ms are mandatory.

This ownership threshold is unrelated to the 250 ms mixed-turn reporting threshold in
Section 13.4.

### 13.3 Contamination algorithm

For each logical segment:

1. intersect the segment with reference active-speaker intervals;
2. exclude silence, ambiguous intervals, and overlap intervals from the contamination numerator and denominator for that stratum;
3. identify the first singleton speaker satisfying the 100 ms ownership threshold;
4. before any different singleton speaker satisfies the same threshold, that speaker owns the turn;
5. once a different qualifying singleton speaker appears, all subsequent qualifying singleton speech until the next hard boundary is contamination, including a later return of the original speaker;
6. do not double-count source samples;
7. report excluded overlap, sub-threshold singleton speech, and unscored duration separately.

This makes a segment containing `A -> B -> C` charge qualifying B and C speech as
contamination of A's logical turn. A premature split before the actual handoff does not
receive false benefit: if the successor segment still becomes owned by A and later
contains B, qualifying B speech remains contamination.

### 13.4 Primary clean/gap headline

The primary product headline is computed only on the predeclared `hard_only` headline
stratum from Section 5.1. `negative_only` exposure contributes to harm and false-action
analysis but not to the contamination denominator. Scored regions with any overlap
reference or stable overlap interval are not allowed to contribute to the primary
clean/gap contamination reduction.

Report:

- `clean_gap_contamination_ratio`: contaminated singleton-speech samples divided by all scorable singleton-speech samples in the clean/gap headline stratum;
- primary paired effect: candidate minus B1 `clean_gap_contamination_ratio` within each source block, with negative values meaning improvement;
- the same paired comparison against B0;
- logical turns containing at least 100, 250, and 500 ms of a second singleton speaker;
- primary mixed-turn rate at the predeclared 250 ms second-speaker threshold;
- clean and gap hard targets remaining merged;
- recovered B0 misses versus accelerated B0 successes;
- contamination avoided per final detector-created hard action.

### 13.5 Overlap-containing contamination is secondary

For `overlap_present` episodes, report all-singleton contamination separately. In
addition, recompute a counterfactual view with detector-created hard actions inside
reference overlap regions suppressed. The difference is labeled
`overlap_hard_action_contamination_contribution` and may not be used to support the
clean/gap headline claim.

A family whose favorable overall contamination result disappears or reverses on the
clean/gap headline stratum is overlap-dependent and cannot be called clean-turn
product-positive.

### 13.6 Exposure and rate reporting

Target-enriched episode pools report raw milliseconds, ratios, and rates per **sampled**
scorable exposure only. They are not converted into natural five-minute/session rates.

Natural five-minute or source-hour contamination estimates are reported only from the
`natural_exposure_validation` pool in Section 16.4. Every rate label states whether its
denominator is target-enriched sampled exposure or unbiased natural exposure.

Transition recall is secondary diagnostic evidence.

## 14. Primary harm: same-speaker fragmentation

Harm flags are evaluated from the emitted boundary location and reference speech state,
independently of whether the action matched a hard reference.

### 14.1 Harmful active split

An action receives `harmful_active_split` when:

- the boundary lies inside a singleton active-speaker interval;
- at least 200 ms of the same singleton speaker is continuously active on both sides;
- the action is not a structural maximum-duration boundary;
- the interval is fully scorable.

A compatible reference match does not suppress this flag. This deliberately captures
an early but reference-matchable boundary that fragments the outgoing speaker.

Report sensitivity at 100 and 300 ms guards without changing the primary 200 ms label.

### 14.2 Lexical split

Where trusted word timing exists, any hard action is a lexical split when it lies inside
a word interval with at least 20 ms of that word on both sides. Word timing source,
revision, and coverage are recorded. Matching a speaker-change reference does not
suppress the lexical flag. Missing word timing produces `not_observable`, not an inferred
negative.

### 14.3 Same-speaker pause split

A detector-created hard action inside silence between two singleton spans of the same
speaker receives `same_speaker_pause_split`. It is not a severe active-speech harm and
receives no speaker-change benefit, but it is not zero-cost: it increases same-speaker
turn fragmentation and is a secondary frontier dimension/tie-breaker.

### 14.4 Duplicate hard boundary

More than one post-fusion hard boundary attributed to one hard reference receives a
duplicate flag. Clustering should suppress most duplicates; remaining duplicates expose
fusion failure.

### 14.5 Overlap hard action

A hard action inside an overlap target or stable overlap interval receives
`overlap_hard_action`. It is neither clean/gap benefit nor same-speaker active-speech
harm. Downstream replay and the overlap counterfactual in Section 13.5 determine its
observed consequences.

### 14.6 Fragmentation metrics

On every target-enriched pool report:

- harmful active splits per sampled active-speech hour and raw count;
- lexical splits per sampled hour of word-aligned speech and raw count;
- duplicate hard actions per sampled source hour and raw count;
- same-speaker pause splits and `same_speaker_extra_turn_count`;
- detector-created fragments with less than 250, 500, and 1000 ms active speech;
- p10/p50/p90 final segment duration and active-speech duration;
- number of consecutive fragments owned by the same speaker;
- legacy unmatched boundary count, labeled historical diagnostic only.

Natural five-minute/session fragmentation rates are reported only from
`natural_exposure_validation`. No target-enriched count is extrapolated to a typical
session rate.

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

Use authorized AMI and AliMeeting sources already in experiment scope. Additional public
corpora are added only when the metadata inventory identifies a concrete missing product
stratum or independent-block shortfall and the addition is separately authorized.

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

- `diagnostic_dev`: controlled synthetic and public development episodes used for signal diagnostics and policy construction;
- `frontier_dev`: development episodes used for final development curves and panel construction;
- `historical_validation`: previously touched Phase 3 held-out sources, never used for confirmatory claims;
- `confirmatory_heldout`: newly selected unused source sessions and speaker groups, inaccessible until freeze;
- `natural_exposure_validation`: deterministic, source-time-uniform bounded windows sampled independently of speaker-transition annotations, used only to estimate natural contamination/fragmentation rates and to compare them with target-enriched estimates.

`diagnostic_dev` and `frontier_dev` are disjoint at the strongest available source-session,
meeting-family, recurring-participant-component, recording, and synthetic-source grouping.
No target, negative, or transformed derivative from a `diagnostic_dev` block may enter
`frontier_dev`.

For `natural_exposure_validation`, window start positions are chosen from source time by
a frozen hash/uniform rule before speaker-transition labels are inspected. Windows use
the same bounded replay machinery and state-equivalence contract. Sampling probabilities,
eligible source duration, and unsampled exclusions are recorded. Five-minute/session and
source-hour rates may be estimated only from this pool (or from complete unbiased source
coverage), never from target-enriched episode pools.

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

The signal gate is executable only after every tested scalar signal registers a frozen
`signal_extractor_id` in `proposal_contract.json`. Each extractor declares its sign,
causal horizon, valid-window rule, and treatment of missing observations.

Primary hard-target AUC uses a 500 ms causal horizon:

- positive example: the strongest declared change-evidence scalar whose observation frontier is no later than hard-target evidence onset + 500 ms;
- matched same-speaker negative: the strongest scalar in the identically sized causal window after the matched pseudo-boundary;
- ERes change strength uses the frozen monotonic transform of cosine similarity declared by the profile (for the standard low-similarity-means-change score this is `1 - cosine`);
- LS uses only explicitly declared hard-target change-strength scalars; overlap-only scores cannot satisfy the hard-target gate.

If no valid observation exists by the horizon, the extractor applies its frozen missing-
observation rule rather than searching later audio. Sensitivity AUCs at 250 and 1000 ms
are reported but do not replace the primary 500 ms gate.

For every frozen scalar signal, calculate ROC-AUC for different-speaker hard targets
versus matched same-speaker acoustic negatives. Compare it with the strongest declared
acoustic-only score on the identical examples.

For each family, use session-block bootstrap on the paired AUC difference:

- `signal_go`: at least one hard-target score has a 95% lower bound greater than zero;
- `signal_limited`: the point estimate is greater than zero but the interval includes zero, or the comparison has fewer than eight independent blocks;
- `signal_stop`: every hard-target score has a 95% upper bound at or below zero.

`signal_go` normally receives the predeclared policy grid. Phase 4 retains the completed
Standard/W24 comparison, but the owner-directed compact Phase 5 excludes W24 from all
new inference and policy replay. It evaluates only the three E-standard extractor IDs
already frozen as `eligible_go` by the accepted Phase 4 disposition:

- `eres_adjacent_change.v1:E-standard:W8000:H500`;
- `eres_prototype_change.v1:E-standard:prototype_memory_4:W8000:S1600:H500`;
- `eres_prototype_change.v1:E-standard:prototype_memory_4:W8000:S4000:H500`.

The adjacent extractor ID is a reference-aligned scalar test and does not encode a
regular proposal step. It maps deterministically to the already declared 100 and 250 ms
steps (`S1600` and `S4000`), so the three eligible Standard signal configurations produce
four executable Phase 5 proposal profiles. No W24, stable-anchor, other window, step,
threshold, confirmation, or state variant is admitted. Each of those four retains the
full clustering, actionization, VAD-fusion, same-proposal ladder, and frequency-control
comparison. The 936 legacy compatibility profiles remain
immutable historical evidence and are not replayed as a new combinatorial sweep; only
B0/B1 and the four compact Standard profiles receive corrected historical-case rescoring.
`signal_limited` receives the same-proposal policy ladder and one sentinel profile per
policy family, but no expanded combinatorial sweep. `signal_stop` receives only B0/B1,
the raw diagnostic report, and the no-neural-policy control. This gate controls compute;
it is not itself a product selection claim.

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

Compare every finalist to non-speaker controls with matched hard-action counts. Control
actions must satisfy the causal/observable rules in Section 7.3.

### F4. Boundary-position shuffle

Shuffle boundary positions only within causally eligible VAD-active regions while
preserving hard-action count and causal availability as closely as possible. Every
shuffled boundary must satisfy `boundary_source_sample <= availability`. Similar benefit
after shuffling falsifies speaker-specific timing.

### F5. Recovery versus acceleration

Report contamination reduction caused by B0-miss recovery separately from earlier
availability of a B0-owned gap boundary.

### F6. Overlap-separated result

Exclude `overlap_present` episodes from the clean/gap headline. Report overlap soft
markers, overlap hard actions, and overlap-hard-action contamination contribution
separately.

### F7. Session robustness

Use leave-one-source-session-out analysis and session-block bootstrap. A conclusion
reversed by removing one session is exploratory.

### F8. Oracle logical-action ceiling

Apply exact hard reference actions at controlled location errors and availability
delays. If the provider-neutral turn assembler cannot conserve audio or reduce
contamination, detector evaluation stops.

### F9. Short-turn refractory stress

Measure loss for 250, 500, and 750 ms successor turns under each refractory setting.

### F10. Same-speaker pause cost

Verify that classifying pause splits as non-severe cannot hide lexical/active-speech
fragmentation, and report the resulting same-speaker extra-turn count separately.

### F11. Episode-state equivalence

Run the source-prefix versus episode-reset comparison from Section 5.4. If proposal or
action parity fails materially, reset-based scored results for that family/profile class
are invalid until source-prefix state is restored.

### F12. Matched-but-harmful boundary

Use fixtures where a hard action falls within the allowed reference matching tolerance
but still lies inside outgoing same-speaker active speech or a word. Verify that the
action can receive benefit attribution and `harmful_active_split`/`lexical_split`
simultaneously.

### F13. Overlap-contribution ablation

Compare all-episode contamination with the clean/gap headline and with overlap hard
actions counterfactually suppressed. A favorable conclusion that depends on overlap hard
actions is not clean-turn product evidence.

### F14. Sampling-rate validity

Compare target-enriched exposure metrics with `natural_exposure_validation`. Verify that
five-minute/hour estimates are emitted only from unbiased natural exposure and that no
report silently rescales enriched counts into natural prevalence.

### F15. Cluster kind/representative consistency

Construct mixed LS-kind clusters and verify that output kind, representative proposal,
boundary coordinate, confidence semantics, and availability are semantically compatible
under Section 9.

### F16. Turn-owner jitter sensitivity

Perturb eligible reference singleton interval edges by deterministic +/-20 and +/-50 ms
views without changing the normative labels. Report sensitivity of the contamination
headline and verify that the 100 ms ownership threshold prevents annotation jitter from
dominating the conclusion.

## 20. Development search and frontier

### 20.1 Search order

1. verify raw model/cache parity;
2. generate raw proposals for every compact Phase 5 `eligible_go` profile;
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
- same-speaker extra turns, including pause splits;
- hard-target recall at every causal deadline;
- causal delay;
- runtime cost;

and is strictly better in at least one. Missing lexical coverage cannot be treated as
zero lexical harm.

The complete integer frontier is preserved. Target-enriched frontiers use raw integer
counts and sampled-exposure metrics. Natural five-minute/hour views are attached only
from `natural_exposure_validation` and are never used as a hidden pre-frontier cap.

### 20.3 Cross-family matched-harm comparison

Use the union of achieved integer harmful-active-split counts on the same development
exposure. At each allowance, select the greatest contamination reduction without
exceeding the allowance; ties prefer fewer same-speaker extra turns, then fewer lexical
splits and duplicates. Report exact achieved cost rather than pretending continuous
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
splits, fewer same-speaker extra turns, lower p95 causal delay, lower runtime cost, and
lexical profile ID. If either
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
- keep target-enriched comparative effects separate from natural-exposure prevalence/rate estimates;
- for `natural_exposure_validation`, aggregate uniformly sampled source-time windows within session before session-level macro/paired summaries and record sampled versus eligible source duration;
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
boundary offset: -500, -300, -200, -100, 0, +100, +200, +300, +500 ms
holdback: 0, 250, 500, 750, 1000, 1500, 2000 ms
```

The +/-500 ms offsets are mandatory sentinels because the hard-reference matcher permits
500 ms localization tolerance. If the development p95 absolute localization error exceeds
500 ms, extend the oracle grid to cover that p95 before provider selection.

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
reviews/phase_0_pre_execution.md
reviews/phase_1_pre_execution.md
reviews/phase_2_pre_execution.md
reviews/phase_3_pre_execution.md
reviews/phase_4_pre_execution.md
reviews/phase_5_pre_execution.md
reviews/phase_6_pre_execution.md
reviews/phase_7_pre_execution.md
reviews/phase_8_pre_execution.md
reviews/phase_9_pre_execution.md
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
10. Overlap soft references cannot raise clean/gap hard-boundary headline recall.
11. `overlap_present` episodes cannot enter the primary clean/gap contamination headline.
12. Warm-up actions cannot enter scored counts.
13. Unscored intervals cannot enter benefit or harm numerators.
14. Contamination source samples are never double-counted.
15. A premature boundary that leaves A and B in the successor turn does not receive false contamination-reduction credit.
16. Turn ownership requires the frozen 100 ms substantive singleton threshold; 50/200 ms sensitivity views are reproducible.
17. Harm flags are independent of benefit matching: a matched boundary can still be an active or lexical split.
18. Harmful active split requires the same singleton speaker on both guarded sides.
19. Missing word timing is not treated as absence of lexical harm.
20. Same-speaker pause splits are non-severe but remain counted as same-speaker extra turns.
21. B1 hard segmentation is exactly identical to B0 when no neural proposals are present.
22. Frequency-matched controls exactly match declared action counts and use no GT to generate actions.
23. Every control/shuffled action satisfies `boundary_source_sample <= observed_source_sample_at_emit`.
24. Cluster output kind and representative boundary come from semantically compatible proposals.
25. `max_confidence` is not used across incompatible confidence semantics.
26. Episode reset-plus-warm-up scored evaluation is forbidden until its family/profile-class state-equivalence test passes; otherwise source-prefix state is required.
27. `diagnostic_dev` and `frontier_dev` are group-disjoint.
28. Bootstrap resamples blocks, not transitions.
29. Cross-split source, speaker, recording, and transformation overlap fails closed.
30. Five-minute/session natural rates are emitted only from unbiased natural exposure or complete source coverage.
31. Held-out cannot open without a valid frozen self-hash.
32. Incomplete held-out sessions cannot produce a decision.
33. Provider-neutral assembly conserves every source sample exactly once.
34. Stale epoch actions cannot mutate the next epoch.
35. Detector safe frontier is monotonic, conservative, and never violated by a later event.
36. Qwen-style safe drain does not release a region before the safe frontier covers it.
37. Deepgram reconnect mapping permits repeated source spans without duplicate normalized words.

## 29. Execution phases, mandatory pre-execution reviews, and gates

### 29.0 Global review rule

Every Phase 0-9 requires an explicit **pre-execution review**. The review is a scientific
and experiment-design gate, not a retrospective report review.

The required ordering is:

```text
previous phase accepted
  -> prepare next-phase review bundle
  -> independent pre-execution review
  -> resolve every required change
  -> review approval recorded
  -> execute the phase
  -> independent evidence verification / phase-exit gate
  -> only then prepare the next phase
```

A phase may not begin experimental execution while its review verdict is `rejected` or
`approved_with_required_changes` with unresolved changes. If the review changes any
normative contract, code hash, manifest, scoring rule, search grid, split, or decision
rule, the affected review bundle must be regenerated and reviewed again before execution.

The pre-execution review must happen **before** the phase performs any action that can
make later correction expensive, leak confirmatory information, or produce apparently
valid but structurally invalid evidence. This includes, as applicable:

- opening or materializing scored audio/annotations beyond what the prior phase approved;
- running new neural inference or a large policy sweep;
- generating scored episode manifests;
- adding data or changing sampling rules;
- constructing or freezing a selection panel;
- opening confirmatory held-out paths;
- using provider credentials or making paid/live provider calls;
- producing a final model/policy recommendation.

Small scaffolding code may be written solely to make the review bundle concrete, but it
must not be used to generate accepted scientific results before approval.

Each phase review produces `reviews/phase_<N>_pre_execution.md` with:

- exact plan/self-hash and source SHA under review;
- phase scope and explicit non-goals;
- prior-phase evidence the phase depends on;
- exact inputs, manifests, caches, code/config hashes, and proposed outputs;
- scoring, causal, split, state, and statistical assumptions relevant to the phase;
- falsification/stop conditions that can prevent the phase from proceeding;
- expected compute/data/provider cost and any irreversible access;
- reviewer findings classified as `blocking`, `major`, or `minor`;
- final verdict: `approved`, `approved_with_required_changes`, or `rejected`;
- every required change and the hash of the corrected artifact when applicable.

`approved_with_required_changes` is not permission to execute until every required change
is resolved and the reviewer records a final `approved` verdict. Review approval never
replaces the phase-exit gate: the phase still must produce and independently verify its
required evidence after execution.

A single blanket review cannot approve multiple later phases. Every phase must receive a
new review using the artifacts actually accepted from the preceding phase.

### Phase 0: checkpoint and contract freeze

**Mandatory pre-execution review timing:** before implementing experiment schemas,
reference/action/fusion logic, replay semantics, or any new model execution. This is the
highest-leverage structure review; if the product question, action semantics, primary
benefit/harm, sampling interpretation, or authority order is wrong, implementation must
not start.

The Phase 0 reviewer must examine at minimum:

- the product question and asymmetric error costs;
- B0/B1 semantics and the exact `logical_finalize` action contract;
- clean/gap versus overlap taxonomy;
- contamination, harm, timing, and natural-exposure definitions;
- proposal -> clustering -> actionization -> VAD-fusion causal structure;
- source-time/epoch/safe-frontier contracts;
- split/held-out authority and the planned statistical unit;
- whether any historical artifact is being treated as normative without justification;
- whether the planned falsification tests can actually invalidate the intended claims.

Deliverables:

- exact restart SHA and dirty-worktree inventory;
- historical artifact hash ledger;
- this plan's self-hash;
- reference/action/fusion schema;
- detector progress and safe-frontier schema;
- B0 logical-finalize replay description;
- approved Phase 0 pre-execution review artifact.

Gate: no held-out access and no new model execution before the action/scoring contract
passes invariant tests and the Phase 0 review verdict is `approved`.

### Phase 1: metadata coverage inventory

**Mandatory pre-execution review timing:** after Phase 0 is accepted, but before building
or using the inventory to add data, change sampling, or open additional corpus material
for scored evaluation.

The Phase 1 reviewer must examine at minimum:

- inventory fields and how each field affects later design decisions;
- independent-block and participant-component grouping rules;
- the distinction between target-enriched and natural-exposure sampling;
- the proposed source-time-uniform sampling frame for natural exposure;
- leakage risks and training-overlap metadata;
- minimum independent-block rules and the exact trigger for adding data;
- whether any proposed data addition targets an observed coverage gap rather than raw hours.

Deliverables:

- all pool/session/speaker/condition counts;
- split-leak graph;
- B0-separated versus B0-missed hard targets;
- negative exposure inventory;
- word-timing observability;
- exact data-gap list and compute/storage forecast;
- natural-exposure eligible duration and frozen source-time sampling frame;
- approved Phase 1 pre-execution review artifact.

Gate: data additions must target an observed gap, not an arbitrary hour count. Any data
addition or sampling-rule change proposed from the inventory must be included in the
accepted Phase 1 evidence and reviewed before it is materialized for Phase 2 scoring.

### Phase 2: episode/reference implementation

**Mandatory pre-execution review timing:** after the Phase 1 inventory is accepted and
before generating scored episode/reference manifests or running stateful model replay on
those episodes.

The Phase 2 reviewer must examine at minimum:

- bounded-episode extraction rules and non-overlap guarantees;
- hard/soft/neutral/unscored reference construction;
- gap interval-valued matching and overlap exclusion from the hard headline;
- turn-owner threshold and annotation-jitter sensitivity plan;
- state-equivalence test design, tolerance, and source-prefix fallback;
- diagnostic/frontier/held-out group-disjointness;
- natural-exposure manifest generation before transition-conditioned inspection;
- sampled waveform/annotation audit procedure.

Deliverables:

- bounded episode builder;
- interval-valued hard references;
- hard/soft/neutral/unscored timelines;
- deterministic manifests;
- sampled waveform/annotation audit;
- episode state-equivalence report and source-prefix snapshot fallback path;
- natural-exposure window manifest generated without transition-label-conditioned placement;
- approved Phase 2 pre-execution review artifact.

Gate: all reference and split invariants pass, sampled episodes agree with source
annotations, and every family/profile class used for reset-based scoring passes the
state-equivalence gate or is switched to source-prefix state.

### Phase 3: provider-neutral logical-action oracle

**Mandatory pre-execution review timing:** before executing the oracle delay/offset/
holdback grid. The reviewer must validate the lifecycle experiment structure before any
neural detector is allowed to consume a full policy sweep.

The Phase 3 reviewer must examine at minimum:

- canonical PCM turn-assembler ownership semantics;
- sample conservation and no-duplication invariants;
- oracle boundary offset, availability-delay, and holdback coverage;
- the +/-500 ms localization sentinels and p95 grid-extension rule;
- safe-frontier drain, timeout, stale-epoch, and late-action behavior;
- contamination and unrecoverable-audio calculations;
- whether a failed assembler test would correctly stop later detector evaluation.

Deliverables:

- canonical PCM turn assembler;
- delay/offset/holdback grid;
- source-sample conservation and ownership evidence;
- safe-frontier drain and timeout evidence;
- contamination ceiling and unrecoverable-late curve;
- approved Phase 3 pre-execution review artifact.

Gate: logical actions must conserve audio and reduce oracle contamination before any
neural family consumes a full policy sweep or confirmatory held-out access.

### Phase 4: raw signal diagnostics

**Mandatory pre-execution review timing:** before any new large neural inference run or
full diagnostic sweep. Cache inspection and tiny parity fixtures may be prepared for the
review, but the production-shaped diagnostic execution starts only after approval.

The Phase 4 reviewer must examine at minimum:

- checkpoint/model/frontend provenance and exact reusable cache identities;
- LS resampler/frontend source-time mapping and causal delay accounting;
- ERes frontend/export parity and exact window coordinates;
- state-equivalence disposition for each family/profile class;
- declared `signal_extractor_id`, sign, causal horizon, missing-observation rule, and AUC construction;
- acoustic-only and same-speaker matched controls;
- `signal_go` / `signal_limited` / `signal_stop` rules and whether they can stop compute without post-hoc interpretation.

Deliverables:

- LS posterior/reducer/oracle report;
- LS 16-to-8 kHz streaming frontend timing and source-mapping parity;
- ERes calibration/anchor/oracle report;
- acoustic-negative controls;
- signal-level go/limited/stop disposition;
- approved Phase 4 pre-execution review artifact.

Gate: only signal-positive or diagnostic-limited families enter the full policy sweep.

### Phase 5: corrected proposal and fusion replay

**Mandatory pre-execution review timing:** before launching the clustering/refractory/VAD-
fusion grid or corrected large development replay.

Phase 5 design regeneration, review, and execution require an explicit owner resume from
the current paused Goal. This document amendment and its review do not resume the Goal.

The Phase 5 reviewer must examine at minimum:

- proposal schema and confidence semantics;
- mixed-kind clustering and representative-selection rules;
- debounce/refractory parameter grid and short-turn stress coverage;
- hard/soft actionization and overlap handling;
- VAD association, replacement/acceleration, and duplicate suppression;
- B0/B1 equivalence;
- same-proposal policy ladder and frequency-matched/shuffle controls;
- orthogonal benefit attribution versus harm flags;
- expected row count, cache reuse, runtime forecast, and completeness checks.

The approved compact planning basis is:

- exactly four E-standard proposal profiles over the currently pinned 878 turn episodes
  and the same four profiles over 204 corrected historical cases;
- 427,566 unique 500 ms E-standard windows, comprising 219,802 verified reusable
  windows and 207,764 windows requiring inference;
- approximately 1.35 hours of benchmark-derived core calculation;
- 2-3 hours of expected Phase 5 execution, including output materialization and the
  frozen 2,048-unit independent audit;
- 3-5 hours of expected elapsed time from explicit Goal resume through design
  regeneration, mandatory pre-execution review, execution, audit, and owner report.

These counts are a pre-review planning baseline, not permission to execute. The Phase 5
design ledger must recompute them from the then-current pinned manifests, cache receipts,
code identities, and declared Phase 4 CPU configuration. Any identity/count mismatch or
an exact execution forecast materially above three hours stops before execution and is
reported to the owner. The superseded 201.6796-hour, 1,024-profile forecast cannot
authorize work.

Deliverables:

- complete causal proposal evidence;
- clustering/refractory rows;
- VAD fusion actions;
- policy-ladder ablations;
- corrected rescoring of historical development caches;
- actual-versus-forecast runtime and cache-reuse receipt for the compact four-profile
  execution;
- approved Phase 5 pre-execution review artifact.

Gate: file/self hashes, expected identities, split completeness, per-session aggregates,
and summary arithmetic are checked exhaustively. B0/B1 equivalence, audio conservation,
and generation-time causal/schema checks remain exhaustive. Independent raw/derived
trace reconstruction for actions, contamination, harm, and timing uses the frozen audit
sample below rather than a second full-grid replay:

- include every mandatory scientific sentinel and every deterministically selected
  failure example;
- include at least one row for every observed checkpoint, proposal-policy class, pool,
  corpus, ladder stage, fusion mode, and control kind;
- fill the remainder to 2,048 distinct physical trace/episode or historical-case units
  by ascending SHA-256 of
  `"turn-episode-v1-phase5-audit-v1" || canonical_unit_id`;
- recompute sampled units from accepted cache/audio/annotation inputs, never from trusted
  derived outputs;
- fail the phase if any sampled reconstruction or exhaustive aggregate check disagrees.

The independent verifier does not expand or replay every logical alias and does not
repeat all neural inference, clustering, control placement, or per-policy scoring.

After the independently verified Phase 5 per-policy results and exit review are accepted,
stop and report them to the owner. No Phase 6 preparation, review, frontier construction,
panel construction, or freeze may begin without a separate explicit owner resume.

### Phase 6: development frontier and freeze

**Mandatory pre-execution review timing:** after Phase 5 evidence is accepted but before
constructing the final selection panel, locking profile IDs, or creating the self-hashed
contract that will authorize confirmatory held-out access.

The post-Phase-5 owner report and separate explicit owner resume are additional entry
conditions for Phase 6; accepted Phase 5 evidence alone does not authorize preparation.

The Phase 6 reviewer must examine at minimum:

- exact primary clean/gap contamination metric and harm dimensions;
- target-enriched versus natural-exposure rate labels;
- Pareto dominance and matched-harm comparison implementation;
- low-harm / frontier-knee / maximum-benefit selection rules;
- treatment of missing lexical observability;
- cross-family comparability and B0/B1 reference handling;
- session-block statistics and leave-one-session-out robustness;
- whether the proposed panel was chosen without held-out information.

Deliverables:

- full product-metric frontiers;
- matched-harm comparisons;
- low/knee/high panels;
- target-enriched versus natural-exposure rate-validity report;
- self-hashed frozen contract;
- no-selection explanation when no positive frontier exists;
- approved Phase 6 pre-execution review artifact covering the exact freeze candidate.

Gate: every expected development row completes before freeze. The frozen contract is not
valid for held-out access until the Phase 6 review approves the exact panel, hashes,
scoring code, split graph, and expected held-out counts that the contract binds.

### Phase 7: confirmatory held-out

**Mandatory pre-execution review timing:** immediately before **any confirmatory held-out
audio path, annotation, manifest payload, or aggregate is opened by the runner**. This is
a hard information-barrier review, not a review after held-out results exist.

The Phase 7 reviewer must examine at minimum:

- the Phase 6 frozen self-hash and all bound code/config/profile IDs;
- proof that held-out groups were not accessed during development;
- expected held-out source sessions/blocks/episodes and completeness checks;
- frozen panel completeness, including deliberately poor/high-cost points;
- bootstrap seed/block graph and no-retuning guarantees;
- clean/gap headline versus overlap reporting masks;
- natural-exposure held-out sampling frame where applicable;
- runner safeguards that prevent partial pooled conclusions or threshold changes.

The runner must require the approved Phase 7 review artifact/hash before resolving or
opening confirmatory held-out paths. An accidental early access invalidates the affected
source group under Section 17.

Deliverables:

- per-session paired evidence;
- session-block uncertainty;
- complete frozen panel including poor points;
- clean/gap headline and separate overlap result;
- natural-exposure contamination/fragmentation rates where the unbiased pool is available;
- runtime evidence;
- approved Phase 7 pre-execution review artifact.

Gate: no pooled conclusion from partial session completion and no confirmatory execution
without the pre-access review approval.

### Phase 8: provider-specific frozen trace replay

**Mandatory pre-execution review timing:** after confirmatory detector traces are accepted
and before provider-specific replay, credential use, paid/live provider calls, or changes
to provider finalization policy arms.

The Phase 8 reviewer must examine at minimum:

- exact frozen detector/action traces to be reused without retuning;
- provider-neutral oracle evidence supporting the proposed lifecycle mechanics;
- source/provider timestamp mapping and reconnect behavior for Deepgram;
- Qwen holdback, safe drain, timeout, and commit ordering;
- Soniox native-speaker baseline and criteria for excluding the local detector;
- one-factor-at-a-time versus explicitly factorial provider arms;
- transcript integrity, loss/duplication, and latency acceptance metrics;
- provider model/API/runtime versions, credential mode, run budget, and sanitized logging.

Deliverables:

- exact same frozen real-detector traces;
- transcript integrity and latency;
- provider-specific feasibility;
- native Soniox comparison where applicable;
- approved Phase 8 pre-execution review artifact.

Gate: boundary metrics alone cannot yield a product-positive recommendation. Provider-
specific execution cannot begin before review approval.

### Phase 9: independent verification and conclusion

**Mandatory pre-execution review timing:** after all evidence-generating phases are closed
but before writing the final model/policy recommendation or implementation handoff. The
review freezes the final-analysis procedure so the conclusion cannot be adapted to the
most favorable observed result.

The Phase 9 reviewer must examine at minimum:

- completeness of every required phase and review artifact;
- exact coordinator recomputation procedure from per-session aggregate evidence, plus
  the frozen sampled raw/derived trace audit rather than duplicate full-grid replay;
- provenance/cache/split/timing audit plan;
- deterministic failure-example selection rule;
- decision-framework mapping to signal-positive, policy-positive, provider-feasible, and product-candidate outcomes;
- negative-outcome rules and external-validity caveat;
- confirmation that no threshold, panel, fusion, or provider policy is being changed after held-out/provider results.

The Phase 9 verifier/reviewer must be independent of the phase executor whose results
are being checked. If the final review discovers a structural flaw that would require
changing a frozen experiment contract, the affected claim is withdrawn or the experiment
returns to the earliest affected phase; it is not repaired by post-hoc rescoring under a
new rule and presented as if it were confirmatory.

Deliverables:

- coordinator recomputation from per-session aggregate evidence and frozen sampled trace
  audit;
- provenance/cache/split/timing audit;
- failure-example audit selected by frozen rule;
- explicit model-signal, policy, provider, and product conclusions;
- implementation handoff or no-selection report;
- approved Phase 9 pre-conclusion review artifact.

Gate: no final implementation recommendation is issued until the Phase 9 review is
`approved` and the independent recomputation/audit agrees with the accepted evidence.

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

A causal clustering/fusion profile reduces primary clean/gap contamination relative to
both B1 and B0 at matched severe harm on development and preserves effect direction
across confirmatory held-out source blocks. If B0 and B1 are not identical, the run is
invalid under the B1 equivalence contract rather than interpreted as a policy result.

### 31.3 Provider-feasible

The frozen actions can be realized without unacceptable audio/text loss, duplication,
or finalization delay.

### 31.4 Product candidate

At least one frozen point:

- has positive held-out clean/gap contamination reduction relative to both B0 and B1 with session-block evidence;
- is better than frequency-matched controls;
- has an explicitly reported harmful-active-split rate on sampled exposure and a separately reported same-speaker extra-turn cost;
- does not depend on overlap hard actions to produce the clean/gap benefit;
- does not materially worsen downstream transcript integrity;
- meets runtime/backpressure requirements;
- binds exact model, frontend, proposal, cluster, fusion, and provider policy.

An exact product tolerance for harmful splits may remain a product-owner decision. The
experiment still reports the full matched-harm frontier and can recommend dominance or
no selection without inventing a cap.

Because newly recorded private/product-domain conversational audio is unavailable, a
successful detector/policy may be labeled `provisional_product_candidate` and handed to
a later implementation task with an explicit external-validity caveat. Lack of such
audio does not by itself invalidate the public/synthetic confirmatory experiment or
forbid implementation-oriented selection. It does forbid claiming measured product-
domain generalization that was not observed.

### 31.5 Required negative outcomes

Use explicit outcomes when appropriate:

- `signal_negative`
- `signal_positive_policy_negative`
- `policy_positive_provider_infeasible`
- `provisional_product_candidate`
- `no_local_detector_selected`

## 32. Independent review, verification, and worker governance

Every Phase 0-9 first receives the mandatory pre-execution review defined in Section 29.
Phase execution does not start until the review verdict is `approved`.
The pre-execution reviewer must be fresh and independent from the Goal executor that will
implement or execute the phase. A reviewer must not inspect
restricted confirmatory held-out content earlier than the phase contract permits.
Each phase-exit evidence verification also uses a fresh reviewer independent from the
Goal executor.

If a reviewer finds a blocking or required change, the coordinator updates the affected
contract/artifact, records the new hash, and requests re-review. The Goal executor may
not treat `approved_with_required_changes` as permission to start. One review may not be
reused to approve later phases whose inputs have changed.

The active Goal executor directly implements and executes each approved phase. OpenCode
must not be launched or delegated to unless the user explicitly authorizes it later. The
Goal executor:

- receives the exact approved phase scope, review artifact/hash, and timeout requirements;
- does not expand the experiment beyond the approved review scope;
- does not retry or restart a failed full run without a new coordinator instruction;
- reports completion or escalation at the phase boundary.

Reviewer or executor reports are not scientific evidence by themselves. The coordinator
independently verifies file/self hashes, expected identities, split completeness,
per-session/pool/block aggregates, summary arithmetic, provider audio conservation, and
review-gate provenance exhaustively. For high-cardinality raw or derived rows, it uses
the phase's frozen deterministic stratified sample unless that phase explicitly requires
an exhaustive sentinel. Accordingly, the coordinator verifies:

- file and self hashes;
- expected session/profile counts;
- split completeness;
- cache identities and sampled payload recomputation;
- sampled causal timing;
- sampled cluster membership;
- sampled final action taxonomy;
- sampled contamination accounting;
- sampled harmful-split classifications;
- sampled session-block bootstrap inputs;
- aggregate recomputation;
- provider audio conservation;
- presence and approval status of the required pre-execution review artifact for every completed phase;
- proof that no phase produced accepted experimental evidence before its review approval.

Experiment-scientific tests are strict for outcome-critical invariants and frozen
sentinels. Broad harness regression, duplicate end-to-end replay, code style, docstrings,
and unrelated polishing are not acceptance gates.

## 33. Architecture boundary

All work remains under `experiments/speaker_turn_boundary`. Production owners,
composition, settings, provider adapters, and runtime lifecycle code are read as the
behavioral baseline but are not modified during detector/fusion experimentation.

Provider oracle adapters are experiment doubles or replay harnesses. Any later
production integration requires a separate reviewed implementation task. This plan
introduces no intended production architecture change.

## 34. Immediate implementation order

The ordering below is subordinate to the mandatory review gates in Section 29. **Do not
batch multiple phases under one review. Before starting any item that belongs to a new
phase, obtain that phase's explicit pre-execution approval first.**

1. Record SHA, dirty state, historical artifact hashes, and prepare the Phase 0 review bundle.
2. Obtain Phase 0 pre-execution approval before implementing experiment schemas or running new model work.
3. Create `turn_episode_v1` schemas and scientific contract tests; verify the Phase 0 exit gate.
4. Prepare and obtain Phase 1 pre-execution approval.
5. Build metadata-only coverage inventory without opening confirmatory held-out audio; verify the Phase 1 exit gate.
6. Prepare and obtain Phase 2 pre-execution approval before scored episode/reference generation.
7. Implement interval-valued reference actions, bounded episode extraction, source-prefix versus reset-plus-warm-up state-equivalence fixtures, snapshot fallback, contamination/harm scoring fixtures, and the unbiased natural-exposure manifest; verify the Phase 2 exit gate.
8. Prepare and obtain Phase 3 pre-execution approval before running the provider-neutral oracle grid.
9. Implement and pass the provider-neutral PCM oracle grid; verify the Phase 3 exit gate.
10. Prepare and obtain Phase 4 pre-execution approval before any new large neural inference or full diagnostic sweep.
11. Run accepted raw signal diagnostics and verify the Phase 4 signal gate.
12. After explicit owner resume, prepare and obtain Phase 5 pre-execution approval before the compact four-profile clustering/refractory/VAD-fusion development sweep.
13. Implement causal cluster/refractory replay, VAD fusion, complete action evidence, cache reuse verification, and corrected development rescoring; verify the Phase 5 exit gate.
14. Decide any data additions only from the accepted inventory findings and under the Phase 1/2 reviewed sampling contract.
15. Stop after accepted Phase 5 per-policy results, report them to the owner, and wait for a separate explicit owner resume before any Phase 6 preparation.
16. Prepare and obtain Phase 6 pre-execution approval before constructing the final frontier/panel or self-hashed freeze.
17. Build the full development frontier and frozen panel; verify the Phase 6 freeze gate.
18. Prepare and obtain the Phase 7 **pre-held-out-access** approval. Do not resolve or open confirmatory held-out paths before this approval.
19. Run confirmatory held-out exactly under the frozen contract and verify complete-session evidence.
20. Prepare and obtain Phase 8 pre-execution approval before provider-specific replay, credentials, or paid/live provider calls.
21. Run provider-specific frozen-trace replay and verify transcript/runtime evidence.
22. Prepare and obtain Phase 9 pre-conclusion approval before writing the final recommendation or implementation handoff.
23. Perform independent final recomputation/audit and issue the conclusion only after the Phase 9 gate passes.

Until the corrected development evidence and its required review gates through Phase 5 are complete, the accepted experiment conclusion is:

> ERes2NetV2 has accepted `signal_go` raw speaker-change evidence. LS-EEND has accepted
> `signal_stop` evidence and does not enter Phase 5 neural replay. No corrected
> product-level detector/fusion selection has been made.
