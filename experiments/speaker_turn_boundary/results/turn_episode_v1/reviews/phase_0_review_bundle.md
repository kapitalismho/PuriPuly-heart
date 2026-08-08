# Phase 0 pre-execution review bundle — bounded turn-episode speaker-change fusion experiment

Status: review bundle for the mandatory Phase 0 pre-execution review (PRD Section 29, Phase 0;
immediate implementation order Section 34 steps 1-2). No experiment schemas have been
implemented for the new run, and no new model execution has started.

Revision history: rev 1 initial bundle; rev 2 resolves review findings P0-INV-001, P0-B0-001,
P0-COALESCE-001, P0-SCORE-001, P0-SCHEMA-001, P0-CAUSAL-001 and records authority note
AUTH-001.

## 1. Artifacts under review

| Item | Value |
| --- | --- |
| Normative plan | `.agents/specs/prd/bounded_turn_episode_speaker_change_fusion_experiment_review_gated.md` |
| Plan git blob | `24340f488f1bb46c666a5fc15eef2fc87ef1f826` (committed in `96462b0f`) |
| Plan self-hash (SHA-256 of file bytes) | `8c6bed2e564b9ec80e26ee6b73701985c863b7beb68e51590be7a5faf173aad4` |
| Restart commit | `fef0a6b312df34680d9db0fd858e28ae054ace89` (`experiment: add phase 3 evidence dumps`) |
| Work branch | `experiment-v2-speaker-change-turn-boundaries-ls` (no upstream) |
| Dirty-worktree inventory | clean (0 modified, 0 untracked at bundle time) |
| Integration target (informational) | `origin/main` at `848aa0b9f1b35388ded5a250d51a687223eac1c5` |

Authority-order note: authority item 2 (`.agents/specs/prd/speaker_change_turn_boundary_experiment_handoff_en(1).md`)
is **absent from the repository** (verified by `git ls-files .agents` and filesystem search).
GitHub issue #51 is external product/history context. The plan-only authority basis is
therefore **provisional**: if the absent handoff document is recovered, it must be re-audited
against this bundle before it can add any conflicting normative requirement (finding
AUTH-001). Until then the normative plan stands on itself.

## 2. Restart inventory

- Restart SHA: `fef0a6b312df34680d9db0fd858e28ae054ace89`.
- Working tree at restart: clean.
- Historical artifacts are preserved byte-for-byte under
  `experiments/speaker_turn_boundary/results/phase3/`, `results/`, and `data/`; none are
  overwritten by this run. The historical hash ledger is
  `reviews/historical_artifact_ledger.json` (SHA-256 per artifact, computed over file bytes).

Key historical facts (from the ledger):

- `results/phase3/dev_rows_v2.jsonl`: 1,369 rows (SHA-256 `6fc01ce8...`).
- Phase 3 `dev_evidence/` holds 6 profile evidence files; `heldout_evidence/` holds the same
  6 profiles across 3 previously touched held-out groups
  (`ami_held_out_pilot`, `ls_held_out_clean`, `ls_held_out_other`). Per plan decisions 13-15
  and Section 16.1/16.4, these are historical validation inputs, never confirmatory.
- The Phase 2 manifests and validation results under `data/manifests/` and `data/results/`
  are historical inputs for corrected replay.

## 3. Phase 0 scope and non-goals

Scope:

- Freeze the restart SHA, dirty-worktree inventory, plan self-hash, and historical hash ledger.
- Freeze the `turn_episode_v1` reference/action/fusion schema and the detector
  progress/safe-frontier schema as *design contracts* (this bundle), then implement them as
  code with scientific contract tests after review approval.
- Freeze the B0 logical-finalize replay description (Section 5 below), the B1 equivalence
  contract it implies, and the exact boundary of what the current replay trace can prove.
- Produce the approved Phase 0 pre-execution review artifact.

Non-goals (Phase 0):

- No scored episode/reference manifests (Phase 2).
- No new neural inference, no model execution beyond already-committed cached evidence (Phase 4+).
- No clustering/fusion/VAD-fusion replay (Phase 5).
- No held-out access of any kind (Phase 7).
- No provider calls or credentials (Phase 8).
- No coverage inventory (Phase 1), no data additions.

## 4. Product question and asymmetric error costs

Product question (PRD Section 0): can causal speaker-change evidence, after proposal
stabilization and VAD fusion, reduce speech from different speakers being finalized in one
STT/translation turn without causing too many harmful or excessive same-speaker splits?

Asymmetric error costs, as fixed by the plan (Section 1):

- Primary benefit: reduction of mixed-speaker turn contamination on the clean/gap headline
  stratum (decision 10; Section 13.4).
- Primary severe harm: a hard boundary inside stable same-speaker active speech
  (`harmful_active_split`, Section 14.1; 200 ms same-speaker guard both sides).
- Same-speaker pause splits are non-severe but not zero-cost (`same_speaker_pause_split`,
  `same_speaker_extra_turn_count`, decision 8).
- Hard logical boundary ends the current logical STT/translation turn while VAD state,
  detector state, and translation context survive (decision 9; Section 3 `logical_finalize`
  semantics: audio before boundary -> old turn, at/after -> new turn, every source sample
  assigned exactly once unless provider replay explicitly duplicates and is normalized back,
  no synthetic silence reset, LS-EEND state continues, ERes policy state changes only per its
  frozen policy, translation context survives, stale epoch actions cannot mutate the current
  epoch).
- Overlap onset is a soft-marker diagnostic, excluded from the clean/gap headline (decision 7).
- No arbitrary false-split cap removes candidates before the frontier is constructed
  (decision 12); harm is reported on the full frontier, and an exact product tolerance for
  harmful splits may remain a product-owner decision (Section 31.4).

## 5. B0/B1 semantics and the logical-finalize replay description

### 5.1 Production-shaped path (what the product does today)

The current peer path is (PRD Section 3):

```text
mono capture
  -> VAD SpeechStart / SpeechChunk / SpeechEnd
  -> STT backend audio stream
  -> SpeechEnd requests backend finalization
  -> peer logical turn
  -> translation turn
```

Production `SpeechEnd` finalizes audio that has already been transmitted; it does not provide
a source-sample retrospective split primitive. Production finalization, terminal flush, and
maximum-duration (7000 ms peer segment) boundaries are production STT-controller behavior
(read as behavioral baseline; not modified during this experiment). The peer VAD gating
profile (`src/puripuly_heart/core/vad/gating.py`): `PEER_VAD_SPEECH_THRESHOLD`, start
debounce/commit chunks, `PEER_MAX_SEGMENT_MS = 7000`, and the replay uses 500 ms hangover and
500 ms pre-roll ring buffer.

### 5.2 The canonical B0 replay trace (what the experiment replays)

B0 (Section 7.1) reproduces the current peer VAD configuration with causal event timing and
**no speaker signal**. The committed historical B0 replay is implemented by
`vad_baseline.py` + `run_b0_replay.py`:

- Input: mono 16 kHz PCM16, processed in 512-sample (32 ms) chunks.
- VAD: `create_peer_vad_gating` with speech threshold, start debounce/commit chunks, 7000 ms
  max segment, 500 ms hangover, 500 ms pre-roll ring buffer.
- Emitted events: `SpeakerBoundaryEvent` for a boundary between two utterances, placed at
  the **start sample of the chunk in which the successor `SpeechStart` fires**, observed at
  the **end of that chunk** (`observed_source_sample_at_emit = chunk_end`). Per-epoch
  `DetectorProgress` snapshots are recorded at chunk granularity.
- Important trace boundary (finding P0-B0-001): the canonical B0 replay trace **does not
  emit or retain `SpeechEnd` events**; `SpeechEnd` is consumed by the replay only to
  populate previous-utterance metadata (`speech_end_sample`, trailing silence, reason). A
  terminal `SpeechEnd` or a max-duration structural boundary with no successor utterance is
  therefore **not present** in the trace. B0's trace-visible segmentation consists of
  successor-`SpeechStart` boundaries only.

Consequences for B1:

- The B1 equivalence contract (Section 7.2) is defined over the **trace-visible B0
  segmentation**: with no neural proposals, B1 must be action-for-action and
  source-sample-for-source-sample identical to B0 for every boundary the B0 trace emits.
- `SpeechEnd`-driven finalization semantics (terminal flush, max-duration structural
  boundaries, `structural_max_duration` actions, Section 6.6) are **deferred to the Phase 3
  provider-neutral oracle and Phase 8 provider replay**, where the `logical_finalize`
  lifecycle is validated with oracle traces. Phase 0 records this deferral explicitly; it is
  not silently claimed as proven by the B0 replay.

### 5.3 Historical coalescing is non-normative for B0/B1

`run_b0_replay.py` additionally runs the historical `coalesce_vad_and_detector` helper with a
500 ms window (`VAD_COALESCE_WINDOW_SAMPLES = 8000`, `config.py`). This helper is
**historical/non-normative for the new run** (finding P0-COALESCE-001):

- its `absorbed` duplicate-tracking set is not keyed by epoch (cross-epoch index aliasing);
- it orders detector events by boundary position rather than observation frontier;
- it does not implement the Section 11.4 association rules or the safe-frontier contract.

Its historical outputs remain evidence only. The new run implements epoch-keyed,
observation-ordered causal VAD fusion at Phase 5 (Section 11 of the plan). B0/B1 for the new
run consume only the raw VAD boundary trace plus `DetectorProgress` snapshots.

### 5.4 Desired hard action

```text
logical_finalize(boundary_source_sample)
```

Semantics fixed in Section 3: boundary sample partition, exactly-once source assignment
(provider replay duplicates normalized back to one source span), VAD continues without
synthetic silence reset, LS-EEND state continues inside the episode, ERes policy state
changes only per its frozen proposal policy, translation context survives, stale epoch
actions rejected.

### 5.5 Provider policy vs provider-neutral logical action

Two layers (Section 3): (1) provider-neutral logical action at a canonical source sample;
(2) provider policy realizing the action without loss/duplication, tested later with oracle
traces (Phase 3/8). Provider adapters are experiment doubles or replay harnesses
(Section 33); production wiring is out of scope.

## 6. Draft schemas (turn_episode_v1)

The following design contracts are proposed for implementation after approval. Schema
versioning: `turn_episode_v1`; proposal generation and product actionization use separate
schema versions (Section 8 invariant).

### 6.1 ProposalEvent (Section 8)

```text
ProposalEvent {
    proposal_id: str
    family: "ls_eend" | "eres2netv2" | "control"   # frozen family ids
    checkpoint: str
    profile_id: str
    audio_epoch: int
    source_session_id: str
    proposal_kind: "new_track_onset" | "dominant_replacement" |
                   "overlap_onset" | "track_instability" | "speaker_change_unknown"
    boundary_source_sample: int
    observed_source_sample_at_emit: int
    emitted_monotonic_ns: int
    confidence: float | None
    confidence_semantics_id: str
    state_provenance: str            # e.g. "episode_reset+warmup" | "source_prefix"
    debug_evidence: {...}            # confirmation samples / posterior frames used
}
```

Invariants (validated at construction and by contract tests):

- `observed_source_sample_at_emit >= boundary_source_sample` (invariant 1, 23).
- `audio_epoch >= 0`; epoch and source-session identity mandatory.
- `confidence` is only interpreted under `confidence_semantics_id`; cross-semantics numeric
  comparison is forbidden (invariant 25) unless a frozen calibration contract says otherwise.
- Determinism and "no reads beyond observation frontier" are enforced by the replay harness
  design (Phase 4/5); the schema records the fields needed to audit them.

### 6.2 DetectorProgress / safe frontier (Section 4.10, Section 8 progress invariants)

```text
DetectorProgress {
    audio_epoch: int
    observed_source_sample: int
    safe_boundary_frontier_sample: int
}
```

Invariants (validated at construction; monotonicity over a trace is a contract test,
invariant 35):

- `0 <= safe_boundary_frontier_sample <= observed_source_sample`.
- Within one epoch, observed and safe frontiers are monotonically non-decreasing.
- The safe frontier covers every still-possible retrospective boundary from frontend
  buffering, neural lookback, confirmation, and open-cluster state; no later proposal may
  name a boundary at or before an already published safe frontier.
- Frontier resets with the epoch. A heuristic watermark is not sufficient (Qwen safe drain
  relies on the guarantee, Section 24.2).

### 6.3 Reference action and reference outcome (Sections 6, 12)

```text
ReferenceAction {
    reference_id: str
    audio_epoch: int
    source_session_id: str
    action_kind: "hard_boundary" | "soft_overlap_marker" | "state_update" |
                 "neutral_pause" | "structural" | "unscored"
    target_sample: int | None            # clean handoff / interruption onset: B onset
    acceptable_interval: [start, end]    # gap handoff: [A speech offset, B onset];
                                         # clean: point +/- localization tolerance
    evidence_onset_sample: int           # detector-evidence onset (B onset for hard targets)
    scorable: bool
    primary_case: bool                   # clean/gap hard targets only
    episode_pool_tag: "hard_only" | "overlap_present" | "negative_only"
}
```

Notes:

- Gap handoff localization error is zero inside `[A speech offset, B onset]` and distance to
  the nearest interval edge outside it (Section 6.2).
- Unscored intervals (Section 6.7) produce `action_kind="unscored"`; actions in them are
  counted as `unscored_action`, never inferred correct or harmful (invariant 13).
- Pool tags (Section 5.1) gate the primary clean/gap headline: only `hard_only` contributes
  (invariants 10, 11).

Benefit attribution lives **per reference** (Section 12.3; finding P0-SCHEMA-001), because
`hard_miss`/`soft_miss`/`late_target_action` have no final action carrying them:

```text
ReferenceOutcome {
    reference_id: str
    matched_action_id: str | None
    benefit_attribution: "retained_b0_success" | "recovered_b0_hard_miss" |
                         "accelerated_b0_success" | "correct_soft_marker" |
                         "late_target_action" | "hard_miss" | "soft_miss" | "none"
    availability_delay_ms: int | None
    localization_error_ms: int | None
}
```

### 6.4 Final action and harm axis (Sections 11.2, 12.3)

```text
FinalAction {
    action_id: str
    audio_epoch: int
    source_session_id: str
    action_kind: "retain_vad" | "accelerate_or_replace_vad" | "add_hard_boundary" |
                 "emit_soft_marker" | "suppress_detector_duplicate" |
                 "suppress_vad_duplicate" | "structural_max_duration" | "unscored_action"
    boundary_source_sample: int | None
    observed_source_sample_at_emit: int | None
    emitted_monotonic_ns: int | None
    availability_source_sample: int
    cluster_id: str | None
    matched_reference_id: str | None
    harm_or_structure_flags: [...]
}
```

- Only `retain_vad`, `accelerate_or_replace_vad`, `add_hard_boundary` create final hard
  logical boundaries (Section 11.2).
- Observation/emission provenance (`observed_source_sample_at_emit`,
  `emitted_monotonic_ns`) is retained on the action so location accuracy and causal
  availability are never collapsed into one timestamp (Section 4.9; finding P0-SCHEMA-001).
- Harm flags (Section 12.3) are an independent axis stored on the action; a matched action is
  not presumed harmless (invariant 17). Flags: `harmful_active_split`, `lexical_split`,
  `same_speaker_pause_split`, `duplicate_hard_boundary`, `structural_split`,
  `overlap_hard_action`, `unscored_action`.
- Hard and soft outputs from one cluster cannot both create product actions (Section 9.2/10);
  a final action belongs to at most one cluster (invariant 4).

### 6.5 LogicalBoundaryCluster and actionization contract (Sections 9.3, 10)

```text
LogicalBoundaryCluster {
    cluster_id: str
    audio_epoch: int
    source_session_id: str
    member_proposal_ids: [...]
    output_kind: "overlap_onset" | "dominant_replacement" | "new_track_onset" |
                 "track_instability" | "speaker_change_unknown"
    compatible_representative_subset: [...]
    representative_proposal_id: str
    representative_reason: "first" | "max_confidence" | "fallback_first"
    confidence_semantics_id: str
    suppression_reason: "refractory" | "none"
    open_frontier_sample: int
    close_frontier_sample: int
    availability_source_sample: int
    boundary_spread_samples: int
    confidence_distribution: [...]
    refractory_owner_cluster_id: str | None
    tail_closed: bool
}
```

Actionization mapping (Section 10; finding P0-CAUSAL-001):

- LS `dominant_replacement` may request a hard action.
- LS `overlap_onset` requests a soft marker.
- LS `track_instability` is diagnostic-only and can never directly create a hard action.
- ERes `speaker_change_unknown` requests a hard candidate (ERes does not expose overlap
  state; reference overlap analysis later measures that limitation's cost).
- Cluster output kind and representative boundary always come from semantically compatible
  proposals; hard and soft outputs from one cluster cannot both create product actions.

### 6.6 Frozen tables frozen now

- Cluster output-kind semantic priority (Section 9.2.6):
  `overlap_onset > dominant_replacement > new_track_onset > track_instability`;
  ERes clusters remain `speaker_change_unknown`.
- Representative selection (Section 9.2.7-9.2.8): `first` | `max_confidence`; `max_confidence`
  only when all candidates in the compatible subset share a comparable frozen
  `confidence_semantics_id`, otherwise deterministic fallback `first`; tie order: earlier
  observation, smaller absolute distance to the compatible-subset boundary median, earlier
  boundary, then proposal ID (invariant 24, 25).
- Primary hard localization tolerance 500 ms with a 250 ms view; availability deadlines 250,
  500, 1000, 1500, 2000 ms (Section 12.1).
- Turn-owner threshold 100 ms with mandatory 50/200 ms sensitivity views (Section 13.2;
  invariant 16).
- Harmful-active-split guard 200 ms with 100/300 ms sensitivity views (Section 14.1).
- VAD-fusion association parameters (Section 11.3): `detector_vad_radius_ms V in {250, 500}`,
  `same_silence_interval_association in {false, true}`.

## 7. Scoring, causal, split, state, and statistical assumptions

### 7.1 Causal assumptions

- Proposals/actions never use future audio or reference labels (Sections 8, 10, 11);
  `boundary_source_sample <= observed_source_sample_at_emit` for every event and control
  action (invariant 23). Controls may use only product-observable causal information
  (Section 7.3).

### 7.2 Scoring contract (Section 28 invariants relevant to Phase 0)

The following rules are frozen as the Phase 0 scoring contract so later phases cannot deviate
(finding P0-SCORE-001):

- **Gap-VAD validity (invariant 9):** a pre-existing VAD boundary inside the gap acceptable
  interval is valid product separation and may never be rejected merely because it is more
  than 500 ms from B onset. Its availability is reported as pre-existing, not anticipatory.
- **No anticipatory detector credit (invariant 8):** a detector/speaker-model proposal whose
  observation frontier is before the detector-evidence onset (B onset for hard targets)
  receives no gap speaker-change evidence credit.
- **Overlap exclusion (invariants 10, 11):** `overlap_present` episodes and overlap soft
  references cannot enter the primary clean/gap contamination headline or raise hard-boundary
  headline recall. The overlap counterfactual (`overlap_hard_action_contamination_contribution`,
  Section 13.5) is reported separately and cannot support the clean/gap claim.
- **Warm-up exclusion (invariant 12):** warm-up actions and references never enter scored
  counts.
- **Unscored exclusion (invariant 13):** unscored intervals never enter benefit or harm
  numerators.
- **Contamination algorithm (invariants 14, 15):** source samples are never double-counted;
  turn ownership requires 100 ms continuous singleton speech from the owning speaker before
  ownership (50/200 ms sensitivity views reported); a premature boundary that leaves the
  successor turn still owned by A and later containing B charges qualifying B speech as
  contamination (no false reduction credit); once a different qualifying singleton speaker
  appears, all subsequent qualifying singleton speech until the next hard boundary is
  contamination, including a later return of the original speaker.
- **Harm rules (invariants 17-20):** harm flags are independent of benefit matching
  (a matched boundary can still be an active or lexical split); `harmful_active_split`
  requires the same singleton speaker on both guarded sides (200 ms guard, 100/300 ms views);
  missing word timing produces `not_observable`, never an inferred negative; same-speaker
  pause splits are non-severe but remain counted as `same_speaker_extra_turn_count`.
- **Timing metrics (Section 15):** interval localization error, signed point error,
  causal availability delay, event lookback, cluster debounce delay, VAD association delay,
  wall-clock model service time, end-to-end scheduling completion delay, real-time factor,
  final backlog, peak RSS, model load time, cache-hit/miss execution time. No negative
  detector availability delay; pre-existing gap VAD actions are a separate valid category.
- **Rate labeling (Section 13.6, 16.4):** target-enriched pools report raw ms, ratios, and
  rates per sampled exposure only, and are never converted into natural five-minute/session
  rates. Five-minute/session/source-hour rates are emitted only from
  `natural_exposure_validation` (source-time-uniform windows sampled before
  transition-label inspection) or complete unbiased source coverage (invariant 30).

### 7.3 Split/held-out and statistical assumptions

- `diagnostic_dev` and `frontier_dev` are group-disjoint at the strongest available grouping
  (Section 16.4; invariant 27).
- Bootstrap resamples source-session blocks, never transitions (Section 21; invariant 28).
- Cross-split source, speaker, recording, and transformation overlap fails closed via
  group-graph hashes (Section 17; invariant 29).
- Confirmatory held-out paths cannot be opened until a valid frozen self-hash exists and the
  Phase 7 pre-access review is approved (Sections 17, 22; invariants 31, 32). Phase 0 opens
  nothing.
- State: reset-plus-warm-up scored evaluation is forbidden until the Section 5.4
  state-equivalence contract passes for the family/profile class; otherwise source-prefix
  state (invariant 26). Phase 0 implements the prohibition gate only; the parity run itself
  executes at Phase 2/4.
- Historical artifacts are evidence/cached model output only (Sections 0, 16.1); none is
  treated as normative corrected results.

## 8. Falsification and stop conditions for Phase 0

- A B0/B1 difference in the Phase 5 equivalence test invalidates the run as a policy result
  (Section 31.2) — Phase 0 freezes the contract so that test is defined now, over the
  trace-visible B0 segmentation (Section 5.2).
- Phase 0 fails if the schemas cannot express the Section 8/4.10/6/11/12 contracts above, or
  if any frozen table (Section 6.6) is ambiguous.
- The Phase 0 exit gate (PRD): no held-out access and no new model execution before (a) the
  action/scoring contract passes its invariant tests and (b) the Phase 0 review verdict is
  `approved`.

## 9. Expected cost and irreversible access

- Phase 0: no compute, no data access beyond hashing already-committed files, no provider
  access, no held-out access. Cost: negligible (local test run only).
- Irreversible access: none. The only files created are under
  `experiments/speaker_turn_boundary/results/turn_episode_v1/` and `.agents/runs/opencode/`.

## 10. Reviewer examination checklist (from PRD Phase 0)

1. Product question and asymmetric error costs (Sections 0, 1, 13, 14).
2. B0/B1 semantics and the exact `logical_finalize` action contract (Sections 3, 7.1, 7.2).
3. Clean/gap versus overlap taxonomy (Section 6).
4. Contamination, harm, timing, and natural-exposure definitions (Sections 13, 14, 15, 16.4).
5. Proposal -> clustering -> actionization -> VAD-fusion causal structure (Sections 8-11).
6. Source-time/epoch/safe-frontier contracts (Sections 3, 4.9, 4.10, 8).
7. Split/held-out authority and the planned statistical unit (Sections 16, 17, 21, 22).
8. Whether any historical artifact is being treated as normative without justification
   (Sections 0, 16.1).
9. Whether the planned falsification tests can actually invalidate the intended claims
   (Sections 19, 28, 31).

## 11. Phase 0 scientific contract/gate inventory (post-approval implementation)

### 11.1 Implementation units

- `experiments/speaker_turn_boundary/turn_episode/` package: schema dataclasses with
  construction-time validation (`schemas.py`) and pure invariant/contract functions plus the
  frozen tables (`contracts.py`).
- `tests/test_turn_episode_contracts.py`: scientific contract tests, plus a gate-only test
  module where noted.
- `results/turn_episode_v1/proposal_contract.json` and `fusion_contract.json`: frozen schema
  version, field definitions, and profile/extractor registries (extractor registry completed
  at Phase 4 per Section 18.3).
- `results/turn_episode_v1/reviews/phase_0_pre_execution.md`: the review artifact.

### 11.2 Invariant disposition (Section 28)

Phase 0 implements **synthetic contract tests** (in-memory references, actions, proposals,
clusters, metadata, hashes, block records; no audio, no model):

- **4** no final action in two clusters; **5** no duplicated hard boundary at one source
  sample; **7** gap interval matching accepts any boundary inside the annotated silence;
  **8** no gap credit before B onset; **13** unscored intervals never enter numerators;
  **17** benefit/harm orthogonality (a matched boundary can carry active/lexical split flags);
  **24** cluster output kind/representative compatibility; **25** `max_confidence` restricted
  to comparable confidence semantics; **34** stale epoch actions rejected.
- **2** no cluster member after cluster close (pure policy function on ordered proposals);
  **6** matching ordered one-to-one (pure matching engine over synthetic references/actions);
  **9** pre-existing VAD gap boundary stays valid; **10**/**11** overlap soft references and
  `overlap_present` episodes excluded from the clean/gap headline (pure pool-tag/mask
  functions); **12** warm-up actions excluded from scored counts; **14** contamination
  samples never double-counted; **15** premature-split no-false-credit rule; **16** turn
  ownership 100 ms threshold with reproducible 50/200 ms views; **18** harmful active split
  requires same singleton speaker on both guarded sides; **19** missing word timing is
  `not_observable`, not absence of harm; **20** pause splits counted as same-speaker extra
  turns; **27** diagnostic/frontier group-disjointness (hash/group-graph function);
  **28** bootstrap resamples blocks (block-record function, not full bootstrap); **29**
  cross-split overlap fails closed; **30** natural-rate labels gate (only
  natural_exposure_validation source); **31** held-out open requires valid frozen self-hash;
  **32** incomplete held-out sessions cannot produce a decision (completeness-check
  function).

Phase 0 implements **gate-only checks** (the prohibition/contract gate is tested; the
empirical evidence is produced by later phases):

- **1** schema-level `observed >= boundary` gate plus full frontier-read audit deferred to
  replay harness (Phase 4/5); **26** reset-plus-warm-up prohibition gate only, state parity
  run executes at Phase 2/4; **35** monotonicity/coverage check as a pure trace validator;
  conservative safe-frontier *guarantee* evidence is produced by the Phase 5 fusion replay.

Later-phase execution tests (recorded, not Phase 0): **3** refractory determinism (Phase 5);
**21** B0/B1 equivalence (Phase 5); **22** frequency-matched controls (Phase 5); **23**
control causal availability (Phase 5); **33** provider-neutral sample conservation (Phase 3);
**36** Qwen safe drain (Phase 3/8); **37** Deepgram reconnect deduplication (Phase 8).

## 12. Proposed post-approval implementation (informational)

Covered by Section 11.1; the complete contract/gate inventory in Section 11.2 replaces the
earlier abbreviated invariant list.

## 13. Recorded review findings and dispositions

| id | severity | finding | disposition |
| --- | --- | --- | --- |
| P0-INV-001 | blocker | Phase-0 invariant list incomplete; invariant 26 only partially Phase-0-testable | resolved in Section 11.2 (complete disposition per invariant) |
| P0-B0-001 | blocker | B0 trace does not emit/retain SpeechEnd; max-duration/terminal semantics not trace-visible | resolved in Section 5.2 (trace boundary stated; deferral to Phase 3/8 recorded) |
| P0-COALESCE-001 | important | historical 500 ms coalescer non-epoch-safe, boundary-ordered; omitted from bundle | resolved in Section 5.3 (recorded as non-normative historical evidence) |
| P0-SCORE-001 | important | scoring contract incomplete (gap-VAD validity, overlap masks, warm-up, contamination, harm, timing, natural exposure) | resolved in Section 7.2 |
| P0-SCHEMA-001 | important | benefit attribution must be per-reference; FinalAction lacks observation/emission provenance and match link | resolved in Sections 6.3-6.4 |
| P0-CAUSAL-001 | important | no cluster schema, no explicit actionization mapping | resolved in Section 6.5 |
| AUTH-001 | note | authority #2 absent; plan-only authority provisional | recorded in Section 1; re-audit required if handoff recovered |
