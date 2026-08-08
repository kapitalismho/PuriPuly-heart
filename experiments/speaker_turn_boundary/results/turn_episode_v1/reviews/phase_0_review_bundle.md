# Phase 0 pre-execution review bundle — bounded turn-episode speaker-change fusion experiment

Status: review bundle for the mandatory Phase 0 pre-execution review (PRD Section 29, Phase 0;
immediate implementation order Section 34 steps 1-2). No experiment schemas have been
implemented for the new run, and no new model execution has started.

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
GitHub issue #51 is external product/history context. The normative plan therefore stands on
itself; if the absent handoff document is later recovered, its content must be re-audited
against this bundle before it can add authority.

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
- Freeze the B0 logical-finalize replay description (Section 5 below) and the B1 equivalence
  contract it implies.
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

### 5.1 Current production-shaped path (B0)

B0 reproduces the current peer VAD configuration with causal event timing and **no speaker
signal** (Section 7.1). It is already implemented and verified in the historical run as:

```text
mono 16 kHz PCM
  -> Silero VAD via peer gating profile (create_peer_vad_gating:
     speech threshold PEER_VAD_SPEECH_THRESHOLD, start debounce/commit chunks,
     max segment 7000 ms, hangover 500 ms, 500 ms pre-roll ring buffer)
  -> SpeechStart / SpeechEnd translated to SpeakerBoundaryEvent at 512-sample (32 ms)
     chunk granularity; a boundary between two utterances is placed at the start sample
     of the chunk in which the successor SpeechStart fires, observed at the end of that
     chunk (boundary_source_sample = chunk_start, observed_source_sample_at_emit = chunk_end)
  -> SpeechEnd requests backend finalization (peer logical turn -> translation turn)
```

Historical B0 evidence: `results/result_b0_phase0_*.json`,
`results/sweep_{ls_eend,eres}_summary_b0_phase0.json`; B0 VAD replay code is
`vad_baseline.py` + `run_b0_replay.py`; the peer gating profile is read from
`puripuly_heart/core/vad/gating.py` (behavioral baseline, not modified).

Key B0 characteristics relevant to the new run:

- B0 has no source-sample retrospective split primitive: `SpeechEnd` finalizes audio already
  transmitted (PRD Section 3).
- B0 boundary positions are chunk-quantized to 512-sample (32 ms) grid, which aligns with the
  plan's 32 ms gap binning (Section 16.2).
- B0 VAD maximum-duration behavior (`structural_max_duration` at 7000 ms) is part of B0
  segmentation and must survive B1 identically (Section 7.2; Section 6.6 structural actions).

### 5.2 B1 structural-engine equivalence control

B1 (Section 7.2) routes the exact B0 VAD event stream through the new logical-action,
evidence, and scoring infrastructure but receives no neural proposals. In the absence of
neural proposals, B1 hard segmentation must be action-for-action and source-sample-for-
sample identical to B0, including ordinary VAD boundaries and the 7000 ms maximum-duration
behavior. New bookkeeping, silence-interval attribution, or action schemas may not create,
delete, move, or accelerate a B0 boundary. Any B0/B1 difference is an implementation-contract
failure (Section 31.2), not a product gain.

Phase 0 freezes this contract; the B1 equivalence test itself is executed at Phase 5
(invariant 21 of Section 28).

### 5.3 Desired hard action

```text
logical_finalize(boundary_source_sample)
```

Semantics fixed in Section 3: boundary sample partition, exactly-once source assignment
(provider replay duplicates normalized back to one source span), VAD continues without
synthetic silence reset, LS-EEND state continues inside the episode, ERes policy state
changes only per its frozen proposal policy, translation context survives, stale epoch
actions rejected.

### 5.4 Provider policy vs provider-neutral logical action

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

### 6.3 Reference action (Section 6 taxonomy, Section 12)

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

### 6.4 Final action and benefit/harm axes (Sections 11.2, 12.3)

```text
FinalAction {
    action_id: str
    audio_epoch: int
    source_session_id: str
    action_kind: "retain_vad" | "accelerate_or_replace_vad" | "add_hard_boundary" |
                 "emit_soft_marker" | "suppress_detector_duplicate" |
                 "suppress_vad_duplicate" | "structural_max_duration" | "unscored_action"
    boundary_source_sample: int | None
    availability_source_sample: int
    cluster_id: str | None
    benefit_attribution: "retained_b0_success" | "recovered_b0_hard_miss" |
                         "accelerated_b0_success" | "correct_soft_marker" |
                         "late_target_action" | "hard_miss" | "soft_miss" | "none"
    harm_or_structure_flags: [...]
}
```

- Only `retain_vad`, `accelerate_or_replace_vad`, `add_hard_boundary` create final hard
  logical boundaries (Section 11.2).
- Harm flags (Section 12.3) are an independent axis stored on the action; a matched action is
  not presumed harmless (invariant 17). Flags: `harmful_active_split`, `lexical_split`,
  `same_speaker_pause_split`, `duplicate_hard_boundary`, `structural_split`,
  `overlap_hard_action`, `unscored_action`.
- Hard and soft outputs from one cluster cannot both create product actions (Section 9.2/10);
  a final action belongs to at most one cluster (invariant 4).

### 6.5 Frozen tables frozen now

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

## 7. Assumptions relevant to Phase 0 (causal, split, state, statistical)

- Causal: proposals/actions never use future audio or reference labels (Sections 8, 10, 11);
  `boundary_source_sample <= observed_source_sample_at_emit` for every event and control
  action (invariant 23). Controls may use only product-observable causal information
  (Section 7.3).
- Split/held-out: `diagnostic_dev` and `frontier_dev` are group-disjoint at the strongest
  available grouping (Section 16.4); confirmatory held-out paths are not opened until the
  Phase 6 freeze and Phase 7 pre-access approval (Sections 17, 22). Phase 0 opens nothing.
- State: reset-plus-warm-up scored evaluation is forbidden until the Section 5.4
  state-equivalence contract passes for the family/profile class; otherwise source-prefix
  state (invariant 26). Phase 0 does not evaluate any family.
- Statistical: source session (or synthetic source-connected block) is the primary
  uncertainty unit; bootstrap resamples blocks (Section 21; invariant 28). Target-enriched
  exposure is never rescaled to natural rates (invariant 30).
- Historical artifacts are evidence/cached model output only (Section 0, 16.1); none is
  treated as normative corrected results.

## 8. Falsification and stop conditions for Phase 0

- A B0/B1 difference in the Phase 5 equivalence test invalidates the run as a policy result
  (Section 31.2) — Phase 0 freezes the contract so that test is defined now.
- Phase 0 fails if the schemas cannot express the Section 8/4.10/6/11/12 contracts above, or
  if any frozen table (Section 6.5) is ambiguous.
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

## 11. Proposed post-approval implementation (informational)

- `experiments/speaker_turn_boundary/turn_episode/` package: schema dataclasses with
  construction-time validation (`schemas.py`) and pure invariant/contract functions plus the
  frozen tables (`contracts.py`).
- `tests/test_turn_episode_contracts.py`: scientific contract tests for Section 28 invariants
  implementable without data, clustering, fusion, or model execution (currently identified:
  1, 4, 5, 7, 8, 13, 17, 24, 25, 26, 34, 35 and the frozen-table consistency tests).
- `results/turn_episode_v1/proposal_contract.json` and `fusion_contract.json`: frozen schema
  version, field definitions, and profile/extractor registries (extractor registry completed
  at Phase 4 per Section 18.3).
- `results/turn_episode_v1/reviews/phase_0_pre_execution.md`: the review artifact.
