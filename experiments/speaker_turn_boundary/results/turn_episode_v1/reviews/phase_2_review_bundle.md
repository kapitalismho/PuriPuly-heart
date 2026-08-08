# Phase 2 pre-execution review bundle — episode/reference implementation

Status: review bundle for the mandatory Phase 2 pre-execution review (PRD Section 29,
Phase 2; immediate implementation order Section 34 steps 6-7). The Phase 0 and Phase 1
reviews are approved (`reviews/phase_0_pre_execution.md`, `reviews/phase_1_pre_execution.md`);
the Phase 1 exit gate was verified (including the approved AMI data addition, addendum rev 2).
The scored episode/reference manifests and the natural-exposure window manifest have **not**
been generated yet, and no stateful model replay has run on episodes.

Revision history: rev 1 initial bundle (candidate HEAD at review time).

## 1. Artifacts under review

| Item | Value |
| --- | --- |
| Normative plan | `.agents/specs/prd/bounded_turn_episode_speaker_change_fusion_experiment_review_gated.md` |
| Plan git blob | `24340f488f1bb46c666a5fc15eef2fc87ef1f826` |
| Plan self-hash | `8c6bed2e564b9ec80e26ee6b73701985c863b7beb68e51590be7a5faf173aad4` |
| Restart commit (base) | `fef0a6b312df34680d9db0fd858e28ae054ace89` |
| Work branch | `experiment-v2-speaker-change-turn-boundaries-ls` |
| Review range | `fef0a6b3..HEAD` on the work branch; the exact candidate head is the branch HEAD at review time and is confirmed by the reviewer via `git rev-parse HEAD`. |
| Phase 1 evidence this phase depends on | `reviews/phase_1_pre_execution.md` (approved), `reviews/phase_1_addendum_data_addition.md` (rev 2, approved), `coverage_inventory.json` (self-hashed, exit-gate verified), `coverage_inventory_details.jsonl`, `ami_materialization_manifest.json`, `b0_inventory_replay/{20 sessions}` |
| Historical hash ledger | `reviews/historical_artifact_ledger.json` |

## 2. Phase scope and explicit non-goals

Scope (Phase 2):

- Bounded-episode builder per PRD Section 5.1 over the 20 opened scorable sessions
  (12 AMI + 8 AliMeeting), consuming the Phase 1 frozen target-enriched selection
  (`coverage_inventory.json` `target_enriched.per_session`).
- Interval-valued reference construction (clean/gap hard, overlap soft, neutral pause,
  structural, unscored) per Sections 6, 6.7 into hard/soft/neutral/unscored timelines.
- Deterministic manifests: `episode_manifest_dev.json` (diagnostic_dev + frontier_dev
  episodes), `natural_exposure_manifest.json` (unbiased windows over the opened sessions),
  both self-hashed per Section 27.3.
- State-equivalence fixtures per Section 5.4: source-prefix versus episode-reset parity
  executed for the B0 family/profile class (the only executable family now), tolerance
  frozen, snapshot-fallback path implemented and round-trip-tested for B0.
- Contamination/harm scoring fixtures per Sections 13-14 (pure functions + known-answer
  fixtures + B0 end-to-end smoke over the 20 sessions, labeled baseline/dev evidence).
- Sampled waveform/annotation audit (episode audio slices sample-identical to the source
  wav; episode references re-derived from source annotations match exactly).
- Diagnostic/frontier group-disjointness split of the 20 opened sessions with a frozen
  deterministic rule and fail-closed cross-split overlap check (invariants 27, 29).
- Pool tagging per Section 5.1 (`hard_only`, `overlap_present`, `negative_only`) and
  truncated warm-up/tail recording (never silently accepted).

Non-goals (Phase 2):

- No confirmatory held-out access: the 8 reserved AMI sessions stay unopened
  (invariants 29, 31; Phase 7). `episode_manifest_heldout.json` is **deferred** to the
  Phase 6 freeze / Phase 7 pre-access gate and is recorded as such, not generated.
- No LS/ERes neural inference (Phase 4), no clustering/fusion replay (Phase 5), no
  provider oracle grid (Phase 3), no VAD-fusion policy sweep.
- No B0/B1 structural-engine construction (Phase 3). B1 equivalence is out of scope here.
- No natural five-minute/session rate estimation (Phase 7+ reporting); the Phase 2
  natural manifest only freezes window placement before label-conditioned inspection.
- No data additions or sampling-rule changes.
- No new production code; all work under `experiments/speaker_turn_boundary` (Section 33).

## 3. Prior-phase evidence the phase depends on

- Phase 0 (approved): `turn_episode/schemas.py` (ReferenceAction, FinalAction,
  DetectorProgress, ProposalEvent), `turn_episode/contracts.py` (invariant functions),
  `proposal_contract.json`, `fusion_contract.json`.
- Phase 1 (approved, exit gate verified at range `fef0a6b3..8d6eac02`):
  - `coverage_inventory.json`: 20 scorable sessions, group graph (`component_sessions`,
    `graph_hash`), natural-exposure frame (30 s windows, 1/16 hash inclusion, computed
    before label inspection), frozen target-enriched selection
    (142 hard-positive + 79 negative anchors, ≤12 per session per pool).
  - `coverage_inventory_details.jsonl`: per-session regions, targets (interval-valued
    acceptable intervals per Section 6.2), overlap soft targets, same-speaker pause
    intervals, wav/annotation SHA-256, B0 classification.
  - `b0_inventory_replay/{20}.json`: full-source B0 Silero VAD traces (canonical
    projection, trace hash) — these serve as the `source_prefix` evidence for the
    B0 state-equivalence test.
  - `ami_materialization_manifest.json`: 16 entries (8 development, 8 reserved),
    sha256/size/duration verified on disk.

## 4. Bounded-episode extraction rules (frozen, Section 5.1)

All rules are deterministic and frozen before generation.

### 4.1 Anchor set

- Anchors = `coverage_inventory.json` `target_enriched.per_session[<session>]`
  `hard_positive_selected` (clean/gap hard targets, 142) and `negative_selected`
  (same-speaker pause intervals, 79). Each anchor carries its frozen `selection_digest`
  and rank; selection is never re-run.
- Anchors are processed in deterministic order: per session, sorted by
  (anchor sample, kind, rank).

### 4.2 Candidate window per anchor

With `T` the anchor sample (target sample for hard positives; silence interval midpoint
for negatives) and `session_end` the wav duration in samples:

- scored region `S = [max(0, T - 5 s), min(session_end, T + 5 s)]` (10 s scored default,
  Section 5.1 "10-20 seconds of scored audio");
- full window `W = [max(0, S.start - 5 s), min(session_end, S.end + 3 s)]`
  (≥5 s warm-up before the first scored interval, ≥3 s tail after the last scored target
  "when source context permits").
- Truncation (session start or end) is **recorded**, never silently accepted:
  `warmup_truncated = S.start - W.start < 5 s`, `tail_truncated = W.end - S.end < 3 s`.

### 4.3 Merging and non-overlap

- Two candidate windows that overlap in source time (per session) are merged into one
  episode (union of scored regions, union of full windows). Merging is applied in
  deterministic anchor order.
- A merged full window longer than 30 s is split **only** at an annotated stable
  same-speaker or silence interval boundary at least 2 s away from every hard/soft
  target inside the window (Section 5.1); the split candidate nearest the window
  midpoint is chosen (earliest on ties). If no split candidate exists, the merged
  episode is kept with the exceeding part recorded (`merged_over_30s_no_split_point`)
  and the anchor order preserved.
- **Invariant (fails closed):** after merging, no two scored episodes within the same
  pool share any source sample (`assert` + manifest check, Section 5.1 "no source sample
  appears in more than one scored episode within the same pool").
- Episodes from different sessions can never overlap (different source time domains);
  episodes from different pools come from disjoint sessions (Section 9), so cross-pool
  overlap is impossible by construction and asserted.

### 4.4 Pool tags (Section 5.1)

Per episode, from the scored region's reference timeline (Section 5):

- `overlap_present`: scored region contains an overlap reference or stable overlap
  interval;
- `hard_only`: contains clean/gap hard references and no overlap reference/stable
  overlap;
- `negative_only`: no different-speaker hard reference in the scored region.
- Precedence: `overlap_present` > `hard_only` > `negative_only`.
- An episode whose warm-up frontier is unstable (warm-up < 5 s available because the
  anchor lies < 5 s into the session) is tagged `diagnostic_only`: it is listed in the
  manifest with `scorable=false` and its anchors are recorded as unscored; it never
  enters the target-enriched scored pools (Section 5.1 last paragraph).

### 4.5 Expected counts (estimate, frozen at generation)

- Anchors: 142 hard-positive + 79 negative = 221.
- Estimated merged episodes: ~189 (merge estimate per the rules above; the exact count
  is fixed by the builder output and asserted against the per-session anchor counts).
- Diagnostic-only episodes: expected small (anchors within 5 s of a session start);
  recorded individually.

## 5. Reference construction (hard/soft/neutral/unscored timelines, Sections 6, 6.7)

The Phase 1 classifier `_classify_targets` already produces interval-valued targets with
the Section 6.2 semantics; Phase 2 wraps it into `ReferenceAction` objects per
`turn_episode/schemas.py` and attaches per-episode timelines.

### 5.1 Reference kinds (frozen)

| Source pattern | action_kind | target_sample | acceptable_interval | evidence_onset | primary |
| --- | --- | --- | --- | --- | --- |
| clean handoff `{A}->{B}` | `hard_boundary` | B onset | `[B onset - 500 ms, B onset]` | B onset | yes |
| gap handoff `{A}->{}->{B}`, A != B | `hard_boundary` | B onset | `[A speech offset, B onset]` | B onset | yes |
| interruption `{A}->{A,B}` | `soft_overlap_marker` | B onset | `[B onset - 500 ms, B onset]` | B onset | no |
| departure `{A,B}->{B}` | `state_update` | None | `[change sample, change sample]` | change sample | no |
| same-speaker pause `{A}->{}->{A}` | `neutral_pause` | None | `[A offset, A next onset]` | next onset | no |
| structural (session start, episode edges, VAD max-duration, terminal flush) | `structural` | boundary sample | `[sample, sample]` | sample | no |
| unscored (ambiguous annotation, missing speaker coverage, channel misalignment, insufficient word timing) | `unscored` | None | covering interval | interval start | no |

- Clean/gap hard references are the only `primary_case=true` references and the only
  ones entering the clean/gap headline stratum (Sections 12.1, 13.4).
- `overlap_present` episodes cannot enter the primary clean/gap contamination headline
  (invariant 11); overlap soft references cannot raise clean/gap hard-boundary headline
  recall (invariant 10).
- `unscored` references create `unscored_action` (invariant 13: no benefit or harm
  numerator credit). Ambiguous regions (`SpeakerRegion.ambiguous`) and words with
  missing/insufficient timing produce unscored intervals; missing word timing is never
  treated as absence of lexical harm (invariant 19).

### 5.2 Episode timeline

Per episode: references whose acceptable interval or target intersects the episode's
scored region. References in the warm-up region are excluded from the reference
timeline entirely (Section 5.3 "exclude warm-up actions and references from headline
counts"; warm-up is unscored by construction). Each reference carries
`episode_pool_tag` matching its episode tag.

### 5.3 Determinism and provenance

- Reference ids: `"{session_id}:{episode_id}:{gt_index}"` (hard/soft) /
  `"{session_id}:{episode_id}:pause:{rank}"` / structural/unscored suffixes — unique,
  deterministic.
- Per-episode reference timeline hash + per-manifest content SHA-256 (Section 27.3);
  each episode records wav SHA-256, annotation SHA-256, selection digest, and
  `episode_manifest` identity so the cache contract (Section 27.2) binds correctly.

## 6. Gap interval-valued matching and overlap exclusion (Sections 12.1-12.2, 13.4-13.5)

- Fixtures re-verify `contracts.py` matching rules against built references:
  - any final hard boundary inside `[A speech offset, B onset]` matches the gap target
    (invariant 7) with zero localization error inside the interval and distance to the
    nearest edge outside it (Section 6.2);
  - a detector proposal before B onset receives no gap speaker-change evidence credit
    (invariant 8); a pre-existing VAD gap boundary remains valid product separation
    (invariant 9);
  - matching is ordered one-to-one within an epoch (invariant 6); contamination is
    recomputed after matching and never an input to matching (Section 12.2).
- Primary hard localization tolerance 500 ms, 250 ms view (Section 12.1) — frozen in
  `contracts.py` (`LOCALIZATION_TOLERANCE_MS_PRIMARY/VIEW`), unchanged.
- Overlap exclusion: the clean/gap headline stratum = `hard_only` episodes only; overlap
  soft references and `overlap_present` episodes are excluded from the primary
  contamination denominator (Section 13.5); `overlap_hard_action` reported separately.

## 7. Turn-owner threshold and annotation-jitter sensitivity plan (Sections 13.2, 13.4)

- Turn-owner threshold frozen at 100 ms of continuous scorable singleton speech
  (`TURN_OWNER_THRESHOLD_MS = 100`, Section 13.2); mandatory sensitivity views at 50 and
  200 ms are part of the scoring fixtures (reported, never replacing the primary label).
- Mixed-turn reporting threshold frozen at 250 ms of a second singleton speaker, with
  required tiers at 100 / 250 / 500 ms (Section 13.4).
- Contamination algorithm fixture per Section 13.3 (first qualifying singleton owner;
  subsequent different qualifying singleton speech is contamination including a later
  return of the original speaker; no double counting, invariant 14; premature split
  before the handoff gets no false credit, invariant 15).
- `segment_contamination`, `turn_owner_requires_threshold`, and
  `premature_split_receives_no_false_credit` in `contracts.py` are the implementation
  anchors; known-answer fixtures freeze edge cases (A->B->C segments, premature split,
  sub-threshold singleton runs).

## 8. State-equivalence test design, tolerance, and source-prefix fallback (Sections 5.3, 5.4, invariant 26)

### 8.1 Family/profile classes

- Executable now: `b0` (Silero VAD peer profile; model SHA-256
  `1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3`, frozen in Phase 1).
- Declared, not executed until Phase 4 (checkpoints pinned there): `ls_eend/*`,
  `eres2netv2/*` per `proposal_contract.json`.

### 8.2 Parity procedure (per family/profile class, per episode class)

For each episode of the class:

1. `source_prefix`: replay the original source wav from sample 0 through the target
   region (Phase 1 full-session B0 traces already provide this evidence for B0;
   replayed deterministically by the fixture if needed).
2. `episode_reset`: reset all state at the extracted episode start, replay the declared
   warm-up, then score the same target region.

Compare at aligned source coordinates:

- B0: raw boundary trace within the scored region — `(boundary_source_sample,
  observed_source_sample_at_emit)` identity, exact match on the canonical projection
  (tolerance 0; Silero VAD is deterministic given identical audio, so warm-up
  sufficiency is what the test verifies);
- LS/ERes (declared now, executed Phase 4): raw LS posteriors (max L1 over aligned
  frames in the scored region <= 1e-2), ERes embeddings/similarity (aligned-window
  cosine >= 0.99), proposal count/kinds, proposal boundary positions and observation
  frontiers, post-clustering actions and safe-frontier progression.

### 8.3 Frozen tolerance

Tolerances are frozen here **before** any comparison: B0 exact; LS posterior max L1
1e-2; ERes aligned-window cosine 0.99; proposal/action identity (count, kinds,
boundary samples, frontiers) exact. A failed parity case remains diagnostic evidence
and cannot be hidden by increasing warm-up post hoc (Section 5.4).

### 8.4 Disposition rule (invariant 26)

- A class passes → reset-plus-warm-up scoring allowed for that class.
- A class fails → scored episodes for that class must start from deterministic
  source-prefix state snapshots or source-prefix replay; recorded per class.
- Default before a class is tested: **source-prefix state required** (reset not
  permitted). LS/ERes classes therefore default to source-prefix until their Phase 4
  parity passes.
- The Phase 2 gate requires: every family/profile class used for reset-based scoring
  passes the state-equivalence gate or is switched to source-prefix. Phase 2 executes
  the gate for B0 only; the disposition table records `b0` pass/fail and LS/ERes
  `source_prefix_default` pending Phase 4.

### 8.5 Snapshot fallback path

- `mode=source_prefix`: deterministic full-source replay per episode
  (`replay_wav_epoch` from sample 0; evidence identity binds checkpoint, frontend,
  audio hash, episode manifest hash, capture payload hash — Section 27.2).
- Snapshot mechanism: serialized engine state captured at episode start from a
  full-source pass; round-trip test (capture -> restore -> resume) must reproduce the
  full-source trace exactly. Implemented and tested for the B0 engine in Phase 2;
  generic interface declared for LS/ERes (Phase 4).
- Cache identity binds the snapshot to (checkpoint/frontend hashes, source audio hash,
  episode manifest hash) so a restored snapshot is never reused across a changed
  contract.

## 9. Diagnostic/frontier/held-out group-disjointness (Sections 16.4, 17; invariants 27, 29, 31)

- The 20 opened scorable sessions form 20 distinct group components (verified in the
  Phase 1 group graph: no two scorable sessions share a component; annotation-only
  siblings in a component never produce episodes).
- Frozen pool split: per corpus (ami, alimeeting), sort scorable session ids; assign
  alternately by rank (even -> `diagnostic_dev`, odd -> `frontier_dev`). The split is
  deterministic, corpus-stratified, and asserts that no component is split across pools
  (invariant 27). Synthetic manifests (`ls_dev`, `ls_held_out_clean`, `ls_held_out_other`,
  `mixed_dev_pool`) are assigned wholly to `diagnostic_dev` (groups `synthetic:<name>`);
  their episodes already exist as complete cases and are registered in the dev manifest
  without re-extraction (Section 5.2).
- `historical_validation` status is recorded per session (touched history) but does not
  exclude a session from development pools; it only bars confirmatory claims (Section
  16.4).
- Cross-split overlap fails closed (invariant 29): the manifest generator asserts, per
  session and per episode, that all episodes of a session belong to exactly one pool and
  that the group graph hash bound in Phase 1 is unchanged.
- Confirmatory held-out: the 8 reserved AMI sessions remain **unopened** (no words
  parsing, no region extraction, no label access). `episode_manifest_heldout.json` is
  not generated in Phase 2; the manifest generator records the deferral and the frozen
  rule (invariant 31: held-out cannot open without a valid frozen self-hash from the
  Phase 6 freeze).

## 10. Natural-exposure manifest generation (Section 16.4; invariant 30)

- The window frame was frozen in Phase 1 (30 s grid from sample 0; inclusion iff
  `int(sha256(f"{session_id}:{start_ms}").hexdigest()[:2], 16) < 16`; computed from
  durations only, before any transition-label inspection).
- Phase 2 materializes `natural_exposure_manifest.json` for the **20 opened scorable
  sessions only**: per included window, session_id, start_ms, sample bounds, eligibility,
  sampled/eligible duration, wav/annotation hashes. The selection rule is re-applied and
  asserted equal to the Phase 1 frame for those sessions (determinism proof).
- Windows over the 8 reserved sessions remain duration-only entries in the Phase 1 frame
  and are **not** opened (they belong to the held-out natural frame, Phase 7).
- Window placement never uses transition labels (the rule is source-time only); the
  manifest is generated before any transition-conditioned inspection, and the generator
  asserts it never reads the label timeline to place a window.
- Natural windows are replayed with the same bounded replay machinery and
  state-equivalence contract (Section 16.4); five-minute/session/source-hour rates may
  be estimated only from this pool (invariant 30), and Phase 2 does not estimate them.

## 11. Sampled waveform/annotation audit procedure (Phase 2 gate)

- Deterministic sample: per pool, keep episode iff
  `int(sha256(episode_id).hexdigest()[:2], 16) < 8` (1/32), with a floor of 8 episodes
  per pool (smallest-hash fill) — frozen before audit.
- Waveform audit: re-open the source wav, extract `[episode_start, episode_end)`
  samples, and compare **byte-for-byte** against the episode slice the builder
  recorded; also verify wav header (PCM 16-bit mono 16 kHz).
- Annotation audit: independently re-derive the reference timeline for the episode span
  directly from the source annotations (AMI `words.xml` set; AliMeeting TextGrid
  interval tiers), independent of the builder's cached regions, and require exact
  equality of the reference set (ids, kinds, target samples, acceptable intervals,
  scorable flags).
- Any audit failure fails the phase (stop condition); the audit report records
  per-episode pass/fail with mismatch details.

## 12. Contamination/harm scoring fixtures (Sections 13-14)

- Pure-function scorer over (final hard boundaries, reference timeline, episode):
  logical segmentation (13.1), turn ownership (100 ms primary, 50/200 ms views),
  contamination algorithm (13.3), clean/gap headline ratios and mixed-turn tiers
  (13.4), harm flags (14.1-14.5), fragmentation metrics (14.6) — implemented in
  `turn_episode/scoring.py` wrapping `contracts.py`.
- Known-answer fixtures: hand-built tiny timelines covering invariants 7-20 (gap
  interval match, B-onset credit, overlap exclusion, warm-up exclusion, unscored
  exclusion, no double count, premature-split no-credit, turn-owner thresholds, harm
  orthogonality, lexical-split missing-timing semantics, same-speaker pause split
  counting, duplicate, overlap hard action).
- B0 end-to-end smoke: run the scorer on the Phase 1 B0 full-session traces over the
  20 sessions (B0 hard boundaries = raw VAD boundaries), producing baseline
  contamination/harm rows. These are **fixture verification / baseline dev evidence**
  labeled `baseline`, never confirmatory claims and never natural rates (Section 13.6).

## 13. Deterministic manifests (Section 27.3)

- `episode_manifest_dev.json`: all diagnostic_dev + frontier_dev episodes (public + 
  synthetic registrations), self-hashed; per-episode: id, pool, session, epoch, sample
  bounds (warm-up/scored/full), tag, references, selection digest, hashes.
- `natural_exposure_manifest.json`: natural windows (Section 10), self-hashed.
- `state_equivalence_report.json`: per-class disposition, parity results, tolerance
  constants, snapshot round-trip evidence (Section 8).
- `scoring_fixture_report.json`: known-answer results + B0 baseline smoke (Section 12).
- `audit_report.json`: sampled audit results (Section 11).
- `episode_manifest_heldout.json`: **deferred** to Phase 6/7 (recorded deferral entry in
  the dev manifest header).
- All JSON artifacts carry canonical content hashes; row files record direct byte
  SHA-256 (Section 27.3).
- New code under `turn_episode/`: `build_episodes.py`, `state_equivalence.py`,
  `scoring.py`, `audit.py` (plus `contracts.py` extensions for frozen tolerances and
  pool-split constants).

## 14. Falsification and stop conditions for Phase 2

- Any cross-pool source-sample overlap, any session assigned to two pools, any
  component split across pools, or any change to the Phase 1 group graph hash → phase
  stops, fails closed.
- Non-overlap assertion among scored episodes within a pool fails → phase stops.
- Reference timeline rebuild disagrees with the Phase 1 classifier output on any shared
  target → phase stops (determinism defect).
- Audit sample mismatch (waveform bytes or annotation re-derivation) → phase stops.
- B0 state-equivalence failure for an episode class → that class is recorded
  `source_prefix_required`; the fixture report must show the disposition table complete
  with no untested class used for reset-based scoring.
- Natural-exposure manifest selection differs from the Phase 1 frame for the opened
  sessions → phase stops (frame was frozen before label inspection; a difference means
  the frame was regenerated).
- Reserved sessions: any attempt to parse/read their annotations or audio in the
  builder/manifest code fails closed with an explicit error (invariant 31).

## 15. Expected compute/data/provider cost and irreversible access

- Compute: episode building/classification CPU minutes; B0 episode-reset replays over
  ~189 episodes (~13-18 s audio each) CPU well under an hour; scoring/audit CPU minutes.
  No GPU, no provider, no credentials, no downloads.
- Irreversible access: none. Reserved AMI sessions are never opened; writes only under
  `results/turn_episode_v1/`.

## 16. Reviewer examination checklist (from PRD Phase 2)

1. Bounded-episode extraction rules and non-overlap guarantees (Section 4).
2. Hard/soft/neutral/unscored reference construction (Section 5).
3. Gap interval-valued matching and overlap exclusion from the hard headline (Section 6).
4. Turn-owner threshold and annotation-jitter sensitivity plan (Section 7).
5. State-equivalence test design, tolerance, and source-prefix fallback (Section 8).
6. Diagnostic/frontier/held-out group-disjointness (Section 9).
7. Natural-exposure manifest generation before transition-conditioned inspection (Section 10).
8. Sampled waveform/annotation audit procedure (Section 11).

## 17. Recorded review findings and dispositions

| id | severity | finding | disposition |
| --- | --- | --- | --- |
| (none yet) | | | |
