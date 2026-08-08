# Phase 2 pre-execution review bundle — episode/reference implementation

Status: review bundle for the mandatory Phase 2 pre-execution review (PRD Section 29,
Phase 2; immediate implementation order Section 34 steps 6-7). The Phase 0 and Phase 1
reviews are approved (`reviews/phase_0_pre_execution.md`, `reviews/phase_1_pre_execution.md`);
the Phase 1 exit gate was verified (including the approved AMI data addition, addendum rev 2).
The scored episode/reference manifests and the natural-exposure window manifest have **not**
been generated yet, and no stateful model replay has run on episodes.

Revision history: rev 1 initial bundle (candidate HEAD `22c45dd9`); rev 2 resolves
P2-001 through P2-013 (candidate `0ad094df`); rev 3 (this revision) resolves P2-014
through P2-026 per the round-2 review (candidate HEAD at review time, confirmed via
`git rev-parse HEAD`).

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

### 1.1 Exact input/code/config hash ledger (frozen, finding P2-013)

| Input / code | SHA-256 |
| --- | --- |
| `turn_episode/contracts.py` | `b207d3f8b9720df5dd228aa8bd8b479c54622abb905a9ca04f580820a6fc3c03` |
| `turn_episode/schemas.py` | `8c449b2ed07fba11bb1e45f01cad6b22fe1c98eb8006a2600bee90170f45f2f9` |
| `turn_episode/build_coverage_inventory.py` | `dd360c9e60a5838feaea17e4b335d1fc93cdbd6df4f426077a2eedbd30e1a1e7` |
| `turn_episode/materialize_ami_additions.py` | `bf431bb5b22ec79032ee6fbe876d5ab330893521536481e590f9b054932ccdc7` |
| `vad_baseline.py` | `7a3965fdb01eb7391dde985e5c498162d80b4e5ab565205626d684a66d8ff627` |
| `events.py` | `2193bda0f06ff9e3d4171402c9ce2296ed273f10994de35332ca070d212b347a` |
| `config.py` | `f4eb24e6c81ebcb0bdd71b6c0c9098595ae4bdddf53e05df6bd8eea925d146a6` |
| `ground_truth.py` | `34d2236595c4fb3e105b1aa5da8b4fa05e513f33979ca63c8c6903299d0f820d` |
| `corpus/phase2_schemas.py` | `7a6b4b0c9033b5ebdc97db552943c522a5218f5166e039db4b37f6744861dcf2` |
| `src/puripuly_heart/core/vad/silero.py` | `43079df5bc36ecb924b1aec7991cff2a16c04ab126bb54907c4b2a570e2cd109` |
| Silero ONNX model | `1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3` |
| `coverage_inventory.json` (bytes) | `02a6a118fc90c0d747e9548f07003177b3fc703f33d408d5338427cb6163dd46` |
| `coverage_inventory_details.jsonl` (bytes) | `15b2e4f0efa270985c3bbc6d848ee9ed25496089268e561bff921c5c1be3ef8c` |
| `ami_materialization_manifest.json` (bytes) | `06fe15fff87bb78218df2c086bd711590378f8741164909b59704f56841ab6c9` |
| `corpus/ami.py` | `e77171299cb7d358f2a3f029e78386d91492e94e49ca683bac90ac5173c4d0dc` |
| `corpus/alimeeting.py` | `f69b733c703e0060245a2d9229258caddc602238c18c7c242570713670ea929a` |
| `corpus/librispeech.py` | `05b77dc735f572da37da0c9ebe595415f5acf4b5c40bd22e3f792dd31402d1f4` |
| `corpus/external.py` | `262d40233371905e6a6fac2efee2da1412c6d6df44d91ea4eb8fd2220ba21b56` |
| `corpus/mixing.py` | `ffd707787003051f5e95e9dabb70afb4eb675b970461e75cd6c062d7f121abc6` |
| `corpus/puripuly_like.py` | `5f5df4ce0879bd5f7ffa35d571e700852a490803bd966c9f7a90c0f4ee848642` |
| `corpus/validation.py` | `66d863a0ada96fcc765097dd574ea4b2edfc5eed6ed873aee13df34d65ff964d` |
| `data/manifests/ls_dev.json` | `14347cdbdb2eff4cc73489f1b59d6755723d9098089dad66ae222984e90370dd` |
| `data/manifests/ls_held_out_clean.json` | `c0aabc5ad8c3f00ec53d45f3b372b8ebca7ca9237720a1bb7a70b8de7dda2581` |
| `data/manifests/ls_held_out_other.json` | `f0d169394a9fdee9e708bc9cad46c0547946bf967799fa4e2e1a398ddb984079` |
| `data/manifests/mixed_dev_pool.json` | `1221176c92f50a2b096e4cd64d5da0168527918e3fba539273c614eabf07a398` |
| `proposal_contract.json` | `0448edd933fd1d9d0a0b4d5f9f2631cb0f630c892fc4d46e1a3ec9740e80b7fb` |
| `fusion_contract.json` | `bfda0c3c0ea7b6613ded79e9639692a33449dcf34202b1f2a5e7ec14c45f9873` |

Any change to these inputs invalidates the Phase 2 outputs bound to them; the Phase 2
manifests record these hashes in their provenance headers.

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
- Pool tagging per Section 5.1 (`hard_only`, `overlap_present`, `negative_only`) plus an
  explicit episode status field (`scorable` / `diagnostic_only`), and truncated
  warm-up/tail/scored coverage recording (never silently accepted).

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
    projection, trace hash). Phase 2 re-runs B0 in both parity modes (Section 8.2); the
    Phase 1 traces are used only as a cross-check, not as the parity evidence (finding
    P2-008).
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

With `T` the anchor sample (target sample for hard positives; silence midpoint for
negatives, computed as `silence_start + (silence_end - silence_start) // 2`, integer
floor division — frozen, finding P2-007) and `session_end` the wav duration in samples:

- scored region `S = [max(0, T - 5 s), min(session_end, T + 5 s)]` (10 s scored default,
  Section 5.1 "10-20 seconds of scored audio");
- full window `W = [max(0, S.start - 5 s), min(session_end, S.end + 3 s)]`
  (≥5 s warm-up before the first scored interval, ≥3 s tail after the last scored target
  "when source context permits").
- Truncation (session start or end) is **recorded**, never silently accepted:
  `warmup_truncated = S.start - W.start < 5 s`, `tail_truncated = W.end - S.end < 3 s`.

### 4.3 Merging, non-overlap, 30 s cap, and 10-20 s scored rule (findings P2-001, P2-004)

- Two candidate windows that overlap in source time (per session) are merged into one
  episode (union of scored regions, union of full windows). Merging is applied in
  deterministic anchor order.
- **Scored-exposure rule (frozen):** every `scorable` episode must have a scored region
  of at least 10 s and at most 20 s.
  - Single-anchor episodes: scored region is `[max(0, T - 5 s), min(session_end, T + 5 s)]`;
    if that region is shorter than 10 s (session too short or end-clipping), the episode
    is **not scorable**: it is emitted `diagnostic_only` with recorded reason
    `scored_truncated` (finding P2-004; truncated coverage recorded, never silently
    accepted).
  - Merged episodes: the union of scored regions is capped at 20 s by trimming from the
    end of the union (deterministic). Any selected anchor whose own scored interval
    falls outside the capped scored region is recorded as `coverage_loss` (its target is
    not scored in this episode); the loss is counted and reported in the manifest
    (Section 5.1: truncated coverage is recorded, never silently accepted). If the
    capped scored region is still shorter than 10 s, the episode is `diagnostic_only`
    with reason `scored_truncated`.
  - A merged episode whose scored region is between 10 s and 20 s inclusive is scorable.
- **30 s cap (frozen):** a `scorable` episode never exceeds 30 s total (full window).
  - If a merged full window exceeds 30 s, it is split **only** at an annotated stable
    same-speaker or silence interval boundary at least 2 s away from every hard/soft
    target inside the window (Section 5.1); the split candidate nearest the window
    midpoint is chosen (earliest on ties). Both parts must again satisfy the scored
    rule; a part that cannot is emitted `diagnostic_only` with recorded reason.
  - **If no valid split candidate exists, a deterministic re-bucketing algorithm
    (frozen, finding P2-015) runs instead:** anchors of the merged group are kept in
    sorted order; the last anchor of the group is removed (with `coverage_loss`
    reason `no_split_point`) and the union re-computed; this repeats until the union is
    <= 30 s. Removed anchors are **never emitted** as episodes (no overlap can be
    created by them). The kept group emits one episode. This is applied per merged
    group; the resulting emitted episodes are pairwise disjoint **by construction**
    (each group's emitted window is the union of kept anchors <= 30 s and no other
    episode contains samples of removed anchors) **and** the builder additionally
    asserts pairwise disjointness of all emitted episodes within the pool, failing
    closed on any violation. If the disjointness assertion cannot be satisfied for a
    group even after tail-removal, the whole group is emitted `diagnostic_only` with
    `coverage_loss` for every anchor (fail-closed, never a > 30 s or overlapping
    scorable episode).
- **Invariant (fails closed):** after merging, no two scored episodes within the same
  pool share any source sample (Section 5.1 "no source sample appears in more than one
  scored episode within the same pool"). Episodes from different sessions can never
  overlap (different source time domains); episodes from different pools come from
  disjoint sessions (Section 9), so cross-pool overlap is impossible by construction
  and asserted.

### 4.4 Episode status, pool tags, and scoring-start readiness (findings P2-002, P2-003)

- **Episode status** is a separate field from the pool tag (finding P2-003): status
  `scorable` | `diagnostic_only` (new `EpisodeStatus` literal in the Phase 2 manifest
  schema; the existing `EpisodePoolTag` remains `hard_only` | `overlap_present` |
  `negative_only` for scorable episodes). A `diagnostic_only` episode keeps its
  underlying pool tag for bookkeeping, has `scorable=false`, and never enters the
  target-enriched scored pools; its references are emitted with `scorable=false` and
  are excluded from every scored numerator and denominator (invariant 12).
- **Pool tags (Section 5.1), per scorable episode, from the scored region's reference
  timeline (Section 5):** `overlap_present` if the scored region contains an overlap
  reference or stable overlap interval; else `hard_only` if it contains clean/gap hard
  references; else `negative_only`. Precedence: `overlap_present` > `hard_only` >
  `negative_only`.
- **Scoring-start readiness predicate (frozen, finding P2-002, P2-016):** per
  family/profile class, a frozen predicate `scoring_start_ready(class, episode)` must
  be provable from **replay-derived detector progress evidence** (never from annotation
  boundaries alone) before the episode is scorable:
  - `b0/peer`: after replaying the declared warm-up in `episode_reset` mode, the
    engine's DetectorProgress must show `safe_boundary_frontier_sample <= scored_start`
    at the scored start (PRD Section 4.10 semantics: no pending VAD segment, pre-roll,
    or hangover can emit a boundary inside the scored region from warm-up audio), with
    the progress trace validated monotonic and conservative (invariant 35). If the
    safe frontier cannot be proven at or before the scored start, the episode is
    `diagnostic_only` with reason `unstable_warmup_frontier`. The readiness predicate
    therefore runs the same replay machinery as the state-equivalence test (Section 8)
    and is derived from the same evidence.
  - `ls_eend/*`, `eres2netv2/*`: declared at Phase 4 when checkpoints are pinned; the
    predicate must cover frontend buffering, neural lookback, confirmation, and cluster
    debounce via the safe-frontier contract (Section 4.10) and is frozen there before
    any scored execution. Until then these classes default to source-prefix state
    (Section 8.4).
  - The state-equivalence test (Section 8) is the empirical proof of readiness for the
    scored region; an episode whose class fails parity is switched to source-prefix
    state, never silently kept on reset.
- An episode whose warm-up frontier is unstable (warm-up < 5 s available because the
  anchor lies < 5 s into the session) is tagged `diagnostic_only` (Section 5.1 last
  paragraph), recorded individually.

### 4.5 Expected counts (estimate, frozen at generation)

- Anchors: 142 hard-positive + 79 negative = 221.
- Estimated merged episodes: ~189 (merge estimate per the rules above; the exact count
  is fixed by the builder output and asserted against the per-session anchor counts).
- Diagnostic-only episodes: expected small (anchors within 5 s of a session start, or
  scored/30 s rule failures); recorded individually with reasons.
- Coverage loss: recorded per anchor and summed in the manifest header.

## 5. Reference construction (hard/soft/neutral/unscored timelines, Sections 6, 6.7)

A **new reference builder** (`turn_episode/build_episodes.py::ReferenceBuilder`, finding
P2-005) walks the per-session region timeline (`SpeakerRegion` sequence from the Phase 1
inventory) **plus raw annotation records** (finding P2-018) and produces the
**complete** Phase 2 taxonomy. The Phase 1 classifier (`_classify_targets`) remains the
inventory-count authority; the Phase 2 builder is the authoritative per-episode
reference source and is independently re-derived by the audit (Section 11). The builder
preserves every discarded/ambiguous pattern as an explicit reference (finding P2-005):

- **Raw annotation input (finding P2-018):** because the existing AMI parser discards
  words lacking timing (they never reach the `SpeakerRegion` timeline), the builder
  additionally consumes the raw annotation records — AMI `words.xml` word elements
  (with and without timing) and AliMeeting TextGrid interval records — and derives
  explicit `unscored` intervals from (a) words/spans lacking timing, (b) ambiguous
  regions, (c) missing speaker coverage, and (d) channel-misalignment/coverage gaps
  declared by the corpus parser. Missing annotation coverage is thereby distinguished
  from true silence in the timelines (finding P2-018); the audit (Section 11)
  re-derives the same taxonomy directly from the raw annotation files.

### 5.1 Reference kinds (frozen)

| Source pattern | action_kind | target_sample | acceptable_interval | evidence_onset | primary |
| --- | --- | --- | --- | --- | --- |
| clean handoff `{A}->{B}` | `hard_boundary` | B onset | `[B onset - 500 ms, B onset]` | B onset | yes |
| gap handoff `{A}->{}->{B}`, A != B | `hard_boundary` | B onset | `[A speech offset, B onset]` | B onset | yes |
| interruption `{A}->{A,B}` | `soft_overlap_marker` | B onset | `[B onset - 500 ms, B onset]` | B onset | no |
| departure `{A,B}->{B}` (`speaker_left`) | `state_update` | None | `[change sample, change sample]` | change sample | no |
| same-speaker pause `{A}->{}->{A}` | `neutral_pause` | None | `[A offset, A next onset]` | next onset | no |
| structural (session start, episode edges — Phase 2 scope; VAD max-duration and terminal flushes **explicitly excluded** from Phase 2 timelines/scoring, deferred to Phase 3/8) | `structural` | boundary sample | `[sample, sample]` | sample | no |
| unscored (ambiguous region, missing speaker coverage, channel misalignment, insufficient/missing word timing) | `unscored` | None | covering interval | interval start | no |

- The builder derives departures (`{A,B}->{B}`) and same-speaker pauses that the Phase 1
  classifier did not emit as references, and converts ambiguous regions and
  words lacking timing into explicit `unscored` intervals (never silently skipped;
  finding P2-005). Missing word timing is never treated as absence of lexical harm
  (invariant 19).
- Clean/gap hard references are the only `primary_case=true` references and the only
  ones entering the clean/gap headline stratum (Sections 12.1, 13.4).
- `overlap_present` episodes cannot enter the primary clean/gap contamination headline
  (invariant 11); overlap soft references cannot raise clean/gap hard-boundary headline
  recall (invariant 10).
- `unscored` references create `unscored_action` (invariant 13: no benefit or harm
  numerator credit).
- Stable overlap interval predicate (frozen, finding P2-007): a region with
  `len(speakers) > 1` and `duration >= 100 ms` and `not ambiguous`.
- Stable same-speaker/silence interval predicate for split points (frozen, finding
  P2-007): a region that is singleton same-speaker or empty (silence), `duration >=
  100 ms`, `not ambiguous`, and whose boundary lies at least 2 s (2000 ms) from every
  hard/soft target inside the window.
- Episode edges and the session start produce `structural` references. VAD
  maximum-duration and terminal-flush structural references are **explicitly excluded
  from Phase 2 timelines and scoring** and deferred to Phase 3/8 (finding P2-021):
  Phase 2 does not claim a complete structural taxonomy; the deferral is recorded in
  the manifest header.

### 5.2 Episode timeline

Per episode: references whose acceptable interval or target intersects the episode's
scored region, with **half-open scored clipping** (finding P2-024): a reference is
included iff its acceptable interval intersects `[scored_start, scored_end)` in
half-open terms and its target/evidence onset >= scored_start; an interval crossing
the warm-up/scored boundary is clipped to `[max(interval.start, scored_start),
interval.end]` for matching; a reference whose target lies in warm-up is not scored
(recorded `warmup_excluded`). **All warm-up actions are rejected before matching**
(invariant 12): any final action with boundary < scored_start can never match any
reference in the episode. References in the warm-up region are excluded from the
reference timeline entirely (Section 5.3 "exclude warm-up actions and references from
headline counts"; warm-up is unscored by construction). Each reference carries
`episode_pool_tag` matching its episode tag and `scorable` matching the episode status.

Synthetic episodes (complete cases from the Phase 1 manifests `ls_dev`,
`ls_held_out_clean`, `ls_held_out_other`, `mixed_dev_pool`) are registered in the dev
manifest without re-extraction, **but** their references are re-derived from the
manifest case regions by the same ReferenceBuilder and audited (Sections 5.3, 11,
finding P2-011). The Phase 1 synthetic manifests are the independent annotation
authority for synthetic cases.

**Real-recording exclusion (frozen, finding P2-014):** `mixed_dev_pool.json` contains
two complete real AMI recordings (`ami_ES2003a`, `ami_IS1008a`, wav paths under
`ami/audio/...`). These cases are **excluded from synthetic registration**: their
source samples are already covered by the public sessions ES2003a/IS1008a (scorable
pilot sessions), so registering them as `synthetic:mixed_dev_pool` would place the same
source samples in two pools and violate Section 16.4 group-disjointness and the
non-overlap invariant. Frozen rule: any manifest case whose `wav_relative_path` starts
with `ami/` or `alimeeting/` is not registered as synthetic; it is recorded in the
manifest header as `excluded_real_recording` with its source-session id, and its source
session binds to the corresponding public session component (already covered). The
remaining 202 `mixed_dev_pool` cases are synthetic and registered under
`synthetic:mixed_dev_pool`. The builder asserts that no registered episode's source
samples overlap any other registered episode's source samples within the same pool
(fail-closed, invariant 29).

### 5.3 Determinism and provenance (findings P2-010, P2-011)

- Reference ids: `"{session_id}:{episode_id}:{gt_index}"` (hard/soft) /
  `"{session_id}:{episode_id}:pause:{rank}"` / `...:departure:{index}` /
  `...:structural:{index}` / `...:unscored:{index}` — unique, deterministic.
- **Episode id (frozen, finding P2-010, P2-019):**
  `"{pool}:{session_id}:{scored_start}:{scored_end}:{anchor_suffix}"` where
  `anchor_suffix` is the sorted list of **typed anchor ids** joined by `'.'`:
  `p{gt_index}` for hard-positive anchors and `n{rank}` for negative anchors (negative
  anchors carry `rank`, not `gt_index`, in the Phase 1 selection — finding P2-019);
  empty for natural windows. The id never contains wall-clock or iteration-order data.
- **Non-circular hashing (frozen, finding P2-010, P2-019):** each episode's content
  hash is the SHA-256 of the canonical JSON of its own payload (no self-reference).
  The manifest content hash is the SHA-256 of the canonical JSON of the payload
  **excluding both the manifest's own `content_sha256` field and every per-episode
  `episode_manifest_id` field** (same scheme as `coverage_inventory.json`). The
  per-episode hash never includes the manifest hash, so no circularity is possible.
- Per-episode manifest identity recorded in each episode: `episode_manifest_id` =
  `episode_manifest_dev.json:<content_sha256>` is **filled after** the manifest content
  hash is computed and is stored as a non-hashed annotation field (finding P2-019), so
  the cache contract (Section 27.2) binds a deterministic manifest identity without
  circular hashing.

## 6. Gap interval-valued matching and overlap exclusion (Sections 12.1-12.2, 13.4-13.5)

The Phase 2 scoring module implements and fixture-tests the **complete deterministic
matcher** (finding P2-006), not just interval membership:

- **Eligibility (Section 12.1):** a final action matches a reference only if (1) source
  session and epoch agree; (2) action kind is compatible (`hard_boundary` actions match
  hard references, soft markers match soft references); (3) the boundary lies within the
  reference's stored acceptable interval — **the acceptable interval is the final
  tolerance-encoded eligibility window, applied exactly once (finding P2-020):** for
  clean targets the interval `[B onset - 500 ms, B onset]` already encodes the declared
  500 ms localization tolerance (Section 6.1, same semantics as the Phase 1 B0
  classification rule), and the matcher does **not** expand it a second time; for gap
  targets any boundary inside `[A speech offset, B onset]` matches with zero
  localization error, distance to the nearest interval edge outside it (Section 6.2,
  invariant 7); (4) detector-derived evidence was not available before
  detector-evidence onset (B onset; invariant 8); (5) availability meets the declared
  deadline (250/500/1000/1500/2000 ms); (6) ordered one-to-one matching is preserved
  (invariant 6). Edge fixtures test boundaries at exactly and just beyond each interval
  edge (finding P2-020).
- **Gap pre-existing VAD validity (invariant 9):** a VAD-owned action inside a gap
  acceptable interval is valid product separation even when available before B onset;
  it is reported `pre-existing` rather than rejected, and detector recovery credit
  still requires observation of B.
- **Matching objective (Section 12.2, frozen):** within each epoch, matching maximizes
  in order (1) number of compatible matched references; (2) number of B0-retained hard
  successes; (3) lower causal availability delay; (4) lower interval localization
  distance; (5) deterministic lexical ids. Contamination is never an input to matching
  (recomputed after matching). B0 actions are replayed independently; neural systems
  never reassign a B0 success but may receive acceleration credit when the same logical
  target becomes usable earlier.
- **Overlap exclusion (Sections 13.4-13.5, invariants 10-11):** the clean/gap headline
  stratum = `hard_only` episodes only; overlap soft references and `overlap_present`
  episodes never enter the primary contamination denominator; `overlap_hard_action`
  reported separately with the counterfactual
  `overlap_hard_action_contamination_contribution`.
- `contracts.py` `inside_acceptable_interval` / `gap_boundary_matches_inside_interval`
  remain the interval primitives; the full matcher is new and fixture-tested.

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

For each episode class (finding P2-008: classes are enumerated explicitly — for B0 the
class is `b0/peer` over all scorable episodes; class granularity may be refined per
corpus/tag only by a frozen amendment):

1. `source_prefix`: replay the original source wav from sample 0 through the target
   region (fresh deterministic re-run; Phase 1 traces are cross-checked but not reused
   as parity evidence).
2. `episode_reset`: reset all state at the extracted episode start, replay the declared
   warm-up, then score the same target region.

Compare at aligned source coordinates (finding P2-008 — all state-affecting outputs,
not just boundary pairs):

- B0: (a) raw boundary trace within the scored region — `(boundary_source_sample,
  observed_source_sample_at_emit)` identity, exact match on the canonical projection
  (tolerance 0); (b) **DetectorProgress safe-frontier trace (finding P2-017)** — the
  aligned `(observed_source_sample, safe_boundary_frontier_sample)` rows within the
  scored region are compared and hashed exactly between the two modes, with
  monotonicity of observed and safe frontiers and `safe_boundary_frontier_sample <=
  observed_source_sample` validated per row (Section 4.10, invariant 35);
  (c) boundary count and `trace_hash` over the scored region.
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
- **Reset is permitted only for an explicitly passing class (finding P2-008).** The
  Phase 2 gate requires: every family/profile class used for reset-based scoring passes
  the state-equivalence gate or is switched to source-prefix. Phase 2 executes the gate
  for B0 only; the disposition table records `b0/peer` pass/fail and LS/ERes
  `source_prefix_default` pending Phase 4.

### 8.5 Snapshot fallback path

- `mode=source_prefix`: deterministic full-source replay per episode
  (`replay_wav_epoch` from sample 0).
- Snapshot mechanism: serialized engine state captured at episode start from a
  full-source pass; round-trip test (capture -> restore -> resume) must reproduce the
  full-source trace exactly. Implemented and tested for the B0 engine in Phase 2;
  generic interface declared for LS/ERes (Phase 4).
- **Cache identity (frozen, finding P2-009)** binds the full PRD tuple (Section 27.2)
  before any snapshot/cache reuse:
  - checkpoint hash **and** checkpoint sidecar hashes;
  - frontend and resampler contract (pinned implementation/config hash);
  - source audio hash;
  - episode manifest hash (`episode_manifest_dev.json:<content_sha256>`);
  - model input/output tensor contract hash;
  - capture payload hash;
  - ERes additionally: every window coordinate and embedding payload hash.

## 9. Diagnostic/frontier/held-out group-disjointness (Sections 16.4, 17; invariants 27, 29, 31)

- The 20 opened scorable sessions form 20 distinct group components (verified in the
  Phase 1 group graph: no two scorable sessions share a component; annotation-only
  siblings in a component never produce episodes).
- Frozen pool split: per corpus (ami, alimeeting), sort scorable session ids; assign
  alternately by rank (even -> `diagnostic_dev`, odd -> `frontier_dev`). The split is
  deterministic, corpus-stratified, and asserts that no component is split across pools
  (invariant 27). Synthetic manifests (`ls_dev`, `ls_held_out_clean`, `ls_held_out_other`,
  `mixed_dev_pool`) are assigned wholly to `diagnostic_dev` (groups `synthetic:<name>`).
- **Historical label carried through (findings P2-012, P2-023):** each session and
  episode records its historical status from a frozen normalized set:
  `dev_pilot` (previously touched development), `held_out_pilot` (previously touched
  held-out — historical validation only), `dev_added` (newly materialized development
  additions, never previously touched — normalized as fresh development evidence and
  eligible for panel/development use), `untouched` (reserved, unopened). Historical
  (previously touched) evidence is development-usable but is excluded from every
  confirmatory/panel claim path: the manifest header and later Phase 6 panel
  construction must assert that no confirmatory selection uses a session whose
  historical status is not `untouched` plus the Phase 6-7 approved held-out set, and
  that `dev_added`/`dev_pilot`/`held_out_pilot` sessions never enter confirmatory
  claims.
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
- **Natural-window subrange mapping (frozen, finding P2-022):** a natural window is a
  30 s unit whose full window equals its scored region — `W = S = [start_ms, start_ms
  + 30 s]` (clipped at session end), warm-up and tail are recorded as
  `warmup_truncated=true` / `tail_truncated=true` (zero-length). The Section 5.1
  10-20 s scored rule and the target-enriched warm-up/tail defaults apply to
  target-enriched episodes only; natural windows are scored over their full extent
  with the zero warm-up/tail explicitly recorded, never silently assumed. Natural
  windows therefore do not claim the 5 s warm-up / 3 s tail construction; their scored
  denominator is the full window (sampled vs eligible recorded).
- Natural window episode ids follow the frozen format (Section 5.3) with
  `anchor_suffix` empty and `pool = natural_exposure_validation`.

## 11. Sampled waveform/annotation audit procedure (Phase 2 gate)

- Deterministic sample: per pool, keep episode iff
  `int(sha256(episode_id).hexdigest()[:2], 16) < 8` (1/32), with a floor of 8 episodes
  per pool (smallest-hash fill) — frozen before audit, using the frozen episode-id
  format (Section 5.3, finding P2-010).
- Waveform audit: re-open the source wav, extract `[episode_start, episode_end)`
  samples, and compare **byte-for-byte** against the episode slice the builder
  recorded; also verify wav header (PCM 16-bit mono 16 kHz).
- Annotation audit (public): independently re-derive the reference timeline for the
  episode span directly from the source annotations (AMI `words.xml` set; AliMeeting
  TextGrid interval tiers), **independent of the builder's cached regions and of the
  Phase 1 classifier**, using the same frozen taxonomy (Section 5.1), and require exact
  equality of the reference set (ids, kinds, target samples, acceptable intervals,
  scorable flags). This is the independent re-derivation authority for the taxonomy
  (finding P2-005).
- Annotation audit (synthetic, finding P2-011): independently re-derive the reference
  timeline for each sampled synthetic episode from the Phase 1 synthetic manifest case
  regions (the manifest is the independent annotation authority) and require exact
  equality with the registered episode references.
- Any audit failure fails the phase (stop condition); the audit report records
  per-episode pass/fail with mismatch details.

## 12. Contamination/harm scoring fixtures (Sections 13-14)

- Pure-function scorer over (final hard boundaries, reference timeline, episode):
  logical segmentation (13.1), turn ownership (100 ms primary, 50/200 ms views),
  contamination algorithm (13.3), clean/gap headline ratios and mixed-turn tiers
  (13.4), full deterministic matcher (Section 6), harm flags (14.1-14.5),
  fragmentation metrics (14.6) — implemented in `turn_episode/scoring.py` wrapping
  `contracts.py`.
- Known-answer fixtures: hand-built tiny timelines covering invariants 6-20 (gap
  interval match, tolerance-expanded eligibility, deadlines, B-onset evidence gating,
  pre-existing VAD gap validity, B0-retention priority, overlap exclusion, warm-up
  exclusion, unscored exclusion, no double count, premature-split no-credit,
  turn-owner thresholds, harm orthogonality, lexical-split missing-timing semantics,
  same-speaker pause split counting, duplicate, overlap hard action).
- B0 end-to-end smoke: run the scorer on the B0 full-session traces over the 20
  sessions (B0 hard boundaries = raw VAD boundaries), producing baseline
  contamination/harm rows. These are **fixture verification / baseline dev evidence**
  labeled `baseline`, never confirmatory claims and never natural rates (Section 13.6).

## 13. Deterministic manifests and code layout (Section 27.3)

- `episode_manifest_dev.json`: all diagnostic_dev + frontier_dev episodes (public +
  synthetic registrations), self-hashed (non-circular scheme, Section 5.3); per-episode:
  id, pool, status (`scorable`/`diagnostic_only`), session, epoch, sample bounds
  (warm-up/scored/full), tag, references, selection digest, historical label, hashes.
- `natural_exposure_manifest.json`: natural windows (Section 10), self-hashed.
- `state_equivalence_report.json`: per-class disposition table, parity results,
  tolerance constants, safe-frontier evidence, snapshot round-trip evidence (Section 8).
- `scoring_fixture_report.json`: known-answer results + B0 baseline smoke (Section 12).
- `audit_report.json`: sampled audit results (Section 11).
- `episode_manifest_heldout.json`: **deferred** to Phase 6/7 (recorded deferral entry in
  the dev manifest header).
- All JSON artifacts carry canonical content hashes; row files record direct byte
  SHA-256 (Section 27.3). Provenance headers record the Section 1.1 hash ledger.
- New code under `turn_episode/`: `build_episodes.py` (builder + ReferenceBuilder +
  manifests), `state_equivalence.py`, `scoring.py`, `audit.py` (plus a new
  `EpisodeStatus` literal and any frozen tolerance/pool-split constants in
  `contracts.py`/`schemas.py`).

## 14. Falsification and stop conditions for Phase 2

- Any cross-pool source-sample overlap, any session assigned to two pools, any
  component split across pools, or any change to the Phase 1 group graph hash → phase
  stops, fails closed.
- Non-overlap assertion among scored episodes within a pool fails (including the
  re-bucketing disjointness assertion, finding P2-015) → phase stops.
- Any registered synthetic case whose wav path resolves under `ami/` or `alimeeting/`
  (real-recording exclusion, finding P2-014) → phase stops.
- Any `scorable` episode with total duration > 30 s, or scored region < 10 s or > 20 s,
  or a `diagnostic_only` episode that still contributes to a scored numerator → phase
  stops (findings P2-001, P2-004).
- Reference timeline rebuild disagrees with the Phase 1 classifier output on any shared
  target, or the independent audit re-derivation disagrees on any sampled episode →
  phase stops (determinism defect).
- Audit sample mismatch (waveform bytes or annotation re-derivation) → phase stops.
- B0 state-equivalence failure for the `b0/peer` class → that class is recorded
  `source_prefix_required`; the fixture report must show the disposition table complete
  with no untested class used for reset-based scoring (finding P2-008). Safe-frontier
  trace parity (finding P2-017) is part of the failure condition.
- B0 readiness predicate cannot be proven from replay-derived safe-frontier evidence
  (finding P2-016) for an episode → that episode is `diagnostic_only`; the readiness
  report must account for every scorable episode.
- Natural-exposure manifest selection differs from the Phase 1 frame for the opened
  sessions → phase stops (frame was frozen before label inspection; a difference means
  the frame was regenerated).
- Reserved sessions: any attempt to parse/read their annotations or audio in the
  builder/manifest code fails closed with an explicit error (invariant 31).
- Historical label asserts (Section 9) fail → phase stops.

## 15. Expected compute/data/provider cost and irreversible access

- Compute: episode building/classification CPU minutes; B0 parity replays
  (source-prefix full-session + episode-reset over ~189 episodes, ~13-18 s audio each)
  CPU well under an hour; scoring/audit CPU minutes. No GPU, no provider, no
  credentials, no downloads.
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
| P2-001 | blocker | no-split fallback kept merged episodes > 30 s | resolved in Section 4.3 (scorable episodes never exceed 30 s; fail-closed re-bucketing; `diagnostic_only` + `coverage_loss` recording) |
| P2-002 | blocker | no deterministic scoring-start readiness predicate | resolved in Section 4.4 (frozen per-class `scoring_start_ready` predicate; B0 warm-up >= 5 s + hangover-margin rule; LS/ERes declared at Phase 4; unstable frontier -> diagnostic_only) |
| P2-003 | important | `diagnostic_only` not representable in `EpisodePoolTag` | resolved in Sections 4.4, 13 (new `EpisodeStatus` field separate from pool tag; `scorable=false`; excluded from scored pools) |
| P2-004 | blocker | scored-region truncation unrecorded; merged scored unions could exceed 20 s | resolved in Section 4.3 (10-20 s scored rule; end-trim cap at 20 s with `coverage_loss`; < 10 s -> `diagnostic_only` `scored_truncated`) |
| P2-005 | blocker | Phase 1 classifier taxonomy incomplete (no departures/structural/unscored; missing timing silently skipped) | resolved in Section 5 (new ReferenceBuilder with complete taxonomy; explicit unscored intervals; independent re-derivation) |
| P2-006 | blocker | matcher spec incomplete (no tolerance expansion, deadlines, compatibility, matching objective) | resolved in Section 6 (full deterministic matcher per 12.1-12.2 with fixture tests) |
| P2-007 | important | stability/ambiguity/rounding predicates unspecified | resolved in Sections 4.2, 5.1 (stable overlap/split predicates, negative midpoint floor division) |
| P2-008 | blocker | B0 parity compared only boundary pairs; no DetectorProgress/safe-frontier; no class enumeration; Phase 1 evidence reused | resolved in Section 8 (full state-affecting comparison incl. safe-frontier; fresh re-runs in both modes; class enumeration; reset only for passing classes) |
| P2-009 | important | snapshot/cache identity tuple incomplete | resolved in Section 8.5 (full PRD Section 27.2 tuple incl. sidecars, tensor contract, ERes coordinates/payload) |
| P2-010 | important | episode-id format and hash circularity unspecified | resolved in Section 5.3 (frozen id format; non-circular payload/header hashing) |
| P2-011 | important | synthetic episodes lacked an audit authority | resolved in Sections 5.2, 11 (synthetic manifests as independent authority; synthetic references audited) |
| P2-012 | note | historical label not carried through splits | resolved in Section 9 (historical status per session/episode; excluded from confirmatory/panel claims) |
| P2-013 | note | code/config/input hash ledger missing | resolved in Section 1.1 (full SHA-256 ledger) |
| P2-014 | blocker | `mixed_dev_pool` contains complete real AMI recordings (ES2003a, IS1008a) but was assigned to `diagnostic_dev` as synthetic, creating cross-pool source overlap | resolved in Section 5.2 (real-recording exclusion rule: cases with wav under `ami/`/`alimeeting/` are not registered as synthetic; recorded as `excluded_real_recording`; fail-closed no-overlap assert) |
| P2-015 | blocker | first-fit re-bucketing could create overlapping buckets | resolved in Section 4.3 (tail-removal re-bucketing that never emits removed anchors; pairwise-disjoint by construction plus fail-closed assertion; unresolvable groups -> diagnostic_only) |
| P2-016 | blocker | B0 readiness used annotation boundaries, not detector pending state | resolved in Section 4.4 (replay-derived DetectorProgress safe-frontier <= scored_start; unprovable -> diagnostic_only `unstable_warmup_frontier`) |
| P2-017 | blocker | B0 parity did not compare `safe_boundary_frontier_sample` values | resolved in Section 8.2 (aligned (observed, safe-frontier) trace compared and hashed exactly; monotonic/conservative validation per row, invariant 35) |
| P2-018 | blocker | region-only builder cannot recover words lacking timing (AMI parser discards them) | resolved in Section 5 (builder consumes raw annotation records plus regions; missing-timing spans become explicit unscored intervals; audit re-derives from raw files) |
| P2-019 | blocker | negative anchors lack gt_index (colliding/undefined episode ids); manifest hash circularity via episode_manifest_id | resolved in Section 5.3 (typed anchor ids `p{gt_index}`/`n{rank}`; manifest hash excludes content_sha256 AND per-episode episode_manifest_id; id filled post-hash as non-hashed annotation) |
| P2-020 | important | tolerance applied twice (interval + expansion) | resolved in Section 6 (acceptable interval is the final tolerance-encoded window, applied once; edge fixtures at/just beyond boundaries) |
| P2-021 | important | deferred structural taxonomy contradicted "complete" claim | resolved in Sections 5.1 (VAD max-duration/terminal structural excluded from Phase 2, deferral recorded, no completeness claim) |
| P2-022 | important | natural windows lacked warmup/scored/tail mapping | resolved in Section 10 (natural window: full window = scored region; zero warm-up/tail recorded as truncated; 10-20 s rule is target-enriched-only) |
| P2-023 | important | historical status contract omitted `dev_added` | resolved in Section 9 (frozen normalized status set {dev_pilot, held_out_pilot, dev_added, untouched}; dev_added = fresh development evidence; confirmatory = untouched only) |
| P2-024 | important | warm-up/scored boundary crossing intervals undefined; warm-up actions could match | resolved in Section 5.2 (half-open scored clipping; clipped intervals; targets in warm-up not scored; all warm-up actions rejected before matching, invariant 12) |
| P2-025 | note | provenance ledger omitted corpus parsers, synthetic manifests, contracts | resolved in Section 1.1 (full ledger: corpus parsers, all synthetic manifests, proposal/fusion contracts) |
| P2-026 | note | implementations are prospective, not execution evidence | acknowledged (Section 18): implementation artifacts are produced only after approval and independently verified at the Phase 2 exit gate |

## 18. Execution and exit-gate note (finding P2-026)

- The modules named in this bundle (`build_episodes.py`, `state_equivalence.py`,
  `scoring.py`, `audit.py`) and the scored manifests do not exist yet. This bundle is a
  pre-execution design review: the Phase 2 reviewer examines the **frozen rules** above
  before any scored episode/reference generation (PRD Section 29 Phase 2 timing).
- Implementation proceeds only after the review verdict is `approved`; the Phase 2 exit
  gate then independently verifies the actual artifacts (manifests, state-equivalence
  report, scoring fixture report, audit report) against this bundle and the PRD.
| P2-026 | note | implementations are prospective, not execution evidence | acknowledged (Section 18): implementation artifacts are produced only after approval and independently verified at the Phase 2 exit gate |
