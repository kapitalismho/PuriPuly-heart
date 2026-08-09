# Phase 3 pre-execution review bundle — provider-neutral logical-action oracle

Status: **pending independent pre-execution review**. This bundle freezes the Phase 3
lifecycle experiment before the oracle delay/offset/holdback grid is executed. No Phase 3
oracle result has been generated and no neural detector, confirmatory held-out source,
provider credential, or live provider is used by this phase.

Revision: 3. Rev 2 resolved P3R-001 through P3R-006 except residual findings
P3R-004-R2-A/B, P3R-005-R2, and P3R-003-R2; rev 3 resolves those residuals.
Candidate: `working-tree` based on `d3a054261c14a6caa52b0a1aafe9c2ac87289621`.

## 1. Authority and accepted entry gate

| Item | Value |
| --- | --- |
| Normative plan | `.agents/specs/prd/bounded_turn_episode_speaker_change_fusion_experiment_review_gated.md` |
| Plan Git blob | `24340f488f1bb46c666a5fc15eef2fc87ef1f826` |
| Plan SHA-256 | `8c6bed2e564b9ec80e26ee6b73701985c863b7beb68e51590be7a5faf173aad4` |
| Accepted Phase 2 candidate | `d3a054261c14a6caa52b0a1aafe9c2ac87289621` |
| Integration target | `origin/main` at `848aa0b9f1b35388ded5a250d51a687223eac1c5` |
| Work branch | `experiment-v2-speaker-change-turn-boundaries-ls` |
| Phase 2 review | `reviews/phase_2_pre_execution.md` and accepted Phase 2 exit review recorded at `d3a05426` |

Phase 2 supplies 804 deterministic development episodes, of which 792 are scorable.
The lifecycle contamination workload is the accepted 186 scorable public episodes from
20 already-opened development sessions. The remaining historical synthetic episodes are
not treated as confirmatory evidence; deterministic PCM lifecycle fixtures provide the
synthetic mechanics coverage in this phase.

### 1.1 Frozen input ledger

| Input | Byte SHA-256 | Bound semantic identity |
| --- | --- | --- |
| `episode_manifest_dev.json` | `a5df5e56c2917ba174fca7892fbaa092f2b57705cad7a77bf9a347ba94cddfee` | content SHA-256 `deb1713cd581c93cc13c47103643041c7551993985379869cfa0ca9a407dff68`; 804/804 episode hashes verified at Phase 2 exit |
| `state_equivalence_report.json` | `6e33711632d5f2e3de8e0c22c229b08827d1ccbb873deba2c1681a2ab2c544ec` | content SHA-256 `c5e4836f69686587bad0b24a2293e0f80b336ba5105651889006aee4a7db3c2c`; B0 requires source-prefix state |
| `scoring_fixture_report.json` | `36a9648178f3de1b9924b1a4ef71baddf28eccd39fa7218d34b447f993a145b1` | content SHA-256 `331a8d54394dbc30ad59d208f470ebafc3c5c2abcb9016b4550b8b52522b625a`; 25/25 fixtures pass |
| `audit_report.json` | `901020e864ada40a7918354f8039bad85512dace8d033a1bcba16d3428db36e4` | content SHA-256 `6b9963e8849af2ad13dd5632b813d7bd05b0e65462a739b1f237fa1ebd310475`; zero audit failures |
| `proposal_contract.json` | `0448edd933fd1d9d0a0b4d5f9f2631cb0f630c892fc4d46e1a3ec9740e80b7fb` | Phase 0 proposal contract |
| `fusion_contract.json` | `bfda0c3c0ea7b6613ded79e9639692a33449dcf34202b1f2a5e7ec14c45f9873` | Phase 0 action contract |
| `turn_episode/schemas.py` | `a9fa4571b1bab3cf88d6739a3732c1cc62f753a46d51d59c2db7526468eb8868` | `FinalAction`, `DetectorProgress`, epoch fields |
| `turn_episode/contracts.py` | `b207d3f8b9720df5dd228aa8bd8b479c54622abb905a9ca04f580820a6fc3c03` | scientific invariants 1-37 |
| `turn_episode/scoring.py` | `332a7daf70e684cd5b9918f808b76e8e0c39f6db559008d3f48c38590fb0aa90` | accepted contamination/ownership algorithm |
| `turn_episode/build_episodes.py` | `6deec51274cedf49a70cd299700547f39cbbbc16e200eb8e3056d15887784c7d` | accepted source-session and reference reconstruction |
| `turn_episode/pinned_ledger.py` | `7509c7abea6813051150f1ff2d98e6f61630c5e10a1801a2905326d9f1290aaa` | fail-closed provenance verification |
| `vad_baseline.py` | `7a3965fdb01eb7391dde985e5c498162d80b4e5ab565205626d684a66d8ff627` | accepted B0 hard-boundary projection algorithm |
| `src/puripuly_heart/core/vad/gating.py` | `88d5dec630b8352fd192f1ef5be7aea39b19bdc7d43273810d260400e3217fec` | production peer VAD lifecycle baseline, read-only |
| `src/puripuly_heart/core/vad/silero.py` | `43079df5bc36ecb924b1aec7991cff2a16c04ab126bb54907c4b2a570e2cd109` | production VAD engine adapter baseline, read-only |
| Silero ONNX model | `1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3` | accepted B0 model identity |
| `experiments/speaker_turn_boundary/config.py` | `f4eb24e6c81ebcb0bdd71b6c0c9098595ae4bdddf53e05df6bd8eea925d146a6` | B0 chunk/profile constants |

The 20 `b0_inventory_replay/*.json` traces are resolved through the accepted pinned
ledger and verified before use. Phase 3 slices those source-prefix hard-boundary
projections; it never substitutes reset-plus-warm-up B0 state. Those projections are
not treated as lifecycle traces: they omit the original `SpeechEnd`, max-duration
without a successor, and terminal events. Section 7 freezes the separate lifecycle
producer and its exact parity relationship to the accepted projection.

## 2. Scope and non-goals

Phase 3 implements one experiment-only canonical PCM turn assembler and executes the
provider-neutral oracle prescribed by PRD Sections 23 and 29. It validates ownership,
loss/duplication, late action, safe drain, structural finalization, and contamination
ceilings independently of any neural detector.

Explicit non-goals:

- no LS-EEND or ERes2NetV2 inference or cache generation;
- no proposal clustering, refractory search, VAD-fusion policy sweep, or detector
  selection;
- no confirmatory held-out path resolution, metadata inspection, audio access, or
  aggregate inspection;
- no provider SDK, network, credential, paid/live call, transcript, or provider
  comparison;
- no production owner, composition, settings, provider adapter, runtime lifecycle, or
  public-entrypoint change;
- no product recommendation.

The assembler is implemented only under
`experiments/speaker_turn_boundary/turn_episode/pcm_oracle.py`. It is an experiment
double/replay harness, not production wiring.

## 3. Canonical coordinate and ownership model

All intervals are half-open canonical 16 kHz **absolute source-session** sample spans
`[start, end)`. Coordinates are never rebased. Each isolated episode creates a fresh
epoch with `epoch_origin_source_sample = scored_start`, and initializes observed and
released frontiers to that nonzero origin. The conserved/ingested interval is exactly
`[scored_start, processed_scored_end)`, where `processed_scored_end` is recomputed as
`min(bounds.scored_end, floor_to_512(min(session.duration_samples,
session.wav_length_samples or session.duration_samples)))` under the accepted Phase 2
loader. Warm-up is not re-ingested because Phase 3 has no neural state and consumes the
already accepted source-prefix B0 traces. An ingested PCM
chunk must start at the observed frontier. A chunk is retained as immutable views over
the original PCM payload plus its absolute source span; splitting a view never copies,
reorders, drops, or invents source samples. Synthetic fixtures may store local payload
indices, but every index maps through the explicit `epoch_origin_source_sample` before
it enters an action, progress row, lifecycle event, B0 trace, or evidence row.

For an accepted logical action at boundary `b`:

- every still-recoverable sample `< b` is assigned to the old turn;
- every still-recoverable sample `>= b` is assigned to the new turn;
- already released samples are immutable;
- when `b` is older than the released frontier, the action is labeled `late_unrecoverable`,
  the new turn begins at the released frontier, and `[b, released_frontier)` is recorded
  as unrecoverable ownership error rather than silently reassigned;
- a boundary at the released frontier is recoverable; a boundary before it is not;
- same-boundary duplicate actions are idempotently suppressed and cannot create an
  empty extra turn;
- actions from another epoch are rejected before any state mutation;
- a new epoch is legal only after the prior epoch has terminally released and finalized
  every retained sample exactly once; it then resets frontiers to the new explicit epoch
  origin, clears ring content and pending drains, and starts new turn IDs;
- forced abandonment with retained PCM is a phase-stopping conservation failure, never
  a successful reset path.

The assembler records both ideal oracle ownership and realized ownership. Conservation
is evaluated over realized source spans. Contamination is evaluated over realized
logical turns; ideal ownership is never substituted for a late result.

## 4. Deterministic event ordering

Input PCM is streamed in 512-sample chunks with one final partial chunk when necessary.
No coordinate is rounded for holdback release or action splitting. At observed frontier
`o`, the ordinary release limit is exactly
`max(released_frontier, max(epoch_origin_source_sample, o - holdback_samples))`, capped
at `o`. A limit inside a retained chunk splits its immutable view at that exact sample.
For holdback zero the limit is `o`; for a terminal partial chunk the limit may be the
unaligned processed end.

At each observed frontier, events are processed in this fixed order:

1. append the contiguous PCM chunk;
2. validate and apply all `DetectorProgress` rows at that frontier;
3. apply actions now available, ordered by
   `(availability_source_sample, boundary_source_sample, action_id)`;
4. apply lifecycle `SpeechEnd` or structural maximum-duration/terminal events at that
   frontier;
5. resolve an eligible safe drain or its timeout; while a drain is pending, ordinary
   release is frozen at the drain's captured released frontier;
6. if no drain remains, release through the exact ordinary release limit;
7. record the resulting progress, ring, ownership, and finalization state.

An action with an availability between chunk frontiers is applied at the first observed
frontier that covers it. The requested source availability and the chunk-realized apply
frontier are both recorded. This scheduling quantization is not folded into boundary
location error.

## 5. Oracle actions and full grid

The population is selected by exact membership in the accepted Phase 2
`baseline_smoke.rows`, joined fail-closed to `episode_manifest_dev.json`, then filtered
to `status=scorable`. The canonical identity object has exactly the keys `session_ids`,
`episode_ids`, and `reference_ids`; each array is independently sorted lexically.
Canonicalization is UTF-8 JSON with `sort_keys=True`, separators `(',', ':')`, and
`ensure_ascii=False`. Its SHA-256 is
`cb06483fb82618bf06dbcbe75a946c65bdea9f67109ddfb645a7e06f4dd555bf`.
It binds 20 sessions, 186 episodes, 283 hard references, 142 episodes with a hard
reference, and 44 no-hard-reference conservation controls.

For every scorable hard reference in this population, one oracle action is derived for
every grid point:

```text
availability delay: 250, 500, 750, 1000, 1250, 1500, 2000 ms
boundary offset:    -500, -300, -200, -100, 0, +100, +200, +300, +500 ms
holdback:           0, 250, 500, 750, 1000, 1500, 2000 ms
```

This is exactly 441 grid rows, 82,026 episode/grid detail rows, and 124,803 oracle-action
instances. Each of the seven delay shards has 11,718 detail rows and 17,829 action
instances. The `-500 ms` and `+500 ms` sentinels are mandatory and cannot be pruned.

For offset `k_ms`, the unclamped coordinate is exactly
`reference.target_sample + 16 * k_ms`. It is clamped to the closed action-coordinate
domain `[scored_start, processed_scored_end]`; the unclamped coordinate, clamp direction,
clamped coordinate, and realized signed point offset are all recorded. A boundary at the
processed end finalizes the old turn without creating or counting an empty successor.
Pre-execution recomputation finds exactly 24 clamped reference/offset combinations across
16 references in 15 episodes: 13 below the scored start and 11 above the processed end.
Across seven delays and seven holdbacks these produce exactly 1,176 clamped action
instances. The canonical clamp identity object has schema version
`turn_episode_v1.phase3_clamp_identity` and a `clamped_reference_offsets` array sorted by
episode ID, reference ID, and numeric offset. Each row contains episode ID, reference ID,
offset ms, unclamped boundary, `below_start|above_end` direction, and clamped boundary.
Under the same canonical JSON rules its SHA-256 is
`22b4488a8a93ee1e6b8de03cdfa914613e213f5198ba603605551f9c3404e14c`.
Any count or identity drift fails before aggregation. Requested availability is
`evidence_onset_sample + delay`;
causal availability is
`max(requested_availability, boundary_source_sample)`. Both requested and realized
delays are retained so positive offsets cannot create an impossible pre-boundary event.

All oracle actions for an episode are applied together. Action application uses the
event ordering in Section 4; equal availability ties use boundary and action ID. The
accepted source-prefix B0 hard-boundary projection remains ordinary B0 `retain_vad`
actions, never structural taxonomy. The candidate adds oracle hard actions. Duplicate
B0/oracle cuts are normalized to one logical boundary. The comparison therefore measures
B0 versus B0 plus exact controlled speaker actions, not an oracle-only segmentation with
the product baseline removed.

After the independently accepted Phase 5 exit gate, localization is recomputed separately
for every exact `policy_run_id`, which binds family, detector profile, proposal schema,
cluster/fusion profile, input-cache hash, and code/config hash. Matching never crosses
episode ID, session ID, or epoch. Eligible actions belong to that exact policy run, have
non-null detector `cluster_id`, and have kind `add_hard_boundary` or
`accelerate_or_replace_vad`; `retain_vad`, structural, soft, B0/B1, control-owned, and
provider actions are ineligible.

Within each episode, scorable different-speaker hard references and eligible actions are
matched by an ordered one-to-one dynamic program with no action reuse and no 500 ms
localization cutoff. It maximizes match count, then minimizes total distance to acceptable
intervals, total causal availability, and lexical reference/action IDs. Unmatched
references remain `missing`; they are counted and never replaced by an action from
another episode. Per-policy p95 uses the nearest-rank rule in Section 10 over matched
interval distances; missing count is reported alongside it and is not converted to zero.
The trigger value is the maximum defined per-policy p95 across all complete real-detector
policies; controls, B0, and B1 are reported but do not set it.

At that boundary all work stops and the complete per-policy performance and p95 table is
reported to the user. Phase 6 preparation, reviewer dispatch, frontier construction, and
freeze remain blocked until explicit user resume authorization. If the trigger exceeds
500 ms, Phase 3 must first reopen and extend offsets symmetrically to the smallest 100 ms
multiple covering the trigger, regenerate all hashes/evidence, and receive an independent
accepted Phase 3 exit review. Any Phase 3 reopening blocks every downstream phase while
open; accepted upstream evidence is retained but cannot authorize further execution.

## 6. Safe frontier, SpeechEnd drain, and fallback

The oracle progress trace is conservative by construction. Its initial safe frontier is
`max(0, epoch_origin_source_sample - 1)`. At each step 2 in Section 4, before actions at
that observed frontier are applied, it is exactly the minimum of the observed frontier
and `b - 1` for every not-yet-applied oracle action boundary `b`; with no pending action
it equals the observed frontier. The value must not decrease. If the formula would fall
below the prior safe frontier, or if any later action refers at or before a prior safe
frontier, the case fails instead of repairing the trace. Thus a boundary remains unsafe
through its apply step and the safe frontier may advance beyond it only on the next
progress step. Every row must be monotonic within an epoch and must not exceed the
observed frontier.

At Qwen-style `SpeechEnd` sample `s`, the assembler arms a drain through `s` and does
not release the held region until `safe_boundary_frontier_sample >= s`. Detector
progress continues after SpeechEnd without a VAD, detector, or epoch reset. The frozen
safe-drain timeout is 2000 ms on an injected monotonic scheduler clock beginning when
the drain is armed; source progress does not substitute for this clock. The deadline is
checked on every chunk, progress, action, lifecycle, structural, explicit timer,
end-of-input, and reset request. If the safe frontier still does not cover `s` at the
deadline, the harness releases through `s` via a separately labeled
`safe_drain_timeout_fallback`; ordinary completion and fallback are never pooled.

Distinct SpeechEnd drains form a FIFO queue ordered by event/apply frontier and drain ID;
they never coalesce. Each drain records its own target `s`, released frontier captured at
arm time, arm clock, and deadline `arm_clock + 2000 ms`. An exact duplicate event ID is
idempotently ignored; a new event with a target lower than the preceding queued target is
invalid. While the queue is nonempty, ordinary release remains frozen at the head drain's
captured frontier. The head completes safely only when the safe frontier covers its own
target, then releases through that target and finalizes that turn; it falls back only at
its own deadline. After removing the head, the next drain is evaluated immediately under
its own target and deadline. Later deadlines never postpone an earlier drain.

Completion class and latency are attributed per drain ID: `safe_complete` or
`safe_drain_timeout_fallback`, scheduler latency from that drain's arm clock, and source
release latency from its target to the observed frontier at resolution. Terminal and
reset resolve the FIFO head-to-tail, advancing the injected clock to each unresolved
head's own deadline without PCM as needed; every drain retains its individual label and
latency.

At end-of-input or reset request with a pending drain, the deterministic replay advances
the injected clock to the existing deadline without ingesting PCM, records ordinary safe
completion if progress already covers `s` or timeout fallback otherwise, releases and
finalizes all retained PCM exactly once, then emits terminal completion. A reset cannot
skip this sequence. Failure to resolve, release, or finalize is a phase-stopping error.

The fixture matrix includes safe coverage at 0, 250, 1000, and 2000 ms; a stalled
frontier resolved by timer with no further PCM; an invalid frontier advance followed by
a late event; SpeechEnd immediately adjacent to an oracle boundary; a three-drain FIFO
with distinct safe/fallback outcomes and deadlines; terminal with a pending queue; reset
with held PCM; duplicate event ID suppression; regressing-target rejection; and forced
abandonment as an expected failing fixture. Release before safe coverage is a
phase-stopping error.

## 7. Structural lifecycle coverage and B1 seed

Phase 3 implements `B0LifecycleReplay` inside `turn_episode/pcm_oracle.py`. It invokes
the same production `VadGating` and Silero engine/config used by `VadBoundaryReplay`,
over each full opened source session in source-prefix mode. It retains every raw VAD
lifecycle event before projection in this schema:

```text
B0LifecycleEvent {
  event_id, audio_epoch, source_session_id, normalized_utterance_id,
  event_kind: speech_start | speech_end | terminal,
  reason: start | silence | max_duration | end_of_input,
  event_source_sample, observed_source_sample_at_emit,
  trailing_silence_ms, chunk_index, chunk_samples
}
```

Runtime UUID values are normalized by first-occurrence ordinal while preserving equality
relationships. `speech_start.event_source_sample` is the processing chunk start.
`speech_end.event_source_sample` is
`(chunk_index + 1 - round(trailing_silence_ms / 32 ms)) * 512`, lower-bounded by the
utterance start; max-duration has zero trailing silence and therefore lands at that
chunk's end. `observed_source_sample_at_emit` is `(chunk_index + 1) * 512`. A terminal
event is emitted at the exact full-source processed end even when no successor
SpeechStart exists and records whether active/pending VAD state remained.

The full-source terminal event validates lifecycle completeness. Each isolated oracle
episode additionally emits an experiment-harness terminal action at its own
`processed_scored_end`; that terminal resolves and conserves the episode-local assembler
interval and is never represented as an accepted B0 hard action.

As a binding parity check, the lifecycle replay independently derives ordinary B0 hard
boundaries on successor SpeechStart with the exact accepted `VadBoundaryReplay`
projection. For all 20 sessions its canonical projection and trace hash must equal the
accepted `b0_inventory_replay/*.json` byte-for-byte in semantic content. This proves the
new lifecycle events came from the accepted production-shaped B0 path. Any source,
config, model, projection, or count mismatch stops the phase.

Maximum-duration and terminal finalizations, deferred by Phase 2, become trace-visible
structural actions here:

- maximum duration closes the current logical turn at its exact source sample, preserves
  the epoch and assembler state, and opens the next turn without speaker-change credit;
- terminal flush releases all remaining PCM exactly once and closes the epoch with a
  separately labeled terminal reason;
- ordinary silence `SpeechEnd` remains a lifecycle finalization event; ordinary accepted
  B0 successor-SpeechStart boundaries remain `retain_vad` hard actions. Only actual
  maximum-duration and terminal actions use the Phase 3 structural labels;
- a no-detector B1 seed fixture routes the concrete lifecycle event trace and accepted B0
  hard actions through the same assembler and requires identical ordinary B0 boundary
  coordinates plus identical lifecycle coordinates/reasons and logical segmentation
  after duplicate normalization.

This seed closes the lifecycle mechanics deferral. Full B0/B1 equality under the final
proposal/cluster/fusion engine remains a Phase 5 gate and is not claimed in Phase 3.

## 8. Measurements and exact calculations

For every episode/grid case:

- conservation passes only if the sorted realized ownership spans form exactly one
  gap-free cover of `[scored_start, processed_scored_end)`;
- no-duplication passes only if realized ownership spans are pairwise disjoint and their
  summed length equals the source interval length;
- old/new ownership is compared at each oracle boundary using ideal versus realized
  turn IDs;
- unrecoverable audio is the length of
  `[boundary_source_sample, released_frontier_at_apply)` intersected with the processed
  episode, or zero when the boundary is still in the ring;
- fragment durations are the lengths of every non-empty realized logical turn;
- logical-action finalization latency is chunk-realized apply frontier minus boundary;
- safe-drain latency is the release frontier that completes the drain minus SpeechEnd;
- contamination at 50, 100, and 200 ms owner thresholds is recomputed with the accepted
  Phase 2 singleton-region algorithm for both B0 and B0 plus oracle turns;
- the primary oracle ceiling is the hard-only 100 ms-owner contamination ratio and
  paired difference from B0; overlap-present results remain separate;
- reductions, unchanged cases, and regressions are counted per episode and per source
  session; target-enriched results are never labeled as natural rates.

The unrecoverable-late curve is grouped by requested delay, offset, and holdback and
reports total/mean/p50/p95/max unrecoverable milliseconds, fully recoverable action
fraction, and contamination remaining. A grid row cannot be called successful when its
aggregate contamination improves but any conservation or duplication invariant fails.

## 9. PCM and lifecycle fixtures

Deterministic signed-16-bit mono PCM payloads encode unique sample identities. Fixtures
compare the byte concatenation of all realized turns with the original PCM and also
verify the span ledger. Cases cover boundary-at-zero/end, chunk interior/edge, multiple
boundaries, duplicate boundaries, zero holdback, fully protected boundary, boundary just
inside/outside the ring, positive/negative offsets, action after ring eviction,
same-epoch late ordering, stale epoch, epoch reset, maximum duration, terminal partial
chunk, safe drain success, and safe-drain timeout fallback.

Property checks enumerate short streams of 0-8 chunks, 0-3 boundaries, every grid
holdback, and boundary positions at `chunk_edge + {-1, 0, +1}`. Each case asserts exact
conservation, duplication, turn ordering, and deterministic replay.

## 10. Outputs and size control

The implementation produces:

- `turn_episode/pcm_oracle.py`;
- `turn_episode/verify_pcm_oracle.py`;
- `tests/test_phase3_pcm_oracle.py`;
- `results/turn_episode_v1/oracle_provider_neutral.json`;
- `results/turn_episode_v1/oracle_provider_neutral_verification.json`;
- seven deterministic gzip JSONL detail shards under
  `results/turn_episode_v1/oracle_provider_neutral_details/`, one per availability delay.

The main JSON contains the complete 441-row aggregate grid, lifecycle fixture results,
input/code provenance, population identity digest, shard byte hashes/counts/action
counts/identity digests, deterministic failure lists, and a canonical `content_sha256`
computed with that field omitted. Each gzip member is written with empty filename and
`mtime=0`; each line is canonical JSON and the line order is
`(delay, offset, holdback, session_id, episode_id)`.

Every detail row has these required fields:

- schema/grid/input identity: schema version, grid ID, delay/offset/holdback,
  population SHA-256, episode content SHA-256, session ID, episode ID, pool/tag/status,
  epoch, epoch origin, scored start, processed scored end, and source-region digest;
- oracle action evidence: reference ID/kind, target and acceptable interval, requested
  offset, unclamped boundary, clamp direction, realized boundary/offset, evidence onset,
  requested availability, causal availability, chunk-realized apply frontier,
  released frontier at apply, recoverability label, and unrecoverable span;
- baseline/lifecycle evidence: every B0 hard action ID/coordinate/availability, every
  lifecycle event ID/kind/reason/event coordinate/apply frontier, every progress row or
  its canonical row list plus digest, and structural action reasons;
- ownership evidence: ideal and realized turn IDs with ordered half-open spans, final
  ring span, terminal release record, unrecoverable spans, duplicate-normalization
  records, and conservation/duplication/ordering booleans;
- metric inputs and results: singleton-speaker intervals or their bound canonical list
  and digest, B0/candidate turn spans, 50/100/200 ms contamination numerators and
  denominators, fragment durations, logical-action and drain latencies, and per-action
  late/fallback flags;
- failure evidence: every invariant flag, fallback class, clamp count, and deterministic
  row digest.

Quantiles use the nearest-rank convention: sort finite values ascending and select
one-based rank `ceil(q * n)` for q 0.50 or 0.95; an empty population yields `null` plus
count zero. No interpolation is permitted.

`verify_pcm_oracle.py` is a separately implemented reader/aggregator. It may import only
schema constants and canonical JSON/hash utilities, never assembler, ownership,
contamination, quantile, or aggregation functions from `pcm_oracle.py`. It independently
reconstructs span covers, contamination, counts, quantiles, grid aggregates, shard
identity digests, the population/clamp identity objects, and the
20/186/283/82,026/124,803/1,176 completeness ledger. Its self-hashed
verification JSON binds its own live-code hash, the main artifact hash, every shard byte
hash, recomputed aggregate hash, mismatch list, and pass/fail verdict. Mutation fixtures
must prove it rejects one missing row, duplicated span, altered ownership, altered
contamination numerator, and altered quantile.

A monolithic uncompressed per-case JSON is forbidden. The runner fails if the main JSON
exceeds 10 MiB or a compressed shard exceeds 20 MiB. This keeps reviewable aggregates
small while retaining exact detailed evidence and avoids repeating the previous
165 MiB single-file failure mode.

## 11. Verification and stop conditions

The phase stops and every downstream phase remains blocked if any of the following occurs:

- a frozen input, pinned-ledger item, per-episode hash, or B0 trace hash fails;
- a confirmatory held-out path is resolved or opened;
- any grid case loses, duplicates, reorders, or multiply owns a source sample;
- ideal old/new ownership disagrees in a case labeled fully recoverable;
- a stale-epoch action changes any state or output digest;
- safe drain releases a region before the safe frontier covers it;
- timeout fallback is pooled with ordinary safe completion;
- maximum-duration or terminal structural cuts are absent from the trace;
- B1 no-detector structural output differs from B0 after normalization;
- independent contamination recomputation differs from persisted results;
- no hard-only grid row reduces contamination relative to B0;
- population identity differs from the frozen digest, or counts differ from 20 sessions,
  186 episodes, 283 hard references, 142 hard-positive episodes, or 44 no-hard controls;
- clamp identity differs from the frozen digest, or counts differ from 24
  reference/offset combinations, 16 references, 15 episodes, 13 below-start, 11
  above-end, or 1,176 action instances;
- expected counts differ from 441 grid rows, 82,026 detail rows, 124,803 action
  instances, 11,718 rows per delay shard, or 17,829 actions per delay shard;
- any session/episode/reference/grid-cell identity is missing or duplicated, or any shard
  identity digest is incomplete;
- any generated artifact lacks canonical self-hash or live-code provenance.

Required verification is Black and Ruff over changed Python, the full
`experiments/speaker_turn_boundary/tests` suite, artifact-specific self-hash and shard
verification, the separately implemented verifier and its mutation fixtures, and a fresh
Phase 3 exit-gate Implementation Reviewer over the exact committed candidate.

## 12. Compute, storage, and access forecast

The workload is CPU-only interval/PCM replay over 186 already-opened public development
episodes plus deterministic synthetic fixtures. Expected execution is under 30 minutes,
peak memory under 1 GiB, main JSON under 10 MiB, and seven compressed shards under
20 MiB each. No GPU, download, network, credential, provider charge, irreversible
access, or production write is authorized.

## 13. Recorded review findings and dispositions

| ID | Severity | Disposition in rev 2 |
| --- | --- | --- |
| P3R-001 | blocker | Section 7 freezes `B0LifecycleReplay`, its event schema/coordinates/hashes, full-session source-prefix execution, parity to accepted B0 projections, and correct ordinary-versus-structural taxonomy. |
| P3R-002 | blocker | Sections 3-4 freeze absolute coordinates, nonzero epoch origin, exact conserved interval, release equation, no rounding, event precedence, pending-drain freeze, and partial-chunk behavior. |
| P3R-003 | blocker | Sections 3 and 6 forbid passing abandonment, require terminal conservation before reset, freeze an injected monotonic timeout, and cover EOF/reset with no more PCM. |
| P3R-004 | blocker | Section 5 freezes target-sample anchoring, clamp semantics, population identity SHA-256, and all session/episode/reference/row/action/shard completeness counts. |
| P3R-005 | blocker | Section 5 freezes the per-policy p95 population/rule, mandatory Phase 3 extension, downstream invalidation barrier, and the user-required report/resume gate before Phase 6. |
| P3R-006 | important | Section 10 freezes detail fields, completeness index, nearest-rank quantiles, and a separately implemented verifier with mutation rejection evidence. |
| P3R-004-R2-A | blocker | Section 5 now freezes the exact three identity-object keys, independent lexical array sorting, canonical JSON parameters, and recomputed `cb06483f...` SHA-256. |
| P3R-004-R2-B | blocker | Section 5 freezes 24 clamped reference/offset identities, their `22b4488a...` digest, 16-reference/15-episode/direction counts, and 1,176 grid action instances. |
| P3R-005-R2 | blocker | Section 5 requires identical episode/session/epoch and exact policy-run identity, restricts eligible detector-owned hard actions, and freezes ordered one-to-one no-reuse matching with explicit misses. |
| P3R-003-R2 | important | Section 6 freezes a per-drain FIFO with captured frontiers, independent deadlines, per-drain outcomes/latencies, duplicate suppression, regression rejection, and terminal/reset resolution. |

## 14. Reviewer checklist

The reviewer must return `approved`, `repair_required`, `not_reviewable`, or
`needs_user_decision` and examine at minimum:

1. half-open canonical PCM ownership and late-action semantics;
2. conservation/no-duplication proofs and actual PCM fixture coverage;
3. exact 7 x 9 x 7 grid, both 500 ms sentinels, and p95 extension trigger;
4. causal availability clamping and chunk scheduling quantization;
5. safe-frontier monotonicity, safe drain, timeout labeling, and stale epoch behavior;
6. maximum-duration, terminal, and B1 structural lifecycle coverage;
7. contamination and unrecoverable-audio calculations;
8. Phase-stopping behavior for every assembler or evidence-integrity failure;
9. held-out, provider, credential, and production-wiring exclusions;
10. compact artifact/shard format and independent recomputability.
