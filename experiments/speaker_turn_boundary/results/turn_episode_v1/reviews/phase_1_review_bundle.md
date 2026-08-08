# Phase 1 pre-execution review bundle — metadata coverage inventory

Status: review bundle for the mandatory Phase 1 pre-execution review (PRD Section 29,
Phase 1; immediate implementation order Section 34 steps 4-5). The Phase 0 review was
approved (`reviews/phase_0_pre_execution.md`); Phase 0 deliverables are committed. The
Phase 1 inventory has **not** been built yet.

Revision history: rev 1 initial bundle (candidate `2f5a03db`); rev 2 resolves review
findings P1-RANGE-001, P1-B0-001, P1-B0-002, P1-B0-003, P1-GROUP-001, P1-SAMPLING-001,
P1-PROV-001, P1-COUNT-001 (candidate `00ff8635`); rev 3 resolves P1-RANGE-002,
P1-GROUP-002, P1-PROV-002 (this revision).

## 1. Artifacts under review

| Item | Value |
| --- | --- |
| Normative plan | `.agents/specs/prd/bounded_turn_episode_speaker_change_fusion_experiment_review_gated.md` |
| Plan git blob | `24340f488f1bb46c666a5fc15eef2fc87ef1f826` |
| Plan self-hash | `8c6bed2e564b9ec80e26ee6b73701985c863b7beb68e51590be7a5faf173aad4` |
| Restart commit (base) | `fef0a6b312df34680d9db0fd858e28ae054ace89` |
| Work branch | `experiment-v2-speaker-change-turn-boundaries-ls` |
| Review base..candidate | `fef0a6b3..e3d88151` (rev-3 candidate `e3d88151`; rev-1 candidate `2f5a03db`, rev-2 candidate `00ff8635`; working tree clean at bundle time) |
| Phase 0 evidence this phase depends on | `reviews/phase_0_pre_execution.md` (approved), `reviews/phase_0_review_bundle.md`, `turn_episode/schemas.py`, `turn_episode/contracts.py`, `proposal_contract.json`, `fusion_contract.json` |
| Historical hash ledger | `reviews/historical_artifact_ledger.json` |

## 2. Phase scope and explicit non-goals

Scope (metadata-only unless explicitly noted):

- Build the metadata-only coverage inventory per PRD Section 16.3 over the locally available
  authorized corpora (AMI, AliMeeting Eval, LibriSpeech-derived synthetic) and the existing
  pilot manifests.
- Compute: independent source-session counts; speaker-connected components; source and
  scored duration; hard clean/gap target counts; overlap soft targets; same-speaker pause
  intervals; stable same-speaker active exposure; B0-separated vs B0-missed hard targets;
  short-turn distribution; channel/microphone condition; word-alignment coverage;
  language/corpus; model-training overlap risk.
- **Annotation coverage, materialized-audio, and scorable-session counts are recorded as
  separate numbers.** Only scorable sessions with materialized audio count toward the
  independent-block gates (finding P1-COUNT-001).
- B0 baseline replay: deterministic production VAD replay over **already-materialized** mono
  16 kHz audio (4 AMI + 8 AliMeeting sessions) to classify hard targets as B0-separated vs
  B0-missed. Scope per Section 4 below.
- Split-leak graph construction (Section 6 below), with a bound graph hash.
- Frozen source-time-uniform natural-exposure sampling frame (Section 5 below).
- Frozen per-session target-enriched sampling rule (Section 7 below).
- Exact data-gap list and compute/storage forecast.

Non-goals (Phase 1):

- No scored episode/reference manifests (Phase 2).
- No new LS/ERes or speaker-change neural inference (Phase 4+).
- No clustering/fusion replay (Phase 5).
- No confirmatory held-out access and no opening of held-out audio paths (Phase 7).
- No data additions or downloads (only the inventory may *identify* gaps; adding data is a
  separate authorization).
- No corpus materialization beyond what already exists on disk.

## 3. Local corpus state (verified on this machine)

Corpus root resolution (`corpus/external.py`): `STB_PHASE2_CORPORA_ROOT` env var or default
`%TEMP%/opencode/stb_phase2_corpora` (the default exists locally).

| Corpus | Local state | Notes |
| --- | --- | --- |
| AMI | 171 meetings with per-participant `words.xml` annotations (687 files); **audio materialized for 4 meetings only**: ES2003a, ES2004a, IS1008a, IS1009a (Mix-Headset 16 kHz mono wav) | words.xml is word-level v1.6.2 per-participant and supplies word timing only (no identity); participant `global_name` actor IDs come from `corpusResources/meetings.xml` speaker elements and from the pilot manifests' `condition.partition_meta.agents` mapping |
| AliMeeting Eval | 8 sessions with TextGrid interval tiers per speaker + far-field audio; 8 sessions already materialized as `far_ch0` mono 16 kHz wav | R8001_M8004, R8003_M8001, R8007_M8010, R8007_M8011, R8008_M8013, R8009_M8018, R8009_M8019, R8009_M8020; speakers are `N_SPKxxxx` tier names (session-local) |
| LibriSpeech synthetic | dev-clean (202 cases), test-clean (202), test-other (202) episode manifests exist; generated wavs under `data/generated/` | synthetic source blocks per Section 5.5 |

Previously touched groups (historical validation only, decisions 13-15, Section 16.4):

- `ami_dev_pilot`: ES2003a, IS1008a (touched in Phases 1-3 development).
- `ami_held_out_pilot`: ES2004a, IS1009a (touched as held-out in the previous run; now
  historical validation, never confirmatory).
- `alimeeting_eval_pilot`: all 8 AliMeeting sessions (touched in previous run).
- `ls_dev` / `ls_held_out_clean` / `ls_held_out_other`: LibriSpeech-derived episodes
  (touched; historical).

**Consequence (preliminary, to be confirmed by the inventory):** zero locally untouched AMI
audio sessions and zero untouched AliMeeting sessions exist. The Section 16.3 confirmatory
gate (at least eight independent blocks per corpus) cannot be satisfied by local
materialization alone. The inventory freezes this fact and produces the exact data-gap list
with a materialization proposal (download from the authorized AMI/AliMeeting sources) for
separate authorization. Until then, pooled claims are capped at exploratory or descriptive
status, and no data addition occurs in Phase 1.

## 4. B0 baseline replay scope (frozen)

### 4.1 What runs and what does not

- The Phase 1 B0 replay is the **deterministic production VAD baseline only**. It executes
  the already-approved bundled Silero VAD ONNX model
  (`src/puripuly_heart/data/vad/silero_vad.onnx`, SHA-256
  `1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3`) with the peer gating
  profile (`PEER_VAD_SPEECH_THRESHOLD`, start debounce/commit chunks, 7000 ms max segment,
  500 ms hangover, 500 ms pre-roll ring, 512-sample chunks), exactly as implemented in
  `vad_baseline.py`. This is **not** new LS/ERes or speaker-change neural inference
  (finding P1-B0-002); it is the baseline engine whose identity is bound by the model hash
  above and by the already-committed B0 evidence.
- The legacy `run_b0_replay.py` CLI is **not** used for the inventory: it validates against
  the v1 `DatasetManifest` schema, while the AMI/AliMeeting pilot manifests use
  `experiments.speaker_turn_boundary.manifest.phase2.v1` and would be rejected (finding
  P1-B0-001). Instead, a dedicated inventory script
  (`turn_episode/build_coverage_inventory.py`) builds per-session records directly from the
  pilot manifests and calls `replay_wav_epoch` per session (raw trace path, no
  `DatasetManifest` validation, no coalescer).
- **Only raw VAD boundary traces are consumed**: `epochs[].boundaries` from
  `replay_wav_epoch` (successor-`SpeechStart` boundaries; the only boundaries the trace
  emits). The historical coalescer output (`coalesce_vad_and_detector`, declared
  non-normative in Phase 0) is ignored. Inferred terminal/`SpeechEnd`/max-duration
  boundaries are not emitted by the trace and are excluded from classification; their
  finalization semantics remain deferred to Phase 3/8 (finding P1-B0-003).

### 4.2 B0-separated vs B0-missed classification rule (frozen before replay)

For each hard reference target (Section 6 taxonomy), using only the raw B0 boundary trace:

- **Clean handoff** (target point = B onset): B0-separated iff a raw B0 boundary lies within
  `[B onset - 500 ms, B onset]` (primary localization tolerance, Section 12.1); otherwise
  B0-missed. A 250 ms view is also reported.
- **Gap handoff** (acceptable interval `[A speech offset, B onset]`): B0-separated iff a raw
  B0 boundary lies anywhere inside the acceptable interval (interval-valued matching,
  Section 6.2; invariant 7). Otherwise B0-missed. A B0 boundary inside the gap is valid
  product separation regardless of timing (invariant 9).
- No reference labels are consumed to produce the VAD trace; labels are used only after
  replay to classify.
- Classification happens once, deterministically; the rule is frozen here and in code before
  replay runs.

### 4.3 Provenance (finding P1-PROV-001)

- Bound before replay: session wav SHA-256 per session; annotation file SHA-256 per session
  (words.xml set / TextGrid); script and module code hashes (inventory script,
  `vad_baseline.py`, `events.py`, `config.py`); Silero model hash above; split-graph hash
  (Section 6); runtime settings (chunk size, hangover, pre-roll, max segment).
- Per-session evidence: deterministic trace content hash over a **canonical projection of the
  boundary trace and classification result that excludes `emitted_monotonic_ns` and all
  wall-clock/runtime metadata** (the raw `SpeakerBoundaryEvent.to_dict()` serializes
  `emitted_monotonic_ns` from `time.perf_counter_ns`, which is nondeterministic; finding
  P1-PROV-002). The projection retains those fields in a separate non-hashed section.
  Canonical projection fields: audio_epoch, boundary_source_sample,
  observed_source_sample_at_emit, confidence, source, debug (sorted).
- Completeness: the inventory is accepted only when all 12 expected sessions
  (4 AMI + 8 AliMeeting) have per-session evidence; partial runs are stored per session and
  cannot masquerade as a complete inventory. The `coverage_inventory.json` records the
  expected and completed session sets and a canonical content hash.

## 5. Frozen sampling frame for natural exposure (natural_exposure_validation)

Frozen before any transition-label inspection:

1. Window grid: fixed 30 s (480,000 samples) non-overlapping windows covering each eligible
   session's source timeline from sample 0.
2. Inclusion rule: deterministic hash — keep window starting at `start_ms` of session
   `session_id` iff
   `int(sha256(f"{session_id}:{start_ms}").hexdigest()[:2], 16) < 16`
   (expected 1/16 of windows).
3. Eligible sessions: any session with materialized mono 16 kHz audio and scorable
   annotation coverage (AMI 4, AliMeeting 8 under current materialization).
4. Recorded per window: session_id, start_ms, inclusion decision, eligibility. Unsampled
   exclusions and sampling probability (1/16) are recorded; sampled-vs-eligible duration is
   part of the inventory.
5. The frame is computed by the inventory **before** speaker-transition labels are
   inspected; the labels are only attached to the sampled windows later (Phase 7).
6. Five-minute/session/source-hour rates may be estimated only from this pool (invariant 30).
   Target-enriched metrics are never converted into natural rates (Section 13.6).

## 6. Grouping, split-leak, and training-overlap rules (frozen)

Complete group graph (finding P1-GROUP-001); all rules frozen before the inventory runs:

- Primary uncertainty block: one source session (one AMI meeting, one AliMeeting session,
  one synthetic source group).
- **Keep-together edges** (any edge → same block):
  - complete original source session;
  - meeting series and related submeetings (AMI meeting id prefix series such as ES/IS; any
    two meetings sharing a series prefix are kept together conservatively);
  - recurring participant connected components: AMI via `global_name` actor IDs from the
    **`corpusResources/meetings.xml`** speaker elements (`nxt_agent` -> `global_name`,
    parsed by `corpus/ami.py`); the pilot manifests carry the same mapping as
    `condition.partition_meta.agents` (meeting-local letter -> global participant id) and
    are the authoritative local source for the four materialized meetings; participants
    without a discoverable global id are treated as meeting-local and never linked across
    meetings. The `words.xml` files contain word elements only and are not an identity
    source (finding P1-GROUP-002);
  - all channel views and derivatives of one recording (AliMeeting `far_ch0` and
    `Eval_Ali_far` audio are one recording);
  - original and transformed synthetic audio (gain, codec, noise, prosody derivatives of one
    source utterance);
  - all utterances from one synthetic source speaker;
  - every episode sharing any source sample.
- AliMeeting speaker identity: `N_SPKxxxx` TextGrid tier names are session-local with no
  discoverable cross-session identity; each AliMeeting session is therefore its own speaker
  component (no cross-session linking).
- The group graph is serialized deterministically and its content hash is bound in the
  inventory; cross-split overlap fails closed (invariant 29).
- Training-overlap risk: AMI-trained LS checkpoints on AMI are in-domain model evidence;
  AliMeeting-trained on AliMeeting likewise; corpus and model-domain results stay stratified
  (Section 17).
- Pool roles frozen (Section 16.4): diagnostic_dev and frontier_dev must be group-disjoint;
  previously touched groups are historical_validation; confirmatory_heldout requires newly
  selected unused groups and stays inaccessible until the Phase 6 freeze + Phase 7 approval.

## 7. Frozen target-enriched sampling rule (finding P1-SAMPLING-001)

Deterministic per-session hash-stratified sampling, frozen before Phase 2:

1. Enumerate, per session: all eligible hard-positive reference intervals (clean/gap
   targets) and all eligible negative intervals (same-speaker pause intervals), each with a
   deterministic rank by (interval start sample, interval end sample, reference id).
2. Selection within a session: keep interval `i` of pool `p` iff
   `int(sha256(f"{session_id}:{p}:{rank_i}").hexdigest()[:2], 16) < 16`; take the first
   (by hash-key order) at most 12 hard-positive and at most 12 negative intervals per
   session, subject to non-overlap (a kept interval drops later intervals that overlap it,
   in deterministic rank order) and episode coverage constraints (Section 5.1).
3. All eligible counts before sampling are preserved in the inventory (Section 16.3); the
   sampled subsets are recorded with their hash keys.
4. Target-enriched episode pools report only per-sampled-exposure metrics and are never
   converted into natural five-minute/session rates (Section 13.6, invariant 30).

## 8. Minimum independent-block rules and data-addition trigger

- Confirmatory pooled AMI+Aimeeting product claim: >= 8 independent contributing blocks per
  corpus after participant-component grouping; 4-7 blocks -> corpus-exploratory; < 4 ->
  descriptive rows only (Section 16.3).
- Block gates use **scorable sessions with materialized audio only**; annotation-only
  coverage never counts toward the gates (finding P1-COUNT-001).
- Data additions are triggered **only** by an observed gap in the inventory (missing product
  stratum or independent-block shortfall), never by raw hour count (Phase 1 gate). Any
  addition proposal must be included in the accepted Phase 1 evidence and separately
  authorized before materialization.

## 9. Inventory fields and their effect on later phases

| Field | Later design impact |
| --- | --- |
| independent source-session count (annotation-only vs materialized vs scorable) | statistical block count; exploratory/confirmatory status |
| speaker-connected component count | block inflation risk; split-leak graph |
| source and scored duration | exposure denominators; episode budget |
| hard clean/gap targets (total, per session) | target-enriched pool construction; B0-miss count |
| overlap soft targets | overlap_present pool; soft-marker diagnostics |
| same-speaker pause intervals | negative pools; pause-split cost analysis |
| stable same-speaker active exposure | harm denominators (active-speech-hour rates) |
| B0-separated vs B0-missed hard targets | recovery vs acceleration attribution (F5) |
| short-turn distribution | refractory stress coverage (F9) |
| channel/microphone condition | stress stratum reporting |
| word-alignment coverage | lexical-split observability (invariant 19) |
| language and corpus | stratified reporting |
| model-training overlap risk | in-domain vs unseen-corpus labeling |

## 10. Falsification and stop conditions for Phase 1

- If the inventory cannot freeze the attainable independent-block count (because annotation
  parse fails deterministically), the phase stops and the failure is reported; no audio
  execution proceeds.
- If B0 replay reveals a materialization or decoding defect in the 12 local sessions, the
  affected sessions are excluded with recorded reason; the block count is recomputed.
- Natural-exposure frame and target-enriched selection must be computed before label-based
  classification uses them; if the frame code touches transition labels first, the phase
  fails and the frame is regenerated.
- The inventory is accepted only with all 12 per-session evidence files and a verified
  completeness check; partial results cannot produce `coverage_inventory.json`.
- Phase 1 gate: no data addition or sampling-rule change may proceed from the inventory
  until reviewed and approved; no scored manifests are generated.

## 11. Expected compute/data/provider cost and irreversible access

- Compute: metadata parsing (minutes); B0 Silero VAD replay over 12 sessions of mono 16 kHz
  audio (Silero ONNX CPU, expected wall-clock well under one hour); no GPU, no provider.
- Storage: inventory JSONs small; B0 replay evidence per session small.
- Irreversible access: none. No downloads, no held-out access, no credentials, no writes
  outside `results/turn_episode_v1/`.

## 12. Reviewer examination checklist (from PRD Phase 1)

1. Inventory fields and how each affects later design decisions.
2. Independent-block and participant-component grouping rules.
3. The distinction between target-enriched and natural-exposure sampling.
4. The proposed source-time-uniform sampling frame for natural exposure.
5. Leakage risks and training-overlap metadata.
6. Minimum independent-block rules and the exact trigger for adding data.
7. Whether any proposed data addition targets an observed coverage gap rather than raw hours.
8. That the B0 replay scope is the deterministic production VAD baseline only, binds the
   approved Silero model hash, consumes only raw boundary traces, ignores the historical
   coalescer, and touches only already-materialized audio.

## 13. Recorded review findings and dispositions

| id | severity | finding | disposition |
| --- | --- | --- | --- |
| P1-RANGE-001 | blocker | bundle recorded stale head; candidate is 2f5a03db | resolved in Section 1 (rev-1 range fef0a6b3..2f5a03db) |
| P1-RANGE-002 | blocker | rev-2 bundle still named the rev-1 candidate | resolved in Section 1 (rev-3 range fef0a6b3..e3d88151) |
| P1-B0-001 | blocker | legacy B0 runner rejects phase2-v1 manifests | resolved in Section 4.1 (dedicated inventory script, raw `replay_wav_epoch` path) |
| P1-B0-002 | important | "not neural inference" false; Silero is ONNX | resolved in Section 4.1 (baseline engine wording; model hash bound) |
| P1-B0-003 | important | classification inputs unspecified; coalescer non-normative; terminal boundaries deferred | resolved in Sections 4.1-4.2 (raw traces only; frozen classification rule) |
| P1-GROUP-001 | important | incomplete keep-together rules; identity namespaces | resolved in Section 6 (complete group graph; AMI global_name; AliMeeting session-local) |
| P1-GROUP-002 | important | AMI global_name source misattributed to words.xml | resolved in Section 6 (meetings.xml speaker elements; partition_meta.agents in manifests) |
| P1-SAMPLING-001 | important | target-enriched selection not fully frozen | resolved in Section 7 (deterministic hash-stratified per-session rule) |
| P1-PROV-001 | important | provenance/complete-set requirements unspecified | resolved in Section 4.3 (bound hashes, per-session evidence, completeness gate) |
| P1-PROV-002 | important | trace hash nondeterministic via emitted_monotonic_ns | resolved in Section 4.3 (canonical projection excluding monotonic ns and runtime metadata) |
| P1-COUNT-001 | note | annotation coverage could falsely satisfy gates | resolved in Sections 3 and 8 (separate counts; scorable-only gates) |
