# Phase 1 pre-execution review bundle — metadata coverage inventory

Status: review bundle for the mandatory Phase 1 pre-execution review (PRD Section 29,
Phase 1; immediate implementation order Section 34 steps 4-5). The Phase 0 review was
approved (`reviews/phase_0_pre_execution.md`); Phase 0 deliverables are committed. The
Phase 1 inventory has **not** been built yet.

## 1. Artifacts under review

| Item | Value |
| --- | --- |
| Normative plan | `.agents/specs/prd/bounded_turn_episode_speaker_change_fusion_experiment_review_gated.md` |
| Plan git blob | `24340f488f1bb46c666a5fc15eef2fc87ef1f826` |
| Plan self-hash | `8c6bed2e564b9ec80e26ee6b73701985c863b7beb68e51590be7a5faf173aad4` |
| Restart commit | `fef0a6b312df34680d9db0fd858e28ae054ace89` |
| Work branch | `experiment-v2-speaker-change-turn-boundaries-ls` |
| Current head | `3f0d6f47` (Phase 0 complete) |
| Phase 0 evidence this phase depends on | `reviews/phase_0_pre_execution.md` (approved), `reviews/phase_0_review_bundle.md`, `turn_episode/schemas.py`, `turn_episode/contracts.py`, `proposal_contract.json`, `fusion_contract.json` |
| Historical hash ledger | `reviews/historical_artifact_ledger.json` |

## 2. Phase scope and explicit non-goals

Scope (all metadata-only unless explicitly noted):

- Build the metadata-only coverage inventory per PRD Section 16.3 over the locally available
  authorized corpora (AMI, AliMeeting Eval, LibriSpeech-derived synthetic) and the existing
  pilot manifests.
- Compute: independent source-session counts; speaker-connected components; source and
  scored duration; hard clean/gap target counts; overlap soft targets; same-speaker pause
  intervals; stable same-speaker active exposure; B0-separated vs B0-missed hard targets;
  short-turn distribution; channel/microphone condition; word-alignment coverage;
  language/corpus; model-training overlap risk.
- B0 baseline replay: deterministic Silero VAD replay (the exact B0 replay path from
  `vad_baseline.py`/`run_b0_replay.py`) over **already-materialized** mono 16 kHz audio
  (4 AMI + 8 AliMeeting sessions) to classify hard targets as B0-separated vs B0-missed.
  This is the deterministic baseline engine, not neural inference, and uses no labels as
  input; it is included here because the Phase 1 deliverable list (Section 29 Phase 1)
  requires it.
- Split-leak graph construction over participant/speaker identity strings.
- Frozen source-time-uniform natural-exposure sampling frame (Section 5 below).
- Exact data-gap list and compute/storage forecast.

Non-goals (Phase 1):

- No scored episode/reference manifests (Phase 2).
- No new neural inference or detector execution (Phase 4+).
- No clustering/fusion replay (Phase 5).
- No confirmatory held-out access and no opening of held-out audio paths (Phase 7). The
  inventory never reads or opens scored audio beyond the already-materialized wav files
  listed in Section 4.
- No data additions or downloads (only the inventory may *identify* gaps; adding data is a
  separate authorization).
- No corpus materialization beyond what already exists on disk.

## 3. Local corpus state (verified on this machine)

Corpus root resolution (`corpus/external.py`): `STB_PHASE2_CORPORA_ROOT` env var or default
`%TEMP%/opencode/stb_phase2_corpora` (the default exists locally).

| Corpus | Local state | Notes |
| --- | --- | --- |
| AMI | 171 meetings with per-participant `words.xml` annotations (687 files); **audio materialized for 4 meetings only**: ES2003a, ES2004a, IS1008a, IS1009a (Mix-Headset 16 kHz mono wav) | words.xml is word-level v1.6.2 per-participant (word timing available for all 171 meetings; audio for 4) |
| AliMeeting Eval | 8 sessions with TextGrid interval tiers per speaker + far-field audio; 8 sessions already materialized as `far_ch0` mono 16 kHz wav | R8001_M8004, R8003_M8001, R8007_M8010, R8007_M8011, R8008_M8013, R8009_M8018, R8009_M8019, R8009_M8020 |
| LibriSpeech synthetic | dev-clean (202 cases), test-clean (202), test-other (202) episode manifests exist; generated wavs under `data/generated/` | synthetic source blocks per Section 5.5 |

Previously touched groups (historical validation only, decision 13-15, Section 16.4):

- `ami_dev_pilot`: ES2003a, IS1008a (touched in Phases 1-3 development).
- `ami_held_out_pilot`: ES2004a, IS1009a (touched as held-out in the previous run; now
  historical validation, never confirmatory).
- `alimeeting_eval_pilot`: all 8 AliMeeting sessions (touched in previous run).
- `ls_dev` / `ls_held_out_clean` / `ls_held_out_other`: LibriSpeech-derived episodes
  (touched; historical).

**Consequence (preliminary, to be confirmed by the inventory):** zero locally untouched AMI
audio sessions and zero untouched AliMeeting sessions exist. The Section 16.3 confirmatory
gate (at least eight independent blocks per corpus from each corpus) cannot be satisfied by
local materialization alone. The inventory freezes this fact and produces the exact
data-gap list with a materialization proposal (download from the authorized AMI/AliMeeting
sources) for separate authorization. Until then, pooled claims are capped at exploratory or
descriptive status, and no data addition occurs in Phase 1.

## 4. Exact inputs

- Manifests: `data/manifests/*.json` (b0_phase0, phase1_dev, ami_dev_pilot,
  ami_held_out_pilot, alimeeting_eval_pilot, ls_dev, ls_held_out_clean, ls_held_out_other,
  mixed_dev_pool, puripuly_like_provisional) — hashes in the historical ledger.
- Annotations (read-only): `%TEMP%/opencode/stb_phase2_corpora/ami/annotations/words/*.words.xml`
  (687 files, 171 meetings), `.../alimeeting/Eval_Ali/Eval_Ali_far/textgrid_dir/*.TextGrid`
  (8 files).
- Audio (read-only, for B0 baseline replay only): `.../ami/audio/<meeting>/<meeting>.Mix-Headset.wav`
  (4 files), `.../alimeeting/far_ch0/*.wav` (8 files).
- Production baseline code read-only: `src/puripuly_heart/core/vad/gating.py`,
  `experiments/speaker_turn_boundary/*`.

Proposed outputs:

- `results/turn_episode_v1/coverage_inventory.json` (Section 27.3 minimum artifact;
  canonical content hash included).
- `results/turn_episode_v1/coverage_inventory_details.jsonl` (per-session rows).
- `results/turn_episode_v1/reviews/phase_1_pre_execution.md` (this phase's review artifact).
- B0 replay per-session evidence under `results/turn_episode_v1/b0_inventory_replay/`.

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

## 6. Grouping, split-leak, and training-overlap rules (frozen)

- Primary uncertainty block: one source session (one AMI meeting, one AliMeeting session,
  one synthetic source group).
- Speaker-connected component: connected component over participant identity strings within
  a corpus (AMI participant IDs from words.xml filenames `{meeting}.{letter}.words.xml` with
  participant mapping from the meeting partition metadata; AliMeeting SPK ids from TextGrid
  tier names). Episodes from one component share one block.
- Split-leak graph: edges between any two sessions sharing a participant component, a
  recording, a channel view, or a synthetic source speaker; cross-split overlap fails closed
  (invariant 29).
- Training-overlap risk: AMI-trained LS checkpoints on AMI are in-domain model evidence;
  AliMeeting-trained on AliMeeting likewise; corpus and model-domain results stay
  stratified (Section 17).
- Pool roles frozen (Section 16.4): diagnostic_dev and frontier_dev must be group-disjoint;
  previously touched groups are historical_validation; confirmatory_heldout requires newly
  selected unused groups and stays inaccessible until the Phase 6 freeze + Phase 7 approval.

## 7. Minimum independent-block rules and data-addition trigger

- Confirmatory pooled AMI+Aimeeting product claim: >= 8 independent contributing blocks per
  corpus after participant-component grouping; 4-7 blocks -> corpus-exploratory; < 4 ->
  descriptive rows only (Section 16.3).
- Data additions are triggered **only** by an observed gap in the inventory (missing product
  stratum or independent-block shortfall), never by raw hour count (Phase 1 gate). Any
  addition proposal must be included in the accepted Phase 1 evidence and separately
  authorized before materialization.
- Per source session, default caps of 12 hard-positive and 12 negative episodes apply from
  Phase 2 on; all eligible counts before sampling are preserved in the inventory
  (Section 16.3).

## 8. Inventory fields and their effect on later phases

| Field | Later design impact |
| --- | --- |
| independent source-session count | statistical block count; exploratory/confirmatory status |
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

## 9. Falsification and stop conditions for Phase 1

- If the inventory cannot freeze the attainable independent-block count (because annotation
  parse fails deterministically), the phase stops and the failure is reported; no audio
  execution proceeds.
- If B0 replay reveals a materialization or decoding defect in the 12 local sessions, the
  affected sessions are excluded with recorded reason; the block count is recomputed.
- Natural-exposure frame must be computed before label inspection; if the frame code touches
  transition labels first, the phase fails and the frame is regenerated.
- Phase 1 gate: no data addition or sampling-rule change may proceed from the inventory
  until reviewed and approved; no scored manifests are generated.

## 10. Expected compute/data/provider cost and irreversible access

- Compute: metadata parsing (minutes); B0 Silero VAD replay over ~12 sessions of mono 16 kHz
  audio (Silero ONNX CPU, expected wall-clock well under one hour); no GPU, no provider.
- Storage: inventory JSONs small; B0 replay evidence per session small.
- Irreversible access: none. No downloads, no held-out access, no credentials, no writes
  outside `results/turn_episode_v1/`.

## 11. Reviewer examination checklist (from PRD Phase 1)

1. Inventory fields and how each affects later design decisions.
2. Independent-block and participant-component grouping rules.
3. The distinction between target-enriched and natural-exposure sampling.
4. The proposed source-time-uniform sampling frame for natural exposure.
5. Leakage risks and training-overlap metadata.
6. Minimum independent-block rules and the exact trigger for adding data.
7. Whether any proposed data addition targets an observed coverage gap rather than raw hours.
8. That the B0 replay scope is the deterministic baseline engine only and touches only
   already-materialized audio.
