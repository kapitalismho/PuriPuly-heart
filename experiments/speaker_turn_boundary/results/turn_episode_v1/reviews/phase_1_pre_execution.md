# Phase 1 pre-execution review

Status: **approved** (independent review completed before the metadata coverage inventory
was built).

## Review identity

- Phase: 1 (metadata coverage inventory) — PRD Section 29, Phase 1.
- Reviewer: independent Implementation Reviewer worker (fresh sessions, read-only).
- Review date: 2026-08-08.
- Plan/self-hash under review: `.agents/specs/prd/bounded_turn_episode_speaker_change_fusion_experiment_review_gated.md`,
  git blob `24340f488f1bb46c666a5fc15eef2fc87ef1f826`, SHA-256 of bytes
  `8c6bed2e564b9ec80e26ee6b73701985c863b7beb68e51590be7a5faf173aad4`.
- Source under review: restart base `fef0a6b3`; Phase 1 review bundle rev 6 at branch HEAD
  `9d3db2d0ef1b87edc98f3316ad42e0e8bbf7136a` (range `fef0a6b3..9d3db2d0ef1b87edc98f3316ad42e0e8bbf7136a`).
  Bundle revision history: rev1 `2f5a03db`, rev2 `00ff8635`, rev3 `e3d88151`, rev4 `33550846`,
  rev5 `71c86b53`, rev6 `9d3db2d0` (range identity bound to branch HEAD).

## Phase scope and explicit non-goals

Scope: metadata-only coverage inventory over locally available authorized corpora (AMI,
AliMeeting Eval, LibriSpeech-derived synthetic) plus deterministic B0 production-VAD
baseline replay over the 12 already-materialized sessions; split-leak graph; frozen
natural-exposure frame; frozen target-enriched sampling rule; exact data-gap list and
compute/storage forecast.

Non-goals: no scored episode/reference manifests (Phase 2); no new LS/ERes or
speaker-change neural inference (Phase 4+); no clustering/fusion replay (Phase 5); no
confirmatory held-out access (Phase 7); no data additions or downloads; no corpus
materialization beyond what exists on disk.

## Prior-phase evidence the phase depends on

Phase 0 (approved): `reviews/phase_0_pre_execution.md`, `reviews/phase_0_review_bundle.md`,
`turn_episode/schemas.py`, `turn_episode/contracts.py`, `proposal_contract.json`,
`fusion_contract.json`. Historical hash ledger: `reviews/historical_artifact_ledger.json`.

## Exact inputs, manifests, caches, code/config hashes, and proposed outputs

Inputs: pilot manifests `data/manifests/*.json` (phase2-v1 schema); AMI annotations
`corpusResources/meetings.xml`, `corpusResources/participants.xml`, `annotations/words/*.words.xml`
(687 files, 171 meetings); AliMeeting `Eval_Ali_far/textgrid_dir/*.TextGrid` (8 files);
materialized audio `ami/audio/<meeting>/*.Mix-Headset.wav` (4) and `alimeeting/far_ch0/*.wav`
(8); production baseline code and the bundled Silero VAD ONNX
(`src/puripuly_heart/data/vad/silero_vad.onnx`, SHA-256
`1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3`) read-only.

Outputs: `results/turn_episode_v1/coverage_inventory.json` (canonical content hash);
`results/turn_episode_v1/coverage_inventory_details.jsonl`; per-session B0 replay evidence
under `results/turn_episode_v1/b0_inventory_replay/`; `reviews/phase_1_pre_execution.md`
(this artifact).

## Assumptions relevant to the phase

- B0 replay consumes only raw `replay_wav_epoch` boundary traces (successor-SpeechStart
  boundaries); historical coalescer output ignored; terminal/SpeechEnd/max-duration
  semantics deferred to Phase 3/8.
- B0-separated classification (frozen before replay): clean target B0-separated iff a raw
  boundary lies in `[B onset - 500 ms, B onset]`; gap target B0-separated iff a raw boundary
  lies inside `[A speech offset, B onset]`; 250 ms view reported for clean.
- Natural-exposure frame (frozen before label inspection): 30 s grid windows; inclusion iff
  `int(sha256(f"{session_id}:{start_ms}").hexdigest()[:2], 16) < 16` (1/16); sampled vs
  eligible duration recorded; only this pool may emit natural five-minute/session rates.
- Target-enriched selection (frozen): per-session hash-stratified rule, at most 12
  hard-positive and 12 negative intervals per session, non-overlap, eligible counts
  preserved, never converted to natural rates.
- Grouping (frozen): complete keep-together graph; AMI participant identity from
  `meetings.xml` `nxt_agent`->`global_name` and pilot manifests'
  `condition.partition_meta.agents`; AliMeeting `N_SPKxxxx` session-local; synthetic
  utterance/speaker families; graph hashed and fail-closed on cross-split overlap.
- Provenance: canonical trace-hash projection excludes `emitted_monotonic_ns` and
  wall-clock metadata; per-session evidence; 12-session completeness gate.
- Annotation-only AMI meetings never count toward independent-block gates; only scorable
  sessions with materialized audio do.

## Falsification/stop conditions

- Inventory fails if annotation parsing is not deterministic; phase stops, failure reported.
- B0 replay defects exclude affected sessions with recorded reason; block counts recomputed.
- Natural-exposure frame and target-enriched selection computed before label-based
  classification; touching labels first invalidates the frame.
- `coverage_inventory.json` requires all 12 per-session evidence files and a verified
  completeness check; partial results cannot produce it.
- Phase 1 gate: no data addition or sampling-rule change proceeds without separate review
  approval; no scored manifests are generated.

## Expected compute/data/provider cost and irreversible access

Metadata parsing minutes; B0 Silero VAD replay over 12 sessions (CPU, well under an hour);
no GPU, no provider, no downloads, no held-out access, no credentials; writes only under
`results/turn_episode_v1/`.

## Reviewer findings

- Round 1 (candidate `2f5a03db`): VERDICT `fix` — P1-RANGE-001 (blocker), P1-B0-001
  (blocker), P1-B0-002/003, P1-GROUP-001, P1-SAMPLING-001, P1-PROV-001 (important),
  P1-COUNT-001 (note). Resolved in bundle rev 2 (`00ff8635`).
- Round 2 (candidate `00ff8635`): VERDICT `fix` — P1-RANGE-002 (blocker), P1-GROUP-002,
  P1-PROV-002 (important). Resolved in bundle rev 3 (`e3d88151`).
- Round 3 (candidate `e3d88151`): VERDICT `fix` — P1-RANGE-002 repeat (candidate head
  stale) and Section 3 identity attribution. Resolved in bundle rev 4 (`33550846`).
- Round 4 (candidate `33550846`): VERDICT `fix` — P1-RANGE-003 (candidate head stale after
  range-fix commit). Resolved in bundle rev 5 (`71c86b53`).
- Round 5 (candidate `71c86b53`): VERDICT `fix` — P1-RANGE-004 (a commit cannot name its
  own hash). Resolved in bundle rev 6: candidate identity bound to `fef0a6b3..HEAD` with
  head-at-writing recorded; exact head confirmed by the reviewer at review time.
- Round 6 (candidate `9d3db2d0ef1b87edc98f3316ad42e0e8bbf7136a`): VERDICT `pass`; remaining
  findings: none.

## Final verdict

**approved** — Phase 1 pre-execution review for the bounded turn-episode speaker-change
fusion experiment (plan blob `24340f488f1bb46c666a5fc15eef2fc87ef1f826`, restart
`fef0a6b3`, bundle rev 6 at `9d3db2d0ef1b87edc98f3316ad42e0e8bbf7136a`).

Required changes: none outstanding.

Execution authorization: the metadata-only coverage inventory, the split-leak graph, the
frozen natural-exposure frame, the frozen target-enriched sampling rule, and the
deterministic B0 baseline replay over the 12 already-materialized sessions may now be
built, without any new LS/ERes or speaker-change neural inference, without data additions,
and without any confirmatory held-out access. The exact data-gap list resulting from the
inventory (expected: zero untouched AMI/AliMeeting sessions locally) is to be reported and
any data-addition proposal must be separately authorized before materialization.
