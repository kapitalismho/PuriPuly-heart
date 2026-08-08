# Phase 2 pre-execution review

Status: **approved** (independent review rounds 1-7, findings P2-001..P2-038 resolved;
bundle rev 8 records the implementation-time scored-start alignment amendment; Phase 2
was then executed and the exit gate is verified separately).

## Review identity

- Phase: 2 (episode/reference implementation) — PRD Section 29, Phase 2.
- Reviewer: independent Implementation Reviewer worker (fresh sessions, read-only).
- Review date: 2026-08-08.
- Plan/self-hash under review: `.agents/specs/prd/bounded_turn_episode_speaker_change_fusion_experiment_review_gated.md`,
  git blob `24340f488f1bb46c666a5fc15eef2fc87ef1f826`, SHA-256 of bytes
  `8c6bed2e564b9ec80e26ee6b73701985c863b7beb68e51590be7a5faf173aad4`.
- Review bundle: `reviews/phase_2_review_bundle.md` rev 8 (rounds 1-7; final approval
  verdict in round 7 at candidate `c64b47ee`; rev 8 amendment recorded at candidate
  `61a1807f` and re-verified at the exit gate).

## Phase scope and explicit non-goals

Scope (as approved): bounded-episode extraction over the 20 opened sessions
(Section 5.1); interval-valued references (Sections 6, 6.7); deterministic
self-hashed manifests (Section 27.3); state-equivalence fixtures for B0 with
pending-start inspection and snapshot fallback (Section 5.4); contamination/harm
scoring fixtures (Sections 13-14); unbiased natural-exposure manifest (Section 16.4);
sampled waveform/annotation audit; diagnostic/frontier group-disjointness
(invariants 27, 29).

Non-goals: no confirmatory held-out access (8 reserved AMI sessions unopened;
`episode_manifest_heldout.json` deferred to Phase 6/7); no LS/ERes inference (Phase
4); no clustering/fusion replay (Phase 5); no provider oracle (Phase 3); no natural
rate estimation (Phase 7+).

## Execution record (post-approval)

- `turn_episode/build_episodes.py`: builder + ReferenceBuilder + manifests.
  Outputs `episode_manifest_dev.json` (804 episodes: 198 public
  [186 scorable / 12 diagnostic-only] + 606 synthetic) and
  `natural_exposure_manifest.json` (74 windows, 2,203,833 sampled ms over the 20
  opened sessions; reserved-session windows remain duration-only in the Phase 1
  frame).
- `turn_episode/state_equivalence.py`: B0 readiness (pending-start inspection;
  chunk-aligned scored-start extension) + parity + snapshot round-trip.
  Outputs `state_equivalence_report.json`.
- `turn_episode/scoring.py`: full deterministic matcher (Section 12), turn-owner
  thresholds, contamination algorithm (Section 13), harm flags (Section 14),
  known-answer fixtures, B0 baseline smoke. Outputs `scoring_fixture_report.json`.
- `turn_episode/audit.py`: deterministic per-pool sampling, byte-identical waveform
  slices, independent annotation re-derivation. Outputs `audit_report.json`.
- `schemas.py`: added `EpisodeStatus` and `WindowType` literals.

## Key scientific finding (state equivalence)

The B0/peer class **fails** the state-equivalence gate: the Silero VAD v5 RNN hidden
state carries long context; source_prefix vs episode_reset scored-region traces
differ in 186/186 episodes with the frozen 5 s warm-up. A convergence diagnostic
shows exact parity only from ~60 s warm-up. Per PRD Section 5.4 and invariant 26,
**B0 scored evaluation must use source-prefix state** (full-session replay with the
episode scored region sliced from the source-prefix trace); reset-plus-warm-up is
forbidden for B0/peer. The failed parity cases remain diagnostic evidence and are
not hidden by increasing warm-up. LS/ERes classes default to source-prefix until
their Phase 4 parity passes.

## Reviewer findings

- Rounds 1-7: P2-001..P2-038 (17 blocker, 14 important, 7 note) — all resolved in the
  bundle (see `phase_2_review_bundle.md` Section 17 disposition table).
- Final verdict: **approved** (round 7).
- Rev 8 amendment (post-approval, implementation-time): `scored_start` rounding
  changed from ceil to floor to guarantee the frozen >= 10 s scored minimum
  (10 s = 160000 samples is not a multiple of the 512-sample chunk). Recorded in the
  bundle revision history; re-verified at the exit gate.
