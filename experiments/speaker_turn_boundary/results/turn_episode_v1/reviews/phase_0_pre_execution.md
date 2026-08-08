# Phase 0 pre-execution review

Status: **approved** (independent review completed before any experiment schema
implementation or new model execution).

## Review identity

- Phase: 0 (checkpoint and contract freeze) — PRD Section 29, Phase 0.
- Reviewer: independent Implementation Reviewer worker (fresh session, read-only).
- Review date: 2026-08-08.
- Plan/self-hash under review: `.agents/specs/prd/bounded_turn_episode_speaker_change_fusion_experiment_review_gated.md`,
  git blob `24340f488f1bb46c666a5fc15eef2fc87ef1f826`, SHA-256 of bytes
  `8c6bed2e564b9ec80e26ee6b73701985c863b7beb68e51590be7a5faf173aad4`.
- Source SHA under review: restart `fef0a6b312df34680d9db0fd858e28ae054ace89`; review bundle
  commits `46614cd3` (bundle) and `6e7f492c` (findings resolution) on branch
  `experiment-v2-speaker-change-turn-boundaries-ls`.

## Phase scope and explicit non-goals

Scope: freeze restart SHA, dirty-worktree inventory, plan self-hash, historical artifact
hash ledger; freeze the `turn_episode_v1` reference/action/fusion schema, the detector
progress/safe-frontier schema, and the B0 logical-finalize replay description with the
B1 equivalence boundary; produce this approved review artifact.

Non-goals: no scored episode/reference manifests (Phase 2); no new neural inference or model
execution (Phase 4+); no clustering/fusion/VAD-fusion replay (Phase 5); no held-out access
(Phase 7); no provider calls/credentials (Phase 8); no coverage inventory or data additions
(Phase 1).

## Prior-phase evidence the phase depends on

None (first phase of the new run). Historical artifacts (Phase 1-3 reports, Phase 3 1,369-row
development evidence, previously touched held-out groups) are inputs as historical evidence
and cached model output only, per PRD Sections 0 and 16.1. Their byte hashes are bound by
`reviews/historical_artifact_ledger.json`.

## Exact inputs, manifests, caches, code/config hashes, and proposed outputs

Inputs: plan PRD (blob/hash above); committed historical evidence under
`experiments/speaker_turn_boundary/results/phase3/`, `results/`, `data/` (hash ledger bound);
production baseline code read-only (`src/puripuly_heart/core/vad/gating.py`,
`experiments/speaker_turn_boundary/*`).

Outputs: `results/turn_episode_v1/reviews/phase_0_pre_execution.md` (this artifact);
post-approval: `turn_episode/` schema package with contract tests,
`results/turn_episode_v1/proposal_contract.json`, `fusion_contract.json`.

## Assumptions relevant to the phase

- Causal: `boundary_source_sample <= observed_source_sample_at_emit` for every event and
  control action; no future audio or reference labels in proposal/action generation.
- Split/held-out: Phase 0 opens nothing; diagnostic/frontier disjointness and held-out
  barriers are defined for later phases.
- State: reset-plus-warm-up scored evaluation is prohibited until the Section 5.4
  state-equivalence contract passes for the family/profile class; Phase 0 implements the
  prohibition gate only.
- Statistical: source-session blocks are the uncertainty unit; target-enriched exposure is
  never rescaled to natural rates.
- B0/B1: B1 must be action-for-action and source-sample-for-source-sample identical to B0 for
  the trace-visible B0 segmentation; SpeechEnd/terminal/max-duration finalization semantics
  are deferred to Phase 3/8 oracle validation.

## Falsification/stop conditions

- B0/B1 difference at Phase 5 invalidates policy interpretation (PRD Section 31.2).
- Phase 0 fails if schemas cannot express the Section 8/4.10/6/11/12 contracts or a frozen
  table is ambiguous.
- Phase 0 exit gate: no held-out access and no new model execution before the action/scoring
  contract passes its invariant tests and this review verdict is `approved`.

## Expected compute/data/provider cost and irreversible access

None. Negligible local test cost; no data, provider, credential, or held-out access; no
irreversible access. New files only under `experiments/speaker_turn_boundary/results/turn_episode_v1/`
and `.agents/runs/opencode/`.

## Reviewer findings

First review round (range `fef0a6b3..46614cd3`): VERDICT `fix`.

- P0-INV-001 (blocker): Phase-0 invariant list incomplete; invariant 26 only partially
  Phase-0-testable. Resolved by Section 11.2 of the bundle: complete per-invariant
  disposition (synthetic contract tests: 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
  17, 18, 19, 20, 24, 25, 27, 28, 29, 30, 31, 32, 34; gate-only: 1, 26, 35; later-phase:
  3, 21, 22, 23, 33, 36, 37).
- P0-B0-001 (blocker): B0 replay trace does not emit/retain SpeechEnd; terminal/max-duration
  boundaries not trace-visible. Resolved by bundle Section 5.2 with explicit deferral to
  Phase 3/8.
- P0-COALESCE-001 (important): historical 500 ms coalescer non-epoch-safe and
  boundary-ordered; recorded as non-normative historical evidence (bundle Section 5.3).
- P0-SCORE-001 (important): scoring contract expanded (bundle Section 7.2: gap-VAD validity,
  anticipatory-credit rule, overlap exclusion, warm-up/unscored exclusion, contamination
  algorithm, harm rules, timing metrics, natural-exposure rate labels).
- P0-SCHEMA-001 (important): benefit attribution moved per-reference (ReferenceOutcome);
  FinalAction gains observation/emission provenance and matched_reference_id (bundle
  Sections 6.3-6.4).
- P0-CAUSAL-001 (important): LogicalBoundaryCluster schema and explicit LS/ERes
  actionization mapping added (bundle Section 6.5).
- AUTH-001 (note): authority item #2 (handoff doc) absent; plan-only authority is
  provisional until the handoff is recovered and re-audited.

Second review round (range `fef0a6b3..6e7f492c`): VERDICT `pass`; remaining findings: none;
AUTH-001 remains a recorded non-blocking note.

## Final verdict

**approved** — Phase 0 pre-execution review for the bounded turn-episode speaker-change
fusion experiment (plan blob `24340f488f1bb46c666a5fc15eef2fc87ef1f826`, self-hash
`8c6bed2e564b9ec80e26ee6b73701985c863b7beb68e51590be7a5faf173aad4`, restart
`fef0a6b312df34680d9db0fd858e28ae054ace89`).

Required changes: none outstanding. Corrected artifact hash: review bundle rev 2 committed
as `6e7f492c` (file `experiments/speaker_turn_boundary/results/turn_episode_v1/reviews/phase_0_review_bundle.md`).

Execution authorization: Phase 0 deliverables may now be implemented
(turn_episode_v1 schemas, contract tests, proposal/fusion contract files) without any new
model execution or held-out access.
