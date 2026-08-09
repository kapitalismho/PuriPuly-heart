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
differ in 74/186 episodes with the frozen 5 s warm-up. A convergence diagnostic
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

## Exit-gate verification and remediation

The Phase 2 exit gate was independently verified (fresh reviewer, candidate
`8717d6fe`): VERDICT `fix` with P2-REF-001/002/003, P2-SCORE-001/002, P2-AUDIT-001/002,
P2-STATE-001/002, P2-INTEGRITY-001/002. All findings were resolved in candidate
`1ba8a362` (+ `d51a248c`):

- References: intervals clipped to the processed scored region (P2-REF-001); episode
  tags preserved on every reference incl. diagnostic_only and natural windows
  (P2-REF-002); structural references at episode edges emitted (P2-REF-003).
- Matcher: gap tolerance closure applied exactly once via a `:gap` reference marker
  (P2-029 fixture); warm-up actions rejected via scored-region context; maximum-
  cardinality augmenting-path matching with the lexicographic Section 12.2 objective
  and deadline views (P2-SCORE-001); clean/gap headline masked to hard_only episodes
  with overlap reported separately; lexical splits computed with word-timing
  observability (171 splits, 47 not_observable) plus overlap/lexical fixtures
  (P2-SCORE-002).
- Audit: per-pool sampling with floor 8 per pool; independent re-derivation code
  path; byte-identical slice comparison against build-time slice SHA-256; id/scorable/
  tag consistency checks (P2-AUDIT-001/002).
- State equivalence: real capture/restore round trip (gating fields, pre-roll ring,
  pending start, RNN hidden state) reproducing the source-prefix trace exactly —
  186/186 passed; finding text corrected to 74/186 parity failures (112 trivial
  passes) with disposition source_prefix_required (P2-STATE-001/002).
- Integrity: fail-closed manifest verification in every consumer; full
  `generated_from` ledger and `structural_taxonomy_status` in every artifact
  (P2-INTEGRITY-001/002).

Round 2 independently re-reviewed candidate `1ba8a362` and returned `fix` with nine
findings (six blocker, three important). Candidate `f410c380` remediated reference
coverage, audit independence, ordered matching/scoring, state capture/fallback,
artifact integrity, and split enforcement findings.

Round 3 independently re-reviewed candidate `f410c380` and returned `fix` with seven
findings (five blocker, two important): capture timing at scored start, stable-overlap
episode classification, fail-closed missing-waveform audit, globally lexicographic
ordered matching, deadline-valid B0 acceleration attribution, live-code provenance,
and per-episode content hashes. The working-tree remediation produces:

- `episode_manifest_dev.json`: 804 episodes, content SHA-256
  `deb1713cd581c93cc13c47103643041c7551993985379869cfa0ca9a407dff68`,
  804/804 per-episode hashes verified.
- `natural_exposure_manifest.json`: 74 windows, content SHA-256
  `e7c8562602685925e4ccb1964801d384c555813d3f565d5a67c7750770b088f3`,
  74/74 per-episode hashes verified.
- `state_equivalence_report.json`: 423,276 bytes, content SHA-256
  `74f2d122c40e66e0f9212900b42fe7b6b3ec1d2c722a60a9ea9d58cba7a3eeec`;
  capture/restore round-trip 186/186, capture hashes 186/186, parity 112 pass/74
  fail, disposition `source_prefix_required`.
- `scoring_fixture_report.json`: content SHA-256
  `331a8d54394dbc30ad59d208f470ebafc3c5c2abcb9016b4550b8b52522b625a`,
  25/25 fixtures pass and the B0 baseline smoke covers 186 episodes.
- `audit_report.json`: content SHA-256
  `6b9963e8849af2ad13dd5632b813d7bd05b0e65462a739b1f237fa1ebd310475`;
  12 public, 9 diagnostic, and 25 synthetic samples with zero waveform unavailable,
  waveform, slice, annotation, or tag failures.

All five candidate artifacts pass their canonical self-hash and `generated_from`
checks against the formatted live code. The full experiment test suite collects and
passes 280 tests. Round 4 exit-gate re-review is pending; these results are candidate
evidence until that review accepts the exact commit.

Round 4 independently reviewed exact candidate `933561c3` and returned
`repair_required`. The reviewer found that the three fields labeled
`*_before_resume` were serialized after resumed audio had mutated the restored state,
and that `capture_sha256` included runtime UUID values and monotonic emission times.
The working-tree repair records the pre-resume ring and pending fields immediately
after restore and binds the capture through the declared
`runtime_identity_normalized_v1` hash projection, preserving UUID equality
relationships while excluding runtime identity and timing values.

The regenerated `state_equivalence_report.json` is 435,052 bytes with content SHA-256
`c5e4836f69686587bad0b24a2293e0f80b336ba5105651889006aee4a7db3c2c` and live
`state_equivalence.py` provenance SHA-256
`44af3271f4e076a872e6ed0a18e315c1374d347123daa1ded3833156fb5f0ca8`.
All 186 round trips pass; all 186 capture hashes are present under the declared hash
contract; captured/pre-resume ring, pending-ID, and pending-content mismatches are all
zero; ring and pending fidelity/parity failures are all zero. An independent rerun of
the first real episode reproduces its persisted `capture_sha256`. The state-equivalence
disposition remains `source_prefix_required` with 112 parity passes and 74 failures.
All five Phase 2 artifacts pass their artifact-specific canonical self-hash verifier,
all live-code provenance entries match, and all 878 manifest episode hashes verify.
Black, Ruff, and all 281 collected experiment tests pass. Round 4 repair re-review
remains pending until the coherent repair is committed.
