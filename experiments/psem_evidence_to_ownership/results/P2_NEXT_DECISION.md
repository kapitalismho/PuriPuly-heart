# P2 E2O-1 Next Decision: GATE BLOCKED (repaired evidence; no branch justified)

Freeze `psem.e2o1.freeze.v1` (`FREEZE.json` sha256
`f96dbdf14546ad106d9ac22d9240468bfa255947b8c35392c6190af35fc9ce90`)
plus `FREEZE_ADDENDUM.json` (versioned corrections; original retained).
Replay: `./.venv/Scripts/python.exe
experiments/psem_evidence_to_ownership/replay.py --run` (deterministic;
rerun-identical receipt hashes). Ledger
`results/P2_EVENT_OWNERSHIP_LEDGER.json` (59 records),
`results/evidence_bounds.json` (per-episode first-overall + in-span
filter), `results/contract_smoke.json` (pass, incl. conservation
drop/dup controls), `results/gate_evaluation.json`,
`results/replay_receipt.json` (hash-bound). Adjudications and residual
limits: `REVIEW_RECORD.md`.

## Repaired findings

- P4-SCORED (EN2009d A00343): H7301 (52158400) agrees 6/6 scored words;
  F0 (52139200) disagrees on do/that; control (52156984) reproduces the
  known that-word timing misplacement. Every applied receipt is
  conditional on a NOMINAL SCHEDULED UPPER BOUND (H 4.967 s, F0 5.927 s,
  control 5.778 s) with UNKNOWN send jitter and UNKNOWN model/event lag.
  Conservation is independently accounted (30 groups, 53 tokens, exact).
- P2-G03 / P3-G04: no in-span F0/H request under the frozen per-episode
  single-fire policy, truthfully explained: first-overall fires precede
  the spans (G03 H 6576000; G04 F0/H 9097600), consuming the episode fire.
  Evidence-absent-in-span, never evidence-absent-everywhere. New captures
  (budget 2/2 spent, both ok, text conserved) have no supported
  word-level ownership annotation in-repo: G04 NPZ in-span
  reference/episode speaker is MEE033 (reference-conditioned metadata,
  not an active-speaker count), while existing annotation reuse
  (owner_alignment) mixes A/B/C/D token ownership with single-owner
  tokens C before and after and 47 mixed tokens, so no clean lexical
  ownership change is established; separately, pinned ES2009a XML
  segments show A/B/C/D segment coverage overlapping the span seconds
  (annotation-coverage observation only, not fused with NPZ metadata).
  G03 NPZ MEE008 episode/reference scope exists, but the word-level
  ownership annotations needed for the captured text are absent in-repo,
  so no physical absence of a speaker transition is claimed from
  reference-conditioned metadata alone.
- Guards (episodes derived from session join, no longer vacuous): R1
  first-overalls fall outside the span (F0 3222400, H 4361600); R2
  first-overalls fall outside (A00003 F0 193600 / H 625600, A00004 702400
  above the span); R2 AB confirmed zero-frame invalid support. T1 has
  real in-span events (A00004 F0/H 702400, A00005 F0 734400 / H 732800);
  requested boundaries fall in A/B singleton regions, measured harm null
  (no timed text), uncertainty retained. Fourth guard real UNSUPPORTED
  (SYN-G synthetic only): safety coverage is incomplete, safety_hold false.
- Receiver statuses observed: applied / already_separated / unsupported
  empirically; too_late / invalid_scope in contract smoke only (no
  empirical negative-margin or invalid-scope request occurred).

## Engineering gate

Second-source positive: FAIL (P4 only; no supported word-level ownership
annotation for G03/G04 captured text in-repo). Causal applicability: FAIL
(zero measured PASSes; all applied conditional on nominal bounds).
Guards: no measured harm, but coverage incomplete (fourth guard) plus T1
scenario requests with uncertainty. Conservation: HOLD.

## Decision

Exactly one branch is allowed only if justified. None is: not a stop of
the target branch (P4 H-conditional agreement reproduced; T1 evidence
real), not a freeze (causality unmeasured, safety incomplete). Gate
BLOCKED. Remaining prerequisites: one second independent
attribution-labeled positive where H is silent in-span; measured/bounded
send + compute timing; a real fourth-guard mapping. Further captures
exceed the spent 2/2 budget and require an explicit budget change, and no
unauthorized inference is claimed in their place: the missing
measurements stay missing. The first two prerequisites select the single
next branch (P3-T on conversion failure with H in time; P3-O/P3-R on
indistinguishable states; bounded Audio decision on post-seal help); none
is opened here.

## Terminal review status

Executed probe trustworthy; Issue 142 definition of done is BLOCKED (no
justified listed branch). The prerequisites above are confirmed locally
as actual external data/timing prerequisites, not analysis artifacts.
