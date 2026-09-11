# P2 E2O-1 Review Record: adjudications, repaired verification, residual limits

## Accepted review findings and repairs (all in `experiments/psem_evidence_to_ownership/`)

1. Vacuous guard episodes: ACCEPTED. Guard spans now derive machine
   episode lists from the frozen source spans via session join
   (`episodes_for_span`): R1 [A00018], R2 [A00003, A00004], T1 [A00004,
   A00005, A00006]. Per-episode single-fire `simulate_episode` plus the
   in-span filter (frozen policy, unchanged) now runs over real episodes;
   T1 yields real in-span events (A00004 F0/H 702400, A00005 F0 734400,
   H 732800). First-overall per episode is recorded with an in-span flag
   and a suppression note. R1/R2 in-span empties verified genuine
   (first-overalls outside spans). Fourth guard stays explicitly
   unsupported-real; safety_hold is false.
2. Nominal schedule, not measured send: ACCEPTED. All margins relabeled
   NOMINAL SCHEDULED UPPER BOUNDS (`nominal_send_wall`,
   `nominal_scheduled_upper_bound_s`); send jitter UNKNOWN is separate
   from model/event lag UNKNOWN. No receipt-flip claims.
3. False measured defaults: ACCEPTED. No-request availabilities read
   evidence-absent (unmeasured), none-arm reads baseline-no-request
   (unmeasured); nothing defaults to measured. Evidence-absent is
   distinct from operation refusal.
4. Dead conservation accounting: ACCEPTED. `check_conservation` derives
   accepted-token IDs and text from the independent accepted records and
   compares actual output groups by ID multiplicity and exact text
   (missing/duplicate/unknown group IDs, missing token refs, text
   equality). Smoke proves detection with drop/duplicate negative
   controls; empirical ledgers carry derived (mostly empty) lists.
5. New code comments: ACCEPTED. Hash-comment lines removed from
   replay.py and capture.py (`grep -n #` clean); docstrings retained.
6. Unsupported freeze timestamp: ACCEPTED. Original FREEZE.json retained
   byte-identical (sha256 f96dbdf1...fc9ce90); FREEZE_ADDENDUM.json
   records the nominal day-label, observed session chronology, provenance,
   guard mapping, and timing/receipt/conservation corrections.

## Rejected adjudication

Repeated-emission / rearm claims: REJECTED per adjudication. First event
per episode plus span filter is the frozen policy; the decoder is not
restarted. Suppression is instead recorded truthfully (G03 H 6576000,
G04 F0/H 9097600 precede their spans; N4 H 625600 postdates its span).

## Repaired verification (observed)

- Contract smoke passes: all receiver scenarios plus conservation
  intact/drop/dup controls and no-reference-invention.
- Deterministic replay: two consecutive `--run` executions produce
  identical receipt hashes.
- Conservation empirical: P4 (30 groups / 53 tokens), G03 (18), G04 (17)
  all exact with derived empty missing/duplicate lists.
- Receiver statuses: applied / already_separated / unsupported observed
  empirically; too_late / invalid_scope observed in smoke only.
- Candidate receipt (`results/replay_receipt.json`) binds replay/capture
  sources, freeze + addendum, profiles, manifest, captures, all
  results/decisions, and frozen NPZ/reference inputs.

## Residual limitations

- No measured causal PASS exists: every applied receipt rests on a
  nominal schedule bound with unknown jitter and lag.
- Safety coverage is incomplete (fourth guard unsupported-real; T1
  scenario requests carry uncertainty; no timed text on guards).
- Second-source lexical ground truth: G04 NPZ in-span reference/episode
  speaker is MEE033 (reference-conditioned metadata, not an
  active-speaker count); existing annotation reuse mixes A/B/C/D token
  ownership (single-owner C both sides, 47 mixed tokens), and pinned
  ES2009a XML A/B/C/D segment overlap is a separately attributed
  annotation-coverage observation. G03 NPZ MEE008 scope exists but
  word-level ownership annotations for the captured text are absent
  in-repo, and no physical transition absence is claimed from metadata
  alone. A second positive needs new attribution-labeled material, which
  exceeds the spent 2/2 capture budget and is NOT authorized here; no
  unauthorized inference stands in for it.
- Decision remains GATE BLOCKED; no P3 branch opened.
