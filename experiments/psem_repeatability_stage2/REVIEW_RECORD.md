# Review Record — Stage2 (checkpoint critics + Director corrections)

No claim is accepted here. State remains IMPLEMENTATION_READY (not accepted).

## Checkpoint critics wave (all 3 lanes, as reported at barrier)

- Verdict relayed by Director: no material findings in any of the 3 lanes.
- No critic content is reproduced or answered here beyond the concrete
  adjudication below; nothing was changed on the basis of the critics alone.

## Director concrete adjudication (terminal-gating correction)

Finding (accepted): `partition_ownership` removed GT-straddling matched
words from the wrong-owner count while the none-baseline includes them —
asymmetric denominators. NP3 H showed 1 ambiguous (Well) with
right_assigned_new 3, but RESULTS claimed “clean 4→0”, hiding one excluded
word behind an “all 8” denominator.

## Scoring-proof repair (no model/reference/policy change)

- Whole-word assignment and the uncertainty flag are preserved; the actual
  deterministic provider-end assigned side of a straddling word is NEVER
  discarded and never silently treated as correct.
- Per arm the ledger now reports the fixed matched cohort plus:
  definite errors, uncertain-straddle count (each with its preserved
  assigned side), the error interval [definite, definite+uncertain],
  right-assigned-new definite + uncertain counts, and ±1280 sensitivity in
  the same definite/pessimistic accounting (no tuning).
- Gate strict improvement must hold BOTH definitely and with uncertainty
  pessimistically counted, so it can never be obtained by abstaining.
- NP3 H truth (directly observed per GT ID, identical 8-word cohort on all
  arms): baseline 4 wrong; H definite 0, uncertain 1 (Well GT-straddles
  boundary 2611200 while its consumer end 2614000 assigns right — the
  correct side, flagged uncertain, not hidden); 3 right words definitely
  correct + 1 uncertain; error interval [0,1] vs baseline 4; pessimistic
  reduction 3 > 0 → supported benefit stands as “3 correct + 1 uncertain”.
- RESULTS corrections: NP3 denominator reads matched/unmatched/mixed =
  8/0/0 with a separate per-arm boundary-uncertainty metric (H: 1);
  G03/G04 use ledger denominator names/counts (G03 matched 13 / unmatched
  33 / mixed 3 over 49 payload GT words in 18 groups; G04 10/16/0 over 26
  in 17 groups); gate phrasing is “1 of 3 new cases vs threshold 2”.

## Correction lineage (hashes preserved, nothing overwritten)

- Preliminary (pre-barrier): ledger body `84b16bf3…`, receipt stable
  `d942ff89…` (timing PENDING).
- Integrated (barrier, control fix + 6 invariants): ledger body
  `4e27abc9…`, receipt stable `337ebd4a…`.
- Corrected (this adjudication): see current `receipt.json`
  (`ledger_body_sha256`, `stable_hash`). Old hashes above are the record;
  old result semantics (NP3 “clean 4→0”) are superseded by “3 correct + 1
  uncertain”, same evidence, same cohort, no word hidden.
- No changes to sibling `timing/` or original P2; no new captures/deps;
  P2 R2 report-value diagnosis from integration stands (code-identical,
  report stale).
