# Manifest — psem_pretranslation_ontology (immutable checkpoint)

Freeze `psem.pretranslation_ontology.freeze.v1` frozen 2026-09-10T07:43:43Z (actual clock via datetime.now UTC, before first probe run; never hand-typed futurestamp).
Baseline branch `experiment-v2-speaker-change-turn-boundaries-ls` commit `83aaed984b8b245082f3ffe7bb15d71f3242361f`, upstream 0/0. Snapshot exception: no commits (Director Git writer unavailable). Prior tracked 5M+ plus all prior untracked dirs preserved.
Ledger generated 2026-09-10T07:44:54Z, wall 26.5 s local, smoke 9/9 PASS, zero paid API, zero new captures, zero Git mutations. Only new dir `experiments/psem_pretranslation_ontology/` mutated; accepted `experiments/psem_pretranslation_receiver/` read-only reference.

## Owned deliverables (sha256)

- `RESULTS.md` `6a91d3626773e678dfeb0c509a68d50cc60f5c06bc439eacd207cfdb7e70ab14`
- `WAIT_GATE.md` `e93619973529a23166df1e2d3b79b33132444297b010f918d8006399ad8d2639`
- `NEXT_DECISION.md` `129dcbaba76cb8726a295690b78e67a332877922a5f7bcbd303fa1a6066a8a47`

## Frozen read-only inputs (sha256, verified match)

- `ARCHITECTURE.md` `8971048f129d49b48a3abf6b9d546906e155eee5ed3b1e845d60e281e062d068`
- `experiments/psem_pretranslation_receiver/FREEZE.json` `07527fddb22b3bcdde2ad893be80d6804f83c08b8fef85278c34a79df210358b`
- `experiments/psem_pretranslation_receiver/replay.py` `10494780cdee21c77835999859d461239ea5f72abb16076164cfe1b8fa0840c3`
- `experiments/psem_pretranslation_receiver/ledger.json` `97b2587c56dbea6c0d07f11a99293e496e0f30fecd5197e3b0ed12f827878c2f`
- `experiments/psem_pretranslation_receiver/RESULTS.md` `761d87b3cd902b81ec76b0c5e3195b6a07611e07d754b604e8960d9a3bde1697`
- `experiments/psem_pretranslation_receiver/NEXT_DECISION.md` `bfcdbfae0a51f7d3ab876977da633b48ec400abc08f82bf6dfa4a2b4db57064a`
- `experiments/psem_repeatability_stage2/captures/NP1.json` `a4ed5e84f9fd866263646660edde1c5400f1269de1f58a98befa044b19da5005`
- `experiments/psem_repeatability_stage2/captures/NP2.json` `a67b4a9953a1a08a55225b221f62af6208c767b67bedff6cdd72feef1dd76177`
- `experiments/psem_repeatability_stage2/captures/NP3.json` `d92884ae50dcbc4e841c5dbc13c805665130dcec9fddc43a4745a32ed329b22d`
- `experiments/psem_phase_a_headroom/observations/OBSERVATIONS.json` `3ec7c03c9c01bff859b9d6ce8d020aec40432c9deb7b951e1cf19be11a90b5aa`
- `experiments/psem_phase_a_headroom/old_grid_cache.json` `8f9474eecbd13c235062bbe8616f93de5b8b6b6316695d9864724b87dbb5814a`
- `experiments/psem_decision_sufficiency/FREEZE.json` `d131883413aa969c0aa6ed5eb32ac134f411ed511af372baa3bf6e5d88d5ac67`
- `experiments/psem_decision_sufficiency/replay.py` `2e08a9ece97c0b8f80730450f5f05124e7b3f97d6268c308af2603720700753d`
- `experiments/psem_p3t_rearm/FREEZE.json` `12e0cbeff1c38c239ff45d99032e1e6c66acb8674d01ef51c225a5103233592a`
- `experiments/psem_phase_a_headroom/FREEZE.json` `543eb0c6cfa36e651375cad5c23b7337cc0f5996ace56e74220b5af0bcb5689b`
- `experiments/psem_return_capacity/FREEZE.json` `9a7ab4c9a8b836b4f3076f40b61c53acfe8a43858cad351bda37092fa3a52faf`
- Provenance: the two accepted doc hashes above (RESULTS 761d87b3, NEXT bfcdbfae) are the HISTORICAL frozen inputs recorded in our FREEZE.json and are NOT rewritten. Upstream docs have since had doc-only revisions (RESULTS b3afed1c, NEXT 294c8f8f at this writing; further doc updates expected) while accepted runtime files (FREEZE 07527fdd, replay 10494780, ledger 97b2587c) verify byte-identical to frozen inputs. This comparison reads the accepted runtime only, so no rerun is triggered by doc revisions.

## Candidate count and APIs

- Candidates frozen: 2 (RICH #97, SIMPLE #98A shared-validity minimal contrast; #98B validity-diff excluded).
- No frozen named reference-free candidate identified in reviewed authorities (not selected/excluded permanently).
- New APIs: 0. No train/download/prod/comments. No Git mutation.

## Smoke (actual focused, 9/9 PASS)

simple-merges-overlap-before-confirm; simple-other-only-preserved; simple-none-preserved; gt-simple-maps-overlap-to-current; raw-preconfirm-collapse-no-false-extended-invalid-gap; simple-vs-rich-during-overlap-paired-otherwrong-vs-withheld; same-evidence-availability-loop-causal; exact-conservation-baseline-contexts; drop-dup-text-controls. Tiny fixtures throwaway after proof, no permanent test bloat.

## Rich recompute verify (not re-pinned wording)

Doc corrections (docs only; FREEZE/probe/ledger bytes unchanged): (a) per-word ledger rows read directly show NP2 gross 1 fix C1480 wrong->correct + 1 correct withheld B1653 + 1 wrong withheld C1481 (net correct +0, not gross 0; prior '2 wrong both withheld' shorthand superseded), NP1 gross 3 fixes A1118/1119/1120 + 1 wrong withheld A1117 (net +3), NP3 gross 2 fixes B323/324 + 2 wrong withheld B322/325 (net +2), R1 guard 0 fixes with B32 wrong->UNKNOWN plus A492/A493 correct->UNKNOWN withheld (not a B32 fix); (b) T1 gross R0->RICH 4 fixes B54-57 + 7 wrong withheld B46/48-53 + 7 correct withheld A128/132/134-138 (net 10->7 correct; prior net phrasing superseded), COMBINED gross 7 fixes + 8 wrong withheld + 8 correct withheld with A126 correct->wrong and B41 retained (net 10c2w16u); no universal 'every wrong-zero has correct-withheld' rule (NP1/NP3 wrong-only withheld). 3-NP aggregate verified by sum: R0 12c/10w/1u/1m (24 IDs) -> R2 17c/0w/6u/1m; gross fixes 6=3+1+2, correct withheld 1, wrong withheld 4=1+1+2; guards never summed. Accepted receiver docs keep their wording (upstream, owned separately, notified). Wait verdicts scoped to the available pre-terminal record only; no waiting experiment performed.

## Untouched

Accepted runtime files verified byte-identical to frozen inputs (FREEZE 07527fdd, replay 10494780, ledger 97b2587c); accepted doc revisions (RESULTS b3afed1c, NEXT 294c8f8f) are non-runtime provenance, see above. Soniox reference only (no ASR/speaker labels used); no commits; no new captures; no paid calls; no training; no production adoption. Full NP unchanged single-seal paired exact. Final disposition: HOLD ADOPTION, overlap-policy branch stopped (research whole continues); see NEXT_DECISION.md.

## Reproduce

```powershell
.venv/Scripts/python.exe experiments/psem_pretranslation_ontology/probe.py smoke
.venv/Scripts/python.exe experiments/psem_pretranslation_ontology/probe.py run
```
