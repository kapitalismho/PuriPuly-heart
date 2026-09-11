# P3T Fixed F0 Excursion Rearm — Result (freeze v1, 36/36 PASS)

Freeze `psem.p3t.rearm.freeze.v1` (frozen 2026-09-09T20:46:58Z before comparison).
Ledger `per_case_ledger.json` from `probe.py run` (wall 6.4 s local).
Smoke `probe.py smoke` 13/13 PASS. Run checks 36/36 PASS.
Inputs bound in FREEZE.json and MANIFEST.md. No training, paid, captures, inference, production, publication, or Git mutations.

## Mechanism answer

Rearm reproduces the NP1 conversion with no incremental guard harm versus ORIG.

Positives, fixed cohort, same deadline, legal measured virtual profile:

- NP1 ES2009c: none [4,4], orig invalid_scope [4,4] (early 19682560 pre-payload), rearm applied 19808000 [0,0]. Deltas rearm vs none and vs orig strict -4/-4. Paired rearm vs orig: 4 fixed, 0 new wrong, 0 new pure UNKNOWN. Availability 5.265 vs seal 7.282.
- NP2 ES2009d: none [2,2], orig applied 33472000 [0,0], rearm same [0,0]. Delta rearm vs orig 0/0, no new wrong. Preservation, not independent gain.
- NP3 ES2002b: none [4,4], orig applied 2611200 [0,1], rearm same [0,1] (Well straddle). No additional uncertain, no new wrong.

Reference control_capacity carried GT nonbinding: NP1 [1,1], NP2 [1,1], NP3 [0,0]. Not counted as success.

## Guards, same word sets, SOURCE scope only, ASR UNKNOWN

Full EN2009d epoch 168000..755520 fresh: ORIG single 672000 emit 684896; REARM 672000 plus 733440 emit 746336. Old 374400 verified no-old-support, never narrated as fresh.

- R2 [670592,701312]: none [6,6] vs orig [3,5] vs rearm [3,5]. Rearm vs orig: 0 new wrong, 0 new pure UNKNOWN. Absolute vs none: 2 new pure-A wrong (Yeah words126, We are words128) plus 1 new pure UNKNOWN (it words124) with 4 fixed — residual baseline harm, SAME in orig, not incremental rearm.
- T1 [701760,755520]: none [11,11]; orig scoped none [11,11] because 672000 is OUT OF SCOPE invalid_scope, never applied as benefit; rearm applied 733440 [6,7]. Rearm vs orig: 6 stillwrong (previously wrong B), 4 fixed, 1 wrong to uncertain (re-jig straddle), 0 new pure-A wrong. Mostly previously wrong B corrected or still residual, no introduced A wrong.
- COMBINED [670592,755520] diagnostic nonbinding: rearm applied 672000, second 733440 already_separated. First wins by design; second cannot apply under one-boundary receiver. Application limit exposed, contract unchanged.
- R1 no applied boundary on any arm (10 proxy words, baseline [1,1] reference scope, zero intervention). BC1 oracle carried slot 3 nonbinding, no in-span applied boundary (4 proxy words baseline [4,4], zero intervention). Singletons ES2009c/ES2009d no fires, [0,0]. No new confident wrong or pure UNKNOWN on any mapped guard word rearm vs orig.

Conservation exact every case (48/27, 29/21, 33/20) with DROP DUP TEXT controls detected. Fragmentation: NP1 2 requests (1 invalid_scope plus 1 applied), NP2/NP3 1 applied each, zero redundant already_separated on positives, zero unnecessary fragments. QA merge nullable no grounded pair (157/251/186 resolved). Sensitivity plus minus 1280 nonbinding recorded, never optimized. Decision CPU measured actual Python (rearm 2.2 to 9.7 us, receiver 39 to 67 us), charged into availability; warm backlog 0.92 to 1.01 s never zeroed.

## Verdict

P3T-PASS mechanism, P3T-HOLD adoption: rearm is retained as RESEARCH candidate only, never adopted. Reasons recorded separately from no-regression: R2 absolute harm vs none (2 pure-A false plus 1 pure UNKNOWN), cascade first-win limit on combined object, and unobserved actual ASR safety on all guards (proxy direction only). BC1 oracle missing causal map satisfies nothing about ASR safety. T1 proxy improvement is not an independent actual ASR positive; NP2 preservation is not replication. No Phase B classifier trigger; no old H recovery. Phase A 4Q unchanged except Q2 gap now explained by conversion.
