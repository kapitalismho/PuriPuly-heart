# Phase A Headroom — Live Result (Ledger Owner, Post-Barrier, v1.2 Corrected)

Freeze `psem.phase_a.headroom.freeze.v1.2` (clarification before comparison, actual UTC 2026-09-09T20:33:31Z; no retune). Live consumer `live_headroom.py` reuses the existing decision-sufficiency alignment helper with the frozen ±32000 search regions (no second implementation) on frozen real observations (`OBSERVATIONS.json` sha `3ec7c03c...90b5aa`, bundle revalidated). Ledger `causal_ownership_ledger.json` written. Integration checks 15/15 PASS.

## Ownership on fixed cohorts (definite plus pessimistic, same ASR, valid monotonic contract)

- NP1 ES2009c A00104 b=19805040, cohort 8 matched 0 unmatched 0 mixed:
  none [4,4]; f0_original early fire boundary 19682560 emit 19700576 scope invalid_scope pre-payload with no applied boundary [4,4]; f0_latch_alt second fire 19808000 applied [0,0]; control b applied availability 4.774 deadline 7.282 mpl +2.508 legal-measured-virtual-profile [1,1] with one false-moved left `it is` (words1157): boundary overshoot 320 samples, ASR-GT timestamp difference 3280 samples (ASR end 19805360 vs GT end 19802080). The two deltas are recorded separately, never conflated.
- NP2 ES2009d A00271 b=33469760, cohort 6 matched 1 unmatched 1 mixed:
  none [2,2]; f0_original applied 33472000 availability 5.184 mpl +2.097 [0,0]; control b applied availability 5.184 mpl +2.097 [1,1] with one false-moved left `have` (words1655): boundary overshoot 320 samples, end-GT difference 1920 samples (ASR end 33470080 vs GT end 33468160).
- NP3 ES2002b A00006 b=2609520, cohort 8 matched 0 unmatched 0 mixed:
  none [4,4]; f0_original applied 2611200 availability 4.857 mpl +2.440 [0,1] (`Well` straddles the late boundary); control b applied availability 4.367 mpl +2.930 [0,0].

Full-payload recognition (unmatched kept distinct from accepted omission and duplication): NP1 29 GT words 25 matched 1 unmatched 3 mixed with right exposure `V_R_`, `do` (role A, UNKNOWN never dropped); NP2 23 GT words 20 matched 1 unmatched 2 mixed with right exposure `There`, `is` (role C); NP3 22 GT words 20 matched 1 unmatched 1 mixed with right exposure `I am`, `um` (role B). Support from exact old-grid join matches frozen metadata on all three cases. Conservation exact on every stream with throwaway DROP, DUP, and TEXT controls all detected; per-arm accepted-text partitions output actual left and right texts, never echo input. Binding arms issue one request each with zero unnecessary fragments; ALT applies once per case with zero redundant already-separated.

## Timing on the single unified schedule

One schedule per case from source 0 including init model schedule (load plus sched, diagnostic mel_full excluded) and chunk 0 with natural catchup: pre-payload raw release on session-relative source clock, inside-payload raw support to capture FLUSH end walls, finish equals max of release and prior finish plus actual service. Control and F0 share the containing chunk plus measured decision and receiver overhead. Raw support frontiers recorded actual, never frame starts. Terminal frames whose chunk support exceeds prefix end are invalid (artificial pad); no selected event uses them. Computed warm backlog at capture entry 0.92 to 1.01 s, never zeroed. Reference-ready session times precede every accepted availability. Branch class for all applied events is legal measured virtual profile under scoped benchmark assumptions, never production live proof; future head latency stays UNKNOWN separately, never a blanket model-lag claim. Added latency is measured CPU only (sub-millisecond), kept separate from headroom margin.

## Reference mapping from actual reference episodes

Meeting mapping authoritative: EN2009d anchor A is FEE083 from actual A00003 epoch start 168000 (slot 1, ready sample 170240, 350 support frames) carried into R2 and T1 with no re-enrollment inside overlap; ES2009a anchor A is MEE033 from actual A00018 beginning (slot 3, ready 3104000, 46 support) carried into R1; positives map from their own episode beginnings under the same fixed 100 ms rule (NP1 slot 2, NP2 slot 2, NP3 slot 1). BC1 primary genuine A00047 preroll from 9091552 has no anchor-only support, so the carried slot 3 measures as oracle diagnostic only (non-binding, no pure new reference invented, last 576 samples UNKNOWN retained exact).

## Guards with actual proxy partitions

R1 carried slot 3, no in-span fire: zero-intervention exposure within proxy, no safety PASS claimed. R2 carried slot 1, one in-span event boundary 672000: proxy partition 3 confident wrong-owner plus 2 uncertain with 0 fragmentation; the pattern-A leg is word-mixed at GT level, a separable reference observation, proxy-only. T1 carried slot 1, one in-span event boundary 733440 in the B leg: proxy partition 6 wrong plus 1 uncertain, takeover direction supported, proxy-only with actual ASR quality UNKNOWN (supported direction is not validated zero ASR harm). BC1 oracle diagnostic: no in-span events, zero intervention, non-binding. Singletons ES2009c and ES2009d mapped with no fires and zero confident wrong-owner words, proxy-only. BC2 and BC3 stay excluded per frozen prior record, never fabricated. QA merge: 157, 251, and 186 adjacency pairs fully resolved against word times with zero grounded in the scored windows, reported null with counts as not applicable. Salvage appended nonbinding only: G03 18 groups 38 tokens, G04 17 groups 42 tokens, P4 historical 53 corrected tokens, no fresh inference.

## Four answers (evaluated after repairs on actual results)

- Q1 headroom YES as scoped benchmark: perfect-boundary control strictly reduces wrong-owner versus none on all three positives (4 to 1, 2 to 1, 4 to 0) with legal measured timing. This is capacity under the GT reference assumption, not a net-improvement claim over F0 everywhere.
- Q2 F0 gap: NP1 single-fire conversion failure (early fire consumed pre-payload, reproduces the old 19683200 diagnostic); NP2 no gap; NP3 one straddle uncertainty from a 1680-sample-late boundary. Control is not a net improvement on all: NP2 F0 beats control, NP3 ties definite.
- Q3 no learned-head necessity demonstrated on this scope: the NP1 gap is closed better by the same fresh F0 latch ([0,0] versus control [1,1]), so conversion explains it and no evidence needs a classifier.
- Q4 Phase B not triggered now: Gate A fails as a learned-observation claim (control does not beat F0 on two positives), which blocks auto-B. The head itself is not judged worthless; the finding is scoped to this evidence.

## Limits

Model and event lag beyond the measured schedule stay unmeasured, so receipts are benchmark-conditional. Dominant residuals are ASR word-time skew and unmatched-word exposure. R2 and BC1 reference limits are separable P3R, proxy-only. About twelve mechanism examples only, never statistically representative.

## Compute overheads (fresh F0 run, loaded local, no H restore)

Observed offline service totals 1426.0 s over 68.8 min audio with frontend 14.42 s; init load plus sched charged per source; diagnostic mel_full excluded. Live consumer wall under 8 s local.

## Reproduce

```powershell
.venv/Scripts/python.exe experiments/psem_phase_a_headroom/live_headroom.py smoke
.venv/Scripts/python.exe experiments/psem_phase_a_headroom/live_headroom.py run
```

No training, no H restore, no new captures, no paid calls, no Git mutations, no production adoption.
