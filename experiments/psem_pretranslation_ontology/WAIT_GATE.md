# WAIT_GATE (independent read-only gate; no waiting policy adopted)

Scope: this gate is computed inside `experiments/psem_pretranslation_ontology/probe.py` from the newly generated SIMPLE and RICH full applicable event lists. It mechanically shows event-set availability counts unchanged under [0,100,300 ms] extra wait. No new measurement claim, no stacked waiting-policy hypothesis, no ordinary point-var search; uniform predeclared raw rules only (avail<=terminal+w, in-pay bounds, same schedule/validity both ontologies). User flexible wait preserved; BASELINE ONLY cutoffs, not final policy.

## Cutoffs and coverage

- NP C9 actual `<fin>` cutoffs BASELINE ONLY: 7.282 (NP1) / 7.281 (NP2) / 7.297 (NP3). No extra wait.
- Finite observation coverage: native decode range covers episode+payload; extended check payhi+20000 run for both ontologies (see below).
- Guards synthetic endcap SAME computed prior source0 availability (object-end zero grace, shared all arms); actual end-to-end timing UNKNOWN, no production adoption.

## NP counts ([0,100,300 ms] extra wait; applicable = in-pay boundary AND avail<=terminal+w)

- NP1 RICH actualF0: plus0 1 (b19808000 avail 5.265) / plus100 1 / plus300 1. SIMPLE actualF0: 1/1/1 (same boundary, avail 5.265). GT arms: 1/1/1 both ontologies. Margin 2.017 s before terminal.
- NP2 RICH actualF0: 1/1/1 (b33472000 avail 5.184). SIMPLE actualF0: 1/1/1. GT arms: 1/1/1 both. Margin 2.097 s.
- NP3 RICH actualF0: 1/1/1 (b2611200 avail 4.857). SIMPLE actualF0: 1/1/1. GT arms: 1/1/1 both. Margin 2.440 s.
- Verdict NP: zero additional applicable events at +100/+300 ms for EITHER ontology OR arm. No wait advantage observed within the available pre-terminal record. No waiting experiment was performed in this outcome; user waiting policy stays flexible.

## Extended decode (payhi+20000, same validity/schedule)

- NP1: ext in-pay RICH 1 / SIMPLE 1; new distinct RICH [] / SIMPLE [].
- NP2: ext in-pay RICH 1 / SIMPLE 1; new distinct [] / [].
- NP3: ext in-pay RICH 1 / SIMPLE 1; new distinct [] / [].
- Verdict: no pending distinct late boundary appears within +20000 samples for either ontology. Missing future coverage is NOT treated as no-info EVER; future later model/new stream not addressed.

## Source0 schedule (PhaseA queue actual observed service charged; NOT extended)

- NP1 last supported 19846496 finish 6.7136 (< 7.282 terminal); raw max support 19872096 / raw extent 19872000 / prefix end 19872000. Raw prefix extra ~1.14 s beyond payhi exists but release None for S>=payhi: NOT SCHEDULED, not usable evidence, not charged 0.
- NP2 last supported 33516896 finish 7.1066 (< 7.281); raw max 33536096 / extent 33536000 / prefix 33536000. Same raw-extra-not-usable rule.
- NP3 last supported 2658656 finish 6.6966 (< 7.297); raw max 2684256 / extent 2684160 / prefix 2684160. Same rule.
- Verdict: last usable evidence finishes before terminal on all NP; queue NOT extended with future PCM unsent; no false wait matrices performed. Condition for a WAIT experiment (a withheld applicable event that extra wait would admit) is NOT MET for available actual streams under either ontology.

## Guard counts (synthetic terminal; capacity diagnostic only)

- R1: RICH applicable 2/2/2 (3187200/3191040) vs SIMPLE 0/0/0 across +0/+100/+300. Unchanged within ontology.
- R2: RICH 1/1/1 (672000) vs SIMPLE 1/1/1. Unchanged.
- T1 in-span: RICH 3/3/3 (702720/712960/733440) vs SIMPLE 2/2/2 (702720/733440). Unchanged.
- COMBINED in-span: RICH 4/4/4 vs SIMPLE 3/3/3. Unchanged.
- BC1: no terminal (unmapped), no counts. Singles: 0/0/0 both.
- Verdict guards: extra wait adds zero applicable events within either ontology, observed within the available pre-terminal record only. Overlap-bound differences across ontologies are static semantic (UNRESOLVED vs CURRENT); no wait advantage was observed for them in this record, which is not a generic claim that waiting could never matter.

## Tests in executable

- `applicable_inpay` with extra_s 0/0.1/0.3 per ontology/arm recorded in `ledger.json` `wait_counts`.
- Extended `decode_state_events` on payhi+20000 frames per ontology; new-boundary diff recorded in `extended`.
- `schedule_last_supported` from same schedule arrays (release/finish None handling); raw-max/prefix recorded, never extended.
- Smoke `same-evidence-availability-loop-causal` passes (avail<=terminal enforced by shared `r2_partition` engine).

## Future protocol (only if later measuring actual waiting)

If a later stream/model shows a pending confirmed event with terminal < avail <= terminal+budget AND in-pay AND valid, a bounded wait experiment may be frozen separately with: predeclared budget, same paired RICH/SIMPLE counting, source0 queue extension only from actually sent PCM (never future unsent, never charged 0), and delivery-deadline harm accounting. This outcome does NOT authorize it; no such event exists in current record.
