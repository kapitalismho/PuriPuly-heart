# Soniox realtime reference benchmark v3 — RESULTS (active, corrected)

Only v3 rows are active. v1/v2 snapshots under `historical/` are
withdrawn method records. Baseline HEAD `83aaed9`, upstream `0/0`.
No Git mutations, comments, publish, training, tolerance tuning
(TIME_SUPPORT 48000 predeclared), PSEM change, or teacher use.

## 0. Timestamp correction (review finding 1)

The embedded `FREEZE.json` `frozen_at_utc` (`2026-09-10T06:15:00Z`) is a
FALSE future string, withdrawn as provenance; no preregistration-time
claim is made. Readonly evidence: freeze file mtime
2026-09-10T05:18:03Z, first v3 connect (journal attempt-journaled,
NP1-NATURAL) 2026-09-10T05:19:20Z, all 8 connects 05:19:20–05:32:30Z.
The file was written before the first call; only the embedded string
is false. No backdating performed (`FREEZE.json` bytes unchanged);
correction stamped 2026-09-10T05:39:40Z in `ledger.json`
(`provenance_correction`, generated from mtime+journal at score time).

## 1. Protocol: 8/8 COMPLETE with finished:true, 0 failures

TEXT empty-string EOS sent once per session, no post-EOS frames.
Server `finished:true` all 8 (waits 0.2–0.3s), normal close. Binary EOS
(v2, 8 stalled) vs TEXT EOS (v3, 8 complete) is compat data, not a
violation claim. Config identical: stt-rt-v5, diarization on, endpoint
detection off, hints en; revision unknown; key redacted.
Capture integrity: full public fields incl speaker on 100% of tokens,
dropped fields empty x8, fin_end 0 NATURAL / 1 MANUAL; delta finals
appended once on full (start,end,text,speaker) occurrence order, exact
dups 0 (EN-MAN 1 post-EOS exact-dup token absorbed), never retracted,
0 violations; unknown/mixed-speaker words 0.

## 2. EOS truth: NATURAL recall is END-OF-STREAM-ASSISTED (finding 2)

Machine split, same clock (arrival_wall from audio first send):

| session | emitted finals pre/post | unique words pre/post/never |
|---|---|---|
| NP1-NAT | 20 / 28 | 10 / 17 / 0 |
| NP1-MAN | 49 / 0 | 27 / 0 / 0 |
| NP2-NAT | 11 / 18 | 9 / 12 / 0 |
| NP2-MAN | 31 / 0 | 21 / 0 / 0 |
| NP3-NAT | 20 / 13 | 13 / 7 / 0 |
| NP3-MAN | 34 / 0 | 20 / 0 / 0 |
| EN2009d-NAT | 231 / 21 | 120 / 13 / 0 |
| EN2009d-MAN | 253 / 1 | 133 / 0 / 0 |

Pre-EOS-final-only vs complete-final recognition, same human GT:

| scope | NP1 N/M | NP2 N/M | NP3 N/M | CMB N/M | R2 | T1 N/M |
|---|---|---|---|---|---|---|
| full recall pre | .31/.86 | .30/.65 | .59/.91 | .32/.68 | .67/.67 | .14/.62 |
| full recall complete | .83/.86 | .57/.65 | .91/.91 | .68/.68 | .67/.67 | .62/.62 |
| cohort recall pre | .375/1.0 | .25/.625 | .5/1.0 | — | — | — |
| cohort recall complete | 1.0/1.0 | .625/.625 | 1.0/1.0 | — | — | — |

MANUAL needs no EOS tail (pre==complete); NATURAL depends on it
(NP1 0.31→0.83, T1 0.14→0.62). Deadline prefixes exclude EOS-future
(snapshots capped at min(deadline, EOS)). The latency tradeoff is
exactly this gap: pre-EOS final-only recall at recorded cutoff is the
pre column, not the complete column.

## 3. Corrected recognition (finding 3): whole-payload hyp, frozen IDs

Cohorts use whole-payload hyp aligned once, projected onto the
unchanged frozen 8 IDs. Prior 7/8 / 4/8 / 4/8 reports were REGION
EXCLUSION (scored-span crop dropped hyp `go`/`idea`-class words), not
recognition loss. Correct fixed-ID recall: NP1 8/8 both, NP2 5/8 both
(un 1654/1478, 1479 mixed-or-un), NP3 8/8 both. PSEM comparison needs
no identical hyp-region rule where it would exclude required words;
GT counts and IDs unchanged.
Full-payload recall / Levenshtein WER (proper DP, unit costs, stable
diagonal-deletion-insertion ties): NP1 .83/.86 0.1724/0.1379; NP2
.57/.65 0.4783/0.3478; NP3 .91/.91 0.0909/0.0909; CMB .68/.68 0.3929;
R2 .67/.67 0.4444; T1 .62/.62 0.4286. Cohort-8 WER is I-loaded by
design (8-word ref vs payload hyp) and cited for method honesty only.
Truncated edge words counted once: NP1 1142/A1134, NP2 A2438, NP3 D23,
EN A124/B41/B46/A128. EN overlap R2 9/9, T1 17+4, CMB 24+4 (span
proxy, no DER). Whole-session denom 171 GT, 26 unique scored words;
scope rows overlap, never summed. Profiles converge; no dominance
verdict (order confound); best observed is not an upper bound.

## 4. Speaker, conditional on matched only (no overall diarization)

Session Hungarian, pure A-D roles, one map per session: NP1
{1:B,2:A} 24/24 + 25/25 (pure IDs good); NP3 {1:D,2:B} 20/20 both
(pure IDs good); NP2-NAT {1:D,2:B,3:C} 10/13, NP2-MAN {1:D,2:A,3:C}
11/15 — extra-C errors counted wrong in FULL A-D as required.
Cohort binary L/R (matched-subset only): 8/8, 5/5, 8/8. EN2009d
{1:B,2:A} 19/19 both. Retrospective optimistic, disclosed.

## 5. Latency: occurrence table, negatives never credited (finding 3)

One generic full-list monotone occurrence alignment (full GT list vs
cumulative snapshots before deadline, existing difflib pattern) plus
predeclared 3.0s admissibility; ambiguous pairs unknown; scoped IDs
projected. D35-class false matches rejected (smoke-proved on actual
NP3 data: old 0.83s wrong occurrence rejected; later admissible
occurrence at 3.406 evaluated normally, avail 1.0/2.0s).
First-prov credited medians: NP full 0.58–0.86s; EN CMB 0.86s both,
R2 1.38–1.39s, T1 0.78–0.80s; all credited minima positive (0.39–1.07).
Flagged-not-credited negatives: NP1 full 3 (1126/1129/1134), EN CMB/R2
1 (B.words42); cohort negatives 0.
Final-receipt latency is measured from aligned GT word end to received
final word: NATURAL medians 5.0–5.7s include EOS-assisted finalization;
MANUAL medians 2.7–4.7s occur before EOS. These measurements include
the configured waiting/finalization policy and network/service time;
they are not intrinsic model latency. NATURAL has post-EOS final
tokens; MANUAL has no newly finalized words after EOS (EN has one
duplicate final token). Finished control frames are counted separately.

## 6. Conclusions

TEXT EOS completes 8/8; NATURAL pre-EOS final-only recall at recorded
cutoff trails its EOS-assisted totals by the measured gaps;
completed-protocol rows above are the only active results. No ranking
vs proxy rows (R2 [3,5]/[4,7], T1 [6,7], CMB [12,14]/[7,12]/[9,13],
NP1 P3T 4->[0,0] cited parallel only). No async, no DER, no upper
bound. Scope done.
