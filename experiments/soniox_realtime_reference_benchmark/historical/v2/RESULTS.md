# Soniox realtime reference benchmark v2 — RESULTS

Freeze `psem.soniox_rt_reference_benchmark.freeze.v2` (`FREEZE.json`,
frozen 2026-09-10T05:30:00Z before any v2 connect). Baseline
`experiment-v2-speaker-change-turn-boundaries-ls` HEAD `83aaed9`,
upstream `0/0`. v1 bytes preserved under `historical/v1/` (hashes in
v1 MANIFEST, re-verified). Prior dirs read-only. No Git mutations, no
code comments, no publish, no training, no tolerance tuning
(TIME_SUPPORT_TOL 48000 unchanged), no PSEM change, no Soniox teacher use.

## Protocol outcome: 8/8 INCOMPLETE on finished, 0 failures, 0 retries

Physical empty-binary EOS was sent exactly once per session (send
start/end walls in captures; offline smoke proved single-send). The
server never returned top-level `finished:true` within the 15s bound in
any of the 8 sessions, so every session is recorded INCOMPLETE per the
frozen rule; quiesce was never treated as success. Observed server
behavior post-EOS: session stays open, answers keepalives with further
hypothesis messages, never finishes, never closes. Empty-binary is
therefore NOT an accepted EOS signal on this endpoint/config; the true
end-of-stream signal is unknown (close-handshake-final is the open
candidate, needs new authorization, not attempted). Token evidence is
complete and scored as-is; finished-dependent claims are absent.

Manual finalize (MANUAL) was sent once per MANUAL session only after the
200ms pre-trail audio end time was reached in TIME (causal discipline,
waits 0.00–0.03s recorded). Config identical all sessions: stt-rt-v5,
diarization on, endpoint detection off, hints en. Server revision never
advertised: `unknown`. Key process-only redacted.

## Capture integrity

All raw tokens keep full public fields incl speaker; speaker present on
100% of tokens all sessions; dropped fields empty x8; `fin_end` 0
NATURAL / 1 MANUAL. Delta discipline: finals appended once keyed on full
(start,end,text,speaker) occurrence order; exact repeats 0 everywhere;
same-end different words coexist; nothing retracted on absence; zero
violations. Unknown-speaker words 0, mixed-speaker words 0.

## Recognition vs human GT (strict recall + WER S/D/I)

NP frozen cohorts reproduce prior splits (8/0/0, 6/1/1, 8/0/0).
WER ref = fixed full-payload non-punct GT in time order, hyp = committed
words ending in scope; overlaps serialized in time order (bounds WER
meaning, disclosed). Truncated edge words counted once, reported.

| session | cohort rec | full rec | full WER (S/D/I/N) | trunc |
|---|---|---|---|---|
| NP1-NAT | 2/8 | 9/29 0.31 | 0.6897 (1/19/0/29) | 1142, A1134 |
| NP1-MAN | 7/8 | 25/29 0.86 | 0.1379 (2/2/0/29) | same |
| NP2-NAT | 1/8 | 7/23 0.30 | 0.6957 (2/14/0/23) | A2438 |
| NP2-MAN | 4/8 | 13/23 0.57 | 0.4783 (9/1/1/23) | same |
| NP3-NAT | 1/8 | 13/22 0.59 | 0.4091 (0/9/0/22) | D23 |
| NP3-MAN | 4/8 | 20/22 0.91 | 0.0909 (0/2/0/22) | same |
| EN2009d-NAT | CMB 9/28 R2 6/9 T1 3/21 | — | CMB 0.7143 (0/19/1) R2 0.4444 T1 0.8571 | A124,B41;B46,A128 |
| EN2009d-MAN | CMB 19/28 R2 6/9 T1 13/21 | — | CMB 0.4286 (2/7/3) R2 0.4444 T1 0.4762 | same |

EN2009d overlap class: R2 9/9 overlap, T1 17+4, COMBINED 24+4 (span
proxy, no DER claim). Whole-session denom EN2009d: 171 GT words total,
28/26 unique scored-scope words; R2/T1/COMBINED rows overlap, never summed.

## Speaker (one per-session map, pure A-D roles, retrospective)

Session Hungarian over full-payload (NP) / COMBINED (EN2009d) matched
pairs, applied consistently to all scopes, never scope-independent.
NP cohort binary L/R kept separate explicit (not diarization):
MAN 7/7, 4/4, 4/4; NAT single-side only.
EN2009d both profiles {1:B,2:A}: NAT 9/9, MAN 19/19 on COMBINED matched.
Miss/unresolved reported via recall denominators; punc excluded only.

## Frame-causal occurrence deadlines + latency (origin audio first send)

Availability = target occurrence aligns in the latest snapshot at or
before the deadline (sequence prefix + run attribution, never bag,
never final text). NP full: 200ms 0 everywhere; 500ms 0–1; 1s 4–7;
2s 4–7 of matched sets. First-prov medians NP 0.52–0.77s, zero
negatives. Final-receipt medians MAN 3.0–4.7s, NAT 5.1–5.7s
(trailing silence + drain). All finals arrived pre-EOS (verified per
message: post-EOS traffic is exactly one keepalive-elicited
provisional-only hypothesis in each NATURAL session, none in MANUAL);
EOS-assisted finals contributed nothing.
EN2009d: one flagged early alignment (A.words124, truncated edge word,
first-prov 7.78s vs GT end ~42.5s: repeated chatter run ambiguity,
flagged time-alignment, not prediction proof; provider token-position
vs word-timestamp error acknowledged). Otherwise first-prov 0.65–2.8s.

## Conclusions (bounded, no forced success)

1. Finished-protocol finding is the headline: empty-binary EOS elicits
   no `finished:true`; all 8 INCOMPLETE with full token evidence.
2. MANUAL vs NATURAL numeric gaps (recall, WER) are reported without a
   dominance verdict: fixed NATURAL-first order confound plus
   unfinished protocol forbid it; best observed complete profile is not
   an upper bound; async not run.
3. Where words were recognized, server speaker labels map onto GT
   roles/sides under the disclosed retrospective map, with the NP2-MAN
   minority-role failure mode visible. No turn-accuracy claim beyond
   matched sets; no DER.
4. Withdrawn v1 claims stand withdrawn (dominance, coarse-band,
   merge-cut revisions, bag availability, per-scope maps, 52-sum).
