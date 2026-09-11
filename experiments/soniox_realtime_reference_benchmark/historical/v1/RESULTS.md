# Soniox realtime reference benchmark — RESULTS

Freeze `psem.soniox_rt_reference_benchmark.freeze.v1` (`FREEZE.json`,
frozen 2026-09-10T04:52:30Z before first connect).
Baseline `experiment-v2-speaker-change-turn-boundaries-ls` HEAD
`83aaed984b8b245082f3ffe7bb15d71f3242361f`, upstream `0/0`.
Prior dirs read-only, untouched. No Git mutations, no code comments,
no publish, no training, no threshold or rule change.

## Execution

8 sessions, 8 attempts, 0 failures, 0 retries. Fixed order NATURAL then
MANUAL per case: NP1, NP2, NP3, EN2009d. Attempt journal
`captures/attempt_journal.jsonl` holds 8 attempt-journaled + 8 opened +
8 completed events, written before connect per session.
Audio per profile 80.22s (3x10s NP + 50.22s EN2009d); both profiles
160.44s audio. Post-EOS drain bounded 15s with 3s quiesce; all sessions
exited on quiesce, never on the cap. Config per freeze:
`stt-rt-v5`, `enable_speaker_diarization=true`,
`enable_endpoint_detection=false`, `language_hints=["en"]`,
no language identification, no translation, no context.
Server revision advertised in no response: recorded `unknown` per freeze.
Key `SONIOX_API_KEY` process-only, 64 chars, never logged, redacted in
captures (`config_sent_redacted`).

## Capture integrity

Every raw token preserved all public fields: text, start_ms, end_ms,
is_final, language, confidence, speaker. Speaker present on 100% of
tokens in all 8 sessions (5855/5855 EN2009d-MANUAL down to 395/395
NP2-MANUAL). `dropped_field_names` empty in all 8: the server sent no
other public fields. `fin_end` markers: 0 in all NATURAL, 1 in all
MANUAL (finalize ack), as choreographed. Zero final-revision protocol
violations, zero ignored out-of-order batches, zero unknown-speaker
words, zero mixed-speaker words in all 8 sessions.
EOS split: all committed finals arrived pre-EOS (during the 3s trailing
silence); post-EOS finals 0 everywhere; never-final words 0 everywhere.
Response before EOS evaluated separately from EOS-assisted finals per
freeze; with zero post-EOS finals the distinction is recorded moot.

## Intrinsic recognition vs human GT (matched / GT denominator)

NP frozen cohorts reproduce prior splits exactly (region pad 32000,
same helper): NP1 8/0/0, NP2 6/1/1, NP3 8/0/0 over the same 8 GT word IDs.

| session | cohort | full-payload | speaker on matched (retro map) |
|---|---|---|---|
| NP1-NATURAL | 3/8 (un 1117,1118,1120; mix 1157,1119) | 9/29 | 3/3 map {1:L} single-side only |
| NP1-MANUAL | 8/8 | 25/29 | 8/8 map {1:L,2:R} |
| NP2-NATURAL | 2/8 (un 1654,1478,1479,1480,1481; mix 1655) | 7/23 | 2/2 map {2:L} single-side only |
| NP2-MANUAL | 5/8 (un 1654,1478; mix 1479) | 13/23 | 5/5 map {2:L,3:R} |
| NP3-NATURAL | 4/8 (un 322,323,325; mix 324) | 13/22 | 4/4 map {1:L} single-side only |
| NP3-MANUAL | 8/8 | 20/22 | 8/8 map {1:L,2:R} |
| EN2009d-NATURAL | COMBINED 9/28, R2 6/9, T1 3/21 | — | 9/9, 6/6, 3/3 map {1:B,2:A} |
| EN2009d-MANUAL | COMBINED 19/28, R2 6/9, T1 13/21 | — | 19/19, 13/13, 6/6 map {1:B,2:A} |

EN2009d GT overlap class: R2 9/9 overlap, T1 17 overlap + 4 nonoverlap,
COMBINED 24 + 4. No DER claim from sparse word coverage; overlap is a
GT-span proxy, not physical VAD. Speaker accuracy is conditional on
recognition under a retrospective per-session permutation (optimistic,
disclosed); NATURAL single-label sessions cannot assess turn
discrimination. Unmapped/unknown counted, none occurred.

## Frame-causal deadlines and latency (wall origin: audio first send)

Provisional availability at fixed deadlines after each GT word end
(matched-word denominators above):

- NP1-MANUAL cohort: 1/8 at 200/500ms, 2/8 at 1/2s. NP2-MANUAL: 1,2,3,3
  of 5. NP3-MANUAL: 2,2,3,3 of 8. EN2009d-MANUAL COMBINED: 5,5,8,9 of 19.
- First-provision medians 0.45–0.78s across sessions (small N, counts in
  ledger). One negative first-prov (-0.425s, EN2009d both profiles,
  disclosed ambiguous near-time alignment, not zero-latency credit).
- Final-receipt medians: MANUAL 2.7–4.7s, NATURAL 5.1–5.7s (trailing
  silence + drain; EOS-assisted finals reported, no future benefit at
  deadline). p95/min/max per session in `ledger.json` per-word rows.

## Conclusions (evidence only, no forced success)

1. MANUAL finalize profile strictly dominates NATURAL on recognition in
   this sample (cohort 21/24 vs 9/24; full 58/74 vs 29/74) under
   identical audio: omitting finalize leaves the transcript partially
   flushed. Order confound fixed NATURAL-first is reported, not
   corrected; no repeatability claim.
2. Best observed RT profile in this sample is MANUAL, not a theoretical
   upper bound; async not run.
3. Where words were recognized, the server speaker labels aligned with
   GT sides/roles under retrospective mapping (52/52 MANUAL matched
   pairs across all scopes), but single-side NATURAL sessions and the
   optimistic map forbid any turn-accuracy claim beyond the matched set.
4. No protocol violations, no missing attempts, no hidden failures: all
   results good and bad are in `ledger.json` with per-word rows.
