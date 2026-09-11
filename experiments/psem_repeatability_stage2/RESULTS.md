# Stage2 Results & Decision — INTEGRATED (barrier consumed)

Freeze `psem.repeatability.stage2.freeze.v1`
(`FREEZE.json` sha256 `6b38e0f3f9eabe72327546deb98b905749a49b2939dc0e0c235af6935e4ecf61`;
chronology: first-written `c685bd08…` governed the NP1 capture only
(journal 09:03:12–09:03:24Z); corrected hash governs NP2/NP3 (journal
09:04:45Z onward); NP1 payload bytes identical in both, never re-captured).
Replay: `./.venv/Scripts/python.exe experiments/psem_repeatability_stage2/replay.py --run`
(deterministic: consecutive runs identical ledger/receipt hashes).
Ledger `LEDGER.json`, receipt `receipt.json` (binds root + timing snapshot
`e5c88982…`, read-only), smoke `smoke.json` (12/12 PASS).

## Gate rollup (H strict wrong-owner reduction vs none; ownership MEASURED)
| Case | match/unmatch/mixed | none wrong | F0 | H7301 | H verdict |
| ---- | ------------------- | ---------- | -- | ----- | --------- |
| NP1 ES2009c | 8/0/0 | 4 | unsupported (fire 19683200 precedes span) | unsupported (fire 19681600 precedes span) | FAILED (measured zero) |
| NP2 ES2009d | 6/1/1 | 2 | unsupported (no event, 12-frame episode) | unsupported (no event) | FAILED (measured zero) |
| NP3 ES2002b | 8/0/0 + boundary-uncertainty H:1 | 4 | applied @2611200, interval [0,1] | applied @2611200, interval [0,1] | PASSED (pessimistic-proof) |

- Useful effect, PREDECLARED (H, threshold 2=new sessions): FAILED — 1 of 3
  new cases pass (NP3), NP1/NP2 measured-zero. Diversity leg (NP3) met, count
  leg unmet.
- NP3 supported benefit (exact truth): 3 right words definitely correct + 1
  uncertain (Well GT-straddles the applied boundary while its consumer end
  assigns the correct side; side preserved, flagged, counted pessimistically).
  Gate passes only because the strict improvement holds pessimistically
  (4→[0,1]); abstaining on words can never pass. Full per-word proof and the
  Director adjudication record: `REVIEW_RECORD.md`.
## Integrated verdict (barrier consumed; material claims)

- Ownership (matched-pessimistic): NP3 H/F0 interval [0,1] vs baseline 4 —
  supported benefit “3 correct + 1 uncertain”; NP1/NP2 measured-zero.
  Predeclared gate (threshold 2 new sessions): 1 of 3 → FAILED.
- Causal timing: actual F0 non-parity compute measured (NP1 .239 s / NP2
  .248 s / NP3 .247 s, isolated CPU, no prefix state); actual H compute
  missing (no weights; ~1.1 ms synthetic head never substituted). Causal
  applicability: UNMEASURED — not pending, not passed.
- H increment over F0: none demonstrated (NP3 identical fire at 2611200).
- Positive qualification: NP3 NEW conditional positive + P4 historical = 2
  distinct cohorts observed, but the predeclared ≥2-NEW-sessions gate is
  unmet → explicit probe completed negative; stage2 positive qualification
  NOT achieved. No stage2-accepted is invented.
- Conservation: PASS exact on all streams (not blanket-blocked: text
  measurement succeeded wherever the frozen replay reaches).
- Safety: UNMEASURED/incomplete (B+C guard unfulfilled; proxies
  semantic-only). Overall: FAILED-or-UNMET. State: IMPLEMENTATION_READY
  (not accepted).
## Control correction (integration invariant 1)

The correct-transition control boundary is the annotation transition b
UNCHANGED (NP1 19805040 / NP2 33469760 / NP3 2609520 — identical to the
anchor line). The 100 ms confirmation affects source support/availability
ONLY (frontier b+1600, max-lag accounting), never the requested boundary.
Headroom honesty: control/anchor false-move one left word on NP1 (`it's`,
provider end 205 ms past GT end) and NP2 (`have`, +120 ms) — real provider
timestamp error bounds any headroom claim.

## Unsupported ≠ unmeasured ownership (invariant 2)

NP1/NP2 evidence arms issue no request (suppression / no event), but the
frozen replay conserves their text exactly, so the ownership partition
equals baseline by measurement: improvement is MEASURED zero (FAILED), not
unknown. Unknown is the model-service timing, recorded separately per arm.

## R2 drift reconciled (invariant 3)

P2’s original `raw_first_event` callable, run read-only here on the exact
same masks/frames/scores, returns R2-A00003 F0 374400 / H 644800 and
R2-A00004 702400/702400 — identical to this ledger. P2’s reported
193600/625600 is therefore a stale report value (pre-repair run carried into
text), not a computation difference; P2 files are untouched. Verdicts agree
(0 in-span requests on R1/R2; T1 real in-span requests; no timed text on
guards → harm null-measured, uncertainty retained).

## Guards & salvage/history

- Singleton proxies (word GT, NOT ASR): ES2009c_A → A00003, ES2009d_A →
  A00005; no in-span F0/H requests → projected harm null, limits explicit.
- B+C reference-scope guard: UNFULFILLED-real → safety stays INCOMPLETE.
- G03 (matched 13 / unmatched 33 / mixed 3 over 49 payload GT words in 18
  groups; `to` ambiguous-enumerated, unscored) and G04 (matched 10 /
  unmatched 16 / mixed 0 over 26 in 17 groups) show mixed-role ownership:
  no clean split, no second positive here. P4 historical: 6/6 matched;
  wrong-owner none 4 / F0 2 / H 0 / control 1 / anchor 1 — reproduces P2.
  Reference only.

## Budget, scope, reproduce

- Provider sessions: exactly 3/3 completed, 0 failed; 0 translation/LLM on
  the ownership side. Sibling timing/ ran bounded F0 compute (non-parity)
  under its own authority — the “no inference” scope below covers this
  deliverable’s replay only. No new captures/deps in integration.
- P2 directory and sibling `timing/` preserved; timing snapshot consumed
  read-only. No commits.

```
./.venv/Scripts/python.exe experiments/psem_repeatability_stage2/replay.py --smoke
./.venv/Scripts/python.exe experiments/psem_repeatability_stage2/replay.py --run
```
