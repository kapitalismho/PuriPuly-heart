# Stage2 Repeatability Manifest (ownership probe; timing/ owned by sibling)

State: FROZEN.

- Freeze `psem.repeatability.stage2.freeze.v1` in `FREEZE.json`
  (sha256 `6b38e0f3f9eabe72327546deb98b905749a49b2939dc0e0c235af6935e4ecf61`),
  recorded BEFORE any capture/scoring. Erratum: first-written hash
  `c685bd08...` carried a one-digit transcription drop in NP2's payload
  sha (63 chars); corrected pre-capture to the true file hash after the
  payload guard refused the NP2 session with zero budget consumed.
  No reviewed claim yet.
## Authority

- Parent/governing: revised #132 PSEM program; #142 definition preserved.
- Continuation: LOCAL user authorization in-chat (bounded stage2
  investigation), 2026-09-09T08:59:42Z actual clock. NOT a remote #132
  amendment. Original P2 2-capture budget NOT reused (spent 2/2).
- Baseline `experiment-v2-speaker-change-turn-boundaries-ls` HEAD
  `83aaed984b8b245082f3ffe7bb15d71f3242361f`.
- New finite budget: exactly THREE provider sessions (NP1/NP2/NP3, <=7s
  each). No replacements/retries beyond pre-connect zero-session failures.
  Zero translation/LLM calls.

## Outcome map

- Owner A (this deliverable): everything here EXCEPT `timing/`.
- Owner B (sibling `repeatability-timing-owner`, exclusive): `timing/` ONLY.
- Barrier: both receipts land, Director sends integration message, only then
  is `timing/results.json` consumed. Preliminary result: timing PENDING.

## Scope (GT-driven, mechanism examples, not sampling)

- NP1 `ami_ES2009c` B(MEE034)->A(MEE033), b=19805040, episode A00104.
- NP2 `ami_ES2009d` B(MEE034)->C(MEE035), b=33469760, episode A00271.
- NP3 `ami_ES2002b` D(MEE008)->B(FEE005), b=2609520, episode A00006
  (different cohort; required for the gate diversity leg).
- Scored set: frozen 8-word GT window per case (4 left + 4 right), all 8
  scored, never a favorable subset. Full intervals, episodes, payload hashes,
  and native-join support in `FREEZE.json`.
- Guards: retained P2 R1/R2/T1 (read-only, no new captures); two pure-A
  singleton word proxies (labeled NOT ASR); B+C reference-scope guard is
  UNFULFILLED-real (both GT candidates rejected pre-score, reasons frozen).
- Salvage: existing P2 G03/G04 streams with word GT where unambiguous
  (reported separately); historical P4 recomputed for reference only.

## Method

- Evidence: `load_validated_export` + `frontier_sweep.simulate_episode`
  (F0 tau 0.5; H7301 tau 0.5887844788775033; 1600-sample confirmation).
  First event per FULL episode, then scored-span window filter. No rearming,
  no threshold tuning. GT-derived references never called causal enrollment.
- Text: one captured Soniox stream per case (stt-rt-v5, chunk 512, realtime
  32ms pacing, trailing 100ms, one finalize), shared by ALL arms. Actual
  websocket send completion instrumented experiment-locally (schedule /
  enqueue / flush walls per chunk); seal = accepted-final arrival, same clock.
- Word rule: end <= boundary stays left else right; straddlers intact and
  uncertain; +-1280 sensitivity, never optimized.
- Receiver: fake Audio research-only; receipts applied / already_separated /
  too_late / unsupported / invalid_scope; conservation exact and separate
  from accuracy; unknown exposure + guard harms tracked; no reference
  invention.
- Comparators per case: no-PSEM, F0, H7301, correct-transition control
  (annotation boundary b UNCHANGED; +100ms confirmation charged to source
  support/availability only), #98 Simple Anchor (historical, non-causal).

## Executable replay

```
./.venv/Scripts/python.exe experiments/psem_repeatability_stage2/capture.py --case NP1
./.venv/Scripts/python.exe experiments/psem_repeatability_stage2/replay.py --run
./.venv/Scripts/python.exe experiments/psem_repeatability_stage2/replay.py --smoke
```

`--run` writes `LEDGER.json` and `receipt.json` (hash-bound). `--smoke`
runs the synthetic receiver contract only (labeled synthetic, never
empirical) plus conservation DROP/DUP controls.

## Constraints / non-goals

No production edits, Git mutations, publish, training, sweeps, new datasets,
provider replacement, translation/LLM calls on the ownership probe,
reference lifecycle changes, threshold/lifecycle changes, success selection,
or capture expansion to force a positive. New model inference inside
`--run`/`--smoke`: none (frozen NPZ evidence + captures only). Sibling
`timing/` ran bounded F0 compute under its own authority (non-parity;
consumed read-only at barrier). Unknown never becomes PASS/zero-cost. Never
substitute synthetic head latency for H actual compute timing. Timing
snapshot consumed only after the barrier message.
