# Soniox realtime reference benchmark v3 — MANIFEST (active, corrected)

Owner dir `experiments/soniox_realtime_reference_benchmark/` only.
v1 snapshot `historical/v1/` and v2 snapshot `historical/v2/` (+
`historical/V2_HASHES.json`) preserved byte-exact, INACTIVE. All other
files read-only, raw captures and historical untouched. Baseline
`83aaed9`, upstream `0/0`. No Git mutations, comments, publish,
deploy, PSEM/production change, or teacher use.

## Timestamp correction (actual clocks, no manual strings)

`FREEZE.json` embedded `frozen_at_utc` (06:15:00Z) is WITHDRAWN as
provenance (false future string). Readonly evidence: freeze file mtime
2026-09-10T05:18:03Z, first v3 connect 05:19:20Z (journal), last
05:32:30Z; correction stamped 2026-09-10T05:39:40Z in `ledger.json`
(`provenance_correction`, mtime+journal read at score time). No
backdating; `FREEZE.json` bytes unchanged (`bd8fe7fa…1438a75`).

## Owned v3 artifacts (sha256)

- `FREEZE.json` `bd8fe7fa…1438a75` (bytes frozen pre-call; embedded
  stamp withdrawn per above)
- `probe.py` `35a2f0ba…28bba2fd8` (docstring v3 TEXT EOS; no runtime change)
- `score.py` `0ff257ae…13b22f70` (full-table occurrence, Levenshtein,
  whole-payload cohorts, EOS splits, pre-EOS views)
- `captures/attempt_journal.jsonl` `5112d814…86226a` (24 lines, 0 fails)
- `captures/NP1-NATURAL.json` `90553e30…957df1` (sentinel)
- `captures/NP1-MANUAL.json` `6b757bba…c2ef89`
- `captures/NP2-NATURAL.json` `585020b2…419b`
- `captures/NP2-MANUAL.json` `d886c92d…e4982`
- `captures/NP3-NATURAL.json` `bbab0ba6…60131`
- `captures/NP3-MANUAL.json` `4ed2b24b…aa6b`
- `captures/EN2009d-NATURAL.json` `b50b2d17…20dc57`
- `captures/EN2009d-MANUAL.json` `49456f47…f17389`
- `ledger.json` `10fcc669…491952780` (with provenance_correction)
- `RESULTS.md` `c57d62cb…1089bced94` (receipt-delay wording fix, metrics unchanged)
- `COMPARISON.md` `d1a6a7b2…eaed39`
- This `MANIFEST.md` (self-described, hashed on read)

## Budget and scope compliance

v3 8/8 attempts (total 24 with v1+v2), 0 failures, 0 retries; ~160s
audio v3 (~481s total); finished waits 0.2–0.3s; ceiling $1, est
~$0.02, billing unknown unclaimed. No new API calls in this revision
(scoring/docs only). Final edge smoke (fake transport + synthetic +
actual D35 old/new, 8 proofs) passed BEFORE docs; throwaway removed
after receipt. Only v3 claims active. Scope done, no TODO.
