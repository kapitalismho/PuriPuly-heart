# Soniox realtime reference benchmark — MANIFEST

Owner dir `experiments/soniox_realtime_reference_benchmark/` only.
All other files read-only. No Git mutations, no code comments, no
publish, no deploy. Baseline `83aaed984b8b245082f3ffe7bb15d71f3242361f`,
branch `experiment-v2-speaker-change-turn-boundaries-ls`, upstream `0/0`.

## Original read inputs (frozen, verified)

- NP1 `ES2009c.Mix-Headset.wav` [19741040,19853040] sha
  `953164ee…195e` match; NP2 `ES2009d` [33405760,33517760] sha
  `461e7f5d…f1` match; NP3 `ES2002b` [2553520,2665520] sha
  `e9d38552…26` match; EN2009d source [0,755520] sha
  `2e7226c2…97c33fd8` (this benchmark).
- `ARCHITECTURE.md` `8971048f…062d068`; provider `soniox.py`
  `686f3113…04d` (untouched, diarization via owned direct websocket);
  `replay.py` `2e08a9ec…308af` (helpers reused, not copied);
  stage2 `capture.py` `a469346b…50466f`, stage2 `FREEZE.json`
  `6b38e0f3…46deb`; return ledger `d01f07ac…32691`; P3T ledger
  `65b3cff0…695c75`.
- AMI annotations: public CC BY 4.0, archive hash per stage2 freeze;
  GT readers are existing helpers. Secret: `SONIOX_API_KEY` from
  `.env.local`, 64 chars, process-only, redacted everywhere.

## Owned artifacts (sha256)

- `FREEZE.json` `778b10a1…35f651e1a` (frozen 2026-09-10T04:52:30Z,
  before first connect)
- `probe.py` `2cb7e23b…21c44c85b` (owned direct-websocket capture)
- `score.py` `9656f5d7…0d7be308b` (owned scoring, existing helpers)
- `captures/attempt_journal.jsonl` `4657b519…71e24aa` (24 lines:
  8 journaled-before-connect + 8 opened + 8 completed)
- `captures/NP1-NATURAL.json` `87a54164…be3ca0db`
- `captures/NP1-MANUAL.json` `e41779fd…09d15552`
- `captures/NP2-NATURAL.json` `8d2a243d…39eece72`
- `captures/NP2-MANUAL.json` `a780ce27…99bf1118`
- `captures/NP3-NATURAL.json` `606150ec…388a186a`
- `captures/NP3-MANUAL.json` `bd307352…9402bc39`
- `captures/EN2009d-NATURAL.json` `fcefcdea…d577ea4cb`
- `captures/EN2009d-MANUAL.json` `90f0c1ed…9d577ea4cb`
- `ledger.json` `12e3207b…f0a788a3909`
- `RESULTS.md` `3b6ef6cd…9a45155cb`
- `COMPARISON.md` `dfd021a5…a6af3ff59`
- This `MANIFEST.md` (self-describing, hashed by reviewer on read)

## Budget and scope compliance

8 sessions / 8 attempts, 0 retries, 0 failures; 160.44s audio both
profiles; post-EOS receive bounded 15s (all exited on 3s quiesce);
ceiling $1, est <$0.02, no billing hard guarantee claimed.
No audio bytes stored in outputs (sanitized token records + walls only).
No Soniox-as-GT, no tuning, no PSEM change, no production touch.
