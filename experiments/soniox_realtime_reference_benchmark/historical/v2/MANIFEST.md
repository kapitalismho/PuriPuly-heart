# Soniox realtime reference benchmark v2 — MANIFEST

Owner dir `experiments/soniox_realtime_reference_benchmark/` only.
v1 bytes preserved under `historical/v1/` (v1 FREEZE `778b10a1…`,
ledger `12e3207b…`, 8 captures + 24-line journal, re-verified after
move). All other files read-only. Baseline `83aaed9`, upstream `0/0`.
No Git mutations, no code comments, no publish, no deploy, no PSEM or
production change, no Soniox teacher use.

## Original read inputs (verified)

NP1 [19741040,19853040] `953164ee…195e`; NP2 [33405760,33517760]
`461e7f5d…f1`; NP3 [2553520,2665520] `e9d38552…26`; EN2009d [0,755520]
`2e7226c2…97c33fd8`. ARCHITECTURE `8971048f…062d068`; provider
`soniox.py` `686f3113…04d` untouched; `replay.py` `2e08a9ec…308af`
helpers reused; stage2 FREEZE `6b38e0f3…46deb`; return ledger
`d01f07ac…32691`; P3T ledger `65b3cff0…695c75`. AMI public CC BY 4.0.
Secret 64ch process-only redacted. No audio bytes stored (tokens/walls).

## Owned v2 artifacts (sha256)

- `FREEZE.json` `a8439628…70d4578` (2026-09-10T05:30:00Z, pre-call)
- `probe.py` `46fb16b0…78c8c` (physical EOS + finished-or-incomplete)
- `score.py` `1c6afa93…49914` (delta/prefix/WER/session-map disciplines)
- `captures/attempt_journal.jsonl` `5366130a…21b07b` (32 lines: 8
  journaled-before-connect + 8 opened + 8 completed + 8 post-eos-incomplete)
- `captures/NP1-NATURAL.json` `0b3fd824…dce7a`
- `captures/NP1-MANUAL.json` `ef7011ec…bbb141`
- `captures/NP2-NATURAL.json` `f34aab02…1c7bd98`
- `captures/NP2-MANUAL.json` `2b777c90…d4e033`
- `captures/NP3-NATURAL.json` `731d0069…0af850`
- `captures/NP3-MANUAL.json` `43796c4b…99709`
- `captures/EN2009d-NATURAL.json` `cd9d5a44…7481b`
- `captures/EN2009d-MANUAL.json` `3706db87…ba2d43`
- `ledger.json` `da4a9ef0…afba59136`
- `RESULTS.md` `22e27418…b38d3d1b`
- `COMPARISON.md` `1781c092…81f5e4731`
- This `MANIFEST.md` (self-described, hashed on read)

## Budget and scope compliance

v2 8/8 attempts (total 16 with v1), 0 failures, 0 retries; ~160.44s
audio v2 (~321s total); 15s finished bound honored all sessions;
ceiling $1, est <$1, actual billing unknown unclaimed. Offline protocol
smoke (fake transport, 5 proofs incl EOS-once, finished-absent
failure, delta normal, duplicate disambiguation, snapshot past-only)
passed BEFORE calls; throwaway removed after receipt.
