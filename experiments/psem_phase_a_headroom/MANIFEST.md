# Phase A Headroom Manifest — Ledger Owner (Post-Barrier, v1.2 COMPLETE)

Freeze `psem.phase_a.headroom.freeze.v1.2` (clarification before comparison, actual UTC 2026-09-09T20:33:31Z; no retune). Baseline branch `experiment-v2-speaker-change-turn-boundaries-ls`, commit `83aaed984b8b245082f3ffe7bb15d71f3242361f` (40 chars). Live comparison COMPLETE under repaired contract. No new-H trigger.

## Owned deliverables (sha256)

- `AUTHORITY.md` `dce9a5ff08b1d940debcb02b1e0c3af23f860f913dd516c736e35d263736c03d`
- `FREEZE.json` `543eb0c6cfa36e651375cad5c23b7337cc0f5996ace56e74220b5af0bcb5689b`
- `replay.py` `1d5659fbb41f5a22a4940fe807ce0937aca75b2a6df7fe02ce08f21593ab8d04` (zero `#` comment lines; consumer contract plus 15/15 smoke)
- `live_headroom.py` `e9d1137d32e0ef99acf321341df5388db8fb8ddad0b64ddb5b2a65c303f57974` (zero comment lines; `#` only inside NITE href string literals; reuses the existing decision-sufficiency alignment helper with no second implementation)
- `old_grid_cache.json` `8f9474eecbd13c235062bbe8616f93de5b8b6b6316695d9864724b87dbb5814a` (read-only derived old-grid spans for exact-epoch join; old paths unmutated)
- `smoke.json` `608dd79861c21c54a264f0812f9c14e0419085380b866587fd80e6d765f88c67` (15/15 PASS)
- `synthetic/OBSERVATIONS.synthetic.json` `29b4ccfe13b5927fdc15ca908ed3f0f71672f6425ae196dd573e672a5dd92e7c` (pre-barrier fixture for smoke only)
- `causal_ownership_ledger.json` `52bd206b9b2ac265a7bf64376f87821fc2d3ddea8b28531604643cd70549e283` (live v1.2 result, obs sha bound inside, 15/15 integration checks)
- `PHASE_A_HEADROOM.md` `11e8c929247e4fb99025298a54a4a1dcaeb768460c03ae8c9ea36f141e444d61`
- `NEXT_DECISION.md` `6dc08c9cdd72d697c1d9b3583e919950bec80153ebc3f6a18caec1d8d762bb06`

## Explicit archive, not active (sha256)

- `archive/ARCHIVE.md` `84fc354890d8548cecf17f7478b4e16ce671e75a45557da2f285c2e02879574f`
- `archive/causal_ownership_ledger.pending.json` `4bb9a2cd5599f4cf9635fa9130dcb52cb67d9ebe1870b355c03bf5afa6897c4a` (pre-barrier placeholder, superseded)
- `archive/causal_ownership_ledger.synthetic.json` `139bb96c5467a258619cea6b29daeccbdcd8f10bf9d8363f8a73f29e6a024d58` (pre-barrier path proof, superseded)

## Frozen read-only inputs (sha256)

- Captures: NP1 `a4ed5e84f9fd866263646660edde1c5400f1269de1f58a98befa044b19da5005`, NP2 `a67b4a9953a1a08a55225b221f62af6208c767b67bedff6cdd72feef1dd76177`, NP3 `d92884ae50dcbc4e841c5dbc13c805665130dcec9fddc43a4745a32ed329b22d`, P2-G03 `8c70c42509525124b8ae49864a861a7b5736435872dae48e7f557dc1fbce730b`, P3-G04 `40121a3287b80f1a5845a81d0cdc161ced56025cb770940821406a53a61c508e`, P4 repair `c013facbbe07e6bab89954c1c8ec3b0ee9f84273ebf344b4d0876f60b1f25018`.
- Annotations ZIP `b56e5babb2496b8795deeeda7e71178d7fbc9963f94276cf2a3f4b56ebbc9f9d`; member byte identity carried from decision-sufficiency FREEZE (verified local sha match on read path).
- Case metadata: decision-sufficiency FREEZE, LEDGER, replay and stage-2 FREEZE plus captures, read-only, unmutated.

## Sibling observations (immutable, this root binds actual coverage)

- `observations/OBSERVATIONS.json` `3ec7c03c9c01bff859b9d6ce8d020aec40432c9deb7b951e1cf19be11a90b5aa` (bundle revalidated by root: per-source probs, hidden, logits, trace shas recomputed; continuity; parity maxabs 0; tail N plus 96; prefix multiples of 1280 below source totals).
- `observations/README.md` `e1ca73b12d6dd110de3b7da2de9c6c504e8c59acef3d725dc728847680e2617f`, `observations/repro.py` `e0f25aede7ffd6464dee0c0522de1c7095a5997fad0a7b37bd1f1d3bc80092f6` (added after barrier; root rehashes at bind time; repro smoke PASS).
- `observations/FREEZE.json` `3e14351a9d1947c3d6556a11250113a70a0e9441cb3b60e35d7719bdadf4674b`, `observations/psem_phase_a_timing.patch` `411226b6fc513166c689f898e0465690a917d1e25ec93cc856640633a7d68e39` (matches profile patch hash).
- Inner `observations/MANIFEST.json` `0135591dce6829dc3d34e1c45a2209f7b95cffa9b640fb3ca462d7837fab460b` is labeled a sealed subset that predates README and repro; root coverage above is actual, not claimed from the inner manifest.
- Feature and trace hashes bound per source as listed in OBSERVATIONS.json feature_files and trace_sha256 (ami_ES2009c, ami_ES2009d, ami_ES2002b, ami_ES2009a, ami_EN2009d); exe `3706db55ecf78c09188516cde829eccfcc04373a069d4f1834be0be22597540d`.

## Untouched

Prior dirs `psem_decision_sufficiency/`, `psem_evidence_to_ownership/`, `psem_repeatability_stage2/`, `psem_h7301_recovery/`, `psem_vulkan_fp16_features/` read-only; sibling observation files unedited; no commits; no new captures or sessions; no paid calls; no training; no H restore; no production adoption.

## Reproduce

```powershell
.venv/Scripts/python.exe experiments/psem_phase_a_headroom/live_headroom.py smoke
.venv/Scripts/python.exe experiments/psem_phase_a_headroom/live_headroom.py run
```
