# P3T Rearm Manifest — freeze v1 COMPLETE

Freeze `psem.p3t.rearm.freeze.v1` (frozen 2026-09-09T20:46:58Z before comparison, branch `experiment-v2-speaker-change-turn-boundaries-ls` commit `83aaed984b8b245082f3ffe7bb15d71f3242361f` upstream 0/0). Probe `probe.py` reuses Phase A and decision-sufficiency helpers with no second implementation. Smoke 13/13, run 36/36 PASS. No training, paid, captures, inference, production, publication, or Git mutations.

## Owned deliverables (sha256)

- `FREEZE.json` `12e0cbeff1c38c239ff45d99032e1e6c66acb8674d01ef51c225a5103233592a`
- `probe.py` `bf351addc2e1e0c8ba50b9f5b63cc1a44436759efb4e43a3a82f43ad46d3c5ec` (zero hash-sign lines; reuses existing alignment, partition, native table, anchor support, slot, fire, schedule, conservation, QA helpers)
- `per_case_ledger.json` `65b3cff0f9be72e7fe86d6e3a205fa8b3c309a28c7f1a6f93f1cceb695c7527a` (generated 2026-09-09T20:49:07Z, obs bound inside, 36/36 checks)
- `P3T_RESULT.md` `c18a27e1f3a73a8c7b2921e7b65ab2113f18fbfddd7b813b5e3574ea7aa49f13`
- `NEXT_DECISION.md` `d7da6c7019366b1c21f0f32cd00f941de77c9f16f3eaabccc9cb10a64fffd77c`

## Frozen read-only inputs (sha256)

- Observations bundle `3ec7c03c9c01bff859b9d6ce8d020aec40432c9deb7b951e1cf19be11a90b5aa`; old grid `8f9474eecbd13c235062bbe8616f93de5b8b6b6316695d9864724b87dbb5814a`
- Helpers `live_headroom.py` `e9d1137d32e0ef99acf321341df5388db8fb8ddad0b64ddb5b2a65c303f57974`, `replay.py` `2e08a9ece97c0b8f80730450f5f05124e7b3f97d6268c308af2603720700753d`
- Phase A freeze `543eb0c6cfa36e651375cad5c23b7337cc0f5996ace56e74220b5af0bcb5689b`, ledger `52bd206b9b2ac265a7bf64376f87821fc2d3ddea8b28531604643cd70549e283`
- Captures NP1 `a4ed5e84f9fd866263646660edde1c5400f1269de1f58a98befa044b19da5005`, NP2 `a67b4a9953a1a08a55225b221f62af6208c767b67bedff6cdd72feef1dd76177`, NP3 `d92884ae50dcbc4e841c5dbc13c805665130dcec9fddc43a4745a32ed329b22d`
- Observation auxiliaries FREEZE `3e14351a9d1947c3d6556a11250113a70a0e9441cb3b60e35d7719bdadf4674b`, MANIFEST `0135591dce6829dc3d34e1c45a2209f7b95cffa9b640fb3ca462d7837fab460b`
- Decision-sufficiency FREEZE `d131883413aa969c0aa6ed5eb32ac134f411ed511af372baa3bf6e5d88d5ac67`
- Feature and trace hashes per source as listed in FREEZE.json inputs (EN2009d, ES2002b, ES2009a, ES2009c, EN2009d); profile FP16 Vulkan NO_MUL_MAT_VEC F32_HEAD LOWLATENCY; ARCHITECTURE `8971048f129d49b48a3abf6b9d546906e155eee5ed3b1e845d60e281e062d068`
- Phase A reports MANIFEST `5ae3cb2e9bca2f078da911c2629b84008fb0d737582819eeba6d9e21849107df`, HEADROOM `11e8c929247e4fb99025298a54a4a1dcaeb768460c03ae8c9ea36f141e444d61`, NEXT `6dc08c9cdd72d697c1d9b3583e919950bec80153ebc3f6a18caec1d8d762bb06`

## Untouched

Prior dirs `psem_decision_sufficiency/`, `psem_phase_a_headroom/`, `psem_repeatability_stage2/`, `psem_evidence_to_ownership/`, `psem_h7301_recovery/`, `psem_vulkan_fp16_features/` read-only; sibling observation files unedited; no commits; no new captures or sessions; no paid calls; no training; no H restore; no production adoption.

## Reproduce

```powershell
python experiments/psem_p3t_rearm/probe.py smoke
python experiments/psem_p3t_rearm/probe.py run
```
