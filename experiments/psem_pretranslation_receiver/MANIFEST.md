# Manifest — psem_pretranslation_receiver (immutable checkpoint + C13 repair)

Freeze `psem.pretranslation_receiver.freeze.v1` frozen 2026-09-10T07:20:40Z + revision 2026-09-10T07:36:07Z (actual clocks, original C13 contract restoration not new ontology; prior freeze d060ef86 audited, not backdated).
Baseline branch `experiment-v2-speaker-change-turn-boundaries-ls` commit `83aaed984b8b245082f3ffe7bb15d71f3242361f`, upstream 0/0.
Ledger generated 2026-09-10T07:36:38Z (after revision), wall 26.4 s local, smoke 18/18 PASS, zero paid API.

## Owned deliverables (sha256)

- `FREEZE.json` `07527fddb22b3bcdde2ad893be80d6804f83c08b8fef85278c34a79df210358b`
- `replay.py` `10494780cdee21c77835999859d461239ea5f72abb16076164cfe1b8fa0840c3`
- `ledger.json` `97b2587c56dbea6c0d07f11a99293e496e0f30fecd5197e3b0ed12f827878c2f`
- `RESULTS.md` `51fa0f680ca4844406c9dd68824f4a7eb007f278daaffa929cd3fca11caf9a4f`
- `NEXT_DECISION.md` `a9e8c25326e1a57e5a955e14c48addab4741ca67844213a6e75d87d3310f5959`

## Old hash audit (pre-fix, not backdated)

- Prior `FREEZE.json` `d060ef869707ad349d4003d8716178301cc7bb48af512debf4bed4ddbce4fdd0`
- Prior `replay.py` `24497b69a702f468585ef1936fcbc4295464fac43b5839d1cebb1238fd3b0009` (single first-wins defect)
- Prior `ledger.json` `c1465c335bfc812b88d33d10247014e19652d5d320710c317a7a7217ba3c40af`
- PREFIX proof: synthetic 2-distinct-X before-fix 1 applied fails, after-fix 2 applied passes; actual COMBINED 2 seals X672000 Z684896 + X733440 Z746336 observed.

## Frozen read-only inputs (sha256, verified match)

- `ARCHITECTURE.md` `8971048f129d49b48a3abf6b9d546906e155eee5ed3b1e845d60e281e062d068`
- `experiments/psem_decision_sufficiency/FREEZE.json` `d131883413aa969c0aa6ed5eb32ac134f411ed511af372baa3bf6e5d88d5ac67`
- `experiments/psem_decision_sufficiency/replay.py` `2e08a9ece97c0b8f80730450f5f05124e7b3f97d6268c308af2603720700753d`
- `experiments/psem_p3t_rearm/FREEZE.json` `12e0cbeff1c38c239ff45d99032e1e6c66acb8674d01ef51c225a5103233592a`
- `experiments/psem_phase_a_headroom/FREEZE.json` `543eb0c6cfa36e651375cad5c23b7337cc0f5996ace56e74220b5af0bcb5689b`
- `experiments/psem_phase_a_headroom/live_headroom.py` `e9d1137d32e0ef99acf321341df5388db8fb8ddad0b64ddb5b2a65c303f57974`
- `experiments/psem_phase_a_headroom/observations/OBSERVATIONS.json` `3ec7c03c9c01bff859b9d6ce8d020aec40432c9deb7b951e1cf19be11a90b5aa`
- `experiments/psem_phase_a_headroom/old_grid_cache.json` `8f9474eecbd13c235062bbe8616f93de5b8b6b6316695d9864724b87dbb5814a`
- `experiments/psem_repeatability_stage2/captures/NP1.json` `a4ed5e84f9fd866263646660edde1c5400f1269de1f58a98befa044b19da5005`
- `experiments/psem_repeatability_stage2/captures/NP2.json` `a67b4a9953a1a08a55225b221f62af6208c767b67bedff6cdd72feef1dd76177`
- `experiments/psem_repeatability_stage2/captures/NP3.json` `d92884ae50dcbc4e841c5dbc13c805665130dcec9fddc43a4745a32ed329b22d`
- `experiments/psem_return_capacity/FREEZE.json` `9a7ab4c9a8b836b4f3076f40b61c53acfe8a43858cad351bda37092fa3a52faf`

## Smoke (actual focused, 18/18 PASS)

freeze<=terminal inclusive; 1 epsilon late denied; raw future invalid; revised same-interval no accum; invalid-scope; return-unsupported-R1; R2-return; prospective 2-distinct-X cuts 2; same-rev duplicates not cut; old-X already separated; return does not stop future OTHER; straddle conserved; token conservation + DROP/DUP/TEXT; no translations; real mapping in run.

## Untouched

Prior dirs read-only; Soniox reference only; no commits; no new captures; no paid calls; no training; no production adoption. Full NP unchanged single-seal paired exact.

## Reproduce

```powershell
.venv/Scripts/python.exe experiments/psem_pretranslation_receiver/replay.py smoke
.venv/Scripts/python.exe experiments/psem_pretranslation_receiver/replay.py run
```
