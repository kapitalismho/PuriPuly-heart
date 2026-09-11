# Decision-Sufficiency Manifest — IMPLEMENTATION_READY

Freeze `psem.decision_sufficiency.freeze.v1` (frozen 2026-09-09T09:58:30Z, before any run). Final run 2026-09-09T10:39:44Z. Review barrier complete. No pending.

## Stable artifact report (sha256; this manifest lists others only, never itself)

- `FREEZE.json` `d131883413aa969c0aa6ed5eb32ac134f411ed511af372baa3bf6e5d88d5ac67`
- `DECISION_RULES.json` `4742494bae17236b91a854e28f139bd8f44359b10b45249cdde578f0c97bd07e`
- `CAPABILITY_PROFILE.json` `84066336109d93a89614b8cccbf712f27a8c270a0c37f97b899114013e2926ef`
- `CAPABILITY_PROFILE.md` `f482adc1d379d7abdd09145505a04ecf2072061c1448330bff1530d17d3a0be7`
- `replay.py` `2e08a9ece97c0b8f80730450f5f05124e7b3f97d6268c308af2603720700753d`
- `LEDGER.json` file `29db23e33571497d876ae937b39ce9177368c8bd0eeb99257ff36c0c0f038f2c`, body `bf683eeecd05d16018b3c472a6536ebea9b88fb8b97194af81e4bcadb916de24`
- `receipt.json` file `185330c5de78942fb9fbfa2c8f0b8e03604bb26f32503cf8fcfd474db4984fe8`, stable `304f2ada44a423258e06328b98c8a4ec5469c0fcfe8fac95c2eaf32ccfafac55`
- `smoke.json` `138919d4ad469b895ca1137bc278c4444a5250843d6f470bbf0044791072471a` (25/25 PASS)
- `RESULTS.md` `45ad5383983c246e062558d444e290e797a31243be9b8c0430e1829f872ae0c8`
- `DECISION.md` `cc8479bb1551912d991c03c54956663d2d8077b1ef7e24d75c65e67196b89acf` (DECISION_READY)
- `DECISION_ADDENDUM.json` `cea3d74b8c54ab84acfba45a610e9b093a34c8da41f48399ebd7b241f6656685` (v3)
- `REVIEW_RECORD.md` `39b804d51d5c6469dae906222a5632e726500f473769bf749d0b9da1f736d300`
- Captures NP1 `a4ed5e84…` NP2 `a67b4a99…` NP3 `d92884ae…` P2-G03 `8c70c425…` P3-G04 `40121a32…`; annotations ZIP `b56e5babb2496b8795deeeda7e71178d7fbc9963f94276cf2a3f4b56ebbc9f9d` (full values in `receipt.json`).

## Provenance consumed read-only (sibling-owned, hash-bound in LEDGER/receipt)

- `provenance/RESULT.json` `a8d803f77f222e153e5fe3d64115ed0a5504adfd55801438d7b641cbbc8c6067` (`fix-complete-with-discriminator-and-prefix-probe`; pre-freeze prefix misstatement corrected)
- `provenance/README.md` `afcf778b…`, `provenance/PREFIX_ADDENDUM.json` `c4edc190…` (RECONSTRUCTED-UNCERTAIN, `frozen_at_utc` null; governing terms via handoff, no clocked durable pre-file proof)
- `provenance/prefix_probe.py` `a935a44d…` (byte-identical executed), `provenance/prefix_results.json` `be3dafd0…`, `provenance/PREFIX_ENV.md` `a860f37f…`
- `provenance/q8_discriminator.py` `7e307848…`, `provenance/q8_discriminator.json` `71ad6638…` (transition_restored True, drop 0.954; different-model observation)
- `provenance/audit.py` `208b136f…`, `provenance/audit_output.json` `2c6694a1…`, `provenance/FREEZE.json` `d8f74463…`, `provenance/oldfail_newpass_proof.txt` `0cea89a9…`, `provenance/pre_fix/SNAPSHOT.json` `c4ee00cf…`
- Source fix bound: `frame_alignment.py` `a60a8f90…`, `material.py` `595745e3…`, plus in-scope tests `51c63803…` / `15caffda…` (25 + 10 receipts = 35 passed measured; old export stays bound to `a3d9003a`, never relabeled).

## Reproduce

```powershell
./.venv/Scripts/python.exe experiments/psem_decision_sufficiency/replay.py --smoke
./.venv/Scripts/python.exe experiments/psem_decision_sufficiency/replay.py --run
```

Prior dirs (`psem_evidence_to_ownership`, `psem_repeatability_stage2`) untouched read-only; no commits; no new captures/sessions; no paid calls; no production adoption.
