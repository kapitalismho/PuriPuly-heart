# Review Record — decision sufficiency (durable)

Final run 2026-09-09T10:39:44Z. Ledger body `bf683eee…`, receipt stable `304f2ada…`, smoke 25/25 PASS.

## Review barrier (complete)

Three critics (source, causal, claims) reviewed the workstream. No material findings against the source fix, the causal probe evidence, or the root replay mechanics. One ACCEPT:

- Claims ACCEPT 1 (pre-freeze prefix misstatement): provenance `RESULT.json` described `PREFIX_ADDENDUM.json` as frozen BEFORE inference while the addendum is RECONSTRUCTED-UNCERTAIN (`frozen_at_utc` null). Assigned to the provenance owner ONLY. Completed: RESULT `a8d803f77f222e153e5fe3d64115ed0a5504adfd55801438d7b641cbbc8c6067`, README `afcf778b…`, PREFIX_ADDENDUM `c4edc190…` unchanged; governing terms via handoff, no clocked durable pre-file proof claimed. No code/results/timing changed by that fix. Root rehashed after completion (this record).

## Verification (measured, not asserted)

- Source fix: 25 targeted tests (`test_material_contract.py` + `test_frame_alignment.py`) pass; receipts suite (`test_receipts.py`) 10 pass — 35 passed total, measured 2026-09-09. Durable proof: `provenance/oldfail_newpass_proof.txt` (verbatim pre-fix execution saturates + stays silent; repaired tree passes) and `provenance/pre_fix/SNAPSHOT.json`.
- Prefix probe: 3/3 clips bit-exact charged-prefix stable (worst 0.0 at 1e-6), tail sensitive beyond lookahead (`provenance/prefix_results.json` `be3dafd0…`, executable `prefix_probe.py` `a935a44d…` byte-identical).
- Q8 discriminator: transition_restored True, drop 0.954, broken projection flat (`provenance/q8_discriminator.py` `7e307848…`, json `71ad6638…`). Different-model observation: falsifies all-observations-missing only.
- Root replay: deterministic existing-stream replay + read-only provenance consumption; smoke 25/25 (12 synthetic contract + 13 real incl. provenance/integrity invariants); ledger/receipt hash-bound.

## Adjudication (Director, recorded verbatim in substance)

- Final decision: restore source-aligned evidence before selecting/adopting model branch; source fix already implemented + validated.
- P3T is a SECONDARY research lead under the explicit conditional valid subset, NOT a second selected decision. No P3O-RESOLVED-toward-P3T (different Q8 model resolves nothing about the original scalar ambiguity). No false binary (authorise-rematerialization-OR-close withdrawn).
- Actual prerequisite: recover the EXACT missing H7301 head/cache for a comparable paired export (new binding, old preserved). Prior search (local, known archives, authenticated pod/volumes) empty; the user-owned external artifact is the only blocker, not general autonomy. F0 solo run possible but insufficient for H comparability and unneeded for the integrity decision. No recovery-impossibility or retraining claim.
- Stage #142 six branches DEFERRED behind the data-integrity prerequisite; THIS task COMPLETE evidence-backed (not generic BLOCKED/batchpause).

## Old NP2 conclusions explicitly withdrawn

1. NP2 null events prove the model cannot detect the B→C transition — WITHDRAWN (join artifact; scores were stale copies).
2. NP2 no-fire falsifies model repair by speed alone — WITHDRAWN as model evidence (raw no-fire forensic retained; no-fire at any nonnegative L is the artifact, not a speed limit).
3. DEV tail scores are genuine per-span model outputs — WITHDRAWN (collapse blocks, now source-fixed).
4. valid+mapped implies model support — WITHDRAWN (annotation grid, not support).
5. Full-pass 1<<30 invocation proves whole-source future context — WITHDRAWN as proof either way.
6. Q8 posterior_sessions as same-upstream NeMo evidence — WITHDRAWN as use (independent observation only).
7. Old export implicitly re-validated by the fix — NOT CLAIMED (old binding `a3d9003a` retained).
8. NP1 fragmentation-1 wording — WITHDRAWN and corrected (total emissions 2; in-payload fragmentation 0; no manufactured wrong-run).
9. P3O RESOLVED-toward-P3T — WITHDRAWN (different-model observation).
10. Authorise-rematerialization-OR-close-NeMo-line binary — WITHDRAWN (replaced by the exact-H prerequisite).
