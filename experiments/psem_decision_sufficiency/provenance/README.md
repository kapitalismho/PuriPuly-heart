# Provenance audit — decision sufficiency (owner B, exclusive)

Independent source/evidence audit of the exported-score plateau. Owns ONLY
`experiments/psem_decision_sufficiency/provenance/`. Sibling owns
temporal/receiver/report at the workstream root. P2/stage2 dirs and production
are read-only and untouched. No Git changes, no commits, no paid calls, no
training, no captures, no model inference.

## Files

- `FREEZE.json` — audit-source freeze (written BEFORE the audit run):
  exact 4 sources + why each + pinned producer identities.
- `audit.py` — executable audit (stdlib + numpy + repo session loaders only).
- `audit_output.json` — machine-readable executed evidence (written by audit.py).
- `RESULT.json` — verdict, stable schema
  `{status,inputs,executed_checks,constant_output_evidence,alignment_provenance,causal_support,available_observations,unavailable_exact_artifacts,branch_constraints,next_discriminator}`.

## Reproduce

From the repository root:

```powershell
$env:PYTHONPATH="."
./.venv/Scripts/python.exe experiments/psem_decision_sufficiency/provenance/audit.py
```

Observed: `wrote .../provenance/audit_output.json (24346 bytes)` in ~5 s.
Every number in `RESULT.json` is read from that output; unmeasured states are
`null` there and stay `null` here.

## Result (one paragraph)

The NP2 "flat tail" is a PROVEN out-of-range gather collapse, not a model/head
finding. `infer_dev_raw_logits` (material.py:1356-1362) maps session action
ends to native emitted frames with `action_sample_indices` (searchsorted
right-1) and no upper-bound guard, so every action end past the native grid end
saturates onto the last native frame. The tail-to-last block equals the longest
bit-exact f0-constant block in ALL FOUR sources (EN2009d@35207, ES2002b@13479,
ES2009c@13220 incl. the exact-equality boundary frame, ES2009d@13453 — all
contiguous to the record end). f0 locks instantly (pure
selected→sigmoid→1-prob→logit projection); H/cand follows after a GRU transient
(2023 frames on ES2009d, final step 2.4e-7) into a bit-exact fixed point
(residual const -0.7159323692321777). Ruled out by executed repro:
clamp rails (±13.815509557963773, nothing on-rail, plateau interior),
silence input (ES2009d RMS 355/376/378 with max 10232 across the collapse),
windowing/detach as a forward cause (detach touches autograd only; TRAIN slices
carry the 1e-5 chunk-vs-full gate, DEV ran full-pass only — a receipt gap, not
a cause), and any finite fill branch (none exists in the DEV f0 path).
Consequence: NP2 episode A00271 frames 18615..18626 sit inside the collapse, so
both arms' null first events (LEDGER) are join artifacts — the joined scores
are copies of native frame 18790 (waveform [24051200,24052480), ~589 s before
the scored span), NOT span audio. The GT B→C word transition exists, but it
does not prove the real-time model fails; the rejected unit is the DEV EXPORT
JOIN, not the model or the F0/H policy (STOP-on-policy from this artifact would
be an untested stop). Causal support for the DEV invocation is UNKNOWN
(bounded-support plausible: chunk 6 / right 7 / fifo+spkcache 188, 1040 ms
charged delay; no stored prefix receipt in gate1_diagnostics.json; whole-pass
invocation alone proves nothing either way).

## Required next observation (existing, no training)

Q8 `posterior_sessions.npz` per-episode posteriors overlapping the NP2 scored
span [33458560,33490400) with episode/speaker mapping (file local,
sha-verified `27b7eaaa…cee8`). Independent historical streaming mechanism
(transcribe.cpp Q8, 480 ms chunks, 1040 ms lookahead) on the TRUE span audio —
tests span discriminability for P3T-vs-P3O routing without claiming the same
upstream feature. Q8 chain (gguf `a5dadc…`, commit `d42c3bb`) vs #121 NeMo chain
(checkpoint `8abd3283`, NeMo rev `1a3c291`, code `a3d9003a`, head `bb5029`
declared-but-missing) are proven independent historical observations.

## Director-owned probe prerequisite (contract-changing, NOT executed)

Only if the Q8-span observation is discriminative AND the branch needs causal
NeMo span scores: bounded prefix probe (max 3 already-captured 7 s payload
clips, full-clip vs prefix-only streaming outputs, restored F0 `.nemo`
present-unexecuted at `.cache/issue-107-assets/checkpoints/…v2.1.nemo`,
471367680 bytes) must be proposed to the Director BEFORE any inference.
Local CPU ~0.24 s crops are NONPARITY per LEDGER and are never a PASS bound.
This task completes the audit with no model inference needed.

## Artifact hashes (executed)

- NPZ inputs: ES2009c `5e175c6f…`, ES2009d `0ab81949…`, ES2002b `ef0e84b1…`,
  EN2009d `06114896…`, P4 text `c013facb…` — 5/5 match.
- Payload slices re-verified 3/3 vs stage2 freeze; full wavs match
  RuntimeSession waveform_sha 3/3 (ES2009c `350661f9…`, ES2009d `b5ef4233…`,
  ES2002b `977fbf6c…`).
- gpu_export_manifest `7ff72366…`; posterior_sessions file match True.

## Source fix + discriminator + prefix probe (Director-authorized extension)

- Pre-fix snapshot: `pre_fix/SNAPSHOT.json` (git blobs `01ff8bfb` /
  `7b1fd9cd`, worktree was CLEAN; audit cites defective code against these).
- Fix: `material.py` DEV source-clock coverage + `waveform` record;
  `frame_alignment.py` fail-closed upper bound. TRAIN paths untouched.
  Old-fail (verbatim pre-fix execution) / new-pass proof:
  `oldfail_newpass_proof.txt`. Targeted tests: 25 passed
  (`test_material_contract.py`, `test_frame_alignment.py`).
- Q8 discriminator: `q8_discriminator.py` → `q8_discriminator.json`
  (INDEPENDENT_OBSERVATION; transition_restored True, drop 0.954;
  broken projection flat; never a NeMo/H claim).
- Prefix probe: `PREFIX_ADDENDUM.json` (RECONSTRUCTED-UNCERTAIN: governing
  terms predate the run through the Director handoff, but no clocked durable
  file-freeze proof exists; `frozen_at_utc` null) →
  `prefix_results.json` (3/3 clips bit-exact stable, sensitive tail).
  Isolated env `C:/tmp/psem-timing-iso`, checkpoint `8abd3283` verified
  pre-load, NeMo tree rev verified `1a3c291`, exact streaming preparation.
- Verdict: `RESULT.json` (status `fix-complete-with-discriminator-and-prefix-probe`,
  includes `fix`, `invalidated_claims`, `q8_discriminator`, `prefix_probe`).
  Old export stays bound to code hash `a3d9003a`; repaired hashes recorded;
  historic results/captures/freezes preserved; no commits.

## Source-fix notice: exact outdated bindings + changed files/tests

Outdated (pre-fix code hash `a3d9003a…`; DO NOT relabel, DO NOT overwrite):

- `…/issue-121-h7301-persistence-v1/export/gpu_export/gpu_export_manifest.json`
  (`7ff72366…`) and all 10 DEV score NPZs, in particular
  `dev_ami_ES2009c.npz` (`5e175c6f…`), `dev_ami_ES2009d.npz` (`0ab81949…`),
  `dev_ami_ES2002b.npz` (`ef0e84b1…`), `dev_ami_EN2009d.npz` (`06114896…`).
  Every joined tail score in the tail_to_last blocks (audit: EN2009d@35207,
  ES2002b@13479, ES2009c@13220, ES2009d@13453, all to record end) is a
  last-native-frame copy, not span audio.
- Downstream of those scores (same binding): `canonical/dev_frontier.json.gz`,
  `canonical/gate1_diagnostics.json`, `calibration_metrics.json`,
  `persistence_analysis.json`, and every stage2/workstream ledger receipt that
  consumed DEV scores (LEDGER null-events on collapsed spans are join
  artifacts, not model no-fires). Captures, freezes, and annotations are
  unaffected (no audio/text/GT bytes change).

Changed files (only these; no commits):

- `experiments/psem_sortformer_adaptation_depth/frame_alignment.py`
  repaired sha256 `a60a8f90…` (was blob `7b1fd9cd…`): fail-closed upper bound.
- `experiments/psem_state_corrected_adaptation_gate/material.py`
  repaired sha256 `595745e3…` (was blob `01ff8bfb…`): DEV source-clock
  coverage + additive `waveform` record.
- `experiments/psem_state_corrected_adaptation_gate/tests/test_material_contract.py`
  (`15caffda…`): `DevSourceClockCoverageTest` (2 new) + mismatch test updated
  to short-coverage.
- `experiments/psem_sortformer_adaptation_depth/tests/test_frame_alignment.py`
  (`51c63803…`): 2 new (upper-bound reject, partial-in-range convention) +
  `_native_rows` floor→ceil grid correction.
- `experiments/psem_state_corrected_adaptation_gate/README.md`: fix notice
  appended (historic content untouched).
- This directory: `FREEZE.json`, `audit.py`, `audit_output.json`,
  `q8_discriminator.py`, `q8_discriminator.json`, `prefix_probe.py`,
  `prefix_results.json`, `PREFIX_ADDENDUM.json`, `PREFIX_ENV.md`,
  `pre_fix/SNAPSHOT.json`, `oldfail_newpass_proof.txt`, `RESULT.json`.

Regression rationale (earns its place): non-uniform action grids longer than
rows×80 ms are the exact plausible-bug shape (all four DEV sources); the tests
assert observable behavior (per-row source-clock evidence values, fail-closed
raise), not plumbing. Re-exec commands: see `PREFIX_ENV.md`.
