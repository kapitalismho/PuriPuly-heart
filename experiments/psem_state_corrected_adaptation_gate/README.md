# Issue 121 H7301 CPU postprocess

This records the completed H7301 CPU product evaluation. The GPU export and trained
head were treated as immutable; this is postprocessing only, not a GPU retrain or
identity change.

## Frozen inputs and GPU identity

The export binding names 53 FIT sources. The actual frozen NPZ payload is 21
files: 11 CALIB and 10 DEV sources. The export manifest is
`7ff72366ffa6182e5b3ef7824507294f8dc66073c99287744039a6a7701bc131`.

The unchanged GPU identity is recorded by the export binding:

- GPU/source code hash: `a3d9003a76ea167c33c644f1e3d15862e0181ed633a0378c91cb6d0fccaa263a`
- trained head hash: `bb5029da54b01a84763d0513a544cdbcd99bb1592f73b9eef71a1916e59aea3f`
- checkpoint hash: `8abd32832159c6ac1148c926b7276f35ba34582c444e559dce1f1253fea42ef8`
- input hash: `72e37c1a2612a365cd575bc7ca36646690ac77c1ce04066c1dbf29b0189c0ed8`
- partition hash: `e5190d423c2b038668831e558cc6771184d7e7b1c06b6ef6934f595fbffbdf6a`
- weights hash: `225f41be27b43450e9b9062349fed52b325d8faedcee23038aa1a511d16166f5`
- seed: `7301`

## Exact CPU command

Executed from the repository root with `OMP_NUM_THREADS=1`,
`OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`,
`TOKENIZERS_PARALLELISM=false`, and torch intra/inter-op thread counts of 1:

```text
uv run python -u -m experiments.psem_state_corrected_adaptation_gate.run_h_arm \
  --command postprocess \
  --export-dir .cache/issue-121-h-profile-staging/rerun-export/gpu_export \
  --out-dir .cache/issue-121-h-profile-staging/rerun-postprocess \
  --workers 8
```

The CPU implementation uses exact global threshold grids formed from the union of
member grids. It sweeps threshold events once per source and reuses immutable
primitive rows through the indexed interval-contamination path. Direct primitive
comparison, the real 1,830-row source check, distinct-member global-grid checks,
and the interval oracle all matched the legacy implementation bit-for-bit.

The final run used one ordered `spawn` pool with eight workers, emitted 42,857
score tasks, and reused 8,917,633 primitive rows. The wave receipt reports
1,693.8040342 seconds; observed end-to-end wall time was 29m18s (measured, not an
estimate). The process exited cleanly.

## Outputs and gate state

The canonical outputs and posthoc persistence evidence are now packaged in the
[durable H7301 publication bundle](results/issue-121-h7301-persistence-v1/README.md).
The bundle is prepared for Director review and is not yet committed, pushed, or
posted.

Canonical postprocess files and source-byte hashes:

- [`canonical/calibration_metrics.json`](results/issue-121-h7301-persistence-v1/canonical/calibration_metrics.json) —
  SHA-256 `c1a6b05ff5da589c793f641f604f3f8150607e58224c65e13f84c7dc4308adf8`
- [`canonical/dev_frontier.json.gz`](results/issue-121-h7301-persistence-v1/canonical/dev_frontier.json.gz) —
  decompresses byte-for-byte to `dev_frontier.json`, SHA-256
  `11b8195bbcf9a301a1524729a659762e834a89bd2a5ecbeaa365c2343a6f0345`
- [`canonical/gate1_diagnostics.json`](results/issue-121-h7301-persistence-v1/canonical/gate1_diagnostics.json) —
  SHA-256 `8df0d02a98f996178fc755a5805353a5fbc3399b2b4b6cfa11753b4bdf9dad84`
- [`canonical/gate1_decision_evidence.md`](results/issue-121-h7301-persistence-v1/canonical/gate1_decision_evidence.md) —
  SHA-256 `6f4dd8eff971076f0a302e366700bf64bfcb21509453d4d558d2be08db48dcab`

The bundle also includes the immutable export manifest, training metrics, all 11
CALIB and 10 DEV numeric NPZ files, and the deterministic analysis outputs.
`bundle_manifest.json` records every durable file's size and SHA-256 plus the
original 1.06 GB frontier size/hash and gzip metadata. No audio, transcripts,
PII, checkpoints, credentials, or process logs are included.

Canonical validation passed for the 11-source calibration metrics and the
10-source `R-H-SC` development frontier. The execution contract is verified, but
this is not a formal immutable candidate and is not terminal acceptance. The
Director's scientific disposition is recorded in
[STATE_CORRECTED_ADAPTATION_DECISION.md](STATE_CORRECTED_ADAPTATION_DECISION.md):
`STOP / inconclusive`; retain F0 operationally. The scientific gate decision is
closed: no Gate 1 receipt was emitted, and no T2 or evaluation arm was opened
automatically. Formal commit review remains outstanding; this is not scientific
gate analysis pending.
