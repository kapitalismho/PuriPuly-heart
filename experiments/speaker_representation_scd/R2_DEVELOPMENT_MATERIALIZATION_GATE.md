# R2-L Legacy Common-GT Validation and Coordinate Gate

## Status

- Scope: exact existing legacy common-GT data only
- Upstream checkpoint: accepted R1 model acquisition, four-encoder smoke, and extractor parity
- Current action: adapt and run legacy-only data validation and coordinate materialization
- New corpus download or archive extraction: forbidden
- Neural inference: forbidden until a separate owner approval
- Training: forbidden

This document supersedes the former Zeroth/JVS acquisition-oriented R2 procedure. The old
`r2_execute archives` path and any public-corpus materialization path are historical implementation
artifacts and must not be run for the current study.

## Input Population

| Item | Frozen identity |
| --- | --- |
| Manifest | `experiments/speaker_turn_boundary/results/turn_episode_v1/episode_manifest_dev.json` |
| Manifest byte SHA-256 | `a5df5e56c2917ba174fca7892fbaa092f2b57705cad7a77bf9a347ba94cddfee` |
| Canonical content SHA-256 | `deb1713cd581c93cc13c47103643041c7551993985379869cfa0ca9a407dff68` |
| Total episodes | 804 |
| Diagnostic episodes | 695 |
| Source identities | 616 |
| Unique WAV bytes | 600 |
| Candidate inventory | 450 positive and 360 negative rows |
| Existing matched pairs | 313 |

These are the same audio and GT identities used by the existing ERes/LS-EEND comparison. The panel
is not resplit and is not promoted to confirmatory evidence. Legacy result files are read-only.

## Required R2-L Work

1. Revalidate the exact manifest byte/content hashes.
2. Resolve every referenced legacy WAV and annotation and verify its recorded identity.
3. Import or derive the minimum independent inventory needed by R3/R4: active-speaker sets, event
   classes, positive/negative rows, pair blocks, source/session blocks, and missingness.
4. Freeze at most the eligible existing 810 candidate rows for R3 while preserving shared rows
   across all four encoders. Do not manufacture replacement rows for absent conditions.
5. Freeze an R4 source subset of at most six source hours before any new encoder score is observed.
6. Generate trailing-window coordinates for 100, 300, and 500 ms contexts at a 1,600-sample
   (100 ms) hop.
7. Use the accepted R1 smoke measurements plus the completed coordinate ledger to forecast actual
   R3/R4 wall time, peak memory, and derived-cache storage.
8. Report exact R3/R4 inputs, model/layer/context grids, commands, and forecast to the owner, then
   stop for explicit approval.

R2-L may reference the existing WAVs in place when stable identity and read-only access can be
proven. Canonical copies in the external experiment cache are allowed only when required for stable
addressing; copying existing bytes is not corpus acquisition.

## Coordinate Contract

For each eligible waveform or bounded source region, the observation uses a causal trailing window
ending at frontier `t`. Context is one of 1,600, 4,800, or 8,000 samples. Frontiers advance by
exactly 1,600 samples. Padding is not observed audio, and coordinates that cannot supply the full
required context are excluded with a recorded reason.

The ledger must bind each coordinate to its source identity, WAV SHA-256, source sample interval,
context, frontier, event/negative label, and block identity. The same coordinate rows are consumed
by every encoder.

## Outputs

The implementation may retain the existing R2 external-cache namespace, but current receipts must
identify the source scope as `legacy-common-gt-v1` only. At minimum it produces:

```text
manifests/r2/legacy_common_gt/validation_receipt.json
manifests/r2/legacy_common_gt/coordinate_ledger.json
data/r2/legacy_common_gt/waveform_inventory.jsonl
data/r2/legacy_common_gt/source_metadata.jsonl
data/r2/legacy_common_gt/coordinates/*.jsonl
manifests/r2/legacy_common_gt/reduced_r3_r4_forecast.json
```

Exact names may be adjusted minimally to fit the current harness, but the source scope and receipt
semantics must remain unambiguous.

## Explicitly Disabled

```text
Zeroth/JVS archive download or extraction     no
any new public-corpus acquisition             no
D5 selection or access                       no
ERes-final inference rerun                    no
LS-EEND inference rerun                       no
pooled neural feature extraction              no, until owner approves R3/R4
full hidden-state persistence                 no
training or learned probes                    no
production changes                            no
```

## Exit and Approval Boundary

R2-L is complete only when the legacy identities and reduced coordinate ledger validate, the cost
forecast is available, and the owner has received the exact proposed R3/R4 invocation. The worker
must stop at that point. Only an explicit owner approval may start neural R3/R4 measurement.
