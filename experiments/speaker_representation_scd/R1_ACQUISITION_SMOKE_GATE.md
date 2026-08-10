# R1 Acquisition and Smoke Gate

## Purpose

This checkpoint authorizes only the smallest execution boundary needed to establish R1 extraction
validity. It does not authorize a corpus download, full feature extraction, confirmatory access, or
training.

## Preconditions

- The accepted R0 contracts remain byte-identical.
- The independent legacy Phase-4 completion and verification artifacts both pass their self-hash
  checks.
- No process command references `experiments/speaker_turn_boundary`, and every potentially
  relevant process command line is inspectable.
- `SRSCD_CACHE_ROOT` is an absolute path on `C:` outside the repository.
- At least 55 GiB is free before environment or model acquisition.
- The R1 gate, source registry, environment lock, and legacy-release evidence match their pinned
  hashes.

The gate is revalidated immediately before every environment sync, model acquisition, and neural
smoke command. A single external lease prevents concurrent R1 actions. The worker verifies the
lease owner by PID and process creation time within at most two ancestor hops, which permits only
the observed Windows virtual-environment launcher hop. Before a worker can run, it is created
suspended and assigned with all descendants to a Windows Job Object. The Job Object
uses a conservative job-memory limit with 1 GiB reserved headroom under the 24 GiB contract,
kills the tree when its handle closes, and supplies the authoritative peak-memory value stored in
the usage receipt. The supervisor additionally scans every 250 ms for legacy work,
process-inspection failure, diagnostic process-tree RSS, per-action wall time, and cumulative R1
wall time; it terminates the Job on violation.

Every child action receipt embeds the lease execution ID and its expected final relative path. A
receipt becomes authoritative only when exactly one self-hashed `status=completed` usage
attestation records the same execution ID, action, receipt path, byte hash, self hash, and hard Job
accounting. Acquisition and smoke reject receipts without that completed attestation. If a child
receipt survives an aborted supervisor run, the next exclusive lease moves it intact to
`control/orphans/` with self-hashed quarantine metadata before a safe retry; it cannot unlock a
downstream phase or permanently occupy the final receipt path.

## Authorized actions

```text
metadata read                 yes
locked environment sync       yes
official source checkout      yes
four exact model downloads    yes
D0 fixture materialization    yes
ten-fixture neural smoke      yes
100-window single benchmark   yes
100-window batch benchmark    yes
corpus download               no
full extraction               no
confirmatory access           no
training                      no
```

Downloads are sequential and are verified against the R0 artifact size and SHA-256 before model
loading. Hugging Face remote code is disabled. PyTorch pickle checkpoints are accepted only from
the pinned official repositories and are loaded through a weights-only path.
Existing mismatched files, dirty source checkouts, foreign origins, and existing result receipts
cause an abort; acquisition and smoke never silently overwrite prior evidence.

## Locked environment

The experiment-local `environment/pyproject.toml` and `environment/uv.lock` select Python 3.12.10,
CPU-only PyTorch 2.7.1, TorchAudio 2.7.1, and Transformers 4.52.3. The environment is separate from
the product environment and its package set is checked at runtime before artifact loading.

Environment synchronization uses the external cache root:

```powershell
$env:SRSCD_CACHE_ROOT = 'C:\srscd-cache'
uv run python -m experiments.speaker_representation_scd.r1_execute sync-environment
```

After synchronization, the supervisor launches model acquisition and smoke under the
experiment-local interpreter. Direct worker execution is rejected. Sync, acquisition, smoke, and
resource-usage receipts preserve exact requested/worker argv, cwd, Git commit and dirty state,
CPU/RAM identity, interpreter/packages, deterministic controls, gate/code identities, and direct
model/frontend/license contracts. The completed resource-usage receipt additionally binds the
exact child action receipt; a child receipt alone has no downstream authority.

## Smoke outputs

Each model produces a self-hashed report with:

- exact gate, execution-code manifest, source registry, acquisition receipt, and interpreter identities
- verified model/source identities
- parameter count and cold-load time
- ten deterministic D0 fixture outcomes whose effective trailing windows prove the named event
  and required pre/post active-speaker regions are present
- silence, one-speaker, clean change, gap, overlap, backchannel, gain/noise, impulse-coordinate,
  and channel-change D0 scenarios
- independently calculated input/output length and feature shape by layer/tap
- empirical source-coordinate response spans per SSL layer and ERes tap, with exact window-end
  availability and no unsupported post-context frame-localization claim
- repeated-run, future-mutation, and batch/single deltas
- ERes `FUSED` reconstruction parity
- 100-window single and batch timing
- sampled process-tree RSS plus the Windows Job Object's hard limit, reserved headroom, and
  authoritative peak job-memory accounting under the 24 GiB ceiling

Passing smoke does not enable full extraction. A separate reviewed forecast checkpoint must combine
the measured seconds/window and cache bytes with the complete R2 coordinate ledger before
`full_extraction` can become true.

## Technical-validity and forecast split

`results/r1/technical_validity.json` records the accepted four-extractor G0 result separately from
R1 phase exit. Its external receipt references are valid only when the exact completed usage
attestations and report identities revalidate under `SRSCD_CACHE_ROOT`.

`configs/r1/full_job_forecast_contract.json` defines the next fail-closed calculation. Until the
verified development acquisition receipt, complete development coordinate ledger, and measured
pooled-cache serialization calibration all exist at their canonical external-cache paths, the
forecast status is `not_ready`, `forecast_approved` remains false, and full extraction remains
disabled. The public calculator accepts only `SRSCD_CACHE_ROOT`; it derives those three paths and
does not accept arbitrary ledger or calibration paths. It verifies the exact frozen source/split
contracts, every source and coordinate-shard file, independent source/context/total row recounts,
and actual float32 NPZ shapes plus sample-manifest membership before calculating a candidate. The
development acquisition receipt also binds a complete 16 kHz mono PCM waveform inventory. For
each waveform, the calculator independently regenerates the reduced trailing-window coordinate shard:
the first frontier is `eligible_start + context`, subsequent frontiers advance by exactly 1600
samples (100 ms), and the final frontier cannot exceed `eligible_end`. Every coordinate binds the
waveform SHA-256 and exact window start, window end, observation frontier, context, and hop. Missing,
extra, reordered, or geometrically altered coordinates fail even when all declared counts and shard
hashes are internally consistent.

Runtime uses a conservative per-window upper bound equal to ten times each model's balanced
ten-fixture single-window mean, followed by the 1.25 safety multiplier. The factor is valid because
the accepted smoke benchmark runs each of the ten recorded fixture contexts exactly ten times.
Storage checks separately enforce the 25 GiB source-download limit, the recorded 55 GiB
pre-download free-space condition, the 20 GiB derived-cache limit, and the 50 GiB total external
root limit. The last total includes the current external root, including environment, models,
sources, receipts, and retained evidence, plus the projected derived cache.

The required development scope contains only the legacy common-GT panel, Zeroth-Korean development
speakers, and JVS development speakers. Canonical paths must remain under the development namespace
after resolution. Confirmatory VoxConverse, AISHELL-4, Zeroth-Korean test, reserved JVS coordinates,
aliases, and sealed D5 paths are rejected before they are opened.

Every calculator result, including `not_ready`, binds the authority, forecast-contract byte/self
identity, calculator byte identity, unique execution ID, process/time identity, Git commit and dirty
status, argv, interpreter, and host. A modified calculator or contract therefore cannot emit a
structurally equivalent ceiling candidate under the reviewed identity.
