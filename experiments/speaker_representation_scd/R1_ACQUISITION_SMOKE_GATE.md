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
smoke command. A single external lease prevents concurrent R1 actions. Before a worker can run, it
is created suspended and assigned with all descendants to a Windows Job Object. The Job Object
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
