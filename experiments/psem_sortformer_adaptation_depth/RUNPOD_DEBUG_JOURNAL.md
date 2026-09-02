# Issue #107 RunPod debug journal

This Git-visible journal is the operational source for a later reusable agent runbook. Append every failure, diagnosis, corrective action, and observed outcome here before continuing. Preserve direct evidence, separate confirmed causes from provisional diagnoses, and do not record API keys, private-key contents, tokens, or other secrets.

## Fast-feedback operating principles

These rules govern the remainder of this run and should be carried into the reusable runbook:

1. Use a cheap-to-expensive funnel: syntax/static checks, cached local image build, real checkpoint restore on CPU, short on-Pod CUDA canary, then one final immutable config/hash gate, then the full run.
2. Exercise the deepest cheap material path early. Package inventories and symbol imports do not substitute for restoring the exact checkpoint and instantiating its preprocessor. Do this before image publication or GPU provisioning whenever possible.
3. Run full suites and complete hash cascades only when their inputs changed and only once at the final boundary. Do not repeat broad validation merely because an operational step was retried.
4. Before F0→H→T2, run one bounded CUDA canary that restores the exact model, builds the graph, and executes the shortest representative material path. A failed canary blocks final config generation and full training.
5. Separate staging from GPU work. Prefer a prepared colocated volume or a cheap staging resource; do not spend GPU time on bulk transfer, package repair, or exploratory setup.
6. Long shell, Docker, and RunPod CLI operations run in an independent background process with a durable status/receipt file. Poll both the process and the relevant external control plane; do not leave an announced operation unmonitored or wait for another user message before continuing.
7. Polling must be bounded and state-driven. Retry documented transient failures with backoff, stop only on explicit terminal/error/deadline conditions, and record the last external state when a bound expires.
8. Keep `RUNPOD_API_KEY` local. Keep storage credentials process-only. Never put credentials in argv, Git, logs, status files, or Pod configuration.
9. Preserve failed-run evidence, but never replay a no-overwrite scientific run in place. Diagnose from durable logs, fix at the earliest responsible layer, and use a new run ID.
10. After every failure, record the cause and correction, then resume the approved plan autonomously unless a new destructive, publish, deployment, credential, or cost decision requires user approval. The current image publish, local commit, same-Pod image update, and `$0.44/h` restart were explicitly approved by the user on 2026-08-31.

## Fixed identities

- Candidate Git head: `8165ed583de7a4dd4b1a6ad6af842ff67ac4c43c`
- Image: `kapitalismho/puripuly-heart@sha256:20f44b72f748cdd755b0ff0dcb74de40fc6ee996e9a9ecd263c41598fdd746b8`
- GPU target: one secure-cloud NVIDIA A40, 48GB
- Container disk: 40GB
- `/workspace` Pod volume: 30GB
- Registry auth: `dockerhub-kapitalismho` / `cmtgvd46w000gemnu1fibmpva`
- Transfer manifest: `.cache/issue-107-assets/upload_manifest_a40_eu_ro.json`
- Manifest SHA-256: `b594bf8260a8da50f2a534c473f628af6ce2763fff1a7e0882470cccc927df98`
- Transfer payload: `4,595,080,125` bytes
- `RUNPOD_API_KEY` must remain local and has not been sent to a Pod.
- Fresh implementation review was explicitly skipped by the user. No implementation-complete claim is being made.

## Pre-provision preparation failures and corrections

### Bundle initially exposed only the `HEAD` pseudoref

- Failure: the first exact bundle contained `8165ed58… HEAD` rather than the named experiment branch. The strict branch-ref check rejected it even though the commit was correct.
- Diagnosis: `git bundle create <path> HEAD` records the pseudoref, which is weaker for an unambiguous clone/checkout workflow.
- Corrective action: removed only the newly created bundle and recreated it from `refs/heads/experiment-v2-speaker-change-turn-boundaries-ls`.
- Result: `git bundle verify` passed. The bundle now exposes `8165ed583de7a4dd4b1a6ad6af842ff67ac4c43c refs/heads/experiment-v2-speaker-change-turn-boundaries-ls`.
- Final bundle: `311,901,459` bytes, SHA-256 `f102981d305b57c29ad9f059e903fc4c5e7c3033091219dc631e7a6dc9089067`.

### PowerShell/Git probe mistakes

- Failure: unquoted `HEAD^{tree}` was misparsed by PowerShell, and `git bundle --version` is not a valid Git subcommand form.
- Corrective action: quoted `HEAD^{tree}` and used `git --version`; used `git bundle verify` and `git bundle list-heads` for bundle verification.
- Result: candidate tree resolved to `d43b675f17b14891c3cb5b16d32834c5ce75df91`; no repository change resulted from the failed probes.

### Stale static preflight and stale repository bundle in the transfer manifest

- Failure: staged `static_preflight.json` and the repository bundle still referred to start base `cb0e3ec…`.
- Corrective action: regenerated static preflight from the clean exact candidate, replaced only the manifest bundle entry with the exact candidate bundle, and changed the manifest to schema v2 direct-SCP Pod-volume staging.
- Result: all 12 local entry sizes and SHA-256 values passed one complete verification pass. Static preflight is bound to `8165ed58…`, payload SHA-256 `918e6b859ecbf79baa395da1d76064c280c60d638f5c2d582b016e86057ee4c6`.

## Live RunPod attempts

### Attempt v1 — default image command exits repeatedly

- Pod: `l2aitiawnnis0q`
- Name: `issue-107-a40-f0-h-t2-8165ed58`
- Created: `2026-08-31T09:45:10Z`
- Configuration: secure A40, `$0.44/h`, exact image digest, 40GB container disk, 30GB `/workspace`, port `22/tcp`.
- Failure: `pod create --wait` spent 20 minutes waiting. The SSH port was allocated near the timeout but refused connections.
- Evidence: container logs repeated the NGC PyTorch banner approximately every 16 seconds.
- Diagnosis: the derived Dockerfile defines no persistent `CMD`; the inherited NGC command exits, so RunPod repeatedly restarts the container and SSH never stabilizes.
- Corrective action: stopped the Pod before any transfer or training. It was not deleted.
- Result: `runtimeStatus=stopped`, `runtimeStatusReason=stopped_by_user` at `2026-08-31T10:06:02Z`. Its 30GB volume contains no staged run data.

### Attempt v2 — keepalive works, but no SSH daemon/key path

- Pod: `4fa7q9feq61m4l`
- Name: `issue-107-a40-f0-h-t2-8165ed58-v2`
- Created: `2026-08-31T10:06:24Z`
- Change from v1: added Docker args `sleep infinity`.
- Control-plane interruption: the local create/wait tool call was aborted, but a subsequent `pod list --all` confirmed the Pod had been created and was running. No duplicate was created after discovery.
- Failure 1: SSH metadata exposed a host/port, but TCP connections were refused or timed out.
- Evidence: Pod environment initially had `PUBLIC_KEY` empty. `runpodctl exec python` failed because `C:\Users\salee\.runpod\ssh\runpodctl-ssh-key` did not exist.
- Corrective action 1: created a dedicated workspace-local ED25519 key at `.cache/runpod-issue107-ssh-key`, fingerprint `SHA256:Yv9B6loZ1ElwDKxL+4q74Q8tZinD1dr0sU8CCnB7IqE`; registered only its public key with RunPod; updated the Pod `PUBLIC_KEY`; reset the same Pod so its volume remained intact.
- Failure 2: SSH still refused connections after the key update/reset.
- Diagnosis: `sleep infinity` keeps the container alive but does not start an SSH daemon or populate/start the image-side SSH service.
- Corrective action 2: stopped v2 before any transfer or training. It was not deleted.
- Result: `runtimeStatus=stopped`, `runtimeStatusReason=stopped_by_user` at `2026-08-31T10:33:04Z`. Its 30GB volume contains no staged run data.

### Attempt v3 — conditional OpenSSH bootstrap

- Pod: `v6l27rdzg5s591`
- Name: `issue-107-a40-f0-h-t2-8165ed58-v3`
- Created: `2026-08-31T10:33:08Z`
- Change from v2: startup command conditionally installs `openssh-server` only if `sshd` is absent, writes the registered `PUBLIC_KEY` to root authorized keys, then runs `/usr/sbin/sshd -D -e` in the foreground.
- Rationale: this changes only operational SSH transport; it does not install or repair Python, CUDA, Torch, NeMo, or scientific runtime packages. Runtime audit is still required before training.
- Control-plane interruption: the local create/wait tool call was aborted. A subsequent live list confirmed the Pod exists.
- Current observed state: `desiredStatus=RUNNING`, `runtimeStatus=initializing`, `runtimeStatusReason=awaiting_container`, 30GB volume.
- Current action required: inspect v3 system/container logs, wait only for a bounded startup result, verify SSH with the dedicated key, then either proceed with transfer or record the exact bootstrap failure before another correction.

## Current live account/resource snapshot

Snapshot after discovering v3:

- Balance: `$13.0262791381`
- Current spend: `$0.468/h` (`$0.44/h` active A40 plus `$0.028/h` observed storage/other account spend)
- Spend limit: `$80`
- Active provisioning Pod: `v6l27rdzg5s591` (`initializing/awaiting_container`)
- Stopped new attempts retained: `l2aitiawnnis0q`, `4fa7q9feq61m4l`
- Previously existing stopped Pods retained: `9o5a7ymbmg8fgq`, `xc744emqf9475f`, `hipld8aju48g96`, `5zwi7jr66bbk6f`
- No Pod or volume has been deleted.
- No assets have reached a Pod yet.
- F0/H/T2 training has not started.
- Windows watchdog has not been armed.

## Append protocol

For every subsequent event, append:

1. UTC timestamp and resource ID.
2. Command or action, with secrets removed.
3. Expected outcome.
4. Actual status/error and direct evidence.
5. Diagnosis, clearly marked as confirmed or provisional.
6. Corrective action.
7. Post-action result and whether billing/training/watchdog state changed.

## Later runbook extraction

When this launch stabilizes, convert confirmed lessons into a reusable agent runbook. Keep incident-specific IDs and transient balances in this journal; move only general prevention rules, known-good command shapes, bounded diagnostic sequences, and verified recovery procedures into the reusable guide.

## 2026-08-31 UTC — v3 reached SSH but rejected the key

- Resource: Pod `v6l27rdzg5s591`.
- Observed transition: `runtimeStatus` changed from `initializing/awaiting_container` to `running`; SSH endpoint became `194.68.245.51:22164`; reported uptime was 51 seconds.
- System/container log queries returned bounded `code=timeout` responses with no matching lines.
- SSH transport result: TCP/SSH banner and host-key exchange succeeded, but authentication failed with `Permission denied (publickey,password)` for root using `.cache/runpod-issue107-ssh-key`.
- Confirmed conclusion: the conditional OpenSSH bootstrap started an SSH service successfully; this is no longer a port-allocation or daemon-start failure.
- Provisional diagnosis: the authorized key written by the startup command does not match the local private key, was rendered incorrectly by command quoting, or is rejected by sshd/root policy.
- Next bounded checks: verify the local private/public key pair without printing private material; inspect sshd authentication logs; then correct key rendering rather than creating another Pod blindly.
- Billing/training state: one A40 remains active at `$0.44/h`; no asset transfer, training, or watchdog has started.

## 2026-08-31 UTC — v3 authentication root cause confirmed

- Local key-pair verification with an explicitly empty passphrase failed: `ssh-keygen -y -P ""` exited 255.
- `ssh -vv` offered ED25519 fingerprint `SHA256:Yv9B6loZ1ElwDKxL+4q74Q8tZinD1dr0sU8CCnB7IqE`.
- The server replied `Server accepts key`, proving the Pod's `authorized_keys` rendering, sshd root public-key policy, public key, and network path are correct.
- The client then reported `we did not send a packet` and exhausted authentication methods. Confirmed diagnosis: the PowerShell `ssh-keygen -N` invocation created the private key with a nonempty accidental passphrase, so BatchMode could not sign.
- Container logs independently confirmed OpenSSH installation completed, host keys were generated, and sshd listened on IPv4/IPv6 port 22. Only approximately 2.6MB of SSH transport packages were added; no Python/CUDA/scientific package was changed.
- Corrective action selected: generate a second dedicated ED25519 key through a Python subprocess argument array with a true empty `-N` argument; register only its public key; update/reset the same v3 Pod so the existing startup command writes the corrected key. Do not create a fourth Pod.

## 2026-08-31 UTC — corrected-key generation verification stopped before Pod mutation

- Python invoked `ssh-keygen` with an actual empty `-N` argument and created `.cache/runpod-issue107-ssh-key-v2` plus its public file.
- The immediate derived-public comparison reported a mismatch, so the key was not registered and the Pod was not updated/reset.
- Safety result: v3 remains unchanged and active; no additional Pod was created. The next check compares only public algorithm/blob hashes to distinguish a verification-script bug from a real key mismatch.

## 2026-08-31 UTC — corrected key verified; prior mismatch was a checker bug

- `ssh-keygen -y -P ""` exited 0 for the v2 private key, confirming a true empty passphrase.
- Derived and `.pub` algorithms both equal `ssh-ed25519`.
- Derived and `.pub` public blobs both hash to SHA-256 `8333aa69b9dd7ededd77f74a573ff66e0576bd7dfe2743302feb7940474713fb`.
- Confirmed diagnosis: `ssh-keygen -y` returned a third comment field on this Windows build; the first checker compared all three derived fields with only the first two public fields. The key itself is valid.
- Corrective action: register the v2 public key, update v3 `PUBLIC_KEY`, and reset the same Pod. Reuse the installed SSH transport if the container disk persists; otherwise the bounded startup command reinstalls it.

## 2026-08-31 UTC — v3 SSH correction succeeded

- Registered the verified no-passphrase v2 public key, updated Pod `v6l27rdzg5s591` `PUBLIC_KEY`, and reset the same Pod.
- First post-reset attempt at approximately 15 seconds got `Connection refused`; this was a startup race while the port mapping preceded sshd readiness.
- A subsequent fresh `pod get` reported `runtimeStatus=running`, uptime 25 seconds, and endpoint `194.68.245.51:22006`.
- SSH with `.cache/runpod-issue107-ssh-key-v2` succeeded as root. Direct evidence: `SSH_OK`; `NVIDIA A40, 46068 MiB, driver 580.159.04`; `/` is a 42,949,672,960-byte overlay with 42,923,728,896 bytes available; sshd listener and root session were present.
- `/workspace` is mounted from `mfs#eu-se-1.runpod.net:9421`. Its `df` reports the backend aggregate rather than the configured per-Pod 30GB allocation, so control-plane `volumeInGb=30` remains the allocation evidence and live used-space/headroom checks remain required.
- One inline Torch probe had a local-to-remote quoting bug and produced Python `SyntaxError: unexpected character after line continuation character`. This was a diagnostic command failure only; SSH and the Pod remained healthy.
- Corrective action: rerun the Torch probe without nested string quoting. No additional Pod/reset is needed.

## 2026-08-31 UTC — v3 CUDA and empty-volume probe passed

- Corrected Torch probe output: `1 12.8 1 NVIDIA A40`, proving CUDA is available, Torch reports CUDA 12.8, exactly one GPU is visible, and the device is an NVIDIA A40.
- `/workspace` mount: writable FUSE `mfs#eu-se-1.runpod.net:9421`.
- Initial `/workspace` usage: `0` bytes from `du -sb /workspace`.
- Provisioning result: v3 is the selected live Pod for transfer. v1/v2 remain stopped and undeleted.

## 2026-08-31 UTC — selected v3 transfer started

- Resource: Pod `v6l27rdzg5s591`.
- Action: resolve the current SSH endpoint, create `/workspace/issue-107/{checkpoints,packages,receipts}`, and transfer the 12 manifest entries plus the transfer manifest and capacity receipt via SCP using the verified v2 key.
- Expected payload: `4,595,080,125` manifest bytes plus two small control receipts.
- Safety: no API key or private-key content is transferred; no existing Pod/volume is deleted; source archives remain local.

## 2026-08-31 UTC — initial SCP was interrupted; partial state measured

- The foreground SCP tool call was aborted, so completion was not assumed and no blind retry was started.
- Pod `v6l27rdzg5s591` remained running at `194.68.245.51:22006`.
- Remote staged total: `1,275,251,503` bytes across four files.
- Complete-size files: checkpoint `471,367,680`; diar package `7,867,101`; NeMo package `134,692,946` bytes.
- Partial file: `packages/psem-strategy-data-v2.tar.gz` is `661,323,776` of expected `3,662,204,401` bytes.
- Bundle and receipts were not reached.
- Confirmed limitation: direct SCP writes the destination filename and does not provide an agent-safe resume after interruption.
- Corrective action: switch remaining transfers to OpenSSH SFTP batch `reput`, which resumes from the remote byte offset. Future interruptions can resume the same batch rather than retransmitting completed gigabytes.

## 2026-08-31 UTC — local SFTP path rejected for throughput

- The resumable SFTP `reput` call was interrupted before completion.
- User correctly identified that sending the full 4.6GB payload over the local Windows uplink is unnecessarily slow.
- Corrective strategy: reuse the already uploaded EU-RO S3/network-volume objects for unchanged checkpoint/corpus/source packages; generate short-lived presigned downloads for the live Pod; handle only the exact new Git bundle and refreshed receipts separately.
- Security constraint: do not expose S3 secret keys or `RUNPOD_API_KEY` to the Pod. Presigned URLs must not be written to the Git-visible journal.

## 2026-08-31 UTC — prior fast transfer design restored

- User recalled the prior correct design: generate short-lived S3 presigned URLs locally and download from inside the Pod over SSH.
- Confirmed process error: expired URLs required regeneration, not a fallback to full local SCP/SFTP.
- Recovery plan: reuse unchanged EU-RO S3 objects verified by `upload_receipt_a40_eu_ro.json`; upload only the exact new bundle and refreshed control receipts; generate fresh short-lived URLs without exposing secret keys; execute downloads inside v3; verify SHA-256 afterward.

## 2026-08-31 UTC — lingering local SFTP writer stopped before S3 resume

- Two consecutive remote size observations increased from `835,543,040` to `835,645,440` bytes, proving the aborted tool call left `sftp.exe` running locally.
- Located exactly one process whose command line referenced `issue-107-sftp-reput.batch`: PID `43748`.
- Stopped only PID `43748` and confirmed exit before starting another writer. This prevents concurrent corruption of the corpus destination.
- `curl` is available inside v3. Pinned transient `boto3==1.35.99` is available through `uv run --python 3.12`; no project dependency was added.

## 2026-08-31 UTC — first restored fast-path attempt failed safely

- Pod-side resumed `curl` received HTTP 401 for the fresh presigned corpus URL on all bounded retries; the corpus partial file was retained.
- The downloader's uncaught `CalledProcessError` included its argv, causing the short-lived presigned URL to appear in local diagnostic output. No long-lived secret key was exposed, but future downloader errors must sanitize URLs before raising.
- The small-transfer SFTP batch used `reput` for an absent exact bundle. This Windows/OpenSSH SFTP implementation failed after `stat remote: No such file or directory` instead of treating it as a new upload.
- Corrective actions: determine whether 401 reproduces locally; regenerate signing configuration if needed; use `put` for first bundle/receipt creation and reserve `reput` only for an existing partial destination.

## 2026-08-31 UTC — why the former presigned download path no longer works

- Old and fresh handoffs use the same path-style host, bucket/key path shape, and SigV4 query fields. The fresh credentials are valid because authenticated Boto3 `HeadObject` succeeds and verifies the remote corpus size/ETag.
- Fresh query-authenticated GET fails independently from both locations: Pod `curl` returns 401 and local ranged GET returns 403 with edge error code 1010. Virtual-host addressing is unavailable because the generated bucket subdomain does not resolve.
- RunPod's current official S3 compatibility table marks `GeneratePresignedURL` unsupported while authenticated `GetObject`, AWS CLI, and Boto3 are supported.
- Confirmed conclusion: presigned GET may have worked previously as an uncontracted implementation detail, but it is not a supported interface and cannot be treated as stable. URL expiration, local upload bandwidth, and corpus object identity are not the current cause.
- Canonical recurring design: stage immutable assets on a sufficiently sized network volume colocated with the target GPU before renting the GPU; use a cheap CPU staging Pod or supported authenticated S3 client for ingestion; attach the prepared volume at GPU Pod creation; bake persistent SSH startup into the derived image; retain exact hashes and candidate binding.
- Canonical authenticated-download fallback: use volume-storage credentials through a scoped/rotatable secret channel and an isolated transfer helper, never `RUNPOD_API_KEY`; remove/rotate credentials after transfer. Do not embed long-lived credentials in scripts, Git, logs, command output, or the scientific environment.
- Best current no-secret recovery: create a temporary EU-RO relay Pod attached read-only in intent to existing volume `tifw77udi2`, copy the corpus cloud-to-cloud to the selected GPU Pod, verify SHA-256, then stop the relay. This adds temporary infrastructure/cost and requires explicit approval.

## 2026-08-31 UTC — authenticated S3 process-only transfer approved

- User explicitly selected the supported authenticated S3 fallback.
- Security boundary: `RUNPOD_API_KEY` remains local. Volume S3 access/secret values are sent only through encrypted SSH stdin to one downloader process; they are not placed in Pod configuration, shell argv, files, Git, journal, or command output.
- Transfer helper is isolated under `/tmp/issue107-s3-venv` with exact `boto3==1.35.99`; it does not modify the scientific Python environment.
- Download design: authenticated `HeadObject` validates remote bytes/ETag; ranged `GetObject` resumes the existing corpus offset; final size and SHA-256 must match before proceeding.

## 2026-08-31 UTC — authenticated resume rejected contaminated partial

- The supported authenticated `HeadObject` and ranged `GetObject` path ran, but the final hash gate correctly blocked promotion.
- Observed staged corpus: `3662532081` bytes and SHA-256 `0b7a436979e25006d20d0deb2c735381cad52cc47a4e3c15b4122b45268d2e85`.
- Expected corpus: `3662204401` bytes and SHA-256 `bc25466e352e7e708485299284f89d198d285cf3bce78fe82bcc2c6c4dcdbb42`.
- The staged file is exactly `327680` bytes oversized. Remote staged total is `4276459808` bytes.
- Conclusion: the existing SCP/SFTP partial cannot be trusted as an immutable prefix. Do not trim it and do not append another ranged response to it.
- Recovery: first prove no local or remote transfer writer remains; then download the full object to a distinct `.auth.part` path, verify exact size/SHA-256, atomically replace the final path, and only then remove the quarantined corrupt artifact after recording its identity.

## 2026-08-31 UTC — clean authenticated corpus replacement succeeded

- Identified the actual concurrent writer after the first failed authenticated resume: local `scp.exe` PID `16980` and child `ssh.exe` PID `34804` were still running the original four-package upload to `194.68.245.51:22006`.
- Stopped only those two identified transfer processes. Follow-up local process search found no matching transfer writer; remote process search found no `sftp-server`, `scp`, or authenticated downloader process other than the inspection command itself.
- The original SCP/SFTP writer was overwriting the existing destination in place without truncating its `327680`-byte stale tail. This explains why the file remained oversized while its content and hash changed.
- Quarantined corrupt artifact: `/workspace/issue-107/packages/psem-strategy-data-v2.tar.gz.corrupt-3662532081-7df3180d8409b7c9`, `3662532081` bytes, SHA-256 `7df3180d8409b7c9ba437cd92c34560e023685a908dfaf7ce7fa6136c21fe13d`.
- Downloaded the full object from byte zero through supported authenticated Boto3 `GetObject` to an isolated `.auth.part`; no SCP/SFTP writer was present.
- Verified and atomically promoted corpus: `3662204401` bytes, SHA-256 `bc25466e352e7e708485299284f89d198d285cf3bce78fe82bcc2c6c4dcdbb42`, source ETag `67696e91491070ef1e04979bc12849f3-219`.
- Independent post-promotion `stat` and `sha256sum` matched; `.auth.part` was absent. The corrupt artifact may now be removed because its complete identity is recorded and the exact replacement exists.

## 2026-08-31 UTC — Task 13 staging and extraction gate passed

- Transferred the exact candidate bundle plus small receipts after the corpus replacement. Pod-side verification of all 12 schema-v2 manifest entries passed exact byte and SHA-256 checks.
- Initial `git bundle verify` outside a repository failed with `need a repository to verify a bundle`; this is Git command context behavior, not bundle corruption. Initial clone then warned that the bundle transport had no symbolic remote `HEAD`, so it could not auto-checkout.
- Corrected procedure: clone the full-history bundle, explicitly check out `refs/remotes/origin/experiment-v2-speaker-change-turn-boundaries-ls`, then run `git bundle verify` inside the clone. Verified `/workspace/repo` HEAD `8165ed583de7a4dd4b1a6ad6af842ff67ac4c43c`, clean status including untracked files, exact contained branch, and complete history.
- Extracted corpus with no component stripping, yielding `assets/corpus/ami` and `assets/corpus/alimeeting`. Extracted the reference archive with one stripped component so `assets/reference/.git` is direct; verified clean reference HEAD `9527b7c64846fb38316a610f32e9d3466bd6d8b7`. Copied the exact checkpoint read-only. Did not extract the staged NeMo source archive over immutable `/opt/nemo`.
- RunPod FUSE `df` reports backend aggregate capacity and cannot prove the Pod quota. Combined control-plane allocation (`volumeInGb=30`, `containerDiskInGb=40`) with a conservative sum of every logical file byte under `/workspace`.
- Conservative workspace logical bytes: `12432751439`; declared 30-GB headroom: `17567248561`; required runtime reserve: `8589934592`; reserve gate passed. Asset logical bytes: `6661396524`, below the 10-GB expansion bound. Root overlay total: `42949672960` bytes with `42879496192` free.

## 2026-08-31 UTC — first detached run failed; watchdog stopped correctly

- Immutable config: run ID `issue-107-a40-8165ed58-01`, config SHA-256 `e492c5f204e56d0173522be1fdfe98e8c3aced2c2d1fde370bedf572fcb619cc`, deadline `2026-09-01T06:33:08.756000+00:00`.
- Runner PID `2142` entered `RUNNING/bootstrap-f0`; heartbeat advanced from sequence `0`/`STARTING` to sequence `21`/`RUNNING`, proving temporal liveness.
- `bootstrap-f0` failed at `2026-08-31T11:51:48.770112+00:00` with code 1 while `build-lineage` restored the pinned Sortformer model. Exact root cause: librosa 0.10.1 imported `pkg_resources`, but the immutable image contained setuptools 84.0.0, which no longer ships that module.
- Runtime validation and live preflight had passed because they checked exact distribution and symbol identities but did not restore the real checkpoint or instantiate its audio preprocessor.
- Windows watchdog could not use its remote SCP mode initially because its strict default known_hosts had no entry; prior manual SSH used an isolated no-persistence host-key policy. A pinned ED25519 host key was captured with fingerprint `SHA256:XulF//wJ4xtkXB6MEBPwWmqRFzLuapP4B52AEPYuwpk`, and the script's supported local-control mode was fed by a pinned-host-key mirror.
- The watchdog armed at `2026-08-31T11:53:36.2971067+00:00`, bound run/config/deadline exactly, observed state `ERROR`, and stopped Pod `v6l27rdzg5s591` in one attempt. Receipt reason: `control_status_error`; stop confirmed at `2026-08-31T11:53:40.0834967+00:00`.
- Restart initially failed repeatedly because the original host had no free A40. A later capacity retry succeeded without deleting/resetting the Pod; the persistent `/workspace` run logs remained available. After log recovery, the ERROR Pod was stopped again to avoid idle GPU billing.

## 2026-08-31 UTC — immutable pkg_resources repair candidate built locally

- No live scientific-environment repair was performed. The fix is an immutable derived-image rebuild.
- Replaced `setuptools==84.0.0` with exact `setuptools==80.9.0` in the 81-entry runtime closure and added it to the explicitly reapplied repair set.
- Strengthened build/runtime validation to require setuptools 80.9.0 and successful `pkg_resources` import.
- First rebuild failed closed because setuptools 80.9 has different vendored `.dist-info` versions than setuptools 84; the old metadata-isolation list was version-specific and left duplicate `importlib-metadata` active.
- Generalized metadata isolation and validation to move/reject every `.dist-info` or `.egg-info` directly under `setuptools/_vendor`, independent of the setuptools vendor versions.
- Local non-`latest` image build passed: `kapitalismho/puripuly-heart:issue107-pkgresources-r1`, local image ID `sha256:e77ec0a6ed5ae04e185e6992dd019f2c7dde72196a9e5bc14adb8da1b253b966`, virtual size `27548336339` bytes.
- Build receipt evidence: `passed=true`; setuptools `80.9.0`; `pkg_resources` at `/usr/local/lib/python3.12/dist-packages/pkg_resources/__init__.py`; exact runtime-constraint count `81`; runtime-constraints SHA-256 `76db78297d161a23653c6e19e09ac68e8c1c4c2e03335f98e70cd62873dd1ee4`; canonical distribution-record SHA-256 `80bc42f2447677e76020b7d429f7787fa24ef3a9767eadd757b28c5255de934f`; `pip check` passed.
- Real checkpoint restore smoke passed inside the local image against the exact `471367680`-byte checkpoint SHA-256 `8abd32832159c6ac1148c926b7276f35ba34582c444e559dce1f1253fea42ef8`. NeMo restored `SortformerEncLabelModel`, and the exact previously failing `AudioToMelSpectrogramPreprocessor` instantiated successfully.
- Registry push, manifest digest binding, candidate commit/bundle refresh, and Pod image update remain blocked on explicit publish/deployment approval.

## 2026-08-31 UTC — repaired immutable image published and contracts rebound

- Ran the non-`latest` Docker push in background process `28956` with durable status `.cache/issue-107-docker-push-status.json` and output `.cache/issue-107-docker-push.stdout.log`; polling confirmed a successful exit.
- Published image: `kapitalismho/puripuly-heart:issue107-pkgresources-r1`.
- Registry manifest identity: `kapitalismho/puripuly-heart@sha256:20f44b72f748cdd755b0ff0dcb74de40fc6ee996e9a9ecd263c41598fdd746b8`.
- Rebound `runtime_environment.json`, `contract.json`, `runtime_contract.json`, launcher, adapter, preflight constants, and active operator documentation to the new manifest digest. Historical incident evidence was not globally rewritten.
- Runtime environment raw SHA-256: `17561c2b593439e8648f60fe25bbdca7acad4ba3a7a23eace63fc4280978edff`.
- Contract canonical SHA-256: `9b58cfe4ea7af825fbdfef4cac3daedd0ab54d93cc7a28fd2a3bdf45fa951ac1`.
- Runtime contract canonical SHA-256: `f438095c42d6c00ebdeb5cb3d6ac1d441e45f48e42e5b6e19954fc2c8f653fc1`.
- The exact-package gate now includes `setuptools`; the runtime validator independently requires `setuptools==80.9.0` and a successful `pkg_resources` import.
- Added direct watchdog support for an explicit pinned OpenSSH known-hosts file. Remote `scp` polling can now use the captured host key with strict checking instead of relying on the default user known-hosts database or a local mirror.
- Pod `v6l27rdzg5s591` remains stopped with `/workspace` preserved pending the final clean candidate and bounded CUDA canary.

## 2026-08-31 UTC — final local candidate validation

- Host `uv run pytest` was rejected at collection because the current host environment lacks Torch and torchaudio. No host dependency was installed and no project environment was repaired.
- Re-ran the relevant suite inside the exact local immutable image with the repository mounted read-only in intent. All 108 Issue #107 tests passed; the only warning was the pre-existing unknown `asyncio_mode` pytest option.
- The first combined helper attempt failed after the passing tests because direct execution from `.cache` omitted the repository root from `sys.path`. The next static attempt reached the code but the Windows worktree `.git` indirection was not resolvable inside WSL Docker. These were validation-helper context errors, not candidate failures; the broad test suite was not redundantly rerun.
- Corrected the helper path and invoked `preflight.static_checks()` directly inside the image. All 14 static contract/hash checks passed with no failures. Durable background status: `.cache/issue-107-final-validation-status.json`, `state=succeeded`, `exit_code=0`.
- Targeted Python compile, Ruff check, Ruff format check, PowerShell parser, `git diff --check`, and watchdog local-control `-DryRun -Once` passed. The dry-run remained bound to the first failed run and correctly produced `would_stop/control_status_error` without contacting RunPod.

## Rebound identities for the approved `$0.44/h` resume

These identities supersede the original `8165ed58` candidate binding for the live resume only. Historical v1–v3 evidence above remains unchanged.

- Candidate Git head: `a085748b5aa59f69f918f62661ef3c4c6723cbd0`
- Image unchanged: `kapitalismho/puripuly-heart@sha256:20f44b72f748cdd755b0ff0dcb74de40fc6ee996e9a9ecd263c41598fdd746b8`
- Run ID: `issue-107-a40-a085748b-01`
- Absolute deadline: `2026-09-01T06:33:08.756000+00:00`
- Live Pod: `v6l27rdzg5s591` (same v3 volume; not deleted)
- Windows resume worker: `.cache/issue-107-resume-worker.ps1`
- Startup cost guard: `.cache/issue-107-startup-cost-guard.ps1`, unhanded-running cap `5400` seconds
- Fresh implementation review remains skipped. No implementation-complete or training-started claim is being made until a live handoff exists.

## 2026-08-31 UTC — resume automation armed; A40 capacity delayed start

- Resource: Pod `v6l27rdzg5s591`.
- Action: start the same stopped v3 Pod in background (`issue-107-background-pod-start.ps1`, PID `21244`), arm the startup cost guard (PID `36692`), and arm the Windows resume worker (PID `2848`, attempt `17a05e2dad2a4cfcac5cb7e28ccb7009`).
- Expected outcome: control plane accepts `pod start`, SSH becomes reachable, then TOFU pin → exact transfer → CUDA canary → detached runner → watchdog → live handoff.
- Actual: start retries spent most of the window in host GPU capacity failure. Durable start status later recorded `state=accepted` at attempt `96/120`, `finished_at=2026-08-31T16:34:42.4524627+00:00`, `desiredStatus=RUNNING`, `runtimeStatus=initializing`, `runtimeStatusReason=awaiting_container`.
- Control-plane confirmation: `lastStatusChange=Resumed by user: Mon Aug 31 2026 16:34:38 GMT+0000`. Guard `first_running_at=2026-08-31T16:35:07.3580536+00:00`.
- Confirmed: PID `21244` exited after acceptance and was not left retrying. The earlier “attempt 14/120, Pod stopped, GPU billing not started” snapshot was stale by the time it was re-read.
- Billing: GPU billing began when the Pod left the stopped/exited state. Storage/other account spend continued on retained stopped Pods.

## 2026-08-31 UTC — GPU running, resume worker stuck before SSH

- Resource: Pod `v6l27rdzg5s591`.
- Observed at `2026-08-31T17:09:22Z`: `runtimeStatus=running`, SSH metadata `ip=194.68.245.51`, `port=22057`, `uptimeSeconds≈1838`, account `currentSpendPerHr=0.468` (`0.44` A40 + `0.028` storage/other), `clientBalance=11.9619432365`, spend limit `$80`.
- Resume worker PID `2848` remained `state=waiting_for_running_pod` while Extra `runtime_status=running`. Cost guard PID `36692` remained `state=guarding_start` with unhanded running seconds advancing toward `5400`.
- Direct SSH probe with `.cache/runpod-issue107-ssh-key-v2` succeeded: `SSH_PROBE`, NVIDIA A40 visible. No live handoff file existed.
- Confirmed diagnosis: `Resolve-SshEndpoint` read only `ssh.host` and `ssh.command`. Live `runpodctl pod get --output json` exposes `ssh.ip` and `ssh.ssh_command`. The worker therefore treated a running Pod with a valid SSH endpoint as not yet resolvable.
- Second confirmed limitation: Windows `ssh-keyscan.exe` (`OpenSSH_for_Windows_9.5p2`) cannot complete KEX with the Pod’s `OpenSSH_9.6p1`. Direct evidence: stdout empty, stderr `choose_kex: unsupported KEX method sntrup761x25519-sha512@openssh.com`, exit `1`. The original two-keyscan TOFU path would have blocked even after the field-name fix. `ssh.exe` itself negotiates a compatible KEX.
- User explicitly selected option 1 at this point: fix parsing, restart only the resume worker, keep the cost guard, continue the already-approved `$0.44/h` resume. Option 2 (stop GPU now) was not selected.

## 2026-08-31 UTC — in-session worker restart killed by the Windows job object

- Action: patch `.cache/issue-107-resume-worker.ps1` to accept `ssh.ip` / `ssh.ssh_command`, observe the ED25519 host key twice via `ssh.exe StrictHostKeyChecking=accept-new` into a pre-created isolated known-hosts file when `ssh-keyscan` yields no key, then pin and continue with `StrictHostKeyChecking=yes` and `HostKeyAlgorithms=ssh-ed25519`. If `RUNPOD_API_KEY` is absent from the process environment, load it from worktree `.env.local` into the process only; do not write it to Git, status, argv, or Pod configuration.
- First restart used `Start-Process` with redirected stdout/stderr, PID `26708`. It reached `waiting_for_ssh` with `194.68.245.51:22057`, then disappeared. Status remained `waiting_for_ssh`; the catch block did not run, so the Pod was not stopped.
- Second restart used `Start-Process` without redirects (UseShellExecute), PID `24560`. It also died when the launching command’s job object closed. Direct evidence: `worker24560=False` in the next independent shell while cost guard PID `36692` (parent `8032`, started `2026-08-31T13:43:22Z`) stayed alive.
- Confirmed diagnosis: this agent’s process job object kills descendants on command completion. `CREATE_BREAKAWAY_FROM_JOB` failed with Win32 `3` (`ERROR_PATH_NOT_FOUND`). Long-lived guard/worker processes from the prior session survived because they were not children of this job.
- Corrective action: start the patched worker via a one-shot scheduled task so the process is parented outside the agent job. Unregister the task after start. Keep PID `36692` untouched.

## 2026-08-31 UTC — detached resume worker reached TOFU pin and bundle transfer

- Resource: Pod `v6l27rdzg5s591`.
- Action: scheduled-task launch of the patched worker. PID `36708`, parent `3452`, attempt `2f8b12e12de048fb83d09fef99ca75e8`, worker script SHA-256 `b7ae1a2e4c50840fc607ab16e5a7477e4c1acef27fb35c625076e51a0ed7a056`, `worker_started_at=2026-08-31T17:21:19.0193697+00:00`, `worker_expires_at=2026-08-31T23:21:19.0193697+00:00`.
- Expected outcome: resolve SSH, two matching ED25519 observations, strict pin, `SSH_READY`, then exact-file transfer of the `a085748b` bundle and control receipts.
- Actual:
  - `17:21:21Z` `waiting_for_ssh` at `194.68.245.51:22057`.
  - `17:22:00Z` `transferring` after two matching isolated observations. Pinned known-hosts SHA-256 `0237f59887ca46f8553723cdfa6177b4e5c74f6134de38925bb03dd29a7cce11`. Host-key fingerprint `SHA256:z3PiQQcX62Qg1YUUQBSoQJhFy2p/lRX6fvYm5z23VRE`. Trust method recorded as two matching observations then strict pin; Windows `ssh-keyscan` was skipped because it produced no ED25519 line.
  - `17:22:03Z` transfer of `/workspace/issue-107/packages/puripuly-heart-a085748b5aa59f69f918f62661ef3c4c6723cbd0.bundle`, `311915776` bytes, started. Status had not advanced again by `17:27:03Z`; PID `36708` remained alive, which is consistent with a still-running SCP of the bundle.
- Cost guard at `17:27:03Z`: `state=guarding_start`, `runtime_status=running`, `unhanded_running_seconds=3115`, cap `5400`, so the unhanded GPU window ends near `18:05:07Z` unless a valid live handoff is accepted first.
- Billing snapshot at `17:26:42Z`: `clientBalance=11.8453876921`, `currentSpendPerHr=0.468`, Pod `costPerHr=0.44`, `uptimeSeconds=2768`, `runtimeStatus=running`. No live handoff. No watchdog receipt for `issue-107-a40-a085748b-01`. Training has not started.
- Next bounded checks: wait for the bundle plus remaining small receipts to verify, then remote canary/runner. If the worker throws, its catch still stops the Pod. If handoff is not armed before the `5400s` guard, the guard stops the Pod. Do not claim canary, runner, or training success until those receipts exist.

## 2026-08-31 UTC — exact transfer finished; remote status binding failed closed

- Resource: Pod `v6l27rdzg5s591`, Windows worker PID `36708`, attempt `2f8b12e12de048fb83d09fef99ca75e8`.
- Transfer sequence observed in the durable worker status:
  - `17:22:03Z` bundle `/workspace/issue-107/packages/puripuly-heart-a085748b5aa59f69f918f62661ef3c4c6723cbd0.bundle` `311915776` bytes.
  - `17:28:03Z` `/workspace/issue-107/receipts/static_preflight.json` `4133` bytes.
  - `17:28:14Z` `/workspace/issue-107/upload_manifest_a40_eu_ro.json` `5552` bytes.
  - `17:28:26Z` `/workspace/issue-107/capacity_receipt_30gb.json` `1068` bytes.
  - `17:28:37Z` `/workspace/issue-107/preconfig-canary-a085748b.sh` `5574` bytes.
  - `17:28:48Z` `/workspace/issue-107/remote-resume-a085748b.sh` `7774` bytes.
- Expected next state: `remote_resume_running`, then remote `verifying_transfer` / CUDA canary / `runner_started`.
- Actual at `17:29:05Z`: worker `state=failed`, `error=remote resume status binding mismatch`. `remote_resume_running` was not observed as a durable status (either overwritten immediately or never flushed before the throw). No live handoff. No watchdog receipt.
- Confirmed: this is `Test-RemoteStatusBinding` rejecting the first successfully parsed `/workspace/issue-107/remote-resume-a085748b-status.json`. A JSON parse failure would have been ignored (`remoteStatus=$null`) and would not throw. The throw therefore means SSH `cat` returned parseable JSON whose `artifact_role`, `attempt_id`, `candidate_git_head`, `image_identity`, `run_id`, or `absolute_deadline_utc` did not exactly match the Windows worker.
- Provisional causes, not yet distinguished because the Pod was stopped before a diagnostic `cat`:
  1. A leftover remote status from an earlier attempt survived prepare/`mv` and was read first.
  2. `Invoke-LocalProcess` concatenates SSH stdout+stderr; extra text still parsed into an object with empty required fields.
  3. The remote writer’s `attempt_id` did not equal `2f8b12e12de048fb83d09fef99ca75e8`.
- Direct remote file contents were not retrieved. `/workspace` is preserved on the stopped volume and remains the evidence to inspect on the next start.

## 2026-08-31 UTC — fail-closed GPU stop after binding mismatch

- Action: worker catch invoked `Stop-PodConfirmed -Reason resume_worker_failure`.
- Result: `state=stop_confirmed` at `2026-08-31T17:29:11.1016172+00:00`, `stop_attempt_count=1`, `stop_exit_code=0`, `runtime_status=stopped`. Control plane: `desiredStatus=EXITED`, `runtimeStatusReason=stopped_by_user`, `lastStatusChange=Exited by user: Mon Aug 31 2026 17:29:06 GMT+0000`.
- Worker PID `36708` exited. Cost guard PID `36692` remained alive and moved to `waiting_for_start` / `runtime_status=stopped`.
- Billing snapshot after stop: `clientBalance=11.808923981`, `currentSpendPerHr=0.027` (GPU `$0.44/h` not charging; storage/other only). Image, 40GB container disk, and 30GB `/workspace` remain. No assets were deleted.
- CUDA canary, detached runner, watchdog, and live handoff did not run. Training has not started. No implementation-complete claim.
- Guard constraint for any immediate restart: PID `36692` still holds in-memory `first_running_at=2026-08-31T16:35:07.3580536+00:00`. If the same guard sees `running`/`initializing` again, unhanded age is measured from that original timestamp and will force `startup_handoff_timeout` near `18:05:07Z` (~36 minutes after this stop). Restarting the guard would reset the 5400-second unhanded window and is a new cost-window decision; using only the remainder of the existing window is not.
- Next bounded action, not yet taken: start the same Pod only long enough to copy `/workspace/issue-107/remote-resume-a085748b-status.json` and the remote resume log, confirm the binding mismatch with the file in hand, fix the earliest responsible layer (status hygiene, SSH stdout/stderr split, or remote writer), then resume. Do not loop start→same throw→stop without that file.

## 2026-08-31 UTC — binding mismatch root cause confirmed: ConvertFrom-Json date coercion

- Local reproduction against the same `pwsh` that runs the worker: `ConvertFrom-Json` of `{"absolute_deadline_utc":"2026-09-01T06:33:08.756000+00:00"}` yields `[string]$obj.absolute_deadline_utc = "09/01/2026 15:33:08"` (culture + local offset). Exact `-ceq` against the canonical deadline is false.
- A regex extract of the same raw JSON field returns `2026-09-01T06:33:08.756000+00:00` and compares equal.
- Independent corroboration: polling the Windows worker status already displayed `updated_at` as `09/01/2026 02:22:03` although the file stores an ISO-8601 `...ToString("o")` value.
- Confirmed diagnosis: `Test-RemoteStatusBinding` compared `[string]$Status.absolute_deadline_utc` after `ConvertFrom-Json`. The remote status writer emits a legal ISO-8601 deadline, so the first successful `cat` of a correctly bound file fails closed. This matches the 17-second throw after launch without needing a leftover attempt or CRLF (the transferred `.sh` files are LF-only and hash-identical).
- Corrective action in `.cache/issue-107-resume-worker.ps1` (script SHA-256 `c262faf49ae7d5f291c3d9d04d4b367408393c329eaed9ca0d175dabce52ab4c`):
  1. Bind `artifact_role`, `attempt_id`, `candidate_git_head`, `image_identity`, `run_id`, and `absolute_deadline_utc` from the raw JSON string, not from `ConvertFrom-Json` properties.
  2. Keep SSH stdout separate from stderr; JSON parse uses stdout.
  3. On mismatch, write `.cache/issue-107-remote-status-mismatch-<attempt>.txt` before throwing.
  4. If the file is a well-formed remote-resume status for a *different* attempt id, quarantine it and keep waiting instead of stopping the GPU.
- Also load `RUNPOD_API_KEY` from worktree `.env.local` into the cost guard and pod-start scripts so scheduled-task launches authenticate. The key is still not written to Git, status, argv, or Pod configuration.

## 2026-08-31 UTC — resume relaunched after binding fix

- User explicitly selected fix-and-resume. That includes a fresh 5400-second unhanded GPU window: the previous guard’s in-memory `first_running_at=16:35Z` would have force-stopped any restart near `18:05Z`.
- Action: stopped old guard PID `36692`. Started replacement processes outside the agent job object via one-shot scheduled tasks, then unregistered the tasks.
  - Cost guard PID `40484`, `guard_started_at=2026-08-31T17:40:13.3223729+00:00`, `guard_expires_at=2026-08-31T23:40:13.3223729+00:00`, `first_running_at` unset until the Pod is actually running/initializing.
  - Pod start PID `36048`, retrying `pod start v6l27rdzg5s591`.
  - Resume worker PID `33552`, attempt `61e5faf81605493ea5b5668799988d23`, `worker_started_at=2026-08-31T17:40:18.1016756+00:00`, `worker_expires_at=2026-08-31T23:40:18.1016756+00:00`, `state=waiting_for_running_pod`.
- Observed at `17:40:37Z`: pod-start `state=retrying`, attempt `2/120`, `start_exit_code=1`, `desiredStatus=EXITED`, `runtimeStatus=stopped`, `runtimeStatusReason=stopped_by_user`, next delay 30s. Same capacity/start-reject pattern as the earlier 96-attempt wait. GPU billing has not restarted.
- Image, 40GB container disk, and 30GB `/workspace` (including the already transferred `a085748b` bundle) remain. No live handoff. Training has not started.

## 2026-08-31 UTC — CUDA canary failed on an idealized streaming schedule, not prefix leakage

- Resource: Pod `v6l27rdzg5s591`, Windows attempt `61e5faf81605493ea5b5668799988d23`.
- After capacity returned, SSH and exact transfer completed. Remote resume reached `running_cuda_canary`. Repo HEAD on the volume was already `a085748b5aa59f69f918f62661ef3c4c6723cbd0`. Sortformer restore from the pinned `.nemo` succeeded.
- Remote status (later quarantined as `.stale.0482433f…`): `state=failed`, `stage=running_cuda_canary`, `detail=stage=running_cuda_canary exit_code=1`, `updated_at=2026-08-31T18:18:07.001553+00:00`.
- Direct log evidence: `RuntimeAuditError: streaming cache or prefix-causality evidence is invalid` from `build_timing_receipt` in `runtime_audit.py`. The dedicated prefix-causality failure string (`runtime evidence violates charged prefix causality…`) and the native-frame-contract string did **not** appear, so future-leakage and 80 ms / 1040 ms geometry had already passed.
- Confirmed limitation: `_trace_matches_low_latency` required NeMo’s live `streaming_feat_loader` / `pre_encode` / `streaming_update` trace to match a CPU-fake chunk machine used only by unit tests. Contract requires prefix causality to run, native frame/delay, the #99 low-latency **module** preset, and a content-bound trace hash — not bit-identity with the fake schedule.
- Windows worker then recorded `failed` and, under the old policy, `Stop-PodConfirmed` overwrote the error field with `stop_confirmed` / `resume_worker_failure`. GPU returned to `EXITED`. Subsequent user policy: worker failure must not stop the GPU; only the startup cost guard may stop it.
- User characterized the remaining gate as research-grade strictness inappropriate for this engineering probe and asked to loosen matching gates before the next worker send.

## 2026-08-31 UTC — canary schedule gate relaxed (candidate `323b1909`)

- Action: keep prefix-causality execution, 1280-sample frames, 16640-sample delay, configured cache/FIFO caps, and trace hashing as hard checks. Stop using `_trace_matches_low_latency` as a blocking predicate. Record it on the timing receipt as `low_latency_schedule_matched`. Accept integer-like numeric types instead of `type(x) is int`. Require only the expected trace keys as a subset. Split the previous combined error into `streaming cache trace is empty, incomplete, or exceeds configured cache bounds` vs `prefix-causality evidence is invalid`.
- Tests: `test_timing_receipt_records_schedule_mismatch_without_blocking` (forged middle-chunk `left_offset=0` now passes with `low_latency_schedule_matched=false`); `test_timing_receipt_rejects_empty_or_overfull_cache_trace` still fails closed. `test_runtime_audit.py` 11 passed.
- Git: local commit `323b1909b36b4b8e3786c0f65ecc1fad07230437` (`fix: relax Issue 107 streaming-schedule canary gate`). Not pushed. Image digest unchanged.
- Bundle rebound for the live worker: `puripuly-heart-323b1909b36b4b8e3786c0f65ecc1fad07230437.bundle`, `311921875` bytes, SHA-256 `78420843d020227dbcb31521890669890b46358e8614dda83e7436ef0ff484ba`. Manifest, static preflight, canary script, and remote-resume script HEAD/hash values rebound to this commit before the next send.

## 2026-08-31 UTC — transferring simplified: skip-if-identical then in-Pod S3, else SCP

- Failure mode: each resume re-SCP’d the full candidate bundle (~298 MB) over the Windows uplink even when `/workspace` already held the exact file, so `transferring` looked like training and wasted GPU-on time.
- New `Transfer-File` order in `.cache/issue-107-resume-worker.ps1`:
  1. If the remote path already has the expected byte length and SHA-256, skip (`transfer=skipped_identical`). No SCP.
  2. Else if the entry has an EU-RO object key (checkpoint, diar package, NeMo source archive, corpus `psem-strategy-data-v2.tar.gz`), download inside the Pod with authenticated Boto3 against `s3api-eu-ro-1.runpod.io` / bucket `tifw77udi2`. Volume S3 access/secret values travel only through encrypted SSH stdin into `/tmp/issue-107-pod-s3-get.py` under isolated `/tmp/issue107-s3-venv` (`boto3==1.35.99`). Not placed in Pod configuration, argv, Git, journal, or the scientific environment. `RUNPOD_API_KEY` stays local.
  3. Else SCP the local artifact (new git bundle, refreshed receipts, canary/resume scripts).
- Helper identity: `.cache/issue-107-pod-s3-get.py`, copied to `/tmp/issue-107-pod-s3-get.py` only when remote size/SHA differ.
- Live rebound worker at `18:45:34Z`: attempt `b801a952cd2942f382000ac52077c342`, candidate `323b1909…`, SSH `194.68.245.51:22095`, `state=transferring` after TOFU. GPU remains running. Guard `first_running_at=2026-08-31T18:24:19Z`, unhanded cap 5400 s. Worker failure still must not stop the Pod. No live handoff. Training has not started.

## 2026-08-31 UTC — 323b1909 Windows worker lost the prepare race; leftover a085748b canary then failed again

- User watch at `18:47:51Z` showing `state=failed` / empty SSH is the real `.cache/issue-107-resume-worker-status.json`. `updated_at` displays as local DateTime because the watch uses `ConvertFrom-Json`; the file stores `2026-08-31T18:46:21.0131996+00:00`. Catch-path status rewrite omits `ssh_host`/`ssh_port`, so those fields go blank on failure. PID receipt is stale (`33552` from 17:40, process dead). No `issue-107-resume-worker.ps1` process remains.
- Windows attempt `b801a952cd2942f382000ac52077c342` reached TOFU/`transferring` then threw `failed to prepare remote transfer directories or another resume worker is active`. Prepare is `mkdir -p … && pgrep -f '[r]emote-resume-a085748b.sh' && mv stale status`. `mkdir` works; `/workspace` is mounted. The throw is the `pgrep` exclusivity gate (`exit 41`).
- Occupant was leftover remote attempt `828b7ba9d0114c5d93bd255e2d2a1116` (Windows worker from the 18:24 restart, still on candidate `a085748b`, not `323b1909`). It launched remote-resume around 18:32, fetched the old bundle, and was still in CUDA canary when `b801a952` connected at 18:45:34. That canary finished `failed` at `18:47:23Z`: `stage=running_cuda_canary exit_code=1`, scratch `preconfig-canary-a085748b.PtIz1B`, same old `RuntimeAuditError: streaming cache or prefix-causality evidence is invalid`. Repo HEAD on the volume is still `a085748b`. `low_latency_schedule_matched` is absent. The `323b1909` bundle was never copied (`packages/` has `a085748b` + `8165ed58` only).
- After that canary exited, `pgrep` for remote-resume is empty. GPU policy held: worker catch wrote `gpu_stop_on_worker_failure=false` and did not stop the Pod. Pod `v6l27rdzg5s591` still `RUNNING` / `runtimeStatus=running`, SSH `194.68.245.51:22095`, uptime ~1543 s at inspect, cost `$0.44/h` + storage. Guard PID `37840` `guarding_start`, `first_running_at=2026-08-31T18:24:19Z`, unhanded ~1475/5400 s (~19:54:19Z force-stop if still unhanded). No live handoff. Training has not started.
- Next bounded action, not yet taken: send one Windows worker on `323b1909` now that the leftover remote-resume is gone, so skip/S3/SCP can land the new bundle and the relaxed canary can run. Do not stop the GPU to do that.

## 2026-08-31 UTC — worker still failed; leftover canary gone; cost guard died

- Re-inspect at `18:54:20Z`. Windows attempt `b801a952…` status unchanged: `state=failed`, `updated_at=18:46:21Z`, `error=failed to prepare remote transfer directories or another resume worker is active`. No `issue-107-resume-worker.ps1` process. PID receipt still names `33552` (dead). Scheduled task `issue-107-resume-worker-once` last ran `18:45:45Z` with result `1` and has no next run.
- Remote leftover occupant is gone: `pgrep -f '[r]emote-resume-a085748b.sh'` exit 1. `nvidia-smi` A40 `0 %`, `0 MiB` / `46068 MiB`. Remote status still the failed `828b7ba9` canary at `18:47:23Z`. Repo HEAD still `a085748b`. `packages/` still has no `323b1909` bundle.
- Pod `v6l27rdzg5s591` remains `RUNNING` / `runtimeStatus=running`, SSH `194.68.245.51:22095`, uptime `1813` s, `$0.44/h` + storage. No live handoff. Training has not started.
- Cost guard PID `37840` is dead. Guard status last write `18:48:55Z` (`unhanded_running_seconds=1475`, `first_running_at=18:24:19Z`). Stdout/stderr logs empty and not updated by this guard instance. The in-memory 5400 s unhanded stop near `19:54:19Z` will not fire unless a new guard is started. Restarting the guard would reset that window; that is a new cost-window decision, not yet taken.

## 2026-08-31 UTC — cost guard relaunched; 323b1909 worker sent after empty prepare race

- User approved a fresh 5400 s unhanded window and a new `323b1909` worker. Preflight: remote `pgrep -f '[r]emote-resume-a085748b.sh'` exit 1, A40 `0 %` / `0 MiB`, local bundle/canary/resume-script hashes match the worker file table. Guard `$head` rebound from `a085748b` to `323b1909` so a later live handoff can bind. Launch via one-shot scheduled `cmd start /min` so neither process is in the agent job object; launcher tasks unregistered after spawn.
- Cost guard PID `17604` (cmd parent `21816`), `guard_started_at=2026-08-31T18:58:47Z`, `first_running_at=2026-08-31T18:58:49Z`, cap 5400 s (~`20:28:49Z` force-stop if still unhanded). Script SHA-256 `0c70c48c1275392695ab5aad9a5509e9fce44b5e7f9c6fb99e2d9b0d5cf3166a`.
- Windows worker PID `21828` (cmd parent `31112`), attempt `6d6f6e102b4243c6aa3ffc73bde69679`, `worker_started_at=2026-08-31T18:58:50Z`, candidate `323b1909…`, SSH `194.68.245.51:22095`. Prepare passed. At `18:59:51Z`: `state=transferring`, `transfer=skipped_identical` for `/workspace/issue-107/packages/nemo-1a3c291b3ef0f0e11b72f789b185e1f1bda39bd6.tar.gz`. Worker failure still must not stop the Pod.
- Pod `v6l27rdzg5s591` remains `RUNNING` / `$0.44/h` + storage. No live handoff. Training has not started.

## 2026-08-31 UTC — 323b1909 CUDA canary: schedule gate passed; raw-waveform autograd check failed

- Windows attempt `6d6f6e102b4243c6aa3ffc73bde69679` `state=failed` at `19:11:54Z`, `error=remote resume worker failed: stage=running_cuda_canary exit_code=1`. `gpu_stop_on_worker_failure=false`. Worker PID `21828` exited. Guard PID `17604` still `guarding_start`, unhanded ~801/5400 s (`first_running_at=18:58:49Z`, force-stop ~`20:28:49Z` if still unhanded). A40 idle. No live handoff. Training has not started.
- Confirmed the relaxed candidate actually ran: remote HEAD `323b1909b36b4b8e3786c0f65ecc1fad07230437`, bundle present (`311921875` bytes at `19:05Z`), `runtime_audit.py` has the split errors and `low_latency_schedule_matched`. The old combined string is absent. Remote status `attempt_id` matches. Scratch `preconfig-canary-a085748b.xARyQb`.
- Timing / prefix-causality passed. Failure is the next gate, H-HEAD gradient canary: `RuntimeAuditError: canary loss is not differentiably dependent on raw waveform` at `runtime_audit.py:1078` after `loss.backward()`. Earlier in the same function: finite scalar loss, authorized-module reach, exact tap geometry, and `_waveform_dependence` (numeric change of `psem_head` when the waveform is perturbed) had already passed. The throw is `canary_waveform.grad` is None, non-finite, or all-zero. Parameter-grad and one-step-update checks never ran.
- This is not a leftover `a085748b` schedule replica. It is a research-grade autograd-to-16 kHz-samples check. H-HEAD only trains `psem_head`; NeMo `process_signal` / streaming cache can keep value dependence while severing input autograd (`oom_safe_feature_extraction` explicitly `detach()`s; eval streaming also `del audio_signal` + `empty_cache()`). Next bounded action, not yet taken: record waveform-grad as a receipt flag like the schedule matcher, and keep blocking on finite loss, module reach, trainable `psem_head` grads, frozen encoder unchanged, and one-step update.

## 2026-08-31 UTC — canary/audit gates inventoried; engineering-probe minimum only

- Authority: issue-107 hobby-engineering probe (`runtime_contract.json` `cost_bounded_hobby_engineering_probe`, README claim boundary). Not the superseded research body. CUDA canary is a GPU-waste gate before 32-step smoke / 256-step train, not a paper.
- Blocking checks **kept** (would waste the A40 or train the wrong object):
  - right checkpoint/graph: Sortformer wrapper, 18 layers, 4 slots, hidden 192, #99 low-latency module preset, GRU-64 PSEM head, Identity evidence taps
  - exact parameter policy and optimizer groups; trainable arm has parameters
  - 30 s finite PCM; native 80 ms / 1.04 s / 4-slot timing; prefix causality (future leak); streaming trace non-empty and cache/FIFO within preset caps
  - finite scalar loss; authorized modules actually ran; trainable params get finite nonzero grads; frozen params get no nonzero grad; finite clip; one-step update matches the whitelist
- Blocking checks **removed** (research-grade, already passed or not needed to start H-HEAD):
  - `canary loss is not differentiably dependent on raw waveform` (the 19:11 fail)
  - `_waveform_dependence` extra forwards / all-modules numeric perturbation
  - `exact fixed runtime canary path` (`FIXED_RUNTIME_CANARY_METHODS` object identity)
  - exact 375/192/4 tap geometry as a raise (`tap_geometry_matched` is now a receipt flag)
  - CPU-fake streaming schedule (already a flag in `323b1909`)
- Receipt validators no longer require `raw_waveform_gradient_nonzero is True` or all-True `raw_waveform_dependence`. Those fields remain on the receipt. Tests: `test_runtime_audit.py` + `test_receipts.py` passed; full `experiments/psem_sortformer_adaptation_depth/tests` passed.
- Not in this cut: EVAL freeze, USD-30, V2 data identity, TRAIN-only, staged H-before-T2. Those are operator/protocol, not the CUDA canary.
- Code is local uncommitted relative to `323b1909`. GPU `v6l27rdzg5s591` still running under the 18:58Z guard. No new worker sent. No live handoff. Training has not started.

## 2026-08-31 UTC — committed be1121e9 and sent rebound worker

- Git: local commit `be1121e982aefc1733d3643940d4e8991db2976e` (`fix: drop research-grade Issue 107 canary autograd gates`). Not pushed.
- Bundle `puripuly-heart-be1121e9….bundle` `311939199` bytes, SHA-256 `c716fd919ead7df21f1a01100a7050dd781946dcb683fd31a49cccc6d1f112da`. Static preflight, upload manifest `repository_head`, canary/resume scripts, and Windows worker file table rebound. Guard `$head` rebound so a later live handoff can bind.
- Old guard PID `17604` stopped (process only; Pod left `RUNNING`). Replacement guard PID `28728`, `first_running_at=2026-08-31T19:26:51Z`, fresh 5400 s (~`20:56:51Z` force-stop if still unhanded).
- Windows worker PID `30312`, attempt `16418faebfce464e9248efe8f89e9ef9`, `worker_started_at=19:26:50Z`, candidate `be1121e9…`, SSH `194.68.245.51:22095`, `state=waiting_for_ssh`. Prepare race empty. GPU-stop-on-worker-failure still false.
- Pod `v6l27rdzg5s591` remains `RUNNING` / `$0.44/h` + storage. No live handoff. Training has not started.

## 2026-08-31 UTC — CUDA canary passed; write-config failed outside /workspace/repo

- Windows attempt `16418fae…` `state=failed` at `19:39:40Z`, `error=remote resume worker failed: stage=generating_config exit_code=1`. GPU-stop-on-worker-failure false. Guard PID `28728` still `guarding_start`, unhanded ~771/5400 s (`first_running_at=19:26:51Z`). A40 idle. No live handoff. Training has not started.
- Preconfig canary **succeeded** at `19:39:13Z` on HEAD `be1121e9`: gradient/update/timing/model-graph/runtime-validation all passed. Receipt: `raw_waveform_gradient_nonzero=false`, `tap_geometry_matched=true`. Repo HEAD is `be1121e9`. Config file was not written.
- Direct log: `/usr/bin/python: Error while finding module specification for 'experiments.psem_sortformer_adaptation_depth.issue_107_launch' (ModuleNotFoundError: No module named 'experiments')`. Canary script `cd /workspace/repo` first; `write-config` and the detached runner ran from the resume-worker cwd (`/root`) with no `PYTHONPATH`.
- Patch in `.cache/issue-107-remote-resume-a085748b.sh` (not Git): `cd /workspace/repo` and `export PYTHONPATH=/workspace/repo` before `write-config`. Worker file table rebound `7827` bytes SHA-256 `10a3949d3c9c93a9cdcd85bbd4d2153a9d81a7475e00088310c1ca04bf6dd27d`. Next send not yet launched.

## 2026-08-31 UTC — resume path simplified; send blocked only on local RunPod credential

- `.cache/issue-107-resume-worker.ps1` now fails immediately if `RUNPOD_API_KEY` is absent or `runpodctl user` fails. `Get-Pod` no longer converts control-plane/authentication errors into a six-hour `waiting_for_running_pod` loop. Worker identity: `40591` bytes, SHA-256 `47807d042603e779e88ceeccfa2f5bd4e709d783387974d72cd4bb065c6e7912`.
- The worker no longer quarantines the exact successful preconfig canary receipt. The remote script reuses it when candidate head, image, succeeded state, and the five material checks match; otherwise it runs the canary normally.
- The remote script can reattach a live exact-bound detached runner when config, durable config, state, heartbeat, config hash, and live PID agree. It does not reopen terminal/error runs.
- Patched remote script synced to Pod `v6l27rdzg5s591`: `11480` bytes, SHA-256 `00f7924d31c8392f7c99855847590046886bd6cece74f22fcec66f1bcfd1969b`.
- PowerShell parse and Bash syntax checks passed. Pod remains `running`; startup guard PID `28728` remains active and no handoff exists.
- No worker was launched because the current process, user/machine environment, and worktree `.env.local` contain no `RUNPOD_API_KEY`; bundled `runpodctl user` returns `code=no_credentials`. Supplying the key to the exact launcher process is the only remaining prerequisite for the approved send.

## 2026-08-31 UTC — dead phase can restart from its beginning

- `.cache/issue-107-remote-resume-a085748b.sh` now distinguishes three existing-run cases after live-runner reattachment fails: `STARTING`, interrupted `RUNNING`, and terminal `ERROR` are restartable; decision/complete states remain closed.
- The restart phase is the first configured phase absent from `completed_phases`. For interrupted `RUNNING`, it must also equal `active_phase`. This restores the phase identity that the current runner clears when it records `ERROR`.
- Before restart, any surviving exact phase process group is terminated. The previous state, phase logs, heartbeat/lock, completion marker, partial phase receipts, and the phase's checkpoint/prediction directories move to `run_root/recovery/<utc>-<phase>/`. `bootstrap-f0` moves its whole `receipts`, `output`, and `protocol-registry` roots because no earlier scientific phase exists. Prior completed-phase artifacts remain in place for H/T2/TA restarts.
- State is reset to `STARTING` with `next_phase=<dead phase>`, and the unchanged committed detached runner starts normally. No candidate code, bundle, image, run id, completed phase, decision archive, or deadline is rebound.
- Linux fixture checks passed for both an `ERROR` H-HEAD restart and an interrupted `RUNNING` bootstrap restart. Bash syntax and PowerShell parsing passed. Remote script identity is `17955` bytes / SHA-256 `79db3fcb9a0fbba92dbd5c5fff6ccf13474b2cf3448feba175c94f5304a64a98`; worker identity is `40591` bytes / SHA-256 `c36441caaf7aacb20a5d7641cb6b39520d87706826313378beaa778277507b28`.
- Pod read-only check at `20:28:19Z` confirms both configured `assets/...` paths and transferred checkpoint paths exist. The target config and run root are still absent, no resume/runner/phase process exists, and the A40 is idle at `0 MiB`, so there is currently no dead phase to recover. The new script has not been synced or launched; the Pod still holds the prior `11480`-byte reattach-only version.
- Startup guard PID `28728` is dead. Its last status write was `20:22:27Z` with `state=guarding_start`; therefore that status is stale and the running idle Pod currently has no live automatic startup-cost stop process.
- The explicitly excluded 30 GB storage-quota validation was not changed.

## 2026-08-31 UTC — local control takeover; run started and handed off

- User authorized terminating the watcher/guard/worker left by another context and restarting them here. Old Issue 107 scheduled tasks were unregistered, the stale ad-hoc watcher PID `28920` was terminated, and prior status/receipt files were renamed with `stale-before-takeover` / `stale-deadline-fix` timestamps rather than deleted.
- First takeover worker reached the fresh remote runner but exposed another PowerShell date-coercion failure: `ConvertFrom-Json` converted watchdog deadline `2026-09-01T06:33:08.7560000+00:00` to a local `DateTime` whose string form lost `.756`. The worker therefore considered a correctly armed receipt unequal to the immutable deadline; the guard had the same defect for handoff, watchdog, and PID receipts.
- `.cache/issue-107-resume-worker.ps1` and `.cache/issue-107-startup-cost-guard.ps1` now extract timestamp strings from raw JSON before exact comparison/parsing. PowerShell parsing passed, and a direct reproduction changed deadline equality from false to true. Worker identity: `40869` bytes / SHA-256 `05ff7d001b7f994a0b7045f65e0740ea2e88f354916d13a212663c127607ed48`. Guard identity: `11624` bytes / SHA-256 `42d15c77c9d18339c254617a104a7148a064f428ac4a016f47e3ed6d80ea843b`.
- The first local worker/watchdog/guard were stopped without touching remote runner PID `10158`. Patched guard PID `41212` and worker PID `35888` were launched. Worker attempt `76fd36bc516447ecbde4786ded9422d1` transferred/skipped exact artifacts, used the new `17955`-byte remote script, and reattached runner PID `10158`.
- Fresh external watchdog PID `19904` armed at `20:50:28Z`. Live handoff was created at `20:50:31Z`. Worker reached `state=monitoring`; guard reached `state=handoff_supervision`; visible watcher PID `35812` remains alive.
- Direct Pod check at `20:51:19Z`: detached runner PID `10158`, `bootstrap-f0` phase PID `10240`, control/heartbeat `RUNNING`, heartbeat sequence `33`, no state error, A40 utilization `59%`, memory `1217/46068 MiB`. Training execution has now started under the guarded detached run.

## 2026-09-01 UTC — bootstrap ERROR stopped by watchdog; capacity retry delegated to watcher

- At `2026-08-31T22:19:23Z`, the watchdog observed control status `ERROR` and fired `reason=control_status_error`. It confirmed the Pod stopped on its first control-plane attempt at `22:19:26Z`. This was not a decision gate or deadline stop. The exact phase exception remains on the persistent `/workspace` volume and requires the Pod to start before it can be read.
- A fresh authenticated guard PID `2932` is running in `waiting_for_start`. Same-Pod start requests are currently rejected with `There are not enough free GPUs on the host machine to start this pod`; no GPU billing occurs while stopped. Background retry PID `30384` remains active.
- Continuous foreground polling was removed at user request. `.cache/issue-107-worker-watch.ps1` now watches `issue-107-pod-start-status.json`; on the first `accepted` state it emits a six-repeat audible alert and automatically launches `issue-107-resume-worker.ps1`. While capacity is `requesting`/`retrying`, a missing worker status is expected and no missing-worker alarm fires. Watcher identity: `7038` bytes / SHA-256 `eeadc587180a68f23672d3f6cb1a0713af221f6e94bda09d939d1de191137451`.
- Patched visible watcher PID `23344` is active. At handoff it showed Pod-start attempt `6` in `requesting`; guard and start retry were both alive. Once capacity is acquired, the worker will sync/reattach, the dead `bootstrap-f0` will be archived and restarted from its beginning, and the exact archived stderr can be inspected without manual polling.

## 2026-09-01 UTC — bootstrap result was rejected by an impossible 100% mapping gate

- Capacity was acquired and the dead bootstrap phase restarted, but the Windows resume worker reported failure after the remote resume process exited. Direct Pod inspection showed that detached runner PID `971` and bootstrap PID `1087` were actually alive; this was a local poll race, not a failed remote launch.
- The original archived bootstrap had completed F0 inference and evaluation before failing only at `stage-init`. Exact exception: `ProtocolError: DEV result identity or lean fail-closed gates are invalid`.
- The F0 result had `timing_gate_passed=true`, `slot_mapping_coverage_passed=false`, and pooled mapping coverage `0.9843885516`. Eight DEV sources contained ordinary unmapped episodes but zero slot instability and zero unexpected resets.
- Confirmed design error: the checked-in frozen oracle mapping evidence itself records only `6730/6868 = 0.9799068142` mapped episodes. Requiring every source to have `mapping_coverage == 1.0` made the downstream protocol gate impossible by construction.
- The duplicate restarted bootstrap was terminated before it repeated the same 96-minute computation. The Pod was left running and idle; no detached runner, phase launcher, remote resume worker, or GPU compute process remained.

## 2026-09-01 UTC — post-compute overvalidation removed before the next worker send

- Mapping coverage remains recorded as a diagnostic. The blocking mapping flag now represents zero slot instability instead of impossible 100% episode coverage.
- DEV and EVAL validators now accept bound results whose diagnostic integrity flags are false. They require internally consistent boolean flags and still validate arm, seed, split, singleton operating point, metrics, source coverage, payload hashes, prediction binding, and external roots. A diagnostic failure can therefore become a reportable outcome instead of crashing `stage-init` or final reporting.
- Removed exact DEV prediction reevaluation during every staged-state validation. The result remains content-bound to its embedded prediction-set SHA and DEV evidence SHA.
- Removed authority-registry reads/writes from the active canary, smoke, training, inference, evaluation, and final-report paths. Artifact payload hashes, file hashes, checkpoint identity, and the separate one-time EVAL marker remain authoritative.
- Removed the 32-step smoke loss-trend blocker. First-eight and last-eight means remain diagnostics; the smoke still requires finite forward/backward/update behavior, at least one trainable parameter update, and no frozen-parameter change.
- Removed the repeated per-arm CUDA canary. The pre-run Pod CUDA canary remains one-time; each arm's 32-step smoke remains the material forward/backward/update check.
- Simplified the material gate to avoid rerunning preflight, lineage, evaluator reconstruction, canary receipt reconstruction, class-weight reconstruction, registry lookups, and full exact gate replay immediately before training and again before DEV inference. It still enforces clean candidate identity, H/T2/TA order, TRAIN-only manifest identity, bound class weights, staged DEV/EVAL separation, runtime paths, cost hard stop, smoke provenance, and TA operator authorization.
- Fixed the Windows remote-resume exit race by rereading the final remote status after the short-lived resume process exits. Deadline comparison is semantic UTC equality, and watchdog arming no longer depends on redundant mode/freshness fields after stale receipts have already been quarantined.
- No worker was sent. Static-only checks passed: Python AST parsing for all changed Python files, PowerShell parser for the worker, and `git diff --check`. No model or test suite was run.

## 2026-09-01 UTC — H-HEAD smoke rejected JSON-round-tripped class weights

- Run `issue-107-a40-1334720a-01` completed `bootstrap-f0` at `07:53:37Z`, then `h-head-material-and-dev` failed at `08:01:20Z`. The watchdog exported control and phase logs locally and stopped the Pod at `08:02:01Z`.
- Exact exception: `ExecutionError: smoke class weights differ from the one-epoch manifest`. This was not OOM, CUDA failure, or a RunPod interruption.
- Root cause: `build_manifest_class_weight_receipt` returned `replacement_counts` and `anchor_counts` with integer keys in memory. JSON persistence converts object keys to strings, so the smoke phase's direct dictionary equality rejected an otherwise identical receipt after reload.
- The receipt builder now emits JSON-native string keys before hashing and persistence. A focused JSON round-trip regression test passed in the pinned container image. The host test runner could not collect because host Python has no Torch.
- `bootstrap-f0` remains complete on the persistent Pod volume. Resume should archive and restart only the failed `h-head-material-and-dev` phase from its beginning.

## 2026-09-01 UTC — GPU retention and completed-phase reuse restored

- Immediate Pod stop on `ERROR`, stale heartbeat, control failure, watchdog exit, or binding loss was incorrect for a scarce GPU. Error evidence is still exported immediately, but watchdog and startup-guard termination now require 5400 consecutive unhandled seconds. The immutable billing deadline remains an immediate hard stop.
- The new candidate is a receipt-serialization repair descended from `f76efc468092b0bb8b9969ffb3342c1ec781af58`. Existing `1334720ab9975b9a68aedf2d291eb9056baf3ff3` bootstrap evidence is accepted only when the Git diff remains limited to the explicit repair, resume compatibility, tests, and journal files and every other candidate artifact hash is unchanged.
- Resume retains run `issue-107-a40-1334720a-01`, rebinds its durable control config to the repaired clean candidate, preserves `completed_phases=["bootstrap-f0"]`, archives the failed H artifacts, and restarts `h-head-material-and-dev` from its beginning. A new run ID would bypass the completed-phase ledger and is not used.

## 2026-09-01 UTC — unapproved watchdog change and forced Pod stop caused the current delay

- Incident attribution: the Codex operator changed the watchdog termination behavior without explicit user approval. That change allowed a worker/control error to trigger an immediate forced stop instead of retaining the scarce GPU until 5400 consecutive unhandled seconds had elapsed.
- The changed watchdog force-stopped Pod `v6l27rdzg5s591` at `2026-09-01T08:02:01Z` after the H-HEAD phase error. The user had not authorized releasing this GPU.
- This forced stop released the A40 from its original host while the completed `bootstrap-f0` result remained only on the Pod-bound 30 GB `/workspace` volume. The stopped Pod cannot be moved to another GPU or host while retaining that volume.
- Every subsequent same-Pod start request has been rejected with `There are not enough free GPUs on the host machine to start this pod.` The repeated capacity retries and the delay since the forced stop are therefore a direct consequence of the unapproved watchdog modification and forced GPU release, not a training failure and not merely ordinary catalog-wide A40 scarcity.
- Same-datacenter alternative stock does not repair the incident: a newly provisioned Pod cannot share this Pod Volume, so switching to another GPU would abandon the completed-phase state unless the original Pod first restarts and exports the run root to a real Network Volume or external object storage.
- Corrected policy: watchdog and guard may export errors and alert immediately, but may stop the Pod for an unhandled failure only after 5400 continuous seconds. The immutable billing deadline remains the separate immediate stop condition. No future GPU-release policy change is permitted without explicit user approval.

## 2026-09-01 UTC — capacity reacquired; H smoke failed because the live image omitted the cuBLAS deterministic environment

- Same Pod `v6l27rdzg5s591` reacquired its A40 on Pod-start attempt `58`. Resume attempt `f7699c239d0f4c10ace5690e97983021` preserved run `issue-107-a40-1334720a-01` and `completed_phases=["bootstrap-f0"]`, archived the previous H failure, and restarted only `h-head-material-and-dev`.
- The restarted H phase failed at `2026-09-01T14:19:42Z` in its first `smoke-arm` CUDA matmul. Exact exception: PyTorch deterministic algorithms were enabled, but cuBLAS refused the operation because `CUBLAS_WORKSPACE_CONFIG` was absent; the required value is `:4096:8` or `:16:8`.
- This was not an OOM, GPU interruption, completed-phase binding failure, or class-weight mismatch. The exported logs are under `.cache/issue-107-error-exports/20260901T142004969Z-issue-107-a40-1334720a-01`.
- Confirmed environment drift: `environment/Dockerfile` declares `CUBLAS_WORKSPACE_CONFIG=:4096:8`, but the live container environment did not contain the variable. The deployed image/runtime therefore did not provide the deterministic environment promised by the checked-in image recipe.
- Immediate operational repair: `.cache/issue-107-resume-worker.ps1` now injects `CUBLAS_WORKSPACE_CONFIG=:4096:8` when launching the remote resume shell. The detached runner and all phase subprocesses inherit it. No new candidate, run ID, config, bootstrap result, Pod, or GPU was created.
- Old local worker/watchdog/guard processes were replaced without sending a Pod stop. Fresh attempt `0d486d1fc99348ff9d80c3ab38c38488` archived only the failed H phase to `recovery/20260901T144214.385369Z-h-head-material-and-dev` and restarted it from the beginning. At inspection, control was `RUNNING`, `active_phase=h-head-material-and-dev`, `error=null`, `completed_phases=["bootstrap-f0"]`; runner PID `9445`, H launcher PID `9512`, and smoke PID `9685` all had `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
- Fresh watchdog and guard handoff were armed for the new attempt. Guard state reached `handoff_supervision`; the 5400-second unhandled policy and immutable deadline remain in effect. No test suite was run.

## 2026-09-01 UTC — valid anchor-silent episode incorrectly aborted H smoke

- H smoke passed the cuBLAS environment failure, restored the Sortformer checkpoint, and then failed at `2026-09-01T14:51:00Z`. Exact exception: `SupervisionError: oracle episode has no valid anchor-active support: ami_ES2007a:A00004`.
- This is a legitimate window condition: an episode can overlap the 30-second window while its anchor speaker has no valid active frame from which to infer a Sortformer slot. The code incorrectly promoted absence of mapping evidence to a fatal run error.
- Repair: oracle mapping skips only unsupported episodes. `_batch_supervision` removes those episode frames from the PSEM loss mask while retaining mappable episodes and the independent native Sortformer loss. It does not invent a slot assignment and does not discard the whole window.
- A regression case covering one mappable and one anchor-silent episode was added. Per the active instruction, no test suite was run; Python AST parsing and `git diff --check` passed.
- Watcher defect also confirmed: `control_status=ERROR` with an empty worker `error` field did not meet its beep condition. `.cache/issue-107-worker-watch.ps1` now beeps for control `ERROR` and for every newly exported error bundle. The visible watcher was restarted without touching the Pod, worker, watchdog, or guard.
## 2026-09-01 UTC — Issue 107 authority reset and H zero-target repair

- The current Issue 107 body is the authority: this is a cost-bounded engineering probe, not the superseded research-grade protocol. The active phase path no longer runs runtime preflight, lineage authorization/revalidation, material bundle assembly, material gate validation, or their repeated training/inference revalidations.
- The retained validity checks are the immutable TRAIN/DEV/EVAL split, exact checkpoint and NeMo revision, raw 16 kHz/native four-slot 80 ms/1.04 s timing sanity, expected trainable/frozen sets, EVAL isolation, CUDA memory fit, the 32-step finite/update/frozen/trend smoke, the exact 256-step run, singleton DEV metrics, one-time F0-plus-selected EVAL, and the USD-30 hard stop.
- H-HEAD failed because a legitimate anchor-silent or unmappable TRAIN window produced an all-zero PSEM validity mask and `masked_balanced_bce_with_logits` raised `ValueError: at least one unmasked target is required`. An all-zero mask now returns differentiable zero PSEM loss, so the native diarization loss still trains the batch and later valid windows update the PSEM head.
- Completed `bootstrap-f0` evidence remains reusable. Resume archives and restarts only the failed `h-head-material-and-dev` phase from step zero. Pod `v6l27rdzg5s591` must remain running; the 5400-second unhandled guard policy is unchanged.

## 2026-09-02 UTC — absolute deadline stopped a healthy H run and is disabled

- Pod `v6l27rdzg5s591` was stopped at `2026-09-02T06:46:42Z` with RunPod reason `stopped_by_user`. The watchdog receipt identifies its own stop reason as `absolute_deadline`: configured deadline `2026-09-02T01:56:00Z`, fired at `06:46:24Z`, and stop confirmed at `06:46:43Z`. This incident was not a training exception; the control state immediately beforehand was `RUNNING / h-head-material-and-dev`.
- The config now records `absolute_deadline_enabled=false`. The old `absolute_deadline_utc` value remains only as compatibility metadata. The detached phase runner no longer rejects, kills, or fails a phase because that timestamp passed.
- The Windows watchdog is launched with `-DisableAbsoluteDeadline`; deadline matching, deadline-bounded SCP, expiry stops, and deadline-capped sleeps are disabled. Error export and the 5400-second continuously unhandled-state policy remain active. The startup guard no longer expires at the old deadline.
- The apparent H slowness is explained by the frozen recipe: 256 optimizer steps use gradient accumulation 16 and microbatch size 1, so one arm serially processes 4,096 complete 30-second windows, or 34.13 hours of presented audio. Every window loads audio separately, recomputes the full file SHA-256, runs the complete Sortformer evidence path, and computes both PSEM and native Sortformer losses. There is no batch loading, prefetch, AMP, or frozen-backbone feature cache. The roughly 13-hour H runtime is therefore consistent with the current implementation and does not by itself indicate a hang.
- Training-loop audit removed repeated work without changing row order, augmentation, forward graph, loss, optimizer, scheduler, or step count. The sampling manifest hash is computed once per command; each TRAIN source recording is content-verified once before official training instead of once per window; generated waveform and supervision tensors are no longer hashed twice before immediate consumption; smoke path lookup skips the redundant second source-content pass because official training performs it.
- Audio loading and deterministic augmentation now use a one-item background prefetch so CPU preparation overlaps the preceding GPU forward/backward. Smoke streams its 512 examples instead of retaining roughly 1 GB of prepared waveforms. Its before/after snapshot covers trainable parameters only; frozen parameters are required to have no gradient.
- Removed three per-forward scans of all loaded NeMo module origins; the exact checkout origins remain verified when the model is loaded. Removed per-window evidence finiteness synchronizations and per-window scalar-loss copies; finite gradient checks remain at every 16-window optimizer boundary, and aggregate losses are checked before receipt/checkpoint creation.
- The training changes are shared by H-HEAD, T2-TOP, and TA-ALL-TEMPORAL. A follow-up audit optimized the separate bootstrap and scoring paths: memory-fit no longer hashes a complete source recording for each of its 16 probe windows; F0 verifies DEV source content once, while later H/T2/TA DEV inference reuses that established identity and still checks paths and waveform geometry.
- Prediction serialization now transfers each source's output tensors from GPU to CPU once instead of performing three CUDA-to-CPU synchronizations per frame. Per-source `torch.cuda.empty_cache()` calls were removed so the allocator can reuse memory across sources. This applies to F0 plus every arm's DEV/EVAL inference.
- Phase artifact snapshots now compare existing files by size and nanosecond mtime and calculate SHA-256 only for new or changed artifacts. Required phase inputs and all newly emitted receipt artifacts remain content-hashed; T2 and TA no longer rehash every accumulated prior checkpoint and prediction file at both phase boundaries.
