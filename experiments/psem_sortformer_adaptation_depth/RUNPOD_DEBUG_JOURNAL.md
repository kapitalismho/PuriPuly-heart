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
