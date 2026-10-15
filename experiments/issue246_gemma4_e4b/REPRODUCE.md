# Reproducing the issue246 experiment

Read [RESULTS.md](RESULTS.md) and [PROTOCOL.md](PROTOCOL.md) first. These are bounded Windows experiment drivers, not product runtime settings or a general benchmark framework. They deliberately preserve the measured9800X3D/RX7900XTX profile4/12/1 and Vulkan0. A different host/profile is a new experiment, not a validated extrapolation.

## Preconditions and isolation

- Use the product sources from immutable baseline `7a8151be6114a950f062e2430bd36f899d57e4de`, with this task directory copied into a **disposable local checkout**. Source identity constants in these task records describe that baseline; they do not automatically certify a different checkout. Do not operate an installed/live user application.
- Keep the published `results/` intact. In the disposable copy, start with a new/empty task results directory; scripts write to their own adjacent `results/`, and `requests.jsonl` is append-only. External run/app labels must also be new: server/app directories use `exist_ok=False`. Do not append a second experiment to this report's ledger.
- Use Python3.14.7 with this checkout's project dependencies, psutil7.2.2 and httpx0.28.1. Execution reused an existing interpreter; it did not install/update the live user's environment. Create any replacement environment separately.
- Provide the exact verified target/drafter plus `installed-manifest.json` from the selected public model installation. Both hashes are asserted by `prepare.py`; changing precision/quantization is outside this reproduction. Model files are never modified. App staging uses hardlinks, so the model and external root must be on the same hardlink-capable volume.
- Allow public GitHub release downloads during preparation. Inference binds only loopback; app probes have isolated user/cache/temp/HF roots, offline HF, telemetryoff, captureoff, encrypted-file secrets, synthetic process discovery, a synthetic nonworking unused-STT construction credential and a Python non-loopback network guard. No actual credential is needed. Do not enable capture/verify Soniox or publish generated secret stores/configurations.
- Leave unrelated processes/power settings untouched. Do not run another experiment in parallel. RAM safety bounds stop soaks; a stop must remain censored, not converted into a pass. Do not flush the system page cache or claim that process restart is OS-cold.

## Preparation

The following PowerShell commands replace the sanitized interpreter/root placeholders used by the actual execution. Run from the disposable repository root. `PP246_MODEL_DIR` defaults to the existing selected public installation; the optional override is useful for a separately staged copy. `PP246_THREAD_PROFILE` can point to an owned numeric-only JSON file containing `{"threads":4,"threads_batch":12,"draft_threads":1}`; absent that override, preparation narrowly reads the normal numeric profile file, not a full settings file.

```powershell
$Python = "<absolute-path-to-project-Python3.14.7>"
$E = "experiments/issue246_gemma4_e4b"
$env:PP246_EXTERNAL = "$env:LOCALAPPDATA/puripuly-heart-issue246-replay"
$env:PP246_MODEL_DIR = "<verified-selected-model-installation-directory>"
$env:PP246_THREAD_PROFILE = "<numeric-only-thread-profile-json>"
& $Python "$E/prepare.py"
& $Python "$E/prepare_followon.py"
```

Preparation asserts archive/model sizes and SHA-256, records all extracted-file hashes, inspects GGUF metadata and calls each actual binary's `--help`, `--version`, `--list-devices`. Official URLs and exact archive identities are in [identity.json](results/identity.json) /[followon-identity.json](results/followon-identity.json). Do not replace only an EXE or mix DLL packages.

Original hardware snapshot commands are retained in [environment.json](results/environment.json), including CIM CPU/OS/DIMM/video data, `powercfg /getactivescheme`, Vulkan information and observer versions. Capture a new equivalent snapshot when reproducing; the committed file describes the original host, not the next machine. Producer workflow commands/links are separately retained in [build-provenance.json](results/build-provenance.json).

## Standalone stages, serially

The stage registry is the exact bounded set in [stages.py](stages.py). Original CPU compatibility ran directly before the registry's Vulkan compatibility stage:

```powershell
& $Python "$E/measure.py" --release b10423 --backend cpu --label old-cpu-quality --phase quality --count 12
& $Python "$E/measure.py" --release b11537 --backend cpu --label new-cpu-quality --phase quality --count 12
& $Python "$E/stages.py" compatibility
& $Python "$E/stages.py" ple-tuning
& $Python "$E/stages.py" final-update
& $Python "$E/stages.py" resource-tuning
& $Python "$E/stages.py" followon-placement
& $Python "$E/parity.py"
& $Python "$E/stages.py" final-cache0
& $Python "$E/stages.py" repack-diagnostic
& $Python "$E/stages.py" soak-default
& $Python "$E/stages.py" soak-cache0
& $Python "$E/stages.py" literal-confirmation
& $Python "$E/stages.py" cache-eviction-attribution
```

Stop at any newly observed compatibility/quality failure rather than treating the command's exit status as the only gate. Inspect request terminal flags, semantic outputs, per-run resource_stop, native checkpoints/errors, counts and actual collected arms. Initial development decisions and later literal-format bridge are distinguished in the report; the literal stage is not a new tuning set.

`measure.py` imports the current product prompt assembly/message wrapper and command builder. Native argv are fully retained in each `<label>-run.json`; ephemeral ports/path placeholders are expected to differ. Extra native arguments on a direct `measure.py` invocation follow `--`, for example `-- --cache-ram 0`. The stage registry avoids ambiguous shell quoting.

CPU MTP-on/off sanity is separate temperature0. The current `parity.py` constructs a genuine non-MTP command by omitting speculative arguments. The original failed OFF preflight used generated MTP arguments plus an empty drafter override and is preserved in `new-cpu-greedy-off-{run,failure}.json` /producer text. Reproduction does not need to recreate that known bad harness construction. Likewise the original app's empty-unused-Soniox-store failure is preserved; the corrected controller deliberately seeds the authorized synthetic construction token before first start.

## Actual application, serially

These commands exercise the real current CLI/settings/translation pipeline. `app_host.py` adds task-local capture, exact process-local runtime identity and controlled unused discovery/network boundaries; maintained sources stay unchanged. Do not launch the ordinary user app with a new binary and old prefix identity.

```powershell
& $Python "$E/app_probe.py" --release b10423 --backend cpu --label app-old-cpu-guarded --cycles 2
& $Python "$E/app_probe.py" --release b11537 --backend cpu --label app-new-cpu-guarded --cycles 2 --lazy auto --foreign-cache-label app-old-cpu-guarded
& $Python "$E/app_probe.py" --release b10423 --backend vulkan --label app-old-vulkan-guarded --cycles 2
& $Python "$E/app_probe.py" --release b11537 --backend vulkan --label app-new-vulkan-guarded --cycles 2 --lazy auto --foreign-cache-label app-old-vulkan-guarded
& $Python "$E/app_probe.py" --release b11537 --backend cpu --label app-new-cpu-cache0 --cycles 3 --lazy auto --cache-ram 0 --warm-turns 30 --foreign-cache-label app-old-cpu-guarded
& $Python "$E/app_probe.py" --release b11537 --backend vulkan --label app-new-vulkan-cache0 --cycles 3 --lazy auto --cache-ram 0 --warm-turns 30 --foreign-cache-label app-old-vulkan-guarded
```

Each controller prequeries capabilities/choices/current/status, subscribes before submission, checks terminal applied plus a completed translation event correlated to child turn and instance, performs the real four-pair/three-slot sequence and fresh host/native restarts, then shuts down owned resources. `--foreign-cache-label` copies only another **own experiment** cache into an isolated root to prove rejection through the injected new identity; never point it at normal user state or intentionally restore cross-version KV. Native producer logs, slot save/restore HTTP response bodies and effective backend/slot/resource traces are decisive; an HTTP restore200 is not itself a cache hit.

## Controlled contention

The corrected CPU controls pin/account the whole owned Windows launcher/worker process trees. Initial unaccounted launcher-only controls remain in the original ledger but should not be used to establish an affinity/CPU-cost claim.

```powershell
& $Python "$E/contention.py" --release b10423 --backend cpu --label contend-old-cpu-default-accounted
& $Python "$E/contention.py" --release b11537 --backend cpu --label contend-new-cpu-cache0-accounted --lazy auto --cache-ram 0
& $Python "$E/contention.py" --release b10423 --backend vulkan --label contend-old-vulkan-default
& $Python "$E/contention.py" --release b11537 --backend vulkan --label contend-new-vulkan-cache0 --lazy auto --cache-ram 0
```

CPU load is six180s-bounded integer-loop workers, actual logical affinities0…5, normal priority; no claim of six physical cores. GPU load is a second fixed-b10423 same-Q4 Vulkan server/cache0 doing real translations. Both controllers record measured start/end, background work and owned cleanup; it is not a live VRChat/game simulation. The original exact chronology was initial CPU/GPU controls, followed by repaired CPU accounting controls; existing evidence is not rewritten to imply a different chronology.

## Reduction and publication boundary

```powershell
& $Python "$E/summarize.py"
& $Python "$E/finish_evidence.py"
```

`summarize.py` computes counts/distributions, paired results,20-request memory bins and quality parity from actual ledger rows; it does not rerun inference. `finish_evidence.py` sanitizes paths, projects unrelated audio/device/config details out of app evidence, retains one canonical template with hash references, compacts producer logs while keeping errors/warnings/checkpoints/cache state/placement, and indexes published result sizes/hashes. It canonicalizes every indexed text result to UTF-8 LF **before hashing**, and the JSON writer emits LF for the index too. Thus hashes/byte counts describe stored publication/Git blob bytes, not Windows CRLF worktree sizes. It preserves prior compaction metadata on repeated reduction. Full original producer text stays only in the owned external storage; normalized sanitized pre-compaction hashes retain their original meaning and are not hashes of raw private-path/original-line-ending files or the compact final file.

For a candidate that already contains the particular results set being checked, the publishing owner can perform the explicit read-only blob audit:

```powershell
& $Python "$E/verify_publication.py" --candidate "<commit-containing-this-results-set>"
```

It checks every indexed working file's UTF-8 LF bytes/hash/size and every corresponding actual Git blob, verifies the index total/LF declaration, and writes `PUBLICATION_AUDIT.json` outside the indexed set to avoid a self-referential checksum. It only invokes read-only `git rev-parse`/`git cat-file --batch`; it does not commit or alter the index. The original repair audit against `8e6c015b` matched all283 stored result blobs at13300418 bytes; no new inference occurred. A fresh reproduction has new timing records and must be compared with its own published candidate, not expected to match the historical data byte-for-byte.

Perform a privacy/evidence review before publishing. Do not commit external archives/DLLs/GGUF/KV files, encrypted secret/config stores, auth/lease records or whole device/process inventories. Keep failed/degraded/censored arms and absent counters explicit. The report recommends no product change; Git/GitHub publication and any later integration/acceptance belong to their authorized owner.
