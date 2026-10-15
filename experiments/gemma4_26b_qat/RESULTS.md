# Issue #245 — Gemma 4 26B A4B QAT Q4 on Windows/Vulkan

## Result and recommendation

**Positive, bounded manual-translation feasibility on the tested machine.** The exact requested artifact served repeated synthetic translations through the unmodified PuriPuly application using pinned **llama.cpp b11537** and actual Vulkan execution. GPU expert caching works; it is not automatically faster than CPU expert computation. Useful serving configurations below the preferred memory target also work. Above-target configurations were measured and remain eligible.

**Experiment-selected recommendation:** `new-ncpu12-cache2048-swa-nocram`:

```text
b11537 / 8e2d31e0eb658d0382d3106a61369dff81f0c080
--n-gpu-layers 99 --n-cpu-moe 12 --moe-cache-mib 2048
--swa-full --cache-ram 0
```

Keep the common profile below, including total context **4608**, **three slots**, mmap, Vulkan, 4/4 compute threads, f16 KV, and reasoning off. This recommendation **does not meet the preferred 8-GB target**: actual-app sampled server dedicated GPU peak was **12,780,429,312 bytes = 12.780 GB = 11.903 GiB**. It is a measured latency/resource knee, not a memory-cutoff decision. It completed 30/30 actual-app translations with median **0.795 s**, p95 **1.412 s**, maximum **3.841 s**. Maximum-layer offload used another 2.658 GB for a median improvement of approximately 0.094 s; sub-8-GB alternatives took approximately 1.39–1.81 s in the same app sequence.

This is an **experimental serving recommendation, not an authorized deployment or an approved latency SLA**. The product owner must accept or reject the additional GPU/private-commit cost versus the documented low-memory alternatives. **Retain shipped E4B until separately authorized managed-runtime integration and rollout.** The demonstrated endpoint is user-managed; managed 26B provisioning, prefix ownership, fallback, and process lifecycle have not been integrated or approved.

## Coverage and evidence index

Execution date: 2026-10-15. Product baseline: `7a8151be6114a950f062e2430bd36f899d57e4de`; initially clean experimental branch. Only task-local experiment files were authored. No production source, live settings, user transcripts, credentials, Git history, branch/index, or remote publication was changed by the execution worker.

- **27 completed standalone profiles × 43 requests = 1,161 completions**, plus one retained model-load failure. This includes the same-model old-runtime control and E4B reference.
- **Eight actual-app profiles × 30 = 240 matched completed translation events**, across **16 fresh app instances**. One earlier app attempt stopped before submission and remains in the failed-attempt ledger.
- Only **12 distinct standalone source sentences** and **five actual-app source sentences**; repetitions are serving evidence, not independent quality samples.
- All owned model processes were released. All app hosts in successful probes were independently observed exited; model shutdowns were graceful after the documented initial console-handler correction.

| Evidence | Contents |
| --- | --- |
| `results/identity.json` | Hardware/OS, model hashes/metadata/tensor histograms, archive and executable/DLL identities |
| `results/build-provenance.json` | Binary-reported compiler, pinned release-workflow settings, storage/PCIe facts and provenance limits |
| `results/summary.json` | All standalone outcomes, exact arguments, phase distributions, memory units, CPU costs, cache/placement logs |
| `results/<profile>.json` | Individual request inputs/outputs, real token/timing counts, hashes, quality flags, failures and shutdown outcome |
| `results/<profile>.samples.jsonl` | One-second buckets preserving underlying sampled maxima, available-RAM minima and sample counts |
| `results/app-summary.json` | App completion distributions, fresh-instance settings, native timing records, resources and cleanup |
| `results/app-*-attempt*.json` | Actual supported CLI commands/receipts, admitted-turn-matched content events and controlled degradation |
| `results/app-*.samples.jsonl` | Separate server/app sampled resource ledger |
| `results/mechanisms.json` | Source-backed CPU/prefill/cache interpretation and actual cache execution counters |
| `results/quality-unique.json`, `QUALITY.md` | Every unique standalone output with occurrences; independent initial review and bounded fidelity caveats |
| `results/resume-headroom.json`, `results/release.json` | Background headroom before execution and final release observation |

Raw 250-ms samples, native logs, complete acquisition manifests, isolated app state and downloaded/extracted artifacts remain outside Git at **`<ARTIFACT_ROOT>`**, the original user environment's `%LOCALAPPDATA%/Temp/puripuly-issue245`. No model binaries or raw private process/device catalogs belong in the report area. Public evidence uses path placeholders.

## Frozen identities

### Host and runtime

| Item | Observed identity |
| --- | --- |
| OS | Windows 11 Education, 10.0.22631, build 22631, x64 |
| CPU | AMD Ryzen 7 9800X3D; 8 physical / 16 logical cores |
| RAM | 2 × 16 GiB DDR5, configured 5600 MT/s; OS-visible 33,138,100 KiB |
| Power | High performance, GUID `8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c` |
| GPU | AMD Radeon RX 7900 XTX; runtime reports 24,560 MiB usable (25,753,026,560 bytes), a 24-GiB-class card |
| Vulkan / driver | Instance 1.4.341; device API 1.4.349; AMD proprietary 26.8.1 (LLPC), driver version 2.0.395; Windows driver 32.0.31041.1004 |
| PCIe | PnP current/max link-speed enum 4 and width x16: PCIe 4 / 16 GT/s, x16 |
| Artifact storage | C: on SOLIDIGM SSDPFKKW020X7, NVMe, 2,048,408,248,320 bytes; 512-byte logical / 4096-byte physical sectors |
| New runtime | b11537, `8e2d31e0eb658d0382d3106a61369dff81f0c080` |
| Shipped control | b10423, `a94d563ed801d1da1b8c2432946de07d0231bb3d` |
| Compiler | Both server version outputs: Clang 20.1.8, Windows x86_64 |
| App interpreter | Python 3.14.7, MSC v.1944, AMD64; launcher SHA256 recorded; imports explicitly resolved to the execution checkout |
| Collector | Windows GetProcessMemoryInfo/GetProcessTimes/GlobalMemoryStatusEx and English PDH counters via ctypes; psutil 7.2.2 for mapped-file observations and owned-process exit observation |

Official pinned Windows x64 Vulkan archives were acquired, not a floating latest build. Their SHA256 values are:

- b11537, 33,458,882 bytes: `ae34d6a850c5d362cdfc0ef8f88f31fc499dab6df8853cec7887a32891aa5bbe`.
- b10423, 34,563,676 bytes: `510447fb021c80a264b2181c885b5f2ce9cc5b66c65d447cd1f9ce7ba81dc222` — matches the shipped distribution contract.

The server executables are small DLL launchers; **all archive EXE/DLL hashes**, including the actual server implementation and Vulkan backend, are recorded, not only the launcher. Release workflows specify Release, dynamic backends, CPU variants, OpenMP and Vulkan SDK 1.4.357.0. Exact release-run tool versions/Vulkan-DLL compiler patch details not exposed by the binaries are unavailable; workflow settings are not mislabeled as a recovered CMakeCache. No local native compilation or upstream runtime modification is claimed.

### Fixed models

| Role | Artifact | Size / SHA256 |
| --- | --- | --- |
| Target | `unsloth/gemma-4-26B-A4B-it-qat-GGUF`, revision `7b92b5b28818151e8669af2e45e88d6086f490dd`, `gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf` | 14,249,047,104 bytes; `a7c5bc715f5ff8e99a3e8901ce7d2b42b402c669bf24f7c5250747633d0f5891` — verified |
| Product reference | Existing read-only E4B QAT Q4 installation matching #75's verified size/hash, `gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf` | 4,215,695,776 bytes; `df0fd4ee07072c607c29a0a1cb4f98918426cca12f45a2776bdd6ee6d09a4de3` — verified |

The fixed target's **actual** GGUF histogram is **392 f32 + 266 q4_0 tensors**, file-type metadata Q4_0, architecture Gemma4, 30 blocks, 128 experts, eight selected experts. Runtime metadata reports 25.23B parameters. The published filename does not override this inspected tensor evidence, and no substitute quantization was used. E4B has 321 f32 + 345 q4_0 tensors. File sizes and active-parameter counts are not resident-memory results. Text only: no projector, other target model, or MTP drafter was loaded.

## Workload, controls and measurement

Common effective server arguments, with model/executable/ephemeral port supplied per run:

```text
--alias issue245 --load-mode mmap --ctx-size 4608 --parallel 3
--batch-size 512 --ubatch-size 512 --threads 4 --threads-batch 4
--cache-type-k f16 --cache-type-v f16 --cache-prompt
--reasoning off --reasoning-budget 0 --warmup --perf --metrics
--no-webui --threads-http 1 --host 127.0.0.1 --port <owned-port>
--prio -1 --device Vulkan0 --flash-attn on --fit off --verbosity 4
```

Placement/cache flags are in the tables and exact per-run command arrays. Omitted MoE cache means the supported default **0/off**, not an unknown cache size. The first failing 4096-MiB preflight used verbosity 3. Every tested runtime reports **1536 tokens per slot**, total 4608, three slots, non-unified KV. No context/slot reduction is presented as equivalent.

The frozen checkout's real prompt is rendered with target-language rules and pair examples through `get_default_prompt` / `render_translation_prompt_template`, not the template alone. System-token counts measured on the new target are:

| Pair | Raw system tokens | Chat-wrapped system + blank-user prompt |
| --- | ---: | ---: |
| en→ko | 812 | 828 |
| ko→en | 810 | 826 |
| ja→en | 806 | 822 |
| zh-CN→en | 821 | 837 |

Rendered content hashes are in `prefix_metadata`; full real request token counts are recorded per request. The first three complete prompts have 912/908/903 tokens including their wrapper, input and history. The standalone fixture has **five fixed synthetic English source-context lines**, identical across configurations, not a claim to reproduce app rolling-history state. Inputs vary across names, quantities, negation, instructions and short social utterances in four pairs.

- Three first-prefix requests, one on each fresh slot, are diagnostic and use temperature 0.
- Five retained same-pair requests, five four-pair alternations (20 requests), and five rounds with all three slots occupied (15 requests) use **temperature 0.6, seed 42, output limit 96**, normal EOS. No fixed-length throughput fiction is applied to naturally short translations.
- Effective target sampling logs show top-k 64, top-p .95, min-p .05 and repetition penalty 1. Actual token counts and timing distributions remain in JSON.
- Standalone requests call `/apply-template`, then native streaming `/completion` with explicit `id_slot`. **TTFT includes template HTTP overhead and time to first nonempty SSE content**, not pure kernel latency.
- Cold readiness is a fresh-process health observation, **n=1 per profile**, not filesystem-/shader-cache-cold latency or a repeated-cold median. No OS/file/shader cache flushing or unrelated process termination was performed.
- Median and nearest-rank p95 are reported; per-run min/max/spread and sample counts are retained. Same-pair n=5 is a small sample; p95 is its maximum. Three-slot latency includes queueing/continuous batching and is not single-stream decode throughput.

### Memory and CPU accounting

The collector starts immediately after owned server creation and remains through load, prefix, repeated serving and shutdown. Nominal interval is 250 ms; actual cadence is retained. For the recommended standalone profile: **113 samples, median interval .257 s, p95 .259 s, maximum gap .335 s**. These are **observed sampled GPU peaks/lower bounds on a true transient peak**, not a lifetime GPU high-water guarantee.

PDH rows are restricted to the owned PID and exact observed RX adapter **`luid_0x00000000_0x0000F60B_phys_0`**, separately from the other virtual-adapter LUID. The owned Vulkan process registers on that adapter. Whole-adapter memory is measured independently, not reconstructed by summing processes. GPU utilization is the **busiest owned process engine**, not the invalid sum of engine percentages. Missing process GPU rows, including the headless app's absent GPU row, are unavailable rather than fabricated zeroes. The server is the only owned model/GPU runtime; whole-device sampling encloses all allocations and background use.

Host evidence includes working set, API peak working set, private committed bytes, peak commit (`PeakPagefileUsage`, **not actual pagefile residence**), cumulative process faults, system available/commit, and system page-input/read rates. Fault counts include soft faults; system hard-page activity is not attributable exclusively to this process. Buffer logs and mapped-file observations are retained; psutil's reported GGUF resident mapping is **not a complete inventory of driver/import/repacked memory**. No unexposed CPU_REPACK allocation is invented as zero. Do not add working set to private commit or GPU counters as disjoint memory.

CPU table values are sampled serving-window process CPU time / wall time ×100: **100% = one logical core-equivalent; divide by 16 for whole-machine percentage**. For the recommended profile this is 1.703 sampled CPU seconds over 22.468 serving seconds, 7.58% of one core, approximately .47% normalized to all 16 logical processors. This is CPU-time cost, not watts. App summaries separately retain server CPU time and both app instances' sampled CPU totals; short-lived CLI client CPU cost is not included in those process counters.

Initial default-cache profiles reached very low available RAM and substantial paging activity. Their tail differences are **not clean cache-size-only performance proof**. Refined finalists retained approximately 7–8 GB minimum available RAM in their standalone/app observations, with similar modest shared-GPU allocations rather than multi-GB WDDM spill replacing intended placement. Physical PCIe traffic/bandwidth, per-request cache hits, process-attributed hard faults, board power and an exact GPU lifetime peak are unavailable.

## Complete standalone matrix

All successful rows have **43/43 completed requests**. Times are seconds. `alt` is four-pair alternation, n=20; `3-slot` n=15. GPU is sampled **server dedicated GB**; host is sampled WS / private commit GB. `S` = `--swa-full`; `R0` = `--cache-ram 0`. Everything here is b11537 unless identified old/E4B.

| Profile / mechanism | GPU GB | Host WS / private GB | Hot med n=5 | Alt med / p95 | 3-slot med | CPU % one-core |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| E4B, b10423 reference | 2.856 | 4.76 / 5.01 | .300 | .349 / .616 | .566 | 10.5 |
| 26B old b10423, ngl99 | 15.432 | 23.28 / 24.72 | 1.954 | 2.033 / 2.240 | 5.867 | 12.2 |
| ngl99 | 15.437 | 23.76 / 27.22 | 1.023 | 1.082 / 1.180 | 2.944 | 10.2 |
| partial ngl14 | 7.379 | 24.24 / 20.00 | 1.519 | 1.734 / 1.971 | 4.944 | 116.9 |
| partial ngl20 | 10.338 | 23.59 / 21.79 | 1.174 | 1.284 / 1.489 | 3.761 | 118.3 |
| cpu-moe, cache off | 2.903 | 23.24 / 14.96 | 1.932 | 2.141 / 2.388 | 7.021 | 99.1 |
| cpu-moe, cache1024 | 3.976 | 22.86 / 15.17 | 2.327 | 3.083 / 3.645 | 9.501 | 25.0 |
| cpu-moe, cache2048¹ | 5.052 | 19.31 / 16.76 | 1.999 | 2.773 / 5.113 | 8.819 | 23.1 |
| cpu-moe, cache3072 | 6.128 | 23.57 / 17.41 | 1.680 | 2.111 / 2.744 | 6.107 | 19.7 |
| cpu-moe, cache4096 | failed | 2.06 / 2.10 before failure | — | — | — | — |
| n-cpu-moe24, off | 5.479 | 24.17 / 17.36 | 1.727 | 1.879 / 2.080 | 5.211 | 91.2 |
| n-cpu-moe24, cache2048 | 7.627 | 24.30 / 19.30 | 1.561 | 1.867 / 2.329 | 5.855 | 16.4 |
| n-cpu-moe12, off | 10.630 | 24.04 / 21.28 | 1.091 | 1.217 / 1.410 | 3.312 | 96.2 |
| n-cpu-moe12, cache2048 | 12.779 | 24.07 / 23.37 | .860 | 1.021 / 1.166 | 2.883 | 16.0 |
| ngl99 + S | 15.438 | 17.44 / 18.68 | .285 | .324 / .668 | .468 | 9.0 |
| ngl99 + R0 | 15.437 | 17.68 / 18.91 | 1.000 | 1.032 / 1.100 | 2.838 | 9.4 |
| ngl99 + S + R0 | 15.438 | 15.11 / 16.39 | .262 | .313 / .674 | .482 | 7.3 |
| partial14 + S | 7.380 | 17.92 / 11.55 | .822 | 1.009 / 1.435 | 2.128 | 166.5 |
| partial14 + S + R0 | 7.380 | 15.26 / 9.01 | .806 | .999 / 1.402 | 2.095 | 166.5 |
| cpu-moe, off + S | 2.905 | 17.33 / 6.72 | .900 | 1.131 / 1.929 | 1.983 | 152.4 |
| cpu-moe, off + S + R0 | 2.905 | 14.46 / 4.18 | .900 | 1.138 / 1.909 | 2.103 | 142.7 |
| cpu-moe, cache2048 + S | 5.054 | 16.83 / 8.38 | 1.186 | 1.606 / 2.216 | 3.009 | 15.4 |
| cpu-moe, cache3072 + S | 6.129 | 16.83 / 9.45 | .977 | 1.325 / 1.747 | 2.330 | 7.4 |
| cpu-moe, cache3072 + S + R0 | 6.129 | 14.33 / 7.15 | .966 | 1.311 / 1.773 | 2.452 | 8.9 |
| n24/cache2048 + S | 7.629 | 17.14 / 11.16 | .915 | 1.223 / 1.599 | 1.921 | 8.1 |
| n24/cache2048 + S + R0 | 7.629 | 14.44 / 8.60 | .932 | 1.221 / 1.590 | 2.751 | 8.6 |
| n12/cache2048 + S | 12.780 | 17.47 / 16.32 | .388 | .569 / .794 | .794 | 6.4 |
| **n12/cache2048 + S + R0** | **12.780** | **14.78 / 13.71** | **.416** | **.561 / .783** | **.791** | **7.6** |

¹ Serving completed, but the first console shutdown was ignored; the owned process was forcibly stopped after the recorded timeout. Its destructor cache statistics are unavailable. This was a harness console-inheritance issue, not a silent successful graceful shutdown. The launcher was corrected to clear inherited Ctrl+C-ignore state; subsequent completed matrix/app servers exited gracefully with code 0. The original record was not overwritten.

### What the refinements explain

1. **Runtime upgrade alone is not a universal decode win.** Same-model old/new maximum-offload alternation medians were 2.033/1.082 s, while their reported short decode medians were approximately 98/92.5 tok/s. Prefill/prefix behavior matters more than advertising a decode number.
2. **Prefix handling was the dominant avoidable cost.** Default SWA checkpoint restoration often reused only approximately 390–420 of a roughly 900-token request and re-evaluated about 500 tokens. With S, representative hot requests retained 884–911 tokens and evaluated 1–28 incremental tokens. ngl99 alternation fell from 1.082 to .324 s with the same measured dedicated allocation. Context, slots, input semantics and model were not reduced.
3. **Host RAM cache and SWA are different mechanisms.** R0 alone reduced maximum-offload private commit from 27.22 to 18.91 GB, but did not eliminate the approximately one-second prefix cost. S alone reduced that cost; combining them reduced private commit to 16.39 GB. R0 is not guaranteed latency-neutral: the n24 three-slot median increased from 1.921 to 2.751 s in the recorded comparison. Retained slot KV is distinct from the server's separate RAM prompt cache.
4. **Dynamic cache trades CPU for transfer cost.** CPU expert decode can be faster than a small dynamic cache, but consumes far more CPU. All-host cache3072/S/R0 used about 8.9% of one core versus CPU experts' 142.7%, yet app medians were 1.748 versus 1.811 s. Cache misses/upload synchronization remain substantial.
5. **Placement matters as much as cache budget.** Keeping only the first 12 layers' experts host-backed makes the 2048-MiB cache cover a much larger useful fraction. It beats the 3072-MiB all-host configuration despite a smaller cache budget, at greater static GPU residency. Above-8-GB refinement was therefore necessary and was not excluded.

## Vulkan execution and cache limits

The selected profile reports 31/31 offloaded layers, **CPU_Mapped model buffer 5663.88 MiB**, **Vulkan model buffer 8673.35 MiB**, and **2045.07 MiB cache for 4900.50 MiB of host experts**. Standard maximum-layer offload still retains a 396-MiB CPU-mapped model buffer; “ngl99/full” does not mean every tensor resides exclusively in VRAM. Ordinary partial14 reports 14/31 layers and 7857.00/6112.86 MiB CPU-mapped/Vulkan model buffers, plus CPU/GPU KV and compute buffers.

Pinned source distinguishes ordinary host placement from GPU expert caching:

- Without cache, small expert decode operations remain CPU-scheduled; sufficiently large prefill operations can transfer/offload to Vulkan. Vulkan's host-operation offload threshold defaults to batch size **32**. This is a source-backed interpretation corroborated by placement and CPU cost, not a per-kernel profiler trace.
- Cache banks are device weights. Selected misses are uploaded; hits reuse bank slots. `get_slot_map` bypasses **n_tokens >32**, zero-token and insufficient-slot cases. Tensor-parallel mode is explicitly rejected.
- The >32 bypass does **not** disable every cache benefit during large prefill: `copy_experts` can read existing banks into large-batch staging without evicting the decode LRU.

Actual graceful-destructor counters prove execution, not merely allocation:

| Profile | ≤8-token hits / misses / hit rate | ≤8 upload MiB | >8 hits / misses / upload MiB | Large-batch cached copies |
| --- | --- | ---: | --- | --- |
| cache1024, default SWA | 51,208 / 128,634 / 28.47% | 410,397.73 | 58 / 5,739 / 18,309.88 | 26,334 experts / 42,008.39 MiB |
| all-host cache3072/S/R0 | 109,468 / 37,782 / 74.34% | 120,540.81 | 5,466 / 17,667 / 56,365.32 | 34,506 / 55,044.48 MiB |
| **n12/cache2048/S/R0** | **52,584 / 6,695 / 88.71%** | **21,359.93** | **5,028 / 3,524 / 11,243.07** | **22,682 / 36,182.66 MiB** |

These are context-lifetime logical transfer counters, including warmup/prefill/serving, not measured physical PCIe bandwidth or pure-decode-only rates. The `>8` slot-map counters cover eligible small batches; separate large-batch-copy counters preserve the prefill distinction.

**Retained failure:** cache4096 could not initialize a **2,861,438,976-byte bank**, with `Requested buffer size exceeds device buffer size limit: ErrorOutOfDeviceMemory`. Pinned Vulkan buffer code rejects `size > device.max_buffer_size` before allocation. Cache3072 works. This is a per-bank/backend/device limit, **not total 24-GiB exhaustion and not an 8-GB eligibility rule**. Larger 6/8-GiB budgets were not claimed tested; monotonic bank growth after this limit is not a useful sweep without upstream changes, which are out of scope. No unsupported size override, CUDA substitution, model pruning or quantization substitution was attempted. Direct Strata implementation provenance is not established.

## Actual unmodified PuriPuly probe

Each profile used its own source CLI instance and owned native server. Before first startup the probe seeded **new** canonical settings, probe-only encrypted-file secrets, telemetry/clipboard off, keyless Soniox selections, capture off and OSC connection mode off. LOCALAPPDATA, APPDATA, USERPROFILE, HOME and temp roots were redirected; inherited provider/native overrides were removed. No live settings/secrets/logs/audio were used.

**OSC off does not silence Self publication.** An exclusively bound worker-owned loopback UDP sink replaced the destination, with both configured send-port fields pointing to it. It stayed bound through teardown. Receiver/OSCQuery were effectively off, overlay desired/effective off, microphone/capture inactive, and source process observation was read-only against an isolated scene-log root. Each successful run received 90 synthetic datagrams (typing/output); this is local sink evidence, not VRChat delivery or GUI presentation.

The supported CLI supplied current `capabilities` and necessary `settings choices`, enabled translation, applied language changes with expected revisions, submitted UTF-8 synthetic files, and followed content-enabled translation events. Every completion was matched to an admitted parent/child turn ID. Persisted language and effective translation-enabled/ready state were rechecked. The second app instance started with the intentionally persisted **en→ko** pair rather than the original **ko→en**, a new instance/PID, and translation disabled until explicitly enabled again. External server/KV survived app-only restart; this is **not** managed model-restart/prefix-disk-cache readiness.

### Controlled inactive-STT degradation

The first probe aborted before text submission because language application returned terminal **degraded**, `settings_commit_success_runtime_degraded`, specifically `provider.stt / stt.provider_not_ready`: intentionally keyless Soniox cannot build. That failed attempt remains recorded. The Director then explicitly authorized **manual translation with inactive STT**, not a fake provider or a product repair.

Successful probes retain **31 degraded language-settings receipts per profile (248 total)**, verify their exact STT attribution, and separately prove persisted languages, effective local translation and completed content. No broad healthy-settings/ASR/live-audio claim is made. Expected redundant final `app stop` calls in earlier successful harness versions can report `instance_not_found` after the two already observed host exits; those receipts remain and are not translation failures. Later cleanup avoids that redundant call.

The actual app uses non-streaming OpenAI-compatible requests to the owned `local_llm`/`ollama` endpoint, real rolling **last-five source entries / 60 s / pair-matched history**, real prompt rendering, temperature .6, and reasoning disabled. App requests do not set a normal output limit or seed: the probe explicitly adds **server limit 96 and seed42**, labeled experimental. App TTFT is unavailable. Completion wall time includes CLI startup/admission/typing-completion/publication observation, unlike standalone streaming TTFT. Native timing records and actual evaluated/output counts are preserved and match each run's 30 completions.

| Actual-app profile | n | Server GPU GB / GiB | Server WS / private GB | Completed wall median / p95 / max s | Whole-device dedicated GB |
| --- | ---: | ---: | ---: | ---: | ---: |
| E4B, b10423 reference | 30 | 2.855 / 2.659 | 3.72 / 3.97 | .702 / .774 / 3.328 | 5.925 |
| cpu-moe off/S/R0 | 30 | 2.905 / 2.706 | 14.32 / 4.26 | 1.811 / 2.007 / 6.415 | 5.983 |
| cpu-moe cache3072/S | 30 | 6.130 / 5.709 | 14.25 / 7.20 | 1.769 / 2.279 / 6.459 | 9.207 |
| cpu-moe cache3072/S/R0 | 30 | 6.130 / 5.709 | 14.25 / 7.20 | 1.748 / 2.194 / 6.435 | 9.208 |
| partial14/S/R0 | 30 | 7.380 / 6.874 | 15.31 / 9.11 | 1.388 / 1.782 / 4.945 | 10.458 |
| n24/cache2048/S/R0 | 30 | 7.629 / 7.105 | 14.40 / 8.65 | 1.574 / 2.044 / 5.660 | 10.706 |
| **n12/cache2048/S/R0** | **30** | **12.780 / 11.903** | **14.78 / 13.75** | **.795 / 1.412 / 3.841** | **15.857** |
| ngl99/S/R0 | 30 | 15.438 / 14.378 | 15.17 / 16.44 | .700 / 1.203 / 2.780 | 18.514 |

The headless app additionally used approximately **.117–.119 GB sampled working set** and **.60 GB private commit**; lifetime API peaks and separate process ledgers are retained. Do not mistake these for GPU allocation or add non-coincident peaks as one measured instantaneous total. App GPU process counters had no registered row and were not replaced by zero. The recommended app run's server shared GPU peak was **609,263,616 bytes**, whole-device dedicated **15,857,037,312 bytes**, leaving approximately **9.896 GB** of the runtime-reported device capacity; minimum available host RAM was **7,886,368,768 bytes**. No physical 8-GB card or VRChat co-rendering workload was tested. Even a sub-8-GB server may exceed an 8-GB whole-device budget with the observed desktop background allocations.

## Quality and final selection reasoning

See `QUALITY.md`. Independent review covered the first 344 outputs. The execution owner then inspected every unique output across the complete **1,161-output / 12-source standalone corpus**, plus all **240 app outputs / five sources**. No configuration-specific material meaning reversal, quantity/negation corruption, pathological repetition, visible reasoning or truncation was observed. These are bounded spot-checks, not global quality certification or evidence of 26B superiority over E4B.

Caveats are retained, not masked by automatic booleans: all standalone profiles assume Japanese **Kobayashi** for ambiguous Chinese **小林**; actual QAT app probes use **Xiao Lin**, but the source alone still does not establish intended reading. **44/156 QAT occurrences** of “in five minutes” become “within five minutes”; this also appears in the old control, not a placement-specific collapse. E4B sometimes adds unsupported gender and omits an explicit ticket noun. Correct “twice/three” outputs intentionally retain literal-digit-check false flags.

The measured frontier has distinct, non-dominated resource choices:

- **Minimum dedicated/private commit:** cpu-moe off/S/R0, approximately 2.905-GB GPU and 4.18-GB standalone private commit, but approximately 1.43 core-equivalents serving CPU and 1.811-s app median.
- **Sub-8-GB, low residual CPU:** all-host cache3072/S/R0, approximately 6.13-GB GPU, approximately .09 core-equivalents and 1.748-s app median; cache misses/transfers limit latency. n24/cache2048 uses 7.63 GB and improves app median to 1.574 s.
- **Ordinary partial placement:** 7.38-GB partial14/S/R0 reaches 1.388-s app median but costs approximately 1.67 core-equivalents. It is not an equal-CPU-cost alternative.
- **Recommended latency/resource knee:** n12/cache2048/S/R0 reaches .795-s app median at 12.78 GB, low residual CPU, and lower GPU/private commit than maximum offload. Its 88.71% small-batch cache hit rate explains why a smaller cache with more static GPU placement can beat a larger all-host cache.
- **Lowest measured app latency:** ngl99/S/R0 at .700 s and 15.44 GB. It remains eligible, but its additional 2.66 GB buys only approximately .094 s median against the knee; p95 and cold-first differences are separately visible.

The recommendation is based on the measured short-subtitle/app knee, not a claim that .795 s satisfies an approved SLA. **Owner decision before adoption:** approve the >8-GB GPU/private-commit premium for the measured latency benefit, or prioritize GPU headroom and select a documented sub-8-GB option. Current E4B remains a separate product-latency/footprint reference with much lower host memory; this small quality set does not prove that changing models is worth that footprint.

## Reproduction and handoff

**Preserve the frozen records:** reproduce only in another copy of the same baseline checkout plus these experiment files, with an empty task-local results directory and a fresh owned artifact root. Seed that root by reusing/copying/linking only the verified models, downloads and extracted runtimes; do not reuse the recorded `matrix/<profile>` or app-state directories. Profile basenames are not an attempt-aware standalone history store. Never rerun these commands over the initial eight records or the final ledger. Identity capture includes local capability catalogs; keep its raw output external and project only relevant hardware/driver facts before publication.

Use the supplied interpreter environment with repository dependencies (psutil included), the frozen checkout and a new owned artifact/result directory. Do not point the app probe at live settings. Preserve existing verified downloads rather than reacquiring them.

```powershell
$python = '<APP_PYTHON>/python.exe'
$root = '<NEW_OWNED_ARTIFACT_ROOT>'
$e4b = '<EXISTING_VERIFIED_E4B_GGUF>'
& $python experiments/gemma4_26b_qat/acquire.py --root $root
& $python experiments/gemma4_26b_qat/capture_identity.py --root $root --e4b $e4b --output "$root/identity-private.json"
& $python experiments/gemma4_26b_qat/run_matrix.py --root $root --e4b $e4b --profiles new-ncpu12-cache2048-swa-nocram --repetitions 5
& $python experiments/gemma4_26b_qat/app_probe.py --root $root --e4b $e4b --profile new-ncpu12-cache2048-swa-nocram --attempt 1
& $python experiments/gemma4_26b_qat/summarize.py --results experiments/gemma4_26b_qat/results --raw-root $root
& $python experiments/gemma4_26b_qat/summarize_apps.py --results experiments/gemma4_26b_qat/results --raw-root $root
```

`run_matrix.py` contains every tested profile; per-run command arrays are the exact resolved commands, including observed ports. Run inference sequentially with exclusive benchmark ownership. The sampler's adapter LUID is the observed boot's RX LUID; re-establish it before another boot/device and do not accept missing counters as zero. `app_probe.py` reconstructs the server from its standalone record; the owned app state directory must be new. Root placeholders mean path substitution, not model substitution. Raw model/native logs stay external. Normalized UTF-8 text-log hashes in run records are not advertised as raw-file checksums.

No production build, pytest suite, linter or formatter was run; this assignment exercised model/app experiments, not product edits. The Director should freeze/review the task-local candidate and its evidence, then publish the authorized English result/status update. Additional GPU benchmarking is not required to read this evidence. Production rollout requires a separately authorized implementation of the 26B managed artifact/profile/lifecycle contract; the generic endpoint result does not establish that readiness.

### Issue #245 checklist disposition

| Requirement | Disposition |
| --- | --- |
| Exact target verified; old/new identities, metadata, hardware/toolchain frozen | Done, with explicit unavailable release-run details and metric limits |
| Required pinned runtime upgrade exercised, actual Vulkan and cache execution | Done on b11537, including native hit/miss/upload and large-batch-copy counters |
| Ordinary partial, CPU expert offload, matched multiple caches, below/near/above preferred target | Done; 1024/2048/3072 budgets, both n-cpu placements, 4-GiB failed bank retained |
| Real prompt, multilingual inputs, repeated cold/retained/pair/three-slot workload at4608/3 | Done; diagnostic cold n and fixed-vs-app history distinctions explicit |
| Refine gains one mechanism at a time; retain failures; full sampled resource ledger | Done: runtime, SWA, host RAM cache, cache size and placement; memory-pressure rows qualified |
| Actual app completions, language changes, fresh-process restart and release | Done for manual/user-managed endpoint; inactive STT degradation retained; not ASR or managed 26B integration |
| Quality, configuration table and memory/latency/CPU frontier, E4B comparison | Done, bounded to synthetic spot-checks with named caveats |
| Final profile/rationale and preferred-target answer | Recommended n12/cache2048/S/R0 is above target; useful sub-target alternatives demonstrated; product-owner adoption tradeoff remains explicit |
| Integration/rollout recommendation | Retain shipped E4B until separate authorization; no deployment/merge or production modifications performed |

### Primary sources

- [Issue #245 contract](https://github.com/kapitalismho/PuriPuly-heart/issues/245), [#75 including amendments](https://github.com/kapitalismho/PuriPuly-heart/issues/75), [#64 decisions](https://github.com/kapitalismho/PuriPuly-heart/issues/64).
- [Pinned target artifact](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-qat-GGUF/blob/7b92b5b28818151e8669af2e45e88d6086f490dd/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf).
- [b11537 release](https://github.com/ggml-org/llama.cpp/releases/tag/b11537), [cache implementation](https://github.com/ggml-org/llama.cpp/blob/8e2d31e0eb658d0382d3106a61369dff81f0c080/src/llama-moe-cache.cpp), [Vulkan buffer limit](https://github.com/ggml-org/llama.cpp/blob/8e2d31e0eb658d0382d3106a61369dff81f0c080/ggml/src/ggml-vulkan/ggml-vulkan-buffers.cpp#L28-L33), [Vulkan offload policy](https://github.com/ggml-org/llama.cpp/blob/8e2d31e0eb658d0382d3106a61369dff81f0c080/ggml/src/ggml-vulkan/ggml-vulkan.cpp#L16019-L16023).
- [Microsoft WDDM GPU accounting](https://devblogs.microsoft.com/directx/gpus-in-the-task-manager/): API-independent accounting, dedicated/shared meanings and busiest-engine utilization semantics.
- Frozen local `docs/architecture.md`, `docs/cli.md`, `src/puripuly_heart/core/local_translation/runtime_profile.py`, prompt assembly, application-control and provider sources at the product baseline.
