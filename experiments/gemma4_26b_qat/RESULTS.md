# Issue #245 — Gemma 4 26B A4B QAT Q4 on Windows/Vulkan

## Result and recommendation

**Positive, bounded manual-translation feasibility on the tested machine.** The exact requested artifact served repeated synthetic translations through the unmodified PuriPuly application using pinned **llama.cpp b11537** and actual Vulkan execution. GPU expert caching works; it is not automatically faster than CPU expert computation. Useful serving configurations below the preferred memory target also work. Above-target configurations were measured and remain eligible.

**Revised experimental recommendation after F1 refinement:** `new-ncpu12-cache3072-swa-nocram`:

```text
b11537 / 8e2d31e0eb658d0382d3106a61369dff81f0c080
--n-gpu-layers 99 --n-cpu-moe 12 --moe-cache-mib 3072
--swa-full --cache-ram 0
```

Keep the frozen common profile, including total context **4608**, **three1536-token slots**, mmap, Vulkan, 4/4 threads, f16 KV and reasoning off. This recommendation **exceeds the preferred8-GB target**. Maximum observed dedicated allocation across its standalone/app runs was **13,856,235,520 bytes =13.856GB =12.905GiB**; actual-app peak was13,855,969,280 bytes. It completed30/30 actual-app texts with **text-submit CLI start → matched translation-completion** median **.681s**, p95 **1.395s**, max **4.193s**. The contemporaneous2048 control was .807/1.409/3.964s at12.780GB. Thus3072 buys approximately .126s median and substantially better occupied-slot latency for approximately1.076GB extra dedicated/private commitment, but **does not improve every tail/cold observation**. These times exclude audio collection, endpointing, STT and GUI rendering; they are not speech end-to-end latency.

This is an **experimental serving recommendation, not an authorized deployment or an approved latency SLA**. The product owner must accept or reject the additional GPU/private-commit cost versus the documented low-memory alternatives. **Retain shipped E4B until separately authorized managed-runtime integration and rollout.** The demonstrated endpoint is user-managed; managed 26B provisioning, prefix ownership, fallback, and process lifecycle have not been integrated or approved.

Checkpoint `84f04d09b6235f07da5fb0248ec7571e75befcc7` had accepted finding **F1**, a selection-evidence gap rather than a product defect: n12/S/R0 lacked matched OFF and nearby budgets. That gap is now addressed with separately named OFF/1024/fresh2048/3072 records, actual pressure qualification and30-text app probes for all four. The original2048 recommendation is **superseded by3072**, not retroactively rewritten. All original per-run records, failures and original-context observations remain. Owner approval of the adoption tradeoff is still separate from this experimental recommendation.

## Coverage and evidence index

Execution date: 2026-10-15. Product baseline: `7a8151be6114a950f062e2430bd36f899d57e4de`; initially clean experimental branch. Only task-local experiment files were authored. No production source, live settings, user transcripts, credentials, Git history, branch/index, or remote publication was changed by the execution worker.

- **34 completed standalone run identities ×43 =1,462 completions**, including separately named matched-condition repeats, plus the retained4096-MiB load failure. Original27/1,161 observations remain; F1 added301 observations in seven runs, three pressure-qualified.
- **13 completed actual-app runs ×30 =390 matched completions**, across **26 fresh app hosts**:12 generic-endpoint runs (360 texts) and one true managed-E4B run (30 texts). The first failed pre-submission app attempt remains. The managed probe's initial missing-legacy-log classification is preserved and resolved by a separate correlated-diagnostic recovery, not an inference rerun.
- Only **12 distinct standalone source sentences** and **five actual-app source sentences**; repetitions are serving evidence, not independent quality samples.
- Two bounded memory-attribution runs added six diagnostic translations and four no-touch VA/residency snapshots; these are not counted as matrix ranking runs or new quality source cases.
- Owned model/app processes were released. Successful generic-app servers shut down gracefully; true managed children were observed exited after their app owners stopped (native exit code unavailable). The original forced cache2048 shutdown and first failed app's weaker cleanup evidence remain disclosed.

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
| `results/resume-headroom.json`, `results/release.json` | Original-checkpoint headroom/release; preserved historical observations |
| `results/repair-matrix.json`, `results/repair-headroom*.json` | Accepted-F1 matched region, recovered-condition controls and pressure exclusions |
| `TIMINGS.md`, `results/timing-summary.json`, `results/timings/` | **Every** matrix/app option: split native prefill/decode ms/TPS, counts, n/median/p95/min/max, including failed/N/A and separately labeled diagnostic work |
| `results/managed-e4b-recovery.json` | True managed-E4B correlated native response timings and separate prefix-evaluation/restore windows |
| `results/managed-extraction-check.json` | Final changed capture helper/extraction-only CLI exercised against retained synthetic logs:30 unique terminal/completion joins and native phase values; no model/app rerun |
| `results/memory-correction.json`, `results/memory-attribution-attempt1.json` | Withdrawn psutil interpretation; actual64-bit allocation/working-set evidence |
| `results/repair-release.json`, `results/decision.json` | Current release state and revised measured recommendation/owner tradeoff |

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
| Collector | Windows GetProcessMemoryInfo/GetProcessTimes/GlobalMemoryStatusEx and English PDH counters via ctypes; original psutil7.2.2 mapped-file interpretation withdrawn below; corrected VirtualQueryEx64/QueryVirtualMemoryInformation/QueryWorkingSetEx observations; psutil also observes owned process exits |

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

The collector starts immediately after owned server creation and remains through load, prefix, repeated serving and shutdown. Nominal interval is 250 ms; actual cadence is retained. For the original 2048-MiB standalone profile: **113 samples, median interval .257 s, p95 .259 s, maximum gap .335 s**. These are **observed sampled GPU peaks/lower bounds on a true transient peak**, not a lifetime GPU high-water guarantee.

PDH rows are restricted to the owned PID and exact observed RX adapter **`luid_0x00000000_0x0000F60B_phys_0`**, separately from the other virtual-adapter LUID. The owned Vulkan process registers on that adapter. Whole-adapter memory is measured independently, not reconstructed by summing processes. GPU utilization is the **busiest owned process engine**, not the invalid sum of engine percentages. Missing process GPU rows, including the headless app's absent GPU row, are unavailable rather than fabricated zeroes. The server is the only owned model/GPU runtime; whole-device sampling encloses all allocations and background use.

Host evidence includes working set, API peak working set, private committed bytes, peak commit (`PeakPagefileUsage`, **not actual pagefile residence**), cumulative process faults, system available/commit, and system page-input/read rates. Fault counts include soft faults; system hard-page activity is not attributable exclusively to this process. **Correction:** original `mapped_resident_*` / `rss_bytes` fields are withdrawn as both residency and total-mapping evidence: psutil7.2.2 Windows returns `VirtualQueryEx.RegionSize`, not residency, and truncates it to unsigned32. The retained1,364,148,224 value is page-rounded full GGUF size modulo2^32, not1.36GB resident RAM. Originals remain immutable; see `results/memory-correction.json` and new pointer-width/no-touch allocation/residency snapshots. No unexposed CPU_REPACK allocation is invented as zero. Do not add working set to private commit or GPU counters as disjoint memory.

CPU table values are sampled serving-window process CPU time / wall time ×100: **100% = one logical core-equivalent; divide by16 for whole-machine percentage**. The revised3072 standalone row used3.54% of one core, approximately .22% normalized across16 logical processors. This is CPU time, not watts. App summaries separately retain observed server/app CPU segments, including both fresh managed model processes; short-lived CLI client CPU is not included.

Initial default-cache rows and the first F1 OFF/1024/2048 cohort had actual low free RAM/paging; they are **not clean fine-grained cache-only comparisons**. Recovered F1 standalone rows retained approximately3.9–5.5GB free with similar modest shared-GPU allocations. App OFF/1024 reached .72/1.01GB free and paging bursts: keep their functional results but qualify close wall-time rankings. The current2048/3072 app runs retained4.77/2.02GB minimum free, with no multi-GB shared-GPU growth; remaining global paging bursts may affect tails. No perfectly idle-machine, physical-PCIe, process-hard-fault, power, or true transient GPU-high-water claim is made.

### Where the apparent “5GB experts +8GB RAM overhead” actually goes

That subtraction mixes accounting views. **4900.50MiB =5.139GB host-expert tensor payload is read-only file-backed data**, not an established subset of13.75GB private commit. CPU_Mapped5663.88MiB is an existing mapping's tensor-offset span, not a private allocation. The Windows mmap implementation maps the whole GGUF and `unmap_fragment` is a no-op even for offloaded weights.

The bounded OFF/2048 observations use64-bit `VirtualQueryEx`, `QueryVirtualMemoryInformation.CommitSize` **once per allocation**, filename basenames, and batched `QueryWorkingSetEx` Valid/Shared bits, without reads, touches or locks. All four scans succeeded, with zero unavailable allocation queries/pages; each took2.67–2.84s, so the brackets are **not atomic partitions**.

| n12/S/R0 snapshot | GGUF mapped GB | GGUF valid shareable GB | MEM_PRIVATE allocation commit GB | MEM_PRIVATE valid-private GB | Process WS bracket GB |
| --- | ---: | ---: | ---: | ---: | --- |
| OFF ready |14.249050|9.912771|11.259503|.691593|10.635415→10.635415|
| OFF after three texts |14.249050|13.919584|11.608564|.849433|14.935011→14.794236|
| cache2048 ready |14.249050|9.912762|13.413482|.701706|10.645418→10.645418|
| cache2048 after three texts |14.249050|13.919584|13.588746|.773743|14.770565→14.712574|

The GGUF mapping's per-allocation **private commit is0**. Post-work, approximately **13.92GB valid shareable whole-model file pages** dominate the working set; private valid pages are approximately .774GB plus small image/other contributions. Conversely, **12.773777GB of2048's private commitment is in write-combined private VA ranges**, closely matching simultaneous dedicated GPU12.776014GB; only **.126640GB** in those ranges has Valid private working-set bits. OFF's corresponding write-combined charge is10.626781GB. This directly distinguishes GPU-sized commitment from an equally large resident CPU heap or an asserted permanent DRAM shadow.

At the original app's actual private-peak sample, private13,750,480,896 bytes coincided with GPU committed13,385,277,440 bytes: numerical difference365,203,456 bytes, **not8GB**. This is not a causal disjoint subtraction either. Exact driver allocator/callsites/permanent-system-memory flags remain unavailable. File-backed pages can be reclaimed under pressure, but later expert/prefill access may incur page-in cost; no forced working-set trim or cheap-reclaim claim was used to improve these results.

Source and exact byte evidence are in `results/memory-correction.json` and `results/memory-attribution-attempt1.json`. The legacy1.36GB “resident” value is withdrawn, not silently replaced in historical records.

## Original checkpoint standalone matrix (preserved)

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

## Accepted F1: matched final-policy n12 cache region

Every row below uses the identical fixed43-request fixture, S/R0 policy and actual26-token median hot evaluation, rather than the unrefined511-token hot path. Original binaries/model/prompt hashes, temperature, seed, output cap and4608/3 are unchanged. No original run or compact per-run ledger was overwritten; guards now reject duplicate run identities and aggregation leaves existing compact ledgers untouched.

The first2048/OFF/1024 cohort encountered actual .48–.92GB free RAM and paging. Background memory released naturally during1024; no unrelated process was changed. The3072 run then retained5.26GB free. Only OFF/1024/2048 were repeated with new `repair2` identities. Global paging bursts remain, but recovered rows did not exhibit the near-zero-headroom condition; medians and full tails remain, not a fastest-sample selection.

| Final-policy n12 run | Dedicated GB | WS / private GB | Min free GB | Hot median | Pair-alt median / p95 |3-slot median / p95|CPU % one core|
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
|2048 `repair-control`, pressure-qualified|12.780|14.75/13.64|.482|.412|.560/.804|.832/1.158|6.69|
|OFF initial, pressure-qualified|10.631|14.93/11.72|.531|.513|.662/1.080|1.124/1.660|98.74|
|1024 initial, condition changed mid-run|11.704|14.78/12.63|.820|.605|.784/1.044|1.552/1.862|7.57|
|OFF `repair2`|10.631|14.94/11.72|3.899|.517|.652/1.015|1.095/1.570|126.37|
|1024 `repair2`|11.704|14.78/12.63|4.618|.596|.781/1.034|1.605/1.978|7.43|
|2048 `repair2-control`|12.780|14.78/13.71|5.096|.398|.536/.774|.824/1.356|7.95|
|**3072**|**13.856**|**14.78/14.79**|**5.263**|**.343**|**.397/.635**|**.544/.842**|**3.54**|

OFF is faster than1024 in this standalone fixture but consumes approximately1.26 cores rather than .074.2048 improves latency and avoids that CPU cost;3072 further improves pair and occupied-slot latency for approximately1.076GB extra dedicated/private charge. All four remain genuine resource tradeoffs, so all four were exercised through the same isolated30-text/restart/release app workload before revising selection.

### Split native prefill and decode — local final-policy alternation

Durations are **median/p95 ms**, rates median tokens/s, n20 per row. P is actual evaluated uncached prompt tokens (median28 in each row), not the approximately900-token total cached prompt.

| n12 cache | prefill_ms | prefill_tps | decode_ms | decode_tps |
| --- | ---: | ---: | ---: | ---: |
|OFF `repair2`|178.51/531.69|158.75|354.24/488.62|44.04|
|1024 `repair2`|284.70/472.46|109.63|408.86/536.43|37.37|
|2048 `repair2-control`|183.17/397.40|197.70|281.44/361.89|55.97|
|**3072**|**92.55/335.50**|**392.31**|**226.01/298.07**|**68.04**|

**Complete all-options tables, not finalists only:** `TIMINGS.md` covers all35 matrix identities (including the failed4096 row), every app/reference attempt, first-prefix/hot/pair-alternation/three-slot matrix phases, actual adjacent hot/changed-pair app groups, separate first-app-text groups and the six attribution texts. `results/timings/` supplies per-run machine-readable sidecars; `results/timing-summary.json` retains every observed count plus n/median/p95/min/max for each duration/rate. App three-slot timing is N/A because app submissions were sequential; standalone occupancy is not falsely relabeled app concurrency.

Native output-count conventions matter: an observed21-token output/printed perf count with423.682ms has reported47.205215tokens/s, corresponding to a rate numerator20. We preserve native TPS, not21/duration. `timings.cache_n` is prior prompt reuse; native response `tokens_cached` is final KV occupancy, **not** a prompt-hit count. Managed printed decode count is unavailable (stock stderr DEVNULL), not estimated from output. Generic app native timings are ordered associations; concurrent standalone logs are assigned only across phase barriers, not false per-request joins.

## Vulkan execution and cache limits

The revised selected3072 profile reports31/31 offloaded layers, **CPU_Mapped5663.88MiB**, **Vulkan model8673.35MiB**, **3069.19MiB cache for4900.50MiB of host experts**, GPU KV990MiB and compute328.88MiB. These allocation logs are distinct from OS residency/commit. Standard maximum-layer offload still retains a396-MiB CPU-mapped wrapper; “ngl99/full” does not mean every tensor resides exclusively in VRAM. Ordinary partial14 reports14/31 layers and7857.00/6112.86MiB CPU-mapped/Vulkan buffers plus CPU/GPU KV/compute.

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
| **n12/cache3072/S/R0** | **57,185 /1,809 /96.93%** | **5,771.49** | **10,226 /667 /2,128.02** | **29,046 /46,334.61MiB** |

These are context-lifetime logical transfer counters, including warmup/prefill/serving, not measured physical PCIe bandwidth or pure-decode-only rates. The `>8` slot-map counters cover eligible small batches; separate large-batch-copy counters preserve the prefill distinction.

**Retained failure:** cache4096 could not initialize a **2,861,438,976-byte bank**, with `Requested buffer size exceeds device buffer size limit: ErrorOutOfDeviceMemory`. Pinned Vulkan buffer code rejects `size > device.max_buffer_size` before allocation. Cache3072 works. This is a per-bank/backend/device limit, **not total 24-GiB exhaustion and not an 8-GB eligibility rule**. Larger 6/8-GiB budgets were not claimed tested; monotonic bank growth after this limit is not a useful sweep without upstream changes, which are out of scope. No unsupported size override, CUDA substitution, model pruning or quantization substitution was attempted. Direct Strata implementation provenance is not established.

## Actual unmodified PuriPuly probe

Each profile used its own source CLI instance and owned native server. Before first startup the probe seeded **new** canonical settings, probe-only encrypted-file secrets, telemetry/clipboard off, keyless Soniox selections, capture off and OSC connection mode off. LOCALAPPDATA, APPDATA, USERPROFILE, HOME and temp roots were redirected; inherited provider/native overrides were removed. No live settings/secrets/logs/audio were used.

**OSC off does not silence Self publication.** An exclusively bound worker-owned loopback UDP sink replaced the destination, with both configured send-port fields pointing to it. It stayed bound through teardown. Receiver/OSCQuery were effectively off, overlay desired/effective off, microphone/capture inactive, and source process observation was read-only against an isolated scene-log root. Each successful run received 90 synthetic datagrams (typing/output); this is local sink evidence, not VRChat delivery or GUI presentation.

The supported CLI supplied current `capabilities` and necessary `settings choices`, enabled translation, applied language changes with expected revisions, submitted UTF-8 synthetic files, and followed content-enabled translation events. Every completion was matched to an admitted parent/child turn ID. Persisted language and effective translation-enabled/ready state were rechecked. The second app instance started with the intentionally persisted **en→ko** pair rather than the original **ko→en**, a new instance/PID, and translation disabled until explicitly enabled again. External server/KV survived app-only restart; this is **not** managed model-restart/prefix-disk-cache readiness.

### Controlled inactive-STT degradation

The first probe aborted before text submission because language application returned terminal **degraded**, `settings_commit_success_runtime_degraded`, specifically `provider.stt / stt.provider_not_ready`: intentionally keyless Soniox cannot build. That failed attempt remains recorded. The Director then explicitly authorized **manual translation with inactive STT**, not a fake provider or a product repair.

Successful probes retain **31 degraded language-settings receipts per run (403 total across13 completed app runs)**, verify exact STT attribution, and separately prove persisted languages, effective translation and content. No broad healthy-settings/ASR/audio claim is made. Earlier redundant final `app stop` missing-instance receipts remain; they occurred after already observed host exits, not as translation failures.

The **generic-endpoint** app probes use non-streaming `local_llm`/`ollama`, real rolling last-five source entries/60s/pair-matched history, actual rendered prompts, temperature .6 and reasoning off. Their server-side limit96/seed42 are experimental overrides, not normal product request settings. **Wall metric = text-submit CLI launch → matched completed translation event**, including CLI/admission/translation/observation; excluding audio collection, endpointing, STT and GUI rendering. It is not pure native inference or audio end-to-end latency. App TTFT is unavailable. Split native timings are separate.

| Original generic-endpoint app profile |n|Server GPU GB/GiB|Server WS/private GB|**Text-submit→translation-completion** median/p95/max s|Whole-device dedicated GB|
| --- | ---: | ---: | ---: | ---: | ---: |
| E4B b10423 **generic endpoint, NOT managed product latency** |30|2.855/2.659|3.72/3.97|.702/.774/3.328|5.925|
| cpu-moe off/S/R0 | 30 | 2.905 / 2.706 | 14.32 / 4.26 | 1.811 / 2.007 / 6.415 | 5.983 |
| cpu-moe cache3072/S | 30 | 6.130 / 5.709 | 14.25 / 7.20 | 1.769 / 2.279 / 6.459 | 9.207 |
| cpu-moe cache3072/S/R0 | 30 | 6.130 / 5.709 | 14.25 / 7.20 | 1.748 / 2.194 / 6.435 | 9.208 |
| partial14/S/R0 | 30 | 7.380 / 6.874 | 15.31 / 9.11 | 1.388 / 1.782 / 4.945 | 10.458 |
| n24/cache2048/S/R0 | 30 | 7.629 / 7.105 | 14.40 / 8.65 | 1.574 / 2.044 / 5.660 | 10.706 |
| Original n12/cache2048/S/R0, historical context |30|12.780/11.903|14.78/13.75|.795/1.412/3.841|15.857|
| ngl99/S/R0 | 30 | 15.438 / 14.378 | 15.17 / 16.44 | .700 / 1.203 / 2.780 | 18.514 |

The original headless generic app used approximately .117–.119GB sampled WS and .60GB private commit. Revised3072 used .119GB/.603GB; true managed app used .135GB/.619GB. No app GPU-process row was registered, so absence remains unavailable rather than zero; whole-device memory separately encloses background allocations. Revised3072 app server shared peak was614,989,824 bytes; whole-device dedicated17,429,037,056 bytes leaves approximately8.324GB of the observed25.753-GB capacity. Its minimum available RAM was2,016,010,240 bytes. No8-GB card, GUI or VRChat co-render workload was tested; a sub8GB server can still exceed an8GB whole-device budget with desktop background.

### Current-condition F1 app comparison

All four completed30 texts, persisted en→ko into a fresh app host, explicitly reenabled translation and released both hosts/native server. The original2048 app result remains valid for its original context; subtle current comparisons use the new2048 control.

| n12/S/R0 generic endpoint |n|Dedicated GB/GiB|WS/private GB|**Text-submit→translation-completion** med/p95/max s|Server CPU seconds over observed window|Min free GB|
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
|OFF `repair2`|30|10.629/9.899|14.97/11.81|1.053/1.515/5.316|31.19 over76.25s|.722|
|1024 `repair2`|30|11.754/10.947|14.79/12.68|1.056/1.644/3.936|10.30 over72.85s|1.007|
|2048 `repair2-control`|30|12.780/11.903|14.78/13.76|.807/1.409/3.964|8.72 over66.29s|4.772|
|**3072**|**30**|**13.856/12.904**|**14.77/14.83**|**.681/1.395/4.193**|**5.59 over62.99s**|**2.016**|

OFF/1024 app close comparisons are pressure-qualified, not proof that their .003s median difference matters. Both remain demonstrated manual-translation alternatives; clean standalone observations establish their CPU/cache tradeoff.3072 improves median and standalone occupancy over current2048, while app p95 is nearly equal and max is worse. Thus it is not a universal tail improvement or a guaranteed production SLA.

### True managed E4B reference — not the earlier .7s generic result

The original `e4b-product` app used `/v1/chat/completions` and **did not exercise managed language-pair slot ownership or prefix preparation**. Its .702s wall,348.825ms prefill/183ms decode and82tok/s remain valid generic-endpoint observations, not current managed product latency.

The additional unmodified app selects **managed_gemma/gpu/Vulkan0**, stages both exact E4B and installation-required MTP assets plus the authoritative manifest into redirected state, and validates both SHA256 values before activation (MTP59,678,016 bytes, SHA423074e537504b4f9ec5eafed5c639fac82c96631626efccacdd3c4039b20605). GPU serving does not load the drafter. The verified complete b10423 Vulkan DLL tree is under an explicit owned `runtime/vulkan` root; CPU executable is absent to **fail closed**, not test stock CPU fallback. No provisioner download or product edit was needed.

Two fresh hosts owned two distinct model processes; exact stock commands, three1536-token slots, explicit Vulkan device and nonzero RX process allocations are observed. Both model/host processes exited with their owners. Persisted en→ko, normal translation disabled-on-start, completed UUIDs and controlled STT degradation are retained. The isolated persistent prefix cache contains four `.vulkan.bin` files; the managed path prepares/pins three slots and uses save/restore/LRU rather than generic endpoint slot choice. Some restored/prepared requests nevertheless re-evaluate nearly the full prompt in this four-pair workload: **22/30 have zero reported prompt-cache hits**; this is observed behavior, not a product fix or an assumption that explicit preparation guarantees hot-prefix speed.

Stock native stderr is DEVNULL, and legacy `[ManagedGemma][Performance]` text was privacy-redacted. The harness preserved its initial `inconclusive_effective_backend_or_metrics` classification. Separate recovery uses **30 structured `[Diagnostic][LlmLatency]` native response rows**, correlated by host PID/request ID through request_end to completed utterance UUID. Vulkan evidence is explicit owned Vulkan children, nonzero GPU allocation and absent CPU executable, not merely requested `models.status.backend=gpu`. No inference rerun or logging-policy bypass was used; absent native printed decode counter remains N/A.

The final capture helper now consumes those structured diagnostics directly. Its **same extraction/classification function and extraction-only CLI were exercised against the retained synthetic log** into an external throwaway record:30 distinct successful terminal PID/request records joined to30 completed UUIDs, prefill median1035.382ms/805.379tok/s and decode196.4105ms/81.8045tok/s, plus3 native prefix-evaluation and28 zero-evaluation apply/restore windows. `results/managed-extraction-check.json` records the command, exit0 and phase values. This is not a claim that the late live-launch path was rerun; original actual-app evidence and initial classification remain immutable.

| True managed E4B stock b10423 |n|**Text-submit→translation-completion** med/p95/max s|Dedicated bytes /GB/GiB|WS/private GB|Min free GB|
| --- | ---: | ---: | ---: | ---: | ---: |
| Four-pair, two fresh model/app processes |30|**1.408/1.499/1.506**|2,855,346,176 /2.855346 /2.659248|4.667/4.914|11.679|

**Prefix preparation is outside that text timer.** Translation-enable+initial-prefix wall was8.136s first process and2.753s fresh restart. Language apply/prefix prepare-or-restore n30 wall median .265s, p95 1.284s, max1.331s; before-restart persistence is a separate .268s operation. Native bracketed evaluation windows n3 had prefill median979.38ms, p95 1004.02ms, median821.70tok/s with median819 evaluated tokens.28 bracketed apply/restore windows evaluated zero new tokens, so TPS is unavailable rather than inflated. Two unbracketed initial-enable native windows are N/A; cumulative snapshots are not individual durations. Prefix n_predict1's1µs decode timer floor is not a decode-throughput result.

### Split native translation timings — app comparison

All rows n30. Times are native median/p95 **milliseconds**, TPS is median native rate; these do not include CLI/observer overhead or prior managed prefix preparation. P is actual uncached evaluated prompt count, D is the native printed count (not a traced pure forward-pass count); generic C/total/API-output counters were not persisted.

| App path/profile |prefill_ms|prefill_tps|decode_ms|decode_tps|P median /C median /D printed median|
| --- | ---: | ---: | ---: | ---: | --- |
| n12 OFF `repair2` |513.50/1010.53|430.53|378.69/495.69|41.59|222.5 /N/A /17|
| n12 cache1024 `repair2` |443.69/1053.81|502.65|446.88/509.98|35.36|222.5 /N/A /17|
| n12 cache2048 current control |365.08/945.25|611.86|281.95/340.48|56.14|222.5 /N/A /17|
| **n12 cache3072** |**285.18/987.10**|**806.20**|**229.17/342.39**|**67.39**|222.5 /N/A /17|
| E4B generic endpoint, historical |348.83/378.37|626.55|183.00/220.37|82.00|218.5 /N/A /15|
| **E4B true managed stock** |**1035.38/1099.24**|**805.38**|**196.41/268.34**|**81.80**|863 /0 /N/A|

The managed row's actual total-prompt median877 and API-output median17 are separately recorded. Its large uncached-token count explains the one-second native prefill despite comparable decode TPS. This is not evidence that26B intrinsically outperforms E4B: model/provider prefix policy and sampling limits differ. `TIMINGS.md` contains **all** earlier/refined/failure rows and app groups, not only this compact decision table.

## Quality and final selection reasoning

See `QUALITY.md`. Independent review remains the first344 outputs. The execution owner inspected every unique pair in the **1,462-output/12-source matrix corpus**, all **390 app outputs/five sources** via their unique variants, and the six bounded attribution texts. No observed configuration-specific material meaning reversal, central quantity/negation corruption, repetition, visible reasoning or truncation was found. This remains synthetic spot-check evidence, not broad quality certification or26B superiority.

Caveats remain: ambiguous 小林 reading (standalone Kobayashi, QAT app Xiao Lin); **61/198 QAT five-minute occurrences** use “within” for “in,” also in old controls; E4B adds unsupported gender and sometimes omits a standalone noun. True managed E4B keeps the same name/gender caveat but preserves the central negation/Friday change. Literal-digit false flags on correct “twice/three” remain; no assertion was removed, weakened or suppressed.

The measured frontier has distinct, non-dominated resource choices:

- **Minimum dedicated/private commit:** cpu-moe off/S/R0, approximately 2.905-GB GPU and 4.18-GB standalone private commit, but approximately 1.43 core-equivalents serving CPU and 1.811-s app median.
- **Sub-8-GB, low residual CPU:** all-host cache3072/S/R0, approximately 6.13-GB GPU, approximately .09 core-equivalents and 1.748-s app median; cache misses/transfers limit latency. n24/cache2048 uses 7.63 GB and improves app median to 1.574 s.
- **Ordinary partial placement:** 7.38-GB partial14/S/R0 reaches 1.388-s app median but costs approximately 1.67 core-equivalents. It is not an equal-CPU-cost alternative.
- **Revised low-CPU latency/resource recommendation:** n12/cache3072/S/R0 at13.856GB sampled dedicated, .681s current text-submit→completion median and96.93% small-batch hit rate. It saves approximately1.58GB dedicated versus historical maximum offload while retaining a much stronger local frontier than1024/2048.
- **Current2048 alternative:**12.780GB, .807s current wall median, almost equal p95 and lower cold max than3072. It saves1.076GB dedicated/private charge and remains the explicit owner alternative if headroom matters more than .126s median/occupied-slot gains.
- **Historical maximum-offload reference:**15.438GB and .700s original wall median. The different-context .700 versus current .681 is **not** a matched claim that caching beats full offload. Current selection uses the contemporaneous n12 budget comparison; original maximum offload remains eligible context evidence.

The revised recommendation is a measured experimental choice, not an approved SLA or owner-approved deployment. **Owner decision before adoption:** accept3072's extra1.076GB GPU/private commitment for the current median/occupancy gain over2048, or choose2048 / a documented lower-memory option. Actual resident RAM is a separate whole-GGUF mmap cost, not the private-commit number. Retain shipped E4B until separately authorized managed26B integration/rollout; the true managed reference and generic endpoint reference are now explicitly separate.

## Reproduction and handoff

**Preserve frozen records:** use another copy of the same baseline checkout/task files, an empty result directory and a new owned root seeded only with verified artifacts. Do not reuse `runs/<profile>` or app-state directories. Duplicate matrix IDs are rejected; aggregators retain existing per-run compact ledgers. Original eight records and all failed/pressure attempts remain immutable. Raw identity capture stays external until relevant facts are projected/sanitized.

Use the supplied interpreter environment with repository dependencies (psutil included), the frozen checkout and a new owned artifact/result directory. Do not point the app probe at live settings. Preserve existing verified downloads rather than reacquiring them.

```powershell
$python = '<APP_PYTHON>/python.exe'
$root = '<NEW_OWNED_ARTIFACT_ROOT>'
$e4b = '<EXISTING_VERIFIED_E4B_GGUF>'
& $python experiments/gemma4_26b_qat/acquire.py --root $root
& $python experiments/gemma4_26b_qat/capture_identity.py --root $root --e4b $e4b --output "$root/identity-private.json"
& $python experiments/gemma4_26b_qat/run_matrix.py --root $root --e4b $e4b --profiles new-ncpu12-cache3072-swa-nocram --repetitions 5
& $python experiments/gemma4_26b_qat/app_probe.py --root $root --e4b $e4b --profile new-ncpu12-cache3072-swa-nocram --attempt 1
& $python experiments/gemma4_26b_qat/managed_e4b_probe.py --root $root --e4b $e4b --attempt 1
& $python experiments/gemma4_26b_qat/summarize_timings.py --results experiments/gemma4_26b_qat/results --raw-root $root --markdown experiments/gemma4_26b_qat/TIMINGS.md
& $python experiments/gemma4_26b_qat/summarize.py --results experiments/gemma4_26b_qat/results --raw-root $root
& $python experiments/gemma4_26b_qat/summarize_apps.py --results experiments/gemma4_26b_qat/results --raw-root $root
```

`run_matrix.py` contains every tested profile; per-run command arrays are the exact resolved commands, including observed ports. Run inference sequentially with exclusive benchmark ownership. The sampler's adapter LUID is the observed boot's RX LUID; re-establish it before another boot/device and do not accept missing counters as zero. `app_probe.py` reconstructs the server from its standalone record; the owned app state directory must be new. Root placeholders mean path substitution, not model substitution. Raw model/native logs stay external. Normalized UTF-8 text-log hashes in run records are not advertised as raw-file checksums.

No production build, pytest suite, linter or formatter was run; this assignment exercised model/app experiments, not product edits. The Director should freeze/review the task-local candidate and its evidence, then publish the authorized experimental branch and English result/status update. **Unresolved product-owner adoption is not a blocker to completing, reviewing or publishing this experimental report.** Additional GPU benchmarking is not required to read this evidence. Production rollout requires a separately authorized implementation of the26B managed artifact/profile/lifecycle contract; the generic endpoint result does not establish that readiness.

### Issue #245 checklist disposition

| Requirement | Disposition |
| --- | --- |
| Exact target verified; old/new identities, metadata, hardware/toolchain frozen | Done, with explicit unavailable release-run details and metric limits |
| Required pinned runtime upgrade exercised, actual Vulkan and cache execution | Done on b11537, including native hit/miss/upload and large-batch-copy counters |
| Ordinary partial, CPU expert offload, matched multiple caches, below/near/above preferred target | Done; 1024/2048/3072 budgets, both n-cpu placements, 4-GiB failed bank retained |
| Real prompt, multilingual inputs, repeated cold/retained/pair/three-slot workload at4608/3 | Done; diagnostic cold n and fixed-vs-app history distinctions explicit |
| Refine one mechanism at a time; failed/pressure records/resource ledger | Addressed F1 with matched n12 OFF/1024/fresh2048/3072 final-policy rows, pressure-qualified first cohort/recovered repeats and same30-text app probes |
| Actual app language/restart/release, true E4B reference | Done for manual text:12 generic runs plus true managed E4B; STT degradation retained; no ASR/GUI or managed26B readiness claim |
| Every option's split prefill/decode ms/TPS and real counts | `TIMINGS.md` and per-run sidecars cover all35 matrix/14 app attempts, phase groups and failures; unavailable counters explicit |
| Exact RAM attribution / incorrect psutil fields | Withdrawn old interpretation; four successful no-touch64-bit snapshots distinguish mapped GGUF, valid private pages, private commitment and GPU-sized write-combined ranges |
| Quality/frontier/selected profile | Revised n12/cache3072/S/R0 selected above preferred target;2048/lower-memory choices and owner tradeoff explicit; bounded quality caveats retained |
| Integration/rollout recommendation | Retain shipped E4B until separate authorization; no deployment/merge or production modifications performed |

### Primary sources

- [Issue #245 contract](https://github.com/kapitalismho/PuriPuly-heart/issues/245), [#75 including amendments](https://github.com/kapitalismho/PuriPuly-heart/issues/75), [#64 decisions](https://github.com/kapitalismho/PuriPuly-heart/issues/64).
- [Pinned target artifact](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-qat-GGUF/blob/7b92b5b28818151e8669af2e45e88d6086f490dd/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf).
- [b11537 release](https://github.com/ggml-org/llama.cpp/releases/tag/b11537), [cache implementation](https://github.com/ggml-org/llama.cpp/blob/8e2d31e0eb658d0382d3106a61369dff81f0c080/src/llama-moe-cache.cpp), [Vulkan buffer limit](https://github.com/ggml-org/llama.cpp/blob/8e2d31e0eb658d0382d3106a61369dff81f0c080/ggml/src/ggml-vulkan/ggml-vulkan-buffers.cpp#L28-L33), [Vulkan offload policy](https://github.com/ggml-org/llama.cpp/blob/8e2d31e0eb658d0382d3106a61369dff81f0c080/ggml/src/ggml-vulkan/ggml-vulkan.cpp#L16019-L16023).
- [Microsoft WDDM GPU accounting](https://devblogs.microsoft.com/directx/gpus-in-the-task-manager/): API-independent accounting, dedicated/shared meanings and busiest-engine utilization semantics.
- Frozen local `docs/architecture.md`, `docs/cli.md`, `src/puripuly_heart/core/local_translation/runtime_profile.py`, prompt assembly, application-control and provider sources at the product baseline.
