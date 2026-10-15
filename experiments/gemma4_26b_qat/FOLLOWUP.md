# Issue245 follow-up — public cases, Vulkan staging, and current-RAM mmap/read tradeoff

## Result and boundary

**Completed the approved bounded investigation: four logical arms plus one justified mmap bracket, five native-server lifecycles,173 completed synthetic translations.** Four runs completed43/43 and exited gracefully0; the first all-host mmap run completed1/43 before an owned memory-safety cutoff. No additional32/33 sweep, profiler,4096-cache repeat, model/backend change or app launch was performed.

The direct, source-backed local finding is that the supported **`--load-mode none` (read/non-mmap)** changes the observed model host buffers from `CPU_Mapped` to **`Vulkan_Host`** and materially improves these cached-MoE native workloads. For n24/cache2048, the gain remains after a later mmap control has substantially better headroom. This is a measured **mode/placement/OS-memory tradeoff**, not an exact partition of every millisecond into transfer, paging and host synchronization.

**Original product recommendation and app claims remain unchanged.** This follow-up tests standalone serving only. Original actual-app feasibility exercised mmap endpoints; none-mode app latency, managed26B lifecycle, audio/STT, GUI and actual VRChat co-rendering are not demonstrated. Retain the published recommendation/owner adoption tradeoff and shipped E4B pending separate integration/rollout authority. Do not promote this follow-up into deployment approval.

Experiment baseline is published/reviewed `eff651d517c7fd15ef2f96a019b348cccf662d06`, product baseline `7a8151be6114a950f062e2430bd36f899d57e4de`. All original per-run records, failures, reports and selection are preserved. New records are under `results/followup/`; raw outputs are external under `<ARTIFACT_ROOT>/followup/`.

## 1. Published cases were read first

These are primary-source reports, not our reproductions. Complete pins, attribution and missing fields are in [`public-cases.json`](results/followup/public-cases.json).

| Published case | What it actually measures | Why it does not equal our baseline |
| --- | --- | --- |
| [PolyStrata physical8GiB Gemma](https://github.com/VecSzn/PolyStrata/blob/c566ff4787c8ab188ecf33c6f1907a9f90d9f152/docs/BACKENDS.md#on-a-laptop-with-8-gb-of-vram) | Custom CUDA engine1212621baadf;4070Laptop8GiB/i7-14700HX/32GB/Win10/13expert workers. Non-QAT Gemma UD-Q4_K_M;actual31,063-token document2146–2148prefill tok/s,85short-chat decode,61after long prompt;card7.7–7.8GiB,three runs with first omitted. | Different Q4_K/Q5_1 artifact,1024-token staged prefill,hybrid CPU misses/GPU hits,guessed tokens and warm adaptive profile;one-at-a-time rather than our3-slot workload. Whole-card7.7–7.8GiB≈8.27–8.38decimalGB is not a6GB runtime. Exact draft acceptance/output count/PCIe/storage and isolated process peaks absent. Its17GB file is larger than ours: quantization alone is not an adequate explanation. |
| [Original Strata](https://github.com/Niko1221/Strata/blob/296b59d2f73c8f04a805df29c2fa97655adace78/docs/DETAILS.md#speed-measured) | Qwen125B customQ2/IQ,5070 12GiB/64GB/PCIe5x16;large prompts,256output,MTP. Example0.1.26 Q2:1299pp/93TG at3562P,2171pp/82TG at31566P;3.23tokens/verify at4K. | Not Gemma or measured6–8GB total runtime. Do not apply Qwen's3.23 acceptance to Gemma. Newer headline94/2650 is another engine version/cell. |
| [Independent RX6600 physical8GiB](https://github.com/Niko1221/Strata/pull/1378) | QwenIQ3_XXS,HIP/EPYC/128GB/8.6K fresh prefix/512output,n2:~263prefill,24decode;1.62GiB expert cache. | Different model/backend;startup layout,not lifetime peak. Public8GiB results are not uniformly100tok/s. |
| [Closest Gemma/Vulkan mmap report](https://github.com/ggml-org/llama.cpp/issues/21590#issuecomment-4207665784) | RX9070XT/Gemma26B/CPUexperts:pp128116→291 with no-mmap;TG27.0→26.4. | Quant/build/OS/CPU/RAM/complete commands/repeats absent. Maintainer explains device-mapped host weights avoid pageable staging. Proposed direct BAR write violates async command ordering and was not adopted. |
| [Upstream cache introduction](https://github.com/ggml-org/llama.cpp/pull/29887), [multiGPU follow-up](https://github.com/ggml-org/llama.cpp/pull/30112) | Explicit **qvac-fabric port**, not established Strata adoption. Qwen/EPYC/CUDA/512output;two4090 casePP440.7→321.5 whileTG34.78→64.39. Sparse7900XTX Vulkan anecdote~35end-to-end tok/s lacks precise budget/PP/TG/OS. | Upstream uploads each miss then computes it on GPU;it is not concurrent CPU-miss/GPU-hit execution. Expert allocation is not total VRAM. Author says Vulkan batched upload not fully optimized. |
| [Actual Strata-inspired llama extension](https://github.com/ggml-org/llama.cpp/pull/30594) | Unmerged fork,implementationf6dea514,documentation2ce49abf;CPU misses/GPU hits,profile fill/adaptation. Fast one-token mailbox is NVIDIA CUDA-only;benchmarks gpt-oss/Qwen on4060Laptop/WSL2. | No Vulkan mailbox/benchmark or exact GemmaQAT reproduction. Adaptive swaps still upload/synchronize. Not an available supported knob in our pinned runtime. |

PolyStrata's pinned engine overlaps CPU misses with GPU hits via the expert-worker GPU callback, overlaps window halves, uses pinned asynchronous adaptive cache following and layer-major double-buffered prefill staging (default1024,stage>=32) to reuse a layer's staged experts. Its Gemma engine guesses tokens, but acceptance for the advertised pass is unavailable. These architectural/speculation/batch differences matter alongside kernels and quantization. Newer Strata's semantic tail-skipping must not be imported; the GemmaP1 case is **not** thereby alleged to skip experts.

No inspected source establishes an exact external Windows/Vulkan/QAT replication. This is bounded research, not proof none exists. Newer Vulkan kernel PRs are not demonstrated fixes for our short-context/cache case, so no compiler/runtime update was attempted here.

## 2. Fixed controls and user-approved current-load boundary

Same verified target: `gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf`,14,249,047,104B,SHA256 `a7c5bc715f5ff8e99a3e8901ce7d2b42b402c669bf24f7c5250747633d0f5891`. Actual expert tensors remain q4_0,not a substituted quantization. Same official b11537/`8e2d31e0eb658d0382d3106a61369dff81f0c080`,Windows11/9800X3D/32GiB/RX7900XTX25,753,026,560 usable runtime bytes;identity/toolchain sources remain the original report. Each new analysis records actual server executable SHA256;current API/driver are reobserved below.

Common native controls stay totalctx4608/parallel3/effective1536 each,Vulkan0,4/4threads,batch512/ubatch512,f16K/V,mmap-or-none,flash attention on,fit off,reasoning off/budget0,warmup,perf,metrics,HTTPthreads1 and the original owned-process priority. No CPU/GPU fallback override,forced pin/lock/trim,driver/system change,other model,draft or paid compute.

Placements: all-host experts+3072MiB cache and first24 host-expert layers+2048MiB cache,each ngl99/SWA-full/cache-ram0. Only load mode changes within each pair. Source-known GGML controls are captured **before and after removal in the owned child**;all effective values are absent,including limit overrides and profiling flags. Presence-based flags are not disabled by `=0`. This is an allowlist,not a secret/full-environment dump. Historical `build-provenance.json:18` recorded GGML* empty before refinement,but cannot reconstruct every exited historical child's environment;neither contamination nor complete historical absence is inferred.

The exact original43-request fixture is reused:12source sentences,four pairs,real rendered prompt plus five fixed synthetic context lines,first-prefix3(temp0),hot5,pair-alternation20,three-slot15(temp.6/seed42/limit96/EOS). Context/slots/history/quality gates were not narrowed. All43 n24 request identities,prompt hashes,order,**actual uncached prompt_n and cache_n** match acrossA1/read/A2;two concurrent read outputs change length by±1token with equivalent wording,retained in [`request-matching.json`](results/followup/request-matching.json). API timings are per request;concurrent native printed counters are joined only at phase barriers,not invented per-request associations.

Preflight initially found11.63GB free,no other model runtimes,low CPU. The user explicitly approved existing limited RAM as a realistic concurrent-load condition rather than freeing RAM. We changed/terminated no unrelated processes. This is **not an actual VRChat render/frame-time benchmark**. Background/OS headroom later rose through15.15,16.12 and17.22GB;we observed it rather than forcing it. Consequently the observation is practical viability plus bounded mechanism evidence,not perfectly isolated transfer causality.

An owned-only watchdog stops after8continuous seconds below256MiB physical available **or**512MiB commit headroom. Existing360s readiness/300s request bounds remain. The cutoff is a safety policy,not a runtime OOM test or permission to alter system memory.

## 3. Observed startup, completion and all-option native phases

**Native prefill/decode includes engine CPU/GPU/synchronization/paging work,not pure GPU kernel time.** Standalone HTTP wall includes `/apply-template` and streaming completion. Neither is actual-app CLI text-submit→completion or speech/audio E2E.

| Profile | Ready n1, s | Ready→first submit², s | First HTTP completion, s | Process launch→first completion, s |
| --- | ---: | ---: | ---: | ---: |
| All-host3072 mmapA1 |5.992|1.851|22.725|30.569|
| All-host3072 readB1 |10.397|0.279|1.550|12.226|
| n24/cache2048 mmapA1 |5.782|1.848|4.427|12.056|
| n24/cache2048 readB1 |7.886|0.298|1.375|9.559|
| n24/cache2048 mmapA2 bracket |4.934|1.796|4.469|11.199|

² Includes existing mapping/residency observation and prefix token-metadata queries,not only idle delay. There is no OS/shader/file-cache flush;native warmup is on and global expert/OS warmth differs from per-slot KV reuse. n1 startup observations are not a repeated startup distribution.

For the identical initialP912 source,first mmap prefill21.304s/42.81tok/s becomes read1.030s/885.61tok/s,but **this is not a21×total-cold speedup**. Read does more loading before ready(10.397vs5.992s);observed launch→firstcompletion is12.226vs30.569s and includes harness preparation. The mmap run then safety-stops,so it cannot supply new hot/pair/occupied medians. Historical mmap hot results remain descriptive for their old conditions,not a clean current comparator.

Every new option/phase is below,including failedN0 phases;[`FOLLOWUP_TIMINGS.md`](FOLLOWUP_TIMINGS.md) and [`firstwave/timing-summary.json`](results/followup/firstwave/timing-summary.json) retain distributions/min/max and per-run sidecars. P=actual uncached prompt evaluations,C=prior prompt KV hits,O=API output,D=native **printed** generation counter. Native TPS is retained:printedD/O can include a token not in the rate numerator;never divide full cached prompt by prefill time or substituteO for unavailable traced forwards.

| Phase / profile | n | P / C / O / D printed med | prefill_ms med/p95 | prefill_tps med | decode_ms med/p95 | decode_tps med |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| first_prefix: All-host3072 mmapA1 |1|912.0 / 0.0 / 21.0 / 21.0|21303.53/21303.53|42.81|1394.55/1394.55|14.34|
| first_prefix: All-host3072 readB1 |3|908.0 / 0.0 / 13.0 / 13.0|958.94/1029.80|941.66|350.24/485.10|34.26|
| first_prefix: n24/cache2048 mmapA1 |3|908.0 / 0.0 / 13.0 / 13.0|1902.71/3662.22|474.59|582.19/735.62|23.93|
| first_prefix: n24/cache2048 readB1 |3|908.0 / 0.0 / 13.0 / 13.0|860.30/897.55|1049.63|311.19/444.52|41.64|
| first_prefix: n24/cache2048 mmapA2 bracket |3|908.0 / 0.0 / 13.0 / 13.0|2064.84/3653.41|439.74|576.33/798.88|24.71|
| hot_same_pair: All-host3072 mmapA1 |0|N/A / N/A / N/A / N/A|N/A|N/A|N/A|N/A|
| hot_same_pair: All-host3072 readB1 |5|26.0 / 884.0 / 15.0 / 15.0|176.92/212.45|142.09|338.39/518.84|41.56|
| hot_same_pair: n24/cache2048 mmapA1 |5|26.0 / 884.0 / 15.0 / 15.0|457.80/537.88|53.74|516.57/799.06|28.04|
| hot_same_pair: n24/cache2048 readB1 |5|26.0 / 884.0 / 15.0 / 15.0|164.25/192.56|154.07|303.52/446.98|46.13|
| hot_same_pair: n24/cache2048 mmapA2 bracket |5|26.0 / 884.0 / 15.0 / 15.0|452.42/550.27|56.44|450.14/690.91|31.10|
| pair_alternation: All-host3072 mmapA1 |0|N/A / N/A / N/A / N/A|N/A|N/A|N/A|N/A|
| pair_alternation: All-host3072 readB1 |20|28.0 / 878.0 / 16.5 / 16.5|204.54/418.38|153.71|400.63/522.94|36.55|
| pair_alternation: n24/cache2048 mmapA1 |20|28.0 / 878.0 / 16.5 / 16.5|529.05/829.71|61.03|630.97/854.85|22.79|
| pair_alternation: n24/cache2048 readB1 |20|28.0 / 878.0 / 16.5 / 16.5|184.96/378.45|168.34|363.51/504.88|40.52|
| pair_alternation: n24/cache2048 mmapA2 bracket |20|28.0 / 878.0 / 16.5 / 16.5|492.03/806.21|62.77|610.84/809.50|23.44|
| three_slots: All-host3072 mmapA1 |0|N/A / N/A / N/A / N/A|N/A|N/A|N/A|N/A|
| three_slots: All-host3072 readB1 |15|26.0 / 882.0 / 15.0 / 15.0|243.00/419.14|110.10|700.35/953.65|21.00|
| three_slots: n24/cache2048 mmapA1 |15|26.0 / 882.0 / 15.0 / 15.0|573.27/1089.57|45.35|1350.27/2412.81|10.62|
| three_slots: n24/cache2048 readB1 |15|26.0 / 882.0 / 15.0 / 15.0|234.80/573.81|109.01|583.59/1020.30|23.51|
| three_slots: n24/cache2048 mmapA2 bracket |15|26.0 / 882.0 / 15.0 / 15.0|568.11/871.48|44.01|1296.40/1837.74|11.01|

The n24 read benefit persists against the higher-headroom mmapA2 bracket:hot452.4→164.3ms prefill and450.1→303.5ms decode;pair492.0→185.0/610.8→363.5ms;three-slot568.1→234.8/1296.4→583.6ms. Actual evaluated/cache-hit counts match43/43. This rules out **only** an explanation confined to the first catastrophic low-RAM failure;it does not assign an exact percentage to copies,CPU waiting or paging.

## 4. Full sampled resource ledger and actual host allocation

Bytes and per-phase n/median/p95/min/max are in each original new run,`.analysis.json`,`.samples.jsonl` and [`summary.json`](results/followup/summary.json). Tables use decimalGB;dedicated also givesGiB. Sampled peaks are lower bounds and can occur at different times;do not add overlapping WS/private/GPU counters.100%CPU=one logical core;whole-host fraction divides by16.

| Profile | Complete / intended | Dedicated GB /GiB | Shared GB | WS /private GB | Free before /minimum GB | CPU seconds /one-core% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| All-host3072 mmapA1 |1/43|6.125/5.705|0.605|12.203/7.028|11.468/0.019|9.422/25.69|
| All-host3072 readB1 |43/43|6.155/5.732|13.394|13.628/19.847|11.254/0.007|3.406/8.77|
| n24/cache2048 mmapA1 |43/43|7.629/7.105|0.614|14.438/8.597|15.154/0.640|8.188/13.19|
| n24/cache2048 readB1 |43/43|7.649/7.124|10.949|11.253/18.921|16.118/4.566|2.391/6.99|
| n24/cache2048 mmapA2 bracket |43/43|7.629/7.105|0.614|14.439/8.597|17.220/2.896|7.391/12.54|

| Profile | Whole-device dedicated /shared peak GB | Global commit peak GB /headroom min GB | Pages input/s median /p95 (global) |
| --- | ---: | ---: | ---: |
| All-host3072 mmapA1 |10.302/1.043|62.124/14.760|88805.61/253958.87|
| All-host3072 readB1 |10.232/13.824|72.671/5.735|298.29/651797.18|
| n24/cache2048 mmapA1 |11.447/1.030|65.525/11.358|19.40/29096.05|
| n24/cache2048 readB1 |11.393/11.348|71.062/6.478|49.83/639305.19|
| n24/cache2048 mmapA2 bracket |11.529/1.027|58.616/18.267|0.00/7295.83|

Observed model-host buffers:
- mmap all-host:`CPU_Mapped13570.59MiB`;mmap n24:`CPU_Mapped10935.02MiB`(a mapped span,not a private allocation/residency claim).
- read all-host:**`Vulkan_Host12647.27MiB≈13.261GB`**;read n24:**`Vulkan_Host10197.01MiB≈10.693GB`**. No pinned-allocation-failure warnings were observed. Placement/import/pinning-related lines are preserved as `host_allocation_log`;flag alone is not our evidence.

Read shared-GPU peaks13.394/10.949GB are consistent with intentional GPU-visible host weights of13.261/10.693GB. **Do not call all shared bytes uncontrolled WDDM VRAM spill.** Conversely,host-visible allocation/its name is not a trace proving every miss is asynchronous or every page physically locked. Exact driver allocator/backing flags/per-upload branch trace were not collected. Read n24 observed WS11.253GB rather than mmap14.44GB while privatecommit rose8.60→18.92GB and shared accounting rose;these are different quantities,not interchangeable physical-RAM savings.

Both load modes can impose significant pressure. All-host read completed with a brief7.2MB-free sample,not a guarantee of safe long-running co-rendering;min headroom was5.735GB on the commit metric. The user-approved cutoff required sustained low available memory,which this run did not trigger. n24 read retained4.566GB physical free;the final mmap bracket retained2.896GB. Process page-fault counts include soft faults;global pages-input/read counters cannot attribute hard faults/disk traffic to this process or measure PCIe bandwidth. GlobalMemoryStatusEx-derived commit/headroom is not disk-pagefile residency and may be bounded by process/system commit limits.

### Preserved first safety stop and classification correction

At35.675s the first all-host mmap run had been below256MiB for8.0055s;trigger sample free93,351,936B,minimum18,579,456B,commit headroom17.151GB. OneP912 translation completed correctly;the nextP908 prompt was admitted then interrupted,remaining requests did not complete. No runtime OOM or complete-profile quality pass is claimed.

The watchdog called termination on **its owned server**. A race with `finally` left original `shutdown=graceful_console_ctrl_c` despite graceful-helper exit1/native exit1. [`safety-classification-correction.json`](results/followup/safety-classification-correction.json) withdraws that shutdown interpretation without rewriting raw evidence. Final post-join classification calls the pure `classify_owned_safety_stop` helper. That exact final helper was **exercised offline in a fresh interpreter on copies of all five real retained records**:the observed-cutoff record becomes `owned_memory_safety_cutoff`,four successful records stay unchanged,and all other observed fields are unchanged;exit0. [`safety-classification-check.json`](results/followup/safety-classification-check.json) preserves executed code/results. Later runs did not naturally trigger the cutoff,so we do not claim another live watchdog event or reenact failure to test it. Four subsequent native processes exited gracefully0 and emitted destructor cache counters.

## 5. What the source and measurements establish

[Primary Gemma/Vulkan discussion#21590](https://github.com/ggml-org/llama.cpp/issues/21590#issuecomment-4203766005),[pinned set_tensor_async path](https://github.com/ggml-org/llama.cpp/blob/8e2d31e0eb658d0382d3106a61369dff81f0c080/ggml/src/ggml-vulkan/ggml-vulkan.cpp#L13194-L13255),[recognized-host buffer lookup](https://github.com/ggml-org/llama.cpp/blob/8e2d31e0eb658d0382d3106a61369dff81f0c080/ggml/src/ggml-vulkan/ggml-vulkan-buffers.cpp#L432-L497):

- **Source:**Recognized pinned source copies use that source buffer;unrecognized pageable source falls back to memcpy into reusable synchronous staging and `ggml_vk_synchronize`. A destination being host-visible does not make unordered CPU writes valid. No unsafe BAR-write patch was applied.
- **Measured:**read creates actualVulkan_Host model allocations,no observed pinning fallback,matched-token native speed improvement persists in mmap bracket with better headroom. **Inference:**avoiding pageable staging is a concrete contributor consistent with source;loading/residency/OS behavior also changes. We do not measure exact host-wait or copy-cost percentages.
- **Source:**upstream cache reads selected router IDs then explicitly backend-synchronizes per cache layer,even for hits,plansLRU on CPU,uploads missing experts and slotmap;misses run on GPU. Read mode cannot remove that router barrier. CPU slotmap upload and other synchronization remain;no claim every call becomes nonblocking.
- **Measured cache execution:**read/mmap n24 small-batch hit rates70.38%/70.31%/70.37%(A2),with logical uploaded110857.86/110870.62/111499.14MiB. Gain is not a new~100%hit ratio. Large-batch bucket hit rates13.17%/14.35%/15.34%,uploads66721.46/53350.37/41631.92MiB;bucket/lifetime differences include batching/interleaving,not equal per-phase transfer work. All-host read small/large74.23%/23.63%,uploads122260.46/52996.23MiB. Raw native counters retain existing-cache GPU→GPU copy counts too. Interrupted first run has no destructor statistics.
- **Source:**`get_slot_map` bypassesn_tokens>32,but `copy_experts` can still read already-cached experts for large prefill. Native<=8/>8 counters are lifecycle buckets,not all transfers or phase-only counters. Logical MiB/wall is **not hardware PCIe bandwidth**.

This also explains why the public non-cache CPU-expert report can improvePP withoutTG,while our dynamic cache can improveTG:the ordinary small-decode CPU path does not upload every missed expert,but the cache path does. More CPU workers or a CUDA hybrid mailbox is not an available equivalent switch in this Vulkan experiment.

**Stopping decision:**source+direct load-mode A/B+improved-headroom bracket answer the bounded bottleneck question. We did not run32/33 micro-prefill or forced long generation. Exact slot-map versus batched path share,host-sync percentage,long256-token warm decode and30K-document performance remain unmeasured. P26TPS must not be compared as equal workload to publicPP128/512/31K.

The runtime has source-backed `GGML_VK_PERF_LOGGER`,CONCURRENT,FREQUENCY andSYNC_LOGGER controls. GPU timestamps/op groups and barrier names are partial profiling,not physical PCIe/CPU-wait measurements. [Even CONCURRENT inserts graph submit/fence waits](https://github.com/ggml-org/llama.cpp/blob/8e2d31e0eb658d0382d3106a61369dff81f0c080/ggml/src/ggml-vulkan/ggml-vulkan.cpp#L14657-L14670);SYNC_LOGGER logs GPU dependency barriers,not durations of the cache's host-facing synchronization. No instrumented numbers were mixed into ranking,and no profiling run was needed for this result.

## 6.4096MiB cache: property-only query settles current cap

Installed `<SYSTEM32>/vulkaninfo.exe`1.4.341.0,SHA256 `3b91a8841c15f7894ef0d1255441c54deb2e8cd0ef426821f33022dd89257016`,full property query exited0 at13:10:59UTC. It used its compiled Vulkan headers,not guessed ctypes ABI. RX7900XTX API1.4.349,driver2.0.395/AMD proprietary26.8.1LLPC;vendor1002/device744c.

| Observation | Actual value |
| --- | --- |
| maintenance4 `maxBufferSize` (promoted Vulkan13 property) | **2,147,483,648B=2GiB** |
| maintenance3 `maxMemoryAllocationSize` (promoted Vulkan11) | **2,147,483,648B** |
| `maxStorageBufferRange` |4,294,967,295B,distinct and larger |
| `minStorageBufferOffsetAlignment` |4B |
| KHR_maintenance4 extension |Present,revision2;maintenance3 also present |
| EXT_shader_64bit_indexing / shader64BitIndexing |Extension not advertised;feature value not emitted,N/A rather than fabricated queriedfalse |
| shaderInt64 / bufferDeviceAddress |Supported,true;not a blanket bypass for expert-bank limits |

[`vulkan-capabilities.json`](results/followup/vulkan-capabilities.json) records selected properties,identity,allowlisted controls(all absent),warning and raw-text hash. The loader warned an OBS hook layerAPI1.3 was older than requested1.4;query still succeeded,we changed no layer/settings and infer no timing cause from it. Current absent controls do not reconstruct exited historical processes.

Pinned GGUF/source geometry:fused gate/up2,230,272B per expert plane,down1,115,136B,Q4_0;one identical-layout group perdevice.3072budget has961usable+onepad=962planes,fused2,145,521,664B/down1,072,760,832B(total3,218,282,496B),which fits.4096has1282usable+onepad=1283planes,fused**2,861,438,976B**/down1,430,719,488B:exactly the retained rejected allocation.

The [pinned buffer guard](https://github.com/ggml-org/llama.cpp/blob/8e2d31e0eb658d0382d3106a61369dff81f0c080/ggml/src/ggml-vulkan/ggml-vulkan-buffers.cpp#L28-L66) rejects that bank against `max_buffer_size` beforeVkBuffer creation/allocation. [Initialization](https://github.com/ggml-org/llama.cpp/blob/8e2d31e0eb658d0382d3106a61369dff81f0c080/ggml/src/ggml-vulkan/ggml-vulkan.cpp#L4193-L4210) normally honors advertised maintenance4/fallback allocation limits. **Current stack really advertises2GiB;this is not total24GiB OOM or an arbitrary hardcoded universal AMD/Windows physical cap.** No force override,unbound-buffer allocation probe or4096inference repeat was needed or performed.

No supported CLI splits this fixed same-layout fused bank into smaller banks. Changing ngl,n-cpu count,suballocation block,ubatch,KV/context/SWA/cache-ram or `--no-repack` does not unfuse the stored tensor. Supported working budgets/static placement remain alternatives;multiGPU can split a global budget only on genuine separately authorized hardware. New bank/shader sharding,driver/backend/model change is outside scope. See normative [maxBufferSize](https://docs.vulkan.org/refpages/latest/refpages/source/VkPhysicalDeviceMaintenance4Properties.html).

## 7. Quality, cleanup and reproduction

All173 completed outputs across12distinct sources/16unique input-output pairs were inspected;four full runs retain43/43EOS completions,no observed truncation/reasoning/repetition/material load-mode-specific collapse. This is bounded self-review,not an independent model-quality ranking. Preserve:小林→Kobayashi ambiguous reading(not exact-name pass),13/24 `in5minutes` sources use `within5minutes`,and digit-only false alarms on correct`twice/three`. New`2장만`/`are not`/`is arriving` variants preserve core contrasts. [`quality.json`](results/followup/quality.json) preserves every unique pair and occurrence.

Final release13:45:23UTC:all owned native processes exited,model-runtime enumeration empty,no owned job/observer/sink remains;available16,159,735,808B,whole-device dedicated3,772,809,216B/shared403,521,536B,counter errors empty. Full [`release.json`](results/followup/release.json) retains exit evidence. Windows reusedPID58360 between sequential n24A1/readB1;use run IDs/start timestamps/native exits,notPIDalone. No new app or UDP sink was launched. Artifacts/raw synthetic logs/samples stay external;no unrelated processes were terminated.

Exact model commands,owned ports,PID,controls and per-request inputs/sampling/tokens are in each `firstwave/<run-id>.json`. Reuse verified original models/runtimes,do not substitute:

```text
<APP_PYTHON>/python.exe experiments/gemma4_26b_qat/followup_probes.py --root <ARTIFACT_ROOT> --placement allhost3072 --load-mode mmap --run-id followup-allhost3072-mmap-a1 --allow-pressure
<APP_PYTHON>/python.exe experiments/gemma4_26b_qat/followup_probes.py --root <ARTIFACT_ROOT> --placement allhost3072 --load-mode none --run-id followup-allhost3072-read-b1 --allow-pressure
<APP_PYTHON>/python.exe experiments/gemma4_26b_qat/followup_probes.py --root <ARTIFACT_ROOT> --placement n24cache2048 --load-mode mmap --run-id followup-n24cache2048-mmap-a1 --allow-pressure
<APP_PYTHON>/python.exe experiments/gemma4_26b_qat/followup_probes.py --root <ARTIFACT_ROOT> --placement n24cache2048 --load-mode none --run-id followup-n24cache2048-read-b1 --allow-pressure
<APP_PYTHON>/python.exe experiments/gemma4_26b_qat/followup_probes.py --root <ARTIFACT_ROOT> --placement n24cache2048 --load-mode mmap --run-id followup-n24cache2048-mmap-a2-bracket --allow-pressure
<APP_PYTHON>/python.exe experiments/gemma4_26b_qat/summarize_timings.py --results experiments/gemma4_26b_qat/results/followup/firstwave --raw-root <ARTIFACT_ROOT>/followup --markdown experiments/gemma4_26b_qat/FOLLOWUP_TIMINGS.md
```

These commands are the recorded sequence,**not instructions to overwrite/reconfirm failures**;runner refuses existing identities. The first command intentionally returned failure after recorded safety cutoff,so it was not chained with`&&`to skip the read comparison. Preserve new identifiers for any separately justified later work. Raw samples remain external and compactJSONL is public;resource summaries group those same samples by their recorded phase.

No product source/architecture,live settings/secrets/providers/audio/output,Git/GitHub mutation,product build/test/linter/formatter or runtime update was performed. Director owns final static/JSON/immutability checks,independent review and authorized experimental publication. No further GPU benchmarking is needed to interpret this bounded result;none-mode actual-app adoption remains a separately requested question,not silently added scope.
