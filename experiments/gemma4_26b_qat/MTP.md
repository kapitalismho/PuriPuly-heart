# Issue 254 follow-up — Q4_0 MTP with unchanged and reduced expert cache

## Outcome and authority

**MTP genuinely ran on the pinned Vulkan runtime, but did not improve the requested short Self/Peer workload.** Both requested max-draft-4 arms completed two fresh counterbalanced lifecycles. At cache2890, MTP worsened pooled simultaneous-pair median/p95 by **133.897 / 293.739 ms** versus matched MTP-off; at cache2700 the penalties were **99.784 / 312.963 ms**. Every max4 MTP run median was above every corresponding same-cache off run median. The bounded max2 exploration improved on max4, but its retained first run was not competitive with the fresh off controls.

**Both cache2890/MTP4 and cache2700/MTP4 measured within the 8-GiB PID-dedicated target.** The unchanged-cache result corrects the initial file-size-based estimate that cache reduction would be necessary. It does **not** mean MTP weights have zero cost or that physical MTP memory cost is only about4MiB: real native drafter model/compute/host buffers exist, and working set, private commitment, shared GPU and GPU-commit views increase separately. Individual allocation backing/residency is not established by the Windows counter deltas.

Completed **9 native lifecycles / 396 EOS translations / 5 configurations**, all graceful exit0, no watchdog cutoff/native failure/observed competing named model runtime, and **135/135 simultaneous rounds with native active-slot overlap**. Independent bounded semantic review and final original-JSON/ledger coverage audit cover ten exact input/output pairs across six sources. This is a separate owner-authorized MTP cohort; the [previous pinned evidence](https://github.com/kapitalismho/PuriPuly-heart/commit/c451f1b04a940f8d3e297ca58d13eefc338baee8) is unchanged and is not pooled into current comparisons.

Experimental recommendation: **do not enable these tested MTP profiles for this workload; retain the prior experimental n24/cache2890/MTP-off recommendation**. That is not owner adoption: #245’s owner-selected n24/cache2048 profile remains authoritative pending the owner’s separate #254 disposition. No product/settings/maintained-docs/runtime/backend/driver/target-model change, PR/merge or rollout was performed by the worker.

Sources: [owner-authorized scope comment](https://github.com/kapitalismho/PuriPuly-heart/issues/254#issuecomment-6205074308), [pinned model MTP README](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-qat-GGUF/raw/7b92b5b28818151e8669af2e45e88d6086f490dd/MTP/README.md), [source-backed runtime/timing/allocation research](results/mtp/runtime-research.json), [evidence index](results/mtp/evidence-index.json), [all native timing tables](MTP_TIMINGS.md), [machine-readable analysis](results/mtp/summary.json). Director owns stable-candidate review, pinning and publication; owner retains disposition.

## Fixed prerequisites and unchanged workload

- Execution baseline: `c451f1b04a940f8d3e297ca58d13eefc338baee8`, with experimental helper extensions only. Existing `TUNING8GIB.md`, timing tables and all prior result files were not modified.
- Target remains `gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf`: **14,249,047,104 bytes**, SHA256 `a7c5bc715f5ff8e99a3e8901ce7d2b42b402c669bf24f7c5250747633d0f5891`.
- Pinned drafter: `unsloth/gemma-4-26B-A4B-it-qat-GGUF`, revision `7b92b5b28818151e8669af2e45e88d6086f490dd`, `MTP/mtp-gemma-4-26B-A4B-it-Q4_0.gguf`: **251,939,328 bytes = 240.268066 MiB**, SHA256 `7272d97595f0d4c74bd7b623492b7dbdaafd8b7c72f329a8270ba4eca68f768a`. This is the same published Q4_0 object as the root `mtp-gemma-4-26B-A4B-it.gguf`; the explicit MTP/Q4_0 path was acquired. The downloaded file and complete target file were freshly stream-hashed before any model launch; drafter and executable identities were checked again per run. No quantization/target substitution.
- Runtime remains llama.cpp **b11537 / `8e2d31e0eb658d0382d3106a61369dff81f0c080`**, executable SHA256 `7b36a3bc3a24e552d0fcc18441893bc198abaeb0fa77d653241d762077c995c6`; fresh `--version` output and identities are in [prerequisites.json](results/mtp/prerequisites.json). No runtime upgrade or upstream changes.
- Same Windows11 / Ryzen7 9800X3D 8C16T /32-GiB-class RAM / RX7900XTX 24-GiB-class Vulkan host. Exact adapter is `luid_0x00000000_0x0000f60b_phys_0`. Existing constrained RAM and background applications remained accepted and untouched.
- All target controls fixed: n24, `--n-gpu-layers99 --load-mode none --swa-full --cache-ram0`, context4608/parallel3/1536 per slot, threads4/4, batch/ubatch512, f16KV, flash-attn on, fit off, reasoning off/budget0, same native warmup/perf/verbosity/local HTTP settings. Only cache and MTP/max-draft change. Exact token-separated native command arrays, rather than compact prose, are authoritative.
- Same44-request fixture: Self Korean→English slot0, Peer English→Korean slot1, slot2 unused;2 first-prefix warmups at temperature0,6 serial pair rounds/12requests,15 barrier-released simultaneous pair rounds/30requests; original prompt/history/CASES, temperature.6, seed42, max96. No app-level concurrency implementation.
- [Fixture inheritance check](results/mtp/workload-inheritance.json): fresh2890/off matches all44 prior pinned request/prompt/sampling identities and actual prompt-evaluated/cache-hit counts. All nine new runs also match all44 current identities, and **P/C counts match at every corresponding request**. Eighteen recorded corresponding-request count differences are all generated-output-token differences and remain disclosed, not forced equal.

### Four-arm plan and bounded refinement

Primary order was **A1→B1→C1→D1→D2→C2→B2→A2**: A=2890/off, B=2890/MTP4, C=2700/MTP4, D=2700/off. Every lifecycle used a fresh server and only one owned model server was loaded at a time. [Exact plan](results/mtp/max4-plan.json). Both requested on arms and both matched off controls have two runs.

Cache2700 started from the owner-discussed estimate. Once unchanged2890/MTP already measured target-class, the Director explicitly kept2700 as the distinctly reduced arm rather than pointlessly back-filling it toward2890. Both2700/MTP repeats measured target-class, so no cache adjustment was needed. This is not a new largest-cache frontier search.

After both max4 arms were negative, the Director authorized a single max2 exploration at unchanged2890 first. The retained E1 median815.041ms exceeds both2890/off medians745.365/768.326ms; the nearest gap46.715ms exceeds off’s22.961ms observed run-median spread. It therefore cannot satisfy every-candidate-run-below-every-control by adding favorable repeats. **No E1 was discarded, no rescue repeat, no reduced-max2 arm and no open sweep**. [Bounded max2 decision](results/mtp/max2-decision.json). This n1 point is exploratory, not repeated proof or universal max2 inferiority.

## Matched pair, serial and memory results

Pair latency is common barrier release→both complete, including apply-template/local HTTP. Serial makespan is start before first routine→second complete. Pooled percentiles use original pair observations, not run percentiles. Eligibility/classification uses the **largest whole-lifecycle sampled server-PID dedicated peak across every run of that configuration**, including E1. Target≤8,589,934,592B; tolerance ceiling8,697,308,774B. All observed configurations are target-class; max2 is still n1, not a selection finalist.

| Configuration | Runs | Every run pair median ms | Pooled pair median / p95 ms | Serial median / p95 ms | PID dedicated bytes / GiB / GB |
| --- | ---: | --- | ---: | ---: | --- |
|c2890-off|2|745.365, 768.326|761.733 /953.179|1009.348 /1133.097|8,525,266,944 /7.939774 /8.525267|
|c2890-mtp4|2|879.382, 898.992|895.630 /1246.918|1046.170 /1344.799|8,529,027,072 /7.943275 /8.529027|
|c2700-off|2|818.780, 793.799|818.100 /970.732|1048.991 /1163.576|8,327,872,512 /7.755936 /8.327873|
|c2700-mtp4|2|901.716, 931.157|917.884 /1283.695|1073.033 /1344.194|8,331,100,160 /7.758942 /8.331100|
|c2890-mtp2|1|815.041|815.041 /1075.524|958.576 /1148.987|8,529,551,360 /7.943764 /8.529551|

### What changed, and what did not

- **Unchanged cache MTP4 is repeatedly slower in this cohort:**879.382/898.992ms versus745.365/768.326ms off. Pooled median increases133.897ms(17.58%), p95 increases293.739ms; serial median/p95 also worsen36.822/211.702ms. This is a matched bounded workload result, not “MTP never helps.”
- **Reduced cache MTP4 is also repeatedly slower:**901.716/931.157ms versus793.799/818.780ms off. Pooled median/p95 worsen99.784/312.963ms; serial median/p95 worsen24.042/180.618ms.
- **Cache reduction has a separately measured cost:**2700/off pooled pair median is56.367ms slower than2890/off while saving197,394,432 dedicated bytes(188.25MiB). Both2890/off run medians are below both2700/off medians. It is not correct to attribute all reduced-arm latency to MTP.
- **Max2 made legitimate progress but did not win:** acceptance rose to66.90% and its815.041-ms pair median improved versus both max4 run medians. Its observed serial median was50.772ms below pooled2890/off, but serialp95 was15.890ms worse and primary pair median/p95 were53.308/122.345ms worse. All max2 statements are descriptive n1; no repeated or general superiority/inferiority claim.

### Margins and observed run variation

| Configuration | Target margin bytes / MiB | Ceiling margin bytes / MiB | Peak variation bytes / MiB | Boundary qualification |
| --- | ---: | ---: | ---: | --- |
|c2890-off|64,667,648 /61.671875|172,041,830 /164.071875|0 /0.000000|Neither boundary marginal by observed same-config range|
|c2890-mtp4|60,907,520 /58.085938|168,281,702 /160.485937|0 /0.000000|Neither boundary marginal by observed same-config range|
|c2700-off|262,062,080 /249.921875|369,436,262 /352.321875|0 /0.000000|Neither boundary marginal by observed same-config range|
|c2700-mtp4|258,834,432 /246.843750|366,208,614 /349.243750|0 /0.000000|Neither boundary marginal by observed same-config range|
|c2890-mtp2|60,383,232 /57.585938|167,757,414 /159.985937|0 /0.000000|n1: variability unmeasured|

GiB=2^30B,MiB=2^20B,GB=10^9B. Peaks are250-ms sampled lower bounds, not exact lifetime high-water marks. A positive margin below the same configuration’s observed peak range would be marginal. Max4 n2 happened to show zero dedicated-peak range; that is not zero uncertainty. None of this establishes physical8-GiB-card fit or free desktop/VRChat capacity.

## Native MTP execution, acceptance and timing semantics

The pinned README’s exact documented controls were used: `--model-draft <pinned Q4_0 file> --spec-type draft-mtp --spec-draft-n-max4`, with max2 only for E1. Default draft-GPU auto placement followed the target; actual logs prove **5/5 drafter layers offloaded to Vulkan0**, `gemma4-assistant`, model buffer225.21MiB and speculative implementation `draft-mtp` initialized with the requested n_max. All three server slots report speculative context enabled when on; slot2 remains unused. Nonzero per-request proposals/accepted prefixes prove actual execution, not merely initialized intent. Native target-KV sharing lines and API/native count reconciliation are retained.

Each off lifecycle was a separate fresh profile with no draft path, draft-HF path or earlier enabled spec type, clean LLAMA_* environment, all `/slots` speculative:false and no draft counters/model initialization. The collected off commands contained one redundant explicit `--spec-type none`, never appended after an enabled type. Research correctly warns that appending none after draft-mtp would not override it and a retained draft path may auto-infer/load MTP; that situation did not occur. The final helper omits even the redundant off spec flag. Original observed commands remain unchanged.

**Use native API server-slot statistics, not libllama destructor TPS.** In this pinned server, `/completion.timings.predicted_per_second = 1000×(predicted_n−1)/predicted_ms`. `predicted_n` counts generated results including the first token/EOS; elapsed includes draft generation, target verification, synchronization and slot scheduling. The formula was checked against every original request. No speculative proposal count is substituted for delivered/generated-output throughput; no corrective surrogate TPS is necessary. P=`timings.prompt_n` is newly evaluated prompt work; C=`timings.cache_n` is prior KV reuse. Top-level `tokens_evaluated` is full prompt size and `tokens_cached` is final retained KV length, **not P/C**.

`draft_n` counts proposals; `draft_n_accepted` counts accepted speculative prefixes, excludes bonus target tokens and can include an accepted suffix past EOS. Its ratio is acceptance, **not delivered-output throughput**. Common native `#gen tokens/#acc tokens` traces are cumulative snapshots: the analyzer compares only each run’s last snapshot to sums of per-request API counters, never sums snapshots. libllama destructor evaluation rates can categorize multi-token verification as prompt work and are not used for output-TPS comparison. The source report contains pinned serializer/counter/allocator citations.

| Configuration | API rows with draft counters | Proposed / accepted prefixes | Weighted acceptance% | Actual P / C / O totals |
| --- | ---: | ---: | ---: | ---: |
|c2890-off|0|unemitted/off|N/A|5,734 /74,276 /1,391|
|c2890-mtp4|88|1,668 /919|55.096|5,734 /74,276 /1,393|
|c2700-off|0|unemitted/off|N/A|5,734 /74,276 /1,390|
|c2700-mtp4|88|1,676 /918|54.773|5,734 /74,276 /1,393|
|c2890-mtp2|44|568 /380|66.901|2,867 /37,138 /694|

### Simultaneous per-request timings by channel

Independent medians use actual tokens and native API rates; they are not aggregate concurrent throughput or ratios of pooled medians. All first-prefix/serial/simultaneous channels, n/median/p95/min/max and counts are in MTP_TIMINGS.md/summary.json.

| Configuration / channel | Prefill ms / TPS median | Decode ms / TPS median | P/C/O median | HTTP median / p95 ms | TTFT median / p95 ms | Draft acceptance% |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
|c2890-off/Self|189.298 /144.622|463.095 /33.373|26.0/882.0/16.0|721.600 /828.176|205.651 /381.259|N/A|
|c2890-off/Peer|189.247 /136.321|522.576 /34.173|26.0/884.0/15.0|717.068 /953.081|213.923 /398.323|N/A|
|c2890-mtp4/Self|188.878 /139.829|471.721 /28.677|26.0/882.0/16.0|788.800 /893.284|208.706 /464.684|73.864|
|c2890-mtp4/Peer|190.903 /134.953|560.692 /25.329|26.0/884.0/15.0|892.814 /1246.824|215.156 /450.874|43.642|
|c2700-off/Self|193.618 /143.125|483.600 /30.936|26.0/882.0/16.0|746.670 /837.837|206.636 /396.066|N/A|
|c2700-off/Peer|194.075 /133.664|531.491 /32.880|26.0/884.0/15.0|763.152 /970.429|232.347 /391.682|N/A|
|c2700-mtp4/Self|191.662 /137.515|482.372 /28.159|26.0/882.0/16.0|807.874 /929.675|206.495 /467.544|73.636|
|c2700-mtp4/Peer|193.630 /136.231|560.134 /25.131|26.0/884.0/15.0|917.701 /1283.410|220.211 /507.369|43.143|
|c2890-mtp2/Self|189.341 /147.291|439.722 /34.556|26/882/16|752.356 /817.682|236.718 /426.184|84.375|
|c2890-mtp2/Peer|183.124 /138.658|528.349 /28.826|26/884/15|814.776 /1075.248|207.205 /440.549|54.425|

## Memory views and the corrected file-size estimate

The initial discussion estimated that a240.27-MiB file might require reducing cache2890 to about2700 to stay near8GiB. That was a **planning proxy, not a measured result**. Measured maxima instead put unchanged2890/MTP4 at **8,529,027,072B = 7.943275GiB**, below target by60,907,520B(58.085938MiB). Both requested variants were still measured; no back-fill or hidden change converted the reduced arm into the unchanged arm.

### Separate lifecycle accounting views — do not add

Each entry is its own all-run maximum, potentially at a different timestamp. WS=host working set; private=private commitment, not resident physical RAM. Process GPU committed/shared/dedicated are distinct accounting views; none is summed here. Whole-device usage includes unrelated background work and is not the eligibility metric. Exact bytes and per-phase distributions are retained in summary.json and memory-audit.json.

| Configuration | Host WS GiB | Private commitment GiB | PID shared GPU GiB | PID GPU committed GiB | Whole-device dedicated GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
|c2890-off|10.496685|18.447861|10.189205|18.033730|12.026829|
|c2890-mtp4|11.134090|19.032227|10.556519|18.498322|12.122028|
|c2700-off|10.497265|18.261845|10.179440|17.849892|11.785526|
|c2700-mtp4|11.092682|18.813942|10.556519|18.313988|11.799511|
|c2890-mtp2|11.102207|19.004314|10.552612|18.495392|11.996532|

Native logs prove real additional buffers: **Vulkan0 drafter model225.21MiB**, **Vulkan_Host model144.00MiB**, and **Vulkan0 drafter compute62.12MiB**, plus output/sampler/hidden-state and verification bookkeeping. Shared target-KV aliases avoid another independent copy of the referenced target KV tensors. Logical drafter KV size lines18+108MiB count references/aliases and are **not** fresh126MiB allocation proof. The source-backed duplicated input/output embedding path explains a real host component; it is not a fabricated host-only replacement for the GPU drafter.

### Own PID/adapter and phase audit

[Memory audit](results/mtp/memory-audit.json) independently projects original per-run samples/logs: all nine runs have no counter-add errors, exact owned PID and single exact LUID instance for process dedicated/shared/committed, load and every workload-phase coverage. A1/B1 examples use PID51512/61988 respectively. Native target weights/cache/KV/compute controls remain unchanged. Dedicated differences persist within matched active phases, so the small lifecycle delta is **not explained merely by substituting a steady snapshot or comparing an off startup peak to an on active snapshot**.

Below, each cell is MTP4 all-run phase maximum minus same-cache2890/off all-run phase maximum. They are separate measurements, not a time-coincident allocation equation.

| Phase | PID dedicated delta MiB | PID shared delta MiB | PID GPU committed delta MiB |
| --- | ---: | ---: | ---: |
|load|1.824219|350.859375|461.363281|
|first_prefix|3.585938|467.164062|470.750000|
|serial_pairs|3.585938|467.164062|469.738281|
|simultaneous_pairs|3.578125|472.164062|475.742188|
|lifecycle|3.585938|376.128906|475.742188|

The native logical GPU model225.21MiB and the lifecycle PID-dedicated delta3.585938MiB therefore do **not** reconcile per allocation from these observations. Higher shared/committed/WS/private views separately show nonzero resource costs, but adding them or assigning all their bytes to one buffer would be incorrect. The pinned allocator tries `DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT` then `DEVICE_LOCAL`; **HOST_VISIBLE does not mean system RAM**. No per-buffer physical residency/backing trace was taken. Do not infer zero-cost weights, missing drafter, CPU fallback, duplicate physical RAM or uncontrolled WDDM spill. Source research retains its load-step observations and limits. Extended driver tracing was not necessary or authorized to answer the paired latency question.

### Sampling, pressure and safety

Same exact-adapter PID-specific Windows GPU counter method as the prior experiment, over load/readiness/prefix/serial/simultaneous/shutdown, nominal250ms. There are **1,145 samples**; per-run cadence medians248.573–248.873ms, largest observed interval265.842ms. Additional final post-exit-handle samples capture lifecycle CPU. Sampled maxima remain lower bounds. Shutdown snapshots are not synchronized allocator-progress fences and are not interpreted as per-buffer cost.

Fresh preflight found no model-runtime conflict before each launch. During lifecycles, **1,136 name/PID-only runtime observations** found no other known native model process. Differently named/Python-hosted or wholly between-sample models remain an explicit detection limit. No unrelated command lines were captured for this observation and no unrelated process was stopped.

Existing constrained RAM was accepted. The owner-only watchdog policy was to stop only the owned server after available physical RAM was below256MiB **or** commit headroom below512MiB continuously for8s, classifying that as a safety cutoff, not OOM. It never fired. Minimum observed physical availability was2,679,996,416B; minimum commit headroom4,531,937,280B. Background conditions/paging/file/shader caches vary; global paging counters are not attributed exclusively to this process.

## Cache allocation/use and actual internal work

Native target expert cache remains a single shared n24 group of3,072 host experts,8 selected/token, with whole expert slots. Requested2890 gives904 usable slots/effective2887.34MiB;2700 gives845/effective2699.10MiB. The reduction is59 slots, not a literal190MiB allocation decrement. MTP drafter initialization disables a duplicate MoE cache; it does not create another target expert cache.

Weighted lifecycle cache statistics below include native warmup and all original requests. Logical upload/copy MiB is not measured physical PCIe traffic. MTP changes internal verification batch mix and deduplicated expert plans, so bucket hit rates/counts are not an identical-operation denominator and are not used as a causal latency decomposition.

| Configuration | Request / effective MiB / slots | Small hits / misses / hit% | >8 hits / misses / hit% | Small / >8 upload MiB | Large tensor-bank copies / MiB |
| --- | --- | --- | --- | ---: | ---: |
|c2890-off|2890 /2887.34 /904|193,355 /44,860 /81.168|17,349 /46,352 /27.235|143122.67 /147882.79|26,332 /42005.20|
|c2890-mtp4|2890 /2887.34 /904|78,545 /25,705 /75.343|65,615 /73,062 /47.315|82009.99 /233099.17|29,428 /46943.98|
|c2700-off|2700 /2699.10 /845|187,765 /49,610 /79.101|17,495 /54,543 /24.286|158277.21 /174015.61|19,704 /31432.11|
|c2700-mtp4|2700 /2699.10 /845|75,812 /28,416 /72.737|58,545 /78,794 /42.628|90659.25 /251386.72|29,718 /47406.60|
|c2890-mtp2|2890 /2887.34 /904|68,163 /26,216 /72.223|11,270 /25,781 /30.418|83640.30 /82252.47|9,736 /15531.01|

All runs retain native allocation/group lines, both cache-destructor buckets and large-copy evidence. There are no cache-off or initialization-failure rows hidden as small-cache results. No known Vulkan bank-cap failure was rerun or worked around.

## Every lifecycle: startup, CPU, pressure and release

Actual chronological order below. CPU100%=one logical core; divide by16 for whole-host normalization. CPU covers the sampled full server lifecycle, not a measured per-request CPU attribution. Launch→first completion includes load, readiness, no-touch observations/prefix metadata and warmup request. Each startup is one lifecycle observation, not an independently controlled cold-disk/shader-cache distribution.

| Run | Cache / draft max | Ready s | Launch→first s | CPU s / one-core% | Min physical / commit headroom GiB | Samples |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
|mtp-n24-c2890-off-a1|2890 /0|8.325|9.908|2.250000 /7.511|2.556282 /4.573151|121|
|mtp-n24-c2890-d4-b1|2890 /4|8.693|10.399|2.578125 /7.786|2.495941 /4.729206|134|
|mtp-n24-c2700-d4-c1|2700 /4|8.466|10.115|3.109375 /9.167|3.230999 /4.732277|137|
|mtp-n24-c2700-off-d1|2700 /0|7.504|9.111|3.718750 /12.430|3.153057 /4.220695|121|
|mtp-n24-c2700-off-d2|2700 /0|7.224|8.774|2.593750 /8.819|3.773354 /4.659451|119|
|mtp-n24-c2700-d4-c2|2700 /4|8.279|9.964|2.875000 /8.617|3.430683 /4.627434|135|
|mtp-n24-c2890-d4-b2|2890 /4|8.323|9.984|3.078125 /9.210|3.528790 /4.649292|135|
|mtp-n24-c2890-off-a2|2890 /0|7.646|9.225|2.281250 /7.766|4.122635 /4.354774|119|
|mtp-n24-c2890-d2-e1|2890 /2|8.071|9.636|2.421875 /7.886|3.932621 /4.363968|124|

No safety stop, native failure, cache-off result or later observed runtime conflict occurred. All nine original run identities remain retained, including negative E1. Failed/unsupported paths would have remained explicit; none is invented as a successful fallback.

## Bounded quality and output/count differences

**396/396 original completed requests are EOS, ten exact pairs/six synthetic sources, zero missing/duplicate/unreviewed occurrences.** The independent final reviewer read all nine original per-run JSON files and matched the full `(run, phase, channel, round, input, output, stop)` multisets to the unique-pair ledger; checked required predicates on every original request; and found no anomalies. [Final independent confirmation](results/mtp/quality-confirmation.json), [unique strings/occurrences](results/mtp/quality-unique.json), [new tenth-pair direct judgment](results/mtp/quality-review-eight.json). The original nine string judgments are reused only for identical source **and** output strings, not their earlier occurrence/timing counts. The final audit scope is original-JSON/ledger plus direct semantic review, **not raw native EOS reproduction or full-model quality certification**.

The tenth string, ID`50c90b165fa41d4a`, is `미나는 티켓을 12장 산 게 아니라, 2장 샀어요.` for “Mina did not buy12tickets; she bought2.” It occurs twice in the max4 cohort and was independently reviewed: the topic marker는 versus가 is a mild discourse distinction, not loss of the name/quantities/corrective negation. It is a distinct reviewed string, not silently merged with another.

Names/numbers/negations and corrective contrasts remain preserved in this bounded fixture; no observed material configuration-specific collapse, truncation, pathological repetition or reasoning/instruction leakage. Real inherited caveats remain: **29 within-five-minutes** outputs versus**34 five-minutes-later**; **33 only-two emphasis** outputs versus**39 other Mina variants**, including the new topic form. Implicit-subject explicitation and minor punctuation caveats are not erased. This is not an unconditional exact-fidelity pass.

Every input/prompt/sampling identity and actual P/C count matches the fresh control. All **18 O-count differences** remain in `prompt_and_output_token_differences`; minor wording/topic/emphasis changes are not forced into output equality under speculative sampling. Draft acceptance is not used as a quality proxy.

## Exercised checks and reproduction

The final fresh-interpreter commands below both exited0 on the frozen9-run/396-output set:

```text
<APP_PYTHON> experiments/gemma4_26b_qat/summarize_mtp.py --check
<APP_PYTHON> experiments/gemma4_26b_qat/mtp_memory_audit.py
```

[Focused check](results/mtp/focused-check.json) and per-run booleans exercised44 requests and2/12/30phase counts; exact original identities/sampling; EOS/bounded surface predicates; fresh graceful0/no cutoff; exact PID/adapter load+phase memory coverage; absent backend controls;15 native overlaps/run and unused slot2; actual cache execution; MTP proposals/accepted prefixes; requested n_max; real drafter Vulkan allocation/sharedKV; final cumulative native counters matching summed API counters; native API decode-rate formula. Independent semantic/final occurrence audit is separate. Acquisition performed fresh full target/drafter hashes and runtime hash/version checks before model launch. No product build/test/linter/formatter/type-check, app smoke, GUI/audio or physical-card/coexistence check was run.

The existing request routine gained only an optional final-response metadata capture switch; request body/workload are retained. Separate new MTP helpers and destinations avoid modifying the prior reports/results. Exact per-run **native `command` and probe `probe_command` token arrays**, complete native final-response metadata, sanitized logs, slot-event line ledgers and compact250ms samples are indexed in [evidence-index.json](results/mtp/evidence-index.json). `<ARTIFACT_ROOT>`, `<CHECKOUT>`, `<MODEL>`, `<APP_PYTHON>` replace private paths; the verified existing Python3.14.7 environment haspsutil/httpx. Raw path-bearing records remain external under `<ARTIFACT_ROOT>/mtp/runs/<run-id>/`; model/drafter binaries remain external. Native-log text hashes refer to retained original decoded log text, not byte-identical redacted public copies.

Acquisition and per-run CLI forms actually used:

```text
<APP_PYTHON> experiments/gemma4_26b_qat/mtp_acquire.py --root <ARTIFACT_ROOT> --adapter luid_0x00000000_0x0000f60b_phys_0
<APP_PYTHON> experiments/gemma4_26b_qat/mtp_probes.py --root <ARTIFACT_ROOT> --adapter luid_0x00000000_0x0000f60b_phys_0 --cache-mib 2890 --draft-max 0 --group control --run-id mtp-n24-c2890-off-a1 --allow-pressure
<APP_PYTHON> experiments/gemma4_26b_qat/mtp_probes.py --root <ARTIFACT_ROOT> --adapter luid_0x00000000_0x0000f60b_phys_0 --cache-mib 2890 --draft-max 4 --group unchanged --run-id mtp-n24-c2890-d4-b1 --allow-pressure
<APP_PYTHON> experiments/gemma4_26b_qat/mtp_probes.py --root <ARTIFACT_ROOT> --adapter luid_0x00000000_0x0000f60b_phys_0 --cache-mib 2700 --draft-max 4 --group reduced --run-id mtp-n24-c2700-d4-c1 --allow-pressure
<APP_PYTHON> experiments/gemma4_26b_qat/mtp_probes.py --root <ARTIFACT_ROOT> --adapter luid_0x00000000_0x0000f60b_phys_0 --cache-mib 2890 --draft-max 2 --group max2 --run-id mtp-n24-c2890-d2-e1 --allow-pressure
```

The exact remaining off/reverse commands are in max4-plan.json and each run record. Reproduction must use new run IDs/evidence destinations: overwrite protection preserves earlier measurements. The final helper’s redundant-off-flag omission is documented above; original observed native arrays remain authoritative.

[Release](results/mtp/release.json): **2026-10-16T00:33:11.667950+00:00**, named model-runtime list empty; all nine owned servers exited0/gracefully, sampler threads and Windows handles closed. No owned app/observer/UDP/provider service remains; unrelated processes were untouched. Raw evidence and downloaded pinned drafter intentionally remain external for reproduction.

## Recommendation and limits

**Do not enable the tested Q4_0 MTP profiles for this specific fixed short Self/Peer workload.** Both owner-requested max4 cache variants work natively on Vulkan and preserve the bounded quality set, but repeatedly worsen simultaneous-pair latency and tails versus their matched off controls. Reducing cache saves about188MiB of this PID-dedicated counter but adds an independently observed latency cost and is not necessary merely to satisfy the measured PID target on this host. Max2 is a retained negative exploratory refinement, not a repeated finding.

Retain the prior **experimental** n24/cache2890/load-mode-none/MTP-off recommendation; the current follow-up does not supersede the earlier placement/cache evidence or constitute owner adoption. Official adopt-refined versus keep-n24/2048 disposition is still the project owner’s decision. [Machine-readable recommendation](results/mtp/recommendation.json).

The measurements use a24-GiB-class7900XTX and sampled Windows PID counters. They do not prove physical8-GiB-card fit, individual buffer residency, desktop/VRChat coexistence, another vendor’s behavior, long-generation MTP performance, managed26B/app-level Self/Peer integration, GUI or audio E2E. **No product/settings adoption or rollout is performed by this experiment.**
