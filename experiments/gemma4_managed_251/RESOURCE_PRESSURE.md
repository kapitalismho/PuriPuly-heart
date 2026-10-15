# Memory pressure and accounting

The original sections below analyze preserved candidate8786 runs; the separate671 after-proof section records the new source-native observations. Neither is a RAM-safety verdict. Reproducible analyzer: `pressure.py`; original timestamps, phase/PID samples and threshold runs: [memory-pressure.json](results/memory-pressure.json). Recorded managed-spawn-to-owned-exit windows are used for attribution; samples emitted by a monitor after that PID exited are excluded.

## Observed low-memory intervals

Physical available memory is a **system-wide** counter. MiB means1,048,576bytes. A consecutive-low sample span is last-low timestamp minus first-low timestamp, with all intervening recorded observations low. It is not a claim that memory remained continuously below the threshold between observations. Surrounding above-threshold samples provide the larger outer bracket within which a continuous low interval could have occurred.

| Native process/run | Physical minimum | Phase at minimum | Longest sampled span below256MiB / outer bracket | Longest sampled span below512MiB / outer bracket |
| --- | ---: | --- | ---: | ---: |
|26B Vulkan,PID58212|12,550,144B(11.969MiB)|`model-load`|0.519124s /1.038275s|3.114608s /3.632947s|
|E4B CPU,PID65036|333,004,800B(317.578MiB)|`route-drain` after active requests|No low observation|5.714365s /6.232777s|
|Injected E4B fallback's actual CPU,PID59756|262,643,712B(250.477MiB)|`opposite:overlap-0`|One low observation:0s sampled span /0.518893s bracket|1.296874s /1.817328s|

-26B longest below256 observations: monotonic95412.2773474–95412.7964714; above bounds95412.0178350–95413.0561102. Longest below512:95410.7204586–95413.8350663, above bounds95410.4618584–95414.0948057. Both are **loading**, before initial native prefix generation. Six samples total were below256 and17 below512; the JSON retains shorter runs as well.
-CPU longest below512:96130.3902032–96136.1045686, above bounds96130.1314472–96136.3642246. It crosses `opposite:overlap-4`, `opposite:overlap-5` and `route-drain`, not loading or shutdown. No physical-below256 sample occurred.
-Fallback below256: single sample96200.9804868, above bounds96200.7214181–96201.2403114. Longest below512:96199.6836131–96200.9804868, above bounds96199.4229836–96201.2403114, across `opposite:serial-1` and `opposite:overlap-0`.

Median sample gaps were approximately0.259–0.260s; maxima were0.305983s for26B,0.312836s for CPU and0.315035s for fallback. There is no sparse eight-second hole in these recorded live-PID windows. The maximum possible below-threshold windows bracketed by recorded above observations are the outer brackets in the table (CPU below256 maximum gap0.312836s). Exact crossing times and instantaneous rebounds remain unknown; discrete sampling does not establish exact continuous durations.

The26B scenario subsequently launched E4B CPU PID63348 and E4B Vulkan PID61372 for actual transitions. Their physical minima were6,739,107,840B during CPU-to-GPU transition and8,565,489,664B during shutdown respectively; neither had observations below either physical threshold. ABBA A1/B1/B2/A2 also had no observations below either threshold. The analyzer retains loading, first-prefix, active, transition and shutdown phases separately instead of attributing all minima to inference.

## Watchdog fact versus safety claim

The configured watchdog is **physical available<256MiB OR commit headroom<512MiB for8s**, stopping only owned work. Physical512MiB is an additional analysis threshold, not its physical cutoff. No watchdog-cutoff event occurred. In the three principal live windows, longest sampled watchdog-predicate spans were0.519124s, none, and a single low observation respectively. The above-observation brackets do not allow an unseen eight-second continuous violation inside those recorded windows.

This explains the absence of a cutoff; it does **not** make an11.969MiB physical minimum acceptable, establish a memory SLA or prove safety under VRChat/background load. No OOM, uncontrolled spill or physical residency conclusion is inferred from accounting alone.

## Baseline limitation

A true **prelaunch** available-memory/commit baseline was **not captured**: the original monitor starts after native spawn. It cannot be reconstructed from later counters. Earliest postspawn observations are explicitly proxies, not renamed prelaunch measurements:

| Run | Delay from recorded spawn | Available physical | Commit headroom | Committed / reported limit |
| --- | ---: | ---: | ---: | ---: |
|26B|0.256058s|9,865,506,816B|16,924,934,144B|59,958,153,216 /76,883,087,360B|
|CPU|0.183678s|10,474,201,088B|14,106,136,576B|62,776,950,784 /76,883,087,360B|
|Fallback CPU|0.258516s|10,487,324,672B|15,587,274,752B|64,053,460,992 /79,640,735,744B|

Loading had already begun. The reported system commit limit was not constant across runs. Background memory and paging were not controlled by terminating unrelated applications. No new model run was made merely to fill this missing original baseline.

## Process versus device and global paging

Decimal GB below; sampled highwaters are lower bounds. Private commitment is not physical RAM. Process GPU shared accounting does not by itself prove spill. Null CPU-process GPU counters mean unavailable/not reported, not a measured global zero. Device-wide counters include other processes.

| Principal native process metric |26B Vulkan|E4B CPU|Fallback CPU|
| --- | ---: | ---: | ---: |
|Working-set sampled maximum|11.165GB|10.715GB|11.678GB|
|Private-commit sampled maximum|18.837GB|9.502GB|9.548GB|
|Process dedicated-GPU sampled maximum|7.647GB|not reported|not reported|
|Process shared-GPU sampled maximum|10.934GB|not reported|not reported|
|Device-wide dedicated-GPU sampled maximum|13.221GB|6.079GB|6.104GB|
|Device-wide shared-GPU sampled maximum|11.525GB|0.697GB|0.710GB|

System paging context, median /p95 /max per second:

| Counter |26B|CPU|Fallback CPU|
| --- | ---: | ---: | ---: |
|Pages input|291.33 /526640.41 /729638.73|123.05 /12966.93 /44925.76|181.43 /29654.28 /194737.22|
|Page reads|50.40 /4246.15 /9745.75|15.37 /1674.47 /6202.66|30.81 /1944.11 /6899.40|

These are global rates, not attribution of every read to the model. High paging and severe transient available-memory depletion are material limitations. The24GB-class GPU workstation observations do not establish8GB-card residency, physical-RAM headroom or live-game frame-time guarantees. Full resource samples and PID-specific phase minima remain in the durable execution records.

## Separate671 after-proof observations

These are the two authorized **source-native after-repair cases**, not a rerun of the8786 comparison or CPU/fallback matrix. Complete per-PID timestamps, samples, stages, threshold brackets, true prelaunch observations and accounting: [afterproof-memory-pressure.json](results/afterproof-memory-pressure.json). Actual source candidate was `671ca2ac68a184a2d6915e9efd36464b52d5b0f5` in both lifecycles. The source window completed and all owned children/ports/tasks were released before offline analysis.

### 26B retry: severe loading depletion despite completion

Actual26B child69864 completed six native translations with the exact fixed Vulkan profile. Its **true prelaunch** observation at monotonic100200.8270583 reported5,045,506,048B available physical,9,941,684,224B commit headroom,74,494,599,168B committed and84,436,283,392B reported commit limit. This sample preceded actual process creation, unlike the original postspawn proxy.

Available physical reached **155,648B(152KiB)** at100204.7551752 during `model-load`. It is material transient depletion, not a RAM-safety pass:

| Threshold | Consecutive-low sampled observations | Sample-supported span | Surrounding above-threshold bracket |
| --- | --- | ---: | --- |
|Physical<256MiB|21 samples,100202.9036697–100208.2886809|5.385011s|100202.6392162–100208.5534793,5.914263s|
|Physical<512MiB|22 samples,100202.9036697–100208.5534793|5.649810s|100202.6392162–100208.8176277,6.178412s|

All low observations occurred during loading, before initial prefix work. The153 live-PID samples had median gap0.263958s and maximum0.342108s; first live observation followed recorded spawn by0.467344s and last preceded exit by0.026481s. The configured watchdog's longest possible low window bounded by recorded above observations was5.914263s, below its8s threshold; no cutoff event occurred. These discrete spans/brackets do not establish exact continuous crossing times or safety under another workload.

Loading minimum commit headroom was3,380,461,568B; at the physical-minimum sample the reported commit limit had changed to94,224,793,600B. System commit limits and background conditions were not fixed. Subsequent startup-first-prefix available minimum was2,794,233,856B, first-language-prefix2,775,625,728B, and overlapping active-group minima2,960,510,976B/2,911,354,880B. Do not attribute the loading minimum to steady inference.

|26B live-PID accounting|Sampled maximum|
|---|---:|
|Process working set|11,171,667,968B|
|Process private commitment|18,818,478,080B|
|Process dedicated/shared GPU|7,639,064,576B /10,940,575,744B|
|Device-wide dedicated/shared GPU|20,268,748,800B /12,102,938,624B|

Global pages-input rate median/p95/max was4,569.16/616,033.72/690,310.15 per second; page-read rate561.67/3,314.84/5,732.07. These are global rates and device totals, not attribution to the model alone. Highwaters remain sampled lower bounds; commitment is not physical residency and shared GPU bytes alone do not diagnose spill.

### E4B GPU→CPU/MTP→GPU

All three owned processes received true prelaunch system observations:

| PID/backend | Prelaunch available physical /commit headroom | Live physical minimum and phase | Maximum sample gap |
|---|---|---|---:|
|50884 Vulkan|14,094,209,024B /18,025,148,416B|2,780,524,544B,`prefix-lru-en-ko`|0.295022s|
|72040 CPU/MTP|12,788,613,120B /19,310,800,896B|2,063,863,808B,`active-backend-transition`|0.289212s|
|69380 returned Vulkan|8,409,595,904B /14,624,907,264B|3,444,895,744B,`shutdown`|0.265965s|

No sample in these live windows was below either physical threshold or the configured watchdog predicate. Independent STT degraded receipts remain truthful; this is effective managed-translation progress, not full-app health. Detailed per-phase accounting and unavailable CPU-process GPU counters remain in the JSON rather than being replaced by global zeros.

### Owned resources and baseline-comparison limits

No retained task-owned native child or prior model host was identified as active before the26B launch; original cleanup evidence and the new terminal supervisors remain distinct observations. This worker did not investigate private/live unrelated processes, terminate unrelated applications, trim working sets or repeat a model run.

The fixture uses streaming `hashlib.file_digest` and hardlinks, not a Python buffer holding model weights; native stdout drains to files. Required model-integrity hashing warms filesystem cache and may change the available-memory context, but no causal contribution is established. The source observer's own prelaunch working set/private commitment was **not sampled**, so its contribution is unmeasured. Offline JSON/raw-journal/quality containers were created only after both model cases and hardware release; they cannot explain the earlier5.045GB baseline and were explicitly dropped after durable records were written, without claiming measured memory recovery.

The original9,865,506,816B value was an earliest **postspawn proxy** under different run conditions, not a matched prelaunch baseline. Do not subtract the two as a measured regression or infer OOM, uncontrolled spill, guaranteed physical residency,8GB-card support or live-game safety. After this severe pressure was reported, the Director authorized the separate native owner's full build only; further model execution remains held for a fresh postbuild baseline/safety assessment.

