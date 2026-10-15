# Issue 254 — Gemma 4 26B QAT expert placement and 8-GiB cache tuning

## Result and authority

**Experimental recommendation: n24 / requested cache2890 MiB / load-mode none**, retaining every other fixed control. Its lifecycle sampled PID dedicated peak was **8,526,270,464 bytes = 7.940708 GiB = 8.526270 GB**. Across two fresh lifecycles its simultaneous-pair medians were **755.156 and 760.529 ms**, below every matched control median (**866.735–910.656 ms**, three lifecycles). Pooled original-pair median improved **116.517 ms (13.31%)**, from875.094 to758.577 ms.

**The tolerance band did not buy a repeatable improvement.** n23/2642 had a lower pooled median, but its second run violated the required every-candidate-run-below-every-target-run standard. No run is dropped to rescue that comparison.

Completed **49 clean native lifecycles, 2,156/2,156 EOS translations, 34 measured configurations, 735/735 simultaneous rounds with native active-slot overlap**, all graceful exit0 and no watchdog stop. Every configuration maximum includes all its clean boundary-sweep and repeated runs. Four earlier contaminated identities/162 completed translations are retained separately and excluded from both timing comparison and clean memory classification.

This is native-only experimental evidence and a recommendation, **not owner adoption, product rollout, managed26B integration, or app/Self-Peer parallelization**. No product source, maintained documentation, live settings, driver/backend, Git history or GitHub state was changed by the measurement worker. The Director owns review, pinning/publication and routing of the project owner’s disposition. The [#245 final owner decision](https://github.com/kapitalismho/PuriPuly-heart/issues/245#issuecomment-6196060251) remains authoritative until separately superseded.

Primary sources: [issue254](https://github.com/kapitalismho/PuriPuly-heart/issues/254), [pinned original Self/Peer report](https://github.com/kapitalismho/PuriPuly-heart/blob/c3da5bb5f97ede9b2cfb31ce1d0f1f590e7e2ec6/experiments/gemma4_26b_qat/SELFPEER.md). [Evidence index](results/tuning8gib/evidence-index.json), [complete machine-readable analysis](results/tuning8gib/clean/summary.json), [every configuration/run/phase/channel timing table](TUNING8GIB_TIMINGS.md).

## Fixed identity, workload and observation boundary

- Execution baseline: `ccab84af268ba58ce47d501132b69e7a583d861b`, plus task-local experimental helpers only. Historical workload source: `c3da5bb5f97ede9b2cfb31ce1d0f1f590e7e2ec6`. The inherited original-product baseline label in the first seven clean JSON files is explicitly corrected in [metadata-corrections.json](results/tuning8gib/clean/metadata-corrections.json); `experiment_baseline` was correct throughout. Later records distinguish current baseline from fixture-origin product commit.
- Verified artifact: `unsloth/gemma-4-26B-A4B-it-qat-GGUF` / `gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf`, revision `7b92b5b28818151e8669af2e45e88d6086f490dd`, **14,249,047,104 bytes**, SHA256 `a7c5bc715f5ff8e99a3e8901ce7d2b42b402c669bf24f7c5250747633d0f5891`. A prerequisite researcher freshly hashed the complete file before launch authorization; identity was reused, not re-downloaded or substituted.
- Runtime: **llama.cpp b11537 / `8e2d31e0eb658d0382d3106a61369dff81f0c080`**, executable SHA256 `7b36a3bc3a24e552d0fcc18441893bc198abaeb0fa77d653241d762077c995c6`; official archive SHA256 `ae34d6a850c5d362cdfc0ef8f88f31fc499dab6df8853cec7887a32891aa5bbe`. Each run records executable hash and effective native command.
- Host: Windows11, Ryzen7 9800X3D(8C/16T),32-GiB RAM class, RX7900XTX24-GiB class; driver32.0.31041.1004 / Vulkan26.8.1. [Sanitized prerequisite evidence](results/tuning8gib/prerequisites.json).
- Fixed native controls: Vulkan0, `--n-gpu-layers99 --load-mode none --swa-full --cache-ram0`, context4608/parallel3/effective1536 per slot, threads4/4,batch/ubatch512,f16KV,flash-attn on,fit off,reasoning off/budget0,warmup; **MTP off**. Full token-separated flags are in every run JSON; compact prose is not a substitute for those arrays.
- Unchanged44-request workload: Self Korean→English pinned slot0, Peer English→Korean pinned slot1, slot2 unused. Two first-prefix warmups(temp0), six serial pair rounds(12requests), fifteen barrier-released simultaneous pair rounds(30requests), original CASES/history/prompt, temperature.6/seed42/max96. Serial first-channel order alternates. The original workload and request functions were retained.
- [Workload inheritance check](results/tuning8gib/clean/workload-inheritance.json): all44 request identities/prompt hashes and actual evaluated/cache-hit prompt counts match the pinned #245 control. All49 clean runs also match the current control at every corresponding identity and P/C count. **95 corresponding-request count differences versus the first clean control are retained; all are output-token differences**, not prompt/cache mismatch or forced equality. The unique wording set remains nine pairs.

Timing conventions are inherited: API prompt/decode durations and TPS belong to the original request; TPS is **per request, not summed concurrent throughput**. P=uncached prompt evaluations,C=prior KV hits,O=API predicted output. Full cached prompt size is not divided by prefill time. Native printed decode-evaluation counts are unavailable in API summaries, not invented from O; raw native timing lines remain in each sanitized log. HTTP wall and TTFT begin before apply-template and end at complete response/first nonempty content. Pair latency is common release→both complete; serial makespan is start before the first routine→second complete, not a sum of stage medians. Native slot intervals prove concurrent active requests, **not simultaneous GPU kernels**. Native timings are neither pure kernel time nor app/audio E2E. Serial always precedes simultaneous, so their ratio is descriptive, not an isolated app-concurrency causal estimate.

## Contamination exclusion and clean restart

The owner reported a brief overlap with another local-model experiment and explicitly requested remeasurement. That statement is ground truth; the past overlap was not rechecked. **All pre-restart runs below are contaminated and comparison-ineligible**, including their dedicated-memory peaks. Their original JSON/logs/samples remain at `results/tuning8gib/`; the only current performance cohort is `results/tuning8gib/clean/`.

| Excluded identity | UTC start | Completed | Outcome |
| --- | --- | ---: | --- |
| tuning8gib-n24-c2048-a1 | 22:43:48 |44|Complete, graceful0; excluded|
| tuning8gib-n23-c2048-curve |22:44:48|44|Complete, graceful0; excluded|
| tuning8gib-n22-c2048-curve |22:45:46|44|Complete, graceful0; excluded|
| tuning8gib-n21-c2048-curve |22:46:18|30|Owner-requested graceful interruption of owned PID4048; subsequent HTTP rejection; excluded|

All starts above are2026-10-15 UTC. The fourth record retains its original `failed`/HTTP-error classification and an explicit correction: **owner-requested interruption, not OOM, native instability or watchdog**. Only the verified owned server was signalled. Its failing wrapper prevented queued n20/n19/n18 launches. [Exact exclusion/stop ledger](results/tuning8gib/contamination.json).

Fresh empty-model preflight: **2026-10-15T22:48:04.335131+00:00**. First clean launch: **2026-10-15T22:48:04.872402+00:00**. No cross-session hub endpoint was available; the Director authorized restart on a fresh empty runtime-name preflight. Every clean lifecycle then observed known model executable names/PIDs at the250-ms sampler cadence, retaining transitions and total observations, without unrelated command lines or process stops. **5,825 lifecycle conflict observations found no other named model runtime**. This is a bounded observation: a differently named/python-hosted model or a process entirely between samples could be missed. Any detected new overlap would have excluded the full run and stopped the dependent sequence.

## Memory accounting and boundary method

Eligibility uses **maximum whole-lifecycle sampled server-PID dedicated usage across all clean runs of the configuration**: target≤8,589,934,592B; band>target and≤8,697,308,774B; over-ceiling descriptive only. Units: GiB=2^30B,MiB=2^20B,GB=10^9B. Process dedicated counters use exact adapter **`luid_0x00000000_0x0000f60b_phys_0`** and the actual owned PID, with counter instances retained. Load, prefix, serial, simultaneous and shutdown are covered; a final post-exit handle sample retains lifecycle CPU. Actual sampled counter coverage is checked during load and every workload phase.

Nominal cadence250ms; **5,874 total samples**, per-run cadence medians248.658–248.949ms, largest observed interval266.676ms. Windows scheduling and the additional final shutdown sample explain non-exact intervals. Peaks are **sampled lower bounds**, not exact lifetime high-water marks. Whole-device dedicated usage, process shared-GPU accounting, working set and private commitment are separate, overlapping views and **must not be added**. Private commitment is not resident physical RAM; large shared accounting is consistent with the logged deliberate Vulkan_Host weight buffers and is not by itself proof of uncontrolled spill. Global paging/background counters remain in samples but are not attributed exclusively to this process.

Cache boundaries are searched in **effective expert-slot space**, not arbitrary MiB grids. For this pinned homogeneous single-device group, [source-backed sizing](results/tuning8gib/cache-semantics.json) is A(s)=(s+1)×3,345,408B, or3.1904296875MiB per extra usable slot, including one padding plane. Canonical integer request R(s)=ceil(A(s)/2^20); all requests R(s)..R(s+1)−1 share the same effective allocation. Actual native group/slot and rounded effective-size logs are checked on every run. Starting estimates come from each measured fixed2048 point, not a universal assumed layer slope; measured lifecycle peaks decide classification. A highest observed passing slot and next distinct observed over-boundary slot bracket every final target placement and each investigated band placement.

These are measured local frontiers, not a mathematical proof of global monotonicity of noisy sampled process peaks. Margins and observed same-config peak ranges are below. n1 means variation is **unmeasured**, not known zero. A positive margin smaller than that configuration’s observed variation is labelled marginal in JSON. Extremely small positive margins remain fragile even when n2/n3 happens to show zero variation; this is not a physical-card fit guarantee.

## Fixed2048-MiB placement curve

Pooled original observations per configuration; simultaneous per-request native medians. All points use640 usable cache slots/effective2045.07MiB. Per-phase/channel/tail/token detail is in TUNING8GIB_TIMINGS.md. Host-memory views for every row are also listed in the all-configuration resource table below.

| Placement | Runs | Dedicated bytes / GiB / GB | Class | Small-batch hit% | Prefill ms / TPS | Decode ms / TPS | Pair median / p95 ms |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |
|n24|3|7,640,506,368 /7.115776 /7.640506|target|70.01|203.083 /126.657|538.019 /28.490|875.094 /1039.176|
|n23|2|8,068,718,592 /7.514580 /8.068719|target|71.59|199.181 /130.574|507.315 /29.909|837.501 /1028.812|
|n22|2|8,497,324,032 /7.913750 /8.497324|target|72.97|189.380 /135.067|489.945 /30.627|817.785 /973.638|
|n21|1|8,925,536,256 /8.312553 /8.925536|over_ceiling|74.60|180.188 /142.695|474.377 /32.196|771.482 /944.886|
|n20|1|9,354,010,624 /8.711601 /9.354011|over_ceiling|76.02|173.952 /148.004|492.011 /33.457|757.132 /910.233|
|n19|1|9,782,353,920 /9.110527 /9.782354|over_ceiling|77.59|172.552 /150.592|420.257 /36.306|709.646 /855.999|
|n18|1|10,210,828,288 /9.509575 /10.210828|over_ceiling|79.26|165.025 /162.533|399.053 /36.471|689.887 /796.303|

### One additional GPU-resident expert layer at fixed2048

Changes versus the preceding placement, not causal kernel attribution. Small steps are not interpreted as improvements when they are below observed run spread; single-run points do not establish zero uncertainty. Native decode duration can move differently from pair latency because each independently calculated median includes actual per-request contention/output counts.

| Step | Δdedicated GiB | Δprefill ms / TPS | Δdecode ms / TPS | Δpair median / p95 ms | ΔWS GiB | Δprivate GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
|n24-c2048→n23-c2048|+0.398804|-3.902 /+3.917|-30.704 /+1.419|-37.594 /-10.363|-0.383724|-0.006630|
|n23-c2048→n22-c2048|+0.399170|-9.801 /+4.493|-17.371 /+0.718|-19.716 /-55.174|-0.393051|+0.007935|
|n22-c2048→n21-c2048|+0.398804|-9.192 /+7.628|-15.567 /+1.569|-46.303 /-28.752|-0.308693|-0.016960|
|n21-c2048→n20-c2048|+0.399048|-6.235 /+5.309|+17.633 /+1.262|-14.350 /-34.653|-0.418610|-0.000534|
|n20-c2048→n19-c2048|+0.398926|-1.400 /+2.588|-71.754 /+2.848|-47.485 /-54.234|-0.340607|+0.023621|
|n19-c2048→n18-c2048|+0.399048|-7.527 /+11.941|-21.204 /+0.166|-19.759 /-59.696|-0.381035|+0.013603|

Most per-step dedicated changes are about0.399GiB, but the measured points—not this slope—determined target caches. The n24→n23 pooled pair change is smaller than the control’s43.921-ms run-median range. Later small single-run curve differences remain descriptive. The full phase/channel delta object is `placement_curve_steps` in summary.json.

## Largest measured effective target caches after all repeats

The integer ranges below are allocation plateaus, not additional measured configurations. Next-slot peaks use **all-run maxima**, including the later n24 reclassification. [Initial search](results/tuning8gib/clean/search-target1.json); [final n24 search](results/tuning8gib/clean/search-target2.json).

| N | Actual request / equivalent integer MiB | Effective native MiB / slots | Runs | Peak bytes / GiB / GB | Target margin MiB | Ceiling margin MiB | Next slot request / observed peak bytes |
| --- | --- | --- | ---: | --- | ---: | ---: | --- |
|24|2945 /2945–2947|2944.77 /922|1|8,586,092,544 /7.996422 /8.586093|3.664062|106.064062|2948 /8,590,503,936|
|23|2540 /2540–2542|2539.58 /795|3|8,588,566,528 /7.998726 /8.588567|1.304688|103.704687|2543 /8,591,712,256|
|22|2132 /2132–2134|2131.21 /667|3|8,587,763,712 /7.997978 /8.587764|2.070312|104.470312|2135 /8,590,909,440|
|21|1723 /1723–1726|1722.83 /539|3|8,587,362,304 /7.997604 /8.587362|2.453125|104.853125|1727 /8,590,508,032|
|20|1318 /1318–1320|1317.65 /412|1|8,589,574,144 /7.999664 /8.589574|0.343750|102.743750|1321 /8,593,244,160|
|19|910 /910–912|909.27 /284|1|8,589,303,808 /7.999413 /8.589304|0.601562|103.001562|913 /8,593,453,056|
|18|501 /501–504|500.90 /156|1|8,588,369,920 /7.998543 /8.588370|1.492188|103.892187|505 /8,591,777,792|

**Mandatory reclassification example:** n24/2948 initially appeared target-class at8,589,500,416B, only434,176B below target. Its third clean lifecycle peaked8,590,503,936B. Across all three runs it is therefore **BAND**, not target; variation1,003,520B(.957031MiB) exceeded its original.414063MiB target margin. All three timings and the larger peak remain in the same configuration. The final target bracket backed off to922slots/request2945, with923slots/request2948 as the observed over-target neighbor. The source search ledger `target1` is historical, not the final classification.

Every n24–n18 placement has an enabled measured target-class configuration. n20–n18 target-filling medians were882.456,887.537 and1,085.415ms respectively; their neighboring smaller/larger-slot measurements also remain in the ledger. They did not lead this workload, but this is not a broad model/placement inferiority claim.

## Tolerance-band frontiers

[Band search](results/tuning8gib/clean/search-band1.json). Leading target placements and nearby contenders were measured to their next-slot ceiling brackets; no full factorial sweep was needed.

| N / request MiB | Effective MiB / slots | Runs | Peak bytes / GiB / GB | Ceiling margin MiB | Run pair medians ms | Next slot request / over-ceiling peak bytes |
| --- | --- | ---: | --- | ---: | --- | --- |
|n24/3051|3050.05 /955|2|8,696,455,168 /8.099205 /8.696455|0.814062|781.236, 784.430|3054 /8,699,600,896|
|n23/2642|2641.68 /827|2|8,695,521,280 /8.098335 /8.695521|1.704687|734.330, 764.128|2645 /8,698,667,008|
|n22/2234|2233.30 /699|1|8,695,250,944 /8.098083 /8.695251|1.962500|799.204|2237 /8,698,396,672|
|n21/1825|1824.93 /571|1|8,694,317,056 /8.097214 /8.694317|2.853125|834.164|1829 /8,698,466,304|

n22/2234 and n21/1825 remain descriptive n1: their first run medians799.204/834.164ms already exceed the leading target’s retained minimum755.156ms. Additional runs cannot make **every** candidate run lower than **every** target run. This is a specific failure of the prescribed recommendation rule, not a universal claim of inferiority. All near-target band neighbors are retained as well.

**Band rule fails:** n23/2642 medians734.330/764.128ms overlap n24/2890’s755.156/760.529ms. Its slower repeat is not below the target’s fastest repeat. It also fails against the n23/2540 target set(min754.423ms). Pooled n23/2642 buys only6.242ms versus the selected refinement for169,250,816B(161.410156MiB) extra dedicated memory, or3.867ms/100MiB; pairp95 is **13.186ms worse**. Those pooled numbers do not override the negative repeat result. No further run was added to rescue the failed rule.

## Repeated finalists, lower-memory alternatives and tradeoffs

All original pair observations are pooled before percentiles; no percentile-of-percentile. Run counts include boundary sweeps. Every repeated configuration was fresh-lifecycle and only one server was loaded. Forward placement order24→23→22→21 was reversed21→22→23→24; band24→23 was reversed23→24. Lower-memory curve alternatives reversed23→22 to22→23. The2890 refinement bracketed the reverse block; controlA2/A3 bracketed the finalist/band work, with A1 retained. [Explicit reverse plan](results/tuning8gib/clean/repeat-plan.json); the lifecycle table below is actual execution order.

| Config | Class / runs | All run pair medians ms | Pooled pair median / p95 ms | Serial median / p95 ms | Peak GiB |
| --- | --- | --- | ---: | ---: | ---: |
|n24-c2048|target /3|910.656, 866.735, 872.872|875.094 /1039.176|1122.033 /1216.332|7.115776|
|n23-c2048|target /2|840.446, 836.516|837.501 /1028.812|1085.551 /1209.751|7.514580|
|n22-c2048|target /2|827.768, 815.939|817.785 /973.638|1054.347 /1176.719|7.913750|
|n24-c2890|target /2|755.156, 760.529|758.577 /905.861|1000.620 /1184.467|7.940708|
|n24-c2945|target /1|762.144|762.144 /924.904|990.219 /1068.960|7.996422|
|n24-c2948|tolerance_band /3|758.024, 777.823, 793.072|777.823 /929.311|1008.657 /1120.148|8.000530|
|n23-c2540|target /3|772.173, 754.423, 789.894|784.059 /921.825|1028.705 /1134.705|7.998726|
|n22-c2132|target /3|822.223, 815.192, 831.528|822.223 /984.395|1036.731 /1159.908|7.997978|
|n21-c1723|target /3|825.221, 810.848, 824.649|824.649 /974.481|1054.570 /1168.381|7.997604|
|n24-c3051|tolerance_band /2|781.236, 784.430|783.649 /967.870|974.544 /1116.120|8.099205|
|n23-c2642|tolerance_band /2|734.330, 764.128|752.335 /919.047|993.740 /1117.067|8.098335|
|n22-c2234|tolerance_band /1|799.204|799.204 /947.690|1013.958 /1103.554|8.098083|
|n21-c1825|tolerance_band /1|834.164|834.164 /983.184|1051.368 /1140.644|8.097214|

- **Target improvement is repeatable:** worst2890 run760.529ms < best control866.735ms. Pooled median/p95 improve116.517/133.314ms; serial median improves121.414ms. No latency SLA is invented.
- **n23/2540 versus2890:** medians overlap; no repeatable superiority is claimed between them. Their pooled25.483-ms difference is within n23’s35.471-ms run-median range.2890 uses62,296,064 fewer dedicated bytes, so the equivalent-spread rule favors2890. Its serialp95 is **49.762ms worse** than n23/2540(1,184.467 vs1,134.705ms), despite better pooled serial median; retain that tail tradeoff.
- **n23/2048 is not equivalent:** both medians836.516/840.446ms exceed both2890 medians; the nearest separation is75.987ms, versus3.930ms observed n23 spread and5.373ms2890 spread. n22/2048 is likewise not equivalent:815.939/827.768ms, nearest separation55.410ms versus11.830ms n22 spread. Both use less dedicated memory, but their latency penalty is larger than the observed variation. For n22/2048 the selected cache costs only28,946,432B more; lower memory alone does not overturn the latency result.
- **Final n24/2945 max-target point:**762.144ms, n1, higher VRAM than2890 and within its5.373-ms observed spread. Its retained first median already prevents every2945 run from being below every2890 run; a larger-memory equivalent candidate does not displace2890. No extra repetition was decision-relevant.
- **Serial/band tradeoffs are not hidden:** n24/3051 has faster serial median/p95 than2890 but worse simultaneous median/p95 and fails the primary rule. n23/2642 improves serial median/p95 versus2890 but worsens simultaneousp95 and fails the repeat rule.

### Simultaneous per-request finalists by channel

| Config / channel | Prefill ms / TPS | Decode ms / TPS | P/C/O median | HTTP median / p95 ms | TTFT median / p95 ms |
| --- | ---: | ---: | --- | ---: | ---: |
|n24-c2048/Self|205.651 /124.571|524.184 /28.483|26/882/16|829.077 /933.992|242.191 /421.672|
|n24-c2048/Peer|198.768 /128.127|599.459 /28.497|26/884/15|851.841 /1038.901|215.304 /417.275|
|n24-c2890/Self|191.636 /140.061|419.219 /34.480|26.0/882.0/16.0|692.353 /802.484|217.220 /367.823|
|n24-c2890/Peer|187.486 /138.766|516.103 /34.001|26.0/884.0/15.0|672.239 /905.753|205.453 /376.725|
|n23-c2540/Self|190.455 /145.065|433.474 /31.874|26/882/16|707.205 /834.440|213.735 /389.403|
|n23-c2540/Peer|190.402 /135.420|522.574 /33.254|26/884/15|751.516 /921.547|207.644 /375.405|
|n23-c2642/Self|185.285 /139.805|455.038 /32.255|26.0/882.0/16.0|717.623 /809.532|203.827 /390.308|
|n23-c2642/Peer|182.167 /140.018|431.686 /33.016|26.0/884.0/15.0|717.438 /918.953|211.812 /371.459|

## Every measured configuration: memory class and margins

All-run lifecycle process-dedicated maximum. Negative margins are outside that boundary. Variation is observed max−min peak; a zero withn1 is unmeasured variability. JSON’s `marginal` flag compares positive boundary margin to **that same configuration’s** observed variation. Very small margins remain lower-bound/limited-repeat observations, not guaranteed fit. The selected2890 profile has60.714844MiB target margin and163.114843MiB ceiling margin versus.957031MiB observed peak variation; neither boundary is marginal by the specified test.

| Config | Runs | Class | Dedicated bytes / GiB / GB | Target / ceiling margin MiB | Peak range MiB |
| --- | ---: | --- | --- | ---: | ---: |
|n24-c2048|3|target|7,640,506,368 /7.115776 /7.640506|905.445312 /1007.845312|0.000000|
|n24-c2890|2|target|8,526,270,464 /7.940708 /8.526270|60.714844 /163.114843|0.957031|
|n24-c2945|1|target|8,586,092,544 /7.996422 /8.586093|3.664062 /106.064062|n1: unmeasured|
|n24-c2948|3|tolerance_band|8,590,503,936 /8.000530 /8.590504|-0.542969 /101.857031|0.957031|
|n24-c2952|1|tolerance_band|8,593,649,664 /8.003460 /8.593650|-3.542969 /98.857031|n1: unmeasured|
|n24-c3051|2|tolerance_band|8,696,455,168 /8.099205 /8.696455|-101.585938 /0.814062|0.000000|
|n24-c3054|1|over_ceiling|8,699,600,896 /8.102135 /8.699601|-104.585938 /-2.185938|n1: unmeasured|
|n23-c2048|2|target|8,068,718,592 /7.514580 /8.068719|497.070312 /599.470312|0.000000|
|n23-c2540|3|target|8,588,566,528 /7.998726 /8.588567|1.304688 /103.704687|0.000000|
|n23-c2543|1|tolerance_band|8,591,712,256 /8.001656 /8.591712|-1.695312 /100.704687|n1: unmeasured|
|n23-c2642|2|tolerance_band|8,695,521,280 /8.098335 /8.695521|-100.695312 /1.704687|0.000000|
|n23-c2645|1|over_ceiling|8,698,667,008 /8.101265 /8.698667|-103.695312 /-1.295313|n1: unmeasured|
|n22-c2048|2|target|8,497,324,032 /7.913750 /8.497324|88.320312 /190.720312|0.000000|
|n22-c2132|3|target|8,587,763,712 /7.997978 /8.587764|2.070312 /104.470312|0.000000|
|n22-c2135|1|tolerance_band|8,590,909,440 /8.000908 /8.590909|-0.929688 /101.470312|n1: unmeasured|
|n22-c2234|1|tolerance_band|8,695,250,944 /8.098083 /8.695251|-100.437500 /1.962500|n1: unmeasured|
|n22-c2237|1|over_ceiling|8,698,396,672 /8.101013 /8.698397|-103.437500 /-1.037500|n1: unmeasured|
|n21-c1723|3|target|8,587,362,304 /7.997604 /8.587362|2.453125 /104.853125|0.000000|
|n21-c1727|1|tolerance_band|8,590,508,032 /8.000534 /8.590508|-0.546875 /101.853125|n1: unmeasured|
|n21-c1825|1|tolerance_band|8,694,317,056 /8.097214 /8.694317|-99.546875 /2.853125|n1: unmeasured|
|n21-c1829|1|over_ceiling|8,698,466,304 /8.101078 /8.698466|-103.503906 /-1.103907|n1: unmeasured|
|n21-c2048|1|over_ceiling|8,925,536,256 /8.312553 /8.925536|-320.054688 /-217.654688|n1: unmeasured|
|n20-c1315|1|target|8,586,428,416 /7.996735 /8.586428|3.343750 /105.743750|n1: unmeasured|
|n20-c1318|1|target|8,589,574,144 /7.999664 /8.589574|0.343750 /102.743750|n1: unmeasured|
|n20-c1321|1|tolerance_band|8,593,244,160 /8.003082 /8.593244|-3.156250 /99.243750|n1: unmeasured|
|n20-c2048|1|over_ceiling|9,354,010,624 /8.711601 /9.354011|-728.679688 /-626.279688|n1: unmeasured|
|n19-c907|1|target|8,585,895,936 /7.996239 /8.585896|3.851562 /106.251562|n1: unmeasured|
|n19-c910|1|target|8,589,303,808 /7.999413 /8.589304|0.601562 /103.001562|n1: unmeasured|
|n19-c913|1|tolerance_band|8,593,453,056 /8.003277 /8.593453|-3.355469 /99.044531|n1: unmeasured|
|n19-c2048|1|over_ceiling|9,782,353,920 /9.110527 /9.782354|-1137.179688 /-1034.779688|n1: unmeasured|
|n18-c498|1|target|8,585,965,568 /7.996304 /8.585966|3.785156 /106.185156|n1: unmeasured|
|n18-c501|1|target|8,588,369,920 /7.998543 /8.588370|1.492188 /103.892187|n1: unmeasured|
|n18-c505|1|tolerance_band|8,591,777,792 /8.001717 /8.591778|-1.757812 /100.642187|n1: unmeasured|
|n18-c2048|1|over_ceiling|10,210,828,288 /9.509575 /10.210828|-1545.804688 /-1443.404688|n1: unmeasured|

### Separate overlapping memory views — do not add

Each column is its own lifecycle maximum, possibly at a different time. Primary raw bytes and GB/GiB conversions are in summary.json. Shared-GPU accounting is not counted toward either target. The selected profile’s maxima are WS11,267,280,896B(10.493473GiB), private19,809,718,272B(18.449238GiB), sharedGPU10,938,478,592B(10.187252GiB), whole-device dedicated12,800,798,720B(11.921673GiB). These are not four additive allocations.

| Config | Host WS GiB | Private commit GiB | Process shared GPU GiB | Whole-device dedicated GiB |
| --- | ---: | ---: | ---: | ---: |
|n24-c2048|10.519531|17.630974|10.222393|11.217590|
|n24-c2890|10.493473|18.449238|10.187252|11.921673|
|n24-c2945|10.496170|18.483330|10.175533|11.980072|
|n24-c2948|10.502842|18.505539|10.216549|12.037201|
|n24-c2952|10.495659|18.487373|10.181393|12.042816|
|n24-c3051|10.495701|18.601830|10.208736|12.136810|
|n24-c3054|10.514088|18.611263|10.224361|12.125580|
|n23-c2048|10.135807|17.624344|9.837269|11.604431|
|n23-c2540|10.142620|18.111782|9.843128|12.036331|
|n23-c2543|10.128193|18.110371|9.833363|12.040482|
|n23-c2642|10.121864|18.208492|9.829456|12.137894|
|n23-c2645|10.127823|18.207085|9.833363|12.082230|
|n22-c2048|9.742756|17.632278|9.444324|12.028992|
|n22-c2132|9.796974|17.706127|9.475559|12.035583|
|n22-c2135|9.723190|17.703400|9.430653|12.038513|
|n22-c2234|9.719521|17.813847|9.428699|12.079048|
|n22-c2237|9.721939|17.816116|9.430653|12.081978|
|n21-c1723|9.384865|17.306274|9.102142|12.035210|
|n21-c1727|9.437149|17.302719|9.104095|12.038139|
|n21-c1825|9.384190|17.400192|9.072861|12.078178|
|n21-c1829|9.392467|17.404690|9.078720|12.082043|
|n21-c2048|9.434063|17.615318|9.104111|12.403870|
|n20-c1315|9.000435|16.903442|8.683823|12.020180|
|n20-c1318|9.015369|16.902000|8.693588|12.042641|
|n20-c1321|9.029781|16.910473|8.703354|12.040688|
|n20-c2048|9.015453|17.614784|8.693588|12.810730|
|n19-c907|8.634525|16.510525|8.306499|12.033844|
|n19-c910|8.630821|16.546349|8.304546|12.022858|
|n19-c913|8.670460|16.511055|8.329937|12.039948|
|n19-c2048|8.674847|17.638405|8.333843|13.299011|
|n18-c498|8.286430|16.134533|7.940899|12.032974|
|n18-c501|8.307289|16.131638|7.954571|12.037613|
|n18-c505|8.273304|16.134014|7.933086|12.041275|
|n18-c2048|8.293812|17.652008|7.946758|13.608704|

Actual `Vulkan_Host` model buffers shrink from10,197.01MiB atn24 to7,746.76MiB atn18. They are not multiplied by occupied request channels. Each run retains actual Vulkan_Host/Vulkan0 model, KV and compute-buffer lines; separate load/first-prefix/serial/simultaneous/shutdown resources remain in JSON. Cache-heavy n24 does not materially reduce host working set, and its extra private commitment is a cost, not hidden physical-RAM savings.

## Every configuration: cache allocation and execution

Requested cache is not total VRAM. All host-expert layers in this model’s one homogeneous group share the cache;128 experts/layer,8 active. LowerN increases a fixed cache’s fraction of remaining host experts. Native effective sizes are rounded to.01MiB; exact source-derived bytes are explicitly labelled calculated, not byte-exact native observation. Full native group lines and both destructor buckets/large-batch copies are retained per run. Pool hit rates below use summed hits/(hits+misses), not an average of percentages.

| Config | Requested / native effective MiB | Usable slots / host-expert coverage% | Small hits / misses / hit% | >8 hit% | Small / >8 uploaded MiB | Large tensor-bank copies / copied MiB |
| --- | ---: | --- | --- | ---: | ---: | ---: |
|n24-c2048|2048 /2045.07|640 /20.833|248,451 /106,450 /70.006|13.101|339621.23 /283351.63|27,532 /43919.46|
|n24-c2890|2890 /2887.34|904 /29.427|192,903 /44,793 /81.155|28.237|142908.91 /128784.89|30,620 /48845.48|
|n24-c2945|2945 /2944.77|922 /30.013|97,104 /21,651 /81.768|28.174|69075.99 /75520.66|12,562 /20039.09|
|n24-c2948|2948 /2947.96|923 /30.046|291,023 /64,960 /81.752|28.940|207250.31 /227152.22|36,394 /58056.25|
|n24-c2952|2952 /2951.15|924 /30.078|96,983 /21,365 /81.947|28.071|68163.53 /90426.35|8,472 /13514.66|
|n24-c3051|3051 /3050.05|955 /31.087|195,840 /40,775 /82.767|30.146|130089.77 /170949.61|18,626 /29712.47|
|n24-c3054|3054 /3053.24|956 /31.120|97,545 /20,368 /82.726|30.620|64982.67 /82766.13|10,098 /16108.48|
|n23-c2048|2048 /2045.07|640 /21.739|162,530 /64,499 /71.590|14.440|205779.52 /159508.72|21,618 /34485.35|
|n23-c2540|2540 /2539.58|795 /27.004|268,785 /73,421 /78.545|23.377|234244.55 /210156.79|38,920 /62085.77|
|n23-c2543|2543 /2542.77|796 /27.038|90,224 /24,586 /78.585|23.869|78439.90 /61463.63|15,032 /23979.27|
|n23-c2642|2642 /2641.68|827 /28.091|181,722 /45,684 /79.911|25.518|145751.59 /139396.25|25,920 /41347.97|
|n23-c2645|2645 /2644.87|828 /28.125|90,225 /22,803 /79.825|24.612|72751.37 /90991.05|7,712 /12302.30|
|n22-c2048|2048 /2045.07|640 /22.727|157,870 /58,483 /72.969|16.057|186585.90 /166017.20|18,130 /28921.25|
|n22-c2132|2132 /2131.21|667 /23.686|242,009 /83,151 /74.428|17.049|265287.42 /266614.64|24,272 /38719.05|
|n22-c2135|2135 /2134.40|668 /23.722|81,370 /27,911 /74.459|18.633|89048.08 /61524.25|13,092 /20884.55|
|n22-c2234|2234 /2233.30|699 /24.822|82,365 /26,131 /75.915|19.085|83369.12 /82565.13|9,018 /14385.65|
|n22-c2237|2237 /2236.49|700 /24.858|83,271 /25,866 /76.300|19.868|82523.65 /66514.08|12,548 /20016.76|
|n21-c1723|1723 /1722.83|539 /20.052|214,691 /96,333 /69.027|11.973|307343.67 /235610.05|26,292 /41941.39|
|n21-c1727|1727 /1726.02|540 /20.089|72,002 /31,810 /69.358|12.325|101487.57 /68855.85|10,314 /16453.05|
|n21-c1825|1825 /1824.93|571 /21.243|73,446 /30,677 /70.538|13.049|97872.81 /86033.13|7,638 /12184.25|
|n21-c1829|1829 /1828.12|572 /21.280|73,088 /30,045 /70.868|13.194|95856.46 /80306.31|8,794 /14028.32|
|n21-c2048|2048 /2045.07|640 /23.810|77,178 /26,273 /74.603|17.275|83822.16 /75871.61|9,558 /15247.06|
|n20-c1315|1315 /1314.46|411 /16.055|60,870 /38,468 /61.276|7.227|122729.45 /64258.44|8,846 /14111.27|
|n20-c1318|1318 /1317.65|412 /16.094|60,194 /38,382 /61.064|6.920|122455.07 /92484.18|5,244 /8365.31|
|n20-c1321|1321 /1320.84|413 /16.133|61,156 /38,072 /61.632|7.118|121466.04 /69902.31|8,232 /13131.81|
|n20-c2048|2048 /2045.07|640 /25.000|74,851 /23,615 /76.017|19.441|75342.00 /74799.62|8,196 /13074.38|
|n19-c907|907 /906.08|283 /11.637|49,296 /44,908 /52.329|2.969|143275.82 /63499.12|6,174 /9848.86|
|n19-c910|910 /909.27|284 /11.678|48,887 /44,557 /52.317|2.919|142155.98 /80322.26|4,712 /7516.65|
|n19-c913|913 /912.46|285 /11.719|49,237 /44,581 /52.481|2.933|142232.55 /75922.66|5,218 /8323.83|
|n19-c2048|2048 /2045.07|640 /26.316|73,302 /21,176 /77.586|21.647|67560.54 /48143.58|13,240 /21120.64|
|n18-c498|498 /497.71|155 /6.727|17,868 /70,290 /20.268|0.793|224255.30 /83407.40|2,374 /3787.04|
|n18-c501|501 /500.90|156 /6.771|16,956 /71,724 /19.120|0.672|228830.38 /78730.23|2,662 /4246.46|
|n18-c505|505 /504.09|157 /6.814|17,137 /71,253 /19.388|0.719|227327.69 /83656.26|2,394 /3818.94|
|n18-c2048|2048 /2045.07|640 /27.778|69,938 /18,300 /79.261|25.183|58384.86 /57998.82|8,926 /14238.89|

Counters cover full lifecycles including native warmup, not isolated phases or measured PCIe bandwidth. Large-copy counts are per expert **tensor-bank copy**, not necessarily distinct routed experts. Missing source-suppressed counters would be unavailable/unemitted rather than invented measured zero; all49 clean lifecycles emitted the retained allocation/group, destructor buckets and large-copy evidence. Cache coverage is not cache hit rate.

Source correction to the issue’s warning-only wording: in this pinned [cache implementation](https://github.com/ggml-org/llama.cpp/blob/8e2d31e0eb658d0382d3106a61369dff81f0c080/src/llama-moe-cache.cpp#L221-L450), an undersized group warns/disables, and if **no enabled group remains** the constructor throws. This single-group model therefore must not be assumed to run successfully cache-off below the minimum. The calculated enabled minimum is8slots/request29MiB. No undersized-cache or cache-off run was needed here; minimum tested caches still had155slots. Every measured positive-cache result is labelled from actual logs, not requested intent. The known >3072MiB Vulkan bank failure was not rerun or worked around; source/backend/model remained pinned.

## Efficiency versus current control, per GiB added/saved dedicated memory

Signed metric change divided by signed dedicated-GiB change. Negative time/host-memory values mean lower cost; positive TPS means higher rate. These ratios are descriptive, not marginal derivatives or causal transfer attribution. Control has zero denominator and isN/A. The table uses simultaneous pooled per-request timings; **every phase and channel** is retained at `configurations.<key>.versus_control_per_GiB.phases`, alongside exact host bytes/GiB. Host columns here convert that numerator toGiB/GiB. Every configuration is shown, including over-ceiling and negative performers.

| Config | Δdedicated GiB | Pair median / p95 ms/GiB | Prefill ms / TPS perGiB | Decode ms / TPS perGiB | WS / private GiB perGiB |
| --- | ---: | ---: | ---: | ---: | ---: |
|n24-c2048|0|N/A|N/A|N/A|N/A|
|n24-c2890|+0.824932|-141.245 /-161.607|-14.376 /+14.678|-97.806 /+6.843|-0.031588 /+0.991917|
|n24-c2945|+0.880646|-128.258 /-129.759|-19.289 /+17.439|-97.409 /+6.725|-0.026527 /+0.967876|
|n24-c2948|+0.884754|-109.941 /-124.176|-15.330 /+17.108|-90.895 /+5.633|-0.018863 /+0.988484|
|n24-c2952|+0.887684|-95.258 /-130.050|-26.201 /+20.808|-20.350 /+6.021|-0.026893 /+0.964757|
|n24-c3051|+0.983429|-92.986 /-72.507|-23.664 /+21.728|-38.019 /+4.220|-0.024232 /+0.987215|
|n24-c3054|+0.986359|-110.427 /-122.117|-22.775 /+24.903|-95.994 /+4.320|-0.005519 /+0.993847|
|n23-c2048|+0.398804|-94.266 /-25.986|-9.784 /+9.823|-76.989 /+3.557|-0.962188 /-0.016625|
|n23-c2540|+0.882950|-103.103 /-132.907|-14.332 /+11.440|-84.492 /+4.701|-0.426877 /+0.544548|
|n23-c2543|+0.885880|-113.195 /-99.066|-13.743 /+9.680|-119.283 /+5.651|-0.441751 /+0.541154|
|n23-c2642|+0.982559|-124.939 /-122.261|-20.751 /+13.598|-89.062 /+4.318|-0.404726 /+0.587770|
|n23-c2645|+0.985489|-81.924 /-112.145|-23.174 /+21.970|-26.243 /+4.691|-0.397476 /+0.584594|
|n22-c2048|+0.797974|-71.818 /-82.130|-17.172 /+10.539|-60.246 /+2.678|-0.973435 /+0.001635|
|n22-c2132|+0.882202|-59.931 /-62.096|-17.327 /+16.410|-15.910 /+3.518|-0.819038 /+0.085188|
|n22-c2135|+0.885132|-93.802 /-100.893|-12.792 /+9.413|-84.189 /+3.673|-0.899686 /+0.081825|
|n22-c2234|+0.982307|-77.257 /-93.133|-16.909 /+14.927|-39.432 /+4.710|-0.814420 /+0.186167|
|n22-c2237|+0.985237|-103.386 /-87.967|-17.727 /+13.334|-91.076 /+4.565|-0.809543 /+0.187917|
|n21-c1723|+0.881828|-57.205 /-73.364|-16.282 /+11.215|-54.161 /+2.373|-1.286720 /-0.368212|
|n21-c1727|+0.884758|-84.401 /-17.644|-16.430 /+8.932|-70.256 /+2.269|-1.223365 /-0.371011|
|n21-c1825|+0.981438|-41.704 /-57.051|-15.976 /+13.330|-22.997 /+3.438|-1.156815 /-0.235146|
|n21-c1829|+0.985302|-88.142 /-80.949|-16.836 /+12.322|-62.466 /+2.372|-1.143876 /-0.229660|
|n21-c2048|+1.196777|-86.576 /-78.786|-19.131 /+13.401|-53.177 /+3.096|-0.906993 /-0.013081|
|n20-c1315|+0.880959|-22.699 /-51.693|-0.339 /+4.758|-49.696 /+0.193|-1.724368 /-0.825841|
|n20-c1318|+0.883888|+8.329 /+51.305|-17.452 /+10.105|+42.955 /+0.676|-1.701756 /-0.824735|
|n20-c1321|+0.887306|-51.492 /-30.651|-10.900 /+4.627|-19.386 /+0.977|-1.678958 /-0.812009|
|n20-c2048|+1.595825|-73.919 /-80.800|-18.254 /+13.377|-28.830 /+3.113|-0.942508 /-0.010145|
|n19-c907|+0.880463|+24.214 /-99.199|+2.263 /-0.450|+9.031 /-1.246|-2.140927 /-1.272569|
|n19-c910|+0.883636|+14.081 /-7.121|-14.935 /+7.724|+34.546 /-0.978|-2.137429 /-1.227456|
|n19-c913|+0.887501|+28.675 /-5.293|-10.140 /+6.540|-36.389 /-1.209|-2.083459 /-1.261879|
|n19-c2048|+1.994751|-82.942 /-91.829|-15.306 /+11.999|-59.036 /+3.918|-0.924769 /+0.003725|
|n18-c498|+0.880527|+221.769 /+172.111|-17.737 /+19.305|+227.151 /-9.521|-2.536094 /-1.699482|
|n18-c501|+0.882767|+238.252 /+172.194|-11.057 /+4.541|+182.129 /-9.652|-2.506033 /-1.698451|
|n18-c505|+0.885941|+250.626 /+171.922|-18.185 /+11.337|+204.267 /-9.629|-2.535415 /-1.689684|
|n18-c2048|+2.393799|-77.370 /-101.459|-15.899 /+14.987|-58.053 /+3.334|-0.929786 /+0.008787|

**Which use of headroom was most efficient?** For this workload and tested headroom, more cache atn24 wins the recommendation:2890 gives141.245ms less pair median/GiB added versus control, with stronger repeat separation than the lower-memory curve choices and lower memory than the leading mixed n23/2540 trade. Putting more experts onGPU at fixed2048 improves the descriptive curve but costs about.399GiB/layer. Under the target, trading too much cache for more resident layers reduces small-batch hit rates: n20/1318≈61.06%,n19/910≈52.32%,n18/501≈19.12%, versus81.16% for n24/2890. The mixed n23/2540 result remains competitive within observed spread, not repeatably superior. This does not assert that cache is universally better for other prompts, hardware or long-prefill workloads.

## Every clean lifecycle: order, startup, CPU and pressure

Run IDs below omit only the common `tuning8gib-clean-` prefix; all exact IDs and commands are indexed. CPU is sampled lifecycle server CPU time, with final exited-process handle accounting;100%=one logical core (divide by16 for full-host normalization), not attributed per request. Startup is n1/lifecycle, not an independent repeated cold-cache benchmark. Launch→first completion includes loading/readiness, no-touch observations/prefix metadata and the first warmup request.

| Order / run suffix | Group | Ready s | Launch→first s | CPU s / one-core% | Min physical / commit headroom GiB | Samples |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
|1 /n24-c2048-a1|control|7.842|9.443|5.500000 /16.970|1.755943 /4.981956|131|
|2 /n23-c2048-curve|curve|7.318|8.885|1.937500 /6.383|2.597889 /5.031124|123|
|3 /n22-c2048-curve|curve|7.353|8.882|2.953125 /9.756|2.871197 /4.167870|122|
|4 /n21-c2048-curve|curve|7.300|8.789|3.937500 /13.562|3.606293 /4.474869|118|
|5 /n20-c2048-curve|curve|7.236|8.734|3.000000 /10.599|4.152931 /4.458683|115|
|6 /n19-c2048-curve|curve|7.392|8.808|4.859375 /17.836|4.258141 /5.256004|110|
|7 /n18-c2048-curve|curve|7.191|8.655|3.546875 /13.344|4.738209 /5.275192|108|
|8 /n24-c2948-target1-s923|target|8.193|9.777|1.609375 /5.473|2.203587 /5.034023|119|
|9 /n24-c2952-target1-s924|target|7.300|8.891|5.859375 /19.934|3.081688 /5.160881|119|
|10 /n23-c2540-target1-s795|target|7.504|9.035|1.437500 /4.965|3.706348 /4.639305|117|
|11 /n23-c2543-target1-s796|target|7.261|8.894|1.921875 /6.673|3.486240 /4.647575|117|
|12 /n22-c2132-target1-s667|target|7.295|8.832|2.625000 /8.781|4.006645 /4.499752|121|
|13 /n22-c2135-target1-s668|target|7.270|8.755|2.984375 /10.316|3.938271 /4.490913|117|
|14 /n21-c1723-target1-s539|target|7.020|8.506|2.484375 /8.454|4.966278 /5.212292|119|
|15 /n21-c1727-target1-s540|target|7.057|8.529|1.609375 /5.502|4.916397 /5.163246|118|
|16 /n20-c1315-target1-s411|target|7.373|8.830|2.296875 /7.617|4.623222 /4.975700|122|
|17 /n20-c1318-target1-s412|target|7.393|8.870|3.515625 /11.355|5.141689 /5.202168|125|
|18 /n20-c1321-target1-s413|target|7.530|9.028|2.734375 /8.996|5.487831 /5.135403|123|
|19 /n19-c907-target1-s283|target|6.905|8.456|2.031250 /6.659|5.951618 /5.126396|123|
|20 /n19-c910-target1-s284|target|7.012|8.468|3.421875 /11.148|5.989075 /5.124760|124|
|21 /n19-c913-target1-s285|target|7.049|8.565|2.609375 /8.477|5.952480 /5.245163|125|
|22 /n18-c498-target1-s155|target|7.204|8.815|2.156250 /6.337|6.458622 /6.112904|138|
|23 /n18-c501-target1-s156|target|7.310|8.871|2.171875 /6.377|6.471359 /5.179527|138|
|24 /n18-c505-target1-s157|target|7.038|8.639|2.343750 /6.889|6.398823 /5.210358|138|
|25 /n24-c2048-a2|control|8.101|9.686|2.500000 /7.861|3.979019 /6.094753|129|
|26 /n24-c2948-r2|repeat|7.044|8.632|3.328125 /11.636|3.977978 /4.573330|116|
|27 /n23-c2540-r2|repeat|7.233|8.732|2.125000 /7.449|4.518177 /5.537998|116|
|28 /n22-c2132-r2|repeat|7.035|8.551|1.859375 /6.354|4.769268 /5.241379|118|
|29 /n21-c1723-r2|repeat|7.017|8.571|3.843750 /13.015|5.264275 /5.592842|120|
|30 /n24-c3051-band1-s955|band|7.524|9.121|4.343750 /14.977|4.177711 /4.937126|117|
|31 /n24-c3054-band1-s956|band|7.031|8.629|3.203125 /11.210|4.393280 /4.591232|116|
|32 /n23-c2642-band1-s827|band|7.060|8.583|2.578125 /9.147|4.778961 /5.361996|114|
|33 /n23-c2645-band1-s828|band|7.451|8.943|2.171875 /7.467|4.632847 /5.378288|118|
|34 /n22-c2234-band1-s699|band|6.898|8.417|1.359375 /4.738|5.028996 /5.640526|116|
|35 /n22-c2237-band1-s700|band|7.249|8.744|2.203125 /7.747|5.439728 /5.786396|115|
|36 /n21-c1825-band1-s571|band|6.869|8.354|3.000000 /10.336|5.901398 /5.834202|118|
|37 /n21-c1829-band1-s572|band|7.304|8.789|1.562500 /5.342|6.035461 /5.402504|118|
|38 /n24-c2890-refine-r1|repeat|7.456|9.056|2.250000 /7.901|4.666122 /3.883785|115|
|39 /n21-c1723-r3|repeat|6.855|8.313|1.937500 /6.686|5.989655 /5.544334|117|
|40 /n22-c2132-r3|repeat|6.998|8.460|1.640625 /5.667|5.589901 /5.268528|117|
|41 /n22-c2048-r2|repeat|7.032|8.543|2.796875 /9.605|5.455841 /5.738167|118|
|42 /n23-c2642-b2|repeat|7.202|8.718|1.437500 /5.057|4.993023 /4.534294|115|
|43 /n23-c2540-r3|repeat|7.212|8.714|3.328125 /11.436|5.117382 /5.744061|118|
|44 /n23-c2048-r2|repeat|7.184|8.782|4.515625 /14.869|4.977913 /5.867264|123|
|45 /n24-c3051-b2|repeat|7.351|8.932|1.671875 /5.760|4.575863 /4.009022|118|
|46 /n24-c2948-r3|repeat|7.199|8.736|2.265625 /7.875|4.494152 /4.503311|117|
|47 /n24-c2890-refine-r2|repeat|7.155|8.682|4.781250 /16.851|4.632992 /5.291931|115|
|48 /n24-c2048-a3|control|7.112|8.628|4.625000 /14.985|4.890697 /5.541233|125|
|49 /n24-c2945-target2-s922|target|7.176|8.732|2.250000 /7.903|4.877163 /5.225220|115|

Existing constrained RAM was explicitly accepted; no unrelated process was stopped or memory-trimmed. Owned watchdog: below256MiB available physical **or**512MiB commit headroom continuously8s→stop only owned server; a cutoff would be a safety result, not OOM. It never fired. Minimum observed available physical1,885,429,760B and commit headroom4,170,182,656B remain separate global views. Background state/paging/file/shader caches vary; neither lifecycle CPU nor a global paging spike is interpreted as isolated GPU-copy time.

## Bounded quality, negative outcomes and checks

**All2,156 outputs are covered by direct independent semantic review of nine exact input/output pairs across six synthetic sources, plus a final independent ledger coverage audit.** [Full per-ID judgments and clean occurrence ledger](results/tuning8gib/clean/quality-review.json); [final independent confirmation](results/tuning8gib/clean/quality-confirmation.json); [original independent judgments](results/tuning8gib/quality-review-initial.json). Exact strings alone were reused from the initially contaminated cohort; clean occurrence/EOS counts were recomputed and independently read from the final49-run ledgers. Final audit found no missing, duplicate or unreviewed occurrence, exact equality of both final ledgers,49×44 outputs,2,156EOS and all recorded automated predicates true. Its boundary is ledger inspection plus direct semantic audit, **not raw runtime reproduction**.

Names/numbers/negations and corrective contrasts remain preserved in this bounded fixture: Yuna4not3,Mina2not12,Minsu ownership,rednotblue,and joking correction. No visible truncation, pathological repetition, reasoning trace or instruction leakage was observed in the reviewed strings. Preserve real fidelity limits: **192** outputs say within5minutes versus**151** five-minutes-later; **151** add only-two emphasis versus**241** plain-two. Implicit-subject explicitation and minor punctuation caveats remain in per-ID judgments. This is not an unconditional exact-fidelity pass or broad model-quality ranking.

Negative/limited outcomes remain visible: n21–n18/2048 are over-ceiling; all boundary upper neighbors remain; n24/2948 is reclassified band after its repeat; tolerance-band repeat rule fails; target trades n20–n18 do not lead; smaller-memory n23/n22 curve choices are measurably slower than the selected refinement. Clean cohort has zero native failures, watchdog stops or cache-off runs. Contaminated owner-interrupted n21 and all earlier records remain excluded, not relabelled successful measurements. [Probe-development ledger](results/tuning8gib/probe-development.json) retains the early summarizer SyntaxError and cache-slot whitespace parsing omission; explicit cache-correction sidecars preserve recovery from already-retained full native logs. Neither repair weakened workload, counts or acceptance assertions.

Final exercised commands:

```text
<APP_PYTHON> experiments/gemma4_26b_qat/summarize_tuning8gib.py --check
<APP_PYTHON> experiments/gemma4_26b_qat/quality_tuning8gib.py
```

Both exited0 against the final49-run/2,156-output set. [Focused check](results/tuning8gib/clean/focused-check.json) exercised actual collected curve points;44requests/run and2/12/30phases;EOS/nonempty/no-reasoning/no-truncation;all identities andP/C;Self0/Peer1/unused2;15native overlaps/run;effective backend controls absent;actual PID/exact-adapter lifecycle memory coverage;slot/effective-size arithmetic;cache access/upload/destructor evidence;actual Vulkan_Host;49graceful0/no-cutoff and no observed runtime conflicts. Per-run check booleans retain the exact assertions. Independent final semantic/coverage audit is separate. No product build/test/linter/formatter/type-check, app smoke, GUI/audio, physical8GiB or coexistence measurement is claimed.

## Commands, evidence and release

Exact token-separated **native `command` and probe `probe_command` arrays are preserved in every run JSON**, indexed in [evidence-index.json](results/tuning8gib/evidence-index.json). This includes each ephemeral bound localhost port and full effective flags; `<ARTIFACT_ROOT>`, `<CHECKOUT>`, `<MODEL>` and `<APP_PYTHON>` replace private paths only. `<MODEL>` is `<ARTIFACT_ROOT>/models`; `<APP_PYTHON>` is the verified existing Python3.14.7 environment withpsutil/httpx. Model/runtime/driver identity is fixed above. Commands below are the actual initial clean probe and controller/aggregation entrypoints; per-run arrays and the chronological table give all manually sequenced forward/reverse children, not a narrowed workload recipe.

```text
<APP_PYTHON> experiments/gemma4_26b_qat/selfpeer_probes.py --root <ARTIFACT_ROOT> --n-cpu-moe 24 --cache-mib 2048 --adapter luid_0x00000000_0x0000f60b_phys_0 --group control --run-id tuning8gib-clean-n24-c2048-a1 --allow-pressure
<APP_PYTHON> experiments/gemma4_26b_qat/bracket_tuning8gib.py --root <ARTIFACT_ROOT> --adapter luid_0x00000000_0x0000f60b_phys_0 --stage target --placements 24 23 22 21 20 19 18 --search-id target1
<APP_PYTHON> experiments/gemma4_26b_qat/bracket_tuning8gib.py --root <ARTIFACT_ROOT> --adapter luid_0x00000000_0x0000f60b_phys_0 --stage band --placements 24 23 22 21 --search-id band1
<APP_PYTHON> experiments/gemma4_26b_qat/bracket_tuning8gib.py --root <ARTIFACT_ROOT> --adapter luid_0x00000000_0x0000f60b_phys_0 --stage target --placements 24 --search-id target2
<APP_PYTHON> experiments/gemma4_26b_qat/summarize_tuning8gib.py --check
<APP_PYTHON> experiments/gemma4_26b_qat/quality_tuning8gib.py
```

Run/search identities are overwrite-protected; reproduction must use fresh identities/evidence destinations, not overwrite the published observations. New raw path-bearing records remain external at `<ARTIFACT_ROOT>/tuning8gib/runs/<run-id>/`. Durable sanitized per-run JSON,250-ms compact samplesJSONL, slot event/line ledgers and full sanitized native logs are under `results/tuning8gib/clean/`. The native log text hash in JSON refers to the retained original decoded log text; the sanitized public log is a redacted derivative, not claimed byte-identical. Prior raw #245 records were not overwritten. Five conclusively unused imported acquisition/identity/app/attribution/stale-resume files were removed; needed inherited workload/memory/aggregation code was preserved.

[Release observation](results/tuning8gib/clean/release.json): **2026-10-15T23:17:25.809617+00:00**, named model runtime list empty; all49 owned native exits0/graceful, per-run sampler/thread/Windows handles closed. No owned model/app/observer/UDP service remains. Raw evidence directories intentionally remain. Unrelated application/Python tasks were untouched.

## Recommendation

**Recommend the experimental refined native profile:** pinned b11537/Vulkan and fixed verified QAT artifact, `--n-cpu-moe24 --load-mode none --moe-cache-mib2890 --n-gpu-layers99 --swa-full --cache-ram0`, with the unchanged4608/3,f16KV,threads4/4,batch/ubatch512,flash-attn on,fit off,reasoning off and MTP-off controls. Token-separated arguments, not compact prose, are authoritative.

1. **Does a target configuration repeatably beat n24/2048? Yes.**2890’s every-run pair median is below every control run; lower-memory curve alternatives do not match its latency within observed spread. It preserves the bounded reviewed quality set and has a larger target margin than the boundary-filling choices.
2. **Does8.0–8.1GiB buy repeatable improvement over the best target result? No.** The leading band repeat violates the required rule; its pooled apparent gain and serial benefits do not erase that failure or its worse simultaneousp95.
3. **Most efficient use of this workload’s headroom:** more cache atn24, not aggressively swapping cache for additional GPU-resident expert layers. n23/mixed remains an equivalent-spread alternative with less host working set, but costs more dedicated memory than2890; serial-tail tradeoffs are disclosed above.

The recommendation’s peak **7.940708GiB** is a sampled server-process lower bound measured on a24-GiB-class7900XTX. It does **not** establish fit/performance on a physical8-GiB card, leave guaranteed desktop/VRChat headroom, or validate application/GUI/audio/coexistence behavior. [Machine-readable recommendation](results/tuning8gib/clean/recommendation.json). **The project owner must still record adopt-refined or keep-n24/2048; no product/settings adoption is performed by this evidence.**
