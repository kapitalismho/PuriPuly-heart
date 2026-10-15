# Issue245 — n18/n24 with3072MiB cache, mmap versus non-mmap

## Outcome and scope

All four required native-server arms completed **43/43 translations each:172/172 total**, all EOS, graceful exit0, no owned safety cutoff or native failure. No app, UDP sink, profiler, new runtime/model/backend, extra sweep or control bracket was run. Product settings, historical reports and recommendation are unchanged. This is **native-only evidence, not actual-app or managed26B integration/adoption approval**.

**Non-mmap n18 is faster and uses less observed process working set than non-mmap n24, at about2.57GB more dedicated VRAM.** Non-mmap improves these workloads versus mmap at either placement; it raises private commitment and shared-GPU accounting. Near20GB private commitment is **not20GB physical RAM**.

Experiment baseline `ee4620b5ce96b56202c17a06bf979852cb6d6cbd`; unchanged product baseline `7a8151be6114a950f062e2430bd36f899d57e4de`. Reused verified fixed QAT artifact14,249,047,104B/SHA256 `a7c5bc715f5ff8e99a3e8901ce7d2b42b402c669bf24f7c5250747633d0f5891`, b11537/`8e2d31e0eb658d0382d3106a61369dff81f0c080`, Windows11/9800X3D8C16T/32GiB/RX7900XTX24GiB-class. Original identity manifest remains `results/identity.json`; per-run executable hashes, exact commands, adapter/counter instances and effective controls are retained in new records.

Common arguments remain ngl99, cache3072, SWA-full/cache-ram0, Vulkan0, ctx4608/parallel3/effective1536 per slot,4/4threads,512/512batch, f16KV, flash-attn on, fit off, reasoning off/budget0, warmup, seed42/limit96 and original prompt/sampling. Original43-request fixture:3first-prefix,5hot,20pair-alternation,15three-slot;12sources. Offline assertions establish all43 ordered request identities/prompt hashes and **actual evaluated/cache token counts match across all four arms**. No narrowed workload.

## Timing and resources

[All-phase timing report](PLACEMENT3072_TIMINGS.md) includes every option's prefill/decode durations and TPS, counts and tails. [JSON summary](results/placement3072/summary.json) retains n/median/p95/min/max, actual P/cache/output/printed-native counts, HTTP wall/streaming TTFT, per-phase resources and destructor telemetry. Native TPS is retained, not recomputed from full cached prompts or substituted API output counts. Concurrent native lines are associated only at phase barriers, never fabricated per-request joins.

Hot same-pair n5 medians:

| Placement/mode | Prefill ms /TPS | Decode ms /TPS | Dedicated bytes /GB /GiB | WS /privatecommit GB | SharedGPU GB | Free minimum GB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| n18 mmap |189.446 /125.54|241.744 /56.25|11,280,859,136 /11.281 /10.506|14.560 /12.185|.615|.577|
| n18 none |92.988 /273.64|235.054 /59.56|11,296,030,720 /11.296 /10.520|8.925 /20.022|8.545|6.298|
| n24 mmap |332.448 /75.32|355.898 /39.34|8,705,212,416 /8.705 /8.107|14.439 /9.666|.614|1.227|
| n24 none |133.590 /191.95|278.047 /50.35|8,725,446,656 /8.725 /8.126|11.296 /20.003|10.978|3.608|

Memory columns are lifecycle sampled peaks, not just hot-phase snapshots; GB is decimal. Dedicated is PID-specific on exact adapter `luid_0x00000000_0x0000f60b_phys_0`. None of these new arms meets the preferred8decimalGB target, which is not an eligibility gate.

| Placement/mode | Cold ready s n1 | First HTTP s | Launch→first complete s | Whole-device dedicated peak GB | Commit headroom min GB | Global pages-input/s p95 | CPU seconds /one-core% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| n18 mmap |7.064|8.434|17.405|15.297|13.001|159618|5.172 /11.33|
| n18 none |7.240|1.178|8.691|15.328|7.243|678801|1.844 /7.10|
| n24 mmap |4.720|5.425|11.940|12.712|17.367|60718|7.656 /14.67|
| n24 none |7.451|1.373|9.130|12.719|8.068|701536|2.266 /7.37|

Cold ready, HTTP streaming wall (including apply-template), harness preparation and launch→first completion are separate observations, **not app or audio E2E**. Startup n1 is not a distribution. CPU100%=one logical core; divide by16 for whole-host fraction. All counters and whole-device shared accounting remain separately available in JSON; do not add overlapping GPU/WS/private counters.

Actual host allocation switches from `CPU_Mapped8299.45MiB` to **`Vulkan_Host7746.76MiB`** at n18; n24 switches `CPU_Mapped10935.02MiB` to **`Vulkan_Host10197.01MiB`**. No pinned-allocation failure warning observed. Large none-mode shared accounting is consistent with deliberately GPU-visible host buffers; **not all shared memory is uncontrolled spill**. Source-backed mode-gain interpretation remains the existing FOLLOWUP.md explanation; there is no new quantified per-upload copy/synchronization proof.

Native cache executes in all four lifecycles: small-batch hit rates n18 mmap/none90.54/90.30%, n24 mmap/none82.58/82.43%. Logical small-batch uploads26,579/27,323MiB and65,614/66,170MiB respectively. Large buckets and existing-cache copy counts are retained in each analysis/summary. These include warmup and full lifecycle; logical upload MiB is not physical PCIe bandwidth or phase-specific transfer attribution.

## Current-load qualification, quality and stopping

Counterbalanced sequence was n18 mmap→none, then n24 none→mmap. Preflight rejects conflicting unowned model runtimes; none was present. Source-known GGML allowlist was captured inherited/effective and removed only in owned children; effective values all absent. Existing constrained RAM was explicitly authorized as realistic background load; no unrelated process was stopped, no system flags/pinning/trim changed. Owned watchdog retained <256MiB available physical **or**512MiB commit headroom continuously8s. No arm triggered it. No native-OOM claim.

Preflight available physical RAM was approximately14.011,15.475,15.190,15.507GB in execution order. Later n24 mmap had more headroom yet remained slower than none; no extra bracket was needed for the bounded conclusion. Background/OS/file-cache/driver warmth are uncontrolled; especially n18 mmap's .577GB minimum is meaningful pressure, not an isolated transfer benchmark. Global paging is not attributable exclusively to the model and page faults include soft faults.250ms sampled peaks are lower bounds, not lifetime high-water guarantees. No physical8GB card or actual VRChat render/frame-time test.

All172 completed outputs and18unique input-output pairs across12sources were inspected ([preserved pairs](results/placement3072/quality-unique.json)). Negations, numeric contrasts12vs2/3vs4/twice-vs-three/1500, Minsu/Tanaka, red-vs-blue and appointment-not-cancelled meanings remain. No observed configuration-specific material collapse, truncation, reasoning or repetition. Preserve known caveats: 小林→Kobayashi is ambiguous name reading, not an exact-name pass; `within5minutes` versus `in5minutes` changes nuance; Korean `2장만` adds emphasis. Automated digit-only false flags on correct `twice/three` remain unchanged. This small self-reviewed set is not an independent quality ranking or proof of superiority over E4B.

Prior n24/cache2048 observations in FOLLOWUP.md are **historical descriptive evidence only**, not a matched3072-vs2048 cache-effect estimate under these conditions. No new2GiB control was added. Original app recommendation/owner decision remains unchanged; none-mode actual-app behavior was not exercised.

## Reproduction, verification and release

Runner extension adds only two placement profiles and optional results/baseline arguments. New evidence is isolated under `results/placement3072/`; old files remain untouched. Raw logs, full samples and results stay external at `<ARTIFACT_ROOT>/followup/runs/<run-id>/`. Each public run JSON preserves sanitized exact native command/PID/start/port, inputs and sampling; compact samples are public JSONL.

Use app Python with the existing followup_probes.py, common arguments:

```text
--root <ARTIFACT_ROOT> --results experiments/gemma4_26b_qat/results/placement3072 --experiment-baseline ee4620b5ce96b56202c17a06bf979852cb6d6cbd --allow-pressure
```

Actual ordered placement/load-mode/run-id triples:

```text
n18cache3072 mmap followup-placement3072-n18-mmap-a1
n18cache3072 none followup-placement3072-n18-read-b1
n24cache3072 none followup-placement3072-n24-read-b1
n24cache3072 mmap followup-placement3072-n24-mmap-a1
```

Pass each as `--placement`, `--load-mode`, `--run-id`; existing identity overwrite is refused. This records commands, not instructions to repeat failures. Aggregation actually exercised:

```text
<APP_PYTHON> experiments/gemma4_26b_qat/summarize_placement3072.py --results experiments/gemma4_26b_qat/results/placement3072
<APP_PYTHON> experiments/gemma4_26b_qat/summarize_timings.py --results experiments/gemma4_26b_qat/results/placement3072 --raw-root <ARTIFACT_ROOT>/followup --markdown experiments/gemma4_26b_qat/PLACEMENT3072_TIMINGS.md
```

Both exited0:4runs/172completed/18unique outputs,4native timing identities/172timed requests/0app runs. Focused fresh-interpreter offline assertions passed four43-request graceful0 lifecycles,172EOS, exact ordered request/prompt/evaluated/cache equality and all effective allowlist values absent ([check record](results/placement3072/focused-check.json)). No helper changes followed actual runs. No product build/tests/linter claimed; Director owns final parsing, independent review and publishing.

[Durable release](results/placement3072/release.json): all four owned native processes exited gracefully0; final enumeration has **no model-runtime processes**, observer handles closed by existing runner/sampler. No owned GPU job/server, app or sink remains. No Git/GitHub mutation performed. Retain shipped E4B pending separately authorized integration/rollout.
