# Issue245 — n24 non-mmap,2048/3072MiB cache, two occupied Self/Peer slots

## Execution outcome and authority

**Four fresh native-server lifecycles completed44/44 requests each,176/176 total; all EOS, graceful exit0, no safety cutoff or native failure.** Sequence:2048A1→3072B1→3072B2→2048A2, only one owned server at a time. This report collects evidence only. **Performance judgment, configuration selection and adoption decisions remain with the Director.** No recommendation or historical cache-effect inference is made.

Experiment baseline `f052922ae80e26b7365cc66f63f8081e4941fd87`; unchanged product baseline `7a8151be6114a950f062e2430bd36f899d57e4de`. Same verified QAT model14,249,047,104B/SHA256 `a7c5bc715f5ff8e99a3e8901ce7d2b42b402c669bf24f7c5250747633d0f5891`, pinned b11537/`8e2d31e0eb658d0382d3106a61369dff81f0c080`, Windows11/9800X3D8C16T/32GiB/RX7900XTX24GiB-class. Existing `results/identity.json` retains model/runtime/host identity. Each new record adds executable SHA256 and exact sanitized native command.

Native-only, **not actual-app/Self/Peer orchestration integration, GUI, audio E2E or actual VRChat coexistence**. No product source/settings, MTP, model/backend/driver change, mmap arm, other cache size or further run was added. Historical records/reports unchanged.

## Fixed workload and timing boundaries

Common native controls: `--load-mode none --n-gpu-layers99 --n-cpu-moe24 --swa-full --cache-ram0`, Vulkan0, context4608/parallel3/effective1536 each, threads4/4,batch/ubatch512,f16KV,flash-attn on,fit off,reasoning off/budget0,warmup,prio-1. Exact token-separated arguments are in run JSON, not the compact prose above. Only cache2048/3072 changes.

Exactly two occupied slots:
- **Self:** Korean→English, pinned slot0, original CASES1/5/9.
- **Peer:** English→Korean, pinned slot1, CASES0/4/8.
- Slot2 stays unused; total server context/parallelism is not reduced.

Per run:2first-prefix warmups(one/channel,temp0);6serial pair rounds(12requests, alternating channel-first order);15simultaneous pair rounds(30requests, common three-party barrier release). Original real rendered prompt, fixed five synthetic history lines, request routine, temp.6/seed42/max96/EOS retained. Cases cycle every three rounds. Warmups are separate, not included in occupied-phase pooled request timing.

Per-request **API** prefill/decode ms and native-reported TPS belong to that original request. P=actual uncached prompt evaluations,C=prior prompt cache hits,O=API output. No cached full-prompt division, output-count substitution for traced forwards or concurrent native timing-to-request join. `decode_eval_tokens` distributions are explicitly unavailable in these API summaries; raw native printed timings remain external.

HTTP wall and streaming TTFT start **before apply-template** and end at response completion/first nonempty content respectively. Simultaneous pair makespan starts at common barrier release before both request routines and ends when both complete; serial reference starts before its first routine and ends after its second. These include client scheduling/local HTTP and are not pure GPU-kernel time or app E2E.

## Individual and pooled pair observations

Seconds; median/p95. Every simultaneous row n15pairs, every serial row n6pairs. Pooled rows use30simultaneous/12serial **original pair observations**, not percentiles of percentiles.

| Cache/run | Simultaneous release→bothcomplete | Serial pair makespan | Simultaneous client start skew med |
| --- | ---: | ---: | ---: |
|2048A1|.817728 /1.012320|1.067072 /1.143300|.000174|
|3072B1|.715896 /.875468|.920623 /.973375|.000175|
|3072B2|.729206 /.865248|.924015 /1.040580|.000171|
|2048A2|.811748 /1.008336|1.048039 /1.135783|.000180|
|2048 pooled|.815606 /1.008336|1.061025 /1.143300|.000178|
|3072 pooled|.722551 /.870050|.924015 /1.040580|.000174|

**Server-side occupancy evidence:** all15 simultaneous rounds in **each** run have positive native launch→release interval overlap for slot0/slot1. Each slot has22complete task intervals/run; no slot2 task or unreleased task. Pooled native active-overlap median/p95:2048 .589548/.643475s;3072 .511517/.587827s, n30each. Serial native overlap is zero n12/cache. These prove concurrent server-active requests, not simultaneous GPU kernel execution. Per-slot ordered pairing uses exactly one request/channel/round and22complete native intervals/slot; it does not join printed native timing counters to API requests. Public `.slots.json` retains native task IDs, timestamps and raw-log line references; client overlap/skew and common-release records remain separate.

## Per-request native API timing

Pooled simultaneous phase, **n30 per channel/cache**, medians below. [Every run, all phases/channels and pooled timing distributions](SELFPEER_TIMINGS.md) includes first-prefix and serial reference; [summary.json](results/selfpeer/summary.json) retains n/median/p95/min/max for every observable, including P/C/O, wall and TTFT.

| Cache/channel | Prefill ms /TPS | Decode ms /TPS | P /C /O med | HTTP wall s med/p95 | TTFT s med/p95 |
| --- | ---: | ---: | --- | ---: | ---: |
|2048 Self|195.229 /130.67|486.535 /31.89|26 /882 /16|.761104 /.862861|.212577 /.399873|
|2048 Peer|191.962 /134.89|564.668 /31.05|26 /884 /15|.752435 /1.008063|.205886 /.400781|
|3072 Self|175.181 /154.79|423.669 /34.34|26 /882 /16|.687318 /.769597|.192559 /.383256|
|3072 Peer|167.152 /149.03|473.926 /35.80|26 /884 /15|.699283 /.869781|.194156 /.339069|

Pooled both-channel simultaneous requests n60/cache:2048 prefill192.590ms/132.83TPS,decode490.670ms/31.07TPS;3072 prefill169.369ms/149.06TPS,decode430.230ms/34.66TPS. These TPS are per-request rates, not aggregate throughput from summing concurrent rates.

Exact prompt/workload identities match44/44 across all four runs. Actual P/C counts also match at every corresponding request. One output-count difference is preserved:3072B2 Peer simultaneous round6 generated21tokens versus reference2048A1's22(P28/C884 unchanged), selecting the equivalent `2장 샀어요` versus `2장만 샀어요` wording. `token_count_differences` records it; equality was not forced.

| Run | Ready s n1 | Launch→first completion s n1 | CPU seconds /one-core% |
| --- | ---: | ---: | ---: |
|2048A1|8.624|10.176|1.797 /5.76|
|3072B1|7.778|9.287|2.016 /7.17|
|3072B2|7.835|9.276|1.813 /6.44|
|2048A2|7.171|8.712|1.563 /5.28|

First-prefix API durations/TPS/wall are separate in the timing report. Startup n1/run is not a repeated-startup distribution; loading precedes ready and preparation precedes first submit. CPU100%=one logical core; divide by16 for whole-host fraction. CPU cost covers sampled full lifecycle, not an attributed per-request CPU duration.

## Resource observations and current-load limits

All rows below are **lifecycle250ms-sampled peaks**; GB is decimal. PID-specific GPU counters use exact adapter `luid_0x00000000_0x0000f60b_phys_0`, with actual counter instances retained. Working set, private commitment and GPU accounting are different, overlapping views: **do not add them; private commitment is not physical RAM**.

| Run | PID dedicated bytes /GB /GiB | PID sharedGB | WS /privatecommitGB | Whole-device dedicated/sharedGB | Preflight free /minfreeGB | Commit headroom minGB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
|2048A1|7,649,181,696 /7.649 /7.124|10.972|11.281 /18.924|11.819 /11.342|13.639 /2.348|6.867|
|3072B1|8,775,426,048 /8.775 /8.173|10.930|11.266 /20.036|12.860 /11.330|14.257 /1.081|4.784|
|3072B2|8,725,315,584 /8.725 /8.126|10.930|11.253 /19.995|12.961 /11.331|13.509 /2.440|4.810|
|2048A2|7,649,181,696 /7.649 /7.124|10.941|11.264 /18.899|11.776 /11.342|15.650 /4.147|6.080|

Active serial+simultaneous phases, separate sampled peaks (GB):

| Run | PID dedicated/shared | WS/privatecommit | Whole-device dedicated |
| --- | ---: | ---: | ---: |
|2048A1|7.649 /10.830|11.104 /18.805|11.804|
|3072B1|8.725 /10.830|11.107 /19.877|12.825|
|3072B2|8.725 /10.830|11.159 /19.931|12.961|
|2048A2|7.649 /10.830|11.104 /18.802|11.773|

All raw sampled bytes and separate load/first-prefix/serial/simultaneous/shutdown distributions are in JSON. The3072B1 lifecycle dedicated peak8.775GB differs from active8.725GB; retain both instead of substituting a selected steady snapshot. Peaks can occur at different times and are sampled lower bounds, not exact lifetime high-water marks.

Actual model buffers in every run: **Vulkan_Host10197.01MiB** and Vulkan0 model3772.85MiB. Host-allocation log lines retained; large shared accounting is consistent with intentional GPU-visible host weights and is **not proof all shared memory is uncontrolled VRAM spill**. No quantified per-upload asynchronous/copy contribution was measured.

Native destructor small-batch hit rates, in execution order:70.51/82.89/82.85/70.27%; corresponding hits/misses83,745/35,020;98,158/20,259;98,349/20,356;83,671/35,399. Logical small-batch uploads111,728.85/64,634.92/64,944.39/112,938.02MiB. Large-batch buckets and cached-copy counts are retained per run. Counters include warmup/full lifecycle, not phase-only physical PCIe transfers or bandwidth.

Preflight current RAM was explicitly accepted; no unrelated process was stopped or forced memory trim/pin/system flag applied. Inherited source-known GGML controls were captured and removed only for owned child environments; all effective allowlist values absent. Existing watchdog remained <256MiB physical available OR512MiB commit headroom continuously8s, owner-only termination. It never fired. Preflight found no conflicting unowned model runtime.

Global pages-input/s p95 in execution order:645080/689679/651637/700802. These and global commit/headroom cannot be attributed exclusively to the model; process page faults include soft faults. Background, paging/file/shader cache and OS state vary. No clean-kernel/transfer causality, long-running coexistence guarantee or physical8GB-card result is asserted. Earlier n24/cache measurements use different workloads/conditions and are not a historical causal control here.

## Quality, verification, release and reproduction

All176 outputs covered by manual review of **9unique input-output pairs across6sources** ([pairs](results/selfpeer/quality-unique.json), [review](results/selfpeer/quality-review.json)). Observed names/numbers/negations preserve Yuna4not3,Mina2not12,Minsu ownership,rednotblue and joking correction. No observed cache-specific material collapse, reasoning, repetition or truncation;176EOS. Preserve caveats:15`within5minutes` versus13`5minutes later` outputs, and15`only2` emphasis variants. This small self-review is not an independent quality/model ranking. Automated predicates unchanged.

Final fresh-interpreter `summarize_selfpeer.py --check` exited0:4runs,176completed,9unique outputs,1disclosed output-count difference. Assertions exercised exact workload/prompt match,44requests/run with2/12/30phase counts,176EOS/nonempty/no-reasoning/not-truncated,pinned slots0/1 with no native slot2 usage,15positive server overlaps/run,actual Vulkan_Host,clean effective allowlist and four graceful0/no-cutoff lifecycles. The final workload/lifecycle helper was exercised by all four actual runs; final aggregation/helper version was exercised offline against all retained logs. No permanent test framework or product build/test/linter added or claimed.

[Durable release](results/selfpeer/release.json): all four owned native exits0/graceful; final model-runtime enumeration empty, observer handles closed. No owned model/app/observer/UDP job remains. Raw logs/full samples/results external under `<ARTIFACT_ROOT>/selfpeer/runs/<run-id>/`; sanitized public run JSON, slot event ledgers and compact samplesJSONL under `results/selfpeer/`. Historical records untouched; no Git/GitHub mutation.

Actual ordered commands (same existing app Python and verified artifact root):

```text
<APP_PYTHON> experiments/gemma4_26b_qat/selfpeer_probes.py --root <ARTIFACT_ROOT> --cache-mib 2048 --run-id selfpeer-cache2048-a1 --allow-pressure
<APP_PYTHON> experiments/gemma4_26b_qat/selfpeer_probes.py --root <ARTIFACT_ROOT> --cache-mib 3072 --run-id selfpeer-cache3072-b1 --allow-pressure
<APP_PYTHON> experiments/gemma4_26b_qat/selfpeer_probes.py --root <ARTIFACT_ROOT> --cache-mib 3072 --run-id selfpeer-cache3072-b2 --allow-pressure
<APP_PYTHON> experiments/gemma4_26b_qat/selfpeer_probes.py --root <ARTIFACT_ROOT> --cache-mib 2048 --run-id selfpeer-cache2048-a2 --allow-pressure
<APP_PYTHON> experiments/gemma4_26b_qat/summarize_selfpeer.py --root <ARTIFACT_ROOT> --check
```

Public exact native commands preserve token arrays. Existing run identities are overwrite-protected. Director owns review, judgment, commit and authorized publication; no adoption decision is embedded in this evidence handoff.
