# Corrected n23 placement — native MTP experiment

**Decision: do not enable the tested Q4_0 max-draft-4 MTP profiles for this fixed short Self/Peer workload at the owner-selected n23 placement.** Both requested cache arms completed two fresh counterbalanced lifecycles; each MTP4 run median was slower than every same-cache off run median. Max2 made measurable progress, but its retained first exploratory run did not qualify. This is an experimental result, not product/settings adoption.

## Authority and separate cohort

- Owner corrected the placement to **n23**, then authorized actual n23 execution: [scope correction](https://github.com/kapitalismho/PuriPuly-heart/issues/254#issuecomment-6207640516). Baseline: `270c0a4b7dd9a8aa85612695f4ad02596452035d`.
- Cache2540MiB is the unchanged **experiment baseline**, not an adopted cache policy. Cache2520MiB is the independently measured reduced arm. The owner selected placement; this report does not select product settings on their behalf.
- Prior [placement/cache report](TUNING8GIB.md), [n24 MTP report](MTP.md), and their evidence remain unchanged. No prior n24 latency, memory, repeats, or contamination records are pooled or relabeled as n23 evidence. Only unchanged fixture definitions, pinned-source semantics, and exact-string semantic judgments are reused.
- All new evidence is in [results/mtp_n23/evidence-index.json](results/mtp_n23/evidence-index.json). [MTP_N23_TIMINGS.md](MTP_N23_TIMINGS.md) contains every run/configuration/phase/channel timing table; [summary.json](results/mtp_n23/summary.json) retains original-observation distributions and every count difference.

## Artifacts, host and invariant controls

- Target: `gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf`,14,249,047,104bytes, SHA256 `a7c5bc715f5ff8e99a3e8901ce7d2b42b402c669bf24f7c5250747633d0f5891`.
- Q4_0 drafter: `MTP/mtp-gemma-4-26B-A4B-it-Q4_0.gguf`,251,939,328bytes, SHA256 `7272d97595f0d4c74bd7b623492b7dbdaafd8b7c72f329a8270ba4eca68f768a`. Both use `unsloth/gemma-4-26B-A4B-it-qat-GGUF` revision `7b92b5b28818151e8669af2e45e88d6086f490dd`.
- Runtime: b11537 / `8e2d31e0eb658d0382d3106a61369dff81f0c080`, Vulkan; executable SHA256 `7b36a3bc3a24e552d0fcc18441893bc198abaeb0fa77d653241d762077c995c6`. Fresh full streaming hashes of both models, executable identity and actual `--version` succeeded in [prerequisites.json](results/mtp_n23/prerequisites.json) at2026-10-16T03:50:10.650491+00:00. Existing verified files reused; no new download/runtime/driver change.
- Host inherited: Windows11, Ryzen7 9800X3D,32GiB RAM, RX7900XTX24GiB-class; driver32.0.31041.1004/Vulkan26.8.1 from the unchanged prerequisite pin. No new driver inspection or physical8GiB-card test is claimed.
- Every actual native command uses `--n-cpu-moe 23 --n-gpu-layers 99 --load-mode none --swa-full --cache-ram 0`, context4608/parallel3/1536 per slot, threads4/4,batch/ubatch512,f16KV,flash-attention on,fit off,reasoning off. The only arm differences are requested cache and MTP flags. Requested n23 is confirmed separately by actual command arrays and the native23-layer cache group in all nine runs.
- Original44-request workload is unchanged:2 first-prefix requests,6 serial pair rounds,15 barrier-released simultaneous pair rounds; Self Korean→English slot0,Peer English→Korean slot1,slot2 unused; original prompt/history,temperature0.6,seed42,max96. All135 simultaneous rounds have positive native slot overlap. [workload-inheritance.json](results/mtp_n23/workload-inheritance.json) compares44 fixture identities/hashes/P/C with the old fixture solely to prove inheritance, not performance.

## Design, measured reduced cache and every lifecycle

Order: **A1 B1 C1 D1 D2 C2 B2 A2**, then bounded **E1**. A=2540/off,B=2540/MTP4,C=2520/MTP4,D=2520/off,E=2540/MTP2. Only one owned model server was loaded at a time. Every primary arm has two fresh full lifecycles. C1 was the initial reduced-cache observation and remains the first final-arm run; it was not replaced.

The reduced-cache starting point followed measured headroom, **not the240.27MiB drafter file-size proxy**. A1 peaked8,588,566,528B (1.305MiB below target); B1 peaked8,590,745,600B (0.773MiB above target, band); C1 at2520 peaked8,567,017,472B (21.855MiB below target). Keep2520 was recorded before remaining runs in [reduced-cache-choice.json](results/mtp_n23/reduced-cache-choice.json). C2 raised the reduced-arm maximum by0.625MiB, leaving21.230MiB target margin. No adaptive cache change or hidden preliminary run occurred.

| Run (full profile prefix `mtp-n23-`) | Started UTC2026-10-16 | Pair median / p95 ms | Serial median / p95 ms | PID dedicated peak bytes | P/C/O |
| --- | --- | ---: | ---: | ---: | --- |
|[c2540-off-a1](results/mtp_n23/mtp-n23-c2540-off-a1.json)|03:51:37.148141+00:00|814.871/1010.144|1086.351/1180.092|8,588,566,528|2867/37138/695|
|[c2540-d4-b1](results/mtp_n23/mtp-n23-c2540-d4-b1.json)|03:52:19.322192+00:00|927.620/1216.620|1091.830/1342.301|8,590,745,600|2867/37138/693|
|[c2520-d4-c1](results/mtp_n23/mtp-n23-c2520-d4-c1.json)|03:52:56.096907+00:00|918.707/1347.052|1077.119/1406.988|8,567,017,472|2867/37138/696|
|[c2520-off-d1](results/mtp_n23/mtp-n23-c2520-off-d1.json)|03:54:13.743541+00:00|828.641/980.127|1013.673/1146.327|8,564,973,568|2867/37138/695|
|[c2520-off-d2](results/mtp_n23/mtp-n23-c2520-off-d2.json)|03:54:46.022454+00:00|843.702/957.231|1027.462/1152.217|8,564,973,568|2867/37138/695|
|[c2520-d4-c2](results/mtp_n23/mtp-n23-c2520-d4-c2.json)|03:55:18.011613+00:00|928.780/1298.218|1062.207/1320.881|8,567,672,832|2867/37138/696|
|[c2540-d4-b2](results/mtp_n23/mtp-n23-c2540-d4-b2.json)|03:55:53.987872+00:00|923.843/1281.828|1102.439/1339.591|8,591,265,792|2867/37138/696|
|[c2540-off-a2](results/mtp_n23/mtp-n23-c2540-off-a2.json)|03:56:30.019657+00:00|799.003/976.985|1016.658/1128.720|8,588,566,528|2867/37138/695|
|[c2540-d2-e1](results/mtp_n23/mtp-n23-c2540-d2-e1.json)|03:57:39.178094+00:00|867.011/1116.409|992.642/1225.604|8,589,692,928|2867/37138/696|

P=actual newly evaluated prompt tokens,C=prior prompt-KV hits,O=native generated results includingEOS. All fresh request identities, prompt/system hashes, sampling, P and C match A1. **All20 O-count differences** remain in summary.json; output counts/strings were not forced to match. Each ordinary run hasP2867/C37138; per-configuration totals:

| Configuration | Runs | Actual P / C / O |
| --- | ---: | --- |
|c2540-off|2|5734/74276/1390|
|c2540-mtp4|2|5734/74276/1389|
|c2520-off|2|5734/74276/1390|
|c2520-mtp4|2|5734/74276/1392|
|c2540-mtp2|1|2867/37138/696|

## Matched performance and repeat rule

Pair latency is common release to both requests complete, not an average of individual HTTP times. Percentiles pool original pair observations, not run percentiles; nearest-rank p95 and median conventions are inherited. Rates are **per request**, not summed concurrent throughput. Both channels, first-prefix, serial and simultaneous distributions—including native prefill/decode durations and rates, P/C/O, HTTP wall and TTFT—are in the timing companion and summary. HTTP/TTFT include apply-template; native timing is not pure GPU-kernel or app/audio E2E time.

| n23 cache / MTP | Runs | Pair median / p95 ms | Serial median / p95 ms | Simultaneous prefill ms / TPS med | Simultaneous decode ms / TPS med |
| --- | ---: | ---: | ---: | ---: | ---: |
|c2540-off|2|800.871/998.190|1058.208/1180.092|186.195/138.838|516.567/30.287|
|c2540-mtp4|2|925.732/1269.234|1091.830/1342.301|191.414/140.655|544.115/25.527|
|c2520-off|2|837.858/962.094|1013.848/1152.217|180.435/145.364|542.272/28.081|
|c2520-mtp4|2|925.443/1298.218|1074.149/1406.988|195.073/140.687|626.426/25.238|
|c2540-mtp2|1|867.011/1116.409|992.642/1225.604|192.524/138.497|512.987/30.355|

- **2540/MTP4:** run medians927.620/923.843ms exceed both2540/off medians814.871/799.003ms. Pooled pair median worsens124.861ms (15.59%), p95 worsens271.043ms; serial median/p95 worsen33.622/162.209ms. Its all-run memory class is band; it fails the band rule as well as the matched-off repeat criterion.
- **2520/MTP4:** run medians918.707/928.780ms exceed both2520/off medians828.641/843.702ms. Pooled pair median worsens87.585ms (10.45%), p95 worsens336.124ms; serial median/p95 worsen60.301/254.770ms. It fits the measured target, but does not establish an MTP benefit.
- **Cache cost isolated by off controls:**2520/off saves23,592,960B=22.5MiB of PID dedicated accounting versus2540/off, while its pair median worsens36.987ms. Every2540/off run median is below every2520/off median. The tradeoffs favor2520 on pair p95 by36.097ms and serial median/p95 by44.360/27.875ms; these are not hidden or attributed to MTP.
- **Bounded max2 E1:**867.011ms is52.140ms above the slower2540/off repeat and68.008ms above the faster one; off median spread is15.868ms. Its first run already fails the strict all-candidate-below-all-matched-control criterion, so no rescue repeat was added. Pooled-versus-off pair median/p95 worsen66.140/118.219ms. Positive progress is retained: observed serial median improves65.566ms and acceptance rises67.668%; serial p95 worsens45.512ms. This is **n1 negative exploratory evidence**, not repeated or universal max2 inferiority.
- Reducedmax2 was not measured. E1 also exceeds both reduced-off medians, while the measured max4 cache change shifts pooled pair median only0.289ms within overlapping repeat ranges. That supplies no observed decision-changing signal for expanding the bounded refinement. This is an explicit scope rationale, **not a proven prediction of unmeasured reducedmax2 performance**. See [max2-plan.json](results/mtp_n23/max2-plan.json) and [max2-decision.json](results/mtp_n23/max2-decision.json).

## Native MTP execution and token-time semantics

All five MTP-enabled lifecycles retain native `gemma4-assistant`,5/5 drafter layers offloaded toVulkan0, positive225.21MiB Vulkan drafter weights, requested `n_max=4` or2, and target-KV sharing. Three native speculative slot contexts initialize; the workload still uses only slots0/1. Per-request proposals and accepted prefixes prove workload execution, not merely loading a drafter. Each final native cumulative `#gen/#acc` pair agrees with the summed per-request API counters; repeated cumulative snapshots are **never summed**.

| Configuration | API proposed / accepted prefix tokens | Weighted acceptance |
| --- | ---: | ---: |
|c2540-off|unemitted/unemitted|unemitted (off)|
|c2540-mtp4|1636/924|56.479%|
|c2520-off|unemitted/unemitted|unemitted (off)|
|c2520-mtp4|1660/918|55.301%|
|c2540-mtp2|566/383|67.668%|

**Off is a true no-drafter control:** its actual commands have no `--model-draft`, no `--spec-type`, and no `--spec-draft-n-max`; `/slots` is non-speculative, drafter initialization/counters are absent. No append-`none` override is used. Every record captures empty effective `LLAMA_*` environment, with inherited known GGML backend overrides removed. HF credential variables are removed from the child environment. This is a bounded environment policy, not a claim to scrub arbitrary unrelated secrets.

Pinned-source interpretation is reused from [runtime-research.json](results/mtp/runtime-research.json), because runtime/model identities are unchanged; none of its n24 measurements is used as n23 outcome evidence:

- Use `/completion.timings.prompt_n/cache_n/predicted_n` for P/C/O, not top-level `tokens_evaluated/tokens_cached`. Top-level values refer to full prompt/final retainedKV, not newly processed prompt/prior hits.
- Native API decodeTPS=`1000*(predicted_n-1)/predicted_ms`. Generated results include first token/EOS; elapsed includes drafting, verification, synchronization and scheduling. This exact formula was checked for every request; no fabricated proposal-based surrogate TPS is needed.
- `draft_n_accepted` counts accepted prefixes excluding bonus target tokens and may include an accepted suffix beyondEOS. It is neither emitted-output count nor a valid deliveredTPS numerator.
- libllama destructor evaluation counters may account verification batches differently; those context rates are not substituted for API outputTPS. API acceptance and cache-destructor counts answer different questions.

## Cache allocation and actual use

Every native log proves one **23-layer** Q4_0 group with128 experts per layer (2944 host experts,active8). Requested2540 gives795slots/effective2,662,944,768B=2539.582031MiB (native2539.58),27.004% slot coverage. Requested2520 gives788slots/effective2,639,526,912B=2517.249023MiB (native2517.25),26.766% coverage. The effective difference is7slots=22.333008MiB, not the literal20MiB request difference.

Pinned-source allocation is `(slots+1)*3,345,408B`, with one padding plane; `slots=floor(requestMiB*2^20/3,345,408)-1`. Native rounded effective MiB and group lines independently confirm each request. No cache-off/initialization failure occurred. Prompt-cache-disabled warnings refer to `--cache-ram0`, not disabled expert cache.

| Run suffix | <=8 hits/misses ; uploaded MiB | >8 hits/misses ; uploaded MiB | Large-batch cache copies / MiB |
| --- | ---: | ---: | ---: |
|c2540-off-a1|88925/24148 ;77042.50|8197/27586 ;88011.19|8834/14092.13|
|c2540-d4-b1|35039/13929 ;44439.50|26351/44046 ;140525.67|10148/16188.24|
|c2520-d4-c1|36152/14717 ;46953.55|26692/49782 ;158825.97|6272/10005.19|
|c2520-off-d1|88303/24511 ;78200.62|9025/30766 ;98156.76|6236/9947.76|
|c2520-off-d2|88141/24492 ;78140.00|9330/32567 ;103902.72|4946/7889.93|
|c2520-d4-c2|35855/14536 ;46376.09|26863/45650 ;145643.12|8742/13945.37|
|c2540-d4-b2|35316/14162 ;45182.87|27104/42621 ;135979.30|11348/18102.50|
|c2540-off-a2|88844/24150 ;77048.88|8292/27186 ;86735.02|8816/14063.41|
|c2540-d2-e1|61504/29349 ;93635.92|8012/23136 ;73813.78|11298/18022.74|

Weighted hit rates per configuration (sum hits / sum hits+misses, whole lifecycle including native warmup):

| Configuration | <=8 hit rate | >8 hit rate |
| --- | ---: | ---: |
|c2540-off|78.636%|23.139%|
|c2540-mtp4|71.466%|38.149%|
|c2520-off|78.264%|22.470%|
|c2520-mtp4|71.111%|35.946%|
|c2540-mtp2|67.696%|25.722%|

MTP verification changes batch mixes and counter denominators. These hit rates/logical uploads/copies are not measured PCIe traffic, physical residency, or an isolated causal explanation of performance.

## Whole-lifecycle memory eligibility, variation and margins

Exact adapter: `luid_0x00000000_0x0000f60b_phys_0`. PID-specific PDH dedicated samples cover load, first-prefix, serial, simultaneous and shutdown, using the same nominal250ms fixed schedule as prior evidence. Every per-run counter instance contains its actual serverPID and that adapter; all counter-errors maps are empty. There are1,235 samples and1,226 known-name runtime observations. Per-run median intervals248.867–249.181ms; largest interval265.425ms; the final exited-process sample can produce a shorter final interval. Sampling peaks are **lower bounds**, not continuous allocation/residency traces.

Each configuration is classified by its **maximum full-lifecycle sampled PID dedicated peak across all its clean runs**, not active-phase or favorable-run selection. All nine runs are clean in this cohort; no failed/contaminated run was removed. Target<=8,589,934,592B; band<=8,697,308,774B. Positive margins below are headroom; negative target margin means band.

| Configuration | Class / runs | PID peak bytes / GiB / decimal GB | Target / ceiling margin MiB | Observed repeat peak range MiB |
| --- | --- | ---: | ---: | ---: |
|c2540-off|target /2|8,588,566,528 /7.998725891 /8.588566528|1.304688/103.704687|0.000000|
|c2540-mtp4|tolerance_band /2|8,591,265,792 /8.001239777 /8.591265792|-1.269531/101.130468|0.496094|
|c2520-off|target /2|8,564,973,568 /7.976753235 /8.564973568|23.804688/126.204687|0.000000|
|c2520-mtp4|target /2|8,567,672,832 /7.979267120 /8.567672832|21.230469/123.630468|0.625000|
|c2540-mtp2|target /1|8,589,692,928 /7.999774933 /8.589692928|0.230469/102.630468|unmeasured n1|

- The repeated2520/MTP4 target margin21.230MiB exceeds its0.625MiB observed peak range.2540/MTP4 is band across the configuration, with0.496094MiB observed variation and101.130MiB ceiling margin; no unfavorable peak is dropped.
-2540/off has only1.305MiB target margin. Its two sampled peaks are identical, so it is not marginal under the formal margin-versus-observed-range rule (range0), but this is **not robust future headroom or a physical fit guarantee**.
- E1 has only0.230MiB target margin and no measured repeat variation. Its n1 range/marginality fields are explicitly unavailable, not a known zero-variation proof. A future repeat or unsampled transient could change classification; no strong target-margin claim follows.

Separate all-run lifecycle maxima follow. Each view may peak at a different timestamp. **Do not add these overlapping accounting views**; private commitment is not physical RAM, shared GPU accounting is not an isolated allocation, and whole-device dedicated includes unrelated desktop/application usage. Only the PID dedicated table determines eligibility.

| Configuration | Separate view | Bytes | GiB | Decimal GB |
| --- | --- | ---: | ---: | ---: |
|c2540-off|Host working set|10,841,157,632|10.096614838|10.841157632|
|c2540-off|Host private commitment|19,443,486,720|18.108158112|19.443486720|
|c2540-off|PID shared GPU accounting|10,552,201,216|9.827503204|10.552201216|
|c2540-off|PID GPU committed accounting|18,998,779,904|17.693992615|18.998779904|
|c2540-off|Whole-device dedicated|14,592,278,528|13.590118408|14.592278528|
|c2540-mtp4|Host working set|11,417,292,800|10.633182526|11.417292800|
|c2540-mtp4|Host private commitment|19,973,267,456|18.601554871|19.973267456|
|c2540-mtp4|PID shared GPU accounting|10,909,122,560|10.159912109|10.909122560|
|c2540-mtp4|PID GPU committed accounting|19,497,762,816|18.158706665|19.497762816|
|c2540-mtp4|Whole-device dedicated|14,359,011,328|13.372871399|14.359011328|
|c2520-off|Host working set|10,893,742,080|10.145587921|10.893742080|
|c2520-off|Host private commitment|19,399,872,512|18.067539215|19.399872512|
|c2520-off|PID shared GPU accounting|10,571,075,584|9.845081329|10.571075584|
|c2520-off|PID GPU committed accounting|18,975,186,944|17.672019958|18.975186944|
|c2520-off|Whole-device dedicated|14,331,813,888|13.347541809|14.331813888|
|c2520-mtp4|Host working set|11,484,315,648|10.695602417|11.484315648|
|c2520-mtp4|Host private commitment|19,999,551,488|18.626033783|19.999551488|
|c2520-mtp4|PID shared GPU accounting|10,909,122,560|10.159912109|10.909122560|
|c2520-mtp4|PID GPU committed accounting|19,474,169,856|18.136734009|19.474169856|
|c2520-mtp4|Whole-device dedicated|14,430,048,256|13.439029694|14.430048256|
|c2540-mtp2|Host working set|11,482,361,856|10.693782806|11.482361856|
|c2540-mtp2|Host private commitment|20,040,220,672|18.663909912|20.040220672|
|c2540-mtp2|PID shared GPU accounting|10,904,928,256|10.156005859|10.904928256|
|c2540-mtp2|PID GPU committed accounting|19,494,617,088|18.155776978|19.494617088|
|c2540-mtp2|Whole-device dedicated|14,457,032,704|13.464160919|14.457032704|

### Why real MTP buffers do not equal its small PID-dedicated delta

The initial earlier file-size proxy is explicitly **not used** here. In this n23 cohort,2540/MTP4 raises lifecycle PID dedicated maximum by2,699,264B=2.574219MiB versus2540/off—not zero—but crosses8GiB because the off baseline has only1.305MiB headroom. The reduced2520 arm supplies measured margin. This does **not** mean physical MTP memory costs only2.6MiB.

Native n23 target buffers report Vulkan model4181.22MiB and Vulkan_Host model9788.64MiB. Additional real drafter buffers remain Vulkan0 model225.21MiB,Vulkan_Host model144.00MiB,Vulkan0 compute62.12MiB, plus execution/output/sampling/verification buffers. Draft layer3 shares targetKV29 and draft layers0/1/2 share targetKV28. Logical18+108MiB KV reports are aliases, not another126MiB allocation.

The source-backed Vulkan allocator first tries DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT, then DEVICE_LOCAL. HOST_VISIBLE does not mean systemRAM. Input/output embedding duplication explains host/drafter buffers in source, not per-buffer physical residency. No driver-level allocation backing/residency trace was collected. Do not claim zero cost, spill, CPU fallback, or physical8GiB fit; do not sum file size/native buffers/PDH views.

The exactPID/adapter and phase observations show the small dedicated delta is not merely choosing lifecycle peak versus active allocation. Independent same-phase maxima for2540/MTP4 minus2540/off are:

| Phase | PID dedicated delta MiB | PID shared delta MiB | PID GPU committed delta MiB | Host WS / private delta MiB |
| --- | ---: | ---: | ---: | ---: |
|lifecycle|2.574219|340.386719|475.867188|549.445312/505.238281|
|load|1.316406|312.984375|460.855469|553.562500/599.453125|
|first_prefix|2.574219|469.675781|470.750000|480.238281/505.238281|
|serial_pairs|1.074219|467.164062|469.738281|627.250000/663.093750|
|simultaneous_pairs|3.578125|474.296875|475.867188|610.933594/634.550781|

These are differences of independently timed maxima, **not a time-coincident allocation equation**. Higher shared/commit/host views provide additional cost observations, but do not identify individual buffer residency. [memory-audit.json](results/mtp_n23/memory-audit.json) retains every run/phase min/median/max, exact counter instance, cadence, and allocation/alias lines for both cache sizes and max2. Shutdown is not a synchronized GPU-progress fence; residual shutdown counters are not attributed to a particular allocation.

## Startup, CPU, pressure, safety and conflict limits

| Run suffix | Ready s | Launch→first completion s | Lifecycle elapsed s | Lifecycle CPU s / one-core % | Minimum physical / commit headroom MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
|c2540-off-a1|16.822|18.580|40.523|2.812500/6.940|3.465/4988.133|
|c2540-d4-b1|9.598|11.341|35.335|2.015625/5.704|747.926/4462.477|
|c2520-d4-c1|9.169|10.905|36.038|1.953125/5.420|974.953/3706.551|
|c2520-off-d1|7.872|9.454|30.822|1.781250/5.779|1355.633/3614.027|
|c2520-off-d2|7.661|9.200|30.530|1.640625/5.374|1619.180/3869.688|
|c2520-d4-c2|8.669|10.287|34.552|2.234375/6.467|1440.629/3724.129|
|c2540-d4-b2|8.679|10.339|34.462|3.250000/9.431|1186.293/3823.746|
|c2540-off-a2|8.013|9.558|30.223|2.484375/8.220|1036.996/3611.047|
|c2540-d2-e1|9.299|10.873|32.566|2.312500/7.101|1243.887/3404.324|

CPU seconds are observed server-process lifecycle CPU, including the final exited-process observation; one-core% is CPU/elapsed, not normalized across16 hardware threads. First A1 had a16.822s ready time versus8.013s on A2 under different host/paging conditions. Do not interpret pooled startup values as proof MTP improves startup. Serial/pair results are the fixed post-first-prefix phases, not cold-start claims.

**Pressure was real and retained.** Existing constrained RAM was owner-accepted; no unrelated process was stopped. The owner-only watchdog stops the owned run below256MiB available physical OR512MiB commit headroom for8continuous seconds. It did not fire. A1 nevertheless reached **3,633,152B physical availability**, with three consecutive sampled-low spans5.487530s(load),0.498442s(load),0.746063s(simultaneous). All recovered before the8s cutoff; no other run sampled below256MiB. Minimum commit headroom across all runs was3,569,692,672B. [safety-observations.json](results/mtp_n23/safety-observations.json) preserves episodes, sample counts and recovery observations. No pressure-free, OOM, or safety-stopped result is claimed. A1 remains in all comparisons; the negative MTP comparison also holds versus A2, without substituting an easier pressure-free workload.

Fresh prelaunch known-model-runtime lists were empty in every run. Lightweight name/PID-only conflict observation continued each nominal250ms; no competing named model runtime was observed. Newly authorized behavior terminates **only the owned affected server** and marks overlap contaminated, then stops the dependent sequence. That conflict-stop branch was not triggered/exercised in this cohort. Known-name observation can miss differently named/Python-hosted or between-sample models. No unavailable hub or unrelated private command lines were used.

All9 runs completed44EOS requests and exited0 via graceful consoleCtrl-C. There were no native failures, cache-off runs, watchdog cutoffs, or excluded/contaminated n23 runs; [failure-ledger.json](results/mtp_n23/failure-ledger.json) retains the full run set and negative classifications. This is not proof all unexercised failure/recovery paths work.

## Every-output bounded quality and independent final confirmation

[quality-confirmation.json](results/mtp_n23/quality-confirmation.json) preserves the independent read-only reviewer's full final result, per-ID judgments, original JSON pointers and limits. The reviewer directly reconciled **all9 original runJSON request arrays** against the unique ledger as complete multisets of run/phase/channel/round/input/output/stop, checked the exact44 identities per run, every required predicate and each retained native completion flag. All396 original requests have stop=true,stop_type=eos,truncated=false; zero missing, duplicate, extra, anomalous or unreviewed outputs.

The eleven exact input/output pairs across six sources were manually inspected. Ten exactly match earlier reviewed strings; the new combined topic/only-two wording was independently judged rather than inferred from its components. Core names, numbers, negations, ownership/destination/time corrections and joking reversal are preserved within this bounded fixture; there is no visible truncation, pathological repetition, reasoning trace or explanatory leakage. This is **not unconditional exact fidelity or full-model quality**.

| Exact pair ID | n23 occurrences | Judgment / retained caveat |
| --- | ---: | --- |
|`1351f036a4fcd63f`|28|Mina two, not twelve; added only-two emphasis.|
|`27ca1947fd98eff1`|39|Wait here; five minutes later. Return contextual in come.|
|`2e37d3748acd5be0`|9|Yuna arriving at4,not3; no inventedAM/PM.|
|`4e2fb418d6a53458`|36|Mina two,not twelve; no added only marker.|
|`50c90b165fa41d4a`|7|Same correction with topic 는; mild discourse nuance.|
|`6b4ce7e12d25567b`|63|Key belongs toMinsu,not speaker; minor comma splice.|
|`8eeb9630ba4e5b92`|24|Within5min is an upper bound, not exactly in5min.|
|`97422f3e8ec0311c`|1|NEW topic 는 + only-two 만; core correction intact, both nuances retained.|
|`9db61efbd35b1374`|63|Red,notblue; I mentioned resolves implicit subject.|
|`ccfe38c884f599bd`|63|Yuna arrives at4,not3; natural aspect choice.|
|`ffb4415c4591f5ca`|63|Joking belief corrected; natural implicit Korean subjects.|

Temporal caveat:24 outputs say within-five-minutes versus39 five-minutes-later.29 Mina outputs add only-two emphasis (28 nominative+1 newtopic),43 do not;8 use topic 는. New`97422f3e8ec0311c` occurs once at`mtp-n23-c2540-d2-e1`,simultaneous Peer round9 (`#/requests/33`): `미나는 티켓을 12장 산 게 아니라, 2장만 샀어요.` It preserves Mina, rejects12 and affirms2, with disclosed topic and exclusivity nuance.

Quality boundary: original requestJSON supplies accumulated streaming output plus final completion metadata. Final response `content` is empty for these streaming requests; it is not a second independent copy of delivered text. Raw streaming chunks/native EOS were not replayed by the reviewer. Ledger/original reconciliation and manual semantic coverage do not validate performance, physical memory backing, universal translation quality or causal MTP quality effects.

## Exact commands and durable evidence

`<APP_PYTHON>` is the verified application virtualenv Python3.14.7; `<ARTIFACT_ROOT>` is the external verified-artifact root. Personal absolute paths remain external. Choose fresh run IDs/destinations; helper guards preserve prior identities and require separate n23 result/report paths. Actual sanitized native/probe command arrays and loopback ports are retained in every original runJSON.

Fresh prerequisite command executed:

```text
<APP_PYTHON> experiments/gemma4_26b_qat/mtp_acquire.py --root <ARTIFACT_ROOT> --adapter luid_0x00000000_0x0000f60b_phys_0 --baseline 270c0a4b7dd9a8aa85612695f4ad02596452035d --output experiments/gemma4_26b_qat/results/mtp_n23/prerequisites.json
```

Each probe command uses this exact shared form; the following table supplies every actual varying argument in execution order:

```text
<APP_PYTHON> experiments/gemma4_26b_qat/mtp_probes.py --root <ARTIFACT_ROOT> --adapter luid_0x00000000_0x0000f60b_phys_0 --n-cpu-moe 23 --baseline 270c0a4b7dd9a8aa85612695f4ad02596452035d --results experiments/gemma4_26b_qat/results/mtp_n23 --cache-mib <CACHE> --draft-max <MAX> --group <GROUP> --run-id <ID> --allow-pressure
```

| ID | CACHE | MAX | GROUP |
| --- | ---: | ---: | --- |
|mtp-n23-c2540-off-a1|2540|0|control|
|mtp-n23-c2540-d4-b1|2540|4|unchanged|
|mtp-n23-c2520-d4-c1|2520|4|reduced|
|mtp-n23-c2520-off-d1|2520|0|control|
|mtp-n23-c2520-off-d2|2520|0|repeat|
|mtp-n23-c2520-d4-c2|2520|4|repeat|
|mtp-n23-c2540-d4-b2|2540|4|repeat|
|mtp-n23-c2540-off-a2|2540|0|repeat|
|mtp-n23-c2540-d2-e1|2540|2|max2|

Actual complete native command arrays for A1(off) and B1(MTP4); other runs differ only in cache/max and ephemeral port as recorded, not hidden controls:

```json
{
  "mtp-n23-c2540-off-a1": [
    "<ARTIFACT_ROOT>\\runtimes\\b11537\\llama-server.exe",
    "--model",
    "<MODEL>/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf",
    "--alias",
    "issue245",
    "--load-mode",
    "none",
    "--ctx-size",
    "4608",
    "--parallel",
    "3",
    "--batch-size",
    "512",
    "--ubatch-size",
    "512",
    "--threads",
    "4",
    "--threads-batch",
    "4",
    "--cache-type-k",
    "f16",
    "--cache-type-v",
    "f16",
    "--cache-prompt",
    "--reasoning",
    "off",
    "--reasoning-budget",
    "0",
    "--warmup",
    "--perf",
    "--metrics",
    "--no-webui",
    "--threads-http",
    "1",
    "--host",
    "127.0.0.1",
    "--port",
    "54669",
    "--prio",
    "-1",
    "--device",
    "Vulkan0",
    "--flash-attn",
    "on",
    "--fit",
    "off",
    "--n-gpu-layers",
    "99",
    "--n-cpu-moe",
    "23",
    "--moe-cache-mib",
    "2540",
    "--swa-full",
    "--cache-ram",
    "0",
    "--verbosity",
    "4"
  ],
  "mtp-n23-c2540-d4-b1": [
    "<ARTIFACT_ROOT>\\runtimes\\b11537\\llama-server.exe",
    "--model",
    "<MODEL>/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf",
    "--alias",
    "issue245",
    "--load-mode",
    "none",
    "--ctx-size",
    "4608",
    "--parallel",
    "3",
    "--batch-size",
    "512",
    "--ubatch-size",
    "512",
    "--threads",
    "4",
    "--threads-batch",
    "4",
    "--cache-type-k",
    "f16",
    "--cache-type-v",
    "f16",
    "--cache-prompt",
    "--reasoning",
    "off",
    "--reasoning-budget",
    "0",
    "--warmup",
    "--perf",
    "--metrics",
    "--no-webui",
    "--threads-http",
    "1",
    "--host",
    "127.0.0.1",
    "--port",
    "61590",
    "--prio",
    "-1",
    "--device",
    "Vulkan0",
    "--flash-attn",
    "on",
    "--fit",
    "off",
    "--n-gpu-layers",
    "99",
    "--n-cpu-moe",
    "23",
    "--moe-cache-mib",
    "2540",
    "--swa-full",
    "--cache-ram",
    "0",
    "--model-draft",
    "<ARTIFACT_ROOT>\\models\\mtp-gemma-4-26B-A4B-it-Q4_0.gguf",
    "--spec-type",
    "draft-mtp",
    "--spec-draft-n-max",
    "4",
    "--verbosity",
    "4"
  ]
}
```

Final analysis and memory audit commands exercised (both exit0):

```text
<APP_PYTHON> experiments/gemma4_26b_qat/summarize_mtp.py --results experiments/gemma4_26b_qat/results/mtp_n23 --n-cpu-moe 23 --reference-cache-mib 2540 --baseline 270c0a4b7dd9a8aa85612695f4ad02596452035d --markdown experiments/gemma4_26b_qat/MTP_N23_TIMINGS.md --check
<APP_PYTHON> experiments/gemma4_26b_qat/mtp_memory_audit.py --results experiments/gemma4_26b_qat/results/mtp_n23
```

Checks actually exercised: nine complete owned native lifecycles; final summary/check over396 requests with all per-run predicates true, matchingP/C and input/sampling, native EOS metadata, requested n23/group/slot-granularity, true-off omission/environment clearing, exactPID/adapter/all phases, native cache use, MTP API/cumulative-counter agreement, requestedmax/nativeGPU/sharedKV, decodeTPS formula,135 native overlaps and graceful exits. Independent reviewer reconciled final396 original requests and all11 semantic strings. Static inherited-fixture comparison found zero identity/P/C differences. All failures/pressure/negative results remain. No product build, test suite, linter, formatter, type checker, app/GUI/audio smoke, physical-card or coexistence check was run or claimed.

Minimal tooling changes: parameterized the existing MTP probe/acquisition baseline and summary placement/reference/output; n23 writes cannot silently enter oldmtp paths, and mixed placements cannot pool. Added exact placement/cache-slot/P-C/finalEOS/off-env audits and explicit n1-variation unavailability. Existing lifecycle helper now records cleared LLAMA environment and supports owner-only conflict termination (opt-in by MTP probe); inherited workload, sampling schedule and safety thresholds remain unchanged. No product/maintained-doc/settings/runtime/backend/driver changes or Git/GitHub mutations were made by this worker.

## Release, limits and recommendation

[release.json](results/mtp_n23/release.json) records2026-10-16T04:00:20.613957+00:00: no named model runtime remains, all9 owned serverPIDs exited0 gracefully, per-run sampler threads/PDH handles closed. No owned app/provider/UDP service was created; unrelated processes untouched. Unique external raw logs/samples, target/runtime/drafter are intentionally retained. Sanitized repository evidence includes every runJSON, native log, slots ledger, compact samples and headroom record; raw personal paths remain external.

**Within this n23 cohort,2540/off is the best repeated primary-pair target result, but with only1.305MiB sampled target margin.** Reduced2520/off offers23.805MiB margin and better pair-tail/serial outcomes at a36.987ms primary median cost. Cache adoption is an owner policy choice, not implied by using2540 as control.

**Recommend MTP off for the tested owner-selected n23 workload.** Neither target2520/MTP4 nor band2540/MTP4 repeatably improves the matched off controls; band MTP does not beat the best target. Max2 is a retained failed n1 exploratory candidate, with positive acceptance/serial-median progress but no qualifying pair benefit; reducedmax2/other draft lengths/long outputs remain unmeasured. No n24 outcome is used to infer this n23 result.

All observations come from a24GiB-class7900XTX under accepted host pressure. Sampled PID accounting near8GiB does not establish total fit/performance on a physical8GiB card, other drivers/vendors, actual app/GUI/audio, VRChat/desktop coexistence, or general MTP quality/performance. Publishing/pinning/review remain with Director; owner-selected placement is n23, while cache/MTP adoption and product handoff remain separate.
