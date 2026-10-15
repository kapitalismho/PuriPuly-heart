# Issue 246: Gemma 4 E4B Q4, newer llama.cpp and PLE

## Decision

**Retain the shipped b10423 pin and current CPU/Vulkan defaults in this report-only change.** There is no demonstrated short-translation latency win from updating alone, and forcing lazy PLE does not add a measured warm benefit on this artifact/hardware. No product source, pin, default, normal settings, model selection, index, history or remote was changed by execution.

There is a separate, useful candidate result: **b11537 restores saved context checkpoints and actually reuses the disk-saved prefix after process restart; b10423 reports successful restore but then fully prefills the first translation.** This is a KV/checkpoint benefit, not a PLE speedup. The measured alternative is b11537 with `--cache-ram 0`, otherwise the current profiles. It merits a separately authorized integration decision if restart/pair-reentry cost matters enough to accept slightly slower hot turns and additional checkpoint memory. It is **not** a recommendation to point the unmodified app at a different runtime directory: that override retains stale shipped build identity.

The current native default prompt-cache policy also creates substantial **intentional retained-state RAM pressure** in both revisions. Default-policy soaks were censored by the experiment's RAM guard, earlier on b11537. Cache0 completed 200 requests per revision/backend, and a bounded 512MiB-cache control demonstrated eviction and a plateau. This is not evidence of an unbounded leak, nor proof that the uncompleted default-policy soak would plateau safely on this desktop.

### Evidence map

- [Frozen identities, archive and extracted-file hashes](results/identity.json), [hardware/driver/power snapshot](results/environment.json), [producer build provenance](results/build-provenance.json), [bounded b11630 identity](results/followon-identity.json).
- [Exact per-run arguments, prefixes and resource records: evidence index](results/evidence-index.json). Representative [CPU command](results/new-cpu-cache0-final-a-run.json) and [Vulkan command](results/new-vulkan-cache0-final-a-run.json).
- [2239 compact standalone request records](results/requests.jsonl), [counts](results/execution-counts.json), [per-arm distributions/buffers/errors](results/summary.json), [paired comparisons](results/comparisons.json), [20-request memory bins](results/memory-soak-bins.json).
- [Actual-app lifecycle/correlation summary](results/app-summary.json), [CPU cache0 app evidence](results/app-new-cpu-cache0-app.json), [Vulkan cache0 app evidence](results/app-new-vulkan-cache0-app.json).
- [Quality review and limits](QUALITY.md), [workload-policy provenance](results/workload-policy.json), [protocol](PROTOCOL.md), [upstream findings and immutable citations](UPSTREAM.md), [reproduction](REPRODUCE.md).
- [Producer compaction policy and normalized-source digests](results/producer-compaction.json), [effective sampling/template summary](results/sampling-template-summary.json).

All inputs are synthetic. Published paths are placeholders. Runtime/model binaries, isolated configurations, encrypted secret stores, control credentials and bulk original producer logs are outside this directory. The index gives SHA-256/size for each result's **canonical UTF-8 LF publication/Git blob bytes**, not Windows-native CRLF bytes; it deliberately excludes its own hash.

### Publication-byte repair

The publishing checkpoint `8e6c015b2f5afc41316aa011ead1e936de7bb149` exposed Windows CRLF→Git LF normalization warnings. The original index had counted Windows worktree bytes. `finish_evidence.py` now writes every indexed text result as canonical UTF-8 LF before hashing; the shared JSON writer explicitly emits LF, including the index. Canonical indexed total is **13300418 bytes across283 result files**, replacing13491464 Windows bytes (191046 newline bytes removed). No inference or scientific record was rerun or semantically changed.

[Read-only publication audit](PUBLICATION_AUDIT.json) compares the repaired index and actual LF working bytes against all283 stored result blobs in that checkpoint: **283/283 match, zero failures**. The repaired index/helper/docs themselves are proposed local changes, not claimed present in that old checkpoint. [Audit helper](verify_publication.py) permits the publishing owner to repeat the comparison at the final candidate. Original normalized-sanitized pre-compaction digests retain their original meaning; the complete `producer-compaction.json` matches the checkpoint blob unchanged. These digests are distinct from the index hashes of compact published files.

## 1. Frozen baseline and artifacts

Execution checkout: `7a8151be6114a950f062e2430bd36f899d57e4de`, initially clean, branch `experiment-optimize-gemma-4-e4b-on-cpu-vulkan-wi`, initially equal to origin/dev. The current code supersedes old #75 two-slot amendments: **three slots, total context4608, 1536 per slot**. #64's Q4 selection and later CPU T4/B12, draft1/1, max/min3/2 decisions were preserved; rejected whole-model Q2 and other runtimes were not revisited. [Issue and criteria](https://github.com/kapitalismho/PuriPuly-heart/issues/246), [prior selection](https://github.com/kapitalismho/PuriPuly-heart/issues/64), [integration history](https://github.com/kapitalismho/PuriPuly-heart/issues/75).

Scientific baseline remains `7a8151be6114a950f062e2430bd36f899d57e4de`. Later origin/dev `29a77232` and #247/#248 are not incorporated into this experiment; the publication repair does not rebase or reinterpret the frozen measurements.

### Measured host

| Surface | Frozen observation |
| --- | --- |
| OS | Windows11 Education, 10.0.22631 |
| CPU | AMD Ryzen7 9800X3D, 8 physical /16 logical cores |
| RAM | Two16GiB DIMMs, configured5600MT/s; OS reports32361MiB total |
| GPU | RX7900XTX, Vulkan0, 24560MiB; vendor1002/device744c |
| Driver/API | AMD26.8.1(LLPC), Windows32.0.31041.1004; device Vulkan1.4.349, loader1.4.341 |
| Power | Existing High Performance plan `8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c`, unchanged |
| Python observer/app | Python3.14.7, psutil7.2.2, httpx0.28.1; current checkout package path observed inside actual hosts |
| Persisted numeric profile | `threads=4`, `threads_batch=12`, `draft_threads=1`, narrowly read from the numeric profile file |
| Background policy | Unrelated desktop processes, including an existing user app, left untouched; serial owned inference, per-arm before/after host CPU samples. Not a clean-room OS or gaming benchmark. |

### Fixed model pair

Both artifacts are from `unsloth/gemma-4-E4B-it-qat-GGUF`, revision `8c5a9e4fd5482e2be20fe0bf013b4c262a8f4265`, and matched the repository's selected sizes/hashes. Existing public model files were reused read-only; isolated app installations use task-owned hardlinks and a public installed manifest.

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf` |4215695776|`df0fd4ee07072c607c29a0a1cb4f98918426cca12f45a2776bdd6ee6d09a4de3`|
| `mtp-gemma-4-E4B-it.gguf` |59678016|`423074e537504b4f9ec5eafed5c639fac82c96631626efccacdd3c4039b20605`|

The filename is not the tensor histogram: target **321F32 +345Q4_0**, 666 tensors; drafter **26F32 +23Q4_0**,49 tensors. No K-quant kernel gain can be assumed from `Q4_K_XL` in the filename.

| Relevant target tensor | Actual type/shape | Payload bytes |
| --- | --- | ---: |
| `per_layer_token_embd.weight` |Q4_0 `[10752,262144]`|1585446912 =1512MiB;6048bytes/row|
| PLE input projection |Q4_0 `[2560,10752]`|15482880|
| PLE input norm |F32 `[256]`|1024|
| Ordinary token embedding |Q4_0 `[2560,262144]`|377487360 =360MiB|

### Pinned runtime/build identities

| Build | Commit | CPU ZIP SHA-256 | Vulkan ZIP SHA-256 |
| --- | --- | --- | --- |
|b10423|`a94d563ed801d1da1b8c2432946de07d0231bb3d`|`b5a396f113a344578c0766331704bd541fd743c4c8e92858bea18440ee0ab19a`|`510447fb021c80a264b2181c885b5f2ce9cc5b66c65d447cd1f9ce7ba81dc222`|
|b11537 primary|`8e2d31e0eb658d0382d3106a61369dff81f0c080`|`a05c2d35ebe9787489277d6e039b391fd006384a6c49d069395f3a7b47186fd2`|`ae34d6a850c5d362cdfc0ef8f88f31fc499dab6df8853cec7887a32891aa5bbe`|
|b11630 ON diagnostic only|`6868a52cb337450658006ad2f9281c2eb63f1666`|`82463bff7b42c41e8dc934202fb60ba28e1120c04264484b8d349b4a5958190d`|`597684a3573bc0d0993a50cfa78433b8af7041db5ad008dc693ac6105b353e35`|

Official release URLs, archive sizes, and **every extracted file's hash** are in the identity records. CPU/Vulkan packages share the same server EXE within each release; matching EXE hashes alone do not identify the backend DLLs. Server EXE SHA-256 values are respectively `c5f9d7db35cb636d93c66562d787de14189a14f5920743b326533f7c869ce2a0`, `7b36a3bc3a24e552d0fcc18441893bc198abaeb0fa77d653241d762077c995c6`, and `f141b0587033f06258ad89e91f904f9298033c6180aca472e4edbee267053c73`.

All actual server `--version` outputs identify Clang20.1.8, **not the Vulkan compiler**. Windows Vulkan ZIPs merge CPU-job server/shared libraries with a separately MSVC-built Vulkan DLL. Research recovered actual successful producer job logs:

- b10423: [run31731999834](https://github.com/ggml-org/llama.cpp/actions/runs/31731999834), CPUjob94611104974 /Clang20.1.8; Vulkanjob94611104924 /MSVC19.51.36252.0.
- b11537: [run37968245796](https://github.com/ggml-org/llama.cpp/actions/runs/37968245796), CPUjob113983515958 /Clang20.1.8; Vulkanjob113981686904 /MSVC19.51.36260.0.
- Both Vulkan jobs use SDK1.4.357.0 and observed configure command `cmake -S . -B build -DGGML_VULKAN=ON -DGGML_NATIVE=OFF -DGGML_CPU=OFF -DGGML_BACKEND_DL=ON -DLLAMA_BUILD_BORINGSSL=ON`, then Release target `ggml-vulkan`.
- CPU configure uses the LLVM toolchain, Ninja Multi-Config, native off, all CPU variants, dynamic backends, OpenMP, BoringSSL, tools/server/RPC on and examples/tests off. b11537 adds `GGML_OPENMP_FETCH=ON`; packaging changes from `libomp140.x86_64.dll` to `libomp.dll` plus its LLVM license. Complete observed commands and source links are in [build-provenance.json](results/build-provenance.json). These are observed configure flags, not recovered complete CMakeCache/expanded compiler invocations or cryptographic publication attestations. The separate b11630 DLL producer compiler was not recovered.

### Effective settings and prompt

The harness imports the current product command builder. Shared: mmap,4608 context/3slots,512 batch/ubatch,f16 K/V, cache-prompt, reasoning off/budget0,warmup,perf,metrics, no web UI, HTTP1, loopback,priority-1. CPU uses4/12, FAoff, CPU target/drafter, MTP draft1/1,max/min3/2,p-min0,f16 draft KV. Vulkan uses4/4,FAon,Vulkan0,ngl99; logs confirm43/43 target layers offloaded, no drafter. New defaults retain greedy draft sampling and inherited draft ubatch; no invented old flag is passed. Actual `--help`, props, command arrays and producer logs—not argument names alone—supply effective-option evidence.

The GGUF/native chat-template hash is `241c50d86bdfe5e43307da87f559cd2416aacd67a8de46c15acc0105ef2200b7`. Current public prompt assembly is unchanged. English→Korean rendered system SHA is `4ffb791e10dbe6bffd9ef2e2a68ebb019f541799b2478a2cb7587158ede61d77`; other pairs and formatted hashes are in each run's prefix records and retained prompt text files.

Standalone performance requests use temperature0.6,seed42,normal completion and a128-token safety ceiling; effective top-k64,top-p0.95,min-p0.05,repeat-penalty1 and sampler order are retained in [sampling-template-summary](results/sampling-template-summary.json). Actual app requests use current temperature0.6 with no seed override, so app/stochastic outputs are not pooled into seed42 timings.

## 2. Workload, gates and sample accounting

Twelve frozen short evaluation inputs cover English/Korean/Japanese/Chinese, names, decimal/time/room numbers, negation, fragments and quoted speech. Three separate development inputs drive tuning. Final timing rotates the first three evaluation pairs through all three slots. A changing `Step N:` near the start of history prevents an exact-whole-user-message cache hit, while retaining the language-pair system prefix.

**Representation disclosure:** the primary controlled workload uses five correctly ordered labelled contents and current message wrapping, but omitted the real resolver's bullet/quote punctuation. It is not claimed byte-identical to application context. A separate no-tuning confirmation passed those same five contents through the **real ContextResolver**, correct self/opposite-peer language pairs, and real bullets/quotes. Wrong-pair and61s-old synthetic entries were required to be filtered. Current integrated request history is five entries in60s; older channel-query defaults3/30s are not the current request resolver. [Exact policy and source references](results/workload-policy.json). The actual app then filled its genuine five-entry context with30 consecutive turns per finalist.

Final controlled runs use old/new/new/old order, two fresh processes per revision/backend/profile,33 requests/process, first three excluded from timing: **60 measured per version in each comparison**. Literal confirmation is12/process,9 measured per version, not pooled. Request wall spans `/apply-template` through completed `/completion` SSE; TTFT is time to the first nonempty SSE content chunk, not an instrumented kernel clock. Pair preparation/readiness are separate. Actual app transport walls are non-streaming native HTTP components, not CLI-start-to-publication latency or GUI latency.

Observed standalone totals: **2239 completed requests across61 arms**:72 quality/deterministic,174 tuning,528 final,1178 soak,96 literal,191 contention including background/initial-observation controls. No recorded request error, output-limit stop or mechanical quality flag. This does not erase the preflight failure, app setup degradation, semantic caveats or four censored soaks below. Prefix warmup's intentional one-token preparation requests are diagnostic and are not counted as completed translation samples.

Standalone `exit_code=1` with `owned_process_exited=true` records deliberate Windows `terminate()` cleanup. It is neither evidence of a graceful native shutdown nor a crash by itself. Request completion, resource-guard stop and process termination are separate observations.

The initial48 compatibility translations have12/12 exact output/hash parity across all four old/new/backend profiles, but **not an unqualified12/12 semantic pass**. Independent read-only scrutiny found “Maybe later, after lunch.” → “たぶん、お昼過ぎかな。” in all four: the conventional after-noon meaning can lose the meal-completion condition and weaken “later.” This is a shared fidelity concern, not an update regression. Candidate greedy MTP on/off then completed12+12 with identical outputs. Automatic sanity success and parity do not establish general correctness or replace #64's648-sample model-quality evidence.

Historical quality rows have matching system/formatted-prefix hashes but lack a full user/context hash or rendered user request. [QUALITY.md](QUALITY.md) preserves the inspected unchanged five-line workload definition and clearly labels it definition provenance, not a retroactively captured hash. Later rows carry explicit history metadata. No prompt/model fix or additional quality sweep was opened for this supported negative recommendation.

All eight literal arms at request_index0 produce `미나는 12장짜리 티켓을 사지 않았대요.`: shared ticket-count attachment ambiguity **and added hearsay**, not just cosmetic fluency. The actual candidate Vulkan warm source explicitly says17 tickets and produces `17장짜리 티켓`: the same attachment concern, **not a12→17 numeric substitution**. `Jisu`/`Jisoo` is acceptable romanization, not a failure. Names/numerals/negation surviving is insufficient semantic protection. No broad quality improvement or newly proven runtime-specific regression is claimed; these observations prevent an unconditional rollout-ready quality result. See [QUALITY.md](QUALITY.md).

## 3. Update-only and separate resource-policy results

All values below are milliseconds, median **[p10–p90]**, with60 measured completed translations/version. These are workload distributions, not confidence intervals.

| Profile | b10423 wall | b11537 wall | b10423 TTFT | b11537 TTFT | Median paired new−old wall |
| --- | ---: | ---: | ---: | ---: | ---: |
| CPU, native default cache |978[726–1306]|966[764–1242]|525[491–742]|569[524–664]|+26ms; p10–p90 −218…+163|
| Vulkan, native default cache |402[360–491]|402[350–483]|232[215–249]|223[202–238]|−6ms; −29…+17|
| CPU, **both cache0** |850[661–963]|874[688–990]|462[441–498]|496[470–524]|+31ms; +4…+54|
| Vulkan, **both cache0** |354[309–429]|370[320–447]|191[177–208]|194[175–207]|+11ms; −10…+33|

The CPU update-only pooled median being slightly lower does **not** establish a gain: paired median is worse, repetition medians cross (old986/971,new1026/951), and a completed new-CPU6.409s outlier is retained (old maximum2.318s). Vulkan repetition medians are old403/397,new399/403, with essentially unchanged completed wall. There is no robust update-only short-translation win to recommend.

Separate cache0 tuning is useful against each revision's own default: candidate pooled medians change966→874ms CPU and402→370ms Vulkan, with lower retained-state cost. But old cache0 is itself faster850/354ms. Do **not** relabel cache-policy improvement as a new-runtime or PLE win. At fixed cache0 the paired ratios are approximately+3.9% CPU and+3.1% Vulkan, with the full spread above; this is a measured cost, not a universal regression percentage.

Native medians, actual counts and CPU cost provide the breakdown:

| Track | Prefill tok/s old→new | Decode tok/s old→new | Process CPU seconds/request old→new |
| --- | ---: | ---: | ---: |
| CPU default |193.5→181.1|38.2→41.2|4.602→4.617|
| Vulkan default |480.1→524.8|73.7→72.5|0.0156→0.0078|
| CPU cache0 |212.7→199.6|47.2→46.2|3.695→3.609|
| Vulkan cache0 |547.4→553.3|79.3→74.4|0.0000→0.0000|

Windows CPU-time quantization is15.625ms; zero Vulkan median is **not** zero CPU work. Native decode-rate conventions include MTP/native timing behavior; they are not substituted for completed wall. Typical final outputs are11–19 actual tokens, incremental prefill roughly94–104 tokens. Exact sums/distributions are in [comparisons](results/comparisons.json) and [requests](results/requests.jsonl).

### Literal current-resolver confirmation

Nine measured/version per row; first3 warmups retained separately. No retuning.

| Literal policy | Wall old→new | TTFT old→new | Paired wall delta median [p10–p90] |
| --- | ---: | ---: | ---: |
| CPU default |892→1010|509→576|+81[+29…+388]|
| CPU cache0 |867→882|484→509|+17[+12…+32]|
| Vulkan default |368→401|210→225|+10[−5…+37]|
| Vulkan cache0 |374→386|202→207|+9[−17…+29]|

This small bridge does not reverse the recommendation: CPU has no measured update gain; Vulkan differences overlap request spread. It is not an excuse to discard the earlier identical-input controlled attribution or to expand into another broad matrix.

## 4. PLE attribution and bounded tuning

The baseline already executes PLE lookup/input projection host-side; this is not new. Candidate GET_ROWS graph ordering and lazy host-page prefetch are released changes. Per-layer gate/project operations are different operations and may follow GPU layers. [Source-level distinctions](UPSTREAM.md).

**AUTO is inactive on the verified Q4.** Its1512MiB marked table is below the strictly-greater-than4GiB tensor threshold; even the entire file is smaller than4GiB. Actual default/AUTO and OFF logs do not activate lazy prefetch. Forced ON logs both lazy reads and prefetch for `per_layer_token_embd.weight`, and quantized Q4 lookup completes correctly. AUTO/ON each received18 development translations/backend (15 measured); OFF received3 sanity completions/backend after eligibility/log agreement, not a redundant full inactive AUTO/OFF matrix.

| b11537 development arm | CPU wall /TTFT | Vulkan wall /TTFT |
| --- | ---: | ---: |
| AUTO |934 /564|379 /216|
| ON |945 /566|385 /209|

**No measured warm completed-wall gain from ON. This is not a cold-OS-cache verdict.** Hashing/earlier reads warmed the OS cache; restarting the process did not flush it. Windows `PrefetchVirtualMemory` acquires host file pages, not a GPU-resident cache. ON also changes lazy buffer assignment/mapping, so ON/OFF is not a pure prefetch ablation. No alternative PLE precision/model artifact was needed; no released purpose-built Vulkan PLE cache was found. Historical closed/unmerged `on-direct` experiments were not used.

Measured native buffers, MiB:

- CPU default/AUTO: target mapped4004.96 and REPACK2491.17; drafter mapped40.80 and REPACK41.84; target KV72+180. Candidate compute120.26 versus baseline125.26. ON adds/separates a1512 mapped lazy buffer; b11537 still prints the broad4004.96 mapping. Printed regions can overlap and must not be added as resident RAM.
- Vulkan default/AUTO: host input mapping1872 (=360 ordinary embedding +1512 PLE), target Vulkan2493.31, KV72+120, compute124.77 plus Vulkan_Host43.61. ON splits host mapping360+1512; full43/43 transformer offload remains. Allocation movement does not eliminate host cost.
- b11630 ON diagnostic,9 translations/backend, was selected only to inspect foreign-mmap exclusion/busy-slot fixes. CPU printed host mappings shrink to small non-lazy ranges plus1512 lazy, while REPACK remains2491.17. **Observed model-ready working set/private commit stays approximately5.835/3.467GB**, like b11537, and Vulkan placement is materially unchanged. Smaller printed mappings did not establish physical-memory or latency improvement. No full b11630 qualification/claim follows from this diagnostic.

Other bounded development branches:

- Candidate CPU cache0, FAoff:809ms wall/465ms TTFT. FAon:889/532, saving only about77MiB private commit in this short development window. Retain CPU FAoff; no KV-precision or batch/thread sweep was justified by this result.
- Candidate CPU cache0 `--no-repack`:6 completions,3 measured,1400ms wall/928ms TTFT, about1065MiB private versus3624MiB with repack. This is a real low-RAM/latency tradeoff already considered and non-selected in #64, **not a dominated point** and not a qualified new recommendation. It was explicitly pruned without long stability/app qualification.
- Greedy CPU MTP on/off diagnostic:12+12 identical-output cases, medians870/1228ms. Current CPU MTP remains on. In the200-request cache0 soaks, native cumulative draft statistics record3012 generated draft tokens in both, accepted1866(old)/1736(new), mean accepted length2.86/2.73, draft-generation8483.358/8402.390ms. Begin/accept timing fields are retained verbatim; **target verification cost is not separately exposed**. Do not invent it from residual wall or rename those fields. No optional Vulkan-MTP profile was opened.

## 5. Retention, stability, peaks and contention

### Native default cache: incomplete planned soaks

The runtime advertises8192MiB prompt cache. Planned96 translations/arm stop after a completed request when available physical RAM falls below1536MiB or owned private commit exceeds14GiB. All four stopped on the available-RAM bound:

| Track | Completed/planned | Native retained prompts /MiB at stop | End private /working set, GiB | Available physical bytes |
| --- | ---: | ---: | ---: | ---: |
| old CPU |55/96|54 /4688|8.02 /10.32|1470521344|
| new CPU |34/96|33 /4525|8.03 /10.33|1508290560|
| old Vulkan |81/96|80 /6945|10.52 /10.28|1567207424|
| new Vulkan |48/96|47 /6581|10.35 /10.12|1530679296|

These are **censored/failed-to-complete planned soaks**, not96 passes and not native crashes. Their limited windows do not establish default-cache eviction/plateau or an unbounded leak. At equal33-request final workloads candidate private commit reaches8.464GB CPU /8.802GB Vulkan versus old6.601GB /6.912GB. The update has a real host-memory cost even though cache0 can remove much of the retention.

### Cache0 and bounded eviction control

All four cache0 arms completed **200/200**, with no request error, crash, corruption, runaway repetition, truncation or observed MTP assertion. First3 are excluded from latency distributions, not completion counts. These are a few thousand generated tokens/arm, not a claimed50k-token endurance run.

| Cache0 track | Wall median[p10–p90] ms | End private /WS MiB | Last20 private MiB range |
| --- | ---: | ---: | ---: |
| old CPU |842[662–970]|3510.6 /5882.0|3510.625–3510.629|
| new CPU |867[688–988]|3744.7 /6114.8|3744.703–3744.707|
| old Vulkan |362[307–440]|3807.9 /3578.4|3807.809–3807.852|
| new Vulkan |369[323–451]|4067.3 /3838.3|4067.328–4067.336|

Candidate private commit steps up by about60MiB around requests103–105, one approximately20MiB step per slot, after the Step marker changes prompt shape at100. It then plateaus through199. This is compatible with retained per-slot context checkpoints, **not proof for arbitrary prompt lengths**. We do not extrapolate the first few growing requests linearly into a leak. Cache0 removes global prompt-cache storage, not three resident KV slots, per-slot checkpoints or the app's disk prefix cache.

A separate `--cache-ram 512` control completed80/80 CPU and80/80 Vulkan. Native logs repeatedly remove oldest entries, report at most4 prompts/507.453MiB and finish with3/440.495MiB. Last20 private commit ranges are4126.027–4126.062MiB CPU and4448.277–4448.695MiB Vulkan. This directly supports **bounded retained-cache cost** for that policy. Median878/395ms was not a better latency/memory point than cache0;512 is an attribution control, not an app-qualified recommendation.

[Memory bins](results/memory-soak-bins.json) retain every20-request interval. [Summary](results/summary.json) retains model-ready/end samples, per-request25ms sampled maxima, OS lifetime peak counters and all warnings. M0 before model load was not observed; mapped-region `rss` from the Windows observer is a region-size observation, not a resident-byte proof. GPU memory is phase-sampled, not peak-sampled.

At cache0-soak end, old/new Vulkan dedicated bytes are2855018496/2854768640; shared787374080/803819520. This is approximately2.66GiB dedicated plus0.73/0.75GiB shared per owned process, with43/43 offload and a24GiB device. Shared memory is reported, not hidden or automatically labeled spill. CPU probes have no matching GPU counter instance; older records say unavailable(empty), newer records explicitly record no matching instance and null byte counters. Neither is a measured GPU zero.

### Reproducible controlled contention

Finalist probe changes **runtime plus cache policy** versus shipped baseline; it is not update-only attribution. Each primary arm completes21 requests,18 measured after warmup.

| Controlled load | Baseline default wall/TTFT ms | Candidate cache0 wall/TTFT ms | Outcome |
| --- | ---: | ---: | --- |
| Six integer-loop workers, logical affinities0…5, normal priority |1230 /755|1224 /765|All42 primary completions; no latency improvement claim |
| Second same-Q4 Vulkan server, fixed b10423/cache0 |606 /328|580 /294|All42 primary completions plus33/32 background completions; no errors |

CPU worker **process trees**, not only Windows venv launcher PIDs, have recorded creation times, actual affinity and start/end CPU counters. Physical-core mapping is not asserted. [Baseline accounted run](results/contend-old-cpu-default-accounted-contention.json), [candidate accounted run](results/contend-new-cpu-cache0-accounted-contention.json). Initial CPU probe records only launcher counters (zero CPU), so those42 requests remain as an **observation/accounting failure**, not the qualified contention evidence; the corrected arms are separately named, not overwritten. Initial workers had a180s self-bound; corrected owned trees were explicitly terminated and waited.

Accounted CPU measurement intervals are42318.7600982→42344.7762924s(old) and42371.6655084→42397.4870734s(new), monotonic:26.016/25.822s. The six owned trees consume143.828/143.328CPU-seconds in those intervals (approximately553%/555% of one core). These are observed load costs, not a claim of six fully saturated physical cores or zero launcher overhead.

GPU load is real owned inference, not a VRChat/game simulation. [Baseline](results/contend-old-vulkan-default-contention.json), [candidate](results/contend-new-vulkan-cache0-contention.json) retain measured start/end monotonic times and cleanup; companion run records retain both processes' GPU counters. Unrelated workloads were not stopped. No unrestricted translation thread-budget increase was used to gain speed.

## 6. Actual isolated PuriPuly confirmation

The current source app, real CLI, settings transactions, orchestrator, managed owner, native HTTP transport and model performed **116 completed translations** across six successful profile tracks /14 fresh host processes. Four old/new default-cache tracks have9 translations in2 processes each; candidate cache0 CPU and Vulkan each have40 translations in3 processes (38 initial,1+1 after fresh restarts). Every submitted text had terminal `applied` admission **and** a matching completed translation event with child-turn/instance correlation. Admission alone was not accepted as completion. CPU and Vulkan effective backends and exact package/native identities were captured independently; UI intent alone was not backend proof.

Current `capabilities`, `settings choices/current`, capture/OSC/overlay status and secrets presence were queried. Native/slot identities are not exposed by that public CLI, and normal owners discard child stdout; task-local transparent capture instrumentation supplies those observations. No translation substitute was injected.

### Isolation beyond `--config`

- Existing dedicated LOCALAPPDATA/APPDATA/USERPROFILE/TEMP/HF roots, offline HF and verified model hardlinks; current numeric4/12/1 seeded in the own profile. No private conversations or whole normal settings were opened.
- Preseeded `encrypted_file` secrets with a throwaway passphrase, not OS keyring; environment API/runtime overrides scrubbed. Telemetry disabled before first start; capture, overlay and clipboard remain off.
- Scene service starts unconditionally and choices can enumerate processes. The actual discovery boundary is replaced process-locally with **synthetic empty process snapshots**, and the profile/home roots contain no user scene logs. Unrelated audio/device inventory returned by choices is omitted from publication.
- OSC modeoff alone does not disable sender construction. Sender is deliberately directed to an owned bound loopback UDP sink, activation noticeoff. The two40-translation finalists each produced120 observed sink datagrams; no live user OSC destination was operated. This proves local destination receipt, not a physical GUI/VRChat presentation claim.
- Initial empty Soniox store caused terminal `settings_commit_success_runtime_degraded` /`stt.provider_not_ready`. [Original failure](results/app-old-cpu-app.json) is retained. Corrected probes use an explicitly authorized synthetic nonworking Soniox construction token only in the isolated encrypted store, with fail-closed **Python** non-loopback DNS/connect/connect_ex guard. No capture or credential verification is attempted; corrected traces record no blocked outbound attempt. There is **no Soniox readiness/verification claim**. This is an unused STT/network boundary, not translation substitution; it is not an OS-wide firewall for arbitrary native code.

### Slots, identities and disk restore

The real owner exercised en→ko slot0,ko→en1,en→ko0,ja→en2,de→en1,en→ko0,ko→en2(disk restore after eviction),en→ko0. Three slots are actually occupied; fourth-pair LRU eviction, retained A prefix and B reentry are observed. Thirty additional same-pair turns per cache0 finalist fill the real resolver's0,1,2,3,4,5 history entries. Native resources remain bounded in this shorter app window; the separate200-request soaks provide the longer resource observation.

Authorized **process-local identity injection** replaces only runtime module build/commit constants with the exact candidate identity, never maintained source constants. Baseline en→ko prefix key is `91c0e8283e668fcef62f6f5d060303721afad6f439910244dc374aaedfec7370`; candidate is `4691068f6f7c2cc4bec182191c3a0b7b4fd6dd9c0a8738f638a3f4cf6994adf7`. Own old cache files were copied into isolated candidate roots; actual lookup marks them ineligible and does not restore foreign KV. Candidate-produced cache is restored only by the matching candidate. **Unmodified runtime-root override still uses shipped constants and does not establish safe cross-version invalidation.** A pin/identity integration is required before a real rollout; the experiment does not fix that compatibility gap.

Restore transport success and useful prefix reuse differ:

| Fresh-process first translation | HTTP restore | Native checkpoint evidence | Cached/evaluated prompt tokens | Translation wall /prompt eval ms |
| --- | --- | --- | ---: | ---: |
| b10423 CPU default |true|No appended checkpoint restore observed|0 /843|3692 /3269|
| b10423 Vulkan default |true; n_read34508316,n_restored825|No appended checkpoint restore observed|0 /843|1292 /1121|
| b11537 CPU default |true|2 restored checkpoints|820 /23|537 /175|
| b11537 Vulkan default |true|2 restored checkpoints|820 /23|266 /95|

Baseline restore is **not** a lifecycle failure: it succeeds, and the following translation completes correctly after full prefill. It is not a prefix hit either. Initial baseline CPU `n_read` was not captured, so no byte count is invented for that row. New checkpoints explain a useful version effect independently of lazy PLE; [upstream033df86b](https://github.com/ggml-org/llama.cpp/commit/033df86b69ec1a333eb241f0c16325a4d43dcff5) changed slot-save checkpoint serialization.

For candidate cache0, matching input hash `f4f457dfeb29b9844f0eb9c47f2f84ced1fcd8c4625e4a79e0b59900a6c902db` is observed in the fresh-prefix and both fresh-process restart paths:

| Backend/path | Prefix prepare or restore wall ms | First completed translation wall ms | Prompt eval ms | Cached/eval/completion tokens |
| --- | ---: | ---: | ---: | --- |
| CPU fresh-prefix |3345.5 prepare|656.2|161.5|820 /23 /14|
| CPU restart1 |48.0 restore|589.7|193.7|820 /23 /15|
| CPU restart2 |46.8 restore|656.6|167.7|820 /23 /14|
| Vulkan fresh-prefix |826.2 prepare|274.9|90.2|820 /23 /15|
| Vulkan restart1 |30.8 restore|276.9|95.6|820 /23 /15|
| Vulkan restart2 |30.5 restore|273.7|93.1|820 /23 /15|

CPU restart n_read68311380, Vulkan68311364, each n_restored825 and **2 actual restored checkpoints**. No new prefix preparation is called after those restores. Thus the saved-prefix path avoids a separate3345ms CPU /826ms Vulkan first-prefix preparation component, while the first short translation is similar. These small n1/n2 app component observations exclude model loading, intervening CLI work and OS-cold behavior; they are not a universal startup multiplier. App sampling is stochastic and completion counts are retained rather than forced equal. Native evidence: [CPU restart1](results/app-new-cpu-cache0-cycle1-native-0.txt), [Vulkan restart1](results/app-new-vulkan-cache0-cycle1-native-0.txt).

All14 successful hosts exit0. Managed clients close, owner process/transport references clear, native children record controlled Windows termination exit1, and cache0 app GPU queries after native exit have **no matching process instances**, with null byte values. This is process/resource release evidence, not a global GPU-memory-zero claim. Followers/UDP sinks and corrected CPU load trees are closed. Initial failed host's terminal exit code was not captured in its cycle record; retained failure/cleanup trace is not promoted to a successful cycle. External public runtimes, full original logs and isolated synthetic state remain deliberately local for reproduction; no credentials/configuration binaries are published.

## 7. Failed/pruned branches, gaps and exact recommendation

Retained failures/limits are part of the result:

1. Initial MTP-off preflight attempted to override generated speculative arguments with an empty drafter; native still attempted target MTP and exited1. [Failure](results/new-cpu-greedy-off-failure.json), [producer](results/new-cpu-greedy-off-producer.txt), [exact arguments](results/new-cpu-greedy-off-run.json). Correct OFF construction removes speculative options entirely; clean12-case result is separate. This is a harness-configuration failure, not evidence of candidate inference instability.
2. Initial actual-app idle Soniox construction degradation, preserved rather than silently called applied. Corrected authorized unused-boundary setup is distinguished above.
3. Four default-policy RAM-guard censored soaks; no default96-pass or unbounded-leak claim.
4. Initial CPU load launcher-only accounting; separately repaired process-tree evidence, no use of false zero counters.
5. PLE ON no warm gain; true cold-OS-cache behavior **inconclusive/not run**. No admin cache flush or disturbance of live user workloads. b11630 only a placement diagnostic, not a second finalist.
6. No-repack is an unqualified preexisting low-RAM tradeoff, not dominated; FAon is non-selected from bounded development. Whole-model Q2, alternate runtimes/vendor backends, multimodal, precision conversions, unmerged direct-read patches, unrestricted threads and Vulkan MTP were not opened.
7. Per-operator PLE timings, page/hard-fault/prefetch-hit counters, isolated target verification cost, GPU utilization/peak GPU memory and pre-load M0 are unavailable/not collected. Producer buffer sizes are not physical-residency measurements. No other hardware, OS-cold SLA, GUI interaction/presentation, packaged-app build acceptance or arbitrary-prompt-length endurance claim is made.

**Immediate recommendation:** no shipped changes. Keep b10423 /`a94d563ed801d1da1b8c2432946de07d0231bb3d`, exact existing Q4/drafter, three slots/4608, current CPU4/12+MTP1/1,max/min3/2,FAoff and Vulkan4/4,FAon/full target offload,MTPoff. The report does not certify the existing8192MiB native cache as safe under every desktop workload; its observed pressure and ineffective old disk-prefix reuse are explicit product decision inputs.

**Small measured alternative, conditional rather than an authorized rollout:** b11537 /`8e2d31e0eb658d0382d3106a61369dff81f0c080`, same model/profile, `--cache-ram 0`, lazy defaultAUTO (inactive here), repack retained. Expected measured benefit is bounded native prompt-cache retention and effective same-runtime disk-prefix restoration, not a PLE gain or faster hot kernel. Costs are modestly slower cache0 hot turns than old cache0, about234MiB extra CPU /259MiB extra Vulkan process private commit at the end of the200-request windows, and larger saved checkpoint files (candidate en→ko about68.3MB versus baseline Vulkan34.5MB). CPU remains much more CPU-time intensive than Vulkan. Quality caveats and finite soak scope remain.

Any implementation decision must separately approve the native cache resource policy, integrate exact pin/build identity and full matching DLL/LLVM-license packaging, isolate/invalidate cache namespaces and preserve three-slot/LRU semantics, then perform normal product acceptance. **Rollback** means selecting the complete old runtime package with the old identity/namespace, retaining model hashes and existing settings; never feed candidate KV bytes to old runtime or vice versa. Do not copy only `llama-server.exe`, reuse foreign checkpoint files, or advertise the current unmodified override as safe integration.

This is a supported negative PLE/hot-update result with a separately attributable checkpoint benefit and a concrete resource-policy decision, not unfinished kernel development or a claim that newer llama.cpp is universally worse.
