# Issue353: fresh EN/JA CPU comparison

## Decision

- **English: promising replacement on this sample, not a production cutover.** Multilingual Granite Q8_0 reduced clean and paired noisy WER, used substantially less measured process-tree RAM, and had similar natural-turn final latency without queue growth. Short-clip decode was slower; long-clip/tail performance was favorable on this run. Retain current production routing until a separate integration decision addresses the raw no-speech-marker policy and broader accuracy requirements. Memory pressure prevents a strong absolute speed claim.
- **Japanese: retain the current sherpa CTC INT8 path.** Granite's lower RAM/download size did not compensate for worse clean/noisy surface CER, wrong-language short responses, and slower inference/replay. This is evidence about these model/runtime combinations and frozen conversational samples, not universal superiority.

All740 primary calls and eight real120s replay runs completed. No production sources, routing, model catalog, settings or behavior were changed. No historical result is presented as a fresh baseline. No alternate runtime build or converter was necessary: the pinned candidate loaded and decoded on the existing app worker.

## Execution identity and limits

Checkout/import: `96a0d2cfb32a7a05136e510eebe55ec28431bcb4`, `src/puripuly_heart/__init__.py`, local locked CPython3.14.10 environment. Host: Windows11 build26300, AMD Ryzen7 9800X3D,8cores/16logical,33,727,868,928bytes RAM. Run metadata records the actual import, artifact SHA256/size, source hashes, package versions and native DLL/executable hashes.

- Existing staged worker build `da3ad834a7f144dd981a4c75f99d4b13`; worker SHA256 `52ae3f80050d54bf3a9d73bc220ce44495f178f93c8c395813a141cc1451492e`, transcribe DLL `a0ffb59f75d6291c770c268f8ef880555a90dded161bd88628419c60ee45bc8d`; app transcribe.cpp pin `8facde6fcd8fb9caa585a5788ab450df6e8ec7db`. CPU activation was observed, not inferred from the presence of Vulkan DLLs. The selected individual dynamic CPU kernel was not separately instrumented.
- sherpa-onnx/core1.13.8, ONNX Runtime1.28.0; actual Python native-library hashes are in each main run's metadata. Japanese uses `OfflineNemoEncDecCtcModelConfig`, CPU,2threads,80feature bins,greedy_search—not TDT.
- [Model configurations](../models.json) pin all artifacts. Granite is the multilingual `17slever17` revision `62f64d9d5ad35989b06caf199cafe7155944a22b`, SHA256 `f3efdc05b85b3a764bb5bb7022e69f61cafcecc11605c881c6396a6ed07ec91b`. Q8_0 means8-bit weight quantization, not an all-INT8 arithmetic guarantee or ONNX INT8 equivalence.
- CPU only,2requested inference threads,batch1,resident model/session,serial candidates. No affinity/thread sweeps. No prompts, vocabulary, context carryover or external correction. Native ITN follows the unchanged app defaults when advertised by the model. Granite language hint is unset and detected language remains unavailable.
- Preloaded mono16k PCM16 bytes→app `decode_pcm16le`→returned text is the primary clock. GGUF includes PCM/float conversion, requantized temporary WAV/write/read and authenticated IPC; sherpa includes conversion and the resident in-process owned-thread CTC call. The existing GGUF writer scales by32767, so its internal requantization is retained, not silently replaced by a new decoder path. Native GGUF decode clocks exclude file/IPC. Median outside-native time was8.04ms Orukeet,8.17ms EN Granite,7.45ms JA Granite. No native-only sherpa substage was instrumented.
- Initialization and first decode are separate; then two unmeasured warmups and one measured pass. Full artifact hash verification precedes timing and warms filesystem caches: initialization is **not cold-disk startup**. First/warmup transcripts/times remain in metadata. No significance test or exhaustive listening review was performed.

### Memory pressure and resource measurement

Primary available RAM before loading was1.29GiB EN Orukeet,1.70GiB EN Granite,1.55GiB JA sherpa,1.71GiB JA Granite. Initial JA smoke had11.48GiB available. Preserve the original results: no timing rerun or survivor filtering was performed. There is no evidence assigning this pressure to preparation; its finite large-array subprocesses had ended before the gate, and the data owner subsequently released only owned metadata/HTTP kernel state. No unrelated process was stopped.

Replay available RAM before each model pair was4.83GiB EN Orukeet,4.82GiB EN Granite,6.59GiB JA sherpa,7.11GiB JA Granite. A [during-replay snapshot](memory-observation.json) recorded3.29GiB available and3.27GiB swap used. Windows swap-in/out zeros do not establish absence of paging. Per-process hard faults/paging and external CPU contention were **not measured**. Accuracy and the clear JA direction remain useful; small speed differences must not be generalized from this host state.

Resource sampling uses existing psutil APIs every50ms for the benchmark process and recursive inference children. Peak tree RSS is a sampled sum, not PSS/unique memory and not a guaranteed instantaneous maximum. Metadata also retains the sum of each process's OS peak working set (an upper bound, not simultaneous peak). Parent interpreter/preloaded PCM and observer overhead are included. CPU seconds sum user+system process time including the native child; Windows counter granularity applies. The table's CPU totals cover measured185 calls only; metadata separately retains whole-run totals including load/warmups. No resource errors were observed.

## Inputs, accuracy and failure behavior

[Preparation evidence](../data-preparation.json), [frozen inputs](../inputs/) and [rerun/access instructions](../README.md) are part of this result. English is the approved **reconstructed** pinned AMI IHM test120,40/40/40 duration mix,4meetings/12source-speaker labels, using human/Kaldi-normalized references. Exact individual-headset recovery failed403; it is not the exact #308 follow-up selection. The genuine English replay uses the original Mix-Headset window32.368–152.368s, not the old mean of4headsets; shared ends were recomputed. Japanese reuses the recovered120 selection and human EAF references:108short/12medium,5conversations/10conversation-local tiers, not proven globally distinct people. Its genuine Zoom window86.04–206.04s is preserved.

Each cell uses identical frozen bytes/references within language:120clean +30pink/fan15dB +30pink/fan5dB +5silence/noise controls. The original seeded active-speech noise recipe, clipping prevention, PCM16 conversion and source/license pins are documented by preparation. AMI training overlap is unknown (IBM lists AMI among its training sources); IngCrowd is unsplit. No model output became ground truth.

Pooled EN WER / JA CER use NFKC+casefold and UnicodeL/N/M retention equally on references/hypotheses. EN punctuation becomes word separators; JA punctuation/whitespace disappear. No kana/kanji/digit-equivalence rewrite hides orthographic errors. Empty speech stays in the denominator; failed/timeout speech would be retained, enumerated and scored as empty hypotheses rather than dropped. All740 statuses were `ok` (not all transcripts correct).

| Cell | Clean120 | Paired clean30 | Noise15 (30) | Noise5 (30) | Empty speech / failed / timeout | Raw nonempty controls / lexical false |
|---|---:|---:|---:|---:|---:|---:|
| en-orukeet | 9.44% | 8.54% | 10.37% | 24.09% | 1 / 0 / 0 | 0 / 0 |
| en-granite | 6.75% | 5.49% | 7.62% | 15.55% | 0 / 0 / 0 | 5 / 0 |
| ja-sherpa-ja | 14.65% | 13.56% | 15.37% | 22.97% | 1 / 0 / 0 | 5 / 4 |
| ja-granite | 22.16% | 20.07% | 22.06% | 27.49% | 0 / 0 / 0 | 5 / 0 |

Clean denominators:1260EN words and1584JA characters; paired/noisy denominators328EN words and553JA characters each. Paired clean30 is the only appropriate comparator for each noisy set. [Saved summary](summary.json) contains pooled edit counts, all clean/noisy duration groups, status counts, timing/RTF distributions and failed IDs.

Raw nonempty control output is **not automatically lexical invention**. Granite returned `<|nospeech|>` for all5controls in each language:5raw outputs,0lexical inventions. These are unsuppressed native adapter results and an integration-policy gap, not decoded conversational words. Sherpa returned `あっ`, `。`, `うん`, `うん`, `うん。`:5raw nonempty,4lexical false outputs,1punctuation-only. Orukeet returned5empty controls. [Control review](control-output-review.json) preserves every classification/output. Granite also returned `<|nospeech|>.` on one JA speech row; that row remains in CER, separately flagged, rather than being silently treated as empty/successful speech. Orukeet and sherpa each had one empty speech row.

[Ten inspected error examples](error-examples.json) retain full reference, both raw outputs and edits. EN examples include Granite losing the negation in NOT AN OPTION→another option, a5dB design-specification omission/corruption, Participant three recovered by Granite but not Orukeet, omitted number context and company-logo corruption in both. JA examples include 確かに→That's funny/That's fine/That's one under Granite, repeatedうん→no-speech marker, 見習おう→ミラーを, frequency/step-count errors and visibleなんか/何か・色々/いろいろ/色いろ distinctions. This is reference/output inspection, not audio reannotation. Granite also preserves some fillers omitted by sherpa; rejection is based on the pooled and latency direction, not cherry-picked failures.

## Primary speed and resources

| Cell | Download bytes | Init / first (ms) | Warm p50 / p95 / max (ms),185 calls | Measured CPU seconds | Peak process-tree RSS (MiB) |
|---|---:|---:|---:|---:|---:|
| en-orukeet | 739,508,608 | 259.8 / 179.6 | 304.4 / 800.1 / 1069.3 | 136.59 | 1235.0 |
| en-granite | 505,663,136 | 96.0 / 229.2 | 314.6 / 621.9 / 908.2 | 130.88 | 621.2 |
| ja-sherpa-ja | 655,571,161 | 1946.3 / 162.6 | 99.2 / 233.8 / 340.4 | 49.12 | 880.2 |
| ja-granite | 505,663,136 | 103.6 / 419.5 | 230.6 / 486.1 / 668.0 | 103.48 | 606.0 |

| Cell | Clean duration (n) | PCM→text p50 / p95 / max (ms) | RTF p50 / p95 / max |
|---|---|---:|---:|
| en-orukeet | 0.5-2s (40) | 123.3 / 183.2 / 188.9 | 0.114 / 0.147 / 0.152 |
| en-orukeet | >2-5s (40) | 303.9 / 401.4 / 432.9 | 0.091 / 0.096 / 0.100 |
| en-orukeet | >5-15s (40) | 589.6 / 876.0 / 1069.3 | 0.086 / 0.090 / 0.094 |
| en-granite | 0.5-2s (40) | 183.3 / 222.6 / 246.1 | 0.173 / 0.316 / 0.338 |
| en-granite | >2-5s (40) | 312.9 / 385.9 / 391.0 | 0.091 / 0.113 / 0.116 |
| en-granite | >5-15s (40) | 503.3 / 779.4 / 908.2 | 0.075 / 0.084 / 0.086 |
| ja-sherpa-ja | 0.5-2s (68) | 68.4 / 92.5 / 98.2 | 0.063 / 0.093 / 0.113 |
| ja-sherpa-ja | >2-5s (40) | 131.3 / 181.2 / 189.7 | 0.044 / 0.050 / 0.052 |
| ja-sherpa-ja | >5-15s (12) | 228.4 / 307.2 / 340.4 | 0.037 / 0.039 / 0.039 |
| ja-granite | 0.5-2s (68) | 179.9 / 221.2 / 238.2 | 0.172 / 0.298 / 0.322 |
| ja-granite | >2-5s (40) | 290.4 / 368.5 / 384.2 | 0.098 / 0.114 / 0.119 |
| ja-granite | >5-15s (12) | 478.4 / 621.3 / 668.0 | 0.078 / 0.091 / 0.097 |

All cells completed185calls. The all185 speed table includes clean,noisy and controls; the duration table is clean only. Full noise15/noise5 duration p50/p95/max and RTF remain in `summary.json`. Percentiles use linear interpolation at(n−1)q. GGUF native decode-only and outside-native distributions are also saved; do not compare native-only with another path's PCM→text clock.

## Eight independently paced natural-conversation replays

The producer admitted512-sample chunks independently at1× wall pace; decode never gated the next admission. Every120s window's ingress finished in120.008–120.018s. The runner reused current Silero Self onset0.3/exit0.2,nominal500ms hangover/500ms preroll, then **actual scoped LocalOfflineSTTSession** with current FIFO pendingTTL12s,30s seal watchdog and queued-work preservation after scoped failure. No #308 legacy failure semantics were imported. Models remained resident for clean+noise15.

The endpoint output enters a small experiment terminal consumer queue. This bypasses capture-owner pre-provider retention/ingress limits and downstream translation,TTS,GUI,overlay/network output: **not whole-app GUI latency**. First-partial latency is N/A for these final-only paths. Shared clean-source energy estimates within human annotations—not file end/enqueue and not acoustic ground truth—anchor response latency. They can be biased by overlap/background. Replay decode_ms is the current session float32→text call; file/IPC remains combined for GGUF.

| Replay | Delivered / empty | Segments | Source ends with nonempty final / total | Source-end→consumer p50 / p95 / max (s) | Segment-final p50 / p95 / max (s) | Recovery after120s (s) |
|---|---:|---:|---:|---:|---:|---:|
| en-orukeet-clean | 25 / 0 | 25 | 32 / 33 | 2.36 / 9.59 / 10.50 | 1.17 / 2.93 / 3.30 | 0.14 |
| en-orukeet-noise15 | 16 / 1 | 17 | 26 / 33 | 2.83 / 19.63 / 22.37 | 0.91 / 2.63 / 2.69 | 0.12 |
| en-granite-clean | 25 / 0 | 25 | 32 / 33 | 2.34 / 9.49 / 10.40 | 1.17 / 2.87 / 3.23 | 0.19 |
| en-granite-noise15 | 17 / 0 | 17 | 27 / 33 | 2.55 / 19.14 / 21.92 | 0.93 / 2.45 / 2.55 | 0.20 |
| ja-sherpa-ja-clean | 6 / 0 | 6 | 49 / 49 | 13.13 / 39.75 / 42.56 | 1.51 / 2.38 / 2.49 | 1.78 |
| ja-sherpa-ja-noise15 | 10 / 0 | 10 | 48 / 49 | 6.01 / 14.94 / 19.59 | 1.19 / 3.89 / 5.91 | 0.44 |
| ja-granite-clean | 6 / 0 | 6 | 49 / 49 | 13.85 / 40.64 / 43.45 | 2.08 / 3.19 / 3.38 | 2.68 |
| ja-granite-noise15 | 10 / 0 | 10 | 48 / 49 | 6.25 / 15.49 / 20.16 | 1.65 / 4.24 / 6.14 | 0.87 |

All eight runs: failed0,expired0,dropped0,replaced0,queue-merged0. Max pending including active was1; pending at ingress end1 and after bounded recovery0. No backlog stop, watchdog orTTL event was triggered, so those failure/recovery branches are reused policy rather than exercised failure claims. Stop guards are three failed calls or>12pending,45s drain; they were not needed. One EN Orukeet noise segment was empty; it remains visible alongside16nonempty deliveries.

The source-end column means an estimated source end fell within input of a segment that later produced a nonempty final. It does **not** establish lexical delivery/whole-turn coverage. Clean EN input-end coverage32/33; noisy27/33, with26/33nonempty for Orukeet and27/33for Granite. JA clean49/49,noisy48/49. No-source-end segments still remain in delivered counts, not latency survivors: segment-end latency has15/25ENclean eligible samples,10/17Orukeet-noise,11/17Granite-noise,6/6JAclean,10/10JA-noise.

Long source-turn tails are important: the endpoint groups multiple annotated turns even without a decode backlog. All6JA clean segments cover multiple source turns; source-end p95 is39.75s sherpa/40.64s Granite, not the much smaller latency from each segment's last estimated speech end. EN noisy source-end p95 is19.63/19.14s. Endpoint grouping is not coordinator queue merging. This result does not claim that stable queues imply timely individual conversational turns.

| Replay | Endpoint p50 / p95 (ms) | Queue p50 / p95 (ms) | PCM/float decode p50 / p95 / max (ms) |
|---|---:|---:|---:|
| en-orukeet-clean | 772.0 / 2342.9 | 0.047 / 0.068 | 259.7 / 954.7 / 1197.0 |
| en-orukeet-noise15 | 663.8 / 2040.5 | 0.053 / 0.086 | 316.8 / 858.0 / 1976.9 |
| en-granite-clean | 772.0 / 2344.4 | 0.051 / 0.063 | 276.8 / 868.4 / 905.9 |
| en-granite-noise15 | 659.3 / 2042.8 | 0.054 / 0.074 | 341.7 / 708.3 / 1543.9 |
| ja-sherpa-ja-clean | 654.9 / 1002.1 | 0.063 / 0.070 | 823.8 / 1692.1 / 1882.9 |
| ja-sherpa-ja-noise15 | 709.7 / 3461.6 | 0.050 / 0.072 | 401.6 / 675.9 / 707.2 |
| ja-granite-clean | 655.3 / 1003.6 | 0.045 / 0.059 | 1315.4 / 2541.3 / 2776.4 |
| ja-granite-noise15 | 704.5 / 3461.4 | 0.047 / 0.066 | 811.8 / 1183.2 / 1286.7 |

Terminal producer→experiment consumer queue p95 was<0.026ms in every run. Raw replay JSONL retains source segment extents, grouped source IDs, all texts/status, enqueue/producer/consumer times and native timings; per-replay summaries retain the full queue timeline and source-end coverage/latency observations. The windows are too small to claim general endpoint quality, but they distinguish decode throughput from endpoint-delayed finals.

## Reuse and evidence index

The CLI defaults smoke to3clean clips; explicit `--limit` overrides row selection. Primary still defaults to all185. A final no-limit EN Granite smoke exercised exactly3distinct clean rows and the unchanged app-native path; its raw/meta files are in [reuse-smoke](reuse-smoke/). The offline scorer then regenerated `summary.json` from saved rows and replay summaries:4cells,740primary rows,8replays,without importing/loading a model. No test framework, CI, UI or runtime conversion project was added.

Exact portable rerun examples and licenses are in [README](../README.md). [Execution evidence](../execution-evidence.json) indexes commands, hashes, outcomes and limits. Complete per-call transcripts/status and run metadata are in this directory (`*-primary.jsonl`, `*-primary-meta.json`, `*-replay.jsonl`, `*-replay-summary.json`, `*-replay-meta.json`, `*-smoke*`). The only post-primary CLI semantic change bounded default smoke; the primary/decode path was not rerun or changed. The replay EOF adapter was corrected against the actual SpeechEnd-or-None API before any replay run. Scoring additions expose native-boundary and raw-marker evidence without changing normalization or hypotheses.

Publication/integration/review remain with the Director. This experiment proposes no production cutover; the follow-up decision is whether EN's sample-specific gain merits an integration task with explicit no-speech marker policy. No further measurement is justified here absent a concrete decision-changing anomaly.
