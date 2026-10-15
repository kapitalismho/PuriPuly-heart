# Managed Gemma issue251 execution evidence

## E4B loading-mode decision: retain mmap

**Director-accepted decision: RETAIN the existing mmap default. No product default or tuning was changed.** The Director accepted the mixed small timing benefit, repeated working-set/commit/shared-GPU tradeoff, paging variability and identical semantic failures as insufficient evidence for a default change. This decision is not whole-issue acceptance or a blanket translation-quality pass.

The final comparable experiment is four fresh managed lifecycles in **mmap / none / none / mmap** order,44 completed translations each,176 total. Product candidate `8786d1fcd8fcb966f0ae164acee289e535a7c862` and Python-source-tree SHA256 `0c7fd7e654c939b1226c129157bde3cde103d4f4128de5dec66c910f821361b2` were identical in all four arms. Actual runtime was official b11537 on the RX7900XTX; the PDH adapter was discovered from native process instances, not assumed. Exact target/draft/runtime hashes are in [asset-identities.json](results/asset-identities.json).

Evidence: [per-arm records](results/e4b-abba-runs.json), [strict comparison](results/e4b-loading-comparison.json), [human quality review](results/e4b-human-quality-review.json). The exact successful five-module fixture is archived as [abba-harness-v3.zip](results/abba-harness-v3.zip), SHA256 `264632b58f2b858c1195bf902fb1c4011ff2f3ae2e1ed7935df380cef210c4d2`. Later safety-fixture changes do not replace this snapshot or enter its comparison.

### Controls and actual observations

- Same exact existing E4B profile in all arms except the approved task-local replacement of its existing load-mode argument. No26B tuning or new product switch was introduced.
- Actual outgoing and native-effective seed `2512026` was fixed only for comparison. All other original product sampling/output parameters were preserved and checked. All matched request prompt hashes, raw native echo hashes and effective sampling parameters agreed across arms.
- Fixed synthetic source history was seeded only between settled measurement groups through existing channel history APIs. The real resolver, prompt rendering, translation admission, provider and output owners remained active.
- Native adds one leading literal `<bos>` to the echoed representation. The exact remaining UTF8 bytes equal the outgoing prompt; both raw hashes are retained. This representation allowance does not modify the outgoing request or permit other normalization.
- **48/48 simultaneous pairs had positive native slot-processing overlap**: six same-language-pair and six opposite-language-pair pairs per arm. Native maximum active tasks, including prefix generation, was two, not three. One managed process/model served each arm; sampled `/slots` also reached two.
- Actual source/channel/turn IDs, native slots/tasks, completed application translation events and owned OSC destination packets were correlated. Positive Self chatbox delivery and22 explicit Peer chatbox denials were retained per arm. No observer failures or owner-history evictions occurred.
- All four arms exited successfully; all observed native processes exited after owned termination, server ports closed and zero fixture tasks remained. Windows native exit1 is paired with the explicit owned termination request, not described as an unexplained crash.
- Lifecycle start/end Unix seconds: A1 `1792084640.160798–1792084685.064802`; B1 `1792084685.065252–1792084729.859972`; B2 `1792084729.860347–1792084774.612678`; A2 `1792084774.613079–1792084819.399269`. These are supervisor lifecycle windows, not pure native startup or inference latency. Per-request monotonic timings/native prefill/decode are retained separately.

### Request latency and makespan

Milliseconds; median / nearest-rank p95. Pooled per mode across its two lifecycles. First-prefix requests are excluded from repeated-request groups and retained separately in the records.

| Language pair and measurement | n per mode | mmap | none |
| --- | ---: | ---: | ---: |
| Same pair, single request |8|485.4 /1030.5|488.4 /1014.5|
| Same pair, overlapping request |24|711.8 /1182.1|701.3 /1278.1|
| Same pair, overlapping pair makespan |12|737.2 /1273.2|734.6 /1281.1|
| Same pair, serial pair makespan |4|999.0 /1031.6|1002.8 /1007.9|
| Opposite pair, single request |8|460.5 /532.7|445.4 /518.8|
| Opposite pair, overlapping request |24|964.4 /1039.7|952.2 /1015.5|
| Opposite pair, overlapping pair makespan |12|1008.2 /1042.0|997.6 /1017.4|
| Opposite pair, serial pair makespan |4|977.7 /1019.4|982.6 /1026.2|

The two AB brackets show none's pooled overlapping-request median lower by23.0ms(2.67%) and21.1ms(2.44%); single-request median lower by10.9ms(2.32%) and17.6ms(3.66%). Serial-request changes reverse sign between brackets. Same-pair overlap p95 worsens under none despite nearly equal pair median. Native single-request decode medians remain approximately80tokens/s; same-pair overlap approximately60tokens/s and opposite-pair overlap approximately38–39tokens/s in both modes. These small, mixed benefits do not demonstrate a repeatable useful overall advantage sufficient to recommend changing the existing default.

Actual per-request cache/input/output counts and native prefill/decode durations/tokens-per-second are retained, not inferred from output character length. `timings.cache_n` is cached input; top-level `tokens_cached` is the slot's post-completion count. They are not interchangeable.

### Resource tradeoff

Decimal GB below. Process counters are not device-wide counters; both are retained separately. Steady medians are ranges across the two arms, not physical-capacity estimates.

| Process metric | mmap | none |
| --- | ---: | ---: |
| Dedicated GPU sampled highwater |2.855GB|2.858GB|
| Shared GPU steady median |0.804GB|2.391GB|
| Working set steady median |5.379GB|4.367–4.372GB|
| Working set sampled highwater |6.630–6.631GB|5.617–5.622GB|
| Private commitment steady median |5.623–5.627GB|7.220–7.230GB|
| Private commitment sampled highwater |6.876–6.882GB|8.472–8.483GB|

None repeatedly reduces process working set by approximately1.01GB, but increases shared-GPU accounting by approximately1.588GB and private commitment by approximately1.60GB. **Private commitment is not physical RAM, and shared-GPU bytes alone do not prove spill.** No claim is made that the counters can simply be added to obtain physical residency.

System available-physical minima were3.860–4.804GB for mmap and5.918–5.937GB for none; commit-headroom minima9.410–10.582GB versus9.075–9.113GB. Background paging varied materially, especially A2; pages-input/read rates and device-wide GPU counters remain in the original records. No owned memory-watchdog cutoff occurred. Sampled maxima are lower bounds on true highwater. Integrity hashing warms filesystem cache, so these fresh lifecycles do **not** establish cold-disk startup performance. Results on this24GB-class card do not establish an8GB-card, live VRChat or frame-time guarantee.

### Human quality: material failures preserved

All44 matched request positions produced identical outputs across all four arms. That is **no observed loading-mode quality difference**, not universal semantic correctness.

Human readback covered all13 unique source/target/output cards representing176 completions. Eight outputs—two per arm, opposite-pair overlap1 and5—reverse a corrected meeting time:

> Source: 정정할게요. 회의는 목요일13시40분이 아니라 화요일14시30분이에요.
>
> Output: Let me correct that. The meeting isn't on Tuesday at14:30, it's on Thursday at13:40.

The source affirms Tuesday14:30 and rejects Thursday13:40; output does the opposite. Automatic word/number checks missed this directional error. The exact spaced original source and every result ID are preserved in the quality JSON. Cause is undiagnosed; this is not attributed to loading mode, application routing, native caching or model quality alone.

Another eight outputs render Korean `표` as blue signs instead of the fixture's intended tickets. The source noun is ambiguous; this lexical concern remains explicit. Seven/nine/three are correctly spelled out, so digit-surface flags do not establish lost quantities. Other cards preserve names, quantities, negation and destinations; minor register, verb and emphasis variations are documented rather than erased. No input/history/sampler was retuned to hide any outcome.

### Failed fixture attempts retained

[Attempt1](results/abba-attempt1-failure.json) and [attempt2](results/abba-attempt2-failure.json) each completed a real native startup-prefix request but zero application translations, then failed an over-strict raw-echo equality assertion. Attempt1 did not journal the native response before assertion; attempt2 did, allowing diagnosis of the native-added single leading `<bos>`. Seed2512026 was confirmed. Native `add_bos_token=true` metadata supports the representation explanation. The Director approved the bounded exact-remainder check. All four arms were restarted fresh as attempt3; no failed or pre-correction record was pooled into comparison. Both failed lifecycles released their native processes/ports and tasks.

## Remaining matrix: exercised results and candidate failures

| Original8786 case | Actual outcome | Durable evidence |
| --- | --- | --- |
|26B fixed profile plus safety|Exit0;69 recognition-request completions and76 actual native translation completions retained. Native maximum2 on26B GPU,1 after transition to E4B CPU/MTP,2 after return to E4B GPU. Actual ordering, both-direction cancellation, injected response rejection/survivor, slot reuse, manual capacity and active shutdown were exercised.|[Execution](results/26b-execution.json),[audit](results/26b-audit.json)|
|E4B CPU/MTP|Exit0;44 completions, native maximum1,1,566 native draft tokens; detected8C/16T yields target/batch4/12 and draft1/1.|[Execution](results/cpu-execution.json),[audit](results/cpu-audit.json)|
|Injected E4B Vulkan failure|Exit0; one injected pre-spawn rejection followed by actual product-owned CPU/MTP fallback,44 completions, maximum1 and1,566 native draft tokens.|[Execution](results/fallback-execution.json),[audit](results/fallback-audit.json)|
|Injected26B Vulkan failure|Original assertion failed. No native/fallback process spawned; managed26B/GPU state failed. Effective translation incorrectly remained enabled after warmup failure. The historical assertion also incorrectly interpreted provider-binding availability as native readiness; its complete failed record remains preserved.|[Original](results/26b-no-fallback-attempt1.json),[audit](results/26b-no-fallback-audit.json)|
|E4B safety attempt1|70 recognition-request completions before failure. CPU-to-GPU settings intent committed under inactive-STT degradation, but actual CPU/MTP remained. Two-active GPU shutdown assertion timed out; not a GPU-return pass.|[Original](results/e4b-safety-attempt1.json),[audit](results/e4b-safety-attempt1-audit.json)|
|E4B supported recovery attempt2|66 recognition-request completions before failure. Actual off/on commands both applied and prepared a new Vulkan child, but first Self/Peer work failed and the GPU exited. No post-recovery native translation completion; timeout retained.|[Original](results/e4b-safety-attempt2.json),[audit](results/e4b-safety-attempt2-audit.json),[diagnosis](results/recovery-diagnosis.json)|
|Supported stock sourceCLI|Two distinct fresh hosts; actual E4B manual translation, explicit26B selection, persisted selection readback and actual26B manual translation after restart. Receipt IDs joined to public translation events and owned OSC packets. Both stop commands applied; all three observed native children released.|[SourceCLI proof](results/supported-source-cli.json)|

The26B shutdown record includes legitimate progress during ordered shutdown: one Peer translation completed and was published, while the waiting manual native result was not published. Do not substitute “zero publication after close was requested” for the observed lifecycle contract. All explicitly cancelled IDs and the injected rejected-response IDs remained unpublished throughout each complete journal; owned processes, ports and fixture tasks drained.

All additional native outcomes—including rejected/cancelled native completions and manual/shutdown outcomes not counted by recognition request observers—are retained in the audit files and [extra-native quality record](results/matrix-extra-native-quality.json). Human readback of175 distinct cards representing293 recognition-request completions found **seven additional factual correction errors**, including wrong rejected weekday and reversed/changed meeting times, plus ten lexical concerns. Counts include legitimate progress inside failed attempts, not additional passing lifecycles. See [all cards](results/matrix-quality-cards.json) and [human review](results/matrix-human-quality-review.json). The two stock sourceCLI outputs preserve the fixed synthetic facts; no blanket quality pass is made.

### Adjudicated readiness contract and671 after-proof

`app.status.translation_runtime_ready` intentionally reports **provider binding**, and was already true while translation was disabled. The actual original enable defect was `translation_enabled` remaining true after managed warmup failed; its failed record remains unchanged. Replacement protection was exercised successfully at frozen671: enabled=false, provider binding=true, exact managed26B/GPU failed, actual `owner.runtime.readiness is None`, zero native/fallback children, then explicit unchanged-selection retry with six actual managed/native completions. This preserves operational/native readiness protection rather than dropping the old assertion.

The original E4B failure was a separate partial-apply signature problem: LLM rebuilding preceded STT convergence, but signature publication followed it. Successful CPU replacement plus STT failure could leave the cached signature at GPU, skipping the next provider replacement; connection-only changes also skipped managed-prefix refresh. Canonical enable demand could prepare a GPU child while the provider remained CPU-bound. Source paths and limits of the original rejected-argv observation remain in the [diagnosis](results/recovery-diagnosis.json). Actual671 after-proof below now demonstrates effective GPU rebinding through repeated completions/overlap and same-child two-active shutdown with admitted manual work; original failed attempts remain failures.

The final clean signature-writer/initial-binding migration was frozen as `671ca2ac68a184a2d6915e9efd36464b52d5b0f5`. The intentionally cancelled7cff native build is not a model/product failure or new installed proof. After the Director's review/integration barrier and explicit source-first GO, both targeted source-native cases completed and released hardware. The later final671 full build, isolated installation and nonmodel checksum staging passed. After the severe-memory warning, the user explicitly skipped latest installed actual inference; no installed control-host/model probe or prepared watchdog ran, and that criterion remains unverified.

Both lifecycles captured exact671 HEAD/import and identical source-tree SHA256 `cbd004e0bd448c1b6efb20f0836b247a5bb0c116adf3b5d736a878a81eb496c5`, identical fixture hashes and pinned b11537/exact asset identities. No original setup, completed matrix or ABBA run was repeated. Attempted argv was retained before the unchanged strict profile guard. Detailed typed receipts and guarantees: [afterproof-evidence.json](results/afterproof-evidence.json); executed protocol: [ledger](EXECUTION.md#executed-at-frozen-product671).

The exact executed seven-module after-proof fixture is [archived](results/afterproof-harness-671.zip), SHA256 `6412fa95e7b99276cbc6280d14d6cd2e527c3dbf8aff412f2ab76ceb8df49307`. Later Black-only formatting preserves every module's parsed AST; native observations and their original fixture hashes are unchanged. [Repair verification](results/evidence-repair-verification.json) also records the corrected quality-provenance field: its values identify original journals, not audit JSON bytes.

|671 source-native after-proof|Actual observation|
|---|---|
|26B rejected spawn and same-selection retry|Exit0; failed spawn left effective off/provider bound/native not ready and zero native/fallback children. Same26B/GPU retry launched exact child69864 and completed6 native/application translations, maximum2. Complete fixed profile and pre-guard argv retained in [summary](results/26b-afterproof.json)/[audit](results/26b-afterproof-audit.json).|
|E4B full workload/safety|Exit0;70 recognition completions and78 native translation completions. Actual GPU50884→CPU/MTP72040→GPU69380, native maxima2/1/2. Seven independent inactive-Soniox STT degraded receipts remain degraded; both backend settings commits were among them. No extra off/on recovery was needed. [Summary](results/e4b-afterproof.json)/[audit](results/e4b-afterproof-audit.json).|
|Returned GPU progress and shutdown|Two Self/Peer pairs completed4 native translations on retained GPU PID69380/slots0–1; native overlaps1.036750s and1.289447s. Same PID then had2 fresh active native requests plus admitted waiting manual work during shutdown. Both channel-order, cancellation/survivor/reuse, response-fault survivor, manual capacity, prefix LRU and transition paths completed.|
|Release|All4 native children recorded owned exits;4 ports closed, both task registries0 and supervisors completed. One legitimate Peer result published during ordered shutdown; waiting manual native result remained unpublished. No unrelated process was terminated. [Release](results/afterproof-release.json).|
|New quality limitations|59 distinct cards representing76 application completions and8 additional native outcomes all read. E4B produced2 factual errors: blue boxes substituted for Korean표, and a reversed corrected meeting time. Two separate lexical concerns: signs/cards versus intended tickets. All6 retry outputs and4 returned-GPU outputs preserve their tested facts; no blanket quality pass. [Full review](results/afterproof-human-quality-review.json).|

These after-proof outputs are separate unforced-sampling observations, not a new loading-mode comparison or explanation of original semantic errors. Original8786/ABBA attribution, failures and exact quality counts remain unchanged.

### Pressure and release

[Resource pressure analysis](RESOURCE_PRESSURE.md) attributes the26B12,550,144B physical minimum to loading, not steady inference; CPU/fallback lows occurred in active/settled groups. It gives below256/512MiB spans and surrounding timestamp bounds, sampling gaps, global paging and process/device accounting. The true prelaunch baseline was not captured; earliest postspawn proxies are not mislabeled. No eight-second watchdog cutoff occurred, which is not a RAM-safety guarantee.

Original cleanup records show exit/closed ports. A single later read-only inspection found all33 recorded owned PIDs absent and all19 recorded ports closed: [final-cleanup.json](results/final-cleanup.json). No unrelated process was terminated. Models/GPU/CPU resources are released; raw original records remain in isolated temporary storage.

New671 [pressure evidence](results/afterproof-memory-pressure.json) captures4 true prelaunch observations, separate from the missing original baseline.26B prelaunch available physical was5,045,506,048B with9,941,684,224B commit headroom; loading reached **155,648B available physical**, a material depletion. Below256MiB sampled span5.385011s/outer5.914263s; below512MiB5.649810s/outer6.178412s; maximum sample gap0.342108s. No8s watchdog cutoff occurred, but this is not RAM safety or an OOM/spill diagnosis. New E4B phase/PID minima, global paging and process/device-separated accounting are in the pressure record/report. Owned integrity hashing warmed filesystem cache; Python prelaunch-host memory contribution was not sampled, and no private/live-process investigation or causal attribution was performed.

Initial native load/control startup and first-channel-prefix observations are separated in [e4b-startup.json](results/e4b-startup.json), not pooled into repeated-request medians.


## Separate boundary proofs

- [Original distribution/build/installed proof](results/distribution-evidence.json): supplied by the runtime-distribution owner at frozen8786, including full native build and two fresh isolated installed26B completion cycles. This is not installed CPU, concurrency or GUI inference proof, nor fresh installed proof for current671.
- [Final distribution and authorized omission](results/distribution-final-evidence.json): actual671 full build,552 matching packaged source hashes, genuine isolated installation and exact26B nonmodel checksum staging. Latest installed native inference was explicitly user-skipped after the pressure warning; zero current installed inference completions. Both unique test installations/registrations were removed; bounded permission-denied temporary remainders are disclosed.
- [GUI selection/provisioning proof](results/selection-evidence.json): supplied by the selection owner; actual GUI selection, provisioning/cancellation and restart boundaries. This is not model inference or benchmark evidence.
- [GUI model-resource cleanup](results/gui-resource-cleanup.json): the task-created14.25GB product copy and separate HF cache hardlink were unlinked after completed proof. The original issue245 model identity/size/mtime and original manifest/log/screenshot records remain; no fresh checksum, memory improvement or new readiness claim is inferred.
- [Enable rollback repair record](results/enable-repair-evidence.json): separate owner's48 targeted tests and controlled composed-control smoke. Its native process/model/transport boundaries were controlled; it is not actual Vulkan inference or quality/resource proof.
- [Original provider signature repair record](results/provider-repair-evidence.json): separate owner's277 targeted tests and controlled consumer/composition evidence. It does not itself replace native proof; original failed E4B attempts remain failures and actual671 source after-proof is recorded separately above.
- [Final signature-ownership repair evidence](results/signature-ownership-evidence.json): separate owner's473 affected tests/controlled composed smoke plus post-pin source hashes, with its pre-pin execution versus later hash-capture provenance preserved. This native source after-proof separately records actual in-run671 identity.
- [Native-free enable GUI evidence](results/enable-gui-evidence.json): actual Flet TRANS failure/cancel/explicit retry at exact7cff, with that owner's limits. It is not relabeled as671 GUI execution or native inference proof.
- [Execution ledger](EXECUTION.md): functional smokes, original safety/CPU/source-control outcomes and completed671 source after-proof are tracked independently. Final separate-owner build/installed boundaries and issue acceptance are not inferred from comparison completeness.
- [Portable product verification](results/verification-evidence.json) and [evidence repair checks](results/evidence-repair-verification.json): preserve exact test/style/review boundaries, original failures, CI-wide Black correction, unchanged harness semantics and original artifact identity.
- [Completion record](COMPLETION.md): implementation links, effective profiles, user-visible changes and the explicitly unverified latest-installed criterion. No issue closure, push, merge or release is implied.
