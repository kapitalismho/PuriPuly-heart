# #204 Soniox continuous-to-segmented gap: decision and evidence

## Decision

**Defer production adoption; retain current behavior.** The authorized investigation reproduced the corrected baseline, completed a four-case A–D factorial (16 calls), completed a two-case S200-only ablation (2 calls), and exercised the real application consumption owners with model-free inputs. It did not establish a production-safe quality–latency operating point or a supported timestamp correction. This is an evidence-backed **inconclusive operating-point disposition**, not rejection of every possible intervention and not a claim that every issue completion criterion passed.

- Frequent manual finalization increased SDK-declared DER in all eight matched-PCM contrasts observed. Conditional native recognized-word identity did **not** uniformly worsen with DER.
- Removing S200 improved overall attribution on `brief_interjections`, but reduced recognized-word count and correctly attributed short-interjection coverage. It did not repair the returning-speaker failure. Blanket S200 removal is not recommended.
- Fewer finalizations expose a quality/arrival tradeoff: complete final-word evidence arrived later after source transmission. Current peer publication is terminal-driven, so continuous recognition requires an explicit ownership/completion design, not a configuration toggle.
- Native clock displacement is observed, including actual processed-audio counters, but independently verified acoustic anchors and a protocol-supported correction are unavailable. Literal native timing is retained.

No production source, endpoint/padding defaults, session lifetime, cancellation, native-only attribution, translation correspondence, or #190 first-readable color behavior changed. No merge, push, deployment, issue closure, private recording, observer stream, fallback provider, or rented compute was performed. No architecture drift was introduced. User-visible application behavior is unchanged; the new surface is the isolated experiment and its reports.

## Authority, revision and reproducible inputs

Authority: [issue #204](https://github.com/kapitalismho/PuriPuly-heart/issues/204). Production/execution baseline: [`4dc46fdac8c771d3bb58e46bd44304b86fde9c47`](https://github.com/kapitalismho/PuriPuly-heart/tree/4dc46fdac8c771d3bb58e46bd44304b86fde9c47), clean `soniox_diar_test`, equal to its remote at investigation start. The offline research checkpoint is [`e8739a267a14a4688498f32cda3d41bd641e6615`](https://github.com/kapitalismho/PuriPuly-heart/commit/e8739a267a14a4688498f32cda3d41bd641e6615). These local research commits are not claimed published. The live checkpoint is the commit containing this report and the frozen artifacts linked below; its exact final SHA belongs to the delivery/review record rather than being retroactively written into execution receipts.

Obtainable inputs, not personal paths alone:

- [Benchmark reproduction instructions](../../../README.md#reproduce-locally), [manifest](../../../manifest.json), and [attribution/license](../../../ATTRIBUTION.md) provide public AMI audio sources, pinned mirror revision, fetch/build commands, clip windows and identities. Raw audio remains untracked.
- [Public word/turn inventory](ami_word_reference_inventory.json) records pinned acquisition URLs, hashes, release and licensing for the retained `ami_words/` XML. Published annotation times are not independently audited waveform clock anchors.
- [Offline evidence](offline_evidence.json) records seven WAV/PCM identities, source maps, 28 native receipt identities, SDK projection reproduction, segmentation/configuration and runtime versions.
- [Original plan](factorial/run_plan.json), [amended plan](factorial/run_plan_v2.json), [configuration](experiment_config.json), [frozen traces](factorial/traces/), and [sender history](factorial/history/) preserve exact byte maps, model/configuration, cut/control positions, executed sender hashes and the pre-C/D packetization amendment.
- [S200-only plan](factorial/ablation/run_plan.json) and [configuration](factorial/ablation/config.json) separately freeze the two additional calls and exact source-packet preservation/remapping. Original A–D outputs are unchanged.

Inference used `stt-rt-v5`, `wss://stt-rt.soniox.com/transcribe-websocket`, English hints, diarization enabled, endpoint detection disabled, PCM16/16 kHz/mono, one connection per case/arm. The isolated endpoint choice does not restore historical settings in production. Current integrated segmentation was rechecked: persisted peer threshold 0.4, real Silero/ListenDelivery/SmartTurn components, current selective S200 and float-to-PCM encoding. Archived segmentation used threshold 0.3 and different language metadata/encoding. D is not archived or production-adapter parity.

Projection remains `@soniox/node@2.3.0` `segmentTranscript(tokens, {group_by: ['speaker']})` in provider time **before** source mapping. Scoring remains `pyannote.metrics==4.0.0`, zero collar, overlap included, full clip UEM, one optimal whole-case speaker-label mapping. SDK silence bridges and their false alarms remain. No pulse-score revival, reference-informed VAD clipping, guessed clock shift or per-turn relabeling.

Runtime replacement is explicit: Windows Python 3.12.10 for current-component trace generation/provider execution; WSL Python 3.12.3 for pinned scoring/lexical diagnosis; the model-free consumer probe used Windows Python 3.14.7 and NumPy 2.3.5, not the application NumPy pin. Exact versions and source hashes are in each evidence artifact. Host UTC receipts remain literal; the initial configuration date has a documented session-date erratum in the factorial report, not rewritten authorization history.

## Results: distinct measurements

### Corrected retained baseline

The exact retained `compare_providers.py` command was run in a temporary benchmark copy. Generated comparison JSON matched the retained corrected result semantically; original comparison JSON/Markdown bytes were unchanged. All 28 Soniox native receipts and official SDK/source projections were reproduced.

| Retained arm | Six-primary DER % | Separate dependent far-field DER % |
| --- | ---: | ---: |
| Continuous | 48.27 | 61.44 |
| Fixed-six-second finalize | 70.28 | 80.88 |
| Archived app-segmented | 68.15 | 80.17 |
| Whole-file async | 48.27 | 69.49 |

Full miss/false-alarm/confusion, counts, per-case outcomes and traceable examples: [offline report](offline_report.md). The six primary windows are exposed annotation-selected diagnostics, not an untouched holdout or independent participant sample; the far-field microphone pair is dependent and not pooled with them. Archived `forced` is not an app-cut arm.

### Four-case controlled comparison

A/B transmit identical complete original PCM; C/D transmit identical frozen current-app PCM including S200/replay. A/C finalize once at genuine end; B/D at frozen app cuts plus terminal completion. Matched pairs have identical observed packet/sample/source-availability signatures. All arms use a concurrent receiver and do not wait for intermediate `<fin>` before successor PCM.

| Case | A DER % | B DER % | C DER % | D DER % | B−A pp | D−C pp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| long_context | 42.74 | 73.67 | 43.13 | 71.80 | +30.93 | +28.67 |
| brief_interjections | 62.60 | 69.84 | 58.45 | 83.26 | +7.24 | +24.82 |
| low_overlap | 15.39 | 18.04 | 17.83 | 20.44 | +2.65 | +2.61 |
| long_return_gap | 44.62 | 47.68 | 33.65 | 43.66 | +3.06 | +10.01 |

Four-case pooled DER A/B/C/D: 42.46/63.14/40.91/63.47%. This is not the retained six-primary aggregate. [Scores](factorial/scores.json) retain every component/count; [factorial report](factorial_report.md) and [live identity report](live_identity_report.md) separate causes and limits.

**Native identity is not DER:** on long_context, A→B conditional known-anchored-word error is 1.82%→1.12% despite DER rising 30.93 points. Correct reference-word coverage is 69.28%→68.13%. In contrast, brief C→D native error is 20.79%→36.23% and correct coverage 48.57%→38.68%. Returning-speaker C correctly attributes 38 anchored initial/return words; D misattributes all 38 under its one whole-case mapping, providing a native failure witness independent of token widths. No independent per-turn remapping hides it.

Lexical diagnostics use separately published reference words/turns, conservative timing-independent exact anchors, one whole-case mapping, and explicit unmatched/ambiguous/missing/mixed/overlap/interjection denominators. Conditional error and correct-reference coverage are neither full ASR accuracy nor caption-color accuracy. Unequal matched coverage limits causal interpretation.

**Pacing limits:** A/C and B/D comparisons combine waveform selection, replay, padding, current PCM quantization, availability and necessary packet boundaries. They do not isolate each PCM factor. C/D initially inherited avoidable 512-sample writes; before their first connection, a versioned amendment coalesced byte-identical PCM to nominal 800-sample packets, preserving cuts/maps and maximum included source availability. Original A/B writes/pacing remained unchanged. Actual paired wall times still differ: long-context CD write-time differences reach 6.234 s. Host pacing creates large measured source backlogs; these are disclosed, not called provider processing latency. Archived segmented wall time includes offline trace generation and cannot be compared to new WebSocket duration as a cost/latency improvement.

### Smallest PCM ablation tested: S200 only

The material PCM result justified a separately user-approved two-call test. Remove only S200 packets; keep exact source bytes, replay, packet boundaries, cut/control availability, configuration and completion policy, remapping provider positions by deleted sample counts. Removing paced padding also changes possible control/input timing; it is not a wall-clock-matched pure acoustic estimate.

| Case | D→S200-off DER % | Conditional native error % | Correct reference coverage % | Recognized words |
| --- | --- | --- | --- | --- |
| brief_interjections | 83.26→69.81 | 36.23→22.96 | 38.68→45.71 | 369→353 |
| long_return_gap | 43.66→44.03 | 9.45→9.50 | 80.71→80.27 | 433→432 |

Brief correctly attributed short-interjection words fall **2→1 of 17**; unmatched reference words increase 139→145. Return retains the same 38 wrong initial/return words and two predicted IDs. Post-input final-word medians fall modestly, but correct final-word exposure at actual finalize writes falls 4.84%→3.30% and 5.99%→4.21%, respectively. These deadlines are not first-readable application publication and use retrospective diagnostic identity mapping.

Decision: **do not adopt blanket S200-off** from these observations. It is a case-specific attribution improvement with text/stratum/readiness costs, not a universal gain. See [ablation execution](factorial/ablation/report.md), [identity/text/readiness report](ablation_identity_report.md), and [machine-readable evidence](ablation_identity_evidence.json). One run per arm is not a significance claim.

## Application feasibility, timing and resources

[Consumer probe](consumption_report.md): 16 assertion-backed model-free checks execute actual adapter writer/receiver, scoped engine/normalizer, peer callback/terminal admission, translation owners, output runtime and presenter. Native responses and translation are controlled doubles, not live recognition. A→B→A, A→missing→A, repeated text/context, overlap, zero-duration tokens, padding, source discontinuities, session replacement, cancellation and first-readable immutability are covered.

`FinalSpeakerRun.source_start_ms/source_end_ms` remain literal native stream times through consumers; their names do not provide canonical source coordinates. Text correspondence uses source character ranges and ownership. Peer callbacks discard stable updates, accepted pending `<fin>` clears turn-local state and permits reuse, and successor input waits for completion. Stable-update overlap metadata can differ from terminal aggregation. Unkeyed old-looking text injected after successor admission demonstrates a correlation limitation, not an observed Soniox barrier violation. No safe continuous consumer follows from merely removing finalize or enabling overlap.

Single-occurrence provider-word receipts arrive roughly 7 s after actual source-word-end writes for terminal-only arms versus 3–4 s for cut-finalized arms. This includes provider/wire/control behavior and published-reference timing uncertainty. Sender backlog is separately substantial. No live application publication, correctly colored caption coverage, sustained caption backlog or full candidate memory/latency frontier was measured. No added fixed color wait or recoloring is proposed.

R4 remains **blocked/unresolved**: long-context processed-counter differences B−A +7,080 ms and D−C +8,400 ms occur with the same 121 additional finalize requests. Matched-word displacement supports the observation but does not identify clock cause. Published AMI timings are forced-aligned; independent acoustic anchors with quantified uncertainty and a separately frozen validation example were not obtained. No calibration calls or timestamp correction were performed.

Actual cumulative paid use: **18 requests, 4,347.928 transmitted audio seconds, 5,586.032 conservative WebSocket-envelope seconds, 487/487 finalize receipts, zero retries**. Original approval was 24/6,000/7,200; the additional two-call approval narrowed the execution boundary to 18 requests, 4,347.928 audio seconds and at most 5,745.829 envelope seconds. Both bounds were respected. [Original quota ledger](factorial/quota_ledger.json) and [ablation ledger](factorial/ablation/quota_ledger.json) are separate and preserve reservations/history. No further calls are authorized by unused quota alone.

Declared harness bounds include max 600 source seconds/19.2 MB PCM, nominal 1,600-byte packets, ≤50 ms coalescing availability window (observed 32 ms), 64 MiB received-data cap per call, WebSocket max-size 4 MiB/max-queue 16, and ≤256 pending controls (observed two). Resident-memory high water was not instrumented. Model-free consumer memory figures apply only to its finite fixtures. Full-stream billing assumptions, transmitted PCM, socket duration and replay wall time are distinct; no invoice or monetary saving is claimed.

## Reproduce without new provider calls

From repository root, after obtaining public audio and the existing benchmark dependencies:

```text
# WSL/Linux benchmark venv, pinned pyannote.metrics 4.0.0 and Node SDK 2.3.0
experiments/ami_benchmark/.venv/bin/python experiments/ami_benchmark/providers/soniox/gap/diagnose.py
experiments/ami_benchmark/.venv/bin/python experiments/ami_benchmark/providers/soniox/gap/live_diagnose.py
experiments/ami_benchmark/.venv/bin/python experiments/ami_benchmark/providers/soniox/gap/ablation_identity.py

# Windows application-capable Python; default factorial operation is offline
python experiments/ami_benchmark/providers/soniox/gap/consumption_probe.py
python experiments/ami_benchmark/providers/soniox/gap/factorial.py
python -B experiments/ami_benchmark/providers/soniox/gap/factorial/ablation/run.py
```

Diagnostic commands regenerate their task-local derived reports/evidence; use a disposable checkout/copy to keep this recorded execution unchanged. Native responses, plans, byte identities and execution receipts are not replaceable by rerunning a report. `--live` is not part of offline reproduction, refuses changed frozen sources/existing results, and does not grant new authorization. Detailed commands, dependency versions and expected-failure checks are in the linked reports.

## Completion and remaining decisions

| Requirement / check | Status | Evidence / exact limit |
| --- | --- | --- |
| R1 frozen provenance, obtainable audio, corrected baseline | Passed, historical-version gaps disclosed | Offline report/evidence; original baseline untouched; current app configuration rechecked separately |
| R2 native/timing/projection/text/consumer separation | Passed within declared diagnostic scope | Independent lexical witnesses plus real-owner model-free probe; not live color accuracy |
| R3 matched finalization/PCM matrix | Passed on predeclared four-case subset | 16 completed calls, eight matched PCM signatures, actual pacing/ACK differences logged |
| R3 necessary single-factor PCM ablation | Passed, mixed result | Two approved S200-only calls; exact retained source packets/controls verified |
| Full seven-case new matrix and repeats | Not run | No validated shortlist to expand; another 12 calls would exceed original 24-call cap after the 18 performed. Not marked empirically unnecessary or passed |
| R4 calibrated clock cause/correction | Blocked/unresolved | Independent acoustic verification and uncertainty absent; no guessed correction, calibration or corrected score |
| R5 bounded quality/text/provider-arrival/resource comparison | Passed as an inconclusive evaluation | S200-off tradeoff measured; fewer-finalize arrival frontier recorded |
| R5 live production operating point / first-readable candidate | Not run / unmet | Raw sender has no intermediate ACK gate; current consumer cannot accept continuous output as a toggle; no matched live publication proof |
| Current source/cancellation/ownership/#190 behavior preserved | Passed | No production edit; actual consumer fixtures exercise existing contracts |
| Physical microphone/GUI/VRChat/HMD, Korean/Japanese | Not run | Recorded public English AMI and headless owner path only |
| Billing invoice / monetary savings / sustained RSS | Not run | Audio/connection/quota receipts are not invoices or sustained-resource proof |
| R6 explicit decision and durable artifacts | Defer, with gaps retained | This report and linked raw/derived receipts; no issue-closure or production-enablement claim |

**Failures:** no outstanding provider failure, missing completed score, smoke assertion or quota overrun. Expected missing-`<fin>` and quota-rejection probes passed by rejecting the invalid operation. Development-only logger/rendering/projection-tolerance/summary-association errors were repaired and disclosed in the component reports; original live receipts and baseline scores were not rewritten.

**Integration checks:** read-only completion execution verified all 90 native-response/write-log/SDK/RTTM digests across 18 receipts, current sender/report pins, 4,347.928 audio seconds, 5,586.032 connection-envelope seconds, 487 barriers and zero retries. All 34 local documentation links in the benchmark README and this report resolved. These checks bind the integrated evidence; they do not add provider accuracy or application-readiness coverage.

**Review boundary:** independent whole-checkpoint review of `4dc46fda..e8739a26` found no material findings and checked retained digest/lexical accounting plus actual consumer ownership evidence. Live-code and terminal completion reviews apply to their pinned integrated candidates in the delivery record; that offline review alone does not certify the later changes.

Remaining ownership:

1. An experiment executor with independently verified public-audio acoustic anchors must freeze calibration/validation examples before any clock correction claim. The maintainer authorizes any new finite calls; current unused quota is not an open-ended sweep.
2. The maintainer decides whether added waiting/context/background transmission or ownership/publication amendments are acceptable. No permissible latency regression or color deadline is invented here.
3. A production follow-up owner must exercise the real ACK-gated adapter and first-readable publication with matched text, coverage, startup/returning-speaker, backlog and resource evidence before adopting an operating point. Current padding/finalization contracts remain unchanged.
4. Additional seven-case/generalization runs need a frozen useful shortlist and fresh authorization. Missing indispensable evidence remains unmet, not an accepted narrowed substitute.
