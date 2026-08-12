# R8 Streaming Sortformer Feasibility and Compute-Cost Experiment Plan

## 1. Status and Scope

This document defines a bounded, no-training feasibility experiment following the failed R7-B
frozen-ERes and ERes-plus-PCM arms. The owner authorized implementation and execution on
2026-08-12. Product integration, deployment, and publication remain unauthorized.

R8 asks one practical question:

> Can an existing overlap-aware streaming diarizer turn the same mono meeting audio used by R7-B
> into useful `new_speaker_onset` events at the required low-false-event operating points, and can
> the target machine sustain its low-latency compute schedule on CPU or Vulkan?

R8 intentionally does not train ERes, mHuBERT, a new decoder, or any other model. It evaluates one
pretrained Streaming Sortformer port and one quantization as the primary system.

## 2. Why This Experiment Is Next

R5, R6, R7-A, and R7-B tested several decision forms over frozen ERes evidence. All found aggregate
speaker information, but none created a useful extreme-low-false-event operating region. R7-B B0
reached 1.3% Recall@250 at 9.511 false events/hour, and B1 reached 1.4% at 9.934 false events/hour.

The R7-B target panel is dominated by overlap onset and silence-gap change rather than clean direct
speaker replacement. Streaming Sortformer is relevant because it predicts up to four simultaneous
speaker-activity tracks directly from mono audio and carries speaker slots across internal chunks.
It therefore changes the representation and task formulation rather than adding another head to
ERes.

This is a falsification-first experiment. A pretrained direct diarizer should be measured before
authorizing new representation training.

## 3. Claims R8 May and May Not Support

R8 may support:

- internal evidence that low-latency overlap-aware diarization can recover R7-B change events;
- an event-level recall-versus-false-events curve on the exact R7-B source exposure;
- isolated CPU and Vulkan throughput, chunk deadline, memory, and backlog measurements on the
  target machine;
- a decision on whether a later live push-audio integration or domain adaptation study is warranted.

R8 may not support:

- an untouched or confirmatory generalization claim;
- a claim that the exact R7 latency budget is met, because the published low-latency preset has
  approximately 1,040 ms of algorithmic lookahead before measured compute;
- a live-microphone API claim, because the pinned transcribe.cpp entrypoint accepts a whole
  recording even though its compute core processes internal streaming chunks;
- a combined ASR-plus-diarization resource claim;
- product integration, production lifecycle ownership, or deployment readiness;
- performance with more speaker slots than the model's four-output limit.

## 4. Locked Third-Party System

### 4.1 Runtime source

Use `handy-computer/transcribe.cpp` at the validated commit prefix `d42c3bb`. Acquisition must resolve
and record the full immutable commit SHA before building. A different revision requires a reviewed
plan amendment rather than an unreported upgrade.

The source receipt must record:

- repository URL and full commit SHA;
- vendored ggml revision recorded by that checkout;
- compiler, CMake, build type, and enabled backends;
- Vulkan SDK/runtime version, device name, and driver version;
- every local telemetry-only patch and its SHA-256.

### 4.2 Model

Primary model:

```text
handy-computer/diar_streaming_sortformer_4spk-v2.1-gguf
diar_streaming_sortformer_4spk-v2.1-Q8_0.gguf
```

The acquired Hugging Face repository revision resolved to
`7ef0c15dc8f9d717e9d24fac29a6e6551e9c6ddf` and is recorded in the frozen execution
configuration and model receipt.

The Q8_0 model is approximately 139 MB and is the only full-panel quantization authorized. The F16
model may be acquired only for the fixed parity smoke described below. No K-quant, alternate
Sortformer release, or model sweep is included.

The acquisition receipt must record the exact download URL, repository revision when available,
file size, SHA-256, and NVIDIA Open Model License disposition. Model binaries and third-party build
trees remain outside Git under `SRSCD_CACHE_ROOT`.

### 4.3 Operating point

Use only:

```text
TRANSCRIBE_SORTFORMER_PRESET_LOW_LATENCY
```

The published geometry is six 80 ms chunk frames with seven frames of right context, corresponding
to approximately 1,040 ms of algorithmic lookahead. `DEFAULT`, `HIGH_LATENCY`, and
`VERY_HIGH_LATENCY` are excluded from the primary result because they do not answer the low-latency
question.

## 5. Evidence Mode and Dataset

R8 remains in `fast_internal_development_known` mode. It reuses all ten R7-B meetings and no other
corpus:

| Fold | Meetings |
| --- | --- |
| 1 | `alimeeting_R8001_M8004`, `ami_IS1009a` |
| 2 | `alimeeting_R8008_M8013`, `ami_EN2001d` |
| 3 | `alimeeting_R8009_M8019`, `ami_TS3006a` |
| 4 | `ami_ES2003a`, `alimeeting_R8007_M8010` |
| 5 | `ami_TS3009b`, `ami_ES2015d` |

The exposure is 4.731 source hours with 4,619 R7-B `new_speaker_onset` references. R8 must reuse the
R7-B waveform inventory, reference labels, source durations, meeting identifiers, event matcher,
and error-stratum definitions by hash. Audio is 16 kHz mono. No waveform may be regenerated from a
different channel or annotation source.

These meetings are development-known, and exact upstream Sortformer training-session exclusion is
not established. A passing result is therefore an internal product-feasibility signal only. It
must stop with a request to freeze a new natural panel before any confirmatory claim.

## 6. Fixed Smoke Panel

Before inspecting Sortformer predictions, select and freeze twenty 30-second excerpts using only
R7-B annotations:

- four clean or nearest-available direct changes;
- four overlap onsets;
- four silence-gap changes;
- four short backchannel or return regions;
- four same-speaker hard-negative regions.

If fewer than four clean direct changes exist, fill the shortfall with additional overlap and
silence-gap clips and state that limitation. Store clip source meeting, start/end sample, reference
types, waveform hash, and selection reason. Do not select clips by listening to or ranking model
output.

The smoke panel is used for build validation, F16/Q8_0 parity, backend parity, and repeated compute
measurement. It is not the primary accuracy result.

## 7. Probability and Timing Extraction

The existing transcribe.cpp Sortformer validation path can dump the internal `diar.probs` tensor.
R8 requires, for every meeting:

- the raw `T x 4` speaker-activity probabilities at the documented 80 ms frame rate;
- the speaker segments emitted by the unmodified library;
- total inference wall time;
- model load time;
- per-internal-chunk compute start, stop, and cache-compression timing;
- backend and resolved device identity.

If the pinned runtime does not expose per-chunk timing, one telemetry-only patch may be applied to
the external checkout. It may record timestamps and raw probabilities but must not alter tensor
math, cache decisions, post-processing, or scheduling geometry. On the deterministic Sortformer
fixture, patched and unpatched runs must produce byte-identical Q8_0 probabilities and speaker
segments. Failure of that check invalidates all patched results.

The shipped API is whole-recording `transcribe_run`, not live `push_audio`. R8 therefore simulates
the published internal streaming geometry on complete files. It does not claim that events were
emitted incrementally to an application.

## 8. From Speaker Tracks to Change Events

The primary event decoder operates on raw probabilities rather than the library's one fixed
post-processing point.

For one shared activity threshold `theta`:

1. Binarize each of the four speaker tracks independently at every 80 ms frame.
2. Treat an inactive-to-active transition as a speaker onset.
3. Do not count the first speaker onset in a recording when no preceding speaker exists.
4. Count `A -> A+B` when the B track becomes active while A remains active.
5. Count `A -> silence -> B` only when B differs from the most recent active speaker and the silence
   gap is no more than the locked R7-B maximum of 500 ms.
6. Do not count `A -> silence -> A` as a change.
7. Preserve short returns and apply the same 200 ms duplicate suppression used by R7-B.
8. Keep the source event timestamp at the first active 80 ms frame. Report decision availability
   separately as algorithmic lookahead plus measured compute.

All four tracks use the same threshold. Per-meeting, per-speaker, or per-stratum thresholds are
forbidden.

The threshold search uses a deterministic dense pass followed by exact refinement over every unique
probability value inside the interval that brackets each 1/5/10/20 false-events/hour operating
point. This avoids repeating the coarse-threshold error found during R7-A.

The library's documented fixed post-processing output is reported as a contextual row only. It does
not replace the raw-probability curve.

## 9. Accuracy Protocol

The primary accuracy view is the aggregate continuous curve over all ten meetings, matching R7-B's
fast internal evidence mode. Report:

- Recall@100, Recall@250, and Recall@500;
- false-event count and false events per source hour;
- the best Recall@250 at no more than 1, 5, 10, and 20 false events/hour;
- precision and F1 as context only;
- overlap-onset, silence-gap-change, clean-change, and short-return recall;
- maximum single-meeting share of matched true positives;
- per-meeting curves and counts;
- speaker-slot discontinuity examples that create false changes.

Add a threshold-transfer robustness view using the existing five folds. For each held-out pair,
select each threshold on the other eight meetings and apply it unchanged to the held-out pair.
Report realized held-out false events/hour and recall. This transfer view is diagnostic because no
new model is trained, but it exposes meeting-specific calibration failure.

Do not use diarization error rate as the primary metric. DER may be reported only as contextual
evidence because the product target is event onset, not complete who-spoke-when reconstruction.

## 10. Compute-Cost Protocol

### 10.1 Backends

Measure exactly two Q8_0 backends on the same target machine:

- CPU;
- Vulkan.

Execution amendment, 2026-08-12: after both backend smoke runs, the owner explicitly limited the
full-panel replay to CPU and requested that the Vulkan full replay be omitted. Vulkan therefore
retains build, device-resolution, parity, repeated-smoke, memory, RTF, and chunk-timing evidence,
but it has no full-panel compute-gate result. This is reported as `not_run_by_owner_decision`, not
as a Vulkan failure. Accuracy uses CPU Q8_0 probabilities only.

Use one fixed resolved CPU thread count and record it before material timing. Do not tune thread
count, Vulkan device selection, chunk geometry, or graph options after inspecting results. If
Vulkan silently falls back to CPU for any material graph, mark the Vulkan arm invalid rather than
reporting it as accelerated.

F16 is restricted to one CPU parity run on the twenty smoke clips. It is not a third performance
arm.

### 10.2 Warm-up and repetitions

For each backend:

1. Start one fresh process and record model load time and peak memory.
2. Run one unreported warm-up clip.
3. Run the twenty fixed smoke clips three times in the same order.
4. Run each full meeting once, preserving one model instance for the entire meeting.

Do not batch different meetings through one persistent speaker cache. Reset cache state at meeting
boundaries only.

### 10.3 Required measurements

Report for each backend:

- model load milliseconds;
- total wall seconds and real-time factor per clip and meeting;
- audio-seconds processed per wall-second;
- per-chunk compute p50, p95, p99, and maximum;
- cache-compression call count and timing percentiles;
- simulated streaming backlog over each meeting;
- peak process RSS/private bytes;
- mean and peak CPU utilization;
- Vulkan device, driver, peak dedicated GPU memory when observable, and sampled GPU utilization;
- output-probability and event agreement against CPU Q8_0.

The low-latency chunk contributes 480 ms of new audio. Simulated backlog is updated in source order:

```text
backlog[n] = max(0, backlog[n-1] + chunk_compute[n] - 480 ms)
```

Report maximum backlog and the proportion of chunks whose compute time exceeds 480 ms. Aggregate
RTF alone is insufficient because a system may be fast on average while missing individual live
deadlines.

Energy measurement is optional and must not block completion. CPU, memory, and backend identity are
mandatory. Missing Vulkan memory/utilization counters must be reported as unavailable rather than
estimated.

### 10.4 Forecast stop

After the smoke panel, forecast full-panel wall time separately for CPU and Vulkan. Stop one backend
before full-panel execution if its forecast exceeds 24 hours or peak committed memory reaches the
existing 24 GiB experiment ceiling. The other valid backend may continue. A stopped backend retains
its smoke measurements and is classified as compute-infeasible for this experiment.

## 11. Predeclared Gates

### 11.1 Event-accuracy gate

The direct Sortformer event path passes the inherited R7-B usefulness gate only if all conditions
hold:

- aggregate Recall@250 is at least 30% at no more than 10 false events/hour;
- aggregate Recall@250 is at least 50% at no more than 20 false events/hour;
- overlap-onset and silence-gap-change recall are both non-zero;
- no meeting contributes more than half of matched true positives;
- no held-out fold realizes more than twice its transferred 10 or 20 false-events/hour target.

The final condition uses the threshold-transfer view. Failure must be reported even if the aggregate
curve passes.

### 11.2 Compute gate

CPU and Vulkan are judged independently. A backend passes minimum real-time compute feasibility only
if:

- full-panel aggregate RTF is below 1.0;
- per-chunk p99 compute is no more than 480 ms;
- maximum simulated backlog is no more than 480 ms;
- it completes without backend fallback, non-finite probabilities, cache corruption, or memory-cap
  violation.

Also report a preferred deployment-headroom view at RTF at or below 0.5. This is not a hard R8 gate
because concurrent ASR is outside scope, but lack of headroom must be visible in the recommendation.

The approximately 1,040 ms algorithmic lookahead is reported before compute and already exceeds the
old exact 1,000 ms R7 input frontier by approximately 40 ms. Passing the compute gate therefore
means sustainable near-one-second streaming, not compliance with the old exact latency boundary.

## 12. Outcomes

### Outcome A — Accuracy and compute are both feasible

The event-accuracy gate passes and at least one backend passes the compute gate. Conclude that the
product goal is realistic at the measured near-one-second operating point on this internal panel.
Stop and request approval for a live push-audio adapter study plus a newly frozen evaluation panel.

### Outcome B — Accuracy passes, compute fails

Sortformer shows the required event information, but neither backend sustains the schedule. Do not
change the model or latency inside R8. Recommend a separately scoped runtime optimization, alternate
hardware, or smaller-model study.

### Outcome C — Compute passes, accuracy fails

The runtime is affordable but the pretrained model does not solve the product event tail. Do not
integrate it. Use the stratum failures to decide whether domain adaptation or a different
frame-level representation is justified.

### Outcome D — Accuracy and compute both fail

Stop the direct pretrained Sortformer path. Do not proceed to live integration or combine it with
ERes/mHuBERT without a new scientific rationale.

### Outcome E — Invalid or inconclusive

Use only for probability-dump mismatch, backend fallback, corrupted cache continuity, irreconcilable
timestamp mapping, missing source evidence, or an exceeded execution ceiling before one valid full
accuracy run completes.

## 13. Execution Sequence

1. Freeze the R8 configuration and hash the R7-B input inventory and references.
2. Freeze the twenty-clip annotation-selected smoke panel.
3. Acquire and receipt the pinned transcribe.cpp source and Q8_0 model outside Git.
4. Build CPU and Vulkan variants and prove resolved backend identity.
5. Run the deterministic upstream fixture and unmodified output smoke.
6. Add telemetry-only extraction if required and prove byte-identical outputs.
7. Run F16/Q8_0 and CPU/Vulkan parity on the smoke panel.
8. Run the repeated compute smoke and apply the 24-hour forecast stop.
9. Run Q8_0 low-latency inference on the ten meetings for every surviving backend.
10. Decode the raw probability thresholds and score the aggregate and transfer views.
11. Produce the compute, accuracy, stratum, and representative timeline reports.
12. Select exactly one predeclared outcome and stop.

No result authorizes the next outcome's follow-up work automatically.

## 14. Execution Ownership

The coordinator owns repository code, configuration, scoring, and interpretation. Long model
downloads, builds, probability extraction, repeated CPU/Vulkan inference, and full-panel replay are
CPU- or device-time-heavy and must run through an OpenCode worker controlled by the Orca CLI. Worker
jobs must receive exact commands and artifact paths and must not modify product code. Long-running
jobs are supervised with event-driven waits or approximately 15-minute waits rather than rapid
polling.

## 15. Artifacts

Store all material outputs outside the repository under:

```text
%SRSCD_CACHE_ROOT%/results/r8/streaming_sortformer_feasibility_v1/
```

Required artifacts:

```text
config.json
source_receipt.json
model_receipt.json
hardware_receipt.json
input_inventory.json
smoke_panel.json
telemetry_patch.diff
telemetry_validation.json
probabilities/<backend>/<meeting>.npz
speaker_segments/<backend>/<meeting>.json
events/<backend>/<meeting>.json
accuracy_metrics.json
threshold_transfer_metrics.json
compute_metrics.json
backend_parity.json
recall_false_event_curve.png
chunk_compute_backlog.png
representative_timelines/
REPORT.md
```

Every receipt and final metric file records its own SHA-256 or is covered by a hashed inventory.
Aborted or invalid runs are preserved and clearly marked; they are never overwritten as if valid.

## 16. Product Architecture Boundary

R8 changes no production module, so no architecture drift is expected.

If Outcome A is later confirmed and live integration is separately approved, Sortformer would be a
long-lived native process or worker resource. Its owner would need to preserve the speaker cache for
one capture generation, reset it at session replacement, reject late events after retirement, and
keep blocking native inference off the Python application event loop. Because Sortformer emits
speaker activity rather than transcript events, it must not be inserted into the STT provider port
without an explicit diarization/change-event contract.

The current transcribe.cpp whole-recording API is not sufficient for that integration. A push-audio
surface or equivalent persistent streaming adapter is a separate reviewed task.

## 17. Completion and Approval Boundary

R8 planning is complete when this document fixes the source, model, data, event semantics,
thresholding, compute measurements, gates, artifacts, and outcome rules.

Execution was explicitly authorized by the owner on 2026-08-12. Acquisition, inference, and the
telemetry-only external checkout patch are within that execution approval; product integration,
publication, and follow-up experiments remain outside it.
