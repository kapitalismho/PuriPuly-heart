# SmartTurn CPU execution (issue #179): ADOPT-F12-2026-09-23

Production after this change is P12/F12: `SmartTurnOnnxInference` with ORT
1/2 (ORT_SEQUENTIAL, ORT_ENABLE_ALL, CPUExecutionProvider, inter-op 1,
intra-op 2) performing one default-executor `asyncio.to_thread` blocking
operation (`prepare_smart_turn_audio` -> `compute_whisper_log_mel_features` ->
`session.run` -> scalar extraction). The completion callback and controller
stay on the original event loop. This adoption is a structural simplification,
not a measured speedup or CPU-gain claim.

Historical #179 measurements and the earlier retain-S12 disposition below are
retained unchanged for audit and are not rewritten as production acceptance.
The tracked sanitized packet is `docs/smartturn-cpu-179-evidence.json`.
Raw per-call records are under ignored `.data/smartturn-179/`.

## Authority, setup, and superseded evidence

- Baseline: `13274569769d3c1ec7a896a2d15b919b76136a6e`.
- Maintainer approval of the frozen comparator is published as
  [CURRENT-LISTEN-SMARTTURN-2026-09-23](https://github.com/kapitalismho/PuriPuly-heart/issues/134#issuecomment-5784588222)
  (updated 2026-09-22T21:37:31Z), with the
  [#179 confirmation](https://github.com/kapitalismho/PuriPuly-heart/issues/179#issuecomment-5784589142).
  It matches the conversation's `POLICY-179-CURRENT`: timely-incomplete-only
  800 ms extension, current automatic/manual language eligibility, and existing
  192/128 ms age steps. Missing or invalid evidence seals at 512 ms.
  Historical receipts remain historical; no endpoint policy is changed here.
- The lifecycle owner reported 91 focused tests passing before the corrected
  measurements. The post-correction controller, CPU, finalist recheck, and
  paced runs started afterward; no correctness run overlapped them.
- Superseded evidence is retained only for audit. The old controller used a
  repeated 32 ms snippet, delivered spans with future end times, measured a
  total-segment boundary rather than the source pause boundary, and filled the
  decision from a later seal. The old co-load ran VAD after SmartTurn and
  sampled RSS before the request. The old finalist recheck predated the
  instrumentation corrections. None of those results supports a disposition.

Runtime was Python 3.14.7, NumPy 2.5.1, ONNX Runtime 1.28.0 on an AMD Ryzen 7
9800X3D (8C/16T), with the Windows high-performance scheme
`8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c` recorded before setup and unchanged
(`power_scheme_drift: false`). OMP/OpenBLAS/MKL thread environment variables
were unset. `threadpoolctl` measured OpenBLAS 0.3.33.112.0, pthreads layer,
effective 16 threads. All arms used `ORT_SEQUENTIAL`, `ORT_ENABLE_ALL`,
`CPUExecutionProvider`, inter-op 1, and the stated intra-op setting.

The Director-selected audio source is LibriSpeech ASR `test.clean`, CC-BY-4.0,
from pinned `openslr/librispeech_asr` revision
`71cacbfb7e2354c4226d01e70d77d5fca3d04ba1`, parquet
`all/test.clean/0000.parquet`, SHA-256
`7113aa4c3cf963fb54697145719a7725f984c8836d1c494a554cbb9f1a017df0`. Eight
distinct-speaker windows are hashed in the audio manifest and tracked evidence;
the five synthetic boundary guards are hashed in the tracked evidence and raw
matrix fixture sequence (not the speech-only audio manifest). This report
makes no speech-quality claim about observed model scores.

Probe-only setup and fetch commands:

```text
uv venv .venv --python 3.14
uv pip install --python .venv/Scripts/python.exe numpy==2.5.1 onnxruntime==1.28.0 psutil==7.2.2 threadpoolctl==3.7.0 pyarrow==25.0.1 soundfile pytest pytest-asyncio
PYTHONPATH=src .venv/Scripts/python.exe scripts/bench_smart_turn_179.py fetch
```

## Measurement contract (historical probe)

The historical probe wrapped the exact production numerical functions
`prepare_smart_turn_audio`, `compute_whisper_log_mel_features`, and
`session.run`, recording worker boundaries, queue waits, callback entry, and
input hashes without adding production logging.

Historical arms (experimental comparators, not production after ADOPT-F12):
S12 was the pre-adoption split path with ORT 1/2; S11 changed only intra-op
to 1; F12 fused the same numerical operations into one owned offload with ORT
1/2; F11 was F12 with intra-op 1.
- Matrix timing is retained from the corrected post-lifecycle run because its
  source path is independent of the controller source-time and active-RSS
  fixes. Each round uses the same fixture for all arms and rotates execution
  order; receipt is measured at callback entry.
- Controller frames are delivered at
  `origin + (frame_index + 1) * 32 ms`, exactly at source end. Each fixture
  contributes its real contiguous first 16,384 samples (1,024 ms), hashed and
  recorded, followed by silence. The run captures the decision at source pause
  512 ms, not total segment age 512 ms, and records the actual seal pause
  length. The full source timeline stays below four seconds.
- The co-load replay delivers each 512-sample/32 ms VAD frame at source end and
  runs bundled Silero concurrently with SmartTurn. Active RSS and thread
  samples come from a 10 ms watcher only while the request/VAD window is live;
  no pre-request sample is labeled active.
- Isolated CPU accounting uses five warmups and 32 measured SmartTurn requests
  per arm, with no VAD co-load. It aggregates process CPU over the bounded
  measured batch before dividing by request count because Windows `psutil`
  process-time deltas are coarse. The observed nonzero per-call quanta were
  15.625–46.875 ms, depending on arm/run.

The tracked packet also preserves per-arm scheduling summaries: submission to
predict entry (the historical `task_start_ms` label includes nested admission),
each worker completion to loop return, predict return to callback, and
completion to controller receipt. These are separate from worker compute time.

Historical runs did not emit a probe-script hash. Their binding to committed
probe `feda47ceefe8ff452919fb416179c7f79cadafef` was independently checked by
inspection, including the retained matrix's unchanged measurement path; an
exact historical script hash cannot be reconstructed and is not backfilled.
New probe outputs record the actual script SHA-256 and checkout HEAD.

## Corrected matrix (retained historical)

Artifact: `.data/smartturn-179/matrix_post_lifecycle.json`.
Command:

```text
PYTHONPATH=src .venv/Scripts/python.exe scripts/bench_smart_turn_179.py --out .data/smartturn-179/matrix_post_lifecycle.json matrix --calls 40 --warmup 5
```

All 40 admissions started for every arm (`busy=0`, `unavailable=0`). Times
below are median / p95 / worst milliseconds:

| Arm | ORT | Dispatch | Submit-to-receipt | Feature queue / worker | Model queue / worker |
| --- | --- | --- | ---: | ---: | ---: |
| S12 | 1/2 | split | 43.327 / 43.958 / 44.883 | 0.061 / 2.608 | 0.035 / 39.454 |
| S11 | 1/1 | split | 80.107 / 81.029 / 81.567 | 0.057 / 2.664 | 0.035 / 76.279 |
| F12 | 1/2 | fused | 43.331 / 44.897 / 50.681 | — | 0.063 / 42.875* |
| F11 | 1/1 | fused | 79.869 / 80.794 / 81.886 | — | 0.063 / 79.479* |

`*` F12/F11 model columns are fused inference queue / worker timings. All
fixture-keyed comparisons have score maximum absolute difference 0.0 and exact
prepared-audio, feature, and model-input hashes. The 1/1 change adds about
37 ms/request without a queue or admission benefit in this workload.

## Corrected LISTEN controller (retained historical)

Artifact: `.data/smartturn-179/controller_post_contract.json`.
Command:

```text
PYTHONPATH=src .venv/Scripts/python.exe scripts/bench_smart_turn_179.py --out .data/smartturn-179/controller_post_contract.json controller --arms S12,S11,F12,F11
```

Each arm ran eight real-clock probes. The contiguous speech prefix is 1,024 ms;
the source time at pause boundary is 1,536 ms (1,024 ms speech plus a 512 ms
pause). Decisions are captured at that pause frontier before any later fallback
seal. The actual seal pause is 512 ms for the one early fixture and 800 ms for
the seven incomplete fixtures on every arm.

| Arm | Receipt median / p95 / worst | Minimum deadline slack | Decision at pause 512 | Seal pause | Late |
| --- | ---: | ---: | --- | --- | ---: |
| S12 | 44.219 / 48.319 / 48.319 ms | 234.770 ms | early 1, incomplete 7 | 512 / 800 ms | 0 |
| S11 | 81.362 / 83.898 / 83.898 ms | 197.729 ms | early 1, incomplete 7 | 512 / 800 ms | 0 |
| F12 | 45.225 / 46.207 / 46.207 ms | 232.552 ms | early 1, incomplete 7 | 512 / 800 ms | 0 |
| F11 | 81.499 / 84.803 / 84.803 ms | 192.104 ms | early 1, incomplete 7 | 512 / 800 ms | 0 |

The score range is 0.0102628–0.9829914 as observed numerical output, not a
quality judgment. Probe status was `started` for all requests and decision
parity remained true. The measured `_request_probe` start-to-return call was
recorded separately (ranges 0.0532–0.1090 ms across arms); the old
scenario-start-to-submit field was removed. Seal records include
`pause_ms_at_seal`, and no post-hoc decision fill is used.

## Isolated process CPU and finalist recheck (retained historical)

Four-arm artifact: `.data/smartturn-179/cpu_post_contract.json`.
Command:

```text
PYTHONPATH=src .venv/Scripts/python.exe scripts/bench_smart_turn_179.py --out .data/smartturn-179/cpu_post_contract.json cpu --arms S12,S11,F12,F11 --calls 32 --warmup 5
```

The CPU values are isolated SmartTurn process CPU (no VAD), aggregated over 32
warmed requests. They are not inference-only thread CPU and are not mixed with
idle gaps:

| Arm | Process CPU total | Process CPU/request | Receipt median / p95 / worst |
| --- | ---: | ---: | ---: |
| S12 | 2.765625 s | 0.086426 s | 43.946 / 44.956 / 45.336 ms |
| S11 | 1.859375 s | 0.058105 s | 80.613 / 82.016 / 84.011 ms |
| F12 | 2.234375 s | 0.069824 s | 43.606 / 44.516 / 44.724 ms |
| F11 | 2.093750 s | 0.065430 s | 80.283 / 81.478 / 81.526 ms |

Corrected selected-vs-baseline recheck artifact:
`.data/smartturn-179/recheck_post_contract.json`.
Command:

```text
PYTHONPATH=src .venv/Scripts/python.exe scripts/bench_smart_turn_179.py --out .data/smartturn-179/recheck_post_contract.json recheck --arms S12,F12 --calls 32 --warmup 5
```

Both arms started all 32 requests. S12 measured 1.703125 s total / 0.053223
s/request and 43.210 / 44.627 / 51.568 ms receipt median/p95/worst. F12
measured 2.312500 s total / 0.072266 s/request and 43.824 / 46.101 / 49.230
ms. This recheck does not establish a F12 benefit. S12 process CPU varied from
0.086426 to 0.053223 s/request between the two batches, so these small samples
do not establish a stable CPU reduction percentage or a universal winner.

## Corrected concurrent Silero co-load (retained historical)

Artifact: `.data/smartturn-179/paced_post_contract.json`.
Command:

```text
PYTHONPATH=src .venv/Scripts/python.exe scripts/bench_smart_turn_179.py --out .data/smartturn-179/paced_post_contract.json paced --arms S12,S11,F12,F11 --rounds 1 --gap 1.0 --coload silero
```

Each arm processed the same eight fixtures and 1,522 VAD frames, with zero
missed 32 ms slots. Endpoint counts were eight `SpeechStart` and zero
`SpeechEnd` per arm because the external replay boundary ends each clip while
the gate remains open. `coload_process_cpu_per_request` includes the complete
concurrent SmartTurn plus VAD request window; it is not inference-only CPU.

| Arm | Receipt median / p95 / worst | Co-load process CPU/request | Active RSS median / p95 | Active threads | Loop lag p95 / worst |
| --- | ---: | ---: | ---: | ---: | ---: |
| S12 | 52.682 / 58.908 / 58.908 ms | 0.257812 s | 131.24 / 131.46 MB | 25 | 10.613 / 19.589 ms |
| S11 | 90.185 / 91.006 / 91.006 ms | 0.167969 s | 133.01 / 133.30 MB | 21 (p95 24) | 10.723 / 17.438 ms |
| F12 | 50.601 / 55.294 / 55.294 ms | 0.214844 s | 133.67 / 134.40 MB | 19 (p95 22) | 10.621 / 15.602 ms |
| F11 | 89.270 / 90.895 / 90.895 ms | 0.191406 s | 135.02 / 135.94 MB | 18 | 10.640 / 15.819 ms |

Capture lag was 5.58–5.78 ms median and 12.05–12.13 ms p95, with no missed
slots. Idle-gap CPU was 0.0 s for every arm. Active RSS values are watcher
samples taken while compute was live, not the earlier pre-request samples.

## Bounded Hann allocation probe (retained historical)

Artifact: `.data/smartturn-179/hann_post_contract.json`.
Command:

```text
PYTHONPATH=src .venv/Scripts/python.exe scripts/bench_smart_turn_179.py --out .data/smartturn-179/hann_post_contract.json hann --repeats 30
```

The probe compares the current `_HANN_WINDOW.astype(np.float64)` path with a
probe-local cached-window path over the same prepared fixture. Outputs are
bitwise equal (`max_abs_output_diff=0.0`). Current versus cached full-feature
cost was 3.103 ms versus 3.063 ms median; the delta was -0.040 ms median and
+0.218 ms at p95, while the cast operation alone was 0.0003 ms median. The
bounded, noisy end-to-end result is negligible relative to SmartTurn inference;
disposition: explicitly reject this allocation change, with no production edit.

## ADOPT-F12-2026-09-23 production evidence

Production `SmartTurnOnnxInference` now performs one default-executor blocking
operation per request. ORT SEQUENTIAL, ENABLE_ALL, CPU, inter 1/intra 2 and the
numerical functions are unchanged. Adoption is structural simplification only;
no speedup or CPU-gain claim is made.

Authority: the maintainer explicitly selected `1/2 + single offload` in the
conversation after reviewing the tradeoff. This supersedes the historical
retain-S12 disposition, not its measured results or limitations. Adoption
baseline: `c33692591a76cb6790f4cd2abad2c2a0501ac63d`.

Current #177 handoff: ORT 1/2, single default-executor offload, unchanged input
snapshots, preparation, feature precision, spinning and numerical-library
settings. This remains independent of embedded-package acceptance.
Rollback: restore the split `SmartTurnOnnxInference.predict` implementation
from the adoption baseline; do not change controller policy or preprocessing.

```text
PYTHONPATH=src .venv/Scripts/python.exe scripts/bench_smart_turn_179.py --audio-dir .data/smartturn-179/audio --out .data/smartturn-179/adopt_f12_parity.json matrix --calls 13 --warmup 2
PYTHONPATH=src .venv/Scripts/python.exe scripts/bench_smart_turn_179.py --audio-dir .data/smartturn-179/audio --out .data/smartturn-179/adopt_f12_controller.json controller --arms P12,S12,F12
```

Matrix (13 measured calls/arm, 2 warmup, rotated order, Python 3.14.7 /
NumPy 2.5.1 / ORT 1.28.0, model SHA
`2bb026316b14a660486a75b1733cd3fbab8c2fd0314dc9af7be49f8cca967e4f`):
all arms started 13/13 with outcome complete. Score max abs diff 0.0 and exact
prepared/feature/model-input hashes for P12 vs S12/S11/F12/F11.
Submit-to-receipt median/p95/worst ms: P12 44.545/52.748/52.748,
S12 44.011/47.713/47.713, F12 43.879/48.227/48.227. Descriptive only.

Controller (8 real-clock probes/arm, P12 actual production owner):
decision parity true; early 1, incomplete 7, seal delivery_pause 8, late 0 on
every arm. Receipt median/p95/worst ms: P12 45.122/49.317/49.317
(slack 223.236 ms), S12 45.184/52.316/52.316 (slack 227.907 ms),
F12 44.961/47.401/47.401 (slack 225.983 ms). Scores identical per fixture
(range 0.0102628-0.9829914, observed output only).
All 13 guards+speech production owner scores complete; direct predict checks
confirm sample-rate/shape validation, no-outputs and closed-session errors.

## Selected disposition and migration handoff (historical retain-S12)

- **R6 disposition:** supported tradeoff/inconclusive investigation; retain
  S12 (ORT 1/2, split offloads). No production changes need rollback.
- **1/1:** lower CPU was observed in the four-arm and co-load samples, at
  roughly 37 ms more request latency. All observed controller deadlines were
  met, but this does not establish headroom under every application workload.
  The investigation does not adopt or reject that product tradeoff.
- **Fused dispatch:** no consistent benefit in the corrected recheck. Keep
  experimental fused paths in the probe only.
- **Dedicated executor:** no-change. Matrix queue waits were below 0.138 ms;
  concurrent co-load maxima were 4.265/3.063/3.190/3.883 ms for S12/S11/F12/F11.
  This bounded co-load does not justify another resource lifecycle; it does
  not establish absence of contention under every application workload.
- **ORT spinning:** support was checked, but no profile sweep was performed.
  Zero observed idle-gap CPU is subject to Windows counter granularity, not
  proof of zero CPU use. Retain the existing configuration.
- **Hann allocation:** reject as negligible/noisy; no production change.
- **#177 handoff:** retain ORT_SEQUENTIAL, ORT_ENABLE_ALL, CPUExecutionProvider,
  inter-op 1/intra-op 2 and the current split dispatch, unchanged numerical
  functions and pinned preprocessing fixture. No BLAS/environment, model,
  precision, endpoint, setting, or production dependency change. This optional
  investigation does not block the migration, already landed through #181.
  Source-checkout CPython evidence does not claim embedded/native-package
  performance; that composition's acceptance remains separate.

## Verification and limits

After review repairs and shared formatting, the integration check passed:
**144 tests** (90 SmartTurn runtime/controller and 54 existing peer-capture tests).
The redundant twin-harness equality test was removed; the existing explicit
threshold-equality and receipt-deadline tests remain. Controller snapshot
isolation now observes the submitted reference, not a stub-created copy.

```text
.venv/Scripts/python.exe -m pytest tests/core/test_smart_turn_runtime.py tests/core/test_smart_turn_delivery.py tests/core/runtime/test_peer_capture_session.py -o addopts= -q
.venv/Scripts/python.exe -m ruff check src/puripuly_heart/core/audio/smart_turn.py scripts/bench_smart_turn_179.py tests/core/test_smart_turn_runtime.py tests/core/test_smart_turn_delivery.py
```

The unchanged golden feature hash passed on NumPy 2.5.1. Regression coverage
includes snapshot isolation, natural reset and synthetic-rollover continuity,
stale/discontinuous context, injected threshold/receipt behavior, preparation
timeout and late reclamation, and feature/ONNX lifetime barriers.

One Windows CPU and small samples only; p95 is descriptive. The co-load is
local Silero frame replay, not live microphone/OS-device capture, full desktop
application load, cloud ASR, or an HMD campaign. No hardware-drop guarantee,
second-machine result, stable CPU percentage, packaged embedded-Python
performance, or speech-quality claim follows from these measurements.
Background system load was not sampled per batch. Coarse CPU counters and
unrecorded background variation limit attribution of the cross-batch CPU
differences; neither the historical disposition nor this adoption establishes
a stable CPU or latency gain.
No architecture ownership or dependency boundary changed.

## Adopted production path: current co-load validation

These operational checks use actual production P12 at
`bda5b565361125ef9e8ee9556cba2cbfe8e96de7`, not the probe-local F12 prototype.
Each arm processes eight requests and 1,522 concurrently paced Silero frames.
The machine's background load was not sampled; power-scheme drift was false.

```text
.venv/Scripts/python.exe scripts/bench_smart_turn_179.py --out .data/smartturn-179/adopt_f12_coload.json paced --arms P12 --rounds 1 --gap 1.0 --coload silero
.venv/Scripts/python.exe scripts/bench_smart_turn_179.py --out .data/smartturn-179/adopt_f12_coload_paired.json paced --arms S12,P12 --rounds 1 --gap 1.0 --coload silero
```

| Run | Arm | Receipt median / p95 / worst ms | Frames processed at least 32 ms late |
| --- | --- | ---: | ---: |
| Initial adopted-path check | P12 | 63.480 / 129.956 / 129.956 | 3 |
| Contemporaneous comparison | S12 | 71.190 / 89.642 / 89.642 | 8 |
| Contemporaneous comparison | P12 | 70.736 / 99.097 / 99.097 | 0 |

Every request completed; all frames were processed. All three runs recorded
eight speech starts with clips ending at external replay boundaries, not
natural endpoints. The late-frame counter is replay scheduling lateness,
not evidence of lost audio or a hardware-drop measurement. The paired run
uses sequential arms in one process, not randomized repeated batches.

The initial P12 lateness is retained, not discarded. This check did not
reproduce a candidate-specific scheduling regression: lateness also appeared
on S12 and was absent on the paired P12 run. It does not prove that fusion
improves scheduling or that application-wide interference is absent. The
earlier zero-late historical runs do not characterize current machine load.
Controller decision/receipt parity is covered separately above.
