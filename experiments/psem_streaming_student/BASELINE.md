# #164 Vulkan teacher/receiver baseline

## Status

**Baseline execution complete; R2 complete-path cost is INCOMPLETE.** This is an exposed-DEV engineering probe, not a teacher-quality verdict, production admission measurement, API/display run, or training result.

The maintainer authorized local Vulkan baseline execution and selected “Baseline before training discussion.” One native pass ran for each approved source. No training/backward/update, paid or cloud API, HOLDOUT/EVAL access, production mutation, or #160 implementation occurred.

## Fixed execution

- ES2009d: source zero through 180 s plus a real 16 s context tail; evaluation ends at 180 s.
- EN2009d: source zero through 48.4 s.
- Total approved audio: 244.4 s; combined model wall: 261.526 s, below the 1,800 s cap.
- Backend/device: Vulkan on the local AMD Radeon RX 7900 XTX.
- Profile: chunk 6, left 1, right 7, FIFO 188, speaker cache 188, update 144; 1,280-sample frames, 1,600-sample confirmation, threshold 0.5, 16 kHz.
- Executable SHA-256: `1a1ad34ed2a778ffbfcd0a360443db30591a16587d5a30ac6f271c3062ae7098`.
- Model SHA-256: `62faec7b99ad23e323087597604b50728abe85089b6364970b019a845547bf99`.

Each source used one continuous native process and one reference/producer generation from source zero. Logical parent seals did not reset model/reference state. Source end was the only discontinuity.

## What was actually measured

Native availability, receiver receipt/consumption, confirmation, and experiment admission invocation use the same Windows QueryPerformanceCounter clock. The admission rule was deliberately no-wait: a frozen accepted-text parent became due at its retained source-span end. **That cutoff is an experimental replay choice, not measured ASR-final or production translation-admission timing.** No historical admission was interpolated, but no production ASR, translation API, or display was executed either.

ES emitted 9 native transitions; 3 of 6 annotated non-overlap transitions matched within 250 ms. EN emitted 6; 2 of 3 matched. The approved prefixes also contain annotation overlap witnesses. Retained admitted-parent guards covered ES 9 same-speaker / 19 verified-change / 3 mixed-or-overlap parents and EN 4 / 6 / 4 respectively.

Confirmed events were consumed before the replay cutoff in both event-bearing ES parents and all four event-bearing EN parents. All six still fell back to a whole-parent unit because complete evidence coverage was unavailable at that cutoff; one ES parent also contains an unmapped accepted token. One later event per source remained post-commit and was not applied retrospectively.

The fixed selected predicate was preserved exactly: suppress only a boundary with the same transition/uncertainty key, differing relation, and one UNKNOWN; a merged unit remains UNKNOWN. Direct CURRENT↔OTHER and straddle/transition keys remain boundaries, and whole-parent eligibility/coverage/fallback remains unchanged. Because coverage blocked every applicable parent, the predicate changed 0 parents and removed 0 boundaries. Accepted text/order/source mapping was conserved with 0 failures. The no-active-PSEM arm is identical accepted-text, one-unit replay.

This is a **receiver-scope coverage/admission blocker**, not evidence that the teacher needs fine-tuning and not a universal prerequisite prohibiting R4 training. After maintainer discussion, matched GT/KD training may proceed under its own R4 authority while the downstream receiver gap is retained, separately repaired, or scoped out. This report does not authorize changing the fixed policy or receiver contract.

## Cost and limitations

| Source | paced path wall | native CPU | native peak working set | attributable dedicated GPU peak |
|---|---:|---:|---:|---:|
| ES2009d | 196.573 s | 1.016 s | 339,570,688 B | 167,706,624 B |
| EN2009d | 48.737 s | 0.125 s | 314,396,672 B | 167,706,624 B |

GPU memory is the Windows per-native-PID `GPU Process Memory` dedicated-usage counter (194 ES samples, 48 EN samples). It excludes shared memory and driver allocations and is not global GPU usage. GPU processing time/compute RTF was not separately observable from paced wall.

The original receiver-wrapper CPU and RSS calls returned raw zero because a Windows pseudo-handle was passed without 64-bit ctypes signatures. Those raw failures remain recorded and are not reinterpreted as zero cost. Consequently, native RSS is only a complete-path lower bound: **receiver cost, complete-path peak RSS, and therefore R2 cost completion remain unavailable.** The helper now uses a real process handle and explicit ctypes signatures for future runs; its current-process smoke check does not revise this run’s measurements.

## Evidence and retrieval

- Summary and decision: `RESULT.json`, `FINDINGS.json`.
- Envelope and identities: `baseline_config.json`, `READINESS.json`.
- Verification: `VERIFICATION.json`.
- Corrected helper smoke proof: `INSTRUMENTATION_SMOKE.json`.
- Per-source timing, transitions, cost, and artifact hashes: `runs/ES2009d/RESULT.json`, `runs/EN2009d/RESULT.json`.
- Required compact raw evidence: `runs/*/native-events.jsonl.gz`, `runs/*/receiver-assignments.jsonl.gz`, and `runs/*/dump/diar.probs.{f32,json}`. Native rows retain source support/consumed frontier, validity, generation, effective profile, native QPC, and receiver receipt QPC; the retained native probability tensor and its metadata preserve the frame-level soft outputs.
- Clock/profile trace and per-process GPU counter: `runs/*/dump/diar.trace.json`, `runs/*/gpu-process-memory.csv`.

The source projections (`runs/*/input.wav`) and large hidden/logit tensors (`runs/*/dump/diar.{hidden,logits}.f32`) remain locally available but are ignored by `.gitignore`; they are not required by verification on a fresh checkout. The required `diar.probs.f32` soft-output tensor is not ignored. Ignored files are derived from the source WAV paths and exact sample bounds in `baseline_config.json`; projection hashes are in each source `RESULT.json`. Retrieve original WAVs from the config paths and the external executable/weights from the pinned paths in the same config. Those external personal-machine paths are an audit/reproduction limitation, and this report does not authorize a model rerun merely to retrieve measured evidence.

Commands used:

```text
python -B experiments/psem_streaming_student/run_baseline.py prepare
python -B experiments/psem_streaming_student/run_baseline.py execute
python -B experiments/psem_streaming_student/analyze_baseline.py
python -B experiments/psem_streaming_student/run_baseline.py verify
```

Architecture boundary: experiment-local scripts plus pinned runtime archive/override only; product source and production behavior did not change.
