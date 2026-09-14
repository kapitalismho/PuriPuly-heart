# #164 Vulkan teacher/receiver baseline

## Status

**Cutoff-conditional partial baseline; R2 complete-path cost is incomplete.** This is an exposed-DEV engineering probe, not a teacher-quality verdict, production admission measurement, API/display run, early-stop result, or training result.

The maintainer authorized local Vulkan baseline execution and selected “Baseline before training discussion.” One native pass ran for each approved source. No training/backward/update, paid or cloud API, HOLDOUT/EVAL access, production mutation, or #160 implementation occurred.

## Fixed execution and provenance

- ES2009d: source zero through 180 s plus a real 16 s context tail; evaluation ends at 180 s.
- EN2009d: source zero through 48.4 s.
- Total approved audio: 244.4 s; combined model wall: 261.526 s, below the 1,800 s cap.
- Backend/device: Vulkan on the local AMD Radeon RX 7900 XTX.
- Profile: chunk 6, left 1, right 7, FIFO 188, speaker cache 188, update 144; 1,280-sample frames, 1,600-sample confirmation, threshold 0.5, 16 kHz.
- Executable SHA-256: `1a1ad34ed2a778ffbfcd0a360443db30591a16587d5a30ac6f271c3062ae7098`.
- Model SHA-256: `62faec7b99ad23e323087597604b50728abe85089b6364970b019a845547bf99`.
- Runtime archive, ownership override, and decoder hashes are pinned in `baseline_config.json` and `RESULT.json`.

The exact runner revision used for the two native passes was not pinned and cannot be recovered from the available history. `runner_sha256_after_analysis_fix` is explicitly post-hoc and is not the executed-runner identity. This is partial runner-revision provenance; no executed hash is inferred.

Each source used one continuous native process and one reference/producer generation from source zero. Logical parent seals did not reset model/reference state. Source end was the only discontinuity.

## What was actually measured

Native availability, receiver receipt/consumption, confirmation, and experiment admission invocation use recorded Windows QueryPerformanceCounter values. The admission rule was deliberately no-wait: a frozen accepted-text parent became due at its retained source-span end. **That cutoff is an experimental replay choice, not measured ASR-final or production translation-admission timing.** No historical admission was interpolated, but no production ASR, translation API, or display was executed.

Offline verification reconstructed the transition decoder from the raw chunk probabilities and receipts and checked the recorded assignment cutoffs. For all 38 admitted parents (28 ES, 10 EN), the native frame covering the parent span end arrived after the artificial cutoff: 0.620–1.046 s for ES and 0.616–1.016 s for EN. Thus full-span evidence was unavailable for every parent at that cutoff. This affects all 38 parents; it is not proof of a receiver defect or an actual ASR-admission blocker.

Six parents nevertheless contained a confirmed transition received before their cutoff: 2 ES and 4 EN. ES emitted 9 native transitions and matched 3 of 6 annotated non-overlap transitions within 250 ms; EN emitted 6 and matched 2 of 3. One ES accepted token in the relevant material remained unmapped. These observations establish partial timely evidence, not complete downstream coverage.

The raw source records retain `source.annotation_coverage.retained_parent_guards={}`. A post-hoc join against annotations of the recorded accepted-parent inputs yields ES 9 same-speaker / 19 verified-change / 3 mixed-or-overlap and EN 4 / 6 / 4. Those counts characterize the recorded inputs; they are not proof that runtime guard labels were present during execution. The prefixes also contain annotation overlap witnesses.

The fixed selected predicate was preserved exactly: suppress only a boundary with the same transition/uncertainty key, differing relation, and one UNKNOWN; a merged unit remains UNKNOWN. Direct CURRENT↔OTHER and straddle/transition keys remain boundaries, and whole-parent eligibility/coverage/fallback remains unchanged. The predicate changed 0 parents and removed 0 boundaries. Accepted text/order/source mapping was conserved with 0 failures. The no-active-PSEM arm is identical accepted-text, one-unit replay.

No wait, deadline, or fixed-policy change is recommended or authorized by this result. Training may proceed after its separate approval despite this named cutoff gap; policy repair is not a mandatory prerequisite.

## Learning target and limits

The retained teacher outputs are usable raw soft targets: four independent probability slots per frame, with source support, consumed frontier, validity/generation metadata, and the exact `diar.probs.f32` tensor plus JSON geometry. They must not be interpreted as one mutually exclusive four-speaker class.

This baseline does not support a general early stop, a downstream/translation benefit, or a negative teacher result. It also does not establish full-path cost, so it cannot yet support a general teacher-versus-student cost benefit.

## Cost and limitations

| Source | paced path wall | native CPU | native peak working set | attributable dedicated GPU peak |
|---|---:|---:|---:|---:|
| ES2009d | 196.573 s | 1.016 s | 339,570,688 B | 167,706,624 B |
| EN2009d | 48.737 s | 0.125 s | 314,396,672 B | 167,706,624 B |

GPU memory is the Windows per-native-PID `GPU Process Memory` dedicated-usage counter (194 ES samples, 48 EN samples). It excludes shared memory and driver allocations and is not global GPU usage. GPU processing time/compute RTF was not separately observable from paced wall.

The original receiver-wrapper CPU and RSS calls returned raw zero because a Windows pseudo-handle was passed without 64-bit ctypes signatures. Those raw failures remain recorded and are not reinterpreted as zero cost. Consequently, native RSS is only a complete-path lower bound: **receiver cost, complete-path peak RSS, and therefore R2 cost completion remain unavailable.** The helper now uses a real process handle and explicit ctypes signatures for future runs; its current-process smoke check does not revise the original measurements.

## Evidence and verification

- Summary and decision: `RESULT.json`, `FINDINGS.json`.
- Envelope and identities: `baseline_config.json`, `READINESS.json`.
- Focused offline verification: `VERIFICATION.json`.
- Corrected helper smoke proof: `INSTRUMENTATION_SMOKE.json`.
- Per-source timing, transitions, cost, and artifact manifest: `runs/ES2009d/RESULT.json`, `runs/EN2009d/RESULT.json`.
- Required raw evidence: `runs/*/native-events.jsonl.gz`, `runs/*/receiver-assignments.jsonl.gz`, and `runs/*/dump/diar.probs.{f32,json}`.

The verifier validates manifest hashes, reconstructs transition boundaries/frontiers/slots and QPC receipts from `native-events.jsonl.gz`, recomputes zero-wait cutoff mapping and timely-parent counts against `receiver-assignments.jsonl.gz`, and checks the retained four-slot tensor geometry and values against raw chunk probabilities. It requires neither ignored input WAVs nor hidden/logit tensors.
Scratch-copy tamper checks refreshed the affected manifest hash to avoid testing only the hash gate: the verifier rejected a changed causal cutoff as `raw no-wait cutoff mapping` and a changed soft value as `soft tensor/raw correspondence`. The implementation-owner execution receipt is retained separately in `REPAIR_CHECKS.json`, so rerunning the ordinary verifier does not overwrite it; scratch copies were removed.

The source projections (`runs/*/input.wav`) and large hidden/logit tensors (`runs/*/dump/diar.{hidden,logits}.f32`) remain locally available but are ignored by `.gitignore`. The required probability tensors are tracked. Original WAV locations and sample bounds are in `baseline_config.json`; projection hashes remain in each source result. External personal-machine paths are a reproduction limitation, and this report does not authorize a model rerun.

Offline derivation and verification commands:

```text
python -B experiments/psem_streaming_student/analyze_baseline.py
python -B experiments/psem_streaming_student/run_baseline.py verify
```

The earlier `prepare` and `execute` commands in `FINDINGS.json` are historical execution provenance, not permission to repeat a native run. Architecture remained experiment-local; product source and production behavior did not change.
