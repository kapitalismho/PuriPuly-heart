# #164 Vulkan teacher/receiver baseline

## Status

**Cutoff-conditional partial baseline; the authorized separate rerun completes paced-wall and sampled process-memory observation only.** Process CPU and GPU-kernel compute remain unquantified. This is an exposed-DEV engineering probe, not a teacher-quality verdict, production admission measurement, API/display run, early-stop result, or training result.

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

The original raw source records retain `source.annotation_coverage.retained_parent_guards={}`. A post-hoc join against annotations of the recorded accepted-parent inputs yields ES 9 same-speaker / 19 verified-change / 3 mixed-or-overlap and EN 4 / 6 / 4. Those counts characterize the recorded inputs; they are not proof that runtime guard labels were present during execution. The rerun's populated counts were computed before native launch by annotating the accepted input parents; they likewise are not runtime receiver guard proof. The prefixes also contain annotation overlap witnesses.

The fixed selected predicate was preserved exactly: suppress only a boundary with the same transition/uncertainty key, differing relation, and one UNKNOWN; a merged unit remains UNKNOWN. Direct CURRENT↔OTHER and straddle/transition keys remain boundaries, and whole-parent eligibility/coverage/fallback remains unchanged. The predicate changed 0 parents and removed 0 boundaries. Accepted text/order/source mapping was conserved with 0 failures. The no-active-PSEM arm is identical accepted-text, one-unit replay.

No wait, deadline, or fixed-policy change is recommended or authorized by this result. Training may proceed after its separate approval despite this named cutoff gap; policy repair is not a mandatory prerequisite.

## Learning target and limits

The retained teacher outputs are usable raw soft targets: four independent probability slots per frame, with source support, consumed frontier, validity/generation metadata, and the exact `diar.probs.f32` tensor plus JSON geometry. They must not be interpreted as one mutually exclusive four-speaker class.

This baseline does not support a general early stop, a downstream/translation benefit, or a negative teacher result. The original run did not establish full-path process memory. The separately labeled authorized rerun observes paced wall and sampled native-plus-wrapper process memory, but it does not quantify process CPU, GPU-kernel compute, or teacher-versus-student benefit.

## Original-run cost and limitations

| Source | paced path wall | native CPU API reading (diagnostic) | native peak working set | attributable dedicated GPU peak |
|---|---:|---:|---:|---:|
| ES2009d | 196.573 s | 1.015625 s | 339,570,688 B | 167,706,624 B |
| EN2009d | 48.737 s | 0.125000 s | 314,396,672 B | 167,706,624 B |

GPU memory is the Windows per-native-PID `GPU Process Memory` dedicated-usage counter (194 ES samples, 48 EN samples). It excludes shared memory and driver allocations and is not global GPU usage. GPU processing time/compute RTF was not separately observable from paced wall.

The original receiver-wrapper CPU and RSS API calls returned raw zero. The executed original runner revision is unavailable, so the cause cannot be established and no specific API-misuse, bias, or correction-factor explanation is asserted. Those raw failures remain recorded and are not reinterpreted as zero cost. Native process CPU readings are retained only as quantum-limited diagnostics, and native RSS remains only a complete-path lower bound.

## Evidence and verification

- Summary and decision: `RESULT.json`, `FINDINGS.json`.
- Envelope and identities: `baseline_config.json`, `READINESS.json`.
- Focused offline verification: `VERIFICATION.json`.
- Process-helper smoke observation and its CPU precision limit: `INSTRUMENTATION_SMOKE.json`.
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

## Authorized cost rerun: wall and sampled memory complete

`#164-BASELINE-RERUN-1` executed exactly one additional Vulkan native pass for each unchanged source/profile under the 1,800 s combined cap. The run processed the same 244.4 s of audio in 261.185 s combined paced orchestration wall. The two timed source loops sum to 245.073 s (196.364 s ES plus 48.708 s EN); the remaining 16.112 s is host preparation, serialization, and between-source orchestration outside those loops. The combined wall is therefore a conservative cap comparison, not per-source preparation CPU. The run performed no training/backward, install, cloud/API, policy, added-wait, HOLDOUT/EVAL, or debugging pass.

The executed revision was pinned before launch as `d49a1b5d6da7888ba7b19f8cb5bf229fe0b7ca0c`. The committed runner/config/executed-analyzer blobs matched before either native process started. Recorded SHA-256 identities are:

- runner `8de3980f56b36ef20ae952809363982101ea2a37ec66a315d6379ee8ad5ad7fe`
- executed analyzer `71676688c440569e72469f3fd6d19bfb93b059d805e03ff99307bccdccaf9b7a`
- post-hoc reporting analyzer `5cd695d0ee7dda89cd90bcf26edc290d5e284f1cfe3b10462245b48f4cc50a44`
- rerun config `13ef051d6243c8c2d0ece46580cb46a5407e08a61bccbf06f56345fca5a53c20`
- executable `1a1ad34ed2a778ffbfcd0a360443db30591a16587d5a30ac6f271c3062ae7098`
- model `62faec7b99ad23e323087597604b50728abe85089b6364970b019a845547bf99`

Runtime archive, ownership override, and decoder hashes are in `cost_rerun/RESULT.json`. The post-hoc analyzer change only repairs interpretation and metadata; it does not re-pin the executed run. All mutable evidence is isolated under `cost_rerun/`; the original `runs/` and top-level result records retain their original failed receiver measurements and provenance.

### Process wall and sampled memory

| Source | paced wall | native CPU API reading (diagnostic) | wrapper CPU API reading (diagnostic) | native sampled peak | wrapper sampled peak | paired whole-path peak | dedicated GPU peak |
|---|---:|---:|---:|---:|---:|---:|---:|
| ES2009d | 196.364 s | 0.484375 s | 0.015625 s | 340,058,112 B | 82,956,288 B | 422,780,928 B | 167,706,624 B |
| EN2009d | 48.708 s | 0.234375 s | 0.015625 s | 314,597,376 B | 81,678,336 B | 395,227,136 B | 167,706,624 B |

Every process CPU reading is a multiple of the 15.625 ms API quantum, and each wrapper reading is exactly one quantum at the instrument floor. Native readings also varied materially under the same geometry: the original ES value is 2.10 times the rerun value, while the rerun EN value is 1.875 times the original. These raw values are retained for diagnosis, but true native and wrapper CPU cost is unquantified; no speedup, efficiency result, universal bias, combined CPU total, or correction factor is claimed.

RSS used 13,319 ES and 3,532 EN QPC-stamped sample pairs. Readings within each pair were back-to-back and near-simultaneous, not atomic; median sampling interval was 15.506 ms ES and 15.504 ms EN, with a maximum of 21.186 ms. At the actual aggregate peaks, ES was 339,947,520 B native plus 82,833,408 B wrapper, and EN was 314,597,376 B plus 80,629,760 B. The whole-path peak is the maximum sampled pair, not the sum of independent process peaks. Wrapper RSS includes Python orchestration and measurement instrumentation, not a production-only receiver.

Dedicated GPU memory came from 1 Hz `typeperf` per-process counters for native PID 30788 (ES, 194 samples) and native PID 35808 (EN, 48 samples). Paced wall is not compute time or compute RTF. Native `service_us` summed to 196.001 s ES and 48.433 s EN and includes pacing. The named per-chunk trace components (`frontend_us + graph_a_us + graph_b_us + host_us`) summed to 13.397 s ES and 3.440 s EN; initialization was 0.261 s and 0.214 s. These are native-reported component timers, not a measured GPU-kernel total, and no compute RTF is claimed.

The repeated causal result remained cutoff-conditional: all 38 parents lacked the end-covering frame at the synthetic source-end cutoff, while 2 ES and 4 EN parents contained timely confirmed transitions. Rerun retained-parent counts were pre-launch input annotations, not runtime guard proof. The selected policy changed 0 parents, removed 0 boundaries, and conserved accepted text. This does not convert the replay cutoff into actual ASR/translation/display timing or establish downstream benefit.

### Rerun evidence

- Config and preparation: `cost_rerun_config.json`, `cost_rerun/READINESS.json`, `cost_rerun/PREEXECUTION_SMOKE.json`.
- Summary and analysis: `cost_rerun/RESULT.json`, `cost_rerun/FINDINGS.json`.
- Final offline execution-artifact verification: `cost_rerun/VERIFICATION.json` (`passed`, no failures; executed runner/config/analyzer validated against immutable `d49a1b5...` blobs and current post-hoc analyzer validated separately).
- Post-hoc reporting repair verification: `REPORTING_REPAIR_VERIFICATION.json` (both original and rerun verifiers passed on the repaired candidate; executed-hash and source-revision tampering were rejected; 22/22 raw artifact hashes matched and raw artifacts are byte-identical to candidate `977c22ab`).
- Per-source raw events, assignments, soft tensors, trace, GPU counter, paired memory samples, and manifests: `cost_rerun/runs/{ES2009d,EN2009d}/`.

Remaining uncertainty: RSS is dense, near-simultaneous sampling rather than an atomic continuous maximum; wrapper RSS includes Python orchestration/instrumentation; the dedicated-GPU counter excludes shared/driver allocations; true process CPU and GPU-kernel compute were not quantified; and the synthetic no-wait cutoff is not production ASR admission. The machine-cost result is paced wall plus sampled process memory only. No further baseline rerun, training, or policy repair is a prerequisite implied by this report.

Executed commands:

```text
python -B experiments/psem_streaming_student/run_baseline.py execute --config experiments/psem_streaming_student/cost_rerun_config.json --source-revision d49a1b5d6da7888ba7b19f8cb5bf229fe0b7ca0c
python -B experiments/psem_streaming_student/analyze_baseline.py --config experiments/psem_streaming_student/cost_rerun_config.json
python -B experiments/psem_streaming_student/run_baseline.py verify --config experiments/psem_streaming_student/cost_rerun_config.json
```

The authorization is consumed: these commands are provenance, not permission for another native pass.
