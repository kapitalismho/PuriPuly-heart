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

## Staged pause/resume control

`stages.py` is a durable sequential controller for future stage-based work. The currently shipped `stage_plan.json` contains only two implemented, bounded environment checks: the isolated WSL environment's `pip check`, followed by the default no-autograd GPU backend probe. It contains no student, teacher, dataset, training, backward, optimizer, API, cloud, or HOLDOUT/EVAL stage. Real training stages can be added only after they exist and are separately authorized.

Use one stable state root for a run. These commands are repository-relative:

```text
python -B experiments/psem_streaming_student/stages.py run --state-root experiments/psem_streaming_student/.stage-control/issue-164
python -B experiments/psem_streaming_student/stages.py pause --state-root experiments/psem_streaming_student/.stage-control/issue-164
python -B experiments/psem_streaming_student/stages.py status --state-root experiments/psem_streaming_student/.stage-control/issue-164
python -B experiments/psem_streaming_student/stages.py resume --state-root experiments/psem_streaming_student/.stage-control/issue-164
```

Map “잠시 중단” to the `pause` command and “재개” to `resume`. An accepted pause is durable across controller-process restarts. If no stage is running, it prevents the next stage from launching. If a stage is running, that child is allowed to finish; only after exit code zero, required outputs, their hashes, logs, and the successful outcome are atomically stored does the controller stop before the next stage. `pause` and `status` read the frozen run state and keep working if the original plan path is changed or removed. `status` reports the current stage, completed stages, next unfinished stage, pause request, run status, and any interrupted-child warning.

`resume` still verifies the requested plan and refuses changed settings. If the original external plan is no longer available, explicitly select the immutable copy:

```text
python -B experiments/psem_streaming_student/stages.py resume --state-root experiments/psem_streaming_student/.stage-control/issue-164 --plan experiments/psem_streaming_student/.stage-control/issue-164/frozen_plan.json
```

The manifest has exactly two top-level keys: integer `schema_version` and non-empty `stages`. Each stage accepts exactly `id`, non-empty string-array `argv`, integer `timeout_seconds` from 1 through 86,400, and run-root-relative `required_outputs`. Supported substitutions are `{python}`, `{run_root}`, `{run_root_wsl}`, `{workflow_root}`, and `{workflow_root_wsl}`. Commands use `shell=False`; unknown keys, duplicate or unsafe IDs, unbounded timeouts, and escaping output paths are rejected.

The frozen plan hash and per-command hashes bind completed state to that run. One driver lock prevents concurrent `run`/`resume`, failures retain their logs and block later stages, and fixed stage timeouts terminate only timed-out owned child process trees. `pause` never kills the current stage. A killed driver can leave its recorded child still running and writing outputs. On the next `resume`, the state becomes `INTERRUPTED` and exposes the last recorded stage/PID plus a warning that the child may still run and that its uncommitted outputs are untrusted; the controller neither blindly kills that PID nor claims completion or retries it. `INTERRUPTED` therefore does not mean the process tree stopped. This is graceful stage-boundary pause/resume, not crash recovery or an intra-training checkpoint promise.

If initialization was interrupted after writing only a valid frozen plan and lock files, the next matching `run` completes initialization once. Any mismatched plan or other content in that partial state root fails closed without deleting or overwriting it.

The harmless two-subprocess controller smoke and focused boundary/locking/failure/plan-identity checks are recorded in `STAGED_CONTROL_VERIFICATION.json`. Runtime state under `.stage-control/` is intentionally ignored; its state, frozen plan, logs, and receipts must remain together for resume.

## Bounded WSL GT probe: original failed attempts

The maintainer authorized four stages: GPU backward/update verification, a minimal aligned FIT input and independent student, a few GT-only updates with checkpoint reload/inference, and a measured report. The Director fixed one seed and architecture, at most 300 unique FIT audio seconds, and 20 total optimizer updates: one backend fixture plus nineteen GT updates. The approximately two-hour target was not an authorization for extra runs, KD, new teacher passes, evaluation audio, paid calls, or production changes.

The actual runs used WSL Ubuntu, ROCm 7.2.1, PyTorch `2.9.1+rocm7.2.1.gitff65f5bc`, and the RX 7900 XTX. `GT_PROBE_RESULT.json` records the observed boundaries and missing evidence.

| Stage | Actual result |
| --- | --- |
| GPU backend | One Conv1d/GRU/linear fixture update completed with finite gradients, loss 0.6978195906, and nonzero convolution-weight delta 0.2022929192. |
| FIT and student preparation | Passed. One permitted TRAIN source, `ami_ES2005a`, from sample 0 through 4,620,800: 288.8 seconds at 16 kHz. The independent student has 5,940,740 trainable parameters. |
| GT updates and checkpoint | The configured nineteen-update loop reached its post-loop optimizer-state audit, which raised before checkpoint persistence. No reloadable checkpoint remains. |
| Report | Failure and resource accounting are retained here; learned inference, quality, and cost conclusions remain unavailable. |

The student is fixed causal 64-bin log-Mel, three stride-two causal convolution blocks, two unidirectional GRU layers of width 512, and four independent activity outputs. It does not need Sortformer at inference. Raw, unsnapped source intervals produce fractional per-speaker occupancy in half-open 1,280-sample bins; overlap remains multi-active. The 397 raw intervals and 23 relation-only nonlexical masks are retained in the prepared input. Relation-only masks do not remove otherwise valid activity supervision. The first three outputs lack the full 4,880-sample causal receptive field and are excluded. Output frontiers match their target-bin ends without future audio.

### Two distinct implementation failures

1. At `400caac18ee6e5105789b6a1b886aa9a5ad1e39b`, Windows backslashes were consumed by the WSL command boundary. `readonly RUN_ROOT="$(wslpath ...)"` also hid the conversion failure. The backend completed, but wrote its receipt and frozen configuration to the workflow directory; the controller correctly failed on missing required outputs. Those files were recovered unchanged, and the backend was not rerun. The path repair adds `{run_root_wsl}`, uses direct WSL execution, and rejects an invalid launcher root before Python starts.
2. At `b9711f1b9a5bc595204b1f43571bce6b0976c644`, preparation succeeded in the correct directory. Training then failed at the post-loop counter audit: `value` was already a scalar optimizer-step tensor, but code indexed `value["step"]`. The same error was present in reload diagnostics. Both reads have been repaired to extract the scalar directly, and checkpoint persistence now precedes post-training diagnostics. The scalar-counter repair was checked without model execution or another optimizer update; the repaired learned save/reload path has not yet been exercised.

The nineteen GT updates are charged from the pinned code and the post-loop traceback, not from persisted per-step receipts. The loop's finite/change checks precede that traceback, but its loss values, step timings, final numeric state, and memory statistics were only in memory and were lost. They are not reconstructed or presented as measured results. The conservative cumulative budget is therefore **20 spent, zero additional updates authorized**.

Retained host command walls were 16.969345 seconds for the backend command, 13.054090 seconds for preparation, and 8.570759 seconds for the failed training command. These include process/WSL overhead and are not isolated GPU compute, training-loop timing, mean step latency, or inference RTF.

### Retained artifacts and next decision

- `.stage-control/issue-164-gt-probe/`: original failed state and logs, recovered backend receipt/configuration, and host launch provenance.
- `.stage-control/issue-164-gt-continuation/`: separate failed state, explicit continuation lineage, the prepared input, preparation receipt, and exact training traceback.
- `prepared_fit.pt`: 18,608,719 bytes, SHA-256 `b367960d2dae9ea8b7b9571af23015f723ac5d72f24125d3c8a5a9ac7576b682`. It contains the authorized waveform prefix, raw GT provenance and projected targets; inspect with the pinned environment's `torch.load(..., map_location="cpu", weights_only=False)`. This is an input artifact, not a learned checkpoint.

Both failed controller states remain unchanged and refuse automatic continuation. No failed state was relabeled as successful. Runtime artifact directories are local and ignored; retain each complete directory with its configuration and logs. No audio or model artifact was published.

At this initial blocked point, the next decision was authorization for two additional GT updates using the same architecture, seed, and the first 30.4 seconds of the already authorized source. The maintainer subsequently approved exactly those two updates, bringing the cumulative cap to 22; the successful completion is recorded below. The earlier runner failure is not evidence against acoustic PSEM or against the student architecture. The complete GT/KD comparison in issue #164 remains open.

Independent review was unavailable: its transport failed before reading the candidate, and fresh reviewer allocation could not resolve a configured model. No independent-review pass is claimed. Direct checks and actual runtime evidence are distinguished above. Product code and the selected downstream policy were unchanged; no system architecture change is proposed by this experiment.

## Two-update checkpoint completion: passed

After explicit approval of two additional GT updates, the frozen implementation `a58a48edc894f655aaa0f3bc13f9d925369b7d01` ran `gt_probe_checkpoint_plan.json` in actual WSL. The backend was not repeated. The separate run `.stage-control/issue-164-gt-checkpoint/` completed preparation, two updates with fresh-process checkpoint reload/inference, and report generation, all with exit code zero. A fresh controller process independently read its persisted `COMPLETED` state.

This completes the requested four-stage **training-path smoke**, not issue #164's full quality-cost comparison. The saved checkpoint contains exactly **two new GT updates from the original initialization**; it does not recover or continue the lost nineteen-update weights. `GT_CHECKPOINT_RESULT.json` is a byte-identical tracked copy of the successful runtime result, SHA-256 `3a172eeb237c8b23fe91b343b0167dcaa11f642c4ee61a2f2d3fdecf12f4bfd8`. `GT_PROBE_RESULT.json` remains the historical failure record.

### Actual observations

| Check | Result |
| --- | --- |
| Training device | RX 7900 XTX, WSL Ubuntu, PyTorch 2.9.1 / ROCm 7.2.1 |
| Student | 5,940,740 parameters; unchanged architecture and seed 20260915 |
| Current learned input | Source-zero `ami_ES2005a`, 486,400 samples / 30.4 seconds |
| GT update 1 | Loss 0.6919456124; finite gradients; first-parameter delta norm 0.0402700007 |
| GT update 2 | Loss 0.6822207570; finite gradients; delta norm 0.0322614498; numeric state carried from update 1 |
| Synchronized training loop | 2.178627211 seconds; first update 2.124919743 seconds includes first-use overhead; second update 0.052244433 seconds |
| Frozen-weight chunk partition | Maximum logit delta 1.6391e-7; tolerance 1e-4 |
| Midstream numeric-state handoff | Maximum logit delta 1.7881e-7; maximum state delta 4.7684e-7 |
| Fresh-process reload | Optimizer counters equal 2, zero further updates, finite four-output inference, prediction delta exactly 0 |
| Full student inference block | 15.2 seconds of audio processed in 0.025567173 seconds, including H2D, log-Mel, convolutions, GRU, logits, sigmoid, and D2H |
| Inference exclusions | Checkpoint load 0.340519757 seconds, CPU optimizer validation 0.562113972 seconds, and warmup 0.858929030 seconds are separate |
| PyTorch GPU allocation peaks | Training plus frozen-weight diagnostics: 275.90 MiB allocated / 294 MiB reserved; measured inference after warmup: 153.23 MiB allocated / 176 MiB reserved |

These losses come from different chronological chunks and are not evidence of convergence or generalization. The inference number is a single warmed, 15.2-second block replay, not measured 80-ms admission cadence, live latency, or a throughput guarantee. PyTorch allocation counters are not complete process or device memory measurements and are not directly comparable to the native teacher's sampled counters. No teacher/student cost-reduction claim follows from this smoke.

### Checkpoint and budget

The checkpoint is `.stage-control/issue-164-gt-checkpoint/student_gt_checkpoint.pt`: **71,402,202 bytes**, SHA-256 `1928a9ae14d257282a9eefbc8e819876de1f8db325e0565831875dded8a434ac`. It includes model parameters, optimizer state, configuration, global step, RNG, consumed-source identity/frontier, and the actual detached streaming state. Its file size includes training state; it is not an inference-only model size. The saved state follows the declared TBPTT trajectory and is not claimed to equal replaying the entire prefix with the final weights.

Retain this complete run directory, including `state.json`, frozen plan/configuration, continuation lineage, prior failure evidence, `prepared_fit.pt`, checkpoint, `train_receipt.json`, `reload_receipt.json`, `result.json`, and logs. The final model and its inputs remain local and ignored; no upload occurred. An already completed `resume` has no next stage and performs no additional training.

Final optimizer accounting is **1 prior backend fixture + 19 conservatively charged failed-attempt updates + 2 newly checkpointed updates = 22**, with zero reload updates. Current input is 30.4 seconds; cumulative unique FIT audio remains 288.8 seconds, while cumulative training exposure including the approved repetition is 319.2 seconds. Both earlier failed controller states and their evidence remain unchanged.

**Decision:** the isolated WSL GPU training, state-carry, persistence, and standalone inference path is executable. A later bounded GT/KD comparison can now be designed around measured execution rather than an assumed backward backend. Acoustic usefulness, calibration, source-disjoint quality, downstream effects, and complete teacher/student cost remain unmeasured. No additional updates, teacher passes, KD, evaluation audio, API/cloud work, production integration, or issue closure are authorized by this completion. The independent-review limitation above still applies.

## Clean ROCm 10 environment reset

After the maintainer updated the Windows display driver and explicitly requested a clean Linux ROCm setup, the RX 7900 XTX driver was observed as DriverStore `32.0.31041.1004`, matching Adrenalin 26.8.1. The scoped removal transaction removed the 23 inventoried ROCm 7.2.1/ROCDXG packages without an upgrade or `autoremove`, followed by the old dedicated PyTorch environment and obsolete installer directories. The WSL distribution, unrelated development tools, code, datasets, labels, checkpoints, and historical results were retained.

The current runtime is `/opt/psem-streaming-student/rocm-10.0.0-pytorch-2.13.0`: Python 3.12.3, PyTorch `2.13.0+rocm10.0.0`, ROCm SDK packages 10.0.0, and ROCDXG 1.2.2. AMD's released Linux pip packages were installed in the dedicated environment and exercised under WSL; this is not described as the exact package-manager recipe shown by AMD's WSL selector. No Linux `amdgpu-dkms` driver was installed, and this setup did not modify the Windows driver.

`torch.version.hip` reports **7.15.26333**, and both HIP runtime/driver version APIs return `71526333`. These observed identities are retained rather than relabeled as HIP 10. Loaded HIP and HSA libraries come from the new environment's `_rocm_sdk_core` directory; the bridge is `/opt/rocm/lib/librocdxg.so.1.2.2`. The former ROCm 7.2.1 runtime and active launcher references were removed.

Current package pins, archive URLs/hashes, loaded-library paths, removal scope, and preservation checks are in `environment/ENVIRONMENT.json` and `environment/requirements.lock`. The previous receipt and lock remain explicitly historical as `environment/ENVIRONMENT_ROCM_7_2_1_HISTORICAL.json` and `environment/requirements-rocm-7.2.1-historical.lock`. Existing frozen run receipts/configurations remain unchanged. The default environment stage plan and both active launchers now select the new environment.

### Existing checkpoint inference

`environment/ROCM10_INFERENCE_RESULT.json` retains 30 raw timings from a fresh-process, no-grad replay of the same checkpoint and first 243,200 prepared samples (15.2 seconds). Each repetition starts from reset streaming state and measures synchronized H2D waveform transfer, log-Mel, convolution, GRU, logits, sigmoid, and D2H probabilities. Three warmup repetitions are excluded; the model is FP32/eval with one CPU thread.

| Check | Observed result |
| --- | --- |
| Warmed block latency, median | 20.064929 ms |
| Warmed block latency, p90 | 21.265417 ms |
| Output | Finite probabilities, shape `[1, 190, 4]` |
| Source frontiers | Exact agreement with the historical reference |
| Maximum probability delta from historical reference | 5.9604645e-8 |
| PyTorch peak allocated / reserved | 127,120,896 / 150,994,944 bytes |
| Additional optimizer updates or backward calls | Zero |

Checkpoint, prepared input, student source, and successful historical GT result hashes match the pre-removal inventory. The cumulative optimizer count remains **22**. The previous 25.567173-ms observation was a single run under the older driver/framework and a different measurement protocol; it is not a matched baseline for a controlled speedup percentage. These measurements are block inference, not live 80-ms admission latency, training speed, or evidence of acoustic quality. Earlier backward/training proof remains evidence for the historical ROCm 7.2.1 environment, not fresh backward verification on ROCm 10.

This reset changes the isolated experiment environment, not the product architecture or student model. Issue #164 remains open/In progress; full GT/KD training, teacher passes, and quality-cost acceptance remain outside this reset.

The Director exercised the updated default stage plan after WSL restarted: `.stage-control/rocm10-clean-env-smoke/`, run `59ead2543e854300bf36764e34380d68`, completed `dependency-check` and `default-gpu-environment-check`, both with exit code zero. The focused `experiments/psem_streaming_student/tests/test_stages.py` run also passed (nine tests). No new permanent test or benchmark script was added; the temporary inference script was removed after retaining its result.
