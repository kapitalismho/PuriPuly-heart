# Gemma 4 E4B CPU Inference Optimization Plan

## 1. Objective

Optimize CPU-only inference for Gemma 4 E4B QAT with MTP for low-latency translation requests.

The main target environment runs alongside VRChat, so the goal is not maximum throughput at any cost. The optimization should balance:

- Low end-to-end request latency
- Low TTFT and prefill latency
- Efficient MTP speculative decoding
- Controlled CPU usage
- Stable behavior across different user CPUs
- No degradation in translation quality

Polling and process-priority tuning are intentionally excluded from this plan.

---

## 2. Reference Workload

Representative request shape:

```text
Total input context      ~900 tokens
Fixed system prefix      ~700 tokens
New prefill              ~200 tokens
Average output           ~12 tokens
Maximum real context     <1500 tokens
Parallel                 1
CPU-only inference
```

The fixed ~700-token system prompt is assumed to hit the prompt cache.

Therefore, the steady-state optimization target is approximately:

```text
~200-token uncached prefill
+
~12-token generation
```

---

## 3. Current Baseline Configuration

```text
--device none
--n-gpu-layers 0

--threads 4
--threads-batch 4

--ctx-size 4096
--parallel 1
--batch-size 512
--ubatch-size 512

--cache-type-k f16
--cache-type-v f16
--cache-prompt

--spec-type draft-mtp
--spec-draft-n-max 4
--spec-draft-n-min 1

--spec-draft-device none
--spec-draft-ngl 0
--spec-draft-threads 4
--spec-draft-threads-batch 4

--spec-draft-type-k f16
--spec-draft-type-v f16

--flash-attn auto
```

This configuration is the baseline for all experiments.

---

# Stage 1. Reduce Context Size

## Change

From:

```text
--ctx-size 4096
```

To:

```text
--ctx-size 2048
```

## Reason

The actual context is guaranteed to remain below 1500 tokens, so 2048 provides sufficient safety margin.

Expected benefits:

- Lower KV cache reservation
- Lower context-related memory usage
- Possible cache-locality improvement
- Less unnecessary RAM usage

This does **not** halve the compute cost of a 900-token prompt. The main gain is memory-related.

## Experiment

```text
A0: ctx=4096
A1: ctx=2048
```

Keep all other settings identical.

If there is no regression, adopt 2048 as the default.

---

# Stage 2. Flash Attention Comparison

## Compare

```text
B0:
--flash-attn auto

B1:
--flash-attn off
```

Keep target and draft KV cache types fixed to:

```text
K = f16
V = f16
```

## Reason

Flash Attention behavior on CPU is hardware-dependent.

For this workload:

```text
uncached prefill ~200 tokens
output           ~12 tokens
```

short CPU prefill latency matters a lot.

This stage isolates Flash Attention and avoids changing unrelated parameters.

## Selection Criteria

Prioritize:

1. End-to-end median latency
2. p95 latency
3. TTFT
4. Prompt-eval time

Use the winner as the baseline for later stages.

---

# Stage 3. Target CPU Thread Tuning

Current fixed values:

```text
--threads 4
--threads-batch 4
```

Dynamic CPU budgeting is **not** implemented yet.

First determine the best fixed values experimentally.

## Suggested Candidates

Use only values that are valid on the test machine.

Example:

```text
C1:
threads=2
threads-batch=2

C2:
threads=4
threads-batch=4

C3:
threads=6
threads-batch=6

C4:
threads=8
threads-batch=8
```

## Separate Prefill Thread Test

After identifying the best `threads` value, vary only `threads-batch`.

Example if `threads=4` wins:

```text
D1:
threads=4
threads-batch=4

D2:
threads=4
threads-batch=6

D3:
threads=4
threads-batch=8
```

## Goal

`--threads` primarily affects token generation.

`--threads-batch` primarily affects prompt/prefill processing.

The goal is to determine whether they should remain equal or use different values.

---

# Stage 4. MTP Draft Thread Tuning

After target-thread tuning, optimize the MTP drafter independently.

Current values:

```text
--spec-draft-threads 4
--spec-draft-threads-batch 4
```

## Minimum Experiment Set

```text
E1:
draft_threads=1
draft_threads_batch=1

E2:
draft_threads=2
draft_threads_batch=2

E3:
draft_threads=4
draft_threads_batch=4
```

Only test larger values if the results justify it.

## Reason

The E4B MTP drafter is much smaller than the target model.

Too many draft threads may cause:

- Thread synchronization overhead
- Unnecessary competition with other applications
- Little or no MTP latency improvement

`2` draft threads is therefore an important candidate.

---

# Stage 5. MTP Draft-Length Tuning

First optimize `n_max` while keeping:

```text
--spec-draft-p-min 0
--spec-draft-n-min 1
```

## Experiment

```text
F1:
n_max=2

F2:
n_max=3

F3:
n_max=4
```

## Reason

The average output is only about 12 tokens.

Long speculative drafts may increase potential progress per target verification, but they also increase:

- Wasted draft work
- Drafter CPU cost
- Rejected speculative tokens

## Selection Criterion

Choose the value with the lowest **real 12-token request end-to-end latency**.

Do not optimize for acceptance rate alone.

---

# Stage 6. MTP `p_min` Tuning

Keep the winning `n_max` from Stage 5 fixed.

## First Comparison

```text
G1:
p_min=0.0

G2:
p_min=0.7
```

## Goal

`p_min` allows the drafter to stop early when confidence becomes low.

On CPU, this may reduce wasted speculative work.

## Optional Fine-Tuning

Only if `0.7` clearly beats `0.0`, test:

```text
0.6
0.7
0.8
```

If the difference is negligible, skip further tuning.

---

# Stage 7. MTP `n_min` Tuning

Test `n_min` only after a non-zero `p_min` has been selected.

## Experiment

```text
H1:
n_min=1

H2:
n_min=2
```

## Meaning

If the drafter stops after generating only one token:

```text
n_min=1
-> use the one-token speculative draft

n_min=2
-> discard it and fall back to the normal target path
```

This experiment determines whether one-token speculative attempts are worthwhile on CPU.

---

# Stage 8. KV Cache Quantization - Optional

Run this stage only after latency optimization is mostly complete.

Default:

```text
target K/V = f16/f16
draft  K/V = f16/f16
```

## Candidates

```text
I1:
target f16/f16
draft  f16/f16

I2:
target q8_0/q8_0
draft  f16/f16

I3:
target q8_0/q8_0
draft  q8_0/q8_0
```

Account for runtime constraints such as Flash Attention requirements when using quantized V cache.

## Reason

With a maximum context below 1500 tokens, KV cache memory is unlikely to be a major bottleneck.

Treat Q8 KV primarily as a **RAM optimization**, not a guaranteed latency optimization.

If end-to-end latency worsens, keep F16 KV.

---

# Stage 9. Final Fixed CPU Profile

After Stages 1-8, finalize:

```text
ctx_size

flash_attention

target_threads
target_threads_batch

draft_threads
draft_threads_batch

mtp_n_max
mtp_n_min
mtp_p_min

target_kv_type
draft_kv_type
```

An example candidate might look like:

```text
--ctx-size 2048
--parallel 1

--batch-size 512
--ubatch-size 512

--threads 4
--threads-batch 4

--cache-type-k f16
--cache-type-v f16
--cache-prompt

--spec-type draft-mtp
--spec-draft-n-max 2
--spec-draft-n-min 1
--spec-draft-p-min 0.7

--spec-draft-threads 2
--spec-draft-threads-batch 2

--spec-draft-type-k f16
--spec-draft-type-v f16

--flash-attn off
```

These values are only an example. The final values must come from benchmark results.

---

# Stage 10. Dynamic CPU Budget `B` - Requires Explicit User Approval

This stage must be considered **only after Stages 1-9 are complete**.

Do not implement it automatically.

First report the benchmark results to the user and obtain **explicit approval** before changing runtime behavior.

## Goal

Replace a fixed setting such as:

```text
threads=4
```

with a CPU-dependent thread budget.

Because the application normally runs alongside VRChat, the runtime should not automatically consume all available CPU cores.

Initial concept:

```text
P = number of physical CPU cores available to the current process

B = ceil(P * CPU_USAGE_FRACTION)
```

For example:

```text
CPU_USAGE_FRACTION = 0.5
```

would produce approximately:

| CPU | Physical cores P | CPU budget B |
|---|---:|---:|
| 4C | 4 | 2 |
| 6C | 6 | 3 |
| 8C | 8 | 4 |
| 12C | 12 | 6 |
| 16C | 16 | 8 |

The initial target mapping could be:

```text
threads       = B
threads_batch = B
```

The MTP drafter policy should be derived from Stage 4 results, for example:

```text
draft_threads = min(B, 2)
```

or:

```text
draft_threads = min(B, 4)
```

The exact formula must be based on benchmark data.

## Why This Comes Last

A simple rule such as:

```text
B = physical_cores * 0.5
```

is not guaranteed to be optimal on every CPU.

The fixed-thread experiments may show something like:

```text
2 threads -> low CPU use, slightly slower
4 threads -> best balance
6 threads -> faster translation but excessive contention
8 threads -> little additional benefit
```

That data should shape the dynamic policy.

Required process:

```text
Fixed-thread experiments
        |
        v
Benchmark result analysis
        |
        v
Propose dynamic-B policy
        |
        v
Report results to user
        |
        v
Obtain explicit user approval
        |
        v
Implement dynamic B in runtime_profile.py
```

---

# Benchmark Metrics

Use a workload close to the real translation request:

```text
~700 cached tokens
+
~200 new tokens
->
~12 output tokens
```

Run at least 30 measured repetitions per configuration after warm-up.

Prioritize:

1. End-to-end median latency
2. p95 latency
3. TTFT
4. Prompt-eval time
5. Generation time
6. MTP drafted-token count
7. MTP accepted-token count
8. Process RSS
9. CPU utilization

The final optimization target is **real translation-request latency**, not raw tokens/sec.

---

# Execution Order

```text
Baseline
   |
   v
Stage 1
ctx 4096 -> 2048
   |
   v
Stage 2
Flash Attention auto vs off
   |
   v
Stage 3
Target threads / threads-batch
   |
   v
Stage 4
MTP draft threads
   |
   v
Stage 5
MTP n_max 2 / 3 / 4
   |
   v
Stage 6
p_min 0 / 0.7
   |
   v
Stage 7
n_min 1 / 2
   |
   v
Stage 8
Optional KV Q8 test
   |
   v
Stage 9
Finalize fixed CPU profile
   |
   v
Benchmark result report
   |
   v
================================
Explicit user approval required
================================
   |
   v
Stage 10
Implement dynamic CPU budget B
```

Core principle:

**First find the best fixed configuration under controlled conditions. Then, and only with explicit user approval, convert those findings into a dynamic CPU-budget policy.**
