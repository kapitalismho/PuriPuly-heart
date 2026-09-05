# Handoff: Policy-Only Smart Turn Evaluator Correction and High-Repetition Validation

## Repository

- Repository: [`https://github.com/kapitalismho/PuriPuly-heart`](https://github.com/kapitalismho/PuriPuly-heart)
- Target branch: `dev`

Assume no prior knowledge of the previous implementation.

---

# 1. Assignment

Correct the known policy-evaluator issues and rerun only the policy analysis.

Do not rerun:

- Smart Turn audio inference.
- CPU latency benchmarks.
- Cold-start benchmarks.
- One-thread versus two-thread benchmarks.
- Provider performance tests.

Reuse the existing CPU-int8 prediction artifacts generated with:

```text
Smart Turn v3.2 CPU int8
CPUExecutionProvider
2 intra-op threads
1 inter-op thread
sequential execution

```

The main decision is whether PuriPuly should proceed to shadow testing with:

```text
P1:
224 ms Smart Turn probe
800 ms hard timeout

```

or:

```text
P2:
224 ms Smart Turn probe
512 ms fresh Smart Turn probe
800 ms hard timeout

```

Evaluate Korean, Japanese, English, and Chinese independently.

---

# 2. Existing artifacts

Reuse:

```text
cpu_predictions_ko.parquet
cpu_predictions_ja.parquet
cpu_predictions_en.parquet
cpu_predictions_zh.parquet

```

Each row must contain or be joinable to:

```text
language
conversation_id
turn_id
span_id
label
span_duration_ms

score_224
inference_latency_224_ms

score_512
inference_latency_512_ms

```

Do not substitute the earlier external GPU prediction artifacts.

Do not regenerate scores unless the existing CPU prediction files fail integrity validation.

---

# 3. Known evaluator problems to correct

## Problem A: hard-timeout rate was calculated over all spans

The previous evaluator treated all unaccepted pause spans as hard timeouts.

That incorrectly includes hold pauses where the speaker resumed before 800 ms.

For example:

```text
300 ms hold pause
Smart Turn does not accept
speaker resumes at 300 ms

```

This is not an 800 ms timeout.

Create two separate metrics.

### EOT timeout rate

```text
eot_timeout_rate =
number of true EOT spans that end through the 800 ms timeout
/
number of true EOT spans

```

This is the user-facing completion-delay metric.

Use this metric in policy-selection gates.

### Unresolved span rate

```text
unresolved_span_rate =
number of all spans with no accepted Smart Turn result
/
number of all spans

```

This is a diagnostic metric only.

Do not use `unresolved_span_rate` as the product timeout gate.

---

## Problem B: P3 matching failed sanity validation

The previous P3 results were labeled as matched to P1 even though the false-cutoff rates were far apart.

Any matched comparison must satisfy:

```text
absolute training false-cutoff difference <= 0.5 percentage points

```

If no candidate satisfies this condition:

```text
matched candidate = unavailable

```

Do not return the first candidate, boundary candidate, or nearest candidate without explicitly recording the mismatch.

Required assertion:

```python
assert abs(
    selected_training_false_cutoff
    - target_training_false_cutoff
) <= 0.005

```

If the assertion cannot be satisfied, return a structured unavailable result rather than raising during the full experiment.

---

## Problem C: threshold selection and evaluation need stronger separation

Thresholds must never be selected using the outer held-out fold.

Use nested group cross-validation.

---

# 4. Policies

## B0: current fixed VAD

```text
Endpoint at 512 ms

```

## B1: fixed long VAD

```text
Endpoint at 800 ms

```

## P1: one Smart Turn probe

```text
At 224 ms:
    run Smart Turn

If the accepted result arrives while the pause is active:
    end at result-arrival time

Otherwise:
    end at 800 ms

```

## P2: two probes with one shared threshold

```text
At 224 ms:
    run Smart Turn

If not ended, at 512 ms:
    run a fresh Smart Turn evaluation

First valid accepted result wins.

Otherwise:
    end at 800 ms

```

Use the same threshold for both P2 probes in the primary experiment.

## P3: independent thresholds

```text
T512 >= T224

```

Do not evaluate P3 initially.

Evaluate P3 only for a language where P2 passes the predefined value gate against P1.

This is a staged experiment:

```text
Stage 1:
B0, B1, P1, P2

Stage 2:
P3 only for languages where P2 proves useful

```

---

# 5. Runtime-aware simulation

Use the existing per-span measured CPU inference latency.

Do not rerun latency tests.

## First probe

```text
arrival_224 =
224 ms + inference_latency_224_ms

```

The result can act only if:

```text
span_duration_ms > arrival_224

```

If the speaker resumes before result arrival, the result is stale.

## Second probe

```text
scheduled_512 = 512 ms

```

Use one inference worker.

If the worker is free:

```text
start_512 = 512 ms

```

Otherwise:

```text
start_512 = time when the first inference finishes

```

Then:

```text
arrival_512 =
start_512 + inference_latency_512_ms

```

The result can act only if:

```text
span_duration_ms > arrival_512
and
arrival_512 < 800 ms

```

## Hard timeout

```text
800 ms always wins over a model result arriving at or after 800 ms

```

## Authoritative decision time

```text
first valid accepted result arrival
or
800 ms

```

Nominal 224/512/800 ms timing may be reported only as a secondary diagnostic.

---

# 6. High-repetition nested cross-validation

The goal is 1,000 outer held-out policy evaluations across four languages.

For each language, run:

```text
5 outer group folds
×
50 repeated group-split seeds
=
250 outer held-out evaluations

```

Across four languages:

```text
250 × 4 = 1,000 held-out evaluations

```

These are not 1,000 new independent audio samples.

They measure sensitivity to conversation grouping and train/test splits.

## Outer split

The outer test fold is used only for final policy evaluation.

## Inner split

Inside each outer training partition, use:

```text
5-fold group cross-validation

```

to select:

- P1 threshold.
- P2 shared threshold.
- Requested operating point.

The outer test fold must not influence threshold selection, tie-breaking, or candidate availability.

---

# 7. Grouping and leakage prevention

Use this grouping priority:

```text
conversation_id
recording/session ID
full user-turn ID

```

Never split pause spans from the same conversation or turn across training and test partitions.

## Fold validity

Each outer test fold should contain at least:

```text
20 true EOT spans
20 eligible hold spans

```

An eligible hold is a hold pause long enough for at least one tested decision to occur.

If a seed creates an invalid fold:

1. Reject that split.
2. Generate another deterministic seed.
3. Record the rejected seed and reason.

Do not silently evaluate folds with only one class.

If the dataset lacks enough conversation groups for valid five-fold evaluation, reduce to four or three folds and preserve approximately 250 valid outer evaluations per language.

Document any reduction.

---

# 8. Threshold candidates

Do not rely on a uniform 0.01 threshold grid.

Use actual observed training-fold scores as candidates.

## P1 candidates

```text
unique score_224 values from the inner-training data

```

## P2 candidates

```text
union of unique score_224 and score_512 values
from the inner-training data

```

Include boundaries:

```text
0.0
1.0

```

Use deterministic threshold ordering and deterministic tie-breaking.

The held-out fold must not contribute threshold candidates.

---

# 9. Operating-point selection

Select two candidates per policy and language.

## Low-latency candidate

Choose the shortest runtime-aware mean endpoint latency subject to:

```text
relative false-cutoff reduction versus B0 >= 20%

```

## Stability candidate

Choose the shortest runtime-aware mean endpoint latency subject to:

```text
relative false-cutoff reduction versus B0 >= 35%

```

Both candidates must satisfy:

```text
held-out false-cutoff rate <= B0
runtime-aware mean endpoint <= 600 ms
P50 endpoint <= 600 ms
EOT timeout rate <= 25%

```

If no candidate satisfies a target in the inner training evaluation:

```text
candidate unavailable

```

Do not replace an unavailable target with a baseline-matched threshold in the primary results.

A baseline-matched point may be reported separately as a diagnostic.

---

# 10. Inner-CV threshold selection

For each candidate threshold:

1. Evaluate it across the inner validation folds.
2. Aggregate inner held-out predictions.
3. Calculate the operating constraints.
4. Select the valid candidate with the lowest mean endpoint latency.

Do not select a threshold using its in-sample training performance.

Tie-breaking order:

```text
1. lower runtime-aware mean endpoint
2. lower false-cutoff rate
3. lower EOT timeout rate
4. higher threshold

```

Record all ties and the final tie-break reason.

---

# 11. Required metrics

Report per language, policy, operating target, outer repeat, and outer fold.

## Span metrics

- Total EOT spans.
- Total hold spans.
- Eligible hold spans.
- False cutoffs.
- False-cutoff rate over all hold spans.
- False-cutoff rate over eligible hold spans.
- Relative false-cutoff reduction versus B0.
- Mean endpoint latency.
- P50.
- P90.
- P95.
- P99.
- 224 ms acceptance rate.
- 512 ms acceptance rate.
- EOT timeout rate.
- Unresolved span rate.
- Stale-result rate.
- Probe-overlap rate.

## Turn metrics

- Turns with at least one false split.
- Percentage of turns with at least one false split.
- False splits per 100 turns.
- Mean false splits per affected turn.

## Threshold stability

- Median selected threshold.
- P10.
- P25.
- P75.
- P90.
- Minimum.
- Maximum.
- IQR.
- Percentage of outer evaluations where the target was available.

---

# 12. Paired P1 versus P2 comparison

P1 and P2 must be compared on the same:

- Outer split.
- Test conversations.
- Operating target.
- Language.

Do not compare independently averaged operating points without pairing.

For every outer evaluation, calculate:

```text
P2 mean endpoint - P1 mean endpoint
P2 false cutoff - P1 false cutoff
P2 EOT timeout - P1 EOT timeout
P2 turn fragmentation - P1 turn fragmentation

```

Negative endpoint and timeout deltas favor P2.

---

# 13. Value gate for the 512 ms probe

Keep P2 for a language only if it satisfies either:

```text
paired mean endpoint improvement >= 20 ms

```

or:

```text
paired EOT-timeout reduction >= 5 percentage points

```

while:

```text
false-cutoff regression <= 0.5 percentage points

```

Additional robustness requirements:

```text
the value condition is met in at least 80% of valid outer evaluations

```

and:

```text
the 95% confidence interval does not cross
the allowed false-cutoff regression boundary

```

If P2 fails this gate, prefer P1.

Do not retain a second inference probe for a 3–12 ms point-estimate improvement that is unstable across splits.

---

# 14. Conditional P3 experiment

Run P3 only for languages where P2 passes the value gate.

P3 uses:

```text
T512 >= T224

```

Use nested inner-CV selection exactly as for P1 and P2.

P3 must beat P2 through either:

```text
paired mean endpoint improvement >= 15 ms
at matched false-cutoff rate

```

or:

```text
false-cutoff improvement >= 1 percentage point
at matched mean endpoint latency

```

The matched condition must pass the explicit 0.5 percentage-point sanity assertion.

If P2 fails, mark P3:

```text
not evaluated: second probe did not pass complexity gate

```

---

# 15. Bootstrap confidence intervals

Use:

```text
10,000 conversation-level bootstrap resamples

```

Do not resample individual spans.

Use a fixed documented seed.

Calculate paired 95% confidence intervals for:

- P1 relative false-cutoff reduction versus B0.
- P2 relative false-cutoff reduction versus B0.
- P2 minus P1 mean endpoint.
- P2 minus P1 false-cutoff rate.
- P2 minus P1 EOT timeout rate.
- P2 minus P1 turn-fragmentation rate.

Bootstrap each outer test result from the conversations in that test partition.

Also produce an aggregate hierarchical bootstrap:

```text
resample outer repeats
then resample conversations inside selected repeats

```

Keep ordinary per-repeat metrics separate from the hierarchical aggregate.

---

# 16. Interpreting the increased run count

Do not report the 1,000 outer evaluations as 1,000 independent datasets.

The effective unique evidence remains the number of real conversations and pause spans.

Repeated nested CV answers:

```text
How sensitive is policy and threshold selection
to the particular conversation split?

```

Conversation bootstrap answers:

```text
How uncertain are the measured policy differences
given the available conversations?

```

Running the same audio through Smart Turn repeatedly would not increase policy accuracy evidence and is outside scope.

---

# 17. Final per-language decisions

Choose one decision for each language.

## `P1_SHADOW`

Use:

```text
224 ms Smart Turn probe
800 ms hard timeout

```

Conditions:

- P1 passes at least the low-latency target.
- P2 does not pass the second-probe value gate.

## `P2_SHADOW`

Use:

```text
224 ms Smart Turn probe
512 ms Smart Turn probe
shared threshold
800 ms hard timeout

```

Conditions:

- P2 passes the second-probe value gate.
- P3 is not materially better.

## `P3_SHADOW`

Use:

```text
224 ms with T224
512 ms with T512
800 ms hard timeout

```

Conditions:

- P2 passes.
- P3 passes its additional complexity gate.

## `BASELINE_ONLY`

Use current fixed 512 ms VAD for that language.

Conditions include:

- No valid P1 target.
- P1 does not improve B0 reliably.
- Latency or EOT timeout constraints fail.
- Threshold behavior is too unstable.

## `NEEDS_MORE_DATA`

Use when:

- Point estimates are promising.
- Confidence intervals remain too wide.
- Threshold availability is low.
- Fold-to-fold results are unstable.

---

# 18. Global decision

Choose one:

```text
PROCEED_TO_LANGUAGE_SPECIFIC_SHADOW
PROCEED_TO_UNIFIED_SHADOW
NEEDS_MORE_POLICY_DATA
STOP_SMART_TURN_INTEGRATION

```

A language-specific shadow result is acceptable.

For example:

```text
ko: P1_SHADOW
ja: P1_SHADOW
en: P1_SHADOW
zh: BASELINE_ONLY

```

Do not force Chinese into Smart Turn merely to keep one global architecture.

The implementation may share the state machine while disabling Smart Turn for selected languages.

---

# 19. Required output files

Produce:

```text
input_validation.json
evaluator_corrections.md

nested_cv_all.csv
nested_cv_ko.csv
nested_cv_ja.csv
nested_cv_en.csv
nested_cv_zh.csv

outer_split_manifest.csv
rejected_split_manifest.csv

threshold_stability.csv
candidate_availability.csv

p1_vs_p2_paired.csv
p3_conditional_results.csv

span_metrics.csv
turn_fragmentation.csv

bootstrap_confidence_intervals.csv
hierarchical_bootstrap.csv

selected_operating_points.json
language_decisions.json
summary.json
report.md

```

Do not regenerate or duplicate the CPU prediction parquet files.

---

# 20. Required tests

## Corrected timeout metrics

1. A 300 ms hold that resumes before 800 ms is not an EOT timeout.
2. An EOT rejected by all probes and ended at 800 ms is an EOT timeout.
3. `unresolved_span_rate` and `eot_timeout_rate` are calculated independently.
4. Selection gates use `eot_timeout_rate`.

## Matched comparison

5. A candidate more than 0.5 percentage points from the target is rejected.
6. No matched candidate returns `unavailable`.
7. A boundary threshold is not silently substituted.

## Nested CV

8. Outer test conversations are never used by inner CV.
9. Threshold candidates come only from inner-training scores.
10. Inner-validation metrics select the threshold.
11. The outer fold is evaluated exactly once per selected candidate.
12. Conversation groups do not cross folds.

## Repeated CV

13. Exactly 50 accepted repeats are produced per language.
14. Invalid folds are rejected and documented.
15. Fixed seeds reproduce the same splits and results.

## Paired policy comparison

16. P1 and P2 differences use identical outer test groups.
17. Missing policy candidates are not treated as zero-valued results.
18. Availability rates are reported.

## Bootstrap

19. Conversations, not spans, are resampled.
20. Exactly 10,000 bootstrap samples are generated.
21. Fixed seeds reproduce confidence intervals.

---

# 21. Non-goals

Do not:

- Rerun Smart Turn inference.
- Rerun CPU latency benchmarks.
- Compare one versus two CPU threads again.
- Change production behavior.
- Add shadow mode in this task.
- Remove speculative translation.
- Use TTS.
- Treat repeated splits as new audio samples.
- Select thresholds from outer test data.
- Force P3 evaluation when P2 fails.
- Force one policy across all four languages.
- Hide candidate unavailability through fallback substitution.

---

# 22. Central questions

The report must answer:

```text
1. After fixing timeout accounting, does P1 reliably beat
   the current 512 ms VAD on held-out conversations?

2. Does the 512 ms second probe provide enough paired,
   statistically stable value to justify its additional complexity?

3. Which languages should proceed to shadow mode?

4. Are the selected thresholds stable across 50 repeated
   conversation-group splits?

```

Do not conclude from point estimates alone.

Use held-out paired results, target availability, and confidence intervals.