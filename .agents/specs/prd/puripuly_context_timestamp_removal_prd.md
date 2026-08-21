# PRD: Remove LLM-Visible Context Timestamps

## Document Status

- **Status:** Proposed
- **Product:** PuriPuly Heart
- **Area:** Translation Context / Prompt Assembly / Local LLM Prefill Optimization
- **Primary Goal:** Improve prompt-cache stability and reduce repeated prefill work
- **Scope:** All translation-provider paths that consume formatted conversation context
- **Out of Scope:** Context expiration policy, context ordering semantics, CPU thread tuning, Flash Attention, KV-cache quantization

---

## 1. Summary

PuriPuly currently stores a timestamp for each translation-context entry and serializes that timestamp into the LLM-visible prompt as a relative age such as:

```text
- [self, 5s ago] "hello"
- [peer, 3s ago] "hi"
```

The relative age changes every time a new translation request is created, even when the underlying context entry has not changed.

This makes otherwise identical context prefixes produce different token sequences over time, reducing the effectiveness of prompt/KV-prefix reuse and increasing repeated CPU prefill work.

This change will:

1. Keep timestamps internally for context expiration and chronological ordering.
2. Remove timestamps entirely from LLM-visible context serialization.
3. Preserve speaker identity and chronological ordering.
4. Update prompts, examples, and tests so all LLM-facing context uses a timestamp-free format.

Target format:

```text
<context>
- [self] "hello"
- [peer] "hi"
</context>
```

The LLM will be told that entries are ordered chronologically from older to newer, but it will not receive explicit absolute or relative time values.

---

## 2. Problem Statement

### 2.1 Current Behavior

A context entry contains internal metadata including:

```text
text
channel
source_language
target_language
timestamp
```

The timestamp is currently used for two different responsibilities:

1. **Runtime context management**
   - Determine whether an entry is still inside the configured time window.
   - Sort self/peer entries chronologically for integrated context.

2. **Prompt presentation**
   - Convert the timestamp to relative age.
   - Serialize it as `5s ago`, `12s ago`, etc.

These responsibilities should be separated.

### 2.2 Prefill Impact

Assume the stored context itself has not changed:

```text
hello
```

At request A:

```text
- [self, 5s ago] "hello"
```

At request B:

```text
- [self, 12s ago] "hello"
```

The semantic context is identical, but the token sequence is different.

As a result, the beginning of the dynamic `<context>` region becomes unstable purely because time has passed.

For CPU inference, where prefill is currently a significant latency bottleneck, this is undesirable.

### 2.3 Desired Behavior

Time remains part of the runtime model:

```text
timestamp -> expiration
timestamp -> ordering
```

but must not cross the LLM serialization boundary:

```text
timestamp
   |
   +--> TTL filtering
   +--> chronological ordering
   +--> internal diagnostics if required
   |
   X
   |
LLM prompt
```

The LLM-visible representation should depend only on stable semantic information:

```text
channel
text
```

---

## 3. Goals

### G1. Preserve existing context expiration behavior

The configured time windows must continue to work exactly as before.

Examples:

```text
context_time_window_s = 30
integrated_context_time_window_s = 40
```

Entries older than the applicable limit must still be excluded.

### G2. Preserve chronological ordering

For integrated context, self and peer entries must remain ordered using their internal timestamps before serialization.

The timestamp is removed from presentation, not from ordering logic.

### G3. Make unchanged context serialization time-invariant

Given the same context entries in the same order, the formatted LLM context must be byte-for-byte identical regardless of the current clock time.

### G4. Apply the behavior consistently to all LLM providers

Timestamp removal must happen before provider-specific code.

The following conceptual path should apply:

```text
ChannelRuntime
    |
    v
ContextResolver
    |  filtering + ordering
    v
LLM context serialization
    |  timestamp removed here
    v
TranslationBackendRequest.context
    |
    +--> Local OpenAI
    +--> Gemini
    +--> OpenRouter
    +--> other providers
```

### G5. Keep speaker information

The LLM must still be able to distinguish local-user and peer history.

Initial target syntax:

```text
- [self] "..."
- [peer] "..."
```

---

## 4. Non-Goals

This change will **not**:

- Remove `ContextEntry.timestamp`.
- Change context time-window settings.
- Change `context_max_entries`.
- Change integrated-context entry limits.
- Change context selection based on language pair.
- Change context ordering rules.
- Introduce `S:` / `P:` shorthand yet.
- Introduce context token budgets.
- Introduce append-only context epochs.
- Change the `<context>` / `<input>` ordering.
- Change prompt-cache implementation inside llama.cpp.
- Modify CPU inference parameters.
- Modify MTP behavior.
- Modify Flash Attention configuration.
- Modify KV-cache precision.

Those may be evaluated separately after this change.

---

## 5. User / Product Rationale

PuriPuly is a live translation application where local LLM inference may run concurrently with VRChat.

The workload is latency-sensitive and typically has:

```text
large stable system prefix
+
small dynamic conversation context
+
current utterance
+
short translation output
```

Once the stable system prefix is cached, the dynamic context region becomes a major part of remaining prefill work.

Removing unnecessarily changing timestamps makes that region more stable without reducing the actual amount of usable conversation history.

This is preferred over immediately reducing context length because it has low expected translation-quality risk.

---

## 6. Current Architecture

### 6.1 Internal Context Entry

`ContextEntry` currently stores:

```python
text: str
source_language: str
target_language: str
timestamp: float
channel: ChannelId
```

The timestamp is necessary runtime metadata and remains part of this structure.

Relevant file:

```text
src/puripuly_heart/core/orchestrator/channel_runtime.py
```

### 6.2 Context Filtering

`ChannelRuntime.get_valid_context()` uses:

```python
(now - entry.timestamp) < time_window_s
```

along with:

- source-language matching
- target-language matching
- maximum-entry selection
- minimum text length

This behavior must remain unchanged.

### 6.3 Integrated Context Ordering

Integrated context combines entries from self and peer channels and sorts them by:

```python
entry.timestamp
```

This ordering must remain unchanged.

Relevant file:

```text
src/puripuly_heart/core/orchestrator/context.py
```

### 6.4 Current Serialization

Current conceptual output:

```text
- [self, 12s ago] "hello"
- [peer, 7s ago] "hey"
```

The relative age is currently generated during context formatting.

This is the primary implementation point to change.

---

## 7. Proposed Design

### 7.1 Architectural Principle

Treat timestamps as **internal-only context metadata**.

The serialization boundary is the point at which `ContextEntry` becomes the string that is eventually passed to the LLM.

Before boundary:

```text
ContextEntry
├── text
├── channel
├── source_language
├── target_language
└── timestamp
```

After boundary:

```text
- [self] "..."
- [peer] "..."
```

### 7.2 Target Serialization

Local context:

```text
<context>
- [self] "오늘 사람 많네요."
- [self] "그러게요."
</context>
```

Integrated context:

```text
<context>
- [self] "오늘 사람 많네요."
- [peer] "평소보다 많은 것 같아요."
- [self] "그러게요."
</context>
```

No timestamp-derived value may appear in the generated context string.

### 7.3 Ordering Contract

The prompt should explicitly state:

```text
Context entries are ordered chronologically from older to newer.
```

This preserves the temporal signal that is useful to the model while avoiding changing numeric metadata.

### 7.4 Speaker Contract

Retain:

```text
[self]
[peer]
```

Definitions:

```text
[self] = earlier utterance from the local user
[peer] = earlier utterance from the peer audio channel
```

No shorthand conversion is part of this change.

---

## 8. Implementation Requirements

### R1. Keep `ContextEntry.timestamp`

Do not remove or make the field optional.

It remains required for:

- expiration
- chronological sorting
- possible internal diagnostics

### R2. Keep all TTL filtering unchanged

The following behavior must remain equivalent before and after the change:

```text
entry newer than time limit -> included
entry older than time limit -> excluded
```

### R3. Keep integrated sorting unchanged

Integrated history must still sort by internal timestamp before formatting.

### R4. Remove relative-age formatting

The LLM formatter should no longer compute:

```text
_relative_age(entry.timestamp)
```

for prompt generation.

If `_relative_age()` has no remaining callers after the change, remove it.

### R5. Use one shared serialization path

Local and integrated contexts should continue to use the same entry formatter.

Avoid adding provider-specific timestamp stripping.

Bad design:

```text
ContextResolver
    |
    v
timestamp-bearing string
    |
    +--> Local OpenAI strips timestamp
    +--> Gemini keeps timestamp
```

Required design:

```text
ContextResolver
    |
    v
timestamp-free string
    |
    +--> every provider
```

### R6. Preserve `<context>` before `<input>`

Current user-message ordering should remain:

```text
<context>
...
</context>

<input>
...
</input>
```

This change must not reorder prompt sections.

---

## 9. Prompt Changes

The shared translation prompt currently references timestamps as context metadata.

Any instruction equivalent to:

```text
Treat timestamps and speaker hints as metadata for tracking conversation flow.
```

should be replaced.

Recommended wording:

```text
* `<context>` is a multilingual history of prior utterances.
* Context entries are ordered chronologically from older to newer.
* `[self]` means the local user's earlier utterance.
* `[peer]` means the other speaker from the peer audio channel; the channel may occasionally include more than one person.
```

Do not tell the model that timestamps exist.

Relevant file:

```text
prompts/translation_prompt.md
```

---

## 10. Example-Prompt Changes

Language-pair examples currently use timestamp-bearing context syntax.

Example before:

```text
<context>
[self, 5s ago] 아까 그분 목소리 진짜 좋았는데.
</context>
```

Example after:

```text
<context>
[self] 아까 그분 목소리 진짜 좋았는데.
</context>
```

Prefer matching runtime syntax exactly.

If runtime serialization includes list markers and quotes:

```text
- [self] "..."
```

examples should use the same representation:

```text
<context>
- [self] "아까 그분 목소리 진짜 좋았는데."
</context>
```

All language-pair examples should be migrated consistently.

Relevant directory:

```text
prompts/prompt-examples/language-pair/
```

---

## 11. Diagnostics Policy

### 11.1 LLM-visible diagnostics

Any diagnostics that capture the already-formatted context string will naturally become timestamp-free.

That is acceptable and desirable.

### 11.2 Internal timing diagnostics

If debugging later requires actual entry age or timestamp, expose that through structured internal diagnostic metadata.

Do not reintroduce timestamps into the LLM context string for debugging convenience.

Preferred separation:

```text
LLM-facing:
  channel
  text

Internal-only:
  timestamp
  age
  expiration decision
```

---

## 12. Testing Strategy

### 12.1 Existing TTL Tests

Keep tests verifying:

- expired entries are excluded
- recent entries are included
- configured time windows remain effective

Expected result: no behavioral changes.

### 12.2 Existing Max-Entry Tests

Keep tests verifying:

```text
max_entries=N
```

selects the correct recent entries.

Expected result: no behavioral changes.

### 12.3 Language-Pair Filtering Tests

Keep tests verifying entries from unrelated source/target language pairs are excluded.

Expected result: no behavioral changes.

### 12.4 Local Formatting Tests

Update expected output.

Before:

```text
- [self, 12s ago] "안녕"
```

After:

```text
- [self] "안녕"
```

### 12.5 Integrated Formatting Tests

Verify:

- self/peer markers are preserved
- timestamp text is absent
- chronological order remains correct

Example:

```text
timestamp=100 self: A
timestamp=105 peer: B
```

must serialize as:

```text
- [self] "A"
- [peer] "B"
```

### 12.6 Provider Request Test

Verify the final `context` passed to the translation provider contains no relative-time syntax.

Example assertion:

```python
assert call["context"] == '- [self] "hello"'
```

### 12.7 Cache-Stability Regression Test

Add an explicit regression test for the primary performance property.

Concept:

```python
entry = ContextEntry(
    text="hello",
    source_language="en",
    target_language="ko",
    timestamp=100.0,
)

clock = 105.0
first = format_context([entry])

clock = 115.0
second = format_context([entry])

assert first == second
```

Required output:

```text
first  = '- [self] "hello"'
second = '- [self] "hello"'
```

Suggested test name:

```text
test_context_serialization_is_stable_as_time_advances
```

This test should prevent future changes from accidentally making prompt serialization clock-dependent again.

### 12.8 No-Timestamp Contract Test

Add a direct contract assertion such as:

```python
assert "ago" not in context
assert "seconds" not in context
```

Prefer structural output equality where possible, with keyword checks as a supplementary guard.

---

## 13. Acceptance Criteria

The feature is complete when all of the following are true.

### Functional

- [ ] Context entries still expire using configured time limits.
- [ ] Integrated context remains chronologically ordered.
- [ ] Source/target language filtering is unchanged.
- [ ] Max-entry behavior is unchanged.
- [ ] Self/peer speaker identity is preserved.

### LLM Serialization

- [ ] No runtime context entry sent to an LLM contains relative age.
- [ ] No runtime context entry sent to an LLM contains an absolute timestamp.
- [ ] Identical context entries serialize identically as wall-clock time advances.
- [ ] Local and integrated context use the same timestamp-free entry syntax.
- [ ] All providers receive the timestamp-free form.

### Prompt Consistency

- [ ] Shared prompt no longer instructs the model to use timestamps.
- [ ] Shared prompt states that context is ordered oldest to newest.
- [ ] Language-pair examples no longer contain timestamp syntax.
- [ ] Example syntax matches runtime syntax.

### Regression

- [ ] Existing context-expiration tests continue to pass.
- [ ] Existing context-selection tests continue to pass.
- [ ] New cache-stability regression test passes.
- [ ] Provider request tests confirm timestamps are absent.

---

## 14. Performance Validation

This change is primarily motivated by prompt-cache stability.

After implementation, compare sequential translation requests where context entries remain unchanged.

### Scenario A: Before

Request 1:

```text
- [self, 4s ago] "A"
```

Request 2:

```text
- [self, 9s ago] "A"
```

Expected:

```text
different token sequence
```

### Scenario B: After

Request 1:

```text
- [self] "A"
```

Request 2:

```text
- [self] "A"
```

Expected:

```text
identical token sequence
```

### Metrics

Measure when available:

- prompt tokens
- cached/reused prompt tokens
- actually evaluated prompt tokens
- prompt-eval latency
- TTFT
- end-to-end request latency

This PRD does not require a specific latency improvement percentage because cache behavior also depends on context-window rotation and llama.cpp runtime behavior.

The required performance property is:

> Passage of time alone must no longer invalidate an otherwise unchanged LLM context prefix.

---

## 15. Risks

### Risk 1: Loss of useful recency signal

The model no longer knows whether two entries were separated by 2 seconds or 20 seconds.

Mitigation:

- Context is already bounded by runtime time windows.
- Entries remain chronologically ordered.
- Translation quality should be validated with representative conversational cases.

### Risk 2: Prompt/examples diverge from runtime syntax

If only runtime formatting is changed, examples may continue teaching the old representation.

Mitigation:

- Update all language-pair examples in the same change.
- Add search-based or test-based checks if useful.

### Risk 3: Future code reintroduces time-derived prompt text

Mitigation:

- Add the explicit time-invariance regression test.
- Document timestamps as internal-only metadata.

---

## 16. Rollout Strategy

### Phase 1: Code and Prompt Update

Modify:

```text
src/puripuly_heart/core/orchestrator/context.py
prompts/translation_prompt.md
prompts/prompt-examples/language-pair/*
relevant context tests
```

### Phase 2: Regression Testing

Verify:

- TTL behavior
- integrated ordering
- provider serialization
- prompt/example consistency

### Phase 3: Latency Benchmark

Compare old and new builds using repeated real-style requests.

Suggested workload:

```text
stable system prefix
+
1-3 unchanged context entries
+
current input
```

Track prompt-eval latency and cached-token reuse.

### Phase 4: Follow-Up Decision

Only after validating this change independently, consider separate optimizations such as:

- compact `S:` / `P:` context syntax
- lower context entry count
- token-budget-based context
- append-only context epochs
- STT partial prefill

Those should remain separate experiments so their quality and latency effects are attributable.

---

## 17. Files Expected to Change

Primary:

```text
src/puripuly_heart/core/orchestrator/context.py
prompts/translation_prompt.md
prompts/prompt-examples/language-pair/*.md
tests/core/test_context_memory.py
```

Potential secondary test files depending on exact assertions:

```text
tests/providers/test_llm_user_messages.py
tests/core/test_orchestrator_pipeline.py
tests/core/test_peer_channel_routing.py
tests/core/test_translation_owner_branch_coverage.py
```

No provider-specific production code should need timestamp-removal logic.

---

## 18. Final Design Principle

The implementation should enforce this separation:

```text
TIME
 |
 +---- runtime eligibility
 |
 +---- runtime ordering
 |
 +---- internal diagnostics
 |
 X
 |
LLM SERIALIZATION
 |
 +---- speaker
 |
 +---- utterance text
```

In one sentence:

> **Time determines whether and where a context entry appears, but time itself is never shown to the LLM.**
