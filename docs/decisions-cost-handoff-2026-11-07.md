# Decisions classifier cost and prompt-bloat handoff

- Date: 2026-11-07.
- Audience: the next engineer investigating Puri's D classifier costs.
- Status: research, earlier live measurements, and offline candidate sizing are complete. No product optimization has been implemented.
- This document does not replace owner decisions. It distinguishes `[MEASURED]`, `[CODE]`, `[SPEC]`, `[EXTERNAL]`, `[OFFLINE ESTIMATE]`, and `[INFERENCE]`.
- The research phase made no additional paid model calls. It reused actual API results from the user's earlier, separately requested local measurements.

## 1. Executive findings

1. `[MEASURED]` Each D request used approximately 3.8K input tokens. All eight measured D calls reported zero cache reads, cache writes, and output tokens.
2. `[CODE]` D no longer selects only route and effort. One Decisions request contains ten independent questions covering route, effort, app-context modules, and task relation.
3. `[SPEC]` Decisions caches matching prefixes of shared `input`, not the contents of `questions`. Most lengthy, constant D instructions currently reside in `questions`.
4. `[OFFLINE ESTIMATE]` Keeping all ten question names, types, and choice values, consolidating repeated policies and choice descriptions reduced a consistently counted local payload estimate from 3,209 to 1,955 tokens. This is not a verified implementation, API billing reduction, cache hit, or classification-quality result.
5. `[INFERENCE]` The first candidate to evaluate is question-text deduplication plus a server-owned shared policy prefix, not a model replacement or a merger of independent questions. Decide only after matched-version cost and quality measurements.

## 2. User intent and authorization boundary

The user is concerned about chatbot operating costs. They requested cached/uncached estimates, local measurements, inspection of D input volume, research into inexpensive decision-model usage, and this handoff.

Writing this document does not execute prompt changes, a model replacement, another paid evaluation campaign, deployment, account reauthentication, GitHub publication, issue changes, or a PR. Before implementing or running a new experiment, check the latest user instruction and applicable owner decisions.

Read first:

- [System map](architecture.md#agent-conversation).
- [Current owner decisions](../agent-relay/prompts/agent/decisions.md), especially `Prompt cache structure` and `Follow-up experiments and boundaries`.
- [Agent Relay development and isolation instructions](../agent-relay/README.md).
- [Application control and observation boundaries](cli.md#agent-chat-control).

The decisions file retains historical route/effort-only policy. A subsequent same-day decision authorized module selection. Do not use the historical paragraph to revert D to two questions.

## 3. Current architecture and source map

### 3.1 D request

- Model and endpoint: `gpt-6-luna`, `POST https://api.openai.com/v1/decisions`.
- The normal entry path makes one D request per submitted user message. It does not rerun D for every tool continuation or approval-result notice.
- The existing classifier deadline is one second. Timeouts and incomplete answers use conservative fallback behavior.
- Trusted questions are server-owned. Clients must not gain permission to provide arbitrary questions or trusted instructions.

| File | Responsibility |
| --- | --- |
| [scope.ts](../agent-relay/src/scope.ts) | Combines route, effort, and generated context questions into one array |
| [scope.md](../agent-relay/prompts/agent/scope.md) | Route boundaries, examples, and M effort-selection policy |
| [generate-context.py](../agent-relay/generate-context.py) | Generates questions from the code-owned module registry |
| [context-questions.ts](../agent-relay/src/context-questions.ts) | Generated module and task-relation questions; do not hand-edit only this output |
| [index.ts](../agent-relay/src/index.ts) | Builds upstream requests; the classification branch currently lacks M's fixed developer-message prefix |
| [routing.py](../src/puripuly_heart/app/services/agent_chat/routing.py) | Serializes D input and interprets choices, uncertainty, and refusals |
| [owner.py](../src/puripuly_heart/app/services/agent_chat/owner.py) | Enforces the deadline, applies D decisions, and owns task context and conversation lifetime |
| [modules.py](../src/puripuly_heart/app/services/agent_chat/modules.py) | Owns module definitions and model-context projection |
| [usage.ts](../agent-relay/src/usage.ts) | Computes D/M token costs and records usage in D1 |

D's questions:

| Question | Allowed choices |
| --- | --- |
| `route` | `work`, `social`, `out_of_scope`, `unclear` |
| `effort` | `low`, `high`, `xhigh` |
| `context_audio` | `include`, `omit`, `unclear` |
| `context_recognition` | `include`, `omit`, `unclear` |
| `context_translation` | `include`, `omit`, `unclear` |
| `context_accounts` | `include`, `omit`, `unclear` |
| `context_languages` | `include`, `omit`, `unclear` |
| `context_output` | `include`, `omit`, `unclear` |
| `context_general` | `include`, `omit`, `unclear` |
| `context_relation` | `new`, `continue`, `unclear` |

Variable input consists of `current_message`, up to four recent display messages, `active_task`, `last_puri_action`, and `previous_modules`. Recent messages are capped at 600 characters each; task and last-action summaries are capped at 240 characters each. Current serialization starts with `current_message`, so it has no substantial stable prefix. Do not solve the cost problem by exposing raw app state, private logs, or secrets to D.

### 3.2 Selected app state is different from M's fixed prompt

`[CODE]` `AgentChatOwner` currently defaults to `context_mode="selected"`. For work requests, the new app-state projection contains mandatory `core` plus D-selected modules. Collecting an overview locally is not the same as transmitting that entire overview to the model.

Broader context can still appear through:

- Full-module fallback for missing, incomplete, uncertain, or diagnostic-entry selections.
- Modules previously selected or explicitly read for the same task.
- Approval and key-verification notices carrying their originating task scope plus code-mapped relevant modules.
- Existing conversation and operation-receipt history.

Behavior instructions, the product guide, cost and benchmark references, and tool definitions remain shared fixed work-M context. Social requests receive display dialogue only; out-of-scope requests receive only the current message.

The selected/full comparison flag changes app-state projection. Both modes use the expanded ten-question D. It is not a comparison with the old route/effort-only classifier.

## 4. Observed provider usage

### 4.1 D input volume

`[MEASURED]` These are D usage counts from eight conversations using actual Luna API calls and synthetic app state.

| Request category | Runs | D input tokens per run |
| --- | ---: | ---: |
| CAPTIONS feature explanation | 2 | 3,822 |
| Enable translation | 2 | 3,815 |
| Change UI language and approve | 2 | 3,824 |
| Compare free and paid translation models | 1 | 3,848 |
| Diagnose missing recognition/captions | 1 | 3,834 |

Every D call reported `cached_tokens = 0`, `cache_write_tokens = 0`, and `output_tokens = 0`.

- D input total: 30,604 tokens; mean: 3,825.5 tokens.
- D total at the standard input rate: `$0.0030604`.
- Mean D cost: `$0.00038255`, approximately KRW 0.536 at an illustrative exchange rate of KRW 1,400/USD.
- D accounted for approximately 62% of the cached simple-explanation turn, but approximately 31% of this entire eight-turn mix. Do not generalize the former to every conversation.

### 4.2 Whole-turn cost and output variation

Costs include D, M, tool-result continuations, and the response following settings approval. Output and reasoning columns sum M usage; reasoning is a subset of output.

| Case | Repetition | Total API requests | M output | Of which reasoning | Usage-based USD |
| --- | ---: | ---: | ---: | ---: | ---: |
| Feature explanation | 1 | 2 | 39 | 0 | 0.001642250 |
| Feature explanation | 2 | 2 | 53 | 0 | 0.000614020 |
| Enable translation | 1 | 3 | 35 | 0 | 0.000858835 |
| Enable translation | 2 | 3 | 37 | 0 | 0.000859835 |
| Language change and approval | 1 | 4 | 300 | 0 | 0.001189210 |
| Language change and approval | 2 | 4 | 307 | 0 | 0.001193720 |
| Model recommendation | 1 | 2 | 3,256 | 3,103 | 0.002355745 |
| Troubleshooting | 1 | 3 | 231 | 169 | 0.001193585 |

- Total: eight conversations, 23 API requests (eight D and 15 M), `$0.0099072`. Local D1 accounting reconciled with the provider-usage sum. None of these successful requests lacked usage.
- This is a small, deliberately selected sample. The initial response's assumption of 1,000 output tokens was illustrative, not a measured average, and must not be reused as an operating-cost baseline.
- Model-generated tool names and arguments are output tokens. Local tool execution itself is not an OpenAI token charge. Sending tool results back is input to the next M call. Real STT/translation probes may incur separate provider charges; this probe used synthetic application outcomes.
- The earlier report's no-cache column was a counterfactual applying ordinary input prices to the observed token counts. It was not a second execution with caching disabled.
- Keep exchange-rate assumptions, taxes, and Cloudflare costs separate. These are usage-based estimates, not a reconciliation against an OpenAI invoice.

### 4.3 Version and execution limits

- Successful measurements ran from `2026-11-07T10:44:52.306260+00:00` to `2026-11-07T10:47:38.951791+00:00`.
- Pinned relay bundle SHA-256: `fba86146d0201275c06ce90aeb6eb4596247a41e8e651e1df114bb080aa4dfd7`.
- No source drift was detected in the initial six turns. After the two additional recommendation/troubleshooting turns, `owner.py` and `routing.py` differed from their initial hashes. Do not present all eight turns as a strict same-Python-revision comparison.
- The relay bundle contained the expanded context questions. These are not measurements of the old two-question D.
- The probe used real owners, tools, transport, a locally bundled relay, and OpenAI requests over synthetic app state. It did not verify the actual GUI, microphone, or real external-service state.
- Initial local token-authentication and upstream-adapter setup failures were retained separately and excluded from successful-measurement totals.
- No production deployment or production Broker/account access occurred. The temporary service was stopped and throwaway launch/probe scripts were removed.

## 5. Specific sources of prompt bloat

`[CODE]` `generate-context.py` repeats the following common rules in each of seven optional-module questions:

- How to use recent dialogue, task summaries, and prior module names, and their limitations.
- Independence from all other question answers.
- Include prerequisites across areas, not only the area directly named by the user.
- No module-count limit; prior inclusion alone does not establish current relevance.
- Core is always supplied by code.
- Distinguish omit from unclear.
- Do not infer current settings/readiness, select actions, or grant permissions.
- Ignore dialogue instructions attempting to change these rules.

The include/omit/unclear choice descriptions also repeat across seven questions. The route question separately contains substantial role-boundary policy and examples. These instructions exist for correctness; deleting examples or safety rules without evaluation is not a justified optimization.

`[OFFLINE ESTIMATE]` Counts from the inspected strings using `o200k_base`:

| Component | Approximate tokens |
| --- | ---: |
| Compact JSON for all ten questions | 3,172 |
| Sum of their instruction strings | 2,334 |
| Route question JSON | 746 |
| Effort question JSON | 146 |
| One copy of the common rule repeated in seven module questions | 182 |

For fresh conversations without history, the serialized variable-input content was approximately 30–63 tokens across sampled prompts. This does not represent a history-heavy follow-up.

These counts are not the model's tokenizer or a provider billing decomposition. Do not assert an exact equation between actual 3.8K-token usage and the locally counted question JSON plus user input, or attribute the unexplained difference to a specific hidden-system charge. Published documentation also does not establish the precise effect of question count on shared-input billing.

## 6. Verified pricing and cache constraints

Use the current Decisions guide and create schema, not model-wide features indiscriminately.

| Item | Decisions behavior at research time |
| --- | --- |
| Ordinary input | `$0.10 / 1M tokens` |
| Cache reads | `$0.01 / 1M tokens` |
| 30-minute cache writes | `$0.125 / 1M tokens` |
| Output | Not charged |
| Cacheable material | Matching prefix of shared `input` |
| Material outside reusable prefix | `questions`, including question instructions and images |
| Minimum length | 1,024 visible tokens in the cacheable shared input; question tokens do not count |
| Requestable TTL | Decisions create schema documents `30m` only |

Important details:

- `cached_tokens` and `cache_write_tokens` are subsets of `input_tokens`. Do not add them to input again. Write pricing replaces ordinary input pricing for those tokens; it is not an additional surcharge on top of ordinary input.
- Shared input supports developer/system text and explicit cache breakpoints. The schema does not guarantee that moving decision rubrics there preserves classification behavior.
- Test a boundary after the fixed prefix and before changing user text. Explicit-only mode is a candidate to avoid writing unique user text. Do not claim implicit mode cannot coexist with explicit breakpoints.
- Adding `prompt_cache_key` does not make short input cacheable. Do not introduce unnecessary key partitioning or sharding for GPT-6.
- Do not pad prompts to reach the cache minimum. Account for writes and actual reuse. Consolidating naturally required policy is different from adding unnecessary text.
- Launch articles and generic pricing mention 12-hour writes, but the Decisions create schema documents only a requestable 30-minute TTL. Do not copy Responses-only options into D.
- The official Batch endpoint list does not include `/v1/decisions`. Multiple independent questions in one request are not the asynchronous Batch API discount. Do not assume Batch/Flex halves this classifier's bill.
- Responses-only prompt-cache diagnostics must not be assumed to work on Decisions. Confirm D cache behavior from its actual usage fields.

## 7. External research and evidence strength

No public reproducible billing report was found for our exact shape: short changing input plus ten long fixed questions. The following sources are useful but demonstrate different things.

| Source | Reported result or pattern | Evidence and applicability limits |
| --- | --- | --- |
| [Sierra in OpenAI's launch article](https://openai.com/index/decisions-api/) | Next-action selection in live conversations; customer quote of 90% lower per-call cost and under 280ms p95 versus its previous approach | No published previous-stack details, token bill, question count, or cache-hit methodology. Not evidence of another 90% reduction for an existing Decisions deployment |
| [Ramp in the same article](https://openai.com/index/decisions-api/) | AI-cost attribution; customer quote of 30% lower cost and 4x speed versus ordinary Luna | A transition from general generation to Decisions, not a measurement of compressing our D questions |
| [Official model-router recipe](https://github.com/openai/openai-agents-python/blob/main/examples/decisions/model_router.py) | One short question and short choices select a later generative model | Implementation pattern, no measured cost reduction. Not a reason to merge our ten independent questions |
| [Fixed-question cache discussion](https://community.openai.com/t/decisions-api-is-now-generally-available-with-new-updates/1408934) | Practitioners report poor cache fit for fixed questions with changing input, and uncertainty about moving rubrics before the input | Closely matches our shape, but is not a demonstrated savings or quality result |
| [Independent Decisions/Responses benchmark](https://github.com/SVB808/openai-decisions-api-benchmark) | Same-model support classification: Decisions p95 183.70ms versus Responses p95 2,263.81ms | One environment, 200 synthetic strings derived from 40 correlated scenarios. Explicitly does not compare dollars. No guarantee for Korean multi-question routing |
| [LiteLLM decision-model benchmark](https://docs.litellm.ai/docs/decisions/benchmark) | Alternative models differ in accuracy, calibration, and missed decisions across routing, guardrail, and judge tasks | Mostly synthetic data; not proof that a cheaper model preserves our Korean module-selection behavior |
| [Cost-safe patterns tutorial](https://www.kunalganglani.com/blog/openai-decisions-api-tutorial) | Gate-then-generate, rules-to-AI cascades, grouped questions, exact-match result caching | Example traffic and hit rates are hypothetical. Its application-level result cache is not OpenAI prompt caching |

Pricing comparison sites can identify candidates, not establish equal-quality savings. Do not apply obsolete beta claims of no cache-write charge, a 500-token-per-decision cost example, or uncalibrated confidence thresholds to our actual measurements.

## 8. Offline candidate sizing already performed

`[OFFLINE ESTIMATE]` No product files were changed and no model calls were made. Candidate question arrays were assembled in Python memory and counted with `o200k_base`. Every variant used the same fresh CAPTIONS question without conversation history.

| Variant | Questions | Question JSON | Fixed shared input | Variable input | Total proxy tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| Current | 10 | 3,172 | 0 | 37 | 3,209 |
| A: deduplicate common module rules | 10 | 1,947 | 187 | 37 | 2,171 |
| B: consolidate policies and choice descriptions | 10 | 388 | 1,530 | 37 | 1,955 |

How the prototypes were constructed:

- A: Verify that the seven module questions have the same common rule suffix, then place that suffix once in shared input. Keep module-specific descriptions/questions and existing choice descriptions, with a short reference to the shared rules.
- B: Put route, effort, and task-relation policies and choice descriptions, the common module policy and choice meanings, and module definitions once in server-owned shared input. Replace per-question instructions with short policy references; retain the same choice values without repeating their descriptions. Omitted descriptions are schema-permitted, but correct interpretation remains unmeasured.
- Offline assertions checked equal question names, types, and ordered choice values, and retention of original policy sentences and choice descriptions in B's shared policy. The original per-question instruction wording itself was changed.
- Total proxy-token reductions were 32.3% for A and 39.1% for B. These are string-size reductions, not measured invoice savings.
- A's shared prefix remains below the cache minimum in the local estimate. Deduplication is still a separate optimization hypothesis.
- B creates a naturally substantial fixed prefix without padding, making it a candidate for a cache experiment. Confirm the actual model's visible-token length and actual cache reads/writes through the API.

The prototype does not establish:

- Equivalent decisions after policy relocation.
- Equality between billed tokens and local string counts.
- Guaranteed cache hits or a particular cost-reduction percentage.
- Preserved refusal or prompt-injection behavior merely because the policy words remain present.

## 9. Preliminary judgments, not approved implementation decisions

### 9.1 First experiment to consider

`[INFERENCE]` Compare A and B with the existing D before replacing models. This directly targets the large fixed question payload while retaining independent questions, choice values, and code-owned authority boundaries.

Likely implementation touchpoints are `generate-context.py`, the trusted policy assembly, and `index.ts`'s upstream classification body. Avoid editing generated output alone, or copying policies into shared input while leaving the same long policies in every question. Do not elevate client-provided user input into developer authority.

### 9.2 Changes not to make first

- Merge seven module decisions into a single choice: multiple modules can be needed simultaneously, so this breaks the contract.
- Remove unclear, refusal handling, task relation, or full-context fallback to make the prompt shorter.
- Split route and remaining decisions into two D calls: work-heavy traffic can acquire extra latency and requests, and this changes the current one-call/one-second contract.
- Force universally shorter M answers: simple actions already used tens of output tokens; recommendation reasoning is a different cost source.
- Treat cache keys, a 12-hour TTL, or Batch/Flex pricing as an automatic solution.
- Broadly bypass free-form user messages using greeting/keyword rules without context: unresolved work, references, and injected instructions can change their meaning.
- Infer an exact-match result-cache hit rate from the repeated measurement prompts: deliberate duplicates are not representative traffic.
- Immediately switch vendors or move to Responses: include output charges, schema input, latency, and Korean classification quality.

A deterministic bypass may be worth a separate proposal for program events or explicit internal commands whose complete decisions are already known. Removing D from every tool/approval continuation is not a new saving: the current implementation already avoids those calls.

## 10. Contracts to preserve

Re-read the latest owner decisions and implementation before making changes. Preserve:

- One D call per submitted message, the existing one-second deadline, budgets, authentication, cancellation, and generation fencing.
- Ten independent questions, the code-owned module registry, and mandatory core context.
- Valid route/effort decisions when only context selection fails.
- Full-context fallback for named context-question refusals; separate out-of-scope/low precedence for route, effort, unknown, and unnamed refusals.
- Conservative full context for incomplete/uncertain selection, unclear routing, and diagnostic entry.
- Task continuation/change, reset, expiry, and supporting-history-eviction rules for context scope.
- Originating task scope for approval queues and key-verification notices, without another D call.
- Social/out-of-scope context and tool isolation.
- App-owned approval, tool-execution, and secret-handling authority. D confidence never grants permissions.
- Separation of D compression from changes to M's guide, references, or tool definitions.

## 11. Recommended validation sequence

1. **Pin versions and preserve evidence.**
   - Pin the Python code and emitted relay together. Do not equate the handoff snapshot, the earlier measured bundle, and the deployed service.
   - Prepare identical labeled inputs for current/A/B. Inspect actual upstream question names, choices, policy placement, and prefix boundaries.
   - Use new report paths. Retain failed and blocked attempts rather than replacing them with reruns until green.
2. **Probe billing and caching at minimum scope.**
   - Vary question count and shared-input length in controlled small probes if the billing decomposition matters. Do not infer undocumented billing behavior from local string sums.
   - Separate cold writes from warm reads and vary real synthetic user text after the fixed prefix. Local prefix equality is not a provider cache hit.
   - Record `input_tokens`, `cached_tokens`, `cache_write_tokens`, `output_tokens`, latency, question-array version, and policy version. Use synthetic content only.
3. **Evaluate D decisions.**
   - Cover Korean and existing supported languages, first turns, follow-ups/references, new tasks, continuing tasks, uncertainty, pending-approval detours, social/out-of-scope, and prompt injection.
   - Compare route, effort, each module's include/omit/unclear, task relation, and refusal precedence separately.
   - Count harmful module omissions, unnecessary full-context fallbacks, unnecessary high/xhigh, and timeouts, not only overall agreement.
   - Observe the actual path with its deadline. A candidate that only works after extending the deadline is not an improvement under the same contract.
4. **Verify complete D+M behavior and cost.**
   - Check whether omitted modules cause extra M reads, inappropriate recommendations, or costly recovery.
   - Include M requests, reasoning, tool continuations, approval outcomes, cancellation, and failure behavior, not just D cost.
   - Do not substitute a cheap D result for app-control correctness or actual GUI verification. Use current capabilities/CLI guidance and isolated synthetic state/local resources for relevant actual-app checks.
5. **Make the decision and report limits.**
   - Present actual costs, quality, and latency together. Policy relocation without cache/quality evidence is not a completed optimization.
   - A permanent implementation must update the generator, callers, relevant contract checks, and permitted documentation together. Do not edit protected README, architecture, privacy, or contributing documents without a separate request; report any resulting drift.

## 12. Corrections to earlier conversational assumptions

- Saying the current implementation always transmits every app-state module was inaccurate. Default behavior is D-selected modules; M's fixed knowledge, existing history, and fallback are separate concerns.
- The initial 1,000-output-token estimate was an illustrative assumption. Actual simple actions and recommendation reasoning differed substantially.
- The local 3,172-token questions JSON count is not an exact attribution of provider input usage. Tokenization, rendering, and billing decomposition can differ.
- Smaller local payload size does not establish the same percentage reduction in actual billed cost.
- Schema support for a shared developer-message rubric is not an official guarantee of equivalent classification accuracy.
- Assuming constant D questions receive automatic prompt-cache discounts is incompatible with the documented `questions` boundary.
- This sample does not establish production-average cost, user repetition rates, or cross-user cache hits at scale. Earlier estimates of conversations per $200 were arithmetic repetitions of this small sample mix, not a production forecast.

## 13. Additional evaluation records noticed while writing

These are records in the latest `decisions.md`, produced by separate work. This session did not rerun them or independently audit all raw results. Do not merge them with the eight-turn cost probe above.

- The `puri-context-selection-20261107-c48f` comparison used emitted bundle `f22b39eb752e4ff132776f66bf34f7deb2970cd7cd447c3076abf213d33dc5b7`, eight cases, and three fresh runs per mode.
- Selected recorded 22 passes and two failures; full recorded 23 passes and one original judge network failure. Both met the existing two-of-three criterion for every case.
- M input totals were 3,279,658 tokens selected versus 4,008,883 full. Conversations and call counts differed, so this is descriptive accounting, not causal savings or quality equivalence.
- A separate visible-input D probe recorded 28 inputs, median latency 0.336 seconds, four uncertain selections requiring fallback, and one omission of a preregistered recognition-module label. Replaying the original result through M yielded appropriate microphone diagnosis/approval using audio context, but that single-turn probe did not exercise approval or subsequent capture.
- The dialogue comparison also recorded two D timeouts per mode. All standalone D probes meeting one second does not prove a universal end-to-end deadline guarantee.

These records reinforce the need to measure module omission and downstream recovery, not just smaller D requests.

## 14. Evidence locations and resumption

This document includes the essential findings and limitations so it can be used without the original chat session. Detailed JSON is supplemental evidence.

| Artifact | Location and purpose |
| --- | --- |
| Eight live turns and D1 reconciliation | Current session: `local://puri-chat-cost-20261107.json` |
| Research sources and offline variants | Current session: `local://decisions-cost-research-20261107.json` |
| Session-local storage | `~/.omp/agent/sessions/-Documents-dev-puripuly_heart/2026-11-07T10-30-56-345Z_01a1b58f-b459-73d8-a569-295563e3439b/local/` |
| Preserved WSL measurement workspace | Ubuntu account: `~/puri-cost-20261107-6c48c8d1/`, containing `results.json`, `results-initial-six.json`, `results-setup-*.json`, `measured-worker.mjs`, and the relay source snapshot |

`local://` is session-scoped. The same URI in another session may not resolve to the same artifact. When raw evidence is needed, preserve it from the existing storage location into a separate evidence directory. Do not import keys, private logs, or live-user conversations.

Core source SHA-256 values captured for this handoff:

| File | SHA-256 |
| --- | --- |
| `agent-relay/generate-context.py` | `8bd72eb34cc4e0a41eccbb976678487d890292f63ad36fd0ed4f6e2818f9510d` |
| `agent-relay/src/context-questions.ts` | `b37de11c0da3ff187c5978ccdf35e01dfafa2ccf2ed833063e4ed156b6607f1a` |
| `agent-relay/src/scope.ts` | `faa03b218082cb6a60ef0edcd9cc3b6f6bc9f632d449e6fadcb30426d8f5e0ad` |
| `agent-relay/src/index.ts` | `a93a1d66fd5883d99f89fd1da8d5125bd582924a47d51421c6d87432a85b39c8` |
| `agent-relay/prompts/agent/scope.md` | `3760c078eb79b56382df65dfda2e41fabbdd98e00a8dbf57450a15bc6db93274` |
| `agent-relay/prompts/agent/decisions.md` | `0d014787807b6111d080f11e3b6faf7919bf212ce571137ec649e3dffc328496` |
| `src/puripuly_heart/app/services/agent_chat/routing.py` | `c4d808e1e40dadb987ecb52e2e1ed0d06a840150516de359908841d423b9a600` |
| `src/puripuly_heart/app/services/agent_chat/owner.py` | `daccd5bb3ba2c624609f8a5c603fc4cf5445b023663b27f2cb20afca596f0c2f` |

Initial Python hashes for the eight-turn measurement are recorded separately in the detailed measurement JSON. Do not claim these handoff hashes identify that earlier run. If sources change before resumption, treat unrelated edits as existing work to accommodate, not changes to revert.

## 15. Primary external references

- [Decisions guide and cache boundaries](https://developers.openai.com/api/docs/guides/decisions#reuse-shared-input-with-prompt-caching).
- [Decisions create schema](https://developers.openai.com/api/reference/resources/decisions/methods/create).
- [GPT-6 Luna pricing](https://developers.openai.com/api/docs/models/gpt-6-luna).
- [General prompt-caching guide](https://developers.openai.com/api/docs/guides/prompt-caching).
- [Batch endpoint eligibility](https://developers.openai.com/api/docs/guides/batch).
- [OpenAI launch article and customer quotes](https://openai.com/index/decisions-api/).
- [Official Decisions recipes](https://github.com/openai/openai-agents-python/tree/main/examples/decisions).
- [Official model-router source](https://github.com/openai/openai-agents-python/blob/main/examples/decisions/model_router.py).
- [Practitioner discussion of fixed-question caching](https://community.openai.com/t/decisions-api-is-now-generally-available-with-new-updates/1408934).
- [Independent Decisions/Responses benchmark](https://github.com/SVB808/openai-decisions-api-benchmark).
- [LiteLLM decision-model benchmark](https://docs.litellm.ai/docs/decisions/benchmark).
- [Cost-safe patterns tutorial](https://www.kunalganglani.com/blog/openai-decisions-api-tutorial).
