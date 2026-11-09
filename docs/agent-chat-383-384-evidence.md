# Agent chat #383 / #384 implementation evidence

Recorded on 2026-11-09 for [#383](https://github.com/kapitalismho/PuriPuly-heart/issues/383) and [#384](https://github.com/kapitalismho/PuriPuly-heart/issues/384).

Completion is not accepted. The latest paired comparison completed with retained failures and blocked attempts; recommendation explanation/update criteria and independent review remain unmet. Earlier results are not substituted for the latest candidate.

## Version boundaries

- Shared branch: `issue-383`.
- Immediately pre-change baseline: `2c72d30f98878d881bf5367447cf947b30d96d20`, original app and relay protocol 5.
- Structural checkpoint, before intentional policy changes: `e89f294669db52d421def23acc34d97e83daca9e`.
- Policy and common simulator harness: `80428edaa3740d7bd560fa8d8d2d1e7815252beb`.
- First complete live-campaign candidate after review repairs: `32ec8cb9a15fdbfd9427e5f6f3703689fa6e243d`, app protocol 6.
- Recommendation-rationale repair: `1c4ba253eaf48445bbd81a20495dbefd6249a716`; its paired campaign was interrupted by network failures.
- Current code candidate: `5ee65e41c94069d871380b3203d08cab6ff4d117`. Its guidance additionally distinguishes historical connection preferences from usable access and requires later subscription disclosure to change the first recommendation. The final campaign below shows that model adherence is still incomplete.
- No production deployment, branch push, merge, or release was performed. Deploy the additive relay before releasing the new app client.

## Capability and ownership cutover

`execution.py` is the canonical schema/target mapping. The generated v6 contract exposes three stable tools with a scalar `request.action`; it does not admit nested command arrays or expose CLI-only capabilities.

| Existing capability | v6 destination |
| --- | --- |
| `get_app_context` | `read_app: context`; account controls require `execute_app: ui.accounts` |
| `control_app` queries | `read_app` with the existing query action |
| `control_app` direct commands | `execute_app: overlay.set`, `audio.retry`, `models.retry` |
| `control_app` confirmation commands | `propose_app: secrets.delete`, `auth.logout`, `overlay.position.reset` |
| `change_settings` | `propose_app: settings`, typed sparse `changes` |
| `withdraw_confirmation` | `execute_app: confirmation.withdraw` |
| `set_capture`, `set_translation` | `execute_app: capture.set`, `translation.set` |
| `diagnose_microphone`, `retry_osc_discovery` | `execute_app: audio.diagnose`, `osc.discovery.retry` |
| `probe_self_recognition`, `probe_peer_recognition` | `execute_app: recognition.probe`, explicit `self`/`peer` channel |
| `probe_translation` | `execute_app: translation.probe` |
| `get_provider_setup` | `read_app: provider.setup` for metadata; `execute_app: ui.provider_setup` stages the existing controls and returns metadata |
| `open_settings` | `execute_app: ui.settings` |
| `propose_report` | `propose_app: report`, existing user-click upload flow |

`read_app: operation` observes only receipts retained by this conversation. It is not arbitrary CLI operation access or a second job store. ApplicationControl retains submission, ordering, request-ID deduplication, revisions, waiting, and supported cancellation. Conversation generations, settings approval cards, setup controls, and reporting stay in their existing owners.

Independent reads are admitted together; mixed calls execute in supplied order. Every call receives an explicit result. Approval, staged user controls, unresolved accepted work, blocking failures, and cancellation prevent subsequent dependent effects. An independent read failure does not discard other reads. Existing per-response, per-turn tool, and model-request limits remain unchanged.

Sparse settings preserve omitted-versus-explicit values, coupled model/connection and audio-route validation, approval, stale-revision rejection, and withdrawal. Identical schema fragments share `$defs`; the actual model API accepted both the original v5 and generated v6 work schemas. v5 tools and context semantics remain supported by the final relay. The shared recommendation policy intentionally changes after the structural checkpoint.

Model compaction preserves status, scope, failure, effect, cleanup/restoration, and configured-versus-exercised evidence. Stable activity timestamps avoid age-only core churn. Local/archive evidence remains separate. Report offered/displayed/clicked/sending/sent/failed states and late completion notices remain observable.

## Focused verification

- Structural Python checks: 502 passed initially; five legacy simulator-contract expectation failures were corrected and all five passed on the focused recheck. These are 507 distinct covered cases, not a claim that the whole command was rerun.
- Relay contract/respond/context-export tests: 39 passed. TypeScript `tsc --noEmit --project tsconfig.json` passed.
- Recommendation/simulator suite after #384: 275 passed.
- Controlled simulator smoke for all 25 cases plus report-offer diagnostic-boundary regressions: three tests passed. Controlled results are not live-model evaluation passes.
- Final review repair checks: 137 passed across `test_agent_chat_unified.py`, `test_agent_chat_tools.py`, `test_agent_chat_control_execution.py`, and `test_agent_chat_context_updates.py`.
- Ruff checks/formatting ran on the changed Python implementation paths.

Focused coverage includes the rejected three-query example; ordered read/action/read and proposal barriers; independent partial read failure; pending operation recovery without a second submission; generation cancellation; sparse coupled validation and approval transitions; context age/trim rehydration; provider metadata purity; retained probe scope; and late report completion.

Self-review reproduced and repaired two defects before freezing the combined candidate: recovered probe receipts lost recognition scope/configured-versus-exercised fields; and opening settings failed to stop a later mutation in the same batch. Independent review agents could not start: initial provider dispatch failed, and the retained-session retry had no message history to revive. No independent review is claimed.

Fresh reviewers on resume failed before provider dispatch, with zero model requests or tool calls, including a retry after model access recovered. Whole-checkpoint review and the independent terminal audit remain unperformed. Upstream `d03acbf28459817f3525f6f6cc4edf7911917744` adds only `tests/core/test_peer_caption_lifecycle.py` coverage; it was not merged into this frozen comparison branch.

## Actual presentation evidence

An isolated Flet web instance used the real AgentChatOwner, AgentChatView, and secure-entry controller, with synthetic application state and controlled transports. Chromium inspection exercised the actual rendered controls:

- The account action displayed ChatGPT account controls.
- Explicit OpenRouter setup displayed the Create API Key control. Clicking it displayed the password-style secure-entry field; no real credential was entered.
- Clicking the report button produced one upload and a rendered sent ticket. The owner recorded `displayed=true`, `clicked=true`, `status=sent`, and the archive contained the later support-report completion notice.

Screenshots and the synthetic owner-state record preserve the observations. This proves the controlled presentation/lifecycle path, not real provider signup, credential verification, remote support delivery, or real VRChat receipt. The GUI inspection preceded the final receipt/settings-barrier repair; those repairs did not change these widgets.

Retained images: [secure-entry control](evidence/agent-chat-383-384/gui-provider-secure-entry.webp) and [report sent](evidence/agent-chat-383-384/gui-report-sent.webp).

## Matched structural measurements

Compact JSON character counts are not token or billing measurements. Static definitions/prefixes use the baseline and structural-checkpoint emitted exports. The scripted real-owner/context replay was reconfirmed against the frozen baseline and combined code candidate without invoking recommendation models.

The static work number is the compact JSON size of the complete exported `contexts.work` object, including instructions, tools, and options.

| Measurement | Baseline | New structure |
| --- | ---: | ---: |
| Tool entry points | 14 | 3 |
| Emitted tool definitions, characters | 20,827 | 12,127 |
| Total static work prefix including relocated instructions, characters | 43,433 | 35,601 |
| Two-field model/connection arguments, characters | 1,050 | 120 |
| Initial synthetic core, characters | 986 | 1,079 |
| Same core after only relative age changes, characters | 989 | 129 |
| Bounded single-candidate VAD fragment, characters | 421 | 290 |
| Three-query replay M round trips | 5 | 2 |
| Three-query replay rejected calls | 3 | 0 |

The VAD fragment passes through the existing diagnostic projection and uses 16 distinct probability samples, retained extrema, and the threshold. Existing four-candidate/16-sample bounds are not counted as new savings. The initial-core increase is the explicit observation-time cost. The replay demonstrates avoided read retries, not a whole-conversation savings percentage.

## Documentation drift

Protected documents were intentionally not edited:

- `agent-relay/README.md:18` describes `app-context.md` as the current-protocol format; v6 uses `app-context-v6.md`.
- `README.md:231` and `docs/readme/README.ko.md:235`, `README.ja.md:217`, `README.ru.md:218`, `README.zh-CN.md:218` give unconditional Gemma/OpenRouter setup recommendations without the new subscription-first and input-specific precedence.
- `README.md:237`, `docs/readme/README.ja.md:223`, and `README.ru.md:224` recommend starting with Deepgram, unlike the clarified free-ASR Gemini-first guidance.

`docs/cli.md` was updated for the changed interface and simulator evidence fields. No architecture-map mismatch was identified; `docs/architecture.md`, `PRIVACY.md`, and `CONTRIBUTING.md` remain unchanged.

## Combined comparison and retained failures

The [derived evidence index](evidence/agent-chat-383-384/comparison-evidence.json) retains every planned attempt's original status, code outcome, judgment, and timing, plus source-report SHA256 identities. Private local paths and bulky repeated model inputs are omitted. It is not replay/rejudgment input. Full original `puri-agent-simulator-1` reports remain unchanged locally.

### Common conditions

- Same 13 scenarios, three logical runs each, six spoken-message limit, zero replacement retries, and at most one invalid-judgment rejudgment.
- Common simulator/scenario/judge code from `80428edaa` runs over each arm's own production app. Baseline v5 tools, prompt, owners, and behavior remain original. New-policy alignment applies equally to both arms; grounding uses each version's actual trusted input. Later setup success cannot rescue incorrect initial advice.
- D/M: unchanged `gpt-6-luna`, existing routing/effort updates. User: `gpt-6.1-terra` medium. Judge: `gpt-6.1-sol` xhigh.
- Selected context; parallel arms, three concurrent dialogues and two judge calls per arm, identical case order. No supported seeds.
- Isolated loopback workerd, synthetic UUIDv7/Ed25519 installations, ephemeral D1/support R2, no webhook. Fixture registration limit: 200 per 15 minutes for both arms. App work limits are unchanged.
- Provider-managed 30-minute D explicit / M implicit cache; no flushing. One work-prefix health request per arm. Later campaigns may retain prior provider cache; actual reads/writes are recorded.
- Baseline relay entry SHA256: `503711cb32f163afdffcb0b8e6296e6582d75c55b8ac20e3e7c4d1299447623b`.
- First live candidate relay: `f98b7f917573966a61165ad40f33fc65f4623f24b34621827983385d31799059`.
- Current candidate relay: `52678b72bf22722b295583f767ca3da2504f08fa112097f866f357a59bdeb525`.
- Shared judge revision: `1848eafad7adcdf0f9cde7acdf881d24e1e013bde46b73335588d00bf4416307`.

### Campaign history

| Campaign | Baseline attempts | Candidate attempts |
| --- | --- | --- |
| Initial transport failure | 39 blocked | 39 blocked at `80428edaa` |
| First complete live campaign | 23 passed, 15 failed, 1 invalid judgment | 32 passed, 7 failed at `32ec8cb9a` |
| Refined, network-interrupted | 10 passed, 3 failed, 25 blocked, 1 harness error | 11 passed, 3 failed, 25 blocked at `1c4ba253e` |

The first transport campaign used a temporary bridge that forwarded invalid transport headers. It retained 36 failed D/M attempts and three user timeouts per arm; unknown usage is not zero. The bridge was repaired without changing product networking.

The refined campaign's network failures and baseline simulated-browser action without observed control/guidance remain blocked/harness outcomes, not silently retried product failures.

On resume, both app work-route health requests and the user-model probe succeeded, but the required judge timed out after 300 seconds. A separate Node transport check subsequently succeeded, followed by the exact existing simulator judge probe. The fixed model and effort were not changed. The fresh paired campaign at `5ee65e41c` completed all 78 planned attempts, including blocked outcomes; all prior reports remain intact.

### First complete live campaign: outcomes

Successful logical runs out of three; two successes is the existing scenario threshold. These results apply to `32ec8cb9a`, not the latest repaired candidate.

| Scenario | Baseline | Candidate |
| --- | ---: | ---: |
| recognition-microphone | 3 | 3 |
| debug-translation-auth | 3 | 3 |
| unsolvable-internal | 2 | 3 |
| extra-reject-paid | 3 | 3 |
| routing-pending-approval | 1 | 2 |
| recommend-chatgpt | 0 | 3 |
| recommend-deepseek | 2 | 3 |
| recommend-gemma | 0 | 2 |
| policy-subscription-priority | 3 | 3 |
| policy-language-not-input | 0 | 0 |
| policy-asr-english-quality | 3 | 2 |
| policy-asr-chinese-quality | 0 | 2 |
| policy-asr-free | 3 | 3 |

Scenario thresholds: 8/13 baseline versus 12/13 candidate. All 39 attempts per arm passed existing code prohibition checks. Recommendation explanations passed 11/24 versus 18/24, while final recommendation state was accepted in all 24 attempts per arm. Thus conversion/unchanged advice-only state was not substituted for explanation quality.

Observed failures drove scoped policy repairs:

- General complaints about unnatural Korean/Japanese-to-Chinese translation repeatedly selected Luna/API instead of ordinary Gemma/OpenRouter advice. General improvement is now distinguished from an explicit maximum-quality/benchmark-winner request.
- One Gemma answer summarized a proposal without a selection reason. Some Soniox answers stated the quality-first preference without its experiential reason. The policy now requires substantive rationale in the recommendation itself. The uncalibrated judge was inconsistent about accepting “quality-first” alone; original judgments were not changed.
- The interrupted campaign exposed historical managed preferences being treated as current access, and later subscription disclosure being answered as availability rather than a new first recommendation. Both relationships are now explicit.
- One routing attempt chose work for an unrelated translation request while a card was pending, but still declined the request and later withdrew the card. D routing remains unchanged and outside this repair scope; its failed outcome is retained.

### First complete live campaign: actual usage

Known token sums across all 39 attempts per arm:

| Arm / role | Uncached input | Cache read | Cache write | Output | Reasoning subset |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline D | 132,465 | 123,300 | 0 | 0 | 0 |
| Candidate D | 149,541 | 138,096 | 0 | 0 | 0 |
| Baseline M | 2,951 | 3,340,079 | 337,524 | 104,223 | 88,488 |
| Candidate M | 7,632 | 2,618,696 | 276,235 | 102,395 | 90,322 |

D usage coverage is 100/103 baseline and 112/114 candidate dispatches; cancelled dispatch usage is unknown. All 238 baseline and 203 candidate M calls have usage. Input components partition input; reasoning is already included in output. D/M currency cost is unavailable.

Simulated-user/judge overhead is separate:

| Arm / role | Uncached input | Cache read | Cache write | Output | Reasoning subset |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline user | 618 | 52,363 | 337,480 | 12,296 | 4,176 |
| Candidate user | 651 | 53,787 | 350,947 | 13,117 | 4,441 |
| Baseline judge | 120 | 0 | 3,723,256 | 106,223 | 90,407 |
| Candidate judge | 117 | 0 | 3,273,381 | 89,353 | 75,401 |

Baseline overhead: 206 user/40 judge calls, including a rejudgment. Candidate overhead: 217 user/39 judge calls. Recorded standard-price overhead estimates: $11.56428715 and $9.58482560, not invoices or assistant cost.

### Matched successful-run efficiency and latency

Only the 22 identical scenario/run IDs passing in both arms enter this summary. The other 17 pairs, including policy mismatches, the invalid judgment, and other failures, remain in the index but are excluded from matched-success efficiency. This measures combined structure and policy, not either change's isolated contribution.

| Measurement, 22 paired runs | Baseline | Candidate |
| --- | ---: | ---: |
| M round trips | 143 | 113 |
| M input tokens, including cache | 2,250,483 | 1,706,487 |
| M output tokens, including reasoning | 60,302 | 53,603 |
| Tool calls | 80 | 55 |
| Rejected calls | 5 | 0 |
| Identical name/argument repeats | 8 | 5 |
| Median first displayed response, seconds | 16.49 | 8.91 |
| Median accepted-dialogue duration, seconds | 24.48 | 20.57 |

Across all 39 attempts, rejections were 10 versus zero, all baseline rejections being `invalid_tool_batch`; identical repeats were 14 versus nine. Repeated reads/restaging are not automatically unnecessary. Synthetic-fixture reads also returned `query_unavailable` 17 versus 10 times; these are not invalid calls.

First response runs from first D/M dispatch to first nonempty displayed assistant message, excluding greetings. This is a display-time proxy, not a semantically timed first useful answer. Accepted-dialogue duration ends before judging and includes user/control time, not the precise first successful-effect timestamp. Exact usefulness/effect-completion latency remains unmeasured.

### Current-candidate live policy observations

A deterministic-user probe exercised the real ScenarioApp, AgentChatOwner, auth/broker, and D/M relay at `5ee65e41c`. Seven conversations/ten turns completed with unchanged settings, no confirmation cards, and no tool calls. Observed responses selected ordinary Gemma/OpenRouter, changed to Luna/ChatGPT after subscription disclosure, switched to official DeepSeek for actual Chinese input or informed choice, distinguished Chinese UI/output from input, preferred Soniox for English/Chinese quality with experiential/prepaid rationale, and recommended Cloud Free Tier Auto/Gemini for free use.

These are direct behavior observations, not judge verdicts or a replacement paired simulator pass. The first probe launch failed before model calls because the temporary script omitted the required localizer; the corrected probe used the existing localization path.

### Final frozen candidate: outcomes and unmet criteria

Both processes completed and saved 39 attempts, but exited with `failed_or_blocked`, not a release pass. Baseline: 18 passed, 12 failed, 9 blocked; candidate: 28 passed, 6 failed, 5 blocked. Existing scenario thresholds were met in 7/13 baseline and 10/13 candidate cases. A blocked scenario status does not mean its other attempts had no product failures.

| Scenario, successful runs out of three | Baseline | Candidate |
| --- | ---: | ---: |
| recognition-microphone | 2 | 2 |
| debug-translation-auth | 3 | 3 |
| unsolvable-internal | 2 | 3 |
| extra-reject-paid | 2 | 2 |
| routing-pending-approval | 0 | 1 |
| recommend-chatgpt | 0 | 0 |
| recommend-deepseek | 2 | 1 |
| recommend-gemma | 0 | 3 |
| policy-subscription-priority | 3 | 2 |
| policy-language-not-input | 0 | 2 |
| policy-asr-english-quality | 3 | 3 |
| policy-asr-chinese-quality | 0 | 3 |
| policy-asr-free | 1 | 3 |

All 39 attempts per arm passed the existing code prohibition checks. Recommendation explanations: baseline 9 pass / 10 fail / 5 unjudged; candidate 17 pass / 5 fail / 2 unjudged. Recommendation state: baseline 23 accepted and one incomplete; candidate all 24 accepted. State acceptance includes unchanged advice-only settings and does not establish correct initial advice.

Candidate failures remain explicit unmet completion criteria, not rewritten verdicts:

- `recommend-chatgpt/run-2`: the later answer selected Luna first but omitted the substantive subscription-based rationale; the initial Gemma answer omitted material OpenRouter preparation/cost. `run-3`: later subscription disclosure produced availability/login advice without updating the initial Gemma recommendation. The full authoritative guide is included in every work prompt (`instructions.ts:39-44`); selecting only the accounts context does not remove the recommendation block. Both subscription turns used the existing low-effort D decision. This does not prove low effort caused the failures; routing/effort changes are excluded from this issue.
- `policy-subscription-priority/run-3`: the correct Luna/subscription choice and later informed DeepSeek choice lacked explicit selection rationales. Correct option names and setup caveats did not satisfy explanation quality.
- `recommend-deepseek/run-3`: initial generic advice chose Gemma before Chinese-input use became known. The subsequent DeepSeek answer omitted its selection reason/material preparation; later setup repeated instructions without a completed transition before departure. Preserve the failed explanation/cost/tone judgments rather than treating the initial language alone as a Chinese-input requirement.
- `policy-language-not-input/run-3`: initial Gemma advice was correct. After an explicit comparison and challenge using the supplied benchmark/cost evidence, the assistant changed to official DeepSeek and later rewrote a sample sentence. The judge failed explanation/context use. The user's comparison makes the default-versus-informed-choice boundary ambiguous under the issue's explicit exception; this uncalibrated failure remains recorded, not reclassified as a pass. No general chat-translation capability is claimed.
- `routing-pending-approval/run-3`: work was selected instead of the required out-of-scope route. The displayed answer declined standalone translation, kept the proposal pending, then withdrew it after refusal. The routing failure is retained; changing D routing is outside scope.

Candidate blocks: two judge timeouts, two `relay_unavailable`, one `relay_request_failed`. Baseline blocks: two relay request failures, one relay unavailable, two judge access refusals, one judge timeout, one user timeout, one user network failure, and one user access refusal. Recovery probes did not guarantee campaign-wide availability; no blocked attempt was replaced.

### Final frozen candidate: usage and tradeoffs

Known token sums across all 39 attempts per arm:

| Arm / role | Uncached input | Cache read | Cache write | Output | Reasoning subset |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline D | 132,335 | 123,300 | 0 | 0 | 0 |
| Candidate D | 137,342 | 128,232 | 0 | 0 | 0 |
| Baseline M | 1,950 | 3,201,428 | 303,651 | 94,720 | 79,448 |
| Candidate M | 4,220 | 2,291,874 | 247,264 | 87,366 | 76,895 |
| Baseline user | 555 | 41,296 | 304,812 | 11,351 | 3,830 |
| Candidate user | 564 | 45,704 | 291,254 | 11,337 | 3,661 |
| Baseline judge | 90 | 0 | 2,740,744 | 81,102 | 69,741 |
| Candidate judge | 102 | 0 | 2,607,593 | 72,162 | 60,082 |

Usage coverage, known/dispatches: baseline D 100/102, M 218/221, user 185/188, judge 30/33; candidate D 104/104, M 181/184, user 188/188, judge 34/36. Missing usage is unknown, not zero. Reasoning is part of output. D/M currency cost remains unavailable. Known user/judge standard-price estimates total $8.10344980 baseline and $7.66440820 candidate, excluding unknown calls; these are not invoices or total assistant cost.

Cache treatment is not cold-cache equivalent: both prefixes received health requests and could retain previous provider cache; the candidate additionally received the seven policy probes while judge access recovered. Actual cache usage is reported rather than assuming identical warm-up state.

The following summary includes only the 16 identical attempt IDs passing in both arms. All other 23 pairs remain visible in the index but are excluded from success-matched efficiency/latency. The index retains exact included IDs and per-role metrics.

| Measurement, 16 paired runs | Baseline | Candidate |
| --- | ---: | ---: |
| M round trips | 106 | 83 |
| M input tokens, including cache | 1,764,701 | 1,267,665 |
| M output tokens, including reasoning | 34,649 | 40,449 |
| Tool calls | 62 | 45 |
| Rejected calls | 9 | 2 |
| Identical name/argument repeats | 8 | 3 |
| Median first displayed response, seconds | 13.77 | 10.35 |
| Median accepted-dialogue duration, seconds | 25.08 | 24.40 |

Fewer round trips/input tokens did not mean lower output: matched candidate output, including reasoning, increased. This is combined structure/policy evidence, not isolated causal attribution or a universal savings claim. The same display-time/completion-proxy limitations stated above apply.

Across all attempts, tool calls were 114 versus 77; rejected calls 21 versus two; identical repeats 20 versus four. Baseline rejections were `invalid_tool_batch`. The two candidate `invalid_tool_call` rejections omitted the required `request` wrapper for `read_app` in translation-auth and internal-diagnostic attempts; both conversations subsequently passed. Synthetic `query_unavailable` results were 13 versus 11. Repeats are not automatically unnecessary and invalid model arguments were not silently accepted.

The final evidence index preserves all four campaigns (312 planned attempts), original judgments, final failed-candidate dialogues, source hashes, and both success-matched subsets. No historical outcome was rejudged or deleted.

### Acceptance limits and next actions

Targeted simulation is not full release evidence. Owner calibration remains blocked; no calibrated `live_pass` is claimed. GUI evidence is separately scoped above. Synthetic providers/audio/signup/chatbox do not establish real hardware/provider/VRChat delivery.

Resolve the retained recommendation-update/explanation failures and obtain independent checkpoint/terminal review before accepting completion. The completed comparison still contains blocked measurements and is not a passing acceptance result. Calibrate the ambiguous comparison/informed-choice judgment before drawing a stronger conclusion; do not rewrite the original result. Preserve the unchanged criteria and historical reports. Do not substitute another judge, rejudge historical failures to improve scores, or close either issue using the older candidate's statistics.
