---
id: PRD-LLM-TAIL-LATENCY-POLICY-SEAM-001
status: reviewed
source: .agents/specs/prd/drafts/llm-tail-latency-policy-seam.source.r1.md
baseline_ref: dev@c41aa7596d0396f0afc2e962c7bb2c69c91f9791
integration_target: dev
document_review_verdict: ready
blocking_open_decisions: 0
---

# Outcome

PuriPuly Heart gains the minimum foundation required to later change when a secondary LLM
attempt starts, without changing any observable product behavior in this contract. The
secondary-start decision becomes a replaceable seam that receives its timing from the
already-resolved fallback plan; the existing persisted race event carries enough additional
context to judge tail behavior, cost, and route identity after the fact; and a suppressed
developer entrypoint can execute real bring-your-own-key OpenRouter calls and emit a
reproducible measurement report. Choosing a different delay, adopting an incident mode, or
pinning provider pairs are explicitly outside this contract and remain user decisions to be
taken after the measurement report exists.

# Established Baseline

## Code baseline

- Canonical baseline is `dev@c41aa7596d0396f0afc2e962c7bb2c69c91f9791` with a clean working
  tree.
- `FallbackRacingLLMProvider` starts the primary attempt, starts the fallback attempt either
  when a 2000ms timer elapses or when the primary attempt raises, adopts the first
  successful translation, grants the losing attempt a 50ms grace window, and then cancels it.
- The provider performs no error classification. Any exception from the primary attempt
  triggers the fallback attempt.
- `ResolvedLLMFallbackPlan` carries `timeout_ms=2000` and `loser_grace_ms=50` and is the only
  place those values are decided. Neither value is exposed through settings.
- The race is composed *inside* the concurrency-limiting provider wrapper, so a second
  in-flight attempt does not consume an additional concurrency slot and concurrent outbound
  calls can reach twice the configured concurrency limit.
- OpenRouter provider preferences already pin upstream providers per model. The recommended
  translation model uses an explicit ordered provider list; another model uses an explicit
  allow list combined with latency sorting; the default path uses latency sorting with an
  ignore list.
- Account-level usage cost is read from a separate OpenRouter key-metadata endpoint, not from
  the completion response. The completion response is parsed for neither per-request cost nor
  token usage, and the identity of the upstream provider that actually served the request is
  not captured anywhere.
- Runtime logging applies per-sink redaction filters to the persisted sink.

## User-visible surfaces

- Translation fallback presence and target, selectable only from the five established
  translation-fallback presets, including the preset that disables fallback entirely.
- Perceived translation completion latency in the running GUI, whose current distribution is
  not established by this contract and is one of the things the measurement report exists to
  characterize.
- The persisted diagnostic log file that users can share for support.

## Actual product entrypoints

- Production GUI: `python -m puripuly_heart.main run-gui`.
- A new developer subcommand of `puripuly_heart.main`, suppressed from help output as the
  two existing suppressed subcommands are, and requiring an explicit report output path as
  the established evidence-producing subcommands do.

## Platform and environment

- Windows with the repository `.venv`, Python 3.12.
- Python application surface only. The Broker service and the Rust VR overlay are
  unaffected and require no verification for this contract.
- Real OpenRouter access through user-supplied bring-your-own-key credentials is required to
  satisfy the harness acceptance criterion. Managed and broker-issued credentials are not
  acceptable for that evidence.

## Compatibility baseline

- The persisted race event keeps its `[Persisted][Fallback] ` message prefix and the names
  and meanings of every key it already emits. The established tests over this surface assert
  the prefix and the presence of individual key-value substrings, and separately parse the
  payload as JSON; no established test asserts payload equality or a closed key set.
  Additive fields therefore need not break them, and updating those assertions is permitted
  but not required.
- Settings serialization stays round-trippable with no new key and no new persisted field,
  so no forward migration and no persisted-data backup are required.
- Secrets continue to load through SecretStore with unchanged key compatibility.
- The `LLMProvider` translation contract continues to return a completed translation, and
  provider teardown continues to await `close()`.
- Translation prompts, provider aliases, fallback availability, output routing, and the
  translation-fallback preset set remain unchanged.
- Document approval, implementation authorization, measurement execution, hypothesis
  judgement, merge, push, and release remain separate decisions.

# Scope

## Included

- Separating the decision of whether and when to start the secondary attempt from the
  execution of the race, with the decision receiving its timing values by injection from the
  already-resolved fallback plan.
- Extending the existing persisted race event with additional context sufficient to judge
  tail behavior, duplicate-call rate, cost, and route identity after the fact.
- Capturing the upstream provider identity actually used by an OpenRouter request when the
  provider response exposes it.
- A suppressed developer entrypoint that executes real bring-your-own-key OpenRouter calls
  and writes a reproducible measurement report to an explicit output path.
- Confining any comparison of OpenRouter route configurations to that entrypoint.

## Non-goals

### NG-001 — Changing the effective secondary-start timing
The production default secondary-start timing and loser grace window remain exactly as
established. Selecting a different value is deferred to a separate decision informed by the
measurement report.

### NG-002 — Incident mode, provider pairing, and provider performance tracking
Time-based incident state, explicit upstream provider pair selection in production, a
provider performance database, and a circuit breaker are not introduced.

### NG-002a — Automatic delay computation
No real-time percentile-based or otherwise adaptive computation of the secondary-start delay
is introduced. The timing remains a static resolved value.

### NG-003 — Route resolution in production composition
No route-resolution abstraction is introduced into production composition, and no
same-model race path becomes reachable in production. The meaning of every
translation-fallback preset, including the preset that disables fallback, is unchanged.

### NG-004 — LLM token streaming and hard timeouts
Token streaming is not reintroduced. No mechanism that forcibly terminates a request after a
deadline is introduced. The two-second figure remains a performance objective, not a timeout.

### NG-005 — Error classification refinement
The existing behavior in which any primary exception triggers the secondary attempt is
retained. Distinguishing retryable from non-retryable failures is not in scope.

### NG-006 — Settings, UI, and localization changes
No settings key, settings UI control, locale string, or persisted schema field is added or
changed.

### NG-007 — Hypothesis conclusions
Deciding whether earlier secondary starts help, whether explicit provider separation is
worth its complexity, or whether tails cluster in time is not part of completion. Producing
an inconclusive measurement report satisfies this contract.

# Requirements

## REQ-001 — Replaceable secondary-start seam
The decision of whether and when to start the secondary attempt is expressed as a distinct,
substitutable collaborator of the race execution, separate from attempt supervision, winner
selection, failure handling, and instrumentation. The seam receives its timing values by
injection from the resolved fallback plan and introduces no independent source of truth for
them.

## REQ-002 — Observable behavior preserved
With the established resolved values, the composed production provider produces the same
externally observable behavior as the baseline: the same secondary-start trigger conditions,
the same winner selection, the same failure outcomes, the same returned translation
contract, and an unchanged meaning for every race event key that already exists. Fields added
under REQ-003 are the only permitted difference in the emitted event.

## REQ-003 — Additive instrumentation
The existing persisted race event is extended only by adding fields. The message prefix and
every existing key name and meaning are preserved. The added context is sufficient to
determine, per logical translation, when and why the secondary attempt started, the logical
completion latency, whether that latency exceeded the relevant performance thresholds,
token usage and cost attributable to the attempts that were started rather than only the
winner, and a place to record incident state if one is ever introduced. The performance
thresholds recorded are two, three, and four seconds of logical completion latency; two
seconds is the performance objective and none of the three terminates a request.

## REQ-004 — Upstream provider identity
When an OpenRouter response exposes the identity of the upstream provider that served the
request, that identity is captured and carried into the instrumentation for both attempts.
When the response does not expose it, the corresponding field is emitted as absent or null
and the remaining instrumentation is still complete.

## REQ-005 — Harness entrypoint
A developer entrypoint of the production application, suppressed from help output, executes
real OpenRouter translation calls, accepts the sample count as an explicit argument with no
default so that omitting it fails rather than spending, accepts an explicit report output
path, rejects managed and broker-issued credentials and proceeds only with
bring-your-own-key credentials, and loads those credentials through SecretStore.

## REQ-006 — Harness report
The harness writes a report to the requested path containing the measured per-attempt
latencies, the observed upstream provider identity when available, usage cost, error
outcomes, the route configurations compared, and the reproduction parameters including the
sample count actually executed. The report contains no conversational content as constrained
by INV-P-001. The report states plainly when the sample is insufficient to distinguish the
compared configurations rather than asserting a conclusion.

## REQ-007 — Production composition untouched
Production composition continues to build the primary attempt from the selected translation
target and the secondary attempt from the preset fallback target. No route-resolution
abstraction and no same-model race path is reachable from any production entrypoint.

# Protected Invariants

## Product invariants

### INV-P-001 — No conversational content in instrumentation or reports
No utterance text, translation output, prompt text, or any fragment thereof appears in the
instrumentation payload or in the harness report artifact in any form. Only aggregate values
such as lengths, token counts, timings, identifiers, and cost are permitted.

### INV-P-002 — Channel separation
Peer utterances are never routed to the VRChat chatbox, and self, peer, and system outputs
remain separate product channels.

### INV-P-003 — Unchanged production defaults and user surfaces
The effective secondary-start timing, the loser grace window, the translation-fallback
preset set and their meanings, settings persistence, and every localized string remain as
established.

## Durable architecture invariants

### INV-A-001 — Two active external attempts
At most two external attempts are active concurrently for a single logical translation.

### INV-A-002 — Provider contract and lifecycle
The translation call continues to return a completed translation to its caller, provider
teardown continues to await `close()`, and no attempt outlives the shutdown path that owns
it.

### INV-A-003 — Persisted event compatibility
The persisted race event keeps its message prefix, and no existing key is renamed, removed,
or given a changed meaning.

### INV-A-004 — Credential and persistence safety in the harness
The harness loads credentials through SecretStore, never writes settings, never mutates
stored secrets, never calls the Broker, and writes only to the requested report path.

# Approved Decisions

- **D1** Scope is limited to the seam, the instrumentation, and the harness. Production
  behavior is unchanged.
- **D2** Route comparison is confined to the harness; production composition gains no route
  resolution and no same-model race.
- **D2a** The comparison baseline for route configuration is the current OpenRouter provider
  preference output as it exists today, not an idealized pure latency-sort configuration.
- **D3** Instrumentation is additive on the existing persisted event; where an established
  assertion over that surface is disturbed by additive fields, updating it is authorized as
  an intended contract change rather than required.
- **D4** The harness is a suppressed subcommand, minimal in scale, bring-your-own-key only,
  with the sample count supplied explicitly and recorded in the report, and with no monetary
  ceiling clause in this contract. This contract additionally requires that the sample-count
  argument have no default, so that omitting it fails before any call is issued; that
  requirement is an intentional safety addition of this contract, consistent with the
  required-argument precedent already used by the established evidence subcommands.
- **D5** No settings key and no persisted field are added; the resolved fallback plan remains
  the single authority for timing values.
- **D6** Harness completion is the production of the report. Reaching a conclusion is not a
  completion condition.

# Open Product Decisions

`None`

Deferred items outside this contract, each requiring a user decision before any future work:
selection of a different secondary-start timing; adoption of incident mode; adoption of
explicit upstream provider pairing in production; adoption of a circuit breaker. Owner: user.
Resolution boundary: after the REQ-006 report exists.

# Acceptance Criteria

| AC | Verifies | Evidence class | Required environment | Pass condition |
|---|---|---|---|---|
| AC-001 | REQ-001, REQ-002, INV-A-001, INV-A-002 | automated | Windows `.venv`; `black src tests`, `ruff check src tests`, `python -m pytest` | Formatting and lint report no findings, and the full existing test suite passes; any established persisted-log assertion that additive fields disturb may be updated under D3, and no other test is weakened or removed. The secondary-start decision is exercised through a substituted seam without altering outcomes. |
| AC-002 | REQ-001, INV-A-001, INV-A-002 | automated | Windows `.venv`, `python -m pytest` with test doubles and injected delays | Under controlled timing, the secondary attempt starts at the injected point, the first valid success is adopted, a late-finishing loser never replaces the adopted result, a single-side failure does not abort the surviving attempt, and no third attempt becomes active. This evidence is explicitly not accepted as evidence of real provider tail improvement. |
| AC-003 | REQ-003, INV-A-003, INV-P-001 | automated + manual log inspection | Windows `.venv`; emitted persisted log file from a real GUI session | The emitted persisted line retains the established prefix and every prior key with unchanged meaning, carries the added context, passes the persisted-sink redaction filter, and contains no utterance, translation, or prompt text. |
| AC-004 | REQ-004 | manual + real-provider execution | Windows `.venv` with real bring-your-own-key OpenRouter credentials | The raw OpenRouter response is inspected and either the upstream provider identity is captured into the instrumentation for both attempts, or the response is demonstrated not to expose it and the field is emitted absent or null with the remaining instrumentation complete. |
| AC-005 | REQ-005, INV-A-004 | manual + real-provider execution | Windows `.venv`, suppressed subcommand of the production entry, real bring-your-own-key credentials | Invoking the entrypoint without the sample-count argument fails without issuing any call; invoking it with managed or broker-issued credentials is refused; credentials are demonstrably obtained through SecretStore rather than an environment variable or literal introduced for this entrypoint; a successful run leaves settings files and stored secrets byte-identical and writes only the requested report path. |
| AC-006 | REQ-006, INV-P-001, D2a, D6 | manual + real-provider execution + comparative | Windows `.venv` with real bring-your-own-key OpenRouter credentials | A report artifact exists at the requested path containing per-attempt latencies, upstream provider identity when available, usage cost, error outcomes, and the executed sample count; the compared route configurations are identified, one of which is the current OpenRouter provider preference output as it exists today; the artifact contains no utterance, translation, or prompt text; and where the sample cannot distinguish the compared configurations the report says so instead of asserting a conclusion. |
| AC-007 | REQ-007, NG-003, INV-P-003 | automated + manual | Windows `.venv`; running GUI at `python -m puripuly_heart.main run-gui` | No production entrypoint can reach a route-resolution abstraction or a same-model race path; the translation-fallback preset set and their effects are unchanged; the effective secondary-start timing and loser grace window equal the established values; settings files round-trip with no new key. |
| AC-008 | INV-P-002 | automated | Windows `.venv`, `python -m pytest` | Peer-channel routing tests continue to pass unchanged, demonstrating peer utterances still never reach the VRChat chatbox and channel separation is intact. |
| AC-009 | NG-001, NG-002, NG-002a, NG-004, NG-005, NG-006 | manual diff review | Candidate diff against `baseline_ref` | The candidate introduces no timing default change, no incident state, no provider pairing, no provider performance store, no adaptive delay computation, no streaming, no request-terminating deadline, no error classification branch, and no settings, UI, or locale change. |

# Decision Authority

## Executor may decide

- reversible implementation details
- private types and internal APIs
- file and helper placement
- names of added instrumentation fields
- harness sample count and the specific route configurations compared
- implementation sequence
- tests and diagnostics

## Independent review required

- durable boundary reliance
- production cutover
- legacy, compatibility, migration, rollback, or fallback path removal
- persistence, security, lifecycle, concurrency, or public API change
- material strategy pivot
- terminal completion

## User decision required

- observable product behavior
- scope or non-goal
- compatibility break
- irreversible migration
- security posture
- supported platform or provider
- required evidence weakening

# Completion Rule

Every acceptance criterion must be directly proven in its required environment and evidence
class. Automated tests alone cannot replace platform, production-composition, comparative,
temporal, or manual evidence. In particular, AC-004, AC-005, and AC-006 require real
bring-your-own-key provider execution and are not satisfiable by test doubles, and the
controlled-timing evidence of AC-002 may never be presented as evidence that real provider
tail latency improved.
