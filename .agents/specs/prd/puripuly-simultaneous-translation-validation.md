---
id: PRD-PURIPULY-SIMULTANEOUS-VALIDATION-001
status: reviewed
source: .agents/specs/prd/drafts/puripuly-bounded-dual-2flight.source.r5.md
baseline_ref: dev@b3b4cc4c0fbb1b25140881fb2c41031cb2bc8ad6
integration_target: dev
predecessor_contract_ref: .agents/specs/prd/puripuly-simultaneous-translation-engine.md@f2a030b28a66f1f8e153c3e55a2307811bd23630
document_review_verdict: ready
blocking_open_decisions: 0
---

# Outcome

PuriPuly Heart evaluates one exact accepted simultaneous-translation implementation candidate against real provider routes, representative speech traces, supported Windows production composition, user experience, quality, latency, stability, and cost. The result is a durable product-owner decision to approve a versioned policy for developer preview, limited preview, or general availability, or to withhold activation with explicit repair evidence. Validation never weakens channel safety, authoritative-final independence, bounded concurrency, compatibility, or the accepted implementation contract.

# Established Baseline

## Code baseline

- The original product baseline is canonical `dev@b3b4cc4c0fbb1b25140881fb2c41031cb2bc8ad6`.
- The predecessor authority is the exact reviewed blob of `PRD-PURIPULY-SIMULTANEOUS-ENGINE-001` recorded in frontmatter. Validation cannot start until that contract has an accepted exact implementation candidate and complete predecessor evidence.
- The predecessor candidate is disabled for product rollout, keeps `TURN_BASED` as the default, enables no unverified live provider route, and exposes deterministic fixtures, diagnostics, versioned policy inputs, and safe degradation needed by this contract.
- At validation activation, the GitHub scope snapshot and Goal state must pin the accepted predecessor candidate SHA, latest integration-target head, provider/model/route candidates, evidence corpus identity, and approved credential/cost boundary.

## User-visible surfaces

- Simultaneous-mode selection and availability, degraded, finalizing, failure, and rollback presentation.
- Self and eligible Peer live overlay previews and their authoritative final replacement.
- Existing Self final chatbox behavior, unconditional Peer chatbox denial, manual final-only translation, context/history, settings persistence, shutdown, and restart.
- Developer preview, limited preview, general-availability eligibility, and runtime rollback controls.

## Actual product entrypoints

- Source and packaged Windows GUI entry through `puripuly_heart.main.main` and the production Flet application.
- Real production capture, translation, provider, output, settings, UI-application, and shutdown owners rather than a test-only alternate composition.
- Desktop overlay and, when affected, native VR overlay presentation; isolated VRChat OSC/chatbox verification for Self and Peer denial.
- Exact supported provider/model/routes through production provider configuration and SecretStore-backed credentials.

## Platform and environment

- Supported Windows x64 with packaged Python 3.12.
- Source and PyInstaller-packaged production compositions.
- Representative Self and Peer audio and STT partial traces, including short, medium, long, rewrite, mixed-latency, provider failure, fallback, cancellation, and shutdown scenarios.
- Real provider access and paid request execution require separate credential and cost authorization and sanitized evidence handling.

## Compatibility baseline

- Every product and architecture invariant in the reviewed predecessor implementation contract remains binding as established candidate behavior.
- `TURN_BASED` is compared with the original baseline and cannot regress because simultaneous validation succeeds or fails.
- Final prompts, context, provider aliases, fallback availability, output routing, settings and SecretStore compatibility, Broker `/v1`, native protocols, executable/installer identity, and user-data locations remain compatible.
- Implementation completion, validation completion, preview approval, GA approval, merge, push, deployment, release, evidence publication, and cleanup are separate decisions.

# Scope

## Included

- Exact route/model conformance for assistant prefill, response shape, requested and observed provenance, cancellation behavior, usage/cost reporting, and safe degradation.
- Sanitized real STT partial-trace replay and controlled temporal/fault injection across final-only, free one-flight, bounded-dual one-flight, bounded-dual two-flight, and verified-only comparison policies.
- Quantitative characterization of correctness, final quality, live utility, first-draft latency, freshness, display stability, frontier risk, completion usage, total cost, useful/wasted flights, final priority, and shutdown.
- Versioned selection of provider capability matrix and policy profile within the predecessor contract.
- Source and packaged Windows production-composition acceptance, all supported locale inputs, and explicit product-owner UX, cost, and activation decision.
- Runtime rollback proof and durable evidence sufficient to approve one preview/availability stage or withhold activation.

## Non-goals

### NG-V001 — Silent implementation redesign

Validation does not change frontier provenance, canonical/display separation, snapshot lineage, maximum-two-flight semantics, result disposition, owner boundaries, channel safety, final authority, persistence compatibility, or other durable predecessor meaning.

### NG-V002 — Automatic release

A successful validation decision does not merge, push, deploy, release, publish credentials/evidence, change production defaults, or clean a worktree without separate approval.

### NG-V003 — Universal provider claims

Conformance and performance conclusions apply only to the exact tested model, route, provider, request shape, prompt, normalizer, unitizer, policy, application, and evidence versions.

### NG-V004 — Unsupported coverage expansion

Very-long-turn segmentation, multi-target live output, mixed-language live output, manual live translation, and live Peer output without an eligible stable language hint remain outside validation.

### NG-V005 — Training and research productionization

No model training, LoRA/DoRA, survival model, grammar engine, n-best/RALCP product path, or model-internal attention requirement is introduced.

# Requirements

## REQ-V001 — Exact predecessor and evidence identity

Validation operates on one exact accepted implementation candidate. Every result records candidate SHA, original baseline, latest reconciled integration target, reviewed implementation-contract blob, provider/model/route identity, prompt version, normalizer version, unitizer version, policy profile, trace-corpus identity, environment, and evidence timestamp. Results from another identity cannot be combined as if they belonged to the candidate.

## REQ-V002 — Provider conformance matrix

Each route eligible for live traffic is directly tested for assistant-prefill request acceptance, suffix-only or full-echo response behavior, normalization, requested and observed provider provenance, streaming policy, optional log-probability shape, concurrency/capacity, timeout, fallback, cancellation, usage, and cost reporting. Unsupported, ignored, unknown, or changed behavior selects a documented safe degraded policy or makes the route ineligible; no metadata or billing stop is inferred.

## REQ-V003 — Representative and private trace corpus

The evaluation corpus contains representative short, medium, and long Self and Peer turns; clean/soft appends; tail and mid rewrites; language/context/route changes; completion inversion; provider tail latency; source stop; finalization; fallback; capacity pressure; cancellation; shutdown; and restart. Corpus composition and exclusions are reported. Raw source, audio, translation, and provider bodies are collected only with explicit opt-in, remain sanitized and access-controlled, and are not placed in default logs or public evidence.

## REQ-V004 — Comparative policy evaluation

The exact same eligible corpus and context inputs compare final-only, free one-flight retranslation, bounded-dual one-flight, bounded-dual two-flight, and verified-only two-flight policies where supported. Comparisons separate route unavailability and missing language hints from eligible denominators and report paired distributions rather than only aggregate means.

## REQ-V005 — Correctness and safety dominance

Activation is ineligible if validation observes any Peer or live chatbox publication, partial history write, final use of live prefill/state, display-only canonical promotion, verified-provenance violation, provisional bound violation, stale-lineage mutation, screen regression, final overwrite/starvation, closed-turn resurrection, forbidden flight/capacity count, unbounded lifecycle wait, post-close output, secret disclosure, or fabricated provenance/cost. Performance evidence cannot offset a correctness or safety violation.

## REQ-V006 — Final quality and live utility

Authoritative final quality is compared with the original turn-based baseline using the same complete source, context, final prompt family, and fallback availability. Live utility is evaluated separately for comprehensibility, correctness, timeliness, correction burden, and final replacement distance. Source-understanding and non-source-understanding users are distinguished where practical. A route or policy cannot be activated when the product owner judges final quality materially inferior or live preview harm greater than its latency benefit.

## REQ-V007 — Characterization targets and policy selection

The following source-r3 values are mandatory reported targets, not automatic claims: at least 90% of ordinary turns at or below two seconds create zero live logical flights; at least 90% of eligible turns at or above six seconds show a draft before source final; median first-draft latency is at most 3.0 seconds; p95 first-draft latency is at most 5.5 seconds; median completion units improve by at least 20% against free live retranslation; median turn cost improves by at least 15% against free live retranslation; median turn cost is at most 2.5 times final-only; and wasted live logical-flight rate is at most 30%. Two-flight freshness improvement, final-quality margin, rollback budget, useful-second-flight floor, cost ceiling, and any approved alternative to these targets require an explicit product-owner decision recorded with the evidence. No value is promoted by silently tuning against the acceptance corpus.

## REQ-V008 — Versioned and bounded activation profile

Any approved profile records live gate, source normalization, unit boundary, frontier lead, bootstrap cap, verified guard, generation and wall-time TTL, overlap, economical-prefill threshold, audit cadence/debt, evidence horizon, soft/hard deadline, cooldown, duplicate policy, freshness window, request/branch budgets, final reserve, optional mask behavior, prompt, and provider conformance versions. Profile selection stays within predecessor invariants, has an explicit rollback target, and does not require language-specific grammar rules.

## REQ-V009 — Windows production composition

The selected profile is exercised through the real supported Windows source and packaged applications for Self short/long, eligible Peer short/long, Self/Peer overlap, manual translation, mode switch, context update, provider delay/failure/degradation, prefill degradation, final fallback, desktop overlay, native VR overlay if affected, Self chatbox, Peer denial, toggle-off, settings migration, shutdown, and restart. The candidate leaves no owned task, provider, overlay/helper, or resource after terminal shutdown.

## REQ-V010 — UX, locale, and accessibility decision

The product owner reviews reproducible simultaneous-mode states for all supported locales, including available, unavailable, degraded, first draft, mutable update, alternate display, finalizing, final replacement, live failure, and rollback. Mask hide/fade behavior, same-revision alternate-to-canonical replacement, finalizing indicator, degraded copy, and live-failure behavior are explicitly approved or disabled. Preview tooling remains inert and hidden outside its debug flag.

## REQ-V011 — Activation decision and rollback

The terminal record selects exactly one outcome: `not_approved`, `developer_preview`, `limited_preview`, or `general_availability`. Any approval names the exact eligible routes, channels, language-hint rules, policy profile, settings/UI exposure, evidence limitations, runtime rollback controls, and remaining monitoring. `Not_approved` identifies the failed claims and whether the next action is evidence repair, bounded implementation repair, a new product decision, or abandonment. Actual deployment or release remains separate.

## REQ-V012 — Bounded validation changes

Validation may add or repair evidence fixtures, diagnostics, sanitized export, versioned policy data, approved localized copy, and provider-capability records. A finding that requires changing product behavior, durable algorithm meaning, ownership, persistence, security, compatibility, supported scope, or required evidence stops validation and requires a reviewed repair or successor implementation authority.

# Protected Invariants

## Product invariants

### INV-P-V001 — Zero channel-safety violations

Live chatbox publication and Peer chatbox publication remain zero in every evaluated and approved configuration.

### INV-P-V002 — Independent authoritative final

The final result remains a no-prefill translation of complete source with frozen final context and cannot be derived from or overwritten by live state.

### INV-P-V003 — Turn-based fallback

`TURN_BASED` remains safe, compatible, and available when simultaneous mode, a route, Peer live eligibility, prefill, optional log-probability use, masking, or two-flight behavior is disabled.

### INV-P-V004 — Evidence before availability

No model/route/channel/profile is presented as available beyond the exact evidence that supports it.

## Durable architecture invariants

### INV-A-V001 — Frozen predecessor meaning

Validation cannot reinterpret or weaken the reviewed predecessor implementation contract.

### INV-A-V002 — Reproducible profile and evidence

Every activation decision is reproducible from exact code, contract, route, prompt, unitizer, normalizer, policy, trace, environment, and evidence identities.

### INV-A-V003 — Safe degradation and rollback

Capability loss, provider variance, threshold failure, or runtime risk moves to a documented narrower or disabled policy without changing authoritative final or channel behavior.

### INV-A-V004 — Privacy and credential separation

Provider credentials remain in SecretStore, credential access is separately approved, and raw private evaluation material never becomes default or implicitly published evidence.

# Approved Decisions

- Validation is a successor Goal and a GitHub sub-issue of implementation; it cannot start before an accepted exact predecessor candidate exists.
- Implementation and validation use separate reviewed PRD authorities and separate Goal completion boundaries.
- Correctness and safety have zero-tolerance acceptance and dominate latency, cost, coverage, and user-preference results.
- Source-r3 numerical candidates are evaluated and reported, but the activation record must identify which exact targets and budgets the product owner approved.
- A provider capability is established by exact conformance evidence, not provider documentation, parameter acceptance, or requested route alone.
- Missing provenance, remote cancellation effect, billing stop, and unknown cost remain explicit unknowns.
- Product rollout is staged and retains switches for simultaneous off, prefill off, two-flight off, Peer live off, optional log-probability off, and mask off.
- A failed validation does not authorize weaker evidence or an opportunistic redesign.

# Open Product Decisions

None before validation execution. The profile, UX, cost, and activation outcome are deliberate evidence-dependent decisions owned by REQ-V007, REQ-V010, and REQ-V011; validation is incomplete until the product owner records them. If the product owner is unavailable or declines to decide, the Goal remains blocked and no activation is inferred.

# Acceptance Criteria

| AC | Verifies | Evidence class | Required environment | Pass condition |
|---|---|---|---|---|
| AC-V001 | REQ-V001, INV-A-V001, INV-A-V002 | provenance audit | exact accepted predecessor candidate and latest reconciled `dev` | Every artifact resolves to one candidate, reviewed predecessor blob, route/profile/corpus/environment identity, and no result from an incompatible identity contributes to a claim. |
| AC-V002 | REQ-V002, INV-P-V004, INV-A-V003 | real-provider conformance + controlled faults | each proposed provider/model/route using separately authorized credentials and cost | Each route has an exact capability disposition and raw sanitized proof; unknown/unsupported behavior is degraded or ineligible; fabricated provenance, cost, or cancellation claims are zero. |
| AC-V003 | REQ-V003, REQ-V004 | trace inventory + deterministic replay | versioned sanitized corpus and identical comparison inputs | Required scenario classes and exclusions are enumerated; all applicable policies replay the same eligible inputs; denominators and paired distributions are reported; default/public artifacts contain no unapproved raw data. |
| AC-V004 | REQ-V005, INV-P-V001, INV-P-V002, INV-P-V003 | automated/fault + real Windows production composition | source and packaged Windows, real owners, isolated overlay/chatbox endpoints | Every enumerated correctness and safety count is measured and attributed. An approval outcome requires every violation count to be zero and `TURN_BASED` to remain compatible. Any nonzero count is recorded as a failed gate, makes activation ineligible, and can satisfy this criterion only through a terminal `not_approved` decision with simultaneous traffic disabled. Missing execution or an unmeasured count is incomplete evidence rather than a failed result. |
| AC-V005 | REQ-V006 | comparative quality + user evaluation | complete-source baseline and candidate finals; representative live traces; product-owner review | Final and live results are separately reported; the product owner records whether final quality is non-inferior and live utility exceeds correction harm, with no live result promoted to final evidence. |
| AC-V006 | REQ-V007, REQ-V008, INV-A-V002 | statistical characterization + configuration audit | exact corpus, exact provider routes, versioned profiles | Every mandatory target and unpinned budget is reported with distribution/sample size and uncertainty; the selected profile and any approved deviation are explicit; evaluation-corpus tuning is disclosed; no unversioned value is activated. |
| AC-V007 | REQ-V009, INV-A-V003 | source and packaged production-composition + temporal/manual evidence | supported Windows x64/Python 3.12 with real GUI and affected external paths | Every required source and packaged scenario is executed and its result is recorded at one exact candidate/profile. An approval outcome requires the complete matrix to pass, no final starvation, terminal shutdown, correct settings restart and output/channel behavior, and no owned resource after close. An observed failure can satisfy this criterion only through a terminal `not_approved` decision that records the failed gate and leaves simultaneous traffic disabled; an unexecuted required scenario remains incomplete evidence. |
| AC-V008 | REQ-V010 | reproducible comparative visual/interaction + locale parity | accepted production UI line, all supported locales, inert debug-preview states and representative real states | Every required state is reviewable; locale inputs are in parity; preview side effects are zero; the product owner records each UX decision or disables the undecided behavior. |
| AC-V009 | REQ-V011, INV-P-V004, INV-A-V003 | decision record + rollback exercise | exact validated candidate/profile and production composition | Exactly one terminal activation outcome is recorded; any approval is no broader than evidence and has working rollback controls; `not_approved` leaves simultaneous traffic disabled and identifies the next authority boundary. |
| AC-V010 | REQ-V012, INV-A-V001 | diff classification + independent review | complete validation candidate including staged, unstaged, and relevant untracked changes | Every code/config change is within the bounded validation surface; no durable predecessor meaning changed; material findings have a separate reviewed repair authority rather than a weakened validation claim. |
| AC-V011 | all requirements and invariants | evidence inventory + fresh independent implementation review | one exact terminal candidate reconciled with latest `dev` | Provider matrix, corpus manifest, comparative report, policy profile, quality/UX decision, source and packaged Windows evidence, test disposition, open findings, rollback record, and terminal decision are complete with no unowned or falsely proven claim. |

# Decision Authority

## Executor may decide

- Reversible evidence-harness details, corpus tooling, private metrics representation, statistical presentation, diagnostic implementation, and validation execution order.
- Policy candidates within the predecessor contract, provided they remain versioned and are not activated before the required product-owner decision.

## Independent review required

- Provider capability acceptance, production-composition evidence, bounded validation code/config changes, policy activation wiring, compatibility or rollback claims, and terminal completion.
- Any conclusion that evidence remains valid after integration-target, provider, route, prompt, policy, corpus, or environment drift.

## User decision required

- Credential and paid-provider access, raw-data collection, cost budget, final-quality judgment, UX choices, approved policy targets, provider/channel/language coverage, preview or GA outcome, evidence publication, and any weakening or scope change.

# Completion Rule

Every acceptance criterion must be directly evaluated in its required environment and evidence class. For an approval outcome, every activation gate and behavioral pass condition must pass. For `not_approved`, an executed gate may fail and the criterion is complete when the required evidence is fully captured, the failure is explicitly attributed in the decision record, simultaneous traffic remains disabled, and no missing required execution is represented as a failure result. Validation completes only when the exact evidence package and explicit product-owner activation decision are accepted by a fresh independent reviewer. A `not_approved` decision is not feature-delivery acceptance. Merge, push, deployment, release, credential access, evidence publication, and cleanup remain separately approved actions.
