---
id: PRD-PURIPULY-SIMULTANEOUS-ENGINE-001
status: reviewed
source: .agents/specs/prd/drafts/puripuly-bounded-dual-2flight.source.r5.md
baseline_ref: dev@b3b4cc4c0fbb1b25140881fb2c41031cb2bc8ad6
integration_target: dev
document_review_verdict: ready
blocking_open_decisions: 0
---

# Outcome

PuriPuly Heart gains a bounded simultaneous-translation engine for eligible long Self and Peer speech turns while preserving the established turn-based experience. In simultaneous mode, mutable live translations may appear only on the overlay before speech ends; the completed source turn is then translated independently without live prefill and atomically becomes the authoritative result. The engine remains disabled for product rollout until the separate validation and activation-decision contract is completed.

# Established Baseline

## Code baseline

- The normative code baseline is canonical `dev@b3b4cc4c0fbb1b25140881fb2c41031cb2bc8ad6`.
- The current translation-turn owner accepts final transcripts, creates ordered child translations, owns cancellation and terminality, and submits typed final output. It has no live-revision lane.
- Self and Peer capture have separate production owners. The provider runtime, output runtime, settings persistence, application shutdown, and UI-facing application boundary are established production owners.
- The output runtime owns destination eligibility, overlay delivery, Self chatbox publication, unconditional Peer chatbox denial, replacement, cancellation, and diagnostics.
- The generic LLM boundary and fallback/concurrency wrappers translate complete requests. They do not expose a reviewed live-run, assistant-prefill, response-normalization, or observed-provider-conformance contract.
- Canonical vNext settings use schema version 31 with legacy compatibility projection and backup-before-forward-migration behavior.
- The production desktop UI at this baseline uses Flet 0.28.3. Any later accepted UI-line change must be reconciled without weakening this contract or treating experimental or reverted UI as the design baseline.

## User-visible surfaces

- Self microphone capture, transcription, translation, desktop/native overlay presentation, optional Self VRChat chatbox publication, and history/context behavior.
- Peer audio capture, automatic or manual source-language behavior, transcription, translation, overlay presentation, and unconditional chatbox denial.
- Translation mode selection, availability and degraded-state presentation, settings persistence, restart behavior, and localized user feedback.
- Manual text translation, which remains final-only.

## Actual product entrypoints

- The Windows desktop product enters through `puripuly_heart.main.main`, whose GUI path launches the Flet application and delegates to `puripuly_heart.ui.app.main_gui`.
- The current production composition constructs the UI application boundary and the established capture, translation, provider, output, settings, and shutdown owners through the application/controller/orchestrator wiring.
- Desktop overlay and native VR overlay delivery remain established output destinations; the native overlay protocol and startup contract are unchanged unless direct evidence proves an explicitly reviewed compatible extension.

## Platform and environment

- Supported product platform: Windows x64.
- Supported packaged runtime: Python 3.12 and the established PyInstaller composition.
- Focused automated verification uses the repository `.venv` on Windows.
- Platform-sensitive implementation acceptance uses the official Windows GUI entrypoint and real owner wiring with deterministic offline source/provider/output fixtures and isolated destinations. Real-provider and representative-audio source/packaged production acceptance belongs to the successor validation contract.

## Compatibility baseline

- `TURN_BASED` behavior, final prompts, context policy, provider aliases, fallback availability, output routing, and history semantics remain compatible.
- Fast Translation remains fixed on and is not redefined as simultaneous translation.
- Context remains integrated-preferred with the established local fallback behavior.
- Existing settings files and keys remain loadable and round-trippable. Forward migration creates a backup before mutation and fails safely.
- Credentials remain compatible with existing SecretStore keys and are loaded through SecretStore. Encrypted-file storage still requires `PURIPULY_HEART_SECRETS_PASSPHRASE`.
- Self, Peer, and System remain separate product channels. Peer utterances never route to the VRChat chatbox.
- Broker `/v1`, native overlay, installer, executable identity, provider alias, and prompt fallback contracts remain unchanged.

# Scope

## Included

- A persisted `TURN_BASED` or `SIMULTANEOUS` translation-interaction mode captured at turn open and applied to that complete logical turn.
- A live-revision lane for eligible Self and Peer speech turns with immutable context identity, bounded prefix reuse, at most two live logical flights, latest-pending coalescing, semantic result classification, and authoritative final replacement.
- Explicit separation of canonical draft, display draft, verified prefix, provisional prefix, speculative tail, provider branch, logical flight, source final, and translation final.
- Provider capability and conformance representation, safe prefill normalization, priority admission, final capacity reservation, cancellation semantics, degradation, and diagnostics.
- Application-boundary state and intent for the interaction mode, localized availability/degraded state, compatible settings migration, and inert debug-preview coverage.
- Deterministic temporal, ordering, fault, property, persistence, architecture, lifecycle, and offline production-entrypoint wiring evidence sufficient to establish a disabled implementation candidate. Real-provider, representative-audio, packaged end-to-end, UX, performance, cost, and activation evidence belongs to the successor validation contract.

## Non-goals

### NG-001 — Product activation

This contract does not authorize developer preview, limited preview, general availability, deployment, release, or enabling a real provider route for live product traffic.

### NG-002 — Live chatbox output

No live, provisional, alternate, failed, or source-final-only translation is sent to the VRChat chatbox.

### NG-003 — Learned or language-specific commitment

No SimulMT training, LoRA/DoRA, survival predictor, target-language grammar engine, absolute log-probability threshold, or per-language syntax rule is required.

### NG-004 — Unbounded partial translation

The product does not translate every ASR partial, admit unbounded parallel requests, or assume provider completions arrive in request order.

### NG-005 — Long-form segmentation

Synthetic seven-second finals, soft translation segments, chunked final authority, and very-long-turn context/output redesign remain outside this contract.

### NG-006 — Expanded live coverage

Manual translation remains final-only. Multi-target live output, mixed-language live output, and live output without a stable Peer language hint remain outside the initial implementation contract.

### NG-007 — Research replication

Beam search, RALCP, n-best production decoding, model-internal attention scheduling, and full MLLP-VRAIN replication are not required.

# Requirements

## REQ-I001 — Interaction modes and turn snapshot

`TURN_BASED` preserves the established final-only path and creates no live actor or live flight. `SIMULTANEOUS` may create live work only after the automatic eligibility gate. A mode change affects the next opened turn and never changes the mode of an active turn. Manual text remains on the established final-only path.

## REQ-I002 — Source lifecycle and immutable context

Each live-capable speech turn has a stable logical identity, monotonically ordered source revisions, explicit source epoch, language epoch, context epoch, route epoch, and terminal generation. Partial source revisions never enter completed conversation history. Source final commits the completed source turn exactly once even when translation fails. Every live and final request for a turn uses a reproducible immutable context lease built only from previously completed logical turns, and the authoritative final lease excludes its own source.

## REQ-I003 — Live and final authority

Live output is mutable, overlay-only, non-authoritative, and excluded from final history and chatbox publication. Source final is not translation final and never promotes the current live draft. Finalization stops new live admission, gives the authoritative final priority over live or audit work, and independently translates the complete source without assistant prefill or live prompt state. An accepted final atomically replaces live presentation and cannot be overwritten by a late live result.

## REQ-I004 — Source stabilization and automatic admission

The engine distinguishes exact repeats, clean or soft appends, tail rewrites, mid-turn rewrites, language changes, context changes, and completed source. Repeats create no work; append-like revisions may preserve compatible frontier state; tail rewrites shrink or audit provisional state; mid-turn, language, context, or route changes reset incompatible state. Short turns may finish without a live request. Live admission requires meaningful source growth, dwell/refresh eligibility, remaining turn budget, provider availability, and non-finalizing state.

## REQ-I005 — Reversible target boundary

All frontier operations use a versioned reversible target-unit boundary that can reconstruct the exact original target string and never slices inside an invalid Unicode or provider-required boundary. Unknown or incompatible normalization degrades safely rather than creating evidence.

## REQ-I006 — Bounded dual frontier

The canonical live draft used for future frontier calculations is separate from the latest display draft. A verified prefix may advance only through exact agreement among eligible no-prefill free or audit generations with compatible source, context, route, language, normalizer, and unitizer identity. A provisional prefix may use compatible regeneration overlap but must remain a prefix of the canonical draft, contain the verified prefix, stay within a finite configured lead over verified, and expire by accepted-generation age or wall-clock age. Forced prefill text never counts as independent verification. Expiry, mismatch, rewrite, or incompatible lineage rolls back or audits provisional state. Display-only alternate results never become canonical draft, frontier evidence, or future prefill.

## REQ-I007 — Immutable prefill lineage and response normalization

Every turn captures one immutable policy-profile ID and fingerprint, and every prefilled run captures an immutable snapshot of that profile together with the exact canonical draft, verified and provisional boundaries, actual forced prefix, overlap reserve, context, route, language, source-unit metric, normalizer, target unitizer, prompt, and generation parameters used at admission. A configuration change applies only to a subsequently opened turn and never reinterprets an admitted request. A prefilled result may affect canonical or provisional state only when that snapshot remains compatible at completion. Provider responses are normalized from an explicitly conformed suffix-only or full-echo shape; unknown, ignored, duplicated, or incompatible shapes cannot become frontier evidence and degrade to a safe no-prefill policy.

## REQ-I008 — Adaptive two-flight scheduling

Normal operation admits one live logical flight. A second live logical flight is permitted only after a configured freshness or first-draft/audit condition, meaningful newer source or explicitly allowed same-source escape, available budget and capacity, cooldown, and non-finalizing state. Each turn has at most two live logical flights and one coalesced latest pending source snapshot. Provider fallback branches are counted separately from logical flights. Duplicate same-source/hash flights are denied unless an explicit bounded escape policy applies. Cancellation is an optimization rather than a correctness dependency.

## REQ-I009 — Semantic result disposition

Every completion is structurally checked before state mutation and is classified as display-and-evidence, display-only, evidence-only, or metrics-only. Display freshness, evidence eligibility, and canonical promotion are independent decisions. Older eligible no-prefill results may contribute evidence within a bounded horizon without regressing the screen. Older prefilled results cannot mutate provisional state. Results from an old context, route, language, terminal generation, rolled-back snapshot, closed turn, or post-final live phase cannot resurrect or mutate the turn.

## REQ-I010 — Provider execution and final capacity

Requested route and observed provider provenance are distinct. Missing observed provenance remains unknown and is never synthesized. Live prefill is disabled unless the exact enabled model/route has evidence for request acceptance and response shape. Live automatic provider racing is off by default; an emergency alternate may be displayed but remains non-canonical. The global admission policy reserves the capacity required for authoritative final work, preempts or suppresses live work when needed, and uses sequential final fallback when total branch capacity is one. Provider teardown awaits `close()`.

## REQ-I011 — Lifecycle ownership and serialization

One semantic translation owner remains authoritative for turn and live meaning. Each live turn serializes all accepted source events, completions, rollback, finalization, and closure through one state owner. Capture owners emit source/turn facts, provider runtime owns external execution and resources, output runtime owns destination routing and delivery, settings persistence owns serialization, shutdown coordination owns terminal admission and teardown, and the UI owns only localized intent and presentation. Every long-running task has explicit ownership, cancellation, shutdown, bounded wait, diagnostics, and rejection of work after close.

## REQ-I012 — Semantic presentation ordering

Overlay transport sequence never substitutes for semantic ordering. Presentation compares turn identity, source epoch and revision, context/route/language identity, live/final phase, and terminal generation explicitly. A terminal tombstone prevents late resurrection after final, close, eviction, context clear, language change, provider change, toggle-off, shutdown, or restart. A mode-setting change alone does not terminate or suppress an active turn; its snapshotted mode remains authoritative until that turn closes, and the pending mode becomes effective only at the next turn boundary. Same-revision replacement is permitted only by explicit disposition policy. Masking of an unstable tail, if implemented, affects presentation only and never canonical text or evidence.

## REQ-I013 — Settings, application boundary, and UI compatibility

The interaction mode persists with a default of `TURN_BASED`; old settings remain valid and round-trippable, schema changes are sequential, and forward migration backs up before mutation. The UI sends a mode intent and consumes localized mode, availability, degraded, finalizing, and failure state through the established application boundary. It does not read or control frontier, scheduler, provider, or actor internals. All user-facing text is localized with all supported locale bundles in parity. Async Flet callbacks use `page.run_task`. Debug-preview states remain hidden without the explicit debug flag and cannot persist settings, mutate secrets, or call providers, Broker, audio, chatbox, or overlays.

## REQ-I014 — Diagnostics and privacy

Diagnostics expose stable identities and bounded state needed to reconstruct admission, flight/branch counts, result disposition, frontier provenance, rollback, final priority, shutdown, policy-profile ID/fingerprint, source-unit/normalizer/target-unitizer/prompt versions, and exact generation parameters. Default logs and exported diagnostics contain no raw source, raw translation, credentials, headers, request bodies, secrets, or fabricated cost/provenance. Raw trace capture is explicit opt-in, sanitized, and separately retained. Local cancellation and remote generation or billing are reported as distinct facts.

## REQ-I015 — Disabled implementation boundary

Implementation completion leaves simultaneous product traffic disabled by default, real live provider routes unenabled, and `TURN_BASED` as the product default. The implementation exposes safe deterministic fixtures and the versioned policy/capability mechanisms required by the successor validation contract. No implementation result alone claims latency, cost, quality, provider conformance, limited-preview readiness, or general-availability readiness.

# Normative State and Transition Algorithms

The names in this section describe durable semantic state and events. Private type names and file placement may differ, but the same ordered events and inputs must produce the same safety-relevant transition, disposition, and output authority.

## ALG-I001 — Turn events and causal identity

A live-capable turn processes the following immutable event meanings through one serialized actor queue:

1. `TurnOpened` establishes logical turn, speech session, channel, start time, and the interaction-mode snapshot.
2. `SourceRevision` carries monotonic revision, source epoch, raw text and hash, comparison hash, creation time, optional provider revision, stable boundary, detected language hint, and language epoch.
3. `SegmentFinalized` reports only an STT/internal segment boundary and never starts authoritative translation finalization.
4. `TurnFinalized` carries the complete final source and is the only source event that starts authoritative finalization.
5. `RunCompleted` carries the original causal ticket and provider result back to the actor; the completion callback cannot mutate presentation or frontier state directly.

Within one logical turn and source epoch, revision numbers are strictly increasing. Text length, timestamps, and provider revision IDs never substitute for revision ordering. When revision `R(n+1)` is accepted, the actor records the time at which `R(n)` became superseded; future superseded time is never placed in an immutable request ticket.

At `TurnOpened`, the turn also captures one immutable policy-profile ID and fingerprint covering source-safe unit metric/version, source normalizer, target unitizer, live prompt, live generation parameters, gate, frontier, audit, scheduler, capacity, freshness, masking, and provider-route policy. A policy/configuration change applies only to the next opened turn unless it is an explicit destructive provider/context/language event already defined by this contract; outstanding work is never reinterpreted under a new profile.

Every run ticket fixes logical flight and turn, channel, run kind, source revision/epoch/hash/snapshot time, policy-profile ID/fingerprint, source-normalizer version, context epoch/hash, frontier snapshot/epoch/version and prefill hash/count, provider generation, route epoch, language epoch, prompt version, unitizer version, exact generation parameters, issue time, and soft/hard deadlines. A completion is interpreted only through that ticket and current actor state.

The only allowed phase transitions are:

```text
COLLECTING -> LIVE
COLLECTING -> FINALIZING
LIVE -> FINALIZING
COLLECTING or LIVE -> CANCELLED
FINALIZING -> FINAL
FINAL -> CLOSED
CANCELLED -> CLOSED
```

There is no reverse transition. Request ID, logical-flight ID, event epoch, source-normalizer version, and terminal generation make `RunCompleted`, `TurnFinalized`, `TurnCancelled`, `ContextChanged`, and `ProviderChanged` idempotent. The first accepted completion may transition state; a duplicate completion is metrics-only. Repeated finalization commits source and admits authoritative final exactly once. Context/provider changes after closed are diagnostic-only. Cancellation cannot reopen finalizing, final, closed, or cancelled state.

## ALG-I002 — Source normalization and classification

The provider always receives the complete raw source text. A separately versioned comparison form is used only for revision classification and may normalize line endings, outer/repeated whitespace, an explicitly pinned Unicode form, and provider/language-neutral punctuation or casing differences proven by fixtures. It cannot discard meaning-bearing characters.

Destructive metadata is processed before text equality. A retreating accepted stable boundary, an accepted stable-language change, or a source-normalizer version change first performs its required destructive transition even when raw/comparison hashes are equal. When no destructive metadata change exists, text classification uses this exact precedence:

1. Equal raw hash or equal comparison hash: `EXACT_REPEAT`.
2. New raw text has the previous raw text as an exact prefix: `CLEAN_APPEND`.
3. The previous comparison text is preserved as a prefix and raw differences are limited to configured presentation normalization: `SOFT_APPEND`.
4. The strict raw longest common prefix ends within a bounded configured tail window and remains at or beyond a configured stable-prefix floor: `TAIL_REWRITE`.
5. All other meaningful changes, a retreating provider stable boundary, a meaningful earlier normalized change, or a destructive language change: `MID_REWRITE`.

The corresponding transition is:

| Revision kind | Verified | Provisional | Canonical/display | Next critical policy |
|---|---|---|---|---|
| `EXACT_REPEAT` | unchanged | unchanged | unchanged | no request; retain or deduplicate pending hash |
| `CLEAN_APPEND` | retained | retained if still compatible | retained until a result | normal free/prefilled admission with base overlap |
| `SOFT_APPEND` | retained | retained if still compatible | retained until a result | normal admission with optionally enlarged bounded overlap; never evidence of target stability by itself |
| `TAIL_REWRITE` | retained only as an unaligned candidate | clamp to verified by default | prior display may remain while canonical awaits a result | mark audit due; use no-prefill free or verified-only prefill |
| `MID_REWRITE` | reset | reset | canonical reset; prior display cannot become evidence | increment source and frontier epochs, obsolete pending/in-flight live work, issue latest-source free when eligible |

If an STT provider exposes a stable prefix, its unit and meaning must be adapter-defined; retreat is destructive. Without such a boundary, admission uses raw/comparison longest common prefix plus revision dwell time. A stable-boundary increase, safe raw append of the configured minimum, tail/mid rewrite, changed hash at a pause checkpoint, or maximum refresh interval constitutes a meaningful delta. Translation never infers capture-private stability.

A stable source-language change increments language epoch, recomputes route and prompt language fields, destructively resets frontier state, obsoletes pending/in-flight live work, and makes the next eligible run no-prefill free. A transient low-confidence hint is not accepted as a language change unless the capture boundary marks it stable.

## ALG-I003 — Context and source-final transaction

At turn open or first translation admission, the actor freezes one context lease containing conversation ID, context epoch and hash, rendered bytes, applied local/integrated mode, included completed turn IDs, renderer version, and creation time. Rendering is byte-identical for the lease lifetime and excludes relative-time or mutable diagnostic content.

Partial revisions never write conversation history. One global conversation-ledger operation serializes concurrent Self and Peer source finals. Under one ledger lock or single owner it atomically deduplicates logical turn ID, builds the final self-excluding lease from turns committed before this admission, assigns deterministic sequence, commits source exactly once, increments context epoch, and returns whether the turn was newly committed. Concurrent final order is ledger-operation admission order, not wall-clock timestamp. Outside the ledger operation, the new context epoch is broadcast to other active actors, authoritative final uses the returned frozen lease, and a successful final translation may attach to the already committed turn. Provider failure never removes committed source.

Clear, context, language, route, prompt, source-normalizer, or unitizer changes increment the applicable epoch and invalidate incompatible live results; no completion may be reinterpreted under a newer lease. A concurrent committed turn causes every other collecting/live actor to obsolete old-context work, increment frontier epoch, clear verified/provisional/canonical state, optionally retain the old display as stale-only, obtain a new lease, and make its next eligible request no-prefill free. A finalizing or final actor retains the final lease returned by its own atomic ledger operation and does not restart or invalidate its authoritative-final admission; only remaining live work is obsolete and rejected.

## ALG-I004 — Automatic live gate and request budget

The first live request is always no-prefill free. The gate is evaluated only in `SIMULTANEOUS` and admits work only when the turn is collecting/live and not finalized, raw source is non-empty and not a repeat, a safe live provider policy exists, meaningful/dwell conditions pass, and turn/global budgets remain. A turn finalized before the first gate creates no live flight.

Initial live coverage is exact: Self speech with exactly one active target may be eligible; Peer speech with an explicit manual source language and exactly one active target may be eligible; automatic-language Peer additionally requires a stable capture-owned language hint; manual text, multiple live targets, mixed-language source, missing/unstable automatic-language hint, or unavailable safe route admits zero live work. Those ineligible cases retain the established authoritative final-only path or established source-only degraded behavior rather than inventing a live language/target choice.

After the first result, refresh is considered only for meaningful source delta, pause checkpoint, tail/mid rewrite, maximum refresh interval, context/provider reset, or a pending-latest revision after capacity opens. Minimum request interval and second-flight soft deadline are distinct controls.

The turn budget counts free, prefilled, audit, and escape logical flights but not authoritative-final provider branches. When exhausted, new live/audit work stops, the latest display may remain, and final is still attempted. With scarce remaining budget, priority is latest critical source, context/reset free, required audit, optional display refresh, then diagnostics. If an audit is due but cannot be admitted, actual prefill is reduced to verified state rather than allowing unaudited provisional advance.

## ALG-I005 — Frontier state and invariant relations

Each turn maintains:

- `verified`: exact prefix independently supported by eligible no-prefill free/audit generations;
- `provisional`: canonical prefix allowed ahead of verified only within configured lead and support lifetime;
- `canonical_draft`: latest accepted canonical-route result used for frontier calculation and future prefill;
- `display_draft`: latest user-visible result, which may be alternate or display-only;
- frontier epoch/version, canonical/display generations, immutable snapshot registry, no-prefill hypotheses keyed by source revision, provisional support generation/time, cumulative forced units since audit, accepted prefilled runs since audit, and the source-normalizer version participating in the lineage fingerprint.

At all accepted states:

```text
verified is an exact target-unit prefix of provisional
provisional is an exact target-unit prefix of canonical_draft
units(provisional) - units(verified) <= W
display_draft may differ from canonical_draft
```

Strict longest-common-prefix operations compare exact reversible target units. Case folding, punctuation removal, fuzzy similarity, semantic equivalence, and stabilized-LCP heuristics cannot advance verified. A destructive reset or rollback increments frontier epoch and invalidates every older snapshot lineage; ordinary compatible advance increments frontier version. Authoritative final never reads frontier state.

## ALG-I006 — Bootstrap and verified recomputation

On the first eligible canonical no-prefill result `F1`:

```text
canonical_draft := F1.complete_text
display_draft := F1.complete_text only if display disposition permits
verified := empty
provisional := unit-safe prefix(remove_guard_tail(F1), bootstrap_cap)
```

The bootstrap prefix is unverified and immediately subject to lead, generation-age, wall-time, audit-debt, and reset rules. A conservative profile may set the cap to zero.

A no-prefill free, audit, or escape result enters the evidence ledger only when logical turn, source/context/language/route epochs, context hash, provider fingerprint, source-normalizer version, prompt version, unitizer version, canonical-route eligibility, normalized complete output, no-prefill status, evidence-eligible provenance, and evidence horizon are compatible.

Eligible hypotheses are ordered by source revision, never arrival time. When at least two distinct revisions exist, verified candidate is:

```text
candidate := remove_verified_guard_tail(
    strict_target_unit_lcp(latest_eligible_revision_minus_one, latest_eligible_revision)
)
```

If candidate extends verified, verified advances and frontier version increments. If it is shorter or incompatible, verified rolls back, provisional clamps to compatible supported text, frontier epoch and version increment, and older snapshots become unusable. Verified advance never automatically promotes the rest of canonical draft; provisional retains only previously supported or independently supported exact span within `verified + W`.

For every later no-prefill free/audit/escape completion, transition order is deterministic: store eligible evidence and immediately recompute verified; update display only when disposition permits; then, only for canonical-route `DISPLAY_AND_EVIDENCE`, replace canonical draft with complete text and compute provisional. The later-free provisional end is the greatest of verified units and the unit count of the guarded complete text, capped by complete-text length and `units(verified) + W`; it therefore uses only independently generated complete text and never forced input. For the first canonical no-prefill result, the additional bootstrap cap applies. For a noncanonical or display-only result, canonical and provisional do not change. Every canonical replacement increments canonical generation and frontier version, and records fresh provisional support generation/time only for the exact accepted provisional span.

## ALG-I007 — Immutable prefill construction

Immediately before a prefilled request, the actor stores an immutable snapshot of frontier epoch/version, verified, provisional, canonical draft, unit counts, context/source/language/route epochs, provider generation/fingerprint, policy-profile ID/fingerprint, source-normalizer, prompt and unitizer versions, exact generation parameters, and creation time. A snapshot referenced by outstanding work cannot be deleted; after destructive reset it remains diagnostic history but is lineage-incompatible.

Prefill is calculated only from that snapshot:

```text
overlap := bounded_overlap(
    base_overlap,
    minimum_rewrite_reserve,
    latest_source_kind,
    mismatch_history,
    optional_rank_signal
)
overlap >= minimum_rewrite_reserve
prefill_end := max(0, units(snapshot.provisional) - overlap)
assistant_prefill := unit_safe_prefix(snapshot.provisional, prefill_end)
```

`assistant_prefill` must be an exact unit-safe prefix of snapshot provisional. `remove_guard_tail(text, guard)` and `remove_verified_guard_tail(text, guard)` are deterministic versioned transformations returning the longest unit-safe prefix with exactly `min(guard, units(text))` trailing units removed. `bounded_overlap` starts from the exact revision-kind overlap in the active profile, adds only the profile-defined mismatch/rank increments, clamps between the profile minimum rewrite reserve and available provisional units, and returns an integer unit count. The disabled seed profile below defines every input and bound. If prefill unit count is below the configured economical threshold, the run changes to no-prefill free and carries no assistant prefix. This economy gate changes cost/latency policy only and never correctness.

The provider request uses the same context lease and current raw source as the snapshot lineage. Its live prompt is separately versioned, shares the established translation tone/context rules, states that the source is incomplete, forbids inventing future negation/conditions/conclusions/intent, permits a fragmentary target, preserves already expressed tone/emotion/honorifics, and emits translation only. Prompt-version change destructively resets frontier lineage.

## ALG-I008 — Prefill response normalization

Conformance identifies an exact route response as `SUFFIX_ONLY`, `FULL_ECHO`, `PARTIAL_ECHO`, or `UNKNOWN`. The adapter must produce both complete target text including prefill and the suffix actually generated by this call.

For suffix-only output, complete text is exact request prefill concatenated with the returned suffix. For full echo, the returned text must begin with the exact request prefill at a valid unit boundary and generated suffix is the remaining text. Partial echo, ambiguous echo, silently rewritten prefill, incompatible whitespace/Unicode boundary, truncation represented as complete, or an output that does not exactly begin with the request prefix is not conformed.

An unconformed prefilled result is a structural rejection and is `METRICS_ONLY`: it cannot affect display, canonical, verified, provisional, or the free ledger. The route becomes prefill-degraded and the next critical run is no-prefill free. A provider that may silently ignore prefill cannot be enabled for prefilled live work without exact conformance evidence.

## ALG-I009 — Prefilled lineage, regenerated support, rollback, and audit

A prefilled result can support provisional state only when its frontier epoch matches current state; snapshot exists; ticket prefill hash/count match the snapshot-derived prefix; current verified and provisional still contain that prefix; context, source, language, route, provider generation, policy-profile fingerprint, source-normalizer, prompt, unitizer, and generation-parameter identities match; provenance is canonical/evidence-eligible; and its source revision is newer than the provisional-application watermark. A later compatible frontier version may be accepted only if the original prefill remains an exact current prefix; any frontier-epoch or policy-profile change rejects it.

Forced prefill is never evidence. Support begins only after the actual prefill boundary:

```text
expected := snapshot.canonical_draft after assistant_prefill
observed := normalized generated_suffix
supported_suffix := strict_target_unit_lcp(expected, observed)
supported_end := units(assistant_prefill) + units(supported_suffix)
candidate_end := min(
    supported_end,
    units(current.verified) + W,
    units(result.complete_text)
)
candidate := unit_safe_prefix(result.complete_text, candidate_end)
```

The candidate must contain current verified exactly. On match, complete text becomes canonical draft, canonical generation increments, candidate becomes provisional, support generation/time update, and frontier version increments. Display updates separately through result disposition.

On ordinary overlap mismatch, unverified extension is removed and provisional is no longer than the greater of verified and the still-compatible actual prefill; audit becomes due. If audit or source rewrite contradicts the prefill itself, provisional clamps to verified. A destructive source/context/language/route/source-normalizer/prompt/unitizer change clears verified, provisional, and canonical and increments frontier epoch. A contradictory no-prefill audit recomputes verified from eligible exact agreement, clamps provisional to verified plus exact supported extension, and increments frontier epoch.

Provisional expires when either accepted canonical generations since last support reach generation TTL or wall time since support reaches wall-time TTL. Expiry or any due audit blocks actual prefill beyond verified until an eligible no-prefill audit is accepted. Critical latest-source work may still outrank scheduling the audit, but that critical run must be no-prefill free or use only a unit-safe verified prefix. If no audit budget exists, actual prefill remains clamped to verified.

Audit debt is separate from current provisional lead. Each accepted canonical prefilled run adds its forced prefix unit count and increments the accepted-prefilled-run count. Audit becomes due on configured run cadence, cumulative forced-unit debt, provisional expiry, overlap mismatch, tail rewrite, context boundary, or optional rank risk. An accepted no-prefill audit resets both debt counters. Optional log-probability/rank data may enlarge overlap, pause provisional advance, accelerate audit, or support diagnostics; it can never verify text or count forced prefix as generated evidence.

## ALG-I010 — Two-flight admission and coalescing

The scheduler maintains in-flight live logical flights, one `pending_latest`, last issue times, live logical-flight and provider-branch counts, cooldown, finalizing, and closed state. A new non-repeat revision either starts an immediately eligible flight or replaces `pending_latest`; intermediate pending revisions are deliberately coalesced because authoritative final uses the complete final source separately.

The first slot requires simultaneous mode, open live gate, live phase, zero current live flights, meaningful source, remaining logical-flight budget, and provider-branch capacity. Run-kind priority is: audit only when due and critical display is fresh; otherwise compatible economical prefilled; otherwise free. Critical latest-source freshness outranks an older-source audit.

A second flight is admitted only when all are true:

1. Exactly one live logical flight is in flight.
2. Its route/run-kind soft deadline has elapsed; soft deadline is an admission checkpoint, not failure.
3. A pending latest revision exists or bounded first-draft escape applies.
4. The new revision is meaningfully newer, or an explicitly allowed no-draft same-source escape applies.
5. No flight already has the same source hash and run kind.
6. Phase is not finalizing, final, closed, or cancelled.
7. Turn logical-flight, global provider-branch, rate, and cost budgets allow it.
8. Second-flight cooldown has elapsed.

Second-slot priority is latest-source hedge, context/reset free, audit sibling, then diagnostics. A latest-source sibling uses the most recent accepted snapshot available at its own issue time and never waits for or descends from the first flight; without compatible snapshot/provenance it is escape-free. Same-source duplicate is allowed only when there is no draft, source has stopped, the first flight is a hard-tail candidate, provider policy permits duplication, and cost budget permits it. While two flights run, new revisions only replace `pending_latest`.

Hard deadline is a resource boundary, not a correctness boundary. The actor removes the matching logical flight from the live in-flight set, records its request/flight tombstone as hard-deadline obsolete, frees the logical slot, and immediately reconsiders pending latest under normal gate, cooldown, budget, and capacity rules. Local cancellation and transport abort are attempted. Each provider branch remains counted as active capacity until its local task reports terminal or the bounded branch-cleanup deadline moves it to a separate cancellation-debt registry. Cancellation debt has a configured count/age bound; reaching it disables new live admission while preserving reserved authoritative-final admission. Remote work and billing remain unknown. A late branch completion returns through the tombstone-aware classifier and cannot consume or remove a newer logical flight.

## ALG-I011 — Completion classification and watermarks

Every completion first validates that its request and logical-flight identities belong to this turn and match the currently active flight or an exact retired-flight tombstone. Only a matching active flight is removed and terminally accounted. A foreign, malformed, or duplicate completion leaves all active-flight and capacity state unchanged and is metrics-only. A completion for a known retired tombstone performs only its allowed provider-branch/debt terminal accounting before classification and cannot retire a newer flight. Logical-turn mismatch; final/closed/cancelled phase; live result after finalizing; context/source/language/route/provider-generation mismatch; source-normalizer, prompt, or unitizer mismatch; incompatible prefilled snapshot; or failed prefill normalization produces `METRICS_ONLY`.

Presentation, provisional, free-evidence, and terminal watermarks are separate. A result from a lower source revision than the display watermark cannot display. An equal-revision replacement requires explicit same-revision policy and a higher presentation generation. A newer result whose source was already superseded must be within the configured no-draft or with-draft superseded-age window; a result whose source hash is still latest has zero superseded age regardless of request age.

Disposition then follows:

| Disposition | Preconditions | State effects |
|---|---|---|
| `DISPLAY_AND_EVIDENCE` | structural lineage valid; display revision/freshness valid; canonical/evidence-eligible provenance; run kind/snapshot eligible | may update display and the eligible canonical/free/provisional state |
| `DISPLAY_ONLY` | display is useful but route/provenance/snapshot is not evidence-eligible | update display only; canonical, verified, provisional, and free ledger unchanged |
| `EVIDENCE_ONLY` | older eligible no-prefill free/audit/escape result within evidence horizon, while display is newer | store no-prefill evidence and immediately run ALG-I006 verified recomputation/rollback; display and provisional-application watermark remain unchanged |
| `METRICS_ONLY` | structural rejection, obsolete/incompatible prefilled result, or no valid display/evidence use | record latency, cost, inversion, cancellation, and rejection reason only |

An older prefilled result is never evidence-only for provisional advance. For the same source revision, canonical evidence eligibility outranks run purpose, known normalization outranks completion time, and an alternate result arriving first can be display-only before a later canonical replacement. Canonical-to-canonical paraphrase replacement at the same revision is denied unless an explicit reviewed policy permits it; corrective audit replacement requires its configured mismatch condition.

After disposition, if phase is live and a slot is open, issue a newer pending critical run, otherwise a due audit, otherwise remain idle. A pending source whose hash is already in flight is not reissued.

## ALG-I012 — Finalization and global capacity

On `TurnFinalized`, the actor atomically enters finalizing, increments terminal generation, clears pending latest, closes all new live/audit admission, marks in-flight live work logically obsolete, attempts cancellation/abort, and requests authoritative final at highest priority. Every later live completion becomes metrics-only because of phase/terminal generation regardless of cancellation success.

Global admission counts provider branches separately from logical flights and reserves configured authoritative-final branch width. With capacity at least final branch width, final primary and permitted fallback capacity are protected from live work. With capacity one, live work is preempted, final primary runs first, and fallback runs sequentially only after primary failure or timeout. Zero capacity is configuration failure. Preemption order is diagnostics, audit, alternate display-only, then oldest obsolete live. Correctness depends on final admission and late-result rejection, not successful remote cancellation.

## ALG-I013 — Reset and terminal matrix

| Event | Frontier action | Outstanding live action | Presentation action |
|---|---|---|---|
| exact/clean append | retain compatible frontier | continue; coalesce latest | retain until newer accepted result |
| soft append | retain compatible frontier; optionally enlarge bounded overlap | continue; coalesce latest | retain |
| tail rewrite | mark audit due; provisional clamps toward verified | older prefilled lineage restricted or obsolete | prior display may remain as mutable/stale until replacement |
| mid rewrite | clear verified/provisional/canonical; increment source/frontier epochs | obsolete all older live work | prior display cannot become canonical/evidence |
| context, route/provider, language, source-normalizer, prompt, or unitizer epoch/version change | destructive reset and epoch increment | obsolete incompatible pending/in-flight work | retain only where explicit display policy permits; never evidence |
| turn finalized | freeze then retire live frontier from authority | close live admission; obsolete/cancel; admit final | live may remain visually only until independent final/failure state |
| final/closed/evicted | no live mutation allowed | reject/metrics-only | terminal tombstone prevents resurrection |

## Characterization seed profile

The disabled implementation must support a versioned profile initialized from source r3 so deterministic fixtures can exercise the intended control shape. These values remain validation candidates and do not establish product performance:

| Policy | Source-r3 seed |
|---|---:|
| profile ID | `simul-disabled-seed-v1` |
| profile fingerprint | SHA-256 over canonical serialized profile |
| source-safe unit metric/version | Unicode extended-grapheme clusters / `simul-source-units-v1` |
| source normalizer version | `simul-source-normalizer-v1` |
| target unitizer version | Unicode extended-grapheme-safe / `simul-target-units-v1` |
| live prompt version | `simul-live-prompt-v1` |
| live temperature | `0.0` |
| live maximum new target units | 100 |
| optional top-logprobs | disabled |
| first live delay | 1700 ms |
| minimum source | 10 source-safe units |
| meaningful source delta | 3 source-safe units |
| minimum request interval | 750 ms |
| tail rewrite window | 12 source-safe units |
| stable-prefix floor | 8 source-safe units |
| maximum live logical flights started per turn | 10 |
| bootstrap provisional cap | 8 target units |
| bootstrap guard tail | 2 target units |
| verified guard tail | 2 target units |
| provisional lead `W` | 16 target units |
| base regeneration overlap | 6 target units |
| soft-append overlap | 8 target units |
| tail-rewrite overlap | 12 target units |
| mismatch overlap increment/cap | add 2 target units, cap at 12 |
| optional rank-risk overlap increment/cap | add 2 target units, cap at 12 |
| minimum rewrite reserve | 2 target units |
| minimum economical prefill | 8 target units |
| provisional generation TTL | 2 accepted canonical generations |
| provisional wall-time TTL | 4.0 seconds |
| audit cadence | 2 accepted prefilled runs |
| forced-unit audit debt | 20 cumulative target units |
| free-evidence horizon | latest 4 revisions or 8 seconds |
| soft deadline with insufficient samples | 1000 ms |
| measured soft deadline | route/run-kind p80 clamped to 800–1300 ms |
| live hard deadline | 5000 ms |
| provider-branch local cleanup deadline after hard deadline | 1000 ms |
| maximum cancellation-debt branches per turn/global | 2 / 8 |
| cancellation-debt diagnostic retention | 30 seconds |
| second-flight cooldown | 750 ms |
| superseded display age with no draft | at most 2000 ms |
| superseded display age with an existing draft | at most 1000 ms |
| optional presentation mask tail | 2 target units |

# Protected Invariants

## Product invariants

### INV-P-I001 — Peer chatbox denial

Peer utterances never route to the VRChat chatbox in any mode, state, fallback, error, cancellation, replacement, finalization, shutdown, or restart path.

### INV-P-I002 — Channel separation

Self, Peer, and System outputs remain distinct product channels and cannot be relabeled through live processing.

### INV-P-I003 — Authoritative final independence

The authoritative final uses the complete source and frozen final context lease with no live prefill, live prompt, display draft, or provisional frontier state.

### INV-P-I004 — Turn-based compatibility

With `TURN_BASED` selected or simultaneous availability disabled, established final translation, history, overlay, Self chatbox, Peer denial, prompt, provider, fallback, and lifecycle behavior is preserved.

### INV-P-I005 — Monotonic terminal presentation

No late or older live result regresses source-revision presentation, overwrites an accepted final, or resurrects a closed or evicted turn.

## Durable architecture invariants

### INV-A-I001 — Owner boundaries

The feature extends the established translation, capture, provider, output, settings, shutdown, and UI-application owners without creating a parallel semantic translation owner or a new multi-responsibility orchestrator.

### INV-A-I002 — Frontier provenance

Verified state is supported only by compatible no-prefill evidence; provisional state is bounded, expiring, lineage-checked, and never self-certified by forced prefill or display-only output.

### INV-A-I003 — Bounded concurrency

Per-turn live logical-flight count, pending source count, global live capacity, final admission delay, cancellation debt, and terminal wait remain explicitly bounded.

### INV-A-I004 — Persistence and credentials

Settings remain round-trippable and backward compatible, forward migration is backed up, and secrets remain exclusively under compatible SecretStore ownership.

### INV-A-I005 — Provider and output separation

Provider adapters do not decide display disposition or output routing, and output delivery does not decide translation evidence or frontier state.

# Approved Decisions

- The feature is a separate `TURN_BASED` or `SIMULTANEOUS` interaction mode, not a reinterpretation of Fast Translation.
- Live output is mutable overlay-only output. Only an authoritative Self final may be eligible for chatbox publication; Peer chatbox publication remains impossible.
- Completed logical source turns are the only conversation-history units. Partial revisions replace one current source turn rather than becoming history.
- The frontier consists of verified, bounded provisional, speculative, and not-yet-generated regions, with separate canonical and display drafts.
- Verified progress requires compatible no-prefill exact agreement. Prefill-conditioned output cannot independently verify forced text.
- One live logical flight is normal and two is the hard turn-local maximum; only the newest pending source is retained.
- Logical flight count and provider branch count remain distinct.
- Live alternate fallback output is display-only by default. Authoritative final fallback preserves established availability behavior and capacity-one sequential execution.
- Live temperature `0.0`, bootstrap cap, frontier lead, TTL, overlap, economical-prefill threshold, deadlines, cooldown, audit cadence, and mask depth from source r3 are characterization candidates, not implementation acceptance guarantees.
- Cancellation never proves remote computation or billing stopped.
- Seven seconds is an operational long-turn hint, not a synthetic final boundary.
- Very-long-turn segmentation is a separate future product decision.

# Open Product Decisions

None for this disabled implementation boundary. Exact live-provider enablement, policy values, quantitative performance targets, mask presentation, finalizing/degraded copy, cost budget, and preview/GA approval are intentionally owned by `PRD-PURIPULY-SIMULTANEOUS-VALIDATION-001` and cannot be decided or implied by this implementation Goal.

# Acceptance Criteria

| AC | Verifies | Evidence class | Required environment | Pass condition |
|---|---|---|---|---|
| AC-I001 | REQ-I001, INV-P-I004 | automated + persistence | Windows `.venv`; canonical settings repository and production-owner composition | `TURN_BASED` creates zero live actors/flights; `SIMULTANEOUS` intent while availability is disabled also creates zero live actors/flights and preserves the complete established final-only translation, history, overlay, Self chatbox, Peer denial, prompt, provider, fallback, and lifecycle behavior; a mode or policy-profile change applies only to the next opened turn; old settings load and round-trip; restart default remains `TURN_BASED`. |
| AC-I002 | REQ-I002, REQ-I003, ALG-I001, ALG-I003, INV-P-I003 | automated property + temporal fault injection | deterministic production-owner composition with concurrent Self/Peer final and delayed-provider fixtures | Partial history writes are zero; source final commits exactly once despite provider failure; same context identity renders byte-identically; concurrent Self/Peer finals serialize by ledger admission order with deterministic sequence/context membership and self-excluding leases; each already-finalizing actor retains its returned frozen lease and exactly one final admission while rejecting obsolete live work. |
| AC-I003 | REQ-I003, INV-P-I001, INV-P-I002 | automated architecture/routing + offline Windows owner wiring | real Self/Peer output runtime with deterministic source/provider inputs and chatbox/overlay fakes or isolated endpoints | Live chatbox publications are zero, Peer chatbox publications are zero, source-final live promotion is zero, and channel identities remain unchanged across success, failure, fallback, cancellation, and shutdown. |
| AC-I004 | REQ-I004, REQ-I005, ALG-I002, ALG-I004 | table/property state-sequence tests | non-empty deterministic source traces including every revision kind, stable-boundary retreat, stable-language change, normalizer change, Unicode, emoji, combining characters, gate-open/closed, mode-change-during-turn, eligible Self/manual-language Peer, automatic-language Peer with stable versus missing/unstable capture hint, manual text, multi-target, and mixed-language cases | Every fixture asserts exact classification, eligibility reason, epoch/reset, pending/admission, and intermediate/final state; ineligible fixtures admit zero live work and retain their established final-only or source-only degraded path; repeat work is zero; active mode is unchanged by a pending setting; unitization exactly rejoins original target text and never produces an invalid boundary. |
| AC-I005 | REQ-I006, REQ-I007, ALG-I005, ALG-I006, ALG-I007, ALG-I008, ALG-I009, INV-A-I002 | property + model-based state-sequence tests | non-empty deterministic completions exercising bootstrap, later canonical free replacement/seeding, two-revision verified advance, verified rollback, economical free fallback, prefill support, overlap mismatch, expiry, due-audit prefill clamp, accepted audit, and destructive reset | Exact canonical, display, verified, provisional, generation, support-time, debt, and epoch state is asserted after every step; verified-provenance violations, forced-text verification, provisional lead violations, expired/due-audit prefill beyond verified, display-only canonical use, unknown-shape evidence, and rolled-back snapshot application are zero. |
| AC-I006 | REQ-I008, REQ-I010, ALG-I010, ALG-I012, INV-A-I003 | scheduler model/property + virtual-time state-sequence tests | non-empty capacity 1, 2, and 3+ scenarios exercising first slot, latest-source hedge, first-draft escape, audit sibling, same-source denial/allowed escape, two-slot coalescing, hard deadline, branch cleanup/debt bounds, pending release, and final preemption | Exact issued run kind/source/snapshot, logical-slot retirement, tombstone, active-branch count, cancellation-debt state, pending reconsideration, and final reserve are asserted after every event; live in-flight never exceeds two, pending never exceeds one, debt is bounded, finalizing live admission is zero, and capacity-one final fallback is sequential. |
| AC-I007 | REQ-I009, REQ-I012, ALG-I011, ALG-I013, INV-P-I005 | temporal/fault state-sequence tests | non-empty scenarios for all four dispositions, foreign/malformed/duplicate completions, retired-flight late completion, same-revision alternate/canonical replacement, on-time and late eligible no-prefill free/audit/escape evidence-only verified recomputation, completion inversion, 3–4 second delays, rollback, context/route change, eviction, authoritative final replacement, final failure, and shutdown races | Each disposition and presentation/frontier watermark is asserted exactly; the late eligible escape fixture updates evidence and verified state without regressing display or provisional-application watermark; foreign/duplicate completions leave active flights unchanged; screen regression, old-lineage mutation, final overwrite, late prefilled mutation, evidence ordering error, duplicate final admission, duplicate completion mutation, and closed-turn resurrection are all zero. |
| AC-I008 | REQ-I007, REQ-I010, ALG-I007, ALG-I008, ALG-I009, INV-A-I005 | provider contract state-sequence fixtures + architecture tests | non-empty suffix-only, full-echo, partial/ambiguous echo, ignored/rewritten prefill, malformed/truncated output, missing-provenance, cancellation, and safe free-degradation simulations | Conformed suffix/full-echo fixtures produce the exact complete text and generated suffix; every failure is metrics-only and selects the exact degraded next policy; missing provenance stays unknown; adapters do not make display/evidence decisions; output runtime does not make frontier decisions. |
| AC-I009 | REQ-I011, ALG-I001, INV-A-I001, INV-A-I003 | architecture + lifecycle/idempotency fault injection | production owner composition | Exactly one semantic owner mutates each turn; every allowed phase transition is exercised; duplicate completion/finalization/cancellation/context/provider events have the specified idempotent result without retiring unrelated active work; provider/output/settings/UI boundaries remain intact; teardown awaits close; unbounded task wait and post-terminal output are zero. |
| AC-I010 | REQ-I012, INV-P-I005 | reducer/property tests + affected overlay integration | desktop overlay and native bridge protocol fixture if touched | Semantic ordering is independent of transport sequence; final tombstones dominate live state; masking never changes canonical text; same-revision replacement occurs only under an explicit disposition. |
| AC-I011 | REQ-I013, INV-A-I004 | migration, application-port, locale, and debug-preview tests + Windows GUI inspection | supported Windows GUI at the accepted UI line and all supported locales | Settings migration backs up and round-trips; UI uses only intent/state contracts; locale keys are in parity; async callbacks use the supported task path; hidden preview performs no persistence, secret, provider, Broker, audio, chatbox, or overlay side effect. |
| AC-I012 | REQ-I014 | diagnostics schema/property + privacy inspection | deterministic traces and sanitized export | Required identities, counts, disposition, frontier, priority, policy-profile ID/fingerprint, exact source-unit/normalizer/target-unitizer/prompt versions and generation parameters, cancellation, and shutdown facts are reconstructable; raw text, translation, credentials, headers, bodies, secrets, and fabricated cost/provenance occurrences are zero by default. |
| AC-I013 | REQ-I015 | configuration/startup + architecture evidence | source and packaged configuration inspection | Simultaneous traffic and real live routes remain disabled, `TURN_BASED` remains default, validation fixtures are available, and no implementation artifact claims rollout readiness. |
| AC-I014 | all requirements and invariants | focused automated + smallest crossed-boundary regression + offline Windows production-entrypoint wiring | Windows x64, Python 3.12, official GUI entrypoint, real production owners, deterministic injected Self/Peer short/long inputs, isolated provider/output destinations, affected overlay fixtures | The complete feature path is wired through production owners behind the disabled boundary without real provider traffic; unaffected turn-based, manual, output, settings, prompt, provider, shutdown, overlay, and channel behavior remains compatible with no blocking regression. |
| AC-I015 | all requirements and invariants | evidence inventory + independent implementation review | one exact candidate SHA reconciled with the latest `dev` integration target | The candidate has a test-disposition ledger, architecture report, deterministic state/fault report, settings/locale report, offline Windows entrypoint-wiring record, open-finding ledger, rollback meaning, and fresh independent accepted review with no blocking claim; no real-provider, packaged end-to-end, UX, performance, cost, or activation claim is included. |

# Decision Authority

## Executor may decide

- Reversible implementation details, private types and internal APIs, module/helper placement, implementation sequence, test structure, and diagnostic representation.
- The inactive developer characterization profile, provided every value is versioned, observable, safely bounded, and not represented as validated or product-enabled.
- Whether optional log-probability data is collected for diagnostics, provided correctness never depends on its availability.

## Independent review required

- Any durable owner, lifecycle, concurrency, provider-capability, persistence, presentation-ordering, or output-routing boundary.
- Any production cutover, real-route enablement, compatibility/fallback removal, protocol change, or terminal completion claim.
- Any material strategy pivot from bounded frontier, immutable lineage, authoritative final independence, or maximum-two-flight semantics.

## User decision required

- Any observable product behavior beyond this contract, scope or non-goal change, compatibility break, supported provider/platform change, irreversible migration, security posture change, evidence weakening, cost budget, UX choice, or rollout decision.

# Completion Rule

Every acceptance criterion must be directly proven in its required environment and evidence class at one exact candidate SHA. Automated tests alone cannot replace required offline Windows entrypoint wiring, temporal, architecture, persistence, locale, privacy, or manual evidence. Real-provider, representative-audio, packaged end-to-end, UX, performance, cost, and activation evidence is reserved for the successor validation PRD. Completion means a disabled implementation candidate is `implementation_complete`; merge, push, provider credential access, validation, activation, deployment, release, evidence publication, and cleanup remain separately approved actions.
