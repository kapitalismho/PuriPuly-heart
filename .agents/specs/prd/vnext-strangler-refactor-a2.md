---
id: PRD-VNEXT-STRANGLER-A2-001
status: reviewed
source: .agents/specs/prd/drafts/vnext-strangler-refactor-a2.source.r11.md
baseline_ref: dev@3fb5ce83e4840ef1fd49f2b5480952a09af66527
integration_target: dev
document_review_verdict: ready
blocking_open_decisions: 0
owner_amendment: GOAL-G19 manual terminal evidence override approved 2026-07-31
---

# Outcome

PuriPuly Heart completes its owner-based architecture with an exact Flet 0.86.1 UI on packaged Python 3.12 without sacrificing the accepted canonical `dev` product: the replacement consumes the proven UI-facing application boundary, preserves the established appearance and interaction at 1136 by 850, five locales, Local ASR CPU/GPU, provider handoff, settings and credentials, overlays, packaging, lifecycle, and channel behavior, applies only the approved fixed-window and retired-card exceptions, and only then permits evidence-gated retirement of the current UI, `GuiController`, unused failed-refactor UI residue, and remaining multi-responsibility `ClientHub` ownership.

# Established Baseline

## Code baseline

- The normative code baseline is canonical `dev@3fb5ce83e4840ef1fd49f2b5480952a09af66527`.
- G00 established and repaired the functional and visual baseline; G02 through G09 transferred the approved backend responsibilities; and G10 proved the UI-facing application boundary through the accepted production UI. The branch-administration commits after G10 changed release metadata, lockfiles, ignore rules, and ancestry without approving a product behavior change.
- Uncommitted working-tree changes present during PRD authorship are excluded from the baseline.
- The current production path combines the accepted Flet 0.28.3 UI with the proven UI-facing application boundary and the production owners for output, Local ASR provisioning, provider runtime, Self capture, Peer capture, translation turns, settings persistence, and application shutdown. `GuiController` and `ClientHub` remain compatibility surfaces pending later retirement rather than authority for new application behavior.
- The reviewed Issue #20 Local ASR contract and the observable behavior present at this baseline are compatibility inputs to this refactor. Existing focused and release evidence may be reused, but the implementation merge is not itself a verified G00 functional baseline.
- The reverted B0 integration and failed candidate UI are not part of the production baseline or a design reference.
- The canonical baseline already includes the approved fixed Fast Translation and integrated-preferred Context policies and removal of their choice surfaces. All replacement-UI comparisons use the accepted production UI at the pinned baseline; earlier G00 evidence remains supporting provenance, not permission to restore the retired choices.

## User-visible surfaces

- Main frameless Flet application shell, title bar, navigation, dashboard, logs, about surface, settings views, cards, controls, status displays, and all dialogs and modals.
- Self microphone capture, STT, transcription presentation, translation, and configured outputs.
- Peer audio capture, STT, translation, desktop or VR subtitle presentation, and peer-specific consent and state.
- Manual text translation and its loading, success, partial, empty, retry, and error states.
- Provider selection, managed authentication, API credential entry and verification, provider status, and fallback behavior.
- Settings editing, cancellation, persistence, restart restoration, migration, and SecretStore-backed credentials.
- VRChat chatbox output, desktop subtitle overlay, and native VR subtitle overlay.
- English, Japanese, Korean, Russian, and Simplified Chinese UI locales.
- Normal startup, background operation, default-size main-window presentation, shutdown, restart, and debug UI preview.
- Local ASR CPU Auto and direct-model selection, GPU provider and device selection, model installation/repair/progress/failure/retry, channel loading state, and release of Local ASR resources on explicit capture toggle-off.
- Peer automatic-language selection, expected-language settings, final-language-run translation and presentation, and every associated success, fallback, cancellation, and error state.

## Actual product entrypoints

- Packaged Windows executable `PuriPulyHeart.exe`, entering through `puripuly_heart.main` and launching the Flet GUI by default.
- Source entrypoints `python -m puripuly_heart.main` and `python -m puripuly_heart.main run-gui`.
- Desktop overlay entrypoints exposed through `puripuly_heart.main` for the production renderer, preview, and diagnostic repro.
- Separately built native Windows VR overlay launched through the established overlay startup protocol.
- Separately built native Windows GPU worker launched and supervised by the desktop application only when GPU STT execution requires it; packaged Local ASR download helpers are internal application processes rather than supported user entrypoints.
- There is no supported product headless entrypoint.

## Platform and environment

- Product and production-composition evidence run on supported Windows x64 systems.
- Automated Python evidence uses the repository `.venv` on Windows.
- The baseline GUI uses Flet and Flet Desktop 0.28.3. The replacement composition uses exactly Flet 0.86.1, its coherent desktop runtime and build/runtime tooling, and packaged Python 3.12; mixed Flet protocol versions, an uncontrolled global CLI, or an implicit Python 3.13/3.14 runtime are not accepted evidence.
- At the pinned baseline, the main window defaults to 1136 by 850 but source and focused tests configure it as user-resizable with minimum dimensions 1024 by 760. The approved replacement intentionally changes that window behavior to fixed and non-resizable at 1136 by 850; this is a reviewed product delta rather than baseline parity.
- Visual comparison uses identical Windows version, theme, display scale, locale, settings, data, window state, and content for baseline and replacement captures.
- Required main-window comparison uses only the established 1136 by 850 size. The replacement main window is fixed at that size; minimum-size behavior, user resizing, resize-responsive behavior, and rendering at other dimensions are outside the parity evidence contract.
- Native VR overlay evidence includes a rebuilt Windows overlay when its changed surface requires it. Local GPU execution evidence uses the separately built worker on supported physical Vulkan hardware when the affected contract requires it. Installer smoke evidence uses an alternate AppId and isolated installation directory.
- Broker verification, when its changed surface requires it, runs only in a Linux-native workspace with Linux-installed dependencies.
- Source and packaged verification must include launch from an arbitrary current working directory, read-only bundled-resource discovery, compatible writable user-data locations, and clean process/resource shutdown.

## Compatibility baseline

- Existing product behavior and output semantics at the verified functional baseline are authoritative unless this PRD explicitly changes them. The main-window parity contract is the fixed 1136 by 850 presentation defined by REQ-008.
- Self, peer, and system are separate product channels, and peer utterances never route to the VRChat chatbox.
- Existing settings files and keys remain loadable, forward migration backs up before rewrite, persistence remains round-trippable, and defaults remain safe. Historical values retain their meaning except that `stt.low_latency_mode=false` and `ui.integrated_context_enabled=false` no longer restore retired choices and are normalized to the fixed policies in REQ-012 and REQ-013.
- Existing SecretStore service names and keys remain compatible. Secrets are loaded through SecretStore and encrypted-file storage continues to require `PURIPULY_HEART_SECRETS_PASSPHRASE`.
- Existing provider aliases, configured provider behavior, prompt fallback behavior, and current safe error-detail behavior remain compatible. Local ASR compatibility includes the established `local_qwen` identity, CPU Auto and direct CPU identities, the GPU provider identity and shared device selection, explicit GPU installation intent, Peer manual/automatic source mode, expected-language settings, and backed-up canonical migrations through v31.
- Local ASR model availability, verified installation and repair, provider handoff, pending-work expiry, terminal-final preservation, explicit toggle-off resource release, shared-GPU serialization, and ordinary worker/download failure containment retain their accepted product meaning.
- Existing Broker `/v1` behavior, desktop overlay behavior, native overlay protocol and startup contract, packaged executable identity, installer identity, and user-data profile remain compatible.
- Debug UI preview remains explicitly gated and inert with respect to settings persistence, secrets, and external systems.

# Scope

## Included

- Preservation of the accepted canonical `dev` functional and visual baseline through the remaining UI migration and retirement work.
- Continued use of the production owners and UI-facing application boundary already accepted through G10.
- Continued use of active and correct settings, provider, safety, runtime, output, overlay, and compatibility assets.
- Preservation of the established Issue #20 Local ASR/GPU, Peer automatic-language, settings v31, native worker, package, installer, and user-feedback behavior throughout backend and UI cutovers.
- Migration of the accepted UI to exactly Flet 0.86.1 through the stable UI-facing application contract, without redesigning the UI or changing observable product behavior beyond the fixed-window and retired-card deltas authorized by REQ-008 and REQ-013.
- Verification that the Flet 0.86.1 runtime bridge remains subordinate to the accepted application shutdown owner and does not expose Flet controls, sessions, events, or runtime types across the application boundary.
- Preservation of source and PyInstaller-packaged Windows startup from arbitrary working directories, compatible read-only and writable paths, Python 3.12, and the established executable, installer, overlay, GPU-worker, and helper-process contracts.
- Exact visual, interaction, locale, settings, provider, audio, translation, output, overlay, lifecycle, packaging, and restart parity against the verified baseline at the 1136 by 850 main-window size, subject only to the fixed-window and retired-card deltas authorized by REQ-008 and REQ-013.
- Retirement of the Fast Translation and Integrated Context user choices, removal of their settings cards and selection UI, and safe normalization of their historical persisted values to the approved fixed policies.
- Evidence-gated retirement of the legacy UI, `GuiController`, and multi-responsibility application ownership in `ClientHub`.
- Evidence-gated removal from the final canonical release tree and package of dormant UI implementations, presentation wiring, components, assets, adapters, and tests created by the failed prior refactor that are not required by the verified production path and differ from the approved UI appearance or behavior.

## Non-goals

### NG-001 — B0 or candidate resurrection

Reactivating the reverted B0 integration, adopting the failed candidate UI, or continuing its horizontal layer-completion plan is not part of this contract.

### NG-002 — Return to the pre-transition product

Resetting canonical `dev` to either archived pre-transition branch tip or semantically reconstructing the old product line is not part of the selected strategy.

### NG-003 — Unapproved product redesign

Changing information architecture, visual design, copy, navigation, workflows, defaults, features, provider behavior, output semantics, or window behavior beyond the retired settings changes in REQ-012 and REQ-013 and the fixed non-resizable replacement-window delta in REQ-008 is not authorized.

### NG-004 — Product headless mode

A supported headless microphone, stdin, translation, or release runtime is not introduced. Test-only drivers may use the production composition for evidence.

### NG-005 — Framework-driven behavior change

Flet 0.86.1 migration does not authorize visual or interaction differences at the fixed default size, backend contract redesign, simultaneous backend responsibility changes, a new application lifecycle owner, or a packaging-tool change.

### NG-006 — Premature legacy deletion

Legacy code is not removed merely because a replacement compiles, passes isolated tests, or exists behind test-only wiring.

### NG-007 — Replacement god object

Moving the current responsibilities into a differently named controller, hub, host, coordinator, service locator, or generic settings API without establishing single responsibility and explicit boundaries is not completion.

### NG-008 — Release authorization

PRD review and implementation completion do not authorize merge, push, deployment, public release, or worktree cleanup.

### NG-009 — Repository history rewriting

Removing failed-refactor UI residue from the final integration target does not require deleting historical commits or branches, rewriting Git history, or destroying archived evidence.

### NG-010 — Local ASR feature redesign

Changing the established CPU/GPU model choices, automatic-selection meaning, installation consent, worker fault boundary, channel behavior, settings identities, Peer automatic-language behavior, resource-release semantics, or user-visible Local ASR workflow is not authorized by the architecture refactor. Reversible download transports and private implementation details may change only while preserving the accepted product contract.

# Requirements

## REQ-001 — Functional baseline authority

The accepted G00-G10 evidence and canonical `dev` commit define the remaining program baseline for startup, self STT and translation, peer STT and translation, channel-correct output, settings restoration, configured provider use, established Local ASR CPU/GPU paths, overlays, native worker lifecycle, UI-facing application behavior, and clean shutdown. If the pinned baseline cannot reproduce a required behavior in its required environment, remaining cutover work stops for the smallest repair or reviewed contract decision rather than redefining the baseline silently.

## REQ-002 — Incremental production ownership

Each cutover must transfer one coherent responsibility to one explicit owner in the real product composition. The owner must define its inputs, outputs, state, failure behavior, diagnostics, and lifecycle obligations appropriate to that responsibility. Unrelated responsibilities and UI presentation must remain unchanged during that cutover.

## REQ-003 — Behavior-preserving cutover

Every owner cutover must preserve the verified baseline's user-observable behavior, data meaning, output routing, timing-sensitive interaction semantics, and failure containment except for the approved changes in REQ-012 and REQ-013. The old and new side-effecting paths must not both execute. The corresponding legacy responsibility remains available until the new owner is proven in production composition and may be removed only after the required evidence passes.

## REQ-004 — Stable UI-facing application boundary

Before replacement UI cutover, the existing UI must prove a stable application boundary in which UI code submits explicit user intents and renders localized view state or presentation events without owning provider creation, credential storage, settings persistence, capture or translation orchestration, output routing policy, or long-running backend task lifecycle.

## REQ-005 — Explicit lifecycle and diagnostics

Every async or long-running production responsibility must have an explicit owner, cancellation path, coordinated shutdown behavior, and actionable diagnostics. Provider replacement and application shutdown must await provider `close()` and must not leave active capture, translation, overlay, Local ASR download, decode, transition, GPU worker, or provider tasks and resources behind. Explicit capture toggle-off must preserve the established release semantics for the affected Local ASR channel.

## REQ-006 — Compatibility preservation

Owner replacement and UI migration must preserve existing settings and persistence meaning except for the two retired values defined by REQ-013, and must preserve backup and migration safety, SecretStore compatibility, provider aliases, prompt fallback behavior, Local ASR identities and canonical migrations through v31, Broker `/v1`, overlay and GPU-worker protocols and startup behavior, executable and installer identity, locale availability, and established user-data locations.

## REQ-007 — Isolated Flet migration

The Flet 0.86.1 UI must consume the UI-facing application boundary already proven through the baseline UI. Flet controls, pages, sessions, events, bridges, and runtime types remain adapter details and must not cross into backend owners. A UI migration cutover must not introduce or complete a backend ownership transfer, and backend contract changes discovered during UI migration require separate review and evidence before the UI migration resumes.

## REQ-008 — Exact UI appearance and interaction parity

The replacement UI must be visually and interactively identical to the verified baseline UI at the established 1136 by 850 main-window size, not merely similar or functionally equivalent, except for two approved deltas: the Fast Translation and Integrated Context settings cards and selection UI are absent as required by REQ-013, and the replacement window changes from the baseline's resizable 1136 by 850 default with 1024 by 760 minimums to fixed and non-resizable at 1136 by 850. Only the positioning and sizing of remaining cards in the directly affected settings rows may change as necessary to close the two vacated card positions. Outside those window-behavior and narrow layout exceptions, the UI must preserve the window shell, title bar, navigation, screen composition, control types, ordering, positioning, visibility, enabled and selected states, labels, localized text, icons, colors, typography, spacing, sizing, scrolling, focus behavior, modal stacking and dismissal, loading and error feedback, and established interaction behavior.

Parity applies to the main views, all remaining settings surfaces, all remaining established dialogs and modals, all accepted Local ASR/GPU provider, device, installation, progress, loading, failure, retry, and Peer automatic-language states, and the baseline states reachable in normal and explicitly gated debug-preview operation across every supported locale at 1136 by 850. One further delta is approved for the explicitly gated debug-preview surface only: the replacement's debug-preview action list may carry the additional foundation-primitives preview action introduced with the migrated token and primitive layer, so that surface may show one more action row and a correspondingly taller popover than the baseline. Every other debug-preview state, and the position of every baseline action row, remain under exact parity, and this delta must not appear in any non-debug surface or weaken INV-P-007 inertness. The two removed controls must not remain as disabled controls, empty cards, or inactive placeholders; closure of their vacated layout may reposition or resize only the remaining cards in the directly affected rows, must preserve their relative order, established design language, content, and behavior, and must pass comparative product-owner review. Minimum-window-size behavior, resize-responsive behavior, and rendering or interaction at any other main-window dimension are outside this contract. Any other user-perceptible difference at the fixed size is a failed acceptance result unless the product owner explicitly approves it through a reviewed PRD revision.

## REQ-009 — Evidence-gated legacy retirement

The legacy UI and `GuiController` may cease to be production entrypoints only after Flet 0.86.1 visual and behavioral parity passes. `ClientHub` must no longer own multiple application responsibilities; any retained Hub must be independently reviewed as a thin compatibility or channel facade. Removal must not weaken compatibility, rollback diagnosis, or required evidence.

## REQ-010 — Production release composition

The final product must remain operable through the established PyInstaller-packaged Windows executable and supported source GUI entrypoints on Python 3.12, with desktop and native VR overlay integration, the independent GPU worker and Local ASR download/install composition, installer behavior, provider access, settings restoration, and shutdown behavior preserved. Flet adoption does not authorize a build-tool, executable identity, installer identity, AppId, or user-data-location change. Test-only composition is not sufficient for terminal acceptance.

## REQ-011 — Failed-refactor UI residue removal

The final canonical release tree and packaged product must not retain or ship dormant alternative UI implementations, presentation wiring, components, assets, adapters, or implementation-coupled tests created by the failed prior refactor when they are outside the verified production path and differ from the approved baseline appearance or behavior. Such code must not be used as the basis of the Flet 0.86.1 UI merely because it reflects a newer architecture.

Before removal, reachability and compatibility evidence must distinguish failed-refactor residue from the working baseline UI, required application contracts, migration or compatibility paths, reusable non-UI assets, and evidence fixtures. Active or required behavior must not be removed by classifying it as residue.

## REQ-012 — Canonical Fast Translation and context policies

Fast Translation is the single enabled translation policy and is not a user-selectable mode. A historical value, restart, or compatibility path must not select the retired normal mode in resolved runtime configuration, STT orchestration, the active translation coordinator, provider verification, or provider construction. Real configured provider operation, verification, failure containment, established Qwen low-latency asynchronous selection, and established provider fallback behavior must continue to work without restoring an off choice.

Integrated context is the single preferred context policy and is not a user-selectable mode. When effective peer translation applies and eligible peer context is available, translation requests use integrated context. Whenever effective peer translation does not apply, or peer context is otherwise unavailable or inapplicable, the runtime automatically uses local context without failing, crossing channels, or exposing a user mode selector.

## REQ-013 — Retired settings UI and compatibility boundary

The Fast Translation and Integrated Context settings cards, their selection dialogs or modals, their user-facing selectable states, and any user interaction path capable of restoring either retired choice are absent in every supported locale. No disabled card, empty card, inactive placeholder, or alternate control may preserve either retired choice.

Historical settings containing `stt.low_latency_mode` or `ui.integrated_context_enabled` remain loadable. Their historical `false` values are accepted at the compatibility boundary but normalize to the fixed policies in REQ-012, cannot restore the retired choices after restart, and cannot flow into the product as user-selectable domain state. Any forward rewrite is backed up before mutation, failure-safe, and round-trippable in its canonical representation. Other historical settings and keys retain their established meaning.

## REQ-014 — Local ASR provisioning ownership

Local ASR model catalog and manifest authority, integrity validation, availability, explicit installation intent, download, repair, reinstall, progress, cancellation, failure containment, and diagnostics must have one production provisioning owner behind explicit application contracts. The owner must preserve independent model validity, CPU Auto completeness semantics, targeted direct-model usability and repair, GPU installation consent, verified promotion only after integrity checks, safe cancellation and cleanup, and application availability under ordinary download or asset failure. It must not own microphone capture, provider execution, settings serialization, translation, output routing, or Flet rendering.

## REQ-015 — Provider runtime and GPU-worker ownership

Provider construction, replacement, generic utterance-boundary handoff, discovery, activation, readiness, shared GPU execution, worker supervision, model residency, manual retry, resource release, awaited close, bounded forced termination, and lifecycle diagnostics must have one production provider runtime owner behind explicit ports. The independent GPU worker remains the fault boundary for GPU model operations and remains separate from the Python application and native overlay. Self and Peer owners retain channel-specific capture, admission, and the decision to request or commit handoff; the provider runtime does not absorb those channel responsibilities.

## REQ-016 — Established Local ASR and automatic-language continuity

The refactor must preserve the accepted Local ASR product behavior: independently valid CPU assets; CPU Auto routing by pinned provider capability; direct CPU choices; the established persistent manual-language mismatch switch to `local_qwen`; the explicit strict-Vulkan GPU provider and shared device selection; selected-GPU installation behavior; verified and cancellable model provisioning; non-blocking discovery and activation states; provider handoff without lost terminal finals; 12-second `speech_end` pending-work expiry; shared-GPU non-preemptive global `speech_end` FIFO; advisory available-VRAM treatment; no automatic GPU retry or CPU fallback; explicit toggle-off release; ordinary worker/download failure containment; decode-only RTF diagnostics; Peer automatic-language final-run translation and presentation; settings and provider compatibility through v31; localized UI states; independent GPU-worker packaging; and peer-chatbox denial. Reversible private transports may change, but no cutover may silently substitute providers, weaken installation integrity or consent, retain released resources, reorder retained GPU work, mix channels, or remove an established UI state.

## REQ-017 — Flet 0.86.1 runtime and storage compatibility

Before replacement-screen implementation or cutover proceeds, the exact Flet 0.86.1 compatibility gate must prove a coherent Flet package, desktop runtime, and official startup composition on Python 3.12. The in-process runtime bridge remains subordinate to the accepted application-shutdown owner and cannot create a competing owner, accept late work after shutdown admission closes, resurrect resources, or bypass application diagnostics. Existing private `flet_desktop` hooks are migration risks, not product contracts, and must be removed or isolated behind an accepted adapter unless direct Windows verification and independent review justify temporary retention. A failed gate blocks screen migration and requires the smallest reviewed framework, packaging, or boundary decision; it does not authorize fallback to Flet 0.85 or a broad rewrite.

The source and packaged application must locate bundled assets, prompts, locale data, metadata, and other read-only resources independently of the process current working directory. Settings, secrets metadata, logs, downloaded models, caches, temporary data, and other mutable state must remain in compatible writable locations. Flet adoption does not absorb the established ownership or process contracts of the GPU worker, native VR overlay, desktop overlay, Local ASR helpers, or Broker.

# Protected Invariants

## Product invariants

### INV-P-001 — Peer chatbox denial

Peer utterances never route to the VRChat chatbox under success, partial, retry, fallback, error, or shutdown conditions.

### INV-P-002 — Channel separation

Self, peer, and system messages retain distinct identity, state, presentation, and output eligibility throughout capture, transcription, translation, fallback, and rendering.

### INV-P-003 — Exact UI preservation

The verified canonical baseline UI appearance and interaction at the 1136 by 850 main-window size, including the absence of selectable retired-setting controls, are the authoritative replacement target. The approved exceptions are making the replacement window fixed and non-resizable rather than retaining the baseline's resize behavior, and repositioning or resizing remaining cards in the directly affected settings rows to close the retired-card vacancies while preserving their relative order, design, content, and behavior. No other redesign, interaction change, or visible deviation is permitted at that size.

### INV-P-004 — Core translation continuity

Self and peer capture, STT, translation, presentation, and configured output flows continue to work through the real Windows product composition after every relevant cutover.

### INV-P-005 — Persistence and credential continuity

Valid existing settings and credentials remain usable with the same meaning except for the two retired settings values explicitly normalized by REQ-013. Settings remain loadable and round-trippable, and forward migration remains backed up and failure-safe.

### INV-P-006 — Provider and prompt compatibility

Established provider aliases, configured provider selection, managed and user credential behavior, and prompt fallback semantics remain compatible.

### INV-P-007 — Inert debug preview

Debug preview remains explicitly gated and cannot persist settings, mutate secrets, or call external systems merely by exercising hidden UI states.

### INV-P-008 — Overlay and packaging compatibility

Desktop and native VR overlay behavior, native protocol and startup contract, packaged executable behavior, and installer safety remain compatible.

### INV-P-009 — Retired choices stay retired

Fast Translation remains enabled as the sole product policy, context remains integrated-preferred with automatic local fallback, and no UI or persisted historical value restores either retired user choice.

### INV-P-010 — Local ASR continuity and isolation

Established Local ASR CPU/GPU choices, availability, installation, capture, handoff, finalization, automatic-language, presentation, failure containment, and diagnostics remain usable without silent provider substitution or channel mixing; Peer-derived output remains denied to the VRChat chatbox.

### INV-P-011 — Explicit GPU intent and resource release

GPU model installation begins only from the established explicit user selection/application workflow, an inactive unselected path does not consume GPU execution resources, Self and Peer share the established worker/model resource, and explicit channel toggle-off and application shutdown release the resources required by the accepted lifecycle.

### INV-P-012 — Runtime, resource, and data-location coherence

Source and packaged execution use a coherent Flet 0.86.1 composition on Python 3.12, launch independently of current working directory, discover every required bundled resource, and preserve established writable user-data, executable, installer, and AppId identities.

## Durable architecture invariants

### INV-A-001 — One production owner per responsibility

A production responsibility has one active owner and one side-effecting execution path at a time.

### INV-A-002 — Proof before removal

Existing production behavior is not removed until its replacement is active in production composition and has passed the required evidence.

### INV-A-003 — Explicit boundaries

Cross-boundary behavior uses explicit input, output, lifecycle, message, rendering, persistence, or provider contracts rather than hidden cross-layer state access.

### INV-A-004 — Owned concurrency

Every background task and external resource has an explicit owner, cancellation and shutdown path, and diagnostics.

### INV-A-005 — Presentation does not own application behavior

The final UI owns rendering, localization, transient presentation state, and user interaction only; it does not own backend orchestration, persistence, providers, output policy, or resource lifecycle.

### INV-A-006 — Migration isolation

Backend ownership cutover and Flet framework migration are separately reviewable changes with separate evidence.

### INV-A-007 — No redistributed monolith

Completion requires reduced responsibility concentration and dependency direction that can be enforced; distributing the same hidden ownership across generic services or a replacement hub is not sufficient.

### INV-A-008 — One shipped UI implementation

The final release tree and package contain one approved Flet 0.86.1 production UI path and no dormant alternative UI produced by the failed refactor.

### INV-A-009 — Provisioning and execution remain separate

Local ASR asset provisioning and provider execution have separate explicit owners and contracts; neither is hidden in UI presentation, settings persistence, a capture owner, or a replacement god object.

### INV-A-010 — Independent GPU fault boundary

GPU discovery, model activation, inference, cancellation, and unload remain owned by the separately supervised GPU worker boundary, independent of the Python application and native overlay, with authenticated local communication, observable lifecycle, awaited shutdown, and bounded termination.

### INV-A-011 — Verified model authority

A Local ASR model is usable only when its own pinned installation contract validates; another model's validity cannot satisfy it, and failure or cancellation cannot promote an incomplete staged asset.

### INV-A-012 — Flet runtime remains an adapter

Flet controls, pages, events, sessions, bridges, and runtime processes remain below the UI adapter boundary. The accepted application owner controls admission, cancellation, shutdown, and diagnostics, and private Flet Desktop hooks do not become cross-layer contracts.

# Approved Decisions

- A2 remains the selected strategy; G00-G10 ownership work is accepted on canonical `dev`, and the remaining contract covers Flet migration and evidence-gated retirement.
- The accepted canonical `dev` production UI is the visual and behavioral oracle until replacement surfaces are proven.
- The reverted B0 integration and failed candidate UI are excluded.
- Existing active and correct canonical foundations are retained rather than rewritten by default; reuse of any specific private class or adapter remains an implementation decision contingent on production reachability and contract fit.
- Local ASR provisioning and provider runtime execution are separate durable responsibilities, while channel-specific capture, admission, and handoff decisions remain with Self and Peer owners.
- An implementation merge or focused test result is not production proof. Required Issue #20 compatibility evidence is established by the functional baseline, affected owner cutovers, and terminal acceptance rather than by a separate verification claim in this PRD.
- Flet 0.86.1 migration follows the proven UI-facing application boundary and remains separate from backend ownership changes.
- The Flet 0.86.1 UI must preserve the canonical baseline appearance exactly at 1136 by 850, including the approved absence of the Fast Translation and Integrated Context settings cards and selection UI; only remaining cards in the directly affected rows may reposition or resize to close those vacancies while preserving relative order, design, content, and behavior, and no other UI redesign is authorized.
- The baseline window is configured as resizable with 1024 by 760 minimums. The approved Flet 0.86.1 replacement intentionally changes it to fixed 1136 by 850 and non-resizable; minimum-size, resize-responsive, and alternate-size rendering are not parity commitments.
- The exact Flet package, desktop runtime, and build/runtime tooling must form a coherent 0.86.1 composition on packaged Python 3.12; mixed protocol versions, uncontrolled global tooling, or implicit newer Python selection are not evidence.
- Flet's runtime bridge remains subordinate to the accepted application shutdown owner, and Flet types remain UI-adapter details.
- PyInstaller, executable and installer identity, AppId, existing user-data locations, arbitrary-working-directory launch, read-only bundle discovery, and writable-data compatibility remain preserved unless a later reviewed product decision explicitly changes them.
- Fast Translation is fixed on as the sole product policy and is no longer user-selectable.
- Context is fixed to an integrated-preferred policy with automatic local fallback when eligible peer context is unavailable or inapplicable, and is no longer user-selectable.
- The two retired settings cards and their selection UI are removed in all five locales without inactive placeholders; every other fixed-size UI surface remains under exact parity.
- Historical values for the two retired choices remain loadable but normalize to the fixed policies through a backed-up, failure-safe, round-trippable compatibility boundary.
- The legacy UI and controller are removed only after replacement parity, and ClientHub's multi-responsibility ownership is eliminated.
- Unused UI code created by the failed prior refactor is also removed from the final canonical release tree and package after reachability and compatibility review proves it is not required production, migration, compatibility, reusable non-UI, or evidence code.
- There is no supported product headless runtime.
- The explicitly gated debug-preview action list may carry one additional foundation-primitives preview action beyond the baseline, approved by the product owner during G17. Every baseline action row keeps its exact position, and this delta is confined to the debug-gated surface.
- The blank settings card already present in the canonical `dev` baseline settings grid is pre-existing established behavior, not a retired-choice placeholder. The product owner ruled during G17 that it remains as-is, so it does not block the retired-card absence criterion and is not reworked by the migration.

# Open Product Decisions

None.

# Acceptance Criteria

| AC | Verifies | Evidence class | Required environment | Pass condition |
|---|---|---|---|---|
| AC-001 | REQ-001, REQ-016, INV-P-001 through INV-P-011 | automated + production-composition + manual | Clean worktree at an exact commit, repository `.venv`, supported Windows x64, configured real audio and providers, established desktop and VR output environment | The exact commit and environment are recorded; the application starts; self and peer STT and translation complete; outputs retain channel eligibility; peer never reaches chatbox; settings survive restart; configured cloud and Local ASR providers and overlays operate to the extent required by the accepted configuration; Local ASR/GPU availability and pending physical evidence are recorded rather than assumed; shutdown releases owned Local ASR and worker resources; every required failure is repaired and the full matrix rerun before the functional baseline is accepted. |
| AC-002 | REQ-002, REQ-003, INV-A-001, INV-A-002 | automated + production-composition + comparative + manual | Exact functional baseline and cutover commits on supported Windows x64 using the same scenario data and configured dependencies | For every owner cutover, traceable production wiring proves the new owner is active, only one side-effecting path executes, focused contracts and affected architecture guards pass, and the before/after Windows scenario has no unapproved behavior, state, output, failure, or timing-semantic difference beyond the deltas authorized by REQ-012 and REQ-013 before legacy responsibility removal. |
| AC-003 | INV-P-001, INV-P-002, INV-P-004 | automated + production-composition + fault-injection | Repository `.venv` and supported Windows x64 across self, peer, system, partial, fallback, retry, error, and shutdown scenarios | Self, peer, and system identity remain distinct at every observed boundary; peer output is denied to the VRChat chatbox in every scenario; allowed desktop and VR presentation still occurs exactly once. |
| AC-004 | REQ-004, INV-A-003, INV-A-005, INV-A-007 | automated architecture + production-composition + independent review | Repository source and tests at the proposed UI-boundary commit, plus the existing Windows UI using that boundary | Dependency enforcement and independent review find no UI ownership of providers, credentials, persistence, capture or translation orchestration, output policy, or backend task lifecycle; the baseline UI exercises all required commands and presentation states through the boundary in production composition. |
| AC-005 | REQ-005, INV-A-004 | automated + temporal + production-composition + fault-injection | Repository `.venv` and supported Windows x64 with active capture, Local ASR download, queued and active decode, provider transition, GPU worker, provider calls, overlays, cancellation, provider swap, explicit channel toggle-off, and repeated shutdown/restart | Tasks and resources have observable owners and diagnostics; cancellation, toggle-off, and shutdown complete within established baseline expectations; provider `close()` is awaited; no active owned task, audio stream, Local ASR download/helper, queued or active decode, provider transition, GPU worker/model, overlay session, or provider resource remains beyond its accepted lifecycle; failed replacement preserves a usable prior state. |
| AC-006 | REQ-006, INV-P-005, INV-P-006 | automated compatibility + production-composition + restart + migration | Historical supported settings fixtures, SecretStore test environments, configured providers, supported Windows x64, and Linux-native Broker workspace only if Broker changes | Settings round-trip and supported forward migration through v31 preserve meaning and backups except for the two explicitly normalized values in REQ-013; credentials remain available through compatible SecretStore keys; provider aliases and prompts resolve as before; restart restores the same state except that retired choices cannot return; any touched Broker or provider compatibility checks pass in their required environment. |
| AC-007 | REQ-007, INV-A-006 | comparative source review + automated + production-composition | Proven backend-boundary baseline and separate Flet 0.86.1 cutover revisions, repository `.venv`, supported Windows x64 | The Flet cutover consumes the previously proven application boundary, exposes no Flet runtime type to backend owners, contains no backend ownership transfer, handles discovered backend contract changes separately, and reaches the same real Windows backend behaviors through the official UI startup path. |
| AC-008 | REQ-008, INV-P-003 | comparative visual + comparative interaction + automated window-constraint inspection + manual product-owner review | Baseline Flet 0.28.3 and replacement Flet 0.86.1 on identical supported Windows x64 environment, theme, display scale, locale, settings, data, window state, and content; both compared at 1136x850, with baseline resize behavior separately observed | Side-by-side and overlaid captures plus direct interaction comparison at 1136x850 cover the shell, every main view, every remaining settings surface, every remaining established dialog/modal, accepted Local ASR/GPU selection, device, installation, progress, loading, failure, retry, and Peer automatic-language states, normal states, error/loading/disabled/selected states, gated debug-preview states, and all five locale bundles; the retired cards and selection UI are absent without disabled controls, empty cards, or inactive placeholders; only remaining cards in the directly affected settings rows may reposition or resize to close the vacancies while preserving relative order, design, content, and behavior; all geometry and interaction outside the approved card and window-behavior exceptions match the canonical baseline; the closure passes product-owner visual review; and inspection and direct manipulation prove the baseline is resizable with its recorded minimums while the replacement is fixed at 1136x850 and cannot be user-resized. |
| AC-009 | REQ-009, INV-A-002, INV-A-005, INV-A-007 | automated architecture + production-composition + independent review + manual | Final proposed legacy-retirement revision on supported Windows x64 after AC-007 and AC-008 pass | The Flet 0.86.1 UI is the production entrypoint; the legacy UI and `GuiController` are unreachable and removable without regression; ClientHub has no multi-responsibility application ownership or is removed; dependency guards pass; the full Windows baseline matrix still passes after deletion. |
| AC-010 | INV-P-007 | automated + manual inertness | Supported Windows x64 with debug preview explicitly enabled and external calls, persistence, and SecretStore writes observable | Exercising every debug-preview state performs no settings persistence, secret mutation, provider, Broker, audio, chatbox, desktop overlay, or VR overlay external action. |
| AC-011 | INV-P-008 | automated protocol + production-composition + build + isolated installer smoke | Supported Windows x64, rebuilt native overlay when touched, packaged application, alternate installer AppId, isolated installation directory | Desktop and VR overlays retain protocol, startup, rendering, and shutdown behavior; the independent GPU worker and packaged Local ASR download helper retain their process, packaging, startup, cancellation, and shutdown contracts when applicable; the packaged app launches normally; isolated install, upgrade scenario when applicable, launch, and uninstall do not alter the production installation or user profile unexpectedly. |
| AC-012 | REQ-010, INV-P-004 through INV-P-008 | full automated regression + packaged production-composition + manual + temporal | Release-candidate packaged Windows application, supported configured providers, audio inputs, desktop and VR outputs, persisted settings and secrets, and established installer environment | The packaged application completes the full accepted baseline matrix, repeated restart and shutdown, settings restoration, provider use, self and peer translation, output routing, overlays, and exact UI comparison without relying on a test-only or headless runtime. |
| AC-013 | REQ-011, INV-A-008 | reachability analysis + source and package inventory + automated architecture + independent review + production-composition | Final canonical source tree, release-candidate package, exact accepted UI contract, and supported Windows x64 production composition after AC-008 passes | An inventory classifies every retained and removed candidate by production reachability, visual and behavioral conformance, compatibility, migration, reusable non-UI, and evidence purpose; every item proven to belong only to the visually or behaviorally incorrect failed-refactor alternative UI and proven unnecessary for production, compatibility, migration, reusable non-UI behavior, and evidence is absent from the release tree and package; required production paths, application contracts, compatibility and migration paths, reusable non-UI assets, and evidence fixtures and tests are preserved; architecture checks and the full Windows baseline still pass after removal. |
| AC-014 | REQ-012, INV-P-004, INV-P-009 | automated behavioral + traceable production wiring + production-composition + provider + fault-injection + manual | Historical `low_latency_mode=false` settings, repository `.venv`, and supported Windows x64 with real configured self and peer audio, configured Qwen and other supported provider paths as applicable, effective peer translation available and unavailable, restart, retry, fallback, and error scenarios | After loading the historical false value and restarting, traceable evidence proves the enabled policy reaches resolved runtime configuration, STT orchestration, the active translation coordinator, provider verification, and provider construction; Qwen selects the established low-latency asynchronous path rather than the retired synchronous normal-mode path; configured provider verification and translation work; effective peer translation with eligible context produces integrated-context requests; every scenario without effective peer translation, or with unavailable or inapplicable peer context, produces local-context requests without failure or channel mixing; established fallback and error containment remain effective; self and peer translation complete with peer chatbox denial preserved. |
| AC-015 | REQ-013, REQ-006, INV-P-003, INV-P-005, INV-P-009 | automated migration + persistence round-trip + restart + comparative visual and interaction + manual product-owner review | Historical settings fixtures containing both values of the retired keys, isolated settings copies, repository `.venv`, and supported Windows x64 at fixed 1136x850 across all five locales | Historical files load without error; a forward rewrite creates a backup before mutation and is failure-safe; both historical false values normalize to the fixed policies and cannot return after save and restart; the canonical representation round-trips while unrelated settings retain meaning; both cards and all selection UI are absent without inactive placeholders or stale localized choices; only remaining cards in the directly affected rows move or resize to close the vacancies, their relative order, design, content, and behavior remain unchanged, the closure is approved, and every surface outside that narrow geometry exception remains visually and interactively identical. |
| AC-016 | REQ-014, REQ-016, INV-P-010, INV-P-011, INV-A-009, INV-A-011 | automated contract + compatibility + temporal + production-composition + fault-injection | Exact owner-cutover commit in repository `.venv` and supported Windows x64 with complete, partial, missing, invalid, cancelled, failed, repaired, CPU Auto, direct CPU, manual-language mismatch, and selected-GPU installation scenarios | Production wiring uses one provisioning owner; independent model validity and CPU Auto completeness are correct; CPU Auto routes by the pinned capability sets; direct valid models remain usable; the approved manual-language mismatch persistently selects `local_qwen` and no other silent substitution occurs; selected-GPU installation follows explicit application intent; progress and diagnostics are safe; cancellation or failure promotes no incomplete asset and leaves no helper or late status; targeted repair succeeds; UI/controller no longer owns provisioning lifecycle; and established provider, settings, package, and application availability behavior remains unchanged. |
| AC-017 | REQ-005, REQ-015, REQ-016, INV-P-010, INV-P-011, INV-A-004, INV-A-009, INV-A-010 | automated contract + temporal + concurrency + production-composition + process fault-injection + diagnostics + manual | Exact provider-runtime cutover commit in repository `.venv` and supported packaged Windows x64 on a physical Vulkan device, with delayed discovery, activation, Self-only, Peer-only, shared-GPU, handoff, queue-expiry, low-reported-VRAM, retry, worker failure, toggle-off, and shutdown scenarios | Production wiring uses one provider runtime owner; discovery remains non-blocking and delayed discovery is pending rather than failed; activation distinguishes validation, load, warmup, readiness, and failure; generic handoff preserves terminal finals while channel owners retain their decisions; one worker/model is shared without channel mixing; retained work executes non-preemptively in global `speech_end` FIFO, work waiting 12 seconds expires before start without cancelling active decode, and started CPU/Vulkan attempts expose decode-only RTF with queue/load time separate; low reported VRAM remains advisory; ordinary failure is isolated, retains the configured selection and asset, exposes manual retry, and performs neither automatic retry nor CPU fallback; explicit toggle-off and shutdown reject or discard new work, cancel and close owned work, release required resources, await provider close, and leave no worker or task; the overlay and unaffected providers remain usable; and UI/controller no longer owns provider execution lifecycle. G00 may record unavailable physical evidence as pending, but this cutover and terminal acceptance cannot pass until the required physical and packaged evidence is complete. |
| AC-018 | REQ-008, REQ-016, INV-P-001 through INV-P-003, INV-P-010, INV-P-011, INV-A-005 | automated UI-boundary + translation lifecycle + comparative visual and interaction + production-composition + manual | Accepted baseline and replacement UI compared at 1136x850 on identical supported Windows x64 environment across all five locales and deterministic Local ASR/GPU/Peer automatic-language states, including mixed-language, whole-utterance, missing-language, unsupported-language, fallback, cancellation, and error cases | The UI-facing boundary exposes the required commands and localized view state without owning provisioning or provider execution; every accepted Local ASR/GPU and Peer automatic-language state is visually and interactively identical to the baseline except for the fixed-window and retired-card deltas authorized by REQ-008; explicit selection/install and retry actions preserve their meaning; provider-neutral final-language runs retain order and unique identity, produce terminal translated or established source-only/cancelled child outcomes, close the parent only after all children are terminal, preserve the configured manual-language fallback meaning, and publish only peer-channel output with chatbox denial; debug preview remains inert; and no old surface is retired before the evidence is accepted. |
| AC-019 | REQ-010, REQ-017, INV-P-008, INV-P-012, INV-A-004, INV-A-012 | dependency inspection + automated + source production-composition + packaged production-composition + temporal + independent review | Clean exact compatibility-gate candidate before replacement-screen implementation, repository `.venv` on Windows, exact Flet 0.86.1 desktop composition, packaged Python 3.12 build, ordinary and arbitrary process working directories, isolated writable user profile, and observable child processes/resources | A recorded accepted gate exists before replacement-screen implementation proceeds; dependency and runtime evidence proves a coherent Flet 0.86.1 composition with no uncontrolled global CLI or mixed protocol; source and packaged entrypoints launch from arbitrary working directories; every required bundled resource is found; mutable data stays in compatible writable locations; each private `flet_desktop` hook is either removed, isolated behind an accepted adapter, or temporarily retained only with direct Windows verification and independent-review acceptance; Flet types do not cross the application boundary; shutdown rejects late work, awaits owned teardown, leaves no Flet, overlay, helper, GPU-worker, or provider residue, and reports actionable diagnostics; executable, installer, AppId, user-data, overlay, worker, helper, and Broker contracts remain unchanged. |

# Decision Authority

## Executor may decide

- Reversible implementation details.
- Private types and internal APIs.
- File and helper placement.
- Implementation sequence within the currently authorized responsibility boundary.
- Focused tests, diagnostics, and evidence collection mechanics that do not weaken required evidence.

## Independent review required

- Durable boundary reliance.
- Production cutover.
- Legacy, compatibility, migration, rollback, or fallback path removal.
- Persistence, security, lifecycle, concurrency, or public API change.
- Any claim that a retained Hub or facade is thin rather than an application owner.
- Material strategy pivot.
- Terminal completion.

## User decision required

- Observable product behavior.
- Any visual or interaction difference from the verified baseline UI at the 1136 by 850 main-window size outside the approved retired-card and fixed-window exceptions.
- Scope or non-goal.
- Compatibility break.
- Irreversible migration.
- Security posture.
- Supported platform, locale, or provider.
- Required evidence weakening.

# Completion Rule

Every acceptance criterion must be directly proven in its required environment and evidence class. Automated tests, static architecture checks, screenshots, provenance records, or sub-issue completion alone cannot replace required Windows production-composition, comparative visual and interaction, provider, audio, overlay, installer, temporal, or manual evidence. PRD review and Goal completion remain separate from merge, push, deployment, release, and worktree cleanup approval.

For GOAL-G19 only, the product owner's 2026-07-31 direct statement that all owner-operated checks work, followed by the explicit instruction to approve and move past the remaining manual-artifact gap under the `start-goal` owner-priority rule, satisfies the missing exact-candidate Peer and packaged owner-operated manual observation slots. This narrow evidence exception accepts the owner's firsthand product judgment despite the absent Peer/package machine-readable artifacts and does not generalize to another Goal. It does not waive the existing automated, source, package, configured-provider, physical GPU, UI, installer, privacy, architecture, rollback, or lifecycle evidence; the Self/Peer/System separation and unconditional Peer chatbox-denial invariants; fresh independent terminal implementation review; or the separation of Goal completion from merge, push, deployment, release, and cleanup approval.
