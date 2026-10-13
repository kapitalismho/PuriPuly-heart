# Issue 218: practical speaker attribution and semantic source composition

## Agreement and starting point

The maintainer supplied the practical-policy proposal and requested this Markdown plan **and implementation against it**. This is the task's approved local implementation agreement, not a claim that the feature or either Nemotron profile is already qualified.

Starting candidate: `b2cfc0c42bfa07d1570c243fb17ab2e547b8da1e`, branch `listen-integrate-the-initial-fp16-vulkan-nemotro`, clean working tree, no configured upstream. Issue [#218](https://github.com/kapitalismho/PuriPuly-heart/issues/218) is OPEN and already In progress. Earlier measured failures and their limits are recorded in the [issue comment](https://github.com/kapitalismho/PuriPuly-heart/issues/218#issuecomment-6139799554) and `native/nemotron/issue218_*` evidence. They are the failing-before evidence; do not rerun them merely to reconfirm the failures.

The new agreement supersedes, **for this implementation**, the earlier proposal restricted to complete source-plus-context coverage and same-speaker interior holes. It also narrows the old prohibition on majority attribution: never assign clearly distinct speakers' actual speech to a majority speaker, but permit representative attribution of a single utterance despite brief ambiguous boundaries, low-confidence holes or reused context. Remote issue bodies have not been amended by this file. Publication, push, merge, release, worktree cleanup and production enablement remain unapproved.

The system map remains [architecture.md](architecture.md). This work changes attribution/composition policy within existing owners, not the two-strategy architecture or the immutable `SourceChild` boundary.

## Intended observable results

1. Clear local A/B speech receives useful distinct speaker identities; returning A preserves its identity within the same valid epoch. The all-gray result is not an acceptable substitute.
2. Each local request uses eligible speaker evidence available at **its own native final**, without adding a wait or changing its submitted audio/options.
3. Short ambiguity at the edge or inside one utterance does not automatically make the entire utterance gray. Actual short other-speaker speech, including “No.” and numeric corrections, is not absorbed or discarded.
4. A confidence transition such as gray → B does not turn one already-finalized phrase into multiple translation jobs. `Gentlemen of the press.` remains a useful translation unit. Actual A → B → A remains separate source units.
5. Parakeet's native returning-speech loss is diagnosed with the exact failed PCM and corrected where supported, independently of color policy. If a native limitation remains, contain harmful optional cuts only on that route and disclose the limitation rather than silently accepting lost content.
6. Gemini final-content coverage and missing source correspondence remain separate questions. Preserve all authoritative final fragments; do not fabricate mapping or promote speculative interim text.

## Preserved invariants and non-goals

- Nemotron remains the sole identity authority, with the same F16 artifact, Vulkan neural inference, eight-slot limitation and full 1.04/0.64 geometries. No replacement diarizer, alignment model, source separation or provider speaker-ID fallback.
- Capture/order/request/contribution identities, exactly-once source/context registration, cancellation, stale-result fences, source-order publication and bounded retention stay strict.
- Submitted PCM, actual source/context ranges, submission cursor, native options and grouping identity freeze at submission. Attribution freezes before immutable source admission and translation. No published recoloring or guessed text partition.
- Keep 500 ms genuine onset recovery and 300 ms hard-rollover context. Onset recovery can contain new first syllables; it is not disposable replay. Reused context cannot outweigh new speech. A brief proven replay edge may consume the same bounded ambiguity allowance; substantial foreign replay, overlap and clear competing new speech still prevent representative assignment.
- No fixed speaker wait, future evidence, re-decoding parent plus children, cloud pre-send hold, extra model or new translation layer. No automatic silence padding, fuzzy deletion or global minimum utterance duration.
- Trimming remains OFF. No exhaustive tuning grid, zero-repeat certification or new general telemetry system.
- No VRChat/SteamVR launch. Non-VR checks do not qualify HMD readability, photon latency or VR coexistence.
- Reuse recorded native Deepgram events for deterministic changed-path checks and actual local inference/translation/rendering; no fresh Deepgram call is planned. After the initial plan, the maintainer explicitly authorized necessary Gemini calls without repeated approval: “제미나이 공짜니깐 승인 받지 말고 필요한 만큼 써”. Execute proportionate raw-wire/SDK/final probes with `GOOGLE_API_KEY2`, accounting for every attempt and submitted audio; this does not authorize other providers. Do not describe replay as a fresh provider run.
- Neither profile becomes a production default. Route-specific remaining limitations do not block unrelated reachable improvements, but they remain visible and cannot be reported as feature acceptance.

## Outcome A — Local final-time attribution and practical evidence policy

Owner: speaker/source implementation workstream. Surfaces: `core/stt/batch_speaker.py`, related source attribution/composition code and behavior tests. Shared classifier changes, if useful, belong to this same owner rather than competing implementations.

### Decisions

- At submission retain immutable audio provenance, not a final speaker verdict.
- On each request's own native terminal, snapshot attribution once from evidence already delivered and immutable then. Do not postpone the decision to ordered release behind another request.
- Retain unresolved requests' necessary source/onset footprints in the existing bounded evidence lifetime. Release that need after the snapshot. Cache exhaustion, discontinuity and reset must remain explicit; never grow retention without bound or reuse a newer epoch's equal slot number for old speech.
- Prefer unique new/onset speech support over already-used continuation context. Treat short uncertain onset/fade/transfer edges and interior holes as bounded acoustic uncertainty, not automatic whole-result failure.
- Use a small, explicit rule in existing attribution owners: a clear representative speaker plus bounded ambiguous evidence, with no supported competing new utterance/overlap or broken source/epoch identity. A long speaker cannot outvote a clear short “No.” from another speaker. A tiny already-used foreign-speaker replay edge may consume the ambiguity budget; the same samples as new/onset speech cannot.
- Initial implementation limits: active probability 0.5; clear probability 0.8 with runner-up at most 0.2; competing probability 0.35; ambiguous evidence at most 120 ms continuously, 200 ms total and 20% of unique clear speech. These are reversible starting heuristics, not measured optima. Missing new-speech coverage remains unavailable. Record counterexamples rather than accumulating special cases until one selected recording becomes colored.

### Verification

Behavior coverage: evidence arriving during ASR; out-of-order native finals; no evidence after own-final changing a decision; pending retention and hard bounds; reset/capture/epoch fencing; onset versus reused context; bounded edges/holes; true short A-B-A, overlap and missing central evidence. Exercise actual GPU Qwen/Nemotron/Gemma at the production 50 ms evidence drain and observe actual rendered identity/color, not only fabricated metadata tests.

## Outcome B — Phrase-preserving streaming composition

Owner: the same speaker/source workstream, to keep local and streaming policy interactions coherent. Surfaces: `core/stt/streaming_speaker.py`, affected source contracts/callers and behavior tests.

### Decisions

- Keep one already-finalized source/phrase together when only confidence changes. Prefer one representative identity where supported; otherwise keep the coherent unit gray.
- Split on supported distinct speaker transitions, not unknown as if it were a separate speaker. Do not merge across genuine A-B-A, incompatible epochs, unsupported clocks, independent release boundaries or unowned text.
- Qualified estimated word timing may support representative attribution; it does not repair an unknown clock. Keep the raw timing and exact text/range accounting.
- Do not wait for a later server sentence. Do not alter physical cloud input control or introduce post-translation text splitting.

### Verification

Use the recorded Deepgram `Gentlemen of the press.` and `Washed the detritus from his hair.` cases through the current source join and real local translation. Report resulting source units, exact concatenated text, history registrations, translation count/meaning and rendered styles. Protect a short real B between A spans. Existing route contract suites cover unchanged clocks, marker/terminal tails, cancellation and ordering. Recorded-provider replay is explicitly not new cloud-service qualification.

## Outcome C — Parakeet native loss

Owner: Parakeet runtime/provider workstream. Surfaces: provider-local decode/runtime dependencies, relevant tests and task-local probe/report files; no concurrent changes to the speaker/source owner's files.

1. Verify actual installed/pinned sherpa versions and the claimed upstream `NemoNormalizePerFeature` stability fix/release using primary sources. The attachment's 1.13.5 claim is a lead, not evidence.
2. Reconstruct the exact failed input from the existing corpus/provenance. Compare the changed runtime on those same PCM bytes/model/options, preserving hashes. Existing repeated `Yeah.`/empty results supply the old-side evidence.
3. If a verified runtime correction fixes the loss, update the compatible dependency/distribution surfaces and exercise actual product decoding. Do not add arbitrary tail/context as a substitute for understanding preprocessing.
4. If it does not fix the loss, investigate a bounded input-length/trailing-silence/normalization distinction and choose an evidenced route-local containment for harmful optional early cuts. Do not suppress ordinary short speech or conceal a remaining native failure.

Acceptance records distinguish a causal repair, an effective but unexplained native change, and containment with a remaining limitation. Test package/runtime compatibility as well as the selected recording.

## Outcome D — Gemini finality and mapping disposition

Owner: focused read-only investigation followed by a separately owned validation-only wire probe; any evidenced product repair is assigned explicitly.

- Inspect existing decoded SDK messages, receive/drain lifecycle, installed SDK parsing and official contracts/upstream reports.
- Account for all 30 observed authoritative final fragments. Missing interim-only content is not proof the application discarded final text.
- Read-only investigation found no demonstrated loss of observed final text in the product or an ordinary-text SDK decoding defect. Existing recordings cannot distinguish stripped/unobserved wire fields or finals after the observation window. A live task-local probe now compares raw inbound messages before parsing, decoded SDK messages, adapter results and actual EOF/close boundaries; begin with one public clear-A-B input and expand only for a decision-changing uncertainty.
- Preserve conservative unmapped finals and existing complete text behavior. No nearby-time speaker guessing, alternate credential fallback or speculative interim promotion.
- Live Gemini probes use only `GOOGLE_API_KEY2` from the maintainer-designated parent `.env.local`, without recording credentials. The first probe is bounded to one start, at most 8 seconds of submitted public PCM and at most 10 seconds of observation after final EOF, with a 30-second session deadline. Further necessary probes are authorized, not an invitation to a profile sweep or arbitrary retries.

Completion of this investigation is not a claim that the service's final coverage or rich speaker mapping has been repaired. Explicit route limitations are permitted by the maintainer's supplied proposal; they must not prevent completing A/B/C where actionable.

## Execution and evidence sequence

1. Record this agreement and inspect current implementation/evidence. Delegate independent Parakeet and Gemini investigation while one owner implements coherent speaker/source changes.
2. Implementation owners run focused behavior checks and actual changed-path smoke checks. Reuse previous failures; preserve existing raw evidence rather than overwriting it with new outcomes.
3. The validation owner runs the integrated non-VR local scenario at 50 ms drain with real GPU ASR, Nemotron and Gemma; verify actual Flet output separately. Compare both full profiles and matched baseline on a proportionate reused case set: clear A-B, returning A-B-A, short “No.”/numeric correction and actual six-second continuation. Report useful content as well as color.
4. Replay qualified recorded Deepgram native data through current code and real translation/rendering. Do not mistake fewer jobs alone for correct meaning or blanket merging for correct speaker preservation.
5. Director updates architecture/reference documentation to describe implemented lifetimes and acoustic policy, integrates and commits the checkpoint, then obtains an independent pinned baseline-to-candidate review. Repair accepted findings and verify only affected checks.
6. An independent completion-evidence audit accounts for every outcome, unchanged invariants, actual measurements and unresolved prerequisites. Record its exact candidate and the Director's scoped acceptance below.

No repository-wide audit or new profile/parameter search is planned. A meaningful unexpected regression expands only the relevant investigation. The branch has no configured upstream; check current remote state before final review without pushing or silently rebasing the measured implementation.

## Completion ledger

| Outcome | Status | Evidence / unresolved criterion |
| --- | --- | --- |
| Plan and revised local agreement | Recorded | This file; current user instruction; prior #218 evidence |
| A: final-time and practical local attribution | Implemented; focused checks passed | Own-final snapshot, bounded retention and representative policy; actual integrated GPU/color qualification pending |
| B: semantic streaming composition | Implemented; focused checks passed | Current recorded-native join smoke keeps phrases coherent; real translation/renderer comparison pending |
| C: Parakeet | Implemented and smoke-verified; review pending | Matched 1.13.5 runtime restores both exact returning-A sentences; `hand/hands` substitution and empty common hard-rollover tail remain. [Report](../native/nemotron/issue218_parakeet_report.md) |
| D: Gemini | Investigation and live wire probe completed; route limitations remain | One valid exact-input session conserved the sole authoritative B final across raw wire/SDK/adapter/engine; A appeared only in interim. No additional final through EOF + 9.015 s. [Findings](../native/nemotron/issue218_gemini_wire_findings.json) |
| Integrated non-VR verification | Preparing execution | Production cadence, both profiles, content/identity/renderer evidence required |
| Independent checkpoint and completion acceptance | Pending | Stable candidate, reviewed diff and complete evidence index |

All implementation heuristics, commands, observed outputs, review dispositions and remaining prerequisites will be recorded here or in linked task-local reports before completion. Do not label an acoustic failure or unavailable external observation as passed.

## Recorded implementation evidence

### Parakeet package-level repair

Primary-source verification confirmed sherpa-onnx [v1.13.5](https://github.com/k2-fsa/sherpa-onnx/releases/tag/v1.13.5) contains the [normalization stability correction](https://github.com/k2-fsa/sherpa-onnx/commit/0967a08db705d8eec9cf5cef962c8ec57c16e4a8). The old actual runtime was 1.13.4. Only the matched wrapper/core pair changed; the model, PCM, options, NumPy/ONNX Runtime and Gemini SDK were preserved.

Both exact returning-A requests recover sentence content through the real installed product scoped queue/terminal. This proves a package-level repair, not single-function causality. No synthetic silence, extra context, recut, minimum-duration rule or retry fallback was added. The returned `hand/hands` instead of `hair`, and the existing empty hard-rollover tail, remain explicit content failures.

The isolated and actual product checks each decoded seven original requests. Focused provider/package/license tests yielded 93 passes and one failing existing installer inventory guard; that guard was corrected to use the canonical local-STT manifest inventory and its rerun passed. No immutable installer payload was rebuilt. Exact commands, PCM/model/native hashes and outputs are retained in the [Parakeet report](../native/nemotron/issue218_parakeet_report.md) and linked JSON evidence.

### Gemini raw-wire disposition

The authorized probe used the exact current `gemini-3.5-transcribe-live` adapter, Google SDK 2.21.0 and `GOOGLE_API_KEY2`. In one valid public clear-A-B session, cumulative interim text contained both sentences, but the preparse WebSocket supplied only `Gentlemen of the press.` as authoritative `inputTranscription`. That exact final was conserved through SDK, adapter and engine. The receive loop remained active after the normal 2.005-second drain and received no additional message through EOF + 9.015 seconds.

This observed case is not a received final discarded by the app or SDK, and does not support an arbitrary longer-drain repair. It establishes bounded wire absence, not a universal service defect or guaranteed non-arrival after close. Final-to-capture correspondence remains unqualified; voice-activity offsets alone do not establish it. No Gemini product change, interim promotion, guessed speaker assignment or SDK upgrade was made.

Total new usage, including two invalid task-driver attempts retained separately: three starts and 11.44575 seconds of successfully submitted PCM. The valid attempt used 5.83775 seconds. The invalid attempts were instrumentation/base64 and structural-protocol proxy errors, not provider failures. Their accounting, corrections, raw/decoded/engine sequences and actual close boundary are retained in the [wire findings](../native/nemotron/issue218_gemini_wire_findings.json) and the linked archives. The user authorized necessary Gemini calls; no new Deepgram call was made.

### Speaker/source implementation readiness

The [speaker-policy report](../native/nemotron/issue218_speaker_policy_report.json) records 261 passing source/translation checks, 510 passing provider-route checks and 172 passing scoped-engine cancellation/order/drain checks: 943 total under the actual repository `.venv` interpreter. Focused Ruff checks passed after removing two obsolete imports. Earlier global-interpreter attempts lacked the declared `truststore` dependency and are invalid environment observations, not product failures or acceptance evidence.

A throwaway causal session exercised evidence arriving during native ASR: attribution identified the local source at its own final, with unchanged 16,000-sample PCM and English options, then one source/history/translation admission. A recorded Deepgram join smoke produced coherent A/B/returning phrases in three jobs without physical cuts. These are execution/protocol checks, not the pending actual GPU/translation/renderer measurements.

Validation ownership transferred from `profile-experiment` to `policy-validation` after repeated internal session-dispatch failures. The former owner made no new validation-driver edits; the replacement retains the same scope, evidence and native tier. No product or acceptance requirement changed.
