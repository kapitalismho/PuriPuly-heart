# Issue 180 — Self end-to-end and Peer software-latency diagnosis

## Scope clarification and disposition

Issue #180 was originally published with two Peer-oriented boundaries. After the maintainer clarified that the observed symptom is **Self speech feeling slow until overlay subtitles appear** and requested “self까지 포함해서 다 조사해줘” (“investigate all, including Self”), this run extends the same bounded Outcome through Self capture, scoped STT, translation, output, and Python presenter acceptance. The earlier Peer findings remain valid, but they no longer constitute a complete diagnosis of the reported Self symptom. The original safety constraints and non-goals remain unchanged: this is not a production fix, policy change, provider benchmark, private-speech capture, device/native-renderer investigation, or claim of a field regression.

| Question | Scoped disposition |
| --- | --- |
| Does a delayed local STT terminal delay the next recognition turn? | **Mechanism demonstrated for both production dispatchers.** Peer retained the previously reported 200 ms B source-to-dispatch shift under a 250 ms A-terminal delay. Self's actual capture dispatcher/adapter/owner chain retained B and, in the final real-loop control, moved B begin from 74 ms to 174 ms when A release moved from 44 ms to 173 ms. Both wrote the same 2 context + 8 content samples; Self B terminal moved from 164 ms to 202 ms, less than its start shift because its queued end followed immediately. |
| Where does Self wait before subtitles appear? | **Two independent software waits demonstrated.** First isolated Self original text applied 0.451 ms after its STT release; translated text applied 0.552 ms after the controlled translation completed. For a short B turn, STT can first wait in Self's ordered capture dispatcher behind A terminal. After B STT release, B's original `self_transcript_final` can separately wait behind A's translation/publication ownership: 76.137 ms in the immediate-terminal/180 ms translation case and 166.095 ms in the delayed-terminal case. A same-schedule zero-delay-A-translation control reduced that B release-to-original-application interval to 0.237 ms. |
| Does completed Peer output wait beyond intended replacement policy? | **Not reproduced in the retained controlled cases.** The five-item burst accumulated the expected applications at 0, 0, 1000, 2000, and 3000 logical ms. The protected-row case advanced on the actual Self clear at 400 logical ms. The final real-clock control observed 1,002.304 ms ready-to-application around one one-second gate and 148 µs from eligible recheck to application. |

**Final next decision: maintainer architecture decision; no production change in this Outcome.** Decide separately (1) whether Self/Peer capture should continue serializing B behind A terminal, and (2) whether the Self presentation contract should keep B's original text behind A's translation completion/close or allow original-text progress without violating Self ordering and identity. The Peer evidence supports no pacing implementation change. No timing constant should be tuned from this synthetic run.

## Frozen execution record

- Actual execution HEAD: `091ecce6b07e9be2050abb77065ac7aa1a5355a0`.
- Verified production baseline: `13274569769d3c1ec7a896a2d15b919b76136a6e`. The harness requires this SHA to be an ancestor and rejects every changed or untracked path outside `experiments/issue_180/`, so artifact-only descendant commits remain runnable without accepting production drift.
- Issue publication's inspected-current reference: `4daaeb4ac71892a783acb2e785bd062fdf80905d`; structural historical reference: `4e967df9d03649106faa8348c3ec611009529ffe`.
- Runtime: Windows 11 `10.0.22631`, CPython `3.14.7`, `uv 0.9.17`.
- Production owners exercised: Self `_GenerationGuardedVadSink` → `SelfCaptureVadSinkAdapter` → `SelfTranslationChannelOwner` → `ScopedRecognitionEngine`; `TranslationTurnLifecycleOwner`/`TranslationRequestOwner`/`TranslationOutputProjectionOwner`; `OutputRuntime`; and `OverlayPresenter`. Peer owner coverage is retained unchanged.
- Doubles/settings: deterministic scoped STT session, synthetic float arrays, deterministic UUIDs, no-I/O chatbox, and a deterministic translation provider with A/B delays of 180/40 ms (plus a zero-delay-A control). Self uses the actual event loop's monotonic clock because the capture dispatcher stamps `asyncio` loop time; its values are one-shot wall scheduling observations, not a benchmark. Peer injected-clock values remain causal logical schedule values. No paid call, model, private speech, capture device, native renderer, or HMD was used.
- Self source profile: 16 kHz; two context samples + eight content samples per turn; 20 ms declared endpoint/trailing-silence interval; A start/seal targets 0/40 ms; B start/seal targets 70/110 ms; immediate A/B terminal targets 40/150 ms; delayed A/B terminal targets 160/200 ms. Actual timestamps are reported below.
- Command:

  ```text
  uv run python experiments/issue_180/probe.py --output experiments/issue_180/trace.jsonl
  ```

- Result: exit 0; all focused assertions passed; 383 compact trace rows across eleven scenarios. Raw trace: [`trace.jsonl`](trace.jsonl). Runnable harness: [`probe.py`](probe.py).

## A. STT turn handoff

### Fixture and fixed source schedule

Both A/B cases use `PeerAudioSegmentLedger`, the production `_GenerationGuardedVadSink` ordered dispatcher, `ScopedRecognitionEngine`, `STTProviderEventBuffer`, and a deterministic implementation of the scoped session port. Provider/settings are fixed as `local_qwen`, 16 kHz, identical signatures.

| Source/provider event | Immediate A terminal | Delayed A terminal |
| --- | ---: | ---: |
| A source start / provider begin / first write | 0 / 0 / 0 ms | 0 / 0 / 0 ms |
| A local seal / provider seal | 100 / 100 ms | 100 / 100 ms |
| A provider terminal / engine release / end-handler return | 100 / 100 / 100 ms | 350 / 350 / 350 ms |
| B source availability | 150 ms | 150 ms |
| B local seal availability | 300 ms | 300 ms |
| B dispatch attempt / provider begin / first write | 150 / 150 / 150 ms | 350 / 350 / 350 ms |
| B provider seal | 300 ms | 350 ms |
| B provider terminal / engine release | 350 / 350 ms | 400 / 400 ms |

Trace anchors: immediate [rows 212–243](trace.jsonl#L212-L243); delayed [rows 244–275](trace.jsonl#L244-L275).

### Attribution

- **A terminal wait:** 0 ms in the immediate case versus 250 ms from A seal to terminal in the delayed case.
- **B source-to-dispatch/admission wait:** 0 ms versus 200 ms. In the delayed case B `SpeechStart` is available at 150 ms but its production-dispatch attempt is at 350 ms.
- **Where the wait occurs:** `_GenerationGuardedVadSink._run()` awaits each sink call before dequeuing the next event (`core/runtime/peer_channel.py`). A's call reaches `ScopedRecognitionEngine._handle_end()`, whose `seal_turn()` is followed by `_await_terminal()` and `_finish_turn()`. Therefore the production dispatcher is still awaiting A's end handler. B does **not** reach the engine's explicit `await resolved.wait()` loop in this path; that loop is a second safety boundary for concurrent direct callers.
- **B provider work versus readiness:** the controlled session applies the same 50 ms post-B-seal terminal delay in both cases. B terminal/readiness moved only 50 ms later (350 → 400), not the full 200 ms start shift, because B's source end was already queued by 300 ms and was dispatched immediately behind its delayed start at 350 ms. This demonstrates partial absorption of the initial lag rather than assuming a later start implies an equally later final.
- **Buffered content:** B retains and writes the same two-sample context payload plus eight-sample content payload in both cases. In the delayed trace, those writes carry original source ranges ending at 150 and 300 ms, and B end reaches dispatch with source age 50 ms. This is retained audio, not silence or loss. The experiment does not evaluate ASR accuracy.
- **Correlation:** every trace row retains segment order; provider begin/write/terminal rows retain the generated provider-turn ID. Engine terminal release is matched to the same segment identity.

### Current and historical adapter verification

This is not an overlap claim based on controller shape alone.

- At the verified production baseline, `ScopedRecognitionEngine._handle_end()` waits for the terminal before `_finish_turn()` sets `_turn_resolved`. The selected current local adapter, `providers/stt/local_qwen_sherpa.py`, clears/collects the turn buffer in `seal_turn()`, enqueues a `LocalDecodeCoordinator` job, and emits the scoped terminal only from `_handle_decode_completion()` (or a failure/expiry path). Thus, for this adapter, the engine's terminal wait includes queue/decode completion rather than only a cheap seal acknowledgement.
- At historical `4e967df9d03649106faa8348c3ec611009529ffe`, `ManagedSTTProvider._on_speech_end()` appends pending-final identity and awaits `session.on_speech_end()` but has no controller-level wait for the corresponding final before a later `_on_speech_start()` sends audio. The historical `LocalQwenSherpa` adapter's `on_speech_end()` copies/enqueues the current buffer and clears it; its next-turn `send_audio_f32()` can populate a fresh buffer while `LocalDecodeCoordinator` and the backend decode lock serialize actual inference. **Structural comparator conclusion:** historical code allowed next-turn capture/buffering to overlap preceding decode, but did not make two local decodes concurrent. This is not a historical runtime benchmark.

## B. Peer output and presenter pacing

All output scenarios exercise production `OutputRuntime.publish_overlay_event()` admission, the ordered Peer overlay writer, destination batch readiness, `OverlayPresenter.emit_peer_when_admissible()`, state reduction, and `OverlayApplicationReceipt`. `physical_ack=false` is preserved: application acceptance is not a renderer/HMD acknowledgement. Routing observer rows are captured live with `decision.decision`; no after-the-fact snapshot is assigned a later timestamp.

### Replacement control

Two new occupants fill two free slots at 0 logical ms. A same-logical-occupant update for occupant 2 and its real `utterance_closed` event both apply at 0 with no replacement wait. The occupant UUID, event type, source order, accepted handoff, and application receipt remain visible in [rows 276–295](trace.jsonl#L276-L295). This confirms original/update/close events for one logical turn are not automatically new replacement opportunities.

### Five-item ready burst

All five translations are ready and accepted into output at 0 logical ms. Destination batches are active immediately in this unmanaged-event fixture; therefore the observed destination-readiness wait is zero. The additional delayed-destination fixture was explicitly deferred as out of this repair's scope. The writer is strictly ordered.

| Item / logical source order | First presenter check | Application | Queue/inherited delay before presenter | Intentional replacement wait | Protection | Logical residual after eligible recheck |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0 ms (free slot) | 0 ms | 0 ms | 0 ms | 0 ms | 0 logical ms |
| 2 | 0 ms (free slot) | 0 ms | 0 ms | 0 ms | 0 ms | 0 logical ms |
| 3 | 0 ms | 1000 ms | 0 ms | 1000 ms | 0 ms | 0 logical ms |
| 4 | 1000 ms | 2000 ms | 1000 ms behind item 3 | 1000 ms | 0 ms | 0 logical ms |
| 5 | 2000 ms | 3000 ms | 2000 ms behind items 3–4 | 1000 ms | 0 ms | 0 logical ms |

The expected opportunity is reconstructed from actual occupant history: slots fill with 1 and 2 at 0; each later new occupant is eligible one second after the preceding new occupant (3 at 1000, 4 at 2000, 5 at 3000). The total 2–3 second delays are policy-driven ordered backlog, not a flat one-second subtraction and not evidence of an extra defect. Because `controlled_sleep` advances the injected clock and yields, zero logical residual here is **not** a wall-scheduling measurement. Live decisions and gates are [rows 296–335](trace.jsonl#L296-L335).

### Protected-row release and progress

With one slot, an actual `self_active_update` creates a protected visible row at 0 logical ms. The waiting Peer item records `protected_rows`. At 400 ms an actual `self_active_clear` goes through `OutputRuntime`, publishes revision 2, signals admission change, and frees the slot. The Peer item is rechecked, becomes eligible by the free-slot exemption, and receives application revision 3 at the same injected-clock instant.

Attribution is 400 logical ms of protection lifetime, zero observed queue/destination wait before the initial presenter check, and zero remaining replacement interval. Wall scheduling from eligibility to application is unmeasured in this injected-clock scenario. The item advances on its blocker-release wakeup; no lost wakeup or circular wait appears in logical state progression. See [rows 336–349](trace.jsonl#L336-L349).

### Head-of-line update/close contract

The focused head-of-line scenario fills two slots with source orders 1 and 2, then holds a source-order-3 new occupant at its one-second replacement gate. While the writer head is held, the probe submits an update and close for the already-visible order-2 occupant through `OutputRuntime.publish_overlay_event()` **without overriding identity**. `OutputRuntime` resolves the remembered production identity `(generation=1, source_order=2)`. Because `_latest_peer_source_order` is already 3, both events are rejected at admission as `stale_source_order`; they never become admissible queued work behind the paced head.

This grounds why a production-valid already-visible Peer update/close cannot be exercised *behind* the later paced head under the preserved source-order contract: to admit it with a higher order would change the logical turn's frozen source identity, which is not a valid profile. In valid ordering, that occupant's update/close must be admitted before the later source-order replacement and therefore precedes its wait. The independent Self clear in the protection scenario uses a separate path and progresses while the Peer writer waits. Evidence: [rows 350–370](trace.jsonl#L350-L370). This is a contract result, not a pacing bypass or fix proposal.

### Real-clock post-eligibility observation

A separate one-slot control uses the real monotonic clock and real `asyncio.sleep`. After the first occupant applies, the second is ready at 135 µs, begins its replacement wait at 188 µs, is rechecked eligible at 1,002,291 µs, and completes presenter application at 1,002,439 µs. The single observed ready-to-application interval is 1,002.304 ms; the observed eligible-recheck-to-application interval is 148 µs. Focused assertions bound these at 0.9–1.5 seconds and 0–100 ms respectively. This is sufficient to observe this boundary once, not a benchmark or general scheduler-latency claim. Evidence: [rows 371–384](trace.jsonl#L371-L384).

### Existing diagnostic-field limits

- `logical_pacing_wait`, `handoff_wait_ms`, `wait_reason`, and `pending_batches` are present in the live decision trace.
- For burst items 4 and 5, `handoff_wait_ms` is only 1000 although post-translation totals are 2000 and 3000 logical ms, because its timer begins at the first presenter wait after earlier writer/queue time.
- `wait_reason` retains only the first reason. In the protection case it reports `protected_rows` and 400 ms; it would not alone expose a later transition to `replacement_gate`.
- The probe-local gate rows fill the reason-transition observation gap without changing production behavior.

## C. Self end-to-end software latency

### Actual owner chain and first-turn control

The Self scenarios enter the production Self capture dispatcher with raw `SpeechStart`/`SpeechEnd`, pass through the production `SelfCaptureVadSinkAdapter` and `SelfTranslationChannelOwner`, and reach a production `ScopedRecognitionEngine`. The scoped session and translation provider are deterministic doubles; translation/output/presenter owners are production implementations. All rows carry source/segment UUID, provider-turn ID, translation UUID, publication ID, and presenter occupant UUID as applicable. For one logical turn these identities resolve to the same source UUID; event IDs distinguish original, translated, and close publications.

The isolated turn establishes that no predecessor is required for the symptom:

| Isolated A boundary | Actual real-loop time |
| --- | ---: |
| Source available / provider begin | 2.772 / 3.038 ms |
| Acoustic last-sample offset / local seal | 20 / 51.418 ms |
| Provider terminal + engine release | 81.472 ms |
| Original `self_transcript_final` application | 81.923 ms |
| Translation start / completion | 82.312 / 262.343 ms |
| Translated `translation_final` application | 262.895 ms |

The 20 ms acoustic value is the configured last-speech-to-local-seal interval, not when the trace callback ran. Original application followed release by 0.451 ms; translated application followed deterministic provider completion by 0.552 ms. These are Python presenter receipts (`scene_revision` 1 then 2), not physical pixels. Evidence: [rows 2–31](trace.jsonl#L2-L31).

### Same-schedule successive turns and controlled attribution

| B boundary | Immediate A terminal, 180 ms A translation | Delayed A terminal, 180 ms A translation |
| --- | ---: | ---: |
| Source available / local seal | 73.970 / 118.992 ms | 81.660 / 111.199 ms |
| Dispatch + provider begin | 74.144 ms | 173.905 ms |
| B terminal + engine release | 163.731 ms | 202.291 ms |
| B original subtitle application | 239.868 ms | 368.386 ms |
| B translation start / completion | 240.109 / 280.136 ms | 368.595 / 408.609 ms |
| B translated subtitle application | 280.493 ms | 408.826 ms |

- **Capture/STT wait:** Self `_GenerationGuardedVadSink._run()` awaits `SelfCaptureVadSinkAdapter.handle_vad_event()`, which awaits `SelfTranslationChannelOwner.handle_vad_event()`, which awaits scoped recognition. A's end call cannot return until its terminal is received and its Self callback finishes. Therefore delayed A holds queued B before the engine. B's source-to-begin age is 0.174 ms in the immediate case versus 92.245 ms in the delayed case.
- **Retained content and lag absorption:** B writes the same two context and eight content samples in both cases. In the delayed case, queued start and end dispatch back-to-back after A; B end is already 62.884 ms old when its provider seal executes. B release shifts only 38.560 ms (163.731 → 202.291), less than the 92.071 ms increase in its source-to-begin age.
- **Translation/application wait after B STT:** `ScopedRecognitionEngine._finish_turn()` sets `_turn_resolved` before awaiting the Self event callback. The callback reaches `_handle_transcript()`, whose `self_transcript_presentation()` takes `_self_publish_lock`/`_self_surface_lock`. A retains the ordered Self presentation chain until A translation is submitted, applied, and closed. B's engine release is therefore visible at 163.731/202.291 ms, but B original text cannot apply until 239.868/368.386 ms; only then does B translation start. This is neither Peer replacement pacing nor provider decode.
- **Controlled translation attribution:** with the same immediate-terminal source schedule but A translation delay changed from 180 ms to zero, B released at 152.619 ms and its original applied at 152.856 ms (0.237 ms later), versus 76.137 ms later with the 180 ms A translation. This isolates the post-STT wait to Self translation/presentation ownership rather than presenter scheduling. Evidence: immediate [rows 32–91](trace.jsonl#L32-L91), zero-delay control [rows 92–151](trace.jsonl#L92-L151), delayed terminal [rows 152–211](trace.jsonl#L152-L211).

### Applicable boundaries and historical comparator

- Low-latency mode is off, so speculative selection, resume debounce, `low_latency_finalize_wait_ms`, and awaiting-VAD timeout are not on the exercised path. No negative claim is made about those unexercised policies.
- The applicable pending-final boundary is the scoped engine's terminal future plus the Self dispatcher's awaited end call. Translation predecessor/presentation ownership is separately observable after engine release. Self output calls `OverlayPresenter.emit()`, not Peer `emit_peer_when_admissible()`, so the one-second replacement gate and `protected_rows` are not applicable to Self original/translated updates in these scenarios.
- At historical `4e967df9d03649106faa8348c3ec611009529ffe`, the Self generation guard directly awaited the Self VAD adapter, and the adapter directly awaited `SelfTranslationChannelOwner.handle_vad_event()`. That owner reached `ManagedSTTProvider._on_speech_end()`, which recorded pending-final identity and awaited `session.on_speech_end()` but did not await the matching final event. The historical local Qwen session copied/enqueued its buffer and cleared it on speech end, allowing subsequent Self audio into a fresh buffer while `LocalDecodeCoordinator` serialized decode. This verifies both historical Self dispatch and adapter before the bounded structural statement: historical Self could overlap next-turn buffering with prior decode; it did not run concurrent local decodes. This is not a historical runtime benchmark and is not, by itself, a regression claim.

## Limits, missing observations, and architecture drift

- Injected-clock Peer timings are causal schedule values, not latency distributions, wall performance, or predictions of field magnitude. The Self and Peer real-loop scenarios are one-shot bounded observations, not a statistical campaign.
- Scoped STT and translation work is deterministic; no model/provider comparison was performed. Current and historical selected adapters were verified structurally so overlap claims are bounded correctly.
- Controlled translation-runtime timing was measured through the production Self lifecycle and output owners. A delayed destination-ready parent, real provider, audio device, native overlay, VR runtime, HMD, and physical acknowledgement were not measured. The delayed Peer destination fixture remains out of scope; destination wait was explicitly zero in the selected Peer scenarios.
- Whether the maintainer's reported real-session magnitude reproduces with a real model/device remains unknown. `application_accepted` proves Python presenter acceptance only.
- No production source, policy, ownership, ordering, generation fence, content/expiry rule, or receipt meaning changed. Architecture drift: **none observed** relative to `docs/architecture.md`; exercised ownership remains Self/Peer capture dispatch → scoped recognition owner → translation lifecycle/output runtime → presenter owner. No actual drift is suspected from this evidence: the waits conform to current owner contracts, while whether those contracts match the desired UX is the maintainer decision identified above.
