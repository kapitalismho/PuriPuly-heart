# Issue 180 — Self end-to-end and Peer software-latency diagnosis

## Approved implementation outcome (2026-09-23)

The maintainer subsequently approved implementation with “수행”. That approval creates a new implementation outcome and does not retroactively change issue #180's investigation-only authority or the historical observations below. The implementation baseline is `9c630d3cb1528d63b3cb6e06ddf4c7c1fe012832`; the earlier production comparator remains `13274569769d3c1ec7a896a2d15b919b76136a6e`.

### Probe-fidelity correction

The historical Self/Peer probe constructed `ScopedRecognitionEngine(event_sink=...)`. Production STT composition instead uses `ProviderRuntimeHandle.start()`, which calls `engine.bind_event_sink(...)` and sends engine output through the bounded `STTProviderEventBuffer` and its FIFO dispatch task. The direct fixture therefore incorrectly allowed output-callback time to propagate into capture-handler time. The historical trace remains preserved as the record that was executed, but its claim that production capture awaited the translation/output callback is withdrawn.

The corrected harness now starts a real `ProviderRuntimeHandle` around the engine. It separates:

- the provider-terminal boundary, which can still hold a non-overlap-capable session's sealed A turn;
- engine terminal retirement and source-ordered enqueue into the bounded event buffer; and
- later FIFO callback delivery, which cannot hold capture after enqueue.

The implementation probe accepts only the explicit approved implementation paths, records the clean implementation baseline plus per-file content hashes, and writes [`trace_after.jsonl`](trace_after.jsonl). Command:

```text
uv run python experiments/issue_180/probe.py --output experiments/issue_180/trace_after.jsonl
```

The retained run used CPython 3.14.7 on Windows 11, exited 0, asserted 478 rows across twelve scenarios, and bound the harness as SHA-256 `05010e9ba8aa90054ca986fa34ed51c7f157988079904e136fb1a51d43bd5f75`. The prose timings below were refreshed from that final retained artifact after it was written; consequently, the trace metadata's pre-write report content hash describes the input report rather than this derived timing update.

### Implemented STT boundary

`ScopedRecognitionEngine` now distinguishes one open audio-input turn from identity-scoped sealed turns awaiting terminal. A session must explicitly expose `allows_sealed_turn_overlap`; the engine otherwise preserves the terminal wait. `STTSessionEventProjection` owns payload/update sequences, sealing, terminal authority, and retirement per admitted identity. Engine terminals are drained in source order even when provider completions arrive out of order, and bound output delivery remains FIFO and generation-fenced.

The local Qwen/Parakeet CPU family and local GPU adapter declare sealed-turn overlap after their seal methods transfer the whole audio segment into owned bounded work. CPU inference remains serialized by `LocalDecodeCoordinator`; an active CPU decode failure terminalizes only that turn while a successor already admitted to the bounded queue is still decoded, without retrying A or increasing inference concurrency. The GPU path continues through the shared bounded runtime: its single automatic decode-worker restart preserves queued successors, while fatal device/recovery failures remain explicit `FAILED` transitions and terminalize discarded work rather than promising recovery. The engine does not create parallel local decode. Abort invalidates authority first and terminalizes all admitted identities; close, provider retirement, settings changes, retention accounting, and terminal deduplication cover the full admitted set.

The corrected controlled Peer comparison retains the non-capable boundary as a control: with delayed A terminal, B begins at 350 logical ms. The overlap-capable case uses the same source schedule and true deferred provider binding: B begins and writes at 150 ms, before A's terminal at 350 ms, with the same 2 context + 8 content samples. In the final retained Self delayed-A run, B begins at 79.129 ms while A terminal is not received until 170.848 ms.

Remaining provider waits are intentional and exact: every remote protocol adapter currently remains non-overlap-capable, because its acknowledgement/result correlation has not been proven safe for a second open turn. Those adapters still wait for A terminal or protocol retirement before B. Provider `begin_turn`, audio writes, and `seal_turn` themselves remain bounded awaited operations. Backend queue/resource admission and source retention limits remain in force.

### Implemented Self original-output boundary

Unmanaged `SelfTranscriptFinal` and source-only `UtteranceClosed` overlay events now use the `self:original` destination-admission lane rather than the managed `self` translation-parent lane. They still route through `OutputRuntime`; no presenter, authority, duplicate, expiry, generation, receipt, or destination policy is bypassed. Managed translation UI, chatbox, overlay, and close work retain their original parent identity and ordering.

In the final retained immediate case, B terminal is delivered at 156.941 ms, B original is inserted active in `self:original` at 157.321 ms, and its application receipt arrives at 157.641 ms while A translation remains pending until 227.846 ms. In the delayed-terminal case, B terminal is delivered at 201.518 ms and B original applies at 201.900 ms while A translation remains pending until 351.638 ms. A's later managed translation and close retain A's identity; source-only B close releases B's independent lane so C can progress without allowing late A work to close or revive B. Managed B translation still follows managed A translation order; only B's original subtitle lifecycle was separated from A's parent lifetime.

Peer one-second replacement pacing, protected rows, source ordering, and multi-destination translation behavior are unchanged. The retained output controls still apply at logical 0, 0, 1000, 2000, and 3000 ms, and the real-clock pacing control remains within its asserted 0.9–1.5 second bound.

### Focused regression evidence

- `tests/core/test_stt_scoped_engine.py`: successor admission after seal, B-before-A completion, source-ordered terminals, deferred sink delivery, successor-begin failure retirement that still drains A's actual final, abort authority, retention, timeouts, settings rotation, and drain behavior.
- `tests/core/test_stt_session_projection.py`: multiple sealed identities retain independent sequence and terminal authority, while explicit retirement/end-of-epoch rejects late callbacks.
- `tests/providers/test_local_qwen_sherpa.py`, `tests/providers/test_local_cpu_backends.py`, and `tests/providers/test_local_gpu_backend.py`: local adapter compatibility, CPU queued-successor failure isolation, honest GPU terminal behavior, and bounded lifecycle behavior.
- `tests/providers/test_protocol_a_scoped_sessions.py` and `tests/integration/test_stt_connection_reuse.py`: remote adapters retain their conservative one-turn protocol boundary.
- `tests/core/runtime/test_output_runtime.py` and existing presenter/projection suites: B original and source-only close advance while A parent is pending, C can follow, and late/removed identity, ordering, duplicate, destination, and tombstone behavior stays owned by the existing output and presenter contracts.


## Scope clarification and disposition

Issue #180's first boundary (STT handoff) was channel-agnostic; its second boundary (overlay pacing) was Peer-display-specific. The earlier experiment exercised the first boundary only through Peer capture and therefore omitted Self from boundary A. After the maintainer clarified that the observed symptom is **Self speech feeling slow until overlay subtitles appear** and requested “self까지 포함해서 다 조사해줘” (“investigate all, including Self”), this run adds Self coverage to the channel-agnostic handoff boundary and expands through full Self output. The prior Peer findings remain valid. Original safety constraints and non-goals remain unchanged: this is not a production fix, policy change, provider benchmark, private-speech capture, device/native-renderer investigation, or claim of a field regression.

| Question | Scoped disposition |
| --- | --- |
| Does a delayed STT terminal delay the next recognition turn? | **Mechanism demonstrated with controlled sessions for both production ordered dispatchers.** Peer retains the known 200 ms B source-to-dispatch shift under a 250 ms A-terminal delay. Self retains B and delays B provider admission behind A's awaited terminal/callback. The Self session double does not model a serialized local decode worker, so its exact lag absorption is fixture-specific rather than a real-provider forecast. |
| Where does successive Self original text wait? | **Initial blocker identified.** B original publication `evt-2` is inserted inactive in destination-batch scope `self` behind active A translation-parent batch. In the immediate case A's `complete_target(..., "applied")` releases the A batch and activates `evt-2` 76.992 ms after B insertion; B applies 0.293 ms later. The delayed-terminal case shows the same release reason and successor activation. The Self projection locks are held while this await propagates, but are not the initial blocker. |
| Does completed Peer output wait beyond intended replacement policy? | **Not reproduced in the retained controls.** The five-item burst applies at 0, 0, 1000, 2000, and 3000 logical ms; protected-row progress follows the actual Self clear at 400 logical ms. The final real-clock control observes 1,013.485 ms ready-to-application around the intended one-second gate and 179 µs from eligible recheck to application. |

**Final next decision: maintainer architecture decision; no production change in this Outcome.** Decide separately (1) whether Self/Peer capture should serialize B behind A terminal/callback and (2) whether an unmanaged B original Self-overlay publication should share destination-batch scope `self` with A's managed translation parent, thereby waiting for A parent output completion. The Peer evidence supports no pacing implementation change. This evidence does not support bypassing admission or tuning a timing constant.

## Frozen execution record

- Ambient execution HEAD: `7988eff3c3d86e538e3f87892c163a81b1a4cef1`. This identifies surrounding committed source, not the experiment artifact bytes.
- Verified production baseline: `13274569769d3c1ec7a896a2d15b919b76136a6e`. The harness requires this SHA to be an ancestor and rejects every changed or untracked path outside `experiments/issue_180/`.
- Executed working-tree harness SHA256: `65e7e0addf246d31ce39d4f9b6c164b5c50929f19ed8597e7df344fc18ed9997` for `experiments/issue_180/probe.py`. Trace metadata binds this content directly; it does not imply that the artifact was committed at execution time, and it remains reproducible after an artifact-only commit.
- Issue publication's inspected-current reference: `4daaeb4ac71892a783acb2e785bd062fdf80905d`; structural historical reference: `4e967df9d03649106faa8348c3ec611009529ffe`.
- Runtime: Windows 11 `10.0.22631`, CPython `3.14.7`, `uv 0.9.17`.
- Production owners exercised: Self `_GenerationGuardedVadSink` → `SelfCaptureVadSinkAdapter` → `SelfTranslationChannelOwner` → `ScopedRecognitionEngine`; production `TranslationChannelOwnerCallbacks`; translation lifecycle/request/projection; `OutputRuntime` destination-batch admission; and `OverlayPresenter`. Peer owner coverage is retained.
- Controlled components: deterministic scoped STT session, synthetic float arrays, deterministic UUIDs, no-I/O chatbox, and deterministic translation provider with A/B delays of 180/40 ms (plus zero-delay-A control). No paid call, model, private speech, capture device, native renderer, or HMD was used.
- Self declared source schedule: 16 kHz; 2 context + 8 content samples per turn; 20 ms endpoint interval; A start/seal 0/40 ms; B start/seal 70/110 ms; isolated A terminal target 70 ms; successive immediate A/B terminal targets 40/150 ms; delayed A/B targets 160/200 ms. Actual real-loop delivery varies by scheduler; the final immediate/delayed B source deliveries differ by about 5 ms. Cross-case differences below 10 ms are not interpreted.
- Command:

  ```text
  uv run python experiments/issue_180/probe.py --output experiments/issue_180/trace.jsonl
  ```

- Result: exit 0; all focused assertions passed; 446 compact trace rows across eleven scenarios. Raw trace: [`trace.jsonl`](trace.jsonl). Runnable harness: [`probe.py`](probe.py).

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

Trace anchors: immediate [rows 275–306](trace.jsonl#L275-L306); delayed [rows 307–338](trace.jsonl#L307-L338).

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

Two new occupants fill two free slots at 0 logical ms. A same-logical-occupant update for occupant 2 and its real `utterance_closed` event both apply at 0 with no replacement wait. The occupant UUID, event type, source order, accepted handoff, and application receipt remain visible in [rows 339–358](trace.jsonl#L339-L358). This confirms original/update/close events for one logical turn are not automatically new replacement opportunities.

### Five-item ready burst

All five translations are ready and accepted into output at 0 logical ms. Destination batches are active immediately in this unmanaged-event fixture; therefore the observed destination-readiness wait is zero. The additional delayed-destination fixture was explicitly deferred as out of this repair's scope. The writer is strictly ordered.

| Item / logical source order | First presenter check | Application | Queue/inherited delay before presenter | Intentional replacement wait | Protection | Logical residual after eligible recheck |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0 ms (free slot) | 0 ms | 0 ms | 0 ms | 0 ms | 0 logical ms |
| 2 | 0 ms (free slot) | 0 ms | 0 ms | 0 ms | 0 ms | 0 logical ms |
| 3 | 0 ms | 1000 ms | 0 ms | 1000 ms | 0 ms | 0 logical ms |
| 4 | 1000 ms | 2000 ms | 1000 ms behind item 3 | 1000 ms | 0 ms | 0 logical ms |
| 5 | 2000 ms | 3000 ms | 2000 ms behind items 3–4 | 1000 ms | 0 ms | 0 logical ms |

The expected opportunity is reconstructed from actual occupant history: slots fill with 1 and 2 at 0; each later new occupant is eligible one second after the preceding new occupant (3 at 1000, 4 at 2000, 5 at 3000). The total 2–3 second delays are policy-driven ordered backlog, not a flat one-second subtraction and not evidence of an extra defect. Because `controlled_sleep` advances the injected clock and yields, zero logical residual here is **not** a wall-scheduling measurement. Live decisions and gates are [rows 359–398](trace.jsonl#L359-L398).

### Protected-row release and progress

With one slot, an actual `self_active_update` creates a protected visible row at 0 logical ms. The waiting Peer item records `protected_rows`. At 400 ms an actual `self_active_clear` goes through `OutputRuntime`, publishes revision 2, signals admission change, and frees the slot. The Peer item is rechecked, becomes eligible by the free-slot exemption, and receives application revision 3 at the same injected-clock instant.

Attribution is 400 logical ms of protection lifetime, zero observed queue/destination wait before the initial presenter check, and zero remaining replacement interval. Wall scheduling from eligibility to application is unmeasured in this injected-clock scenario. The item advances on its blocker-release wakeup; no lost wakeup or circular wait appears in logical state progression. See [rows 399–412](trace.jsonl#L399-L412).

### Head-of-line update/close contract

The focused head-of-line scenario fills two slots with source orders 1 and 2, then holds a source-order-3 new occupant at its one-second replacement gate. While the writer head is held, the probe submits an update and close for the already-visible order-2 occupant through `OutputRuntime.publish_overlay_event()` **without overriding identity**. `OutputRuntime` resolves the remembered production identity `(generation=1, source_order=2)`. Because `_latest_peer_source_order` is already 3, both events are rejected at admission as `stale_source_order`; they never become admissible queued work behind the paced head.

This grounds why a production-valid already-visible Peer update/close cannot be exercised *behind* the later paced head under the preserved source-order contract: to admit it with a higher order would change the logical turn's frozen source identity, which is not a valid profile. In valid ordering, that occupant's update/close must be admitted before the later source-order replacement and therefore precedes its wait. The independent Self clear in the protection scenario uses a separate path and progresses while the Peer writer waits. Evidence: [rows 413–433](trace.jsonl#L413-L433). This is a contract result, not a pacing bypass or fix proposal.

### Real-clock post-eligibility observation

A separate one-slot control uses the real monotonic clock and real `asyncio.sleep`. After the first occupant applies, the second is ready at 258 µs, begins its replacement wait at 344 µs, is rechecked eligible at 1,013,564 µs, and completes presenter application at 1,013,743 µs. The single observed ready-to-application interval is 1,013.485 ms; the observed eligible-recheck-to-application interval is 179 µs. Focused assertions bound these at 0.9–1.5 seconds and 0–100 ms respectively. This is sufficient to observe this boundary once, not a benchmark or general scheduler-latency claim. Evidence: [rows 434–447](trace.jsonl#L434-L447).

### Existing diagnostic-field limits

- `logical_pacing_wait`, `handoff_wait_ms`, `wait_reason`, and `pending_batches` are present in the live decision trace.
- For burst items 4 and 5, `handoff_wait_ms` is only 1000 although post-translation totals are 2000 and 3000 logical ms, because its timer begins at the first presenter wait after earlier writer/queue time.
- `wait_reason` retains only the first reason. In the protection case it reports `protected_rows` and 400 ms; it would not alone expose a later transition to `replacement_gate`.
- The probe-local gate rows fill the reason-transition observation gap without changing production behavior.

## C. Self end-to-end software latency

### Exercised path, fixture fidelity, and first-turn control

The Self scenarios enter production `_GenerationGuardedVadSink` with raw `SpeechStart`/`SpeechEnd`, then exercise `SelfCaptureVadSinkAdapter`, `SelfTranslationChannelOwner`, `ScopedRecognitionEngine`, `TranslationChannelOwnerCallbacks.self_event_handler`, translation lifecycle/request/projection, `OutputRuntime`, destination-batch admission, and `OverlayPresenter`. Trace rows correlate segment/source UUID, provider-turn ID, translation parent UUID, overlay publication ID, and occupant UUID; these identities are intentionally not collapsed into one field.

Three fixture substitutions bound the result:

1. `SelfDispatcherOwner` supplies `is_current_generation()` and failure recording to the production generation guard. A production `SelfCaptureSessionOwner.is_current_generation()` also requires `RUNNING` state and a live loop task. The fixture models an already-running, current generation; it does not cover start/stop, stale generation, or loop-task death.
2. `SelfEngineRuntimeBridge` sends the adapter's owned Self event to the production scoped engine. It omits `LocalASRProviderRuntimeOwner.handle_owned_vad_event()` channel validation, `_operation` gate, and provider resolution. The selected path is steady-state, channel=`self`, with one already-resolved deterministic session and no provider mutation. Claims therefore cover ordered dispatch/scoped-engine behavior in that state, not reconfiguration races or provider ownership.
3. The repaired probe now uses production `TranslationChannelOwnerCallbacks.self_event_handler()` rather than calling `SelfTranslationChannelOwner.handle_stt_event()` directly. It executes before/after session-state handling and records the event. A probe `SelfTerminalObserver` receives `note_recognition_terminal()` in place of a full capture owner and records that callback; it does not mutate capture-session state.

The scoped session double emits controlled terminals but does **not** model a serialized local decode worker, queue contention, model execution, or adapter buffer handoff. Consequently, all numeric lag absorption in these Self cases is specific to this double. Structural observations about the real current/historical local adapter remain separate below.

The isolated control supports **absence of predecessor blocking**, not a separate latency symptom:

| Isolated A boundary | Actual real-loop time |
| --- | ---: |
| Source available / provider begin | 2.423 / 2.698 ms |
| Acoustic last-sample offset / local seal | 20 / 51.653 ms |
| Provider terminal + engine release | 82.026 ms |
| Original `self_transcript_final` application | 82.570 ms |
| Translation start / completion | 82.890 / 262.923 ms |
| Translated `translation_final` application | 263.445 ms |

The declared isolated terminal target is 70 ms (successive immediate A uses 40 ms), and A translation is configured for 180 ms. Thus the isolated elapsed time is configured STT/translation delay plus one-shot scheduling, not demonstrated extra waiting. Original application follows engine release by 0.544 ms; translated application follows controlled completion by 0.522 ms. These are Python presenter receipts, not physical pixels. Evidence: [rows 2–40](trace.jsonl#L2-L40).

### Declared-schedule successive turns

| B boundary | Immediate A terminal, 180 ms A translation | Delayed A terminal, 180 ms A translation |
| --- | ---: | ---: |
| Source available / local seal | 74.315 / 119.256 ms | 79.531 / 124.931 ms |
| Dispatch + provider begin | 74.574 ms | 172.052 ms |
| B terminal + engine release | 164.231 ms | 200.721 ms |
| B original subtitle application | 241.721 ms | 353.226 ms |
| B translation start / completion | 241.942 / 281.957 ms | 353.438 / 393.456 ms |
| B translated subtitle application | 282.192 ms | 393.713 ms |

These cases have the same **declared** 70/110 ms B schedule, not identical physical delivery. Final B source delivery differs by 5.216 ms between cases; no sub-10 ms cross-case inference is made.

- **Capture/STT serialization:** production Self `_GenerationGuardedVadSink._run()` awaits the adapter, Self channel, bridge, scoped engine, and callback before dequeuing. Delayed A therefore holds queued B before the engine. B source-to-provider-begin age is 0.259 ms in the immediate case and 92.521 ms in the delayed case. B retains the same 2 context + 8 content samples. Its terminal timing is a consequence of this deterministic session's already-queued end/terminal schedule; it is not a real local-provider forecast.
- **Exact post-STT blocker:** after B engine release at 164.231 ms, `OutputRuntime.publish_overlay_event()` inserts B original publication `evt-2` at 164.436 ms as inactive in scope `self`, with active parent `8c4d6668-fb65-50ed-ba47-9e72465fbbd8` (A). A translation completes at 240.431 ms; A translated output and close apply; `complete_translation_parent_output()` reaches `complete_target()`, which releases A with disposition `applied` at 241.428 ms and activates successor `evt-2`. B original applies at 241.721 ms. The delayed case records the same chain: B `evt-2` parks at 200.913 ms, A releases `applied` and activates it at 352.934 ms, and B applies at 353.226 ms.
- **Why locks appear blocked:** `SelfTranslationChannelOwner._handle_transcript()` has entered the Self presentation context, so `_self_publish_lock`/`_self_surface_lock` remain held while the call propagates into `OutputRuntime` and awaits `DestinationBatch.ready` at `publish_overlay_event()`. They propagate the wait to other Self work but are not its initial blocker. The initial blocker is destination-batch admission's active A parent in scope `self`; `DestinationBatchAdmission.release()` is the exact wakeup.
- **Translation-delay control:** with the same declared immediate schedule and zero A translation delay, B release-to-original-application is 0.371 ms, versus 77.490 ms with 180 ms A translation. This supports attribution to the A parent batch lifetime, not Peer replacement pacing or baseline presenter application. Evidence: immediate [rows 41–118](trace.jsonl#L41-L118), zero-delay control [rows 119–196](trace.jsonl#L119-L196), delayed terminal [rows 197–274](trace.jsonl#L197-L274).

`TracingDestinationBatchAdmission` is probe-local observational instrumentation: it subclasses the production admission owner, calls production `_insert()`/`release()` unchanged, then records scope, active/waiting parent identities, disposition, and activated successor. Focused assertions require B to be inactive behind A and require A's `applied` release to activate B.

### Applicable boundaries and historical comparator

- Low-latency mode is off, so speculative selection, resume debounce, `low_latency_finalize_wait_ms`, and awaiting-VAD timeout are not on the exercised path. No negative claim is made about those policies.
- The applicable pending-final boundary is the scoped engine's terminal future plus the Self dispatcher's awaited end call. The subsequent B original-text wait is destination-batch admission. Self output calls `OverlayPresenter.emit()`, not Peer `emit_peer_when_admissible()`, so the one-second replacement gate and `protected_rows` are not applicable to these Self events.
- At historical `4e967df9d03649106faa8348c3ec611009529ffe`, the Self generation guard directly awaited the Self VAD adapter, and the adapter directly awaited `SelfTranslationChannelOwner.handle_vad_event()`. That owner reached `ManagedSTTProvider._on_speech_end()`, which recorded pending-final identity and awaited `session.on_speech_end()` but did not await the matching final event. The historical local Qwen session copied/enqueued its buffer and cleared it on speech end, allowing subsequent Self audio into a fresh buffer while `LocalDecodeCoordinator` serialized decode. Bounded structural conclusion: historical Self could overlap next-turn buffering with prior decode but did not run concurrent local decodes. This is not a historical runtime benchmark or regression claim.

## Limits, missing observations, and architecture drift

- Injected-clock Peer timings are causal schedule values. Self and Peer real-loop scenarios are one-shot observations, not latency distributions, performance benchmarks, or field-magnitude predictions.
- The deterministic Self session does not model serialized local decode work; the deterministic translation provider does not model a real provider. No model/provider comparison was performed.
- Controlled translation and destination-admission timing was measured through production owners. A real audio device, native overlay, VR runtime, HMD, and physical acknowledgement were not measured.
- Whether the maintainer's reported real-session magnitude reproduces with a real model/device remains unknown. `application_accepted` proves Python presenter acceptance only.
- No production source, policy, ownership, ordering, generation fence, content/expiry rule, or receipt meaning changed. Architecture drift: **none observed** relative to `docs/architecture.md`. No actual drift is suspected: the wait conforms to current destination-batch ownership; whether original Self text should share that scope is the maintainer decision above.
