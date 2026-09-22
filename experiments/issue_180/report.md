# Issue 180 — STT handoff and Peer display-delay diagnosis

## Scope and disposition

This is the bounded software-mechanism experiment requested by issue #180. It does not establish a real-session regression and does not measure physical overlay/HMD presentation.

| Question | Scoped disposition |
| --- | --- |
| Does a delayed local STT terminal delay the next recognition turn? | **Mechanism demonstrated.** In the exercised production Peer dispatch path, the ordered upstream worker remains inside A's `SpeechEnd` handler until A terminal/release. B is retained in the upstream queue and does not enter `ScopedRecognitionEngine.handle_owned_vad_event()` until then. With A terminal delayed 250 ms and an otherwise identical source schedule, B dispatch/begin/first write moved from 150 ms to 350 ms. No B audio was lost. |
| Does completed Peer output wait beyond the intended replacement policy? | **Not reproduced in these controlled cases.** The five-item burst accumulated the expected ordered replacement backlog (applications at 0, 0, 1000, 2000, and 3000 ms). A protected-row case advanced immediately when an actual `self_active_clear` released protection at 400 ms. No residual post-eligibility wait, repeated gate, or double pacing was observed. Existing diagnostic fields under-report total post-translation time for later burst items because their timer begins only when each item reaches the presenter. |

**Final next decision: maintainer architecture/policy decision.** Decide whether the correlation/terminal-safety contract should continue to serialize local Peer recognition through A terminal, or whether a separately authorized design should admit and buffer B at the provider boundary while A decode remains authoritative. The output evidence supports **no pacing implementation change** from this experiment; changing the one-second policy is a separate maintainer product decision.

## Frozen execution record

- Actual execution revision: `13274569769d3c1ec7a896a2d15b919b76136a6e`.
- Issue publication's inspected-current reference: `4daaeb4ac71892a783acb2e785bd062fdf80905d` (historical relative to this execution, not treated as the executed baseline).
- Structural historical reference: `4e967df9d03649106faa8348c3ec611009529ffe` (source comparison only; not executed as an application).
- Runtime: Windows 11 `10.0.22631`, CPython `3.14.7`, `uv 0.9.17`.
- Deterministic doubles only: an event-controlled scoped STT session, synthetic float arrays, deterministic UUID inputs, a controlled monotonic clock, a no-I/O chatbox, and the real `OutputRuntime`/`OverlayPresenter` owners. No provider call, model, private speech, capture device, native renderer, or HMD was used.
- Probe-local observation: `TracingPresenter` records the result of the real `_peer_replacement_delay()` and then calls the production method unchanged. It does not bypass pacing, ordering, protection, receipt, or ownership checks.
- Command:

  ```text
  uv run python experiments/issue_180/probe.py --output experiments/issue_180/trace.jsonl
  ```

- Result: exit 0; 141 compact trace rows across five scenarios. Raw trace: [`trace.jsonl`](trace.jsonl). Runnable harness: [`probe.py`](probe.py).

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

Trace anchors: immediate [rows 2–33](trace.jsonl#L2-L33); delayed [rows 34–65](trace.jsonl#L34-L65).

### Attribution

- **A terminal wait:** 0 ms in the immediate case versus 250 ms from A seal to terminal in the delayed case.
- **B source-to-dispatch/admission wait:** 0 ms versus 200 ms. In the delayed case B `SpeechStart` is available at 150 ms but its production-dispatch attempt is at 350 ms.
- **Where the wait occurs:** `_GenerationGuardedVadSink._run()` awaits each sink call before dequeuing the next event (`core/runtime/peer_channel.py`). A's call reaches `ScopedRecognitionEngine._handle_end()`, whose `seal_turn()` is followed by `_await_terminal()` and `_finish_turn()`. Therefore the production dispatcher is still awaiting A's end handler. B does **not** reach the engine's explicit `await resolved.wait()` loop in this path; that loop is a second safety boundary for concurrent direct callers.
- **B provider work versus readiness:** the controlled session applies the same 50 ms post-B-seal terminal delay in both cases. B terminal/readiness moved only 50 ms later (350 → 400), not the full 200 ms start shift, because B's source end was already queued by 300 ms and was dispatched immediately behind its delayed start at 350 ms. This fixture therefore demonstrates partial absorption of the initial lag rather than assuming a later start implies an equally later final.
- **Buffered content:** B retains and writes the same two-sample context payload plus eight-sample content payload in both cases. In the delayed trace, those writes carry original source ranges ending at 150 and 300 ms, and B end reaches dispatch with source age 50 ms. This is retained audio, not silence or loss. The experiment does not evaluate ASR accuracy.
- **Correlation:** every trace row retains segment order; provider begin/write/terminal rows retain the generated provider-turn ID. Engine terminal release is matched to the same segment identity.

### Current and historical adapter verification

This is not an overlap claim based on controller shape alone.

- At the executed revision, `ScopedRecognitionEngine._handle_end()` waits for the terminal before `_finish_turn()` sets `_turn_resolved`. The selected current local adapter, `providers/stt/local_qwen_sherpa.py`, clears/collects the turn buffer in `seal_turn()`, enqueues a `LocalDecodeCoordinator` job, and emits the scoped terminal only from `_handle_decode_completion()` (or a failure/expiry path). Thus, for this adapter, the engine's terminal wait includes queue/decode completion rather than only a cheap seal acknowledgement.
- At historical `4e967df9d03649106faa8348c3ec611009529ffe`, `ManagedSTTProvider._on_speech_end()` appends pending-final identity and awaits `session.on_speech_end()` but has no controller-level wait for the corresponding final before a later `_on_speech_start()` sends audio. The historical `LocalQwenSherpa` adapter's `on_speech_end()` copies/enqueues the current buffer and clears it; its next-turn `send_audio_f32()` can populate a fresh buffer while `LocalDecodeCoordinator` and the backend decode lock serialize actual inference. **Structural comparator conclusion:** historical code allowed next-turn capture/buffering to overlap preceding decode, but did not make two local decodes concurrent. This is not a historical runtime benchmark.

## B. Peer output and presenter pacing

All output scenarios exercise the production `OutputRuntime.publish_overlay_event()` admission, ordered Peer overlay writer, destination batch readiness, `OverlayPresenter.emit_peer_when_admissible()`, state reduction, and `OverlayApplicationReceipt`. `physical_ack=false` is preserved: application acceptance is not a renderer/HMD acknowledgement.

### Replacement control

Two new occupants fill two free slots at 0 ms. A same-logical-occupant update for occupant 2 and its real `utterance_closed` event both apply at 0 ms with no replacement wait. The occupant UUID, event type, source order, accepted handoff, and application receipt remain visible in [rows 66–85](trace.jsonl#L66-L85). This confirms original/update/close events for one logical turn are not counted as new replacement opportunities.

### Five-item ready burst

All five translations are ready and accepted into output at 0 ms. Destination batches are active immediately in this unmanaged-event fixture; therefore there is no separate destination-readiness wait. The writer is strictly ordered.

| Item / logical source order | First presenter eligibility check | Application receipt | Queue/inherited delay before presenter | Intentional replacement wait for this item | Protection | Residual after eligibility |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0 ms (free slot) | 0 ms | 0 ms | 0 ms | 0 ms | 0 ms |
| 2 | 0 ms (free slot) | 0 ms | 0 ms | 0 ms | 0 ms | 0 ms |
| 3 | 0 ms | 1000 ms | 0 ms | 1000 ms | 0 ms | 0 ms |
| 4 | 1000 ms | 2000 ms | 1000 ms behind item 3 | 1000 ms | 0 ms | 0 ms |
| 5 | 2000 ms | 3000 ms | 2000 ms behind items 3–4 | 1000 ms | 0 ms | 0 ms |

The expected opportunity is reconstructed from actual occupant history: slots fill with 1 and 2 at 0; each later new occupant is eligible one second after the preceding new occupant (3 at 1000, 4 at 2000, 5 at 3000). The total 2–3 second delays are policy-driven ordered backlog, not a flat one-second subtraction and not evidence of an extra defect. Compact event/gate trace: [rows 86–112](trace.jsonl#L86-L112). Production diagnostic snapshots: [rows 113–125](trace.jsonl#L113-L125).

### Protected-row release and progress

With one slot, an actual `self_active_update` creates a protected visible row at 0 ms. The waiting Peer item records `protected_rows`. At 400 ms an actual `self_active_clear` goes through `OutputRuntime`, publishes revision 2, signals admission change, and frees the slot. The Peer item is rechecked, becomes eligible immediately by the free-slot exemption, and receives application revision 3 at 400 ms.

Attribution is therefore 400 ms protection lifetime, 0 ms queue/destination wait before the initial presenter check, 0 ms remaining replacement interval, and 0 ms post-eligibility residual. The item advances as soon as its actual blocker clears; no lost wakeup or circular wait appears. See [rows 126–142](trace.jsonl#L126-L142).

### Head-of-line and diagnostic-field limits

- The Peer writer is head-of-line by design while a new occupant waits, so later Peer batches cannot overtake it. Under valid source ordering, predecessor update/close events are admitted before a later source-order replacement and therefore precede that wait. The control confirms update/close progress in that order.
- A diagnostic negative control submitted older source-order update and close events after the newer item. Both were rejected at output admission as `stale_source_order`; they were not silently held behind the writer head. This deliberately invalid ordering is a check of protection/order semantics, **not** a proposed bypass or supported production sequence.
- The independent Self clear progressed while the Peer writer was waiting and released protection, showing the blocked Peer head did not block the destination's non-Peer release path.
- No otherwise-admissible visible-Peer update or progress-releasing close was observed stuck behind a blocked head in the valid-order scenarios. This is a bounded negative result, not proof for every production interleaving.
- `logical_pacing_wait`, `handoff_wait_ms`, `wait_reason`, and `pending_batches` behaved as documented. For burst items 4 and 5, `handoff_wait_ms` is only 1000 even though post-translation totals are 2000 and 3000 ms, because timing begins at the item's first presenter wait after earlier writer/queue time. `wait_reason` retains only the first reason. In the protection case it reports `protected_rows` and 400 ms; it would not by itself expose a later transition to `replacement_gate`. The probe-local gate rows fill that observation gap without changing production behavior.

## Limits, missing observations, and architecture drift

- The numeric timings are a controlled-clock causal schedule, not latency distribution, wall-clock performance, or a prediction of field magnitude.
- Local provider work is deterministic; no model/provider comparison was performed. The selected adapter was verified structurally so overlap claims are bounded correctly.
- No upstream translation-runtime timing, real provider, audio device, native overlay, VR runtime, HMD, or physical acknowledgement was measured. Whether the maintainer's reported real-session slowdown occurs, its magnitude, and physical HMD latency remain unknown.
- `application_accepted` proves Python presenter acceptance only.
- No production source, policy, ownership, ordering, generation fence, content/expiry rule, or receipt meaning changed. Architecture drift: **none observed** relative to `docs/architecture.md`; the exercised ownership is Peer capture dispatch → scoped recognition owner and output runtime → presenter owner.
- No core probe was blocked. Destination readiness was immediate in the selected fixture; a deliberately delayed destination-ready parent was not added because the issue asks causal separation, and the observed burst already separates pre-presenter ordered queue delay from presenter pacing without creating a new framework.
