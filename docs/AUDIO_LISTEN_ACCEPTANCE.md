# AUDIO_LISTEN_ACCEPTANCE

## Authority, candidate, and installed composition

- Implementation issue: [#139](https://github.com/kapitalismho/PuriPuly-heart/issues/139), body updated `2026-09-11T10:52:15Z`; the installed pre-SELF-cutover scope is governed by `PRE-SELF-CUTOVER-ISOLATION`.
- Canonical contract: [#134](https://github.com/kapitalismho/PuriPuly-heart/issues/134), `AUDIO-LISTEN-1` with the `SMART-TURN-COMMON-075-NOHASH` amendment.
- Production-source baseline: `56404af0fc152a1d66a4140152df690be87d2cec`. Checkpoints `60a394e254a97561b79453c390ef857fdab3ffc6` and `659736c46751b613ee86e91a6fa70bf6c4295844` were not accepted. Final production-watchdog, reconnect and writer-retirement test content is pinned to `6f9c07d5999ce40d35cc18768ce38973018ffcbf`; this later receipt-only revision introduces no product or test changes. Earlier successful-final evidence that enlarged the writer timeout is superseded and is not acceptance evidence.
- Accepted LISTEN core: #135 at `41210e2fd36874280996526a83b3039e0618b53a`; receipt `docs/AUDIO_CORE_ACCEPTANCE.md`.
- Accepted Smart Turn implementation: #136 shipped at the working baseline; receipt `docs/AUDIO_SMART_TURN_ACCEPTANCE.md` and [accepted handoff](https://github.com/kapitalismho/PuriPuly-heart/issues/136#issuecomment-5632724285).
- Installed SELF path: the existing legacy `ManagedSTTProvider` projection. #143 shared extraction and #144 SELF policy migration are not installed. The superseded intrinsic SELF nonblocking criterion remains historically unmet; the amended cross-channel isolation criterion is assessed on the installed composition.
- Environment: Windows 11 x64, Python 3.12.10, PuriPuly Heart 2.6.1, NumPy 2.5.1, sounddevice 0.5.5, PyAudioWPatch 0.2.12.8, proc-tap 1.1.1, soxr 1.1.0, httpx 0.28.1, websockets 16.1.1, google-genai 2.21.0, deepgram-sdk 5.3.4, elevenlabs 2.65.0, dashscope 1.26.4, sherpa-onnx 1.13.4, and ONNX Runtime 1.28.0 with `CPUExecutionProvider` available.

The installed production composition has one `PeerCaptureSessionOwner`, one LISTEN delivery controller, scoped LISTEN recognition through `ScopedRecognitionEngine`, the existing translation lifecycle owner, and the existing output runtime. SELF remains on its legacy event projection. This describes implementation, not Director acceptance. No second LISTEN engine, replacement writer, replay layer, PSEM producer, generic buffering platform, or partial SELF migration was added. The #141 Gemini executor/client/teardown implementation and its lifecycle tests are preserved.

## Historical diagnosis and acceptance result

#126 observed a blocked channel consumer followed by capture-queue pressure, including 127 SELF and 32 desktop-peer dropped callback blocks, and separately identified synchronous SDK/native work blocking a shared event loop. Those values were callback-block counts, not seconds, words, WER, or individual loss timestamps. Corrected Gemini setup runs off the event loop. However, a legacy SELF acoustic consumer still directly awaits `ManagedSTTProvider.handle_vad_event`; when that call opens a new SDK session, the shared event loop and LISTEN may progress, but that SELF consumer does not consume another frame until setup returns.

Consequently the original #139 reverse “both actual consumers progress” row remains historical and unmet, while amended `PRE-SELF-CUTOVER-ISOLATION` requires opposite-channel and common-loop progress in both setup directions plus LISTEN own progress. #143 R5 forbids importing LISTEN duration bounds into unbounded SELF, while #144 S2/S6/S7 owns the shared-ready cutover, finite 2,880,000-sample-equivalent SELF policy, and nonblocking acoustic consumer. No #139-local dispatcher or new SELF admission policy is introduced.

## Capture and route inventory

### Capture and normalized source paths

| Input | Installed path | Tested/configured facts |
| --- | --- | --- |
| SELF microphone | `SelfCaptureSessionOwner` -> self capture adapter -> legacy provider projection | Actual pre-#144 SELF behavior; separate generation, source, VAD and provider lifecycle. |
| Desktop LISTEN | PyAudioWPatch WASAPI loopback -> `DesktopAudioCaptureSource` -> mono-first streaming resampler -> 16 kHz peer owner | Default callback queue 64 blocks; source sequence is assigned before admission; default-device fallback is reported. |
| Process LISTEN | proc-tap 1.1.1, 48 kHz stereo -> `ProcessAudioCaptureSource` -> mono-first streaming resampler -> 16 kHz peer owner | Default callback queue 64 blocks; missing selected process is not replaced by desktop capture. |
| Current workstation discovery | sounddevice enumerated MME, DirectSound, WASAPI and WDM-KS; defaults reported input index 1 and output index 5 | Discovery only. No physical microphone, loopback, process target, or driver scheduling certification was run. Device names are omitted from this privacy-safe receipt. |

Known callback loss advances source coordinates and is attached to the next admitted frame. Unknown loss starts an explicit discontinuity/capture epoch. The actual-owner gap regressions preserve already emitted PCM, fail unsafe open ownership, reject stale Smart Turn evidence, and prevent pre-gap context from being concatenated with post-gap content. Orderly EOF flushes genuine resampler residue; abort and discontinuity do not manufacture silence.

### All 15 configured selectors

| Selector | Concrete LISTEN resolution and completion authority |
| --- | --- |
| `local_cpu_auto` | Catalog/language-selected CPU delegate; one scoped decode job. |
| `local_parakeet_v3` | Parakeet transducer CPU adapter; decode completion/failure. |
| `local_parakeet_ja` | Parakeet Japanese CTC CPU adapter; decode completion/failure. |
| `local_qwen` | Local Qwen CPU adapter; decode completion including explicit hallucination suppression. |
| `local_qwen_gpu` | Shared GPU worker request and device coordinator; scoped worker completion. |
| `deepgram` | Stable fragments, actual ordered Finalize write, `from_finalize` acknowledgement, then bounded CloseStream/second drain if absent. |
| `gemini_transcribe` | Authoritative input transcription and ActivityEnd/protocol acknowledgement as separate barriers; declared timeout fallback only. |
| `elevenlabs_scribe` | Manual commit and committed transcript, including authoritative empty. |
| `soniox` | Ordered final tokens/language runs and the pending manual finalize's `<fin>`. |
| `qwen_asr` | Native commit/item identity and item completion/failure. |
| `qwen_audio` | Native task identity, stable sentence assembly, and `task-finished`. |
| `rolling_free` | Configured Scribe/Gemini/Deepgram member order, quota and cooldown; switch creates a provider epoch. |
| `custom` | Alias resolved through stored mode to one of the two concrete custom contracts. |
| `custom_offline` | One sealed PCM HTTP request; HTTP/parse failure is not empty. |
| `custom_realtime` | Client commit and keyed item completion where present; unkeyed completion retires the epoch. |

Factory coverage resolves every selector to its route-specific watchdogs. Aliases and rolling selection do not acquire a generic LISTEN fallback. The installed custom-offline acceptance regression holds the real HTTP request while a later acoustic episode is captured and sealed by `PeerCaptureSessionOwner`; both 1,024-sample segments remain ordered, then both receive authoritative finals after release.

## Deterministic pressure matrix

| Pressure point | Evidence and result |
| --- | --- |
| Synchronous SDK setup | Corrected Gemini resource construction runs in a worker thread. The actual-owner matrix holds initial setup and running-session handoff/reconnect in both directions. During each hold the opposite `SelfCaptureSessionOwner` or `PeerCaptureSessionOwner` source and a common event-loop sentinel continue advancing; during running handoffs the held channel's already-open source also continues. During SELF's initial setup its source is not yet open, preserving the explicit pre-#144 limitation. All opened sources close under their owners. |
| Asynchronous initial open | Peer-owner tests consume and seal source content before provider ingress becomes ready. Eight wholly unsent sealed segments remain the exact admission budget; overflow and TTL produce explicit terminal receipts. |
| Actual writer after SDK enqueue | The owner regression uses the production Soniox watchdog resolver with its unchanged `write_timeout_s=5`; the test supplies `drain_timeout_s=0.1` for cleanup and does not certify production drain timing. It holds the scoped session's actual `send_turn_audio`. For Smart Turn OFF and shipped ON, the independent hard timer seals normalized `[0,92160)` at six seconds within the asserted `[-0.020,+0.250)` controlled-runtime lag envelope. The native scoped engine then delivers `provider_send_timeout` through canonical owner ingress; no fabricated final crosses that terminal. After the blocked native write is released and cleanup debt reaches zero, a fresh provider epoch accepts a later capture, seals it by pause before the six-second hard limit, delivers an authoritative final through the engine and owner callbacks, and produces exactly one downstream peer final. |
| Missing/delayed finalization | Later capture seals independently while dispatch is held. Route suites cover Deepgram two-drain, Gemini's two barriers, Soniox `<fin>`, Scribe commit, Qwen item/task barriers, duplicate/late messages, EOF and send/end/drain failures. |
| Idle rotation/expiry/reconnect | Idle epoch end after empty or final A creates a fresh epoch for B; stale/foreign epoch-end callbacks cannot retire B. Healthy 180 s rotation occurs only at a barrier. Recovery is exactly three attempts with 0.8 s then 1.6 s backoff. Three sessions that reach ready and then fail before any successful turn exhaust that same episode; a fourth segment cannot reset it by reopening. |
| Custom offline HTTP | New actual-owner regression holds the HTTP coroutine while all later source frames are consumed and the second segment seals. Releasing HTTP admits finals in source order. Production phase bounds remain 5/30/10/5 s with a 50 s logical total. |
| Local decode/native stall | CPU and GPU suites distinguish pending 12 s expiry from the 30 s active decode timeout and physical capture loss. A busy native resource remains quarantined; logical expiry does not free it or allocate a replacement. |
| Slow translation/destination | Translation's eight-parent/12 s waiting envelope, per-child finite watchdog, source-only expiry, predecessor release, and output's independent destination lanes are covered. Provider/capture progress does not await LLM or sink completion. |

Within the controlled LISTEN envelope, the demonstrated cases lose no accepted source interval. Outside it, callback gaps, recognition overload/expiry, provider buffer overflow, semantic expiry and destination overload have distinct failed/expired/rejected receipts. The full default callback queue case injects 66 four-sample callbacks into capacity 64: 65 are admitted/consumed, one known four-sample interval is missing, the successor is `[260,264)`, and `unknown_discontinuity_count` remains zero. No test enlarges a production capacity. The unresolved legacy SELF setup-consumer row is outside this claim.

## Exact bounded envelope and observed boundaries

| Resource | Current bound and terminal disposition exercised |
| --- | --- |
| Physical callback data | 64 blocks by default; callback nonblocking; known drop range or unknown discontinuity survives a full data queue. |
| Recognition waiting | Eight wholly unsent sealed segments, excluding provider-active and current capture; oldest wholly unsent retires as `overload`. |
| Recognition freshness | 12 s from local seal; `expired_before_recognition`. Five fresh segments spanning more than 12 aggregate seconds remain valid because age is per seal. |
| Recognition control | 32 reserved terminal/control events; the 33rd is rejected without evicting admitted PCM. |
| Provider events | 256; provisional snapshots coalesce first, stable/terminal pressure fails and retires the scoped epoch. |
| Text and language runs | 1 MiB UTF-8 assembly and 256 runs; oversize text fails, invalid run metadata falls back to complete unknown-language text. |
| Translation waiting | Eight not-started peer parents and 12 s from parent admission; oldest retires source-only/expired and releases order. |
| Output handoff | Eight unsent parent batches plus one writer per destination; oldest unsent is `output_overload`; another destination remains independent. |
| Output identity | In-flight plus 4,096 completed peer publication IDs; generation/source order still rejects stale callbacks after cache eviction. |
| Smart Turn | One executing worker, no pending queue; busy skips the pause and occupied native work is not replaced. |
| PCM | Current capture plus provider-active plus eight wholly unsent sealed slots, each at most six seconds of source content, plus snapshotted prefix/context. |

The earlier `9d0c36de` smoke and suite claims are superseded and are not acceptance evidence because that regression enlarged the production write watchdog and injected a final outside the scoped engine. The repaired isolated run uses the production route resolver: OFF and ON each hard-seal `[0,92160)` while the writer is blocked, observe the real five-second `provider_send_timeout`, release the quarantined native write, then admit a fresh scoped session whose authoritative final reaches semantic/output ownership. Exact current-run commands and results are recorded below; no physical-device or vendor call is implied.

Applicable accepted #136 scheduling evidence at this same source baseline recorded two OFF local seals at wall offsets 0.0021866 s and 0.0044042 s, with 0.0022176 s between seals, provider B injected before A at 0.0044930 s, translation idle at 0.0052395 s and output idle at 0.0053444 s. Those are controlled scheduling observations, not physical-device certification or source-to-display latency. Deterministic owner tests separately seal the real accepted frontier when no callback arrives and exercise the <=32 ms frame assumption; the <=32 ms production scheduler half of C5's 64 ms allowance remains unmeasured on a physical capture device.

## Lifecycle, replacement, and simultaneous channels

- Simultaneous cloud setup: the actual `LocalASRProviderRuntimeOwner` admits independent SELF/Deepgram and LISTEN/Soniox constructions concurrently. Both remain pending at the injected barriers; release installs each only in its channel. Peer release does not evict SELF.
- Corrected setup isolation: actual `SelfCaptureSessionOwner` and `PeerCaptureSessionOwner` are run in both held-setup directions. The common loop and opposite source consumer advance in each case; LISTEN also consumes during its own held setup. During SELF's own held setup its source is not yet open/consuming, which is the explicit pre-#144 disposition rather than a failed amended criterion.
- Shared GPU: SELF and LISTEN activate on one GPU runtime and one device. While SELF VAD ingress is held, peer release completes, deactivates only peer, leaves SELF and its model resident, allocates no second runtime, and closes the physical runtime once at owner shutdown.
- CPU/local transitions retain the selected model lease, serialize decode on the actual lease, reject device changes requiring quiescence, and prevent late delegate/session construction from resurrecting a closed owner.
- Continuous rollover keeps acoustic continuity and source order while old provider work drains. Exact age-step, pause and hard-seal ties collapse to one local seal; hard timers seal only accepted source ranges.
- LISTEN abort/re-enable now has an actual `PeerCaptureSessionOwner` plus scoped-engine regression: OFF occurs while `send_turn_audio` remains unresolved, a fresh generation and fresh scoped engine are active before that old write completes, and a late old authoritative callback is rejected by native owner admission. Releasing the old write yields once-only stop/close, zero scoped cleanup debt, one accepted new-generation final and no old-generation output. Other phase-specific generation tests remain separate evidence rather than substitutes for this writer phase.
- Known/unknown gaps, source format changes, device/process restart and superseded target resolution end the affected capture generation. A selected missing process never falls back to desktop.
- Empty, failed, expired, cancelled and suppressed recognition each retire one source-order slot. A later successful segment waits only for that terminality, not for nonexistent text. Legitimate identical later speech is not deduplicated.
- Slow predecessor translation always releases its semantic gate on final, failed, cancelled, source-only or expiry. Capture and provider work continue independently.
- Output denies peer chatbox, records accepted handoff separately from destination delivery, rejects retired generations, protects a newer active caption, and reports destination overload/failure without replaying recognition or translation.
- OFF and supported-ON pressure use the same LISTEN ledger, dispatch owner, production route watchdogs and scoped engine. Both prove the canonical six-second hard boundary under a five-second write timeout. Their recovered successor pause ranges differ according to the shipped endpoint profiles (`39` frames OFF and `35` frames ON from source sample `92160`); neither policy, timeout nor capacity is enlarged.
- PSEM is absent in the base. Injected prospective hypotheses cover accepted, already-separated, too-late, invalid, duplicate and retracted receipts with at most one seal; no producer or second timer is enabled.

A cancelled await is not treated as a killed native thread. The dedicated OFF/re-enable regression keeps the old scoped writer unresolved across the new generation, rejects its late terminal at owner admission, and then proves finite reclamation. The separate timeout/recovery regression observes a failed first provider epoch/turn and an authoritative final from a fresh epoch/turn, with once-only source/session cleanup and zero debt.

## Protocol evidence class and external limitations

Concrete adapter tests use actual installed SDK message classes where available: Deepgram, Gemini and Scribe SDK shapes plus documented Soniox payloads. Qwen, custom, local CPU and local GPU suites exercise their concrete request/item/task/decode contracts with controlled transports or runtimes. These are deterministic application/protocol-handling checks, not vendor wire certification.

Environment inspection found `DEEPGRAM_API_KEY`, `GEMINI_API_KEY`, `ELEVENLABS_API_KEY`, `QWEN_API_KEY`, `DASHSCOPE_API_KEY`, `SONIOX_API_KEY`, and `OPENAI_API_KEY` unset. Integration is opt-in. Therefore no paid provider call was made, and live Deepgram, Gemini, Scribe, Soniox, Qwen, rolling-member, or custom-server conformance is not claimed. No physical microphone/loopback/process capture, live GPU model decode, Smart Turn model rerun, live LLM, Flet window, overlay process, VRChat chatbox, or remote display acknowledgement was exercised for #139. These remain explicitly untested, not PASS and not removal of support.

## Evidence join and diagnostics

The pressure case joins channel (`peer`), activation generation, capture epoch, exact source/normalized ranges, segment ID/order, provider epoch/turn, actual writer state, local seal, terminal receipt, downstream final and cleanup debt without recording PCM or transcript text. A minimal test-only sampler reads the existing guarded owner queue accounting during the held writer: both profiles reach an exact high-water of `179` waiting events holding `92160` PCM/content samples and zero context samples; the oldest sealed event's waiting age is sampled from its dispatch timestamp (observed `0.0 s` at the controlled clock resolution). Queue IDs/orders are asserted against the failed segment frontier and terminal receipt. No LISTEN telemetry field or legacy `ManagedSTTProvider` diagnostic is claimed.

A sink coroutine or queue acceptance proves only local handoff. It is not physical overlay/UI display. Raw PCM, transcript text, API keys and device names are not added to default diagnostics. This repair adds no synchronous per-frame logger or observability subsystem.

## Verification and AC mapping

Isolated-clone commands use the original environment interpreter and explicit isolated imports:

```text
PYTHONPATH=<isolated-clone>/src <original-.venv>/Scripts/python.exe -m pytest tests/core/runtime/test_peer_capture_session.py -q
PYTHONPATH=<isolated-clone>/src <original-.venv>/Scripts/python.exe -m pytest tests/core/runtime/test_peer_capture_session.py -k "actual_capture_owners_isolate_cross_channel_setup_pressure or canonical_delivery_boundaries_survive_production_write_timeout_and_recover or off_then_reenable_during_actual_scoped_write_stall" -q
PYTHONPATH=<isolated-clone>/src <original-.venv>/Scripts/python.exe -m pytest -q --ignore=tests/integration
```

The repaired peer-owner file passed all **51** cases in `17.37 s`; the seven repaired setup/handoff, timeout/recovery and OFF/re-enable cases passed together in `15.58 s`. The final integration-excluded isolated full suite exited successfully in `194.76 s`; its eight existing skips and 267 warnings (266 Flet deprecations plus one existing unawaited API-key verification warning) remain explicit. Import-path assertions resolved `peer_channel.py`, `scoped_engine.py`, and `listen_delivery.py` under the isolated clone's `src`, excluding the original moving worktree. A throwaway evidence print was removed byte-for-byte after capturing both profiles, and the final suite used the strengthened exact high-water assertion.

| Acceptance | Result and evidence |
| --- | --- |
| AC01 | Production baseline, rejected checkpoint and final repair test-content revision are pinned; the isolated committed suite excludes concurrent unrelated product changes. Selectors and legacy UI/OSC meaning remain unchanged. |
| AC02 | Candidate evidence: prefix/onset, finite tail, continuous rollover and no empty successor are covered by source and owner suites. |
| AC03 | Candidate evidence: OFF and shipped ON preserve the independent canonical six-second timer under the production five-second write watchdog; the timeout is terminal, and a later fresh scoped session seals and completes authoritatively. |
| AC04 | Candidate evidence: the default 64-block callback queue has an exact 66-callback overflow case with 260 consumed, four known missing, and unknown count zero. |
| AC05 | Candidate evidence under `PRE-SELF-CUTOVER-ISOLATION`: actual owners prove opposite-channel/common-loop progress during initial setup and running handoff/reconnect in both directions, including both already-running sources during handoff. Intrinsic legacy SELF own-initial-setup consumption remains assigned to #144. |
| AC06 | Existing evidence only: every selector resolves; concrete route suites cover final/empty/error/timeout/EOF/duplicate with external limits explicit. |
| AC07 | Existing evidence: Gemini barrier reorder/missing/late-A and Deepgram no-ACK two-drain. |
| AC08 | Existing evidence: three-attempt recovery, idle/age rotation, rolling member changes, expiry and permanent errors are finite and do not replay uncertain audio. |
| AC09 | Candidate evidence: actual setup/final, scoped writer, shared decode, translation and output owners each re-enable before retired work/callback completion; generation admission rejects old publication and cleanup remains finite. |
| AC10 | Candidate joined evidence: exact capacities/TTL/control/text/output/cache/debt cases remain; the repaired pressure case joins capture→writer→failed recognition and fresh recognition→semantic→output, while separate generation tests own retirement. |
| AC11 | Existing evidence: complete text/run conservation, repetition, empty/suppressed/failure terminality, stable children and source order. |
| AC12 | Existing evidence: source-only, predecessor release, destination rejection, caption priority, truthful handoff and chatbox denial. |
| AC13 | Existing evidence: prospective receiver receipts and one-seal arbitration; PSEM producer absent. |
| AC14 | Existing evidence: next-segment endpoint snapshots, old-config drain, successful/failed handoff and requested/effective state. |
| AC15 | Candidate pre-cutover composition evidence: #141 retained; actual two-direction capture-owner setup isolation, runtime cloud overlap and shared GPU ownership pass; shipped ON runs under the same actual-writer pressure. Final post-cutover both-consumer guarantee remains #144. |

## Completion disposition

The installed LISTEN ownership path remains the only production peer recognition path, and the repaired pressure, lifecycle and evidence-join cases above are candidate acceptance evidence under amended `PRE-SELF-CUTOVER-ISOLATION`. The amended F1 setup/isolation row is demonstrated while the original intrinsic SELF criterion remains historical/unmet for #144. #136 remains previously shipped and accepted. Final independent review is required before Director acceptance. No partial SELF migration or policy is introduced here.

No model quality, WER, multilingual superiority, physical-device scheduling, vendor SLA, exactly-once remote display, or future shared/SELF migration claim is made. The pre-existing `AGENTS.md` change is untouched.
