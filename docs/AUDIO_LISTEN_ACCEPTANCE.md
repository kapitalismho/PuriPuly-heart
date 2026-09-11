# AUDIO_LISTEN_ACCEPTANCE

## Authority, candidate, and installed composition

- Implementation issue: [#139](https://github.com/kapitalismho/PuriPuly-heart/issues/139), body updated `2026-09-09T08:13:04Z`.
- Canonical contract: [#134](https://github.com/kapitalismho/PuriPuly-heart/issues/134), `AUDIO-LISTEN-1` with the `SMART-TURN-COMMON-075-NOHASH` amendment.
- Working baseline and tested source revision: `56404af0fc152a1d66a4140152df690be87d2cec`.
- Accepted LISTEN core: #135 at `41210e2fd36874280996526a83b3039e0618b53a`; receipt `docs/AUDIO_CORE_ACCEPTANCE.md`.
- Accepted Smart Turn implementation: #136 shipped at the working baseline; receipt `docs/AUDIO_SMART_TURN_ACCEPTANCE.md` and [accepted handoff](https://github.com/kapitalismho/PuriPuly-heart/issues/136#issuecomment-5632724285).
- Installed SELF path: the existing legacy `ManagedSTTProvider` projection. #143 shared extraction and #144 SELF policy migration were not installed and are not prerequisites for this acceptance.
- Environment: Windows 11 x64, Python 3.12.10, PuriPuly Heart 2.6.1, NumPy 2.5.1, sounddevice 0.5.5, PyAudioWPatch 0.2.12.8, proc-tap 1.1.1, soxr 1.1.0, httpx 0.28.1, websockets 16.1.1, google-genai 2.21.0, deepgram-sdk 5.3.4, elevenlabs 2.65.0, dashscope 1.26.4, sherpa-onnx 1.13.4, and ONNX Runtime 1.28.0 with `CPUExecutionProvider` available.

The accepted production composition has one `PeerCaptureSessionOwner`, one LISTEN delivery controller, scoped LISTEN recognition through `ScopedRecognitionEngine`, the existing translation lifecycle owner, and the existing output runtime. SELF remains on its legacy event projection. No second LISTEN engine, replacement writer, replay layer, PSEM producer, or generic buffering platform was added. The #141 Gemini executor/client/teardown implementation and its lifecycle tests are preserved.

## Historical diagnosis and acceptance result

#126 observed a blocked channel consumer followed by capture-queue pressure, including 127 SELF and 32 desktop-peer dropped callback blocks, and separately identified synchronous SDK/native work blocking a shared event loop. Those values were callback-block counts, not seconds, words, WER, or individual loss timestamps. This acceptance does not recreate the obsolete synchronous Gemini start premise: it verifies the corrected executor-owned Gemini setup, actual legacy SELF compatibility, independent LISTEN ownership, and cross-channel runtime composition.

The deterministic pressure matrix passes on the installed composition. One implementation hardening regression was added for a custom-offline network wait through the actual peer capture owner and scoped engine. Two cross-channel resource regressions were added for simultaneous cloud construction and shared GPU channel ownership. No product limit, timing, retry, endpoint, or model policy changed.

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
| Synchronous SDK setup | Gemini ownership tests run corrected synchronous SELF resource construction in a worker thread while the scoped LISTEN engine and both controlled capture loops progress. Cancellation leaves one owned late resource and releases client plus sync/async transports once. |
| Asynchronous initial open | Peer-owner tests consume and seal source content before provider ingress becomes ready. Eight wholly unsent sealed segments remain the exact admission budget; overflow and TTL produce explicit terminal receipts. |
| Actual writer after SDK enqueue | Scoped-engine tests hold begin/payload/seal independently and prove final waiting starts only after the actual seal writer completes. Gemini and Soniox concrete send-loop tests hold the native writer after queue admission; enqueue is not writer completion. The no-existing-bound writer ceiling remains 5 s. |
| Missing/delayed finalization | Later capture seals independently while dispatch is held. Route suites cover Deepgram two-drain, Gemini's two barriers, Soniox `<fin>`, Scribe commit, Qwen item/task barriers, duplicate/late messages, EOF and send/end/drain failures. |
| Idle rotation/expiry/reconnect | Idle epoch end after empty or final A creates a fresh epoch for B; stale/foreign epoch-end callbacks cannot retire B. Healthy 180 s rotation occurs only at a barrier. Recovery is exactly three attempts with 0.8 s then 1.6 s backoff. Three sessions that reach ready and then fail before any successful turn exhaust that same episode; a fourth segment cannot reset it by reopening. |
| Custom offline HTTP | New actual-owner regression holds the HTTP coroutine while all later source frames are consumed and the second segment seals. Releasing HTTP admits finals in source order. Production phase bounds remain 5/30/10/5 s with a 50 s logical total. |
| Local decode/native stall | CPU and GPU suites distinguish pending 12 s expiry from the 30 s active decode timeout and physical capture loss. A busy native resource remains quarantined; logical expiry does not free it or allocate a replacement. |
| Slow translation/destination | Translation's eight-parent/12 s waiting envelope, per-child finite watchdog, source-only expiry, predecessor release, and output's independent destination lanes are covered. Provider/capture progress does not await LLM or sink completion. |

Within the controlled envelope no accepted source interval is lost. Outside it, callback gaps, recognition overload/expiry, provider buffer overflow, semantic expiry and destination overload have distinct failed/expired/rejected receipts. No test enlarges a production capacity.

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

The retained production-owner smoke used three 512-sample 16 kHz frames. The single segment owned normalized `[0,1536)`, sealed at that exact accepted frontier for silence, and produced this actual writer sequence: `begin_written`, payload writes 1/2/3 of 1,024 bytes each, `seal_written:silence`, `stop`, `close`. Its terminal was authoritative `final`, source closed once, and cleanup debt was zero. The throwaway probe was removed after execution.

Applicable accepted #136 scheduling evidence at this same source baseline recorded two OFF local seals at wall offsets 0.0021866 s and 0.0044042 s, with 0.0022176 s between seals, provider B injected before A at 0.0044930 s, translation idle at 0.0052395 s and output idle at 0.0053444 s. Those are controlled scheduling observations, not physical-device certification or source-to-display latency. Deterministic owner tests separately seal the real accepted frontier when no callback arrives and exercise the <=32 ms frame assumption; the <=32 ms production scheduler half of C5's 64 ms allowance remains unmeasured on a physical capture device.

## Lifecycle, replacement, and simultaneous channels

- Simultaneous cloud setup: the actual `LocalASRProviderRuntimeOwner` admits independent SELF/Deepgram and LISTEN/Soniox constructions concurrently. Both remain pending at the injected barriers; release installs each only in its channel. Peer release does not evict SELF.
- Corrected Gemini setup: legacy SELF setup executes off-loop while actual scoped LISTEN capture reaches its provider seal and terminal. The reverse channel direction is covered by independent provider-runtime setup/replacement. The legacy peer controller case remains SELF-compatibility evidence only and is not cited as LISTEN. The concrete Gemini tests retain #141's executor/client/transport ownership rather than restoring synchronous event-loop setup.
- Shared GPU: SELF and LISTEN activate on one GPU runtime and one device. While SELF VAD ingress is held, peer release completes, deactivates only peer, leaves SELF and its model resident, allocates no second runtime, and closes the physical runtime once at owner shutdown.
- CPU/local transitions retain the selected model lease, serialize decode on the actual lease, reject device changes requiring quiescence, and prevent late delegate/session construction from resurrecting a closed owner.
- Continuous rollover keeps acoustic continuity and source order while old provider work drains. Exact age-step, pause and hard-seal ties collapse to one local seal; hard timers seal only accepted source ranges.
- LISTEN OFF during setup, handoff, writer/final wait, local decode, Smart Turn inference, translation and output invalidates publication first. Re-enable starts a new generation; old callbacks and source-only fallbacks remain rejected while physical cleanup is still owned.
- Known/unknown gaps, source format changes, device/process restart and superseded target resolution end the affected capture generation. A selected missing process never falls back to desktop.
- Empty, failed, expired, cancelled and suppressed recognition each retire one source-order slot. A later successful segment waits only for that terminality, not for nonexistent text. Legitimate identical later speech is not deduplicated.
- Slow predecessor translation always releases its semantic gate on final, failed, cancelled, source-only or expiry. Capture and provider work continue independently.
- Output denies peer chatbox, records accepted handoff separately from destination delivery, rejects retired generations, protects a newer active caption, and reports destination overload/failure without replaying recognition or translation.
- OFF and supported-ON pressure use the same capture ledger, dispatch owner, scoped engine, translation and output owners. ON adds only the accepted C5 evidence worker. Provider-stall tests exercise pending inference and the 800 ms fallback while capture continues; OFF retains persisted hangover before the unchanged 4 s step and 6 s hard limit.
- PSEM is absent in the base. Injected prospective hypotheses cover accepted, already-separated, too-late, invalid, duplicate and retracted receipts with at most one seal; no producer or second timer is enabled.

A cancelled await is not treated as a killed native thread. Setup, inference, decode and protocol cleanup tests distinguish prompt logical invalidation from eventual physical release, retain one named cleanup/quarantine scope, reject replacement while occupied, and prove eventual once-only reclamation.

## Protocol evidence class and external limitations

Concrete adapter tests use actual installed SDK message classes where available: Deepgram, Gemini and Scribe SDK shapes plus documented Soniox payloads. Qwen, custom, local CPU and local GPU suites exercise their concrete request/item/task/decode contracts with controlled transports or runtimes. These are deterministic application/protocol-handling checks, not vendor wire certification.

Environment inspection found `DEEPGRAM_API_KEY`, `GEMINI_API_KEY`, `ELEVENLABS_API_KEY`, `QWEN_API_KEY`, `DASHSCOPE_API_KEY`, `SONIOX_API_KEY`, and `OPENAI_API_KEY` unset. Integration is opt-in. Therefore no paid provider call was made, and live Deepgram, Gemini, Scribe, Soniox, Qwen, rolling-member, or custom-server conformance is not claimed. No physical microphone/loopback/process capture, live GPU model decode, Smart Turn model rerun, live LLM, Flet window, overlay process, VRChat chatbox, or remote display acknowledgement was exercised for #139. These remain explicitly untested, not PASS and not removal of support.

## Evidence join and diagnostics

Source identity carries channel, activation generation, capture epoch, callback order, source and normalized ranges, segment ID/order, provider epoch/turn, and immutable settings. Known missing intervals retain ranges; unknown intervals retain null extent and a new epoch. Recognition evidence distinguishes queued operation from actual writer completion, seal, terminal outcome, failure/expiry, cleanup debt and eventual release. Translation and output retain parent/child/publication identity, semantic terminal, accepted or rejected handoff, and per-destination result.

A sink coroutine or queue acceptance proves only local handoff. It is not physical overlay/UI display. Raw PCM, transcript text, API keys and device names are not added to default diagnostics. The bounded owners expose queue depth/capacity, retained receipts and cleanup debt used by the deterministic checks; this issue does not add a synchronous per-frame logger or observability subsystem.

## Verification and AC mapping

Focused current-tree commands:

```text
uv run --frozen pytest -q tests/core/runtime/test_peer_capture_session.py tests/core/test_stt_scoped_engine.py tests/core/test_smart_turn_delivery.py tests/core/test_smart_turn_runtime.py
uv run --frozen pytest -q tests/core/runtime/test_local_asr_provider_runtime.py tests/providers/test_gemini_transcribe_lifecycle.py tests/providers/test_gemini_transcribe_ownership.py tests/app/test_runtime_pipeline_composition.py
uv run --frozen pytest -q tests/providers/test_protocol_a_scoped_sessions.py tests/providers/test_custom_stt.py tests/providers/test_qwen_audio.py tests/providers/test_qwen_asr_session.py tests/providers/test_local_cpu_backends.py tests/providers/test_local_gpu_backend.py tests/core/test_stt_rolling.py
uv run --frozen pytest -q tests/core/test_translation_turn_owner.py tests/core/test_peer_translation_channel_owner.py tests/core/runtime/test_output_runtime.py tests/core/test_translation_output_projection_owner.py
```

The consolidated focused run passed `473` cases in `12.72 s`. The final current-tree command `uv run --frozen python -m pytest -o addopts= -q --ignore=tests/integration` passed `5914` cases with `8` skips and `267` warnings in `147.13 s`. The skips remain subprocess, unavailable Windows PowerShell, and real-process Flet cases described by the accepted receipts; no new #139 regression was skipped. Ruff formatting and lint passed for all four changed Python test files.

| Acceptance | Result and evidence |
| --- | --- |
| AC01 | Baseline/revisions pinned; all selectors retained; legacy SELF/manual and UI/OSC meaning unchanged. |
| AC02 | Prefix/onset, finite tail, continuous rollover and no empty successor covered by source/owner suites and smoke. |
| AC03 | OFF and shipped ON strict 512/800, common 0.75, 4 s step, 6 s hard timer, busy/resumption/ties and exact frontiers covered. |
| AC04 | Default 64-block capture queues, known/unknown loss, no-callback behavior, discontinuity and source overflow accounting covered. |
| AC05 | Setup, actual writer, finalization, HTTP and decode stalls do not stop acoustic consumption; identity is retained. |
| AC06 | Every selector resolves; concrete route terminal matrices cover final/empty/error/timeout/EOF/duplicate with external limits explicit. |
| AC07 | Gemini barrier reorder/missing/late-A and Deepgram no-ACK two-drain covered with concrete message handling. |
| AC08 | Three-attempt recovery, idle/age rotation, rolling member changes, expiry and permanent errors are finite and do not replay uncertain audio. |
| AC09 | OFF/restart across blocking phases retires publication; cleanup/quarantine and re-enable reject late work without double free. |
| AC10 | Exact C8 capacities, TTL, controls, text/runs, output/cache, native cleanup debt and cross-channel progress covered. |
| AC11 | Complete text/run conservation, repetition, empty/suppressed/failure terminality, stable children and source order covered. |
| AC12 | Source-only, predecessor release, destination rejection, caption priority, truthful handoff and chatbox denial covered. |
| AC13 | Prospective receiver receipts and one-seal arbitration covered; PSEM producer absent. |
| AC14 | Next-segment endpoint snapshots, old-config drain, successful/failed handoff and requested/effective state covered. |
| AC15 | #141 retained; actual runtime cross-channel setup and shared GPU ownership regressions pass; shipped ON runs under provider pressure. |

## Completion disposition

The existing LISTEN ownership path is the only production peer recognition path. Its pressure, protocol, lifecycle, replacement, output and simultaneous-channel obligations are accepted on the installed pre-#143/pre-#144 composition. #136 is shipped and exercised, not deferred. #143 may reuse and rerun these cases during extraction; #144 owns a later two-policy SELF/LISTEN rerun and is not certified here.

No model quality, WER, multilingual superiority, physical-device scheduling, vendor SLA, exactly-once remote display, or future shared/SELF migration claim is made. The pre-existing `AGENTS.md` change is untouched.
