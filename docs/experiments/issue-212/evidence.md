# Issue #212: offline adapter evidence

Revision: **OFFLINE-212-2**. Author: `offline-probe-owner`. Investigation authority: [#212](https://github.com/kapitalismho/PuriPuly-heart/issues/212), with shared invariants from [#211](https://github.com/kapitalismho/PuriPuly-heart/issues/211). This is a task-local execution record, not maintained policy, product implementation, a new permanent test suite, or provider certification.

## Result and reproducibility

**43 fixture checks passed; 0 failed; 0 blocked in the final execution.** A passed check can mean that an expected information loss or unsafe-to-reuse condition was reproduced. It does not mean that a route supports speaker-aware streaming, that a provider actually emits the constructed event sequence, or that a native model produced the synthetic transcript.

Artifacts:

- [`fixtures.json`](fixtures.json): 25 bounded adapter sequences with public-style fields, sanitized text, fake request/task/item identities, and expected observations.
- [`probe.py`](probe.py): standalone executable using existing production adapters, parser methods, event projection, normalizer, consumption ledger, capture spans, input map, local decode coordinator, VAD gating, Listen delivery controller, CPU Auto and Rolling wrappers. Seven local-runtime cases, six buffered-context cases, one Rolling case and four shared-path cases are defined directly in this executable.
- [`results.json`](results.json): exact runtime versions, HEAD, fixture/probe/imported-source SHA-256 values, checks, control writes, projected events, consumption pieces, mapping slices and source-boundary records. No PCM is persisted. Bytes are represented by length and SHA-256; arrays by sample count and dtype. Random UUIDs receive consistent run-local aliases; equality, receipt order and distinct boundaries are preserved. Dataclass fields whose value is `None` or an empty collection are omitted; empty text and boolean values are retained. Per-case event logs are bounded synthetic traces, not raw provider/application logs.

Run from repository root with the supplied repository-parent Python runtime:

```powershell
../../../.venv/Scripts/python.exe docs/experiments/issue-212/probe.py --output docs/experiments/issue-212/results.json
```

To retain the checked-in observation rather than overwrite it:

```powershell
../../../.venv/Scripts/python.exe docs/experiments/issue-212/probe.py --output docs/experiments/issue-212/results-rerun.json
```

The final command above using `results.json` was exercised. The alternative output-path command is a reproduction suggestion, not an observed run. Final execution exited **0**, printed `passed=43, failed=0, blocked=0`, recorded **0 socket connection attempts**, and took **0.5228 s inside the probe / 1.03 s tool wall time**. This is probe runtime, not ASR/provider/source-release latency. The probe's socket connection guard is installed after asyncio constructs its Windows internal socketpair; all experimental operations thereafter forbid socket connects. It uses no credentials, API/model calls, real/private audio, installed application, capture device, native worker, VR/OSC connection, or paid service.

HEAD at both initial inspection and recorded execution: `97c0ad25ea1d1d7b37a396c389b7774797a77af0`. Imported production-source hashes are preserved so evidence can be invalidated by actual file differences, not only by a commit label. No Git mutation was performed.

### Exact observed environment

| Component | Installed during execution |
| --- | --- |
| Python | 3.14.7, CPython, MSC v.1944, AMD64 |
| Platform | Windows-11-10.0.22631-SP0 |
| NumPy | 2.5.1 |
| httpx | 0.28.1 |
| websockets | 16.1.1 |
| google-genai | 2.21.0 |
| deepgram-sdk | 5.3.4 |
| elevenlabs | 2.65.0 |
| dashscope | 1.26.4 |
| soxr | 1.1.0 |
| sherpa-onnx / onnxruntime | 1.13.4 / 1.28.0 |

The final run uses the supplied repository-parent virtual environment, whose recorded versions agree with the baseline requirements for the listed components. No environment installation or upgrade was performed. Early runs used global Python with NumPy 2.3.5, websockets 15.0.1, google-genai 2.20.0, Deepgram 5.3.0, dashscope 1.25.4 and no sherpa/ONNX packages; those older observations were superseded by the final virtual-environment run. Both environments emitted Deepgram's warning: `Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater.` It was retained, not suppressed or treated as provider certification failure. Actual SDK event construction and parser checks passed in the final environment; no model/binary readiness or complete packaging qualification follows.

Execution history, without hiding harness failures:

1. Initial `python -c` runtime/package inspection identified Python 3.14.7, then failed at missing `sherpa-onnx` metadata. Subsequent bounded metadata inspection recorded the installed versions above.
2. First standalone execution exited 1 before any fixture ran: installing the socket guard before event-loop construction blocked Windows asyncio's internal socketpair. The guard was moved into the running loop, before experimental operations; it was not loosened to allow provider connections.
3. Next execution produced 34 passed / 0 failed / 1 blocked, exit 1. The synthetic Deepgram word omitted SDK-required `confidence`. The fixture was corrected to include `confidence: 1`; no production parser or SDK was patched.
4. A global-Python run including the production-normalizer terminal-only-tail probe produced 36/0/0. After the additional buffered-context and wrapper assignment, the supplied repository-parent virtual environment ran all 43 cases with 43/0/0. A final metadata-label correction was followed by a 43/0/0 repeat; only this current-probe output is checked in. Repository-wide tests, builds, linters and formatters were not run.

## Actual methods exercised and fixture boundaries

The fake objects replace transports/inference, not transcript reduction rules. Soniox and Gemini run their production send loops against fake writers. Deepgram subclasses only `_write_thread_payload` to record outgoing controls instead of starting its SDK socket thread; its SDK `ListenV1ResultsEvent` is parsed by production `_build_transcript_event`. Qwen calls production `_handle_server_message` and real seal/finish methods; initial active task state is seeded, and closure prevents opening another network task. This is not a handshake or next-task qualification. ElevenLabs runs production `start` with an injected fake connect factory, recording actual registered SDK event handlers and invoking them by name. Custom streaming runs production `_receive_loop` with a fake asynchronous websocket; Custom offline executes production HTTP-response extraction with a fake client. Gemini uses actual installed `google.genai.types` message objects.

Local CPU sessions use a pre-injected fake recognizer, bypassing asset acquisition/readiness and real sherpa loading. The production `decode_f32`, `_decode_f32_sync`, decode coordinator, sealing and result projection execute. GPU uses the production buffered session with a fake `submit_pcm16` implementation returning the real `GpuWorkerTranscription` type; no device/model activation occurs. VAD uses synthetic probabilities derived from zero/one arrays, not a neural model. The actual `create_peer_vad_gating`, `ListenDeliveryController`, `PeerAudioSegmentLedger` and capture-span methods execute. VAD/source completion and local dispatch are separate component probes, not an end-to-end speaker feature.

### Adapter observations

| Production path / fixture IDs | Observed output and exact constraint |
| --- | --- |
| Soniox: `soniox-repeat-tail`, `soniox-duplicate-delivery` | Mutable non-final `wrong` is ignored. Distinct final contributions `no`, ` no`, and late ` three` preserve genuine repeated speech as `no no three`; normalizer/ledger consume it once. Re-delivering the same synthetic final token, same request ID and same times, creates another append and terminal `nono`. Request ID alone is not a token delivery identity, and the adapter does not deduplicate that constructed replay. This says nothing about whether the server legally replays it. |
| Soniox: `soniox-speaker-disabled-time-loss`, `soniox-reversed-times` | With native speaker handling enabled, token times appear only in native `FinalSpeakerRun` fields. A later token whose 20–80 ms range follows a 1000–1100 ms range is retained; terminal native run marks `overlaps_previous=true` rather than rejecting a discontinuity. With native speaker handling disabled, the same timed final token produces no speaker runs and no independent timing provenance. The experiment's native speaker IDs are fixtures, not authorized speaker authority. #211 requires Nemotron-only authority; simply disabling native runs currently also loses this route's exposed token times. |
| Soniox: `soniox-empty-fin`, `soniox-early-fin`, `soniox-multiple-fin` | Sealed empty turn + one `<fin>` produces authoritative empty, reusable epoch. `<fin>` before seal produces failed terminal and retires. A second `<fin>` after terminal retires the epoch as protocol ambiguity. Real seal writes one `finalize` control; no boundary/seal was fabricated by a proposed cutter. |
| Deepgram: `deepgram-repeat-tail`, `deepgram-duplicate-delivery` | Mutable partial is ignored. Final segments and final text on `from_finalize=true` append into `no no three`. Constructed identical final result delivery produces `nono`; native request ID, start/duration and word data do not provide retained per-contribution delivery identity. Supplied SDK word times and segment start/duration are absent from result projection. Stable text is the application adapter classification, not independent server finality proof. |
| Deepgram: `deepgram-empty-ack`, `deepgram-early-ack`, `deepgram-multiple-ack` | Seal writes `Finalize`. Empty finalize result with `is_final=false` produces authoritative empty reusable terminal. Finalize ACK before seal with nonempty text yields degraded `no`, reason `deepgram_finalize_ack_before_seal`, retire. A second ACK after terminal retires as `deepgram_idle_result`. Missing-ACK timed escalation to CloseStream was inspected in code but not exercised by this probe. |
| Qwen: `qwen-id-repeat-duplicate-correction`, `qwen-stale-id-zero-empty` | Sentence partial is ignored. Same completed sentence ID is suppressed; different IDs with equal `no` text survive, and `three` is appended. Three stable **replace/cumulative** updates normalize to `no no three`, with native sentence/task IDs retained. Supplied word intervals and `fixed` are discarded. Sentence ID `0` and stale task are ignored. Early task-finished is ignored; task-finished after real seal produces terminal; another task-finished is ignored. Empty task returns authoritative empty. Seal writes finish-task; test then closes rather than qualifying continued task startup. |
| ElevenLabs: `elevenlabs-paired-metadata`, `elevenlabs-missing-metadata-short`, `elevenlabs-metadata-only` | Registered `partial_transcript`, `final_transcript`, and `final_transcript_with_timestamps` handlers all produce provisional replacement text. Their timestamps are discarded. Real seal calls commit. `committed_transcript` alone returns authoritative final, even without timestamp metadata. `committed_transcript_with_timestamps` has **no registered handler** in this adapter; the paired event is ignored, and metadata-only produces **no terminal** in the bounded observation. No server delivery/order or missing-metadata timeout is certified. |
| ElevenLabs: `elevenlabs-early-commit`, `elevenlabs-duplicate-commit` | Commit arriving before the application's seal is ignored. Later sealed empty commit succeeds. Duplicate `committed_transcript` after successful terminal retires as unsolicited commit. Fake transport accepts 320 ms PCM and commit; this does **not** establish the provider's minimum audio/control duration. |
| Gemini: `gemini-early-late-multiple-empty` | Early `input_transcription` with `finished=false` still becomes a recognition unit. A final delivered after input turn 1 seals and turn 2 begins belongs to the stream/epoch, **not automatically turn 2**. Sequence is `no`, `late tail`, empty, `no`, receipt sequences 1–4. Separate input terminals for turns 1 and 2 are `submitted`; they are not final text or ACK. Supplied word offsets are retained as raw strings in `NativeTranscriptionEvidence`; no source interval/order coverage is projected or validated by this path. Interim text and ACTIVITY_END messages produce no transcript/control terminal here. |
| Gemini: `gemini-duplicate-delivery`, `gemini-empty-controls` | Equal synthetic final message deliveries become two distinct UUID/receipt units. There is no wire delivery identity in this fixture; string dedup would also destroy genuine repetition. Empty final still produces a unit. ACKs, including repeated/early ACKs, are ignored. Empty input seal emits submitted without an audio_stream_end write. Nonempty input writes one audio_stream_end per audio fence. None of this verifies Hybrid source-join correctness or server finality. |
| Custom offline: `custom-buffered-time-loss` | Synthetic response text `no three` survives one buffered request. Response `words` and `segments` are discarded by production text extraction; no timed source join follows merely from an OpenAI-compatible envelope. Endpoint/model are explicitly fake, not an actual selectable Custom server. |
| Custom realtime: `custom-realtime-keyed-tail`, `custom-realtime-out-of-order` | Commit barrier associates an item ID. Provisional keyed delta does not enter consumed final source; completed `no three` does. Re-delivered completed item/event ID is ignored. Keyed completion preceding the commit correlation barrier yields failed, retired protocol-error terminal. An unknown Custom server's semantics remain unknown despite this adapter parser behavior. |

Production code at the pinned baseline:

- [Soniox parser, native run construction and seal](https://github.com/kapitalismho/PuriPuly-heart/blob/97c0ad25ea1d1d7b37a396c389b7774797a77af0/src/puripuly_heart/providers/stt/soniox.py#L464-L744), [seal/padding controls](https://github.com/kapitalismho/PuriPuly-heart/blob/97c0ad25ea1d1d7b37a396c389b7774797a77af0/src/puripuly_heart/providers/stt/soniox.py#L1016-L1058).
- [Deepgram parser and terminal reconciliation](https://github.com/kapitalismho/PuriPuly-heart/blob/97c0ad25ea1d1d7b37a396c389b7774797a77af0/src/puripuly_heart/providers/stt/deepgram.py#L152-L294).
- [Qwen sentence and task completion](https://github.com/kapitalismho/PuriPuly-heart/blob/97c0ad25ea1d1d7b37a396c389b7774797a77af0/src/puripuly_heart/providers/stt/qwen_audio.py#L662-L770).
- [ElevenLabs actual handler registration](https://github.com/kapitalismho/PuriPuly-heart/blob/97c0ad25ea1d1d7b37a396c389b7774797a77af0/src/puripuly_heart/providers/stt/elevenlabs_scribe.py#L287-L335), [partial/committed handling](https://github.com/kapitalismho/PuriPuly-heart/blob/97c0ad25ea1d1d7b37a396c389b7774797a77af0/src/puripuly_heart/providers/stt/elevenlabs_scribe.py#L349-L440).
- [Gemini stream recognition projection](https://github.com/kapitalismho/PuriPuly-heart/blob/97c0ad25ea1d1d7b37a396c389b7774797a77af0/src/puripuly_heart/providers/stt/gemini_transcribe.py#L600-L639), [input seal distinct from final text](https://github.com/kapitalismho/PuriPuly-heart/blob/97c0ad25ea1d1d7b37a396c389b7774797a77af0/src/puripuly_heart/providers/stt/gemini_transcribe.py#L846-L866).
- [Custom offline response extraction](https://github.com/kapitalismho/PuriPuly-heart/blob/97c0ad25ea1d1d7b37a396c389b7774797a77af0/src/puripuly_heart/providers/stt/custom.py#L376-L427), [realtime correlation and completion](https://github.com/kapitalismho/PuriPuly-heart/blob/97c0ad25ea1d1d7b37a396c389b7774797a77af0/src/puripuly_heart/providers/stt/custom.py#L801-L947).

### Local timing and buffered short-input dispatch

Six CPU cases execute both a text-only and a fabricated timestamp-rich recognizer result for each actual backend class:

- `local_qwen`: `qwen3-asr-0.6b-int8-sherpa`.
- `local_parakeet_v3`: `parakeet-tdt-0.6b-v3-int8-sherpa`.
- `local_parakeet_ja`: `parakeet-tdt-ctc-0.6b-ja-int8-sherpa`.

Each receives **1600 samples / 100 ms** synthetic PCM, has zero decode calls before seal, and dispatches exactly one decode after `source_eof`. `no three` survives as a final. In the rich fixture, `tokens`, `timestamps` and `durations` disappear because actual `_decode_f32_sync` returns only `str(result.text).strip()`. These are synthetic field-preservation observations, **not evidence that any shipped model returns those fields, units, intervals or alignments**. CPU Auto's existing-resolved-delegate path is exercised separately below; actual install inspection/model selection is not.

The GPU buffered session also dispatches exactly one 100 ms PCM decode after seal against a fake runtime. Text and detected-language run survive. Actual `GpuWorkerTranscription` fields are `text`, `detected_language`, `audio_seconds`, `decode_seconds`, `rtf`; there are no token/word/segment timestamps in that Python contract. This does not establish timestamp absence in every native binary or underlying model; it establishes the current adapter-visible contract. GPU model ID and device are deliberately fixture values; no Vulkan/CPU/CUDA inference claim is made.

Code: [CPU text-only reduction](https://github.com/kapitalismho/PuriPuly-heart/blob/97c0ad25ea1d1d7b37a396c389b7774797a77af0/src/puripuly_heart/providers/stt/local_qwen_sherpa.py#L296-L310), [buffered CPU seal](https://github.com/kapitalismho/PuriPuly-heart/blob/97c0ad25ea1d1d7b37a396c389b7774797a77af0/src/puripuly_heart/providers/stt/local_qwen_sherpa.py#L407-L437), [GPU worker result](https://github.com/kapitalismho/PuriPuly-heart/blob/97c0ad25ea1d1d7b37a396c389b7774797a77af0/src/puripuly_heart/core/gpu_worker.py#L29-L35), [GPU buffered seal](https://github.com/kapitalismho/PuriPuly-heart/blob/97c0ad25ea1d1d7b37a396c389b7774797a77af0/src/puripuly_heart/providers/stt/local_gpu.py#L181-L201).
### Buffered rollover context is not excluded from decoding

Six additional `<route>-rollover-context-buffer` cases exercise actual Qwen CPU, Parakeet v3 CPU, Japanese Parakeet CPU, local GPU, Custom offline and CPU Auto sessions. CPU Auto uses its actual `open_session` against an injected already-resolved Qwen delegate; this tests delegation and retained model/session identity, not installation discovery or language-based model selection.

Each uses two separate turn identities:

1. Turn 1 sends source `[0,4800)` as content: **4800 samples / 300 ms**, PCM16 marker value **1000**. It seals and fake-decodes to `again`.
2. Turn 2 sends exactly the same `[0,4800)` PCM with `context_only=true`, then new source `[4800,6400)` with `context_only=false`: **1600 samples / 100 ms**, marker value **2000**. Its declared sealed content range is **only `[4800,6400)`**.
3. Actual buffered adapters pass **all 6400 samples / 400 ms** to the fake recognizer/runtime or actual generated WAV HTTP request. Recorded sample counts/value frequencies show both markers. CPU decode receives the expected float32 values after real PCM conversion; GPU and Custom WAV receive PCM16 values 1000/2000.
4. The fake decode is explicitly defined to output `again new` when the new marker is present. This is not neural ASR evidence. It exposes old context in the decoded input and demonstrates that the current whole-terminal contract cannot exclude its text by `context_only` or sealed source ranges. Both terminals are consumed as distinct sources: `again`, `again new`. Gray styling would not remove the repeated ownership/content.

This is a supported negative **buffer-exposure and result-contract** observation, not proof that every real model repeats pre-roll words or a measurement of repetition frequency. Baseline 300 ms hard-rollover behavior is unchanged. Enhanced batch cuts need disjoint selected content or a verified way to exclude context-derived source text before no-double-ownership enablement; this probe does not implement that choice. The actual VAD/ledger keeps pre-roll as context, but downstream buffered decoding still includes those bytes.

### Direct Rolling selection and late member result

`actual-rolling-late-member-result` opens actual `RollingSTTBackend`/`_RollingSession` over production Gemini and Deepgram sessions with fake transports. First it selects Gemini; a fixture configuration toggle then makes a second open select Deepgram. A late Gemini final is consumed through the original wrapper, retaining its original `rolling-gemini` stream settings scope and member identity. Gemini's independent-recognition capability is true on the first session and false on the second Deepgram session. Both use the same fixture epoch deliberately, so this does not certify the production epoch generator or capability-in-result propagation. No quota/error-classification/server behavior is claimed. Direct Custom offline/realtime paths are already exercised in the adapter table above.

Code: [Auto resolved-delegate session opening](https://github.com/kapitalismho/PuriPuly-heart/blob/97c0ad25ea1d1d7b37a396c389b7774797a77af0/src/puripuly_heart/providers/stt/local_cpu.py#L124-L137), [Rolling selection and member wrapper](https://github.com/kapitalismho/PuriPuly-heart/blob/97c0ad25ea1d1d7b37a396c389b7774797a77af0/src/puripuly_heart/core/stt/rolling.py#L268-L289), [Rolling recognition remapping](https://github.com/kapitalismho/PuriPuly-heart/blob/97c0ad25ea1d1d7b37a396c389b7774797a77af0/src/puripuly_heart/core/stt/rolling.py#L503-L546).


### Existing source minimum, six-second safety and continuation

`existing-224ms-source-completion` uses actual Peer VAD construction (three-chunk start commitment), actual ledger and actual Listen controller, delivery profile off and explicit **128 ms hangover**. Three 32 ms speech chunks plus four silence chunks naturally seal **3584 samples / 224 ms** with `delivery_pause`. This is a supported component regression counterexample to a **universal one-second source-duration floor**. It is not a newly implemented speaker cut, not neural short-speech accuracy, and not a promise that every provider accepts 224 ms input. The hangover is an explicit fixture setting, not a newly selected default. Separate actual buffered session probes show 100 ms dispatch with exactly one fake decode, preserving a short correction.

`existing-six-second-rollover` sends continuous synthetic speech. The capture-frontier branch seals at **6.016 s**, the first 32 ms frame frontier at or after the existing **6.0 s** hard limit. The next frame opens another segment with `genuine_onset=false`, **4800 samples / 300 ms** pre-roll context, and **512 samples** new content. Existing source ownership does not reclaim the old content as new content. This exercises the acoustic/frontier safety branch and VAD rollover; it does not measure scheduling of the real wall-clock deadline timer or qualify provider pre-roll duplication handling.

Code: [Listen hard limit and capture-frontier branch](https://github.com/kapitalismho/PuriPuly-heart/blob/97c0ad25ea1d1d7b37a396c389b7774797a77af0/src/puripuly_heart/core/audio/listen_delivery.py#L38-L48), [natural pause and deadline decisions](https://github.com/kapitalismho/PuriPuly-heart/blob/97c0ad25ea1d1d7b37a396c389b7774797a77af0/src/puripuly_heart/core/audio/listen_delivery.py#L134-L211), [rollover state and 300 ms constant](https://github.com/kapitalismho/PuriPuly-heart/blob/97c0ad25ea1d1d7b37a396c389b7774797a77af0/src/puripuly_heart/core/vad/gating.py#L487-L575).

### Existing mapping, discontinuities and terminal-only tail

`actual-capture-span-and-stream-map` calls production methods, not a proposed-policy mapper:

- Source **48 kHz samples [96000,100800)**, monotonic **[2.0,2.1) s**, maps to normalized **16 kHz [32000,33600)**. Slicing at normalized 32800 returns source boundary **98400**, preserving the known-loss marker only on the slice retaining the original left edge.
- Input-map replay of already sent `[0,1600)` returns empty PCM; overlapping `[800,2400)` sends only `[1600,2400)` (**800 samples**) and total sent samples becomes **2400**.
- A normalized gap raises `stream source discontinuity`; capture-epoch change raises `stream capture epoch changed without retirement`.
- A span with a **contiguous normalized range but explicit discontinuity marker** is accepted by this low-level map. Its existing logic checks normalized position/epoch, not `discontinuity_before`. No claim is made that upper capture/engine layers ignore or correctly resolve that marker. Policy integration must keep marker handling and piecewise source mapping explicit rather than infer coverage from a continuous numeric frontier.

`existing-normalizer-terminal-only-tail` directly executes the production `STTScopedTurnNormalizer` and `STTContributionConsumptionLedger`: provisional `four` can change to `no` without being consumed; two genuinely repeated stable `no` contributions with distinct native IDs survive; duplicate native ID is suppressed; terminal `no no three` contributes only **` three`** after consumed `no no`; terminal replay contributes empty. Stable `four` followed by terminal correction `no` raises `provider_stable_prefix_inconsistent`. This shows the current immutable-prefix assumption, not permission to classify a provider's mutable result as stable. The ledger consumes whole declared contributions; no intra-contribution audio/time split was implemented or exercised.

Code: [AudioCaptureSpan slicing](https://github.com/kapitalismho/PuriPuly-heart/blob/97c0ad25ea1d1d7b37a396c389b7774797a77af0/src/puripuly_heart/core/audio/format.py#L77-L123), [actual stream input map](https://github.com/kapitalismho/PuriPuly-heart/blob/97c0ad25ea1d1d7b37a396c389b7774797a77af0/src/puripuly_heart/core/stt/stream_input.py), [normalizer](https://github.com/kapitalismho/PuriPuly-heart/blob/97c0ad25ea1d1d7b37a396c389b7774797a77af0/src/puripuly_heart/core/stt/scoped_normalizer.py#L68-L215), [consumption ledger](https://github.com/kapitalismho/PuriPuly-heart/blob/97c0ad25ea1d1d7b37a396c389b7774797a77af0/src/puripuly_heart/core/stt/backend.py#L116-L178).

## Evidence limits and integration disposition

**Fixture-tested:** the specific production parser/projection/control, buffered-context, resolved-delegate, member-wrapper and component sequences above. **Not run / unknown:** live server finality, upstream event ordering/duplicate guarantees, legal control minima, timestamps' actual source meaning, endpoint/region service behavior, rate/cost/quotas, actual model outputs, device/runtime readiness, actual CPU Auto install/model selection, production Rolling epoch/capability payload, Hybrid common-engine source coverage/admission, real short-speech accuracy, overlap diarization, provider pre-roll attribution, and wall-clock/UI/VR latency. Models, credentials and spend authorization were neither requested nor used. The final environment has sherpa/ONNX packages, but no model/device readiness or inference was inspected/executed. This assignment does not decide unsupported upstream contracts by guessing.

For the Director's policy matrix:

1. Separate adapter-tested observations from the read-only researchers' official contract evidence. The fake endpoint/model labels do not establish production route resolution.
2. Preserve Soniox timing separately from native speaker authority; preserve Deepgram/Qwen/ElevenLabs/Custom/local timings only where verified upstream/runtime data exists. The current losses are recorded, not repaired here.
3. Distinguish same projected contribution replay (ledger suppresses it) from identical provider delivery interpreted as a fresh contribution (Soniox/Deepgram/Gemini fixtures preserve both). Text equality alone cannot resolve repeated speech versus replay.
4. Preserve Gemini stream/epoch receipt identity and raw evidence without inventing source coverage or attaching late final text to the newest input turn. Input submitted, transport fence, ACK and text are different observed surfaces.
5. Reject a universal one-second source floor on the component evidence above, while independently deciding lawful route-specific controls. Preserve the existing six-second safety path and rollover continuation/context ownership. ElevenLabs server minimum remains outside this offline result.
6. Treat missing committed timestamp subscription/pairing as an observed ElevenLabs information-preservation gap, not a completed pairing reducer or provider guarantee. Metadata-only is unresolved in this bounded run.
7. Resolve buffered context ownership before enhanced-route enablement: observed actual batch decoding includes context-only bytes and ignores sealed content ranges. Preserve disjoint content for enhanced batch cuts as selected by the Director; baseline rollover remains unchanged here. Untimed gray output is not sufficient to prevent duplicate source ownership.

No maintained guidance, product code or tests were changed. No repaired adapter or new speaker-cut implementation is delivered. The Director owns policy integration and any necessary product decisions. If these artifacts are incorporated, the focused verification command is the repository-parent standalone probe above; no repository-wide verification is requested by this evidence assignment. Broader product/provider qualification remains separately authorized work.
