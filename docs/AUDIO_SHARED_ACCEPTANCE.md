# AUDIO_SHARED_ACCEPTANCE

## Revision and scope

- Contract: GitHub issue #143, `AUDIO-SHARED-1`, I0-I2.
- Requested characterization baseline: `d022c8a22eb9fb8c62165f8e55a7ca5a778b4d47`.
- Accepted Audio Core input: implementation `41210e2fd36874280996526a83b3039e0618b53a`, receipt `cf5d038ee9e23d1daab367a00343142e8e504f6a`.
- Remote development pin inspected by the Director: `6395fd8d0f6b3711529bfcc051618b2911bbc923`; it is not an ancestor of the requested local baseline.
- Original checkpoint candidate `e0e62a23030bd04fa3d704abd36d539a01507abc` was rejected for F1-F4 and evidence gaps. First repair candidate `bdf8ab6844871b68b78f8af7610eee99a847db4f` closed those findings but was **rejected** for R1: normalizing each append deleted trailing-style inter-fragment separators. Text-boundary repair implementation: `bff08d9813934e1a4a5b53cc3f17dedbf1fc1d76`; independent verification and Director acceptance are pending.
- Production SELF remains on its legacy projection. The scoped SELF-like client used here is a non-production conformance binding only; production adoption and retirement are owned by #144.

## I0 characterization and owner map

The requested baseline retained these deterministic SELF facts: microphone capture and mute gating are owned by `SelfCaptureSessionOwner`; unscoped provider attachment and staged handoff are owned by `ProviderRuntimeHandle` under `LocalASRProviderRuntimeOwner`; legacy partial/final projection is consumed by `SelfTranslationChannelOwner`; translation turns and output remain under `TranslationTurnLifecycleOwner` and `OutputRuntime`. Manual text bypasses capture and recognition.

The frozen deterministic SELF traces at `d022c8a2` are:

| Case | Observation / recognition / publication trace |
| --- | --- |
| Stable before end | accepted microphone/VAD onset and prefix -> legacy provider stable event -> current SELF merge source and speculative translation -> real `SpeechEnd` -> provider final -> existing target/output commit |
| End before final | accepted onset/content -> real `SpeechEnd` and provider drain -> legacy provider final callback -> existing finalize grace/translation -> output commit |
| Smooth provider handoff | old resolved configuration remains effective through the current utterance boundary -> old provider drains -> handle commits prepared replacement -> next onset uses the new configuration |
| TALK OFF | capture admission closes -> provider abort and legacy SELF channel reset -> late speech callback rejected; baseline channel-wide cancellation behavior remains for #144 to correct |
| Manual text | UI text intent -> manual SELF translation parent -> existing output owner, with no capture, VAD, recognition request or audio budget |

The shared/LISTEN path owner map is:

| Responsibility | One owner |
| --- | --- |
| Source coordinates, mono-first normalization, resampler residue, callback gaps | capture source and `audio_vad_loop` stream instance |
| Accepted source ranges, context, seal, terminal order | `PeerAudioSegmentLedger` (the current shared request-ledger implementation; its retained asymmetric name does not add a second implementation) |
| LISTEN pause/age/hard-cut policy | `ListenDeliveryController` |
| Ordered request begin/payload/seal, provider epoch, watchdogs, cleanup | `ScopedRecognitionEngine` |
| Native assembly, deduplication, contribution provenance, stable-prefix consistency | `STTScopedTurnNormalizer` |
| Attachment generations and staged replacement | `ProviderRuntimeHandle` |
| Concrete SDK/session resource and native completion barrier | the selected `providers/stt` adapter |
| Shared GPU process, compatible model/device and admitted FIFO | `SharedGpuASRRuntime` |
| LISTEN final-parent admission | `PeerTranslationChannelOwner` and existing translation lifecycle owner |
| Publication and destination delivery | `OutputRuntime` |

Baseline SELF observation/recognition/publication characterization is covered by the existing self capture/provider adapter and low-latency owner fixtures listed under verification. The extraction does not route production SELF through the scoped exchange and does not change its publication behavior.

## Shared exchange and resolved profiles

`ProviderRuntimeBuildRequest.recognition_projection` selects legacy or scoped projection explicitly; `auto` preserves production compatibility by selecting scoped for LISTEN and legacy for SELF. A non-production SELF-like binding selects `scoped` explicitly and receives the same `ScopedRecognitionEngine`, backend factory and concrete adapter used by LISTEN. Every instance has channel-local request state, epoch, event buffer, queue and contribution state.

Stable updates preserve the provider's raw cumulative assembly privately, normalize the complete assembled boundary once for publication, and expose a request-local contribution ID with an exact `[text_start, text_end)` range in that normalized whole. A trailing separator from one fragment remains available and becomes the prefix of the next contribution when a later fragment makes it internal; leading-style and multilingual separators behave identically. A terminal exposes the complete normalized transcript plus the exact previously emitted contributions it includes. `STTContributionConsumptionLedger` demonstrates early consumption without closing the recognition request: consuming A before seal and then terminal A+B yields only B. Cumulative replacement compares normalized incoming text with the normalized accepted prefix, so harmless raw leading whitespace does not falsely contradict it. A genuine prefix change fails as `provider_stable_prefix_inconsistent`; no substring repair, fuzzy matching, or provider-specific spacing rule is used.

Retention numbers are binding-owned:

- LISTEN: recognition accounting covers the already-accepted dispatch envelope of eight wholly-unsent sealed segments, one segment being recognized, and one source-open segment. Each slot is the existing 6,000 ms hard source range plus configured prefix at the resolved sample rate. `ListenDeliveryController`, not recognition memory accounting, remains the sole segment endpoint owner; a legal chunk crossing 6 seconds is retained and transcribed.
- SELF-like: 2,880,000 normalized mono sample-equivalents and four bytes of accounting capacity per equivalent. This is a retained-memory failure ceiling, not an endpoint or a successful 180-second split.
- Streaming scoped routes release the engine reservation after ordered adapter writer completion. Local CPU, local GPU, and custom offline sessions selected through either `custom` or `custom_offline` retain it until terminality; custom realtime releases after completed writes.
- GPU public and scoped admission: eight pending jobs per channel, excluding active work. Admitted jobs preserve global speech-end then sequence order. A ninth pending job for that channel terminalizes with `failure_reason="pending_capacity"`; it does not evict the other channel.

LISTEN endpoint values continue to come only from `ListenDeliveryController` and the resolved peer settings snapshot. No generic channel-name inference adds those endpoint values to SELF.

## Concrete provider inventory

All configured selectors are retained and resolve through the existing backend factory and scoped-session projection:

`local_cpu_auto`, `local_parakeet_v3`, `local_parakeet_ja`, `local_qwen`, `local_qwen_gpu`, `deepgram`, `gemini_transcribe`, `elevenlabs_scribe`, `qwen_asr`, `qwen_audio`, `soniox`, `rolling_free`, `custom`, `custom_offline`, `custom_realtime`.

Aliases and rolling members resolve to those concrete protocol/resource paths; no second SELF or LISTEN parser, socket owner, reconnect loop, recognition controller, global scheduler, manager or endpoint abstraction was added.

## Executable I2 case map

The I2 proof is not inferred from a general suite pass. These named cases execute the shared production classes:

| Case | Real path exercised | Required observation |
| --- | --- | --- |
| `test_shared_engine_real_owner_and_deepgram_adapter_serve_both_clients` | Two `ScopedRecognitionEngine` instances attached simultaneously to the real `LocalASRProviderRuntimeOwner`/`ProviderRuntimeHandle` pair, each opening the concrete `_DeepgramSDKSession` parser/writer projection | LISTEN ignores stable updates and receives one terminal `AB`; SELF-like consumes stable A before seal, B is injected after the adapter seal, terminal `AB` declares both contributions and the consumption ledger yields only B. Both use the same engine class and concrete adapter implementation with distinct epochs/state. |
| same case, retained-writer phase | Real engine -> concrete Deepgram `_write_thread_payload` outbound barrier | While the adapter writer is blocked, the channel retention snapshot remains 4 samples/8 bytes and the source dispatch task remains pending. After actual writer completion it returns to zero. Twenty further four-sample chunks are accepted and written before the same request seals: 84 total samples traverse a profile whose instantaneous bound is four, proving no aggregate duration endpoint while every retained transfer remains charged. |
| `test_two_scoped_deepgram_clients_isolate_abort_and_native_late_result` | Same real owner, two scoped engines, two concrete Deepgram sessions | SELF-like abort emits only its scoped cancellation; a native late SELF result is rejected; the simultaneously open LISTEN request reaches `peer-ok` and its runtime channel remains running. |
| `test_scoped_configuration_handoff_is_channel_local_with_concrete_adapter` | Real runtime owner/handles and concrete Deepgram sessions across an explicit SELF-like scoped settings handoff | The replacement accepts only the new runtime signature; the already-open LISTEN session/identity is unchanged and later terminals normally as `peer-survived`. |
| `test_real_shared_gpu_runtime_serves_two_scoped_concrete_adapters` | One real `SharedGpuASRRuntime`, two concrete `LocalGpuSTTBackend` scoped sessions, one fake native worker process boundary | SELF-like cancellation leaves native work owned, LISTEN remains queued/admitted, then both native jobs settle in FIFO order. Closing SELF releases only its lease while the worker and LISTEN lease remain; the last close stops the shared worker. |
| `test_global_speech_end_fifo_has_finite_per_channel_public_admission` and existing GPU failure/recovery cases in the same file | Real shared GPU owner with fake worker process boundary | Eight pending jobs per channel are admitted, the ninth is finite `pending_capacity`, admitted speech-end/sequence order is unchanged, and actual worker failures/cancel quarantine remain owned by the one shared coordinator. |
| `test_each_concrete_streaming_protocol_serves_self_and_peer_concurrently` | Concrete Deepgram, Gemini Transcribe, Soniox and ElevenLabs Scribe adapters, two simultaneous scoped native boundaries each | Every distinct streaming protocol projection accepts SELF-like and LISTEN requests concurrently, publishes each native result only to its matching identity, and closes both concrete adapter resources. |
| `test_local_cpu_scoped_decode_snapshots_pcm_and_emits_one_terminal` and `test_cpu_auto_aliases_delegate_scoped_turn_contract` (both parametrized by `channel`) | Direct Qwen/Parakeet v3/Parakeet JA CPU adapters and all `local_cpu_auto` delegate aliases | The concrete batch adapters execute their scoped contract under both SELF-like and LISTEN request channels. |
| `test_scoped_native_items_reject_duplicate_late_and_unsolicited_terminals`, `test_scoped_task_uses_native_task_barrier_and_stable_sentence_updates`, and the two parametrized custom scoped tests | Qwen ASR, Qwen Audio, custom offline and custom realtime concrete adapters | Both request channels exercise actual protocol projections, native IDs/barriers, and terminal behavior; custom offline additionally uses the production resolver under both `custom` and `custom_offline` selectors. |
| `test_production_retention_profiles_accept_listen_boundary_and_long_self_like` | Real engine with production `_recognition_retention_profile` | A legal 6.144-second LISTEN crossing chunk plus 500 ms prefix finishes `final`; a seven-second SELF-like request (strictly older than LISTEN's timer) also finishes `final` without a time cut. |
| `test_abort_immediately_invalidates_authority_while_native_phase_is_blocked` | Real engine with controlled native open, write and final-wait barriers | OFF returns with the logical turn/epoch authority invalidated in every phase; later native completion is cleanup work and cannot publish a final. This does not claim native work physically stopped. |
| `test_real_soniox_shared_engine_preserves_trailing_token_separator` | Real `ScopedRecognitionEngine` plus concrete `_SonioxSession` parser/send/receive tasks and fake websocket boundary | Native trailing-style `"same " + "世界"` updates publish `"same"` then `"same 世界"`; contribution ranges are `[0,4)` and `[4,7)`, early consumption returns `"same"`, terminal consumption returns `" 世界"`, and terminal language runs conserve the exact text. |
| `test_append_separators_and_cumulative_raw_whitespace_preserve_exact_suffixes` | Shared normalizer and public consumption ledger | Trailing-style multilingual append and raw-leading cumulative replace preserve exact IDs/ranges/suffixes; normalized cumulative prefix comparison accepts `"  hello "` -> `"  hello world  "` without weakening contradiction detection. |

## RI01-RI10 result

| ID | Result and evidence |
| --- | --- |
| RI01 | PASS. Deterministic LISTEN owner/output suites remain unchanged, and the requested `d022c8a2` Deepgram timed-token/pretranslation delta is rerun in `test_stt_timed_tokens.py`, `test_peer_pretranslation_path.py`, `test_pretranslation_ownership.py`, `test_translation_ownership_children.py`, and the concrete Deepgram session/conformance files. |
| RI02 | PASS by `test_shared_engine_real_owner_and_deepgram_adapter_serve_both_clients`: both clients are simultaneously attached to the real runtime owner/handles and execute the same `ScopedRecognitionEngine`, `STTScopedTurnNormalizer`, event buffer and concrete Deepgram parser/writer implementation. Engines, epochs, queues, ledgers and consumers are separate instances. |
| RI03 | PASS by the same concrete case. Deepgram stable A is observed while its adapter projection is unsealed; B is injected only after `sealed` becomes true; terminal `AB` occurs once and lists contribution ranges `[0,1)` and `[1,2)`. |
| RI04 | PASS by the same concrete case. The SELF-like consumer commits A while open, deliberately does not consume B's update, then consumes terminal `AB` as B only. A second terminal consumption is empty in the focused provenance regression. |
| RI05 | PASS after text-boundary repair by `test_stable_contribution_provenance_preserves_suffix_and_detects_contradiction`, `test_whitespace_stable_contributions_match_normalized_terminal_ranges`, `test_append_separators_and_cumulative_raw_whitespace_preserve_exact_suffixes`, and the real Soniox+engine regression. Append delta, cumulative replace, both whitespace separator styles, multilingual runs, identical disjoint append text, duplicate native ID, stable-prefix contradiction, terminal-only text, authoritative empty and failure outcomes remain distinct without fuzzy matching. |
| RI06 | PASS after repair by `test_abort_immediately_invalidates_authority_while_native_phase_is_blocked` for open, write, and final-wait phases plus the two-client Deepgram late-result case. Abort increments logical authority before waiting on the input path, detaches the current turn/session, emits scoped cancellation, and quarantines native completion under cleanup ownership. It does not claim the native operation stops immediately. Configuration replacement remains covered by the concrete channel-local handoff case. |
| RI07 | PASS after repair by the concrete retained-writer case, production profile tests, and focused `buffer_exhausted` regression. The legal LISTEN 6.144-second crossing chunk plus prefix succeeds; a seven-second SELF-like request proves independence from LISTEN age/timer; custom offline under both selectors remains charged exactly with its real adapter buffer; custom realtime and concrete streaming writers release only after write completion. |
| RI08 | PASS by concrete Deepgram abort/handoff cases and the repaired immediate-authority test. Scoped SELF-like abort/native lateness and settings handoff do not replace, stop, re-identify, or publish into scoped LISTEN. |
| RI09 | PASS by the real shared-GPU two-adapter case, finite admission test, and scoped pending-capacity regression. Cancelling SELF-like preserves LISTEN and native ownership; scoped failure reason retains `pending_capacity`. |
| RI10 | PASS. In addition to real owner/engine Deepgram and GPU cases, every distinct streaming protocol and each local/custom/Qwen mode is exercised under both request channels. Rolling-free delegates to the already-covered Scribe/Gemini/Deepgram concrete member sessions and adds only selection/error classification, covered by `test_stt_rolling.py`; it is not a distinct parser or buffering profile. Fakes stop at SDK/network/native-process boundaries. |

## Temporary facade and #144 removal list

The following remains solely because production SELF migration is not authorized in #143:

- `ManagedSTTProvider` legacy event projection and legacy backend transcript events.
- `LEGACY_STT_SESSION_PROJECTION` and the SELF `auto -> legacy` factory branch.
- `LocalASRProviderRuntimeOwner.handle_vad_event` unowned SELF dispatch.
- SELF-only legacy VAD handoff commit and the matching capture/provider adapter calls.
- Legacy SELF event callback projection into `SelfTranslationChannelOwner`.

Issue #144 owns production SELF binding, its contribution/publication policy, callsite migration, and deletion of this facade/removal list. None of these paths is a fallback for LISTEN or the scoped SELF-like fixture.

## Verification and limits

Focused commands and pass counts below are from the repaired worktree. Tests use controlled audio, fake SDK transports, fake GPU workers and real runtime owners/adapters unless noted. The original 828-pass claim did not cover the rejected F1-F4 interactions and is retained here only as history; it is not repair evidence.

- `.venv/Scripts/python.exe -m pytest -o addopts= -q tests/providers` — 519 passed.
- Focused core recognition/GPU/rolling/audio/translation coverage — 191 passed.
- Focused app wiring/SELF/runtime coverage — 168 passed.
- Final post-cleanup confirmations: 229 repaired recognition/adapter/rolling cases, 95 GPU/runtime cases, and 50 capture-envelope cases passed.
- Actual shared-runtime smoke selection — 13 passed: real runtime owner + dual concrete Deepgram clients, every distinct streaming protocol under both clients, real `SharedGpuASRRuntime` + dual concrete GPU adapters, production LISTEN/SELF-like profiles, real offline custom buffering under both selectors, and open/write/final-wait abort races.
- Ruff over all repaired Python sources/tests passed; `compileall -q` over repaired source modules completed without output/errors.
- Text-boundary repair direct verification: `test_stt_scoped_engine.py` plus `test_protocol_a_scoped_sessions.py` — 40 passed. Actual shared-runtime smoke — 3 passed: real Soniox+engine trailing separator, real owner+dual Deepgram clients, and deterministic trailing/multilingual plus cumulative raw-whitespace provenance.

No live provider credentials, provider service quality, real microphone/loopback device, real GPU model decode, timing-performance envelope or physical remote-display acknowledgement is certified by these fake/controlled fixtures.
