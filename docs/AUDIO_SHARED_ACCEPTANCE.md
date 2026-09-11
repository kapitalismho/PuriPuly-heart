# AUDIO_SHARED_ACCEPTANCE

## Revision and scope

- Contract: GitHub issue #143, `AUDIO-SHARED-1`, I0-I2.
- Requested characterization baseline: `d022c8a22eb9fb8c62165f8e55a7ca5a778b4d47`.
- Accepted Audio Core input: implementation `41210e2fd36874280996526a83b3039e0618b53a`, receipt `cf5d038ee9e23d1daab367a00343142e8e504f6a`.
- Remote development pin inspected by the Director: `6395fd8d0f6b3711529bfcc051618b2911bbc923`; it is not an ancestor of the requested local baseline.
- Implementation result: `72f54772df26a68b2ed61220cb52e108686453d4`. Independent review and Director acceptance are pending; the RI results below are implementation verification claims.
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

Stable updates expose a request-local contribution ID and exact `[text_start, text_end)` range in the normalized assembled transcript. A terminal exposes the complete transcript plus the exact previously emitted contributions it includes. `STTContributionConsumptionLedger` demonstrates early consumption without closing the recognition request: consuming A before seal and then terminal A+B yields only B. Duplicate native IDs remain suppressed; identical disjoint append contributions retain distinct IDs. A cumulative replacement that changes an accepted stable prefix fails as `provider_stable_prefix_inconsistent`; no substring repair or word alignment is used.

Retention numbers are binding-owned:

- LISTEN: the existing configured 6,000 ms hard source range plus configured prefix at the resolved 16 kHz sample rate; existing eight wholly-unsent sealed-segment and 12-second freshness envelope remains in the capture dispatch owner.
- SELF-like: 2,880,000 normalized mono sample-equivalents and four bytes of accounting capacity per equivalent. This is a retained-memory failure ceiling, not an endpoint or a successful 180-second split.
- Streaming scoped routes release the engine reservation after ordered adapter writer completion. Local CPU, local GPU and custom-offline batch routes retain it until terminality.
- GPU public admission: eight pending jobs per channel, excluding active work. Admitted jobs preserve global speech-end then sequence order. A ninth pending job for that channel fails as `pending_capacity`; it does not evict the other channel.

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

## RI01-RI10 result

| ID | Result and evidence |
| --- | --- |
| RI01 | PASS. Deterministic LISTEN owner/output suites remain unchanged, and the requested `d022c8a2` Deepgram timed-token/pretranslation delta is rerun in `test_stt_timed_tokens.py`, `test_peer_pretranslation_path.py`, `test_pretranslation_ownership.py`, `test_translation_ownership_children.py`, and the concrete Deepgram session/conformance files. |
| RI02 | PASS by `test_shared_engine_real_owner_and_deepgram_adapter_serve_both_clients`: both clients are simultaneously attached to the real runtime owner/handles and execute the same `ScopedRecognitionEngine`, `STTScopedTurnNormalizer`, event buffer and concrete Deepgram parser/writer implementation. Engines, epochs, queues, ledgers and consumers are separate instances. |
| RI03 | PASS by the same concrete case. Deepgram stable A is observed while its adapter projection is unsealed; B is injected only after `sealed` becomes true; terminal `AB` occurs once and lists contribution ranges `[0,1)` and `[1,2)`. |
| RI04 | PASS by the same concrete case. The SELF-like consumer commits A while open, deliberately does not consume B's update, then consumes terminal `AB` as B only. A second terminal consumption is empty in the focused provenance regression. |
| RI05 | PASS by `test_stable_contribution_provenance_preserves_suffix_and_detects_contradiction` plus concrete provider conformance: append delta, cumulative replace, identical disjoint append text, duplicate native ID, stable-prefix contradiction, terminal-only text, authoritative empty and failure outcomes are distinct without fuzzy matching. |
| RI06 | PASS by scoped engine begin/send/seal barrier tests and `test_two_scoped_deepgram_clients_isolate_abort_and_native_late_result`: SELF-like abort is prompt and local, its late concrete native callback cannot publish, and the concurrently open LISTEN request finishes. Configuration replacement is additionally exercised by the named concrete handoff case. |
| RI07 | PASS by the retained-writer phase of the two-client concrete Deepgram case and the focused `buffer_exhausted` regression. Reservation remains charged while the concrete SDK writer barrier owns the payload and releases only on completion; 84 samples traverse one unsegmented request under a four-sample instantaneous streaming bound. Batch/local bindings charge four actual bytes per retained float32 sample until terminal. The selected 2,880,000-sample SELF-like profile is a memory failure bound, never a timer. |
| RI08 | PASS by the two concrete Deepgram isolation cases. Scoped SELF-like abort/native lateness and scoped SELF-like configuration handoff do not replace, stop, re-identify or publish into the simultaneously open scoped LISTEN client. Existing bounded-pressure/timeout owner tests cover the same channel-local handles and queues. |
| RI09 | PASS by `test_real_shared_gpu_runtime_serves_two_scoped_concrete_adapters`: two concrete scoped local-GPU sessions use one real shared GPU owner and fake process boundary; cancelling SELF-like leaves LISTEN queued and native ownership intact, closing SELF releases only its lease, and the worker survives until LISTEN closes. The finite admission and existing real-owner worker failure/recovery cases cover FIFO, shared physical failure and quarantine without a legacy SELF provider. |
| RI10 | PASS. The named I2 fixtures use real `PeerAudioSegmentLedger`, `ScopedRecognitionEngine`, `LocalASRProviderRuntimeOwner`, `ProviderRuntimeHandle`, concrete Deepgram/local-GPU adapters and real `SharedGpuASRRuntime`. Fakes stop at SDK websocket or native worker process boundaries; no alternate controller supplies lifecycle behavior. |

## Temporary facade and #144 removal list

The following remains solely because production SELF migration is not authorized in #143:

- `ManagedSTTProvider` legacy event projection and legacy backend transcript events.
- `LEGACY_STT_SESSION_PROJECTION` and the SELF `auto -> legacy` factory branch.
- `LocalASRProviderRuntimeOwner.handle_vad_event` unowned SELF dispatch.
- SELF-only legacy VAD handoff commit and the matching capture/provider adapter calls.
- Legacy SELF event callback projection into `SelfTranslationChannelOwner`.

Issue #144 owns production SELF binding, its contribution/publication policy, callsite migration, and deletion of this facade/removal list. None of these paths is a fallback for LISTEN or the scoped SELF-like fixture.

## Verification and limits

Focused commands and pass counts are recorded from the implementation worktree. Tests use controlled audio, fake SDK transports, fake GPU workers and real runtime owners/adapters unless noted.

- Baseline research receipt: 267 passed across the accepted owner/provider/audio suites before mutation.
- `.venv/Scripts/python.exe -m pytest -o addopts= -q tests/providers` — 504 passed. This includes the three concrete two-client Deepgram owner/adapter cases and the concrete two-client local-GPU adapter case, with fakes only at SDK/network/native boundaries.
- `.venv/Scripts/python.exe -m pytest -o addopts= -q tests/core/test_stt_scoped_engine.py tests/core/test_audio_ownership.py tests/core/test_audio_vad_loop.py tests/core/runtime/test_peer_capture_session.py tests/core/test_peer_owned_provider_runtime.py tests/core/test_stt_timed_tokens.py tests/core/test_peer_pretranslation_path.py tests/core/test_pretranslation_ownership.py tests/core/test_translation_ownership_children.py tests/core/test_peer_translation_channel_owner.py` — 126 passed.
- `.venv/Scripts/python.exe -m pytest -o addopts= -q tests/core/runtime/test_local_asr_provider_runtime.py tests/core/runtime/test_gpu_asr.py tests/core/runtime/test_self_capture_session.py tests/core/test_stt_session_projection.py` — 114 passed, including the real shared GPU runtime with two concrete scoped adapters.
- `.venv/Scripts/python.exe -m pytest -o addopts= -q tests/app/test_self_capture_provider_adapter.py tests/app/test_self_capture_vad_adapter.py tests/core/test_self_translation_low_latency.py tests/app/test_wiring_local_asr_provider_runtime.py` — 84 passed.
- Final affected total: 828 passed.
- `.venv/Scripts/ruff.exe check` over every changed Python source and test — all checks passed. Final `compileall` over the affected source packages completed without output/errors.
- A throwaway executable smoke instantiated the real `ScopedRecognitionEngine`, request ledger, event buffer, SELF-like binding and consumption ledger. It observed two stable updates, one terminal `AB`, consumed prefix `A` and suffix `B`, and then the throwaway file was removed. The final source revision's same real runtime scenarios reported `{'shared_runtime_smoke': 'passed', 'projection': 'self-scoped', 'retention': 'buffer_exhausted'}`.

No live provider credentials, provider service quality, real microphone/loopback device, real GPU model decode, timing-performance envelope or physical remote-display acknowledgement is certified by these fake/controlled fixtures.
