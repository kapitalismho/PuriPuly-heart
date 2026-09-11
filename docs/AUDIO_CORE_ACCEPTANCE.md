# AUDIO_CORE_ACCEPTANCE

## Authority and reproducibility

- Implementation issue: [#135](https://github.com/kapitalismho/PuriPuly-heart/issues/135), body updated `2026-09-09T08:12:06Z`.
- Consumed contract: [#134 AUDIO-LISTEN-1, C0-C15](https://github.com/kapitalismho/PuriPuly-heart/issues/134), body updated `2026-09-09T07:04:53Z`. Its finalized body supersedes the earlier experimental proposals and scope-only comments.
- Required baseline and implementation start: `4e967df9d03649106faa8348c3ec611009529ffe`.
- Implementation HEAD: `9bd46b4fb147b43491175909eae9df370cdb481b`. This receipt is committed separately from that validated code tree.
- Branch: `audio-a0-audio-processing-architecture-source-ti`. The initially older branch was fast-forwarded to the required baseline with explicit approval. The pre-existing user change in `AGENTS.md` was preserved and excluded from implementation commits.
- Setup: Windows 11 x64, Python 3.12.10, uv 0.9.17; project dependencies resolved with `uv run --frozen --extra dev`. Timeout-enabled suite invocation additionally uses `--with pytest-timeout`.
- Scope: LISTEN/peer base with Smart Turn OFF; AC01-AC02, AC04-AC14, OFF/controller portions of AC03, and retained #141 regressions. No production SELF migration, ON inference, speaker model, retrospective partition, new provider, or alternate rollback engine.

This receipt records implementation and verification evidence, not permission to push, deploy, or release. Vendor conformance and physical-device scheduling certification are not inferred from controlled-provider tests.

## Existing owner to changed seam

Paths below are relative to `src/puripuly_heart/`.

| Existing owner | Clauses | Changed seam | Preserved behavior |
| --- | --- | --- | --- |
| Desktop/process source and `run_audio_vad_loop` | C3-C5, C8 | `core/audio/{source,desktop_source,process_source,streaming_resampler}.py`; `core/runtime/audio_vad_loop.py`: capture coordinates, normalized spans, loss, prefix/context and orderly residue | Shared capture transports and SELF acoustic confirmation policy |
| `PeerCaptureSessionOwner` | C2-C6, C8, C13 | `core/runtime/peer_channel.py`; `core/audio/{ownership,listen_delivery,psem_receiver}.py`: activation ledger, sealed ranges, OFF controller, bounded dispatch and prospective receipts | Existing target resolution, provider attachment and explicit retry/enable control |
| Local-ASR runtime and provider handles | C2, C6-C10 | `core/runtime/local_asr_provider_runtime.py` and existing provider-handle/resource owners: peer scoped ingress, configuration barriers, owned cancellation and quarantine | SELF/manual ingress, CPU/GPU leases, process ownership, local idle retention and #141 Gemini setup/teardown |
| STT controller/factory | C7-C11, C14 | `core/stt/{backend,scoped_engine,scoped_event_buffer,scoped_normalizer}.py`; `app/wiring/wiring_local_asr_provider_runtime.py`: typed request/update/terminal/epoch-end contract and peer factory cutover | Existing resolved providers, credentials, models, language hints, vocabulary and resource configuration |
| Concrete STT adapters | C7, C9-C10 | `providers/stt/`: native request/fragment provenance, actual ordered writes, route-specific completion and bounded cleanup | Compatible native transports and unmigrated SELF behavior |
| Peer translation and `TranslationTurnLifecycleOwner` | C11-C12 | Existing `core/orchestrator/` owners: source-order terminal admission, deterministic language/target children, finite semantic work and predecessor release | Request/scene/context preparation, manual behavior, source-only results and child identity |
| Existing output runtime/history owners | C8, C12 | Existing output owners: per-destination bounded peer handoff, one writer, generation retirement, dedupe and explicit rejection/failure | Caption ordering, newer-caption priority, history ownership and peer chatbox denial |

`ARCHITECTURE.md` is updated to describe these owners and the production scoped route. No new root AudioManager, revived peer-final scheduler, or second Audio LLM/output scheduler was introduced.

## Actual selector and protocol inventory

All 15 values in `config/provider_values.py::STTProviderName` remain supported through actual factory resolution.

| Selector(s) | Concrete path and terminal barrier | Connection/resource disposition |
| --- | --- | --- |
| `local_cpu_auto` | Existing catalog/language resolver selects the local CPU decoder; one sealed PCM decode job | Existing loaded-model lease; expiry does not free a running recognizer |
| `local_parakeet_v3`, `local_parakeet_ja` | Parakeet CPU adapters and shared local decode job completion/failure | No duplicate model or overlapping decode on quarantined lease |
| `local_qwen` | Existing local Qwen CPU selection and scoped decode completion, including explicit suppression | Compatible hotwords/model configuration and native ownership |
| `local_qwen_gpu` | Existing shared GPU worker/request path; scoped request completion | Existing process-safe cancellation/device admission; no replacement while native work remains owned |
| `deepgram` | Accumulate stable `is_final` fragments; ordered Finalize and `from_finalize` barrier. Missing ACK uses CloseStream and a second resolved drain interval | Unresolved/disordered completion is explicit; no generic-final-to-segment FIFO. Unkeyed terminal completion retires the epoch before subsequent input |
| `gemini_transcribe` | Authoritative input transcription and protocol completion are distinct barriers. ACK alone cannot promote interim; expiry retains authoritative text or uses only the declared degraded interim fallback | Missing barrier retires the epoch. Real SDK unkeyed message shapes do not authorize cross-turn reuse; #141 setup/teardown ownership remains |
| `elevenlabs_scribe` | Manual commit; committed transcript, including authoritative empty, is terminal. Partial/timestamp variants do not create extra terminals | Unkeyed completed epoch retires before another turn |
| `soniox` | Ordered final tokens/language runs; only the pending manual finalize's `<fin>` is its barrier | Unsolicited markers do not create finals; unkeyed completion and unresolved timeout retire the epoch |
| `qwen_asr` | Commit/item identity; completed/failed transcription resolves that item | Native item IDs are required for safe correlation; missing IDs fail visibly. Session-finish budget is teardown, not the final timer |
| `qwen_audio` | Native task identity, stable-sentence assembly and `task-finished` | Task barrier precedes the next task; keepalive is context-only; native error categories remain |
| `custom_offline` | One sealed PCM HTTP request; completion/error resolves that request | Existing HTTP phase limits plus finite logical total; no HTTP wait on acoustic consumption |
| `custom_realtime` | Client-owned commit; keyed native item completion where available | Omitted turn detection resolves to null; incompatible explicit server-turn configuration is rejected. Keyed reuse is scoped; unkeyed completion retires before the next turn |
| `custom` | Existing mode alias resolves to `custom_offline` or `custom_realtime` | Uses the selected concrete contract, never a generic fallback |
| `rolling_free` | Existing configured member selection resolves to concrete cloud adapters | Preserve member order/quota/cooldown; member switch creates an epoch. No uncertain audio replay or unconfigured paid fallback |

Request identity is independent of native fragment provenance. Provider provisional/stable updates remain representable inside the scoped seam; LISTEN admits only a terminal result. Final wait starts after actual ordered end/commit completion, not SDK enqueue. Idle epoch-end notification also retires the current session before subsequent input; stale/foreign epoch-end callbacks cannot close a replacement.

## Source, timer and terminal observations

Controlled sources/providers exercise actual owners rather than treating a CUT log or transport enqueue as proof:

- Finite uneven 16 kHz PCM frames of 333, 517 and 1450 samples passed through `PeerCaptureSessionOwner`, `run_audio_vad_loop`, owned ingress and `ScopedRecognitionEngine`. The terminal source and normalized envelope was exactly `[0, 2300)`, contiguous and attributed to the same activation/capture epoch/segment as the provider terminal. Final text was admitted once; stop closed the source, retired the generation and produced zero late admissions. The sink exposed only owned ingress. The throwaway smoke script was removed.
- No-callback hard-seal probe closed the real accepted range `[0, 1536)`, rather than extending ownership to manufactured silence. Subsequent source content used the successor segment without reacquiring onset or duplicating prefix. The regression is `test_no_callback_deadline_seals_exact_range_and_next_content_rolls_over` in `tests/core/runtime/test_peer_capture_session.py`.
- OFF uses persisted peer hangover before 4 s, immediately preserves/compares observed silence against 224 ms from 4 s, and independently seals at the 6 s deadline. Missing callbacks never advance pause silence. The 64 ms supported-envelope allowance is a requirement under the contract's frame/scheduler assumptions, not measured production certification.
- Source tests cover orderly sub-chunk residue, prefix reused as context rather than new content, known/unknown loss, capture epoch changes, overflow control capacity, immutable terminal snapshots and bounded retired receipts. Normal EOF drains accepted content; abort/loss cannot turn an unresolved provider result into clean empty recognition.
- Blocked setup/write/final/decode probes retain source progress and exact failed/expired scope. Wholly unsent sealed admission has eight slots excluding active/current work; freshness is 12 s from local seal. Physical capture loss is distinct from local-job expiry.
- Idle-end repair probes exercised empty A and final A followed by epoch end and immediate B: B used a fresh provider epoch, retained B identity/text, and did not replay A. Repeated old/foreign callbacks did not retire the replacement.

## Normalization, downstream, settings and PSEM evidence

- `tests/core/test_stt_scoped_engine.py` covers actual write ordering/final-timer start, empty A/late A/duplicate isolation, provisional coalescing, stable overflow, 1 MiB text/256-run bounds, finite retry, configuration/age barriers, idle epoch retirement, and native/cloud quarantine without replacement leaks.
- `tests/providers/test_protocol_a_scoped_sessions.py` uses actual Deepgram/Gemini/Scribe SDK message shapes and documented Soniox payload shapes, not fabricated per-turn IDs. The concrete Qwen/custom/local suites cover their request/item/task/decode barriers and errors. These establish internal protocol behavior, not live vendor conformance.
- Complete text and language-run conservation, honest unknown-language fallback, legitimate identical speech, explicit empty/suppressed/failed/cancelled/degraded outcomes, and source-order parent admission are exercised with the existing translation owners. Deterministic children and predecessor release are retained.
- Peer publication is activation-scoped, including source-only, cancellation, provider-error and local-drain paths. OFF rejects late publication immediately. Each destination has eight unsent parent batches plus one writer; output overload rejects the oldest unsent batch explicitly. Sink failure does not replay ASR/LLM work, and enqueue is not called physical delivery. Completed peer publication dedupe is bounded to 4096 IDs with generation/order rejection of stale callbacks.
- Caption/overlay enablement is destination-only and does not disable LISTEN capture. Peer output remains denied to chatbox. Tests cover source-only behavior, output rejection, newer-caption priority and semantic predecessor release.
- Endpoint settings are snapshotted for the next segment. Blocked/failed provider applies retain truthful current settings and callbacks; successful handoff retires the old scope. Existing persisted drain/model/provider/hint/vocabulary configuration is preserved except the explicit LISTEN migrations in C14.
- Injected prospective speaker evidence exercises applicable cut, already-separated ownership, too-late/invalid evidence, duplicate/retracted evidence and simultaneous reasons. Source ownership around rollover determines the receipt; segment-ID mismatch alone does not reject evidence or force another cut. No live PSEM producer, speaker enrollment, extra confirmation timer, retrospective receiver or holdback was added.

## Verification and coverage

Final code-tree command:

```text
uv run --frozen --extra dev --with pytest-timeout python -m pytest -o addopts= -q --timeout=30 tests/core tests/app tests/providers tests/config tests/architecture tests/release_evidence tests/ui tests/domain tests/scripts
5857 passed, 8 skipped, 268 warnings in 142.86s
```

The expanded suite initially exposed two unmigrated unattended release-evidence sinks; both were migrated to real owned activation/ledger ingress before this passing run. Warnings were 266 Flet `ElevatedButton` deprecations and two API-key UI coroutine warnings; they were not suppressed or represented as audio lifecycle evidence.

Additional executed evidence:

- `uv run --frozen --extra dev python -m puripuly_heart.main gui-startup-check`: exit 0. This is startup-path verification, not visual/physical-device certification.
- Black and Ruff on the final 24 cutover Python files: passed. Earlier integration formatting/lint checks also passed.
- Actual source-to-scoped-terminal smoke above: passed; throwaway script removed.
- Retained #141 tests: `tests/providers/test_gemini_transcribe_lifecycle.py`, `test_gemini_transcribe_ownership.py`, plus the actual provider/runtime integration suites.
- Manual/SELF regressions include clipboard/manual fallback, generic manual/self/peer terminal admission, provider-apply vertical flow, SELF adapter/source ownership, and shared local CPU/GPU runtime tests.

| Acceptance | Evidence surface |
| --- | --- |
| AC01 | Baseline record; application/UI/config/provider-selector suites; SELF/manual regressions |
| AC02 | `tests/core/test_audio_ownership.py`, `test_audio_vad_loop.py`; peer capture session tests and finite-source smoke |
| AC03, OFF/controller only | Peer capture no-callback/rollover/profile/settings tests and independent hard-seal smoke |
| AC04 | Desktop/process source, resampler/audio loop, source ownership and capacity/discontinuity tests |
| AC05 | Actual capture-owner blocked setup/write paths; scoped engine ordered-write/late-terminal tests |
| AC06 | Concrete provider suites and actual-shape protocol tests; live limitations explicitly retained |
| AC07 | Deepgram two-drain and Gemini distinct-authoritative/protocol barrier cases |
| AC08 | Scoped engine retry/age/config/idle-end tests; rolling provider selection/recovery tests |
| AC09 | Capture OFF/restart, runtime/provider handle, scoped quarantine, translation/output and #141 lifecycle tests |
| AC10 | Capture queue/TTL/control cases; scoped event/text/run limits; existing output capacity/dedupe tests; shared-resource cleanup tests |
| AC11 | Scoped normalization and existing translation-turn/language-run/source-order tests |
| AC12 | Translation/output streaming and projection tests; application overlay/peer-control tests; release-evidence generation gate |
| AC13 | Prospective speaker receiver cases through actual peer capture ownership, including rollover |
| AC14 | Peer capture blocked/failed/successful handoff and endpoint snapshot cases; runtime resolution/serialization and provider-apply tests |
| AC15, retained #141 portion | Gemini lifecycle/ownership suites; broader ON/cross-route stress remains #139's separate scope |

## Removed LISTEN paths and retained compatibility

- Production peer factory uses `ScopedRecognitionEngine`; legacy `ManagedSTTProvider` remains for unmigrated SELF, not as a peer fallback.
- Removed `handle_peer_vad_event`, raw peer sink ingress and owned-to-raw fallback. Peer capture requires `handle_owned_vad_event`; runtime legacy VAD and commit-handoff entrypoints reject peer use.
- Removed LISTEN-specific native-final/pending-UUID correlation, duplicate legacy boundary/finalize handoff and old peer VAD delivery cuts from the production path. No legacy/new LISTEN setting remains.
- Migrated production Local ASR, unattended-runtime and Windows process-isolation evidence callers to the current owned contract. Evidence probes do not offer another LISTEN engine.
- Retained shared transports, SELF acoustic/provisional/manual behavior, existing local-resource ownership and compatible concrete provider parsing. #144 owns the later SELF policy migration; #143/#144 own convergence of the temporary SELF compatibility path.

## AUDIO-SHARED-1 handoff

[#143](https://github.com/kapitalismho/PuriPuly-heart/issues/143) can extract the accepted mechanisms rather than create duplicate owners:

| Mechanism | Current seam | Boundary to preserve during extraction |
| --- | --- | --- |
| Source identity/content ledger | `core/audio/ownership.py`, capture spans and `run_audio_vad_loop` | Accepted source vs context, epoch/loss, immutable seal/terminal and order |
| Delivery policy | `core/audio/listen_delivery.py`, segment settings snapshot | Keep LISTEN 4 s/224 ms/6 s policy outside reusable wire/resource components |
| Request/connection execution | `core/stt/scoped_engine.py`, existing runtime/provider handle | Bounded work, actual writer progress, finite recovery, barrier reuse and owned cleanup/quarantine |
| Native protocol adapters | `providers/stt/`, typed contract in `core/stt/backend.py` | Request identity vs native provenance; provisional/stable updates vs request terminality |
| Resolved resource/profile configuration | Existing runtime resolution and STT factory wiring | Compatible credentials/models/hints/drain and CPU/GPU/Gemini resource ownership |
| Normalized terminal seam | `core/stt/scoped_normalizer.py` to existing peer/translation-turn owners | Text/run conservation, explicit outcomes and source-order admission |
| Publication/PSEM client policy | Existing peer/output owners and `core/audio/psem_receiver.py` | Activation-scoped publication and prospective receipts stay client policy, not provider wire policy |

No unresolved base safety decision is deferred to #143. #136 supplies ON inference; #139 expands adversarial composition and actual-device/cross-route/simultaneous-channel evidence; #144 is a separate SELF migration. No new paid vendor calls, packaged-model download or physical audio-device conformance run was performed for this receipt. Those empirical limits do not remove any configured route or certify it using another provider.
