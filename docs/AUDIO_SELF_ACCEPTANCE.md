# AUDIO-SELF-1 acceptance receipt

## Authority and revision identity

- Contract: [#144](https://github.com/kapitalismho/PuriPuly-heart/issues/144), `AUDIO-SELF-1`.
- Inspected development pin required by #144: `4e967df9d03649106faa8348c3ec611009529ffe`.
- Accepted Audio Core implementation: `41210e2fd36874280996526a83b3039e0618b53a`.
- Accepted #143 shared implementation: `2811638c4d37eb8d0943ff3c64312b7f1bc6489f`; independent whole-Goal verification revision: `05fc7f0c6fefd8f3949e74a96e93647ec5bd9e2a`.
- #144 working-tree input baseline: `28a87a90bba38f809182bb1f7337f31b4ee7b826`.
- LISTEN regression authorities: `docs/AUDIO_LISTEN_ACCEPTANCE.md` and `docs/AUDIO_SMART_TURN_ACCEPTANCE.md`.
- Environment: Windows 11 x64 with the repository frozen `uv` environment.

Production now constructs `SharedSTTProviderFactory`. SELF and LISTEN both use `ScopedRecognitionEngine`, separate channel epochs/queues/request state/cancellation/retention profiles, and the same concrete provider sessions. The existing CPU/GPU physical-resource owners remain shared.

## Installed path and exact SELF configuration

```text
microphone callback -> mute gate -> normalized 16 kHz mono frames
-> SELF Silero VAD -> generation-owned source ranges and segment identity
-> nonblocking serial SELF dispatcher -> ScopedRecognitionEngine
-> scoped stable/update/terminal consumer -> SELF merge/speculation
-> TranslationTurnLifecycleOwner -> OutputRuntime -> UI/chatbox/overlay
```

The checked-in configuration selects `local_cpu_auto`, source `ko`, primary target `en`, no secondary target, threshold `0.4`, 500 ms Fast Translation hangover, 500 ms pre-roll, 600 ms merge gap, 10 speculative retries, 400 ms finalize wait, and 3.0 s awaiting-VAD/resume-end watchdog. SELF has no LISTEN age step, hard rollover, Smart Turn, or successful 180-second endpoint.

Cloud routes require their provider credential/network. Local CPU routes require their selected model artifact. `local_qwen_gpu` requires a compatible worker/device. Custom routes require a compatible endpoint/model. Physical microphone, paid-provider, packaged-app, live VRChat, WER, and translation-quality checks were unavailable and are not marked PASS.

## Preserved versus corrected behavior

| Preserve | Deliberately replace |
| --- | --- |
| SELF actual onset/hangover/mute and no ordinary LISTEN time segmentation | Blocking provider/LLM/output work on the acoustic consumer |
| Early stable-driven speculation, merge/resume, post-end grace and awaiting-end fallback | Treating each native final as one completed recognition request |
| Reuse only matching speculative text/config/provider; current fallback retranslation | Whole-request tombstones losing a suffix after early publication; string-only deduplication of separate speech |
| Explicit TALK OFF versus permitted internal drain; existing staged apply intent | Blanket dropped graceful-handoff finals in low-latency mode; incidental manual cancellation on speech-only reset |
| Dual-target concurrent execution/progressive output and current primary projection | Unbounded backlog/native replacement and successful-empty disguises for failures |
| Existing shared GPU scheduler/leases and CPU backend ownership | Duplicate SELF protocol/reconnect/cleanup implementation after cutover |
| Existing permitted output destinations, history and system disclosure | Ambiguous cross-epoch bridge replay, unsafe ACK promotion and stale result publication |

## Provider selector inventory

The production selector inventory is `local_cpu_auto`, `local_parakeet_v3`, `local_parakeet_ja`, `local_qwen`, `local_qwen_gpu`, `deepgram`, `gemini_transcribe`, `elevenlabs_scribe`, `qwen_asr`, `qwen_audio`, `soniox`, `rolling_free`, `custom`, `custom_offline`, and `custom_realtime`. Selection aliases and rolling/free policy resolve before `SharedSTTProviderFactory`; every resulting concrete session enters the same scoped engine used by LISTEN.

## Active finite bounds

| Resource | Bound / terminal disposition |
| --- | --- |
| SELF retained PCM | 2,880,000 normalized mono sample-equivalents; unsafe admission closes speech with `buffer_exhausted`, never a successful split. |
| Recognition request admission | Eight wholly unsent requests and 12 s pre-dispatch age through the shared profile; oldest safely reclaimable unsent work retires first. |
| Recognition control/provider events | 32 reserved control events; 256 native events with provisional coalescing before stable/terminal pressure fails the epoch. |
| Text/language assembly | 1 MiB UTF-8 and 256 language runs. |
| SELF speech translation | Two running parents, eight not-started parents, 12 s from parent admission; all target children share one parent slot. |
| SELF speech output | One active submission and eight unsent speech messages, 12 s from output handoff; `output_overload`/`output_timeout`. |
| Contribution/publication identity | Contribution identity retained through terminality with bounded 4,096 request identities; existing committed-publication tombstones remain bounded. |
| Manual/peer | Not candidates for SELF speech expiry/eviction; peer destinations and generations remain separate. |

## Exact AS01-AS15 evidence map

Each row restates the original #144 obligation; no row is renumbered or substituted.

| ID | Original required observable proof | Named executable evidence and result |
| --- | --- | --- |
| AS01 | Current SELF onset/prefix/hangover, uneven frames/residue and mute known/unknown/grace states match baseline; masked audio never leaks. | `test_adapter_constructs_engine_and_exact_self_gating_policy`, `test_vad_gating_emits_start_and_end_with_hangover`, `test_vad_gating_pre_roll_contains_previous_audio`, `test_run_audio_vad_loop_applies_audio_gate_before_forwarding_to_sink`, and `tests/core/test_vrc_mic_gate.py`: PASS. |
| AS02 | Long ongoing speech does not acquire LISTEN's policy; retained PCM/metadata is bounded and resource exhaustion is explicit failure, not silent success splitting. | `test_vad_gating_default_max_segment_disabled_does_not_force_continuous_speech`, `test_self_retention_profile_uses_exact_selected_sample_ceiling` (2,880,000 samples/11,520,000 actual bytes), and `test_self_like_binding_has_no_time_cut_and_fails_at_retained_pcm_bound`: PASS. |
| AS03 | Stable-before-end and end-before-final each retain speculative/grace behavior without blocking source progress. | Stable/end ordering cases in `test_self_translation_low_latency.py`, `test_late_and_stale_generation_callbacks_cannot_reach_self_sink`, and blocked-provider capture cases in `test_self_capture_session.py`: PASS. The direct real-owner smoke below also observed source/VAD completion before scoped terminal publication. |
| AS04 | Multiple stable contributions, cumulative terminal inclusion, duplicate native event and disjoint repeated words conserve intended text exactly. | Contribution conservation/duplicate cases in `test_stt_scoped_engine.py`, `test_scoped_request_keeps_suffix_after_early_publication`, and `test_repeated_text_from_distinct_requests_is_preserved`: PASS. |
| AS05 | Awaiting-end watchdog publishes A while request remains open; later B survives and terminal A+B does not republish A or fabricate a new onset. | `test_scoped_request_keeps_suffix_after_early_publication` publishes A, opens a distinct successor for B, consumes duplicate terminal A+B as empty, and retains the single acoustic segment identity: PASS. |
| AS06 | Pending/confirmed resume, real no-text resume, true duplicate update, new identical speech and late terminal all resolve without stale reuse or hanging merge. | Resume/no-final/no-text/duplicate/late-final cases in `test_self_translation_low_latency.py` plus `TestLowLatencyDisjointSpeech`: PASS; retired overlap-helper expectations were deleted. |
| AS07 | LLM generation/config changes invalidate ready/running speculation; compatible speculation is reused; fallback completion cannot hang or publish stale output. | `test_replaced_llm_provider_late_spec_completion_cannot_update_low_latency_state`, `test_replaced_llm_provider_late_spec_completion_falls_back_without_hanging_commit`, and `test_ready_spec_result_is_invalidated_by_provider_replacement_during_grace`: PASS. |
| AS08 | Smooth handoff retains allowed old-request finals; destructive restart/OFF rejects them. Effective runtime signatures advance only on actual convergence. | `test_provider_apply_intent_full_vertical_rolling_gemini_soniox_reverse_and_speech_end`, `test_running_provider_handoff_preserves_capture_and_prior_state_on_non_apply`, `test_retired_provider_terminal_failure_cannot_fault_handoff_session`, and provider-handle scoped handoff tests: PASS. |
| AS09 | TALK OFF during every phase cancels speech only; existing and newly submitted manual text remains usable. LISTEN is unaffected; global shutdown still retires all scopes. | `test_cloud_explicit_toggle_off_routes_to_abort_at_each_delay` (0/0.1/0.3/1.0 s), `test_self_provider_reset_cancels_speech_without_erasing_manual_or_peer_state`, `test_self_speech_cancellation_leaves_manual_parent_running`, shared GPU channel-disable tests, and the direct smoke's before/after manual messages: PASS. |
| AS10 | Dual-target secondary-first, primary failure, later parent ready first and progressive chatbox revision cases retain current execution/projection order. | `test_end_to_end_secondary_first_publishes_progressive_parent_snapshots`, `test_dual_target_observability_records_all_terminal_time_after_failure`, `test_newer_transcript_visibility_suppresses_older_primary_latest_surfaces`, and `test_newer_self_parent_starts_while_older_secondary_is_in_flight`: PASS. |
| AS11 | Source-only, empty, suppression, failure, expiry and sink rejection retire semantic/parent obligations without duplicate output or false remote-delivery claims. | Terminal-path cases in `test_translation_turn_owner.py`, output rejection/failure cases in `test_output_runtime.py`, `test_failed_page_is_dropped_without_retrying_or_blocking_later_pages`, and scoped empty/failed terminals: PASS. |
| AS12 | Queue/sample/text/parent/output caps and TTLs hold at exact limits; provider/native/LLM stalls do not silently drop capture or restart retry budgets. | Exact limit cases in `test_stt_scoped_engine.py`, `test_self_speech_queue_has_two_running_and_eight_waiting_parents`, `test_self_speech_waiting_capacity_evicts_oldest_without_dropping_manual`, `test_self_speech_waiting_ttl_expires_speech_but_preserves_manual`, and GPU pending/TTL/retry cases: PASS. |
| AS13 | Shared GPU leases/order/cancellation/actual process failure affect only the correct set; CPU/cloud independent failures stay isolated. | `test_one_worker_and_model_are_shared_until_last_channel_deactivates`, `test_channel_disable_discards_only_its_work_and_retains_shared_worker`, `test_shared_gpu_keeps_peer_and_self_owned_during_self_stall_and_peer_release`, process-failure/recovery tests in `test_gpu_asr.py`, and `test_self_and_peer_cloud_setup_overlap_without_cross_channel_eviction`: PASS. |
| AS14 | All concrete-provider completion fixtures run through the same implementation as LISTEN; only product consumers/profiles differ. | `test_protocol_a_scoped_sessions.py`, scoped Deepgram/Gemini/Scribe/Soniox/Qwen/custom/local CPU/local GPU provider suites, `test_wiring_local_asr_provider_runtime.py`, and production request wiring: PASS for all configured selectors. No live-provider quality claim. |
| AS15 | Final composition has no production SELF legacy recognition engine; LISTEN #134 conformance and accepted #136 behavior remain unchanged after migration. | Composition/factory exclusivity tests, absence search for `core.stt.controller`/`ManagedSTTProvider`, shared/LISTEN regression suites, and the final non-PSEM scoped repository suite: PASS. The unrelated exclusion is disclosed below; no unqualified whole-suite PASS is claimed. |


## Direct non-pytest real-owner smoke

A temporary script was executed and then deleted. It used the real `run_audio_vad_loop`, `PeerAudioSegmentLedger`, `ScopedRecognitionEngine`, `SelfTranslationChannelOwner`, `TranslationTurnLifecycleOwner`, and `OutputRuntime`, with a controlled three-frame source/VAD boundary and controlled scoped session.
Command: `uv run python scripts/_audio_self_owner_smoke.py` (the throwaway file was removed immediately after the successful run).


Observed JSON:

```json
{"trace":["source","vad","owned-segment","scoped-stable","scoped-terminal","self-publication","speech-only-off","manual-publication"],"session_stable":true,"vad_chunks":3,"segments":1,"scoped_terminal":true,"session_terminal":true,"scoped_events":["STTProviderTurnTerminal"],"dispatch_done":false,"terminal_texts":["spoken"],"terminal_outcomes":[["final",null]],"dispatch_error":null,"history":[],"speech_ui":["TRANSCRIPT_FINAL","OSC_SENT"],"chatbox":["spoken","manual-before-off","manual-after-off"],"manual_history":[],"speech_state_after_off":[]}
```

The scripted `scoped-stable` trace label denotes successful scoped audio admission (`session_stable=true`); the literal `scoped_events` field shows that this controlled session emitted only its authoritative terminal. The run proves the controlled source reached scoped recognition and SELF UI/chatbox publication, TALK OFF removed speech state, and independently submitted manual output remained usable both before and after OFF. It is ownership/order evidence, not physical-device or provider-latency evidence.

## S3 retirement inventory

Deleted obsolete lifecycle/product tests:

- `src/puripuly_heart/core/stt/controller.py` — the entire `ManagedSTTProvider` final-FIFO/session/reset/reconnect/bridge/event lifecycle.
- `tests/core/test_stt_controller.py` — tests of the deleted lifecycle.
- `tests/core/test_orchestrator_pipeline.py` — legacy controller pipeline tests replaced by scoped real-owner/provider tests.
- `tests/providers/test_gemini_transcribe_ownership.py` — legacy-controller Gemini ownership fixtures superseded by scoped Gemini/protocol and shared-engine suites.
- `tests/integration/test_e2e_latency_measurement.py` and `tests/integration/test_qwen_asr_llm_integration.py` — legacy controller integrations; no obsolete alternate production entrypoint remains.
- Legacy-only controller cases in `tests/core/test_audio_vad_loop.py` and `tests/app/test_wiring_providers.py`, plus substring-overlap and retired-legacy-final assertions, were removed rather than repinned.

`FinalTranscriptSuppressedNotification` moved to `core/stt/notifications.py`; it is a data notification, not a recognition owner. Concrete provider adapters still normalize their native protocol through `STTSessionEventProjection`, but production factory construction is scoped-only and there is no remaining `ManagedSTTProvider`, legacy SELF controller, reconnect loop, bridge owner, or final-FIFO consumer. Provider-specific native completion code remains because it is the common scoped implementation used by both channels, not a second lifecycle.

## Verification commands

- `uv run --frozen pytest -o addopts= -q` over the 28 named SELF capture/VAD, scoped engine, low-latency/translation/output, provider-apply, GPU/local runtime, every concrete-provider scoped completion, Smart Turn and peer-capture modules used in the AS rows: **738 passed in 30.95 s**.
- `uv run --frozen pytest -o addopts= -q tests/core/test_vrc_mic_gate.py`: **14 passed in 0.14 s**.
- Exact SELF ceiling plus reduced-limit exhaustion nodes: **4 passed in 0.66 s**.
- `uv run --frozen pytest --ignore=tests/integration --ignore=tests/core/test_psem_r2_live_runner.py --ignore=tests/core/test_psem_r2_budget.py --ignore=tests/core/test_psem_r2_offline_pipeline.py --ignore=tests/core/test_psem_r2_metrics.py`: **5900 passed, 8 skipped in 212.02 s**.
- No unqualified whole-suite PASS is claimed. The first current-tree run produced **5952 passed, 8 skipped, 2 failed**: the deleted controller remained in an architecture test's inspected-path tuple (fixed; its exact node then passed) and unrelated concurrent `test_psem_r2_live_runner.py::test_network_runner_reserves_before_sdk_open_and_llm` failed. Its traceback first rendered `approx(deepgram[0][\"settled_usd\"])`; the immediate rerun rendered source `approx(0.0)` while rewritten failure data still expected `0.00025666666666666665`, demonstrating moving PSEM source. After excluding that module, `test_psem_r2_budget.py::test_healthy_meeting_reserves_one_pass_and_settles_verified_pcm` also failed inside the same concurrently changed PSEM experiment (`0.00154` versus `0.0012833333333333334`). Those pre-existing/concurrent PSEM files were not changed for #144 and are excluded only from the scoped final verification above.
- `git diff --name-only --diff-filter=ACM -- '*.py' | xargs uv run ruff check`: **All checks passed**.
- `pyright` is not installed and no Python type-checker is configured; no static-type PASS is claimed.
