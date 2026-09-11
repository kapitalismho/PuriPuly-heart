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
| SELF retained audio | Two simultaneous per-channel ceilings shared by dispatcher arrays, PCM16 conversion, and provider/native copies across current and draining engines: 2,880,000 retained normalized sample-equivalents and 11,520,000 actual bytes. Each distinct copy charges its full sample count and bytes, so float32 plus PCM16 copies of the same $N$ samples cost $2N$ sample-equivalents and $6N$ bytes; aliases reserve once, streaming writes release after progress, and unsafe admission closes speech with `buffer_exhausted`, never a successful split. |
| Recognition request admission | Eight wholly unsent requests and 12 s pre-dispatch age through the shared profile; oldest safely reclaimable unsent work retires first. |
| Recognition control/provider events | 32 reserved control events; 256 native events with provisional coalescing before stable/terminal pressure fails the epoch. |
| Text/language assembly | 1 MiB UTF-8 and 256 language runs. |
| SELF speech translation | Two running parents, eight not-started parents, 12 s from parent admission; all target children share one parent slot. |
| SELF speech output | One active submission and eight unsent speech parent batches, 12 s from output handoff; target-language revisions replace/coalesce inside their parent batch and do not consume extra capacity; `output_overload`/`output_timeout`. |
| Contribution/publication identity | Contribution identity retained through terminality with bounded 4,096 request identities; existing committed-publication tombstones remain bounded. |
| Manual/peer | Not candidates for SELF speech expiry/eviction; peer destinations and generations remain separate. |

## Exact AS01-AS15 evidence map

Each row restates the original #144 obligation; no row is renumbered or substituted.

| ID | Original required observable proof | Named executable evidence and result |
| --- | --- | --- |
| AS01 | Current SELF onset/prefix/hangover, uneven frames/residue and mute known/unknown/grace states match baseline; masked audio never leaks. | `test_adapter_constructs_engine_and_exact_self_gating_policy`, `test_vad_gating_emits_start_and_end_with_hangover`, `test_vad_gating_pre_roll_contains_previous_audio`, `test_run_audio_vad_loop_applies_audio_gate_before_forwarding_to_sink`, and `tests/core/test_vrc_mic_gate.py`: PASS. |
| AS02 | Long ongoing speech does not acquire LISTEN's policy; retained PCM/metadata is bounded and resource exhaustion is explicit failure, not silent success splitting. | `test_vad_gating_default_max_segment_disabled_does_not_force_continuous_speech`, `test_self_retention_profile_uses_exact_selected_sample_ceiling`, `test_retention_budget_counts_alias_once_and_distinct_copies_at_actual_bytes`, `test_retention_budget_rejects_sample_equivalent_limit_before_byte_limit`, `test_shared_budget_counts_old_local_retention_against_new_scoped_engine`, `test_streaming_scoped_engine_releases_native_copy_after_each_write`, and `test_production_self_adapter_routes_exact_pressure_terminals_and_preserves_other_work`: PASS. Together they enforce simultaneous 2,880,000-sample-equivalent and 11,520,000-byte SELF-channel ceilings across dispatcher, PCM16 conversion, and provider/native retention, including overlapping old/new engines and local float32 versus PCM16 copies. |
| AS03 | Stable-before-end and end-before-final each retain speculative/grace behavior without blocking source progress. | Stable/end ordering cases in `test_self_translation_low_latency.py`, `test_scoped_successor_before_real_end_uses_post_end_grace`, `test_bound_event_sink_does_not_block_speech_end_on_downstream_delivery`, and `test_self_recognition_admission_keeps_exactly_eight_unsent_and_expires_by_original_age`: PASS. The direct real-owner smoke below also observed source/VAD completion with six owned events queued while the scoped provider write remained blocked. |
| AS04 | Multiple stable contributions, cumulative terminal inclusion, duplicate native event and disjoint repeated words conserve intended text exactly. | Contribution conservation/duplicate cases in `test_stt_scoped_engine.py`, `test_scoped_request_keeps_suffix_after_early_publication`, and `test_repeated_text_from_distinct_requests_is_preserved`: PASS. |
| AS05 | Awaiting-end watchdog publishes A while request remains open; later B survives and terminal A+B does not republish A or fabricate a new onset. | `test_scoped_request_keeps_suffix_after_early_publication` publishes A, opens a distinct successor for B, consumes duplicate terminal A+B as empty, and retains the single acoustic segment identity. `test_scoped_successor_before_real_end_uses_post_end_grace` proves that B-before-real-`SpeechEnd` transfers the endpoint to the successor publication and uses post-end grace instead of the awaiting-end watchdog: PASS. |
| AS06 | Pending/confirmed resume, real no-text resume, true duplicate update, new identical speech and late terminal all resolve without stale reuse or hanging merge. | Resume/no-final/no-text/duplicate/late-final cases in `test_self_translation_low_latency.py` plus `TestLowLatencyDisjointSpeech`: PASS; retired overlap-helper expectations were deleted. |
| AS07 | LLM generation/config changes invalidate ready/running speculation; compatible speculation is reused; fallback completion cannot hang or publish stale output. | `test_recognition_configuration_change_is_merge_barrier_but_epoch_rotation_is_not` proves distinct provider-setting scopes cannot mix in one publication while same-scope epoch rotation does not split. `test_replaced_llm_provider_late_spec_completion_cannot_update_low_latency_state`, `test_replaced_llm_provider_late_spec_completion_falls_back_without_hanging_commit`, and `test_ready_spec_result_is_invalidated_by_provider_replacement_during_grace` prove speculation invalidation/fallback: PASS. |
| AS08 | Smooth handoff retains allowed old-request finals; destructive restart/OFF rejects them. Effective runtime signatures advance only on actual convergence. | `test_local_scope_change_and_capture_restart_replace_engines_without_losing_recognition` proves a local source-language change installs an engine with the requested immutable scope, the next utterance reaches it, a changed-device restart replaces the drain-closed engine, the pre-restart turn reaches its scoped terminal boundary, and recognition succeeds again. `test_failed_scoped_language_handoff_does_not_look_applied` covers cloud and local rollback. The full vertical apply test, `test_running_provider_handoff_preserves_capture_and_prior_state_on_non_apply`, `test_retired_provider_terminal_failure_cannot_fault_handoff_session`, and provider-handle scoped handoff tests also pass. |
| AS09 | TALK OFF during every phase cancels speech only; existing and newly submitted manual text remains usable. LISTEN is unaffected; global shutdown still retires all scopes. | `test_cloud_explicit_toggle_off_routes_to_abort_at_each_delay` (0/0.1/0.3/1.0 s), `test_self_provider_reset_cancels_speech_without_erasing_manual_or_peer_state`, `test_self_speech_cancellation_leaves_manual_parent_running`, shared GPU channel-disable tests, and the direct smoke's before/after manual messages: PASS. |
| AS10 | Dual-target secondary-first, primary failure, later parent ready first and progressive chatbox revision cases retain current execution/projection order. | `test_end_to_end_secondary_first_publishes_progressive_parent_snapshots`, `test_dual_target_observability_records_all_terminal_time_after_failure`, `test_newer_transcript_visibility_suppresses_older_primary_latest_surfaces`, and `test_newer_self_parent_starts_while_older_secondary_is_in_flight`: PASS. |
| AS11 | Source-only, empty, suppression, failure, expiry and sink rejection retire semantic/parent obligations without duplicate output or false remote-delivery claims. | `test_dual_target_source_only_children_publish_one_parent_fallback` proves two target-child fallbacks produce one observable parent message. Terminal-path cases in `test_translation_turn_owner.py`, output rejection/failure cases in `test_output_runtime.py`, `test_failed_page_is_dropped_without_retrying_or_blocking_later_pages`, and scoped empty/failed terminals: PASS. |
| AS12 | Queue/sample/text/parent/output caps and TTLs hold at exact limits; provider/native/LLM stalls do not silently drop capture or restart retry budgets. | `test_self_recognition_admission_keeps_exactly_eight_unsent_and_expires_by_original_age` proves two overload evictions at the 11th/12th admissions and TTL from original source age. `test_production_self_adapter_routes_exact_pressure_terminals_and_preserves_other_work` repeats capacity, TTL and buffer pressure through `SelfCaptureVadSinkAdapter` → `SelfTranslationChannelOwner` → settings-scoped local-ASR routing → `ScopedRecognitionEngine`, observes all 11 scoped failure UI reports, and preserves manual/peer state. `test_self_speech_queue_has_two_running_and_eight_waiting_parents` proves two running plus exactly eight not-started parents and source-only overload retirement; `test_self_speech_waiting_parent_expires_to_observable_source_only` proves source-only TTL retirement. Exact scoped PCM/native/text bounds, paginator parent-revision replacement/capacity tests, and GPU pending/TTL/retry cases also pass. |
| AS13 | Shared GPU leases/order/cancellation/actual process failure affect only the correct set; CPU/cloud independent failures stay isolated. | `test_one_worker_and_model_are_shared_until_last_channel_deactivates`, `test_channel_disable_discards_only_its_work_and_retains_shared_worker`, `test_shared_gpu_keeps_peer_and_self_owned_during_self_stall_and_peer_release`, process-failure/recovery tests in `test_gpu_asr.py`, and `test_self_and_peer_cloud_setup_overlap_without_cross_channel_eviction`: PASS. |
| AS14 | All concrete-provider completion fixtures run through the same implementation as LISTEN; only product consumers/profiles differ. | `test_protocol_a_scoped_sessions.py`, scoped Deepgram/Gemini/Scribe/Soniox/Qwen/custom/local CPU/local GPU provider suites, `test_wiring_local_asr_provider_runtime.py`, and production request wiring: PASS for all configured selectors. No live-provider quality claim. |
| AS15 | Final composition has no production SELF legacy recognition engine; LISTEN #134 conformance and accepted #136 behavior remain unchanged after migration. | Composition/factory exclusivity tests, absence search for `core.stt.controller`/`ManagedSTTProvider`, shared/LISTEN regression suites, and the post-repair non-PSEM repository selection all pass. The exact exclusions and totals are disclosed below; no unqualified whole-suite or live-provider quality claim is made. |


## Direct non-pytest real-owner smoke

A temporary script was executed and then deleted. It used the real `run_audio_vad_loop`, SELF generation guard and `PeerAudioSegmentLedger`, `ScopedRecognitionEngine`, `SelfTranslationChannelOwner`, `TranslationTurnLifecycleOwner`, and `OutputRuntime`. Its controlled provider published stable text and then blocked its first write while six owned VAD events accumulated; the source loop was required to finish before that write was released. Releasing it crossed a deliberately reduced retained-PCM bound.

Command: `uv run --frozen python -m scripts._audio_self_repair_smoke` (the throwaway module was removed immediately after the successful run).

Observed JSON:

```json
{"source_completed_while_provider_blocked":true,"queued_while_blocked":6,"session_send_count":1,"capture_state":"faulted","capture_failure":"session_failed","buffer_terminal_count":1,"buffer_terminal_outcomes":["degraded"],"ledger_live_segments":0,"session_state":"DISCONNECTED","ui_types":["SESSION_STATE_CHANGED","ERROR","SESSION_STATE_CHANGED","TRANSCRIPT_FINAL","OSC_SENT","TRANSCRIPT_FINAL","OSC_SENT"],"chatbox":["spoken","manual-after-failure"],"manual_usable":true,"source_closed":1}
```

The run proves source/VAD progress did not await blocked provider I/O, buffer exhaustion immediately produced one failure-bearing degraded terminal for the stable text, the source ledger had no live segments afterward, capture closed/faulted, session/UI failure state was actionable, and independently submitted manual output still published. It is controlled ownership/order/failure evidence, not physical-device or live-provider latency evidence.

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

- Dispatcher-terminal wiring repair: production adapter/unit pressure nodes plus the raw-user-error architecture guard: **11 passed in 3.11 s**.
- Exact production adapter pressure node: **1 passed in 0.93 s**; it asserts the production 8-request/12-second limits before accelerating the TTL clock for deterministic execution.
- Full architecture suite after the typed-error repair: **232 passed in 65.22 s**.
- Directly impacted SELF adapter, capture, settings-scoped local-ASR routing, scoped recognition, translation owner/low-latency/output and runtime-composition suites: **269 passed in 7.44 s**.
- Final repair-focused owner/scoped-engine/capture/translation/output/composition/lifecycle command (the 11 modules named in the delivery report): **251 passed in 6.82 s**.
- LISTEN/Smart Turn isolation command over `test_audio_vad_loop.py`, `test_peer_capture_session.py`, `test_smart_turn_delivery.py`, `test_smart_turn_runtime.py`, `test_vrc_mic_gate.py`, and `test_local_asr_provider_runtime.py`: **149 passed in 17.01 s**.
- Concrete-provider command over `tests/providers`, production local-ASR wiring, and the local-ASR architecture contracts: **551 passed in 14.68 s**.
- Channel-specific immediate/deferred failure timing plus degraded-session and dual-target fallback nodes: **5 passed in 13.02 s**.
- The non-pytest command and exact JSON above are the final end-to-end failure-path smoke proof.
- Terminal-review candidate evidence: the reviewer-matched non-PSEM/non-integration selection finished with **5911 passed, 8 skipped** after one untouched GPU lifecycle-diagnostics teardown race did not reproduce in the isolated node (4/4), full file (35/35), or repeated selection.
- Post-repair evidence with the same selection and exclusions finished with **5916 passed, 8 skipped, 0 failed in 214.38 s**. Command: `PYTHONDONTWRITEBYTECODE=1 uv run --frozen pytest -p no:cacheprovider --ignore=tests/integration --ignore=tests/core/test_psem_r2_budget.py --ignore=tests/core/test_psem_r2_live_runner.py --ignore=tests/core/test_psem_r2_metrics.py --ignore=tests/core/test_psem_r2_offline_pipeline.py`.
- The focused repair command covering dual retained-audio ceilings, identity/copies, overlapping old/new engines, streaming release, production adapter pressure, immutable local/cloud handoff rollback, peer callback retirement, local next-speech convergence, and restart-after-drain recognition finished with **67 passed, 0 failed in 6.23 s**.
- `uv run ruff check` over every repaired Python source and regression-test path: **All checks passed**.
- `pyright` is not installed and no Python type-checker is configured; no static-type PASS is claimed.
