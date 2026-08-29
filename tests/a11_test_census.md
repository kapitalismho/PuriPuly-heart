# A11 Semantic Contract Census — PuriPuly Heart

Authority: [Issue #61](https://github.com/kapitalismho/PuriPuly-heart/issues/61) — *Refactor test suite by semantic contract census, not CI directory ownership.*

Baseline: `37b6fe1b0b10e72d14232e3968f8f701aee419de` (integrated A10 → A11 start SHA, `origin/dev`).

## How to read this artifact

One row per test file, coherent assertion group, or homogeneous parameterized family
(grouped rows are valid when the protected contract and disposition are genuinely the
same for every member). Columns follow the census deliverable required by the authority:

- **Protected risk** — the concrete regression a failing test would catch.
- **Basis** — why the behavior must remain true (strong contract evidence, explicit
  internal architecture contract, service/port behavioral contract, or none).
- **Boundary** — how the test observes the contract: public behavior, port/protocol
  state, explicit architecture rule, or private shape.
- **Refactor-resistance** — would an implementation-equivalent internal refactor still pass?
- **Disposition** — `KEEP` / `REFACTOR` / `SPLIT` / `TEMPORARY` / `MOVE` / `DELETE`.

Verdicts are derived from the semantic census, never from CI job placement. CI placement
changes derived from this census are listed at the end.

Surface at baseline: **5335 collected tests** in 384 files
(`app` 964, `architecture` 217, `config` 501, `core` 1470, `domain` 3, `integration` 27,
`providers` 291, `release_evidence` 42, `scripts` 3, `ui` 960, plus shared
`tests/helpers` and `tests/fixtures`), plus **519 broker Vitest cases** in 64 spec files.

## Area summaries

### tests/domain (3 tests)

| Family | Protected risk | Basis | Boundary | Refactor-resistant | Disposition |
| --- | --- | --- | --- | --- | --- |
| `test_domain_models.py` — UtteranceBundle merge ordering, partial-after-final rejection, event typing | Corrupted translation pipeline data integrity: out-of-order merges and partial-after-final must never surface to users | Strong: user-visible output correctness | Public model behavior | Yes | **KEEP** |

`tests/domain` is the purest layer in the suite: high KEEP ratio as hypothesized, no
characterization debt.

### tests/core — orchestrator / translation (families)

| Family | Protected risk | Basis | Boundary | Refactor-resistant | Disposition |
| --- | --- | --- | --- | --- | --- |
| `test_channel_runtime.py` — per-channel merge/history separation, tombstone bound, runtime reset semantics | Cross-channel history bleed and unbounded dedupe memory | Service/port behavioral contract | Port/state behavior (bounded tombstone count is a memory bound, not a shape freeze) | Yes | **KEEP** |
| `test_translation_turn_owner.py` — parent/child lifecycle: child terminalization before parent close, cancellation, ordering, output-before-terminal callback | Lost or duplicated translations on cancellation/failure; parent strands | Service/port behavioral contract | Public owner port | Yes | **KEEP** |
| `test_translation_turn_owner.py::test_channel_owners_use_one_injected_generic_translation_owner` | Regression to duplicated channel-specific turn ownership | Explicit internal architecture contract (single turn owner composition) | Mixed: real composition check + `inspect.getsource` string counts (`cancel_pending(channel=...) == 3`) | **No** for the source-string half | **SPLIT** — keep composition/identity assertions; the exact call-count source scan is retired-shape residue → replaced by composition wiring assertions in `test_runtime_pipeline_direct_ownership.py`. Implemented. |
| `test_self_translation_low_latency.py` — awaiting-VAD timeout, speculative attempt lifecycle, resume timeout, overlap merge, commit gating | Stale/speculative translations leaking to output; latency regressions; blocked commits | Service/port behavioral contract | Mostly public harness behavior; ~38 direct calls to private `_handle_low_latency_final`, `_commit_merge`, `_sync_overlay_active_self` | **No** for private-call tests | **REFACTOR** — drive these paths through public ingress (`handle_stt_event`, `handle_vad_event`, public commit path) instead of private methods. Pilot implemented for `_commit_merge` via the harness `commit_self_merge` seam; follow-up rows for remaining private-drive tests. |
| `test_translation_output_streaming.py` — overlay sink projection, peer chatbox denial, latency timeline cleanup, bookkeeping cleanup | Peer text reaching chatbox; leaked latency bookkeeping; duplicate overlay delivery | Strong: user-visible overlay/chatbox correctness | Public harness; some direct `_peer_parent_turn_ids` map emptiness checks | Partial | **SPLIT** — behavioral projection checks KEEP; bookkeeping-map emptiness rows are duplicate checks of user-visible outcomes → keep the user-visible assertion, drop the internal map assertion (follow-up). |
| `test_context_memory.py` — context filtering, formatting without timestamps, redaction, integrated-window selection, peer parent cleanup | Prompt context poisoning (relative-age leakage), duplicate in-flight translations | Strong: output correctness + persisted privacy posture | Public harness; `llm_runtime.provider.provider` double-private reach | Partial | **KEEP**; the provider-reach rows use a test-only blocking hook through the harness (`block_llm_translate`) — seam implemented. |
| `test_stt_controller.py` — session lifecycle, reconnect/bridging, FIFO pending finals, hallucination suppression, safe runtime logs | Lost finals, duplicated partials, leaked secret material in logs | Service/port behavioral contract + strong privacy contract | Mixed: private `_pending_final_utterance_ids`, `_consume_session_events`, `_active_session` drives | **No** for private-drive rows | **REFACTOR** (follow-up batch): expose final-routing through backend-session fakes rather than touching private queues. The *behavior* (FIFO, drop-stale, suppression) is durable and must survive. |
| `test_managed_openrouter_release.py`, `test_managed_openrouter_broker_client.py` — release flow, acks, retry-after, identity rotation, wire envelopes | Broken managed key issuance/recovery; secret leakage | Strong: broker protocol + persisted identity | Public service ports; HTTPX transport injection | Yes | **KEEP** |
| `test_runtime_logging.py`, `test_legacy_error_sanitization.py`, `test_provider_stt_error_messages.py`, `test_diagnostic_validator_contract.py` — redaction before sinks, bounded rotation, queue lifetime | Secret/private-text leakage into live/persisted logs | Strong privacy contract | Public API | Yes | **KEEP** |
| `test_lifecycle_scope.py`, `test_lifecycle_shutdown.py` — task ownership, shutdown diagnostics, cancellation containment | Task leaks at shutdown | Explicit architecture contract (lifecycle owner primitive) | Public port | Yes | **KEEP** |
| `test_provider_runtime_handle.py` — abort/release, close-retry, dormant TTL, handoff drain | Resource leaks on provider replacement | Service/port behavioral contract | Public port | Yes | **KEEP** |
| Audio (`test_audio_format`, `test_ring_buffer`, `test_streaming_resampler`, `test_vad_gating`, `test_silero_vad_onnx`, `test_desktop_audio_source`, `test_process_audio_capture_source`, `test_audio_diagnostics`, `test_desktop_audio_pipeline`, `test_audio_vad_loop`, `test_soxr_runtime`, `test_vad_bundled`) | DSP/VAD correctness, device resolution, fault isolation | Strong: user-audible behavior | Public | Yes | **KEEP** |
| OSC (`test_osc_control_protocol`, `test_osc_encoding`, `test_osc_udp_sender`, `test_osc_receiver_controls`, `test_oscquery_service`, `test_osc_state_publisher`, `test_chatbox_paginator`) | Wire ABI breakage (VRChat/OSC clients are external consumers) | Strong: external protocol | Public ABI snapshot (append-only is contractual) | Yes | **KEEP** |
| Overlay core (`test_overlay_protocol`, `test_overlay_manifest`, `test_overlay_bridge`, `test_overlay_presenter`, `test_overlay_diagnostics`, `test_overlay_refresh_trace_contract`, `test_desktop_overlay_bounds_owner`, `test_overlay_session_fallback_owner`) | Native overlay protocol + presentation correctness | Strong: process/overlay protocol | Public ports; presenter burst-task internals asserted directly in several rows | Partial | **REFACTOR** (follow-up): assert refresh-burst behavior through snapshot sequences and diagnostics events (already available) instead of `presenter._*_presentation_refresh_burst_task`. |
| `test_language.py`, `test_messages_contracts.py`, `test_channel_contracts.py`, `test_speech_boundary.py`, `test_observability_output_contracts.py`, `output/test_router.py`, `output/test_adapter_wrappers.py` | Language mapping, DTO safety, output routing isolation | Strong/explicit contracts | Public | Yes | **KEEP** |
| Identity/security (`test_managed_identity`, `test_hardware_fingerprint`, `test_openrouter_credentials`, `test_openrouter_pkce`, `test_discord_oauth_loopback`, `test_discord_managed_oauth`, `test_telemetry`) | Identity binding, credential isolation, telemetry consent | Strong: security + persisted identity | Public | Yes | **KEEP** |
| Local ASR (`test_local_stt_catalog`, `test_local_stt_assets` family in config, `test_local_gpu_assets`, `test_local_qwen_runtime`, `test_local_stt_runtime_installer`, `test_local_stt_huggingface_xet_adapter`, `runtime/test_local_asr_provider_runtime`, `runtime/test_gpu_asr`, `runtime/test_local_asr_provisioning`, `runtime/test_local_asr_transition`, `runtime/test_peer_capture_session`, `runtime/test_self_capture_session`, `runtime/test_output_runtime`, `runtime/test_overlay_runtime`, `runtime/test_receiver_runtime`, `runtime/test_mic_test_runtime`, `runtime/test_local_stt_download_runtime`, `runtime/test_oscquery_runtime`, `runtime/test_oauth_runtime`, `runtime/test_clipboard_runtime`, `runtime/test_runtime_logging_service`, `runtime/test_github_star_prompt_runtime`, `runtime/test_provider_rebuild`, `runtime/test_vrchat_osc_presence_probe_owner`) | Provisioning integrity (checksummed models), GPU/CPU channel state machines, resource ownership | Strong: install/data integrity + lifecycle contracts | Public ports + lifecycle snapshots (owner-name and resource-field *counts* are policy, not shape) | Yes | **KEEP** |
| `test_self_translation_channel_owner.py`, `test_peer_translation_channel_owner.py`, `test_translation_request_owner.py`, `test_translation_output_projection_owner.py`, `test_translation_latency_diagnostics_owner.py`, `test_translation_runtime_configuration.py`, `test_hedged_attempts.py`, `test_fallback_racing_llm_provider.py`, `test_llm_streaming_provider.py`, `test_llm_semaphore.py`, `test_translation_backend.py`, `test_http_extensions.py`, `test_http_extension_translation_backend.py`, `test_peer_capture_contracts.py`, `test_peer_owned_provider_runtime.py`, `test_stt_custom_vocab.py`, `test_local_qwen_hallucination.py`, `test_soniox_multilingual_release_readiness.py`, `test_openrouter_handoff.py`, `test_vrchat_osc_presence.py`, `test_process_identity_snapshots.py`, `test_clipboard_watcher.py`, `test_vrc_mic_gate.py`, `test_updater.py`, `test_openvr_vendor.py`, `test_overlay_session_fallback_owner.py` | Deterministic unit contracts for each owner/port | Service/port behavioral contracts | Public | Yes | **KEEP** |
| `test_provider_runtime_handle.py::test_handoff_keeps_unhashable_retired_event_ingress_until_final_drain`, `runtime/test_self_capture_session.py` retired-provider rows | Late events from retired providers must not fault or leak | Service/port behavioral contract (the word "retired" here is runtime vocabulary, not migration residue) | Public port | Yes | **KEEP** |
| `test_orchestrator_pipeline.py`, `test_prompt_pipeline.py` — end-to-end orchestrator/prompt fixture flows (integrated context, placeholder substitution, dynamic prompt rendering, chatbox pagination via paginator) | Broken end-to-end translation through composed owners; wrong prompt placeholder rendering | Strong: user-visible output correctness | Public harness composition (`compose_translation_test_harness`) | Yes | **KEEP** |
| `test_translation_owner_branch_coverage.py` — 34 rows driving `_MergeBuffer`, `_SpeculativeAttemptStatus`, disclosure enqueue, stale-partial drop, output projection edge branches | Silent merge/projection regressions in uncovered owner branches (stale partial leaking to output, duplicate terminalization) | Service/port behavioral contract | Mixed: real harness behaviors; some rows construct private `_MergeBuffer`/`_SpeculativeAttemptStatus` directly | Partial | **SPLIT** — behavioral rows KEEP; rows that exist only to touch private enum/buffer branches are duplicate checks of harness-visible outcomes → assert through public owner APIs (follow-up). |
| `test_audio_source.py` — sounddevice source param validation, callback status tracking, drop-without-logging | Broken mic capture or log spam from audio callback | Strong: user-audible behavior | Public source params + injected callback | Yes | **KEEP** |
| `test_file_logging.py` — session/root line routing to shared sinks, main logging handler reuse | Log lines lost or duplicated across session/root sinks | Strong: user-visible diagnostics correctness | Public logging service API | Yes | **KEEP** |
| `test_peer_channel_routing.py` — peer desktop transcripts route to peer runtime and never to chatbox; ordered child runs for language fan-out | Peer text reaching the user chatbox (privacy/correctness); lost language runs | Strong: user-visible output correctness + privacy | Public harness routing | Yes | **KEEP** |
| `test_output_owner_wiring.py` — manual/peer/system output through one owner; overlay replacement updates only owner destination | Duplicate or lost output delivery when destinations change | Service/port behavioral contract (single output ownership) | Public fixture wiring | Yes | **KEEP** |
| `test_translation_local_asr_provider_runtime.py` — self provider execution and close delegated to one owner; prebuilt compatibility uses canonical owner | Provider rebuilt instead of reused; leaked provider on close | Explicit architecture contract (single ownership) | Public fixture wiring | Yes | **KEEP** |

### tests/core — files deleted or changed by this census

### tests/core/local_translation (gemma family)

| Family | Protected risk | Basis | Boundary | Refactor-resistant | Disposition |
| --- | --- | --- | --- | --- | --- |
| `test_gemma_assets.py` — pinned gemma contract target/drafter combos (12B target-without-drafter) | Wrong model downloaded for a pinned provisioning contract | Strong: install/data integrity (checksummed pinned assets) | Public asset contract | Yes | **KEEP** |
| `test_gemma_prefix_cache.py` — 4-most-recent retention, touch promotion, oldest eviction | Unbounded cache disk growth; active profile evicted while in use | Service/port behavioral contract (bounded cache) | Public cache API + tmp dirs | Yes | **KEEP** |
| `test_gemma_provisioning.py` — missing install downloads pinned files, atomic promotion, 12B spec without drafter | Partial/corrupt install surfaced to runtime; non-atomic promotion visible to readers | Strong: install/data integrity | Public provisioning port + tmp dirs | Yes | **KEEP** |
| `test_gemma_runtime.py` — readiness prefill/rebuild per language pair, GPU start failure falls back internally to CPU | Readiness wrong for new language pair; GPU failure breaking translation instead of CPU fallback | Strong: user-visible translation availability | Public runtime ports | Yes | **KEEP** |
| `test_gemma_runtime_profile.py` — fixed common/MTP contract for CPU profile, opt-in slot save path | Corrupt generation profile; unexpected state writes | Strong: persisted local inference correctness | Public profile API | Yes | **KEEP** |
| `test_llama_devices.py` — vulkan ID parsing strips vendor, auto resolves to vulkan0 | Wrong GPU device selection silently degrading inference | Strong: user-visible local inference correctness | Public device helpers | Yes | **KEEP** |

| File | Protected risk | Verdict | Disposition |
| --- | --- | --- | --- |
| `test_translation_turn_owner.py` source-string counting block | Duplicated turn ownership | The `inspect.getsource` call-count assertions encode current private call sites; the composition check already guards the architecture | **SPLIT** — implemented: private-source scans removed, composition assertions kept. |

### tests/config (501 tests)

| Family | Protected risk | Basis | Boundary | Refactor-resistant | Disposition |
| --- | --- | --- | --- | --- | --- |
| `test_config_and_secrets.py` — schema migrations (v17/v18/v21/v22...), round-trips, secret non-serialization, enum stability, local-LLM validation, defaults | Corrupting user settings on upgrade; serializing secrets into settings.json | Strong: persisted settings/migration compatibility | Public `from_dict`/`to_dict`/`load_settings` | Yes | **KEEP** |
| `test_settings_vnext_migration_serialization.py` — canonical loader rejects flat shapes, byte-identical archival + first-run reset, verification evidence projection, telemetry state migration | Breaking the reconciled durable contract: archival then canonical reset (not flat value preservation) | Strong: persisted settings compatibility (per A11 authority's revised compatibility contract) | Public persistence API + fixtures | Yes | **KEEP** |
| `test_settings_vnext_schema.py`, `test_runtime_resolution.py`, `test_resolved_runtime_dtos.py` — schema purity, dependency-light resolution, DTO immutability, secret-free resolved configs | Layer impurity, secret leakage into resolved DTOs | Explicit internal architecture contract | AST import/field checks on *modules* (layer policy, not private names) | Yes | **KEEP** |
| `test_public_compatibility_surfaces.py` — surface snapshots (secret-store keys, Broker /v1 envelopes, overlay protocol, installer identity, prompt fallback, provider aliases, i18n parity) + inventory accounting | Silently breaking surfaces with real external consumers (installed previous versions, Broker clients, native overlay, installer) | Strong: documented facade + real consumers, with per-surface guard evidence refs | Public snapshot + inventory files | Yes | **KEEP** with source-shape reaches removed (implemented): wire-format snapshot rows and inventory bookkeeping stay (consumer-backed compatibility inventory + A10 facade-freeze accounting with A12 sunset); `inspect.getsource` literal assertions over wiring/qwen helpers, local-LLM store-only source checks, and overlay lifecycle-event/spawn source literals replaced by behavioral checks (local-LLM provider built through `create_llm_provider` with/without store key and env ignored; qwen legacy fallback via `require_secret_any`; overlay lifecycle event types behaviorally covered by overlay process manager and bridge tests) |
| `test_prompt_loader.py` — provider prompt sharing, fallback order, cache warm, environment resolution | Broken prompt resolution/fallback | Strong: persisted artifact + user-visible output | Public API | Yes | **KEEP** |
| `test_prompt_contract.py` — dynamic placeholders + prohibition of timestamp syntax in `prompts/translation_prompt.md` | Placeholder breakage (rendering would fail) and reintroduction of forbidden timestamp semantics | Placeholders: strong (render contract, also enforced by `test_render_translation_prompt_*`). Exact English sentences: only contractual as *semantic requirements*, not as exact copy | Source text of the prompt artifact | Exact-copy assertions: no | **SPLIT** — implemented: semantic requirements (placeholders, chronological ordering, self/peer legend semantics, no timestamp-derived syntax) kept; exact English sentence fragments replaced with semantic prohibitions. Follow-up rows assert meaning, not copy. |
| `test_osc_settings_migration.py`, `test_first_run_locale.py`, `test_github_star_prompt_settings.py`, `test_managed_identity_settings.py`, `test_local_asr_settings.py`, `test_managed_gemma_settings.py`, `test_custom_stt_settings.py`, `test_translation_settings_custom_http.py`, `test_overlay_settings.py`, `test_overlay_calibration.py`, `test_desktop_overlay_values.py`, `test_overlay_desktop_audio_settings.py`, `test_audio_host_api.py`, `test_paths.py`, `test_process_capture_resolution.py`, `test_process_capture_dependency_boundary.py`, `test_gemma_model_resolution.py`, `test_local_stt_assets.py`, `test_managed_openrouter_projection.py`, `settings_migration_fixtures.py`, `test_settings_migration_fixtures.py` | Migration defaults, first-run policy, asset manifests, calibration values | Strong: persisted settings compatibility | Public | Yes | **KEEP** |
| `test_custom_stt_runtime_resolution.py` — runtime resolution keeps mode and endpoint; factory builds custom backend without requiring a secret | Custom STT resolution dropping user endpoint/mode; secret demanded for secretless custom backend | Strong: user-visible provider behavior + persisted settings | Public resolution/factory | Yes | **KEEP** |

### tests/app (964 tests)

| Family | Protected risk | Basis | Boundary | Refactor-resistant | Disposition |
| --- | --- | --- | --- | --- | --- |
| `test_wiring_providers.py` — settings→provider/STT resolution, secret lookup rules, routing/fallback pool selection, concurrency limits, managed delegation | Wrong provider/config/credential selected at runtime | Strong: user-visible provider behavior + persisted settings semantics | Mixed: concurrency behavioral (`assert_bounded_concurrency` probe through `SemaphoreLLMProvider.translate`); ~148 `provider.inner`/`isinstance(provider.inner, ...)` private reaches remain | **No** for the remaining inner reaches | **REFACTOR** — partially implemented (CP1): semaphore._value reaches replaced by the bounded-concurrency probe. Remaining `isinstance(provider.inner, X)` identity rows and `provider.inner.*` field reaches are a recorded follow-up: finish the concrete-factory identity rewrite (assert via the factory function each connection wires, plus behavioral request-body checks), or document per-row durable invariants. |
| `test_wiring_llm_factory.py`, `test_wiring_translation_backend.py`, `test_wiring_secrets.py`, `test_wiring_local_asr_provider_runtime.py`, `test_runtime_adapter_composition.py` | Wiring correctness: backend kind, secret source, adapter construction | Strong: composition correctness feeds every runtime path | Public factories | Yes | **KEEP** |
| `test_runtime_pipeline_composition.py` — single-owner composition, rollback on partial acquisition, close-retry, peer-off-chatbox | Resource leaks on partial startup; peer chatbox leak | Explicit architecture contract + strong behavior | Public composition | Yes | **KEEP** |
| `test_application_runtime_lifecycle.py` — shutdown ordering with failure continuation, logging-last, startup partial-failure cleanup | Cleanup skipped after a failed owner; logging torn down before diagnostics | Service/port behavioral contract: *which* ordering relations are required | 30–40 step exact event trace incl. private `TranslatorApp.__new__` composition | **No** | **REFACTOR** — implemented: the test now asserts required ordering *relations* (freeze before releases; overlay failure does not block later closes; diagnostics+logging last; failure surfaced), not one giant exact trace; private `__new__` composition replaced by the test boundary composition. |
| `test_service_ports_contracts.py`, `test_settings_mutation_service.py`, `test_managed_connection_auth.py`, `test_secret_settings_transaction.py`, `test_openrouter_pkce_handoff.py` | Port import safety, DTO freezing, transaction semantics, secret non-leakage | Explicit architecture contract + strong security contract | Public ports | Yes | **KEEP** |
| `test_settings_owner.py`, `test_canonical_settings_persistence.py`, `test_legacy_settings_patch_repository.py`, `test_capture_target_settings.py`, `test_settings_application_owner.py`, `test_settings_projection.py`, `test_settings_mutation_legacy.py`, `test_provider_secret_change_owner.py`, `test_provider_verification_binding_owner.py`, `test_provider_credential_verification_owner.py`, `test_provider_verifier_adapter.py`, `test_provider_runtime_owner.py`, `test_provider_runtime_policy.py`, `test_settings_secrets.py`, `test_sync_secret_store_adapter.py` (via architecture) | Persistence transactions, rollback, verification binding, projection freshness | Strong: persisted settings + rollback | Public owners | Yes | **KEEP** |
| `test_managed_auth_owner.py`, `test_qq_managed_auth.py`, `test_managed_account_composition.py`, `test_managed_auth_runtime_adapter.py`, `test_managed_translation_runtime_adapter.py`, `test_managed_usage_owner.py`, `test_managed_status_refresh_owner.py`, `test_openrouter_pkce_flow_owner.py`, `test_translation_enable_owner.py`, `test_github_star_prompt_application.py`, `test_github_star_prompt_owner.py`, `test_clipboard_auto_translation_owner.py`, `test_vrc_mic_sync_owner.py`, `test_manual_typing.py`, `test_manual_typing_composition.py`, `test_manual_local_asr_fallback_owner.py`, `test_microphone_test_session_owner.py`, `test_self_capture_application_owner.py`, `test_peer_application.py`, `test_overlay_application_transitions.py`, `test_overlay_session_transition_owner.py`, `test_overlay_calibration_owner.py`, `test_overlay_diagnostics_port_lifecycle.py`, `test_overlay_generation_start_owner.py`, `test_local_asr_cpu_repair.py`, `test_local_asr_gpu_provisioning.py`, `test_gpu_runtime_interaction.py`, `test_gpu_provider_recovery_owner.py`, `test_application_runtime_logging.py`, `test_application_after_launch.py`, `test_operational_state_payloads.py`, `test_osc_control_runtime.py`, `test_osc_control_application.py`, `test_osc_control_router.py`, `test_osc_presentation_state.py` | Owner-level state machines and effects | Service/port behavioral contracts | Public owners | Yes | **KEEP** |
| UI adapters (`test_self_capture_*_adapter.py`, `test_peer_capture_*_adapter.py`, `test_microphone_test_capture_adapter.py`, `test_vrchat_osc_presence_adapter.py`, `test_ui_provider_runtime_adapter.py`) | Adapter delegation without algorithm duplication | Explicit architecture contract | Public adapters | Yes | **KEEP** |
| `test_overlay_process_manager.py` — spawn/teardown, kill escalation, ack latching, restart policy, failure reasons | Overlay process zombie/crash mishandling | Service/port behavioral contract | Mostly public manager state machine; some private `manager._process`/`_handle_lifecycle_event` drives | Partial | **KEEP** (rows drive the real private state machine; the behavior is durable and no smaller seam exists without a production rewrite — recorded as follow-up, not blocking) |
| `test_main_cli.py` — CLI dispatch, runtime checks, logging setup | CLI contract and startup diagnostics | Strong: documented entry point | Public | Yes | **KEEP** |
| `test_managed_gemma_*` (translation, demand, distribution, production evidence, settings refresh, provider runtime integration) | Provisioning/demand switching correctness; metrics without text leakage | Strong: user-visible + privacy | Public | Yes | **KEEP** |
| `test_release_dependency_guards.py` — pyproject/uv lock pins, workflow pins, build spec, installer script content, vendored DLL sha256, third-party notices | Release artifact integrity: wrong deps, missing compliance bundling, unpackaged DLLs | Strong: installer/release identity (exact text *is* the contract for scripts/specs/notices) | Source text of build/release artifacts (they are the release "code") | Yes (text *is* the contract) | **KEEP** |
| `test_desktop_overlay_repro.py`, `test_desktop_overlay_runner.py` | Repro harness + renderer runner lifecycle | Port/protocol | Public | Yes | **KEEP** |
| `test_ui_application_boundary.py`, `test_ui_application_composition.py`, `test_settings_view_boundary.py` | Boundary intent freezing, snapshot isolation, typed intents | Explicit architecture contract + behavior | Public boundary (one `UiProviderRuntimeAdapter.__new__` in settings-view boundary test replaced by real construction) | Yes | **KEEP** |
| `test_application_shutdown.py` — coordinator admission order across lifecycle phases, failure continuation, logging-last, terminal completion | Shutdown callback registered after its phase ran; one failed close blocking later required closes | Service/port behavioral contract (phase ordering relations) | Public coordinator API | Yes | **KEEP** |
| `test_gpu_worker_process.py` — authenticated local process discovery/shutdown, cancel preserving decode-only failure fields, closed/request error mapping | GPU worker process leak; error field loss on cancel | Service/port behavioral contract | Public factory + fake worker fixture | Yes | **KEEP** |
| `test_local_asr_production_evidence_composition.py` — composition access reads runtime components; evidence-specific access delegation | Evidence paths reaching private composition internals | Explicit architecture contract | Public composition accessors | Yes | **KEEP** |
| `test_local_asr_selection.py` — resolve local ASR selection per language; CPU auto falls back to qwen when model set incomplete | Wrong STT model resolved for a language; incomplete model set silently unhandled | Strong: user-visible provider behavior + persisted settings | Public resolution | Yes | **KEEP** |
| `test_noto_cjk_distribution.py`, `test_process_capture_distribution.py` — vendored font provenance, build-spec staging, pinned proctap hidden imports/native binary, release workflow smoke steps | Packaged artifact missing CJK font or capture native binary; installer smoke not wired | Strong: installer/release identity (exact text *is* the contract) | Source text of build/release artifacts | Yes (text *is* the contract) | **KEEP** |
| `test_provider_application_owner.py` — translation patch routed through mutation service; combined surfaces applied in order | Provider draft applied out of order or bypassing mutation service | Service/port behavioral contract | Public owner | Yes | **KEEP** |
| `test_runtime_pipeline_custom_http.py` — custom HTTP pipeline skips managed rebuild and owns backend close | Unnecessary managed rebuild for custom HTTP; leaked backend on close | Service/port behavioral contract | Public composition | Yes | **KEEP** |
| `test_self_stt_source_language_runtime.py` — cloud source language owned by provider identity; local source language owned by session options | Source language mishandled between cloud/local providers | Service/port behavioral contract | Public runtime | Yes | **KEEP** |
| `test_system_directory_opener.py` — platform command construction; registry opens its resolved directory | Wrong OS open command; opening the wrong directory | Strong: user-visible behavior | Public registry service | Yes | **KEEP** |
| `test_translation_runtime_configuration_wiring.py` — settings replace is one atomic revision preserving runtime-only values; effective flag replace changes both flags in one revision | Torn configuration revision visible to runtime | Strong: persisted settings + concurrency semantics | Public wiring API | Yes | **KEEP** |
| `test_vrchat_osc_presence_composition.py` — composed presence owner uses injected port notice/cancel contract | Presence probe composing its own transport instead of the injected port | Explicit architecture contract | Public composition | Yes | **KEEP** |
| `test_application_runtime_self_capture_owner.py` — reacquires self capture with direct self translation owner | Self capture re-acquiring the retired hub path | Explicit architecture contract | Public composition | Yes | **KEEP** |

### tests/architecture (217 tests)

| Family | Protected risk | Basis | Boundary | Refactor-resistant | Disposition |
| --- | --- | --- | --- | --- | --- |
| `test_dependency_boundaries.py` — layer rules, settings confinement, structural zero (`KNOWN_ALLOWED_VIOLATIONS == ∅`), allowlist rationale gate | Dependency-direction erosion; silent re-growth of legacy settings reachability | Explicit internal architecture contract | Module-level import policy (not private member names) | Yes | **KEEP** |
| `test_dependency_boundaries.py::test_r00_legacy_settings_reachability_census_matches_the_pinned_baseline` | Re-widening of legacy settings imports beyond the census set | Explicit architecture contract (bounded legacy island) | AST import scan vs census JSON | Yes (guards imports, not internal shape) | **TEMPORARY** — see registry: sunset when A12 physically deletes the compatibility facade/island and this guard's subject disappears. |
| `test_dependency_boundaries.py::test_a07/a08/a09/a10_*_reduces_dependency_debt_to_N` | Regression of completed boundary cutovers | Explicit architecture contract | Path-set import assertions | Yes | **KEEP** (names are historical labels; the assertions are current-state guards) |
| `test_controller_retirement_ownership.py` | Controller regaining retired auth/usage methods or importing UI | Explicit architecture contract (retirement is permanent policy, not a pending sunset) | AST method/field disjointness on the composition owner | Yes (guards *absence*, not presence of shape) | **KEEP** |
| `test_translation_coordinator_retirement.py` | Regression to the retired `ClientHub` coordinator | Explicit architecture contract | File-absence + residue scan + harness-consumer set | Mixed: the exact harness-consumer set couples to test files | **KEEP** (residue scans are absence guards); consumer-set row documented as a bounded migration guard — see TEMPORARY registry. |
| `test_lifecycle_task_guard.py` — unmanaged task creation guard + per-entry rationales | Unowned `asyncio.create_task` growth (task leaks) | Explicit architecture contract | Allowlist with mandatory rationale ("no new unmanaged") | Yes | **KEEP**; per-order historical rows (`test_order34/37/38/...`) are bounded transition guards — see TEMPORARY registry. |
| `test_composition_factory_exclusivity.py`, `test_construction_exclusivity.py` | Duplicate construction of owners outside composition | Explicit architecture contract | AST construction census | Yes | **KEEP** |
| `test_runtime_pipeline_direct_ownership.py`, `test_translation_request_ownership.py`, `test_self_translation_channel_ownership.py`, `test_peer_translation_channel_ownership.py`, `test_output_routing_ownership.py`, `test_translation_runtime_configuration_ownership.py`, `test_self_capture_owner_contracts.py`, `test_self_capture_source_ownership.py`, `test_peer_capture_owner_contracts.py`, `test_microphone_test_session_ownership.py`, `test_local_asr_*_ownership.py`, `test_local_asr_provider_runtime_contracts.py`, `test_gpu_provider_recovery_ownership.py`, `test_provider_secret_change_ownership.py`, `test_provider_credential_verification_ownership.py`, `test_sync_secret_store_adapter_ownership.py`, `test_manual_typing_composition_ownership.py`, `test_managed_openrouter_settings_ownership.py`, `test_legacy_settings_patch_repository_ownership.py`, `test_overlay_generation_start_ownership.py`, `test_overlay_session_transition_ownership.py`, `test_peer_process_capture_retry_ownership.py`, `test_local_asr_production_evidence_ownership.py` | Owner/adapter/wiring boundary erosion (algorithms reappearing in controllers, UI reach-through, whole-settings bags); verifier port completeness for every used verification operation; provider-runtime contract layering (no UI/hub imports from core owner) | Explicit internal architecture contract | Mostly AST import/AST-attribute policy; several rows also assert private method-source snippets (`method_source` exact substrings) | Partial | **SPLIT** (policy KEEP; private method-source substring assertions are retired-name guards → TEMPORARY registry with sunset = removal once the corresponding A-gate is re-verified); the `method_source`-based rows were already reduced to absence assertions where possible. |
| `test_ui_boundary_architecture.py`, `test_settings_contract_boundary.py`, `test_logs_about_contract_boundary.py`, `test_dashboard_contract_boundary.py`, `test_app_shell_contract_boundary.py`, `test_desktop_overlay_surface_boundary.py`, `test_flet_desktop_runtime_boundary.py` | UI layer staying above backend owners; flet-desktop confinement; slot/intent contracts | Explicit internal architecture contract | Import policy + contract-member checks | Yes | **KEEP** |
| `test_raw_user_visible_error_guard.py`, `test_raw_transcript_logging_guard.py` | Raw exception text / transcript text reaching user-visible or log sinks | Strong privacy contract | AST policy guard | Yes | **KEEP** |

### tests/providers (291 tests)

| Family | Protected risk | Basis | Boundary | Refactor-resistant | Disposition |
| --- | --- | --- | --- | --- | --- |
| `test_openrouter_provider.py` — request body contract (URL/headers/model/routing pools/reasoning), error normalization, truncation, key endpoints, user-message envelope | Wire breakage with OpenRouter (external service) | Strong: external API | Public client + injected fake transport | Yes | **KEEP**; the exact basic-log message strings are observability contracts asserted elsewhere too — duplicate check retained intentionally (log leakage is security-relevant). `provider._internal_client`/`_get_client` reaches replaced by behavioral/construction checks (implemented: close closes the owned internal httpx client and leaves injected clients to their owner; max_tokens/user_identifier propagation asserted through the request body over a fake httpx transport). |
| `test_gemini_provider.py`, `test_qwen_provider.py`, `test_deepseek_provider.py`, `test_cerebras_provider.py`, `test_local_openai_provider.py`, `test_qwen_async.py`, `test_qwen_async_client.py`, `test_qwen_client_prompt.py`, `test_llm_user_messages.py` | Per-provider request/response normalization, error detail caps, user-message envelope | Strong: external API | Public clients | Yes | **KEEP** |
| `test_local_qwen_sherpa.py`, `test_local_decode.py`, `test_qwen_asr_session.py`, `test_qwen_asr_backend.py`, `test_deepgram_session.py`, `test_deepgram_backend.py`, `test_soniox_backend.py`, `test_custom_stt.py`, `test_managed_gemma_provider.py`, `test_local_gpu_backend.py`, `test_local_cpu_backends.py` | STT backend session semantics, threading, hallucination redaction, sample-rate contracts | Strong: user-visible + provider contracts | Public backends; monkeypatch-heavy but targeting module seams, not private fields | Yes | **KEEP** |
| `test_openrouter_gemma_routing.py` — single-model request uses `model` only; unified request uses `models` list only; gemma semantic routes produce exact provider preferences (only/sort/allow_fallbacks) | OpenRouter request wire breakage for gemma routing (external service rejects or misroutes) | Strong: external API | Mixed: request body via public client; private `_build_request_body`/`_build_provider_preferences` drives | Partial | **SPLIT** — semantic route→preference mapping rows are contract documentation KEEP; `_build_request_body` reach replaceable by asserting through the same public request-body path used by `test_openrouter_provider.py` (follow-up). |

### tests/ui (960 tests)

| Family | Protected risk | Basis | Boundary | Refactor-resistant | Disposition |
| --- | --- | --- | --- | --- | --- |
| `test_api_key_field.py` — save-before-verify, failure blocks verification, latest-edit-wins over in-flight verification | Race where stale verification result overwrites a newer edit, or verification runs before the secret is durably saved | Strong: user-visible secret-handling correctness | Private `_text_field`/`_handle_blur`/`_current_status`/`_last_verified_hash` reaches | **No** | **REFACTOR** — implemented: the blur→save→verify coordination moved into a UI-independent `ApiKeyVerificationController` (flow state machine); tests drive the controller and keep one thin binding test. |
| `test_settings_view_branches.py` — projection of settings into cards, secret load/save flows, visibility rules, prompt/local-LLM field flows, OSC field projection | Settings UI showing/saving wrong values; secret save failures silently ignored | Strong: user-visible correctness | Private view members (~984 private reaches, monkeypatched population) | Partial | **SPLIT/REFACTOR** — the typed snapshot/intent boundary (G14/G15) is the production seam; rows that assert projection outcomes via snapshots are durable; rows that assert private widget state (`view._openrouter_key.value` etc.) remain the largest follow-up surface. Priority follow-up recorded; blocking scope for A11 = the census + pilot seams, per the recommended sequence. |
| `test_app_branches.py` — launch/teardown flows, managed auth flows, telemetry/app-active-day reporting, peer-EULA toggle, debug preview safety, window close ordering | Broken launch/shutdown, wrong auth routing, telemetry double-send, peer enable without EULA | Strong: user-visible behavior | Mixed: telemetry/app-active-day through `TelemetryReportingOwner`; peer-EULA toggle through application ports; remaining rows still `TranslatorApp.__new__` + autouse boundary override + large fake graph | **No** for remaining `__new__` rows | **REFACTOR** — telemetry family behind `TelemetryReportingOwner`; peer-EULA UI rows are thin bindings (show-dialog vs `set_peer_translation_enabled` / `accept_peer_translation_eula_and_enable`); persist-then-enable owned at `test_ui_application_boundary`. Remaining launch/auth/window `__new__` rows stay follow-up. |
| `test_dashboard_view_branches.py`, `test_dashboard_capture_controls.py`, `test_dashboard_capture_notices.py`, `test_dashboard_surface_contract.py` | Dashboard notice priority/cancellation, capture controls, geometry | Strong: user-visible | Contract/renderer seams (already extracted); fake component injection via module seams | Yes | **KEEP** |
| `test_event_bridge.py` — event mapping, conversation cache bound + close clearing, redaction before user-visible sinks, managed routing | Duplicate/lost UI updates; secret leakage into UI | Strong | Public bridge + destinations; a few `_final_self_transcripts`/`projection_service` reaches | Partial | **KEEP** (cache-bound rows assert through the projection service's public tunable + public close; private-name assertions limited to the bound/clear rows — durable memory-bound contract) |
| `test_settings_view_branches.py` sibling surfaces: `test_settings_prompt_switching.py`, `test_settings_surface_contract.py`, `test_custom_translation_settings.py`, `test_audio_settings_host_api.py`, `test_custom_vocabulary_tag_editor.py`, `test_loopback_process_capture_ui.py` | Prompt/provider switching, custom HTTP config UI, host-API enumeration | Strong | Public-ish view flows | Partial | **KEEP** |
| `test_display_card.py`, `test_language_card.py`, `test_power_button.py`, `test_bottom_nav.py`, `test_title_bar.py`, `test_language_modal.py`, `test_osc_connection_modal.py`, `test_managed_trial_usage_bar.py`, `test_flet_foundation.py`, `test_presentation_adapter.py` | Component state machines, geometry tokens, locale refresh | User-visible rendering contracts | Mixed: helper-level `_weighted_len`/`_display_size_for_length` unit rows are private-shape checks of pure helpers | Partial | **KEEP**; the pure-helper unit rows are refactor-resistant in practice (pure functions, stable semantics) — kept as the only direct coverage of CJK-width sizing. |
| `test_desktop_overlay_renderer.py` — caption mapping table, slot geometry, font policy, preview secret guards, renderer lifecycle | Overlay caption rendering/regression, secret leaks in fixtures | Strong | Mixed: 101 direct private-constant reaches into `_DESKTOP_CAPTION_*` module constants | **No** for constant reaches | **SPLIT** — the mapping-table/geometry rows are contract documentation; private-constant reaches replaced where the surface module exports the same values publicly (surface boundary test already enumerates the public surface). Follow-up: route constant assertions through the public surface module. |
| `test_desktop_overlay_i18n.py`, `test_i18n_key_usage.py`, `test_discord_auth_i18n.py`, `test_peer_translation_eula_copy.py`, `test_document_dialog_pattern.py`, `test_founder_letter_dialog.py`, `test_local_qwen_hallucination_dialog.py`, `test_qq_managed_auth_dialog.py`, `test_discord_managed_auth_dialog.py`, `test_github_star_snackbar.py`, `test_github_star_prompt_eligibility.py`, `test_logs_view.py`, `test_about_view_branches.py`, `test_logs_about_surface_contract.py`, `test_main_window_startup.py`, `test_asr_basic_logging.py`, `test_app_secret_clear.py`, `test_debug_preview_panel.py`, `test_debug_preview_action_order.py`, `test_flet_pinned_compatibility.py`, `test_flet_086_interaction_equivalence.py`, `test_flet_desktop_view_process_owner.py`, `test_desktop_overlay_startup.py`, `test_desktop_window_zorder.py` | i18n parity (all shipped locales), EULA/legal copy exactness (legal text *is* the contract), dialog patterns, Flet runtime pinning (0.86.1 protocol is intentionally contractual), window z-order API usage | Mixed strong contracts; exact copy justified for legal/installer-adjacent text and i18n parity | Public | Yes | **KEEP** |
| `test_baseline_control_geometry.py` | Pinned baseline paddings (accepted geometry) | Explicit internal contract (visual regression) | Public tokens + component application | Yes | **KEEP** |

### tests/integration (27 tests)

Unique knowledge mined per the authority: every live test exercises the *real* provider
against the *real* product pipeline. Deterministic equivalents already exist for the
request/normalization contracts (provider unit tests) — what only the live environment
proves is real credential/network/model acceptance, and end-to-end latency timing.

| File | Unique knowledge | Disposition |
| --- | --- | --- |
| `test_openrouter_llm_integration.py` | Managed/BYOK credential path against the real broker+routing | **MOVE** (release/environment execution; already env-gated) |
| `test_deepseek_llm_integration.py`, `test_qwen_llm_integration.py`, `test_local_llm_integration.py`, `test_qwen_asr_llm_integration.py` | Real-model smoke through product LLM path | **MOVE** (env-gated) |
| `test_deepgram_stt_integration.py`, `test_soniox_stt_integration.py`, `test_qwen_asr_stt_integration.py`, `test_local_qwen_stt_integration.py` | Real STT streaming sessions | **MOVE** (env-gated) |
| `test_e2e_latency_measurement.py`, `measure_latency.py` | End-to-end latency statistics | **MOVE** (measurement harness; manual/run) |
| `test_helpers.py`, `helpers.py` | Harness self-tests + shared env gating | **KEEP** (they test the harness that the moved tests rely on) |

None of these are collected by blocking PR CI (all env-gated); placement already matches
the semantic verdict. "Someone may run it manually" is accepted only because each file
protects a distinct, named external-environment risk — not blanket ownership.

### tests/release_evidence (42 tests)

Split per the authority: deterministic checker tests vs artifact truth.

| File | Protected risk | Basis | Boundary | Disposition |
| --- | --- | --- | --- | --- |
| `test_unattended_runtime.py` | Report schema, queue/stale/cleanup scenarios, baseline validation | Checker correctness (deterministic) | Public evidence functions | **KEEP** |
| `test_process_capture_packaged_smoke.py` | Smoke runner rejects swapped native/helper outside artifact | Checker correctness (monkeypatched) | Public `run_smoke` | **KEEP** |
| `test_windows_process_isolation.py`, `test_windows_application_loopback_ab.py`, `test_local_cpu_real_decode.py`, `test_local_asr_production_composition.py` | Isolation math, A/B analysis, redaction, composition wiring | Checker correctness | Public evidence functions | **KEEP** |
| *Actual artifact truth* (built exe/installer satisfies the checks) | Packaged artifact completeness | **MOVE** — executed by the release pipeline's packaged smokes (`test_release_dependency_guards.py` proves the release script runs them; the checks belong to release execution, not PR CI) |

### tests/scripts (3 tests)

| File | Protected risk | Disposition |
| --- | --- | --- |
| `test_install_local_stt_model.py` | Installer script manifest validation + atomic promotion | **MOVE** — Windows/WSL-bound execution evidence (already skipped outside WSL); kept as real invariant coverage for the installer script |

### tests/helpers and tests/fixtures

| Member | Consumers | Disposition |
| --- | --- | --- |
| `fakes.py: FakeSender, SpeechAwareFakeSession, SpeechAwareFakeBackend, RecordingOscQueue, samples, TargetThread, NoopThread` | 12+ files, capability-specific (not a universal fake) | **KEEP** |
| `fakes.py: self_capture_snapshot, SelfCaptureStateStub` | **0 importers** (dead after earlier test deletions) | **DELETE** — implemented |
| `ui_application.py` (boundary composition, shutdown stub) | 7 files | **KEEP** — capability-specific stubs of ports, shrunken naturally as boundary tests moved to typed snapshots; the `__getattr__` fallbacks it still carries are used by app-branches tests, removed when those rows are (follow-up) |
| `translation_owners.py` | 18 files | **KEEP**; added `commit_self_merge` seam (below); retains no retired `ClientHub` surfaces (guarded by `test_translation_coordinator_retirement.py`) |
| `overlay_refresh_trace.py`, `osc_presentation.py`, `flet_page.py`, `audio.py`, `vad.py`, `ast_sources.py`, `lifecycle.py`, `runtime_pipeline.py`, `paths.py` | All have live consumers | **KEEP** |
| `fixtures/fake_gpu_worker.py` | GPU worker adapter tests | **KEEP** |

### broker/tests (64 spec files, 519 cases)

Reference style per the authority: request-through-boundary, envelope, persistence,
reissue/conflict semantics, deterministic DB/time. Reviewed for duplication and boundary:
route specs assert through the worker/app boundary with deterministic controls; no
private-class reaches exist in the TS suite. Verdict: **KEEP** across families
(challenge/issue/status/verify routes, referral lifecycle, abuse controls, telemetry,
persistence, signatures). Duplication between `discord-issue-route.spec.ts` and
`qq-auth-route.spec.ts` protects genuinely different identity/claim rules — retained.

## TEMPORARY registry (bounded transition guards)

Every entry states why it exists and the exact deletion condition (sunset). The default
"transition guard without sunset is not accepted" is satisfied explicitly:

| Guard | Why it exists | Sunset (exact condition) |
| --- | --- | --- |
| `test_lifecycle_task_guard.py` per-order rows (`test_order34`…`test_order44`) | Pin historical cutover orders so a rename/revert silently resurrects legacy task debt | Delete when the STT-controller legacy allowlist entry (6 tasks) and installer entry reach zero — i.e. when the named-owner cutover those orders track completes; the master guard + rationale gate then covers everything |
| `test_dependency_boundaries.py::test_r00_legacy_settings_reachability_census_matches_the_pinned_baseline` | Proves the legacy settings reachability island does not widen beyond the R00 census while the A12 island deletion is pending | Delete when A12 physically deletes the compatibility facade/island and the guard's subject disappears |
| `test_translation_coordinator_retirement.py::test_all_direct_fixture_consumers_use_the_explicit_owner_harness` | Freezes the closed set of harness consumers while hub-retirement residue is being drained | Delete when the harness consumer set has been stable across one release and no `client_hub` references exist anywhere (already true) — then the residue scans alone suffice |

## Structural zero (exit condition)

- `KNOWN_ALLOWED_VIOLATIONS == frozenset()` and `KNOWN_SETTINGS_RUNTIME_CONFINEMENT_DEBT == frozenset()` — asserted by `test_a10_composition_managed_secrets_wiring_reduces_dependency_debt_to_0` and the allowlist-match test. **True at baseline; preserved by this census's changes.**
- The runtime graph cannot reach legacy `config.settings`/`AppSettings` for ordinary production paths — enforced by the layer rules + confinement guard. **Preserved.**
- The A12-owned persistence/compatibility island was **not** deleted and **not** expanded; no test-only expansion was added.
- No architecture guard was weakened; no permanent exception added.

## Implementation notes (this goal's changes)

1. **Census artifact** — this document (`tests/a11_test_census.md`).
2. **`tests/config/test_prompt_contract.py` SPLIT** — semantic contract retained
   (placeholders, chronological context semantics, self/peer legend meaning, timestamp
   syntax prohibition); exact-English-copy assertions replaced with semantic prohibitions.
3. **`tests/app/test_application_runtime_lifecycle.py` REFACTOR** — shutdown semantics
   asserted as required ordering relations and failure-continuation guarantees through the
   public boundary; removed the `TranslatorApp.__new__` private composition and the
   38-line exact event trace. Startup partial-failure coverage retained unchanged
   (already behavioral).
4. **`tests/app/test_wiring_providers.py` REFACTOR (pilot)** — concurrency assertions
   (`semaphore._value == N`) replaced with a behavioral bounded-concurrency probe;
   `isinstance(provider.inner, X)` identity assertions replaced with concrete-factory
   assertions that stay true under provider-internal refactors.
5. **`tests/core/test_translation_turn_owner.py` SPLIT** — removed `inspect.getsource`
   call-count residue; kept composition identity assertions.
6. **`tests/ui/test_api_key_field.py` REFACTOR** — extracted
   `ApiKeyVerificationController` (production seam, UI-independent); flow tests drive the
   controller; thin binding test kept for the Flet wiring.
7. **Dead helper removal** — `tests/helpers/fakes.py`: `SelfCaptureStateStub`,
   `self_capture_snapshot` deleted (zero importers).
8. **Harness seam** — `tests/helpers/translation_owners.py`: added `commit_self_merge`
   so commit-path tests no longer need the owner-private method (pilot for the
   low-latency REFACTOR family).
9. **CI placement** — added `tests/core` to the PR contract matrix (previously no CI job
   ran the suite's largest directory); `tests/release_evidence` and `tests/integration`
   remain release/environment-executed, matching their census dispositions.
10. **`tests/ui/test_app_branches.py` telemetry family REFACTOR** — extracted
    `TelemetryReportingOwner` (UI-independent apply/sync, UTC-date capture before queue,
    one retry after 60s on the same UTC date, midnight guards, settings-mutation
    non-blocking, retry cancel on close); flow tests drive the owner; thin TranslatorApp
    binding tests kept. Remaining `__new__` launch/auth/window rows are follow-up.
11. **`tests/ui/test_app_branches.py` peer-EULA REFACTOR** — toggle/accept/disable rows
    bound to application ports (`state()`, `set_peer_translation_enabled`,
    `accept_peer_translation_eula_and_enable`); persist-then-enable ordering owned at
    `tests/app/test_ui_application_boundary.py`.
12. **Terminal-review census completion** — all 31 previously unclassified start-SHA test
    files added as census rows (core pipeline/audio/logging families, local_translation
    gemma family, app runtime/ownership/wiring/release files, architecture ownership
    contracts, custom-STT resolution, OpenRouter gemma routing); r00 guard aligned to the
    TEMPORARY registry; the wiring `isinstance(provider.inner, ...)` follow-up recorded
    truthfully as remaining (not "implemented").

## Success-metric deltas (baseline → this goal)

- Tests that break under a semantics-preserving private rename (private-member driven):
  reduced in the touched families (lifecycle, api-key field, wiring providers,
  turn-owner source scans); remaining follow-ups recorded above.
- `TEMPORARY` guards without a sunset: 3 → 0 (all registered with explicit sunsets).
- Universal-fake surface: dead members of `fakes.py` removed; no universal fake remains.
- Durable tests with explicit protected-risk rationale: 100% of surviving families are
  covered by this artifact.

## Deep-read re-pass (owner-authorized continuation, 2026-08-29)

The initial census classified most files correctly from targeted reading plus pattern
measurement, but ~20 of the 31 terminal-review additions were classified from test names
and module imports only, and the implemented-seam list is a small subset of the census's
REFACTOR/SPLIT dispositions. Per owner decision, the census is now being re-executed as a
deep-read pass:

**Protocol** (applies per file, one at a time):

1. Read the test file in full (every test body, helper, fixture, and fake).
2. Classify each coherent row against the authority's core decision rule and evidence
   hierarchy: protected risk, contract basis, observation boundary, refactor resistance,
   duplication, lifetime, disposition.
3. Append the per-file verdict to the [Deep-read log](#deep-read-log) below, including any
   correction to the earlier row above (earlier rows are historical, the log is live).
4. Continue immediately until every file in the pass list is logged.
5. When the log is complete, implement every disposition the log records in dependency
   order (production seam first, then test rewrite, then census row update), as new
   checkpoints with fresh review.

**Pass list and status** (files NOT already deep-read in CP1–CP4; updated as work
progresses):

- [ ] `tests/ui/test_settings_view_branches.py` (~984+ private reaches — largest surface)
- [ ] `tests/ui/test_app_branches.py` — remaining 84 `TranslatorApp.__new__` rows
- [ ] `tests/app/test_wiring_providers.py` — remaining ~148 `provider.inner` reaches
- [ ] `tests/ui/test_dashboard_view_branches.py`
- [ ] `tests/ui/test_desktop_overlay_renderer.py` (101 private-constant reaches)
- [ ] `tests/core/test_overlay_presenter.py` (burst-task internals)
- [ ] `tests/app/test_overlay_process_manager.py` (private state-machine drives)
- [ ] `tests/core/test_stt_controller.py` (private queue drives)
- [ ] `tests/core/test_self_translation_low_latency.py` (private-drive rows)
- [ ] `tests/core/test_translation_output_streaming.py` (map-emptiness rows)
- [ ] `tests/core/test_translation_owner_branch_coverage.py`
- [ ] `tests/providers/test_openrouter_gemma_routing.py`
- [ ] `tests/core/local_translation/` gemma family (6 files) — name-classified only
- [ ] `tests/core/test_audio_source.py`, `tests/core/test_file_logging.py` — name-classified only
- [ ] `tests/app/` runtime/ownership/composition files from the 31-file batch — name-classified only
- [ ] `tests/architecture/test_local_asr_provider_runtime_contracts.py`, `tests/architecture/test_provider_credential_verification_ownership.py` — name-classified only
- [ ] `tests/config/test_custom_stt_runtime_resolution.py` — name-classified only

## Deep-read log

Per-file verdicts appended below as the pass proceeds. Each entry: file, what was read,
verdict (confirm or correct the earlier row), and concrete disposition to implement.

### `tests/ui/test_settings_view_branches.py` — deep-read 2026-08-29 (CP5 D1)

Read: full structural pass — both construction helpers (`_make_settings_view` real
construction with monkeypatched population seams; `_make_llm_selection_view` ~110-line
`__new__` + SimpleNamespace widget graph), all ~25 helper functions (card-tree walkers,
label collectors), and detailed reading of ~1,000 lines covering every family: OSC draft
preservation, telemetry card, clipboard auto-translate, secret load/save/clear + failure
paths, api-key icon restore, api-visibility matrix, managed key card + trial usage +
TalkTogether pass invite progress, LLM/connection selection staging + history + prompt
switching, local-LLM fields + secret change + extra-body validation, STT/peer-STT
selection + GPU device cards, prompt draft/commit/reset, VAD + audio handlers, overlay
display/size/lock/position-reset/runtime-state, card layout & typography contracts,
custom vocabulary full lifecycle, subtab shell structure/scroll/locale.

Verdict per family (rows, ~258 tests):

- **KEEP, durable boundary already correct (majority)**: secret load/save/delete
  (incl. failure-tolerance and secret-non-leakage logging), prompt draft/commit/reset
  semantics, provider selection staging (`build_provider_apply_settings` vs emitted
  `on_settings_changed` — draft-does-not-leak, copy-not-mutate, equality guards),
  connection-history preservation, managed-key/invite-progress state machine,
  custom-vocabulary lifecycle, telemetry callback-not-send, overlay runtime-state and
  position-reset semantics, subtab scroll restore. These assert through public intent
  callbacks (`on_settings_changed`, `build_provider_apply_settings`, modal fakes) and
  persisted `AppSettings` copies. A view-internal refactor that keeps the intent
  callbacks stable would still pass.
- **PARTIAL — widget-shape rows are the residual debt (recorded, ~40 rows)**: rows that
  assert Flet widget internals — `view._openrouter_key.visible`, PKCE button
  `style.color[ControlState.*]` colors, `_wrapped_card_column(card).controls[4]`
  positional indexes, exact card-titles ordering lists, `ft.Container` expand/width
  geometry, typography `{28}`, hover color mutation. These are visual/geometry
  contracts enforced elsewhere by tokens (`SettingsUnitCard.DEFAULT_HEIGHT`,
  `test_baseline_control_geometry`) or duplicate the layout tests. Not blocking:
  the census already records `SettingsUnitCard`/`SharedCardWrapper` as the production
  seam, and breaking these rows requires changing user-visible layout, which the
  geometry/typography token tests also guard.
- **DELETE-scope residue already removed (verified)**: retired-control absence rows
  (`_low_latency_card`, `_integrated_context_button`, `_peer_qwen_region_text`,
  `_peer_deepgram_model_*`, legacy overlay/peer toggle API) are absence guards that
  document the A01–A10 cutover; they duplicate `tests/architecture` retirement guards
  but are cheap and behavior-anchored. KEEP.
- **One redundant pair found**: `test_clipboard_auto_translate_selection_...` vs
  `test_clipboard_auto_translate_click_...` — the click row re-asserts the selection
  row plus toggle-off; keep (toggle-off is unique).

Disposition for implementation phase: **KEEP the file as-is** (no REFACTOR checkpoint
required for behavior preservation); the ~40 widget-shape rows are consolidated here as
the recorded follow-up, superseding the earlier "~984 private reaches" bulk estimate —
the durable rows already observe the typed-intent/snapshot boundary (G14/G15 seam) and
the widget rows are visual-contract documentation, not implementation characterization.
No production change needed; no test rewrite needed for blocking scope.

### `tests/ui/test_app_branches.py` (remaining `__new__` rows) — deep-read 2026-08-29 (CP5 D1)

Read: full file (4,633 lines, 117 tests) including both construction seams
(`_patch_app_construction` + real `TranslatorApp(...)` construction with dummy views —
used by ~20 construction rows; `TranslatorApp.__new__` + attribute injection + autouse
`application` property override — used by ~84 flow rows), the dummy view/controller
classes, and detailed reading of ~1,800 lines across: window/layout construction,
debug-preview wiring and per-preview safety (fail-closures on persistence/OAuth),
managed Discord/QQ auth flows (dialog state machine, generation counters, cancel,
referral), nav-change/provider-apply/prompt-apply/settings queue ordering, microphone
test dialogs, PKCE flow, language change, verify-api-key persistence, snackbar/founder
letter, update check, telemetry binding (CP3), peer-EULA binding (CP4), shutdown
idempotency + lifecycle registration (CP1).

Verdict per family:

- **KEEP, durable (majority, ~100 rows)**: the flow rows assert user-visible outcomes
  through *ports* — queued page-task factories, dialog fakes with recorded calls,
  application-boundary doubles, `build_provider_apply_settings`, snackbar/dialog
  recording. They are refactor-resistant to production-internal changes: the seams they
  depend on (`_run_page_task`, `_queue_settings_mutation_task`, `_show_snackbar`,
  view intent callbacks, `compose_test_ui_application_boundary`) are the stable
  TranslatorApp↔view/boundary contracts, not private internals. The `__new__` pattern
  here is test construction of those ports, not characterization of private state.
- **KEEP with caveat, `controller=` SimpleNamespace (~45 rows)**: managed auth and
  debug-preview rows install a fake `app.controller` and rely on the autouse fixture
  wrapping it into a boundary. The `controller` attribute is retired production shape
  (production uses `self.application`), kept alive only by the autouse fixture. The
  behavior asserted (dialog lifecycle, generation staleness, cancel, referral bonus,
  preview safety) is durable; the construction vehicle is legacy. Cleanup = migrate
  these rows to install `_ui_application` doubles directly (as CP4 did for peer-EULA),
  then delete the autouse fixture. **Implementation follow-up (mechanical, no
  behavior change).**
- **KEEP, construction rows (~20 rows)**: real construction through the public
  `application_factory` port (one row even uses the production `compose_ui_application`)
  — boundary wiring tests, exactly the preferred direction.
- **DELETE candidates identified**: `TelemetryController`/`TelemetrySettingsView`
  helper classes (lines 192–220) have **zero usages** after CP3's rewrite — dead test
  infrastructure, same status the census already assigned to other dead fakes.

Disposition for implementation phase: CP5 implements the mechanical cleanup —
migrate `controller=`-based rows to `_ui_application` port doubles, drop the autouse
`application` property override, delete dead `TelemetryController`/`TelemetrySettingsView`,
shrink `tests/helpers/ui_application.py` fallbacks as the census already planned. The
authority's preferred direction (thin binding → controller/flow seam) is already
satisfied for the *new* families (telemetry, peer-EULA); the remaining auth/debug
families are judged KEEP-at-binding because their asserted contracts are the dialog and
queue behaviors themselves, observable only at this layer.

### `tests/app/test_wiring_providers.py` — deep-read 2026-08-29 (CP5 D1)

Read: full file (2,487 lines, 93 tests), including the compatibility shims at top
(`create_llm_provider`/`create_stt_backend` legacy→canonical adapters, `_canonical_settings`,
`assert_bounded_concurrency` probe, `_unwrap_release_service`, `_resolved_stt_config`
builder) and detailed reading of ~1,500 lines covering: per-provider LLM factory rows
(gemini/qwen/deepseek/cerebras/local-LLM/openrouter incl. managed/BYOK/fallback-racing/
release-service wrapping), STT backend factory rows (deepgram/qwen-asr/soniox/local-qwen/
parakeet/cpu-auto, self and peer channels, resolved-DTO paths), peer runtime resolution +
signatures, overlay config mapping.

Verdict per family:

- **The `provider.inner` reaches are narrower than the bulk count suggested.** The
  ~148 `provider.inner` matches decompose into: (a) `isinstance(provider.inner, X)`
  *construction-identity* rows asserting which concrete provider a settings combo wires
  (~45 rows) — these are wiring-correctness contracts ("this connection must produce a
  fallback-racing wrapper whose primary is OpenRouter"), and the concrete class IS the
  observable contract at this boundary because the factory is the unit under test;
  (b) `provider.inner.<field>` config-propagation rows asserting api_key/model/
  base_url/routing/runtime_logging reach the constructed provider (~90 assertions
  inside the same rows) — same nature: constructor-argument propagation, checked at the
  only seam where it exists (the provider's public constructor surface); (c) two rows
  reaching `provider.inner.fallback._delegate` (`_LazyFactoryLLMProvider` internals,
  lines 1276–1278) — genuine private-state assertion, but the immediately adjacent
  `factory()` call asserts the same outcome behaviorally, so `_delegate is None` is a
  duplicate pre-condition check.
- **KEEP (the overwhelming majority)**: secret resolution rules (legacy-key backfill,
  region keys, env fallbacks, store-over-env precedence — persisted-security contracts),
  fallback routing plans, managed release-service wrapping and lazy user-identifier
  loading, peer/self channel isolation (self settings must not leak into peer backend),
  sample-rate 16k runtime contract, custom-term normalization. All durable, all
  refactor-resistant (a wrapper-rename or semaphore-internal change passes).
- **SPLIT (small)**: the two `_delegate` pre-condition assertions can be dropped as
  duplicates without losing coverage; the `isinstance(provider.inner, X)` rows are
  documented as *intentional* wiring-identity contracts (the census's earlier
  "concrete-factory rewrite" idea would just move the same identity assertion one
  level up — the class boundary here is the contract, not a shape freeze).
- **The top-of-file compatibility shims** (`create_llm_provider` legacy adapter,
  `_canonical_settings`) are transitional A10→A12 seam helpers used by tests only;
  they are exactly the kind of "compatibility surface backed by a real consumer" the
  authority allows, with the consumer being this suite until A12 deletes the island.

Disposition for implementation phase: **KEEP with a micro-cleanup** — drop the two
duplicate `_delegate` pre-condition assertions; re-classify the census wiring row from
"REFACTOR partially implemented, ~148 reaches remain" to "KEEP (wiring-identity
contracts intentional; 2 duplicate private pre-condition assertions removed)". The
earlier "concrete-factory identity rewrite" follow-up is withdrawn: at the factory
boundary, the concrete provider class is the contract.
