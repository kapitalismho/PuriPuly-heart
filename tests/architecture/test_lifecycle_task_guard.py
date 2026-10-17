from __future__ import annotations

import ast
from collections import Counter

from tests.helpers.paths import REPO_ROOT
from tests.helpers.task_creation import (
    ASYNCIO_CREATE_TASK,
    ASYNCIO_ENSURE_FUTURE,
    ASYNCIO_TASK,
    BARE_RUN_TASK,
    LOOP_CREATE_TASK,
    RUN_TASK,
    TaskClassification,
    TaskDisposition,
    TaskIdentity,
    assert_task_ownership,
    scan_task_creation_sites,
)

PROTECTED_TASK_SOURCE_ROOTS = ("src/puripuly_heart",)


def _reviews(
    module: str,
    functions: dict[str, int],
    rationale: str,
    *,
    classification: TaskClassification = "named_owner",
    family: str = ASYNCIO_CREATE_TASK,
) -> tuple[tuple[TaskIdentity, TaskDisposition], ...]:
    return tuple(
        (
            TaskIdentity(f"src/puripuly_heart/{module}", function, family),
            TaskDisposition(classification, maximum, rationale),
        )
        for function, maximum in functions.items()
    )


TASK_CREATION_REVIEWS = (
    *_reviews(
        "core/lifecycle.py",
        {"LifecycleScope.create_task": 1},
        "LifecycleScope is the owning primitive: named task registration, closed-scope rejection, cancellation and draining on close",
        classification="owner_primitive",
    ),
    *_reviews(
        "core/lifecycle.py",
        {"LifecycleScope.create_task": 1},
        "LifecycleScope registers and drains the eager-start Task through the same owning primitive",
        classification="owner_primitive",
        family=ASYNCIO_TASK,
    ),
    *_reviews(
        "app/adapters/gpu_worker_process.py",
        {"_DefaultGpuWorkerClient.close": 1, "_DefaultGpuWorkerClient._close_writer_resource": 1},
        "GPU worker client retains close and writer-close tasks; shielded close or force-close owns terminal resource cleanup",
    ),
    *_reviews(
        "app/services/application_control.py",
        {"ApplicationControlOwner.submit": 1},
        "ApplicationControlOwner retains each operation independently of its client; freeze_ingress interrupts operations and drain_operations gathers them",
    ),
    *_reviews(
        "app/services/application_shutdown.py",
        {
            "ApplicationShutdownCoordinator.shutdown": 1,
            "ApplicationShutdownCoordinator._run_callback": 1,
            "ApplicationShutdownCoordinator._emit_diagnostic": 1,
        },
        "ApplicationShutdownCoordinator owns bounded callback and diagnostic tasks, cancels them on deadlines, and awaits terminal cleanup",
    ),
    *_reviews(
        "app/services/canonical_settings_persistence.py",
        {"SettingsOwner.persist_async": 1},
        "SettingsOwner locally shield-awaits the thread persistence task, including after cancellation, before releasing the call boundary",
    ),
    *_reviews(
        "app/services/index_selection.py",
        {"IndexSelectionOwner._schedule": 1, "IndexSelectionOwner._run_owned": 1},
        "IndexSelectionOwner shares one bounded selection result, fences and cancels timed-out work while retaining its owned blocking cleanup, and drains supervisor plus worker tasks on close",
    ),
    *_reviews(
        "app/services/llm_connection_readiness.py",
        {"LlmConnectionReadinessOwner._start_mode": 1},
        "LlmConnectionReadinessOwner retains mode transition tasks and gathers the remaining set on close",
        family=LOOP_CREATE_TASK,
    ),
    *_reviews(
        "app/services/manual_typing.py",
        {"ManualTypingOwner._reschedule_idle_timeout": 1},
        "ManualTypingOwner owns its bounded idle timeout task and cancels it on input transitions and application shutdown",
        family=LOOP_CREATE_TASK,
    ),
    *_reviews(
        "app/services/overlay/overlay_application.py",
        {
            "OverlayApplicationOwner._retire_fallback_runtime": 1,
            "OverlayApplicationOwner.watch_runtime": 1,
            "OverlayApplicationOwner._run_desktop_startup_recovery": 1,
        },
        "OverlayApplicationOwner retains retirement and recovery tasks and cancels or gathers them across shutdown, close and replacement barriers",
    ),
    *_reviews(
        "app/services/overlay/overlay_calibration_application.py",
        {"OverlayCalibrationApplicationOwner._schedule_owned": 1},
        "OverlayCalibrationApplicationOwner retains calibration tasks, observes completion failures and exposes wait_pending for the application shutdown barrier",
    ),
    *_reviews(
        "app/services/peer_application.py",
        {"PeerApplicationOwner._await_runtime_replacement_cleanup": 1},
        "PeerApplicationOwner retains shielded replacement cleanup and observes its result; the previous runtime remains owned until cleanup commits",
    ),
    *_reviews(
        "cli/host.py",
        {"GuiControlBoundary.start": 1},
        "Bounded grandfathering: GUI stop cancels its retained stop-request watcher without joining it; this is existing incomplete lifetime enforcement",
        classification="legacy_exception",
    ),
    *_reviews(
        "cli/host.py",
        {"GuiControlBoundary._close_host": 1},
        "GuiControlBoundary retains the host close task and shield-awaits it from stop and the stop-request watcher",
    ),
    *_reviews(
        "cli/transport.py",
        {"ControlServer._serve": 2},
        "ControlServer owns subscriber disconnect and next-event tasks within aclosing; its finally block cancels and gathers both",
    ),
    *_reviews(
        "composition/headless_application.py",
        {"HeadlessApplicationPresentation.schedule_task": 1},
        "Headless presentation retains scheduled tasks, observes their outcomes and cancels plus gathers them in close_after_launch_tasks",
    ),
    *_reviews(
        "core/audio/listen_delivery.py",
        {"ListenDeliveryController._arm_timers": 3},
        "ListenDeliveryController retains the pause-step and hard-deadline timers; close cancels and gathers the current timer set",
    ),
    *_reviews(
        "core/audio/smart_turn.py",
        {
            "SmartTurnInferenceOwner.request_prepare": 1,
            "SmartTurnInferenceOwner.submit": 1,
            "SmartTurnInferenceOwner.close": 1,
            "SmartTurnInferenceOwner._prepare": 1,
        },
        "SmartTurnInferenceOwner retains prepare, execution and close tasks; shielded resource close waits for preparation and inference before releasing the model",
    ),
    *_reviews(
        "core/audio/smart_turn.py",
        {"_await_owned_operation": 1},
        "SmartTurn's local operation boundary shield-awaits work and awaits completion after cancellation before propagating cancellation",
        family=ASYNCIO_ENSURE_FUTURE,
    ),
    *_reviews(
        "core/llm/fallback_racing.py",
        {
            "FallbackRacingLLMProvider._translate": 1,
            "FallbackRacingLLMProvider.close": 1,
            "FallbackRacingLLMProvider._create_tracked_task": 1,
        },
        "fallback racing owns short-lived contender tasks and cancels losers within the provider call boundary; tracked cleanup and shielded close retire owned work",
        classification="legacy_exception",
    ),
    *_reviews(
        "core/local_asr/local_stt_runtime_installer.py",
        {"_ensure_local_stt_installed_with_lease": 1},
        "legacy installer download task remains deferred to the local STT download runtime owner cutover",
        classification="legacy_exception",
    ),
    *_reviews(
        "core/local_translation/provisioning.py",
        {"_await_download_cleanup": 1},
        "Gemma download boundary signals cancellation or cancels the download and waits until it settles before releasing staging ownership",
    ),
    *_reviews(
        "core/local_translation/runtime.py",
        {"ManagedGemmaRuntimeOwner._wait_for_startup": 2},
        "ManagedGemmaRuntimeOwner races readiness against process exit and cancels plus gathers both local waiters in finally",
    ),
    *_reviews(
        "core/local_translation/runtime.py",
        {"ManagedGemmaRuntimeOwner._await_server_work": 1},
        "ManagedGemmaRuntimeOwner retains server work in _server_tasks: admission permits at most "
        "two Vulkan workers or one CPU worker, including prefix preparation serialized under the "
        "same capacity gate; fallback drains active slots before replacement preparation. Each "
        "call shield-drains and consumes its worker outcome before releasing the slot; lifecycle "
        "shutdown applies bounded grace, stops the owned server, cancels remaining workers and "
        "drains their operation owners",
        family=ASYNCIO_ENSURE_FUTURE,
    ),
    *_reviews(
        "core/orchestrator/self_translation_channel.py",
        {
            "SelfTranslationChannelOwner._start_awaiting_vad_timeout": 1,
            "SelfTranslationChannelOwner._start_resume_end_timeout": 1,
            "SelfTranslationChannelOwner._restart_post_end_grace": 1,
            "SelfTranslationChannelOwner._maybe_restart_spec": 1,
        },
        "Self translation owner owns per-utterance timeout and speculation tasks and cancels them through its explicit channel lifecycle",
    ),
    *_reviews(
        "core/orchestrator/translation_output_projection.py",
        {
            "TranslationUiMessageQueue._ensure_self_writer": 1,
            "TranslationUiMessageQueue._ensure_peer_writer": 1,
        },
        "TranslationUiMessageQueue retains destination writer tasks; close retires publication and cancels plus gathers both writers",
    ),
    *_reviews(
        "core/osc/receiver.py",
        {"VrcOscReceiver.mute_handler": 1},
        "OSC receiver mute-state callback schedules adapter-local async work on the owning loop until receiver runtime ownership fully wraps it",
        classification="legacy_exception",
        family=LOOP_CREATE_TASK,
    ),
    *_reviews(
        "core/overlay/bridge.py",
        {"OverlayBridge._create_task": 1},
        "overlay bridge adapter wraps a named task factory and closes tasks through its adapter lifecycle",
        classification="legacy_exception",
    ),
    *_reviews(
        "core/overlay/presenter.py",
        {"OverlayPresenter._create_task": 1},
        "overlay presenter adapter wraps a named task factory and cancels presenter work during close",
        classification="legacy_exception",
    ),
    *_reviews(
        "core/overlay/presenter.py",
        {"OverlayPresenter.emit_peer_when_admissible": 2},
        "OverlayPresenter locally owns replacement wake and deadline waiters and cancels plus gathers both in finally",
    ),
    *_reviews(
        "core/overlay/process.py",
        {
            "OverlayProcessManager.start": 3,
            "OverlayProcessManager._create_task": 1,
            "OverlayProcessManager._create_cleanup_task": 1,
            "OverlayProcessManager._next_shutdown_lifecycle_event": 2,
            "OverlayProcessManager._fail": 1,
        },
        "overlay process-manager owner retains spawn/reaper, monitor and diagnostic tasks; local shutdown waiters are cancelled and gathered before releasing process state",
    ),
    *_reviews(
        "core/overlay/process_adapter.py",
        {"_AsyncioOverlayProcess._create_task": 1},
        "Overlay process adapter retains stream reader tasks and finish_readers gathers them through the process-manager close boundary",
    ),
    *_reviews(
        "core/overlay/process_adapter.py",
        {"_AsyncioOverlayProcess.terminate": 1},
        "Bounded grandfathering: shielded exit-confirmation waiter is awaited normally but is not retained, cancelled or joined when termination times out",
        classification="legacy_exception",
    ),
    *_reviews(
        "core/runtime/audio_vad_loop.py",
        {"_frames_with_progress": 1},
        "Audio frame iterator locally owns its progress monitor; finally signals finished and awaits the monitor",
    ),
    *_reviews(
        "core/runtime/clipboard.py",
        {"ClipboardRuntime.submit_from_loop": 1},
        "ClipboardRuntime is the named lifecycle owner for clipboard watcher and submission tasks",
    ),
    *_reviews(
        "core/runtime/desktop_overlay_bounds.py",
        {"DesktopOverlayBoundsOwner.schedule_persistence": 1},
        "DesktopOverlayBoundsOwner owns the debounced bounds persistence task and cancels or gathers it before resets, setting changes, and shutdown",
    ),
    *_reviews(
        "core/runtime/github_star_prompt.py",
        {
            "GithubStarPromptRuntime.start_launch_prompt": 1,
            "GithubStarPromptRuntime.start_translation_success_observation": 1,
        },
        "GithubStarPromptRuntime is the named lifecycle owner for prompt observation/launch tasks",
    ),
    *_reviews(
        "core/runtime/gpu_asr.py",
        {"SharedGpuASRRuntime._begin_shutdown": 1},
        "SharedGpuASRRuntime retains its shutdown task and shield-awaits it on channel deactivation or close; shutdown gathers worker tasks",
    ),
    *_reviews(
        "core/runtime/local_asr_provisioning.py",
        {"LocalASRProvisioningOwner._schedule_install_result_delivery": 1},
        "LocalASRProvisioningOwner owns install-result delivery tasks and cancels and awaits them during close",
    ),
    *_reviews(
        "core/runtime/local_stt_download.py",
        {"LocalSTTDownloadRuntime.start": 1},
        "LocalSTTDownloadRuntime is the named lifecycle owner for download tasks",
    ),
    *_reviews(
        "core/runtime/mic_test.py",
        {"MicTestRuntime.start": 1, "MicTestRuntime.create_frame_task": 1},
        "MicTestRuntime is the named lifecycle owner for microphone test tasks",
    ),
    *_reviews(
        "core/runtime/oauth.py",
        {"OAuthRuntime.create_auth_task": 1},
        "OAuthRuntime is the named lifecycle owner for managed-auth tasks",
    ),
    *_reviews(
        "core/runtime/output.py",
        {"OutputRuntime._submit_peer_overlay_event": 1, "OutputRuntime._create_task": 1},
        "OutputRuntime is the named lifecycle owner for chatbox flush, overlay delivery, and UI bridge tasks",
    ),
    *_reviews(
        "core/runtime/overlay.py",
        {"OverlayRuntimeHandle._create_task": 1},
        "OverlayRuntimeHandle is the named lifecycle owner for overlay tasks",
    ),
    *_reviews(
        "core/runtime/overlay_session_fallback.py",
        {"_create_overlay_session_fallback_task": 1},
        "OverlaySessionFallbackOwner owns its named deferred fallback task and cancels and awaits it during close",
    ),
    *_reviews(
        "core/runtime/peer_channel.py",
        {
            "_GenerationGuardedVadSink._submit": 1,
            "_GenerationGuardedVadSink._arm_expiry_timer": 1,
            "PeerCaptureSessionOwner.close": 1,
            "PeerCaptureSessionOwner._create_task": 1,
        },
        "PeerCaptureSessionOwner owns its session loop, generation-scoped VAD dispatch/expiry and shielded close; generation retirement drains the owned work",
    ),
    *_reviews(
        "core/runtime/provider_handle.py",
        {"ProviderRuntimeHandle._create_task": 1},
        "ProviderRuntimeHandle is the named lifecycle owner for provider event draining, idle release and retirement",
    ),
    *_reviews(
        "core/runtime/receiver.py",
        {"VrcMicReceiverRuntime.handle_mute_packet": 1},
        "VrcMicReceiverRuntime is the named lifecycle owner for receiver tasks",
    ),
    *_reviews(
        "core/runtime/self_capture.py",
        {
            "_GenerationGuardedVadSink.handle_stream_input": 1,
            "_GenerationGuardedVadSink.handle_vad_event": 1,
            "_GenerationGuardedVadSink._admit_unsent_segment": 1,
            "SelfCaptureSessionOwner.note_recognition_failure": 1,
            "SelfCaptureSessionOwner.apply_intent": 1,
            "SelfCaptureSessionOwner._start_generation": 1,
            "SelfCaptureSessionOwner._on_loop_task_done": 1,
        },
        "SelfCaptureSessionOwner owns intent transitions, its session loop, contained fault teardown and generation-scoped VAD dispatch/expiry",
    ),
    *_reviews(
        "core/runtime/vrchat_osc_presence.py",
        {"_create_vrchat_osc_probe_task": 1},
        "VrchatOscPresenceProbeOwner owns its named probe task and cancels and awaits it during close",
    ),
    *_reviews(
        "core/stt/scoped_engine.py",
        {
            "ScopedRecognitionEngine.bind_event_sink": 1,
            "ScopedRecognitionEngine._run_stream_write": 1,
            "ScopedRecognitionEngine.abort": 1,
            "ScopedRecognitionEngine.close_backend": 1,
            "ScopedRecognitionEngine._handle_end": 1,
            "ScopedRecognitionEngine._ensure_session": 2,
            "ScopedRecognitionEngine._prepare_qwen_turn_audio": 1,
            "ScopedRecognitionEngine._run_write": 1,
            "ScopedRecognitionEngine._retire_current_session": 1,
            "ScopedRecognitionEngine._schedule_lifetime_check": 1,
            "ScopedRecognitionEngine._bounded_cleanup_call": 1,
            "ScopedRecognitionEngine._schedule_late_factory_reclaim": 1,
        },
        "ScopedRecognitionEngine retains event, operation, session, factory and cleanup tasks; retirement transfers cleanup debt and close_backend awaits cleanup before backend release",
    ),
    *_reviews(
        "core/stt/scoped_engine.py",
        {"ScopedRecognitionEngine._notify_terminal_failure": 1},
        "Bounded grandfathering: terminal notification is retained and its callback consumes exceptions, but close does not drain notification tasks",
        classification="legacy_exception",
    ),
    *_reviews(
        "core/stt/scoped_engine.py",
        {"ScopedRecognitionEngine._normalization_diagnostic": 1},
        "Bounded grandfathering: normalization notification is not retained; its callback consumes exceptions, but close does not drain it",
        classification="legacy_exception",
    ),
    *_reviews(
        "providers/llm/chatgpt_plan.py",
        {
            "ChatGptPlanLLMProvider._maintain": 1,
            "ChatGptPlanLLMProvider._start_opening": 1,
            "ChatGptPlanLLMProvider.close": 1,
            "ChatGptPlanLLMProvider._run_reserved": 1,
            "ChatGptPlanLLMProvider._current_token_generation": 1,
        },
        "ChatGPT plan provider retains maintenance, opening, request and auth tasks; shielded close drains those sets before closing the connection pool",
    ),
    *_reviews(
        "providers/llm/index_translate.py",
        {"IndexTranslateProvider.close": 1},
        "Index provider retains one client-close task, shields and drains it despite repeated caller cancellation, and releases its client reference only after terminal cleanup",
    ),
    *_reviews(
        "providers/stt/custom.py",
        {
            "_OfflineOpenAITranscriptionSession.seal_turn": 1,
            "_StreamingOpenAIRealtimeSession.start": 1,
            "_StreamingOpenAIRealtimeSession.seal_turn": 1,
        },
        "Custom provider sessions retain transcription, websocket receive and final-timeout tasks under provider session close semantics",
        classification="legacy_exception",
    ),
    *_reviews(
        "providers/stt/deepgram.py",
        {
            "_DeepgramSDKSession.start": 1,
            "_DeepgramSDKSession._run": 3,
            "_DeepgramSDKSession.seal_turn": 1,
        },
        "Deepgram session retains SDK, send, receive, keepalive and drain tasks; _run finally and close cancel plus gather the task sets",
    ),
    *_reviews(
        "providers/stt/elevenlabs_scribe.py",
        {"_ElevenLabsScribeSession.start": 2},
        "ElevenLabs Scribe session owns its connection-event drain and keepalive tasks under provider session close semantics",
        classification="legacy_exception",
    ),
    *_reviews(
        "providers/stt/gemini_transcribe.py",
        {
            "_GeminiTranscribeLiveSession.start": 3,
            "_GeminiTranscribeLiveSession._start_teardown": 1,
        },
        "Gemini Transcribe Live session owns send, receive, cancellable handshake-enter and single teardown tasks and cancels plus joins them through the session teardown owner on close, failure, timeout and replacement; obsolete finalize-timeout debt is removed",
        classification="legacy_exception",
    ),
    *_reviews(
        "providers/stt/local_gpu.py",
        {
            "_LocalGpuSTTSession.seal_turn": 1,
            "_LocalGpuSTTSession.on_speech_end": 1,
            "_LocalGpuSTTSession._transcribe": 1,
        },
        "Local GPU STT sessions own transcription and submission tasks in an explicit set and cancel or gather tasks during stop and close",
    ),
    *_reviews(
        "providers/stt/local_qwen_sherpa.py",
        {"_LocalQwenSherpaSession.seal_turn": 1},
        "Local Qwen Sherpa session retains per-job decode watchdogs and cancels plus gathers the watchdog set on close",
    ),
    *_reviews(
        "providers/stt/qwen_audio.py",
        {
            "_QwenAudioSession._ready": 1,
            "_QwenAudioSession._seal": 1,
        },
        "Each Qwen channel owns two slot receivers and per-task final watchdogs, cancelled and joined on slot reuse, stop, or abort",
    ),
    *_reviews(
        "providers/stt/sdk_network.py",
        {"ExternalScribeRealtime._connect_audio": 1},
        "External Scribe adapter transfers the message task into RealtimeConnection._message_task; the pinned SDK connection close cancels and awaits it",
    ),
    *_reviews(
        "providers/stt/soniox.py",
        {"_SonioxSession.start": 3},
        "Soniox session owns send/receive/keepalive tasks under provider session close semantics",
        classification="legacy_exception",
    ),
    *_reviews(
        "release_evidence/cli_control_speech.py",
        {"_recognize": 1},
        "Speech evidence coroutine owns channel subscriber tasks and cancels plus gathers the set in finally; subscribers terminate their subprocesses",
    ),
    *_reviews(
        "release_evidence/local_asr_production_composition.py",
        {"_execute": 1},
        "Production composition evidence locally awaits provider handoff and cancels plus gathers it on failure before leaving the evidence boundary",
    ),
    *_reviews(
        "release_evidence/local_asr_production_composition.py",
        {"_run_stage": 1},
        "Bounded grandfathering: evidence stage cancels and gathers its waiter on timeout, but has no waiter cleanup on caller cancellation",
        classification="legacy_exception",
        family=ASYNCIO_ENSURE_FUTURE,
    ),
    *_reviews(
        "ui/components/settings/api_key_field.py",
        {"ApiKeyField._handle_blur": 1},
        "Flet API-key field callback uses page.run_task at the UI boundary for async verification",
        classification="legacy_exception",
        family=RUN_TASK,
    ),
    *_reviews(
        "ui/components/subtab_shell.py",
        {"_ScrollBody.restore_scroll": 1},
        "Flet subtab shell uses the page task runner for one bounded scroll restoration callback required by the asynchronous scroll API",
        classification="legacy_exception",
        family=BARE_RUN_TASK,
    ),
    *_reviews(
        "ui/desktop_overlay.py",
        {
            "FletDesktopRendererWindow.start": 2,
            "FletDesktopRendererWindow._schedule_bounds_sample": 1,
            "FletDesktopRendererWindow._run_page_task": 1,
            "DesktopOverlayRenderer._start_runtime_tasks": 5,
            "DesktopOverlayRenderer._ui_update_loop": 2,
        },
        "desktop overlay adapter owns renderer/app/websocket/window tasks and cancels them through overlay shutdown",
        classification="legacy_exception",
    ),
    *_reviews(
        "ui/desktop_overlay.py",
        {"FletDesktopRendererWindow._run_page_task": 1},
        "desktop overlay adapter has exactly one injected UI callback task-runner call site tracked by the renderer scheduler",
        classification="legacy_exception",
        family=BARE_RUN_TASK,
    ),
    *_reviews(
        "ui/desktop_overlay_repro.py",
        {"StaticCheckerboardBackdrop.start": 1, "DesktopOverlayReproOwner.run": 2},
        "DesktopOverlayReproOwner owns and gathers its renderer, diagnostic consumer, and static-backdrop tasks",
    ),
    *_reviews(
        "ui/flet_desktop_runtime.py",
        {
            "FletDesktopViewProcessOwner.close": 1,
            "FletDesktopViewProcessOwner._wait_for_process_exit": 1,
            "patch_hidden_view_launcher.launch": 1,
        },
        "FletDesktopViewProcessOwner owns close, process-wait, and bounded hidden-view spawn tasks and shield-awaits with cancel-drain before releasing process state",
    ),
    *_reviews(
        "ui/flet_runtime.py",
        {"run_control_method": 1},
        "Flet runtime helper schedules bounded control coroutines such as focus through the owning page task runner",
        classification="legacy_exception",
        family=BARE_RUN_TASK,
    ),
    *_reviews(
        "ui/foundation/runtime.py",
        {"FletFoundationRuntime.run_page_task": 1},
        "FletFoundationRuntime funnels UI callbacks through the owning page task runner and cancels them through application shutdown",
        family=RUN_TASK,
    ),
    *_reviews(
        "ui/foundation/runtime.py",
        {"FletFoundationRuntime.run_application_shutdown_task": 1},
        "FletFoundationRuntime retains shutdown-driver tasks separately from ordinary callbacks; shutdown driver is excluded from close self-cancellation, not drained by close",
        family=RUN_TASK,
    ),
    *_reviews(
        "ui/presentation_adapter.py",
        {"FletUiPresentationAdapter.schedule_task": 1},
        "Flet presentation adapter owns one injected page task-runner call site for UI callback scheduling",
        classification="legacy_exception",
        family=BARE_RUN_TASK,
    ),
    *_reviews(
        "ui/views/dashboard.py",
        {"DashboardView._run_notice_action": 1},
        "DashboardView uses the page task runner for one bounded async GPU notice action callback",
        classification="legacy_exception",
        family=BARE_RUN_TASK,
    ),
    *_reviews(
        "ui/views/settings.py",
        {"SettingsView._schedule_page_task": 1},
        "SettingsView uses page.run_task at the UI boundary for bounded asynchronous UI callbacks, including credential saving, while keeping the modal responsive",
        classification="legacy_exception",
        family=RUN_TASK,
    ),
)
REVIEWED_TASK_CREATION = dict(TASK_CREATION_REVIEWS)


def _allowlist_summary(classification: TaskClassification) -> Counter[tuple[str, str]]:
    summary: Counter[tuple[str, str]] = Counter()
    for identity, disposition in TASK_CREATION_REVIEWS:
        if disposition.classification == classification:
            summary[(identity.module, identity.family)] += disposition.max_count
    return summary


LEGACY_TASK_CREATION_ALLOWLIST = _allowlist_summary("legacy_exception")
NAMED_LIFECYCLE_OWNER_TASK_ALLOWLIST = _allowlist_summary("named_owner")
TASK_CREATION_ALLOWLIST_RATIONALES = {
    (identity.module, identity.family): disposition.rationale
    for identity, disposition in TASK_CREATION_REVIEWS
    if disposition.classification != "owner_primitive"
}


def test_first_party_creation_sites_have_reviewed_bounded_ownership() -> None:
    sites = scan_task_creation_sites(REPO_ROOT, PROTECTED_TASK_SOURCE_ROOTS)
    assert sites
    assert_task_ownership(sites, REVIEWED_TASK_CREATION)


def test_reviewed_task_identities_are_unique_and_have_lifetime_rationales() -> None:
    assert len(REVIEWED_TASK_CREATION) == len(TASK_CREATION_REVIEWS)
    assert_task_ownership([], REVIEWED_TASK_CREATION)


def test_lifecycle_scope_file_is_the_allowed_task_owner_primitive() -> None:
    assert (REPO_ROOT / "src" / "puripuly_heart" / "core" / "lifecycle.py").is_file()


def test_task_creation_allowlists_have_explicit_gate6_rationale() -> None:
    expected = LEGACY_TASK_CREATION_ALLOWLIST + NAMED_LIFECYCLE_OWNER_TASK_ALLOWLIST

    assert set(TASK_CREATION_ALLOWLIST_RATIONALES) == set(expected)
    assert all(
        rationale and "unclassified" not in rationale
        for rationale in TASK_CREATION_ALLOWLIST_RATIONALES.values()
    )


def test_order34_named_owner_allowlist_has_no_retired_stt_controller_tasks() -> None:
    stt_controller_tasks = (
        "src/puripuly_heart/core/stt/controller.py",
        ASYNCIO_CREATE_TASK,
    )

    assert stt_controller_tasks not in LEGACY_TASK_CREATION_ALLOWLIST
    assert stt_controller_tasks not in NAMED_LIFECYCLE_OWNER_TASK_ALLOWLIST


def test_order37_named_owner_allowlist_retires_controller_bounds_task_debt() -> None:
    assert (
        "src/puripuly_heart/ui/foundation/runtime.py",
        RUN_TASK,
    ) in NAMED_LIFECYCLE_OWNER_TASK_ALLOWLIST
    assert (
        "src/puripuly_heart/core/openrouter/managed_openrouter_release.py",
        ASYNCIO_CREATE_TASK,
    ) not in LEGACY_TASK_CREATION_ALLOWLIST
    assert (
        "src/puripuly_heart/core/runtime/oauth.py",
        ASYNCIO_CREATE_TASK,
    ) in NAMED_LIFECYCLE_OWNER_TASK_ALLOWLIST
    assert (
        "src/puripuly_heart/core/runtime/clipboard.py",
        ASYNCIO_CREATE_TASK,
    ) in NAMED_LIFECYCLE_OWNER_TASK_ALLOWLIST


def test_order38_named_owner_allowlist_preserves_installer_legacy_task_debt() -> None:
    assert (
        "src/puripuly_heart/core/local_asr/local_stt_runtime_installer.py",
        ASYNCIO_CREATE_TASK,
    ) in LEGACY_TASK_CREATION_ALLOWLIST
    assert (
        "src/puripuly_heart/core/runtime/local_stt_download.py",
        ASYNCIO_CREATE_TASK,
    ) in NAMED_LIFECYCLE_OWNER_TASK_ALLOWLIST
    assert (
        "src/puripuly_heart/core/runtime/mic_test.py",
        ASYNCIO_CREATE_TASK,
    ) in NAMED_LIFECYCLE_OWNER_TASK_ALLOWLIST


def test_order39_named_owner_allowlist_adds_receiver_prompt_and_bounds_owners() -> None:
    assert (
        "src/puripuly_heart/core/runtime/receiver.py",
        ASYNCIO_CREATE_TASK,
    ) in NAMED_LIFECYCLE_OWNER_TASK_ALLOWLIST
    assert (
        "src/puripuly_heart/core/runtime/github_star_prompt.py",
        ASYNCIO_CREATE_TASK,
    ) in NAMED_LIFECYCLE_OWNER_TASK_ALLOWLIST
    assert (
        "src/puripuly_heart/core/runtime/desktop_overlay_bounds.py",
        ASYNCIO_CREATE_TASK,
    ) in NAMED_LIFECYCLE_OWNER_TASK_ALLOWLIST
    assert (
        "src/puripuly_heart/ui/foundation/runtime.py",
        RUN_TASK,
    ) in NAMED_LIFECYCLE_OWNER_TASK_ALLOWLIST


def test_order40_named_owner_allowlist_adds_vrchat_osc_presence_owner() -> None:
    assert (
        "src/puripuly_heart/core/runtime/vrchat_osc_presence.py",
        ASYNCIO_CREATE_TASK,
    ) in NAMED_LIFECYCLE_OWNER_TASK_ALLOWLIST


def test_order41_managed_refresh_scheduling_uses_its_named_owner() -> None:
    controller_path = (
        REPO_ROOT / "src" / "puripuly_heart" / "composition" / "application_runtime.py"
    )
    owner_path = (
        REPO_ROOT / "src" / "puripuly_heart" / "app" / "services" / "managed" / "managed_usage.py"
    )
    controller_methods = {
        node.name: ast.unparse(node)
        for node in ast.walk(ast.parse(controller_path.read_text(encoding="utf-8")))
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    }
    owner_methods = {
        node.name: ast.unparse(node)
        for node in ast.walk(ast.parse(owner_path.read_text(encoding="utf-8")))
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    }

    assert "_schedule_owned_referral_id_status_refresh" not in controller_methods
    assert "schedule_status_refresh" in owner_methods["schedule_status_refresh"]
    assert "schedule_trial_usage_refresh" in owner_methods["schedule_usage_refresh"]


def test_order43_output_runtime_owns_ui_bridge_startup_waiter() -> None:
    controller_path = (
        REPO_ROOT / "src" / "puripuly_heart" / "composition" / "application_runtime.py"
    )
    output_path = REPO_ROOT / "src" / "puripuly_heart" / "core" / "runtime" / "output.py"
    controller_tree = ast.parse(controller_path.read_text(encoding="utf-8"))
    output_tree = ast.parse(output_path.read_text(encoding="utf-8"))
    controller_methods = {
        node.name: ast.unparse(node)
        for node in ast.walk(controller_tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    }
    output_methods = {
        node.name: ast.unparse(node)
        for node in ast.walk(output_tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    }

    composition_wait = controller_methods["wait_for_event_bridge"]
    assert "_ui_background_scope" not in composition_wait
    assert "output_runtime.wait_for_ui_event_bridge_started" in composition_wait
    assert "hub.output_runtime.wait_for_ui_event_bridge_started" not in composition_wait
    assert (
        "_ui_event_bridge_started_wait_task" in output_methods["wait_for_ui_event_bridge_started"]
    )
    assert "_cancel_ui_event_bridge_started_wait_task" in output_methods["close"]


def test_order44_local_asr_owner_retires_controller_background_scope() -> None:
    controller_path = (
        REPO_ROOT / "src" / "puripuly_heart" / "composition" / "application_runtime.py"
    )
    cpu_repair_path = (
        REPO_ROOT
        / "src"
        / "puripuly_heart"
        / "app"
        / "services"
        / "local_asr"
        / "local_asr_cpu_repair.py"
    )
    provisioning_path = (
        REPO_ROOT / "src" / "puripuly_heart" / "core" / "runtime" / "local_asr_provisioning.py"
    )
    controller_source = controller_path.read_text(encoding="utf-8")
    cpu_repair_source = cpu_repair_path.read_text(encoding="utf-8")
    provisioning_source = provisioning_path.read_text(encoding="utf-8")

    assert "_ui_background_scope" not in controller_source
    assert "result_handler=" not in controller_source
    assert "result_handler=lambda result: self.handle_install_result(" in cpu_repair_source
    assert "_result_delivery_tasks" in provisioning_source
    assert "LocalASRProvisioningOwner:install-result-" in provisioning_source
