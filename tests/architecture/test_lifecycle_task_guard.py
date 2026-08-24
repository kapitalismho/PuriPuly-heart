from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path

from tests.helpers.paths import REPO_ROOT, SOURCE_ROOT

ASYNCIO_CREATE_TASK = "asyncio.create_task"
ASYNCIO_ENSURE_FUTURE = "asyncio.ensure_future"
LOOP_CREATE_TASK = "loop.create_task"
BARE_RUN_TASK = "run_task(...)"
RUN_TASK = ".run_task"

LIFECYCLE_OWNER_PRIMITIVES = frozenset(
    {
        "src/puripuly_heart/core/lifecycle.py",
    }
)

LEGACY_TASK_CREATION_ALLOWLIST = Counter(
    {
        ("src/puripuly_heart/core/llm/fallback_racing.py", ASYNCIO_CREATE_TASK): 1,
        (
            "src/puripuly_heart/core/local_asr/local_stt_runtime_installer.py",
            ASYNCIO_CREATE_TASK,
        ): 1,
        ("src/puripuly_heart/core/stt/controller.py", ASYNCIO_CREATE_TASK): 6,
        ("src/puripuly_heart/core/overlay/bridge.py", ASYNCIO_CREATE_TASK): 1,
        ("src/puripuly_heart/core/overlay/presenter.py", ASYNCIO_CREATE_TASK): 1,
        ("src/puripuly_heart/providers/stt/custom.py", ASYNCIO_CREATE_TASK): 1,
        ("src/puripuly_heart/providers/stt/soniox.py", ASYNCIO_CREATE_TASK): 3,
        ("src/puripuly_heart/ui/components/subtab_shell.py", BARE_RUN_TASK): 1,
        ("src/puripuly_heart/ui/flet_runtime.py", BARE_RUN_TASK): 1,
        ("src/puripuly_heart/ui/components/settings/api_key_field.py", RUN_TASK): 1,
        ("src/puripuly_heart/ui/views/dashboard.py", BARE_RUN_TASK): 1,
        ("src/puripuly_heart/ui/views/settings.py", RUN_TASK): 1,
        ("src/puripuly_heart/ui/presentation_adapter.py", BARE_RUN_TASK): 1,
        ("src/puripuly_heart/ui/desktop_overlay.py", ASYNCIO_CREATE_TASK): 11,
        ("src/puripuly_heart/ui/desktop_overlay.py", BARE_RUN_TASK): 1,
        ("src/puripuly_heart/core/osc/receiver.py", LOOP_CREATE_TASK): 1,
    }
)

NAMED_LIFECYCLE_OWNER_TASK_ALLOWLIST = Counter(
    {
        (
            "src/puripuly_heart/core/orchestrator/self_translation_channel.py",
            ASYNCIO_CREATE_TASK,
        ): 4,
        ("src/puripuly_heart/core/runtime/peer_channel.py", ASYNCIO_CREATE_TASK): 1,
        ("src/puripuly_heart/core/runtime/provider_handle.py", ASYNCIO_CREATE_TASK): 1,
        ("src/puripuly_heart/core/runtime/self_capture.py", ASYNCIO_CREATE_TASK): 3,
        ("src/puripuly_heart/core/runtime/overlay.py", ASYNCIO_CREATE_TASK): 1,
        ("src/puripuly_heart/core/runtime/output.py", ASYNCIO_CREATE_TASK): 2,
        ("src/puripuly_heart/core/runtime/oauth.py", ASYNCIO_CREATE_TASK): 1,
        ("src/puripuly_heart/core/runtime/clipboard.py", ASYNCIO_CREATE_TASK): 1,
        (
            "src/puripuly_heart/core/runtime/desktop_overlay_bounds.py",
            ASYNCIO_CREATE_TASK,
        ): 1,
        ("src/puripuly_heart/core/overlay/process.py", ASYNCIO_CREATE_TASK): 3,
        (
            "src/puripuly_heart/core/runtime/overlay_session_fallback.py",
            ASYNCIO_CREATE_TASK,
        ): 1,
        (
            "src/puripuly_heart/core/runtime/vrchat_osc_presence.py",
            ASYNCIO_CREATE_TASK,
        ): 1,
        (
            "src/puripuly_heart/core/runtime/github_star_prompt.py",
            ASYNCIO_CREATE_TASK,
        ): 2,
        (
            "src/puripuly_heart/core/runtime/local_asr_provisioning.py",
            ASYNCIO_CREATE_TASK,
        ): 1,
        ("src/puripuly_heart/core/runtime/local_stt_download.py", ASYNCIO_CREATE_TASK): 1,
        ("src/puripuly_heart/core/runtime/mic_test.py", ASYNCIO_CREATE_TASK): 2,
        ("src/puripuly_heart/core/runtime/receiver.py", ASYNCIO_CREATE_TASK): 1,
        (
            "src/puripuly_heart/app/services/application_shutdown.py",
            ASYNCIO_CREATE_TASK,
        ): 3,
        ("src/puripuly_heart/app/services/manual_typing.py", LOOP_CREATE_TASK): 1,
        ("src/puripuly_heart/providers/stt/local_gpu.py", ASYNCIO_CREATE_TASK): 1,
        ("src/puripuly_heart/ui/desktop_overlay_repro.py", ASYNCIO_CREATE_TASK): 3,
        ("src/puripuly_heart/ui/flet_desktop_runtime.py", ASYNCIO_CREATE_TASK): 2,
        ("src/puripuly_heart/ui/foundation/runtime.py", RUN_TASK): 1,
    }
)

TASK_CREATION_ALLOWLIST_RATIONALES = {
    (
        "src/puripuly_heart/core/llm/fallback_racing.py",
        ASYNCIO_CREATE_TASK,
    ): "fallback racing owns short-lived contender tasks and cancels losers within the provider call boundary",
    (
        "src/puripuly_heart/core/local_asr/local_stt_runtime_installer.py",
        ASYNCIO_CREATE_TASK,
    ): "legacy installer download task remains deferred to the local STT download runtime owner cutover",
    (
        "src/puripuly_heart/core/stt/controller.py",
        ASYNCIO_CREATE_TASK,
    ): "managed STT provider still owns session consumer and reset timers until STT lifecycle is folded into an explicit runtime owner",
    (
        "src/puripuly_heart/core/orchestrator/self_translation_channel.py",
        ASYNCIO_CREATE_TASK,
    ): "Self translation owner owns per-utterance timeout and speculation tasks and cancels them through its explicit channel lifecycle",
    (
        "src/puripuly_heart/core/overlay/bridge.py",
        ASYNCIO_CREATE_TASK,
    ): "overlay bridge adapter wraps a named task factory and closes tasks through its adapter lifecycle",
    (
        "src/puripuly_heart/core/overlay/presenter.py",
        ASYNCIO_CREATE_TASK,
    ): "overlay presenter adapter wraps a named task factory and cancels presenter work during close",
    (
        "src/puripuly_heart/core/overlay/process.py",
        ASYNCIO_CREATE_TASK,
    ): "overlay subprocess and process-manager owners name reader, monitor, and locally awaited shutdown-cleanup tasks and gather them before releasing process state",
    (
        "src/puripuly_heart/providers/stt/custom.py",
        ASYNCIO_CREATE_TASK,
    ): "Custom realtime session owns its websocket receive task under provider session close semantics",
    (
        "src/puripuly_heart/providers/stt/soniox.py",
        ASYNCIO_CREATE_TASK,
    ): "Soniox session owns send/receive/keepalive tasks under provider session close semantics",
    (
        "src/puripuly_heart/ui/foundation/runtime.py",
        RUN_TASK,
    ): "FletFoundationRuntime funnels UI callbacks through the owning page task runner and cancels them through application shutdown",
    (
        "src/puripuly_heart/ui/components/subtab_shell.py",
        BARE_RUN_TASK,
    ): "Flet subtab shell uses the page task runner for one bounded scroll restoration callback required by the asynchronous scroll API",
    (
        "src/puripuly_heart/ui/flet_runtime.py",
        BARE_RUN_TASK,
    ): "Flet runtime helper schedules bounded control coroutines such as focus through the owning page task runner",
    (
        "src/puripuly_heart/ui/components/settings/api_key_field.py",
        RUN_TASK,
    ): "Flet API-key field callback uses page.run_task at the UI boundary for async verification",
    (
        "src/puripuly_heart/ui/views/dashboard.py",
        BARE_RUN_TASK,
    ): "DashboardView uses the page task runner for one bounded async GPU notice action callback",
    (
        "src/puripuly_heart/ui/views/settings.py",
        RUN_TASK,
    ): "SettingsView uses page.run_task at the UI boundary for loopback process capture options while keeping the modal responsive",
    (
        "src/puripuly_heart/ui/presentation_adapter.py",
        BARE_RUN_TASK,
    ): "Flet presentation adapter owns one injected page task-runner call site for UI callback scheduling",
    (
        "src/puripuly_heart/ui/desktop_overlay.py",
        ASYNCIO_CREATE_TASK,
    ): "desktop overlay adapter owns renderer/app/websocket/window tasks and cancels them through overlay shutdown",
    (
        "src/puripuly_heart/ui/desktop_overlay.py",
        BARE_RUN_TASK,
    ): "desktop overlay adapter has exactly one injected UI callback task-runner call site tracked by the renderer scheduler",
    (
        "src/puripuly_heart/ui/flet_desktop_runtime.py",
        ASYNCIO_CREATE_TASK,
    ): "FletDesktopViewProcessOwner owns close and process-wait tasks and awaits their terminal cleanup before releasing process state",
    (
        "src/puripuly_heart/core/osc/receiver.py",
        LOOP_CREATE_TASK,
    ): "OSC receiver mute-state callback schedules adapter-local async work on the owning loop until receiver runtime ownership fully wraps it",
    (
        "src/puripuly_heart/core/runtime/peer_channel.py",
        ASYNCIO_CREATE_TASK,
    ): "PeerChannelRuntime is the named lifecycle owner for its session loop",
    (
        "src/puripuly_heart/core/runtime/provider_handle.py",
        ASYNCIO_CREATE_TASK,
    ): "ProviderRuntimeHandle is the named lifecycle owner for provider event draining",
    (
        "src/puripuly_heart/core/runtime/self_capture.py",
        ASYNCIO_CREATE_TASK,
    ): "SelfCaptureSessionOwner owns intent transitions, its session loop, and contained fault teardown",
    (
        "src/puripuly_heart/core/runtime/overlay.py",
        ASYNCIO_CREATE_TASK,
    ): "OverlayRuntimeHandle is the named lifecycle owner for overlay tasks",
    (
        "src/puripuly_heart/core/runtime/overlay_session_fallback.py",
        ASYNCIO_CREATE_TASK,
    ): "OverlaySessionFallbackOwner owns its named deferred fallback task and cancels and awaits it during close",
    (
        "src/puripuly_heart/core/runtime/vrchat_osc_presence.py",
        ASYNCIO_CREATE_TASK,
    ): "VrchatOscPresenceProbeOwner owns its named probe task and cancels and awaits it during close",
    (
        "src/puripuly_heart/core/runtime/output.py",
        ASYNCIO_CREATE_TASK,
    ): "OutputRuntime is the named lifecycle owner for chatbox flush, overlay delivery, and UI bridge tasks",
    (
        "src/puripuly_heart/core/runtime/oauth.py",
        ASYNCIO_CREATE_TASK,
    ): "OAuthRuntime is the named lifecycle owner for managed-auth tasks",
    (
        "src/puripuly_heart/core/runtime/clipboard.py",
        ASYNCIO_CREATE_TASK,
    ): "ClipboardRuntime is the named lifecycle owner for clipboard watcher tasks",
    (
        "src/puripuly_heart/core/runtime/desktop_overlay_bounds.py",
        ASYNCIO_CREATE_TASK,
    ): "DesktopOverlayBoundsOwner owns the debounced bounds persistence task and cancels or gathers it before resets, setting changes, and shutdown",
    (
        "src/puripuly_heart/core/runtime/github_star_prompt.py",
        ASYNCIO_CREATE_TASK,
    ): "GithubStarPromptRuntime is the named lifecycle owner for prompt observation/launch tasks",
    (
        "src/puripuly_heart/core/runtime/local_asr_provisioning.py",
        ASYNCIO_CREATE_TASK,
    ): "LocalASRProvisioningOwner owns install-result delivery tasks and cancels and awaits them during close",
    (
        "src/puripuly_heart/core/runtime/local_stt_download.py",
        ASYNCIO_CREATE_TASK,
    ): "LocalSTTDownloadRuntime is the named lifecycle owner for download tasks",
    (
        "src/puripuly_heart/core/runtime/mic_test.py",
        ASYNCIO_CREATE_TASK,
    ): "MicTestRuntime is the named lifecycle owner for microphone test tasks",
    (
        "src/puripuly_heart/core/runtime/receiver.py",
        ASYNCIO_CREATE_TASK,
    ): "VrcMicReceiverRuntime is the named lifecycle owner for receiver tasks",
    (
        "src/puripuly_heart/app/services/application_shutdown.py",
        ASYNCIO_CREATE_TASK,
    ): "ApplicationShutdownCoordinator owns bounded callback and diagnostic tasks, cancels them on deadlines, and awaits terminal cleanup",
    (
        "src/puripuly_heart/app/services/manual_typing.py",
        LOOP_CREATE_TASK,
    ): "ManualTypingOwner owns its bounded idle timeout task and cancels it on input transitions and application shutdown",
    (
        "src/puripuly_heart/providers/stt/local_gpu.py",
        ASYNCIO_CREATE_TASK,
    ): "Local GPU STT sessions own transcription tasks in an explicit set and cancel or gather every task during stop and close",
    (
        "src/puripuly_heart/ui/desktop_overlay_repro.py",
        ASYNCIO_CREATE_TASK,
    ): "DesktopOverlayReproOwner owns and gathers its renderer, diagnostic consumer, and static-backdrop tasks",
}


def _repo_path(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def _is_asyncio_create_task_call(node: ast.Call) -> bool:
    return (
        isinstance(node.func, ast.Attribute)
        and node.func.attr == "create_task"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "asyncio"
    )


def _is_asyncio_ensure_future_call(node: ast.Call) -> bool:
    return (
        isinstance(node.func, ast.Attribute)
        and node.func.attr == "ensure_future"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "asyncio"
    )


def _is_loop_create_task_call(node: ast.Call) -> bool:
    if not isinstance(node.func, ast.Attribute) or node.func.attr != "create_task":
        return False
    if isinstance(node.func.value, ast.Name) and node.func.value.id == "asyncio":
        return False
    return True


def _is_bare_run_task_call(node: ast.Call) -> bool:
    return isinstance(node.func, ast.Name) and node.func.id == "run_task"


def _is_run_task_call(node: ast.Call) -> bool:
    return isinstance(node.func, ast.Attribute) and node.func.attr == "run_task"


def _task_creation_counts() -> Counter[tuple[str, str]]:
    counts: Counter[tuple[str, str]] = Counter()
    for source_file in sorted(SOURCE_ROOT.rglob("*.py")):
        relative_path = _repo_path(source_file)
        if relative_path in LIFECYCLE_OWNER_PRIMITIVES:
            continue

        tree = ast.parse(source_file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if _is_asyncio_create_task_call(node):
                counts[(relative_path, ASYNCIO_CREATE_TASK)] += 1
            elif _is_loop_create_task_call(node):
                counts[(relative_path, LOOP_CREATE_TASK)] += 1
            elif _is_asyncio_ensure_future_call(node):
                counts[(relative_path, ASYNCIO_ENSURE_FUTURE)] += 1
            elif _is_bare_run_task_call(node):
                counts[(relative_path, BARE_RUN_TASK)] += 1
            elif _is_run_task_call(node):
                counts[(relative_path, RUN_TASK)] += 1
    return counts


def test_lifecycle_scope_file_is_the_allowed_task_owner_primitive() -> None:
    assert (REPO_ROOT / "src" / "puripuly_heart" / "core" / "lifecycle.py").is_file()


def test_no_new_unmanaged_task_creation_outside_lifecycle_allowlist() -> None:
    actual = _task_creation_counts()
    expected = LEGACY_TASK_CREATION_ALLOWLIST + NAMED_LIFECYCLE_OWNER_TASK_ALLOWLIST
    unexpected = actual - expected
    stale = expected - actual

    assert not unexpected and not stale, (
        "Unmanaged background task inventory changed. New async work must go "
        "through LifecycleScope or a named lifecycle owner method; legacy "
        "exceptions must be reviewed before updating this allowlist.\n"
        f"Unexpected occurrences: {dict(unexpected)}\n"
        f"Stale allowlist entries: {dict(stale)}"
    )


def test_task_creation_allowlists_have_explicit_gate6_rationale() -> None:
    expected = LEGACY_TASK_CREATION_ALLOWLIST + NAMED_LIFECYCLE_OWNER_TASK_ALLOWLIST

    assert set(TASK_CREATION_ALLOWLIST_RATIONALES) == set(expected)
    assert all(
        rationale and "unclassified" not in rationale
        for rationale in TASK_CREATION_ALLOWLIST_RATIONALES.values()
    )


def test_order34_named_owner_allowlist_does_not_claim_stt_controller_legacy_tasks() -> None:
    stt_controller_tasks = (
        "src/puripuly_heart/core/stt/controller.py",
        ASYNCIO_CREATE_TASK,
    )

    assert stt_controller_tasks in LEGACY_TASK_CREATION_ALLOWLIST
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


def test_application_composition_does_not_retain_dead_shutdown_or_provider_algorithms() -> None:
    controller_path = (
        REPO_ROOT / "src" / "puripuly_heart" / "composition" / "application_runtime.py"
    )
    provider_settings_path = (
        REPO_ROOT
        / "src"
        / "puripuly_heart"
        / "app"
        / "services"
        / "provider"
        / "provider_settings.py"
    )
    provider_runtime_path = (
        REPO_ROOT
        / "src"
        / "puripuly_heart"
        / "app"
        / "services"
        / "provider"
        / "provider_runtime_apply.py"
    )
    startup_path = (
        REPO_ROOT / "src" / "puripuly_heart" / "app" / "services" / "application_startup.py"
    )
    startup_adapter_path = (
        REPO_ROOT / "src" / "puripuly_heart" / "composition" / "application_startup.py"
    )
    shutdown_path = (
        REPO_ROOT
        / "src"
        / "puripuly_heart"
        / "app"
        / "services"
        / "application_runtime_shutdown.py"
    )
    source = controller_path.read_text(encoding="utf-8")
    provider_settings_source = provider_settings_path.read_text(encoding="utf-8")
    provider_runtime_source = provider_runtime_path.read_text(encoding="utf-8")
    startup_source = startup_path.read_text(encoding="utf-8")
    startup_adapter_source = startup_adapter_path.read_text(encoding="utf-8")
    shutdown_source = shutdown_path.read_text(encoding="utf-8")

    assert "_stop_lock" not in source
    assert "_stt_switch_lock" not in source
    assert "_provider_secret_change_lock" not in source
    assert "_provider_secret_change_serialization_owner" not in source
    assert "_provider_secret_change_owner" not in source
    assert "provider_settings: ProviderSettingsOwner | None" in source
    assert "secret_change: ProviderSecretChangeOwner" in provider_settings_source
    assert "class ProviderApplicationOwner" in provider_settings_source
    assert "class ProviderRuntimeOwner" in provider_runtime_source
    assert "class LlmProviderRebuildOwner" in provider_runtime_source
    assert "_build_provider_runtime_apply_plan" not in source
    assert "_apply_provider_runtime_plan" not in source
    assert "_apply_order21_order22_order24_provider_settings" not in source
    assert "_apply_translation_provider_settings_via_mutation_service" not in source
    assert "_apply_stt_language_audio_provider_settings_via_mutation_service" not in source
    assert "_apply_providers_direct" not in source
    assert "_rebuild_llm_provider" not in source
    assert "SettingsRuntimeApplyBoundary" not in provider_runtime_source
    assert "_ControllerSttLanguageAudioRuntimeApply" not in provider_runtime_source
    assert "_gpu_provider_recovery_lock" not in source
    assert "gpu_recovery: GpuProviderRecoveryApplicationOwner | None" in source
    assert "_overlay_lock" not in source
    assert "overlay: OverlayApplicationOwner | None" in source
    assert "def application_shutdown_callbacks" not in source
    assert "ApplicationShutdownCoordinator" not in source
    assert "application_shutdown_callback(" not in source
    assert "class ApplicationStartupOwner" in startup_source
    assert "_get_" not in startup_source
    assert "Any" not in startup_source
    assert "await self.provisioning.inspect_cpu()" in startup_adapter_source
    assert "await self.pipeline_launcher.launch(" in startup_adapter_source
    assert "compose_application_runtime_shutdown_callbacks" in shutdown_source
    assert "SHUTDOWN_PHASE_CLOSE_LOGGING_DIAGNOSTICS" in shutdown_source
