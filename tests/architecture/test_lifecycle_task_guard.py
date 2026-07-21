from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / "src" / "puripuly_heart"

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
            "src/puripuly_heart/core/local_stt_runtime_installer.py",
            ASYNCIO_CREATE_TASK,
        ): 1,
        ("src/puripuly_heart/core/stt/controller.py", ASYNCIO_CREATE_TASK): 6,
        ("src/puripuly_heart/core/orchestrator/hub.py", ASYNCIO_CREATE_TASK): 4,
        ("src/puripuly_heart/core/overlay/bridge.py", ASYNCIO_CREATE_TASK): 1,
        ("src/puripuly_heart/core/overlay/presenter.py", ASYNCIO_CREATE_TASK): 1,
        ("src/puripuly_heart/core/overlay/process.py", ASYNCIO_CREATE_TASK): 2,
        ("src/puripuly_heart/providers/stt/soniox.py", ASYNCIO_CREATE_TASK): 3,
        ("src/puripuly_heart/ui/app.py", RUN_TASK): 1,
        ("src/puripuly_heart/ui/components/settings/api_key_field.py", RUN_TASK): 1,
        ("src/puripuly_heart/ui/views/dashboard.py", BARE_RUN_TASK): 1,
        ("src/puripuly_heart/ui/views/settings.py", RUN_TASK): 1,
        ("src/puripuly_heart/ui/controller.py", ASYNCIO_CREATE_TASK): 1,
        ("src/puripuly_heart/ui/controller.py", LOOP_CREATE_TASK): 3,
        ("src/puripuly_heart/ui/controller.py", BARE_RUN_TASK): 5,
        ("src/puripuly_heart/ui/desktop_overlay.py", ASYNCIO_CREATE_TASK): 11,
        ("src/puripuly_heart/ui/desktop_overlay.py", BARE_RUN_TASK): 1,
        ("src/puripuly_heart/core/osc/receiver.py", LOOP_CREATE_TASK): 1,
    }
)

NAMED_LIFECYCLE_OWNER_TASK_ALLOWLIST = Counter(
    {
        ("src/puripuly_heart/core/runtime/peer_channel.py", ASYNCIO_CREATE_TASK): 1,
        ("src/puripuly_heart/core/runtime/provider_handle.py", ASYNCIO_CREATE_TASK): 1,
        ("src/puripuly_heart/core/runtime/self_capture.py", ASYNCIO_CREATE_TASK): 3,
        ("src/puripuly_heart/core/runtime/overlay.py", ASYNCIO_CREATE_TASK): 1,
        ("src/puripuly_heart/core/runtime/output.py", ASYNCIO_CREATE_TASK): 2,
        ("src/puripuly_heart/core/runtime/oauth.py", ASYNCIO_CREATE_TASK): 1,
        ("src/puripuly_heart/core/runtime/clipboard.py", ASYNCIO_CREATE_TASK): 1,
        (
            "src/puripuly_heart/core/runtime/github_star_prompt.py",
            ASYNCIO_CREATE_TASK,
        ): 2,
        ("src/puripuly_heart/core/runtime/local_stt_download.py", ASYNCIO_CREATE_TASK): 1,
        ("src/puripuly_heart/core/runtime/mic_test.py", ASYNCIO_CREATE_TASK): 2,
        ("src/puripuly_heart/core/runtime/receiver.py", ASYNCIO_CREATE_TASK): 1,
        (
            "src/puripuly_heart/app/services/application_shutdown.py",
            ASYNCIO_CREATE_TASK,
        ): 3,
        ("src/puripuly_heart/providers/stt/local_gpu.py", ASYNCIO_CREATE_TASK): 1,
        ("src/puripuly_heart/ui/desktop_overlay_repro.py", ASYNCIO_CREATE_TASK): 3,
    }
)

TASK_CREATION_ALLOWLIST_RATIONALES = {
    (
        "src/puripuly_heart/core/llm/fallback_racing.py",
        ASYNCIO_CREATE_TASK,
    ): "fallback racing owns short-lived contender tasks and cancels losers within the provider call boundary",
    (
        "src/puripuly_heart/core/local_stt_runtime_installer.py",
        ASYNCIO_CREATE_TASK,
    ): "legacy installer download task remains deferred to the local STT download runtime owner cutover",
    (
        "src/puripuly_heart/core/stt/controller.py",
        ASYNCIO_CREATE_TASK,
    ): "managed STT provider still owns session consumer and reset timers until STT lifecycle is folded into an explicit runtime owner",
    (
        "src/puripuly_heart/core/orchestrator/hub.py",
        ASYNCIO_CREATE_TASK,
    ): "orchestrator owns per-utterance timeout/speculation tasks with explicit buffer cleanup; broader owner extraction is deferred work",
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
    ): "overlay process manager wraps subprocess monitor tasks with named factories and shutdown cleanup",
    (
        "src/puripuly_heart/providers/stt/soniox.py",
        ASYNCIO_CREATE_TASK,
    ): "Soniox session owns send/receive/keepalive tasks under provider session close semantics",
    (
        "src/puripuly_heart/ui/app.py",
        RUN_TASK,
    ): "TranslatorApp funnels Flet async callbacks through one tracked page.run_task helper and cancels them during shutdown",
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
    ): "SettingsView uses page.run_task at the UI boundary to load loopback process capture options asynchronously while keeping the modal responsive",
    (
        "src/puripuly_heart/ui/controller.py",
        ASYNCIO_CREATE_TASK,
    ): "controller retains one bounded UI task handle for desktop-bounds persistence after G09 moved other task families behind lifecycle scopes",
    (
        "src/puripuly_heart/ui/controller.py",
        LOOP_CREATE_TASK,
    ): "controller retains three loop-bound UI scheduling call sites for overlay updates and owner-scoped manual typing idle timeout after G09 lifecycle cutover",
    (
        "src/puripuly_heart/ui/controller.py",
        BARE_RUN_TASK,
    ): "controller has exactly five injected UI task-runner call sites for overlay, calibration, and runtime callback scheduling",
    (
        "src/puripuly_heart/ui/desktop_overlay.py",
        ASYNCIO_CREATE_TASK,
    ): "desktop overlay adapter owns renderer/app/websocket/window tasks and cancels them through overlay shutdown",
    (
        "src/puripuly_heart/ui/desktop_overlay.py",
        BARE_RUN_TASK,
    ): "desktop overlay adapter has exactly one injected UI callback task-runner call site tracked by the renderer scheduler",
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
        "src/puripuly_heart/core/runtime/github_star_prompt.py",
        ASYNCIO_CREATE_TASK,
    ): "GithubStarPromptRuntime is the named lifecycle owner for prompt observation/launch tasks",
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


def test_order37_named_owner_allowlist_preserves_remaining_legacy_ui_task_debt() -> None:
    assert (
        "src/puripuly_heart/ui/controller.py",
        ASYNCIO_CREATE_TASK,
    ) in LEGACY_TASK_CREATION_ALLOWLIST
    assert ("src/puripuly_heart/ui/app.py", RUN_TASK) in LEGACY_TASK_CREATION_ALLOWLIST
    assert (
        "src/puripuly_heart/core/managed_openrouter_release.py",
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
        "src/puripuly_heart/core/local_stt_runtime_installer.py",
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


def test_order39_named_owner_allowlist_adds_receiver_and_prompt_owners() -> None:
    assert (
        "src/puripuly_heart/core/runtime/receiver.py",
        ASYNCIO_CREATE_TASK,
    ) in NAMED_LIFECYCLE_OWNER_TASK_ALLOWLIST
    assert (
        "src/puripuly_heart/core/runtime/github_star_prompt.py",
        ASYNCIO_CREATE_TASK,
    ) in NAMED_LIFECYCLE_OWNER_TASK_ALLOWLIST
    assert (
        "src/puripuly_heart/ui/controller.py",
        ASYNCIO_CREATE_TASK,
    ) in LEGACY_TASK_CREATION_ALLOWLIST
    assert ("src/puripuly_heart/ui/app.py", RUN_TASK) in LEGACY_TASK_CREATION_ALLOWLIST
