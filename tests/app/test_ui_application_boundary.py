from __future__ import annotations

import asyncio
import inspect
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from puripuly_heart.app.ports.ui_models import OverlayPeerPresentationState
from puripuly_heart.app.services.application_shutdown import (
    ApplicationIntentRejectedError,
    ApplicationShutdownRuntimeState,
    application_shutdown_callback,
)
from puripuly_heart.app.services.ui_application import (
    UI_APPLICATION_USER_INTENT_METHODS,
)
from puripuly_heart.app.services.ui_application import (
    UiApplicationBoundary as ProductionUiApplicationBoundary,
)
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext
from puripuly_heart.core.lifecycle import SHUTDOWN_PHASE_FREEZE_INGRESS
from tests.helpers.ui_application import compose_test_ui_application_boundary


def UiApplicationBoundary(
    backend: object,
    *,
    osc_state_publisher=None,
) -> ProductionUiApplicationBoundary:
    return compose_test_ui_application_boundary(
        backend,
        osc_state_publisher=osc_state_publisher,
    )


class RecordingBackend:
    def __init__(self) -> None:
        self.config_path = Path("settings.json")
        self.settings = replace(
            AppSettingsVNext(),
            intent=replace(
                AppSettingsVNext().intent,
                translation=replace(AppSettingsVNext().intent.translation, model="local_llm"),
                overlay=replace(AppSettingsVNext().intent.overlay, target="desktop"),
            ),
        )
        self.hub = SimpleNamespace(
            translation_enabled=True,
            llm=object(),
            stt_session_state=lambda channel: f"{channel}-listening",
        )
        self.audio_test_active = True
        self.desktop_overlay_captions_locked = True
        self.last_discord_managed_auth_referral_bonus_applied = True
        self.peer_retry_result = True
        self.events: list[tuple[object, ...]] = []

    async def start(self) -> None:
        self.events.append(("start",))

    def emit_application_shutdown_diagnostic(self, diagnostic: object) -> None:
        self.events.append(("shutdown-diagnostic", diagnostic))

    def log_basic(self, message: str, *, level: int) -> None:
        self.events.append(("log-basic", message, level))

    def log_diagnostic(self, message: str, *, level: int) -> None:
        self.events.append(("log-detailed", message, level))

    async def submit_text(self, text: str) -> None:
        self.events.append(("submit", text))

    async def set_translation_enabled(
        self, enabled: bool, *, allow_authorization: bool = True
    ) -> bool:
        _ = allow_authorization
        self.events.append(("translation", enabled))
        return enabled

    async def set_stt_enabled(self, enabled: bool) -> bool:
        self.events.append(("self", enabled))
        return enabled

    async def set_peer_translation_enabled(self, enabled: bool) -> bool:
        self.events.append(("peer", enabled))
        return enabled

    async def set_overlay_enabled(self, enabled: bool) -> bool:
        self.events.append(("overlay", enabled))
        return enabled

    async def retry_peer_process_capture(self) -> bool:
        self.events.append(("peer-retry",))
        return self.peer_retry_result

    async def apply_settings(self, settings: object, *, provider_requested: bool = False) -> None:
        self.settings = settings
        self.events.append(("settings", settings))

    def refresh_settings_projection(
        self,
        *,
        preserve_custom_vocab_draft: bool = False,
    ) -> bool:
        self.events.append(("settings-projection", preserve_custom_vocab_draft))
        return True

    def refresh_settings_after_openrouter_pkce_success(self) -> bool:
        self.events.append(("settings-pkce-projection",))
        return True

    async def apply_providers(self, *args, **kwargs) -> None:
        self.events.append(("providers", args, kwargs))

    async def install_selected_gpu_model_if_needed(self) -> None:
        self.events.append(("gpu-install",))

    async def ensure_gpu_device_discovery(self) -> None:
        self.events.append(("gpu-discovery",))

    async def apply_telemetry_enabled(self, enabled: bool) -> object:
        self.events.append(("telemetry", enabled))
        return self.settings

    def persist_settings(self) -> None:
        self.events.append(("persist",))

    def clear_provider_verification(self, provider: str) -> None:
        from puripuly_heart.config.settings_vnext.schema import ProviderVerificationEntry

        verification = self.settings.state.provider_verification
        self.settings = replace(
            self.settings,
            state=replace(
                self.settings.state,
                provider_verification=replace(
                    verification,
                    **{provider: ProviderVerificationEntry(status="unknown")},
                ),
            ),
        )
        self.events.append(("clear-verification", provider))

    async def verify_api_key(self, provider: str, key: str) -> tuple[bool, str]:
        self.events.append(("verify", provider, key))
        return True, "verified"

    def persist_api_key_verification(self, provider: str, key: str, success: bool) -> None:
        self.events.append(("persist-verification", provider, key, success))

    async def persist_provider_secret_change(self, key: str, value: str) -> bool:
        self.events.append(("secret", key, value))
        return True

    async def start_qq_managed_auth_from_dialog(self, **kwargs: object) -> object:
        self.events.append(("qq-auth", kwargs))
        return "qq-result"

    async def start_discord_managed_auth_from_dialog(self, **kwargs: object) -> object:
        self.events.append(("discord-auth", kwargs))
        return "discord-result"

    def reopen_discord_managed_auth_browser(self) -> bool:
        self.events.append(("discord-reopen",))
        return True

    def cancel_discord_managed_auth(self) -> bool:
        self.events.append(("discord-cancel",))
        return True

    async def set_desktop_overlay_captions_locked(self, locked: bool) -> None:
        self.events.append(("overlay-lock", locked))

    async def set_desktop_overlay_size_preset(self, size_preset: str) -> None:
        self.events.append(("overlay-size", size_preset))

    async def reset_desktop_overlay_position(self) -> None:
        self.events.append(("overlay-reset",))

    def begin_overlay_calibration(self) -> object:
        self.events.append(("calibration-begin",))
        return "calibration"

    def set_overlay_calibration_field(self, field: str, value: object) -> object:
        self.events.append(("calibration-field", field, value))
        return value

    def apply_overlay_calibration(self) -> object:
        self.events.append(("calibration-apply",))
        return True

    def cancel_overlay_calibration(self) -> object:
        self.events.append(("calibration-cancel",))
        return True

    def overlay_peer_presentation_state(self) -> OverlayPeerPresentationState:
        self.events.append(("overlay-peer-contract",))
        return OverlayPeerPresentationState(
            overlay_intent_enabled=True,
            overlay_state="connected",
            overlay_failure_reason=None,
            peer_intent_enabled=True,
            peer_effective_enabled=True,
            peer_warning_reason=None,
            peer_activation_starting=False,
        )

    def cycle_debug_capture_fault_profile(self) -> str:
        self.events.append(("capture-fault",))
        return "capture-failure"

    def cycle_debug_stt_fault_profile(self) -> str:
        self.events.append(("stt-fault",))
        return "stt-failure"

    def clear_debug_audio_fault_profiles(self) -> None:
        self.events.append(("fault-clear",))

    def handle_gpu_notice_action(self, action: str) -> object:
        self.events.append(("gpu-retry", action))
        return "retrying"

    async def persist_github_star_prompt_opened(
        self,
        *,
        should_open=None,
    ) -> bool:
        self.events.append(("github-open", should_open))
        return should_open is None or bool(should_open())


def test_state_is_a_semantic_snapshot_without_exposing_backend_objects() -> None:
    backend = RecordingBackend()
    boundary = UiApplicationBoundary(backend)

    state = boundary.state()

    assert state.config_path == Path("settings.json")
    assert state.translation_enabled is True
    assert state.stt_state == "self-listening"
    assert state.peer_translation_eula_accepted is False
    assert state.audio_test_active is True
    assert state.provider_name == "local_llm"
    assert state.overlay_target == "desktop"
    assert state.desktop_overlay_captions_locked is True
    assert state.managed_auth_referral_bonus_applied is True
    assert not hasattr(state, "hub")
    assert not hasattr(state, "settings")


def test_compatibility_settings_is_detached_and_missing_ui_state_stays_unknown() -> None:
    backend = RecordingBackend()
    boundary = UiApplicationBoundary(backend)

    detached = boundary.compatibility_settings()
    assert detached == backend.settings
    assert detached is not backend.settings
    _ = replace(
        detached,
        state=replace(
            detached.state,
            peer_translation=replace(detached.state.peer_translation, eula_accepted=True),
        ),
    )
    assert backend.settings.state.peer_translation.eula_accepted is False
    backend.settings = None
    assert boundary.state().peer_translation_eula_accepted is None


@pytest.mark.asyncio
async def test_start_failure_runs_owned_shutdown_and_preserves_original_error() -> None:
    class FailingBackend(RecordingBackend):
        async def start(self) -> None:
            self.events.append(("start",))
            raise RuntimeError("pipeline failed")

    backend = FailingBackend()
    boundary = UiApplicationBoundary(backend)
    cleanup: list[str] = []
    boundary.register_application_shutdown_callbacks(
        (
            application_shutdown_callback(
                phase=SHUTDOWN_PHASE_FREEZE_INGRESS,
                owner_name="TestRuntime",
                callback_name="cleanup",
                callback=lambda: cleanup.append("cleanup"),
            ),
        )
    )

    with pytest.raises(RuntimeError, match="pipeline failed"):
        await boundary.start()

    assert cleanup == ["cleanup"]
    assert boundary.application_lifecycle().is_terminal


@pytest.mark.asyncio
async def test_eula_acceptance_is_owned_at_the_boundary_before_peer_enable() -> None:
    backend = RecordingBackend()
    backend.audio_test_active = False
    boundary = UiApplicationBoundary(backend)

    result = await boundary.accept_peer_translation_eula_and_enable()

    assert result is True
    assert backend.events[0][0] == "settings"
    assert backend.settings.state.peer_translation.eula_accepted is True
    assert backend.events[1] == ("peer", True)


@pytest.mark.asyncio
async def test_explicit_capture_choice_during_audio_test_takes_over_from_the_test() -> None:
    backend = RecordingBackend()

    async def stop_audio_test() -> None:
        backend.events.append(("audio-test-stop",))
        backend.audio_test_active = False

    backend.stop_audio_test = stop_audio_test
    backend.release_audio_test_capture_restore = lambda channel: backend.events.append(
        ("audio-test-release", channel)
    )
    boundary = UiApplicationBoundary(backend)

    await boundary.set_peer_translation_enabled(False)
    await boundary.set_stt_enabled(True)
    await boundary.set_peer_translation_enabled(True)

    assert backend.events == [
        ("audio-test-release", "peer"),
        ("peer", False),
        ("audio-test-release", "self"),
        ("audio-test-stop",),
        ("self", True),
        ("audio-test-release", "peer"),
        ("peer", True),
    ]


@pytest.mark.asyncio
async def test_frozen_boundary_rejects_mutating_intents_but_keeps_stall_diagnostic_callable() -> (
    None
):
    backend = RecordingBackend()
    boundary = UiApplicationBoundary(backend)
    freeze_started = asyncio.Event()
    backend.application_shutdown_runtime_states = lambda: (
        ApplicationShutdownRuntimeState(
            owner_name="SelfCaptureSessionOwner",
            generation=7,
            active_native_operations=("capture-loop",),
            child_states=("audio-helper:pid=123:running",),
        ),
    )
    release_freeze = asyncio.Event()

    async def freeze() -> None:
        freeze_started.set()
        await release_freeze.wait()

    boundary.register_application_shutdown_callbacks(
        (
            application_shutdown_callback(
                phase=SHUTDOWN_PHASE_FREEZE_INGRESS,
                owner_name="Application",
                callback_name="freeze",
                callback=freeze,
            ),
        )
    )
    lifecycle = boundary.application_lifecycle()
    shutdown_task = asyncio.create_task(lifecycle.shutdown())
    await freeze_started.wait()
    backend.events.clear()

    for intent_name in sorted(UI_APPLICATION_USER_INTENT_METHODS):
        intent = getattr(boundary, intent_name)
        with pytest.raises(ApplicationIntentRejectedError) as exc_info:
            if inspect.iscoroutinefunction(intent):
                await intent()
            else:
                intent()
        assert exc_info.value.intent_name == intent_name

    diagnostic = boundary.capture_application_shutdown_stall_diagnostic()
    assert diagnostic.coordinator_state == "shutting_down"
    assert diagnostic.coordinator_terminal is False
    assert diagnostic.phase == SHUTDOWN_PHASE_FREEZE_INGRESS
    assert diagnostic.active_owner_name == "Application"
    assert diagnostic.active_callback_name == "freeze"
    assert diagnostic.runtime_states == (
        ApplicationShutdownRuntimeState(
            owner_name="SelfCaptureSessionOwner",
            generation=7,
            active_native_operations=("capture-loop",),
            child_states=("audio-helper:pid=123:running",),
        ),
    )
    assert "application-shutdown-coordinator" in diagnostic.task_await_graphs
    assert diagnostic.task_await_graphs["application-shutdown-coordinator"]

    assert backend.events == []
    release_freeze.set()
    await shutdown_task

    with pytest.raises(ApplicationIntentRejectedError):
        await boundary.submit_text("late")
    assert backend.events == []


@pytest.mark.asyncio
async def test_boundary_preserves_settings_failure_and_restart_projection() -> None:
    class FailingBackend(RecordingBackend):
        async def apply_settings(
            self, settings: object, *, provider_requested: bool = False
        ) -> None:
            self.events.append(("settings-failed", settings))
            raise RuntimeError("settings apply failed")

    failing_backend = FailingBackend()
    with pytest.raises(RuntimeError, match="settings apply failed"):
        await UiApplicationBoundary(failing_backend).apply_settings(object())

    restored_backend = RecordingBackend()
    restored_backend.settings = replace(
        restored_backend.settings,
        state=replace(
            restored_backend.settings.state,
            peer_translation=replace(
                restored_backend.settings.state.peer_translation,
                eula_accepted=True,
            ),
        ),
    )
    restored = UiApplicationBoundary(restored_backend)

    assert restored.state().peer_translation_eula_accepted is True
    assert failing_backend.events[0][0] == "settings-failed"


@pytest.mark.asyncio
async def test_boundary_stop_owns_managed_auth_task_cancellation_and_terminal_close() -> None:
    boundary = UiApplicationBoundary(RecordingBackend())
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def auth_task() -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    handle = boundary.start_managed_auth_task(
        task_runner=lambda factory: asyncio.create_task(factory()),
        task_factory=auth_task,
        task_name="discord-managed-auth-dialog",
        generation=1,
    )
    await started.wait()

    assert boundary.managed_auth_task_names() == ("discord-managed-auth-dialog",)
    await boundary.stop()
    await asyncio.gather(handle, return_exceptions=True)

    assert handle.cancelled() is True
    assert cancelled.is_set() is True
    assert boundary.managed_auth_tasks_open() is False
    assert boundary.application_lifecycle().is_terminal


@pytest.mark.asyncio
async def test_boundary_stop_owns_github_prompt_generation_and_cancellation() -> None:
    boundary = UiApplicationBoundary(RecordingBackend())
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def prompt(generation: int) -> bool:
        assert boundary.is_current_github_star_prompt_generation(generation)
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
        return True

    task = boundary.start_github_star_prompt(prompt)
    await started.wait()
    await boundary.stop()

    assert await task is False
    assert cancelled.is_set() is True
    assert boundary.application_lifecycle().is_terminal


@pytest.mark.asyncio
@pytest.mark.parametrize("paused_channel", ("self", "peer"))
@pytest.mark.parametrize("override_channel", ("self", "peer"))
async def test_audio_start_orders_gated_pause_before_explicit_capture_takeover(
    paused_channel: str, override_channel: str
) -> None:
    from tests.app.test_audio_test_session_owner import Harness

    harness = Harness()
    pause_entered = asyncio.Event()
    release_pause = asyncio.Event()
    original = harness.control.set_enabled

    async def set_enabled(channel: str, enabled: bool) -> None:
        await original(channel, enabled)
        if channel == paused_channel and not enabled:
            pause_entered.set()
            await release_pause.wait()

    harness.control.set_enabled = set_enabled

    class Backend(RecordingBackend):
        def __getattribute__(self, name):
            if name == "audio_test_active":
                return harness.owner.active
            return super().__getattribute__(name)

        async def start_audio_test(self, **kwargs):
            return await harness.owner.start(**kwargs)

        async def stop_audio_test(self):
            await harness.owner.stop()

        def release_audio_test_capture_restore(self, channel):
            harness.owner.release_capture_restore(channel)

        async def set_stt_enabled(self, enabled):
            await original("self", enabled)

        async def set_peer_translation_enabled(self, enabled):
            await original("peer", enabled)

    boundary = UiApplicationBoundary(Backend())
    starting = asyncio.create_task(boundary.start_audio_test())
    takeover = None
    try:
        await pause_entered.wait()
        assert harness.owner.active
        assert harness.owner.snapshot["state"] == "pending"
        operation = (
            boundary.set_stt_enabled
            if override_channel == "self"
            else boundary.set_peer_translation_enabled
        )
        takeover = asyncio.create_task(operation(True))
        await asyncio.sleep(0)
        assert not takeover.done()
        assert not any(harness.control.enabled.values()) or paused_channel == "self"
        release_pause.set()
        assert await starting
        await takeover
        assert not harness.owner.active
        assert harness.owner.snapshot["state"] == "off"
        assert harness.control.enabled == {"self": True, "peer": True}
        assert override_channel not in harness.owner.snapshot["restore_capture"]
    finally:
        release_pause.set()
        await asyncio.gather(
            *(task for task in (starting, takeover) if task is not None), return_exceptions=True
        )
        await harness.owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ("stop", "close"))
async def test_audio_stop_interrupts_startup_while_capture_pause_is_gated(operation: str) -> None:
    from tests.app.test_audio_test_session_owner import Harness

    harness = Harness()
    paused = asyncio.Event()
    release = asyncio.Event()
    original = harness.control.set_enabled

    async def set_enabled(channel: str, enabled: bool) -> None:
        await original(channel, enabled)
        if channel == "peer" and not enabled:
            paused.set()
            await release.wait()

    harness.control.set_enabled = set_enabled
    starting = asyncio.create_task(harness.owner.start())
    stopping = None
    try:
        await paused.wait()
        stopping = asyncio.create_task(getattr(harness.owner, operation)())
        await asyncio.sleep(0)
        release.set()
        assert await starting is False
        await stopping
        assert harness.owner.snapshot["state"] == "off"
        assert harness.control.enabled == {"self": operation == "stop", "peer": operation == "stop"}
        assert all(
            harness.owner.runtime_for(channel).source is None for channel in ("self", "peer")
        )
    finally:
        release.set()
        await asyncio.gather(
            *(task for task in (starting, stopping) if task is not None), return_exceptions=True
        )
        await harness.owner.close()
