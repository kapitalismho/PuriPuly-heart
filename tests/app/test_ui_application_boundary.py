from __future__ import annotations

import asyncio
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from puripuly_heart.app.ports.ui_models import OverlayPeerPresentationState
from puripuly_heart.app.services.application_shutdown import (
    ApplicationIntentRejectedError,
    application_shutdown_callback,
)
from puripuly_heart.app.services.ui_application import (
    UI_APPLICATION_USER_INTENT_METHODS,
)
from puripuly_heart.app.services.ui_application import (
    UiApplicationBoundary as ProductionUiApplicationBoundary,
)
from puripuly_heart.core.lifecycle import SHUTDOWN_PHASE_FREEZE_INGRESS
from tests.helpers.ui_application import compose_test_ui_application_boundary


def UiApplicationBoundary(backend: object) -> ProductionUiApplicationBoundary:
    return compose_test_ui_application_boundary(backend)


class RecordingBackend:
    def __init__(self) -> None:
        self.config_path = Path("settings.json")
        self.runtime_logging_mode = "detailed"
        self.settings = SimpleNamespace(
            ui=SimpleNamespace(peer_translation_eula_accepted=False),
            provider=SimpleNamespace(llm=SimpleNamespace(value="local_llm")),
            overlay=SimpleNamespace(target=SimpleNamespace(value="desktop")),
            api_key_verified=SimpleNamespace(openrouter=True),
        )
        self.hub = SimpleNamespace(
            translation_enabled=True,
            llm=object(),
            stt_session_state=lambda channel: f"{channel}-listening",
        )
        self.microphone_test_active = True
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

    def log_detailed(self, message: str, *, level: int) -> None:
        self.events.append(("log-detailed", message, level))

    async def submit_text(self, text: str) -> None:
        self.events.append(("submit", text))

    async def set_translation_enabled(self, enabled: bool) -> bool:
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

    async def apply_settings(self, settings: object) -> None:
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

    async def apply_telemetry_consent(self, consent: str) -> object:
        self.events.append(("telemetry", consent))
        return self.settings

    def persist_settings(self) -> None:
        self.events.append(("persist",))

    def clear_provider_verification(self, provider: str) -> None:
        setattr(self.settings.api_key_verified, provider, False)
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
    assert state.runtime_logging_mode == "detailed"
    assert state.translation_enabled is True
    assert state.stt_state == "self-listening"
    assert state.peer_translation_eula_accepted is False
    assert state.microphone_test_active is True
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
    detached.ui.peer_translation_eula_accepted = True

    assert backend.settings.ui.peer_translation_eula_accepted is False
    backend.settings = None
    assert boundary.state().peer_translation_eula_accepted is None


def test_settings_projection_operations_delegate_without_exposing_the_view() -> None:
    backend = RecordingBackend()
    boundary = UiApplicationBoundary(backend)

    assert boundary.refresh_settings_projection(preserve_custom_vocab_draft=True) is True
    assert boundary.refresh_settings_after_openrouter_pkce_success() is True

    assert backend.events == [
        ("settings-projection", True),
        ("settings-pkce-projection",),
    ]


@pytest.mark.asyncio
async def test_primary_intents_delegate_once_and_preserve_results() -> None:
    backend = RecordingBackend()
    boundary = UiApplicationBoundary(backend)

    await boundary.start()
    await boundary.submit_text("hello")
    result = await boundary.set_translation_enabled(True)
    await boundary.stop()

    assert result is True
    assert backend.events == [
        ("start",),
        ("submit", "hello"),
        ("translation", True),
    ]
    assert boundary.application_lifecycle().is_terminal


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
    boundary = UiApplicationBoundary(backend)

    result = await boundary.accept_peer_translation_eula_and_enable()

    assert result is True
    assert backend.events[0][0] == "settings"
    assert backend.settings.ui.peer_translation_eula_accepted is True
    assert backend.events[1] == ("peer", True)


@pytest.mark.asyncio
async def test_provider_apply_preserves_no_argument_and_forced_rebuild_contracts() -> None:
    backend = RecordingBackend()
    boundary = UiApplicationBoundary(backend)

    await boundary.apply_providers()
    await boundary.apply_providers(force_rebuild_llm=True)
    pending = object()
    await boundary.apply_providers(pending)
    await boundary.apply_providers(
        force_rebuild_llm=True,
        persist_settings=False,
        refresh_ui=False,
    )

    assert backend.events == [
        ("providers", (), {}),
        ("providers", (), {"force_rebuild_llm": True}),
        ("providers", (pending,), {}),
        (
            "providers",
            (None,),
            {
                "force_rebuild_llm": True,
                "persist_settings": False,
                "refresh_ui": False,
            },
        ),
    ]


@pytest.mark.asyncio
async def test_lifecycle_callbacks_diagnostics_and_logging_stay_behind_boundary() -> None:
    backend = RecordingBackend()
    boundary = UiApplicationBoundary(backend)
    diagnostic = object()
    closed: list[str] = []

    boundary.register_application_shutdown_callbacks(
        (
            application_shutdown_callback(
                phase=SHUTDOWN_PHASE_FREEZE_INGRESS,
                owner_name="TestRuntime",
                callback_name="close",
                callback=lambda: closed.append("closed"),
            ),
        )
    )
    assert boundary.emit_application_shutdown_diagnostic(diagnostic) is None
    boundary.log_basic("basic", level=10)
    boundary.log_detailed("detailed", level=20)
    await boundary.stop()

    assert backend.events == [
        ("shutdown-diagnostic", diagnostic),
        ("log-basic", "basic", 10),
        ("log-detailed", "detailed", 20),
    ]
    assert closed == ["closed"]


@pytest.mark.asyncio
async def test_every_user_intent_is_rejected_after_freeze_without_backend_invocation() -> None:
    backend = RecordingBackend()
    boundary = UiApplicationBoundary(backend)
    freeze_started = asyncio.Event()
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

    assert backend.events == []
    release_freeze.set()
    await shutdown_task

    with pytest.raises(ApplicationIntentRejectedError):
        await boundary.submit_text("late")
    assert backend.events == []


@pytest.mark.asyncio
async def test_self_peer_overlay_and_retry_intents_preserve_channel_results() -> None:
    backend = RecordingBackend()
    boundary = UiApplicationBoundary(backend)

    assert await boundary.set_stt_enabled(False) is False
    assert await boundary.set_peer_translation_enabled(True) is True
    assert await boundary.set_overlay_enabled(True) is True
    assert await boundary.retry_peer_process_capture() is True

    assert backend.events == [
        ("self", False),
        ("peer", True),
        ("overlay", True),
        ("peer-retry",),
    ]


@pytest.mark.asyncio
async def test_retry_intent_preserves_false_backend_result_exactly_once() -> None:
    backend = RecordingBackend()
    backend.peer_retry_result = False
    boundary = UiApplicationBoundary(backend)

    assert await boundary.retry_peer_process_capture() is False
    assert backend.events == [("peer-retry",)]


@pytest.mark.asyncio
async def test_local_asr_gpu_install_discovery_failure_and_retry_intents_delegate() -> None:
    backend = RecordingBackend()
    boundary = UiApplicationBoundary(backend)

    await boundary.install_selected_gpu_model_if_needed()
    await boundary.ensure_gpu_device_discovery()
    assert boundary.cycle_debug_capture_fault_profile() == "capture-failure"
    assert boundary.cycle_debug_stt_fault_profile() == "stt-failure"
    boundary.clear_debug_audio_fault_profiles()
    assert boundary.handle_gpu_notice_action("restart") == "retrying"

    assert backend.events == [
        ("gpu-install",),
        ("gpu-discovery",),
        ("capture-fault",),
        ("stt-fault",),
        ("fault-clear",),
        ("gpu-retry", "restart"),
    ]


@pytest.mark.asyncio
async def test_provider_verification_secret_and_managed_auth_transitions_delegate() -> None:
    backend = RecordingBackend()
    boundary = UiApplicationBoundary(backend)

    assert await boundary.verify_api_key("openrouter", "key") == (True, "verified")
    boundary.persist_api_key_verification("openrouter", "key", True)
    assert await boundary.persist_provider_secret_change("llm", "secret") is True
    assert await boundary.start_qq_managed_auth_from_dialog(stage="retry") == "qq-result"
    assert (
        await boundary.start_discord_managed_auth_from_dialog(stage="waiting") == "discord-result"
    )
    assert boundary.supports_discord_managed_auth_reopen() is False
    assert boundary.reopen_discord_managed_auth_browser() is None
    assert boundary.cancel_discord_managed_auth() is None

    assert backend.events == [
        ("verify", "openrouter", "key"),
        ("persist-verification", "openrouter", "key", True),
        ("secret", "llm", "secret"),
        ("qq-auth", {"stage": "retry"}),
        ("discord-auth", {"stage": "waiting"}),
    ]


@pytest.mark.asyncio
async def test_overlay_projection_calibration_apply_cancel_and_reset_delegate() -> None:
    backend = RecordingBackend()
    boundary = UiApplicationBoundary(backend)

    await boundary.set_desktop_overlay_captions_locked(True)
    await boundary.set_desktop_overlay_size_preset("large")
    await boundary.reset_desktop_overlay_position()
    assert boundary.begin_overlay_calibration() == "calibration"
    assert boundary.set_overlay_calibration_field("opacity", 0.8) == 0.8
    assert boundary.apply_overlay_calibration() is True
    assert boundary.cancel_overlay_calibration() is True
    assert boundary.overlay_peer_presentation_state() == OverlayPeerPresentationState(
        overlay_intent_enabled=True,
        overlay_state="connected",
        overlay_failure_reason=None,
        peer_intent_enabled=True,
        peer_effective_enabled=True,
        peer_warning_reason=None,
        peer_activation_starting=False,
    )

    assert backend.events == [
        ("overlay-lock", True),
        ("overlay-size", "large"),
        ("overlay-reset",),
        ("calibration-begin",),
        ("calibration-field", "opacity", 0.8),
        ("calibration-apply",),
        ("calibration-cancel",),
        ("overlay-peer-contract",),
    ]


@pytest.mark.asyncio
async def test_boundary_preserves_settings_failure_and_restart_projection() -> None:
    class FailingBackend(RecordingBackend):
        async def apply_settings(self, settings: object) -> None:
            self.events.append(("settings-failed", settings))
            raise RuntimeError("settings apply failed")

    failing_backend = FailingBackend()
    with pytest.raises(RuntimeError, match="settings apply failed"):
        await UiApplicationBoundary(failing_backend).apply_settings(object())

    restored_backend = RecordingBackend()
    restored_backend.settings.ui.peer_translation_eula_accepted = True
    restored = UiApplicationBoundary(restored_backend)

    assert restored.state().peer_translation_eula_accepted is True
    assert failing_backend.events[0][0] == "settings-failed"


@pytest.mark.asyncio
async def test_github_prompt_open_persistence_accepts_optional_callable_contract() -> None:
    backend = RecordingBackend()
    boundary = UiApplicationBoundary(backend)

    assert await boundary.persist_github_star_prompt_opened() is True
    assert await boundary.persist_github_star_prompt_opened(should_open=lambda: False) is False

    assert backend.events[0] == ("github-open", None)
    assert backend.events[1][0] == "github-open"
    assert callable(backend.events[1][1])
    assert backend.events[1][1]() is False


@pytest.mark.asyncio
async def test_telemetry_and_verification_mutations_stay_behind_named_intents() -> None:
    backend = RecordingBackend()
    boundary = UiApplicationBoundary(backend)

    returned = await boundary.apply_telemetry_consent("allow")
    boundary.clear_provider_verification("openrouter")

    assert returned is backend.settings
    assert backend.settings.api_key_verified.openrouter is False
    assert backend.events == [
        ("telemetry", "allow"),
        ("clear-verification", "openrouter"),
    ]


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
