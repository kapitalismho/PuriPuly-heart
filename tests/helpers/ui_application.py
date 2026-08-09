from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import cast

from puripuly_heart.app.ports.application_runtime_logging import (
    ApplicationRuntimeLoggingPort,
)
from puripuly_heart.app.ports.application_runtime_shutdown import (
    ApplicationRuntimeShutdownPort,
)
from puripuly_heart.app.ports.ui_application_intents import (
    UiDiagnosticsRuntimePort,
    UiEngagementRuntimePort,
    UiInputRuntimePort,
    UiManagedRuntimePort,
    UiMicrophoneRuntimePort,
    UiOverlayRuntimePort,
    UiPeerCaptureRuntimePort,
    UiProviderRuntimePort,
    UiSettingsRuntimePort,
)
from puripuly_heart.app.ports.ui_application_state import (
    UiApplicationStateRuntimePort,
)
from puripuly_heart.app.services.application_shutdown import (
    ApplicationShutdownContext,
    ApplicationShutdownDiagnostic,
)
from puripuly_heart.app.services.application_startup import ApplicationStartupOwner
from puripuly_heart.app.services.ui_application import UiApplicationBoundary
from puripuly_heart.app.services.ui_application_state import UiApplicationStateOwner


class ApplicationRuntimeLoggingStub:
    def __init__(self, backend: object) -> None:
        self._backend = backend

    @property
    def mode(self) -> str:
        return str(getattr(self._backend, "runtime_logging_mode", "basic"))

    def emit_basic(self, message: str, *, level: int = logging.INFO) -> None:
        sink = getattr(self._backend, "log_basic", None)
        if callable(sink):
            sink(message, level=level)

    def emit_detailed(
        self,
        message: str,
        *,
        level: int = logging.INFO,
        exception: BaseException | None = None,
    ) -> bool:
        sink = getattr(self._backend, "log_detailed", None)
        if callable(sink):
            sink(message, level=level)
        return exception is not None


class UiApplicationRuntimeStub:
    def __init__(self, backend: object) -> None:
        self._backend = backend

    def __getattr__(self, name: str) -> object:
        return getattr(self._backend, name)

    def capture_settings_view_change(self, settings: object) -> object:
        capture = getattr(self._backend, "capture_settings_view_change", None)
        return capture(settings) if callable(capture) else settings

    def merge_settings_view_change_with_current(self, captured: object) -> object:
        merge = getattr(self._backend, "merge_settings_view_change_with_current", None)
        return merge(captured) if callable(merge) else captured

    def refresh_settings_projection(
        self,
        *,
        preserve_custom_vocab_draft: bool = False,
    ) -> bool:
        refresh = getattr(self._backend, "refresh_settings_projection", None)
        return (
            bool(
                refresh(
                    preserve_custom_vocab_draft=preserve_custom_vocab_draft,
                )
            )
            if callable(refresh)
            else False
        )

    def refresh_settings_after_openrouter_pkce_success(self) -> bool:
        refresh = getattr(
            self._backend,
            "refresh_settings_after_openrouter_pkce_success",
            None,
        )
        return bool(refresh()) if callable(refresh) else False

    async def stop_microphone_test(self) -> None:
        stop = getattr(self._backend, "stop_microphone_test", None)
        if not callable(stop):
            return
        result = stop()
        if hasattr(result, "__await__"):
            await result

    def overlay_peer_presentation_state(self) -> object | None:
        state = getattr(self._backend, "overlay_peer_presentation_state", None)
        return state() if callable(state) else None

    def dashboard_managed_auth_action(self) -> str:
        action = getattr(self._backend, "dashboard_managed_auth_action", None)
        return str(action()) if callable(action) else "continue"

    def dashboard_managed_auth_prompt_kind(self) -> str:
        prompt = getattr(self._backend, "dashboard_managed_auth_prompt_kind", None)
        return str(prompt()) if callable(prompt) else "discord"

    async def apply_telemetry_consent(self, consent: str) -> object | None:
        apply = getattr(self._backend, "apply_telemetry_consent", None)
        return await apply(consent) if callable(apply) else None

    def reopen_openrouter_pkce_authorization_url(self) -> object:
        reopen = getattr(
            self._backend,
            "reopen_openrouter_pkce_authorization_url",
            None,
        )
        return reopen() if callable(reopen) else None

    def clear_provider_verification(self, provider: str) -> None:
        clear = getattr(self._backend, "clear_provider_verification", None)
        if callable(clear):
            clear(provider)

    async def record_telemetry_translation_success_day(self) -> None:
        record = getattr(
            self._backend,
            "record_telemetry_translation_success_day",
            None,
        )
        if callable(record):
            await record()

    def should_show_github_star_prompt(self) -> bool:
        should_show = getattr(self._backend, "should_show_github_star_prompt", None)
        return bool(should_show()) if callable(should_show) else False

    async def persist_github_star_prompt_eligible_launch(self) -> bool:
        persist = getattr(
            self._backend,
            "persist_github_star_prompt_eligible_launch",
            None,
        )
        return bool(await persist()) if callable(persist) else False

    async def refresh_openrouter_usage_after_launch(self) -> bool:
        refresh = getattr(self._backend, "refresh_openrouter_usage_after_launch", None)
        return bool(await refresh()) if callable(refresh) else False

    async def prepare_runtime_after_launch(self) -> None:
        prepare = getattr(self._backend, "prepare_runtime_after_launch", None)
        if callable(prepare):
            await prepare()

    async def persist_github_star_prompt_opened(
        self,
        *,
        should_open=None,
    ) -> bool:
        persist = getattr(self._backend, "persist_github_star_prompt_opened", None)
        return bool(await persist(should_open=should_open)) if callable(persist) else False

    async def persist_github_star_prompt_clicked(self) -> None:
        persist = getattr(self._backend, "persist_github_star_prompt_clicked", None)
        if callable(persist):
            await persist()


class UiApplicationStateRuntimeStub:
    def __init__(self, backend: object) -> None:
        self._backend = backend

    @property
    def config_path(self) -> Path:
        return Path(getattr(self._backend, "config_path", "settings.json"))

    @property
    def compatibility_settings(self) -> object | None:
        return getattr(self._backend, "settings", None)

    @property
    def translation_enabled(self) -> bool:
        hub = getattr(self._backend, "hub", None)
        return bool(getattr(hub, "translation_enabled", False))

    @property
    def translation_runtime_ready(self) -> bool | None:
        hub = getattr(self._backend, "hub", None)
        return getattr(hub, "llm", None) is not None if hub is not None else None

    @property
    def stt_state(self) -> object | None:
        hub = getattr(self._backend, "hub", None)
        state = getattr(hub, "stt_session_state", None)
        return state("self") if callable(state) else None

    @property
    def peer_translation_eula_accepted(self) -> bool | None:
        settings = self.compatibility_settings
        ui = getattr(settings, "ui", None)
        return (
            bool(getattr(ui, "peer_translation_eula_accepted", False)) if ui is not None else None
        )

    @property
    def microphone_test_active(self) -> bool:
        return bool(getattr(self._backend, "microphone_test_active", False))

    @property
    def provider_name(self) -> str | None:
        settings = self.compatibility_settings
        provider = getattr(getattr(settings, "provider", None), "llm", None)
        value = getattr(provider, "value", provider)
        return str(value) if value is not None else None

    @property
    def overlay_target(self) -> str | None:
        settings = self.compatibility_settings
        target = getattr(getattr(settings, "overlay", None), "target", None)
        value = getattr(target, "value", target)
        return str(value) if value is not None else None

    @property
    def desktop_overlay_captions_locked(self) -> bool:
        return bool(getattr(self._backend, "desktop_overlay_captions_locked", False))

    @property
    def managed_auth_referral_bonus_applied(self) -> bool:
        return (
            getattr(
                self._backend,
                "last_discord_managed_auth_referral_bonus_applied",
                False,
            )
            is True
        )

    @property
    def overlay_calibration(self) -> object:
        return getattr(self._backend, "overlay_calibration", None)


class ApplicationRuntimeShutdownStub:
    def __init__(self, backend: object) -> None:
        self._backend = backend

    def freeze_application_ingress(self) -> None:
        return None

    def stop_github_star_prompt_ingress(self) -> None:
        return None

    async def release_manual_typing(self) -> None:
        return None

    async def close_clipboard_runtime(self) -> None:
        return None

    async def cancel_vrchat_osc_presence_probe(self) -> None:
        return None

    async def stop_self_capture_ingress(self) -> None:
        return None

    async def close_vrc_mic_receiver_runtime(self) -> None:
        return None

    async def close_overlay_runtime(self) -> None:
        return None

    async def close_peer_runtime(self) -> None:
        return None

    async def close_github_star_prompt_owner(self) -> None:
        return None

    async def close_openrouter_oauth_runtime(self) -> None:
        return None

    async def close_local_asr_provisioning(self) -> None:
        return None

    async def close_microphone_test_runtime(self) -> None:
        return None

    async def close_self_capture_owner(self) -> None:
        return None

    async def close_runtime_logging_background_tasks(self) -> None:
        return None

    async def close_managed_auth_owner(self) -> None:
        return None

    async def close_translation_enable_owner(self) -> None:
        return None

    async def close_managed_usage_owner(self) -> None:
        return None

    async def close_runtime_pipeline_launcher(self) -> None:
        return None

    async def close_peer_capture_owner(self) -> None:
        return None

    async def close_self_translation_ingress(self) -> None:
        return None

    async def close_peer_translation_ingress(self) -> None:
        return None

    async def close_translation_turns(self) -> None:
        return None

    async def close_output_runtime(self) -> None:
        return None

    async def close_self_channel_runtime(self) -> None:
        return None

    async def close_peer_channel_runtime(self) -> None:
        return None

    async def close_local_asr_runtime(self) -> None:
        return None

    async def close_llm_runtime(self) -> None:
        return None

    def close_vrchat_sender(self) -> None:
        return None

    async def close_managed_openrouter_release_service(self) -> None:
        return None

    def emit_final_application_shutdown_diagnostics(
        self,
        context: ApplicationShutdownContext,
    ) -> None:
        return None

    def close_runtime_logging(
        self,
        context: ApplicationShutdownContext,
    ) -> None:
        return None

    def emit_application_shutdown_diagnostic(
        self,
        diagnostic: ApplicationShutdownDiagnostic,
    ) -> None:
        sink = getattr(self._backend, "emit_application_shutdown_diagnostic", None)
        if callable(sink):
            sink(diagnostic)


def compose_test_ui_application_boundary(
    backend: object,
    *,
    runtime_shutdown: ApplicationRuntimeShutdownPort | None = None,
    runtime_logging: ApplicationRuntimeLoggingPort | None = None,
    osc_state_publisher: Callable[[], object] | None = None,
) -> UiApplicationBoundary:
    logging_port = runtime_logging or ApplicationRuntimeLoggingStub(backend)
    state_runtime = cast(
        UiApplicationStateRuntimePort,
        UiApplicationStateRuntimeStub(backend),
    )
    runtime = UiApplicationRuntimeStub(backend)
    return UiApplicationBoundary(
        startup=cast(ApplicationStartupOwner, runtime),
        input_runtime=cast(UiInputRuntimePort, runtime),
        peer_capture=cast(UiPeerCaptureRuntimePort, runtime),
        settings=cast(UiSettingsRuntimePort, runtime),
        provider=cast(UiProviderRuntimePort, runtime),
        microphone=cast(UiMicrophoneRuntimePort, runtime),
        overlay=cast(UiOverlayRuntimePort, runtime),
        managed=cast(UiManagedRuntimePort, runtime),
        engagement=cast(UiEngagementRuntimePort, runtime),
        diagnostics=cast(UiDiagnosticsRuntimePort, runtime),
        state=UiApplicationStateOwner(
            state_runtime,
            runtime_logging=logging_port,
        ),
        runtime_shutdown=runtime_shutdown or ApplicationRuntimeShutdownStub(backend),
        runtime_logging=logging_port,
        osc_state_publisher=osc_state_publisher,
    )
