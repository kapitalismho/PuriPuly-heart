from __future__ import annotations

from pathlib import Path

from puripuly_heart.ui.controller import GuiController


class ControllerUiApplicationStateAdapter:
    def __init__(self, runtime: GuiController) -> None:
        self._runtime = runtime

    @property
    def config_path(self) -> Path:
        return self._runtime.config_path

    @property
    def compatibility_settings(self) -> object | None:
        return self._runtime.settings

    @property
    def translation_enabled(self) -> bool:
        hub = self._runtime.hub
        return hub.translation_enabled if hub is not None else False

    @property
    def translation_runtime_ready(self) -> bool | None:
        hub = self._runtime.hub
        return hub.llm is not None if hub is not None else None

    @property
    def stt_state(self) -> object | None:
        hub = self._runtime.hub
        return hub.stt_session_state("self") if hub is not None else None

    @property
    def peer_translation_eula_accepted(self) -> bool | None:
        settings = self._runtime.settings
        return bool(settings.ui.peer_translation_eula_accepted) if settings is not None else None

    @property
    def microphone_test_active(self) -> bool:
        return self._runtime.microphone_test_active

    @property
    def provider_name(self) -> str | None:
        settings = self._runtime.settings
        return settings.provider.llm.value if settings is not None else None

    @property
    def overlay_target(self) -> str | None:
        settings = self._runtime.settings
        return str(settings.overlay.target) if settings is not None else None

    @property
    def desktop_overlay_captions_locked(self) -> bool:
        return self._runtime.desktop_overlay_captions_locked

    @property
    def managed_auth_referral_bonus_applied(self) -> bool:
        return self._runtime.last_discord_managed_auth_referral_bonus_applied is True

    @property
    def overlay_calibration(self) -> object:
        return self._runtime.overlay_calibration
