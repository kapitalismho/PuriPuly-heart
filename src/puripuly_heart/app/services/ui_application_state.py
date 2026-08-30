from __future__ import annotations

import copy

from puripuly_heart.app.ports.application_runtime_logging import (
    ApplicationRuntimeLoggingPort,
)
from puripuly_heart.app.ports.ui_application import UiApplicationState
from puripuly_heart.app.ports.ui_application_state import (
    UiApplicationStateRuntimePort,
)
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext


class UiApplicationStateOwner:
    def __init__(
        self,
        runtime: UiApplicationStateRuntimePort,
        *,
        runtime_logging: ApplicationRuntimeLoggingPort,
    ) -> None:
        self._runtime = runtime
        self._runtime_logging = runtime_logging

    def snapshot(self) -> UiApplicationState:
        runtime = self._runtime
        return UiApplicationState(
            config_path=runtime.config_path,
            runtime_logging_mode=self._runtime_logging.mode,
            translation_enabled=runtime.translation_enabled,
            stt_state=runtime.stt_state,
            peer_translation_eula_accepted=runtime.peer_translation_eula_accepted,
            microphone_test_active=runtime.microphone_test_active,
            provider_name=runtime.provider_name,
            overlay_target=runtime.overlay_target,
            desktop_overlay_captions_locked=runtime.desktop_overlay_captions_locked,
            managed_auth_referral_bonus_applied=(runtime.managed_auth_referral_bonus_applied),
            translation_runtime_ready=runtime.translation_runtime_ready,
        )

    def compatibility_settings(self) -> AppSettingsVNext | None:
        settings = self._runtime.compatibility_settings
        return copy.deepcopy(settings) if settings is not None else None

    def overlay_calibration(self) -> object:
        return copy.deepcopy(self._runtime.overlay_calibration)
