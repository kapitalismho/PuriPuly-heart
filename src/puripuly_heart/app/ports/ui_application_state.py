from __future__ import annotations

from pathlib import Path
from typing import Protocol

from puripuly_heart.app.ports.ui_application import UiApplicationState
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext


class UiApplicationStateRuntimePort(Protocol):
    @property
    def config_path(self) -> Path: ...

    @property
    def compatibility_settings(self) -> AppSettingsVNext | None: ...

    @property
    def translation_enabled(self) -> bool: ...

    @property
    def translation_runtime_ready(self) -> bool | None: ...

    @property
    def stt_state(self) -> object | None: ...

    @property
    def peer_translation_eula_accepted(self) -> bool | None: ...

    @property
    def microphone_test_active(self) -> bool: ...

    @property
    def provider_name(self) -> str | None: ...

    @property
    def overlay_target(self) -> str | None: ...

    @property
    def desktop_overlay_captions_locked(self) -> bool: ...

    @property
    def managed_auth_referral_bonus_applied(self) -> bool: ...

    @property
    def overlay_calibration(self) -> object: ...


class UiApplicationStatePort(Protocol):
    def snapshot(self) -> UiApplicationState: ...

    def compatibility_settings(self) -> AppSettingsVNext | None: ...

    def overlay_calibration(self) -> object: ...
