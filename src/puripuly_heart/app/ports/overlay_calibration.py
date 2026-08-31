from __future__ import annotations

from typing import Protocol

from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext


class OverlayCalibrationRuntimeEffectsPort(Protocol):
    def sync_from_settings(self, settings: AppSettingsVNext | None = None) -> None: ...


__all__ = ["OverlayCalibrationRuntimeEffectsPort"]
