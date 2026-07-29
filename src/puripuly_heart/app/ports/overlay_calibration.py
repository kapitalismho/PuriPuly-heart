from __future__ import annotations

from typing import Protocol


class OverlayCalibrationRuntimeEffectsPort(Protocol):
    def sync_from_settings(self, settings: object | None = None) -> None: ...


__all__ = ["OverlayCalibrationRuntimeEffectsPort"]
