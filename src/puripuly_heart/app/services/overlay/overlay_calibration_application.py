from __future__ import annotations

import copy
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from puripuly_heart.app.services.canonical_settings_persistence import SettingsOwner
from puripuly_heart.app.services.settings_application import SettingsApplicationOwner
from puripuly_heart.config.overlay_calibration import OverlayCalibration

from .overlay_application import OverlayApplicationOwner
from .overlay_calibration import OverlayCalibrationOwner


@dataclass(slots=True)
class OverlayCalibrationApplicationOwner:
    settings: SettingsOwner
    settings_application_provider: Callable[[], SettingsApplicationOwner]
    overlay_provider: Callable[[], OverlayApplicationOwner]
    schedule_task: Callable[[Callable[[], Awaitable[None]]], bool]
    log_detailed: Callable[..., object]
    ingress_available: Callable[[], bool]
    _owner: OverlayCalibrationOwner = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._owner = OverlayCalibrationOwner(
            schedule_task=self.schedule_task,
            persist=self._persist,
            emit=self._emit,
            can_persist=lambda: self.settings.current is not None,
            can_emit=lambda: (
                self.ingress_available() and self.overlay_provider().current_presenter() is not None
            ),
            log_detailed=self.log_detailed,
        )

    @property
    def current(self) -> OverlayCalibration:
        return self._owner.current

    @property
    def draft(self) -> OverlayCalibration | None:
        return self._owner.draft

    def replace_current(self, calibration: OverlayCalibration) -> None:
        self._owner.replace_current(calibration)

    def replace_draft(self, calibration: OverlayCalibration | None) -> None:
        self._owner.replace_draft(calibration)

    def begin(self) -> OverlayCalibration:
        return self._owner.begin()

    def set_field(self, field_name: str, value: object) -> OverlayCalibration:
        return self._owner.set_field(field_name, value)

    def apply(self) -> OverlayCalibration:
        return self._owner.apply()

    def cancel(self) -> OverlayCalibration:
        return self._owner.cancel()

    def schedule_emit(self) -> None:
        self._owner.schedule_emit()

    async def emit_current(self) -> None:
        await self._owner.emit_current()

    def sync_from_settings(self, settings: object | None = None) -> None:
        resolved = settings if settings is not None else self.settings.current
        calibration = getattr(getattr(resolved, "overlay", None), "calibration", None)
        if isinstance(calibration, OverlayCalibration):
            self._owner.replace_current(calibration)

    async def _persist(self, calibration: OverlayCalibration) -> None:
        current = self.settings.current
        if current is None:
            return
        updated = copy.deepcopy(current)
        updated.overlay.calibration = calibration.copy()
        await self.settings_application_provider().apply_overlay_osc_output(updated)

    async def _emit(self, calibration: OverlayCalibration) -> None:
        presenter = self.overlay_provider().current_presenter()
        if presenter is not None:
            await presenter.update_calibration(calibration.copy())


__all__ = ["OverlayCalibrationApplicationOwner"]
