from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from puripuly_heart.config.overlay_calibration import OverlayCalibration


@dataclass(slots=True)
class OverlayCalibrationOwner:
    schedule_task: Callable[[Callable[[], Awaitable[None]]], bool]
    persist: Callable[[OverlayCalibration], Awaitable[None]]
    emit: Callable[[OverlayCalibration], Awaitable[None]]
    can_persist: Callable[[], bool]
    can_emit: Callable[[], bool]
    log_detailed: Callable[..., object]
    _current: OverlayCalibration = field(
        init=False,
        default_factory=OverlayCalibration,
        repr=False,
    )
    _draft: OverlayCalibration | None = field(init=False, default=None, repr=False)

    @property
    def current(self) -> OverlayCalibration:
        return self._current

    @property
    def draft(self) -> OverlayCalibration | None:
        return self._draft

    def replace_current(self, calibration: OverlayCalibration) -> None:
        self._current = calibration.copy()

    def replace_draft(self, calibration: OverlayCalibration | None) -> None:
        self._draft = calibration.copy() if calibration is not None else None

    def begin(self) -> OverlayCalibration:
        if self._draft is None:
            self._draft = self._current.copy()
        return self._draft.copy()

    def set_field(self, field_name: str, value: object) -> OverlayCalibration:
        if self._draft is None:
            self._draft = self._current.copy()
        if field_name not in OverlayCalibration.__dataclass_fields__:
            raise ValueError(f"unknown overlay calibration field: {field_name}")
        if field_name == "anchor":
            setattr(self._draft, field_name, str(value))
        else:
            setattr(self._draft, field_name, float(value))
        self._draft.validate()
        return self._draft.copy()

    def apply(self) -> OverlayCalibration:
        if self._draft is None:
            return self._current.copy()
        self._draft.validate()
        self._current = self._draft.copy()
        self._draft = None
        self.schedule_persistence(self._current.copy())
        self.schedule_emit()
        return self._current.copy()

    def cancel(self) -> OverlayCalibration:
        self._draft = None
        return self._current.copy()

    async def persist_calibration(self, calibration: OverlayCalibration) -> None:
        await self.persist(calibration.copy())

    def schedule_persistence(self, calibration: OverlayCalibration) -> None:
        if not self.can_persist():
            return
        scheduled_calibration = calibration.copy()

        async def task() -> None:
            await self.persist_calibration(scheduled_calibration)

        try:
            if self.schedule_task(task):
                return
        except Exception:
            self.log_detailed(
                "[Overlay] Calibration persistence skipped reason=page_run_task_failed",
                level=logging.WARNING,
            )
            return
        self.log_detailed(
            "[Overlay] Calibration persistence skipped reason=page_run_task_unavailable",
            level=logging.WARNING,
        )

    async def emit_current(self) -> None:
        if not self.can_emit():
            return
        try:
            await self.emit(self._current.copy())
        except Exception:
            return

    def schedule_emit(self) -> None:
        if not self.can_emit():
            return
        try:
            if self.schedule_task(self.emit_current):
                return
        except Exception as exc:
            self.log_detailed(
                "[Overlay] Failed to schedule calibration update via page.run_task",
                level=logging.WARNING,
                exception=exc,
            )
            return
        self.log_detailed(
            "[Overlay] Skipping calibration update; page.run_task unavailable",
            level=logging.WARNING,
        )
