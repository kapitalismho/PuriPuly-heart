"""UI-independent telemetry toggle and app-active-day reporting flows."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import Any

APP_ACTIVE_DAY_RETRY_DELAY_S = 60.0


class TelemetryReportingOwner:
    """Owns telemetry-enable flow and app-active-day reporting semantics.

    Ports are injected callables so the flows are testable without a UI:
    - ``application``: apply_telemetry_enabled / settings_general_snapshot /
      record_app_active_day
    - ``run_background``: schedules a coroutine factory as a page-owned task
    - ``queue_settings_mutation``: enqueues a task factory onto the ordered
      settings-mutation queue (single worker, no reentrancy)
    - ``is_shutting_down``: returns True once shutdown ingress froze
    - ``sync_telemetry``: optional view callback for snapshot sync
    - ``clock``/``sleep``: time control (defaults: UTC now / asyncio.sleep)
    """

    def __init__(
        self,
        application: Any,
        *,
        run_background: Callable[..., object],
        queue_settings_mutation: Callable[[Callable[[], Awaitable[None]]], None],
        is_shutting_down: Callable[[], bool] | None = None,
        sync_telemetry: Callable[[Any], None] | None = None,
        clock: Callable[[], datetime] | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        self._application = application
        self._run_background = run_background
        self._queue_settings_mutation = queue_settings_mutation
        self._is_shutting_down = is_shutting_down or (lambda: False)
        self._sync_telemetry = sync_telemetry
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._sleep = sleep or asyncio.sleep
        self._retry_task_handle: object | None = None

    def apply_telemetry_enabled(self, enabled: bool) -> None:
        if not isinstance(enabled, bool):
            return

        async def _task() -> None:
            await self._application.apply_telemetry_enabled(enabled)
            settings = self._application.settings_general_snapshot()
            if self._sync_telemetry is not None and settings is not None:
                self._sync_telemetry(settings)

        self._run_background(_task)

    def schedule_app_active_day_report(self) -> None:
        active_date_utc = self._clock().date().isoformat()

        async def _send() -> None:
            result = await self._application.record_app_active_day(active_date_utc)
            if getattr(result, "status", None) == "send_failed":
                self._schedule_app_active_day_retry(active_date_utc)

        self._queue_settings_mutation(_send)

    def _schedule_app_active_day_retry(self, active_date_utc: str) -> None:
        existing = self._retry_task_handle
        still_running = getattr(existing, "done", None)
        if existing is not None and callable(still_running) and not still_running():
            return
        if self._is_shutting_down():
            return

        handle = None

        async def _retry_after_delay() -> None:
            try:
                await self._sleep(APP_ACTIVE_DAY_RETRY_DELAY_S)
                if self._is_shutting_down():
                    return
                if self._clock().date().isoformat() != active_date_utc:
                    return

                async def _retry() -> None:
                    if self._is_shutting_down():
                        return
                    if self._clock().date().isoformat() != active_date_utc:
                        return
                    await self._application.record_app_active_day(active_date_utc)

                self._queue_settings_mutation(_retry)
            finally:
                if self._retry_task_handle is handle:
                    self._retry_task_handle = None

        handle = self._run_background(_retry_after_delay)
        self._retry_task_handle = handle

    async def cancel_retry(self) -> None:
        handle = self._retry_task_handle
        if handle is None:
            return
        still_running = getattr(handle, "done", None)
        cancel = getattr(handle, "cancel", None)
        if callable(still_running) and not still_running() and callable(cancel):
            cancel()
        try:
            await asyncio.gather(handle, return_exceptions=True)
        finally:
            if self._retry_task_handle is handle:
                self._retry_task_handle = None
