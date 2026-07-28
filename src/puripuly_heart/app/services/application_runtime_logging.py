from __future__ import annotations

import asyncio
import logging
import traceback
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from puripuly_heart.app.ports.ui_presentation import UiPresentationPort
from puripuly_heart.app.services.application_shutdown import (
    ApplicationShutdownContext,
    ApplicationShutdownDiagnostic,
)
from puripuly_heart.core.lifecycle import LifecycleScope, start_lifecycle_task


def _runtime_logging_task_scope() -> LifecycleScope:
    return LifecycleScope("ApplicationRuntimeLoggingOwner")


def _callback_unavailable() -> bool:
    return False


@dataclass(slots=True)
class ApplicationRuntimeLoggingOwner:
    presentation: UiPresentationPort
    service_factory: Callable[[], Any]
    fallback_logger: logging.Logger
    overlay_logging_mode_update: Callable[[], Awaitable[None]] | None = None
    overlay_logging_mode_update_available: Callable[[], bool] = _callback_unavailable
    _service: Any | None = field(init=False, default=None, repr=False)
    _task_scope: LifecycleScope = field(
        init=False,
        default_factory=_runtime_logging_task_scope,
        repr=False,
    )
    _task_sequence: int = field(init=False, default=0, repr=False)
    _ingress_stopped: bool = field(init=False, default=False, repr=False)

    @property
    def service(self) -> Any:
        if self._service is None:
            self._service = self.service_factory()
        self.presentation.attach_runtime_log_sink(self._service)
        return self._service

    @property
    def installed_service(self) -> Any | None:
        return self._service

    def install_service(self, service: Any | None) -> None:
        self._service = service

    @property
    def mode(self) -> str:
        return str(self.service.mode.value)

    def set_mode(
        self,
        mode: object,
        *,
        detailed_enabled: Callable[[], None],
        mode_changed: Callable[[str], None],
    ) -> None:
        service = self.service
        previous_mode = service.mode
        service.set_mode(mode)
        normalized_mode = str(service.mode.value)
        previous_mode_value = str(getattr(previous_mode, "value", previous_mode))
        if previous_mode_value != "detailed" and normalized_mode == "detailed":
            detailed_enabled()
        mode_changed(normalized_mode)

    @property
    def active_task_names(self) -> tuple[str, ...]:
        return self._task_scope.active_task_names

    def schedule_audio_environment_snapshot(self) -> None:
        if self._ingress_stopped:
            return
        try:
            if self.presentation.schedule_task(self.log_audio_environment_snapshot):
                return
        except Exception as exc:
            self.emit_detailed(
                "[AudioDiag][Snapshot] failed to schedule via page.run_task",
                level=logging.WARNING,
                exception=exc,
            )
        self._start_owned_fallback_task(
            self.log_audio_environment_snapshot,
            task_prefix="audio-environment-snapshot",
            no_loop_message="[AudioDiag][Snapshot] skipped reason=no_running_loop",
            create_failure_message="[AudioDiag][Snapshot] skipped reason=create_task_failed",
        )

    async def log_audio_environment_snapshot(self) -> None:
        from puripuly_heart.core.audio.diagnostics import (
            collect_pyaudiowpatch_snapshot_lines,
            collect_sounddevice_snapshot_lines,
        )

        sounddevice_lines, loopback_lines = await asyncio.gather(
            asyncio.to_thread(collect_sounddevice_snapshot_lines),
            asyncio.to_thread(collect_pyaudiowpatch_snapshot_lines),
        )
        for line in sounddevice_lines:
            self.emit_detailed(line)
        for line in loopback_lines:
            self.emit_detailed(line)

    def schedule_overlay_logging_mode_update(self) -> None:
        if self._ingress_stopped or not self.overlay_logging_mode_update_available():
            return
        update = self.overlay_logging_mode_update
        if update is None:
            return
        try:
            if self.presentation.schedule_task(update):
                return
        except Exception as exc:
            self.emit_detailed(
                "[Overlay] Failed to schedule logging mode update via page.run_task",
                level=logging.WARNING,
                exception=exc,
            )
            return
        self._start_owned_fallback_task(
            update,
            task_prefix="overlay-logging-mode-update",
            no_loop_message=(
                "[Overlay] Skipping logging mode update; "
                "no running loop and page.run_task unavailable"
            ),
            create_failure_message="[Overlay] Skipping logging mode update; create_task failed",
        )

    def stop_ingress(self) -> None:
        self._ingress_stopped = True

    async def close_background_tasks(self) -> None:
        self.stop_ingress()
        await self._task_scope.close()

    def _start_owned_fallback_task(
        self,
        task_factory: Callable[[], Awaitable[None]],
        *,
        task_prefix: str,
        no_loop_message: str,
        create_failure_message: str,
    ) -> None:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            self.emit_detailed(no_loop_message, level=logging.WARNING)
            return

        async def run() -> None:
            try:
                await task_factory()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.emit_detailed(
                    f"[Lifecycle][ApplicationRuntimeLoggingOwner] task_failed task={task_prefix}",
                    level=logging.WARNING,
                    exception=exc,
                )

        self._task_sequence += 1
        coroutine = run()
        try:
            start_lifecycle_task(
                self._task_scope,
                coroutine,
                name=f"{task_prefix}-{self._task_sequence}",
            )
        except Exception as exc:
            coroutine.close()
            self.emit_detailed(
                create_failure_message,
                level=logging.WARNING,
                exception=exc,
            )

    def emit_basic(self, message: str, *, level: int = logging.INFO) -> None:
        try:
            self.service.emit_basic(message, level=level)
        except Exception:
            self.fallback_logger.log(level, message)

    def emit_detailed(
        self,
        message: str,
        *,
        level: int = logging.INFO,
        exception: BaseException | None = None,
    ) -> bool:
        rendered_message = message
        exc_info = None
        if exception is not None:
            exc_info = (type(exception), exception, exception.__traceback__)
            rendered_message = (
                f"{message}\n{''.join(traceback.format_exception(*exc_info)).rstrip()}"
            )
        try:
            return bool(self.service.emit_detailed(rendered_message, level=level))
        except Exception:
            self.fallback_logger.log(level, message, exc_info=exc_info)
            return True

    def emit_detailed_lazy(
        self,
        build_message: Callable[[], str],
        *,
        level: int = logging.INFO,
        exception: BaseException | None = None,
    ) -> bool:
        exc_info = None
        if exception is not None:
            exc_info = (type(exception), exception, exception.__traceback__)

        def render_message() -> str:
            rendered_message = build_message()
            if exc_info is None:
                return rendered_message
            return f"{rendered_message}\n{''.join(traceback.format_exception(*exc_info)).rstrip()}"

        try:
            return bool(self.service.emit_detailed_lazy(render_message, level=level))
        except Exception:
            self.fallback_logger.log(level, build_message(), exc_info=exc_info)
            return True

    def emit_terminal_summary(self, context: ApplicationShutdownContext) -> None:
        service = self._service
        if service is None:
            return
        emit_persisted = getattr(service, "emit_persisted", None)
        if not callable(emit_persisted):
            return
        emit_persisted(
            "[Lifecycle][Shutdown] coordinator_terminal "
            "owner=ApplicationShutdownCoordinator "
            f"failure_count={len(context.failures)}",
            level=logging.INFO,
        )

    def close_after_producers_stop(self, context: ApplicationShutdownContext) -> None:
        service = self._service
        if service is None:
            return
        service.close_after_producers_stop(
            cleanup_failures=context.cleanup_exceptions,
        )

    def emit_shutdown_diagnostic(
        self,
        diagnostic: ApplicationShutdownDiagnostic,
    ) -> None:
        message = (
            "[Lifecycle][Shutdown] callback_failed "
            f"phase={diagnostic.phase} "
            f"owner={diagnostic.owner_name} "
            f"callback={diagnostic.callback_name} "
            f"exception_class={diagnostic.exception_class} "
            f"timed_out={str(diagnostic.timed_out).lower()}"
        )
        service = self._service
        if service is not None:
            emit_persisted = getattr(service, "emit_persisted", None)
            if callable(emit_persisted):
                emit_persisted(message, level=logging.ERROR)
                return
        self.fallback_logger.error(message)
