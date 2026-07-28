from __future__ import annotations

import logging
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from puripuly_heart.app.ports.ui_presentation import UiPresentationPort
from puripuly_heart.app.services.application_shutdown import (
    ApplicationShutdownContext,
    ApplicationShutdownDiagnostic,
)


@dataclass(slots=True)
class ApplicationRuntimeLoggingOwner:
    presentation: UiPresentationPort
    service_factory: Callable[[], Any]
    fallback_logger: logging.Logger
    _service: Any | None = field(init=False, default=None, repr=False)

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
