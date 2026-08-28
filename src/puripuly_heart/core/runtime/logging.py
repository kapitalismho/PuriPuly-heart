from __future__ import annotations

import hashlib
import logging
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol

from puripuly_heart.core.observability import RealtimeLogSink, SessionLoggingMode


class RuntimeLoggingAdapterPort(Protocol):
    mode: SessionLoggingMode
    log_file: Path

    def set_mode(self, mode: SessionLoggingMode | str) -> None: ...

    def attach_realtime_sink(self, sink: RealtimeLogSink) -> None: ...

    def detach_realtime_sink(self) -> None: ...

    def emit_basic(self, message: str, *, level: int = logging.INFO) -> None: ...

    def emit_detailed(self, message: str, *, level: int = logging.INFO) -> bool: ...

    def emit_detailed_lazy(
        self,
        build_message: Callable[[], str],
        *,
        level: int = logging.INFO,
    ) -> bool: ...

    def emit_persisted(self, message: str, *, level: int = logging.INFO) -> None: ...

    def close_terminal_owner(self) -> None: ...


class RuntimeLoggingCloseError(RuntimeError):
    """Bounded terminal logging close failure summary."""

    def __init__(self, failures: Sequence[BaseException]) -> None:
        self.failure_types = tuple(type(failure).__name__ for failure in failures)
        self._failure_count = len(self.failure_types)
        failure_type_text = ",".join(self.failure_types) if self.failure_types else "none"
        super().__init__(
            "runtime logging shutdown failed "
            f"failure_count={self._failure_count} failure_types={failure_type_text}"
        )

    @property
    def failure_count(self) -> int:
        return self._failure_count


class RuntimeLoggingService:
    """Lifecycle owner that keeps runtime logging open until terminal shutdown."""

    resource_fields = (
        "live log sink",
        "queue listener",
        "file handlers",
        "persisted diagnostics",
    )
    stop_ingress = "stop after producers have stopped"
    shutdown_policy = "flush final shutdown summary, then close handlers"
    late_callback_rule = "late logs after close go to bounded fallback/stderr only"

    def __init__(
        self,
        *,
        session_service: RuntimeLoggingAdapterPort | None = None,
        session_factory: Callable[[], RuntimeLoggingAdapterPort] | None = None,
        fallback_logger: logging.Logger | None = None,
    ) -> None:
        if session_service is not None and session_factory is not None:
            raise ValueError("Provide either session_service or session_factory, not both")
        if session_service is None and session_factory is None:
            raise ValueError("A runtime logging adapter or factory is required")
        self._session = session_service if session_service is not None else session_factory()
        self._fallback_logger = fallback_logger
        self._closed = False
        self._mode = self._session.mode

    @property
    def owner_name(self) -> str:
        return "RuntimeLoggingService"

    @property
    def is_closed(self) -> bool:
        return self._closed

    @property
    def mode(self) -> SessionLoggingMode:
        if not self._closed:
            self._mode = self._session.mode
        return self._mode

    @property
    def log_file(self) -> Path:
        return self._session.log_file

    def lifecycle_owner_snapshot(self) -> dict[str, object]:
        return {
            "owner": self.owner_name,
            "resource_fields": self.resource_fields,
            "stop_ingress": self.stop_ingress,
            "shutdown_policy": self.shutdown_policy,
            "late_callback_rule": self.late_callback_rule,
        }

    def set_mode(self, mode: SessionLoggingMode | str) -> None:
        self._mode = SessionLoggingMode(mode)
        if self._closed:
            return
        self._session.set_mode(self._mode)

    def attach_realtime_sink(self, sink: RealtimeLogSink) -> None:
        if self._closed:
            return
        self._session.attach_realtime_sink(sink)

    def detach_realtime_sink(self) -> None:
        if self._closed:
            return
        self._session.detach_realtime_sink()

    def emit_basic(self, message: str, *, level: int = logging.INFO) -> None:
        if self._closed:
            self._emit_fallback(message, level=level)
            return
        self._session.emit_basic(message, level=level)

    def emit_detailed(self, message: str, *, level: int = logging.INFO) -> bool:
        if self.mode is not SessionLoggingMode.DETAILED:
            return False
        if self._closed:
            self._emit_fallback(message, level=level)
            return True
        return self._session.emit_detailed(message, level=level)

    def emit_detailed_lazy(
        self,
        build_message: Callable[[], str],
        *,
        level: int = logging.INFO,
    ) -> bool:
        if self.mode is not SessionLoggingMode.DETAILED:
            return False
        if self._closed:
            self._emit_fallback(build_message(), level=level)
            return True
        return self._session.emit_detailed_lazy(build_message, level=level)

    def emit_persisted(self, message: str, *, level: int = logging.INFO) -> None:
        if self._closed:
            self._emit_fallback(message, level=level)
            return
        self._session.emit_persisted(message, level=level)

    def close_after_producers_stop(
        self,
        *,
        cleanup_failures: Sequence[BaseException] = (),
    ) -> None:
        if self._closed:
            return

        close_failures: list[BaseException] = []
        try:
            self._session.emit_persisted(
                _format_final_shutdown_summary(cleanup_failures),
                level=logging.INFO,
            )
        except Exception as exc:
            close_failures.append(exc)

        try:
            self._close_session_as_logging_owner()
        except Exception as exc:
            close_failures.append(exc)
        finally:
            self._closed = True
            self._mode = self._session.mode

        if close_failures:
            self._emit_close_failure_diagnostic(close_failures)
            _raise_runtime_logging_close_error(close_failures)

    def close(self) -> None:
        self.close_after_producers_stop()

    def _emit_close_failure_diagnostic(self, failures: Sequence[BaseException]) -> None:
        failure_types = ",".join(type(failure).__name__ for failure in failures) or "none"
        self._emit_fallback(
            "[Lifecycle][Shutdown] runtime_logging_close_failed "
            f"failure_count={len(failures)} failure_types={failure_types}",
            level=logging.ERROR,
            message_is_safe=True,
        )

    def _close_session_as_logging_owner(self) -> None:
        self._session.close_terminal_owner()

    def _emit_fallback(
        self,
        message: str,
        *,
        level: int,
        message_is_safe: bool = False,
    ) -> None:
        fallback_message = (
            message if message_is_safe else _format_late_log_fallback(message, level=level)
        )
        if self._emit_to_fallback_logger(fallback_message, level=level):
            return
        stream = getattr(sys, "stderr", None)
        if stream is None:
            return
        try:
            stream.write(f"{logging.getLevelName(level)}: {fallback_message}\n")
            stream.flush()
        except Exception:
            pass

    def _emit_to_fallback_logger(self, message: str, *, level: int) -> bool:
        if self._fallback_logger is None:
            return False
        try:
            if self._fallback_logger.disabled or not self._fallback_logger.isEnabledFor(level):
                return False
            record = self._fallback_logger.makeRecord(
                self._fallback_logger.name,
                level,
                fn="",
                lno=0,
                msg=message,
                args=(),
                exc_info=None,
            )
            if not self._fallback_logger.filter(record):
                return False
        except Exception:
            return False

        emitted = False
        for handler in list(self._fallback_logger.handlers):
            if level < handler.level:
                continue
            try:
                handler.handle(record)
                emitted = True
            except Exception:
                pass
        return emitted


def _format_final_shutdown_summary(failures: Sequence[BaseException]) -> str:
    failure_types = ",".join(type(failure).__name__ for failure in failures) or "none"
    status = "failed" if failures else "ok"
    return (
        "[Lifecycle][Shutdown] final_summary "
        "owner=RuntimeLoggingService "
        f"status={status} "
        f"failure_count={len(failures)} "
        f"failure_types={failure_types}"
    )


def _format_late_log_fallback(message: str, *, level: int) -> str:
    level_name = logging.getLevelName(level)
    if not isinstance(level_name, str):
        level_name = str(level_name)
    message_hash = hashlib.sha256(message.encode("utf-8", errors="replace")).hexdigest()[:16]
    return (
        "[Lifecycle][Shutdown] late_runtime_log_dropped "
        f"level={level_name.replace(' ', '_')} "
        f"message_len={len(message)} "
        f"message_sha256={message_hash}"
    )


def _raise_runtime_logging_close_error(failures: Sequence[BaseException]) -> None:
    raise RuntimeLoggingCloseError(failures) from None
