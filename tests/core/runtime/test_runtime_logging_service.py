from __future__ import annotations

import hashlib
import logging
from logging.handlers import QueueHandler
from pathlib import Path
from uuid import uuid4

import pytest

from puripuly_heart.core.runtime import RuntimeLoggingCloseError, RuntimeLoggingService
from puripuly_heart.core.runtime_logging import (
    SessionLoggingMode,
    SessionRuntimeLoggingService,
    configure_main_logging,
)
from tests.helpers.lifecycle import assert_lifecycle_structure


class FakeSessionRuntimeLogging:
    def __init__(
        self,
        *,
        persisted_error: BaseException | None = None,
        close_error: BaseException | None = None,
    ) -> None:
        self.mode = SessionLoggingMode.BASIC
        self.log_file = Path("runtime.log")
        self.persisted_error = persisted_error
        self.close_error = close_error
        self.basic_messages: list[tuple[int, str]] = []
        self.detailed_messages: list[tuple[int, str]] = []
        self.persisted_messages: list[tuple[int, str]] = []
        self.events: list[str] = []
        self.attached_sinks: list[object] = []
        self.close_calls = 0

    def set_mode(self, mode: SessionLoggingMode | str) -> None:
        self.mode = SessionLoggingMode(mode)

    def attach_realtime_sink(self, sink: object) -> None:
        self.attached_sinks.append(sink)

    def detach_realtime_sink(self) -> None:
        self.events.append("detach_realtime_sink")

    def emit_basic(self, message: str, *, level: int = logging.INFO) -> None:
        self.basic_messages.append((level, message))

    def emit_detailed(self, message: str, *, level: int = logging.INFO) -> bool:
        if self.mode is not SessionLoggingMode.DETAILED:
            return False
        self.detailed_messages.append((level, message))
        return True

    def emit_detailed_lazy(
        self,
        build_message,
        *,
        level: int = logging.INFO,
    ) -> bool:
        if self.mode is not SessionLoggingMode.DETAILED:
            return False
        self.detailed_messages.append((level, build_message()))
        return True

    def emit_persisted(self, message: str, *, level: int = logging.INFO) -> None:
        self.persisted_messages.append((level, message))
        self.events.append("persisted_summary")
        if self.persisted_error is not None:
            raise self.persisted_error

    def close(self) -> None:
        self.close_calls += 1
        self.events.append("session_close")
        if self.close_error is not None:
            raise self.close_error


class _FallbackCapture(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.messages: list[tuple[int, str]] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append((record.levelno, record.getMessage()))


class _RealtimeSink:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def append_log(self, line: str) -> None:
        self.lines.append(line)


class _RealtimeHandler(logging.Handler):
    def __init__(self, sink: _RealtimeSink) -> None:
        super().__init__()
        self._sink = sink
        self.closed = False
        self.setFormatter(logging.Formatter("%(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        self._sink.append_log(self.format(record))

    def close(self) -> None:
        self.closed = True
        super().close()


def _fallback_logger() -> tuple[logging.Logger, _FallbackCapture]:
    logger = logging.getLogger(f"test.runtime_logging.fallback.{uuid4()}")
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.INFO)
    handler = _FallbackCapture()
    logger.addHandler(handler)
    return logger, handler


def _wait_for_text(path: Path, text: str) -> None:
    content = path.read_text(encoding="utf-8")
    assert text in content


def _bounded_late_log_hash(message: str) -> str:
    return hashlib.sha256(message.encode("utf-8", errors="replace")).hexdigest()[:16]


def _assert_bounded_late_log(message: str, *, raw_message: str, level: int) -> None:
    assert message.startswith("[Lifecycle][Shutdown] late_runtime_log_dropped ")
    assert f"level={logging.getLevelName(level)}" in message
    assert f"message_len={len(raw_message)}" in message
    assert f"message_sha256={_bounded_late_log_hash(raw_message)}" in message
    assert raw_message not in message


def _contains_object_identity(value: object, target: object) -> bool:
    if value is target:
        return True
    if isinstance(value, dict):
        return any(
            _contains_object_identity(item, target) for pair in value.items() for item in pair
        )
    if isinstance(value, (tuple, list, set, frozenset)):
        return any(_contains_object_identity(item, target) for item in value)
    return False


def _assert_close_error_is_diagnostic_safe(
    error: RuntimeLoggingCloseError,
    raw_failures: tuple[BaseException, ...],
) -> None:
    expected_failure_types = tuple(type(failure).__name__ for failure in raw_failures)
    expected_failure_type_text = ",".join(expected_failure_types)

    assert error.failure_count == len(raw_failures)
    assert error.failure_types == expected_failure_types
    assert f"failure_count={len(raw_failures)}" in str(error)
    assert f"failure_types={expected_failure_type_text}" in str(error)
    assert error.__cause__ is None
    assert error.__context__ is None

    public_attrs = {name: value for name, value in vars(error).items() if not name.startswith("_")}
    assert "failures" not in public_attrs
    for raw_failure in raw_failures:
        raw_message = str(raw_failure)
        assert raw_message not in str(error)
        assert raw_message not in repr(error)
        for attr_name, attr_value in public_attrs.items():
            assert not _contains_object_identity(attr_value, raw_failure), attr_name
            assert raw_message not in repr(attr_value), attr_name


def test_runtime_logging_service_exposes_lifecycle_inventory() -> None:
    service = RuntimeLoggingService(session_service=FakeSessionRuntimeLogging())

    snapshot = service.lifecycle_owner_snapshot()

    assert_lifecycle_structure(snapshot)
    assert snapshot["owner"] == "RuntimeLoggingService"


def test_final_shutdown_summary_is_metadata_only_and_emitted_before_close() -> None:
    session = FakeSessionRuntimeLogging()
    service = RuntimeLoggingService(session_service=session)

    service.close_after_producers_stop(
        cleanup_failures=[
            RuntimeError("provider secret raw text must not be logged"),
            ValueError("api-key-looking message must not be logged"),
        ]
    )

    assert session.events == ["persisted_summary", "session_close"]
    assert len(session.persisted_messages) == 1
    _, summary = session.persisted_messages[0]
    assert summary.startswith("[Lifecycle][Shutdown] final_summary ")
    assert "status=failed" in summary
    assert "failure_count=2" in summary
    assert "failure_types=RuntimeError,ValueError" in summary
    assert "secret raw text" not in summary
    assert "api-key-looking" not in summary


def test_close_detaches_realtime_sink_closes_queue_and_is_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_logger = logging.getLogger(f"test.runtime_logging.owner.root.{uuid4()}")
    root_logger.handlers.clear()
    root_logger.propagate = False
    monkeypatch.setattr("puripuly_heart.core.runtime_logging.user_config_dir", lambda: tmp_path)

    session = SessionRuntimeLoggingService(
        root_logger=root_logger, ui_handler_factory=_RealtimeHandler
    )
    service = RuntimeLoggingService(session_service=session)
    sink = _RealtimeSink()
    service.attach_realtime_sink(sink)

    service.emit_basic("before terminal close")
    service.close_after_producers_stop()
    service.close_after_producers_stop()

    log_file = session.log_file
    _wait_for_text(log_file, "before terminal close")
    content = log_file.read_text(encoding="utf-8")
    assert content.count("[Lifecycle][Shutdown] final_summary") == 1
    assert "status=ok" in content
    assert not any(isinstance(handler, QueueHandler) for handler in root_logger.handlers)
    assert sink.lines == ["before terminal close"]


def test_terminal_close_force_closes_preconfigured_main_queue_and_persists_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_logger = logging.getLogger(f"test.runtime_logging.main_preconfigured.{uuid4()}")
    root_logger.handlers.clear()
    root_logger.propagate = False
    monkeypatch.setattr("puripuly_heart.core.runtime_logging.user_config_dir", lambda: tmp_path)
    main_sinks = configure_main_logging(root_logger=root_logger, log_dir=tmp_path)

    try:
        queue_handler = main_sinks.file_queue_handler
        assert isinstance(queue_handler, QueueHandler)
        assert queue_handler in root_logger.handlers

        service = RuntimeLoggingService(
            session_factory=lambda: SessionRuntimeLoggingService(root_logger=root_logger)
        )
        assert service.log_file == main_sinks.log_file

        service.emit_basic("queued before terminal close")
        service.close_after_producers_stop()

        content = main_sinks.log_file.read_text(encoding="utf-8")
        assert "queued before terminal close" in content
        assert "[Lifecycle][Shutdown] final_summary" in content
        assert "status=ok" in content
        assert queue_handler not in root_logger.handlers
        assert not any(isinstance(handler, QueueHandler) for handler in root_logger.handlers)
    finally:
        main_sinks.close(force=True)


def test_late_logs_after_terminal_close_do_not_propagate_to_preconfigured_file_queue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_logger = logging.getLogger(f"test.runtime_logging.late_preconfigured.{uuid4()}")
    root_logger.handlers.clear()
    root_logger.propagate = False
    monkeypatch.setattr("puripuly_heart.core.runtime_logging.user_config_dir", lambda: tmp_path)
    main_sinks = configure_main_logging(root_logger=root_logger, log_dir=tmp_path)
    fallback_logger = logging.getLogger(f"{root_logger.name}.fallback")
    fallback_logger.handlers.clear()
    fallback_logger.propagate = True
    fallback_logger.setLevel(logging.INFO)
    fallback = _FallbackCapture()
    fallback_logger.addHandler(fallback)
    raw_late_message = "late user text with secret=raw-value"

    try:
        service = RuntimeLoggingService(
            session_factory=lambda: SessionRuntimeLoggingService(root_logger=root_logger),
            fallback_logger=fallback_logger,
        )
        service.close_after_producers_stop()

        service.emit_basic(raw_late_message, level=logging.WARNING)
        if main_sinks.file_queue is not None:
            main_sinks.file_queue.join()

        content = main_sinks.log_file.read_text(encoding="utf-8")
        assert raw_late_message not in content
        assert fallback.messages
        fallback_level, fallback_message = fallback.messages[-1]
        assert fallback_level == logging.WARNING
        _assert_bounded_late_log(
            fallback_message,
            raw_message=raw_late_message,
            level=logging.WARNING,
        )
    finally:
        main_sinks.close(force=True)


def test_late_logs_after_close_use_fallback_without_session_writes() -> None:
    session = FakeSessionRuntimeLogging()
    fallback_logger, fallback = _fallback_logger()
    service = RuntimeLoggingService(session_service=session, fallback_logger=fallback_logger)
    service.set_mode(SessionLoggingMode.DETAILED)
    service.close_after_producers_stop()
    session.basic_messages.clear()
    session.detailed_messages.clear()

    service.emit_basic("late basic", level=logging.WARNING)
    detailed_result = service.emit_detailed("late detail", level=logging.ERROR)

    assert detailed_result is True
    assert session.basic_messages == []
    assert session.detailed_messages == []
    assert [level for level, _message in fallback.messages] == [logging.WARNING, logging.ERROR]
    _assert_bounded_late_log(
        fallback.messages[0][1],
        raw_message="late basic",
        level=logging.WARNING,
    )
    _assert_bounded_late_log(
        fallback.messages[1][1],
        raw_message="late detail",
        level=logging.ERROR,
    )


@pytest.mark.parametrize("include_persisted_failure", [False, True])
def test_close_failure_is_bounded_and_surfaced_after_final_summary_attempt(
    include_persisted_failure: bool,
) -> None:
    close_failure = OSError("disk path with raw close details")
    persisted_failure = RuntimeError("provider payload with raw summary details")
    raw_failures = (
        (persisted_failure, close_failure) if include_persisted_failure else (close_failure,)
    )
    session = FakeSessionRuntimeLogging(
        persisted_error=persisted_failure if include_persisted_failure else None,
        close_error=close_failure,
    )
    fallback_logger, fallback = _fallback_logger()
    service = RuntimeLoggingService(session_service=session, fallback_logger=fallback_logger)

    with pytest.raises(RuntimeLoggingCloseError) as exc_info:
        service.close_after_producers_stop()

    assert session.events == ["persisted_summary", "session_close"]
    _assert_close_error_is_diagnostic_safe(exc_info.value, raw_failures)
    expected_failure_types = ",".join(type(failure).__name__ for failure in raw_failures)
    assert fallback.messages == [
        (
            logging.ERROR,
            "[Lifecycle][Shutdown] runtime_logging_close_failed "
            f"failure_count={len(raw_failures)} "
            f"failure_types={expected_failure_types}",
        )
    ]
    for raw_failure in raw_failures:
        assert str(raw_failure) not in fallback.messages[0][1]
