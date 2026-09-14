from __future__ import annotations

import io
import logging
import re
import time
from collections.abc import Awaitable
from dataclasses import dataclass
from logging.handlers import QueueHandler, RotatingFileHandler
from uuid import uuid4

import pytest

from puripuly_heart.core.messages import (
    CONTENT_POLICY_METADATA_ONLY,
    CONTENT_POLICY_RAW_USER_TEXT_ALLOWED,
    DIAGNOSTIC_CATEGORY_NETWORK,
    DIAGNOSTIC_CATEGORY_UNKNOWN,
    DIAGNOSTIC_VISIBILITY_BASIC,
    DIAGNOSTIC_VISIBILITY_DETAILED,
    DIAGNOSTIC_VISIBILITY_DIAGNOSTIC_ONLY,
    SEVERITY_ERROR,
    SEVERITY_INFO,
    SEVERITY_WARNING,
    ErrorDiagnostics,
)
from puripuly_heart.core.observability import (
    ConversationRecord,
    DiagnosticEvent,
    PersistedDiagnosticRecord,
    ProviderObservationEvent,
    RuntimeLogEvent,
)
from puripuly_heart.core.output.models import OutputRoutingDecision
from puripuly_heart.core.runtime_logging import (
    SessionLoggingMode,
    SessionRuntimeLoggingService,
    configure_main_logging,
)


@dataclass
class _SharedSinkBundle:
    stream_handler: logging.Handler
    file_handler: logging.Handler
    log_file: object


class _ObservabilityRunner:
    def __init__(self) -> None:
        self.awaitables: list[Awaitable[None]] = []

    def __call__(self, awaitable: Awaitable[None]) -> None:
        self.awaitables.append(awaitable)

    async def drain(self) -> None:
        while self.awaitables:
            await self.awaitables.pop(0)


class _StructuredObservabilitySink:
    def __init__(self) -> None:
        self.runtime_events: list[RuntimeLogEvent] = []
        self.diagnostic_events: list[DiagnosticEvent] = []
        self.provider_events: list[ProviderObservationEvent] = []
        self.conversation_records: list[ConversationRecord] = []
        self.persisted_records: list[PersistedDiagnosticRecord] = []

    async def emit_runtime_log(self, event: RuntimeLogEvent) -> None:
        self.runtime_events.append(event)

    async def emit_diagnostic(self, event: DiagnosticEvent) -> None:
        self.diagnostic_events.append(event)

    async def emit_provider_observation(self, event: ProviderObservationEvent) -> None:
        self.provider_events.append(event)

    async def record_conversation(self, record: ConversationRecord) -> None:
        self.conversation_records.append(record)

    async def persist_diagnostic(self, record: PersistedDiagnosticRecord) -> None:
        self.persisted_records.append(record)


class _SynchronousFailingRuntimeLogSink:
    def emit_runtime_log(self, _event: RuntimeLogEvent) -> Awaitable[None]:
        raise RuntimeError("synchronous runtime-log sink failure")


class _CloseRaisingAwaitable:
    def __init__(self) -> None:
        self.closed = False

    def __await__(self):
        if False:
            yield None
        return None

    def close(self) -> None:
        self.closed = True
        raise RuntimeError("awaitable close failure")


class _CloseRaisingRuntimeLogSink:
    def __init__(self) -> None:
        self.awaitable = _CloseRaisingAwaitable()

    def emit_runtime_log(self, _event: RuntimeLogEvent) -> Awaitable[None]:
        return self.awaitable


class _RaisingObservabilityRunner:
    def __init__(self) -> None:
        self.awaitables: list[Awaitable[None]] = []

    def __call__(self, awaitable: Awaitable[None]) -> None:
        self.awaitables.append(awaitable)
        raise RuntimeError("observability runner failure")


class _DelayingForwardingHandler(logging.Handler):
    def __init__(
        self,
        target: logging.Handler,
        *,
        delayed_message: str,
        delay_s: float = 0.2,
    ) -> None:
        super().__init__()
        self._target = target
        self._delayed_message = delayed_message
        self._delay_s = delay_s
        self.started = False

    def emit(self, record: logging.LogRecord) -> None:
        if record.getMessage() == self._delayed_message:
            self.started = True
            time.sleep(self._delay_s)
        self._target.handle(record)


class _RaisingHandler(logging.Handler):
    def emit(self, _record: logging.LogRecord) -> None:
        raise RuntimeError("synthetic handler failure")


def _format_with_handler(handler: logging.Handler) -> str:
    record = logging.LogRecord(
        name="test.runtime",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="hello",
        args=(),
        exc_info=None,
    )
    record.created = 0.0
    record.msecs = 123.0
    return handler.format(record)


def _wait_for_log_text(log_file, text: str) -> None:
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        if text in log_file.read_text(encoding="utf-8"):
            return
        time.sleep(0.01)
    assert text in log_file.read_text(encoding="utf-8")


def _wait_until(condition) -> None:
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(0.01)
    assert condition()


def _make_runtime_logging_capture() -> tuple[SessionRuntimeLoggingService, io.StringIO]:
    stream = io.StringIO()
    stream_handler = logging.StreamHandler(stream)

    root_logger = logging.getLogger(f"test.runtime_logging.root.{uuid4()}")
    root_logger.handlers.clear()
    root_logger.propagate = False

    session_logger = logging.getLogger(f"test.runtime_logging.session.{uuid4()}")
    session_logger.handlers.clear()
    session_logger.propagate = False

    runtime_logging = SessionRuntimeLoggingService(
        root_logger=root_logger,
        session_logger=session_logger,
        sinks=_SharedSinkBundle(
            stream_handler=stream_handler,
            file_handler=logging.NullHandler(),
            log_file="runtime.log",
        ),
    )
    return runtime_logging, stream


def test_configure_main_logging_formats_new_handlers_with_millisecond_resolution(tmp_path) -> None:
    root_logger = logging.getLogger(f"test.runtime_logging.configure.new.{uuid4()}")
    root_logger.handlers.clear()
    root_logger.propagate = False

    sinks = configure_main_logging(root_logger=root_logger, log_dir=tmp_path)

    try:
        formatted_stream = _format_with_handler(sinks.stream_handler)
        formatted_file = _format_with_handler(sinks.file_handler)

        assert re.fullmatch(
            r"\d{2}:\d{2}:\d{2}\.123 \[INFO\] test\.runtime: hello", formatted_stream
        )
        assert re.fullmatch(r"\d{2}:\d{2}:\d{2}\.123 \[INFO\] test\.runtime: hello", formatted_file)
    finally:
        sinks.close()


def test_configure_main_logging_reused_handlers_get_millisecond_resolution_formatter(
    tmp_path,
) -> None:
    root_logger = logging.getLogger(f"test.runtime_logging.configure.reused.{uuid4()}")
    root_logger.handlers.clear()
    root_logger.propagate = False

    existing_stream = logging.StreamHandler(io.StringIO())
    existing_stream.setFormatter(logging.Formatter("%(message)s"))
    existing_file = RotatingFileHandler(
        tmp_path / "puripuly_heart.log",
        maxBytes=4096,
        backupCount=0,
        encoding="utf-8",
    )
    existing_file.setFormatter(logging.Formatter("%(message)s"))
    root_logger.addHandler(existing_stream)
    root_logger.addHandler(existing_file)

    sinks = configure_main_logging(root_logger=root_logger, log_dir=tmp_path)

    try:
        assert sinks.stream_handler is existing_stream
        assert sinks.file_handler is existing_file
        assert re.fullmatch(
            r"\d{2}:\d{2}:\d{2}\.123 \[INFO\] test\.runtime: hello",
            _format_with_handler(existing_stream),
        )
        assert re.fullmatch(
            r"\d{2}:\d{2}:\d{2}\.123 \[INFO\] test\.runtime: hello",
            _format_with_handler(existing_file),
        )
    finally:
        sinks.close()


def test_configure_main_logging_routes_file_writes_through_queue(tmp_path) -> None:
    root_logger = logging.getLogger(f"test.runtime_logging.queue.{uuid4()}")
    root_logger.handlers.clear()
    root_logger.propagate = False

    sinks = configure_main_logging(root_logger=root_logger, log_dir=tmp_path)

    try:
        assert isinstance(sinks.file_queue_handler, QueueHandler)
        assert isinstance(sinks.file_handler, RotatingFileHandler)
        assert sinks.file_queue_handler in root_logger.handlers
        assert sinks.file_handler not in root_logger.handlers

        root_logger.info("queued file record")
        _wait_for_log_text(sinks.log_file, "queued file record")
    finally:
        sinks.close()


def test_configured_main_logging_redacts_direct_logger_records_before_live_and_file(
    tmp_path,
) -> None:
    root_logger = logging.getLogger(f"test.runtime_logging.direct_redaction.{uuid4()}")
    root_logger.handlers.clear()
    root_logger.propagate = False
    stream = io.StringIO()
    root_logger.addHandler(logging.StreamHandler(stream))
    sinks = configure_main_logging(root_logger=root_logger, log_dir=tmp_path)
    unsafe = "provider_response_body={'token':'direct-log-secret'}"

    try:
        root_logger.error(unsafe)
        _wait_for_log_text(sinks.log_file, "provider-response-body-redacted")

        combined = stream.getvalue() + sinks.log_file.read_text(encoding="utf-8")
        assert "direct-log-secret" not in combined
        assert "provider_response_body" not in combined
        assert "[provider-response-body-redacted]" in combined
    finally:
        sinks.close()


def test_configured_main_logging_drops_exception_and_stack_details_before_live_and_file(
    tmp_path,
) -> None:
    root_logger = logging.getLogger(f"test.runtime_logging.exception_redaction.{uuid4()}")
    root_logger.handlers.clear()
    root_logger.propagate = False
    stream = io.StringIO()
    root_logger.addHandler(logging.StreamHandler(stream))
    sinks = configure_main_logging(root_logger=root_logger, log_dir=tmp_path)

    try:
        root_logger.info("ordinary safe exception-neighbor message")
        try:
            raise RuntimeError(
                "provider_response_body={'token':'exception-provider-secret'} "
                "authorization=Bearer exception-token-secret"
            )
        except RuntimeError:
            root_logger.exception("provider call failed safely")
        root_logger.error("stack-only failure breadcrumb", stack_info=True)
        _wait_for_log_text(sinks.log_file, "stack-only failure breadcrumb")

        combined = stream.getvalue() + sinks.log_file.read_text(encoding="utf-8")
        assert "ordinary safe exception-neighbor message" in combined
        assert "provider call failed safely" in combined
        assert "stack-only failure breadcrumb" in combined
        assert "exception-provider-secret" not in combined
        assert "exception-token-secret" not in combined
        assert "provider_response_body" not in combined
        assert "Traceback" not in combined
        assert "Stack (most recent call last)" not in combined
        assert 'File "' not in combined
        assert "RuntimeError" not in combined
    finally:
        sinks.close()


def test_session_runtime_logging_redacts_direct_records_with_injected_sinks(tmp_path) -> None:
    root_logger = logging.getLogger(f"test.runtime_logging.injected.root.{uuid4()}")
    root_logger.handlers.clear()
    root_logger.propagate = False
    session_logger = logging.getLogger(f"test.runtime_logging.injected.session.{uuid4()}")
    session_logger.handlers.clear()
    session_logger.propagate = False
    stream = io.StringIO()
    stream_handler = logging.StreamHandler(stream)
    stream_handler.setFormatter(logging.Formatter("%(message)s"))
    log_file = tmp_path / "injected-direct.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    runtime_logging = SessionRuntimeLoggingService(
        root_logger=root_logger,
        session_logger=session_logger,
        sinks=_SharedSinkBundle(
            stream_handler=stream_handler,
            file_handler=file_handler,
            log_file=log_file,
        ),
    )

    try:
        root_logger.info("ordinary safe injected direct record")
        root_logger.error("provider_response_body={'token':'injected-root-secret'}")
        try:
            raise RuntimeError("raw_exception=token=injected-exception-secret")
        except RuntimeError:
            session_logger.exception("session exception breadcrumb")
        session_logger.error("session stack breadcrumb", stack_info=True)
        file_handler.flush()

        combined = stream.getvalue() + log_file.read_text(encoding="utf-8")
        assert "ordinary safe injected direct record" in combined
        assert "session exception breadcrumb" in combined
        assert "session stack breadcrumb" in combined
        assert "injected-root-secret" not in combined
        assert "injected-exception-secret" not in combined
        assert "provider_response_body" not in combined
        assert "raw_exception" not in combined
        assert "Traceback" not in combined
        assert "Stack (most recent call last)" not in combined
        assert 'File "' not in combined
        assert "RuntimeError" not in combined
    finally:
        runtime_logging.close()
        file_handler.close()


def test_configure_main_logging_uses_bounded_rotation_policy(tmp_path) -> None:
    root_logger = logging.getLogger(f"test.runtime_logging.rotation_policy.{uuid4()}")
    root_logger.handlers.clear()
    root_logger.propagate = False

    sinks = configure_main_logging(root_logger=root_logger, log_dir=tmp_path)

    try:
        assert isinstance(sinks.file_handler, RotatingFileHandler)
        assert sinks.file_handler.maxBytes == 10 * 1024 * 1024
        assert sinks.file_handler.backupCount == 1
    finally:
        sinks.close()


def test_configure_main_logging_names_single_backup_with_log_extension(tmp_path) -> None:
    root_logger = logging.getLogger(f"test.runtime_logging.rotation_name.{uuid4()}")
    root_logger.handlers.clear()
    root_logger.propagate = False

    sinks = configure_main_logging(root_logger=root_logger, log_dir=tmp_path)

    try:
        default_backup_name = str(sinks.log_file) + ".1"
        assert sinks.file_handler.rotation_filename(default_backup_name) == str(
            tmp_path / "puripuly_heart.backup.log"
        )

        sinks.file_handler.stream.write("old log line\n")
        sinks.file_handler.flush()
        sinks.file_handler.doRollover()

        assert (tmp_path / "puripuly_heart.backup.log").exists()
        assert not (tmp_path / "puripuly_heart.log.1").exists()
    finally:
        sinks.close()


def test_configure_main_logging_reuses_existing_queue_handler(tmp_path) -> None:
    root_logger = logging.getLogger(f"test.runtime_logging.queue.reuse.{uuid4()}")
    root_logger.handlers.clear()
    root_logger.propagate = False

    first = configure_main_logging(root_logger=root_logger, log_dir=tmp_path)
    second = configure_main_logging(root_logger=root_logger, log_dir=tmp_path)

    try:
        queue_handlers = [
            handler for handler in root_logger.handlers if isinstance(handler, QueueHandler)
        ]
        assert queue_handlers == [first.file_queue_handler]
        assert second.file_queue_handler is first.file_queue_handler
        assert second.file_handler is first.file_handler
    finally:
        first.close()
        second.close()


def test_configure_main_logging_keeps_shared_queue_alive_until_last_close(tmp_path) -> None:
    root_logger = logging.getLogger(f"test.runtime_logging.queue.refcount.{uuid4()}")
    root_logger.handlers.clear()
    root_logger.propagate = False

    first = configure_main_logging(root_logger=root_logger, log_dir=tmp_path)
    second = configure_main_logging(root_logger=root_logger, log_dir=tmp_path)

    try:
        assert second.file_queue_handler is first.file_queue_handler

        first.close()
        assert second.file_queue_handler in root_logger.handlers

        root_logger.info("second still writes after first close")
        _wait_for_log_text(second.log_file, "second still writes after first close")

        second.close()
        assert second.file_queue_handler not in root_logger.handlers
    finally:
        first.close()
        second.close()


def test_force_close_shared_queue_releases_even_with_outstanding_refs(tmp_path) -> None:
    root_logger = logging.getLogger(f"test.runtime_logging.queue.force_close.{uuid4()}")
    root_logger.handlers.clear()
    root_logger.propagate = False

    first = configure_main_logging(root_logger=root_logger, log_dir=tmp_path)
    second = configure_main_logging(root_logger=root_logger, log_dir=tmp_path)

    try:
        assert second.file_queue_handler is first.file_queue_handler

        first.close(force=True)

        assert first.file_queue_handler not in root_logger.handlers
        second.close()
        assert second.file_queue_handler not in root_logger.handlers
    finally:
        first.close()
        second.close()


def test_default_session_logging_services_share_queue_until_last_close(
    tmp_path, monkeypatch
) -> None:
    root_logger = logging.getLogger(f"test.runtime_logging.service.refcount.{uuid4()}")
    root_logger.handlers.clear()
    root_logger.propagate = False
    monkeypatch.setattr("puripuly_heart.core.runtime_logging.user_config_dir", lambda: tmp_path)

    first = SessionRuntimeLoggingService(root_logger=root_logger)
    second = SessionRuntimeLoggingService(root_logger=root_logger)

    try:
        assert second.log_file == first.log_file

        first.close()
        second.emit_basic("service two writes after service one close")
        _wait_for_log_text(second.log_file, "service two writes after service one close")

        second.close()
        queue_handlers = [
            handler for handler in root_logger.handlers if isinstance(handler, QueueHandler)
        ]
        assert queue_handlers == []
    finally:
        first.close()
        second.close()


def test_configure_main_logging_removes_stale_queue_when_log_dir_changes(tmp_path) -> None:
    root_logger = logging.getLogger(f"test.runtime_logging.queue.stale.{uuid4()}")
    root_logger.handlers.clear()
    root_logger.propagate = False
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"

    first = configure_main_logging(root_logger=root_logger, log_dir=first_dir)
    root_logger.info("before log dir switch")
    _wait_for_log_text(first.log_file, "before log dir switch")

    second = configure_main_logging(root_logger=root_logger, log_dir=second_dir)

    try:
        assert first.file_queue_handler not in root_logger.handlers
        assert second.file_queue_handler in root_logger.handlers

        root_logger.info("after log dir switch")
        _wait_for_log_text(second.log_file, "after log dir switch")
        assert "after log dir switch" not in first.log_file.read_text(encoding="utf-8")
    finally:
        first.close()
        second.close()


def test_bounded_file_queue_drops_under_pressure_without_blocking_producers(tmp_path) -> None:
    root_logger = logging.getLogger(f"test.runtime_logging.queue.pressure.{uuid4()}")
    root_logger.handlers.clear()
    root_logger.propagate = False
    sinks = configure_main_logging(root_logger=root_logger, log_dir=tmp_path)
    sinks.stream_handler.setLevel(logging.CRITICAL)
    assert sinks.file_queue_listener is not None
    assert sinks.file_queue_handler is not None
    delaying_handler = _DelayingForwardingHandler(
        sinks.file_handler,
        delayed_message="hold-listener",
    )
    sinks.file_queue_listener.handlers = (delaying_handler,)

    try:
        root_logger.info("hold-listener")
        _wait_until(lambda: delaying_handler.started)
        started = time.monotonic()
        for index in range(3000):
            root_logger.info("pressure-%s", index)
        elapsed = time.monotonic() - started

        assert elapsed < 1.0
        assert sinks.file_queue_handler.dropped_records > 0
    finally:
        sinks.close()


def test_terminal_file_record_survives_queue_pressure_and_later_reports_loss(tmp_path) -> None:
    root_logger = logging.getLogger(f"test.runtime_logging.queue.terminal.{uuid4()}")
    root_logger.handlers.clear()
    root_logger.propagate = False
    sinks = configure_main_logging(root_logger=root_logger, log_dir=tmp_path)
    sinks.stream_handler.setLevel(logging.CRITICAL)
    assert sinks.file_queue_listener is not None
    assert sinks.file_queue_handler is not None
    delaying_handler = _DelayingForwardingHandler(
        sinks.file_handler,
        delayed_message="hold-terminal-listener",
        delay_s=0.5,
    )
    sinks.file_queue_listener.handlers = (delaying_handler,)
    runtime_logging = SessionRuntimeLoggingService(
        root_logger=root_logger,
        sinks=sinks,
    )

    try:
        root_logger.info("hold-terminal-listener")
        _wait_until(lambda: delaying_handler.started)
        assert sinks.file_queue is not None
        while not sinks.file_queue.full():
            root_logger.info("queued-pressure")

        runtime_logging.emit_persisted("[Lifecycle] terminal-one")
        _wait_for_log_text(sinks.log_file, "[Lifecycle] terminal-one")
        runtime_logging.emit_persisted("[Lifecycle] terminal-two")
        _wait_for_log_text(sinks.log_file, "[Lifecycle] terminal-two")

        terminal_two = next(
            line
            for line in sinks.log_file.read_text(encoding="utf-8").splitlines()
            if "[Lifecycle] terminal-two" in line
        )
        assert "logging_delivery_dropped=1" in terminal_two
        assert "logging_delivery_failures=0" in terminal_two
    finally:
        runtime_logging.close()
        sinks.close()


def test_file_listener_survives_handler_exceptions(tmp_path) -> None:
    root_logger = logging.getLogger(f"test.runtime_logging.queue.failure.{uuid4()}")
    root_logger.handlers.clear()
    root_logger.propagate = False
    sinks = configure_main_logging(root_logger=root_logger, log_dir=tmp_path)
    sinks.stream_handler.setLevel(logging.CRITICAL)
    assert sinks.file_queue_listener is not None
    sinks.file_queue_listener.handlers = (_RaisingHandler(),)

    try:
        root_logger.info("first failure")
        root_logger.info("second failure")
        _wait_until(lambda: sinks.file_queue_listener.delivery_failures == 2)
        assert sinks.file_queue_listener._thread is not None
        assert sinks.file_queue_listener._thread.is_alive()
    finally:
        sinks.close()


def test_emit_persisted_delivers_through_file_queue(tmp_path) -> None:
    root_logger = logging.getLogger(f"test.runtime_logging.persisted.{uuid4()}")
    root_logger.handlers.clear()
    root_logger.propagate = False
    sinks = configure_main_logging(root_logger=root_logger, log_dir=tmp_path)
    runtime_logging = SessionRuntimeLoggingService(root_logger=root_logger, sinks=sinks)

    try:
        runtime_logging.emit_persisted("persisted critical record")
        _wait_for_log_text(sinks.log_file, "persisted critical record")
    finally:
        runtime_logging.close()
        sinks.close()


def test_emit_persisted_preserves_queued_record_order_before_direct_write(tmp_path) -> None:
    root_logger = logging.getLogger(f"test.runtime_logging.persisted.order.{uuid4()}")
    root_logger.handlers.clear()
    root_logger.propagate = False
    sinks = configure_main_logging(root_logger=root_logger, log_dir=tmp_path)
    runtime_logging = SessionRuntimeLoggingService(root_logger=root_logger, sinks=sinks)
    delaying_handler = _DelayingForwardingHandler(
        sinks.file_handler,
        delayed_message="basic before persisted",
    )
    assert sinks.file_queue_listener is not None
    sinks.file_queue_listener.handlers = (delaying_handler,)

    try:
        runtime_logging.emit_basic("basic before persisted")
        _wait_until(lambda: delaying_handler.started)

        runtime_logging.emit_persisted("persisted after queued basic")

        _wait_for_log_text(sinks.log_file, "basic before persisted")
        _wait_for_log_text(sinks.log_file, "persisted after queued basic")
        log_lines = sinks.log_file.read_text(encoding="utf-8").splitlines()
        basic_index = next(
            index for index, line in enumerate(log_lines) if "basic before persisted" in line
        )
        persisted_index = next(
            index for index, line in enumerate(log_lines) if "persisted after queued basic" in line
        )
        assert basic_index < persisted_index
    finally:
        runtime_logging.close()
        sinks.close()


def test_technical_redactor_removes_url_userinfo_from_arbitrary_logger_names() -> None:
    stream = io.StringIO()
    stream_handler = logging.StreamHandler(stream)
    stream_handler.setFormatter(logging.Formatter("%(message)s"))
    root_logger = logging.getLogger(f"test.runtime_logging.url-userinfo.{uuid4()}")
    root_logger.handlers.clear()
    root_logger.propagate = False
    root_logger.setLevel(logging.INFO)
    session_logger = logging.getLogger(f"{root_logger.name}.session")
    session_logger.handlers.clear()
    session_logger.propagate = False
    runtime_logging = SessionRuntimeLoggingService(
        root_logger=root_logger,
        session_logger=session_logger,
        sinks=_SharedSinkBundle(
            stream_handler=stream_handler,
            file_handler=logging.NullHandler(),
            log_file="runtime.log",
        ),
    )
    third_party_logger = logging.getLogger(f"{root_logger.name}.websocket-client")
    third_party_logger.handlers.clear()
    third_party_logger.propagate = True
    third_party_logger.setLevel(logging.INFO)

    try:
        third_party_logger.error("connection failed wss://alice:super-secret@example.test/socket")

        rendered = stream.getvalue()
        assert "connection failed" in rendered
        assert "alice:super-secret" not in rendered
        assert "wss://[redacted]@example.test/socket" in rendered
    finally:
        runtime_logging.close()


def test_session_runtime_logging_redacts_unsafe_legacy_text_before_live_and_persisted_sinks(
    tmp_path,
) -> None:
    root_logger = logging.getLogger(f"test.runtime_logging.redaction.root.{uuid4()}")
    root_logger.handlers.clear()
    root_logger.propagate = False
    session_logger = logging.getLogger(f"test.runtime_logging.redaction.session.{uuid4()}")
    session_logger.handlers.clear()
    session_logger.propagate = False
    stream = io.StringIO()
    stream_handler = logging.StreamHandler(stream)
    stream_handler.setFormatter(logging.Formatter("%(message)s"))
    log_file = tmp_path / "runtime-redaction.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    runtime_logging = SessionRuntimeLoggingService(
        root_logger=root_logger,
        session_logger=session_logger,
        sinks=_SharedSinkBundle(
            stream_handler=stream_handler,
            file_handler=file_handler,
            log_file=log_file,
        ),
    )
    unsafe_text = (
        "Provider failure provider_response_body="
        "{'error':'bad','token':'provider-secret-live-123'}\n"
        "Traceback (most recent call last):\n"
        '  File "provider.py", line 42, in translate\n'
        "RuntimeError: authorization=Bearer provider-token-live-456"
    )

    try:
        runtime_logging.emit_basic("ordinary safe line")
        runtime_logging.emit_basic(unsafe_text, level=logging.ERROR)
        runtime_logging.set_mode(SessionLoggingMode.DETAILED)
        runtime_logging.emit_detailed(unsafe_text, level=logging.WARNING)
        runtime_logging.emit_persisted(unsafe_text, level=logging.ERROR)
        file_handler.flush()

        live_text = stream.getvalue()
        persisted_text = log_file.read_text(encoding="utf-8")
        combined = live_text + persisted_text
        assert "ordinary safe line" in combined
        assert "provider-secret-live-123" not in combined
        assert "provider-token-live-456" not in combined
        assert "provider_response_body" not in combined
        assert "Traceback" not in combined
        assert 'File "provider.py"' not in combined
        assert "[provider-response-body-redacted]" in combined or "[redacted]" in combined
    finally:
        runtime_logging.close()
        file_handler.close()


def test_session_runtime_logging_redacts_unsafe_text_assignment_keys_before_sinks(
    tmp_path,
) -> None:
    root_logger = logging.getLogger(f"test.runtime_logging.assignment_keys.root.{uuid4()}")
    root_logger.handlers.clear()
    root_logger.propagate = False
    session_logger = logging.getLogger(f"test.runtime_logging.assignment_keys.session.{uuid4()}")
    session_logger.handlers.clear()
    session_logger.propagate = False
    stream = io.StringIO()
    stream_handler = logging.StreamHandler(stream)
    stream_handler.setFormatter(logging.Formatter("%(message)s"))
    log_file = tmp_path / "unsafe-assignment-keys.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    runtime_logging = SessionRuntimeLoggingService(
        root_logger=root_logger,
        session_logger=session_logger,
        sinks=_SharedSinkBundle(
            stream_handler=stream_handler,
            file_handler=file_handler,
            log_file=log_file,
        ),
    )
    unsafe_text = (
        "file_contents=user document text must not persist\n"
        "raw_exception=ValueError('raw provider exception text')\n"
        'stack_trace=File "provider.py", line 42, in translate'
    )

    try:
        runtime_logging.emit_basic(unsafe_text, level=logging.ERROR)
        runtime_logging.emit_persisted(unsafe_text, level=logging.ERROR)
        file_handler.flush()

        combined = stream.getvalue() + log_file.read_text(encoding="utf-8")
        assert "file_contents" not in combined
        assert "user document text" not in combined
        assert "raw_exception" not in combined
        assert "raw provider exception text" not in combined
        assert "stack_trace" not in combined
        assert 'File "provider.py"' not in combined
        assert "[redacted]" in combined
    finally:
        runtime_logging.close()
        file_handler.close()


def test_session_runtime_logging_redacts_token_assignment_variants_before_sinks(
    tmp_path,
) -> None:
    root_logger = logging.getLogger(f"test.runtime_logging.token_variants.root.{uuid4()}")
    root_logger.handlers.clear()
    root_logger.propagate = False
    session_logger = logging.getLogger(f"test.runtime_logging.token_variants.session.{uuid4()}")
    session_logger.handlers.clear()
    session_logger.propagate = False
    stream = io.StringIO()
    stream_handler = logging.StreamHandler(stream)
    stream_handler.setFormatter(logging.Formatter("%(message)s"))
    log_file = tmp_path / "token-variant-redaction.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    runtime_logging = SessionRuntimeLoggingService(
        root_logger=root_logger,
        session_logger=session_logger,
        sinks=_SharedSinkBundle(
            stream_handler=stream_handler,
            file_handler=file_handler,
            log_file=log_file,
        ),
    )
    unsafe_text = (
        "provider failed access_token=runtime-access-secret "
        "refreshToken=runtime-refresh-secret "
        "idToken=runtime-id-secret"
    )

    try:
        runtime_logging.emit_basic(unsafe_text, level=logging.ERROR)
        runtime_logging.set_mode(SessionLoggingMode.DETAILED)
        runtime_logging.emit_detailed(unsafe_text, level=logging.WARNING)
        runtime_logging.emit_persisted(unsafe_text, level=logging.ERROR)
        file_handler.flush()

        combined = stream.getvalue() + log_file.read_text(encoding="utf-8")
        assert "runtime-access-secret" not in combined
        assert "runtime-refresh-secret" not in combined
        assert "runtime-id-secret" not in combined
        assert "[redacted]" in combined
    finally:
        runtime_logging.close()
        file_handler.close()


def test_configure_main_logging_reconfigures_after_close(tmp_path) -> None:
    root_logger = logging.getLogger(f"test.runtime_logging.queue.close_reconfigure.{uuid4()}")
    root_logger.handlers.clear()
    root_logger.propagate = False

    first = configure_main_logging(root_logger=root_logger, log_dir=tmp_path)
    first.close()

    second = configure_main_logging(root_logger=root_logger, log_dir=tmp_path)

    try:
        assert second.file_queue_handler is not first.file_queue_handler
        assert second.file_queue_handler in root_logger.handlers
        root_logger.info("after reconfigure")
        _wait_for_log_text(second.log_file, "after reconfigure")
    finally:
        second.close()


def test_emit_detailed_lazy_checks_mode_before_formatting() -> None:
    runtime_logging, stream = _make_runtime_logging_capture()
    builder_calls = 0

    def builder() -> str:
        nonlocal builder_calls
        builder_calls += 1
        return "lazy detail"

    try:
        assert runtime_logging.emit_detailed_lazy(builder) is False
        assert builder_calls == 0
        assert stream.getvalue() == ""

        runtime_logging.set_mode(SessionLoggingMode.DETAILED)

        assert runtime_logging.emit_detailed_lazy(builder) is True
        assert builder_calls == 1
        assert stream.getvalue().splitlines() == [
            "[Logging] mode_changed requested=detailed effective=detailed previous=basic",
            "lazy detail",
        ]
    finally:
        runtime_logging.close()


@pytest.mark.asyncio
async def test_session_runtime_logging_emits_structured_events_without_changing_text() -> None:
    stream = io.StringIO()
    stream_handler = logging.StreamHandler(stream)
    runner = _ObservabilityRunner()
    sink = _StructuredObservabilitySink()
    runtime_logging = SessionRuntimeLoggingService(
        root_logger=logging.getLogger(f"test.runtime_logging.structured.root.{uuid4()}"),
        session_logger=logging.getLogger(f"test.runtime_logging.structured.session.{uuid4()}"),
        sinks=_SharedSinkBundle(
            stream_handler=stream_handler,
            file_handler=logging.NullHandler(),
            log_file="runtime.log",
        ),
        runtime_log_sink=sink,
        diagnostics_sink=sink,
        persisted_diagnostic_store=sink,
        observability_runner=runner,
    )

    try:
        runtime_logging.emit_basic("legacy basic text", level=logging.WARNING)
        assert runtime_logging.emit_detailed("hidden detailed text") is False
        runtime_logging.set_mode(SessionLoggingMode.DETAILED)
        assert runtime_logging.emit_detailed("legacy detailed text", level=logging.ERROR) is True
        runtime_logging.emit_persisted("legacy persisted text", level=logging.ERROR)
        await runner.drain()

        assert stream.getvalue().splitlines() == [
            "legacy basic text",
            "[Logging] mode_changed requested=detailed effective=detailed previous=basic",
            "legacy detailed text",
        ]
        assert [event.visibility for event in sink.runtime_events] == [
            DIAGNOSTIC_VISIBILITY_BASIC,
            DIAGNOSTIC_VISIBILITY_BASIC,
            DIAGNOSTIC_VISIBILITY_DETAILED,
        ]
        assert [event.severity for event in sink.runtime_events] == [
            SEVERITY_WARNING,
            SEVERITY_INFO,
            SEVERITY_ERROR,
        ]
        for event in sink.runtime_events:
            assert event.category == DIAGNOSTIC_CATEGORY_UNKNOWN
            assert event.content_policy == CONTENT_POLICY_METADATA_ONLY
            assert isinstance(event.correlation_id, str)
            assert event.message is None
            assert event.diagnostics is None
            assert event.fields["renderer"] == "legacy_text"
            field_text = repr(dict(event.fields))
            assert "legacy basic text" not in field_text
            assert "legacy detailed text" not in field_text

        assert [event.visibility for event in sink.diagnostic_events] == [
            DIAGNOSTIC_VISIBILITY_BASIC,
            DIAGNOSTIC_VISIBILITY_BASIC,
            DIAGNOSTIC_VISIBILITY_DETAILED,
        ]
        persisted = sink.persisted_records[-1]
        assert persisted.storage_key == "runtime.log"
        assert persisted.diagnostic.severity == SEVERITY_ERROR
        assert persisted.diagnostic.visibility == DIAGNOSTIC_VISIBILITY_DIAGNOSTIC_ONLY
        assert persisted.diagnostic.content_policy == CONTENT_POLICY_METADATA_ONLY
        assert "legacy persisted text" not in repr(dict(persisted.diagnostic.fields))
    finally:
        runtime_logging.close()


@pytest.mark.asyncio
async def test_session_runtime_logging_dispatches_provider_and_conversation_events() -> None:
    runner = _ObservabilityRunner()
    sink = _StructuredObservabilitySink()
    runtime_logging, _stream = _make_runtime_logging_capture()
    runtime_logging.configure_structured_observability(
        provider_observation_sink=sink,
        conversation_record_sink=sink,
        observability_runner=runner,
    )
    diagnostics = ErrorDiagnostics(
        component="provider.openrouter",
        operation="translate",
        code="provider.network",
        category=DIAGNOSTIC_CATEGORY_NETWORK,
        visibility=DIAGNOSTIC_VISIBILITY_BASIC,
        content_policy=CONTENT_POLICY_METADATA_ONLY,
        status_code=503,
        retry_after_ms=None,
        fields={"status_code": 503},
    )

    try:
        runtime_logging.observe_provider_operation(
            provider="openrouter",
            operation="translate",
            outcome="failure",
            severity=SEVERITY_ERROR,
            diagnostics=diagnostics,
            fields={"status_code": 503},
        )
        runtime_logging.record_conversation_observation(
            utterance_id="utt-1",
            speaker_channel="self",
            transcript_text="hello",
            translation_text="bonjour",
            source_language="en",
            target_language="fr",
            metadata={"transcript_len": 5, "translation_len": 7},
        )
        runtime_logging.record_conversation_observation(
            utterance_id="utt-1",
            speaker_channel="self",
            transcript_text="duplicate source",
            translation_text="duplicate translation",
            source_language="en",
            target_language="fr",
            metadata={"transcript_len": 16, "translation_len": 21},
        )
        await runner.drain()

        provider_event = sink.provider_events[-1]
        assert provider_event.provider == "openrouter"
        assert provider_event.operation == "translate"
        assert provider_event.outcome == "failure"
        assert provider_event.category == DIAGNOSTIC_CATEGORY_NETWORK
        assert provider_event.severity == SEVERITY_ERROR
        assert provider_event.visibility == DIAGNOSTIC_VISIBILITY_BASIC
        assert provider_event.content_policy == CONTENT_POLICY_METADATA_ONLY
        assert isinstance(provider_event.correlation_id, str)
        assert provider_event.diagnostics is diagnostics
        assert dict(provider_event.fields) == {"status_code": 503}
        assert len(sink.conversation_records) == 1

        conversation = sink.conversation_records[-1]
        assert conversation.utterance_id == "utt-1"
        assert conversation.speaker_channel == "self"
        assert conversation.transcript_text == "hello"
        assert conversation.translation_text == "bonjour"
        assert conversation.category == DIAGNOSTIC_CATEGORY_UNKNOWN
        assert conversation.severity == SEVERITY_INFO
        assert conversation.visibility == DIAGNOSTIC_VISIBILITY_BASIC
        assert conversation.content_policy == CONTENT_POLICY_RAW_USER_TEXT_ALLOWED
        assert isinstance(conversation.correlation_id, str)
        assert dict(conversation.metadata) == {"transcript_len": 5, "translation_len": 7}
    finally:
        runtime_logging.close()


@pytest.mark.asyncio
async def test_session_runtime_logging_redacts_diagnostics_before_structured_sinks() -> None:
    runner = _ObservabilityRunner()
    sink = _StructuredObservabilitySink()
    runtime_logging, _stream = _make_runtime_logging_capture()
    runtime_logging.configure_structured_observability(
        provider_observation_sink=sink,
        observability_runner=runner,
    )
    diagnostics = ErrorDiagnostics(
        component="provider.openrouter",
        operation="translate",
        code="provider.invalid_response",
        category=DIAGNOSTIC_CATEGORY_NETWORK,
        visibility=DIAGNOSTIC_VISIBILITY_DETAILED,
        content_policy="redacted",
        status_code=502,
        retry_after_ms=None,
        fields={
            "raw_response_body": "{'error':'bad','token':'provider-secret-structured'}",
            "provider": "openrouter",
        },
    )

    try:
        runtime_logging.observe_provider_operation(
            provider="openrouter",
            operation="translate",
            outcome="failure",
            severity=SEVERITY_ERROR,
            diagnostics=diagnostics,
            fields={"provider_response_body": "token=provider-secret-field"},
        )
        await runner.drain()

        assert len(sink.provider_events) == 1
        event = sink.provider_events[0]
        rendered = repr(event)
        assert "provider-secret-structured" not in rendered
        assert "provider-secret-field" not in rendered
        assert event.diagnostics is not None
        assert event.diagnostics.fields["raw_response_body"] == "[provider-response-body-redacted]"
        assert event.fields["provider_response_body"] == "[provider-response-body-redacted]"
    finally:
        runtime_logging.close()


def test_session_runtime_logging_ignores_provider_and_conversation_after_close() -> None:
    runner = _ObservabilityRunner()
    sink = _StructuredObservabilitySink()
    runtime_logging, _stream = _make_runtime_logging_capture()
    runtime_logging.configure_structured_observability(
        provider_observation_sink=sink,
        conversation_record_sink=sink,
        observability_runner=runner,
    )
    runtime_logging.close()

    runtime_logging.observe_provider_operation(
        provider="openrouter",
        operation="translate",
        outcome="success",
    )
    runtime_logging.record_conversation_observation(
        utterance_id="utt-after-close",
        speaker_channel="self",
        transcript_text="hello",
        translation_text=None,
        source_language="en",
        target_language=None,
    )

    assert runner.awaitables == []
    assert sink.provider_events == []
    assert sink.conversation_records == []


def test_session_runtime_logging_preserves_text_when_observability_construction_fails() -> None:
    runner = _ObservabilityRunner()
    runtime_logging, stream = _make_runtime_logging_capture()
    runtime_logging.configure_structured_observability(
        runtime_log_sink=_SynchronousFailingRuntimeLogSink(),
        observability_runner=runner,
    )

    try:
        runtime_logging.emit_basic("legacy text survives construction failure")

        assert stream.getvalue().splitlines() == ["legacy text survives construction failure"]
        assert runner.awaitables == []
    finally:
        runtime_logging.close()


def test_session_runtime_logging_preserves_text_when_observability_cleanup_fails() -> None:
    runner = _RaisingObservabilityRunner()
    sink = _CloseRaisingRuntimeLogSink()
    runtime_logging, stream = _make_runtime_logging_capture()
    runtime_logging.configure_structured_observability(
        runtime_log_sink=sink,
        observability_runner=runner,
    )

    try:
        runtime_logging.emit_basic("legacy text survives cleanup failure")

        assert stream.getvalue().splitlines() == ["legacy text survives cleanup failure"]
        assert runner.awaitables == [sink.awaitable]
        assert sink.awaitable.closed is True
    finally:
        runtime_logging.close()


@pytest.mark.asyncio
async def test_session_runtime_logging_observes_destination_result_in_basic() -> None:
    runtime_logging, stream = _make_runtime_logging_capture()
    decision = OutputRoutingDecision(
        decision="denied",
        route="self_chatbox",
        publication_id="peer-utterance-1",
        publication_kind="peer_subtitle",
        reason="peer_chatbox_denied",
        metadata={"channel": "peer", "unsafe": "secret peer transcript"},
    )

    try:
        await runtime_logging.observe_output_routing(decision)

        log_text = stream.getvalue()
        assert "[Output] destination_result" in log_text
        assert "decision=denied" in log_text
        assert "route=self_chatbox" in log_text
        assert "publication_kind=peer_subtitle" in log_text
        assert "reason=peer_chatbox_denied" in log_text
        assert "unsafe" not in log_text
        assert "secret peer transcript" not in log_text
    finally:
        runtime_logging.close()


def test_close_after_producers_stop_preserves_safe_first_cleanup_cause() -> None:
    runtime_logging, stream = _make_runtime_logging_capture()

    runtime_logging.close_after_producers_stop(
        cleanup_failures=(RuntimeError("secret cleanup detail"), ValueError("later")),
    )

    log_text = stream.getvalue()
    assert "[Lifecycle][Shutdown] logging_close" in log_text
    assert "cleanup_failure_count=2" in log_text
    assert "first_cleanup_exception_type=RuntimeError" in log_text
    assert "secret cleanup detail" not in log_text
