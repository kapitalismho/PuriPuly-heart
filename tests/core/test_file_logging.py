"""Tests for RotatingFileHandler file logging."""

import inspect
import io
import logging
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from uuid import uuid4

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


class _RealtimeSink:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def append_log(self, line: str) -> None:
        self.lines.append(line)


class _RealtimeHandler(logging.Handler):
    def __init__(self, sink: _RealtimeSink) -> None:
        super().__init__()
        self._sink = sink
        self.setFormatter(logging.Formatter("%(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        self._sink.append_log(self.format(record))


def test_session_runtime_logging_service_routes_root_and_session_lines_to_shared_sinks(tmp_path):
    assert "sinks" in inspect.signature(SessionRuntimeLoggingService).parameters
    assert callable(getattr(SessionRuntimeLoggingService, "emit_basic", None))
    assert callable(getattr(SessionRuntimeLoggingService, "emit_detailed", None))

    stream = io.StringIO()
    log_file = tmp_path / "main.log"
    stream_handler = logging.StreamHandler(stream)
    stream_handler.setFormatter(logging.Formatter("%(message)s"))
    file_handler = RotatingFileHandler(log_file, maxBytes=4096, backupCount=0, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(message)s"))

    root_logger = logging.getLogger(f"test.runtime.root.{uuid4()}")
    root_logger.handlers.clear()
    root_logger.propagate = False

    session_logger = logging.getLogger(f"test.runtime.session.{uuid4()}")
    session_logger.handlers.clear()
    session_logger.propagate = False

    service = SessionRuntimeLoggingService(
        root_logger=root_logger,
        session_logger=session_logger,
        sinks=_SharedSinkBundle(
            stream_handler=stream_handler,
            file_handler=file_handler,
            log_file=log_file,
        ),
        ui_handler_factory=_RealtimeHandler,
    )

    sink = _RealtimeSink()
    service.attach_realtime_sink(sink)

    root_logger.info("root info")
    service.emit_basic("basic line")
    service.emit_detailed("hidden detail")
    service.set_mode(SessionLoggingMode.DETAILED)
    service.emit_detailed("visible detail")
    service.set_mode(SessionLoggingMode.BASIC)
    service.emit_detailed("hidden after reset")
    service.close()

    content = log_file.read_text(encoding="utf-8")
    assert "root info" in content
    assert "basic line" in content
    assert "visible detail" in content
    assert "hidden detail" not in content
    assert "hidden after reset" not in content
    assert sink.lines == ["root info", "basic line", "visible detail"]


def test_configure_main_logging_reuses_existing_root_stream_handler(tmp_path):
    root_logger = logging.getLogger(f"test.runtime.configure.{uuid4()}")
    root_logger.handlers.clear()
    root_logger.propagate = False

    existing_stream = logging.StreamHandler(io.StringIO())
    root_logger.addHandler(existing_stream)

    sinks = configure_main_logging(root_logger=root_logger, log_dir=tmp_path)

    try:
        stream_handlers = [
            handler
            for handler in root_logger.handlers
            if isinstance(handler, logging.StreamHandler)
            and not isinstance(handler, RotatingFileHandler)
        ]
        assert sinks.stream_handler is existing_stream
        assert stream_handlers == [existing_stream]
    finally:
        sinks.close()


def test_default_session_runtime_logging_services_do_not_share_session_logger(tmp_path):
    root_logger_one = logging.getLogger(f"test.runtime.default.root.one.{uuid4()}")
    root_logger_one.handlers.clear()
    root_logger_one.propagate = False
    root_logger_two = logging.getLogger(f"test.runtime.default.root.two.{uuid4()}")
    root_logger_two.handlers.clear()
    root_logger_two.propagate = False

    log_file_one = tmp_path / "session-one.log"
    log_file_two = tmp_path / "session-two.log"
    file_handler_one = RotatingFileHandler(
        log_file_one,
        maxBytes=4096,
        backupCount=0,
        encoding="utf-8",
    )
    file_handler_one.setFormatter(logging.Formatter("%(message)s"))
    file_handler_two = RotatingFileHandler(
        log_file_two,
        maxBytes=4096,
        backupCount=0,
        encoding="utf-8",
    )
    file_handler_two.setFormatter(logging.Formatter("%(message)s"))

    service_one = SessionRuntimeLoggingService(
        root_logger=root_logger_one,
        sinks=_SharedSinkBundle(
            stream_handler=logging.StreamHandler(io.StringIO()),
            file_handler=file_handler_one,
            log_file=log_file_one,
        ),
    )
    service_two = SessionRuntimeLoggingService(
        root_logger=root_logger_two,
        sinks=_SharedSinkBundle(
            stream_handler=logging.StreamHandler(io.StringIO()),
            file_handler=file_handler_two,
            log_file=log_file_two,
        ),
    )

    service_one.emit_basic("from one")
    service_two.emit_basic("from two")
    service_one.close()
    service_two.close()

    assert log_file_one.read_text(encoding="utf-8") == "from one\n"
    assert log_file_two.read_text(encoding="utf-8") == "from two\n"


def test_closing_runtime_logging_service_detaches_session_handlers(tmp_path):
    session_logger = logging.getLogger(f"test.runtime.shared.session.{uuid4()}")
    session_logger.handlers.clear()
    session_logger.propagate = False

    root_logger_one = logging.getLogger(f"test.runtime.shared.root.one.{uuid4()}")
    root_logger_one.handlers.clear()
    root_logger_one.propagate = False
    root_logger_two = logging.getLogger(f"test.runtime.shared.root.two.{uuid4()}")
    root_logger_two.handlers.clear()
    root_logger_two.propagate = False

    log_file_one = tmp_path / "closed-one.log"
    log_file_two = tmp_path / "closed-two.log"
    file_handler_one = RotatingFileHandler(
        log_file_one,
        maxBytes=4096,
        backupCount=0,
        encoding="utf-8",
    )
    file_handler_one.setFormatter(logging.Formatter("%(message)s"))
    file_handler_two = RotatingFileHandler(
        log_file_two,
        maxBytes=4096,
        backupCount=0,
        encoding="utf-8",
    )
    file_handler_two.setFormatter(logging.Formatter("%(message)s"))

    first_service = SessionRuntimeLoggingService(
        root_logger=root_logger_one,
        session_logger=session_logger,
        sinks=_SharedSinkBundle(
            stream_handler=logging.StreamHandler(io.StringIO()),
            file_handler=file_handler_one,
            log_file=log_file_one,
        ),
    )
    first_service.emit_basic("before close")
    first_service.close()

    second_service = SessionRuntimeLoggingService(
        root_logger=root_logger_two,
        session_logger=session_logger,
        sinks=_SharedSinkBundle(
            stream_handler=logging.StreamHandler(io.StringIO()),
            file_handler=file_handler_two,
            log_file=log_file_two,
        ),
    )
    second_service.emit_basic("after close")
    second_service.close()

    assert log_file_one.read_text(encoding="utf-8") == "before close\n"
    assert log_file_two.read_text(encoding="utf-8") == "after close\n"


def test_session_runtime_logging_service_persists_file_only_events_in_basic_mode(tmp_path):
    stream = io.StringIO()
    log_file = tmp_path / "persisted.log"
    stream_handler = logging.StreamHandler(stream)
    stream_handler.setFormatter(logging.Formatter("%(message)s"))
    file_handler = RotatingFileHandler(log_file, maxBytes=4096, backupCount=0, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(message)s"))

    root_logger = logging.getLogger(f"test.runtime.persisted.root.{uuid4()}")
    root_logger.handlers.clear()
    root_logger.propagate = False

    session_logger = logging.getLogger(f"test.runtime.persisted.session.{uuid4()}")
    session_logger.handlers.clear()
    session_logger.propagate = False

    service = SessionRuntimeLoggingService(
        root_logger=root_logger,
        session_logger=session_logger,
        sinks=_SharedSinkBundle(
            stream_handler=stream_handler,
            file_handler=file_handler,
            log_file=log_file,
        ),
        ui_handler_factory=_RealtimeHandler,
    )

    sink = _RealtimeSink()
    service.attach_realtime_sink(sink)
    service.emit_basic("basic line")
    service.emit_persisted(
        '[Persisted][Fallback] {"dual_bill_candidate": true, "event": "race_finished", '
        '"fallback_credential_source": "managed", "fallback_model": '
        '"google/gemini-2.5-flash-lite", "fallback_triggered": true, '
        '"primary_credential_source": "managed", "primary_model": '
        '"google/gemma-4-26b-a4b-it", "returned_source": "fallback", '
        '"winner": "fallback"}'
    )
    service.close()

    assert stream.getvalue().splitlines() == ["basic line"]
    assert sink.lines == ["basic line"]
    log_lines = log_file.read_text(encoding="utf-8").splitlines()
    assert log_lines[0] == "basic line"
    assert log_lines[1].startswith("[Persisted][Fallback] ")
    assert '"event": "race_finished"' in log_lines[1]
    assert '"primary_model": "google/gemma-4-26b-a4b-it"' in log_lines[1]
    assert '"fallback_model": "google/gemini-2.5-flash-lite"' in log_lines[1]
    assert '"winner": "fallback"' in log_lines[1]
