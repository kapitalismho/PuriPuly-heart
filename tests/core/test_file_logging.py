"""Tests for RotatingFileHandler file logging."""

import io
import json
import logging
from logging.handlers import RotatingFileHandler
from types import SimpleNamespace
from uuid import uuid4

from puripuly_heart.app.services.application_runtime_logging import ApplicationRuntimeLoggingOwner
from puripuly_heart.core.llm.latency import observe_attempt, observe_request
from puripuly_heart.core.messages import ErrorDiagnostics
from puripuly_heart.core.runtime.logging import RuntimeLoggingService
from puripuly_heart.core.runtime_logging import (
    RuntimeLoggingSinks,
    SessionRuntimeLoggingService,
    configure_main_logging,
)


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
        sinks=RuntimeLoggingSinks(
            stream_handler=stream_handler,
            file_handler=file_handler,
            log_file=log_file,
        ),
        ui_handler_factory=_RealtimeHandler,
    )

    sink = _RealtimeSink()
    service.attach_realtime_sink(sink)

    root_logger.info("root arbitrary prose")
    service.emit_basic("basic line")
    diagnostics = [
        ErrorDiagnostics(
            component="provider.llm",
            operation="translate",
            code=code,
            category="network",
            visibility="diagnostic_only",
            content_policy="metadata_only",
            status_code=None,
            retry_after_ms=None,
            fields={},
        )
        for code in ("provider.network", "provider.timeout")
    ]
    for diagnostic in diagnostics:
        service.emit_failure(diagnostic)
    service.close()

    content = log_file.read_text(encoding="utf-8")
    assert "root arbitrary prose" not in content
    assert "untrusted_record_redacted" in content
    assert "basic line" in content
    records = [
        json.loads(line.partition("[Failure] ")[2])
        for line in content.splitlines()
        if line.startswith("[Failure] {")
    ]
    assert [record["diagnostic_id"] for record in records] == [
        diagnostic.diagnostic_id for diagnostic in diagnostics
    ]
    assert [record["code"] for record in records] == ["provider.network", "provider.timeout"]
    assert sink.lines == ["basic line"]


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
        sinks=RuntimeLoggingSinks(
            stream_handler=logging.StreamHandler(io.StringIO()),
            file_handler=file_handler_one,
            log_file=log_file_one,
        ),
    )
    service_two = SessionRuntimeLoggingService(
        root_logger=root_logger_two,
        sinks=RuntimeLoggingSinks(
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
        sinks=RuntimeLoggingSinks(
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
        sinks=RuntimeLoggingSinks(
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
        sinks=RuntimeLoggingSinks(
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
    assert log_lines[1].startswith("[Logging] untrusted_record_redacted ")
    assert "race_finished" not in log_lines[1]
    assert "google/gemma" not in log_lines[1]


def test_translation_latency_uses_async_file_writer_without_ui_console_or_late_fallback(
    tmp_path, capsys
):
    stream = io.StringIO()
    root_logger = logging.getLogger(f"test.runtime.latency.root.{uuid4()}")
    root_logger.propagate = False
    root_logger.addHandler(logging.StreamHandler(stream))
    sinks = configure_main_logging(root_logger=root_logger, log_dir=tmp_path)
    session = SessionRuntimeLoggingService(
        root_logger=root_logger,
        session_logger=logging.getLogger(f"test.runtime.latency.session.{uuid4()}"),
        sinks=sinks,
        ui_handler_factory=_RealtimeHandler,
    )
    runtime_logging = RuntimeLoggingService(session_service=session)
    live_sink = _RealtimeSink()
    owner = ApplicationRuntimeLoggingOwner(
        presentation=SimpleNamespace(
            attach_runtime_log_sink=lambda service: service.attach_realtime_sink(live_sink),
        ),
        service_factory=lambda: runtime_logging,
        fallback_logger=root_logger,
    )

    def translate():
        with observe_request(
            sink=owner,
            utterance_id=uuid4(),
            channel="self",
            kind="manual",
            source_language="ko",
            target_language="en",
            provider_generation=1,
            input_chars=12,
            prompt_chars=30,
            context_chars=0,
        ):
            with observe_attempt(provider="openrouter", model="google/gemma-4"):
                pass

    try:
        owner.emit_basic("basic line")
        translate()
        assert not owner.emit_translation_latency(
            "[Diagnostic][LlmLatency] request_end request_id=1 prompt=secret"
        )
        runtime_logging.close()
        content = sinks.log_file.read_text(encoding="utf-8")
        translate()
        assert sinks.log_file.read_text(encoding="utf-8") == content
    finally:
        session.close()
        sinks.close()

    latency_lines = [line for line in content.splitlines() if "[LlmLatency]" in line]
    assert len(latency_lines) == 2
    assert "attempt_end" in latency_lines[0]
    assert "model=google/gemma-4" in latency_lines[0]
    assert "request_end" in latency_lines[1]
    assert "prompt=secret" not in content
    assert "[LlmLatency]" not in stream.getvalue()
    assert live_sink.lines == ["basic line"]
    captured = capsys.readouterr()
    assert "[LlmLatency]" not in captured.out + captured.err
