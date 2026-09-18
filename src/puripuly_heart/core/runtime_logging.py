from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import queue
import re
import time
from collections import deque
from collections.abc import Awaitable, Mapping, Sequence
from dataclasses import dataclass
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler
from pathlib import Path
from typing import Callable
from uuid import uuid4

from puripuly_heart.config.paths import user_config_dir
from puripuly_heart.core.diagnostic_validation import (
    DIAGNOSTIC_REDACTION_MARKER,
    DIAGNOSTIC_SINK_BASIC_LOGS,
    DIAGNOSTIC_SINK_PERSISTED_LOGS,
    DIAGNOSTIC_VALIDATION_STATUS_ACCEPTED,
    DiagnosticSink,
    redact_conversation_text_for_sink,
    redact_diagnostics_for_sink,
    redact_text_for_sink,
    validate_diagnostics_for_sink,
)
from puripuly_heart.core.language import get_language_info
from puripuly_heart.core.messages import (
    CONTENT_POLICY_METADATA_ONLY,
    CONTENT_POLICY_RAW_USER_TEXT_ALLOWED,
    DIAGNOSTIC_CATEGORY_UNKNOWN,
    DIAGNOSTIC_VISIBILITY_BASIC,
    DIAGNOSTIC_VISIBILITY_DIAGNOSTIC_ONLY,
    SEVERITY_ERROR,
    SEVERITY_INFO,
    SEVERITY_WARNING,
    ContentPolicy,
    DiagnosticCategory,
    DiagnosticFieldValue,
    DiagnosticVisibility,
    ErrorDiagnostics,
    Severity,
)
from puripuly_heart.core.observability import (
    ConversationRecord,
    ConversationRecordChannel,
    ConversationRecordSink,
    DiagnosticEvent,
    DiagnosticsSink,
    PersistedDiagnosticRecord,
    PersistedDiagnosticStore,
    ProviderObservationEvent,
    ProviderObservationOutcome,
    ProviderObservationSink,
    RealtimeLogSink,
    RuntimeLogEvent,
    RuntimeLogSink,
)
from puripuly_heart.core.output.models import OutputRoutingDecision
from puripuly_heart.core.runtime.logging import (
    LIVE_AUDIENCE_BASIC,
    LIVE_AUDIENCE_RECORD_ATTRIBUTE,
)

MAIN_LOG_FILENAME = "puripuly_heart.log"
MAIN_LOG_BACKUP_FILENAME = "puripuly_heart.backup.log"
_MAIN_STREAM_HANDLER_NAME = "puripuly_heart.main.stream"
_MAIN_FILE_HANDLER_NAME = "puripuly_heart.main.file"
_MAIN_FILE_QUEUE_HANDLER_NAME = "puripuly_heart.main.file.queue"
_SESSION_LOGGER_NAME = "puripuly_heart.runtime.session"
_QUEUE_HANDLER_LOG_FILE_ATTR = "_puripuly_heart_log_file"
_QUEUE_HANDLER_FILE_HANDLER_ATTR = "_puripuly_heart_file_handler"
_QUEUE_HANDLER_LISTENER_ATTR = "_puripuly_heart_queue_listener"
_QUEUE_HANDLER_CLOSED_ATTR = "_puripuly_heart_queue_closed"
_QUEUE_HANDLER_CLOSING_ATTR = "_puripuly_heart_queue_closing"
_QUEUE_HANDLER_REFCOUNT_ATTR = "_puripuly_heart_queue_refcount"
_QUEUE_HANDLER_QUEUE_ATTR = "_puripuly_heart_queue"
_CONTENT_CATEGORY_ATTR = "_puripuly_heart_content_category"
_CONVERSATION_CATEGORY = "accepted_conversation"
_CONTEXT_CATEGORY = "request_context"
_LIVE_AUDIENCE_ATTR = LIVE_AUDIENCE_RECORD_ATTRIBUTE
_LIVE_AUDIENCE_BASIC = LIVE_AUDIENCE_BASIC
_TERMINAL_RECORD_ATTR = "_puripuly_heart_terminal_record"
_MAIN_FILE_QUEUE_CAPACITY = 2048
_MAIN_FILE_PRIORITY_RESERVE = 64
_FILE_DRAIN_TIMEOUT_S = 2.0
_TERMINAL_ENQUEUE_TIMEOUT_S = 0.25
_FILE_BATCH_MAX_AGE_S = 1.0
_FILE_BATCH_MAX_BYTES = 64 * 1024
_FILE_BATCH_MAX_RECORDS = 128
_MAIN_LOG_MAX_BYTES = 20 * 1024 * 1024
_METADATA_PREFIX_RE = re.compile(r"^(?:\[[A-Za-z][A-Za-z0-9_-]*\])+(?:\s+|$)")
_METADATA_TOKEN_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.:-]*$")
_METADATA_FIELD_RE = re.compile(
    r"^[A-Za-z][A-Za-z0-9_.-]*=(?:[A-Za-z0-9_.:+,-]+|\[[A-Za-z0-9_.:+,-]*\])$"
)


LOG_FORMAT = "%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s: %(message)s"
LOG_DATE_FORMAT = "%H:%M:%S"


class _StableSessionNameFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        original_name = record.name
        if original_name.startswith(f"{_SESSION_LOGGER_NAME}."):
            record.name = _SESSION_LOGGER_NAME
        try:
            return super().format(record)
        finally:
            record.name = original_name


def _main_formatter() -> logging.Formatter:
    return _StableSessionNameFormatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)


@dataclass(frozen=True, slots=True)
class LatencyTracePointContract:
    name: str
    timing_semantics: str
    acceptance_expectation: str


LATENCY_TRACE_POINT_CONTRACTS: dict[str, LatencyTracePointContract] = {
    "last_speech": LatencyTracePointContract(
        name="last_speech",
        timing_semantics="Last observed source speech content boundary supplied by the channel owner.",
        acceptance_expectation="Use the owned source content frontier and observed trailing silence; leave unavailable origins unmeasured.",
    ),
    "speech_end": LatencyTracePointContract(
        name="speech_end",
        timing_semantics="Actual source utterance seal accepted by the channel owner.",
        acceptance_expectation="Retain this post-VAD boundary for the stage breakdown without using it as the end-to-end origin.",
    ),
    "stt_final": LatencyTracePointContract(
        name="stt_final",
        timing_semantics="Recorded when the channel owner accepts the final STT transcript that will feed the final output path.",
        acceptance_expectation="Emit at most once per output path using the final transcript text that survives to output publication.",
    ),
    "llm_request_start": LatencyTracePointContract(
        name="llm_request_start",
        timing_semantics="Recorded immediately before the translation request owner calls the provider for the output path.",
        acceptance_expectation="Use the request that contributes to the published output, not cancelled exploratory retries.",
    ),
    "llm_first_chunk": LatencyTracePointContract(
        name="llm_first_chunk",
        timing_semantics="Recorded when the translation request owner receives the first streaming chunk for the output path.",
        acceptance_expectation="Emit only for streaming paths and only on the first chunk that belongs to the published output.",
    ),
    "llm_done": LatencyTracePointContract(
        name="llm_done",
        timing_semantics="Recorded when the translation request owner has completed text ready for publication.",
        acceptance_expectation="Use the completed translation that is about to be published, whether it came from a streaming or non-streaming provider.",
    ),
    "self_chatbox_send": LatencyTracePointContract(
        name="self_chatbox_send",
        timing_semantics="Recorded after the first self chatbox page is successfully handed to the OSC UDP sender.",
        acceptance_expectation="This is a software send observation, not a VRChat or physical display acknowledgement.",
    ),
    "peer_overlay_applied": LatencyTracePointContract(
        name="peer_overlay_applied",
        timing_semantics="Recorded when the peer presenter returns an applied application receipt after logical pacing.",
        acceptance_expectation="This is presenter application receipt, not native renderer or physical HMD acknowledgement.",
    ),
    "dashboard_translation_applied": LatencyTracePointContract(
        name="dashboard_translation_applied",
        timing_semantics="Recorded after the dashboard translation destination accepts the update.",
        acceptance_expectation="This is dashboard application receipt, not a physical display acknowledgement.",
    ),
    "peer_overlay_first_render": LatencyTracePointContract(
        name="peer_overlay_first_render",
        timing_semantics="Reported by the native overlay's existing correlated first-render diagnostic.",
        acceptance_expectation="This is a local renderer observation, not compositor, physical HMD, or VRChat display acknowledgement.",
    ),
}


class RealtimeLogHandler(logging.Handler):
    def __init__(self, sink: RealtimeLogSink):
        super().__init__()
        self._sink = sink
        self.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
        )

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
            append_threadsafe = getattr(self._sink, "append_log_threadsafe", None)
            if callable(append_threadsafe):
                append_threadsafe(message)
            else:
                self._sink.append_log(message)
        except Exception:
            return


ObservabilityRunner = Callable[[Awaitable[None]], None]


def _is_metadata_only_text(message: str) -> bool:
    if "\n" in message or "\r" in message:
        return False
    prefix = _METADATA_PREFIX_RE.match(message)
    if prefix is None:
        return False
    tokens = message[prefix.end() :].split()
    if not tokens or _METADATA_TOKEN_RE.fullmatch(tokens[0]) is None:
        return False
    return all(_METADATA_FIELD_RE.fullmatch(token) is not None for token in tokens[1:])


def _metadata_only_log_envelope(record: logging.LogRecord, message: str) -> str:
    level_name = logging.getLevelName(record.levelno)
    return (
        "[Logging] untrusted_record_redacted "
        f"level={str(level_name).replace(' ', '_')} "
        f"logger_sha256={hashlib.sha256(record.name.encode('utf-8', errors='replace')).hexdigest()[:16]} "
        f"message_len={len(message)} "
        f"message_sha256={hashlib.sha256(message.encode('utf-8', errors='replace')).hexdigest()[:16]}"
    )


class _DiagnosticRedactionFilter(logging.Filter):
    def __init__(self, sink: DiagnosticSink) -> None:
        super().__init__()
        self.sink = sink

    def filter(self, record: logging.LogRecord) -> bool:
        if record.name == "httpx" or record.name.startswith("httpcore"):
            return False
        with contextlib.suppress(Exception):
            message = record.getMessage()
            category = getattr(record, _CONTENT_CATEGORY_ATTR, None)
            if category in {_CONVERSATION_CATEGORY, _CONTEXT_CATEGORY}:
                safe_message = message
            elif (
                self.sink == DIAGNOSTIC_SINK_BASIC_LOGS
                or getattr(record, _LIVE_AUDIENCE_ATTR, None) == _LIVE_AUDIENCE_BASIC
            ):
                safe_message = _redact_legacy_text_for_sink(message, self.sink)
            else:
                redacted = _redact_legacy_text_for_sink(message, self.sink)
                safe_message = (
                    redacted
                    if redacted == message and _is_metadata_only_text(message)
                    else _metadata_only_log_envelope(record, message)
                )
            if record.exc_info is not None or record.stack_info is not None:
                record.msg = safe_message
                record.args = ()
                record.exc_info = None
                record.exc_text = None
                record.stack_info = None
            elif safe_message != message:
                record.msg = safe_message
                record.args = ()
        return True


class _LiveAudienceFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return (
            getattr(record, _LIVE_AUDIENCE_ATTR, None) == _LIVE_AUDIENCE_BASIC
            or getattr(record, _CONTENT_CATEGORY_ATTR, None) == _CONVERSATION_CATEGORY
        )


class _BatchingRotatingFileHandler(RotatingFileHandler):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.oversized_records = 0
        self.delivery_failures = 0

    def emit_batch(self, records: list[logging.LogRecord]) -> None:
        if not records:
            return
        try:
            with self.lock:
                if self.stream is None:
                    self.stream = self._open()
                pending: list[str] = []
                pending_bytes = 0
                current_bytes = self._stream_size()
                for record in records:
                    if not self.filter(record):
                        continue
                    text = f"{self.format(record)}{self.terminator}"
                    encoded_size = len(text.encode(self.encoding or "utf-8", errors="replace"))
                    if self.maxBytes > 0 and encoded_size > self.maxBytes:
                        self.oversized_records += 1
                        continue
                    if (
                        self.maxBytes > 0
                        and current_bytes + pending_bytes > 0
                        and current_bytes + pending_bytes + encoded_size > self.maxBytes
                    ):
                        self._write_pending(pending)
                        pending = []
                        pending_bytes = 0
                        self.doRollover()
                        current_bytes = 0
                    pending.append(text)
                    pending_bytes += encoded_size
                self._write_pending(pending)
                if pending:
                    self.flush()
        except Exception:
            self.delivery_failures += len(records)

    def emit(self, record: logging.LogRecord) -> None:
        self.emit_batch([record])

    def _stream_size(self) -> int:
        if self.stream is None:
            return 0
        self.stream.seek(0, 2)
        return int(self.stream.tell())

    def _write_pending(self, pending: list[str]) -> None:
        if pending and self.stream is not None:
            self.stream.write("".join(pending))


def _estimated_record_bytes(record: logging.LogRecord) -> int:
    try:
        return len(record.getMessage().encode("utf-8", errors="replace")) + 96
    except Exception:
        return 96


def _message_is_definitely_oversized(message: str) -> bool:
    return len(message) >= _MAIN_LOG_MAX_BYTES


def _record_is_definitely_oversized(record: logging.LogRecord) -> bool:
    message = record.msg
    return (
        isinstance(message, str) and not record.args and _message_is_definitely_oversized(message)
    )


def _is_priority_file_record(record: logging.LogRecord) -> bool:
    return record.levelno >= logging.WARNING or bool(getattr(record, _TERMINAL_RECORD_ATTR, False))


class _BoundedFileQueue(queue.Queue[logging.LogRecord]):
    def admit(self, record: logging.LogRecord, *, force: bool = False) -> tuple[bool, int]:
        priority = _is_priority_file_record(record)
        with self.not_full:
            limit = self.maxsize if priority else self.maxsize - _MAIN_FILE_PRIORITY_RESERVE
            evicted = 0
            if self._qsize() >= limit and priority:
                for index, queued in enumerate(self.queue):
                    if not _is_priority_file_record(queued):
                        del self.queue[index]
                        self.unfinished_tasks -= 1
                        evicted = 1
                        if self.unfinished_tasks == 0:
                            self.all_tasks_done.notify_all()
                        self.not_full.notify()
                        break
            if self._qsize() >= limit and force and self._qsize():
                self._get()
                self.unfinished_tasks -= 1
                evicted += 1
                if self.unfinished_tasks == 0:
                    self.all_tasks_done.notify_all()
                self.not_full.notify()
            if self._qsize() >= limit:
                return False, evicted
            self._put(record)
            self.unfinished_tasks += 1
            self.not_empty.notify()
            return True, evicted


class _BoundedQueueHandler(QueueHandler):
    def __init__(self, file_queue: _BoundedFileQueue) -> None:
        super().__init__(file_queue)
        self.dropped_records = 0
        self.oversized_records = 0

    def handle(self, record: logging.LogRecord) -> bool:
        if getattr(self, _QUEUE_HANDLER_CLOSING_ATTR, False):
            self.dropped_records += 1
            return False
        if _record_is_definitely_oversized(record):
            self.oversized_records += 1
            return False
        return super().handle(record)

    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)

    def enqueue(self, record: logging.LogRecord) -> None:
        accepted, evicted = self.queue.admit(record)
        self.dropped_records += evicted
        if not accepted:
            self.dropped_records += 1


class _SafeQueueListener(QueueListener):
    def __init__(self, file_queue: queue.Queue[logging.LogRecord], *handlers: logging.Handler):
        super().__init__(file_queue, *handlers, respect_handler_level=True)
        self.delivery_failures = 0
        self.shutdown_drops = 0
        self.cleanup_failures: list[Exception] = []
        self.cleanup_complete = False
        self._owned_handlers = handlers
        self._sentinel_enqueued = False

    def enqueue_sentinel(self) -> None:
        if self._sentinel_enqueued:
            return
        while True:
            try:
                self.queue.put_nowait(self._sentinel)
                self._sentinel_enqueued = True
                return
            except queue.Full:
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    continue
                self.queue.task_done()
                self.shutdown_drops += 1

    def stop_bounded(self, *, timeout_s: float) -> bool:
        thread = self._thread
        if thread is None:
            return self.cleanup_complete
        if not thread.is_alive():
            self._thread = None
            return self.cleanup_complete
        self.enqueue_sentinel()
        thread.join(max(0.0, timeout_s))
        if thread.is_alive():
            return False
        self._thread = None
        return self.cleanup_complete

    def _monitor(self) -> None:
        batch: list[logging.LogRecord] = []
        batch_bytes = 0
        oldest_at = 0.0
        try:
            while True:
                timeout = None
                if batch:
                    timeout = max(
                        0.0,
                        _FILE_BATCH_MAX_AGE_S - (time.monotonic() - oldest_at),
                    )
                try:
                    record = (
                        self.dequeue(block=True)
                        if timeout is None
                        else self.queue.get(True, timeout)
                    )
                except queue.Empty:
                    self._deliver_batch(batch)
                    for _ in batch:
                        self.queue.task_done()
                    batch = []
                    batch_bytes = 0
                    continue
                if record is self._sentinel:
                    self._deliver_batch(batch)
                    for _ in batch:
                        self.queue.task_done()
                    self.queue.task_done()
                    return
                prepared = self.prepare(record)
                if not batch:
                    oldest_at = time.monotonic()
                batch.append(prepared)
                batch_bytes += _estimated_record_bytes(prepared)
                if (
                    len(batch) >= _FILE_BATCH_MAX_RECORDS
                    or batch_bytes >= _FILE_BATCH_MAX_BYTES
                    or prepared.levelno >= logging.WARNING
                ):
                    self._deliver_batch(batch)
                    for _ in batch:
                        self.queue.task_done()
                    batch = []
                    batch_bytes = 0
        finally:
            for handler in self._owned_handlers:
                try:
                    _close_file_handler(handler)
                except Exception as exc:
                    self.cleanup_failures.append(exc)
            self.cleanup_complete = True

    def _deliver_batch(self, records: list[logging.LogRecord]) -> None:
        if not records:
            return
        for handler in self.handlers:
            eligible = [
                record
                for record in records
                if not self.respect_handler_level or record.levelno >= handler.level
            ]
            if not eligible:
                continue
            try:
                emit_batch = getattr(handler, "emit_batch", None)
                if callable(emit_batch):
                    emit_batch(eligible)
                else:
                    for record in eligible:
                        handler.handle(record)
            except Exception:
                self.delivery_failures += len(eligible)


@dataclass(slots=True)
class RuntimeLoggingSinks:
    stream_handler: logging.Handler
    file_handler: logging.Handler
    log_file: Path
    owner_logger: logging.Logger | None = None
    file_queue_handler: logging.Handler | None = None
    file_queue_listener: QueueListener | None = None
    file_queue: queue.Queue[logging.LogRecord] | None = None
    abandoned_records: int = 0
    _closed: bool = False

    def close(self, *, force: bool = False) -> None:
        if self._closed and not force:
            return
        self._closed = True
        if self.owner_logger is not None and self.file_queue_handler is not None:
            _release_main_file_queue_handler(
                self.owner_logger,
                self.file_queue_handler,
                force=force,
            )
            return
        _close_file_handler(self.file_handler)


def default_main_log_file(*, log_dir: Path | None = None) -> Path:
    resolved_log_dir = log_dir or user_config_dir()
    resolved_log_dir.mkdir(parents=True, exist_ok=True)
    return resolved_log_dir / MAIN_LOG_FILENAME


def _main_log_backup_namer(default_name: str) -> str:
    backup_path = Path(default_name)
    if backup_path.name == f"{MAIN_LOG_FILENAME}.1":
        return str(backup_path.with_name(MAIN_LOG_BACKUP_FILENAME))
    return default_name


def configure_main_logging(
    *,
    root_logger: logging.Logger | None = None,
    log_dir: Path | None = None,
) -> RuntimeLoggingSinks:
    target_logger = root_logger or logging.getLogger()
    log_file = default_main_log_file(log_dir=log_dir)

    stream_handler = _find_main_stream_handler(target_logger)
    if stream_handler is None:
        stream_handler = logging.StreamHandler()
        stream_handler.set_name(_MAIN_STREAM_HANDLER_NAME)
        target_logger.addHandler(stream_handler)
    stream_handler.setFormatter(_main_formatter())
    _ensure_redaction_filter(stream_handler, DIAGNOSTIC_SINK_BASIC_LOGS)
    _ensure_live_audience_filter(stream_handler)

    _remove_stale_main_file_queue_handlers(target_logger, log_file=log_file)
    existing_queue = _find_main_file_queue_handler(target_logger, log_file=log_file)
    if existing_queue is None:
        file_handler = _find_main_file_handler(target_logger, log_file=log_file)
        if file_handler is not None:
            with contextlib.suppress(Exception):
                target_logger.removeHandler(file_handler)
            with contextlib.suppress(Exception):
                file_handler.close()
        file_handler = _BatchingRotatingFileHandler(
            log_file,
            maxBytes=_MAIN_LOG_MAX_BYTES,
            backupCount=1,
            encoding="utf-8",
        )
        file_handler.namer = _main_log_backup_namer
        file_handler.set_name(_MAIN_FILE_HANDLER_NAME)
        file_handler.setFormatter(_main_formatter())
        _ensure_redaction_filter(file_handler, DIAGNOSTIC_SINK_PERSISTED_LOGS)
        file_queue = _BoundedFileQueue(maxsize=_MAIN_FILE_QUEUE_CAPACITY)
        file_queue_handler = _BoundedQueueHandler(file_queue)
        file_queue_handler.set_name(_MAIN_FILE_QUEUE_HANDLER_NAME)
        _ensure_redaction_filter(file_queue_handler, DIAGNOSTIC_SINK_PERSISTED_LOGS)
        file_queue_listener = _SafeQueueListener(file_queue, file_handler)
        setattr(file_queue_handler, _QUEUE_HANDLER_LOG_FILE_ATTR, str(log_file.resolve()))
        setattr(file_queue_handler, _QUEUE_HANDLER_FILE_HANDLER_ATTR, file_handler)
        setattr(file_queue_handler, _QUEUE_HANDLER_LISTENER_ATTR, file_queue_listener)
        setattr(file_queue_handler, _QUEUE_HANDLER_CLOSED_ATTR, False)
        setattr(file_queue_handler, _QUEUE_HANDLER_CLOSING_ATTR, False)
        setattr(file_queue_handler, _QUEUE_HANDLER_REFCOUNT_ATTR, 1)
        setattr(file_queue_handler, _QUEUE_HANDLER_QUEUE_ATTR, file_queue)
        target_logger.addHandler(file_queue_handler)
        file_queue_listener.start()
    else:
        file_queue_handler, file_handler, file_queue_listener = existing_queue
        file_queue = _main_file_queue_for_handler(file_queue_handler)
        _ensure_redaction_filter(file_queue_handler, DIAGNOSTIC_SINK_PERSISTED_LOGS)
        setattr(
            file_queue_handler,
            _QUEUE_HANDLER_REFCOUNT_ATTR,
            int(getattr(file_queue_handler, _QUEUE_HANDLER_REFCOUNT_ATTR, 1)) + 1,
        )
        file_handler.namer = _main_log_backup_namer
        file_handler.setFormatter(_main_formatter())
        _ensure_redaction_filter(file_handler, DIAGNOSTIC_SINK_PERSISTED_LOGS)

    target_logger.setLevel(logging.INFO)
    return RuntimeLoggingSinks(
        stream_handler=stream_handler,
        file_handler=file_handler,
        log_file=log_file,
        owner_logger=target_logger,
        file_queue_handler=file_queue_handler,
        file_queue_listener=file_queue_listener,
        file_queue=file_queue,
    )


class SessionRuntimeLoggingService:
    def __init__(
        self,
        *,
        root_logger: logging.Logger | None = None,
        session_logger: logging.Logger | None = None,
        sinks: RuntimeLoggingSinks | None = None,
        ui_handler_factory: Callable[[RealtimeLogSink], logging.Handler] | None = None,
        runtime_log_sink: RuntimeLogSink | None = None,
        diagnostics_sink: DiagnosticsSink | None = None,
        provider_observation_sink: ProviderObservationSink | None = None,
        conversation_record_sink: ConversationRecordSink | None = None,
        persisted_diagnostic_store: PersistedDiagnosticStore | None = None,
        observability_runner: ObservabilityRunner | None = None,
    ) -> None:
        self._root_logger = root_logger or logging.getLogger()
        self._owns_sinks = sinks is None
        self._session_logger = session_logger or logging.getLogger(_new_session_logger_name())
        self._sinks = sinks or configure_main_logging(root_logger=self._root_logger)
        self._root_logger.setLevel(logging.INFO)
        self._session_logger.setLevel(logging.INFO)
        self._session_logger.propagate = False
        self._ui_handler_factory = ui_handler_factory
        self._runtime_log_sink = runtime_log_sink
        self._diagnostics_sink = diagnostics_sink
        self._provider_observation_sink = provider_observation_sink
        self._conversation_record_sink = conversation_record_sink
        self._persisted_diagnostic_store = persisted_diagnostic_store
        self._observability_runner = observability_runner
        self._realtime_sink: RealtimeLogSink | None = None
        self._ui_handler: logging.Handler | None = None
        self._session_handlers: list[logging.Handler] = []
        self._closed = False
        self._conversation_record_keys: set[tuple[str, str, int | None, int | None, str]] = set()
        self._conversation_record_order: deque[tuple[str, str, int | None, int | None, str]] = (
            deque()
        )
        self._request_context_keys: set[tuple[str, int | None]] = set()
        self._request_context_order: deque[tuple[str, int | None]] = deque()

        file_output_handler = (
            getattr(self._sinks, "file_queue_handler", None) or self._sinks.file_handler
        )
        _ensure_redaction_filter(self._sinks.stream_handler, DIAGNOSTIC_SINK_BASIC_LOGS)
        _ensure_live_audience_filter(self._sinks.stream_handler)
        _ensure_redaction_filter(file_output_handler, DIAGNOSTIC_SINK_PERSISTED_LOGS)
        if file_output_handler is not self._sinks.file_handler:
            _ensure_redaction_filter(self._sinks.file_handler, DIAGNOSTIC_SINK_PERSISTED_LOGS)
        _ensure_handler(self._root_logger, self._sinks.stream_handler)
        _ensure_handler(self._root_logger, file_output_handler)
        if _ensure_handler(self._session_logger, self._sinks.stream_handler):
            self._session_handlers.append(self._sinks.stream_handler)
        if _ensure_handler(self._session_logger, file_output_handler):
            self._session_handlers.append(file_output_handler)

    @property
    def log_file(self) -> Path:
        return self._sinks.log_file

    def configure_structured_observability(
        self,
        *,
        runtime_log_sink: RuntimeLogSink | None = None,
        diagnostics_sink: DiagnosticsSink | None = None,
        provider_observation_sink: ProviderObservationSink | None = None,
        conversation_record_sink: ConversationRecordSink | None = None,
        persisted_diagnostic_store: PersistedDiagnosticStore | None = None,
        observability_runner: ObservabilityRunner | None = None,
    ) -> None:
        if runtime_log_sink is not None:
            self._runtime_log_sink = runtime_log_sink
        if diagnostics_sink is not None:
            self._diagnostics_sink = diagnostics_sink
        if provider_observation_sink is not None:
            self._provider_observation_sink = provider_observation_sink
        if conversation_record_sink is not None:
            self._conversation_record_sink = conversation_record_sink
        if persisted_diagnostic_store is not None:
            self._persisted_diagnostic_store = persisted_diagnostic_store
        if observability_runner is not None:
            self._observability_runner = observability_runner

    def attach_realtime_sink(self, sink: RealtimeLogSink) -> None:
        if self._closed:
            return
        if self._realtime_sink is sink:
            return

        self.detach_realtime_sink()
        self._realtime_sink = sink
        if self._ui_handler_factory is None:
            return

        handler = self._ui_handler_factory(sink)
        _ensure_redaction_filter(handler, DIAGNOSTIC_SINK_BASIC_LOGS)
        _ensure_live_audience_filter(handler)
        self._ui_handler = handler
        _ensure_handler(self._root_logger, handler)
        _ensure_handler(self._session_logger, handler)

    def detach_realtime_sink(self) -> None:
        self._detach_realtime_sink(suppress_errors=True)

    def _detach_realtime_sink(self, *, suppress_errors: bool) -> None:
        failures: list[Exception] = []
        if self._ui_handler is not None:
            try:
                self._root_logger.removeHandler(self._ui_handler)
            except Exception as exc:
                if not suppress_errors:
                    failures.append(exc)
            try:
                self._session_logger.removeHandler(self._ui_handler)
            except Exception as exc:
                if not suppress_errors:
                    failures.append(exc)
            try:
                self._ui_handler.close()
            except Exception as exc:
                if not suppress_errors:
                    failures.append(exc)
        self._realtime_sink = None
        self._ui_handler = None
        _raise_close_failures("Runtime logging realtime sink close failed", failures)

    def emit_basic(self, message: str, *, level: int = logging.INFO) -> None:
        if self._closed:
            return
        if _message_is_definitely_oversized(message):
            _note_oversized_rejection(self._sinks)
            self._session_logger.warning(
                "[Logging] oversized_record_rejected audience=basic "
                f"character_count={len(message)} file_cap_bytes={_MAIN_LOG_MAX_BYTES}",
                extra={_LIVE_AUDIENCE_ATTR: _LIVE_AUDIENCE_BASIC},
            )
            return
        safe_message = _redact_legacy_text_for_sink(message, DIAGNOSTIC_SINK_BASIC_LOGS)
        self._session_logger.log(
            level,
            safe_message,
            extra={_LIVE_AUDIENCE_ATTR: _LIVE_AUDIENCE_BASIC},
        )
        self._emit_structured_runtime_log(
            safe_message,
            level=level,
            visibility=DIAGNOSTIC_VISIBILITY_BASIC,
        )

    def emit_diagnostic(self, message: str, *, level: int = logging.INFO) -> bool:
        if self._closed:
            return False
        if _message_is_definitely_oversized(message):
            _note_oversized_rejection(self._sinks)
            return False
        safe_message = _redact_legacy_text_for_sink(message, DIAGNOSTIC_SINK_PERSISTED_LOGS)
        record = self._session_logger.makeRecord(
            self._session_logger.name,
            level,
            fn="",
            lno=0,
            msg=safe_message,
            args=(),
            exc_info=None,
        )
        file_output_handler = (
            getattr(self._sinks, "file_queue_handler", None) or self._sinks.file_handler
        )
        file_output_handler.handle(record)
        self._persist_structured_diagnostic(safe_message, level=level)
        return True

    def emit_diagnostic_lazy(
        self,
        build_message: Callable[[], str],
        *,
        level: int = logging.INFO,
    ) -> bool:
        if self._closed:
            return False
        return self.emit_diagnostic(build_message(), level=level)

    def record_output_routing_decision(self, decision: OutputRoutingDecision) -> None:
        if self._closed:
            return
        self.emit_basic(_format_output_routing_decision(decision))

    async def observe_output_routing(self, decision: OutputRoutingDecision) -> None:
        self.record_output_routing_decision(decision)

    def emit_persisted(self, message: str, *, level: int = logging.INFO) -> None:
        if self._closed:
            return
        if _message_is_definitely_oversized(message):
            _note_oversized_rejection(self._sinks)
            return
        safe_message = _redact_legacy_text_for_sink(message, DIAGNOSTIC_SINK_PERSISTED_LOGS)
        loss_suffix = _file_delivery_loss_suffix(self._sinks)
        if loss_suffix:
            safe_message = f"{safe_message} {loss_suffix}"
        record = self._session_logger.makeRecord(
            self._session_logger.name,
            level,
            fn="",
            lno=0,
            msg=safe_message,
            args=(),
            exc_info=None,
        )
        _enqueue_terminal_file_record(self._sinks, record)
        self._persist_structured_diagnostic(safe_message, level=level)

    def observe_provider_operation(
        self,
        *,
        provider: str,
        operation: str,
        outcome: ProviderObservationOutcome,
        severity: Severity = SEVERITY_INFO,
        diagnostics: ErrorDiagnostics | None = None,
        fields: Mapping[str, DiagnosticFieldValue] | None = None,
        category: DiagnosticCategory | None = None,
        visibility: DiagnosticVisibility | None = None,
        content_policy: ContentPolicy | None = None,
        correlation_id: str | None = None,
    ) -> None:
        if self._closed:
            return
        event_visibility = visibility or (
            diagnostics.visibility
            if diagnostics is not None
            else DIAGNOSTIC_VISIBILITY_DIAGNOSTIC_ONLY
        )
        event_content_policy = content_policy or (
            diagnostics.content_policy if diagnostics is not None else CONTENT_POLICY_METADATA_ONLY
        )
        sink = _sink_for_live_visibility(event_visibility)
        safe_diagnostics = _redact_diagnostics_for_observability_sink(diagnostics, sink)
        safe_fields = _redact_observability_fields_for_sink(
            fields or {},
            sink,
            visibility=event_visibility,
            content_policy=event_content_policy,
        )
        event = ProviderObservationEvent(
            provider=provider,
            operation=operation,
            outcome=outcome,
            correlation_id=correlation_id or _new_correlation_id("provider"),
            diagnostics=safe_diagnostics,
            fields=safe_fields,
            category=category
            or (diagnostics.category if diagnostics is not None else DIAGNOSTIC_CATEGORY_UNKNOWN),
            severity=severity,
            visibility=event_visibility,
            content_policy=event_content_policy,
        )
        self._dispatch_observability(
            self._provider_observation_sink,
            lambda sink: sink.emit_provider_observation(event),
        )

    def record_conversation_observation(
        self,
        *,
        utterance_id: str,
        speaker_channel: ConversationRecordChannel,
        transcript_text: str | None,
        translation_text: str | None,
        source_language: str | None,
        target_language: str | None,
        metadata: Mapping[str, DiagnosticFieldValue] | None = None,
        category: DiagnosticCategory = DIAGNOSTIC_CATEGORY_UNKNOWN,
        severity: Severity = SEVERITY_INFO,
        visibility: DiagnosticVisibility = DIAGNOSTIC_VISIBILITY_BASIC,
        content_policy: ContentPolicy = CONTENT_POLICY_RAW_USER_TEXT_ALLOWED,
        correlation_id: str | None = None,
    ) -> None:
        if self._closed:
            return
        values = dict(metadata or {})
        target_index_value = values.get("target_index")
        target_index = target_index_value if isinstance(target_index_value, int) else None
        segment_index_value = values.get("segment_index")
        segment_index = segment_index_value if isinstance(segment_index_value, int) else None
        disposition = str(values.get("disposition") or "accepted")
        record_kind = "translation" if translation_text is not None else "source"
        key = (speaker_channel, utterance_id, segment_index, target_index, record_kind)
        if key in self._conversation_record_keys:
            return
        self._remember_conversation_key(key)
        safe_source, source_redacted = _safe_conversation_text(transcript_text)
        safe_translation, translation_redacted = _safe_conversation_text(translation_text)
        omission = "secret_redacted" if source_redacted or translation_redacted else "none"
        safe_metadata = _redact_observability_fields_for_sink(
            values,
            DIAGNOSTIC_SINK_PERSISTED_LOGS,
            visibility=visibility,
            content_policy=CONTENT_POLICY_METADATA_ONLY,
        )
        record = ConversationRecord(
            utterance_id=utterance_id,
            speaker_channel=speaker_channel,
            transcript_text=safe_source,
            translation_text=safe_translation,
            source_language=source_language,
            target_language=target_language,
            metadata=safe_metadata,
            category=category,
            severity=severity,
            visibility=visibility,
            content_policy=content_policy,
            correlation_id=correlation_id or f"conversation:{speaker_channel}:{utterance_id}",
        )
        line = _format_conversation_record(record, disposition=disposition, omission=omission)
        self._session_logger.log(
            _level_for_severity(severity),
            line,
            extra={_CONTENT_CATEGORY_ATTR: _CONVERSATION_CATEGORY},
        )
        self._append_realtime_conversation(record, disposition=disposition)
        self._dispatch_observability(
            self._conversation_record_sink,
            lambda sink: sink.record_conversation(record),
        )

    def record_request_context(
        self,
        *,
        utterance_id: str,
        context_texts: Sequence[str],
        segment_index: int | None = None,
    ) -> None:
        if self._closed:
            return
        key = (utterance_id, segment_index)
        if key in self._request_context_keys:
            return
        self._remember_request_context_key(key)
        if _message_is_definitely_oversized(utterance_id):
            _note_oversized_rejection(self._sinks)
            return
        safe_texts: list[str] = []
        omission = False
        for text in context_texts:
            safe_text, redacted = _safe_conversation_text(text)
            if safe_text is None:
                continue
            safe_texts.append(safe_text)
            omission = omission or redacted
        line = _format_request_context(
            utterance_id=utterance_id,
            texts=safe_texts,
            omission=omission,
        )
        if _message_is_definitely_oversized(line):
            _note_oversized_rejection(self._sinks)
            return
        record = self._session_logger.makeRecord(
            self._session_logger.name,
            logging.INFO,
            fn="",
            lno=0,
            msg=line,
            args=(),
            exc_info=None,
            extra={_CONTENT_CATEGORY_ATTR: _CONTEXT_CATEGORY},
        )
        file_output_handler = (
            getattr(self._sinks, "file_queue_handler", None) or self._sinks.file_handler
        )
        file_output_handler.handle(record)

    def _remember_conversation_key(
        self,
        key: tuple[str, str, int | None, int | None, str],
    ) -> None:
        self._conversation_record_keys.add(key)
        self._conversation_record_order.append(key)
        while len(self._conversation_record_order) > 4096:
            expired = self._conversation_record_order.popleft()
            self._conversation_record_keys.discard(expired)

    def _remember_request_context_key(self, key: tuple[str, int | None]) -> None:
        self._request_context_keys.add(key)
        self._request_context_order.append(key)
        while len(self._request_context_order) > 4096:
            expired = self._request_context_order.popleft()
            self._request_context_keys.discard(expired)

    def _append_realtime_conversation(
        self,
        record: ConversationRecord,
        *,
        disposition: str,
    ) -> None:
        sink = self._realtime_sink
        append = getattr(sink, "append_conversation_record", None)
        if not callable(append):
            return
        metadata = record.metadata
        try:
            append(
                source=str(
                    metadata.get("source")
                    or ("Listen" if record.speaker_channel == "peer" else "Mic")
                ),
                channel=record.speaker_channel,
                utterance_id=record.utterance_id,
                source_text=record.transcript_text,
                translated_text=record.translation_text,
                source_language=record.source_language,
                target_language=record.target_language,
                target_index=metadata.get("target_index"),
                disposition=disposition,
                origin_wall_clock_ms=metadata.get("origin_wall_clock_ms"),
                turn_kind=metadata.get("turn_kind"),
            )
        except Exception as exc:
            self.emit_basic(
                "[Logging] conversation_ui_delivery_failed "
                f"cause=unclassified exception_type={type(exc).__name__}",
                level=logging.ERROR,
            )

    def _emit_structured_runtime_log(
        self,
        message: str,
        *,
        level: int,
        visibility: DiagnosticVisibility,
    ) -> None:
        if self._observability_runner is None:
            return
        if self._runtime_log_sink is None and self._diagnostics_sink is None:
            return

        correlation_id = _new_correlation_id("runtime-log")
        fields = _legacy_text_observability_fields(
            message,
            level=level,
            visibility=visibility,
        )
        runtime_event = RuntimeLogEvent(
            category=DIAGNOSTIC_CATEGORY_UNKNOWN,
            severity=_severity_for_level(level),
            visibility=visibility,
            content_policy=CONTENT_POLICY_METADATA_ONLY,
            correlation_id=correlation_id,
            message=None,
            diagnostics=None,
            fields=fields,
        )
        diagnostic_event = DiagnosticEvent(
            category=runtime_event.category,
            severity=runtime_event.severity,
            visibility=runtime_event.visibility,
            content_policy=runtime_event.content_policy,
            correlation_id=runtime_event.correlation_id,
            diagnostics=None,
            fields=runtime_event.fields,
        )
        self._dispatch_observability(
            self._runtime_log_sink,
            lambda sink: sink.emit_runtime_log(runtime_event),
        )
        self._dispatch_observability(
            self._diagnostics_sink,
            lambda sink: sink.emit_diagnostic(diagnostic_event),
        )

    def _persist_structured_diagnostic(self, message: str, *, level: int) -> None:
        if self._persisted_diagnostic_store is None or self._observability_runner is None:
            return
        diagnostic = DiagnosticEvent(
            category=DIAGNOSTIC_CATEGORY_UNKNOWN,
            severity=_severity_for_level(level),
            visibility=DIAGNOSTIC_VISIBILITY_DIAGNOSTIC_ONLY,
            content_policy=CONTENT_POLICY_METADATA_ONLY,
            correlation_id=_new_correlation_id("persisted-log"),
            diagnostics=None,
            fields=_legacy_text_observability_fields(
                message,
                level=level,
                visibility=DIAGNOSTIC_VISIBILITY_DIAGNOSTIC_ONLY,
            ),
        )
        persisted = PersistedDiagnosticRecord(
            diagnostic=diagnostic,
            storage_key=_persisted_storage_key(self._sinks.log_file),
            metadata={"renderer": "legacy_text"},
        )
        self._dispatch_observability(
            self._persisted_diagnostic_store,
            lambda store: store.persist_diagnostic(persisted),
        )

    def _dispatch_observability(
        self,
        sink: object | None,
        build_awaitable: Callable[[object], Awaitable[None]],
    ) -> None:
        runner = self._observability_runner
        if sink is None or runner is None:
            return
        try:
            awaitable = build_awaitable(sink)
        except Exception:
            return
        try:
            runner(awaitable)
        except Exception:
            close = getattr(awaitable, "close", None)
            if callable(close):
                with contextlib.suppress(Exception):
                    close()

    def close(self) -> None:
        self._close(force_owned_sinks=False)

    def close_terminal_owner(self) -> None:
        self._close(force_owned_sinks=True)

    def close_after_producers_stop(
        self,
        *,
        cleanup_failures: tuple[BaseException, ...] = (),
    ) -> None:
        if self._closed:
            return
        first_cleanup = type(cleanup_failures[0]).__name__ if cleanup_failures else "none"
        self.emit_basic(
            "[Lifecycle][Shutdown] logging_close "
            f"cleanup_failure_count={len(cleanup_failures)} "
            f"first_cleanup_exception_type={first_cleanup}",
            level=logging.WARNING if cleanup_failures else logging.INFO,
        )
        self.close_terminal_owner()

    def _close(self, *, force_owned_sinks: bool) -> None:
        if self._closed:
            return
        self._closed = True
        failures: list[Exception] = []
        try:
            self._detach_realtime_sink(suppress_errors=False)
        except Exception as exc:
            failures.append(exc)
        for handler in self._session_handlers:
            try:
                self._session_logger.removeHandler(handler)
            except Exception as exc:
                failures.append(exc)
        self._session_handlers.clear()
        if self._owns_sinks:
            try:
                self._sinks.close(force=force_owned_sinks)
            except Exception as exc:
                failures.append(exc)
        _raise_close_failures("Runtime logging session close failed", failures)


def _ensure_handler(logger: logging.Logger, handler: logging.Handler) -> bool:
    if handler not in logger.handlers:
        logger.addHandler(handler)
        return True
    return False


def _ensure_redaction_filter(handler: logging.Handler, sink: DiagnosticSink) -> None:
    for existing in handler.filters:
        if isinstance(existing, _DiagnosticRedactionFilter) and existing.sink == sink:
            return
    handler.addFilter(_DiagnosticRedactionFilter(sink))


def _ensure_live_audience_filter(handler: logging.Handler) -> None:
    if any(isinstance(existing, _LiveAudienceFilter) for existing in handler.filters):
        return
    handler.filters.insert(0, _LiveAudienceFilter())


def _new_session_logger_name() -> str:
    return f"{_SESSION_LOGGER_NAME}.{uuid4()}"


def _new_correlation_id(prefix: str) -> str:
    return f"{prefix}-{uuid4()}"


def _severity_for_level(level: int) -> Severity:
    if level >= logging.ERROR:
        return SEVERITY_ERROR
    if level >= logging.WARNING:
        return SEVERITY_WARNING
    return SEVERITY_INFO


def _level_for_severity(severity: Severity) -> int:
    if severity == SEVERITY_ERROR:
        return logging.ERROR
    if severity == SEVERITY_WARNING:
        return logging.WARNING
    return logging.INFO


def _safe_conversation_text(value: str | None) -> tuple[str | None, bool]:
    if value is None:
        return None, False
    result = redact_conversation_text_for_sink(value, DIAGNOSTIC_SINK_PERSISTED_LOGS)
    return result.text, result.redacted


def _format_conversation_record(
    record: ConversationRecord,
    *,
    disposition: str,
    omission: str,
) -> str:
    metadata = record.metadata
    channel = "Self" if record.speaker_channel == "self" else "Peer"
    turn_kind = metadata.get("turn_kind")
    if turn_kind == "manual":
        origin = "Manual"
    else:
        source = metadata.get("source")
        origin = (
            str(source)
            if _safe_routing_token(source)
            else ("Listen" if channel == "Peer" else "Mic")
        )
    source_language = _language_display_name(record.source_language)
    target_language = _language_display_name(record.target_language)
    if record.translation_text is not None:
        header = f"[Conversation] {channel} · {origin} → {target_language}"
        parts = [header, f"Translation {json.dumps(record.translation_text, ensure_ascii=False)}"]
    else:
        header = f"[Conversation] {channel} · {origin} · Original ({source_language})"
        parts = [header]
        if record.transcript_text is not None:
            parts.append(f"Original {json.dumps(record.transcript_text, ensure_ascii=False)}")
    if disposition not in {"accepted", "translated"}:
        parts.append(f"Outcome {disposition.replace('_', ' ')}")
    failure_code = metadata.get("failure_code")
    if _safe_routing_token(failure_code):
        parts.append(f"Cause {str(failure_code).replace('_', ' ')}")
    if omission != "none":
        parts.append("Content redacted")
    return " · ".join(parts)


def _format_request_context(
    *,
    utterance_id: str,
    texts: Sequence[str],
    omission: bool,
) -> str:
    parts = [
        "[Context]",
        f"utterance_id={utterance_id}",
        f"context_count={len(texts)}",
    ]
    if texts:
        parts.append(f"texts={json.dumps(list(texts), ensure_ascii=False)}")
    if omission:
        parts.append("Content redacted")
    return " ".join(parts)


def _language_display_name(code: str | None) -> str:
    if not code:
        return "Unknown language"
    info = get_language_info(code)
    return info.name if info is not None else code


def _redact_legacy_text_for_sink(message: str, sink: DiagnosticSink) -> str:
    result = redact_text_for_sink(message, sink)
    if result.status == DIAGNOSTIC_VALIDATION_STATUS_ACCEPTED and result.text is not None:
        return result.text
    return DIAGNOSTIC_REDACTION_MARKER


def emit_basic_log(
    logger: logging.Logger,
    message: str,
    *args: object,
    level: int = logging.INFO,
) -> None:
    logger.log(
        level,
        message,
        *args,
        extra={_LIVE_AUDIENCE_ATTR: _LIVE_AUDIENCE_BASIC},
    )


def _sink_for_live_visibility(visibility: DiagnosticVisibility) -> DiagnosticSink:
    if visibility == DIAGNOSTIC_VISIBILITY_BASIC:
        return DIAGNOSTIC_SINK_BASIC_LOGS
    return DIAGNOSTIC_SINK_PERSISTED_LOGS


def _redact_diagnostics_for_observability_sink(
    diagnostics: ErrorDiagnostics | None,
    sink: DiagnosticSink,
) -> ErrorDiagnostics | None:
    if diagnostics is None:
        return None
    validation = validate_diagnostics_for_sink(diagnostics, sink)
    if validation.status == DIAGNOSTIC_VALIDATION_STATUS_ACCEPTED:
        return diagnostics
    result = redact_diagnostics_for_sink(diagnostics, sink)
    if result.status != DIAGNOSTIC_VALIDATION_STATUS_ACCEPTED:
        return None
    return result.diagnostics


def _redact_observability_fields_for_sink(
    fields: Mapping[str, DiagnosticFieldValue],
    sink: DiagnosticSink,
    *,
    visibility: DiagnosticVisibility,
    content_policy: ContentPolicy,
) -> Mapping[str, DiagnosticFieldValue]:
    if not fields:
        return {}
    diagnostics = ErrorDiagnostics(
        component="observability",
        operation="emit",
        code="observability.fields",
        category=DIAGNOSTIC_CATEGORY_UNKNOWN,
        visibility=visibility,
        content_policy=content_policy,
        status_code=None,
        retry_after_ms=None,
        fields=fields,
    )
    validation = validate_diagnostics_for_sink(diagnostics, sink)
    if validation.status == DIAGNOSTIC_VALIDATION_STATUS_ACCEPTED:
        return fields
    result = redact_diagnostics_for_sink(diagnostics, sink)
    if result.status != DIAGNOSTIC_VALIDATION_STATUS_ACCEPTED:
        return {}
    if result.diagnostics is None:
        return {}
    return result.diagnostics.fields


def _legacy_text_observability_fields(
    message: str,
    *,
    level: int,
    visibility: DiagnosticVisibility,
) -> Mapping[str, DiagnosticFieldValue]:
    level_name = logging.getLevelName(level)
    if not isinstance(level_name, str):
        level_name = str(level_name)
    return {
        "renderer": "legacy_text",
        "visibility": visibility,
        "levelno": int(level),
        "level_name": level_name,
        "text_len": len(message),
    }


def _persisted_storage_key(log_file: object) -> str | None:
    try:
        storage_path = Path(log_file)
    except TypeError:
        return None
    return storage_path.name or None


def _format_output_routing_decision(decision: OutputRoutingDecision) -> str:
    parts = [
        "[Output] destination_result",
        f"decision={decision.decision}",
        f"route={decision.route}",
        f"publication_id={decision.publication_id}",
        f"publication_kind={decision.publication_kind}",
        f"reason={decision.reason}",
    ]
    metadata = decision.metadata
    for name in ("stage", "outcome"):
        value = metadata.get(name)
        if _safe_routing_token(value):
            parts.append(f"{name}={value}")
    physical_ack = metadata.get("physical_ack")
    if isinstance(physical_ack, bool):
        parts.append(f"physical_ack={str(physical_ack).lower()}")
    wait_reason = metadata.get("wait_reason")
    if wait_reason in {"replacement_gate", "protected_rows"}:
        parts.append(f"wait_reason={wait_reason}")
    for name in ("handoff_wait_ms", "pending_batches"):
        value = metadata.get(name)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            parts.append(f"{name}={value}")
    return " ".join(parts)


def _safe_routing_token(value: object) -> bool:
    return (
        isinstance(value, str)
        and 0 < len(value) <= 64
        and all(character.isalnum() or character in "_-" for character in value)
    )


def _find_main_stream_handler(logger: logging.Logger) -> logging.Handler | None:
    fallback: logging.Handler | None = None
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(
            handler, RotatingFileHandler
        ):
            if handler.get_name() == _MAIN_STREAM_HANDLER_NAME:
                return handler
            fallback = fallback or handler
    if fallback is not None:
        fallback.set_name(_MAIN_STREAM_HANDLER_NAME)
    return fallback


def _find_main_file_handler(logger: logging.Logger, *, log_file: Path) -> logging.Handler | None:
    expected_path = str(log_file.resolve())
    for handler in logger.handlers:
        if not isinstance(handler, RotatingFileHandler):
            continue
        if handler.get_name() == _MAIN_FILE_HANDLER_NAME:
            return handler
        if str(Path(handler.baseFilename).resolve()) == expected_path:
            handler.set_name(_MAIN_FILE_HANDLER_NAME)
            return handler
    return None


def _raise_close_failures(message: str, failures: list[Exception]) -> None:
    if not failures:
        return
    if len(failures) == 1:
        raise failures[0]
    raise ExceptionGroup(message, failures)


def _close_file_handler(file_handler: logging.Handler) -> None:
    failures: list[Exception] = []
    try:
        file_handler.flush()
    except Exception as exc:
        failures.append(exc)
    try:
        file_handler.close()
    except Exception as exc:
        failures.append(exc)
    _raise_close_failures("Runtime logging file handler close failed", failures)


def _main_file_queue_for_handler(
    handler: logging.Handler,
) -> queue.Queue[logging.LogRecord] | None:
    file_queue = getattr(handler, _QUEUE_HANDLER_QUEUE_ATTR, None)
    if isinstance(file_queue, queue.Queue):
        return file_queue
    if isinstance(handler, QueueHandler) and isinstance(handler.queue, queue.Queue):
        setattr(handler, _QUEUE_HANDLER_QUEUE_ATTR, handler.queue)
        return handler.queue
    return None


def _join_pending_file_queue(
    sinks: RuntimeLoggingSinks,
    *,
    timeout_s: float | None = None,
) -> bool:
    file_queue_handler = getattr(sinks, "file_queue_handler", None)
    if file_queue_handler is None:
        return True
    if getattr(file_queue_handler, _QUEUE_HANDLER_CLOSED_ATTR, False):
        return True
    file_queue = getattr(sinks, "file_queue", None) or _main_file_queue_for_handler(
        file_queue_handler
    )
    if file_queue is None:
        return True
    timeout = _FILE_DRAIN_TIMEOUT_S if timeout_s is None else timeout_s
    deadline = time.monotonic() + max(0.0, timeout)
    while file_queue.unfinished_tasks and time.monotonic() < deadline:
        time.sleep(0.005)
    return file_queue.unfinished_tasks == 0


def _enqueue_terminal_file_record(
    sinks: RuntimeLoggingSinks,
    record: logging.LogRecord,
) -> None:
    handler = getattr(sinks, "file_queue_handler", None)
    file_queue = getattr(sinks, "file_queue", None)
    if handler is None or file_queue is None:
        sinks.file_handler.handle(record)
        return
    setattr(record, _TERMINAL_RECORD_ATTR, True)
    if isinstance(file_queue, _BoundedFileQueue):
        accepted, evicted = file_queue.admit(record, force=True)
        sinks.abandoned_records = int(getattr(sinks, "abandoned_records", 0)) + evicted
        if not accepted:
            sinks.abandoned_records += 1
        return
    deadline = time.monotonic() + _TERMINAL_ENQUEUE_TIMEOUT_S
    while True:
        try:
            file_queue.put(record, timeout=max(0.0, deadline - time.monotonic()))
            return
        except queue.Full:
            try:
                file_queue.get_nowait()
            except queue.Empty:
                continue
            file_queue.task_done()
            sinks.abandoned_records = int(getattr(sinks, "abandoned_records", 0)) + 1


def _file_delivery_loss_suffix(sinks: RuntimeLoggingSinks) -> str:
    dropped = int(getattr(sinks, "abandoned_records", 0))
    rejected = 0
    failures = 0
    handler = getattr(sinks, "file_queue_handler", None)
    if isinstance(handler, _BoundedQueueHandler):
        dropped += handler.dropped_records
        rejected += handler.oversized_records
    file_handler = getattr(sinks, "file_handler", None)
    if isinstance(file_handler, _BatchingRotatingFileHandler):
        rejected += file_handler.oversized_records
        failures += file_handler.delivery_failures
    listener = getattr(sinks, "file_queue_listener", None)
    if isinstance(listener, _SafeQueueListener):
        dropped += listener.shutdown_drops
        failures += listener.delivery_failures
    if not dropped and not rejected and not failures:
        return ""
    return (
        f"logging_delivery_dropped={dropped} "
        f"logging_oversized_rejected={rejected} "
        f"logging_delivery_failures={failures}"
    )


def _note_oversized_rejection(sinks: RuntimeLoggingSinks) -> None:
    handler = getattr(sinks, "file_queue_handler", None)
    if isinstance(handler, _BoundedQueueHandler):
        handler.oversized_records += 1
        return
    file_handler = getattr(sinks, "file_handler", None)
    if isinstance(file_handler, _BatchingRotatingFileHandler):
        file_handler.oversized_records += 1


def _close_main_file_queue_handler(logger: logging.Logger, handler: logging.Handler) -> None:
    failures: list[Exception] = []
    already_closing = bool(getattr(handler, _QUEUE_HANDLER_CLOSING_ATTR, False))
    setattr(handler, _QUEUE_HANDLER_REFCOUNT_ATTR, 0)
    setattr(handler, _QUEUE_HANDLER_CLOSING_ATTR, True)

    file_queue = _main_file_queue_for_handler(handler)
    listener = getattr(handler, _QUEUE_HANDLER_LISTENER_ATTR, None)
    file_handler = getattr(handler, _QUEUE_HANDLER_FILE_HANDLER_ATTR, None)
    sinks = RuntimeLoggingSinks(
        stream_handler=logging.NullHandler(),
        file_handler=file_handler,
        log_file=Path(getattr(handler, _QUEUE_HANDLER_LOG_FILE_ATTR)),
        file_queue_handler=handler,
        file_queue_listener=listener,
        file_queue=file_queue,
    )
    drained = _join_pending_file_queue(sinks)
    initial_delivery_failures = (
        file_handler.delivery_failures
        if isinstance(file_handler, _BatchingRotatingFileHandler)
        else 0
    )
    dropped = handler.dropped_records if isinstance(handler, _BoundedQueueHandler) else 0
    rejected = handler.oversized_records if isinstance(handler, _BoundedQueueHandler) else 0
    if isinstance(file_handler, _BatchingRotatingFileHandler):
        rejected += file_handler.oversized_records
    delivery_failures = initial_delivery_failures
    if isinstance(listener, _SafeQueueListener):
        dropped += listener.shutdown_drops
        delivery_failures += listener.delivery_failures
    if not already_closing and (dropped or rejected or delivery_failures or not drained):
        record = logger.makeRecord(
            _SESSION_LOGGER_NAME,
            logging.WARNING,
            fn="",
            lno=0,
            msg=(
                "[Lifecycle][Shutdown] logging_delivery_terminal "
                f"dropped={dropped} oversized_rejected={rejected} "
                f"delivery_failures={delivery_failures} drain_complete={str(drained).lower()}"
            ),
            args=(),
            exc_info=None,
        )
        _enqueue_terminal_file_record(sinks, record)

    listener_stopped = True
    if isinstance(listener, _SafeQueueListener):
        listener_stopped = listener.stop_bounded(timeout_s=_TERMINAL_ENQUEUE_TIMEOUT_S)
        failures.extend(listener.cleanup_failures)
    elif isinstance(listener, QueueListener) and drained:
        try:
            listener.stop()
        except Exception as exc:
            failures.append(exc)
            listener_stopped = False
        if listener_stopped and isinstance(file_handler, logging.Handler):
            try:
                _close_file_handler(file_handler)
            except Exception as exc:
                failures.append(exc)

    if listener_stopped:
        try:
            logger.removeHandler(handler)
        except Exception as exc:
            failures.append(exc)
        setattr(handler, _QUEUE_HANDLER_CLOSED_ATTR, True)
        if (
            isinstance(file_handler, _BatchingRotatingFileHandler)
            and file_handler.delivery_failures
        ):
            failures.append(
                RuntimeError(
                    "Runtime logging file delivery failed "
                    f"record_count={file_handler.delivery_failures}"
                )
            )
    else:
        failures.append(
            TimeoutError(
                "Runtime logging file listener stop timed out; cleanup remains listener-owned"
            )
        )
    _raise_close_failures("Runtime logging queue handler close failed", failures)


def _release_main_file_queue_handler(
    logger: logging.Logger,
    handler: logging.Handler,
    *,
    force: bool = False,
) -> None:
    if getattr(handler, _QUEUE_HANDLER_CLOSED_ATTR, False):
        return
    if force:
        _close_main_file_queue_handler(logger, handler)
        return
    refcount = int(getattr(handler, _QUEUE_HANDLER_REFCOUNT_ATTR, 1))
    remaining_refcount = max(0, refcount - 1)
    setattr(handler, _QUEUE_HANDLER_REFCOUNT_ATTR, remaining_refcount)
    if remaining_refcount > 0:
        return
    _close_main_file_queue_handler(logger, handler)


def _remove_stale_main_file_queue_handlers(logger: logging.Logger, *, log_file: Path) -> None:
    expected_path = str(log_file.resolve())
    for handler in list(logger.handlers):
        if handler.get_name() != _MAIN_FILE_QUEUE_HANDLER_NAME:
            continue
        same_file = getattr(handler, _QUEUE_HANDLER_LOG_FILE_ATTR, None) == expected_path
        usable = not getattr(handler, _QUEUE_HANDLER_CLOSED_ATTR, False) and not getattr(
            handler, _QUEUE_HANDLER_CLOSING_ATTR, False
        )
        if same_file and usable:
            continue
        _close_main_file_queue_handler(logger, handler)


def _find_main_file_queue_handler(
    logger: logging.Logger,
    *,
    log_file: Path,
) -> tuple[logging.Handler, logging.Handler, QueueListener] | None:
    expected_path = str(log_file.resolve())
    for handler in logger.handlers:
        if handler.get_name() != _MAIN_FILE_QUEUE_HANDLER_NAME:
            continue
        if getattr(handler, _QUEUE_HANDLER_CLOSED_ATTR, False):
            continue
        if getattr(handler, _QUEUE_HANDLER_LOG_FILE_ATTR, None) != expected_path:
            continue
        file_handler = getattr(handler, _QUEUE_HANDLER_FILE_HANDLER_ATTR, None)
        listener = getattr(handler, _QUEUE_HANDLER_LISTENER_ATTR, None)
        if isinstance(listener, QueueListener) and isinstance(file_handler, logging.Handler):
            return handler, file_handler, listener
    return None
