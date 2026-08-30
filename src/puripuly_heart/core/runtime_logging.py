from __future__ import annotations

import contextlib
import logging
import queue
from collections.abc import Awaitable, Mapping
from dataclasses import dataclass
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler
from pathlib import Path
from typing import Callable
from uuid import uuid4

from puripuly_heart.config.paths import user_config_dir
from puripuly_heart.core.diagnostic_validation import (
    DIAGNOSTIC_REDACTION_MARKER,
    DIAGNOSTIC_SINK_BASIC_LOGS,
    DIAGNOSTIC_SINK_DETAILED_LOGS,
    DIAGNOSTIC_SINK_PERSISTED_LOGS,
    DIAGNOSTIC_VALIDATION_STATUS_ACCEPTED,
    DiagnosticSink,
    redact_diagnostics_for_sink,
    redact_text_for_sink,
    validate_diagnostics_for_sink,
)
from puripuly_heart.core.messages import (
    CONTENT_POLICY_METADATA_ONLY,
    CONTENT_POLICY_RAW_USER_TEXT_ALLOWED,
    DIAGNOSTIC_CATEGORY_UNKNOWN,
    DIAGNOSTIC_VISIBILITY_BASIC,
    DIAGNOSTIC_VISIBILITY_DETAILED,
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
    SessionLoggingMode,
)
from puripuly_heart.core.output.models import OutputRoutingDecision

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
_QUEUE_HANDLER_REFCOUNT_ATTR = "_puripuly_heart_queue_refcount"
_QUEUE_HANDLER_QUEUE_ATTR = "_puripuly_heart_queue"


LOG_FORMAT = "%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s: %(message)s"
LOG_DATE_FORMAT = "%H:%M:%S"


def _main_formatter() -> logging.Formatter:
    return logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)


@dataclass(frozen=True, slots=True)
class LatencyTracePointContract:
    name: str
    timing_semantics: str
    acceptance_expectation: str


LATENCY_TRACE_POINT_CONTRACTS: dict[str, LatencyTracePointContract] = {
    "speech_end": LatencyTracePointContract(
        name="speech_end",
        timing_semantics="Shared latency zero boundary recorded when the channel owner accepts SpeechEnd for the utterance.",
        acceptance_expectation="Record the post-VAD SpeechEnd boundary; published e2e_ms adds the channel-specific VAD hangover for user-facing latency.",
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
    "self_chatbox_enqueue": LatencyTracePointContract(
        name="self_chatbox_enqueue",
        timing_semantics="Recorded when the output projection enqueues the final self output into ChatboxPaginator.",
        acceptance_expectation="This is the official self Basic latency end boundary because it is the final self output handoff point owned by the output projection.",
    ),
    "peer_overlay_first_emit": LatencyTracePointContract(
        name="peer_overlay_first_emit",
        timing_semantics="Recorded at the first peer overlay output emitted by the output projection: paired source+translation when translation succeeds, or source-only fallback when translation is unavailable, fails, or is cancelled.",
        acceptance_expectation="Use the first overlay_sink.emit call that carries peer-visible text for that peer logical turn; when translation is enabled and succeeds, wait for the paired source+translation overlay output.",
    ),
    "peer_overlay_first_render": LatencyTracePointContract(
        name="peer_overlay_first_render",
        timing_semantics="Recorded by the local overlay when the first local visible peer source or translation overlay output for the logical turn appears on this client.",
        acceptance_expectation="Emit once per peer logical turn after peer_overlay_first_emit at the first local visible peer source or translation overlay output for that turn; do not wait for lifecycle completion, cleanup, or any channel-owner terminal summary stage.",
    ),
}


def format_basic_latency_summary(
    *,
    channel: str,
    e2e_ms: int,
) -> str:
    parts = [
        f"channel={channel}",
        f"e2e_ms={e2e_ms}",
    ]
    return f"[Basic][Latency] {' '.join(parts)}"


def format_detailed_latency_trace(
    *,
    channel: str,
    utterance_id: str,
    stage: str,
    elapsed_ms: int,
    parent_utterance_id: str | None = None,
    target_index: int | None = None,
    target_language: str | None = None,
    turn_generation: int | None = None,
    turn_order: int | None = None,
) -> str:
    parts = [
        f"[Detailed][Latency] channel={channel}",
        f"utterance_id={utterance_id}",
        f"stage={stage}",
        f"elapsed_ms={elapsed_ms}",
    ]
    if target_language is not None:
        parts.extend(
            (
                f"parent_utterance_id={parent_utterance_id}",
                f"target_index={target_index}",
                f"target_language={target_language}",
                f"turn_generation={turn_generation}",
                f"turn_order={turn_order}",
            )
        )
    return " ".join(parts)


def format_detailed_latency_breakdown(
    *,
    channel: str,
    e2e_ms: int,
    speech_end_to_stt_final_ms: int | None = None,
    stt_final_to_final_output_ms: int | None = None,
) -> str:
    parts = [
        f"channel={channel}",
        f"e2e_ms={e2e_ms}",
    ]
    if speech_end_to_stt_final_ms is not None:
        parts.append(f"speech_end_to_stt_final_ms={speech_end_to_stt_final_ms}")
    if stt_final_to_final_output_ms is not None:
        parts.append(f"stt_final_to_final_output_ms={stt_final_to_final_output_ms}")
    return f"[Detailed][LatencyBreakdown] {' '.join(parts)}"


def format_translation_ready_for_output(
    *,
    channel: str,
    utterance_id: str,
    update_id: str,
    origin_wall_clock_ms: int | None,
    session_scope: str | None,
    source_text_hash: str | None,
    source_text_len: int | None,
    logical_turn_key: str | None,
    translation_len: int,
    elapsed_ms: int | None,
    parent_utterance_id: str | None = None,
    target_index: int | None = None,
    target_language: str | None = None,
    turn_generation: int | None = None,
    turn_order: int | None = None,
) -> str:
    parts = [
        "[Detailed][Translation] translation_ready_for_output",
        f"channel={channel}",
        f"utterance_id={utterance_id}",
        f"update_id={update_id}",
        f"origin_wall_clock_ms={origin_wall_clock_ms}",
        f"session_scope={session_scope}",
        f"source_text_hash={source_text_hash}",
        f"source_text_len={source_text_len}",
        f"logical_turn_key={logical_turn_key}",
        f"translation_len={translation_len}",
    ]
    if target_language is not None:
        parts.extend(
            (
                f"parent_utterance_id={parent_utterance_id}",
                f"target_index={target_index}",
                f"target_language={target_language}",
                f"turn_generation={turn_generation}",
                f"turn_order={turn_order}",
            )
        )
    if elapsed_ms is not None:
        parts.append(f"elapsed_ms={elapsed_ms}")
    return " ".join(parts)


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


class _DiagnosticRedactionFilter(logging.Filter):
    def __init__(self, sink: DiagnosticSink) -> None:
        super().__init__()
        self.sink = sink

    def filter(self, record: logging.LogRecord) -> bool:
        with contextlib.suppress(Exception):
            message = record.getMessage()
            safe_message = _redact_legacy_text_for_sink(message, self.sink)
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


@dataclass(slots=True)
class RuntimeLoggingSinks:
    stream_handler: logging.Handler
    file_handler: logging.Handler
    log_file: Path
    owner_logger: logging.Logger | None = None
    file_queue_handler: logging.Handler | None = None
    file_queue_listener: QueueListener | None = None
    file_queue: queue.Queue[logging.LogRecord] | None = None
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

    _remove_stale_main_file_queue_handlers(target_logger, log_file=log_file)
    existing_queue = _find_main_file_queue_handler(target_logger, log_file=log_file)
    if existing_queue is None:
        file_handler = _find_main_file_handler(target_logger, log_file=log_file)
        if file_handler is not None:
            with contextlib.suppress(Exception):
                target_logger.removeHandler(file_handler)
        else:
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=10 * 1024 * 1024,
                backupCount=1,
                encoding="utf-8",
            )
        file_handler.namer = _main_log_backup_namer
        file_handler.set_name(_MAIN_FILE_HANDLER_NAME)
        file_handler.setFormatter(_main_formatter())
        _ensure_redaction_filter(file_handler, DIAGNOSTIC_SINK_PERSISTED_LOGS)
        file_queue: queue.Queue[logging.LogRecord] = queue.Queue()
        file_queue_handler = QueueHandler(file_queue)
        file_queue_handler.set_name(_MAIN_FILE_QUEUE_HANDLER_NAME)
        _ensure_redaction_filter(file_queue_handler, DIAGNOSTIC_SINK_PERSISTED_LOGS)
        file_queue_listener = QueueListener(file_queue, file_handler, respect_handler_level=True)
        setattr(file_queue_handler, _QUEUE_HANDLER_LOG_FILE_ATTR, str(log_file.resolve()))
        setattr(file_queue_handler, _QUEUE_HANDLER_FILE_HANDLER_ATTR, file_handler)
        setattr(file_queue_handler, _QUEUE_HANDLER_LISTENER_ATTR, file_queue_listener)
        setattr(file_queue_handler, _QUEUE_HANDLER_CLOSED_ATTR, False)
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
        self._sinks = sinks or configure_main_logging(root_logger=self._root_logger)
        self._session_logger = session_logger or logging.getLogger(_new_session_logger_name())
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
        self._mode = SessionLoggingMode.BASIC
        self._closed = False

        file_output_handler = (
            getattr(self._sinks, "file_queue_handler", None) or self._sinks.file_handler
        )
        _ensure_redaction_filter(self._sinks.stream_handler, DIAGNOSTIC_SINK_BASIC_LOGS)
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
    def mode(self) -> SessionLoggingMode:
        return self._mode

    @property
    def log_file(self) -> Path:
        return self._sinks.log_file

    def set_mode(self, mode: SessionLoggingMode | str) -> None:
        self._mode = SessionLoggingMode(mode)

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
        safe_message = _redact_legacy_text_for_sink(message, DIAGNOSTIC_SINK_BASIC_LOGS)
        self._session_logger.log(level, safe_message)
        self._emit_structured_runtime_log(
            safe_message,
            level=level,
            visibility=DIAGNOSTIC_VISIBILITY_BASIC,
        )

    def emit_detailed(self, message: str, *, level: int = logging.INFO) -> bool:
        if self._closed:
            return False
        if self._mode is not SessionLoggingMode.DETAILED:
            return False
        safe_message = _redact_legacy_text_for_sink(message, DIAGNOSTIC_SINK_DETAILED_LOGS)
        self._session_logger.log(level, safe_message)
        self._emit_structured_runtime_log(
            safe_message,
            level=level,
            visibility=DIAGNOSTIC_VISIBILITY_DETAILED,
        )
        return True

    def emit_detailed_lazy(
        self,
        build_message: Callable[[], str],
        *,
        level: int = logging.INFO,
    ) -> bool:
        if self._closed:
            return False
        if self._mode is not SessionLoggingMode.DETAILED:
            return False
        message = _redact_legacy_text_for_sink(
            build_message(),
            DIAGNOSTIC_SINK_DETAILED_LOGS,
        )
        self._session_logger.log(level, message)
        self._emit_structured_runtime_log(
            message,
            level=level,
            visibility=DIAGNOSTIC_VISIBILITY_DETAILED,
        )
        return True

    async def observe_output_routing(self, decision: OutputRoutingDecision) -> None:
        with contextlib.suppress(Exception):
            self.emit_detailed_lazy(lambda: _format_output_routing_decision(decision))

    def emit_persisted(self, message: str, *, level: int = logging.INFO) -> None:
        if self._closed:
            return
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
        _join_pending_file_queue(self._sinks)
        self._sinks.file_handler.handle(record)
        with contextlib.suppress(Exception):
            self._sinks.file_handler.flush()
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
            diagnostics.visibility if diagnostics is not None else DIAGNOSTIC_VISIBILITY_DETAILED
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
        safe_metadata = _redact_observability_fields_for_sink(
            metadata or {},
            _sink_for_live_visibility(visibility),
            visibility=visibility,
            content_policy=content_policy,
        )
        record = ConversationRecord(
            utterance_id=utterance_id,
            speaker_channel=speaker_channel,
            transcript_text=transcript_text,
            translation_text=translation_text,
            source_language=source_language,
            target_language=target_language,
            metadata=safe_metadata,
            category=category,
            severity=severity,
            visibility=visibility,
            content_policy=content_policy,
            correlation_id=correlation_id or _new_correlation_id("conversation"),
        )
        self._dispatch_observability(
            self._conversation_record_sink,
            lambda sink: sink.record_conversation(record),
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


def _redact_legacy_text_for_sink(message: str, sink: DiagnosticSink) -> str:
    result = redact_text_for_sink(message, sink)
    if result.status == DIAGNOSTIC_VALIDATION_STATUS_ACCEPTED and result.text is not None:
        return result.text
    return DIAGNOSTIC_REDACTION_MARKER


def _sink_for_live_visibility(visibility: DiagnosticVisibility) -> DiagnosticSink:
    if visibility == DIAGNOSTIC_VISIBILITY_BASIC:
        return DIAGNOSTIC_SINK_BASIC_LOGS
    return DIAGNOSTIC_SINK_DETAILED_LOGS


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
    metadata_keys = ",".join(sorted(str(key) for key in decision.metadata))
    return (
        "[Detailed][OutputRouter] routing_decision "
        f"decision={decision.decision} "
        f"route={decision.route} "
        f"publication_id={decision.publication_id} "
        f"publication_kind={decision.publication_kind} "
        f"reason={decision.reason} "
        f"metadata_keys={metadata_keys}"
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


def _join_pending_file_queue(sinks: RuntimeLoggingSinks) -> None:
    file_queue_handler = getattr(sinks, "file_queue_handler", None)
    if file_queue_handler is None:
        return
    if getattr(file_queue_handler, _QUEUE_HANDLER_CLOSED_ATTR, False):
        return
    file_queue = getattr(sinks, "file_queue", None) or _main_file_queue_for_handler(
        file_queue_handler
    )
    if file_queue is not None:
        file_queue.join()


def _close_main_file_queue_handler(logger: logging.Logger, handler: logging.Handler) -> None:
    failures: list[Exception] = []
    try:
        logger.removeHandler(handler)
    except Exception as exc:
        failures.append(exc)
    setattr(handler, _QUEUE_HANDLER_CLOSED_ATTR, True)
    setattr(handler, _QUEUE_HANDLER_REFCOUNT_ATTR, 0)

    listener = getattr(handler, _QUEUE_HANDLER_LISTENER_ATTR, None)
    if isinstance(listener, QueueListener):
        try:
            listener.stop()
        except Exception as exc:
            failures.append(exc)

    file_handler = getattr(handler, _QUEUE_HANDLER_FILE_HANDLER_ATTR, None)
    if isinstance(file_handler, logging.Handler):
        try:
            _close_file_handler(file_handler)
        except Exception as exc:
            failures.append(exc)
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
        if getattr(handler, _QUEUE_HANDLER_LOG_FILE_ATTR, None) == expected_path and not getattr(
            handler, _QUEUE_HANDLER_CLOSED_ATTR, False
        ):
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
