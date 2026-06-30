from __future__ import annotations

import contextlib
import logging
import queue
from dataclasses import dataclass
from enum import Enum
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler
from pathlib import Path
from typing import Callable, Protocol
from uuid import uuid4

from puripuly_heart.config.paths import user_config_dir

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

LATENCY_DOMINANT_STAGE_STT_FINALIZATION = "stt_finalization"
LATENCY_DOMINANT_STAGE_POST_STT_OUTPUT = "post_stt_output"
LATENCY_DOMINANT_STAGE_NORMAL = "normal"
LATENCY_DOMINANT_STAGE_THRESHOLD_MS = 1500
LATENCY_CAUSE_E2E_THRESHOLD_MS = 5000


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
        timing_semantics="Shared latency zero boundary recorded when the hub accepts SpeechEnd for the utterance.",
        acceptance_expectation="Record the post-VAD SpeechEnd boundary; published e2e_ms adds the channel-specific VAD hangover for user-facing latency.",
    ),
    "stt_final": LatencyTracePointContract(
        name="stt_final",
        timing_semantics="Recorded when the hub accepts the final STT transcript that will feed the final output path.",
        acceptance_expectation="Emit at most once per output path using the final transcript text that survives to output publication.",
    ),
    "llm_request_start": LatencyTracePointContract(
        name="llm_request_start",
        timing_semantics="Recorded immediately before the hub calls the translation provider for the output path.",
        acceptance_expectation="Use the request that contributes to the published output, not cancelled exploratory retries.",
    ),
    "llm_first_chunk": LatencyTracePointContract(
        name="llm_first_chunk",
        timing_semantics="Recorded when the hub receives the first streaming translation chunk for the output path.",
        acceptance_expectation="Emit only for streaming paths and only on the first chunk that belongs to the published output.",
    ),
    "llm_done": LatencyTracePointContract(
        name="llm_done",
        timing_semantics="Recorded when the hub has the completed translation text ready for publication.",
        acceptance_expectation="Use the completed translation that is about to be published, whether it came from a streaming or non-streaming provider.",
    ),
    "self_chatbox_enqueue": LatencyTracePointContract(
        name="self_chatbox_enqueue",
        timing_semantics="Recorded when the hub enqueues the final self output into ChatboxPaginator.",
        acceptance_expectation="This is the official self Basic latency end boundary because it is the final self output handoff point owned by the hub.",
    ),
    "peer_overlay_first_emit": LatencyTracePointContract(
        name="peer_overlay_first_emit",
        timing_semantics="Recorded at the first peer overlay output emitted by the hub: paired source+translation when translation succeeds, or source-only fallback when translation is unavailable, fails, or is cancelled.",
        acceptance_expectation="Use the first overlay_sink.emit call that carries peer-visible text for that peer logical turn; when translation is enabled and succeeds, wait for the paired source+translation overlay output.",
    ),
    "peer_overlay_first_render": LatencyTracePointContract(
        name="peer_overlay_first_render",
        timing_semantics="Recorded by the local overlay when the first local visible peer source or translation overlay output for the logical turn appears on this client.",
        acceptance_expectation="Emit once per peer logical turn after peer_overlay_first_emit at the first local visible peer source or translation overlay output for that turn; do not wait for lifecycle completion, cleanup, or any hub terminal summary stage.",
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
) -> str:
    return (
        f"[Detailed][Latency] channel={channel} utterance_id={utterance_id} "
        f"stage={stage} elapsed_ms={elapsed_ms}"
    )


def format_detailed_latency_breakdown(
    *,
    channel: str,
    e2e_ms: int,
    speech_end_to_stt_final_ms: int | None = None,
    stt_final_to_final_output_ms: int | None = None,
    dominant_stage: str | None = None,
) -> str:
    parts = [
        f"channel={channel}",
        f"e2e_ms={e2e_ms}",
    ]
    if speech_end_to_stt_final_ms is not None:
        parts.append(f"speech_end_to_stt_final_ms={speech_end_to_stt_final_ms}")
    if stt_final_to_final_output_ms is not None:
        parts.append(f"stt_final_to_final_output_ms={stt_final_to_final_output_ms}")
    if dominant_stage is not None:
        parts.append(f"dominant_stage={dominant_stage}")
    return f"[Detailed][LatencyBreakdown] {' '.join(parts)}"


def compute_latency_dominant_stage(
    *,
    speech_end_to_stt_final_ms: int | None,
    stt_final_to_final_output_ms: int | None,
) -> str:
    stt_final_ms = speech_end_to_stt_final_ms if speech_end_to_stt_final_ms is not None else 0
    post_stt_ms = stt_final_to_final_output_ms if stt_final_to_final_output_ms is not None else 0
    if stt_final_ms >= LATENCY_DOMINANT_STAGE_THRESHOLD_MS and stt_final_ms >= post_stt_ms:
        return LATENCY_DOMINANT_STAGE_STT_FINALIZATION
    if post_stt_ms >= LATENCY_DOMINANT_STAGE_THRESHOLD_MS:
        return LATENCY_DOMINANT_STAGE_POST_STT_OUTPUT
    return LATENCY_DOMINANT_STAGE_NORMAL


def format_latency_cause_metric(
    *,
    channel: str,
    e2e_ms: int,
    dominant_stage: str,
    speech_end_to_stt_final_ms: int | None = None,
    stt_final_to_final_output_ms: int | None = None,
) -> str:
    parts = [
        "[Metric] latency_cause",
        f"channel={channel}",
        f"e2e_ms={e2e_ms}",
        f"dominant_stage={dominant_stage}",
    ]
    if speech_end_to_stt_final_ms is not None:
        parts.append(f"speech_end_to_stt_final_ms={speech_end_to_stt_final_ms}")
    if stt_final_to_final_output_ms is not None:
        parts.append(f"stt_final_to_final_output_ms={stt_final_to_final_output_ms}")
    return " ".join(parts)


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
) -> str:
    parts = [
        "[Detailed][Hub] translation_ready_for_output",
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
    if elapsed_ms is not None:
        parts.append(f"elapsed_ms={elapsed_ms}")
    return " ".join(parts)


class RealtimeLogSink(Protocol):
    def append_log(self, line: str) -> None: ...


class SessionLoggingMode(str, Enum):
    BASIC = "basic"
    DETAILED = "detailed"


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
        file_queue: queue.Queue[logging.LogRecord] = queue.Queue()
        file_queue_handler = QueueHandler(file_queue)
        file_queue_handler.set_name(_MAIN_FILE_QUEUE_HANDLER_NAME)
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
        setattr(
            file_queue_handler,
            _QUEUE_HANDLER_REFCOUNT_ATTR,
            int(getattr(file_queue_handler, _QUEUE_HANDLER_REFCOUNT_ATTR, 1)) + 1,
        )
        file_handler.namer = _main_log_backup_namer
        file_handler.setFormatter(_main_formatter())

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
    ) -> None:
        self._root_logger = root_logger or logging.getLogger()
        self._owns_sinks = sinks is None
        self._sinks = sinks or configure_main_logging(root_logger=self._root_logger)
        self._session_logger = session_logger or logging.getLogger(_new_session_logger_name())
        self._root_logger.setLevel(logging.INFO)
        self._session_logger.setLevel(logging.INFO)
        self._session_logger.propagate = False
        self._ui_handler_factory = ui_handler_factory
        self._realtime_sink: RealtimeLogSink | None = None
        self._ui_handler: logging.Handler | None = None
        self._session_handlers: list[logging.Handler] = []
        self._mode = SessionLoggingMode.BASIC

        file_output_handler = (
            getattr(self._sinks, "file_queue_handler", None) or self._sinks.file_handler
        )
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

    def attach_realtime_sink(self, sink: RealtimeLogSink) -> None:
        if self._realtime_sink is sink:
            return

        self.detach_realtime_sink()
        self._realtime_sink = sink
        if self._ui_handler_factory is None:
            return

        handler = self._ui_handler_factory(sink)
        self._ui_handler = handler
        _ensure_handler(self._root_logger, handler)
        _ensure_handler(self._session_logger, handler)

    def detach_realtime_sink(self) -> None:
        if self._ui_handler is not None:
            with contextlib.suppress(Exception):
                self._root_logger.removeHandler(self._ui_handler)
            with contextlib.suppress(Exception):
                self._session_logger.removeHandler(self._ui_handler)
            with contextlib.suppress(Exception):
                self._ui_handler.close()
        self._realtime_sink = None
        self._ui_handler = None

    def emit_basic(self, message: str, *, level: int = logging.INFO) -> None:
        self._session_logger.log(level, message)

    def emit_detailed(self, message: str, *, level: int = logging.INFO) -> bool:
        if self._mode is not SessionLoggingMode.DETAILED:
            return False
        self._session_logger.log(level, message)
        return True

    def emit_detailed_lazy(
        self,
        build_message: Callable[[], str],
        *,
        level: int = logging.INFO,
    ) -> bool:
        if self._mode is not SessionLoggingMode.DETAILED:
            return False
        self._session_logger.log(level, build_message())
        return True

    def emit_persisted(self, message: str, *, level: int = logging.INFO) -> None:
        record = self._session_logger.makeRecord(
            self._session_logger.name,
            level,
            fn="",
            lno=0,
            msg=message,
            args=(),
            exc_info=None,
        )
        _join_pending_file_queue(self._sinks)
        self._sinks.file_handler.handle(record)
        with contextlib.suppress(Exception):
            self._sinks.file_handler.flush()

    def close(self) -> None:
        self.detach_realtime_sink()
        for handler in self._session_handlers:
            with contextlib.suppress(Exception):
                self._session_logger.removeHandler(handler)
        self._session_handlers.clear()
        if self._owns_sinks:
            self._sinks.close()


def _ensure_handler(logger: logging.Logger, handler: logging.Handler) -> bool:
    if handler not in logger.handlers:
        logger.addHandler(handler)
        return True
    return False


def _new_session_logger_name() -> str:
    return f"{_SESSION_LOGGER_NAME}.{uuid4()}"


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


def _close_file_handler(file_handler: logging.Handler) -> None:
    with contextlib.suppress(Exception):
        file_handler.flush()
    with contextlib.suppress(Exception):
        file_handler.close()


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
    with contextlib.suppress(Exception):
        logger.removeHandler(handler)
    setattr(handler, _QUEUE_HANDLER_CLOSED_ATTR, True)
    setattr(handler, _QUEUE_HANDLER_REFCOUNT_ATTR, 0)

    listener = getattr(handler, _QUEUE_HANDLER_LISTENER_ATTR, None)
    if isinstance(listener, QueueListener):
        with contextlib.suppress(Exception):
            listener.stop()

    file_handler = getattr(handler, _QUEUE_HANDLER_FILE_HANDLER_ATTR, None)
    if isinstance(file_handler, logging.Handler):
        _close_file_handler(file_handler)


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
