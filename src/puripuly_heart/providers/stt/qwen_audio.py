from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
import time
import uuid
from collections import deque
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, ClassVar, Literal

from puripuly_heart.core import network_clients
from puripuly_heart.core.audio.format import AudioCaptureSpan
from puripuly_heart.core.audio.ownership import SegmentTerminalOutcome
from puripuly_heart.core.diagnostics import current_diagnostic_operation, emit_failure
from puripuly_heart.core.messages import ErrorDiagnostics
from puripuly_heart.core.speech_boundary import SpeechBoundaryReason, boundary_wait_ms
from puripuly_heart.core.stt.backend import (
    LEGACY_STT_SESSION_PROJECTION,
    RecoverableSTTSessionError,
    STTBackend,
    STTBackendSession,
    STTBackendTranscriptEvent,
    STTNativeProvenance,
    STTProviderInputTerminal,
    STTProviderTurnIdentity,
    STTProviderTurnRequest,
    STTProviderTurnTerminal,
    STTProviderTurnUpdate,
    STTRecognitionUnit,
    STTSessionProjection,
)
from puripuly_heart.core.stt.diagnostics import recognition_failure_diagnostics
from puripuly_heart.core.stt.session_projection import STTSessionEventProjection
from puripuly_heart.core.stt.stream_input import STTStreamInputMap, STTStreamWrite
from puripuly_heart.domain.recognition import (
    NativeAudioCoverage,
    NativeTranscriptionEvidence,
    NativeWordEvidence,
    RecognitionStreamIdentity,
    RecognitionUnitIdentity,
)

logger = logging.getLogger(__name__)


def _safe_exception_class(exc: BaseException | None) -> str:
    if exc is None:
        return "none"
    cls = type(exc)
    if cls.__module__ == "builtins" or cls.__module__.startswith(("asyncio.", "websockets.")):
        return cls.__name__
    return "ExternalError"


def _safe_numeric_code(value: object, minimum: int, maximum: int) -> int | str:
    return value if type(value) is int and minimum <= value <= maximum else "none"


def _close_code(exc: BaseException | None, ws: Any) -> int | str:
    received = getattr(exc, "rcvd", None) if exc is not None else None
    code = getattr(received, "code", None)
    if code is None:
        code = getattr(exc, "code", None) if exc is not None else None
    if code is None and ws is not None:
        code = getattr(ws, "close_code", None)
    return _safe_numeric_code(code, 1000, 4999)


QWEN_AUDIO_MODEL = "qwen-audio-3.1-asr-flash-streaming"
QWEN_AUDIO_DEFAULT_ENDPOINT = "wss://dashscope.aliyuncs.com/api-ws/v1/inference"
QWEN_AUDIO_DEFAULT_HOTWORD_WEIGHT = 4
QWEN_AUDIO_LANGUAGE_HINTS_LIMIT = 4
QWEN_AUDIO_KEEPALIVE_INTERVAL_S = 15.0
QWEN_AUDIO_KEEPALIVE_SILENCE_MS = 100
QWEN_AUDIO_KEEPALIVE_TICK_S = 1.0
QWEN_PEER_SENTENCE_SILENCE_MS = 800
QWEN_PEER_NO_FINAL_S = 12
QWEN_PEER_TASK_MAX_S = 60
QWEN_PEER_MAX_SENTENCES = 256
QWEN_PEER_MAX_MAP_RECORDS = 4096
QWEN_PEER_MAX_METADATA_BYTES = 256 * 1024


class QwenAudioSessionState(str, Enum):
    CONNECTING = "connecting"
    TASK_ACTIVE = "task_active"
    READY = "task_active"
    FINISHING_TASK = "finishing_task"
    STARTING_NEXT_TASK = "starting_next_task"
    FAILED = "failed"
    CLOSING = "closing"


class QwenAudioProtocolError(RecoverableSTTSessionError):
    pass


class QwenAudioTaskFailedError(QwenAudioProtocolError):
    def __init__(self, error_code: str, error_message: str, *, hotwords_rejected: bool = False):
        self.error_code = error_code or "UNKNOWN"
        self.error_message = error_message or "Qwen Audio task failed"
        self.hotwords_rejected = hotwords_rejected
        label = "hotword parameters rejected" if hotwords_rejected else "task failed"
        super().__init__(f"Qwen Audio {label}: {self.error_code}: {self.error_message}")


WebSocketFactory = Callable[..., Awaitable[Any]]
HotwordInput = Mapping[str, int] | Sequence[str]


def _normalized_vocabulary(
    hotwords: HotwordInput | None,
    *,
    default_weight: int = QWEN_AUDIO_DEFAULT_HOTWORD_WEIGHT,
) -> dict[str, int]:
    if not isinstance(default_weight, int) or isinstance(default_weight, bool):
        default_weight = QWEN_AUDIO_DEFAULT_HOTWORD_WEIGHT
    if default_weight not in (1, 2, 3, 4, 5, 50):
        default_weight = QWEN_AUDIO_DEFAULT_HOTWORD_WEIGHT
    if hotwords is None:
        return {}
    result: dict[str, int] = {}
    if isinstance(hotwords, Mapping):
        source = hotwords.items()
    else:
        source = ((term, default_weight) for term in hotwords)
    for raw_term, raw_weight in source:
        if not isinstance(raw_term, str):
            continue
        term = raw_term.strip()
        if not term or term in result:
            continue
        weight = (
            raw_weight
            if isinstance(raw_weight, int) and not isinstance(raw_weight, bool)
            else default_weight
        )
        if weight not in (1, 2, 3, 4, 5, 50):
            weight = default_weight
        result[term] = weight
    return result


def _join_sentences(sentences: Sequence[str]) -> str:
    output = ""
    for sentence in sentences:
        text = sentence.strip()
        if not text:
            continue
        if not output:
            output = text
            continue
        previous = output[-1]
        first = text[0]
        if previous.isspace() or first.isspace() or first in ',.!?;:，。！？；：、)]}」』】》”’"':
            output += text
        elif ord(previous) >= 0x3000 or ord(first) >= 0x3000:
            output += text
        else:
            output += " " + text
    return output


@dataclass(slots=True)
class QwenAudioStreamingSTTBackend(STTBackend):
    api_key: str
    language_hints: tuple[str, ...] = ()
    model: str = QWEN_AUDIO_MODEL
    endpoint: str = QWEN_AUDIO_DEFAULT_ENDPOINT
    sample_rate_hz: int = 16000
    connect_timeout_s: float = 5.0
    task_start_timeout_s: float = 5.0
    task_finish_timeout_s: float = 5.0
    send_timeout_s: float = 5.0
    keepalive_interval_s: float = QWEN_AUDIO_KEEPALIVE_INTERVAL_S
    keepalive_silence_ms: int = QWEN_AUDIO_KEEPALIVE_SILENCE_MS
    hotwords: HotwordInput = ()
    hotword_weight: int = QWEN_AUDIO_DEFAULT_HOTWORD_WEIGHT
    websocket_factory: WebSocketFactory | None = None
    channel: Literal["self", "peer"] = "self"

    async def open_session(
        self,
        *,
        projection: STTSessionProjection = LEGACY_STT_SESSION_PROJECTION,
    ) -> STTBackendSession:
        if self.sample_rate_hz not in (8000, 16000):
            raise ValueError("sample_rate_hz must be 8000 or 16000")
        if not self.api_key:
            raise ValueError("api_key must be non-empty")
        if any(not hint for hint in self.language_hints):
            raise ValueError("language hints must be non-empty")
        if not self.endpoint:
            raise ValueError("endpoint must be non-empty")
        for name, value in (
            ("connect_timeout_s", self.connect_timeout_s),
            ("task_start_timeout_s", self.task_start_timeout_s),
            ("task_finish_timeout_s", self.task_finish_timeout_s),
            ("send_timeout_s", self.send_timeout_s),
            ("keepalive_interval_s", self.keepalive_interval_s),
            ("keepalive_silence_ms", self.keepalive_silence_ms),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be > 0")
        if int(self.sample_rate_hz * self.keepalive_silence_ms / 1000) * 2 <= 0:
            raise ValueError("keepalive_silence_ms must produce at least one audio frame")
        if self.channel == "peer" and projection.mode != "scoped":
            raise ValueError("Peer Qwen Audio requires scoped source ownership")
        session_type = _QwenPeerAudioSession if self.channel == "peer" else _QwenAudioSession
        session = session_type(
            api_key=self.api_key,
            language_hints=self.language_hints,
            model=self.model,
            endpoint=self.endpoint,
            sample_rate_hz=self.sample_rate_hz,
            connect_timeout_s=self.connect_timeout_s,
            task_start_timeout_s=self.task_start_timeout_s,
            task_finish_timeout_s=self.task_finish_timeout_s,
            send_timeout_s=self.send_timeout_s,
            keepalive_interval_s=self.keepalive_interval_s,
            keepalive_silence_ms=self.keepalive_silence_ms,
            hotwords=self.hotwords,
            hotword_weight=self.hotword_weight,
            websocket_factory=self.websocket_factory,
            projection=projection,
        )
        try:
            await session.start()
        except BaseException:
            with contextlib.suppress(BaseException):
                await session.abort_for_toggle_off()
            with contextlib.suppress(BaseException):
                await session.close()
            raise
        return session

    @staticmethod
    async def verify_api_key(
        api_key: str,
        *,
        endpoint: str = QWEN_AUDIO_DEFAULT_ENDPOINT,
        model: str = QWEN_AUDIO_MODEL,
    ) -> bool:
        if not api_key:
            return False
        backend = QwenAudioStreamingSTTBackend(api_key=api_key, endpoint=endpoint, model=model)
        session = await backend.open_session()
        try:
            await asyncio.wait_for(session.stop(), timeout=7.0)
            verified = session.task_finished_count == 1
            if not verified:
                diagnostics = recognition_failure_diagnostics(
                    session._failure,
                    provider="qwen_audio",
                    channel=None,
                    operation="probe",
                    reason="provider_not_ready",
                )
                emit_failure(
                    logger,
                    replace(
                        diagnostics,
                        **current_diagnostic_operation(),
                        fields={
                            **diagnostics.fields,
                            "verification_scope": "key_probe",
                            "recognition_readiness": "unobserved",
                        },
                    ),
                    stage="probe",
                )
            return verified
        finally:
            await session.close()


@dataclass(slots=True)
class _QwenAudioBoundary:
    sequence: int


@dataclass(slots=True)
class _QwenAudioSession(STTBackendSession):
    api_key: str
    language_hints: tuple[str, ...]
    model: str
    endpoint: str
    sample_rate_hz: int
    connect_timeout_s: float
    task_start_timeout_s: float
    task_finish_timeout_s: float
    send_timeout_s: float
    keepalive_interval_s: float
    keepalive_silence_ms: int
    hotwords: HotwordInput = ()
    hotword_weight: int = QWEN_AUDIO_DEFAULT_HOTWORD_WEIGHT
    websocket_factory: WebSocketFactory | None = None
    projection: STTSessionProjection = LEGACY_STT_SESSION_PROJECTION
    sentence_silence_ms: ClassVar[int] = 6000
    synthetic_keepalive: ClassVar[bool] = True

    _event_projection: STTSessionEventProjection = field(init=False, repr=False)
    _ws: Any = field(init=False, default=None, repr=False)
    _recv_task: asyncio.Task[None] | None = field(init=False, default=None, repr=False)
    _start_timeout_task: asyncio.Task[None] | None = field(init=False, default=None, repr=False)
    _finish_timeout_task: asyncio.Task[None] | None = field(init=False, default=None, repr=False)
    _start_future: asyncio.Future[None] | None = field(init=False, default=None, repr=False)
    _task_id: str | None = field(init=False, default=None, repr=False)
    _sentences: list[str] = field(init=False, default_factory=list, repr=False)
    _sentence_ids: set[str] = field(init=False, default_factory=set, repr=False)
    _audio_queue: deque[bytes] = field(init=False, default_factory=deque, repr=False)
    _post_boundary_audio_queue: deque[bytes] = field(init=False, default_factory=deque, repr=False)
    _pending_boundaries: deque[_QwenAudioBoundary] = field(
        init=False, default_factory=deque, repr=False
    )
    _active_boundary: _QwenAudioBoundary | None = field(init=False, default=None, repr=False)
    _next_boundary_sequence: int = field(init=False, default=1, repr=False)
    _state: QwenAudioSessionState = field(
        init=False, default=QwenAudioSessionState.CONNECTING, repr=False
    )
    _accept_terminals: bool = field(init=False, default=True, repr=False)
    _closing_requested: bool = field(init=False, default=False, repr=False)
    _events_closed: bool = field(init=False, default=False, repr=False)
    _drain_complete: asyncio.Event = field(init=False, repr=False)
    _flushing_audio: bool = field(init=False, default=False, repr=False)
    _boundary_requested: bool = field(init=False, default=False, repr=False)
    _flush_task: asyncio.Task[None] | None = field(init=False, default=None, repr=False)
    _deferred_finish_task: asyncio.Task[None] | None = field(init=False, default=None, repr=False)
    _keepalive_task: asyncio.Task[None] | None = field(init=False, default=None, repr=False)
    _last_audio_send_at: float | None = field(init=False, default=None, repr=False)
    _inflight_send_task: asyncio.Task[Any] | None = field(init=False, default=None, repr=False)
    _admission_lock: asyncio.Lock = field(init=False, repr=False)
    _send_lock: asyncio.Lock = field(init=False, repr=False)
    _loop: asyncio.AbstractEventLoop | None = field(init=False, default=None, repr=False)
    _failure: BaseException | None = field(init=False, default=None, repr=False)
    _scoped_task_id: str | None = field(init=False, default=None, repr=False)
    _task_finished_count: int = field(init=False, default=0, repr=False)
    _startup_at: float = field(init=False, default=0.0, repr=False)
    _stage_at: float = field(init=False, default=0.0, repr=False)
    _startup_stage: str = field(init=False, default="connect", repr=False)

    def __post_init__(self) -> None:
        self._event_projection = STTSessionEventProjection(self.projection)
        self._drain_complete = asyncio.Event()
        self._send_lock = asyncio.Lock()
        self._admission_lock = asyncio.Lock()

    @property
    def state(self) -> QwenAudioSessionState:
        return self._state

    @property
    def bridge_reset_preserves_vad(self) -> bool:
        return True

    @property
    def task_id(self) -> str | None:
        return self._task_id

    @property
    def task_finished_count(self) -> int:
        return self._task_finished_count

    def update_hotwords(self, hotwords: HotwordInput) -> None:
        self.hotwords = hotwords

    def _vocabulary(self) -> dict[str, int]:
        return _normalized_vocabulary(self.hotwords, default_weight=self.hotword_weight)

    def _log_failure(
        self, reason: str, *, cause: BaseException | None = None, code: object = None
    ) -> None:
        now = time.monotonic()
        response = getattr(cause, "response", None) if cause is not None else None
        status = getattr(response, "status_code", None)
        if status is None and cause is not None:
            status = getattr(cause, "status_code", None)
        logger.warning(
            "[QwenAudio] failure stage=%s reason=%s stage_ms=%d startup_ms=%d "
            "cause_class=%s nested_class=%s http_status=%s ws_close_code=%s provider_code=%s",
            self._startup_stage,
            reason,
            int((now - self._stage_at) * 1000) if self._stage_at else 0,
            int((now - self._startup_at) * 1000) if self._startup_at else 0,
            _safe_exception_class(cause),
            _safe_exception_class(cause.__cause__ if cause is not None else None),
            _safe_numeric_code(status, 100, 599),
            _close_code(cause, self._ws),
            _safe_numeric_code(code, 0, 999999),
        )

    async def start(self) -> None:
        self._startup_at = self._stage_at = time.monotonic()
        self._startup_stage = "connect"
        logger.info("[QwenAudio] startup stage=connect")
        self._loop = asyncio.get_running_loop()
        factory = self.websocket_factory
        if factory is None:
            factory = network_clients.external_websocket_connect
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            try:
                connection = factory(
                    self.endpoint,
                    additional_headers=headers,
                    ping_interval=None,
                    open_timeout=self.connect_timeout_s,
                )
            except TypeError:
                connection = factory(self.endpoint, headers=headers)
            self._ws = await connection
            self._startup_stage = "run_task_send"
            self._stage_at = time.monotonic()
            logger.info(
                "[QwenAudio] startup stage=connected stage_ms=%d",
                int((self._stage_at - self._startup_at) * 1000),
            )
        except Exception as exc:
            self._log_failure("connect_failed", cause=exc)
            self._state = QwenAudioSessionState.FAILED
            raise QwenAudioProtocolError("Qwen Audio connection failed") from exc
        self._recv_task = asyncio.create_task(self._recv_loop(), name="qwen-audio-recv")
        if self.synthetic_keepalive:
            self._keepalive_task = asyncio.create_task(
                self._keepalive_loop(), name="qwen-audio-keepalive"
            )
        try:
            await self._begin_task(initial=True)
        except BaseException:
            abandoned = self._start_future
            if abandoned is not None and abandoned.done() and not abandoned.cancelled():
                abandoned.exception()
            raise
        future = self._start_future
        if future is None:
            self._log_failure("start_future_missing")
            raise QwenAudioProtocolError("Qwen Audio task start was not requested")
        try:
            await asyncio.wait_for(asyncio.shield(future), timeout=self.task_start_timeout_s)
        except asyncio.TimeoutError as exc:
            await self._fail(
                QwenAudioProtocolError("Qwen Audio task-started timeout"),
                reason="task_start_timeout",
                cause=exc,
            )
            raise QwenAudioProtocolError("Qwen Audio task-started timeout") from exc
        except asyncio.CancelledError:
            raise
        except Exception:
            raise

    async def _begin_task(self, *, initial: bool = False) -> None:
        if not self._accept_terminals or self._ws is None:
            return
        task_id = str(uuid.uuid4())
        self._task_id = task_id
        self._sentences.clear()
        self._sentence_ids.clear()
        self._start_future = self._loop.create_future() if self._loop is not None else None
        self._state = (
            QwenAudioSessionState.CONNECTING
            if initial
            else QwenAudioSessionState.STARTING_NEXT_TASK
        )
        parameters: dict[str, object] = {
            "format": "pcm",
            "sample_rate": self.sample_rate_hz,
            "semantic_punctuation_enabled": False,
            "max_sentence_silence": self.sentence_silence_ms,
            "multi_threshold_mode_enabled": False,
            "heartbeat": True,
        }
        if self.language_hints:
            parameters["language_hints"] = list(self.language_hints)
        vocabulary = self._vocabulary()
        if vocabulary:
            parameters["vocabulary"] = vocabulary
        payload = {
            "header": {"action": "run-task", "task_id": task_id, "streaming": "duplex"},
            "payload": {
                "task_group": "audio",
                "task": "asr",
                "function": "recognition",
                "model": self.model,
                "parameters": parameters,
                "input": {},
            },
        }
        if initial:
            self._startup_stage = "run_task_send"
            self._stage_at = time.monotonic()
        try:
            await self._send_json(payload)
            if initial and self._state is not QwenAudioSessionState.FAILED:
                sent_at = time.monotonic()
                logger.info(
                    "[QwenAudio] startup stage=run_task_sent startup_ms=%d send_ms=%d",
                    int((sent_at - self._startup_at) * 1000),
                    int((sent_at - self._stage_at) * 1000),
                )
                if self._state is QwenAudioSessionState.CONNECTING:
                    self._startup_stage = "task_start_wait"
                    self._stage_at = sent_at
                elif self._state is QwenAudioSessionState.TASK_ACTIVE:
                    self._startup_stage = "active"
                    self._stage_at = sent_at
        except Exception as exc:
            await self._fail(
                QwenAudioProtocolError("Qwen Audio run-task send failed"),
                reason="run_task_send_failed",
                cause=exc,
            )
            raise
        if not initial:
            self._cancel_start_timeout()
            self._start_timeout_task = asyncio.create_task(
                self._wait_for_task_started(task_id), name="qwen-audio-task-start-timeout"
            )

    async def _wait_for_task_started(self, task_id: str) -> None:
        try:
            await asyncio.sleep(self.task_start_timeout_s)
            if self._task_id == task_id and self._state is QwenAudioSessionState.STARTING_NEXT_TASK:
                await self._fail(
                    QwenAudioProtocolError("Qwen Audio task-started timeout"),
                    reason="task_start_timeout",
                )
        except asyncio.CancelledError:
            return

    async def _send_json(self, payload: Mapping[str, object]) -> None:
        async with self._admission_lock:
            async with self._send_lock:
                await self._send_json_locked(payload)

    async def _send_json_locked(self, payload: Mapping[str, object]) -> None:
        await self._send_value(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))

    async def _send_value(self, value: str | bytes) -> None:
        ws = self._ws
        if ws is None:
            raise QwenAudioProtocolError("Qwen Audio socket is unavailable")
        send_task = asyncio.create_task(ws.send(value), name="qwen-audio-send")
        self._inflight_send_task = send_task
        try:
            await asyncio.wait_for(asyncio.shield(send_task), timeout=self.send_timeout_s)
        except asyncio.TimeoutError as exc:
            send_task.cancel()
            await asyncio.gather(send_task, return_exceptions=True)
            raise QwenAudioProtocolError("Qwen Audio send timeout") from exc
        except asyncio.CancelledError:
            send_task.cancel()
            await asyncio.gather(send_task, return_exceptions=True)
            if self._closing_requested or self._state in (
                QwenAudioSessionState.CLOSING,
                QwenAudioSessionState.FAILED,
            ):
                return
            raise

        finally:
            if self._inflight_send_task is send_task:
                self._inflight_send_task = None

    async def _send_finish(self, task_id: str) -> None:
        async with self._admission_lock:
            async with self._send_lock:
                await self._send_finish_locked(task_id)

    async def _send_finish_locked(self, task_id: str) -> None:
        await self._send_json_locked(
            {
                "header": {"action": "finish-task", "task_id": task_id, "streaming": "duplex"},
                "payload": {"input": {}},
            }
        )

    async def _recv_loop(self) -> None:
        try:
            while self._ws is not None:
                message = await self._ws.recv()
                if message is None:
                    break
                await self._handle_server_message(message)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if self._state not in (QwenAudioSessionState.CLOSING, QwenAudioSessionState.FAILED):
                await self._fail(
                    QwenAudioProtocolError("Qwen Audio socket failure"),
                    reason="receive_failed",
                    cause=exc,
                )
        finally:
            if (
                self._state not in (QwenAudioSessionState.CLOSING, QwenAudioSessionState.FAILED)
                and self._accept_terminals
            ):
                await self._fail(
                    QwenAudioProtocolError("Qwen Audio socket closed before task completion"),
                    reason="receive_eof",
                )
            if not self._events_closed and self._state is not QwenAudioSessionState.FAILED:
                self._put_event(None)

    @staticmethod
    def _message_dict(message: object) -> dict[str, Any] | None:
        if isinstance(message, Mapping):
            return dict(message)
        if isinstance(message, bytes):
            message = message.decode("utf-8", errors="replace")
        if not isinstance(message, str):
            return None
        try:
            value = json.loads(message)
        except TypeError, ValueError:
            return None
        return dict(value) if isinstance(value, Mapping) else None

    async def _handle_server_message(self, message: object) -> None:
        response = self._message_dict(message)
        if response is None:
            logger.debug("Qwen Audio ignored non-JSON provider message")
            return
        header = response.get("header")
        header = header if isinstance(header, Mapping) else {}
        event_type = str(header.get("event") or response.get("event") or "")
        event_task_id = str(header.get("task_id") or response.get("task_id") or "").strip()
        if event_type == "task-started":
            await self._handle_task_started(event_task_id)
        elif event_type == "result-generated":
            self._handle_result_generated(event_task_id, response)
        elif event_type == "task-finished":
            await self._handle_task_finished(event_task_id)
        elif event_type == "task-failed":
            await self._handle_task_failed(event_task_id, header)
        else:
            logger.debug("Qwen Audio ignored provider event")

    async def inject_event(self, event: Mapping[str, object]) -> None:
        await self._handle_server_message(event)

    async def _handle_task_started(self, event_task_id: str) -> None:
        if not event_task_id or event_task_id != self._task_id:
            logger.debug("Qwen Audio stale task-started ignored")
            return
        if self._state not in (
            QwenAudioSessionState.CONNECTING,
            QwenAudioSessionState.STARTING_NEXT_TASK,
        ):
            logger.debug("Qwen Audio duplicate task-started ignored")
            return
        self._cancel_start_timeout()
        if self._state is QwenAudioSessionState.CONNECTING:
            now = time.monotonic()
            ack_during_send = self._startup_stage == "run_task_send"
            logger.info(
                "[QwenAudio] startup stage=task_started startup_ms=%d wait_ms=%d "
                "ack_during_send=%s",
                int((now - self._startup_at) * 1000),
                0 if ack_during_send else int((now - self._stage_at) * 1000),
                str(ack_during_send).lower(),
            )
            if not ack_during_send:
                self._startup_stage = "active"
                self._stage_at = now
        self._state = QwenAudioSessionState.TASK_ACTIVE
        self._last_audio_send_at = time.monotonic()
        if self._post_boundary_audio_queue:
            self._audio_queue.extend(self._post_boundary_audio_queue)
            self._post_boundary_audio_queue.clear()
        if self._pending_boundaries and self._active_boundary is None:
            self._active_boundary = self._pending_boundaries.popleft()
            self._boundary_requested = True
        future = self._start_future
        if future is not None and not future.done():
            future.set_result(None)
        self._flushing_audio = True
        self._flush_task = asyncio.create_task(
            self._flush_queued_audio(),
            name="qwen-audio-queued-audio-flush",
        )

    async def _flush_queued_audio(self) -> None:
        current_task = asyncio.current_task()
        try:
            async with self._send_lock:
                while self._audio_queue and self._state is QwenAudioSessionState.TASK_ACTIVE:
                    await self._send_audio_now(self._audio_queue.popleft())
                self._flushing_audio = False
                if (
                    self._active_boundary is not None
                    and self._state is QwenAudioSessionState.TASK_ACTIVE
                ):
                    await self._finish_active_task_locked()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._fail(
                QwenAudioProtocolError("Qwen Audio queued audio send failed"),
                cause=exc,
                reason="qwen_write_failed",
            )
        finally:
            self._flushing_audio = False
            if self._flush_task is current_task:
                self._flush_task = None

    def _handle_result_generated(self, event_task_id: str, response: Mapping[str, object]) -> None:
        if event_task_id != self._task_id or self._state not in (
            QwenAudioSessionState.TASK_ACTIVE,
            QwenAudioSessionState.FINISHING_TASK,
        ):
            logger.debug("Qwen Audio stale result ignored")
            return
        payload = response.get("payload")
        payload = payload if isinstance(payload, Mapping) else {}
        output = payload.get("output")
        output = output if isinstance(output, Mapping) else {}
        sentence = output.get("sentence")
        sentence = sentence if isinstance(sentence, Mapping) else {}
        if bool(sentence.get("heartbeat")):
            return
        raw_sentence_id = sentence.get("sentence_id")
        sentence_id = str(raw_sentence_id).strip() if raw_sentence_id is not None else ""
        if not sentence_id or sentence_id == "0":
            return
        if not bool(sentence.get("sentence_end")):
            return
        if sentence_id in self._sentence_ids:
            logger.debug("Qwen Audio duplicate sentence ignored")
            return
        self._sentence_ids.add(sentence_id)
        text = str(sentence.get("text") or "").strip()
        if text:
            self._sentences.append(text)
            identity = self._event_projection.active_identity
            if identity is not None and self._scoped_task_id == event_task_id:
                sequence = self._event_projection.next_update_sequence(identity)
                if sequence is not None:
                    self._event_projection.put_update(
                        STTProviderTurnUpdate(
                            identity=identity,
                            sequence=sequence,
                            stability="stable",
                            assembly="replace",
                            text=_join_sentences(self._sentences),
                            provenance=STTNativeProvenance(
                                native_event_id=sentence_id,
                                native_task_id=event_task_id,
                                barrier="sentence_end",
                            ),
                        )
                    )

    async def _finish_after_flush(self, event_task_id: str) -> None:
        try:
            flush_task = self._flush_task
            if flush_task is not None and flush_task is not asyncio.current_task():
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await flush_task
            if self._state not in (QwenAudioSessionState.CLOSING, QwenAudioSessionState.FAILED):
                await self._handle_task_finished(event_task_id)
        finally:
            if self._deferred_finish_task is asyncio.current_task():
                self._deferred_finish_task = None

    async def _handle_task_finished(self, event_task_id: str) -> None:
        if event_task_id != self._task_id:
            logger.debug("Qwen Audio stale task-finished ignored")
            return
        if (
            self._state is QwenAudioSessionState.TASK_ACTIVE
            and self._active_boundary is not None
            and self._flush_task is not None
            and not self._flush_task.done()
        ):
            if self._deferred_finish_task is None:
                self._deferred_finish_task = asyncio.create_task(
                    self._finish_after_flush(event_task_id),
                    name="qwen-audio-finish-after-flush",
                )
            return
        if self._state is not QwenAudioSessionState.FINISHING_TASK or self._active_boundary is None:
            logger.debug("Qwen Audio duplicate task-finished ignored")
            return
        self._task_finished_count += 1
        self._cancel_finish_timeout()
        self._cancel_start_timeout()
        terminal_text = _join_sentences(self._sentences)
        if self._accept_terminals and self._event_projection.is_legacy:
            self._put_event(STTBackendTranscriptEvent(text=terminal_text, is_final=True))
        identity = self._event_projection.active_identity
        if identity is not None and self._scoped_task_id == event_task_id:
            self._terminalize_scoped(
                identity,
                outcome="final" if terminal_text else "empty",
                text=terminal_text,
                provenance=STTNativeProvenance(
                    native_task_id=event_task_id,
                    barrier="task-finished",
                ),
            )
        self._active_boundary = None
        self._boundary_requested = False
        self._task_id = None
        self._sentences.clear()
        self._sentence_ids.clear()
        if self._closing_requested and not self._pending_boundaries:
            self._state = QwenAudioSessionState.CLOSING
            await self._close_socket()
            self._drain_complete.set()
            return
        try:
            await self._begin_task()
        except Exception:
            return

    async def _handle_task_failed(self, event_task_id: str, header: Mapping[str, object]) -> None:
        if event_task_id != self._task_id:
            logger.debug("Qwen Audio stale task-failed ignored")
            return
        error_code = str(header.get("error_code") or "UNKNOWN")
        error_message = str(header.get("error_message") or "Qwen Audio task failed")
        normalized = f"{error_code} {error_message}".casefold()
        hotwords_rejected = bool(self._vocabulary()) and any(
            token in normalized for token in ("vocabulary", "hotword", "workspace", "sub-workspace")
        )
        await self._fail(
            QwenAudioTaskFailedError(
                error_code,
                error_message,
                hotwords_rejected=hotwords_rejected,
            ),
            reason="provider_task_failed",
            code=header.get("error_code"),
        )

    async def _finish_active_task(self) -> None:
        async with self._send_lock:
            await self._finish_active_task_locked()

    async def _finish_active_task_locked(self) -> None:
        if (
            self._state is not QwenAudioSessionState.TASK_ACTIVE
            or self._active_boundary is None
            or self._task_id is None
        ):
            return
        task_id = self._task_id
        self._state = QwenAudioSessionState.FINISHING_TASK
        try:
            await self._send_finish_locked(task_id)
            self._finish_timeout_task = asyncio.create_task(
                self._wait_for_task_finished(task_id), name="qwen-audio-task-finish-timeout"
            )
        except Exception as exc:
            await self._fail(
                QwenAudioProtocolError("Qwen Audio finish-task send failed"),
                cause=exc,
                reason="qwen_write_failed",
            )

    async def _send_audio_now(self, pcm16le: bytes) -> None:
        if not pcm16le:
            return
        await self._send_value(pcm16le)
        self._last_audio_send_at = time.monotonic()

    async def _keepalive_loop(self) -> None:
        try:
            while not self._closing_requested:
                await asyncio.sleep(min(QWEN_AUDIO_KEEPALIVE_TICK_S, self.keepalive_interval_s))
                if not self._keepalive_eligible():
                    continue
                silence = b"\x00" * int(self.sample_rate_hz * self.keepalive_silence_ms / 1000) * 2
                if not silence:
                    continue
                try:
                    async with self._send_lock:
                        if not self._keepalive_eligible():
                            continue
                        await self._send_audio_now(silence)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    await self._fail(
                        QwenAudioProtocolError("Qwen Audio keepalive send failed"),
                        cause=exc,
                        reason="qwen_keepalive_failed",
                    )
                    return
                logger.debug(
                    "Qwen Audio keepalive silence sent bytes=%s interval_s=%s",
                    len(silence),
                    self.keepalive_interval_s,
                )
        except asyncio.CancelledError:
            raise

    def _keepalive_eligible(self) -> bool:
        if self._closing_requested or self._ws is None:
            return False
        if not self._accept_terminals:
            return False
        if self._state is not QwenAudioSessionState.TASK_ACTIVE:
            return False
        if self._boundary_requested or self._flushing_audio:
            return False
        if self._audio_queue or self._post_boundary_audio_queue:
            return False
        last = self._last_audio_send_at
        if last is None:
            return False
        return time.monotonic() - last >= self.keepalive_interval_s

    async def begin_turn(self, request: STTProviderTurnRequest) -> None:
        if not self._accept_terminals:
            raise RuntimeError("Qwen Audio session is unavailable")
        if self._state is QwenAudioSessionState.STARTING_NEXT_TASK:
            future = self._start_future
            if future is not None:
                await asyncio.wait_for(
                    asyncio.shield(future),
                    timeout=self.task_start_timeout_s,
                )
        if self._state is not QwenAudioSessionState.TASK_ACTIVE or self._task_id is None:
            raise RuntimeError("Qwen Audio task is not active")
        self._event_projection.begin(request)
        self._scoped_task_id = self._task_id

    async def send_turn_audio(
        self,
        identity: STTProviderTurnIdentity,
        pcm16le: bytes,
        *,
        payload_sequence: int,
        source_ranges: tuple[AudioCaptureSpan, ...],
        context_only: bool,
    ) -> None:
        _ = source_ranges, context_only
        self._event_projection.validate_payload(identity, payload_sequence)
        if not pcm16le:
            self._event_projection.payload_written(identity, payload_sequence)
            return
        try:
            async with self._admission_lock:
                async with self._send_lock:
                    if (
                        self._state is not QwenAudioSessionState.TASK_ACTIVE
                        or self._task_id != self._scoped_task_id
                        or self._boundary_requested
                    ):
                        raise RuntimeError("Qwen Audio task cannot accept scoped audio")
                    await self._send_audio_now(pcm16le)
                    self._event_projection.payload_written(identity, payload_sequence)
        except Exception as exc:
            await self._fail(
                QwenAudioProtocolError("Qwen Audio audio send failed"),
                cause=exc,
                reason="qwen_write_failed",
            )

    async def seal_turn(
        self,
        identity: STTProviderTurnIdentity,
        *,
        sealed_content_ranges: tuple[AudioCaptureSpan, ...],
        seal_reason: str,
        observed_trailing_silence_ms: int | None,
    ) -> None:
        _ = sealed_content_ranges, seal_reason, observed_trailing_silence_ms
        self._event_projection.seal(identity)
        async with self._admission_lock:
            if (
                self._state is not QwenAudioSessionState.TASK_ACTIVE
                or self._task_id != self._scoped_task_id
                or self._active_boundary is not None
            ):
                raise RuntimeError("Qwen Audio task cannot be finished")
            self._active_boundary = _QwenAudioBoundary(self._next_boundary_sequence)
            self._next_boundary_sequence += 1
            self._boundary_requested = True
            async with self._send_lock:
                await self._finish_active_task_locked()

    async def abort_turn(self, identity: STTProviderTurnIdentity, *, reason: str) -> None:
        self._event_projection.require_open(identity)
        self._terminalize_scoped(
            identity,
            outcome="cancelled",
            failure_reason=reason,
            retire=True,
            provenance=STTNativeProvenance(
                native_task_id=self._scoped_task_id,
                barrier="abort",
            ),
        )
        self._event_projection.end_epoch(
            orderly=False,
            reason=reason,
            provider_turn_id=identity.provider_turn_id,
        )
        await self.abort_for_toggle_off()

    async def turn_events(self):
        async for event in self._event_projection.turn_events():
            yield event

    def _terminalize_scoped(
        self,
        identity: STTProviderTurnIdentity,
        *,
        outcome: SegmentTerminalOutcome,
        text: str = "",
        failure_reason: str | None = None,
        retire: bool = False,
        provenance: STTNativeProvenance,
        failure_diagnostics: ErrorDiagnostics | None = None,
    ) -> None:
        if not self._event_projection.is_current(identity):
            return
        self._event_projection.terminal(
            STTProviderTurnTerminal(
                identity=identity,
                outcome=outcome,
                text=text,
                text_authority="authoritative" if outcome in ("final", "empty") else "none",
                failure_reason=failure_reason,
                epoch_disposition="retire" if retire else "reuse",
                provenance=(provenance,),
                failure_diagnostics=failure_diagnostics,
            )
        )
        self._scoped_task_id = None

    async def send_audio(self, pcm16le: bytes) -> None:
        if not isinstance(pcm16le, bytes):
            raise TypeError("pcm16le must be bytes")
        if not pcm16le:
            return
        async with self._admission_lock:
            if not self._accept_terminals:
                return
            if (
                self._state is QwenAudioSessionState.TASK_ACTIVE
                and not self._flushing_audio
                and not self._boundary_requested
            ):
                try:
                    async with self._send_lock:
                        if (
                            self._state is QwenAudioSessionState.TASK_ACTIVE
                            and not self._flushing_audio
                            and not self._boundary_requested
                        ):
                            await self._send_audio_now(pcm16le)
                            return
                except Exception as exc:
                    await self._fail(
                        QwenAudioProtocolError("Qwen Audio audio send failed"),
                        cause=exc,
                        reason="qwen_write_failed",
                    )
                    return
            if self._state is QwenAudioSessionState.TASK_ACTIVE:
                if self._boundary_requested:
                    self._post_boundary_audio_queue.append(pcm16le)
                else:
                    self._audio_queue.append(pcm16le)
                return
            if self._state in (
                QwenAudioSessionState.CONNECTING,
                QwenAudioSessionState.FINISHING_TASK,
                QwenAudioSessionState.STARTING_NEXT_TASK,
            ):
                self._audio_queue.append(pcm16le)
                return
            return

    async def on_speech_end(
        self,
        *,
        trailing_silence_ms: int | None = None,
        reason: SpeechBoundaryReason | None = None,
    ) -> None:
        if not self._accept_terminals:
            return
        async with self._admission_lock:
            if not self._accept_terminals:
                return
            existing_ms = max(int(trailing_silence_ms or 0), 0)
            logger.info(
                "[STT][Tail] provider=qwen_audio boundary_reason=%s observed_tail_ms=%s "
                "injected_padding_ms=0 declared_trim_ms=0 boundary_wait_ms=%s",
                reason,
                existing_ms,
                boundary_wait_ms(reason, observed_tail_ms=existing_ms),
            )
            boundary = _QwenAudioBoundary(self._next_boundary_sequence)
            self._next_boundary_sequence += 1
            if self._state is QwenAudioSessionState.TASK_ACTIVE and self._active_boundary is None:
                self._active_boundary = boundary
                self._boundary_requested = True
                if not self._flushing_audio:
                    await self._finish_active_task_locked()
                return
            if self._state is QwenAudioSessionState.TASK_ACTIVE or self._state in (
                QwenAudioSessionState.CONNECTING,
                QwenAudioSessionState.FINISHING_TASK,
                QwenAudioSessionState.STARTING_NEXT_TASK,
            ):
                self._pending_boundaries.append(boundary)
                return
            if self._state is QwenAudioSessionState.FAILED:
                await self._resolve_pending_empty()

    async def _resolve_pending_empty(self) -> None:
        identity = self._event_projection.active_identity
        if identity is not None:
            self._terminalize_scoped(
                identity,
                outcome="failed",
                failure_reason="task_finished_without_terminal",
                retire=True,
                provenance=STTNativeProvenance(
                    native_task_id=self._scoped_task_id,
                    barrier="task_terminal_missing",
                ),
            )
        if not self._accept_terminals:
            return
        count = (1 if self._active_boundary is not None else 0) + len(self._pending_boundaries)
        self._active_boundary = None
        self._pending_boundaries.clear()
        self._audio_queue.clear()
        self._post_boundary_audio_queue.clear()
        for _ in range(count):
            self._put_event(STTBackendTranscriptEvent(text="", is_final=True))
        self._drain_complete.set()

    def _cancel_start_timeout(self) -> None:
        task = self._start_timeout_task
        self._start_timeout_task = None
        if task is not None and not task.done():
            task.cancel()

    def _cancel_finish_timeout(self) -> None:
        task = self._finish_timeout_task
        self._finish_timeout_task = None
        if task is not None and not task.done():
            task.cancel()

    async def _stop_keepalive(self) -> None:
        task = self._keepalive_task
        self._keepalive_task = None
        if task is None or task is asyncio.current_task():
            return
        if not task.done():
            task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    async def _wait_for_inflight_send(self) -> None:
        send_task = self._inflight_send_task
        if send_task is None or send_task.done() or send_task is asyncio.current_task():
            return
        try:
            await asyncio.wait_for(asyncio.shield(send_task), timeout=self.task_finish_timeout_s)
        except asyncio.TimeoutError:
            send_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await send_task
        except asyncio.CancelledError:
            raise
        except Exception:
            return

    async def _wait_for_task_finished(self, task_id: str) -> None:
        try:
            await asyncio.sleep(self.task_finish_timeout_s)
            if self._task_id == task_id and self._state is QwenAudioSessionState.FINISHING_TASK:
                await self._fail(
                    QwenAudioProtocolError("Qwen Audio task-finished timeout"),
                    reason="task_finish_timeout",
                    cause=TimeoutError(),
                )
        except asyncio.CancelledError:
            return

    async def _fail(
        self,
        exc: BaseException,
        *,
        reason: str = "session_failed",
        cause: BaseException | None = None,
        code: object = None,
    ) -> None:
        if self._state is QwenAudioSessionState.FAILED:
            return
        if self._state is QwenAudioSessionState.CLOSING and self._failure is None:
            return
        initial_connect = self._state is QwenAudioSessionState.CONNECTING
        self._log_failure(reason, cause=cause, code=code)
        self._failure = exc
        self._state = QwenAudioSessionState.FAILED
        self._accept_terminals = True
        self._cancel_start_timeout()
        future = self._start_future
        if initial_connect and future is not None and not future.done():
            future.set_exception(exc)
        self._cancel_finish_timeout()
        identity = self._event_projection.active_identity
        diagnostics = recognition_failure_diagnostics(
            cause or exc,
            provider="qwen_audio",
            channel=self._event_projection.channel or "unknown",
            operation="receive",
            reason=reason,
            identity=identity,
        )
        provider_code = code
        if isinstance(exc, QwenAudioTaskFailedError):
            provider_code = exc.error_code
        if (
            isinstance(provider_code, str)
            and len(provider_code) <= 6
            and provider_code.isascii()
            and provider_code.isdecimal()
        ):
            provider_code = int(provider_code)
        if type(provider_code) is int and 0 <= provider_code <= 999999:
            diagnostics = replace(
                diagnostics, fields={**diagnostics.fields, "provider_code": provider_code}
            )
        exc.failure_diagnostics = diagnostics
        if cause is not None:
            exc.__cause__ = cause
        reason = (
            reason
            if reason
            in {
                "provider_task_failed",
                "task_start_timeout",
                "task_finish_timeout",
                "qwen_write_failed",
                "qwen_keepalive_failed",
                "run_task_send_failed",
                "receive_failed",
                "receive_eof",
                "session_failed",
            }
            else "session_failed"
        )
        if identity is not None:
            self._terminalize_scoped(
                identity,
                outcome="failed",
                failure_reason=reason,
                retire=True,
                failure_diagnostics=diagnostics,
                provenance=STTNativeProvenance(
                    native_task_id=self._scoped_task_id,
                    barrier="task-failed",
                ),
            )
        self._event_projection.end_epoch(
            orderly=False,
            reason=reason,
            provider_turn_id=identity.provider_turn_id if identity is not None else None,
            failure_diagnostics=diagnostics,
        )
        await self._resolve_pending_empty()
        self._accept_terminals = False
        self._closing_requested = True
        await self._close_socket()
        self._put_event(exc)
        self._put_event(None)

    async def stop(self) -> None:
        if self._state is QwenAudioSessionState.CLOSING:
            return
        if self._state is QwenAudioSessionState.FAILED:
            await self._close_socket()
            return
        self._closing_requested = True
        await self._wait_for_inflight_send()
        if self._state is QwenAudioSessionState.TASK_ACTIVE and self._active_boundary is None:
            await self.on_speech_end()
        if (
            self._active_boundary is None
            and not (
                self._state is QwenAudioSessionState.STARTING_NEXT_TASK and self._pending_boundaries
            )
            and self._state is not QwenAudioSessionState.FINISHING_TASK
        ):
            self._state = QwenAudioSessionState.CLOSING
            self._drain_complete.set()
            await self._close_socket()
            return
        if self._state is QwenAudioSessionState.STARTING_NEXT_TASK and self._pending_boundaries:
            try:
                await asyncio.wait_for(
                    self._drain_complete.wait(), timeout=self.task_finish_timeout_s
                )
            except asyncio.TimeoutError:
                self._accept_terminals = True
                await self._resolve_pending_empty()
                self._accept_terminals = False
                self._state = QwenAudioSessionState.CLOSING
                await self._close_socket()
            return
        try:
            await asyncio.wait_for(self._drain_complete.wait(), timeout=self.task_finish_timeout_s)
        except asyncio.TimeoutError:
            self._accept_terminals = True
            await self._resolve_pending_empty()
            self._accept_terminals = False
            self._state = QwenAudioSessionState.CLOSING
            await self._close_socket()

    async def abort_for_toggle_off(self) -> None:
        self._accept_terminals = False
        identity = self._event_projection.active_identity
        if identity is not None:
            self._terminalize_scoped(
                identity,
                outcome="cancelled",
                failure_reason="toggle_off",
                retire=True,
                provenance=STTNativeProvenance(
                    native_task_id=self._scoped_task_id,
                    barrier="abort",
                ),
            )
        self._closing_requested = True
        self._state = QwenAudioSessionState.CLOSING
        self._audio_queue.clear()
        self._pending_boundaries.clear()
        self._post_boundary_audio_queue.clear()
        self._active_boundary = None
        self._cancel_start_timeout()
        self._drain_complete.set()
        await self._close_socket(cancel_receiver=True)
        self._clear_events()

    def _clear_events(self) -> None:
        self._event_projection.discard_legacy()

    async def _close_socket(self, *, cancel_receiver: bool = False) -> None:
        current_task = asyncio.current_task()
        await self._stop_keepalive()
        deferred_task = self._deferred_finish_task
        if (
            deferred_task is not None
            and not deferred_task.done()
            and deferred_task is not current_task
        ):
            deferred_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await deferred_task
        self._deferred_finish_task = None
        flush_task = self._flush_task
        if flush_task is not None and not flush_task.done() and flush_task is not current_task:
            flush_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await flush_task
        self._flush_task = None
        send_task = self._inflight_send_task
        if send_task is not None and not send_task.done() and send_task is not current_task:
            send_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await send_task
        self._inflight_send_task = None
        ws = self._ws
        self._ws = None
        close_at = time.monotonic()
        close_error: BaseException | None = None
        if ws is not None:
            try:
                await ws.close()
            except Exception as exc:
                close_error = exc
                diagnostics = recognition_failure_diagnostics(
                    exc,
                    provider="qwen_audio",
                    channel=self._event_projection.channel,
                    operation="cleanup",
                    reason="provider_cleanup_failed",
                )
                primary = getattr(self._failure, "failure_diagnostics", None)
                emit_failure(
                    logger,
                    replace(
                        diagnostics,
                        cause_id=primary.diagnostic_id if primary else None,
                        fields={**diagnostics.fields, "cleanup_confirmed": False},
                    ),
                    severity="warning",
                    disposition="degraded",
                    stage="cleanup",
                )
            logger.info(
                "[QwenAudio] close elapsed_ms=%d startup_ms=%d ws_close_code=%s "
                "outcome=%s cause_class=%s",
                int((time.monotonic() - close_at) * 1000),
                int((time.monotonic() - self._startup_at) * 1000) if self._startup_at else 0,
                _close_code(close_error, ws),
                "failed" if close_error is not None else "completed",
                _safe_exception_class(close_error),
            )
        recv_task = self._recv_task
        if recv_task is not None and not recv_task.done() and recv_task is not current_task:
            recv_task.cancel()
        if recv_task is not None and recv_task is not current_task:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await recv_task
        self._recv_task = None

    def _put_event(self, event: STTBackendTranscriptEvent | BaseException | None) -> None:
        if self._events_closed or not self._event_projection.is_legacy:
            return
        if event is None:
            self._events_closed = True
        self._event_projection.put_legacy(event)

    async def close(self) -> None:
        if self._state not in (QwenAudioSessionState.CLOSING, QwenAudioSessionState.FAILED):
            with contextlib.suppress(Exception):
                await self.stop()
        await self._close_socket(cancel_receiver=True)
        self._event_projection.close()

    async def events(self):
        async for event in self._event_projection.events():
            yield event


@dataclass(frozen=True, slots=True)
class _QwenTaskCapture:
    input_start: int
    input_end: int
    capture: AudioCaptureSpan
    context_only: bool


@dataclass(slots=True)
class _QwenPeerAudioSession(_QwenAudioSession):
    sentence_silence_ms: ClassVar[int] = QWEN_PEER_SENTENCE_SILENCE_MS
    synthetic_keepalive: ClassVar[bool] = False
    accepts_stream_input: ClassVar[bool] = True
    independent_recognition_units: ClassVar[bool] = True
    _stream: RecognitionStreamIdentity | None = field(init=False, default=None)
    _input_map: STTStreamInputMap = field(init=False, default_factory=STTStreamInputMap)
    _task_capture: list[_QwenTaskCapture] = field(init=False, default_factory=list)
    _task_samples: int = field(init=False, default=0)
    _last_final_samples: int = field(init=False, default=0)
    _last_sentence_id: int = field(init=False, default=0)
    _metadata_bytes: int = field(init=False, default=0)
    _receipt_sequence: int = field(init=False, default=0)
    _task_terminal: asyncio.Future[None] | None = field(init=False, default=None)
    _rotation_task: asyncio.Task[None] | None = field(init=False, default=None)
    _age_task: asyncio.Task[None] | None = field(init=False, default=None)
    _stream_write_lock: asyncio.Lock = field(init=False, default_factory=asyncio.Lock)
    _ending: bool = field(init=False, default=False)

    async def _begin_task(self, *, initial: bool = False) -> None:
        self._task_capture.clear()
        self._task_samples = 0
        self._last_final_samples = 0
        self._last_sentence_id = 0
        self._metadata_bytes = 0
        self._task_terminal = asyncio.get_running_loop().create_future()
        await _QwenAudioSession._begin_task(self, initial=initial)

    async def _handle_task_started(self, event_task_id: str) -> None:
        starting = self._state in (
            QwenAudioSessionState.CONNECTING,
            QwenAudioSessionState.STARTING_NEXT_TASK,
        )
        await _QwenAudioSession._handle_task_started(self, event_task_id)
        if (
            starting
            and event_task_id == self._task_id
            and self._state is QwenAudioSessionState.TASK_ACTIVE
        ):
            await self._cancel_peer_task(self._age_task)
            self._age_task = asyncio.create_task(self._task_age_limit(), name="qwen-peer-task-age")

    async def _task_age_limit(self) -> None:
        try:
            await asyncio.sleep(QWEN_PEER_TASK_MAX_S)
            self._request_rotation()
        except asyncio.CancelledError:
            return
        finally:
            if self._age_task is asyncio.current_task():
                self._age_task = None

    def _request_rotation(self) -> None:
        if (
            self._ending
            or not self._accept_terminals
            or self._state is QwenAudioSessionState.FAILED
        ):
            return
        if self._rotation_task is None:
            self._rotation_task = asyncio.create_task(
                self._rotate_task(), name="qwen-peer-task-rotation"
            )

    async def _rotate_task(self) -> None:
        try:
            async with self._stream_write_lock:
                await self._finish_peer_task()
                if not self._ending and self._accept_terminals:
                    await self._begin_task()
                    future = self._start_future
                    if future is None:
                        raise QwenAudioProtocolError("Qwen Audio task readiness is missing")
                    await asyncio.wait_for(asyncio.shield(future), self.task_start_timeout_s)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._fail(
                QwenAudioProtocolError("Qwen Audio task rotation failed"),
                reason="session_failed",
                cause=exc,
            )
        finally:
            if self._rotation_task is asyncio.current_task():
                self._rotation_task = None

    async def _finish_peer_task(self) -> None:
        terminal = self._task_terminal
        if self._state is QwenAudioSessionState.TASK_ACTIVE:
            async with self._admission_lock:
                async with self._send_lock:
                    self._active_boundary = _QwenAudioBoundary(self._next_boundary_sequence)
                    self._next_boundary_sequence += 1
                    self._boundary_requested = True
                    await self._finish_active_task_locked()
        if self._failure is not None:
            raise self._failure
        if terminal is not None and not terminal.done():
            try:
                await asyncio.wait_for(asyncio.shield(terminal), self.task_finish_timeout_s)
            except asyncio.TimeoutError as exc:
                await self._fail(
                    QwenAudioProtocolError("Qwen Audio task-finished timeout"),
                    reason="task_finish_timeout",
                    cause=exc,
                )
                raise QwenAudioProtocolError("Qwen Audio task-finished timeout") from exc
        if self._failure is not None:
            raise self._failure

    async def _handle_task_finished(self, event_task_id: str) -> None:
        if event_task_id != self._task_id:
            return
        if self._state is not QwenAudioSessionState.FINISHING_TASK:
            await self._fail(
                QwenAudioProtocolError("Qwen Audio task ended without input closure"),
                reason="session_failed",
            )
            return
        self._task_finished_count += 1
        self._cancel_finish_timeout()
        self._active_boundary = None
        self._boundary_requested = False
        self._task_id = None
        self._state = QwenAudioSessionState.STARTING_NEXT_TASK
        terminal = self._task_terminal
        if terminal is not None and not terminal.done():
            terminal.set_result(None)

    async def begin_stream(self, stream: RecognitionStreamIdentity) -> None:
        if (
            self._ending
            or not self._accept_terminals
            or stream.provider_epoch_id != self._event_projection.provider_epoch_id
            or stream.channel != "peer"
            or (self._stream is not None and self._stream != stream)
        ):
            raise QwenAudioProtocolError("Qwen Audio stream ownership is unavailable")
        self._stream = stream

    async def begin_turn(self, request: STTProviderTurnRequest) -> None:
        identity = request.identity
        await self.begin_stream(
            RecognitionStreamIdentity(
                request.channel,
                identity.segment.activation_generation,
                identity.segment.capture_epoch,
                identity.provider_epoch_id,
                identity.settings_scope,
            )
        )
        self._event_projection.begin(request)

    async def send_turn_audio(
        self,
        identity: STTProviderTurnIdentity,
        pcm16le: bytes,
        *,
        payload_sequence: int,
        source_ranges: tuple[AudioCaptureSpan, ...],
        context_only: bool,
    ) -> None:
        self._event_projection.validate_payload(identity, payload_sequence)
        await self._submit_peer_audio(pcm16le, source_ranges, context_only)
        self._event_projection.payload_written(identity, payload_sequence)

    async def send_stream_audio(
        self, pcm16le: bytes, *, source_ranges: tuple[AudioCaptureSpan, ...]
    ) -> None:
        await self._submit_peer_audio(pcm16le, source_ranges, False)

    async def send_audio(self, pcm16le: bytes) -> None:
        raise RuntimeError("Peer Qwen Audio requires owned source coverage")

    async def on_speech_end(
        self,
        *,
        trailing_silence_ms: int | None = None,
        reason: SpeechBoundaryReason | None = None,
    ) -> None:
        raise RuntimeError("Peer Qwen Audio local input closes through seal_turn")

    async def _submit_peer_audio(
        self, pcm16le: bytes, ranges: tuple[AudioCaptureSpan, ...], context_only: bool
    ) -> None:
        if self._stream is None:
            raise QwenAudioProtocolError("Qwen Audio stream has no source owner")
        _, write = self._input_map.prepare(pcm16le, ranges)
        if write is None:
            return
        cursor = 0
        for span in ranges:
            start, end = span.normalized_start_sample, span.normalized_end_sample
            assert start is not None and end is not None
            left = max(start, write.source_start)
            while left < end:
                rotation = self._rotation_task
                if rotation is not None:
                    await rotation
                async with self._stream_write_lock:
                    if self._rotation_task is not None:
                        continue
                    if self._ending or self._state is not QwenAudioSessionState.TASK_ACTIVE:
                        raise QwenAudioProtocolError("Qwen Audio task cannot accept input")
                    room = min(
                        QWEN_PEER_NO_FINAL_S * self.sample_rate_hz
                        - (self._task_samples - self._last_final_samples),
                        QWEN_PEER_TASK_MAX_S * self.sample_rate_hz - self._task_samples,
                    )
                    right = min(end, left + room)
                    if room <= 0:
                        self._request_rotation()
                        continue
                    capture = span.slice_normalized(left, right)
                    merged = self._can_merge_capture(capture, context_only)
                    if not merged and (
                        len(self._task_capture) >= QWEN_PEER_MAX_MAP_RECORDS
                        or self._metadata_bytes + 192 > QWEN_PEER_MAX_METADATA_BYTES
                    ):
                        self._request_rotation()
                        continue
                    offset = (cursor + left - start) * 2
                    selected = pcm16le[offset : offset + (right - left) * 2]
                    async with self._admission_lock:
                        if self._ending or self._state is not QwenAudioSessionState.TASK_ACTIVE:
                            raise QwenAudioProtocolError("Qwen Audio task cannot accept input")
                        if not merged and self._metadata_bytes + 192 > QWEN_PEER_MAX_METADATA_BYTES:
                            self._request_rotation()
                            continue
                        async with self._send_lock:
                            try:
                                await self._send_audio_now(selected)
                            except Exception as exc:
                                await self._fail(
                                    QwenAudioProtocolError("Qwen Audio audio send failed"),
                                    reason="qwen_write_failed",
                                    cause=exc,
                                )
                                raise QwenAudioProtocolError(
                                    "Qwen Audio audio send failed"
                                ) from exc
                        self._append_capture(capture, context_only, merged)
                        self._input_map.commit(
                            STTStreamWrite(capture.capture_epoch, left, right, right - left)
                        )
                    left = right
                    if (
                        self._task_samples - self._last_final_samples
                        >= QWEN_PEER_NO_FINAL_S * self.sample_rate_hz
                        or self._task_samples >= QWEN_PEER_TASK_MAX_S * self.sample_rate_hz
                    ):
                        self._request_rotation()
            cursor += end - start

    def _can_merge_capture(self, capture: AudioCaptureSpan, context_only: bool) -> bool:
        if not self._task_capture:
            return False
        previous = self._task_capture[-1].capture
        return (
            self._task_capture[-1].context_only == context_only
            and previous.capture_epoch == capture.capture_epoch
            and previous.source_sample_rate_hz == capture.source_sample_rate_hz
            and previous.normalized_sample_rate_hz == capture.normalized_sample_rate_hz
            and previous.source_end_sample == capture.source_start_sample
            and previous.normalized_end_sample == capture.normalized_start_sample
            and previous.source_end_monotonic_s == capture.source_start_monotonic_s
            and capture.discontinuity_before is None
        )

    def _append_capture(self, capture: AudioCaptureSpan, context_only: bool, merged: bool) -> None:
        count = capture.normalized_sample_count
        if merged:
            previous = self._task_capture[-1]
            self._task_capture[-1] = replace(
                previous,
                input_end=self._task_samples + count,
                capture=replace(
                    previous.capture,
                    source_end_sample=capture.source_end_sample,
                    normalized_end_sample=capture.normalized_end_sample,
                    source_end_monotonic_s=capture.source_end_monotonic_s,
                ),
            )
        else:
            self._task_capture.append(
                _QwenTaskCapture(
                    self._task_samples, self._task_samples + count, capture, context_only
                )
            )
            self._metadata_bytes += 192
        self._task_samples += count

    def recognition_source_covers(self, ranges: tuple[AudioCaptureSpan, ...]) -> bool:
        return self._input_map.covers(ranges)

    async def seal_turn(
        self,
        identity: STTProviderTurnIdentity,
        *,
        sealed_content_ranges: tuple[AudioCaptureSpan, ...],
        seal_reason: str,
        observed_trailing_silence_ms: int | None,
    ) -> None:
        if sealed_content_ranges and not self._input_map.covers(sealed_content_ranges):
            raise QwenAudioProtocolError("Qwen Audio sealed input is not fully submitted")
        self._event_projection.seal(identity)
        self._event_projection.terminal(STTProviderInputTerminal(identity, "submitted", "peer"))

    async def _handle_server_message(self, message: object) -> None:
        response = self._message_dict(message)
        if response is None:
            return
        header = response.get("header") or {}
        if header.get("event") != "result-generated":
            await _QwenAudioSession._handle_server_message(self, response)
            return
        async with self._admission_lock:
            await self._admit_peer_sentence(response, header)

    async def _admit_peer_sentence(
        self, response: Mapping[str, object], header: Mapping[str, object]
    ) -> None:
        task_id = str(header.get("task_id") or "")
        if task_id != self._task_id or self._state not in (
            QwenAudioSessionState.TASK_ACTIVE,
            QwenAudioSessionState.FINISHING_TASK,
        ):
            return
        sentence = response.get("payload", {}).get("output", {}).get("sentence", {})
        if sentence.get("heartbeat") or not sentence.get("sentence_end"):
            return
        raw_id = sentence.get("sentence_id")
        if raw_id == 0 or raw_id == "0":
            return
        try:
            sentence_number = int(str(raw_id))
        except ValueError:
            sentence_number = -1
        sentence_id = str(sentence_number)
        if sentence_id in self._sentence_ids:
            return
        if sentence_number <= self._last_sentence_id:
            await self._fail(
                QwenAudioProtocolError("Qwen Audio final sentence order is inconsistent"),
                reason="session_failed",
            )
            return
        text = sentence.get("text")
        if not isinstance(text, str) or self._stream is None:
            await self._fail(
                QwenAudioProtocolError("Qwen Audio final sentence has no admitted identity"),
                reason="session_failed",
            )
            return
        identity_bytes = len(sentence_id.encode("utf-8")) + 64
        if (
            len(self._sentence_ids) >= QWEN_PEER_MAX_SENTENCES
            or self._metadata_bytes + identity_bytes > QWEN_PEER_MAX_METADATA_BYTES
        ):
            await self._fail(
                QwenAudioProtocolError("Qwen Audio task evidence capacity exceeded"),
                reason="session_failed",
            )
            return
        self._sentence_ids.add(sentence_id)
        self._last_sentence_id = sentence_number
        self._metadata_bytes += identity_bytes
        self._last_final_samples = self._task_samples
        begin = self._native_time(sentence.get("begin_time"))
        end = self._native_time(sentence.get("end_time"))
        words = tuple(
            NativeWordEvidence(
                word.get("text") if isinstance(word.get("text"), str) else None,
                word.get("punctuation") if isinstance(word.get("punctuation"), str) else None,
                self._native_time(word.get("begin_time")),
                self._native_time(word.get("end_time")),
            )
            for word in (sentence.get("words") or ())
            if isinstance(word, Mapping)
        )
        native = NativeTranscriptionEvidence(
            native_task_id=task_id,
            native_sentence_id=sentence_id,
            timestamp_basis="task_relative_estimated",
            begin_time_ms=begin,
            end_time_ms=end,
            input_sample_rate_hz=self.sample_rate_hz,
            task_input_samples=self._task_samples,
            timed_words=words,
            audio_coverage=self._sentence_coverage(begin, end),
        )
        self._receipt_sequence += 1
        unit = STTRecognitionUnit(
            RecognitionUnitIdentity(
                self._stream,
                uuid.uuid5(
                    uuid.NAMESPACE_URL, f"{self._stream.provider_epoch_id}/{task_id}/{sentence_id}"
                ),
                self._receipt_sequence,
            ),
            text,
            STTNativeProvenance(
                native_event_id=sentence_id,
                native_task_id=task_id,
                barrier="sentence_end",
                transcription=native,
            ),
        )
        if not self._event_projection.put_recognition(unit):
            await self._fail(
                QwenAudioProtocolError("Qwen Audio recognition capacity exceeded"),
                reason="session_failed",
            )
            return
        if (
            len(self._sentence_ids) >= QWEN_PEER_MAX_SENTENCES
            or self._metadata_bytes >= QWEN_PEER_MAX_METADATA_BYTES
        ):
            self._request_rotation()

    @staticmethod
    def _native_time(value: object) -> float | None:
        return float(value) if type(value) in (int, float) and math.isfinite(value) else None

    def _sentence_coverage(
        self, begin_ms: float | None, end_ms: float | None
    ) -> tuple[NativeAudioCoverage, ...]:
        left = math.floor(begin_ms * self.sample_rate_hz / 1000) if begin_ms is not None else 0
        right = (
            math.ceil(end_ms * self.sample_rate_hz / 1000)
            if end_ms is not None
            else self._task_samples
        )
        coverage = []
        for entry in self._task_capture:
            start, end = max(left, entry.input_start), min(right, entry.input_end)
            if start >= end:
                continue
            source_start = entry.capture.normalized_start_sample
            assert source_start is not None
            capture = entry.capture.slice_normalized(
                source_start + start - entry.input_start,
                source_start + end - entry.input_start,
            )
            coverage.append(
                NativeAudioCoverage(
                    start,
                    end,
                    capture.capture_epoch,
                    capture.source_sample_rate_hz,
                    capture.source_start_sample,
                    capture.source_end_sample,
                    capture.normalized_sample_rate_hz,
                    capture.normalized_start_sample,
                    capture.normalized_end_sample,
                    capture.source_start_monotonic_s,
                    capture.source_end_monotonic_s,
                    entry.context_only,
                    discontinuity_kind=(
                        capture.discontinuity_before.kind if capture.discontinuity_before else None
                    ),
                    discontinuity_observed_at_monotonic_s=(
                        capture.discontinuity_before.observed_at_monotonic_s
                        if capture.discontinuity_before
                        else None
                    ),
                    lost_source_samples=(
                        capture.discontinuity_before.lost_source_samples
                        if capture.discontinuity_before
                        else None
                    ),
                )
            )
        return tuple(coverage)

    def _terminalize_scoped(
        self,
        identity: STTProviderTurnIdentity,
        *,
        outcome: SegmentTerminalOutcome,
        text: str = "",
        failure_reason: str | None = None,
        retire: bool = False,
        provenance: STTNativeProvenance,
        failure_diagnostics: ErrorDiagnostics | None = None,
    ) -> None:
        self._event_projection.terminal(
            STTProviderInputTerminal(
                identity,
                outcome if outcome in ("failed", "expired", "cancelled") else "failed",
                "peer",
                failure_reason,
                "retire" if retire else "reuse",
                failure_diagnostics=failure_diagnostics,
            )
        )

    async def end_stream(self, *, reason: str) -> None:
        self._ending = True
        await self._cancel_peer_task(self._age_task)
        rotation = self._rotation_task
        if rotation is not None:
            await rotation
        try:
            async with self._stream_write_lock:
                await self._finish_peer_task()
                await asyncio.sleep(0)
                self._closing_requested = True
                self._state = QwenAudioSessionState.CLOSING
                await self._close_socket()
        except asyncio.CancelledError:
            await self.abort_for_toggle_off()
            raise

    async def abort_turn(self, identity: STTProviderTurnIdentity, *, reason: str) -> None:
        self._terminalize_scoped(
            identity,
            outcome="cancelled",
            failure_reason=reason,
            retire=True,
            provenance=STTNativeProvenance(barrier="abort"),
        )
        await self.abort_for_toggle_off()

    async def stop(self) -> None:
        await self.abort_for_toggle_off()

    async def abort_for_toggle_off(self) -> None:
        self._ending = True
        await _QwenAudioSession.abort_for_toggle_off(self)

    async def _close_socket(self, *, cancel_receiver: bool = False) -> None:
        await self._cancel_peer_task(self._age_task)
        await self._cancel_peer_task(self._rotation_task)
        self._age_task = None
        self._rotation_task = None
        self._cancel_start_timeout()
        self._cancel_finish_timeout()
        await _QwenAudioSession._close_socket(self, cancel_receiver=cancel_receiver)
        self._task_capture.clear()
        self._sentence_ids.clear()
        self._metadata_bytes = 0
        terminal = self._task_terminal
        if terminal is not None and not terminal.done():
            terminal.set_result(None)

    @staticmethod
    async def _cancel_peer_task(task: asyncio.Task[None] | None) -> None:
        if task is not None and task is not asyncio.current_task():
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)


__all__ = [
    "QWEN_AUDIO_DEFAULT_ENDPOINT",
    "QWEN_AUDIO_DEFAULT_HOTWORD_WEIGHT",
    "QWEN_AUDIO_KEEPALIVE_INTERVAL_S",
    "QWEN_AUDIO_KEEPALIVE_SILENCE_MS",
    "QWEN_AUDIO_KEEPALIVE_TICK_S",
    "QWEN_AUDIO_MODEL",
    "QwenAudioProtocolError",
    "QwenAudioSessionState",
    "QwenAudioStreamingSTTBackend",
    "QwenAudioTaskFailedError",
]
