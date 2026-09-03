from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import uuid
from collections import deque
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from puripuly_heart.core.speech_boundary import SpeechBoundaryReason, boundary_wait_ms
from puripuly_heart.core.stt.backend import (
    RecoverableSTTSessionError,
    STTBackend,
    STTBackendSession,
    STTBackendTranscriptEvent,
)

logger = logging.getLogger(__name__)

QWEN_AUDIO_MODEL = "qwen-audio-3.0-asr-flash-streaming"
QWEN_AUDIO_DEFAULT_ENDPOINT = "wss://dashscope.aliyuncs.com/api-ws/v1/inference"
QWEN_AUDIO_DEFAULT_HOTWORD_WEIGHT = 4
QWEN_AUDIO_LANGUAGE_HINTS_LIMIT = 4


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
    hotwords: HotwordInput = ()
    hotword_weight: int = QWEN_AUDIO_DEFAULT_HOTWORD_WEIGHT
    websocket_factory: WebSocketFactory | None = None

    async def open_session(self) -> STTBackendSession:
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
        ):
            if value <= 0:
                raise ValueError(f"{name} must be > 0")
        session = _QwenAudioSession(
            api_key=self.api_key,
            language_hints=self.language_hints,
            model=self.model,
            endpoint=self.endpoint,
            sample_rate_hz=self.sample_rate_hz,
            connect_timeout_s=self.connect_timeout_s,
            task_start_timeout_s=self.task_start_timeout_s,
            task_finish_timeout_s=self.task_finish_timeout_s,
            send_timeout_s=self.send_timeout_s,
            hotwords=self.hotwords,
            hotword_weight=self.hotword_weight,
            websocket_factory=self.websocket_factory,
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
    async def verify_api_key(api_key: str, *, endpoint: str = QWEN_AUDIO_DEFAULT_ENDPOINT) -> bool:
        if not api_key:
            return False
        import websockets

        try:
            async with websockets.connect(
                endpoint,
                additional_headers={"Authorization": f"Bearer {api_key}"},
                ping_interval=None,
                open_timeout=5,
            ):
                return True
        except Exception:
            return False


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
    hotwords: HotwordInput = ()
    hotword_weight: int = QWEN_AUDIO_DEFAULT_HOTWORD_WEIGHT
    websocket_factory: WebSocketFactory | None = None

    _events: asyncio.Queue[STTBackendTranscriptEvent | BaseException | None] = field(
        init=False, repr=False
    )
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
    _inflight_send_task: asyncio.Task[Any] | None = field(init=False, default=None, repr=False)
    _admission_lock: asyncio.Lock = field(init=False, repr=False)
    _send_lock: asyncio.Lock = field(init=False, repr=False)
    _loop: asyncio.AbstractEventLoop | None = field(init=False, default=None, repr=False)
    _failure: BaseException | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        self._events = asyncio.Queue()
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

    def update_hotwords(self, hotwords: HotwordInput) -> None:
        self.hotwords = hotwords

    def _vocabulary(self) -> dict[str, int]:
        return _normalized_vocabulary(self.hotwords, default_weight=self.hotword_weight)

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        factory = self.websocket_factory
        if factory is None:
            import websockets

            factory = websockets.connect
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
        except Exception as exc:
            self._state = QwenAudioSessionState.FAILED
            raise QwenAudioProtocolError(f"Qwen Audio connection failed: {exc}") from exc
        self._recv_task = asyncio.create_task(self._recv_loop(), name="qwen-audio-recv")
        await self._begin_task(initial=True)
        future = self._start_future
        if future is None:
            raise QwenAudioProtocolError("Qwen Audio task start was not requested")
        try:
            await asyncio.wait_for(asyncio.shield(future), timeout=self.task_start_timeout_s)
        except asyncio.TimeoutError as exc:
            await self._fail(QwenAudioProtocolError("Qwen Audio task-started timeout"))
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
            "max_sentence_silence": 6000,
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
        try:
            await self._send_json(payload)
        except Exception as exc:
            await self._fail(QwenAudioProtocolError(f"Qwen Audio run-task send failed: {exc}"))
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
                await self._fail(QwenAudioProtocolError("Qwen Audio task-started timeout"))
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
                await self._fail(QwenAudioProtocolError(f"Qwen Audio socket failure: {exc}"))
        finally:
            if (
                self._state not in (QwenAudioSessionState.CLOSING, QwenAudioSessionState.FAILED)
                and self._accept_terminals
            ):
                await self._fail(
                    QwenAudioProtocolError("Qwen Audio socket closed before task completion")
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
        except (TypeError, ValueError):
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
            logger.debug(
                "Qwen Audio ignored provider event=%s task_id=%s", event_type, event_task_id
            )

    async def inject_event(self, event: Mapping[str, object]) -> None:
        await self._handle_server_message(event)

    async def _handle_task_started(self, event_task_id: str) -> None:
        if not event_task_id or event_task_id != self._task_id:
            logger.debug("Qwen Audio stale task-started ignored task_id=%s", event_task_id)
            return
        if self._state not in (
            QwenAudioSessionState.CONNECTING,
            QwenAudioSessionState.STARTING_NEXT_TASK,
        ):
            logger.debug("Qwen Audio duplicate task-started ignored task_id=%s", event_task_id)
            return
        self._cancel_start_timeout()
        self._state = QwenAudioSessionState.TASK_ACTIVE
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
            await self._fail(QwenAudioProtocolError(f"Qwen Audio queued audio send failed: {exc}"))
        finally:
            self._flushing_audio = False
            if self._flush_task is current_task:
                self._flush_task = None

    def _handle_result_generated(self, event_task_id: str, response: Mapping[str, object]) -> None:
        if event_task_id != self._task_id or self._state not in (
            QwenAudioSessionState.TASK_ACTIVE,
            QwenAudioSessionState.FINISHING_TASK,
        ):
            logger.debug("Qwen Audio stale result ignored task_id=%s", event_task_id)
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
            logger.debug(
                "Qwen Audio duplicate sentence ignored task_id=%s sentence_id=%s",
                event_task_id,
                sentence_id,
            )
            return
        self._sentence_ids.add(sentence_id)
        text = str(sentence.get("text") or "").strip()
        if text:
            self._sentences.append(text)

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
            logger.debug("Qwen Audio stale task-finished ignored task_id=%s", event_task_id)
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
            logger.debug("Qwen Audio duplicate task-finished ignored task_id=%s", event_task_id)
            return
        self._cancel_finish_timeout()
        self._cancel_start_timeout()
        if self._accept_terminals:
            self._put_event(
                STTBackendTranscriptEvent(text=_join_sentences(self._sentences), is_final=True)
            )
        self._active_boundary = None
        self._boundary_requested = False
        self._task_id = None
        self._sentences.clear()
        self._sentence_ids.clear()
        if self._closing_requested and not self._pending_boundaries:
            self._state = QwenAudioSessionState.CLOSING
            self._drain_complete.set()
            await self._close_socket()
            return
        try:
            await self._begin_task()
        except Exception:
            return

    async def _handle_task_failed(self, event_task_id: str, header: Mapping[str, object]) -> None:
        if event_task_id != self._task_id:
            logger.debug("Qwen Audio stale task-failed ignored task_id=%s", event_task_id)
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
            )
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
            await self._fail(QwenAudioProtocolError(f"Qwen Audio finish-task send failed: {exc}"))

    async def _send_audio_now(self, pcm16le: bytes) -> None:
        if not pcm16le:
            return
        await self._send_value(pcm16le)

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
                    await self._fail(QwenAudioProtocolError(f"Qwen Audio audio send failed: {exc}"))
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
                await self._fail(QwenAudioProtocolError("Qwen Audio task-finished timeout"))
        except asyncio.CancelledError:
            return

    async def _fail(self, exc: BaseException) -> None:
        if self._state is QwenAudioSessionState.FAILED:
            return
        if self._state is QwenAudioSessionState.CLOSING and self._failure is None:
            return
        initial_connect = self._state is QwenAudioSessionState.CONNECTING
        self._failure = exc
        self._state = QwenAudioSessionState.FAILED
        self._accept_terminals = True
        self._cancel_start_timeout()
        future = self._start_future
        if initial_connect and future is not None and not future.done():
            future.set_exception(exc)
        self._cancel_finish_timeout()
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
        while True:
            try:
                self._events.get_nowait()
            except asyncio.QueueEmpty:
                return

    async def _close_socket(self, *, cancel_receiver: bool = False) -> None:
        current_task = asyncio.current_task()
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
        if ws is not None:
            with contextlib.suppress(Exception):
                await ws.close()
        recv_task = self._recv_task
        if recv_task is not None and not recv_task.done() and recv_task is not current_task:
            recv_task.cancel()
        if recv_task is not None and recv_task is not current_task:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await recv_task
        self._recv_task = None

    def _put_event(self, event: STTBackendTranscriptEvent | BaseException | None) -> None:
        if self._events_closed:
            return
        if event is None:
            self._events_closed = True
        self._events.put_nowait(event)

    async def close(self) -> None:
        if self._state not in (QwenAudioSessionState.CLOSING, QwenAudioSessionState.FAILED):
            with contextlib.suppress(Exception):
                await self.stop()
        await self._close_socket(cancel_receiver=True)

    async def events(self):
        while True:
            event = await self._events.get()
            if event is None:
                return
            if isinstance(event, BaseException):
                raise event
            yield event


__all__ = [
    "QWEN_AUDIO_DEFAULT_ENDPOINT",
    "QWEN_AUDIO_DEFAULT_HOTWORD_WEIGHT",
    "QWEN_AUDIO_MODEL",
    "QwenAudioProtocolError",
    "QwenAudioSessionState",
    "QwenAudioStreamingSTTBackend",
    "QwenAudioTaskFailedError",
]
