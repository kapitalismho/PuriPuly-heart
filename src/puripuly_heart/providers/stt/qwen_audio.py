from __future__ import annotations

import asyncio
import json
import logging
import math
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Literal

from puripuly_heart.core import network_clients
from puripuly_heart.core.audio.format import AudioCaptureSpan
from puripuly_heart.core.speech_boundary import SpeechBoundaryReason
from puripuly_heart.core.stt.backend import (
    LEGACY_STT_SESSION_PROJECTION,
    RecoverableSTTSessionError,
    STTBackend,
    STTBackendSession,
    STTBackendTranscriptEvent,
    STTNativeProvenance,
    STTProviderTurnIdentity,
    STTProviderTurnRequest,
    STTProviderTurnTerminal,
    STTProviderTurnUpdate,
    STTSessionProjection,
)
from puripuly_heart.core.stt.diagnostics import recognition_failure_diagnostics
from puripuly_heart.core.stt.session_projection import STTSessionEventProjection
from puripuly_heart.domain.recognition import (
    NativeAudioCoverage,
    NativeTranscriptionEvidence,
    NativeWordEvidence,
    RecognitionStreamIdentity,
    RecognitionTextOrigin,
    RecognitionUnitIdentity,
)

logger = logging.getLogger(__name__)

QWEN_AUDIO_MODEL = "qwen-audio-3.1-asr-flash-streaming"
QWEN_AUDIO_DEFAULT_ENDPOINT = "wss://dashscope.aliyuncs.com/api-ws/v1/inference"
QWEN_AUDIO_DEFAULT_HOTWORD_WEIGHT = 4
QWEN_AUDIO_LANGUAGE_HINTS_LIMIT = 4
QWEN_AUDIO_MAX_SENTENCES = 256
QWEN_AUDIO_MAX_MAP_RECORDS = 4096
QWEN_AUDIO_MAX_METADATA_BYTES = 256 * 1024


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
    if type(default_weight) is not int or default_weight not in (1, 2, 3, 4, 5, 50):
        default_weight = QWEN_AUDIO_DEFAULT_HOTWORD_WEIGHT
    source = (
        hotwords.items()
        if isinstance(hotwords, Mapping)
        else ((term, default_weight) for term in (hotwords or ()))
    )
    result: dict[str, int] = {}
    for raw_term, raw_weight in source:
        if not isinstance(raw_term, str):
            continue
        term = raw_term.strip()
        if term and term not in result:
            result[term] = (
                raw_weight
                if type(raw_weight) is int and raw_weight in (1, 2, 3, 4, 5, 50)
                else default_weight
            )
    return result


def _separator(left: str, right: str) -> str:
    if not left or not right or left[-1].isspace() or right[0].isspace():
        return ""
    if right[0] in ',.!?;:，。！？；：、)]}」』】》”’"' or (
        ord(left[-1]) >= 0x3000 or ord(right[0]) >= 0x3000
    ):
        return ""
    return " "


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
    channel: Literal["self", "peer"] = "self"

    async def open_session(
        self,
        *,
        projection: STTSessionProjection = LEGACY_STT_SESSION_PROJECTION,
    ) -> STTBackendSession:
        if self.sample_rate_hz not in (8000, 16000):
            raise ValueError("Qwen Audio requires 8000 or 16000 Hz PCM")
        if not self.api_key.strip() or not self.endpoint.strip():
            raise ValueError("Qwen Audio requires an API key and endpoint")
        if self.channel not in ("self", "peer"):
            raise ValueError("invalid Qwen Audio channel")
        if len(self.language_hints) > QWEN_AUDIO_LANGUAGE_HINTS_LIMIT or any(
            not isinstance(hint, str) or not hint.strip() for hint in self.language_hints
        ):
            raise ValueError("invalid Qwen Audio language hints")
        for value in (
            self.connect_timeout_s,
            self.task_start_timeout_s,
            self.task_finish_timeout_s,
            self.send_timeout_s,
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Qwen Audio timeouts must be positive and finite")
        return _QwenAudioSession(self, projection)

    @staticmethod
    async def verify_api_key(
        api_key: str,
        *,
        endpoint: str = QWEN_AUDIO_DEFAULT_ENDPOINT,
        model: str = QWEN_AUDIO_MODEL,
        websocket_factory: WebSocketFactory | None = None,
    ) -> bool:
        if not api_key.strip():
            return False
        backend = QwenAudioStreamingSTTBackend(
            api_key=api_key, endpoint=endpoint, model=model, websocket_factory=websocket_factory
        )
        session = _QwenAudioSession(backend, LEGACY_STT_SESSION_PROJECTION)
        try:
            task = await session._begin(None)
            await session._ready(task)
            await session._seal(task)
            await asyncio.wait_for(task.done.wait(), backend.task_finish_timeout_s)
            if task.failure is not None:
                raise task.failure
            return True
        finally:
            await session.close()


@dataclass(frozen=True, slots=True)
class _QwenTaskCapture:
    input_start: int
    input_end: int
    capture: AudioCaptureSpan
    context_only: bool


@dataclass(slots=True)
class _QwenTask:
    task_id: str
    identity: STTProviderTurnIdentity | None
    slot: _QwenSlot
    ready: asyncio.Event = field(default_factory=asyncio.Event)
    done: asyncio.Event = field(default_factory=asyncio.Event)
    started: bool = False
    sealed: bool = False
    failure: Exception | None = None
    readiness_owned: bool = False
    samples: int = 0
    captures: list[_QwenTaskCapture] = field(default_factory=list)
    sentence_ids: set[int] = field(default_factory=set)
    last_sentence_id: int = 0
    metadata_bytes: int = 0
    text: str = ""
    origins: tuple[RecognitionTextOrigin, ...] = ()
    provenance: list[STTNativeProvenance] = field(default_factory=list)
    final_wait: asyncio.Task[None] | None = None


@dataclass(slots=True)
class _QwenSlot:
    number: int
    websocket: Any = None
    receiver: asyncio.Task[None] | None = None
    task: _QwenTask | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class _QwenAudioSession(STTBackendSession):
    allows_sealed_turn_overlap = True
    max_inflight_turns = 2
    bridge_reset_preserves_vad = True
    readiness_retry_safe = True

    def __init__(self, backend: QwenAudioStreamingSTTBackend, projection: STTSessionProjection):
        self.backend = backend
        self._event_projection = STTSessionEventProjection(
            projection, allows_sealed_turn_overlap=True
        )
        self._slots = (_QwenSlot(0), _QwenSlot(1))
        self._next_slot = 0
        self._tasks: dict[STTProviderTurnIdentity, _QwenTask] = {}
        self._legacy_current: _QwenTask | None = None
        self._legacy_order: list[_QwenTask] = []
        self._legacy_events: asyncio.Queue[STTBackendTranscriptEvent | Exception | None] = (
            asyncio.Queue(maxsize=3)
        )
        self._legacy_capacity = asyncio.Semaphore(2)
        self._admission_lock = asyncio.Lock()
        self._closing = False
        self._aborted = False
        self._task_finished_count = 0
        self._stopped = False

    @property
    def state(self) -> QwenAudioSessionState:
        if self._closing:
            return QwenAudioSessionState.CLOSING
        task = self._legacy_current or next(reversed(self._tasks.values()), None)
        if task is None:
            return QwenAudioSessionState.STARTING_NEXT_TASK
        if task.failure is not None:
            return QwenAudioSessionState.FAILED
        if task.sealed:
            return QwenAudioSessionState.FINISHING_TASK
        return (
            QwenAudioSessionState.TASK_ACTIVE if task.started else QwenAudioSessionState.CONNECTING
        )

    @property
    def task_id(self) -> str | None:
        task = self._legacy_current or next(reversed(self._tasks.values()), None)
        return task.task_id if task is not None else None

    @property
    def task_finished_count(self) -> int:
        return self._task_finished_count

    def update_hotwords(self, hotwords: HotwordInput) -> None:
        self.backend.hotwords = hotwords

    async def _begin(self, identity: STTProviderTurnIdentity | None) -> _QwenTask:
        if self._closing:
            raise QwenAudioProtocolError("Qwen Audio session is closed")
        slot = self._slots[self._next_slot]
        previous = slot.task
        if previous is not None and not previous.done.is_set():
            await asyncio.wait_for(previous.done.wait(), self.backend.task_finish_timeout_s)
        if previous is not None and previous.final_wait is not None:
            await asyncio.gather(previous.final_wait, return_exceptions=True)
        if self._closing:
            raise QwenAudioProtocolError("Qwen Audio session is closed")
        task = _QwenTask(uuid.uuid4().hex, identity, slot)
        slot.task = task
        self._next_slot = 1 - self._next_slot
        return task

    async def begin_turn(self, request: STTProviderTurnRequest) -> None:
        if request.channel != self.backend.channel:
            raise QwenAudioProtocolError("Qwen Audio channel ownership mismatch")
        async with self._admission_lock:
            if len(self._tasks) >= 2:
                raise QwenAudioProtocolError("Qwen Audio turn capacity exceeded")
            self._event_projection.begin(request)
            self._tasks[request.identity] = await self._begin(request.identity)

    async def prepare_turn_audio(self, identity: STTProviderTurnIdentity) -> None:
        self._event_projection.require_open(identity)
        task = self._tasks[identity]
        task.readiness_owned = True
        try:
            await self._ready(task, publish_failure=False)
        finally:
            task.readiness_owned = False

    async def _discard_readiness_attempt(self, task: _QwenTask) -> None:
        try:
            await self._close_slot(task.slot)
        except BaseException:
            self._closing = True
            task.sealed = True
            task.done.set()
            raise
        task.done.set()
        identity = task.identity
        if (
            not self._aborted
            and not self._closing
            and identity is not None
            and self._tasks.get(identity) is task
        ):
            replacement = _QwenTask(uuid.uuid4().hex, identity, task.slot)
            task.slot.task = replacement
            self._tasks[identity] = replacement

    async def _ready(self, task: _QwenTask, *, publish_failure: bool = True) -> None:
        if task.started:
            return
        try:
            async with asyncio.timeout(self.backend.task_start_timeout_s):
                slot = task.slot
                if slot.websocket is None:
                    factory = (
                        self.backend.websocket_factory or network_clients.external_websocket_connect
                    )
                    headers = {"Authorization": f"Bearer {self.backend.api_key}"}
                    try:
                        connection = factory(
                            self.backend.endpoint,
                            additional_headers=headers,
                            ping_interval=None,
                            open_timeout=self.backend.connect_timeout_s,
                            max_size=QWEN_AUDIO_MAX_METADATA_BYTES,
                            max_queue=2,
                        )
                    except TypeError:
                        connection = factory(self.backend.endpoint, headers=headers)
                    slot.websocket = await connection
                    if self._aborted:
                        await self._close_slot(slot)
                        raise QwenAudioProtocolError("Qwen Audio admission revoked")
                    slot.receiver = asyncio.create_task(
                        self._receive(slot), name=f"qwen-audio-receive:{slot.number}"
                    )
                parameters: dict[str, object] = {
                    "format": "pcm",
                    "sample_rate": self.backend.sample_rate_hz,
                    "semantic_punctuation_enabled": False,
                    "max_sentence_silence": 800 if self.backend.channel == "peer" else 6000,
                    "multi_threshold_mode_enabled": False,
                    "heartbeat": True,
                }
                if self.backend.language_hints:
                    parameters["language_hints"] = list(self.backend.language_hints)
                vocabulary = _normalized_vocabulary(
                    self.backend.hotwords, default_weight=self.backend.hotword_weight
                )
                if vocabulary:
                    parameters["vocabulary"] = vocabulary
                await self._send(
                    slot,
                    json.dumps(
                        {
                            "header": {
                                "action": "run-task",
                                "task_id": task.task_id,
                                "streaming": "duplex",
                            },
                            "payload": {
                                "task_group": "audio",
                                "task": "asr",
                                "function": "recognition",
                                "model": self.backend.model,
                                "parameters": parameters,
                                "input": {},
                            },
                        },
                        ensure_ascii=False,
                    ),
                )
                await task.ready.wait()
                if task.failure is not None:
                    raise task.failure
        except asyncio.CancelledError:
            if publish_failure:
                await self._fail(
                    task,
                    QwenAudioProtocolError("Qwen Audio readiness cancelled"),
                    "qwen_readiness_cancelled",
                )
            else:
                await self._discard_readiness_attempt(task)
            raise
        except Exception as exc:
            if publish_failure:
                await self._fail(task, exc, "qwen_readiness_failed")
            else:
                await self._discard_readiness_attempt(task)
            raise

    async def _send(self, slot: _QwenSlot, value: str | bytes) -> None:
        if self._aborted or slot.websocket is None:
            raise QwenAudioProtocolError("Qwen Audio socket is unavailable")
        async with asyncio.timeout(self.backend.send_timeout_s):
            await slot.websocket.send(value)

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
        task = self._tasks[identity]
        await self._audio(task, pcm16le, source_ranges, context_only)
        self._event_projection.payload_written(identity, payload_sequence)

    async def _audio(
        self, task: _QwenTask, pcm: bytes, ranges: tuple[AudioCaptureSpan, ...], context: bool
    ) -> None:
        if not isinstance(pcm, bytes) or len(pcm) % 2:
            raise ValueError("Qwen Audio requires complete PCM16 samples")
        if not pcm:
            return
        if ranges and sum(span.normalized_sample_count for span in ranges) * 2 != len(pcm):
            raise QwenAudioProtocolError("Qwen Audio capture coverage does not match PCM")
        try:
            await self._ready(task)
            async with task.slot.lock:
                if task.sealed or task.done.is_set() or self._aborted:
                    raise QwenAudioProtocolError("Qwen Audio task cannot accept input")
                if len(task.captures) + len(ranges) > QWEN_AUDIO_MAX_MAP_RECORDS:
                    raise QwenAudioProtocolError("Qwen Audio capture map capacity exceeded")
                if task.metadata_bytes + len(ranges) * 192 > QWEN_AUDIO_MAX_METADATA_BYTES:
                    raise QwenAudioProtocolError("Qwen Audio evidence capacity exceeded")
                await self._send(task.slot, pcm)
                for span in ranges:
                    self._append_capture(task, span, context)
                task.samples += len(pcm) // 2 if not ranges else 0
        except asyncio.CancelledError:
            await self._fail(
                task, QwenAudioProtocolError("Qwen Audio write cancelled"), "qwen_write_cancelled"
            )
            raise
        except Exception as exc:
            await self._fail(task, exc, "qwen_write_failed")
            raise

    def _append_capture(self, task: _QwenTask, capture: AudioCaptureSpan, context: bool) -> None:
        end = task.samples + capture.normalized_sample_count
        previous = task.captures[-1] if task.captures else None
        if previous is not None and (
            previous.context_only == context
            and previous.capture.capture_epoch == capture.capture_epoch
            and previous.capture.source_sample_rate_hz == capture.source_sample_rate_hz
            and previous.capture.normalized_sample_rate_hz == capture.normalized_sample_rate_hz
            and previous.capture.source_end_sample == capture.source_start_sample
            and previous.capture.normalized_end_sample == capture.normalized_start_sample
            and previous.capture.source_end_monotonic_s == capture.source_start_monotonic_s
            and capture.discontinuity_before is None
        ):
            task.captures[-1] = replace(
                previous,
                input_end=end,
                capture=replace(
                    previous.capture,
                    source_end_sample=capture.source_end_sample,
                    normalized_end_sample=capture.normalized_end_sample,
                    source_end_monotonic_s=capture.source_end_monotonic_s,
                ),
            )
        else:
            task.captures.append(_QwenTaskCapture(task.samples, end, capture, context))
            task.metadata_bytes += 192
        task.samples = end

    async def seal_turn(
        self,
        identity: STTProviderTurnIdentity,
        *,
        sealed_content_ranges: tuple[AudioCaptureSpan, ...],
        seal_reason: str,
        observed_trailing_silence_ms: int | None,
    ) -> None:
        self._event_projection.seal(identity)
        await self._seal(self._tasks[identity])

    async def _seal(self, task: _QwenTask) -> None:
        if task.done.is_set() or task.sealed:
            return
        try:
            async with task.slot.lock:
                task.sealed = True
                if not task.started:
                    await self._complete(task)
                    return
                await self._send(
                    task.slot,
                    json.dumps(
                        {
                            "header": {
                                "action": "finish-task",
                                "task_id": task.task_id,
                                "streaming": "duplex",
                            },
                            "payload": {"input": {}},
                        }
                    ),
                )
                if not task.done.is_set():
                    task.final_wait = asyncio.create_task(
                        self._wait_final(task), name=f"qwen-audio-final:{task.task_id}"
                    )
        except asyncio.CancelledError:
            await self._fail(
                task, QwenAudioProtocolError("Qwen Audio finish cancelled"), "qwen_write_cancelled"
            )
            raise
        except Exception as exc:
            await self._fail(task, exc, "qwen_write_failed")
            raise

    async def _wait_final(self, task: _QwenTask) -> None:
        try:
            await asyncio.wait_for(task.done.wait(), self.backend.task_finish_timeout_s)
        except asyncio.TimeoutError as exc:
            await self._fail(task, exc, "qwen_final_timeout")

    async def _receive(self, slot: _QwenSlot) -> None:
        try:
            while slot.websocket is not None and not self._aborted:
                message = await slot.websocket.recv()
                if message is None:
                    raise QwenAudioProtocolError("Qwen Audio socket closed")
                await self._handle_message(slot, message)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            task = slot.task
            if task is not None and not task.done.is_set() and not self._aborted:
                await self._fail(task, exc, "qwen_socket_failed")
            else:
                await self._close_slot(slot)

    async def _handle_message(self, slot: _QwenSlot, message: object) -> None:
        if isinstance(message, (bytes, str)):
            if len(message) > QWEN_AUDIO_MAX_METADATA_BYTES:
                raise QwenAudioProtocolError("Qwen Audio event capacity exceeded")
            message = json.loads(message)
        if not isinstance(message, Mapping):
            return
        header = message.get("header")
        if not isinstance(header, Mapping):
            return
        task = slot.task
        if (
            task is None
            or task.done.is_set()
            or self._aborted
            or header.get("task_id") != task.task_id
        ):
            return
        event = header.get("event")
        if event == "task-started":
            if not task.started and not task.sealed:
                task.started = True
                task.ready.set()
            return
        async with slot.lock:
            if task.done.is_set():
                return
            if event == "result-generated" and task.started:
                await self._sentence(task, message)
            elif event == "task-finished":
                if not task.sealed:
                    await self._fail(
                        task,
                        QwenAudioProtocolError("Qwen Audio task ended before local end"),
                        "qwen_unexpected_terminal",
                    )
                else:
                    self._task_finished_count += 1
                    await self._complete(task)
            elif event == "task-failed":
                code = str(header.get("error_code") or "UNKNOWN")
                detail = str(header.get("error_message") or "Qwen Audio task failed")
                rejected = bool(self.backend.hotwords) and any(
                    token in f"{code} {detail}".casefold()
                    for token in ("vocabulary", "hotword", "workspace", "sub-workspace")
                )
                await self._fail(
                    task,
                    QwenAudioTaskFailedError(code, detail, hotwords_rejected=rejected),
                    "provider_task_failed",
                )

    async def inject_event(self, event: Mapping[str, object]) -> None:
        header = event.get("header")
        task_id = header.get("task_id") if isinstance(header, Mapping) else None
        for slot in self._slots:
            if slot.task is not None and slot.task.task_id == task_id:
                await self._handle_message(slot, event)
                return

    async def _sentence(self, task: _QwenTask, message: Mapping[str, object]) -> None:
        payload = message.get("payload")
        output = payload.get("output") if isinstance(payload, Mapping) else None
        sentence = output.get("sentence") if isinstance(output, Mapping) else None
        if (
            not isinstance(sentence, Mapping)
            or sentence.get("heartbeat")
            or not sentence.get("sentence_end")
        ):
            return
        try:
            sentence_id = int(str(sentence.get("sentence_id")))
        except ValueError:
            raise QwenAudioProtocolError("Qwen Audio final sentence identity is missing") from None
        if sentence_id == 0 or sentence_id in task.sentence_ids:
            return
        if sentence_id <= task.last_sentence_id:
            raise QwenAudioProtocolError("Qwen Audio final sentence order is inconsistent")
        text = sentence.get("text")
        if not isinstance(text, str):
            raise QwenAudioProtocolError("Qwen Audio final text is missing")
        begin = self._native_time(sentence.get("begin_time"))
        end = self._native_time(sentence.get("end_time"))
        words_value = sentence.get("words") or ()
        if (
            not isinstance(words_value, (tuple, list))
            or len(words_value) > QWEN_AUDIO_MAX_MAP_RECORDS
        ):
            raise QwenAudioProtocolError("Qwen Audio word evidence capacity exceeded")
        native = NativeTranscriptionEvidence(
            native_task_id=task.task_id,
            native_sentence_id=str(sentence_id),
            timestamp_basis="task_relative_estimated",
            begin_time_ms=begin,
            end_time_ms=end,
            input_sample_rate_hz=self.backend.sample_rate_hz,
            task_input_samples=task.samples,
            timed_words=tuple(
                NativeWordEvidence(
                    word.get("text") if isinstance(word.get("text"), str) else None,
                    word.get("punctuation") if isinstance(word.get("punctuation"), str) else None,
                    self._native_time(word.get("begin_time")),
                    self._native_time(word.get("end_time")),
                )
                for word in words_value
                if isinstance(word, Mapping)
            ),
            audio_coverage=self._coverage(task, begin, end),
        )
        size = native.retained_bytes + len(text.encode("utf-8")) + 128
        if (
            len(task.sentence_ids) >= QWEN_AUDIO_MAX_SENTENCES
            or task.metadata_bytes + size > QWEN_AUDIO_MAX_METADATA_BYTES
        ):
            raise QwenAudioProtocolError("Qwen Audio sentence evidence capacity exceeded")
        task.sentence_ids.add(sentence_id)
        task.last_sentence_id = sentence_id
        task.metadata_bytes += size
        start = len(task.text) + len(_separator(task.text, text))
        task.text += _separator(task.text, text) + text
        provenance = STTNativeProvenance(
            native_event_id=str(sentence_id),
            native_task_id=task.task_id,
            barrier="sentence_end",
            transcription=native,
        )
        task.provenance.append(provenance)
        identity = task.identity
        if identity is not None and text:
            stream = RecognitionStreamIdentity(
                self.backend.channel,
                identity.segment.activation_generation,
                identity.segment.capture_epoch,
                identity.provider_epoch_id,
                identity.settings_scope,
            )
            task.origins += (
                RecognitionTextOrigin(
                    RecognitionUnitIdentity(
                        stream,
                        uuid.uuid5(
                            uuid.NAMESPACE_URL,
                            f"{identity.provider_epoch_id}/{task.task_id}/{sentence_id}",
                        ),
                        len(task.sentence_ids),
                    ),
                    start,
                    len(task.text),
                    native,
                    source_binding="local_segment",
                    order_basis="source_order",
                ),
            )
            if self.backend.channel == "self":
                sequence = self._event_projection.next_update_sequence(identity)
                if sequence is not None and not self._event_projection.put_update(
                    STTProviderTurnUpdate(
                        identity,
                        sequence,
                        "stable",
                        "replace",
                        task.text,
                        provenance=provenance,
                        recognition_origins=task.origins,
                    )
                ):
                    raise QwenAudioProtocolError("Qwen Audio update capacity exceeded")

    @staticmethod
    def _native_time(value: object) -> float | None:
        return float(value) if type(value) in (int, float) and math.isfinite(value) else None

    def _coverage(
        self, task: _QwenTask, begin: float | None, end: float | None
    ) -> tuple[NativeAudioCoverage, ...]:
        rate = self.backend.sample_rate_hz
        left = math.floor(begin * rate / 1000) if begin is not None else 0
        right = math.ceil(end * rate / 1000) if end is not None else task.samples
        coverage = []
        for entry in task.captures:
            start, stop = max(left, entry.input_start), min(right, entry.input_end)
            if start >= stop:
                continue
            source_start = entry.capture.normalized_start_sample
            assert source_start is not None
            capture = entry.capture.slice_normalized(
                source_start + start - entry.input_start, source_start + stop - entry.input_start
            )
            gap = capture.discontinuity_before
            coverage.append(
                NativeAudioCoverage(
                    start,
                    stop,
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
                    gap.kind if gap else None,
                    gap.observed_at_monotonic_s if gap else None,
                    gap.lost_source_samples if gap else None,
                )
            )
        return tuple(coverage)

    async def _complete(self, task: _QwenTask, *, reason: str | None = None) -> None:
        if task.done.is_set():
            return
        identity = task.identity
        if identity is not None:
            self._event_projection.terminal(
                STTProviderTurnTerminal(
                    identity=identity,
                    outcome=(
                        "failed"
                        if task.failure is not None
                        else ("final" if task.text else "empty")
                    ),
                    text=task.text,
                    text_authority=(
                        "degraded"
                        if task.failure is not None and task.text
                        else "authoritative" if task.failure is None else "none"
                    ),
                    failure_reason=reason,
                    epoch_disposition="retire" if task.failure is not None else "reuse",
                    provenance=tuple(task.provenance)
                    + (
                        STTNativeProvenance(
                            native_task_id=task.task_id,
                            barrier="task-failed" if task.failure else "task-finished",
                        ),
                    ),
                    recognition_origins=task.origins,
                    failure_diagnostics=(
                        recognition_failure_diagnostics(
                            task.failure,
                            provider="qwen_audio",
                            channel=self.backend.channel,
                            operation="receive",
                            reason=reason or "qwen_task_failed",
                            identity=identity,
                        )
                        if task.failure is not None
                        else None
                    ),
                )
            )
        task.done.set()
        task.ready.set()
        final_wait = task.final_wait
        if final_wait is not None and final_wait is not asyncio.current_task():
            final_wait.cancel()
        if identity is None:
            await self._flush_legacy()

    async def _fail(self, task: _QwenTask, exc: Exception, reason: str) -> None:
        if task.done.is_set():
            return
        task.failure = exc
        if task.readiness_owned:
            task.ready.set()
            return
        self._closing = True
        logger.warning(
            "[QwenAudio] task failed channel=%s reason=%s cause_class=%s",
            self.backend.channel,
            reason,
            type(exc).__name__,
        )
        try:
            await self._close_slot(task.slot)
        finally:
            await self._complete(task, reason=reason)

    async def _close_slot(self, slot: _QwenSlot) -> None:
        websocket = slot.websocket
        receiver, slot.receiver = slot.receiver, None
        if receiver is not None and receiver is not asyncio.current_task():
            receiver.cancel()
            await asyncio.gather(receiver, return_exceptions=True)
        if websocket is not None:
            try:
                async with asyncio.timeout(self.backend.send_timeout_s):
                    await websocket.close()
            except Exception:
                abort = getattr(getattr(websocket, "transport", None), "abort", None)
                wait_closed = getattr(websocket, "wait_closed", None)
                if not callable(abort) or not callable(wait_closed):
                    self.readiness_retry_safe = False
                    raise
                try:
                    abort()
                    async with asyncio.timeout(self.backend.send_timeout_s):
                        await wait_closed()
                except BaseException:
                    self.readiness_retry_safe = False
                    raise
            except BaseException:
                self.readiness_retry_safe = False
                raise
            if slot.websocket is websocket:
                slot.websocket = None

    async def abort_turn(self, identity: STTProviderTurnIdentity, *, reason: str) -> None:
        await self.abort_for_toggle_off()

    async def abort_for_toggle_off(self) -> None:
        if self._aborted:
            return
        self._aborted = self._closing = True
        for slot in self._slots:
            task = slot.task
            if task is not None:
                task.done.set()
                task.ready.set()
                if task.final_wait is not None:
                    task.final_wait.cancel()
        await asyncio.gather(*(self._close_slot(slot) for slot in self._slots))
        timers = [
            slot.task.final_wait
            for slot in self._slots
            if slot.task is not None and slot.task.final_wait is not None
        ]
        if timers:
            await asyncio.gather(*timers, return_exceptions=True)
        self._tasks.clear()
        self._legacy_current = None
        self._legacy_order.clear()
        self._event_projection.close()
        if self._event_projection.is_legacy:
            while not self._legacy_events.empty():
                self._legacy_events.get_nowait()
            self._legacy_events.put_nowait(None)
            self._legacy_capacity.release()
            self._legacy_capacity.release()
        self._stopped = True

    async def turn_events(self):
        async for event in self._event_projection.turn_events():
            if isinstance(event, STTProviderTurnTerminal):
                self._tasks.pop(event.identity, None)
            yield event

    async def send_audio(self, pcm16le: bytes) -> None:
        if not self._event_projection.is_legacy:
            raise RuntimeError("Qwen Audio scoped sessions require owned turn input")
        if not pcm16le:
            return
        async with self._admission_lock:
            if self._closing:
                raise QwenAudioProtocolError("Qwen Audio session is closed")
            task = self._legacy_current
            if task is None:
                await asyncio.wait_for(
                    self._legacy_capacity.acquire(), self.backend.task_finish_timeout_s
                )
                task = await self._begin(None)
                self._legacy_current = task
                self._legacy_order.append(task)
            await self._audio(task, pcm16le, (), False)

    async def on_speech_end(
        self, *, trailing_silence_ms: int | None = None, reason: SpeechBoundaryReason | None = None
    ) -> None:
        async with self._admission_lock:
            task, self._legacy_current = self._legacy_current, None
            if task is not None:
                await self._seal(task)

    async def _flush_legacy(self) -> None:
        while self._legacy_order and self._legacy_order[0].done.is_set():
            task = self._legacy_order.pop(0)
            await self._legacy_events.put(
                task.failure or STTBackendTranscriptEvent(task.text, True)
            )

    async def events(self):
        while True:
            event = await self._legacy_events.get()
            if event is None:
                return
            self._legacy_capacity.release()
            if isinstance(event, Exception):
                raise event
            yield event

    async def stop(self) -> None:
        if self._stopped:
            return
        self._closing = True
        for slot in self._slots:
            task = slot.task
            if task is not None and not task.done.is_set():
                await self._seal(task)
        pending = [slot.task.done.wait() for slot in self._slots if slot.task is not None]
        if pending:
            await asyncio.gather(*pending)
        timers = [
            slot.task.final_wait
            for slot in self._slots
            if slot.task is not None and slot.task.final_wait is not None
        ]
        if timers:
            await asyncio.gather(*timers, return_exceptions=True)
        await asyncio.gather(*(self._close_slot(slot) for slot in self._slots))
        self._event_projection.close()
        if self._event_projection.is_legacy:
            await self._legacy_events.put(None)
        self._stopped = True

    async def close(self) -> None:
        await self.stop()


__all__ = [
    "QWEN_AUDIO_DEFAULT_ENDPOINT",
    "QWEN_AUDIO_DEFAULT_HOTWORD_WEIGHT",
    "QWEN_AUDIO_MODEL",
    "QwenAudioProtocolError",
    "QwenAudioSessionState",
    "QwenAudioStreamingSTTBackend",
    "QwenAudioTaskFailedError",
]
