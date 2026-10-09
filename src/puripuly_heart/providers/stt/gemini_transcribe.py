"""Gemini 3.5 Transcribe Live STT Backend using the official google-genai SDK."""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import logging
import math
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, ClassVar, Sequence
from uuid import uuid4

from puripuly_heart.core import network_clients
from puripuly_heart.core.audio.format import AudioCaptureSpan
from puripuly_heart.core.speech_boundary import SpeechBoundaryReason
from puripuly_heart.core.stt.backend import (
    LEGACY_STT_SESSION_PROJECTION,
    STTBackend,
    STTBackendSession,
    STTBackendTranscriptEvent,
    STTNativeProvenance,
    STTProviderInputTerminal,
    STTProviderTurnIdentity,
    STTProviderTurnRequest,
    STTRecognitionUnit,
    STTSessionProjection,
)
from puripuly_heart.core.stt.session_projection import STTSessionEventProjection
from puripuly_heart.core.stt.stream_input import STTStreamInputMap
from puripuly_heart.domain.recognition import (
    NativeTranscriptionEvidence,
    RecognitionStreamIdentity,
    RecognitionUnitIdentity,
)

logger = logging.getLogger(__name__)

GEMINI_TRANSCRIBE_STT_MODEL = "gemini-3.5-transcribe-live"
GEMINI_TRANSCRIBE_SAMPLE_RATE_HZ = 16000
GEMINI_TRANSCRIBE_DRAIN_TIMEOUT_S = 2.0
GEMINI_TRANSCRIBE_MAX_SESSION_AGE_S = 9.0 * 60.0


@dataclass(frozen=True, slots=True)
class _EndStream:
    reason: str
    completion: asyncio.Future[None]


@dataclass(frozen=True, slots=True)
class _AudioWrite:
    pcm16le: bytes
    source_ranges: tuple[AudioCaptureSpan, ...]
    completion: asyncio.Future[None]


@dataclass(slots=True)
class _GeminiClientResources:
    client: Any
    sync_transport: Any
    async_transport: Any
    config: Any


def _build_live_config_sync(language_codes: Sequence[str], custom_vocabulary: Sequence[str]) -> Any:
    from google.genai import types

    transcription_config_kwargs: dict[str, Any] = {"mode": "VERBATIM"}
    if language_codes:
        transcription_config_kwargs["language_codes"] = list(language_codes)
    if custom_vocabulary:
        transcription_config_kwargs["custom_vocabulary"] = list(custom_vocabulary)
    return types.LiveConnectConfig(
        response_modalities=["TEXT"],
        input_audio_transcription=types.AudioTranscriptionConfig(**transcription_config_kwargs),
        realtime_input_config=types.RealtimeInputConfig(
            automatic_activity_detection=types.AutomaticActivityDetection(
                prefix_padding_ms=500,
                silence_duration_ms=400,
            ),
        ),
    )


def _create_transports_sync() -> tuple[Any, Any]:
    sync_transport = network_clients.external_client(timeout=None, follow_redirects=True)
    try:
        async_transport = network_clients.external_async_client(timeout=None, follow_redirects=True)
    except BaseException:
        with contextlib.suppress(Exception):
            sync_transport.close()
        raise
    return sync_transport, async_transport


def _build_http_options_sync(sync_transport: Any, async_transport: Any) -> Any:
    from google.genai import types

    return types.HttpOptions(
        **network_clients.genai_http_options(
            sync_transport=sync_transport, async_transport=async_transport
        )
    )


def _create_genai_client_sync(api_key: str, http_options: Any) -> Any:
    from google import genai

    from .genai_network import configure_live_network

    client = genai.Client(api_key=api_key, http_options=http_options)
    configure_live_network(client, http_options)
    return client


def _prepare_gemini_resources_sync(
    api_key: str, language_codes: Sequence[str], custom_vocabulary: Sequence[str]
) -> _GeminiClientResources:
    config = _build_live_config_sync(language_codes, custom_vocabulary)
    sync_transport, async_transport = _create_transports_sync()
    try:
        http_options = _build_http_options_sync(sync_transport, async_transport)
        client = _create_genai_client_sync(api_key, http_options)
    except BaseException:
        with contextlib.suppress(Exception):
            sync_transport.close()
        with contextlib.suppress(Exception):
            asyncio.run(async_transport.aclose())
        raise
    return _GeminiClientResources(
        client=client,
        sync_transport=sync_transport,
        async_transport=async_transport,
        config=config,
    )


def gemini_transcribe_language_codes(source_language: str | None) -> list[str]:
    if not source_language:
        return []
    from puripuly_heart.core.language import gemini_transcribe_language_hint

    mapped = gemini_transcribe_language_hint(source_language)
    return [mapped] if mapped else []


_SAFE_EXCEPTION_CLASSES = frozenset(
    {
        "APIError",
        "ClientError",
        "ServerError",
        "GoAway",
        "GoAwayError",
        "BrokenPipeError",
        "ConnectionError",
        "ConnectionResetError",
        "ConnectionAbortedError",
        "ConnectionClosed",
        "ConnectionClosedError",
        "ConnectionClosedOK",
        "InvalidStatus",
        "InvalidStatusCode",
        "OSError",
        "RuntimeError",
        "TimeoutError",
        "TypeError",
        "ValueError",
        "ValidationError",
    }
)
_SAFE_API_STATUSES = frozenset(
    {
        "OK",
        "CANCELLED",
        "UNKNOWN",
        "INVALID_ARGUMENT",
        "DEADLINE_EXCEEDED",
        "NOT_FOUND",
        "ALREADY_EXISTS",
        "PERMISSION_DENIED",
        "RESOURCE_EXHAUSTED",
        "FAILED_PRECONDITION",
        "ABORTED",
        "OUT_OF_RANGE",
        "UNIMPLEMENTED",
        "INTERNAL",
        "UNAVAILABLE",
        "DATA_LOSS",
        "UNAUTHENTICATED",
        "GO_AWAY",
    }
)
_SAFE_RETIREMENT_REASONS = frozenset(
    {
        "gemini_write_failed",
        "gemini_receive_failed",
        "gemini_go_away",
        "gemini_connection_ended",
        "gemini_recognition_buffer_overflow",
        "gemini_write_cancelled",
    }
)


def _transport_failure_fields(exc: BaseException | None) -> tuple[str, int | str, str, str]:
    if exc is None:
        return "none", "none", "none", "none"
    name = type(exc).__name__
    exception_class = name if name in _SAFE_EXCEPTION_CLASSES else "unclassified"
    code = getattr(exc, "code", None)
    if code is None:
        code = getattr(getattr(exc, "rcvd", None), "code", None)
    if code is None:
        code = getattr(getattr(exc, "response", None), "status_code", None)
    api_code = code if type(code) is int and 0 <= code <= 4999 else "none"
    status = getattr(exc, "status", None)
    api_status = (
        status
        if type(status) is str and status in _SAFE_API_STATUSES
        else ("none" if status is None else "unclassified")
    )
    if exception_class in {"GoAway", "GoAwayError"} or api_status == "GO_AWAY":
        message_kind = "go_away"
    elif (
        isinstance(exc, ConnectionError)
        or exception_class in {"ConnectionClosed", "ConnectionClosedError", "ConnectionClosedOK"}
        or api_status == "UNAVAILABLE"
    ):
        message_kind = "connection_closed"
    elif isinstance(exc, TimeoutError) or api_status == "DEADLINE_EXCEEDED":
        message_kind = "timeout"
    elif (
        api_code in {400, 422}
        or api_status == "INVALID_ARGUMENT"
        or exception_class == "ValidationError"
    ):
        message_kind = "validation"
    else:
        message_kind = "other"
    return exception_class, api_code, api_status, message_kind


def _safe_epoch(value: str | None) -> str:
    if (
        value
        and len(value) <= 64
        and value.isascii()
        and all(character.isalnum() or character in "._-" for character in value)
    ):
        return value
    return "none"


@dataclass(slots=True)
class GeminiTranscribeSTTBackend(STTBackend):
    """Gemini 3.5 Transcribe Live STT Backend using the official google-genai SDK."""

    api_key: str
    language_codes: Sequence[str] = ()
    custom_vocabulary: Sequence[str] = ()
    model: str = GEMINI_TRANSCRIBE_STT_MODEL
    sample_rate_hz: int = GEMINI_TRANSCRIBE_SAMPLE_RATE_HZ
    connect_timeout_s: float = 10.0
    drain_timeout_s: float = GEMINI_TRANSCRIBE_DRAIN_TIMEOUT_S
    live_connect_factory: Callable[[str, Any], Any] | None = None

    async def open_session(
        self,
        *,
        projection: STTSessionProjection = LEGACY_STT_SESSION_PROJECTION,
    ) -> STTBackendSession:
        if self.sample_rate_hz != GEMINI_TRANSCRIBE_SAMPLE_RATE_HZ:
            raise ValueError(
                "sample_rate_hz must be 16000 for Gemini Transcribe Live transcription"
            )
        if not self.api_key:
            raise ValueError("api_key must be non-empty")
        if self.connect_timeout_s <= 0:
            raise ValueError("connect_timeout_s must be > 0")
        if self.drain_timeout_s <= 0:
            raise ValueError("drain_timeout_s must be > 0")
        session = _GeminiTranscribeLiveSession(
            api_key=self.api_key,
            language_codes=list(self.language_codes),
            custom_vocabulary=list(self.custom_vocabulary),
            model=self.model,
            sample_rate_hz=self.sample_rate_hz,
            connect_timeout_s=self.connect_timeout_s,
            drain_timeout_s=self.drain_timeout_s,
            live_connect_factory=self.live_connect_factory,
            projection=projection,
        )
        try:
            await session.start()
        except BaseException:
            with contextlib.suppress(BaseException):
                await session.close()
            raise
        return session

    @staticmethod
    async def verify_api_key(api_key: str) -> bool:
        if not api_key:
            return False

        def _check() -> bool:
            with network_clients.external_client(timeout=5, follow_redirects=True) as client:
                response = client.get(
                    "https://generativelanguage.googleapis.com/v1beta/models",
                    headers={"x-goog-api-key": api_key},
                )
                response.raise_for_status()
                return response.status_code == 200

        return await asyncio.to_thread(_check)


@dataclass(slots=True)
class _GeminiTranscribeLiveSession(STTBackendSession):
    """Internal session wrapping a google-genai Live API AsyncSession."""

    api_key: str
    language_codes: list[str]
    custom_vocabulary: list[str]
    model: str
    sample_rate_hz: int
    connect_timeout_s: float
    drain_timeout_s: float
    live_connect_factory: Callable[[str, Any], Any] | None = None
    projection: STTSessionProjection = LEGACY_STT_SESSION_PROJECTION
    max_session_age_s: ClassVar[float] = GEMINI_TRANSCRIBE_MAX_SESSION_AGE_S

    _event_projection: STTSessionEventProjection = field(init=False, repr=False)
    _send_queue: asyncio.Queue[_EndStream | _AudioWrite] = field(init=False, repr=False)
    _live_context: Any = field(init=False, default=None, repr=False)
    _live_session: Any = field(init=False, default=None, repr=False)
    _send_task: asyncio.Task[None] | None = field(init=False, default=None, repr=False)
    _recv_task: asyncio.Task[None] | None = field(init=False, default=None, repr=False)
    _stopped: bool = field(init=False, default=False)
    _stream: RecognitionStreamIdentity | None = field(init=False, default=None, repr=False)
    _input_map: STTStreamInputMap = field(init=False, default_factory=STTStreamInputMap)
    _receipt_sequence: int = field(init=False, default=0)
    _queued_audio_bytes: int = field(init=False, default=0)
    _audio_since_fence: bool = field(init=False, default=False)
    _protocol_failed: bool = field(init=False, default=False)
    _client_resources: _GeminiClientResources | None = field(init=False, default=None, repr=False)
    _connected_at_s: float | None = field(init=False, default=None, repr=False)
    _go_away_timer: asyncio.TimerHandle | None = field(init=False, default=None, repr=False)
    _go_away_active: bool = field(init=False, default=False)
    _setup_future: Any = field(init=False, default=None, repr=False)
    _setup_executor: Any = field(init=False, default=None, repr=False)
    _handshake_task: asyncio.Task[Any] | None = field(init=False, default=None, repr=False)
    _teardown_task: asyncio.Task[None] | None = field(init=False, default=None, repr=False)
    _teardown_done: bool = field(init=False, default=False)
    _scoped_request: STTProviderTurnRequest | None = field(init=False, default=None, repr=False)
    _sealing: bool = field(init=False, default=False)
    allows_sealed_turn_overlap: ClassVar[bool] = True
    accepts_stream_input: ClassVar[bool] = True
    independent_recognition_units: ClassVar[bool] = True

    def __post_init__(self) -> None:
        self._event_projection = STTSessionEventProjection(self.projection)
        self._send_queue = asyncio.Queue(maxsize=258)

    async def start(self) -> None:
        if self._stopped:
            raise RuntimeError("Gemini Transcribe Live session is closed")
        self._teardown_done = False
        self._teardown_task = None
        self._setup_future = None
        self._setup_executor = None
        self._handshake_task = None
        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="gemini-stt-setup"
        )
        self._setup_executor = executor
        try:
            raw = executor.submit(
                _prepare_gemini_resources_sync,
                self.api_key,
                list(self.language_codes),
                list(self.custom_vocabulary),
            )
        except BaseException:
            self._setup_executor = None
            executor.shutdown(wait=False)
            raise
        self._setup_future = raw
        try:
            resources = await asyncio.wrap_future(raw)
        except BaseException:
            try:
                await self._teardown()
            except asyncio.CancelledError, Exception:
                pass
            raise
        if self._stopped:
            try:
                await self._teardown()
            except asyncio.CancelledError, Exception:
                pass
            raise RuntimeError("Gemini Transcribe Live session closed during setup")
        self._client_resources = resources
        executor.shutdown(wait=False)
        try:
            factory = self.live_connect_factory
            if factory is None:
                factory = resources.client.aio.live.connect
            live_context = factory(model=self.model, config=resources.config)
        except BaseException:
            try:
                await self._teardown()
            except asyncio.CancelledError, Exception:
                pass
            raise
        self._live_context = live_context
        handshake_task = asyncio.create_task(live_context.__aenter__())
        self._handshake_task = handshake_task
        try:
            live_session = await asyncio.wait_for(handshake_task, timeout=self.connect_timeout_s)
        except BaseException as exc:
            if isinstance(exc, Exception) and self.live_connect_factory is None:
                from .genai_network import annotate_live_error

                annotate_live_error(exc, resources.client)
            try:
                await self._teardown()
            except asyncio.CancelledError, Exception:
                pass
            raise
        self._handshake_task = None
        if self._stopped:
            try:
                await self._teardown()
            except asyncio.CancelledError, Exception:
                pass
            raise RuntimeError("Gemini Transcribe Live session closed during connect")
        self._live_session = live_session
        self._connected_at_s = asyncio.get_running_loop().time()
        self._send_task = asyncio.create_task(self._send_loop())
        self._recv_task = asyncio.create_task(self._recv_loop())

    def _start_teardown(self) -> asyncio.Task[None] | None:
        if self._teardown_task is None and not self._teardown_done:
            self._teardown_task = asyncio.create_task(self._teardown_body())
        return self._teardown_task

    async def _teardown(self) -> None:
        task = self._start_teardown()
        if task is None:
            return
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue

    async def _teardown_body(self) -> None:
        self._cancel_go_away_timer()
        raw = self._setup_future
        if raw is not None and not raw.done():
            try:
                await asyncio.wrap_future(raw)
            except asyncio.CancelledError:
                body_task = asyncio.current_task()
                if body_task is not None and body_task.cancelling():
                    raise
            except Exception:
                pass
        outcome = None
        if raw is not None and raw.done():
            try:
                outcome = raw.result()
            except BaseException:
                outcome = None
        if outcome is not None and outcome is not self._client_resources:
            await self._close_resources(outcome)
        executor, self._setup_executor = self._setup_executor, None
        if executor is not None:
            executor.shutdown(wait=False)
        handshake_task, self._handshake_task = self._handshake_task, None
        if handshake_task is not None:
            if not handshake_task.done():
                handshake_task.cancel()
            try:
                await handshake_task
            except asyncio.CancelledError:
                body_task = asyncio.current_task()
                if body_task is not None and body_task.cancelling():
                    raise
            except Exception:
                pass
        for task in (self._send_task, self._recv_task):
            if task is not None:
                task.cancel()
        pending_tasks = [task for task in (self._send_task, self._recv_task) if task is not None]
        if pending_tasks:
            await asyncio.gather(*pending_tasks, return_exceptions=True)
        self._send_task = None
        self._recv_task = None
        live_context, self._live_context = self._live_context, None
        live_session, self._live_session = self._live_session, None
        if live_context is not None:
            with contextlib.suppress(Exception):
                await live_context.__aexit__(None, None, None)
        elif live_session is not None:
            with contextlib.suppress(Exception):
                await live_session.close()
        await self._release_client_resources()
        self._teardown_done = True

    async def _close_resources(self, resources: _GeminiClientResources) -> None:
        with contextlib.suppress(Exception):
            await resources.client.aio.aclose()
        with contextlib.suppress(Exception):
            await asyncio.to_thread(resources.client.close)
        with contextlib.suppress(Exception):
            await resources.async_transport.aclose()
        with contextlib.suppress(Exception):
            await asyncio.to_thread(resources.sync_transport.close)

    async def _release_client_resources(self) -> None:
        resources, self._client_resources = self._client_resources, None
        if resources is None:
            return
        await self._close_resources(resources)

    async def _send_loop(self) -> None:
        item: _EndStream | _AudioWrite | None = None
        try:
            while not self._stopped:
                item = await self._send_queue.get()
                if self._protocol_failed:
                    self._resolve_write(item.completion, RuntimeError("Gemini writer retired"))
                    return
                if item.completion.cancelled():
                    if isinstance(item, _AudioWrite):
                        self._queued_audio_bytes -= len(item.pcm16le)
                    continue
                if isinstance(item, _EndStream):
                    if self._audio_since_fence:
                        await self._send_realtime(audio_stream_end=True)
                        self._audio_since_fence = False
                else:
                    self._queued_audio_bytes -= len(item.pcm16le)
                    pcm, mapped = (
                        self._input_map.prepare(
                            item.pcm16le,
                            item.source_ranges,
                        )
                        if self._event_projection.is_scoped
                        else (item.pcm16le, None)
                    )
                    if pcm:
                        await self._send_realtime(
                            audio={
                                "data": pcm,
                                "mime_type": f"audio/pcm;rate={self.sample_rate_hz}",
                            }
                        )
                        self._audio_since_fence = True
                        if mapped is not None:
                            self._input_map.commit(mapped)
                self._resolve_write(item.completion, None)
        except asyncio.CancelledError:
            self._resolve_write(
                getattr(item, "completion", None), RuntimeError("Gemini writer cancelled")
            )
            raise
        except Exception as exc:
            self._resolve_write(getattr(item, "completion", None), exc)
            self._log_transport_diagnostic("transport_failure", "send", exception=exc)
            self._put_event(exc)
            self._scoped_transport_failure("gemini_write_failed", exception=exc)
        finally:
            self._fail_pending_writes()

    async def _recv_loop(self) -> None:
        try:
            while not self._stopped and not self._protocol_failed:
                live_session = self._live_session
                if live_session is None:
                    return
                async for message in live_session.receive():
                    self._handle_message(message)
                    if self._protocol_failed:
                        return
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._log_transport_diagnostic("transport_failure", "receive", exception=exc)
            self._put_event(exc)
            self._scoped_transport_failure("gemini_receive_failed", exception=exc)
        finally:
            self._put_event(None)
            if not self._stopped:
                self._scoped_transport_failure(
                    "gemini_go_away" if self._go_away_active else "gemini_connection_ended"
                )
                if self._go_away_active:
                    self._start_teardown()

    def _handle_message(self, message: Any) -> None:
        if self._protocol_failed or self._stopped:
            return
        content = getattr(message, "server_content", None)
        final = getattr(content, "input_transcription", None)
        if final is not None:
            text = str(getattr(final, "text", None) or "")
            if self._event_projection.is_legacy:
                self._put_event(STTBackendTranscriptEvent(text=text, is_final=True))
            elif self._stream is not None:
                self._receipt_sequence += 1
                unit = STTRecognitionUnit(
                    identity=RecognitionUnitIdentity(self._stream, uuid4(), self._receipt_sequence),
                    text=text,
                    provenance=STTNativeProvenance(
                        transcription=NativeTranscriptionEvidence(
                            finished=getattr(final, "finished", None),
                            language_code=getattr(final, "language_code", None),
                            speaker_label=getattr(final, "speaker_label", None),
                            words=tuple(
                                (
                                    getattr(word, "word", None),
                                    (
                                        str(word.start_offset)
                                        if getattr(word, "start_offset", None) is not None
                                        else None
                                    ),
                                    (
                                        str(word.end_offset)
                                        if getattr(word, "end_offset", None) is not None
                                        else None
                                    ),
                                )
                                for word in (getattr(final, "words", None) or ())
                            ),
                        )
                    ),
                )
                if not self._event_projection.put_recognition(unit):
                    self._scoped_transport_failure("gemini_recognition_buffer_overflow")
        go_away = getattr(message, "go_away", None)
        if go_away is not None and not self._go_away_active:
            self._go_away_active = True
            time_left = getattr(go_away, "time_left", None)
            try:
                grace_s = float(time_left[:-1]) if time_left and time_left.endswith("s") else 0.0
            except ValueError, TypeError:
                grace_s = 0.0
            if not math.isfinite(grace_s) or grace_s <= 0:
                self._scoped_transport_failure("gemini_go_away")
                self._start_teardown()
                return
            if self._connected_at_s is not None:
                grace_s = min(
                    grace_s,
                    max(
                        0.0,
                        self.max_session_age_s
                        - (asyncio.get_running_loop().time() - self._connected_at_s),
                    ),
                )
            if grace_s <= 0:
                self._scoped_transport_failure("gemini_go_away")
                self._start_teardown()
                return
            self._go_away_timer = asyncio.get_running_loop().call_later(
                grace_s, self._expire_go_away
            )

    def _cancel_go_away_timer(self) -> None:
        timer, self._go_away_timer = self._go_away_timer, None
        if timer is not None:
            timer.cancel()

    def _expire_go_away(self) -> None:
        self._go_away_timer = None
        if self._stopped or self._protocol_failed:
            return
        self._scoped_transport_failure("gemini_go_away")
        self._start_teardown()

    @staticmethod
    def _resolve_write(
        completion: asyncio.Future[None] | None,
        error: BaseException | None,
    ) -> None:
        if completion is None or completion.done():
            return
        if error is None:
            completion.set_result(None)
        else:
            completion.set_exception(error)

    def _fail_pending_writes(self) -> None:
        error = RuntimeError("Gemini Transcribe Live writer stopped")
        self._queued_audio_bytes = 0
        while not self._send_queue.empty():
            try:
                item = self._send_queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            self._resolve_write(getattr(item, "completion", None), error)

    def _log_transport_diagnostic(
        self,
        event: str,
        operation: str,
        *,
        reason: str = "none",
        exception: BaseException | None = None,
    ) -> None:
        channel = (
            self._stream.channel
            if self._stream is not None
            else (self._scoped_request.channel if self._scoped_request is not None else "none")
        )
        exception_class, api_code, api_status, message_kind = _transport_failure_fields(exception)
        logger.log(
            logging.ERROR if exception is not None else logging.INFO,
            "[GeminiTranscribe] %s provider=gemini_transcribe channel=%s epoch=%s "
            "operation=%s reason=%s exception_class=%s api_code=%s api_status=%s "
            "message_kind=%s go_away=%s",
            event,
            channel if channel in {"self", "peer"} else "none",
            _safe_epoch(self._event_projection.provider_epoch_id),
            operation,
            reason if reason in _SAFE_RETIREMENT_REASONS else "none",
            exception_class,
            api_code,
            api_status,
            message_kind,
            str(self._go_away_active).lower(),
        )

    def _scoped_transport_failure(
        self,
        reason: str,
        *,
        orderly: bool = False,
        exception: BaseException | None = None,
    ) -> None:
        self._cancel_go_away_timer()
        if self._protocol_failed:
            return
        self._log_transport_diagnostic(
            "epoch_retired", "retire", reason=reason, exception=exception
        )
        self._protocol_failed = True
        request, self._scoped_request = self._scoped_request, None
        if request is not None:
            self._event_projection.terminal(
                STTProviderInputTerminal(
                    request.identity, "failed", request.channel, reason, "retire"
                )
            )
        self._event_projection.end_epoch(
            orderly=orderly,
            reason=reason,
            provider_turn_id=request.identity.provider_turn_id if request else None,
        )
        self._fail_pending_writes()

    def _require_scoped_turn(self, identity: STTProviderTurnIdentity) -> STTProviderTurnRequest:
        self._event_projection.require_open(identity)
        request = self._scoped_request
        if request is None or request.identity != identity:
            raise RuntimeError("Gemini scoped input identity mismatch")
        return request

    async def begin_stream(self, stream: RecognitionStreamIdentity) -> None:
        if self._stopped or self._protocol_failed:
            raise RuntimeError("Gemini Transcribe Live session is closed")
        if not self._event_projection.is_scoped:
            raise RuntimeError("Scoped Gemini stream requires source ownership")
        if self._stream is not None and self._stream != stream:
            raise RuntimeError("Gemini stream ownership changed without retirement")
        if stream.provider_epoch_id != self._event_projection.provider_epoch_id:
            raise RuntimeError("Gemini stream belongs to a different provider epoch")
        self._stream = stream

    async def begin_turn(self, request: STTProviderTurnRequest) -> None:
        identity = request.identity
        stream = RecognitionStreamIdentity(
            request.channel,
            identity.segment.activation_generation,
            identity.segment.capture_epoch,
            identity.provider_epoch_id,
            identity.settings_scope,
        )
        await self.begin_stream(stream)
        self._event_projection.begin(request)
        self._scoped_request = request

    async def _enqueue(self, item: _AudioWrite | _EndStream) -> None:
        if self._stopped or self._protocol_failed:
            raise RuntimeError("Gemini Transcribe Live session is closed")
        if self._send_queue.full():
            raise RuntimeError("Gemini Transcribe Live audio queue overflow")
        if isinstance(item, _AudioWrite):
            if self._queued_audio_bytes + len(item.pcm16le) > 4 * 1024 * 1024:
                raise RuntimeError("Gemini Transcribe Live audio byte capacity exceeded")
            self._queued_audio_bytes += len(item.pcm16le)
        self._send_queue.put_nowait(item)
        try:
            await item.completion
        except asyncio.CancelledError:
            self._scoped_transport_failure("gemini_write_cancelled")
            if self._send_task is not None:
                self._send_task.cancel()
            self._start_teardown()
            raise

    async def send_turn_audio(
        self,
        identity: STTProviderTurnIdentity,
        pcm16le: bytes,
        *,
        payload_sequence: int,
        source_ranges: tuple[AudioCaptureSpan, ...],
        context_only: bool,
    ) -> None:
        self._require_scoped_turn(identity)
        if self._sealing:
            raise RuntimeError("Gemini scoped input is sealing")
        self._event_projection.validate_payload(identity, payload_sequence)
        await self._enqueue(
            _AudioWrite(
                pcm16le,
                source_ranges,
                asyncio.get_running_loop().create_future(),
            )
        )
        self._event_projection.payload_written(identity, payload_sequence)

    async def send_stream_audio(
        self, pcm16le: bytes, *, source_ranges: tuple[AudioCaptureSpan, ...]
    ) -> None:
        if self._stream is None:
            raise RuntimeError("Gemini stream has no admitted source")
        await self._enqueue(
            _AudioWrite(pcm16le, source_ranges, asyncio.get_running_loop().create_future())
        )

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
        request = self._require_scoped_turn(identity)
        if self._sealing:
            raise RuntimeError("Gemini scoped input is already sealing")
        self._sealing = True
        await self._enqueue(_EndStream(seal_reason, asyncio.get_running_loop().create_future()))
        if sealed_content_ranges and not self._input_map.covers(sealed_content_ranges):
            raise RuntimeError("Gemini sealed input contains unsubmitted audio")
        self._event_projection.seal(identity)
        self._event_projection.terminal(
            STTProviderInputTerminal(identity, "submitted", request.channel)
        )
        self._scoped_request = None
        self._sealing = False

    async def end_stream(self, *, reason: str) -> None:
        await self._enqueue(_EndStream(reason, asyncio.get_running_loop().create_future()))
        if reason == "source_eof":
            await asyncio.sleep(self.drain_timeout_s)

    async def abort_turn(self, identity: STTProviderTurnIdentity, *, reason: str) -> None:
        request = self._require_scoped_turn(identity)
        self._event_projection.terminal(
            STTProviderInputTerminal(identity, "cancelled", request.channel, reason, "retire")
        )
        self._scoped_request = None
        self._scoped_transport_failure(reason)

    async def turn_events(self):
        async for event in self._event_projection.turn_events():
            yield event

    async def send_audio(self, pcm16le: bytes) -> None:
        if not self._event_projection.is_legacy:
            raise RuntimeError("Scoped Gemini audio requires source ownership")
        await self._enqueue(_AudioWrite(pcm16le, (), asyncio.get_running_loop().create_future()))

    async def on_speech_end(
        self,
        *,
        trailing_silence_ms: int | None = None,
        reason: SpeechBoundaryReason | None = None,
    ) -> None:
        await self._enqueue(
            _EndStream(str(reason or "local_end"), asyncio.get_running_loop().create_future())
        )

    async def _send_realtime(self, **kwargs: Any) -> None:
        live_session = self._live_session
        if live_session is None:
            raise RuntimeError("Gemini Transcribe Live transport is unavailable")
        await live_session.send_realtime_input(**kwargs)

    async def stop(self) -> None:
        if self._stopped:
            return
        self._cancel_go_away_timer()
        request, self._scoped_request = self._scoped_request, None
        if request is not None:
            self._event_projection.terminal(
                STTProviderInputTerminal(
                    request.identity, "cancelled", request.channel, "stopped", "retire"
                )
            )
        self._event_projection.end_epoch(orderly=True, reason="stopped")
        self._stopped = True
        self._fail_pending_writes()
        if self._send_task is not None:
            self._send_task.cancel()
        self._put_event(None)

    async def close(self) -> None:
        await self.stop()
        current_task = asyncio.current_task()
        try:
            await self._teardown()
        except asyncio.CancelledError, Exception:
            pass
        self._event_projection.close()
        if current_task is not None and current_task.cancelling():
            raise asyncio.CancelledError

    async def events(self) -> AsyncIterator[STTBackendTranscriptEvent]:
        async for event in self._event_projection.events():
            yield event

    def _put_event(self, event: STTBackendTranscriptEvent | BaseException | None) -> None:
        self._event_projection.put_legacy(event)
