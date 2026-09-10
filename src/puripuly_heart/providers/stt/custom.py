from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
from collections import deque
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import httpx

from puripuly_heart.core.audio.format import AudioCaptureSpan
from puripuly_heart.core.audio.ownership import SegmentTerminalOutcome
from puripuly_heart.core.speech_boundary import SpeechBoundaryReason
from puripuly_heart.core.stt.backend import (
    STTBackend,
    STTBackendSession,
    STTBackendTranscriptEvent,
    STTNativeProvenance,
    STTProviderEpochEnded,
    STTProviderTurnIdentity,
    STTProviderTurnRequest,
    STTProviderTurnTerminal,
    STTProviderTurnUpdate,
)
from puripuly_heart.core.stt.custom import (
    CUSTOM_STT_CAPABILITY_LANGUAGE_HINT,
    CUSTOM_STT_COMPAT_OPENAI_REALTIME,
    CUSTOM_STT_COMPAT_OPENAI_TRANSCRIPTION,
    CUSTOM_STT_MODE_OFFLINE,
    CUSTOM_STT_MODE_REALTIME,
    CUSTOM_STT_VALIDATION_AUTH_FAILURE,
    CUSTOM_STT_VALIDATION_COMPATIBILITY_MISMATCH,
    CUSTOM_STT_VALIDATION_MODEL_UNAVAILABLE,
    CUSTOM_STT_VALIDATION_UNREACHABLE,
    CustomSTTConfigurationError,
    append_custom_stt_query,
    classify_http_failure,
    compatibility_supports,
    language_hint_for_source,
    normalize_custom_stt_extra,
    resolve_openai_realtime_url,
    resolve_openai_transcription_url,
    sanitize_custom_stt_text,
    sanitize_endpoint_for_display,
    validate_mode_compatibility,
)
from puripuly_heart.core.stt.custom_connection import (
    authorization_headers,
    extract_transcript_text,
    parse_realtime_event,
    pcm16le_to_wav,
    safe_body_excerpt,
)
from puripuly_heart.core.stt.scoped_event_buffer import STTProviderEventBuffer

logger = logging.getLogger(__name__)

_OFFLINE_TIMEOUT = httpx.Timeout(connect=5.0, read=30.0, write=10.0, pool=5.0)
_OFFLINE_TOTAL_TIMEOUT_S = 50.0
_STREAM_CONNECT_TIMEOUT_S = 5.0
_STREAM_FINAL_TIMEOUT_S = 20.0
_FINAL_EVENT_TYPES = frozenset(
    {
        "conversation.item.input_audio_transcription.completed",
        "conversation.item.input_audio_transcription.failed",
        "input_audio_transcription.completed",
        "input_audio_transcription.failed",
    }
)
_PARTIAL_EVENT_TYPES = frozenset(
    {
        "conversation.item.input_audio_transcription.delta",
        "input_audio_transcription.delta",
        "response.audio_transcript.delta",
        "transcript.delta",
    }
)


class CustomSTTRequestError(RuntimeError):
    def __init__(self, message: str, *, category: str) -> None:
        super().__init__(message)
        self.category = category


@dataclass(slots=True)
class CustomSTTBackend(STTBackend):
    mode: str
    compatibility: str
    endpoint: str
    model: str
    api_key: str = ""
    source_language: str = ""
    sample_rate_hz: int = 16000
    extra: Mapping[str, object] = field(default_factory=dict)
    http_client_factory: Callable[..., httpx.AsyncClient] = httpx.AsyncClient
    websocket_connect: Callable[..., Any] | None = None

    def __post_init__(self) -> None:
        validate_mode_compatibility(self.mode, self.compatibility)
        if self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be > 0")
        self.extra = normalize_custom_stt_extra(self.extra)

    async def open_session(self) -> STTBackendSession:
        if self.mode == CUSTOM_STT_MODE_OFFLINE:
            if self.compatibility != CUSTOM_STT_COMPAT_OPENAI_TRANSCRIPTION:
                raise CustomSTTConfigurationError(
                    f"unsupported offline compatibility: {self.compatibility}"
                )
            session = _OfflineOpenAITranscriptionSession(
                endpoint=self.endpoint,
                model=self.model,
                api_key=self.api_key,
                source_language=self.source_language,
                sample_rate_hz=self.sample_rate_hz,
                extra=self.extra,
                http_client_factory=self.http_client_factory,
            )
            await session.start()
            return session
        if self.mode == CUSTOM_STT_MODE_REALTIME:
            if self.compatibility != CUSTOM_STT_COMPAT_OPENAI_REALTIME:
                raise CustomSTTConfigurationError(
                    f"unsupported realtime compatibility: {self.compatibility}"
                )
            session = _StreamingOpenAIRealtimeSession(
                endpoint=self.endpoint,
                model=self.model,
                api_key=self.api_key,
                source_language=self.source_language,
                sample_rate_hz=self.sample_rate_hz,
                extra=self.extra,
                websocket_connect=self.websocket_connect,
            )
            await session.start()
            return session
        raise CustomSTTConfigurationError(f"unsupported Custom STT mode: {self.mode}")


@dataclass(slots=True)
class _OfflineOpenAITranscriptionSession(STTBackendSession):
    endpoint: str
    model: str
    api_key: str
    source_language: str
    sample_rate_hz: int
    http_client_factory: Callable[..., httpx.AsyncClient]
    extra: Mapping[str, object] = field(default_factory=dict)

    _events: asyncio.Queue[STTBackendTranscriptEvent | BaseException | None] = field(
        init=False, repr=False
    )
    _buffer: bytearray = field(init=False, repr=False)
    _transcribe_lock: asyncio.Lock = field(init=False, repr=False)
    _client: httpx.AsyncClient | None = field(init=False, default=None, repr=False)
    _stopped: bool = field(init=False, default=False)
    _url: str = field(init=False, default="", repr=False)
    _scoped_events: STTProviderEventBuffer = field(
        init=False, default_factory=STTProviderEventBuffer, repr=False
    )
    _scoped_identity: STTProviderTurnIdentity | None = field(init=False, default=None, repr=False)
    _scoped_payload_sequence: int = field(init=False, default=0, repr=False)
    _scoped_buffer: bytearray = field(init=False, repr=False)
    _scoped_task: asyncio.Task[None] | None = field(init=False, default=None, repr=False)
    _scoped_epoch_retired: bool = field(init=False, default=False, repr=False)

    def __post_init__(self) -> None:
        self._events = asyncio.Queue()
        self._buffer = bytearray()
        self._transcribe_lock = asyncio.Lock()
        self._scoped_buffer = bytearray()
        self._url = resolve_openai_transcription_url(self.endpoint)

    async def start(self) -> None:
        self._client = self.http_client_factory(
            timeout=_OFFLINE_TIMEOUT,
            trust_env=False,
            follow_redirects=False,
        )
        logger.info(
            "[STT] Custom offline session ready endpoint=%s",
            sanitize_endpoint_for_display(self._url),
        )

    async def send_audio(self, pcm16le: bytes) -> None:
        if self._stopped or not pcm16le:
            return
        self._buffer.extend(pcm16le)

    async def begin_turn(self, request: STTProviderTurnRequest) -> None:
        if self._stopped or self._scoped_epoch_retired:
            raise RuntimeError("Custom offline STT session is unavailable")
        if self._scoped_identity is not None:
            raise RuntimeError("Custom offline STT session already has an unresolved turn")
        self._scoped_identity = request.identity
        self._scoped_payload_sequence = 0
        self._scoped_buffer.clear()

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
        self._require_scoped_identity(identity)
        if payload_sequence <= self._scoped_payload_sequence:
            raise ValueError("payload_sequence must increase")
        self._scoped_payload_sequence = payload_sequence
        self._scoped_buffer.extend(pcm16le)

    async def seal_turn(
        self,
        identity: STTProviderTurnIdentity,
        *,
        sealed_content_ranges: tuple[AudioCaptureSpan, ...],
        seal_reason: str,
        observed_trailing_silence_ms: int | None,
    ) -> None:
        _ = sealed_content_ranges, seal_reason, observed_trailing_silence_ms
        self._require_scoped_identity(identity)
        if self._scoped_task is not None and not self._scoped_task.done():
            raise RuntimeError("Custom offline transcription is already running")
        utterance = bytes(self._scoped_buffer)
        self._scoped_buffer.clear()
        self._scoped_task = asyncio.create_task(
            self._run_scoped_transcription(identity, utterance),
            name="custom-offline-scoped-transcription",
        )

    async def _run_scoped_transcription(
        self,
        identity: STTProviderTurnIdentity,
        pcm16le: bytes,
    ) -> None:
        try:
            async with self._transcribe_lock:
                text = await asyncio.wait_for(
                    self._request_transcription(pcm16le),
                    timeout=_OFFLINE_TOTAL_TIMEOUT_S,
                )
        except asyncio.CancelledError:
            return
        except TimeoutError:
            self._terminalize_scoped(
                identity,
                outcome="failed",
                failure_reason="offline_total_timeout",
            )
            return
        except Exception as exc:
            sanitized = _sanitized_error(exc, secret=self.api_key)
            reason = (
                sanitized.category
                if isinstance(sanitized, CustomSTTRequestError)
                else type(sanitized).__name__
            )
            self._terminalize_scoped(
                identity,
                outcome="failed",
                failure_reason=reason,
            )
            return
        self._terminalize_scoped(
            identity,
            outcome="final" if text else "empty",
            text=text,
        )

    async def abort_turn(self, identity: STTProviderTurnIdentity, *, reason: str) -> None:
        self._require_scoped_identity(identity)
        task = self._scoped_task
        if task is not None and not task.done():
            task.cancel()
        self._terminalize_scoped(
            identity,
            outcome="cancelled",
            failure_reason=reason,
        )

    async def turn_events(self):
        async for event in self._scoped_events.events():
            yield event

    def _require_scoped_identity(self, identity: STTProviderTurnIdentity) -> None:
        if self._scoped_identity != identity:
            raise RuntimeError("unknown or retired Custom offline STT turn")

    def _terminalize_scoped(
        self,
        identity: STTProviderTurnIdentity,
        *,
        outcome: SegmentTerminalOutcome,
        text: str = "",
        failure_reason: str | None = None,
    ) -> None:
        if self._scoped_identity != identity:
            return
        retire = outcome in ("failed", "cancelled")
        self._scoped_events.put(
            STTProviderTurnTerminal(
                identity=identity,
                outcome=outcome,
                text=text,
                text_authority="authoritative" if outcome in ("final", "empty") else "none",
                failure_reason=failure_reason,
                epoch_disposition="retire" if retire else "reuse",
            )
        )
        self._scoped_identity = None
        self._scoped_epoch_retired = retire

    async def on_speech_end(
        self,
        *,
        trailing_silence_ms: int | None = None,
        reason: SpeechBoundaryReason | None = None,
    ) -> None:
        _ = trailing_silence_ms, reason
        if self._stopped:
            return
        utterance = bytes(self._buffer)
        self._buffer.clear()
        async with self._transcribe_lock:
            if self._stopped:
                return
            try:
                await self._transcribe(utterance)
            except Exception as exc:
                logger.warning(
                    "[STT] Custom offline utterance failed: %s",
                    _sanitized_error(exc, secret=self.api_key),
                )
                await self._events.put(STTBackendTranscriptEvent(text="", is_final=True))

    async def stop(self) -> None:
        await self.close()

    async def close(self) -> None:
        if self._stopped:
            return
        identity = self._scoped_identity
        if identity is not None:
            self._terminalize_scoped(
                identity,
                outcome="failed",
                failure_reason="session_closed",
            )
        self._stopped = True
        task = self._scoped_task
        self._scoped_task = None
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        async with self._transcribe_lock:
            client = self._client
            self._client = None
            if client is not None:
                with contextlib.suppress(Exception):
                    await client.aclose()
        await self._events.put(None)
        self._scoped_events.close()

    async def events(self) -> AsyncIterator[STTBackendTranscriptEvent]:
        while True:
            item = await self._events.get()
            if item is None:
                return
            if isinstance(item, BaseException):
                raise item
            yield item

    async def _transcribe(self, pcm16le: bytes) -> None:
        text = await self._request_transcription(pcm16le)
        await self._events.put(STTBackendTranscriptEvent(text=text, is_final=True))

    async def _request_transcription(self, pcm16le: bytes) -> str:
        client = self._client
        if client is None:
            raise RuntimeError("Custom STT offline session is not started")
        wav_bytes = pcm16le_to_wav(pcm16le, sample_rate_hz=self.sample_rate_hz)
        data: dict[str, str] = {}
        if self.model and "model" not in self.extra:
            data["model"] = self.model
        if compatibility_supports(
            CUSTOM_STT_COMPAT_OPENAI_TRANSCRIPTION,
            CUSTOM_STT_CAPABILITY_LANGUAGE_HINT,
        ):
            language = language_hint_for_source(self.source_language)
            if language and "language" not in self.extra:
                data["language"] = language
        for key, value in self.extra.items():
            data[key] = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        url = append_custom_stt_query(self._url, self.extra)
        try:
            response = await client.post(
                url,
                headers=authorization_headers(self.api_key),
                data=data,
                files={"file": ("speech.wav", wav_bytes, "audio/wav")},
            )
        except httpx.HTTPError as exc:
            raise CustomSTTRequestError(
                f"Custom STT endpoint unreachable ({sanitize_endpoint_for_display(url)})",
                category=CUSTOM_STT_VALIDATION_UNREACHABLE,
            ) from exc
        if response.status_code >= 400:
            excerpt = safe_body_excerpt(response.text)
            category = classify_http_failure(response.status_code, excerpt)
            raise CustomSTTRequestError(
                f"Custom STT transcription failed ({category})",
                category=category,
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise CustomSTTRequestError(
                "Custom STT compatibility mismatch",
                category=CUSTOM_STT_VALIDATION_COMPATIBILITY_MISMATCH,
            ) from exc
        text = extract_transcript_text(payload)
        if text is None:
            raise CustomSTTRequestError(
                "Custom STT compatibility mismatch",
                category=CUSTOM_STT_VALIDATION_COMPATIBILITY_MISMATCH,
            )
        normalized = text.strip()
        logger.info("[STT] Custom offline final text_len=%s", len(normalized))
        return normalized


@dataclass(slots=True)
class _StreamingOpenAIRealtimeSession(STTBackendSession):
    endpoint: str
    model: str
    api_key: str
    source_language: str
    sample_rate_hz: int
    websocket_connect: Callable[..., Any] | None = None
    extra: Mapping[str, object] = field(default_factory=dict)

    _events: asyncio.Queue[STTBackendTranscriptEvent | BaseException | None] = field(
        init=False, repr=False
    )
    _ws: Any = field(init=False, default=None, repr=False)
    _recv_task: asyncio.Task[None] | None = field(init=False, default=None, repr=False)
    _stopped: bool = field(init=False, default=False)
    _url: str = field(init=False, default="", repr=False)
    _pending_finals: int = field(init=False, default=0)
    _held_audio: bytearray = field(init=False, repr=False)
    _final_ready: asyncio.Event = field(init=False, repr=False)
    _send_lock: asyncio.Lock = field(init=False, repr=False)
    _scoped_events: STTProviderEventBuffer = field(
        init=False, default_factory=STTProviderEventBuffer, repr=False
    )
    _terminal_native_item_ids: set[str] = field(init=False, default_factory=set, repr=False)
    _terminal_native_event_ids: set[str] = field(init=False, default_factory=set, repr=False)
    _terminal_native_order: deque[tuple[str | None, str | None]] = field(
        init=False, default_factory=deque, repr=False
    )
    _scoped_identity: STTProviderTurnIdentity | None = field(init=False, default=None, repr=False)
    _scoped_payload_sequence: int = field(init=False, default=0, repr=False)
    _scoped_update_sequence: int = field(init=False, default=0, repr=False)
    _scoped_sealed: bool = field(init=False, default=False, repr=False)
    _scoped_epoch_retired: bool = field(init=False, default=False, repr=False)
    _scoped_final_timeout_task: asyncio.Task[None] | None = field(
        init=False, default=None, repr=False
    )

    def __post_init__(self) -> None:
        self._events = asyncio.Queue()
        self._held_audio = bytearray()
        self._final_ready = asyncio.Event()
        self._final_ready.set()
        self._send_lock = asyncio.Lock()
        self._url = append_custom_stt_query(resolve_openai_realtime_url(self.endpoint), self.extra)
        if self.model and "model" not in self.extra:
            self._url = append_custom_stt_query(self._url, {"model": self.model})

    async def start(self) -> None:
        connect = self.websocket_connect
        if connect is None:
            import websockets

            connect = websockets.connect
        headers = {
            **authorization_headers(self.api_key),
            "OpenAI-Beta": "realtime=v1",
        }
        try:
            self._ws = await asyncio.wait_for(
                connect(
                    self._url,
                    additional_headers=headers,
                    open_timeout=_STREAM_CONNECT_TIMEOUT_S,
                    ping_interval=None,
                ),
                timeout=_STREAM_CONNECT_TIMEOUT_S,
            )
        except TypeError:
            self._ws = await asyncio.wait_for(
                connect(
                    self._url,
                    extra_headers=headers,
                    open_timeout=_STREAM_CONNECT_TIMEOUT_S,
                    ping_interval=None,
                ),
                timeout=_STREAM_CONNECT_TIMEOUT_S,
            )
        except Exception as exc:
            raise CustomSTTRequestError(
                f"Custom STT endpoint unreachable ({sanitize_endpoint_for_display(self._url)})",
                category=CUSTOM_STT_VALIDATION_UNREACHABLE,
            ) from exc
        try:
            await self._send_json(self._session_update_payload())
        except Exception as exc:
            with contextlib.suppress(Exception):
                await self._ws.close()
            self._ws = None
            raise CustomSTTRequestError(
                f"Custom STT endpoint unreachable ({sanitize_endpoint_for_display(self._url)})",
                category=CUSTOM_STT_VALIDATION_UNREACHABLE,
            ) from exc
        self._recv_task = asyncio.create_task(self._receive_loop())
        logger.info(
            "[STT] Custom realtime session ready endpoint=%s",
            sanitize_endpoint_for_display(self._url),
        )

    async def send_audio(self, pcm16le: bytes) -> None:
        if self._stopped or not pcm16le:
            return
        if not self._final_ready.is_set():
            self._held_audio.extend(pcm16le)
            return
        with contextlib.suppress(Exception):
            await self._append_audio(pcm16le)

    async def begin_turn(self, request: STTProviderTurnRequest) -> None:
        if self.extra.get("turn_detection") is not None:
            raise CustomSTTConfigurationError("Custom realtime LISTEN requires turn_detection=null")
        if self._stopped or self._scoped_epoch_retired:
            raise RuntimeError("Custom realtime STT session is unavailable")
        if self._scoped_identity is not None:
            raise RuntimeError("Custom realtime STT session already has an unresolved turn")
        if not self._final_ready.is_set() or self._pending_finals:
            raise RuntimeError("Custom realtime commit barrier is unresolved")
        self._scoped_identity = request.identity
        self._scoped_payload_sequence = 0
        self._scoped_sealed = False
        self._scoped_update_sequence = 0
        self._held_audio.clear()

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
        self._require_scoped_identity(identity)
        if self._scoped_sealed:
            raise RuntimeError("Custom realtime STT turn is already sealed")
        if payload_sequence <= self._scoped_payload_sequence:
            raise ValueError("payload_sequence must increase")
        self._scoped_payload_sequence = payload_sequence
        if not pcm16le:
            return
        try:
            async with self._send_lock:
                await self._append_audio(pcm16le)
        except Exception:
            self._terminalize_scoped(
                identity,
                outcome="failed",
                failure_reason="audio_write_failed",
                retire=True,
                provenance=STTNativeProvenance(barrier="audio_write"),
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
        self._require_scoped_identity(identity)
        if self._scoped_sealed:
            raise RuntimeError("Custom realtime STT turn is already sealed")
        self._scoped_sealed = True
        self._final_ready.clear()
        self._pending_finals = 1
        try:
            async with self._send_lock:
                await self._send_json({"type": "input_audio_buffer.commit"})
        except Exception:
            self._pending_finals = 0
            self._final_ready.set()
            self._terminalize_scoped(
                identity,
                outcome="failed",
                failure_reason="commit_write_failed",
                retire=True,
                provenance=STTNativeProvenance(barrier="commit_write"),
            )
            return
        self._scoped_final_timeout_task = asyncio.create_task(
            self._wait_scoped_final(identity),
            name="custom-realtime-scoped-final-timeout",
        )

    async def _wait_scoped_final(self, identity: STTProviderTurnIdentity) -> None:
        try:
            await asyncio.sleep(_STREAM_FINAL_TIMEOUT_S)
        except asyncio.CancelledError:
            return
        if self._scoped_identity != identity:
            return
        self._pending_finals = 0
        self._final_ready.set()
        self._terminalize_scoped(
            identity,
            outcome="failed",
            failure_reason="final_timeout",
            retire=True,
            provenance=STTNativeProvenance(barrier="commit_timeout"),
        )
        await self.close()

    async def abort_turn(self, identity: STTProviderTurnIdentity, *, reason: str) -> None:
        self._require_scoped_identity(identity)
        self._terminalize_scoped(
            identity,
            outcome="cancelled",
            failure_reason=reason,
            retire=True,
            provenance=STTNativeProvenance(barrier="abort"),
        )
        await self.close()

    async def turn_events(self):
        async for event in self._scoped_events.events():
            yield event

    def _require_scoped_identity(self, identity: STTProviderTurnIdentity) -> None:
        if self._scoped_identity != identity:
            raise RuntimeError("unknown or retired Custom realtime STT turn")

    def _terminalize_scoped(
        self,
        identity: STTProviderTurnIdentity,
        *,
        outcome: SegmentTerminalOutcome,
        text: str = "",
        failure_reason: str | None = None,
        retire: bool = False,
        provenance: STTNativeProvenance,
    ) -> None:
        if self._scoped_identity != identity:
            return
        timeout_task = self._scoped_final_timeout_task
        self._scoped_final_timeout_task = None
        if timeout_task is not None and timeout_task is not asyncio.current_task():
            timeout_task.cancel()
        self._scoped_events.put(
            STTProviderTurnTerminal(
                identity=identity,
                outcome=outcome,
                text=text,
                text_authority="authoritative" if outcome in ("final", "empty") else "none",
                failure_reason=failure_reason,
                epoch_disposition="retire" if retire else "reuse",
                provenance=(provenance,),
            )
        )
        self._scoped_identity = None
        self._scoped_sealed = False
        self._scoped_epoch_retired = retire

    async def on_speech_end(
        self,
        *,
        trailing_silence_ms: int | None = None,
        reason: SpeechBoundaryReason | None = None,
    ) -> None:
        _ = trailing_silence_ms, reason
        if self._stopped:
            return
        await self._wait_for_previous_final()
        if self._stopped:
            return
        if self._held_audio:
            held = bytes(self._held_audio)
            self._held_audio.clear()
            with contextlib.suppress(Exception):
                await self._append_audio(held)
            if self._stopped:
                return
        self._final_ready.clear()
        self._pending_finals += 1
        try:
            await self._send_json({"type": "input_audio_buffer.commit"})
        except Exception:
            if self._pending_finals > 0:
                self._pending_finals -= 1
            self._final_ready.set()
            return
        logger.info("[STT] Custom realtime finalize sent")

    async def stop(self) -> None:
        await self.close()

    async def close(self) -> None:
        identity = self._scoped_identity
        if identity is not None:
            self._terminalize_scoped(
                identity,
                outcome="failed",
                failure_reason="session_closed",
                retire=True,
                provenance=STTNativeProvenance(barrier="close"),
            )
        if self._stopped:
            return
        self._stopped = True
        recv_task = self._recv_task
        self._recv_task = None
        if recv_task is not None:
            recv_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await recv_task
        ws = self._ws
        self._ws = None
        if ws is not None:
            with contextlib.suppress(Exception):
                await ws.close()
        self._scoped_events.close()
        await self._events.put(None)

    async def events(self) -> AsyncIterator[STTBackendTranscriptEvent]:
        while True:
            item = await self._events.get()
            if item is None:
                return
            if isinstance(item, BaseException):
                raise item
            yield item

    @staticmethod
    def _is_fatal_error_message(lowered: str) -> bool:
        if not lowered:
            return False
        fatal_tokens = (
            "unauthorized",
            "api key",
            "invalid api",
            "authentication",
            "model not found",
            "model_not_found",
            "unknown model",
            "insufficient",
            "forbidden",
        )
        return any(token in lowered for token in fatal_tokens)

    def _session_update_payload(self) -> dict[str, Any]:
        transcription: dict[str, Any] = {}
        if self.model:
            transcription["model"] = self.model
        if compatibility_supports(
            CUSTOM_STT_COMPAT_OPENAI_REALTIME,
            CUSTOM_STT_CAPABILITY_LANGUAGE_HINT,
        ):
            language = language_hint_for_source(self.source_language)
            if language:
                transcription["language"] = language
        session: dict[str, Any] = {
            "input_audio_transcription": transcription,
            "turn_detection": self.extra.get("turn_detection"),
        }
        return {
            "type": "session.update",
            "session": session,
        }

    async def _append_audio(self, pcm16le: bytes) -> None:
        await self._send_json(
            {
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(pcm16le).decode("ascii"),
            }
        )

    async def _wait_for_previous_final(self) -> None:
        if self._final_ready.is_set():
            return
        try:
            await asyncio.wait_for(self._final_ready.wait(), timeout=_STREAM_FINAL_TIMEOUT_S)
        except TimeoutError:
            if self._pending_finals > 0:
                self._pending_finals -= 1
            self._final_ready.set()
            await self._events.put(STTBackendTranscriptEvent(text="", is_final=True))

    async def _send_json(self, payload: dict[str, Any]) -> None:
        ws = self._ws
        if ws is None:
            raise RuntimeError("Custom STT realtime session is not started")
        try:
            await asyncio.wait_for(ws.send(json.dumps(payload)), timeout=5.0)
        except Exception as exc:
            await self._events.put(_sanitized_error(exc, secret=self.api_key))
            raise

    async def _receive_loop(self) -> None:
        ws = self._ws
        if ws is None:
            return
        try:
            async for raw in ws:
                event = parse_realtime_event(raw)
                if event is None:
                    continue
                event_type = str(event.get("type") or "")
                if event_type in _PARTIAL_EVENT_TYPES or event_type.endswith(".delta"):
                    identity = self._scoped_identity
                    if identity is not None:
                        delta = str(event.get("delta") or event.get("text") or "").strip()
                        if delta:
                            self._scoped_update_sequence += 1
                            self._scoped_events.put(
                                STTProviderTurnUpdate(
                                    identity=identity,
                                    sequence=self._scoped_update_sequence,
                                    stability="provisional",
                                    assembly="append",
                                    text=delta,
                                    provenance=STTNativeProvenance(
                                        native_event_id=str(event.get("event_id") or "").strip()
                                        or None,
                                        native_item_id=str(event.get("item_id") or "").strip()
                                        or None,
                                        barrier=event_type,
                                    ),
                                )
                            )
                    continue
                if event_type in {"session.created", "session.updated"}:
                    continue
                if event_type == "error" or event.get("error"):
                    error = event.get("error")
                    message = ""
                    if isinstance(error, dict):
                        message = str(error.get("message") or error.get("code") or "")
                    lowered = message.lower()
                    if self._is_fatal_error_message(lowered):
                        category = CUSTOM_STT_VALIDATION_COMPATIBILITY_MISMATCH
                        if "auth" in lowered or "unauthorized" in lowered or "api key" in lowered:
                            category = CUSTOM_STT_VALIDATION_AUTH_FAILURE
                        elif "model" in lowered:
                            category = CUSTOM_STT_VALIDATION_MODEL_UNAVAILABLE
                        raise CustomSTTRequestError(
                            "Custom STT realtime session failed",
                            category=category,
                        )
                    if self._scoped_identity is not None:
                        raise CustomSTTRequestError(
                            "Custom STT realtime provider error",
                            category="provider_error",
                        )
                    continue
                if event_type not in _FINAL_EVENT_TYPES:
                    continue
                native_item_id = str(event.get("item_id") or "").strip() or None
                item = event.get("item")
                if native_item_id is None and isinstance(item, Mapping):
                    native_item_id = str(item.get("id") or "").strip() or None
                native_event_id = str(event.get("event_id") or "").strip() or None
                if (
                    native_item_id is not None and native_item_id in self._terminal_native_item_ids
                ) or (
                    native_event_id is not None
                    and native_event_id in self._terminal_native_event_ids
                ):
                    continue
                if self._pending_finals <= 0:
                    continue
                text = extract_transcript_text(event) or ""
                normalized_text = text.strip()
                identity = self._scoped_identity
                if identity is not None and self._scoped_sealed:
                    self._remember_native_terminal(native_item_id, native_event_id)
                    failed = event_type.endswith(".failed")
                    self._terminalize_scoped(
                        identity,
                        outcome="failed" if failed else ("final" if normalized_text else "empty"),
                        text="" if failed else normalized_text,
                        failure_reason="native_transcription_failed" if failed else None,
                        retire=failed,
                        provenance=STTNativeProvenance(
                            native_event_id=native_event_id,
                            native_item_id=native_item_id,
                            barrier=event_type,
                        ),
                    )
                self._pending_finals -= 1
                self._final_ready.set()
                logger.info("[STT] Custom realtime final text_len=%s", len(normalized_text))
                await self._events.put(
                    STTBackendTranscriptEvent(text=normalized_text, is_final=True)
                )

        except asyncio.CancelledError:
            return
        except Exception as exc:
            await self._events.put(_sanitized_error(exc, secret=self.api_key))
            identity = self._scoped_identity
            if identity is not None:
                sanitized = _sanitized_error(exc, secret=self.api_key)
                reason = (
                    sanitized.category
                    if isinstance(sanitized, CustomSTTRequestError)
                    else type(sanitized).__name__
                )
                self._terminalize_scoped(
                    identity,
                    outcome="failed",
                    failure_reason=reason,
                    retire=True,
                    provenance=STTNativeProvenance(barrier="receive_error"),
                )
                self._scoped_events.put(
                    STTProviderEpochEnded(
                        provider_epoch_id=identity.provider_epoch_id,
                        orderly=False,
                        reason=reason,
                        provider_turn_id=identity.provider_turn_id,
                    )
                )
        else:
            await self._events.put(
                CustomSTTRequestError(
                    "Custom STT realtime session ended",
                    category=CUSTOM_STT_VALIDATION_UNREACHABLE,
                )
            )
            identity = self._scoped_identity
            if identity is not None:
                self._terminalize_scoped(
                    identity,
                    outcome="failed",
                    failure_reason="connection_eof",
                    retire=True,
                    provenance=STTNativeProvenance(barrier="connection_eof"),
                )
                self._scoped_events.put(
                    STTProviderEpochEnded(
                        provider_epoch_id=identity.provider_epoch_id,
                        orderly=False,
                        reason="connection_eof",
                        provider_turn_id=identity.provider_turn_id,
                    )
                )

    def _remember_native_terminal(
        self,
        native_item_id: str | None,
        native_event_id: str | None,
    ) -> None:
        if len(self._terminal_native_order) >= 4096:
            old_item_id, old_event_id = self._terminal_native_order.popleft()
            if old_item_id is not None:
                self._terminal_native_item_ids.discard(old_item_id)
            if old_event_id is not None:
                self._terminal_native_event_ids.discard(old_event_id)
        self._terminal_native_order.append((native_item_id, native_event_id))
        if native_item_id is not None:
            self._terminal_native_item_ids.add(native_item_id)
        if native_event_id is not None:
            self._terminal_native_event_ids.add(native_event_id)


def _sanitized_error(exc: BaseException, *, secret: str) -> Exception:
    if isinstance(exc, CustomSTTRequestError):
        return CustomSTTRequestError(
            sanitize_custom_stt_text(str(exc), secret=secret),
            category=exc.category,
        )
    return RuntimeError(sanitize_custom_stt_text(str(exc), secret=secret))
