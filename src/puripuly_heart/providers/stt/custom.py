from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import httpx

from puripuly_heart.core.speech_boundary import SpeechBoundaryReason
from puripuly_heart.core.stt.backend import (
    STTBackend,
    STTBackendSession,
    STTBackendTranscriptEvent,
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

logger = logging.getLogger(__name__)

_OFFLINE_TIMEOUT = httpx.Timeout(connect=5.0, read=30.0, write=10.0, pool=5.0)
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

    def __post_init__(self) -> None:
        self._events = asyncio.Queue()
        self._buffer = bytearray()
        self._transcribe_lock = asyncio.Lock()
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
        self._stopped = True
        async with self._transcribe_lock:
            client = self._client
            self._client = None
            if client is not None:
                with contextlib.suppress(Exception):
                    await client.aclose()
        await self._events.put(None)

    async def events(self) -> AsyncIterator[STTBackendTranscriptEvent]:
        while True:
            item = await self._events.get()
            if item is None:
                return
            if isinstance(item, BaseException):
                raise item
            yield item

    async def _transcribe(self, pcm16le: bytes) -> None:
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
        logger.info("[STT] Custom offline final text_len=%s", len(text.strip()))
        await self._events.put(STTBackendTranscriptEvent(text=text.strip(), is_final=True))


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

    def __post_init__(self) -> None:
        self._events = asyncio.Queue()
        self._held_audio = bytearray()
        self._final_ready = asyncio.Event()
        self._final_ready.set()
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
        }
        if "turn_detection" in self.extra:
            session["turn_detection"] = self.extra["turn_detection"]
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
            await ws.send(json.dumps(payload))
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
                    continue
                if event_type not in _FINAL_EVENT_TYPES:
                    continue
                if self._pending_finals <= 0:
                    continue
                text = extract_transcript_text(event) or ""
                self._pending_finals -= 1
                self._final_ready.set()
                logger.info("[STT] Custom realtime final text_len=%s", len(text.strip()))
                await self._events.put(STTBackendTranscriptEvent(text=text.strip(), is_final=True))
        except asyncio.CancelledError:
            return
        except Exception as exc:
            await self._events.put(_sanitized_error(exc, secret=self.api_key))
        else:
            await self._events.put(
                CustomSTTRequestError(
                    "Custom STT realtime session ended",
                    category=CUSTOM_STT_VALIDATION_UNREACHABLE,
                )
            )


def _sanitized_error(exc: BaseException, *, secret: str) -> Exception:
    if isinstance(exc, CustomSTTRequestError):
        return CustomSTTRequestError(
            sanitize_custom_stt_text(str(exc), secret=secret),
            category=exc.category,
        )
    return RuntimeError(sanitize_custom_stt_text(str(exc), secret=secret))
