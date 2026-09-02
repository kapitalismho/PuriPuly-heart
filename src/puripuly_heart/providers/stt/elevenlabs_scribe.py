"""ElevenLabs Scribe v2 Realtime STT Backend using the official elevenlabs SDK."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Sequence

from puripuly_heart.core.speech_boundary import SpeechBoundaryReason, boundary_wait_ms
from puripuly_heart.core.stt.backend import (
    STTBackend,
    STTBackendSession,
    STTBackendTranscriptEvent,
)

logger = logging.getLogger(__name__)

ELEVENLABS_SCRIBE_STT_MODEL = "scribe_v2_realtime"
ELEVENLABS_SCRIBE_SAMPLE_RATE_HZ = 16000
MAX_SCRIBE_KEYTERMS = 50
MAX_SCRIBE_KEYTERM_CHARS = 20

_CLOSED = object()


def scribe_keyterms(terms: Sequence[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    for term in terms:
        candidate = str(term).strip()[:MAX_SCRIBE_KEYTERM_CHARS]
        if not candidate or candidate in normalized:
            continue
        normalized.append(candidate)
        if len(normalized) >= MAX_SCRIBE_KEYTERMS:
            break
    return tuple(normalized)


def scribe_language_code(source_language: str | None) -> str | None:
    if not source_language:
        return None
    normalized = source_language.strip()
    if not normalized or normalized.lower() == "auto":
        return None
    base = normalized.split("-")[0].lower()
    return base or None


@dataclass(slots=True)
class ElevenLabsScribeSTTBackend(STTBackend):
    """ElevenLabs Scribe v2 Realtime STT Backend using the official SDK."""

    api_key: str
    language_code: str | None = None
    keyterms: Sequence[str] = ()
    model: str = ELEVENLABS_SCRIBE_STT_MODEL
    sample_rate_hz: int = ELEVENLABS_SCRIBE_SAMPLE_RATE_HZ
    connect_timeout_s: float = 10.0
    scribe_connect_factory: Callable[[Any], Any] | None = None

    async def open_session(self) -> STTBackendSession:
        if self.sample_rate_hz != ELEVENLABS_SCRIBE_SAMPLE_RATE_HZ:
            raise ValueError("sample_rate_hz must be 16000 for Scribe realtime transcription")
        if not self.api_key:
            raise ValueError("api_key must be non-empty")
        if self.connect_timeout_s <= 0:
            raise ValueError("connect_timeout_s must be > 0")

        session = _ElevenLabsScribeSession(
            api_key=self.api_key,
            language_code=scribe_language_code(self.language_code),
            keyterms=scribe_keyterms(self.keyterms),
            model=self.model,
            sample_rate_hz=self.sample_rate_hz,
            connect_timeout_s=self.connect_timeout_s,
            scribe_connect_factory=self.scribe_connect_factory,
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
            import urllib.error
            import urllib.request

            req = urllib.request.Request(
                "https://api.elevenlabs.io/v1/user/subscription",
                headers={"xi-api-key": api_key},
            )
            try:
                with urllib.request.urlopen(req, timeout=5) as response:
                    return response.status == 200
            except urllib.error.HTTPError as e:
                raise Exception(f"HTTP {e.code}: {e.reason}")
            except Exception as e:
                raise Exception(f"Connection failed: {e}")

        return await asyncio.to_thread(_check)


@dataclass(slots=True)
class _ElevenLabsScribeSession(STTBackendSession):
    """Internal session wrapping an elevenlabs RealtimeConnection."""

    api_key: str
    language_code: str | None
    keyterms: tuple[str, ...]
    model: str
    sample_rate_hz: int
    connect_timeout_s: float
    scribe_connect_factory: Callable[[Any], Any] | None = None

    _events: asyncio.Queue[STTBackendTranscriptEvent | BaseException | None] = field(
        init=False, repr=False
    )
    _connection: Any = field(init=False, default=None, repr=False)
    _queue_task: asyncio.Task[None] | None = field(init=False, default=None, repr=False)
    _stopped: bool = field(init=False, default=False)
    _connection_events: asyncio.Queue[Any] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._events = asyncio.Queue()
        self._connection_events = asyncio.Queue()

    async def start(self) -> None:
        from elevenlabs.realtime import (
            AudioFormat,
            CommitStrategy,
            RealtimeAudioOptions,
            RealtimeEvents,
            ScribeRealtime,
        )

        options = RealtimeAudioOptions(
            model_id=self.model,
            audio_format=AudioFormat.PCM_16000,
            sample_rate=self.sample_rate_hz,
            commit_strategy=CommitStrategy.MANUAL,
            language_code=self.language_code,
            keyterms=list(self.keyterms) if self.keyterms else None,
        )
        logger.info("[STT] Scribe realtime connecting (timeout=%.1fs)", self.connect_timeout_s)
        start_at = time.monotonic()
        if self.scribe_connect_factory is not None:
            connection = await asyncio.wait_for(
                self.scribe_connect_factory(options), timeout=self.connect_timeout_s
            )
        else:
            scribe = ScribeRealtime(api_key=self.api_key)
            connection = await asyncio.wait_for(
                scribe.connect(options), timeout=self.connect_timeout_s
            )
        self._connection = connection
        elapsed = time.monotonic() - start_at
        logger.info("[STT] Scribe realtime connected in %.2fs", elapsed)

        self._connection.on(RealtimeEvents.PARTIAL_TRANSCRIPT, self._on_partial)
        self._connection.on(RealtimeEvents.FINAL_TRANSCRIPT, self._on_partial)
        self._connection.on(RealtimeEvents.FINAL_TRANSCRIPT_WITH_TIMESTAMPS, self._on_partial)
        self._connection.on(RealtimeEvents.COMMITTED_TRANSCRIPT, self._on_committed)
        self._connection.on(RealtimeEvents.QUOTA_EXCEEDED, self._on_error_event)
        self._connection.on(RealtimeEvents.AUTH_ERROR, self._on_error_event)
        self._connection.on(RealtimeEvents.RATE_LIMITED, self._on_error_event)
        self._connection.on(RealtimeEvents.ERROR, self._on_error_event)
        self._connection.on(RealtimeEvents.TRANSCRIBER_ERROR, self._on_error_event)
        self._connection.on(RealtimeEvents.INVALID_REQUEST, self._on_error_event)
        self._connection.on(RealtimeEvents.QUEUE_OVERFLOW, self._on_error_event)
        self._connection.on(RealtimeEvents.RESOURCE_EXHAUSTED, self._on_error_event)
        self._connection.on(RealtimeEvents.SESSION_TIME_LIMIT_EXCEEDED, self._on_error_event)
        self._connection.on(RealtimeEvents.INPUT_ERROR, self._on_error_event)
        self._connection.on(RealtimeEvents.CHUNK_SIZE_EXCEEDED, self._on_error_event)
        self._connection.on(RealtimeEvents.INSUFFICIENT_AUDIO_ACTIVITY, self._on_error_event)
        self._connection.on(RealtimeEvents.CLOSE, self._on_closed)

        self._queue_task = asyncio.create_task(self._drain_connection_events())

    @staticmethod
    def _event_name(data: Any) -> str:
        if isinstance(data, dict):
            return str(data.get("message_type") or data.get("type") or "")
        return str(getattr(data, "type", "") or "")

    @staticmethod
    def _event_text(data: Any) -> str:
        if isinstance(data, dict):
            return str(data.get("text") or "")
        return str(getattr(data, "text", "") or "")

    def _on_partial(self, data: Any) -> None:
        logger.debug(
            "[STT] Scribe %s non-authoritative text_len=%s",
            self._event_name(data),
            len(self._event_text(data)),
        )

    def _on_committed(self, data: Any) -> None:
        text = self._event_text(data)
        logger.info("[STT] Transcript final text_len=%s", len(text))
        self._connection_events.put_nowait(STTBackendTranscriptEvent(text=text, is_final=True))

    def _on_error_event(self, data: Any) -> None:
        event_name = self._event_name(data)
        logger.warning("[STT] Scribe provider event %s", event_name)
        self._connection_events.put_nowait(RuntimeError(f"Scribe realtime error: {event_name}"))

    def _on_closed(self, data: Any) -> None:
        _ = data
        logger.debug("[STT] Scribe connection closed")
        self._connection_events.put_nowait(_CLOSED)

    async def _drain_connection_events(self) -> None:
        try:
            while True:
                item = await self._connection_events.get()
                if item is _CLOSED:
                    return
                self._put_event(item)
                if isinstance(item, BaseException):
                    self._stopped = True
                    return
        except asyncio.CancelledError:
            raise
        finally:
            self._put_event(None)

    async def send_audio(self, pcm16le: bytes) -> None:
        if self._stopped or self._connection is None:
            return
        await self._connection.send({"audio_base_64": base64.b64encode(pcm16le).decode("ascii")})

    async def on_speech_end(
        self,
        *,
        trailing_silence_ms: int | None = None,
        reason: SpeechBoundaryReason | None = None,
    ) -> None:
        if self._stopped:
            return
        observed_tail_ms = max(int(trailing_silence_ms or 0), 0)
        wait_ms = boundary_wait_ms(reason, observed_tail_ms=observed_tail_ms)
        logger.info(
            "[STT][Tail] provider=scribe boundary_reason=%s observed_tail_ms=%s "
            "boundary_wait_ms=%s",
            reason,
            observed_tail_ms,
            wait_ms,
        )
        if self._connection is None:
            return
        await self._connection.commit()
        logger.info("[STT] Scribe commit sent (finalize)")

    async def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        self._connection_events.put_nowait(_CLOSED)

    async def close(self) -> None:
        await self.stop()
        if self._queue_task is not None:
            self._queue_task.cancel()
            await asyncio.gather(self._queue_task, return_exceptions=True)
            self._queue_task = None
        self._put_event(None)
        if self._connection is not None:
            with contextlib.suppress(Exception):
                result = self._connection.close()
                if asyncio.iscoroutine(result):
                    await result
            self._connection = None

    async def events(self) -> AsyncIterator[STTBackendTranscriptEvent]:
        while True:
            item = await self._events.get()
            if item is None:
                return
            if isinstance(item, BaseException):
                raise item
            yield item

    def _put_event(self, event: STTBackendTranscriptEvent | BaseException | None) -> None:
        self._events.put_nowait(event)
