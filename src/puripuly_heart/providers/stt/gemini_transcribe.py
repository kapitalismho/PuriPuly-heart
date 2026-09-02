"""Gemini 3.5 Transcribe Live STT Backend using the official google-genai SDK."""

from __future__ import annotations

import asyncio
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

GEMINI_TRANSCRIBE_STT_MODEL = "gemini-3.5-transcribe-live"
GEMINI_TRANSCRIBE_SAMPLE_RATE_HZ = 16000


def gemini_transcribe_language_codes(source_language: str | None) -> list[str]:
    if not source_language or source_language == "auto":
        return []
    normalized = source_language.strip()
    if not normalized:
        return []
    return [normalized]


@dataclass(slots=True)
class GeminiTranscribeSTTBackend(STTBackend):
    """Gemini 3.5 Transcribe Live STT Backend using the official google-genai SDK."""

    api_key: str
    language_codes: Sequence[str] = ()
    custom_vocabulary: Sequence[str] = ()
    model: str = GEMINI_TRANSCRIBE_STT_MODEL
    sample_rate_hz: int = GEMINI_TRANSCRIBE_SAMPLE_RATE_HZ
    connect_timeout_s: float = 10.0
    live_connect_factory: Callable[[str, Any], Any] | None = None

    async def open_session(self) -> STTBackendSession:
        if self.sample_rate_hz != GEMINI_TRANSCRIBE_SAMPLE_RATE_HZ:
            raise ValueError(
                "sample_rate_hz must be 16000 for Gemini Transcribe Live transcription"
            )
        if not self.api_key:
            raise ValueError("api_key must be non-empty")
        if self.connect_timeout_s <= 0:
            raise ValueError("connect_timeout_s must be > 0")

        session = _GeminiTranscribeLiveSession(
            api_key=self.api_key,
            language_codes=list(self.language_codes),
            custom_vocabulary=list(self.custom_vocabulary),
            model=self.model,
            sample_rate_hz=self.sample_rate_hz,
            connect_timeout_s=self.connect_timeout_s,
            live_connect_factory=self.live_connect_factory,
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
                "https://generativelanguage.googleapis.com/v1beta/models",
                headers={"x-goog-api-key": api_key},
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
class _GeminiTranscribeLiveSession(STTBackendSession):
    """Internal session wrapping a google-genai Live API AsyncSession."""

    api_key: str
    language_codes: list[str]
    custom_vocabulary: list[str]
    model: str
    sample_rate_hz: int
    connect_timeout_s: float
    live_connect_factory: Callable[[str, Any], Any] | None = None

    _events: asyncio.Queue[STTBackendTranscriptEvent | BaseException | None] = field(
        init=False, repr=False
    )
    _live_session: Any = field(init=False, default=None, repr=False)
    _recv_task: asyncio.Task[None] | None = field(init=False, default=None, repr=False)
    _stopped: bool = field(init=False, default=False)
    _activity_open: bool = field(init=False, default=False)
    _pending_finalize_requests: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self._events = asyncio.Queue()

    async def start(self) -> None:
        from google.genai import types

        transcription_config_kwargs: dict[str, Any] = {"mode": "VERBATIM"}
        if self.language_codes:
            transcription_config_kwargs["language_codes"] = list(self.language_codes)
        if self.custom_vocabulary:
            transcription_config_kwargs["custom_vocabulary"] = list(self.custom_vocabulary)

        config = types.LiveConnectConfig(
            response_modalities=["TEXT"],
            input_audio_transcription=types.AudioTranscriptionConfig(**transcription_config_kwargs),
            realtime_input_config=types.RealtimeInputConfig(
                automatic_activity_detection=types.AutomaticActivityDetection(disabled=True),
            ),
        )

        factory = self.live_connect_factory
        if factory is None:
            from google import genai

            client = genai.Client(api_key=self.api_key)
            factory = client.aio.live.connect
        logger.info(
            "[STT] Gemini Transcribe Live connecting (timeout=%.1fs)", self.connect_timeout_s
        )
        start_at = time.monotonic()
        live_context = factory(model=self.model, config=config)
        self._live_session = await asyncio.wait_for(
            live_context.__aenter__(), timeout=self.connect_timeout_s
        )
        elapsed = time.monotonic() - start_at
        logger.info("[STT] Gemini Transcribe Live connected in %.2fs", elapsed)

        self._recv_task = asyncio.create_task(self._recv_loop())

    async def _recv_loop(self) -> None:
        try:
            live_session = self._live_session
            if live_session is None:
                return
            async for message in live_session.receive():
                self._handle_message(message)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("Gemini Transcribe Live recv loop error")
            self._put_event(exc)
        finally:
            self._put_event(None)

    def _handle_message(self, message: Any) -> None:
        content = message.server_content
        if content is None:
            return
        interim = content.interim_input_transcription
        if interim is not None and interim.text:
            logger.debug("[STT] Gemini Transcribe Live interim text_len=%s", len(interim.text))
        final = content.input_transcription
        if final is None:
            return
        text = str(final.text or "")
        if self._pending_finalize_requests <= 0:
            logger.debug(
                "[STT] Gemini Transcribe Live final ignored without pending finalize text_len=%s",
                len(text),
            )
            return
        self._pending_finalize_requests -= 1
        logger.info("[STT] Transcript final text_len=%s", len(text))
        self._put_event(STTBackendTranscriptEvent(text=text, is_final=True))

    async def send_audio(self, pcm16le: bytes) -> None:
        if self._stopped:
            return
        if not self._activity_open:
            await self._send_realtime(activity_start={})
            self._activity_open = True
            logger.info("[STT] Gemini Transcribe Live activityStart sent")
        await self._send_realtime(
            audio={"data": pcm16le, "mime_type": f"audio/pcm;rate={self.sample_rate_hz}"},
        )

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
            "[STT][Tail] provider=gemini_transcribe boundary_reason=%s observed_tail_ms=%s "
            "boundary_wait_ms=%s",
            reason,
            observed_tail_ms,
            wait_ms,
        )
        if self._activity_open:
            self._activity_open = False
            self._pending_finalize_requests += 1
            await self._send_realtime(activity_end={})
            logger.info("[STT] Gemini Transcribe Live activityEnd sent (finalize)")

    async def _send_realtime(self, **kwargs: Any) -> None:
        live_session = self._live_session
        if live_session is None:
            return
        await live_session.send_realtime_input(**kwargs)

    async def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        if self._pending_finalize_requests > 0:
            logger.warning(
                "[STT] Gemini Transcribe Live session closed with unresolved finalize requests count=%s",
                self._pending_finalize_requests,
            )
            self._pending_finalize_requests = 0
        self._put_event(None)

    async def close(self) -> None:
        await self.stop()
        if self._recv_task is not None:
            self._recv_task.cancel()
            await asyncio.gather(self._recv_task, return_exceptions=True)
            self._recv_task = None
        live_session = self._live_session
        self._live_session = None
        if live_session is not None:
            with contextlib.suppress(Exception):
                await live_session.close()

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
