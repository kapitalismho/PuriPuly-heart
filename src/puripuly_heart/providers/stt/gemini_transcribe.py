"""Gemini 3.5 Transcribe Live STT Backend using the official google-genai SDK."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Sequence

from puripuly_heart.core.speech_boundary import SpeechBoundaryReason, boundary_wait_ms
from puripuly_heart.core.stt.backend import (
    RecoverableSTTSessionError,
    STTBackend,
    STTBackendSession,
    STTBackendTranscriptEvent,
)

logger = logging.getLogger(__name__)

GEMINI_TRANSCRIBE_STT_MODEL = "gemini-3.5-transcribe-live"
GEMINI_TRANSCRIBE_SAMPLE_RATE_HZ = 16000
GEMINI_TRANSCRIBE_FINALIZE_TIMEOUT_S = 2.0


class GeminiTranscribeFinalizeTimeout(RecoverableSTTSessionError):
    pass


@dataclass(slots=True, eq=False)
class _PendingTurn:
    latest_interim: str = ""
    final_emitted: bool = False
    activity_end_ack: asyncio.Event = field(default_factory=asyncio.Event)
    timeout_task: asyncio.Task[None] | None = None


@dataclass(frozen=True, slots=True)
class _StartTurn:
    turn: _PendingTurn


@dataclass(frozen=True, slots=True)
class _EndTurn:
    turn: _PendingTurn


_STOP = object()


def gemini_transcribe_language_codes(source_language: str | None) -> list[str]:
    if not source_language:
        return []
    from puripuly_heart.core.language import gemini_transcribe_language_hint

    mapped = gemini_transcribe_language_hint(source_language)
    return [mapped] if mapped else []


def _recv_failure_fields(exc: BaseException) -> tuple[str, object, object, str]:
    exception_class = type(exc).__name__
    api_code = getattr(exc, "code", None)
    api_status = getattr(exc, "status", None)
    return exception_class, api_code, api_status, _recv_message_kind(exc, api_code, api_status)


def _recv_message_kind(exc: BaseException, api_code: object, api_status: object) -> str:
    class_name = type(exc).__name__.lower().replace("_", "")
    status_text = str(api_status or "").lower()
    if "goaway" in class_name or "go_away" in status_text:
        return "go_away"
    if (
        "connection" in class_name
        or "closed" in class_name
        or "websocket" in class_name
        or "unavailable" in status_text
    ):
        return "connection_closed"
    if (
        api_code in {400, 422}
        or "invalid" in status_text
        or "validation" in class_name
        or "invalidargument" in class_name
    ):
        return "validation"
    return "other"


@dataclass(slots=True)
class GeminiTranscribeSTTBackend(STTBackend):
    """Gemini 3.5 Transcribe Live STT Backend using the official google-genai SDK."""

    api_key: str
    language_codes: Sequence[str] = ()
    custom_vocabulary: Sequence[str] = ()
    model: str = GEMINI_TRANSCRIBE_STT_MODEL
    sample_rate_hz: int = GEMINI_TRANSCRIBE_SAMPLE_RATE_HZ
    connect_timeout_s: float = 10.0
    finalize_timeout_s: float = GEMINI_TRANSCRIBE_FINALIZE_TIMEOUT_S
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
        if self.finalize_timeout_s <= 0:
            raise ValueError("finalize_timeout_s must be > 0")

        session = _GeminiTranscribeLiveSession(
            api_key=self.api_key,
            language_codes=list(self.language_codes),
            custom_vocabulary=list(self.custom_vocabulary),
            model=self.model,
            sample_rate_hz=self.sample_rate_hz,
            connect_timeout_s=self.connect_timeout_s,
            finalize_timeout_s=self.finalize_timeout_s,
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
    finalize_timeout_s: float
    live_connect_factory: Callable[[str, Any], Any] | None = None

    _events: asyncio.Queue[STTBackendTranscriptEvent | BaseException | None] = field(
        init=False, repr=False
    )
    _send_queue: asyncio.Queue[_StartTurn | _EndTurn | bytes | object] = field(
        init=False, repr=False
    )
    _live_context: Any = field(init=False, default=None, repr=False)
    _live_session: Any = field(init=False, default=None, repr=False)
    _send_task: asyncio.Task[None] | None = field(init=False, default=None, repr=False)
    _recv_task: asyncio.Task[None] | None = field(init=False, default=None, repr=False)
    _stopped: bool = field(init=False, default=False)
    _capture_turn: _PendingTurn | None = field(init=False, default=None, repr=False)
    _streaming_turn: _PendingTurn | None = field(init=False, default=None, repr=False)
    _pending_turns: deque[_PendingTurn] = field(init=False, default_factory=deque, repr=False)
    _protocol_failed: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        self._events = asyncio.Queue()
        self._send_queue = asyncio.Queue()

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
        self._live_context = live_context
        self._live_session = await asyncio.wait_for(
            live_context.__aenter__(), timeout=self.connect_timeout_s
        )
        elapsed = time.monotonic() - start_at
        logger.info("[STT] Gemini Transcribe Live connected in %.2fs", elapsed)

        self._send_task = asyncio.create_task(self._send_loop())
        self._recv_task = asyncio.create_task(self._recv_loop())

    async def _send_loop(self) -> None:
        try:
            while not self._stopped:
                item = await self._send_queue.get()
                if item is _STOP:
                    return
                if isinstance(item, _StartTurn):
                    from google.genai import types

                    self._streaming_turn = item.turn
                    await self._send_realtime(activity_start=types.ActivityStart())
                    logger.info("[STT] Gemini Transcribe Live activityStart sent")
                    continue
                if isinstance(item, _EndTurn):
                    from google.genai import types

                    turn = item.turn
                    self._pending_turns.append(turn)
                    await self._send_realtime(activity_end=types.ActivityEnd())
                    if turn in self._pending_turns:
                        turn.timeout_task = asyncio.create_task(self._finalize_timeout(turn))
                    logger.info("[STT] Gemini Transcribe Live activityEnd sent (finalize)")
                    await turn.activity_end_ack.wait()
                    self._streaming_turn = None
                    if self._protocol_failed:
                        return
                    continue
                if isinstance(item, bytes):
                    await self._send_realtime(
                        audio={
                            "data": item,
                            "mime_type": f"audio/pcm;rate={self.sample_rate_hz}",
                        },
                    )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("Gemini Transcribe Live send loop error")
            self._put_event(exc)

    async def _recv_loop(self) -> None:
        try:
            while not self._stopped:
                live_session = self._live_session
                if live_session is None:
                    return
                async for message in live_session.receive():
                    self._handle_message(message)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            exception_class, api_code, api_status, message_kind = _recv_failure_fields(exc)
            logger.exception(
                "Gemini Transcribe Live recv loop error exception_class=%s "
                "api_code=%s api_status=%s message_kind=%s",
                exception_class,
                api_code,
                api_status,
                message_kind,
            )
            self._put_event(exc)
        finally:
            self._put_event(None)

    def _handle_message(self, message: Any) -> None:
        if self._protocol_failed:
            return
        content = message.server_content
        if content is not None:
            interim = content.interim_input_transcription
            if interim is not None and interim.text:
                text = str(interim.text)
                turn = self._turn_for_interim()
                if turn is not None:
                    turn.latest_interim = text
                logger.debug("[STT] Gemini Transcribe Live interim text_len=%s", len(text))
            final = content.input_transcription
            if final is not None:
                self._handle_final(str(final.text or ""))

        voice_activity = getattr(message, "voice_activity", None)
        activity_type = getattr(voice_activity, "voice_activity_type", None)
        activity_value = getattr(activity_type, "value", activity_type)
        if str(activity_value or "").upper() == "ACTIVITY_END":
            self._handle_activity_end_ack()

    def _turn_for_interim(self) -> _PendingTurn | None:
        return self._streaming_turn

    def _handle_final(self, text: str) -> None:
        turn = next((item for item in self._pending_turns if not item.final_emitted), None)
        if turn is None:
            logger.debug(
                "[STT] Gemini Transcribe Live final ignored without pending finalize text_len=%s",
                len(text),
            )
            return
        self._emit_turn_final(turn, text)

    def _handle_activity_end_ack(self) -> None:
        if not self._pending_turns:
            logger.debug("[STT] Gemini Transcribe Live activityEnd ack without pending finalize")
            return
        turn = self._pending_turns.popleft()
        self._cancel_turn_timeout(turn)
        if not turn.final_emitted:
            self._emit_turn_final(turn, turn.latest_interim)
        turn.activity_end_ack.set()

    def _emit_turn_final(self, turn: _PendingTurn, text: str) -> None:
        if turn.final_emitted:
            return
        turn.final_emitted = True
        if text:
            logger.info("[STT] Transcript final text_len=%s", len(text))
        else:
            logger.debug("[STT] Gemini Transcribe Live empty finalize ack")
        self._put_event(STTBackendTranscriptEvent(text=text, is_final=True))

    def _cancel_turn_timeout(self, turn: _PendingTurn) -> None:
        task = turn.timeout_task
        turn.timeout_task = None
        if task is not None and task is not asyncio.current_task():
            task.cancel()

    async def _finalize_timeout(self, turn: _PendingTurn) -> None:
        try:
            await asyncio.sleep(self.finalize_timeout_s)
        except asyncio.CancelledError:
            return
        if self._stopped or self._protocol_failed or turn not in self._pending_turns:
            return
        self._protocol_failed = True
        self._pending_turns.remove(turn)
        self._emit_turn_final(turn, turn.latest_interim)
        turn.activity_end_ack.set()
        logger.warning(
            "[STT] Gemini Transcribe Live finalize timed out after %.2fs; recycling session",
            self.finalize_timeout_s,
        )
        self._put_event(
            GeminiTranscribeFinalizeTimeout(
                f"Gemini Transcribe Live finalize timed out after {self.finalize_timeout_s:.2f}s"
            )
        )

    async def send_audio(self, pcm16le: bytes) -> None:
        if self._stopped or self._protocol_failed:
            return
        if self._capture_turn is None:
            self._capture_turn = _PendingTurn()
            await self._send_queue.put(_StartTurn(self._capture_turn))
        await self._send_queue.put(pcm16le)

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
        if self._capture_turn is not None:
            turn = self._capture_turn
            self._capture_turn = None
            await self._send_queue.put(_EndTurn(turn))

    async def _send_realtime(self, **kwargs: Any) -> None:
        live_session = self._live_session
        if live_session is None:
            return
        await live_session.send_realtime_input(**kwargs)

    async def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        await self._send_queue.put(_STOP)
        if self._pending_turns:
            logger.warning(
                "[STT] Gemini Transcribe Live session closed with unresolved finalize requests count=%s",
                len(self._pending_turns),
            )
        for turn in self._pending_turns:
            self._cancel_turn_timeout(turn)
            turn.activity_end_ack.set()
        self._pending_turns.clear()
        self._capture_turn = None
        self._streaming_turn = None
        self._put_event(None)

    async def close(self) -> None:
        await self.stop()
        tasks = [self._send_task, self._recv_task]
        for task in tasks:
            if task is not None:
                task.cancel()
        await asyncio.gather(*(task for task in tasks if task is not None), return_exceptions=True)
        self._send_task = None
        self._recv_task = None
        live_context = self._live_context
        live_session = self._live_session
        self._live_context = None
        self._live_session = None
        if live_context is not None:
            with contextlib.suppress(Exception):
                await live_context.__aexit__(None, None, None)
            return
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
