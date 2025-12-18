from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from ajebal_daera_translator.core.stt.backend import (
    STTBackend,
    STTBackendSession,
    STTBackendTranscriptEvent,
)


@dataclass(slots=True)
class AlibabaModelStudioRealtimeSTTBackend(STTBackend):
    api_key: str
    model: str = "paraformer-realtime-v2"
    endpoint: str = "wss://dashscope-intl.aliyuncs.com/api-ws/v1/inference"
    sample_rate_hz: int = 16000
    semantic_punctuation_enabled: bool = False
    heartbeat: bool = True
    max_retries: int = 3
    retry_backoff_s: float = 0.5

    async def open_session(self) -> STTBackendSession:
        if self.sample_rate_hz not in (8000, 16000):
            raise ValueError("sample_rate_hz must be 8000 or 16000")
        if not self.api_key:
            raise ValueError("api_key must be non-empty")
        if not self.endpoint:
            raise ValueError("endpoint must be non-empty")

        import dashscope  # type: ignore
        from dashscope.audio.asr import Recognition, RecognitionCallback, RecognitionResult  # type: ignore

        dashscope.api_key = self.api_key
        dashscope.base_websocket_api_url = self.endpoint

        session = _DashScopeRealtimeSession(
            Recognition=Recognition,
            RecognitionCallback=RecognitionCallback,
            RecognitionResult=RecognitionResult,
            model=self.model,
            sample_rate_hz=self.sample_rate_hz,
            semantic_punctuation_enabled=self.semantic_punctuation_enabled,
            heartbeat=self.heartbeat,
            max_retries=self.max_retries,
            retry_backoff_s=self.retry_backoff_s,
        )
        await session.start()
        return session


_STOP = object()
_RECONNECT = object()


@dataclass(slots=True)
class _DashScopeRealtimeSession(STTBackendSession):
    Recognition: Any
    RecognitionCallback: Any
    RecognitionResult: Any
    model: str
    sample_rate_hz: int
    semantic_punctuation_enabled: bool
    heartbeat: bool
    max_retries: int
    retry_backoff_s: float

    _control_q: asyncio.Queue[bytes | object] = field(init=False, repr=False)
    _events: asyncio.Queue[STTBackendTranscriptEvent | BaseException | None] = field(init=False, repr=False)
    _loop: asyncio.AbstractEventLoop | None = field(init=False, default=None, repr=False)
    _recognition: Any | None = field(init=False, default=None, repr=False)
    _worker: asyncio.Task[None] | None = field(init=False, default=None, repr=False)
    _stopped: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        self._control_q = asyncio.Queue()
        self._events = asyncio.Queue()

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._worker = asyncio.create_task(self._run())

    async def send_audio(self, pcm16le: bytes) -> None:
        if self._stopped:
            return
        await self._control_q.put(pcm16le)

    async def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        await self._control_q.put(_STOP)

    async def close(self) -> None:
        await self.stop()
        if self._worker is not None:
            await asyncio.gather(self._worker, return_exceptions=True)
            self._worker = None

        if self._recognition is not None:
            with contextlib.suppress(Exception):
                self._recognition.stop()
            self._recognition = None

    async def events(self) -> AsyncIterator[STTBackendTranscriptEvent]:
        while True:
            item = await self._events.get()
            if item is None:
                return
            if isinstance(item, BaseException):
                raise item
            yield item

    async def _run(self) -> None:
        assert self._loop is not None

        retries = 0
        try:
            while True:
                message = await self._control_q.get()
                if message is _STOP:
                    break
                if message is _RECONNECT:
                    await self._restart_with_backoff(retries=retries)
                    retries = min(retries + 1, self.max_retries)
                    continue

                audio = message
                retries = 0
                await self._ensure_started()

                while True:
                    try:
                        await asyncio.to_thread(self._recognition.send_audio_frame, audio)  # type: ignore[union-attr]
                        break
                    except Exception:
                        retries += 1
                        if retries > self.max_retries:
                            raise
                        await self._restart_with_backoff(retries=retries)
                        await self._ensure_started()
        except BaseException as exc:
            await self._events.put(exc)
        finally:
            if self._recognition is not None:
                with contextlib.suppress(Exception):
                    self._recognition.stop()
                self._recognition = None
            await self._events.put(None)

    async def _restart_with_backoff(self, *, retries: int) -> None:
        if self._recognition is not None:
            with contextlib.suppress(Exception):
                self._recognition.stop()
        self._recognition = None

        delay = self.retry_backoff_s * max(retries, 1)
        await asyncio.sleep(delay)

    async def _ensure_started(self) -> None:
        if self._recognition is not None:
            return

        callback = self._make_callback()
        recognition = self.Recognition(
            model=self.model,
            format="pcm",
            sample_rate=self.sample_rate_hz,
            semantic_punctuation_enabled=self.semantic_punctuation_enabled,
            heartbeat=self.heartbeat,
            callback=callback,
        )
        recognition.start()
        self._recognition = recognition

    def _make_callback(self) -> Any:
        session = self

        class _Callback(session.RecognitionCallback):
            def on_event(self, result):  # type: ignore[no-untyped-def]
                sentence = result.get_sentence()
                text = sentence.get("text") if isinstance(sentence, dict) else None
                if not text:
                    return
                is_final = False
                with contextlib.suppress(Exception):
                    is_final = bool(session.RecognitionResult.is_sentence_end(sentence))
                event = STTBackendTranscriptEvent(text=str(text).strip(), is_final=is_final)
                session._loop.call_soon_threadsafe(session._events.put_nowait, event)  # type: ignore[union-attr]

            def on_error(self, message):  # type: ignore[no-untyped-def]
                # Transient errors should trigger reconnect attempts; only raise when retries are exhausted.
                session._loop.call_soon_threadsafe(session._control_q.put_nowait, _RECONNECT)  # type: ignore[union-attr]

            def on_close(self) -> None:
                return

            def on_complete(self) -> None:
                return

        return _Callback()
