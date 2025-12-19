from __future__ import annotations

import asyncio
import contextlib
import queue
import threading
import time
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

    _control_q: queue.Queue[bytes | object] = field(init=False, repr=False)
    _events: queue.Queue[STTBackendTranscriptEvent | BaseException | None] = field(init=False, repr=False)
    _recognition: Any | None = field(init=False, default=None, repr=False)
    _thread: threading.Thread | None = field(init=False, default=None, repr=False)
    _stopped: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        self._control_q = queue.Queue()
        self._events = queue.Queue()

    async def start(self) -> None:
        self._thread = threading.Thread(target=self._thread_main, name="dashscope-asr", daemon=True)
        self._thread.start()

    async def send_audio(self, pcm16le: bytes) -> None:
        if self._stopped:
            return
        self._control_q.put_nowait(pcm16le)

    async def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        self._control_q.put_nowait(_STOP)

    async def close(self) -> None:
        await self.stop()
        if self._thread is not None:
            self._thread.join()
            self._thread = None

        if self._recognition is not None:
            with contextlib.suppress(Exception):
                self._recognition.stop()
            self._recognition = None

    async def events(self) -> AsyncIterator[STTBackendTranscriptEvent]:
        while True:
            try:
                item = self._events.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.01)
                continue
            if item is None:
                return
            if isinstance(item, BaseException):
                raise item
            yield item

    def _thread_main(self) -> None:
        retries = 0
        try:
            while True:
                message = self._control_q.get()
                if message is _STOP:
                    break
                if message is _RECONNECT:
                    self._restart_with_backoff(retries=retries)
                    retries = min(retries + 1, self.max_retries)
                    continue

                audio = message
                retries = 0
                self._ensure_started()

                while True:
                    try:
                        self._recognition.send_audio_frame(audio)  # type: ignore[union-attr]
                        break
                    except Exception:
                        retries += 1
                        if retries > self.max_retries:
                            raise
                        self._restart_with_backoff(retries=retries)
                        self._ensure_started()
        except BaseException as exc:
            self._events.put(exc)
        finally:
            if self._recognition is not None:
                with contextlib.suppress(Exception):
                    self._recognition.stop()
                self._recognition = None
            self._events.put(None)

    def _restart_with_backoff(self, *, retries: int) -> None:
        if self._recognition is not None:
            with contextlib.suppress(Exception):
                self._recognition.stop()
        self._recognition = None

        delay = self.retry_backoff_s * max(retries, 1)
        time.sleep(delay)

    def _ensure_started(self) -> None:
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
                session._events.put(event)

            def on_error(self, message):  # type: ignore[no-untyped-def]
                # Transient errors should trigger reconnect attempts; only raise when retries are exhausted.
                session._control_q.put_nowait(_RECONNECT)

            def on_close(self) -> None:
                return

            def on_complete(self) -> None:
                return

        return _Callback()
