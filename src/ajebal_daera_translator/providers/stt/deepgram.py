"""Deepgram Realtime STT Backend using official SDK.

WebSocket-based Speech-to-Text using Deepgram's nova-3 model via deepgram-sdk.
"""

from __future__ import annotations

import asyncio
import contextlib
import queue
import threading
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from ajebal_daera_translator.core.stt.backend import (
    STTBackend,
    STTBackendSession,
    STTBackendTranscriptEvent,
)


@dataclass(slots=True)
class DeepgramRealtimeSTTBackend(STTBackend):
    """Deepgram Realtime STT Backend using official deepgram-sdk."""

    api_key: str
    model: str = "nova-3"
    language: str = "ko"
    sample_rate_hz: int = 16000

    async def open_session(self) -> STTBackendSession:
        if self.sample_rate_hz not in (8000, 16000):
            raise ValueError("sample_rate_hz must be 8000 or 16000")
        if not self.api_key:
            raise ValueError("api_key must be non-empty")

        session = _DeepgramSDKSession(
            api_key=self.api_key,
            model=self.model,
            language=self.language,
            sample_rate_hz=self.sample_rate_hz,
        )
        await session.start()
        return session


_STOP = object()


@dataclass(slots=True)
class _DeepgramSDKSession(STTBackendSession):
    """Internal session using Deepgram SDK for WebSocket management."""

    api_key: str
    model: str
    language: str
    sample_rate_hz: int

    _events: asyncio.Queue[STTBackendTranscriptEvent | BaseException | None] = field(init=False, repr=False)
    _audio_q: queue.Queue[bytes | object] = field(init=False, repr=False)
    _connection: Any = field(init=False, default=None, repr=False)
    _thread: threading.Thread | None = field(init=False, default=None, repr=False)
    _stopped: bool = field(init=False, default=False)
    _loop: asyncio.AbstractEventLoop | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        self._events = asyncio.Queue()
        self._audio_q = queue.Queue()

    async def start(self) -> None:
        self._loop = asyncio.get_event_loop()
        self._thread = threading.Thread(target=self._run_sync, name="deepgram-sdk", daemon=True)
        self._thread.start()

    def _run_sync(self) -> None:
        """Run Deepgram SDK in a separate thread with its own event loop."""
        import asyncio as aio

        try:
            aio.run(self._async_main())
        except BaseException as exc:
            self._put_event(exc)
        finally:
            self._put_event(None)

    async def _async_main(self) -> None:
        """Main async function running in the thread."""
        from deepgram import DeepgramClient, LiveTranscriptionEvents, LiveOptions  # type: ignore

        client = DeepgramClient(self.api_key)
        connection = client.listen.websocket.v("1")

        # Register event handlers
        connection.on(LiveTranscriptionEvents.Transcript, self._on_transcript)
        connection.on(LiveTranscriptionEvents.Error, self._on_error)

        options = LiveOptions(
            model=self.model,
            language=self.language,
            encoding="linear16",
            sample_rate=self.sample_rate_hz,
            channels=1,
        )

        if not connection.start(options):
            raise RuntimeError("Failed to start Deepgram connection")

        self._connection = connection

        # Audio sending loop
        try:
            while True:
                try:
                    data = self._audio_q.get(timeout=0.1)
                except queue.Empty:
                    continue

                if data is _STOP:
                    break

                if isinstance(data, bytes) and self._connection:
                    self._connection.send(data)
        finally:
            if self._connection:
                with contextlib.suppress(Exception):
                    self._connection.finish()
                self._connection = None

    def _on_transcript(self, _client: Any, result: Any, **kwargs: Any) -> None:
        """Handle transcript events from Deepgram SDK."""
        _ = kwargs
        try:
            channel = result.channel
            if not channel or not channel.alternatives:
                return

            transcript = channel.alternatives[0].transcript
            if not transcript:
                return

            is_final = result.is_final if hasattr(result, "is_final") else False

            event = STTBackendTranscriptEvent(text=transcript.strip(), is_final=is_final)
            self._put_event(event)
        except Exception:
            pass

    def _on_error(self, _client: Any, error: Any, **kwargs: Any) -> None:
        """Handle error events from Deepgram SDK."""
        _ = kwargs
        self._put_event(RuntimeError(f"Deepgram error: {error}"))

    def _put_event(self, event: STTBackendTranscriptEvent | BaseException | None) -> None:
        """Thread-safe event posting to the asyncio queue."""
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._events.put_nowait, event)

    async def send_audio(self, pcm16le: bytes) -> None:
        if self._stopped:
            return
        self._audio_q.put_nowait(pcm16le)

    async def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        self._audio_q.put_nowait(_STOP)

    async def close(self) -> None:
        await self.stop()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    async def events(self) -> AsyncIterator[STTBackendTranscriptEvent]:
        while True:
            item = await self._events.get()
            if item is None:
                return
            if isinstance(item, BaseException):
                raise item
            yield item
