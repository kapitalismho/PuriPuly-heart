"""Deepgram Realtime STT Backend using raw WebSocket connection.

WebSocket-based Speech-to-Text using Deepgram's nova-3 model.
Uses websocket-client for direct WebSocket control similar to Alibaba STT backend.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
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
    """Deepgram Realtime STT Backend using raw WebSocket."""

    api_key: str
    model: str = "nova-3"
    language: str = "ko"
    sample_rate_hz: int = 16000

    async def open_session(self) -> STTBackendSession:
        if self.sample_rate_hz not in (8000, 16000):
            raise ValueError("sample_rate_hz must be 8000 or 16000")
        if not self.api_key:
            raise ValueError("api_key must be non-empty")

        session = _DeepgramWebSocketSession(
            api_key=self.api_key,
            model=self.model,
            language=self.language,
            sample_rate_hz=self.sample_rate_hz,
        )
        await session.start()
        return session


_STOP = object()


@dataclass(slots=True)
class _DeepgramWebSocketSession(STTBackendSession):
    """Internal session using raw WebSocket for Deepgram Streaming API."""

    api_key: str
    model: str
    language: str
    sample_rate_hz: int

    _events: asyncio.Queue[STTBackendTranscriptEvent | BaseException | None] = field(init=False, repr=False)
    _audio_q: queue.Queue[bytes | object] = field(init=False, repr=False)
    _ws: Any = field(init=False, default=None, repr=False)
    _thread: threading.Thread | None = field(init=False, default=None, repr=False)
    _stopped: bool = field(init=False, default=False)
    _loop: asyncio.AbstractEventLoop | None = field(init=False, default=None, repr=False)
    _connected: threading.Event = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._events = asyncio.Queue()
        self._audio_q = queue.Queue()
        self._connected = threading.Event()

    async def start(self) -> None:
        self._loop = asyncio.get_event_loop()
        self._thread = threading.Thread(target=self._run_sync, name="deepgram-ws", daemon=True)
        self._thread.start()

    def _build_url(self) -> str:
        """Build Deepgram WebSocket URL with query parameters."""
        base = "wss://api.deepgram.com/v1/listen"
        params = [
            f"model={self.model}",
            f"language={self.language}",
            f"encoding=linear16",
            f"sample_rate={self.sample_rate_hz}",
            f"channels=1",
            f"interim_results=true",
            f"punctuate=true",
        ]
        return f"{base}?{'&'.join(params)}"

    def _run_sync(self) -> None:
        """Run WebSocket connection in a separate thread."""
        import websocket

        try:
            url = self._build_url()
            headers = {"Authorization": f"Token {self.api_key}"}

            self._ws = websocket.WebSocketApp(
                url,
                header=headers,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
                on_open=self._on_open,
            )

            # Start WebSocket in its own thread with run_forever
            ws_thread = threading.Thread(target=self._ws.run_forever, daemon=True)
            ws_thread.start()

            # Wait for WebSocket connection to be established
            if not self._connected.wait(timeout=5.0):
                self._put_event(RuntimeError("Deepgram WebSocket connection timeout"))
                return

            # Audio sending loop
            while True:
                try:
                    data = self._audio_q.get(timeout=0.1)
                except queue.Empty:
                    if self._stopped:
                        break
                    continue

                if data is _STOP:
                    break

                if isinstance(data, bytes) and self._ws:
                    try:
                        self._ws.send(data, opcode=websocket.ABNF.OPCODE_BINARY)
                    except Exception:
                        break

            # Send CloseStream message
            if self._ws:
                with contextlib.suppress(Exception):
                    self._ws.send(json.dumps({"type": "CloseStream"}))

        except BaseException as exc:
            self._put_event(exc)
        finally:
            if self._ws:
                with contextlib.suppress(Exception):
                    self._ws.close()
            self._put_event(None)

    def _on_open(self, ws: Any) -> None:
        """Called when WebSocket connection is established."""
        print("[DEBUG] Deepgram WS Sent Open", flush=True)
        self._connected.set()
        _ = ws

    def _on_message(self, ws: Any, message: str) -> None:
        """Handle incoming messages from Deepgram."""
        _ = ws
        try:
            data = json.loads(message)
            msg_type = data.get("type", "")

            if msg_type == "Results":
                channel = data.get("channel", {})
                alternatives = channel.get("alternatives", [])
                if alternatives:
                    transcript = alternatives[0].get("transcript", "")
                    # print(f"[DEBUG] Deepgram Msg: {transcript}", flush=True)
                    if transcript:
                        is_final = data.get("is_final", False)
                        event = STTBackendTranscriptEvent(text=transcript.strip(), is_final=is_final)
                        self._put_event(event)
            else:
                 print(f"[DEBUG] Deepgram Non-Result: {msg_type}", flush=True)

        except Exception as e:
            print(f"[DEBUG] Deepgram Parse Error: {e}", flush=True)

    def _on_error(self, ws: Any, error: Any) -> None:
        """Handle WebSocket errors."""
        _ = ws
        print(f"[DEBUG] Deepgram WS Error: {error}", flush=True)
        self._put_event(RuntimeError(f"Deepgram WebSocket error: {error}"))

    def _on_close(self, ws: Any, close_status_code: Any, close_msg: Any) -> None:
        """Handle WebSocket close."""
        print(f"[DEBUG] Deepgram WS Closed: {close_status_code} {close_msg}", flush=True)
        _ = ws, close_status_code, close_msg

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
