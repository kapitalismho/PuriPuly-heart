"""Soniox Realtime STT Backend using WebSocket API.

Uses raw WebSocket streaming with manual finalize and keepalive control messages.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Sequence

from puripuly_heart.core.stt.backend import (
    STTBackend,
    STTBackendSession,
    STTBackendTranscriptEvent,
)

logger = logging.getLogger(__name__)

_STOP = object()
_FINALIZE = object()


@dataclass(slots=True)
class SonioxRealtimeSTTBackend(STTBackend):
    """Soniox Realtime STT Backend using WebSocket API."""

    api_key: str
    language_hints: Sequence[str]
    model: str = "stt-rt-v3"
    endpoint: str = "wss://stt-rt.soniox.com/transcribe-websocket"
    sample_rate_hz: int = 16000
    keepalive_interval_s: float = 10.0
    trailing_silence_ms: int = 100

    async def open_session(self) -> STTBackendSession:
        if self.sample_rate_hz not in (8000, 16000):
            raise ValueError("sample_rate_hz must be 8000 or 16000")
        if not self.api_key:
            raise ValueError("api_key must be non-empty")
        if not self.endpoint:
            raise ValueError("endpoint must be non-empty")
        if self.keepalive_interval_s <= 0:
            raise ValueError("keepalive_interval_s must be > 0")
        if self.trailing_silence_ms < 0:
            raise ValueError("trailing_silence_ms must be >= 0")

        session = _SonioxSession(
            api_key=self.api_key,
            model=self.model,
            endpoint=self.endpoint,
            sample_rate_hz=self.sample_rate_hz,
            language_hints=list(self.language_hints),
            keepalive_interval_s=self.keepalive_interval_s,
            trailing_silence_ms=self.trailing_silence_ms,
        )
        await session.start()
        return session

    @staticmethod
    async def verify_api_key(
        api_key: str, *, endpoint: str = "wss://stt-rt.soniox.com/transcribe-websocket"
    ) -> bool:
        if not api_key:
            return False

        import websockets

        async def _check() -> bool:
            try:
                async with websockets.connect(endpoint, ping_interval=None, open_timeout=5) as ws:
                    config = {
                        "api_key": api_key,
                        "model": "stt-rt-v3",
                        "audio_format": "pcm_s16le",
                        "sample_rate": 16000,
                        "num_channels": 1,
                        "enable_endpoint_detection": False,
                    }
                    await ws.send(json.dumps(config))
                    try:
                        message = await asyncio.wait_for(ws.recv(), timeout=3.0)
                    except asyncio.TimeoutError:
                        return True
                    if isinstance(message, bytes):
                        message = message.decode("utf-8", errors="ignore")
                    data = json.loads(message)
                    if "error" in data or "error_code" in data:
                        raise Exception(data.get("error") or data.get("error_code"))
                    return True
            except Exception as exc:
                raise Exception(f"Connection failed: {exc}") from exc

        return await _check()


@dataclass(slots=True)
class _SonioxSession(STTBackendSession):
    """Internal session using Soniox WebSocket API."""

    api_key: str
    model: str
    endpoint: str
    sample_rate_hz: int
    language_hints: list[str]
    keepalive_interval_s: float
    trailing_silence_ms: int

    _events: asyncio.Queue[STTBackendTranscriptEvent | BaseException | None] = field(
        init=False, repr=False
    )
    _audio_q: asyncio.Queue[bytes | object] = field(init=False, repr=False)
    _ws: Any = field(init=False, default=None, repr=False)
    _send_task: asyncio.Task[None] | None = field(init=False, default=None, repr=False)
    _recv_task: asyncio.Task[None] | None = field(init=False, default=None, repr=False)
    _keepalive_task: asyncio.Task[None] | None = field(init=False, default=None, repr=False)
    _stopped: bool = field(init=False, default=False)
    _last_send_at: float | None = field(init=False, default=None)
    _final_tokens: list[str] = field(init=False, default_factory=list)
    _last_final_end_ms: int | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self._events = asyncio.Queue()
        self._audio_q = asyncio.Queue()

    async def start(self) -> None:
        import websockets

        config: dict[str, Any] = {
            "api_key": self.api_key,
            "model": self.model,
            "audio_format": "pcm_s16le",
            "sample_rate": self.sample_rate_hz,
            "num_channels": 1,
            "enable_endpoint_detection": False,
        }
        if self.language_hints:
            config["language_hints"] = self.language_hints

        self._ws = await websockets.connect(self.endpoint, ping_interval=None, open_timeout=5)
        await self._ws.send(json.dumps(config))
        self._last_send_at = time.monotonic()

        self._send_task = asyncio.create_task(self._send_loop())
        self._recv_task = asyncio.create_task(self._recv_loop())
        self._keepalive_task = asyncio.create_task(self._keepalive_loop())

    async def _send_loop(self) -> None:
        if self._ws is None:
            return
        try:
            while True:
                data = await self._audio_q.get()
                if data is _STOP:
                    return
                if data is _FINALIZE:
                    payload = {
                        "type": "finalize",
                        "trailing_silence_ms": self.trailing_silence_ms,
                    }
                    await self._ws.send(json.dumps(payload))
                    self._last_send_at = time.monotonic()
                    continue
                if isinstance(data, bytes):
                    await self._ws.send(data)
                    self._last_send_at = time.monotonic()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("Soniox send loop error")
            self._put_event(exc)

    async def _recv_loop(self) -> None:
        if self._ws is None:
            return
        try:
            while True:
                message = await self._ws.recv()
                if message is None:
                    return
                self._handle_message(message)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            try:
                from websockets.exceptions import ConnectionClosedOK

                if isinstance(exc, ConnectionClosedOK):
                    return
            except Exception:
                pass
            logger.exception("Soniox recv loop error")
            self._put_event(exc)
        finally:
            self._stopped = True
            self._put_event(None)

    async def _keepalive_loop(self) -> None:
        if self._ws is None:
            return
        try:
            while not self._stopped:
                await asyncio.sleep(self.keepalive_interval_s)
                if self._stopped or self._ws is None:
                    return
                now = time.monotonic()
                last = self._last_send_at or 0.0
                if now - last >= self.keepalive_interval_s:
                    await self._ws.send(json.dumps({"type": "keepalive"}))
                    self._last_send_at = now
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.debug(f"Soniox keepalive failed: {exc}")

    def _handle_message(self, message: str | bytes) -> None:
        if isinstance(message, bytes):
            message = message.decode("utf-8", errors="ignore")
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            logger.debug("Soniox message parse error")
            return

        if "error" in data or "error_code" in data:
            error_msg = data.get("error") or data.get("error_code") or "Unknown error"
            self._put_event(RuntimeError(f"Soniox error: {error_msg}"))
            return

        tokens = data.get("tokens") or []
        if not isinstance(tokens, list):
            return

        for token in tokens:
            if not isinstance(token, dict):
                continue
            text = str(token.get("text", "") or "")
            is_final = bool(token.get("is_final"))
            if not is_final:
                continue
            if text in ("<fin>", "<end>"):
                self._flush_final()
                continue
            end_ms = token.get("end_ms")
            if isinstance(end_ms, (int, float)):
                end_ms = int(end_ms)
                if self._last_final_end_ms is not None and end_ms <= self._last_final_end_ms:
                    continue
                self._last_final_end_ms = end_ms
            self._final_tokens.append(text)

    def _flush_final(self) -> None:
        if not self._final_tokens:
            return
        text = "".join(self._final_tokens).strip()
        self._final_tokens.clear()
        self._last_final_end_ms = None
        if not text:
            return
        self._put_event(STTBackendTranscriptEvent(text=text, is_final=True))

    def _put_event(self, event: STTBackendTranscriptEvent | BaseException | None) -> None:
        self._events.put_nowait(event)

    async def send_audio(self, pcm16le: bytes) -> None:
        if self._stopped:
            return
        await self._audio_q.put(pcm16le)

    async def on_speech_end(self) -> None:
        if self._stopped:
            return

        import numpy as np

        silence_samples = int(self.sample_rate_hz * (self.trailing_silence_ms / 1000.0))
        if silence_samples > 0:
            silence = np.zeros(silence_samples, dtype=np.float32)
            pcm16 = (silence * 32767).astype(np.int16).tobytes()
            await self._audio_q.put(pcm16)
            logger.info(
                f"[STT] Trailing silence sent ({silence_samples} samples, {len(pcm16)} bytes)"
            )

        await self._audio_q.put(_FINALIZE)

    async def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        if self._ws is not None:
            with contextlib.suppress(Exception):
                await self._ws.send("")
        await self._audio_q.put(_STOP)

    async def close(self) -> None:
        await self.stop()
        tasks = [self._send_task, self._recv_task, self._keepalive_task]
        for task in tasks:
            if task is None:
                continue
            task.cancel()
        await asyncio.gather(*(t for t in tasks if t is not None), return_exceptions=True)
        if self._ws is not None:
            with contextlib.suppress(Exception):
                await self._ws.close()
            self._ws = None

    async def events(self) -> AsyncIterator[STTBackendTranscriptEvent]:
        while True:
            item = await self._events.get()
            if item is None:
                return
            if isinstance(item, BaseException):
                raise item
            yield item


import contextlib  # placed at bottom to keep the main logic compact
