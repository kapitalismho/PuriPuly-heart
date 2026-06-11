"""60db Realtime STT Backend using the streaming WebSocket API.

Uses 60db's ``/ws/stt`` endpoint with raw ``websockets`` streaming.

Unlike Deepgram/Soniox, 60db exposes no per-utterance "finalize" control
message - the only client controls are ``start`` / ``audio`` / ``stop``. The
server emits a final ``transcription`` (``speech_final=true``) once its own
endpointing detects end-of-utterance. We rely on local VAD to gate speech and
then top up trailing silence on speech-end so the server's endpointing fires
promptly. The session stays open between utterances via ``continuous_mode``.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Sequence
from urllib.parse import quote

from puripuly_heart.core.stt.backend import (
    STTBackend,
    STTBackendSession,
    STTBackendTranscriptEvent,
)

logger = logging.getLogger(__name__)

_STOP = object()
_FINALIZE = object()

# 60db rejects utterance_end_ms below this floor.
_MIN_UTTERANCE_END_MS = 300


def _build_ws_url(endpoint: str, api_key: str) -> str:
    """Append the apiKey query parameter (60db authenticates via the URL)."""
    separator = "&" if "?" in endpoint else "?"
    return f"{endpoint}{separator}apiKey={quote(api_key, safe='')}"


@dataclass(slots=True)
class SixtyDBRealtimeSTTBackend(STTBackend):
    """60db Realtime STT Backend using the streaming WebSocket API."""

    api_key: str
    language_codes: Sequence[str]
    context_terms: Sequence[str] = ()
    endpoint: str = "wss://api.60db.ai/ws/stt"
    sample_rate_hz: int = 16000
    utterance_end_ms: int = 300
    trailing_silence_ms: int = 400
    connect_timeout_s: float = 5.0

    async def open_session(self) -> STTBackendSession:
        if self.sample_rate_hz not in (8000, 16000, 24000, 44100, 48000):
            raise ValueError("sample_rate_hz must be one of 8000/16000/24000/44100/48000")
        if not self.api_key:
            raise ValueError("api_key must be non-empty")
        if not self.endpoint:
            raise ValueError("endpoint must be non-empty")
        if self.utterance_end_ms < _MIN_UTTERANCE_END_MS:
            raise ValueError(f"utterance_end_ms must be >= {_MIN_UTTERANCE_END_MS}")
        if self.trailing_silence_ms < 0:
            raise ValueError("trailing_silence_ms must be >= 0")
        if self.connect_timeout_s <= 0:
            raise ValueError("connect_timeout_s must be > 0")

        session = _SixtyDBSession(
            api_key=self.api_key,
            endpoint=self.endpoint,
            sample_rate_hz=self.sample_rate_hz,
            language_codes=list(self.language_codes),
            context_terms=list(self.context_terms),
            utterance_end_ms=self.utterance_end_ms,
            trailing_silence_ms=self.trailing_silence_ms,
            connect_timeout_s=self.connect_timeout_s,
        )
        await session.start()
        return session

    @staticmethod
    async def verify_api_key(
        api_key: str, *, endpoint: str = "wss://api.60db.ai/ws/stt"
    ) -> bool:
        if not api_key:
            return False

        import websockets

        async def _check() -> bool:
            url = _build_ws_url(endpoint, api_key)
            try:
                async with websockets.connect(
                    url, ping_interval=None, open_timeout=5
                ) as ws:
                    try:
                        message = await asyncio.wait_for(ws.recv(), timeout=3.0)
                    except asyncio.TimeoutError:
                        # Connection stayed open without rejecting the key.
                        return True
                    if isinstance(message, bytes):
                        message = message.decode("utf-8", errors="ignore")
                    data = json.loads(message)
                    if "connection_established" in data:
                        return True
                    if "error" in data or data.get("type") == "error":
                        raise Exception(data.get("error") or data.get("message"))
                    # Any other first frame (e.g. "connecting") implies the
                    # handshake is proceeding, so the key was accepted.
                    return True
            except Exception as exc:
                raise Exception(f"Connection failed: {exc}") from exc

        return await _check()


@dataclass(slots=True)
class _SixtyDBSession(STTBackendSession):
    """Internal session using the 60db /ws/stt WebSocket API."""

    api_key: str
    endpoint: str
    sample_rate_hz: int
    language_codes: list[str]
    context_terms: list[str]
    utterance_end_ms: int
    trailing_silence_ms: int
    connect_timeout_s: float

    _events: asyncio.Queue[STTBackendTranscriptEvent | BaseException | None] = field(
        init=False, repr=False
    )
    _audio_q: asyncio.Queue[bytes | object] = field(init=False, repr=False)
    _ws: Any = field(init=False, default=None, repr=False)
    _send_task: asyncio.Task[None] | None = field(init=False, default=None, repr=False)
    _recv_task: asyncio.Task[None] | None = field(init=False, default=None, repr=False)
    _stopped: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        self._events = asyncio.Queue()
        self._audio_q = asyncio.Queue()

    def _build_start_message(self) -> dict[str, Any]:
        config: dict[str, Any] = {
            "type": "start",
            "config": {
                "encoding": "linear",
                "sample_rate": self.sample_rate_hz,
                "utterance_end_ms": self.utterance_end_ms,
                "continuous_mode": True,
            },
        }
        if self.language_codes:
            config["languages"] = self.language_codes
        if self.context_terms:
            config["context"] = {"terms": self.context_terms}
        return config

    async def start(self) -> None:
        import websockets

        url = _build_ws_url(self.endpoint, self.api_key)
        logger.info("[STT] 60db connecting (timeout=%.1fs)", self.connect_timeout_s)
        start_at = time.monotonic()
        self._ws = await websockets.connect(
            url, ping_interval=None, open_timeout=self.connect_timeout_s
        )

        # Drain the handshake until the connection is established (or rejected).
        await self._await_connection_established()

        elapsed = time.monotonic() - start_at
        logger.info("[STT] 60db connected in %.2fs", elapsed)
        await self._ws.send(json.dumps(self._build_start_message()))

        self._send_task = asyncio.create_task(self._send_loop())
        self._recv_task = asyncio.create_task(self._recv_loop())

    async def _await_connection_established(self) -> None:
        while True:
            raw = await asyncio.wait_for(self._ws.recv(), timeout=self.connect_timeout_s)
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", errors="ignore")
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if "error" in data or data.get("type") == "error":
                raise RuntimeError(
                    f"60db error: {data.get('error') or data.get('message') or 'auth failed'}"
                )
            if "connection_established" in data:
                return
            # Ignore transient "connecting" frames and keep waiting.

    async def _send_loop(self) -> None:
        if self._ws is None:
            return
        try:
            while True:
                data = await self._audio_q.get()
                if data is _STOP:
                    with contextlib.suppress(Exception):
                        await self._ws.send(json.dumps({"type": "stop"}))
                    return
                if data is _FINALIZE:
                    # 60db has no per-utterance finalize control; trailing
                    # silence (already enqueued) drives server endpointing.
                    continue
                if isinstance(data, bytes):
                    payload = {
                        "type": "audio",
                        "audio": base64.b64encode(data).decode("ascii"),
                        "encoding": "linear",
                        "sample_rate": self.sample_rate_hz,
                    }
                    await self._ws.send(json.dumps(payload))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("60db send loop error")
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
            logger.exception("60db recv loop error")
            self._put_event(exc)
        finally:
            self._stopped = True
            self._put_event(None)

    def _handle_message(self, message: str | bytes) -> None:
        if isinstance(message, bytes):
            message = message.decode("utf-8", errors="ignore")
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            logger.debug("60db message parse error")
            return

        if "error" in data or data.get("type") == "error":
            error_msg = data.get("error") or data.get("message") or "Unknown error"
            self._put_event(RuntimeError(f"60db error: {error_msg}"))
            return

        if data.get("type") != "transcription":
            # speech_started / language_changed / session_stopped / connected /
            # test_response — not transcript-bearing.
            return

        # Interim partials are ignored; the app only emits finals.
        if not bool(data.get("is_final")):
            return
        # Two-phase refinement: skip the fast dict-corrected emit and wait for
        # the canonical (LLM-refined) result. When no context is supplied 60db
        # sends a single event with speech_final=true, so this is a no-op there.
        if not bool(data.get("speech_final")):
            return

        text = str(data.get("text", "") or "").strip()
        if not text:
            # Empty final = speech-end-no-result (silence / low confidence).
            return
        logger.info("[STT] Transcript: '%s' (final)", text)
        self._put_event(STTBackendTranscriptEvent(text=text, is_final=True))

    def _put_event(self, event: STTBackendTranscriptEvent | BaseException | None) -> None:
        self._events.put_nowait(event)

    async def send_audio(self, pcm16le: bytes) -> None:
        if self._stopped:
            return
        await self._audio_q.put(pcm16le)

    async def on_speech_end(self, *, trailing_silence_ms: int | None = None) -> None:
        if self._stopped:
            return

        # Top up trailing silence so 60db's server-side endpointing fires and
        # emits the final transcript for this utterance.
        existing_ms = max(int(trailing_silence_ms or 0), 0)
        target_ms = max(int(self.trailing_silence_ms), 0)
        missing_ms = max(target_ms - existing_ms, 0)
        if missing_ms > 0:
            import numpy as np

            silence_samples = int(self.sample_rate_hz * (missing_ms / 1000.0))
            if silence_samples > 0:
                silence = np.zeros(silence_samples, dtype=np.float32)
                pcm16 = (silence * 32767).astype(np.int16).tobytes()
                await self._audio_q.put(pcm16)
                logger.info(
                    "[STT] Trailing silence sent (%sms, %s samples, %s bytes)",
                    missing_ms,
                    silence_samples,
                    len(pcm16),
                )

        await self._audio_q.put(_FINALIZE)

    async def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        await self._audio_q.put(_STOP)

    async def close(self) -> None:
        await self.stop()
        tasks = [self._send_task, self._recv_task]
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
