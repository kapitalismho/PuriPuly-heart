"""Soniox Realtime STT Backend using WebSocket API.

Uses raw WebSocket streaming with manual finalize and keepalive control messages.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
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


@dataclass(frozen=True, slots=True)
class _FinalizeRequest:
    trailing_silence_ms: int | None = None


@dataclass(frozen=True, slots=True)
class _FinalToken:
    text: str
    end_ms: int | None


@dataclass(slots=True)
class SonioxRealtimeSTTBackend(STTBackend):
    """Soniox Realtime STT Backend using WebSocket API."""

    api_key: str
    language_hints: Sequence[str]
    context_terms: Sequence[str] = ()
    model: str = "stt-rt-v5"
    endpoint: str = "wss://stt-rt.soniox.com/transcribe-websocket"
    sample_rate_hz: int = 16000
    keepalive_interval_s: float = 10.0
    trailing_silence_ms: int = 100
    connect_timeout_s: float = 5.0

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
        if self.connect_timeout_s <= 0:
            raise ValueError("connect_timeout_s must be > 0")

        session = _SonioxSession(
            api_key=self.api_key,
            model=self.model,
            endpoint=self.endpoint,
            sample_rate_hz=self.sample_rate_hz,
            language_hints=list(self.language_hints),
            context_terms=list(self.context_terms),
            keepalive_interval_s=self.keepalive_interval_s,
            trailing_silence_ms=self.trailing_silence_ms,
            connect_timeout_s=self.connect_timeout_s,
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
                        "model": "stt-rt-v5",
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
    context_terms: list[str]
    keepalive_interval_s: float
    trailing_silence_ms: int
    connect_timeout_s: float

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
    _pending_tokens: list[_FinalToken] = field(init=False, default_factory=list)
    _pending_last_end_ms: int | None = field(init=False, default=None)
    _final_tokens: list[_FinalToken] = field(init=False, default_factory=list)
    _pending_finalize_requests: int = field(init=False, default=0)

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
        if self.context_terms:
            config["context"] = {"terms": self.context_terms}

        logger.info("[STT] Soniox connecting (timeout=%.1fs)", self.connect_timeout_s)
        start_at = time.monotonic()
        self._ws = await websockets.connect(
            self.endpoint, ping_interval=None, open_timeout=self.connect_timeout_s
        )
        elapsed = time.monotonic() - start_at
        logger.info("[STT] Soniox connected in %.2fs", elapsed)
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
                if isinstance(data, _FinalizeRequest):
                    payload = {"type": "finalize"}
                    if data.trailing_silence_ms is not None:
                        payload["trailing_silence_ms"] = data.trailing_silence_ms
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

        if tokens:
            logger.debug("[STT] Soniox tokens received count=%s", len(tokens))

        for token in tokens:
            if not isinstance(token, dict):
                continue
            text = str(token.get("text", "") or "")
            is_final = bool(token.get("is_final"))
            if not is_final:
                continue
            if text in ("<fin>", "<end>"):
                logger.debug(
                    "[STT] Soniox token finalize pending_tokens=%s", len(self._pending_tokens)
                )
                self._flush_final()
                continue
            end_ms = token.get("end_ms")
            if isinstance(end_ms, (int, float)):
                end_ms = int(end_ms)
                if self._pending_last_end_ms is not None and end_ms <= self._pending_last_end_ms:
                    logger.debug(
                        "[STT] Soniox token skipped end_ms=%s last_end_ms=%s",
                        end_ms,
                        self._pending_last_end_ms,
                    )
                    continue
                self._pending_last_end_ms = end_ms
            preview = text if len(text) <= 80 else f"{text[:80]}..."
            logger.debug(
                "[STT] Soniox token final text=%r end_ms=%s pending_tokens=%s",
                preview,
                end_ms,
                len(self._pending_tokens) + 1,
            )
            self._pending_tokens.append(_FinalToken(text=text, end_ms=end_ms))

    def _flush_final(self) -> None:
        if not self._pending_tokens:
            if self._consume_pending_finalize_request():
                self._final_tokens.clear()
                self._emit_empty_final_ack()
            return
        updated = self._merge_pending_tokens()
        self._pending_tokens.clear()
        self._pending_last_end_ms = None
        if not updated:
            if self._consume_pending_finalize_request():
                if not self._emit_final_text():
                    self._emit_empty_final_ack()
                self._final_tokens.clear()
            return
        emitted = self._emit_final_text()
        if self._consume_pending_finalize_request():
            self._final_tokens.clear()
            if not emitted:
                self._emit_empty_final_ack()

    def _consume_pending_finalize_request(self) -> bool:
        if self._pending_finalize_requests <= 0:
            return False
        self._pending_finalize_requests -= 1
        return True

    def _merge_pending_tokens(self) -> bool:
        new_tokens = self._pending_tokens
        if not new_tokens:
            return False
        if not self._final_tokens:
            self._final_tokens = list(new_tokens)
            return True

        new_max = self._max_end_ms(new_tokens)
        existing_max = self._max_end_ms(self._final_tokens)
        if new_max is not None and existing_max is not None and new_max < existing_max:
            logger.debug(
                "[STT] Soniox final batch out-of-order max_end_ms=%s existing_max_end_ms=%s",
                new_max,
                existing_max,
            )
            return False

        new_first = self._min_end_ms(new_tokens)
        if new_first is None:
            self._final_tokens.extend(new_tokens)
            return True

        cut_idx = None
        for idx, token in enumerate(self._final_tokens):
            if token.end_ms is None:
                continue
            if token.end_ms >= new_first:
                cut_idx = idx
                break

        logger.debug(
            "[STT] Soniox final merge cut_idx=%s new_first_end_ms=%s new_max_end_ms=%s existing_max_end_ms=%s",
            cut_idx,
            new_first,
            new_max,
            existing_max,
        )
        if cut_idx is None:
            self._final_tokens.extend(new_tokens)
        elif cut_idx == 0:
            self._final_tokens = list(new_tokens)
        else:
            self._final_tokens = self._final_tokens[:cut_idx] + list(new_tokens)
        return True

    def _emit_final_text(self) -> bool:
        if not self._final_tokens:
            return False
        text = "".join(token.text for token in self._final_tokens).strip()
        if not text:
            return False
        # 문두 문장부호+공백 패턴 제거 (이전 발화의 잔여 문장부호 방어)
        # 예: ". 안녕" -> "안녕", "? 다음" -> "다음"
        text = re.sub(r"^[.,:;!?。，；：！？]+\s+", "", text)
        if not text:
            return False
        logger.info("[STT] Transcript: '%s' (final)", text)
        logger.debug(
            "[STT] Soniox final flush tokens=%s text_len=%s",
            len(self._final_tokens),
            len(text),
        )
        self._put_event(STTBackendTranscriptEvent(text=text, is_final=True))
        return True

    def _emit_empty_final_ack(self) -> None:
        logger.debug("[STT] Soniox empty finalize ack")
        self._put_event(STTBackendTranscriptEvent(text="", is_final=True))

    def _min_end_ms(self, tokens: Sequence[_FinalToken]) -> int | None:
        values = [token.end_ms for token in tokens if token.end_ms is not None]
        if not values:
            return None
        return min(values)

    def _max_end_ms(self, tokens: Sequence[_FinalToken]) -> int | None:
        values = [token.end_ms for token in tokens if token.end_ms is not None]
        if not values:
            return None
        return max(values)

    def _put_event(self, event: STTBackendTranscriptEvent | BaseException | None) -> None:
        self._events.put_nowait(event)

    async def send_audio(self, pcm16le: bytes) -> None:
        if self._stopped:
            return
        await self._audio_q.put(pcm16le)

    async def on_speech_end(self, *, trailing_silence_ms: int | None = None) -> None:
        if self._stopped:
            return

        self._pending_finalize_requests += 1
        silence_ms = max(int(self.trailing_silence_ms), 0)
        if trailing_silence_ms is None and silence_ms > 0:
            silence_samples = int(self.sample_rate_hz * (silence_ms / 1000.0))
            if silence_samples > 0:
                import numpy as np

                silence = np.zeros(silence_samples, dtype=np.float32)
                pcm16 = (silence * 32767).astype(np.int16).tobytes()
                await self._audio_q.put(pcm16)
                logger.info(
                    "[STT] Trailing silence sent (%sms, %s samples, %s bytes)",
                    silence_ms,
                    silence_samples,
                    len(pcm16),
                )

        await self._audio_q.put(
            _FinalizeRequest(trailing_silence_ms=silence_ms if silence_ms > 0 else None)
        )

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
