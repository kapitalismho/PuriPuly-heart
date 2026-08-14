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

from puripuly_heart.core.speech_boundary import SpeechBoundaryReason, boundary_wait_ms
from puripuly_heart.core.stt.backend import (
    STTBackend,
    STTBackendSession,
    STTBackendTranscriptEvent,
)
from puripuly_heart.domain.models import FinalLanguageRun

logger = logging.getLogger(__name__)

_STOP = object()


@dataclass(frozen=True, slots=True)
class _FinalizeRequest:
    pass


@dataclass(frozen=True, slots=True)
class _FinalToken:
    text: str
    end_ms: int | None
    language: str = ""


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
    enable_language_identification: bool = False
    language_hints_strict: bool = False
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
        if self.language_hints_strict and not self.language_hints:
            raise ValueError("language_hints_strict requires language_hints")

        session = _SonioxSession(
            api_key=self.api_key,
            model=self.model,
            endpoint=self.endpoint,
            sample_rate_hz=self.sample_rate_hz,
            language_hints=list(self.language_hints),
            context_terms=list(self.context_terms),
            keepalive_interval_s=self.keepalive_interval_s,
            trailing_silence_ms=self.trailing_silence_ms,
            enable_language_identification=self.enable_language_identification,
            language_hints_strict=self.language_hints_strict,
            connect_timeout_s=self.connect_timeout_s,
        )
        try:
            await session.start()
        except BaseException:
            with contextlib.suppress(BaseException):
                await session.close()
            raise
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
    enable_language_identification: bool = False
    language_hints_strict: bool = False

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
            "enable_language_identification": self.enable_language_identification,
        }
        if self.language_hints:
            config["language_hints"] = self.language_hints
            if self.language_hints_strict:
                config["language_hints_strict"] = True
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
                    await self._ws.send("")
                    self._last_send_at = time.monotonic()
                    return
                if isinstance(data, _FinalizeRequest):
                    payload = {"type": "finalize"}
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
            self._put_event(RuntimeError("Soniox request failed"))
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
                        "[STT] Soniox token timestamp non-increasing end_ms=%s last_end_ms=%s",
                        end_ms,
                        self._pending_last_end_ms,
                    )
                self._pending_last_end_ms = end_ms
            logger.debug(
                "[STT] Soniox token final text_len=%s end_ms=%s pending_tokens=%s",
                len(text),
                end_ms,
                len(self._pending_tokens) + 1,
            )
            language = ""
            if self.enable_language_identification:
                raw_language = token.get("language")
                if isinstance(raw_language, str):
                    language = raw_language.strip().lower()
            self._pending_tokens.append(_FinalToken(text=text, end_ms=end_ms, language=language))

    def _flush_final(self) -> None:
        if not self._consume_pending_finalize_request():
            logger.debug(
                "[STT] Soniox finalize marker retained without pending request tokens=%s",
                len(self._pending_tokens),
            )
            return
        self._final_tokens = list(self._pending_tokens)
        self._pending_tokens.clear()
        self._pending_last_end_ms = None
        if not self._emit_final_text():
            self._emit_empty_final_ack()

    def _consume_pending_finalize_request(self) -> bool:
        if self._pending_finalize_requests <= 0:
            return False
        self._pending_finalize_requests -= 1
        return True

    def _emit_final_text(self) -> bool:
        if not self._final_tokens:
            return False
        self._final_tokens = self._normalized_final_tokens()
        text = "".join(token.text for token in self._final_tokens)
        if not text:
            return False
        logger.info("[STT] Transcript final text_len=%s", len(text))
        logger.debug(
            "[STT] Soniox final flush tokens=%s text_len=%s",
            len(self._final_tokens),
            len(text),
        )
        self._put_event(
            STTBackendTranscriptEvent(
                text=text,
                is_final=True,
                final_language_runs=self._final_language_runs(),
            )
        )
        return True

    def _normalized_final_tokens(self) -> list[_FinalToken]:
        source = "".join(token.text for token in self._final_tokens)
        start = len(source) - len(source.lstrip())
        end = len(source.rstrip())
        if start >= end:
            return []

        normalized: list[_FinalToken] = []
        offset = 0
        for token in self._final_tokens:
            token_end = offset + len(token.text)
            overlap_start = max(start, offset)
            overlap_end = min(end, token_end)
            if overlap_start < overlap_end:
                normalized.append(
                    _FinalToken(
                        text=token.text[overlap_start - offset : overlap_end - offset],
                        end_ms=token.end_ms,
                        language=token.language,
                    )
                )
            offset = token_end
        return normalized

    def _final_language_runs(self) -> tuple[FinalLanguageRun, ...]:
        if not self.enable_language_identification:
            return ()
        runs: list[FinalLanguageRun] = []
        for token in self._final_tokens:
            if runs and runs[-1].language == token.language:
                previous = runs[-1]
                runs[-1] = FinalLanguageRun(
                    text=previous.text + token.text,
                    language=previous.language,
                )
            else:
                runs.append(FinalLanguageRun(text=token.text, language=token.language))
        return tuple(runs)

    def _emit_empty_final_ack(self) -> None:
        logger.debug("[STT] Soniox empty finalize ack")
        self._put_event(STTBackendTranscriptEvent(text="", is_final=True))

    def _put_event(self, event: STTBackendTranscriptEvent | BaseException | None) -> None:
        self._events.put_nowait(event)

    async def send_audio(self, pcm16le: bytes) -> None:
        if self._stopped:
            return
        await self._audio_q.put(pcm16le)

    async def on_speech_end(
        self,
        *,
        trailing_silence_ms: int | None = None,
        reason: SpeechBoundaryReason | None = None,
    ) -> None:
        if self._stopped:
            return

        self._pending_finalize_requests += 1
        observed_tail_ms = max(int(trailing_silence_ms or 0), 0)
        wait_ms = boundary_wait_ms(reason, observed_tail_ms=observed_tail_ms)
        logger.info(
            "[STT][Tail] provider=soniox boundary_reason=%s observed_tail_ms=%s "
            "boundary_wait_ms=%s",
            reason,
            observed_tail_ms,
            wait_ms,
        )
        await self._audio_q.put(_FinalizeRequest())

    async def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
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
