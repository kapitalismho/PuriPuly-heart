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

from puripuly_heart.core.audio.format import AudioCaptureSpan
from puripuly_heart.core.speech_boundary import SpeechBoundaryReason, boundary_wait_ms
from puripuly_heart.core.stt.backend import (
    LEGACY_STT_SESSION_PROJECTION,
    STTBackend,
    STTBackendSession,
    STTBackendTranscriptEvent,
    STTNativeProvenance,
    STTProviderTurnIdentity,
    STTProviderTurnRequest,
    STTProviderTurnTerminal,
    STTProviderTurnUpdate,
    STTSessionProjection,
)
from puripuly_heart.core.stt.session_projection import STTSessionEventProjection
from puripuly_heart.domain.models import FinalLanguageRun

logger = logging.getLogger(__name__)

_STOP = object()


@dataclass(frozen=True, slots=True)
class _FinalizeRequest:
    completion: asyncio.Future[None] | None = None


@dataclass(frozen=True, slots=True)
class _AudioWrite:
    pcm16le: bytes
    completion: asyncio.Future[None]


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

    async def open_session(
        self,
        *,
        projection: STTSessionProjection = LEGACY_STT_SESSION_PROJECTION,
    ) -> STTBackendSession:
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
            projection=projection,
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
    projection: STTSessionProjection = LEGACY_STT_SESSION_PROJECTION

    _event_projection: STTSessionEventProjection = field(init=False, repr=False)
    _audio_q: asyncio.Queue[bytes | _AudioWrite | object] = field(init=False, repr=False)
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
    _scoped_provenance: list[STTNativeProvenance] = field(
        init=False, default_factory=list, repr=False
    )
    _scoped_tokens: list[_FinalToken] = field(init=False, default_factory=list, repr=False)

    def __post_init__(self) -> None:
        self._event_projection = STTSessionEventProjection(self.projection)
        self._audio_q = asyncio.Queue(maxsize=258)

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
        data: bytes | _AudioWrite | object = _STOP
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
                    self._resolve_write(data.completion, None)
                    continue
                if isinstance(data, _AudioWrite):
                    await self._ws.send(data.pcm16le)
                    self._last_send_at = time.monotonic()
                    self._resolve_write(data.completion, None)
                    continue
                if isinstance(data, bytes):
                    await self._ws.send(data)
                    self._last_send_at = time.monotonic()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._resolve_write(getattr(data, "completion", None), exc)
            logger.exception("Soniox send loop error")
            self._put_event(exc)
            self._scoped_transport_failure("soniox_write_failed", orderly=False)
        finally:
            self._fail_pending_writes()

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
            self._scoped_transport_failure("soniox_receive_failed", orderly=False)
        finally:
            self._stopped = True
            self._put_event(None)
            self._scoped_transport_failure("soniox_connection_ended", orderly=True)

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
            self._put_event(exc)
            self._scoped_transport_failure("soniox_keepalive_failed", orderly=False)
            self._stopped = True

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
            self._scoped_transport_failure("soniox_request_failed", orderly=False)
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
            if text == "<fin>":
                logger.debug(
                    "[STT] Soniox token finalize pending_tokens=%s", len(self._pending_tokens)
                )
                self._flush_final()
                self._resolve_scoped_fin(data)
                continue
            if text == "<end>":
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
            final_token = _FinalToken(text=text, end_ms=end_ms, language=language)
            self._pending_tokens.append(final_token)
            self._emit_scoped_token(final_token, token, data)

    def _emit_scoped_token(
        self,
        final_token: _FinalToken,
        token: dict[str, Any],
        message: dict[str, Any],
    ) -> None:
        identity = self._event_projection.active_identity
        if identity is None:
            return
        request_id = message.get("request_id")
        provenance = STTNativeProvenance(
            native_request_id=str(request_id) if request_id is not None else None,
        )
        self._scoped_tokens.append(final_token)
        self._scoped_provenance.append(provenance)
        sequence = self._event_projection.next_update_sequence(identity)
        if sequence is None:
            return
        runs = ()
        if self.enable_language_identification:
            runs = (FinalLanguageRun(text=final_token.text, language=final_token.language),)
        self._event_projection.put_update(
            STTProviderTurnUpdate(
                identity=identity,
                sequence=sequence,
                stability="stable",
                assembly="append",
                text=final_token.text,
                final_language_runs=runs,
                provenance=provenance,
            )
        )

    def _resolve_scoped_fin(self, message: dict[str, Any]) -> None:
        identity = self._event_projection.active_identity
        if identity is None or not self._event_projection.sealed:
            return
        request_id = message.get("request_id")
        provenance = STTNativeProvenance(
            native_request_id=str(request_id) if request_id is not None else None,
            barrier="manual_finalize",
        )
        self._scoped_provenance.append(provenance)
        text = "".join(token.text for token in self._scoped_tokens)
        runs = self._language_runs_for_tokens(self._scoped_tokens)
        self._event_projection.terminal(
            STTProviderTurnTerminal(
                identity=identity,
                outcome="final" if text else "empty",
                text=text,
                final_language_runs=runs,
                text_authority="authoritative",
                epoch_disposition="retire",
                provenance=tuple(self._scoped_provenance),
            )
        )
        self._clear_scoped_turn()

    def _language_runs_for_tokens(
        self,
        tokens: list[_FinalToken],
    ) -> tuple[FinalLanguageRun, ...]:
        if not self.enable_language_identification:
            return ()
        runs: list[FinalLanguageRun] = []
        for token in tokens:
            if runs and runs[-1].language == token.language:
                previous = runs[-1]
                runs[-1] = FinalLanguageRun(
                    text=previous.text + token.text,
                    language=previous.language,
                )
            else:
                runs.append(FinalLanguageRun(text=token.text, language=token.language))
        return tuple(runs)

    def _clear_scoped_turn(self) -> None:
        self._scoped_provenance.clear()
        self._scoped_tokens.clear()

    def _scoped_transport_failure(self, reason: str, *, orderly: bool) -> None:
        identity = self._event_projection.active_identity
        provider_turn_id = identity.provider_turn_id if identity is not None else None
        if identity is not None:
            text = "".join(token.text for token in self._scoped_tokens)
            self._event_projection.terminal(
                STTProviderTurnTerminal(
                    identity=identity,
                    outcome="degraded" if text else "failed",
                    text=text,
                    final_language_runs=self._language_runs_for_tokens(self._scoped_tokens),
                    text_authority="degraded" if text else "none",
                    failure_reason=reason,
                    epoch_disposition="retire",
                    provenance=tuple(self._scoped_provenance),
                )
            )
            self._clear_scoped_turn()
        self._event_projection.end_epoch(
            orderly=orderly,
            reason=reason,
            provider_turn_id=provider_turn_id,
        )

    @staticmethod
    def _resolve_write(
        completion: asyncio.Future[None] | None,
        error: BaseException | None,
    ) -> None:
        if completion is None or completion.done():
            return
        if error is None:
            completion.set_result(None)
        else:
            completion.set_exception(error)

    def _fail_pending_writes(self) -> None:
        error = RuntimeError("Soniox writer stopped")
        while not self._audio_q.empty():
            try:
                item = self._audio_q.get_nowait()
            except asyncio.QueueEmpty:
                return
            self._resolve_write(getattr(item, "completion", None), error)

    def _flush_final(self) -> None:
        if not self._consume_pending_finalize_request():
            logger.debug(
                "[STT] Soniox finalize marker retained without pending request tokens=%s",
                len(self._pending_tokens),
            )
            return
        if self._event_projection.is_scoped:
            self._pending_tokens.clear()
            self._pending_last_end_ms = None
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
        self._event_projection.put_legacy(event)

    async def begin_turn(self, request: STTProviderTurnRequest) -> None:
        if self._stopped or self._ws is None:
            raise RuntimeError("Soniox session is closed")
        self._event_projection.begin(request)
        self._scoped_provenance.clear()
        self._scoped_tokens.clear()

    async def send_turn_audio(
        self,
        identity: STTProviderTurnIdentity,
        pcm16le: bytes,
        *,
        payload_sequence: int,
        source_ranges: tuple[AudioCaptureSpan, ...],
        context_only: bool,
    ) -> None:
        self._event_projection.validate_payload(identity, payload_sequence)
        _ = source_ranges, context_only
        completion = asyncio.get_running_loop().create_future()
        await self._audio_q.put(_AudioWrite(pcm16le, completion))
        await completion
        self._event_projection.payload_written(identity, payload_sequence)

    async def seal_turn(
        self,
        identity: STTProviderTurnIdentity,
        *,
        sealed_content_ranges: tuple[AudioCaptureSpan, ...],
        seal_reason: str,
        observed_trailing_silence_ms: int | None,
    ) -> None:
        self._event_projection.seal(identity)
        _ = sealed_content_ranges, seal_reason, observed_trailing_silence_ms
        self._pending_finalize_requests += 1
        completion = asyncio.get_running_loop().create_future()
        await self._audio_q.put(_FinalizeRequest(completion))
        await completion

    async def abort_turn(self, identity: STTProviderTurnIdentity, *, reason: str) -> None:
        self._event_projection.require_open(identity)
        self._event_projection.terminal(
            STTProviderTurnTerminal(
                identity=identity,
                outcome="cancelled",
                text_authority="none",
                failure_reason=reason,
                epoch_disposition="retire",
            )
        )
        self._clear_scoped_turn()
        self._event_projection.end_epoch(
            orderly=False,
            reason=reason,
            provider_turn_id=identity.provider_turn_id,
        )

    async def turn_events(self):
        async for event in self._event_projection.turn_events():
            yield event

    async def send_audio(self, pcm16le: bytes) -> None:
        if self._stopped:
            return
        if self._audio_q.qsize() >= 256:
            raise RuntimeError("Soniox audio queue overflow")
        self._audio_q.put_nowait(pcm16le)

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
        self._event_projection.close()

    async def events(self) -> AsyncIterator[STTBackendTranscriptEvent]:
        async for event in self._event_projection.events():
            yield event


import contextlib  # placed at bottom to keep the main logic compact
