"""Soniox Realtime STT Backend using WebSocket API.

Uses raw WebSocket streaming with manual finalize and keepalive control messages.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Literal, Sequence
from uuid import uuid4

from puripuly_heart.core.audio.format import AudioCaptureSpan
from puripuly_heart.core.speech_boundary import SpeechBoundaryReason
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
from puripuly_heart.domain.models import FinalLanguageRun, FinalSpeakerRun

logger = logging.getLogger(__name__)

_STOP = object()
_SELECTIVE_PADDING_MS = 200
_SELECTIVE_PAUSE_MIN_MS = 4000
_SELECTIVE_PAUSE_MAX_MS = 7000
_MAX_TURN_FINAL_TOKENS = 16384


@dataclass(frozen=True, slots=True)
class _FinalizeRequest:
    completion: asyncio.Future[None] | None = None
    padding_pcm16le: bytes = b""


@dataclass(frozen=True, slots=True)
class _AudioWrite:
    pcm16le: bytes
    completion: asyncio.Future[None]


@dataclass(frozen=True, slots=True)
class _FinalToken:
    text: str
    start_ms: int | None
    end_ms: int | None
    confidence: float | None = None
    language: str = ""
    speaker_id: str | None = None


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
    enable_speaker_diarization: bool = True
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
            enable_speaker_diarization=self.enable_speaker_diarization,
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
                raise Exception("Connection failed") from exc

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
    enable_speaker_diarization: bool = True
    language_hints_strict: bool = False
    projection: STTSessionProjection = LEGACY_STT_SESSION_PROJECTION

    speaker_session_scope: str = field(init=False, default_factory=lambda: uuid4().hex)
    _event_projection: STTSessionEventProjection = field(init=False, repr=False)
    _audio_q: asyncio.Queue[bytes | _AudioWrite | object] = field(init=False, repr=False)
    _ws: Any = field(init=False, default=None, repr=False)
    _send_task: asyncio.Task[None] | None = field(init=False, default=None, repr=False)
    _recv_task: asyncio.Task[None] | None = field(init=False, default=None, repr=False)
    _keepalive_task: asyncio.Task[None] | None = field(init=False, default=None, repr=False)
    _send_lock: asyncio.Lock = field(init=False, default_factory=asyncio.Lock, repr=False)
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
    _scoped_channel: Literal["self", "peer"] | None = field(init=False, default=None, repr=False)

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
            "enable_speaker_diarization": self.enable_speaker_diarization,
        }
        if self.language_hints:
            config["language_hints"] = self.language_hints
            if self.language_hints_strict:
                config["language_hints_strict"] = True
        if self.context_terms:
            config["context"] = {"terms": self.context_terms}

        self._ws = await websockets.connect(
            self.endpoint, ping_interval=None, open_timeout=self.connect_timeout_s
        )
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
                    async with self._send_lock:
                        await self._ws.send("")
                        self._last_send_at = time.monotonic()
                    return
                if isinstance(data, _FinalizeRequest):
                    async with self._send_lock:
                        if data.padding_pcm16le:
                            await self._ws.send(data.padding_pcm16le)
                        payload = {"type": "finalize"}
                        await self._ws.send(json.dumps(payload))
                        self._last_send_at = time.monotonic()
                    self._resolve_write(data.completion, None)
                    continue
                if isinstance(data, _AudioWrite):
                    async with self._send_lock:
                        await self._ws.send(data.pcm16le)
                        self._last_send_at = time.monotonic()
                    self._resolve_write(data.completion, None)
                    continue
                if isinstance(data, bytes):
                    async with self._send_lock:
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
                    async with self._send_lock:
                        await self._ws.send(json.dumps({"type": "keepalive"}))
                        self._last_send_at = time.monotonic()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.debug("Soniox keepalive failed cause=%s", type(exc).__name__)
            self._put_event(exc)
            self._scoped_transport_failure("soniox_keepalive_failed", orderly=False)
            self._stopped = True

    def _handle_message(self, message: str | bytes) -> None:
        if isinstance(message, bytes):
            message = message.decode("utf-8", errors="ignore")
        try:
            data = json.loads(message)
        except json.JSONDecodeError, UnicodeDecodeError:
            if self._event_projection.is_scoped:
                self._scoped_transport_failure("soniox_protocol_ambiguity", orderly=False)
            return
        if not isinstance(data, dict):
            if self._event_projection.is_scoped:
                self._scoped_transport_failure("soniox_protocol_ambiguity", orderly=False)
            return

        if "error" in data or "error_code" in data:
            self._put_event(RuntimeError("Soniox request failed"))
            self._scoped_transport_failure("soniox_request_failed", orderly=False)
            return

        tokens = data.get("tokens", [])
        if tokens is None:
            tokens = []
        if not isinstance(tokens, list):
            if self._event_projection.is_scoped:
                self._scoped_transport_failure("soniox_protocol_ambiguity", orderly=False)
            return

        for token in tokens:
            if not isinstance(token, dict):
                if self._event_projection.is_scoped:
                    self._scoped_transport_failure("soniox_protocol_ambiguity", orderly=False)
                    return
                continue
            text = str(token.get("text", "") or "")
            is_final = bool(token.get("is_final"))
            if not is_final:
                continue
            if text == "<fin>":
                if self._event_projection.is_scoped:
                    self._resolve_scoped_fin(data)
                    if self._event_projection.retired:
                        return
                else:
                    self._flush_final()
                continue
            if text == "<end>":
                continue
            if self._event_projection.is_scoped and self._event_projection.active_identity is None:
                self._scoped_transport_failure("soniox_idle_authoritative_text", orderly=False)
                return
            if len(self._pending_tokens) >= _MAX_TURN_FINAL_TOKENS:
                if self._event_projection.is_scoped:
                    self._scoped_transport_failure("soniox_token_buffer_overflow", orderly=False)
                else:
                    self._put_event(RuntimeError("Soniox token buffer overflow"))
                    self._pending_tokens.clear()
                    self._final_tokens.clear()
                    self._pending_last_end_ms = None
                    self._pending_finalize_requests = 0
                return
            start_ms = token.get("start_ms")
            if isinstance(start_ms, (int, float)) and not isinstance(start_ms, bool):
                start_ms = int(start_ms)
            else:
                start_ms = None
            end_ms = token.get("end_ms")
            if isinstance(end_ms, (int, float)):
                end_ms = int(end_ms)
                self._pending_last_end_ms = end_ms
            else:
                end_ms = None
            confidence = token.get("confidence")
            if (
                isinstance(confidence, (int, float))
                and not isinstance(confidence, bool)
                and 0.0 <= float(confidence) <= 1.0
            ):
                confidence = float(confidence)
            else:
                confidence = None
            language = ""
            if self.enable_language_identification:
                raw_language = token.get("language")
                if isinstance(raw_language, str):
                    language = raw_language.strip().lower()
            speaker_id = None
            if self.enable_speaker_diarization:
                raw_speaker = token.get("speaker")
                if isinstance(raw_speaker, str | int) and not isinstance(raw_speaker, bool):
                    speaker_id = str(raw_speaker).strip() or None
            final_token = _FinalToken(
                text=text,
                start_ms=start_ms,
                end_ms=end_ms,
                confidence=confidence,
                language=language,
                speaker_id=speaker_id,
            )
            self._pending_tokens.append(final_token)
            self._emit_scoped_token(final_token, token, data)

        if data.get("finished") is True and self._event_projection.is_scoped:
            self._scoped_transport_failure("soniox_stream_finished", orderly=True)
            self._stopped = True

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
        speaker_runs = ()
        if self.enable_speaker_diarization:
            speaker_runs = (
                FinalSpeakerRun(
                    text=final_token.text,
                    speaker_id=final_token.speaker_id,
                    session_scope=self.speaker_session_scope,
                    source_start_ms=final_token.start_ms,
                    source_end_ms=final_token.end_ms,
                    speaker_confidence=final_token.confidence,
                ),
            )
        self._event_projection.put_update(
            STTProviderTurnUpdate(
                identity=identity,
                sequence=sequence,
                stability="stable",
                assembly="append",
                text=final_token.text,
                final_language_runs=runs,
                final_speaker_runs=speaker_runs,
                provenance=provenance,
            )
        )

    def _resolve_scoped_fin(self, message: dict[str, Any]) -> None:
        identity = self._event_projection.active_identity
        if (
            identity is None
            or not self._event_projection.sealed
            or self._pending_finalize_requests != 1
        ):
            self._scoped_transport_failure("soniox_protocol_ambiguity", orderly=False)
            return
        self._pending_finalize_requests -= 1
        request_id = message.get("request_id")
        provenance = STTNativeProvenance(
            native_request_id=str(request_id) if request_id is not None else None,
            barrier="manual_finalize",
        )
        self._scoped_provenance.append(provenance)
        text = "".join(token.text for token in self._scoped_tokens)
        runs = self._language_runs_for_tokens(self._scoped_tokens)
        speaker_runs = self._speaker_runs_for_tokens(self._scoped_tokens)
        self._event_projection.terminal(
            STTProviderTurnTerminal(
                identity=identity,
                outcome="final" if text else "empty",
                text=text,
                final_language_runs=runs,
                final_speaker_runs=speaker_runs,
                text_authority="authoritative",
                epoch_disposition="reuse",
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

    def _speaker_runs_for_tokens(
        self,
        tokens: list[_FinalToken],
    ) -> tuple[FinalSpeakerRun, ...]:
        if not self.enable_speaker_diarization:
            return ()
        runs: list[FinalSpeakerRun] = []
        previous_token_end_ms: int | None = None
        has_previous_token = False
        for token in tokens:
            overlaps_previous = has_previous_token and (
                previous_token_end_ms is None
                or token.start_ms is None
                or token.start_ms < previous_token_end_ms
            )
            if runs and runs[-1].speaker_id == token.speaker_id:
                previous = runs[-1]
                runs[-1] = FinalSpeakerRun(
                    text=previous.text + token.text,
                    speaker_id=token.speaker_id,
                    session_scope=self.speaker_session_scope,
                    source_start_ms=previous.source_start_ms,
                    source_end_ms=token.end_ms,
                    speaker_confidence=(
                        min(previous.speaker_confidence, token.confidence)
                        if previous.speaker_confidence is not None and token.confidence is not None
                        else None
                    ),
                    overlaps_previous=previous.overlaps_previous or overlaps_previous,
                )
            else:
                runs.append(
                    FinalSpeakerRun(
                        text=token.text,
                        speaker_id=token.speaker_id,
                        session_scope=self.speaker_session_scope,
                        source_start_ms=token.start_ms,
                        source_end_ms=token.end_ms,
                        speaker_confidence=token.confidence,
                        overlaps_previous=overlaps_previous,
                    )
                )
            previous_token_end_ms = token.end_ms
            has_previous_token = True
        return tuple(runs)

    def _clear_scoped_turn(self) -> None:
        self._scoped_provenance.clear()
        self._scoped_tokens.clear()
        self._scoped_channel = None
        self._pending_tokens.clear()
        self._final_tokens.clear()
        self._pending_last_end_ms = None
        self._pending_finalize_requests = 0

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
                    final_speaker_runs=self._speaker_runs_for_tokens(self._scoped_tokens),
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
        self._put_event(
            STTBackendTranscriptEvent(
                text=text,
                is_final=True,
                final_language_runs=self._final_language_runs(),
                final_speaker_runs=self._final_speaker_runs(),
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
                        start_ms=token.start_ms,
                        end_ms=token.end_ms,
                        confidence=token.confidence,
                        language=token.language,
                        speaker_id=token.speaker_id,
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

    def _final_speaker_runs(self) -> tuple[FinalSpeakerRun, ...]:
        return self._speaker_runs_for_tokens(self._final_tokens)

    def _emit_empty_final_ack(self) -> None:
        self._put_event(STTBackendTranscriptEvent(text="", is_final=True))

    def _put_event(self, event: STTBackendTranscriptEvent | BaseException | None) -> None:
        self._event_projection.put_legacy(event)

    async def begin_turn(self, request: STTProviderTurnRequest) -> None:
        if self._stopped or self._ws is None:
            raise RuntimeError("Soniox session is closed")
        self._event_projection.begin(request)
        self._clear_scoped_turn()
        self._scoped_channel = request.channel

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
        _ = observed_trailing_silence_ms
        padding_pcm16le = self._selective_padding_pcm16le(
            sealed_content_ranges=sealed_content_ranges,
            seal_reason=seal_reason,
        )
        self._pending_finalize_requests += 1
        completion = asyncio.get_running_loop().create_future()
        await self._audio_q.put(_FinalizeRequest(completion, padding_pcm16le))
        await completion

    def _selective_padding_pcm16le(
        self,
        *,
        sealed_content_ranges: tuple[AudioCaptureSpan, ...],
        seal_reason: str,
    ) -> bytes:
        if self._scoped_channel != "peer":
            return b""
        if seal_reason == "delivery_deadline":
            return bytes(self.sample_rate_hz * _SELECTIVE_PADDING_MS // 1000 * 2)
        if seal_reason != "delivery_pause":
            return b""
        content_samples = sum(span.normalized_sample_count for span in sealed_content_ranges)
        if not (
            self.sample_rate_hz * _SELECTIVE_PAUSE_MIN_MS
            <= content_samples * 1000
            < self.sample_rate_hz * _SELECTIVE_PAUSE_MAX_MS
        ):
            return b""
        return bytes(self.sample_rate_hz * _SELECTIVE_PADDING_MS // 1000 * 2)

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
