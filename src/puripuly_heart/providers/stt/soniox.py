"""Soniox Realtime STT Backend using WebSocket API.

Uses raw WebSocket streaming with manual finalize and keepalive control messages.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, ClassVar, Literal, Sequence
from uuid import uuid4

from websockets.exceptions import ProtocolError

from puripuly_heart.core import network_clients
from puripuly_heart.core.audio.format import AudioCaptureSpan
from puripuly_heart.core.speech_boundary import SpeechBoundaryReason
from puripuly_heart.core.stt.backend import (
    LEGACY_STT_SESSION_PROJECTION,
    PermanentSTTScopedSessionError,
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
from puripuly_heart.core.stt.diagnostics import recognition_cause
from puripuly_heart.core.stt.session_projection import STTSessionEventProjection
from puripuly_heart.domain.models import FinalLanguageRun, FinalSpeakerRun

logger = logging.getLogger(__name__)

_STOP = object()
_SELECTIVE_PADDING_MS = 200
_SELECTIVE_PAUSE_MIN_MS = 4000
_SELECTIVE_PAUSE_MAX_MS = 7000
_MAX_TURN_FINAL_TOKENS = 16384
SONIOX_MAX_SESSION_AGE_S = 299.0 * 60.0
_RETRYABLE_SERVER_ERRORS = {
    408: "request_timeout",
    413: "max_duration_reached",
    429: "limit_exceeded",
    500: "internal_error",
    503: "service_unavailable",
}
_RETRYABLE_TRANSPORT_REASONS = frozenset(
    {
        "soniox_write_failed",
        "soniox_receive_failed",
        "soniox_keepalive_failed",
        "soniox_connection_ended",
        "soniox_stream_finished",
    }
)
_RETRYABLE_WS_CLOSE_CODES = frozenset({1000, 1001, 1006, 1011, 1012, 1013})

_SAFE_EXCEPTION_TYPES = frozenset(
    {
        "BrokenPipeError",
        "ConnectionClosed",
        "ConnectionClosedError",
        "ConnectionClosedOK",
        "ConnectionError",
        "ConnectionResetError",
        "JSONDecodeError",
        "OSError",
        "RuntimeError",
        "TimeoutError",
        "TypeError",
        "UnicodeDecodeError",
        "ValueError",
    }
)


def _safe_exception_type(exc: BaseException | None) -> str:
    if exc is None:
        return "none"
    name = type(exc).__name__
    return name if name in _SAFE_EXCEPTION_TYPES else "unclassified"


def _safe_code(value: object, *, maximum: int) -> int | str:
    if isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= maximum:
        return value
    if isinstance(value, str) and 0 < len(value) <= 6 and value.isascii() and value.isdecimal():
        number = int(value)
        if number <= maximum:
            return number
    return "unclassified"


def _retryable_server_error(data: dict[str, Any], code: int | str) -> bool:
    expected_type = _RETRYABLE_SERVER_ERRORS.get(code) if isinstance(code, int) else None
    if expected_type is None:
        return False
    error_type = data.get("error_type")
    error = data.get("error")
    return ("error_type" not in data or error_type == expected_type) and (
        "error" not in data or isinstance(error, str)
    )


def _retryable_transport_failure(reason: str, exception: BaseException | None, ws: Any) -> bool:
    if reason not in _RETRYABLE_TRANSPORT_REASONS:
        return False
    if isinstance(exception, ProtocolError):
        return False
    close = getattr(exception, "rcvd", None)
    code = getattr(close, "code", None)
    if code is None:
        code = getattr(exception, "code", None)
    if code is None and ws is not None:
        code = getattr(ws, "close_code", None)
    return code is None or _safe_code(code, maximum=65535) in _RETRYABLE_WS_CLOSE_CODES


def _elapsed_ms(now: float, then: float | None) -> int | str:
    return max(0, int((now - then) * 1000)) if then is not None else "none"


@dataclass(frozen=True, slots=True)
class _FinalizeRequest:
    completion: asyncio.Future[None] | None = None
    padding_pcm16le: bytes = b""
    identity: STTProviderTurnIdentity | None = None
    turn: _TurnDiagnostics | None = None


@dataclass(slots=True)
class _TurnDiagnostics:
    identity: STTProviderTurnIdentity
    channel: Literal["self", "peer"]
    finalize_requested_at: float | None = None
    finalize_written_at: float | None = None
    fin_received: bool = False
    fin_accepted: bool = False
    logged: bool = False


@dataclass(frozen=True, slots=True)
class _AudioWrite:
    pcm16le: bytes
    completion: asyncio.Future[None]


@dataclass(frozen=True, slots=True)
class _FinalToken:
    text: str
    start_ms: int | None
    end_ms: int | None
    language: str = ""
    speaker_id: str | None = None
    attribution_state: str = "missing"


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
            raise PermanentSTTScopedSessionError("sample_rate_hz must be 8000 or 16000")
        if not self.api_key:
            raise PermanentSTTScopedSessionError("api_key must be non-empty")
        if not self.endpoint:
            raise PermanentSTTScopedSessionError("endpoint must be non-empty")
        if self.keepalive_interval_s <= 0:
            raise PermanentSTTScopedSessionError("keepalive_interval_s must be > 0")
        if self.trailing_silence_ms < 0:
            raise PermanentSTTScopedSessionError("trailing_silence_ms must be >= 0")
        if self.connect_timeout_s <= 0:
            raise PermanentSTTScopedSessionError("connect_timeout_s must be > 0")
        if self.language_hints_strict and not self.language_hints:
            raise PermanentSTTScopedSessionError("language_hints_strict requires language_hints")

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


        async def _check() -> bool:
            try:
                async with network_clients.external_websocket_connect(endpoint, ping_interval=None, open_timeout=5) as ws:
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
    max_session_age_s: ClassVar[float] = SONIOX_MAX_SESSION_AGE_S

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
    _turn_diagnostics: _TurnDiagnostics | None = field(init=False, default=None, repr=False)
    _session_open_at: float | None = field(init=False, default=None, repr=False)
    _last_rx_at: float | None = field(init=False, default=None, repr=False)
    _session_fault_logged: bool = field(init=False, default=False, repr=False)
    _local_cleanup_started: bool = field(init=False, default=False, repr=False)

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

        try:
            self._ws = await network_clients.external_websocket_connect(self.endpoint, ping_interval=None, open_timeout=self.connect_timeout_s)
        except websockets.exceptions.InvalidStatus as exc:
            status = _safe_code(exc.response.status_code, maximum=999999)
            if status in _RETRYABLE_SERVER_ERRORS:
                raise
            raise PermanentSTTScopedSessionError("soniox_connection_rejected") from exc
        except (
            websockets.exceptions.InvalidHandshake,
            websockets.exceptions.InvalidURI,
            ValueError,
        ) as exc:
            raise PermanentSTTScopedSessionError("soniox_connection_rejected") from exc
        self._session_open_at = time.monotonic()
        try:
            configuration = json.dumps(config)
        except (TypeError, ValueError) as exc:
            raise PermanentSTTScopedSessionError("soniox_configuration_invalid") from exc
        try:
            await self._ws.send(configuration)
        except Exception as exc:
            if not _retryable_transport_failure("soniox_write_failed", exc, self._ws):
                raise PermanentSTTScopedSessionError("soniox_connection_rejected") from exc
            raise
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
                    if (
                        data.turn is not None
                        and data.turn is self._turn_diagnostics
                        and data.identity == data.turn.identity
                    ):
                        if data.turn.finalize_requested_at is None:
                            data.turn.finalize_requested_at = time.monotonic()
                    async with self._send_lock:
                        if data.padding_pcm16le:
                            await self._ws.send(data.padding_pcm16le)
                        payload = {"type": "finalize"}
                        await self._ws.send(json.dumps(payload))
                        written_at = time.monotonic()
                        self._last_send_at = written_at
                    if (
                        data.turn is not None
                        and data.turn is self._turn_diagnostics
                        and data.identity == data.turn.identity
                    ):
                        data.turn.finalize_written_at = written_at
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
            logger.error("Soniox send loop error exception_type=%s", _safe_exception_type(exc))
            self._put_event(exc)
            self._scoped_transport_failure("soniox_write_failed", orderly=False, exception=exc)
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
            logger.error("Soniox recv loop error exception_type=%s", _safe_exception_type(exc))
            self._put_event(exc)
            self._scoped_transport_failure("soniox_receive_failed", orderly=False, exception=exc)
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
            logger.debug("Soniox keepalive failed cause=%s", _safe_exception_type(exc))
            self._put_event(exc)
            self._scoped_transport_failure("soniox_keepalive_failed", orderly=False, exception=exc)
            self._stopped = True

    def _handle_message(self, message: str | bytes) -> None:
        self._last_rx_at = time.monotonic()
        if isinstance(message, bytes):
            message = message.decode("utf-8", errors="ignore")
        try:
            data = json.loads(message)
        except json.JSONDecodeError, UnicodeDecodeError:
            if self._event_projection.is_scoped:
                self._scoped_transport_failure(
                    "soniox_protocol_ambiguity", orderly=False, protocol_detail="invalid_json"
                )
            return
        if not isinstance(data, dict):
            if self._event_projection.is_scoped:
                self._scoped_transport_failure(
                    "soniox_protocol_ambiguity",
                    orderly=False,
                    protocol_detail="invalid_message_shape",
                )
            return

        if "error" in data or "error_code" in data or "error_type" in data:
            self._put_event(RuntimeError("Soniox request failed"))
            server_error_code = (
                _safe_code(data.get("error_code"), maximum=999999)
                if "error_code" in data
                else "none"
            )
            self._scoped_transport_failure(
                "soniox_request_failed",
                orderly=False,
                server_error_code=server_error_code,
                failure_retryable=_retryable_server_error(data, server_error_code),
            )
            return

        tokens = data.get("tokens", [])
        if tokens is None:
            tokens = []
        if not isinstance(tokens, list):
            if self._event_projection.is_scoped:
                self._scoped_transport_failure(
                    "soniox_protocol_ambiguity",
                    orderly=False,
                    protocol_detail="invalid_tokens_shape",
                )
            return

        for token in tokens:
            if not isinstance(token, dict):
                if self._event_projection.is_scoped:
                    self._scoped_transport_failure(
                        "soniox_protocol_ambiguity",
                        orderly=False,
                        protocol_detail="invalid_token_shape",
                    )
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
            attribution_state = "identified" if speaker_id is not None else "missing"
            if self.enable_speaker_diarization and "speaker" in token and speaker_id is None:
                attribution_state = "malformed"
            final_token = _FinalToken(
                text=text,
                start_ms=start_ms,
                end_ms=end_ms,
                language=language,
                speaker_id=speaker_id,
                attribution_state=attribution_state,
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
                    source="soniox",
                    attribution_state=final_token.attribution_state,
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
        diagnostic = self._turn_diagnostics
        if diagnostic is not None:
            diagnostic.fin_received = True
        identity = self._event_projection.active_identity
        if identity is None:
            detail = "fin_without_turn"
        elif not self._event_projection.sealed:
            detail = "fin_before_seal"
        elif self._pending_finalize_requests != 1:
            detail = "fin_pending_count_mismatch"
        else:
            detail = None
        if detail is not None:
            self._scoped_transport_failure(
                "soniox_protocol_ambiguity", orderly=False, protocol_detail=detail
            )
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
        if diagnostic is not None:
            diagnostic.fin_accepted = True
        self._log_scoped_summary(trigger="fin")
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
            if (
                runs
                and runs[-1].speaker_id == token.speaker_id
                and runs[-1].attribution_state == token.attribution_state
            ):
                previous = runs[-1]
                runs[-1] = FinalSpeakerRun(
                    text=previous.text + token.text,
                    speaker_id=token.speaker_id,
                    session_scope=self.speaker_session_scope,
                    source_start_ms=previous.source_start_ms,
                    source_end_ms=token.end_ms,
                    overlaps_previous=previous.overlaps_previous or overlaps_previous,
                    source="soniox",
                    attribution_state=previous.attribution_state,
                )
            else:
                runs.append(
                    FinalSpeakerRun(
                        text=token.text,
                        speaker_id=token.speaker_id,
                        session_scope=self.speaker_session_scope,
                        source_start_ms=token.start_ms,
                        source_end_ms=token.end_ms,
                        overlaps_previous=overlaps_previous,
                        source="soniox",
                        attribution_state=token.attribution_state,
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
        self._turn_diagnostics = None

    def _log_scoped_summary(
        self,
        *,
        trigger: str,
        reason: str = "none",
        protocol_detail: str = "none",
        server_error_code: int | str = "none",
        exception: BaseException | None = None,
        idle_fin_received: bool = False,
    ) -> None:
        if not self._event_projection.is_scoped:
            return
        turn = self._turn_diagnostics
        if turn is None:
            if trigger in ("abort", "local_stop", "local_close"):
                return
            if self._session_fault_logged or self._local_cleanup_started:
                return
            self._session_fault_logged = True
        elif turn.logged:
            return
        else:
            turn.logged = True
        now = time.monotonic()
        requested_at = turn.finalize_requested_at if turn is not None else None
        written_at = turn.finalize_written_at if turn is not None else None
        close = getattr(exception, "rcvd", None)
        code = getattr(close, "code", None)
        if code is None:
            code = getattr(exception, "code", None)
        if code is None and self._ws is not None:
            code = getattr(self._ws, "close_code", None)
        logger.info(
            "[Soniox] %s channel=%s utterance_id=%s epoch=%s turn=%s "
            "trigger=%s reason=%s session_age_ms=%s diarization=%s "
            "finalize_requested=%s finalize_written=%s finalize_queue_ms=%s "
            "since_finalize_requested_ms=%s since_finalize_written_ms=%s "
            "fin_received=%s fin_accepted=%s protocol_detail=%s "
            "last_rx_age_ms=%s server_error_code=%s ws_close_code=%s exception_type=%s",
            "turn_end" if turn is not None else "session_fault",
            turn.channel if turn is not None else "none",
            turn.identity.segment.segment_id if turn is not None else "none",
            (
                turn.identity.provider_epoch_id
                if turn is not None
                else self._event_projection.provider_epoch_id
            ),
            turn.identity.provider_turn_id if turn is not None else "none",
            trigger,
            reason,
            _elapsed_ms(now, self._session_open_at),
            str(self.enable_speaker_diarization).lower(),
            str(requested_at is not None).lower(),
            str(written_at is not None).lower(),
            _elapsed_ms(written_at, requested_at) if written_at is not None else "none",
            _elapsed_ms(now, requested_at),
            _elapsed_ms(now, written_at),
            str(turn.fin_received if turn is not None else idle_fin_received).lower(),
            str(turn.fin_accepted if turn is not None else False).lower(),
            protocol_detail,
            _elapsed_ms(now, self._last_rx_at),
            server_error_code,
            _safe_code(code, maximum=65535) if code is not None else "none",
            _safe_exception_type(exception),
        )

    def _scoped_transport_failure(
        self,
        reason: str,
        *,
        orderly: bool,
        protocol_detail: str = "none",
        server_error_code: int | str = "none",
        exception: BaseException | None = None,
        failure_retryable: bool | None = None,
    ) -> None:
        if failure_retryable is None:
            failure_retryable = _retryable_transport_failure(reason, exception, self._ws)
        identity = self._event_projection.active_identity
        if identity is None and self._event_projection.retired:
            return
        trigger = (
            "provider_error"
            if reason == "soniox_request_failed"
            else (
                "protocol_error"
                if reason
                in (
                    "soniox_protocol_ambiguity",
                    "soniox_idle_authoritative_text",
                    "soniox_token_buffer_overflow",
                )
                else "transport_error"
            )
        )
        self._log_scoped_summary(
            trigger=trigger,
            reason=reason,
            protocol_detail=protocol_detail,
            server_error_code=server_error_code,
            exception=exception,
            idle_fin_received=protocol_detail == "fin_without_turn",
        )
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
                    failure_retryable=failure_retryable,
                    epoch_disposition="retire",
                    provenance=tuple(self._scoped_provenance),
                )
            )
        self._clear_scoped_turn()
        self._event_projection.end_epoch(
            orderly=orderly,
            reason=reason,
            provider_turn_id=provider_turn_id,
            failure_retryable=failure_retryable,
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
                        language=token.language,
                        speaker_id=token.speaker_id,
                        attribution_state=token.attribution_state,
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
        self._turn_diagnostics = _TurnDiagnostics(request.identity, request.channel)

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
        turn = self._turn_diagnostics
        await self._audio_q.put(_FinalizeRequest(completion, padding_pcm16le, identity, turn))
        if turn is not None and turn is self._turn_diagnostics:
            if turn.finalize_requested_at is None:
                turn.finalize_requested_at = time.monotonic()
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
        self._log_scoped_summary(trigger="abort", reason=recognition_cause(reason))
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
        self._log_scoped_summary(trigger="local_stop")
        self._local_cleanup_started = True
        self._stopped = True
        await self._audio_q.put(_STOP)

    async def close(self) -> None:
        self._log_scoped_summary(trigger="local_close")
        self._local_cleanup_started = True
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
