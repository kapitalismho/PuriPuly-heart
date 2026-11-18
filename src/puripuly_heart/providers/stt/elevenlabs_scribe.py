from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from typing import Any, AsyncIterator, ClassVar, Sequence
from urllib.parse import urlencode

from websockets.asyncio.client import ClientConnection
from websockets.exceptions import ConnectionClosedOK

from puripuly_heart.config.provider_values import ELEVENLABS_SCRIBE_STT_MODEL
from puripuly_heart.core import network_clients
from puripuly_heart.core.audio.format import AudioCaptureSpan
from puripuly_heart.core.diagnostics import current_diagnostic_operation, emit_failure
from puripuly_heart.core.messages import ErrorDiagnostics
from puripuly_heart.core.speech_boundary import SpeechBoundaryReason
from puripuly_heart.core.stt.backend import (
    LEGACY_STT_SESSION_PROJECTION,
    STTBackend,
    STTBackendSession,
    STTBackendTranscriptEvent,
    STTFinalSpeakerEvidence,
    STTFinalSpeakerRevision,
    STTNativeProvenance,
    STTNativeSegmentRef,
    STTNativeSpeakerMerge,
    STTProviderTurnIdentity,
    STTProviderTurnRequest,
    STTProviderTurnTerminal,
    STTProviderTurnUpdate,
    STTSessionProjection,
)
from puripuly_heart.core.stt.diagnostics import (
    recognition_cleanup_failure,
    recognition_failure_diagnostics,
    retryable_transport_failure,
)
from puripuly_heart.core.stt.session_projection import STTSessionEventProjection
from puripuly_heart.domain.models import FinalSpeakerRun

logger = logging.getLogger(__name__)

ELEVENLABS_SCRIBE_SAMPLE_RATE_HZ = 16000
ELEVENLABS_SCRIBE_VERIFY_CONNECT_TIMEOUT_S = 10.0
ELEVENLABS_SCRIBE_VERIFY_SETTLE_TIMEOUT_S = 10.0
ELEVENLABS_SCRIBE_KEEPALIVE_INTERVAL_S = 10.0
MAX_SCRIBE_KEYTERMS = 50
MAX_SCRIBE_KEYTERM_CHARS = 20

_SCRIBE_URL = "wss://api.elevenlabs.io/v1/speech-to-text/realtime"
_SCRIBE_SOURCE = "elevenlabs_scribe"
_MAX_UNCOMMITTED_BYTES = 30 * ELEVENLABS_SCRIBE_SAMPLE_RATE_HZ * 2
_MAX_RETAINED_COMMITS = 32
_MAX_SPEAKER_IDS = 4096
_MAX_SPEAKER_CHARS = 128
_MAX_SPEAKER_RUNS = 256
_MAX_COMMIT_CHARS = 65536
_SCRIBE_ERRORS = frozenset(
    {
        "auth_error",
        "chunk_size_exceeded",
        "error",
        "input_error",
        "insufficient_audio_activity",
        "invalid_request",
        "queue_overflow",
        "quota_exceeded",
        "rate_limited",
        "resource_exhausted",
        "server_error",
        "session_time_limit_exceeded",
        "transcriber_error",
    }
)
ScribeWebSocketFactory = Callable[..., Awaitable[ClientConnection]]
_RECOVERABLE_SCRIBE_EVENTS = frozenset(
    {
        "session_time_limit_exceeded",
        "insufficient_audio_activity",
    }
)


def scribe_keyterms(terms: Sequence[str]) -> tuple[str, ...]:
    input_terms = list(terms)
    normalized: list[str] = []
    truncated_count = 0
    for term in input_terms:
        raw = str(term).strip()
        if len(raw) > MAX_SCRIBE_KEYTERM_CHARS:
            truncated_count += 1
        candidate = raw[:MAX_SCRIBE_KEYTERM_CHARS]
        if not candidate or candidate in normalized:
            continue
        normalized.append(candidate)
        if len(normalized) >= MAX_SCRIBE_KEYTERMS:
            break
    if (
        len(normalized)
        < len(
            {
                str(term).strip()[:MAX_SCRIBE_KEYTERM_CHARS]
                for term in input_terms
                if str(term).strip()
            }
        )
        or truncated_count
    ):
        logger.debug(
            "Scribe keyterms normalized input=%s kept=%s truncated_chars=%s",
            len(input_terms),
            len(normalized),
            truncated_count,
        )
    return tuple(normalized)


async def _connect_scribe(
    api_key: str,
    *,
    language: str | None = None,
    keyterms: tuple[str, ...] = (),
    diarize: bool = False,
    websocket_factory: ScribeWebSocketFactory | None = None,
) -> ClientConnection:
    options = {
        "model_id": ELEVENLABS_SCRIBE_STT_MODEL,
        "audio_format": "pcm_16000",
        "commit_strategy": "manual",
        "diarize": "true" if diarize else "false",
        "include_timestamps": "true" if diarize else "false",
        "filter_background_audio": "false",
    }
    if language:
        options["language_code"] = language
    elif diarize:
        options["multilang_mode"] = "true"
    query = urlencode((*options.items(), *(("keyterms", term) for term in keyterms)))
    connect = websocket_factory or network_clients.external_websocket_connect
    return await connect(f"{_SCRIBE_URL}?{query}", additional_headers={"xi-api-key": api_key})


def _read_frame(raw: str | bytes) -> dict[str, Any] | None:
    try:
        frame = json.loads(raw)
    except ValueError, RecursionError:
        return None
    return frame if isinstance(frame, dict) else None


async def _read_session_id(connection: ClientConnection) -> str:
    try:
        frame = _read_frame(await connection.recv())
    except ConnectionClosedOK as exc:
        raise RuntimeError("Scribe connection closed before session start") from exc
    if frame is not None and frame.get("message_type") == "session_started":
        session_id = frame.get("session_id")
        if isinstance(session_id, str) and session_id.strip() and len(session_id) <= 256:
            return session_id
    event_name = frame.get("message_type") if frame is not None else None
    reason = "scribe_session_started_missing_id"
    if isinstance(event_name, str) and event_name in _SCRIBE_ERRORS:
        reason = f"scribe_{event_name}"
    failure = RuntimeError(f"Scribe realtime readiness failed: {reason}")
    failure.failure_diagnostics = recognition_failure_diagnostics(
        None, provider=_SCRIBE_SOURCE, channel=None, operation="readiness", reason=reason
    )
    raise failure


@dataclass(slots=True)
class _Commit:
    identity: STTProviderTurnIdentity | None
    index: int
    sent: bool = False
    raw_text: str | None = None
    observed_at: float | None = None
    speaker_runs: tuple[FinalSpeakerRun, ...] | None = None


async def verify_scribe_realtime_connection(
    api_key: str,
    *,
    websocket_factory: ScribeWebSocketFactory | None = None,
    connect_timeout_s: float = ELEVENLABS_SCRIBE_VERIFY_CONNECT_TIMEOUT_S,
    settle_timeout_s: float = ELEVENLABS_SCRIBE_VERIFY_SETTLE_TIMEOUT_S,
) -> bool:
    """Verify an API key by opening a Scribe realtime session.

    Account-level probes (for example ``GET /v1/user/subscription``) reject
    scope-restricted keys that are still valid for transcription, so
    verification performs the same realtime handshake production sessions use.
    Returns True once the server starts a session. Raises with the server
    reason on auth/permission failures and on connection problems.
    """
    if not api_key:
        return False
    try:
        connection = await asyncio.wait_for(
            _connect_scribe(api_key, websocket_factory=websocket_factory),
            timeout=connect_timeout_s,
        )
    except (asyncio.TimeoutError, TimeoutError) as exc:
        raise TimeoutError(
            "Scribe realtime verification timed out while connecting "
            f"after {connect_timeout_s:.0f}s"
        ) from exc
    try:
        primary_failure = None
        try:
            await asyncio.wait_for(_read_session_id(connection), timeout=settle_timeout_s)
        except (asyncio.TimeoutError, TimeoutError) as exc:
            raise TimeoutError(
                "Scribe realtime verification timed out waiting for session start "
                f"after {settle_timeout_s:.0f}s"
            ) from exc
        return True
    except Exception as exc:
        primary_failure = recognition_failure_diagnostics(
            exc,
            provider="elevenlabs_scribe",
            channel=None,
            operation="probe",
            reason="provider_not_ready",
        )
        primary_failure = replace(
            primary_failure,
            **current_diagnostic_operation(),
            fields={
                **primary_failure.fields,
                "verification_scope": "key_probe",
                "recognition_readiness": "unobserved",
            },
        )
        exc.failure_diagnostics = primary_failure
        emit_failure(logger, primary_failure, stage="probe")
        raise
    finally:
        try:
            await connection.close()
        except Exception as exc:
            diagnostics = recognition_failure_diagnostics(
                exc,
                provider="elevenlabs_scribe",
                channel=None,
                operation="cleanup",
                reason="provider_cleanup_failed",
            )
            emit_failure(
                logger,
                replace(
                    diagnostics,
                    **current_diagnostic_operation(),
                    cause_id=primary_failure.diagnostic_id if primary_failure else None,
                    fields={**diagnostics.fields, "cleanup_confirmed": False},
                ),
                severity="warning",
                disposition="degraded",
                stage="cleanup",
            )


def scribe_language_code(source_language: str | None) -> str | None:
    if not source_language:
        return None
    normalized = source_language.strip()
    if not normalized or normalized.lower() == "auto":
        return None
    base = normalized.split("-")[0].lower()
    return base or None


@dataclass(slots=True)
class ElevenLabsScribeSTTBackend(STTBackend):
    """ElevenLabs Scribe v3 Live recognition backend."""

    api_key: str
    language_code: str | None = None
    keyterms: Sequence[str] = ()
    model: ClassVar[str] = ELEVENLABS_SCRIBE_STT_MODEL
    sample_rate_hz: int = ELEVENLABS_SCRIBE_SAMPLE_RATE_HZ
    connect_timeout_s: float = 10.0
    keepalive_interval_s: float = ELEVENLABS_SCRIBE_KEEPALIVE_INTERVAL_S
    diarize: bool = False
    websocket_factory: ScribeWebSocketFactory | None = None
    _starting_sessions: list[STTBackendSession] = field(
        init=False, default_factory=list, repr=False
    )

    async def open_session(
        self,
        *,
        projection: STTSessionProjection = LEGACY_STT_SESSION_PROJECTION,
    ) -> STTBackendSession:
        if self.sample_rate_hz != ELEVENLABS_SCRIBE_SAMPLE_RATE_HZ:
            raise ValueError("sample_rate_hz must be 16000 for Scribe realtime transcription")
        if not self.api_key:
            raise ValueError("api_key must be non-empty")
        if self.connect_timeout_s <= 0:
            raise ValueError("connect_timeout_s must be > 0")
        if self.keepalive_interval_s <= 0:
            raise ValueError("keepalive_interval_s must be > 0")

        if self._starting_sessions:
            await self.close()
        session = _ElevenLabsScribeSession(
            api_key=self.api_key,
            language_code=scribe_language_code(self.language_code),
            keyterms=scribe_keyterms(self.keyterms),
            sample_rate_hz=self.sample_rate_hz,
            connect_timeout_s=self.connect_timeout_s,
            keepalive_interval_s=self.keepalive_interval_s,
            diarize=self.diarize,
            websocket_factory=self.websocket_factory,
            projection=projection,
        )
        try:
            await session.start()
        except BaseException as primary:
            diagnostics = recognition_failure_diagnostics(
                primary,
                provider="elevenlabs_scribe",
                channel=None,
                operation="readiness",
                reason="provider_not_ready",
            )
            primary.failure_diagnostics = diagnostics
            session._failure_diagnostics = diagnostics
            try:
                await session.close()
            except Exception as cleanup:
                self._starting_sessions.append(session)
                primary.failure_diagnostics = replace(
                    diagnostics,
                    causes=(*diagnostics.causes, cleanup.failure_diagnostics),
                )
            raise
        return session

    async def close(self) -> None:
        for session in tuple(self._starting_sessions):
            await session.close()
            self._starting_sessions.remove(session)

    @staticmethod
    async def verify_api_key(
        api_key: str,
        *,
        websocket_factory: ScribeWebSocketFactory | None = None,
        connect_timeout_s: float = ELEVENLABS_SCRIBE_VERIFY_CONNECT_TIMEOUT_S,
        settle_timeout_s: float = ELEVENLABS_SCRIBE_VERIFY_SETTLE_TIMEOUT_S,
    ) -> bool:
        if not api_key:
            return False
        return await verify_scribe_realtime_connection(
            api_key,
            websocket_factory=websocket_factory,
            connect_timeout_s=connect_timeout_s,
            settle_timeout_s=settle_timeout_s,
        )


@dataclass(slots=True)
class _ElevenLabsScribeSession(STTBackendSession):
    """Owns one native Scribe websocket and its manual commit slot."""

    api_key: str
    language_code: str | None
    keyterms: tuple[str, ...]
    sample_rate_hz: int
    connect_timeout_s: float
    keepalive_interval_s: float = ELEVENLABS_SCRIBE_KEEPALIVE_INTERVAL_S
    diarize: bool = False
    websocket_factory: ScribeWebSocketFactory | None = None
    projection: STTSessionProjection = LEGACY_STT_SESSION_PROJECTION

    _event_projection: STTSessionEventProjection = field(init=False, repr=False)
    _connection: ClientConnection | None = field(init=False, default=None, repr=False)
    _receive_task: asyncio.Task[None] | None = field(init=False, default=None, repr=False)
    _keepalive_task: asyncio.Task[None] | None = field(init=False, default=None, repr=False)
    _stopped: bool = field(init=False, default=False)
    _last_send_at: float = field(init=False, default=0.0, repr=False)
    _session_id: str = field(init=False, default="", repr=False)
    _next_index: int = field(init=False, default=0, repr=False)
    _uncommitted_bytes: int = field(init=False, default=0, repr=False)
    _commit: _Commit | None = field(init=False, default=None, repr=False)
    _committed: OrderedDict[int, _Commit] = field(
        init=False, default_factory=OrderedDict, repr=False
    )
    _aliases: dict[str, str] = field(init=False, default_factory=dict, repr=False)
    _speaker_evidence_enabled: bool = field(init=False, default=True, repr=False)
    _send_lock: asyncio.Lock = field(init=False, default_factory=asyncio.Lock, repr=False)
    _failure_diagnostics: ErrorDiagnostics | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        self._event_projection = STTSessionEventProjection(self.projection)

    async def start(self) -> None:
        self._connection = await asyncio.wait_for(
            _connect_scribe(
                self.api_key,
                language=self.language_code,
                keyterms=self.keyterms,
                diarize=self.diarize,
                websocket_factory=self.websocket_factory,
            ),
            timeout=self.connect_timeout_s,
        )
        self._session_id = await asyncio.wait_for(
            _read_session_id(self._connection), timeout=self.connect_timeout_s
        )
        self._last_send_at = time.monotonic()
        self._receive_task = asyncio.create_task(self._receive())
        self._keepalive_task = asyncio.create_task(self._keepalive_loop())

    async def _receive(self) -> None:
        connection = self._connection
        assert connection is not None
        try:
            while not self._stopped:
                frame = _read_frame(await connection.recv())
                if self._stopped:
                    return
                if frame is None:
                    continue
                event = frame.get("message_type")
                if event == "partial_transcript":
                    self._on_partial(frame)
                elif event == "committed_transcript":
                    self._on_committed(frame)
                elif self.diarize and event == "diarized_transcript":
                    self._on_diarized(frame)
                elif self.diarize and event == "speakers_merged":
                    self._on_merge(frame)
                elif isinstance(event, str) and event in _SCRIBE_ERRORS:
                    self._on_error_event(event)
                elif event == "session_started":
                    self._protocol_failure("scribe_duplicate_session_started")
        except ConnectionClosedOK:
            self._scoped_transport_failure("scribe_connection_closed", orderly=True)
            self._end_connection_stream()
        except Exception as exc:
            self._scoped_transport_failure("scribe_connection_ended", orderly=False, exception=exc)
            self._end_connection_stream(exc)

    def _on_partial(self, data: dict[str, Any]) -> None:
        text = data.get("text")
        if not isinstance(text, str) or len(text) > _MAX_COMMIT_CHARS:
            return
        identity = self._event_projection.active_identity
        if identity is None:
            return
        sequence = self._event_projection.next_update_sequence(identity)
        if sequence is None:
            return
        self._event_projection.put_update(
            STTProviderTurnUpdate(
                identity=identity,
                sequence=sequence,
                stability="provisional",
                assembly="replace",
                text=text,
            )
        )

    def _on_committed(self, data: dict[str, Any]) -> None:
        index = data.get("segment_index")
        if self.diarize or index is not None:
            if type(index) is not int or index < 0:
                self._protocol_failure("scribe_invalid_segment_index")
                return
            if index < self._next_index:
                return
            if index != self._next_index:
                self._protocol_failure("scribe_skipped_segment_index")
                return
        commit = self._commit
        if commit is None or (
            commit.identity is not None
            and not (
                self._event_projection.is_current(commit.identity) and self._event_projection.sealed
            )
        ):
            self._protocol_failure("scribe_unsolicited_committed_transcript")
            return
        if commit.raw_text is not None:
            if index == commit.index:
                return
            self._protocol_failure("scribe_duplicate_committed_transcript")
            return
        text = data.get("text")
        if not isinstance(text, str) or len(text) > _MAX_COMMIT_CHARS:
            reason = (
                "scribe_committed_transcript_missing_text"
                if "text" not in data
                else "scribe_invalid_committed_transcript"
            )
            self._protocol_failure(reason)
            return
        commit.raw_text = text
        commit.observed_at = time.monotonic()
        self._finish_committed(commit)

    def _finish_committed(self, commit: _Commit) -> None:
        if (
            self._stopped
            or self._commit is not commit
            or not commit.sent
            or commit.raw_text is None
        ):
            return
        identity = commit.identity
        self._next_index += 1
        self._uncommitted_bytes = 0
        if identity is None:
            accepted = self._event_projection.put_legacy(
                STTBackendTranscriptEvent(text=commit.raw_text, is_final=True)
            )
        else:
            text = commit.raw_text.strip()
            native_segment = (
                STTNativeSegmentRef(self._session_id, commit.index) if self.diarize else None
            )
            if self.diarize and text and commit.speaker_runs is None:
                runs = (self._speaker_run(text, None),)
            else:
                runs = commit.speaker_runs or ()
            accepted = self._event_projection.terminal(
                STTProviderTurnTerminal(
                    identity=identity,
                    outcome="final" if text else "empty",
                    text=text,
                    final_speaker_runs=runs,
                    text_authority="authoritative",
                    provenance=(
                        STTNativeProvenance(
                            native_item_id=str(commit.index) if native_segment else None,
                            native_task_id=self._session_id,
                            barrier="committed_transcript",
                        ),
                    ),
                    native_segment=native_segment,
                    observed_at=commit.observed_at,
                )
            )
            if accepted and native_segment is not None and text and self._speaker_evidence_enabled:
                self._committed[commit.index] = commit
                if len(self._committed) > _MAX_RETAINED_COMMITS:
                    self._committed.popitem(last=False)
        self._commit = None
        if not accepted:
            self._protocol_failure("scribe_connection_event_overflow")

    def _canonical(self, speaker_id: str) -> str:
        while speaker_id in self._aliases:
            speaker_id = self._aliases[speaker_id]
        return speaker_id

    def _speaker_run(self, text: str, speaker_id: str | None) -> FinalSpeakerRun:
        return FinalSpeakerRun(
            text,
            speaker_id,
            self._session_id,
            source=_SCRIBE_SOURCE,
            attribution_state="identified" if speaker_id is not None else "missing",
        )

    @staticmethod
    def _valid_speaker_id(value: object) -> bool:
        return (
            isinstance(value, str) and 0 < len(value) <= _MAX_SPEAKER_CHARS and bool(value.strip())
        )

    def _on_diarized(self, data: dict[str, Any]) -> None:
        index = data.get("segment_index")
        if not self._speaker_evidence_enabled or type(index) is not int or index < 0:
            return
        pending = self._commit
        commit = (
            pending
            if pending is not None and pending.index == index
            else self._committed.get(index)
        )
        if commit is None or not commit.raw_text or commit.speaker_runs is not None:
            return
        words = data.get("words")
        if data.get("text") != commit.raw_text or not isinstance(words, list) or not words:
            return
        raw = commit.raw_text
        start = len(raw) - len(raw.lstrip())
        end = len(raw.rstrip())
        runs: list[FinalSpeakerRun] = []
        parts: list[str] = []
        current_id: str | None = None
        offset = 0
        for word in words:
            if not isinstance(word, dict):
                return
            text = word.get("text")
            speaker_id = word.get("speaker_id")
            if (
                not isinstance(text, str)
                or not raw.startswith(text, offset)
                or (speaker_id is not None and not self._valid_speaker_id(speaker_id))
            ):
                return
            stop = offset + len(text)
            left, right = max(start, offset), min(end, stop)
            offset = stop
            if left >= right:
                continue
            if speaker_id is not None:
                speaker_id = self._canonical(speaker_id)
            if parts and current_id != speaker_id:
                runs.append(self._speaker_run("".join(parts), current_id))
                if len(runs) == _MAX_SPEAKER_RUNS:
                    return
                parts.clear()
            current_id = speaker_id
            parts.append(raw[left:right])
        if offset != len(raw) or not parts:
            return
        runs.append(self._speaker_run("".join(parts), current_id))
        commit.speaker_runs = tuple(runs)
        if commit is not pending:
            self._emit_speaker_evidence((commit,))

    def _on_merge(self, data: dict[str, Any]) -> None:
        origin = data.get("from_speaker_id")
        target = data.get("to_speaker_id")
        if (
            not self._speaker_evidence_enabled
            or not self._valid_speaker_id(origin)
            or not self._valid_speaker_id(target)
        ):
            return
        origin = self._canonical(origin)
        target = self._canonical(target)
        if origin == target:
            return
        if len(self._aliases) == _MAX_SPEAKER_IDS:
            self._speaker_evidence_enabled = False
            self._committed.clear()
            self._aliases.clear()
            return
        self._aliases[origin] = target
        revised = []
        pending = self._commit
        for commit in (*self._committed.values(), pending):
            if commit is None or not commit.speaker_runs:
                continue
            if not any(run.speaker_id == origin for run in commit.speaker_runs):
                continue
            commit.speaker_runs = tuple(
                replace(run, speaker_id=target) if run.speaker_id == origin else run
                for run in commit.speaker_runs
            )
            if commit is not pending:
                revised.append(commit)
        self._emit_speaker_evidence(
            tuple(revised), speaker_merge=STTNativeSpeakerMerge(self._session_id, origin, target)
        )

    def _emit_speaker_evidence(
        self, commits: tuple[_Commit, ...], *, speaker_merge: STTNativeSpeakerMerge | None = None
    ) -> None:
        epoch = self._event_projection.provider_epoch_id
        if epoch is None or (not commits and speaker_merge is None):
            return
        self._event_projection.put_speaker_evidence(
            STTFinalSpeakerEvidence(
                epoch,
                tuple(
                    STTFinalSpeakerRevision(
                        commit.identity,
                        STTNativeSegmentRef(self._session_id, commit.index),
                        commit.speaker_runs,
                    )
                    for commit in commits
                    if commit.identity is not None and commit.speaker_runs is not None
                ),
                time.monotonic(),
                is_current=lambda: not self._stopped,
                speaker_merge=speaker_merge,
            )
        )

    def _protocol_failure(self, reason: str) -> None:
        self._scoped_transport_failure(reason, orderly=False)
        self._end_connection_stream(RuntimeError(reason))

    def _on_error_event(self, event_name: str) -> None:
        logger.warning("[STT] Scribe provider event %s", event_name)
        self._scoped_transport_failure(f"scribe_{event_name}", orderly=False)
        self._end_connection_stream(
            None
            if event_name in _RECOVERABLE_SCRIBE_EVENTS
            else RuntimeError(f"scribe_{event_name}")
        )

    def _end_connection_stream(self, legacy_failure: Exception | None = None) -> None:
        if self._stopped:
            return
        self._stopped = True
        self._commit = None
        self._committed.clear()
        self._aliases.clear()
        if self._event_projection.is_legacy:
            self._event_projection.put_legacy(legacy_failure)

    def _scoped_transport_failure(
        self, reason: str, *, orderly: bool, exception: BaseException | None = None
    ) -> None:
        identity = self._event_projection.active_identity
        provider_turn_id = identity.provider_turn_id if identity is not None else None
        diagnostics = recognition_failure_diagnostics(
            exception,
            provider="elevenlabs_scribe",
            channel=self._event_projection.channel or "unknown",
            operation="receive",
            reason=reason,
            identity=identity,
        )
        if self._failure_diagnostics is None and not orderly:
            self._failure_diagnostics = diagnostics
        retryable = reason in {
            "scribe_session_time_limit_exceeded",
            "scribe_insufficient_audio_activity",
            "scribe_connection_closed",
            "scribe_connection_ended",
            "scribe_rate_limited",
            "scribe_resource_exhausted",
            "scribe_server_error",
        } or retryable_transport_failure(exception)
        if identity is not None:
            self._event_projection.terminal(
                STTProviderTurnTerminal(
                    identity=identity,
                    outcome="failed",
                    text_authority="none",
                    failure_reason=reason,
                    epoch_disposition="retire",
                    failure_retryable=retryable,
                    failure_diagnostics=diagnostics,
                )
            )
        self._event_projection.end_epoch(
            orderly=orderly,
            reason=reason,
            provider_turn_id=provider_turn_id,
            failure_retryable=retryable,
            failure_diagnostics=diagnostics,
        )

    async def _send(self, pcm16le: bytes, *, commit: _Commit | None = None) -> None:
        async with self._send_lock:
            if self._stopped or self._connection is None:
                raise RuntimeError("Scribe session is closed")
            if len(pcm16le) + self._uncommitted_bytes > _MAX_UNCOMMITTED_BYTES:
                self._protocol_failure("scribe_manual_commit_audio_limit")
                raise RuntimeError("Scribe manual commit audio limit")
            if commit is not None:
                if self._commit is not None:
                    self._protocol_failure("scribe_overlapping_manual_commits")
                    raise RuntimeError("Scribe already has an outstanding manual commit")
                self._commit = commit
            await self._connection.send(
                json.dumps(
                    {
                        "message_type": "input_audio_chunk",
                        "audio_base_64": base64.b64encode(pcm16le).decode("ascii"),
                        "sample_rate": self.sample_rate_hz,
                        "commit": commit is not None,
                    },
                    separators=(",", ":"),
                )
            )
            self._uncommitted_bytes += len(pcm16le)
            self._last_send_at = time.monotonic()

    async def _keepalive_loop(self) -> None:
        interval_s = self.keepalive_interval_s
        try:
            while not self._stopped:
                await asyncio.sleep(interval_s)
                if self._stopped or self._connection is None:
                    return
                if time.monotonic() - self._last_send_at < interval_s:
                    continue
                try:
                    await self._send(b"")
                except Exception as exc:
                    self._scoped_transport_failure(
                        "scribe_keepalive_failed", orderly=False, exception=exc
                    )
                    self._end_connection_stream()
                    return
        except asyncio.CancelledError:
            raise

    async def begin_turn(self, request: STTProviderTurnRequest) -> None:
        if self._stopped or self._connection is None:
            raise RuntimeError("Scribe session is closed")
        self._event_projection.begin(request)

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
        connection = self._connection
        if self._stopped or connection is None:
            raise RuntimeError("Scribe session is closed")
        try:
            await self._send(pcm16le)
        except Exception as exc:
            self._scoped_transport_failure("scribe_write_failed", orderly=False, exception=exc)
            self._end_connection_stream()
            raise
        if (
            self._stopped
            or self._connection is not connection
            or not self._event_projection.is_current(identity)
        ):
            raise RuntimeError("Scribe turn lost authority during audio write")
        self._event_projection.payload_written(identity, payload_sequence)

    async def seal_turn(
        self,
        identity: STTProviderTurnIdentity,
        *,
        sealed_content_ranges: tuple[AudioCaptureSpan, ...],
        seal_reason: str,
        observed_trailing_silence_ms: int | None,
    ) -> None:
        self._event_projection.require_open(identity)
        _ = sealed_content_ranges, seal_reason, observed_trailing_silence_ms
        connection = self._connection
        if self._stopped or connection is None:
            raise RuntimeError("Scribe session is closed")
        self._event_projection.seal(identity)
        await self._send_commit(identity)

    async def _send_commit(self, identity: STTProviderTurnIdentity | None) -> None:
        commit = _Commit(identity, self._next_index)
        try:
            await self._send(b"", commit=commit)
        except Exception as exc:
            self._scoped_transport_failure("scribe_commit_failed", orderly=False, exception=exc)
            self._end_connection_stream()
            raise
        commit.sent = True
        self._finish_committed(commit)

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
        self._event_projection.end_epoch(
            orderly=False,
            reason=reason,
            provider_turn_id=identity.provider_turn_id,
        )
        self._end_connection_stream()

    async def turn_events(self):
        async for event in self._event_projection.turn_events():
            yield event

    async def send_audio(self, pcm16le: bytes) -> None:
        if self._stopped or self._connection is None:
            return
        try:
            await self._send(pcm16le)
        except Exception as exc:
            self._scoped_transport_failure("scribe_write_failed", orderly=False, exception=exc)
            self._end_connection_stream()
            return

    async def on_speech_end(
        self,
        *,
        trailing_silence_ms: int | None = None,
        reason: SpeechBoundaryReason | None = None,
    ) -> None:
        if self._stopped:
            return
        if self._connection is None:
            return
        try:
            await self._send_commit(None)
        except Exception:
            return

    async def stop(self) -> None:
        if self._stopped:
            return
        self._scoped_transport_failure("scribe_connection_ended", orderly=True)
        self._end_connection_stream()

    async def close(self) -> None:
        await self.stop()
        tasks = tuple(
            task for task in (self._receive_task, self._keepalive_task) if task is not None
        )
        self._receive_task = None
        self._keepalive_task = None
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if self._connection is not None:
            try:
                await self._connection.close()
            except Exception as exc:
                diagnostics = recognition_cleanup_failure(
                    exc,
                    provider="elevenlabs_scribe",
                    channel=self._event_projection.channel,
                    primary=self._failure_diagnostics,
                )
                emit_failure(
                    logger, diagnostics, severity="warning", disposition="degraded", stage="cleanup"
                )
                raise
            self._connection = None
        self._event_projection.close()

    async def events(self) -> AsyncIterator[STTBackendTranscriptEvent]:
        async for event in self._event_projection.events():
            yield event
