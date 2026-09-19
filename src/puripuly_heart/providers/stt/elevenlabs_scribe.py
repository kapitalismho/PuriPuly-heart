"""ElevenLabs Scribe v2 Realtime STT Backend using the official elevenlabs SDK."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Sequence

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

logger = logging.getLogger(__name__)

ELEVENLABS_SCRIBE_STT_MODEL = "scribe_v2_realtime"
ELEVENLABS_SCRIBE_SAMPLE_RATE_HZ = 16000
ELEVENLABS_SCRIBE_VERIFY_CONNECT_TIMEOUT_S = 10.0
ELEVENLABS_SCRIBE_VERIFY_SETTLE_TIMEOUT_S = 10.0
ELEVENLABS_SCRIBE_KEEPALIVE_INTERVAL_S = 10.0
MAX_SCRIBE_KEYTERMS = 50
MAX_SCRIBE_KEYTERM_CHARS = 20

_CLOSED = object()
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


def _scribe_verify_event_detail(data: Any) -> str:
    if isinstance(data, dict):
        name = str(data.get("message_type") or data.get("type") or "error")
        extra = data.get("error") or data.get("message") or ""
    else:
        name = str(getattr(data, "type", "") or "error")
        extra = str(getattr(data, "error", "") or getattr(data, "message", "") or "")
    extra_text = str(extra).strip()
    return name if not extra_text else f"{name}: {extra_text}"


async def verify_scribe_realtime_connection(
    api_key: str,
    *,
    scribe_connect_factory: Callable[[Any], Any] | None = None,
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
    from elevenlabs.realtime import (
        AudioFormat,
        CommitStrategy,
        RealtimeAudioOptions,
        RealtimeEvents,
        ScribeRealtime,
    )

    if not api_key:
        return False
    outcomes: asyncio.Queue[Any] = asyncio.Queue()

    def _on_started(data: Any) -> None:
        _ = data
        outcomes.put_nowait("started")

    def _on_error(data: Any) -> None:
        outcomes.put_nowait(("error", _scribe_verify_event_detail(data)))

    def _on_closed() -> None:
        outcomes.put_nowait(("closed",))

    options = RealtimeAudioOptions(
        model_id=ELEVENLABS_SCRIBE_STT_MODEL,
        audio_format=AudioFormat.PCM_16000,
        sample_rate=ELEVENLABS_SCRIBE_SAMPLE_RATE_HZ,
        commit_strategy=CommitStrategy.MANUAL,
    )
    try:
        if scribe_connect_factory is not None:
            connection = await asyncio.wait_for(
                scribe_connect_factory(options), timeout=connect_timeout_s
            )
        else:
            scribe = ScribeRealtime(api_key=api_key)
            connection = await asyncio.wait_for(scribe.connect(options), timeout=connect_timeout_s)
    except (asyncio.TimeoutError, TimeoutError) as exc:
        raise TimeoutError(
            "Scribe realtime verification timed out while connecting "
            f"after {connect_timeout_s:.0f}s"
        ) from exc
    except Exception as exc:
        raise Exception(f"Scribe realtime verification connection failed: {exc}") from exc
    try:
        connection.on(RealtimeEvents.SESSION_STARTED, _on_started)
        connection.on(RealtimeEvents.ERROR, _on_error)
        connection.on(RealtimeEvents.CLOSE, _on_closed)
        try:
            outcome = await asyncio.wait_for(outcomes.get(), timeout=settle_timeout_s)
        except (asyncio.TimeoutError, TimeoutError) as exc:
            raise TimeoutError(
                "Scribe realtime verification timed out waiting for session start "
                f"after {settle_timeout_s:.0f}s"
            ) from exc
        if outcome == "started":
            return True
        if outcome[0] == "closed":
            raise Exception(
                "Scribe realtime verification failed: connection closed before session start"
            )
        detail = str(outcome[1])
        if "auth" in detail.lower():
            raise Exception(f"unauthorized: Scribe rejected the API key ({detail})")
        raise Exception(f"Scribe realtime verification failed ({detail})")
    finally:
        with contextlib.suppress(Exception):
            result = connection.close()
            if asyncio.iscoroutine(result):
                await result


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
    """ElevenLabs Scribe v2 Realtime STT Backend using the official SDK."""

    api_key: str
    language_code: str | None = None
    keyterms: Sequence[str] = ()
    model: str = ELEVENLABS_SCRIBE_STT_MODEL
    sample_rate_hz: int = ELEVENLABS_SCRIBE_SAMPLE_RATE_HZ
    connect_timeout_s: float = 10.0
    keepalive_interval_s: float = ELEVENLABS_SCRIBE_KEEPALIVE_INTERVAL_S
    scribe_connect_factory: Callable[[Any], Any] | None = None

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

        session = _ElevenLabsScribeSession(
            api_key=self.api_key,
            language_code=scribe_language_code(self.language_code),
            keyterms=scribe_keyterms(self.keyterms),
            model=self.model,
            sample_rate_hz=self.sample_rate_hz,
            connect_timeout_s=self.connect_timeout_s,
            keepalive_interval_s=self.keepalive_interval_s,
            scribe_connect_factory=self.scribe_connect_factory,
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
        api_key: str,
        *,
        scribe_connect_factory: Callable[[Any], Any] | None = None,
        connect_timeout_s: float = ELEVENLABS_SCRIBE_VERIFY_CONNECT_TIMEOUT_S,
        settle_timeout_s: float = ELEVENLABS_SCRIBE_VERIFY_SETTLE_TIMEOUT_S,
    ) -> bool:
        if not api_key:
            return False
        return await verify_scribe_realtime_connection(
            api_key,
            scribe_connect_factory=scribe_connect_factory,
            connect_timeout_s=connect_timeout_s,
            settle_timeout_s=settle_timeout_s,
        )


@dataclass(slots=True)
class _ElevenLabsScribeSession(STTBackendSession):
    """Internal session wrapping an elevenlabs RealtimeConnection."""

    api_key: str
    language_code: str | None
    keyterms: tuple[str, ...]
    model: str
    sample_rate_hz: int
    connect_timeout_s: float
    keepalive_interval_s: float = ELEVENLABS_SCRIBE_KEEPALIVE_INTERVAL_S
    scribe_connect_factory: Callable[[Any], Any] | None = None
    projection: STTSessionProjection = LEGACY_STT_SESSION_PROJECTION

    _event_projection: STTSessionEventProjection = field(init=False, repr=False)
    _connection: Any = field(init=False, default=None, repr=False)
    _queue_task: asyncio.Task[None] | None = field(init=False, default=None, repr=False)
    _keepalive_task: asyncio.Task[None] | None = field(init=False, default=None, repr=False)
    _stopped: bool = field(init=False, default=False)
    _last_send_at: float = field(init=False, default=0.0, repr=False)
    _connection_events: asyncio.Queue[Any] = field(init=False, repr=False)
    _scoped_provenance: list[STTNativeProvenance] = field(
        init=False, default_factory=list, repr=False
    )
    _commit_identity: STTProviderTurnIdentity | None = field(init=False, default=None, repr=False)
    _commit_write_in_flight: bool = field(init=False, default=False, repr=False)
    _pending_committed: tuple[STTProviderTurnIdentity, str, STTNativeProvenance] | None = field(
        init=False, default=None, repr=False
    )

    def __post_init__(self) -> None:
        self._event_projection = STTSessionEventProjection(self.projection)
        self._connection_events = asyncio.Queue(maxsize=258)

    async def start(self) -> None:
        from elevenlabs.realtime import (
            AudioFormat,
            CommitStrategy,
            RealtimeAudioOptions,
            RealtimeEvents,
            ScribeRealtime,
        )

        options = RealtimeAudioOptions(
            model_id=self.model,
            audio_format=AudioFormat.PCM_16000,
            sample_rate=self.sample_rate_hz,
            commit_strategy=CommitStrategy.MANUAL,
            language_code=self.language_code,
            keyterms=list(self.keyterms) if self.keyterms else None,
        )
        if self.scribe_connect_factory is not None:
            connection = await asyncio.wait_for(
                self.scribe_connect_factory(options), timeout=self.connect_timeout_s
            )
        else:
            scribe = ScribeRealtime(api_key=self.api_key)
            connection = await asyncio.wait_for(
                scribe.connect(options), timeout=self.connect_timeout_s
            )
        self._connection = connection

        self._connection.on(RealtimeEvents.PARTIAL_TRANSCRIPT, self._on_partial)
        self._connection.on(RealtimeEvents.FINAL_TRANSCRIPT, self._on_partial)
        self._connection.on(RealtimeEvents.FINAL_TRANSCRIPT_WITH_TIMESTAMPS, self._on_partial)
        self._connection.on(RealtimeEvents.COMMITTED_TRANSCRIPT, self._on_committed)
        self._connection.on(RealtimeEvents.QUOTA_EXCEEDED, self._on_error_event)
        self._connection.on(RealtimeEvents.AUTH_ERROR, self._on_error_event)
        self._connection.on(RealtimeEvents.RATE_LIMITED, self._on_error_event)
        self._connection.on(RealtimeEvents.ERROR, self._on_error_event)
        self._connection.on(RealtimeEvents.TRANSCRIBER_ERROR, self._on_error_event)
        self._connection.on(RealtimeEvents.INVALID_REQUEST, self._on_error_event)
        self._connection.on(RealtimeEvents.QUEUE_OVERFLOW, self._on_error_event)
        self._connection.on(RealtimeEvents.RESOURCE_EXHAUSTED, self._on_error_event)
        self._connection.on(RealtimeEvents.SESSION_TIME_LIMIT_EXCEEDED, self._on_error_event)
        self._connection.on(RealtimeEvents.INPUT_ERROR, self._on_error_event)
        self._connection.on(RealtimeEvents.CHUNK_SIZE_EXCEEDED, self._on_error_event)
        self._connection.on(RealtimeEvents.INSUFFICIENT_AUDIO_ACTIVITY, self._on_error_event)
        self._connection.on(RealtimeEvents.CLOSE, self._on_closed)

        self._last_send_at = time.monotonic()
        self._queue_task = asyncio.create_task(self._drain_connection_events())
        self._keepalive_task = asyncio.create_task(self._keepalive_loop())

    @staticmethod
    def _event_name(data: Any) -> str:
        if isinstance(data, dict):
            return str(data.get("message_type") or data.get("type") or "")
        return str(getattr(data, "message_type", "") or getattr(data, "type", "") or "")

    @staticmethod
    def _event_text(data: Any) -> str:
        if isinstance(data, dict):
            return str(data.get("text") or "")
        return str(getattr(data, "text", "") or "")

    def _on_partial(self, data: Any) -> None:
        text = self._event_text(data)
        identity = self._event_projection.active_identity
        if identity is None:
            return
        provenance = self._event_provenance(data)
        sequence = self._event_projection.next_update_sequence(identity)
        if sequence is None:
            return
        self._scoped_provenance.append(provenance)
        self._event_projection.put_update(
            STTProviderTurnUpdate(
                identity=identity,
                sequence=sequence,
                stability="provisional",
                assembly="replace",
                text=text,
                provenance=provenance,
            )
        )

    def _on_committed(self, data: Any) -> None:
        try:
            text = self._committed_text(data)
        except TypeError, ValueError:
            self._protocol_failure("scribe_committed_transcript_missing_text")
            return
        if self._event_projection.is_legacy:
            self._enqueue_connection_event(STTBackendTranscriptEvent(text=text, is_final=True))
            return
        identity = self._event_projection.active_identity
        if identity is None:
            self._protocol_failure("scribe_unsolicited_committed_transcript")
            return
        if self._commit_identity != identity or not self._event_projection.sealed:
            return
        provenance = self._event_provenance(data, barrier="committed_transcript")
        if self._commit_write_in_flight:
            if self._pending_committed is not None:
                self._protocol_failure("scribe_duplicate_committed_transcript")
                return
            self._pending_committed = (identity, text, provenance)
            return
        self._finish_committed(identity, text, provenance)

    @staticmethod
    def _committed_text(data: Any) -> str:
        if isinstance(data, dict):
            if "text" not in data:
                raise ValueError("committed transcript is missing text")
            text = data["text"]
        else:
            if not hasattr(data, "text"):
                raise ValueError("committed transcript is missing text")
            text = data.text
        if not isinstance(text, str):
            raise TypeError("committed transcript text must be a string")
        return text

    def _finish_committed(
        self,
        identity: STTProviderTurnIdentity,
        text: str,
        provenance: STTNativeProvenance,
    ) -> None:
        if (
            self._stopped
            or self._commit_write_in_flight
            or self._commit_identity != identity
            or not self._event_projection.is_current(identity)
            or not self._event_projection.sealed
        ):
            return
        self._scoped_provenance.append(provenance)
        accepted = self._event_projection.terminal(
            STTProviderTurnTerminal(
                identity=identity,
                outcome="final" if text else "empty",
                text=text,
                text_authority="authoritative",
                epoch_disposition="reuse",
                provenance=tuple(self._scoped_provenance),
            )
        )
        self._clear_scoped_turn()
        if not accepted:
            self._event_projection.end_epoch(
                orderly=False,
                reason="scribe_connection_event_overflow",
                provider_turn_id=identity.provider_turn_id,
            )
            self._end_connection_stream()

    def _protocol_failure(self, reason: str) -> None:
        if self._event_projection.is_scoped:
            self._scoped_transport_failure(reason, orderly=False)
            self._end_connection_stream()
            return
        if self._stopped:
            return
        self._stopped = True
        self._enqueue_connection_event(RuntimeError(reason))

    def _on_error_event(self, data: Any) -> None:
        if self._stopped:
            return
        event_name = self._event_name(data)
        if event_name in _RECOVERABLE_SCRIBE_EVENTS:
            self._scoped_transport_failure(f"scribe_{event_name}", orderly=False)
            self._end_connection_stream()
            return
        logger.warning("[STT] Scribe provider event %s", event_name)
        self._stopped = True
        self._scoped_transport_failure(f"scribe_{event_name}", orderly=False)
        self._enqueue_connection_event(RuntimeError(f"Scribe realtime error: {event_name}"))

    def _on_closed(self) -> None:
        if self._stopped:
            return
        self._scoped_transport_failure("scribe_connection_closed", orderly=True)
        self._end_connection_stream()

    def _end_connection_stream(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        self._enqueue_connection_event(_CLOSED)

    @staticmethod
    def _event_provenance(
        data: Any,
        *,
        barrier: str | None = None,
    ) -> STTNativeProvenance:
        _ = data
        return STTNativeProvenance(barrier=barrier)

    def _enqueue_connection_event(self, event: object) -> None:
        if self._event_projection.is_scoped and event is not _CLOSED:
            if isinstance(event, BaseException):
                event = _CLOSED
            else:
                return
        try:
            self._connection_events.put_nowait(event)
        except asyncio.QueueFull:
            self._scoped_transport_failure("scribe_connection_event_overflow", orderly=False)
            self._stopped = True

    def _clear_scoped_turn(self) -> None:
        self._scoped_provenance.clear()
        self._commit_identity = None
        self._commit_write_in_flight = False
        self._pending_committed = None

    def _scoped_transport_failure(self, reason: str, *, orderly: bool) -> None:
        identity = self._event_projection.active_identity
        provider_turn_id = identity.provider_turn_id if identity is not None else None
        if identity is not None:
            self._event_projection.terminal(
                STTProviderTurnTerminal(
                    identity=identity,
                    outcome="failed",
                    text_authority="none",
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

    async def _drain_connection_events(self) -> None:
        try:
            while True:
                item = await self._connection_events.get()
                if item is _CLOSED:
                    return
                self._put_event(item)
                if isinstance(item, BaseException):
                    self._stopped = True
                    return
        except asyncio.CancelledError:
            raise
        finally:
            self._scoped_transport_failure("scribe_connection_ended", orderly=True)
            self._put_event(None)

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
                    await self._connection.send({"audio_base_64": ""})
                except Exception:
                    self._scoped_transport_failure("scribe_keepalive_failed", orderly=False)
                    self._end_connection_stream()
                    return
                self._last_send_at = time.monotonic()
        except asyncio.CancelledError:
            raise

    async def begin_turn(self, request: STTProviderTurnRequest) -> None:
        if self._stopped or self._connection is None:
            raise RuntimeError("Scribe session is closed")
        self._event_projection.begin(request)
        self._scoped_provenance.clear()

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
            await connection.send({"audio_base_64": base64.b64encode(pcm16le).decode("ascii")})
        except Exception:
            self._scoped_transport_failure("scribe_write_failed", orderly=False)
            self._end_connection_stream()
            raise
        if (
            self._stopped
            or self._connection is not connection
            or not self._event_projection.is_current(identity)
        ):
            raise RuntimeError("Scribe turn lost authority during audio write")
        self._last_send_at = time.monotonic()
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
        self._commit_identity = identity
        self._commit_write_in_flight = True
        try:
            await connection.commit()
        except Exception:
            self._scoped_transport_failure("scribe_commit_failed", orderly=False)
            self._end_connection_stream()
            raise
        if (
            self._stopped
            or self._connection is not connection
            or not self._event_projection.is_current(identity)
            or self._commit_identity != identity
        ):
            return
        self._commit_write_in_flight = False
        pending = self._pending_committed
        self._pending_committed = None
        if pending is not None:
            pending_identity, text, provenance = pending
            if pending_identity == identity:
                self._finish_committed(identity, text, provenance)

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
        if self._stopped or self._connection is None:
            return
        try:
            await self._connection.send(
                {"audio_base_64": base64.b64encode(pcm16le).decode("ascii")}
            )
        except Exception:
            self._end_connection_stream()
            return
        self._last_send_at = time.monotonic()

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
            await self._connection.commit()
        except Exception:
            self._end_connection_stream()
            return

    async def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        self._enqueue_connection_event(_CLOSED)

    async def close(self) -> None:
        await self.stop()
        tasks = tuple(task for task in (self._queue_task, self._keepalive_task) if task is not None)
        self._queue_task = None
        self._keepalive_task = None
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._put_event(None)
        if self._connection is not None:
            with contextlib.suppress(Exception):
                result = self._connection.close()
                if asyncio.iscoroutine(result):
                    await result
            self._connection = None
        self._event_projection.close()

    async def events(self) -> AsyncIterator[STTBackendTranscriptEvent]:
        async for event in self._event_projection.events():
            yield event

    def _put_event(self, event: STTBackendTranscriptEvent | BaseException | None) -> None:
        self._event_projection.put_legacy(event)
