"""Deepgram Realtime STT Backend using official SDK v5.

WebSocket-based Speech-to-Text using Deepgram's nova-3 model.
Uses the official deepgram-sdk v5 with manual KeepAlive messages (every 5 seconds)
to prevent the 10-second timeout (NET-0001 error).
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field, replace
from typing import Any, AsyncIterator, Sequence

from puripuly_heart.core.audio.format import AudioCaptureSpan
from puripuly_heart.core.messages import ErrorDiagnostics
from puripuly_heart.core.speech_boundary import SpeechBoundaryReason
from puripuly_heart.core.stt.backend import (
    LEGACY_STT_SESSION_PROJECTION,
    STTAudioClock,
    STTBackend,
    STTBackendSession,
    STTBackendTranscriptEvent,
    STTNativeProvenance,
    STTProviderTurnIdentity,
    STTProviderTurnRequest,
    STTProviderTurnTerminal,
    STTProviderTurnUpdate,
    STTSessionProjection,
    STTSourceCapabilities,
    STTSourceRelease,
    STTTextContribution,
    STTTextTiming,
)
from puripuly_heart.core.stt.diagnostics import recognition_failure_diagnostics
from puripuly_heart.core.stt.session_projection import STTSessionEventProjection
from puripuly_heart.domain.models import FinalLanguageRun
from puripuly_heart.domain.recognition import NativeTranscriptionEvidence

logger = logging.getLogger(__name__)
_DEEPGRAM_KEYTERM_MODEL = "nova-3"


def _native_language_runs(
    text: str, words: Sequence[Any], language: str | None
) -> tuple[FinalLanguageRun, ...]:
    runs = []
    cursor = 0
    for word in words:
        word_text = getattr(word, "punctuated_word", None) or getattr(word, "word", None)
        word_language = getattr(word, "language", None) or language
        if not isinstance(word_text, str) or not word_text or not word_language:
            return (FinalLanguageRun(text, language),) if text and language else ()
        offset = text.find(word_text, cursor)
        if offset < 0 or text[cursor:offset].strip():
            return (FinalLanguageRun(text, language),) if text and language else ()
        end = offset + len(word_text)
        runs.append(FinalLanguageRun(text[cursor:end], str(word_language)))
        cursor = end
    if cursor < len(text):
        if runs and not text[cursor:].strip():
            previous = runs[-1]
            runs[-1] = FinalLanguageRun(previous.text + text[cursor:], previous.language)
        else:
            return (FinalLanguageRun(text, language),) if text and language else ()
    return tuple(runs)


@dataclass(slots=True)
class DeepgramRealtimeSTTBackend(STTBackend):
    """Deepgram Realtime STT Backend using official SDK v5."""

    api_key: str
    language: str  # Required: passed from wiring.py via get_deepgram_language()
    model: str = "nova-3"
    sample_rate_hz: int = 16000
    connect_timeout_s: float = 5.0
    keyterms: Sequence[str] = ()
    stream_label: str | None = None
    drain_timeout_s: float = 1.5

    async def open_session(
        self,
        *,
        projection: STTSessionProjection = LEGACY_STT_SESSION_PROJECTION,
    ) -> STTBackendSession:
        if self.sample_rate_hz not in (8000, 16000):
            raise ValueError("sample_rate_hz must be 8000 or 16000")
        if not self.api_key:
            raise ValueError("api_key must be non-empty")
        if self.connect_timeout_s <= 0:
            raise ValueError("connect_timeout_s must be > 0")
        if self.drain_timeout_s <= 0:
            raise ValueError("drain_timeout_s must be > 0")

        session = _DeepgramSDKSession(
            api_key=self.api_key,
            model=self.model,
            language=self.language,
            sample_rate_hz=self.sample_rate_hz,
            connect_timeout_s=self.connect_timeout_s,
            keyterms=list(self.keyterms),
            stream_label=self.stream_label,
            drain_timeout_s=self.drain_timeout_s,
            projection=projection,
        )
        await session.start()
        return session

    @staticmethod
    async def verify_api_key(api_key: str) -> bool:
        if not api_key:
            return False

        from puripuly_heart.core import network_clients

        def _check():
            with network_clients.external_client(timeout=5, follow_redirects=True) as client:
                response = client.get(
                    "https://api.deepgram.com/v1/projects",
                    headers={"Authorization": f"Token {api_key}"},
                )
                response.raise_for_status()
                return response.status_code == 200

        return await asyncio.to_thread(_check)


_STOP = object()
_FINALIZE = object()
_CLOSE_STREAM = object()


@dataclass(frozen=True, slots=True)
class _AudioWrite:
    payload: bytes | object
    completion: asyncio.Future[None]


@dataclass(slots=True)
class _DeepgramSDKSession(STTBackendSession):

    api_key: str
    model: str
    language: str
    sample_rate_hz: int
    connect_timeout_s: float
    keyterms: list[str]
    stream_label: str | None = None
    drain_timeout_s: float = 1.5
    projection: STTSessionProjection = LEGACY_STT_SESSION_PROJECTION

    _event_projection: STTSessionEventProjection = field(init=False, repr=False)
    _audio_q: asyncio.Queue[bytes | object | _AudioWrite] = field(init=False, repr=False)
    _run_task: asyncio.Task[None] | None = field(init=False, default=None, repr=False)
    _send_task: asyncio.Task[None] | None = field(init=False, default=None, repr=False)
    _recv_task: asyncio.Task[None] | None = field(init=False, default=None, repr=False)
    _keepalive_task: asyncio.Task[None] | None = field(init=False, default=None, repr=False)
    _send_lock: asyncio.Lock = field(init=False, default_factory=asyncio.Lock, repr=False)
    _stopped: bool = field(init=False, default=False)
    _close_deadline: float | None = field(init=False, default=None, repr=False)
    _connected: asyncio.Event = field(init=False, default_factory=asyncio.Event, repr=False)
    _startup_done: asyncio.Event = field(init=False, default_factory=asyncio.Event, repr=False)
    _startup_error: BaseException | None = field(init=False, default=None, repr=False)
    _error_reported: bool = field(init=False, default=False, repr=False)
    _scoped_fragments: list[str] = field(init=False, default_factory=list, repr=False)
    _scoped_provenance: list[STTNativeProvenance] = field(
        init=False, default_factory=list, repr=False
    )
    _scoped_language_runs: list[FinalLanguageRun] = field(
        init=False, default_factory=list, repr=False
    )
    _scoped_close_sent: bool = field(init=False, default=False, repr=False)
    _scoped_drain_task: asyncio.Task[None] | None = field(init=False, default=None, repr=False)
    _drain_tasks: set[asyncio.Task[None]] = field(init=False, default_factory=set, repr=False)
    _scoped_contributions: list[STTTextContribution] = field(
        init=False, default_factory=list, repr=False
    )
    _final_segment_ids: dict[tuple[object, ...], None] = field(
        init=False, default_factory=dict, repr=False
    )
    _final_receipts: dict[int, Any] = field(init=False, default_factory=dict, repr=False)
    _sent_samples: int = field(init=False, default=0, repr=False)
    _finalize_control_id: str | None = field(init=False, default=None, repr=False)
    _scoped_close_written: bool = field(init=False, default=False, repr=False)
    _finalize_sent_sample: int | None = field(init=False, default=None, repr=False)
    _scoped_text_length: int = field(init=False, default=0, repr=False)
    _last_final_start: float | None = field(init=False, default=None, repr=False)
    _clock_discontinuous: bool = field(init=False, default=False, repr=False)

    def __post_init__(self) -> None:
        self._event_projection = STTSessionEventProjection(self.projection)
        self._audio_q = asyncio.Queue(maxsize=258)

    def _supports_keyterms(self) -> bool:
        return self.model.strip().lower() == _DEEPGRAM_KEYTERM_MODEL

    def _build_transcript_event(self, result: Any) -> STTBackendTranscriptEvent | None:
        message_type = getattr(result, "type", None)
        if message_type is not None and message_type != "Results":
            return None
        speech_final = getattr(result, "speech_final", False)
        is_final = getattr(result, "is_final", False)
        metadata = getattr(result, "metadata", None)
        from_finalize = bool(
            getattr(result, "from_finalize", False) or getattr(metadata, "from_finalize", False)
        )
        channel = getattr(result, "channel", None)
        alternatives = getattr(channel, "alternatives", None)
        if message_type == "Results" and (channel is None or alternatives is None):
            raise ValueError("deepgram result shape invalid")
        if alternatives:
            alternative = alternatives[0]
            raw_transcript = getattr(alternative, "transcript", "") or ""
            if not isinstance(raw_transcript, str):
                raise TypeError("deepgram transcript shape invalid")
        elif from_finalize:
            raw_transcript = ""
        else:
            return None
        transcript = raw_transcript
        request_id = getattr(metadata, "request_id", None)
        model_info = getattr(metadata, "model_info", None)
        effective_model = getattr(model_info, "name", None) or self.model
        words = (getattr(alternative, "words", None) or ()) if alternatives else ()
        languages = getattr(alternative, "languages", None) if alternatives else None
        language = str(languages[0]) if languages and len(languages) == 1 else None
        language_runs = _native_language_runs(raw_transcript, words, language)
        model_metadata = tuple(
            (key, str(value))
            for key, value in (
                ("name", getattr(model_info, "name", None)),
                ("version", getattr(model_info, "version", None)),
                ("arch", getattr(model_info, "arch", None)),
                ("model_uuid", getattr(metadata, "model_uuid", None)),
            )
            if value is not None
        )
        qualified = self.model == "nova-3" and effective_model in {"nova-3", "general-nova-3"}
        start = getattr(result, "start", None)
        duration = getattr(result, "duration", None)
        segment_key = None
        source_order_unresolved = False
        if is_final and raw_transcript:
            receipt_id = id(result)
            if self._final_receipts.get(receipt_id) is result:
                return None
            self._final_receipts[receipt_id] = result
            if len(self._final_receipts) > 256:
                del self._final_receipts[next(iter(self._final_receipts))]
            if isinstance(start, (int, float)) and isinstance(duration, (int, float)):
                segment_key = (
                    request_id,
                    tuple(getattr(result, "channel_index", ()) or ()),
                    start,
                    duration,
                )
                source_order_unresolved = segment_key in self._final_segment_ids
                if source_order_unresolved or (
                    self._last_final_start is not None and start < self._last_final_start
                ):
                    self._clock_discontinuous = True
                self._last_final_start = start
                self._final_segment_ids[segment_key] = None
                if len(self._final_segment_ids) > 256:
                    del self._final_segment_ids[next(iter(self._final_segment_ids))]
        qualified = qualified and not self._clock_discontinuous
        clock_id = f"deepgram:{self.projection.provider_epoch_id or request_id or 'connection'}"
        provenance = STTNativeProvenance(
            native_request_id=str(request_id) if request_id is not None else None,
            barrier="finalize_ack" if from_finalize else ("is_final" if is_final else None),
            from_finalize=from_finalize,
            capabilities=STTSourceCapabilities(
                backend_id="deepgram",
                model_id=str(effective_model),
                policy="timed" if qualified else "unknown",
            ),
            audio_clocks=(STTAudioClock(clock_id, self.sample_rate_hz, 0, qualified),),
            sent_until_sample=self._sent_samples,
            control_id=self._finalize_control_id if from_finalize else None,
            control_kind="Finalize" if from_finalize else None,
            control_sent_sample=self._finalize_sent_sample if from_finalize else None,
            transcription=NativeTranscriptionEvidence(
                finished=bool(speech_final),
                language_code=language,
                words=tuple(
                    (
                        getattr(word, "punctuated_word", None) or getattr(word, "word", None),
                        str(word.start) if getattr(word, "start", None) is not None else None,
                        str(word.end) if getattr(word, "end", None) is not None else None,
                    )
                    for word in words
                ),
            ),
            native_model_metadata=model_metadata,
        )
        if self._event_projection.is_scoped:
            identity = self._event_projection.active_identity
            if identity is not None and is_final and raw_transcript:
                text_start = self._scoped_text_length
                contribution_id = (
                    f"{identity.provider_turn_id}:segment:{len(self._scoped_contributions) + 1}"
                )
                timings = []
                if isinstance(start, (int, float)) and isinstance(duration, (int, float)):
                    timings.append(
                        STTTextTiming(
                            contribution_id,
                            text_start,
                            text_start + len(raw_transcript),
                            "result_span",
                            start,
                            start + duration,
                            clock_id=clock_id,
                            complete_boundary=True,
                            estimated=True,
                        )
                    )
                cursor = 0
                for word in words:
                    word_text = getattr(word, "punctuated_word", None) or getattr(
                        word, "word", None
                    )
                    if not isinstance(word_text, str) or not word_text:
                        continue
                    offset = raw_transcript.find(word_text, cursor)
                    if offset < 0 or raw_transcript[cursor:offset].strip():
                        continue
                    cursor = offset + len(word_text)
                    timings.append(
                        STTTextTiming(
                            contribution_id,
                            text_start + offset,
                            text_start + cursor,
                            "word_interval",
                            getattr(word, "start", None),
                            getattr(word, "end", None),
                            clock_id=clock_id,
                            complete_boundary=True,
                            estimated=True,
                        )
                    )
                provenance = replace(provenance, text_timing=tuple(timings))
            if source_order_unresolved:
                reason = "deepgram_final_source_order_unresolved"
                self._terminalize_scoped(
                    provenance=replace(provenance, unresolved_source_text=raw_transcript),
                    epoch_disposition="retire",
                    degraded_reason=reason,
                    empty_is_success=False,
                )
                self._event_projection.end_epoch(
                    orderly=False,
                    reason=reason,
                    provider_turn_id=identity.provider_turn_id if identity is not None else None,
                )
                self._stopped = True
                return None
            self._handle_scoped_result(
                identity,
                raw_transcript,
                bool(is_final),
                from_finalize,
                provenance,
                language_runs,
            )
        if self._event_projection.is_scoped or not (is_final or from_finalize):
            return None
        if not transcript:
            return STTBackendTranscriptEvent(text="", is_final=True)
        return STTBackendTranscriptEvent(
            text=transcript, is_final=True, final_language_runs=language_runs
        )

    def _handle_scoped_result(
        self,
        identity: STTProviderTurnIdentity | None,
        text: str,
        is_final: bool,
        from_finalize: bool,
        provenance: STTNativeProvenance,
        language_runs: tuple[FinalLanguageRun, ...] = (),
    ) -> None:
        if identity is None:
            if is_final or from_finalize:
                self._event_projection.end_epoch(
                    orderly=False,
                    reason="deepgram_idle_result",
                )
            return
        if not self._event_projection.is_current(identity):
            return
        if is_final and text:
            self._scoped_fragments.append(text)
            self._scoped_provenance.append(provenance)
            self._scoped_language_runs.extend(language_runs or (FinalLanguageRun(text, ""),))
            text_start = self._scoped_text_length
            self._scoped_text_length += len(text)
            contribution_id = (
                provenance.text_timing[0].contribution_id
                if provenance.text_timing
                else f"{identity.provider_turn_id}:segment:{len(self._scoped_contributions) + 1}"
            )
            contribution = STTTextContribution(contribution_id, text_start, text_start + len(text))
            self._scoped_contributions.append(contribution)
            sequence = self._event_projection.next_update_sequence(identity)
            if sequence is not None:
                self._event_projection.put_update(
                    STTProviderTurnUpdate(
                        identity=identity,
                        sequence=sequence,
                        stability="stable",
                        assembly="append",
                        text=text,
                        final_language_runs=language_runs,
                        provenance=provenance,
                        contribution=contribution,
                        source_release=STTSourceRelease(
                            f"{contribution_id}:final", 0, contribution.text_end, "end"
                        ),
                    )
                )
        if from_finalize:
            if not self._event_projection.sealed:
                self._terminalize_scoped(
                    provenance=provenance,
                    epoch_disposition="retire",
                    degraded_reason="deepgram_finalize_ack_before_seal",
                    empty_is_success=False,
                )
                return
            self._terminalize_scoped(
                provenance=provenance,
                epoch_disposition="retire" if self._scoped_close_sent else "reuse",
            )

    def _terminalize_scoped(
        self,
        *,
        provenance: STTNativeProvenance | None = None,
        epoch_disposition: str,
        degraded_reason: str | None = None,
        empty_is_success: bool = True,
        failure_diagnostics: ErrorDiagnostics | None = None,
    ) -> None:
        identity = self._event_projection.active_identity
        if identity is None:
            return
        if provenance is not None:
            self._scoped_provenance.append(provenance)
        task = self._scoped_drain_task
        self._scoped_drain_task = None
        if task is not None and task is not asyncio.current_task():
            task.cancel()
        text = "".join(self._scoped_fragments)
        if provenance is not None and provenance.unresolved_source_text is not None:
            outcome = "failed"
            authority = "none"
        elif degraded_reason is not None and text:
            outcome = "degraded"
            authority = "degraded"
        elif degraded_reason is not None and not empty_is_success:
            outcome = "failed"
            authority = "none"
        else:
            outcome = "final" if text else "empty"
            authority = "authoritative"
        if outcome in {"failed", "degraded"} and failure_diagnostics is None:
            failure_diagnostics = recognition_failure_diagnostics(
                None,
                provider="deepgram",
                channel=self._event_projection.channel or self.stream_label or "unknown",
                operation="seal",
                reason=degraded_reason or "deepgram_transport_error",
                identity=identity,
            )
        self._event_projection.terminal(
            STTProviderTurnTerminal(
                identity=identity,
                outcome=outcome,
                text=text,
                final_language_runs=tuple(self._scoped_language_runs),
                text_authority=authority,
                failure_reason=degraded_reason,
                epoch_disposition=epoch_disposition,
                provenance=tuple(self._scoped_provenance),
                failure_diagnostics=failure_diagnostics,
                included_contributions=tuple(self._scoped_contributions),
            )
        )
        self._scoped_fragments.clear()
        self._scoped_provenance.clear()
        self._scoped_contributions.clear()
        self._scoped_text_length = 0
        self._scoped_language_runs.clear()

    def _scoped_transport_end(
        self, orderly: bool, reason: str, exception: BaseException | None = None
    ) -> None:
        diagnostics = (
            None
            if orderly
            else recognition_failure_diagnostics(
                exception,
                provider="deepgram",
                channel=self._event_projection.channel or self.stream_label or "unknown",
                operation={
                    "deepgram_protocol_failed": "protocol",
                    "deepgram_write_failed": "send",
                    "deepgram_keepalive_failed": "send",
                }.get(reason, "receive"),
                reason=reason,
                identity=self._event_projection.active_identity,
            )
        )
        identity = self._event_projection.active_identity
        provider_turn_id = identity.provider_turn_id if identity is not None else None
        if identity is not None:
            if (
                orderly
                and self._scoped_close_written
                and self._event_projection.sealed
                and reason == "deepgram_connection_closed"
            ):
                self._terminalize_scoped(
                    epoch_disposition="retire",
                    empty_is_success=True,
                )
            else:
                self._terminalize_scoped(
                    epoch_disposition="retire",
                    degraded_reason=reason,
                    empty_is_success=False,
                    failure_diagnostics=diagnostics,
                )
        self._event_projection.end_epoch(
            orderly=orderly,
            reason=reason,
            provider_turn_id=provider_turn_id,
            failure_diagnostics=diagnostics,
        )

    async def _emit_test_final(
        self,
        *,
        text: str,
    ) -> None:
        self._event_projection.put_legacy(
            STTBackendTranscriptEvent(
                text=text,
                is_final=True,
            )
        )

    async def start(self) -> None:
        self._run_task = asyncio.create_task(self._run(), name="deepgram-sdk")
        try:
            try:
                await asyncio.wait_for(self._startup_done.wait(), self.connect_timeout_s)
            except TimeoutError as cause:
                exc = RuntimeError("Deepgram SDK connection timeout")
                logger.warning(
                    "[STT] Deepgram connection timeout after %.1fs", self.connect_timeout_s
                )
                self._report_error(exc)
                raise exc from cause
            if self._startup_error is not None:
                raise self._startup_error
            if not self._connected.is_set():
                raise RuntimeError("Deepgram connection closed")
        except BaseException:
            await self.close()
            raise

    async def _run(self) -> None:
        try:
            from deepgram import AsyncDeepgramClient
            from deepgram.core.events import EventType

            from puripuly_heart.core import network_clients

            from .sdk_network import deepgram_listen_connect

            connect_kwargs: dict[str, Any] = {
                "model": self.model,
                "language": self.language,
                "encoding": "linear16",
                "sample_rate": self.sample_rate_hz,
                "channels": 1,
                "interim_results": False,
                "punctuate": True,
                "vad_events": False,
                "endpointing": False,
            }
            if self.keyterms and self._supports_keyterms():
                connect_kwargs["keyterm"] = self.keyterms

            async with (
                network_clients.external_async_client() as http_client,
                deepgram_listen_connect(
                    AsyncDeepgramClient(api_key=self.api_key, httpx_client=http_client),
                    _close_deadline=self._begin_close,
                    **connect_kwargs,
                ) as connection,
            ):

                def on_message(result: Any) -> None:
                    self._handle_message(result)

                def on_error(error: Any) -> None:
                    if not self._stopped:
                        exc = (
                            error
                            if isinstance(error, BaseException)
                            else RuntimeError("Deepgram transport failed")
                        )
                        logger.warning("Deepgram transport failed")
                        self._report_error(exc)
                        self._scoped_transport_end(
                            False,
                            (
                                "deepgram_protocol_failed"
                                if isinstance(exc, json.JSONDecodeError)
                                else "deepgram_transport_error"
                            ),
                            exc,
                        )
                        self._stopped = True

                def on_close(close_event: Any) -> None:
                    _ = close_event
                    orderly = self._scoped_close_sent or self._stopped
                    self._scoped_transport_end(orderly, "deepgram_connection_closed")
                    if not orderly:
                        self._report_error(RuntimeError("Deepgram connection closed"))
                    self._stopped = True

                def on_open(open_event: Any) -> None:
                    _ = open_event
                    self._connected.set()
                    self._startup_done.set()

                connection.on(EventType.OPEN, on_open)
                connection.on(EventType.MESSAGE, on_message)
                connection.on(EventType.ERROR, on_error)
                connection.on(EventType.CLOSE, on_close)
                self._recv_task = asyncio.create_task(
                    self._listen_loop(connection), name="deepgram-listen"
                )
                self._send_task = asyncio.create_task(
                    self._send_loop(connection), name="deepgram-send"
                )
                self._keepalive_task = asyncio.create_task(
                    self._keepalive_loop(connection), name="deepgram-keepalive"
                )
                tasks = (self._recv_task, self._send_task, self._keepalive_task)
                try:
                    await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                finally:
                    self._stopped = True
                    tasks += tuple(self._drain_tasks)
                    for task in tasks:
                        task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
                    self._scoped_drain_task = None
                    self._fail_pending_writes()
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            logger.warning("Deepgram SDK failed cause=%s", type(exc).__name__)
            if not self._connected.is_set():
                self._startup_error = exc
            self._report_error(exc)
            self._scoped_transport_end(False, "deepgram_transport_error", exc)
        finally:
            self._stopped = True
            self._startup_done.set()
            self._put_event(None)
            self._scoped_transport_end(False, "deepgram_writer_ended")

    async def _listen_loop(self, connection: Any) -> None:
        try:
            await connection.start_listening()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._report_error(exc)
            self._scoped_transport_end(
                False,
                (
                    "deepgram_protocol_failed"
                    if isinstance(exc, ValueError | TypeError)
                    else "deepgram_transport_error"
                ),
                exc,
            )

    async def _send_loop(self, connection: Any) -> None:
        from deepgram.extensions.types.sockets import ListenV1ControlMessage

        data: bytes | object | _AudioWrite = _STOP
        try:
            while True:
                if self._stopped and self._audio_q.empty():
                    return
                data = await self._audio_q.get()
                if data is _STOP:
                    return
                payload = data.payload if isinstance(data, _AudioWrite) else data
                async with self._send_lock:
                    if payload is _FINALIZE:
                        await connection.send_control(ListenV1ControlMessage(type="Finalize"))
                    elif payload is _CLOSE_STREAM:
                        await connection.send_control(ListenV1ControlMessage(type="CloseStream"))
                    elif isinstance(payload, bytes):
                        await connection.send_media(payload)
                self._resolve_write(getattr(data, "completion", None), None)
        except asyncio.CancelledError:
            self._resolve_write(
                getattr(data, "completion", None), RuntimeError("Deepgram writer stopped")
            )
            raise
        except Exception as exc:
            logger.warning("Deepgram writer failed cause=%s", type(exc).__name__)
            self._resolve_write(getattr(data, "completion", None), exc)
            self._report_error(exc)
            self._scoped_transport_end(False, "deepgram_write_failed", exc)
        finally:
            self._fail_pending_writes()

    async def _keepalive_loop(self, connection: Any) -> None:
        from deepgram.extensions.types.sockets import ListenV1ControlMessage

        try:
            while not self._stopped:
                await asyncio.sleep(5.0)
                if self._stopped:
                    return
                async with self._send_lock:
                    await connection.send_control(ListenV1ControlMessage(type="KeepAlive"))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.debug("Deepgram keepalive failed cause=%s", type(exc).__name__)
            self._report_error(exc)
            self._scoped_transport_end(False, "deepgram_keepalive_failed", exc)

    def _handle_message(self, result: Any) -> None:
        if self._stopped:
            return
        try:
            event = self._build_transcript_event(result)
            if event is not None:
                self._put_event(event)
        except Exception as exc:
            self._report_error(exc)
            self._scoped_transport_end(False, "deepgram_protocol_failed", exc)
            self._stopped = True

    def _report_error(self, exc: BaseException) -> None:
        if self._error_reported:
            return
        self._error_reported = True
        self._put_event(exc)

    def _put_event(self, event: STTBackendTranscriptEvent | BaseException | None) -> None:
        if self._event_projection.is_legacy:
            self._event_projection.put_legacy(event)

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
        error = RuntimeError("Deepgram writer stopped")
        while not self._audio_q.empty():
            item = self._audio_q.get_nowait()
            self._resolve_write(getattr(item, "completion", None), error)

    async def _write_payload(self, payload: bytes | object) -> None:
        if self._stopped:
            raise RuntimeError("Deepgram session is closed")
        completion = asyncio.get_running_loop().create_future()
        self._audio_q.put_nowait(_AudioWrite(payload=payload, completion=completion))
        await completion

    async def begin_turn(self, request: STTProviderTurnRequest) -> None:
        if self._stopped or self._scoped_close_sent:
            raise RuntimeError("Deepgram session is closed")
        self._event_projection.begin(request)
        self._scoped_fragments.clear()
        self._scoped_provenance.clear()
        self._scoped_contributions.clear()
        self._scoped_text_length = 0
        self._scoped_language_runs.clear()
        self._finalize_control_id = None
        self._finalize_sent_sample = None

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
        await self._write_payload(pcm16le)
        self._sent_samples += len(pcm16le) // 2
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
        self._finalize_control_id = f"{identity.provider_turn_id}:finalize"
        self._finalize_sent_sample = self._sent_samples
        await self._write_payload(_FINALIZE)
        if self._event_projection.is_current(identity):
            task = asyncio.create_task(self._missing_finalize_ack(identity), name="deepgram-drain")
            self._scoped_drain_task = task
            self._drain_tasks.add(task)
            task.add_done_callback(self._drain_tasks.discard)

    async def _missing_finalize_ack(self, identity: STTProviderTurnIdentity) -> None:
        try:
            await asyncio.sleep(self.drain_timeout_s)
            if not self._event_projection.is_current(identity):
                return
            self._scoped_close_sent = True
            await self._write_payload(_CLOSE_STREAM)
            self._scoped_close_written = True
            if not self._event_projection.is_current(identity):
                return
            self._scoped_provenance.append(
                STTNativeProvenance(
                    barrier="close_stream",
                    control_id=f"{identity.provider_turn_id}:close",
                    control_kind="CloseStream",
                    control_sent_sample=self._sent_samples,
                    sent_until_sample=self._sent_samples,
                )
            )
            await asyncio.sleep(self.drain_timeout_s)
        except asyncio.CancelledError:
            return
        except Exception as exc:
            if self._event_projection.is_current(identity):
                self._terminalize_scoped(
                    epoch_disposition="retire",
                    degraded_reason="deepgram_close_stream_failed",
                    empty_is_success=False,
                    failure_diagnostics=recognition_failure_diagnostics(
                        exc,
                        provider="deepgram",
                        channel=self.stream_label or "self",
                        operation="seal",
                        reason="deepgram_close_stream_failed",
                        identity=identity,
                    ),
                )
            return
        if self._event_projection.is_current(identity):
            self._terminalize_scoped(
                epoch_disposition="retire",
                degraded_reason="deepgram_finalize_ack_missing",
                empty_is_success=False,
            )

    async def abort_turn(self, identity: STTProviderTurnIdentity, *, reason: str) -> None:
        self._event_projection.require_open(identity)
        task = self._scoped_drain_task
        self._scoped_drain_task = None
        if task is not None:
            task.cancel()
        self._event_projection.terminal(
            STTProviderTurnTerminal(
                identity=identity,
                outcome="cancelled",
                text_authority="none",
                failure_reason=reason,
                epoch_disposition="retire",
            )
        )
        self._scoped_fragments.clear()
        self._scoped_provenance.clear()
        self._scoped_contributions.clear()
        self._scoped_text_length = 0
        self._scoped_language_runs.clear()
        self._scoped_close_sent = True
        if not self._stopped:
            await self._write_payload(_CLOSE_STREAM)
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
            raise RuntimeError("Deepgram audio queue overflow")
        self._audio_q.put_nowait(pcm16le)

    async def on_speech_end(
        self,
        *,
        trailing_silence_ms: int | None = None,
        reason: SpeechBoundaryReason | None = None,
    ) -> None:
        """Handle end of speech and finalize."""
        if self._stopped:
            return

        self._audio_q.put_nowait(_FINALIZE)

    def _begin_close(self) -> float:
        if self._close_deadline is None:
            self._close_deadline = asyncio.get_running_loop().time() + 5.0
        return self._close_deadline

    async def stop(self) -> None:
        self._begin_close()
        if self._stopped:
            return
        self._stopped = True
        try:
            self._audio_q.put_nowait(_STOP)
        except asyncio.QueueFull:
            if self._send_task is None or self._send_task.done():
                self._fail_pending_writes()

    async def close(self) -> None:
        deadline = self._begin_close()
        try:
            try:
                async with asyncio.timeout_at(deadline):
                    await self.stop()
                    self._scoped_drain_task = None
                    drain_tasks = tuple(
                        task for task in self._drain_tasks if task is not asyncio.current_task()
                    )
                    for task in drain_tasks:
                        task.cancel()
                    await asyncio.gather(*drain_tasks, return_exceptions=True)
                    task = self._run_task
                    if task is not None:
                        if not self._connected.is_set():
                            task.cancel()
                        try:
                            await asyncio.shield(task)
                        except asyncio.CancelledError:
                            if asyncio.current_task().cancelling():
                                raise
            except TimeoutError:
                pass
        finally:
            task = self._run_task
            if task is not None and not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            self._run_task = None
            self._fail_pending_writes()
            self._event_projection.close()

    async def events(self) -> AsyncIterator[STTBackendTranscriptEvent]:
        async for event in self._event_projection.events():
            yield event
