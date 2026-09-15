"""Deepgram Realtime STT Backend using official SDK v5.

WebSocket-based Speech-to-Text using Deepgram's nova-3 model.
Uses the official deepgram-sdk v5 with manual KeepAlive messages (every 5 seconds)
to prevent the 10-second timeout (NET-0001 error).
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
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

logger = logging.getLogger(__name__)
_DEEPGRAM_KEYTERM_MODEL = "nova-3"


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

        import urllib.error
        import urllib.request

        def _check():
            req = urllib.request.Request(
                "https://api.deepgram.com/v1/projects",
                headers={"Authorization": f"Token {api_key}"},
            )
            try:
                with urllib.request.urlopen(req, timeout=5) as response:
                    if response.status == 200:
                        return True
                    return False
            except urllib.error.HTTPError as e:
                raise Exception(f"HTTP {e.code}: {e.reason}")
            except Exception as e:
                raise Exception(f"Connection failed: {e}")

        return await asyncio.to_thread(_check)


_STOP = object()
_FINALIZE = object()
_CLOSE_STREAM = object()


@dataclass(frozen=True, slots=True)
class _ThreadWrite:
    payload: bytes | object
    completion: asyncio.Future[None]


@dataclass(slots=True)
class _DeepgramSDKSession(STTBackendSession):
    """Internal session using official Deepgram SDK v5 with threading."""

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
    _audio_q: queue.Queue[bytes | object | _ThreadWrite] = field(init=False, repr=False)
    _thread: threading.Thread | None = field(init=False, default=None, repr=False)
    _stopped: bool = field(init=False, default=False)
    _loop: asyncio.AbstractEventLoop | None = field(init=False, default=None, repr=False)
    _connected: threading.Event = field(init=False, repr=False)
    _connect_started_at: float | None = field(init=False, default=None, repr=False)
    _error_reported: bool = field(init=False, default=False, repr=False)
    _emitted_finals: int = field(init=False, default=0, repr=False)
    _empty_final_acks: int = field(init=False, default=0, repr=False)
    _summary_logged: bool = field(init=False, default=False, repr=False)
    _scoped_fragments: list[str] = field(init=False, default_factory=list, repr=False)
    _scoped_provenance: list[STTNativeProvenance] = field(
        init=False, default_factory=list, repr=False
    )
    _scoped_close_sent: bool = field(init=False, default=False, repr=False)
    _scoped_drain_task: asyncio.Task[None] | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        self._event_projection = STTSessionEventProjection(self.projection)
        self._audio_q = queue.Queue(maxsize=258)
        self._connected = threading.Event()

    def _supports_keyterms(self) -> bool:
        return self.model.strip().lower() == _DEEPGRAM_KEYTERM_MODEL

    def _build_transcript_event(self, result: Any) -> STTBackendTranscriptEvent | None:
        speech_final = getattr(result, "speech_final", False)
        is_final = getattr(result, "is_final", False)
        metadata = getattr(result, "metadata", None)
        from_finalize = bool(
            getattr(result, "from_finalize", False) or getattr(metadata, "from_finalize", False)
        )
        channel = getattr(result, "channel", None)
        alternatives = getattr(channel, "alternatives", None)
        if alternatives:
            alternative = alternatives[0]
            raw_transcript = str(getattr(alternative, "transcript", "") or "")
        elif from_finalize:
            raw_transcript = ""
        else:
            return None
        transcript = raw_transcript.strip()
        request_id = getattr(metadata, "request_id", None)
        provenance = STTNativeProvenance(
            native_request_id=str(request_id) if request_id is not None else None,
            barrier="finalize_ack" if from_finalize else None,
            from_finalize=from_finalize,
        )
        if self._event_projection.is_scoped:
            self._schedule_scoped_result(
                raw_transcript,
                is_final=bool(is_final),
                from_finalize=from_finalize,
                provenance=provenance,
            )
        logger.info(
            "[STT] Transcript metadata text_len=%s is_final=%s speech_final=%s",
            len(transcript),
            is_final,
            speech_final,
        )
        if self._event_projection.is_scoped or not (is_final or speech_final or from_finalize):
            return None
        if not transcript:
            if self.stream_label == "peer":
                self._empty_final_acks += 1
                logger.info(
                    "[STT][peer] Empty final transcript acknowledged "
                    "(is_final=%s, speech_final=%s)",
                    is_final,
                    speech_final,
                )
            return STTBackendTranscriptEvent(text="", is_final=True)
        if self.stream_label == "peer":
            self._emitted_finals += 1
        return STTBackendTranscriptEvent(text=transcript, is_final=True)

    def _schedule_scoped_result(
        self,
        text: str,
        *,
        is_final: bool,
        from_finalize: bool,
        provenance: STTNativeProvenance,
    ) -> None:
        loop = self._loop
        if loop is None:
            return
        identity = self._event_projection.active_identity
        loop.call_soon_threadsafe(
            self._handle_scoped_result,
            identity,
            text,
            is_final,
            from_finalize,
            provenance,
        )

    def _handle_scoped_result(
        self,
        identity: STTProviderTurnIdentity | None,
        text: str,
        is_final: bool,
        from_finalize: bool,
        provenance: STTNativeProvenance,
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
            sequence = self._event_projection.next_update_sequence(identity)
            if sequence is not None:
                self._event_projection.put_update(
                    STTProviderTurnUpdate(
                        identity=identity,
                        sequence=sequence,
                        stability="stable",
                        assembly="append",
                        text=text,
                        provenance=provenance,
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
        if degraded_reason is not None and text:
            outcome = "degraded"
            authority = "degraded"
        elif degraded_reason is not None and not empty_is_success:
            outcome = "failed"
            authority = "none"
        else:
            outcome = "final" if text else "empty"
            authority = "authoritative"
        self._event_projection.terminal(
            STTProviderTurnTerminal(
                identity=identity,
                outcome=outcome,
                text=text,
                text_authority=authority,
                failure_reason=degraded_reason,
                epoch_disposition=epoch_disposition,
                provenance=tuple(self._scoped_provenance),
            )
        )
        self._scoped_fragments.clear()
        self._scoped_provenance.clear()

    def _scoped_transport_end(self, orderly: bool, reason: str) -> None:
        identity = self._event_projection.active_identity
        provider_turn_id = identity.provider_turn_id if identity is not None else None
        if identity is not None:
            if orderly and self._scoped_close_sent and self._event_projection.sealed:
                self._terminalize_scoped(
                    epoch_disposition="retire",
                    empty_is_success=True,
                )
            else:
                self._terminalize_scoped(
                    epoch_disposition="retire",
                    degraded_reason=reason,
                    empty_is_success=False,
                )
        self._event_projection.end_epoch(
            orderly=orderly,
            reason=reason,
            provider_turn_id=provider_turn_id,
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
        self._loop = asyncio.get_running_loop()
        self._connect_started_at = time.monotonic()
        self._thread = threading.Thread(target=self._run_sync, name="deepgram-sdk", daemon=True)
        self._thread.start()

        # Wait for connection to be established
        logger.info("[STT] Deepgram connecting (timeout=%.1fs)", self.connect_timeout_s)
        connected = await asyncio.to_thread(self._connected.wait, self.connect_timeout_s)
        if not connected:
            exc = RuntimeError("Deepgram SDK connection timeout")
            logger.warning("[STT] Deepgram connection timeout after %.1fs", self.connect_timeout_s)
            self._report_error(exc)
            await self.stop()
            raise exc

    def _run_sync(self) -> None:
        """Run Deepgram SDK connection in a separate thread."""
        try:
            from deepgram import DeepgramClient
            from deepgram.core.events import EventType
            from deepgram.extensions.types.sockets import ListenV1ControlMessage

            # Create client with api_key
            client = DeepgramClient(api_key=self.api_key)

            # Connect with streaming options using v1.connect() API
            connect_kwargs: dict[str, Any] = {
                "model": self.model,
                "language": self.language,
                "encoding": "linear16",
                "sample_rate": self.sample_rate_hz,
                "channels": 1,
                "interim_results": False,
                "punctuate": True,
                "vad_events": False,  # Disabled: using local VAD + Finalize
                "endpointing": False,  # Disabled: using local VAD for speech boundaries
            }
            if self.keyterms and self._supports_keyterms():
                connect_kwargs["keyterm"] = self.keyterms

            with client.listen.v1.connect(
                **connect_kwargs,
            ) as connection:
                # Set up event handlers
                def on_message(result: Any) -> None:
                    try:
                        event = self._build_transcript_event(result)
                        if event is not None:
                            self._put_event(event)
                    except Exception as e:
                        logger.debug(f"Deepgram parse error: {e}")

                def on_error(error: Any) -> None:
                    logger.warning(f"Deepgram error: {error}")
                    if not self._stopped:
                        exc = RuntimeError(f"Deepgram error: {error}")
                        self._report_error(exc)
                        if self._loop is not None:
                            self._loop.call_soon_threadsafe(
                                self._scoped_transport_end,
                                False,
                                "deepgram_transport_error",
                            )
                        self._stopped = True
                        try:
                            self._audio_q.put_nowait(_STOP)
                        except Exception:
                            pass

                def on_close(close_event: Any) -> None:
                    _ = close_event
                    logger.debug("Deepgram: Connection closed")
                    orderly = self._scoped_close_sent or self._stopped
                    if self._loop is not None:
                        self._loop.call_soon_threadsafe(
                            self._scoped_transport_end,
                            orderly,
                            "deepgram_connection_closed",
                        )
                    if not self._stopped and not self._scoped_close_sent:
                        self._report_error(RuntimeError("Deepgram connection closed"))
                        self._stopped = True
                        try:
                            self._audio_q.put_nowait(_STOP)
                        except Exception:
                            pass

                def on_open(open_event: Any) -> None:
                    _ = open_event
                    logger.debug("Deepgram: Connection opened")
                    if self._connect_started_at is not None:
                        elapsed = time.monotonic() - self._connect_started_at
                        logger.info("[STT] Deepgram connected in %.2fs", elapsed)
                    self._connected.set()

                connection.on(EventType.OPEN, on_open)
                connection.on(EventType.MESSAGE, on_message)
                connection.on(EventType.ERROR, on_error)
                connection.on(EventType.CLOSE, on_close)

                # Start listening in a separate thread (it's blocking)
                def listening_thread():
                    try:
                        connection.start_listening()
                    except Exception as e:
                        logger.debug(f"Listening thread ended: {e}")

                listen_thread = threading.Thread(target=listening_thread, daemon=True)
                listen_thread.start()

                logger.debug("Deepgram SDK connection and listening started")

                # Start keepalive thread (sends KeepAlive every 5 seconds to prevent 10-second timeout)
                def keepalive_thread():
                    while not self._stopped:
                        time.sleep(5.0)
                        if self._stopped:
                            break
                        try:
                            connection.send_control(ListenV1ControlMessage(type="KeepAlive"))
                            logger.debug("[STT] KeepAlive sent")
                        except Exception as e:
                            logger.debug(f"KeepAlive failed: {e}")
                            break

                ka_thread = threading.Thread(target=keepalive_thread, daemon=True)
                ka_thread.start()

                # Audio sending loop
                audio_chunks_sent = 0
                while True:
                    try:
                        data = self._audio_q.get(timeout=0.1)
                    except queue.Empty:
                        if self._stopped:
                            break
                        continue

                    if data is _STOP:
                        logger.debug(
                            f"Deepgram: Stop signal received after {audio_chunks_sent} chunks"
                        )
                        self._put_event(None)
                        break

                    completion: asyncio.Future[None] | None = None
                    payload = data
                    if isinstance(data, _ThreadWrite):
                        completion = data.completion
                        payload = data.payload
                    try:
                        if payload is _FINALIZE:
                            connection.send_control(ListenV1ControlMessage(type="Finalize"))
                            logger.info("[STT] Finalize message sent to Deepgram")
                        elif payload is _CLOSE_STREAM:
                            connection.send_control(ListenV1ControlMessage(type="CloseStream"))
                            logger.info("[STT] CloseStream message sent to Deepgram")
                        elif isinstance(payload, bytes):
                            connection.send_media(payload)
                            audio_chunks_sent += 1
                            if audio_chunks_sent == 1:
                                logger.info(
                                    f"[STT] First audio chunk sent to Deepgram ({len(payload)} bytes)"
                                )
                            elif audio_chunks_sent % 50 == 0:
                                logger.debug(f"[STT] Audio chunks sent: {audio_chunks_sent}")
                    except Exception as exc:
                        logger.warning("Deepgram writer failed: %s", exc)
                        self._resolve_thread_write(completion, exc)
                        if self._loop is not None:
                            self._loop.call_soon_threadsafe(
                                self._scoped_transport_end,
                                False,
                                "deepgram_write_failed",
                            )
                        break
                    else:
                        self._resolve_thread_write(completion, None)

        except BaseException as exc:
            logger.exception("Deepgram SDK thread error")
            self._put_event(exc)
        finally:
            self._put_event(None)
            if self._loop is not None:
                self._loop.call_soon_threadsafe(
                    self._scoped_transport_end,
                    self._scoped_close_sent or self._stopped,
                    "deepgram_writer_ended",
                )

    def _report_error(self, exc: BaseException) -> None:
        if self._error_reported:
            return
        self._error_reported = True
        self._put_event(exc)

    def _put_event(self, event: STTBackendTranscriptEvent | BaseException | None) -> None:
        if self._event_projection.is_legacy and self._loop is not None:
            self._loop.call_soon_threadsafe(self._event_projection.put_legacy, event)

    def _resolve_thread_write(
        self,
        completion: asyncio.Future[None] | None,
        error: BaseException | None,
    ) -> None:
        loop = self._loop
        if completion is None or loop is None:
            return

        def resolve() -> None:
            if completion.done():
                return
            if error is None:
                completion.set_result(None)
            else:
                completion.set_exception(error)

        loop.call_soon_threadsafe(resolve)

    async def _write_thread_payload(self, payload: bytes | object) -> None:
        if self._stopped:
            raise RuntimeError("Deepgram session is closed")
        completion = asyncio.get_running_loop().create_future()
        self._audio_q.put_nowait(_ThreadWrite(payload=payload, completion=completion))
        await completion

    async def begin_turn(self, request: STTProviderTurnRequest) -> None:
        if self._stopped or self._scoped_close_sent:
            raise RuntimeError("Deepgram session is closed")
        self._event_projection.begin(request)
        self._scoped_fragments.clear()
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
        await self._write_thread_payload(pcm16le)
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
        await self._write_thread_payload(_FINALIZE)
        if self._event_projection.is_current(identity):
            self._scoped_drain_task = asyncio.create_task(self._missing_finalize_ack(identity))

    async def _missing_finalize_ack(self, identity: STTProviderTurnIdentity) -> None:
        try:
            await asyncio.sleep(self.drain_timeout_s)
            if not self._event_projection.is_current(identity):
                return
            self._scoped_close_sent = True
            await self._write_thread_payload(_CLOSE_STREAM)
            await asyncio.sleep(self.drain_timeout_s)
        except asyncio.CancelledError:
            return
        except Exception:
            if self._event_projection.is_current(identity):
                self._terminalize_scoped(
                    epoch_disposition="retire",
                    degraded_reason="deepgram_close_stream_failed",
                    empty_is_success=False,
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
        self._scoped_close_sent = True
        if not self._stopped:
            await self._write_thread_payload(_CLOSE_STREAM)
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

        existing_ms = max(int(trailing_silence_ms or 0), 0)
        wait_ms = boundary_wait_ms(reason, observed_tail_ms=existing_ms)

        logger.info(
            "[STT][Tail] provider=deepgram boundary_reason=%s observed_tail_ms=%s "
            "boundary_wait_ms=%s",
            reason,
            existing_ms,
            wait_ms,
        )
        self._audio_q.put_nowait(_FINALIZE)

    async def stop(self) -> None:
        self._log_summary_once()
        if self._stopped:
            return
        self._stopped = True
        self._scoped_close_sent = True
        self._audio_q.put_nowait(_STOP)

    async def close(self) -> None:
        self._log_summary_once()
        task = self._scoped_drain_task
        self._scoped_drain_task = None
        if task is not None and task is not asyncio.current_task():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        await self.stop()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        self._event_projection.close()

    async def events(self) -> AsyncIterator[STTBackendTranscriptEvent]:
        async for event in self._event_projection.events():
            yield event

    def _log_summary_once(self) -> None:
        if self.stream_label != "peer" or self._summary_logged:
            return
        self._summary_logged = True
        total_finals_seen = self._emitted_finals + self._empty_final_acks
        empty_ratio = self._empty_final_acks / total_finals_seen if total_finals_seen > 0 else 0.0
        logger.info(
            "[STT][peer] Session summary: emitted_finals=%s empty_final_acks=%s total_finals_seen=%s empty_ratio=%.3f",
            self._emitted_finals,
            self._empty_final_acks,
            total_finals_seen,
            empty_ratio,
        )
