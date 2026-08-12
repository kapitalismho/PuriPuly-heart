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
from collections import deque
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Sequence

from puripuly_heart.core.stt.backend import (
    STTBackend,
    STTBackendSession,
    STTBackendTranscriptEvent,
)

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
    finalize_timeout_s: float = 1.0

    async def open_session(self) -> STTBackendSession:
        if self.sample_rate_hz not in (8000, 16000):
            raise ValueError("sample_rate_hz must be 8000 or 16000")
        if not self.api_key:
            raise ValueError("api_key must be non-empty")
        if self.connect_timeout_s <= 0:
            raise ValueError("connect_timeout_s must be > 0")
        if self.finalize_timeout_s <= 0:
            raise ValueError("finalize_timeout_s must be > 0")

        session = _DeepgramSDKSession(
            api_key=self.api_key,
            model=self.model,
            language=self.language,
            sample_rate_hz=self.sample_rate_hz,
            connect_timeout_s=self.connect_timeout_s,
            keyterms=list(self.keyterms),
            stream_label=self.stream_label,
            finalize_timeout_s=self.finalize_timeout_s,
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


@dataclass(slots=True)
class _PendingFinalize:
    sequence: int
    awaiting_fence: bool
    event: STTBackendTranscriptEvent | None = None
    fence_received: threading.Event = field(default_factory=threading.Event)


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
    finalize_timeout_s: float = 1.0

    _events: asyncio.Queue[STTBackendTranscriptEvent | BaseException | None] = field(
        init=False, repr=False
    )
    _audio_q: queue.Queue[bytes | object] = field(init=False, repr=False)
    _thread: threading.Thread | None = field(init=False, default=None, repr=False)
    _stopped: bool = field(init=False, default=False)
    _loop: asyncio.AbstractEventLoop | None = field(init=False, default=None, repr=False)
    _connected: threading.Event = field(init=False, repr=False)
    _connect_started_at: float | None = field(init=False, default=None, repr=False)
    _error_reported: bool = field(init=False, default=False, repr=False)
    _emitted_finals: int = field(init=False, default=0, repr=False)
    _empty_final_acks: int = field(init=False, default=0, repr=False)
    _summary_logged: bool = field(init=False, default=False, repr=False)
    _finalize_lock: threading.Lock = field(init=False, repr=False)
    _pending_finalizes: deque[_PendingFinalize] = field(init=False, repr=False)
    _segment_buffer: list[str] = field(init=False, repr=False)
    _next_finalize_sequence: int = field(init=False, default=1, repr=False)
    _outstanding_finalizes: int = field(init=False, default=0, repr=False)
    _all_finalized: asyncio.Event = field(init=False, repr=False)
    _accept_results: bool = field(init=False, default=True, repr=False)
    _active_finalize: _PendingFinalize | None = field(init=False, default=None, repr=False)
    _wire_audio_open: bool = field(init=False, default=False, repr=False)
    _segment_ranges: set[tuple[float, float]] = field(init=False, repr=False)
    _finalize_fence_ranges: deque[tuple[float, float]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._events = asyncio.Queue()
        self._audio_q = queue.Queue()
        self._connected = threading.Event()
        self._finalize_lock = threading.Lock()
        self._pending_finalizes = deque()
        self._segment_buffer = []
        self._segment_ranges = set()
        self._finalize_fence_ranges = deque(maxlen=32)
        self._all_finalized = asyncio.Event()
        self._all_finalized.set()

    def _supports_keyterms(self) -> bool:
        return self.model.strip().lower() == _DEEPGRAM_KEYTERM_MODEL

    @staticmethod
    def _result_range(result: Any) -> tuple[float, float] | None:
        start = getattr(result, "start", None)
        duration = getattr(result, "duration", None)
        if not isinstance(start, (int, float)) or not isinstance(duration, (int, float)):
            return None
        return float(start), float(duration)

    def _build_transcript_events(self, result: Any) -> list[STTBackendTranscriptEvent]:
        if not hasattr(result, "channel") or not hasattr(result.channel, "alternatives"):
            return []
        if not result.channel.alternatives:
            return []

        alternative = result.channel.alternatives[0]
        transcript = str(getattr(alternative, "transcript", "") or "").strip()
        speech_final = getattr(result, "speech_final", False)
        is_final = getattr(result, "is_final", False)
        from_finalize = getattr(result, "from_finalize", False)
        result_range = self._result_range(result)
        logger.info(
            "[STT] Transcript metadata text_len=%s is_final=%s speech_final=%s "
            "from_finalize=%s start=%s duration=%s",
            len(transcript),
            is_final,
            speech_final,
            from_finalize,
            result_range[0] if result_range is not None else None,
            result_range[1] if result_range is not None else None,
        )
        with self._finalize_lock:
            if not self._accept_results:
                return []
            pending = self._active_finalize
            if from_finalize:
                if pending is None:
                    logger.debug("[STT] Deepgram finalize fence ignored without pending request")
                    return []
                if pending.event is not None:
                    logger.debug(
                        "[STT] Deepgram duplicate finalize fence ignored sequence=%s",
                        pending.sequence,
                    )
                    return []
                if (
                    result_range is not None
                    and result_range in self._finalize_fence_ranges
                ):
                    logger.warning(
                        "[STT] Deepgram stale finalize fence ignored sequence=%s start=%s duration=%s",
                        pending.sequence,
                        result_range[0],
                        result_range[1],
                    )
                    return []
            if (is_final or speech_final) and transcript and self._wire_audio_open:
                if result_range is None or result_range not in self._segment_ranges:
                    self._segment_buffer.append(transcript)
                    if result_range is not None:
                        self._segment_ranges.add(result_range)
            if not from_finalize:
                return []
            text = " ".join(self._segment_buffer).strip()
            self._segment_buffer.clear()
            self._segment_ranges.clear()
            self._wire_audio_open = False
            if result_range is not None:
                self._finalize_fence_ranges.append(result_range)
            pending.awaiting_fence = False
            pending.event = STTBackendTranscriptEvent(text=text, is_final=True)
            pending.fence_received.set()
            return []

    async def _emit_test_final(
        self,
        *,
        text: str,
    ) -> None:
        await self._events.put(
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
                        for event in self._build_transcript_events(result):
                            self._put_event(event)
                    except Exception as e:
                        logger.debug(f"Deepgram parse error: {e}")

                def on_error(error: Any) -> None:
                    logger.warning(f"Deepgram error: {error}")
                    if not self._stopped:
                        self._fail_finalize_session(
                            RuntimeError(f"Deepgram error: {error}"),
                            status="provider_error",
                        )

                def on_close(close_event: Any) -> None:
                    _ = close_event
                    logger.debug("Deepgram: Connection closed")
                    if not self._stopped:
                        self._fail_finalize_session(
                            RuntimeError("Deepgram connection closed"),
                            status="connection_closed",
                        )

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
                audio_bytes_since_finalize = 0
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
                        self._put_event(None)  # Signal consumer immediately before SDK cleanup
                        break

                    if data is _FINALIZE:
                        pending = self._register_finalize(
                            has_audio=audio_bytes_since_finalize > 0
                        )
                        audio_bytes_since_finalize = 0
                        if not pending.awaiting_fence:
                            continue
                        try:
                            connection.send_control(ListenV1ControlMessage(type="Finalize"))
                            logger.info("[STT] Finalize message sent to Deepgram")
                        except Exception as e:
                            logger.warning(f"Failed to send Finalize: {e}")
                            self._fail_finalize_session(
                                RuntimeError("Deepgram finalize send failed"),
                                status="finalize_send_failed",
                            )
                            break
                        if not pending.fence_received.wait(self.finalize_timeout_s):
                            self._fail_finalize_session(
                                RuntimeError(
                                    f"Deepgram finalize fence timeout sequence={pending.sequence}"
                                ),
                                status="finalize_timeout",
                                sequence=pending.sequence,
                            )
                            break
                        for event in self._complete_active_finalize(pending):
                            self._put_event(event)
                        continue

                    if isinstance(data, bytes):
                        try:
                            with self._finalize_lock:
                                self._wire_audio_open = True
                            connection.send_media(data)
                            audio_chunks_sent += 1
                            audio_bytes_since_finalize += len(data)
                            if audio_chunks_sent == 1:
                                logger.info(
                                    f"[STT] First audio chunk sent to Deepgram ({len(data)} bytes)"
                                )
                            elif audio_chunks_sent % 50 == 0:
                                logger.debug(f"[STT] Audio chunks sent: {audio_chunks_sent}")
                        except Exception as e:
                            logger.warning(f"Failed to send audio: {e}")
                            self._fail_finalize_session(
                                RuntimeError("Deepgram audio send failed"),
                                status="audio_send_failed",
                            )
                            break

        except BaseException as exc:
            logger.exception("Deepgram SDK thread error")
            self._fail_finalize_session(exc, status="thread_error")
        finally:
            self._put_event(None)

    def _report_error(self, exc: BaseException) -> None:
        if self._error_reported:
            return
        self._error_reported = True
        self._put_event(exc)

    def _put_event(self, event: STTBackendTranscriptEvent | BaseException | None) -> None:
        """Thread-safe event posting to the asyncio queue."""
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._events.put_nowait, event)

    def _set_all_finalized(self) -> None:
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._set_all_finalized_if_idle)
        elif self._outstanding_finalizes == 0:
            self._all_finalized.set()

    def _set_all_finalized_if_idle(self) -> None:
        with self._finalize_lock:
            if self._outstanding_finalizes == 0:
                self._all_finalized.set()

    def _record_terminal(self, event: STTBackendTranscriptEvent) -> None:
        if self.stream_label != "peer":
            return
        if event.text:
            self._emitted_finals += 1
        else:
            self._empty_final_acks += 1

    def _drain_ready_finalizes_locked(self) -> list[STTBackendTranscriptEvent]:
        ready: list[STTBackendTranscriptEvent] = []
        while self._pending_finalizes and self._pending_finalizes[0].event is not None:
            pending = self._pending_finalizes.popleft()
            event = pending.event
            self._outstanding_finalizes -= 1
            self._record_terminal(event)
            ready.append(event)
        if self._outstanding_finalizes == 0:
            self._set_all_finalized()
        return ready

    def _register_finalize(self, *, has_audio: bool) -> _PendingFinalize:
        ready: list[STTBackendTranscriptEvent] = []
        with self._finalize_lock:
            pending = _PendingFinalize(
                sequence=self._next_finalize_sequence,
                awaiting_fence=has_audio,
            )
            self._next_finalize_sequence += 1
            self._pending_finalizes.append(pending)
            if has_audio:
                if self._active_finalize is not None:
                    raise RuntimeError("Deepgram finalize serialization violated")
                self._wire_audio_open = True
                self._active_finalize = pending
            else:
                pending.event = STTBackendTranscriptEvent(text="", is_final=True)
                ready = self._drain_ready_finalizes_locked()
        for event in ready:
            self._put_event(event)
        return pending

    def _complete_active_finalize(
        self,
        pending: _PendingFinalize,
    ) -> list[STTBackendTranscriptEvent]:
        with self._finalize_lock:
            if self._active_finalize is not pending or pending.event is None:
                return []
            self._active_finalize = None
            return self._drain_ready_finalizes_locked()

    def _fail_finalize_session(
        self,
        exc: BaseException,
        *,
        status: str,
        sequence: int | None = None,
    ) -> None:
        ready: list[STTBackendTranscriptEvent] = []
        with self._finalize_lock:
            if not self._accept_results:
                return
            if sequence is not None and not any(
                pending.sequence == sequence and pending.awaiting_fence
                for pending in self._pending_finalizes
            ):
                return
            self._accept_results = False
            first_unresolved = True
            for pending in self._pending_finalizes:
                if pending.event is not None:
                    continue
                text = " ".join(self._segment_buffer).strip() if first_unresolved else ""
                first_unresolved = False
                pending.awaiting_fence = False
                pending.event = STTBackendTranscriptEvent(text=text, is_final=True)
                pending.fence_received.set()
            self._segment_buffer.clear()
            self._segment_ranges.clear()
            self._wire_audio_open = False
            self._active_finalize = None
            ready = self._drain_ready_finalizes_locked()
            while self._outstanding_finalizes > 0:
                event = STTBackendTranscriptEvent(text="", is_final=True)
                self._record_terminal(event)
                ready.append(event)
                self._outstanding_finalizes -= 1
            if self._outstanding_finalizes == 0:
                self._set_all_finalized()
        logger.warning("[STT] Deepgram finalize session failed status=%s", status)
        for event in ready:
            self._put_event(event)
        self._report_error(exc)
        self._stopped = True
        self._discard_audio_queue()
        self._audio_q.put_nowait(_STOP)

    def _discard_audio_queue(self) -> None:
        while True:
            try:
                self._audio_q.get_nowait()
            except queue.Empty:
                return

    async def send_audio(self, pcm16le: bytes) -> None:
        if self._stopped:
            return
        self._audio_q.put_nowait(pcm16le)

    async def on_speech_end(self, *, trailing_silence_ms: int | None = None) -> None:
        """Handle end of speech and finalize."""
        if self._stopped:
            return

        with self._finalize_lock:
            self._outstanding_finalizes += 1
            self._all_finalized.clear()

        existing_ms = max(int(trailing_silence_ms or 0), 0)
        missing_ms = 0

        logger.info(
            "[STT][Tail] provider=deepgram observed_tail_ms=%s injected_padding_ms=%s "
            "declared_trim_ms=0 boundary_wait_ms=unknown",
            existing_ms,
            missing_ms,
        )
        self._audio_q.put_nowait(_FINALIZE)

    async def stop(self) -> None:
        self._log_summary_once()
        if self._stopped:
            return
        try:
            await asyncio.wait_for(
                self._all_finalized.wait(),
                timeout=self.finalize_timeout_s + 0.25,
            )
        except asyncio.TimeoutError:
            self._fail_finalize_session(
                RuntimeError("Deepgram finalize drain timeout"),
                status="stop_timeout",
            )
            return
        self._stopped = True
        self._audio_q.put_nowait(_STOP)

    async def abort_for_toggle_off(self) -> None:
        self._stopped = True
        with self._finalize_lock:
            self._accept_results = False
            for pending in self._pending_finalizes:
                pending.fence_received.set()
            self._pending_finalizes.clear()
            self._segment_buffer.clear()
            self._segment_ranges.clear()
            self._wire_audio_open = False
            self._active_finalize = None
            self._outstanding_finalizes = 0
            self._all_finalized.set()
        self._discard_audio_queue()
        self._audio_q.put_nowait(_STOP)

    async def close(self) -> None:
        self._log_summary_once()
        await self.stop()
        if self._thread is not None:
            await asyncio.to_thread(self._thread.join, 5.0)
            self._thread = None

    async def events(self) -> AsyncIterator[STTBackendTranscriptEvent]:
        while True:
            item = await self._events.get()
            if item is None:
                return
            if isinstance(item, BaseException):
                raise item
            yield item

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
