"""Qwen ASR Realtime STT Backend using DashScope SDK.

WebSocket-based Speech-to-Text using Alibaba's qwen3-asr-flash-realtime model.
Uses Manual Mode (no server VAD) for consistent behavior with local VAD control.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import logging
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from puripuly_heart.core.speech_boundary import SpeechBoundaryReason, boundary_wait_ms
from puripuly_heart.core.stt.backend import (
    STTBackend,
    STTBackendSession,
    STTBackendTranscriptEvent,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class QwenASRRealtimeSTTBackend(STTBackend):
    """Qwen ASR Realtime STT Backend using DashScope SDK."""

    api_key: str
    language: str  # Required: passed from wiring.py via get_qwen_asr_language()
    model: str = "qwen3-asr-flash-realtime"
    endpoint: str = "wss://dashscope-intl.aliyuncs.com/api-ws/v1/realtime"
    sample_rate_hz: int = 16000
    connect_timeout_s: float = 5.0
    finish_timeout_s: float = 1.0

    async def open_session(self) -> STTBackendSession:
        if self.sample_rate_hz not in (8000, 16000):
            raise ValueError("sample_rate_hz must be 8000 or 16000")
        if not self.api_key:
            raise ValueError("api_key must be non-empty")
        if self.connect_timeout_s <= 0:
            raise ValueError("connect_timeout_s must be > 0")
        if self.finish_timeout_s <= 0:
            raise ValueError("finish_timeout_s must be > 0")

        session = _QwenASRSession(
            api_key=self.api_key,
            model=self.model,
            language=self.language,
            endpoint=self.endpoint,
            sample_rate_hz=self.sample_rate_hz,
            connect_timeout_s=self.connect_timeout_s,
            finish_timeout_s=self.finish_timeout_s,
        )
        try:
            await session.start()
        except BaseException:
            with contextlib.suppress(BaseException):
                await session.abort_for_toggle_off()
            with contextlib.suppress(BaseException):
                await session.close()
            raise
        return session

    @staticmethod
    async def verify_api_key(api_key: str) -> bool:
        """Verify Alibaba API key by making a test request."""
        if not api_key:
            return False

        # Use the same verification as Qwen LLM (shared API key)
        from puripuly_heart.providers.llm.qwen import QwenLLMProvider

        return await QwenLLMProvider.verify_api_key(api_key)


_STOP = object()
_COMMIT = object()
_END_SESSION = object()


@dataclass(slots=True)
class _PendingCommit:
    sequence: int
    item_id: str | None = None
    terminal_status: str | None = None
    event: STTBackendTranscriptEvent | None = None


@dataclass(slots=True)
class _QwenASRSession(STTBackendSession):
    """Internal session using DashScope SDK with threading."""

    api_key: str
    model: str
    language: str
    endpoint: str
    sample_rate_hz: int
    connect_timeout_s: float
    finish_timeout_s: float = 1.0

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
    _connect_error: BaseException | None = field(init=False, default=None, repr=False)
    _commit_lock: threading.Lock = field(init=False, repr=False)
    _pending_commits: deque[_PendingCommit] = field(init=False, repr=False)
    _terminal_item_ids: set[str] = field(init=False, repr=False)
    _terminal_event_ids: set[str] = field(init=False, repr=False)
    _next_commit_sequence: int = field(init=False, default=1, repr=False)
    _accept_terminals: bool = field(init=False, default=True, repr=False)

    def __post_init__(self) -> None:
        self._events = asyncio.Queue()
        self._audio_q = queue.Queue()
        self._connected = threading.Event()
        self._commit_lock = threading.Lock()
        self._pending_commits = deque()
        self._terminal_item_ids = set()
        self._terminal_event_ids = set()

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._connect_started_at = time.monotonic()
        self._thread = threading.Thread(target=self._run_sync, name="qwen-asr-sdk", daemon=True)
        self._thread.start()

        # Wait for connection to be established
        logger.info("[STT] Qwen ASR connecting (timeout=%.1fs)", self.connect_timeout_s)
        connected = await asyncio.to_thread(self._connected.wait, self.connect_timeout_s)
        if not connected or self._connect_error is not None:
            exc = self._connect_error or RuntimeError("Qwen ASR SDK connection timeout")
            logger.warning("[STT] Qwen ASR connection failed: %s", exc)
            await self.stop()
            raise exc

    def _run_sync(self) -> None:
        """Run Qwen ASR SDK connection in a separate thread."""
        try:
            import dashscope
            from dashscope.audio.qwen_omni import (
                MultiModality,
                OmniRealtimeCallback,
                OmniRealtimeConversation,
            )
            from dashscope.audio.qwen_omni.omni_realtime import TranscriptionParams

            # Set API key
            dashscope.api_key = self.api_key

            class Callback(OmniRealtimeCallback):
                def __init__(cb_self, parent: "_QwenASRSession"):
                    cb_self.parent = parent
                    cb_self.conversation = None

                def on_open(cb_self):
                    logger.debug("Qwen ASR: Connection opened")
                    if cb_self.parent._connect_started_at is not None:
                        elapsed = time.monotonic() - cb_self.parent._connect_started_at
                        logger.info("[STT] Qwen ASR connected in %.2fs", elapsed)
                    cb_self.parent._connected.set()

                def on_close(cb_self, code, msg):
                    logger.debug(f"Qwen ASR: Connection closed, code: {code}, msg: {msg}")
                    if not cb_self.parent._stopped:
                        cb_self.parent._report_error(
                            RuntimeError(f"Qwen ASR connection closed: {code} {msg}")
                        )
                        cb_self.parent._stopped = True
                        cb_self.parent._signal_stop()

                def on_event(cb_self, response):
                    try:
                        cb_self.parent._handle_provider_event(response)

                    except Exception as e:
                        logger.debug(f"Qwen ASR callback error: {e}")

            callback = Callback(self)

            # Create conversation with Manual Mode (no server VAD)
            conversation = OmniRealtimeConversation(
                model=self.model,
                url=self.endpoint,
                callback=callback,
            )
            callback.conversation = conversation

            # Connect
            conversation.connect()

            # Update session configuration (Manual Mode)
            transcription_params = TranscriptionParams(
                language=self.language, sample_rate=self.sample_rate_hz, input_audio_format="pcm"
            )

            conversation.update_session(
                output_modalities=[MultiModality.TEXT],
                enable_input_audio_transcription=True,
                enable_turn_detection=False,  # Manual Mode: no server VAD
                transcription_params=transcription_params,
            )

            # Signal that connection is established
            self._connected.set()
            logger.debug("Qwen ASR SDK connection and session update complete")

            # Keepalive: send 100ms silence every 50 seconds to prevent 60s timeout
            import numpy as np

            last_activity = time.monotonic()
            KEEPALIVE_INTERVAL = 50.0  # seconds
            SILENCE_DURATION_MS = 100  # milliseconds

            def send_keepalive_silence():
                """Send 100ms of silence as keepalive."""
                nonlocal last_activity
                silence_samples = int(self.sample_rate_hz * SILENCE_DURATION_MS / 1000)
                silence = np.zeros(silence_samples, dtype=np.int16).tobytes()
                audio_b64 = base64.b64encode(silence).decode("ascii")
                conversation.append_audio(audio_b64)
                last_activity = time.monotonic()
                logger.debug(f"[STT] Keepalive silence sent ({SILENCE_DURATION_MS}ms)")

            # Audio sending loop
            audio_chunks_sent = 0
            while True:
                try:
                    data = self._audio_q.get(timeout=0.1)
                except queue.Empty:
                    if not self._stopped and time.monotonic() - last_activity > KEEPALIVE_INTERVAL:
                        try:
                            send_keepalive_silence()
                        except Exception as e:
                            logger.warning(f"Keepalive failed: {e}")
                    continue

                if data is _STOP:
                    logger.debug(f"Qwen ASR: Stop signal received after {audio_chunks_sent} chunks")
                    break

                if data is _END_SESSION:
                    try:
                        conversation.end_session(timeout=self.finish_timeout_s)
                    except Exception as exc:
                        logger.debug("Qwen ASR end_session failed: %s", exc)
                    finally:
                        self._resolve_all_pending_empty("session_finished_without_terminal")
                    break

                if data is _COMMIT:
                    if not self._send_commit(conversation):
                        break
                    continue

                if isinstance(data, bytes):
                    if not self._append_audio(conversation, data):
                        break
                    audio_chunks_sent += 1
                    last_activity = time.monotonic()
                    if audio_chunks_sent == 1:
                        logger.info(f"[STT] First audio chunk sent to Qwen ASR ({len(data)} bytes)")
                    elif audio_chunks_sent % 50 == 0:
                        logger.debug(f"[STT] Audio chunks sent: {audio_chunks_sent}")

            # Close conversation
            try:
                conversation.close()
            except Exception as e:
                logger.debug(f"Error closing conversation: {e}")

        except BaseException as exc:
            logger.exception("Qwen ASR SDK thread error")
            self._resolve_all_pending_empty("session_error")
            self._report_error(exc)
        finally:
            self._put_event(None)

    def _report_error(self, exc: BaseException) -> None:
        if self._error_reported:
            return
        self._error_reported = True
        if self._connect_error is None:
            self._connect_error = exc
        self._connected.set()
        self._put_event(exc)

    def _signal_stop(self) -> None:
        try:
            self._audio_q.put_nowait(_STOP)
        except Exception:
            pass

    @staticmethod
    def _response_item_id(response: dict[str, Any]) -> str | None:
        item_id = response.get("item_id")
        if not item_id and isinstance(response.get("item"), dict):
            item_id = response["item"].get("id")
        value = str(item_id or "").strip()
        return value or None

    def _register_commit(self) -> _PendingCommit | None:
        with self._commit_lock:
            if not self._accept_terminals:
                return None
            pending = _PendingCommit(sequence=self._next_commit_sequence)
            self._next_commit_sequence += 1
            self._pending_commits.append(pending)
            return pending

    def _send_commit(self, conversation: Any) -> bool:
        pending = self._register_commit()
        if pending is None:
            return False
        try:
            conversation.commit()
            logger.info("[STT] Commit sent to Qwen ASR (finalize)")
            return True
        except Exception as exc:
            logger.warning("Failed to send commit: %s", exc)
            self._fail_worker_session(
                RuntimeError("Qwen ASR commit send failed"),
                status="commit_send_failed",
            )
            return False

    def _append_audio(self, conversation: Any, data: bytes) -> bool:
        try:
            audio_b64 = base64.b64encode(data).decode("ascii")
            conversation.append_audio(audio_b64)
            return True
        except Exception as exc:
            logger.warning("Failed to send audio: %s", exc)
            self._fail_worker_session(
                RuntimeError("Qwen ASR audio send failed"),
                status="audio_send_failed",
            )
            return False

    def _assign_committed_item(self, response: dict[str, Any]) -> None:
        item_id = self._response_item_id(response)
        event_id = str(response.get("event_id") or "").strip() or "none"
        with self._commit_lock:
            pending = next(
                (
                    item
                    for item in self._pending_commits
                    if item.item_id is None and item.terminal_status is None
                ),
                None,
            )
            if pending is not None and item_id is not None:
                pending.item_id = item_id
            sequence = pending.sequence if pending is not None else None
        logger.info(
            "[STT] Qwen ASR committed sequence=%s event_id=%s item_id=%s",
            sequence,
            event_id,
            item_id or "none",
        )

    def _resolve_terminal(
        self,
        response: dict[str, Any],
        *,
        status: str,
        text: str,
    ) -> None:
        item_id = self._response_item_id(response)
        event_id = str(response.get("event_id") or "").strip() or None
        ready: list[STTBackendTranscriptEvent] = []
        with self._commit_lock:
            if not self._accept_terminals:
                return
            if event_id is not None and event_id in self._terminal_event_ids:
                logger.debug("[STT] Qwen ASR duplicate terminal ignored event_id=%s", event_id)
                return
            if item_id is not None and item_id in self._terminal_item_ids:
                logger.debug("[STT] Qwen ASR duplicate terminal ignored item_id=%s", item_id)
                return
            pending = None
            if item_id is not None:
                pending = next(
                    (item for item in self._pending_commits if item.item_id == item_id),
                    None,
                )
            if pending is None:
                pending = next(
                    (
                        item
                        for item in self._pending_commits
                        if item.item_id is None and item.terminal_status is None
                    ),
                    None,
                )
                if pending is not None and item_id is not None:
                    pending.item_id = item_id
            if pending is None or pending.terminal_status is not None:
                logger.debug(
                    "[STT] Qwen ASR terminal ignored without pending commit item_id=%s status=%s",
                    item_id or "none",
                    status,
                )
                return
            if event_id is not None:
                self._terminal_event_ids.add(event_id)
            pending.terminal_status = status
            pending.event = STTBackendTranscriptEvent(text=text, is_final=True)
            ready = self._drain_ready_terminals_locked()
        for event in ready:
            self._put_event(event)

    def _resolve_all_pending_empty(self, status: str) -> None:
        ready: list[STTBackendTranscriptEvent] = []
        resolved = 0
        with self._commit_lock:
            if not self._accept_terminals:
                return
            for pending in self._pending_commits:
                if pending.terminal_status is None:
                    pending.terminal_status = status
                    pending.event = STTBackendTranscriptEvent(text="", is_final=True)
                    resolved += 1
            ready = self._drain_ready_terminals_locked()
        if resolved:
            logger.warning(
                "[STT] Qwen ASR unresolved commits closed status=%s count=%s",
                status,
                resolved,
            )
        for event in ready:
            self._put_event(event)

    def _discard_audio_queue(self) -> None:
        while True:
            try:
                self._audio_q.get_nowait()
            except queue.Empty:
                return

    def _fail_worker_session(self, exc: BaseException, *, status: str) -> None:
        self._resolve_all_pending_empty(status)
        with self._commit_lock:
            self._accept_terminals = False
        self._discard_audio_queue()
        self._report_error(exc)
        self._stopped = True

    def _drain_ready_terminals_locked(self) -> list[STTBackendTranscriptEvent]:
        ready: list[STTBackendTranscriptEvent] = []
        while self._pending_commits and self._pending_commits[0].event is not None:
            pending = self._pending_commits.popleft()
            if pending.item_id is not None:
                self._terminal_item_ids.add(pending.item_id)
            ready.append(pending.event)
        return ready

    def _handle_provider_event(self, response: dict[str, Any]) -> None:
        event_type = str(response.get("type", "") or "")
        if event_type == "session.created":
            session_id = response.get("session", {}).get("id", "unknown")
            logger.debug("Qwen ASR: Session created: %s", session_id)
            return
        if event_type == "input_audio_buffer.committed":
            self._assign_committed_item(response)
            return
        if event_type == "conversation.item.input_audio_transcription.completed":
            transcript = str(response.get("transcript", "") or "").strip()
            logger.info(
                "[STT] Qwen ASR transcript terminal status=completed text_len=%s item_id=%s",
                len(transcript),
                self._response_item_id(response) or "none",
            )
            self._resolve_terminal(response, status="completed", text=transcript)
            return
        if event_type == "conversation.item.input_audio_transcription.failed":
            error = response.get("error")
            logger.warning(
                "[STT] Qwen ASR transcript terminal status=failed item_id=%s error=%s",
                self._response_item_id(response) or "none",
                error,
            )
            self._resolve_terminal(response, status="failed", text="")
            return
        if event_type == "conversation.item.input_audio_transcription.text":
            text = str(response.get("text", "") or "").strip()
            stash = str(response.get("stash", "") or "").strip()
            if text or stash:
                logger.debug(
                    "Qwen ASR: Intermediate text_len=%s stash_len=%s",
                    len(text),
                    len(stash),
                )
            return
        if event_type == "session.finished":
            logger.info("[STT] Qwen ASR session finished")
            return
        if event_type == "error":
            error_msg = response.get("error", {}).get("message", "Unknown error")
            logger.warning("Qwen ASR error: %s", error_msg)
            self._resolve_all_pending_empty("session_error")
            if not self._stopped:
                self._report_error(RuntimeError(f"Qwen ASR error: {error_msg}"))
                self._stopped = True
                self._signal_stop()

    def _put_event(self, event: STTBackendTranscriptEvent | BaseException | None) -> None:
        """Thread-safe event posting to the asyncio queue."""
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._events.put_nowait, event)

    async def send_audio(self, pcm16le: bytes) -> None:
        if self._stopped:
            return
        self._audio_q.put_nowait(pcm16le)

    async def on_speech_end(
        self,
        *,
        trailing_silence_ms: int | None = None,
        reason: SpeechBoundaryReason | None = None,
    ) -> None:
        if self._stopped:
            return

        existing_ms = max(int(trailing_silence_ms or 0), 0)
        missing_ms = 0
        wait_ms = boundary_wait_ms(reason, observed_tail_ms=existing_ms)

        logger.info(
            "[STT][Tail] provider=qwen boundary_reason=%s observed_tail_ms=%s "
            "injected_padding_ms=%s "
            "declared_trim_ms=0 boundary_wait_ms=%s",
            reason,
            existing_ms,
            missing_ms,
            wait_ms,
        )
        self._audio_q.put_nowait(_COMMIT)

    async def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        self._audio_q.put_nowait(_END_SESSION)

    async def abort_for_toggle_off(self) -> None:
        self._stopped = True
        with self._commit_lock:
            self._accept_terminals = False
            self._pending_commits.clear()
        self._discard_audio_queue()
        self._signal_stop()

    async def close(self) -> None:
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
