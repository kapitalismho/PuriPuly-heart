"""Qwen ASR Realtime STT Backend using DashScope SDK.

WebSocket-based Speech-to-Text using Alibaba's qwen3-asr-flash-realtime model.
Uses Manual Mode (no server VAD) for consistent behavior with local VAD control.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import AsyncIterator

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

    async def open_session(self) -> STTBackendSession:
        if self.sample_rate_hz not in (8000, 16000):
            raise ValueError("sample_rate_hz must be 8000 or 16000")
        if not self.api_key:
            raise ValueError("api_key must be non-empty")
        if self.connect_timeout_s <= 0:
            raise ValueError("connect_timeout_s must be > 0")

        session = _QwenASRSession(
            api_key=self.api_key,
            model=self.model,
            language=self.language,
            endpoint=self.endpoint,
            sample_rate_hz=self.sample_rate_hz,
            connect_timeout_s=self.connect_timeout_s,
        )
        await session.start()
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


@dataclass(slots=True)
class _QwenASRSession(STTBackendSession):
    """Internal session using DashScope SDK with threading."""

    api_key: str
    model: str
    language: str
    endpoint: str
    sample_rate_hz: int
    connect_timeout_s: float

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

    def __post_init__(self) -> None:
        self._events = asyncio.Queue()
        self._audio_q = queue.Queue()
        self._connected = threading.Event()

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
                        event_type = response.get("type", "")

                        if event_type == "session.created":
                            session_id = response.get("session", {}).get("id", "unknown")
                            logger.debug(f"Qwen ASR: Session created: {session_id}")

                        elif event_type == "conversation.item.input_audio_transcription.completed":
                            # Final transcript
                            transcript = response.get("transcript", "").strip()
                            if transcript:
                                logger.info(
                                    "[STT] Qwen ASR transcript final text_len=%s",
                                    len(transcript),
                                )
                                event = STTBackendTranscriptEvent(text=transcript, is_final=True)
                                cb_self.parent._put_event(event)

                        elif event_type == "conversation.item.input_audio_transcription.text":
                            # Intermediate result (stash)
                            text = response.get("text", "").strip()
                            stash = response.get("stash", "").strip()
                            if text or stash:
                                logger.debug(
                                    "Qwen ASR: Intermediate text_len=%s stash_len=%s",
                                    len(text),
                                    len(stash),
                                )

                        elif event_type == "input_audio_buffer.committed":
                            logger.debug("Qwen ASR: Audio buffer committed")

                        elif event_type == "error":
                            error_msg = response.get("error", {}).get("message", "Unknown error")
                            logger.warning(f"Qwen ASR error: {error_msg}")
                            if not cb_self.parent._stopped:
                                cb_self.parent._report_error(
                                    RuntimeError(f"Qwen ASR error: {error_msg}")
                                )
                                cb_self.parent._stopped = True
                                cb_self.parent._signal_stop()

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
                    if self._stopped:
                        break
                    # Check if keepalive needed
                    if time.monotonic() - last_activity > KEEPALIVE_INTERVAL:
                        try:
                            send_keepalive_silence()
                        except Exception as e:
                            logger.warning(f"Keepalive failed: {e}")
                    continue

                if data is _STOP:
                    logger.debug(f"Qwen ASR: Stop signal received after {audio_chunks_sent} chunks")
                    break

                if data is _COMMIT:
                    try:
                        conversation.commit()
                        logger.info("[STT] Commit sent to Qwen ASR (finalize)")
                    except Exception as e:
                        logger.warning(f"Failed to send commit: {e}")
                    continue

                if isinstance(data, bytes):
                    try:
                        # Qwen ASR requires base64-encoded audio
                        audio_b64 = base64.b64encode(data).decode("ascii")
                        conversation.append_audio(audio_b64)
                        audio_chunks_sent += 1
                        last_activity = time.monotonic()  # Update activity time
                        if audio_chunks_sent == 1:
                            logger.info(
                                f"[STT] First audio chunk sent to Qwen ASR ({len(data)} bytes)"
                            )
                        elif audio_chunks_sent % 50 == 0:
                            logger.debug(f"[STT] Audio chunks sent: {audio_chunks_sent}")
                    except Exception as e:
                        logger.warning(f"Failed to send audio: {e}")
                        break

            # Close conversation
            try:
                conversation.close()
            except Exception as e:
                logger.debug(f"Error closing conversation: {e}")

        except BaseException as exc:
            logger.exception("Qwen ASR SDK thread error")
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

    def _put_event(self, event: STTBackendTranscriptEvent | BaseException | None) -> None:
        """Thread-safe event posting to the asyncio queue."""
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._events.put_nowait, event)

    async def send_audio(self, pcm16le: bytes) -> None:
        if self._stopped:
            return
        self._audio_q.put_nowait(pcm16le)

    async def on_speech_end(self, *, trailing_silence_ms: int | None = None) -> None:
        """Handle end of speech: top up trailing silence if needed, then commit."""
        if self._stopped:
            return

        min_silence_ms = 100
        existing_ms = max(int(trailing_silence_ms or 0), 0)
        missing_ms = max(min_silence_ms - existing_ms, 0)

        if missing_ms > 0:
            import numpy as np

            silence_samples = int(self.sample_rate_hz * (missing_ms / 1000.0))
            if silence_samples > 0:
                silence = np.zeros(silence_samples, dtype=np.float32)
                pcm16 = (silence * 32767).astype(np.int16).tobytes()
                self._audio_q.put_nowait(pcm16)
                logger.info(
                    "[STT] Trailing silence sent (%sms, %s samples)", missing_ms, silence_samples
                )

        # Send commit (equivalent to Deepgram's Finalize)
        self._audio_q.put_nowait(_COMMIT)

    async def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        self._signal_stop()

    async def close(self) -> None:
        await self.stop()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    async def events(self) -> AsyncIterator[STTBackendTranscriptEvent]:
        while True:
            item = await self._events.get()
            if item is None:
                return
            if isinstance(item, BaseException):
                raise item
            yield item
