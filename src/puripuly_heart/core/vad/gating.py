from __future__ import annotations

import contextlib
import logging
import math
import uuid
from dataclasses import dataclass
from typing import Callable, Protocol
from uuid import UUID

import numpy as np

from puripuly_heart.core.audio.diagnostics import compute_audio_frame_metrics
from puripuly_heart.core.audio.format import AudioFrameF32
from puripuly_heart.core.audio.ring_buffer import RingBufferF32
from puripuly_heart.core.speech_boundary import SpeechBoundaryReason

logger = logging.getLogger(__name__)


class VadEngine(Protocol):
    def speech_probability(self, samples: np.ndarray, *, sample_rate_hz: int) -> float: ...
    def reset(self) -> None: ...


@dataclass(frozen=True, slots=True)
class SpeechStart:
    utterance_id: UUID
    pre_roll: np.ndarray
    chunk: np.ndarray


@dataclass(frozen=True, slots=True)
class SpeechChunk:
    utterance_id: UUID
    chunk: np.ndarray


@dataclass(frozen=True, slots=True)
class SpeechEnd:
    utterance_id: UUID
    trailing_silence_ms: int = 0
    reason: SpeechBoundaryReason = "silence"


VadEvent = SpeechStart | SpeechChunk | SpeechEnd


def default_chunk_samples(sample_rate_hz: int) -> int:
    if sample_rate_hz == 16000:
        return 512
    if sample_rate_hz == 8000:
        return 256
    raise ValueError("Silero VAD streaming supports only 8000 or 16000 Hz")


@dataclass(slots=True)
class VadGating:
    engine: VadEngine
    sample_rate_hz: int
    speech_threshold: float
    hangover_chunks: int
    chunk_samples: int
    start_debounce_chunks: int
    start_commit_chunks: int
    max_segment_ms: int | None
    soft_boundary_start_ms: int | None
    soft_pause_ms: int | None
    candidate_log_label: str | None
    diagnostic_event_callback: Callable[[str], object] | None
    diagnostics_enabled: Callable[[], bool] | None
    diagnostic_label: str
    _ring: RingBufferF32
    _in_speech: bool
    _utterance_id: UUID | None
    _silence_run: int
    _pending_start_id: UUID | None
    _pending_start_pre_roll: np.ndarray | None
    _pending_start_prob: float | None
    _pending_start_chunks: list[np.ndarray]
    _pending_debounce_reached: bool
    _speech_chunk_count: int
    _speech_sample_count: int

    def __init__(
        self,
        engine: VadEngine,
        *,
        sample_rate_hz: int,
        ring_buffer_ms: int = 500,
        speech_threshold: float = 0.4,
        hangover_ms: int = 1100,
        max_segment_ms: int | None = None,
        chunk_samples: int | None = None,
        start_debounce_chunks: int = 1,
        start_commit_chunks: int = 1,
        candidate_log_label: str | None = None,
        soft_boundary_start_ms: int | None = None,
        soft_pause_ms: int | None = None,
        diagnostic_event_callback: Callable[[str], object] | None = None,
        diagnostics_enabled: Callable[[], bool] | None = None,
        diagnostic_label: str = "self",
    ) -> None:
        if sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be > 0")
        if ring_buffer_ms <= 0:
            raise ValueError("ring_buffer_ms must be > 0")
        if hangover_ms < 0:
            raise ValueError("hangover_ms must be >= 0")
        if start_debounce_chunks <= 0:
            raise ValueError("start_debounce_chunks must be > 0")
        if start_commit_chunks <= 0:
            raise ValueError("start_commit_chunks must be > 0")
        if start_commit_chunks < start_debounce_chunks:
            raise ValueError("start_commit_chunks must be >= start_debounce_chunks")
        if max_segment_ms is not None and max_segment_ms <= 0:
            raise ValueError("max_segment_ms must be > 0")
        if (soft_boundary_start_ms is None) != (soft_pause_ms is None):
            raise ValueError("soft boundary start and pause must be configured together")
        if soft_boundary_start_ms is not None:
            if soft_boundary_start_ms <= 0:
                raise ValueError("soft_boundary_start_ms must be > 0")
            if soft_pause_ms is None or soft_pause_ms <= 0:
                raise ValueError("soft_pause_ms must be > 0")
            if max_segment_ms is None or soft_boundary_start_ms >= max_segment_ms:
                raise ValueError("soft boundary start must be below max_segment_ms")

        self.engine = engine
        self.sample_rate_hz = sample_rate_hz
        self.speech_threshold = speech_threshold
        self.chunk_samples = chunk_samples or default_chunk_samples(sample_rate_hz)
        self.start_debounce_chunks = start_debounce_chunks
        self.start_commit_chunks = start_commit_chunks
        self.max_segment_ms = max_segment_ms
        self.soft_boundary_start_ms = soft_boundary_start_ms
        self.soft_pause_ms = soft_pause_ms
        self.candidate_log_label = candidate_log_label
        self.diagnostic_event_callback = diagnostic_event_callback
        self.diagnostics_enabled = diagnostics_enabled
        self.diagnostic_label = diagnostic_label

        chunk_ms = (self.chunk_samples / self.sample_rate_hz) * 1000.0
        self.hangover_chunks = int(math.ceil(hangover_ms / chunk_ms)) if hangover_ms > 0 else 0

        capacity_samples = int(self.sample_rate_hz * (ring_buffer_ms / 1000.0))
        self._ring = RingBufferF32(capacity_samples=capacity_samples)

        self._in_speech = False
        self._utterance_id = None
        self._silence_run = 0
        self._pending_start_id = None
        self._pending_start_pre_roll = None
        self._pending_start_prob = None
        self._pending_start_chunks = []
        self._pending_debounce_reached = False
        self._speech_chunk_count = 0
        self._speech_sample_count = 0

    @property
    def in_speech(self) -> bool:
        return self._in_speech

    @property
    def utterance_id(self) -> UUID | None:
        return self._utterance_id

    def reset(self) -> None:
        self.engine.reset()
        self._ring.clear()
        self._in_speech = False
        self._utterance_id = None
        self._silence_run = 0
        self._reset_pending_start()
        self._speech_chunk_count = 0
        self._speech_sample_count = 0

    def process_chunk(self, chunk: np.ndarray) -> list[VadEvent]:
        chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)
        if chunk.size != self.chunk_samples:
            raise ValueError(f"chunk must have {self.chunk_samples} samples")

        prob = self.engine.speech_probability(chunk, sample_rate_hz=self.sample_rate_hz)

        events: list[VadEvent] = []

        if not self._in_speech:
            if prob >= self.speech_threshold:
                events.extend(self._handle_pending_start(chunk, prob))
            else:
                self._drop_pending_start()
            self._ring.append(chunk)
            return events

        # in speech
        events.append(SpeechChunk(self._utterance_id, chunk=chunk.copy()))  # type: ignore[arg-type]
        self._speech_chunk_count += 1
        self._speech_sample_count += int(chunk.size)

        if prob >= self.speech_threshold:
            self._silence_run = 0
            if self._max_segment_reached():
                self._emit_max_duration_end(events)
            self._ring.append(chunk)
            return events

        self._silence_run += 1
        trailing_silence_ms = self._trailing_silence_ms()
        if self._peer_hard_cap_reached():
            self._emit_max_duration_end(
                events,
                trailing_silence_ms=trailing_silence_ms,
            )
        elif self._soft_pause_reached(trailing_silence_ms):
            self._emit_soft_pause_end(events, trailing_silence_ms=trailing_silence_ms)
        elif self._silence_run >= self.hangover_chunks:
            logger.info(
                "[VAD] SpeechEnd: id=%s, trailing_silence_ms=%s",
                str(self._utterance_id)[:8],
                trailing_silence_ms,
            )
            with contextlib.suppress(Exception):
                if self._diagnostics_enabled():
                    speech_audio_ms = self._speech_sample_count * 1000.0 / self.sample_rate_hz
                    assert self.diagnostic_event_callback is not None
                    self.diagnostic_event_callback(
                        f"[AudioDiag][VAD][{self.diagnostic_label}] event=SpeechEnd "
                        f"utterance_id={str(self._utterance_id)[:8]} "
                        f"reason=silence "
                        f"trailing_silence_ms={trailing_silence_ms} "
                        f"speech_audio_ms={speech_audio_ms:.1f} "
                        f"chunk_count={self._speech_chunk_count}"
                    )

            events.append(
                SpeechEnd(
                    self._utterance_id,
                    trailing_silence_ms=trailing_silence_ms,
                    reason="silence",
                )
            )  # type: ignore[arg-type]
            self._reset_active_segment()
            self.engine.reset()

        self._ring.append(chunk)
        return events

    def _max_segment_reached(self) -> bool:
        if self.max_segment_ms is None:
            return False
        speech_audio_ms = self._speech_sample_count * 1000.0 / self.sample_rate_hz
        return speech_audio_ms >= self.max_segment_ms

    def _soft_pause_reached(self, trailing_silence_ms: int) -> bool:
        if self.soft_boundary_start_ms is None or self.soft_pause_ms is None:
            return False
        speech_samples_before_tail = (
            self._speech_sample_count - self._silence_run * self.chunk_samples
        )
        speech_audio_ms_before_tail = speech_samples_before_tail * 1000.0 / self.sample_rate_hz
        return (
            speech_audio_ms_before_tail >= self.soft_boundary_start_ms
            and trailing_silence_ms >= self.soft_pause_ms
        )

    def _peer_hard_cap_reached(self) -> bool:
        return self.soft_boundary_start_ms is not None and self._max_segment_reached()

    def _trailing_silence_ms(self) -> int:
        return int(round(self._silence_run * (self.chunk_samples / self.sample_rate_hz) * 1000.0))

    def _reset_active_segment(self) -> None:
        self._in_speech = False
        self._utterance_id = None
        self._silence_run = 0
        self._speech_chunk_count = 0
        self._speech_sample_count = 0

    def _handle_pending_start(self, chunk: np.ndarray, prob: float) -> list[VadEvent]:
        if self._pending_start_id is None:
            self._pending_start_id = uuid.uuid4()
            self._pending_start_pre_roll = self._ring.get_last_samples(self._ring.capacity_samples)
            self._pending_start_prob = prob
            self._pending_start_chunks = [chunk.copy()]
            self._pending_debounce_reached = self.start_debounce_chunks <= 1
            self._log_candidate("start", prob=prob)
        else:
            self._pending_start_chunks.append(chunk.copy())

        if (
            not self._pending_debounce_reached
            and len(self._pending_start_chunks) >= self.start_debounce_chunks
        ):
            self._pending_debounce_reached = True

        if len(self._pending_start_chunks) < self.start_commit_chunks:
            return []

        utterance_id = self._pending_start_id
        if utterance_id is None:
            return []

        self._in_speech = True
        self._silence_run = 0
        self._utterance_id = utterance_id

        pre_roll = self._pending_start_pre_roll
        if pre_roll is None:
            pre_roll = np.empty((0,), dtype=np.float32)
        start_prob = self._pending_start_prob if self._pending_start_prob is not None else prob
        buffered_chunks = list(self._pending_start_chunks)
        self._log_candidate("committed", buffered_chunks=len(buffered_chunks))
        logger.info("[VAD] SpeechStart: id=%s, prob=%.2f", str(utterance_id)[:8], start_prob)
        self._speech_chunk_count = len(buffered_chunks)
        self._speech_sample_count = sum(int(buffered.size) for buffered in buffered_chunks)

        with contextlib.suppress(Exception):
            if self._diagnostics_enabled():
                metrics = compute_audio_frame_metrics(
                    AudioFrameF32(
                        samples=buffered_chunks[0],
                        sample_rate_hz=self.sample_rate_hz,
                        channels=1,
                    )
                )
                assert self.diagnostic_event_callback is not None
                self.diagnostic_event_callback(
                    f"[AudioDiag][VAD][{self.diagnostic_label}] event=SpeechStart "
                    f"utterance_id={str(utterance_id)[:8]} "
                    f"prob={start_prob:.3f} threshold={self.speech_threshold} "
                    f"pre_roll_ms={len(pre_roll) * 1000.0 / self.sample_rate_hz:.1f} "
                    f"rms_db={metrics.rms_db:.1f} peak_db={metrics.peak_db:.1f}"
                )

        self._reset_pending_start()

        events: list[VadEvent] = [
            SpeechStart(utterance_id, pre_roll=pre_roll, chunk=buffered_chunks[0])
        ]
        events.extend(
            SpeechChunk(utterance_id, chunk=buffered.copy()) for buffered in buffered_chunks[1:]
        )
        if self._max_segment_reached():
            self._emit_max_duration_end(events)
        return events

    def _emit_soft_pause_end(
        self,
        events: list[VadEvent],
        *,
        trailing_silence_ms: int,
    ) -> None:
        utterance_id = self._utterance_id
        if utterance_id is None:
            return

        speech_audio_ms = self._speech_sample_count * 1000.0 / self.sample_rate_hz
        logger.info(
            "[VAD] SpeechEnd: id=%s, reason=soft_pause, trailing_silence_ms=%s, "
            "speech_audio_ms=%.1f",
            str(utterance_id)[:8],
            trailing_silence_ms,
            speech_audio_ms,
        )
        with contextlib.suppress(Exception):
            if self._diagnostics_enabled():
                assert self.diagnostic_event_callback is not None
                self.diagnostic_event_callback(
                    f"[AudioDiag][VAD][{self.diagnostic_label}] event=SpeechEnd "
                    f"utterance_id={str(utterance_id)[:8]} "
                    f"reason=soft_pause trailing_silence_ms={trailing_silence_ms} "
                    f"speech_audio_ms={speech_audio_ms:.1f} "
                    f"chunk_count={self._speech_chunk_count}"
                )

        events.append(
            SpeechEnd(
                utterance_id,
                trailing_silence_ms=trailing_silence_ms,
                reason="soft_pause",
            )
        )
        self._reset_active_segment()

    def _emit_max_duration_end(
        self,
        events: list[VadEvent],
        *,
        trailing_silence_ms: int = 0,
    ) -> None:
        utterance_id = self._utterance_id
        if utterance_id is None:
            return

        logger.info(
            "[VAD] SpeechEnd: id=%s, reason=max_duration, trailing_silence_ms=%s, "
            "speech_audio_ms=%.1f",
            str(utterance_id)[:8],
            trailing_silence_ms,
            self._speech_sample_count * 1000.0 / self.sample_rate_hz,
        )
        with contextlib.suppress(Exception):
            if self._diagnostics_enabled():
                speech_audio_ms = self._speech_sample_count * 1000.0 / self.sample_rate_hz
                assert self.diagnostic_event_callback is not None
                self.diagnostic_event_callback(
                    f"[AudioDiag][VAD][{self.diagnostic_label}] event=SpeechEnd "
                    f"utterance_id={str(utterance_id)[:8]} "
                    f"reason=max_duration trailing_silence_ms={trailing_silence_ms} "
                    f"speech_audio_ms={speech_audio_ms:.1f} "
                    f"chunk_count={self._speech_chunk_count}"
                )

        events.append(
            SpeechEnd(
                utterance_id,
                trailing_silence_ms=trailing_silence_ms,
                reason="max_duration",
            )
        )
        self._reset_active_segment()

    def _drop_pending_start(self) -> None:
        if self._pending_start_id is None:
            return
        self._log_candidate("dropped", buffered_chunks=len(self._pending_start_chunks))
        self._reset_pending_start()

    def _reset_pending_start(self) -> None:
        self._pending_start_id = None
        self._pending_start_pre_roll = None
        self._pending_start_prob = None
        self._pending_start_chunks = []
        self._pending_debounce_reached = False

    def _log_candidate(
        self,
        action: str,
        *,
        prob: float | None = None,
        buffered_chunks: int | None = None,
    ) -> None:
        if not self.candidate_log_label:
            return
        utterance = (
            str(self._pending_start_id)[:8] if self._pending_start_id is not None else "unknown"
        )
        if action == "start":
            logger.info(
                "[VAD][TEST] %s candidate start: id=%s, prob=%.2f",
                self.candidate_log_label,
                utterance,
                0.0 if prob is None else prob,
            )
            return
        if action == "dropped":
            logger.info(
                "[VAD][TEST] %s candidate dropped: id=%s, buffered_chunks=%s",
                self.candidate_log_label,
                utterance,
                buffered_chunks,
            )
            return
        if action == "committed":
            logger.info(
                "[VAD][TEST] %s candidate committed: id=%s, buffered_chunks=%s",
                self.candidate_log_label,
                utterance,
                buffered_chunks,
            )

    def _diagnostics_enabled(self) -> bool:
        if self.diagnostic_event_callback is None:
            return False
        if self.diagnostics_enabled is None:
            return True
        with contextlib.suppress(Exception):
            return bool(self.diagnostics_enabled())
        return False


PEER_VAD_SPEECH_THRESHOLD = 0.5
PEER_VAD_START_DEBOUNCE_CHUNKS = 3
PEER_VAD_START_COMMIT_CHUNKS = 3
PEER_SOFT_BOUNDARY_START_MS = 5000
PEER_SOFT_PAUSE_MS = 160
PEER_MAX_SEGMENT_MS = 7000


def create_peer_vad_gating(
    engine: VadEngine,
    *,
    sample_rate_hz: int,
    ring_buffer_ms: int,
    speech_threshold: float = PEER_VAD_SPEECH_THRESHOLD,
    hangover_ms: int,
    diagnostic_event_callback: Callable[[str], object] | None = None,
    diagnostics_enabled: Callable[[], bool] | None = None,
    diagnostic_label: str = "peer",
) -> VadGating:
    return VadGating(
        engine=engine,
        sample_rate_hz=sample_rate_hz,
        ring_buffer_ms=max(1, ring_buffer_ms),
        speech_threshold=speech_threshold,
        hangover_ms=hangover_ms,
        max_segment_ms=PEER_MAX_SEGMENT_MS,
        start_debounce_chunks=PEER_VAD_START_DEBOUNCE_CHUNKS,
        start_commit_chunks=PEER_VAD_START_COMMIT_CHUNKS,
        soft_boundary_start_ms=PEER_SOFT_BOUNDARY_START_MS,
        soft_pause_ms=PEER_SOFT_PAUSE_MS,
        candidate_log_label="Peer",
        diagnostic_event_callback=diagnostic_event_callback,
        diagnostics_enabled=diagnostics_enabled,
        diagnostic_label=diagnostic_label,
    )
