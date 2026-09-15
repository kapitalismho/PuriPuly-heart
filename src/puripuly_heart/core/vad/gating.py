from __future__ import annotations

import math
import uuid
from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

import numpy as np

from puripuly_heart.config.resolved import vad_exit_threshold
from puripuly_heart.core.audio.format import AudioCaptureSpan
from puripuly_heart.core.audio.ring_buffer import RingBufferF32
from puripuly_heart.core.speech_boundary import SpeechBoundaryReason


class VadEngine(Protocol):
    def speech_probability(self, samples: np.ndarray, *, sample_rate_hz: int) -> float: ...
    def reset(self) -> None: ...


@dataclass(frozen=True, slots=True)
class SpeechStart:
    utterance_id: UUID
    pre_roll: np.ndarray
    chunk: np.ndarray
    pre_roll_capture: tuple[AudioCaptureSpan, ...] = ()
    chunk_capture: tuple[AudioCaptureSpan, ...] = ()
    genuine_onset: bool = True


@dataclass(frozen=True, slots=True)
class SpeechChunk:
    utterance_id: UUID
    chunk: np.ndarray

    chunk_capture: tuple[AudioCaptureSpan, ...] = ()


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
    continuation_threshold: float
    hangover_chunks: int
    chunk_samples: int
    start_debounce_chunks: int
    start_commit_chunks: int
    external_delivery_boundaries: bool
    _ring: RingBufferF32
    _in_speech: bool
    _utterance_id: UUID | None
    _silence_run: int
    _pending_start_id: UUID | None
    _pending_start_pre_roll: np.ndarray | None
    _pending_start_pre_roll_capture: tuple[AudioCaptureSpan, ...]
    _pending_start_prob: float | None
    _pending_start_chunks: list[np.ndarray]
    _pending_start_capture: list[tuple[AudioCaptureSpan, ...]]
    _pending_debounce_reached: bool
    _speech_sample_count: int
    _last_observation_was_speech: bool

    _ring_capture: list[AudioCaptureSpan]
    _rollover_pending: bool
    _rollover_silence_run: int
    _pending_segment_settings: tuple[float, float, int, int] | None
    _hard_rollover_pre_roll_samples: int
    _hard_rollover_pre_roll: np.ndarray | None
    _hard_rollover_pre_roll_capture: tuple[AudioCaptureSpan, ...]

    def __init__(
        self,
        engine: VadEngine,
        *,
        sample_rate_hz: int,
        ring_buffer_ms: int = 500,
        speech_threshold: float = 0.4,
        continuation_threshold: float | None = None,
        hangover_ms: int = 1100,
        chunk_samples: int | None = None,
        start_debounce_chunks: int = 1,
        start_commit_chunks: int = 1,
        external_delivery_boundaries: bool = False,
        hard_rollover_pre_roll_ms: int = 0,
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
        if hard_rollover_pre_roll_ms < 0:
            raise ValueError("hard_rollover_pre_roll_ms must be >= 0")

        self.engine = engine
        self.sample_rate_hz = sample_rate_hz
        self.speech_threshold = speech_threshold
        self.continuation_threshold = (
            speech_threshold if continuation_threshold is None else continuation_threshold
        )
        self.chunk_samples = chunk_samples or default_chunk_samples(sample_rate_hz)
        self.start_debounce_chunks = start_debounce_chunks
        self.start_commit_chunks = start_commit_chunks
        self.external_delivery_boundaries = external_delivery_boundaries

        chunk_ms = (self.chunk_samples / self.sample_rate_hz) * 1000.0
        self.hangover_chunks = int(math.ceil(hangover_ms / chunk_ms)) if hangover_ms > 0 else 0

        capacity_samples = int(self.sample_rate_hz * (ring_buffer_ms / 1000.0))
        self._ring = RingBufferF32(capacity_samples=capacity_samples)
        self._hard_rollover_pre_roll_samples = int(
            self.sample_rate_hz * hard_rollover_pre_roll_ms / 1000.0
        )

        self._in_speech = False
        self._utterance_id = None
        self._silence_run = 0
        self._pending_start_id = None
        self._pending_start_pre_roll = None
        self._pending_start_pre_roll_capture = ()
        self._pending_start_prob = None
        self._pending_start_chunks = []
        self._pending_debounce_reached = False
        self._pending_start_capture = []
        self._speech_sample_count = 0
        self._ring_capture = []
        self._rollover_pending = False
        self._rollover_silence_run = 0
        self._last_observation_was_speech = False
        self._pending_segment_settings = None
        self._hard_rollover_pre_roll = None
        self._hard_rollover_pre_roll_capture = ()

    @property
    def in_speech(self) -> bool:
        return self._in_speech

    @property
    def continuation_pending(self) -> bool:
        return self._rollover_pending

    @property
    def last_observation_was_speech(self) -> bool:
        return self._last_observation_was_speech

    @property
    def utterance_id(self) -> UUID | None:
        return self._utterance_id

    def reset(self) -> None:
        self.engine.reset()
        self._ring.clear()
        self._ring_capture.clear()
        self._in_speech = False
        self._utterance_id = None
        self._silence_run = 0
        self._rollover_pending = False
        self._rollover_silence_run = 0
        self._clear_hard_rollover_pre_roll()
        self._reset_pending_start()
        self._last_observation_was_speech = False
        self._speech_sample_count = 0
        self._apply_pending_segment_settings()

    def reconfigure_next_segment(
        self,
        *,
        speech_threshold: float,
        continuation_threshold: float | None = None,
        hangover_ms: int,
        ring_buffer_ms: int,
    ) -> None:
        if hangover_ms < 0:
            raise ValueError("hangover_ms must be >= 0")
        if ring_buffer_ms <= 0:
            raise ValueError("ring_buffer_ms must be > 0")
        chunk_ms = self.chunk_samples * 1000.0 / self.sample_rate_hz
        hangover_chunks = int(math.ceil(hangover_ms / chunk_ms)) if hangover_ms > 0 else 0
        capacity_samples = int(self.sample_rate_hz * ring_buffer_ms / 1000.0)
        self._pending_segment_settings = (
            speech_threshold,
            speech_threshold if continuation_threshold is None else continuation_threshold,
            hangover_chunks,
            capacity_samples,
        )
        if not self._in_speech:
            self._drop_pending_start()
            self._apply_pending_segment_settings()

    def _apply_pending_segment_settings(self) -> None:
        pending = self._pending_segment_settings
        if pending is None:
            return
        self._pending_segment_settings = None
        speech_threshold, continuation_threshold, hangover_chunks, capacity_samples = pending
        self.speech_threshold = speech_threshold
        self.continuation_threshold = continuation_threshold
        self.hangover_chunks = hangover_chunks
        if self._ring.capacity_samples == capacity_samples:
            return
        retained = self._ring.get_last_samples(capacity_samples)
        retained_capture = self._capture_suffix(retained.size)
        self._ring = RingBufferF32(capacity_samples=capacity_samples)
        self._ring.append(retained)
        self._ring_capture = list(retained_capture)

    def process_chunk(self, chunk: np.ndarray) -> list[VadEvent]:
        return self.process_owned_chunk(chunk, ())

    def process_owned_chunk(
        self,
        chunk: np.ndarray,
        capture: tuple[AudioCaptureSpan, ...],
    ) -> list[VadEvent]:
        chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)
        if chunk.size != self.chunk_samples:
            raise ValueError(f"chunk must have {self.chunk_samples} samples")

        prob = self.engine.speech_probability(chunk, sample_rate_hz=self.sample_rate_hz)
        observation_threshold = (
            self.continuation_threshold
            if self._in_speech or self._rollover_pending
            else self.speech_threshold
        )
        self._last_observation_was_speech = bool(prob >= observation_threshold)

        events: list[VadEvent] = []

        if not self._in_speech and self._rollover_pending:
            if self._last_observation_was_speech:
                events.extend(self._start_rollover(chunk, capture, prob))
                self._append_ring(chunk, capture)
                return events
            self._rollover_silence_run += 1
            if self._rollover_silence_run >= max(1, self.hangover_chunks):
                self._rollover_pending = False
                self._rollover_silence_run = 0
                self._clear_hard_rollover_pre_roll()
                self.engine.reset()
            self._append_ring(chunk, capture)
            return events

        if not self._in_speech:
            if prob >= self.speech_threshold:
                events.extend(self._handle_pending_start(chunk, prob, capture))
            else:
                self._drop_pending_start()
            self._append_ring(chunk, capture)
            return events

        events.append(
            SpeechChunk(
                self._utterance_id,
                chunk=chunk.copy(),
                chunk_capture=capture,
            )
        )  # type: ignore[arg-type]
        self._speech_sample_count += int(chunk.size)

        if self._last_observation_was_speech:
            self._silence_run = 0
            self._append_ring(chunk, capture)
            return events

        self._silence_run += 1
        trailing_silence_ms = self._trailing_silence_ms()
        if self.external_delivery_boundaries:
            self._append_ring(chunk, capture)
            return events
        elif self._silence_run >= self.hangover_chunks:

            events.append(
                SpeechEnd(
                    self._utterance_id,
                    trailing_silence_ms=trailing_silence_ms,
                    reason="silence",
                )
            )  # type: ignore[arg-type]
            self._reset_active_segment()
            self._rollover_pending = False
            self._rollover_silence_run = 0
            self.engine.reset()

        self._append_ring(chunk, capture)
        return events

    def _trailing_silence_ms(self) -> int:
        return int(round(self._silence_run * (self.chunk_samples / self.sample_rate_hz) * 1000.0))

    def _reset_active_segment(self) -> None:
        self._in_speech = False
        self._utterance_id = None
        self._silence_run = 0
        self._speech_sample_count = 0
        self._apply_pending_segment_settings()

    def _handle_pending_start(
        self,
        chunk: np.ndarray,
        prob: float,
        capture: tuple[AudioCaptureSpan, ...],
    ) -> list[VadEvent]:
        if self._pending_start_id is None:
            self._pending_start_id = uuid.uuid4()
            self._pending_start_pre_roll = self._ring.get_last_samples(self._ring.capacity_samples)
            self._pending_start_pre_roll_capture = self._capture_suffix(
                len(self._pending_start_pre_roll)
            )
            self._pending_start_prob = prob
            self._pending_start_chunks = [chunk.copy()]
            self._pending_start_capture = [capture]
            self._pending_debounce_reached = self.start_debounce_chunks <= 1
        else:
            self._pending_start_chunks.append(chunk.copy())
            self._pending_start_capture.append(capture)

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
        pre_roll_capture = self._pending_start_pre_roll_capture
        buffered_chunks = list(self._pending_start_chunks)
        buffered_capture = list(self._pending_start_capture)
        self._speech_sample_count = sum(int(buffered.size) for buffered in buffered_chunks)

        self._reset_pending_start()

        events: list[VadEvent] = [
            SpeechStart(
                utterance_id,
                pre_roll=pre_roll,
                chunk=buffered_chunks[0],
                pre_roll_capture=pre_roll_capture,
                chunk_capture=buffered_capture[0],
            )
        ]
        events.extend(
            SpeechChunk(
                utterance_id,
                chunk=buffered.copy(),
                chunk_capture=buffered_capture[index],
            )
            for index, buffered in enumerate(buffered_chunks[1:], start=1)
        )
        return events

    def _drop_pending_start(self) -> None:
        if self._pending_start_id is None:
            return
        self._reset_pending_start()

    def _reset_pending_start(self) -> None:
        self._pending_start_id = None
        self._pending_start_pre_roll = None
        self._pending_start_pre_roll_capture = ()
        self._pending_start_prob = None
        self._pending_start_chunks = []
        self._pending_start_capture = []
        self._pending_debounce_reached = False

    def _start_rollover(
        self,
        chunk: np.ndarray,
        capture: tuple[AudioCaptureSpan, ...],
        prob: float,
    ) -> list[VadEvent]:
        utterance_id = uuid.uuid4()
        self._in_speech = True
        self._utterance_id = utterance_id
        self._silence_run = 0
        self._rollover_pending = False
        self._rollover_silence_run = 0
        self._speech_sample_count = int(chunk.size)
        pre_roll = self._hard_rollover_pre_roll
        if pre_roll is None:
            pre_roll = np.empty((0,), dtype=np.float32)
        pre_roll_capture = self._hard_rollover_pre_roll_capture
        self._clear_hard_rollover_pre_roll()
        return [
            SpeechStart(
                utterance_id,
                pre_roll=pre_roll,
                chunk=chunk.copy(),
                pre_roll_capture=pre_roll_capture,
                chunk_capture=capture,
                genuine_onset=False,
            )
        ]

    def seal_active(self, *, reason: SpeechBoundaryReason) -> SpeechEnd | None:
        utterance_id = self._utterance_id
        if utterance_id is None:
            self.reset()
            return None
        event = SpeechEnd(
            utterance_id,
            trailing_silence_ms=self._trailing_silence_ms(),
            reason=reason,
        )
        self.reset()
        return event

    def seal_active_for_rollover(self, *, reason: SpeechBoundaryReason) -> SpeechEnd | None:
        utterance_id = self._utterance_id
        if utterance_id is None:
            return None
        event = SpeechEnd(
            utterance_id,
            trailing_silence_ms=self._trailing_silence_ms(),
            reason=reason,
        )
        sample_count = (
            min(
                self._hard_rollover_pre_roll_samples,
                self._speech_sample_count,
                self._ring.capacity_samples,
            )
            if reason == "delivery_deadline"
            else 0
        )
        self._hard_rollover_pre_roll = self._ring.get_last_samples(sample_count)
        self._hard_rollover_pre_roll_capture = self._capture_suffix(sample_count)
        self._reset_active_segment()
        self._rollover_pending = True
        self._rollover_silence_run = 0
        return event

    def _append_ring(
        self,
        chunk: np.ndarray,
        capture: tuple[AudioCaptureSpan, ...],
    ) -> None:
        self._ring.append(chunk)
        self._ring_capture.extend(capture)
        self._ring_capture = list(self._capture_suffix(self._ring.capacity_samples))

    def _capture_suffix(self, sample_count: int) -> tuple[AudioCaptureSpan, ...]:
        if sample_count <= 0:
            return ()
        remaining = sample_count
        selected: list[AudioCaptureSpan] = []
        for item in reversed(self._ring_capture):
            item_count = item.normalized_sample_count
            if item_count <= 0:
                continue
            if item_count <= remaining:
                selected.append(item)
                remaining -= item_count
            else:
                end = item.normalized_end_sample
                if end is None:
                    continue
                selected.append(item.slice_normalized(end - remaining, end))
                remaining = 0
            if remaining == 0:
                break
        selected.reverse()
        return tuple(selected)

    def _clear_hard_rollover_pre_roll(self) -> None:
        self._hard_rollover_pre_roll = None
        self._hard_rollover_pre_roll_capture = ()


PEER_VAD_SPEECH_THRESHOLD = 0.5
PEER_VAD_START_DEBOUNCE_CHUNKS = 3
PEER_VAD_START_COMMIT_CHUNKS = 3
PEER_VAD_DELIVERY_BOUNDARIES_EXTERNAL = True
PEER_HARD_ROLLOVER_PRE_ROLL_MS = 300


def create_peer_vad_gating(
    engine: VadEngine,
    *,
    sample_rate_hz: int,
    ring_buffer_ms: int,
    speech_threshold: float = PEER_VAD_SPEECH_THRESHOLD,
    hangover_ms: int,
) -> VadGating:
    return VadGating(
        engine=engine,
        sample_rate_hz=sample_rate_hz,
        ring_buffer_ms=max(1, ring_buffer_ms),
        speech_threshold=speech_threshold,
        continuation_threshold=vad_exit_threshold(speech_threshold),
        hangover_ms=hangover_ms,
        start_debounce_chunks=PEER_VAD_START_DEBOUNCE_CHUNKS,
        start_commit_chunks=PEER_VAD_START_COMMIT_CHUNKS,
        external_delivery_boundaries=PEER_VAD_DELIVERY_BOUNDARIES_EXTERNAL,
        hard_rollover_pre_roll_ms=PEER_HARD_ROLLOVER_PRE_ROLL_MS,
    )
