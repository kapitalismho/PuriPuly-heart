from __future__ import annotations

import contextlib
import logging
import math
import uuid
from dataclasses import dataclass
from typing import Callable, Protocol
from uuid import UUID

import numpy as np

from puripuly_heart.config.resolved import vad_exit_threshold
from puripuly_heart.core.audio.diagnostics import compute_audio_frame_metrics
from puripuly_heart.core.audio.format import AudioCaptureSpan, AudioFrameF32
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
    candidate_log_label: str | None
    diagnostic_event_callback: Callable[[str], object] | None
    diagnostics_enabled: Callable[[], bool] | None
    external_delivery_boundaries: bool
    diagnostic_label: str
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
    _speech_chunk_count: int
    _speech_sample_count: int
    _last_observation_was_speech: bool
    _diag_run_class: str | None
    _diag_run_start_sample: int
    _diag_run_samples: int
    _diag_prob_min: float
    _diag_prob_max: float
    _diag_observed_samples: int
    _diag_non_speech_samples: int
    _diag_max_non_speech_samples: int
    _diag_band_frames: int

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
        candidate_log_label: str | None = None,
        diagnostic_event_callback: Callable[[str], object] | None = None,
        diagnostics_enabled: Callable[[], bool] | None = None,
        diagnostic_label: str = "self",
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
        self.candidate_log_label = candidate_log_label
        self.diagnostic_event_callback = diagnostic_event_callback
        self.diagnostics_enabled = diagnostics_enabled
        self.diagnostic_label = diagnostic_label
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
        self._speech_chunk_count = 0
        self._speech_sample_count = 0
        self._ring_capture = []
        self._rollover_pending = False
        self._rollover_silence_run = 0
        self._last_observation_was_speech = False
        self._reset_diagnostics()
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
        self._flush_diagnostic_run()
        self._reset_diagnostics()
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
        self._speech_chunk_count = 0
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
        self._speech_chunk_count += 1
        self._speech_sample_count += int(chunk.size)
        self._observe_diagnostics(prob, self._speech_sample_count - int(chunk.size))

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
            logger.info(
                "[VAD] SpeechEnd: id=%s, trailing_silence_ms=%s",
                str(self._utterance_id)[:8],
                trailing_silence_ms,
            )
            self._log_speech_end("silence")

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
        self._flush_diagnostic_run()
        self._reset_diagnostics()
        self._in_speech = False
        self._utterance_id = None
        self._silence_run = 0
        self._speech_chunk_count = 0
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
            self._log_candidate("start", prob=prob)
        else:
            self._pending_start_chunks.append(chunk.copy())
            self._pending_start_capture.append(capture)

        self._observe_diagnostics(prob, (len(self._pending_start_chunks) - 1) * self.chunk_samples)

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
        start_prob = self._pending_start_prob if self._pending_start_prob is not None else prob
        buffered_chunks = list(self._pending_start_chunks)
        buffered_capture = list(self._pending_start_capture)
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
        self._log_candidate("dropped", buffered_chunks=len(self._pending_start_chunks))
        self._flush_diagnostic_run()
        self._reset_diagnostics()
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
        self._speech_chunk_count = 1
        self._speech_sample_count = int(chunk.size)
        pre_roll = self._hard_rollover_pre_roll
        if pre_roll is None:
            pre_roll = np.empty((0,), dtype=np.float32)
        pre_roll_capture = self._hard_rollover_pre_roll_capture
        self._clear_hard_rollover_pre_roll()
        logger.info("[VAD] Speech rollover: id=%s, prob=%.2f", str(utterance_id)[:8], prob)
        self._observe_diagnostics(prob, 0)
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
        self._log_speech_end(reason)
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
        self._log_speech_end(reason)
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

    def _reset_diagnostics(self) -> None:
        self._diag_run_class = None
        self._diag_run_start_sample = 0
        self._diag_run_samples = 0
        self._diag_prob_min = 0.0
        self._diag_prob_max = 0.0
        self._diag_observed_samples = 0
        self._diag_non_speech_samples = 0
        self._diag_max_non_speech_samples = 0
        self._diag_band_frames = 0

    def _observe_diagnostics(self, prob: float, start_sample: int) -> None:
        if not self._diagnostics_enabled():
            self._diag_run_class = None
            self._diag_run_samples = 0
            self._diag_non_speech_samples = 0
            return
        classification = (
            "speech"
            if prob >= self.speech_threshold
            else "band" if prob >= self.continuation_threshold else "non_speech"
        )
        if classification != self._diag_run_class:
            self._flush_diagnostic_run()
            self._diag_run_class = classification
            self._diag_run_start_sample = start_sample
            self._diag_prob_min = prob
            self._diag_prob_max = prob
        else:
            self._diag_prob_min = min(self._diag_prob_min, prob)
            self._diag_prob_max = max(self._diag_prob_max, prob)
        self._diag_run_samples += self.chunk_samples
        self._diag_observed_samples += self.chunk_samples
        if classification == "non_speech":
            self._diag_non_speech_samples += self.chunk_samples
            self._diag_max_non_speech_samples = max(
                self._diag_max_non_speech_samples, self._diag_non_speech_samples
            )
        else:
            self._diag_non_speech_samples = 0
        if classification == "band":
            self._diag_band_frames += 1

    def _flush_diagnostic_run(self) -> None:
        classification = self._diag_run_class
        samples = self._diag_run_samples
        self._diag_run_class = None
        self._diag_run_samples = 0
        if classification is None or not self._diagnostics_enabled():
            return
        with contextlib.suppress(Exception):
            assert self.diagnostic_event_callback is not None
            utterance_id = self._utterance_id or self._pending_start_id
            self.diagnostic_event_callback(
                f"[AudioDiag][VAD][{self.diagnostic_label}] event=VadRun "
                f"utterance_id={str(utterance_id)[:8]} class={classification} "
                f"start_audio_ms={self._diag_run_start_sample * 1000.0 / self.sample_rate_hz:.1f} "
                f"duration_ms={samples * 1000.0 / self.sample_rate_hz:.1f} "
                f"frame_count={samples // self.chunk_samples} "
                f"prob_min={self._diag_prob_min:.6f} prob_max={self._diag_prob_max:.6f} "
                f"onset_threshold={self.speech_threshold:.6f} "
                f"continuation_threshold={self.continuation_threshold:.6f}"
            )

    def _log_speech_end(self, reason: SpeechBoundaryReason) -> None:
        self._flush_diagnostic_run()
        with contextlib.suppress(Exception):
            if not self._diagnostics_enabled():
                return
            assert self.diagnostic_event_callback is not None
            self.diagnostic_event_callback(
                f"[AudioDiag][VAD][{self.diagnostic_label}] event=SpeechEnd "
                f"utterance_id={str(self._utterance_id)[:8]} reason={reason} "
                f"trailing_silence_ms={self._trailing_silence_ms()} "
                f"speech_audio_ms={self._speech_sample_count * 1000.0 / self.sample_rate_hz:.1f} "
                f"chunk_count={self._speech_chunk_count} "
                f"onset_threshold={self.speech_threshold:.6f} "
                f"continuation_threshold={self.continuation_threshold:.6f} "
                f"max_non_speech_ms={self._diag_max_non_speech_samples * 1000.0 / self.sample_rate_hz:.1f} "
                f"band_frame_count={self._diag_band_frames} "
                f"observed_audio_ms={self._diag_observed_samples * 1000.0 / self.sample_rate_hz:.1f} "
                f"observation_complete={str(self._diag_observed_samples == self._speech_sample_count).lower()}"
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
PEER_VAD_DELIVERY_BOUNDARIES_EXTERNAL = True
PEER_HARD_ROLLOVER_PRE_ROLL_MS = 300


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
        continuation_threshold=vad_exit_threshold(speech_threshold),
        hangover_ms=hangover_ms,
        start_debounce_chunks=PEER_VAD_START_DEBOUNCE_CHUNKS,
        start_commit_chunks=PEER_VAD_START_COMMIT_CHUNKS,
        external_delivery_boundaries=PEER_VAD_DELIVERY_BOUNDARIES_EXTERNAL,
        hard_rollover_pre_roll_ms=PEER_HARD_ROLLOVER_PRE_ROLL_MS,
        candidate_log_label="Peer",
        diagnostic_event_callback=diagnostic_event_callback,
        diagnostics_enabled=diagnostics_enabled,
        diagnostic_label=diagnostic_label,
    )
