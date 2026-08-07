from __future__ import annotations

import time
import wave
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from experiments.speaker_turn_boundary.config import (
    B0_SOURCE,
    B0_VAD_HANGOVER_MS,
    B0_VAD_MAX_SEGMENT_MS,
    B0_VAD_PRE_ROLL_MS,
    B0_VAD_PROFILE,
    B0_VAD_SPEECH_THRESHOLD,
    B0_VAD_START_COMMIT_CHUNKS,
    B0_VAD_START_DEBOUNCE_CHUNKS,
    CANONICAL_CHUNK_SAMPLES,
    CANONICAL_SAMPLE_RATE_HZ,
)
from experiments.speaker_turn_boundary.events import (
    DetectorProgress,
    SpeakerBoundaryEvent,
)
from puripuly_heart.core.vad.gating import (
    VadEngine,
    VadGating,
    create_peer_vad_gating,
)


class CanonicalAudioError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class CompletedUtterance:
    utterance_id: str
    start_sample: int
    speech_end_sample: int
    trailing_silence_ms: int
    reason: str


@dataclass(slots=True)
class VadBoundaryReplay:
    engine_factory: Callable[[], VadEngine]
    sample_rate_hz: int = CANONICAL_SAMPLE_RATE_HZ
    chunk_samples: int = CANONICAL_CHUNK_SAMPLES
    ring_buffer_ms: int = B0_VAD_PRE_ROLL_MS
    hangover_ms: int = B0_VAD_HANGOVER_MS
    monotonic_ns: Callable[[], int] = time.perf_counter_ns
    _gating: VadGating | None = field(init=False, default=None, repr=False)
    _audio_epoch: int | None = field(init=False, default=None)
    _chunk_index: int = field(init=False, default=0)
    _current_utterance: CompletedUtterance | None = field(init=False, default=None)
    _prev_utterance: CompletedUtterance | None = field(init=False, default=None)
    _utterance_seq: int = field(init=False, default=0)
    _pre_roll_samples: int = field(init=False, default=0)
    _boundaries: list[SpeakerBoundaryEvent] = field(init=False, default_factory=list)
    _progress: list[DetectorProgress] = field(init=False, default_factory=list)

    @property
    def audio_epoch(self) -> int | None:
        return self._audio_epoch

    def start_epoch(self, audio_epoch: int) -> None:
        self._gating = create_peer_vad_gating(
            self.engine_factory(),
            sample_rate_hz=self.sample_rate_hz,
            ring_buffer_ms=self.ring_buffer_ms,
            hangover_ms=self.hangover_ms,
        )
        self._audio_epoch = audio_epoch
        self._chunk_index = 0
        self._current_utterance = None
        self._prev_utterance = None
        self._utterance_seq = 0
        self._pre_roll_samples = 0
        self._boundaries = []
        self._progress = []

    def process_chunk(self, chunk: np.ndarray) -> list[SpeakerBoundaryEvent]:
        if self._audio_epoch is None or self._gating is None:
            raise RuntimeError("start_epoch must be called before process_chunk")
        chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)
        if chunk.size != self.chunk_samples:
            raise ValueError(f"chunk must have {self.chunk_samples} samples")
        events = self._gating.process_chunk(chunk)
        boundaries = self._translate_events(events)
        self._boundaries.extend(boundaries)
        frontier_start = self._chunk_index * self.chunk_samples
        self._progress.append(
            DetectorProgress(
                audio_epoch=self._audio_epoch,
                observed_source_sample=frontier_start + self.chunk_samples,
                safe_boundary_frontier_sample=frontier_start,
            )
        )
        self._chunk_index += 1
        return boundaries

    def progress_snapshot(self) -> DetectorProgress:
        if self._audio_epoch is None or not self._progress:
            raise RuntimeError("no epoch has been processed")
        return self._progress[-1]

    @property
    def boundaries(self) -> list[SpeakerBoundaryEvent]:
        return list(self._boundaries)

    @property
    def progress(self) -> list[DetectorProgress]:
        return list(self._progress)

    def _translate_events(self, events: list[object]) -> list[SpeakerBoundaryEvent]:
        if self._audio_epoch is None:
            return []
        boundaries: list[SpeakerBoundaryEvent] = []
        chunk_index = self._chunk_index
        start_sample = chunk_index * self.chunk_samples
        for event in events:
            kind = type(event).__name__
            if kind == "SpeechStart":
                pre_roll = getattr(event, "pre_roll", None)
                if pre_roll is not None:
                    self._pre_roll_samples = int(np.asarray(pre_roll).size)
                if self._prev_utterance is not None:
                    boundary_sample = start_sample
                    prev = self._prev_utterance
                    boundaries.append(
                        SpeakerBoundaryEvent(
                            audio_epoch=self._audio_epoch,
                            boundary_source_sample=boundary_sample,
                            observed_source_sample_at_emit=start_sample + self.chunk_samples,
                            emitted_monotonic_ns=self.monotonic_ns(),
                            confidence=None,
                            source=B0_SOURCE,
                            debug={
                                "profile": B0_VAD_PROFILE,
                                "speech_threshold": B0_VAD_SPEECH_THRESHOLD,
                                "start_debounce_chunks": B0_VAD_START_DEBOUNCE_CHUNKS,
                                "start_commit_chunks": B0_VAD_START_COMMIT_CHUNKS,
                                "max_segment_ms": B0_VAD_MAX_SEGMENT_MS,
                                "hangover_ms": self.hangover_ms,
                                "ring_buffer_ms": self.ring_buffer_ms,
                                "chunk_samples": self.chunk_samples,
                                "start_chunk_index": chunk_index,
                                "pre_roll_samples": self._pre_roll_samples,
                                "prev_utterance_seq": self._utterance_seq,
                                "prev_utterance_start_sample": prev.start_sample,
                                "prev_speech_end_sample": prev.speech_end_sample,
                                "gap_samples": boundary_sample - prev.speech_end_sample,
                                "prev_trailing_silence_ms": prev.trailing_silence_ms,
                                "prev_end_reason": prev.reason,
                            },
                        )
                    )
                self._utterance_seq += 1
                current_id = str(getattr(event, "utterance_id"))
                self._current_utterance = CompletedUtterance(
                    utterance_id=current_id,
                    start_sample=start_sample,
                    speech_end_sample=start_sample,
                    trailing_silence_ms=0,
                    reason="",
                )
            elif kind == "SpeechEnd":
                current = self._current_utterance
                if current is None:
                    continue
                trailing_silence_ms = int(getattr(event, "trailing_silence_ms", 0))
                reason = str(getattr(event, "reason", "silence"))
                chunk_ms = self.chunk_samples / self.sample_rate_hz * 1000.0
                silence_run = int(round(trailing_silence_ms / chunk_ms))
                speech_end_sample = (chunk_index + 1 - silence_run) * self.chunk_samples
                if speech_end_sample < current.start_sample:
                    speech_end_sample = current.start_sample
                completed = CompletedUtterance(
                    utterance_id=current.utterance_id,
                    start_sample=current.start_sample,
                    speech_end_sample=speech_end_sample,
                    trailing_silence_ms=trailing_silence_ms,
                    reason=reason,
                )
                self._prev_utterance = completed
                self._current_utterance = None
        return boundaries


@dataclass(frozen=True, slots=True)
class EpochReplayResult:
    audio_epoch: int
    length_samples: int
    boundaries: list[SpeakerBoundaryEvent]
    progress: list[DetectorProgress]

    def to_dict(self) -> dict[str, Any]:
        return {
            "audio_epoch": self.audio_epoch,
            "length_samples": self.length_samples,
            "boundaries": [boundary.to_dict() for boundary in self.boundaries],
            "progress": [snapshot.to_dict() for snapshot in self.progress],
        }


def load_canonical_wav(wav_path: Path) -> np.ndarray:
    with wave.open(str(wav_path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        frame_rate = wav_file.getframerate()
        if frame_rate != CANONICAL_SAMPLE_RATE_HZ:
            raise CanonicalAudioError(
                f"canonical audio must be {CANONICAL_SAMPLE_RATE_HZ} Hz mono PCM16, "
                f"got {frame_rate} Hz"
            )
        if channels != 1:
            raise CanonicalAudioError(
                f"canonical audio must be {CANONICAL_SAMPLE_RATE_HZ} Hz mono PCM16, "
                f"got {channels} channels"
            )
        if sample_width != 2:
            raise CanonicalAudioError(
                f"canonical audio must be {CANONICAL_SAMPLE_RATE_HZ} Hz mono PCM16, "
                f"got {sample_width * 8}-bit samples"
            )
        frames = wav_file.readframes(wav_file.getnframes())
    samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    return samples


def replay_wav_epoch(
    wav_path: Path,
    *,
    audio_epoch: int,
    engine_factory: Callable[[], VadEngine],
    monotonic_ns: Callable[[], int] = time.perf_counter_ns,
    chunk_samples: int = CANONICAL_CHUNK_SAMPLES,
    hangover_ms: int = B0_VAD_HANGOVER_MS,
) -> EpochReplayResult:
    samples = load_canonical_wav(wav_path)
    replay = VadBoundaryReplay(
        engine_factory=engine_factory,
        chunk_samples=chunk_samples,
        hangover_ms=hangover_ms,
        monotonic_ns=monotonic_ns,
    )
    replay.start_epoch(audio_epoch)
    offset = 0
    while offset < samples.size:
        chunk = samples[offset : offset + chunk_samples]
        if chunk.size < chunk_samples:
            break
        replay.process_chunk(chunk)
        offset += chunk_samples
    return EpochReplayResult(
        audio_epoch=audio_epoch,
        length_samples=samples.size,
        boundaries=replay.boundaries,
        progress=replay.progress,
    )
