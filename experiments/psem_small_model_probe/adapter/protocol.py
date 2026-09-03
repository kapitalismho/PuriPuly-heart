from __future__ import annotations
from pathlib import Path

from dataclasses import dataclass, field
from typing import Protocol


class BindingError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class StepOut:
    speech: float | None
    anchor: float
    aux: dict = field(default_factory=dict)
    source_time_ms: int = 0


def frame_bytes(frame_ms: int, sample_rate_hz: int = 16000) -> int:
    return sample_rate_hz * frame_ms // 1000 * 2


def load_wav_mono16k(path: str | Path) -> bytes:
    import wave

    with wave.open(str(path), "rb") as reader:
        if reader.getframerate() != 16000 or reader.getnchannels() != 1 or reader.getsampwidth() != 2:
            raise ValueError(
                f"expected mono 16 kHz PCM16 WAV: {path} "
                f"(got {reader.getframerate()} Hz, {reader.getnchannels()} ch, "
                f"{reader.getsampwidth() * 8}-bit)"
            )
        return reader.readframes(reader.getnframes())


def validate_pcm16_chunk(chunk: bytes, *, frame_ms: int, sample_rate_hz: int = 16000) -> int:
    if sample_rate_hz != 16000:
        raise ValueError(f"sample_rate_hz must be 16000, got {sample_rate_hz}")
    unit = frame_bytes(frame_ms, sample_rate_hz)
    if len(chunk) == 0 or len(chunk) % unit != 0:
        raise ValueError(
            f"chunk must be a non-empty multiple of one {frame_ms} ms frame "
            f"({unit} bytes mono 16 kHz int16 LE), got {len(chunk)} bytes"
        )
    return len(chunk) // unit


class PSEMObservationAdapter(Protocol):
    frame_ms: int
    sample_rate_hz: int = 16000

    def reset(self) -> None:
        ...

    def bind(self, reference_pcm16: bytes) -> None:
        ...

    def step(self, pcm16_chunk: bytes) -> StepOut:
        ...
