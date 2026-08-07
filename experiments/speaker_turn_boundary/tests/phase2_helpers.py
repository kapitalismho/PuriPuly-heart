from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from experiments.speaker_turn_boundary.config import CANONICAL_SAMPLE_RATE_HZ


def write_pcm16_flac(path: Path, samples: np.ndarray) -> None:
    pcm16 = np.clip(samples, -1.0, 1.0)
    pcm16 = np.round(pcm16 * 32767.0).astype(np.int16)
    result = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "s16le",
            "-ar",
            str(CANONICAL_SAMPLE_RATE_HZ),
            "-ac",
            "1",
            "-i",
            "pipe:0",
            "-c:a",
            "flac",
            str(path),
        ],
        input=pcm16.tobytes(),
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode("utf-8", "replace")[-500:])


def write_pcm16_wav(
    path: Path, samples: np.ndarray, sample_rate_hz: int = CANONICAL_SAMPLE_RATE_HZ
) -> None:
    import wave

    pcm16 = np.clip(samples, -1.0, 1.0)
    pcm16 = np.round(pcm16 * 32767.0).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate_hz)
        handle.writeframes(pcm16.tobytes())


def speech_like(
    duration_s: float, seed: int, sample_rate_hz: int = CANONICAL_SAMPLE_RATE_HZ
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(duration_s * sample_rate_hz)
    t = np.arange(n) / sample_rate_hz
    tone = 0.3 * np.sin(2 * np.pi * 220.0 * t)
    tone += 0.15 * np.sin(2 * np.pi * 440.0 * t)
    noise = 0.05 * rng.standard_normal(n)
    envelope = np.clip(np.sin(np.pi * np.arange(n) / n) * 2.0, 0.0, 1.0)
    return ((tone + noise) * envelope).astype(np.float32)
