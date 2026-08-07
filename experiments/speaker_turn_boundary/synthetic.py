from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from experiments.speaker_turn_boundary.config import (
    CANONICAL_SAMPLE_RATE_HZ,
    DEFAULT_GENERATOR_SEED,
)


def silence(duration_s: float, *, sample_rate_hz: int = CANONICAL_SAMPLE_RATE_HZ) -> np.ndarray:
    return np.zeros(int(round(duration_s * sample_rate_hz)), dtype=np.float32)


def envelope_edges(samples: np.ndarray, *, fade_s: float = 0.05) -> np.ndarray:
    out = samples.astype(np.float32, copy=True)
    fade = int(round(fade_s * CANONICAL_SAMPLE_RATE_HZ))
    if fade <= 0 or out.size == 0:
        return out
    fade = min(fade, out.size // 2)
    if fade > 0:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        out[:fade] *= ramp
        out[-fade:] *= ramp[::-1]
    return out


def harmonic_stack(
    duration_s: float,
    *,
    base_hz: float,
    seed: int,
    n_harmonics: int = 6,
    amplitude: float = 0.55,
    sample_rate_hz: int = CANONICAL_SAMPLE_RATE_HZ,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    count = int(round(duration_s * sample_rate_hz))
    t = np.arange(count, dtype=np.float64) / sample_rate_hz
    signal = np.zeros(count, dtype=np.float32)
    for harmonic in range(1, n_harmonics + 1):
        freq = base_hz * harmonic + float(rng.uniform(-6.0, 6.0))
        phase = float(rng.uniform(0.0, 2.0 * math.pi))
        amplitude_partial = amplitude / harmonic
        signal += (amplitude_partial * np.sin(2.0 * math.pi * freq * t + phase)).astype(np.float32)
    return envelope_edges(signal)


def _resonator_cascade(
    samples: np.ndarray,
    formants: list[tuple[float, float]],
    *,
    sample_rate_hz: int,
) -> np.ndarray:
    output = samples
    for center_hz, bandwidth_hz in formants:
        radius = float(np.exp(-math.pi * bandwidth_hz / sample_rate_hz))
        theta = 2.0 * math.pi * center_hz / sample_rate_hz
        a1 = -2.0 * radius * math.cos(theta)
        a2 = radius * radius
        b0 = 1.0 - radius * radius
        length = int(output.size)
        y = np.zeros(length, dtype=np.float64)
        x = output.astype(np.float64)
        for index in range(length):
            y[index] = b0 * x[index]
            if index >= 1:
                y[index] -= a1 * y[index - 1]
            if index >= 2:
                y[index] -= a2 * y[index - 2]
        output = y.astype(np.float32)
    return output


def formant_vowel(
    duration_s: float,
    *,
    formants: list[tuple[float, float]],
    seed: int,
    amplitude: float = 0.9,
    am_rate_hz: float = 4.2,
    sample_rate_hz: int = CANONICAL_SAMPLE_RATE_HZ,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    count = int(round(duration_s * sample_rate_hz))
    noise = rng.normal(0.0, 0.9, count).astype(np.float32)
    filtered = _resonator_cascade(noise, formants, sample_rate_hz=sample_rate_hz)
    t = np.arange(count, dtype=np.float64) / sample_rate_hz
    amplitude_modulation = (0.75 + 0.25 * np.sin(2.0 * math.pi * am_rate_hz * t)).astype(np.float32)
    filtered = (filtered * amplitude_modulation).astype(np.float32)
    peak = float(np.abs(filtered).max())
    if peak > 0.0:
        filtered = (filtered / peak * amplitude).astype(np.float32)
    return envelope_edges(filtered)


FORMANT_VOWEL_A = [(730.0, 100.0), (1090.0, 120.0), (2440.0, 140.0)]
FORMANT_VOWEL_I = [(270.0, 90.0), (2290.0, 110.0), (3010.0, 130.0)]
FORMANT_VOWEL_O = [(500.0, 90.0), (900.0, 120.0), (2500.0, 140.0)]


@dataclass(frozen=True, slots=True)
class CaseSpec:
    case_id: str
    seed: int
    segments: list[tuple[str, np.ndarray]]

    @property
    def audio(self) -> np.ndarray:
        return np.concatenate([segment for _, segment in self.segments]).astype(np.float32)


def _region_speakers(segments: list[tuple[str, np.ndarray]]) -> list[dict[str, object]]:
    regions: list[dict[str, object]] = []
    start_sample = 0
    for kind, samples in segments:
        end_sample = start_sample + int(samples.size)
        if kind == "speech_a":
            speakers: list[str] = ["A"]
        elif kind == "speech_b":
            speakers = ["B"]
        elif kind == "speech_ab":
            speakers = ["A", "B"]
        else:
            speakers = []
        regions.append(
            {
                "start_sample": start_sample,
                "end_sample": end_sample,
                "speakers": speakers,
            }
        )
        start_sample = end_sample
    return regions


def build_default_cases(
    seed: int = DEFAULT_GENERATOR_SEED,
) -> list[CaseSpec]:
    speech_a = formant_vowel(
        2.0,
        formants=FORMANT_VOWEL_A,
        seed=31,
        amplitude=1.2,
        sample_rate_hz=CANONICAL_SAMPLE_RATE_HZ,
    )
    speech_b = formant_vowel(
        1.6,
        formants=FORMANT_VOWEL_I,
        seed=32,
        amplitude=1.2,
        sample_rate_hz=CANONICAL_SAMPLE_RATE_HZ,
    )
    lead = silence(0.5)
    tail = silence(0.5)
    gap_400 = silence(0.4)
    cases: list[CaseSpec] = [
        CaseSpec(
            case_id="golden_two_utterance_gap400",
            seed=seed,
            segments=[
                ("silence", lead),
                ("speech_a", speech_a),
                ("silence", gap_400),
                ("speech_b", speech_b),
                ("silence", tail),
            ],
        ),
        CaseSpec(
            case_id="golden_single_utterance",
            seed=seed,
            segments=[
                ("silence", lead),
                ("speech_a", speech_a),
                ("silence", tail),
            ],
        ),
        CaseSpec(
            case_id="golden_silence",
            seed=seed,
            segments=[
                ("silence", silence(2.0)),
            ],
        ),
    ]
    return cases


def case_regions(case: CaseSpec) -> list[dict[str, object]]:
    return _region_speakers(case.segments)


def pcm16_bytes(samples: np.ndarray) -> bytes:
    scaled = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    pcm = np.round(scaled * 32767.0).astype(np.int16)
    return pcm.tobytes()
