from __future__ import annotations

import numpy as np

from ajebal_daera_translator.core.audio.format import (
    float32_to_pcm16le_bytes,
    mixdown_to_mono_f32,
    normalize_audio_f32,
    pcm16le_bytes_to_float32,
    resample_f32_linear,
)
from ajebal_daera_translator.core.audio.ring_buffer import RingBufferF32


def test_mixdown_to_mono():
    stereo = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    mono = mixdown_to_mono_f32(stereo)
    assert mono.shape == (2,)
    assert np.allclose(mono, np.array([0.5, 0.5], dtype=np.float32))


def test_float32_pcm16_roundtrip():
    samples = np.array([-1.0, -0.5, 0.0, 0.5, 0.9999], dtype=np.float32)
    data = float32_to_pcm16le_bytes(samples)
    restored = pcm16le_bytes_to_float32(data)
    assert restored.shape == samples.shape
    assert np.all(restored <= 1.0)
    assert np.all(restored >= -1.0)


def test_resample_length_ratio():
    src = np.linspace(-1.0, 1.0, num=480, dtype=np.float32)
    dst = resample_f32_linear(src, from_rate_hz=48000, to_rate_hz=16000)
    assert dst.shape[0] == 160


def test_normalize_audio_resamples_only_when_needed():
    raw = np.linspace(-1.0, 1.0, num=480, dtype=np.float32)
    first = normalize_audio_f32(raw, input_sample_rate_hz=48000, target_sample_rate_hz=16000)
    second = normalize_audio_f32(first.samples, input_sample_rate_hz=16000, target_sample_rate_hz=16000)
    assert first.sample_rate_hz == 16000
    assert second.sample_rate_hz == 16000
    assert second.samples.shape == first.samples.shape


def test_ring_buffer_returns_last_samples():
    rb = RingBufferF32(capacity_samples=10)
    rb.append(np.arange(6, dtype=np.float32))
    rb.append(np.arange(6, 16, dtype=np.float32))
    last = rb.get_last_samples(5)
    assert np.allclose(last, np.array([11, 12, 13, 14, 15], dtype=np.float32))

