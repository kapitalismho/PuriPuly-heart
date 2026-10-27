from __future__ import annotations

import numpy as np

from puripuly_heart.core.audio.format import (
    AudioFrameF32,
    float32_to_pcm16le_bytes,
    mixdown_to_mono_f32,
    normalize_audio_f32,
    normalize_audio_frame_f32,
    pcm16le_bytes_to_float32,
    resample_f32_linear,
    reshape_audio_samples_f32,
)


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
    second = normalize_audio_f32(
        first.samples, input_sample_rate_hz=16000, target_sample_rate_hz=16000
    )
    assert first.sample_rate_hz == 16000
    assert second.sample_rate_hz == 16000
    assert second.samples.shape == first.samples.shape


def test_reshape_audio_samples_reconstructs_interleaved_frames():
    interleaved = np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float32)
    reshaped = reshape_audio_samples_f32(interleaved, channels=2)
    assert reshaped.shape == (2, 2)
    assert np.allclose(reshaped, np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32))


def test_normalize_audio_frame_uses_channel_metadata_for_mixdown():
    frame = AudioFrameF32(
        samples=np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float32),
        sample_rate_hz=16000,
        channels=2,
    )
    normalized = normalize_audio_frame_f32(frame, target_sample_rate_hz=16000)
    assert normalized.channels == 1
    assert np.allclose(normalized.samples, np.array([0.5, 0.5], dtype=np.float32))


def test_normalize_audio_f32_same_rate_stereo_only_mixdowns() -> None:
    raw = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    normalized = normalize_audio_f32(
        raw,
        input_sample_rate_hz=16000,
        target_sample_rate_hz=16000,
    )
    assert normalized.sample_rate_hz == 16000
    assert normalized.channels == 1
    assert np.allclose(normalized.samples, np.array([0.5, 0.5], dtype=np.float32))
