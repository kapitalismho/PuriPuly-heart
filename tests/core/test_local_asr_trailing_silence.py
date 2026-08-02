from __future__ import annotations

import numpy as np
import pytest

from puripuly_heart.core.local_asr.trailing_silence import (
    LOCAL_ASR_TRAILING_SILENCE_SAFETY_TAIL_MS,
    trim_local_asr_trailing_silence,
)


@pytest.mark.parametrize("trailing_silence_ms", [None, 0, 64, 128])
def test_local_asr_trim_retains_audio_at_or_below_safety_tail(
    trailing_silence_ms: int | None,
) -> None:
    samples = np.arange(16_000, dtype=np.float32)

    trim = trim_local_asr_trailing_silence(
        samples,
        sample_rate_hz=16_000,
        trailing_silence_ms=trailing_silence_ms,
    )

    assert LOCAL_ASR_TRAILING_SILENCE_SAFETY_TAIL_MS == 128
    assert np.array_equal(trim.samples_f32, samples)
    assert trim.audio_ms_before == 1000.0
    assert trim.reported_trailing_silence_ms == trailing_silence_ms
    assert trim.trimmed_samples == 0
    assert trim.actual_trimmed_ms == 0.0
    assert trim.submitted_audio_ms == 1000.0


def test_local_asr_trim_removes_only_suffix_beyond_safety_tail() -> None:
    samples = np.arange(16_000, dtype=np.float32)

    trim = trim_local_asr_trailing_silence(
        samples,
        sample_rate_hz=16_000,
        trailing_silence_ms=400,
    )

    assert trim.trimmed_samples == 4352
    assert trim.actual_trimmed_ms == 272.0
    assert trim.submitted_audio_ms == 728.0
    assert np.array_equal(trim.samples_f32, samples[:11_648])


def test_local_asr_trim_preserves_leading_preroll_and_internal_pauses() -> None:
    samples = np.ones(32_000, dtype=np.float32)
    samples[:512] = 0.0
    samples[8_000:9_000] = 0.0

    trim = trim_local_asr_trailing_silence(
        samples,
        sample_rate_hz=16_000,
        trailing_silence_ms=400,
    )

    assert np.array_equal(trim.samples_f32, samples[:27_648])
    assert np.count_nonzero(trim.samples_f32[:512]) == 0
    assert np.count_nonzero(trim.samples_f32[8_000:9_000]) == 0


def test_local_asr_trim_bounds_overreported_silence_to_actual_buffer() -> None:
    samples = np.arange(16_000, dtype=np.float32)

    trim = trim_local_asr_trailing_silence(
        samples,
        sample_rate_hz=16_000,
        trailing_silence_ms=5_000,
    )

    assert trim.samples_f32.size == 2048
    assert np.array_equal(trim.samples_f32, samples[:2048])
    assert trim.actual_trimmed_ms == 872.0
    assert trim.submitted_audio_ms == 128.0


def test_local_asr_trim_keeps_very_short_valid_utterance_nonempty() -> None:
    samples = np.arange(1000, dtype=np.float32)

    trim = trim_local_asr_trailing_silence(
        samples,
        sample_rate_hz=16_000,
        trailing_silence_ms=5_000,
    )

    assert np.array_equal(trim.samples_f32, samples)
    assert trim.trimmed_samples == 0
    assert trim.submitted_audio_ms == 62.5


def test_local_asr_trim_preserves_empty_buffer() -> None:
    trim = trim_local_asr_trailing_silence(
        np.empty((0,), dtype=np.float32),
        sample_rate_hz=16_000,
        trailing_silence_ms=400,
    )

    assert trim.samples_f32.size == 0
    assert trim.audio_ms_before == 0.0
    assert trim.actual_trimmed_ms == 0.0
    assert trim.submitted_audio_ms == 0.0


def test_local_asr_trim_max_duration_zero_report_keeps_input() -> None:
    samples = np.arange(160_000, dtype=np.float32)

    trim = trim_local_asr_trailing_silence(
        samples,
        sample_rate_hz=16_000,
        trailing_silence_ms=0,
    )

    assert np.array_equal(trim.samples_f32, samples)
    assert trim.submitted_audio_ms == 10_000.0


def test_local_asr_trim_rejects_invalid_sample_rate() -> None:
    with pytest.raises(ValueError, match="sample_rate_hz must be > 0"):
        trim_local_asr_trailing_silence(
            np.ones(1, dtype=np.float32),
            sample_rate_hz=0,
            trailing_silence_ms=400,
        )
