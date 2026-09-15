from __future__ import annotations

import numpy as np
import pytest

from puripuly_heart.core.audio.diagnostics import (
    EXPECTED_FAULT_SIGNATURES,
    AudioFaultProfile,
    AudioFrameMetrics,
    FaultInjectingAudioSource,
    apply_audio_fault_profile,
    compute_audio_frame_metrics,
)
from puripuly_heart.core.audio.format import AudioFrameF32


def test_compute_audio_frame_metrics_reports_channel_values() -> None:
    frame = AudioFrameF32(
        samples=np.array([[0.0, 1.0], [0.0, -1.0]], dtype=np.float32),
        sample_rate_hz=48000,
        channels=2,
    )

    metrics = compute_audio_frame_metrics(frame)

    assert isinstance(metrics, AudioFrameMetrics)
    assert metrics.samples == 4
    assert round(metrics.audio_ms, 3) == 0.042
    assert metrics.peak_db == 0.0
    assert metrics.channel_rms_db[0] <= -119.0
    assert metrics.channel_rms_db[1] == 0.0


def test_apply_audio_fault_profiles_are_deterministic() -> None:
    frame = AudioFrameF32(
        samples=np.array([[1.0, 0.5], [-1.0, -0.5]], dtype=np.float32),
        sample_rate_hz=16000,
        channels=2,
    )

    muted = apply_audio_fault_profile(frame, AudioFaultProfile.CAPTURE_SILENT_FIRST_CHANNEL)
    attenuated = apply_audio_fault_profile(frame, AudioFaultProfile.CAPTURE_ATTENUATE_40DB)
    noisy = apply_audio_fault_profile(
        frame, AudioFaultProfile.CAPTURE_NEAR_SILENCE_NOISE, sequence_index=3
    )
    noisy_again = apply_audio_fault_profile(
        frame, AudioFaultProfile.CAPTURE_NEAR_SILENCE_NOISE, sequence_index=3
    )
    noisy_other = apply_audio_fault_profile(
        frame, AudioFaultProfile.CAPTURE_NEAR_SILENCE_NOISE, sequence_index=4
    )
    dropped = apply_audio_fault_profile(
        frame, AudioFaultProfile.CAPTURE_BUFFER_DROPOUTS, sequence_index=1
    )

    np.testing.assert_allclose(muted.samples[:, 0], np.array([0.0, 0.0], dtype=np.float32))
    np.testing.assert_allclose(muted.samples[:, 1], np.array([0.5, -0.5], dtype=np.float32))
    np.testing.assert_allclose(attenuated.samples, frame.samples * np.float32(0.01))
    assert float(np.max(np.abs(noisy.samples))) <= 0.004
    np.testing.assert_allclose(noisy.samples, noisy_again.samples)
    assert not np.array_equal(noisy.samples, noisy_other.samples)
    np.testing.assert_allclose(dropped.samples, np.zeros_like(frame.samples))
    assert "stt_input_low_snr_vad_pass" in EXPECTED_FAULT_SIGNATURES


class StubAudioSource:
    def __init__(self, frames: list[AudioFrameF32]) -> None:
        self._frames = frames
        self.closed = False

    async def frames(self):
        for frame in self._frames:
            yield frame

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_diagnostic_audio_source_fault_provider_failure_yields_original_frame() -> None:
    frame = AudioFrameF32(
        samples=np.ones(16000, dtype=np.float32), sample_rate_hz=16000, channels=1
    )

    def fail_fault_profile():
        raise RuntimeError("fault profile unavailable")

    wrapper = FaultInjectingAudioSource(
        source=StubAudioSource([frame]),
        fault_profile_provider=fail_fault_profile,
    )

    frames = [frame async for frame in wrapper.frames()]

    assert len(frames) == 1
    assert frames[0] is frame
