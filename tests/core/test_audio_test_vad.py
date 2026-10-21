from __future__ import annotations

import numpy as np
import pytest

from puripuly_heart.core.audio.audio_test_vad import AudioTestVadProbe, AudioTestVadSettings
from puripuly_heart.core.audio.format import AudioFrameF32


class LoudnessEngine:
    def __init__(self) -> None:
        self.calls: list[int] = []

    def speech_probability(self, samples: np.ndarray, *, sample_rate_hz: int) -> float:
        assert sample_rate_hz == 16000
        self.calls.append(int(samples.size))
        return 0.9 if float(np.max(np.abs(samples))) > 0.1 else 0.05

    def reset(self) -> None:
        return None


def _settings(**overrides) -> AudioTestVadSettings:
    values = {"speech_threshold": 0.5, "hangover_ms": 64, "ring_buffer_ms": 300}
    values.update(overrides)
    return AudioTestVadSettings(**values)


def _mono(amplitude: float, chunks: int) -> AudioFrameF32:
    return AudioFrameF32(
        samples=np.full(512 * chunks, amplitude, dtype=np.float32),
        sample_rate_hz=16000,
    )


def test_device_rate_stereo_frames_are_normalized_into_vad_chunks() -> None:
    engine = LoudnessEngine()
    probe = AudioTestVadProbe(engine=engine, settings=_settings())
    stereo = np.column_stack(
        (np.full(48000, 0.4, dtype=np.float32), np.full(48000, 0.2, dtype=np.float32))
    )

    samples = probe.process(AudioFrameF32(samples=stereo, sample_rate_hz=48000, channels=2))

    assert 29 <= len(samples) <= 31
    assert set(engine.calls) == {512}
    assert samples[-1].probability == pytest.approx(0.9)
    assert samples[-1].level == pytest.approx(0.3, abs=0.02)
    assert samples[-1].threshold == pytest.approx(0.5)


def test_partial_chunks_carry_over_between_frames() -> None:
    probe = AudioTestVadProbe(engine=LoudnessEngine(), settings=_settings())
    half = AudioFrameF32(samples=np.zeros(256, dtype=np.float32), sample_rate_hz=16000)

    assert probe.process(half) == ()
    assert len(probe.process(half)) == 1


def test_speech_follows_gating_debounce_and_hangover_not_raw_probability() -> None:
    probe = AudioTestVadProbe(
        engine=LoudnessEngine(),
        settings=_settings(start_debounce_chunks=3, start_commit_chunks=3),
    )

    onset = probe.process(_mono(0.5, 3))
    assert [sample.speech for sample in onset] == [False, False, True]

    trailing = probe.process(_mono(0.0, 3))
    assert trailing[0].speech is True
    assert trailing[-1].speech is False


def test_threshold_changes_apply_to_following_chunks() -> None:
    probe = AudioTestVadProbe(engine=LoudnessEngine(), settings=_settings())
    assert probe.process(_mono(0.5, 1))[-1].speech is True
    probe.process(_mono(0.0, 4))

    probe.reconfigure(_settings(speech_threshold=0.95))
    samples = probe.process(_mono(0.5, 2))

    assert [sample.speech for sample in samples] == [False, False]
    assert samples[-1].threshold == pytest.approx(0.95)


def test_format_change_and_invalid_pcm_are_rejected() -> None:
    probe = AudioTestVadProbe(engine=LoudnessEngine(), settings=_settings())
    probe.process(_mono(0.0, 1))

    with pytest.raises(ValueError, match="capture_format_changed"):
        probe.process(AudioFrameF32(samples=np.zeros(480, dtype=np.float32), sample_rate_hz=48000))

    fresh = AudioTestVadProbe(engine=LoudnessEngine(), settings=_settings())
    with pytest.raises(ValueError, match="invalid_pcm"):
        fresh.process(
            AudioFrameF32(samples=np.array([np.nan] * 512, dtype=np.float32), sample_rate_hz=16000)
        )
