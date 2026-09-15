from __future__ import annotations

import contextlib
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import AsyncIterator

import numpy as np

from puripuly_heart.core.audio.format import AudioFrameF32, reshape_audio_samples_f32
from puripuly_heart.core.audio.source import AudioSource


class AudioFaultProfile(StrEnum):
    NONE = "none"
    CAPTURE_SILENT_FIRST_CHANNEL = "capture_silent_first_channel"
    CAPTURE_ATTENUATE_40DB = "capture_attenuate_40db"
    CAPTURE_NEAR_SILENCE_NOISE = "capture_near_silence_noise"
    CAPTURE_BUFFER_DROPOUTS = "capture_buffer_dropouts"
    STT_INPUT_LOW_SNR_VAD_PASS = "stt_input_low_snr_vad_pass"


EXPECTED_FAULT_SIGNATURES = {
    "capture_silent_first_channel": (
        "Capture logs show one muted channel; VAD may miss speech after mono mixdown."
    ),
    "capture_attenuate_40db": (
        "Capture RMS and VAD-input RMS drop sharply; VAD may miss or local_qwen may decode "
        "garbage."
    ),
    "capture_near_silence_noise": (
        "Capture RMS is very low with near-silence noise; VAD should usually stay idle."
    ),
    "capture_buffer_dropouts": (
        "Capture logs alternate normal and zeroed chunks; VAD/STT chunk counts reveal dropout "
        "sensitivity."
    ),
    "stt_input_low_snr_vad_pass": (
        "VAD can start from real capture while STT-input and local_qwen logs show injected "
        "low-SNR audio."
    ),
}


@dataclass(frozen=True, slots=True)
class AudioFrameMetrics:
    samples: int
    audio_ms: float
    rms_db: float
    peak_db: float
    zero_ratio: float
    channel_rms_db: tuple[float, ...] = ()
    channel_peak_db: tuple[float, ...] = ()


def _safe_db(value: float) -> float:
    if value <= 0.0:
        return -120.0
    return round(float(20.0 * np.log10(max(value, 1e-6))), 1)


def compute_audio_frame_metrics(frame: AudioFrameF32) -> AudioFrameMetrics:
    samples = np.asarray(frame.samples, dtype=np.float32)
    if samples.size == 0:
        return AudioFrameMetrics(
            samples=0,
            audio_ms=0.0,
            rms_db=-120.0,
            peak_db=-120.0,
            zero_ratio=1.0,
        )

    reshaped = reshape_audio_samples_f32(samples, channels=frame.channels)
    sample_frames = int(reshaped.shape[0])
    audio_ms = (
        sample_frames * 1000.0 / float(frame.sample_rate_hz) if frame.sample_rate_hz > 0 else 0.0
    )
    rms = float(np.sqrt(np.mean(np.square(samples))))
    peak = float(np.max(np.abs(samples)))
    zero_ratio = float(np.mean(np.abs(samples) < 1e-6))

    channel_rms: list[float] = []
    channel_peak: list[float] = []
    channels = (
        (reshaped,)
        if reshaped.ndim == 1
        else tuple(reshaped[:, idx] for idx in range(reshaped.shape[1]))
    )
    for channel in channels:
        channel_rms.append(
            _safe_db(float(np.sqrt(np.mean(np.square(channel)))) if channel.size else 0.0)
        )
        channel_peak.append(_safe_db(float(np.max(np.abs(channel))) if channel.size else 0.0))

    return AudioFrameMetrics(
        samples=int(samples.size),
        audio_ms=audio_ms,
        rms_db=_safe_db(rms),
        peak_db=_safe_db(peak),
        zero_ratio=round(zero_ratio, 3),
        channel_rms_db=tuple(channel_rms),
        channel_peak_db=tuple(channel_peak),
    )


def normalize_audio_fault_profile(profile: AudioFaultProfile | str | None) -> AudioFaultProfile:
    if profile is None:
        return AudioFaultProfile.NONE
    return AudioFaultProfile(profile)


def apply_audio_fault_profile(
    frame: AudioFrameF32,
    profile: AudioFaultProfile | str | None,
    *,
    sequence_index: int = 0,
) -> AudioFrameF32:
    resolved = normalize_audio_fault_profile(profile)
    if resolved in (AudioFaultProfile.NONE, AudioFaultProfile.STT_INPUT_LOW_SNR_VAD_PASS):
        return frame

    samples = np.asarray(frame.samples, dtype=np.float32).copy()
    if resolved is AudioFaultProfile.CAPTURE_SILENT_FIRST_CHANNEL:
        reshaped = reshape_audio_samples_f32(samples, channels=frame.channels).copy()
        if reshaped.ndim == 2:
            reshaped[:, 0] = 0.0
            return replace(frame, samples=reshaped)
        return replace(frame, samples=np.zeros_like(samples))

    if resolved is AudioFaultProfile.CAPTURE_ATTENUATE_40DB:
        return replace(frame, samples=samples * np.float32(0.01))

    if resolved is AudioFaultProfile.CAPTURE_NEAR_SILENCE_NOISE:
        flat = np.arange(samples.size, dtype=np.float32) + np.float32(sequence_index * 17)
        noise = np.sin(flat * np.float32(12.9898)) * np.float32(0.003)
        return replace(frame, samples=noise.reshape(samples.shape).astype(np.float32))

    if resolved is AudioFaultProfile.CAPTURE_BUFFER_DROPOUTS:
        if sequence_index % 2 == 1:
            samples.fill(0.0)
        return replace(frame, samples=samples)

    raise AssertionError(f"Unhandled audio fault profile: {resolved}")


@dataclass(slots=True)
class FaultInjectingAudioSource(AudioSource):
    source: AudioSource
    fault_profile: AudioFaultProfile | str = AudioFaultProfile.NONE
    fault_profile_provider: Callable[[], AudioFaultProfile | str | None] | None = None
    _sequence_index: int = field(init=False, default=0)

    def _current_fault_profile(self) -> AudioFaultProfile:
        if self.fault_profile_provider is not None:
            return normalize_audio_fault_profile(self.fault_profile_provider())
        return normalize_audio_fault_profile(self.fault_profile)

    async def frames(self) -> AsyncIterator[AudioFrameF32]:
        async for frame in self.source.frames():
            profile = self._safe_current_fault_profile()
            if profile is AudioFaultProfile.NONE:
                yield frame
                continue

            output = self._safe_apply_audio_fault_profile(
                frame,
                profile,
                sequence_index=self._sequence_index,
            )
            self._sequence_index += 1
            yield output

    def _safe_current_fault_profile(self) -> AudioFaultProfile:
        with contextlib.suppress(Exception):
            return self._current_fault_profile()
        return AudioFaultProfile.NONE

    def _safe_apply_audio_fault_profile(
        self,
        frame: AudioFrameF32,
        profile: AudioFaultProfile,
        *,
        sequence_index: int,
    ) -> AudioFrameF32:
        with contextlib.suppress(Exception):
            return apply_audio_fault_profile(frame, profile, sequence_index=sequence_index)
        return frame

    async def close(self) -> None:
        await self.source.close()
