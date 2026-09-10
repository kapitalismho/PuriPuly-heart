from __future__ import annotations

import contextlib
import logging
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field, replace

import numpy as np

from puripuly_heart.core.audio.diagnostics import compute_audio_frame_metrics
from puripuly_heart.core.audio.format import AudioCaptureSpan, AudioFrameF32
from puripuly_heart.core.audio.source import AudioSource
from puripuly_heart.core.audio.streaming_resampler import MonoFirstStreamingResampler

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class DesktopPeerPipeline:
    source: AudioSource
    target_sample_rate_hz: int = 16000
    is_detailed_enabled: Callable[[], bool] | None = None
    log_detailed: Callable[[str], object] | None = None
    _logged_formats: set[tuple[int, int]] = field(default_factory=set, init=False, repr=False)
    _diag_accumulated_audio_ms: float = field(default=0.0, init=False, repr=False)

    async def frames(self) -> AsyncIterator[AudioFrameF32]:
        resampler: MonoFirstStreamingResampler | None = None
        source_format: tuple[int, int] | None = None
        normalized_epoch: int | None = None
        normalized_next_sample = 0
        pending_capture: AudioCaptureSpan | None = None
        last_capture: AudioCaptureSpan | None = None

        async for frame in self.source.frames():
            format_key = (frame.sample_rate_hz, frame.channels)
            if format_key not in self._logged_formats:
                self._logged_formats.add(format_key)
                logger.info(
                    "Desktop peer audio format: source_rate=%sHz source_channels=%s -> target_rate=%sHz",
                    frame.sample_rate_hz,
                    frame.channels,
                    self.target_sample_rate_hz,
                )

            frame_format = (frame.sample_rate_hz, frame.channels)
            if source_format is None:
                source_format = frame_format
                resampler = self._new_resampler(frame)
            elif frame_format != source_format:
                raise ValueError(
                    "source audio format changed during streaming: "
                    f"expected {source_format[0]}Hz/{source_format[1]}ch, "
                    f"got {frame.sample_rate_hz}Hz/{frame.channels}ch"
                )

            capture = frame.capture
            if capture is not None:
                discontinuous = capture.discontinuity_before is not None
                if normalized_epoch is None:
                    normalized_epoch = capture.capture_epoch
                    normalized_next_sample = 0
                elif capture.capture_epoch != normalized_epoch:
                    normalized_epoch = capture.capture_epoch
                    normalized_next_sample = 0
                    discontinuous = True
                elif (
                    capture.discontinuity_before is not None
                    and capture.discontinuity_before.kind == "known_loss"
                    and capture.discontinuity_before.lost_source_samples is not None
                ):
                    normalized_next_sample += round(
                        capture.discontinuity_before.lost_source_samples
                        * self.target_sample_rate_hz
                        / capture.source_sample_rate_hz
                    )
                if discontinuous and resampler is not None:
                    resampler = self._new_resampler(frame)
                    pending_capture = None
                pending_capture = self._combine_capture(pending_capture, capture)
                last_capture = capture

            assert resampler is not None
            normalized = resampler.resample_chunk(frame.samples)
            self._maybe_log_peer_diagnostics(
                source_rate=frame.sample_rate_hz,
                source_channels=frame.channels,
                normalized=normalized,
            )
            if normalized.size:
                output_capture = None
                if pending_capture is not None:
                    output_capture = pending_capture.with_normalized_range(
                        sample_rate_hz=self.target_sample_rate_hz,
                        start_sample=normalized_next_sample,
                        end_sample=normalized_next_sample + int(normalized.size),
                    )
                    normalized_next_sample += int(normalized.size)
                    pending_capture = None
                yield self._build_output_frame(
                    normalized.reshape(-1),
                    capture=output_capture,
                )

        if resampler is None:
            return

        tail = resampler.flush()
        if tail.size:
            tail_capture = pending_capture or last_capture
            output_capture = None
            if tail_capture is not None:
                output_capture = tail_capture.with_normalized_range(
                    sample_rate_hz=self.target_sample_rate_hz,
                    start_sample=normalized_next_sample,
                    end_sample=normalized_next_sample + int(tail.size),
                )
            yield self._build_output_frame(tail.reshape(-1), capture=output_capture)

    async def close(self) -> None:
        await self.source.close()

    def _new_resampler(self, frame: AudioFrameF32) -> MonoFirstStreamingResampler:
        return MonoFirstStreamingResampler(
            input_sample_rate_hz=frame.sample_rate_hz,
            output_sample_rate_hz=self.target_sample_rate_hz,
            input_channels=frame.channels,
        )

    @staticmethod
    def _combine_capture(
        pending: AudioCaptureSpan | None,
        current: AudioCaptureSpan,
    ) -> AudioCaptureSpan:
        if pending is None or pending.capture_epoch != current.capture_epoch:
            return current
        return replace(
            pending,
            source_end_sample=current.source_end_sample,
            source_end_monotonic_s=current.source_end_monotonic_s,
        )


    def _maybe_log_peer_diagnostics(
        self,
        *,
        source_rate: int,
        source_channels: int,
        normalized: np.ndarray,
    ) -> None:
        if self.is_detailed_enabled is None or self.log_detailed is None:
            return
        detailed_enabled = False
        with contextlib.suppress(Exception):
            detailed_enabled = bool(self.is_detailed_enabled())
        if not detailed_enabled:
            return

        with contextlib.suppress(Exception):
            frame = AudioFrameF32(
                samples=normalized.reshape(-1),
                sample_rate_hz=self.target_sample_rate_hz,
                channels=1,
            )
            metrics = compute_audio_frame_metrics(frame)
            self._diag_accumulated_audio_ms += metrics.audio_ms
            if self._diag_accumulated_audio_ms < 1000.0:
                return

            self._diag_accumulated_audio_ms = 0.0
            self.log_detailed(
                f"[AudioDiag][PeerPipeline] source_rate={source_rate} "
                f"source_channels={source_channels} target_rate={self.target_sample_rate_hz} "
                f"samples={metrics.samples} audio_ms={metrics.audio_ms:.1f} "
                f"rms_db={metrics.rms_db:.1f} peak_db={metrics.peak_db:.1f} "
                f"zero_ratio={metrics.zero_ratio:.3f}"
            )

    def _build_output_frame(
        self,
        samples: np.ndarray,
        *,
        capture: AudioCaptureSpan | None = None,
    ) -> AudioFrameF32:
        return AudioFrameF32(
            samples=samples,
            sample_rate_hz=self.target_sample_rate_hz,
            channels=1,
            capture=capture,
        )
