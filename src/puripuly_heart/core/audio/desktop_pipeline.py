from __future__ import annotations

import contextlib
import logging
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field

import numpy as np

from puripuly_heart.core.audio.diagnostics import compute_audio_frame_metrics
from puripuly_heart.core.audio.format import (
    AudioCaptureDiscontinuity,
    AudioCaptureSpan,
    AudioFrameF32,
)
from puripuly_heart.core.audio.source import AudioSource
from puripuly_heart.core.audio.streaming_resampler import CaptureMappedStreamingResampler

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class DesktopPeerPipeline:
    source: AudioSource
    target_sample_rate_hz: int = 16000
    is_detailed_enabled: Callable[[], bool] | None = None
    log_detailed: Callable[[str], object] | None = None
    _logged_formats: set[tuple[int, int]] = field(default_factory=set, init=False, repr=False)
    _diag_accumulated_audio_ms: float = field(default=0.0, init=False, repr=False)
    _terminal_discarded_capture: tuple[AudioCaptureSpan, ...] = field(
        default=(),
        init=False,
        repr=False,
    )

    async def frames(self) -> AsyncIterator[AudioFrameF32]:
        normalizer: CaptureMappedStreamingResampler | None = None
        source_format: tuple[int, int] | None = None

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
                normalizer = CaptureMappedStreamingResampler(
                    input_sample_rate_hz=frame.sample_rate_hz,
                    output_sample_rate_hz=self.target_sample_rate_hz,
                    input_channels=frame.channels,
                )
            elif frame_format != source_format:
                raise ValueError(
                    "source audio format changed during streaming: "
                    f"expected {source_format[0]}Hz/{source_format[1]}ch, "
                    f"got {frame.sample_rate_hz}Hz/{frame.channels}ch"
                )

            assert normalizer is not None
            normalized, output_capture, discarded = normalizer.process(
                frame.samples,
                frame.capture,
            )
            self._maybe_log_peer_diagnostics(
                source_rate=frame.sample_rate_hz,
                source_channels=frame.channels,
                normalized=normalized,
            )
            if normalized.size or discarded:
                yield self._build_output_frame(
                    normalized.reshape(-1),
                    capture=output_capture,
                    discarded_capture_before=discarded,
                    discontinuity_before=(
                        frame.capture.discontinuity_before
                        if frame.capture is not None
                        else frame.discontinuity_before
                    ),
                )

        if normalizer is None:
            return
        terminal_reason = self.terminal_reason
        orderly = terminal_reason in {None, "closed"}
        tail, output_capture, discarded = normalizer.finish(orderly=orderly)
        if not orderly:
            self._terminal_discarded_capture = discarded
            return
        if tail.size:
            yield self._build_output_frame(
                tail.reshape(-1),
                capture=output_capture,
            )

    async def close(self) -> None:
        await self.source.close()

    @property
    def terminal_reason(self) -> str | None:
        reason = getattr(self.source, "terminal_reason", None)
        return reason if isinstance(reason, str) else None

    @property
    def terminal_discarded_capture(self) -> tuple[AudioCaptureSpan, ...]:
        return self._terminal_discarded_capture



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
        discarded_capture_before: tuple[AudioCaptureSpan, ...] = (),
        discontinuity_before: AudioCaptureDiscontinuity | None = None,
    ) -> AudioFrameF32:
        return AudioFrameF32(
            samples=samples,
            sample_rate_hz=self.target_sample_rate_hz,
            channels=1,
            capture=capture,
            discarded_capture_before=discarded_capture_before,
            discontinuity_before=discontinuity_before,
        )
