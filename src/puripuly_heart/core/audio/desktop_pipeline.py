from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import numpy as np

from puripuly_heart.core.audio.format import (
    AudioCaptureDiscontinuity,
    AudioCaptureSpan,
    AudioFrameF32,
)
from puripuly_heart.core.audio.source import AudioSource
from puripuly_heart.core.audio.streaming_resampler import CaptureMappedStreamingResampler


@dataclass(slots=True)
class DesktopPeerPipeline:
    source: AudioSource
    target_sample_rate_hz: int = 16000
    _terminal_discarded_capture: tuple[AudioCaptureSpan, ...] = field(
        default=(),
        init=False,
        repr=False,
    )

    async def frames(self) -> AsyncIterator[AudioFrameF32]:
        normalizer: CaptureMappedStreamingResampler | None = None
        source_format: tuple[int, int] | None = None

        async for frame in self.source.frames():

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
