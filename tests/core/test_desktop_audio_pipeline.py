from __future__ import annotations

import numpy as np
import pytest

import puripuly_heart.core.audio.desktop_pipeline as desktop_pipeline_module
from puripuly_heart.core.audio.desktop_pipeline import DesktopPeerPipeline
from puripuly_heart.core.audio.format import (
    AudioCaptureDiscontinuity,
    AudioCaptureSpan,
    AudioFrameF32,
)


class StubDesktopAudioSource:
    def __init__(self, frames):
        self._frames = frames
        self.closed = False

    async def frames(self):
        for frame in self._frames:
            yield frame

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_desktop_pipeline_outputs_16khz_vad_ready_frames():
    source = StubDesktopAudioSource(
        frames=[
            AudioFrameF32(
                sample_rate_hz=48000,
                samples=np.ones(4800, dtype=np.float32),
            )
        ]
    )
    pipeline = DesktopPeerPipeline(source=source)

    frames = [frame async for frame in pipeline.frames()]
    combined = np.concatenate([frame.samples for frame in frames])

    assert len(frames) >= 1
    assert all(frame.sample_rate_hz == 16000 for frame in frames)
    assert all(frame.samples.dtype == np.float32 for frame in frames)
    assert all(frame.samples.ndim == 1 for frame in frames)
    assert all(frame.channels == 1 for frame in frames)
    assert combined.shape == (1600,)


@pytest.mark.asyncio
async def test_desktop_pipeline_downmixes_interleaved_multichannel_frames():
    source = StubDesktopAudioSource(
        frames=[
            AudioFrameF32(
                sample_rate_hz=16000,
                channels=2,
                samples=np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float32),
            )
        ]
    )
    pipeline = DesktopPeerPipeline(source=source)

    frame = await pipeline.frames().__anext__()

    assert frame.sample_rate_hz == 16000
    assert np.allclose(frame.samples, np.array([0.5, 0.5], dtype=np.float32))


@pytest.mark.asyncio
async def test_desktop_pipeline_logs_post_resample_diagnostics_when_detailed() -> None:
    log_lines: list[str] = []
    source = StubDesktopAudioSource(
        frames=[
            AudioFrameF32(
                sample_rate_hz=16000,
                channels=1,
                samples=np.ones(16000, dtype=np.float32),
            )
        ]
    )
    pipeline = DesktopPeerPipeline(
        source=source,
        target_sample_rate_hz=16000,
        is_detailed_enabled=lambda: True,
        log_detailed=log_lines.append,
    )

    frames = [frame async for frame in pipeline.frames()]

    assert len(frames) == 1
    assert any("[AudioDiag][PeerPipeline]" in line for line in log_lines)
    assert any("source_rate=16000" in line and "target_rate=16000" in line for line in log_lines)


@pytest.mark.asyncio
async def test_desktop_pipeline_skips_metrics_when_not_detailed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        desktop_pipeline_module,
        "compute_audio_frame_metrics",
        lambda _frame: (_ for _ in ()).throw(
            AssertionError("Basic mode must not compute peer pipeline metrics")
        ),
    )
    source = StubDesktopAudioSource(
        frames=[
            AudioFrameF32(
                sample_rate_hz=16000,
                channels=1,
                samples=np.ones(16000, dtype=np.float32),
            )
        ]
    )
    pipeline = DesktopPeerPipeline(
        source=source,
        target_sample_rate_hz=16000,
        is_detailed_enabled=lambda: False,
        log_detailed=lambda _message: (_ for _ in ()).throw(
            AssertionError("Basic mode must not log peer AudioDiag")
        ),
    )

    frames = [frame async for frame in pipeline.frames()]

    assert len(frames) == 1


@pytest.mark.asyncio
async def test_desktop_pipeline_detailed_predicate_failure_still_yields_frames() -> None:
    def fail_detailed_enabled():
        raise RuntimeError("detailed predicate failed")

    source = StubDesktopAudioSource(
        frames=[
            AudioFrameF32(
                sample_rate_hz=16000,
                channels=1,
                samples=np.ones(16000, dtype=np.float32),
            )
        ]
    )
    pipeline = DesktopPeerPipeline(
        source=source,
        target_sample_rate_hz=16000,
        is_detailed_enabled=fail_detailed_enabled,
        log_detailed=lambda _message: (_ for _ in ()).throw(
            AssertionError("failed detailed predicate must not log")
        ),
    )

    frames = [frame async for frame in pipeline.frames()]

    assert len(frames) == 1
    np.testing.assert_allclose(frames[0].samples, np.ones(16000, dtype=np.float32))


@pytest.mark.asyncio
async def test_desktop_pipeline_metric_failure_still_yields_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        desktop_pipeline_module,
        "compute_audio_frame_metrics",
        lambda _frame: (_ for _ in ()).throw(RuntimeError("peer metrics failed")),
    )
    source = StubDesktopAudioSource(
        frames=[
            AudioFrameF32(
                sample_rate_hz=16000,
                channels=1,
                samples=np.ones(16000, dtype=np.float32),
            )
        ]
    )
    pipeline = DesktopPeerPipeline(
        source=source,
        target_sample_rate_hz=16000,
        is_detailed_enabled=lambda: True,
        log_detailed=lambda _message: None,
    )

    frames = [frame async for frame in pipeline.frames()]

    assert len(frames) == 1
    np.testing.assert_allclose(frames[0].samples, np.ones(16000, dtype=np.float32))


@pytest.mark.asyncio
async def test_desktop_pipeline_log_failure_still_yields_frames() -> None:
    def fail_log(_message: str) -> None:
        raise RuntimeError("peer diagnostic log failed")

    source = StubDesktopAudioSource(
        frames=[
            AudioFrameF32(
                sample_rate_hz=16000,
                channels=1,
                samples=np.ones(16000, dtype=np.float32),
            )
        ]
    )
    pipeline = DesktopPeerPipeline(
        source=source,
        target_sample_rate_hz=16000,
        is_detailed_enabled=lambda: True,
        log_detailed=fail_log,
    )

    frames = [frame async for frame in pipeline.frames()]

    assert len(frames) == 1
    np.testing.assert_allclose(frames[0].samples, np.ones(16000, dtype=np.float32))


@pytest.mark.asyncio
async def test_desktop_pipeline_close_closes_underlying_source():
    source = StubDesktopAudioSource(frames=[])
    pipeline = DesktopPeerPipeline(source=source)

    await pipeline.close()

    assert source.closed is True

@pytest.mark.asyncio
async def test_desktop_pipeline_discards_resampler_residue_on_unexpected_source_loss():
    capture = AudioCaptureSpan(
        capture_epoch=1,
        callback_sequence=0,
        source_sample_rate_hz=48000,
        source_start_sample=0,
        source_end_sample=1024,
        source_start_monotonic_s=0.0,
        source_end_monotonic_s=1024 / 48000,
    )

    class TerminalSource(StubDesktopAudioSource):
        terminal_reason = "target_exited"

    pipeline = DesktopPeerPipeline(
        source=TerminalSource(
            [AudioFrameF32(np.ones((1024,), dtype=np.float32), 48000, capture=capture)]
        )
    )

    frames = [frame async for frame in pipeline.frames()]
    assert frames == []
    assert len(pipeline.terminal_discarded_capture) == 1
    discarded = pipeline.terminal_discarded_capture[0]
    assert (discarded.source_start_sample, discarded.source_end_sample) == (0, 1024)


@pytest.mark.asyncio
async def test_desktop_pipeline_preserves_raw_residue_at_known_discontinuity():
    captures = [
        AudioCaptureSpan(
            capture_epoch=1,
            callback_sequence=sequence,
            source_sample_rate_hz=48000,
            source_start_sample=start,
            source_end_sample=end,
            source_start_monotonic_s=start / 48000,
            source_end_monotonic_s=end / 48000,
            discontinuity_before=discontinuity,
        )
        for sequence, start, end, discontinuity in (
            (0, 0, 1024, None),
            (1, 1024, 2048, None),
            (2, 2048, 2148, None),
            (
                3,
                3072,
                4096,
                AudioCaptureDiscontinuity(
                    kind="known_loss",
                    observed_at_monotonic_s=3072 / 48000,
                    lost_source_samples=924,
                ),
            ),
        )
    ]
    pipeline = DesktopPeerPipeline(
        source=StubDesktopAudioSource(
            [
                AudioFrameF32(
                    np.ones((capture.source_sample_count,), dtype=np.float32),
                    48000,
                    capture=capture,
                )
                for capture in captures
            ]
        )
    )

    frames = [frame async for frame in pipeline.frames()]
    prior_ranges = [
        span
        for frame in frames
        for span in (
            *((frame.capture,) if frame.capture is not None else ()),
            *frame.discarded_capture_before,
        )
        if span.source_start_sample < 2148
    ]
    prior_ranges.sort(key=lambda item: item.source_start_sample)

    assert prior_ranges[0].source_start_sample == 0
    assert prior_ranges[-1].source_end_sample == 2148
    assert all(
        left.source_end_sample == right.source_start_sample
        for left, right in zip(prior_ranges, prior_ranges[1:])
    )
    discontinuity_frame = next(frame for frame in frames if frame.discontinuity_before)
    assert discontinuity_frame.discarded_capture_before
