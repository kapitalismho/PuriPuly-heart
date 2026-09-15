from __future__ import annotations

import builtins
import importlib.util
import sys

import numpy as np
import pytest

import puripuly_heart.core.audio.streaming_resampler as streaming_resampler
import puripuly_heart.core.runtime.audio_vad_loop as audio_vad_loop
from puripuly_heart.core.audio.desktop_pipeline import DesktopPeerPipeline
from puripuly_heart.core.audio.format import AudioFrameF32
from puripuly_heart.core.audio.streaming_resampler import MonoFirstStreamingResampler


def test_streaming_resampler_module_import_does_not_import_soxr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "soxr":
            raise AssertionError("module import should not import soxr")
        return real_import(name, globals, locals, fromlist, level)

    shadow_name = "puripuly_heart._test_shadow_streaming_resampler"
    shadow_spec = importlib.util.spec_from_file_location(shadow_name, streaming_resampler.__file__)
    assert shadow_spec is not None and shadow_spec.loader is not None

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    shadow_module = importlib.util.module_from_spec(shadow_spec)
    monkeypatch.setitem(sys.modules, shadow_name, shadow_module)
    shadow_spec.loader.exec_module(shadow_module)

    resampler = shadow_module.MonoFirstStreamingResampler(16000, 16000, 2)
    out = resampler.resample_chunk(np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32))
    np.testing.assert_allclose(out, np.array([0.5, 0.5], dtype=np.float32))


def test_resample_chunk_mixes_down_before_streaming_soxr_with_mq_quality(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream_calls: list[tuple[object, ...]] = []

    class FakeResampleStream:
        def __init__(
            self,
            in_rate: int,
            out_rate: int,
            channels: int,
            *,
            dtype: str,
            quality: str,
        ) -> None:
            stream_calls.append(("init", in_rate, out_rate, channels, dtype, quality))

        def resample_chunk(self, samples: np.ndarray, *, last: bool = False) -> np.ndarray:
            stream_calls.append(("chunk", samples.copy(), last, None, None))
            return np.asarray(samples * 2.0, dtype=np.float32)

    class FakeSoxrModule:
        ResampleStream = FakeResampleStream

    monkeypatch.setattr(
        streaming_resampler,
        "_import_soxr",
        lambda: FakeSoxrModule,
        raising=False,
    )

    resampler = MonoFirstStreamingResampler(
        input_sample_rate_hz=48000,
        output_sample_rate_hz=16000,
        input_channels=2,
    )

    output = resampler.resample_chunk(np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float32))

    assert stream_calls[0] == ("init", 48000, 16000, 1, "float32", "MQ")
    np.testing.assert_allclose(stream_calls[1][1], np.array([0.5, 0.5], dtype=np.float32))
    assert stream_calls[1][2] is False
    np.testing.assert_allclose(output, np.array([1.0, 1.0], dtype=np.float32))
    assert output.dtype == np.float32


def test_16khz_noop_path_mixdowns_without_building_soxr_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        streaming_resampler,
        "_import_soxr",
        lambda: pytest.fail("16k no-op path should not import soxr"),
        raising=False,
    )

    resampler = MonoFirstStreamingResampler(
        input_sample_rate_hz=16000,
        output_sample_rate_hz=16000,
        input_channels=2,
    )

    output = resampler.resample_chunk(np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32))

    np.testing.assert_allclose(output, np.array([0.5, 0.5], dtype=np.float32))
    assert output.dtype == np.float32


def test_flush_uses_last_true_and_rejects_future_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[np.ndarray, bool]] = []

    class FakeResampleStream:
        def __init__(
            self,
            in_rate: int,
            out_rate: int,
            channels: int,
            *,
            dtype: str,
            quality: str,
        ) -> None:
            assert (in_rate, out_rate, channels, dtype, quality) == (
                48000,
                16000,
                1,
                "float32",
                "MQ",
            )

        def resample_chunk(self, samples: np.ndarray, *, last: bool = False) -> np.ndarray:
            calls.append((samples.copy(), last))
            if last:
                return np.array([0.25, -0.25], dtype=np.float32)
            return np.empty((0,), dtype=np.float32)

    class FakeSoxrModule:
        ResampleStream = FakeResampleStream

    monkeypatch.setattr(
        streaming_resampler,
        "_import_soxr",
        lambda: FakeSoxrModule,
        raising=False,
    )

    resampler = MonoFirstStreamingResampler(input_sample_rate_hz=48000, output_sample_rate_hz=16000)

    first = resampler.resample_chunk(np.array([0.0, 1.0], dtype=np.float32))
    tail = resampler.flush()

    assert first.size == 0
    assert calls[0][1] is False
    assert calls[1][1] is True
    assert calls[1][0].dtype == np.float32
    assert calls[1][0].size == 0
    np.testing.assert_allclose(tail, np.array([0.25, -0.25], dtype=np.float32))

    with pytest.raises(RuntimeError, match="already been flushed"):
        resampler.resample_chunk(np.array([0.0], dtype=np.float32))


class _StubAudioSource:
    def __init__(self, frames: list[AudioFrameF32]) -> None:
        self._frames = frames
        self.closed = False

    async def frames(self):
        for frame in self._frames:
            yield frame

    async def close(self) -> None:
        self.closed = True


class _StubVad:
    def __init__(self, *, chunk_samples: int) -> None:
        self.chunk_samples = chunk_samples
        self.chunks: list[np.ndarray] = []

    def process_chunk(self, chunk: np.ndarray) -> list[np.ndarray]:
        copied = chunk.copy()
        self.chunks.append(copied)
        return [copied]


class _StubSink:
    def __init__(self) -> None:
        self.events: list[np.ndarray] = []

    async def handle_vad_event(self, event: np.ndarray) -> None:
        self.events.append(event)


@pytest.mark.asyncio
async def test_run_audio_vad_loop_raises_on_source_sample_rate_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeResampler:
        def __init__(
            self,
            input_sample_rate_hz: int,
            output_sample_rate_hz: int = 16000,
            input_channels: int = 1,
        ) -> None:
            assert (input_sample_rate_hz, output_sample_rate_hz, input_channels) == (
                48000,
                16000,
                2,
            )

        def resample_chunk(self, samples: np.ndarray) -> np.ndarray:
            return np.asarray(samples[:0], dtype=np.float32)

        def flush(self) -> np.ndarray:
            return np.empty((0,), dtype=np.float32)

    monkeypatch.setattr(audio_vad_loop, "MonoFirstStreamingResampler", FakeResampler, raising=False)

    source = _StubAudioSource(
        [
            AudioFrameF32(
                samples=np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float32),
                sample_rate_hz=48000,
                channels=2,
            ),
            AudioFrameF32(
                samples=np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32),
                sample_rate_hz=44100,
                channels=2,
            ),
        ]
    )

    with pytest.raises(ValueError, match="source audio format changed"):
        await audio_vad_loop.run_audio_vad_loop(
            source=source,
            vad=_StubVad(chunk_samples=2),
            sink=_StubSink(),
            target_sample_rate_hz=16000,
        )


@pytest.mark.asyncio
async def test_desktop_pipeline_raises_on_source_channel_change() -> None:

    pipeline = DesktopPeerPipeline(
        source=_StubAudioSource(
            [
                AudioFrameF32(
                    samples=np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float32),
                    sample_rate_hz=48000,
                    channels=2,
                ),
                AudioFrameF32(
                    samples=np.array([0.0, 1.0], dtype=np.float32),
                    sample_rate_hz=48000,
                    channels=1,
                ),
            ]
        ),
        target_sample_rate_hz=16000,
    )

    with pytest.raises(ValueError, match="source audio format changed"):
        _ = [frame async for frame in pipeline.frames()]
