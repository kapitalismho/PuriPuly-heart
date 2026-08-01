from __future__ import annotations

import contextlib
import inspect
from collections.abc import Callable

import numpy as np

from puripuly_heart.core.audio.diagnostics import compute_audio_frame_metrics
from puripuly_heart.core.audio.format import AudioFrameF32
from puripuly_heart.core.audio.gate import VrcMicAudioGate
from puripuly_heart.core.audio.source import AudioSource
from puripuly_heart.core.audio.streaming_resampler import MonoFirstStreamingResampler
from puripuly_heart.core.vad.gating import VadGating
from puripuly_heart.core.vad.sink import VadEventSink
from puripuly_heart.core.vad.smart_turn import (
    SmartTurnEventSinkFactory,
    SmartTurnExperimentConfig,
)


async def run_audio_vad_loop(
    *,
    source: AudioSource,
    vad: VadGating,
    sink: VadEventSink,
    target_sample_rate_hz: int,
    audio_gate: VrcMicAudioGate | None = None,
    channel_label: str = "self",
    is_detailed_enabled: Callable[[], bool] | None = None,
    log_detailed: Callable[[str], object] | None = None,
    smart_turn_config: SmartTurnExperimentConfig | None = None,
    vad_event_sink_factory: SmartTurnEventSinkFactory | None = None,
) -> None:
    event_sink: VadEventSink = sink
    if smart_turn_config is not None and vad_event_sink_factory is not None:
        event_sink = vad_event_sink_factory(
            sink=sink,
            vad=vad,
            config=smart_turn_config,
            channel_label=channel_label,
            sample_rate_hz=target_sample_rate_hz,
        )
    try:
        await _run_audio_vad_loop(
            source=source,
            vad=vad,
            sink=event_sink,
            target_sample_rate_hz=target_sample_rate_hz,
            audio_gate=audio_gate,
            channel_label=channel_label,
            is_detailed_enabled=is_detailed_enabled,
            log_detailed=log_detailed,
        )
    finally:
        if event_sink is not sink:
            close = getattr(event_sink, "close", None)
            if callable(close):
                result = close()
                if inspect.isawaitable(result):
                    await result


async def _run_audio_vad_loop(
    *,
    source: AudioSource,
    vad: VadGating,
    sink: VadEventSink,
    target_sample_rate_hz: int,
    audio_gate: VrcMicAudioGate | None = None,
    channel_label: str = "self",
    is_detailed_enabled: Callable[[], bool] | None = None,
    log_detailed: Callable[[str], object] | None = None,
) -> None:
    chunk_samples = vad.chunk_samples
    buffer = np.empty((0,), dtype=np.float32)
    resampler: MonoFirstStreamingResampler | None = None
    source_format: tuple[int, int] | None = None
    gate_gated_audio_ms = 0.0
    gate_passed_audio_ms = 0.0
    gate_log_accumulated_ms = 0.0
    vad_input_accumulated_audio_ms = 0.0

    def _diagnostics_enabled() -> bool:
        if is_detailed_enabled is None or log_detailed is None:
            return False
        with contextlib.suppress(Exception):
            return bool(is_detailed_enabled())
        return False

    def _log_detailed_best_effort(message: str) -> None:
        if log_detailed is None:
            return
        with contextlib.suppress(Exception):
            log_detailed(message)

    async def _process_buffered_chunks() -> None:
        nonlocal buffer, gate_gated_audio_ms, gate_passed_audio_ms, gate_log_accumulated_ms
        while buffer.size >= chunk_samples:
            chunk = buffer[:chunk_samples]
            buffer = buffer[chunk_samples:]
            original_chunk = chunk
            if audio_gate is not None:
                chunk = audio_gate.process_chunk(chunk)
                if _diagnostics_enabled():
                    with contextlib.suppress(Exception):
                        chunk_ms = chunk.size * 1000.0 / float(target_sample_rate_hz)
                        gate_log_accumulated_ms += chunk_ms
                        if np.any(original_chunk) and not np.any(chunk):
                            gate_gated_audio_ms += chunk_ms
                        else:
                            gate_passed_audio_ms += chunk_ms
                        if gate_log_accumulated_ms >= 1000.0:
                            _log_detailed_best_effort(
                                f"[AudioDiag][Gate][{channel_label}] "
                                f"enabled={audio_gate.enabled} "
                                f"receiver_active={audio_gate.receiver_active} "
                                f"gated_audio_ms={gate_gated_audio_ms:.1f} "
                                f"passed_audio_ms={gate_passed_audio_ms:.1f}"
                            )
                            gate_log_accumulated_ms = 0.0
                            gate_gated_audio_ms = 0.0
                            gate_passed_audio_ms = 0.0
            for ev in vad.process_chunk(chunk):
                await sink.handle_vad_event(ev)

    async for frame in source.frames():
        frame_format = (frame.sample_rate_hz, frame.channels)
        if source_format is None:
            source_format = frame_format
            resampler = MonoFirstStreamingResampler(
                input_sample_rate_hz=frame.sample_rate_hz,
                output_sample_rate_hz=target_sample_rate_hz,
                input_channels=frame.channels,
            )
        elif frame_format != source_format:
            raise ValueError(
                "source audio format changed during streaming: "
                f"expected {source_format[0]}Hz/{source_format[1]}ch, "
                f"got {frame.sample_rate_hz}Hz/{frame.channels}ch"
            )

        assert resampler is not None
        normalized = resampler.resample_chunk(frame.samples)
        if normalized.size:
            if _diagnostics_enabled():
                with contextlib.suppress(Exception):
                    vad_input_frame = AudioFrameF32(
                        samples=normalized.reshape(-1),
                        sample_rate_hz=target_sample_rate_hz,
                        channels=1,
                    )
                    vad_input_metrics = compute_audio_frame_metrics(vad_input_frame)
                    vad_input_accumulated_audio_ms += vad_input_metrics.audio_ms
                    if vad_input_accumulated_audio_ms >= 1000.0:
                        vad_input_accumulated_audio_ms = 0.0
                        _log_detailed_best_effort(
                            f"[AudioDiag][VADInput][{channel_label}] "
                            f"source_rate={frame.sample_rate_hz} "
                            f"source_channels={frame.channels} "
                            f"target_rate={target_sample_rate_hz} "
                            f"samples={vad_input_metrics.samples} "
                            f"audio_ms={vad_input_metrics.audio_ms:.1f} "
                            f"rms_db={vad_input_metrics.rms_db:.1f} "
                            f"peak_db={vad_input_metrics.peak_db:.1f} "
                            f"zero_ratio={vad_input_metrics.zero_ratio:.3f}"
                        )
            buffer = np.concatenate([buffer, normalized.reshape(-1)])
            await _process_buffered_chunks()

    if resampler is None:
        return

    tail = resampler.flush()
    if tail.size:
        buffer = np.concatenate([buffer, tail.reshape(-1)])
    await _process_buffered_chunks()
