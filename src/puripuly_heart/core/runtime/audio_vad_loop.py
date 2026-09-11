from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Callable

import numpy as np

from puripuly_heart.core.audio.diagnostics import compute_audio_frame_metrics
from puripuly_heart.core.audio.format import (
    AudioCaptureSpan,
    AudioFrameF32,
    reshape_audio_samples_f32,
)
from puripuly_heart.core.audio.gate import VrcMicAudioGate
from puripuly_heart.core.audio.listen_delivery import ListenOffDeliveryController
from puripuly_heart.core.audio.ownership import PeerAudioSegmentLedger
from puripuly_heart.core.audio.source import AudioSource
from puripuly_heart.core.audio.streaming_resampler import CaptureMappedStreamingResampler
from puripuly_heart.core.vad.gating import VadGating
from puripuly_heart.core.vad.sink import VadEventSink


def _capture_prefix(
    ranges: list[AudioCaptureSpan],
    sample_count: int,
) -> tuple[AudioCaptureSpan, ...]:
    remaining = sample_count
    selected: list[AudioCaptureSpan] = []
    while remaining > 0 and ranges:
        item = ranges[0]
        item_count = item.normalized_sample_count
        if item_count <= remaining:
            selected.append(item)
            ranges.pop(0)
            remaining -= item_count
            continue
        start = item.normalized_start_sample
        end = item.normalized_end_sample
        if start is None or end is None:
            raise ValueError("normalized capture range is missing")
        split = start + remaining
        selected.append(item.slice_normalized(start, split))
        ranges[0] = item.slice_normalized(split, end)
        remaining = 0
    return tuple(selected)


def _terminal_reason(source: object) -> str | None:
    current = source
    for _ in range(4):
        reason = getattr(current, "terminal_reason", None)
        if isinstance(reason, str):
            return reason
        current = getattr(current, "source", None)
        if current is None:
            return None
    return None


def _terminal_discarded_capture(source: object) -> tuple[AudioCaptureSpan, ...]:
    current = source
    for _ in range(4):
        discarded = getattr(current, "terminal_discarded_capture", None)
        if isinstance(discarded, tuple):
            return discarded
        current = getattr(current, "source", None)
        if current is None:
            return ()
    return ()


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
    segment_ledger: PeerAudioSegmentLedger | None = None,
    monotonic_clock: Callable[[], float] = time.monotonic,
) -> None:
    chunk_samples = vad.chunk_samples
    buffer = np.empty((0,), dtype=np.float32)
    capture_buffer: list[AudioCaptureSpan] = []
    normalizer: CaptureMappedStreamingResampler | None = None
    source_format: tuple[int, int] | None = None
    synthetic_source_next_sample = 0
    synthetic_sequence = 0
    gate_gated_audio_ms = 0.0
    gate_passed_audio_ms = 0.0
    gate_log_accumulated_ms = 0.0
    vad_input_accumulated_audio_ms = 0.0
    delivery_controller: ListenOffDeliveryController | None = None

    async def _emit_owned(owned: object) -> None:
        await sink.handle_owned_vad_event(owned)

    if segment_ledger is not None and bool(getattr(vad, "external_delivery_boundaries", False)):
        delivery_controller = ListenOffDeliveryController(
            vad=vad,
            ledger=segment_ledger,
            emit=_emit_owned,
            monotonic_clock=monotonic_clock,
        )
        owner_task = asyncio.current_task()
        if owner_task is not None:
            owner_task.add_done_callback(lambda _task: delivery_controller.cancel())

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

    async def _dispatch(event: object) -> None:
        if segment_ledger is None:
            await sink.handle_vad_event(event)
            return
        if delivery_controller is not None:
            await delivery_controller.handle_vad_event(event)
            return
        owned = segment_ledger.observe_vad_event(
            event,
            now_monotonic_s=monotonic_clock(),
        )
        await _emit_owned(owned)

    async def _process_buffered_chunks() -> None:
        nonlocal buffer, gate_gated_audio_ms, gate_passed_audio_ms, gate_log_accumulated_ms
        while buffer.size >= chunk_samples:
            chunk = buffer[:chunk_samples]
            buffer = buffer[chunk_samples:]
            chunk_capture = _capture_prefix(capture_buffer, chunk_samples)
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
            process_owned = getattr(vad, "process_owned_chunk", None)
            events = (
                process_owned(chunk, chunk_capture)
                if callable(process_owned)
                else vad.process_chunk(chunk)
            )
            for event in events:
                await _dispatch(event)
            if delivery_controller is not None:
                await delivery_controller.observe_acoustic_chunk(
                    speech_observed=bool(getattr(vad, "last_observation_was_speech", False)),
                    capture=chunk_capture,
                )

    async def _handle_discontinuity(
        discarded_capture: tuple[AudioCaptureSpan, ...] = (),
    ) -> None:
        nonlocal buffer, capture_buffer
        segment_id = segment_ledger.current_open_segment_id if segment_ledger is not None else None
        if segment_ledger is not None:
            segment_ledger.claim_open_content_for_failure((*capture_buffer, *discarded_capture))
        buffer = np.empty((0,), dtype=np.float32)
        capture_buffer = []
        seal_active = getattr(vad, "seal_active", None)
        sealed = seal_active(reason="source_discontinuity") if callable(seal_active) else None
        if sealed is not None:
            await _dispatch(sealed)
        elif hasattr(vad, "reset"):
            vad.reset()
        if segment_ledger is not None and segment_id is not None:
            segment_ledger.terminalize(
                segment_id,
                outcome="failed",
                now_monotonic_s=monotonic_clock(),
            )

    def _source_capture(frame: AudioFrameF32) -> AudioCaptureSpan:
        nonlocal synthetic_source_next_sample, synthetic_sequence
        capture = frame.capture
        if capture is not None:
            return capture
        reshaped = reshape_audio_samples_f32(frame.samples, channels=frame.channels)
        sample_count = int(reshaped.shape[0])
        observed_at = monotonic_clock()
        start = synthetic_source_next_sample
        synthetic_source_next_sample += sample_count
        sequence = synthetic_sequence
        synthetic_sequence += 1
        return AudioCaptureSpan(
            capture_epoch=0,
            callback_sequence=sequence,
            source_sample_rate_hz=frame.sample_rate_hz,
            source_start_sample=start,
            source_end_sample=start + sample_count,
            source_start_monotonic_s=observed_at - sample_count / frame.sample_rate_hz,
            source_end_monotonic_s=observed_at,
        )

    async for frame in source.frames():
        capture = frame.capture
        if capture is None and frame.samples.size:
            capture = _source_capture(frame)
        frame_format = (frame.sample_rate_hz, frame.channels)
        if source_format is None:
            source_format = frame_format
            normalizer = CaptureMappedStreamingResampler(
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

        assert normalizer is not None
        normalized, normalized_capture, normalizer_discarded = normalizer.process(
            frame.samples,
            capture,
        )
        discontinuity = frame.discontinuity_before or (
            capture.discontinuity_before if capture is not None else None
        )
        discarded = (*frame.discarded_capture_before, *normalizer_discarded)
        if discontinuity is not None:
            await _handle_discontinuity(discarded)
        elif discarded and segment_ledger is not None:
            segment_ledger.claim_open_content_for_failure(discarded)

        if normalized.size:
            if normalized_capture is None:
                raise RuntimeError("normalized audio has no capture mapping")
            if _diagnostics_enabled():
                with contextlib.suppress(Exception):
                    vad_input_frame = AudioFrameF32(
                        samples=normalized.reshape(-1),
                        sample_rate_hz=target_sample_rate_hz,
                        channels=1,
                        capture=normalized_capture,
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
            capture_buffer.append(normalized_capture)
            await _process_buffered_chunks()

    terminal_reason = _terminal_reason(source)
    if normalizer is None:
        return
    orderly = terminal_reason in {None, "closed"}
    tail, tail_capture, discarded = normalizer.finish(orderly=orderly)
    discarded = (*discarded, *_terminal_discarded_capture(source))
    if not orderly:
        await _handle_discontinuity(discarded)
        return
    if discarded and segment_ledger is not None:
        segment_ledger.claim_open_content_for_failure(discarded)
    if tail.size:
        if tail_capture is None:
            raise RuntimeError("resampler tail has no capture mapping")
        buffer = np.concatenate([buffer, tail.reshape(-1)])
        capture_buffer.append(tail_capture)
    await _process_buffered_chunks()

    if buffer.size and (
        getattr(vad, "in_speech", False) or getattr(vad, "continuation_pending", False)
    ):
        real_tail_count = int(buffer.size)
        buffer = np.concatenate(
            [
                buffer,
                np.zeros((chunk_samples - real_tail_count,), dtype=np.float32),
            ]
        )
        await _process_buffered_chunks()

    seal_active = getattr(vad, "seal_active", None)
    sealed = seal_active(reason="source_eof") if callable(seal_active) else None
    if sealed is not None:
        await _dispatch(sealed)
    if delivery_controller is not None:
        await delivery_controller.close()
