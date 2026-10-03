from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import time
from collections.abc import AsyncIterator, Callable

import numpy as np

from puripuly_heart.core.audio.format import (
    AudioCaptureSpan,
    AudioFrameF32,
    reshape_audio_samples_f32,
)
from puripuly_heart.core.audio.gate import VrcMicAudioGate
from puripuly_heart.core.audio.listen_delivery import ListenDeliveryController
from puripuly_heart.core.audio.ownership import CaptureStreamInput, PeerAudioSegmentLedger
from puripuly_heart.core.audio.smart_turn import SmartTurnInferenceOwner
from puripuly_heart.core.audio.source import AudioSource
from puripuly_heart.core.audio.streaming_resampler import CaptureMappedStreamingResampler
from puripuly_heart.core.vad.gating import SpeechChunk, SpeechEnd, SpeechStart, VadGating
from puripuly_heart.core.vad.sink import VadEventSink

logger = logging.getLogger(__name__)


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


async def _frames_with_progress(
    source: AudioSource,
    *,
    channel_label: str,
    log_basic: Callable[[str], object] | None,
    monotonic_clock: Callable[[], float],
    no_frame_timeout_s: float,
) -> AsyncIterator[AudioFrameF32]:
    if log_basic is None:
        async for frame in source.frames():
            yield frame
        return

    last_frame_at = monotonic_clock()
    no_frames_reported = False
    finished = asyncio.Event()

    def emit(message: str) -> None:
        with contextlib.suppress(Exception):
            log_basic(message)

    async def monitor() -> None:
        nonlocal no_frames_reported
        while not finished.is_set():
            try:
                await asyncio.wait_for(finished.wait(), timeout=no_frame_timeout_s)
            except asyncio.TimeoutError:
                age_s = max(0.0, monotonic_clock() - last_frame_at)
                if age_s >= no_frame_timeout_s and not no_frames_reported:
                    no_frames_reported = True
                    emit(
                        f"[Capture] progress channel={channel_label} "
                        f"state=no_frames wait_ms={int(age_s * 1000)}"
                    )

    monitor_task = asyncio.create_task(
        monitor(),
        name=f"capture-progress:{channel_label}",
    )
    try:
        async for frame in source.frames():
            last_frame_at = monotonic_clock()
            if no_frames_reported:
                no_frames_reported = False
                emit(f"[Capture] progress channel={channel_label} state=frames_resumed")
            yield frame
    finally:
        finished.set()
        await monitor_task


async def run_audio_vad_loop(
    *,
    source: AudioSource,
    vad: VadGating,
    sink: VadEventSink,
    target_sample_rate_hz: int,
    audio_gate: VrcMicAudioGate | None = None,
    channel_label: str = "self",
    log_basic: Callable[[str], object] | None = None,
    segment_ledger: PeerAudioSegmentLedger | None = None,
    monotonic_clock: Callable[[], float] = time.monotonic,
    smart_turn_owner: SmartTurnInferenceOwner | None = None,
    peer_diagnostic_interval_audio_ms: float = 10_000.0,
    no_frame_timeout_s: float = 10.0,
) -> None:
    chunk_samples = vad.chunk_samples
    buffer = np.empty((0,), dtype=np.float32)
    capture_buffer: list[AudioCaptureSpan] = []
    normalizer: CaptureMappedStreamingResampler | None = None
    source_format: tuple[int, int] | None = None
    synthetic_source_next_sample = 0
    last_capture_epoch: int | None = None
    synthetic_sequence = 0
    delivery_controller: ListenDeliveryController | None = None
    gate_stream_blocked = False
    peer_diagnostics = channel_label.lower() == "peer"
    window_samples = 0
    window_square_sum = 0.0
    window_max_probability: float | None = None
    window_threshold_min: float | None = None
    window_threshold_max: float | None = None
    window_threshold_hits = 0
    window_discarded = 0
    window_max_discarded_chunks = 0
    window_committed = 0
    window_rollovers = 0
    window_generation = getattr(vad, "diagnostic_generation", None)
    last_discarded_count = getattr(vad, "discarded_candidate_count", 0)
    window_settings: tuple[float | None, float | None, int | str, int | str] | None = None

    def _score(value: float | None) -> str:
        return "unknown" if value is None else f"{value:.4f}"

    def _flush_peer_window(reason: str) -> None:
        nonlocal window_samples, window_square_sum, window_max_probability
        nonlocal window_threshold_min, window_threshold_max, window_threshold_hits
        nonlocal window_discarded, window_max_discarded_chunks, window_committed
        nonlocal window_rollovers, window_generation, last_discarded_count
        nonlocal window_settings
        if not peer_diagnostics:
            return
        discarded_count = getattr(vad, "discarded_candidate_count", last_discarded_count)
        if discarded_count > last_discarded_count:
            window_discarded += discarded_count - last_discarded_count
            window_max_discarded_chunks = max(
                window_max_discarded_chunks,
                getattr(vad, "last_discarded_candidate_chunks", 0),
            )
            last_discarded_count = discarded_count
        if window_samples or window_discarded:
            rms = math.sqrt(window_square_sum / window_samples) if window_samples else 0.0
            chunk_ms = chunk_samples * 1000.0 / target_sample_rate_hz
            with contextlib.suppress(Exception):
                logger.info(
                    "[VAD] window channel=peer reason=%s generation=%s observed_audio_ms=%d "
                    "input_rms=%.5f max_probability=%s threshold_min=%s threshold_max=%s "
                    "threshold_hits=%d candidate_discarded=%d candidate_max_run_ms=%.1f "
                    "candidate_pending_chunks=%s committed=%d rollovers=%d "
                    "onset_threshold=%s continuation_threshold=%s "
                    "start_debounce_chunks=%s start_commit_chunks=%s",
                    reason,
                    window_generation if window_generation is not None else "unknown",
                    round(window_samples * 1000.0 / target_sample_rate_hz),
                    rms,
                    _score(window_max_probability),
                    _score(window_threshold_min),
                    _score(window_threshold_max),
                    window_threshold_hits,
                    window_discarded,
                    window_max_discarded_chunks * chunk_ms,
                    getattr(vad, "pending_candidate_chunks", "unknown"),
                    window_committed,
                    window_rollovers,
                    _score(window_settings[0] if window_settings else None),
                    _score(window_settings[1] if window_settings else None),
                    window_settings[2] if window_settings else "unknown",
                    window_settings[3] if window_settings else "unknown",
                )
        window_samples = 0
        window_square_sum = 0.0
        window_max_probability = None
        window_threshold_min = None
        window_threshold_max = None
        window_threshold_hits = 0
        window_discarded = 0
        window_max_discarded_chunks = 0
        window_committed = 0
        window_rollovers = 0
        window_generation = getattr(vad, "diagnostic_generation", None)
        window_settings = None

    async def _emit_owned(owned: object) -> None:
        await sink.handle_owned_vad_event(owned)

    if segment_ledger is not None and bool(getattr(vad, "external_delivery_boundaries", False)):
        delivery_controller = ListenDeliveryController(
            vad=vad,
            ledger=segment_ledger,
            emit=_emit_owned,
            monotonic_clock=monotonic_clock,
            smart_turn_owner=smart_turn_owner,
            activity_log=log_basic,
        )
        owner_task = asyncio.current_task()
        if owner_task is not None:
            owner_task.add_done_callback(lambda _task: delivery_controller.cancel())

    def _log_vad_activity(event: object) -> None:
        if log_basic is None:
            return
        label = "Peer" if channel_label.lower() == "peer" else "Self"
        message: str | None = None
        if isinstance(event, SpeechStart):
            message = (
                f"[{label} · VAD] Speech started."
                if event.genuine_onset
                else f"[{label} · VAD] Segment continued after rollover."
            )
        elif isinstance(event, SpeechEnd):
            suffix = (
                f" Trailing silence {event.trailing_silence_ms} ms."
                if event.trailing_silence_ms > 0
                else ""
            )
            message = f"[{label} · VAD] Speech ended.{suffix}"
        if message is not None:
            with contextlib.suppress(Exception):
                log_basic(message)

    async def _dispatch(event: object) -> None:
        if peer_diagnostics and isinstance(event, SpeechStart):
            with contextlib.suppress(Exception):
                logger.info(
                    "[VAD] committed channel=peer utterance_id=%s kind=%s "
                    "applied_threshold=%s onset_threshold=%s continuation_threshold=%s "
                    "start_commit_chunks=%s",
                    event.utterance_id,
                    "onset" if event.genuine_onset else "rollover",
                    (
                        f"{vad.last_applied_threshold:.4f}"
                        if getattr(vad, "last_applied_threshold", None) is not None
                        else "unknown"
                    ),
                    getattr(vad, "speech_threshold", "unknown"),
                    getattr(vad, "continuation_threshold", "unknown"),
                    getattr(vad, "start_commit_chunks", "unknown"),
                )
        _log_vad_activity(event)
        if (
            segment_ledger is not None
            and isinstance(event, SpeechChunk | SpeechEnd)
            and segment_ledger.is_segment_terminal(event.utterance_id)
        ):
            if delivery_controller is not None:
                await delivery_controller.handle_vad_event(event)
            return
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
        nonlocal gate_stream_blocked
        nonlocal buffer
        nonlocal window_samples, window_square_sum, window_max_probability
        nonlocal window_threshold_min, window_threshold_max, window_threshold_hits
        nonlocal window_discarded, window_max_discarded_chunks, window_committed
        nonlocal window_rollovers, window_generation, last_discarded_count, window_settings
        while buffer.size >= chunk_samples:
            chunk = buffer[:chunk_samples]
            buffer = buffer[chunk_samples:]
            chunk_capture = _capture_prefix(capture_buffer, chunk_samples)
            permitted = True
            if audio_gate is not None:
                gated = audio_gate.process_chunk(chunk)
                permitted = gated is chunk
                chunk = gated
            if peer_diagnostics:
                generation = getattr(vad, "diagnostic_generation", None)
                if generation != window_generation:
                    _flush_peer_window("settings_change")
                window_generation = generation
                if window_settings is None:
                    window_settings = (
                        getattr(vad, "speech_threshold", None),
                        getattr(vad, "continuation_threshold", None),
                        getattr(vad, "start_debounce_chunks", "unknown"),
                        getattr(vad, "start_commit_chunks", "unknown"),
                    )
            async with (
                delivery_controller.capture_frame_lock
                if delivery_controller is not None
                else contextlib.nullcontext()
            ):
                process_owned = getattr(vad, "process_owned_chunk", None)
                events = (
                    process_owned(chunk, chunk_capture)
                    if callable(process_owned)
                    else vad.process_chunk(chunk)
                )
                speech_observed = bool(getattr(vad, "last_observation_was_speech", False))
                if peer_diagnostics:
                    window_samples += chunk.size
                    window_square_sum += float(np.dot(chunk, chunk))
                    probability = getattr(vad, "last_probability", None)
                    threshold = getattr(vad, "last_applied_threshold", None)
                    if probability is not None:
                        window_max_probability = (
                            probability
                            if window_max_probability is None
                            else max(window_max_probability, probability)
                        )
                    if threshold is not None:
                        window_threshold_min = (
                            threshold
                            if window_threshold_min is None
                            else min(window_threshold_min, threshold)
                        )
                        window_threshold_max = (
                            threshold
                            if window_threshold_max is None
                            else max(window_threshold_max, threshold)
                        )
                    window_threshold_hits += speech_observed
                    discarded_count = getattr(
                        vad, "discarded_candidate_count", last_discarded_count
                    )
                    if discarded_count > last_discarded_count:
                        window_discarded += discarded_count - last_discarded_count
                        window_max_discarded_chunks = max(
                            window_max_discarded_chunks,
                            getattr(vad, "last_discarded_candidate_chunks", 0),
                        )
                        last_discarded_count = discarded_count
                    for event in events:
                        if isinstance(event, SpeechStart):
                            window_committed += 1
                            window_rollovers += not event.genuine_onset
                    if getattr(vad, "diagnostic_generation", None) != window_generation:
                        _flush_peer_window("settings_change")
                observe_source_activity = getattr(sink, "observe_source_activity", None)
                if callable(observe_source_activity):
                    await observe_source_activity(
                        speech_observed=speech_observed,
                        observed_at_monotonic_s=monotonic_clock(),
                    )
                handle_stream_input = getattr(sink, "handle_stream_input", None)
                if callable(handle_stream_input):
                    if not permitted:
                        if not gate_stream_blocked:
                            await handle_stream_input(
                                CaptureStreamInput(
                                    chunk=chunk[:0],
                                    capture=(),
                                    boundary_reason="source_discontinuity",
                                )
                            )
                        gate_stream_blocked = True
                    elif chunk_capture:
                        gate_stream_blocked = False
                        real_samples = sum(span.normalized_sample_count for span in chunk_capture)
                        await handle_stream_input(
                            CaptureStreamInput(
                                chunk=chunk[:real_samples],
                                capture=chunk_capture,
                                speech_observed=speech_observed,
                            )
                        )
                for event in events:
                    await _dispatch(event)
            if (
                peer_diagnostics
                and window_samples * 1000.0 / target_sample_rate_hz
                >= peer_diagnostic_interval_audio_ms
            ):
                _flush_peer_window("interval")
            if delivery_controller is not None:
                await delivery_controller.observe_acoustic_chunk(
                    speech_observed=speech_observed,
                    capture=chunk_capture,
                )

    async def _handle_discontinuity(
        discarded_capture: tuple[AudioCaptureSpan, ...] = (),
    ) -> None:
        nonlocal buffer, capture_buffer
        nonlocal gate_stream_blocked
        gate_stream_blocked = False
        handle_stream_input = getattr(sink, "handle_stream_input", None)
        if callable(handle_stream_input):
            await handle_stream_input(
                CaptureStreamInput(
                    chunk=np.empty((0,), dtype=np.float32),
                    capture=(),
                    boundary_reason="source_discontinuity",
                )
            )
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
        _flush_peer_window("discontinuity")
        if delivery_controller is not None:
            delivery_controller.invalidate_context()
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

    try:
        async for frame in _frames_with_progress(
            source,
            channel_label=channel_label,
            log_basic=log_basic,
            monotonic_clock=monotonic_clock,
            no_frame_timeout_s=no_frame_timeout_s,
        ):
            capture = frame.capture
            if capture is None and frame.samples.size:
                capture = _source_capture(frame)
            epoch_changed = (
                capture is not None
                and last_capture_epoch is not None
                and capture.capture_epoch != last_capture_epoch
            )
            if capture is not None:
                last_capture_epoch = capture.capture_epoch
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
            if discontinuity is not None or epoch_changed:
                await _handle_discontinuity(discarded)
            elif discarded and segment_ledger is not None:
                segment_ledger.claim_open_content_for_failure(discarded)

            if normalized.size:
                if normalized_capture is None:
                    raise RuntimeError("normalized audio has no capture mapping")
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

        if buffer.size and not (
            getattr(vad, "in_speech", False) or getattr(vad, "continuation_pending", False)
        ):
            handle_stream_input = getattr(sink, "handle_stream_input", None)
            if callable(handle_stream_input):
                tail_chunk = audio_gate.process_chunk(buffer) if audio_gate is not None else buffer
                if tail_chunk is buffer:
                    await handle_stream_input(
                        CaptureStreamInput(chunk=buffer, capture=tuple(capture_buffer))
                    )
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
        _flush_peer_window("eof")
        if delivery_controller is not None:
            await delivery_controller.close()
        handle_stream_input = getattr(sink, "handle_stream_input", None)
        if callable(handle_stream_input):
            await handle_stream_input(
                CaptureStreamInput(
                    chunk=np.empty((0,), dtype=np.float32),
                    capture=(),
                    boundary_reason="source_eof",
                )
            )
    except asyncio.CancelledError:
        _flush_peer_window("cancel")
        raise
    finally:
        _flush_peer_window("exit")
