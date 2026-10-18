"""Read-only recording of policy inputs for offline simulation.

Records full speaker evidence items and the capture-owned inputs that speaker policy does not
change: recognition request lifecycles, submitted audio ranges and native ASR results.
"""

from __future__ import annotations

import functools
import time

from bench.observe import emit, serial


def span(value) -> list:
    discontinuity = value.discontinuity_before
    return [
        value.capture_epoch,
        value.callback_sequence,
        value.source_sample_rate_hz,
        value.source_start_sample,
        value.source_end_sample,
        round(value.source_start_monotonic_s, 6),
        round(value.source_end_monotonic_s, 6),
        value.normalized_sample_rate_hz,
        value.normalized_start_sample,
        value.normalized_end_sample,
        (
            None
            if discontinuity is None
            else [
                discontinuity.kind,
                discontinuity.lost_source_samples,
                discontinuity.observed_at_monotonic_s,
            ]
        ),
    ]


def evidence_item(item) -> dict:
    frame = item.frame
    return {
        "session": item.capture_session_id,
        "epoch": item.speaker_epoch,
        "ready": item.observation_ready_monotonic_s,
        "processed": item.processed_through_sample,
        "immutable": item.immutable_through_sample,
        "range_state": item.range_state,
        "reason": item.reason,
        "frame": (
            None
            if frame is None
            else [
                frame.start_sample,
                frame.end_sample,
                list(frame.probabilities),
                list(frame.active_slots),
            ]
        ),
        "map": [
            [
                p.compact_start_sample,
                p.compact_end_sample,
                p.kind,
                None if p.capture is None else span(p.capture),
            ]
            for p in item.source_map
        ],
    }


def _identity(identity) -> dict:
    segment = identity.segment
    return {
        "turn": identity.provider_turn_id,
        "segment_id": str(segment.segment_id),
        "segment_order": segment.segment_order,
        "activation_generation": segment.activation_generation,
        "capture_epoch": segment.capture_epoch,
    }


def _wrap_async(cls, name, describe):
    original = getattr(cls, name)

    @functools.wraps(original)
    async def wrapper(self, *args, **kwargs):
        try:
            emit(
                "policy_input",
                session=type(self).__name__,
                owner=id(self),
                call=name,
                **describe(*args, **kwargs),
            )
        except Exception as exc:
            emit("observation_error", where=f"{cls.__name__}.{name}", error=repr(exc))
        return await original(self, *args, **kwargs)

    setattr(cls, name, wrapper)


def _wrap_sync(cls, name, describe):
    original = getattr(cls, name)

    @functools.wraps(original)
    def wrapper(self, *args, **kwargs):
        try:
            emit(
                "policy_input",
                session=type(self).__name__,
                owner=id(self),
                call=name,
                **describe(*args, **kwargs),
            )
        except Exception as exc:
            emit("observation_error", where=f"{cls.__name__}.{name}", error=repr(exc))
        return original(self, *args, **kwargs)

    setattr(cls, name, wrapper)


def _begin(request):
    return {
        "identity": _identity(request.identity),
        "display_group_id": serial(request.display_group_id),
        "isolate_failure": request.isolate_failure,
        "optional_cutoff_at_monotonic_s": request.optional_cutoff_at_monotonic_s,
    }


def _audio(identity, pcm16le, *, payload_sequence, source_ranges, context_only):
    return {
        "identity": _identity(identity),
        "payload_sequence": payload_sequence,
        "samples": len(pcm16le) // 2,
        "context_only": context_only,
        "ranges": [span(s) for s in source_ranges],
    }


def _seal(
    identity, *, sealed_content_ranges=(), seal_reason=None, observed_trailing_silence_ms=None, **_
):
    return {
        "identity": _identity(identity),
        "sealed": [span(s) for s in sealed_content_ranges],
        "reason": seal_reason,
        "trailing_silence_ms": observed_trailing_silence_ms,
    }


def _abort(identity, *, reason):
    return {"identity": _identity(identity), "reason": reason}


def _native(event):
    return {
        "event_type": type(event).__name__,
        "identity": _identity(event.identity),
        "observed_at": time.monotonic(),
        "native": serial(event),
    }


def _stream_audio(pcm16le, *, source_ranges):
    return {"samples": len(pcm16le) // 2, "ranges": [span(s) for s in source_ranges]}


def install() -> None:
    from puripuly_heart.core.speaker import evidence as evidence_module
    from puripuly_heart.core.stt.batch_speaker import CausalBatchSession
    from puripuly_heart.core.stt.streaming_speaker import StreamingSpeakerSession

    view = evidence_module.SpeakerEvidenceView
    original_observe = view.observe

    @functools.wraps(original_observe)
    def observe(self, items):
        items = tuple(items)
        try:
            emit(
                "evidence_items",
                view=id(self),
                generation=self.generation,
                items=[evidence_item(i) for i in items],
            )
        except Exception as exc:
            emit("observation_error", where="SpeakerEvidenceView.observe", error=repr(exc))
        return original_observe(self, items)

    view.observe = observe

    for cls in (CausalBatchSession, StreamingSpeakerSession):
        _wrap_async(cls, "begin_turn", _begin)
        _wrap_async(cls, "send_turn_audio", _audio)
        _wrap_async(cls, "seal_turn", _seal)
        _wrap_async(cls, "abort_turn", _abort)
        _wrap_sync(cls, "_snapshot_native_event", _native)
    _wrap_async(StreamingSpeakerSession, "send_stream_audio", _stream_audio)
    _wrap_async(StreamingSpeakerSession, "begin_stream", lambda stream: {"stream": serial(stream)})
    _wrap_async(StreamingSpeakerSession, "end_stream", lambda *, reason: {"reason": reason})
