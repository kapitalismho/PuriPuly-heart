from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field, replace
from uuid import uuid4

from puripuly_heart.core.audio.format import AudioCaptureSpan
from puripuly_heart.core.speaker.contracts import SpeakerEvidence
from puripuly_heart.core.stt.backend import (
    STTProviderTurnIdentity,
    STTProviderTurnRequest,
    STTProviderTurnTerminal,
    STTProviderTurnUpdate,
    STTSourceAudio,
    STTSourceRelease,
    STTTextContribution,
)
from puripuly_heart.core.stt.scoped_event_buffer import STTProviderEventBuffer
from puripuly_heart.core.stt.speaker_attribution import (
    SpeakerFrame,
    attribute_speaker,
    partition_replay,
    probability_state,
)
from puripuly_heart.domain.models import FinalLanguageRun, FinalSpeakerRun

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _Parent:
    request: STTProviderTurnRequest
    payloads: deque = field(default_factory=deque)
    children: deque = field(default_factory=deque)
    sealed: bool = False
    cancelled: bool = False
    sequence: int = 0
    text: str = ""
    speakers: tuple = ()
    languages: tuple = ()
    contributions: tuple = ()
    audio: tuple = ()
    failures: list[str] = field(default_factory=list)
    submitted_cursor: int = -1
    optional_cuts_disabled: bool = False


@dataclass(slots=True)
class _Slice:
    parent: _Parent
    identity: STTProviderTurnIdentity
    audio: STTSourceAudio
    unique_context: tuple
    replayed_context: tuple
    speaker: FinalSpeakerRun | None = None
    evidence_valid: bool = True
    result: STTProviderTurnTerminal | None = None


class CausalBatchSession:
    allows_sealed_turn_overlap = True
    MAX_PENDING_SLICES = 8
    MAX_FRAMES = 2048

    def __init__(self, inner, *, clock=time.monotonic):
        self.inner = inner
        self.clock = clock
        self._events = STTProviderEventBuffer()
        self._parents = deque()
        self._active = None
        self._slices = {}
        self._frames = []
        self._submitted_ranges = deque(maxlen=self.MAX_FRAMES)
        self._epoch = None
        self._immutable = 0
        self._closed = False
        self._consumer = asyncio.create_task(self._consume(), name="causal-batch-results")
        self._lock = asyncio.Lock()

    async def begin_turn(self, request):
        async with self._lock:
            if self._closed or self._active is not None:
                raise RuntimeError("batch capture request is unavailable")
            snapshot = getattr(self.inner, "snapshot_decode_options", None)
            if callable(snapshot):
                request = replace(request, native_options=snapshot())
            self._active = _Parent(request)
            self._parents.append(self._active)

    async def send_turn_audio(
        self, identity, pcm16le, *, payload_sequence, source_ranges, context_only
    ):
        async with self._lock:
            parent = self._require(identity)
            if payload_sequence != parent.sequence + 1:
                raise ValueError("batch payload sequence must be contiguous")
            parent.sequence = payload_sequence
            if pcm16le:
                parent.payloads.append((pcm16le, source_ranges, context_only))
            await self._cut_available(parent)

    def _require(self, identity):
        parent = self._active
        if parent is None or parent.request.identity != identity:
            raise RuntimeError("unknown batch capture request")
        return parent

    async def observe_speaker_evidence(self, evidence):
        async with self._lock:
            if self._closed:
                return
            for item in evidence:
                if not isinstance(item, SpeakerEvidence):
                    continue
                if self._epoch is not None and item.speaker_epoch < self._epoch:
                    continue
                if item.speaker_epoch != self._epoch:
                    if self._epoch is not None:
                        for child in self._slices.values():
                            child.evidence_valid = False
                    self._frames.clear()
                    self._submitted_ranges.clear()
                    self._immutable = 0
                    self._epoch = item.speaker_epoch
                previous_immutable = self._immutable
                self._immutable = max(self._immutable, item.immutable_through_sample)
                if item.reason == "omitted_silence" and item.frame is None:
                    for piece in item.source_map:
                        if (
                            piece.kind == "unobserved"
                            and piece.capture is not None
                            and piece.capture.normalized_start_sample is not None
                            and piece.capture.normalized_end_sample is not None
                        ):
                            self._frames.append(
                                SpeakerFrame(
                                    piece.capture,
                                    item.speaker_epoch,
                                    piece.compact_end_sample,
                                    item.observation_ready_monotonic_s,
                                    True,
                                    (),
                                    False,
                                    False,
                                    False,
                                )
                            )
                    continue
                if item.reason not in (None, "vad_pause", "source_eof"):
                    self._frames.clear()
                    self._submitted_ranges.clear()
                    for child in self._slices.values():
                        child.evidence_valid = False
                    continue
                if item.frame is None:
                    continue
                state = probability_state(item.frame.probabilities)
                if state is None:
                    continue
                slots, clear, uncertain, competing = state
                for piece in item.source_map:
                    span = piece.capture
                    if span is None or piece.kind != "original":
                        continue
                    start, end = span.normalized_start_sample, span.normalized_end_sample
                    if start is None or end is None:
                        continue
                    self._frames = [
                        frame
                        for frame in self._frames
                        if not (
                            frame.span.capture_epoch == span.capture_epoch
                            and frame.span.normalized_start_sample < end
                            and start < frame.span.normalized_end_sample
                            and not frame.closed
                            and frame.compact_end > previous_immutable
                        )
                    ]
                    if any(
                        frame.span.capture_epoch == span.capture_epoch
                        and (frame.closed or frame.compact_end <= previous_immutable)
                        and frame.span.normalized_start_sample < end
                        and start < frame.span.normalized_end_sample
                        for frame in self._frames
                    ):
                        continue
                    self._frames.append(
                        SpeakerFrame(
                            span,
                            item.speaker_epoch,
                            piece.compact_end_sample,
                            item.observation_ready_monotonic_s,
                            item.range_state == "closed",
                            slots,
                            clear,
                            uncertain,
                            competing,
                        )
                    )
            self._frames.sort(
                key=lambda frame: (frame.span.capture_epoch, frame.span.normalized_start_sample)
            )
            self._prune_frames()
            if self._active is not None:
                await self._cut_available(self._active)

    def _prune_frames(self):
        needed = [
            span
            for child in self._slices.values()
            for span in (*child.audio.source_ranges, *child.audio.context_ranges)
        ]
        if self._active is not None:
            needed.extend(span for _, ranges, _ in self._active.payloads for span in ranges)
        floors = {}
        for span in needed:
            if span.normalized_start_sample is not None:
                floors[span.capture_epoch] = min(
                    floors.get(span.capture_epoch, span.normalized_start_sample),
                    span.normalized_start_sample,
                )
        newest = self._frames[-1].span if self._frames else None
        if newest is not None:
            floors[newest.capture_epoch] = min(
                floors.get(newest.capture_epoch, newest.normalized_end_sample),
                newest.normalized_end_sample - 8000,
            )
        self._frames = [
            frame
            for frame in self._frames
            if frame.span.capture_epoch in floors
            and frame.span.normalized_end_sample > floors[frame.span.capture_epoch]
        ][-self.MAX_FRAMES :]

    async def _cut_available(self, parent):
        if parent.optional_cuts_disabled:
            return
        cutoff = parent.request.optional_cutoff_at_monotonic_s
        if cutoff is not None:
            frontier = max(
                (
                    span.source_end_monotonic_s
                    for _, ranges, _ in parent.payloads
                    for span in ranges
                ),
                default=self.clock(),
            )
            observed_at = max(self.clock(), frontier)
            if observed_at >= cutoff:
                parent.optional_cuts_disabled = True
                logger.info(
                    "[BatchSpeaker] request=%s coalesced=safety optional_cutoff=%s observed_at=%s",
                    parent.request.identity.provider_turn_id,
                    cutoff,
                    observed_at,
                )
                return
        if sum(len(item.children) for item in self._parents) >= self.MAX_PENDING_SLICES:
            parent.optional_cuts_disabled = True
            return
        if any(
            sum(span.normalized_sample_count for span in ranges) * 2 != len(pcm)
            for pcm, ranges, _ in parent.payloads
        ):
            return
        content = [span for _, ranges, context in parent.payloads if not context for span in ranges]
        if not content:
            return
        start = content[0].normalized_start_sample
        end = content[-1].normalized_end_sample
        epoch = content[0].capture_epoch
        if start is None or end is None:
            return
        frames = [
            frame
            for frame in self._frames
            if frame.span.capture_epoch == epoch
            and (frame.closed or frame.compact_end <= self._immutable)
        ]
        last_slot = previous_slot = last_end = boundary = None
        qualified = previous_qualified = False
        for frame in frames:
            if last_end is not None and last_end != frame.span.normalized_start_sample:
                last_slot = previous_slot = None
                qualified = previous_qualified = False
            last_end = frame.span.normalized_end_sample
            if len(frame.slots) > 1 or frame.competing:
                last_slot = previous_slot = None
                qualified = previous_qualified = False
                continue
            if not frame.slots:
                continue
            slot = frame.slots[0]
            if slot != last_slot:
                previous_slot, previous_qualified = last_slot, qualified
                last_slot, boundary, qualified = slot, frame.span.normalized_start_sample, False
            qualified |= frame.clear
            if (
                qualified
                and previous_qualified
                and previous_slot != slot
                and start < boundary < end
                and boundary > parent.submitted_cursor
            ):
                await self._submit(parent, boundary, "speaker", frame.ready_at)
                await self._cut_available(parent)
                return

    def _partition(self, parent, boundary):
        selected, remaining = [], deque()
        for pcm, ranges, context in parent.payloads:
            if boundary is None or context:
                selected.append((pcm, ranges, context))
                continue
            if sum(span.normalized_sample_count for span in ranges) * 2 != len(pcm):
                raise ValueError("batch speaker selection requires exact audio ownership")
            offset = 0
            for span in ranges:
                start, end = span.normalized_start_sample, span.normalized_end_sample
                split = min(end, max(start, boundary))
                count = span.normalized_sample_count * 2
                data = pcm[offset : offset + count]
                left = (split - start) * 2
                if left:
                    selected.append((data[:left], (span.slice_normalized(start, split),), False))
                if left < count:
                    remaining.append((data[left:], (span.slice_normalized(split, end),), False))
                offset += count
        parent.payloads = remaining
        return selected

    def _attribute(self, source, context, replayed=()):
        return attribute_speaker(
            (*source, *context), self._frames, self._immutable, replayed=replayed
        )

    async def _submit(self, parent, boundary, reason, ready_at=None):
        selected = self._partition(parent, boundary)
        if not selected:
            return
        source = tuple(span for _, ranges, context in selected if not context for span in ranges)
        context = tuple(span for _, ranges, context in selected if context for span in ranges)
        child_id = replace(parent.request.identity, provider_turn_id=uuid4().hex)
        unique_context, replayed_context = partition_replay(context, self._submitted_ranges)
        native_backend = parent.request.native_options
        if not hasattr(native_backend, "provider_id"):
            native_backend = getattr(self.inner, "backend", None)
        audio = STTSourceAudio(
            child_id.provider_turn_id,
            source,
            context,
            parent.request.display_group_id,
        )
        child = _Slice(parent, child_id, audio, unique_context, replayed_context)
        parent.children.append(child)
        self._slices[child_id] = child
        await self.inner.begin_turn(
            replace(
                parent.request,
                identity=child_id,
                isolate_failure=True,
            )
        )
        sequence = 0
        for data, ranges, is_context in selected:
            sequence += 1
            await self.inner.send_turn_audio(
                child_id,
                data,
                payload_sequence=sequence,
                source_ranges=ranges,
                context_only=is_context,
            )
        await self.inner.seal_turn(
            child_id,
            sealed_content_ranges=source,
            seal_reason=reason,
            observed_trailing_silence_ms=None,
        )
        if source:
            parent.submitted_cursor = source[-1].normalized_end_sample
        for span in (*context, *source):
            if span.normalized_start_sample is not None and span.normalized_end_sample is not None:
                self._submitted_ranges.append(span)
        logger.info(
            "[BatchSpeaker] request=%s slice=%s reason=%s source_boundary=%s "
            "evidence_ready=%s submitted_at=%s submitted_cursor=%s context_samples=%s "
            "capture_epoch=%s group=%s model=%s backend=%s",
            parent.request.identity.provider_turn_id,
            child_id.provider_turn_id,
            reason,
            boundary if boundary is not None else parent.submitted_cursor,
            ready_at,
            self.clock(),
            parent.submitted_cursor,
            sum(span.normalized_sample_count for span in context),
            parent.request.identity.segment.capture_epoch,
            parent.request.display_group_id,
            getattr(native_backend, "model_id", None),
            getattr(native_backend, "provider", "Vulkan"),
        )

    async def seal_turn(
        self, identity, *, sealed_content_ranges, seal_reason, observed_trailing_silence_ms
    ):
        async with self._lock:
            parent = self._require(identity)
            await self._submit(parent, None, seal_reason)
            parent.sealed = True
            self._active = None
            await self._release()

    async def _consume(self):
        try:
            async for event in self.inner.turn_events():
                if isinstance(event, STTProviderTurnTerminal):
                    async with self._lock:
                        child = self._slices.pop(event.identity, None)
                        if child is not None:
                            child.speaker = (
                                self._attribute(
                                    child.audio.source_ranges,
                                    child.unique_context,
                                    child.replayed_context,
                                )
                                if child.evidence_valid
                                and all(
                                    span.capture_epoch == child.identity.segment.capture_epoch
                                    for span in (*child.audio.source_ranges, *child.audio.context_ranges)
                                )
                                else FinalSpeakerRun(
                                    "", None, "", source="nemotron", attribution_state="unavailable"
                                )
                            )
                            child.audio = replace(
                                child.audio, speaker_attribution=child.speaker.attribution
                            )
                            child.result = event
                            self._prune_frames()
                            await self._release()
        finally:
            self._events.close()

    async def _release(self):
        while self._parents:
            parent = self._parents[0]
            while parent.children and parent.children[0].result is not None:
                child = parent.children.popleft()
                result = child.result
                if parent.cancelled:
                    continue
                if result.outcome not in ("final", "empty", "suppressed"):
                    parent.failures.append(result.failure_reason or result.outcome)
                if not result.text:
                    continue
                text = result.text
                languages = result.final_language_runs or (FinalLanguageRun(text, ""),)
                parent.languages += languages
                start = len(parent.text)
                parent.text += text
                sequence = len(parent.contributions) + 1
                contribution = STTTextContribution(
                    f"{parent.request.identity.provider_turn_id}:{sequence}",
                    start,
                    len(parent.text),
                )
                parent.contributions += (contribution,)
                parent.audio += (
                    replace(child.audio, contribution_id=contribution.contribution_id),
                )
                parent.speakers += (replace(child.speaker, text=text),)
                if parent.sealed and not parent.children:
                    continue
                self._events.put(
                    STTProviderTurnUpdate(
                        parent.request.identity,
                        sequence,
                        "stable",
                        "append",
                        text,
                        final_language_runs=languages,
                        final_speaker_runs=(replace(child.speaker, text=text),),
                        source_release=STTSourceRelease(
                            contribution.contribution_id, start, len(parent.text), "boundary"
                        ),
                        source_audio=parent.audio,
                    )
                )
            if not parent.sealed or parent.children:
                return
            self._parents.popleft()
            outcome = (
                "cancelled"
                if parent.cancelled
                else (
                    "degraded"
                    if parent.failures and parent.text
                    else "failed"
                    if parent.failures
                    else "final"
                    if parent.text
                    else "empty"
                )
            )
            self._events.put(
                STTProviderTurnTerminal(
                    parent.request.identity,
                    outcome,
                    parent.text,
                    final_speaker_runs=parent.speakers,
                    final_language_runs=parent.languages,
                    text_authority="authoritative" if outcome in ("final", "empty") else "none",
                    failure_reason=parent.failures[0] if parent.failures else None,
                    included_contributions=parent.contributions,
                    source_audio=parent.audio,
                )
            )

    async def abort_turn(self, identity, *, reason):
        async with self._lock:
            parent = next(
                (item for item in self._parents if item.request.identity == identity), None
            )
            if parent is None:
                return
            parent.payloads.clear()
            parent.cancelled = parent.sealed = True
            parent.failures.append(reason)
            for child in parent.children:
                self._slices.pop(child.identity, None)
            parent.children.clear()
            self._prune_frames()
            if self._active is parent:
                self._active = None
            await self._release()

    async def turn_events(self):
        async for event in self._events.events():
            yield event

    async def stop(self):
        await self.inner.stop()
        await self.close()

    async def close(self):
        if self._closed:
            return
        self._closed = True
        await self.inner.close()
        self._consumer.cancel()
        await asyncio.gather(self._consumer, return_exceptions=True)
        self._parents.clear()
        self._slices.clear()
        self._frames.clear()
        self._submitted_ranges.clear()
        self._events.close()
