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
from puripuly_heart.domain.models import FinalLanguageRun, FinalSpeakerRun

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _Frame:
    span: AudioCaptureSpan
    epoch: int
    slots: tuple[int, ...]
    compact_end: int
    ready_at: float
    closed: bool
    clear: bool
    uncertain: bool
    competing: bool


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
    speaker: FinalSpeakerRun
    result: STTProviderTurnTerminal | None = None


class CausalBatchSession:
    allows_sealed_turn_overlap = True
    MAX_PENDING_SLICES = 8
    MAX_FRAMES = 2048
    ACTIVE_PROBABILITY = 0.5
    CLEAR_PROBABILITY = 0.8
    QUIET_PROBABILITY = 0.2
    COMPETING_PROBABILITY = 0.35
    PARAKEET_RIGHT_CONTEXT_SAMPLES = 32000
    _PARAKEET_RIGHT_CONTEXT_PCM = bytes(PARAKEET_RIGHT_CONTEXT_SAMPLES * 2)

    def __init__(self, inner, *, clock=time.monotonic):
        self.inner = inner
        self.clock = clock
        self._events = STTProviderEventBuffer()
        self._parents = deque()
        self._active = None
        self._slices = {}
        self._frames = []
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
                    self._frames.clear()
                    self._immutable = 0
                    self._epoch = item.speaker_epoch
                previous_immutable = self._immutable
                if item.frame is None:
                    self._immutable = max(self._immutable, item.immutable_through_sample)
                if item.reason not in (None, "vad_pause", "source_eof"):
                    self._frames.clear()
                    continue
                if item.frame is None:
                    continue
                for piece in item.source_map:
                    span = piece.capture
                    if span is None or piece.kind != "original":
                        continue
                    start, end = span.normalized_start_sample, span.normalized_end_sample
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
                    probabilities = item.frame.probabilities
                    slots = tuple(
                        index
                        for index, value in enumerate(probabilities)
                        if value >= self.ACTIVE_PROBABILITY
                    )
                    ranked = sorted(probabilities, reverse=True)
                    competing = len(slots) < 2 and ranked[1] >= self.COMPETING_PROBABILITY
                    self._frames.append(
                        _Frame(
                            span,
                            item.speaker_epoch,
                            slots,
                            piece.compact_end_sample,
                            item.observation_ready_monotonic_s,
                            item.range_state == "closed",
                            len(slots) == 1
                            and ranked[0] >= self.CLEAR_PROBABILITY
                            and ranked[1] <= self.QUIET_PROBABILITY,
                            (not slots and ranked[0] >= self.QUIET_PROBABILITY) or competing,
                            competing,
                        )
                    )
            self._frames.sort(
                key=lambda frame: (frame.span.capture_epoch, frame.span.normalized_start_sample)
            )
            self._frames = self._frames[-self.MAX_FRAMES :]
            if self._active is not None:
                if self._active.payloads:
                    ranges = [span for _, spans, _ in self._active.payloads for span in spans]
                    if ranges:
                        floor = (
                            min(
                                (
                                    span.normalized_start_sample
                                    for span in ranges
                                    if span.normalized_start_sample is not None
                                ),
                                default=0,
                            )
                            - 8000
                        )
                        self._frames = [
                            frame
                            for frame in self._frames
                            if frame.span.capture_epoch == ranges[0].capture_epoch
                            and frame.span.normalized_end_sample > floor
                        ]
                await self._cut_available(self._active)

    async def _cut_available(self, parent):
        if parent.optional_cuts_disabled:
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

    def _attribute(self, source, context):
        slots, epochs = set(), set()
        missing, overlap, uncertain = False, False, False
        for span in (*source, *context):
            cursor = span.normalized_start_sample
            end = span.normalized_end_sample
            if cursor is None or end is None:
                missing = True
                continue
            for frame in self._frames:
                if frame.span.capture_epoch != span.capture_epoch or (
                    not frame.closed and frame.compact_end > self._immutable
                ):
                    continue
                left = max(cursor, frame.span.normalized_start_sample)
                right = min(end, frame.span.normalized_end_sample)
                if left >= right:
                    continue
                if left != cursor:
                    missing = True
                slots.update(frame.slots)
                epochs.add(frame.epoch)
                overlap |= len(frame.slots) > 1
                uncertain |= frame.uncertain
                cursor = right
            missing |= cursor != end
        state = (
            "mixed"
            if overlap or len(slots) > 1 or len(epochs) > 1
            else (
                "unavailable"
                if missing or not slots
                else "uncertain"
                if uncertain
                else "identified"
            )
        )
        return FinalSpeakerRun(
            "",
            str(next(iter(slots))) if state == "identified" else None,
            str(next(iter(epochs))) if len(epochs) == 1 else "",
            source="nemotron",
            attribution_state=state,
        )

    async def _submit(self, parent, boundary, reason, ready_at=None):
        selected = self._partition(parent, boundary)
        if not selected:
            return
        source = tuple(span for _, ranges, context in selected if not context for span in ranges)
        context = tuple(span for _, ranges, context in selected if context for span in ranges)
        child_id = replace(parent.request.identity, provider_turn_id=uuid4().hex)
        speaker = self._attribute(source, context)
        native_backend = parent.request.native_options
        if not hasattr(native_backend, "provider_id"):
            native_backend = getattr(self.inner, "backend", None)
        padding = (
            self.PARAKEET_RIGHT_CONTEXT_SAMPLES
            if parent.submitted_cursor >= 0
            and getattr(native_backend, "provider_id", None) == "local_parakeet_v3"
            else 0
        )
        padding_rate = getattr(native_backend, "sample_rate_hz", None) if padding else None
        audio = STTSourceAudio(
            child_id.provider_turn_id,
            source,
            context,
            parent.request.display_group_id,
            speaker.attribution,
            padding,
            padding_rate,
        )
        child = _Slice(parent, child_id, audio, speaker)
        parent.children.append(child)
        self._slices[child_id] = child
        await self.inner.begin_turn(
            replace(
                parent.request,
                identity=child_id,
                isolate_failure=True,
                synthetic_right_context_samples=padding,
                synthetic_context_sample_rate_hz=padding_rate,
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
        if padding:
            await self.inner.send_turn_audio(
                child_id,
                self._PARAKEET_RIGHT_CONTEXT_PCM,
                payload_sequence=sequence + 1,
                source_ranges=(),
                context_only=True,
            )
        await self.inner.seal_turn(
            child_id,
            sealed_content_ranges=source,
            seal_reason=reason,
            observed_trailing_silence_ms=None,
        )
        if source:
            parent.submitted_cursor = source[-1].normalized_end_sample
        logger.info(
            "[BatchSpeaker] request=%s slice=%s reason=%s source_boundary=%s "
            "evidence_ready=%s submitted_at=%s submitted_cursor=%s context_samples=%s "
            "synthetic_right_context_samples=%s attribution=%s capture_epoch=%s "
            "speaker_epoch=%s group=%s model=%s backend=%s",
            parent.request.identity.provider_turn_id,
            child_id.provider_turn_id,
            reason,
            boundary if boundary is not None else parent.submitted_cursor,
            ready_at,
            self.clock(),
            parent.submitted_cursor,
            sum(span.normalized_sample_count for span in context),
            padding,
            child.speaker.attribution_state,
            parent.request.identity.segment.capture_epoch,
            child.speaker.session_scope,
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
                            child.result = event
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
        self._events.close()
