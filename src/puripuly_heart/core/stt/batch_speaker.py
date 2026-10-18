from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field, replace
from uuid import uuid4

from puripuly_heart.core.speaker.evidence import SpeakerEvidenceView
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
    partition_replay,
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
    speaker: FinalSpeakerRun | None = None
    evidence_valid: bool = True
    result: STTProviderTurnTerminal | None = None


class CausalBatchSession:
    allows_sealed_turn_overlap = True
    MAX_PENDING_SLICES = 8
    MAX_SUBMITTED_RANGES = 2048

    def __init__(self, inner, *, clock=time.monotonic, evidence_view=None):
        self.inner = inner
        self.clock = clock
        self._events = STTProviderEventBuffer()
        self._parents = deque()
        self._active = None
        self._slices = {}
        self.evidence = evidence_view if evidence_view is not None else SpeakerEvidenceView()
        self._owns_evidence = evidence_view is None
        self._submitted_ranges = deque(maxlen=self.MAX_SUBMITTED_RANGES)
        self._evidence_generation = self.evidence.generation
        self._closed = False
        self._cleanup_complete = False
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
        if self._owns_evidence:
            self.evidence.observe(evidence)
        async with self._lock:
            if self._closed:
                return
            if self._evidence_generation != self.evidence.generation:
                self._evidence_generation = self.evidence.generation
                self._submitted_ranges.clear()
                for child in self._slices.values():
                    child.evidence_valid = False
            if self._active is not None:
                await self._cut_available(self._active)

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
        for boundary, _, successor in self.evidence.transitions():
            if (
                boundary.capture_epoch == epoch
                and start < boundary.normalized_sample < end
                and boundary.normalized_sample > parent.submitted_cursor
            ):
                cut = boundary.normalized_sample
                for frame in reversed(self.evidence.frames):
                    if frame.span.capture_epoch != epoch or frame.span.normalized_end_sample > cut:
                        continue
                    if (
                        frame.span.normalized_end_sample != cut
                        or not (frame.closed or frame.compact_end <= self.evidence.immutable)
                        or frame.competing
                        or frame.slots != successor.slots
                    ):
                        break
                    cut = frame.span.normalized_start_sample
                if cut <= start or cut <= parent.submitted_cursor:
                    cut = boundary.normalized_sample
                await self._submit(
                    parent,
                    cut,
                    "speaker",
                    boundary.observed_at_monotonic_s,
                )
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
            speaker_proof_id=self.evidence.pin(
                (*source, *unique_context), replayed=replayed_context
            ),
        )
        child = _Slice(parent, child_id, audio)
        parent.children.append(child)
        self._slices[child_id] = child
        await self.inner.begin_turn(
            replace(
                parent.request,
                identity=child_id,
                isolate_failure=True,
                on_native_terminal=self._snapshot_native_terminal,
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

    def _snapshot_native_terminal(self, event):
        child = self._slices.get(event.identity)
        if child is None or child.parent.cancelled or child.speaker is not None:
            return
        child.speaker = (
            self.evidence.resolve(child.audio.speaker_proof_id)
            if child.evidence_valid
            and all(
                span.capture_epoch == child.identity.segment.capture_epoch
                for span in (*child.audio.source_ranges, *child.audio.context_ranges)
            )
            else FinalSpeakerRun("", None, "", source="nemotron", attribution_state="unavailable")
        )
        child.audio = replace(child.audio, speaker_attribution=child.speaker.attribution)

    async def _consume(self):
        try:
            async for event in self.inner.turn_events():
                if isinstance(event, STTProviderTurnTerminal):
                    async with self._lock:
                        child = self._slices.pop(event.identity, None)
                        if child is not None:
                            if child.speaker is None:
                                raise RuntimeError(
                                    "native terminal did not snapshot speaker evidence"
                                )
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
                    self.evidence.release(child.audio.speaker_proof_id)
                    continue
                if result.outcome not in ("final", "empty", "suppressed"):
                    parent.failures.append(result.failure_reason or result.outcome)
                if not result.text:
                    self.evidence.release(child.audio.speaker_proof_id)
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
                    else "failed" if parent.failures else "final" if parent.text else "empty"
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
                self.evidence.release(child.audio.speaker_proof_id)
            parent.children.clear()
            if self._active is parent:
                self._active = None
            await self._release()

    async def turn_events(self):
        async for event in self._events.events():
            yield event

    async def _settle_consumer(self):
        if not self._consumer.done():
            self._consumer.cancel()
        await asyncio.gather(self._consumer, return_exceptions=True)
        for parent in self._parents:
            for child in parent.children:
                self.evidence.release(child.audio.speaker_proof_id)
        self._parents.clear()
        self._slices.clear()
        if self._owns_evidence:
            self.evidence.reset()
        self._submitted_ranges.clear()
        self._events.close()

    async def stop(self):
        self._closed = True
        try:
            await self.inner.stop()
        except BaseException:
            await self._settle_consumer()
            raise
        await self.close()

    async def close(self):
        if self._cleanup_complete:
            return
        self._closed = True
        try:
            await self.inner.close()
            await asyncio.gather(self._consumer, return_exceptions=True)
        finally:
            await self._settle_consumer()
        self._cleanup_complete = True
