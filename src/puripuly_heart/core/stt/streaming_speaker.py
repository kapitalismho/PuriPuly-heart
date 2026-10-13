from __future__ import annotations

import asyncio
import math
import time
from collections import deque
from dataclasses import dataclass, field, replace

from puripuly_heart.core.speaker.contracts import SpeakerEvidence
from puripuly_heart.core.stt.backend import (
    STTAudioBoundary,
    STTProviderAudioBoundary,
    STTProviderEpochEnded,
    STTProviderInputTerminal,
    STTProviderTurnTerminal,
    STTProviderTurnUpdate,
    STTRecognitionUnit,
    STTSourceAudio,
    STTSourceRelease,
    STTTextContribution,
)
from puripuly_heart.core.stt.scoped_event_buffer import STTProviderEventBuffer
from puripuly_heart.core.stt.scoped_normalizer import STTNormalizationError, STTScopedTurnNormalizer
from puripuly_heart.core.stt.speaker_attribution import (
    SpeakerFrame,
    attribute_speaker,
    probability_state,
)
from puripuly_heart.domain.models import FinalSpeakerRun


class _Timeline:
    MAX_FRAMES = 2048

    def __init__(self):
        self.frames = []
        self.epoch = None
        self.immutable = 0

    def observe(self, evidence):
        for item in evidence:
            if not isinstance(item, SpeakerEvidence):
                continue
            if self.epoch is not None and item.speaker_epoch < self.epoch:
                continue
            if self.epoch != item.speaker_epoch:
                self.frames.clear()
                self.epoch = item.speaker_epoch
                self.immutable = 0
            previous = self.immutable
            self.immutable = max(self.immutable, item.immutable_through_sample)
            if item.reason == "omitted_silence" and item.frame is None:
                for piece in item.source_map:
                    if (
                        piece.kind == "unobserved"
                        and piece.capture is not None
                        and piece.capture.normalized_start_sample is not None
                        and piece.capture.normalized_end_sample is not None
                    ):
                        self.frames.append(
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
                self.frames.clear()
                continue
            if item.frame is None:
                continue
            state = probability_state(item.frame.probabilities)
            if state is None:
                continue
            slots, clear, uncertain, competing = state
            for piece in item.source_map:
                span = piece.capture
                if piece.kind != "original" or span is None:
                    continue
                start, end = span.normalized_start_sample, span.normalized_end_sample
                if start is None or end is None:
                    continue
                overlapping = [
                    frame
                    for frame in self.frames
                    if frame.span.capture_epoch == span.capture_epoch
                    and frame.span.normalized_start_sample < end
                    and start < frame.span.normalized_end_sample
                ]
                if any(frame.closed or frame.compact_end <= previous for frame in overlapping):
                    continue
                self.frames = [frame for frame in self.frames if frame not in overlapping]
                self.frames.append(
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
        self.frames.sort(
            key=lambda frame: (frame.span.capture_epoch, frame.span.normalized_start_sample)
        )
        self.frames = self.frames[-self.MAX_FRAMES :]

    def stable_frames(self):
        return (
            frame for frame in self.frames if frame.closed or frame.compact_end <= self.immutable
        )

    def attribute(self, ranges):
        return attribute_speaker(ranges, self.frames, self.immutable)

    def boundaries(self):
        return (boundary for boundary, _, _ in self.transitions())

    def transitions(self):
        previous = None
        last_epoch = last_end = None
        for frame in self.stable_frames():
            if (
                last_epoch != frame.span.capture_epoch
                or last_end != frame.span.normalized_start_sample
                or frame.span.discontinuity_before is not None
            ):
                previous = None
            last_epoch, last_end = frame.span.capture_epoch, frame.span.normalized_end_sample
            if frame.competing or len(frame.slots) > 1:
                previous = None
                continue
            if not frame.clear:
                continue
            if previous is not None and previous.slots != frame.slots:
                yield (
                    STTAudioBoundary(
                        f"nemotron:{frame.epoch}:{frame.span.capture_epoch}:{frame.span.normalized_start_sample}",
                        frame.span.capture_epoch,
                        frame.span.normalized_start_sample,
                        frame.ready_at,
                        "speaker",
                        str(frame.epoch),
                        transition_start_sample=previous.span.normalized_end_sample,
                        transition_end_sample=frame.span.normalized_start_sample,
                    ),
                    previous,
                    frame,
                )
            previous = frame


@dataclass(slots=True)
class _Turn:
    request: object
    normalizer: STTScopedTurnNormalizer
    content: list = field(default_factory=list)
    context: list = field(default_factory=list)
    timings: dict = field(default_factory=dict)
    clocks: dict = field(default_factory=dict)
    latest: STTProviderTurnUpdate | None = None
    released: int = 0
    sequence: int = 0
    boundaries: set = field(default_factory=set)
    mapping_available: bool = True
    source_endpoints: set = field(default_factory=set)


class StreamingSpeakerSession:
    MAX_TURNS = 64
    MAX_MAP_PIECES = 4096
    MAX_TIMINGS = 4096

    def __init__(self, inner, *, clock=time.monotonic):
        self.inner = inner
        self.clock = clock
        self.sent_until = 0
        self.audio_boundaries = deque(maxlen=256)
        self._maps = deque(maxlen=self.MAX_MAP_PIECES)
        self._timeline = _Timeline()
        self._turns = {}
        self._events = STTProviderEventBuffer()
        self._closed = False
        self._lock = asyncio.Lock()
        observer = getattr(inner, "set_source_mapping_observer", None)
        if callable(observer):
            observer(self._synthetic_sent)
        self._consumer = asyncio.create_task(self._consume(), name="streaming-speaker-results")

    def __getattr__(self, name):
        return getattr(self.inner, name)

    @property
    def accepts_stream_input(self):
        return bool(getattr(self.inner, "accepts_stream_input", False))

    @property
    def independent_recognition_units(self):
        return bool(getattr(self.inner, "independent_recognition_units", False))

    async def begin_stream(self, stream):
        await self.inner.begin_stream(stream)

    async def send_stream_audio(self, pcm16le, *, source_ranges):
        await self.inner.send_stream_audio(pcm16le, source_ranges=source_ranges)

    async def end_stream(self, *, reason):
        await self.inner.end_stream(reason=reason)

    def recognition_source_covers(self, ranges):
        covers = getattr(self.inner, "recognition_source_covers", None)
        return bool(covers(ranges)) if callable(covers) else False

    def _synthetic_sent(self, sample_count):
        if sample_count < 0:
            raise ValueError("synthetic sent count must be nonnegative")
        self._maps.append((self.sent_until, self.sent_until + sample_count, None, True))
        self.sent_until += sample_count

    async def begin_turn(self, request):
        async with self._lock:
            if len(self._turns) >= self.MAX_TURNS:
                raise RuntimeError("streaming speaker request capacity exceeded")
            await self.inner.begin_turn(request)
            self._turns[request.identity] = _Turn(
                request, STTScopedTurnNormalizer(request.identity)
            )
            self._record_boundaries()

    async def send_turn_audio(
        self, identity, pcm16le, *, payload_sequence, source_ranges, context_only
    ):
        async with self._lock:
            turn = self._turns[identity]
            start = self.sent_until
            await self.inner.send_turn_audio(
                identity,
                pcm16le,
                payload_sequence=payload_sequence,
                source_ranges=source_ranges,
                context_only=context_only,
            )
            count = len(pcm16le) // 2
            exact = sum(span.normalized_sample_count for span in source_ranges) == count
            if self.sent_until != start or not exact or len(pcm16le) % 2:
                turn.mapping_available = False
                self._maps.append((self.sent_until, self.sent_until + count, None, context_only))
            else:
                cursor = self.sent_until
                for span in source_ranges:
                    end = cursor + span.normalized_sample_count
                    self._maps.append((cursor, end, span, context_only))
                    cursor = end
            self.sent_until += count
            target = turn.context if context_only else turn.content
            target.extend(source_ranges)
            if len(turn.content) + len(turn.context) > self.MAX_MAP_PIECES:
                turn.content.clear()
                turn.context.clear()
                turn.mapping_available = False
            self._record_boundaries()
            self._release_available(turn)

    async def seal_turn(self, identity, **kwargs):
        await self.inner.seal_turn(identity, **kwargs)

    async def abort_turn(self, identity, *, reason):
        await self.inner.abort_turn(identity, reason=reason)
        self._turns.pop(identity, None)

    async def observe_speaker_evidence(self, evidence):
        async with self._lock:
            self._timeline.observe(evidence)
            self._record_boundaries()
            for turn in self._turns.values():
                self._release_available(turn)

    def _record_boundaries(self):
        for boundary in self._timeline.boundaries():
            for identity, turn in self._turns.items():
                if (
                    identity.segment.capture_epoch != boundary.capture_epoch
                    or boundary.boundary_id in turn.boundaries
                ):
                    continue
                if not any(
                    span.capture_epoch == boundary.capture_epoch
                    and span.normalized_start_sample is not None
                    and span.normalized_end_sample is not None
                    and span.normalized_start_sample
                    <= boundary.normalized_sample
                    <= span.normalized_end_sample
                    for span in turn.content
                ):
                    continue
                if len(turn.boundaries) >= 256:
                    continue
                turn.boundaries.add(boundary.boundary_id)
                self.audio_boundaries.append((identity, boundary))
                self._events.put(STTProviderAudioBoundary(identity, boundary))

    def _mapped(self, turn, timing):
        return tuple(span for span, _ in self._mapped_pieces(turn, timing))

    def _mapped_pieces(self, turn, timing):
        clock = turn.clocks.get(timing.clock_id)
        if (
            not turn.mapping_available
            or clock is None
            or not clock.qualified
            or clock.origin_sent_sample is None
            or timing.semantics == "token_point"
            or timing.raw_start is None
            or timing.raw_end is None
        ):
            return ()
        if clock.sample_rate_hz != turn.request.settings.target_sample_rate_hz:
            return ()
        scale = clock.sample_rate_hz / (1000 if timing.unit == "ms" else 1)
        raw_start, raw_end = timing.raw_start * scale, timing.raw_end * scale
        if (
            not math.isfinite(raw_start)
            or not math.isfinite(raw_end)
            or raw_start < 0
            or raw_end <= raw_start
        ):
            return ()
        start = clock.origin_sent_sample + math.floor(raw_start)
        end = clock.origin_sent_sample + math.ceil(raw_end)
        cursor, ranges = start, []
        for left, right, span, context_only in self._maps:
            a, b = max(cursor, left), min(end, right)
            if a >= b:
                continue
            if (
                a != cursor
                or span is None
                or span.capture_epoch != turn.request.identity.segment.capture_epoch
            ):
                return ()
            ranges.append(
                (
                    span.slice_normalized(
                        span.normalized_start_sample + a - left,
                        span.normalized_start_sample + b - left,
                    ),
                    context_only,
                )
            )
            cursor = b
        return tuple(ranges) if cursor == end else ()

    def _speaker(self, turn, timing):
        ranges = self._mapped(turn, timing)
        return self._timeline.attribute(ranges)

    def _complete_text_groups(self, turn, event, timings):
        endpoints = {0, len(event.text)}
        endpoints.update(point for point in turn.source_endpoints if point <= len(event.text))
        endpoints.update(
            point
            for timing in timings
            if timing.complete_boundary and timing.semantics == "result_span"
            for point in (timing.text_start, timing.text_end)
        )
        if isinstance(event, STTProviderTurnUpdate) and event.source_release is not None:
            endpoints.update((event.source_release.text_start, event.source_release.text_end))
        fixed_endpoints = endpoints.copy()
        lexical = [
            timing
            for timing in timings
            if timing.semantics != "result_span"
            and event.text[timing.text_start : timing.text_end].strip()
        ]
        mapped = [self._mapped(turn, timing) for timing in lexical]
        speakers = [self._speaker(turn, timing) for timing in lexical]
        final = isinstance(event, STTProviderTurnTerminal) or event.source_release is not None
        for boundary, predecessor, successor in self._timeline.transitions():
            transition_start = (
                boundary.transition_start_sample
                if boundary.transition_start_sample is not None
                else boundary.normalized_sample
            )
            preceding = following = None
            for index, timing in enumerate(lexical):
                ranges = mapped[index]
                if (
                    timing.semantics != "word_interval"
                    or not ranges
                    or speakers[index].attribution_state != "identified"
                    or any(span.capture_epoch != boundary.capture_epoch for span in ranges)
                ):
                    continue
                if (
                    timing.complete_boundary
                    and (speakers[index].session_scope, speakers[index].speaker_id)
                    == (str(predecessor.epoch), str(predecessor.slots[0]))
                    and all(span.normalized_end_sample <= transition_start for span in ranges)
                ):
                    preceding = index
                if (
                    following is None
                    and index > 0
                    and lexical[index - 1].complete_boundary
                    and lexical[index - 1].text_end <= timing.text_start
                    and (speakers[index].session_scope, speakers[index].speaker_id)
                    == (str(successor.epoch), str(successor.slots[0]))
                    and all(
                        span.normalized_start_sample >= boundary.normalized_sample
                        for span in ranges
                    )
                ):
                    following = index
            if preceding is not None and not final:
                timing = lexical[preceding]
                cut = timing.text_end
                if preceding + 1 < len(lexical):
                    next_start = lexical[preceding + 1].text_start
                    if not event.text[cut:next_start].strip():
                        cut = next_start
                elif not event.text[cut:].strip():
                    cut = len(event.text)
                endpoints.add(cut)
            if (
                following is not None
                and (preceding is None or preceding < following)
                and (
                    not final
                    or (
                        preceding is not None
                        and speakers[preceding].attribution.key
                        != speakers[following].attribution.key
                    )
                )
            ):
                cut = lexical[following].text_start
                if final and cut not in fixed_endpoints:
                    lower = max(point for point in fixed_endpoints if point < cut)
                    upper = min(point for point in fixed_endpoints if point > cut)
                    owned = [
                        span
                        for timing, ranges in zip(lexical, mapped)
                        if lower <= timing.text_start and timing.text_end <= upper
                        for span in ranges
                    ]
                    supported = {
                        (frame.epoch, frame.slots)
                        for frame in self._timeline.stable_frames()
                        if frame.clear
                        and any(
                            span.capture_epoch == frame.span.capture_epoch
                            and span.normalized_start_sample < frame.span.normalized_end_sample
                            and frame.span.normalized_start_sample < span.normalized_end_sample
                            for span in owned
                        )
                    }
                    if not all(
                        (frame.epoch, frame.slots) in supported
                        for frame in (predecessor, successor)
                    ):
                        continue
                endpoints.add(cut)
        ordered = sorted(point for point in endpoints if 0 <= point <= len(event.text))
        if not final:
            speakers = [
                self._group_speaker(turn, event.text, start, end, timings)
                for start, end in zip(ordered, ordered[1:])
            ]
            ordered = [
                point
                for index, point in enumerate(ordered)
                if point in fixed_endpoints
                or speakers[index - 1].attribution_state == "identified"
                or speakers[index].attribution_state == "identified"
            ]
        return zip(ordered, ordered[1:])

    def _group_timings(self, text, start, end, timings):
        cursor = start
        covered = []
        for timing in timings:
            if (
                timing.semantics == "result_span"
                or timing.text_start < cursor
                or timing.text_end > end
                or not text[timing.text_start : timing.text_end].strip()
            ):
                continue
            if text[cursor : timing.text_start].strip():
                return ()
            covered.append(timing)
            cursor = timing.text_end
        return tuple(covered) if covered and not text[cursor:end].strip() else ()

    def _group_speaker(self, turn, text, start, end, timings):
        covered = self._group_timings(text, start, end, timings)
        if not covered:
            result = next(
                (
                    item
                    for item in timings
                    if item.semantics == "result_span"
                    and item.text_start == start
                    and item.text_end == end
                ),
                None,
            )
            return (
                self._speaker(turn, result)
                if result is not None
                else FinalSpeakerRun(
                    "", None, "", source="nemotron", attribution_state="unavailable"
                )
            )
        ranges = []
        for timing in covered:
            mapped = self._mapped(turn, timing)
            if not mapped:
                return FinalSpeakerRun(
                    "", None, "", source="nemotron", attribution_state="unavailable"
                )
            ranges.extend(mapped)
        return self._timeline.attribute(tuple(ranges))

    def _annotate(self, turn, event):
        text = event.text
        provenance = (
            (event.provenance,) if isinstance(event, STTProviderTurnUpdate) else event.provenance
        )
        capability = next(
            (item.capabilities for item in reversed(provenance) if item.capabilities is not None),
            None,
        )
        allowed = capability is not None and capability.policy != "unknown"
        timings = (
            sorted(turn.timings.values(), key=lambda item: (item.text_start, item.text_end))
            if allowed and capability.policy == "timed"
            else []
        )
        unavailable = FinalSpeakerRun(
            "", None, "", source="nemotron", attribution_state="unavailable"
        )
        whole = (
            self._timeline.attribute(tuple(turn.content + turn.context))
            if allowed and turn.mapping_available
            else unavailable
        )
        runs = [
            replace(
                self._group_speaker(turn, text, start, end, timings) if timings else whole,
                text=text[start:end],
            )
            for start, end in self._complete_text_groups(turn, event, timings)
        ]
        contributions = event.included_contributions
        if isinstance(event, STTProviderTurnUpdate) and not contributions:
            contributions = turn.normalizer.contributions
        if isinstance(event, STTProviderTurnTerminal):
            complete = []
            cursor = 0
            for contribution in contributions:
                if cursor < contribution.text_start:
                    complete.append(
                        STTTextContribution(
                            f"terminal:{cursor}:{contribution.text_start}",
                            cursor,
                            contribution.text_start,
                        )
                    )
                complete.append(contribution)
                cursor = contribution.text_end
            if cursor < len(text):
                complete.append(
                    STTTextContribution(f"terminal:{cursor}:{len(text)}", cursor, len(text))
                )
            contributions = tuple(complete)
        audio = []
        for contribution in contributions:
            associated = [
                timing
                for timing in timings
                if timing.contribution_id == contribution.contribution_id
            ]
            coverage = next(
                (timing for timing in associated if timing.semantics == "result_span"), None
            )
            if coverage is None:
                intervals = [
                    timing
                    for timing in associated
                    if timing.semantics != "token_point"
                    and timing.raw_start is not None
                    and timing.raw_end is not None
                ]
                if intervals and len({(timing.clock_id, timing.unit) for timing in intervals}) == 1:
                    coverage = replace(
                        intervals[0],
                        raw_start=min(timing.raw_start for timing in intervals),
                        raw_end=max(timing.raw_end for timing in intervals),
                    )
            mapped = self._mapped_pieces(turn, coverage) if coverage is not None else ()
            source = (
                tuple(span for span, is_context in mapped if not is_context)
                if mapped
                else tuple(turn.content)
            )
            context = (
                tuple(span for span, is_context in mapped if is_context)
                if mapped
                else tuple(turn.context)
            )
            attribution = (
                self._timeline.attribute(source + context)
                if allowed and turn.mapping_available
                else unavailable
            )
            audio.append(
                STTSourceAudio(
                    contribution.contribution_id,
                    source,
                    context,
                    turn.request.display_group_id,
                    attribution.attribution,
                )
            )
        return replace(
            event,
            final_speaker_runs=tuple(runs),
            source_audio=tuple(audio),
            included_contributions=contributions,
        )

    def _release_available(self, turn):
        event = turn.latest
        if (
            event is None
            or turn.released >= len(event.text)
            or event.provenance.capabilities is None
            or event.provenance.capabilities.policy != "timed"
        ):
            return
        timings = sorted(turn.timings.values(), key=lambda item: (item.text_start, item.text_end))
        complete = turn.released
        for start, end in self._complete_text_groups(turn, event, timings):
            if end <= complete:
                continue
            if start != complete:
                break
            speaker = self._group_speaker(turn, event.text, start, end, timings)
            if speaker.attribution_state != "identified":
                break
            covered = self._group_timings(event.text, start, end, timings) or tuple(
                timing
                for timing in timings
                if timing.semantics == "result_span"
                and timing.complete_boundary
                and timing.text_start == start
                and timing.text_end == end
            )
            ranges = tuple(span for timing in covered for span in self._mapped(turn, timing))
            if not ranges or not any(
                all(
                    span.capture_epoch == boundary.capture_epoch
                    and span.normalized_end_sample
                    <= (
                        boundary.transition_start_sample
                        if boundary.transition_start_sample is not None
                        else boundary.normalized_sample
                    )
                    for span in ranges
                )
                for boundary in self._timeline.boundaries()
            ):
                break
            complete = end
        if complete <= turn.released:
            return
        turn.sequence += 1
        release = STTSourceRelease(
            f"nemotron:{turn.request.identity.provider_turn_id}:{complete}",
            turn.released,
            complete,
            "boundary",
        )
        turn.released = complete
        if len(turn.source_endpoints) < 256:
            turn.source_endpoints.add(complete)
        outgoing = replace(
            event,
            sequence=turn.sequence,
            provenance=replace(event.provenance, native_event_id=None),
            source_release=release,
        )
        self._events.put(self._annotate(turn, outgoing))

    async def _consume(self):
        try:
            async for event in self.inner.turn_events():
                async with self._lock:
                    if isinstance(event, STTProviderTurnUpdate):
                        turn = self._turns.get(event.identity)
                        if turn is None:
                            continue
                        for clock in event.provenance.audio_clocks:
                            turn.clocks[clock.clock_id] = clock
                        for timing in event.provenance.text_timing:
                            turn.timings[
                                (timing.contribution_id, timing.text_start, timing.text_end)
                            ] = timing
                        if len(turn.timings) > self.MAX_TIMINGS or len(turn.clocks) > 64:
                            turn.timings.clear()
                            turn.clocks.clear()
                            turn.mapping_available = False
                        try:
                            normalized = turn.normalizer.apply_update(
                                replace(event, final_speaker_runs=())
                            )
                        except STTNormalizationError:
                            turn.sequence += 1
                            self._events.put(
                                replace(event, sequence=turn.sequence, final_speaker_runs=())
                            )
                            self._turns.pop(event.identity, None)
                            continue
                        if normalized is None:
                            continue
                        turn.sequence += 1
                        normalized = replace(normalized, sequence=turn.sequence)
                        if normalized.stability == "stable":
                            turn.latest = normalized
                        if normalized.source_release is not None:
                            turn.released = max(turn.released, normalized.source_release.text_end)
                            if len(turn.source_endpoints) < 256:
                                turn.source_endpoints.add(normalized.source_release.text_end)
                        self._events.put(self._annotate(turn, normalized))
                        self._release_available(turn)
                    elif isinstance(event, STTProviderTurnTerminal):
                        turn = self._turns.pop(event.identity, None)
                        if turn is not None:
                            for provenance in event.provenance:
                                for clock in provenance.audio_clocks:
                                    turn.clocks[clock.clock_id] = clock
                                for timing in provenance.text_timing:
                                    turn.timings[
                                        (timing.contribution_id, timing.text_start, timing.text_end)
                                    ] = timing
                            try:
                                event = turn.normalizer.apply_terminal(
                                    replace(event, final_speaker_runs=())
                                )
                            except STTNormalizationError:
                                event = replace(event, final_speaker_runs=())
                            else:
                                event = self._annotate(turn, event)
                        else:
                            event = replace(event, final_speaker_runs=())
                        self._events.put(event)
                    elif isinstance(event, STTRecognitionUnit):
                        self._events.put(replace(event, final_speaker_runs=()))
                    else:
                        self._events.put(event)
                        if (
                            isinstance(event, STTProviderInputTerminal)
                            and self.independent_recognition_units
                        ):
                            self._turns.pop(event.identity, None)
                        if isinstance(event, STTProviderEpochEnded):
                            self._turns.clear()
        finally:
            self._events.close()

    async def turn_events(self):
        async for event in self._events.events():
            yield event

    async def stop(self):
        await self.inner.stop()
        await self._consumer

    async def close(self):
        if self._closed:
            return
        self._closed = True
        await self.inner.close()
        if not self._consumer.done():
            self._consumer.cancel()
        try:
            await self._consumer
        except asyncio.CancelledError:
            pass
        self._turns.clear()
        self._maps.clear()
        self._events.close()
