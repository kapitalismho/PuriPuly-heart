from __future__ import annotations

import asyncio
import math
import time
from collections import deque
from dataclasses import dataclass, field, replace

from puripuly_heart.core.speaker.evidence import SpeakerEvidenceView
from puripuly_heart.core.stt.backend import (
    MAX_SPEAKER_AUDIO_PARTS,
    STTProviderAudioBoundary,
    STTProviderEpochEnded,
    STTProviderInputTerminal,
    STTProviderTurnTerminal,
    STTProviderTurnUpdate,
    STTRecognitionUnit,
    STTSourceAudio,
    STTSourceRelease,
    STTSpeakerAudioPart,
    STTTextContribution,
)
from puripuly_heart.core.stt.scoped_event_buffer import STTProviderEventBuffer
from puripuly_heart.core.stt.scoped_normalizer import STTNormalizationError, STTScopedTurnNormalizer
from puripuly_heart.core.stt.speaker_attribution import (
    attribute_speaker,
    leading_slot,
    partition_replay,
)
from puripuly_heart.domain.models import FinalSpeakerRun


@dataclass(slots=True)
class _Turn:
    request: object
    normalizer: STTScopedTurnNormalizer
    sent_start: int = 0
    content: list = field(default_factory=list)
    context: list = field(default_factory=list)
    timings: dict = field(default_factory=dict)
    timing_frontiers: dict = field(default_factory=dict)
    clocks: dict = field(default_factory=dict)
    latest: STTProviderTurnUpdate | None = None
    released: int = 0
    sequence: int = 0
    boundaries: set = field(default_factory=set)
    mapping_available: bool = True
    source_endpoints: set = field(default_factory=set)
    lexical_endpoints: set = field(default_factory=set)
    native_inputs: dict = field(default_factory=dict)
    proofs: dict = field(default_factory=dict)
    transferred_proofs: set = field(default_factory=set)
    input_proof: str | None = None


class StreamingSpeakerSession:
    MAX_TURNS = 64
    MAX_MAP_PIECES = 4096
    MAX_TIMINGS = 4096

    def __init__(self, inner, *, clock=time.monotonic, evidence_view=None):
        self.inner = inner
        self.clock = clock
        self.sent_until = 0
        self.audio_boundaries = deque(maxlen=256)
        self._maps = deque(maxlen=self.MAX_MAP_PIECES)
        self._timeline = evidence_view if evidence_view is not None else SpeakerEvidenceView()
        self._owns_evidence = evidence_view is None
        self._turns = {}
        self._events = STTProviderEventBuffer()
        self._closed = False
        self._lock = asyncio.Lock()
        self._cleanup_complete = False
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
                request, STTScopedTurnNormalizer(request.identity), sent_start=self.sent_until
            )
            self._record_boundaries()

    async def send_turn_audio(
        self, identity, pcm16le, *, payload_sequence, source_ranges, context_only
    ):
        async with self._lock:
            turn = self._turns[identity]
            start = self.sent_until
            proof = self._timeline.pin(tuple(turn.content + turn.context) + tuple(source_ranges))
            self._timeline.release(turn.input_proof)
            turn.input_proof = proof
            try:
                await self.inner.send_turn_audio(
                    identity,
                    pcm16le,
                    payload_sequence=payload_sequence,
                    source_ranges=source_ranges,
                    context_only=context_only,
                )
            except BaseException:
                self._timeline.release(turn.input_proof)
                turn.input_proof = None
                raise
            count = len(pcm16le) // 2
            exact = sum(span.normalized_sample_count for span in source_ranges) == count
            if context_only and turn.content:
                turn.mapping_available = False
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
        turn = self._turns.pop(identity, None)
        if turn is not None:
            self._release_proofs(turn)

    async def observe_speaker_evidence(self, evidence):
        async with self._lock:
            if self._owns_evidence:
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

    def _remember_native_inputs(self, turn, event):
        for origin in event.recognition_origins:
            native = origin.native
            if (
                origin.source_binding != "local_segment"
                or native is None
                or native.native_task_id is None
                or native.timestamp_basis != "task_relative_estimated"
                or native.input_sample_rate_hz != turn.request.settings.target_sample_rate_hz
            ):
                continue
            coverage = turn.native_inputs.setdefault(native.native_task_id, {})
            for item in native.audio_coverage:
                coverage[
                    (
                        item.input_start_sample,
                        item.input_end_sample,
                        item.capture_epoch,
                        item.normalized_start_sample,
                    )
                ] = item
        if sum(len(items) for items in turn.native_inputs.values()) > self.MAX_MAP_PIECES:
            turn.native_inputs.clear()
            turn.mapping_available = False

    def _mapped_native_pieces(self, turn, timing):
        if (
            not turn.mapping_available
            or timing.semantics == "token_point"
            or timing.raw_start is None
            or timing.raw_end is None
        ):
            return ()
        rate = turn.request.settings.target_sample_rate_hz
        scale = rate / (1000 if timing.unit == "ms" else 1)
        start, end = timing.raw_start * scale, timing.raw_end * scale
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
            return ()
        cursor, stop = math.floor(start), math.ceil(end)
        ranges = []
        for item in sorted(
            turn.native_inputs[timing.clock_id].values(),
            key=lambda item: (item.input_start_sample, item.input_end_sample),
        ):
            left, right = max(cursor, item.input_start_sample), min(stop, item.input_end_sample)
            if left >= right:
                continue
            if (
                left != cursor
                or item.capture_epoch != turn.request.identity.segment.capture_epoch
                or item.normalized_sample_rate_hz != rate
                or item.input_end_sample - item.input_start_sample
                != item.normalized_end_sample - item.normalized_start_sample
                or item.discontinuity_kind is not None
            ):
                return ()
            source_start = item.normalized_start_sample + left - item.input_start_sample
            source_end = item.normalized_start_sample + right - item.input_start_sample
            owned = turn.context if item.context_only else turn.content
            for span in owned:
                a, b = (
                    max(source_start, span.normalized_start_sample),
                    min(source_end, span.normalized_end_sample),
                )
                if span.capture_epoch != item.capture_epoch or a >= b:
                    continue
                if a != source_start:
                    return ()
                ranges.append((span.slice_normalized(a, b), item.context_only))
                source_start = b
            if source_start != source_end:
                return ()
            cursor = right
        return tuple(ranges) if cursor == stop else ()

    def _remember_timings(self, turn, provenance, *, final):
        for clock in provenance.audio_clocks:
            turn.clocks[clock.clock_id] = clock
        for timing in provenance.text_timing:
            key = (timing.contribution_id, timing.text_start, timing.text_end)
            turn.timings[key] = timing
            if final:
                turn.timing_frontiers.setdefault(key, self.sent_until)
        if len(turn.timings) > self.MAX_TIMINGS or len(turn.clocks) > 64:
            turn.timings.clear()
            turn.timing_frontiers.clear()
            turn.clocks.clear()
            turn.mapping_available = False

    def _mapped_pieces(self, turn, timing):
        if timing.clock_id in turn.native_inputs:
            return self._mapped_native_pieces(turn, timing)
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
        frontier = turn.timing_frontiers.get(
            (timing.contribution_id, timing.text_start, timing.text_end), self.sent_until
        )
        if end > frontier:
            return ()
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

    def _proof_ranges(self, turn, source, context):
        if not context:
            return source, ()
        if not self._maps or self._maps[0][0] > 0:
            return None
        previous = tuple(span for _, end, span, _ in self._maps if end <= turn.sent_start)
        if any(span is None for span in previous):
            return None
        unique, replayed = partition_replay(context, previous)
        return source + unique, replayed

    def _attribute_ranges(self, turn, source, context):
        proof = self._proof_ranges(turn, source, context)
        if proof is None:
            return FinalSpeakerRun("", None, "", source="nemotron", attribution_state="unavailable")
        primary, replayed = proof
        return attribute_speaker(
            primary, self._timeline.frames, self._timeline.immutable, replayed=replayed
        )

    def _attribute_mapped(self, turn, mapped):
        return self._attribute_ranges(
            turn,
            tuple(span for span, context in mapped if not context),
            tuple(span for span, context in mapped if context),
        )

    def _speaker(self, turn, timing):
        return self._attribute_mapped(turn, self._mapped_pieces(turn, timing))

    def _interval_enclosure(self, turn, text, start, end, timings):
        intervals = tuple(
            timing
            for timing in timings
            if timing.semantics in ("word_interval", "subword_interval")
            and timing.raw_start is not None
            and timing.raw_end is not None
        )
        if (
            not intervals
            or len({(timing.clock_id, timing.unit) for timing in intervals}) != 1
            or self._group_timings(text, start, end, intervals) != intervals
            or not all(self._mapped_pieces(turn, timing) for timing in intervals)
        ):
            return ()
        coverage = replace(
            intervals[0],
            raw_start=min(timing.raw_start for timing in intervals),
            raw_end=max(timing.raw_end for timing in intervals),
        )
        return self._mapped_pieces(turn, coverage)

    def _word_envelope(self, turn, text, start, end, timings):
        if any(
            timing.semantics == "word_interval"
            and (
                timing.text_start < 0
                or timing.text_end <= timing.text_start
                or timing.text_end > len(text)
            )
            for timing in timings
        ):
            return ()
        words = tuple(
            timing
            for timing in timings
            if timing.semantics == "word_interval"
            and timing.text_start < end
            and start < timing.text_end
        )
        if not words or self._group_timings(text, start, end, words) != words:
            return ()
        if len({(word.clock_id, word.unit) for word in words}) != 1:
            return ()
        previous = None
        for word in words:
            if (
                word.unit not in ("s", "ms")
                or not word.complete_boundary
                or not isinstance(word.raw_start, (int, float))
                or isinstance(word.raw_start, bool)
                or not isinstance(word.raw_end, (int, float))
                or isinstance(word.raw_end, bool)
                or not math.isfinite(word.raw_start)
                or not math.isfinite(word.raw_end)
                or word.raw_start < 0
                or word.raw_end <= word.raw_start
                or previous is not None
                and (word.raw_start < previous.raw_start or word.raw_end < previous.raw_end)
                or not self._mapped_pieces(turn, word)
            ):
                return ()
            previous = word
        mapped = self._mapped_pieces(
            turn, replace(words[0], raw_start=words[0].raw_start, raw_end=words[-1].raw_end)
        )
        if any(
            span.capture_epoch != turn.request.identity.segment.capture_epoch
            or span.normalized_sample_rate_hz != turn.request.settings.target_sample_rate_hz
            or span.discontinuity_before is not None
            for span, _ in mapped
        ):
            return ()
        return mapped

    def _transition_band_observed(self, boundary, predecessor, successor):
        cursor = boundary.transition_start_sample
        end = boundary.transition_end_sample
        if cursor is None or end is None:
            return False
        allowed = {predecessor.slots[0], successor.slots[0]}
        for frame in self._timeline.stable_frames():
            if frame.span.capture_epoch != boundary.capture_epoch:
                continue
            left = max(cursor, frame.span.normalized_start_sample)
            right = min(end, frame.span.normalized_end_sample)
            if left >= right:
                continue
            if (
                left != cursor
                or frame.epoch != predecessor.epoch
                or frame.span.discontinuity_before is not None
                or frame.competing
                or len(frame.slots) > 1
                or leading_slot(frame) not in allowed | {None}
            ):
                return False
            cursor = right
        return cursor == end

    def _word_transition_witness(self, ranges, expected, other):
        witnessed = False
        for frame in self._timeline.stable_frames():
            if not any(
                span.capture_epoch == frame.span.capture_epoch
                and span.normalized_start_sample < frame.span.normalized_end_sample
                and frame.span.normalized_start_sample < span.normalized_end_sample
                for span in ranges
            ):
                continue
            if (
                frame.epoch != expected.epoch
                or frame.competing
                or len(frame.slots) > 1
                or leading_slot(frame) not in {expected.slots[0], other.slots[0], None}
                or frame.clear
                and frame.slots != expected.slots
            ):
                return False
            witnessed |= frame.clear and frame.slots == expected.slots
        return witnessed

    def _uncertainty_word_cut(self, turn, event, lexical, mapped, boundary, predecessor, successor):
        band_start = boundary.transition_start_sample
        band_end = boundary.transition_end_sample
        if band_start is None or band_end is None:
            return None
        junctions = []
        for index in range(1, len(lexical)):
            left, right = lexical[index - 1], lexical[index]
            before, after = mapped[index - 1], mapped[index]
            if (
                left.semantics != "word_interval"
                or right.semantics != "word_interval"
                or not left.complete_boundary
                or not right.complete_boundary
                or left.contribution_id != right.contribution_id
                or left.clock_id != right.clock_id
                or left.unit != right.unit
                or left.text_end > right.text_start
                or event.text[left.text_end : right.text_start].strip()
                or not before
                or not after
                or any(span.capture_epoch != boundary.capture_epoch for span in (*before, *after))
                or not band_start
                <= before[-1].normalized_end_sample
                <= after[0].normalized_start_sample
                <= band_end
            ):
                continue
            junctions.append(index)
        if len(junctions) != 1:
            return None
        index = junctions[0]
        right = lexical[index]
        contribution = next(
            (
                item
                for item in turn.normalizer.contributions
                if item.contribution_id == right.contribution_id
            ),
            None,
        )
        associated = tuple(
            timing for timing in lexical if timing.contribution_id == right.contribution_id
        )
        envelope = (
            self._word_envelope(
                turn, event.text, contribution.text_start, contribution.text_end, associated
            )
            if contribution is not None
            else ()
        )
        if (
            not envelope
            or not any(
                span.capture_epoch == boundary.capture_epoch
                and span.normalized_start_sample <= band_start
                for span, _ in envelope
            )
            or not any(span.normalized_end_sample >= band_end for span, _ in envelope)
            or not self._transition_band_observed(boundary, predecessor, successor)
            or not self._word_transition_witness(mapped[index - 1], predecessor, successor)
            or not self._word_transition_witness(mapped[index], successor, predecessor)
            or right.text_start <= turn.released
        ):
            return None
        return right.text_start

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
        supported_regions = {}
        for boundary, predecessor, successor in self._timeline.transitions():
            transition_start = (
                boundary.transition_start_sample
                if boundary.transition_start_sample is not None
                else boundary.normalized_sample
            )
            successor_onset = (
                boundary.successor_onset_sample
                if boundary.successor_onset_sample is not None
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
                    and all(span.normalized_start_sample >= successor_onset for span in ranges)
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
                if len(turn.lexical_endpoints) < 256:
                    turn.lexical_endpoints.add(cut)
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
                    region = (lower, upper)
                    if region not in supported_regions:
                        owned = [
                            span
                            for timing, ranges in zip(lexical, mapped)
                            if lower <= timing.text_start and timing.text_end <= upper
                            for span in ranges
                        ]
                        supported_regions[region] = {
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
                        (frame.epoch, frame.slots) in supported_regions[region]
                        for frame in (predecessor, successor)
                    ):
                        continue
                endpoints.add(cut)
                if len(turn.lexical_endpoints) < 256:
                    turn.lexical_endpoints.add(cut)
            if final and len(turn.lexical_endpoints) < 256:
                cut = self._uncertainty_word_cut(
                    turn, event, lexical, mapped, boundary, predecessor, successor
                )
                if cut is not None:
                    endpoints.add(cut)
                    turn.lexical_endpoints.add(cut)
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
        if all(word.semantics == "word_interval" and word.complete_boundary for word in covered):
            envelope = self._word_envelope(turn, text, start, end, timings)
            return self._attribute_mapped(turn, envelope)
        return self._attribute_mapped(
            turn, self._interval_enclosure(turn, text, start, end, covered)
        )

    def _proof_for(self, turn, contribution, source, context, *, transferred):
        proof = self._proof_ranges(turn, source, context)
        if proof is None:
            return None
        primary, replayed = proof
        footprint = (contribution.contribution_id, primary, replayed)
        token = turn.proofs.get(footprint)
        if token is None or not self._timeline.has_proof(token):
            if footprint not in turn.proofs and len(turn.proofs) >= self.MAX_MAP_PIECES:
                return None
            turn.transferred_proofs.discard(token)
            token = self._timeline.pin(primary, replayed=replayed)
            turn.proofs[footprint] = token
        if transferred and token is not None:
            turn.transferred_proofs.add(token)
        return token

    def _release_proofs(self, turn):
        self._timeline.release(turn.input_proof)
        turn.input_proof = None
        for token in turn.proofs.values():
            if token not in turn.transferred_proofs:
                self._timeline.release(token)

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
            self._attribute_ranges(turn, tuple(turn.content), tuple(turn.context))
            if allowed and turn.mapping_available
            else unavailable
        )
        groups = tuple(self._complete_text_groups(turn, event, timings))
        runs = [
            replace(
                self._group_speaker(turn, text, start, end, timings) if timings else whole,
                text=text[start:end],
                source_boundary_before=start in turn.lexical_endpoints,
            )
            for start, end in groups
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
            mapped = (
                self._mapped_pieces(turn, coverage)
                if coverage is not None
                else self._interval_enclosure(
                    turn, text, contribution.text_start, contribution.text_end, associated
                )
            )
            words_within_contribution = all(
                contribution.text_start <= timing.text_start
                and timing.text_end <= contribution.text_end
                for timing in associated
                if timing.semantics == "word_interval"
            )
            if not mapped and words_within_contribution:
                mapped = self._word_envelope(
                    turn, text, contribution.text_start, contribution.text_end, associated
                )
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
                self._attribute_ranges(turn, source, context)
                if allowed and turn.mapping_available
                else unavailable
            )
            selections = [
                (max(start, contribution.text_start), min(end, contribution.text_end))
                for start, end in groups
                if start < contribution.text_end and contribution.text_start < end
            ]
            parts = []
            if 1 < len(selections) <= MAX_SPEAKER_AUDIO_PARTS:
                for start, end in selections:
                    selected = (
                        self._word_envelope(turn, text, start, end, associated)
                        if words_within_contribution
                        else ()
                    )
                    part_source = tuple(span for span, is_context in selected if not is_context)
                    part_context = tuple(span for span, is_context in selected if is_context)
                    part_attribution = (
                        self._attribute_ranges(turn, part_source, part_context)
                        if selected
                        else unavailable
                    )
                    parts.append(
                        STTSpeakerAudioPart(
                            start,
                            end,
                            part_source if selected else source,
                            part_context if selected else context,
                            part_attribution.attribution,
                            (
                                self._proof_for(
                                    turn,
                                    contribution,
                                    part_source,
                                    part_context,
                                    transferred=isinstance(event, STTProviderTurnTerminal)
                                    or event.source_release is not None,
                                )
                                if selected
                                else None
                            ),
                        )
                    )
            audio.append(
                STTSourceAudio(
                    contribution.contribution_id,
                    source,
                    context,
                    turn.request.display_group_id,
                    attribution.attribution,
                    speaker_proof_id=(
                        self._proof_for(
                            turn,
                            contribution,
                            source,
                            context,
                            transferred=isinstance(event, STTProviderTurnTerminal)
                            or event.source_release is not None,
                        )
                        if allowed
                        and turn.mapping_available
                        and (mapped or capability.policy == "whole_span")
                        else None
                    ),
                    speaker_parts=tuple(parts),
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
                        self._remember_native_inputs(turn, event)
                        self._remember_timings(
                            turn, event.provenance, final=event.source_release is not None
                        )
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
                            self._release_proofs(turn)
                            continue
                        if normalized is None:
                            continue
                        turn.sequence += 1
                        normalized = replace(normalized, sequence=turn.sequence)
                        if normalized.stability == "stable":
                            turn.latest = normalized
                        if normalized.source_release is not None:
                            if len(turn.source_endpoints) < 256:
                                turn.source_endpoints.add(normalized.source_release.text_end)
                        self._events.put(self._annotate(turn, normalized))
                        if normalized.source_release is not None:
                            turn.released = max(turn.released, normalized.source_release.text_end)
                        self._release_available(turn)
                    elif isinstance(event, STTProviderTurnTerminal):
                        turn = self._turns.pop(event.identity, None)
                        if turn is not None:
                            self._remember_native_inputs(turn, event)
                            for provenance in event.provenance:
                                self._remember_timings(turn, provenance, final=True)
                            try:
                                event = turn.normalizer.apply_terminal(
                                    replace(event, final_speaker_runs=())
                                )
                            except STTNormalizationError:
                                event = replace(event, final_speaker_runs=())
                            else:
                                event = self._annotate(turn, event)
                            self._release_proofs(turn)
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
                            turn = self._turns.pop(event.identity, None)
                            if turn is not None:
                                self._release_proofs(turn)
                        if isinstance(event, STTProviderEpochEnded):
                            for turn in self._turns.values():
                                self._release_proofs(turn)
                            self._turns.clear()
        finally:
            self._events.close()

    async def turn_events(self):
        async for event in self._events.events():
            yield event

    async def _settle_consumer(self):
        if not self._consumer.done():
            self._consumer.cancel()
        await asyncio.gather(self._consumer, return_exceptions=True)
        for turn in self._turns.values():
            self._release_proofs(turn)
        self._turns.clear()
        self._maps.clear()
        if self._owns_evidence:
            self._timeline.reset()
        self._events.close()

    async def stop(self):
        self._closed = True
        try:
            await self.inner.stop()
            await asyncio.gather(self._consumer, return_exceptions=True)
        finally:
            await self._settle_consumer()

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
