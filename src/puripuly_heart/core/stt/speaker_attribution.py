from __future__ import annotations

import math
from dataclasses import dataclass

from puripuly_heart.core.audio.format import AudioCaptureSpan
from puripuly_heart.domain.models import FinalSpeakerRun

ACTIVE_PROBABILITY = 0.5
CLEAR_PROBABILITY = 0.8
QUIET_PROBABILITY = 0.2
COMPETING_PROBABILITY = 0.35
MAX_AMBIGUOUS_INTERVAL_S = 0.120
MAX_AMBIGUOUS_TOTAL_S = 0.200
MAX_AMBIGUOUS_SUPPORT_RATIO = 0.20


@dataclass(frozen=True, slots=True)
class SpeakerFrame:
    span: AudioCaptureSpan
    epoch: int
    compact_end: int
    ready_at: float
    closed: bool
    slots: tuple[int, ...]
    clear: bool
    uncertain: bool
    competing: bool
    probabilities: tuple[float, ...] = ()
    native_frame: tuple[int, int] | None = None


def probability_state(values):
    if len(values) != 8 or not all(math.isfinite(value) and 0 <= value <= 1 for value in values):
        return None
    ranked = sorted(values, reverse=True)
    slots = tuple(index for index, value in enumerate(values) if value >= ACTIVE_PROBABILITY)
    clear = len(slots) == 1 and ranked[0] >= CLEAR_PROBABILITY and ranked[1] <= QUIET_PROBABILITY
    competing = ranked[1] >= COMPETING_PROBABILITY
    uncertain = not clear and ranked[0] >= QUIET_PROBABILITY
    return slots, clear, uncertain, competing


def leading_slot(frame):
    if not frame.probabilities or frame.competing:
        return None
    slot = max(range(len(frame.probabilities)), key=frame.probabilities.__getitem__)
    return slot if frame.probabilities[slot] >= COMPETING_PROBABILITY else None


def unique_ranges(ranges):
    ordered = sorted(
        ranges,
        key=lambda span: (span.capture_epoch, span.normalized_start_sample or 0),
    )
    result = []
    for span in ordered:
        start, end = span.normalized_start_sample, span.normalized_end_sample
        if start is None or end is None:
            result.append(span)
            continue
        if result and result[-1].capture_epoch == span.capture_epoch:
            previous_end = result[-1].normalized_end_sample
            if previous_end is not None and start < previous_end:
                if end <= previous_end:
                    continue
                span = span.slice_normalized(previous_end, end)
        result.append(span)
    return tuple(result)


def partition_replay(context, submitted):
    unique, replayed = [], []
    submitted = unique_ranges(submitted)
    for span in context:
        cursor, end = span.normalized_start_sample, span.normalized_end_sample
        if cursor is None or end is None:
            unique.append(span)
            continue
        for previous in submitted:
            if previous.capture_epoch != span.capture_epoch:
                continue
            left = max(cursor, previous.normalized_start_sample)
            right = min(end, previous.normalized_end_sample)
            if left >= right:
                continue
            if cursor < left:
                unique.append(span.slice_normalized(cursor, left))
            replayed.append(span.slice_normalized(left, right))
            cursor = right
        if cursor < end:
            unique.append(span.slice_normalized(cursor, end))
    return tuple(unique), tuple(replayed)


def attribute_speaker(ranges, frames, immutable, *, replayed=()):
    primary = unique_ranges(ranges)
    replayed = unique_ranges(replayed)
    stable = [frame for frame in frames if frame.closed or frame.compact_end <= immutable]
    epochs, clear_slots, active_slots = set(), set(), set()
    pieces = []
    missing = not primary
    overlap = False
    captures = {span.capture_epoch for span in (*primary, *replayed)}
    broken = len(captures) > 1 or any(
        span.discontinuity_before is not None for span in (*ranges, *replayed)
    )
    for is_replay, spans in ((False, primary), (True, replayed)):
        for span in spans:
            cursor, end = span.normalized_start_sample, span.normalized_end_sample
            broken |= span.discontinuity_before is not None
            if cursor is None or end is None or span.normalized_sample_rate_hz is None:
                missing = True
                continue
            for frame in stable:
                if frame.span.capture_epoch != span.capture_epoch:
                    continue
                left = max(cursor, frame.span.normalized_start_sample)
                right = min(end, frame.span.normalized_end_sample)
                if left >= right:
                    continue
                missing |= left != cursor
                if not is_replay:
                    active_slots.update(frame.slots)
                    if frame.clear:
                        clear_slots.update(frame.slots)
                epochs.add(frame.epoch)
                broken |= frame.span.discontinuity_before is not None
                overlap |= len(frame.slots) > 1 or frame.competing
                pieces.append(
                    (
                        span.capture_epoch,
                        left,
                        right,
                        span.normalized_sample_rate_hz,
                        frame,
                        is_replay,
                    )
                )
                cursor = right
            missing |= cursor != end
    state = "mixed" if overlap or len(clear_slots) > 1 or len(epochs) > 1 else None
    candidate = next(iter(clear_slots)) if len(clear_slots) == 1 else None
    support = 0.0
    ambiguity = []
    foreign_replay = False
    foreign_uncertainty = False
    if candidate is not None:
        for capture, left, right, rate, frame, is_replay in pieces:
            foreign = frame.clear and frame.slots != (candidate,)
            if not is_replay and frame.uncertain:
                leader = leading_slot(frame)
                foreign_uncertainty |= leader is not None and leader != candidate
            foreign_replay |= is_replay and foreign
            if not is_replay and frame.clear:
                support += (right - left) / rate
            elif (is_replay and foreign) or (not is_replay and frame.uncertain):
                ambiguity.append((capture, left, right, rate))
    total = sum((right - left) / rate for _, left, right, rate in ambiguity)
    longest = run = 0.0
    previous = None
    for capture, left, right, rate in sorted(ambiguity):
        run = (
            run + (right - left) / rate
            if previous == (capture, left, rate)
            else (right - left) / rate
        )
        longest = max(longest, run)
        previous = (capture, right, rate)
    exceeds = (
        longest > MAX_AMBIGUOUS_INTERVAL_S + 1e-9
        or total > MAX_AMBIGUOUS_TOTAL_S + 1e-9
        or total > support * MAX_AMBIGUOUS_SUPPORT_RATIO + 1e-9
    )
    if state is None:
        state = (
            "unavailable"
            if missing or broken
            else (
                "mixed"
                if foreign_replay and exceeds
                else (
                    "uncertain"
                    if foreign_uncertainty or (candidate is None and active_slots) or exceeds
                    else "unavailable" if candidate is None else "identified"
                )
            )
        )
    return FinalSpeakerRun(
        "",
        str(candidate) if state == "identified" else None,
        str(next(iter(epochs))) if len(epochs) == 1 else "",
        source="nemotron",
        attribution_state=state,
    )
