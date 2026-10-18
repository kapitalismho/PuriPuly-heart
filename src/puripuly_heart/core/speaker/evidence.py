from __future__ import annotations

from dataclasses import dataclass, replace
from uuid import uuid4

from puripuly_heart.core.speaker.contracts import SpeakerEvidence
from puripuly_heart.core.stt.backend import STTAudioBoundary
from puripuly_heart.core.stt.speaker_attribution import (
    SpeakerFrame,
    attribute_speaker,
    leading_slot,
    probability_state,
    unique_ranges,
)
from puripuly_heart.domain.models import FinalSpeakerRun


@dataclass(frozen=True, slots=True)
class _Proof:
    ranges: tuple
    replayed: tuple
    generation: int
    stage: str = "request"


class SpeakerEvidenceView:
    MAX_FRAMES = 2048
    MAX_PROOFS = 128
    MAX_RANGE_PIECES = 4096
    CONFIRM_FRAMES = 2

    def __init__(self):
        self.frames = []
        self.epoch = None
        self.capture_session_id = None
        self.immutable = 0
        self.processed = 0
        self.revision = 0
        self.generation = 0
        self._proofs = {}
        self._listeners = set()

    def subscribe(self, listener):
        self._listeners.add(listener)

    def unsubscribe(self, listener):
        self._listeners.discard(listener)

    def _changed(self):
        self.revision += 1
        for listener in tuple(self._listeners):
            listener()

    def reset(self):
        self.frames.clear()
        self._proofs.clear()
        self.immutable = self.processed = 0
        self.generation += 1
        self._changed()

    def pin(self, ranges, *, replayed=()):
        ranges, replayed = unique_ranges(ranges), unique_ranges(replayed)
        count = len(ranges) + len(replayed)
        if (
            not ranges
            or len(self._proofs) >= self.MAX_PROOFS
            or count + sum(len(p.ranges) + len(p.replayed) for p in self._proofs.values())
            > self.MAX_RANGE_PIECES
        ):
            return None
        token = uuid4().hex
        self._proofs[token] = _Proof(ranges, replayed, self.generation)
        return token

    def release(self, token):
        self._proofs.pop(token, None)

    def adopt(self, token):
        proof = self._proofs.get(token)
        if proof is not None and proof.stage != "source":
            self._proofs[token] = replace(proof, stage="source")

    def publish(self, token):
        proof = self._proofs.get(token)
        if proof is not None and proof.stage == "request":
            self._proofs[token] = replace(proof, stage="contribution")

    def release_unadopted(self, token, *, published=False):
        proof = self._proofs.get(token)
        if (
            proof is not None
            and proof.stage != "source"
            and (published or proof.stage == "request")
        ):
            self.release(token)

    def release_claims(self):
        self._proofs.clear()
        self._changed()

    def has_proof(self, token):
        return token in self._proofs

    def resolve(self, token):
        proof = self._proofs.get(token)
        if proof is None or proof.generation != self.generation:
            return FinalSpeakerRun("", None, "", source="nemotron", attribution_state="unavailable")
        return attribute_speaker(proof.ranges, self.frames, self.immutable, replayed=proof.replayed)

    def observe(self, evidence):
        changed = False
        for item in evidence:
            if not isinstance(item, SpeakerEvidence):
                continue
            if self.epoch is not None and item.speaker_epoch < self.epoch:
                continue
            if not 0 <= item.immutable_through_sample <= item.processed_through_sample:
                self.reset()
                continue
            if item.speaker_epoch != self.epoch or (
                self.capture_session_id is not None
                and item.capture_session_id != self.capture_session_id
            ):
                if self.epoch is not None:
                    self.reset()
                self.epoch = item.speaker_epoch
                self.capture_session_id = item.capture_session_id
            if item.reason not in (None, "vad_pause", "source_eof", "omitted_silence"):
                self.reset()
                continue
            previous = self.immutable
            frontier = max(self.immutable, item.immutable_through_sample)
            processed = max(self.processed, item.processed_through_sample)
            changed |= frontier != self.immutable or processed != self.processed
            self.immutable, self.processed = frontier, processed
            if item.frame is None:
                if item.reason != "omitted_silence":
                    continue
                state = ((), False, False, False)
                probabilities = ()
                kind = "unobserved"
            else:
                state = probability_state(item.frame.probabilities)
                if state is None:
                    continue
                probabilities = item.frame.probabilities
                kind = "original"
            for piece in item.source_map:
                span = piece.capture
                if span is None or piece.kind != kind:
                    continue
                start, end = span.normalized_start_sample, span.normalized_end_sample
                if start is None or end is None or start >= end:
                    continue
                if (
                    piece.compact_start_sample < 0
                    or piece.compact_end_sample > item.processed_through_sample
                    or (
                        item.frame is None
                        and (piece.compact_start_sample, piece.compact_end_sample) != (0, 0)
                    )
                    or (
                        item.frame is not None
                        and piece.compact_end_sample - piece.compact_start_sample != end - start
                    )
                    or (
                        item.frame is not None
                        and (
                            piece.compact_start_sample < item.frame.start_sample
                            or piece.compact_end_sample > item.frame.end_sample
                        )
                    )
                ):
                    continue
                overlaps = [
                    frame
                    for frame in self.frames
                    if frame.span.capture_epoch == span.capture_epoch
                    and frame.span.normalized_start_sample < end
                    and start < frame.span.normalized_end_sample
                ]
                if any(frame.closed or frame.compact_end <= previous for frame in overlaps):
                    continue
                for frame in overlaps:
                    self.frames.remove(frame)
                self.frames.append(
                    SpeakerFrame(
                        span,
                        item.speaker_epoch,
                        piece.compact_end_sample,
                        item.observation_ready_monotonic_s,
                        item.frame is None or item.range_state == "closed",
                        *state,
                        probabilities=probabilities,
                        native_frame=(
                            None
                            if item.frame is None
                            else (item.frame.start_sample, item.frame.end_sample)
                        ),
                    )
                )
                changed = True
        self.frames.sort(
            key=lambda frame: (frame.span.capture_epoch, frame.span.normalized_start_sample)
        )
        if len(self.frames) > self.MAX_FRAMES:
            removed = self.frames[: -self.MAX_FRAMES]
            self.frames = self.frames[-self.MAX_FRAMES :]
            for token, proof in tuple(self._proofs.items()):
                if any(
                    span.capture_epoch == frame.span.capture_epoch
                    and span.normalized_start_sample < frame.span.normalized_end_sample
                    and frame.span.normalized_start_sample < span.normalized_end_sample
                    for span in (*proof.ranges, *proof.replayed)
                    for frame in removed
                ):
                    self.release(token)
        if changed:
            self._changed()

    def stable_frames(self):
        return (
            frame for frame in self.frames if frame.closed or frame.compact_end <= self.immutable
        )

    def attribute(self, ranges):
        return attribute_speaker(ranges, self.frames, self.immutable)

    def transitions(self):
        previous = candidate = onset = None
        confirmations = 0
        last_capture = last_end = last_epoch = last_native = None
        for frame in self.stable_frames():
            if (
                last_capture != frame.span.capture_epoch
                or last_epoch != frame.epoch
                or last_end != frame.span.normalized_start_sample
                or frame.span.discontinuity_before is not None
            ):
                previous = candidate = onset = None
                confirmations = 0
                last_native = None
            last_capture, last_end, last_epoch = (
                frame.span.capture_epoch,
                frame.span.normalized_end_sample,
                frame.epoch,
            )
            if frame.competing or len(frame.slots) > 1:
                previous = candidate = onset = None
                confirmations = 0
                last_native = None
                continue
            leader = leading_slot(frame)
            if previous is not None and leader is not None and (leader,) != previous.slots:
                if onset is None or leading_slot(onset) != leader:
                    onset = frame
            else:
                onset = None
            if not frame.clear:
                candidate = None
                confirmations = 0
                last_native = None
                continue
            if previous is not None and frame.slots == previous.slots:
                previous = frame
                candidate = onset = None
                confirmations = 0
                last_native = None
                continue
            if candidate is None or frame.slots != candidate.slots:
                candidate = frame
                confirmations = 1
                last_native = frame.native_frame
                continue
            if frame.native_frame != last_native:
                confirmations += 1
                last_native = frame.native_frame
            if confirmations < self.CONFIRM_FRAMES:
                continue
            if previous is not None:
                yield (
                    STTAudioBoundary(
                        f"nemotron:{candidate.epoch}:{candidate.span.capture_epoch}:{candidate.span.normalized_start_sample}",
                        candidate.span.capture_epoch,
                        candidate.span.normalized_start_sample,
                        frame.ready_at,
                        "speaker",
                        str(candidate.epoch),
                        transition_start_sample=previous.span.normalized_end_sample,
                        transition_end_sample=candidate.span.normalized_start_sample,
                        successor_onset_sample=onset.span.normalized_start_sample,
                    ),
                    previous,
                    candidate,
                )
            previous = frame
            candidate = onset = None
            confirmations = 0
            last_native = None

    def boundaries(self):
        return (boundary for boundary, _, _ in self.transitions())
