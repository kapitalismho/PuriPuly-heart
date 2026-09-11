from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field, replace
from typing import Literal
from uuid import UUID

from puripuly_heart.core.audio.format import AudioCaptureSpan

SegmentTerminalOutcome = Literal[
    "final",
    "empty",
    "degraded",
    "suppressed",
    "failed",
    "expired",
    "cancelled",
]


@dataclass(frozen=True, slots=True)
class AudioSegmentSettingsSnapshot:
    provider_id: str
    provider_signature: tuple[object, ...]
    runtime_signature: tuple[object, ...]
    source_mode: str
    source_language: str
    expected_languages: tuple[str, ...]
    target_sample_rate_hz: int
    vad_speech_threshold: float
    vad_hangover_ms: int
    vad_pre_roll_ms: int
    delivery_profile_requested: str = "off"
    delivery_profile_effective: str = "off"
    delivery_availability: str = "disabled"
    delivery_threshold: float | None = None
    delivery_input_revision: str | None = None


@dataclass(frozen=True, slots=True)
class AudioSegmentIdentity:
    activation_generation: int
    segment_order: int
    segment_id: UUID
    capture_epoch: int


@dataclass(frozen=True, slots=True)
class AudioSegmentSnapshot:
    identity: AudioSegmentIdentity
    settings: AudioSegmentSettingsSnapshot
    content_ranges: tuple[AudioCaptureSpan, ...]
    context_ranges: tuple[AudioCaptureSpan, ...]
    failed_ranges: tuple[AudioCaptureSpan, ...]
    content_sample_count: int
    context_sample_count: int
    failed_normalized_sample_count: int
    failed_source_sample_count: int
    prefix_context_sample_count: int
    synthetic_context_sample_count: int
    genuine_onset: bool
    state: Literal["open", "sealed", "terminal"]
    opened_at_monotonic_s: float
    sealed_at_monotonic_s: float | None
    seal_reason: str | None


@dataclass(frozen=True, slots=True)
class AudioSegmentTerminalReceipt:
    identity: AudioSegmentIdentity
    outcome: SegmentTerminalOutcome
    segment: AudioSegmentSnapshot
    terminal_at_monotonic_s: float
    provider_epoch_id: str | None = None
    provider_turn_id: str | None = None
    native_request_id: str | None = None
    text_authority: Literal["authoritative", "degraded", "none"] = "none"
    failure_reason: str | None = None


@dataclass(frozen=True, slots=True)
class OwnedVadEvent:
    event: object
    segment: AudioSegmentSnapshot


@dataclass(slots=True)
class _MutableSegment:
    identity: AudioSegmentIdentity
    settings: AudioSegmentSettingsSnapshot
    opened_at_monotonic_s: float
    genuine_onset: bool
    content_ranges: list[AudioCaptureSpan] = field(default_factory=list)
    failed_ranges: list[AudioCaptureSpan] = field(default_factory=list)
    context_ranges: list[AudioCaptureSpan] = field(default_factory=list)
    synthetic_context_sample_count: int = 0
    sealed_at_monotonic_s: float | None = None
    seal_reason: str | None = None
    terminal: AudioSegmentTerminalReceipt | None = None


class PeerAudioSegmentLedger:
    _MAX_RETIRED_RECEIPTS = 4096
    _MAX_CLAIM_INTERVALS_PER_EPOCH = 64

    def __init__(
        self,
        *,
        activation_generation: int,
        settings: AudioSegmentSettingsSnapshot,
    ) -> None:
        self._activation_generation = activation_generation
        self._settings = settings
        self._next_order = 1
        self._next_retirement_order = 1
        self._open_segment_id: UUID | None = None
        self._segments: dict[UUID, _MutableSegment] = {}
        self._segment_ids_by_order: dict[int, UUID] = {}
        self._terminal_by_order: dict[int, AudioSegmentTerminalReceipt] = {}
        self._retired_receipts: OrderedDict[UUID, AudioSegmentTerminalReceipt] = OrderedDict()
        self._ready_terminal_receipts: list[AudioSegmentTerminalReceipt] = []
        self._claimed_ranges: dict[int, list[tuple[int, int]]] = {}
        self._delivery_seal_port: object | None = None

    def rebind(
        self,
        *,
        activation_generation: int,
        settings: AudioSegmentSettingsSnapshot,
    ) -> None:
        self._activation_generation = activation_generation
        self._settings = settings

    @property
    def current_open_segment_id(self) -> UUID | None:
        return self._open_segment_id

    @property
    def snapshots(self) -> tuple[AudioSegmentSnapshot, ...]:
        return tuple(
            self._snapshot(self._segments[self._segment_ids_by_order[order]])
            for order in sorted(self._segment_ids_by_order)
        )

    @property
    def terminal_receipts(self) -> tuple[AudioSegmentTerminalReceipt, ...]:
        receipts = [
            *self._retired_receipts.values(),
            *self._terminal_by_order.values(),
        ]
        return tuple(sorted(receipts, key=lambda item: item.identity.segment_order))

    @property
    def activation_generation(self) -> int:
        return self._activation_generation

    def take_ready_terminal_receipts(self) -> tuple[AudioSegmentTerminalReceipt, ...]:
        receipts = tuple(self._ready_terminal_receipts)
        self._ready_terminal_receipts.clear()
        return receipts

    @property
    def delivery_seal_port(self) -> object | None:
        return self._delivery_seal_port

    def bind_delivery_seal_port(self, port: object) -> None:
        if self._delivery_seal_port is not None and self._delivery_seal_port is not port:
            raise RuntimeError("audio segment ledger delivery authority is already bound")
        self._delivery_seal_port = port

    def source_scope(
        self,
        *,
        capture_epoch: int,
        source_sample: int,
    ) -> Literal["current", "already_separated", "irreversible", "unknown"]:
        current = list(self.snapshots)
        current_ids = {snapshot.identity.segment_id for snapshot in current}
        matching = [
            snapshot
            for snapshot in (
                *current,
                *(
                    receipt.segment
                    for receipt in self.terminal_receipts
                    if receipt.identity.segment_id not in current_ids
                ),
            )
            if snapshot.identity.capture_epoch == capture_epoch
        ]
        open_snapshot = next(
            (snapshot for snapshot in matching if snapshot.state == "open"),
            None,
        )
        if open_snapshot is not None:
            current_ranges = open_snapshot.content_ranges
            if current_ranges:
                current_start = current_ranges[0].normalized_start_sample
                current_end = current_ranges[-1].normalized_end_sample
                if (
                    current_start is not None
                    and current_end is not None
                    and current_start < source_sample <= current_end
                ):
                    return "current"
                if current_start is not None and source_sample <= current_start:
                    for snapshot in matching:
                        if snapshot.state == "open":
                            continue
                        if any(
                            item.normalized_start_sample is not None
                            and item.normalized_end_sample is not None
                            and item.normalized_start_sample
                            < source_sample
                            < item.normalized_end_sample
                            for item in snapshot.content_ranges
                        ):
                            return "irreversible"
                    return "already_separated"
            return "unknown"
        if any(
            item.normalized_start_sample is not None
            and item.normalized_end_sample is not None
            and item.normalized_start_sample < source_sample <= item.normalized_end_sample
            for snapshot in matching
            for item in snapshot.content_ranges
        ):
            return "irreversible"
        return "unknown"

    def observe_vad_event(self, event: object, *, now_monotonic_s: float) -> OwnedVadEvent:
        from puripuly_heart.core.vad.gating import SpeechChunk, SpeechEnd, SpeechStart

        if isinstance(event, SpeechStart):
            if self._open_segment_id is not None:
                raise RuntimeError("cannot open a segment while another segment is open")
            capture_epoch = self._capture_epoch(event.chunk_capture)
            identity = AudioSegmentIdentity(
                activation_generation=self._activation_generation,
                segment_order=self._next_order,
                segment_id=event.utterance_id,
                capture_epoch=capture_epoch,
            )
            self._next_order += 1
            segment = _MutableSegment(
                identity=identity,
                settings=self._settings,
                opened_at_monotonic_s=self._content_opened_at(
                    event.chunk_capture,
                    fallback=now_monotonic_s,
                ),
                genuine_onset=event.genuine_onset,
            )
            self._segments[event.utterance_id] = segment
            self._segment_ids_by_order[identity.segment_order] = event.utterance_id
            self._open_segment_id = event.utterance_id
            segment.context_ranges.extend(event.pre_roll_capture)
            self._append_content(segment, event.chunk_capture, int(event.chunk.size))
            return OwnedVadEvent(event=event, segment=self._snapshot(segment))

        if isinstance(event, SpeechChunk):
            segment = self._require_writable_segment(event.utterance_id)
            self._append_content(segment, event.chunk_capture, int(event.chunk.size))
            return OwnedVadEvent(event=event, segment=self._snapshot(segment))

        if isinstance(event, SpeechEnd):
            segment = self._require_writable_segment(event.utterance_id)
            segment.sealed_at_monotonic_s = now_monotonic_s
            segment.seal_reason = event.reason
            if self._open_segment_id == event.utterance_id:
                self._open_segment_id = None
            return OwnedVadEvent(event=event, segment=self._snapshot(segment))

        raise TypeError(f"unknown VAD event: {type(event)!r}")

    def claim_open_content_for_failure(
        self,
        ranges: tuple[AudioCaptureSpan, ...],
    ) -> None:
        segment_id = self._open_segment_id
        if segment_id is None:
            return
        segment = self._require_writable_segment(segment_id)
        for item in ranges:
            if item.normalized_sample_count:
                segment.failed_ranges.extend(self._claim_ranges((item,)))
            else:
                segment.failed_ranges.append(item)

    def terminalize(
        self,
        segment_id: UUID,
        *,
        outcome: SegmentTerminalOutcome,
        now_monotonic_s: float,
        provider_epoch_id: str | None = None,
        provider_turn_id: str | None = None,
        native_request_id: str | None = None,
        text_authority: Literal["authoritative", "degraded", "none"] = "none",
        failure_reason: str | None = None,
    ) -> AudioSegmentTerminalReceipt:
        retired = self._retired_receipts.get(segment_id)
        if retired is not None:
            return retired
        segment = self._require_segment(segment_id)
        if segment.terminal is not None:
            return segment.terminal
        if segment.sealed_at_monotonic_s is None:
            raise RuntimeError("cannot terminalize an open audio segment")
        return self._terminalize_sealed(
            segment,
            outcome=outcome,
            now_monotonic_s=now_monotonic_s,
            provider_epoch_id=provider_epoch_id,
            provider_turn_id=provider_turn_id,
            native_request_id=native_request_id,
            text_authority=text_authority,
            failure_reason=failure_reason,
        )

    def terminalize_for_failure(
        self,
        segment_id: UUID,
        *,
        now_monotonic_s: float,
        failure_reason: str,
        provider_epoch_id: str | None = None,
        provider_turn_id: str | None = None,
        text_authority: Literal["authoritative", "degraded", "none"] = "none",
        outcome: SegmentTerminalOutcome = "failed",
    ) -> AudioSegmentTerminalReceipt:
        retired = self._retired_receipts.get(segment_id)
        if retired is not None:
            return retired
        segment = self._require_segment(segment_id)
        if segment.terminal is not None:
            return segment.terminal
        if segment.sealed_at_monotonic_s is None:
            segment.sealed_at_monotonic_s = now_monotonic_s
            segment.seal_reason = failure_reason
            if self._open_segment_id == segment_id:
                self._open_segment_id = None
        return self._terminalize_sealed(
            segment,
            outcome=outcome,
            now_monotonic_s=now_monotonic_s,
            provider_epoch_id=provider_epoch_id,
            provider_turn_id=provider_turn_id,
            text_authority=text_authority,
            failure_reason=failure_reason,
        )

    def terminalize_open_for_source_loss(
        self,
        *,
        now_monotonic_s: float,
    ) -> AudioSegmentTerminalReceipt | None:
        segment_id = self._open_segment_id
        if segment_id is None:
            return None
        segment = self._require_writable_segment(segment_id)
        segment.sealed_at_monotonic_s = now_monotonic_s
        segment.seal_reason = "source_discontinuity"
        self._open_segment_id = None
        return self._terminalize_sealed(
            segment,
            outcome="failed",
            now_monotonic_s=now_monotonic_s,
            text_authority="none",
        )

    def cancel_unfinished(
        self, *, now_monotonic_s: float
    ) -> tuple[AudioSegmentTerminalReceipt, ...]:
        receipts: list[AudioSegmentTerminalReceipt] = []
        segment_ids = tuple(
            self._segment_ids_by_order[order] for order in sorted(self._segment_ids_by_order)
        )
        for segment_id in segment_ids:
            segment = self._segments.get(segment_id)
            if segment is None or segment.terminal is not None:
                continue
            if segment.sealed_at_monotonic_s is None:
                segment.sealed_at_monotonic_s = now_monotonic_s
                segment.seal_reason = "cancelled"
                if self._open_segment_id == segment_id:
                    self._open_segment_id = None
            receipts.append(
                self._terminalize_sealed(
                    segment,
                    outcome="cancelled",
                    now_monotonic_s=now_monotonic_s,
                )
            )
        return tuple(receipts)

    def fail_unresolved_after_drain(
        self,
        *,
        now_monotonic_s: float,
    ) -> tuple[AudioSegmentTerminalReceipt, ...]:
        receipts: list[AudioSegmentTerminalReceipt] = []
        segment_ids = tuple(
            self._segment_ids_by_order[order] for order in sorted(self._segment_ids_by_order)
        )
        for segment_id in segment_ids:
            segment = self._segments.get(segment_id)
            if segment is None or segment.terminal is not None:
                continue
            if segment.sealed_at_monotonic_s is None:
                raise RuntimeError("provider drain completed with an open audio segment")
            receipts.append(
                self._terminalize_sealed(
                    segment,
                    outcome="failed",
                    now_monotonic_s=now_monotonic_s,
                    text_authority="none",
                    failure_reason="provider_drain_without_scoped_terminal",
                )
            )
        return tuple(receipts)

    def _append_content(
        self,
        segment: _MutableSegment,
        ranges: tuple[AudioCaptureSpan, ...],
        delivered_sample_count: int,
    ) -> None:
        claimed = self._claim_ranges(ranges)
        self._extend_coalesced_ranges(segment.content_ranges, claimed)
        real_sample_count = sum(item.normalized_sample_count for item in ranges)
        segment.synthetic_context_sample_count += max(0, delivered_sample_count - real_sample_count)

    @staticmethod
    def _extend_coalesced_ranges(
        target: list[AudioCaptureSpan],
        incoming: list[AudioCaptureSpan],
    ) -> None:
        for item in incoming:
            if not target:
                target.append(item)
                continue
            previous = target[-1]
            contiguous = (
                item.discontinuity_before is None
                and previous.capture_epoch == item.capture_epoch
                and previous.source_sample_rate_hz == item.source_sample_rate_hz
                and previous.source_end_sample == item.source_start_sample
                and previous.normalized_sample_rate_hz == item.normalized_sample_rate_hz
                and previous.normalized_end_sample == item.normalized_start_sample
            )
            if not contiguous:
                target.append(item)
                continue
            target[-1] = replace(
                previous,
                source_end_sample=item.source_end_sample,
                source_end_monotonic_s=item.source_end_monotonic_s,
                normalized_end_sample=item.normalized_end_sample,
            )

    def _claim_ranges(
        self,
        ranges: tuple[AudioCaptureSpan, ...],
    ) -> list[AudioCaptureSpan]:
        claimed: list[AudioCaptureSpan] = []
        for item in ranges:
            self._prune_claimed_epochs(item.capture_epoch)
            start = item.normalized_start_sample
            end = item.normalized_end_sample
            if start is None or end is None or end <= start:
                continue
            remaining = [(start, end)]
            prior = self._claimed_ranges.setdefault(item.capture_epoch, [])
            for prior_start, prior_end in prior:
                next_remaining: list[tuple[int, int]] = []
                for current_start, current_end in remaining:
                    if prior_end <= current_start or prior_start >= current_end:
                        next_remaining.append((current_start, current_end))
                        continue
                    if current_start < prior_start:
                        next_remaining.append((current_start, prior_start))
                    if prior_end < current_end:
                        next_remaining.append((prior_end, current_end))
                remaining = next_remaining
            for current_start, current_end in remaining:
                sliced = item.slice_normalized(current_start, current_end)
                claimed.append(sliced)
                prior.append((current_start, current_end))
            prior.sort()
            merged = self._merge_intervals(prior)
            self._claimed_ranges[item.capture_epoch] = merged[
                -self._MAX_CLAIM_INTERVALS_PER_EPOCH :
            ]
        return claimed

    @staticmethod
    def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
        merged: list[tuple[int, int]] = []
        for start, end in intervals:
            if not merged or start > merged[-1][1]:
                merged.append((start, end))
            else:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        return merged

    @staticmethod
    def _capture_epoch(ranges: tuple[AudioCaptureSpan, ...]) -> int:
        return ranges[0].capture_epoch if ranges else 0

    @staticmethod
    def _content_opened_at(
        ranges: tuple[AudioCaptureSpan, ...],
        *,
        fallback: float,
    ) -> float:
        return ranges[0].source_start_monotonic_s if ranges else fallback

    def contains_segment(self, segment_id: UUID) -> bool:
        return segment_id in self._segments or segment_id in self._retired_receipts

    def _terminalize_sealed(
        self,
        segment: _MutableSegment,
        *,
        outcome: SegmentTerminalOutcome,
        now_monotonic_s: float,
        provider_epoch_id: str | None = None,
        provider_turn_id: str | None = None,
        native_request_id: str | None = None,
        text_authority: Literal["authoritative", "degraded", "none"] = "none",
        failure_reason: str | None = None,
    ) -> AudioSegmentTerminalReceipt:
        if outcome == "empty" and text_authority != "authoritative":
            raise ValueError("empty terminal outcome requires authoritative provider completion")
        receipt = AudioSegmentTerminalReceipt(
            identity=segment.identity,
            outcome=outcome,
            segment=self._snapshot(segment, terminal=True),
            terminal_at_monotonic_s=now_monotonic_s,
            provider_epoch_id=provider_epoch_id,
            provider_turn_id=provider_turn_id,
            native_request_id=native_request_id,
            text_authority=text_authority,
            failure_reason=failure_reason,
        )
        segment.terminal = receipt
        self._terminal_by_order[segment.identity.segment_order] = receipt
        self._retire_ready_terminal_receipts()
        return receipt

    def _retire_ready_terminal_receipts(self) -> None:
        while self._next_retirement_order in self._terminal_by_order:
            order = self._next_retirement_order
            receipt = self._terminal_by_order.pop(order)
            self._next_retirement_order += 1
            segment_id = self._segment_ids_by_order.pop(order)
            self._segments.pop(segment_id, None)
            self._retired_receipts[segment_id] = receipt
            self._ready_terminal_receipts.append(receipt)
            self._retired_receipts.move_to_end(segment_id)
            while len(self._retired_receipts) > self._MAX_RETIRED_RECEIPTS:
                self._retired_receipts.popitem(last=False)

    def _require_writable_segment(self, segment_id: UUID) -> _MutableSegment:
        segment = self._require_segment(segment_id)
        if segment.terminal is not None:
            raise RuntimeError("cannot mutate a terminal audio segment")
        if segment.sealed_at_monotonic_s is not None:
            raise RuntimeError("cannot mutate a sealed audio segment")
        return segment

    def _require_segment(self, segment_id: UUID) -> _MutableSegment:
        segment = self._segments.get(segment_id)
        if segment is None:
            if segment_id in self._retired_receipts:
                raise RuntimeError("cannot mutate a terminal audio segment")
            raise KeyError(f"unknown segment: {segment_id}")
        return segment

    def _prune_claimed_epochs(self, current_epoch: int) -> None:
        for capture_epoch in tuple(self._claimed_ranges):
            if capture_epoch != current_epoch:
                self._claimed_ranges.pop(capture_epoch, None)

    @staticmethod
    def _snapshot(
        segment: _MutableSegment,
        *,
        terminal: bool = False,
    ) -> AudioSegmentSnapshot:
        content_ranges = tuple(segment.content_ranges)
        context_ranges = tuple(segment.context_ranges)
        failed_ranges = tuple(segment.failed_ranges)
        content_sample_count = sum(item.normalized_sample_count for item in content_ranges)
        failed_normalized_sample_count = sum(item.normalized_sample_count for item in failed_ranges)
        failed_source_sample_count = sum(item.source_sample_count for item in failed_ranges)
        prefix_context_sample_count = sum(item.normalized_sample_count for item in context_ranges)
        if terminal or segment.terminal is not None:
            state: Literal["open", "sealed", "terminal"] = "terminal"
        elif segment.sealed_at_monotonic_s is not None:
            state = "sealed"
        else:
            state = "open"
        return AudioSegmentSnapshot(
            identity=segment.identity,
            settings=segment.settings,
            content_ranges=content_ranges,
            failed_ranges=failed_ranges,
            context_ranges=context_ranges,
            content_sample_count=content_sample_count,
            failed_normalized_sample_count=failed_normalized_sample_count,
            failed_source_sample_count=failed_source_sample_count,
            context_sample_count=(
                prefix_context_sample_count + segment.synthetic_context_sample_count
            ),
            prefix_context_sample_count=prefix_context_sample_count,
            synthetic_context_sample_count=segment.synthetic_context_sample_count,
            genuine_onset=segment.genuine_onset,
            state=state,
            opened_at_monotonic_s=segment.opened_at_monotonic_s,
            sealed_at_monotonic_s=segment.sealed_at_monotonic_s,
            seal_reason=segment.seal_reason,
        )


__all__ = [
    "AudioSegmentIdentity",
    "AudioSegmentSettingsSnapshot",
    "AudioSegmentSnapshot",
    "AudioSegmentTerminalReceipt",
    "OwnedVadEvent",
    "PeerAudioSegmentLedger",
    "SegmentTerminalOutcome",
]
