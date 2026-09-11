from __future__ import annotations

import asyncio
import math
from collections.abc import Awaitable, Callable
from typing import Literal
from uuid import UUID

import numpy as np

from puripuly_heart.core.audio.format import AudioCaptureSpan
from puripuly_heart.core.audio.ownership import OwnedVadEvent, PeerAudioSegmentLedger
from puripuly_heart.core.audio.smart_turn import (
    SMART_TURN_INPUT_REVISION,
    SmartTurnCompletion,
    SmartTurnInferenceOwner,
    SmartTurnRequestIdentity,
)
from puripuly_heart.core.vad.gating import SpeechChunk, SpeechEnd, SpeechStart

ListenOwnedEventSink = Callable[[OwnedVadEvent], Awaitable[None]]


class ListenDeliveryController:
    STEP_AGE_S = 4.0
    STEP_PAUSE_MS = 224
    HARD_LIMIT_S = 6.0
    SMART_PROBE_MS = 224
    SMART_COMPLETE_MS = 512
    SMART_FALLBACK_MS = 800
    MAX_CONTEXT_SAMPLES = 8 * 16000

    def __init__(
        self,
        *,
        vad: object,
        ledger: PeerAudioSegmentLedger,
        emit: ListenOwnedEventSink,
        monotonic_clock: Callable[[], float],
        smart_turn_owner: SmartTurnInferenceOwner | None = None,
    ) -> None:
        self._vad = vad
        self._ledger = ledger
        self._emit = emit
        self._monotonic_clock = monotonic_clock
        self._smart_turn_owner = smart_turn_owner or SmartTurnInferenceOwner(clock=monotonic_clock)
        self._owns_smart_turn_owner = smart_turn_owner is None
        self._segment_id: UUID | None = None
        self._opened_at_s: float | None = None
        self._pause_samples = 0
        self._pause_id = 0
        self._pause_started_at_s: float | None = None
        self._sample_rate_hz = 0
        self._source_frontier = 0
        self._context_revision = 0
        self._context_parts: list[np.ndarray] = []
        self._context_samples = 0
        self._probe_attempted = False
        self._probe_status: Literal["none", "started", "busy", "unavailable"] = "none"
        self._completion: SmartTurnCompletion | None = None
        self._step_task: asyncio.Task[None] | None = None
        self._hard_task: asyncio.Task[None] | None = None
        self._closed = False
        self._seal_lock = asyncio.Lock()
        self._ledger.bind_delivery_seal_port(self)

    @property
    def current_segment_id(self) -> UUID | None:
        return self._segment_id

    async def seal_prospective_transition(
        self,
        *,
        capture_epoch: int,
        requested_source_sample: int,
    ) -> tuple[
        Literal["sealed", "already_separated", "too_late_for_current_scope", "invalid_source"],
        int | None,
        UUID | None,
    ]:
        async with self._seal_lock:
            scope = self._ledger.source_scope(
                capture_epoch=capture_epoch,
                source_sample=requested_source_sample,
            )
            segment_id = self._segment_id
            if scope == "already_separated":
                return "already_separated", None, segment_id
            if scope == "irreversible":
                return "too_late_for_current_scope", None, segment_id
            if scope != "current" or segment_id is None:
                return "invalid_source", None, segment_id
            snapshot = self._current_snapshot(segment_id)
            actual_frontier = (
                snapshot.content_ranges[-1].normalized_end_sample
                if snapshot is not None and snapshot.content_ranges
                else None
            )
            sealed = await self._seal_locked(
                segment_id,
                reason="prospective_speaker_transition",
                rollover=True,
            )
            if sealed:
                return "sealed", actual_frontier, segment_id
            scope = self._ledger.source_scope(
                capture_epoch=capture_epoch,
                source_sample=requested_source_sample,
            )
            if scope == "already_separated":
                return "already_separated", None, segment_id
            return "too_late_for_current_scope", None, segment_id

    async def handle_vad_event(self, event: object) -> None:
        owned = self._ledger.observe_vad_event(event, now_monotonic_s=self._monotonic_clock())
        if isinstance(event, SpeechStart):
            self._segment_id = event.utterance_id
            self._opened_at_s = owned.segment.opened_at_monotonic_s
            self._pause_samples = 0
            self._pause_started_at_s = None
            self._pause_id += 1
            self._sample_rate_hz = owned.segment.settings.target_sample_rate_hz
            self._reset_probe()
            if event.genuine_onset:
                self._reset_context()
            self._append_context(event.pre_roll)
            self._append_context(event.chunk)
            if owned.segment.settings.delivery_profile_effective == "on":
                self._smart_turn_owner.request_prepare()
            self._arm_timers(event.utterance_id, self._opened_at_s)
        elif isinstance(event, SpeechChunk) and event.utterance_id == self._segment_id:
            self._append_context(event.chunk)
        elif isinstance(event, SpeechEnd) and event.utterance_id == self._segment_id:
            natural_endpoint = event.reason in {
                "delivery_pause",
                "source_eof",
                "silence",
                "soft_pause",
            }
            discontinuity = event.reason == "source_discontinuity"
            self._clear_active_segment()
            if natural_endpoint or discontinuity:
                self._reset_context()
        await self._emit(owned)

    async def observe_acoustic_chunk(
        self,
        *,
        speech_observed: bool,
        capture: tuple[AudioCaptureSpan, ...],
    ) -> None:
        segment_id = self._segment_id
        opened_at_s = self._opened_at_s
        if segment_id is None or opened_at_s is None:
            return
        observed_samples = sum(item.normalized_sample_count for item in capture)
        frontier_s = self._capture_frontier(capture)
        if capture:
            normalized_end = capture[-1].normalized_end_sample
            if normalized_end is not None:
                self._source_frontier = normalized_end
        if speech_observed:
            if self._pause_samples:
                self._pause_id += 1
                self._reset_probe()
            self._pause_samples = 0
            self._pause_started_at_s = None
        else:
            if self._pause_samples == 0 and frontier_s is not None and self._sample_rate_hz > 0:
                self._pause_started_at_s = frontier_s - observed_samples / self._sample_rate_hz
            self._pause_samples += observed_samples
        if frontier_s is None:
            return
        age_s = max(0.0, frontier_s - opened_at_s)
        if age_s >= self.HARD_LIMIT_S:
            await self._seal(segment_id, reason="delivery_deadline", rollover=True)
            return
        pause_ms = self._observed_pause_ms()
        if not speech_observed and age_s >= self.STEP_AGE_S:
            if pause_ms >= self.STEP_PAUSE_MS:
                await self._seal(segment_id, reason="delivery_pause", rollover=False)
            return
        if speech_observed:
            return
        snapshot = self._current_snapshot(segment_id)
        if snapshot is None:
            return
        if snapshot.settings.delivery_profile_effective != "on":
            if pause_ms >= snapshot.settings.vad_hangover_ms:
                await self._seal(segment_id, reason="delivery_pause", rollover=False)
            return
        if pause_ms >= self.SMART_PROBE_MS and not self._probe_attempted:
            self._request_probe(snapshot.identity.activation_generation, segment_id)
        if pause_ms >= self.SMART_COMPLETE_MS and self._timely_complete(
            snapshot.settings.delivery_threshold
        ):
            await self._seal(segment_id, reason="delivery_pause", rollover=False)
            return
        if pause_ms >= self.SMART_FALLBACK_MS:
            await self._seal(segment_id, reason="delivery_pause", rollover=False)

    async def close(self) -> None:
        self._closed = True
        tasks = self._cancel_timers()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if self._owns_smart_turn_owner:
            await self._smart_turn_owner.close()

    def cancel(self) -> None:
        self._closed = True
        self._cancel_timers()

    def invalidate_context(self) -> None:
        self._pause_samples = 0
        self._pause_started_at_s = None
        self._pause_id += 1
        self._reset_probe()
        self._reset_context()

    def _request_probe(self, activation_generation: int, segment_id: UUID) -> None:
        self._probe_attempted = True
        pause_started = self._pause_started_at_s
        if pause_started is None:
            self._probe_status = "unavailable"
            return
        identity = SmartTurnRequestIdentity(
            activation_generation=activation_generation,
            segment_id=segment_id,
            pause_id=self._pause_id,
            context_revision=self._context_revision,
            input_revision=SMART_TURN_INPUT_REVISION,
            source_frontier=self._source_frontier,
            probe_frontier_monotonic_s=pause_started + self.SMART_PROBE_MS / 1000.0,
            complete_deadline_monotonic_s=pause_started + self.SMART_COMPLETE_MS / 1000.0,
        )
        audio = np.concatenate(self._context_parts) if self._context_parts else np.empty(0)
        self._probe_status = self._smart_turn_owner.submit(
            identity, audio, self._receive_completion
        )

    async def _receive_completion(self, completion: SmartTurnCompletion) -> None:
        identity = completion.identity
        if (
            self._closed
            or identity.segment_id != self._segment_id
            or identity.pause_id != self._pause_id
        ):
            return
        if completion.completed_at_monotonic_s >= identity.complete_deadline_monotonic_s:
            self._smart_turn_owner.record_late()
        self._completion = completion

    def _timely_complete(self, threshold: float | None) -> bool:
        result = self._completion
        if (
            result is None
            or threshold is None
            or result.outcome != "complete"
            or result.score is None
        ):
            return False
        return (
            result.completed_at_monotonic_s < result.identity.complete_deadline_monotonic_s
            and result.score >= threshold
        )

    def _arm_timers(self, segment_id: UUID, opened_at_s: float) -> None:
        self._cancel_timers()
        now = self._monotonic_clock()
        self._step_task = asyncio.create_task(
            self._run_step_timer(segment_id, max(0.0, opened_at_s + self.STEP_AGE_S - now)),
            name="listen-step",
        )
        self._hard_task = asyncio.create_task(
            self._run_hard_timer(segment_id, max(0.0, opened_at_s + self.HARD_LIMIT_S - now)),
            name="listen-deadline",
        )

    async def _run_step_timer(self, segment_id: UUID, delay_s: float) -> None:
        await asyncio.sleep(delay_s)
        if (
            not self._closed
            and segment_id == self._segment_id
            and self._observed_pause_ms() >= self.STEP_PAUSE_MS
        ):
            await self._seal(segment_id, reason="delivery_pause", rollover=False)

    async def _run_hard_timer(self, segment_id: UUID, delay_s: float) -> None:
        await asyncio.sleep(delay_s)
        if not self._closed and segment_id == self._segment_id:
            await self._seal(segment_id, reason="delivery_deadline", rollover=True)

    async def _seal(self, segment_id: UUID, *, reason: str, rollover: bool) -> bool:
        async with self._seal_lock:
            return await self._seal_locked(segment_id, reason=reason, rollover=rollover)

    async def _seal_locked(
        self,
        segment_id: UUID,
        *,
        reason: str,
        rollover: bool,
    ) -> bool:
        if segment_id != self._segment_id:
            return False
        method_name = "seal_active_for_rollover" if rollover else "seal_active"
        seal = getattr(self._vad, method_name, None)
        if not callable(seal):
            raise RuntimeError(f"peer VAD does not support {method_name}")
        event = seal(reason=reason)
        if event is None:
            return False
        await self.handle_vad_event(event)
        return True

    def _clear_active_segment(self) -> None:
        self._segment_id = None
        self._opened_at_s = None
        self._pause_samples = 0
        self._pause_started_at_s = None
        self._sample_rate_hz = 0
        self._reset_probe()
        self._cancel_timers()

    def _reset_probe(self) -> None:
        self._probe_attempted = False
        self._probe_status = "none"
        self._completion = None

    def _reset_context(self) -> None:
        self._context_parts.clear()
        self._context_samples = 0
        self._context_revision += 1

    def _append_context(self, audio: np.ndarray) -> None:
        value = np.asarray(audio, dtype=np.float32).reshape(-1)
        if not value.size:
            return
        self._context_parts.append(value.copy())
        self._context_samples += int(value.size)
        excess = self._context_samples - self.MAX_CONTEXT_SAMPLES
        while excess > 0 and self._context_parts:
            first = self._context_parts[0]
            if first.size <= excess:
                self._context_parts.pop(0)
                self._context_samples -= int(first.size)
                excess -= int(first.size)
            else:
                self._context_parts[0] = first[excess:].copy()
                self._context_samples -= excess
                excess = 0
        self._context_revision += 1

    def _cancel_timers(self) -> tuple[asyncio.Task[None], ...]:
        current = asyncio.current_task()
        tasks = tuple(
            task
            for task in (self._step_task, self._hard_task)
            if task is not None and task is not current
        )
        self._step_task = None
        self._hard_task = None
        for task in tasks:
            task.cancel()
        return tasks

    def _observed_pause_ms(self) -> int:
        if self._sample_rate_hz <= 0 or self._pause_samples <= 0:
            return 0
        return int(math.floor(self._pause_samples * 1000 / self._sample_rate_hz))

    def _current_snapshot(self, segment_id: UUID):
        return next(
            (
                snapshot
                for snapshot in self._ledger.snapshots
                if snapshot.identity.segment_id == segment_id
            ),
            None,
        )

    @staticmethod
    def _capture_frontier(capture: tuple[AudioCaptureSpan, ...]) -> float | None:
        return capture[-1].source_end_monotonic_s if capture else None


__all__ = ["ListenDeliveryController", "ListenOwnedEventSink"]
