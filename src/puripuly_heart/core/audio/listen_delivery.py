from __future__ import annotations

import asyncio
import math
from collections.abc import Awaitable, Callable
from typing import Literal
from uuid import UUID

from puripuly_heart.core.audio.format import AudioCaptureSpan
from puripuly_heart.core.audio.ownership import OwnedVadEvent, PeerAudioSegmentLedger
from puripuly_heart.core.vad.gating import SpeechEnd, SpeechStart

ListenOwnedEventSink = Callable[[OwnedVadEvent], Awaitable[None]]


class ListenOffDeliveryController:
    STEP_AGE_S = 4.0
    STEP_PAUSE_MS = 224
    HARD_LIMIT_S = 6.0

    def __init__(
        self,
        *,
        vad: object,
        ledger: PeerAudioSegmentLedger,
        emit: ListenOwnedEventSink,
        monotonic_clock: Callable[[], float],
    ) -> None:
        self._vad = vad
        self._ledger = ledger
        self._emit = emit
        self._monotonic_clock = monotonic_clock
        self._segment_id: UUID | None = None
        self._opened_at_s: float | None = None
        self._pause_samples = 0
        self._sample_rate_hz = 0
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
        owned = self._ledger.observe_vad_event(
            event,
            now_monotonic_s=self._monotonic_clock(),
        )
        if isinstance(event, SpeechStart):
            self._segment_id = event.utterance_id
            self._opened_at_s = owned.segment.opened_at_monotonic_s
            self._pause_samples = 0
            self._sample_rate_hz = owned.segment.settings.target_sample_rate_hz
            self._arm_timers(event.utterance_id, self._opened_at_s)
        elif isinstance(event, SpeechEnd) and event.utterance_id == self._segment_id:
            self._clear_active_segment()
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
        if speech_observed:
            self._pause_samples = 0
        else:
            self._pause_samples += observed_samples
        frontier_s = self._capture_frontier(capture)
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
        if snapshot is not None and pause_ms >= snapshot.settings.vad_hangover_ms:
            await self._seal(segment_id, reason="delivery_pause", rollover=False)

    async def close(self) -> None:
        self._closed = True
        tasks = self._cancel_timers()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def cancel(self) -> None:
        self._closed = True
        self._cancel_timers()

    def _arm_timers(self, segment_id: UUID, opened_at_s: float) -> None:
        self._cancel_timers()
        now = self._monotonic_clock()
        self._step_task = asyncio.create_task(
            self._run_step_timer(
                segment_id,
                max(0.0, opened_at_s + self.STEP_AGE_S - now),
            ),
            name="listen-off-step",
        )
        self._hard_task = asyncio.create_task(
            self._run_hard_timer(
                segment_id,
                max(0.0, opened_at_s + self.HARD_LIMIT_S - now),
            ),
            name="listen-off-deadline",
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
        self._sample_rate_hz = 0
        self._cancel_timers()

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


__all__ = ["ListenOffDeliveryController", "ListenOwnedEventSink"]
