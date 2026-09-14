from __future__ import annotations

import asyncio
import logging
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
from puripuly_heart.core.vad.gating import (
    PEER_HARD_ROLLOVER_PRE_ROLL_MS,
    SpeechChunk,
    SpeechEnd,
    SpeechStart,
)

# The capture dispatch envelope consists of wholly unsent sealed segments,
# one segment being recognized, and one segment open at the source.
LISTEN_MAX_WHOLE_UNSENT_SEGMENTS = 8
LISTEN_RETAINED_SEGMENT_SLOTS = LISTEN_MAX_WHOLE_UNSENT_SEGMENTS + 2

ListenOwnedEventSink = Callable[[OwnedVadEvent], Awaitable[None]]

logger = logging.getLogger(__name__)


class ListenDeliveryController:
    FOUR_SECOND_AGE_S = 4.0
    FOUR_SECOND_PAUSE_MS = 192
    FIVE_SECOND_AGE_S = 5.0
    FIVE_SECOND_PAUSE_MS = 128
    HARD_LIMIT_S = 6.0
    HARD_CUT_OVERLAP_MS = PEER_HARD_ROLLOVER_PRE_ROLL_MS
    SMART_PROBE_MS = 224
    SMART_COMPLETE_MS = 512
    SMART_INCOMPLETE_MS = 800
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
        self._completion_boundary_decision: Literal["none", "early", "incomplete"] = "none"
        self._four_second_task: asyncio.Task[None] | None = None
        self._five_second_task: asyncio.Task[None] | None = None
        self._hard_task: asyncio.Task[None] | None = None
        self._closed = False
        self._seal_lock = asyncio.Lock()

    @property
    def current_segment_id(self) -> UUID | None:
        return self._segment_id

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
        if not speech_observed and age_s >= self.FIVE_SECOND_AGE_S:
            if pause_ms >= self.FIVE_SECOND_PAUSE_MS:
                await self._seal(segment_id, reason="delivery_pause", rollover=False)
            return
        if not speech_observed and age_s >= self.FOUR_SECOND_AGE_S:
            if pause_ms >= self.FOUR_SECOND_PAUSE_MS:
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
        if pause_ms >= self.SMART_COMPLETE_MS:
            if self._completion_boundary_decision == "none":
                self._completion_boundary_decision = (
                    "incomplete"
                    if self._timely_incomplete(snapshot.settings.delivery_threshold)
                    else "early"
                )
                self._log_completion_decision(snapshot.settings.delivery_threshold)
            if self._completion_boundary_decision == "early":
                await self._seal(segment_id, reason="delivery_pause", rollover=False)
                return
        if pause_ms >= self.SMART_INCOMPLETE_MS:
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
            logger.info(
                "[STT][Runtime] smart-turn inference skipped segment=%s pause=%s "
                "reason=unavailable availability=missing_pause_frontier",
                segment_id,
                self._pause_id,
            )
            return
        probe_samples = self.SMART_PROBE_MS * self._sample_rate_hz // 1000
        samples_after_probe = max(0, self._pause_samples - probe_samples)
        source_frontier = self._source_frontier - samples_after_probe
        identity = SmartTurnRequestIdentity(
            activation_generation=activation_generation,
            segment_id=segment_id,
            pause_id=self._pause_id,
            context_revision=self._context_revision,
            input_revision=SMART_TURN_INPUT_REVISION,
            source_frontier=source_frontier,
            probe_frontier_monotonic_s=pause_started + self.SMART_PROBE_MS / 1000.0,
            complete_deadline_monotonic_s=pause_started + self.SMART_COMPLETE_MS / 1000.0,
        )
        audio = np.concatenate(self._context_parts) if self._context_parts else np.empty(0)
        if samples_after_probe:
            audio = audio[: max(0, audio.size - samples_after_probe)]
        self._probe_status = self._smart_turn_owner.submit(
            identity, audio, self._receive_completion
        )

    async def _receive_completion(self, completion: SmartTurnCompletion) -> None:
        identity = completion.identity
        reason = (
            "closed"
            if self._closed
            else (
                "segment_changed"
                if identity.segment_id != self._segment_id
                else "pause_changed" if identity.pause_id != self._pause_id else None
            )
        )
        if reason is not None:
            logger.info(
                "[STT][Runtime] smart-turn result discarded segment=%s pause=%s reason=%s",
                identity.segment_id,
                identity.pause_id,
                reason,
            )
            return
        if completion.completed_at_monotonic_s >= identity.complete_deadline_monotonic_s:
            self._smart_turn_owner.record_late()
            logger.info(
                "[STT][Runtime] smart-turn result late segment=%s pause=%s late_by_ms=%.1f",
                identity.segment_id,
                identity.pause_id,
                (completion.completed_at_monotonic_s - identity.complete_deadline_monotonic_s)
                * 1000.0,
            )
        self._completion = completion

    def _timely_incomplete(self, threshold: float | None) -> bool:
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
            and result.score < threshold
        )

    def _log_completion_decision(self, threshold: float | None) -> None:
        result = self._completion
        reason = "model"
        if result is None:
            reason = "pending" if self._probe_status == "started" else self._probe_status
        elif threshold is None:
            reason = "missing_threshold"
        elif result.outcome != "complete":
            reason = result.outcome
        elif result.score is None:
            reason = "missing_score"
        elif result.completed_at_monotonic_s >= result.identity.complete_deadline_monotonic_s:
            reason = "late"
        incomplete = self._completion_boundary_decision == "incomplete"
        verdict = ("incomplete" if incomplete else "complete") if reason == "model" else "fallback"
        logger.info(
            "[STT][Runtime] smart-turn decision segment=%s pause=%s verdict=%s "
            "score=%s threshold=%s action=%s target_pause_ms=%s reason=%s",
            self._segment_id,
            self._pause_id,
            verdict,
            result.score if result is not None else None,
            threshold,
            "wait" if incomplete else "seal",
            self.SMART_INCOMPLETE_MS if incomplete else self.SMART_COMPLETE_MS,
            reason,
        )

    def _arm_timers(self, segment_id: UUID, opened_at_s: float) -> None:
        self._cancel_timers()
        now = self._monotonic_clock()
        self._four_second_task = asyncio.create_task(
            self._run_pause_step_timer(
                segment_id,
                max(0.0, opened_at_s + self.FOUR_SECOND_AGE_S - now),
                pause_ms=self.FOUR_SECOND_PAUSE_MS,
            ),
            name="listen-four-second-step",
        )
        self._five_second_task = asyncio.create_task(
            self._run_pause_step_timer(
                segment_id,
                max(0.0, opened_at_s + self.FIVE_SECOND_AGE_S - now),
                pause_ms=self.FIVE_SECOND_PAUSE_MS,
            ),
            name="listen-five-second-step",
        )
        self._hard_task = asyncio.create_task(
            self._run_hard_timer(segment_id, max(0.0, opened_at_s + self.HARD_LIMIT_S - now)),
            name="listen-deadline",
        )

    async def _run_pause_step_timer(
        self,
        segment_id: UUID,
        delay_s: float,
        *,
        pause_ms: int,
    ) -> None:
        await asyncio.sleep(delay_s)
        if (
            not self._closed
            and segment_id == self._segment_id
            and self._observed_pause_ms() >= pause_ms
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
        opened_at_s = self._opened_at_s
        if (
            reason != "delivery_deadline"
            and opened_at_s is not None
            and self._monotonic_clock() >= opened_at_s + self.HARD_LIMIT_S
        ):
            reason = "delivery_deadline"
            rollover = True
        if segment_id != self._segment_id:
            return False
        method_name = "seal_active_for_rollover" if rollover else "seal_active"
        seal = getattr(self._vad, method_name, None)
        if not callable(seal):
            raise RuntimeError(f"peer VAD does not support {method_name}")
        event = seal(reason=reason)
        if event is None:
            return False
        snapshot = self._current_snapshot(segment_id)
        if snapshot is not None and snapshot.settings.delivery_profile_effective == "on":
            logger.info(
                "[STT][Runtime] smart-turn segment sealed segment=%s pause=%s "
                "reason=%s rollover=%s observed_pause_ms=%s",
                segment_id,
                self._pause_id,
                reason,
                rollover,
                self._observed_pause_ms(),
            )
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
        self._completion_boundary_decision = "none"

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
            for task in (
                self._four_second_task,
                self._five_second_task,
                self._hard_task,
            )
            if task is not None and task is not current
        )
        self._four_second_task = None
        self._five_second_task = None
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
