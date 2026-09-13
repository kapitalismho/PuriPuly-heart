from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest

from puripuly_heart.core.audio.format import AudioCaptureSpan
from puripuly_heart.core.audio.listen_delivery import ListenDeliveryController
from puripuly_heart.core.audio.ownership import AudioSegmentSettingsSnapshot, PeerAudioSegmentLedger
from puripuly_heart.core.audio.smart_turn import (
    SMART_TURN_COMPLETE_THRESHOLD,
    SmartTurnCompletion,
    smart_turn_language_profile,
)
from puripuly_heart.core.vad.gating import SpeechChunk, SpeechEnd, SpeechStart


@dataclass
class Clock:
    value: float = 0.0

    def __call__(self) -> float:
        return self.value


class Vad:
    def __init__(self) -> None:
        self.segment_id = None
        self.ends: list[SpeechEnd] = []

    def open(self, segment_id) -> None:
        self.segment_id = segment_id

    def seal_active(self, *, reason: str):
        return self._seal(reason)

    def seal_active_for_rollover(self, *, reason: str):
        return self._seal(reason)

    def _seal(self, reason: str):
        if self.segment_id is None:
            return None
        event = SpeechEnd(self.segment_id, reason=reason)
        self.segment_id = None
        self.ends.append(event)
        return event


class InferenceOwner:
    def __init__(self, statuses: list[str] | None = None) -> None:
        self.statuses = statuses or ["started"]
        self.requests = []
        self.audio = []
        self.callbacks = []
        self.prepare_count = 0
        self.late_count = 0
        self.closed = False
        self.snapshot = SimpleNamespace(availability="ready")

    def request_prepare(self) -> None:
        self.prepare_count += 1

    def submit(self, identity, audio, completion):
        status = self.statuses[min(len(self.requests), len(self.statuses) - 1)]
        self.requests.append(identity)
        self.audio.append(audio.copy())
        self.callbacks.append(completion)
        return status

    def record_late(self) -> None:
        self.late_count += 1

    async def close(self) -> None:
        self.closed = True


class Harness:
    def __init__(
        self,
        *,
        profile: str = "on",
        requested: str = "on",
        threshold: float | None = SMART_TURN_COMPLETE_THRESHOLD,
        hangover_ms: int = 500,
        inference: InferenceOwner | None = None,
        source_language: str = "en",
    ) -> None:
        self.clock = Clock()
        self.vad = Vad()
        self.inference = inference or InferenceOwner()
        self.events = []
        settings = AudioSegmentSettingsSnapshot(
            provider_id="test",
            provider_signature=("test",),
            runtime_signature=("test",),
            source_mode="manual",
            source_language=source_language,
            expected_languages=(),
            target_sample_rate_hz=16000,
            vad_speech_threshold=0.5,
            vad_hangover_ms=hangover_ms,
            vad_pre_roll_ms=0,
            delivery_profile_requested=requested,
            delivery_profile_effective=profile,
            delivery_availability=self.inference.snapshot.availability,
            delivery_threshold=threshold,
        )
        self.ledger = PeerAudioSegmentLedger(activation_generation=7, settings=settings)
        self.controller = ListenDeliveryController(
            vad=self.vad,
            ledger=self.ledger,
            emit=self._emit,
            monotonic_clock=self.clock,
            smart_turn_owner=self.inference,
        )
        self.sample = 0
        self.sequence = 0

    async def _emit(self, event) -> None:
        self.events.append(event)

    async def open(self, *, genuine: bool = True, value: float = 1.0):
        segment_id = uuid4()
        self.vad.open(segment_id)
        capture = self._span(32)
        chunk = np.full(512, value, dtype=np.float32)
        await self.controller.handle_vad_event(
            SpeechStart(
                segment_id,
                pre_roll=np.empty(0, dtype=np.float32),
                chunk=chunk,
                chunk_capture=(capture,),
                genuine_onset=genuine,
            )
        )
        await self.controller.observe_acoustic_chunk(speech_observed=True, capture=(capture,))
        return segment_id

    async def feed(self, milliseconds: int, *, speech: bool, value: float = 0.0) -> None:
        assert milliseconds % 32 == 0
        await self.feed_chunks(
            [32] * (milliseconds // 32),
            speech=speech,
            value=value,
        )

    async def feed_chunks(
        self,
        durations_ms: list[int],
        *,
        speech: bool,
        value: float = 0.0,
    ) -> None:
        for milliseconds in durations_ms:
            if self.controller.current_segment_id is None:
                return
            capture = self._span(milliseconds)
            chunk = np.full(milliseconds * 16, value, dtype=np.float32)
            await self.controller.handle_vad_event(
                SpeechChunk(
                    self.controller.current_segment_id,
                    chunk,
                    chunk_capture=(capture,),
                )
            )
            await self.controller.observe_acoustic_chunk(
                speech_observed=speech,
                capture=(capture,),
            )

    async def complete(self, index: int, *, score: float | None, at: float, outcome="complete"):
        self.clock.value = at
        await self.inference.callbacks[index](
            SmartTurnCompletion(
                identity=self.inference.requests[index],
                score=score,
                completed_at_monotonic_s=at,
                duration_s=0.01,
                outcome=outcome,
            )
        )

    def _span(self, milliseconds: int) -> AudioCaptureSpan:
        count = milliseconds * 16
        start = self.sample
        end = start + count
        start_s = start / 16000
        end_s = end / 16000
        self.sample = end
        self.sequence += 1
        self.clock.value = end_s
        return AudioCaptureSpan(
            capture_epoch=1,
            callback_sequence=self.sequence,
            source_sample_rate_hz=16000,
            source_start_sample=start,
            source_end_sample=end,
            source_start_monotonic_s=start_s,
            source_end_monotonic_s=end_s,
            normalized_sample_rate_hz=16000,
            normalized_start_sample=start,
            normalized_end_sample=end,
        )


@pytest.mark.asyncio
async def test_off_never_prepares_or_executes_and_uses_persisted_hangover() -> None:
    harness = Harness(profile="off", requested="off", threshold=None, hangover_ms=480)
    await harness.open()
    await harness.feed(448, speech=False)
    assert not harness.vad.ends
    await harness.feed(32, speech=False)
    assert len(harness.vad.ends) == 1
    assert harness.inference.prepare_count == 0
    assert harness.inference.requests == []


@pytest.mark.asyncio
async def test_unsupported_language_never_prepares_or_infers() -> None:
    harness = Harness(
        profile="unsupported_language",
        requested="on",
        threshold=None,
        hangover_ms=480,
    )
    await harness.open()
    await harness.feed(480, speech=False)
    assert len(harness.vad.ends) == 1
    assert harness.inference.prepare_count == 0
    assert harness.inference.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["unavailable", "busy", "started"])
async def test_unavailable_busy_or_pending_inference_seals_at_512_without_capture_wait(
    status: str,
) -> None:
    harness = Harness(inference=InferenceOwner([status]))
    await harness.open()
    await harness.feed(480, speech=False)
    assert not harness.vad.ends
    await harness.feed(32, speech=False)
    assert len(harness.vad.ends) == 1
    assert harness.ledger.snapshots[0].content_sample_count == (32 + 512) * 16
    assert len(harness.inference.requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("completion_offset", "sealed_at_512"),
    [(0.511999, False), (0.512, True), (0.513, True)],
)
async def test_incomplete_evidence_deadline_is_strict(
    completion_offset: float,
    sealed_at_512: bool,
) -> None:
    harness = Harness()
    await harness.open()
    await harness.feed(224, speech=False)
    request = harness.inference.requests[0]
    await harness.complete(
        0,
        score=0.2,
        at=request.complete_deadline_monotonic_s - 0.512 + completion_offset,
    )
    await harness.feed(288, speech=False)
    assert bool(harness.vad.ends) is sealed_at_512
    if not sealed_at_512:
        await harness.feed(288, speech=False)
        assert len(harness.vad.ends) == 1
        assert harness.ledger.snapshots[0].content_sample_count == (32 + 800) * 16


@pytest.mark.asyncio
async def test_stamp_timely_result_delivered_after_512_cannot_reopen_sealed_segment() -> None:
    harness = Harness()
    await harness.open()
    await harness.feed(224, speech=False)
    request = harness.inference.requests[0]
    await harness.feed(288, speech=False)
    assert len(harness.vad.ends) == 1

    await harness.complete(
        0,
        score=0.2,
        at=request.complete_deadline_monotonic_s - 0.1,
    )
    await harness.feed(288, speech=False)
    assert len(harness.vad.ends) == 1
    assert harness.ledger.snapshots[0].content_sample_count == (32 + 512) * 16


@pytest.mark.asyncio
async def test_nonuniform_frames_clip_probe_input_to_exact_224_frontier() -> None:
    harness = Harness()
    await harness.open(value=0.25)
    await harness.feed_chunks([30] * 7 + [20], speech=False)

    request = harness.inference.requests[0]
    assert request.source_frontier == (32 + 224) * 16
    assert harness.inference.audio[0].size == (32 + 224) * 16
    np.testing.assert_array_equal(harness.inference.audio[0][: 32 * 16], 0.25)
    np.testing.assert_array_equal(harness.inference.audio[0][32 * 16 :], 0.0)


@pytest.mark.asyncio
async def test_hard_timer_seals_actual_owned_range_while_model_is_pending() -> None:
    harness = Harness()
    harness.controller.HARD_LIMIT_S = 0.3
    await harness.open()
    await harness.feed(224, speech=False)
    assert len(harness.inference.requests) == 1

    async with asyncio.timeout(1.0):
        while not harness.vad.ends:
            await asyncio.sleep(0.001)
    snapshot = harness.ledger.snapshots[0]
    assert snapshot.seal_reason == "delivery_deadline"
    assert snapshot.content_sample_count == (32 + 224) * 16
    second_id = await harness.open(genuine=False, value=2.0)
    await harness.feed(224, speech=False)
    assert len(harness.inference.requests) == 2
    assert harness.inference.requests[1].segment_id == second_id
    np.testing.assert_array_equal(harness.inference.audio[1][:512], np.ones(512, dtype=np.float32))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("outcome", "score", "pause_ms"),
    [("complete", 0.2, 800), ("error", None, 512), ("nonfinite", float("nan"), 512)],
)
async def test_only_valid_incomplete_result_extends_pause_beyond_512(
    outcome: str, score: float | None, pause_ms: int
) -> None:
    harness = Harness()
    await harness.open()
    await harness.feed(224, speech=False)
    request = harness.inference.requests[0]
    await harness.complete(
        0,
        score=score,
        at=request.complete_deadline_monotonic_s - 0.1,
        outcome=outcome,
    )
    await harness.feed(pause_ms - 224 - 32, speech=False)
    assert not harness.vad.ends
    await harness.feed(32, speech=False)
    assert len(harness.vad.ends) == 1
    assert harness.ledger.snapshots[0].content_sample_count == (32 + pause_ms) * 16


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "language",
    (
        "ar",
        "zh",
        "da",
        "nl",
        "de",
        "en",
        "fi",
        "fr",
        "hi",
        "id",
        "it",
        "ja",
        "ko",
        "no",
        "pl",
        "pt",
        "ru",
        "es",
        "tr",
        "uk",
        "vi",
    ),
)
@pytest.mark.parametrize(
    ("score", "sealed_at_512"),
    (
        (SMART_TURN_COMPLETE_THRESHOLD, True),
        (math.nextafter(SMART_TURN_COMPLETE_THRESHOLD, 0.0), False),
    ),
)
async def test_common_threshold_equal_qualifies_and_below_falls_back(
    language: str,
    score: float,
    sealed_at_512: bool,
) -> None:
    profile, threshold = smart_turn_language_profile("manual", language)
    assert profile == "on"
    assert threshold == SMART_TURN_COMPLETE_THRESHOLD
    harness = Harness(threshold=threshold, source_language=language)
    await harness.open()
    await harness.feed(224, speech=False)
    request = harness.inference.requests[0]
    await harness.complete(
        0,
        score=score,
        at=request.complete_deadline_monotonic_s - 0.1,
    )
    await harness.feed(288, speech=False)
    assert bool(harness.vad.ends) is sealed_at_512
    if not sealed_at_512:
        await harness.feed(288, speech=False)
        assert len(harness.vad.ends) == 1
        assert harness.ledger.snapshots[0].content_sample_count == (32 + 800) * 16


@pytest.mark.asyncio
async def test_resumption_creates_new_pause_and_busy_worker_does_not_queue_or_reuse() -> None:
    harness = Harness(inference=InferenceOwner(["started", "busy"]))
    await harness.open()
    await harness.feed(224, speech=False)
    first = harness.inference.requests[0]
    await harness.feed(32, speech=True, value=1.0)
    await harness.complete(0, score=0.2, at=first.complete_deadline_monotonic_s - 0.1)
    await harness.feed(224, speech=False)
    assert len(harness.inference.requests) == 2
    assert harness.inference.requests[1].pause_id != first.pause_id
    await harness.feed(288, speech=False)
    assert len(harness.vad.ends) == 1
    assert harness.ledger.snapshots[0].content_sample_count == (32 + 224 + 32 + 512) * 16


@pytest.mark.asyncio
async def test_age_step_preserves_existing_pause_and_revokes_model_authority() -> None:
    harness = Harness()
    await harness.open()
    await harness.feed(3648, speech=True, value=1.0)
    await harness.feed(224, speech=False)
    assert len(harness.inference.requests) == 1
    await harness.feed(96, speech=False)
    assert len(harness.inference.requests) == 1
    assert len(harness.vad.ends) == 1
    request = harness.inference.requests[0]
    await harness.complete(0, score=0.99, at=request.complete_deadline_monotonic_s - 0.1)
    assert len(harness.vad.ends) == 1


@pytest.mark.asyncio
async def test_six_second_step_preserves_pause_support_without_forcing_speech_cut() -> None:
    continuous = Harness(profile="off", requested="off", threshold=None)
    await continuous.open()
    await continuous.feed(5984, speech=True, value=1.0)
    assert continuous.clock.value > 6.0
    assert continuous.vad.ends == []

    await continuous.feed(96, speech=False)
    assert continuous.vad.ends == []
    await continuous.feed(32, speech=False)
    assert len(continuous.vad.ends) == 1
    assert continuous.vad.ends[0].reason == "delivery_pause"

    accumulated = Harness(profile="off", requested="off", threshold=None)
    await accumulated.open()
    await accumulated.feed(5824, speech=True, value=1.0)
    await accumulated.feed(160, speech=False)
    assert accumulated.clock.value > 6.0
    assert len(accumulated.vad.ends) == 1
    assert accumulated.ledger.snapshots[0].seal_reason == "delivery_pause"


@pytest.mark.asyncio
async def test_no_callback_steps_add_no_silence_and_hard_timer_seals_actual_frontier() -> None:
    harness = Harness(profile="off", requested="off", threshold=None)
    harness.controller.FOUR_SECOND_AGE_S = 0.05
    harness.controller.SIX_SECOND_AGE_S = 0.15
    harness.controller.HARD_LIMIT_S = 0.3
    await harness.open()
    await harness.feed(96, speech=False)
    actual_frontier_samples = (32 + 96) * 16

    await asyncio.sleep(0.18)
    assert harness.vad.ends == []
    assert harness.ledger.snapshots[0].content_sample_count == actual_frontier_samples

    async with asyncio.timeout(1.0):
        while not harness.vad.ends:
            await asyncio.sleep(0.001)
    snapshot = harness.ledger.snapshots[0]
    assert snapshot.seal_reason == "delivery_deadline"
    assert snapshot.content_sample_count == actual_frontier_samples


@pytest.mark.asyncio
async def test_six_second_timer_reevaluates_existing_160ms_pause_without_callback() -> None:
    harness = Harness(profile="off", requested="off", threshold=None)
    harness.controller.FOUR_SECOND_AGE_S = 0.05
    harness.controller.SIX_SECOND_AGE_S = 0.25
    harness.controller.HARD_LIMIT_S = 0.5
    await harness.open()
    await harness.feed(160, speech=False)
    assert harness.vad.ends == []

    async with asyncio.timeout(1.0):
        while not harness.vad.ends:
            await asyncio.sleep(0.001)
    assert harness.vad.ends[0].reason == "delivery_pause"
    assert harness.ledger.snapshots[0].content_sample_count == (32 + 160) * 16


@pytest.mark.asyncio
async def test_hard_boundary_wins_simultaneous_pause_decision_once() -> None:
    harness = Harness(profile="off", requested="off", threshold=None)
    harness.controller.HARD_LIMIT_S = 0.16
    harness.controller.SIX_SECOND_AGE_S = 0.1
    await harness.open()
    await harness.feed(128, speech=False)

    assert len(harness.vad.ends) == 1
    assert harness.vad.ends[0].reason == "delivery_deadline"
    assert harness.ledger.snapshots[0].seal_reason == "delivery_deadline"


@pytest.mark.asyncio
async def test_settings_are_snapshotted_per_segment_without_second_old_pause_probe() -> None:
    harness = Harness(profile="off", requested="off", threshold=None, hangover_ms=800)
    await harness.open()
    await harness.feed(224, speech=False)
    next_settings = AudioSegmentSettingsSnapshot(
        provider_id="test",
        provider_signature=("test",),
        runtime_signature=("on",),
        source_mode="manual",
        source_language="en",
        expected_languages=(),
        target_sample_rate_hz=16000,
        vad_speech_threshold=0.5,
        vad_hangover_ms=100,
        vad_pre_roll_ms=0,
        delivery_profile_requested="on",
        delivery_profile_effective="on",
        delivery_availability="ready",
        delivery_threshold=SMART_TURN_COMPLETE_THRESHOLD,
    )
    harness.ledger.rebind(activation_generation=7, settings=next_settings)
    await harness.feed(288, speech=False)
    assert harness.inference.requests == []
    assert not harness.vad.ends
    await harness.feed(288, speech=False)
    assert len(harness.vad.ends) == 1
    await harness.open()
    await harness.feed(224, speech=False)
    assert len(harness.inference.requests) == 1


@pytest.mark.asyncio
async def test_retirement_during_inference_rejects_late_completion() -> None:
    harness = Harness()
    await harness.open()
    await harness.feed(224, speech=False)
    request = harness.inference.requests[0]
    harness.controller.cancel()
    await harness.complete(0, score=0.99, at=request.complete_deadline_monotonic_s - 0.1)
    assert not harness.vad.ends
    assert harness.inference.closed is False
