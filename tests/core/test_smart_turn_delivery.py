from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest

from puripuly_heart.core.audio.format import AudioCaptureSpan
from puripuly_heart.core.audio.listen_delivery import ListenDeliveryController
from puripuly_heart.core.audio.ownership import AudioSegmentSettingsSnapshot, PeerAudioSegmentLedger
from puripuly_heart.core.audio.smart_turn import SmartTurnCompletion
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
        threshold: float | None = 0.5,
        hangover_ms: int = 500,
        inference: InferenceOwner | None = None,
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
            source_language="en",
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
        for _ in range(milliseconds // 32):
            if self.controller.current_segment_id is None:
                return
            capture = self._span(32)
            chunk = np.full(512, value, dtype=np.float32)
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
@pytest.mark.parametrize("profile", ["unsupported_auto", "unsupported_language"])
async def test_unsupported_profiles_never_prepare_or_infer(profile: str) -> None:
    harness = Harness(profile=profile, requested="on", threshold=None, hangover_ms=480)
    await harness.open()
    await harness.feed(480, speech=False)
    assert len(harness.vad.ends) == 1
    assert harness.inference.prepare_count == 0
    assert harness.inference.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["unavailable", "busy"])
async def test_supported_on_missing_loading_or_busy_uses_800_without_capture_wait(
    status: str,
) -> None:
    harness = Harness(inference=InferenceOwner([status]))
    await harness.open()
    await harness.feed(512, speech=False)
    assert not harness.vad.ends
    await harness.feed(288, speech=False)
    assert len(harness.vad.ends) == 1
    assert len(harness.inference.requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("completion_offset", "sealed_at_512"),
    [(0.511999, True), (0.512, False), (0.513, False)],
)
async def test_complete_evidence_deadline_is_strict(
    completion_offset: float,
    sealed_at_512: bool,
) -> None:
    harness = Harness()
    await harness.open()
    await harness.feed(224, speech=False)
    request = harness.inference.requests[0]
    await harness.complete(
        0,
        score=0.9,
        at=request.complete_deadline_monotonic_s - 0.512 + completion_offset,
    )
    await harness.feed(288, speech=False)
    assert bool(harness.vad.ends) is sealed_at_512
    if not sealed_at_512:
        await harness.feed(288, speech=False)
        assert len(harness.vad.ends) == 1
        assert harness.inference.late_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("outcome", "score"),
    [("complete", 0.2), ("error", None), ("nonfinite", float("nan"))],
)
async def test_incomplete_error_and_nonfinite_use_exact_fallback(
    outcome: str, score: float | None
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
    await harness.feed(576, speech=False)
    assert len(harness.vad.ends) == 1


@pytest.mark.asyncio
async def test_resumption_creates_new_pause_and_busy_worker_does_not_queue_or_reuse() -> None:
    harness = Harness(inference=InferenceOwner(["started", "busy"]))
    await harness.open()
    await harness.feed(224, speech=False)
    first = harness.inference.requests[0]
    await harness.feed(32, speech=True, value=1.0)
    await harness.complete(0, score=0.99, at=first.complete_deadline_monotonic_s - 0.1)
    await harness.feed(224, speech=False)
    assert len(harness.inference.requests) == 2
    assert harness.inference.requests[1].pause_id != first.pause_id
    await harness.feed(288, speech=False)
    assert not harness.vad.ends
    await harness.feed(288, speech=False)
    assert len(harness.vad.ends) == 1


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
async def test_natural_endpoint_resets_context_but_synthetic_rollover_retains_it() -> None:
    synthetic = Harness(inference=InferenceOwner(["started", "started"]))
    await synthetic.open(value=0.25)
    await synthetic.feed(224, speech=False)
    first_size = synthetic.inference.audio[0].size
    sealed = await synthetic.controller.seal_prospective_transition(
        capture_epoch=1,
        requested_source_sample=1,
    )
    assert sealed[0] == "sealed"
    await synthetic.open(genuine=False, value=0.5)
    await synthetic.feed(224, speech=False)
    assert synthetic.inference.audio[1].size > first_size

    natural = Harness(inference=InferenceOwner(["started", "started"]))
    await natural.open(value=0.25)
    await natural.feed(224, speech=False)
    first_size = natural.inference.audio[0].size
    request = natural.inference.requests[0]
    await natural.complete(0, score=0.99, at=request.complete_deadline_monotonic_s - 0.1)
    await natural.feed(288, speech=False)
    await natural.open(genuine=True, value=0.5)
    await natural.feed(224, speech=False)
    assert natural.inference.audio[1].size == first_size


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
        delivery_threshold=0.5,
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


@pytest.mark.asyncio
async def test_simultaneous_psem_seal_and_model_completion_produce_at_most_one_seal() -> None:
    harness = Harness()
    segment_id = await harness.open()
    await harness.feed(224, speech=False)
    request = harness.inference.requests[0]
    result = await harness.controller.seal_prospective_transition(
        capture_epoch=1,
        requested_source_sample=1,
    )
    await harness.complete(0, score=0.99, at=request.complete_deadline_monotonic_s - 0.1)
    assert result[0] == "sealed"
    assert result[2] == segment_id
    assert len(harness.vad.ends) == 1
