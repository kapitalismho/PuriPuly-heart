from __future__ import annotations

import asyncio
from dataclasses import dataclass, field, replace
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest

from puripuly_heart.core.audio.ownership import AudioSegmentSettingsSnapshot, PeerAudioSegmentLedger
from puripuly_heart.core.clock import SystemClock
from puripuly_heart.core.orchestrator.translation_channel_callbacks import (
    TranslationChannelOwnerCallbacks,
)
from puripuly_heart.core.osc.chatbox_paginator import ChatboxPaginator
from puripuly_heart.core.stt.backend import (
    STTProviderInputTerminal,
    STTProviderTurnIdentity,
    STTProviderTurnTerminal,
    STTProviderTurnUpdate,
    STTRecognitionUnit,
    STTRecognitionUnitTerminal,
    STTTextContribution,
)
from puripuly_heart.core.vad.gating import SpeechChunk, SpeechEnd, SpeechStart
from puripuly_heart.domain.recognition import RecognitionStreamIdentity, RecognitionUnitIdentity
from tests.helpers.self_stream import Provider, Sender, until
from tests.helpers.translation_owners import compose_translation_test_harness


@dataclass
class ControlledRecognitionProvider:
    scoped_settings_scope = ("synthetic_overlap", ("bounded_overlap",), ("current",))
    provider_name = "synthetic_overlap"
    received: list[object] = field(default_factory=list)
    closed: bool = False

    def is_current_recognition_stream(self, stream):
        return not self.closed and stream.settings_scope == self.scoped_settings_scope

    async def handle_owned_vad_event(self, event):
        self.received.append(event)

    async def close(self):
        self.closed = True


@dataclass
class OwnedIngress:
    harness: object
    clock: SystemClock = field(default_factory=SystemClock)
    ledger: PeerAudioSegmentLedger = field(init=False)

    def __post_init__(self):
        self.ledger = PeerAudioSegmentLedger(
            activation_generation=1,
            settings=AudioSegmentSettingsSnapshot(
                provider_id="synthetic_overlap",
                provider_signature=("bounded_overlap",),
                runtime_signature=("current",),
                source_mode="manual",
                source_language="en",
                expected_languages=("en",),
                target_sample_rate_hz=16000,
                vad_speech_threshold=0.6,
                vad_hangover_ms=900,
                vad_pre_roll_ms=500,
            ),
        )

    async def vad(self, event):
        owned = self.ledger.observe_vad_event(event, now_monotonic_s=self.clock.now())
        await self.harness.self_owner.handle_vad_event(owned)
        settings = owned.segment.settings
        return STTProviderTurnIdentity(
            owned.segment.identity,
            "current-overlap-epoch",
            str(event.utterance_id),
            (settings.provider_id, settings.provider_signature, settings.runtime_signature),
        )

    async def start(self, *, confirmed=True):
        identity = uuid4()
        request = await self.vad(
            SpeechStart(
                identity,
                pre_roll=np.empty(0, dtype=np.float32),
                chunk=np.ones(512, dtype=np.float32),
            )
        )
        if confirmed:
            for _ in range(3):
                await self.vad(SpeechChunk(identity, chunk=np.ones(512, dtype=np.float32)))
        return identity, request

    async def terminal(self, identity, outcome, text=""):
        await self.harness.self_owner.handle_stt_event(
            STTProviderTurnTerminal(
                identity,
                outcome,
                text=text,
                text_authority="authoritative" if text else "none",
                failure_reason="synthetic_terminal" if outcome in {"failed", "expired"} else None,
            )
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["active_prebuffer", "ended_then_false_start"])
@pytest.mark.parametrize("settlement", ["empty", "failed", "cancelled", "expired", "watchdog"])
async def test_owned_delayed_a_settles_only_a_and_matching_b_settlement_progresses(
    phase, settlement
):
    sender = Sender()
    clock = SystemClock()
    recognition = ControlledRecognitionProvider()
    harness = compose_translation_test_harness(
        stt=recognition,
        llm=None,
        osc=ChatboxPaginator(sender=sender, clock=clock),
        clock=clock,
        low_latency_mode=True,
        low_latency_finalize_wait_ms=0,
        low_latency_awaiting_vad_timeout_s=0.1,
        self_target_languages=("en",),
    )
    ingress = OwnedIngress(harness, clock)
    try:
        first, request_a = await ingress.start()
        await ingress.vad(SpeechEnd(first))
        successor, request_b = await ingress.start()
        false_start = None
        if phase == "ended_then_false_start":
            await ingress.vad(SpeechEnd(successor))
            false_start, _ = await ingress.start(confirmed=False)
        await ingress.terminal(request_a, "final", "retained alpha")
        await asyncio.sleep(0.005)
        assert sender.packets == []
        if false_start is not None:
            await ingress.vad(SpeechEnd(false_start))
            await asyncio.sleep(0.005)
            assert sender.packets == []
        if settlement == "watchdog":
            if phase == "active_prebuffer":
                await ingress.vad(SpeechEnd(successor))
        else:
            await ingress.terminal(request_b, settlement)
        await until(lambda: sender.packets == ["retained alpha"])
        await asyncio.wait_for(harness.translation_turns.wait_for_idle(), 1)
        assert sender.packets == ["retained alpha"]
        assert harness.self_runtime.translation_history == []
        assert any(event.segment.identity == request_b.segment for event in recognition.received)
    finally:
        await harness.stop()


@pytest.mark.asyncio
async def test_owned_stable_prefix_watchdog_preview_keeps_open_request_suffix_and_actual_terminal():
    sender = Sender()
    provider = Provider()
    recognition = ControlledRecognitionProvider()
    clock = SystemClock()
    harness = compose_translation_test_harness(
        stt=recognition,
        llm=provider,
        osc=ChatboxPaginator(sender=sender, clock=clock),
        clock=clock,
        low_latency_mode=True,
        low_latency_finalize_wait_ms=10,
        low_latency_awaiting_vad_timeout_s=0.04,
        self_target_languages=("en",),
        chatbox_include_source=False,
    )
    ingress = OwnedIngress(harness, clock)
    try:
        acoustic, request = await ingress.start()
        prefix = STTTextContribution("prefix", 0, 12)
        await harness.self_owner.handle_stt_event(
            STTProviderTurnUpdate(
                request,
                1,
                "stable",
                "append",
                "first prefix",
                contribution=prefix,
            )
        )
        provider.release("first prefix")
        await until(lambda: sender.packets == ["en:first prefix"])
        assert harness.self_owner.merge_buffer is not None
        await until(lambda: harness.self_owner.merge_buffer is None)
        await asyncio.wait_for(harness.translation_turns.wait_for_idle(), 1)
        suffix = STTTextContribution("suffix", 13, 26)
        await harness.self_owner.handle_stt_event(
            STTProviderTurnUpdate(
                request,
                2,
                "stable",
                "append",
                "first prefix second suffix",
                contribution=suffix,
            )
        )
        await until(lambda: len(provider.calls) == 2)
        assert [call["text"] for call in provider.calls] == ["first prefix", "second suffix"]
        provider.release("second suffix")
        await ingress.vad(SpeechEnd(acoustic))
        await harness.self_owner.handle_stt_event(
            STTProviderTurnTerminal(
                request,
                "final",
                text="first prefix second suffix",
                text_authority="authoritative",
                included_contributions=(prefix, suffix),
            )
        )
        await asyncio.wait_for(harness.translation_turns.wait_for_idle(), 1)
        assert sender.packets == ["en:first prefix", "en:second suffix"]
        assert len(provider.calls) == 2
    finally:
        provider.release("first prefix")
        provider.release("second suffix")
        await harness.stop()


@pytest.mark.asyncio
async def test_exact_logical_source_retirement_preserves_other_active_obligations():
    sender = Sender()
    clock = SystemClock()
    harness = compose_translation_test_harness(
        stt=ControlledRecognitionProvider(),
        llm=None,
        osc=ChatboxPaginator(sender=sender, clock=clock),
        clock=clock,
        low_latency_mode=True,
        low_latency_finalize_wait_ms=10,
        self_target_languages=("en",),
    )
    earlier = OwnedIngress(harness, clock)
    other = OwnedIngress(harness, clock)
    try:
        first, request_a = await earlier.start()
        await earlier.vad(SpeechEnd(first))
        _, unadmitted = await earlier.start(confirmed=False)
        _, active = await other.start()
        await earlier.terminal(request_a, "final", "retained alpha")
        receipt = next(
            receipt
            for receipt in earlier.ledger.cancel_unfinished(now_monotonic_s=clock.now())
            if receipt.identity == unadmitted.segment
        )
        await harness.self_owner.handle_source_segment_retired(
            replace(receipt, identity=replace(receipt.identity, capture_epoch=99))
        )
        await asyncio.sleep(0.02)
        assert sender.packets == []
        await harness.self_owner.handle_source_segment_retired(receipt)
        await asyncio.sleep(0.02)
        assert sender.packets == []
        await other.terminal(active, "empty")
        await until(lambda: sender.packets == ["retained alpha"])
        await harness.translation_turns.wait_for_idle()
        assert sender.packets == ["retained alpha"]
    finally:
        await harness.stop()


@pytest.mark.asyncio
async def test_submitted_pcm_input_does_not_complete_scoped_source_obligation():
    sender = Sender()
    clock = SystemClock()
    harness = compose_translation_test_harness(
        stt=ControlledRecognitionProvider(),
        llm=None,
        osc=ChatboxPaginator(sender=sender, clock=clock),
        clock=clock,
        low_latency_mode=True,
        low_latency_finalize_wait_ms=10,
        low_latency_awaiting_vad_timeout_s=0.3,
        self_target_languages=("en",),
    )
    ingress = OwnedIngress(harness, clock)
    callbacks = TranslationChannelOwnerCallbacks(harness.stt_sessions)
    callbacks.bind_self(harness.self_owner)
    try:
        first, request_a = await ingress.start()
        await ingress.vad(SpeechEnd(first))
        second, request_b = await ingress.start()
        await ingress.vad(SpeechEnd(second))
        await ingress.terminal(request_a, "final", "retained alpha")
        await callbacks.self_event_handler(
            STTProviderInputTerminal(request_b, "submitted", channel="self")
        )
        await asyncio.sleep(0.03)
        assert sender.packets == []
        await ingress.terminal(request_b, "empty")
        await until(lambda: sender.packets == ["retained alpha"])
        await harness.translation_turns.wait_for_idle()
        assert sender.packets == ["retained alpha"]
    finally:
        await harness.stop()


@pytest.mark.asyncio
async def test_independent_submitted_before_end_repeats_then_scoped_source_handoff_fences_old_input():
    sender = Sender()
    clock = SystemClock()
    harness = compose_translation_test_harness(
        stt=ControlledRecognitionProvider(),
        llm=None,
        osc=ChatboxPaginator(sender=sender, clock=clock),
        clock=clock,
        low_latency_mode=True,
        low_latency_finalize_wait_ms=0,
        self_target_languages=("en",),
    )
    ingress = OwnedIngress(harness, clock)
    callbacks = TranslationChannelOwnerCallbacks(harness.stt_sessions)
    callbacks.bind_self(harness.self_owner)
    callbacks.bind_self_capture(
        SimpleNamespace(
            snapshot=SimpleNamespace(generation=1),
            note_input_terminal=lambda _event: None,
            is_current_recognition_stream=lambda stream: stream.activation_generation == 1,
        )
    )
    try:
        for order in range(3):
            segment_id, request = await ingress.start()
            await callbacks.self_event_handler(
                STTProviderInputTerminal(
                    request, "submitted", channel="self", independent_recognition=True
                )
            )
            stream = RecognitionStreamIdentity(
                "self",
                request.segment.activation_generation,
                request.segment.capture_epoch,
                request.provider_epoch_id,
                request.settings_scope,
            )
            await callbacks.self_event_handler(
                STTRecognitionUnitTerminal(
                    STTRecognitionUnit(
                        RecognitionUnitIdentity(stream, uuid4(), order + 1), f"independent {order}"
                    ),
                    "final",
                )
            )
            await ingress.vad(SpeechEnd(segment_id))
            await harness.translation_turns.wait_for_idle()
            await until(lambda: sender.packets == [f"independent {i}" for i in range(order + 1)])
        segment_id, request = await ingress.start()
        await callbacks.retired_event_handler(
            STTProviderInputTerminal(
                replace(request, segment=replace(request.segment, activation_generation=0)),
                "submitted",
                channel="self",
                independent_recognition=True,
            )
        )
        await ingress.vad(SpeechEnd(segment_id))
        await ingress.terminal(request, "final", "scoped successor")
        await until(
            lambda: sender.packets
            == ["independent 0", "independent 1", "independent 2", "scoped successor"]
        )
        await harness.translation_turns.wait_for_idle()
        assert [entry.text for entry in harness.self_runtime.translation_history] == []
    finally:
        await harness.stop()
