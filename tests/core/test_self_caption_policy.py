from __future__ import annotations

import asyncio
from uuid import uuid4

import numpy as np
import pytest

from puripuly_heart.config.overlay_calibration import OverlayCalibration
from puripuly_heart.core.audio.ownership import AudioSegmentIdentity
from puripuly_heart.core.clock import FakeClock
from puripuly_heart.core.llm.provider import LLMProvider
from puripuly_heart.core.osc.chatbox_paginator import ChatboxPaginator
from puripuly_heart.core.overlay.presenter import OverlayPresenter
from puripuly_heart.core.overlay.sink import OverlayEventAdapter
from puripuly_heart.core.stt.backend import (
    STTProviderTurnIdentity,
    STTProviderTurnTerminal,
    STTProviderTurnUpdate,
    STTTextContribution,
)
from puripuly_heart.core.vad.gating import SpeechEnd, SpeechStart
from puripuly_heart.domain.events import STTFinalEvent
from puripuly_heart.domain.models import Transcript, Translation
from tests.helpers.fakes import RecordingOscQueue
from tests.helpers.self_stream import Sender
from tests.helpers.translation_owners import compose_translation_test_harness


async def eventually(predicate):
    async with asyncio.timeout(2):
        while not predicate():
            await asyncio.sleep(0)


class HeldTranslations(LLMProvider):
    def __init__(self):
        self.calls = []
        self.responses = {}

    async def translate(self, **values):
        key = (values["text"], values["target_language"])
        self.calls.append(key)
        if key not in self.responses or self.responses[key].done():
            self.responses[key] = asyncio.get_running_loop().create_future()
        response = await self.responses[key]
        return Translation(
            utterance_id=values["utterance_id"],
            text=response,
            source_text=values["text"],
            source_language=values["source_language"],
            target_language=values["target_language"],
        )

    def finish(self, text, response, target="en"):
        self.responses[(text, target)].set_result(response)

    async def close(self):
        return


async def source(harness, source_id, text):
    await harness.self_owner.handle_stt_event(
        STTFinalEvent(source_id, Transcript(source_id, text, is_final=True))
    )


def visible(presenter):
    return [(block.primary_text, block.secondary_text) for block in presenter.snapshot().blocks]


class Scenes:
    def __init__(self):
        self.snapshots = []

    async def replace_snapshot(self, snapshot, **kwargs):
        self.snapshots.append(snapshot)


class SourceUpdates:
    def __init__(self, harness, route):
        self.harness = harness
        self.route = route
        self.uid = uuid4()
        self.identity = STTProviderTurnIdentity(
            AudioSegmentIdentity(1, 1, self.uid, 0), "epoch", "request"
        )
        self.contributions = []

    async def update(self, text):
        if self.route == "final_only":
            await source(self.harness, self.uid, text)
            return
        start = self.contributions[-1].text_end if self.contributions else 0
        contribution = STTTextContribution(str(len(self.contributions)), start, len(text))
        self.contributions.append(contribution)
        await self.harness.self_owner.handle_stt_event(
            STTProviderTurnUpdate(
                self.identity,
                len(self.contributions),
                "stable",
                "append",
                text,
                contribution=contribution,
            )
        )

    async def terminal(self, text):
        if self.route == "stable_append":
            await self.harness.self_owner.handle_stt_event(
                STTProviderTurnTerminal(
                    self.identity,
                    "final",
                    text=text,
                    text_authority="authoritative",
                    included_contributions=tuple(self.contributions),
                )
            )


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["stable_append", "final_only"])
@pytest.mark.parametrize("source_after_end", [False, True])
async def test_fast_results_and_post_end_token_bursts_do_not_define_visible_units(
    route, source_after_end
):
    provider, scenes, sender = HeldTranslations(), Scenes(), Sender()
    presenter = OverlayPresenter(
        calibration=OverlayCalibration(), bridge=scenes, native_retry_enabled=True
    )
    harness = compose_translation_test_harness(
        stt=None,
        llm=provider,
        osc=ChatboxPaginator(sender=sender, clock=FakeClock()),
        overlay_sink=presenter,
        low_latency_mode=True,
        low_latency_finalize_wait_ms=80,
        chatbox_include_source=False,
    )
    updates = SourceUpdates(harness, route)
    try:
        if source_after_end:
            await harness.self_owner.handle_vad_event(SpeechEnd(updates.uid))
        for index, text in enumerate(("alpha", "alpha beta", "alpha beta gamma"), 1):
            await updates.update(text)
            await eventually(lambda: (text, "en") in provider.responses)
            provider.finish(text, f"translation {index}")
            await eventually(lambda: f"translation {index}" in sender.packets)
            assert visible(presenter) == []
        await updates.terminal("alpha beta gamma")
        if not source_after_end:
            await harness.self_owner.handle_vad_event(SpeechEnd(updates.uid))
        assert visible(presenter) == []
        await eventually(lambda: harness.self_owner.merge_buffer is None)
        await harness.translation_turns.wait_for_idle()
        assert visible(presenter) == [("alpha beta gamma", "translation 3")]
        assert sender.packets == ["translation 1", "translation 2", "translation 3"]
        translated_scenes = [
            block.secondary_text
            for snapshot in scenes.snapshots
            for block in snapshot.blocks
            if block.secondary_text
        ]
        assert translated_scenes == ["translation 3"]
    finally:
        await harness.stop()
        await presenter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "first,second",
    [
        ("A complete sentence.", "A complete sentence. Another complete sentence!"),
        ("첫 문장입니다.", "첫 문장입니다. 다음 문장입니다."),
        ("最初の文です。", "最初の文です。次の文です。"),
        ("这是第一句话。", "这是第一句话。这是下一句话！"),
    ],
)
async def test_open_sentence_coverage_advances_atomically_without_duplicate_prefix_slots(
    first, second
):
    provider, osc = HeldTranslations(), RecordingOscQueue()
    presenter = OverlayPresenter(calibration=OverlayCalibration(), native_retry_enabled=True)
    harness = compose_translation_test_harness(
        stt=None,
        llm=provider,
        osc=osc,
        overlay_sink=presenter,
        low_latency_mode=True,
        low_latency_finalize_wait_ms=0,
    )
    uid = uuid4()
    try:
        await source(harness, uid, first)
        await eventually(lambda: (first, "en") in provider.responses)
        provider.finish(first, "first readable unit")
        await eventually(lambda: visible(presenter) == [(first, "first readable unit")])
        original = presenter.snapshot().blocks[0]
        episode = presenter.snapshot().native_quiet_tail_episodes.self
        middle = first + " unfinished"
        await source(harness, uid, middle)
        await eventually(lambda: (middle, "en") in provider.responses)
        provider.finish(middle, "intermediate result")
        await eventually(lambda: any("intermediate result" in m.text for m in osc.messages))
        assert visible(presenter) == [(first, "first readable unit")]
        assert presenter.snapshot().native_quiet_tail_episodes.self == episode
        await source(harness, uid, second)
        await eventually(lambda: (second, "en") in provider.responses)
        assert visible(presenter) == [(first, "first readable unit")]
        provider.finish(second, "second readable unit")
        await eventually(lambda: visible(presenter) == [(second, "second readable unit")])
        current = presenter.snapshot().blocks[0]
        assert (current.id, current.occupant_key, current.appearance_seq) == (
            original.id,
            original.occupant_key,
            original.appearance_seq,
        )
        await harness.self_owner.handle_vad_event(SpeechEnd(uid))
        await harness.translation_turns.wait_for_idle()
        assert visible(presenter) == [(second, "second readable unit")]
    finally:
        await harness.stop()
        await presenter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text", ["Ask Dr.", "Use e.g.", "Version 3.14", "Number 2.", "Still waiting..."]
)
async def test_decimal_abbreviation_and_ellipsis_results_do_not_create_sentence_units(text):
    provider, osc = HeldTranslations(), RecordingOscQueue()
    presenter = OverlayPresenter(calibration=OverlayCalibration())
    harness = compose_translation_test_harness(
        stt=None,
        llm=provider,
        osc=osc,
        overlay_sink=presenter,
        low_latency_mode=True,
        low_latency_finalize_wait_ms=0,
    )
    try:
        uid = uuid4()
        await source(harness, uid, text)
        await eventually(lambda: (text, "en") in provider.responses)
        provider.finish(text, "ready before closure")
        await eventually(lambda: any("ready before closure" in m.text for m in osc.messages))
        assert visible(presenter) == []
        await harness.self_owner.handle_vad_event(SpeechEnd(uid))
        await harness.translation_turns.wait_for_idle()
        assert visible(presenter) == [(text, "ready before closure")]
    finally:
        await harness.stop()
        await presenter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("punctuated", [False, True])
async def test_continuous_source_keeps_cancelled_sentence_results_out_and_closes_without_llm(
    punctuated,
):
    provider, osc = HeldTranslations(), RecordingOscQueue()
    presenter = OverlayPresenter(calibration=OverlayCalibration())
    harness = compose_translation_test_harness(
        stt=None,
        llm=provider,
        osc=osc,
        overlay_sink=presenter,
        low_latency_mode=True,
        low_latency_finalize_wait_ms=0,
    )
    uid = uuid4()
    first = (
        "A complete sentence." if punctuated else ("unpunctuated continuous speech " * 30).rstrip()
    )
    final = first + " still speaking without another boundary"
    try:
        await source(harness, uid, first)
        await eventually(lambda: (first, "en") in provider.responses)
        cancelled = provider.responses[(first, "en")]
        await source(harness, uid, final)
        await eventually(lambda: (final, "en") in provider.responses)
        assert cancelled.cancelled()
        assert not any(block.secondary_text for block in presenter.snapshot().blocks)
        await harness.self_owner.handle_vad_event(SpeechEnd(uid))
        assert harness.self_owner.merge_buffer is None
        assert visible(presenter) == [(final, "")]
        provider.finish(final, "complete continuous translation")
        await harness.translation_turns.wait_for_idle()
        assert visible(presenter) == [(final, "complete continuous translation")]
        assert any("complete continuous translation" in m.text for m in osc.messages)
    finally:
        await harness.stop()
        await presenter.close()


@pytest.mark.asyncio
async def test_resumed_speech_keeps_exact_previous_sentence_coverage_until_new_unit_ready():
    provider = HeldTranslations()
    presenter = OverlayPresenter(calibration=OverlayCalibration())
    harness = compose_translation_test_harness(
        stt=None,
        llm=provider,
        osc=RecordingOscQueue(),
        overlay_sink=presenter,
        low_latency_mode=True,
        low_latency_finalize_wait_ms=80,
    )
    try:
        a, b = uuid4(), uuid4()
        await source(harness, a, "First sentence.")
        await eventually(lambda: ("First sentence.", "en") in provider.responses)
        provider.finish("First sentence.", "first")
        await eventually(lambda: visible(presenter) == [("First sentence.", "first")])
        await harness.self_owner.handle_vad_event(SpeechEnd(a))
        parent = harness.self_owner.merge_buffer.merge_id
        await harness.self_owner.handle_vad_event(
            SpeechStart(b, np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32))
        )
        await source(harness, b, "resumed words")
        combined = "First sentence.resumed words"
        await eventually(lambda: (combined, "en") in provider.responses)
        provider.finish(combined, "combined")
        await eventually(lambda: bool(harness.translation_turns._parents[parent].results))
        assert visible(presenter) == [("First sentence.", "first")]
        await harness.self_owner.handle_vad_event(SpeechEnd(b))
        await eventually(lambda: harness.self_owner.merge_buffer is None)
        await harness.translation_turns.wait_for_idle()
        assert visible(presenter) == [(combined, "combined")]
    finally:
        await harness.stop()
        await presenter.close()


@pytest.mark.asyncio
async def test_successive_ready_turns_keep_previous_translation_until_next_useful_successor():
    provider, clock = HeldTranslations(), FakeClock()
    presenter = OverlayPresenter(calibration=OverlayCalibration(), clock=clock)
    harness = compose_translation_test_harness(
        stt=None,
        llm=provider,
        osc=RecordingOscQueue(),
        overlay_sink=presenter,
        clock=clock,
        low_latency_mode=True,
        low_latency_finalize_wait_ms=0,
    )
    try:
        for text, translated in (("alpha", "A"), ("bravo", "B")):
            uid = uuid4()
            await source(harness, uid, text)
            await eventually(lambda: (text, "en") in provider.responses)
            provider.finish(text, translated)
            await asyncio.sleep(0)
            await harness.self_owner.handle_vad_event(SpeechEnd(uid))
            await eventually(
                lambda: any(b.secondary_text == translated for b in presenter.snapshot().blocks)
            )
            clock.advance(0.2)
        assert visible(presenter) == [("alpha", "A"), ("bravo", "B")]
        retained = presenter.snapshot().blocks[1]
        uid = uuid4()
        await source(harness, uid, "charlie")
        await eventually(lambda: ("charlie", "en") in provider.responses)
        await harness.self_owner.handle_vad_event(SpeechEnd(uid))
        assert visible(presenter) == [("alpha", "A"), ("bravo", "B")]
        clock.advance(0.2)
        provider.finish("charlie", "C")
        await eventually(lambda: visible(presenter) == [("bravo", "B"), ("charlie", "C")])
        assert presenter.snapshot().blocks[0].appearance_seq == retained.appearance_seq
        assert len(presenter.snapshot().blocks) == 2
        print(
            "OWNER_SMOKE ready_turns=true first_retained_until_third_useful_translation=true synthetic_read_window_s=0.6"
        )
    finally:
        await harness.stop()
        await presenter.close()


@pytest.mark.asyncio
async def test_staged_successor_first_translation_survives_newer_dashboard_source_without_reordering():
    provider = HeldTranslations()
    presenter = OverlayPresenter(calibration=OverlayCalibration())
    harness = compose_translation_test_harness(
        stt=None,
        llm=provider,
        osc=RecordingOscQueue(),
        overlay_sink=presenter,
        low_latency_mode=True,
        low_latency_finalize_wait_ms=0,
    )
    try:
        for text in ("alpha", "bravo", "charlie"):
            uid = uuid4()
            await source(harness, uid, text)
            await eventually(lambda: (text, "en") in provider.responses)
            if text == "alpha":
                provider.finish(text, "A")
            await harness.self_owner.handle_vad_event(SpeechEnd(uid))
            if text == "alpha":
                await eventually(lambda: visible(presenter) == [("alpha", "A")])
        assert visible(presenter) == [("alpha", "A"), ("bravo", "")]
        await source(harness, uuid4(), "newer dashboard source")
        await eventually(lambda: ("newer dashboard source", "en") in provider.responses)
        provider.finish("charlie", "C")
        await eventually(lambda: visible(presenter) == [("bravo", ""), ("charlie", "C")])
        ids = [block.id for block in presenter.snapshot().blocks]
        provider.finish("bravo", "B")
        await eventually(lambda: visible(presenter) == [("bravo", "B"), ("charlie", "C")])
        assert visible(presenter) == [("bravo", "B"), ("charlie", "C")]
        assert [block.id for block in presenter.snapshot().blocks] == ids
        assert presenter.active_self_overlay_metadata() is None
    finally:
        await harness.stop()
        await presenter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["disabled", "unavailable", "failed"])
async def test_closed_source_fallback_progresses_without_successful_translation(outcome):
    provider = HeldTranslations() if outcome == "failed" else None
    presenter = OverlayPresenter(calibration=OverlayCalibration())
    harness = compose_translation_test_harness(
        stt=None,
        llm=provider,
        osc=RecordingOscQueue(),
        overlay_sink=presenter,
        low_latency_mode=True,
        low_latency_finalize_wait_ms=0,
        translation_enabled=outcome != "disabled",
        fallback_transcript_only=True,
    )
    try:
        uid = uuid4()
        await source(harness, uid, "source fallback")
        if provider is not None:
            await eventually(lambda: ("source fallback", "en") in provider.responses)
        await harness.self_owner.handle_vad_event(SpeechEnd(uid))
        if provider is not None:
            provider.responses[("source fallback", "en")].set_exception(
                RuntimeError("synthetic failure")
            )
            await eventually(lambda: len(provider.calls) == 2)
            provider.responses[("source fallback", "en")].set_exception(
                RuntimeError("synthetic final failure")
            )
        await harness.translation_turns.wait_for_idle()
        assert visible(presenter) == [("source fallback", "")]
    finally:
        await harness.stop()
        await presenter.close()


@pytest.mark.asyncio
async def test_latest_staged_source_fills_expired_readable_slot_without_replaying_older_stages():
    provider, clock = HeldTranslations(), FakeClock()
    presenter = OverlayPresenter(calibration=OverlayCalibration(), clock=clock)
    harness = compose_translation_test_harness(
        stt=None,
        llm=provider,
        osc=RecordingOscQueue(),
        overlay_sink=presenter,
        clock=clock,
        low_latency_mode=True,
        low_latency_finalize_wait_ms=0,
    )
    try:
        for text in ("alpha", "bravo", "charlie", "delta"):
            uid = uuid4()
            await source(harness, uid, text)
            await eventually(lambda: (text, "en") in provider.responses)
            if text in {"alpha", "bravo"}:
                provider.finish(text, text.upper())
            await harness.self_owner.handle_vad_event(SpeechEnd(uid))
            if text in {"alpha", "bravo"}:
                await eventually(
                    lambda: any(
                        b.secondary_text == text.upper() for b in presenter.snapshot().blocks
                    )
                )
            clock.advance(1)
        assert visible(presenter) == [("alpha", "ALPHA"), ("bravo", "BRAVO")]
        clock.advance(4.1)
        await presenter.update_display_preferences(show_translation=True, show_peer_original=False)
        assert visible(presenter) == [("bravo", "BRAVO"), ("delta", "")]
        provider.finish("charlie", "stale intermediate")
        await eventually(
            lambda: ("charlie", "en") in provider.responses
            and provider.responses[("charlie", "en")].done()
        )
        assert visible(presenter) == [("bravo", "BRAVO"), ("delta", "")]
        provider.finish("delta", "D")
        await harness.translation_turns.wait_for_idle()
        assert visible(presenter) == [("bravo", "BRAVO"), ("delta", "D")]
        clock.advance(9)
        await presenter.update_display_preferences(show_translation=False, show_peer_original=True)
        assert visible(presenter) == []
    finally:
        await harness.stop()
        await presenter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("peer_state", ["older_finalized", "newer_finalized", "older_active"])
async def test_pending_and_failed_successor_stages_only_for_actual_translated_self_victim(
    peer_state,
):
    provider, clock, osc = HeldTranslations(), FakeClock(), RecordingOscQueue()
    presenter = OverlayPresenter(
        calibration=OverlayCalibration(), clock=clock, native_retry_enabled=True
    )
    harness = compose_translation_test_harness(
        stt=None,
        llm=provider,
        osc=osc,
        overlay_sink=presenter,
        clock=clock,
        low_latency_mode=True,
        low_latency_finalize_wait_ms=0,
        fallback_transcript_only=True,
    )
    adapter = harness.output_projection.overlay_event_adapter
    peer_id = uuid4()

    async def publish_peer():
        if peer_state == "older_active":
            await presenter.emit(
                adapter.peer_active_update(
                    text="peer", utterance_id=peer_id, occupant_key=f"peer:{peer_id}"
                )
            )
        else:
            await presenter.emit(
                adapter.translation_final(
                    utterance_id=peer_id,
                    channel="peer",
                    text="peer",
                    source_language="ko",
                    target_language="en",
                    applied_context_mode=None,
                )
            )

    try:
        if peer_state != "newer_finalized":
            await publish_peer()
        clock.advance(0.2)
        uid = uuid4()
        await source(harness, uid, "alpha")
        await eventually(lambda: ("alpha", "en") in provider.responses)
        provider.finish("alpha", "A")
        await harness.self_owner.handle_vad_event(SpeechEnd(uid))
        await harness.translation_turns.wait_for_idle()
        if peer_state == "newer_finalized":
            await publish_peer()
        before = presenter.snapshot()
        alpha = next(block for block in before.blocks if block.primary_text == "alpha")
        clock.advance(0.8)
        uid = uuid4()
        await source(harness, uid, "bravo")
        await eventually(lambda: ("bravo", "en") in provider.responses)
        await harness.self_owner.handle_vad_event(SpeechEnd(uid))
        expected = (
            [("alpha", "A"), ("bravo", "")]
            if peer_state == "older_finalized"
            else [(block.primary_text, block.secondary_text) for block in before.blocks]
        )
        assert visible(presenter) == expected
        assert len(presenter.snapshot().blocks) == 2
        if peer_state == "older_finalized":
            current_alpha, successor = presenter.snapshot().blocks
            assert (current_alpha.id, current_alpha.appearance_seq) == (
                alpha.id,
                alpha.appearance_seq,
            )
            assert alpha.appearance_seq < successor.appearance_seq
            assert presenter.active_self_overlay_metadata().text == "bravo"
            assert presenter.snapshot().native_quiet_tail_episodes.self.generation > (
                before.native_quiet_tail_episodes.self.generation
            )
        else:
            assert presenter.snapshot().native_quiet_tail_episodes.self == (
                before.native_quiet_tail_episodes.self
            )
        provider.responses[("bravo", "en")].set_exception(RuntimeError("synthetic failure"))
        await eventually(lambda: provider.calls.count(("bravo", "en")) == 2)
        provider.responses[("bravo", "en")].set_exception(RuntimeError("synthetic final failure"))
        await harness.translation_turns.wait_for_idle()
        assert visible(presenter) == expected
        assert [message.text for message in osc.messages] == ["alpha (A)", "bravo"]
        assert presenter.active_self_overlay_metadata() is None
        clock.advance(6.9)
        await presenter.update_display_preferences(show_translation=True, show_peer_original=False)
        if peer_state != "older_active":
            assert visible(presenter) == expected
        print(
            f"OWNER_SMOKE mixed_selection={peer_state} pending_and_failed=true "
            f"scene={expected!r} chatbox={[message.text for message in osc.messages]!r} "
            "before_expiry_s=7.9"
        )
    finally:
        await harness.stop()
        await presenter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "show_peer_original,translation_enabled,peer_visible",
    [(True, True, True), (False, True, False), (False, False, True)],
)
async def test_source_only_active_peer_eligibility_controls_successor_admission(
    show_peer_original, translation_enabled, peer_visible
):
    clock = FakeClock()
    presenter = OverlayPresenter(
        calibration=OverlayCalibration(),
        clock=clock,
        show_peer_original=show_peer_original,
        translation_enabled=translation_enabled,
    )
    adapter = OverlayEventAdapter(clock=clock)
    peer_id, first_id, successor_id = uuid4(), uuid4(), uuid4()
    try:
        await presenter.emit(
            adapter.peer_active_update(
                text="peer source", utterance_id=peer_id, occupant_key=f"peer:{peer_id}"
            )
        )
        await presenter.emit(
            adapter.self_active_update(
                utterance_id=first_id,
                occupant_key=f"self:{first_id}",
                text="alpha",
                secondary_text="A",
                caption_source_closed=True,
                caption_complete=True,
                caption_revision=1,
            )
        )
        before = visible(presenter)
        await presenter.emit(
            adapter.self_active_update(
                utterance_id=successor_id,
                occupant_key=f"self:{successor_id}",
                text="bravo",
                caption_source_closed=True,
                caption_complete=False,
                caption_revision=1,
            )
        )
        assert visible(presenter) == (before if peer_visible else [("alpha", "A"), ("bravo", "")])
        assert len(presenter.snapshot().blocks) == 2
        if peer_visible:
            peer = presenter.snapshot().blocks[0]
            assert (peer.primary_text, peer.secondary_text, peer.secondary_enabled) == (
                ("", "peer source", True) if translation_enabled else ("peer source", "", False)
            )
    finally:
        await presenter.close()


@pytest.mark.asyncio
async def test_staged_source_promotes_when_effective_preferences_hide_active_peer():
    clock = FakeClock()
    presenter = OverlayPresenter(
        calibration=OverlayCalibration(), clock=clock, native_retry_enabled=True
    )
    adapter = OverlayEventAdapter(clock=clock)
    peer_id, first_id, successor_id = uuid4(), uuid4(), uuid4()
    try:
        await presenter.emit(
            adapter.peer_active_update(
                text="peer source", utterance_id=peer_id, occupant_key=f"peer:{peer_id}"
            )
        )
        for uid, text, translated in ((first_id, "alpha", "A"), (successor_id, "bravo", "")):
            await presenter.emit(
                adapter.self_active_update(
                    utterance_id=uid,
                    occupant_key=f"self:{uid}",
                    text=text,
                    secondary_text=translated,
                    caption_source_closed=True,
                    caption_complete=True,
                    caption_revision=1,
                )
            )
        assert visible(presenter) == [("", "peer source"), ("alpha", "A")]
        alpha = presenter.snapshot().blocks[1]
        episode = presenter.snapshot().native_quiet_tail_episodes.self
        await presenter.update_display_preferences(show_translation=True, show_peer_original=False)
        assert clock.now() == 0
        assert visible(presenter) == [("alpha", "A"), ("bravo", "")]
        assert presenter.snapshot().blocks[0].appearance_seq == alpha.appearance_seq
        assert presenter.snapshot().native_quiet_tail_episodes.self.generation > episode.generation
        await presenter.update_display_preferences(show_translation=True, show_peer_original=True)
        assert visible(presenter) == [("", "peer source"), ("bravo", "")]
        await presenter.update_display_preferences(show_translation=True, show_peer_original=False)
        assert visible(presenter) == [("bravo", "")]
    finally:
        await presenter.close()


@pytest.mark.asyncio
async def test_staged_source_replaces_older_peer_when_it_loses_active_protection():
    clock = FakeClock()
    presenter = OverlayPresenter(
        calibration=OverlayCalibration(), clock=clock, native_retry_enabled=True
    )
    adapter = OverlayEventAdapter(clock=clock)
    peer_id, first_id, successor_id = uuid4(), uuid4(), uuid4()
    try:
        await presenter.emit(
            adapter.translation_final(
                utterance_id=peer_id,
                channel="peer",
                text="peer",
                source_language="ko",
                target_language="en",
                applied_context_mode=None,
            )
        )
        await presenter.emit(
            adapter.peer_active_update(
                text="peer source", utterance_id=peer_id, occupant_key=f"peer:{peer_id}"
            )
        )
        for uid, text, translated in ((first_id, "alpha", "A"), (successor_id, "bravo", "")):
            await presenter.emit(
                adapter.self_active_update(
                    utterance_id=uid,
                    occupant_key=f"self:{uid}",
                    text=text,
                    secondary_text=translated,
                    caption_source_closed=True,
                    caption_complete=True,
                    caption_revision=1,
                )
            )
        assert visible(presenter) == [("peer", "peer source"), ("alpha", "A")]
        alpha = presenter.snapshot().blocks[1]
        replacement_id = uuid4()
        await presenter.emit(
            adapter.peer_active_update(
                text="", utterance_id=replacement_id, occupant_key=f"peer:{replacement_id}"
            )
        )
        assert clock.now() == 0
        assert visible(presenter) == [("alpha", "A"), ("bravo", "")]
        assert presenter.snapshot().blocks[0].appearance_seq == alpha.appearance_seq
        assert len(presenter.snapshot().blocks) == 2
    finally:
        await presenter.close()


@pytest.mark.asyncio
async def test_mixed_peer_capacity_stages_source_but_serves_latest_translated_scene():
    provider, clock = HeldTranslations(), FakeClock()
    presenter = OverlayPresenter(
        calibration=OverlayCalibration(), clock=clock, native_retry_enabled=True
    )
    harness = compose_translation_test_harness(
        stt=None,
        llm=provider,
        osc=RecordingOscQueue(),
        overlay_sink=presenter,
        clock=clock,
        low_latency_mode=True,
        low_latency_finalize_wait_ms=0,
    )
    adapter = harness.output_projection.overlay_event_adapter
    try:
        uid = uuid4()
        await source(harness, uid, "alpha")
        await eventually(lambda: ("alpha", "en") in provider.responses)
        provider.finish("alpha", "A")
        await harness.self_owner.handle_vad_event(SpeechEnd(uid))
        await eventually(lambda: visible(presenter) == [("alpha", "A")])
        await harness.translation_turns.wait_for_idle()
        await presenter.emit(
            adapter.translation_final(
                utterance_id=uuid4(),
                channel="peer",
                text="peer",
                source_language="ko",
                target_language="en",
                applied_context_mode=None,
            )
        )
        before = presenter.snapshot().native_quiet_tail_episodes.self
        uid = uuid4()
        await source(harness, uid, "bravo")
        await eventually(lambda: ("bravo", "en") in provider.responses)
        await harness.self_owner.handle_vad_event(SpeechEnd(uid))
        assert visible(presenter) == [("alpha", "A"), ("peer", "")]
        assert presenter.snapshot().native_quiet_tail_episodes.self == before
        provider.finish("bravo", "B")
        await harness.translation_turns.wait_for_idle()
        assert visible(presenter) == [("peer", ""), ("bravo", "B")]
        assert presenter.snapshot().native_quiet_tail_episodes.self.generation > before.generation
    finally:
        await harness.stop()
        await presenter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("overlay_mode", ["blocked", "disabled"])
async def test_overlay_delivery_does_not_change_chatbox_corrections_or_translation_progress(
    overlay_mode,
):
    provider, sender = HeldTranslations(), Sender()
    presenter = OverlayPresenter(calibration=OverlayCalibration())
    entered, release = asyncio.Event(), asyncio.Event()

    class BlockedOverlay:
        async def emit(self, event):
            entered.set()
            await release.wait()
            return await presenter.emit(event)

    harness = compose_translation_test_harness(
        stt=None,
        llm=provider,
        osc=ChatboxPaginator(sender=sender, clock=FakeClock()),
        overlay_sink=BlockedOverlay() if overlay_mode == "blocked" else None,
        low_latency_mode=True,
        low_latency_finalize_wait_ms=0,
        chatbox_include_source=False,
    )
    try:
        uid = uuid4()
        for text, target in (("alpha", "A"), ("alpha beta", "AB")):
            await source(harness, uid, text)
            await eventually(lambda: (text, "en") in provider.responses)
            provider.finish(text, target)
            await eventually(lambda: target in sender.packets)
        await harness.self_owner.handle_vad_event(SpeechEnd(uid))
        if overlay_mode == "blocked":
            await entered.wait()
        await source(harness, uuid4(), "successor")
        await eventually(lambda: ("successor", "en") in provider.responses)
        provider.finish("successor", "S")
        await eventually(lambda: "S" in sender.packets)
        assert sender.packets == ["A", "AB", "S"]
        assert visible(presenter) == []
        assert (
            harness.output_runtime.overlay_admission_snapshot()["scopes"]
            .get("preview:self", {})
            .get("unsent", 0)
            <= 8
        )
        release.set()
        if overlay_mode == "blocked":
            await eventually(lambda: visible(presenter) == [("alpha beta", "AB")])
    finally:
        release.set()
        await harness.stop()
        await presenter.close()


@pytest.mark.asyncio
async def test_blocked_ready_burst_delivers_latest_two_units_not_a_caption_backlog():
    from dataclasses import replace

    provider, sender, clock, scenes = HeldTranslations(), Sender(), FakeClock(), Scenes()
    presenter = OverlayPresenter(calibration=OverlayCalibration(), clock=clock, bridge=scenes)
    entered, release = asyncio.Event(), asyncio.Event()
    delivered, snapshots = [], []

    class BlockedOverlay:
        async def emit(self, event):
            entered.set()
            await release.wait()
            delivered.append(event)
            return await presenter.emit(event)

    harness = compose_translation_test_harness(
        stt=None,
        llm=provider,
        osc=ChatboxPaginator(sender=sender, clock=clock),
        overlay_sink=BlockedOverlay(),
        clock=clock,
        low_latency_mode=True,
        low_latency_finalize_wait_ms=0,
        chatbox_include_source=False,
    )
    callback = harness.translation_turns.on_draft_updated

    async def observe(snapshot, request):
        snapshots.append((snapshot, request))
        await callback(snapshot, request)

    harness.translation_turns.on_draft_updated = observe
    try:
        for index in range(9):
            text = f"turn {index}"
            uid = uuid4()
            await source(harness, uid, text)
            await eventually(lambda: (text, "en") in provider.responses)
            provider.finish(text, f"T{index}")
            await eventually(lambda: f"T{index}" in sender.packets)
            await harness.self_owner.handle_vad_event(SpeechEnd(uid))
            await eventually(lambda: harness.self_owner.merge_buffer is None)
            if index == 0:
                await entered.wait()
        await eventually(lambda: not harness.translation_turns._parent_tasks)
        assert len(sender.packets) == 9
        assert (
            harness.output_runtime.overlay_admission_snapshot()["scopes"]["preview:self"]["unsent"]
            <= 2
        )
        release.set()
        await eventually(lambda: not harness.output_runtime.has_active_overlay_deliveries)
        assert visible(presenter) == [("turn 7", "T7"), ("turn 8", "T8")]
        assert {event.text for event in delivered} == {"turn 0", "turn 7", "turn 8"}
        assert not any(
            block.primary_text in {f"turn {index}" for index in range(1, 7)}
            for snapshot in scenes.snapshots
            for block in snapshot.blocks
        )
        old, request = next((s, r) for s, r in reversed(snapshots) if s.source_text == "turn 1")
        await harness.output_projection.project_self_turn(
            replace(old, revision=old.revision + 1000), request.config_snapshot, source="Mic"
        )
        await eventually(lambda: not harness.output_runtime.has_active_overlay_deliveries)
        assert visible(presenter) == [("turn 7", "T7"), ("turn 8", "T8")]
        assert {event.text for event in delivered} == {"turn 0", "turn 7", "turn 8"}
    finally:
        release.set()
        await harness.stop()
        await presenter.close()


@pytest.mark.asyncio
async def test_queued_first_admission_keeps_authority_when_newer_turn_precedes_its_result():
    provider, clock = HeldTranslations(), FakeClock()
    presenter = OverlayPresenter(calibration=OverlayCalibration(), clock=clock)
    entered, release = asyncio.Event(), asyncio.Event()
    delivered = []

    class BlockedOverlay:
        async def emit(self, event):
            entered.set()
            await release.wait()
            delivered.append(event)
            return await presenter.emit(event)

    harness = compose_translation_test_harness(
        stt=None,
        llm=provider,
        osc=RecordingOscQueue(),
        overlay_sink=BlockedOverlay(),
        clock=clock,
        low_latency_mode=True,
        low_latency_finalize_wait_ms=0,
    )
    try:
        for text in ("alpha", "bravo", "charlie"):
            uid = uuid4()
            await source(harness, uid, text)
            await eventually(lambda: (text, "en") in provider.responses)
            if text == "charlie":
                provider.finish(text, "C")
                await eventually(
                    lambda: bool(
                        harness.translation_turns._parents[
                            harness.self_owner.merge_buffer.merge_id
                        ].results
                    )
                )
            await harness.self_owner.handle_vad_event(SpeechEnd(uid))
            if text == "alpha":
                await entered.wait()
        provider.finish("bravo", "B")
        await eventually(
            lambda: any(
                "B" in batch.retained_payloads.values()
                for batch in harness.output_runtime._batch_admission.batches.values()
            )
        )
        release.set()
        await eventually(lambda: not harness.output_runtime.has_active_overlay_deliveries)
        assert visible(presenter) == [("bravo", "B"), ("charlie", "C")]
        first_bravo = next(event for event in delivered if getattr(event, "text", "") == "bravo")
        first_charlie = next(
            event for event in delivered if getattr(event, "text", "") == "charlie"
        )
        assert not first_bravo.retained_only
        assert delivered.index(first_bravo) < delivered.index(first_charlie)
        provider.finish("alpha", "A")
        await harness.translation_turns.wait_for_idle()
        assert visible(presenter) == [("bravo", "B"), ("charlie", "C")]
    finally:
        release.set()
        await harness.stop()
        await presenter.close()


@pytest.mark.asyncio
async def test_simultaneous_expiry_reveals_latest_staged_source_before_its_retention_ends():
    clock = FakeClock()
    presenter = OverlayPresenter(calibration=OverlayCalibration(), clock=clock)
    adapter = OverlayEventAdapter(clock=clock)
    ids = [uuid4() for _ in range(4)]
    try:
        for index, uid in enumerate(ids):
            await presenter.emit(
                adapter.self_active_update(
                    utterance_id=uid,
                    occupant_key=f"self:{uid}",
                    text=f"source {index}",
                    secondary_text=f"T{index}" if index < 2 else "",
                    caption_source_closed=True,
                    caption_complete=index < 2,
                    caption_revision=1,
                    turn_generation=0,
                    turn_order=index,
                )
            )
        assert visible(presenter) == [("source 0", "T0"), ("source 1", "T1")]
        clock.advance(8.1)
        await presenter.update_display_preferences(show_translation=True, show_peer_original=False)
        assert visible(presenter) == [("source 3", "")]
        await presenter.emit(
            adapter.self_active_update(
                utterance_id=ids[2],
                occupant_key=f"self:{ids[2]}",
                text="source 2",
                secondary_text="late retired",
                caption_source_closed=True,
                caption_complete=True,
                caption_revision=2,
                retained_only=True,
                turn_generation=0,
                turn_order=2,
            )
        )
        assert visible(presenter) == [("source 3", "")]
        clock.advance(9)
        await presenter.update_display_preferences(show_translation=True, show_peer_original=True)
        assert visible(presenter) == []
        await presenter.emit(
            adapter.self_active_update(
                utterance_id=ids[3],
                occupant_key=f"self:{ids[3]}",
                text="source 3",
                secondary_text="expired result",
                caption_source_closed=True,
                caption_complete=True,
                caption_revision=2,
                turn_generation=0,
                turn_order=3,
            )
        )
        assert visible(presenter) == []
    finally:
        await presenter.close()


@pytest.mark.asyncio
async def test_target_configuration_change_clears_old_target_without_admitting_nonunit_result():
    provider = HeldTranslations()
    presenter = OverlayPresenter(calibration=OverlayCalibration())
    harness = compose_translation_test_harness(
        stt=None,
        llm=provider,
        osc=RecordingOscQueue(),
        overlay_sink=presenter,
        low_latency_mode=True,
        low_latency_finalize_wait_ms=0,
    )
    uid = uuid4()
    first, final = "A complete sentence.", "A complete sentence. unfinished"
    try:
        await source(harness, uid, first)
        await eventually(lambda: (first, "en") in provider.responses)
        provider.finish(first, "English")
        await eventually(lambda: visible(presenter) == [(first, "English")])
        await source(harness, uid, final)
        await eventually(lambda: (final, "en") in provider.responses)
        old_request = provider.responses[(final, "en")]
        harness.replace_configuration(target_language="fr", self_target_languages=("fr",))
        await source(harness, uid, final)
        await eventually(lambda: (final, "fr") in provider.responses)
        assert old_request.cancelled()
        assert visible(presenter) == [(first, "")]
        provider.finish(final, "French", "fr")
        parent_id = harness.self_owner.merge_buffer.merge_id
        await eventually(lambda: bool(harness.translation_turns._parents[parent_id].results))
        assert visible(presenter) == [(first, "")]
        await harness.self_owner.handle_vad_event(SpeechEnd(uid))
        await harness.translation_turns.wait_for_idle()
        assert visible(presenter) == [(final, "French")]
        await harness.self_owner.reset_provider_channel()
        assert visible(presenter) == [(final, "French")]
        await presenter.clear_for_runtime_detach()
        assert visible(presenter) == []
    finally:
        await harness.stop()
        await presenter.close()


@pytest.mark.asyncio
async def test_queued_failed_source_fallback_cannot_erase_latest_translated_units():
    provider, sender, clock = HeldTranslations(), Sender(), FakeClock()
    presenter = OverlayPresenter(calibration=OverlayCalibration(), clock=clock)
    entered, release = asyncio.Event(), asyncio.Event()

    class BlockedOverlay:
        async def emit(self, event):
            entered.set()
            await release.wait()
            return await presenter.emit(event)

    harness = compose_translation_test_harness(
        stt=None,
        llm=provider,
        osc=ChatboxPaginator(sender=sender, clock=clock),
        overlay_sink=BlockedOverlay(),
        clock=clock,
        low_latency_mode=True,
        low_latency_finalize_wait_ms=0,
        chatbox_include_source=False,
        fallback_transcript_only=True,
    )
    try:
        for text in ("initial", "one", "two"):
            uid = uuid4()
            await source(harness, uid, text)
            await eventually(lambda: (text, "en") in provider.responses)
            provider.finish(text, text.upper())
            await eventually(lambda: text.upper() in sender.packets)
            await harness.self_owner.handle_vad_event(SpeechEnd(uid))
            if text == "initial":
                await entered.wait()
        for text in ("failed one", "failed two"):
            uid = uuid4()
            await source(harness, uid, text)
            await eventually(lambda: (text, "en") in provider.responses)
            await harness.self_owner.handle_vad_event(SpeechEnd(uid))
            provider.responses[(text, "en")].set_exception(RuntimeError("synthetic failure"))
            await eventually(lambda: provider.calls.count((text, "en")) == 2)
            provider.responses[(text, "en")].set_exception(RuntimeError("synthetic final failure"))
            await eventually(lambda: text in sender.packets)
        release.set()
        await eventually(lambda: not harness.output_runtime.has_active_overlay_deliveries)
        assert visible(presenter) == [("one", "ONE"), ("two", "TWO")]
        clock.advance(8.1)
        await presenter.update_display_preferences(show_translation=True, show_peer_original=False)
        assert visible(presenter) == [("failed two", "")]
        assert "failed one" in sender.packets and "failed two" in sender.packets
    finally:
        release.set()
        await harness.stop()
        await presenter.close()
