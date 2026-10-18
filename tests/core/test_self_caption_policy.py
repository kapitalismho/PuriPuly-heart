from __future__ import annotations

import asyncio
from dataclasses import replace
from uuid import uuid4

import numpy as np
import pytest

from puripuly_heart.config.overlay_calibration import OverlayCalibration
from puripuly_heart.core.audio.ownership import AudioSegmentIdentity
from puripuly_heart.core.clock import FakeClock
from puripuly_heart.core.llm.provider import LLMProvider
from puripuly_heart.core.overlay.presenter import OverlayPresenter
from puripuly_heart.core.overlay.sink import OverlayEventAdapter
from puripuly_heart.core.stt.backend import (
    STTProviderTurnIdentity,
    STTProviderTurnTerminal,
    STTProviderTurnUpdate,
    STTTextContribution,
)
from puripuly_heart.core.vad.gating import SpeechEnd, SpeechStart
from puripuly_heart.domain.events import SelfTurnTargetPresentation, STTFinalEvent
from puripuly_heart.domain.models import Transcript, Translation
from tests.helpers.fakes import RecordingOscQueue
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


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["stable_append", "final_only"])
async def test_speech_boundaries_and_completed_translation_advance_one_caption(route):
    provider = HeldTranslations()
    presenter = OverlayPresenter(calibration=OverlayCalibration(), native_retry_enabled=True)
    harness = compose_translation_test_harness(
        stt=None,
        llm=provider,
        osc=RecordingOscQueue(),
        overlay_sink=presenter,
        low_latency_mode=True,
        low_latency_finalize_wait_ms=400,
    )
    source_id = uuid4()
    identity = STTProviderTurnIdentity(AudioSegmentIdentity(1, 1, source_id, 0), "epoch", "request")
    contributions = []
    snapshots = []
    callback = harness.translation_turns.on_draft_updated

    async def observe(snapshot, request):
        snapshots.append(snapshot)
        await callback(snapshot, request)

    harness.translation_turns.on_draft_updated = observe

    async def update(text):
        if route == "final_only":
            await source(harness, source_id, text)
        else:
            start = contributions[-1].text_end if contributions else 0
            contributions.append(STTTextContribution(str(len(contributions)), start, len(text)))
            await harness.self_owner.handle_stt_event(
                STTProviderTurnUpdate(
                    identity,
                    len(contributions),
                    "stable",
                    "append",
                    text,
                    contribution=contributions[-1],
                )
            )

    try:
        await update("alpha")
        await eventually(lambda: ("alpha", "en") in provider.responses)
        await update("alpha beta")
        await eventually(lambda: ("alpha beta", "en") in provider.responses)
        assert visible(presenter) == []
        assert presenter.snapshot().native_quiet_tail_episodes is None
        await harness.self_owner.handle_vad_event(SpeechEnd(source_id))
        assert visible(presenter) == [("alpha beta", "")]
        first = presenter.snapshot().blocks[0]
        first_episode = presenter.snapshot().native_quiet_tail_episodes.self
        assert first_episode.phase == "stream"
        assert snapshots[-1].boundary_source_revision == snapshots[-1].source_revision == 2
        assert len(provider.calls) == 2
        provider.finish("alpha beta", "AB")
        await eventually(lambda: visible(presenter) == [("alpha beta", "AB")])
        translated_episode = presenter.snapshot().native_quiet_tail_episodes.self
        assert translated_episode.generation > first_episode.generation
        source_id = uuid4()
        identity = STTProviderTurnIdentity(
            AudioSegmentIdentity(1, 2, source_id, 0), "epoch", "request-2"
        )
        contributions = []
        await harness.self_owner.handle_vad_event(
            SpeechStart(source_id, np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32))
        )
        await update("gamma")
        await eventually(lambda: ("alpha beta gamma", "en") in provider.responses)
        assert visible(presenter) == [("alpha beta", "AB")]
        assert presenter.snapshot().native_quiet_tail_episodes.self == translated_episode
        await harness.self_owner.handle_vad_event(SpeechEnd(source_id))
        assert visible(presenter) == [("alpha beta gamma", "AB")]
        assert (
            presenter.snapshot().native_quiet_tail_episodes.self.generation
            > translated_episode.generation
        )
        source_id = uuid4()
        identity = STTProviderTurnIdentity(
            AudioSegmentIdentity(1, 3, source_id, 0), "epoch", "request-3"
        )
        contributions = []
        await harness.self_owner.handle_vad_event(
            SpeechStart(source_id, np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32))
        )
        await update("delta")
        await eventually(lambda: ("alpha beta gamma delta", "en") in provider.responses)
        provider.finish("alpha beta gamma delta", "ABGD")
        await eventually(lambda: visible(presenter) == [("alpha beta gamma delta", "ABGD")])
        if route == "stable_append":
            await harness.self_owner.handle_stt_event(
                STTProviderTurnTerminal(
                    identity,
                    "final",
                    text="delta",
                    text_authority="authoritative",
                    included_contributions=tuple(contributions),
                )
            )
        await harness.self_owner.handle_vad_event(SpeechEnd(source_id))
        await eventually(lambda: harness.self_owner.merge_buffer is None)
        await harness.translation_turns.wait_for_idle()
        final = presenter.snapshot().blocks[0]
        assert (final.id, final.occupant_key, final.appearance_seq) == (
            first.id,
            first.occupant_key,
            first.appearance_seq,
        )
        assert final.block_variant == "finalized"
        assert visible(presenter) == [("alpha beta gamma delta", "ABGD")]
        assert snapshots[-1].source_revision == 4
    finally:
        await harness.stop()
        await presenter.close()


@pytest.mark.asyncio
async def test_closed_pending_predecessor_retains_identity_and_late_translation_without_focus():
    provider = HeldTranslations()
    clock = FakeClock()
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
    a, b, c = uuid4(), uuid4(), uuid4()
    try:
        await source(harness, a, "alpha")
        await eventually(lambda: ("alpha", "en") in provider.responses)
        await harness.self_owner.handle_vad_event(SpeechEnd(a))
        first = presenter.snapshot().blocks[0]
        assert first.block_variant == "active_self"
        await source(harness, b, "bravo")
        await eventually(lambda: ("bravo", "en") in provider.responses)
        provider.finish("alpha", "A")
        await eventually(lambda: visible(presenter) == [("alpha", "A")])
        assert presenter.snapshot().blocks[0].id == first.id
        await harness.self_owner.handle_vad_event(SpeechEnd(b))
        assert visible(presenter) == [("alpha", "A"), ("bravo", "")]
        b_block = presenter.snapshot().blocks[-1]
        provider.finish("bravo", "B")
        await eventually(lambda: visible(presenter) == [("alpha", "A"), ("bravo", "B")])
        await source(harness, c, "charlie")
        await harness.self_owner.handle_vad_event(SpeechEnd(c))
        assert visible(presenter) == [("bravo", "B"), ("charlie", "")]
        assert presenter.snapshot().blocks[0].appearance_seq == b_block.appearance_seq
    finally:
        await harness.stop()
        await presenter.close()


@pytest.mark.asyncio
async def test_retained_result_does_not_reorder_or_revive_displaced_caption():
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
    snapshots = []
    callback = harness.translation_turns.on_draft_updated

    async def observe(snapshot, request):
        snapshots.append((snapshot, request))
        await callback(snapshot, request)

    harness.translation_turns.on_draft_updated = observe
    try:
        for text in ("alpha", "bravo"):
            uid = uuid4()
            await source(harness, uid, text)
            await eventually(lambda: (text, "en") in provider.responses)
            await harness.self_owner.handle_vad_event(SpeechEnd(uid))
        identities = [(b.id, b.occupant_key, b.appearance_seq) for b in presenter.snapshot().blocks]
        current = presenter.active_self_overlay_metadata().utterance_id
        provider.finish("alpha", "A")
        await eventually(lambda: visible(presenter) == [("alpha", "A"), ("bravo", "")])
        assert [
            (b.id, b.occupant_key, b.appearance_seq) for b in presenter.snapshot().blocks
        ] == identities
        assert presenter.active_self_overlay_metadata().utterance_id == current
        a_snapshot, request = next(
            (s, r) for s, r in reversed(snapshots) if s.source_text == "alpha"
        )
        uid = uuid4()
        await source(harness, uid, "charlie")
        await harness.self_owner.handle_vad_event(SpeechEnd(uid))
        before = visible(presenter)
        await harness.output_projection.project_self_turn(
            replace(a_snapshot, revision=a_snapshot.revision + 100),
            request.config_snapshot,
            source="Mic",
        )
        assert visible(presenter) == before
    finally:
        await harness.stop()
        await presenter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["disabled", "failed"])
async def test_closed_source_survives_without_matching_translation(outcome):
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
        await source(harness, uid, "alpha")
        if provider is not None:
            await eventually(lambda: ("alpha", "en") in provider.responses)
            provider.finish("alpha", "old")
            await eventually(lambda: visible(presenter) == [("alpha", "old")])
            await source(harness, uid, "alpha changed")
            await eventually(lambda: ("alpha changed", "en") in provider.responses)
        await harness.self_owner.handle_vad_event(SpeechEnd(uid))
        closed = "alpha changed" if provider is not None else "alpha"
        assert visible(presenter) == [(closed, "old" if provider is not None else "")]
        if provider is not None:
            provider.responses[(closed, "en")].set_exception(RuntimeError("failed"))
            await eventually(lambda: len(provider.calls) == 3)
            provider.responses[(closed, "en")].set_exception(RuntimeError("failed final"))
        await asyncio.wait_for(harness.translation_turns.wait_for_idle(), 2)
        assert visible(presenter) == [(closed, "")]
    finally:
        await harness.stop()
        await presenter.close()


@pytest.mark.asyncio
async def test_boundary_only_reuse_target_provenance_resync_and_reset():
    provider = HeldTranslations()
    presenter = OverlayPresenter(calibration=OverlayCalibration(), native_retry_enabled=True)
    harness = compose_translation_test_harness(
        stt=None,
        llm=provider,
        osc=RecordingOscQueue(),
        overlay_sink=presenter,
        low_latency_mode=True,
        low_latency_finalize_wait_ms=400,
        self_target_languages=("en", "fr"),
    )
    snapshots = []
    callback = harness.translation_turns.on_draft_updated

    async def observe(snapshot, request):
        snapshots.append((snapshot, request))
        await callback(snapshot, request)

    harness.translation_turns.on_draft_updated = observe
    uid = uuid4()
    try:
        await source(harness, uid, "alpha")
        await eventually(lambda: ("alpha", "en") in provider.responses)
        buffer = harness.self_owner.merge_buffer
        await harness.self_owner._sync_overlay_active_self(buffer)
        assert visible(presenter) == []
        before = snapshots[-1][0]
        await harness.self_owner.handle_vad_event(SpeechEnd(uid))
        boundary = snapshots[-1][0]
        assert boundary.source_revision == before.source_revision
        assert boundary.boundary_source_revision == boundary.source_revision
        await eventually(lambda: ("alpha", "fr") in provider.responses)
        provider.finish("alpha", "French", "fr")
        await eventually(lambda: snapshots[-1][0].targets[1].status == "ready")
        assert visible(presenter) == [("alpha", "")]
        provider.finish("alpha", "English")
        await eventually(lambda: visible(presenter) == [("alpha", "English")])
        calls = list(provider.calls)
        unchanged_episode = presenter.snapshot().native_quiet_tail_episodes.self
        unchanged_revision = snapshots[-1][0].source_revision
        await source(harness, uid, "alpha")
        assert provider.calls == calls
        assert visible(presenter) == [("alpha", "English")]
        assert presenter.snapshot().native_quiet_tail_episodes.self == unchanged_episode
        assert snapshots[-1][0].source_revision == unchanged_revision
        await source(harness, uid, "alpha.")
        assert provider.calls == calls
        assert visible(presenter) == [("alpha.", "English")]
        current, request = snapshots[-1]
        stale = replace(
            current,
            revision=current.revision + 1,
            source_revision=current.source_revision + 1,
            source_text="changed",
            targets=(
                SelfTurnTargetPresentation(
                    "en", current.source_revision, "ready", current.targets[0].translation
                ),
            ),
        )
        await harness.output_projection.project_self_turn(
            stale, request.config_snapshot, source="Mic"
        )
        assert visible(presenter) == [("alpha.", "English")]
        await harness.translation_turns.close_channel_ingress("self")
        await harness.output_projection.reset_overlay_preview()
        assert presenter.active_self_overlay_metadata() is None
        assert not harness.output_projection._self_captions
        await harness.output_projection.project_self_turn(
            replace(current, revision=1000), request.config_snapshot, source="Mic"
        )
        assert presenter.active_self_overlay_metadata() is None
    finally:
        await harness.stop()
        await presenter.close()


@pytest.mark.asyncio
async def test_blocked_overlay_preserves_predecessor_patch_and_successor_with_bounded_admission():
    provider = HeldTranslations()
    presenter = OverlayPresenter(calibration=OverlayCalibration())
    entered, release = asyncio.Event(), asyncio.Event()

    class BlockedOverlay:
        async def emit(self, event):
            entered.set()
            await release.wait()
            return await presenter.emit(event)

    osc = RecordingOscQueue()
    harness = compose_translation_test_harness(
        stt=None,
        llm=provider,
        osc=osc,
        overlay_sink=BlockedOverlay(),
        low_latency_mode=True,
        low_latency_finalize_wait_ms=0,
    )
    try:
        a, b = uuid4(), uuid4()
        await source(harness, a, "alpha")
        await eventually(lambda: ("alpha", "en") in provider.responses)
        await harness.self_owner.handle_vad_event(SpeechEnd(a))
        await entered.wait()
        provider.finish("alpha", "A")
        await eventually(lambda: bool(osc.messages))
        await source(harness, b, "bravo")
        await eventually(lambda: ("bravo", "en") in provider.responses)
        await harness.self_owner.handle_vad_event(SpeechEnd(b))
        assert visible(presenter) == []
        assert len(provider.calls) == 2
        assert any("A" in message.text for message in osc.messages)
        assert (
            harness.output_runtime._batch_admission.snapshot()["scopes"]["preview:self"]["unsent"]
            <= 8
        )
        release.set()
        await eventually(lambda: visible(presenter) == [("alpha", "A"), ("bravo", "")])
        assert presenter.active_self_overlay_metadata().text == "bravo"
    finally:
        release.set()
        await harness.stop()
        await presenter.close()


@pytest.mark.asyncio
async def test_shared_peer_pressure_keeps_closed_pending_current_and_rejects_expired_results():
    provider = HeldTranslations()
    clock = FakeClock()
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
    adapter = OverlayEventAdapter(clock=clock)
    try:
        uid = uuid4()
        await source(harness, uid, "alpha")
        await eventually(lambda: ("alpha", "en") in provider.responses)
        await harness.self_owner.handle_vad_event(SpeechEnd(uid))
        first = presenter.snapshot().blocks[0]
        for text in ("peer one", "peer two"):
            clock.advance(1.1)
            await presenter.emit_peer_when_admissible(
                adapter.translation_final(
                    utterance_id=uuid4(),
                    channel="peer",
                    text=text,
                    source_language="ko",
                    target_language="en",
                    applied_context_mode=None,
                )
            )
        assert len(presenter.snapshot().blocks) == 2
        assert any(block.id == first.id for block in presenter.snapshot().blocks)
        provider.finish("alpha", "A")
        await eventually(
            lambda: any(block.secondary_text == "A" for block in presenter.snapshot().blocks)
        )
        assert (
            next(
                block for block in presenter.snapshot().blocks if block.id == first.id
            ).appearance_seq
            == first.appearance_seq
        )
        uid = uuid4()
        await source(harness, uid, "late")
        await eventually(lambda: ("late", "en") in provider.responses)
        await harness.self_owner.handle_vad_event(SpeechEnd(uid))
        clock.advance(9)
        await presenter.update_display_preferences(show_translation=False, show_peer_original=True)
        assert not presenter.snapshot().blocks
        provider.finish("late", "too late")
        await harness.translation_turns.wait_for_idle()
        assert not presenter.snapshot().blocks
    finally:
        await harness.stop()
        await presenter.close()


@pytest.mark.asyncio
async def test_speech_reset_and_target_configuration_never_carry_retired_translation():
    provider = HeldTranslations()
    presenter = OverlayPresenter(calibration=OverlayCalibration())
    harness = compose_translation_test_harness(
        stt=None,
        llm=provider,
        osc=RecordingOscQueue(),
        overlay_sink=presenter,
        low_latency_mode=True,
        low_latency_finalize_wait_ms=400,
    )
    try:
        uid = uuid4()
        await source(harness, uid, "alpha")
        await eventually(lambda: ("alpha", "en") in provider.responses)
        provider.finish("alpha", "English")
        await eventually(lambda: visible(presenter) == [("alpha", "English")])
        harness.replace_configuration(target_language="fr", self_target_languages=("fr",))
        await source(harness, uid, "alpha")
        await eventually(lambda: ("alpha", "fr") in provider.responses)
        assert visible(presenter) == [("alpha", "")]
        provider.finish("alpha", "French", "fr")
        await eventually(lambda: visible(presenter) == [("alpha", "French")])
        await harness.self_owner.reset_provider_channel()
        assert not presenter.snapshot().blocks
        assert not harness.output_projection._self_captions
        await source(harness, uuid4(), "bravo")
        assert not presenter.snapshot().blocks
    finally:
        await harness.stop()
        await presenter.close()


@pytest.mark.asyncio
async def test_equal_utterances_keep_separate_caption_identity_without_sticky_carryover():
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
        first_id = uuid4()
        await source(harness, first_id, "same")
        await eventually(lambda: ("same", "en") in provider.responses)
        provider.finish("same", "first")
        await eventually(lambda: visible(presenter) == [("same", "first")])
        await harness.self_owner.handle_vad_event(SpeechEnd(first_id))
        await harness.translation_turns.wait_for_idle()
        first = presenter.snapshot().blocks[0]
        second_id = uuid4()
        await source(harness, second_id, "same")
        await eventually(lambda: len(provider.calls) == 2)
        assert visible(presenter) == [("same", "first")]
        await harness.self_owner.handle_vad_event(SpeechEnd(second_id))
        assert visible(presenter) == [("same", "first"), ("same", "")]
        second = presenter.snapshot().blocks[-1]
        assert second.id != first.id
        assert second.appearance_seq > first.appearance_seq
        provider.finish("same", "second")
        await eventually(lambda: visible(presenter) == [("same", "first"), ("same", "second")])
    finally:
        await harness.stop()
        await presenter.close()
