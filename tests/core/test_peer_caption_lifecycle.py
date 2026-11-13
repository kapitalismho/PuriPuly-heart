from __future__ import annotations

import asyncio
from dataclasses import replace
from types import SimpleNamespace
from uuid import uuid4

import pytest

from puripuly_heart.config.overlay_calibration import OverlayCalibration
from puripuly_heart.config.prompts import get_default_prompt
from puripuly_heart.core.audio.ownership import AudioSegmentTerminalReceipt
from puripuly_heart.core.clock import FakeClock
from puripuly_heart.core.local_asr.local_asr_provider_runtime import ProviderRuntimeBuildRequest
from puripuly_heart.core.orchestrator.channel_runtime import ContextEntry
from puripuly_heart.core.orchestrator.peer_caption import join_action, preview_source
from puripuly_heart.core.orchestrator.translation_channel_callbacks import (
    TranslationChannelOwnerCallbacks,
)
from puripuly_heart.core.overlay.peer_caption_layout import estimate_peer_translation_lines
from puripuly_heart.core.overlay.presenter import OverlayPresenter
from puripuly_heart.core.runtime.local_asr_provider_runtime import LocalASRProviderRuntimeOwner
from puripuly_heart.core.stt.backend import (
    STTProviderEpochEnded,
    STTProviderTurnIdentity,
    STTProviderTurnTerminal,
    STTProviderTurnUpdate,
    STTRecognitionUnit,
    STTRecognitionUnitTerminal,
)
from puripuly_heart.core.translation_backend import TranslationBackend
from puripuly_heart.domain.events import STTFinalEvent, UIEventType
from puripuly_heart.domain.models import FinalLanguageRun, FinalSpeakerRun, Transcript, Translation
from puripuly_heart.domain.recognition import RecognitionStreamIdentity, RecognitionUnitIdentity
from tests.core.runtime.test_local_asr_provider_runtime import (
    FakeGpuRuntimeFactory,
    FakeProviderFactory,
    FakeProvisioningPort,
    _resolved_config,
    _wait_until,
)
from tests.core.test_peer_overlay_policy import Surface, Timeline, caption
from tests.helpers.fakes import RecordingOscQueue
from tests.helpers.translation_owners import compose_translation_test_harness, owned_peer_speech_end


class ControlledBackend(TranslationBackend):
    def __init__(self):
        self.calls = []
        self.block = False
        self.completions = []
        self.failures = set()
        self.translations = {}

    async def translate(self, request):
        self.calls.append(request)
        if self.block:
            completion = asyncio.get_running_loop().create_future()
            self.completions.append(completion)
            await completion
        if request.text in self.failures:
            raise RuntimeError("controlled translation failure")
        return Translation(
            utterance_id=request.utterance_id,
            text=self.translations.get(request.text, f"translated:{request.text}"),
        )

    async def close(self):
        return None


async def turn_loop():
    for _ in range(12):
        await asyncio.sleep(0)


@pytest.fixture
def preview_timeline(monkeypatch):
    timeline = Timeline()
    original_sleep = asyncio.sleep

    async def sleep(delay):
        task = asyncio.current_task()
        if task is not None and task.get_name().startswith("PeerCaptionOwner:peer:timer:"):
            await timeline.sleep(delay)
        else:
            await original_sleep(delay)

    monkeypatch.setattr(asyncio, "sleep", sleep)
    return timeline


def setup(*, overlay_sink=None, clock=None, local_asr_provider_runtime_factory=None):
    backend = ControlledBackend()
    events = []

    async def emit(event):
        events.append(event)

    clock = clock or FakeClock()
    harness = compose_translation_test_harness(
        stt=None,
        llm=backend,
        osc=RecordingOscQueue(),
        clock=clock,
        overlay_sink=overlay_sink
        or SimpleNamespace(emit=emit, active_self_overlay_metadata=lambda: None),
        source_language="en",
        target_language="ko",
        system_prompt=get_default_prompt(),
        peer_translation_enabled=True,
        local_asr_provider_runtime_factory=local_asr_provider_runtime_factory,
    )
    return harness, backend, events, clock


class CommittedProviderRuntimeFactory:
    def __init__(self):
        self.providers = FakeProviderFactory()
        self.owner = None

    def create(self, callbacks):
        self.owner = LocalASRProviderRuntimeOwner(
            provider_factory=self.providers,
            gpu_runtime_factory=FakeGpuRuntimeFactory(),
            provisioning=FakeProvisioningPort(),
            self_event_handler=callbacks.self_event_handler,
            peer_event_handler=callbacks.peer_event_handler,
            retired_event_handler=callbacks.retired_event_handler,
            self_exception_handler=callbacks.self_exception_handler,
            peer_exception_handler=callbacks.peer_exception_handler,
        )
        return self.owner


def soniox_request(speaker_diarization):
    return ProviderRuntimeBuildRequest(
        config=replace(
            _resolved_config("peer", "soniox"),
            provider_options={"enable_speaker_diarization": speaker_diarization},
        )
    )


def transcript(text, order, *, identity=None, final=False):
    return Transcript(
        identity or uuid4(),
        text,
        final,
        channel="peer",
        publication_generation=1,
        source_order=order,
    )


async def partial(harness, item, provider="soniox"):
    await harness.peer_owner.cloud_captions.partial(
        item, provider_id=provider, scope=(1, 0, "epoch"), config=harness.configuration.snapshot()
    )


async def final(harness, item, provider="soniox", sealed=None):
    await harness.peer_owner.cloud_captions.final(
        replace(item, is_final=True),
        provider_id=provider,
        scope=(1, 0, "epoch"),
        config=harness.configuration.snapshot(),
        sealed=sealed,
    )


async def idle(harness):
    await harness.peer_owner.cloud_captions.wait_for_idle()
    await harness.output_runtime.wait_for_peer_output_idle()
    await turn_loop()


@pytest.mark.asyncio
async def test_preview_first_refresh_growth_gate_and_final_settlement():
    harness, backend, events, clock = setup()
    try:
        await harness.start()
        item = transcript("The observed beginning", 1)
        await partial(harness, item)
        clock.advance(4.99)
        await harness.peer_owner.cloud_captions.check_previews()
        assert backend.calls == []
        clock.advance(0.01)
        await harness.peer_owner.cloud_captions.check_previews()
        await idle(harness)
        assert len(backend.calls) == 1
        assert events[-1].type == "translation_stream_update"
        assert events[-1].peer_caption_revision == 1
        assert harness.peer_runtime.translation_history == []
        clock.advance(4)
        await harness.peer_owner.cloud_captions.check_previews()
        await idle(harness)
        assert len(backend.calls) == 1
        await partial(harness, replace(item, text="The observed beginning and later content"))
        clock.advance(4)
        await harness.peer_owner.cloud_captions.check_previews()
        await idle(harness)
        assert len(backend.calls) == 2
        assert events[-1].utterance_id == item.utterance_id
        assert events[-1].peer_caption_revision == 2
        await final(harness, replace(item, text="The complete corrected sentence."))
        await idle(harness)
        assert len(backend.calls) == 3
        assert events[-1].type == "translation_final"
        assert events[-1].peer_caption_revision == 3
        assert [entry.text for entry in harness.peer_runtime.translation_history] == [
            "The complete corrected sentence."
        ]
        assert harness.osc.messages == []
    finally:
        await harness.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("disabled", ["translation", "provider"])
@pytest.mark.parametrize("show_preview", [False, True], ids=["before-preview", "shown-preview"])
async def test_inactive_timer_waits_and_recovers_without_duplicate_previews(
    disabled, show_preview, preview_timeline
):
    initial_tasks = asyncio.all_tasks()
    harness, backend, events, clock = setup(clock=preview_timeline.clock)
    item = transcript("The observed beginning", 1)
    try:
        await harness.start()
        await partial(harness, item)
        await turn_loop()
        await preview_timeline.advance(4.99)
        assert backend.calls == []
        if show_preview:
            await preview_timeline.advance(0.01)
            await idle(harness)
            assert [request.text for request in backend.calls] == [item.text]
            item = replace(item, text="The observed beginning and later content")
            await partial(harness, item)
        calls_before = len(backend.calls)
        events_before = len(events)
        if disabled == "translation":
            harness.replace_configuration(translation_enabled=False)
        else:
            await harness.replace_llm_provider(None)
        await preview_timeline.advance(4 if show_preview else 0.01)
        for _ in range(3):
            waits = [
                deadline - clock.now()
                for deadline, future in preview_timeline.sleepers
                if not future.done()
            ]
            assert waits == pytest.approx([4.0])
            sleep_count = len(preview_timeline.sleepers)
            await turn_loop()
            assert len(preview_timeline.sleepers) == sleep_count
            assert len(backend.calls) == calls_before
            assert len(events) == events_before
            await preview_timeline.advance(4)
        if disabled == "translation":
            harness.replace_configuration(translation_enabled=True)
        else:
            await harness.replace_llm_provider(backend)
        await preview_timeline.advance(3.99)
        assert len(backend.calls) == calls_before
        await preview_timeline.advance(0.01)
        await idle(harness)
        assert len(backend.calls) == calls_before + 1
        assert events[-1].type == "translation_stream_update"
        assert events[-1].source_text == item.text
        assert events[-1].utterance_id == item.utterance_id
        assert any(
            decision.publication_id == events[-1].event_id
            and decision.metadata.get("publication_generation") == item.publication_generation
            and decision.metadata.get("source_order") == item.source_order
            for decision in harness.output_runtime.routing_decisions
        )
        assert events[-1].peer_caption_revision == calls_before + 1
        await preview_timeline.advance(4)
        await idle(harness)
        assert len(backend.calls) == calls_before + 1
        settled = replace(item, text="The authoritative final sentence.")
        await final(harness, settled)
        await idle(harness)
        assert len(backend.calls) == calls_before + 2
        assert events[-1].type == "translation_final"
        assert events[-1].source_text == settled.text
        assert events[-1].utterance_id == item.utterance_id
        assert any(
            decision.publication_id == events[-1].event_id
            and decision.metadata.get("source_order") == item.source_order
            for decision in harness.output_runtime.routing_decisions
        )
        await final(harness, settled)
        await partial(harness, item)
        await preview_timeline.advance(8)
        await idle(harness)
        assert len(backend.calls) == calls_before + 2
        assert [event.type for event in events] == [
            *(["translation_stream_update"] if show_preview else []),
            "translation_stream_update",
            "translation_final",
        ]
        assert harness.osc.messages == []
    finally:
        await asyncio.wait_for(harness.stop(), 1)
        await turn_loop()
        assert harness.peer_owner.cloud_captions._tasks == set()
        assert all(future.done() for _, future in preview_timeline.sleepers)
        assert not {task for task in asyncio.all_tasks() - initial_tasks if not task.done()}


@pytest.mark.asyncio
@pytest.mark.parametrize("disabled", ["translation", "provider"])
@pytest.mark.parametrize("queued_growth", [False, True], ids=["same-source", "latest-source"])
async def test_preview_waiting_for_capacity_stays_inactive_and_recovers(
    disabled, queued_growth, preview_timeline
):
    from puripuly_heart.core.orchestrator.peer_caption import PeerCaptionOwner

    harness, backend, events, clock = setup(clock=preview_timeline.clock)
    current = harness.peer_owner.cloud_captions
    harness.peer_owner.cloud_captions = PeerCaptionOwner(
        current.runtime, current.requests, current.output, clock, max_requests=1
    )
    item = transcript("Queued preview source", 2)
    try:
        await harness.start()
        backend.block = True
        await final(harness, transcript("Earlier final.", 1))
        await turn_loop()
        await partial(harness, item)
        await turn_loop()
        await preview_timeline.advance(5)
        assert [request.text for request in backend.calls] == ["Earlier final."]
        if queued_growth:
            item = replace(item, text="Queued preview source with newer observed content")
            await partial(harness, item)
            await preview_timeline.advance(4)
        if disabled == "translation":
            harness.replace_configuration(translation_enabled=False)
        else:
            await harness.replace_llm_provider(None)
        backend.block = False
        backend.completions[0].set_result(None)
        await idle(harness)
        assert [request.text for request in backend.calls] == ["Earlier final."]
        assert [event.source_text for event in events] == (
            ["Earlier final."] if disabled == "translation" else []
        )
        await preview_timeline.advance(4)
        assert len(backend.calls) == 1
        if disabled == "translation":
            harness.replace_configuration(translation_enabled=True)
        else:
            await harness.replace_llm_provider(backend)
        await preview_timeline.advance(4)
        await idle(harness)
        assert [request.text for request in backend.calls] == ["Earlier final.", item.text]
        assert events[-1].type == "translation_stream_update"
        assert events[-1].source_text == item.text
        assert events[-1].utterance_id == item.utterance_id
        assert any(
            decision.publication_id == events[-1].event_id
            and decision.metadata.get("source_order") == item.source_order
            for decision in harness.output_runtime.routing_decisions
        )
        assert events[-1].peer_caption_revision == 1
        await preview_timeline.advance(4)
        await idle(harness)
        assert len(backend.calls) == 2
        await final(harness, replace(item, text="Recovered final."))
        await idle(harness)
        assert [request.text for request in backend.calls] == [
            "Earlier final.",
            item.text,
            "Recovered final.",
        ]
        assert events[-1].type == "translation_final"
        assert events[-1].utterance_id == item.utterance_id
        assert harness.osc.messages == []
    finally:
        await asyncio.wait_for(harness.stop(), 1)
        await turn_loop()
        assert harness.peer_owner.cloud_captions._tasks == set()
        assert all(future.done() for _, future in preview_timeline.sleepers)


@pytest.mark.asyncio
async def test_provider_recovery_before_capacity_release_admits_current_preview(preview_timeline):
    from puripuly_heart.core.orchestrator.peer_caption import PeerCaptionOwner

    initial_tasks = asyncio.all_tasks()
    harness, backend, events, clock = setup(clock=preview_timeline.clock)
    current = harness.peer_owner.cloud_captions
    owner = harness.peer_owner.cloud_captions = PeerCaptionOwner(
        current.runtime, current.requests, current.output, clock, max_requests=1
    )
    item = transcript("Queued preview source", 2)
    try:
        await harness.start()
        backend.block = True
        await final(harness, transcript("Earlier final.", 1))
        await turn_loop()
        await partial(harness, item)
        await turn_loop()
        await preview_timeline.advance(5)
        assert [request.text for request in backend.calls] == ["Earlier final."]
        harness.peer_runtime.translation_history.append(
            ContextEntry("Later context.", "en", "ko", clock.now(), "peer")
        )
        await harness.replace_llm_provider(None)
        await preview_timeline.advance(4)
        await harness.replace_llm_provider(backend)
        await preview_timeline.advance(4)
        assert [request.text for request in backend.calls] == ["Earlier final."]
        backend.block = False
        backend.completions[0].set_result(None)
        await asyncio.wait_for(idle(harness), 1)
        assert [request.text for request in backend.calls] == ["Earlier final.", item.text]
        assert backend.calls[-1].context == '- [peer] "Earlier final."'
        assert [(event.type, event.source_text) for event in events] == [
            ("translation_stream_update", item.text)
        ]
        assert events[-1].utterance_id == item.utterance_id
        assert events[-1].peer_caption_revision == 1
        assert any(
            decision.publication_id == events[-1].event_id
            and decision.metadata.get("publication_generation") == item.publication_generation
            and decision.metadata.get("source_order") == item.source_order
            for decision in harness.output_runtime.routing_decisions
        )
        for _ in range(3):
            await preview_timeline.advance(4)
            await idle(harness)
        assert len(backend.calls) == 2
        assert owner.request_counts["preview"] == 1
        settled = replace(item, text="Recovered final.")
        await final(harness, settled)
        await asyncio.wait_for(idle(harness), 1)
        await final(harness, settled)
        await partial(harness, item)
        await preview_timeline.advance(4)
        await idle(harness)
        assert [request.text for request in backend.calls] == [
            "Earlier final.",
            item.text,
            settled.text,
        ]
        assert [(event.type, event.source_text) for event in events] == [
            ("translation_stream_update", item.text),
            ("translation_final", settled.text),
        ]
        assert events[-1].utterance_id == item.utterance_id
        assert events[-1].peer_caption_revision == 2
        assert harness.osc.messages == []
    finally:
        await asyncio.wait_for(harness.stop(), 1)
        await turn_loop()
        assert owner._tasks == set()
        assert all(future.done() for _, future in preview_timeline.sleepers)
        assert not {task for task in asyncio.all_tasks() - initial_tasks if not task.done()}


@pytest.mark.asyncio
@pytest.mark.parametrize("disabled", ["translation", "provider"])
@pytest.mark.parametrize("closing", ["retire", "reset", "stop"])
async def test_closing_inactive_caption_releases_timer_and_running_preview(
    disabled, closing, preview_timeline
):
    initial_tasks = asyncio.all_tasks()
    harness, backend, events, clock = setup(clock=preview_timeline.clock)
    item = transcript("Unsettled observed source", 1)
    owner = harness.peer_owner.cloud_captions
    try:
        await harness.start()
        backend.block = True
        await partial(harness, item)
        await turn_loop()
        await preview_timeline.advance(5)
        await turn_loop()
        assert len(backend.calls) == 1
        if disabled == "translation":
            harness.replace_configuration(translation_enabled=False)
        else:
            await harness.replace_llm_provider(None)
        await preview_timeline.advance(4)
        assert [
            deadline - clock.now()
            for deadline, future in preview_timeline.sleepers
            if not future.done()
        ] == pytest.approx([4])
        if closing == "retire":
            await owner.retire(item.utterance_id)
            await idle(harness)
        elif closing == "reset":
            await owner.reset()
        else:
            await harness.stop()
        await turn_loop()
        assert owner._segments == {}
        assert owner._tasks == set()
        assert backend.completions[0].cancelled()
        assert all(future.done() for _, future in preview_timeline.sleepers)
        assert not any(event.type == "translation_stream_update" for event in events)
        assert harness.osc.messages == []
    finally:
        await asyncio.wait_for(harness.stop(), 1)
        await turn_loop()
        assert not {task for task in asyncio.all_tasks() - initial_tasks if not task.done()}


@pytest.mark.asyncio
async def test_final_cancels_slow_preview_without_waiting_for_preview_timer():
    harness, backend, events, clock = setup()
    try:
        await harness.start()
        backend.block = True
        item = transcript("unfinished observed source", 1)
        await partial(harness, item)
        clock.advance(5)
        await harness.peer_owner.cloud_captions.check_previews()
        await turn_loop()
        assert len(backend.calls) == 1
        backend.block = False
        await final(harness, replace(item, text="Final source."))
        await idle(harness)
        assert [event.type for event in events] == ["translation_final"]
        assert backend.completions[0].cancelled()
        assert len(backend.calls) == 2
    finally:
        await harness.stop()


@pytest.mark.asyncio
async def test_merge_preserves_first_context_and_absorbs_preview_atomically(monkeypatch):
    monkeypatch.setattr(
        "puripuly_heart.core.orchestrator.peer_caption.estimate_peer_translation_lines",
        lambda *_, **__: 2.0,
    )
    harness, backend, events, clock = setup()
    try:
        await harness.start()
        harness.peer_runtime.translation_history.append(
            ContextEntry("Earlier.", "en", "ko", 0, "peer")
        )
        first = transcript("The first part,", 1)
        await final(harness, first)
        await idle(harness)
        clock.advance(0.5)
        second = transcript("and the second part", 2)
        await partial(harness, second)
        clock.advance(5)
        await harness.peer_owner.cloud_captions.check_previews()
        await idle(harness)
        assert events[-1].utterance_id == second.utterance_id
        backend.block = True
        await final(harness, second)
        await turn_loop()
        assert events[-1].peer_superseded_unit_ids == ()
        backend.completions[-1].set_result(None)
        await idle(harness)
        repair = events[-1]
        assert repair.type == "translation_final"
        assert repair.utterance_id == first.utterance_id
        assert repair.source_text == "The first part, and the second part"
        assert repair.peer_superseded_unit_ids == (second.utterance_id,)
        assert backend.calls[-1].context == '- [peer] "Earlier."'
        assert [entry.text for entry in harness.peer_runtime.translation_history] == [
            "Earlier.",
            repair.source_text,
        ]
    finally:
        await harness.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("r1_only", [False, True])
async def test_join_above_two_lines_uses_continuation_and_current_context(monkeypatch, r1_only):
    monkeypatch.setattr(
        "puripuly_heart.core.orchestrator.peer_caption.estimate_peer_translation_lines",
        lambda *_, **__: 3.0,
    )
    harness, backend, events, clock = setup()
    sealed = replace(
        owned_peer_speech_end(uuid4(), speech_end_at=1).segment, seal_reason="delivery_deadline"
    )
    try:
        await harness.start()
        first = transcript("A closed statement." if r1_only else "The earlier part,", 1)
        await final(harness, first, sealed=sealed)
        await idle(harness)
        clock.advance(0.3)
        second = transcript("The next content.", 2)
        await partial(harness, second)
        await final(harness, second)
        await idle(harness)
        assert events[-1].utterance_id == second.utterance_id
        assert backend.calls[-1].text == second.text
        assert "## Peer continuation" in backend.calls[-1].system_prompt
        assert backend.calls[-1].context == f'- [peer] "{first.text}"'
        assert [entry.text for entry in harness.peer_runtime.translation_history] == [
            first.text,
            second.text,
        ]
    finally:
        await harness.stop()


@pytest.mark.asyncio
async def test_r1_only_over_four_lines_is_independent(monkeypatch):
    monkeypatch.setattr(
        "puripuly_heart.core.orchestrator.peer_caption.estimate_peer_translation_lines",
        lambda *_, **__: 4.01,
    )
    harness, backend, events, clock = setup()
    sealed = replace(
        owned_peer_speech_end(uuid4(), speech_end_at=1).segment, seal_reason="delivery_deadline"
    )
    try:
        await harness.start()
        await final(harness, transcript("A closed statement.", 1), sealed=sealed)
        await idle(harness)
        clock.advance(0.5)
        second = transcript("The next statement.", 2)
        await partial(harness, second)
        await final(harness, second)
        await idle(harness)
        assert events[-1].utterance_id == second.utterance_id
        assert "## Peer continuation" not in backend.calls[-1].system_prompt
    finally:
        await harness.stop()


@pytest.mark.asyncio
async def test_chained_repairs_out_of_order_keep_cumulative_absorptions(monkeypatch):
    monkeypatch.setattr(
        "puripuly_heart.core.orchestrator.peer_caption.estimate_peer_translation_lines",
        lambda *_, **__: 2.0,
    )
    harness, backend, events, clock = setup()
    try:
        await harness.start()
        first = transcript("A,", 1)
        await final(harness, first)
        await idle(harness)
        backend.block = True
        second = transcript("second,", 2)
        await partial(harness, second)
        await final(harness, second)
        await turn_loop()
        third = transcript("third.", 3)
        await partial(harness, third)
        await final(harness, third)
        await turn_loop()
        backend.completions[1].set_result(None)
        await turn_loop()
        backend.completions[0].set_result(None)
        await idle(harness)
        assert len(events) == 2
        assert events[-1].source_text == "A, second, third."
        assert events[-1].peer_superseded_unit_ids == (second.utterance_id, third.utterance_id)
        assert [entry.text for entry in harness.peer_runtime.translation_history] == [
            events[-1].source_text
        ]
    finally:
        await harness.stop()


@pytest.mark.asyncio
async def test_preview_coalesces_one_running_one_latest_and_reset_fences_work():
    harness, backend, events, clock = setup()
    try:
        await harness.start()
        backend.block = True
        item = transcript("first observed content", 1)
        await partial(harness, item)
        for suffix in ("", " longer", " longer newest"):
            await partial(harness, replace(item, text=item.text + suffix))
            clock.advance(5)
            await harness.peer_owner.cloud_captions.check_previews()
            await turn_loop()
        assert len(backend.calls) == 1
        backend.completions[0].set_result(None)
        await turn_loop()
        assert len(backend.calls) == 2
        assert backend.calls[-1].text.endswith("newest")
        await harness.peer_owner.cloud_captions.reset()
        assert backend.completions[1].cancelled()
        assert harness.peer_runtime.translation_history == []
        assert len(events) <= 1
    finally:
        await harness.stop()


@pytest.mark.asyncio
async def test_owned_provider_ingress_through_output_contract():
    harness, backend, events, clock = setup()
    try:
        await harness.start()
        item = transcript("Observed provisional speech", 1)
        owned = owned_peer_speech_end(item.utterance_id, speech_end_at=9)
        segment = replace(
            owned.segment, settings=replace(owned.segment.settings, provider_id="soniox")
        )
        identity = STTProviderTurnIdentity(segment.identity, "epoch", "turn", ("soniox",))
        await harness.peer_owner.handle_provider_update(
            STTProviderTurnUpdate(identity, 1, "provisional", "replace", item.text, observed_at=0.0)
        )
        clock.advance(5)
        await harness.peer_owner.cloud_captions.check_previews()
        await idle(harness)
        assert events[-1].type == "translation_stream_update"
        receipt = AudioSegmentTerminalReceipt(segment.identity, "final", segment, 9.0)
        await harness.peer_owner.handle_provider_turn_terminal(
            receipt,
            STTProviderTurnTerminal(
                identity, "final", "Authoritative corrected speech.", text_authority="authoritative"
            ),
        )
        await idle(harness)
        assert events[-1].type == "translation_final"
        assert events[0].utterance_id == events[-1].utterance_id
        assert [entry.text for entry in harness.peer_runtime.translation_history] == [
            "Authoritative corrected speech."
        ]
        assert harness.osc.messages == []
    finally:
        await harness.stop()


def test_preview_boundary_selection_and_reply_gap_stops():
    assert (
        preview_source("a short, a much longer unfinished observation")
        == "a short, a much longer unfinished observation"
    )
    assert preview_source("A sufficiently long clause, tail") == "A sufficiently long clause,"
    for gap, incoming in ((None, "continuation"), (1.51, "continuation"), (0.2, "yeah okay")):
        assert (
            join_action(
                "the first,",
                incoming,
                language="en",
                gap_s=gap,
                expected_lines=1,
                provider_id="soniox",
                sealed=None,
            )
            == "independent"
        )


def test_gemini_ignores_acoustic_signals_and_question_stops_r1():
    sealed = replace(
        owned_peer_speech_end(uuid4(), speech_end_at=1).segment, seal_reason="delivery_deadline"
    )
    assert (
        join_action(
            "Completed.",
            "Next",
            language="en",
            gap_s=0.2,
            expected_lines=2,
            provider_id="gemini_transcribe",
            sealed=sealed,
        )
        == "independent"
    )
    assert (
        join_action(
            "Completed?",
            "Next",
            language="en",
            gap_s=0.2,
            expected_lines=2,
            provider_id="soniox",
            sealed=sealed,
        )
        == "independent"
    )


@pytest.mark.asyncio
async def test_continuation_caption_can_merge_and_continue_with_its_original_context(monkeypatch):
    estimate = [3.0]
    monkeypatch.setattr(
        "puripuly_heart.core.orchestrator.peer_caption.estimate_peer_translation_lines",
        lambda *_, **__: estimate[0],
    )
    harness, backend, events, clock = setup()
    try:
        await harness.start()
        first = transcript("The beginning,", 1)
        await final(harness, first)
        await idle(harness)
        second = transcript("the middle,", 2)
        await partial(harness, second)
        await final(harness, second)
        await idle(harness)
        assert "## Peer continuation" in backend.calls[-1].system_prompt
        estimate[0] = 2.0
        third = transcript("and more,", 3)
        await partial(harness, third)
        await final(harness, third)
        await idle(harness)
        assert events[-1].utterance_id == second.utterance_id
        assert backend.calls[-1].text == "the middle, and more,"
        assert "## Peer continuation" not in backend.calls[-1].system_prompt
        assert backend.calls[-1].context == f'- [peer] "{first.text}"'
        estimate[0] = 3.0
        fourth = transcript("the end.", 4)
        await partial(harness, fourth)
        await final(harness, fourth)
        await idle(harness)
        assert events[-1].utterance_id == fourth.utterance_id
        assert "## Peer continuation" in backend.calls[-1].system_prompt
        assert backend.calls[-1].context.splitlines()[-1] == '- [peer] "the middle, and more,"'
        assert [entry.text for entry in harness.peer_runtime.translation_history] == [
            first.text,
            "the middle, and more,",
            fourth.text,
        ]
    finally:
        await harness.stop()


@pytest.mark.asyncio
async def test_independent_finals_finishing_reverse_keep_repetition_and_ordered_source_history():
    harness, backend, events, clock = setup()
    try:
        await harness.start()
        backend.block = True
        first = transcript("Repeated wording.", 1)
        second = transcript(first.text, 2)
        await final(harness, first)
        await final(harness, second)
        await turn_loop()
        backend.completions[1].set_result(None)
        await turn_loop()
        assert [event.utterance_id for event in events] == [second.utterance_id]
        backend.completions[0].set_result(None)
        await idle(harness)
        assert [event.utterance_id for event in events] == [second.utterance_id, first.utterance_id]
        assert [call.utterance_id for call in backend.calls] == [
            first.utterance_id,
            second.utterance_id,
        ]
        ui_events = []
        while not harness.ui_events.empty():
            ui_events.append(harness.ui_events.get_nowait())
        assert [
            event.utterance_id for event in ui_events if event.type is UIEventType.TRANSLATION_DONE
        ] == [
            first.utterance_id,
            second.utterance_id,
        ]
        assert [entry.text for entry in harness.peer_runtime.translation_history] == [
            first.text,
            second.text,
        ]
        assert harness.peer_owner.cloud_captions.request_counts == {
            "preview": 0,
            "final": 2,
            "repair": 0,
            "continuation": 0,
        }
    finally:
        await harness.stop()


@pytest.mark.asyncio
async def test_context_is_five_recent_peer_finals_and_preview_context_is_frozen():
    harness, backend, events, clock = setup()
    try:
        await harness.start()
        clock.advance(61)
        harness.peer_runtime.translation_history.append(
            ContextEntry("Expired.", "en", "ko", 0, "peer")
        )
        for index in range(7):
            harness.peer_runtime.translation_history.append(
                ContextEntry(f"Recent {index}.", "en", "ko", 7, "peer")
            )
        harness.self_runtime.translation_history.append(
            ContextEntry("Private Self.", "en", "ko", 60)
        )
        item = transcript("Observed incomplete", 1)
        await partial(harness, item)
        clock.advance(5)
        await harness.peer_owner.cloud_captions.check_previews()
        await idle(harness)
        assert len(backend.calls) == 1
        assert backend.calls[-1].text == item.text
        expected = "\n".join(f'- [peer] "Recent {index}."' for index in range(2, 7))
        assert backend.calls[-1].context == expected
        harness.peer_runtime.translation_history.append(
            ContextEntry("Later context.", "en", "ko", clock.now(), "peer")
        )
        growing = replace(item, text="Observed incomplete source with more detail")
        await partial(harness, growing)
        clock.advance(4)
        await harness.peer_owner.cloud_captions.check_previews()
        await idle(harness)
        assert clock.now() - 7 > 60
        assert len(backend.calls) == 2
        assert backend.calls[-1].text == growing.text
        assert backend.calls[-1].context == expected
        await final(harness, replace(item, text="Authoritative final."))
        await idle(harness)
        assert len(backend.calls) == 3
        assert backend.calls[-1].text == "Authoritative final."
        assert backend.calls[-1].context == expected
        ui = []
        while not harness.ui_events.empty():
            ui.append(harness.ui_events.get_nowait())
        assert len([event for event in ui if event.type is UIEventType.TRANSLATION_DONE]) == 1
        await final(harness, transcript("A new independent sentence.", 2))
        await idle(harness)
        assert len(backend.calls) == 4
        assert backend.calls[-1].text == "A new independent sentence."
        assert backend.calls[-1].context == (
            '- [peer] "Later context."\n- [peer] "Authoritative final."'
        )
    finally:
        await harness.stop()


@pytest.mark.asyncio
async def test_qwen_empty_partial_retains_last_nonempty_and_text_capacity_is_explicit():
    harness, backend, events, clock = setup()
    try:
        await harness.start()
        item = transcript("Observed content", 1)
        await partial(harness, item, provider="qwen_audio")
        await partial(harness, replace(item, text=""), provider="qwen_audio")
        clock.advance(5)
        await harness.peer_owner.cloud_captions.check_previews()
        await idle(harness)
        assert backend.calls[-1].text == item.text
        harness.peer_owner.cloud_captions.max_source_chars = 20
        with pytest.raises(ValueError, match="text capacity exceeded"):
            await partial(harness, replace(item, text="x" * 21))
        with pytest.raises(ValueError, match="text capacity exceeded"):
            await final(harness, replace(item, text="x" * 21))
        assert harness.peer_runtime.translation_history == []
    finally:
        await harness.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("configured_source", "expected_name"),
    [("zh-CN", "Chinese (Simplified)"), ("zh-TW", "Chinese (Traditional)"), ("ja", "Japanese")],
)
async def test_manual_regional_source_translates_final_without_detected_language(
    configured_source, expected_name
):
    harness, backend, events, _ = setup()
    harness.configuration.replace(
        replace(
            harness.configuration.snapshot().value,
            peer_source_mode="manual",
            peer_source_language=configured_source,
            peer_target_language="ko",
        )
    )
    try:
        await harness.start()
        await final(harness, transcript("你好。", 1), provider="qwen_audio")
        await idle(harness)
        assert [request.text for request in backend.calls] == ["你好。"]
        assert backend.calls[0].source_language == configured_source
        assert expected_name in backend.calls[0].system_prompt
        assert events[-1].type == "translation_final"
        assert events[-1].text == "translated:你好。"
        assert events[-1].source_language == configured_source
    finally:
        await harness.stop()


@pytest.mark.asyncio
async def test_local_peer_uses_unchanged_per_final_translation_without_preview_or_join(monkeypatch):
    harness, backend, events, clock = setup()
    monkeypatch.setattr(
        type(harness.peer_owner), "_current_provider_id", lambda _: "local_cpu_auto"
    )
    try:
        await harness.start()
        first = transcript("An incomplete source,", 1)
        from puripuly_heart.domain.events import STTPartialEvent

        await harness.peer_owner.handle_stt_event(STTPartialEvent(first.utterance_id, first))
        clock.advance(10)
        await harness.peer_owner.cloud_captions.check_previews()
        assert backend.calls == []
        await harness.peer_owner.handle_stt_event(
            STTFinalEvent(first.utterance_id, replace(first, is_final=True))
        )
        await harness.translation_turns.wait_for_idle()
        second = transcript("the next local final", 2)
        await harness.peer_owner.handle_stt_event(
            STTFinalEvent(second.utterance_id, replace(second, is_final=True))
        )
        await harness.translation_turns.wait_for_idle()
        await harness.output_runtime.wait_for_peer_output_idle()
        assert [call.text for call in backend.calls] == [first.text, second.text]
        assert all("## Peer provisional" not in call.system_prompt for call in backend.calls)
        assert all("## Peer continuation" not in call.system_prompt for call in backend.calls)
        assert harness.peer_owner.cloud_captions.request_counts == dict.fromkeys(
            ("preview", "final", "repair", "continuation"), 0
        )
    finally:
        await harness.stop()


@pytest.mark.asyncio
async def test_provider_replacement_discards_inflight_caption_and_fresh_work_progresses():
    harness, backend, events, clock = setup()
    try:
        await harness.start()
        backend.block = True
        await final(harness, transcript("Retired request.", 1))
        await turn_loop()
        replacement = ControlledBackend()
        await harness.replace_llm_provider(replacement)
        backend.completions[0].set_result(None)
        await idle(harness)
        assert events == []
        await final(harness, transcript("Current request.", 2))
        await idle(harness)
        assert events[-1].source_text == "Current request."
    finally:
        await harness.stop()


def test_reply_punctuation_and_overlapping_delivery_gap_follow_join_rules():
    assert (
        join_action(
            "earlier,",
            "yeah, okay!",
            language="en",
            gap_s=0.2,
            expected_lines=1,
            provider_id="soniox",
            sealed=None,
        )
        == "independent"
    )
    assert (
        join_action(
            "earlier,",
            "continued content",
            language="en",
            gap_s=-0.1,
            expected_lines=1,
            provider_id="soniox",
            sealed=None,
        )
        == "merge"
    )


@pytest.mark.asyncio
async def test_shutdown_ingress_drain_keeps_admitted_final_and_cancels_open_preview():
    harness, backend, events, clock = setup()
    try:
        await harness.start()
        backend.block = True
        await final(harness, transcript("Admitted final.", 1))
        await turn_loop()
        await partial(harness, transcript("Unsettled open source", 2))
        draining = asyncio.create_task(harness.peer_owner.close_ingress())
        await turn_loop()
        assert not draining.done()
        backend.completions[0].set_result(None)
        await draining
        await harness.output_runtime.wait_for_peer_output_idle()
        assert [event.source_text for event in events] == ["Admitted final."]
        assert [entry.text for entry in harness.peer_runtime.translation_history] == [
            "Admitted final."
        ]
    finally:
        await harness.stop()


@pytest.mark.asyncio
async def test_request_capacity_backpressures_without_losing_final_sources():
    from puripuly_heart.core.orchestrator.peer_caption import PeerCaptionOwner

    harness, backend, events, clock = setup()
    current = harness.peer_owner.cloud_captions
    harness.peer_owner.cloud_captions = PeerCaptionOwner(
        current.runtime, current.requests, current.output, clock, max_requests=2
    )
    third_admission = None
    try:
        await harness.start()
        backend.block = True
        for order in (1, 2):
            await final(harness, transcript(f"Final {order}.", order))
        third_admission = asyncio.create_task(final(harness, transcript("Final 3.", 3)))
        await turn_loop()
        assert len(backend.calls) == 2 and not third_admission.done()
        backend.completions[0].set_result(None)
        await third_admission
        await turn_loop()
        assert len(backend.calls) == 3
        backend.completions[1].set_result(None)
        backend.completions[2].set_result(None)
        await idle(harness)
        assert [event.source_text for event in events] == ["Final 1.", "Final 2.", "Final 3."]
        assert [entry.text for entry in harness.peer_runtime.translation_history] == [
            "Final 1.",
            "Final 2.",
            "Final 3.",
        ]
    finally:
        if third_admission is not None and not third_admission.done():
            third_admission.cancel()
            await asyncio.gather(third_admission, return_exceptions=True)
        await harness.stop()


@pytest.mark.asyncio
async def test_translation_timeout_preserves_source_and_subsequent_progress():
    harness, backend, events, clock = setup()
    harness.peer_owner.cloud_captions.request_timeout_s = 0.01
    try:
        await harness.start()
        backend.block = True
        await final(harness, transcript("Timeout source.", 1))
        await idle(harness)
        assert events[0].type == "translation_final"
        assert events[0].text == "" and events[0].source_text == "Timeout source."
        assert backend.completions[0].cancelled()
        backend.block = False
        await final(harness, transcript("Recovered source.", 2))
        await idle(harness)
        assert events[-1].type == "translation_final"
        assert events[-1].source_text == "Recovered source."
        assert [entry.text for entry in harness.peer_runtime.translation_history] == [
            "Timeout source.",
            "Recovered source.",
        ]
    finally:
        await harness.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("detected_language", ["unsupported", "xx"])
async def test_unsupported_final_fragment_settles_source_and_releases_sibling_and_successor(
    detected_language, monkeypatch
):
    harness, backend, events, clock = setup()
    submissions = []
    projection = harness.peer_owner.output_projection
    original = type(projection).project_peer_caption

    async def record(owner, submission, **kwargs):
        submissions.append(submission)
        await original(owner, submission, **kwargs)

    monkeypatch.setattr(type(projection), "project_peer_caption", record)
    try:
        await harness.start()
        item = replace(
            transcript("bad good", 1, final=True),
            final_language_runs=(
                FinalLanguageRun("bad ", detected_language),
                FinalLanguageRun("good", "en"),
            ),
        )
        owned = owned_peer_speech_end(item.utterance_id, speech_end_at=0)
        segment = replace(
            owned.segment, settings=replace(owned.segment.settings, provider_id="soniox")
        )
        receipt = AudioSegmentTerminalReceipt(segment.identity, "final", segment, 0.0)
        terminal = STTProviderTurnTerminal(
            STTProviderTurnIdentity(receipt.identity, "epoch", "mixed", ("soniox",)),
            "final",
            item.text,
            final_language_runs=item.final_language_runs,
            text_authority="authoritative",
        )
        await harness.peer_owner.handle_provider_turn_terminal(receipt, terminal)
        await harness.peer_owner.handle_provider_turn_terminal(receipt, terminal)
        await final(harness, transcript("A valid successor.", 2))
        await asyncio.wait_for(idle(harness), 1)
        assert [(item.source_text, item.outcome, item.failure_code) for item in submissions] == [
            ("bad ", "source_only", "unsupported_source_language"),
            ("good", "translated", None),
            ("A valid successor.", "translated", None),
        ]
        assert [request.text for request in backend.calls] == ["good", "A valid successor."]
        assert [(event.source_text, event.text) for event in events] == [
            ("bad ", ""),
            ("good", "translated:good"),
            ("A valid successor.", "translated:A valid successor."),
        ]
        assert all(event.peer_caption_revision == 1 for event in events)
        assert harness.osc.messages == []
    finally:
        await asyncio.wait_for(harness.stop(), 1)


@pytest.mark.asyncio
@pytest.mark.parametrize("retirement", ["empty", "epoch"])
@pytest.mark.parametrize("show_preview", [False, True], ids=["unrequested", "shown"])
async def test_native_empty_or_epoch_retirement_closes_preview_and_unblocks_successor(
    retirement, show_preview
):
    harness, backend, events, clock = setup()
    callbacks = TranslationChannelOwnerCallbacks(harness.stt_sessions)
    callbacks.bind_peer(harness.peer_owner)
    stream = RecognitionStreamIdentity("peer", 1, 0, "native-epoch", ("gemini_transcribe",))
    current = [stream]
    callbacks._peer_capture = SimpleNamespace(
        is_current_recognition_stream=lambda identity: identity == current[0]
    )
    harness.peer_owner.local_asr_runtime = SimpleNamespace(
        snapshot=harness.peer_owner.local_asr_runtime.snapshot,
        is_current_recognition_stream=lambda _channel, identity: identity == current[0],
    )
    unit = STTRecognitionUnit(
        RecognitionUnitIdentity(stream, uuid4(), 1),
        "Retracted partial",
        is_final=False,
        observed_at=clock.now(),
    )
    try:
        await harness.start()
        await callbacks.peer_event_handler(unit)
        if show_preview:
            clock.advance(5)
            await harness.peer_owner.cloud_captions.check_previews()
            await idle(harness)
            assert events[-1].type == "translation_stream_update"
        if retirement == "empty":
            terminal = STTRecognitionUnitTerminal(replace(unit, text="", is_final=True), "empty")
        else:
            terminal = STTProviderEpochEnded(stream.provider_epoch_id, True, "go_away")
        await callbacks.peer_event_handler(terminal)
        await callbacks.peer_event_handler(terminal)
        await callbacks.peer_event_handler(unit)
        assert harness.peer_owner.cloud_captions._segments == {}
        assert harness.peer_owner._native_caption_orders == {}
        current[0] = replace(stream, provider_epoch_id="successor-epoch")
        successor = STTRecognitionUnit(
            RecognitionUnitIdentity(current[0], uuid4(), 2), "Valid successor."
        )
        await callbacks.peer_event_handler(STTRecognitionUnitTerminal(successor, "final"))
        await asyncio.wait_for(idle(harness), 1)
        assert [event.source_text for event in events if event.type == "translation_final"] == [
            "",
            "Valid successor.",
        ]
        assert [request.text for request in backend.calls] == (
            ["Retracted partial", "Valid successor."] if show_preview else ["Valid successor."]
        )
        clock.advance(20)
        await harness.peer_owner.cloud_captions.check_previews()
        await idle(harness)
        for order in range(3, 9):
            pending = STTRecognitionUnit(
                RecognitionUnitIdentity(current[0], uuid4(), order),
                "Another withdrawn source",
                is_final=False,
                observed_at=clock.now(),
            )
            await callbacks.peer_event_handler(pending)
            await callbacks.peer_event_handler(
                STTRecognitionUnitTerminal(replace(pending, text="", is_final=True), "empty")
            )
        await asyncio.wait_for(idle(harness), 1)
        assert harness.peer_owner.cloud_captions._segments == {}
        assert harness.peer_owner._native_caption_orders == {}
        assert [entry.text for entry in harness.peer_runtime.translation_history] == [
            "Valid successor."
        ]
    finally:
        await asyncio.wait_for(harness.stop(), 1)


@pytest.mark.asyncio
@pytest.mark.parametrize("recognizer_id", [None, "gemini_transcribe"], ids=["alias", "member"])
async def test_rolling_free_partials_and_finals_share_the_caption_route(recognizer_id):
    harness, backend, events, clock = setup()
    callbacks = TranslationChannelOwnerCallbacks(harness.stt_sessions)
    callbacks.bind_peer(harness.peer_owner)
    stream = RecognitionStreamIdentity(
        "peer", 1, 0, "rolling-epoch", ("rolling_free",), recognizer_id
    )
    callbacks._peer_capture = SimpleNamespace(
        is_current_recognition_stream=lambda identity: identity == stream
    )
    harness.peer_owner.local_asr_runtime = SimpleNamespace(
        snapshot=harness.peer_owner.local_asr_runtime.snapshot,
        is_current_recognition_stream=lambda _channel, identity: identity == stream,
    )
    sources = [f"Sentence number {order}." for order in range(1, 6)]
    try:
        await harness.start()
        for order, source in enumerate(sources, start=1):
            unit_id = uuid4()
            await callbacks.peer_event_handler(
                STTRecognitionUnit(
                    RecognitionUnitIdentity(stream, unit_id, order * 2 - 1),
                    source[:8],
                    is_final=False,
                    observed_at=clock.now(),
                )
            )
            await callbacks.peer_event_handler(
                STTRecognitionUnitTerminal(
                    STTRecognitionUnit(RecognitionUnitIdentity(stream, unit_id, order * 2), source),
                    "final",
                )
            )
            await asyncio.wait_for(idle(harness), 1)
            assert harness.peer_owner.cloud_captions._segments == {}
        finals = [event for event in events if event.type == "translation_final"]
        assert [event.source_text for event in finals] == sources
        assert all(event.peer_caption_revision >= 1 for event in finals)
        assert [request.text for request in backend.calls] == sources
    finally:
        await asyncio.wait_for(harness.stop(), 1)


def rolling_peer(harness, stream):
    callbacks = TranslationChannelOwnerCallbacks(harness.stt_sessions)
    callbacks.bind_peer(harness.peer_owner)
    callbacks._peer_capture = SimpleNamespace(
        is_current_recognition_stream=lambda identity: identity == stream
    )
    harness.peer_owner.local_asr_runtime = SimpleNamespace(
        snapshot=harness.peer_owner.local_asr_runtime.snapshot,
        is_current_recognition_stream=lambda _channel, identity: identity == stream,
    )
    return callbacks


@pytest.mark.asyncio
async def test_rolling_member_switch_keeps_turn_partial_and_final_on_one_publication_order():
    harness, backend, events, clock = setup()
    harness.output_runtime.activate_peer_generation(1)
    gemini = RecognitionStreamIdentity(
        "peer", 1, 0, "gemini-epoch", ("rolling_free",), "gemini_transcribe"
    )
    callbacks = rolling_peer(harness, gemini)
    gemini_sources = [f"Gemini sentence {order}." for order in range(1, 6)]
    scribe_sources = ["Scribe sentence one.", "Scribe sentence two."]
    try:
        await harness.start()
        for order, source in enumerate(gemini_sources):
            unit_id = uuid4()
            await callbacks.peer_event_handler(
                STTRecognitionUnit(
                    RecognitionUnitIdentity(gemini, unit_id, order * 2 + 1),
                    source[:8],
                    is_final=False,
                    observed_at=clock.now(),
                )
            )
            await callbacks.peer_event_handler(
                STTRecognitionUnitTerminal(
                    STTRecognitionUnit(
                        RecognitionUnitIdentity(gemini, unit_id, order * 2 + 2), source
                    ),
                    "final",
                )
            )
            await asyncio.wait_for(idle(harness), 1)
        for segment_order, source in enumerate(scribe_sources, start=2):
            owned = owned_peer_speech_end(uuid4(), speech_end_at=20.0 + segment_order)
            segment = replace(
                owned.segment,
                identity=replace(owned.segment.identity, segment_order=segment_order),
            )
            identity = STTProviderTurnIdentity(
                segment.identity,
                "scribe-epoch",
                f"turn-{segment_order}",
                ("rolling_free",),
                "elevenlabs_scribe",
            )
            await harness.peer_owner.handle_provider_update(
                STTProviderTurnUpdate(
                    identity, 1, "provisional", "replace", source[:13], observed_at=clock.now()
                )
            )
            clock.advance(5)
            await harness.peer_owner.cloud_captions.check_previews()
            await asyncio.wait_for(idle(harness), 1)
            await harness.peer_owner.handle_provider_turn_terminal(
                AudioSegmentTerminalReceipt(segment.identity, "final", segment, clock.now()),
                STTProviderTurnTerminal(identity, "final", source, text_authority="authoritative"),
            )
            await asyncio.wait_for(idle(harness), 1)
        finals = [event.source_text for event in events if event.type == "translation_final"]
        assert finals == [*gemini_sources, *scribe_sources]
        assert harness.peer_owner.cloud_captions._segments == {}
    finally:
        await asyncio.wait_for(harness.stop(), 1)


@pytest.mark.asyncio
@pytest.mark.parametrize("disabled", ["translation", "provider"])
@pytest.mark.parametrize("show_preview", [False, True], ids=["before-preview", "shown-preview"])
async def test_final_after_translation_stops_closes_its_open_caption(
    disabled, show_preview, monkeypatch, preview_timeline
):
    harness, backend, events, clock = setup(clock=preview_timeline.clock)
    stream = RecognitionStreamIdentity(
        "peer", 1, 0, "rolling-epoch", ("rolling_free",), "gemini_transcribe"
    )
    callbacks = rolling_peer(harness, stream)
    available = [True]
    monkeypatch.setattr(
        type(harness.peer_owner.translation_requests),
        "provider_available",
        property(lambda _owner: available[0]),
    )
    try:
        await harness.start()
        for order in range(1, 5):
            unit_id = uuid4()
            await callbacks.peer_event_handler(
                STTRecognitionUnit(
                    RecognitionUnitIdentity(stream, unit_id, order * 2 - 1),
                    f"partial {order}",
                    is_final=False,
                    observed_at=clock.now(),
                )
            )
            await turn_loop()
            if show_preview:
                await preview_timeline.advance(5)
                await idle(harness)
                assert events[-1].type == "translation_stream_update"
            timer = harness.peer_owner.cloud_captions._segments[unit_id].timer
            if disabled == "translation":
                harness.replace_configuration(translation_enabled=False)
            else:
                available[0] = False
            calls_before = len(backend.calls)
            await preview_timeline.advance(6)
            await callbacks.peer_event_handler(
                STTRecognitionUnitTerminal(
                    STTRecognitionUnit(
                        RecognitionUnitIdentity(stream, unit_id, order * 2), f"Final {order}."
                    ),
                    "final",
                )
            )
            await preview_timeline.advance(6)
            await asyncio.wait_for(idle(harness), 1)
            assert harness.peer_owner.cloud_captions._segments == {}
            assert timer.done()
            assert not any(
                request.text.startswith("partial") for request in backend.calls[calls_before:]
            )
            harness.replace_configuration(translation_enabled=True)
            available[0] = True
    finally:
        await asyncio.wait_for(harness.stop(), 1)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["error", "timeout"])
@pytest.mark.parametrize("repair", [False, True], ids=["settlement", "repair-absorption"])
async def test_failed_or_timed_out_version_clears_preview_and_preserves_authoritative_source(
    failure, repair, monkeypatch
):
    clock = FakeClock()
    presenter = OverlayPresenter(
        calibration=OverlayCalibration(), clock=clock, show_peer_original=False
    )
    harness, backend, _, clock = setup(overlay_sink=presenter, clock=clock)
    submissions = []
    projection = harness.peer_owner.output_projection
    original = type(projection).project_peer_caption

    async def record(owner, submission, **kwargs):
        submissions.append(submission)
        await original(owner, submission, **kwargs)

    monkeypatch.setattr(type(projection), "project_peer_caption", record)
    try:
        await harness.start()
        if repair:
            first = transcript("A,", 1)
            await final(harness, first)
            await idle(harness)
            clock.advance(0.3)
            item = transcript("second", 2)
            authoritative = "A, second"
        else:
            item = transcript("Incomplete preview", 1)
            authoritative = "Correct authoritative source."
        await partial(harness, item)
        clock.advance(5)
        await harness.peer_owner.cloud_captions.check_previews()
        await idle(harness)
        assert any("translated:" in block.primary_text for block in presenter.snapshot().blocks)
        if failure == "timeout":
            backend.block = True
            harness.peer_owner.cloud_captions.request_timeout_s = 0.01
        else:
            backend.failures.add(authoritative)
        await final(harness, item if repair else replace(item, text=authoritative))
        await asyncio.wait_for(idle(harness), 1)
        result = submissions[-1]
        assert result.failure_code == (
            "translation_timeout" if failure == "timeout" else "provider_error"
        )
        assert result.source_text == authoritative and result.translation is None
        assert result.peer_caption_revision == 2
        assert result.peer_superseded_unit_ids == ((item.utterance_id,) if repair else ())
        blocks = presenter.snapshot().blocks
        assert len(blocks) == 1 and blocks[0].primary_text == authoritative
        assert blocks[0].id == f"peer:{first.utterance_id if repair else item.utterance_id}"
        assert not any(
            decision.reason == "stale_source_order"
            for decision in harness.output_runtime.routing_decisions
        )
        assert [entry.text for entry in harness.peer_runtime.translation_history] == [authoritative]
        if failure == "timeout":
            assert backend.completions[-1].cancelled()
    finally:
        await harness.stop()
        await presenter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider", ["soniox", "deepgram", "gemini_transcribe", "elevenlabs_scribe"]
)
async def test_complete_partial_retraction_prevents_non_qwen_preview(provider):
    harness, backend, events, clock = setup()
    try:
        await harness.start()
        item = transcript("Withdrawn cumulative hypothesis", 1)
        await partial(harness, item, provider=provider)
        await partial(harness, replace(item, text=""), provider=provider)
        clock.advance(5)
        await harness.peer_owner.cloud_captions.check_previews()
        await idle(harness)
        assert backend.calls == [] and events == []
        await final(harness, replace(item, text="Supported later final."), provider=provider)
        await asyncio.wait_for(idle(harness), 1)
        assert [request.text for request in backend.calls] == ["Supported later final."]
    finally:
        await harness.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider", "speaker_diarization", "preview_style", "final_style"),
    [
        ("qwen_audio", False, "gold", "gold"),
        ("soniox", False, "gold", "gold"),
        ("deepgram", False, "gold", "gold"),
        ("gemini_transcribe", False, "gold", "gold"),
        ("elevenlabs_scribe", False, "gold", "gold"),
        ("soniox", True, "gray", "gold"),
    ],
)
async def test_long_speech_preview_keeps_non_diarized_peer_caption_gold(
    provider, speaker_diarization, preview_style, final_style
):
    clock = FakeClock()
    presenter = OverlayPresenter(calibration=OverlayCalibration(), clock=clock)
    harness, backend, _, clock = setup(overlay_sink=presenter, clock=clock)
    harness.peer_owner.cloud_captions.speaker_diarization = (
        lambda provider_id: speaker_diarization and provider_id == "soniox"
    )
    try:
        await harness.start()
        item = transcript("We are going to meet at the station later today", 1)
        await partial(harness, item, provider=provider)
        clock.advance(5)
        await harness.peer_owner.cloud_captions.check_previews()
        await idle(harness)
        assert [request.text for request in backend.calls] == [item.text]
        assert [block.speaker_style for block in presenter.snapshot().blocks] == [preview_style]
        if speaker_diarization:
            item = replace(
                item,
                is_final=True,
                final_speaker_runs=(
                    FinalSpeakerRun(
                        item.text,
                        "1",
                        "soniox-scope",
                        source="soniox",
                        attribution_state="identified",
                    ),
                ),
            )
        await final(harness, item, provider=provider)
        await idle(harness)
        assert [block.speaker_style for block in presenter.snapshot().blocks] == [final_style]
    finally:
        await harness.stop()
        await presenter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("committed", "candidate", "preview_style"),
    [
        (True, None, "gray"),
        (False, None, "gold"),
        (True, "pending", "gray"),
        (True, "cancelled", "gray"),
    ],
)
async def test_peer_preview_style_follows_committed_provider_diarization(
    committed, candidate, preview_style
):
    clock = FakeClock()
    presenter = OverlayPresenter(calibration=OverlayCalibration(), clock=clock)
    runtime_factory = CommittedProviderRuntimeFactory()
    harness, backend, _, clock = setup(
        overlay_sink=presenter,
        clock=clock,
        local_asr_provider_runtime_factory=runtime_factory,
    )
    owner = runtime_factory.owner
    handoff = None
    try:
        await harness.start()
        await owner.replace_provider(soniox_request(committed), start=False)
        if candidate is not None:
            runtime_factory.providers.providers[-1].is_at_utterance_boundary = False
            handoff = asyncio.create_task(
                owner.handoff_provider(soniox_request(not committed), start=False)
            )
            await _wait_until(lambda: owner.snapshot.channel_for("peer").pending_handoff)
        if candidate == "cancelled":
            handoff.cancel()
            await asyncio.wait_for(asyncio.gather(handoff, return_exceptions=True), 1)
            await owner.commit_handoff("peer")
        item = transcript("We are going to meet at the station later today", 1)
        await partial(harness, item)
        clock.advance(5)
        await harness.peer_owner.cloud_captions.check_previews()
        await asyncio.wait_for(idle(harness), 1)
        assert [request.text for request in backend.calls] == [item.text]
        assert [block.speaker_style for block in presenter.snapshot().blocks] == [preview_style]
        if committed:
            item = replace(
                item,
                is_final=True,
                final_speaker_runs=(
                    FinalSpeakerRun(
                        item.text,
                        "1",
                        "soniox-scope",
                        source="soniox",
                        attribution_state="identified",
                    ),
                ),
            )
        await final(harness, item)
        await asyncio.wait_for(idle(harness), 1)
        assert len(backend.calls) == 2
        assert [entry.text for entry in harness.peer_runtime.translation_history] == [item.text]
        assert [block.speaker_style for block in presenter.snapshot().blocks] == ["gold"]
        assert harness.osc.messages == []
    finally:
        if handoff is not None:
            handoff.cancel()
            await asyncio.gather(handoff, return_exceptions=True)
        await harness.stop()
        await presenter.close()


@pytest.mark.asyncio
async def test_retraction_cancels_running_preview_and_later_growth_can_progress():
    harness, backend, events, clock = setup()
    try:
        await harness.start()
        item = transcript("Old hypothesis", 1)
        backend.block = True
        await partial(harness, item)
        clock.advance(5)
        await harness.peer_owner.cloud_captions.check_previews()
        await turn_loop()
        assert len(backend.calls) == 1
        await partial(harness, replace(item, text=""))
        await idle(harness)
        assert backend.completions[0].cancelled() and events == []
        backend.block = False
        await partial(harness, replace(item, text="A longer replacement hypothesis"))
        clock.advance(4)
        await harness.peer_owner.cloud_captions.check_previews()
        await idle(harness)
        assert backend.calls[-1].text == "A longer replacement hypothesis"
        assert events[-1].source_text == "A longer replacement hypothesis"
    finally:
        await harness.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["soniox", "qwen_audio"])
async def test_empty_partial_fences_late_preview_without_blocking_replacement(provider):
    initial_tasks = asyncio.all_tasks()
    harness, backend, events, clock = setup()
    entered, cancelled, release = (asyncio.Event() for _ in range(3))
    item = transcript("Old hypothesis", 1)
    idle_waiter = None

    async def late_translation(request):
        backend.calls.append(request)
        if request.text == item.text:
            entered.set()
            while not release.is_set():
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    cancelled.set()
        return Translation(
            utterance_id=request.utterance_id,
            text="translated:" + request.text,
        )

    backend.translate = late_translation
    try:
        await harness.start()
        await partial(harness, item, provider=provider)
        clock.advance(5)
        await harness.peer_owner.cloud_captions.check_previews()
        await asyncio.wait_for(entered.wait(), 1)
        await partial(harness, replace(item, text=""), provider=provider)
        if provider == "qwen_audio":
            await turn_loop()
            assert not cancelled.is_set()
        else:
            await asyncio.wait_for(cancelled.wait(), 1)

        idle_waiter = asyncio.create_task(harness.peer_owner.cloud_captions.wait_for_idle())
        await turn_loop()
        assert not idle_waiter.done()
        release.set()
        await asyncio.wait_for(idle_waiter, 1)
        await asyncio.wait_for(idle(harness), 1)
        retained = [item.text] if provider == "qwen_audio" else []
        assert [event.source_text for event in events] == retained

        replacement = "A longer replacement hypothesis"
        await partial(harness, replace(item, text=replacement), provider=provider)
        clock.advance(4)
        await harness.peer_owner.cloud_captions.check_previews()
        await asyncio.wait_for(idle(harness), 1)
        assert [event.source_text for event in events] == [*retained, replacement]
        assert events[-1].type == "translation_stream_update"
        assert events[-1].text == "translated:" + replacement
        assert [request.text for request in backend.calls] == [item.text, replacement]
        assert harness.peer_runtime.translation_history == []
        assert harness.osc.messages == []
    finally:
        release.set()
        if idle_waiter is not None:
            await asyncio.wait_for(idle_waiter, 1)
        await asyncio.wait_for(harness.stop(), 2)
        await turn_loop()
        assert not {task for task in asyncio.all_tasks() - initial_tasks if not task.done()}


@pytest.mark.asyncio
@pytest.mark.parametrize(("scale", "expected"), [(1.0, "repair"), (1.2, "continuation")])
async def test_join_uses_supported_effective_calibration_scale_without_estimator_replacement(
    scale, expected
):
    clock = FakeClock()
    presenter = OverlayPresenter(calibration=OverlayCalibration(), clock=clock)
    harness, backend, _, clock = setup(overlay_sink=presenter, clock=clock)
    harness.configuration.replace(
        replace(harness.configuration.snapshot().value, peer_target_language="en")
    )
    try:
        await harness.start()
        await presenter.update_calibration(replace(presenter.calibration, text_scale=scale))
        assert harness.peer_owner.output_projection.peer_caption_text_scale == scale
        first = transcript(
            "We will continue with this information and share the next details with everyone who",
            1,
        )
        backend.translations[first.text] = "short prior translation"
        await final(harness, first)
        await idle(harness)
        clock.advance(0.3)
        second = transcript("needs", 2)
        await partial(harness, second)
        await final(harness, second)
        await idle(harness)
        assert harness.peer_owner.cloud_captions.request_counts[expected] == 1
        assert backend.calls[-1].text == (
            first.text + " needs" if expected == "repair" else "needs"
        )
    finally:
        await harness.stop()
        await presenter.close()


@pytest.mark.asyncio
async def test_ready_repair_cancels_absorbed_preview_at_real_protected_upper_gate():
    timeline = Timeline()
    presenter = OverlayPresenter(
        calibration=OverlayCalibration(), clock=timeline.clock, sleep=timeline.sleep
    )
    harness, backend, _, clock = setup(overlay_sink=presenter, clock=timeline.clock)
    upper = transcript("Upper ended.", 1)
    lower = transcript("The preceding,", 2)
    incoming = transcript("second", 3)
    backend.translations[upper.text] = "가" * 40
    try:
        await harness.start()
        await final(harness, upper)
        await idle(harness)
        await timeline.advance(1.5)
        await final(harness, lower)
        await idle(harness)
        await timeline.advance(0.1)
        await partial(harness, incoming)
        await timeline.advance(4.9)
        await presenter.emit(caption(100, upper.utterance_id, "나" * 40, revision=2))
        await timeline.advance(0.1)
        await harness.peer_owner.cloud_captions.check_previews()
        await harness.peer_owner.cloud_captions.wait_for_idle()
        async with asyncio.timeout(1):
            while not any(
                decision.reason == "logical_pacing_wait"
                for decision in harness.output_runtime.routing_decisions
            ):
                await asyncio.sleep(0)
        before = clock.now()
        assert [block.id for block in presenter.snapshot().blocks] == [
            f"peer:{upper.utterance_id}",
            f"peer:{lower.utterance_id}",
        ]
        await final(harness, incoming)
        await asyncio.wait_for(idle(harness), 1)
        assert clock.now() == before
        assert [block.primary_text for block in presenter.snapshot().blocks][-1] == (
            "translated:The preceding, second"
        )
        assert any(
            decision.reason == "application_registered"
            and decision.metadata.get("utterance_id") == str(incoming.utterance_id)
            and decision.metadata["scene_revision"] is None
            for decision in harness.output_runtime.routing_decisions
        )
        assert presenter.peer_fragment_state(incoming.utterance_id, 0) is None
        assert (
            presenter._acceptance.terminal_reason(("peer", incoming.utterance_id))
            == "absorbed_by_repair"
        )
        await timeline.advance(1.1)
        assert all(
            block.id != f"peer:{incoming.utterance_id}" for block in presenter.snapshot().blocks
        )
        assert not any(
            decision.reason == "application_applied"
            and decision.metadata.get("utterance_id") == str(incoming.utterance_id)
            for decision in harness.output_runtime.routing_decisions
        )
    finally:
        await harness.stop()
        await presenter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["error", "timeout"])
async def test_failed_chained_repair_forwards_every_absorbed_identity(failure, monkeypatch):
    clock = FakeClock()
    presenter = OverlayPresenter(calibration=OverlayCalibration(), clock=clock)
    harness, backend, _, clock = setup(overlay_sink=presenter, clock=clock)
    submissions = []
    projection = harness.peer_owner.output_projection
    original = type(projection).project_peer_caption

    async def record(owner, submission, **kwargs):
        submissions.append(submission)
        await original(owner, submission, **kwargs)

    monkeypatch.setattr(type(projection), "project_peer_caption", record)
    first, second, third = transcript("A,", 1), transcript("second", 2), transcript("third", 3)
    try:
        await harness.start()
        await final(harness, first)
        await idle(harness)
        clock.advance(0.3)
        await partial(harness, second)
        clock.advance(5)
        await harness.peer_owner.cloud_captions.check_previews()
        await idle(harness)
        if failure == "timeout":
            backend.block = True
            harness.peer_owner.cloud_captions.request_timeout_s = 0.01
        else:
            backend.failures.update({"A, second", "A, second third"})
        await final(harness, second)
        clock.advance(0.1)
        await partial(harness, third)
        await final(harness, third)
        await asyncio.wait_for(idle(harness), 1)
        latest = submissions[-1]
        assert latest.source_text == "A, second third"
        assert latest.peer_caption_revision == 3
        assert latest.peer_superseded_unit_ids == (second.utterance_id, third.utterance_id)
        assert latest.failure_code == (
            "translation_timeout" if failure == "timeout" else "provider_error"
        )
        assert [(block.id, block.primary_text) for block in presenter.snapshot().blocks] == [
            (f"peer:{first.utterance_id}", "A, second third")
        ]
        assert [entry.text for entry in harness.peer_runtime.translation_history] == [
            "A, second third"
        ]
    finally:
        await harness.stop()
        await presenter.close()


@pytest.mark.asyncio
async def test_final_settlement_keeps_waiting_preview_original_one_second_budget():
    timeline = Timeline()
    presenter = OverlayPresenter(
        calibration=OverlayCalibration(), clock=timeline.clock, sleep=timeline.sleep
    )
    harness, backend, _, clock = setup(overlay_sink=presenter, clock=timeline.clock)
    upper = transcript("Upper ended.", 1)
    lower = transcript("A closed sentence.", 2)
    incoming = transcript("Independent final.", 3)
    backend.translations[upper.text] = "가" * 40
    try:
        await harness.start()
        await final(harness, upper)
        await idle(harness)
        await final(harness, lower)
        await idle(harness)
        await partial(harness, incoming)
        await timeline.advance(4.9)
        await presenter.emit(caption(100, upper.utterance_id, "나" * 40, revision=2))
        await timeline.advance(0.1)
        await harness.peer_owner.cloud_captions.check_previews()
        await harness.peer_owner.cloud_captions.wait_for_idle()
        async with asyncio.timeout(1):
            while not any(
                decision.reason == "logical_pacing_wait"
                for decision in harness.output_runtime.routing_decisions
            ):
                await asyncio.sleep(0)
        original_admission_at = clock.now()
        registration = presenter.peer_fragment_state(incoming.utterance_id, 0)
        await timeline.advance(0.6)
        await final(harness, incoming)
        await harness.peer_owner.cloud_captions.wait_for_idle()
        await turn_loop()
        assert (
            presenter.peer_fragment_state(incoming.utterance_id, 0).registered_at
            == original_admission_at
            == registration.registered_at
        )
        assert (
            presenter.peer_fragment_state(incoming.utterance_id, 0).registration_order
            == registration.registration_order
        )
        assert all(
            block.id != f"peer:{incoming.utterance_id}" for block in presenter.snapshot().blocks
        )
        await timeline.advance(0.39)
        assert all(
            block.id != f"peer:{incoming.utterance_id}" for block in presenter.snapshot().blocks
        )
        await timeline.advance(original_admission_at + 1.0 - clock.now())
        await asyncio.wait_for(harness.output_runtime.wait_for_peer_output_idle(), 1)
        assert clock.now() == pytest.approx(original_admission_at + 1.0)
        assert presenter.snapshot().blocks[-1].id == f"peer:{incoming.utterance_id}"
        assert presenter.snapshot().blocks[-1].primary_text == "translated:Independent final."
        assert harness.peer_owner.cloud_captions.request_counts["repair"] == 0
        assert harness.output_runtime._peer_caption_admission_started_at == {}
    finally:
        await harness.stop()
        await presenter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("punctuation", ["...", " ？！ "])
@pytest.mark.parametrize("show_preview", [False, True], ids=["before-preview", "shown-preview"])
async def test_zero_fragment_final_retires_segment_and_preserves_successor(
    punctuation, show_preview, monkeypatch
):
    timeline = Timeline()
    presenter = OverlayPresenter(
        calibration=OverlayCalibration(), clock=timeline.clock, sleep=timeline.sleep
    )
    harness, backend, _, clock = setup(overlay_sink=presenter, clock=timeline.clock)
    submissions = []
    projection = harness.peer_owner.output_projection
    original = type(projection).project_peer_caption

    async def record(owner, submission, **kwargs):
        submissions.append(submission)
        await original(owner, submission, **kwargs)

    monkeypatch.setattr(type(projection), "project_peer_caption", record)
    retired_ids = []
    try:
        await harness.start()
        for order in range(1, 4):
            item = transcript(f"Withdrawn observed source {order}", order)
            retired_ids.append(item.utterance_id)
            await partial(harness, item)
            if show_preview:
                await timeline.advance(5)
                await harness.peer_owner.cloud_captions.check_previews()
                await asyncio.wait_for(idle(harness), 1)
                assert presenter.snapshot().blocks[-1].id == f"peer:{item.utterance_id}"
            authoritative = replace(item, text=punctuation)
            await final(harness, authoritative)
            await final(harness, authoritative)
            await partial(harness, replace(item, text="Obsolete late partial"))
            await asyncio.wait_for(idle(harness), 1)
            assert harness.peer_owner.cloud_captions.is_completed(item.utterance_id)
            assert harness.peer_owner.cloud_captions._segments == {}
            assert not presenter.snapshot().blocks
            assert harness.peer_runtime.translation_history == []
            await timeline.advance(20)
            await harness.peer_owner.cloud_captions.check_previews()
            await asyncio.wait_for(idle(harness), 1)
        retirements = [
            submission
            for submission in submissions
            if submission.peer_request_kind == "final" and not submission.source_text
        ]
        assert [submission.child_utterance_id for submission in retirements] == retired_ids
        assert all(
            submission.translation is None
            and submission.outcome == "source_only"
            and submission.peer_caption_revision == (2 if show_preview else 1)
            for submission in retirements
        )
        successor = transcript("Valid successor.", 4)
        await final(harness, successor)
        await asyncio.wait_for(idle(harness), 1)
        assert [request.text for request in backend.calls] == (
            [f"Withdrawn observed source {order}" for order in range(1, 4)] if show_preview else []
        ) + [successor.text]
        assert [(block.id, block.primary_text) for block in presenter.snapshot().blocks] == [
            (f"peer:{successor.utterance_id}", "translated:Valid successor.")
        ]
        assert [entry.text for entry in harness.peer_runtime.translation_history] == [
            successor.text
        ]
        ui = []
        while not harness.ui_events.empty():
            ui.append(harness.ui_events.get_nowait())
        assert [
            event.payload.text for event in ui if event.type is UIEventType.TRANSCRIPT_FINAL
        ] == [successor.text]
        assert harness.osc.messages == []
    finally:
        await asyncio.wait_for(harness.stop(), 1)
        await presenter.close()
    assert harness.peer_owner.cloud_captions._tasks == set()
    assert harness.output_runtime._peer_caption_admission_started_at == {}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("detected_language", "configured_source", "boundary", "expected_kind"),
    [
        ("ko", "en", "r0-two", "continuation"),
        ("ko", "en", "r1-two", "continuation"),
        ("ko", "en", "r1-four", "final"),
        (None, "ko", "r0-two", "continuation"),
        (None, "en", "r0-two", "repair"),
    ],
    ids=[
        "detected-r0-continuation",
        "detected-r1-continuation",
        "detected-r1-four-line-stop",
        "configured-korean-fallback",
        "configured-english-fallback",
    ],
)
async def test_join_estimate_uses_preceding_detected_language_or_configured_fallback(
    detected_language, configured_source, boundary, expected_kind
):
    clock = FakeClock()
    presenter = OverlayPresenter(calibration=OverlayCalibration(), clock=clock)
    harness, backend, _, clock = setup(overlay_sink=presenter, clock=clock)
    harness.configuration.replace(
        replace(
            harness.configuration.snapshot().value,
            peer_source_mode="auto",
            peer_source_language=configured_source,
            peer_target_language="en",
        )
    )
    first_source = "가" * (39 if boundary == "r1-four" else 19)
    first_source += "끝." if boundary.startswith("r1") else ""
    second_source = "나" * (35 if boundary == "r1-four" else 21)
    merged = first_source + " " + second_source
    effective_language = detected_language or configured_source
    expected_lines = estimate_peer_translation_lines(merged, effective_language, "en")
    if boundary == "r1-four":
        assert expected_lines > 4
        assert 2 < estimate_peer_translation_lines(merged, configured_source, "en") <= 4
    elif expected_kind == "continuation":
        assert 2 < expected_lines <= 4
        if detected_language:
            assert estimate_peer_translation_lines(merged, configured_source, "en") <= 2
    else:
        assert expected_lines <= 2

    def with_language(item):
        return replace(
            item,
            final_language_runs=(
                (FinalLanguageRun(item.text, detected_language),) if detected_language else ()
            ),
        )

    first = with_language(transcript(first_source, 1, final=True))
    second = with_language(transcript(second_source, 2, final=True))
    sealed = (
        replace(
            owned_peer_speech_end(uuid4(), speech_end_at=1).segment,
            seal_reason="delivery_deadline",
        )
        if boundary.startswith("r1")
        else None
    )
    try:
        await harness.start()
        harness.peer_runtime.translation_history.append(
            ContextEntry("Earlier context.", "en", "en", clock.now(), "peer")
        )
        await final(harness, first, sealed=sealed)
        await idle(harness)
        clock.advance(0.3)
        await partial(harness, replace(second, is_final=False, final_language_runs=()))
        await final(harness, second)
        await asyncio.wait_for(idle(harness), 1)
        counts = harness.peer_owner.cloud_captions.request_counts
        assert counts[expected_kind] == (2 if expected_kind == "final" else 1)
        assert all(
            counts[kind] == 0 for kind in ("repair", "continuation") if kind != expected_kind
        )
        assert [request.text for request in backend.calls] == [
            first_source,
            merged if expected_kind == "repair" else second_source,
        ]
        assert all(
            request.source_language == (detected_language or "auto") for request in backend.calls
        )
        assert ("## Peer continuation" in backend.calls[-1].system_prompt) == (
            expected_kind == "continuation"
        )
        assert presenter.snapshot().blocks[-1].id == (
            f"peer:{first.utterance_id if expected_kind == 'repair' else second.utterance_id}"
        )
        if expected_kind == "repair":
            assert backend.calls[-1].context == backend.calls[0].context
        else:
            assert backend.calls[-1].context.endswith(f'- [peer] "{first_source}"')
        assert [entry.text for entry in harness.peer_runtime.translation_history] == [
            "Earlier context.",
            *([merged] if expected_kind == "repair" else [first_source, second_source]),
        ]
    finally:
        await asyncio.wait_for(harness.stop(), 1)
        await presenter.close()


@pytest.mark.asyncio
async def test_local_revision_zero_owner_output_presenter_keeps_history_and_upper_idle_ttl(
    monkeypatch,
):
    from puripuly_heart.core.overlay.sink import TranslationFinal, UtteranceClosed
    from puripuly_heart.domain.events import STTPartialEvent

    timeline, surface = Timeline(), Surface()
    presenter = OverlayPresenter(
        calibration=OverlayCalibration(),
        clock=timeline.clock,
        sleep=timeline.sleep,
        bridge=surface,
        native_retry_enabled=True,
    )
    harness, backend, _, clock = setup(overlay_sink=presenter, clock=timeline.clock)
    monkeypatch.setattr(
        type(harness.peer_owner), "_current_provider_id", lambda _: "local_cpu_auto"
    )
    observed = []
    original_emit = OverlayPresenter.emit

    async def observe(owner, event):
        observed.append(event)
        return await original_emit(owner, event)

    monkeypatch.setattr(OverlayPresenter, "emit", observe)
    first = transcript("An incomplete local clause,", 1)
    second = transcript("continues as a separate local final.", 2)
    backend.translations.update({first.text: "short A", second.text: "short B"})
    try:
        await harness.start()
        await harness.peer_owner.handle_stt_event(STTPartialEvent(first.utterance_id, first))
        await timeline.advance(10)
        await harness.peer_owner.cloud_captions.check_previews()
        assert backend.calls == []
        assert not presenter.snapshot().blocks
        origin = clock.now()
        await harness.peer_owner.handle_stt_event(
            STTFinalEvent(first.utterance_id, replace(first, is_final=True))
        )
        await harness.translation_turns.wait_for_idle()
        await harness.output_runtime.wait_for_peer_output_idle()
        first_child = backend.calls[0].utterance_id
        assert first_child != first.utterance_id
        assert surface.expirations[f"peer:{first_child}"] - origin == 8
        await timeline.advance(1.5)
        await harness.peer_owner.handle_stt_event(
            STTFinalEvent(second.utterance_id, replace(second, is_final=True))
        )
        await harness.translation_turns.wait_for_idle()
        await harness.output_runtime.wait_for_peer_output_idle()
        second_child = backend.calls[1].utterance_id
        assert second_child != second.utterance_id
        assert [block.id for block in presenter.snapshot().blocks] == [
            f"peer:{first_child}",
            f"peer:{second_child}",
        ]
        assert [block.primary_text for block in presenter.snapshot().blocks] == [
            "short A",
            "short B",
        ]
        assert surface.expirations[f"peer:{first_child}"] - origin == 8
        assert surface.expirations[f"peer:{second_child}"] - origin == 9.5
        for parent, child in ((first, first_child), (second, second_child)):
            publications = [event for event in observed if event.utterance_id == child]
            assert [type(event) for event in publications] == [TranslationFinal, UtteranceClosed]
            assert publications[0].peer_caption_revision == 0
            assert publications[0].parent_utterance_id == parent.utterance_id
            assert publications[1].is_final is True
            assert presenter._entries[("peer", child)].closed_seq is not None
        assert [entry.text for entry in harness.peer_runtime.translation_history] == [
            first.text,
            second.text,
        ]
        assert harness.peer_owner.cloud_captions.request_counts == dict.fromkeys(
            ("preview", "final", "repair", "continuation"), 0
        )
        await timeline.advance(2.1)
        assert [block.primary_text for block in presenter.snapshot().blocks] == [
            "short A",
            "short B",
        ]
        await timeline.advance(4.4)
        assert [block.primary_text for block in presenter.snapshot().blocks] == ["short B"]
        await timeline.advance(1.5)
        assert not presenter.snapshot().blocks
    finally:
        await harness.stop()
        await presenter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("self_contention", [False, True])
async def test_local_revision_zero_overflow_enters_fragments_without_extra_requests_or_history(
    monkeypatch, self_contention
):
    from puripuly_heart.core.overlay.peer_caption_layout import peer_translation_fragments
    from tests.core.test_peer_caption_fragments import own_speech

    timeline, surface = Timeline(), Surface()
    presenter = OverlayPresenter(
        calibration=OverlayCalibration(),
        clock=timeline.clock,
        sleep=timeline.sleep,
        bridge=surface,
        native_retry_enabled=True,
    )
    harness, backend, _, _ = setup(overlay_sink=presenter, clock=timeline.clock)
    monkeypatch.setattr(
        type(harness.peer_owner), "_current_provider_id", lambda _: "local_cpu_auto"
    )
    source = transcript("A local final with an overflowing translation.", 1)
    text = "가" * 56 + "나" * 20
    backend.translations[source.text] = text
    try:
        await harness.start()
        if self_contention:
            await presenter.emit(own_speech())
        await harness.peer_owner.handle_stt_event(
            STTFinalEvent(source.utterance_id, replace(source, is_final=True))
        )
        await harness.translation_turns.wait_for_idle()
        await harness.output_runtime.wait_for_peer_output_idle()
        await timeline.advance(0)
        child = backend.calls[0].utterance_id
        assert child != source.utterance_id
        blocks = [block for block in presenter.snapshot().blocks if block.channel == "peer"]
        assert blocks[0].secondary_text == source.text
        assert blocks[0].secondary_enabled
        if self_contention:
            assert [block.id for block in blocks] == [f"peer:{child}"]
            first = presenter.peer_fragment_state(child, 0)
            tail = presenter.peer_fragment_state(child, 1)
            assert first.visible_since == 0
            assert tail.pending and tail.visible_since is None
            await timeline.advance(0.9)
            assert [
                block.id for block in presenter.snapshot().blocks if block.channel == "peer"
            ] == [f"peer:{child}"]
            await timeline.advance(0.1)
            blocks = [block for block in presenter.snapshot().blocks if block.channel == "peer"]
            first = presenter.peer_fragment_state(child, 0)
            tail = presenter.peer_fragment_state(child, 1)
            assert first.departed and first.visible_since == 0
            assert not tail.pending and tail.visible_since == 1
            assert surface.expirations[tail.id] == 9
            assert [block.primary_text for block in blocks] == [peer_translation_fragments(text)[1]]
            assert blocks[0].secondary_text == ""
            assert not blocks[0].secondary_enabled
        else:
            assert [block.primary_text for block in blocks] == list(
                peer_translation_fragments(text)
            )
            assert len(blocks) == 2
            assert blocks[1].secondary_text == ""
            assert not blocks[1].secondary_enabled
        assert all(
            entry.peer_caption_revision == 0
            for entry in presenter._entries.values()
            if entry.channel == "peer"
        )
        assert presenter.snapshot().native_fresh_render_targets.peer in {
            block.id for block in blocks
        }
        episode = presenter.snapshot().native_quiet_tail_episodes.peer
        assert episode.phase == "final" and episode.generation == 2
        await timeline.advance(3)
        assert [block for block in presenter.snapshot().blocks if block.channel == "peer"] == blocks
        assert presenter.snapshot().native_quiet_tail_episodes.peer == episode
        assert len(backend.calls) == 1
        assert [entry.text for entry in harness.peer_runtime.translation_history] == [source.text]
        assert harness.peer_owner.cloud_captions.request_counts == dict.fromkeys(
            ("preview", "final", "repair", "continuation"), 0
        )
    finally:
        await harness.stop()
        await presenter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("later_first_delay", [0.2, 8.1])
async def test_local_finals_enter_shared_display_in_translation_ready_order(
    monkeypatch, later_first_delay
):
    timeline = Timeline()
    presenter = OverlayPresenter(
        calibration=OverlayCalibration(), clock=timeline.clock, sleep=timeline.sleep
    )
    harness, backend, _, _ = setup(overlay_sink=presenter, clock=timeline.clock)
    monkeypatch.setattr(
        type(harness.peer_owner), "_current_provider_id", lambda _: "local_cpu_auto"
    )
    first, second = transcript("First local final.", 1), transcript("Second local final.", 2)
    backend.block = True
    try:
        await harness.start()
        for source in (first, second):
            await harness.peer_owner.handle_stt_event(
                STTFinalEvent(source.utterance_id, replace(source, is_final=True))
            )
        async with asyncio.timeout(1):
            while len(backend.completions) < 2:
                await asyncio.sleep(0)
        backend.completions[1].set_result(None)
        async with asyncio.timeout(1):
            while not presenter.snapshot().blocks:
                await asyncio.sleep(0)
        ready_child = backend.calls[1].utterance_id
        assert presenter.snapshot().blocks[0].id == f"peer:{ready_child}"
        await timeline.advance(later_first_delay)
        backend.completions[0].set_result(None)
        await harness.translation_turns.wait_for_idle()
        await harness.output_runtime.wait_for_peer_output_idle()
        await timeline.advance(0)
        expected = [f"peer:{backend.calls[0].utterance_id}"]
        if later_first_delay < 8:
            expected.insert(0, f"peer:{ready_child}")
        assert [block.id for block in presenter.snapshot().blocks] == expected
        assert [entry.text for entry in harness.peer_runtime.translation_history] == [
            first.text,
            second.text,
        ]
    finally:
        await harness.stop()
        await presenter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("overflow_preview", [False, True])
async def test_l3_preview_to_ordinary_overflow_is_one_caption_and_one_history_entry(
    overflow_preview,
):
    from puripuly_heart.core.overlay.peer_caption_layout import peer_translation_fragments

    timeline = Timeline()
    presenter = OverlayPresenter(
        calibration=OverlayCalibration(),
        clock=timeline.clock,
        sleep=timeline.sleep,
        native_retry_enabled=True,
    )
    harness, backend, _, _ = setup(overlay_sink=presenter, clock=timeline.clock)
    preview = transcript("A preview source", 1)
    settled = replace(preview, text="A preview source becomes a complete final.")
    text = "가" * 56 + "나" * 20
    preview_text = text if overflow_preview else "short preview"
    backend.translations.update({preview.text: preview_text, settled.text: text})
    try:
        await harness.start()
        await partial(harness, preview)
        await timeline.advance(5)
        await harness.peer_owner.cloud_captions.check_previews()
        await idle(harness)
        blocks = presenter.snapshot().blocks
        assert [block.primary_text for block in blocks] == list(
            peer_translation_fragments(preview_text)
        )
        assert [block.secondary_text for block in blocks] == [preview.text] + [""] * (
            len(blocks) - 1
        )
        assert [block.secondary_enabled for block in blocks] == [True] + [False] * (len(blocks) - 1)
        await timeline.advance(0.2)
        await final(harness, settled)
        await idle(harness)
        await timeline.advance(0)
        assert [block.primary_text for block in presenter.snapshot().blocks] == list(
            peer_translation_fragments(text)
        )
        assert [block.secondary_text for block in presenter.snapshot().blocks] == [settled.text, ""]
        assert [block.secondary_enabled for block in presenter.snapshot().blocks] == [True, False]
        assert all(entry.peer_caption_revision == 2 for entry in presenter._entries.values())
        assert len(backend.calls) == 2
        assert backend.calls[0].utterance_id == backend.calls[1].utterance_id
        assert [entry.text for entry in harness.peer_runtime.translation_history] == [settled.text]
        assert harness.peer_owner.cloud_captions.request_counts == {
            "preview": 1,
            "final": 1,
            "repair": 0,
            "continuation": 0,
        }
    finally:
        await harness.stop()
        await presenter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("predecessor_ending", ["failed", "cancelled"])
async def test_early_local_overlay_is_not_republished_after_predecessor_recovery(
    monkeypatch, predecessor_ending
):
    timeline, surface = Timeline(), Surface()
    presenter = OverlayPresenter(
        calibration=OverlayCalibration(),
        clock=timeline.clock,
        sleep=timeline.sleep,
        bridge=surface,
        native_retry_enabled=True,
    )
    harness, backend, _, _ = setup(overlay_sink=presenter, clock=timeline.clock)
    monkeypatch.setattr(
        type(harness.peer_owner), "_current_provider_id", lambda _: "local_cpu_auto"
    )
    first, second = transcript("A pending predecessor.", 1), transcript("A ready successor.", 2)
    backend.block = True
    if predecessor_ending == "failed":
        backend.failures.add(first.text)
    try:
        await harness.start()
        for source in (first, second):
            await harness.peer_owner.handle_stt_event(
                STTFinalEvent(source.utterance_id, replace(source, is_final=True))
            )
        async with asyncio.timeout(1):
            while len(backend.completions) != 2:
                await asyncio.sleep(0)
        backend.completions[1].set_result(None)
        async with asyncio.timeout(1):
            while not presenter.snapshot().blocks:
                await asyncio.sleep(0)
        child = backend.calls[1].utterance_id
        deadline = surface.expirations[f"peer:{child}"]
        episode = presenter.snapshot().native_quiet_tail_episodes.peer
        await timeline.advance(0.2)
        if predecessor_ending == "failed":
            backend.completions[0].set_result(None)
        else:
            harness.peer_runtime.translation_tasks[backend.calls[0].utterance_id].cancel()
        await asyncio.wait_for(harness.translation_turns.wait_for_idle(), 1)
        await asyncio.wait_for(harness.output_runtime.wait_for_peer_output_idle(), 1)
        events = []
        while not harness.ui_events.empty():
            events.append(harness.ui_events.get_nowait())
        assert (
            sum(
                event.type is UIEventType.TRANSLATION_DONE and event.utterance_id == child
                for event in events
            )
            == 1
        )
        assert [entry.text for entry in harness.peer_runtime.translation_history] == [
            first.text,
            second.text,
        ]
        assert surface.expirations[f"peer:{child}"] == deadline == 8
        assert presenter.snapshot().native_quiet_tail_episodes.peer == episode
        assert harness.peer_owner.output_projection._early_peer_overlays == set()
        assert len(backend.calls) == 2
    finally:
        await harness.stop()
        await presenter.close()


@pytest.mark.asyncio
async def test_g4_owner_full_revisions_keep_one_logical_history_contribution():
    from tests.core.test_peer_caption_fragments import own_speech

    timeline = Timeline()
    presenter = OverlayPresenter(
        calibration=OverlayCalibration(), clock=timeline.clock, sleep=timeline.sleep
    )
    harness, backend, _, _ = setup(overlay_sink=presenter, clock=timeline.clock)
    first = transcript("Initial", 1)
    settled = replace(first, text="Initial source,")
    repair = transcript("ends.", 2)
    unrelated = uuid4()
    backend.translations.update(
        {
            first.text: "가" * 64,
            settled.text: "나" * 128,
            "Initial source, ends.": "다" * 24,
        }
    )
    try:
        await harness.start()
        await presenter.emit(own_speech())
        await partial(harness, first)
        await timeline.advance(5)
        await harness.peer_owner.cloud_captions.check_previews()
        await idle(harness)
        unit = backend.calls[0].utterance_id
        tail = presenter.peer_fragment_state(unit, 1)
        registration = tail.registered_at, tail.registration_order
        await timeline.advance(0.1)
        await presenter.emit(caption(101, unrelated, "B"))
        await timeline.advance(0.1)
        await final(harness, settled)
        await harness.peer_owner.cloud_captions.wait_for_idle()
        await timeline.advance(0)
        pending = sorted(
            (entry for entry in presenter.peer_fragment_states() if entry.pending),
            key=lambda entry: entry.registration_order,
        )
        assert [(entry.logical_id, entry.index) for entry in pending] == [
            (unit, 1),
            (unrelated, 0),
            (unit, 2),
        ]
        assert (pending[0].registered_at, pending[0].registration_order) == registration
        assert all(entry.revision == 2 for entry in presenter.peer_fragment_states(unit))
        await timeline.advance(0.1)
        await partial(harness, repair)
        await final(harness, repair)
        await harness.peer_owner.cloud_captions.wait_for_idle()
        await timeline.advance(0)
        assert [entry.index for entry in presenter.peer_fragment_states(unit)] == [0]
        assert presenter._entries[("peer", unit)].peer_caption_revision == 3
        assert presenter._entries[("peer", unrelated)].peer_pending
        assert [entry.text for entry in harness.peer_runtime.translation_history] == [
            "Initial source, ends."
        ]
        assert len(backend.calls) == 3
        assert len({request.utterance_id for request in backend.calls}) == 1
        assert harness.peer_owner.cloud_captions.request_counts == {
            "preview": 1,
            "final": 1,
            "repair": 1,
            "continuation": 0,
        }
        await timeline.advance(
            presenter._entries[("peer", unrelated)].peer_registered_at + 1 - timeline.clock.now()
        )
        await harness.output_runtime.wait_for_peer_output_idle()
        assert ("peer", unit) not in presenter._entries
        assert any(block.id == f"peer:{unrelated}" for block in presenter.snapshot().blocks)
    finally:
        await harness.stop()
        await presenter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("later_first_delay", [0.2, 8.1])
async def test_cloud_first_finals_display_by_readiness_not_ordered_history(later_first_delay):
    timeline, surface = Timeline(), Surface()
    presenter = OverlayPresenter(
        calibration=OverlayCalibration(),
        clock=timeline.clock,
        sleep=timeline.sleep,
        bridge=surface,
        native_retry_enabled=True,
    )
    harness, backend, _, _ = setup(overlay_sink=presenter, clock=timeline.clock)
    first, second = transcript("Earlier cloud final.", 1), transcript("Later cloud final.", 2)
    backend.translations.update({first.text: "short A", second.text: "short B"})
    backend.block = True
    try:
        await harness.start()
        await final(harness, first)
        await final(harness, second)
        async with asyncio.timeout(1):
            while len(backend.completions) != 2:
                await asyncio.sleep(0)
        backend.completions[1].set_result(None)
        async with asyncio.timeout(1):
            while not presenter.snapshot().blocks:
                await asyncio.sleep(0)
        assert [block.primary_text for block in presenter.snapshot().blocks] == ["short B"]
        b_order = presenter.snapshot().blocks[0].publication_order
        assert b_order == 1
        assert surface.expirations[f"peer:{second.utterance_id}"] == 8
        assert harness.ui_events.empty()
        await timeline.advance(later_first_delay)
        if later_first_delay > 8:
            assert not presenter.snapshot().blocks
        backend.completions[0].set_result(None)
        await asyncio.wait_for(idle(harness), 1)
        blocks = presenter.snapshot().blocks
        assert [block.primary_text for block in blocks] == (
            ["short B", "short A"] if later_first_delay < 8 else ["short A"]
        )
        assert blocks[-1].publication_order == 2
        assert presenter.snapshot().native_fresh_render_generations.peer == 2
        assert presenter.snapshot().native_fresh_render_targets.peer == f"peer:{first.utterance_id}"
        assert surface.expirations[f"peer:{first.utterance_id}"] == later_first_delay + 8
        assert [call.utterance_id for call in backend.calls] == [
            first.utterance_id,
            second.utterance_id,
        ]
        assert [entry.text for entry in harness.peer_runtime.translation_history] == [
            first.text,
            second.text,
        ]
        events = []
        while not harness.ui_events.empty():
            events.append(harness.ui_events.get_nowait())
        assert [
            event.utterance_id for event in events if event.type is UIEventType.TRANSLATION_DONE
        ] == [
            first.utterance_id,
            second.utterance_id,
        ]
        assert harness.peer_owner.cloud_captions.request_counts == {
            "preview": 0,
            "final": 2,
            "repair": 0,
            "continuation": 0,
        }
    finally:
        await harness.stop()
        await presenter.close()


@pytest.mark.asyncio
async def test_expired_cloud_caption_cannot_get_fresh_display_order_after_terminal_cache_rollover():
    from puripuly_heart.core.overlay.presenter_acceptance import CLOSED_TOMBSTONE_LIMIT
    from tests.core.runtime.test_output_runtime import _publish_peer_overlay

    timeline = Timeline()
    presenter = OverlayPresenter(
        calibration=OverlayCalibration(), clock=timeline.clock, sleep=timeline.sleep
    )
    harness, backend, _, _ = setup(overlay_sink=presenter, clock=timeline.clock)
    original = transcript("An expired cloud caption.", 1)
    backend.translations[original.text] = "expired"
    try:
        await harness.start()
        await final(harness, original)
        await idle(harness)
        await timeline.advance(8.1)
        for index in range(CLOSED_TOMBSTONE_LIMIT + 3):
            await _publish_peer_overlay(
                harness.output_runtime,
                caption(index + 10, uuid4(), f"bounded {index}"),
                source_order=index + 2,
            )
            await timeline.advance(0)
            await timeline.advance(1)
            await asyncio.wait_for(harness.output_runtime.wait_for_peer_output_idle(), 1)
        key = ("peer", original.utterance_id)
        assert key not in presenter._acceptance.scene_terminal_keys
        assert key not in presenter._acceptance.terminal_registry
        before = presenter.snapshot()
        await _publish_peer_overlay(
            harness.output_runtime,
            caption(9999, original.utterance_id, "expired", source=original.text),
            source_order=1,
        )
        await harness.output_runtime.wait_for_peer_output_idle()
        assert presenter.snapshot() == before
        assert harness.output_runtime.routing_decisions[-1].reason == "stale"
        assert len(backend.calls) == 1
    finally:
        await harness.stop()
        await presenter.close()
