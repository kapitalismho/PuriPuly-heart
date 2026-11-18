from __future__ import annotations

import asyncio
from urllib.parse import parse_qs, urlsplit

import pytest

from puripuly_heart.app.wiring.wiring_local_asr_provider_runtime import SharedSTTProviderFactory
from puripuly_heart.config.runtime_resolution import STTRuntimeIntent, resolve_stt_config
from puripuly_heart.core import network_clients
from puripuly_heart.core.conversation_history import ConversationHistory
from puripuly_heart.core.local_asr.local_asr_provider_runtime import ProviderRuntimeBuildRequest
from puripuly_heart.core.observability import ConversationRecord
from puripuly_heart.core.orchestrator.translation_channel_callbacks import (
    TranslationChannelOwnerCallbacks,
)
from puripuly_heart.core.runtime.stt_session_projection import SttSessionStateProjection
from puripuly_heart.core.storage.secrets import InMemorySecretStore
from puripuly_heart.core.stt.backend import STTFinalSpeakerEvidence, STTProviderTurnTerminal
from tests.core.runtime.test_peer_capture_session import make_config, make_owner, wait_until
from tests.core.test_peer_caption_lifecycle import ControlledBackend, idle
from tests.core.test_peer_overlay_policy import Surface, Timeline, presenter_for
from tests.helpers.fakes import RecordingOscQueue
from tests.helpers.scoped_provider_sessions import FakeScribeWebSocket, _owned_deepgram_events
from tests.helpers.translation_owners import compose_translation_test_harness


class _ConversationObserver(ConversationHistory):
    def record_conversation_observation(self, **fields):
        self.observe_conversation(ConversationRecord(**fields))

    def emit_basic(self, *args, **kwargs):
        pass


async def _idle(harness):
    async with asyncio.timeout(3):
        await idle(harness)


@pytest.fixture
async def route(monkeypatch):
    timeline = Timeline()
    surface = Surface()
    presenter = presenter_for(timeline, surface)
    await presenter.begin_native_retry_epoch(enabled=True)
    socket = FakeScribeWebSocket("native-scope")
    native_scope = "native-scope"
    urls = []

    async def connect(url, *, additional_headers):
        nonlocal socket, native_scope
        if socket.closed:
            native_scope = f"native-scope-{len(urls)}"
            socket = FakeScribeWebSocket(native_scope)
        assert additional_headers == {"xi-api-key": "controlled"}
        urls.append(url)
        return socket

    monkeypatch.setattr(network_clients, "external_websocket_connect", connect)
    secrets = InMemorySecretStore()
    secrets.set("elevenlabs_scribe_api_key", "controlled")
    config = make_config(provider_id="elevenlabs_scribe")
    engine = await SharedSTTProviderFactory(secrets, timeline.clock, 300.0, None).create(
        ProviderRuntimeBuildRequest(
            config=resolve_stt_config(
                STTRuntimeIntent(
                    channel="peer",
                    provider="elevenlabs_scribe",
                    source_language="en",
                    elevenlabs_scribe_language_code="en",
                )
            ),
            provider_signature=config.provider_signature,
            recognition_projection="scoped",
            runtime_signature=config.runtime_signature,
        ),
        gpu_runtime=None,
    )
    backend = ControlledBackend()
    history = _ConversationObserver()
    harness = compose_translation_test_harness(
        peer_stt=engine,
        llm=backend,
        osc=RecordingOscQueue(),
        clock=timeline.clock,
        overlay_sink=presenter,
        runtime_logging=history,
        source_language="en",
        target_language="ko",
        peer_translation_enabled=True,
    )

    class Sink:
        handle_owned_vad_event = staticmethod(harness.peer_owner.handle_peer_owned_vad_event)

    sink_started = asyncio.get_running_loop().create_future()

    async def run_audio_loop(**kwargs):
        sink_started.set_result(kwargs["sink"])
        await asyncio.Event().wait()

    capture, *_ = make_owner(sink=Sink(), clock=timeline.clock, run_audio_loop=run_audio_loop)
    callbacks = TranslationChannelOwnerCallbacks(SttSessionStateProjection())
    callbacks.bind_self(harness.self_owner)
    callbacks.bind_peer(harness.peer_owner)
    callbacks.bind_peer_capture(capture)
    capture.bind_publication_generation_observer(
        activated=harness.output_runtime.activate_peer_generation,
        retired=harness.output_runtime.retire_peer_generation,
    )
    evidence = []
    terminals = []

    async def observe(event):
        if isinstance(event, STTProviderTurnTerminal):
            terminals.append(event)
        elif isinstance(event, STTFinalSpeakerEvidence):
            evidence.append(event)
        await callbacks.peer_event_handler(event)

    try:
        await harness.start()
        engine.bind_event_sink(observe)
        await capture.apply_intent(config, enabled=True)
        guarded = await asyncio.wait_for(sink_started, 2)
        ledger = capture.segment_ledger
        assert ledger is not None

        async def committed(text, *speakers, partial=False):
            terminal_index = len(terminals)
            index = 0 if socket.closed else socket.commits
            start, chunks, end = _owned_deepgram_events(
                ledger, start_sample=(terminal_index + 1) * 1000
            )
            for event in (start, *chunks, end):
                await guarded.handle_owned_vad_event(event)
            await wait_until(lambda: socket.commits == index + 1, timeout_s=2)
            if partial:
                socket.push("partial_transcript", text=text)
            socket.push("committed_transcript", text=text, segment_index=index)
            if speakers:
                push_diarized(index, speakers)
            await wait_until(lambda: len(terminals) > terminal_index, timeout_s=2)
            await engine.wait_for_event_ingress_drain()
            terminal = terminals[terminal_index]
            assert terminal.text == text and terminal.text_authority == "authoritative"
            assert terminal.native_segment.provider_session_id == native_scope
            assert terminal.native_segment.segment_index == index
            assert any(
                receipt.identity == terminal.identity.segment
                for receipt in ledger.terminal_receipts
            )
            return start.segment.identity.segment_id

        def push_diarized(index, words):
            socket.push(
                "diarized_transcript",
                segment_index=index,
                text="".join(text for text, _ in words),
                words=[{"text": text, "speaker_id": speaker} for text, speaker in words],
            )

        async def diarized(index, *words):
            original = len(evidence)
            push_diarized(index, words)
            await wait_until(lambda: len(evidence) > original, timeout_s=2)
            await engine.wait_for_event_ingress_drain()
            return evidence[-1]

        yield harness, backend, presenter, timeline, surface, history, committed, diarized, socket, urls
    finally:
        await capture.close()
        await harness.stop()
        await presenter.close()


@pytest.mark.asyncio
async def test_peer_scribe_factory_commit_progress_late_identity_then_split_and_merge(route):
    harness, backend, presenter, timeline, _, history, committed, diarized, socket, urls = route
    first = await committed("Alpha. Beta.")
    await _idle(harness)
    first_id = f"peer:{first}"
    assert [block.primary_text for block in presenter.snapshot().blocks] == [
        "translated:Alpha. Beta."
    ]
    assert presenter.snapshot().blocks[0].speaker_style == "gray"
    await timeline.advance(0.2)
    second = await committed("An independent line.")
    await _idle(harness)
    second_id = f"peer:{second}"
    assert len(urls) == 1 and socket.commits == 2
    options = parse_qs(urlsplit(urls[0]).query)
    assert options["model_id"] == ["scribe_v3_live"]
    assert options["language_code"] == ["en"]
    assert options["diarize"] == options["include_timestamps"] == ["true"]
    assert options["filter_background_audio"] == ["false"]
    assert options["commit_strategy"] == ["manual"]
    stable = presenter.peer_fragment_state(second, 0)
    retry = presenter.snapshot().native_fresh_render_generations
    await diarized(1, ("An independent line.", "speaker-2"))
    await _idle(harness)
    assert len(backend.calls) == 2
    assert presenter.peer_fragment_state(second, 0) == stable
    assert presenter.snapshot().native_fresh_render_generations == retry
    assert (
        next(block for block in presenter.snapshot().blocks if block.id == second_id).speaker_style
        != "gray"
    )
    initial_order = presenter.peer_fragment_state(first, 0).registration_order
    await diarized(0, ("Alpha. ", "speaker-0"), ("Beta.", "speaker-1"))
    await _idle(harness)
    blocks = {block.id: block for block in presenter.snapshot().blocks}
    assert blocks[first_id].primary_text == "translated:Alpha. "
    assert blocks[first_id].secondary_text == "Alpha."
    assert presenter.peer_fragment_state(first, 0).registration_order == initial_order
    assert presenter.peer_fragment_state(first, 0).expires_at == pytest.approx(8.2)
    assert [call.text for call in backend.calls] == [
        "Alpha. Beta.",
        "An independent line.",
        "Alpha. ",
        "Beta.",
    ]
    assert [entry.text for entry in harness.peer_runtime.translation_history] == [
        "Alpha. ",
        "Beta.",
        "An independent line.",
    ]
    assert [record.source_text for record in history.conversation_records] == [
        "Alpha.",
        "Beta.",
        "An independent line.",
    ]
    assert presenter.snapshot().blocks[-1].id == second_id
    socket.push("speakers_merged", from_speaker_id="speaker-1", to_speaker_id="speaker-0")
    await wait_until(lambda: len(backend.calls) == 5, timeout_s=2)
    await _idle(harness)
    assert backend.calls[-1].text == "Alpha. Beta."
    assert [record.source_text for record in history.conversation_records] == [
        "Alpha. Beta.",
        "An independent line.",
    ]
    assert [block.id for block in presenter.snapshot().blocks] == [first_id, second_id]
    assert harness.osc.messages == []


@pytest.mark.asyncio
@pytest.mark.parametrize("retirement", ["expiry", "clear", "eviction"])
async def test_dead_scribe_family_split_never_registers_a_new_child(route, retirement):
    harness, backend, presenter, timeline, surface, _, committed, diarized, _, _ = route
    old = await committed("Dead one. Dead two.")
    await idle(harness)
    assert presenter.peer_fragment_state(old, 0) is not None
    if retirement == "expiry":
        await timeline.advance(8.1)
    elif retirement == "clear":
        await presenter.clear_for_runtime_detach()
    else:
        await committed("Fresh A.")
        await idle(harness)
        await committed("Fresh B.")
        await idle(harness)
        await timeline.advance(1.01)
    assert all(block.id != f"peer:{old}" for block in presenter.snapshot().blocks)
    first_scene = len(surface.snapshots)
    before = len(backend.calls)
    await diarized(0, ("Dead one. ", "old-a"), ("Dead two.", "old-b"))
    await idle(harness)
    assert len(backend.calls) == before
    new = await committed("This is current.")
    await idle(harness)
    await timeline.advance(1.01)
    assert any(block.id == f"peer:{new}" for block in presenter.snapshot().blocks)
    assert all(
        "Dead" not in block.primary_text and "Dead" not in block.secondary_text
        for snapshot in surface.snapshots[first_scene:]
        for block in snapshot.blocks
    )


@pytest.mark.asyncio
async def test_scribe_repair_that_finishes_after_expiry_cannot_reenter(route):
    harness, backend, presenter, timeline, surface, _, committed, diarized, _, _ = route
    old = await committed("Too late. Split tail.")
    await idle(harness)
    backend.block = True
    await diarized(0, ("Too late. ", "old-a"), ("Split tail.", "old-b"))
    await wait_until(lambda: len(backend.completions) == 2, timeout_s=2)
    await timeline.advance(8.1)
    assert all(block.id != f"peer:{old}" for block in presenter.snapshot().blocks)
    expired_scene = len(surface.snapshots)
    for completion in reversed(backend.completions):
        completion.set_result(None)
    await idle(harness)
    assert not presenter.snapshot().blocks
    assert not any(snapshot.blocks for snapshot in surface.snapshots[expired_scene:])
    backend.block = False
    ready = await committed("New after old work.")
    await idle(harness)
    assert [block.id for block in presenter.snapshot().blocks] == [f"peer:{ready}"]


@pytest.mark.asyncio
async def test_scribe_available_speakers_skip_redundant_whole_source_translation(route):
    harness, backend, presenter, timeline, _, history, committed, _, _, _ = route
    backend.translations["A longer first group. "] = "가" * 64
    await committed(
        "A longer first group. A different speaker.",
        ("A longer first group. ", "speaker-A"),
        ("A different speaker.", "speaker-B"),
    )
    await _idle(harness)
    assert [call.text for call in backend.calls] == [
        "A longer first group. ",
        "A different speaker.",
    ]
    assert [record.source_text for record in history.conversation_records] == [
        "A longer first group.",
        "A different speaker.",
    ]
    await timeline.advance(0)
    first, continuation = presenter.snapshot().blocks
    assert first.secondary_text == "A longer first group."
    assert continuation.secondary_text == ""
    await timeline.advance(1.01)
    assert all(block.id != first.id for block in presenter.snapshot().blocks)
    visible = {block.id: block for block in presenter.snapshot().blocks}
    assert visible[continuation.id].secondary_text == ""
    assert next(
        block for block in visible.values() if block.id != continuation.id
    ).secondary_text == ("A different speaker.")


@pytest.mark.asyncio
async def test_scribe_ready_regroup_tail_keeps_normal_wait_order_and_survives_head_departure(route):
    harness, backend, presenter, timeline, surface, history, committed, diarized, socket, _ = route
    head = await committed("First speaker. Later sibling.")
    await _idle(harness)
    await committed("Occupies the lower slot.")
    await _idle(harness)
    await timeline.advance(0.2)
    waiter = await committed("Already waiting.")
    await _idle(harness)
    old_wait = presenter.peer_fragment_state(waiter, 0)
    assert old_wait.pending
    await timeline.advance(0.2)
    await diarized(0, ("First speaker. ", "a"), ("Later sibling.", "b"))
    await _idle(harness)
    records = {record.source_text: record for record in history.conversation_records}
    from uuid import UUID

    sibling_id = UUID(records["Later sibling."].utterance_id)
    sibling = presenter.peer_fragment_state(sibling_id, 0)
    assert sibling.pending and sibling.registration_order > old_wait.registration_order
    assert presenter.peer_fragment_state(waiter, 0) == old_wait
    await timeline.advance(0.81)
    assert all(block.id != f"peer:{head}" for block in presenter.snapshot().blocks)
    assert any(block.id == f"peer:{waiter}" for block in presenter.snapshot().blocks)
    assert presenter.peer_fragment_state(sibling_id, 0).pending
    await timeline.advance(1.01)
    shown = next(block for block in presenter.snapshot().blocks if block.id == f"peer:{sibling_id}")
    assert shown.secondary_text == "Later sibling."
    shown_state = presenter.peer_fragment_state(sibling_id, 0)
    calls = len(backend.calls)
    scene_start = len(surface.snapshots)
    socket.push("speakers_merged", from_speaker_id="b", to_speaker_id="a")
    await wait_until(lambda: len(surface.snapshots) > scene_start, timeout_s=2)
    await _idle(harness)
    assert len(backend.calls) == calls
    assert presenter.peer_fragment_state(sibling_id, 0) == shown_state
    assert all(
        block.id != f"peer:{head}"
        for snapshot in surface.snapshots[scene_start:]
        for block in snapshot.blocks
    )


@pytest.mark.asyncio
async def test_scribe_inflight_whole_source_cannot_overwrite_out_of_order_group_results(route):
    harness, backend, presenter, _, surface, history, committed, diarized, _, _ = route
    backend.block = True
    parent = await committed("Initial left. Initial right.")
    await wait_until(lambda: len(backend.completions) == 1, timeout_s=2)
    await diarized(0, ("Initial left. ", "speaker-A"), ("Initial right.", "speaker-B"))
    await wait_until(lambda: len(backend.completions) == 3, timeout_s=2)
    assert not backend.completions[0].done()
    backend.completions[2].set_result(None)
    await wait_until(lambda: bool(presenter.snapshot().blocks), timeout_s=2)
    assert [block.primary_text for block in presenter.snapshot().blocks] == [
        "translated:Initial right."
    ]
    assert not history.conversation_records
    backend.completions[1].set_result(None)
    backend.completions[0].set_result(None)
    await _idle(harness)
    assert [record.source_text for record in history.conversation_records] == [
        "Initial left.",
        "Initial right.",
    ]
    assert any(block.id == f"peer:{parent}" for block in presenter.snapshot().blocks)
    assert all(
        "Initial left. Initial right." not in block.primary_text
        for snapshot in surface.snapshots
        for block in snapshot.blocks
    )


@pytest.mark.asyncio
async def test_admitted_scribe_split_survives_clean_socket_retirement(route):
    harness, backend, presenter, timeline, _, history, committed, diarized, socket, urls = route
    backend.block = True
    await committed("Accepted head. Accepted tail.")
    await wait_until(lambda: len(backend.completions) == 1, timeout_s=2)
    evidence = await diarized(0, ("Accepted head. ", "speaker-A"), ("Accepted tail.", "speaker-B"))
    await wait_until(lambda: len(backend.completions) == 3, timeout_s=2)
    socket.emit_close()
    await wait_until(lambda: socket.closed and not evidence.is_current(), timeout_s=2)
    for completion in backend.completions:
        completion.set_result(None)
    await _idle(harness)
    assert [record.source_text for record in history.conversation_records] == [
        "Accepted head.",
        "Accepted tail.",
    ]
    assert {block.secondary_text for block in presenter.snapshot().blocks} == {
        "Accepted head.",
        "Accepted tail.",
    }
    backend.block = False
    fresh = await committed("Fresh successor.")
    await _idle(harness)
    await timeline.advance(1.1)
    assert len(urls) == 2
    assert history.conversation_records[-1].source_text == "Fresh successor."
    assert any(block.id == f"peer:{fresh}" for block in presenter.snapshot().blocks)


@pytest.mark.asyncio
@pytest.mark.parametrize("speaker", ["speaker-A", "speaker-B"])
async def test_scribe_unknown_continuation_tracks_split_tail_boundary(route, monkeypatch, speaker):
    lines = 3.0
    monkeypatch.setattr(
        "puripuly_heart.core.orchestrator.peer_caption.estimate_peer_translation_lines",
        lambda *_, **__: lines,
    )
    harness, backend, _, timeline, _, history, committed, diarized, _, _ = route
    await committed("The first part, and the second part,")
    await _idle(harness)
    await timeline.advance(0.2)
    unknown = await committed("and an unassigned continuation", partial=True)
    await _idle(harness)
    assert history.conversation_records[-1].utterance_id == str(unknown)
    assert history.conversation_records[-1].source_text == "and an unassigned continuation"
    await diarized(0, ("The first part, ", "speaker-A"), ("and the second part,", "speaker-B"))
    await _idle(harness)
    lines = 2.0
    await timeline.advance(0.2)
    successor = await committed("and another part", ("and another part", speaker), partial=True)
    await _idle(harness)
    matching = speaker == "speaker-B"
    expected = "and an unassigned continuation and another part" if matching else "and another part"
    assert backend.calls[-1].text == expected
    assert history.conversation_records[-1].source_text == expected
    assert history.conversation_records[-1].utterance_id == str(unknown if matching else successor)
