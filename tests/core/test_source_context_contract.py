from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest

from puripuly_heart.core.audio.format import AudioCaptureSpan
from puripuly_heart.core.audio.ownership import (
    AudioSegmentIdentity,
    AudioSegmentSettingsSnapshot,
    AudioSegmentSnapshot,
    AudioSegmentTerminalReceipt,
    PeerAudioSegmentLedger,
)
from puripuly_heart.core.clock import FakeClock
from puripuly_heart.core.orchestrator.translation_channel_callbacks import (
    TranslationChannelOwnerCallbacks,
)
from puripuly_heart.core.orchestrator.translation_request import TranslationProcessRequest
from puripuly_heart.core.orchestrator.translation_turn import TranslationTurnRequest
from puripuly_heart.core.runtime.peer_channel import PeerCaptureSessionOwner
from puripuly_heart.core.stt.backend import (
    STTAudioBoundary,
    STTContributionConsumptionLedger,
    STTProviderEpochEnded,
    STTProviderTurnIdentity,
    STTProviderTurnRequest,
    STTProviderTurnTerminal,
    STTProviderTurnUpdate,
    STTSessionProjection,
    STTSourceRelease,
    STTTextContribution,
)
from puripuly_heart.core.stt.recognition_units import contribution_transcript
from puripuly_heart.core.stt.scoped_engine import ScopedRecognitionEngine
from puripuly_heart.core.stt.scoped_normalizer import STTScopedTurnNormalizer
from puripuly_heart.core.vad.gating import SpeechEnd, SpeechStart
from puripuly_heart.domain.models import FinalLanguageRun, FinalSpeakerRun, Transcript, Translation
from puripuly_heart.domain.recognition import (
    RecognitionStreamIdentity,
    RecognitionTextOrigin,
    RecognitionUnitIdentity,
)
from puripuly_heart.providers.stt.soniox import _SonioxSession
from tests.core.test_stt_scoped_engine import ControlledScopedSession, settings, span, watchdogs
from tests.helpers.fakes import RecordingOscQueue
from tests.helpers.translation_owners import compose_translation_test_harness


def source_fixture(order=1):
    settings = AudioSegmentSettingsSnapshot(
        "fixture", (), (), "manual", "en", ("en",), 16000, 0.5, 200, 500
    )
    segment = AudioSegmentIdentity(1, order, uuid4(), 7)
    span = AudioCaptureSpan(7, order, 48000, 0, 48000, 0.0, 1.0,
                            normalized_sample_rate_hz=16000,
                            normalized_start_sample=0, normalized_end_sample=16000)
    snapshot = AudioSegmentSnapshot(
        segment, settings, (span,), (), (), 16000, 0, 0, 0, 0, 0, True,
        "open", 0.0, None, None, display_group_id=uuid4(),
    )
    return snapshot, STTProviderTurnIdentity(segment, "provider-1", f"request-{order}")


class ControlledProvider:
    def __init__(self):
        self.calls = []
        self.gates = {}
        self.running = 0
        self.high_water = 0

    async def translate(self, *, utterance_id, text, target_language, context="", **kwargs):
        self.calls.append((text, target_language, context))
        self.running += 1
        self.high_water = max(self.high_water, self.running)
        try:
            gate = self.gates.get(text)
            if gate is not None:
                await gate.wait()
            if text == "fail":
                raise RuntimeError("controlled failure")
            return Translation(utterance_id, translated_text=f"{target_language}:{text}")
        finally:
            self.running -= 1

    async def close(self):
        return


class OutputTrace:
    def __init__(self, delegate):
        self.delegate = delegate
        self.items = []

    async def submit_translation_output(self, submission):
        self.items.append(submission)
        return await self.delegate.submit_translation_output(submission)


async def until(predicate):
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(0)


def harness_for(provider):
    harness = compose_translation_test_harness(
        stt=None, llm=provider, osc=RecordingOscQueue(), clock=FakeClock(10),
        source_language="en", target_language="ja", peer_source_language="en",
        peer_target_language="ja", integrated_context_enabled=True, concurrency_limit=2,
        translation_enabled=True, peer_translation_enabled=True,
    )
    harness.output_runtime.activate_peer_generation(1)
    trace = OutputTrace(harness.translation_turns.output)
    harness.translation_turns.output = trace
    return harness, trace


async def finish(harness):
    await harness.translation_turns.wait_for_idle()
    await harness.output_runtime.wait_for_peer_output_idle()


async def close(harness):
    await harness.translation_turns.close()
    await harness.output_runtime.close()


def test_partial_contribution_terminal_tail_and_retired_identity_accounting():
    snapshot, identity = source_fixture()
    normalizer = STTScopedTurnNormalizer(identity)
    update = normalizer.apply_update(STTProviderTurnUpdate(
        identity, 1, "stable", "append", "AB"
    ))
    ledger = STTContributionConsumptionLedger(identity_capacity=1)
    boundary = STTAudioBoundary("speaker-1", 7, 8000, 0.6, "speaker", "speaker-epoch")
    ledger.record_boundary(identity, boundary)
    ledger.seal_audio(identity)
    assert ledger.lifecycle(identity) == (True, False)
    assert ledger.boundaries(identity) == (boundary,)
    assert [part.text for part in ledger.consume_parts(update, text_range=(0, 1))] == ["A"]
    terminal = normalizer.apply_terminal(STTProviderTurnTerminal(
        identity, "final", "ABC", text_authority="authoritative"
    ))
    assert ledger.consume(terminal) == "BC"
    assert ledger.lifecycle(identity) == (True, True)
    assert ledger.consume(terminal) == ""
    _, successor = source_fixture(2)
    assert ledger.consume(STTProviderTurnTerminal(successor, "final", "ABC")) == "ABC"
    assert ledger.consume(update) == ""
    assert ledger.consume(terminal) == ""


def test_missing_or_malformed_terminal_ownership_cannot_become_gray_success():
    _, identity = source_fixture()
    ledger = STTContributionConsumptionLedger()
    update = STTProviderTurnUpdate(identity, 1, "stable", "replace", "A",
                                   contribution=STTTextContribution("c", 0, 1))
    assert ledger.consume(update) == "A"
    with pytest.raises(ValueError, match="association"):
        ledger.consume(STTProviderTurnTerminal(identity, "final", "AB"))
    with pytest.raises(ValueError, match="ranges"):
        ledger.consume(STTProviderTurnTerminal(identity, "final", "AB",
            included_contributions=(STTTextContribution("c", 0, 3),)))
    assert ledger.consume(STTProviderTurnTerminal(identity, "final", "AB",
        included_contributions=(update.contribution,))) == "B"


@pytest.mark.asyncio
async def test_batch_and_streaming_adapter_sources_reach_context_and_output_before_audio_closes():
    provider = ControlledProvider()
    provider.gates["A"] = asyncio.Event()
    harness, trace = harness_for(provider)
    snapshot, identity = source_fixture()
    normalizer = STTScopedTurnNormalizer(identity)
    try:
        first = normalizer.apply_update(STTProviderTurnUpdate(
            identity, 1, "stable", "append", "A", (FinalLanguageRun("A", "en"),),
            source_release=STTSourceRelease("first", 0, 1, "boundary"),
        ))
        first_event = await harness.peer_owner.handle_provider_turn_update(snapshot, first)
        await until(lambda: len(provider.calls) == 1)
        second = normalizer.apply_update(STTProviderTurnUpdate(
            identity, 2, "stable", "replace", "AB", (FinalLanguageRun("AB", "en"),),
            source_release=STTSourceRelease("second", 1, 2, "boundary"),
        ))
        await asyncio.gather(*(
            harness.peer_owner.handle_provider_turn_update(snapshot, second) for _ in range(4)
        ))
        await until(lambda: len(provider.calls) == 2)
        assert [entry.text for entry in harness.peer_runtime.translation_history] == ["A", "B"]
        assert provider.calls[0][2] == ""
        assert '"A"' in provider.calls[1][2]
        assert trace.items == []
        provider.gates["A"].set()
        await finish(harness)
        assert [item.source_text for item in trace.items] == ["A", "B"]
        assert snapshot.state == "open"
        terminal = normalizer.apply_terminal(STTProviderTurnTerminal(
            identity, "final", "ABC", (FinalLanguageRun("ABC", "en"),),
            text_authority="authoritative"
        ))
        sealed = replace(snapshot, state="sealed", sealed_at_monotonic_s=1.0, seal_reason="natural")
        receipt = AudioSegmentTerminalReceipt(identity.segment, "final", sealed, 1.1,
                                               identity.provider_epoch_id, identity.provider_turn_id)
        await harness.peer_owner.handle_provider_turn_terminal(receipt, terminal)
        await harness.peer_owner.handle_provider_turn_terminal(receipt, terminal)
        await finish(harness)
        batch_snapshot, batch_identity = source_fixture(2)
        batch_terminal = STTScopedTurnNormalizer(batch_identity).apply_terminal(
            STTProviderTurnTerminal(batch_identity, "final", "7", text_authority="authoritative")
        )
        batch_receipt = AudioSegmentTerminalReceipt(batch_identity.segment, "final", batch_snapshot, 2)
        await harness.peer_owner.handle_provider_turn_terminal(batch_receipt, batch_terminal)
        await finish(harness)
        assert [item.source_text for item in trace.items] == ["A", "B", "C", "7"]
        assert [entry.text for entry in harness.peer_runtime.translation_history] == ["A", "B", "C", "7"]
        assert [item.source_order for item in trace.items] == [1, 2, 3, 4]
        for item in trace.items:
            source = item.source_child
            assert source.version == 1 and source.effective_text == source.raw_text == item.source_text
            assert source.origins[0].ownership == "request"
            assert source.origins[0].source_ranges
            assert source.display_group_id in {snapshot.display_group_id, batch_snapshot.display_group_id}
        assert first_event.transcript.utterance_id != identity.segment.segment_id
        assert provider.high_water <= 2
    finally:
        provider.gates["A"].set()
        await close(harness)


@pytest.mark.asyncio
async def test_frozen_aba_sources_share_targets_and_increment_prior_sibling_context():
    provider = ControlledProvider()
    harness, trace = harness_for(provider)
    snapshot, identity = source_fixture()
    try:
        update = STTScopedTurnNormalizer(identity).apply_update(STTProviderTurnUpdate(
            identity, 1, "stable", "append", "ABA",
            final_language_runs=(FinalLanguageRun("ABA", "en"),),
            final_speaker_runs=tuple(FinalSpeakerRun(
                letter, letter, "speaker-epoch", source="nemotron", attribution_state="identified"
            ) for letter in "ABA"),
        ))
        transcript = contribution_transcript(
            snapshot, update, STTContributionConsumptionLedger().consume_parts(update),
            created_at=10, publication_order=1,
        )
        request = TranslationTurnRequest(transcript, "Peer", "peer", ("ja", "ko"),
                                         harness.peer_owner.config_snapshot())
        await harness.translation_turns.submit(request)
        await finish(harness)
        assert [entry.text for entry in harness.peer_runtime.translation_history] == ["A", "B", "A"]
        sources = [item.source_child for item in trace.items]
        assert len(trace.items) == 6
        assert sources[0] is sources[1] and sources[2] is sources[3] and sources[4] is sources[5]
        assert len({source.source_id for source in sources}) == 3
        assert [item.context_texts for item in trace.items] == [(), (), ("A",), ("A",), ("A", "B"), ("A", "B")]
        assert await harness.translation_turns.submit(request) == ()
        assert len(harness.peer_runtime.translation_history) == 3
        assert sources[0].display_group_id == snapshot.display_group_id
        assert [source.origins[0].contribution_range for source in sources[::2]] == [
            (0, 1), (1, 2), (2, 3)
        ]
    finally:
        await close(harness)


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["empty", "failed", "cancelled"])
async def test_terminal_resolution_does_not_cancel_admitted_prefix(outcome):
    provider = ControlledProvider()
    provider.gates["A"] = asyncio.Event()
    harness, trace = harness_for(provider)
    snapshot, identity = source_fixture()
    normalizer = STTScopedTurnNormalizer(identity)
    try:
        update = normalizer.apply_update(STTProviderTurnUpdate(
            identity, 1, "stable", "append", "A",
            source_release=STTSourceRelease("prefix", 0, 1, "boundary"),
        ))
        await harness.peer_owner.handle_provider_turn_update(snapshot, update)
        await until(lambda: len(provider.calls) == 1)
        terminal = normalizer.apply_terminal(STTProviderTurnTerminal(identity, outcome))
        receipt = AudioSegmentTerminalReceipt(identity.segment, outcome,
            replace(snapshot, state="sealed", sealed_at_monotonic_s=1, seal_reason="natural"), 1)
        await harness.peer_owner.handle_provider_turn_terminal(receipt, terminal)
        provider.gates["A"].set()
        await finish(harness)
        assert [item.source_text for item in trace.items] == ["A"]
        assert [entry.text for entry in harness.peer_runtime.translation_history] == ["A"]
    finally:
        provider.gates["A"].set()
        await close(harness)


@pytest.mark.asyncio
async def test_soniox_fixture_common_source_releases_are_distinct_from_tokens_and_wire_markers():
    provider = ControlledProvider()
    harness, trace = harness_for(provider)
    snapshot, identity = source_fixture()
    session = _SonioxSession(
        api_key="unused", model="fixture", endpoint="unused", sample_rate_hz=16000,
        language_hints=["en"], context_terms=[], keepalive_interval_s=100,
        trailing_silence_ms=100, connect_timeout_s=5,
        enable_language_identification=True,
        projection=STTSessionProjection("scoped", identity.provider_epoch_id),
    )
    session._ws = object()
    await session.begin_turn(STTProviderTurnRequest(identity, snapshot.settings))
    events = session.turn_events()
    normalizer = STTScopedTurnNormalizer(identity)
    release_reasons = []

    def message(*texts, finished=False):
        session._handle_message(json.dumps({
            "tokens": [{"text": text, "is_final": True, "language": "en"} for text in texts],
            "finished": finished,
        }))

    async def release(current_snapshot, current_normalizer, release_id, start, end, reason):
        ready = current_normalizer.release_source(STTSourceRelease(release_id, start, end, reason))
        release_reasons.append(reason)
        return await harness.peer_owner.handle_provider_turn_update(current_snapshot, ready)

    async def drain(current_snapshot, current_normalizer, terminal_release=None):
        while session._event_projection.scoped_event_depth:
            raw = await anext(events)
            if isinstance(raw, STTProviderTurnUpdate):
                update = current_normalizer.apply_update(raw)
                if update is not None:
                    await harness.peer_owner.handle_provider_turn_update(current_snapshot, update)
            elif isinstance(raw, STTProviderTurnTerminal):
                if terminal_release is not None:
                    await release(
                        current_snapshot, current_normalizer, terminal_release,
                        0, len(current_normalizer.stable_text), terminal_release,
                    )
                terminal = current_normalizer.apply_terminal(raw)
                receipt = AudioSegmentTerminalReceipt(
                    terminal.identity.segment, terminal.outcome, current_snapshot, 11,
                    terminal.identity.provider_epoch_id, terminal.identity.provider_turn_id,
                )
                await harness.peer_owner.handle_provider_turn_terminal(receipt, terminal)

    try:
        boundary = STTAudioBoundary("audio-change", 7, 8000, 0.6, "speaker", "nemotron-1")
        harness.peer_owner.note_audio_boundary(identity, boundary)
        message("Hello", " world")
        await drain(snapshot, normalizer)
        assert provider.calls == trace.items == harness.peer_runtime.translation_history == []
        message("<end>")
        await drain(snapshot, normalizer)
        assert provider.calls == trace.items == harness.peer_runtime.translation_history == []
        await release(snapshot, normalizer, "automatic-end", 0, 11, "end")
        await finish(harness)
        assert [item.source_text for item in trace.items] == ["Hello world"]
        assert len(trace.items[0].source_child.origins) == 2
        assert snapshot.state == "open"
        message("<end>", "<end>")
        await drain(snapshot, normalizer)
        assert await release(snapshot, normalizer, "automatic-end", 0, 11, "end") is None
        assert await release(snapshot, normalizer, "empty-end", 11, 11, "end") is None
        assert len(trace.items) == len(harness.peer_runtime.translation_history) == 1
        assert session._pending_finalize_requests == 0
        session._event_projection.seal(identity)
        session._pending_finalize_requests = 1
        message("Hello", " world", "<end>")
        await drain(snapshot, normalizer)
        assert len(provider.calls) == 1
        assert session._pending_finalize_requests == 1
        message("<fin>")
        await drain(snapshot, normalizer, terminal_release="manual")
        await finish(harness)
        assert [item.source_text for item in trace.items] == ["Hello world", "Hello world"]
        assert [entry.text for entry in harness.peer_runtime.translation_history] == [
            "Hello world", "Hello world"
        ]
        assert session._pending_finalize_requests == 0
        second_snapshot, second_identity = source_fixture(2)
        await session.begin_turn(STTProviderTurnRequest(second_identity, second_snapshot.settings))
        second_normalizer = STTScopedTurnNormalizer(second_identity)
        message("tail", " piece", finished=True)
        await drain(second_snapshot, second_normalizer, terminal_release="drain")
        await finish(harness)
        assert [item.source_text for item in trace.items] == [
            "Hello world", "Hello world", "tail piece"
        ]
        assert [entry.text for entry in harness.peer_runtime.translation_history] == [
            "Hello world", "Hello world", "tail piece"
        ]
        assert release_reasons == ["end", "end", "end", "manual", "drain"]
        assert harness.peer_owner._stt_consumption.boundaries(identity) == (boundary,)
        assert snapshot.opened_at_monotonic_s == 0
        assert snapshot.context_ranges == ()
        assert len(provider.calls) == 3
        assert harness.output_runtime.chatbox_is_denied("peer")
    finally:
        await events.aclose()
        await close(harness)


@pytest.mark.asyncio
async def test_failed_predecessor_releases_successful_source_successor_in_channel_order():
    provider = ControlledProvider()
    provider.gates["fail"] = asyncio.Event()
    harness, trace = harness_for(provider)
    try:
        for order, text in ((1, "fail"), (2, "success")):
            snapshot, identity = source_fixture(order)
            terminal = STTScopedTurnNormalizer(identity).apply_terminal(
                STTProviderTurnTerminal(identity, "final", text, text_authority="authoritative")
            )
            receipt = AudioSegmentTerminalReceipt(identity.segment, "final", snapshot, 11)
            await harness.peer_owner.handle_provider_turn_terminal(receipt, terminal)
        await until(lambda: len(provider.calls) == 2 and provider.running == 1)
        assert trace.items == []
        provider.gates["fail"].set()
        await finish(harness)
        assert [item.source_text for item in trace.items] == ["fail", "success"]
        assert trace.items[1].outcome == "translated"
        assert provider.high_water == 2
    finally:
        provider.gates["fail"].set()
        await close(harness)


@pytest.mark.asyncio
async def test_scoped_prefix_callbacks_fence_old_provider_epochs_settings_stop_and_races():
    provider = ControlledProvider()
    harness, trace = harness_for(provider)
    snapshot, identity = source_fixture()
    capture = object.__new__(PeerCaptureSessionOwner)
    capture._publication_generations = {1}
    capture._segment_ledgers = [SimpleNamespace(snapshots=(snapshot,), terminal_receipts=())]
    callbacks = TranslationChannelOwnerCallbacks(harness.stt_sessions)
    callbacks.bind_peer(harness.peer_owner)
    callbacks.bind_peer_capture(capture)
    current = True
    harness.peer_owner.local_asr_runtime = SimpleNamespace(
        is_current_recognition_stream=lambda channel, stream: current and stream.provider_epoch_id == identity.provider_epoch_id
    )
    update = STTScopedTurnNormalizer(identity).apply_update(
        STTProviderTurnUpdate(
            identity, 1, "stable", "append", "A",
            source_release=STTSourceRelease("prefix", 0, 1, "boundary"),
        )
    )
    try:
        old_epoch = replace(update, identity=replace(identity, provider_epoch_id="old-provider"))
        wrong_settings = replace(update, identity=replace(identity, settings_scope=("wrong",)))
        await callbacks.peer_event_handler(old_epoch)
        await callbacks.peer_event_handler(wrong_settings)
        assert provider.calls == []
        await asyncio.gather(*(callbacks.peer_event_handler(update) for _ in range(4)))
        await callbacks.retired_event_handler(update)
        await finish(harness)
        assert [entry.text for entry in harness.peer_runtime.translation_history] == ["A"]
        assert [item.source_text for item in trace.items] == ["A"]
        current = False
        late = replace(update, sequence=2, text="AB", contribution=STTTextContribution("late", 1, 2))
        await callbacks.peer_event_handler(late)
        capture._publication_generations.clear()
        await harness.peer_owner.close_ingress()
        await callbacks.peer_event_handler(late)
        assert len(provider.calls) == len(trace.items) == 1
    finally:
        await close(harness)


@pytest.mark.asyncio
async def test_prestarted_attempt_and_retry_replay_do_not_add_source_history():
    provider = ControlledProvider()
    harness, trace = harness_for(provider)
    owner = harness.translation_requests
    source_id = uuid4()
    request = TranslationProcessRequest(
        source_id, source_id, 0, "A", "self", "Mic", "ja", "integrated_preferred",
        harness.self_owner.config_snapshot(),
        expected_provider_generation=owner.provider_generation, prestarted=True,
    )
    try:
        result = await owner.process(request)
        assert result.outcome == "translated"
        assert harness.self_runtime.translation_history == []
        admitted = replace(request, prestarted=False)
        prepared = owner.admit((admitted,))
        assert prepared[source_id].context_texts == ()
        await owner.process(admitted, prepared=prepared[source_id])
        await owner.process(admitted)
        assert [entry.text for entry in harness.self_runtime.translation_history] == ["A"]
        assert all(call[2] == "" for call in provider.calls)
    finally:
        await close(harness)


def test_unfinished_source_capacity_keeps_owned_prefix_until_terminal_resolution():
    _, identity = source_fixture()
    _, next_identity = source_fixture(2)
    ledger = STTContributionConsumptionLedger(identity_capacity=1)
    contribution = STTTextContribution("prefix", 0, 1)
    update = STTProviderTurnUpdate(identity, 1, "stable", "replace", "A", contribution=contribution)
    assert ledger.consume(update) == "A"
    with pytest.raises(ValueError, match="unfinished source"):
        ledger.consume(STTProviderTurnTerminal(next_identity, "final", "A"))
    assert ledger.consume(STTProviderTurnTerminal(
        identity, "final", "AB", included_contributions=(contribution,)
    )) == "B"
    assert ledger.consume(STTProviderTurnTerminal(next_identity, "final", "A")) == "A"
    assert ledger.consume(update) == ""


@pytest.mark.asyncio
async def test_malformed_source_origin_is_rejected_before_gray_publication():
    harness, trace = harness_for(ControlledProvider())
    identity = RecognitionUnitIdentity(
        RecognitionStreamIdentity("peer", 1, 7, "provider", ()), uuid4(), 1
    )
    request = TranslationTurnRequest(
        Transcript(uuid4(), "AB", True, channel="peer", publication_generation=1, source_order=1,
                   recognition_origins=(RecognitionTextOrigin(identity, 1, 2),)),
        "Peer", "peer", ("ja",), harness.peer_owner.config_snapshot(),
    )
    try:
        with pytest.raises(ValueError, match="provenance"):
            await harness.translation_turns.submit(request)
        assert trace.items == []
        assert harness.peer_runtime.translation_history == []
    finally:
        await close(harness)


@pytest.mark.asyncio
async def test_failed_middle_parent_does_not_break_channel_predecessor_chain():
    provider = ControlledProvider()
    provider.gates["A"] = asyncio.Event()
    harness, trace = harness_for(provider)
    delegate = harness.translation_turns.on_child_created

    async def created(child):
        if child.transcript.text == "bad":
            raise RuntimeError("controlled child creation failure")
        await delegate(child)

    harness.translation_turns.on_child_created = created
    try:
        for order, text in ((1, "A"), (2, "bad"), (3, "C")):
            snapshot, identity = source_fixture(order)
            terminal = STTScopedTurnNormalizer(identity).apply_terminal(
                STTProviderTurnTerminal(identity, "final", text, text_authority="authoritative")
            )
            await harness.peer_owner.handle_provider_turn_terminal(
                AudioSegmentTerminalReceipt(identity.segment, "final", snapshot, 11), terminal
            )
        await until(lambda: len(provider.calls) == 2 and provider.running == 1)
        assert trace.items == []
        provider.gates["A"].set()
        await finish(harness)
        assert [item.source_text for item in trace.items] == ["A", "C"]
        assert [item.source_order for item in trace.items] == [1, 3]
        assert [entry.text for entry in harness.peer_runtime.translation_history] == ["A", "bad", "C"]
        with pytest.raises(RuntimeError, match="terminally rejected"):
            await harness.peer_owner.submit_translation_output(trace.items[-1])
        assert len(provider.calls) == 2
        assert len(harness.peer_runtime.translation_history) == 3
    finally:
        provider.gates["A"].set()
        await close(harness)


@pytest.mark.asyncio
async def test_explicit_terminal_provenance_preserves_admitted_source_despite_whole_final_disagreement():
    harness, trace = harness_for(ControlledProvider())
    snapshot, identity = source_fixture()
    normalizer = STTScopedTurnNormalizer(identity)
    try:
        update = normalizer.apply_update(STTProviderTurnUpdate(
            identity, 1, "stable", "append", "A",
            source_release=STTSourceRelease("prefix", 0, 1, "boundary"),
        ))
        await harness.peer_owner.handle_provider_turn_update(snapshot, update)
        await finish(harness)
        admitted_source = trace.items[0].source_child
        terminal = normalizer.apply_terminal(STTProviderTurnTerminal(
            identity, "final", "Xtail", text_authority="authoritative",
            included_contributions=(update.contribution,),
        ))
        receipt = AudioSegmentTerminalReceipt(identity.segment, "final", snapshot, 11)
        await harness.peer_owner.handle_provider_turn_terminal(receipt, terminal)
        await finish(harness)
        assert [item.source_text for item in trace.items] == ["A", "tail"]
        assert admitted_source.raw_text == admitted_source.effective_text == "A"
        assert [entry.text for entry in harness.peer_runtime.translation_history] == ["A", "tail"]
        assert trace.items[1].source_child.origins[0].contribution_range == (1, 5)
    finally:
        await close(harness)


@pytest.mark.asyncio
@pytest.mark.parametrize("composition", ["language", "speaker"])
@pytest.mark.parametrize("owned_range", [(0, 1), (1, 2)])
async def test_parent_provenance_hole_cannot_disappear_at_composed_segment_boundary(
    composition, owned_range
):
    provider = ControlledProvider()
    harness, trace = harness_for(provider)
    identity = RecognitionUnitIdentity(
        RecognitionStreamIdentity("peer", 1, 7, "provider", ()), uuid4(), 1
    )
    transcript = Transcript(
        uuid4(), "AB", True, channel="peer", publication_generation=1, source_order=1,
        recognition_origins=(RecognitionTextOrigin(identity, *owned_range),),
        final_language_runs=(
            (FinalLanguageRun("A", "en"), FinalLanguageRun("B", "ko"))
            if composition == "language" else ()
        ),
        final_speaker_runs=(
            (FinalSpeakerRun("A", "1", "epoch"), FinalSpeakerRun("B", "2", "epoch"))
            if composition == "speaker" else ()
        ),
    )
    request = TranslationTurnRequest(
        transcript, "Peer", "peer", ("ja",), harness.peer_owner.config_snapshot()
    )
    try:
        with pytest.raises(ValueError, match="provenance"):
            await harness.translation_turns.submit(request)
        assert provider.calls == trace.items == harness.peer_runtime.translation_history == []
    finally:
        await close(harness)


@pytest.mark.asyncio
async def test_origin_free_manual_legacy_composition_remains_valid():
    harness, trace = harness_for(ControlledProvider())
    request = TranslationTurnRequest(
        Transcript(
            uuid4(), "AB", True, channel="self",
            final_language_runs=(FinalLanguageRun("A", "en"), FinalLanguageRun("B", "ko")),
        ),
        "Manual", "manual", ("ja",), harness.self_owner.config_snapshot(),
    )
    try:
        await harness.translation_turns.submit(request)
        await finish(harness)
        assert [item.source_text for item in trace.items] == ["A", "B"]
        assert all(item.source_child.origins == () for item in trace.items)
        assert all(item.outcome == "translated" for item in trace.items)
    finally:
        await close(harness)


@pytest.mark.asyncio
@pytest.mark.parametrize("retirement", ["stop", "replacement", "forgotten-capture", "epoch"])
async def test_obsolete_peer_retirement_reclaims_ledger_without_cancelling_admitted_source(retirement):
    provider = ControlledProvider()
    provider.gates["A"] = asyncio.Event()
    harness, trace = harness_for(provider)
    harness.peer_owner._stt_consumption = STTContributionConsumptionLedger(
        identity_capacity=1, contribution_capacity=1
    )
    snapshot, identity = source_fixture()
    capture = object.__new__(PeerCaptureSessionOwner)
    capture._publication_generations = {1}
    capture._segment_ledgers = [SimpleNamespace(snapshots=(snapshot,), terminal_receipts=())]
    callbacks = TranslationChannelOwnerCallbacks(harness.stt_sessions)
    callbacks.bind_peer(harness.peer_owner)
    callbacks.bind_peer_capture(capture)
    current_identity = identity
    harness.peer_owner.local_asr_runtime = SimpleNamespace(
        is_current_recognition_stream=lambda channel, stream: (
            stream.provider_epoch_id == current_identity.provider_epoch_id
        )
    )
    normalizer = STTScopedTurnNormalizer(identity)
    normalizer.apply_update(STTProviderTurnUpdate(identity, 1, "stable", "append", "A"))
    ready = normalizer.release_source(STTSourceRelease("prefix", 0, 1, "boundary"))
    try:
        await callbacks.peer_event_handler(ready)
        await until(lambda: len(provider.calls) == 1)
        capture._publication_generations = {2} if retirement == "replacement" else set()
        if retirement == "forgotten-capture":
            capture._segment_ledgers.clear()
        if retirement == "epoch":
            await callbacks.retired_event_handler(
                STTProviderEpochEnded(identity.provider_epoch_id, False, "cancelled")
            )
        else:
            await callbacks.retired_event_handler(STTProviderTurnTerminal(identity, "cancelled"))
        assert harness.peer_owner._stt_consumption.lifecycle(identity)[1]
        assert harness.peer_owner._stt_consumption.consumed_contribution_ids == frozenset()
        provider.gates["A"].set()
        await finish(harness)
        assert [item.source_text for item in trace.items] == ["A"]
        assert trace.items[0].outcome == "translated"
        next_snapshot, next_identity = source_fixture(2)
        current_identity = replace(next_identity, provider_epoch_id="provider-2")
        capture._publication_generations = {1}
        capture._segment_ledgers.append(
            SimpleNamespace(snapshots=(next_snapshot,), terminal_receipts=())
        )
        next_normalizer = STTScopedTurnNormalizer(current_identity)
        next_normalizer.apply_update(
            STTProviderTurnUpdate(current_identity, 1, "stable", "append", "B")
        )
        await callbacks.peer_event_handler(
            next_normalizer.release_source(STTSourceRelease("next", 0, 1, "boundary"))
        )
        await finish(harness)
        assert [item.source_text for item in trace.items] == ["A", "B"]
        assert await harness.peer_owner.handle_provider_turn_update(snapshot, ready) is None
        assert len(provider.calls) == len(harness.peer_runtime.translation_history) == 2
    finally:
        provider.gates["A"].set()
        await close(harness)


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["reset", "close", "language"])
async def test_peer_owner_lifecycle_retires_unresolved_recognition_and_fences_replay(operation):
    provider = ControlledProvider()
    harness, trace = harness_for(provider)
    harness.peer_owner._stt_consumption = STTContributionConsumptionLedger(
        identity_capacity=1, contribution_capacity=1
    )
    snapshot, identity = source_fixture()
    normalizer = STTScopedTurnNormalizer(identity)
    normalizer.apply_update(STTProviderTurnUpdate(identity, 1, "stable", "append", "A"))
    ready = normalizer.release_source(STTSourceRelease("prefix", 0, 1, "boundary"))
    try:
        await harness.peer_owner.handle_provider_turn_update(snapshot, ready)
        await finish(harness)
        if operation == "reset":
            await harness.peer_owner.reset_provider_channel("peer")
        elif operation == "close":
            await harness.peer_owner.close()
            await harness.peer_owner.open_ingress()
        else:
            await harness.peer_owner.clear_language_runtime_state(channel="peer")
        assert harness.peer_owner._stt_consumption.lifecycle(identity)[1]
        assert harness.peer_owner._stt_consumption.consumed_contribution_ids == frozenset()
        next_snapshot, next_identity = source_fixture(2)
        if operation == "close":
            next_segment = replace(next_snapshot.identity, activation_generation=2)
            next_snapshot = replace(next_snapshot, identity=next_segment)
            next_identity = replace(next_identity, segment=next_segment)
            harness.output_runtime.activate_peer_generation(2)
        next_normalizer = STTScopedTurnNormalizer(next_identity)
        next_normalizer.apply_update(
            STTProviderTurnUpdate(next_identity, 1, "stable", "append", "B")
        )
        await harness.peer_owner.handle_provider_turn_update(
            next_snapshot,
            next_normalizer.release_source(STTSourceRelease("next", 0, 1, "boundary")),
        )
        await finish(harness)
        assert [item.source_text for item in trace.items] == ["A", "B"]
        count = len(harness.peer_runtime.translation_history)
        assert await harness.peer_owner.handle_provider_turn_update(snapshot, ready) is None
        assert len(provider.calls) == 2
        assert len(harness.peer_runtime.translation_history) == count
    finally:
        await close(harness)


def test_explicit_release_conserves_multi_contribution_partial_and_overlapping_ranges():
    _, identity = source_fixture()
    ledger = STTContributionConsumptionLedger()
    contributions = (STTTextContribution("one", 0, 2), STTTextContribution("two", 2, 4))
    update = STTProviderTurnUpdate(
        identity, 1, "stable", "replace", "ABCD",
        source_release=STTSourceRelease("first", 0, 3, "end"),
        included_contributions=contributions,
    )
    parts = ledger.consume_parts(update)
    assert [(piece.text, piece.text_start, piece.text_end) for piece in parts] == [
        ("AB", 0, 2), ("C", 2, 3)
    ]
    assert ledger.consume(update) == ""
    with pytest.raises(ValueError, match="release identity changed"):
        ledger.consume(replace(update, source_release=STTSourceRelease("first", 0, 4, "end")))
    with pytest.raises(ValueError, match="release identity changed"):
        ledger.consume(replace(update, source_release=STTSourceRelease("first", 0, 3, "manual")))
    assert ledger.consume(replace(
        update, source_release=STTSourceRelease("second", 1, 4, "manual")
    )) == "D"
    assert ledger.consume(STTProviderTurnTerminal(
        identity, "final", "ABCDE", included_contributions=contributions
    )) == "E"


@pytest.mark.parametrize(
    "malformation",
    ["hole", "missing", "overlap", "duplicate", "range", "unstable", "request", "epoch"],
)
def test_malformed_explicit_release_is_rejected_before_any_range_consumption(malformation):
    _, identity = source_fixture()
    ledger = STTContributionConsumptionLedger()
    complete = (STTTextContribution("one", 0, 2), STTTextContribution("two", 2, 4))
    cases = {
        "hole": (STTTextContribution("one", 0, 2), STTTextContribution("two", 3, 4)),
        "missing": (),
        "overlap": (STTTextContribution("one", 0, 3), STTTextContribution("two", 2, 4)),
        "duplicate": (STTTextContribution("one", 0, 2), STTTextContribution("one", 2, 4)),
        "range": complete,
        "unstable": complete,
        "request": complete,
        "epoch": complete,
    }
    declared_identity = (
        replace(identity, provider_turn_id="") if malformation == "request"
        else replace(identity, provider_epoch_id="") if malformation == "epoch" else identity
    )
    update = STTProviderTurnUpdate(
        declared_identity, 1, "provisional" if malformation == "unstable" else "stable",
        "replace", "ABCD",
        source_release=STTSourceRelease("ready", 0, 5 if malformation == "range" else 4, "end"),
        included_contributions=cases[malformation],
    )
    with pytest.raises(ValueError, match="source release"):
        ledger.consume(update)
    assert ledger.consumed_contribution_ids == frozenset()
    assert ledger._releases == {}
    assert ledger.consume(replace(
        update, identity=identity, stability="stable",
        source_release=STTSourceRelease("ready", 0, 4, "end"),
        included_contributions=complete,
    )) == "ABCD"


def test_retained_release_state_is_bounded_and_reclaimed_without_identity_readmission():
    _, identity = source_fixture()
    ledger = STTContributionConsumptionLedger(identity_capacity=1)
    for index in range(256):
        assert ledger.consume(STTProviderTurnUpdate(
            identity, index, "stable", "replace", "",
            source_release=STTSourceRelease(str(index), 0, 0, "end"),
        )) == ""
    with pytest.raises(ValueError, match="release capacity"):
        ledger.consume(STTProviderTurnUpdate(
            identity, 257, "stable", "replace", "",
            source_release=STTSourceRelease("overflow", 0, 0, "end"),
        ))
    assert ledger.retire_known(identity)
    assert ledger._releases == {}
    _, next_identity = source_fixture(2)
    assert ledger.consume(STTProviderTurnUpdate(
        next_identity, 1, "stable", "replace", "B",
        source_release=STTSourceRelease("new", 0, 1, "boundary"),
        included_contributions=(STTTextContribution("new-token", 0, 1),),
    )) == "B"
    assert ledger.consume(STTProviderTurnUpdate(
        identity, 258, "stable", "replace", "A",
        source_release=STTSourceRelease("late", 0, 1, "boundary"),
        included_contributions=(STTTextContribution("late-token", 0, 1),),
    )) == ""


@pytest.mark.asyncio
async def test_soniox_default_final_tokens_remain_one_terminal_source_without_common_release():
    provider = ControlledProvider()
    harness, trace = harness_for(provider)
    snapshot, identity = source_fixture()
    session = _SonioxSession(
        api_key="unused", model="fixture", endpoint="unused", sample_rate_hz=16000,
        language_hints=["en"], context_terms=[], keepalive_interval_s=100,
        trailing_silence_ms=100, connect_timeout_s=5,
        enable_language_identification=True,
        projection=STTSessionProjection("scoped", identity.provider_epoch_id),
    )
    session._ws = object()
    await session.begin_turn(STTProviderTurnRequest(identity, snapshot.settings))
    events = session.turn_events()
    normalizer = STTScopedTurnNormalizer(identity)
    try:
        session._handle_message(json.dumps({
            "tokens": [{"text": text, "is_final": True, "language": "en"}
                       for text in ("Hello", " world")],
        }))
        while session._event_projection.scoped_event_depth:
            raw = await anext(events)
            assert isinstance(raw, STTProviderTurnUpdate)
            update = normalizer.apply_update(raw)
            assert update.source_release is None
            assert await harness.peer_owner.handle_provider_turn_update(snapshot, update) is None
        assert provider.calls == trace.items == harness.peer_runtime.translation_history == []
        session._handle_message(json.dumps({"tokens": [], "finished": True}))
        raw = await anext(events)
        assert isinstance(raw, STTProviderTurnTerminal)
        terminal = normalizer.apply_terminal(raw)
        await harness.peer_owner.handle_provider_turn_terminal(
            AudioSegmentTerminalReceipt(identity.segment, "final", snapshot, 11), terminal
        )
        await finish(harness)
        assert [item.source_text for item in trace.items] == ["Hello world"]
        assert [entry.text for entry in harness.peer_runtime.translation_history] == ["Hello world"]
        assert len(provider.calls) == 1
        assert len(trace.items[0].source_child.origins) == 2
    finally:
        await events.aclose()
        await close(harness)


@pytest.mark.asyncio
async def test_explicit_release_checks_capture_owner_before_consumption():
    provider = ControlledProvider()
    harness, trace = harness_for(provider)
    snapshot, identity = source_fixture()
    wrong_snapshot, _ = source_fixture(2)
    normalizer = STTScopedTurnNormalizer(identity)
    normalizer.apply_update(STTProviderTurnUpdate(identity, 1, "stable", "append", "AB"))
    ready = normalizer.release_source(STTSourceRelease("ready", 0, 2, "boundary"))
    try:
        with pytest.raises(ValueError, match="source request identity mismatch"):
            await harness.peer_owner.handle_provider_turn_update(wrong_snapshot, ready)
        assert harness.peer_owner._stt_consumption.consumed_contribution_ids == frozenset()
        assert harness.peer_owner._stt_consumption._releases == {}
        await harness.peer_owner.handle_provider_turn_update(snapshot, ready)
        await finish(harness)
        assert [item.source_text for item in trace.items] == ["AB"]
        assert [entry.text for entry in harness.peer_runtime.translation_history] == ["AB"]
    finally:
        await close(harness)


@pytest.mark.parametrize("selection", [None, (1, 2)])
def test_explicit_release_rejects_skipped_prefix_atomically_then_reconciles_tail(selection):
    _, identity = source_fixture()
    normalizer = STTScopedTurnNormalizer(identity)
    normalizer.apply_update(STTProviderTurnUpdate(identity, 1, "stable", "append", "AB"))
    ledger = STTContributionConsumptionLedger()
    later = normalizer.release_source(STTSourceRelease(
        "ready", 1 if selection is None else 0, 2, "end"
    ))
    with pytest.raises(ValueError, match="skips earlier unconsumed"):
        ledger.consume_parts(later, text_range=selection)
    assert ledger.consumed_contribution_ids == frozenset()
    assert ledger._releases == ledger._admitted_prefixes == {}
    prefix = normalizer.release_source(STTSourceRelease("ready", 0, 1, "end"))
    assert ledger.consume(prefix) == "A"
    assert ledger._admitted_prefixes[identity] == 1
    assert ledger.consume(prefix) == ""
    terminal = normalizer.apply_terminal(STTProviderTurnTerminal(
        identity, "final", "ABC", text_authority="authoritative"
    ))
    assert ledger.consume(terminal) == "BC"
    assert ledger._releases == ledger._admitted_prefixes == {}


@pytest.mark.asyncio
async def test_rejected_out_of_order_source_release_adds_no_context_jobs_or_publications():
    provider = ControlledProvider()
    harness, trace = harness_for(provider)
    snapshot, identity = source_fixture()
    normalizer = STTScopedTurnNormalizer(identity)
    normalizer.apply_update(STTProviderTurnUpdate(identity, 1, "stable", "append", "AB"))
    try:
        with pytest.raises(ValueError, match="skips earlier unconsumed"):
            await harness.peer_owner.handle_provider_turn_update(
                snapshot, normalizer.release_source(STTSourceRelease("later", 1, 2, "end"))
            )
        assert provider.calls == trace.items == harness.peer_runtime.translation_history == []
        assert harness.peer_owner._stt_consumption.consumed_contribution_ids == frozenset()
        assert harness.peer_owner._stt_consumption._releases == {}
        await harness.peer_owner.handle_provider_turn_update(
            snapshot, normalizer.release_source(STTSourceRelease("earlier", 0, 1, "end"))
        )
        await until(lambda: len(provider.calls) == 1)
        await harness.peer_owner.handle_provider_turn_update(
            snapshot, normalizer.release_source(STTSourceRelease("later", 1, 2, "end"))
        )
        await finish(harness)
        assert [item.source_text for item in trace.items] == ["A", "B"]
        assert [entry.text for entry in harness.peer_runtime.translation_history] == ["A", "B"]
        assert provider.calls[0][2] == ""
        assert '"A"' in provider.calls[1][2]
    finally:
        await close(harness)


@pytest.mark.asyncio
@pytest.mark.parametrize("orderly", [False, True])
@pytest.mark.parametrize("retired_route", [False, True])
async def test_actual_engine_epoch_before_local_seal_preserves_lawful_terminal_tail(
    orderly, retired_route
):
    provider = ControlledProvider()
    harness, trace = harness_for(provider)
    harness.peer_owner._stt_consumption = STTContributionConsumptionLedger(identity_capacity=1)
    ledger = PeerAudioSegmentLedger(activation_generation=1, settings=settings("soniox"))
    segment_id = uuid4()
    start = ledger.observe_vad_event(
        SpeechStart(
            segment_id, np.empty(0, dtype=np.float32), np.full(4, 0.1, dtype=np.float32),
            chunk_capture=(span(1, 0, 4),),
        ),
        now_monotonic_s=10,
    )
    capture = object.__new__(PeerCaptureSessionOwner)
    capture._publication_generations = {1}
    capture._segment_ledgers = [ledger]
    capture._provider_terminal_events = {}
    capture.clock = FakeClock(10)
    callbacks = TranslationChannelOwnerCallbacks(harness.stt_sessions)
    callbacks.bind_peer(harness.peer_owner)
    callbacks.bind_peer_capture(capture)
    session = ControlledScopedSession()
    delivered = []

    async def receive(event):
        if retired_route and isinstance(event, STTProviderEpochEnded):
            await callbacks.retired_event_handler(event)
        else:
            await callbacks.peer_event_handler(event)
        delivered.append(event)

    engine = ScopedRecognitionEngine(
        session_factory=lambda _settings, _epoch: asyncio.sleep(0, result=session),
        event_sink=receive, watchdog_resolver=lambda _settings: watchdogs(final_timeout_s=0.3),
    )
    harness.peer_owner.local_asr_runtime = SimpleNamespace(
        is_current_recognition_stream=lambda channel, stream: engine.is_current_recognition_stream(stream)
    )
    try:
        await engine.handle_owned_vad_event(start)
        identity = session.requests[0].identity
        session.emit(STTProviderTurnUpdate(
            identity, 1, "stable", "append", "AB",
            source_release=STTSourceRelease("prefix", 0, 1, "boundary"),
        ))
        await until(lambda: len(provider.calls) == 1)
        await finish(harness)
        assert [item.source_text for item in trace.items] == ["A"]
        session.emit(STTProviderTurnTerminal(
            identity, "degraded", "AB", text_authority="degraded",
            epoch_disposition="retire", failure_reason="soniox_stream_finished",
        ))
        session.emit(STTProviderEpochEnded(
            identity.provider_epoch_id, orderly, "soniox_stream_finished"
        ))
        await until(lambda: any(isinstance(event, STTProviderEpochEnded) for event in delivered))
        assert engine._turns[identity].terminal_ready.done()
        assert not engine._turns[identity].local_sealed
        assert ledger.snapshots[0].state == "open"
        assert not any(isinstance(event, STTProviderTurnTerminal) for event in delivered)
        assert harness.peer_owner._stt_consumption.lifecycle(identity) == (False, False)
        assert [entry.text for entry in harness.peer_runtime.translation_history] == ["A"]
        end = ledger.observe_vad_event(
            SpeechEnd(segment_id, trailing_silence_ms=224, reason="silence"),
            now_monotonic_s=11,
        )
        await engine.handle_owned_vad_event(end)
        await finish(harness)
        assert [item.source_text for item in trace.items] == ["A", "B"]
        assert [entry.text for entry in harness.peer_runtime.translation_history] == ["A", "B"]
        assert provider.calls[0][2] == ""
        assert '"A"' in provider.calls[1][2]
        assert harness.peer_owner._stt_consumption.lifecycle(identity) == (True, True)
        assert harness.peer_owner._stt_consumption.consumed_contribution_ids == frozenset()
        assert harness.peer_owner._stt_consumption._releases == {}
        assert harness.peer_owner._stt_consumption._admitted_prefixes == {}
        next_identity = replace(
            identity, segment=replace(identity.segment, segment_order=2, segment_id=uuid4()),
            provider_turn_id="next",
        )
        assert harness.peer_owner._stt_consumption.consume(
            STTProviderTurnTerminal(next_identity, "final", "next")
        ) == "next"
        assert harness.peer_owner._stt_consumption.consume(STTProviderTurnUpdate(
            identity, 2, "stable", "replace", "AB",
            source_release=STTSourceRelease("late", 0, 2, "end"),
            included_contributions=(STTTextContribution(
                f"{identity.provider_turn_id}:1", 0, 2
            ),),
        )) == ""
    finally:
        await engine.close()
        await close(harness)


def test_epoch_reclamation_selects_obsolete_identity_and_preserves_older_pending_tail():
    _, pending = source_fixture(1)
    _, obsolete = source_fixture(2)
    ledger = STTContributionConsumptionLedger(identity_capacity=2)
    contribution = STTTextContribution("pending-text", 0, 2)
    pending_update = STTProviderTurnUpdate(
        pending, 1, "stable", "replace", "AB", contribution=contribution,
        source_release=STTSourceRelease("pending-prefix", 0, 1, "boundary"),
    )
    obsolete_update = STTProviderTurnUpdate(
        obsolete, 1, "stable", "replace", "C", contribution=STTTextContribution("old", 0, 1),
        source_release=STTSourceRelease("old-prefix", 0, 1, "boundary"),
    )
    assert ledger.consume(pending_update) == "A"
    assert ledger.consume(obsolete_update) == "C"
    assert ledger.retire_epoch(
        pending.provider_epoch_id, is_obsolete=lambda identity: identity == obsolete
    )
    assert ledger.lifecycle(pending) == (False, False)
    assert ledger.lifecycle(obsolete) == (False, True)
    assert ledger._admitted_prefixes == {pending: 1}
    _, next_identity = source_fixture(3)
    assert ledger.consume(STTProviderTurnTerminal(next_identity, "final", "next")) == "next"
    assert len(ledger._known_identities) == 2
    assert ledger.consume(STTProviderTurnTerminal(
        pending, "degraded", "AB", included_contributions=(contribution,)
    )) == "B"
    assert ledger.consume(obsolete_update) == ""
    assert ledger._releases == ledger._admitted_prefixes == {}
