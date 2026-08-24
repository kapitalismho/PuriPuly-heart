from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from uuid import uuid4

import pytest

from puripuly_heart.core.clock import FakeClock
from puripuly_heart.core.speech_boundary import SpeechBoundaryReason
from puripuly_heart.core.stt.backend import STTBackendTranscriptEvent
from puripuly_heart.core.stt.controller import ManagedSTTProvider
from puripuly_heart.core.vad.gating import SpeechEnd, SpeechStart
from puripuly_heart.domain.events import STTFinalEvent, UIEventType
from puripuly_heart.domain.models import FinalLanguageRun, Transcript, Translation
from tests.helpers.fakes import RecordingOscQueue, samples
from tests.helpers.translation_owners import compose_translation_test_harness


@dataclass(slots=True)
class FakePeerSession:
    audio: list[bytes] = field(default_factory=list)
    _queue: asyncio.Queue[object | None] = field(default_factory=asyncio.Queue)
    _seen_speech: bool = False

    async def send_audio(self, pcm16le: bytes) -> None:
        self.audio.append(pcm16le)
        if any(byte != 0 for byte in pcm16le):
            self._seen_speech = True

    async def on_speech_end(
        self,
        *,
        trailing_silence_ms: int | None = None,
        reason: SpeechBoundaryReason | None = None,
    ) -> None:
        _ = (trailing_silence_ms, reason)
        if self._seen_speech:
            self._seen_speech = False
            await self._queue.put(STTBackendTranscriptEvent(text="peer final", is_final=True))

    async def stop(self) -> None:
        await self._queue.put(None)

    async def close(self) -> None:
        await self._queue.put(None)

    async def events(self):
        while True:
            item = await self._queue.get()
            if item is None:
                return
            yield item


@dataclass(slots=True)
class FakePeerBackend:
    sessions: list[FakePeerSession] = field(default_factory=list)

    async def open_session(self) -> FakePeerSession:
        session = FakePeerSession()
        self.sessions.append(session)
        return session


@dataclass(slots=True)
class LabelledPeerSession:
    label: str
    audio: list[bytes] = field(default_factory=list)
    _queue: asyncio.Queue[object | None] = field(default_factory=asyncio.Queue)
    _seen_speech: bool = False

    async def send_audio(self, pcm16le: bytes) -> None:
        self.audio.append(pcm16le)
        if any(byte != 0 for byte in pcm16le):
            self._seen_speech = True

    async def on_speech_end(
        self,
        *,
        trailing_silence_ms: int | None = None,
        reason: SpeechBoundaryReason | None = None,
    ) -> None:
        _ = (trailing_silence_ms, reason)
        if self._seen_speech:
            self._seen_speech = False
            await self._queue.put(
                STTBackendTranscriptEvent(text=f"{self.label} final", is_final=True)
            )

    async def stop(self) -> None:
        await self._queue.put(None)

    async def close(self) -> None:
        await self._queue.put(None)

    async def events(self):
        while True:
            item = await self._queue.get()
            if item is None:
                return
            yield item


@dataclass(slots=True)
class LabelledPeerBackend:
    label: str
    sessions: list[LabelledPeerSession] = field(default_factory=list)

    async def open_session(self) -> LabelledPeerSession:
        session = LabelledPeerSession(label=self.label)
        self.sessions.append(session)
        return session


@dataclass(slots=True)
class FakeLLM:
    calls: list[str] = field(default_factory=list)

    async def translate(
        self,
        *,
        utterance_id,
        text: str,
        system_prompt: str,
        source_language: str,
        target_language: str,
        context: str = "",
    ) -> Translation:
        _ = (utterance_id, system_prompt, source_language, target_language, context)
        self.calls.append(text)
        return Translation(utterance_id=utterance_id, text="translated")

    async def close(self) -> None:
        return None


async def _next_transcript_final_event(
    queue: asyncio.Queue[object],
    *,
    timeout_s: float = 0.5,
):
    while True:
        event = await asyncio.wait_for(queue.get(), timeout=timeout_s)
        if getattr(event, "type", None) == UIEventType.TRANSCRIPT_FINAL:
            return event


@pytest.mark.asyncio
async def test_peer_desktop_transcripts_are_routed_to_peer_runtime_and_never_sent_to_chatbox() -> (
    None
):
    osc = RecordingOscQueue()
    harness = compose_translation_test_harness(
        stt=None, llm=None, osc=osc, clock=FakeClock(_now=10.0)
    )

    utterance_id = await harness.handle_peer_transcript_final_for_test(
        text="peer line",
    )

    bundle = harness.bundle_for(utterance_id, channel="peer")
    event = await harness.ui_events.get()

    assert bundle.final is not None
    assert bundle.final.channel == "peer"
    assert bundle.final.text == "peer line"
    assert osc.messages == []
    assert event.type == UIEventType.TRANSCRIPT_FINAL
    assert event.channel == "peer"


@pytest.mark.asyncio
async def test_peer_final_runs_owner_creates_ordered_children_for_language_runs() -> None:
    harness = compose_translation_test_harness(
        stt=None, llm=None, osc=RecordingOscQueue(), clock=FakeClock(_now=10.0)
    )
    parent_utterance_id = uuid4()
    runs = (
        FinalLanguageRun(text="日本語", language="ja"),
        FinalLanguageRun(text="中文", language="zh"),
    )

    child_ids = await harness.translation_turns.submit_parent(
        Transcript(
            utterance_id=parent_utterance_id,
            text="日本語中文",
            is_final=True,
            channel="peer",
            final_language_runs=runs,
        ),
        source="Peer",
    )

    assert len(child_ids) == 2
    assert parent_utterance_id not in child_ids
    assert [
        harness.peer_runtime.utterances[child_id].final.final_language_runs
        for child_id in child_ids
    ] == [
        (runs[0],),
        (runs[1],),
    ]


@pytest.mark.asyncio
async def test_peer_final_event_preserves_language_runs_at_the_current_consumer_boundary() -> None:
    harness = compose_translation_test_harness(
        stt=None, llm=None, osc=RecordingOscQueue(), clock=FakeClock(_now=10.0)
    )
    parent_utterance_id = uuid4()
    runs = (FinalLanguageRun(text="中文", language="zh"),)
    transcript = Transcript(
        utterance_id=parent_utterance_id,
        text="中文",
        is_final=True,
        channel="peer",
        final_language_runs=runs,
    )

    await harness.dispatch_stt_event(STTFinalEvent(parent_utterance_id, transcript))

    event = await harness.ui_events.get()
    assert event.type == UIEventType.TRANSCRIPT_FINAL
    assert isinstance(event.payload, Transcript)
    assert event.payload.final_language_runs == runs


@pytest.mark.asyncio
async def test_integrated_context_always_includes_peer_entries() -> None:
    clock = FakeClock(_now=112.0)
    harness = compose_translation_test_harness(
        stt=None,
        llm=None,
        osc=RecordingOscQueue(),
        clock=clock,
        integrated_context_enabled=True,
        peer_translation_enabled=True,
    )
    harness.replace_configuration(source_language="en")
    harness.replace_configuration(target_language="ko")
    harness.self_runtime.remember_context(
        "self line",
        timestamp=100.0,
        source_language="en",
        target_language="ko",
    )
    harness.peer_runtime.remember_context(
        "peer line",
        timestamp=105.0,
        source_language="en",
        target_language="ko",
    )

    context, mode = harness.translation_requests.context_resolver.resolve_for_request(
        runtime=harness.self_runtime,
        other_runtime=harness.peer_runtime,
        requested_mode="integrated",
        peer_translation_enabled=True,
        source_language="en",
        target_language="ko",
    )

    assert mode == "integrated"
    assert "self line" in context
    assert "peer line" in context


def test_integrated_context_includes_opposite_direction_peer_entries() -> None:
    clock = FakeClock(_now=112.0)
    harness = compose_translation_test_harness(
        stt=None,
        llm=None,
        osc=RecordingOscQueue(),
        clock=clock,
        integrated_context_enabled=True,
        peer_translation_enabled=True,
    )
    harness.replace_configuration(source_language="ko")
    harness.replace_configuration(target_language="en")
    harness.replace_configuration(peer_source_language="en")
    harness.replace_configuration(peer_target_language="ko")
    harness.remember_context("self previous", timestamp=100.0, runtime=harness.self_runtime)
    harness.remember_context("peer previous", timestamp=105.0, runtime=harness.peer_runtime)

    _, self_context, _, self_mode = harness.prepare_translation_request_with_mode(
        "self current",
        runtime=harness.self_runtime,
    )
    _, peer_context, _, peer_mode = harness.prepare_translation_request_with_mode(
        "peer current",
        runtime=harness.peer_runtime,
    )

    assert self_mode == "integrated"
    assert peer_mode == "integrated"
    assert self_context == ('- [self] "self previous"\n- [peer] "peer previous"')
    assert peer_context == ('- [self] "self previous"\n- [peer] "peer previous"')


@pytest.mark.asyncio
async def test_peer_translation_respects_master_translation_toggle() -> None:
    llm = FakeLLM()
    harness = compose_translation_test_harness(
        stt=None,
        llm=llm,
        osc=RecordingOscQueue(),
        clock=FakeClock(_now=10.0),
        translation_enabled=False,
        peer_translation_enabled=True,
    )

    utterance_id = await harness.handle_peer_transcript_final_for_test(text="peer line")
    bundle = harness.bundle_for(utterance_id, channel="peer")
    event = await harness.ui_events.get()

    assert event.type == UIEventType.TRANSCRIPT_FINAL
    assert bundle.translation is None
    assert llm.calls == []


@pytest.mark.asyncio
async def test_peer_transcripts_stay_peer_routed_across_runtime_swap_without_duplicates() -> None:
    old_peer = ManagedSTTProvider(
        backend=LabelledPeerBackend("old"),
        sample_rate_hz=16000,
        channel="peer",
        reset_deadline_s=90.0,
        drain_timeout_s=0.05,
        finalize_grace_s=0.0,
    )
    new_peer = ManagedSTTProvider(
        backend=LabelledPeerBackend("new"),
        sample_rate_hz=16000,
        channel="peer",
        reset_deadline_s=90.0,
        drain_timeout_s=0.05,
        finalize_grace_s=0.0,
    )
    harness = compose_translation_test_harness(
        stt=None,
        peer_stt=old_peer,
        llm=None,
        osc=RecordingOscQueue(),
        clock=FakeClock(_now=10.0),
    )
    await harness.start(auto_flush_osc=False)

    first_id = __import__("uuid").uuid4()
    await harness.peer_owner.handle_peer_vad_event(
        SpeechStart(first_id, pre_roll=samples(0.0), chunk=samples(1.0))
    )
    await harness.peer_owner.handle_peer_vad_event(SpeechEnd(first_id))
    first_final = await _next_transcript_final_event(harness.ui_events)

    await harness.local_asr_runtime.replace_prebuilt_provider("peer", new_peer, start=True)

    second_id = __import__("uuid").uuid4()
    await harness.peer_owner.handle_peer_vad_event(
        SpeechStart(second_id, pre_roll=samples(0.0), chunk=samples(1.0))
    )
    await harness.peer_owner.handle_peer_vad_event(SpeechEnd(second_id))
    second_final = await _next_transcript_final_event(harness.ui_events)

    await asyncio.sleep(0.05)
    remaining_events: list[object] = []
    while not harness.ui_events.empty():
        remaining_events.append(await harness.ui_events.get())

    finals = [first_final, second_final] + [
        event
        for event in remaining_events
        if getattr(event, "type", None) == UIEventType.TRANSCRIPT_FINAL
    ]

    assert len(finals) == 2
    assert [event.channel for event in finals] == ["peer", "peer"]
    assert [event.payload.text for event in finals] == ["old final", "new final"]
    await harness.stop()
