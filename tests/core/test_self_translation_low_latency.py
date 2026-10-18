"""Unit tests for low-latency mode awaiting_vad_end bug fix."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field, replace
from uuid import UUID, uuid4

import numpy as np
import pytest

from puripuly_heart.core.audio.format import AudioCaptureSpan
from puripuly_heart.core.osc.chatbox_paginator import ChatboxPaginator
from puripuly_heart.core.overlay.state import ActiveSelfOverlayMetadata
from puripuly_heart.core.translation_backend import LlmTranslationBackend
from puripuly_heart.core.vad.gating import SpeechChunk, SpeechEnd, SpeechStart
from puripuly_heart.domain.events import (
    STTFinalEvent,
    STTSessionState,
    STTSessionStateEvent,
    UIEventType,
)
from puripuly_heart.domain.models import Transcript, Translation
from tests.core.test_translation_owner_branch_coverage import (
    _make_runtime_logging_capture,
    _runtime_log_messages,
)
from tests.helpers.fakes import FakeSender
from tests.helpers.translation_owners import (
    compose_translation_test_harness,
    owned_peer_speech_end,
)

# ── Mock classes ──────────────────────────────────────────────────────────────


class FakeClock:
    """Fake clock for testing time-based logic."""

    def __init__(self, initial_time: float = 0.0):
        self._time = initial_time

    def now(self) -> float:
        return self._time

    def advance(self, seconds: float) -> None:
        self._time += seconds


@dataclass
class FakeLLMProvider:
    """Fake LLM provider that records calls."""

    calls: list[dict] = field(default_factory=list)
    response_text: str = "translated"
    delay_s: float = 0.01

    async def translate(
        self,
        *,
        utterance_id,
        text: str,
        system_prompt: str,
        source_language: str,
        target_language: str,
        context: str = "",
        scene_participant_count: int | None = None,
    ):
        self.calls.append(
            {
                "utterance_id": utterance_id,
                "text": text,
                "context": context,
                "target_language": target_language,
            }
        )
        await asyncio.sleep(self.delay_s)
        return Translation(utterance_id=utterance_id, text=self.response_text)

    async def close(self) -> None:
        pass


@dataclass
class ProtocolOnlyProviderRuntime:
    provider: object | None
    current_generation: int

    def current_provider_generation(self) -> tuple[object | None, int]:
        return self.provider, self.current_generation

    def is_current_provider_generation(
        self,
        *,
        provider: object,
        generation: int,
    ) -> bool:
        return provider is self.provider and generation == self.current_generation


@dataclass
class ClosingLLMProvider(FakeLLMProvider):
    close_calls: int = 0

    async def close(self) -> None:
        self.close_calls += 1


@dataclass
class BlockingLLMProvider(FakeLLMProvider):
    started: asyncio.Event = field(default_factory=asyncio.Event)
    release: asyncio.Event = field(default_factory=asyncio.Event)
    close_calls: int = 0

    async def translate(
        self,
        *,
        utterance_id,
        text: str,
        system_prompt: str,
        source_language: str,
        target_language: str,
        context: str = "",
        scene_participant_count: int | None = None,
    ):
        self.calls.append(
            {
                "utterance_id": utterance_id,
                "text": text,
                "context": context,
                "target_language": target_language,
            }
        )
        self.started.set()
        await self.release.wait()
        return Translation(utterance_id=utterance_id, text=self.response_text)

    async def close(self) -> None:
        self.close_calls += 1


@dataclass
class PerTargetBlockingLLMProvider(FakeLLMProvider):
    started_targets: set[str] = field(default_factory=set)
    all_started: asyncio.Event = field(default_factory=asyncio.Event)
    releases: dict[str, asyncio.Event] = field(default_factory=dict)

    async def translate(
        self,
        *,
        utterance_id,
        text: str,
        system_prompt: str,
        source_language: str,
        target_language: str,
        context: str = "",
        scene_participant_count: int | None = None,
    ):
        self.calls.append(
            {
                "utterance_id": utterance_id,
                "text": text,
                "context": context,
                "target_language": target_language,
            }
        )
        self.started_targets.add(target_language)
        if len(self.started_targets) == 2:
            self.all_started.set()
        release = self.releases.setdefault(target_language, asyncio.Event())
        await release.wait()
        return Translation(utterance_id=utterance_id, text=f"translated {target_language}")


@dataclass
class FailingSecondaryLLMProvider(FakeLLMProvider):
    primary_started: asyncio.Event = field(default_factory=asyncio.Event)
    primary_release: asyncio.Event = field(default_factory=asyncio.Event)
    secondary_finished: asyncio.Event = field(default_factory=asyncio.Event)
    secondary_calls: int = 0

    async def translate(
        self,
        *,
        utterance_id,
        text: str,
        system_prompt: str,
        source_language: str,
        target_language: str,
        context: str = "",
        scene_participant_count: int | None = None,
    ):
        self.calls.append(
            {
                "utterance_id": utterance_id,
                "text": text,
                "context": context,
                "target_language": target_language,
            }
        )
        if target_language == "ja":
            self.secondary_calls += 1
            if self.secondary_calls == 1:
                self.secondary_finished.set()
                raise RuntimeError("secondary failed")
            return Translation(utterance_id=utterance_id, text="translated ja")
        self.primary_started.set()
        await self.primary_release.wait()
        return Translation(utterance_id=utterance_id, text="translated zh-CN")


@dataclass
class FailingThenSuccessfulLLMProvider(FakeLLMProvider):
    async def translate(
        self,
        *,
        utterance_id,
        text: str,
        system_prompt: str,
        source_language: str,
        target_language: str,
        context: str = "",
        scene_participant_count: int | None = None,
    ):
        self.calls.append(
            {
                "utterance_id": utterance_id,
                "text": text,
                "context": context,
            }
        )
        if len(self.calls) == 1:
            raise RuntimeError("spec failed")
        return Translation(utterance_id=utterance_id, text=self.response_text)


@dataclass
class QueueingSTTProvider:
    channel: str = "self"
    closed: bool = False
    handled: list[object] = field(default_factory=list)
    queue: asyncio.Queue[object | None] = field(default_factory=asyncio.Queue)

    async def handle_vad_event(self, event: object) -> None:
        self.handled.append(event)

    async def close(self) -> None:
        self.closed = True
        await self.queue.put(None)

    async def emit(self, event: object) -> None:
        await self.queue.put(event)

    async def events(self):
        while True:
            item = await self.queue.get()
            if item is None:
                return
            yield item


class PersistentQueueingSTTProvider(QueueingSTTProvider):
    async def close(self) -> None:
        self.closed = True


@dataclass
class FlakyCloseSTTProvider(QueueingSTTProvider):
    close_failures: int = 1
    close_label: str = "stt"
    close_calls: int = 0

    async def close(self) -> None:
        self.close_calls += 1
        if self.close_failures > 0:
            self.close_failures -= 1
            raise RuntimeError(f"{self.close_label} close failed")
        await super().close()


@dataclass
class ClockedTranslateLLMProvider:
    clock: FakeClock
    responses: list[tuple[float, str]]
    calls: list[dict] = field(default_factory=list)

    async def translate(
        self,
        *,
        utterance_id,
        text: str,
        system_prompt: str,
        source_language: str,
        target_language: str,
        context: str = "",
        scene_participant_count: int | None = None,
    ):
        self.calls.append(
            {
                "utterance_id": utterance_id,
                "text": text,
                "system_prompt": system_prompt,
                "source_language": source_language,
                "target_language": target_language,
                "context": context,
            }
        )
        if not self.responses:
            raise AssertionError("no translate response configured")
        delay_s, response_text = self.responses.pop(0)
        self.clock.advance(delay_s)
        return Translation(utterance_id=utterance_id, text=response_text)

    async def close(self) -> None:
        pass


@dataclass
class FakeOscQueue:
    """Fake OSC queue that records enqueued messages."""

    messages: list = field(default_factory=list)

    def enqueue(self, msg) -> None:
        self.messages.append(msg)

    def send_typing(self, on: bool) -> None:
        pass

    def set_typing_reason(self, reason: str, active: bool) -> None:
        pass

    def send_immediate(self, text: str) -> tuple[bool, None]:

        return True, None

    def process_due(self) -> None:
        pass


@dataclass
class RecordingOverlaySink:
    events: list[object] = field(default_factory=list)
    active_self_metadata: ActiveSelfOverlayMetadata | None = None

    async def emit(self, event: object) -> None:
        self.events.append(event)
        event_type = getattr(event, "type", None)
        if event_type == "self_active_update":
            utterance_id = getattr(event, "utterance_id", None)
            if not isinstance(utterance_id, UUID):
                return
            self.active_self_metadata = ActiveSelfOverlayMetadata(
                text=getattr(event, "text", ""),
                secondary_text=getattr(event, "secondary_text", ""),
                utterance_id=utterance_id,
                occupant_key=getattr(event, "occupant_key", ""),
                update_id=getattr(event, "update_id", None),
                origin_wall_clock_ms=getattr(event, "origin_wall_clock_ms", None),
                session_scope=getattr(event, "session_scope", None),
                source_text_hash=getattr(event, "source_text_hash", None),
                source_text_len=getattr(event, "source_text_len", None),
                logical_turn_key=getattr(event, "logical_turn_key", None),
                primary_language=(str(getattr(event, "source_language", "") or "").strip() or None),
                secondary_language=(
                    str(getattr(event, "target_language", "") or "").strip() or None
                    if getattr(event, "secondary_text", "").strip()
                    else None
                ),
            )
        elif event_type == "self_active_clear":
            self.active_self_metadata = None
        elif event_type == "self_transcript_final" and self.active_self_metadata is not None:
            if self.active_self_metadata.utterance_id == getattr(event, "utterance_id", None):
                self.active_self_metadata = None

    def active_self_overlay_metadata(self) -> ActiveSelfOverlayMetadata | None:
        return self.active_self_metadata


def samples(value: float, n: int = 512) -> np.ndarray:
    return np.full((n,), value, dtype=np.float32)


async def _eventually(predicate):
    async with asyncio.timeout(1.0):
        while not predicate():
            await asyncio.sleep(0)


async def _source_revision(harness, source_id, text):
    await harness.self_owner.handle_stt_event(
        STTFinalEvent(
            source_id,
            Transcript(utterance_id=source_id, text=text, is_final=True),
        )
    )


def _drain_ui(harness):
    events = []
    while not harness.ui_events.empty():
        events.append(harness.ui_events.get_nowait())
    return events


# ── Tests ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_public_low_latency_final_uses_declared_provider_generation_port() -> None:
    llm = FakeLLMProvider(response_text="translated", delay_s=0.0)
    sink = RecordingOverlaySink()
    harness = compose_translation_test_harness(
        stt=None,
        llm=llm,
        osc=FakeOscQueue(),
        overlay_sink=sink,
        low_latency_mode=True,
    )
    harness.translation_requests.provider_runtime = ProtocolOnlyProviderRuntime(
        provider=LlmTranslationBackend(llm),
        current_generation=41,
    )
    try:
        await _source_revision(harness, uuid4(), "port contract")
        await _eventually(
            lambda: any(
                getattr(event, "secondary_text", "") == "translated" for event in sink.events
            )
        )
        assert [call["text"] for call in llm.calls] == ["port contract"]
    finally:
        await harness.stop()


@pytest.mark.asyncio
async def test_translation_exposes_named_provider_runtime_handles_and_shutdown_policies() -> None:
    stt = QueueingSTTProvider(channel="self")
    peer_stt = QueueingSTTProvider(channel="peer")
    llm = ClosingLLMProvider(response_text="translated", delay_s=0.0)
    harness = compose_translation_test_harness(
        stt=stt, peer_stt=peer_stt, llm=llm, osc=FakeOscQueue()
    )

    llm_runtime = harness.llm_runtime

    assert llm_runtime.owner_name == "ProviderRuntimeHandle:llm"
    assert harness.local_asr_runtime is not None
    assert harness.local_asr_runtime.snapshot.channel_for("self").has_resources
    assert harness.local_asr_runtime.snapshot.channel_for("peer").has_resources
    assert "await provider.close" in llm_runtime.shutdown_policy

    await harness.start(auto_flush_osc=False)
    assert harness.self_runtime.stt_task is None

    await harness.stop()
    await harness.stop()

    assert stt.closed is True
    assert peer_stt.closed is True
    assert llm.close_calls == 1


@pytest.mark.asyncio
async def test_stop_attempts_all_provider_closes_and_retries_failed_handles() -> None:
    self_stt = FlakyCloseSTTProvider(
        channel="self",
        close_failures=1,
        close_label="self provider",
    )
    peer_stt = FlakyCloseSTTProvider(
        channel="peer",
        close_failures=1,
        close_label="peer provider",
    )
    llm = ClosingLLMProvider(response_text="translated", delay_s=0.0)
    harness = compose_translation_test_harness(
        stt=self_stt, peer_stt=peer_stt, llm=llm, osc=FakeOscQueue()
    )

    await harness.start(auto_flush_osc=False)

    with pytest.raises(ExceptionGroup) as excinfo:
        await harness.stop()

    failure_messages = {str(exc) for exc in excinfo.value.exceptions}
    assert failure_messages == {
        "self provider close failed",
        "peer provider close failed",
    }
    assert self_stt.close_calls == 1
    assert peer_stt.close_calls == 1
    assert llm.close_calls == 1
    assert harness.local_asr_runtime.snapshot.channel_for("self").has_resources
    assert harness.local_asr_runtime.snapshot.channel_for("peer").has_resources
    assert harness.llm_runtime.provider is None

    await harness.stop()

    assert self_stt.close_calls == 2
    assert peer_stt.close_calls == 2
    assert self_stt.closed is True
    assert peer_stt.closed is True
    assert not harness.local_asr_runtime.snapshot.channel_for("self").has_resources
    assert not harness.local_asr_runtime.snapshot.channel_for("peer").has_resources
    assert harness.llm_runtime.provider is None


@pytest.mark.asyncio
async def test_replaced_peer_provider_late_final_cannot_mutate_or_enqueue_chatbox() -> None:
    old_peer = QueueingSTTProvider(channel="peer")
    new_peer = QueueingSTTProvider(channel="peer")
    osc = FakeOscQueue()
    harness = compose_translation_test_harness(stt=None, peer_stt=old_peer, llm=None, osc=osc)

    assert harness.llm_runtime.owner_name == "ProviderRuntimeHandle:llm"

    await harness.start(auto_flush_osc=False)
    await harness.local_asr_runtime.replace_prebuilt_provider("peer", new_peer, start=True)
    stale_utterance_id = uuid4()
    await old_peer.emit(
        STTFinalEvent(
            stale_utterance_id,
            Transcript(
                utterance_id=stale_utterance_id,
                text="stale peer final",
                is_final=True,
                created_at=0.0,
                channel="peer",
            ),
        )
    )
    await asyncio.sleep(0)

    assert harness.peer_runtime.utterances == {}
    assert osc.messages == []

    await harness.stop()


@pytest.mark.asyncio
async def test_replaced_llm_provider_late_final_cleans_peer_runtime_without_output() -> None:
    old_llm = BlockingLLMProvider(response_text="stale translation", delay_s=0.0)
    new_llm = FakeLLMProvider(response_text="current translation", delay_s=0.0)
    osc = FakeOscQueue()
    overlay_sink = RecordingOverlaySink()
    harness = compose_translation_test_harness(
        stt=None,
        llm=old_llm,
        osc=osc,
        overlay_sink=overlay_sink,
        peer_translation_enabled=True,
    )
    utterance_id = uuid4()
    parent_utterance_id = uuid4()
    harness.peer_owner._peer_parent_speech_end_times[parent_utterance_id] = 12.0
    harness.peer_owner._register_peer_logical_turn(
        parent_utterance_id=parent_utterance_id,
        peer_turn_id=utterance_id,
    )

    translate_task = asyncio.create_task(
        harness.process_translation(utterance_id, "peer hello", runtime=harness.peer_runtime)
    )
    await asyncio.wait_for(old_llm.started.wait(), timeout=1.0)

    await harness.replace_llm_provider(new_llm)
    assert old_llm.close_calls == 1

    old_llm.release.set()
    await asyncio.wait_for(translate_task, timeout=1.0)

    assert utterance_id not in harness.peer_runtime.utterances
    assert harness.ui_events.empty()
    assert osc.messages == []
    assert overlay_sink.events == []
    assert not any(
        getattr(event, "type", None) == "translation_final" for event in overlay_sink.events
    )
    assert not any(
        getattr(event, "text", "") == "stale translation" for event in overlay_sink.events
    )
    assert utterance_id not in harness.peer_runtime.utterance_start_times
    assert utterance_id not in harness.peer_runtime.speech_ended_ids
    assert utterance_id not in harness.peer_owner._peer_turn_parent_ids
    assert parent_utterance_id not in harness.peer_owner._peer_parent_turn_ids
    assert parent_utterance_id not in harness.peer_owner._peer_parent_speech_end_times
    assert (
        "peer",
        utterance_id,
    ) not in harness.translation_diagnostics.snapshot().timeline_keys

    await harness.stop()


@pytest.mark.asyncio
async def test_replaced_llm_provider_late_spec_completion_cannot_update_low_latency_state() -> None:
    old = BlockingLLMProvider(response_text="stale speculative translation")
    new = FakeLLMProvider(response_text="current translation", delay_s=0.0)
    sink = RecordingOverlaySink()
    osc = FakeOscQueue()
    harness = compose_translation_test_harness(
        stt=None,
        llm=old,
        osc=osc,
        overlay_sink=sink,
        low_latency_mode=True,
        low_latency_finalize_wait_ms=0,
    )
    source_id = uuid4()
    try:
        await _source_revision(harness, source_id, "hello live")
        await asyncio.wait_for(old.started.wait(), timeout=1.0)
        await harness.replace_llm_provider(new)
        old.release.set()
        await harness.self_owner.handle_vad_event(SpeechEnd(source_id))
        await asyncio.wait_for(harness.translation_turns.wait_for_idle(), timeout=1.0)
        assert old.close_calls == 1
        assert [call["text"] for call in new.calls] == ["hello live"]
        assert osc.messages[-1].text == "hello live (current translation)"
        assert not any(
            "stale speculative" in getattr(event, "secondary_text", "") for event in sink.events
        )
        assert not any(event.type == UIEventType.ERROR for event in _drain_ui(harness))
    finally:
        old.release.set()
        await harness.stop()


@pytest.mark.asyncio
async def test_replaced_llm_provider_late_spec_completion_falls_back_without_hanging_commit() -> (
    None
):
    old = BlockingLLMProvider(response_text="stale speculative translation")
    new = FakeLLMProvider(response_text="current fallback translation", delay_s=0.0)
    osc = FakeOscQueue()
    sink = RecordingOverlaySink()
    harness = compose_translation_test_harness(
        stt=None,
        llm=old,
        osc=osc,
        overlay_sink=sink,
        low_latency_mode=True,
        low_latency_finalize_wait_ms=0,
    )
    source_id = uuid4()
    try:
        await _source_revision(harness, source_id, "hello live")
        await asyncio.wait_for(old.started.wait(), timeout=1.0)
        await harness.self_owner.handle_vad_event(SpeechEnd(source_id))
        assert harness.self_owner.merge_buffer is None
        await harness.replace_llm_provider(new)
        old.release.set()
        await asyncio.wait_for(harness.translation_turns.wait_for_idle(), timeout=1.0)
        assert [call["text"] for call in new.calls] == ["hello live"]
        assert osc.messages[-1].text == "hello live (current fallback translation)"
        snapshots = [
            event.payload
            for event in _drain_ui(harness)
            if event.type == UIEventType.SELF_TURN_UPDATED
        ]
        assert snapshots[-1].status == "complete"
        assert snapshots[-1].targets[0].translation.text == "current fallback translation"
        assert not any(
            "stale speculative" in getattr(event, "secondary_text", "") for event in sink.events
        )
    finally:
        old.release.set()
        await harness.stop()


@pytest.mark.asyncio
async def test_ready_spec_result_is_invalidated_by_provider_replacement_during_grace() -> None:
    old = FakeLLMProvider(response_text="old translation", delay_s=0.0)
    new = FakeLLMProvider(response_text="new translation", delay_s=0.0)
    sink = RecordingOverlaySink()
    osc = FakeOscQueue()
    harness = compose_translation_test_harness(
        stt=None,
        llm=old,
        osc=osc,
        overlay_sink=sink,
        low_latency_mode=True,
        low_latency_finalize_wait_ms=50,
    )
    source_id = uuid4()
    try:
        await harness.self_owner.handle_vad_event(SpeechEnd(source_id))
        await _source_revision(harness, source_id, "hello")
        await _eventually(
            lambda: any(
                getattr(event, "secondary_text", "") == "old translation" for event in sink.events
            )
        )
        await harness.replace_llm_provider(new)
        await _eventually(lambda: harness.self_owner.merge_buffer is None)
        await asyncio.wait_for(harness.translation_turns.wait_for_idle(), timeout=1.0)
        assert len(old.calls) == 1
        assert len(new.calls) == 1
        assert osc.messages[-1].text == "hello (new translation)"
        snapshots = [
            event.payload
            for event in _drain_ui(harness)
            if event.type == UIEventType.SELF_TURN_UPDATED
        ]
        assert snapshots[-1].status == "complete"
        assert snapshots[-1].targets[0].translation.text == "new translation"
    finally:
        await harness.stop()


@pytest.mark.asyncio
async def test_self_stt_toggle_off_keeps_event_ingress_for_later_reenable_events() -> None:
    stt = PersistentQueueingSTTProvider(channel="self")
    harness = compose_translation_test_harness(stt=stt, peer_stt=None, llm=None, osc=FakeOscQueue())

    await harness.start(auto_flush_osc=False)
    try:
        await harness.drain_self_stt_for_toggle_off()

        assert stt.closed is True

        await stt.emit(
            STTSessionStateEvent(
                STTSessionState.STREAMING,
                channel="self",
            )
        )
        ui_event = await asyncio.wait_for(harness.ui_events.get(), timeout=1.0)

        assert ui_event.type == UIEventType.SESSION_STATE_CHANGED
        assert ui_event.payload == STTSessionState.STREAMING
        assert ui_event.channel == "self"
    finally:
        await harness.stop()


class TestSpeechEndedTracking:
    """Test _speech_ended_ids tracking."""

    @pytest.mark.asyncio
    async def test_low_latency_state_stays_on_self_runtime_only(self):
        clock = FakeClock(initial_time=10.0)
        harness = compose_translation_test_harness(
            stt=None,
            llm=None,
            osc=FakeOscQueue(),
            clock=clock,
            low_latency_mode=True,
            low_latency_finalize_wait_ms=0,
        )

        transcript = Transcript(
            utterance_id=uuid4(),
            text="self only",
            is_final=True,
            created_at=clock.now(),
        )
        await harness.dispatch_stt_event(
            STTFinalEvent(utterance_id=transcript.utterance_id, transcript=transcript)
        )

        assert harness.self_runtime.merge_buffer is harness.self_owner.merge_buffer
        assert harness.peer_runtime.merge_buffer is None
        assert harness.self_runtime.translation_history == []
        assert harness.peer_runtime.translation_history == []

        await harness.stop()


class TestRuntimeLatencyLogging:

    @pytest.mark.asyncio
    async def test_low_latency_self_and_peer_success_paths_both_use_translate(
        self,
    ):
        llm = FakeLLMProvider(
            response_text="translated",
            delay_s=0.0,
        )
        overlay_sink = RecordingOverlaySink()
        harness = compose_translation_test_harness(
            stt=None,
            llm=llm,
            osc=FakeOscQueue(),
            overlay_sink=overlay_sink,
            low_latency_mode=True,
            low_latency_finalize_wait_ms=0,
            peer_translation_enabled=True,
        )

        try:
            transcript = Transcript(
                utterance_id=uuid4(),
                text="hello",
                is_final=True,
                created_at=0.0,
            )
            await harness.dispatch_stt_event(
                STTFinalEvent(utterance_id=transcript.utterance_id, transcript=transcript)
            )

            await _eventually(
                lambda: any(
                    getattr(event, "secondary_text", "") == "translated"
                    for event in overlay_sink.events
                )
            )

            assert len(llm.calls) == 1
            assert llm.calls[0]["text"] == "hello"

            await harness.process_translation(uuid4(), "peer hello", runtime=harness.peer_runtime)

            assert [call["text"] for call in llm.calls] == ["hello", "peer hello"]
        finally:
            await harness.stop()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("trailing_silence_ms,reason", [(600, "silence"), (0, "max_duration")])
    async def test_last_speech_latency_uses_capture_tail_not_handler_time_or_settings(
        self, trailing_silence_ms, reason, monkeypatch
    ):
        runtime_logging, log_stream = _make_runtime_logging_capture()
        clock = FakeClock(initial_time=12.0)
        sender = FakeSender()
        osc = ChatboxPaginator(sender=sender, clock=clock)
        harness = compose_translation_test_harness(
            stt=None,
            llm=None,
            osc=osc,
            clock=clock,
            low_latency_mode=True,
            low_latency_finalize_wait_ms=0,
            hangover_s=9.9,
            runtime_logging=runtime_logging,
        )
        osc.stage_recorder = harness.translation_diagnostics.record_chatbox_stage

        async def accept_owned_audio(*args):
            return None

        monkeypatch.setattr(
            type(harness.local_asr_runtime), "handle_owned_vad_event", accept_owned_audio
        )
        utterance_id = uuid4()
        owned = owned_peer_speech_end(
            utterance_id,
            speech_end_at=11.0,
            trailing_silence_ms=trailing_silence_ms,
            reason=reason,
        )
        capture = AudioCaptureSpan(0, 0, 16000, 0, 16000, 9.0, 10.0)
        owned = replace(
            owned,
            segment=replace(owned.segment, content_ranges=(capture,), content_sample_count=16000),
        )
        try:
            await harness.self_owner.handle_vad_event(owned)
            harness.replace_configuration(hangover_s=0.1)
            clock.advance(0.25)
            await harness.dispatch_stt_event(
                STTFinalEvent(
                    utterance_id=utterance_id,
                    transcript=Transcript(
                        utterance_id=utterance_id,
                        text="captured speech",
                        is_final=True,
                        created_at=clock.now(),
                    ),
                )
            )
            await harness.translation_turns.wait_for_idle()

            assert sender.sent == ["captured speech"]
            elapsed_ms = 2250 + trailing_silence_ms
            summaries = [
                message
                for message in _runtime_log_messages(log_stream)
                if "[Basic][Latency]" in message
            ]
            assert len(summaries) == 1
            assert f"last_speech_to_chatbox_send_ms={elapsed_ms}" in summaries[0]
        finally:
            await harness.stop()
            runtime_logging.close()

    @pytest.mark.asyncio
    async def test_speech_end_before_stt_final_uses_post_end_phase(self):
        """SpeechEnd가 먼저 오면 phase=post_end로 처리되어야 함."""
        clock = FakeClock(initial_time=10.0)
        harness = compose_translation_test_harness(
            stt=None,
            llm=FakeLLMProvider(),
            osc=FakeOscQueue(),
            clock=clock,
            low_latency_mode=True,
        )

        uid = uuid4()

        # 1. SpeechStart
        await harness.self_owner.handle_vad_event(
            SpeechStart(uid, pre_roll=samples(0.0), chunk=samples(1.0))
        )

        # 2. SpeechChunk (3개)
        for _ in range(3):
            await harness.self_owner.handle_vad_event(SpeechChunk(uid, chunk=samples(0.5)))

        # 3. SpeechEnd 먼저 도착
        await harness.self_owner.handle_vad_event(SpeechEnd(uid))

        # 4. _speech_ended_ids에 추가되었는지 확인
        assert uid in harness.self_runtime.speech_ended_ids

        # 5. STT Final 이벤트 직접 호출
        transcript = Transcript(
            utterance_id=uid, text="테스트", is_final=True, created_at=clock.now()
        )
        await harness.dispatch_stt_event(
            STTFinalEvent(utterance_id=transcript.utterance_id, transcript=transcript)
        )

        # 6. awaiting_vad_end=False 확인 (post_end로 처리됨)
        buffer = harness.self_owner.merge_buffer
        assert buffer is not None
        assert buffer.awaiting_vad_end is False

        await harness.stop()

    @pytest.mark.asyncio
    async def test_stt_final_before_speech_end_waits_for_vad_end(self):
        """STT Final이 먼저 오면 awaiting_vad_end=True가 되어야 함."""
        clock = FakeClock(initial_time=10.0)
        harness = compose_translation_test_harness(
            stt=None,
            llm=FakeLLMProvider(),
            osc=FakeOscQueue(),
            clock=clock,
            low_latency_mode=True,
            low_latency_awaiting_vad_timeout_s=10.0,  # 긴 타임아웃
        )

        uid = uuid4()

        # SpeechEnd 없이 STT Final 직접 전송
        transcript = Transcript(
            utterance_id=uid, text="테스트", is_final=True, created_at=clock.now()
        )
        await harness.dispatch_stt_event(
            STTFinalEvent(utterance_id=transcript.utterance_id, transcript=transcript)
        )

        # awaiting_vad_end=True 확인
        buffer = harness.self_owner.merge_buffer
        assert buffer is not None
        assert buffer.awaiting_vad_end is True
        assert buffer.awaiting_vad_utterance_id == uid

        # SpeechEnd 전송
        await harness.self_owner.handle_vad_event(SpeechEnd(uid))

        # awaiting_vad_end=False로 클리어됨
        assert buffer.awaiting_vad_end is False

        await harness.stop()

    @pytest.mark.asyncio
    async def test_speech_ended_ids_cleaned_on_commit(self):
        """커밋 시 _speech_ended_ids가 정리되어야 함."""
        clock = FakeClock(initial_time=10.0)
        harness = compose_translation_test_harness(
            stt=None,
            llm=None,  # LLM 없이 직접 커밋
            osc=FakeOscQueue(),
            clock=clock,
            low_latency_mode=True,
            low_latency_finalize_wait_ms=0,  # 즉시 커밋
        )

        uid = uuid4()

        # SpeechEnd 도착
        await harness.self_owner.handle_vad_event(SpeechEnd(uid))
        assert uid in harness.self_runtime.speech_ended_ids

        # STT Final 전송 (LLM 없으므로 바로 커밋)
        transcript = Transcript(
            utterance_id=uid, text="테스트", is_final=True, created_at=clock.now()
        )
        await harness.dispatch_stt_event(
            STTFinalEvent(utterance_id=transcript.utterance_id, transcript=transcript)
        )

        # 커밋 후 정리됨
        assert uid not in harness.self_runtime.speech_ended_ids

        await harness.stop()


class TestAwaitingVadEndTimeout:
    """Test awaiting_vad_end timeout mechanism."""

    @pytest.mark.asyncio
    async def test_awaiting_vad_end_timeout_clears_state(self):
        """타임아웃 후 awaiting_vad_end가 클리어되어야 함."""
        clock = FakeClock(initial_time=10.0)
        harness = compose_translation_test_harness(
            stt=None,
            llm=FakeLLMProvider(),
            osc=FakeOscQueue(),
            clock=clock,
            low_latency_mode=True,
            low_latency_awaiting_vad_timeout_s=0.1,  # 100ms 타임아웃
        )

        uid = uuid4()

        # SpeechEnd 없이 STT Final 전송
        transcript = Transcript(
            utterance_id=uid, text="테스트", is_final=True, created_at=clock.now()
        )
        await harness.dispatch_stt_event(
            STTFinalEvent(utterance_id=transcript.utterance_id, transcript=transcript)
        )

        # awaiting_vad_end=True 확인
        buffer = harness.self_owner.merge_buffer
        assert buffer is not None
        assert buffer.awaiting_vad_end is True

        # 타임아웃 대기 (150ms)
        await asyncio.sleep(0.15)

        # 타임아웃으로 클리어됨
        assert buffer.awaiting_vad_end is False

        await harness.stop()

    @pytest.mark.asyncio
    async def test_timeout_cancelled_when_speech_end_arrives(self):
        """SpeechEnd가 오면 타임아웃이 취소되어야 함."""
        clock = FakeClock(initial_time=10.0)
        harness = compose_translation_test_harness(
            stt=None,
            llm=FakeLLMProvider(),
            osc=FakeOscQueue(),
            clock=clock,
            low_latency_mode=True,
            low_latency_awaiting_vad_timeout_s=0.5,  # 500ms 타임아웃
        )

        uid = uuid4()

        # STT Final 전송 → 타임아웃 시작
        transcript = Transcript(
            utterance_id=uid, text="테스트", is_final=True, created_at=clock.now()
        )
        await harness.dispatch_stt_event(
            STTFinalEvent(utterance_id=transcript.utterance_id, transcript=transcript)
        )

        buffer = harness.self_owner.merge_buffer
        assert buffer is not None
        assert buffer.awaiting_vad_timeout_task is not None

        # SpeechEnd 전송 → 타임아웃 취소
        await harness.self_owner.handle_vad_event(SpeechEnd(uid))

        # 타임아웃 태스크가 취소됨
        assert buffer.awaiting_vad_timeout_task is None

        await harness.stop()


class TestLowLatencyCommitBlocking:
    """Test commit blocking scenarios in low-latency mode."""

    @pytest.mark.asyncio
    async def test_normal_speech_commits_without_delay(self):
        """정상 발화는 지연 없이 커밋되어야 함 (regression)."""
        clock = FakeClock(initial_time=10.0)
        osc = FakeOscQueue()
        harness = compose_translation_test_harness(
            stt=None,
            llm=None,  # 번역 없이 직접 커밋
            osc=osc,
            clock=clock,
            low_latency_mode=True,
            low_latency_finalize_wait_ms=10,  # 짧은 grace period
        )

        uid = uuid4()

        # 정상 시퀀스: SpeechStart → SpeechChunks → SpeechEnd → STT Final
        await harness.self_owner.handle_vad_event(
            SpeechStart(uid, pre_roll=samples(0.0), chunk=samples(1.0))
        )
        for _ in range(3):
            await harness.self_owner.handle_vad_event(SpeechChunk(uid, chunk=samples(0.5)))
        await harness.self_owner.handle_vad_event(SpeechEnd(uid))

        # STT Final
        transcript = Transcript(
            utterance_id=uid, text="정상 발화", is_final=True, created_at=clock.now()
        )
        await harness.dispatch_stt_event(
            STTFinalEvent(utterance_id=transcript.utterance_id, transcript=transcript)
        )

        # grace period 대기
        await asyncio.sleep(0.02)

        # OSC 메시지 전송됨
        assert len(osc.messages) == 1
        assert "정상 발화" in osc.messages[0].text

        await harness.stop()

    @pytest.mark.asyncio
    async def test_speech_end_after_commit_pop_does_not_block(self):
        """이전 커밋에서 pop된 후 SpeechEnd가 와도 블록되지 않아야 함.

        이것은 99be2bfc 버그의 핵심 시나리오입니다:
        1. 첫 번째 버퍼가 utterance_id를 포함하여 커밋됨 (_utterance_start_times pop)
        2. 같은 utterance_id의 SpeechEnd가 나중에 도착
        3. 새 버퍼에서 같은 utterance_id의 STT Final이 도착
        4. _utterance_start_times.get() = None이지만 SpeechEnd가 이미 왔으므로 post_end 처리
        """
        clock = FakeClock(initial_time=10.0)
        harness = compose_translation_test_harness(
            stt=None,
            llm=None,
            osc=FakeOscQueue(),
            clock=clock,
            low_latency_mode=True,
            low_latency_finalize_wait_ms=0,
        )

        uid1 = uuid4()
        uid2 = uuid4()

        # 첫 번째 발화 완료 및 커밋
        await harness.self_owner.handle_vad_event(SpeechEnd(uid1))
        transcript1 = Transcript(
            utterance_id=uid1, text="첫 번째", is_final=True, created_at=clock.now()
        )
        await harness.dispatch_stt_event(
            STTFinalEvent(utterance_id=transcript1.utterance_id, transcript=transcript1)
        )

        # uid1이 _utterance_start_times에서 pop됨
        assert uid1 not in harness.self_runtime.utterance_start_times

        # 하지만 _speech_ended_ids에는 있음 (커밋 시 정리되었지만, 다시 추가될 수 있음)
        # 실제로는 첫 번째 커밋에서 정리되었을 것이므로 없음

        # uid2로 새 발화 시작
        await harness.self_owner.handle_vad_event(SpeechEnd(uid2))
        transcript2 = Transcript(
            utterance_id=uid2, text="두 번째", is_final=True, created_at=clock.now()
        )
        await harness.dispatch_stt_event(
            STTFinalEvent(utterance_id=transcript2.utterance_id, transcript=transcript2)
        )

        # 정상적으로 커밋됨 (블록 없음)
        assert harness.self_owner.merge_buffer is None

        await harness.stop()


class TestLowLatencyDisjointSpeech:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("text,merged", [("same", "same same"), ("안녕", "안녕안녕")])
    @pytest.mark.parametrize("same_turn", [False, True])
    async def test_repeated_text_from_distinct_requests_is_preserved(self, text, merged, same_turn):
        llm = FakeLLMProvider(delay_s=0.0)
        sink = RecordingOverlaySink()
        osc = FakeOscQueue()
        harness = compose_translation_test_harness(
            stt=None,
            llm=llm,
            osc=osc,
            overlay_sink=sink,
            low_latency_mode=True,
            low_latency_finalize_wait_ms=0,
        )
        first_id, second_id = uuid4(), uuid4()
        try:
            await _source_revision(harness, first_id, text)
            await _eventually(
                lambda: any(
                    getattr(event, "secondary_text", "") == "translated" for event in sink.events
                )
            )
            if not same_turn:
                await harness.self_owner.handle_vad_event(SpeechEnd(first_id))
                await asyncio.wait_for(harness.translation_turns.wait_for_idle(), timeout=1.0)
            await _source_revision(harness, second_id, text)
            if same_turn:
                await harness.self_owner.handle_vad_event(SpeechEnd(first_id))
            await harness.self_owner.handle_vad_event(SpeechEnd(second_id))
            await asyncio.wait_for(harness.translation_turns.wait_for_idle(), timeout=1.0)
            expected_sources = [merged] if same_turn else [text, text]
            assert [
                entry.text for entry in harness.self_runtime.translation_history
            ] == expected_sources
            snapshots = {
                event.payload.utterance_id: event.payload
                for event in _drain_ui(harness)
                if event.type == UIEventType.SELF_TURN_UPDATED
                and event.payload.status == "complete"
            }
            assert [snapshot.source_text for snapshot in snapshots.values()] == expected_sources
            assert len({message.utterance_id for message in osc.messages}) == (
                1 if same_turn else 2
            )
            assert len(llm.calls) == 2
        finally:
            await harness.stop()


class TestResumeEndTimeout:
    """Test resume_confirmed timeout when STT Final doesn't arrive (Pattern A)."""

    @pytest.mark.asyncio
    async def test_resume_confirmed_without_stt_final_times_out(self):
        """resume_confirmed 상태에서 STT Final 안 오면 타임아웃 후 커밋."""
        clock = FakeClock(initial_time=10.0)
        osc = FakeOscQueue()
        harness = compose_translation_test_harness(
            stt=None,
            llm=None,  # 번역 없이 직접 커밋
            osc=osc,
            clock=clock,
            low_latency_mode=True,
            low_latency_finalize_wait_ms=5000,  # 긴 grace period (커밋 안 되게)
            low_latency_awaiting_vad_timeout_s=0.1,  # 100ms 타임아웃
        )

        uid1 = uuid4()
        uid2 = uuid4()

        # 1. 첫 번째 발화 시작 및 STT Final
        await harness.self_owner.handle_vad_event(
            SpeechStart(uid1, pre_roll=samples(0.0), chunk=samples(1.0))
        )
        await harness.self_owner.handle_vad_event(SpeechEnd(uid1))
        transcript1 = Transcript(
            utterance_id=uid1, text="첫 번째 발화", is_final=True, created_at=clock.now()
        )
        await harness.dispatch_stt_event(
            STTFinalEvent(utterance_id=transcript1.utterance_id, transcript=transcript1)
        )

        # 버퍼에 텍스트가 있음 (grace period가 길어서 아직 커밋 안 됨)
        buffer = harness.self_owner.merge_buffer
        assert buffer is not None
        assert "첫 번째 발화" in harness.self_owner._merge_text(buffer.parts)

        # 2. 두 번째 발화 (resume) - SpeechStart
        await harness.self_owner.handle_vad_event(
            SpeechStart(uid2, pre_roll=samples(0.0), chunk=samples(1.0))
        )
        assert buffer.resume_pending is True

        # 3. SpeechChunk 3개 → resume_confirmed
        for _ in range(3):
            await harness.self_owner.handle_vad_event(SpeechChunk(uid2, chunk=samples(0.5)))
        assert buffer.resume_confirmed is True

        # 4. SpeechEnd (STT Final 없이) → 타임아웃 시작
        await harness.self_owner.handle_vad_event(SpeechEnd(uid2))
        assert buffer.resume_end_timeout_task is not None
        assert buffer.resume_end_utterance_id == uid2

        # 5. 타임아웃 대기 (150ms)
        await asyncio.sleep(0.15)

        # 6. 타임아웃으로 커밋됨
        assert harness.self_owner.merge_buffer is None
        assert len(osc.messages) == 1
        assert "첫 번째 발화" in osc.messages[0].text

        await harness.stop()

    @pytest.mark.asyncio
    async def test_resume_end_timeout_cancelled_when_stt_final_arrives(self):
        """STT Final이 오면 타임아웃 취소."""
        clock = FakeClock(initial_time=10.0)
        harness = compose_translation_test_harness(
            stt=None,
            llm=None,
            osc=FakeOscQueue(),
            clock=clock,
            low_latency_mode=True,
            low_latency_finalize_wait_ms=5000,  # 긴 grace period
            low_latency_awaiting_vad_timeout_s=0.5,  # 500ms 타임아웃
        )

        uid1 = uuid4()
        uid2 = uuid4()

        # 1. 첫 번째 발화
        await harness.self_owner.handle_vad_event(
            SpeechStart(uid1, pre_roll=samples(0.0), chunk=samples(1.0))
        )
        await harness.self_owner.handle_vad_event(SpeechEnd(uid1))
        transcript1 = Transcript(
            utterance_id=uid1, text="첫 번째", is_final=True, created_at=clock.now()
        )
        await harness.dispatch_stt_event(
            STTFinalEvent(utterance_id=transcript1.utterance_id, transcript=transcript1)
        )

        buffer = harness.self_owner.merge_buffer
        assert buffer is not None

        # 2. resume_confirmed 상태로 만들기
        await harness.self_owner.handle_vad_event(
            SpeechStart(uid2, pre_roll=samples(0.0), chunk=samples(1.0))
        )
        for _ in range(3):
            await harness.self_owner.handle_vad_event(SpeechChunk(uid2, chunk=samples(0.5)))
        assert buffer.resume_confirmed is True

        # 3. SpeechEnd → 타임아웃 시작
        await harness.self_owner.handle_vad_event(SpeechEnd(uid2))
        assert buffer.resume_end_timeout_task is not None

        # 4. STT Final 도착 → 타임아웃 취소 (via _clear_resume_state)
        transcript2 = Transcript(
            utterance_id=uid2, text="두 번째", is_final=True, created_at=clock.now()
        )
        await harness.dispatch_stt_event(
            STTFinalEvent(utterance_id=transcript2.utterance_id, transcript=transcript2)
        )

        # resume 상태 클리어됨 → 타임아웃도 취소됨
        assert buffer.resume_end_timeout_task is None
        assert buffer.resume_confirmed is False

        await harness.stop()

    @pytest.mark.asyncio
    async def test_new_false_start_keeps_older_unresolved_watchdog(self):
        osc = FakeOscQueue()
        harness = compose_translation_test_harness(
            stt=None,
            llm=None,
            osc=osc,
            low_latency_mode=True,
            low_latency_finalize_wait_ms=5000,
            low_latency_awaiting_vad_timeout_s=0.02,
        )
        first_id, unresolved_id, false_start_id = uuid4(), uuid4(), uuid4()
        try:
            await harness.self_owner.handle_vad_event(SpeechEnd(first_id))
            await _source_revision(harness, first_id, "첫 번째")
            buffer = harness.self_owner.merge_buffer
            assert buffer is not None
            await harness.self_owner.handle_vad_event(
                SpeechStart(unresolved_id, pre_roll=samples(0.0), chunk=samples(1.0))
            )
            for _ in range(3):
                await harness.self_owner.handle_vad_event(
                    SpeechChunk(unresolved_id, chunk=samples(0.5))
                )
            await harness.self_owner.handle_vad_event(SpeechEnd(unresolved_id))
            older_watchdog = buffer.resume_end_timeout_task
            assert older_watchdog is not None
            await harness.self_owner.handle_vad_event(
                SpeechStart(false_start_id, pre_roll=samples(0.0), chunk=samples(1.0))
            )
            harness.replace_configuration(low_latency_finalize_wait_ms=0)
            await harness.self_owner.handle_vad_event(SpeechEnd(false_start_id))
            assert not older_watchdog.done()
            assert harness.self_owner.merge_buffer is buffer
            assert osc.messages == []
            await asyncio.wait_for(older_watchdog, timeout=1.0)
            await asyncio.wait_for(harness.translation_turns.wait_for_idle(), timeout=1.0)
            assert harness.self_owner.merge_buffer is None
            assert [message.text for message in osc.messages] == ["첫 번째"]
        finally:
            await harness.stop()

    @pytest.mark.asyncio
    async def test_timeout_only_triggers_for_matched_utterance_id(self):
        """타임아웃은 정확히 매칭되는 utterance_id에서만 트리거."""
        clock = FakeClock(initial_time=10.0)
        harness = compose_translation_test_harness(
            stt=None,
            llm=None,
            osc=FakeOscQueue(),
            clock=clock,
            low_latency_mode=True,
            low_latency_finalize_wait_ms=5000,  # 긴 grace period
            low_latency_awaiting_vad_timeout_s=0.1,
        )

        uid1 = uuid4()
        uid2 = uuid4()

        # 1. 첫 번째 발화
        await harness.self_owner.handle_vad_event(
            SpeechStart(uid1, pre_roll=samples(0.0), chunk=samples(1.0))
        )
        await harness.self_owner.handle_vad_event(SpeechEnd(uid1))
        transcript1 = Transcript(
            utterance_id=uid1, text="첫 번째", is_final=True, created_at=clock.now()
        )
        await harness.dispatch_stt_event(
            STTFinalEvent(utterance_id=transcript1.utterance_id, transcript=transcript1)
        )

        buffer = harness.self_owner.merge_buffer
        assert buffer is not None

        # 2. uid2로 resume_confirmed
        await harness.self_owner.handle_vad_event(
            SpeechStart(uid2, pre_roll=samples(0.0), chunk=samples(1.0))
        )
        for _ in range(3):
            await harness.self_owner.handle_vad_event(SpeechChunk(uid2, chunk=samples(0.5)))

        # 3. 다른 uid의 SpeechEnd → 타임아웃 시작 안 됨
        await harness.self_owner.handle_vad_event(SpeechEnd(uid1))  # uid1의 SpeechEnd
        assert buffer.resume_end_timeout_task is None

        # 4. 정확한 uid의 SpeechEnd → 타임아웃 시작
        await harness.self_owner.handle_vad_event(SpeechEnd(uid2))
        assert buffer.resume_end_timeout_task is not None
        assert buffer.resume_end_utterance_id == uid2

        await harness.stop()

    @pytest.mark.asyncio
    async def test_resume_confirmed_without_stt_keeps_active_secondary_in_same_call(self):
        sink = RecordingOverlaySink()
        harness = compose_translation_test_harness(
            stt=None,
            llm=FakeLLMProvider(response_text="translated live", delay_s=0.0),
            osc=FakeOscQueue(),
            overlay_sink=sink,
            low_latency_mode=True,
        )
        first_id, resumed_id = uuid4(), uuid4()
        try:
            await _source_revision(harness, first_id, "첫 번째")
            await _eventually(
                lambda: any(
                    getattr(event, "secondary_text", "") == "translated live"
                    for event in sink.events
                )
            )
            await harness.self_owner.handle_vad_event(
                SpeechStart(resumed_id, pre_roll=samples(0.0), chunk=samples(1.0))
            )
            for _ in range(3):
                await harness.self_owner.handle_vad_event(
                    SpeechChunk(resumed_id, chunk=samples(0.5))
                )
            metadata = sink.active_self_overlay_metadata()
            assert metadata is not None
            assert metadata.text == "첫 번째"
            assert metadata.secondary_text == "translated live"
            assert harness.self_owner.merge_buffer is not None
        finally:
            await harness.stop()


class TestSpecFinalReconciliation:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "initial,revised",
        [("hello", "hello"), ("hello", "hello..."), ("안녕", "안녕。")],
    )
    @pytest.mark.parametrize("ready", [False, True])
    async def test_equivalent_source_reuses_one_result_through_closure(
        self, initial, revised, ready
    ):
        llm = BlockingLLMProvider(response_text="reused")
        sink = RecordingOverlaySink()
        osc = FakeOscQueue()
        harness = compose_translation_test_harness(
            stt=None,
            llm=llm,
            osc=osc,
            overlay_sink=sink,
            low_latency_mode=True,
            low_latency_finalize_wait_ms=0,
        )
        source_id = uuid4()
        try:
            await _source_revision(harness, source_id, initial)
            await asyncio.wait_for(llm.started.wait(), timeout=1.0)
            turn_id = sink.events[0].utterance_id
            if ready:
                llm.release.set()
                await _eventually(
                    lambda: any(event.secondary_text == "reused" for event in sink.events)
                )
            await _source_revision(harness, source_id, revised)
            await harness.self_owner.handle_vad_event(SpeechEnd(source_id))
            assert harness.self_owner.merge_buffer is None
            llm.release.set()
            await asyncio.wait_for(harness.translation_turns.wait_for_idle(), timeout=1.0)
            assert [call["text"] for call in llm.calls] == [initial]
            assert all(event.utterance_id == turn_id for event in sink.events)
            assert osc.messages[-1].utterance_id == turn_id
            assert osc.messages[-1].text == f"{revised} (reused)"
            assert [entry.text for entry in harness.self_runtime.translation_history] == [revised]
            snapshots = [
                event.payload
                for event in _drain_ui(harness)
                if event.type == UIEventType.SELF_TURN_UPDATED
            ]
            results = [
                target.translation
                for snapshot in snapshots
                for target in snapshot.targets
                if target.translation is not None
            ]
            assert results
            assert len({result.update_id for result in results}) == 1
            assert results[0].update_id
            assert snapshots[-1].source_closed
            assert snapshots[-1].status == "complete"
        finally:
            llm.release.set()
            await harness.stop()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "initial,revised",
        [("hello", "goodbye"), ("hello", "hello?"), ("pay 10", "pay 20"), ("go", "do not go")],
    )
    async def test_meaning_revision_restarts_once_and_rejects_retired_result(
        self, initial, revised
    ):
        llm = BlockingLLMProvider()
        sink = RecordingOverlaySink()
        harness = compose_translation_test_harness(
            stt=None,
            llm=llm,
            osc=FakeOscQueue(),
            overlay_sink=sink,
            low_latency_mode=True,
            low_latency_finalize_wait_ms=0,
        )
        source_id = uuid4()
        try:
            await _source_revision(harness, source_id, initial)
            await asyncio.wait_for(llm.started.wait(), timeout=1.0)
            await _source_revision(harness, source_id, revised)
            await _eventually(lambda: len(llm.calls) == 2)
            await _source_revision(harness, source_id, revised)
            await harness.self_owner.handle_vad_event(SpeechEnd(source_id))
            llm.release.set()
            await asyncio.wait_for(harness.translation_turns.wait_for_idle(), timeout=1.0)
            assert [call["text"] for call in llm.calls] == [initial, revised]
            translated = [
                event
                for event in sink.events
                if getattr(event, "secondary_text", "") == "translated"
            ]
            assert translated
            assert all(event.text == revised for event in translated)
            assert [entry.text for entry in harness.self_runtime.translation_history] == [revised]
        finally:
            llm.release.set()
            await harness.stop()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("translation_change", [False, True])
    async def test_configuration_change_preserves_only_compatible_work(self, translation_change):
        llm = BlockingLLMProvider()
        harness = compose_translation_test_harness(
            stt=None,
            llm=llm,
            osc=FakeOscQueue(),
            low_latency_mode=True,
            low_latency_finalize_wait_ms=0,
        )
        source_id = uuid4()
        try:
            await _source_revision(harness, source_id, "hello")
            await asyncio.wait_for(llm.started.wait(), timeout=1.0)
            harness.replace_configuration(
                **(
                    {"system_prompt": "new prompt"}
                    if translation_change
                    else {"chatbox_include_source": False}
                )
            )
            await _source_revision(harness, source_id, "hello")
            await harness.self_owner.handle_vad_event(SpeechEnd(source_id))
            llm.release.set()
            await asyncio.wait_for(harness.translation_turns.wait_for_idle(), timeout=1.0)
            assert len(llm.calls) == (2 if translation_change else 1)
            assert [entry.text for entry in harness.self_runtime.translation_history] == ["hello"]
        finally:
            llm.release.set()
            await harness.stop()

    @pytest.mark.asyncio
    async def test_failed_attempt_falls_back_once_after_grace(self):
        llm = FailingThenSuccessfulLLMProvider(response_text="fallback", delay_s=0.0)
        osc = FakeOscQueue()
        harness = compose_translation_test_harness(
            stt=None,
            llm=llm,
            osc=osc,
            low_latency_mode=True,
            low_latency_finalize_wait_ms=10,
        )
        source_id = uuid4()
        try:
            await harness.self_owner.handle_vad_event(SpeechEnd(source_id))
            await _source_revision(harness, source_id, "hello")
            await _eventually(lambda: harness.self_owner.merge_buffer is None)
            await asyncio.wait_for(harness.translation_turns.wait_for_idle(), timeout=1.0)
            assert [call["text"] for call in llm.calls] == ["hello", "hello"]
            assert [message.text for message in osc.messages] == ["hello (fallback)"]
            assert [entry.text for entry in harness.self_runtime.translation_history] == ["hello"]
        finally:
            await harness.stop()


class TestSpecCommitPaths:
    @pytest.mark.asyncio
    async def test_dual_target_progresses_secondary_first_and_reuses_both_running_calls(self):
        llm = PerTargetBlockingLLMProvider()
        osc = FakeOscQueue()
        sink = RecordingOverlaySink()
        harness = compose_translation_test_harness(
            stt=None,
            llm=llm,
            osc=osc,
            overlay_sink=sink,
            low_latency_mode=True,
            low_latency_finalize_wait_ms=20,
            source_language="en",
            target_language="zh-CN",
            self_target_languages=("zh-CN", "ja"),
            chatbox_include_source=True,
        )
        source_id = uuid4()
        try:
            await _source_revision(harness, source_id, "hello")
            await _eventually(lambda: llm.started_targets == {"zh-CN"})
            assert harness.self_runtime.translation_history == []
            await harness.self_owner.handle_vad_event(SpeechEnd(source_id))
            await asyncio.wait_for(llm.all_started.wait(), timeout=1.0)
            llm.releases["ja"].set()
            await _eventually(
                lambda: any(message.text == "translated ja" for message in osc.messages)
            )
            assert not llm.releases["zh-CN"].is_set()
            await _eventually(lambda: harness.self_owner.merge_buffer is None)
            assert [call["target_language"] for call in llm.calls] == ["zh-CN", "ja"]
            llm.releases["zh-CN"].set()
            await asyncio.wait_for(harness.translation_turns.wait_for_idle(), timeout=1.0)
            assert osc.messages[-1].text == "translated zh-CN\ntranslated ja"
            assert len({message.utterance_id for message in osc.messages}) == 1
            assert [
                (entry.text, entry.target_language)
                for entry in harness.self_runtime.translation_history
            ] == [
                ("hello", "zh-CN"),
                ("hello", "ja"),
            ]
        finally:
            for release in llm.releases.values():
                release.set()
            await harness.stop()

    @pytest.mark.asyncio
    async def test_resumed_source_restarts_both_targets_without_discarding_sticky_overlay(self):
        llm = PerTargetBlockingLLMProvider()
        sink = RecordingOverlaySink()
        osc = FakeOscQueue()
        harness = compose_translation_test_harness(
            stt=None,
            llm=llm,
            osc=osc,
            overlay_sink=sink,
            low_latency_mode=True,
            low_latency_finalize_wait_ms=5000,
            source_language="en",
            target_language="zh-CN",
            self_target_languages=("zh-CN", "ja"),
            chatbox_include_source=True,
        )
        first_id, resumed_id = uuid4(), uuid4()
        try:
            await harness.self_owner.handle_vad_event(SpeechEnd(first_id))
            await _source_revision(harness, first_id, "hello")
            await asyncio.wait_for(llm.all_started.wait(), timeout=1.0)
            llm.releases["ja"].set()
            await _eventually(
                lambda: any(
                    getattr(event, "secondary_text", "") == "translated ja" for event in sink.events
                )
            )
            await harness.self_owner.handle_vad_event(
                SpeechStart(resumed_id, pre_roll=samples(0.0), chunk=samples(1.0))
            )
            for _ in range(3):
                await harness.self_owner.handle_vad_event(
                    SpeechChunk(resumed_id, chunk=samples(0.5))
                )
            assert sink.active_self_overlay_metadata().secondary_text == "translated ja"
            harness.replace_configuration(low_latency_finalize_wait_ms=0)
            await harness.self_owner.handle_vad_event(SpeechEnd(resumed_id))
            await _source_revision(harness, resumed_id, "continued")
            await _eventually(lambda: len(llm.calls) == 4)
            for release in llm.releases.values():
                release.set()
            await asyncio.wait_for(harness.translation_turns.wait_for_idle(), timeout=1.0)
            assert [(call["text"], call["target_language"]) for call in llm.calls[2:]] == [
                ("hello continued", "zh-CN"),
                ("hello continued", "ja"),
            ]
            assert osc.messages[-1].text == "translated zh-CN\ntranslated ja"
        finally:
            for release in llm.releases.values():
                release.set()
            await harness.stop()

    @pytest.mark.asyncio
    async def test_failed_secondary_prestart_is_silent_and_retried_once_after_admission(self):
        llm = FailingSecondaryLLMProvider()
        osc = FakeOscQueue()
        harness = compose_translation_test_harness(
            stt=None,
            llm=llm,
            osc=osc,
            low_latency_mode=True,
            low_latency_finalize_wait_ms=20,
            source_language="en",
            target_language="zh-CN",
            self_target_languages=("zh-CN", "ja"),
            chatbox_include_source=True,
        )
        source_id = uuid4()
        try:
            await harness.self_owner.handle_vad_event(SpeechEnd(source_id))
            await _source_revision(harness, source_id, "hello")
            await asyncio.wait_for(llm.secondary_finished.wait(), timeout=1.0)
            assert not any(event.type == UIEventType.ERROR for event in _drain_ui(harness))
            llm.primary_release.set()
            await _eventually(lambda: harness.self_owner.merge_buffer is None)
            await asyncio.wait_for(harness.translation_turns.wait_for_idle(), timeout=1.0)
            assert llm.secondary_calls == 2
            assert osc.messages[-1].text == "translated zh-CN\ntranslated ja"
            assert not any(event.type == UIEventType.ERROR for event in _drain_ui(harness))
        finally:
            llm.primary_release.set()
            await harness.stop()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("secondary_ready", [False, True])
    async def test_provider_replacement_invalidates_both_targets(self, secondary_ready):
        old = PerTargetBlockingLLMProvider()
        new = FakeLLMProvider(response_text="new", delay_s=0.0)
        osc = FakeOscQueue()
        sink = RecordingOverlaySink()
        harness = compose_translation_test_harness(
            stt=None,
            llm=old,
            osc=osc,
            overlay_sink=sink,
            low_latency_mode=True,
            low_latency_finalize_wait_ms=50,
            source_language="en",
            target_language="zh-CN",
            self_target_languages=("zh-CN", "ja"),
            chatbox_include_source=True,
        )
        source_id = uuid4()
        try:
            await harness.self_owner.handle_vad_event(SpeechEnd(source_id))
            await _source_revision(harness, source_id, "hello")
            await asyncio.wait_for(old.all_started.wait(), timeout=1.0)
            if secondary_ready:
                old.releases["ja"].set()
                await _eventually(
                    lambda: any(
                        getattr(event, "secondary_text", "") == "translated ja"
                        for event in sink.events
                    )
                )
            await harness.replace_llm_provider(new)
            for release in old.releases.values():
                release.set()
            await _eventually(lambda: harness.self_owner.merge_buffer is None)
            await asyncio.wait_for(harness.translation_turns.wait_for_idle(), timeout=1.0)
            assert [call["target_language"] for call in new.calls] == ["zh-CN", "ja"]
            assert osc.messages[-1].text == "new\nnew"
            assert [
                (entry.text, entry.target_language)
                for entry in harness.self_runtime.translation_history
            ] == [
                ("hello", "zh-CN"),
                ("hello", "ja"),
            ]
        finally:
            for release in old.releases.values():
                release.set()
            await harness.stop()

    @pytest.mark.asyncio
    async def test_source_closes_and_manual_successor_executes_while_translation_pending(self):
        llm = BlockingLLMProvider()
        osc = FakeOscQueue()
        harness = compose_translation_test_harness(
            stt=None,
            llm=llm,
            osc=osc,
            low_latency_mode=True,
            low_latency_finalize_wait_ms=0,
        )
        source_id = uuid4()
        try:
            await _source_revision(harness, source_id, "hello")
            await asyncio.wait_for(llm.started.wait(), timeout=1.0)
            await harness.self_owner.handle_vad_event(SpeechEnd(source_id))
            assert harness.self_owner.merge_buffer is None
            manual = await harness.self_owner.submit_text("manual")
            await _eventually(lambda: len(llm.calls) == 2)
            assert [call["text"] for call in llm.calls] == ["hello", "manual"]
            assert manual.turn_id != llm.calls[0]["utterance_id"]
            assert osc.messages == []
            llm.release.set()
            await asyncio.wait_for(harness.translation_turns.wait_for_idle(), timeout=1.0)
            assert len(llm.calls) == 2
        finally:
            llm.release.set()
            await harness.stop()
