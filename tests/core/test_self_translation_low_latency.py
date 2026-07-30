"""Unit tests for low-latency mode awaiting_vad_end bug fix."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from uuid import UUID, uuid4

import numpy as np
import pytest

from puripuly_heart.core.orchestrator.channel_runtime import _MergeBuffer
from puripuly_heart.core.orchestrator.peer_translation_channel import (
    PeerTranslationChannelOwner,
)
from puripuly_heart.core.orchestrator.ports import compute_latency_dominant_stage
from puripuly_heart.core.orchestrator.self_translation_channel import (
    SelfTranslationChannelOwner,
)
from puripuly_heart.core.overlay.state import ActiveSelfOverlayMetadata
from puripuly_heart.core.runtime_logging import SessionLoggingMode
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
from tests.helpers.translation_owners import compose_translation_test_harness

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
    ):
        self.calls.append(
            {
                "utterance_id": utterance_id,
                "text": text,
                "context": context,
            }
        )
        await asyncio.sleep(self.delay_s)
        return Translation(utterance_id=utterance_id, text=self.response_text)

    async def close(self) -> None:
        pass


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
    ):
        self.calls.append(
            {
                "utterance_id": utterance_id,
                "text": text,
                "context": context,
            }
        )
        self.started.set()
        await self.release.wait()
        return Translation(utterance_id=utterance_id, text=self.response_text)

    async def close(self) -> None:
        self.close_calls += 1


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

    def send_immediate(self, text: str) -> bool:
        return True

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


def active_self_metadata_for_buffer(
    buffer: _MergeBuffer,
    *,
    text: str,
    secondary_text: str,
    source_language: str = "",
    target_language: str = "",
) -> ActiveSelfOverlayMetadata:
    return ActiveSelfOverlayMetadata(
        text=text,
        secondary_text=secondary_text,
        utterance_id=buffer.merge_id,
        occupant_key=f"self:{buffer.merge_id}",
        update_id=None,
        origin_wall_clock_ms=None,
        session_scope=None,
        source_text_hash=None,
        source_text_len=None,
        logical_turn_key=None,
        primary_language=(str(source_language or "").strip() or None),
        secondary_language=(
            (str(target_language or "").strip() or None) if secondary_text.strip() else None
        ),
    )


@dataclass
class RecordingOverlayDiagnostics:
    translation_events: list[dict[str, object]] = field(default_factory=list)

    def record_translation(self, event: str, **fields: object) -> dict[str, object]:
        payload = {"event": event, **fields}
        self.translation_events.append(payload)
        return payload


def samples(value: float, n: int = 512) -> np.ndarray:
    return np.full((n,), value, dtype=np.float32)


# ── Tests ─────────────────────────────────────────────────────────────────────


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
    assert "_peer_stt_task" not in PeerTranslationChannelOwner.__dataclass_fields__

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
    old_llm = BlockingLLMProvider(response_text="stale speculative translation", delay_s=0.0)
    new_llm = FakeLLMProvider(response_text="current translation", delay_s=0.0)
    osc = FakeOscQueue()
    overlay_sink = RecordingOverlaySink()
    harness = compose_translation_test_harness(
        stt=None,
        llm=old_llm,
        osc=osc,
        overlay_sink=overlay_sink,
        low_latency_mode=True,
        low_latency_finalize_wait_ms=0,
        low_latency_awaiting_vad_timeout_s=0,
    )

    await harness.self_owner._handle_low_latency_final(
        Transcript(
            utterance_id=uuid4(),
            text="hello live",
            is_final=True,
            created_at=0.0,
        )
    )
    buffer = harness.self_owner.merge_buffer
    assert buffer is not None
    spec_task = buffer.spec_task
    assert spec_task is not None
    await asyncio.wait_for(old_llm.started.wait(), timeout=1.0)

    await harness.replace_llm_provider(new_llm)
    assert old_llm.close_calls == 1

    old_llm.release.set()
    await asyncio.wait_for(spec_task, timeout=1.0)

    assert buffer.spec_translation is None
    assert osc.messages == []
    assert not any(
        getattr(event, "secondary_text", "") == "stale speculative translation"
        for event in overlay_sink.events
    )
    assert harness.ui_events.empty()

    await harness.stop()


@pytest.mark.asyncio
async def test_replaced_llm_provider_late_spec_completion_falls_back_without_hanging_commit() -> (
    None
):
    old_llm = BlockingLLMProvider(response_text="stale speculative translation", delay_s=0.0)
    new_llm = FakeLLMProvider(response_text="current fallback translation", delay_s=0.0)
    osc = FakeOscQueue()
    overlay_sink = RecordingOverlaySink()
    clock = FakeClock(initial_time=10.0)
    harness = compose_translation_test_harness(
        stt=None,
        llm=old_llm,
        osc=osc,
        overlay_sink=overlay_sink,
        clock=clock,
        low_latency_mode=True,
        low_latency_finalize_wait_ms=0,
        low_latency_awaiting_vad_timeout_s=0,
    )
    utterance_id = uuid4()
    await harness.self_owner.handle_vad_event(SpeechEnd(utterance_id))

    await harness.self_owner._handle_low_latency_final(
        Transcript(
            utterance_id=utterance_id,
            text="hello live",
            is_final=True,
            created_at=clock.now(),
        )
    )
    buffer = harness.self_owner.merge_buffer
    assert buffer is not None
    spec_task = buffer.spec_task
    assert spec_task is not None
    await asyncio.wait_for(old_llm.started.wait(), timeout=1.0)

    await harness.replace_llm_provider(new_llm)
    assert old_llm.close_calls == 1

    old_llm.release.set()
    await asyncio.wait_for(spec_task, timeout=1.0)

    assert harness.self_owner.merge_buffer is None
    assert buffer.spec_translation is None
    assert buffer.spec_latency_stage_times == {}
    assert new_llm.calls and new_llm.calls[-1]["text"] == "hello live"
    assert osc.messages
    assert "current fallback translation" in osc.messages[-1].text

    ui_events = []
    while not harness.ui_events.empty():
        ui_events.append(await harness.ui_events.get())
    assert not any(event.type == UIEventType.ERROR for event in ui_events)
    translation_events = [
        event for event in ui_events if event.type == UIEventType.TRANSLATION_DONE
    ]
    assert len(translation_events) == 1
    assert translation_events[0].payload.text == "current fallback translation"
    assert not any(
        getattr(event, "secondary_text", "") == "stale speculative translation"
        or getattr(event, "text", "") == "stale speculative translation"
        for event in overlay_sink.events
    )

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
        await harness.self_owner._handle_low_latency_final(transcript)

        assert harness.self_runtime.merge_buffer is harness.self_owner.merge_buffer
        assert harness.peer_runtime.merge_buffer is None
        assert harness.self_runtime.translation_history == []
        assert harness.peer_runtime.translation_history == []

        await harness.stop()


class TestRuntimeLatencyLogging:
    def test_compute_latency_dominant_stage_uses_largest_safe_duration(self):
        assert (
            compute_latency_dominant_stage(
                {
                    "speech_end_to_stt_final": 80,
                    "llm_request_to_llm_done": 320,
                    "ignored_missing": None,
                }
            )
            == "llm_request_to_llm_done"
        )

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
            await harness.self_owner._handle_low_latency_final(transcript)

            spec_task = (
                harness.self_owner.merge_buffer.spec_task
                if harness.self_owner.merge_buffer is not None
                else None
            )
            assert spec_task is not None
            await asyncio.gather(spec_task, return_exceptions=True)

            assert len(llm.calls) == 1
            assert llm.calls[0]["text"] == "hello"

            await harness.process_translation(uuid4(), "peer hello", runtime=harness.peer_runtime)

            assert [call["text"] for call in llm.calls] == ["hello", "peer hello"]
        finally:
            await harness.stop()

    @pytest.mark.asyncio
    async def test_basic_latency_summary_includes_self_hangover_without_stage(self):
        runtime_logging, log_stream = _make_runtime_logging_capture()
        clock = FakeClock(initial_time=10.0)
        harness = compose_translation_test_harness(
            stt=None,
            llm=None,
            osc=FakeOscQueue(),
            clock=clock,
            low_latency_mode=True,
            low_latency_finalize_wait_ms=0,
            hangover_s=9.9,
            runtime_logging=runtime_logging,
        )
        utterance_id = uuid4()

        try:
            await harness.self_owner.handle_vad_event(SpeechEnd(utterance_id))
            clock.advance(0.25)

            await harness.self_owner._handle_low_latency_final(
                Transcript(
                    utterance_id=utterance_id,
                    text="official latency",
                    is_final=True,
                    created_at=clock.now(),
                )
            )

            messages = _runtime_log_messages(log_stream)
            latency_message = next(message for message in messages if "[Basic][Latency]" in message)

            assert "channel=self" in latency_message
            assert "e2e_ms=10150" in latency_message
            assert "final_output_stage=" not in latency_message
            assert "speech_end_to_stt_final_ms=" not in latency_message
            assert "stt_final_to_final_output_ms=" not in latency_message
            assert "hangover" not in latency_message
        finally:
            runtime_logging.close()
            await harness.stop()

    @pytest.mark.asyncio
    async def test_detailed_latency_traces_emit_only_in_detailed_mode(self):
        basic_runtime_logging, basic_stream = _make_runtime_logging_capture()
        detailed_runtime_logging, detailed_stream = _make_runtime_logging_capture()
        detailed_runtime_logging.set_mode(SessionLoggingMode.DETAILED)

        basic_clock = FakeClock(initial_time=10.0)
        detailed_clock = FakeClock(initial_time=20.0)
        basic_harness = compose_translation_test_harness(
            stt=None,
            llm=None,
            osc=FakeOscQueue(),
            clock=basic_clock,
            low_latency_mode=True,
            low_latency_finalize_wait_ms=0,
            runtime_logging=basic_runtime_logging,
        )
        detailed_harness = compose_translation_test_harness(
            stt=None,
            llm=None,
            osc=FakeOscQueue(),
            clock=detailed_clock,
            low_latency_mode=True,
            low_latency_finalize_wait_ms=0,
            runtime_logging=detailed_runtime_logging,
        )

        try:
            basic_utterance_id = uuid4()
            await basic_harness.self_owner.handle_vad_event(SpeechEnd(basic_utterance_id))
            basic_clock.advance(0.05)
            await basic_harness.self_owner._handle_low_latency_final(
                Transcript(
                    utterance_id=basic_utterance_id,
                    text="basic only",
                    is_final=True,
                    created_at=basic_clock.now(),
                )
            )

            detailed_utterance_id = uuid4()
            await detailed_harness.self_owner.handle_vad_event(SpeechEnd(detailed_utterance_id))
            detailed_clock.advance(0.05)
            await detailed_harness.self_owner._handle_low_latency_final(
                Transcript(
                    utterance_id=detailed_utterance_id,
                    text="detailed trace",
                    is_final=True,
                    created_at=detailed_clock.now(),
                )
            )

            basic_messages = _runtime_log_messages(basic_stream)
            detailed_messages = _runtime_log_messages(detailed_stream)

            assert not any("[Detailed][Latency]" in message for message in basic_messages)
            assert not any("[Detailed][LatencyBreakdown]" in message for message in basic_messages)
            assert any(
                "[Detailed][Latency]" in message and "stage=speech_end" in message
                for message in detailed_messages
            )
            assert any(
                "[Detailed][Latency]" in message and "stage=stt_final" in message
                for message in detailed_messages
            )
            assert any(
                "[Detailed][Latency]" in message and "stage=self_chatbox_enqueue" in message
                for message in detailed_messages
            )
            assert any(
                "[Detailed][LatencyBreakdown]" in message
                and "channel=self" in message
                and "e2e_ms=1050" in message
                and "speech_end_to_stt_final_ms=50" in message
                and "stt_final_to_final_output_ms=0" in message
                and "final_output_stage=" not in message
                for message in detailed_messages
            )
        finally:
            basic_runtime_logging.close()
            detailed_runtime_logging.close()
            await basic_harness.stop()
            await detailed_harness.stop()

    @pytest.mark.asyncio
    async def test_latency_cause_metric_emits_only_in_detailed_mode_with_safe_facts(self):
        basic_runtime_logging, basic_stream = _make_runtime_logging_capture()
        detailed_runtime_logging, detailed_stream = _make_runtime_logging_capture()
        detailed_runtime_logging.set_mode(SessionLoggingMode.DETAILED)

        basic_clock = FakeClock(initial_time=10.0)
        detailed_clock = FakeClock(initial_time=20.0)
        basic_harness = compose_translation_test_harness(
            stt=None,
            llm=None,
            osc=FakeOscQueue(),
            clock=basic_clock,
            low_latency_mode=True,
            low_latency_finalize_wait_ms=0,
            runtime_logging=basic_runtime_logging,
        )
        detailed_harness = compose_translation_test_harness(
            stt=None,
            llm=None,
            osc=FakeOscQueue(),
            clock=detailed_clock,
            low_latency_mode=True,
            low_latency_finalize_wait_ms=0,
            runtime_logging=detailed_runtime_logging,
        )

        try:
            basic_utterance_id = uuid4()
            await basic_harness.self_owner.handle_vad_event(SpeechEnd(basic_utterance_id))
            basic_clock.advance(0.25)
            await basic_harness.self_owner._handle_low_latency_final(
                Transcript(
                    utterance_id=basic_utterance_id,
                    text="basic raw transcript secret-token",
                    is_final=True,
                    created_at=basic_clock.now(),
                )
            )

            detailed_utterance_id = uuid4()
            await detailed_harness.self_owner.handle_vad_event(SpeechEnd(detailed_utterance_id))
            detailed_clock.advance(0.25)
            await detailed_harness.self_owner._handle_low_latency_final(
                Transcript(
                    utterance_id=detailed_utterance_id,
                    text="detailed raw transcript secret-token",
                    is_final=True,
                    created_at=detailed_clock.now(),
                )
            )

            basic_messages = _runtime_log_messages(basic_stream)
            detailed_messages = _runtime_log_messages(detailed_stream)
            latency_cause = next(
                message for message in detailed_messages if "[Metric] latency_cause" in message
            )

            assert not any("[Metric] latency_cause" in message for message in basic_messages)
            assert "channel=self" in latency_cause
            assert "provider=stt" in latency_cause
            assert "dominant_stage=speech_end_to_stt_final" in latency_cause
            assert "speech_end_to_stt_final_ms=250" in latency_cause
            assert "raw transcript" not in latency_cause
            assert "secret-token" not in latency_cause
        finally:
            basic_runtime_logging.close()
            detailed_runtime_logging.close()
            await basic_harness.stop()
            await detailed_harness.stop()

    @pytest.mark.asyncio
    async def test_latency_cause_metric_prefers_translation_stage_when_llm_dominates(self):
        runtime_logging, log_stream = _make_runtime_logging_capture()
        runtime_logging.set_mode(SessionLoggingMode.DETAILED)
        clock = FakeClock(initial_time=10.0)
        harness = compose_translation_test_harness(
            stt=None,
            llm=ClockedTranslateLLMProvider(
                clock=clock,
                responses=[(0.30, "translated secret-token")],
            ),
            osc=FakeOscQueue(),
            clock=clock,
            low_latency_mode=True,
            low_latency_finalize_wait_ms=0,
            runtime_logging=runtime_logging,
        )
        utterance_id = uuid4()

        try:
            await harness.self_owner.handle_vad_event(SpeechEnd(utterance_id))
            clock.advance(0.05)
            await harness.self_owner._handle_low_latency_final(
                Transcript(
                    utterance_id=utterance_id,
                    text="source secret-token",
                    is_final=True,
                    created_at=clock.now(),
                )
            )
            spec_task = (
                harness.self_owner.merge_buffer.spec_task
                if harness.self_owner.merge_buffer is not None
                else None
            )
            assert spec_task is not None
            await asyncio.gather(spec_task, return_exceptions=True)

            messages = _runtime_log_messages(log_stream)
            latency_cause = next(
                message for message in messages if "[Metric] latency_cause" in message
            )
            assert "provider=llm" in latency_cause
            assert "dominant_stage=llm_request_to_llm_done" in latency_cause
            assert "llm_request_to_llm_done_ms=300" in latency_cause
            assert "source secret-token" not in latency_cause
            assert "translated secret-token" not in latency_cause
        finally:
            runtime_logging.close()
            await harness.stop()

    @pytest.mark.asyncio
    async def test_detailed_latency_trace_survives_basic_to_detailed_mode_switch_mid_utterance(
        self,
    ):
        runtime_logging, log_stream = _make_runtime_logging_capture()
        clock = FakeClock(initial_time=10.0)
        harness = compose_translation_test_harness(
            stt=None,
            llm=None,
            osc=FakeOscQueue(),
            clock=clock,
            low_latency_mode=True,
            low_latency_finalize_wait_ms=0,
            runtime_logging=runtime_logging,
        )
        utterance_id = uuid4()

        try:
            await harness.self_owner.handle_vad_event(SpeechEnd(utterance_id))
            runtime_logging.set_mode(SessionLoggingMode.DETAILED)
            clock.advance(0.05)

            await harness.self_owner._handle_low_latency_final(
                Transcript(
                    utterance_id=utterance_id,
                    text="mode switch",
                    is_final=True,
                    created_at=clock.now(),
                )
            )

            messages = _runtime_log_messages(log_stream)
            assert any(
                "[Detailed][Latency]" in message and "stage=speech_end" in message
                for message in messages
            )
            assert any(
                "[Detailed][Latency]" in message and "stage=stt_final" in message
                for message in messages
            )
        finally:
            runtime_logging.close()
            await harness.stop()

    @pytest.mark.asyncio
    async def test_low_latency_translation_ready_for_output_emits_only_in_detailed_mode(self):
        basic_runtime_logging, basic_stream = _make_runtime_logging_capture()
        detailed_runtime_logging, detailed_stream = _make_runtime_logging_capture()
        detailed_runtime_logging.set_mode(SessionLoggingMode.DETAILED)

        basic_overlay_sink = RecordingOverlaySink()
        detailed_overlay_sink = RecordingOverlaySink()
        basic_harness = compose_translation_test_harness(
            stt=None,
            llm=FakeLLMProvider(response_text="translated body", delay_s=0.0),
            osc=FakeOscQueue(),
            overlay_sink=basic_overlay_sink,
            runtime_logging=basic_runtime_logging,
            low_latency_mode=True,
            low_latency_finalize_wait_ms=0,
        )
        detailed_harness = compose_translation_test_harness(
            stt=None,
            llm=FakeLLMProvider(response_text="translated body", delay_s=0.0),
            osc=FakeOscQueue(),
            overlay_sink=detailed_overlay_sink,
            runtime_logging=detailed_runtime_logging,
            low_latency_mode=True,
            low_latency_finalize_wait_ms=0,
        )

        try:
            basic_utterance_id = uuid4()
            await basic_harness.process_translation(basic_utterance_id, "source body")

            detailed_utterance_id = uuid4()
            await detailed_harness.process_translation(detailed_utterance_id, "source body")

            basic_messages = _runtime_log_messages(basic_stream)
            detailed_messages = _runtime_log_messages(detailed_stream)
            detailed_translation = detailed_harness.bundle_for(detailed_utterance_id).translation
            assert detailed_translation is not None
            detailed_overlay_event = next(
                event
                for event in detailed_overlay_sink.events
                if getattr(event, "type", None) == "translation_final"
            )

            assert not any("translation_ready_for_output" in message for message in basic_messages)
            ready_message = next(
                message
                for message in detailed_messages
                if "translation_ready_for_output" in message
            )
            assert f"update_id={detailed_translation.update_id}" in ready_message
            assert (
                f"origin_wall_clock_ms={detailed_translation.origin_wall_clock_ms}" in ready_message
            )
            assert f"source_text_hash={detailed_translation.source_text_hash}" in ready_message
            assert f"source_text_len={detailed_translation.source_text_len}" in ready_message
            assert "source body" not in ready_message
            assert "translated body" not in ready_message

            assert detailed_overlay_event.update_id == detailed_translation.update_id
            assert (
                detailed_overlay_event.origin_wall_clock_ms
                == detailed_translation.origin_wall_clock_ms
            )
            assert detailed_overlay_event.session_scope == detailed_translation.session_scope
            assert detailed_overlay_event.source_text_hash == detailed_translation.source_text_hash
            assert detailed_overlay_event.source_text_len == detailed_translation.source_text_len
            assert detailed_overlay_event.logical_turn_key == detailed_translation.logical_turn_key
        finally:
            basic_runtime_logging.close()
            detailed_runtime_logging.close()
            await basic_harness.stop()
            await detailed_harness.stop()

    @pytest.mark.asyncio
    async def test_low_latency_reused_spec_translation_logs_llm_stages_on_output_path(self):
        runtime_logging, log_stream = _make_runtime_logging_capture()
        runtime_logging.set_mode(SessionLoggingMode.DETAILED)
        clock = FakeClock(initial_time=10.0)
        llm = FakeLLMProvider(response_text="translated", delay_s=0.0)
        osc = FakeOscQueue()
        harness = compose_translation_test_harness(
            stt=None,
            llm=llm,
            osc=osc,
            clock=clock,
            low_latency_mode=True,
            low_latency_finalize_wait_ms=0,
            runtime_logging=runtime_logging,
        )
        utterance_id = uuid4()

        try:
            await harness.self_owner.handle_vad_event(SpeechEnd(utterance_id))
            clock.advance(0.05)
            await harness.self_owner._handle_low_latency_final(
                Transcript(
                    utterance_id=utterance_id,
                    text="hello live",
                    is_final=True,
                    created_at=clock.now(),
                )
            )

            spec_task = (
                harness.self_owner.merge_buffer.spec_task
                if harness.self_owner.merge_buffer is not None
                else None
            )
            assert spec_task is not None
            await asyncio.gather(spec_task, return_exceptions=True)

            assert len(llm.calls) == 1
            output_utterance_id = osc.messages[0].utterance_id
            output_messages = [
                message
                for message in _runtime_log_messages(log_stream)
                if "[Detailed][Latency]" in message
                and f"utterance_id={str(output_utterance_id)[:8]}" in message
            ]

            assert any("stage=llm_request_start" in message for message in output_messages)
            assert any("stage=llm_done" in message for message in output_messages)
        finally:
            runtime_logging.close()
            await harness.stop()

    @pytest.mark.asyncio
    async def test_low_latency_self_output_path_uses_merge_id_for_detailed_traces(self):
        runtime_logging, log_stream = _make_runtime_logging_capture()
        runtime_logging.set_mode(SessionLoggingMode.DETAILED)
        clock = FakeClock(initial_time=10.0)
        osc = FakeOscQueue()
        overlay_sink = RecordingOverlaySink()
        harness = compose_translation_test_harness(
            stt=None,
            llm=None,
            osc=osc,
            overlay_sink=overlay_sink,
            clock=clock,
            low_latency_mode=True,
            low_latency_finalize_wait_ms=0,
            runtime_logging=runtime_logging,
        )
        source_utterance_id = uuid4()

        try:
            await harness.self_owner.handle_vad_event(SpeechEnd(source_utterance_id))
            clock.advance(0.05)
            await harness.self_owner._handle_low_latency_final(
                Transcript(
                    utterance_id=source_utterance_id,
                    text="merge path",
                    is_final=True,
                    created_at=clock.now(),
                )
            )

            output_utterance_id = osc.messages[0].utterance_id
            active_events = [
                event for event in overlay_sink.events if event.type == "self_active_update"
            ]
            messages = _runtime_log_messages(log_stream)
            output_messages = [
                message
                for message in messages
                if "[Detailed][Latency]" in message
                and f"utterance_id={str(output_utterance_id)[:8]}" in message
            ]
            source_messages = [
                message
                for message in messages
                if "[Detailed][Latency]" in message
                and f"utterance_id={str(source_utterance_id)[:8]}" in message
            ]

            assert len(active_events) == 1
            assert active_events[0].utterance_id == output_utterance_id
            assert any("stage=speech_end" in message for message in output_messages)
            assert any("stage=stt_final" in message for message in output_messages)
            assert any("stage=self_chatbox_enqueue" in message for message in output_messages)
            assert source_messages == []
        finally:
            runtime_logging.close()
            await harness.stop()

    @pytest.mark.asyncio
    async def test_low_latency_final_mismatch_uses_final_request_for_official_llm_traces(self):
        runtime_logging, log_stream = _make_runtime_logging_capture()
        runtime_logging.set_mode(SessionLoggingMode.DETAILED)
        clock = FakeClock(initial_time=10.0)
        llm = ClockedTranslateLLMProvider(
            clock=clock,
            responses=[(0.10, "spec translated"), (0.30, "final translated")],
        )
        osc = FakeOscQueue()
        harness = compose_translation_test_harness(
            stt=None,
            llm=llm,
            osc=osc,
            clock=clock,
            low_latency_mode=True,
            runtime_logging=runtime_logging,
        )
        source_utterance_id = uuid4()
        merge_id = uuid4()
        buffer = _MergeBuffer(
            merge_id=merge_id,
            parts=["final output"],
            utterance_ids=[source_utterance_id],
            start_time=10.0,
            last_end_time=10.0,
            spec_text="spec output",
            spec_attempts=1,
        )
        harness.self_owner.merge_buffer = buffer
        harness.self_runtime.utterance_start_times[source_utterance_id] = 10.0

        try:
            harness.peer_owner._record_latency_stage(
                channel="self",
                utterance_id=source_utterance_id,
                stage="speech_end",
                timestamp=10.0,
                publish_now=False,
            )
            harness.peer_owner._record_latency_stage(
                channel="self",
                utterance_id=source_utterance_id,
                stage="stt_final",
                timestamp=10.05,
                publish_now=False,
            )

            await harness.self_owner._run_spec_translation(merge_id, "spec output", 1)
            await harness.self_owner._commit_merge(buffer, reason="spec_done")

            output_messages = [
                message
                for message in _runtime_log_messages(log_stream)
                if "[Detailed][Latency]" in message
                and f"utterance_id={str(merge_id)[:8]}" in message
            ]

            assert any("stage=llm_request_start" in message for message in output_messages)
            assert any("stage=llm_done" in message for message in output_messages)
            assert any(
                "stage=llm_request_start" in message and "elapsed_ms=100" in message
                for message in output_messages
            )
            assert any(
                "stage=llm_done" in message and "elapsed_ms=400" in message
                for message in output_messages
            )
            assert not any(
                "stage=llm_request_start" in message and "elapsed_ms=50" in message
                for message in output_messages
            )
            assert not any(
                "stage=llm_done" in message and "elapsed_ms=100" in message
                for message in output_messages
            )
        finally:
            runtime_logging.close()
            await harness.stop()

    @pytest.mark.asyncio
    async def test_low_latency_spec_cancel_drops_exploratory_llm_traces(self):
        runtime_logging, log_stream = _make_runtime_logging_capture()
        runtime_logging.set_mode(SessionLoggingMode.DETAILED)
        clock = FakeClock(initial_time=10.0)
        llm = ClockedTranslateLLMProvider(
            clock=clock,
            responses=[(0.10, "spec translated")],
        )
        osc = FakeOscQueue()
        harness = compose_translation_test_harness(
            stt=None,
            llm=llm,
            osc=osc,
            clock=clock,
            low_latency_mode=True,
            low_latency_finalize_wait_ms=0,
            runtime_logging=runtime_logging,
        )
        source_utterance_id = uuid4()
        merge_id = uuid4()
        buffer = _MergeBuffer(
            merge_id=merge_id,
            parts=["final output"],
            utterance_ids=[source_utterance_id],
            start_time=10.0,
            last_end_time=10.0,
            spec_text="spec output",
            spec_attempts=1,
        )
        harness.self_owner.merge_buffer = buffer
        harness.self_runtime.utterance_start_times[source_utterance_id] = 10.0

        try:
            harness.peer_owner._record_latency_stage(
                channel="self",
                utterance_id=source_utterance_id,
                stage="speech_end",
                timestamp=10.0,
                publish_now=False,
            )
            harness.peer_owner._record_latency_stage(
                channel="self",
                utterance_id=source_utterance_id,
                stage="stt_final",
                timestamp=10.05,
                publish_now=False,
            )

            await harness.self_owner._run_spec_translation(merge_id, "spec output", 1)
            assert harness.self_owner._clear_spec_state(buffer, reason="spec_retry") is True
            harness.llm_runtime.attach_provider_reference(None)
            harness.replace_configuration(translation_enabled=False)

            await harness.self_owner._commit_merge(buffer, reason="final_no_llm")

            output_messages = [
                message
                for message in _runtime_log_messages(log_stream)
                if "[Detailed][Latency]" in message
                and f"utterance_id={str(merge_id)[:8]}" in message
            ]

            assert any("stage=speech_end" in message for message in output_messages)
            assert any("stage=stt_final" in message for message in output_messages)
            assert any("stage=self_chatbox_enqueue" in message for message in output_messages)
            assert not any("stage=llm_request_start" in message for message in output_messages)
            assert not any("stage=llm_done" in message for message in output_messages)
        finally:
            runtime_logging.close()
            await harness.stop()

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
        await harness.self_owner._handle_low_latency_final(transcript)

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
        await harness.self_owner._handle_low_latency_final(transcript)

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
        await harness.self_owner._handle_low_latency_final(transcript)

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
        await harness.self_owner._handle_low_latency_final(transcript)

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
        await harness.self_owner._handle_low_latency_final(transcript)

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
        await harness.self_owner._handle_low_latency_final(transcript)

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
        await harness.self_owner._handle_low_latency_final(transcript1)

        # uid1이 _utterance_start_times에서 pop됨
        assert uid1 not in harness.self_runtime.utterance_start_times

        # 하지만 _speech_ended_ids에는 있음 (커밋 시 정리되었지만, 다시 추가될 수 있음)
        # 실제로는 첫 번째 커밋에서 정리되었을 것이므로 없음

        # uid2로 새 발화 시작
        await harness.self_owner.handle_vad_event(SpeechEnd(uid2))
        transcript2 = Transcript(
            utterance_id=uid2, text="두 번째", is_final=True, created_at=clock.now()
        )
        await harness.self_owner._handle_low_latency_final(transcript2)

        # 정상적으로 커밋됨 (블록 없음)
        assert harness.self_owner.merge_buffer is None

        await harness.stop()


class TestLowLatencyMergeOverlap:
    """Test relaxed overlap merge behavior."""

    def test_relaxed_overlap_strips_boundary_punct(self):
        harness = compose_translation_test_harness(stt=None, llm=None, osc=FakeOscQueue())
        merged = harness.self_owner._merge_with_overlap("같으면서.", "같으면서도 안.")
        assert merged == "같으면서도 안."

    def test_relaxed_overlap_min_length(self):
        harness = compose_translation_test_harness(stt=None, llm=None, osc=FakeOscQueue())
        merged = harness.self_owner._merge_with_overlap("가다.", "가다고")
        assert merged == "가다.가다고"


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
        await harness.self_owner._handle_low_latency_final(transcript1)

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
        await harness.self_owner._handle_low_latency_final(transcript1)

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
        await harness.self_owner._handle_low_latency_final(transcript2)

        # resume 상태 클리어됨 → 타임아웃도 취소됨
        assert buffer.resume_end_timeout_task is None
        assert buffer.resume_confirmed is False

        await harness.stop()

    @pytest.mark.asyncio
    async def test_new_resume_cancels_previous_timeout(self):
        """새 resume 시작 시 이전 타임아웃 취소."""
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
        uid3 = uuid4()

        # 1. 첫 번째 발화
        await harness.self_owner.handle_vad_event(
            SpeechStart(uid1, pre_roll=samples(0.0), chunk=samples(1.0))
        )
        await harness.self_owner.handle_vad_event(SpeechEnd(uid1))
        transcript1 = Transcript(
            utterance_id=uid1, text="첫 번째", is_final=True, created_at=clock.now()
        )
        await harness.self_owner._handle_low_latency_final(transcript1)

        buffer = harness.self_owner.merge_buffer
        assert buffer is not None

        # 2. uid2로 resume_confirmed + SpeechEnd → 타임아웃 시작
        await harness.self_owner.handle_vad_event(
            SpeechStart(uid2, pre_roll=samples(0.0), chunk=samples(1.0))
        )
        for _ in range(3):
            await harness.self_owner.handle_vad_event(SpeechChunk(uid2, chunk=samples(0.5)))
        await harness.self_owner.handle_vad_event(SpeechEnd(uid2))

        old_timeout_task = buffer.resume_end_timeout_task
        assert old_timeout_task is not None
        assert buffer.resume_end_utterance_id == uid2

        # 3. uid3로 새 resume 시작 → 이전 타임아웃 취소
        await harness.self_owner.handle_vad_event(
            SpeechStart(uid3, pre_roll=samples(0.0), chunk=samples(1.0))
        )

        # 이전 타임아웃 취소됨
        assert buffer.resume_end_timeout_task is None
        assert buffer.resume_end_utterance_id is None
        # 새 resume 상태
        assert buffer.resume_pending is True
        assert buffer.resume_utterance_id == uid3

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
        await harness.self_owner._handle_low_latency_final(transcript1)

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
        clock = FakeClock(initial_time=10.0)
        overlay_sink = RecordingOverlaySink()
        harness = compose_translation_test_harness(
            stt=None,
            llm=None,
            osc=FakeOscQueue(),
            overlay_sink=overlay_sink,
            clock=clock,
            low_latency_mode=True,
        )
        first_utterance_id = uuid4()
        resumed_utterance_id = uuid4()
        merge_id = uuid4()
        buffer = _MergeBuffer(
            merge_id=merge_id,
            parts=["첫 번째"],
            utterance_ids=[first_utterance_id],
            spec_text="첫 번째",
            spec_translation=Translation(utterance_id=merge_id, text="translated live"),
            resume_pending=True,
            resume_utterance_id=resumed_utterance_id,
            resume_chunk_count=2,
        )
        harness.self_owner.merge_buffer = buffer
        overlay_sink.active_self_metadata = active_self_metadata_for_buffer(
            buffer,
            text="첫 번째",
            secondary_text="translated live",
            source_language=harness.configuration.snapshot().value.source_language,
            target_language=harness.configuration.snapshot().value.target_language,
        )

        await harness.self_owner.handle_vad_event(
            SpeechChunk(resumed_utterance_id, chunk=samples(0.5))
        )

        assert buffer.resume_confirmed is True
        assert buffer.spec_translation is None
        assert overlay_sink.events == []
        metadata = overlay_sink.active_self_overlay_metadata()
        assert metadata is not None
        assert metadata.secondary_text == "translated live"


class TestSpecCommitPaths:
    def test_soft_reuse_mode_accepts_only_safe_boundary_changes(self):
        harness = compose_translation_test_harness(
            stt=None,
            llm=FakeLLMProvider(),
            osc=FakeOscQueue(),
            clock=FakeClock(initial_time=10.0),
            low_latency_mode=True,
        )

        assert harness.output_projection.soft_reuse_mode(" hello ", "hello...") == "soft_boundary"
        assert harness.output_projection.soft_reuse_mode("hello", ",hello") == "soft_boundary"
        assert harness.output_projection.soft_reuse_mode("안녕", "안녕。") == "soft_boundary"
        assert harness.output_projection.soft_reuse_mode("안녕", "안녕，") == "soft_boundary"
        assert harness.output_projection.soft_reuse_mode("hello", "hello?") is None

    @pytest.mark.asyncio
    async def test_commit_merge_reuses_spec_translation_when_text_matches(self):
        clock = FakeClock(initial_time=10.0)
        osc = FakeOscQueue()
        overlay_sink = RecordingOverlaySink()
        harness = compose_translation_test_harness(
            stt=None,
            llm=FakeLLMProvider(),
            osc=osc,
            overlay_sink=overlay_sink,
            clock=clock,
            low_latency_mode=True,
        )
        uid = uuid4()
        merge_id = uuid4()
        buffer = _MergeBuffer(
            merge_id=merge_id,
            parts=["hello"],
            utterance_ids=[uid],
            start_time=clock.now(),
            last_end_time=clock.now(),
            spec_text="hello",
            spec_translation=Translation(utterance_id=merge_id, text="hola"),
        )
        harness.self_owner.merge_buffer = buffer
        harness.self_runtime.utterance_start_times[uid] = clock.now()

        await harness.self_owner._commit_merge(buffer, reason="spec_done")

        assert harness.self_owner.merge_buffer is None
        assert len(osc.messages) == 1
        assert osc.messages[0].text == "hello (hola)"
        assert len(harness.self_runtime.translation_history) == 1
        assert [event.type for event in overlay_sink.events] == [
            "self_transcript_final",
            "translation_final",
            "utterance_closed",
        ]
        assert overlay_sink.events[1].text == "hola"
        assert overlay_sink.events[1].utterance_id == merge_id
        assert overlay_sink.events[-1].channel == "self"
        assert overlay_sink.events[-1].is_final is True

    @pytest.mark.asyncio
    async def test_commit_merge_self_active_update_carries_merge_id_when_clearing_stale_active_secondary_before_finalizing_mismatch(
        self,
    ):
        clock = FakeClock(initial_time=10.0)
        osc = FakeOscQueue()
        overlay_sink = RecordingOverlaySink()
        harness = compose_translation_test_harness(
            stt=None,
            llm=FakeLLMProvider(response_text="nuevo", delay_s=0.0),
            osc=osc,
            overlay_sink=overlay_sink,
            clock=clock,
            low_latency_mode=True,
        )
        source_utterance_id = uuid4()
        merge_id = uuid4()
        buffer = _MergeBuffer(
            merge_id=merge_id,
            parts=["hello live"],
            utterance_ids=[source_utterance_id],
            start_time=clock.now(),
            last_end_time=clock.now(),
            spec_text="hello live",
            spec_translation=Translation(utterance_id=merge_id, text="translated live"),
        )
        harness.self_owner.merge_buffer = buffer
        harness.self_runtime.utterance_start_times[source_utterance_id] = clock.now()

        await harness.self_owner._sync_overlay_active_self(buffer, created_at=clock.now())
        buffer.spec_text = "goodbye live"

        await harness.self_owner._commit_merge(buffer, reason="spec_done")

        assert [event.type for event in overlay_sink.events] == [
            "self_active_update",
            "self_active_update",
            "self_transcript_final",
            "translation_final",
            "utterance_closed",
        ]
        assert overlay_sink.events[0].text == "hello live"
        assert overlay_sink.events[0].secondary_text == "translated live"
        assert overlay_sink.events[0].utterance_id == merge_id
        assert overlay_sink.events[1].text == "hello live"
        assert overlay_sink.events[1].secondary_text == ""
        assert overlay_sink.events[1].utterance_id == merge_id
        assert overlay_sink.events[1].occupant_key == overlay_sink.events[0].occupant_key
        assert overlay_sink.events[2].utterance_id == merge_id
        assert overlay_sink.events[3].text == "nuevo"
        assert overlay_sink.events[3].utterance_id == merge_id

    @pytest.mark.asyncio
    async def test_low_latency_repeated_same_utterance_overlay_updates_use_distinct_update_id(
        self,
    ):
        clock = FakeClock(initial_time=10.0)
        overlay_sink = RecordingOverlaySink()
        harness = compose_translation_test_harness(
            stt=None,
            llm=FakeLLMProvider(response_text="nuevo", delay_s=0.0),
            osc=FakeOscQueue(),
            overlay_sink=overlay_sink,
            clock=clock,
            low_latency_mode=True,
        )
        source_utterance_id = uuid4()
        merge_id = uuid4()
        buffer = _MergeBuffer(
            merge_id=merge_id,
            parts=["hello live"],
            utterance_ids=[source_utterance_id],
            start_time=clock.now(),
            last_end_time=clock.now(),
            spec_text="hello live",
            spec_translation=Translation(utterance_id=merge_id, text="translated live"),
        )
        harness.self_owner.merge_buffer = buffer
        harness.self_runtime.utterance_start_times[source_utterance_id] = clock.now()

        await harness.self_owner._sync_overlay_active_self(buffer, created_at=clock.now())
        buffer.spec_text = "goodbye live"
        await harness.self_owner._commit_merge(buffer, reason="spec_done")

        preview_event = next(
            event
            for event in overlay_sink.events
            if getattr(event, "type", None) == "self_active_update"
            and event.secondary_text == "translated live"
        )
        final_event = next(
            event
            for event in overlay_sink.events
            if getattr(event, "type", None) == "translation_final"
        )

        assert preview_event.utterance_id == merge_id
        assert final_event.utterance_id == merge_id
        assert isinstance(preview_event.update_id, str) and preview_event.update_id
        assert isinstance(final_event.update_id, str) and final_event.update_id
        assert preview_event.update_id != final_event.update_id

    @pytest.mark.asyncio
    async def test_sync_self_active_overlay_records_overlay_emit_with_merge_id(self):
        clock = FakeClock(initial_time=10.0)
        diagnostics = RecordingOverlayDiagnostics()
        overlay_sink = RecordingOverlaySink()
        harness = compose_translation_test_harness(
            stt=None,
            llm=None,
            osc=FakeOscQueue(),
            overlay_sink=overlay_sink,
            overlay_diagnostics=diagnostics,
            clock=clock,
            low_latency_mode=True,
        )
        source_utterance_id = uuid4()
        merge_id = uuid4()
        buffer = _MergeBuffer(
            merge_id=merge_id,
            parts=["hello live"],
            utterance_ids=[source_utterance_id],
            spec_text="hello live",
            spec_translation=Translation(utterance_id=merge_id, text="translated live"),
        )

        await harness.self_owner._sync_overlay_active_self(buffer, created_at=clock.now())

        active_self_emit = next(
            event
            for event in diagnostics.translation_events
            if event["event"] == "overlay_emit" and event["event_kind"] == "active_self"
        )
        assert active_self_emit["utterance_id"] == str(merge_id)
        assert active_self_emit["secondary_len"] == len("translated live")

    @pytest.mark.asyncio
    async def test_sync_self_active_overlay_dedupes_only_within_same_logical_turn(self):
        clock = FakeClock(initial_time=10.0)
        overlay_sink = RecordingOverlaySink()
        harness = compose_translation_test_harness(
            stt=None,
            llm=None,
            osc=FakeOscQueue(),
            overlay_sink=overlay_sink,
            clock=clock,
            low_latency_mode=True,
        )
        first_buffer = _MergeBuffer(
            merge_id=uuid4(),
            parts=["same preview"],
            utterance_ids=[uuid4()],
        )
        second_buffer = _MergeBuffer(
            merge_id=uuid4(),
            parts=["same preview"],
            utterance_ids=[uuid4()],
        )

        await harness.self_owner._sync_overlay_active_self(first_buffer, created_at=clock.now())
        await harness.self_owner._sync_overlay_active_self(second_buffer, created_at=clock.now())

        active_events = [
            event for event in overlay_sink.events if event.type == "self_active_update"
        ]
        assert [event.utterance_id for event in active_events] == [
            first_buffer.merge_id,
            second_buffer.merge_id,
        ]
        assert [event.occupant_key for event in active_events] == [
            f"self:{first_buffer.merge_id}",
            f"self:{second_buffer.merge_id}",
        ]

    @pytest.mark.asyncio
    async def test_sync_self_active_overlay_re_emits_same_preview_when_update_id_changes(self):
        clock = FakeClock(initial_time=10.0)
        overlay_sink = RecordingOverlaySink()
        harness = compose_translation_test_harness(
            stt=None,
            llm=None,
            osc=FakeOscQueue(),
            overlay_sink=overlay_sink,
            clock=clock,
            low_latency_mode=True,
        )
        merge_id = uuid4()
        buffer = _MergeBuffer(
            merge_id=merge_id,
            parts=["same preview"],
            utterance_ids=[uuid4()],
            spec_text="same preview",
            spec_translation=Translation(
                utterance_id=merge_id,
                text="translated live",
                update_id="upd-1",
                origin_wall_clock_ms=111,
                session_scope="session:self",
                source_text_hash="hash-111",
                source_text_len=12,
                logical_turn_key=f"self:{merge_id}",
            ),
        )

        await harness.self_owner._sync_overlay_active_self(buffer, created_at=clock.now())

        buffer.spec_translation = Translation(
            utterance_id=merge_id,
            text="translated live",
            update_id="upd-2",
            origin_wall_clock_ms=222,
            session_scope="session:self",
            source_text_hash="hash-222",
            source_text_len=12,
            logical_turn_key=f"self:{merge_id}",
        )
        await harness.self_owner._sync_overlay_active_self(buffer, created_at=clock.now())

        active_events = [
            event for event in overlay_sink.events if event.type == "self_active_update"
        ]

        assert len(active_events) == 2
        assert [event.text for event in active_events] == ["same preview", "same preview"]
        assert [event.secondary_text for event in active_events] == [
            "translated live",
            "translated live",
        ]
        assert [event.update_id for event in active_events] == ["upd-1", "upd-2"]
        assert [event.origin_wall_clock_ms for event in active_events] == [111, 222]
        assert [event.source_text_hash for event in active_events] == ["hash-111", "hash-222"]

    @pytest.mark.asyncio
    async def test_commit_merge_blanking_active_self_records_overlay_emit_with_merge_id(self):
        clock = FakeClock(initial_time=10.0)
        diagnostics = RecordingOverlayDiagnostics()
        overlay_sink = RecordingOverlaySink()
        harness = compose_translation_test_harness(
            stt=None,
            llm=FakeLLMProvider(response_text="nuevo", delay_s=0.0),
            osc=FakeOscQueue(),
            overlay_sink=overlay_sink,
            overlay_diagnostics=diagnostics,
            clock=clock,
            low_latency_mode=True,
        )
        source_utterance_id = uuid4()
        merge_id = uuid4()
        buffer = _MergeBuffer(
            merge_id=merge_id,
            parts=["hello live"],
            utterance_ids=[source_utterance_id],
            start_time=clock.now(),
            last_end_time=clock.now(),
            spec_text="hello live",
            spec_translation=Translation(utterance_id=merge_id, text="translated live"),
        )
        harness.self_owner.merge_buffer = buffer
        harness.self_runtime.utterance_start_times[source_utterance_id] = clock.now()

        await harness.self_owner._sync_overlay_active_self(buffer, created_at=clock.now())
        buffer.spec_text = "goodbye live"
        await harness.self_owner._commit_merge(buffer, reason="spec_done")

        active_self_emits = [
            event
            for event in diagnostics.translation_events
            if event["event"] == "overlay_emit" and event["event_kind"] == "active_self"
        ]

        assert [event["utterance_id"] for event in active_self_emits] == [
            str(merge_id),
            str(merge_id),
        ]
        assert [event["secondary_len"] for event in active_self_emits] == [
            len("translated live"),
            0,
        ]

    @pytest.mark.asyncio
    async def test_commit_merge_reuses_spec_translation_for_soft_boundary_difference(self):
        clock = FakeClock(initial_time=10.0)
        llm = FakeLLMProvider()
        osc = FakeOscQueue()
        harness = compose_translation_test_harness(
            stt=None,
            llm=llm,
            osc=osc,
            clock=clock,
            low_latency_mode=True,
        )
        uid = uuid4()
        merge_id = uuid4()
        buffer = _MergeBuffer(
            merge_id=merge_id,
            parts=["hello..."],
            utterance_ids=[uid],
            start_time=clock.now(),
            last_end_time=clock.now(),
            spec_text="hello",
            spec_translation=Translation(utterance_id=merge_id, text="hola"),
        )
        harness.self_owner.merge_buffer = buffer
        harness.self_runtime.utterance_start_times[uid] = clock.now()

        await harness.self_owner._commit_merge(buffer, reason="spec_done")

        assert harness.self_owner.merge_buffer is None
        assert llm.calls == []
        assert len(osc.messages) == 1
        assert osc.messages[0].text == "hello... (hola)"

    @pytest.mark.asyncio
    async def test_commit_merge_reuses_spec_translation_for_multilingual_boundary_difference(self):
        clock = FakeClock(initial_time=10.0)
        llm = FakeLLMProvider()
        osc = FakeOscQueue()
        harness = compose_translation_test_harness(
            stt=None,
            llm=llm,
            osc=osc,
            clock=clock,
            low_latency_mode=True,
        )
        uid = uuid4()
        merge_id = uuid4()
        buffer = _MergeBuffer(
            merge_id=merge_id,
            parts=["안녕。"],
            utterance_ids=[uid],
            start_time=clock.now(),
            last_end_time=clock.now(),
            spec_text="안녕",
            spec_translation=Translation(utterance_id=merge_id, text="你好"),
        )
        harness.self_owner.merge_buffer = buffer
        harness.self_runtime.utterance_start_times[uid] = clock.now()

        await harness.self_owner._commit_merge(buffer, reason="spec_done")

        assert harness.self_owner.merge_buffer is None
        assert llm.calls == []
        assert len(osc.messages) == 1
        assert osc.messages[0].text == "안녕。 (你好)"

    @pytest.mark.asyncio
    async def test_try_commit_after_spec_allows_soft_boundary_match(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        harness = compose_translation_test_harness(
            stt=None,
            llm=FakeLLMProvider(),
            osc=FakeOscQueue(),
            clock=FakeClock(initial_time=10.0),
            low_latency_mode=True,
        )
        buffer = _MergeBuffer(
            merge_id=uuid4(),
            parts=["hello..."],
            spec_text="hello",
            spec_translation=Translation(utterance_id=uuid4(), text="translated"),
        )
        harness.self_owner.merge_buffer = buffer
        called: list[str] = []

        async def fake_commit(self, commit_buffer, *, reason: str):  # noqa: ANN001
            _ = (self, commit_buffer)
            called.append(reason)

        monkeypatch.setattr(SelfTranslationChannelOwner, "_commit_merge", fake_commit)

        await harness.self_owner._try_commit_after_spec(
            buffer, reason="spec_done", allow_fallback=False
        )
        assert called == ["spec_done"]

    @pytest.mark.asyncio
    async def test_try_commit_after_spec_skips_when_spec_text_differs(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        harness = compose_translation_test_harness(
            stt=None,
            llm=FakeLLMProvider(),
            osc=FakeOscQueue(),
            clock=FakeClock(initial_time=10.0),
            low_latency_mode=True,
        )
        buffer = _MergeBuffer(
            merge_id=uuid4(),
            parts=["new"],
            spec_text="old",
            spec_translation=Translation(utterance_id=uuid4(), text="translated"),
        )
        harness.self_owner.merge_buffer = buffer
        called: list[str] = []

        async def fake_commit(self, commit_buffer, *, reason: str):  # noqa: ANN001
            _ = (self, commit_buffer)
            called.append(reason)

        monkeypatch.setattr(SelfTranslationChannelOwner, "_commit_merge", fake_commit)

        await harness.self_owner._try_commit_after_spec(
            buffer, reason="spec_done", allow_fallback=False
        )
        assert called == []

    @pytest.mark.asyncio
    async def test_try_commit_after_spec_skips_when_only_excluded_punctuation_differs(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        harness = compose_translation_test_harness(
            stt=None,
            llm=FakeLLMProvider(),
            osc=FakeOscQueue(),
            clock=FakeClock(initial_time=10.0),
            low_latency_mode=True,
        )
        buffer = _MergeBuffer(
            merge_id=uuid4(),
            parts=["hello?"],
            spec_text="hello",
            spec_translation=Translation(utterance_id=uuid4(), text="translated"),
        )
        harness.self_owner.merge_buffer = buffer
        called: list[str] = []

        async def fake_commit(self, commit_buffer, *, reason: str):  # noqa: ANN001
            _ = (self, commit_buffer)
            called.append(reason)

        monkeypatch.setattr(SelfTranslationChannelOwner, "_commit_merge", fake_commit)

        await harness.self_owner._try_commit_after_spec(
            buffer, reason="spec_done", allow_fallback=False
        )
        assert called == []

    @pytest.mark.asyncio
    async def test_commit_merge_retranslates_when_only_excluded_punctuation_differs(self):
        clock = FakeClock(initial_time=10.0)
        llm = FakeLLMProvider(response_text="nuevo")
        osc = FakeOscQueue()
        harness = compose_translation_test_harness(
            stt=None,
            llm=llm,
            osc=osc,
            clock=clock,
            low_latency_mode=True,
        )
        uid = uuid4()
        merge_id = uuid4()
        buffer = _MergeBuffer(
            merge_id=merge_id,
            parts=["hello?"],
            utterance_ids=[uid],
            start_time=clock.now(),
            last_end_time=clock.now(),
            spec_text="hello",
            spec_translation=Translation(utterance_id=merge_id, text="hola"),
        )
        harness.self_owner.merge_buffer = buffer
        harness.self_runtime.utterance_start_times[uid] = clock.now()

        await harness.self_owner._commit_merge(buffer, reason="spec_done")

        assert harness.self_owner.merge_buffer is None
        assert len(llm.calls) == 1
        assert llm.calls[0]["text"] == "hello?"
        assert len(osc.messages) == 1
        assert osc.messages[0].text == "hello? (nuevo)"

    @pytest.mark.asyncio
    async def test_commit_merge_blocks_while_resume_or_waiting_states(self):
        harness = compose_translation_test_harness(
            stt=None,
            llm=None,
            osc=FakeOscQueue(),
            clock=FakeClock(initial_time=10.0),
            low_latency_mode=True,
        )
        buffer = _MergeBuffer(merge_id=uuid4(), parts=["text"], utterance_ids=[uuid4()])
        harness.self_owner.merge_buffer = buffer

        buffer.resume_pending = True
        await harness.self_owner._commit_merge(buffer, reason="blocked_resume")
        assert harness.self_owner.merge_buffer is buffer

        buffer.resume_pending = False
        buffer.awaiting_vad_end = True
        await harness.self_owner._commit_merge(buffer, reason="blocked_waiting")
        assert harness.self_owner.merge_buffer is buffer

        buffer.awaiting_vad_end = False
        buffer.finalize_wait_task = asyncio.create_task(asyncio.sleep(0.1))
        await harness.self_owner._commit_merge(buffer, reason="blocked_grace")
        assert harness.self_owner.merge_buffer is buffer
        buffer.finalize_wait_task.cancel()
        await asyncio.gather(buffer.finalize_wait_task, return_exceptions=True)
