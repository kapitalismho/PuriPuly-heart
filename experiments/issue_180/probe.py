from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast
from uuid import UUID, uuid5

import numpy as np

from puripuly_heart.app.adapters.self_capture.self_capture_vad_sink import (
    SelfCaptureVadSinkAdapter,
)
from puripuly_heart.config.overlay_calibration import OverlayCalibration
from puripuly_heart.core.audio.format import AudioCaptureSpan
from puripuly_heart.core.audio.ownership import (
    SELF_RETAINED_AUDIO_CAPACITY_BYTES,
    SELF_RETAINED_AUDIO_CAPACITY_SAMPLE_EQUIVALENTS,
    AudioRetentionBudget,
    AudioSegmentSettingsSnapshot,
    OwnedVadEvent,
    PeerAudioSegmentLedger,
)
from puripuly_heart.core.clock import Clock, FakeClock
from puripuly_heart.core.orchestrator.translation_channel_callbacks import (
    TranslationChannelOwnerCallbacks,
)
from puripuly_heart.core.overlay.presenter import OverlayPresenter
from puripuly_heart.core.overlay.sink import OverlayEventAdapter, OverlayEventUnion
from puripuly_heart.core.runtime.output import OutputRuntime
from puripuly_heart.core.runtime.output_batch import (
    DestinationBatch,
    DestinationBatchAdmission,
)
from puripuly_heart.core.runtime.peer_channel import (
    _CaptureGeneration,
    _GenerationGuardedVadSink,
)
from puripuly_heart.core.runtime.provider_handle import ProviderRuntimeHandle
from puripuly_heart.core.runtime.self_capture import (
    _CaptureGeneration as _SelfCaptureGeneration,
)
from puripuly_heart.core.runtime.self_capture import (
    _GenerationGuardedVadSink as _SelfGenerationGuardedVadSink,
)
from puripuly_heart.core.runtime.stt_session_projection import SttSessionStateProjection
from puripuly_heart.core.stt.backend import (
    STTProviderTurnEvent,
    STTProviderTurnIdentity,
    STTProviderTurnRequest,
    STTProviderTurnTerminal,
)
from puripuly_heart.core.stt.scoped_engine import ScopedRecognitionEngine, STTRecognitionWatchdogs
from puripuly_heart.core.stt.scoped_event_buffer import STTProviderEventBuffer
from puripuly_heart.core.vad.gating import SpeechEnd, SpeechStart
from puripuly_heart.domain.models import OSCMessage, Transcript, Translation

EXPECTED_SHA = "13274569769d3c1ec7a896a2d15b919b76136a6e"
APPROVED_IMPLEMENTATION_BASELINE = "9c630d3cb1528d63b3cb6e06ddf4c7c1fe012832"
APPROVED_IMPLEMENTATION_PATHS = frozenset(
    {
        "docs/architecture.md",
        "experiments/issue_180/probe.py",
        "experiments/issue_180/report.md",
        "experiments/issue_180/trace_after.jsonl",
        "src/puripuly_heart/core/runtime/output_batch.py",
        "src/puripuly_heart/core/stt/scoped_engine.py",
        "src/puripuly_heart/core/stt/session_projection.py",
        "src/puripuly_heart/providers/stt/local_gpu.py",
        "src/puripuly_heart/providers/stt/local_qwen_sherpa.py",
        "src/puripuly_heart/providers/stt/local_decode.py",
        "tests/core/runtime/test_output_runtime.py",
        "tests/core/test_stt_scoped_engine.py",
        "tests/core/test_stt_session_projection.py",
        "tests/providers/test_local_cpu_backends.py",
    }
)
NAMESPACE = UUID("79ba997d-9247-4a88-b850-a24db11de180")


@dataclass(slots=True)
class Trace:
    scenario: str
    clock: Clock
    rows: list[dict[str, Any]] = field(default_factory=list)

    def add(self, event: str, **data: Any) -> None:
        now = self.clock.now()
        self.rows.append(
            {
                "scenario": self.scenario,
                "t_ms": round(now * 1000),
                "t_us": round(now * 1_000_000),
                "event": event,
                **data,
            }
        )


@dataclass(slots=True)
class RelativeRealClock:
    origin: float = field(default_factory=time.perf_counter)

    def now(self) -> float:
        return time.perf_counter() - self.origin


class DeterministicScopedSession:
    def __init__(self, trace: Trace, *, allows_sealed_turn_overlap: bool = False) -> None:
        self.trace = trace
        self.buffer = STTProviderEventBuffer()
        self.requests: list[STTProviderTurnRequest] = []
        self.sealed: dict[int, asyncio.Event] = {1: asyncio.Event(), 2: asyncio.Event()}
        self.identities: dict[int, STTProviderTurnIdentity] = {}
        self.allows_sealed_turn_overlap = allows_sealed_turn_overlap

    async def begin_turn(self, request: STTProviderTurnRequest) -> None:
        order = request.identity.segment.segment_order
        self.requests.append(request)
        self.identities[order] = request.identity
        self.trace.add(
            "provider_begin", turn=order, provider_turn_id=request.identity.provider_turn_id
        )

    async def send_turn_audio(
        self,
        identity: STTProviderTurnIdentity,
        pcm16le: bytes,
        *,
        payload_sequence: int,
        source_ranges: tuple[AudioCaptureSpan, ...],
        context_only: bool,
    ) -> None:
        self.trace.add(
            "provider_write",
            turn=identity.segment.segment_order,
            provider_turn_id=identity.provider_turn_id,
            payload_sequence=payload_sequence,
            samples=len(pcm16le) // 2,
            context_only=context_only,
            source_start_ms=round(
                min(item.source_start_monotonic_s for item in source_ranges) * 1000
            ),
            source_end_ms=round(max(item.source_end_monotonic_s for item in source_ranges) * 1000),
        )

    async def seal_turn(
        self,
        identity: STTProviderTurnIdentity,
        *,
        sealed_content_ranges: tuple[AudioCaptureSpan, ...],
        seal_reason: str,
        observed_trailing_silence_ms: int | None,
    ) -> None:
        order = identity.segment.segment_order
        self.trace.add(
            "provider_seal",
            turn=order,
            content_samples=sum(item.normalized_sample_count for item in sealed_content_ranges),
            seal_reason=seal_reason,
        )
        self.sealed[order].set()

    async def abort_turn(self, identity: STTProviderTurnIdentity, *, reason: str) -> None:
        self.trace.add("provider_abort", turn=identity.segment.segment_order, reason=reason)

    async def turn_events(self):
        async for event in self.buffer.events():
            yield event

    async def stop(self) -> None:
        return None

    async def close(self) -> None:
        self.buffer.close()

    def terminal(self, order: int) -> None:
        identity = self.identities[order]
        self.trace.add(
            "provider_terminal_receipt", turn=order, provider_turn_id=identity.provider_turn_id
        )
        assert self.buffer.put(
            STTProviderTurnTerminal(
                identity=identity,
                outcome="final",
                text=f"turn-{order}",
                text_authority="authoritative",
            )
        )


@dataclass(slots=True)
class ProbeCaptureRuntime:
    generation: int = 1
    retired: list[tuple[str, str]] = field(default_factory=list)

    def is_current_generation(self, value: int) -> bool:
        return value == self.generation

    def record_segment_terminal(
        self, segment_id: UUID, *, outcome: str, text_authority: str, failure_reason: str
    ) -> None:
        self.retired.append((str(segment_id), failure_reason))


class TracedEngineSink:
    def __init__(self, engine: ScopedRecognitionEngine, trace: Trace) -> None:
        self.engine = engine
        self.trace = trace

    async def handle_owned_vad_event(self, owned: OwnedVadEvent) -> None:
        order = owned.segment.identity.segment_order
        kind = type(owned.event).__name__
        source_at = (
            owned.segment.opened_at_monotonic_s
            if kind == "SpeechStart"
            else owned.segment.sealed_at_monotonic_s
        )
        self.trace.add(
            "dispatch_attempt",
            turn=order,
            kind=kind,
            source_age_ms=(
                None if source_at is None else round((self.trace.clock.now() - source_at) * 1000)
            ),
        )
        await self.engine.handle_owned_vad_event(owned)
        self.trace.add(
            "end_handler_return" if kind == "SpeechEnd" else "dispatch_return",
            turn=order,
            kind=kind,
        )

    async def observe_pending_source_work(self, *, pending: bool) -> None:
        self.trace.add("upstream_pending", pending=pending)
        await self.engine.observe_pending_source_work(pending=pending)


async def spin_until(predicate, label: str) -> None:
    for _ in range(10_000):
        if predicate():
            return
        await asyncio.sleep(0)
    raise RuntimeError(f"probe stalled waiting for {label}")


def stt_settings() -> AudioSegmentSettingsSnapshot:
    return AudioSegmentSettingsSnapshot(
        provider_id="local_qwen",
        provider_signature=("issue-180",),
        runtime_signature=("issue-180",),
        source_mode="desktop",
        source_language="en",
        expected_languages=("en",),
        target_sample_rate_hz=16_000,
        vad_speech_threshold=0.4,
        vad_hangover_ms=800,
        vad_pre_roll_ms=500,
    )


def capture_span(
    sequence: int, start_sample: int, end_sample: int, start_ms: int, end_ms: int
) -> AudioCaptureSpan:
    return AudioCaptureSpan(
        capture_epoch=1,
        callback_sequence=sequence,
        source_sample_rate_hz=16_000,
        source_start_sample=start_sample,
        source_end_sample=end_sample,
        source_start_monotonic_s=start_ms / 1000,
        source_end_monotonic_s=end_ms / 1000,
        normalized_sample_rate_hz=16_000,
        normalized_start_sample=start_sample,
        normalized_end_sample=end_sample,
    )


def stt_turn(
    ledger: PeerAudioSegmentLedger, *, order: int, start_ms: int, end_ms: int
) -> tuple[OwnedVadEvent, OwnedVadEvent]:
    turn_id = uuid5(NAMESPACE, f"stt-{order}")
    base = order * 100
    pre = capture_span(base, base, base + 2, start_ms - 10, start_ms)
    content = capture_span(base + 1, base + 2, base + 10, start_ms, end_ms)
    start = ledger.observe_vad_event(
        SpeechStart(
            turn_id,
            np.array([0.1, 0.1], dtype=np.float32),
            np.array([0.2] * 8, dtype=np.float32),
            pre_roll_capture=(pre,),
            chunk_capture=(content,),
        ),
        now_monotonic_s=start_ms / 1000,
    )
    end = ledger.observe_vad_event(
        SpeechEnd(turn_id, trailing_silence_ms=224, reason="silence"),
        now_monotonic_s=end_ms / 1000,
    )
    return start, end


async def run_stt_case(
    delayed: bool,
    *,
    allows_sealed_turn_overlap: bool = False,
) -> list[dict[str, Any]]:
    clock = FakeClock(_now=0.0)
    scenario = "stt_delayed_terminal" if delayed else "stt_immediate_terminal"
    if allows_sealed_turn_overlap:
        scenario = f"{scenario}_overlap"
    trace = Trace(scenario, clock)
    session = DeterministicScopedSession(
        trace,
        allows_sealed_turn_overlap=allows_sealed_turn_overlap,
    )

    async def on_engine_event(event: STTProviderTurnEvent) -> None:
        if isinstance(event, STTProviderTurnTerminal):
            trace.add(
                "engine_turn_release",
                turn=event.identity.segment.segment_order,
                outcome=event.outcome,
            )

    engine = ScopedRecognitionEngine(
        session_factory=lambda _settings, _epoch: asyncio.sleep(0, result=session),
        channel="peer",
        watchdog_resolver=lambda _settings: STTRecognitionWatchdogs(
            readiness_timeout_s=1,
            write_timeout_s=1,
            final_timeout_s=1,
            drain_timeout_s=1,
            healthy_reset_age_s=180,
            connect_attempts=3,
            connect_retry_base_s=0.001,
            connect_retry_max_s=0.002,
        ),
        monotonic_clock=clock.now,
    )
    provider_handle = ProviderRuntimeHandle(
        name=f"issue_180_{trace.scenario}",
        provider=engine,
        event_handler=on_engine_event,
    )
    await provider_handle.start()
    guarded = _GenerationGuardedVadSink(
        sink=TracedEngineSink(engine, trace),
        runtime=cast(Any, ProbeCaptureRuntime()),
        capture_generation=_CaptureGeneration(1),
        provider_ingress_ready=asyncio.Event(),
    )
    guarded.provider_ingress_ready.set()
    ledger = PeerAudioSegmentLedger(activation_generation=1, settings=stt_settings())
    a_start, a_end = stt_turn(ledger, order=1, start_ms=0, end_ms=100)
    b_start, b_end = stt_turn(ledger, order=2, start_ms=150, end_ms=300)

    trace.add("source_available", turn=1, kind="SpeechStart", buffered_content_samples=8)
    await guarded.handle_owned_vad_event(a_start)
    await spin_until(lambda: 1 in session.identities, "A begin")
    await spin_until(
        lambda: any(
            row["event"] == "provider_write" and row.get("turn") == 1 for row in trace.rows
        ),
        "A first write",
    )
    clock.advance(0.100)
    trace.add("local_seal", turn=1)
    await guarded.handle_owned_vad_event(a_end)
    await session.sealed[1].wait()
    if not delayed:
        session.terminal(1)
        await spin_until(lambda: engine.is_at_turn_boundary, "A immediate release")

    clock.advance(0.050)
    trace.add("source_available", turn=2, kind="SpeechStart", buffered_content_samples=8)
    await guarded.handle_owned_vad_event(b_start)
    if not delayed or allows_sealed_turn_overlap:
        await spin_until(lambda: 2 in session.identities, "B begin")
        await spin_until(
            lambda: any(
                row["event"] == "provider_write" and row.get("turn") == 2 for row in trace.rows
            ),
            "B first write",
        )
    clock.advance(0.150)
    trace.add("local_seal", turn=2)
    await guarded.handle_owned_vad_event(b_end)
    if delayed:
        clock.advance(0.050)
        session.terminal(1)
    await session.sealed[2].wait()
    clock.advance(0.050)
    session.terminal(2)
    await spin_until(lambda: engine.is_at_turn_boundary, "B release")
    await guarded.finish()
    await provider_handle.close()
    return trace.rows


@dataclass(slots=True)
class NullChatbox:
    def enqueue(self, message: OSCMessage) -> OSCMessage | None:
        return message

    def send_immediate(self, text: str) -> bool:
        return True

    def send_typing(self, is_typing: bool) -> None:
        return None

    def set_typing_reason(self, reason: str, active: bool) -> None:
        return None

    def clear_typing_reasons(self) -> None:
        return None

    def process_due(self) -> None:
        return None

    def drop_pending(self) -> None:
        return None


class TracingPresenter(OverlayPresenter):
    def __init__(self, *args: Any, probe_trace: Trace, **kwargs: Any) -> None:
        self.probe_trace = probe_trace
        super().__init__(*args, **kwargs)

    def _peer_replacement_delay(self, event: OverlayEventUnion) -> float | None:
        delay = super()._peer_replacement_delay(event)
        selected = [block.id for block in self.snapshot().blocks]
        self.probe_trace.add(
            "presenter_gate",
            publication_id=event.event_id,
            event_type=event.EVENT_TYPE,
            occupant=str(event.utterance_id),
            selected=selected,
            gate=(
                "protected_rows"
                if delay is None
                else ("eligible" if delay <= 0 else "replacement_gate")
            ),
            remaining_ms=None if delay is None else round(delay * 1000),
        )
        return delay

    async def _emit_serialized(self, event: OverlayEventUnion) -> None:
        self.probe_trace.add(
            "presenter_apply_begin",
            publication_id=event.event_id,
            event_type=event.EVENT_TYPE,
            occupant=str(event.utterance_id),
        )
        await super()._emit_serialized(event)
        self.probe_trace.add(
            "application_receipt_ready",
            publication_id=event.event_id,
            event_type=event.EVENT_TYPE,
            occupant=str(event.utterance_id),
            scene_revision=self.snapshot().revision,
        )


async def make_output(
    scenario: str, slots: int
) -> tuple[Trace, FakeClock, TracingPresenter, OutputRuntime, OverlayEventAdapter]:
    clock = FakeClock(_now=0.0)
    trace = Trace(scenario, clock)

    async def controlled_sleep(delay: float) -> None:
        if delay <= 1.000001:
            trace.add("presenter_sleep", delay_ms=round(delay * 1000))
            clock.advance(delay)
            await asyncio.sleep(0)
            return
        await asyncio.Event().wait()

    presenter = TracingPresenter(
        calibration=OverlayCalibration(),
        clock=clock,
        sleep=controlled_sleep,
        translation_enabled=False,
        visible_window_target_blocks=slots,
        probe_trace=trace,
    )

    def observe(decision: Any) -> None:
        trace.add(
            "output_decision",
            publication_id=decision.publication_id,
            reason=decision.reason,
            status=decision.decision,
            metadata=dict(decision.metadata),
        )

    output = OutputRuntime(
        chatbox=NullChatbox(),
        clock=clock,
        overlay_sink=presenter,
        routing_observer=observe,
    )
    output.activate_peer_generation(1)
    return trace, clock, presenter, output, OverlayEventAdapter(clock=clock)


def peer_final(adapter: OverlayEventAdapter, clock: Clock, index: int, *, text: str | None = None):
    turn = uuid5(NAMESPACE, f"peer-{index}")
    return adapter.transcript_final(
        Transcript(
            utterance_id=turn,
            channel="peer",
            text=text or f"peer {index}",
            is_final=True,
            created_at=clock.now(),
        ),
        source_language="en",
        target_language="ja",
    )


async def publish_peer(output: OutputRuntime, event: OverlayEventUnion, order: int):
    return await output.publish_overlay_event(event, publication_generation=1, source_order=order)


async def run_output_control() -> list[dict[str, Any]]:
    trace, clock, presenter, output, adapter = await make_output("output_replacement_control", 2)
    first = peer_final(adapter, clock, 1)
    second = peer_final(adapter, clock, 2)
    await publish_peer(output, first, 1)
    await publish_peer(output, second, 2)
    await output.wait_for_peer_output_idle()
    update = peer_final(adapter, clock, 2, text="peer 2 updated")
    await publish_peer(output, update, 2)
    close = adapter.utterance_closed(
        utterance_id=cast(UUID, second.utterance_id), channel="peer", created_at=clock.now()
    )
    await publish_peer(output, close, 2)
    await output.wait_for_peer_output_idle()
    trace.add("control_complete", visible=[block.id for block in presenter.snapshot().blocks])
    await output.close()
    await presenter.close()
    return trace.rows


async def run_output_burst() -> list[dict[str, Any]]:
    trace, clock, presenter, output, adapter = await make_output("output_burst", 2)
    events = [peer_final(adapter, clock, index) for index in range(1, 6)]
    for order, event in enumerate(events, 1):
        result = await publish_peer(output, event, order)
        trace.add(
            "translation_ready_and_enqueued",
            publication_id=event.event_id,
            occupant=str(event.utterance_id),
            source_order=order,
            queue_result=result.decision.reason,
        )
    await output.wait_for_peer_output_idle()
    trace.add("burst_complete", visible=[block.id for block in presenter.snapshot().blocks])
    await output.close()
    await presenter.close()
    return trace.rows


async def run_output_protection() -> list[dict[str, Any]]:
    trace, clock, presenter, output, adapter = await make_output("output_protection_release", 1)
    self_turn = uuid5(NAMESPACE, "protected-self")
    seed = adapter.self_active_update(
        text="protected self",
        utterance_id=self_turn,
        occupant_key=f"self:{self_turn}",
        created_at=clock.now(),
    )
    await output.publish_overlay_event(seed)
    peer = peer_final(adapter, clock, 20)
    await publish_peer(output, peer, 1)
    await spin_until(
        lambda: any(
            row["event"] == "presenter_gate" and row.get("gate") == "protected_rows"
            for row in trace.rows
        ),
        "protected wait",
    )
    clock.advance(0.400)
    clear = adapter.self_active_clear(created_at=clock.now())
    clear_result = await output.publish_overlay_event(clear)
    trace.add(
        "protection_release_event",
        publication_id=clear.event_id,
        event_type=clear.EVENT_TYPE,
        result=clear_result.decision.reason,
    )
    await output.wait_for_peer_output_idle()
    await output.close()
    await presenter.close()
    return trace.rows


async def run_output_head_of_line() -> list[dict[str, Any]]:
    clock = FakeClock(_now=0.0)
    trace = Trace("output_head_of_line", clock)
    replacement_release = asyncio.Event()

    async def held_sleep(delay: float) -> None:
        if delay <= 1.000001:
            trace.add("presenter_sleep_held", delay_ms=round(delay * 1000))
            await replacement_release.wait()
            clock.advance(delay)
            return
        await asyncio.Event().wait()

    presenter = TracingPresenter(
        calibration=OverlayCalibration(),
        clock=clock,
        sleep=held_sleep,
        translation_enabled=False,
        visible_window_target_blocks=2,
        probe_trace=trace,
    )

    def observe(decision: Any) -> None:
        trace.add(
            "output_decision",
            publication_id=decision.publication_id,
            reason=decision.reason,
            status=decision.decision,
            metadata=dict(decision.metadata),
        )

    output = OutputRuntime(
        chatbox=NullChatbox(),
        clock=clock,
        overlay_sink=presenter,
        routing_observer=observe,
    )
    output.activate_peer_generation(1)
    adapter = OverlayEventAdapter(clock=clock)
    first = peer_final(adapter, clock, 1)
    visible = peer_final(adapter, clock, 2)
    await publish_peer(output, first, 1)
    await publish_peer(output, visible, 2)
    await output.wait_for_peer_output_idle()

    paced_head = peer_final(adapter, clock, 3)
    await publish_peer(output, paced_head, 3)
    await spin_until(
        lambda: any(row["event"] == "presenter_sleep_held" for row in trace.rows),
        "paced writer head",
    )

    visible_update = peer_final(adapter, clock, 2, text="visible update while head waits")
    update_result = await output.publish_overlay_event(visible_update)
    visible_close = adapter.utterance_closed(
        utterance_id=cast(UUID, visible.utterance_id),
        channel="peer",
        created_at=clock.now(),
    )
    close_result = await output.publish_overlay_event(visible_close)
    trace.add(
        "head_of_line_contract",
        paced_head=cast(UUID, paced_head.utterance_id).hex,
        visible_occupant=cast(UUID, visible.utterance_id).hex,
        remembered_source_order=2,
        latest_source_order=3,
        update_reason=update_result.decision.reason,
        close_reason=close_result.decision.reason,
    )
    replacement_release.set()
    await output.wait_for_peer_output_idle()
    await output.close()
    await presenter.close()
    return trace.rows


async def run_real_clock_pacing() -> list[dict[str, Any]]:
    clock = RelativeRealClock()
    trace = Trace("output_real_clock_pacing", clock)
    presenter = TracingPresenter(
        calibration=OverlayCalibration(),
        clock=clock,
        translation_enabled=False,
        visible_window_target_blocks=1,
        probe_trace=trace,
    )

    def observe(decision: Any) -> None:
        trace.add(
            "output_decision",
            publication_id=decision.publication_id,
            reason=decision.reason,
            status=decision.decision,
            metadata=dict(decision.metadata),
        )

    output = OutputRuntime(
        chatbox=NullChatbox(),
        clock=clock,
        overlay_sink=presenter,
        routing_observer=observe,
    )
    output.activate_peer_generation(1)
    adapter = OverlayEventAdapter(clock=clock)
    first = peer_final(adapter, clock, 31)
    second = peer_final(adapter, clock, 32)
    await publish_peer(output, first, 1)
    await output.wait_for_peer_output_idle()
    trace.add("real_clock_item_ready", publication_id=second.event_id)
    await publish_peer(output, second, 2)
    await output.wait_for_peer_output_idle()

    eligible = next(
        row
        for row in trace.rows
        if row["event"] == "presenter_gate"
        and row["publication_id"] == second.event_id
        and row["gate"] == "eligible"
    )
    applied = next(
        row
        for row in trace.rows
        if row["event"] == "application_receipt_ready" and row["publication_id"] == second.event_id
    )
    ready = next(
        row
        for row in trace.rows
        if row["event"] == "real_clock_item_ready" and row["publication_id"] == second.event_id
    )
    trace.add(
        "real_clock_bound",
        publication_id=second.event_id,
        ready_to_application_us=applied["t_us"] - ready["t_us"],
        eligibility_to_application_us=applied["t_us"] - eligible["t_us"],
    )
    await output.close()
    await presenter.close()
    return trace.rows


@dataclass(slots=True)
class LoopClock:
    def now(self) -> float:
        return asyncio.get_running_loop().time()


@dataclass(slots=True)
class RelativeLoopTrace:
    scenario: str
    origin: float
    rows: list[dict[str, Any]] = field(default_factory=list)

    def add(self, event: str, **data: Any) -> None:
        now = asyncio.get_running_loop().time() - self.origin
        self.rows.append(
            {
                "scenario": self.scenario,
                "t_ms": round(now * 1000),
                "t_us": round(now * 1_000_000),
                "event": event,
                **data,
            }
        )


@dataclass(slots=True)
class ControlledTranslationProvider:
    trace: RelativeLoopTrace
    delays: dict[str, float]

    async def translate(
        self,
        *,
        utterance_id: UUID,
        text: str,
        system_prompt: str,
        source_language: str,
        target_language: str,
        context: str = "",
        scene_participant_count: int | None = None,
    ) -> Translation:
        _ = (system_prompt, context, scene_participant_count)
        self.trace.add(
            "translation_start",
            translation_id=str(utterance_id),
            source_text=text,
            source_language=source_language,
            target_language=target_language,
        )
        await asyncio.sleep(self.delays[text])
        self.trace.add(
            "translation_completion",
            translation_id=str(utterance_id),
            source_text=text,
        )
        return Translation(
            utterance_id=utterance_id,
            text=f"translated-{text}",
            source_text=text,
            source_language=source_language,
            target_language=target_language,
            channel="self",
        )

    async def close(self) -> None:
        return None


@dataclass(slots=True)
class SelfDispatcherOwner:
    generation: int = 1
    failures: list[str] = field(default_factory=list)

    def is_current_generation(self, value: int) -> bool:
        return value == self.generation

    def note_recognition_failure(self, reason: str) -> None:
        self.failures.append(reason)


@dataclass(slots=True)
class SelfTerminalObserver:
    trace: RelativeLoopTrace

    def note_recognition_terminal(self, event: STTProviderTurnTerminal) -> None:
        self.trace.add(
            "self_callback_terminal_noted",
            turn=event.identity.segment.segment_order,
            source_id=str(event.identity.segment.segment_id),
            provider_turn_id=event.identity.provider_turn_id,
            outcome=event.outcome,
        )


class TracingDestinationBatchAdmission(DestinationBatchAdmission):
    def __init__(self, trace: RelativeLoopTrace) -> None:
        super().__init__()
        self.trace = trace

    def _insert(self, batch: DestinationBatch) -> None:
        predecessor = self.active.get(batch.scope)
        super()._insert(batch)
        self.trace.add(
            "destination_batch_insert",
            scope=batch.scope,
            destination=batch.destination,
            parent_id=batch.parent_id,
            turn_generation=batch.turn_generation,
            turn_order=batch.turn_order,
            active=batch.active,
            active_parent_id=(predecessor[1] if predecessor is not None else batch.parent_id),
            waiting_parent_ids=[
                parent_id for _scope, parent_id in self.waiting.get(batch.scope, ())
            ],
        )

    def release(self, batch: DestinationBatch, disposition: str) -> None:
        released_parent_id = batch.parent_id
        scope = batch.scope
        was_active = self.active.get(scope) == (scope, released_parent_id)
        super().release(batch, disposition)
        successor = self.active.get(scope)
        self.trace.add(
            "destination_batch_release",
            scope=scope,
            destination=batch.destination,
            parent_id=released_parent_id,
            disposition=disposition,
            was_active=was_active,
            activated_successor_parent_id=(
                successor[1] if was_active and successor is not None else None
            ),
            waiting_parent_ids=[parent_id for _scope, parent_id in self.waiting.get(scope, ())],
        )


class SelfEngineRuntimeBridge:
    def __init__(self, engine: ScopedRecognitionEngine, trace: RelativeLoopTrace) -> None:
        self.engine = engine
        self.trace = trace

    async def handle_owned_vad_event(self, channel: str, owned: OwnedVadEvent) -> None:
        assert channel == "self"
        order = owned.segment.identity.segment_order
        kind = type(owned.event).__name__
        source_at = (
            owned.segment.opened_at_monotonic_s
            if kind == "SpeechStart"
            else owned.segment.sealed_at_monotonic_s
        )
        self.trace.add(
            "self_owner_to_engine_dispatch",
            source_id=str(owned.segment.identity.segment_id),
            turn=order,
            kind=kind,
            source_age_ms=(
                None
                if source_at is None
                else round((asyncio.get_running_loop().time() - source_at) * 1000)
            ),
        )
        await self.engine.handle_owned_vad_event(owned)
        self.trace.add(
            "self_owner_engine_return",
            source_id=str(owned.segment.identity.segment_id),
            turn=order,
            kind=kind,
        )

    async def handle_vad_event(self, channel: str, event: object) -> None:
        raise AssertionError(f"unowned Self event reached scoped probe: {channel}/{event!r}")

    async def commit_handoff(self, channel: str) -> None:
        assert channel == "self"
        self.trace.add("self_handoff_commit")

    async def observe_pending_source_work(self, channel: str, *, pending: bool) -> None:
        assert channel == "self"
        self.trace.add("self_pending_source_work", pending=pending)
        await self.engine.observe_pending_source_work(pending=pending)

    async def observe_source_activity(
        self,
        channel: str,
        *,
        speech_observed: bool,
        observed_at_monotonic_s: float,
    ) -> None:
        assert channel == "self"
        await self.engine.observe_source_activity(
            speech_observed=speech_observed,
            observed_at_monotonic_s=observed_at_monotonic_s,
        )

    async def reject_owned_segment(
        self,
        channel: str,
        event: OwnedVadEvent,
        *,
        reason: str,
        outcome: str,
    ) -> None:
        assert channel == "self"
        await self.engine.reject_owned_segment(event, reason=reason, outcome=cast(Any, outcome))

    async def fail_owned_segment(
        self,
        channel: str,
        event: OwnedVadEvent,
        *,
        reason: str,
    ) -> None:
        assert channel == "self"
        await self.engine.fail_owned_segment(event, reason=reason)


def self_speech_events(
    *,
    scenario: str,
    order: int,
    start_at: float,
    seal_at: float,
) -> tuple[SpeechStart, SpeechEnd, UUID]:
    source_id = uuid5(NAMESPACE, f"{scenario}:self:{order}")
    sample_base = order * 100
    pre = AudioCaptureSpan(
        capture_epoch=1,
        callback_sequence=sample_base,
        source_sample_rate_hz=16_000,
        source_start_sample=sample_base,
        source_end_sample=sample_base + 2,
        source_start_monotonic_s=start_at - 0.005,
        source_end_monotonic_s=start_at,
        normalized_sample_rate_hz=16_000,
        normalized_start_sample=sample_base,
        normalized_end_sample=sample_base + 2,
    )
    content = AudioCaptureSpan(
        capture_epoch=1,
        callback_sequence=sample_base + 1,
        source_sample_rate_hz=16_000,
        source_start_sample=sample_base + 2,
        source_end_sample=sample_base + 10,
        source_start_monotonic_s=start_at,
        source_end_monotonic_s=seal_at,
        normalized_sample_rate_hz=16_000,
        normalized_start_sample=sample_base + 2,
        normalized_end_sample=sample_base + 10,
    )
    return (
        SpeechStart(
            source_id,
            np.array([0.1, 0.1], dtype=np.float32),
            np.array([0.2] * 8, dtype=np.float32),
            pre_roll_capture=(pre,),
            chunk_capture=(content,),
        ),
        SpeechEnd(source_id, trailing_silence_ms=20, reason="silence"),
        source_id,
    )


async def run_self_end_to_end(
    *,
    scenario: str,
    delayed_a_terminal: bool,
    include_b: bool,
    a_translation_delay_s: float = 0.180,
) -> list[dict[str, Any]]:
    tests_path = str(Path(__file__).resolve().parents[2] / "tests")
    if tests_path not in sys.path:
        sys.path.insert(0, tests_path)
    from helpers.translation_owners import compose_translation_test_harness

    loop = asyncio.get_running_loop()
    origin = loop.time()
    trace = RelativeLoopTrace(scenario, origin)
    clock = LoopClock()
    translation = ControlledTranslationProvider(
        trace=trace,
        delays={"turn-1": a_translation_delay_s, "turn-2": 0.040},
    )
    presenter = TracingPresenter(
        calibration=OverlayCalibration(),
        clock=clock,
        translation_enabled=True,
        visible_window_target_blocks=2,
        probe_trace=trace,
    )
    harness = compose_translation_test_harness(
        osc=NullChatbox(),
        llm=translation,
        overlay_sink=presenter,
        clock=clock,
        source_language="en",
        target_language="ja",
        self_target_languages=("ja",),
        low_latency_mode=False,
    )
    harness.output_runtime._batch_admission = TracingDestinationBatchAdmission(trace)
    callbacks = TranslationChannelOwnerCallbacks(SttSessionStateProjection())
    callbacks.bind_self(harness.self_owner)
    callbacks.bind_self_capture(cast(Any, SelfTerminalObserver(trace)))

    def observe(decision: Any) -> None:
        if decision.route != "subtitle_overlay":
            return
        trace.add(
            "self_output_decision",
            publication_id=decision.publication_id,
            reason=decision.reason,
            status=decision.decision,
            metadata=dict(decision.metadata),
        )

    harness.output_runtime.routing_observer = observe
    session = DeterministicScopedSession(cast(Any, trace), allows_sealed_turn_overlap=True)

    async def on_engine_event(event: STTProviderTurnEvent) -> None:
        if isinstance(event, STTProviderTurnTerminal):
            order = event.identity.segment.segment_order
            trace.add(
                "engine_turn_release",
                turn=order,
                source_id=str(event.identity.segment.segment_id),
                provider_turn_id=event.identity.provider_turn_id,
                outcome=event.outcome,
            )
            await callbacks.self_event_handler(event)
            trace.add(
                "translation_admission_return",
                turn=order,
                source_id=str(event.identity.segment.segment_id),
                provider_turn_id=event.identity.provider_turn_id,
            )

    engine = ScopedRecognitionEngine(
        session_factory=lambda _settings, _epoch: asyncio.sleep(0, result=session),
        channel="self",
        watchdog_resolver=lambda _settings: STTRecognitionWatchdogs(
            readiness_timeout_s=1,
            write_timeout_s=1,
            final_timeout_s=1,
            drain_timeout_s=1,
            healthy_reset_age_s=180,
            connect_attempts=3,
            connect_retry_base_s=0.001,
            connect_retry_max_s=0.002,
        ),
        monotonic_clock=clock.now,
    )
    provider_handle = ProviderRuntimeHandle(
        name=f"issue_180_{scenario}",
        provider=engine,
        event_handler=on_engine_event,
    )
    await provider_handle.start()
    bridge = SelfEngineRuntimeBridge(engine, trace)
    harness.self_owner.local_asr_runtime = cast(Any, bridge)
    adapter = SelfCaptureVadSinkAdapter(runtime_provider=lambda: harness.self_owner)
    ledger = PeerAudioSegmentLedger(activation_generation=1, settings=stt_settings())
    dispatch_owner = SelfDispatcherOwner()
    guarded = _SelfGenerationGuardedVadSink(
        sink=adapter,
        owner=cast(Any, dispatch_owner),
        capture_generation=_SelfCaptureGeneration(1),
        ledger=ledger,
        retention_budget=AudioRetentionBudget(
            capacity_bytes=SELF_RETAINED_AUDIO_CAPACITY_BYTES,
            capacity_sample_equivalents=SELF_RETAINED_AUDIO_CAPACITY_SAMPLE_EQUIVALENTS,
        ),
    )
    await harness.start()

    async def sleep_until(offset_s: float) -> None:
        await asyncio.sleep(max(0.0, origin + offset_s - loop.time()))

    a_terminal_at = 0.160 if delayed_a_terminal else (0.070 if not include_b else 0.040)
    b_terminal_at = 0.200 if delayed_a_terminal else 0.150

    async def deliver_terminal(order: int, at_s: float) -> None:
        await session.sealed[order].wait()
        await sleep_until(at_s)
        session.terminal(order)

    terminals = [asyncio.create_task(deliver_terminal(1, a_terminal_at))]
    if include_b:
        terminals.append(asyncio.create_task(deliver_terminal(2, b_terminal_at)))

    a_start, a_end, a_id = self_speech_events(
        scenario=scenario,
        order=1,
        start_at=origin,
        seal_at=origin + 0.040,
    )
    trace.add(
        "self_source_available",
        turn=1,
        source_id=str(a_id),
        kind="SpeechStart",
        buffered_content_samples=8,
    )
    await guarded.handle_vad_event(a_start)
    await sleep_until(0.040)
    trace.add("acoustic_last_sample", turn=1, source_id=str(a_id), source_offset_ms=20)
    trace.add("local_seal", turn=1, source_id=str(a_id), endpoint_delay_ms=20)
    await guarded.handle_vad_event(a_end)

    if include_b:
        await sleep_until(0.070)
        b_start, b_end, b_id = self_speech_events(
            scenario=scenario,
            order=2,
            start_at=origin + 0.070,
            seal_at=origin + 0.110,
        )
        trace.add(
            "self_source_available",
            turn=2,
            source_id=str(b_id),
            kind="SpeechStart",
            buffered_content_samples=8,
        )
        await guarded.handle_vad_event(b_start)
        await sleep_until(0.110)
        trace.add("acoustic_last_sample", turn=2, source_id=str(b_id), source_offset_ms=90)
        trace.add("local_seal", turn=2, source_id=str(b_id), endpoint_delay_ms=20)
        await guarded.handle_vad_event(b_end)

    await asyncio.gather(*terminals)
    await guarded.finish()
    await harness.translation_turns.wait_for_idle()
    assert not dispatch_owner.failures
    await harness.stop()
    await provider_handle.close()
    await presenter.close()
    return trace.rows


def one_row(
    rows: list[dict[str, Any]],
    scenario: str,
    event: str,
    **matches: Any,
) -> dict[str, Any]:
    found = [
        row
        for row in rows
        if row["scenario"] == scenario
        and row["event"] == event
        and all(row.get(name) == value for name, value in matches.items())
    ]
    if len(found) != 1:
        raise AssertionError(f"expected one {scenario}/{event}/{matches}, found {len(found)}")
    return found[0]


def assert_probe_contract(rows: list[dict[str, Any]]) -> None:
    immediate_begin = one_row(rows, "stt_immediate_terminal", "provider_begin", turn=2)
    delayed_begin = one_row(rows, "stt_delayed_terminal", "provider_begin", turn=2)
    assert immediate_begin["t_ms"] == 150
    assert delayed_begin["t_ms"] == 350

    for scenario in ("stt_immediate_terminal", "stt_delayed_terminal"):
        writes = [
            (row["samples"], row["context_only"])
            for row in rows
            if row["scenario"] == scenario
            and row["event"] == "provider_write"
            and row.get("turn") == 2
        ]
        assert writes == [(2, True), (8, False)]

    burst_applied = [
        row["t_ms"]
        for row in rows
        if row["scenario"] == "output_burst" and row["event"] == "application_receipt_ready"
    ]
    overlap_begin = one_row(
        rows,
        "stt_delayed_terminal_overlap",
        "provider_begin",
        turn=2,
    )
    assert overlap_begin["t_ms"] == 150
    overlap_a_terminal = one_row(
        rows,
        "stt_delayed_terminal_overlap",
        "provider_terminal_receipt",
        turn=1,
    )
    assert overlap_begin["t_ms"] < overlap_a_terminal["t_ms"]
    overlap_writes = [
        (row["samples"], row["context_only"])
        for row in rows
        if row["scenario"] == "stt_delayed_terminal_overlap"
        and row["event"] == "provider_write"
        and row.get("turn") == 2
    ]
    assert overlap_writes == [(2, True), (8, False)]
    assert burst_applied == [0, 0, 1000, 2000, 3000]
    protection_applied = one_row(
        rows,
        "output_protection_release",
        "application_receipt_ready",
        event_type="peer_transcript_final",
    )
    assert protection_applied["t_ms"] == 400

    head_contract = one_row(rows, "output_head_of_line", "head_of_line_contract")
    assert head_contract["update_reason"] == "stale_source_order"
    assert head_contract["close_reason"] == "stale_source_order"

    real_bound = one_row(rows, "output_real_clock_pacing", "real_clock_bound")
    assert 900_000 <= real_bound["ready_to_application_us"] <= 1_500_000
    assert 0 <= real_bound["eligibility_to_application_us"] <= 100_000

    self_immediate = "self_successive_immediate_terminal"
    self_delayed = "self_successive_delayed_terminal"
    immediate_self_begin = one_row(rows, self_immediate, "provider_begin", turn=2)
    delayed_self_begin = one_row(rows, self_delayed, "provider_begin", turn=2)
    assert abs(delayed_self_begin["t_us"] - immediate_self_begin["t_us"]) <= 30_000
    immediate_self_release = one_row(rows, self_immediate, "engine_turn_release", turn=2)
    delayed_self_release = one_row(rows, self_delayed, "engine_turn_release", turn=2)
    assert delayed_self_release["t_us"] - immediate_self_release["t_us"] >= 20_000

    for scenario in (self_immediate, self_delayed):
        writes = [
            (row["samples"], row["context_only"])
            for row in rows
            if row["scenario"] == scenario
            and row["event"] == "provider_write"
            and row.get("turn") == 2
        ]
        b_source_id = one_row(rows, scenario, "self_source_available", turn=2)["source_id"]
        b_original = one_row(
            rows,
            scenario,
            "application_receipt_ready",
            event_type="self_transcript_final",
            occupant=b_source_id,
        )
        b_publication_id = b_original["publication_id"]
        b_original_batch = one_row(
            rows,
            scenario,
            "destination_batch_insert",
            parent_id=b_publication_id,
            active=True,
        )
        assert b_original_batch["scope"] == "self:original"
        assert b_original_batch["active_parent_id"] == b_publication_id
        one_row(
            rows,
            scenario,
            "destination_batch_release",
            parent_id=b_publication_id,
            disposition="applied",
            activated_successor_parent_id=None,
        )
        one_row(
            rows,
            scenario,
            "self_callback_terminal_noted",
            turn=2,
        )
        assert writes == [(2, True), (8, False)]
        assert (
            b_original["t_us"] - one_row(rows, scenario, "engine_turn_release", turn=2)["t_us"]
            <= 50_000
        )
        a_translation_done = one_row(rows, scenario, "translation_completion", source_text="turn-1")
        b_translation_start = one_row(rows, scenario, "translation_start", source_text="turn-2")
        assert b_translation_start["t_us"] >= a_translation_done["t_us"]

        original_applies = [
            row
            for row in rows
            if row["scenario"] == scenario
            and row["event"] == "application_receipt_ready"
            and row["event_type"] == "self_transcript_final"
        ]
        translated_applies = [
            row
            for row in rows
            if row["scenario"] == scenario
            and row["event"] == "application_receipt_ready"
            and row["event_type"] == "translation_final"
        ]
        assert len(original_applies) == 2
        assert len(translated_applies) == 2

    first_original = one_row(
        rows,
        "self_first_isolated",
        "application_receipt_ready",
        event_type="self_transcript_final",
    )
    first_translation_done = one_row(
        rows,
        "self_first_isolated",
        "translation_completion",
        source_text="turn-1",
    )
    first_translated = one_row(
        rows,
        "self_first_isolated",
        "application_receipt_ready",
        event_type="translation_final",
    )
    assert first_original["t_us"] < first_translation_done["t_us"] <= first_translated["t_us"]
    assert first_translated["t_us"] - first_translation_done["t_us"] <= 100_000

    fast_scenario = "self_successive_fast_translation"
    fast_b_release = one_row(rows, fast_scenario, "engine_turn_release", turn=2)
    fast_original_applies = [
        row
        for row in rows
        if row["scenario"] == fast_scenario
        and row["event"] == "application_receipt_ready"
        and row["event_type"] == "self_transcript_final"
    ]
    normal_b_release = one_row(rows, self_immediate, "engine_turn_release", turn=2)
    normal_original_applies = [
        row
        for row in rows
        if row["scenario"] == self_immediate
        and row["event"] == "application_receipt_ready"
        and row["event_type"] == "self_transcript_final"
    ]
    assert len(fast_original_applies) == len(normal_original_applies) == 2
    assert fast_original_applies[1]["t_us"] - fast_b_release["t_us"] <= 50_000
    assert normal_original_applies[1]["t_us"] - normal_b_release["t_us"] <= 50_000


def verify_source_revision() -> tuple[str, tuple[str, ...]]:
    actual_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    for required_ancestor in (EXPECTED_SHA, APPROVED_IMPLEMENTATION_BASELINE):
        ancestry = subprocess.run(
            ["git", "merge-base", "--is-ancestor", required_ancestor, actual_sha],
            check=False,
        )
        if ancestry.returncode != 0:
            raise RuntimeError(
                f"probe HEAD {actual_sha} does not descend from required baseline "
                f"{required_ancestor}"
            )
    tracked_changes = subprocess.check_output(
        ["git", "diff", "--name-only", APPROVED_IMPLEMENTATION_BASELINE, "--"], text=True
    ).splitlines()
    untracked_changes = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard"], text=True
    ).splitlines()
    changed = tuple(
        sorted(
            {
                line.strip().replace("\\", "/")
                for line in (*tracked_changes, *untracked_changes)
                if line.strip()
            }
        )
    )
    unexpected = [path for path in changed if path not in APPROVED_IMPLEMENTATION_PATHS]
    if unexpected:
        raise RuntimeError(
            "implementation probe differs outside its approved paths: " + ", ".join(unexpected)
        )
    return actual_sha, changed


def summarize(
    rows: list[dict[str, Any]],
    *,
    ambient_head: str,
    executed_probe_sha256: str,
    verified_changes: tuple[str, ...],
) -> dict[str, Any]:
    scenarios = sorted({row["scenario"] for row in rows})
    return {
        "ambient_head": ambient_head,
        "historical_production_baseline": EXPECTED_SHA,
        "approved_implementation_baseline": APPROVED_IMPLEMENTATION_BASELINE,
        "executed_probe_path": "experiments/issue_180/probe.py",
        "executed_probe_sha256": executed_probe_sha256,
        "artifact_content_note": (
            "SHA256 binds the executed harness and verified_changes records the explicit "
            "approved implementation delta; historical trace.jsonl remains a separate record"
        ),
        "verified_change_sha256": {
            path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for path in verified_changes
            if Path(path).is_file() and path != "experiments/issue_180/trace_after.jsonl"
        },
        "verified_changes": verified_changes,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "scenarios": scenarios,
        "row_count": len(rows),
    }


async def main(output_path: Path) -> None:
    ambient_head, verified_changes = verify_source_revision()
    executed_probe_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    rows = [
        *(
            await run_self_end_to_end(
                scenario="self_first_isolated",
                delayed_a_terminal=False,
                include_b=False,
            )
        ),
        *(
            await run_self_end_to_end(
                scenario="self_successive_immediate_terminal",
                delayed_a_terminal=False,
                include_b=True,
            )
        ),
        *(
            await run_self_end_to_end(
                scenario="self_successive_fast_translation",
                delayed_a_terminal=False,
                include_b=True,
                a_translation_delay_s=0.0,
            )
        ),
        *(
            await run_self_end_to_end(
                scenario="self_successive_delayed_terminal",
                delayed_a_terminal=True,
                include_b=True,
            )
        ),
        *(await run_stt_case(False)),
        *(await run_stt_case(True)),
        *(await run_stt_case(True, allows_sealed_turn_overlap=True)),
        *(await run_output_control()),
        *(await run_output_burst()),
        *(await run_output_protection()),
        *(await run_output_head_of_line()),
        *(await run_real_clock_pacing()),
    ]
    assert_probe_contract(rows)
    metadata = summarize(
        rows,
        ambient_head=ambient_head,
        executed_probe_sha256=executed_probe_sha256,
        verified_changes=verified_changes,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps({"type": "run_metadata", **metadata}, sort_keys=True) + "\n")
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Issue #180 focused Self/Peer software latency probe"
    )
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("trace.jsonl"))
    args = parser.parse_args()
    asyncio.run(main(args.output))
