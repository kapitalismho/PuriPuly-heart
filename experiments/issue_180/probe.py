from __future__ import annotations

import argparse
import asyncio
import json
import platform
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast
from uuid import UUID, uuid5

import numpy as np

from puripuly_heart.config.overlay_calibration import OverlayCalibration
from puripuly_heart.core.audio.format import AudioCaptureSpan
from puripuly_heart.core.audio.ownership import (
    AudioSegmentSettingsSnapshot,
    OwnedVadEvent,
    PeerAudioSegmentLedger,
)
from puripuly_heart.core.clock import Clock, FakeClock
from puripuly_heart.core.overlay.presenter import OverlayPresenter
from puripuly_heart.core.overlay.sink import OverlayEventAdapter, OverlayEventUnion
from puripuly_heart.core.runtime.output import OutputRuntime
from puripuly_heart.core.runtime.peer_channel import _CaptureGeneration, _GenerationGuardedVadSink
from puripuly_heart.core.stt.backend import (
    STTProviderTurnEvent,
    STTProviderTurnIdentity,
    STTProviderTurnRequest,
    STTProviderTurnTerminal,
)
from puripuly_heart.core.stt.scoped_engine import ScopedRecognitionEngine, STTRecognitionWatchdogs
from puripuly_heart.core.stt.scoped_event_buffer import STTProviderEventBuffer
from puripuly_heart.core.vad.gating import SpeechEnd, SpeechStart
from puripuly_heart.domain.models import OSCMessage, Transcript

EXPECTED_SHA = "13274569769d3c1ec7a896a2d15b919b76136a6e"
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
    def __init__(self, trace: Trace) -> None:
        self.trace = trace
        self.buffer = STTProviderEventBuffer()
        self.requests: list[STTProviderTurnRequest] = []
        self.sealed: dict[int, asyncio.Event] = {1: asyncio.Event(), 2: asyncio.Event()}
        self.identities: dict[int, STTProviderTurnIdentity] = {}

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


async def run_stt_case(delayed: bool) -> list[dict[str, Any]]:
    clock = FakeClock(_now=0.0)
    trace = Trace("stt_delayed_terminal" if delayed else "stt_immediate_terminal", clock)
    session = DeterministicScopedSession(trace)

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
        event_sink=on_engine_event,
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
    if not delayed:
        await spin_until(lambda: 2 in session.identities, "B immediate begin")
        await spin_until(
            lambda: any(
                row["event"] == "provider_write" and row.get("turn") == 2 for row in trace.rows
            ),
            "B immediate first write",
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
    await engine.close()
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


def verify_source_revision() -> tuple[str, tuple[str, ...]]:
    actual_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    ancestry = subprocess.run(
        ["git", "merge-base", "--is-ancestor", EXPECTED_SHA, actual_sha],
        check=False,
    )
    if ancestry.returncode != 0:
        raise RuntimeError(
            f"probe HEAD {actual_sha} does not descend from production baseline {EXPECTED_SHA}"
        )
    tracked_changes = subprocess.check_output(
        ["git", "diff", "--name-only", EXPECTED_SHA, "--"], text=True
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
    unexpected = [path for path in changed if not path.startswith("experiments/issue_180/")]
    if unexpected:
        raise RuntimeError(
            "production baseline differs outside the issue-180 experiment: " + ", ".join(unexpected)
        )
    return actual_sha, changed


def summarize(
    rows: list[dict[str, Any]],
    *,
    actual_sha: str,
    verified_changes: tuple[str, ...],
) -> dict[str, Any]:
    scenarios = sorted({row["scenario"] for row in rows})
    return {
        "actual_head": actual_sha,
        "verified_production_baseline": EXPECTED_SHA,
        "verified_changes": verified_changes,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "scenarios": scenarios,
        "row_count": len(rows),
    }


async def main(output_path: Path) -> None:
    actual_sha, verified_changes = verify_source_revision()
    rows = [
        *(await run_stt_case(False)),
        *(await run_stt_case(True)),
        *(await run_output_control()),
        *(await run_output_burst()),
        *(await run_output_protection()),
        *(await run_output_head_of_line()),
        *(await run_real_clock_pacing()),
    ]
    assert_probe_contract(rows)
    metadata = summarize(
        rows,
        actual_sha=actual_sha,
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
        description="Issue #180 focused STT handoff and Peer output timing probe"
    )
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("trace.jsonl"))
    args = parser.parse_args()
    asyncio.run(main(args.output))
