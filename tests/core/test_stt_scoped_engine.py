from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field, replace
from typing import Any, cast
from uuid import uuid4

import numpy as np
import pytest

from puripuly_heart.core.audio.format import AudioCaptureSpan
from puripuly_heart.core.audio.ownership import (
    AudioRetentionBinding,
    AudioRetentionBudget,
    AudioSegmentSettingsSnapshot,
    CaptureStreamInput,
    OwnedStreamInput,
    OwnedVadEvent,
    PeerAudioSegmentLedger,
)
from puripuly_heart.core.stt.backend import (
    STTContributionConsumptionLedger,
    STTNativeProvenance,
    STTProviderEpochEnded,
    STTProviderInputTerminal,
    STTProviderTurnEvent,
    STTProviderTurnIdentity,
    STTProviderTurnRequest,
    STTProviderTurnTerminal,
    STTProviderTurnUpdate,
    STTRecognitionUnit,
    STTRecognitionUnitTerminal,
)
from puripuly_heart.core.stt.diagnostics import recognition_cause
from puripuly_heart.core.stt.scoped_engine import (
    ScopedRecognitionEngine,
    STTRecognitionWatchdogs,
    STTRetentionProfile,
)
from puripuly_heart.core.stt.scoped_event_buffer import STTProviderEventBuffer
from puripuly_heart.core.stt.scoped_normalizer import (
    STTNormalizationError,
    STTScopedTurnNormalizer,
)
from puripuly_heart.core.vad.gating import SpeechChunk, SpeechEnd, SpeechStart
from puripuly_heart.domain.models import FinalLanguageRun
from puripuly_heart.domain.recognition import RecognitionStreamIdentity, RecognitionUnitIdentity


@dataclass(slots=True)
class ControlledScopedSession:
    buffer: STTProviderEventBuffer
    begin_gate: asyncio.Event
    send_gate: asyncio.Event
    requests: list[STTProviderTurnRequest]
    seal_gate: asyncio.Event
    calls: list[tuple[object, ...]]
    terminal_on_seal: tuple[str, str] | None = None
    allows_sealed_turn_overlap: bool = False

    def __init__(self) -> None:
        self.buffer = STTProviderEventBuffer()
        self.begin_gate = asyncio.Event()
        self.send_gate = asyncio.Event()
        self.seal_gate = asyncio.Event()
        self.begin_gate.set()
        self.send_gate.set()
        self.seal_gate.set()
        self.requests = []
        self.calls = []
        self.terminal_on_seal = None
        self.allows_sealed_turn_overlap = False

    async def begin_turn(self, request: STTProviderTurnRequest) -> None:
        self.requests.append(request)
        self.calls.append(("begin", request.identity))
        await self.begin_gate.wait()
        self.calls.append(("begin_done", request.identity))

    async def send_turn_audio(
        self,
        identity: STTProviderTurnIdentity,
        pcm16le: bytes,
        *,
        payload_sequence: int,
        source_ranges: tuple[AudioCaptureSpan, ...],
        context_only: bool,
    ) -> None:
        self.calls.append(
            ("send", identity, payload_sequence, len(pcm16le), source_ranges, context_only)
        )
        await self.send_gate.wait()
        self.calls.append(("send_done", identity, payload_sequence))

    async def seal_turn(
        self,
        identity: STTProviderTurnIdentity,
        *,
        sealed_content_ranges: tuple[AudioCaptureSpan, ...],
        seal_reason: str,
        observed_trailing_silence_ms: int | None,
    ) -> None:
        self.calls.append(
            (
                "seal",
                identity,
                sealed_content_ranges,
                seal_reason,
                observed_trailing_silence_ms,
            )
        )
        await self.seal_gate.wait()
        self.calls.append(("seal_done", identity))
        if self.terminal_on_seal is not None:
            outcome, text = self.terminal_on_seal
            self.buffer.put(
                STTProviderTurnTerminal(
                    identity=identity,
                    outcome=outcome,
                    text=text,
                    text_authority="authoritative",
                )
            )

    async def abort_turn(self, identity: STTProviderTurnIdentity, *, reason: str) -> None:
        self.calls.append(("abort", identity, reason))

    async def turn_events(self):
        async for event in self.buffer.events():
            yield event

    async def stop(self) -> None:
        self.calls.append(("stop",))

    async def close(self) -> None:
        self.calls.append(("close",))
        self.buffer.close()

    def emit(self, event: object) -> bool:
        return self.buffer.put(cast(STTProviderTurnEvent, event))


class BarrierCleanupScopedSession(ControlledScopedSession):
    __slots__ = ("stop_gate", "close_gate", "stop_cancellations", "close_cancellations")

    def __init__(self) -> None:
        super().__init__()
        self.stop_gate = asyncio.Event()
        self.close_gate = asyncio.Event()
        self.stop_cancellations = 0
        self.close_cancellations = 0

    async def stop(self) -> None:
        self.calls.append(("stop",))
        while not self.stop_gate.is_set():
            try:
                await self.stop_gate.wait()
            except asyncio.CancelledError:
                self.stop_cancellations += 1
                task = asyncio.current_task()
                if task is not None:
                    task.uncancel()
        self.calls.append(("stop_done",))

    async def close(self) -> None:
        self.calls.append(("close",))
        while not self.close_gate.is_set():
            try:
                await self.close_gate.wait()
            except asyncio.CancelledError:
                self.close_cancellations += 1
                task = asyncio.current_task()
                if task is not None:
                    task.uncancel()
        self.calls.append(("close_done",))
        self.buffer.close()


class InterruptibleCleanupScopedSession(ControlledScopedSession):
    __slots__ = ("stop_gate", "close_gate", "stop_cancellations", "close_cancellations")

    def __init__(self) -> None:
        super().__init__()
        self.stop_gate = asyncio.Event()
        self.close_gate = asyncio.Event()
        self.stop_cancellations = 0
        self.close_cancellations = 0

    async def stop(self) -> None:
        self.calls.append(("stop",))
        try:
            await self.stop_gate.wait()
        except asyncio.CancelledError:
            self.stop_cancellations += 1
            raise

    async def close(self) -> None:
        self.calls.append(("close",))
        try:
            await self.close_gate.wait()
        except asyncio.CancelledError:
            self.close_cancellations += 1
            raise


def settings(
    provider_id: str = "deepgram", *, signature: str = "a"
) -> AudioSegmentSettingsSnapshot:
    return AudioSegmentSettingsSnapshot(
        provider_id=provider_id,
        provider_signature=(signature,),
        runtime_signature=(signature,),
        source_mode="desktop",
        source_language="en",
        expected_languages=("en",),
        target_sample_rate_hz=16000,
        vad_speech_threshold=0.4,
        vad_hangover_ms=800,
        vad_pre_roll_ms=500,
    )


def span(sequence: int, start: int, end: int) -> AudioCaptureSpan:
    return AudioCaptureSpan(
        capture_epoch=1,
        callback_sequence=sequence,
        source_sample_rate_hz=16000,
        source_start_sample=start,
        source_end_sample=end,
        source_start_monotonic_s=start / 16000,
        source_end_monotonic_s=end / 16000,
        normalized_sample_rate_hz=16000,
        normalized_start_sample=start,
        normalized_end_sample=end,
    )


def segment_events(
    ledger: PeerAudioSegmentLedger,
    *,
    start_sample: int,
    now: float,
) -> tuple[OwnedVadEvent, OwnedVadEvent, OwnedVadEvent]:
    segment_id = uuid4()
    pre = span(start_sample, start_sample, start_sample + 2)
    first = span(start_sample + 1, start_sample + 2, start_sample + 6)
    second = span(start_sample + 2, start_sample + 6, start_sample + 10)
    start = ledger.observe_vad_event(
        SpeechStart(
            segment_id,
            np.array([0.1, 0.1], dtype=np.float32),
            np.array([0.2] * 4, dtype=np.float32),
            pre_roll_capture=(pre,),
            chunk_capture=(first,),
        ),
        now_monotonic_s=now,
    )
    chunk = ledger.observe_vad_event(
        SpeechChunk(
            segment_id,
            np.array([0.3] * 4, dtype=np.float32),
            chunk_capture=(second,),
        ),
        now_monotonic_s=now + 0.1,
    )
    end = ledger.observe_vad_event(
        SpeechEnd(segment_id, trailing_silence_ms=224, reason="silence"),
        now_monotonic_s=now + 0.2,
    )
    return start, chunk, end


async def wait_until(predicate, *, timeout: float = 1.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() >= deadline:
            raise AssertionError("timed out")
        await asyncio.sleep(0)


def watchdogs(**overrides: float) -> STTRecognitionWatchdogs:
    values: dict[str, Any] = {
        "readiness_timeout_s": 0.2,
        "write_timeout_s": 0.2,
        "final_timeout_s": 0.03,
        "drain_timeout_s": 0.03,
        "idle_timeout_s": 60.0,
        "connect_attempts": 3,
        "connect_retry_base_s": 0.001,
        "connect_retry_max_s": 0.002,
    }
    values.update(overrides)
    return STTRecognitionWatchdogs(**values)


@dataclass(slots=True)
class ControlledMonotonicClock:
    value: float = 0.0
    sleepers: list[tuple[float, asyncio.Future[None]]] = field(default_factory=list)

    def now(self) -> float:
        return self.value

    async def sleep(self, delay_s: float) -> None:
        if delay_s <= 0:
            await asyncio.sleep(0)
            return
        future = asyncio.get_running_loop().create_future()
        self.sleepers.append((self.value + delay_s, future))
        await future

    async def advance_to(self, value: float) -> None:
        self.value = value
        for deadline, future in tuple(self.sleepers):
            if deadline <= value and not future.done():
                future.set_result(None)
        await asyncio.sleep(0)
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_ordered_actual_writes_and_final_wait_starts_after_end_write() -> None:
    ledger = PeerAudioSegmentLedger(activation_generation=3, settings=settings())
    start, chunk, end = segment_events(ledger, start_sample=10, now=2.0)
    session = ControlledScopedSession()
    session.begin_gate.clear()
    session.send_gate.clear()
    session.seal_gate.clear()
    session.terminal_on_seal = ("final", "  repeated repeated  ")
    emitted: list[object] = []
    engine = ScopedRecognitionEngine(
        session_factory=lambda _settings, _epoch: asyncio.sleep(0, result=session),
        event_sink=lambda event: emitted.append(event),
        watchdog_resolver=lambda _settings: watchdogs(write_timeout_s=0.3),
    )

    start_task = asyncio.create_task(engine.handle_owned_vad_event(start))
    await wait_until(lambda: bool(session.calls))
    assert ledger.snapshots[0].state == "sealed"
    assert [call[0] for call in session.calls] == ["begin"]

    session.begin_gate.set()
    await wait_until(lambda: any(call[0] == "send" for call in session.calls))
    session.send_gate.set()
    await start_task
    await engine.handle_owned_vad_event(chunk)

    end_task = asyncio.create_task(engine.handle_owned_vad_event(end))
    await wait_until(lambda: any(call[0] == "seal" for call in session.calls))
    await asyncio.sleep(0.05)
    assert not any(isinstance(item, STTProviderTurnTerminal) for item in emitted)

    session.seal_gate.set()
    await end_task
    operation_order = [call[0] for call in session.calls if call[0] not in ("stop", "close")]
    assert operation_order == [
        "begin",
        "begin_done",
        "send",
        "send_done",
        "send",
        "send_done",
        "send",
        "send_done",
        "seal",
        "seal_done",
    ]
    terminal = next(item for item in emitted if isinstance(item, STTProviderTurnTerminal))
    assert terminal.text == "repeated repeated"
    assert terminal.final_language_runs == ()
    await engine.close()


@pytest.mark.asyncio
async def test_successor_start_waits_without_holding_ingress_lock_for_predecessor_terminal() -> (
    None
):
    ledger = PeerAudioSegmentLedger(activation_generation=1, settings=settings())
    a_start, _a_chunk, a_end = segment_events(ledger, start_sample=100, now=1.0)
    b_start, _b_chunk, b_end = segment_events(ledger, start_sample=200, now=2.0)
    session = ControlledScopedSession()
    engine = ScopedRecognitionEngine(
        session_factory=lambda _settings, _epoch: asyncio.sleep(0, result=session),
        event_sink=lambda _event: None,
        watchdog_resolver=lambda _settings: watchdogs(),
    )

    await engine.handle_owned_vad_event(a_start)
    a_identity = session.requests[0].identity
    a_end_task = asyncio.create_task(engine.handle_owned_vad_event(a_end))
    await wait_until(lambda: any(call[0] == "seal_done" for call in session.calls))
    b_start_task = asyncio.create_task(engine.handle_owned_vad_event(b_start))
    await asyncio.sleep(0)
    assert not b_start_task.done()
    assert len(session.requests) == 1

    session.emit(
        STTProviderTurnTerminal(
            identity=a_identity,
            outcome="empty",
            text_authority="authoritative",
            epoch_disposition="reuse",
        )
    )
    await a_end_task
    await b_start_task
    assert len(session.requests) == 2
    assert session.requests[1].identity.provider_epoch_id == a_identity.provider_epoch_id

    session.terminal_on_seal = ("empty", "")
    await engine.handle_owned_vad_event(b_end)
    await engine.close()


@pytest.mark.asyncio
async def test_sealed_overlap_admits_successor_and_orders_out_of_order_terminals_before_delivery() -> (
    None
):
    ledger = PeerAudioSegmentLedger(activation_generation=1, settings=settings("local_qwen"))
    a_start, _a_chunk, a_end = segment_events(ledger, start_sample=100, now=1.0)
    b_start, _b_chunk, b_end = segment_events(ledger, start_sample=200, now=2.0)
    session = ControlledScopedSession()
    session.allows_sealed_turn_overlap = True
    delivered: list[STTProviderTurnTerminal] = []
    delivery_gate = asyncio.Event()

    async def receive(event: object) -> None:
        if isinstance(event, STTProviderTurnTerminal):
            delivered.append(event)
            if len(delivered) == 1:
                await delivery_gate.wait()

    engine = ScopedRecognitionEngine(
        channel="self",
        session_factory=lambda _settings, _epoch: asyncio.sleep(0, result=session),
        watchdog_resolver=lambda _settings: watchdogs(),
    )
    engine.bind_event_sink(receive)

    await engine.handle_owned_vad_event(a_start)
    a_identity = session.requests[0].identity
    await engine.handle_owned_vad_event(a_end)
    await engine.handle_owned_vad_event(b_start)
    b_identity = session.requests[1].identity
    await engine.handle_owned_vad_event(b_end)

    session.emit(
        STTProviderTurnTerminal(
            identity=b_identity,
            outcome="final",
            text="b",
            text_authority="authoritative",
        )
    )
    await asyncio.sleep(0)
    assert delivered == []

    session.emit(
        STTProviderTurnTerminal(
            identity=a_identity,
            outcome="final",
            text="a",
            text_authority="authoritative",
        )
    )
    await wait_until(lambda: len(delivered) == 1)
    assert delivered[0].identity == a_identity
    assert engine.is_at_turn_boundary

    delivery_gate.set()
    await wait_until(lambda: len(delivered) == 2)
    assert [terminal.identity for terminal in delivered] == [a_identity, b_identity]
    await engine.close()


@pytest.mark.asyncio
async def test_successor_begin_failure_drains_predecessor_terminal_from_retiring_epoch() -> None:
    class FailingSuccessorSession(ControlledScopedSession):
        async def begin_turn(self, request: STTProviderTurnRequest) -> None:
            await super().begin_turn(request)
            if len(self.requests) == 2:
                raise RuntimeError("successor begin failed")

        async def stop(self) -> None:
            self.calls.append(("stop",))
            first = self.requests[0].identity
            self.emit(
                STTProviderTurnTerminal(
                    identity=first,
                    outcome="final",
                    text="a-final-text",
                    text_authority="authoritative",
                )
            )

    ledger = PeerAudioSegmentLedger(
        activation_generation=1,
        settings=settings("local_qwen"),
    )
    a_start, _a_chunk, a_end = segment_events(ledger, start_sample=100, now=1.0)
    b_start, _b_chunk, _b_end = segment_events(ledger, start_sample=200, now=2.0)
    session = FailingSuccessorSession()
    session.allows_sealed_turn_overlap = True
    emitted: list[STTProviderTurnTerminal] = []
    engine = ScopedRecognitionEngine(
        channel="self",
        session_factory=lambda _settings, _epoch: asyncio.sleep(0, result=session),
        event_sink=lambda event: (
            emitted.append(event) if isinstance(event, STTProviderTurnTerminal) else None
        ),
        watchdog_resolver=lambda _settings: watchdogs(final_timeout_s=0.1),
    )

    await engine.handle_owned_vad_event(a_start)
    await engine.handle_owned_vad_event(a_end)
    await engine.handle_owned_vad_event(b_start)
    await wait_until(lambda: len(emitted) == 2, timeout=0.5)

    assert [(item.outcome, item.text, item.failure_reason) for item in emitted] == [
        ("final", "a-final-text", None),
        ("failed", "", "provider_begin_failed:RuntimeError"),
    ]
    assert [item.identity for item in emitted] == [
        session.requests[0].identity,
        session.requests[1].identity,
    ]
    await engine.close()


@pytest.mark.asyncio
async def test_empty_a_late_a_and_duplicates_cannot_shift_b() -> None:
    ledger = PeerAudioSegmentLedger(activation_generation=1, settings=settings())
    a_start, _a_chunk, a_end = segment_events(ledger, start_sample=100, now=1.0)
    b_start, _b_chunk, b_end = segment_events(ledger, start_sample=200, now=2.0)
    session = ControlledScopedSession()
    emitted: list[object] = []
    engine = ScopedRecognitionEngine(
        session_factory=lambda _settings, _epoch: asyncio.sleep(0, result=session),
        event_sink=lambda event: emitted.append(event),
        watchdog_resolver=lambda _settings: watchdogs(),
    )

    await engine.handle_owned_vad_event(a_start)
    a_identity = session.calls[0][1]
    a_end_task = asyncio.create_task(engine.handle_owned_vad_event(a_end))
    await wait_until(lambda: any(call[0] == "seal" for call in session.calls))
    session.emit(
        STTProviderTurnTerminal(
            identity=a_identity,
            outcome="empty",
            text_authority="authoritative",
            provenance=(STTNativeProvenance(native_event_id="a-terminal"),),
        )
    )
    await a_end_task

    await engine.handle_owned_vad_event(b_start)
    b_identity = next(call[1] for call in reversed(session.calls) if call[0] == "begin")
    session.emit(
        STTProviderTurnTerminal(
            identity=a_identity,
            outcome="final",
            text="late-a",
            text_authority="authoritative",
            provenance=(STTNativeProvenance(native_event_id="late-a"),),
        )
    )
    session.emit(
        STTProviderTurnUpdate(
            identity=b_identity,
            sequence=1,
            stability="stable",
            assembly="append",
            text="echo",
            provenance=STTNativeProvenance(native_event_id="b-1"),
        )
    )
    session.emit(
        STTProviderTurnUpdate(
            identity=b_identity,
            sequence=2,
            stability="stable",
            assembly="append",
            text="echo",
            provenance=STTNativeProvenance(native_event_id="b-2"),
        )
    )
    b_end_task = asyncio.create_task(engine.handle_owned_vad_event(b_end))
    await wait_until(lambda: sum(call[0] == "seal" for call in session.calls) == 2)
    b_terminal = STTProviderTurnTerminal(
        identity=b_identity,
        outcome="final",
        text_authority="authoritative",
        provenance=(STTNativeProvenance(native_event_id="b-terminal"),),
    )
    session.emit(b_terminal)
    session.emit(b_terminal)
    await b_end_task
    terminals = [item for item in emitted if isinstance(item, STTProviderTurnTerminal)]
    assert [(item.identity, item.outcome, item.text) for item in terminals] == [
        (a_identity, "empty", ""),
        (b_identity, "final", "echoecho"),
    ]
    await engine.close()


@pytest.mark.asyncio
async def test_peer_recognition_evidence_counts_receipts_and_empty_terminal_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    ledger = PeerAudioSegmentLedger(activation_generation=1, settings=settings())
    start, chunk, end = segment_events(ledger, start_sample=300, now=1.0)
    session = ControlledScopedSession()
    session.send_gate.clear()
    engine = ScopedRecognitionEngine(
        session_factory=lambda _settings, _epoch: asyncio.sleep(0, result=session),
        watchdog_resolver=lambda _settings: watchdogs(write_timeout_s=1.0),
    )
    with caplog.at_level(logging.INFO, logger="puripuly_heart.core.stt.scoped_engine"):
        first = asyncio.create_task(engine.handle_owned_vad_event(start))
        await wait_until(lambda: any(call[0] == "send" for call in session.calls))
        assert not [
            record
            for record in caplog.records
            if record.getMessage().startswith("[Recognition] payload_received ")
        ]
        session.send_gate.set()
        await first
        await engine.handle_owned_vad_event(chunk)
        identity = session.requests[0].identity
        session.terminal_on_seal = ("empty", "")
        await engine.handle_owned_vad_event(end)
        session.emit(STTProviderTurnTerminal(identity=identity, outcome="final", text="late"))
        await asyncio.sleep(0)
        await engine.close()

    messages = [
        record.getMessage()
        for record in caplog.records
        if record.name == "puripuly_heart.core.stt.scoped_engine"
        and record.getMessage().startswith("[Recognition] ")
    ]
    assert len(messages) == 2
    events = {
        message.split()[1]: dict(token.split("=", 1) for token in message.split()[2:])
        for message in messages
    }
    receipt = events["payload_received"]
    terminal = events["terminal"]
    assert receipt["utterance_id"] == terminal["utterance_id"] == str(identity.segment.segment_id)
    assert receipt["epoch"] == terminal["epoch"] == identity.provider_epoch_id
    assert receipt["turn"] == terminal["turn"] == identity.provider_turn_id
    assert receipt["context_only"] == "1"
    assert receipt["successful_bytes"] == "4"
    assert terminal["outcome"] == "empty"
    assert terminal["successful_payloads"] == "3"
    assert terminal["successful_samples"] == "10"
    assert terminal["successful_bytes"] == "20"
    assert terminal["context_bytes"] == "4"
    assert terminal["content_bytes"] == "16"


@pytest.mark.asyncio
async def test_peer_recognition_evidence_distinguishes_never_written_failure(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class FailingBeginSession(ControlledScopedSession):
        async def begin_turn(self, request: STTProviderTurnRequest) -> None:
            raise RuntimeError("private failure detail")

    ledger = PeerAudioSegmentLedger(activation_generation=1, settings=settings())
    start, _chunk, end = segment_events(ledger, start_sample=500, now=2.0)
    engine = ScopedRecognitionEngine(
        session_factory=lambda _settings, _epoch: asyncio.sleep(0, result=FailingBeginSession()),
        watchdog_resolver=lambda _settings: watchdogs(),
    )
    with caplog.at_level(logging.INFO, logger="puripuly_heart.core.stt.scoped_engine"):
        await engine.handle_owned_vad_event(start)
        await engine.handle_owned_vad_event(end)
        await engine.close()

    messages = [
        record.getMessage()
        for record in caplog.records
        if record.name == "puripuly_heart.core.stt.scoped_engine"
        and record.getMessage().startswith("[Recognition] ")
    ]
    assert len(messages) == 1
    fields = dict(token.split("=", 1) for token in messages[0].split()[2:])
    assert fields["utterance_id"] == str(start.segment.identity.segment_id)
    assert fields["outcome"] == "failed"
    assert fields["cause"] == "provider_begin_failed"
    assert fields["final_wait_ms"] == "none"
    assert fields["final_timeout_ms"] == "30"
    assert fields["activation_generation"] == "1"
    assert fields["successful_payloads"] == "0"
    assert fields["successful_bytes"] == fields["content_bytes"] == fields["context_bytes"] == "0"
    assert "private failure detail" not in messages[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("channel", ["self", "peer"])
@pytest.mark.parametrize(
    "reason",
    ["soniox_request_failed", "soniox_protocol_ambiguity", "soniox_receive_failed"],
)
async def test_terminal_evidence_preserves_owned_failure_for_both_channels(
    channel: str,
    reason: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    ledger = PeerAudioSegmentLedger(activation_generation=37, settings=settings("soniox"))
    start, _chunk, end = segment_events(ledger, start_sample=550, now=2.0)
    session = ControlledScopedSession()
    emitted: list[object] = []
    engine = ScopedRecognitionEngine(
        session_factory=lambda _settings, _epoch: asyncio.sleep(0, result=session),
        channel=cast(Any, channel),
        event_sink=lambda event: emitted.append(event),
        watchdog_resolver=lambda _settings: watchdogs(final_timeout_s=0.2),
    )
    with caplog.at_level(logging.INFO, logger="puripuly_heart.core.stt.scoped_engine"):
        await engine.handle_owned_vad_event(start)
        identity = session.requests[0].identity
        ending = asyncio.create_task(engine.handle_owned_vad_event(end))
        await wait_until(
            lambda: engine._turn is not None and engine._turn.final_wait_started_at_s is not None
        )
        session.emit(
            STTProviderTurnTerminal(
                identity=identity,
                outcome="failed",
                text_authority="none",
                failure_reason=f"{reason}:private transcript and response",
                epoch_disposition="retire",
            )
        )
        await ending
        session.emit(
            STTProviderTurnTerminal(
                identity=identity,
                outcome="cancelled",
                text_authority="none",
                failure_reason="toggle_off",
            )
        )
        await engine.close()

    records = [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("[Recognition] terminal ")
    ]
    assert len(records) == 1
    fields = dict(token.split("=", 1) for token in records[0].split()[2:])
    assert fields["channel"] == channel
    assert fields["utterance_id"] == str(identity.segment.segment_id)
    assert fields["epoch"] == identity.provider_epoch_id
    assert fields["turn"] == identity.provider_turn_id
    assert fields["activation_generation"] == "37"
    assert fields["outcome"] == "failed"
    assert fields["cause"] == reason
    assert fields["final_timeout_ms"] == "200"
    assert fields["final_wait_ms"] != "none"
    assert "private transcript and response" not in records[0]
    terminals = [event for event in emitted if isinstance(event, STTProviderTurnTerminal)]
    assert len(terminals) == 1
    assert terminals[0].failure_reason == f"{reason}:private transcript and response"


@pytest.mark.asyncio
async def test_peer_recognition_survives_logging_handler_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger = PeerAudioSegmentLedger(activation_generation=1, settings=settings())
    start, _chunk, end = segment_events(ledger, start_sample=650, now=3.0)
    session = ControlledScopedSession()
    session.terminal_on_seal = ("final", "recognized")
    emitted: list[object] = []
    engine = ScopedRecognitionEngine(
        session_factory=lambda _settings, _epoch: asyncio.sleep(0, result=session),
        event_sink=emitted.append,
        watchdog_resolver=lambda _settings: watchdogs(),
    )

    def broken_handler(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("handler unavailable")

    monkeypatch.setattr("puripuly_heart.core.stt.scoped_engine.logger.info", broken_handler)
    await engine.handle_owned_vad_event(start)
    await engine.handle_owned_vad_event(end)
    assert [
        (event.outcome, event.text)
        for event in emitted
        if isinstance(event, STTProviderTurnTerminal)
    ] == [("final", "recognized")]
    await engine.close()


def test_event_buffer_coalesces_provisional_and_fails_stable_overflow() -> None:
    identity = STTProviderTurnIdentity(
        segment=segment_events(
            PeerAudioSegmentLedger(activation_generation=1, settings=settings()),
            start_sample=300,
            now=3.0,
        )[0].segment.identity,
        provider_epoch_id="epoch",
        provider_turn_id="turn",
    )
    buffer = STTProviderEventBuffer()
    for sequence in range(300):
        assert buffer.put(
            STTProviderTurnUpdate(
                identity=identity,
                sequence=sequence,
                stability="provisional",
                assembly="replace",
                text=str(sequence),
            )
        )
    assert buffer.depth == 1

    stable_buffer = STTProviderEventBuffer()
    for sequence in range(256):
        assert stable_buffer.put(
            STTProviderTurnUpdate(
                identity=identity,
                sequence=sequence,
                stability="stable",
                assembly="append",
                text="x",
            )
        )
    assert not stable_buffer.put(
        STTProviderTurnTerminal(
            identity=identity,
            outcome="final",
            text="done",
            text_authority="authoritative",
        )
    )
    assert stable_buffer.depth == 1
    failed = asyncio.run(stable_buffer.get())
    assert isinstance(failed, STTProviderTurnTerminal)
    assert failed.failure_reason == "provider_event_buffer_overflow"
    assert failed.epoch_disposition == "retire"


def test_independent_final_byte_pressure_is_explicit_and_releases_on_consumption() -> None:
    from puripuly_heart.core.stt.backend import STTRecognitionUnit, STTRecognitionUnitTerminal
    from puripuly_heart.domain.recognition import RecognitionStreamIdentity, RecognitionUnitIdentity

    stream = RecognitionStreamIdentity("peer", 1, 1, "epoch", ())
    buffer = STTProviderEventBuffer()
    text = "가" * (buffer.MAX_RECOGNITION_BYTES // 3)
    first = STTRecognitionUnitTerminal(
        STTRecognitionUnit(RecognitionUnitIdentity(stream, uuid4(), 1), text), "final"
    )
    second = STTRecognitionUnitTerminal(
        STTRecognitionUnit(RecognitionUnitIdentity(stream, uuid4(), 2), text), "final"
    )
    rejected = STTRecognitionUnitTerminal(
        STTRecognitionUnit(RecognitionUnitIdentity(stream, uuid4(), 3), "overflow"), "final"
    )
    assert buffer.put(first)
    assert asyncio.run(buffer.get()) == first
    assert buffer.put(second)
    assert not buffer.put(rejected)
    assert not buffer.put(rejected)
    assert buffer.depth == 2
    assert asyncio.run(buffer.get()) == second
    terminal = asyncio.run(buffer.get())
    assert terminal == STTProviderEpochEnded(
        "epoch", orderly=False, reason="provider_event_buffer_overflow"
    )
    assert buffer.depth == 0
    assert not buffer.put(STTRecognitionUnit(RecognitionUnitIdentity(stream, uuid4(), 4), "late"))


def test_independent_final_count_pressure_preserves_order_and_other_epoch() -> None:
    from puripuly_heart.core.stt.backend import STTRecognitionUnit, STTRecognitionUnitTerminal
    from puripuly_heart.domain.recognition import RecognitionStreamIdentity, RecognitionUnitIdentity

    stream = RecognitionStreamIdentity("peer", 1, 1, "epoch", ())
    other_stream = RecognitionStreamIdentity("peer", 1, 1, "other", ())
    buffer = STTProviderEventBuffer(max_events=2)
    first = STTRecognitionUnitTerminal(
        STTRecognitionUnit(RecognitionUnitIdentity(stream, uuid4(), 1), "first"), "final"
    )
    other = STTRecognitionUnitTerminal(
        STTRecognitionUnit(RecognitionUnitIdentity(other_stream, uuid4(), 1), "other"), "final"
    )
    rejected = STTRecognitionUnitTerminal(
        STTRecognitionUnit(RecognitionUnitIdentity(stream, uuid4(), 2), "rejected"), "final"
    )
    assert buffer.put(first)
    assert buffer.put(other)
    assert not buffer.put(rejected)
    assert not buffer.put(rejected)
    assert buffer.depth == 3
    assert [asyncio.run(buffer.get()) for _ in range(3)] == [
        first,
        other,
        STTProviderEpochEnded("epoch", orderly=False, reason="provider_event_buffer_overflow"),
    ]
    assert not buffer.put(rejected)
    assert buffer.put(STTProviderEpochEnded("other", orderly=True, reason="completed"))
    assert asyncio.run(buffer.get()) == STTProviderEpochEnded(
        "other", orderly=True, reason="completed"
    )


def test_stream_source_overlap_never_replays_already_submitted_samples() -> None:
    from puripuly_heart.core.stt.stream_input import STTStreamInputMap

    mapping = STTStreamInputMap()
    first, write = mapping.prepare(b"aabbccdd", (span(1, 0, 4),))
    assert first == b"aabbccdd" and write is not None
    mapping.commit(write)
    second, write = mapping.prepare(b"ccddeeff", (span(2, 2, 6),))
    assert second == b"eeff" and write is not None
    mapping.commit(write)
    assert mapping.covers((span(3, 0, 6),))
    assert mapping.prepare(b"aabb", (span(4, 0, 2),)) == (b"", None)
    with pytest.raises(ValueError, match="discontinuity"):
        mapping.prepare(b"zz", (span(5, 7, 8),))
    assert mapping.sent_samples == 6


@pytest.mark.parametrize("start,end", [(0, 2), (2, 5), (1, 4)])
def test_context_overlap_submits_only_unseen_suffix(start: int, end: int) -> None:
    from puripuly_heart.core.stt.stream_input import STTStreamInputMap

    mapping = STTStreamInputMap()
    original = b"aabbccddeeff"
    payload, initial = mapping.prepare(original[:8], (span(1, 0, 4),))
    assert payload == original[:8] and initial is not None
    mapping.commit(initial)
    payload, context = mapping.prepare(original[start * 2 : end * 2], (span(2, start, end),))
    assert payload == original[max(4, start) * 2 : end * 2]
    if context is not None:
        mapping.commit(context)
    frontier = max(4, end)
    assert mapping.covers((span(3, 0, frontier),))
    remaining, write = mapping.prepare(original[6:12], (span(4, 3, 6),))
    assert remaining == original[frontier * 2 : 12] and write is not None
    mapping.commit(write)
    assert mapping.sent_samples == 6


def test_initial_context_is_retained_but_later_prefix_is_not_replayed() -> None:
    from puripuly_heart.core.stt.stream_input import STTStreamInputMap

    mapping = STTStreamInputMap()
    payload, initial = mapping.prepare(b"aabb", (span(1, 4, 6),))
    assert payload == b"aabb" and initial is not None
    mapping.commit(initial)
    assert mapping.prepare(b"zzzz", (span(2, 0, 2),)) == (b"", None)
    for ranges in (
        (span(3, 7, 9),),
        (span(4, 6, 7), span(5, 8, 9)),
        (replace(span(6, 4, 6), capture_epoch=2),),
    ):
        with pytest.raises(ValueError, match="discontinuity|epoch"):
            mapping.prepare(b"ccdd", ranges)
    assert mapping.sent_samples == 2
    assert not mapping.covers((span(7, 0, 2),))


def test_stream_source_history_does_not_expire_continuous_input_at_two_minutes() -> None:
    from puripuly_heart.core.stt.stream_input import STTStreamInputMap

    mapping = STTStreamInputMap()
    pcm = b"\x01\x00" * 16000
    for second in range(180):
        payload, write = mapping.prepare(pcm, (span(second, second * 16000, (second + 1) * 16000),))
        assert payload == pcm and write is not None
        mapping.commit(write)
    assert mapping.sent_samples == 180 * 16000
    assert mapping.covers((span(180, 0, 180 * 16000),))


def test_normalizer_enforces_text_and_language_run_bounds() -> None:
    identity = STTProviderTurnIdentity(
        segment=segment_events(
            PeerAudioSegmentLedger(activation_generation=1, settings=settings()),
            start_sample=400,
            now=4.0,
        )[0].segment.identity,
        provider_epoch_id="epoch",
        provider_turn_id="turn",
    )
    accepted = "a" * STTScopedTurnNormalizer.MAX_ASSEMBLY_BYTES
    normalizer = STTScopedTurnNormalizer(identity)
    exact = normalizer.apply_terminal(
        STTProviderTurnTerminal(
            identity=identity,
            outcome="final",
            text=accepted,
            text_authority="authoritative",
        )
    )
    assert exact.text == accepted
    assert exact.final_language_runs == ()

    oversized = STTScopedTurnNormalizer(identity)
    with pytest.raises(STTNormalizationError, match="provider_result_too_large"):
        oversized.apply_terminal(
            STTProviderTurnTerminal(
                identity=identity,
                outcome="final",
                text=accepted + "a",
                text_authority="authoritative",
            )
        )

    diagnostics: list[object] = []
    runs_256 = tuple(
        FinalLanguageRun(text="x", language="en" if index % 2 else "ja") for index in range(256)
    )
    bounded = STTScopedTurnNormalizer(identity, diagnostic_sink=diagnostics.append)
    result_256 = bounded.apply_terminal(
        STTProviderTurnTerminal(
            identity=identity,
            outcome="final",
            text=" " + ("x" * 256) + " ",
            final_language_runs=(
                FinalLanguageRun(text=" ", language="ja"),
                *runs_256,
                FinalLanguageRun(text=" ", language="en"),
            ),
            text_authority="authoritative",
        )
    )
    assert result_256.text == "x" * 256
    assert len(result_256.final_language_runs) == 256
    assert "".join(run.text for run in result_256.final_language_runs) == result_256.text
    assert diagnostics == []

    runs_257 = tuple(
        FinalLanguageRun(text="y", language="en" if index % 2 else "ja") for index in range(257)
    )
    fallback = STTScopedTurnNormalizer(identity, diagnostic_sink=diagnostics.append)
    result_257 = fallback.apply_terminal(
        STTProviderTurnTerminal(
            identity=identity,
            outcome="final",
            text="y" * 257,
            final_language_runs=runs_257,
            text_authority="authoritative",
        )
    )
    assert result_257.final_language_runs == (FinalLanguageRun(text="y" * 257, language=""),)
    assert diagnostics


def test_normalizer_accepts_metadata_free_updates_and_terminal_without_diagnostics() -> None:
    identity = STTProviderTurnIdentity(
        segment=segment_events(
            PeerAudioSegmentLedger(activation_generation=1, settings=settings()),
            start_sample=425,
            now=4.25,
        )[0].segment.identity,
        provider_epoch_id="epoch",
        provider_turn_id="metadata-free",
    )
    diagnostics: list[object] = []
    normalizer = STTScopedTurnNormalizer(identity, diagnostic_sink=diagnostics.append)

    provisional = normalizer.apply_update(
        STTProviderTurnUpdate(
            identity=identity,
            sequence=1,
            stability="provisional",
            assembly="replace",
            text="draft",
        )
    )
    stable = normalizer.apply_update(
        STTProviderTurnUpdate(
            identity=identity,
            sequence=2,
            stability="stable",
            assembly="replace",
            text="hello",
        )
    )
    terminal = normalizer.apply_terminal(
        STTProviderTurnTerminal(
            identity=identity,
            outcome="final",
            text="  hello world  ",
            text_authority="authoritative",
        )
    )

    assert provisional is not None and (provisional.text, provisional.final_language_runs) == (
        "draft",
        (),
    )
    assert stable is not None and (stable.text, stable.final_language_runs) == ("hello", ())
    assert (terminal.text, terminal.final_language_runs) == ("hello world", ())
    assert diagnostics == []


def test_normalizer_append_preserves_known_and_absent_language_regions() -> None:
    identity = STTProviderTurnIdentity(
        segment=segment_events(
            PeerAudioSegmentLedger(activation_generation=1, settings=settings()),
            start_sample=430,
            now=4.3,
        )[0].segment.identity,
        provider_epoch_id="epoch",
        provider_turn_id="mixed-metadata",
    )
    diagnostics: list[object] = []
    normalizer = STTScopedTurnNormalizer(identity, diagnostic_sink=diagnostics.append)

    updates = (
        ("hello ", (FinalLanguageRun(text="hello ", language="en"),)),
        ("there ", ()),
        ("世界", (FinalLanguageRun(text="世界", language="ja"),)),
    )
    result = None
    for sequence, (text, runs) in enumerate(updates, start=1):
        result = normalizer.apply_update(
            STTProviderTurnUpdate(
                identity=identity,
                sequence=sequence,
                stability="stable",
                assembly="append",
                text=text,
                final_language_runs=runs,
            )
        )

    assert result is not None
    assert result.text == "hello there 世界"
    assert result.final_language_runs == (
        FinalLanguageRun(text="hello ", language="en"),
        FinalLanguageRun(text="there ", language=""),
        FinalLanguageRun(text="世界", language="ja"),
    )
    assert diagnostics == []


def test_corrupt_appended_language_metadata_repairs_only_its_text() -> None:
    identity = STTProviderTurnIdentity(
        segment=segment_events(
            PeerAudioSegmentLedger(activation_generation=1, settings=settings()),
            start_sample=440,
            now=4.4,
        )[0].segment.identity,
        provider_epoch_id="epoch",
        provider_turn_id="repaired-metadata",
    )
    diagnostics: list[object] = []
    normalizer = STTScopedTurnNormalizer(identity, diagnostic_sink=diagnostics.append)

    updates = (
        ("hello ", (FinalLanguageRun(text="hello ", language="en"),)),
        ("broken ", (FinalLanguageRun(text="broken ", language="   "),)),
        ("世界", (FinalLanguageRun(text="世界", language="ja"),)),
    )
    result = None
    for sequence, (text, runs) in enumerate(updates, start=1):
        result = normalizer.apply_update(
            STTProviderTurnUpdate(
                identity=identity,
                sequence=sequence,
                stability="stable",
                assembly="append",
                text=text,
                final_language_runs=runs,
            )
        )

    assert result is not None
    assert result.text == "hello broken 世界"
    assert result.final_language_runs == (
        FinalLanguageRun(text="hello ", language="en"),
        FinalLanguageRun(text="broken ", language=""),
        FinalLanguageRun(text="世界", language="ja"),
    )
    assert diagnostics


def test_normalizer_bounds_private_raw_whitespace_and_cumulative_appends() -> None:
    identity = STTProviderTurnIdentity(
        segment=segment_events(
            PeerAudioSegmentLedger(activation_generation=1, settings=settings()),
            start_sample=450,
            now=4.5,
        )[0].segment.identity,
        provider_epoch_id="epoch",
        provider_turn_id="raw-bound",
    )
    oversized_raw = STTScopedTurnNormalizer(identity)
    with pytest.raises(STTNormalizationError, match="provider_result_too_large"):
        oversized_raw.apply_update(
            STTProviderTurnUpdate(
                identity=identity,
                sequence=1,
                stability="stable",
                assembly="append",
                text=" " * (2 * STTScopedTurnNormalizer.MAX_ASSEMBLY_BYTES) + "x",
            )
        )
    assert oversized_raw.stable_text == ""
    recovered = oversized_raw.apply_update(
        STTProviderTurnUpdate(
            identity=identity,
            sequence=2,
            stability="stable",
            assembly="append",
            text="ok",
        )
    )
    assert recovered is not None and recovered.text == "ok"

    cumulative = STTScopedTurnNormalizer(identity)
    first = cumulative.apply_update(
        STTProviderTurnUpdate(
            identity=identity,
            sequence=1,
            stability="stable",
            assembly="append",
            text="hello",
        )
    )
    whitespace = " " * (STTScopedTurnNormalizer.MAX_ASSEMBLY_BYTES // 4)
    middle = cumulative.apply_update(
        STTProviderTurnUpdate(
            identity=identity,
            sequence=2,
            stability="stable",
            assembly="append",
            text=whitespace,
        )
    )
    assert first is not None and middle is not None
    assert middle.text == "hello"
    for sequence in range(3, 7):
        try:
            cumulative.apply_update(
                STTProviderTurnUpdate(
                    identity=identity,
                    sequence=sequence,
                    stability="stable",
                    assembly="append",
                    text=whitespace,
                )
            )
        except STTNormalizationError as exc:
            assert exc.reason == "provider_result_too_large"
            break
    else:
        pytest.fail("raw whitespace accumulation exceeded the assembly limit")
    assert cumulative.stable_text == "hello"


@pytest.mark.asyncio
async def test_local_write_timeout_keeps_one_quarantined_resource() -> None:
    local_settings = settings("local_qwen")
    ledger = PeerAudioSegmentLedger(activation_generation=1, settings=local_settings)
    a_start, _a_chunk, a_end = segment_events(ledger, start_sample=500, now=5.0)
    b_start, _b_chunk, b_end = segment_events(ledger, start_sample=600, now=6.0)
    session = ControlledScopedSession()
    session.send_gate.clear()
    factory_calls = 0
    emitted: list[object] = []

    async def factory(_settings: AudioSegmentSettingsSnapshot, _epoch: str):
        nonlocal factory_calls
        factory_calls += 1
        return session

    engine = ScopedRecognitionEngine(
        session_factory=factory,
        event_sink=lambda event: emitted.append(event),
        watchdog_resolver=lambda _settings: watchdogs(write_timeout_s=0.01),
    )
    await engine.handle_owned_vad_event(a_start)
    await engine.handle_owned_vad_event(a_end)
    assert engine.cleanup_debt == 1

    await engine.handle_owned_vad_event(b_start)
    await engine.handle_owned_vad_event(b_end)
    terminals = [item for item in emitted if isinstance(item, STTProviderTurnTerminal)]
    assert factory_calls == 1
    assert [item.outcome for item in terminals] == ["failed", "failed"]
    assert terminals[1].failure_reason == "provider_not_ready:RuntimeError"

    session.send_gate.set()
    await wait_until(lambda: engine.cleanup_debt == 0)
    await engine.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider_id", "first_outcome", "first_text"),
    [
        ("custom", "empty", ""),
        ("qwen_asr", "final", "alpha"),
    ],
)
async def test_idle_epoch_end_forces_fresh_session_without_replaying_audio(
    provider_id: str,
    first_outcome: str,
    first_text: str,
) -> None:
    provider_settings = settings(provider_id)
    ledger = PeerAudioSegmentLedger(activation_generation=1, settings=provider_settings)
    first_start, _first_chunk, first_end = segment_events(
        ledger,
        start_sample=1100,
        now=11.0,
    )
    second_start, _second_chunk, second_end = segment_events(
        ledger,
        start_sample=1200,
        now=12.0,
    )
    first = ControlledScopedSession()
    first.terminal_on_seal = (first_outcome, first_text)
    second = ControlledScopedSession()
    second.terminal_on_seal = ("final", "bravo")
    sessions = [first, second]
    emitted: list[object] = []
    factory_epochs: list[str] = []

    async def factory(_settings: AudioSegmentSettingsSnapshot, epoch_id: str):
        factory_epochs.append(epoch_id)
        return sessions.pop(0)

    engine = ScopedRecognitionEngine(
        session_factory=factory,
        event_sink=lambda event: emitted.append(event),
        watchdog_resolver=lambda _settings: watchdogs(),
    )
    await engine.handle_owned_vad_event(first_start)
    await engine.handle_owned_vad_event(first_end)
    first_terminal = next(event for event in emitted if isinstance(event, STTProviderTurnTerminal))
    old_epoch_id = first_terminal.identity.provider_epoch_id
    assert first.emit(
        STTProviderEpochEnded(
            provider_epoch_id=old_epoch_id,
            orderly=True,
            reason="native_idle_end",
        )
    )

    await engine.handle_owned_vad_event(second_start)
    assert ("close",) in first.calls
    assert second.emit(
        STTProviderEpochEnded(
            provider_epoch_id=old_epoch_id,
            orderly=True,
            reason="duplicate_old_epoch_end",
        )
    )
    await engine.handle_owned_vad_event(second_end)
    terminals = [event for event in emitted if isinstance(event, STTProviderTurnTerminal)]
    second_terminal = terminals[-1]
    assert len(factory_epochs) == 2
    assert factory_epochs[0] != factory_epochs[1]
    assert second_terminal.identity.segment == second_start.segment.identity
    assert second_terminal.identity.provider_epoch_id == factory_epochs[1]
    assert (second_terminal.outcome, second_terminal.text) == ("final", "bravo")
    first_identities = {call[1] for call in first.calls if call[0] in ("begin", "send", "seal")}
    second_identities = {call[1] for call in second.calls if call[0] in ("begin", "send", "seal")}
    assert first_identities == {first_terminal.identity}
    assert second_identities == {second_terminal.identity}
    await engine.close()


@pytest.mark.asyncio
async def test_cloud_cleanup_quarantines_one_epoch_until_physical_release() -> None:
    cloud_settings = settings("deepgram")
    ledger = PeerAudioSegmentLedger(activation_generation=1, settings=cloud_settings)
    segments = [
        segment_events(ledger, start_sample=600 + index * 100, now=6.0 + index)
        for index in range(3)
    ]
    session = BarrierCleanupScopedSession()
    session.terminal_on_seal = ("failed", "")
    factory_calls = 0
    emitted: list[object] = []
    provider_failures: list[Exception] = []
    backend_closes: list[None] = []

    async def close_backend() -> None:
        backend_closes.append(None)

    async def factory(_settings: AudioSegmentSettingsSnapshot, _epoch: str):
        nonlocal factory_calls
        factory_calls += 1
        return session

    async def release_on_teardown() -> None:
        try:
            await asyncio.sleep(10)
        finally:
            session.stop_gate.set()
            session.close_gate.set()

    teardown_release = asyncio.create_task(release_on_teardown())

    engine = ScopedRecognitionEngine(
        session_factory=factory,
        event_sink=lambda event: emitted.append(event),
        terminal_failure_sink=lambda exc: provider_failures.append(exc),
        backend_close=close_backend,
        watchdog_resolver=lambda _settings: watchdogs(
            readiness_timeout_s=0.01,
            drain_timeout_s=0.01,
        ),
        event_drain_timeout_s=0.01,
    )
    first_start, _first_chunk, first_end = segments[0]
    await engine.handle_owned_vad_event(first_start)
    await engine.handle_owned_vad_event(first_end)
    await wait_until(lambda: session.stop_cancellations == 1)

    for start, _chunk, end in segments[1:]:
        await engine.handle_owned_vad_event(start)
        await engine.handle_owned_vad_event(end)

    terminals = [item for item in emitted if isinstance(item, STTProviderTurnTerminal)]
    assert factory_calls == 1
    assert engine.cleanup_debt == 1
    assert len(terminals) == 3
    assert [item.outcome for item in terminals] == ["failed", "failed", "failed"]
    assert [item.failure_reason for item in terminals[1:]] == [
        "provider_not_ready:RuntimeError",
        "provider_not_ready:RuntimeError",
    ]
    assert len(provider_failures) == 1
    assert session.calls.count(("stop",)) == 1
    assert session.calls.count(("close",)) == 0

    await asyncio.wait_for(engine.abort_for_toggle_off(), timeout=0.05)
    await asyncio.wait_for(engine.close_backend(), timeout=0.05)
    assert engine.cleanup_debt == 1
    assert backend_closes == []
    assert len(provider_failures) == 1

    session.stop_gate.set()
    await wait_until(lambda: session.close_cancellations == 1)
    assert engine.cleanup_debt == 1
    session.close_gate.set()
    await wait_until(lambda: engine.cleanup_debt == 0)
    assert backend_closes == [None]
    assert session.calls.count(("stop_done",)) == 1
    assert session.calls.count(("close",)) == 1
    assert session.calls.count(("close_done",)) == 1
    teardown_release.cancel()
    await asyncio.gather(teardown_release, return_exceptions=True)


@pytest.mark.asyncio
async def test_interruptible_cloud_cleanup_releases_within_drain_and_resumes() -> None:
    cloud_settings = settings("deepgram")
    ledger = PeerAudioSegmentLedger(activation_generation=1, settings=cloud_settings)
    first_start, _first_chunk, first_end = segment_events(
        ledger,
        start_sample=900,
        now=9.0,
    )
    second_start, _second_chunk, second_end = segment_events(
        ledger,
        start_sample=1000,
        now=10.0,
    )
    first = InterruptibleCleanupScopedSession()
    first.terminal_on_seal = ("failed", "")
    second = ControlledScopedSession()
    second.terminal_on_seal = ("empty", "")
    sessions = [first, second]
    emitted: list[object] = []
    provider_failures: list[Exception] = []

    async def factory(_settings: AudioSegmentSettingsSnapshot, _epoch: str):
        return sessions.pop(0)

    engine = ScopedRecognitionEngine(
        session_factory=factory,
        event_sink=lambda event: emitted.append(event),
        terminal_failure_sink=lambda exc: provider_failures.append(exc),
        watchdog_resolver=lambda _settings: watchdogs(drain_timeout_s=0.01),
    )
    await engine.handle_owned_vad_event(first_start)
    await engine.handle_owned_vad_event(first_end)
    await wait_until(lambda: engine.cleanup_debt == 0)
    assert first.stop_cancellations == 1
    assert first.close_cancellations == 1

    await engine.handle_owned_vad_event(second_start)
    await engine.handle_owned_vad_event(second_end)
    terminals = [item for item in emitted if isinstance(item, STTProviderTurnTerminal)]
    assert [item.outcome for item in terminals] == ["failed", "empty"]
    assert provider_failures == []
    assert sessions == []
    await engine.close()


@pytest.mark.asyncio
async def test_recovery_is_three_attempts_with_point_eight_and_one_point_six_backoff() -> None:
    ledger = PeerAudioSegmentLedger(activation_generation=1, settings=settings())
    start, _chunk, end = segment_events(ledger, start_sample=800, now=8.0)
    attempts = 0
    delays: list[float] = []
    emitted: list[object] = []
    terminal_failures: list[Exception] = []

    async def failing_factory(_settings: AudioSegmentSettingsSnapshot, _epoch: str):
        nonlocal attempts
        attempts += 1
        raise ConnectionError("offline")

    async def record_sleep(delay: float) -> None:
        delays.append(delay)

    engine = ScopedRecognitionEngine(
        session_factory=failing_factory,
        event_sink=lambda event: emitted.append(event),
        watchdog_resolver=lambda _settings: STTRecognitionWatchdogs(),
        sleep=record_sleep,
        terminal_failure_sink=lambda failure: terminal_failures.append(failure),
    )
    await engine.handle_owned_vad_event(start)
    await engine.handle_owned_vad_event(end)

    assert attempts == 3
    assert delays == [0.8, 1.6]
    terminal = next(item for item in emitted if isinstance(item, STTProviderTurnTerminal))
    assert terminal.outcome == "failed"
    assert terminal.failure_reason == "provider_not_ready:RuntimeError"
    assert len(terminal_failures) == 1
    assert str(terminal_failures[0]) == "provider_recovery_exhausted"
    await engine.close()


@pytest.mark.asyncio
async def test_ready_then_failed_epochs_exhaust_one_recovery_episode() -> None:
    emitted: list[object] = []
    sessions: list[ControlledScopedSession] = []

    class ReadyThenFailedSession(ControlledScopedSession):
        async def begin_turn(self, request: STTProviderTurnRequest) -> None:
            await super().begin_turn(request)
            self.buffer.put(
                STTProviderEpochEnded(
                    provider_epoch_id=request.identity.provider_epoch_id,
                    orderly=False,
                    reason="ready_then_failed",
                    provider_turn_id=request.identity.provider_turn_id,
                )
            )

    async def factory(_settings: AudioSegmentSettingsSnapshot, _epoch: str):
        session = ReadyThenFailedSession()
        sessions.append(session)
        return session

    engine = ScopedRecognitionEngine(
        session_factory=factory,
        event_sink=lambda event: emitted.append(event),
        watchdog_resolver=lambda _settings: watchdogs(),
    )
    ledger = PeerAudioSegmentLedger(activation_generation=1, settings=settings())

    for index in range(4):
        start, _chunk, end = segment_events(
            ledger,
            start_sample=1000 + index * 20,
            now=10.0 + index,
        )
        await engine.handle_owned_vad_event(start)
        await asyncio.sleep(0)
        await engine.handle_owned_vad_event(end)

    terminals = [item for item in emitted if isinstance(item, STTProviderTurnTerminal)]
    assert len(sessions) == 3
    assert [terminal.outcome for terminal in terminals] == ["failed"] * 4
    assert [terminal.failure_reason for terminal in terminals[:3]] == ["ready_then_failed"] * 3
    assert terminals[3].failure_reason == "provider_not_ready:RuntimeError"
    await engine.close()


async def finish_soniox_turn(
    engine: ScopedRecognitionEngine,
    session: ControlledScopedSession,
    end: OwnedVadEvent,
    *,
    retryable: bool,
    reason: str = "soniox_receive_failed",
) -> STTProviderTurnIdentity:
    identity = session.requests[-1].identity
    session.emit(
        STTProviderTurnTerminal(
            identity=identity,
            outcome="failed",
            failure_reason=reason,
            failure_retryable=retryable,
            epoch_disposition="retire",
        )
    )
    await engine.handle_owned_vad_event(end)
    return identity


@pytest.mark.asyncio
async def test_soniox_transient_recovers_next_utterance_without_replaying_failed_audio() -> None:
    emitted: list[object] = []
    sessions: list[ControlledScopedSession] = []
    delays: list[float] = []

    async def factory(_settings: AudioSegmentSettingsSnapshot, _epoch: str):
        session = ControlledScopedSession()
        sessions.append(session)
        return session

    async def sleep(delay: float) -> None:
        delays.append(delay)

    engine = ScopedRecognitionEngine(
        session_factory=factory,
        channel="self",
        event_sink=emitted.append,
        watchdog_resolver=lambda _settings: watchdogs(
            connect_retry_base_s=0.8, connect_retry_max_s=1.6
        ),
        sleep=sleep,
        session_lifetime_enabled=False,
    )
    ledger = PeerAudioSegmentLedger(activation_generation=1, settings=settings("soniox"))
    first_start, _first_chunk, first_end = segment_events(ledger, start_sample=2000, now=20.0)
    second_start, _second_chunk, second_end = segment_events(ledger, start_sample=2020, now=21.0)
    await engine.handle_owned_vad_event(first_start)
    first_identity = await finish_soniox_turn(engine, sessions[0], first_end, retryable=True)
    await wait_until(lambda: ("close",) in sessions[0].calls)
    await engine.handle_owned_vad_event(second_start)
    second_identity = sessions[1].requests[0].identity
    sessions[1].terminal_on_seal = ("final", "recognized")
    await engine.handle_owned_vad_event(second_end)
    terminals = [event for event in emitted if isinstance(event, STTProviderTurnTerminal)]
    assert [(item.outcome, item.recovery_pending) for item in terminals] == [
        ("failed", True),
        ("final", False),
    ]
    assert terminals[0].failure_reason == "soniox_receive_failed"
    assert terminals[0].failure_retryable
    assert first_identity.provider_epoch_id != second_identity.provider_epoch_id
    assert [call[0] for call in sessions[1].calls].count("send") == 2
    assert delays == [0.8]
    await engine.close()


@pytest.mark.asyncio
async def test_soniox_write_timeout_cancels_write_and_recovers_new_epoch() -> None:
    emitted: list[object] = []
    sessions: list[ControlledScopedSession] = []

    async def factory(_settings: AudioSegmentSettingsSnapshot, _epoch: str):
        session = ControlledScopedSession()
        if not sessions:
            session.send_gate.clear()
        sessions.append(session)
        return session

    engine = ScopedRecognitionEngine(
        session_factory=factory,
        channel="self",
        event_sink=emitted.append,
        watchdog_resolver=lambda _settings: watchdogs(write_timeout_s=0.01),
        session_lifetime_enabled=False,
    )
    ledger = PeerAudioSegmentLedger(activation_generation=1, settings=settings("soniox"))
    start, _chunk, end = segment_events(ledger, start_sample=2100, now=21.0)
    successor_start, _chunk, successor_end = segment_events(ledger, start_sample=2120, now=22.0)
    await engine.handle_owned_vad_event(start)
    await engine.handle_owned_vad_event(end)
    await wait_until(lambda: ("close",) in sessions[0].calls)
    sessions[0].send_gate.set()
    await engine.handle_owned_vad_event(successor_start)
    sessions[1].terminal_on_seal = ("final", "next")
    await engine.handle_owned_vad_event(successor_end)
    terminals = [event for event in emitted if isinstance(event, STTProviderTurnTerminal)]
    assert terminals[0].failure_reason == "provider_send_timeout"
    assert terminals[0].failure_retryable
    assert terminals[0].recovery_pending
    assert terminals[1].outcome == "final"
    assert len([call for call in sessions[0].calls if call[0] == "send_done"]) == 0
    assert len([call for call in sessions[1].calls if call[0] == "send"]) == 2
    await engine.close()


@pytest.mark.asyncio
async def test_soniox_three_transient_turn_failures_exhaust_episode_and_final_resets_it() -> None:
    emitted: list[object] = []
    failures: list[Exception] = []
    sessions: list[ControlledScopedSession] = []
    delays: list[float] = []

    async def factory(_settings: AudioSegmentSettingsSnapshot, _epoch: str):
        session = ControlledScopedSession()
        sessions.append(session)
        return session

    async def sleep(delay: float) -> None:
        delays.append(delay)

    engine = ScopedRecognitionEngine(
        session_factory=factory,
        channel="self",
        event_sink=emitted.append,
        terminal_failure_sink=failures.append,
        watchdog_resolver=lambda _settings: watchdogs(
            connect_retry_base_s=0.8, connect_retry_max_s=1.6
        ),
        sleep=sleep,
        session_lifetime_enabled=False,
    )
    ledger = PeerAudioSegmentLedger(activation_generation=1, settings=settings("soniox"))
    for index in range(4):
        start, _chunk, end = segment_events(
            ledger, start_sample=3000 + 20 * index, now=30.0 + index
        )
        await engine.handle_owned_vad_event(start)
        if index < 3:
            await finish_soniox_turn(engine, sessions[index], end, retryable=True)
            await wait_until(lambda: ("close",) in sessions[index].calls)
        else:
            await engine.handle_owned_vad_event(end)
    terminals = [event for event in emitted if isinstance(event, STTProviderTurnTerminal)]
    assert len(sessions) == 3
    assert [item.recovery_pending for item in terminals] == [True, True, False, False]
    assert [item.failure_reason for item in terminals[:3]] == ["soniox_receive_failed"] * 3
    assert delays == [0.8, 1.6]
    assert len(failures) == 1
    await engine.close()

    reset_sessions: list[ControlledScopedSession] = []
    reset_events: list[object] = []

    async def reset_factory(_settings: AudioSegmentSettingsSnapshot, _epoch: str):
        session = ControlledScopedSession()
        reset_sessions.append(session)
        return session

    reset = ScopedRecognitionEngine(
        session_factory=reset_factory,
        channel="self",
        event_sink=reset_events.append,
        watchdog_resolver=lambda _settings: watchdogs(),
    )
    fresh = PeerAudioSegmentLedger(activation_generation=2, settings=settings("soniox"))
    for index in range(3):
        start, _chunk, end = segment_events(fresh, start_sample=4000 + index * 20, now=40.0 + index)
        await reset.handle_owned_vad_event(start)
        if index == 1:
            reset_sessions[-1].terminal_on_seal = ("final", "recovered")
            await reset.handle_owned_vad_event(end)
            reset_sessions[-1].terminal_on_seal = None
        else:
            await finish_soniox_turn(reset, reset_sessions[-1], end, retryable=True)
        if index != 1:
            await wait_until(lambda: ("close",) in reset_sessions[-1].calls)
    assert [
        item.recovery_pending for item in reset_events if isinstance(item, STTProviderTurnTerminal)
    ] == [True, False, True]
    await reset.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("channel", ["self", "peer"])
async def test_soniox_fatal_provider_terminal_never_opens_another_epoch(channel: str) -> None:
    emitted: list[object] = []
    sessions: list[ControlledScopedSession] = []
    failures: list[Exception] = []

    async def factory(_settings: AudioSegmentSettingsSnapshot, _epoch: str):
        session = ControlledScopedSession()
        sessions.append(session)
        return session

    engine = ScopedRecognitionEngine(
        session_factory=factory,
        channel=cast(Any, channel),
        event_sink=emitted.append,
        terminal_failure_sink=failures.append,
        watchdog_resolver=lambda _settings: watchdogs(),
    )
    ledger = PeerAudioSegmentLedger(activation_generation=1, settings=settings("soniox"))
    for index in range(3):
        start, _chunk, end = segment_events(
            ledger, start_sample=5000 + 20 * index, now=50.0 + index
        )
        await engine.handle_owned_vad_event(start)
        if index == 0:
            await finish_soniox_turn(
                engine, sessions[0], end, retryable=False, reason="soniox_request_failed"
            )
            await wait_until(lambda: ("close",) in sessions[0].calls)
        else:
            await engine.handle_owned_vad_event(end)
    terminals = [event for event in emitted if isinstance(event, STTProviderTurnTerminal)]
    assert len(sessions) == 1
    assert terminals[0].failure_reason == "soniox_request_failed"
    assert not any(item.recovery_pending for item in terminals)
    assert len(failures) == 1
    await engine.close()


def test_normalizer_preserves_provider_retryability_and_engine_recovery_metadata() -> None:
    ledger = PeerAudioSegmentLedger(activation_generation=1, settings=settings("soniox"))
    start, _chunk, _end = segment_events(ledger, start_sample=6000, now=60.0)
    identity = STTProviderTurnIdentity(start.segment.identity, "epoch", "turn")
    normalized = STTScopedTurnNormalizer(identity).apply_terminal(
        STTProviderTurnTerminal(
            identity=identity,
            outcome="failed",
            failure_reason="soniox_request_failed",
            epoch_disposition="retire",
            failure_retryable=True,
            recovery_pending=True,
        )
    )
    assert normalized.failure_reason == "soniox_request_failed"
    assert normalized.failure_retryable is True
    assert normalized.recovery_pending is True


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["backoff", "opening"])
async def test_soniox_toggle_during_recovery_never_installs_late_epoch(phase: str) -> None:
    emitted: list[object] = []
    failures: list[Exception] = []
    sessions: list[ControlledScopedSession] = []
    sleep_gate = asyncio.Event()
    open_gate = asyncio.Event()

    async def factory(_settings: AudioSegmentSettingsSnapshot, _epoch: str):
        session = ControlledScopedSession()
        sessions.append(session)
        if len(sessions) == 2:
            await open_gate.wait()
        return session

    async def sleep(_delay: float) -> None:
        await sleep_gate.wait()

    engine = ScopedRecognitionEngine(
        session_factory=factory,
        channel="self",
        event_sink=emitted.append,
        terminal_failure_sink=failures.append,
        watchdog_resolver=lambda _settings: watchdogs(readiness_timeout_s=0.5),
        sleep=sleep,
        session_lifetime_enabled=False,
    )
    ledger = PeerAudioSegmentLedger(activation_generation=1, settings=settings("soniox"))
    first_start, _chunk, first_end = segment_events(ledger, start_sample=7000, now=70.0)
    next_start, _chunk, _next_end = segment_events(ledger, start_sample=7020, now=71.0)
    await engine.handle_owned_vad_event(first_start)
    await finish_soniox_turn(engine, sessions[0], first_end, retryable=True)
    await wait_until(lambda: ("close",) in sessions[0].calls)
    pending_start = asyncio.create_task(engine.handle_owned_vad_event(next_start))
    await asyncio.sleep(0)
    if phase == "opening":
        sleep_gate.set()
        await wait_until(lambda: len(sessions) == 2)
    await engine.abort_for_toggle_off()
    sleep_gate.set()
    open_gate.set()
    await pending_start
    if phase == "opening":
        await wait_until(lambda: ("close",) in sessions[1].calls)
    assert len(sessions) == (1 if phase == "backoff" else 2)
    if phase == "opening":
        assert not any(call[0] == "begin" for call in sessions[1].calls)
    assert len([event for event in emitted if isinstance(event, STTProviderTurnTerminal)]) == 1
    assert not failures
    await engine.close()


@pytest.mark.asyncio
async def test_soniox_idle_auth_failure_blocks_reopening_and_notifies_once() -> None:
    emitted: list[object] = []
    sessions: list[ControlledScopedSession] = []
    failures: list[Exception] = []

    async def factory(_settings: AudioSegmentSettingsSnapshot, _epoch: str):
        session = ControlledScopedSession()
        sessions.append(session)
        return session

    engine = ScopedRecognitionEngine(
        session_factory=factory,
        channel="self",
        event_sink=emitted.append,
        terminal_failure_sink=failures.append,
        watchdog_resolver=lambda _settings: watchdogs(),
    )
    ledger = PeerAudioSegmentLedger(activation_generation=1, settings=settings("soniox"))
    first_start, _chunk, first_end = segment_events(ledger, start_sample=8000, now=80.0)
    next_start, _chunk, next_end = segment_events(ledger, start_sample=8020, now=81.0)
    await engine.handle_owned_vad_event(first_start)
    sessions[0].terminal_on_seal = ("final", "healthy")
    await engine.handle_owned_vad_event(first_end)
    sessions[0].emit(
        STTProviderEpochEnded(
            provider_epoch_id=sessions[0].requests[0].identity.provider_epoch_id,
            orderly=False,
            reason="soniox_request_failed",
            failure_retryable=False,
        )
    )
    await wait_until(lambda: bool(failures))
    await engine.handle_owned_vad_event(next_start)
    await engine.handle_owned_vad_event(next_end)
    assert len(sessions) == 1
    assert len(failures) == 1
    await engine.close()


@pytest.mark.asyncio
async def test_soniox_idle_transient_epoch_counts_once_and_recovers_on_next_start() -> None:
    sessions: list[ControlledScopedSession] = []
    emitted: list[object] = []
    delays: list[float] = []

    async def factory(_settings: AudioSegmentSettingsSnapshot, _epoch: str):
        session = ControlledScopedSession()
        sessions.append(session)
        return session

    async def sleep(delay: float) -> None:
        delays.append(delay)

    engine = ScopedRecognitionEngine(
        session_factory=factory,
        channel="self",
        event_sink=emitted.append,
        watchdog_resolver=lambda _settings: watchdogs(),
        sleep=sleep,
        session_lifetime_enabled=False,
    )
    ledger = PeerAudioSegmentLedger(activation_generation=1, settings=settings("soniox"))
    first_start, _chunk, first_end = segment_events(ledger, start_sample=9000, now=90.0)
    next_start, _chunk, next_end = segment_events(ledger, start_sample=9020, now=91.0)
    await engine.handle_owned_vad_event(first_start)
    sessions[0].terminal_on_seal = ("final", "healthy")
    await engine.handle_owned_vad_event(first_end)
    sessions[0].emit(
        STTProviderEpochEnded(
            provider_epoch_id=sessions[0].requests[0].identity.provider_epoch_id,
            orderly=False,
            reason="soniox_receive_failed",
            failure_retryable=True,
        )
    )
    await wait_until(lambda: ("close",) in sessions[0].calls)
    await engine.handle_owned_vad_event(next_start)
    sessions[1].terminal_on_seal = ("final", "next")
    await engine.handle_owned_vad_event(next_end)
    assert len(sessions) == 2
    assert delays == [0.001]
    await engine.close()


@pytest.mark.asyncio
async def test_configuration_change_rotates_at_turn_boundary() -> None:
    first_ledger = PeerAudioSegmentLedger(
        activation_generation=1,
        settings=settings(signature="old"),
    )
    second_ledger = PeerAudioSegmentLedger(
        activation_generation=1,
        settings=settings(signature="new"),
    )
    first = segment_events(first_ledger, start_sample=900, now=9.0)
    second = segment_events(second_ledger, start_sample=1000, now=10.0)
    sessions: list[ControlledScopedSession] = []

    async def factory(_settings: AudioSegmentSettingsSnapshot, _epoch: str):
        session = ControlledScopedSession()
        session.terminal_on_seal = ("empty", "")
        sessions.append(session)
        return session

    engine = ScopedRecognitionEngine(
        session_factory=factory,
        event_sink=lambda _event: None,
        watchdog_resolver=lambda _settings: watchdogs(),
    )

    await engine.handle_owned_vad_event(first[0])
    await engine.handle_owned_vad_event(first[2])
    await engine.handle_owned_vad_event(second[0])
    await engine.handle_owned_vad_event(second[2])

    assert len(sessions) == 2
    await wait_until(lambda: any(call[0] == "close" for call in sessions[0].calls))
    await engine.close()


@pytest.mark.asyncio
async def test_idle_closes_at_sixty_seconds_without_callbacks_even_after_long_healthy_age() -> None:
    ledger = PeerAudioSegmentLedger(activation_generation=1, settings=settings())
    first = segment_events(ledger, start_sample=900, now=9.0)
    session = ControlledScopedSession()
    session.terminal_on_seal = ("empty", "")
    clock = ControlledMonotonicClock()
    engine = ScopedRecognitionEngine(
        session_factory=lambda _settings, _epoch: asyncio.sleep(0, result=session),
        event_sink=lambda _event: None,
        watchdog_resolver=lambda _settings: watchdogs(),
        monotonic_clock=clock.now,
        sleep=clock.sleep,
    )

    await engine.observe_source_activity(speech_observed=True)
    await engine.handle_owned_vad_event(first[0])
    await engine.handle_owned_vad_event(first[2])
    await clock.advance_to(300.0)
    assert not any(call[0] == "close" for call in session.calls)

    await engine.observe_source_activity(speech_observed=False)
    await clock.advance_to(359.9)
    assert not any(call[0] == "close" for call in session.calls)
    await engine.observe_source_activity(speech_observed=True)
    await engine.observe_source_activity(speech_observed=False)
    await clock.advance_to(419.8)
    assert not any(call[0] == "close" for call in session.calls)
    await clock.advance_to(419.9)
    await wait_until(lambda: any(call[0] == "close" for call in session.calls))
    await engine.close()


@pytest.mark.asyncio
async def test_unresolved_turn_delays_idle_until_terminal_without_new_source_callback() -> None:
    ledger = PeerAudioSegmentLedger(activation_generation=1, settings=settings())
    first = segment_events(ledger, start_sample=900, now=9.0)
    session = ControlledScopedSession()
    clock = ControlledMonotonicClock()
    emitted: list[object] = []
    engine = ScopedRecognitionEngine(
        session_factory=lambda _settings, _epoch: asyncio.sleep(0, result=session),
        event_sink=emitted.append,
        watchdog_resolver=lambda _settings: watchdogs(),
        monotonic_clock=clock.now,
        sleep=clock.sleep,
    )

    await engine.handle_owned_vad_event(first[0])
    end_task = asyncio.create_task(engine.handle_owned_vad_event(first[2]))
    await wait_until(lambda: any(call[0] == "seal_done" for call in session.calls))
    await clock.advance_to(60.0)
    assert not any(call[0] == "close" for call in session.calls)

    session.emit(
        STTProviderTurnTerminal(
            identity=session.requests[0].identity,
            outcome="final",
            text="complete",
            text_authority="authoritative",
            epoch_disposition="reuse",
        )
    )
    await end_task
    await wait_until(lambda: any(call[0] == "close" for call in session.calls))
    assert [event.text for event in emitted if isinstance(event, STTProviderTurnTerminal)] == [
        "complete"
    ]
    await engine.close()


@pytest.mark.asyncio
async def test_pending_owned_input_blocks_idle_until_queue_drains() -> None:
    ledger = PeerAudioSegmentLedger(activation_generation=1, settings=settings())
    first = segment_events(ledger, start_sample=900, now=9.0)
    session = ControlledScopedSession()
    session.terminal_on_seal = ("empty", "")
    clock = ControlledMonotonicClock()
    engine = ScopedRecognitionEngine(
        session_factory=lambda _settings, _epoch: asyncio.sleep(0, result=session),
        event_sink=lambda _event: None,
        watchdog_resolver=lambda _settings: watchdogs(),
        monotonic_clock=clock.now,
        sleep=clock.sleep,
    )

    await engine.handle_owned_vad_event(first[0])
    await engine.handle_owned_vad_event(first[2])
    await engine.observe_pending_source_work(pending=True)
    await clock.advance_to(90.0)
    assert not any(call[0] == "close" for call in session.calls)
    await engine.observe_pending_source_work(pending=False)
    await wait_until(lambda: any(call[0] == "close" for call in session.calls))
    await engine.close()


@pytest.mark.asyncio
async def test_idle_retirement_reopens_only_for_demand() -> None:
    first = segment_events(
        PeerAudioSegmentLedger(activation_generation=1, settings=settings()),
        start_sample=900,
        now=9.0,
    )
    second = segment_events(
        PeerAudioSegmentLedger(activation_generation=1, settings=settings()),
        start_sample=1000,
        now=10.0,
    )
    sessions: list[ControlledScopedSession] = []
    clock = ControlledMonotonicClock()

    async def factory(_settings: AudioSegmentSettingsSnapshot, _epoch: str):
        session = ControlledScopedSession()
        session.terminal_on_seal = ("empty", "")
        sessions.append(session)
        return session

    engine = ScopedRecognitionEngine(
        session_factory=factory,
        event_sink=lambda _event: None,
        watchdog_resolver=lambda _settings: watchdogs(),
        monotonic_clock=clock.now,
        sleep=clock.sleep,
    )

    await engine.handle_owned_vad_event(first[0])
    await engine.handle_owned_vad_event(first[2])
    await clock.advance_to(59.9)
    assert len(sessions) == 1
    assert not any(call[0] == "close" for call in sessions[0].calls)
    await clock.advance_to(60.0)
    await wait_until(lambda: any(call[0] == "close" for call in sessions[0].calls))
    await wait_until(lambda: engine.cleanup_debt == 0)
    await clock.advance_to(300.0)
    assert len(sessions) == 1
    await engine.handle_owned_vad_event(second[0])
    await engine.handle_owned_vad_event(second[2])
    assert len(sessions) == 2
    await engine.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider_id",
    ["custom_offline", "local_cpu_auto", "local_parakeet_v3", "local_qwen_gpu"],
)
async def test_disabled_session_lifetime_keeps_healthy_session_past_180s(
    provider_id: str,
) -> None:
    provider_settings = settings(provider_id)
    first = segment_events(
        PeerAudioSegmentLedger(activation_generation=1, settings=provider_settings),
        start_sample=900,
        now=9.0,
    )
    second = segment_events(
        PeerAudioSegmentLedger(activation_generation=1, settings=provider_settings),
        start_sample=1000,
        now=10.0,
    )
    sessions: list[ControlledScopedSession] = []
    clock = ControlledMonotonicClock()

    async def factory(_settings: AudioSegmentSettingsSnapshot, _epoch: str):
        session = ControlledScopedSession()
        session.terminal_on_seal = ("empty", "")
        sessions.append(session)
        return session

    engine = ScopedRecognitionEngine(
        session_factory=factory,
        event_sink=lambda _event: None,
        watchdog_resolver=lambda _settings: watchdogs(),
        monotonic_clock=clock.now,
        sleep=clock.sleep,
        session_lifetime_enabled=False,
    )

    await engine.handle_owned_vad_event(first[0])
    await engine.handle_owned_vad_event(first[2])
    await clock.advance_to(300.0)
    await engine.handle_owned_vad_event(second[0])
    await engine.handle_owned_vad_event(second[2])
    assert len(sessions) == 1
    assert not any(call[0] == "close" for call in sessions[0].calls)
    await engine.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("channel", ["self", "peer"])
async def test_ceiling_fails_active_turn_and_releases_successor_without_replaying_audio(
    channel: str,
) -> None:
    class ProviderCappedSession(ControlledScopedSession):
        max_session_age_s = 5.0

    first = segment_events(
        PeerAudioSegmentLedger(activation_generation=1, settings=settings()),
        start_sample=900,
        now=9.0,
    )
    second = segment_events(
        PeerAudioSegmentLedger(activation_generation=1, settings=settings()),
        start_sample=1000,
        now=10.0,
    )
    sessions: list[ControlledScopedSession] = []
    emitted: list[object] = []
    clock = ControlledMonotonicClock()

    async def factory(_settings: AudioSegmentSettingsSnapshot, _epoch: str):
        session = ProviderCappedSession()
        session.terminal_on_seal = ("empty", "")
        sessions.append(session)
        return session

    engine = ScopedRecognitionEngine(
        channel=channel,
        session_factory=factory,
        event_sink=emitted.append,
        watchdog_resolver=lambda _settings: watchdogs(max_session_age_s=8.0),
        monotonic_clock=clock.now,
        sleep=clock.sleep,
    )

    await engine.observe_source_activity(speech_observed=True)
    await engine.handle_owned_vad_event(first[0])
    await clock.advance_to(4.9)
    assert not any(call[0] == "close" for call in sessions[0].calls)
    successor_start = asyncio.create_task(engine.handle_owned_vad_event(second[0]))
    await asyncio.sleep(0)
    assert not successor_start.done()
    await clock.advance_to(5.0)
    await wait_until(lambda: any(call[0] == "close" for call in sessions[0].calls))
    failed = [event for event in emitted if isinstance(event, STTProviderTurnTerminal)]
    if channel == "peer":
        assert failed == []
        assert not successor_start.done()
    else:
        assert len(failed) == 1
    await engine.handle_owned_vad_event(first[2])
    failed = [event for event in emitted if isinstance(event, STTProviderTurnTerminal)]
    assert len(failed) == 1
    assert failed[0].outcome == "failed"
    assert failed[0].failure_reason == "provider_session_lifetime_exceeded"
    await wait_until(lambda: engine.cleanup_debt == 0)
    await successor_start
    await engine.handle_owned_vad_event(second[2])
    assert len(sessions) == 2
    assert len(sessions[1].requests) == 1
    assert len([event for event in emitted if isinstance(event, STTProviderTurnTerminal)]) == 2
    await engine.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("blocked_phase", "holds_cancellation"),
    [("begin", False), ("send", False), ("seal", False), ("send", True)],
)
async def test_session_ceiling_preempts_writes_and_retains_cleanup_ownership(
    blocked_phase: str, holds_cancellation: bool
) -> None:
    class CappedSession(ControlledScopedSession):
        async def send_turn_audio(self, *args, **kwargs) -> None:
            try:
                await super().send_turn_audio(*args, **kwargs)
            except asyncio.CancelledError:
                if not holds_cancellation:
                    raise
                await self.send_gate.wait()

    first = segment_events(
        PeerAudioSegmentLedger(activation_generation=1, settings=settings()),
        start_sample=900,
        now=9.0,
    )
    second = segment_events(
        PeerAudioSegmentLedger(activation_generation=1, settings=settings()),
        start_sample=1000,
        now=10.0,
    )
    session = CappedSession()
    successor = ControlledScopedSession()
    successor.terminal_on_seal = ("final", "recovered")
    sessions = iter((session, successor))
    clock = ControlledMonotonicClock()
    emitted: list[object] = []
    engine = ScopedRecognitionEngine(
        channel="peer",
        session_factory=lambda _settings, _epoch: asyncio.sleep(0, result=next(sessions)),
        event_sink=emitted.append,
        watchdog_resolver=lambda _settings: watchdogs(max_session_age_s=0.05, write_timeout_s=0.5),
        monotonic_clock=clock.now,
        sleep=clock.sleep,
    )
    pending = None
    try:
        await engine.observe_source_activity(speech_observed=True)
        if blocked_phase != "begin":
            await engine.handle_owned_vad_event(first[0])
        getattr(session, f"{blocked_phase}_gate").clear()
        event = first[{"begin": 0, "send": 1, "seal": 2}[blocked_phase]]
        call_count = len(session.calls)
        pending = asyncio.create_task(engine.handle_owned_vad_event(event))
        await wait_until(
            lambda: any(call[0] == blocked_phase for call in session.calls[call_count:])
        )
        await clock.advance_to(0.05)
        await asyncio.wait_for(pending, timeout=2.0)
        if blocked_phase != "seal":
            assert not any(isinstance(event, STTProviderTurnTerminal) for event in emitted)
            await engine.handle_owned_vad_event(first[2])
        failed = [event for event in emitted if isinstance(event, STTProviderTurnTerminal)]
        assert len(failed) == 1
        assert failed[0].failure_reason == "provider_session_lifetime_exceeded"
        if holds_cancellation:
            assert engine.cleanup_debt == 1
            assert ("stop",) not in session.calls
            session.send_gate.set()
        await wait_until(lambda: engine.cleanup_debt == 0)
        assert session.calls.count(("close",)) == 1
        await engine.handle_owned_vad_event(second[0])
        await engine.handle_owned_vad_event(second[2])
        terminals = [event for event in emitted if isinstance(event, STTProviderTurnTerminal)]
        assert [event.text for event in terminals] == ["", "recovered"]
        assert successor.requests[0].identity.segment == second[0].segment.identity
        assert terminals[0].identity.provider_epoch_id != terminals[1].identity.provider_epoch_id
    finally:
        session.begin_gate.set()
        session.send_gate.set()
        session.seal_gate.set()
        if pending is not None:
            await pending
        await engine.close()


@pytest.mark.asyncio
async def test_watchdog_ceiling_retires_idle_session_before_silence_deadline() -> None:
    first = segment_events(
        PeerAudioSegmentLedger(activation_generation=1, settings=settings()),
        start_sample=900,
        now=9.0,
    )
    session = ControlledScopedSession()
    session.terminal_on_seal = ("empty", "")
    clock = ControlledMonotonicClock()
    emitted: list[object] = []
    engine = ScopedRecognitionEngine(
        session_factory=lambda _settings, _epoch: asyncio.sleep(0, result=session),
        event_sink=emitted.append,
        watchdog_resolver=lambda _settings: watchdogs(max_session_age_s=4.0),
        monotonic_clock=clock.now,
        sleep=clock.sleep,
    )

    await engine.observe_source_activity(speech_observed=True)
    await engine.handle_owned_vad_event(first[0])
    await engine.handle_owned_vad_event(first[2])
    await clock.advance_to(3.9)
    assert not any(call[0] == "close" for call in session.calls)
    await clock.advance_to(4.0)
    await wait_until(lambda: any(call[0] == "close" for call in session.calls))
    assert [event.outcome for event in emitted if isinstance(event, STTProviderTurnTerminal)] == [
        "empty"
    ]
    await engine.close()


@pytest.mark.asyncio
async def test_ceiling_rotates_at_boundary_before_admitting_successor() -> None:
    first = segment_events(
        PeerAudioSegmentLedger(activation_generation=1, settings=settings()),
        start_sample=900,
        now=9.0,
    )
    second = segment_events(
        PeerAudioSegmentLedger(activation_generation=1, settings=settings()),
        start_sample=1000,
        now=10.0,
    )
    sessions: list[ControlledScopedSession] = []
    clock = ControlledMonotonicClock()

    async def factory(_settings: AudioSegmentSettingsSnapshot, _epoch: str):
        session = ControlledScopedSession()
        session.terminal_on_seal = ("empty", "")
        sessions.append(session)
        return session

    engine = ScopedRecognitionEngine(
        session_factory=factory,
        event_sink=lambda _event: None,
        watchdog_resolver=lambda _settings: watchdogs(max_session_age_s=5.0),
        monotonic_clock=clock.now,
        sleep=clock.sleep,
    )

    await engine.observe_source_activity(speech_observed=True)
    await engine.handle_owned_vad_event(first[0])
    await engine.handle_owned_vad_event(first[2])
    await clock.advance_to(4.8)
    await engine.handle_owned_vad_event(second[0])
    await engine.handle_owned_vad_event(second[2])
    assert len(sessions) == 2
    assert len(sessions[0].requests) == len(sessions[1].requests) == 1
    await wait_until(lambda: any(call[0] == "close" for call in sessions[0].calls))
    await engine.close()


def test_whitespace_stable_contributions_match_normalized_terminal_ranges() -> None:
    request_identity = STTProviderTurnIdentity(
        segment=PeerAudioSegmentLedger(
            activation_generation=1,
            settings=settings(),
        )
        .observe_vad_event(
            SpeechStart(
                uuid4(),
                np.empty((0,), dtype=np.float32),
                np.ones(1, dtype=np.float32),
            ),
            now_monotonic_s=0.0,
        )
        .segment.identity,
        provider_epoch_id="epoch",
        provider_turn_id="turn",
    )
    normalizer = STTScopedTurnNormalizer(request_identity)
    ledger = STTContributionConsumptionLedger()
    stable = normalizer.apply_update(
        STTProviderTurnUpdate(
            identity=request_identity,
            sequence=1,
            stability="stable",
            assembly="replace",
            text="  hello 世界  ",
            final_language_runs=(
                FinalLanguageRun(text="  hello ", language="en"),
                FinalLanguageRun(text="世界  ", language="ja"),
            ),
        )
    )
    assert stable is not None
    terminal = normalizer.apply_terminal(
        STTProviderTurnTerminal(
            identity=request_identity,
            outcome="final",
            text="  hello 世界  ",
            final_language_runs=(
                FinalLanguageRun(text="  hello ", language="en"),
                FinalLanguageRun(text="世界  ", language="ja"),
            ),
            text_authority="authoritative",
        )
    )

    assert stable.text == "hello 世界"
    assert terminal.text == "hello 世界"
    assert terminal.included_contributions == (stable.contribution,)
    assert ledger.consume(stable) == "hello 世界"
    assert ledger.consume(terminal) == ""
    assert "".join(run.text for run in terminal.final_language_runs) == terminal.text


def test_append_separators_and_cumulative_raw_whitespace_preserve_exact_suffixes() -> None:
    ledger = PeerAudioSegmentLedger(activation_generation=1, settings=settings())
    identity = STTProviderTurnIdentity(
        segment=segment_events(ledger, start_sample=1750, now=17.5)[0].segment.identity,
        provider_epoch_id="epoch",
        provider_turn_id="spaced-turn",
    )
    consumption = STTContributionConsumptionLedger()

    trailing = STTScopedTurnNormalizer(identity)
    first = trailing.apply_update(
        STTProviderTurnUpdate(
            identity=identity,
            sequence=1,
            stability="stable",
            assembly="append",
            text="hello ",
            final_language_runs=(FinalLanguageRun(text="hello ", language="en"),),
        )
    )
    second = trailing.apply_update(
        STTProviderTurnUpdate(
            identity=identity,
            sequence=2,
            stability="stable",
            assembly="append",
            text="世界",
            final_language_runs=(FinalLanguageRun(text="世界", language="ja"),),
        )
    )
    assert first is not None and second is not None
    assert (first.text, second.text) == ("hello", "hello 世界")
    assert first.contribution is not None and (
        first.contribution.text_start,
        first.contribution.text_end,
    ) == (0, 5)
    assert second.contribution is not None and (
        second.contribution.text_start,
        second.contribution.text_end,
    ) == (5, 8)
    assert consumption.consume(first) == "hello"
    trailing_terminal = trailing.apply_terminal(
        STTProviderTurnTerminal(
            identity=identity,
            outcome="final",
            text="hello 世界",
            final_language_runs=(
                FinalLanguageRun(text="hello ", language="en"),
                FinalLanguageRun(text="世界", language="ja"),
            ),
            text_authority="authoritative",
        )
    )
    assert consumption.consume(trailing_terminal) == " 世界"
    assert [item.contribution_id for item in trailing_terminal.included_contributions] == [
        "spaced-turn:1",
        "spaced-turn:2",
    ]

    cumulative_identity = replace(identity, provider_turn_id="cumulative-turn")
    cumulative = STTScopedTurnNormalizer(cumulative_identity)
    cumulative_first = cumulative.apply_update(
        STTProviderTurnUpdate(
            identity=cumulative_identity,
            sequence=1,
            stability="stable",
            assembly="replace",
            text="  hello ",
        )
    )
    cumulative_second = cumulative.apply_update(
        STTProviderTurnUpdate(
            identity=cumulative_identity,
            sequence=2,
            stability="stable",
            assembly="replace",
            text="  hello world  ",
        )
    )
    assert cumulative_first is not None and cumulative_second is not None
    assert (cumulative_first.text, cumulative_second.text) == ("hello", "hello world")
    assert cumulative_second.contribution is not None
    assert (
        cumulative_second.contribution.text_start,
        cumulative_second.contribution.text_end,
    ) == (5, 11)
    cumulative_terminal = cumulative.apply_terminal(
        STTProviderTurnTerminal(
            identity=cumulative_identity,
            outcome="final",
            text="  hello world  ",
            text_authority="authoritative",
        )
    )
    assert STTContributionConsumptionLedger().consume(cumulative_terminal) == "hello world"


@pytest.mark.asyncio
@pytest.mark.parametrize("blocked_phase", ["open", "write", "final"])
async def test_abort_immediately_invalidates_authority_while_native_phase_is_blocked(
    blocked_phase: str,
) -> None:
    provider_settings = settings()
    ledger = PeerAudioSegmentLedger(activation_generation=1, settings=provider_settings)
    start, _chunk, end = segment_events(ledger, start_sample=2000, now=20.0)
    session = ControlledScopedSession()
    factory_gate = asyncio.Event()
    factory_gate.set()
    if blocked_phase == "write":
        session.send_gate.clear()
    emitted: list[object] = []

    async def factory(_settings: AudioSegmentSettingsSnapshot, _epoch: str):
        await factory_gate.wait()
        return session

    if blocked_phase == "open":
        factory_gate.clear()
    engine = ScopedRecognitionEngine(
        session_factory=factory,
        event_sink=emitted.append,
        watchdog_resolver=lambda _settings: watchdogs(
            readiness_timeout_s=1.0,
            write_timeout_s=1.0,
            final_timeout_s=1.0,
        ),
    )
    operation = asyncio.create_task(engine.handle_owned_vad_event(start))
    if blocked_phase == "open":
        await asyncio.sleep(0)
    else:
        await wait_until(lambda: bool(session.calls))
    if blocked_phase == "final":
        await operation
        operation = asyncio.create_task(engine.handle_owned_vad_event(end))
        await wait_until(lambda: ("seal_done", session.requests[0].identity) in session.calls)

    await asyncio.wait_for(engine.abort_for_toggle_off(), timeout=0.05)
    assert engine.is_at_turn_boundary
    assert engine.scoped_settings_scope is None
    assert not any(
        isinstance(event, STTProviderTurnTerminal) and event.outcome == "final" for event in emitted
    )

    factory_gate.set()
    session.send_gate.set()
    await asyncio.wait_for(operation, timeout=1.0)
    await engine.close()


@pytest.mark.asyncio
async def test_abort_during_first_start_payload_prevents_second_payload_write() -> None:
    ledger = PeerAudioSegmentLedger(activation_generation=1, settings=settings())
    start, _chunk, _end = segment_events(ledger, start_sample=2050, now=20.5)
    session = ControlledScopedSession()
    session.send_gate.clear()
    emitted: list[object] = []
    engine = ScopedRecognitionEngine(
        session_factory=lambda _settings, _epoch: asyncio.sleep(0, result=session),
        event_sink=emitted.append,
        watchdog_resolver=lambda _settings: watchdogs(write_timeout_s=1.0),
    )

    start_task = asyncio.create_task(engine.handle_owned_vad_event(start))
    await wait_until(lambda: sum(call[0] == "send" for call in session.calls) == 1)
    await engine.abort_for_toggle_off()
    session.send_gate.set()
    await start_task
    await wait_until(lambda: any(call[0] == "stop" for call in session.calls))

    assert sum(call[0] == "send" for call in session.calls) == 1
    assert sum(call[0] == "send_done" for call in session.calls) == 1
    assert sum(call[0] == "abort" for call in session.calls) == 1
    terminals = [item for item in emitted if isinstance(item, STTProviderTurnTerminal)]
    assert len(terminals) == 1
    assert terminals[0].outcome == "cancelled"
    await engine.close()


@pytest.mark.asyncio
async def test_bound_event_sink_does_not_block_speech_end_on_downstream_delivery() -> None:
    ledger = PeerAudioSegmentLedger(activation_generation=1, settings=settings())
    start, _chunk, end = segment_events(ledger, start_sample=1200, now=12.0)
    session = ControlledScopedSession()
    session.terminal_on_seal = ("final", "ready")
    sink_started = asyncio.Event()
    release_sink = asyncio.Event()
    emitted: list[STTProviderTurnEvent] = []

    async def blocked_sink(event: STTProviderTurnEvent) -> None:
        sink_started.set()
        await release_sink.wait()
        emitted.append(event)

    engine = ScopedRecognitionEngine(
        session_factory=lambda _settings, _epoch: asyncio.sleep(0, result=session),
        watchdog_resolver=lambda _settings: watchdogs(),
    )
    engine.bind_event_sink(blocked_sink)

    await engine.handle_owned_vad_event(start)
    await asyncio.wait_for(engine.handle_owned_vad_event(end), timeout=0.2)
    await asyncio.wait_for(sink_started.wait(), timeout=0.2)
    assert emitted == []

    release_sink.set()
    await asyncio.wait_for(engine.wait_for_event_ingress_drain(), timeout=0.2)
    assert len(emitted) == 1
    assert isinstance(emitted[0], STTProviderTurnTerminal)
    await engine.close()


def test_stable_contribution_provenance_preserves_suffix_and_detects_contradiction() -> None:
    ledger = PeerAudioSegmentLedger(activation_generation=1, settings=settings())
    identity = STTProviderTurnIdentity(
        segment=segment_events(ledger, start_sample=1300, now=13.0)[0].segment.identity,
        provider_epoch_id="epoch",
        provider_turn_id="turn",
    )
    normalizer = STTScopedTurnNormalizer(identity)
    consumption = STTContributionConsumptionLedger()
    stable_a = normalizer.apply_update(
        STTProviderTurnUpdate(
            identity=identity,
            sequence=1,
            stability="stable",
            assembly="append",
            text="A",
        )
    )
    stable_b = normalizer.apply_update(
        STTProviderTurnUpdate(
            identity=identity,
            sequence=2,
            stability="stable",
            assembly="append",
            text="B",
        )
    )
    assert stable_a is not None and consumption.consume(stable_a) == "A"
    assert stable_b is not None
    terminal = normalizer.apply_terminal(
        STTProviderTurnTerminal(
            identity=identity,
            outcome="final",
            text="AB",
            text_authority="authoritative",
        )
    )
    assert consumption.consume(terminal) == "B"
    assert consumption.consume(terminal) == ""
    assert [item.contribution_id for item in terminal.included_contributions] == [
        "turn:1",
        "turn:2",
    ]
    other_ledger = PeerAudioSegmentLedger(activation_generation=2, settings=settings())
    other_identity = STTProviderTurnIdentity(
        segment=segment_events(other_ledger, start_sample=1500, now=15.0)[0].segment.identity,
        provider_epoch_id="other-epoch",
        provider_turn_id="other-turn",
    )
    assert (
        consumption.consume(
            STTProviderTurnTerminal(
                identity=other_identity,
                outcome="final",
                text="terminal-only",
                text_authority="authoritative",
            )
        )
        == "terminal-only"
    )

    inconsistent = STTScopedTurnNormalizer(identity)
    inconsistent.apply_update(
        STTProviderTurnUpdate(
            identity=identity,
            sequence=1,
            stability="stable",
            assembly="replace",
            text="stable",
        )
    )
    with pytest.raises(STTNormalizationError, match="provider_stable_prefix_inconsistent"):
        inconsistent.apply_update(
            STTProviderTurnUpdate(
                identity=identity,
                sequence=2,
                stability="stable",
                assembly="replace",
                text="changed",
            )
        )


def test_terminal_consumption_includes_unpublished_authoritative_tail_once() -> None:
    identity = STTProviderTurnIdentity(
        segment=segment_events(
            PeerAudioSegmentLedger(activation_generation=1, settings=settings()),
            start_sample=1600,
            now=16.0,
        )[0].segment.identity,
        provider_epoch_id="epoch",
        provider_turn_id="tail-turn",
    )
    normalizer = STTScopedTurnNormalizer(identity)
    stable = normalizer.apply_update(
        STTProviderTurnUpdate(
            identity=identity,
            sequence=1,
            stability="stable",
            assembly="append",
            text="A",
        )
    )
    assert stable is not None
    terminal = normalizer.apply_terminal(
        STTProviderTurnTerminal(
            identity=identity,
            outcome="final",
            text="AB",
            text_authority="authoritative",
        )
    )
    assert [
        (item.contribution_id, item.text_start, item.text_end)
        for item in terminal.included_contributions
    ] == [("tail-turn:1", 0, 1)]

    early = STTContributionConsumptionLedger()
    assert early.consume(stable) == "A"
    assert early.consume(terminal) == "B"
    assert early.consume(terminal) == ""

    terminal_only = STTContributionConsumptionLedger()
    assert terminal_only.consume(terminal) == "AB"
    assert terminal_only.consume(terminal) == ""


@pytest.mark.asyncio
async def test_self_like_binding_has_no_time_cut_and_fails_at_retained_pcm_bound() -> None:
    ledger = PeerAudioSegmentLedger(activation_generation=1, settings=settings())
    start, _chunk, end = segment_events(ledger, start_sample=1400, now=14.0)
    session = ControlledScopedSession()
    emitted: list[object] = []
    engine = ScopedRecognitionEngine(
        channel="self",
        session_factory=lambda _settings, _epoch: asyncio.sleep(0, result=session),
        event_sink=emitted.append,
        watchdog_resolver=lambda _settings: watchdogs(),
        retention_profile_resolver=lambda _settings: STTRetentionProfile(
            max_retained_samples=5,
            max_retained_bytes=20,
            release_after_write=False,
        ),
    )

    await engine.handle_owned_vad_event(start)

    terminals = [item for item in emitted if isinstance(item, STTProviderTurnTerminal)]
    assert len(terminals) == 1
    assert terminals[0].failure_reason == "buffer_exhausted"
    assert engine.retention_snapshot.retained_samples == 0
    await engine.handle_owned_vad_event(end)

    assert session.requests[0].channel == "self"
    assert session.requests[0].identity.segment == start.segment.identity
    assert len([item for item in emitted if isinstance(item, STTProviderTurnTerminal)]) == 1
    assert not any(call[0] == "seal" for call in session.calls)
    await engine.close()


@pytest.mark.asyncio
async def test_shared_budget_counts_old_local_retention_against_new_scoped_engine() -> None:
    ledger = PeerAudioSegmentLedger(activation_generation=1, settings=settings())
    old_start, _old_chunk, old_end = segment_events(
        ledger,
        start_sample=1500,
        now=15.0,
    )
    new_start, _new_chunk, _new_end = segment_events(
        ledger,
        start_sample=1600,
        now=16.0,
    )
    budget = AudioRetentionBudget(
        capacity_bytes=100,
        capacity_sample_equivalents=21,
    )

    def bind(owned: OwnedVadEvent) -> tuple[OwnedVadEvent, object]:
        dispatcher_owner = object()
        retained_bytes = sum(
            samples.nbytes
            for samples in (
                getattr(owned.event, "pre_roll", None),
                getattr(owned.event, "chunk", None),
            )
            if samples is not None
        )
        assert budget.try_reserve(
            dispatcher_owner,
            retained_bytes,
            sample_equivalents=retained_bytes // 4,
        )
        return (
            replace(
                owned,
                retention=AudioRetentionBinding(
                    budget=budget,
                    dispatcher_owner=dispatcher_owner,
                ),
            ),
            dispatcher_owner,
        )

    profile = STTRetentionProfile(
        max_retained_samples=100,
        max_retained_bytes=400,
        release_after_write=False,
        retained_bytes_per_sample=4,
    )
    old_session = ControlledScopedSession()
    old_session.terminal_on_seal = ("empty", "")
    old_engine = ScopedRecognitionEngine(
        channel="self",
        session_factory=lambda _settings, _epoch: asyncio.sleep(
            0,
            result=old_session,
        ),
        watchdog_resolver=lambda _settings: watchdogs(),
        retention_profile_resolver=lambda _settings: profile,
    )
    new_session = ControlledScopedSession()
    new_terminals: list[STTProviderTurnTerminal] = []
    new_engine = ScopedRecognitionEngine(
        channel="self",
        session_factory=lambda _settings, _epoch: asyncio.sleep(
            0,
            result=new_session,
        ),
        event_sink=lambda event: (
            new_terminals.append(event) if isinstance(event, STTProviderTurnTerminal) else None
        ),
        watchdog_resolver=lambda _settings: watchdogs(),
        retention_profile_resolver=lambda _settings: profile,
    )

    bound_old, old_dispatcher = bind(old_start)
    await old_engine.handle_owned_vad_event(bound_old)
    budget.release(old_dispatcher)
    assert budget.used_bytes == 24

    bound_new, new_dispatcher = bind(new_start)
    await new_engine.handle_owned_vad_event(bound_new)
    budget.release(new_dispatcher)
    assert new_terminals[-1].failure_reason == "buffer_exhausted"
    assert budget.used_bytes == 24
    assert budget.high_water_bytes == 64
    assert budget.high_water_sample_equivalents == 18
    await old_engine.handle_owned_vad_event(old_end)
    await wait_until(lambda: budget.used_bytes == 0)
    await new_engine.close()
    await old_engine.close()


@pytest.mark.asyncio
async def test_streaming_scoped_engine_releases_native_copy_after_each_write() -> None:
    ledger = PeerAudioSegmentLedger(activation_generation=1, settings=settings())
    start, _chunk, end = segment_events(ledger, start_sample=1700, now=17.0)
    budget = AudioRetentionBudget(
        capacity_bytes=60,
        capacity_sample_equivalents=100,
    )
    dispatcher_owner = object()
    assert budget.try_reserve(
        dispatcher_owner,
        24,
        sample_equivalents=6,
    )
    start = replace(
        start,
        retention=AudioRetentionBinding(
            budget=budget,
            dispatcher_owner=dispatcher_owner,
        ),
    )
    session = ControlledScopedSession()
    session.terminal_on_seal = ("empty", "")
    engine = ScopedRecognitionEngine(
        channel="self",
        session_factory=lambda _settings, _epoch: asyncio.sleep(0, result=session),
        watchdog_resolver=lambda _settings: watchdogs(),
        retention_profile_resolver=lambda _settings: STTRetentionProfile(
            max_retained_samples=100,
            max_retained_bytes=400,
            release_after_write=True,
            retained_bytes_per_sample=4,
        ),
    )

    await engine.handle_owned_vad_event(start)
    assert budget.used_bytes == 24
    assert budget.high_water_bytes == 48
    budget.release(dispatcher_owner)
    assert budget.used_bytes == 0
    await engine.handle_owned_vad_event(end)
    await engine.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("native_outcome", ["final", "failed", None])
async def test_session_ceiling_overrides_cached_terminal_before_peer_source_seal(
    native_outcome: str | None,
) -> None:
    first = segment_events(
        PeerAudioSegmentLedger(activation_generation=1, settings=settings()),
        start_sample=1900,
        now=19.0,
    )
    session = ControlledScopedSession()
    clock = ControlledMonotonicClock()
    emitted: list[object] = []
    engine = ScopedRecognitionEngine(
        channel="peer",
        session_factory=lambda _settings, _epoch: asyncio.sleep(0, result=session),
        event_sink=emitted.append,
        watchdog_resolver=lambda _settings: watchdogs(max_session_age_s=5.0),
        monotonic_clock=clock.now,
        sleep=clock.sleep,
    )
    try:
        await engine.observe_source_activity(speech_observed=True)
        await engine.handle_owned_vad_event(first[0])
        identity = session.requests[0].identity
        if native_outcome is not None:
            session.emit(
                STTProviderTurnTerminal(
                    identity=identity,
                    outcome=native_outcome,
                    text="early result" if native_outcome == "final" else "",
                    text_authority="authoritative" if native_outcome == "final" else "none",
                    failure_reason=None if native_outcome == "final" else "transport_failed",
                    epoch_disposition="retire",
                )
            )
        session.emit(
            STTProviderEpochEnded(
                provider_epoch_id=identity.provider_epoch_id,
                orderly=False,
                reason="transport_failed",
            )
        )
        await wait_until(lambda: any(isinstance(event, STTProviderEpochEnded) for event in emitted))
        await clock.advance_to(5.0)
        await wait_until(lambda: ("close",) in session.calls)
        assert not any(isinstance(event, STTProviderTurnTerminal) for event in emitted)
        await engine.handle_owned_vad_event(first[2])
        terminals = [event for event in emitted if isinstance(event, STTProviderTurnTerminal)]
        assert len(terminals) == 1
        assert terminals[0].outcome == "failed"
        assert terminals[0].failure_reason == "provider_session_lifetime_exceeded"
        assert terminals[0].text == ""
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_timeout_diagnostics_keep_original_epoch_and_frozen_watchdog_after_config_change(
    caplog: pytest.LogCaptureFixture,
) -> None:
    first_start, _first_chunk, first_end = segment_events(
        PeerAudioSegmentLedger(activation_generation=41, settings=settings("soniox")),
        start_sample=2000,
        now=20.0,
    )
    second_start, _second_chunk, second_end = segment_events(
        PeerAudioSegmentLedger(activation_generation=42, settings=settings("soniox")),
        start_sample=2100,
        now=21.0,
    )
    first = ControlledScopedSession()
    second = ControlledScopedSession()
    second.terminal_on_seal = ("final", "next")
    sessions = [first, second]
    clock = ControlledMonotonicClock(value=10.0)
    configured_timeout_s = 0.012
    emitted: list[object] = []

    async def factory(_settings: AudioSegmentSettingsSnapshot, _epoch: str):
        return sessions.pop(0)

    engine = ScopedRecognitionEngine(
        session_factory=factory,
        event_sink=emitted.append,
        watchdog_resolver=lambda _settings: watchdogs(final_timeout_s=configured_timeout_s),
        monotonic_clock=clock.now,
        sleep=clock.sleep,
    )
    with caplog.at_level(logging.INFO, logger="puripuly_heart.core.stt.scoped_engine"):
        await engine.handle_owned_vad_event(first_start)
        first_identity = first.requests[0].identity
        configured_timeout_s = 0.2
        ending = asyncio.create_task(engine.handle_owned_vad_event(first_end))
        await wait_until(
            lambda: engine._turn is not None and engine._turn.final_wait_started_at_s is not None
        )
        clock.value = 10.015625
        await ending
        await wait_until(lambda: ("close",) in first.calls)

        second_start_task = asyncio.create_task(engine.handle_owned_vad_event(second_start))
        await wait_until(
            lambda: any(
                deadline <= clock.value + 0.002 and not future.done()
                for deadline, future in clock.sleepers
            )
        )
        await clock.advance_to(clock.value + 0.001)
        await second_start_task
        second_identity = second.requests[0].identity
        assert second_identity.provider_epoch_id != first_identity.provider_epoch_id
        second.emit(
            STTProviderEpochEnded(
                provider_epoch_id=first_identity.provider_epoch_id,
                orderly=False,
                reason="soniox_receive_failed",
            )
        )
        second.emit(
            STTProviderTurnTerminal(
                identity=first_identity,
                outcome="cancelled",
                text_authority="none",
                failure_reason="toggle_off",
            )
        )
        await engine.handle_owned_vad_event(second_end)
        await engine.close()

    records = [
        dict(token.split("=", 1) for token in record.getMessage().split()[2:])
        for record in caplog.records
        if record.getMessage().startswith("[Recognition] terminal ")
    ]
    assert len(records) == 2
    by_utterance = {record["utterance_id"]: record for record in records}
    original = by_utterance[str(first_identity.segment.segment_id)]
    successor = by_utterance[str(second_identity.segment.segment_id)]
    assert original["epoch"] == first_identity.provider_epoch_id
    assert original["turn"] == first_identity.provider_turn_id
    assert original["activation_generation"] == "41"
    assert original["cause"] == "provider_final_timeout"
    assert original["final_timeout_ms"] == "12"
    assert original["failure_retryable"] == "1"
    assert original["recovery_pending"] == "1"
    assert int(original["final_wait_ms"]) == 15
    assert successor["epoch"] == second_identity.provider_epoch_id
    assert successor["turn"] == second_identity.provider_turn_id
    assert successor["activation_generation"] == "42"
    assert successor["outcome"] == "final"
    assert successor["cause"] == "none"
    assert successor["final_timeout_ms"] == "200"
    assert successor["recovery_pending"] == "0"
    terminals = [event for event in emitted if isinstance(event, STTProviderTurnTerminal)]
    assert [(event.identity, event.outcome) for event in terminals] == [
        (first_identity, "failed"),
        (second_identity, "final"),
    ]


@pytest.mark.parametrize(
    ("reason", "expected"),
    [
        ("soniox_write_failed:private exception and transcript", "soniox_write_failed"),
        ("soniox_keepalive_failed", "soniox_keepalive_failed"),
        ("soniox_connection_ended", "soniox_connection_ended"),
        ("soniox_stream_finished", "soniox_stream_finished"),
        ("soniox_idle_authoritative_text", "soniox_idle_authoritative_text"),
        ("soniox_token_buffer_overflow", "soniox_token_buffer_overflow"),
        ("toggle_off", "toggle_off"),
        ("server_message:soniox_receive_failed", "unclassified"),
        ("provider_reported", "unclassified"),
        (None, "none"),
    ],
)
def test_recognition_cause_exposes_only_owned_reason_prefixes(
    reason: str | None,
    expected: str,
) -> None:
    assert recognition_cause(reason) == expected


class ControlledStreamSession(ControlledScopedSession):
    accepts_stream_input = True
    independent_recognition_units = True

    def __init__(self) -> None:
        from puripuly_heart.core.stt.stream_input import STTStreamInputMap

        super().__init__()
        self.mapping = STTStreamInputMap()
        self.stream: RecognitionStreamIdentity | None = None
        self.audio: list[tuple[bytes, tuple[AudioCaptureSpan, ...]]] = []
        self.failure: Exception | None = None
        self.write_entered = asyncio.Event()
        self.write_cancelled = asyncio.Event()

    async def begin_stream(self, stream: RecognitionStreamIdentity) -> None:
        self.stream = stream

    async def begin_turn(self, request: STTProviderTurnRequest) -> None:
        await super().begin_turn(request)
        await self.begin_stream(
            RecognitionStreamIdentity(
                request.channel,
                request.identity.segment.activation_generation,
                request.identity.segment.capture_epoch,
                request.identity.provider_epoch_id,
                request.identity.settings_scope,
            )
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
        await self.send_stream_audio(pcm16le, source_ranges=source_ranges)

    async def send_stream_audio(
        self, pcm16le: bytes, *, source_ranges: tuple[AudioCaptureSpan, ...]
    ) -> None:
        self.write_entered.set()
        try:
            await self.send_gate.wait()
        except asyncio.CancelledError:
            self.write_cancelled.set()
            raise
        if self.failure is not None:
            raise self.failure
        pcm, write = self.mapping.prepare(pcm16le, source_ranges)
        if write is not None:
            self.mapping.commit(write)
            self.audio.append((pcm, source_ranges))

    def recognition_source_covers(self, ranges: tuple[AudioCaptureSpan, ...]) -> bool:
        return self.mapping.covers(ranges)

    async def end_stream(self, *, reason: str) -> None:
        self.calls.append(("end_stream", reason))


def stream_input(
    ledger: PeerAudioSegmentLedger,
    start: int,
    end: int,
    *,
    speech_observed: bool = False,
    **kwargs: Any,
) -> OwnedStreamInput:
    return OwnedStreamInput(
        CaptureStreamInput(
            np.full(end - start, 0.4, dtype=np.float32),
            (span(start, start, end),),
            speech_observed=speech_observed,
        ),
        ledger,
        ledger.settings,
        ledger.activation_generation,
        **kwargs,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("channel", ["self", "peer"])
@pytest.mark.parametrize("successor_epoch", [1, 2])
async def test_queued_capture_discontinuity_retires_stream_after_successor_arrives(
    channel: str, successor_epoch: int
) -> None:
    from types import SimpleNamespace

    from puripuly_heart.core.runtime import peer_channel, self_capture

    ledger = PeerAudioSegmentLedger(activation_generation=3, settings=settings("gemini_transcribe"))
    start, _, _ = segment_events(ledger, start_sample=0, now=0)
    session = ControlledStreamSession()
    emitted: list[object] = []
    engine = ScopedRecognitionEngine(
        lambda _settings, _epoch: asyncio.sleep(0, result=session),
        channel=channel,
        event_sink=emitted.append,
        watchdog_resolver=lambda _: watchdogs(write_timeout_s=1.0),
    )
    runtime = SimpleNamespace(
        segment_ledger=ledger,
        _current_stream_capture_epoch=None,
        is_current_generation=lambda generation: generation == 3,
    )
    if channel == "self":
        guard = self_capture._GenerationGuardedVadSink(
            sink=engine,
            owner=runtime,
            capture_generation=self_capture._CaptureGeneration(3),
            ledger=ledger,
            retention_budget=AudioRetentionBudget(
                capacity_bytes=4096, capacity_sample_equivalents=1024
            ),
        )
    else:
        ready = asyncio.Event()
        ready.set()
        guard = peer_channel._GenerationGuardedVadSink(
            sink=engine,
            runtime=runtime,
            capture_generation=peer_channel._CaptureGeneration(3),
            provider_ingress_ready=ready,
        )
    try:
        await engine.handle_owned_vad_event(start)
        session.send_gate.clear()
        session.write_entered.clear()
        await guard.handle_stream_input(stream_input(ledger, 6, 10).event)
        await asyncio.wait_for(session.write_entered.wait(), timeout=1.0)
        await guard.handle_stream_input(
            CaptureStreamInput(
                np.empty((0,), dtype=np.float32), (), boundary_reason="source_discontinuity"
            )
        )
        await guard.handle_stream_input(
            CaptureStreamInput(
                np.full(4, 0.4, dtype=np.float32),
                (replace(span(3, 10, 14), capture_epoch=successor_epoch),),
            )
        )
        session.send_gate.set()
        await asyncio.wait_for(guard.finish(), timeout=1.0)
        await wait_until(
            lambda: any(
                isinstance(event, STTProviderInputTerminal)
                and event.outcome == "cancelled"
                and event.failure_reason == "source_discontinuity"
                for event in emitted
            )
        )
        await wait_until(lambda: ("close",) in session.calls)
        assert [
            (capture.capture_epoch, capture.source_start_sample, capture.source_end_sample)
            for _, ranges in session.audio
            for capture in ranges
        ] == [(1, 0, 2), (1, 2, 6), (1, 6, 10)]
        assert guard.retention_budget.used_bytes == 0
    finally:
        session.send_gate.set()
        await guard.abort()
        await engine.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["timeout", "error"])
async def test_continuous_stream_write_failure_recovers_unsent_frames_and_new_finals(
    failure: str,
) -> None:
    ledger = PeerAudioSegmentLedger(activation_generation=3, settings=settings("gemini_transcribe"))
    first_start, _, _ = segment_events(ledger, start_sample=0, now=0)
    sessions: list[ControlledStreamSession] = []
    emitted: list[object] = []

    async def factory(_settings, _epoch):
        session = ControlledStreamSession()
        sessions.append(session)
        return session

    engine = ScopedRecognitionEngine(
        factory,
        channel="self",
        event_sink=emitted.append,
        watchdog_resolver=lambda _: watchdogs(write_timeout_s=0.01),
    )
    try:
        await engine.handle_owned_vad_event(first_start)
        old_stream = sessions[0].stream
        assert old_stream is not None
        sessions[0].write_entered.clear()
        if failure == "timeout":
            sessions[0].send_gate.clear()
        else:
            sessions[0].failure = OSError("wire failed")
        await engine.handle_stream_input(stream_input(ledger, 6, 10, speech_observed=True))
        if failure == "timeout":
            await wait_until(sessions[0].write_cancelled.is_set)
        await engine.handle_stream_input(stream_input(ledger, 6, 10))
        assert len(sessions) == 1
        await engine.handle_stream_input(stream_input(ledger, 8, 14))
        assert len(sessions) == 2
        recovered = sessions[1]
        assert recovered.requests == []
        assert [
            (s.normalized_start_sample, s.normalized_end_sample)
            for _, ranges in recovered.audio
            for s in ranges
        ] == [(10, 14)]
        new_stream = recovered.stream
        assert new_stream is not None and new_stream != old_stream
        assert not engine.is_current_recognition_stream(old_stream)
        assert engine.is_current_recognition_stream(new_stream)
        recovered.emit(
            STTRecognitionUnit(RecognitionUnitIdentity(old_stream, uuid4(), 1), "stale final")
        )
        recovered.emit(
            STTRecognitionUnit(RecognitionUnitIdentity(new_stream, uuid4(), 1), "recovered final")
        )
        await wait_until(lambda: any(isinstance(e, STTRecognitionUnitTerminal) for e in emitted))
        finals = [e for e in emitted if isinstance(e, STTRecognitionUnitTerminal)]
        assert [e.unit.text for e in finals] == ["recovered final"]
        assert finals[0].unit.identity.stream == new_stream
        assert finals[0].unit.estimated_last_speech_at is None
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_stream_recovery_connection_failure_exhausts_once_not_on_every_frame() -> None:
    ledger = PeerAudioSegmentLedger(activation_generation=3, settings=settings("gemini_transcribe"))
    start, _, _ = segment_events(ledger, start_sample=0, now=0)
    session = ControlledStreamSession()
    opens = 0
    failures: list[Exception] = []

    async def factory(_settings, _epoch):
        nonlocal opens
        opens += 1
        if opens > 1:
            raise OSError("connection unavailable")
        return session

    engine = ScopedRecognitionEngine(
        factory,
        channel="self",
        watchdog_resolver=lambda _: watchdogs(write_timeout_s=0.01),
        terminal_failure_sink=failures.append,
    )
    try:
        await engine.handle_owned_vad_event(start)
        session.send_gate.clear()
        await engine.handle_stream_input(stream_input(ledger, 6, 10))
        for left in range(10, 34, 4):
            await engine.handle_stream_input(stream_input(ledger, left, left + 4))
        await wait_until(lambda: len(failures) == 1)
        assert opens == 3
        assert str(failures[0]) == "provider_recovery_exhausted"
    finally:
        await engine.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalidation",
    [
        "stopped",
        "toggle_off",
        "muted",
        "source_discontinuity",
        "settings",
        "activation",
        "capture",
        "cancelled",
    ],
)
async def test_inflight_stream_recovery_cannot_revive_invalidated_capture(
    invalidation: str,
) -> None:
    ledger = PeerAudioSegmentLedger(activation_generation=3, settings=settings("gemini_transcribe"))
    start, _, _ = segment_events(ledger, start_sample=0, now=0)
    sessions = [ControlledStreamSession(), ControlledStreamSession()]
    opening = asyncio.Event()
    release = asyncio.Event()
    capture_current = True
    opens = 0

    async def factory(_settings, _epoch):
        nonlocal opens
        opens += 1
        if opens == 2:
            opening.set()
            await release.wait()
        return sessions[opens - 1]

    engine = ScopedRecognitionEngine(
        factory,
        channel="self",
        watchdog_resolver=lambda _: watchdogs(write_timeout_s=0.01),
    )
    try:
        await engine.handle_owned_vad_event(start)
        sessions[0].send_gate.clear()
        await engine.handle_stream_input(stream_input(ledger, 6, 10))
        pending = asyncio.create_task(
            engine.handle_stream_input(
                stream_input(ledger, 10, 14, is_current=lambda: capture_current)
            )
        )
        await opening.wait()
        if invalidation == "settings":
            ledger.rebind(
                activation_generation=3, settings=settings("gemini_transcribe", signature="b")
            )
        elif invalidation == "activation":
            ledger.rebind(activation_generation=4, settings=ledger.settings)
        elif invalidation == "capture":
            capture_current = False
        elif invalidation == "cancelled":
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
        else:
            await engine.abort(reason=invalidation)
        release.set()
        await asyncio.gather(pending, return_exceptions=True)
        await wait_until(lambda: ("close",) in sessions[1].calls)
        assert sessions[1].stream is None
        assert sessions[1].audio == []
        assert sessions[1].requests == []
        await engine.handle_stream_input(stream_input(ledger, 14, 18))
        assert opens == 2
    finally:
        release.set()
        await engine.close()


@pytest.mark.asyncio
async def test_continuous_stream_recovery_waits_for_physical_cleanup_and_stops_if_quarantined() -> (
    None
):
    class QuarantinedStreamSession(ControlledStreamSession):
        def __init__(self) -> None:
            super().__init__()
            self.release = asyncio.Event()

        async def stop(self) -> None:
            self.calls.append(("stop",))
            while not self.release.is_set():
                try:
                    await self.release.wait()
                except asyncio.CancelledError:
                    pass

    ledger = PeerAudioSegmentLedger(activation_generation=3, settings=settings("gemini_transcribe"))
    start, _, _ = segment_events(ledger, start_sample=0, now=0)
    session = QuarantinedStreamSession()
    opens = 0
    failures: list[Exception] = []

    async def factory(_settings, _epoch):
        nonlocal opens
        opens += 1
        return session

    engine = ScopedRecognitionEngine(
        factory,
        channel="self",
        watchdog_resolver=lambda _: watchdogs(write_timeout_s=0.01, readiness_timeout_s=0.01),
        terminal_failure_sink=failures.append,
    )
    try:
        await engine.handle_owned_vad_event(start)
        session.send_gate.clear()
        await engine.handle_stream_input(stream_input(ledger, 6, 10))
        await engine.handle_stream_input(stream_input(ledger, 10, 14))
        await engine.handle_stream_input(stream_input(ledger, 14, 18))
        assert opens == 1
        assert [str(failure) for failure in failures] == ["provider_resource_quarantined"]
        session.release.set()
        await wait_until(lambda: engine.cleanup_debt == 0)
        await engine.handle_stream_input(stream_input(ledger, 18, 22))
        assert opens == 1
    finally:
        session.release.set()
        await engine.close()


@pytest.mark.asyncio
async def test_independent_scoped_context_deduplicates_seen_audio_and_retains_initial_context() -> (
    None
):
    ledger = PeerAudioSegmentLedger(activation_generation=3, settings=settings("gemini_transcribe"))
    first, _, end = segment_events(ledger, start_sample=0, now=0)
    session = ControlledStreamSession()

    async def factory(_settings, _epoch):
        return session

    engine = ScopedRecognitionEngine(
        factory,
        channel="self",
        watchdog_resolver=lambda _: watchdogs(),
    )
    try:
        await engine.handle_owned_vad_event(first)
        await engine.handle_stream_input(stream_input(ledger, 6, 10))
        await engine.handle_owned_vad_event(end)
        successor = ledger.observe_vad_event(
            SpeechStart(
                uuid4(),
                np.full(2, 0.1, dtype=np.float32),
                np.full(4, 0.2, dtype=np.float32),
                pre_roll_capture=(span(10, 4, 6),),
                chunk_capture=(span(11, 10, 14),),
            ),
            now_monotonic_s=1,
        )
        await engine.handle_owned_vad_event(successor)
        assert [
            (s.normalized_start_sample, s.normalized_end_sample)
            for _, ranges in session.audio
            for s in ranges
        ] == [(0, 2), (2, 6), (6, 10), (10, 14)]
    finally:
        await engine.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("channel", ["self", "peer"])
async def test_independent_final_freezes_latest_source_speech_before_deferred_delivery(
    channel: str,
) -> None:
    ledger = PeerAudioSegmentLedger(activation_generation=3, settings=settings("gemini_transcribe"))
    first, chunk, end = segment_events(ledger, start_sample=0, now=100)
    session = ControlledStreamSession()
    emitted: list[STTRecognitionUnitTerminal] = []
    delivery_entered = asyncio.Event()
    release_delivery = asyncio.Event()

    async def factory(_settings, _epoch):
        return session

    async def sink(event):
        if isinstance(event, STTRecognitionUnitTerminal):
            delivery_entered.set()
            await release_delivery.wait()
            emitted.append(event)

    engine = ScopedRecognitionEngine(
        factory,
        channel=channel,
        monotonic_clock=lambda: 500.0,
        watchdog_resolver=lambda _: watchdogs(),
    )
    engine.bind_event_sink(sink)
    try:
        await engine.observe_source_activity(speech_observed=True, observed_at_monotonic_s=400)
        await engine.handle_stream_input(stream_input(ledger, 6, 10, speech_observed=True))
        await engine.handle_stream_input(stream_input(ledger, 10, 14))
        await engine.handle_owned_vad_event(first)
        await engine.handle_owned_vad_event(chunk)
        await engine.handle_stream_input(stream_input(ledger, 10, 14))
        assert session.stream is not None
        session.emit(STTRecognitionUnit(RecognitionUnitIdentity(session.stream, uuid4(), 1), "one"))
        await asyncio.wait_for(delivery_entered.wait(), timeout=1)
        session.emit(STTRecognitionUnit(RecognitionUnitIdentity(session.stream, uuid4(), 2), "two"))
        await wait_until(lambda: engine._last_receipt_sequence == 2)
        await engine.handle_stream_input(stream_input(ledger, 14, 18, speech_observed=True))
        await engine.handle_owned_vad_event(end)
        release_delivery.set()
        await wait_until(lambda: len(emitted) == 2)
        assert [(event.unit.text, event.unit.estimated_last_speech_at) for event in emitted] == [
            ("one", 10 / 16000),
            ("two", 10 / 16000),
        ]
        session.emit(
            STTRecognitionUnit(RecognitionUnitIdentity(session.stream, uuid4(), 3), "after seal")
        )
        await wait_until(lambda: len(emitted) == 3)
        assert emitted[2].unit.estimated_last_speech_at == 18 / 16000
    finally:
        release_delivery.set()
        await engine.close()


@pytest.mark.asyncio
async def test_independent_final_without_observed_source_speech_has_no_estimate() -> None:
    ledger = PeerAudioSegmentLedger(activation_generation=3, settings=settings("gemini_transcribe"))
    first, _, _ = segment_events(ledger, start_sample=0, now=100)
    session = ControlledStreamSession()
    emitted: list[STTProviderTurnEvent] = []

    async def factory(_settings, _epoch):
        return session

    engine = ScopedRecognitionEngine(
        factory,
        event_sink=emitted.append,
        monotonic_clock=lambda: 500.0,
        watchdog_resolver=lambda _: watchdogs(),
    )
    try:
        await engine.observe_source_activity(speech_observed=True, observed_at_monotonic_s=400)
        await engine.handle_stream_input(stream_input(ledger, 2, 6))
        await engine.handle_owned_vad_event(first)
        await engine.handle_stream_input(stream_input(ledger, 6, 10))
        assert session.stream is not None
        session.emit(
            STTRecognitionUnit(RecognitionUnitIdentity(session.stream, uuid4(), 1), "final")
        )
        await wait_until(lambda: any(isinstance(e, STTRecognitionUnitTerminal) for e in emitted))
        finals = [event for event in emitted if isinstance(event, STTRecognitionUnitTerminal)]
        assert finals[0].unit.estimated_last_speech_at is None
    finally:
        await engine.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "boundary",
    [
        "abort",
        "source_discontinuity",
        "provider_epoch",
        "capture_epoch",
        "capture_epoch_with_speech",
        "activation",
        "settings",
    ],
)
async def test_independent_new_stream_does_not_reuse_retired_speech(boundary: str) -> None:
    ledger = PeerAudioSegmentLedger(activation_generation=3, settings=settings("gemini_transcribe"))
    first, _, end = segment_events(ledger, start_sample=0, now=100)
    sessions: list[ControlledStreamSession] = []
    emitted: list[STTProviderTurnEvent] = []

    async def factory(_settings, _epoch):
        session = ControlledStreamSession()
        sessions.append(session)
        return session

    engine = ScopedRecognitionEngine(
        factory,
        event_sink=emitted.append,
        watchdog_resolver=lambda _: watchdogs(),
    )
    try:
        await engine.handle_stream_input(stream_input(ledger, 2, 6, speech_observed=True))
        await engine.handle_owned_vad_event(first)
        await engine.handle_owned_vad_event(end)
        old_stream = sessions[0].stream
        assert old_stream is not None
        if boundary == "abort":
            await engine.abort(reason="toggle_off")
        elif boundary == "source_discontinuity":
            await engine.handle_stream_input(
                OwnedStreamInput(
                    CaptureStreamInput(
                        np.empty((0,), dtype=np.float32),
                        (),
                        boundary_reason="source_discontinuity",
                    ),
                    ledger,
                    ledger.settings,
                    ledger.activation_generation,
                )
            )
        elif boundary == "provider_epoch":
            sessions[0].emit(
                STTProviderEpochEnded(old_stream.provider_epoch_id, orderly=True, reason="closed")
            )
            await wait_until(lambda: engine._session is None)
        successor_ledger = PeerAudioSegmentLedger(
            activation_generation=4 if boundary == "activation" else 3,
            settings=(
                settings("gemini_transcribe", signature="next")
                if boundary == "settings"
                else ledger.settings
            ),
        )
        successor_capture = replace(
            span(2, 10, 14),
            capture_epoch=2 if boundary.startswith("capture_epoch") else 1,
        )
        successor = successor_ledger.observe_vad_event(
            SpeechStart(
                uuid4(),
                np.empty((0,), dtype=np.float32),
                np.full(4, 0.2, dtype=np.float32),
                chunk_capture=(successor_capture,),
            ),
            now_monotonic_s=200,
        )
        if boundary == "capture_epoch_with_speech":
            await engine.handle_stream_input(
                OwnedStreamInput(
                    CaptureStreamInput(
                        np.full(4, 0.2, dtype=np.float32),
                        (successor_capture,),
                        speech_observed=True,
                    ),
                    successor_ledger,
                    successor_ledger.settings,
                    successor_ledger.activation_generation,
                )
            )
        await engine.handle_owned_vad_event(successor)
        session = sessions[-1]
        assert session.stream is not None
        assert session.stream != old_stream
        session.emit(STTRecognitionUnit(RecognitionUnitIdentity(old_stream, uuid4(), 1), "stale"))
        session.emit(
            STTRecognitionUnit(RecognitionUnitIdentity(session.stream, uuid4(), 1), "current")
        )
        await wait_until(lambda: any(isinstance(e, STTRecognitionUnitTerminal) for e in emitted))
        finals = [event for event in emitted if isinstance(event, STTRecognitionUnitTerminal)]
        assert [(event.unit.text, event.unit.estimated_last_speech_at) for event in finals] == [
            ("current", 14 / 16000 if boundary == "capture_epoch_with_speech" else None)
        ]
    finally:
        await engine.close()
