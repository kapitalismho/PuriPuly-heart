from __future__ import annotations

import asyncio
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
    OwnedVadEvent,
    PeerAudioSegmentLedger,
)
from puripuly_heart.core.stt.backend import (
    STTContributionConsumptionLedger,
    STTNativeProvenance,
    STTProviderEpochEnded,
    STTProviderTurnEvent,
    STTProviderTurnIdentity,
    STTProviderTurnRequest,
    STTProviderTurnTerminal,
    STTProviderTurnUpdate,
)
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


@dataclass(slots=True)
class ControlledScopedSession:
    buffer: STTProviderEventBuffer
    begin_gate: asyncio.Event
    send_gate: asyncio.Event
    requests: list[STTProviderTurnRequest]
    seal_gate: asyncio.Event
    calls: list[tuple[object, ...]]
    terminal_on_seal: tuple[str, str] | None = None
    allows_interim_timeout_fallback: bool = False
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
        self.allows_interim_timeout_fallback = False
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
        "healthy_reset_age_s": 180.0,
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
@pytest.mark.parametrize(
    ("provider_id", "expected_outcome", "expected_text"),
    [
        ("gemini_transcribe", "degraded", "interim only"),
        ("deepgram", "failed", ""),
        ("soniox", "failed", ""),
        ("elevenlabs_scribe", "failed", ""),
    ],
)
async def test_interim_timeout_fallback_is_gemini_adapter_declared_only(
    provider_id: str,
    expected_outcome: str,
    expected_text: str,
) -> None:
    ledger = PeerAudioSegmentLedger(activation_generation=1, settings=settings(provider_id))
    start, _chunk, end = segment_events(ledger, start_sample=700, now=7.0)
    session = ControlledScopedSession()
    session.allows_interim_timeout_fallback = True
    emitted: list[object] = []
    engine = ScopedRecognitionEngine(
        session_factory=lambda _settings, _epoch: asyncio.sleep(0, result=session),
        event_sink=lambda event: emitted.append(event),
        watchdog_resolver=lambda _settings: watchdogs(final_timeout_s=0.01),
    )

    await engine.handle_owned_vad_event(start)
    identity = session.calls[0][1]
    session.emit(
        STTProviderTurnUpdate(
            identity=identity,
            sequence=1,
            stability="provisional",
            assembly="replace",
            text="interim only",
        )
    )
    await wait_until(lambda: any(isinstance(item, STTProviderTurnUpdate) for item in emitted))
    await engine.handle_owned_vad_event(end)

    terminal = next(item for item in emitted if isinstance(item, STTProviderTurnTerminal))
    assert terminal.outcome == expected_outcome
    assert terminal.text == expected_text
    assert terminal.failure_reason == "provider_final_timeout"
    assert terminal.epoch_disposition == "retire"
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
async def test_soft_age_waits_through_continuous_speech_then_expires_without_callback() -> None:
    ledger = PeerAudioSegmentLedger(activation_generation=1, settings=settings())
    first = segment_events(ledger, start_sample=900, now=9.0)
    session = ControlledScopedSession()
    session.terminal_on_seal = ("empty", "")
    clock = ControlledMonotonicClock()
    engine = ScopedRecognitionEngine(
        session_factory=lambda _settings, _epoch: asyncio.sleep(0, result=session),
        event_sink=lambda _event: None,
        watchdog_resolver=lambda _settings: watchdogs(
            healthy_reset_age_s=180.0,
            recent_speech_window_s=10.0,
        ),
        monotonic_clock=clock.now,
        sleep=clock.sleep,
    )

    await engine.observe_source_activity(
        speech_observed=True,
        observed_at_monotonic_s=0.0,
    )
    await engine.handle_owned_vad_event(first[0])
    await engine.handle_owned_vad_event(first[2])
    await asyncio.sleep(0)

    await clock.advance_to(180.0)
    assert not any(call[0] == "close" for call in session.calls)
    await clock.advance_to(300.0)
    assert not any(call[0] == "close" for call in session.calls)

    await engine.observe_source_activity(
        speech_observed=False,
        observed_at_monotonic_s=300.0,
    )
    await asyncio.sleep(0)
    await clock.advance_to(309.9)
    assert not any(call[0] == "close" for call in session.calls)
    await engine.observe_source_activity(
        speech_observed=True,
        observed_at_monotonic_s=309.9,
    )
    await engine.observe_source_activity(
        speech_observed=False,
        observed_at_monotonic_s=309.9,
    )
    await clock.advance_to(319.8)
    assert not any(call[0] == "close" for call in session.calls)
    await clock.advance_to(319.9)
    await wait_until(lambda: any(call[0] == "close" for call in session.calls))
    await wait_until(lambda: engine.cleanup_debt == 0)
    await engine.close()


@pytest.mark.asyncio
async def test_soft_age_waits_for_pending_terminal_and_recent_speech() -> None:
    ledger = PeerAudioSegmentLedger(activation_generation=1, settings=settings())
    first = segment_events(ledger, start_sample=900, now=9.0)
    session = ControlledScopedSession()
    clock = ControlledMonotonicClock()
    engine = ScopedRecognitionEngine(
        session_factory=lambda _settings, _epoch: asyncio.sleep(0, result=session),
        event_sink=lambda _event: None,
        watchdog_resolver=lambda _settings: watchdogs(
            healthy_reset_age_s=180.0,
            recent_speech_window_s=10.0,
        ),
        monotonic_clock=clock.now,
        sleep=clock.sleep,
    )

    await engine.handle_owned_vad_event(first[0])
    await asyncio.sleep(0)
    await clock.advance_to(179.0)
    await engine.observe_source_activity(
        speech_observed=True,
        observed_at_monotonic_s=179.0,
    )
    end_task = asyncio.create_task(engine.handle_owned_vad_event(first[2]))
    await wait_until(lambda: any(call[0] == "seal_done" for call in session.calls))
    await engine.observe_source_activity(
        speech_observed=False,
        observed_at_monotonic_s=179.0,
    )
    await asyncio.sleep(0)
    await clock.advance_to(189.0)
    assert not any(call[0] == "close" for call in session.calls)

    identity = session.requests[0].identity
    session.emit(
        STTProviderTurnTerminal(
            identity=identity,
            outcome="final",
            text="complete",
            text_authority="authoritative",
            epoch_disposition="reuse",
        )
    )
    await end_task
    await wait_until(lambda: any(call[0] == "close" for call in session.calls))
    await engine.close()


@pytest.mark.asyncio
async def test_soft_age_preserves_pending_owned_input_until_source_queue_drains() -> None:
    ledger = PeerAudioSegmentLedger(activation_generation=1, settings=settings())
    first = segment_events(ledger, start_sample=900, now=9.0)
    session = ControlledScopedSession()
    session.terminal_on_seal = ("empty", "")
    clock = ControlledMonotonicClock()
    engine = ScopedRecognitionEngine(
        session_factory=lambda _settings, _epoch: asyncio.sleep(0, result=session),
        event_sink=lambda _event: None,
        watchdog_resolver=lambda _settings: watchdogs(healthy_reset_age_s=180.0),
        monotonic_clock=clock.now,
        sleep=clock.sleep,
    )

    await engine.handle_owned_vad_event(first[0])
    await engine.handle_owned_vad_event(first[2])
    await engine.observe_pending_source_work(pending=True)
    await asyncio.sleep(0)
    await clock.advance_to(300.0)
    assert not any(call[0] == "close" for call in session.calls)

    await engine.observe_pending_source_work(pending=False)
    await wait_until(lambda: any(call[0] == "close" for call in session.calls))
    await engine.close()


@pytest.mark.asyncio
async def test_idle_age_retirement_reconnects_only_when_new_work_is_admitted() -> None:
    first_ledger = PeerAudioSegmentLedger(activation_generation=1, settings=settings())
    second_ledger = PeerAudioSegmentLedger(activation_generation=1, settings=settings())
    first = segment_events(first_ledger, start_sample=900, now=9.0)
    second = segment_events(second_ledger, start_sample=1000, now=10.0)
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
        watchdog_resolver=lambda _settings: watchdogs(healthy_reset_age_s=180.0),
        monotonic_clock=clock.now,
        sleep=clock.sleep,
    )

    await engine.handle_owned_vad_event(first[0])
    await engine.handle_owned_vad_event(first[2])
    await asyncio.sleep(0)
    await clock.advance_to(180.0)
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
    [
        "qwen_audio",
        "custom",
        "custom_realtime",
        "custom_offline",
        "local_cpu_auto",
        "local_parakeet_v3",
        "local_parakeet_ja",
        "local_qwen",
        "local_qwen_gpu",
    ],
)
async def test_non_target_routes_keep_turn_boundary_age_rotation(provider_id: str) -> None:
    provider_settings = settings(provider_id)
    first_ledger = PeerAudioSegmentLedger(
        activation_generation=1,
        settings=provider_settings,
    )
    second_ledger = PeerAudioSegmentLedger(
        activation_generation=1,
        settings=provider_settings,
    )
    first = segment_events(first_ledger, start_sample=900, now=9.0)
    second = segment_events(second_ledger, start_sample=1000, now=10.0)
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
        watchdog_resolver=lambda _settings: watchdogs(healthy_reset_age_s=180.0),
        monotonic_clock=clock.now,
        sleep=clock.sleep,
        deferred_age_rotation_enabled=False,
    )

    await engine.handle_owned_vad_event(first[0])
    await engine.handle_owned_vad_event(first[2])
    await clock.advance_to(300.0)
    assert len(sessions) == 1
    assert not any(call[0] == "close" for call in sessions[0].calls)

    await engine.handle_owned_vad_event(second[0])
    await engine.handle_owned_vad_event(second[2])
    assert len(sessions) == 2
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
