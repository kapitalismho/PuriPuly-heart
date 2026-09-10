from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, cast
from uuid import uuid4

import numpy as np
import pytest

from puripuly_heart.core.audio.format import AudioCaptureSpan
from puripuly_heart.core.audio.ownership import (
    AudioSegmentSettingsSnapshot,
    OwnedVadEvent,
    PeerAudioSegmentLedger,
)
from puripuly_heart.core.stt.backend import (
    STTNativeProvenance,
    STTProviderTurnIdentity,
    STTProviderTurnRequest,
    STTProviderTurnTerminal,
    STTProviderTurnEvent,
    STTProviderTurnUpdate,
)
from puripuly_heart.core.stt.scoped_engine import (
    STTRecognitionWatchdogs,
    ScopedRecognitionEngine,
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
    seal_gate: asyncio.Event
    calls: list[tuple[object, ...]]
    terminal_on_seal: tuple[str, str] | None = None
    allows_interim_timeout_fallback: bool = False

    def __init__(self) -> None:
        self.buffer = STTProviderEventBuffer()
        self.begin_gate = asyncio.Event()
        self.send_gate = asyncio.Event()
        self.seal_gate = asyncio.Event()
        self.begin_gate.set()
        self.send_gate.set()
        self.seal_gate.set()
        self.calls = []
        self.terminal_on_seal = None
        self.allows_interim_timeout_fallback = False

    async def begin_turn(self, request: STTProviderTurnRequest) -> None:
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


def settings(provider_id: str = "deepgram", *, signature: str = "a") -> AudioSegmentSettingsSnapshot:
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
    assert terminal.final_language_runs == (
        FinalLanguageRun(text="repeated repeated", language="unknown"),
    )
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
    accepted = "a" * (STTScopedTurnNormalizer.MAX_ASSEMBLY_BYTES - len("unknown"))
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
        FinalLanguageRun(text="x", language="en" if index % 2 else "ja")
        for index in range(256)
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
    assert len(result_256.final_language_runs) == 256
    assert "".join(run.text for run in result_256.final_language_runs) == result_256.text

    runs_257 = tuple(
        FinalLanguageRun(text="y", language="en" if index % 2 else "ja")
        for index in range(257)
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
    assert result_257.final_language_runs == (
        FinalLanguageRun(text="y" * 257, language="unknown"),
    )
    assert diagnostics


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
    ("provider_id", "expected_outcome", "expected_text"),
    [
        ("gemini_transcribe", "degraded", "interim only"),
        ("deepgram", "failed", ""),
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
    await wait_until(
        lambda: any(isinstance(item, STTProviderTurnUpdate) for item in emitted)
    )
    await engine.handle_owned_vad_event(end)

    terminal = next(item for item in emitted if isinstance(item, STTProviderTurnTerminal))
    assert terminal.outcome == expected_outcome
    assert terminal.text == expected_text
    assert terminal.failure_reason == "provider_final_timeout"
    await engine.close()


@pytest.mark.asyncio
async def test_recovery_is_three_attempts_with_point_eight_and_one_point_six_backoff() -> None:
    ledger = PeerAudioSegmentLedger(activation_generation=1, settings=settings())
    start, _chunk, end = segment_events(ledger, start_sample=800, now=8.0)
    attempts = 0
    delays: list[float] = []
    emitted: list[object] = []

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
    )
    await engine.handle_owned_vad_event(start)
    await engine.handle_owned_vad_event(end)

    assert attempts == 3
    assert delays == [0.8, 1.6]
    terminal = next(item for item in emitted if isinstance(item, STTProviderTurnTerminal))
    assert terminal.outcome == "failed"
    assert terminal.failure_reason == "provider_not_ready:RuntimeError"
    await engine.close()


@pytest.mark.asyncio
async def test_configuration_and_healthy_age_rotate_only_at_turn_barrier() -> None:
    first_ledger = PeerAudioSegmentLedger(
        activation_generation=1,
        settings=settings(signature="old"),
    )
    second_ledger = PeerAudioSegmentLedger(
        activation_generation=1,
        settings=settings(signature="new"),
    )
    third_ledger = PeerAudioSegmentLedger(
        activation_generation=1,
        settings=settings(signature="new"),
    )
    first = segment_events(first_ledger, start_sample=900, now=9.0)
    second = segment_events(second_ledger, start_sample=1000, now=10.0)
    third = segment_events(third_ledger, start_sample=1100, now=11.0)
    sessions: list[ControlledScopedSession] = []
    clock = [0.0]

    async def factory(_settings: AudioSegmentSettingsSnapshot, _epoch: str):
        session = ControlledScopedSession()
        session.terminal_on_seal = ("empty", "")
        sessions.append(session)
        return session

    engine = ScopedRecognitionEngine(
        session_factory=factory,
        event_sink=lambda _event: None,
        watchdog_resolver=lambda _settings: watchdogs(healthy_reset_age_s=10.0),
        monotonic_clock=lambda: clock[0],
    )

    await engine.handle_owned_vad_event(first[0])
    await engine.handle_owned_vad_event(first[2])
    assert len(sessions) == 1

    await engine.handle_owned_vad_event(second[0])
    await engine.handle_owned_vad_event(second[2])
    assert len(sessions) == 2
    await wait_until(lambda: any(call[0] == "close" for call in sessions[0].calls))

    clock[0] = 11.0
    await engine.handle_owned_vad_event(third[0])
    await engine.handle_owned_vad_event(third[2])
    assert len(sessions) == 3
    await wait_until(lambda: any(call[0] == "close" for call in sessions[1].calls))
    await engine.close()
