from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import replace
from typing import Literal
from uuid import uuid4

import numpy as np
import pytest
from puripuly_heart.app.adapters.self_capture_vad_sink import SelfCaptureVadSinkAdapter

from puripuly_heart.config.resolved import vad_exit_threshold
from puripuly_heart.core.audio.format import AudioCaptureSpan
from puripuly_heart.core.audio.ownership import (
    AudioSegmentIdentity,
    CaptureStreamInput,
)
from puripuly_heart.core.runtime.self_capture import SelfCaptureSessionOwner
from puripuly_heart.core.self_capture import (
    SelfCaptureAdmission,
    SelfCaptureAdmissionStatus,
    SelfCaptureDiagnostic,
    SelfCaptureFailureReason,
    SelfCaptureIngressError,
    SelfCaptureProviderMutation,
    SelfCaptureProviderMutationStatus,
    SelfCaptureProviderStatus,
    SelfCaptureSessionConfig,
    SelfCaptureSessionState,
)
from puripuly_heart.core.stt.backend import (
    STTProviderInputTerminal,
    STTProviderTurnIdentity,
    STTProviderTurnRequest,
    STTProviderTurnTerminal,
    STTRecognitionUnit,
    STTRecognitionUnitTerminal,
)
from puripuly_heart.core.stt.scoped_engine import (
    ScopedRecognitionEngine,
    STTRecognitionWatchdogs,
    STTRetentionProfile,
)
from puripuly_heart.core.vad.gating import SpeechChunk, SpeechEnd, SpeechStart, VadGating
from puripuly_heart.domain.recognition import RecognitionStreamIdentity, RecognitionUnitIdentity
from tests.helpers.fakes import RecordingOscQueue
from tests.helpers.translation_owners import compose_translation_test_harness
from tests.helpers.vad import SequenceVadEngine, chunk_samples


class RecordingAdmission:
    def __init__(
        self,
        result: SelfCaptureAdmission | None = None,
        *,
        gate: asyncio.Event | None = None,
    ) -> None:
        self.result = result or SelfCaptureAdmission(SelfCaptureAdmissionStatus.ADMITTED)
        self.gate = gate
        self.calls: list[SelfCaptureSessionConfig] = []

    async def admit(self, config: SelfCaptureSessionConfig) -> SelfCaptureAdmission:
        self.calls.append(config)
        if self.gate is not None:
            await self.gate.wait()
        return self.result


class RecordingProvider:
    def __init__(self, *, ready: bool = False) -> None:
        self.ready = ready
        self.replace_result = SelfCaptureProviderMutation(SelfCaptureProviderMutationStatus.APPLIED)
        self.handoff_result = SelfCaptureProviderMutation(SelfCaptureProviderMutationStatus.APPLIED)
        self.replace_calls: list[tuple[object, bool]] = []
        self.handoff_calls: list[tuple[object, bool]] = []
        self.release_calls: list[tuple[str, float | None]] = []
        self.reconfigure_calls: list[object] = []
        self.cancel_handoff_calls = 0
        self.start_calls = 0
        self.start_failure: Exception | None = None
        self.warmup_calls = 0
        self.terminal_failure_handler = None

    def is_ready(self, config: SelfCaptureSessionConfig) -> bool:
        _ = config
        return self.ready

    async def replace(
        self,
        request: object,
        *,
        start: bool,
        on_terminal_failure: Callable[[Exception], Awaitable[None]],
    ) -> SelfCaptureProviderMutation:
        self.replace_calls.append((request, start))
        self.terminal_failure_handler = on_terminal_failure
        if self.replace_result.status is SelfCaptureProviderMutationStatus.APPLIED:
            self.ready = True
        return self.replace_result

    async def handoff(
        self,
        request: object,
        *,
        start: bool,
        on_terminal_failure: Callable[[Exception], Awaitable[None]],
    ) -> SelfCaptureProviderMutation:
        self.handoff_calls.append((request, start))
        self.terminal_failure_handler = on_terminal_failure
        return self.handoff_result

    async def cancel_handoff(self) -> bool:
        self.cancel_handoff_calls += 1
        return True

    async def start_ingress(self) -> None:
        self.start_calls += 1
        if self.start_failure is not None:
            raise self.start_failure

    async def warmup(self) -> None:
        self.warmup_calls += 1

    async def reconfigure(self, session_options: object) -> None:
        self.reconfigure_calls.append(session_options)

    async def release(
        self,
        *,
        mode: Literal["drain", "abort"],
        release_backend_after: float | None = None,
    ) -> None:
        self.release_calls.append((mode, release_backend_after))
        self.ready = False


class RecordingSource:
    def __init__(self, failures: list[Exception] | None = None) -> None:
        self.failures = list(failures or [])
        self.close_calls = 0

    async def close(self) -> None:
        self.close_calls += 1
        if self.failures:
            raise self.failures.pop(0)


class RecordingSink:
    def __init__(self) -> None:
        self.events: list[object] = []

    async def handle_vad_event(self, event: object) -> None:
        self.events.append(event)


class BlockingRecognitionSink(RecordingSink):
    def __init__(self) -> None:
        super().__init__()
        self.release = asyncio.Event()
        self.started = asyncio.Event()
        self.rejections: list[tuple[object, str, str]] = []
        self.failures: list[tuple[object, str]] = []

    async def handle_vad_event(self, event: object) -> None:
        self.events.append(event)
        if not self.started.is_set():
            self.started.set()
            await self.release.wait()

    async def reject_owned_segment(
        self,
        owned: object,
        *,
        reason: str,
        outcome: str,
    ) -> None:
        self.rejections.append((owned, reason, outcome))

    async def fail_owned_segment(self, owned: object, *, reason: str) -> None:
        self.failures.append((owned, reason))


class BlockingScopedSession:
    def __init__(self) -> None:
        self.events: asyncio.Queue[object | None] = asyncio.Queue()
        self.first_send_started = asyncio.Event()
        self.release_first_send = asyncio.Event()
        self.identities: list[STTProviderTurnIdentity] = []
        self.second_send_started = asyncio.Event()
        self.release_second_send = asyncio.Event()
        self.send_count = 0

    async def begin_turn(self, request: STTProviderTurnRequest) -> None:
        self.identities.append(request.identity)

    async def send_turn_audio(
        self,
        _identity: STTProviderTurnIdentity,
        _pcm: bytes,
        **_kwargs: object,
    ) -> None:
        self.send_count += 1
        if self.send_count == 1:
            self.first_send_started.set()
            await self.release_first_send.wait()
        elif self.send_count == 2:
            self.second_send_started.set()
            await self.release_second_send.wait()

    async def seal_turn(
        self,
        identity: STTProviderTurnIdentity,
        **_kwargs: object,
    ) -> None:
        await self.events.put(
            STTProviderTurnTerminal(
                identity,
                "final",
                text="recognized",
                text_authority="authoritative",
            )
        )

    async def abort_turn(
        self,
        _identity: STTProviderTurnIdentity,
        **_kwargs: object,
    ) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def close(self) -> None:
        await self.events.put(None)

    async def turn_events(self):
        while True:
            event = await self.events.get()
            if event is None:
                return
            yield event


class LoopHarness:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.calls: list[dict[str, object]] = []
        self.failure: Exception | None = None

    async def run(self, **kwargs: object) -> None:
        self.calls.append(kwargs)
        self.started.set()
        if self.failure is not None:
            raise self.failure
        await self.release.wait()


def config(
    suffix: str = "one",
    *,
    local_cpu: bool = False,
    local_gpu: bool = False,
    capture_signature: tuple[object, ...] = ("capture",),
    vad_speech_threshold: float = 0.5,
    vad_hangover_ms: int = 1100,
    ring_buffer_ms: int = 2000,
) -> SelfCaptureSessionConfig:
    return SelfCaptureSessionConfig(
        provider_id=f"provider-{suffix}",
        provider_signature=("provider", suffix),
        runtime_signature=("runtime", suffix),
        capture_signature=capture_signature,
        target_sample_rate_hz=16000,
        vad_speech_threshold=vad_speech_threshold,
        vad_hangover_ms=vad_hangover_ms,
        ring_buffer_ms=ring_buffer_ms,
        session_options=("options", suffix),
        local_cpu=local_cpu,
        local_gpu=local_gpu,
        release_backend_after=600.0 if local_cpu else None,
    )


def build_owner(
    *,
    admission: RecordingAdmission | None = None,
    provider: RecordingProvider | None = None,
    sources: list[RecordingSource] | None = None,
    vad_factory: Callable[[SelfCaptureSessionConfig], object] | None = None,
    loop: LoopHarness | None = None,
    sink: RecordingSink | None = None,
    diagnostics: list[SelfCaptureDiagnostic] | None = None,
    gate_resets: list[str] | None = None,
) -> tuple[
    SelfCaptureSessionOwner,
    RecordingAdmission,
    RecordingProvider,
    list[RecordingSource],
    LoopHarness,
    RecordingSink,
]:
    admission = admission or RecordingAdmission()
    provider = provider or RecordingProvider()
    source_list = sources if sources is not None else []
    loop = loop or LoopHarness()
    sink = sink or RecordingSink()

    def source_factory(_config: SelfCaptureSessionConfig) -> RecordingSource:
        source = RecordingSource()
        source_list.append(source)
        return source

    owner = SelfCaptureSessionOwner(
        admission=admission,
        provider=provider,
        provider_request_factory=lambda request_config, warmup: (
            request_config.provider_id,
            warmup,
        ),
        source_factory=source_factory,
        vad_factory=vad_factory or (lambda _config: object()),
        run_audio_loop=loop.run,
        vad_sink=sink,
        diagnostic_sink=(diagnostics.append if diagnostics is not None else None),
        audio_gate_reset=(
            (lambda: gate_resets.append("reset")) if gate_resets is not None else None
        ),
    )
    return owner, admission, provider, source_list, loop, sink


async def wait_until(predicate: Callable[[], bool], *, timeout_s: float = 1.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while not predicate():
        if loop.time() >= deadline:
            raise AssertionError("timed out waiting for condition")
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_inactive_active_restart_and_explicit_toggle_off_preserve_release_policy() -> None:
    loops = [LoopHarness(), LoopHarness()]
    active_loop = 0

    async def run_loop(**kwargs: object) -> None:
        await loops[active_loop].run(**kwargs)

    owner, _, provider, sources, _, _ = build_owner()
    owner._run_audio_loop = run_loop
    session_config = config(local_cpu=True)

    assert owner.snapshot.state is SelfCaptureSessionState.STOPPED

    await owner.apply_intent(session_config, enabled=True)
    assert owner.snapshot.effective_active is True
    assert provider.replace_calls == [(("provider-one", True), False)]

    active_loop = 1
    await owner.apply_intent(
        session_config,
        enabled=True,
        restart=True,
        explicit_toggle_off=False,
    )
    assert sources[0].close_calls == 1
    assert provider.release_calls == [("drain", 600.0)]
    assert owner.snapshot.effective_active is True

    await owner.apply_intent(session_config, enabled=False)
    assert owner.snapshot.state is SelfCaptureSessionState.STOPPED
    assert sources[1].close_calls == 1
    assert provider.release_calls[-1] == ("abort", None)


@pytest.mark.asyncio
async def test_active_self_vad_settings_remain_frozen_until_successor_episode() -> None:
    engine = SequenceVadEngine(probs=[0.6, 0.55, 0.49, 0.49, 0.7, 0.8])
    gate = VadGating(
        engine,
        sample_rate_hz=16000,
        ring_buffer_ms=64,
        speech_threshold=0.6,
        continuation_threshold=vad_exit_threshold(0.6),
        hangover_ms=64,
    )
    owner, _, provider, sources, _, _ = build_owner(vad_factory=lambda _config: gate)
    initial = config(
        vad_speech_threshold=0.6,
        vad_hangover_ms=64,
        ring_buffer_ms=64,
    )
    await owner.apply_intent(initial, enabled=True)
    start = gate.process_chunk(chunk_samples(1.0, n=gate.chunk_samples))
    assert isinstance(start[0], SpeechStart)

    updated = replace(
        initial,
        runtime_signature=("runtime", "updated-vad"),
        vad_speech_threshold=0.8,
        vad_hangover_ms=96,
        ring_buffer_ms=96,
    )
    await owner.apply_intent(updated, enabled=True)

    assert len(provider.handoff_calls) == 1
    assert len(sources) == 1
    assert gate.speech_threshold == 0.6
    assert gate.continuation_threshold == pytest.approx(0.5)
    assert gate.hangover_chunks == 2
    gate.process_chunk(chunk_samples(2.0, n=gate.chunk_samples))
    gate.process_chunk(chunk_samples(3.0, n=gate.chunk_samples))
    ended = gate.process_chunk(chunk_samples(4.0, n=gate.chunk_samples))
    assert any(isinstance(event, SpeechEnd) for event in ended)
    assert gate.speech_threshold == 0.8
    assert gate.continuation_threshold == pytest.approx(0.7)
    assert gate.hangover_chunks == 3
    assert gate.process_chunk(chunk_samples(5.0, n=gate.chunk_samples)) == []
    successor = gate.process_chunk(chunk_samples(6.0, n=gate.chunk_samples))
    assert isinstance(successor[0], SpeechStart)

    await owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("toggle_delay_s", [0.0, 0.1, 0.3, 1.0])
async def test_cloud_explicit_toggle_off_routes_to_abort_at_each_delay(
    toggle_delay_s: float,
) -> None:
    owner, _, provider, _, _, _ = build_owner()
    session_config = config()

    await owner.apply_intent(session_config, enabled=True)
    await asyncio.sleep(toggle_delay_s)
    snapshot = await owner.apply_intent(session_config, enabled=False)

    assert snapshot.state is SelfCaptureSessionState.STOPPED
    assert provider.release_calls == [("abort", None)]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("admission_result", "state", "desired"),
    [
        (
            SelfCaptureAdmission(SelfCaptureAdmissionStatus.PENDING, reason="install_pending"),
            SelfCaptureSessionState.ADMISSION_PENDING,
            True,
        ),
        (
            SelfCaptureAdmission(SelfCaptureAdmissionStatus.REJECTED, reason="unavailable"),
            SelfCaptureSessionState.FAULTED,
            False,
        ),
        (
            SelfCaptureAdmission(
                SelfCaptureAdmissionStatus.REJECTED,
                reason="consent_pending",
                retain_intent=True,
            ),
            SelfCaptureSessionState.FAULTED,
            True,
        ),
    ],
)
async def test_admission_facts_distinguish_pending_rejected_and_retained_intent(
    admission_result: SelfCaptureAdmission,
    state: SelfCaptureSessionState,
    desired: bool,
) -> None:
    owner, _, provider, sources, _, _ = build_owner(admission=RecordingAdmission(admission_result))

    snapshot = await owner.apply_intent(config(), enabled=True)

    assert snapshot.state is state
    assert snapshot.desired_active is desired
    assert snapshot.admission_reason == admission_result.reason
    assert provider.replace_calls == []
    assert sources == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure_stage", "reason"),
    [
        ("source", SelfCaptureFailureReason.SOURCE_OPEN_FAILED),
        ("vad", SelfCaptureFailureReason.VAD_FAILED),
    ],
)
async def test_source_and_vad_start_failures_abort_provider_and_leave_no_resources(
    failure_stage: str,
    reason: SelfCaptureFailureReason,
) -> None:
    diagnostics: list[SelfCaptureDiagnostic] = []

    def fail_source(_config: SelfCaptureSessionConfig) -> object:
        raise RuntimeError("secret source detail")

    def fail_vad(_config: SelfCaptureSessionConfig) -> object:
        raise RuntimeError("secret vad detail")

    owner, _, provider, _, _, _ = build_owner(
        vad_factory=fail_vad if failure_stage == "vad" else None,
        diagnostics=diagnostics,
    )
    if failure_stage == "source":
        owner._source_factory = fail_source

    snapshot = await owner.apply_intent(config(), enabled=True)

    assert snapshot.state is SelfCaptureSessionState.FAULTED
    assert snapshot.failure_reason is reason
    assert snapshot.has_source is False
    assert snapshot.has_vad is False
    assert snapshot.has_loop_task is False
    assert provider.release_calls == [("abort", None)]
    assert diagnostics[-1].detail == "RuntimeError"
    assert "secret" not in repr(diagnostics[-1])


@pytest.mark.asyncio
async def test_late_and_stale_generation_callbacks_cannot_reach_self_sink() -> None:
    owner, _, _, _, _, sink = build_owner()
    session_config = config()

    await owner.apply_intent(session_config, enabled=True)
    guarded_sink = owner.guard_vad_sink()
    await getattr(guarded_sink, "handle_vad_event")("current")

    await owner.apply_intent(session_config, enabled=False)
    await getattr(guarded_sink, "handle_vad_event")("late")

    assert sink.events == ["current"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutation_status",
    [
        None,
        SelfCaptureProviderMutationStatus.FAILED,
        SelfCaptureProviderMutationStatus.PENDING,
    ],
)
async def test_retained_capture_handles_failure_from_new_segment_after_rebind(
    mutation_status: SelfCaptureProviderMutationStatus | None,
) -> None:
    owner, _, provider, sources, loop, _ = build_owner()
    initial = replace(config(), provider_id="deepgram")
    try:
        await owner.apply_intent(initial, enabled=True)
        await loop.started.wait()
        guarded = loop.calls[0]["sink"]
        if mutation_status is None:
            await owner.apply_intent(initial, enabled=True)
        else:
            provider.handoff_result = SelfCaptureProviderMutation(
                mutation_status, reason="provider_readiness_unavailable"
            )
            await owner.apply_intent(config("two"), enabled=True)
        segment_id = uuid4()
        await guarded.handle_vad_event(
            SpeechStart(
                segment_id,
                pre_roll=np.empty((0,), dtype=np.float32),
                chunk=np.ones((8,), dtype=np.float32),
            )
        )
        await guarded.handle_vad_event(SpeechEnd(segment_id))
        identity = guarded.ledger.snapshots[0].identity
        owner.note_recognition_terminal(
            STTProviderTurnTerminal(
                STTProviderTurnIdentity(identity, "failed-epoch", "failed-turn"),
                "failed",
                failure_reason="provider_final_timeout",
                epoch_disposition="retire",
            )
        )
        await wait_until(lambda: owner.snapshot.state is SelfCaptureSessionState.FAULTED)
        assert not owner.snapshot.desired_active
        assert not owner.snapshot.effective_active
        assert sources[0].close_calls == 1
        assert provider.release_calls[-1] == ("abort", None)
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_scoped_terminals_retire_self_ledgers_and_failure_closes_capture() -> None:
    diagnostics: list[SelfCaptureDiagnostic] = []
    owner, _, provider, sources, _, _ = build_owner(diagnostics=diagnostics)
    await owner.apply_intent(config(), enabled=True)
    guarded = owner.guard_vad_sink()
    ledger = guarded.ledger

    for order in range(1, 46):
        segment_id = uuid4()
        start_sample = order * 8
        span = AudioCaptureSpan(
            capture_epoch=1,
            callback_sequence=order,
            source_sample_rate_hz=16_000,
            source_start_sample=start_sample,
            source_end_sample=start_sample + 8,
            source_start_monotonic_s=start_sample / 16_000,
            source_end_monotonic_s=(start_sample + 8) / 16_000,
            normalized_sample_rate_hz=16_000,
            normalized_start_sample=start_sample,
            normalized_end_sample=start_sample + 8,
        )
        await guarded.handle_vad_event(
            SpeechStart(
                segment_id,
                pre_roll=np.empty((0,), dtype=np.float32),
                chunk=np.ones((8,), dtype=np.float32),
                chunk_capture=(span,),
            )
        )
        await guarded.handle_vad_event(SpeechEnd(segment_id))
        owner.note_recognition_terminal(
            STTProviderTurnTerminal(
                STTProviderTurnIdentity(
                    AudioSegmentIdentity(1, order, segment_id, 1),
                    "epoch",
                    f"turn-{order}",
                ),
                "empty",
                text_authority="authoritative",
            )
        )
    assert ledger.snapshots == ()
    assert len(ledger.terminal_receipts) == 45
    failed_id = uuid4()
    failed_span = AudioCaptureSpan(
        capture_epoch=1,
        callback_sequence=46,
        source_sample_rate_hz=16_000,
        source_start_sample=368,
        source_end_sample=376,
        source_start_monotonic_s=368 / 16_000,
        source_end_monotonic_s=376 / 16_000,
        normalized_sample_rate_hz=16_000,
        normalized_start_sample=368,
        normalized_end_sample=376,
    )
    await guarded.handle_vad_event(
        SpeechStart(
            failed_id,
            pre_roll=np.empty((0,), dtype=np.float32),
            chunk=np.ones((8,), dtype=np.float32),
            chunk_capture=(failed_span,),
        )
    )
    owner.note_recognition_terminal(
        STTProviderTurnTerminal(
            STTProviderTurnIdentity(
                AudioSegmentIdentity(1, 46, failed_id, 1),
                "epoch",
                "failed-turn",
            ),
            "failed",
            text_authority="none",
            failure_reason="buffer_exhausted",
            epoch_disposition="retire",
        )
    )

    await wait_until(lambda: owner.snapshot.state is SelfCaptureSessionState.FAULTED)
    assert owner.snapshot.failure_reason is SelfCaptureFailureReason.SESSION_FAILED
    assert ledger.snapshots == ()
    assert sources[0].close_calls == 1
    assert provider.release_calls[-1] == ("abort", None)
    failures = [
        item for item in diagnostics if item.reason is SelfCaptureFailureReason.SESSION_FAILED
    ]
    assert len(failures) == 1
    failure = failures[0]
    assert failure.recognition_reason == "buffer_exhausted"
    assert failure.utterance_id == failed_id
    assert failure.epoch == "epoch"
    assert failure.turn == "failed-turn"
    assert failure.activation_generation == 1
    assert failure.generation == 1
    assert failure.desired_active_before is True
    assert failure.desired_active_after is False
    assert failure.action == "deactivate"
    assert failure.target_state is SelfCaptureSessionState.FAULTED


@pytest.mark.asyncio
async def test_recoverable_terminal_retires_utterance_without_interrupting_next_capture() -> None:
    owner, _, provider, sources, _, sink = build_owner()
    await owner.apply_intent(config(), enabled=True)
    guarded = owner.guard_vad_sink()
    first_id = uuid4()
    await guarded.handle_vad_event(
        SpeechStart(
            first_id,
            pre_roll=np.empty((0,), dtype=np.float32),
            chunk=np.ones((8,), dtype=np.float32),
        )
    )
    await guarded.handle_vad_event(SpeechEnd(first_id))
    first_identity = guarded.ledger.snapshots[0].identity
    owner.note_recognition_terminal(
        STTProviderTurnTerminal(
            STTProviderTurnIdentity(first_identity, "failed-epoch", "failed-turn"),
            "failed",
            failure_reason="soniox_receive_failed",
            failure_retryable=True,
            recovery_pending=True,
            epoch_disposition="retire",
        )
    )

    assert guarded.ledger.snapshots == ()
    assert guarded.ledger.terminal_receipts[0].failure_reason == "soniox_receive_failed"
    assert owner.snapshot.state is SelfCaptureSessionState.RUNNING
    assert owner.snapshot.desired_active and owner.snapshot.effective_active
    assert owner.snapshot.failure_reason is None
    assert sources[0].close_calls == 0
    assert provider.release_calls == []

    second_id = uuid4()
    await guarded.handle_vad_event(
        SpeechStart(
            second_id,
            pre_roll=np.empty((0,), dtype=np.float32),
            chunk=np.ones((8,), dtype=np.float32),
        )
    )
    await guarded.handle_vad_event(SpeechEnd(second_id))
    second_identity = guarded.ledger.snapshots[0].identity
    owner.note_recognition_terminal(
        STTProviderTurnTerminal(
            STTProviderTurnIdentity(second_identity, "new-epoch", "final-turn"),
            "final",
            text="later success",
            text_authority="authoritative",
        )
    )
    await wait_until(lambda: len(sink.events) == 4)
    assert [item.segment.identity.segment_id for item in sink.events] == [
        first_id,
        first_id,
        second_id,
        second_id,
    ]
    assert [item.provider_epoch_id for item in guarded.ledger.terminal_receipts] == [
        "failed-epoch",
        "new-epoch",
    ]
    assert owner.snapshot.state is SelfCaptureSessionState.RUNNING
    await owner.close()


@pytest.mark.asyncio
async def test_input_terminal_retires_audio_without_faulting_active_self_capture() -> None:
    owner, _, provider, sources, _, _ = build_owner()
    await owner.apply_intent(config(), enabled=True)
    guarded = owner.guard_vad_sink()
    try:
        for order, outcome in enumerate(("failed", "submitted"), start=1):
            segment_id = uuid4()
            await guarded.handle_vad_event(
                SpeechStart(
                    segment_id,
                    pre_roll=np.empty((0,), dtype=np.float32),
                    chunk=np.ones((8,), dtype=np.float32),
                )
            )
            if outcome == "submitted":
                await guarded.handle_vad_event(SpeechEnd(segment_id))
            identity = guarded.ledger.snapshots[-1].identity
            owner.note_input_terminal(
                STTProviderInputTerminal(
                    STTProviderTurnIdentity(identity, "epoch", f"input-{order}"),
                    outcome,
                    channel="self",
                    failure_reason="provider_input_failed" if outcome == "failed" else None,
                    recovery_pending=outcome == "failed",
                )
            )
            if outcome == "failed":
                await guarded.handle_vad_event(
                    SpeechChunk(segment_id, chunk=np.ones((8,), dtype=np.float32))
                )
                await guarded.handle_vad_event(SpeechEnd(segment_id))
        assert [receipt.outcome for receipt in guarded.ledger.terminal_receipts] == [
            "failed",
            "submitted",
        ]
        assert owner.snapshot.state is SelfCaptureSessionState.RUNNING
        assert owner.snapshot.desired_active and owner.snapshot.effective_active
        assert sources[0].close_calls == 0
        assert provider.release_calls == []
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_same_effective_self_intent_preserves_stream_ownership_for_next_input() -> None:
    class StreamSink:
        def __init__(self, engine: ScopedRecognitionEngine) -> None:
            self.engine = engine

        async def handle_vad_event(self, event: object) -> None:
            await self.engine.handle_owned_vad_event(event)

        async def handle_stream_input(self, event: object) -> None:
            await self.engine.handle_stream_input(event)

    class StreamSession(BlockingScopedSession):
        accepts_stream_input = True
        independent_recognition_units = True

        def __init__(self) -> None:
            super().__init__()
            self.stream: RecognitionStreamIdentity | None = None
            self.stream_audio: list[bytes] = []

        async def begin_stream(self, stream: RecognitionStreamIdentity) -> None:
            if self.stream is not None and self.stream != stream:
                raise RuntimeError("stream ownership changed without retirement")
            self.stream = stream

        async def begin_turn(self, request: STTProviderTurnRequest) -> None:
            identity = request.identity
            stream = RecognitionStreamIdentity(
                "self",
                identity.segment.activation_generation,
                identity.segment.capture_epoch,
                identity.provider_epoch_id,
                identity.settings_scope,
            )
            if self.stream is not None and self.stream != stream:
                raise RuntimeError("stream ownership changed without retirement")
            self.stream = stream
            self.identities.append(identity)

        async def send_turn_audio(
            self, _identity: STTProviderTurnIdentity, _pcm: bytes, **_kwargs: object
        ) -> None:
            return None

        async def seal_turn(self, identity: STTProviderTurnIdentity, **_kwargs: object) -> None:
            await self.events.put(STTProviderInputTerminal(identity, "submitted", channel="self"))
            assert self.stream is not None
            await self.events.put(
                STTRecognitionUnit(
                    RecognitionUnitIdentity(self.stream, uuid4(), len(self.identities)),
                    f"utterance-{len(self.identities)}",
                )
            )

        async def send_stream_audio(
            self, pcm16le: bytes, *, source_ranges: tuple[AudioCaptureSpan, ...]
        ) -> None:
            assert source_ranges
            self.stream_audio.append(pcm16le)

        async def end_stream(self, *, reason: str) -> None:
            return None

        def recognition_source_covers(self, _ranges: tuple[AudioCaptureSpan, ...]) -> bool:
            return False

    session = StreamSession()
    captured: list[STTRecognitionUnitTerminal] = []
    capture_box: list[SelfCaptureSessionOwner] = []

    async def on_event(event: object) -> None:
        if isinstance(event, STTProviderInputTerminal):
            capture_box[0].note_input_terminal(event)
        elif isinstance(event, STTRecognitionUnitTerminal):
            if capture_box[0].is_current_recognition_stream(event.unit.identity.stream):
                captured.append(event)

    engine = ScopedRecognitionEngine(
        channel="self",
        session_factory=lambda _settings, _epoch: asyncio.sleep(0, result=session),
        event_sink=on_event,
        watchdog_resolver=lambda _settings: STTRecognitionWatchdogs(write_timeout_s=1.0),
    )
    owner, _, _, _, loop, _ = build_owner(sink=StreamSink(engine))
    capture_box.append(owner)
    session_config = replace(
        config(),
        provider_id="gemini_transcribe",
        provider_signature=("gemini_transcribe",),
    )
    try:
        initial = await owner.apply_intent(session_config, enabled=True)
        await loop.started.wait()
        guarded = loop.calls[0]["sink"]
        initial_loop_task = owner.loop_task
        now = asyncio.get_running_loop().time()
        span = AudioCaptureSpan(
            capture_epoch=1,
            callback_sequence=1,
            source_sample_rate_hz=16000,
            source_start_sample=0,
            source_end_sample=8,
            source_start_monotonic_s=now,
            source_end_monotonic_s=now + 8 / 16000,
            normalized_sample_rate_hz=16000,
            normalized_start_sample=0,
            normalized_end_sample=8,
        )

        async def submit_utterance(order: int) -> None:
            segment_id = uuid4()
            current = replace(
                span,
                callback_sequence=order,
                source_start_sample=order * 8,
                source_end_sample=(order + 1) * 8,
                normalized_start_sample=order * 8,
                normalized_end_sample=(order + 1) * 8,
            )
            await guarded.handle_vad_event(
                SpeechStart(
                    segment_id,
                    pre_roll=np.empty((0,), dtype=np.float32),
                    chunk=np.ones((8,), dtype=np.float32),
                    chunk_capture=(current,),
                )
            )
            await guarded.handle_stream_input(
                CaptureStreamInput(np.ones((8,), dtype=np.float32), (current,))
            )
            await guarded.handle_vad_event(SpeechEnd(segment_id))
            await wait_until(lambda: len(captured) == order)

        await submit_utterance(1)
        applied = await owner.apply_intent(session_config, enabled=True)
        assert applied.generation == initial.generation
        assert owner.loop_task is initial_loop_task
        await submit_utterance(2)
        assert [event.unit.text for event in captured] == ["utterance-1", "utterance-2"]
        assert [item.outcome for item in guarded.ledger.terminal_receipts] == [
            "submitted",
            "submitted",
        ]
        assert session.stream is not None
        assert owner.is_current_recognition_stream(session.stream)
        old_stream = session.stream
        changed = replace(session_config, runtime_signature=("runtime", "changed"))
        switched = await owner.apply_intent(changed, enabled=True)
        assert switched.generation > initial.generation
        assert not owner.is_current_recognition_stream(old_stream)
        assert all(
            receipt.identity.activation_generation == initial.generation
            for receipt in guarded.ledger.terminal_receipts
        )
    finally:
        await owner.close()
        await engine.close_backend()


@pytest.mark.asyncio
async def test_changed_self_settings_release_exact_old_input_without_reviving_old_stream() -> None:
    owner, _, _, _, loop, _ = build_owner()
    first_config = config()
    try:
        started = await owner.apply_intent(first_config, enabled=True)
        await loop.started.wait()
        guarded = loop.calls[0]["sink"]
        old_id = uuid4()
        await guarded.handle_vad_event(
            SpeechStart(
                old_id,
                pre_roll=np.empty((0,), dtype=np.float32),
                chunk=np.ones((8,), dtype=np.float32),
            )
        )
        old_identity = guarded.ledger.snapshots[0].identity
        changed = replace(first_config, runtime_signature=("runtime", "new-settings"))
        switched = await owner.apply_intent(changed, enabled=True)
        assert switched.generation > started.generation
        owner.note_input_terminal(
            STTProviderInputTerminal(
                STTProviderTurnIdentity(
                    replace(old_identity, capture_epoch=old_identity.capture_epoch + 1),
                    "old-epoch",
                    "wrong-input",
                ),
                "failed",
                channel="self",
                failure_reason="wrong_input_failed",
            )
        )
        assert guarded.ledger.current_open_segment_id == old_id
        owner.note_input_terminal(
            STTProviderInputTerminal(
                STTProviderTurnIdentity(old_identity, "old-epoch", "old-input"),
                "failed",
                channel="self",
                failure_reason="old_input_failed",
            )
        )
        await guarded.handle_vad_event(SpeechEnd(old_id))
        assert guarded.ledger.current_open_segment_id is None
        assert guarded.ledger.terminal_receipts[0].identity == old_identity
        assert guarded.ledger.terminal_receipts[0].outcome == "failed"
        assert guarded.ledger.terminal_receipts[0].failure_reason == "old_input_failed"

        next_id = uuid4()
        await guarded.handle_vad_event(
            SpeechStart(
                next_id,
                pre_roll=np.empty((0,), dtype=np.float32),
                chunk=np.ones((8,), dtype=np.float32),
            )
        )
        successor = guarded.ledger.snapshots[0]
        assert successor.identity.activation_generation == switched.generation
        assert successor.settings.runtime_signature == changed.runtime_signature
        assert owner.snapshot.state is SelfCaptureSessionState.RUNNING
    finally:
        await owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_retryable", [False, True])
async def test_permanent_or_exhausted_terminal_faults_capture(failure_retryable: bool) -> None:
    owner, _, provider, sources, _, _ = build_owner()
    await owner.apply_intent(config(), enabled=True)
    generation = owner.snapshot.generation
    owner.note_recognition_terminal(
        STTProviderTurnTerminal(
            STTProviderTurnIdentity(
                AudioSegmentIdentity(generation, 1, uuid4(), 1), "epoch", "turn"
            ),
            "failed",
            failure_reason="soniox_receive_failed" if failure_retryable else "soniox_auth_failed",
            failure_retryable=failure_retryable,
            recovery_pending=False,
            epoch_disposition="retire",
        )
    )
    await wait_until(lambda: owner.snapshot.state is SelfCaptureSessionState.FAULTED)
    assert not owner.snapshot.desired_active
    assert sources[0].close_calls == 1
    assert provider.release_calls == [("abort", None)]


@pytest.mark.asyncio
@pytest.mark.parametrize("recovery_pending", [False, True])
async def test_late_old_generation_terminal_does_not_fault_reactivated_capture(
    recovery_pending: bool,
) -> None:
    owner, _, provider, sources, _, _ = build_owner()
    await owner.apply_intent(config(), enabled=True)
    old_generation = owner.snapshot.generation
    await owner.apply_intent(config(), enabled=False)
    await owner.apply_intent(config(), enabled=True)
    owner.note_recognition_terminal(
        STTProviderTurnTerminal(
            STTProviderTurnIdentity(
                AudioSegmentIdentity(old_generation, 1, uuid4(), 1), "old", "old"
            ),
            "failed",
            failure_reason="soniox_receive_failed",
            recovery_pending=recovery_pending,
            epoch_disposition="retire",
        )
    )
    assert owner.snapshot.state is SelfCaptureSessionState.RUNNING
    assert owner.snapshot.desired_active
    assert sources[1].close_calls == 0
    assert provider.release_calls == [("abort", None)]
    await owner.close()


@pytest.mark.asyncio
async def test_explicit_toggle_off_wins_over_late_recoverable_terminal() -> None:
    owner, _, provider, _, _, _ = build_owner()
    await owner.apply_intent(config(), enabled=True)
    generation = owner.snapshot.generation
    await owner.apply_intent(config(), enabled=False)
    owner.note_recognition_terminal(
        STTProviderTurnTerminal(
            STTProviderTurnIdentity(AudioSegmentIdentity(generation, 1, uuid4(), 1), "old", "old"),
            "failed",
            failure_reason="soniox_receive_failed",
            recovery_pending=True,
            epoch_disposition="retire",
        )
    )
    assert owner.snapshot.state is SelfCaptureSessionState.STOPPED
    assert not owner.snapshot.desired_active
    assert provider.release_calls == [("abort", None)]
    await owner.close()


@pytest.mark.asyncio
async def test_superseded_recognition_fault_does_not_deactivate_new_intent() -> None:
    diagnostics: list[SelfCaptureDiagnostic] = []
    owner, _, provider, _, _, _ = build_owner(diagnostics=diagnostics)
    await owner.apply_intent(config(), enabled=True)
    prior_generation = owner.snapshot.generation
    owner.note_recognition_terminal(
        STTProviderTurnTerminal(
            STTProviderTurnIdentity(
                AudioSegmentIdentity(prior_generation, 1, uuid4(), 1),
                "old_epoch",
                "old_turn",
            ),
            "failed",
            failure_reason="soniox_receive_failed",
            epoch_disposition="retire",
        )
    )
    owner.invalidate_intent()
    refreshed = await owner.apply_intent(config(), enabled=True)
    await asyncio.gather(*tuple(owner._fault_tasks))
    assert refreshed.desired_active
    assert owner.snapshot.state is SelfCaptureSessionState.RUNNING
    assert provider.release_calls == []
    assert not [item for item in diagnostics if item.event.value == "failure"]
    await owner.close()


@pytest.mark.asyncio
async def test_self_recognition_admission_keeps_exactly_eight_unsent_and_expires_by_original_age() -> (
    None
):
    sink = BlockingRecognitionSink()
    adapter = SelfCaptureVadSinkAdapter(runtime_provider=lambda: sink)
    owner, _, _, _, _, _ = build_owner(sink=adapter)
    await owner.apply_intent(config(), enabled=True)
    guarded = owner.guard_vad_sink()
    assert guarded.max_wholly_unsent_segments == 8
    assert guarded.wholly_unsent_ttl_s == 12.0
    guarded.wholly_unsent_ttl_s = 0.05
    segment_ids: list[object] = []

    async def admit_segment(order: int) -> None:
        segment_id = uuid4()
        segment_ids.append(segment_id)
        start_sample = order * 8
        span = AudioCaptureSpan(
            capture_epoch=1,
            callback_sequence=order,
            source_sample_rate_hz=16_000,
            source_start_sample=start_sample,
            source_end_sample=start_sample + 8,
            source_start_monotonic_s=asyncio.get_running_loop().time(),
            source_end_monotonic_s=asyncio.get_running_loop().time() + 8 / 16_000,
            normalized_sample_rate_hz=16_000,
            normalized_start_sample=start_sample,
            normalized_end_sample=start_sample + 8,
        )
        await guarded.handle_vad_event(
            SpeechStart(
                segment_id,
                pre_roll=np.empty((0,), dtype=np.float32),
                chunk=np.ones((8,), dtype=np.float32),
                chunk_capture=(span,),
            )
        )
        await guarded.handle_vad_event(SpeechEnd(segment_id))

    await admit_segment(1)
    await sink.started.wait()
    for order in range(2, 12):
        await admit_segment(order)

    assert len(guarded._unsent_segments) == 8
    assert [
        rejection[0].segment.identity.segment_id for rejection in sink.rejections
    ] == segment_ids[1:3]
    assert all(rejection[1] == "recognition_admission_overload" for rejection in sink.rejections)
    await asyncio.sleep(0.07)
    assert guarded._unsent_segments == {}
    assert {rejection[0].segment.identity.segment_id for rejection in sink.rejections} == set(
        segment_ids[1:]
    )
    assert all(rejection[2] == "expired" for rejection in sink.rejections)

    sink.release.set()
    await guarded.abort()
    await owner.apply_intent(config(), enabled=False)


@pytest.mark.asyncio
async def test_production_adapter_routes_buffer_failure_to_immediate_scoped_terminal() -> None:
    sink = BlockingRecognitionSink()
    adapter = SelfCaptureVadSinkAdapter(runtime_provider=lambda: sink)
    owner, _, provider, sources, _, _ = build_owner(sink=adapter)
    await owner.apply_intent(config(), enabled=True)
    guarded = owner.guard_vad_sink()
    guarded.max_retained_samples = 8
    segment_id = uuid4()
    first = AudioCaptureSpan(
        capture_epoch=1,
        callback_sequence=1,
        source_sample_rate_hz=16_000,
        source_start_sample=0,
        source_end_sample=8,
        source_start_monotonic_s=asyncio.get_running_loop().time(),
        source_end_monotonic_s=asyncio.get_running_loop().time() + 8 / 16_000,
        normalized_sample_rate_hz=16_000,
        normalized_start_sample=0,
        normalized_end_sample=8,
    )
    second = replace(
        first,
        callback_sequence=2,
        source_start_sample=8,
        source_end_sample=16,
        normalized_start_sample=8,
        normalized_end_sample=16,
    )

    await guarded.handle_vad_event(
        SpeechStart(
            segment_id,
            pre_roll=np.empty((0,), dtype=np.float32),
            chunk=np.ones((8,), dtype=np.float32),
            chunk_capture=(first,),
        )
    )
    await sink.started.wait()
    await guarded.handle_vad_event(
        SpeechChunk(
            segment_id,
            np.ones((8,), dtype=np.float32),
            chunk_capture=(second,),
        )
    )

    assert [(item[0].segment.identity.segment_id, item[1]) for item in sink.failures] == [
        (segment_id, "buffer_exhausted")
    ]
    assert guarded.ledger.snapshots == ()
    assert [
        (receipt.outcome, receipt.failure_reason) for receipt in guarded.ledger.terminal_receipts
    ] == [("failed", "buffer_exhausted")]
    await wait_until(lambda: owner.snapshot.state is SelfCaptureSessionState.FAULTED)
    assert sources[0].close_calls == 1
    assert provider.release_calls[-1] == ("abort", None)

    sink.release.set()
    await guarded.abort()
    await owner.close()


@pytest.mark.asyncio
async def test_production_self_adapter_routes_exact_pressure_terminals_and_preserves_other_work() -> (
    None
):
    session = BlockingScopedSession()
    engine = ScopedRecognitionEngine(
        channel="self",
        session_factory=lambda _settings, _epoch: asyncio.sleep(0, result=session),
        watchdog_resolver=lambda _settings: STTRecognitionWatchdogs(write_timeout_s=1.0),
        retention_profile_resolver=lambda _settings: STTRetentionProfile(
            max_retained_samples=2_880_000,
            max_retained_bytes=2_880_000 * 4,
            release_after_write=False,
            retained_bytes_per_sample=4,
        ),
    )
    engine.accepted_settings_scope = (
        "provider-one",
        ("provider", "one"),
        ("runtime", "one"),
    )
    osc = RecordingOscQueue()
    translation = compose_translation_test_harness(
        stt=engine,
        llm=None,
        osc=osc,
        low_latency_mode=True,
        ui_queue_maxsize=64,
    )
    await translation.start()
    adapter = SelfCaptureVadSinkAdapter(runtime_provider=lambda: translation.self_owner)
    capture, _, provider, sources, _, _ = build_owner(sink=adapter)
    await capture.apply_intent(config(), enabled=True)
    guarded = capture.guard_vad_sink()
    assert guarded.max_wholly_unsent_segments == 8
    assert guarded.wholly_unsent_ttl_s == 12.0
    guarded.wholly_unsent_ttl_s = 0.05
    segment_ids: list[object] = []

    async def admit_segment(order: int) -> None:
        segment_id = uuid4()
        segment_ids.append(segment_id)
        start_sample = order * 8
        now = asyncio.get_running_loop().time()
        span = AudioCaptureSpan(
            capture_epoch=1,
            callback_sequence=order,
            source_sample_rate_hz=16_000,
            source_start_sample=start_sample,
            source_end_sample=start_sample + 8,
            source_start_monotonic_s=now,
            source_end_monotonic_s=now + 8 / 16_000,
            normalized_sample_rate_hz=16_000,
            normalized_start_sample=start_sample,
            normalized_end_sample=start_sample + 8,
        )
        await guarded.handle_vad_event(
            SpeechStart(
                segment_id,
                pre_roll=np.empty((0,), dtype=np.float32),
                chunk=np.ones((8,), dtype=np.float32),
                chunk_capture=(span,),
            )
        )
        await guarded.handle_vad_event(SpeechEnd(segment_id))

    await admit_segment(1)
    await session.first_send_started.wait()
    for order in range(2, 12):
        await admit_segment(order)

    assert len(guarded._unsent_segments) == 8
    assert [receipt.failure_reason for receipt in guarded.ledger.terminal_receipts] == [
        "recognition_admission_overload",
        "recognition_admission_overload",
    ]
    await asyncio.sleep(0.07)
    assert guarded._unsent_segments == {}
    assert {receipt.failure_reason for receipt in guarded.ledger.terminal_receipts} == {
        "recognition_admission_overload",
        "recognition_admission_timeout",
    }

    session.release_first_send.set()
    await wait_until(lambda: not guarded._queue)
    await wait_until(lambda: capture._retention_budget.used_bytes == 0)
    capture._retention_budget.capacity_bytes = 80
    buffer_segment_id = uuid4()
    now = asyncio.get_running_loop().time()
    first = AudioCaptureSpan(
        capture_epoch=1,
        callback_sequence=20,
        source_sample_rate_hz=16_000,
        source_start_sample=160,
        source_end_sample=168,
        source_start_monotonic_s=now,
        source_end_monotonic_s=now + 8 / 16_000,
        normalized_sample_rate_hz=16_000,
        normalized_start_sample=160,
        normalized_end_sample=168,
    )
    second = replace(
        first,
        callback_sequence=21,
        source_start_sample=168,
        source_end_sample=176,
        normalized_start_sample=168,
        normalized_end_sample=176,
    )
    await guarded.handle_vad_event(
        SpeechStart(
            buffer_segment_id,
            pre_roll=np.empty((0,), dtype=np.float32),
            chunk=np.ones((8,), dtype=np.float32),
            chunk_capture=(first,),
        )
    )
    await session.second_send_started.wait()
    assert capture._retention_budget.used_bytes == 80
    assert capture._retention_budget.high_water_bytes >= 80
    await guarded.handle_vad_event(
        SpeechChunk(
            buffer_segment_id,
            np.ones((8,), dtype=np.float32),
            chunk_capture=(second,),
        )
    )

    await wait_until(lambda: capture.snapshot.state is SelfCaptureSessionState.FAULTED)
    buffer_receipts = [
        receipt
        for receipt in guarded.ledger.terminal_receipts
        if receipt.identity.segment_id == buffer_segment_id
    ]
    assert [(receipt.outcome, receipt.failure_reason) for receipt in buffer_receipts] == [
        ("failed", "buffer_exhausted")
    ]
    assert sources[0].close_calls == 1
    assert provider.release_calls[-1] == ("abort", None)

    peer_id = uuid4()
    peer_bundle = translation.peer_runtime.get_or_create_bundle(peer_id)
    await translation.self_owner.submit_text("manual-after-production-pressure")
    assert translation.peer_runtime.get_or_create_bundle(peer_id) is peer_bundle
    assert any(message.text == "manual-after-production-pressure" for message in osc.messages)
    await wait_until(
        lambda: (
            sum(event.type.value == "ERROR" for event in tuple(translation.ui_events._queue)) == 11
        )
    )

    session.release_second_send.set()
    await guarded.abort()
    assert capture._retention_budget.used_bytes == 0
    await capture.close()
    await translation.stop()


@pytest.mark.asyncio
async def test_latest_intent_cancels_pending_admission_without_opening_source() -> None:
    gate = asyncio.Event()
    admission = RecordingAdmission(gate=gate)
    owner, _, provider, sources, _, _ = build_owner(admission=admission)
    session_config = config()

    enable_task = asyncio.create_task(owner.apply_intent(session_config, enabled=True))
    await wait_until(lambda: len(admission.calls) == 1)
    disable_snapshot = await owner.apply_intent(session_config, enabled=False)
    gate.set()
    enable_snapshot = await enable_task

    assert disable_snapshot.state is SelfCaptureSessionState.STOPPED
    assert enable_snapshot.state is SelfCaptureSessionState.STOPPED
    assert sources == []
    assert provider.replace_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mutation_status", "state", "provider_status", "failure"),
    [
        (
            SelfCaptureProviderMutationStatus.PENDING,
            SelfCaptureSessionState.ADMISSION_PENDING,
            SelfCaptureProviderStatus.PENDING,
            None,
        ),
        (
            SelfCaptureProviderMutationStatus.FAILED,
            SelfCaptureSessionState.FAULTED,
            SelfCaptureProviderStatus.FAILED,
            SelfCaptureFailureReason.PROVIDER_FAILED,
        ),
        (
            SelfCaptureProviderMutationStatus.SUPERSEDED,
            SelfCaptureSessionState.STOPPED,
            SelfCaptureProviderStatus.DETACHED,
            None,
        ),
    ],
)
async def test_provider_pending_failure_and_supersession_are_explicit(
    mutation_status: SelfCaptureProviderMutationStatus,
    state: SelfCaptureSessionState,
    provider_status: SelfCaptureProviderStatus,
    failure: SelfCaptureFailureReason | None,
) -> None:
    provider = RecordingProvider()
    provider.replace_result = SelfCaptureProviderMutation(mutation_status, reason="provider-state")
    owner, _, _, sources, _, _ = build_owner(provider=provider)

    snapshot = await owner.apply_intent(config(), enabled=True)

    assert snapshot.state is state
    assert snapshot.provider_status is provider_status
    assert snapshot.failure_reason is failure
    assert sources == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutation_status",
    [
        SelfCaptureProviderMutationStatus.APPLIED,
        SelfCaptureProviderMutationStatus.PENDING,
        SelfCaptureProviderMutationStatus.FAILED,
        SelfCaptureProviderMutationStatus.SUPERSEDED,
    ],
)
async def test_running_provider_handoff_preserves_capture_and_prior_state_on_non_apply(
    mutation_status: SelfCaptureProviderMutationStatus,
) -> None:
    provider = RecordingProvider()
    diagnostics: list[SelfCaptureDiagnostic] = []
    owner, _, _, sources, _, _ = build_owner(provider=provider, diagnostics=diagnostics)
    first = config("one")
    second = config("two")

    await owner.apply_intent(first, enabled=True)
    first_task = owner.loop_task
    provider.handoff_result = SelfCaptureProviderMutation(
        mutation_status,
        reason="provider_readiness_unavailable",
    )

    snapshot = await owner.apply_intent(second, enabled=True)

    assert owner.loop_task is first_task
    assert sources[0].close_calls == 0
    assert snapshot.state is SelfCaptureSessionState.RUNNING
    if mutation_status is SelfCaptureProviderMutationStatus.APPLIED:
        assert snapshot.provider_id == second.provider_id
        assert snapshot.failure_reason is None
    else:
        assert snapshot.provider_id == first.provider_id
        if mutation_status is SelfCaptureProviderMutationStatus.FAILED:
            assert snapshot.failure_reason is SelfCaptureFailureReason.PROVIDER_FAILED
            assert [
                diagnostic.detail
                for diagnostic in diagnostics
                if diagnostic.reason is SelfCaptureFailureReason.PROVIDER_FAILED
            ] == ["provider_readiness_unavailable"]

    await owner.close()


@pytest.mark.asyncio
async def test_retained_capture_rebinds_callbacks_and_source_loss_after_provider_handoff() -> None:
    owner, _, _, sources, loop, sink = build_owner()
    first = config("one")
    second = config("two")

    await owner.apply_intent(first, enabled=True)
    guarded_sink = loop.calls[0]["sink"]
    await getattr(guarded_sink, "handle_vad_event")("before")

    await owner.apply_intent(first, enabled=True)
    await getattr(guarded_sink, "handle_vad_event")("after-noop")
    await owner.apply_intent(second, enabled=True)
    await getattr(guarded_sink, "handle_vad_event")("after-handoff")
    loop.release.set()
    await wait_until(lambda: owner.snapshot.state is SelfCaptureSessionState.FAULTED)

    assert sink.events == ["before", "after-noop", "after-handoff"]
    assert sources[0].close_calls == 1
    assert owner.snapshot.failure_reason is SelfCaptureFailureReason.SESSION_FAILED


@pytest.mark.asyncio
async def test_superseded_handoff_keeps_retained_callbacks_current_during_cancellation() -> None:
    class BlockingHandoffProvider(RecordingProvider):
        def __init__(self) -> None:
            super().__init__()
            self.handoff_started = asyncio.Event()
            self.cancel_started = asyncio.Event()
            self.cancel_release = asyncio.Event()
            self.handoff_count = 0

        async def handoff(
            self,
            request: object,
            *,
            start: bool,
            on_terminal_failure: Callable[[Exception], Awaitable[None]],
        ) -> SelfCaptureProviderMutation:
            self.handoff_count += 1
            if self.handoff_count == 1:
                self.handoff_started.set()
                await asyncio.Event().wait()
            return await super().handoff(
                request,
                start=start,
                on_terminal_failure=on_terminal_failure,
            )

        async def cancel_handoff(self) -> bool:
            self.cancel_started.set()
            await self.cancel_release.wait()
            return await super().cancel_handoff()

    provider = BlockingHandoffProvider()
    owner, _, _, sources, loop, sink = build_owner(provider=provider)
    await owner.apply_intent(config("one"), enabled=True)
    guarded_sink = loop.calls[0]["sink"]

    first_handoff = asyncio.create_task(owner.apply_intent(config("two"), enabled=True))
    await provider.handoff_started.wait()
    second_handoff = asyncio.create_task(owner.apply_intent(config("three"), enabled=True))
    await provider.cancel_started.wait()

    await getattr(guarded_sink, "handle_vad_event")("during-cancellation")
    loop.release.set()
    provider.cancel_release.set()
    await asyncio.gather(first_handoff, second_handoff)
    await wait_until(lambda: owner.snapshot.state is SelfCaptureSessionState.FAULTED)

    assert sink.events == ["during-cancellation"]
    assert sources[0].close_calls == 1
    assert owner.snapshot.failure_reason is SelfCaptureFailureReason.SESSION_FAILED


@pytest.mark.asyncio
async def test_superseded_start_retries_incomplete_provider_ingress() -> None:
    class BlockingIngressProvider(RecordingProvider):
        def __init__(self) -> None:
            super().__init__()
            self.ingress_started = asyncio.Event()
            self.completed_starts = 0

        async def start_ingress(self) -> None:
            self.start_calls += 1
            if self.start_calls == 1:
                self.ingress_started.set()
                await asyncio.Event().wait()
            self.completed_starts += 1

    provider = BlockingIngressProvider()
    owner, _, _, sources, _, _ = build_owner(provider=provider)
    session_config = config()

    first_start = asyncio.create_task(owner.apply_intent(session_config, enabled=True))
    await provider.ingress_started.wait()
    assert owner.snapshot.state is SelfCaptureSessionState.STARTING
    second_start = asyncio.create_task(owner.apply_intent(session_config, enabled=True))
    await asyncio.gather(first_start, second_start)

    assert owner.snapshot.state is SelfCaptureSessionState.RUNNING
    assert owner.snapshot.effective_active is True
    assert provider.start_calls == 2
    assert provider.completed_starts == 1
    assert len(sources) == 2
    assert sources[0].close_calls == 1
    assert sources[1].close_calls == 0

    await owner.close()


@pytest.mark.asyncio
async def test_microphone_test_exclusion_stops_ingress_before_abort_release() -> None:
    events: list[str] = []

    class OrderedProvider(RecordingProvider):
        async def release(
            self,
            *,
            mode: Literal["drain", "abort"],
            release_backend_after: float | None = None,
        ) -> None:
            events.append("provider-release")
            await super().release(mode=mode, release_backend_after=release_backend_after)

    class OrderedSource(RecordingSource):
        async def close(self) -> None:
            events.append("source-close")
            await super().close()

    source = OrderedSource()
    provider = OrderedProvider()
    owner, _, _, _, _, _ = build_owner(provider=provider)
    owner._source_factory = lambda _config: source

    await owner.apply_intent(config(local_cpu=True), enabled=True)
    snapshot = await owner.release_for_microphone_test()

    assert snapshot.state is SelfCaptureSessionState.STOPPED
    assert events == ["source-close", "provider-release"]
    assert provider.release_calls == [("abort", None)]


@pytest.mark.asyncio
async def test_terminal_provider_failure_faults_only_current_generation_and_cleans_session() -> (
    None
):
    owner, _, provider, sources, _, _ = build_owner()

    await owner.apply_intent(config(), enabled=True)
    handler = provider.terminal_failure_handler
    assert handler is not None
    await handler(RuntimeError("terminal provider failure"))

    assert owner.snapshot.state is SelfCaptureSessionState.FAULTED
    assert owner.snapshot.failure_reason is SelfCaptureFailureReason.PROVIDER_FAILED
    assert sources[0].close_calls == 1
    assert provider.release_calls == [("abort", None)]


@pytest.mark.asyncio
async def test_prepared_provider_terminal_failure_faults_enabled_session() -> None:
    owner, _, provider, sources, _, _ = build_owner()
    session_config = config()

    await owner.prepare_provider(session_config)
    handler = provider.terminal_failure_handler
    assert handler is not None
    await owner.apply_intent(session_config, enabled=True)
    await handler(RuntimeError("prepared provider terminal failure"))

    assert owner.snapshot.state is SelfCaptureSessionState.FAULTED
    assert owner.snapshot.failure_reason is SelfCaptureFailureReason.PROVIDER_FAILED
    assert sources[0].close_calls == 1
    assert provider.release_calls == [("abort", None)]


@pytest.mark.asyncio
async def test_current_provider_terminal_failure_survives_noop_generation_change() -> None:
    owner, _, provider, sources, _, _ = build_owner()
    session_config = config()

    await owner.apply_intent(session_config, enabled=True)
    handler = provider.terminal_failure_handler
    assert handler is not None
    await owner.apply_intent(session_config, enabled=True)
    await handler(RuntimeError("same provider terminal failure"))

    assert owner.snapshot.state is SelfCaptureSessionState.FAULTED
    assert owner.snapshot.failure_reason is SelfCaptureFailureReason.PROVIDER_FAILED
    assert sources[0].close_calls == 1
    assert provider.release_calls == [("abort", None)]
    assert provider.handoff_calls == []
    assert provider.reconfigure_calls == []


@pytest.mark.asyncio
async def test_retired_provider_terminal_failure_cannot_fault_handoff_session() -> None:
    owner, _, provider, sources, _, _ = build_owner()

    await owner.apply_intent(config("one"), enabled=True)
    retired_handler = provider.terminal_failure_handler
    assert retired_handler is not None
    await owner.apply_intent(config("two"), enabled=True)
    await retired_handler(RuntimeError("retired provider terminal failure"))

    assert owner.snapshot.state is SelfCaptureSessionState.RUNNING
    assert owner.snapshot.provider_id == "provider-two"
    assert owner.snapshot.failure_reason is None
    assert sources[0].close_calls == 0
    assert provider.release_calls == []

    await owner.close()


@pytest.mark.asyncio
async def test_reused_provider_signature_rejects_retired_attachment_failure() -> None:
    owner, _, provider, sources, _, _ = build_owner()

    await owner.apply_intent(config("one"), enabled=True)
    retired_handler = provider.terminal_failure_handler
    assert retired_handler is not None
    await owner.apply_intent(config("two"), enabled=True)
    await owner.apply_intent(config("one"), enabled=True)
    current_handler = provider.terminal_failure_handler
    assert current_handler is not None
    assert current_handler is not retired_handler
    await retired_handler(RuntimeError("retired reused-signature provider failure"))

    assert owner.snapshot.state is SelfCaptureSessionState.RUNNING
    assert owner.snapshot.provider_id == "provider-one"
    assert owner.snapshot.failure_reason is None
    assert sources[0].close_calls == 0
    assert provider.release_calls == []

    await owner.close()


@pytest.mark.asyncio
async def test_same_signature_release_and_rebuild_rejects_retired_attachment_failure() -> None:
    owner, _, provider, sources, _, _ = build_owner()
    session_config = config()

    await owner.apply_intent(session_config, enabled=True)
    retired_handler = provider.terminal_failure_handler
    assert retired_handler is not None
    await owner.apply_intent(session_config, enabled=False)
    await owner.apply_intent(session_config, enabled=True)
    current_handler = provider.terminal_failure_handler
    assert current_handler is not None
    assert current_handler is not retired_handler
    await retired_handler(RuntimeError("released provider terminal failure"))

    assert owner.snapshot.state is SelfCaptureSessionState.RUNNING
    assert owner.snapshot.provider_id == session_config.provider_id
    assert owner.snapshot.failure_reason is None
    assert sources[0].close_calls == 1
    assert sources[1].close_calls == 0
    assert provider.release_calls == [("abort", None)]

    await owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "cause", ["provider_unavailable", "gpu_not_ready", "self_channel_inactive"]
)
async def test_provider_ingress_start_failure_completes_and_releases_owned_resources(
    cause: Literal["provider_unavailable", "gpu_not_ready", "self_channel_inactive"],
) -> None:
    provider = RecordingProvider()
    provider.start_failure = SelfCaptureIngressError(cause)
    diagnostics: list[SelfCaptureDiagnostic] = []
    owner, _, _, sources, _, _ = build_owner(provider=provider, diagnostics=diagnostics)

    snapshot = await asyncio.wait_for(
        owner.apply_intent(config(), enabled=True),
        timeout=1.0,
    )

    assert snapshot.state is SelfCaptureSessionState.FAULTED
    assert snapshot.failure_reason is SelfCaptureFailureReason.PROVIDER_FAILED
    assert snapshot.has_source is False
    assert snapshot.has_vad is False
    assert snapshot.has_loop_task is False
    assert sources[0].close_calls == 1
    assert provider.release_calls == [("abort", None)]
    failures = [
        item for item in diagnostics if item.reason is SelfCaptureFailureReason.PROVIDER_FAILED
    ]
    assert len(failures) == 1
    assert failures[0].failure_stage == "ingress"
    assert failures[0].failure_code == cause
    assert failures[0].failure_type == "SelfCaptureIngressError"
    assert failures[0].desired_active_after is False


@pytest.mark.asyncio
async def test_loop_failure_is_contained_and_releases_all_owned_resources() -> None:
    loop = LoopHarness()
    loop.failure = RuntimeError("loop failure")
    owner, _, provider, sources, _, _ = build_owner(loop=loop)

    await owner.apply_intent(config(), enabled=True)
    await wait_until(lambda: owner.snapshot.state is SelfCaptureSessionState.FAULTED)

    assert owner.snapshot.failure_reason is SelfCaptureFailureReason.SESSION_FAILED
    assert owner.snapshot.has_loop_task is False
    assert sources[0].close_calls == 1
    assert provider.release_calls == [("abort", None)]


@pytest.mark.asyncio
async def test_repeated_close_retries_retained_source_and_shutdown_abort_is_idempotent() -> None:
    source = RecordingSource([RuntimeError("first close failure")])
    provider = RecordingProvider()
    gate_resets: list[str] = []
    owner, _, _, _, _, _ = build_owner(provider=provider, gate_resets=gate_resets)
    owner._source_factory = lambda _config: source

    await owner.apply_intent(config(), enabled=True)

    with pytest.raises(RuntimeError, match="first close failure"):
        await owner.close()

    assert owner.snapshot.cleanup_debt == 1
    assert owner.snapshot.failure_reason is SelfCaptureFailureReason.CLEANUP_FAILED

    await owner.close()
    await owner.close()

    assert owner.snapshot.state is SelfCaptureSessionState.STOPPED
    assert owner.snapshot.cleanup_debt == 0
    assert source.close_calls == 2
    assert provider.release_calls[0] == ("abort", None)
    assert gate_resets


@pytest.mark.asyncio
async def test_close_cancels_pending_admission_and_rejects_future_intent() -> None:
    gate = asyncio.Event()
    admission = RecordingAdmission(gate=gate)
    owner, _, provider, sources, _, _ = build_owner(admission=admission)
    session_config = config()
    transition = asyncio.create_task(owner.apply_intent(session_config, enabled=True))
    await wait_until(lambda: len(admission.calls) == 1)

    await owner.close()
    gate.set()
    await transition

    assert owner.snapshot.closed is True
    assert owner.snapshot.state is SelfCaptureSessionState.STOPPED
    assert sources == []
    assert provider.replace_calls == []
    with pytest.raises(RuntimeError, match="closed"):
        await owner.apply_intent(session_config, enabled=True)


@pytest.mark.asyncio
async def test_prepare_provider_replaces_when_provider_identity_changes() -> None:
    owner, _, provider, sources, _, _ = build_owner()

    await owner.prepare_provider(config("one"))
    snapshot = await owner.prepare_provider(config("two"))

    assert provider.replace_calls == [
        (("provider-one", False), False),
        (("provider-two", False), False),
    ]
    assert snapshot.runtime_signature == config("two").runtime_signature
    assert sources == []
    await owner.close()


@pytest.mark.asyncio
async def test_prepare_provider_reuses_ready_provider_with_same_identity() -> None:
    owner, _, provider, _, _, _ = build_owner()
    session_config = config("one")

    await owner.prepare_provider(session_config)
    await owner.prepare_provider(session_config)

    assert provider.replace_calls == [(("provider-one", False), False)]
    await owner.close()


@pytest.mark.asyncio
async def test_start_replaces_ready_provider_when_identity_changed() -> None:
    owner, _, provider, _, _, _ = build_owner()

    await owner.prepare_provider(config("one"))
    snapshot = await owner.apply_intent(config("two"), enabled=True)

    assert ("provider-two", False) in {request for request, _start in provider.replace_calls}
    assert snapshot.runtime_signature == config("two").runtime_signature
    await owner.close()


@pytest.mark.asyncio
async def test_prepare_provider_attaches_without_opening_capture_resources() -> None:
    owner, _, provider, sources, _, _ = build_owner()
    session_config = config(local_cpu=True)

    snapshot = await owner.prepare_provider(session_config)

    assert snapshot.state is SelfCaptureSessionState.STOPPED
    assert snapshot.provider_status is SelfCaptureProviderStatus.READY
    assert snapshot.desired_active is False
    assert provider.replace_calls == [(("provider-one", False), False)]
    assert sources == []

    await owner.close()


@pytest.mark.asyncio
async def test_suspend_and_recovery_preserve_intent_without_releasing_provider() -> None:
    owner, _, provider, sources, _, _ = build_owner()
    session_config = config(local_gpu=True)

    await owner.apply_intent(session_config, enabled=True)
    retired_handler = provider.terminal_failure_handler
    assert retired_handler is not None
    suspended = await owner.suspend_provider_consumer()

    assert suspended.state is SelfCaptureSessionState.STOPPED
    assert suspended.desired_active is True
    assert suspended.provider_status is SelfCaptureProviderStatus.READY
    assert sources[0].close_calls == 1
    assert provider.release_calls == []

    recovered_handler = owner.prepare_provider_recovery(session_config)
    provider.terminal_failure_handler = recovered_handler
    recovered = await owner.adopt_recovered_provider(
        session_config,
        on_terminal_failure=recovered_handler,
    )
    resumed = await owner.apply_intent(session_config, enabled=True)
    await retired_handler(RuntimeError("retired pre-recovery provider failure"))

    assert recovered.provider_status is SelfCaptureProviderStatus.READY
    assert resumed.state is SelfCaptureSessionState.RUNNING
    assert provider.replace_calls == [(("provider-one", True), False)]
    assert sources[1].close_calls == 0

    await recovered_handler(RuntimeError("recovered provider failure"))

    assert owner.snapshot.state is SelfCaptureSessionState.FAULTED
    assert owner.snapshot.failure_reason is SelfCaptureFailureReason.PROVIDER_FAILED
    assert sources[1].close_calls == 1
    assert provider.release_calls == [("abort", None)]

    await owner.close()


@pytest.mark.asyncio
async def test_recovered_provider_failure_before_adoption_is_contained_on_commit() -> None:
    owner, _, provider, sources, _, _ = build_owner()
    session_config = config(local_gpu=True)

    await owner.apply_intent(session_config, enabled=True)
    await owner.suspend_provider_consumer()
    recovered_handler = owner.prepare_provider_recovery(session_config)
    provider.terminal_failure_handler = recovered_handler
    await recovered_handler(RuntimeError("recovered provider failed before adoption"))

    snapshot = await owner.adopt_recovered_provider(
        session_config,
        on_terminal_failure=recovered_handler,
    )

    assert snapshot.state is SelfCaptureSessionState.FAULTED
    assert snapshot.failure_reason is SelfCaptureFailureReason.PROVIDER_FAILED
    assert snapshot.has_source is False
    assert snapshot.has_vad is False
    assert snapshot.has_loop_task is False
    assert sources[0].close_calls == 1
    assert provider.release_calls == [("abort", None)]


@pytest.mark.asyncio
async def test_current_provider_failure_during_recovery_preparation_is_contained() -> None:
    owner, _, provider, sources, _, _ = build_owner()
    session_config = config(local_gpu=True)

    await owner.apply_intent(session_config, enabled=True)
    current_handler = provider.terminal_failure_handler
    assert current_handler is not None
    pending_handler = owner.prepare_provider_recovery(session_config)
    await current_handler(RuntimeError("current provider failed before quiesce"))

    assert owner.snapshot.state is SelfCaptureSessionState.FAULTED
    assert owner.snapshot.failure_reason is SelfCaptureFailureReason.PROVIDER_FAILED
    assert sources[0].close_calls == 1
    assert provider.release_calls == [("abort", None)]
    assert owner.abort_provider_recovery(pending_handler) is False

    provider.ready = True
    provider.terminal_failure_handler = pending_handler
    with pytest.raises(RuntimeError, match="no matching owner callback"):
        await owner.adopt_recovered_provider(
            session_config,
            on_terminal_failure=pending_handler,
        )

    assert provider.ready is False
    assert provider.release_calls == [("abort", None), ("abort", None)]


@pytest.mark.asyncio
async def test_aborted_provider_recovery_retains_current_failure_callback() -> None:
    owner, _, provider, sources, _, _ = build_owner()
    session_config = config(local_gpu=True)

    await owner.apply_intent(session_config, enabled=True)
    current_handler = provider.terminal_failure_handler
    assert current_handler is not None
    pending_handler = owner.prepare_provider_recovery(session_config)

    assert owner.abort_provider_recovery(pending_handler) is True
    assert owner.abort_provider_recovery(pending_handler) is False
    await current_handler(RuntimeError("current provider failed after recovery abort"))

    assert owner.snapshot.state is SelfCaptureSessionState.FAULTED
    assert owner.snapshot.failure_reason is SelfCaptureFailureReason.PROVIDER_FAILED
    assert sources[0].close_calls == 1
    assert provider.release_calls == [("abort", None)]


@pytest.mark.asyncio
async def test_overlapping_recoveries_adopt_their_exact_provider_callbacks() -> None:
    owner, _, provider, sources, _, _ = build_owner()
    session_config = config(local_gpu=True)

    await owner.apply_intent(session_config, enabled=True)
    await owner.suspend_provider_consumer()
    first_handler = owner.prepare_provider_recovery(session_config)
    second_handler = owner.prepare_provider_recovery(session_config)

    provider.terminal_failure_handler = first_handler
    await owner.adopt_recovered_provider(
        session_config,
        on_terminal_failure=first_handler,
    )
    await owner.apply_intent(session_config, enabled=True)
    await owner.suspend_provider_consumer()
    provider.terminal_failure_handler = second_handler
    await owner.adopt_recovered_provider(
        session_config,
        on_terminal_failure=second_handler,
    )
    await owner.apply_intent(session_config, enabled=True)
    await first_handler(RuntimeError("first recovered provider retired"))

    assert owner.snapshot.state is SelfCaptureSessionState.RUNNING
    assert sources[2].close_calls == 0

    await second_handler(RuntimeError("second recovered provider failed"))

    assert owner.snapshot.state is SelfCaptureSessionState.FAULTED
    assert owner.snapshot.failure_reason is SelfCaptureFailureReason.PROVIDER_FAILED
    assert sources[2].close_calls == 1
    assert provider.release_calls == [("abort", None)]


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["activate", "prepare", "handoff"])
@pytest.mark.parametrize("stage", ["provider_build", "provider_warmup", "readiness"])
async def test_failed_provider_mutation_preserves_startup_failure_identity(
    operation: str,
    stage: str,
) -> None:
    provider = RecordingProvider()
    diagnostics: list[SelfCaptureDiagnostic] = []
    owner, _, _, sources, _, _ = build_owner(provider=provider, diagnostics=diagnostics)
    mutation = SelfCaptureProviderMutation(
        SelfCaptureProviderMutationStatus.FAILED,
        reason="worker_process_exited",
        failure_code="worker_process_exited",
        failure_type="GpuWorkerClosedError",
        failure_stage=stage,
    )
    try:
        if operation == "handoff":
            await owner.apply_intent(config("one"), enabled=True)
            provider.handoff_result = mutation
            snapshot = await owner.apply_intent(config("two"), enabled=True)
            assert snapshot.effective_active is True
            assert snapshot.provider_id == "provider-one"
            assert sources[0].close_calls == 0
        else:
            provider.replace_result = mutation
            if operation == "prepare":
                snapshot = await owner.prepare_provider(config())
            else:
                snapshot = await owner.apply_intent(config(), enabled=True)
            assert snapshot.state is SelfCaptureSessionState.FAULTED
            assert snapshot.desired_active is False
            assert sources == []
        failures = [
            item for item in diagnostics if item.reason is SelfCaptureFailureReason.PROVIDER_FAILED
        ]
        assert len(failures) == 1
        assert failures[0].detail == mutation.reason
        assert failures[0].failure_code == mutation.failure_code
        assert failures[0].failure_type == mutation.failure_type
        assert failures[0].failure_stage == stage
    finally:
        await owner.close()
