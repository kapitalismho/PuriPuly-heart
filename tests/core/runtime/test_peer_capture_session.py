from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from uuid import uuid4

import numpy as np
import pytest

from puripuly_heart.core.audio.format import AudioCaptureSpan, AudioFrameF32
from puripuly_heart.core.clock import FakeClock
from puripuly_heart.core.peer_capture import (
    PeerCaptureAdmission,
    PeerCaptureAdmissionStatus,
    PeerCapturedFinalFacts,
    PeerCaptureFinalLanguageState,
    PeerCaptureLanguageFacts,
    PeerCaptureProviderMutation,
    PeerCaptureProviderMutationStatus,
    PeerCaptureProviderStatus,
    PeerCaptureResolvedTarget,
    PeerCaptureSessionConfig,
    PeerCaptureSessionState,
    PeerCaptureTargetIntent,
    PeerCaptureTargetResolution,
    PeerCaptureTargetStatus,
)
from puripuly_heart.core.runtime.audio_vad_loop import run_audio_vad_loop
from puripuly_heart.core.runtime.peer_channel import PeerCaptureSessionOwner
from puripuly_heart.core.vad.gating import SpeechEnd, SpeechStart, VadGating
from tests.helpers.vad import SequenceVadEngine


@dataclass(slots=True)
class FakeSource:
    close_calls: int = 0
    terminal_reason: str | None = None

    async def close(self) -> None:
        self.close_calls += 1


class FailingCloseSource:
    def __init__(self) -> None:
        self.close_calls = 0

    async def close(self) -> None:
        self.close_calls += 1
        if self.close_calls == 1:
            raise RuntimeError("close failed")


class FakeAdmission:
    def __init__(self) -> None:
        self.result = PeerCaptureAdmission(PeerCaptureAdmissionStatus.ADMITTED)
        self.calls: list[PeerCaptureSessionConfig] = []
        self.gate: asyncio.Event | None = None

    async def admit(self, config: PeerCaptureSessionConfig) -> PeerCaptureAdmission:
        self.calls.append(config)
        if self.gate is not None:
            await self.gate.wait()
        return self.result


class FakeTargetResolver:
    def __init__(self) -> None:
        self.calls: list[PeerCaptureTargetIntent] = []
        self.results: list[PeerCaptureTargetResolution] = []
        self.gate: asyncio.Event | None = None

    async def resolve(
        self,
        target: PeerCaptureTargetIntent,
    ) -> PeerCaptureTargetResolution:
        self.calls.append(target)
        if self.gate is not None:
            await self.gate.wait()
        if self.results:
            return self.results.pop(0)
        return PeerCaptureTargetResolution(
            PeerCaptureTargetStatus.RESOLVED,
            target=PeerCaptureResolvedTarget(intent=target),
        )


class FakeProvider:
    def __init__(self) -> None:
        self.provider_id: str | None = None
        self.requests: list[object] = []
        self.handoffs: list[object] = []
        self.releases: list[tuple[str, float | None]] = []
        self.reconfigurations: list[object] = []
        self.start_calls = 0
        self.warmup_calls = 0
        self.cancel_calls = 0
        self.replace_result = PeerCaptureProviderMutation(PeerCaptureProviderMutationStatus.APPLIED)
        self.start_gate: asyncio.Event | None = None
        self.handoff_result = PeerCaptureProviderMutation(PeerCaptureProviderMutationStatus.APPLIED)
        self.replace_gate: asyncio.Event | None = None
        self.handoff_gate: asyncio.Event | None = None
        self.start_error: Exception | None = None
        self.replace_terminal_error: Exception | None = None
        self.terminal_handlers: list = []

    def is_ready(self, config: PeerCaptureSessionConfig) -> bool:
        return self.provider_id == config.provider_id

    async def replace(self, request, *, start: bool, on_terminal_failure):
        _ = start
        self.requests.append(request)
        self.terminal_handlers.append(on_terminal_failure)
        if self.replace_terminal_error is not None:
            await on_terminal_failure(self.replace_terminal_error)
        if self.replace_gate is not None:
            await self.replace_gate.wait()
        if self.replace_result.status is PeerCaptureProviderMutationStatus.APPLIED:
            self.provider_id = request[0]
        return self.replace_result

    async def handoff(self, request, *, start: bool, on_terminal_failure):
        _ = start
        self.handoffs.append(request)
        self.terminal_handlers.append(on_terminal_failure)
        if self.handoff_gate is not None:
            await self.handoff_gate.wait()
        if self.handoff_result.status is PeerCaptureProviderMutationStatus.APPLIED:
            self.provider_id = request[0]
        return self.handoff_result

    async def cancel_handoff(self) -> bool:
        self.cancel_calls += 1
        return True

    async def start_ingress(self) -> None:
        self.start_calls += 1
        if self.start_gate is not None:
            await self.start_gate.wait()
        if self.start_error is not None:
            raise self.start_error

    async def warmup(self) -> None:
        self.warmup_calls += 1

    async def reconfigure(self, session_options: object) -> None:
        self.reconfigurations.append(session_options)

    async def release(
        self,
        *,
        mode: str,
        release_backend_after: float | None = None,
    ) -> None:
        self.releases.append((mode, release_backend_after))
        if mode == "abort":
            self.provider_id = None


class FakeVadSink:
    def __init__(self) -> None:
        self.events: list[object] = []

    async def handle_vad_event(self, event: object) -> None:
        self.events.append(event)


def make_config(
    *,
    provider_id: str = "soniox",
    target: PeerCaptureTargetIntent | None = None,
    language: PeerCaptureLanguageFacts | None = None,
    capture_signature: tuple[object, ...] | None = None,
) -> PeerCaptureSessionConfig:
    resolved_target = target or PeerCaptureTargetIntent(kind="default_output_device")
    language_facts = language or PeerCaptureLanguageFacts(
        source_mode="manual",
        source_language="ko",
    )
    signature = capture_signature or (resolved_target, 16000, 0.6, 900, 500)
    return PeerCaptureSessionConfig(
        provider_id=provider_id,
        provider_signature=(provider_id, language_facts),
        runtime_signature=(provider_id, signature, language_facts),
        capture_signature=signature,
        capture_target=resolved_target,
        language=language_facts,
        target_sample_rate_hz=16000,
        vad_speech_threshold=0.6,
        vad_hangover_ms=900,
        vad_pre_roll_ms=500,
        local_provider=provider_id.startswith("local_"),
        release_backend_after=600.0 if provider_id == "local_qwen" else None,
    )


def make_owner(
    *,
    admission: FakeAdmission | None = None,
    resolver: FakeTargetResolver | None = None,
    provider: FakeProvider | None = None,
    sources: list[object] | None = None,
    source_factory=None,
    vad_factory=None,
    run_audio_loop=None,
    sink: FakeVadSink | None = None,
    diagnostics: list | None = None,
) -> tuple[
    PeerCaptureSessionOwner,
    FakeAdmission,
    FakeTargetResolver,
    FakeProvider,
    list[object],
    FakeVadSink,
]:
    admission_port = admission or FakeAdmission()
    resolver_port = resolver or FakeTargetResolver()
    provider_port = provider or FakeProvider()
    created_sources = sources if sources is not None else []
    vad_sink = sink or FakeVadSink()

    def create_source(config, target):
        if source_factory is not None:
            return source_factory(config, target)
        source = FakeSource()
        created_sources.append(source)
        return source

    async def default_loop(**_kwargs) -> None:
        await asyncio.Event().wait()

    owner = PeerCaptureSessionOwner(
        admission=admission_port,
        target_resolver=resolver_port,
        provider=provider_port,
        clock=FakeClock(),
        provider_request_factory=lambda config, warmup: (config.provider_id, warmup),
        source_factory=create_source,
        vad_factory=vad_factory or (lambda _config: object()),
        run_audio_loop=run_audio_loop or default_loop,
        vad_sink=vad_sink,
        diagnostic_sink=diagnostics.append if diagnostics is not None else None,
    )
    return owner, admission_port, resolver_port, provider_port, created_sources, vad_sink

@pytest.mark.asyncio
async def test_peer_session_owner_exposes_segment_identity_from_actual_audio_loop() -> None:
    class FiniteSource:
        terminal_reason = None

        async def frames(self):
            yield AudioFrameF32(
                samples=np.ones((16,), dtype=np.float32),
                sample_rate_hz=16000,
            )

        async def close(self) -> None:
            return None

    source = FiniteSource()
    owner, *_ = make_owner(
        source_factory=lambda _config, _target: source,
        vad_factory=lambda _config: VadGating(
            SequenceVadEngine(probs=[0.9, 0.9]),
            sample_rate_hz=16000,
            chunk_samples=8,
            ring_buffer_ms=1,
            hangover_ms=640,
        ),
        run_audio_loop=run_audio_vad_loop,
    )

    started = await owner.apply_intent(make_config(), enabled=True)
    assert started.state is PeerCaptureSessionState.RUNNING

    await wait_until(
        lambda: owner.segment_ledger is not None
        and bool(owner.segment_ledger.snapshots)
        and owner.segment_ledger.snapshots[0].seal_reason == "source_eof"
    )

    ledger = owner.segment_ledger
    assert ledger is not None
    assert len(ledger.snapshots) == 1
    segment = ledger.snapshots[0]
    assert segment.identity.activation_generation == owner.snapshot.generation
    assert segment.identity.segment_order == 1
    assert segment.content_sample_count == 16
    assert segment.seal_reason == "source_eof"
    receipt = owner.record_segment_terminal(
        segment.identity.segment_id,
        outcome="final",
        text_authority="authoritative",
    )
    assert receipt.identity == segment.identity
    assert receipt.outcome == "final"
    assert ledger.drain_ready_terminal_receipts() == (receipt,)

    await owner.close()

@pytest.mark.asyncio
async def test_slow_peer_provider_dispatch_does_not_suspend_acoustic_progress() -> None:
    blocked = asyncio.Event()
    release = asyncio.Event()

    class FiniteSource:
        terminal_reason = None

        def __init__(self) -> None:
            self.yielded = 0

        async def frames(self):
            for value in (1.0, 1.0, 0.0):
                self.yielded += 1
                yield AudioFrameF32(
                    samples=np.full((8,), value, dtype=np.float32),
                    sample_rate_hz=16000,
                )

        async def close(self) -> None:
            return None

    class SlowSink:
        def __init__(self) -> None:
            self.events: list[object] = []

        async def handle_owned_vad_event(self, event: object) -> None:
            self.events.append(event)
            if len(self.events) == 1:
                blocked.set()
                await release.wait()

        async def handle_vad_event(self, event: object) -> None:
            raise AssertionError(f"unowned event reached sink: {event!r}")

    source = FiniteSource()
    sink = SlowSink()
    owner, *_ = make_owner(
        source_factory=lambda _config, _target: source,
        vad_factory=lambda _config: VadGating(
            SequenceVadEngine(probs=[0.9, 0.9, 0.0]),
            sample_rate_hz=16000,
            chunk_samples=8,
            ring_buffer_ms=1,
            hangover_ms=0,
        ),
        run_audio_loop=run_audio_vad_loop,
        sink=sink,
    )

    await owner.apply_intent(make_config(), enabled=True)
    await asyncio.wait_for(blocked.wait(), timeout=0.5)
    await wait_until(
        lambda: owner.segment_ledger is not None
        and bool(owner.segment_ledger.snapshots)
        and owner.segment_ledger.snapshots[0].seal_reason == "silence"
    )
    await wait_until(lambda: source.yielded == 3)
    ledger = owner.segment_ledger
    assert ledger is not None
    assert ledger.snapshots[0].seal_reason == "silence"

    release.set()
    await wait_until(lambda: len(sink.events) == 4)
    await owner.close()

@pytest.mark.asyncio
async def test_capture_waits_for_ingress_and_finite_completion_retires_owner_state() -> None:
    provider = FakeProvider()
    provider.start_gate = asyncio.Event()

    class FiniteSource:
        terminal_reason = None

        def __init__(self) -> None:
            self.yielded = 0

        async def frames(self):
            self.yielded += 1
            yield AudioFrameF32(
                samples=np.ones((16,), dtype=np.float32),
                sample_rate_hz=16000,
            )

        async def close(self) -> None:
            return None

    source = FiniteSource()
    owner, _admission, _resolver, _provider, _sources, sink = make_owner(
        provider=provider,
        source_factory=lambda _config, _target: source,
        vad_factory=lambda _config: VadGating(
            SequenceVadEngine(probs=[0.9, 0.9]),
            sample_rate_hz=16000,
            chunk_samples=8,
            ring_buffer_ms=1,
            hangover_ms=640,
        ),
        run_audio_loop=run_audio_vad_loop,
    )

    start_task = asyncio.create_task(owner.apply_intent(make_config(), enabled=True))
    await wait_until(lambda: provider.start_calls == 1)
    await asyncio.sleep(0.02)
    assert source.yielded == 0
    assert owner.segment_ledger is None

    provider.start_gate.set()
    await start_task
    await wait_until(lambda: owner.snapshot.state is PeerCaptureSessionState.STOPPED)

    ledger = owner.segment_ledgers[-1]
    assert ledger.snapshots[0].content_sample_count == 16
    assert ledger.snapshots[0].seal_reason == "source_eof"
    assert [event.__class__.__name__ for event in sink.events] == [
        "SpeechStart",
        "SpeechChunk",
        "SpeechEnd",
    ]
    assert owner.snapshot.effective_active is False
    assert owner.snapshot.has_loop_task is False
    await owner.close()


@pytest.mark.asyncio
async def test_blocked_provider_handoff_keeps_actual_segment_settings_until_commit() -> None:
    owner, _admission, _resolver, provider, _sources, _sink = make_owner()
    original = make_config(provider_id="soniox")
    requested = make_config(provider_id="deepgram")
    await owner.apply_intent(original, enabled=True)
    ledger = owner.segment_ledger
    assert ledger is not None
    first_id = uuid4()
    first_capture = AudioCaptureSpan(
        capture_epoch=1,
        callback_sequence=0,
        source_sample_rate_hz=16000,
        source_start_sample=0,
        source_end_sample=8,
        source_start_monotonic_s=0.0,
        source_end_monotonic_s=0.0005,
        normalized_sample_rate_hz=16000,
        normalized_start_sample=0,
        normalized_end_sample=8,
    )
    ledger.observe_vad_event(
        SpeechStart(
            first_id,
            pre_roll=np.empty((0,), dtype=np.float32),
            chunk=np.ones((8,), dtype=np.float32),
            chunk_capture=(first_capture,),
        ),
        now_monotonic_s=0.0,
    )

    provider.handoff_gate = asyncio.Event()
    transition = asyncio.create_task(owner.apply_intent(requested, enabled=True))
    await wait_until(lambda: len(provider.handoffs) == 1)

    assert owner.snapshot.provider_id == "soniox"
    assert ledger.snapshots[0].settings.provider_id == "soniox"
    ledger.observe_vad_event(
        SpeechEnd(first_id, trailing_silence_ms=0, reason="silence"),
        now_monotonic_s=0.1,
    )

    provider.handoff_gate.set()
    await transition
    assert owner.snapshot.provider_id == "deepgram"

    second_id = uuid4()
    second_capture = AudioCaptureSpan(
        capture_epoch=1,
        callback_sequence=1,
        source_sample_rate_hz=16000,
        source_start_sample=8,
        source_end_sample=16,
        source_start_monotonic_s=0.0005,
        source_end_monotonic_s=0.001,
        normalized_sample_rate_hz=16000,
        normalized_start_sample=8,
        normalized_end_sample=16,
    )
    owned = ledger.observe_vad_event(
        SpeechStart(
            second_id,
            pre_roll=np.empty((0,), dtype=np.float32),
            chunk=np.ones((8,), dtype=np.float32),
            chunk_capture=(second_capture,),
        ),
        now_monotonic_s=0.2,
    )
    assert owned.segment.settings.provider_id == "deepgram"
    assert ledger.snapshots[0].settings.provider_id == "soniox"
    await owner.close()


@pytest.mark.asyncio
async def test_peer_dispatch_accepts_twelve_seconds_while_downstream_is_stalled() -> None:
    blocked = asyncio.Event()
    release = asyncio.Event()

    class TwelveSecondSource:
        terminal_reason = None

        def __init__(self) -> None:
            self.yielded = 0

        async def frames(self):
            for _ in range(375):
                self.yielded += 1
                yield AudioFrameF32(
                    samples=np.ones((512,), dtype=np.float32),
                    sample_rate_hz=16000,
                )

        async def close(self) -> None:
            return None

    class SlowSink:
        def __init__(self) -> None:
            self.events = 0

        async def handle_owned_vad_event(self, _event: object) -> None:
            self.events += 1
            if self.events == 1:
                blocked.set()
                await release.wait()

        async def handle_vad_event(self, event: object) -> None:
            raise AssertionError(f"unowned event reached sink: {event!r}")

    source = TwelveSecondSource()
    sink = SlowSink()
    owner, *_ = make_owner(
        source_factory=lambda _config, _target: source,
        vad_factory=lambda _config: VadGating(
            SequenceVadEngine(probs=[0.9] * 375),
            sample_rate_hz=16000,
            chunk_samples=512,
            ring_buffer_ms=500,
            hangover_ms=640,
        ),
        run_audio_loop=run_audio_vad_loop,
        sink=sink,
    )

    await owner.apply_intent(make_config(), enabled=True)
    await asyncio.wait_for(blocked.wait(), timeout=0.5)
    await wait_until(lambda: source.yielded == 375)
    await wait_until(
        lambda: owner.segment_ledger is not None
        and bool(owner.segment_ledger.snapshots)
        and owner.segment_ledger.snapshots[0].seal_reason == "source_eof"
    )
    ledger = owner.segment_ledger
    assert ledger is not None
    assert ledger.snapshots[0].content_sample_count == 192000
    assert ledger.snapshots[0].seal_reason == "source_eof"
    assert owner.snapshot.state is PeerCaptureSessionState.RUNNING

    release.set()
    await wait_until(lambda: owner.snapshot.state is PeerCaptureSessionState.STOPPED)
    assert sink.events == 376
    await owner.close()

@pytest.mark.asyncio
async def test_peer_dispatch_reserves_terminal_capacity_for_eight_unsent_segments() -> None:
    blocked = asyncio.Event()
    release = asyncio.Event()

    class EightSegmentSource:
        terminal_reason = None

        def __init__(self) -> None:
            self.yielded = 0

        async def frames(self):
            for probability in [0.9, 0.0] * 8:
                self.yielded += 1
                yield AudioFrameF32(
                    samples=np.full((512,), probability, dtype=np.float32),
                    sample_rate_hz=16000,
                )

        async def close(self) -> None:
            return None

    class SlowSink:
        def __init__(self) -> None:
            self.events = 0

        async def handle_owned_vad_event(self, _event: object) -> None:
            self.events += 1
            if self.events == 1:
                blocked.set()
                await release.wait()

        async def handle_vad_event(self, event: object) -> None:
            raise AssertionError(f"unowned event reached sink: {event!r}")

    source = EightSegmentSource()
    sink = SlowSink()
    owner, *_ = make_owner(
        source_factory=lambda _config, _target: source,
        vad_factory=lambda _config: VadGating(
            SequenceVadEngine(probs=[0.9, 0.0] * 8),
            sample_rate_hz=16000,
            chunk_samples=512,
            ring_buffer_ms=500,
            hangover_ms=0,
        ),
        run_audio_loop=run_audio_vad_loop,
        sink=sink,
    )

    await owner.apply_intent(make_config(), enabled=True)
    await asyncio.wait_for(blocked.wait(), timeout=0.5)
    await wait_until(lambda: source.yielded == 16)
    await wait_until(
        lambda: owner.segment_ledger is not None
        and len(owner.segment_ledger.snapshots) == 8
        and all(
            segment.seal_reason == "silence"
            for segment in owner.segment_ledger.snapshots
        )
    )

    release.set()
    await wait_until(lambda: owner.snapshot.state is PeerCaptureSessionState.STOPPED)
    assert sink.events == 24
    await owner.close()


async def wait_until(predicate, *, timeout_s: float = 1.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while not predicate():
        if loop.time() >= deadline:
            raise AssertionError("timed out waiting for condition")
        await asyncio.sleep(0)


async def test_disabled_enabled_and_release_are_owned() -> None:
    owner, admission, resolver, provider, sources, _sink = make_owner()
    config = make_config()

    started = await owner.apply_intent(config, enabled=True)
    stopped = await owner.apply_intent(config, enabled=False, stop_mode="release")

    assert started.state is PeerCaptureSessionState.RUNNING
    assert started.effective_active is True
    assert stopped.state is PeerCaptureSessionState.STOPPED
    assert stopped.effective_active is False
    assert admission.calls == [config]
    assert resolver.calls == [config.capture_target]
    assert provider.requests == [("soniox", False)]
    assert provider.start_calls == 1
    assert provider.releases == [("abort", None)]
    assert sources[0].close_calls == 1


async def test_prepared_local_provider_is_reused_when_capture_starts() -> None:
    owner, admission, resolver, provider, sources, _sink = make_owner()
    config = make_config(provider_id="local_qwen")

    prepared = await owner.prepare_provider(config)

    assert prepared.state is PeerCaptureSessionState.STOPPED
    assert prepared.provider_status is PeerCaptureProviderStatus.READY
    assert prepared.effective_active is False
    assert admission.calls == []
    assert resolver.calls == []
    assert sources == []
    assert provider.requests == [("local_qwen", True)]

    started = await owner.apply_intent(config, enabled=True)

    assert started.state is PeerCaptureSessionState.RUNNING
    assert provider.requests == [("local_qwen", True)]
    assert provider.start_calls == 1
    assert admission.calls == [config]
    assert resolver.calls == [config.capture_target]
    assert len(sources) == 1

    await owner.close()
    assert provider.start_calls == 1
    assert provider.releases == [("abort", None)]
    assert sources[0].close_calls == 1


@pytest.mark.parametrize(
    ("status", "expected_state", "desired_active"),
    [
        (PeerCaptureAdmissionStatus.PENDING, PeerCaptureSessionState.ADMISSION_PENDING, True),
        (PeerCaptureAdmissionStatus.REJECTED, PeerCaptureSessionState.FAULTED, False),
    ],
)
async def test_admission_pending_and_rejected_do_not_open_resources(
    status: PeerCaptureAdmissionStatus,
    expected_state: PeerCaptureSessionState,
    desired_active: bool,
) -> None:
    admission = FakeAdmission()
    admission.result = PeerCaptureAdmission(status, reason="consent", retain_intent=False)
    owner, _admission, resolver, provider, sources, _sink = make_owner(admission=admission)

    snapshot = await owner.apply_intent(make_config(), enabled=True)

    assert snapshot.state is expected_state
    assert snapshot.desired_active is desired_active
    assert snapshot.admission_reason == "consent"
    assert resolver.calls == []
    assert provider.requests == []
    assert sources == []


async def test_process_target_unavailable_faults_and_retry_resolves_fresh_target() -> None:
    target = PeerCaptureTargetIntent(
        kind="process",
        process_kind="generic_executable",
        executable_identity=r"c:\apps\game\game.exe",
    )
    resolver = FakeTargetResolver()
    resolver.results = [
        PeerCaptureTargetResolution(
            PeerCaptureTargetStatus.UNAVAILABLE,
            reason="no_process",
        ),
        PeerCaptureTargetResolution(
            PeerCaptureTargetStatus.RESOLVED,
            target=PeerCaptureResolvedTarget(
                intent=target,
                capture_descriptor=(81, "instance-a"),
            ),
        ),
    ]
    owner, _admission, _resolver, provider, sources, _sink = make_owner(resolver=resolver)
    config = make_config(target=target)

    failed = await owner.apply_intent(config, enabled=True)
    retried = await owner.retry_process_capture()

    assert failed.state is PeerCaptureSessionState.FAULTED
    assert failed.failure_reason.value == "target_unavailable"
    assert failed.target_reason == "no_process"
    assert failed.retry_available is True
    assert retried is True
    assert resolver.calls == [target, target]
    assert owner.snapshot.resolved_target.capture_descriptor == (81, "instance-a")
    assert provider.start_calls == 1
    assert len(sources) == 1
    await owner.close()


async def test_source_and_vad_failures_are_contained_and_release_provider() -> None:
    source_owner, *_ = make_owner(
        source_factory=lambda _config, _target: (_ for _ in ()).throw(RuntimeError("source"))
    )
    source_snapshot = await source_owner.apply_intent(make_config(), enabled=True)

    vad_owner, *_ = make_owner(
        vad_factory=lambda _config: (_ for _ in ()).throw(RuntimeError("vad"))
    )
    vad_snapshot = await vad_owner.apply_intent(make_config(), enabled=True)

    assert source_snapshot.failure_reason.value == "source_open_failed"
    assert source_snapshot.has_source is False
    assert vad_snapshot.failure_reason.value == "vad_failed"
    assert vad_snapshot.has_vad is False


async def test_terminal_process_loss_faults_and_allows_retry() -> None:
    target = PeerCaptureTargetIntent(
        kind="process",
        process_kind="vrchat",
        executable_identity=r"c:\games\vrchat\vrchat.exe",
    )
    loop_release = asyncio.Event()

    async def run_loop(**_kwargs) -> None:
        await loop_release.wait()

    owner, _admission, _resolver, provider, sources, _sink = make_owner(run_audio_loop=run_loop)
    await owner.apply_intent(make_config(target=target), enabled=True)
    sources[0].terminal_reason = "target_exited"
    loop_release.set()
    await wait_until(lambda: owner.snapshot.state is PeerCaptureSessionState.FAULTED)

    assert owner.snapshot.failure_reason.value == "source_lost"
    assert owner.snapshot.retry_available is True
    assert provider.releases[-1] == ("abort", None)


async def test_language_facts_change_with_same_capture_handoffs_without_reopening_source() -> None:
    owner, _admission, _resolver, provider, sources, _sink = make_owner()
    manual = make_config()
    automatic = replace(
        manual,
        provider_id="local_qwen_gpu",
        provider_signature=("local_qwen_gpu", "auto", ("en", "ko")),
        runtime_signature=("local_qwen_gpu", "auto", ("en", "ko")),
        language=PeerCaptureLanguageFacts("auto", "en", ("en", "ko")),
        local_provider=True,
    )

    await owner.apply_intent(manual, enabled=True)
    snapshot = await owner.apply_intent(automatic, enabled=True)

    assert snapshot.language == automatic.language
    assert provider.handoffs == [("local_qwen_gpu", True)]
    assert len(sources) == 1
    await owner.close()


async def test_captured_final_facts_keep_identity_order_and_language_state() -> None:
    language = PeerCaptureLanguageFacts("auto", "en", ("en", "ko"))
    first = PeerCapturedFinalFacts(
        utterance_id=uuid4(),
        capture_sequence=1,
        language=language,
        language_state=PeerCaptureFinalLanguageState.MIXED,
        detected_languages=("en", "ko"),
    )
    second = replace(
        first,
        utterance_id=uuid4(),
        capture_sequence=2,
        language_state=PeerCaptureFinalLanguageState.MISSING,
        detected_languages=(),
    )

    assert first.capture_sequence < second.capture_sequence
    assert first.language.expected_languages == ("en", "ko")
    assert second.language_state is PeerCaptureFinalLanguageState.MISSING


async def test_provider_pending_failure_and_ingress_failure_publish_truthful_state() -> None:
    pending_provider = FakeProvider()
    pending_provider.replace_result = PeerCaptureProviderMutation(
        PeerCaptureProviderMutationStatus.PENDING,
        reason="loading",
    )
    pending_owner, *_ = make_owner(provider=pending_provider)
    pending = await pending_owner.apply_intent(make_config(), enabled=True)

    failed_provider = FakeProvider()
    failed_provider.replace_result = PeerCaptureProviderMutation(
        PeerCaptureProviderMutationStatus.FAILED,
        reason="offline",
    )
    failed_owner, *_ = make_owner(provider=failed_provider)
    failed = await failed_owner.apply_intent(make_config(), enabled=True)

    ingress_provider = FakeProvider()
    ingress_provider.start_error = RuntimeError("ingress")
    ingress_owner, *_ = make_owner(provider=ingress_provider)
    ingress = await ingress_owner.apply_intent(make_config(), enabled=True)

    assert pending.state is PeerCaptureSessionState.PROVIDER_PENDING
    assert pending.provider_status is PeerCaptureProviderStatus.PENDING
    assert failed.state is PeerCaptureSessionState.FAULTED
    assert failed.failure_reason.value == "provider_failed"
    assert ingress.state is PeerCaptureSessionState.FAULTED
    assert ingress.effective_active is False


async def test_superseded_target_resolution_cannot_open_or_publish_old_generation() -> None:
    resolver = FakeTargetResolver()
    resolver.gate = asyncio.Event()
    owner, _admission, _resolver, provider, sources, _sink = make_owner(resolver=resolver)
    config = make_config()

    first = asyncio.create_task(owner.apply_intent(config, enabled=True))
    await wait_until(lambda: len(resolver.calls) == 1)
    stopped = asyncio.create_task(owner.apply_intent(config, enabled=False))
    resolver.gate.set()
    await asyncio.gather(first, stopped)

    assert owner.snapshot.state is PeerCaptureSessionState.STOPPED
    assert sources == []
    assert provider.start_calls == 0


async def test_stale_vad_and_provider_callbacks_cannot_fault_replacement_generation() -> None:
    captured_sinks: list[object] = []

    async def run_loop(**kwargs) -> None:
        captured_sinks.append(kwargs["sink"])
        await asyncio.Event().wait()

    owner, _admission, _resolver, provider, _sources, sink = make_owner(run_audio_loop=run_loop)
    first = make_config(provider_id="soniox")
    second = make_config(provider_id="deepgram", capture_signature=("new-device",))
    await owner.apply_intent(first, enabled=True)
    await wait_until(lambda: len(captured_sinks) == 1)
    old_terminal = provider.terminal_handlers[-1]
    await owner.apply_intent(second, enabled=True)
    await wait_until(lambda: len(captured_sinks) == 2)

    await captured_sinks[0].handle_vad_event("stale")
    await old_terminal(RuntimeError("late"))
    await captured_sinks[1].handle_vad_event("current")

    assert sink.events == ["current"]
    assert owner.snapshot.state is PeerCaptureSessionState.RUNNING
    await owner.close()


async def test_current_initial_provider_terminal_failure_faults_and_releases() -> None:
    owner, _admission, _resolver, provider, sources, _sink = make_owner()
    await owner.apply_intent(make_config(), enabled=True)

    await provider.terminal_handlers[-1](RuntimeError("terminal"))

    assert owner.snapshot.state is PeerCaptureSessionState.FAULTED
    assert owner.snapshot.failure_reason.value == "provider_failed"
    assert owner.snapshot.has_source is False
    assert owner.snapshot.has_vad is False
    assert owner.snapshot.has_loop_task is False
    assert sources[0].close_calls == 1
    assert provider.releases[-1][0] == "abort"


async def test_terminal_failure_before_initial_attachment_commit_faults_without_leak() -> None:
    provider = FakeProvider()
    provider.replace_terminal_error = RuntimeError("terminal during replace")
    owner, _admission, _resolver, _provider, sources, _sink = make_owner(provider=provider)

    snapshot = await owner.apply_intent(make_config(), enabled=True)

    assert snapshot.state is PeerCaptureSessionState.FAULTED
    assert snapshot.failure_reason.value == "provider_failed"
    assert snapshot.has_source is False
    assert sources == []
    assert provider.releases[-1][0] == "abort"


async def test_retained_reconfiguration_keeps_provider_callback_current() -> None:
    owner, _admission, _resolver, provider, _sources, _sink = make_owner()
    first = make_config()
    await owner.apply_intent(first, enabled=True)
    current_terminal = provider.terminal_handlers[-1]
    reconfigured = replace(
        first,
        runtime_signature=("retained",),
        language=replace(first.language, source_language="en"),
    )

    await owner.apply_intent(reconfigured, enabled=True)
    await current_terminal(RuntimeError("terminal after reconfigure"))

    assert provider.reconfigurations
    assert owner.snapshot.state is PeerCaptureSessionState.FAULTED
    assert owner.snapshot.failure_reason.value == "provider_failed"


async def test_retained_local_provider_callback_remains_current_after_restart() -> None:
    owner, _admission, _resolver, provider, _sources, _sink = make_owner()
    config = make_config(provider_id="local_qwen")
    await owner.apply_intent(config, enabled=True)
    retained_terminal = provider.terminal_handlers[-1]

    await owner.apply_intent(config, enabled=False, stop_mode="retain")
    await owner.apply_intent(config, enabled=True)
    await retained_terminal(RuntimeError("retained terminal"))

    assert len(provider.terminal_handlers) == 1
    assert owner.snapshot.state is PeerCaptureSessionState.FAULTED
    assert owner.snapshot.failure_reason.value == "provider_failed"


async def test_failed_handoff_retires_candidate_callback_and_keeps_current_callback() -> None:
    owner, _admission, _resolver, provider, _sources, _sink = make_owner()
    first = make_config(provider_id="soniox")
    await owner.apply_intent(first, enabled=True)
    current_terminal = provider.terminal_handlers[-1]
    provider.handoff_result = PeerCaptureProviderMutation(
        PeerCaptureProviderMutationStatus.FAILED,
        reason="offline",
    )

    await owner.apply_intent(make_config(provider_id="deepgram"), enabled=True)
    candidate_terminal = provider.terminal_handlers[-1]
    await candidate_terminal(RuntimeError("retired candidate"))

    assert owner.snapshot.state is PeerCaptureSessionState.RUNNING
    await current_terminal(RuntimeError("current terminal"))
    assert owner.snapshot.state is PeerCaptureSessionState.FAULTED
    assert owner.snapshot.failure_reason.value == "provider_failed"


async def test_successful_handoff_retires_old_callback_and_faults_from_new_callback() -> None:
    owner, _admission, _resolver, provider, _sources, _sink = make_owner()
    await owner.apply_intent(make_config(provider_id="soniox"), enabled=True)
    old_terminal = provider.terminal_handlers[-1]

    await owner.apply_intent(make_config(provider_id="deepgram"), enabled=True)
    new_terminal = provider.terminal_handlers[-1]
    await old_terminal(RuntimeError("retired old"))

    assert owner.snapshot.state is PeerCaptureSessionState.RUNNING
    await new_terminal(RuntimeError("current new"))
    assert owner.snapshot.state is PeerCaptureSessionState.FAULTED
    assert owner.snapshot.failure_reason.value == "provider_failed"


async def test_recovery_terminal_failure_before_adoption_faults_recovered_attachment() -> None:
    owner, _admission, _resolver, provider, _sources, _sink = make_owner()
    config = make_config(provider_id="local_qwen_gpu")
    await owner.apply_intent(config, enabled=True)
    await owner.suspend_provider_consumer()
    recovered_terminal = owner.prepare_provider_recovery(config)

    await recovered_terminal(RuntimeError("recovered terminal"))
    await owner.adopt_recovered_provider(
        config,
        on_terminal_failure=recovered_terminal,
    )

    assert owner.snapshot.state is PeerCaptureSessionState.FAULTED
    assert owner.snapshot.failure_reason.value == "provider_failed"
    assert provider.releases[-1][0] == "abort"


async def test_close_cancels_pending_start_and_is_idempotent() -> None:
    admission = FakeAdmission()
    admission.gate = asyncio.Event()
    owner, *_ = make_owner(admission=admission)
    start = asyncio.create_task(owner.apply_intent(make_config(), enabled=True))
    await wait_until(lambda: len(admission.calls) == 1)
    close = asyncio.create_task(owner.close())
    admission.gate.set()
    await asyncio.gather(start, close)
    await owner.close()

    assert owner.snapshot.closed is True
    assert owner.snapshot.state is PeerCaptureSessionState.STOPPED
    assert owner.snapshot.has_source is False
    assert owner.snapshot.has_loop_task is False


async def test_close_retries_cleanup_debt_without_leaking_source() -> None:
    source = FailingCloseSource()
    owner, _admission, _resolver, provider, _sources, _sink = make_owner(
        source_factory=lambda _config, _target: source
    )
    await owner.apply_intent(make_config(), enabled=True)

    with pytest.raises(RuntimeError, match="close failed"):
        await owner.apply_intent(make_config(), enabled=False)

    assert owner.snapshot.cleanup_debt == 1
    await owner.close()
    assert source.close_calls == 2
    assert owner.snapshot.cleanup_debt == 0
    assert provider.releases[-1] == ("abort", None)
