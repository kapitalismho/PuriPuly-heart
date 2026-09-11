from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass, replace
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest

from puripuly_heart.app.services.application_runtime_shutdown import (
    compose_application_runtime_shutdown_callbacks,
)
from puripuly_heart.app.services.application_shutdown import ApplicationShutdownCoordinator
from puripuly_heart.app.wiring.wiring_stt_factory import (
    build_peer_capture_session_config_from_vnext,
)
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext
from puripuly_heart.core.audio import smart_turn as smart_turn_module
from puripuly_heart.core.audio.format import (
    AudioCaptureDiscontinuity,
    AudioCaptureSpan,
    AudioFrameF32,
)
from puripuly_heart.core.audio.listen_delivery import ListenDeliveryController
from puripuly_heart.core.audio.psem_receiver import ProspectiveSpeakerHypothesis
from puripuly_heart.core.audio.smart_turn import (
    SMART_TURN_COMPLETE_THRESHOLD,
    SmartTurnInferenceOwner,
)
from puripuly_heart.core.clock import FakeClock, SystemClock
from puripuly_heart.core.orchestrator.translation_channel_callbacks import (
    TranslationChannelOwnerCallbacks,
)
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
from puripuly_heart.core.runtime.stt_session_projection import SttSessionStateProjection
from puripuly_heart.core.stt.backend import (
    STTProviderTurnIdentity,
    STTProviderTurnTerminal,
    STTSessionProjection,
)
from puripuly_heart.core.stt.scoped_engine import ScopedRecognitionEngine, STTRecognitionWatchdogs
from puripuly_heart.core.vad.gating import (
    SpeechEnd,
    SpeechStart,
    VadGating,
    create_peer_vad_gating,
)
from puripuly_heart.providers.stt.custom import CustomSTTBackend
from tests.helpers.fakes import RecordingOscQueue
from tests.helpers.translation_owners import compose_translation_test_harness
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
        self.release_gate: asyncio.Event | None = None
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
        if self.release_gate is not None:
            await self.release_gate.wait()
        if mode == "abort":
            self.provider_id = None


class FakeVadSink:
    def __init__(self) -> None:
        self.events: list[object] = []

    async def handle_owned_vad_event(self, event: object) -> None:
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
    signature = capture_signature or (resolved_target, 16000)
    return PeerCaptureSessionConfig(
        provider_id=provider_id,
        provider_signature=(provider_id, language_facts),
        runtime_signature=(provider_id, signature, language_facts, 0.6, 900, 500),
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
    smart_turn_owner=None,
    diagnostics: list | None = None,
    clock=None,
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
        clock=clock or FakeClock(),
        provider_request_factory=lambda config, warmup: (config.provider_id, warmup),
        source_factory=create_source,
        vad_factory=vad_factory or (lambda _config: object()),
        run_audio_loop=run_audio_loop or default_loop,
        vad_sink=vad_sink,
        smart_turn_owner=smart_turn_owner,
        diagnostic_sink=diagnostics.append if diagnostics is not None else None,
    )
    return owner, admission_port, resolver_port, provider_port, created_sources, vad_sink


def test_peer_capture_owner_rejects_legacy_unowned_vad_sink() -> None:
    class LegacySink:
        async def handle_vad_event(self, _event: object) -> None:
            return None

    with pytest.raises(TypeError, match="requires owned VAD ingress"):
        make_owner(sink=LegacySink())


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
    provider = FakeProvider()
    provider.release_gate = asyncio.Event()
    owner, *_ = make_owner(
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

    started = await owner.apply_intent(make_config(), enabled=True)
    assert started.state is PeerCaptureSessionState.RUNNING

    ledger = owner.segment_ledgers[-1]
    await wait_until(lambda: bool(provider.releases))
    assert provider.releases[-1][0] == "drain"
    assert owner.snapshot.state is PeerCaptureSessionState.STOPPING
    assert len(ledger.snapshots) == 1
    segment = ledger.snapshots[0]
    assert segment.identity.activation_generation == started.generation
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
    assert ledger.snapshots == ()
    assert ledger.terminal_receipts == (receipt,)
    provider.release_gate.set()
    await wait_until(lambda: owner.snapshot.state is PeerCaptureSessionState.STOPPED)

    await owner.close()


@pytest.mark.asyncio
async def test_live_hangover_change_applies_to_next_segment_without_capture_restart() -> None:
    class ControlledSource:
        terminal_reason = None

        def __init__(self) -> None:
            self.silence_gate = asyncio.Event()
            self.successor_gate = asyncio.Event()
            self.stop_gate = asyncio.Event()

        async def frames(self):
            yield AudioFrameF32(
                samples=np.ones((1536,), dtype=np.float32),
                sample_rate_hz=16000,
            )
            await self.silence_gate.wait()
            for _ in range(29):
                yield AudioFrameF32(
                    samples=np.zeros((512,), dtype=np.float32),
                    sample_rate_hz=16000,
                )
            await self.successor_gate.wait()
            yield AudioFrameF32(
                samples=np.ones((1536,), dtype=np.float32),
                sample_rate_hz=16000,
            )
            await self.stop_gate.wait()

        async def close(self) -> None:
            self.silence_gate.set()
            self.successor_gate.set()
            self.stop_gate.set()

    class ReadySmartTurnOwner:
        def __init__(self) -> None:
            self.snapshot = SimpleNamespace(availability="ready")
            self.prepare_calls = 0
            self.submit_calls = 0

        def request_prepare(self) -> None:
            self.prepare_calls += 1

        def submit(self, _identity, _audio, _completion):
            self.submit_calls += 1
            return "unavailable"

        def record_late(self) -> None:
            raise AssertionError("no result completes")

        async def close(self) -> None:
            self.snapshot.availability = "closed"

    smart_turn = ReadySmartTurnOwner()
    source = ControlledSource()
    probabilities = [0.9] * 3 + [0.0] * 29 + [0.9] * 3
    owner, *_ = make_owner(
        source_factory=lambda _config, _target: source,
        vad_factory=lambda current: create_peer_vad_gating(
            SequenceVadEngine(probs=probabilities),
            sample_rate_hz=current.target_sample_rate_hz,
            ring_buffer_ms=current.vad_pre_roll_ms,
            speech_threshold=current.vad_speech_threshold,
            hangover_ms=current.vad_hangover_ms,
        ),
        run_audio_loop=run_audio_vad_loop,
        smart_turn_owner=smart_turn,
    )
    initial_base = make_config()
    initial = replace(
        initial_base,
        smart_turn_enabled=True,
        runtime_signature=(*initial_base.runtime_signature, "smart-on"),
    )
    changed_language = PeerCaptureLanguageFacts(
        source_mode="manual",
        source_language="ja",
    )
    changed = replace(
        initial,
        language=changed_language,
        smart_turn_enabled=False,
        vad_hangover_ms=1200,
        provider_signature=("soniox", changed_language),
        runtime_signature=(*initial.runtime_signature, "smart-off-ja-hangover-1200"),
    )
    started = await owner.apply_intent(initial, enabled=True)
    ledger = owner.segment_ledger
    vad = owner.vad
    assert ledger is not None
    await wait_until(lambda: len(ledger.snapshots) == 1)
    first_id = ledger.snapshots[0].identity.segment_id

    updated = await owner.apply_intent(changed, enabled=True)
    assert updated.generation == started.generation
    assert owner.segment_ledger is ledger
    assert ledger.snapshots[0].settings.delivery_threshold == SMART_TURN_COMPLETE_THRESHOLD
    assert ledger.snapshots[0].settings.vad_hangover_ms == 900

    pending = owner.snapshot
    assert pending.requested_delivery_profile == "off"
    assert pending.effective_delivery_profile == "on"
    assert pending.requested_vad_hangover_ms == 1200
    assert pending.effective_vad_hangover_ms == 900
    assert pending.requested_language == changed_language
    assert pending.language == initial.delivery_language
    assert pending.effective_language == initial.delivery_language
    assert pending.smart_turn_availability == "ready"
    source.silence_gate.set()
    await wait_until(lambda: ledger.snapshots[0].state == "sealed")
    assert ledger.snapshots[0].identity.segment_id == first_id
    assert ledger.snapshots[0].seal_reason == "delivery_pause"
    assert getattr(vad, "hangover_chunks") == 38

    source.successor_gate.set()
    await wait_until(lambda: len(ledger.snapshots) == 2)
    successor = ledger.snapshots[1]
    assert successor.settings.vad_hangover_ms == 1200
    assert successor.identity.activation_generation == started.generation

    assert successor.settings.source_language == "ja"
    assert successor.settings.delivery_threshold is None
    settled = owner.snapshot
    assert settled.requested_delivery_profile == "off"
    assert settled.effective_delivery_profile == "off"
    assert settled.effective_vad_hangover_ms == 1200
    assert settled.language == changed_language
    assert settled.effective_language == changed_language
    assert settled.smart_turn_availability == "disabled"
    assert smart_turn.prepare_calls == 1
    assert smart_turn.submit_calls == 1
    await owner.close()
    assert [receipt.outcome for receipt in ledger.terminal_receipts] == [
        "cancelled",
        "cancelled",
    ]


@pytest.mark.asyncio
async def test_requested_auto_language_stays_unsupported_when_local_auto_resolves_english() -> None:
    class FiniteSource:
        terminal_reason = None

        def __init__(self) -> None:
            self.yielded = 0

        async def frames(self):
            for value in (*([1.0] * 3), *([0.0] * 16)):
                self.yielded += 1
                yield AudioFrameF32(
                    samples=np.full((512,), value, dtype=np.float32),
                    sample_rate_hz=16000,
                )

        async def close(self) -> None:
            return None

    class NoInferenceOwner:
        def __init__(self) -> None:
            self.snapshot = SimpleNamespace(availability="ready")
            self.prepare_calls = 0
            self.submit_calls = 0

        def request_prepare(self) -> None:
            self.prepare_calls += 1

        def submit(self, _identity, _audio, _completion):
            self.submit_calls += 1
            return "started"

        def record_late(self) -> None:
            raise AssertionError("auto language must not infer")

        async def close(self) -> None:
            return None

    settings = AppSettingsVNext()
    settings = replace(
        settings,
        intent=replace(
            settings.intent,
            desktop_audio=replace(
                settings.intent.desktop_audio,
                smart_turn_enabled=True,
                vad_hangover_ms=480,
            ),
            languages=replace(
                settings.intent.languages,
                peer_source_mode="auto",
                peer_source_language="en",
            ),
        ),
    )
    config = build_peer_capture_session_config_from_vnext(settings)
    assert config.provider_id == "local_cpu_auto"
    assert config.language.source_mode == "manual"
    assert config.language.source_language == "en"
    assert config.delivery_language.source_mode == "auto"
    assert config.delivery_language.source_language == "en"

    source = FiniteSource()
    smart_turn = NoInferenceOwner()
    owner, *_ = make_owner(
        source_factory=lambda _config, _target: source,
        vad_factory=lambda current: create_peer_vad_gating(
            SequenceVadEngine(probs=[0.9] * 3 + [0.0] * 16),
            sample_rate_hz=current.target_sample_rate_hz,
            ring_buffer_ms=current.vad_pre_roll_ms,
            speech_threshold=current.vad_speech_threshold,
            hangover_ms=current.vad_hangover_ms,
        ),
        run_audio_loop=run_audio_vad_loop,
        smart_turn_owner=smart_turn,
    )
    await owner.apply_intent(config, enabled=True)
    await wait_until(lambda: source.yielded == 19)
    ledger = owner.segment_ledger
    assert ledger is not None
    await wait_until(
        lambda: bool(ledger.snapshots) and ledger.snapshots[0].seal_reason == "delivery_pause"
    )
    segment = ledger.snapshots[0]
    assert segment.settings.delivery_profile_effective == "unsupported_auto"
    assert segment.settings.delivery_threshold is None
    assert segment.settings.vad_hangover_ms == 480
    assert segment.content_sample_count == 18 * 512
    assert smart_turn.prepare_calls == 0
    assert smart_turn.submit_calls == 0
    await owner.close()


@pytest.mark.asyncio
async def test_idle_cached_smart_turn_stays_unloaded_until_speech_starts_prepare(
    tmp_path,
) -> None:
    speech_gate = asyncio.Event()
    stop_gate = asyncio.Event()
    factory_entered = threading.Event()
    factory_release = threading.Event()

    class GatedSource:
        terminal_reason = None

        async def frames(self):
            await speech_gate.wait()
            for _ in range(3):
                yield AudioFrameF32(
                    samples=np.ones((512,), dtype=np.float32),
                    sample_rate_hz=16000,
                )
            await stop_gate.wait()

        async def close(self) -> None:
            speech_gate.set()
            stop_gate.set()

    class Inference:
        async def predict(self, _audio, *, sample_rate_hz: int) -> float:
            del sample_rate_hz
            return 0.9

        def close(self) -> None:
            return None

    def factory(_path):
        factory_entered.set()
        assert factory_release.wait(5.0)
        return Inference()

    model_path = tmp_path / "smart-turn-v3.2-cpu.onnx"
    model_path.write_bytes(b"fixture")
    smart_turn = SmartTurnInferenceOwner(
        model_path=model_path,
        inference_factory=factory,
    )
    source = GatedSource()
    owner, *_ = make_owner(
        source_factory=lambda _config, _target: source,
        vad_factory=lambda current: create_peer_vad_gating(
            SequenceVadEngine(probs=[0.9] * 3),
            sample_rate_hz=current.target_sample_rate_hz,
            ring_buffer_ms=current.vad_pre_roll_ms,
            speech_threshold=current.vad_speech_threshold,
            hangover_ms=current.vad_hangover_ms,
        ),
        run_audio_loop=run_audio_vad_loop,
        smart_turn_owner=smart_turn,
    )
    config = replace(make_config(), smart_turn_enabled=True)
    await owner.apply_intent(config, enabled=True)
    assert owner.snapshot.smart_turn_availability == "unloaded"
    assert not factory_entered.is_set()

    speech_gate.set()
    assert await asyncio.to_thread(factory_entered.wait, 1.0)
    assert owner.snapshot.smart_turn_availability == "loading"
    factory_release.set()
    await wait_until(lambda: owner.snapshot.smart_turn_availability == "ready")

    stop_gate.set()
    await owner.close()


@pytest.mark.asyncio
async def test_application_shutdown_deadline_preserves_blocked_peer_native_cleanup(
    tmp_path,
) -> None:
    entered = threading.Event()
    release = threading.Event()
    closed = threading.Event()

    class BlockingInference:
        async def predict(self, _audio, *, sample_rate_hz: int) -> float:
            del sample_rate_hz
            return 0.5

        def close(self) -> None:
            closed.set()

    def blocking_factory(_model_path):
        entered.set()
        assert release.wait(timeout=5.0)
        return BlockingInference()

    model_path = tmp_path / "smart-turn-v3.2-cpu.onnx"
    model_path.write_bytes(b"verified-model")
    smart_turn = SmartTurnInferenceOwner(
        model_path=model_path,
        inference_factory=blocking_factory,
    )
    smart_turn.request_prepare()
    assert await asyncio.to_thread(entered.wait, 1.0)

    owner, *_ = make_owner(smart_turn_owner=smart_turn)
    await owner.apply_intent(make_config(), enabled=True)

    class ShutdownRuntime:
        close_peer_capture_owner = owner.close

        def __getattr__(self, _name):
            async def no_op() -> None:
                return None

            return no_op

    production_callback = next(
        callback
        for callback in compose_application_runtime_shutdown_callbacks(ShutdownRuntime())
        if callback.owner_name == "PeerCaptureSessionOwner"
    )
    assert production_callback.timeout_seconds == 30.0
    coordinator = ApplicationShutdownCoordinator(
        (replace(production_callback, timeout_seconds=0.05),),
        task_settle_timeout_seconds=0.01,
    )

    started_at = time.perf_counter()
    with pytest.raises(TimeoutError):
        await coordinator.shutdown()
    elapsed = time.perf_counter() - started_at
    assert elapsed < 0.5
    assert coordinator.snapshot.failures[0].timed_out is True
    assert owner.snapshot.closed is True
    assert owner.snapshot.desired_active is False
    assert owner._close_task is not None
    assert not owner._close_task.done()
    assert not closed.is_set()

    release.set()
    await owner.close()
    assert closed.is_set()
    assert owner._close_task.done()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("gap_kind", "second_epoch"),
    [("known_loss", 1), ("unknown_loss", 2)],
)
async def test_source_gap_resets_smart_turn_context_and_rejects_old_result(
    gap_kind: str,
    second_epoch: int,
) -> None:
    gap_gate = asyncio.Event()
    resume_gate = asyncio.Event()
    stop_gate = asyncio.Event()

    class GapSource:
        terminal_reason = None

        def __init__(self) -> None:
            self.monotonic_sample = 0

        def frame(
            self,
            *,
            value: float,
            epoch: int,
            source_start: int,
            sequence: int,
            discontinuity=None,
        ) -> AudioFrameF32:
            monotonic_start = self.monotonic_sample / 16000
            self.monotonic_sample += 512
            return AudioFrameF32(
                samples=np.full((512,), value, dtype=np.float32),
                sample_rate_hz=16000,
                capture=AudioCaptureSpan(
                    capture_epoch=epoch,
                    callback_sequence=sequence,
                    source_sample_rate_hz=16000,
                    source_start_sample=source_start,
                    source_end_sample=source_start + 512,
                    source_start_monotonic_s=monotonic_start,
                    source_end_monotonic_s=self.monotonic_sample / 16000,
                    discontinuity_before=discontinuity,
                ),
            )

        async def frames(self):
            source_start = 0
            for sequence, value in enumerate([0.25] * 3 + [0.0] * 7):
                yield self.frame(
                    value=value,
                    epoch=1,
                    source_start=source_start,
                    sequence=sequence,
                )
                source_start += 512
            await gap_gate.wait()
            gap = AudioCaptureDiscontinuity(
                kind=gap_kind,
                observed_at_monotonic_s=self.monotonic_sample / 16000,
                lost_source_samples=512 if gap_kind == "known_loss" else None,
            )
            source_start = source_start + 512 if gap_kind == "known_loss" else 0
            for offset, value in enumerate([0.5] * 3 + [0.0] * 7):
                yield self.frame(
                    value=value,
                    epoch=second_epoch,
                    source_start=source_start,
                    sequence=10 + offset if gap_kind == "known_loss" else offset,
                    discontinuity=gap if offset == 0 else None,
                )
                source_start += 512
            await resume_gate.wait()
            for offset, value in enumerate([0.75] + [0.0] * 7, start=10):
                yield self.frame(
                    value=value,
                    epoch=second_epoch,
                    source_start=source_start,
                    sequence=10 + offset if gap_kind == "known_loss" else offset,
                )
                source_start += 512
            await stop_gate.wait()

        async def close(self) -> None:
            gap_gate.set()
            stop_gate.set()
            resume_gate.set()

    class RecordingSmartTurn:
        def __init__(self) -> None:
            self.snapshot = SimpleNamespace(availability="ready")
            self.requests = []
            self.audio = []
            self.callbacks = []
            self.attempt_count = 0
            self.busy = False

        def request_prepare(self) -> None:
            return None

        def submit(self, identity, audio, callback):
            self.attempt_count += 1
            if self.busy:
                return "busy"
            self.busy = True
            self.requests.append(identity)
            self.audio.append(audio.copy())
            self.callbacks.append(callback)
            return "started"

        def record_late(self) -> None:
            raise AssertionError("the stamp is timely")

        async def close(self) -> None:
            return None

    source = GapSource()
    smart_turn = RecordingSmartTurn()
    owner, *_ = make_owner(
        source_factory=lambda _config, _target: source,
        vad_factory=lambda current: create_peer_vad_gating(
            SequenceVadEngine(probs=([0.9] * 3 + [0.0] * 7) * 2 + [0.9] + [0.0] * 7),
            sample_rate_hz=current.target_sample_rate_hz,
            ring_buffer_ms=current.vad_pre_roll_ms,
            speech_threshold=current.vad_speech_threshold,
            hangover_ms=current.vad_hangover_ms,
        ),
        run_audio_loop=run_audio_vad_loop,
        smart_turn_owner=smart_turn,
    )
    config = replace(make_config(), smart_turn_enabled=True)
    await owner.apply_intent(config, enabled=True)
    await wait_until(lambda: len(smart_turn.requests) == 1)
    old_request = smart_turn.requests[0]

    gap_gate.set()
    await wait_until(lambda: smart_turn.attempt_count == 2)
    assert len(smart_turn.requests) == 1
    ledger = owner.segment_ledger
    assert ledger is not None
    current = ledger.snapshots[-1]
    assert current.identity.capture_epoch == second_epoch
    assert current.state == "open"

    smart_turn.busy = False
    await smart_turn.callbacks[0](
        smart_turn_module.SmartTurnCompletion(
            identity=old_request,
            score=0.99,
            completed_at_monotonic_s=old_request.complete_deadline_monotonic_s - 0.1,
            duration_s=0.01,
            outcome="complete",
        )
    )
    assert ledger.snapshots[-1].identity.segment_id == current.identity.segment_id
    assert ledger.snapshots[-1].state == "open"
    assert ledger.terminal_receipts[0].outcome == "failed"

    resume_gate.set()
    await wait_until(lambda: len(smart_turn.requests) == 2)
    assert smart_turn.audio[1].size == (3 + 7 + 1 + 7) * 512
    np.testing.assert_array_equal(smart_turn.audio[1][: 3 * 512], 0.5)
    np.testing.assert_array_equal(smart_turn.audio[1][3 * 512 : 10 * 512], 0.0)
    np.testing.assert_array_equal(smart_turn.audio[1][10 * 512 : 11 * 512], 0.75)
    np.testing.assert_array_equal(smart_turn.audio[1][11 * 512 :], 0.0)

    stop_gate.set()
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
        lambda: (
            owner.segment_ledger is not None
            and bool(owner.segment_ledger.snapshots)
            and owner.segment_ledger.snapshots[0].seal_reason == "silence"
        )
    )
    await wait_until(lambda: source.yielded == 3)
    ledger = owner.segment_ledger
    assert ledger is not None
    assert ledger.snapshots[0].seal_reason == "silence"

    release.set()
    await wait_until(lambda: len(sink.events) == 4)
    await owner.close()


@pytest.mark.asyncio
async def test_provider_stall_does_not_suspend_pending_smart_turn_or_800ms_fallback() -> None:
    blocked = asyncio.Event()
    release = asyncio.Event()

    class FiniteSource:
        terminal_reason = None

        def __init__(self) -> None:
            self.yielded = 0

        async def frames(self):
            for value in (1.0, 1.0, *([0.0] * 85)):
                self.yielded += 1
                yield AudioFrameF32(
                    samples=np.full((160,), value, dtype=np.float32),
                    sample_rate_hz=16000,
                )

        async def close(self) -> None:
            return None

    class SlowSink:
        async def handle_owned_vad_event(self, _event: object) -> None:
            blocked.set()
            await release.wait()

    class PendingSmartTurnOwner:
        def __init__(self) -> None:
            self.snapshot = type("Snapshot", (), {"availability": "ready"})()
            self.submitted = asyncio.Event()
            self.requests = []

        def request_prepare(self) -> None:
            return None

        def submit(self, identity, audio, completion):
            self.requests.append((identity, audio, completion))
            self.submitted.set()
            return "started"

        def record_late(self) -> None:
            raise AssertionError("inference remains incomplete")

        async def close(self) -> None:
            return None

    source = FiniteSource()
    smart_turn = PendingSmartTurnOwner()
    owner, *_ = make_owner(
        source_factory=lambda _config, _target: source,
        vad_factory=lambda _config: VadGating(
            SequenceVadEngine(probs=[0.9, 0.9, *([0.0] * 85)]),
            sample_rate_hz=16000,
            chunk_samples=160,
            ring_buffer_ms=10,
            hangover_ms=1000,
            external_delivery_boundaries=True,
        ),
        run_audio_loop=run_audio_vad_loop,
        sink=SlowSink(),
        smart_turn_owner=smart_turn,
    )

    await owner.apply_intent(
        replace(make_config(), smart_turn_enabled=True),
        enabled=True,
    )
    await asyncio.wait_for(blocked.wait(), timeout=0.5)
    await asyncio.wait_for(smart_turn.submitted.wait(), timeout=0.5)
    await wait_until(lambda: source.yielded == 87)
    ledger = owner.segment_ledger
    assert ledger is not None
    await wait_until(
        lambda: bool(ledger.snapshots) and ledger.snapshots[0].seal_reason == "delivery_pause"
    )
    assert len(smart_turn.requests) == 1

    release.set()
    await owner.close()


@pytest.mark.asyncio
async def test_custom_offline_http_wait_does_not_block_later_speech_capture() -> None:
    request_started = asyncio.Event()
    release_http = asyncio.Event()

    class Response:
        status_code = 200
        text = '{"text":"recognized"}'

        def json(self):
            return {"text": "recognized"}

    class Client:
        async def post(self, *_args, **_kwargs):
            request_started.set()
            await release_http.wait()
            return Response()

        async def aclose(self) -> None:
            return None

    class Source:
        terminal_reason = None

        def __init__(self) -> None:
            self.yielded = 0

        async def frames(self):
            for probability in (0.9, 0.0, 0.9, 0.0):
                self.yielded += 1
                yield AudioFrameF32(
                    samples=np.full((512,), probability, dtype=np.float32),
                    sample_rate_hz=16000,
                )

        async def close(self) -> None:
            return None

    backend = CustomSTTBackend(
        mode="offline",
        compatibility="openai_transcription",
        endpoint="http://127.0.0.1:8000",
        model="test",
        http_client_factory=lambda **_kwargs: Client(),
    )

    async def session_factory(_settings, epoch_id):
        return await backend.open_session(
            projection=STTSessionProjection(mode="scoped", provider_epoch_id=epoch_id)
        )

    owner_box: list[PeerCaptureSessionOwner] = []
    terminals: list[STTProviderTurnTerminal] = []

    async def terminal_sink(event) -> None:
        if not isinstance(event, STTProviderTurnTerminal):
            return
        terminals.append(event)
        owner_box[0].record_segment_terminal(
            event.identity.segment.segment_id,
            outcome=event.outcome,
            text_authority=event.text_authority,
            failure_reason=event.failure_reason,
        )

    engine = ScopedRecognitionEngine(
        session_factory=session_factory,
        event_sink=terminal_sink,
        watchdog_resolver=lambda _settings: STTRecognitionWatchdogs(
            readiness_timeout_s=0.5,
            write_timeout_s=0.5,
            final_timeout_s=0.5,
            drain_timeout_s=0.1,
            connect_retry_base_s=0.01,
            connect_retry_max_s=0.02,
        ),
    )
    source = Source()
    owner, *_ = make_owner(
        source_factory=lambda _config, _target: source,
        vad_factory=lambda _config: VadGating(
            SequenceVadEngine(probs=[0.9, 0.0, 0.9, 0.0]),
            sample_rate_hz=16000,
            chunk_samples=512,
            ring_buffer_ms=32,
            hangover_ms=0,
        ),
        run_audio_loop=run_audio_vad_loop,
        sink=engine,
    )
    owner_box.append(owner)

    running = asyncio.create_task(
        owner.apply_intent(make_config(provider_id="custom_offline"), enabled=True)
    )
    await asyncio.wait_for(request_started.wait(), timeout=0.5)
    await wait_until(lambda: source.yielded == 4)
    ledger = owner.segment_ledgers[-1]
    await wait_until(lambda: len(ledger.snapshots) == 2)

    assert [segment.state for segment in ledger.snapshots] == ["sealed", "sealed"]
    assert terminals == []
    assert sum(segment.content_sample_count for segment in ledger.snapshots) == 2048

    release_http.set()
    await running
    await wait_until(lambda: len(terminals) == 2)
    await wait_until(lambda: owner.snapshot.state is PeerCaptureSessionState.STOPPED)
    assert [terminal.outcome for terminal in terminals] == ["final", "final"]
    assert [terminal.text for terminal in terminals] == ["recognized", "recognized"]
    assert [receipt.outcome for receipt in ledger.terminal_receipts] == ["final", "final"]
    await owner.close()
    await engine.close_backend()


@pytest.mark.asyncio
async def test_capture_progresses_before_ingress_and_finite_completion_publishes_before_drain() -> (
    None
):
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

    class PublishingSink:
        def __init__(self) -> None:
            self.owner: PeerCaptureSessionOwner | None = None
            self.events: list[object] = []
            self.receipts = []

        async def handle_owned_vad_event(self, owned: object) -> None:
            event = owned.event
            self.events.append(event)
            if isinstance(event, SpeechEnd):
                assert self.owner is not None
                self.receipts.append(
                    self.owner.record_segment_terminal(
                        event.utterance_id,
                        outcome="final",
                        text_authority="authoritative",
                    )
                )

    source = FiniteSource()
    sink = PublishingSink()
    owner, _admission, _resolver, _provider, _sources, _sink = make_owner(
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
        sink=sink,
    )
    sink.owner = owner

    start_task = asyncio.create_task(owner.apply_intent(make_config(), enabled=True))
    await wait_until(lambda: provider.start_calls == 1)
    await asyncio.sleep(0.02)
    assert source.yielded == 1
    assert owner.segment_ledger is not None
    assert sink.events == []

    provider.start_gate.set()
    await start_task
    await wait_until(lambda: owner.snapshot.state is PeerCaptureSessionState.STOPPED)

    ledger = owner.segment_ledgers[-1]
    assert ledger.snapshots == ()
    assert len(sink.receipts) == 1
    receipt = sink.receipts[0]
    assert receipt.outcome == "final"
    assert receipt.segment.content_sample_count == 16
    assert receipt.segment.seal_reason == "source_eof"
    assert ledger.terminal_receipts == (receipt,)
    assert [event.__class__.__name__ for event in sink.events] == [
        "SpeechStart",
        "SpeechChunk",
        "SpeechEnd",
    ]
    assert provider.releases[-1][0] == "drain"
    assert owner.snapshot.effective_active is False
    assert owner.snapshot.has_loop_task is False
    await owner.close()


@pytest.mark.asyncio
async def test_successful_provider_drain_fails_unresolved_segment_without_empty_success() -> None:
    class FiniteSource:
        terminal_reason = None

        async def frames(self):
            yield AudioFrameF32(
                samples=np.ones((16,), dtype=np.float32),
                sample_rate_hz=16000,
            )

        async def close(self) -> None:
            return None

    owner, _admission, _resolver, provider, _sources, _sink = make_owner(
        source_factory=lambda _config, _target: FiniteSource(),
        vad_factory=lambda _config: VadGating(
            SequenceVadEngine(probs=[0.9, 0.9]),
            sample_rate_hz=16000,
            chunk_samples=8,
            ring_buffer_ms=1,
            hangover_ms=640,
        ),
        run_audio_loop=run_audio_vad_loop,
    )

    await owner.apply_intent(make_config(), enabled=True)
    await wait_until(lambda: owner.snapshot.state is PeerCaptureSessionState.STOPPED)

    ledger = owner.segment_ledgers[-1]
    assert provider.releases[-1][0] == "drain"
    assert ledger.snapshots == ()
    assert len(ledger.terminal_receipts) == 1
    receipt = ledger.terminal_receipts[0]
    assert receipt.outcome == "failed"
    assert receipt.text_authority == "none"
    assert receipt.failure_reason == "provider_drain_without_scoped_terminal"
    assert receipt.segment.seal_reason == "source_eof"
    await owner.close()


@pytest.mark.asyncio
async def test_owner_bounds_ten_thousand_terminal_receipts_without_manual_drain() -> None:
    owner, *_ = make_owner()
    await owner.apply_intent(make_config(), enabled=True)
    ledger = owner.segment_ledger
    assert ledger is not None
    first_id = None
    last_id = None
    sample = np.ones((1,), dtype=np.float32)

    for order in range(10_000):
        segment_id = uuid4()
        if first_id is None:
            first_id = segment_id
        last_id = segment_id
        capture = AudioCaptureSpan(
            capture_epoch=1,
            callback_sequence=order,
            source_sample_rate_hz=16000,
            source_start_sample=order,
            source_end_sample=order + 1,
            source_start_monotonic_s=order / 16000,
            source_end_monotonic_s=(order + 1) / 16000,
            normalized_sample_rate_hz=16000,
            normalized_start_sample=order,
            normalized_end_sample=order + 1,
        )
        ledger.observe_vad_event(
            SpeechStart(
                segment_id,
                pre_roll=np.empty((0,), dtype=np.float32),
                chunk=sample,
                chunk_capture=(capture,),
            ),
            now_monotonic_s=float(order),
        )
        ledger.observe_vad_event(
            SpeechEnd(segment_id, trailing_silence_ms=0, reason="silence"),
            now_monotonic_s=float(order) + 0.1,
        )
        owner.record_segment_terminal(
            segment_id,
            outcome=("failed" if order % 2 else "cancelled"),
            text_authority="none",
        )

    assert ledger.snapshots == ()
    assert len(ledger.terminal_receipts) == 4096
    assert first_id is not None and not ledger.contains_segment(first_id)
    assert last_id is not None and ledger.contains_segment(last_id)
    assert [receipt.identity.segment_order for receipt in ledger.terminal_receipts] == list(
        range(5905, 10_001)
    )
    await owner.close()


@pytest.mark.asyncio
async def test_blocked_provider_handoff_keeps_actual_segment_settings_until_commit() -> None:
    owner, _admission, _resolver, provider, _sources, _sink = make_owner()
    original = make_config(provider_id="soniox")
    requested = replace(make_config(provider_id="deepgram"), vad_hangover_ms=1200)
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
    assert owner.snapshot.requested_vad_hangover_ms == 1200
    assert owner.snapshot.effective_vad_hangover_ms == 900

    assert owner.snapshot.provider_id == "soniox"
    assert ledger.snapshots[0].settings.provider_id == "soniox"
    ledger.observe_vad_event(
        SpeechEnd(first_id, trailing_silence_ms=0, reason="silence"),
        now_monotonic_s=0.1,
    )

    provider.handoff_gate.set()
    await transition
    assert owner.snapshot.provider_id == "deepgram"
    assert owner.snapshot.requested_vad_hangover_ms == 1200
    assert owner.snapshot.effective_vad_hangover_ms == 1200

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
async def test_off_cancels_blocked_running_provider_handoff() -> None:
    owner, _admission, _resolver, provider, _sources, _sink = make_owner()
    original = make_config(provider_id="soniox")
    requested = make_config(provider_id="deepgram")
    await owner.apply_intent(original, enabled=True)

    provider.handoff_gate = asyncio.Event()
    transition = asyncio.create_task(owner.apply_intent(requested, enabled=True))
    await wait_until(lambda: len(provider.handoffs) == 1)

    stopped = await asyncio.wait_for(
        owner.apply_intent(requested, enabled=False),
        timeout=0.5,
    )
    await transition

    assert stopped.state is PeerCaptureSessionState.STOPPED
    assert provider.cancel_calls == 1
    assert provider.releases[-1] == ("abort", None)

    await owner.close()


@pytest.mark.asyncio
async def test_peer_dispatch_keeps_five_fresh_segments_over_twelve_total_seconds() -> None:
    blocked = asyncio.Event()
    release = asyncio.Event()

    class FiveSegmentSource:
        terminal_reason = None

        def __init__(self) -> None:
            self.yielded = 0

        async def frames(self):
            probabilities = ([0.9] * 94 + [0.0]) * 5
            started_at = asyncio.get_running_loop().time()
            for sequence, probability in enumerate(probabilities):
                self.yielded += 1
                source_start = sequence * 512
                yield AudioFrameF32(
                    samples=np.full((512,), probability, dtype=np.float32),
                    sample_rate_hz=16000,
                    capture=AudioCaptureSpan(
                        capture_epoch=1,
                        callback_sequence=sequence,
                        source_sample_rate_hz=16000,
                        source_start_sample=source_start,
                        source_end_sample=source_start + 512,
                        source_start_monotonic_s=started_at + sequence * 0.032,
                        source_end_monotonic_s=started_at + (sequence + 1) * 0.032,
                    ),
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

    source = FiveSegmentSource()
    sink = SlowSink()
    config = replace(
        make_config(),
        vad_hangover_ms=32,
        runtime_signature=("soniox", "physical", "hangover", 32),
    )
    owner, *_ = make_owner(
        source_factory=lambda _config, _target: source,
        vad_factory=lambda current: create_peer_vad_gating(
            SequenceVadEngine(probs=([0.9] * 94 + [0.0]) * 5),
            sample_rate_hz=16000,
            ring_buffer_ms=current.vad_pre_roll_ms,
            speech_threshold=current.vad_speech_threshold,
            hangover_ms=current.vad_hangover_ms,
        ),
        run_audio_loop=run_audio_vad_loop,
        sink=sink,
    )

    await owner.apply_intent(config, enabled=True)
    await asyncio.wait_for(blocked.wait(), timeout=0.5)
    await wait_until(lambda: source.yielded == 475)

    ledger = owner.segment_ledger
    assert ledger is not None
    await wait_until(
        lambda: (
            len(ledger.snapshots) == 5
            and all(segment.state == "sealed" for segment in ledger.snapshots)
        )
    )
    assert sum(segment.content_sample_count for segment in ledger.snapshots) == 243200
    assert ledger.terminal_receipts == ()
    assert owner.snapshot.state is PeerCaptureSessionState.RUNNING

    release.set()
    await wait_until(lambda: owner.snapshot.state is PeerCaptureSessionState.STOPPED)
    assert sink.events == 480
    await owner.close()


@pytest.mark.asyncio
async def test_peer_dispatch_accepts_32_reserved_controls_and_rejects_33rd() -> None:
    blocked = asyncio.Event()
    release = asyncio.Event()

    class SlowSink:
        async def handle_owned_vad_event(self, _event: object) -> None:
            blocked.set()
            await release.wait()

    owner, *_ = make_owner(sink=SlowSink())
    await owner.apply_intent(make_config(), enabled=True)
    guarded = owner.guard_vad_sink()
    ledger = owner.segment_ledger
    assert ledger is not None

    for index in range(33):
        utterance_id = uuid4()
        ledger.observe_vad_event(
            SpeechStart(
                utterance_id,
                pre_roll=np.empty(0, dtype=np.float32),
                chunk=np.empty(0, dtype=np.float32),
            ),
            now_monotonic_s=float(index),
        )
        owned_end = ledger.observe_vad_event(
            SpeechEnd(utterance_id),
            now_monotonic_s=float(index),
        )
        if index < 32:
            await guarded.handle_owned_vad_event(owned_end)
            continue
        with pytest.raises(RuntimeError, match="control event budget"):
            await guarded.handle_owned_vad_event(owned_end)

    await guarded.abort()
    release.set()
    await owner.close()


@pytest.mark.asyncio
async def test_peer_dispatch_reserves_capacity_for_eight_wholly_unsent_segments() -> None:
    blocked = asyncio.Event()
    release = asyncio.Event()

    class NineSegmentSource:
        terminal_reason = None

        def __init__(self) -> None:
            self.yielded = 0

        async def frames(self):
            for probability in [0.9, 0.0] * 9:
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

    source = NineSegmentSource()
    sink = SlowSink()
    owner, *_ = make_owner(
        source_factory=lambda _config, _target: source,
        vad_factory=lambda _config: VadGating(
            SequenceVadEngine(probs=[0.9, 0.0] * 9),
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
    await wait_until(lambda: source.yielded == 18)
    await wait_until(
        lambda: (
            owner.segment_ledger is not None
            and len(owner.segment_ledger.snapshots) == 9
            and all(segment.seal_reason == "silence" for segment in owner.segment_ledger.snapshots)
        )
    )

    release.set()
    await wait_until(lambda: owner.snapshot.state is PeerCaptureSessionState.STOPPED)
    assert sink.events == 27
    await owner.close()


@pytest.mark.asyncio
async def test_peer_dispatch_expires_oldest_wholly_unsent_segment_on_overflow() -> None:
    blocked = asyncio.Event()
    release = asyncio.Event()

    class TenSegmentSource:
        terminal_reason = None

        async def frames(self):
            for probability in [0.9, 0.0] * 10:
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

    sink = SlowSink()
    owner, *_ = make_owner(
        source_factory=lambda _config, _target: TenSegmentSource(),
        vad_factory=lambda _config: VadGating(
            SequenceVadEngine(probs=[0.9, 0.0] * 10),
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
    ledger = owner.segment_ledgers[-1]
    await wait_until(
        lambda: len(ledger.snapshots) == 10 and ledger.snapshots[1].state == "terminal"
    )
    assert ledger.snapshots[1].identity.segment_order == 2
    assert ledger.snapshots[1].seal_reason == "silence"
    assert ledger.terminal_receipts[0].outcome == "expired"

    release.set()
    await wait_until(lambda: owner.snapshot.state is PeerCaptureSessionState.STOPPED)
    assert sink.events == 27
    assert [receipt.outcome for receipt in ledger.terminal_receipts[:2]] == [
        "failed",
        "expired",
    ]
    assert ledger.terminal_receipts[0].failure_reason == "provider_drain_without_scoped_terminal"
    assert ledger.terminal_receipts[1].failure_reason == "overload"
    await owner.close()


@pytest.mark.asyncio
async def test_peer_dispatch_expires_wholly_unsent_segment_after_seal_age_timer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    blocked = asyncio.Event()
    release = asyncio.Event()
    monkeypatch.setattr(
        "puripuly_heart.core.runtime.peer_channel._GenerationGuardedVadSink._SEALED_SEGMENT_TTL_S",
        0.01,
    )

    class TwoSegmentSource:
        terminal_reason = None

        async def frames(self):
            for probability in [0.9, 0.0] * 2:
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

    sink = SlowSink()
    owner, *_ = make_owner(
        source_factory=lambda _config, _target: TwoSegmentSource(),
        vad_factory=lambda _config: VadGating(
            SequenceVadEngine(probs=[0.9, 0.0] * 2),
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
    ledger = owner.segment_ledgers[-1]
    await wait_until(lambda: len(ledger.snapshots) == 2 and ledger.snapshots[1].state == "terminal")
    assert ledger.terminal_receipts[0].outcome == "expired"

    release.set()
    await wait_until(lambda: owner.snapshot.state is PeerCaptureSessionState.STOPPED)
    assert sink.events == 3
    assert [receipt.outcome for receipt in ledger.terminal_receipts] == [
        "failed",
        "expired",
    ]
    assert ledger.terminal_receipts[0].failure_reason == "provider_drain_without_scoped_terminal"
    assert ledger.terminal_receipts[1].failure_reason == "expired_before_recognition"
    await owner.close()


@pytest.mark.asyncio
async def test_off_cancels_blocked_provider_setup_after_capture_has_progressed() -> None:
    provider = FakeProvider()
    provider.replace_gate = asyncio.Event()

    class StreamingSource:
        terminal_reason = None

        def __init__(self) -> None:
            self.yielded = 0
            self.close_calls = 0

        async def frames(self):
            for _ in range(6):
                self.yielded += 1
                yield AudioFrameF32(
                    samples=np.ones((512,), dtype=np.float32),
                    sample_rate_hz=16000,
                )
            await asyncio.Event().wait()

        async def close(self) -> None:
            self.close_calls += 1

    source = StreamingSource()
    owner, _admission, _resolver, _provider, _sources, sink = make_owner(
        provider=provider,
        source_factory=lambda _config, _target: source,
        vad_factory=lambda config: create_peer_vad_gating(
            SequenceVadEngine(probs=[0.9] * 6),
            sample_rate_hz=config.target_sample_rate_hz,
            ring_buffer_ms=config.vad_pre_roll_ms,
            hangover_ms=config.vad_hangover_ms,
        ),
        run_audio_loop=run_audio_vad_loop,
    )
    config = make_config()

    start = asyncio.create_task(owner.apply_intent(config, enabled=True))
    await wait_until(lambda: source.yielded == 6)
    ledger = owner.segment_ledgers[-1]
    stop = asyncio.create_task(owner.apply_intent(config, enabled=False))
    await asyncio.wait_for(asyncio.gather(start, stop), timeout=0.5)

    assert owner.snapshot.state is PeerCaptureSessionState.STOPPED
    assert owner.snapshot.effective_active is False
    assert source.close_calls == 1
    assert sink.events == []
    assert [receipt.outcome for receipt in ledger.terminal_receipts] == ["cancelled"]


@pytest.mark.asyncio
@pytest.mark.parametrize("blocked_channel", ["self", "peer"])
@pytest.mark.parametrize("transition_kind", ["initial", "handoff"])
async def test_actual_capture_owners_isolate_cross_channel_setup_pressure(
    blocked_channel: str,
    transition_kind: str,
) -> None:
    from puripuly_heart.core.runtime.self_capture import SelfCaptureSessionOwner
    from puripuly_heart.core.self_capture import (
        SelfCaptureAdmission,
        SelfCaptureAdmissionStatus,
        SelfCaptureProviderMutation,
        SelfCaptureProviderMutationStatus,
        SelfCaptureSessionConfig,
    )

    setup_entered = asyncio.Event()
    setup_release = asyncio.Event()
    loop_ticks = 0

    class StreamingSource:
        terminal_reason = None

        def __init__(self) -> None:
            self.consumed = 0
            self.closed = False

        async def frames(self):
            while not self.closed:
                self.consumed += 1
                yield AudioFrameF32(
                    samples=np.zeros((512,), dtype=np.float32),
                    sample_rate_hz=16_000,
                )
                await asyncio.sleep(0.005)

        async def close(self) -> None:
            self.closed = True

    class SelfAdmission:
        async def admit(self, _config):
            return SelfCaptureAdmission(SelfCaptureAdmissionStatus.ADMITTED)

    class SelfProvider:
        def __init__(self) -> None:
            self.ready = False
            self.handoff_gate: asyncio.Event | None = None

        def is_ready(self, _config) -> bool:
            return self.ready

        async def replace(self, *_args, **_kwargs):
            if blocked_channel == "self" and transition_kind == "initial":
                setup_entered.set()
                await setup_release.wait()
            self.ready = True
            return SelfCaptureProviderMutation(SelfCaptureProviderMutationStatus.APPLIED)

        async def handoff(self, *_args, **_kwargs):
            if self.handoff_gate is not None:
                setup_entered.set()
                await self.handoff_gate.wait()
            return SelfCaptureProviderMutation(SelfCaptureProviderMutationStatus.APPLIED)

        async def cancel_handoff(self) -> bool:
            return True

        async def start_ingress(self) -> None:
            return None

        async def warmup(self) -> None:
            return None

        async def reconfigure(self, _options) -> None:
            return None

        async def release(self, **_kwargs) -> None:
            self.ready = False

    self_source = StreamingSource()
    self_provider = SelfProvider()
    self_owner = SelfCaptureSessionOwner(
        admission=SelfAdmission(),
        provider=self_provider,
        provider_request_factory=lambda *_args: object(),
        source_factory=lambda _config: self_source,
        vad_factory=lambda _config: VadGating(
            SequenceVadEngine(probs=[0.0]),
            sample_rate_hz=16_000,
            chunk_samples=512,
            ring_buffer_ms=32,
            hangover_ms=640,
        ),
        run_audio_loop=run_audio_vad_loop,
        vad_sink=FakeVadSink(),
    )
    self_config = SelfCaptureSessionConfig(
        provider_id="legacy-self",
        provider_signature=("legacy-self",),
        runtime_signature=("legacy-self",),
        capture_signature=("self-mic",),
        target_sample_rate_hz=16_000,
        session_options=None,
    )

    peer_source = StreamingSource()
    peer_provider = FakeProvider()
    if blocked_channel == "peer":
        if transition_kind == "initial":
            peer_provider.replace_gate = setup_release

            async def blocked_replace(request, *, start, on_terminal_failure):
                setup_entered.set()
                return await FakeProvider.replace(
                    peer_provider,
                    request,
                    start=start,
                    on_terminal_failure=on_terminal_failure,
                )

            peer_provider.replace = blocked_replace

        else:
            peer_provider.handoff_gate = setup_release

            async def blocked_handoff(request, *, start, on_terminal_failure):
                setup_entered.set()
                return await FakeProvider.handoff(
                    peer_provider,
                    request,
                    start=start,
                    on_terminal_failure=on_terminal_failure,
                )

            peer_provider.handoff = blocked_handoff
    peer_owner, *_ = make_owner(
        provider=peer_provider,
        source_factory=lambda _config, _target: peer_source,
        vad_factory=lambda config: create_peer_vad_gating(
            SequenceVadEngine(probs=[0.0]),
            sample_rate_hz=config.target_sample_rate_hz,
            ring_buffer_ms=config.vad_pre_roll_ms,
            hangover_ms=config.vad_hangover_ms,
        ),
        run_audio_loop=run_audio_vad_loop,
    )

    async def tick_common_loop() -> None:
        nonlocal loop_ticks
        while not setup_release.is_set():
            loop_ticks += 1
            await asyncio.sleep(0.005)

    tick_task = asyncio.create_task(tick_common_loop())
    if transition_kind == "handoff":
        await asyncio.gather(
            self_owner.apply_intent(self_config, enabled=True),
            peer_owner.apply_intent(make_config(), enabled=True),
        )
        if blocked_channel == "self":
            self_provider.handoff_gate = setup_release
            changed_self = replace(
                self_config,
                provider_id="legacy-self-next",
                provider_signature=("legacy-self-next",),
                runtime_signature=("legacy-self-next",),
            )
            start = asyncio.create_task(self_owner.apply_intent(changed_self, enabled=True))
        else:
            start = asyncio.create_task(
                peer_owner.apply_intent(make_config(provider_id="deepgram"), enabled=True)
            )
        opposite_before = (
            peer_source.consumed if blocked_channel == "self" else self_source.consumed
        )
    elif blocked_channel == "self":
        await peer_owner.apply_intent(make_config(), enabled=True)
        start = asyncio.create_task(self_owner.apply_intent(self_config, enabled=True))
        opposite_before = peer_source.consumed
    else:
        await self_owner.apply_intent(self_config, enabled=True)
        start = asyncio.create_task(peer_owner.apply_intent(make_config(), enabled=True))
        opposite_before = self_source.consumed

    try:
        await asyncio.wait_for(setup_entered.wait(), timeout=1.0)
        await asyncio.sleep(0.05)
        assert loop_ticks > 1
        if blocked_channel == "self":
            assert peer_source.consumed > opposite_before
            if transition_kind == "handoff":
                assert self_source.consumed > 1
            else:
                assert self_source.consumed == 0
        else:
            assert self_source.consumed > opposite_before
            assert peer_source.consumed > 1
        setup_release.set()
        await asyncio.wait_for(start, timeout=1.0)
    finally:
        setup_release.set()
        await asyncio.gather(start, return_exceptions=True)
        await peer_owner.close()
        await self_owner.close()
        await tick_task

    assert peer_source.closed is True
    if blocked_channel == "peer":
        assert self_source.closed is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "smart_turn_enabled",
    [False, True],
    ids=["smart-turn-off", "smart-turn-on"],
)
async def test_canonical_delivery_boundaries_survive_production_write_timeout_and_recover(
    smart_turn_enabled: bool,
) -> None:
    from puripuly_heart.app.wiring.wiring_local_asr_provider_runtime import (
        _recognition_watchdogs,
    )

    writer_entered = asyncio.Event()
    writer_release = asyncio.Event()
    source_release = asyncio.Event()
    continue_source = asyncio.Event()

    class NativeSession:
        allows_interim_timeout_fallback = False

        def __init__(self, *, block_writer: bool, terminal_text: str) -> None:
            self.block_writer = block_writer
            self.terminal_text = terminal_text
            self.events: asyncio.Queue[object | None] = asyncio.Queue()
            self.identities: list[STTProviderTurnIdentity] = []
            self.progress: list[str] = []

        async def begin_turn(self, request) -> None:
            self.identities.append(request.identity)
            self.progress.append("begin_written")

        async def send_turn_audio(self, _identity, _pcm16le, **_kwargs) -> None:
            self.progress.append("sdk_enqueued")
            if self.block_writer:
                writer_entered.set()
                await writer_release.wait()
            self.progress.append("sdk_consumed")

        async def seal_turn(self, identity, **_kwargs) -> None:
            self.progress.append("seal_written")
            self.events.put_nowait(
                STTProviderTurnTerminal(
                    identity=identity,
                    outcome="final",
                    text=self.terminal_text,
                    text_authority="authoritative",
                )
            )

        async def abort_turn(self, _identity, **_kwargs) -> None:
            self.progress.append("abort_written")

        async def turn_events(self):
            while (event := await self.events.get()) is not None:
                yield event

        async def stop(self) -> None:
            self.progress.append("stop")

        async def close(self) -> None:
            self.progress.append("close")
            self.events.put_nowait(None)

    sessions: list[NativeSession] = []

    async def open_session(_settings, _epoch_id):
        session = NativeSession(
            block_writer=not sessions,
            terminal_text="pressure recovered",
        )
        sessions.append(session)
        return session

    watchdog_config = SimpleNamespace(provider="soniox", drain_timeout_s=0.1)
    recognition = ScopedRecognitionEngine(
        channel="peer",
        session_factory=open_session,
        watchdog_resolver=lambda _settings: _recognition_watchdogs(watchdog_config),
    )

    class PressureSource:
        terminal_reason = None

        def __init__(self) -> None:
            self.close_calls = 0

        async def frames(self):
            timeline_start = time.monotonic()
            for sequence in range(180):
                source_end = timeline_start + (sequence + 1) * 0.032
                start = sequence * 512
                yield AudioFrameF32(
                    samples=np.ones((512,), dtype=np.float32),
                    sample_rate_hz=16_000,
                    capture=AudioCaptureSpan(
                        capture_epoch=0,
                        callback_sequence=sequence,
                        source_sample_rate_hz=16_000,
                        source_start_sample=start,
                        source_end_sample=start + 512,
                        source_start_monotonic_s=source_end - 0.032,
                        source_end_monotonic_s=source_end,
                    ),
                )
            await continue_source.wait()
            pause_frames = 30
            continuation_start = time.monotonic()
            for offset in range(10 + pause_frames):
                sequence = 180 + offset
                source_end = continuation_start + (offset + 1) * 0.032
                start = sequence * 512
                yield AudioFrameF32(
                    samples=np.full(
                        (512,),
                        0.0 if offset >= 10 else 1.0,
                        dtype=np.float32,
                    ),
                    sample_rate_hz=16_000,
                    capture=AudioCaptureSpan(
                        capture_epoch=0,
                        callback_sequence=sequence,
                        source_sample_rate_hz=16_000,
                        source_start_sample=start,
                        source_end_sample=start + 512,
                        source_start_monotonic_s=source_end - 0.032,
                        source_end_monotonic_s=source_end,
                    ),
                )
            await source_release.wait()

        async def close(self) -> None:
            self.close_calls += 1
            source_release.set()

    guarded_sinks: list[object] = []
    pressure_observation = SimpleNamespace(
        high_water=0,
        pcm_samples=0,
        content_samples=0,
        context_samples=0,
        segment_ids=set(),
        segment_orders=set(),
        oldest_waiting_age_s=0.0,
    )

    async def observed_audio_loop(**kwargs) -> None:
        guarded = kwargs["sink"]
        guarded_sinks.append(guarded)

        async def sample_pressure() -> None:
            while True:
                queued = tuple(guarded._queue)
                if len(queued) > pressure_observation.high_water:
                    pressure_observation.high_water = len(queued)
                    pressure_observation.pcm_samples = guarded._queued_pcm_samples
                    pressure_observation.content_samples = guarded._queued_content_samples
                    pressure_observation.context_samples = guarded._queued_context_samples
                    pressure_observation.segment_ids = {
                        item.segment_id for item in queued if item.segment_id
                    }
                    pressure_observation.segment_orders = {
                        item.segment_order for item in queued if item.segment_order
                    }
                sealed_entries = [item for item in queued if item.sealed_at_dispatch_s is not None]
                if sealed_entries:
                    age = asyncio.get_running_loop().time() - min(
                        item.sealed_at_dispatch_s for item in sealed_entries
                    )
                    pressure_observation.oldest_waiting_age_s = max(
                        pressure_observation.oldest_waiting_age_s,
                        age,
                    )
                await asyncio.sleep(0.001)

        sampler = asyncio.create_task(sample_pressure())
        try:
            await run_audio_vad_loop(**kwargs)
        finally:
            sampler.cancel()
            await asyncio.gather(sampler, return_exceptions=True)

    peer_source = PressureSource()
    probabilities = [0.9] * (180 + 10) + [0.0] * 30
    owner, *_ = make_owner(
        source_factory=lambda _config, _target: peer_source,
        vad_factory=lambda config: create_peer_vad_gating(
            SequenceVadEngine(probs=probabilities),
            sample_rate_hz=config.target_sample_rate_hz,
            ring_buffer_ms=config.vad_pre_roll_ms,
            hangover_ms=config.vad_hangover_ms,
        ),
        run_audio_loop=observed_audio_loop,
        clock=SystemClock(),
        sink=recognition,
    )

    class RecordingOverlay:
        def __init__(self) -> None:
            self.events: list[object] = []

        async def emit(self, event: object) -> None:
            self.events.append(event)

        def active_self_overlay_metadata(self) -> None:
            return None

    overlay = RecordingOverlay()
    translation = compose_translation_test_harness(
        stt=None,
        llm=None,
        osc=RecordingOscQueue(),
        overlay_sink=overlay,
    )
    callbacks = TranslationChannelOwnerCallbacks(SttSessionStateProjection())
    callbacks.bind_peer_capture(owner)
    callbacks.bind_peer(translation.peer_owner)
    recognition.bind_event_sink(callbacks.peer_event_handler)
    owner.bind_publication_generation_observer(
        activated=translation.output_runtime.activate_peer_generation,
        retired=translation.output_runtime.retire_peer_generation,
    )

    try:
        await translation.start()
        await owner.apply_intent(
            replace(make_config(), smart_turn_enabled=smart_turn_enabled),
            enabled=True,
        )
        await asyncio.wait_for(writer_entered.wait(), timeout=1.0)
        ledger = owner.segment_ledgers[-1]
        await wait_until(lambda: bool(ledger.snapshots))
        segment = ledger.snapshots[0]
        due_at = segment.opened_at_monotonic_s + ListenDeliveryController.HARD_LIMIT_S
        await wait_until(
            lambda: ledger.snapshots[0].seal_reason == "delivery_deadline",
            timeout_s=ListenDeliveryController.HARD_LIMIT_S + 1.0,
        )
        segment = ledger.snapshots[0]
        fired_at = segment.sealed_at_monotonic_s
        assert fired_at is not None
        assert -0.02 <= fired_at - due_at < 0.25
        assert segment.content_sample_count == 180 * 512
        assert (
            segment.content_ranges[0].normalized_start_sample,
            segment.content_ranges[-1].normalized_end_sample,
        ) == (0, 180 * 512)

        guarded = guarded_sinks[0]
        queued = tuple(guarded._queue)
        sealed_entries = [item for item in queued if item.sealed_at_dispatch_s is not None]
        current_oldest_waiting_age_s = asyncio.get_running_loop().time() - min(
            item.sealed_at_dispatch_s for item in sealed_entries
        )
        assert pressure_observation.high_water == 179
        assert pressure_observation.pcm_samples == 180 * 512
        assert pressure_observation.content_samples == 180 * 512
        assert pressure_observation.context_samples == 0
        assert current_oldest_waiting_age_s >= 0
        assert pressure_observation.segment_ids == {segment.identity.segment_id}
        assert pressure_observation.segment_orders == {segment.identity.segment_order}
        assert {item.segment_id for item in queued if item.segment_id} == {
            segment.identity.segment_id
        }
        assert sessions[0].progress == ["begin_written", "sdk_enqueued"]
        writer_release.set()
        await wait_until(lambda: recognition.cleanup_debt == 0)
        await wait_until(lambda: len(ledger.terminal_receipts) == 1)
        failed_receipt = ledger.terminal_receipts[0]
        assert failed_receipt.identity == segment.identity
        assert failed_receipt.outcome == "failed"
        assert failed_receipt.failure_reason == "provider_send_timeout"
        assert overlay.events == []

        continue_source.set()
        await wait_until(
            lambda: (
                len(ledger.snapshots) == 1 and ledger.snapshots[0].seal_reason == "delivery_pause"
            ),
            timeout_s=5.0,
        )
        pause_segment = ledger.snapshots[0]
        await wait_until(lambda: len(ledger.terminal_receipts) == 2)
        expected_pause_frames = 35 if smart_turn_enabled else 39
        assert pause_segment.content_sample_count == expected_pause_frames * 512
        assert (
            pause_segment.content_ranges[0].normalized_start_sample,
            pause_segment.content_ranges[-1].normalized_end_sample,
        ) == (180 * 512, (180 + expected_pause_frames) * 512)
        recovered_receipt = ledger.terminal_receipts[1]
        assert recovered_receipt.identity == pause_segment.identity
        assert recovered_receipt.outcome == "final"
        await translation.translation_turns.wait_for_idle()
        await translation.output_runtime.wait_for_peer_output_idle()
        finals = [
            event
            for event in overlay.events
            if getattr(event, "type", None) == "peer_transcript_final"
        ]
        assert len(finals) == 1
        assert len(sessions) == 2
        assert sessions[1].progress[-1] == "seal_written"
        assert owner.snapshot.cleanup_debt == 0
        assert recognition.cleanup_debt == 0
    finally:
        writer_release.set()
        source_release.set()
        await owner.close()
        await recognition.close_backend()
        await translation.stop()

    assert peer_source.close_calls == 1
    assert sessions[0].progress.count("stop") == 1
    assert sessions[0].progress.count("close") == 1


@pytest.mark.asyncio
async def test_off_then_reenable_during_actual_scoped_write_stall_retires_old_scope() -> None:
    from puripuly_heart.app.wiring.wiring_local_asr_provider_runtime import (
        _recognition_watchdogs,
    )

    writer_entered = asyncio.Event()
    writer_release = asyncio.Event()
    emitted: list[STTProviderTurnTerminal] = []
    owner_box: list[PeerCaptureSessionOwner] = []

    class ScopedSession:
        allows_interim_timeout_fallback = False

        def __init__(self, blocked: bool) -> None:
            self.blocked = blocked
            self.events: asyncio.Queue[object | None] = asyncio.Queue()
            self.identity: STTProviderTurnIdentity | None = None
            self.stop_calls = 0
            self.close_calls = 0

        async def begin_turn(self, request) -> None:
            self.identity = request.identity

        async def send_turn_audio(self, _identity, _pcm16le, **_kwargs) -> None:
            if self.blocked:
                writer_entered.set()
                await writer_release.wait()

        async def seal_turn(self, identity, **_kwargs) -> None:
            self.events.put_nowait(
                STTProviderTurnTerminal(
                    identity=identity,
                    outcome="final",
                    text="new generation",
                    text_authority="authoritative",
                )
            )

        async def abort_turn(self, _identity, **_kwargs) -> None:
            return None

        async def turn_events(self):
            while (event := await self.events.get()) is not None:
                yield event

        async def stop(self) -> None:
            self.stop_calls += 1

        async def close(self) -> None:
            self.close_calls += 1
            self.events.put_nowait(None)

    class ScopedProvider:
        def __init__(self) -> None:
            self.provider_id: str | None = None
            self.sessions: list[ScopedSession] = []
            self.engines: list[ScopedRecognitionEngine] = []
            self.current: ScopedRecognitionEngine | None = None

        def is_ready(self, config) -> bool:
            return self.provider_id == config.provider_id

        async def replace(self, request, **_kwargs):
            session = ScopedSession(blocked=not self.sessions)
            self.sessions.append(session)

            async def terminal_sink(event) -> None:
                if not isinstance(event, STTProviderTurnTerminal):
                    return
                admissions = owner_box[0].admit_provider_terminal(event)
                emitted.extend(terminal for _receipt, terminal in admissions)

            config = SimpleNamespace(provider="soniox", drain_timeout_s=0.1)
            engine = ScopedRecognitionEngine(
                session_factory=lambda _settings, _epoch: asyncio.sleep(0, result=session),
                event_sink=terminal_sink,
                watchdog_resolver=lambda _settings: _recognition_watchdogs(config),
            )
            self.engines.append(engine)
            self.current = engine
            self.provider_id = request[0]
            return PeerCaptureProviderMutation(PeerCaptureProviderMutationStatus.APPLIED)

        async def handoff(self, request, **kwargs):
            return await self.replace(request, **kwargs)

        async def cancel_handoff(self) -> bool:
            return True

        async def start_ingress(self) -> None:
            return None

        async def warmup(self) -> None:
            return None

        async def reconfigure(self, _options) -> None:
            return None

        async def release(self, **_kwargs) -> None:
            retiring = self.current
            self.current = None
            self.provider_id = None
            if retiring is not None:
                await retiring.close()

        async def handle_owned_vad_event(self, event) -> None:
            engine = self.current
            if engine is not None:
                await engine.handle_owned_vad_event(event)

    class GenerationSource:
        terminal_reason = None

        def __init__(self, probabilities: list[float]) -> None:
            self.probabilities = probabilities
            self.closed = asyncio.Event()
            self.close_calls = 0

        async def frames(self):
            for probability in self.probabilities:
                yield AudioFrameF32(
                    samples=np.full((512,), probability, dtype=np.float32),
                    sample_rate_hz=16_000,
                )
            await self.closed.wait()

        async def close(self) -> None:
            self.close_calls += 1
            self.closed.set()

    first_source = GenerationSource([0.9])
    second_source = GenerationSource([0.9] + [0.0] * 30)
    sources = iter((first_source, second_source))
    provider = ScopedProvider()
    owner, *_ = make_owner(
        provider=provider,
        source_factory=lambda _config, _target: next(sources),
        vad_factory=lambda _config: VadGating(
            SequenceVadEngine(probs=[0.9] + [0.0] * 30),
            sample_rate_hz=16_000,
            chunk_samples=512,
            ring_buffer_ms=32,
            hangover_ms=900,
        ),
        run_audio_loop=run_audio_vad_loop,
        sink=provider,
    )
    owner_box.append(owner)

    try:
        await owner.apply_intent(make_config(), enabled=True)
        await asyncio.wait_for(writer_entered.wait(), timeout=1.0)
        old_ledger = owner.segment_ledgers[-1]
        await owner.apply_intent(make_config(), enabled=False)
        await owner.apply_intent(make_config(), enabled=True)
        new_ledger = owner.segment_ledgers[-1]
        assert writer_release.is_set() is False
        assert old_ledger.terminal_receipts[0].outcome == "cancelled"

        old_session = provider.sessions[0]
        assert old_session.identity is not None
        old_session.events.put_nowait(
            STTProviderTurnTerminal(
                identity=old_session.identity,
                outcome="final",
                text="stale generation",
                text_authority="authoritative",
            )
        )
        writer_release.set()
        await wait_until(lambda: provider.engines[0].cleanup_debt == 0)
        await wait_until(lambda: len(new_ledger.terminal_receipts) == 1)
        assert [event.identity.segment.activation_generation for event in emitted] == [
            new_ledger.activation_generation
        ]
        assert new_ledger.terminal_receipts[0].outcome == "final"
        assert old_session.stop_calls == 1
        assert old_session.close_calls == 1
        assert first_source.close_calls == 1
    finally:
        writer_release.set()
        await owner.close()
        for engine in provider.engines:
            await engine.close_backend()

    assert second_source.close_calls == 1
    assert owner.snapshot.cleanup_debt == 0


@pytest.mark.asyncio
async def test_prospective_speaker_receiver_uses_owned_rollover_boundary_once() -> None:
    continue_source = asyncio.Event()

    class PausingSource:
        terminal_reason = None

        async def frames(self):
            for _ in range(3):
                yield AudioFrameF32(
                    samples=np.ones((512,), dtype=np.float32),
                    sample_rate_hz=16000,
                )
            await continue_source.wait()
            yield AudioFrameF32(
                samples=np.ones((512,), dtype=np.float32),
                sample_rate_hz=16000,
            )
            await asyncio.Event().wait()

        async def close(self) -> None:
            return None

    owner, *_ = make_owner(
        source_factory=lambda _config, _target: PausingSource(),
        vad_factory=lambda config: create_peer_vad_gating(
            SequenceVadEngine(probs=[0.9] * 4),
            sample_rate_hz=config.target_sample_rate_hz,
            ring_buffer_ms=config.vad_pre_roll_ms,
            hangover_ms=config.vad_hangover_ms,
        ),
        run_audio_loop=run_audio_vad_loop,
    )
    await owner.apply_intent(make_config(), enabled=True)
    ledger = owner.segment_ledgers[-1]
    await wait_until(
        lambda: bool(ledger.snapshots) and ledger.snapshots[0].content_sample_count == 1536
    )
    base = dict(
        revision=1,
        capture_epoch=ledger.snapshots[0].identity.capture_epoch,
        support_start_sample=0,
        support_end_sample=1536,
        estimated_transition_sample=768,
        observed_frontier_sample=1536,
        available_at_monotonic_s=owner.clock.now(),
        producer_generation="producer-1",
        reference_generation="reference-1",
        producer_valid=True,
        reference_valid=True,
    )
    applied = await owner.receive_prospective_speaker_hypothesis(
        ProspectiveSpeakerHypothesis(hypothesis_id="applied", **base)
    )
    duplicate = await owner.receive_prospective_speaker_hypothesis(
        ProspectiveSpeakerHypothesis(hypothesis_id="applied", **base)
    )
    continue_source.set()
    await wait_until(lambda: len(ledger.snapshots) == 2)
    already = await owner.receive_prospective_speaker_hypothesis(
        ProspectiveSpeakerHypothesis(
            hypothesis_id="already",
            **{
                **base,
                "support_start_sample": 1536,
                "support_end_sample": 1536,
                "estimated_transition_sample": 1536,
            },
        )
    )
    late = await owner.receive_prospective_speaker_hypothesis(
        ProspectiveSpeakerHypothesis(hypothesis_id="late", **base)
    )
    invalid = await owner.receive_prospective_speaker_hypothesis(
        ProspectiveSpeakerHypothesis(
            hypothesis_id="invalid",
            **{**base, "producer_valid": False},
        )
    )
    invalid_reference = await owner.receive_prospective_speaker_hypothesis(
        ProspectiveSpeakerHypothesis(
            hypothesis_id="invalid-reference",
            **{**base, "reference_valid": False},
        )
    )
    retracted = await owner.receive_prospective_speaker_hypothesis(
        ProspectiveSpeakerHypothesis(
            hypothesis_id="retracted",
            **{**base, "producer_valid": False, "retracted": True},
        )
    )
    second_seal = await owner.receive_prospective_speaker_hypothesis(
        ProspectiveSpeakerHypothesis(
            hypothesis_id="second-seal",
            **{
                **base,
                "support_start_sample": 1536,
                "support_end_sample": 2048,
                "estimated_transition_sample": 1792,
                "observed_frontier_sample": 2048,
            },
        )
    )
    identities = [snapshot.identity for snapshot in ledger.snapshots]
    terminal_a = STTProviderTurnTerminal(
        identity=STTProviderTurnIdentity(identities[0], "epoch", "turn-a"),
        outcome="final",
        text="first",
        text_authority="authoritative",
    )
    terminal_b = STTProviderTurnTerminal(
        identity=STTProviderTurnIdentity(identities[1], "epoch", "turn-b"),
        outcome="final",
        text="second",
        text_authority="authoritative",
    )

    class RecordingOverlay:
        def __init__(self) -> None:
            self.events: list[object] = []

        async def emit(self, event: object) -> None:
            self.events.append(event)

        def active_self_overlay_metadata(self) -> None:
            return None

    overlay = RecordingOverlay()
    harness = compose_translation_test_harness(
        stt=None,
        llm=None,
        osc=RecordingOscQueue(),
        overlay_sink=overlay,
    )
    callbacks = TranslationChannelOwnerCallbacks(SttSessionStateProjection())
    callbacks.bind_peer_capture(owner)
    callbacks.bind_peer(harness.peer_owner)
    owner.bind_publication_generation_observer(
        activated=harness.output_runtime.activate_peer_generation,
        retired=harness.output_runtime.retire_peer_generation,
    )
    harness.output_runtime.activate_peer_generation(identities[0].activation_generation)
    await harness.start()
    await callbacks.peer_event_handler(terminal_b)
    assert overlay.events == []
    await callbacks.peer_event_handler(terminal_a)
    await harness.translation_turns.wait_for_idle()
    await harness.output_runtime.wait_for_peer_output_idle()

    assert applied.disposition == "sealed"
    assert applied.requested_transition_sample == 768
    assert applied.actual_applied_sample == 1536
    assert duplicate.disposition == "duplicate"
    assert already.disposition == "already_separated"
    assert late.disposition == "too_late_for_current_scope"
    assert invalid.disposition == "invalid_source"
    assert invalid_reference.disposition == "invalid_reference"
    assert retracted.disposition == "retracted"
    assert ledger.terminal_receipts[0].segment.seal_reason == "prospective_speaker_transition"
    assert second_seal.disposition == "sealed"
    assert [
        getattr(event, "text")
        for event in overlay.events
        if getattr(event, "type", None) == "peer_transcript_final"
    ] == ["first", "second"]
    assert [receipt.outcome for receipt in ledger.terminal_receipts] == ["final", "final"]
    await owner.apply_intent(make_config(), enabled=False)
    assert not harness.output_runtime.peer_publication_is_authorized(
        identities[0].activation_generation,
        2,
    )
    replacement = await owner.apply_intent(make_config(), enabled=True)
    assert replacement.generation != identities[0].activation_generation
    assert harness.output_runtime.peer_publication_is_authorized(replacement.generation, 1)
    await callbacks.peer_event_handler(terminal_b)
    await harness.output_runtime.wait_for_peer_output_idle()
    assert (
        sum(getattr(event, "type", None) == "peer_transcript_final" for event in overlay.events)
        == 2
    )
    await harness.stop()
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
    old_ledger = owner.segment_ledger
    assert old_ledger is not None
    old_segment_id = uuid4()
    old_owned = old_ledger.observe_vad_event(
        SpeechStart(
            old_segment_id,
            pre_roll=np.empty((0,), dtype=np.float32),
            chunk=np.ones((8,), dtype=np.float32),
            chunk_capture=(
                AudioCaptureSpan(
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
                ),
            ),
        ),
        now_monotonic_s=0.0,
    )
    old_terminal = provider.terminal_handlers[-1]
    await owner.apply_intent(second, enabled=True)
    assert [receipt.outcome for receipt in old_ledger.terminal_receipts] == ["cancelled"]
    assert old_ledger.terminal_receipts[0].identity.segment_id == old_segment_id
    await wait_until(lambda: len(captured_sinks) == 2)

    await captured_sinks[0].handle_owned_vad_event(old_owned)
    await old_terminal(RuntimeError("late"))
    current_ledger = owner.segment_ledger
    assert current_ledger is not None
    current_owned = current_ledger.observe_vad_event(
        SpeechStart(
            uuid4(),
            pre_roll=np.empty((0,), dtype=np.float32),
            chunk=np.empty((0,), dtype=np.float32),
        ),
        now_monotonic_s=1.0,
    )
    await captured_sinks[1].handle_owned_vad_event(current_owned)

    assert sink.events == [current_owned]
    assert owner.snapshot.state is PeerCaptureSessionState.RUNNING
    await owner.close()
    assert [receipt.outcome for receipt in old_ledger.terminal_receipts] == ["cancelled"]


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
    assert len(sources) == 1
    assert sources[0].close_calls == 1
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
