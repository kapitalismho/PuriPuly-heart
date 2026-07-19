from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from pathlib import Path

from puripuly_heart.config.process_capture_resolution import ProcessCaptureTargetUnavailableError
from puripuly_heart.config.resolved import (
    CREDENTIAL_SOURCE_SECRET_STORE,
    ResolvedCredentialRequirement,
    ResolvedDesktopAudioCaptureTarget,
    ResolvedSTTConfig,
)
from puripuly_heart.config.settings import STTProviderName
from puripuly_heart.core.clock import FakeClock
from puripuly_heart.core.local_asr_provider_runtime import (
    LocalASRProviderRuntimeSnapshot,
    ProviderRuntimeBuildRequest,
    ProviderRuntimeChannelSnapshot,
    ProviderRuntimeGpuSnapshot,
    ProviderRuntimeMutationResult,
)
from puripuly_heart.core.runtime.local_asr_transition import LocalASRSessionOptions
from puripuly_heart.core.runtime.peer_channel import (
    PeerChannelRuntime,
    PeerChannelRuntimeState,
    PeerRuntimeConfig,
    PeerRuntimeFailureReason,
)


@dataclass(slots=True)
class DummySource:
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


async def fake_run_audio_loop(**_kwargs) -> None:
    await asyncio.Event().wait()


async def wait_until(predicate, *, timeout_s: float = 1.0) -> None:  # noqa: ANN001
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while not predicate():
        if loop.time() >= deadline:
            raise AssertionError("timed out waiting for condition")
        await asyncio.sleep(0)


def make_peer_runtime_config(output_device: str = "Headphones (Loopback)") -> PeerRuntimeConfig:
    backend = ResolvedSTTConfig(
        channel="peer",
        provider=STTProviderName.DEEPGRAM.value,
        source_language="ko",
        model="nova-3",
        endpoint=None,
        region=None,
        credential=ResolvedCredentialRequirement(
            source=CREDENTIAL_SOURCE_SECRET_STORE,
            required=True,
            reference="deepgram:stt",
        ),
        input_host_api=None,
        input_device=None,
        output_device=output_device,
        sample_rate_hz=16000,
        channels=1,
        ring_buffer_ms=500,
        drain_timeout_s=2.0,
        vad_speech_threshold=0.5,
        vad_hangover_ms=900,
        vad_pre_roll_ms=500,
        low_latency_enabled=True,
        low_latency_merge_gap_ms=600,
        low_latency_spec_retry_max=10,
        custom_vocabulary_enabled=True,
        custom_terms={"ko": ("아이리", "시나노")},
        provider_options={},
    )
    provider_signature = (
        backend.provider,
        backend.source_language,
        backend.model,
        backend.custom_terms,
    )
    return PeerRuntimeConfig(
        backend=backend,
        output_device=output_device,
        vad_threshold=0.6,
        vad_hangover_ms=900,
        vad_pre_roll_ms=500,
        provider_signature=provider_signature,
        runtime_signature=(
            backend.source_language,
            output_device,
            0.6,
            900,
            500,
            provider_signature,
        ),
        model_id=backend.model,
        capture_vad_signature=(output_device, 0.6, 900, 500),
    )


def make_process_peer_runtime_config() -> PeerRuntimeConfig:
    config = make_peer_runtime_config()
    target = ResolvedDesktopAudioCaptureTarget(
        kind="process",
        process_kind="generic_executable",
        executable_identity=r"c:\apps\game\game.exe",
    )
    return replace(
        config,
        output_device="",
        runtime_signature=("process", target),
        capture_target=target,
        capture_vad_signature=("process", target),
    )


def make_local_qwen_runtime_config(*, model: str = "local") -> PeerRuntimeConfig:
    config = make_peer_runtime_config()
    backend = replace(
        config.backend,
        provider=STTProviderName.LOCAL_QWEN.value,
        model=model,
        credential=replace(config.backend.credential, required=False, reference=None),
    )
    provider_signature = (backend.provider, backend.source_language, backend.model)
    return replace(
        config,
        backend=backend,
        provider_signature=provider_signature,
        runtime_signature=(config.output_device, provider_signature),
        model_id=model,
    )


class FakePeerOwner:
    def __init__(self) -> None:
        self.releases: list[tuple[str, str, float | None]] = []
        self.reconfigurations: list[LocalASRSessionOptions] = []
        self.warmups = 0
        self._peer = ProviderRuntimeChannelSnapshot(
            channel="peer",
            provider_id=None,
            model_id=None,
            phase="inactive",
            generation=0,
            pending_handoff=False,
            has_resources=False,
        )

    @property
    def snapshot(self) -> LocalASRProviderRuntimeSnapshot:
        return LocalASRProviderRuntimeSnapshot(
            channels=(
                ProviderRuntimeChannelSnapshot(
                    channel="self",
                    provider_id=None,
                    model_id=None,
                    phase="inactive",
                    generation=0,
                    pending_handoff=False,
                    has_resources=False,
                ),
                self._peer,
            ),
            gpu=ProviderRuntimeGpuSnapshot(
                phase="inactive",
                devices=(),
                active_channels=frozenset(),
                pending_count=0,
                worker_pid=None,
                configured_device_id=None,
                model_resident=False,
                retry_required=False,
                failure_code=None,
            ),
        )

    def attach(self, request: ProviderRuntimeBuildRequest, *, start: bool) -> None:
        self._peer = replace(
            self._peer,
            provider_id=request.provider_id,
            model_id=request.model_id,
            phase="running" if start else "dormant",
            generation=self._peer.generation + 1,
            has_resources=True,
        )

    async def release_channel(
        self,
        channel: str,
        *,
        mode: str,
        release_backend_after: float | None = None,
    ) -> None:
        self.releases.append((channel, mode, release_backend_after))
        if mode == "abort":
            self._peer = replace(
                self._peer,
                provider_id=None,
                model_id=None,
                phase="inactive",
                generation=self._peer.generation + 1,
                has_resources=False,
            )


class FakeOwnedPeerHub:
    def __init__(self) -> None:
        self.local_asr_provider_runtime = FakePeerOwner()
        self.requests: list[ProviderRuntimeBuildRequest] = []
        self.handoffs: list[ProviderRuntimeBuildRequest] = []
        self.start_calls = 0
        self.vad_events: list[object] = []
        self.terminal_failure = None

    async def replace_peer_stt_provider_request(
        self,
        request: ProviderRuntimeBuildRequest,
        *,
        start: bool,
        on_terminal_failure,
    ) -> ProviderRuntimeMutationResult:
        self.requests.append(request)
        self.terminal_failure = on_terminal_failure
        self.local_asr_provider_runtime.attach(request, start=start)
        return ProviderRuntimeMutationResult(
            status="applied",
            request=request,
            previous_provider_id=None,
            snapshot=self.local_asr_provider_runtime.snapshot,
        )

    async def handoff_peer_stt_provider_request(
        self,
        request: ProviderRuntimeBuildRequest,
        *,
        start: bool,
        on_terminal_failure,
    ) -> ProviderRuntimeMutationResult:
        self.handoffs.append(request)
        self.terminal_failure = on_terminal_failure
        previous = self.local_asr_provider_runtime.snapshot.channel_for("peer").provider_id
        self.local_asr_provider_runtime.attach(request, start=start)
        return ProviderRuntimeMutationResult(
            status="applied",
            request=request,
            previous_provider_id=previous,
            snapshot=self.local_asr_provider_runtime.snapshot,
        )

    async def cancel_peer_stt_provider_request_handoff(self) -> bool:
        return True

    async def start_peer_stt_provider_ingress(self) -> None:
        self.start_calls += 1
        self.local_asr_provider_runtime._peer = replace(
            self.local_asr_provider_runtime._peer,
            phase="running",
        )

    async def abort_peer_stt_for_toggle_off(self) -> None:
        await self.local_asr_provider_runtime.release_channel("peer", mode="abort")

    async def warmup_stt_channel(self, channel: str) -> None:
        _ = channel
        self.local_asr_provider_runtime.warmups += 1

    async def reconfigure_stt_channel(
        self,
        channel: str,
        options: LocalASRSessionOptions,
    ) -> None:
        _ = channel
        self.local_asr_provider_runtime.reconfigurations.append(options)

    async def handle_peer_vad_event(self, event: object) -> None:
        self.vad_events.append(event)


def make_runtime(
    hub: FakeOwnedPeerHub,
    *,
    source_factory,
    run_audio_loop=fake_run_audio_loop,
    diagnostics=None,
    idle_release_seconds: float | None = None,
) -> PeerChannelRuntime:
    return PeerChannelRuntime(
        hub=hub,
        clock=FakeClock(),
        provider_request_factory=lambda config, warmup: ProviderRuntimeBuildRequest(
            config=config.backend,
            warmup=warmup,
            model_id=config.model_id or config.backend.model,
            session_options=config.session_options,
        ),
        source_factory=source_factory,
        vad_factory=lambda _config, _model_path: "peer-vad",
        vad_model_resolver=lambda: Path("vad.onnx"),
        run_audio_loop=run_audio_loop,
        diagnostic_sink=diagnostics,
        idle_release_seconds=idle_release_seconds,
    )


async def test_start_and_release_delegate_provider_lifecycle_to_owner() -> None:
    hub = FakeOwnedPeerHub()
    sources: list[DummySource] = []
    runtime = make_runtime(
        hub,
        source_factory=lambda _config: sources.append(DummySource()) or sources[-1],
    )
    config = make_peer_runtime_config()

    await runtime.apply_policy(config=config, desired_active=True)
    await runtime.apply_policy(config=config, desired_active=False, stop_mode="release")

    assert runtime.state is PeerChannelRuntimeState.STOPPED
    assert [request.config for request in hub.requests] == [config.backend]
    assert sources[0].close_calls == 1
    assert hub.local_asr_provider_runtime.releases == [("peer", "abort", None)]


async def test_local_qwen_reuses_owner_retained_provider() -> None:
    hub = FakeOwnedPeerHub()
    sources: list[DummySource] = []
    runtime = make_runtime(
        hub,
        source_factory=lambda _config: sources.append(DummySource()) or sources[-1],
        idle_release_seconds=600.0,
    )
    config = make_local_qwen_runtime_config()

    await runtime.apply_policy(config=config, desired_active=True)
    await runtime.apply_policy(config=config, desired_active=False, stop_mode="retain")
    await runtime.apply_policy(config=config, desired_active=True)

    assert len(hub.requests) == 1
    assert hub.start_calls == 2
    assert hub.local_asr_provider_runtime.releases == [("peer", "drain", 600.0)]
    assert [source.close_calls for source in sources] == [1, 0]

    await runtime.close()


async def test_capture_change_restarts_source_through_owner_request() -> None:
    hub = FakeOwnedPeerHub()
    sources: list[DummySource] = []
    runtime = make_runtime(
        hub,
        source_factory=lambda _config: sources.append(DummySource()) or sources[-1],
    )
    first = make_peer_runtime_config("first")
    second = make_peer_runtime_config("second")

    await runtime.apply_policy(config=first, desired_active=True)
    await runtime.apply_policy(config=second, desired_active=True)

    assert len(hub.requests) == 1
    assert sources[0].close_calls == 1
    assert runtime.current_signature == second.runtime_signature

    await runtime.close()


async def test_same_local_model_reconfigures_without_replacing_capture() -> None:
    hub = FakeOwnedPeerHub()
    source = DummySource()
    runtime = make_runtime(hub, source_factory=lambda _config: source)
    first = make_local_qwen_runtime_config()
    options = LocalASRSessionOptions(source_language="ja")
    second = replace(
        first,
        session_options=options,
        provider_signature=(*first.provider_signature, "ja"),
        runtime_signature=(*first.runtime_signature, "ja"),
    )

    await runtime.apply_policy(config=first, desired_active=True)
    await runtime.apply_policy(config=second, desired_active=True)

    assert len(hub.requests) == 1
    assert hub.handoffs == []
    assert hub.local_asr_provider_runtime.reconfigurations == [options]
    assert source.close_calls == 0

    await runtime.close()


async def test_local_model_change_requests_owner_handoff() -> None:
    hub = FakeOwnedPeerHub()
    source = DummySource()
    runtime = make_runtime(hub, source_factory=lambda _config: source)
    first = make_local_qwen_runtime_config(model="first")
    second = replace(
        make_local_qwen_runtime_config(model="second"),
        capture_vad_signature=first.capture_vad_signature,
    )

    await runtime.apply_policy(config=first, desired_active=True)
    await runtime.apply_policy(config=second, desired_active=True)

    assert [request.model_id for request in hub.handoffs] == ["second"]
    assert source.close_calls == 0

    await runtime.close()


async def test_stale_capture_sink_cannot_publish_peer_event() -> None:
    hub = FakeOwnedPeerHub()
    sinks: list[object] = []

    async def capture_loop(**kwargs) -> None:
        sinks.append(kwargs["sink"])
        await asyncio.Event().wait()

    sources: list[DummySource] = []
    runtime = make_runtime(
        hub,
        source_factory=lambda _config: sources.append(DummySource()) or sources[-1],
        run_audio_loop=capture_loop,
    )
    first = make_peer_runtime_config("first")
    second = make_peer_runtime_config("second")

    await runtime.apply_policy(config=first, desired_active=True)
    await wait_until(lambda: len(sinks) == 1)
    stale_sink = sinks[0]
    await runtime.apply_policy(config=second, desired_active=True)
    await stale_sink.handle_vad_event("stale")

    assert hub.vad_events == []

    await runtime.close()


async def test_process_target_unavailable_faults_and_can_retry() -> None:
    hub = FakeOwnedPeerHub()
    diagnostics = []
    attempts = 0
    source = DummySource()

    def source_factory(_config):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ProcessCaptureTargetUnavailableError("no_process")
        return source

    runtime = make_runtime(hub, source_factory=source_factory, diagnostics=diagnostics.append)
    config = make_process_peer_runtime_config()

    await runtime.apply_policy(config=config, desired_active=True)

    assert runtime.state is PeerChannelRuntimeState.FAULTED
    assert diagnostics[-1].reason is PeerRuntimeFailureReason.PROCESS_TARGET_UNAVAILABLE
    assert diagnostics[-1].process_unavailable_reason == "no_process"

    assert await runtime.retry_process_capture(config=config) is True
    assert runtime.state is PeerChannelRuntimeState.RUNNING

    await runtime.close()


async def test_capture_loop_failure_faults_and_releases_owner_channel() -> None:
    hub = FakeOwnedPeerHub()
    diagnostics = []

    async def failing_loop(**_kwargs) -> None:
        raise RuntimeError("capture failed")

    runtime = make_runtime(
        hub,
        source_factory=lambda _config: DummySource(),
        run_audio_loop=failing_loop,
        diagnostics=diagnostics.append,
    )

    await runtime.apply_policy(config=make_peer_runtime_config(), desired_active=True)
    await wait_until(lambda: bool(diagnostics))

    assert diagnostics[-1].reason is PeerRuntimeFailureReason.PEER_RUNTIME_FAILED
    assert hub.local_asr_provider_runtime.releases[-1] == ("peer", "abort", None)


async def test_terminal_process_source_faults_with_target_exited_reason() -> None:
    hub = FakeOwnedPeerHub()
    diagnostics = []
    source = DummySource(terminal_reason="target_exited")

    async def completed_loop(**_kwargs) -> None:
        return

    runtime = make_runtime(
        hub,
        source_factory=lambda _config: source,
        run_audio_loop=completed_loop,
        diagnostics=diagnostics.append,
    )

    await runtime.apply_policy(config=make_process_peer_runtime_config(), desired_active=True)
    await wait_until(lambda: bool(diagnostics))

    assert diagnostics[-1].reason is PeerRuntimeFailureReason.PROCESS_TARGET_EXITED


async def test_terminal_provider_failure_faults_current_generation() -> None:
    hub = FakeOwnedPeerHub()
    diagnostics = []
    runtime = make_runtime(
        hub,
        source_factory=lambda _config: DummySource(),
        diagnostics=diagnostics.append,
    )
    config = make_process_peer_runtime_config()

    await runtime.apply_policy(config=config, desired_active=True)
    await hub.terminal_failure(RuntimeError("provider failed"))

    assert runtime.state is PeerChannelRuntimeState.FAULTED
    assert diagnostics[-1].reason is PeerRuntimeFailureReason.PROCESS_PROVIDER_FAILED


async def test_close_retries_source_cleanup_debt_and_releases_owner() -> None:
    hub = FakeOwnedPeerHub()
    source = FailingCloseSource()
    runtime = make_runtime(hub, source_factory=lambda _config: source)

    await runtime.apply_policy(config=make_peer_runtime_config(), desired_active=True)

    try:
        await runtime.apply_policy(
            config=make_peer_runtime_config(),
            desired_active=False,
            stop_mode="release",
        )
    except RuntimeError as exc:
        assert str(exc) == "close failed"

    await runtime.close()

    assert source.close_calls == 2
    assert runtime.state is PeerChannelRuntimeState.STOPPED


async def test_diagnostic_sink_failure_does_not_break_fault_teardown() -> None:
    hub = FakeOwnedPeerHub()

    async def failing_loop(**_kwargs) -> None:
        raise RuntimeError("capture failed")

    def failing_sink(_diagnostic) -> None:
        raise RuntimeError("sink failed")

    runtime = make_runtime(
        hub,
        source_factory=lambda _config: DummySource(),
        run_audio_loop=failing_loop,
        diagnostics=failing_sink,
    )

    await runtime.apply_policy(config=make_peer_runtime_config(), desired_active=True)
    await wait_until(lambda: runtime.state is PeerChannelRuntimeState.FAULTED)

    assert hub.local_asr_provider_runtime.releases[-1] == ("peer", "abort", None)
