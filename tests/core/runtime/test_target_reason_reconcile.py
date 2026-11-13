from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from puripuly_heart.core.clock import FakeClock
from puripuly_heart.core.peer_capture import (
    PeerCaptureAdmission,
    PeerCaptureAdmissionStatus,
    PeerCaptureLanguageFacts,
    PeerCaptureProviderMutation,
    PeerCaptureProviderMutationStatus,
    PeerCaptureResolvedTarget,
    PeerCaptureSessionConfig,
    PeerCaptureSessionState,
    PeerCaptureTargetIntent,
    PeerCaptureTargetResolution,
    PeerCaptureTargetStatus,
)
from puripuly_heart.core.runtime.peer_channel import PeerCaptureSessionOwner


@dataclass(slots=True)
class FakeSource:
    close_calls: int = 0
    terminal_reason: str | None = None

    async def close(self) -> None:
        self.close_calls += 1


class FakeAdmission:
    def __init__(self) -> None:
        self.result = PeerCaptureAdmission(PeerCaptureAdmissionStatus.ADMITTED)
        self.calls: list[PeerCaptureSessionConfig] = []

    async def admit(self, config: PeerCaptureSessionConfig) -> PeerCaptureAdmission:
        self.calls.append(config)
        return self.result


class FakeTargetResolver:
    def __init__(self) -> None:
        self.calls: list[PeerCaptureTargetIntent] = []
        self.results: list[PeerCaptureTargetResolution] = []

    async def resolve(
        self,
        target: PeerCaptureTargetIntent,
    ) -> PeerCaptureTargetResolution:
        self.calls.append(target)
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
        self.releases: list[tuple[str, float | None]] = []
        self.start_calls = 0
        self.terminal_handlers: list = []

    def is_ready(self, config: PeerCaptureSessionConfig) -> bool:
        return self.provider_id == config.provider_id

    async def replace(self, request, *, start: bool, on_terminal_failure):
        _ = start
        self.requests.append(request)
        self.terminal_handlers.append(on_terminal_failure)
        if self.replace_result_applied():
            self.provider_id = request[0]
        return PeerCaptureProviderMutation(PeerCaptureProviderMutationStatus.APPLIED)

    def replace_result_applied(self) -> bool:
        return True

    async def handoff(self, request, *, start: bool, on_terminal_failure):
        _ = start
        self.terminal_handlers.append(on_terminal_failure)
        return PeerCaptureProviderMutation(PeerCaptureProviderMutationStatus.APPLIED)

    async def cancel_handoff(self) -> bool:
        return True

    async def start_ingress(self) -> None:
        self.start_calls += 1

    async def warmup(self) -> None:
        return None

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

    async def handle_owned_vad_event(self, event: object) -> None:
        self.events.append(event)


def make_config(
    *,
    provider_id: str = "soniox",
    target: PeerCaptureTargetIntent | None = None,
) -> PeerCaptureSessionConfig:
    resolved_target = target or PeerCaptureTargetIntent(kind="default_output_device")
    language_facts = PeerCaptureLanguageFacts(
        source_mode="manual",
        source_language="ko",
    )
    signature = (resolved_target, 16000)
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
        release_backend_after=600.0 if provider_id == "local_fun_asr_mlt_nano" else None,
    )


def make_owner(
    *,
    resolver: FakeTargetResolver | None = None,
) -> tuple[PeerCaptureSessionOwner, FakeTargetResolver, FakeProvider]:
    admission_port = FakeAdmission()
    resolver_port = resolver or FakeTargetResolver()
    provider_port = FakeProvider()

    async def default_loop(**_kwargs) -> None:
        await asyncio.Event().wait()

    owner = PeerCaptureSessionOwner(
        admission=admission_port,
        target_resolver=resolver_port,
        provider=provider_port,
        clock=FakeClock(),
        provider_request_factory=lambda config, warmup: (config.provider_id, warmup),
        source_factory=lambda _config, _target: FakeSource(),
        vad_factory=lambda _config: object(),
        run_audio_loop=default_loop,
        vad_sink=FakeVadSink(),
        smart_turn_owner=None,
    )
    return owner, resolver_port, provider_port


def process_target() -> PeerCaptureTargetIntent:
    return PeerCaptureTargetIntent(
        kind="process",
        process_kind="generic_executable",
        executable_identity=r"c:\apps\game\game.exe",
    )


def unavailable_then_resolved(target: PeerCaptureTargetIntent) -> FakeTargetResolver:
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
    return resolver


@pytest.mark.asyncio
async def test_successful_activation_reconciles_target_reason() -> None:
    target = process_target()
    owner, resolver, _provider = make_owner(resolver=unavailable_then_resolved(target))
    config = make_config(target=target)

    failed = await owner.apply_intent(config, enabled=True)
    assert failed.state is PeerCaptureSessionState.FAULTED
    assert failed.failure_reason.value == "target_unavailable"
    assert failed.target_reason == "no_process"

    assert await owner.retry_process_capture() is True

    healthy = owner.snapshot
    assert healthy.state is PeerCaptureSessionState.RUNNING
    assert healthy.failure_reason is None
    assert healthy.retry_available is False
    assert healthy.target_reason is None, (
        "retained target-unavailable reason survived successful activation; "
        "docs/architecture.md: successful activation reconciles current failure "
        "state independently of retained diagnostic history"
    )
    await owner.close()


@pytest.mark.asyncio
async def test_unrelated_fault_after_target_recovery_does_not_report_target_reason() -> None:
    target = process_target()
    owner, resolver, provider = make_owner(resolver=unavailable_then_resolved(target))
    config = make_config(target=target)

    failed = await owner.apply_intent(config, enabled=True)
    assert failed.target_reason == "no_process"
    assert await owner.retry_process_capture() is True
    assert owner.snapshot.state is PeerCaptureSessionState.RUNNING

    await provider.terminal_handlers[-1](RuntimeError("terminal"))

    faulted = owner.snapshot
    assert faulted.state is PeerCaptureSessionState.FAULTED
    assert faulted.failure_reason.value == "provider_failed"
    assert faulted.target_reason is None, (
        "unrelated provider failure after target recovery carries the stale "
        "target-unavailable reason; consumers classify it as "
        "action_required/select_available_capture_target"
    )
    await owner.close()
