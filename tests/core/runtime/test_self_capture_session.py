from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import replace
from typing import Literal

import pytest

from puripuly_heart.core.runtime.self_capture import SelfCaptureSessionOwner
from puripuly_heart.core.self_capture import (
    SelfCaptureAdmission,
    SelfCaptureAdmissionStatus,
    SelfCaptureDiagnostic,
    SelfCaptureFailureReason,
    SelfCaptureProviderMutation,
    SelfCaptureProviderMutationStatus,
    SelfCaptureProviderStatus,
    SelfCaptureSessionConfig,
    SelfCaptureSessionState,
)


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
) -> SelfCaptureSessionConfig:
    return SelfCaptureSessionConfig(
        provider_id=f"provider-{suffix}",
        provider_signature=("provider", suffix),
        runtime_signature=("runtime", suffix),
        capture_signature=capture_signature,
        target_sample_rate_hz=16000,
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
    owner, _, _, sources, _, _ = build_owner(provider=provider)
    first = config("one")
    second = config("two")

    await owner.apply_intent(first, enabled=True)
    first_task = owner.loop_task
    provider.handoff_result = SelfCaptureProviderMutation(mutation_status)

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
@pytest.mark.parametrize("transition", ["no_op", "reconfigure"])
async def test_provider_terminal_failure_survives_same_provider_generation_changes(
    transition: str,
) -> None:
    owner, _, provider, sources, _, _ = build_owner()
    session_config = config()

    await owner.apply_intent(session_config, enabled=True)
    handler = provider.terminal_failure_handler
    assert handler is not None
    next_config = session_config
    if transition == "reconfigure":
        next_config = replace(
            session_config,
            runtime_signature=("runtime", "reconfigured"),
            session_options=("options", "reconfigured"),
        )
    await owner.apply_intent(next_config, enabled=True)
    await handler(RuntimeError("same provider terminal failure"))

    assert owner.snapshot.state is SelfCaptureSessionState.FAULTED
    assert owner.snapshot.failure_reason is SelfCaptureFailureReason.PROVIDER_FAILED
    assert sources[0].close_calls == 1
    assert provider.release_calls == [("abort", None)]
    assert provider.reconfigure_calls == (
        [("options", "reconfigured")] if transition == "reconfigure" else []
    )


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
async def test_provider_ingress_start_failure_completes_and_releases_owned_resources() -> None:
    provider = RecordingProvider()
    provider.start_failure = RuntimeError("provider ingress failed")
    owner, _, _, sources, _, _ = build_owner(provider=provider)

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
