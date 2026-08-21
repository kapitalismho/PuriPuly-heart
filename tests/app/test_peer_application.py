import asyncio
from dataclasses import dataclass, field

import pytest

from puripuly_heart.app.services.peer_application import (
    PeerApplicationOwner,
    PeerApplicationState,
)
from puripuly_heart.app.wiring import build_peer_capture_session_config
from puripuly_heart.config.settings import AppSettings, STTProviderName
from puripuly_heart.core.peer_capture import PeerCaptureProviderStatus
from puripuly_heart.ui.overlay_peer_contract import build_overlay_peer_consumer_contract


@dataclass(frozen=True)
class RuntimeSnapshot:
    effective_active: bool
    provider_status: PeerCaptureProviderStatus = PeerCaptureProviderStatus.READY


@dataclass
class Runtime:
    retry_result: bool = False
    retry_error: BaseException | None = None
    current_signature: object | None = None
    last_local_asr_transition_status: str = "idle"
    policy_calls: list[tuple[object, bool, str]] = field(default_factory=list)
    retry_configs: list[object] = field(default_factory=list)
    close_error: BaseException | None = None
    close_calls: int = 0
    close_entered: asyncio.Event | None = None
    close_release: asyncio.Event | None = None
    effective_active: bool = True
    prepare_calls: list[object] = field(default_factory=list)
    prepare_entered: asyncio.Event | None = None
    prepare_release: asyncio.Event | None = None
    provider_status: PeerCaptureProviderStatus = PeerCaptureProviderStatus.READY

    @property
    def snapshot(self) -> RuntimeSnapshot:
        return RuntimeSnapshot(
            effective_active=self.effective_active,
            provider_status=self.provider_status,
        )

    async def prepare_provider(self, config: object) -> RuntimeSnapshot:
        self.prepare_calls.append(config)
        if self.prepare_entered is not None:
            self.prepare_entered.set()
        if self.prepare_release is not None:
            await self.prepare_release.wait()
        self.current_signature = getattr(config, "runtime_signature", None)
        return self.snapshot

    async def apply_policy(
        self,
        *,
        config: object,
        desired_active: bool,
        stop_mode: str = "retain",
    ) -> None:
        self.policy_calls.append((config, desired_active, stop_mode))

    async def retry_process_capture(self, *, config: object) -> bool:
        self.retry_configs.append(config)
        if self.retry_error is not None:
            raise self.retry_error
        return self.retry_result

    async def close(self) -> None:
        self.close_calls += 1
        if self.close_entered is not None:
            self.close_entered.set()
        if self.close_release is not None:
            await self.close_release.wait()
        if self.close_error is not None:
            raise self.close_error


@dataclass
class Harness:
    settings: AppSettings = field(default_factory=AppSettings)
    overlay_state: str = "connected"
    overlay_command_available: bool = True
    runtime_available: bool = True
    provider_available: bool = True
    ready: bool = True
    ingress_frozen: bool = False
    events: list[object] = field(default_factory=list)
    translation_demands: list[bool] = field(default_factory=list)
    effective: list[tuple[bool, bool]] = field(default_factory=list)
    lifecycle_traces: list[tuple[str, dict[str, object]]] = field(default_factory=list)
    readiness_entered: asyncio.Event | None = None
    readiness_release: asyncio.Event | None = None

    def state(self) -> PeerApplicationState:
        return PeerApplicationState(
            settings_available=True,
            peer_intent_enabled=self.settings.ui.peer_translation_enabled,
            eula_accepted=self.settings.ui.peer_translation_eula_accepted,
            overlay_intent_enabled=self.settings.ui.overlay_enabled,
            peer_provider_id=self.settings.provider.peer_stt.value,
            runtime_available=self.runtime_available,
            peer_provider_available=self.provider_available,
            overlay_state=self.overlay_state,
            overlay_command_available=self.overlay_command_available,
            ingress_frozen=self.ingress_frozen,
        )

    async def ensure_ready(self, generation: int) -> bool:
        self.events.append(("ready", generation))
        if self.readiness_entered is not None:
            self.readiness_entered.set()
        if self.readiness_release is not None:
            await self.readiness_release.wait()
        return self.ready

    async def begin_overlay(self) -> None:
        self.events.append("overlay_start")

    def owner(self) -> PeerApplicationOwner:
        return PeerApplicationOwner(
            state_provider=self.state,
            config_factory=lambda: build_peer_capture_session_config(self.settings),
            peer_intent_sink=lambda enabled: setattr(
                self.settings.ui,
                "peer_translation_enabled",
                enabled,
            ),
            overlay_intent_sink=lambda enabled: setattr(
                self.settings.ui,
                "overlay_enabled",
                enabled,
            ),
            persist_manual_fallback=lambda: True,
            ensure_local_ready=self.ensure_ready,
            clear_cpu_pending=lambda: self.events.append("clear_cpu"),
            clear_gpu_pending=lambda: self.events.append("clear_gpu"),
            clear_switched_pending=lambda: self.events.append("clear_switched"),
            sync_local_notice=lambda: self.events.append("notice"),
            presentation_changed=lambda: self.events.append("presentation"),
            begin_overlay_start=self.begin_overlay,
            effective_sink=lambda peer, context: self.effective.append((peer, context)),
            disclosure_sink=lambda: self.events.append("disclosure"),
            superseded_sink=lambda: self.events.append("superseded"),
            log_basic=lambda message: self.events.append(("basic", message)),
            log_detailed=lambda message: self.events.append(("detail", message)),
            log_failure=lambda message: self.events.append(("failure", message)),
            lifecycle_trace_sink=lambda event, fields: self.lifecycle_traces.append(
                (event, fields)
            ),
            translation_demand_sink=self._notify_translation_demand,
        )

    async def _notify_translation_demand(self) -> None:
        self.events.append("demand")
        self.translation_demands.append(self.settings.ui.peer_translation_enabled)


@pytest.mark.asyncio
async def test_peer_owner_preserves_eula_and_effective_activation_contract() -> None:
    harness = Harness()
    owner = harness.owner()

    await owner.set_enabled(True)

    assert harness.settings.ui.peer_translation_enabled is False
    assert harness.effective[-1] == (False, False)
    assert "overlay_start" not in harness.events

    harness.settings.ui.peer_translation_eula_accepted = True
    harness.overlay_state = "off"
    await owner.set_enabled(True)

    assert harness.settings.ui.peer_translation_enabled is True
    assert harness.settings.ui.overlay_enabled is True
    assert "overlay_start" in harness.events
    assert owner.snapshot().activation_requested is True
    assert harness.translation_demands == [False, True]


@pytest.mark.asyncio
async def test_peer_owner_notifies_translation_demand_before_local_ready() -> None:
    harness = Harness(
        readiness_entered=asyncio.Event(),
        readiness_release=asyncio.Event(),
    )
    harness.settings.ui.peer_translation_eula_accepted = True
    owner = harness.owner()

    enabling = asyncio.create_task(owner.set_enabled(True))
    await harness.readiness_entered.wait()

    assert harness.translation_demands == [True]
    assert harness.settings.ui.peer_translation_enabled is True

    harness.readiness_release.set()
    await enabling


@pytest.mark.asyncio
async def test_peer_enable_starts_overlay_before_waiting_for_local_ready() -> None:
    harness = Harness(
        overlay_state="off",
        readiness_entered=asyncio.Event(),
        readiness_release=asyncio.Event(),
    )
    harness.settings.ui.peer_translation_eula_accepted = True
    owner = harness.owner()

    enabling = asyncio.create_task(owner.set_enabled(True))
    await harness.readiness_entered.wait()

    assert harness.settings.ui.overlay_enabled is True
    assert "overlay_start" in harness.events
    assert harness.events.index("overlay_start") < harness.events.index("demand")
    assert harness.events.index("demand") < harness.events.index(("ready", 1))

    harness.readiness_release.set()
    await enabling


@pytest.mark.asyncio
async def test_peer_enable_starts_overlay_before_waiting_for_translation_demand() -> None:
    harness = Harness(overlay_state="off")
    harness.settings.ui.peer_translation_eula_accepted = True
    demand_entered = asyncio.Event()
    demand_release = asyncio.Event()

    async def slow_demand() -> None:
        harness.events.append("demand")
        harness.translation_demands.append(harness.settings.ui.peer_translation_enabled)
        demand_entered.set()
        await demand_release.wait()

    harness._notify_translation_demand = slow_demand
    owner = harness.owner()

    enabling = asyncio.create_task(owner.set_enabled(True))
    await demand_entered.wait()

    assert harness.settings.ui.overlay_enabled is True
    assert "overlay_start" in harness.events
    assert harness.events.index("overlay_start") < harness.events.index("demand")

    demand_release.set()
    await enabling


@pytest.mark.asyncio
async def test_peer_owner_rejects_post_readiness_completion_after_newer_intent() -> None:
    harness = Harness(
        readiness_entered=asyncio.Event(),
        readiness_release=asyncio.Event(),
    )
    harness.settings.provider.peer_stt = STTProviderName.SONIOX
    harness.settings.ui.peer_translation_eula_accepted = True
    owner = harness.owner()

    enabling = asyncio.create_task(owner.set_enabled(True))
    await harness.readiness_entered.wait()
    owner.disable_for_overlay()
    harness.readiness_release.set()
    await enabling

    assert harness.settings.ui.peer_translation_enabled is False
    assert "overlay_start" not in harness.events
    assert "disclosure" not in harness.events


@pytest.mark.asyncio
async def test_peer_owner_retry_rebuilds_config_and_clears_warning_only_on_success() -> None:
    harness = Harness()
    harness.settings.ui.peer_translation_enabled = True
    harness.settings.ui.peer_translation_eula_accepted = True
    owner = harness.owner()
    runtime = Runtime()
    owner.bind_runtime(runtime)
    owner.process_warning_reason = "process_target_exited"

    assert await owner.retry_process_capture() is False
    assert owner.process_warning_reason == "process_target_exited"

    runtime.retry_result = True
    assert await owner.retry_process_capture() is True
    assert owner.process_warning_reason is None
    assert len(runtime.retry_configs) == 2

    for error in (RuntimeError("retry failed"), asyncio.CancelledError()):
        runtime.retry_error = error
        owner.process_warning_reason = "process_target_exited"
        presentation_count = harness.events.count("presentation")
        with pytest.raises(type(error)):
            await owner.retry_process_capture()
        assert owner.process_warning_reason == "process_target_exited"
        assert harness.events.count("presentation") == presentation_count
    runtime.retry_error = None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "close_error",
    [RuntimeError("close failed"), asyncio.CancelledError()],
    ids=["exception", "cancellation"],
)
async def test_peer_owner_runtime_replacement_retains_previous_close_debt(
    close_error: BaseException,
) -> None:
    harness = Harness()
    owner = harness.owner()
    previous = Runtime(close_error=close_error)
    replacement = Runtime()
    owner.bind_runtime(previous)

    with pytest.raises(type(close_error)):
        await owner.replace_runtime(replacement)

    assert owner.runtime is previous
    assert previous.close_calls == 1
    assert replacement.close_calls == 0

    previous.close_error = None
    await owner.close()

    assert previous.close_calls == 2
    assert owner.runtime is None


@pytest.mark.asyncio
async def test_peer_owner_shutdown_race_rejects_and_closes_replacement() -> None:
    harness = Harness()
    owner = harness.owner()
    previous = Runtime(
        close_entered=asyncio.Event(),
        close_release=asyncio.Event(),
    )
    replacement = Runtime()
    owner.bind_runtime(previous)

    replacing = asyncio.create_task(owner.replace_runtime(replacement))
    await previous.close_entered.wait()
    closing = asyncio.create_task(owner.close())
    await asyncio.sleep(0)

    assert closing.done() is False

    previous.close_release.set()
    await replacing
    await closing

    assert previous.close_calls == 1
    assert replacement.close_calls == 1
    assert owner.runtime is None


@pytest.mark.asyncio
async def test_peer_owner_post_close_replacement_is_closed_without_resurrection() -> None:
    harness = Harness()
    owner = harness.owner()
    await owner.close()
    replacement = Runtime()

    await owner.replace_runtime(replacement)

    assert replacement.close_calls == 1
    assert owner.runtime is None


@pytest.mark.asyncio
async def test_peer_owner_retry_revalidates_state_generation_and_runtime_after_readiness() -> None:
    harness = Harness()
    harness.settings.ui.peer_translation_enabled = True
    harness.settings.ui.peer_translation_eula_accepted = True
    owner = harness.owner()
    runtime = Runtime(retry_result=True)
    owner.bind_runtime(runtime)
    config = ["before-ready"]

    async def refresh_config(_generation: int) -> bool:
        config[0] = "after-ready"
        return True

    owner.ensure_local_ready = refresh_config
    owner.config_factory = lambda: config[0]

    assert await owner.retry_process_capture() is True
    assert runtime.retry_configs == ["after-ready"]

    for stale_case in ("ingress", "generation", "runtime"):
        case_harness = Harness(
            readiness_entered=asyncio.Event(),
            readiness_release=asyncio.Event(),
        )
        case_harness.settings.ui.peer_translation_enabled = True
        case_harness.settings.ui.peer_translation_eula_accepted = True
        case_owner = case_harness.owner()
        case_runtime = Runtime(retry_result=True)
        replacement = Runtime(retry_result=True)
        case_owner.bind_runtime(case_runtime)

        retry = asyncio.create_task(case_owner.retry_process_capture())
        await case_harness.readiness_entered.wait()
        if stale_case == "ingress":
            case_harness.ingress_frozen = True
        elif stale_case == "generation":
            case_owner.invalidate_activation()
        else:
            case_owner.bind_runtime(replacement)
        case_harness.readiness_release.set()

        assert await retry is False
        assert case_runtime.retry_configs == []
        assert replacement.retry_configs == []


@pytest.mark.asyncio
async def test_peer_owner_applies_runtime_policy_and_retains_failed_close_debt() -> None:
    harness = Harness()
    harness.settings.ui.peer_translation_enabled = True
    harness.settings.ui.peer_translation_eula_accepted = True
    owner = harness.owner()
    runtime = Runtime(close_error=RuntimeError("close failed"))
    owner.bind_runtime(runtime)

    await owner.refresh_runtime(stop_mode="release")

    assert len(runtime.policy_calls) == 1
    config, desired_active, stop_mode = runtime.policy_calls[0]
    assert config == build_peer_capture_session_config(harness.settings)
    assert desired_active is True
    assert stop_mode == "release"
    assert owner.last_runtime_signature == config.runtime_signature

    with pytest.raises(RuntimeError, match="close failed"):
        await owner.close()
    assert owner.runtime is runtime


def _peer_surface_state(owner: PeerApplicationOwner, *, overlay_state: str) -> str:
    snapshot = owner.snapshot()
    state = owner.state_provider()
    contract = build_overlay_peer_consumer_contract(
        overlay_intent_enabled=True,
        overlay_state=overlay_state,
        overlay_failure_reason=None,
        peer_intent_enabled=state.peer_intent_enabled,
        peer_effective_enabled=snapshot.effective_enabled,
        peer_warning_reason=snapshot.process_warning_reason,
        peer_activation_starting=snapshot.activation_starting or snapshot.model_loading,
    )
    return contract.peer.state


@pytest.mark.asyncio
async def test_peer_activation_starting_survives_overlay_connect_until_capture_effective() -> None:
    harness = Harness(overlay_state="off", provider_available=True)
    harness.settings.ui.peer_translation_eula_accepted = True
    owner = harness.owner()
    runtime = Runtime(effective_active=False)
    owner.bind_runtime(runtime)

    await owner.set_enabled(True)

    assert "overlay_start" in harness.events
    assert owner.snapshot().activation_starting is True
    assert _peer_surface_state(owner, overlay_state="starting") == "starting"

    harness.overlay_state = "connected"
    owner.sync_effective_flags()

    assert owner.snapshot().effective_enabled is False
    assert harness.lifecycle_traces == []
    assert owner.snapshot().activation_starting is True
    assert _peer_surface_state(owner, overlay_state="connected") == "starting"

    runtime.effective_active = True
    owner.sync_effective_flags()

    assert owner.snapshot().effective_enabled is True
    assert owner.snapshot().activation_starting is False
    assert _peer_surface_state(owner, overlay_state="connected") == "on"
    assert harness.lifecycle_traces == [
        ("peer_capture_effective", {"activation_generation": 1, "accepted": True})
    ]
    owner.sync_effective_flags()
    assert len(harness.lifecycle_traces) == 1


@pytest.mark.asyncio
async def test_local_provider_prepares_while_overlay_start_is_in_progress() -> None:
    harness = Harness(overlay_state="off", provider_available=False)
    harness.settings.ui.peer_translation_eula_accepted = True
    owner = harness.owner()
    runtime = Runtime(
        effective_active=False,
        prepare_entered=asyncio.Event(),
        prepare_release=asyncio.Event(),
    )
    owner.bind_runtime(runtime)

    enabling = asyncio.create_task(owner.set_enabled(True))
    await runtime.prepare_entered.wait()

    assert "overlay_start" in harness.events
    assert owner.snapshot().model_loading is True
    harness.overlay_state = "connected"
    await owner.refresh_runtime()
    assert runtime.policy_calls == []

    runtime.prepare_release.set()
    await enabling

    assert len(runtime.prepare_calls) == 1
    assert len(runtime.policy_calls) == 1
    assert runtime.policy_calls[0][1] is True
    assert owner.snapshot().model_loading is False


@pytest.mark.asyncio
async def test_local_peer_attaches_after_overlay_connects_during_local_ready() -> None:
    harness = Harness(
        overlay_state="off",
        provider_available=False,
        readiness_entered=asyncio.Event(),
        readiness_release=asyncio.Event(),
    )
    harness.settings.ui.peer_translation_eula_accepted = True
    owner = harness.owner()
    runtime = Runtime(effective_active=False)
    owner.bind_runtime(runtime)

    enabling = asyncio.create_task(owner.set_enabled(True))
    await harness.readiness_entered.wait()

    assert "overlay_start" in harness.events
    harness.overlay_state = "connected"
    await owner.refresh_runtime()
    assert runtime.policy_calls == []

    harness.readiness_release.set()
    await enabling

    assert len(runtime.prepare_calls) == 1
    assert len(runtime.policy_calls) == 1
    assert runtime.policy_calls[0][1] is True


@pytest.mark.asyncio
async def test_failed_local_provider_preparation_is_not_retried_by_activation() -> None:
    harness = Harness(overlay_state="connected", provider_available=False)
    harness.settings.ui.peer_translation_eula_accepted = True
    owner = harness.owner()
    runtime = Runtime(
        effective_active=False,
        provider_status=PeerCaptureProviderStatus.FAILED,
    )
    owner.bind_runtime(runtime)

    await owner.set_enabled(True)

    assert len(runtime.prepare_calls) == 1
    assert runtime.policy_calls == []
    assert owner.snapshot().activation_starting is False
    assert owner.snapshot().model_loading is False


@pytest.mark.asyncio
async def test_peer_activation_starting_cleared_by_terminal_overlay_or_process_warning() -> None:
    harness = Harness(overlay_state="off", provider_available=False)
    harness.settings.ui.peer_translation_eula_accepted = True
    owner = harness.owner()
    await owner.set_enabled(True)
    assert owner.snapshot().activation_starting is True

    owner.cancel_activation_starting()
    assert owner.snapshot().activation_starting is False
    assert _peer_surface_state(owner, overlay_state="failed") == "warning"

    await owner.set_enabled(True)
    assert owner.snapshot().activation_starting is True

    diagnostic = _process_diagnostic()
    owner.on_runtime_diagnostic(diagnostic)

    assert owner.snapshot().activation_starting is False
    assert owner.process_warning_reason is not None
    assert _peer_surface_state(owner, overlay_state="connected") == "warning"


@pytest.mark.asyncio
async def test_peer_disable_intent_clears_activation_starting() -> None:
    harness = Harness(overlay_state="off", provider_available=False)
    harness.settings.ui.peer_translation_eula_accepted = True
    owner = harness.owner()
    await owner.set_enabled(True)
    assert owner.snapshot().activation_starting is True

    owner.disable_for_overlay()

    assert owner.snapshot().activation_starting is False
    assert _peer_surface_state(owner, overlay_state="off") == "off"


def _process_diagnostic():
    from puripuly_heart.core.peer_capture import (
        PeerCaptureDiagnostic,
        PeerCaptureDiagnosticEvent,
        PeerCaptureFailureReason,
        PeerCaptureSessionState,
    )

    return PeerCaptureDiagnostic(
        event=PeerCaptureDiagnosticEvent.FAILURE,
        generation=1,
        state=PeerCaptureSessionState.FAULTED,
        provider_id="local_cpu_auto",
        capture_kind="process",
        reason=PeerCaptureFailureReason.PROCESS_TARGET_UNAVAILABLE,
        detail="no_process",
    )
