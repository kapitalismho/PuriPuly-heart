from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from dataclasses import replace

import pytest
from puripuly_heart.app.services.local_asr_gpu_provisioning import (
    LocalASRGpuProvisioningDiagnostic,
    LocalASRGpuProvisioningEffect,
    LocalASRGpuProvisioningOwner,
    LocalASRGpuProvisioningState,
)
from puripuly_heart.core.local_asr_provisioning import (
    LocalASRInstallRequest,
    LocalASRInstallResult,
    LocalASRModelProvisioningState,
    LocalASRProvisioningActivity,
    LocalASRProvisioningSnapshot,
)
from puripuly_heart.core.local_stt_assets import LOCAL_QWEN_GPU_MODEL_ID


def _snapshot(
    status: str,
    *,
    activity: bool = False,
) -> LocalASRProvisioningSnapshot:
    activities = (
        (
            LocalASRProvisioningActivity(
                backend="gpu",
                model_id=LOCAL_QWEN_GPU_MODEL_ID,
                origin="manual",
                progress_percent=20,
                generation=1,
            ),
        )
        if activity
        else ()
    )
    return LocalASRProvisioningSnapshot(
        models=(
            LocalASRModelProvisioningState(
                model_id=LOCAL_QWEN_GPU_MODEL_ID,
                backend="gpu",
                integrity=status,
            ),
        ),
        required_cpu_model_ids=(),
        gpu_model_id=LOCAL_QWEN_GPU_MODEL_ID,
        activities=activities,
    )


class RecordingProvisioning:
    def __init__(
        self,
        snapshot: LocalASRProvisioningSnapshot,
        *,
        release: asyncio.Event | None = None,
        result: str = "ready",
        failure: BaseException | None = None,
        inspect_hook: Callable[[], None] | None = None,
    ) -> None:
        self._snapshot = snapshot
        self.release = release
        self.result = result
        self.failure = failure
        self.inspect_hook = inspect_hook
        self.inspect_calls: list[tuple[bool, bool]] = []
        self.requests: list[LocalASRInstallRequest] = []

    @property
    def snapshot(self) -> LocalASRProvisioningSnapshot:
        return self._snapshot

    async def inspect_gpu(
        self,
        *,
        explicit_intent: bool,
        verify_checksums: bool = False,
    ) -> LocalASRProvisioningSnapshot:
        self.inspect_calls.append((explicit_intent, verify_checksums))
        if self.inspect_hook is not None:
            self.inspect_hook()
        return self._snapshot

    def start_install(
        self,
        request: LocalASRInstallRequest,
        *,
        result_handler=None,
    ) -> asyncio.Task[LocalASRInstallResult]:
        self.requests.append(request)
        self._snapshot = replace(
            self._snapshot,
            activities=(
                LocalASRProvisioningActivity(
                    backend="gpu",
                    model_id=LOCAL_QWEN_GPU_MODEL_ID,
                    origin=request.origin,
                    progress_percent=0,
                    generation=1,
                ),
            ),
        )

        async def finish_and_deliver() -> LocalASRInstallResult:
            result = await self._finish(request)
            if result_handler is not None:
                outcome = result_handler(result)
                if inspect.isawaitable(outcome):
                    await outcome
            return result

        return asyncio.create_task(finish_and_deliver())

    async def _finish(
        self,
        request: LocalASRInstallRequest,
    ) -> LocalASRInstallResult:
        if self.release is not None:
            await self.release.wait()
        if self.failure is not None:
            raise self.failure
        self._snapshot = _snapshot("ready" if self.result == "ready" else "missing")
        return LocalASRInstallResult(
            request=request,
            installed_model_ids=((LOCAL_QWEN_GPU_MODEL_ID,) if self.result == "ready" else ()),
            failed_model_ids=((LOCAL_QWEN_GPU_MODEL_ID,) if self.result == "failed" else ()),
            cancelled=self.result == "cancelled",
            snapshot=self._snapshot,
        )


def _owner(
    provisioning: RecordingProvisioning,
    state_box: list[LocalASRGpuProvisioningState],
    *,
    effects: list[LocalASRGpuProvisioningEffect] | None = None,
    retries: list[str] | None = None,
    diagnostics: list[LocalASRGpuProvisioningDiagnostic] | None = None,
):
    recorded_effects = effects if effects is not None else []
    recorded_retries = retries if retries is not None else []
    recorded_diagnostics = diagnostics if diagnostics is not None else []

    async def retry() -> None:
        recorded_retries.append("retry")

    return LocalASRGpuProvisioningOwner(
        provisioning_provider=lambda: provisioning,
        state_provider=lambda: state_box[0],
        effect_sink=recorded_effects.append,
        retry_activation=retry,
        diagnostic_sink=recorded_diagnostics.append,
    )


@pytest.mark.asyncio
async def test_selected_install_skips_unselected_active_and_ready_states() -> None:
    state_box = [
        LocalASRGpuProvisioningState(
            selected_provider_requires_model=False,
            locale="ko",
            pending_channels=frozenset(),
        )
    ]
    unselected = RecordingProvisioning(_snapshot("missing"))
    assert await _owner(unselected, state_box).install_selected_model_if_needed() is False
    assert unselected.inspect_calls == []

    state_box[0] = replace(state_box[0], selected_provider_requires_model=True)
    active = RecordingProvisioning(_snapshot("missing", activity=True))
    assert await _owner(active, state_box).install_selected_model_if_needed() is False
    assert active.inspect_calls == []

    ready = RecordingProvisioning(_snapshot("ready"))
    assert await _owner(ready, state_box).install_selected_model_if_needed() is False
    assert ready.inspect_calls == [(True, False)]
    assert ready.requests == []


@pytest.mark.asyncio
async def test_selected_install_rechecks_provider_after_explicit_inspection() -> None:
    state_box = [
        LocalASRGpuProvisioningState(
            selected_provider_requires_model=True,
            locale="ja",
            pending_channels=frozenset(),
        )
    ]
    provisioning = RecordingProvisioning(
        _snapshot("missing"),
        inspect_hook=lambda: state_box.__setitem__(
            0,
            replace(state_box[0], selected_provider_requires_model=False),
        ),
    )

    assert await _owner(provisioning, state_box).install_selected_model_if_needed() is False
    assert provisioning.inspect_calls == [(True, False)]
    assert provisioning.requests == []


@pytest.mark.asyncio
async def test_request_install_starts_background_activation_origin() -> None:
    state_box = [
        LocalASRGpuProvisioningState(
            selected_provider_requires_model=True,
            locale="ko",
            pending_channels=frozenset(),
        )
    ]
    release = asyncio.Event()
    provisioning = RecordingProvisioning(_snapshot("missing"), release=release)
    effects: list[LocalASRGpuProvisioningEffect] = []
    owner = _owner(provisioning, state_box, effects=effects)

    assert owner.request_install() is True
    assert owner.request_install() is False
    await asyncio.sleep(0)
    assert provisioning.requests == [
        LocalASRInstallRequest(
            backend="gpu",
            model_ids=(LOCAL_QWEN_GPU_MODEL_ID,),
            locale="ko",
            origin="activation",
            explicit_gpu_intent=True,
        )
    ]
    assert effects[0] == LocalASRGpuProvisioningEffect(
        state="installing",
        origin="activation",
        progress_percent=0,
        publish_notice=True,
    )
    release.set()
    while effects[-1].state != "installed":
        await asyncio.sleep(0)
    assert effects[-1] == LocalASRGpuProvisioningEffect(
        state="installed",
        origin="activation",
    )


@pytest.mark.asyncio
async def test_selected_install_forwards_exact_request_and_terminal_effects() -> None:
    state_box = [
        LocalASRGpuProvisioningState(
            selected_provider_requires_model=True,
            locale="en",
            pending_channels=frozenset(),
        )
    ]
    provisioning = RecordingProvisioning(_snapshot("missing"))
    effects: list[LocalASRGpuProvisioningEffect] = []

    assert (
        await _owner(
            provisioning,
            state_box,
            effects=effects,
        ).install_selected_model_if_needed()
        is True
    )

    assert provisioning.inspect_calls == [(True, False)]
    assert provisioning.requests == [
        LocalASRInstallRequest(
            backend="gpu",
            model_ids=(LOCAL_QWEN_GPU_MODEL_ID,),
            locale="en",
            origin="settings_exit",
            explicit_gpu_intent=True,
        )
    ]
    assert effects == [
        LocalASRGpuProvisioningEffect(
            state="installing",
            origin="settings_exit",
            progress_percent=0,
            publish_notice=True,
        ),
        LocalASRGpuProvisioningEffect(
            state="installed",
            origin="settings_exit",
        ),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("result", ["failed", "cancelled"])
async def test_install_result_preserves_failure_and_cancelled_effects(result: str) -> None:
    state_box = [
        LocalASRGpuProvisioningState(
            selected_provider_requires_model=True,
            locale=None,
            pending_channels=frozenset(),
        )
    ]
    provisioning = RecordingProvisioning(_snapshot("missing"), result=result)
    effects: list[LocalASRGpuProvisioningEffect] = []

    await _owner(provisioning, state_box, effects=effects).install_or_repair()

    expected = [
        LocalASRGpuProvisioningEffect(
            state="installing",
            origin="manual",
            progress_percent=0,
            publish_notice=True,
        )
    ]
    if result == "failed":
        expected.append(
            LocalASRGpuProvisioningEffect(
                state="install_failed",
                origin="manual",
                publish_notice=True,
            )
        )
    assert effects == expected


@pytest.mark.asyncio
async def test_install_retries_only_channels_pending_at_completion() -> None:
    state_box = [
        LocalASRGpuProvisioningState(
            selected_provider_requires_model=True,
            locale="ko",
            pending_channels=frozenset({"self"}),
        )
    ]
    release = asyncio.Event()
    provisioning = RecordingProvisioning(_snapshot("missing"), release=release)
    retries: list[str] = []
    install = asyncio.create_task(
        _owner(provisioning, state_box, retries=retries).install_or_repair()
    )
    await asyncio.sleep(0)
    state_box[0] = replace(state_box[0], pending_channels=frozenset())
    release.set()
    await install
    assert retries == []

    release = asyncio.Event()
    provisioning = RecordingProvisioning(_snapshot("missing"), release=release)
    install = asyncio.create_task(
        _owner(provisioning, state_box, retries=retries).install_or_repair()
    )
    await asyncio.sleep(0)
    state_box[0] = replace(state_box[0], pending_channels=frozenset({"peer"}))
    release.set()
    await install
    assert retries == ["retry"]


@pytest.mark.asyncio
async def test_install_propagates_cancellation() -> None:
    state_box = [
        LocalASRGpuProvisioningState(
            selected_provider_requires_model=True,
            locale="ko",
            pending_channels=frozenset(),
        )
    ]
    provisioning = RecordingProvisioning(
        _snapshot("missing"),
        release=asyncio.Event(),
    )
    install = asyncio.create_task(_owner(provisioning, state_box).install_or_repair())
    await asyncio.sleep(0)
    install.cancel()

    with pytest.raises(asyncio.CancelledError):
        await install


@pytest.mark.asyncio
async def test_unexpected_install_failure_emits_safe_diagnostic_and_failure_effect() -> None:
    state_box = [
        LocalASRGpuProvisioningState(
            selected_provider_requires_model=True,
            locale="ko",
            pending_channels=frozenset(),
        )
    ]
    failure = RuntimeError("private detail")
    provisioning = RecordingProvisioning(
        _snapshot("missing"),
        failure=failure,
    )
    effects: list[LocalASRGpuProvisioningEffect] = []
    diagnostics: list[LocalASRGpuProvisioningDiagnostic] = []

    await _owner(
        provisioning,
        state_box,
        effects=effects,
        diagnostics=diagnostics,
    ).install_or_repair(origin="repair")

    assert diagnostics == [
        LocalASRGpuProvisioningDiagnostic(
            event="model_install",
            outcome="failed",
            origin="repair",
            failure_type="RuntimeError",
            exception=failure,
        )
    ]
    assert diagnostics[0].exception is failure
    assert effects[-1] == LocalASRGpuProvisioningEffect(
        state="install_failed",
        origin="repair",
        publish_notice=True,
    )


@pytest.mark.asyncio
async def test_diagnostic_sink_failure_is_contained() -> None:
    state_box = [
        LocalASRGpuProvisioningState(
            selected_provider_requires_model=True,
            locale="ko",
            pending_channels=frozenset(),
        )
    ]
    provisioning = RecordingProvisioning(
        _snapshot("missing"),
        failure=RuntimeError("install"),
    )
    effects: list[LocalASRGpuProvisioningEffect] = []
    owner = LocalASRGpuProvisioningOwner(
        provisioning_provider=lambda: provisioning,
        state_provider=lambda: state_box[0],
        effect_sink=effects.append,
        retry_activation=lambda: asyncio.sleep(0),
        diagnostic_sink=lambda _diagnostic: (_ for _ in ()).throw(RuntimeError("sink")),
    )

    await owner.install_or_repair()

    assert effects[-1].state == "install_failed"
