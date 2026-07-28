from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import replace

import pytest

from puripuly_heart.app.services.local_asr_cpu_repair import (
    LocalASRCpuRepairEffect,
    LocalASRCpuRepairEffectType,
    LocalASRCpuRepairOwner,
    LocalASRCpuRepairRequest,
    LocalASRCpuRepairRuntimeState,
)
from puripuly_heart.app.wiring_composition import create_local_asr_cpu_repair_owner
from puripuly_heart.core.local_asr_provisioning import (
    LocalASRInstallRequest,
    LocalASRInstallResult,
    LocalASRInstallResultHandler,
    LocalASRModelProvisioningState,
    LocalASRProvisioningActivity,
    LocalASRProvisioningSnapshot,
)
from puripuly_heart.core.local_stt_assets import (
    LOCAL_STT_MODEL_ID,
    PARAKEET_JAPANESE_MODEL_ID,
    PARAKEET_V3_MODEL_ID,
    REQUIRED_CPU_LOCAL_STT_MODEL_IDS,
)

LOCAL_QWEN_PROVIDER = "local_qwen"
LOCAL_AUTO_PROVIDER = "local_cpu_auto"


def _snapshot(
    *,
    qwen: str = "ready",
    parakeet_v3: str = "ready",
    parakeet_ja: str = "ready",
    active: bool = False,
) -> LocalASRProvisioningSnapshot:
    return LocalASRProvisioningSnapshot(
        models=(
            LocalASRModelProvisioningState(
                model_id=PARAKEET_V3_MODEL_ID,
                backend="cpu",
                integrity=parakeet_v3,
            ),
            LocalASRModelProvisioningState(
                model_id=PARAKEET_JAPANESE_MODEL_ID,
                backend="cpu",
                integrity=parakeet_ja,
            ),
            LocalASRModelProvisioningState(
                model_id=LOCAL_STT_MODEL_ID,
                backend="cpu",
                integrity=qwen,
            ),
        ),
        required_cpu_model_ids=REQUIRED_CPU_LOCAL_STT_MODEL_IDS,
        gpu_model_id="gpu",
        activities=(
            (
                LocalASRProvisioningActivity(
                    backend="cpu",
                    model_id=LOCAL_STT_MODEL_ID,
                    origin="manual",
                    progress_percent=0,
                    generation=1,
                ),
            )
            if active
            else ()
        ),
    )


class RecordingProvisioning:
    def __init__(
        self,
        snapshot: LocalASRProvisioningSnapshot,
        *,
        start_error: RuntimeError | None = None,
    ) -> None:
        self._snapshot = snapshot
        self.start_error = start_error
        self.requests: list[LocalASRInstallRequest] = []
        self.handlers: list[LocalASRInstallResultHandler | None] = []
        self.start_attempts = 0

    @property
    def snapshot(self) -> LocalASRProvisioningSnapshot:
        return self._snapshot

    def start_install(
        self,
        request: LocalASRInstallRequest,
        *,
        result_handler: LocalASRInstallResultHandler | None = None,
    ):
        self.start_attempts += 1
        if self.start_error is not None:
            raise self.start_error
        self.requests.append(request)
        self.handlers.append(result_handler)
        self._snapshot = replace(
            self._snapshot,
            activities=(
                LocalASRProvisioningActivity(
                    backend="cpu",
                    model_id=request.model_ids[0],
                    origin=request.origin,
                    progress_percent=0,
                    generation=self.start_attempts,
                ),
            ),
        )
        return None

    async def deliver(
        self,
        result: LocalASRInstallResult,
        *,
        index: int = -1,
    ) -> None:
        handler = self.handlers[index]
        assert handler is not None
        outcome = handler(result)
        if inspect.isawaitable(outcome):
            await outcome


def _state(
    *,
    settings_available: bool = True,
    locale: str | None = "ko",
    self_provider: str | None = LOCAL_QWEN_PROVIDER,
    peer_provider: str | None = LOCAL_QWEN_PROVIDER,
    self_provider_local: bool = True,
    peer_requested: bool = False,
    self_generation: int = 7,
    peer_generation: int = 11,
    self_desired: bool = True,
) -> LocalASRCpuRepairRuntimeState:
    return LocalASRCpuRepairRuntimeState(
        settings_available=settings_available,
        locale=locale,
        self_provider=self_provider,
        peer_provider=peer_provider,
        self_provider_local=self_provider_local,
        peer_requested=peer_requested,
        self_activation_generation=self_generation,
        peer_activation_generation=peer_generation,
        self_desired=self_desired,
    )


def _model_ids(provider: str) -> tuple[str, ...]:
    if provider == LOCAL_AUTO_PROVIDER:
        return REQUIRED_CPU_LOCAL_STT_MODEL_IDS
    if provider == LOCAL_QWEN_PROVIDER:
        return (LOCAL_STT_MODEL_ID,)
    return ()


def _result(
    provisioning: RecordingProvisioning,
    *,
    origin: str = "manual",
    failed: bool = False,
    cancelled: bool = False,
) -> LocalASRInstallResult:
    request = LocalASRInstallRequest(
        backend="cpu",
        model_ids=(LOCAL_STT_MODEL_ID,),
        locale="ko",
        origin=origin,
    )
    return LocalASRInstallResult(
        request=request,
        installed_model_ids=(() if failed or cancelled else (LOCAL_STT_MODEL_ID,)),
        failed_model_ids=((LOCAL_STT_MODEL_ID,) if failed else ()),
        cancelled=cancelled,
        snapshot=provisioning.snapshot,
    )


def _owner(
    provisioning: RecordingProvisioning,
    state_box: list[LocalASRCpuRepairRuntimeState],
    *,
    effects: list[LocalASRCpuRepairEffect] | None = None,
    calls: list[str] | None = None,
    rebuild_hook: Callable[[LocalASRCpuRepairOwner], None] | None = None,
    self_resume_result: bool = True,
    status_by_provider: dict[str, str] | None = None,
    model_ids_for_provider: Callable[[str], tuple[str, ...]] = _model_ids,
) -> LocalASRCpuRepairOwner:
    recorded_effects = effects if effects is not None else []
    recorded_calls = calls if calls is not None else []
    owner_box: list[LocalASRCpuRepairOwner] = []

    async def rebuild() -> None:
        recorded_calls.append("rebuild")
        if rebuild_hook is not None:
            rebuild_hook(owner_box[0])

    async def resume_self() -> bool:
        recorded_calls.append("self")
        return self_resume_result

    async def resume_peer() -> None:
        recorded_calls.append("peer")

    owner = create_local_asr_cpu_repair_owner(
        provisioning_provider=lambda: provisioning,
        state_provider=lambda: state_box[0],
        model_ids_for_provider=model_ids_for_provider,
        status_for_provider=lambda provider: (status_by_provider or {}).get(
            provider,
            "ready",
        ),
        effect_sink=recorded_effects.append,
        rebuild_self_provider=rebuild,
        resume_self=resume_self,
        resume_peer=resume_peer,
    )
    owner_box.append(owner)
    return owner


def test_self_repair_admission_preserves_effect_order_exact_request_and_single_flight() -> None:
    provisioning = RecordingProvisioning(_snapshot(qwen="missing"))
    state_box = [_state()]
    effects: list[LocalASRCpuRepairEffect] = []
    owner = _owner(provisioning, state_box, effects=effects)

    assert (
        owner.request_repair(
            LocalASRCpuRepairRequest(
                status="missing",
                channel="self",
                activation_generation=7,
            )
        )
        is False
    )
    assert owner.snapshot.self_pending is True
    assert owner.snapshot.self_activation_generation == 7
    assert effects == [
        LocalASRCpuRepairEffect(LocalASRCpuRepairEffectType.DISABLE_SELF_INTENT),
        LocalASRCpuRepairEffect(LocalASRCpuRepairEffectType.DISABLE_SELF_DASHBOARD),
        LocalASRCpuRepairEffect(LocalASRCpuRepairEffectType.SYNC_NOTICE),
    ]
    assert provisioning.requests == [
        LocalASRInstallRequest(
            backend="cpu",
            model_ids=(LOCAL_STT_MODEL_ID,),
            locale="ko",
            origin="manual",
        )
    ]

    owner.request_repair(
        LocalASRCpuRepairRequest(
            status="missing",
            channel="self",
            activation_generation=7,
        )
    )

    assert len(provisioning.requests) == 1
    assert provisioning.start_attempts == 1


def test_repair_admission_rejects_stale_self_and_peer_generations() -> None:
    provisioning = RecordingProvisioning(_snapshot(qwen="missing"))
    state_box = [_state(peer_requested=True)]
    effects: list[LocalASRCpuRepairEffect] = []
    owner = _owner(provisioning, state_box, effects=effects)

    owner.request_repair(
        LocalASRCpuRepairRequest(
            status="missing",
            channel="self",
            activation_generation=8,
        )
    )
    owner.request_repair(
        LocalASRCpuRepairRequest(
            status="missing",
            channel="peer",
            activation_generation=12,
        )
    )

    assert owner.snapshot == owner.snapshot.__class__(
        self_pending=False,
        self_activation_generation=None,
        peer_pending=False,
    )
    assert effects == []
    assert provisioning.requests == []


def test_repair_without_generation_preserves_legacy_admission_and_missing_settings_skip() -> None:
    provisioning = RecordingProvisioning(_snapshot(qwen="missing"))
    state_box = [_state(self_desired=False)]
    effects: list[LocalASRCpuRepairEffect] = []
    owner = _owner(provisioning, state_box, effects=effects)

    owner.request_repair(
        LocalASRCpuRepairRequest(
            status="missing",
            channel="self",
        )
    )

    assert owner.snapshot.self_pending is True
    assert owner.snapshot.self_activation_generation == 7
    assert len(provisioning.requests) == 1

    unavailable = RecordingProvisioning(_snapshot(qwen="missing"))
    unavailable_effects: list[LocalASRCpuRepairEffect] = []
    unavailable_owner = _owner(
        unavailable,
        [_state(settings_available=False)],
        effects=unavailable_effects,
    )

    unavailable_owner.request_repair(
        LocalASRCpuRepairRequest(
            status="missing",
            channel="self",
        )
    )

    assert unavailable_owner.snapshot.self_pending is False
    assert unavailable_effects == []
    assert unavailable.requests == []


def test_repair_re_resolves_provisioning_after_model_selection_effects() -> None:
    inspected = RecordingProvisioning(_snapshot(qwen="missing"))
    installed = RecordingProvisioning(_snapshot(qwen="missing"))
    provider_box = [inspected]
    state_box = [_state()]
    effects: list[LocalASRCpuRepairEffect] = []

    async def idle() -> None:
        return None

    async def resume_self() -> bool:
        return True

    def apply_effect(effect: LocalASRCpuRepairEffect) -> None:
        effects.append(effect)
        if effect.type is LocalASRCpuRepairEffectType.SYNC_NOTICE:
            provider_box[0] = installed

    owner = create_local_asr_cpu_repair_owner(
        provisioning_provider=lambda: provider_box[0],
        state_provider=lambda: state_box[0],
        model_ids_for_provider=_model_ids,
        status_for_provider=lambda _provider: "ready",
        effect_sink=apply_effect,
        rebuild_self_provider=idle,
        resume_self=resume_self,
        resume_peer=idle,
    )

    owner.request_repair(
        LocalASRCpuRepairRequest(
            status="missing",
            channel="self",
            activation_generation=7,
        )
    )

    assert inspected.requests == []
    assert installed.requests == [
        LocalASRInstallRequest(
            backend="cpu",
            model_ids=(LOCAL_STT_MODEL_ID,),
            locale="ko",
            origin="manual",
        )
    ]


def test_repair_uses_required_models_when_snapshot_has_no_unavailable_identity() -> None:
    provisioning = RecordingProvisioning(_snapshot())
    state_box = [
        _state(
            self_provider=LOCAL_AUTO_PROVIDER,
            self_provider_local=True,
        )
    ]
    owner = _owner(provisioning, state_box)

    owner.request_repair(
        LocalASRCpuRepairRequest(
            status="invalid",
            channel="self",
        )
    )

    assert provisioning.requests[0].model_ids == REQUIRED_CPU_LOCAL_STT_MODEL_IDS


def test_repair_with_explicit_models_skips_provider_model_resolution() -> None:
    provisioning = RecordingProvisioning(_snapshot(qwen="missing"))
    state_box = [
        _state(
            self_provider=LOCAL_AUTO_PROVIDER,
            self_provider_local=True,
        )
    ]

    def unexpected_model_resolution(_provider: str) -> tuple[str, ...]:
        raise AssertionError("explicit model IDs must bypass provider model resolution")

    owner = _owner(
        provisioning,
        state_box,
        model_ids_for_provider=unexpected_model_resolution,
    )

    owner.request_repair(
        LocalASRCpuRepairRequest(
            status="invalid",
            channel="self",
            model_ids=(PARAKEET_V3_MODEL_ID,),
        )
    )

    assert provisioning.requests[0].model_ids == (PARAKEET_V3_MODEL_ID,)


def test_repair_with_empty_models_uses_unavailable_subset_before_required_fallback() -> None:
    provisioning = RecordingProvisioning(_snapshot(qwen="missing"))
    state_box = [
        _state(
            self_provider=LOCAL_AUTO_PROVIDER,
            self_provider_local=True,
        )
    ]
    owner = _owner(provisioning, state_box)

    owner.request_repair(
        LocalASRCpuRepairRequest(
            status="missing",
            channel="self",
            model_ids=(),
        )
    )

    assert provisioning.requests[0].model_ids == (LOCAL_STT_MODEL_ID,)


def test_start_install_runtime_error_is_a_false_single_flight_admission() -> None:
    provisioning = RecordingProvisioning(
        _snapshot(qwen="missing"),
        start_error=RuntimeError("raced"),
    )
    state_box = [_state()]
    owner = _owner(provisioning, state_box)

    assert owner.request_install(origin="manual") is False
    assert provisioning.start_attempts == 1
    assert provisioning.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("origin", "failed", "cancelled", "expected_effects"),
    [
        (
            "manual",
            True,
            False,
            [LocalASRCpuRepairEffect(LocalASRCpuRepairEffectType.SHOW_DOWNLOAD_FAILED)],
        ),
        ("settings", True, False, []),
        ("manual", False, True, []),
    ],
)
async def test_failed_and_cancelled_results_preserve_pending_intent(
    origin: str,
    failed: bool,
    cancelled: bool,
    expected_effects: list[LocalASRCpuRepairEffect],
) -> None:
    provisioning = RecordingProvisioning(_snapshot())
    state_box = [_state()]
    effects: list[LocalASRCpuRepairEffect] = []
    calls: list[str] = []
    owner = _owner(provisioning, state_box, effects=effects, calls=calls)
    owner.retain_pending("self", activation_generation=7)

    await owner.handle_install_result(
        _result(
            provisioning,
            origin=origin,
            failed=failed,
            cancelled=cancelled,
        ),
        origin=origin,
    )

    assert owner.snapshot.self_pending is True
    assert effects == expected_effects
    assert calls == []


@pytest.mark.asyncio
async def test_successful_manual_result_resumes_self_then_peer() -> None:
    provisioning = RecordingProvisioning(_snapshot())
    state_box = [_state(peer_requested=True)]
    calls: list[str] = []
    owner = _owner(provisioning, state_box, calls=calls)
    owner.retain_pending("self", activation_generation=7)
    owner.retain_pending("peer")

    await owner.handle_install_result(
        _result(provisioning),
        origin="manual",
    )

    assert calls == ["rebuild", "self", "peer"]
    assert owner.snapshot.self_pending is False
    assert owner.snapshot.peer_pending is False


@pytest.mark.asyncio
async def test_successful_result_waits_for_ready_provider_status() -> None:
    provisioning = RecordingProvisioning(_snapshot(qwen="missing"))
    state_box = [_state()]
    calls: list[str] = []
    owner = _owner(
        provisioning,
        state_box,
        calls=calls,
        status_by_provider={LOCAL_QWEN_PROVIDER: "missing"},
    )
    owner.retain_pending("self", activation_generation=7)

    await owner.handle_install_result(
        _result(provisioning),
        origin="manual",
    )

    assert calls == []
    assert owner.snapshot.self_pending is True


@pytest.mark.asyncio
async def test_self_provider_switch_suppresses_self_without_losing_peer_resume() -> None:
    provisioning = RecordingProvisioning(_snapshot())
    state_box = [
        _state(
            self_provider="deepgram",
            self_provider_local=False,
            peer_requested=True,
        )
    ]
    calls: list[str] = []
    owner = _owner(provisioning, state_box, calls=calls)
    owner.retain_pending("self", activation_generation=7)
    owner.retain_pending("peer")

    await owner.handle_install_result(
        _result(provisioning),
        origin="manual",
    )

    assert calls == ["peer"]
    assert owner.snapshot.self_pending is False
    assert owner.snapshot.peer_pending is False


@pytest.mark.asyncio
async def test_peer_disable_suppresses_peer_resume() -> None:
    provisioning = RecordingProvisioning(_snapshot())
    state_box = [_state(peer_requested=False)]
    calls: list[str] = []
    owner = _owner(provisioning, state_box, calls=calls)
    owner.retain_pending("peer")

    await owner.handle_install_result(
        _result(provisioning),
        origin="manual",
    )

    assert calls == []
    assert owner.snapshot.peer_pending is False


@pytest.mark.asyncio
async def test_self_generation_change_during_rebuild_returns_before_peer_resume() -> None:
    provisioning = RecordingProvisioning(_snapshot())
    state_box = [_state(peer_requested=True)]
    calls: list[str] = []
    owner = _owner(
        provisioning,
        state_box,
        calls=calls,
        rebuild_hook=lambda current: current.set_self_activation_generation(8),
    )
    owner.retain_pending("self", activation_generation=7)
    owner.retain_pending("peer")

    await owner.handle_install_result(
        _result(provisioning),
        origin="manual",
    )

    assert calls == ["rebuild"]
    assert owner.snapshot.self_pending is True
    assert owner.snapshot.peer_pending is True


@pytest.mark.asyncio
async def test_stale_self_resume_result_returns_before_peer_resume() -> None:
    provisioning = RecordingProvisioning(_snapshot())
    state_box = [_state(peer_requested=True)]
    calls: list[str] = []
    owner = _owner(
        provisioning,
        state_box,
        calls=calls,
        self_resume_result=False,
    )
    owner.retain_pending("self", activation_generation=7)
    owner.retain_pending("peer")

    await owner.handle_install_result(
        _result(provisioning),
        origin="manual",
    )

    assert calls == ["rebuild", "self"]
    assert owner.snapshot.self_pending is False
    assert owner.snapshot.peer_pending is True


@pytest.mark.asyncio
async def test_non_manual_success_clears_switched_intent_without_resume() -> None:
    provisioning = RecordingProvisioning(_snapshot())
    state_box = [
        _state(
            self_provider="deepgram",
            self_provider_local=False,
            peer_requested=False,
        )
    ]
    calls: list[str] = []
    owner = _owner(provisioning, state_box, calls=calls)
    owner.retain_pending("self", activation_generation=7)
    owner.retain_pending("peer")

    await owner.handle_install_result(
        _result(provisioning, origin="settings"),
        origin="settings",
    )

    assert calls == []
    assert owner.snapshot.self_pending is False
    assert owner.snapshot.peer_pending is False


def test_owner_lifecycle_inventory_declares_state_without_resources() -> None:
    provisioning = RecordingProvisioning(_snapshot())
    owner = _owner(provisioning, [_state()])

    inventory = owner.lifecycle_owner_snapshot()

    assert inventory["owner"] == "LocalASRCpuRepairOwner"
    assert inventory["state_fields"] == (
        "_self_pending",
        "_self_activation_generation",
        "_peer_pending",
    )
    assert inventory["shutdown_policy"] == "no task or external resource is retained"
