from __future__ import annotations

from types import SimpleNamespace

import pytest

from puripuly_heart.app.services.local_asr.local_asr_cpu_repair import (
    LocalASRCpuRepairOwner,
    LocalASRCpuRepairRuntimeState,
)
from puripuly_heart.app.services.local_asr.local_asr_readiness import (
    LocalASRReadinessEffectType,
    LocalASRReadinessOwner,
    LocalASRReadinessState,
)
from puripuly_heart.app.services.local_asr.local_asr_selection import required_local_asr_model_ids
from puripuly_heart.core.asr_worker import ASRWorkerRequestError
from puripuly_heart.core.local_asr.local_asr_provisioning import (
    LocalASRModelProvisioningState,
    LocalASRProvisioningSnapshot,
)
from puripuly_heart.core.local_asr.local_stt_assets import (
    FUN_ASR_MLT_NANO_MODEL_ID,
    REQUIRED_CPU_LOCAL_STT_MODEL_IDS,
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "outcome", ["ready", "unsupported", "model_missing", "model_invalid", "backend_failure"]
)
async def test_cpu_readiness_preserves_selection_and_only_repairs_asset_failures(
    outcome: str,
) -> None:
    class Provisioning:
        def __init__(self):
            self.snapshot = LocalASRProvisioningSnapshot(
                models=tuple(
                    LocalASRModelProvisioningState(
                        model_id=model_id, backend="cpu", integrity="ready"
                    )
                    for model_id in REQUIRED_CPU_LOCAL_STT_MODEL_IDS
                ),
                required_cpu_model_ids=REQUIRED_CPU_LOCAL_STT_MODEL_IDS,
                gpu_model_id="gpu",
            )
            self.installs = []
            self.invalidated = []

        async def inspect_cpu(self, _model_ids):
            return self.snapshot

        async def report_model_validation_failure(self, model_id, **_kwargs):
            self.invalidated.append(model_id)

        def start_install(self, request, **_kwargs):
            self.installs.append(request)

    provisioning = Provisioning()
    repair_effects = []
    readiness_effects = []
    probes = []
    logs = []
    state = LocalASRReadinessState(
        settings_available=True,
        runtime_available=True,
        self_provider="local_cpu_auto",
        peer_provider="local_cpu_auto",
        self_source_language="fr" if outcome == "unsupported" else "ko-KR",
        peer_source_language="en",
        self_desired=True,
        peer_requested=False,
        self_activation_generation=1,
        peer_activation_generation=1,
    )

    async def idle():
        return None

    async def probe():
        probes.append(True)
        if outcome not in {"ready", "unsupported"}:
            raise ASRWorkerRequestError(outcome)

    repair = LocalASRCpuRepairOwner(
        provisioning_provider=lambda: provisioning,
        state_provider=lambda: LocalASRCpuRepairRuntimeState(
            settings_available=True,
            locale="en",
            self_provider=state.self_provider,
            peer_provider=state.peer_provider,
            self_provider_local=True,
            peer_requested=False,
            self_activation_generation=1,
            peer_activation_generation=1,
            self_desired=True,
        ),
        model_ids_for_provider=required_local_asr_model_ids,
        status_for_provider=lambda _provider: "ready",
        effect_sink=repair_effects.append,
        rebuild_self_provider=idle,
        resume_self=idle,
        resume_peer=idle,
    )
    owner = LocalASRReadinessOwner(
        provisioning_provider=lambda: provisioning,
        cpu_repair_owner=repair,
        state_provider=lambda: state,
        effect_sink=readiness_effects.append,
        self_provider_available=lambda: True,
        self_channel_provider=lambda: SimpleNamespace(
            phase="available", model_id=FUN_ASR_MLT_NANO_MODEL_ID
        ),
        rebuild_self_provider=idle,
        probe_self_provider=probe,
        validate_gpu_activation=idle,
        gpu_state_provider=lambda: None,
        retain_gpu_pending=lambda _channel: pytest.fail("CPU readiness cannot activate GPU"),
        load_log_sink=lambda **fields: logs.append(fields),
    )

    if outcome == "backend_failure":
        with pytest.raises(ASRWorkerRequestError, match="backend_failure"):
            await owner.ensure_self_ready(activation_generation=1)
    else:
        assert await owner.ensure_self_ready(activation_generation=1) is (outcome == "ready")

    assert state.self_provider == "local_cpu_auto"
    assert state.self_source_language == ("fr" if outcome == "unsupported" else "ko-KR")
    if outcome in {"model_missing", "model_invalid"}:
        assert len(provisioning.installs) == 1
        assert provisioning.installs[0].model_ids == (FUN_ASR_MLT_NANO_MODEL_ID,)
        assert repair.snapshot.self_pending
    else:
        assert provisioning.installs == []
    assert provisioning.invalidated == (
        [FUN_ASR_MLT_NANO_MODEL_ID] if outcome == "model_invalid" else []
    )
    if outcome == "unsupported":
        assert probes == []
        assert [effect.type for effect in readiness_effects] == [
            LocalASRReadinessEffectType.DISABLE_SELF_UNSUPPORTED
        ]
    else:
        assert probes == [True]
    if outcome == "ready":
        assert logs[0]["outcome"] == "ready"
        assert logs[0]["model_id"] == FUN_ASR_MLT_NANO_MODEL_ID
        assert readiness_effects[-1].type is LocalASRReadinessEffectType.SYNC_NOTICE
