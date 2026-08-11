from __future__ import annotations

import pytest
from puripuly_heart.core.local_asr_provider_runtime import (
    LocalASRProviderRuntimeSnapshot,
    ProviderRuntimeChannelSnapshot,
    ProviderRuntimeGpuSnapshot,
)

from puripuly_heart.app.services.gpu_runtime_interaction import (
    GpuRuntimeInteractionOwner,
    GpuRuntimeInteractionState,
)
from puripuly_heart.core.gpu_worker import GpuWorkerDevice


class InstallingRuntime:
    async def inspect_gpu_readiness(
        self,
        *,
        explicit_intent: bool,
        device_id: str,
    ) -> LocalASRProviderRuntimeSnapshot:
        assert explicit_intent is True
        assert device_id == "auto"
        channels = tuple(
            ProviderRuntimeChannelSnapshot(
                channel=channel,
                provider_id=None,
                model_id=None,
                phase="inactive",
                generation=0,
                pending_handoff=False,
                has_resources=False,
            )
            for channel in ("self", "peer")
        )
        device = GpuWorkerDevice(
            device_id="vulkan-index-0",
            registry_index=0,
            name="GPU 0",
            description="GPU 0",
            device_type="discrete",
            memory_total_bytes=8_000_000_000,
            memory_free_bytes=4_000_000_000,
        )
        return LocalASRProviderRuntimeSnapshot(
            channels=channels,
            gpu=ProviderRuntimeGpuSnapshot(
                phase="installing",
                devices=(device,),
                active_channels=frozenset(),
                pending_count=0,
                worker_pid=None,
                configured_device_id=None,
                model_resident=False,
                retry_required=False,
                failure_code="downloading",
            ),
        )


@pytest.mark.asyncio
async def test_validate_activation_keeps_active_download_in_installing_state() -> None:
    presentations = []

    async def retry_activation() -> None:
        return None

    owner = GpuRuntimeInteractionOwner(
        runtime_provider=lambda: InstallingRuntime(),
        provisioning_provider=lambda: pytest.fail("provisioning should not be queried directly"),
        state_provider=lambda: GpuRuntimeInteractionState(
            settings_available=True,
            selected_provider_requires_model=True,
            locale="ko",
            device_id="auto",
        ),
        presentation_sink=presentations.append,
        detailed_log_sink=lambda _message: None,
        retry_activation=retry_activation,
    )

    assert await owner.validate_activation() is False
    assert owner.snapshot.ui_state == "installing"
    assert presentations[-1].state == "installing"
    assert presentations[-1].notice is not None
    assert presentations[-1].notice.status == "installing"
    assert presentations[-1].publish_notice is True
