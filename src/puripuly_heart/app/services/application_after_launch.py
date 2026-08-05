from __future__ import annotations

from dataclasses import dataclass

from puripuly_heart.app.services.gpu_runtime_interaction import GpuRuntimeInteractionOwner
from puripuly_heart.core.runtime.vrchat_osc_presence import (
    VrchatOscPresenceProbeOwner,
)


@dataclass(slots=True)
class ApplicationAfterLaunchOwner:
    vrchat_presence: VrchatOscPresenceProbeOwner
    gpu: GpuRuntimeInteractionOwner

    async def prepare(self) -> None:
        self.vrchat_presence.schedule(force=True)
        await self.gpu.preload_saved_device_discovery()


__all__ = ["ApplicationAfterLaunchOwner"]
