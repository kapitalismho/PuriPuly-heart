from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from puripuly_heart.app.services.gpu_runtime_interaction import GpuRuntimeInteractionOwner
from puripuly_heart.core.audio.process_identity import PsutilCurrentUserProcessSnapshots
from puripuly_heart.core.runtime.vrchat_osc_presence import (
    VrchatOscPresenceProbeOwner,
)


@dataclass(slots=True)
class ApplicationAfterLaunchOwner:
    schedule_task: Callable[[Callable[[], Awaitable[None]]], bool]
    vrchat_presence: VrchatOscPresenceProbeOwner
    gpu: GpuRuntimeInteractionOwner
    _process_discovery_scheduled: bool = field(
        init=False,
        default=False,
        repr=False,
    )

    async def prepare(self) -> None:
        self.schedule_process_discovery()
        self.vrchat_presence.schedule(force=True)
        await self.gpu.preload_saved_device_discovery()

    def schedule_process_discovery(self) -> None:
        if self._process_discovery_scheduled:
            return
        self._process_discovery_scheduled = True
        with contextlib.suppress(Exception):
            self.schedule_task(self.prepare_process_discovery)

    async def prepare_process_discovery(self) -> None:
        with contextlib.suppress(Exception):
            await asyncio.to_thread(lambda: tuple(PsutilCurrentUserProcessSnapshots().snapshots()))


__all__ = ["ApplicationAfterLaunchOwner"]
