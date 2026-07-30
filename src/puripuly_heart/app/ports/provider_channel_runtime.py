from __future__ import annotations

from typing import Protocol

from puripuly_heart.core.local_asr_provider_runtime import ProviderRuntimeChannel


class ProviderChannelResetPort(Protocol):
    async def reset_provider_channel(self, channel: ProviderRuntimeChannel) -> None: ...


__all__ = ["ProviderChannelResetPort"]
