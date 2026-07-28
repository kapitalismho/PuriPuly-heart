from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol, cast


class PeerProcessCaptureRetryRuntimePort(Protocol):
    async def retry_process_capture(self, *, config: object) -> bool: ...


@dataclass(slots=True)
class PeerProcessCaptureRetryOwner:
    settings_provider: Callable[[], object | None]
    runtime_provider: Callable[[], PeerProcessCaptureRetryRuntimePort | None]
    should_be_active: Callable[[object], bool]
    ensure_ready: Callable[[], Awaitable[bool]]
    build_config: Callable[[object], object]
    on_retry_succeeded: Callable[[], None]
    sync_effective_flags: Callable[[object], None]
    refresh_consumers: Callable[[], None]

    async def retry(self) -> bool:
        initial_settings = self.settings_provider()
        if initial_settings is None:
            return False
        initial_runtime = self.runtime_provider()
        if initial_runtime is None or not self.should_be_active(initial_settings):
            return False
        if not await self.ensure_ready():
            return False
        current_settings = cast(object, self.settings_provider())
        config = self.build_config(current_settings)
        current_runtime = cast(
            PeerProcessCaptureRetryRuntimePort,
            self.runtime_provider(),
        )
        retried = await current_runtime.retry_process_capture(config=config)
        if retried:
            self.on_retry_succeeded()
        self.sync_effective_flags(cast(object, self.settings_provider()))
        self.refresh_consumers()
        return retried
