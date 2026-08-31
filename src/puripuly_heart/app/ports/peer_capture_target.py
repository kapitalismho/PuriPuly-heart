from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from puripuly_heart.config.process_capture_resolution import ProcessCaptureCandidate
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext


class ProcessCaptureInventoryPort(Protocol):
    def candidates(self) -> Sequence[ProcessCaptureCandidate]: ...


class LoopbackDeviceInventoryPort(Protocol):
    def names(self) -> Sequence[str]: ...


class PeerCaptureTargetRuntimeEffectsPort(Protocol):
    async def apply_capture_target(self, settings: AppSettingsVNext) -> None: ...


__all__ = [
    "LoopbackDeviceInventoryPort",
    "PeerCaptureTargetRuntimeEffectsPort",
    "ProcessCaptureInventoryPort",
]
