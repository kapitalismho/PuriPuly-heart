from __future__ import annotations

import contextlib
import importlib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from puripuly_heart.config.process_capture_resolution import (
    ProcessCaptureCandidate,
    ProcessCaptureResolver,
)
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext
from puripuly_heart.core.audio.process_identity import PsutilCurrentUserProcessSnapshots


class WindowsProcessCaptureInventoryAdapter:
    def candidates(self) -> tuple[ProcessCaptureCandidate, ...]:
        return ProcessCaptureResolver(
            snapshots=PsutilCurrentUserProcessSnapshots()
        ).enumerate_candidates()


class WindowsLoopbackDeviceInventoryAdapter:
    def names(self) -> tuple[str, ...]:
        names: list[str] = []
        manager = None
        try:
            pyaudio = importlib.import_module("pyaudiowpatch")
            manager = pyaudio.PyAudio()
            seen: set[str] = set()
            for info in manager.get_loopback_device_info_generator():
                name = str(info.get("name", "") or "").strip()
                if not name or name in seen:
                    continue
                seen.add(name)
                names.append(name)
        except Exception:
            return tuple(names)
        finally:
            if manager is not None:
                with contextlib.suppress(Exception):
                    manager.terminate()
        return tuple(names)


@dataclass(frozen=True, slots=True)
class PeerCaptureTargetRuntimeEffectsAdapter:
    refresh_peer: Callable[[], Awaitable[None]]
    sync_effective_flags: Callable[[object], None]
    refresh_presentation: Callable[[], None]

    async def apply_capture_target(self, settings: AppSettingsVNext) -> None:
        await self.refresh_peer()
        self.sync_effective_flags(settings)
        self.refresh_presentation()


__all__ = [
    "PeerCaptureTargetRuntimeEffectsAdapter",
    "WindowsLoopbackDeviceInventoryAdapter",
    "WindowsProcessCaptureInventoryAdapter",
]
