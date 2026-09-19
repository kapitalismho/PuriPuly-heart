from __future__ import annotations

import platform
import sys
from dataclasses import dataclass
from typing import Final

PROCESS_CAPTURE_MIN_WINDOWS_BUILD: Final = 20348


@dataclass(frozen=True, slots=True)
class ProcessCapturePlatformAvailability:
    available: bool
    reason: str | None = None


def get_process_capture_platform_availability() -> ProcessCapturePlatformAvailability:
    version = sys.version_info
    return evaluate_process_capture_platform(
        system_name=platform.system(),
        implementation=platform.python_implementation(),
        python_version=(version.major, version.minor),
        machine=platform.machine(),
        windows_build=_current_windows_build(),
    )


def evaluate_process_capture_platform(
    *,
    system_name: str,
    implementation: str,
    python_version: tuple[int, int],
    machine: str,
    windows_build: int | None,
) -> ProcessCapturePlatformAvailability:
    if system_name != "Windows":
        return ProcessCapturePlatformAvailability(available=False, reason="unsupported_system")
    if implementation.casefold() != "cpython":
        return ProcessCapturePlatformAvailability(
            available=False, reason="unsupported_implementation"
        )
    if python_version != (3, 14):
        return ProcessCapturePlatformAvailability(available=False, reason="unsupported_python")
    if machine.casefold() != "amd64":
        return ProcessCapturePlatformAvailability(available=False, reason="unsupported_machine")
    if windows_build is None or windows_build < PROCESS_CAPTURE_MIN_WINDOWS_BUILD:
        return ProcessCapturePlatformAvailability(
            available=False, reason="unsupported_windows_build"
        )
    return ProcessCapturePlatformAvailability(available=True)


def _current_windows_build() -> int | None:
    if sys.platform != "win32":
        return None
    getwindowsversion = getattr(sys, "getwindowsversion", None)
    if not callable(getwindowsversion):
        return None
    build = getattr(getwindowsversion(), "build", None)
    return build if isinstance(build, int) else None
