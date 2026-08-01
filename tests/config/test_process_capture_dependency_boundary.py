from __future__ import annotations

import importlib
import re
import sys
import tomllib

import pytest

from puripuly_heart.config.process_capture_platform import (
    PROCESS_CAPTURE_MIN_WINDOWS_BUILD,
    evaluate_process_capture_platform,
    get_process_capture_platform_availability,
)
from tests.helpers.paths import REPO_ROOT as ROOT

PROCESS_CAPTURE_MARKER = (
    "platform_system == 'Windows' and platform_python_implementation == 'CPython' "
    "and python_version == '3.12' and platform_machine == 'AMD64'"
)
PROCESS_CAPTURE_LOCK_MARKER = (
    "python_full_version == '3.12.*' and platform_machine == 'AMD64' "
    "and platform_python_implementation == 'CPython' and sys_platform == 'win32'"
)


def test_process_capture_dependencies_use_the_exact_supported_platform_marker() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = pyproject["project"]["dependencies"]

    assert f"proc-tap==1.1.1; {PROCESS_CAPTURE_MARKER}" in dependencies
    assert f"psutil>=5.9; {PROCESS_CAPTURE_MARKER}" in dependencies


def test_uv_lock_covers_pinned_proctap_and_windows_process_dependency() -> None:
    uv_lock = (ROOT / "uv.lock").read_text(encoding="utf-8")

    proctap = re.search(
        r'\[\[package\]\]\s+name = "proc-tap"\s+version = "([^"]+)"',
        uv_lock,
        re.MULTILINE,
    )
    psutil = re.search(
        r'\[\[package\]\]\s+name = "psutil"\s+version = "([^"]+)"',
        uv_lock,
        re.MULTILINE,
    )

    assert proctap is not None
    assert proctap.group(1) == "1.1.1"
    assert psutil is not None
    assert tuple(int(part) for part in psutil.group(1).split(".")[:2]) >= (5, 9)
    assert "proc_tap-1.1.1-cp312-cp312-win_amd64.whl" in uv_lock
    assert (
        '{ name = "proc-tap", marker = "'
        f"{PROCESS_CAPTURE_LOCK_MARKER}"
        '", specifier = "==1.1.1" }'
    ) in uv_lock
    assert (
        '{ name = "psutil", marker = "' f"{PROCESS_CAPTURE_LOCK_MARKER}" '", specifier = ">=5.9" }'
    ) in uv_lock


@pytest.mark.parametrize(
    ("system_name", "implementation", "python_version", "machine", "windows_build", "reason"),
    [
        ("Linux", "CPython", (3, 12), "AMD64", 20348, "unsupported_system"),
        ("Windows", "PyPy", (3, 12), "AMD64", 20348, "unsupported_implementation"),
        ("Windows", "CPython", (3, 13), "AMD64", 20348, "unsupported_python"),
        ("Windows", "CPython", (3, 12), "ARM64", 20348, "unsupported_machine"),
        ("Windows", "CPython", (3, 12), "AMD64", 20347, "unsupported_windows_build"),
        ("Windows", "CPython", (3, 12), "AMD64", None, "unsupported_windows_build"),
    ],
)
def test_process_capture_platform_validation_fails_closed(
    system_name: str,
    implementation: str,
    python_version: tuple[int, int],
    machine: str,
    windows_build: int | None,
    reason: str,
) -> None:
    availability = evaluate_process_capture_platform(
        system_name=system_name,
        implementation=implementation,
        python_version=python_version,
        machine=machine,
        windows_build=windows_build,
    )

    assert availability.available is False
    assert availability.reason == reason


def test_process_capture_platform_validation_accepts_only_the_supported_target() -> None:
    availability = evaluate_process_capture_platform(
        system_name="Windows",
        implementation="CPython",
        python_version=(3, 12),
        machine="AMD64",
        windows_build=PROCESS_CAPTURE_MIN_WINDOWS_BUILD,
    )

    assert availability.available is True
    assert availability.reason is None


def test_unsupported_current_platform_does_not_import_proctap(monkeypatch) -> None:
    monkeypatch.delitem(sys.modules, "proctap", raising=False)
    module = importlib.import_module("puripuly_heart.config.process_capture_platform")
    monkeypatch.setattr(module.platform, "system", lambda: "Linux")

    availability = get_process_capture_platform_availability()

    assert availability.available is False
    assert availability.reason == "unsupported_system"
    assert "proctap" not in sys.modules
