from __future__ import annotations

import base64
import hashlib
import platform
import re
import subprocess
from pathlib import Path

try:
    import winreg
except ImportError:  # pragma: no cover - only exercised on non-Windows hosts
    winreg = None  # type: ignore[assignment]

_LINUX_MACHINE_ID_PATHS = (Path("/etc/machine-id"), Path("/var/lib/dbus/machine-id"))
_MACOS_IOREG_COMMAND = ("ioreg", "-rd1", "-c", "IOPlatformExpertDevice")
_MACOS_PLATFORM_UUID_PATTERN = re.compile(r'"IOPlatformUUID"\s*=\s*"([^"]+)"')


def get_raw_hardware_fingerprint() -> str:
    system_name = platform.system()
    if system_name == "Windows":
        return _get_windows_raw_hardware_fingerprint()
    if system_name == "Linux":
        return _get_linux_raw_hardware_fingerprint()
    if system_name == "Darwin":
        return _get_macos_raw_hardware_fingerprint()
    raise RuntimeError(f"hardware fingerprint unsupported platform: {system_name}")


def compute_hardware_hash(*, fingerprint_salt: str, raw_fingerprint: str) -> str:
    validated_fingerprint_salt = _validate_non_empty_text(
        fingerprint_salt,
        error_type=ValueError,
        message="fingerprint_salt must be a non-empty string",
    )
    validated_raw_fingerprint = _validate_non_empty_text(
        raw_fingerprint,
        error_type=ValueError,
        message="raw_fingerprint must be a non-empty string",
    )
    digest = hashlib.sha256(
        f"{validated_fingerprint_salt}{validated_raw_fingerprint}".encode("utf-8")
    ).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


def _get_windows_raw_hardware_fingerprint() -> str:
    if winreg is None:
        raise RuntimeError("Windows hardware fingerprint unavailable: winreg is not available")

    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Cryptography") as key:
            machine_guid, _value_type = winreg.QueryValueEx(key, "MachineGuid")
    except OSError as exc:
        raise RuntimeError(
            "Windows hardware fingerprint unavailable: MachineGuid could not be read"
        ) from exc

    return _require_non_empty_text(
        machine_guid,
        error_type=RuntimeError,
        message="Windows hardware fingerprint unavailable: MachineGuid was empty",
    )


def _get_linux_raw_hardware_fingerprint() -> str:
    for path in _LINUX_MACHINE_ID_PATHS:
        machine_id = _read_non_empty_text_file(path)
        if machine_id is not None:
            return machine_id
    raise RuntimeError("Linux hardware fingerprint unavailable: machine-id not found")


def _get_macos_raw_hardware_fingerprint() -> str:
    try:
        result = subprocess.run(
            _MACOS_IOREG_COMMAND,
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError("macOS hardware fingerprint unavailable: ioreg failed") from exc

    if result.returncode != 0:
        raise RuntimeError("macOS hardware fingerprint unavailable: ioreg failed")

    match = _MACOS_PLATFORM_UUID_PATTERN.search(result.stdout)
    if match is None:
        raise RuntimeError("macOS hardware fingerprint unavailable: IOPlatformUUID not found")

    return _require_non_empty_text(
        match.group(1),
        error_type=RuntimeError,
        message="macOS hardware fingerprint unavailable: IOPlatformUUID was empty",
    )


def _read_non_empty_text_file(path: Path) -> str | None:
    try:
        return _normalize_optional_text(path.read_text(encoding="utf-8"))
    except OSError, UnicodeDecodeError:
        return None


def _normalize_optional_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _require_non_empty_text(value: object, *, error_type: type[Exception], message: str) -> str:
    normalized = _normalize_optional_text(value)
    if normalized is None:
        raise error_type(message)
    return normalized


def _validate_non_empty_text(value: object, *, error_type: type[Exception], message: str) -> str:
    if _normalize_optional_text(value) is None:
        raise error_type(message)
    assert isinstance(value, str)
    return value
