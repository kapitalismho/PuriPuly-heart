from __future__ import annotations

import hashlib
from pathlib import Path
from typing import NamedTuple

from puripuly_heart.runtime_layout import current_runtime_layout

OPENVR_VENDOR_REPOSITORY_REF = "ValveSoftware/openvr@v2.15.6"
OPENVR_VENDOR_DLL_URL = (
    "https://raw.githubusercontent.com/ValveSoftware/openvr/v2.15.6/bin/win64/openvr_api.dll"
)
OPENVR_VENDOR_LICENSE_URL = "https://raw.githubusercontent.com/ValveSoftware/openvr/v2.15.6/LICENSE"
OPENVR_VENDOR_DLL_SHA256 = "bab8ac6ef64e68a9ca53315b0014d131088584b2efdfa6db511d67ec03cfcb4a"
OPENVR_VENDOR_SHA256_LINE = f"{OPENVR_VENDOR_DLL_SHA256} *openvr_api.dll"
OPENVR_VENDOR_BUNDLE_RELATIVE_DIR = Path("third_party/openvr")
OPENVR_VENDOR_DLL_RELATIVE_PATH = Path("win64/openvr_api.dll")
OPENVR_VENDOR_SHA256_RELATIVE_PATH = Path("win64/openvr_api.dll.sha256")
OPENVR_VENDOR_LICENSE_RELATIVE_PATH = Path("LICENSE")
OPENVR_VENDOR_README_RELATIVE_PATH = Path("README.md")
OPENVR_VENDOR_PACKAGED_RUNTIME_RELATIVE_DIR = "."


class VendoredOpenVrBundle(NamedTuple):
    bundle_dir: Path
    dll_path: Path
    sha256_path: Path
    license_path: Path
    readme_path: Path
    dll_sha256: str


def validate_openvr_runtime_dll(
    dll_path: Path, *, expected_sha256: str = OPENVR_VENDOR_DLL_SHA256
) -> Path:
    if not dll_path.is_file():
        raise FileNotFoundError(f"Vendored OpenVR runtime DLL not found: {dll_path}")

    normalized_expected_sha256 = expected_sha256.strip().lower()
    actual_sha256 = _sha256_file(dll_path)
    if actual_sha256 != normalized_expected_sha256:
        raise ValueError(
            "Vendored OpenVR runtime DLL sha256 mismatch: expected "
            f"{normalized_expected_sha256}, got {actual_sha256}"
        )

    return dll_path


def validate_vendored_openvr_bundle(
    bundle_dir: Path | None = None, *, expected_sha256: str = OPENVR_VENDOR_DLL_SHA256
) -> VendoredOpenVrBundle:
    resolved_bundle_dir = (bundle_dir or _default_bundle_dir()).resolve()
    dll_path = resolved_bundle_dir / OPENVR_VENDOR_DLL_RELATIVE_PATH
    sha256_path = resolved_bundle_dir / OPENVR_VENDOR_SHA256_RELATIVE_PATH
    license_path = resolved_bundle_dir / OPENVR_VENDOR_LICENSE_RELATIVE_PATH
    readme_path = resolved_bundle_dir / OPENVR_VENDOR_README_RELATIVE_PATH

    for required_path in (dll_path, sha256_path, license_path, readme_path):
        if not required_path.is_file():
            raise FileNotFoundError(f"Vendored OpenVR bundle file not found: {required_path}")

    normalized_expected_sha256 = expected_sha256.strip().lower()
    expected_sha256_line = f"{normalized_expected_sha256} *openvr_api.dll"
    sha256_text = sha256_path.read_text(encoding="utf-8")
    if sha256_text not in {expected_sha256_line, f"{expected_sha256_line}\n"}:
        raise ValueError(
            "Vendored OpenVR sha256 file must be a single-line sha256sum entry for "
            f"openvr_api.dll: {expected_sha256_line}"
        )

    validate_openvr_runtime_dll(dll_path, expected_sha256=normalized_expected_sha256)

    return VendoredOpenVrBundle(
        bundle_dir=resolved_bundle_dir,
        dll_path=dll_path,
        sha256_path=sha256_path,
        license_path=license_path,
        readme_path=readme_path,
        dll_sha256=normalized_expected_sha256,
    )


def collect_vendored_openvr_runtime_binaries() -> list[tuple[str, str]]:
    bundle = validate_vendored_openvr_bundle()
    return [(str(bundle.dll_path), OPENVR_VENDOR_PACKAGED_RUNTIME_RELATIVE_DIR)]


def _default_bundle_dir() -> Path:
    return current_runtime_layout().resource(*OPENVR_VENDOR_BUNDLE_RELATIVE_DIR.parts)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "OPENVR_VENDOR_BUNDLE_RELATIVE_DIR",
    "OPENVR_VENDOR_DLL_SHA256",
    "OPENVR_VENDOR_DLL_URL",
    "OPENVR_VENDOR_LICENSE_URL",
    "OPENVR_VENDOR_PACKAGED_RUNTIME_RELATIVE_DIR",
    "OPENVR_VENDOR_REPOSITORY_REF",
    "OPENVR_VENDOR_SHA256_LINE",
    "VendoredOpenVrBundle",
    "collect_vendored_openvr_runtime_binaries",
    "validate_openvr_runtime_dll",
    "validate_vendored_openvr_bundle",
]
