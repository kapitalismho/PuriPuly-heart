from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest

from tests.helpers.paths import REPO_ROOT as ROOT

MODULE_PATH = ROOT / "src" / "puripuly_heart" / "core" / "overlay" / "openvr_vendor.py"
PINNED_OPENVR_VENDOR_REF = "ValveSoftware/openvr@v2.15.6"
PINNED_OPENVR_DLL_URL = (
    "https://raw.githubusercontent.com/ValveSoftware/openvr/v2.15.6/bin/win64/openvr_api.dll"
)
PINNED_OPENVR_LICENSE_URL = "https://raw.githubusercontent.com/ValveSoftware/openvr/v2.15.6/LICENSE"
PINNED_OPENVR_DLL_SHA256 = "bab8ac6ef64e68a9ca53315b0014d131088584b2efdfa6db511d67ec03cfcb4a"
PINNED_OPENVR_SHA256_LINE = f"{PINNED_OPENVR_DLL_SHA256} *openvr_api.dll"


def _load_openvr_vendor_module():
    assert MODULE_PATH.is_file(), f"Missing OpenVR vendor module at {MODULE_PATH}"

    spec = importlib.util.spec_from_file_location("test_openvr_vendor_module", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_openvr_vendor_module_exposes_pinned_bundle_contract() -> None:
    module = _load_openvr_vendor_module()

    assert module.OPENVR_VENDOR_REPOSITORY_REF == PINNED_OPENVR_VENDOR_REF
    assert module.OPENVR_VENDOR_DLL_URL == PINNED_OPENVR_DLL_URL
    assert module.OPENVR_VENDOR_LICENSE_URL == PINNED_OPENVR_LICENSE_URL
    assert module.OPENVR_VENDOR_DLL_SHA256 == PINNED_OPENVR_DLL_SHA256
    assert module.OPENVR_VENDOR_SHA256_LINE == PINNED_OPENVR_SHA256_LINE
    assert callable(module.validate_vendored_openvr_bundle)
    assert callable(module.validate_openvr_runtime_dll)
    assert callable(module.collect_vendored_openvr_runtime_binaries)


def test_validate_vendored_openvr_bundle_accepts_repo_bundle() -> None:
    module = _load_openvr_vendor_module()

    bundle = module.validate_vendored_openvr_bundle(ROOT / "third_party" / "openvr")

    assert bundle.bundle_dir == ROOT / "third_party" / "openvr"
    assert bundle.dll_path == bundle.bundle_dir / "win64" / "openvr_api.dll"
    assert bundle.sha256_path == bundle.bundle_dir / "win64" / "openvr_api.dll.sha256"
    assert bundle.license_path == bundle.bundle_dir / "LICENSE"
    assert bundle.readme_path == bundle.bundle_dir / "README.md"
    assert bundle.dll_sha256 == PINNED_OPENVR_DLL_SHA256


def test_validate_vendored_openvr_bundle_rejects_non_sha256sum_line(tmp_path: Path) -> None:
    module = _load_openvr_vendor_module()
    bundle_dir = tmp_path / "openvr"
    win64_dir = bundle_dir / "win64"
    win64_dir.mkdir(parents=True)
    (bundle_dir / "LICENSE").write_text("license", encoding="utf-8")
    (bundle_dir / "README.md").write_text("readme", encoding="utf-8")
    (win64_dir / "openvr_api.dll").write_bytes(b"test-dll")
    (win64_dir / "openvr_api.dll.sha256").write_text(
        f"{PINNED_OPENVR_DLL_SHA256} openvr_api.dll\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="sha256sum"):
        module.validate_vendored_openvr_bundle(bundle_dir)


def test_validate_openvr_runtime_dll_validates_explicit_expected_sha256(tmp_path: Path) -> None:
    module = _load_openvr_vendor_module()
    dll_path = tmp_path / "openvr_api.dll"
    dll_path.write_bytes(b"vendored-openvr-test")
    expected_sha256 = hashlib.sha256(dll_path.read_bytes()).hexdigest()

    assert module.validate_openvr_runtime_dll(dll_path, expected_sha256=expected_sha256) == dll_path

    with pytest.raises(ValueError, match="sha256"):
        module.validate_openvr_runtime_dll(dll_path, expected_sha256="0" * 64)
