from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest

from tests.helpers.paths import REPO_ROOT as ROOT

MODULE_PATH = ROOT / "src" / "puripuly_heart" / "core" / "overlay" / "openvr_vendor.py"
PINNED_OPENVR_DLL_SHA256 = "bab8ac6ef64e68a9ca53315b0014d131088584b2efdfa6db511d67ec03cfcb4a"


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
    bundle = module.validate_vendored_openvr_bundle(ROOT / "third_party" / "openvr")

    assert (
        module.OPENVR_VENDOR_DLL_SHA256 == bundle.dll_sha256
    ), "module hash pin must match the disk-hashed vendored bundle"
    assert module.OPENVR_VENDOR_SHA256_LINE == f"{bundle.dll_sha256} *openvr_api.dll"


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


def test_default_vendored_bundle_uses_source_resource_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_openvr_vendor_module()
    monkeypatch.setattr(module, "__file__", str(tmp_path / "relocated" / "openvr_vendor.pyc"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PURIPULY_HEART_NATIVE_RESOURCE_ROOT", raising=False)

    bundle = module.validate_vendored_openvr_bundle()

    assert bundle.bundle_dir == ROOT / "third_party" / "openvr"
    assert bundle.dll_sha256 == PINNED_OPENVR_DLL_SHA256
