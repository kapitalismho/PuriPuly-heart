from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

MODULE_NAME = "puripuly_heart.core.soxr_runtime"


def _load_runtime_module():
    existing = sys.modules.get(MODULE_NAME)
    if existing is not None:
        return importlib.reload(existing)
    try:
        return importlib.import_module(MODULE_NAME)
    except ModuleNotFoundError as exc:
        pytest.fail(f"{MODULE_NAME} is missing: {exc}")


def test_runtime_resolution_rejects_missing_extension_spec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_module = _load_runtime_module()

    monkeypatch.setattr(runtime_module.importlib.util, "find_spec", lambda _name: None)

    with pytest.raises(
        runtime_module.SoxrRuntimeAvailabilityError,
        match="soxr extension module spec is unavailable",
    ):
        runtime_module.resolve_soxr_runtime_paths()


def test_runtime_resolution_wraps_find_spec_lookup_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_module = _load_runtime_module()

    def raise_lookup_failure(_name: str) -> None:
        raise ModuleNotFoundError("No module named 'soxr'")

    monkeypatch.setattr(runtime_module.importlib.util, "find_spec", raise_lookup_failure)

    with pytest.raises(
        runtime_module.SoxrRuntimeAvailabilityError,
        match="failed to resolve soxr extension module spec",
    ):
        runtime_module.resolve_soxr_runtime_paths()


def test_runtime_validation_rejects_missing_sibling_dll(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_module = _load_runtime_module()
    runtime_dir = tmp_path / "soxr"
    runtime_dir.mkdir()
    extension_path = runtime_dir / "soxr_ext.cp312-win_amd64.pyd"
    extension_path.write_bytes(b"")

    monkeypatch.setattr(
        runtime_module.importlib.util,
        "find_spec",
        lambda _name: type("Spec", (), {"origin": str(extension_path)})(),
    )

    with pytest.raises(
        runtime_module.SoxrRuntimeAvailabilityError,
        match=r"missing required soxr sibling DLL: .*[/\\]soxr\.dll$",
    ):
        runtime_module.validate_soxr_runtime_paths(runtime_module.resolve_soxr_runtime_paths())


def test_runtime_validation_returns_runtime_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_module = _load_runtime_module()
    runtime_dir = tmp_path / "soxr"
    runtime_dir.mkdir()
    extension_path = runtime_dir / "soxr_ext.cp312-win_amd64.pyd"
    extension_path.write_bytes(b"")
    sibling_dll_path = runtime_dir / runtime_module.SOXR_SIBLING_DLL_NAME
    sibling_dll_path.write_bytes(b"")

    monkeypatch.setattr(
        runtime_module.importlib.util,
        "find_spec",
        lambda _name: type("Spec", (), {"origin": str(extension_path)})(),
    )

    runtime_paths = runtime_module.validate_soxr_runtime_paths(
        runtime_module.resolve_soxr_runtime_paths()
    )

    assert runtime_paths.extension_path == extension_path.resolve()
    assert runtime_paths.runtime_dir == runtime_dir.resolve()
    assert runtime_paths.sibling_dll_path == sibling_dll_path.resolve()
