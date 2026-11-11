from __future__ import annotations

import importlib
import os
import sys
import types
from pathlib import Path

import pytest

MODULE_NAME = "puripuly_heart.core.local_asr.local_sherpa_runtime"


def _load_runtime_module():
    existing = sys.modules.get(MODULE_NAME)
    if existing is not None:
        return importlib.reload(existing)
    try:
        return importlib.import_module(MODULE_NAME)
    except ModuleNotFoundError as exc:
        pytest.fail(f"{MODULE_NAME} is missing: {exc}")


def test_resolve_local_sherpa_runtime_dir_uses_declared_native_layout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import builtins

    original_import = builtins.__import__

    def reject_onnx_import(name, *args, **kwargs):
        if name == "onnxruntime":
            pytest.fail("packaged preload resolution must not import ONNX Runtime")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_onnx_import)
    runtime_module = _load_runtime_module()
    runtime_root = tmp_path / "installed runtime"
    runtime_root.mkdir()
    monkeypatch.setattr(runtime_module.sys, "platform", "win32")
    monkeypatch.setenv("PURIPULY_HEART_NATIVE_RESOURCE_ROOT", str(tmp_path / "resources"))
    monkeypatch.setenv("PURIPULY_HEART_NATIVE_RUNTIME_ROOT", str(runtime_root))

    assert runtime_module.resolve_local_sherpa_runtime_dir() == (
        runtime_root / "site-packages/onnxruntime/capi"
    )


def test_source_runtime_resolution_keeps_the_developer_onnx_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime_module = _load_runtime_module()
    monkeypatch.delenv("PURIPULY_HEART_NATIVE_RESOURCE_ROOT", raising=False)
    package = types.ModuleType("onnxruntime")
    package.__file__ = str(tmp_path / "developer-onnx/onnxruntime/__init__.py")
    monkeypatch.setitem(sys.modules, "onnxruntime", package)
    monkeypatch.setattr(runtime_module.sys, "platform", "win32")

    assert runtime_module.resolve_local_sherpa_runtime_dir() == (
        tmp_path / "developer-onnx/onnxruntime/capi"
    )


def test_ensure_local_sherpa_windows_runtime_rejects_missing_required_dlls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_module = _load_runtime_module()
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    (runtime_dir / runtime_module.REQUIRED_LOCAL_SHERPA_RUNTIME_DLLS[0]).write_bytes(b"")

    monkeypatch.setattr(runtime_module.sys, "platform", "win32")
    monkeypatch.setattr(runtime_module, "resolve_local_sherpa_runtime_dir", lambda: runtime_dir)
    monkeypatch.setattr(
        runtime_module.os,
        "add_dll_directory",
        lambda _path: pytest.fail("should not register invalid runtime dir"),
        raising=False,
    )

    with pytest.raises(
        runtime_module.LocalSherpaRuntimeBootstrapError,
        match="onnxruntime_providers_shared\\.dll",
    ):
        runtime_module.ensure_local_sherpa_windows_runtime()


def test_ensure_local_sherpa_windows_runtime_registers_directory_only_once_and_prepends_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_module = _load_runtime_module()
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    for dll_name in runtime_module.REQUIRED_LOCAL_SHERPA_RUNTIME_DLLS:
        (runtime_dir / dll_name).write_bytes(b"")

    calls: list[str] = []
    loaded_libraries: list[str] = []
    directory_handle = object()
    library_handle = object()

    def fake_add_dll_directory(path: str) -> object:
        calls.append(path)
        return directory_handle

    def fake_win_dll(path: str) -> object:
        assert calls == [str(runtime_dir)]
        loaded_libraries.append(path)
        return library_handle

    monkeypatch.setattr(runtime_module.sys, "platform", "win32")
    monkeypatch.setattr(runtime_module, "resolve_local_sherpa_runtime_dir", lambda: runtime_dir)
    monkeypatch.setattr(
        runtime_module.os,
        "add_dll_directory",
        fake_add_dll_directory,
        raising=False,
    )
    monkeypatch.setattr(runtime_module.ctypes, "WinDLL", fake_win_dll, raising=False)
    monkeypatch.setenv("PATH", r"C:\\Windows\\System32")

    first = runtime_module.ensure_local_sherpa_windows_runtime()
    second = runtime_module.ensure_local_sherpa_windows_runtime()

    assert first == runtime_dir
    assert second == runtime_dir
    assert calls == [str(runtime_dir)]
    assert loaded_libraries == [str(runtime_dir / "onnxruntime.dll")]
    assert runtime_module._REGISTERED_LOCAL_SHERPA_RUNTIME_HANDLE is directory_handle
    assert runtime_module._REGISTERED_LOCAL_SHERPA_RUNTIME_LIBRARY is library_handle

    path_entries = os.environ["PATH"].split(os.pathsep)
    assert path_entries[0] == str(runtime_dir)
    assert path_entries.count(str(runtime_dir)) == 1


def test_ensure_local_sherpa_windows_runtime_rejects_unloadable_runtime_dll(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_module = _load_runtime_module()
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    for dll_name in runtime_module.REQUIRED_LOCAL_SHERPA_RUNTIME_DLLS:
        (runtime_dir / dll_name).write_bytes(b"")

    monkeypatch.setattr(runtime_module.sys, "platform", "win32")
    monkeypatch.setattr(runtime_module, "resolve_local_sherpa_runtime_dir", lambda: runtime_dir)
    monkeypatch.setattr(runtime_module.os, "add_dll_directory", lambda _path: object())
    monkeypatch.setattr(
        runtime_module.ctypes,
        "WinDLL",
        lambda _path: (_ for _ in ()).throw(OSError("load failed")),
        raising=False,
    )

    with pytest.raises(
        runtime_module.LocalSherpaRuntimeBootstrapError,
        match="failed to load local sherpa runtime DLL",
    ):
        runtime_module.ensure_local_sherpa_windows_runtime()
