from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest

MODULE_NAME = "puripuly_heart.core.local_asr.local_qwen_runtime"


def _load_runtime_module():
    existing = sys.modules.get(MODULE_NAME)
    if existing is not None:
        return importlib.reload(existing)
    try:
        return importlib.import_module(MODULE_NAME)
    except ModuleNotFoundError as exc:
        pytest.fail(f"{MODULE_NAME} is missing: {exc}")


def test_resolve_local_qwen_runtime_dir_uses_packaged_dir_for_frozen_windows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_module = _load_runtime_module()
    executable = tmp_path / "dist" / "PuriPulyHeart.exe"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"")

    monkeypatch.setattr(runtime_module.sys, "platform", "win32")
    monkeypatch.setattr(runtime_module.sys, "frozen", True, raising=False)
    monkeypatch.setattr(runtime_module.sys, "executable", str(executable))

    assert runtime_module.resolve_local_qwen_runtime_dir() == (
        executable.parent / runtime_module.LOCAL_QWEN_PACKAGED_RUNTIME_RELATIVE_DIR
    )


def test_resolve_local_qwen_runtime_dir_falls_back_to_internal_packaged_dir_for_pyinstaller_layout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_module = _load_runtime_module()
    executable = tmp_path / "dist" / "PuriPulyHeart.exe"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"")
    fallback_runtime_dir = (
        executable.parent / "_internal" / runtime_module.LOCAL_QWEN_PACKAGED_RUNTIME_RELATIVE_DIR
    )
    fallback_runtime_dir.mkdir(parents=True)

    monkeypatch.setattr(runtime_module.sys, "platform", "win32")
    monkeypatch.setattr(runtime_module.sys, "frozen", True, raising=False)
    monkeypatch.setattr(runtime_module.sys, "executable", str(executable))

    assert runtime_module.resolve_local_qwen_runtime_dir() == fallback_runtime_dir


def test_ensure_local_qwen_windows_runtime_rejects_missing_required_dlls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_module = _load_runtime_module()
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    (runtime_dir / runtime_module.REQUIRED_LOCAL_QWEN_RUNTIME_DLLS[0]).write_bytes(b"")

    monkeypatch.setattr(runtime_module.sys, "platform", "win32")
    monkeypatch.setattr(runtime_module, "resolve_local_qwen_runtime_dir", lambda: runtime_dir)
    monkeypatch.setattr(
        runtime_module.os,
        "add_dll_directory",
        lambda _path: pytest.fail("should not register invalid runtime dir"),
        raising=False,
    )

    with pytest.raises(
        runtime_module.LocalQwenRuntimeBootstrapError,
        match="onnxruntime_providers_shared\\.dll",
    ):
        runtime_module.ensure_local_qwen_windows_runtime()


def test_ensure_local_qwen_windows_runtime_registers_directory_only_once_and_prepends_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_module = _load_runtime_module()
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    for dll_name in runtime_module.REQUIRED_LOCAL_QWEN_RUNTIME_DLLS:
        (runtime_dir / dll_name).write_bytes(b"")

    calls: list[str] = []

    def fake_add_dll_directory(path: str) -> object:
        calls.append(path)
        return object()

    monkeypatch.setattr(runtime_module.sys, "platform", "win32")
    monkeypatch.setattr(runtime_module, "resolve_local_qwen_runtime_dir", lambda: runtime_dir)
    monkeypatch.setattr(
        runtime_module.os,
        "add_dll_directory",
        fake_add_dll_directory,
        raising=False,
    )
    monkeypatch.setenv("PATH", r"C:\\Windows\\System32")

    first = runtime_module.ensure_local_qwen_windows_runtime()
    second = runtime_module.ensure_local_qwen_windows_runtime()

    assert first == runtime_dir
    assert second == runtime_dir
    assert calls == [str(runtime_dir)]

    path_entries = os.environ["PATH"].split(os.pathsep)
    assert path_entries[0] == str(runtime_dir)
    assert path_entries.count(str(runtime_dir)) == 1
