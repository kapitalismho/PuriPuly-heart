from __future__ import annotations

import os
from pathlib import Path

from puripuly_heart.config.prompts import get_prompts_dir
from puripuly_heart.core.local_translation.runtime_profile import (
    LLAMA_CPP_RUNTIME_DIRNAME,
    default_llama_runtime_root,
)
from puripuly_heart.runtime_layout import current_runtime_layout


def test_source_layout_is_independent_of_working_directory(
    tmp_path: Path,
    monkeypatch,
) -> None:
    unrelated = tmp_path / "unrelated cwd 한글"
    unrelated.mkdir()
    monkeypatch.chdir(unrelated)
    monkeypatch.delenv("PURIPULY_HEART_PROMPTS_DIR", raising=False)
    monkeypatch.delenv("PURIPULY_HEART_LLAMA_CPP_ROOT", raising=False)

    layout = current_runtime_layout()

    assert layout.host_kind == "source"
    assert (layout.app_resource_root / "pyproject.toml").is_file()
    assert get_prompts_dir() == layout.resource("prompts")
    assert get_prompts_dir().is_dir()
    assert default_llama_runtime_root() == layout.native(LLAMA_CPP_RUNTIME_DIRNAME)
    assert layout.user_data_root != unrelated
    assert layout.model_cache_root.parent == layout.user_data_root
    assert layout.log_root.parent == layout.user_data_root


def test_native_layout_keeps_host_python_resources_and_writable_roots_distinct(
    tmp_path: Path,
    monkeypatch,
) -> None:
    install_root = tmp_path / "설치 경로 with spaces"
    resources = install_root / "resources"
    runtime = install_root / "python-runtime"
    host = install_root / "PuriPulyHeart.exe"
    python = runtime / "python.exe"
    for path in (resources, runtime):
        path.mkdir(parents=True)
    host.write_bytes(b"host")
    python.write_bytes(b"python")
    monkeypatch.setenv("PURIPULY_HEART_NATIVE_RESOURCE_ROOT", str(resources))
    monkeypatch.setenv("PURIPULY_HEART_NATIVE_RUNTIME_ROOT", str(runtime))
    monkeypatch.setenv("PURIPULY_HEART_NATIVE_HOST_EXECUTABLE", str(host))
    monkeypatch.setenv("PURIPULY_HEART_NATIVE_PYTHON_EXECUTABLE", str(python))

    layout = current_runtime_layout()

    assert layout.host_kind == "native"
    assert layout.app_resource_root == resources.resolve()
    assert layout.native_runtime_root == runtime.resolve()
    assert layout.host_executable == host.resolve()
    assert layout.python_executable == python.resolve()
    assert layout.host_executable != layout.python_executable
    assert not os.path.commonpath((layout.user_data_root, resources)) == str(resources)


def test_native_python_child_environment_is_installed_only_and_non_recursive(
    tmp_path: Path,
    monkeypatch,
) -> None:
    install_root = tmp_path / "설치 경로 with spaces"
    resources = install_root / "app"
    resources.mkdir(parents=True)
    monkeypatch.setenv("PURIPULY_HEART_NATIVE_RESOURCE_ROOT", str(resources))
    monkeypatch.setenv("PURIPULY_HEART_NATIVE_RUNTIME_ROOT", str(install_root))
    monkeypatch.setenv(
        "PURIPULY_HEART_NATIVE_HOST_EXECUTABLE",
        str(install_root / "PuriPulyHeart.exe"),
    )
    monkeypatch.setenv(
        "PURIPULY_HEART_NATIVE_PYTHON_EXECUTABLE",
        str(install_root / "python.exe"),
    )
    layout = current_runtime_layout()

    environment = layout.python_child_environment(
        {
            "SystemRoot": r"C:\Windows",
            "PATH": "poisoned",
            "PYTHONPATH": "poisoned",
            "FLET_DART_BRIDGE_PORT": "41",
            "FLET_DART_BRIDGE_EXIT_PORT": "42",
        }
    )

    assert environment["PYTHONHOME"] == str(install_root.resolve())
    assert environment["PYTHONPATH"].split(os.pathsep) == [
        str(resources.resolve()),
        str(install_root.resolve() / "site-packages"),
    ]
    assert environment["PATH"].split(os.pathsep) == [
        str(install_root.resolve()),
        str(install_root.resolve() / "DLLs"),
        str(install_root.resolve() / "site-packages"),
        r"C:\Windows\System32",
    ]
    assert environment["PYTHONDONTWRITEBYTECODE"] == "1"
    assert environment["PYTHONOPTIMIZE"] == "0"
    assert "FLET_DART_BRIDGE_PORT" not in environment
    assert "FLET_DART_BRIDGE_EXIT_PORT" not in environment
