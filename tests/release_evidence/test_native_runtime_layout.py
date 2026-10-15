from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from puripuly_heart.config.prompts import get_prompts_dir
from puripuly_heart.core.local_translation.runtime_profile import (
    LLAMA_CPP_RUNTIME_DIRNAME,
    default_llama_runtime_root,
)
from puripuly_heart.release_evidence.native_distribution import bundle_runtime, compile_runtime
from puripuly_heart.runtime_layout import current_runtime_layout
from tests.helpers.paths import SOURCE_ROOT


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
    assert layout.package_resource("data", "THIRD_PARTY_NOTICES.txt") == (
        SOURCE_ROOT / "data" / "THIRD_PARTY_NOTICES.txt"
    )
    assert layout.package_resource("data", "THIRD_PARTY_NOTICES.txt").is_file()
    assert (
        default_llama_runtime_root()
        == layout.app_resource_root / "build" / "llama.cpp" / LLAMA_CPP_RUNTIME_DIRNAME
    )
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
        str(resources.resolve() / "python.zip"),
        str(install_root.resolve() / "python314.zip"),
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


def test_frozen_package_resources_and_child_environment_preserve_runtime_boundary(
    tmp_path: Path,
    monkeypatch,
) -> None:
    extraction_root = tmp_path / "frozen 한글"
    notices = extraction_root / "puripuly_heart" / "data" / "THIRD_PARTY_NOTICES.txt"
    notices.parent.mkdir(parents=True)
    shutil.copy2(SOURCE_ROOT / "data" / "THIRD_PARTY_NOTICES.txt", notices)
    monkeypatch.delenv("PURIPULY_HEART_NATIVE_RESOURCE_ROOT", raising=False)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(extraction_root), raising=False)

    layout = current_runtime_layout()

    assert layout.host_kind == "pyinstaller"
    assert layout.package_resource("data", "THIRD_PARTY_NOTICES.txt") == notices
    assert notices.read_text(encoding="utf-8") == (
        SOURCE_ROOT / "data" / "THIRD_PARTY_NOTICES.txt"
    ).read_text(encoding="utf-8")
    environment = {"PYTHONPATH": "inherited", "PATH": "inherited"}
    assert layout.python_child_environment(environment) == environment


def test_native_child_consumes_production_bundle_resources_without_installed_writes(
    tmp_path: Path,
    staged_archive_runtime: tuple[Path, Path, dict[str, object]],
) -> None:
    layout_path, bytecode, _ = staged_archive_runtime
    package = tmp_path / "app" / "puripuly_heart"
    shutil.copytree(
        SOURCE_ROOT,
        package,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "data"),
    )
    resource_paths = (
        "data/vad/silero_vad.onnx",
        "data/fonts/NotoSansCJK-Medium.ttc",
        "data/THIRD_PARTY_NOTICES.txt",
        "data/licenses/PYTHON-3.14.7-LICENSE.txt",
        "data/models/qwen3-asr-0.6b-int8-sherpa.manifest.json",
    )
    for relative in resource_paths:
        destination = package / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SOURCE_ROOT / relative, destination)
    shutil.copytree(SOURCE_ROOT / "data/i18n", package / "data/i18n")
    bytecode.write_text(json.dumps(compile_runtime(tmp_path, layout_path)), encoding="utf-8")
    bundle_runtime(tmp_path, layout_path, bytecode)
    app_root = tmp_path / "relocated app 한글"
    (tmp_path / "app").rename(app_root)
    archive_path = app_root / "python.zip"
    (app_root / "puripuly_heart/__init__.py").write_text(
        "raise AssertionError('loose application fallback')\n", encoding="utf-8"
    )
    installed_roots = (app_root, tmp_path / "site-packages", tmp_path / "Lib")
    before = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for root in installed_roots
        for path in root.rglob("*")
        if path.is_file()
    }
    script = (
        "import hashlib, json, os, sys, zipimport\n"
        f"sys.path[:0] = {[str(archive_path), str(app_root), str(tmp_path / 'site-packages')]!r}\n"
        f"os.environ['PURIPULY_HEART_NATIVE_RESOURCE_ROOT'] = {str(app_root)!r}\n"
        f"os.environ['PURIPULY_HEART_NATIVE_RUNTIME_ROOT'] = {str(tmp_path)!r}\n"
        "from _puripuly_native_runtime import install\n"
        f"install({str(tmp_path)!r})\n"
        "def reject_compilation(*args, **kwargs):\n"
        "    raise AssertionError('archive source compilation')\n"
        "zipimport._compile_source = reject_compilation\n"
        """
from puripuly_heart import runtime_layout
from puripuly_heart.core.vad import bundled
from puripuly_heart.core.local_asr import local_stt_assets
from puripuly_heart.ui import fonts, i18n

layout = runtime_layout.current_runtime_layout()
model = bundled.bundled_silero_vad_onnx_path()
font = fonts.fonts_dir() / "NotoSansCJK-Medium.ttc"
resources = [
    model,
    font,
    layout.package_resource("data", "THIRD_PARTY_NOTICES.txt"),
    layout.package_resource("data", "licenses", "PYTHON-3.14.7-LICENSE.txt"),
]
manifest = local_stt_assets.load_local_stt_asset_manifest()
print(json.dumps({
    "host_kind": layout.host_kind,
    "origins": [runtime_layout.__file__, bundled.__file__, fonts.__file__, i18n.__file__,
                local_stt_assets.__file__],
    "resources": [
        {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for path in resources
    ],
    "font_url": fonts.font_asset_path(fonts.FONT_FAMILY_NOTO_SANS_CJK_JP),
    "manifest_id": manifest.model_id,
    "locales": i18n.available_locales(),
    "title": i18n.t("app.title"),
}))
"""
    )
    environment = dict(os.environ)
    environment["LOCALAPPDATA"] = str(tmp_path / "profile")
    unrelated = tmp_path / "unrelated working directory"
    unrelated.mkdir()
    completed = subprocess.run(
        [sys.executable, "-I", "-B", "-c", script],
        cwd=unrelated,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["host_kind"] == "native"
    for origin in result["origins"]:
        assert origin.replace("\\", "/").startswith(archive_path.as_posix() + "/")
        assert origin.endswith(".pyc")
    assert result["resources"] == [
        {
            "path": str(app_root / "puripuly_heart" / relative),
            "sha256": hashlib.sha256((SOURCE_ROOT / relative).read_bytes()).hexdigest(),
        }
        for relative in resource_paths[:4]
    ]
    assert result["font_url"] == "/fonts/NotoSansCJK-Medium.ttc"
    assert (
        result["manifest_id"]
        == json.loads((SOURCE_ROOT / resource_paths[4]).read_text(encoding="utf-8"))["model_id"]
    )
    assert sorted(result["locales"]) == sorted(
        path.stem for path in (SOURCE_ROOT / "data/i18n").glob("*.json")
    )
    assert (
        result["title"]
        == json.loads((SOURCE_ROOT / "data/i18n/en.json").read_text(encoding="utf-8"))["app.title"]
    )
    assert {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for root in installed_roots
        for path in root.rglob("*")
        if path.is_file()
    } == before
