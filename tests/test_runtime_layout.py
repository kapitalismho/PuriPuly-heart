from __future__ import annotations

import json
import os
import py_compile
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from puripuly_heart.config.prompts import get_prompts_dir
from puripuly_heart.core.local_translation.runtime_profile import (
    LLAMA_CPP_RUNTIME_DIRNAME,
    default_llama_runtime_root,
)
from puripuly_heart.runtime_layout import RuntimeLayout, current_runtime_layout
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


def test_native_child_imports_archived_bytecode_and_filesystem_package_resources(
    tmp_path: Path,
) -> None:
    app_root = tmp_path / "relocated app 한글"
    app_root.mkdir()
    archive_path = app_root / "python.zip"
    bytecode_root = tmp_path / "bytecode"
    directories: set[str] = set()
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for source in sorted(SOURCE_ROOT.rglob("*.py")):
            relative = source.relative_to(SOURCE_ROOT.parent)
            for parent in relative.parents:
                if parent != Path("."):
                    directory = parent.as_posix() + "/"
                    if directory not in directories:
                        archive.writestr(directory, b"")
                        directories.add(directory)
            bytecode = bytecode_root / relative.with_suffix(".pyc")
            bytecode.parent.mkdir(parents=True, exist_ok=True)
            py_compile.compile(
                str(source),
                cfile=str(bytecode),
                dfile=relative.as_posix(),
                doraise=True,
                optimize=0,
            )
            archive.write(bytecode, relative.with_suffix(".pyc").as_posix())
            archive.write(source, relative.as_posix())
    resource_paths = (
        "data/vad/silero_vad.onnx",
        "data/fonts/NotoSansCJK-Medium.ttc",
        "data/THIRD_PARTY_NOTICES.txt",
        "data/licenses/PYTHON-3.14.7-LICENSE.txt",
        "data/models/qwen3-asr-0.6b-int8-sherpa.manifest.json",
        "data/i18n/en.json",
        "data/i18n/ko.json",
    )
    for relative_path in resource_paths:
        destination = app_root / "puripuly_heart" / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SOURCE_ROOT / relative_path, destination)
    layout = RuntimeLayout(
        host_kind="native",
        app_resource_root=app_root,
        native_runtime_root=Path(sys.base_prefix),
        host_executable=tmp_path / "PuriPulyHeart.exe",
        python_executable=Path(sys.executable),
        user_data_root=tmp_path / "profile",
        model_cache_root=tmp_path / "profile" / "models",
        log_root=tmp_path / "profile" / "logs",
    )
    script = """
import hashlib
import json
from puripuly_heart import runtime_layout
from puripuly_heart.core.vad import bundled
from puripuly_heart.core.local_asr import local_stt_assets
from puripuly_heart.ui import fonts, i18n

layout = runtime_layout.current_runtime_layout()
notices = layout.package_resource("data", "THIRD_PARTY_NOTICES.txt")
license_path = layout.package_resource("data", "licenses", "PYTHON-3.14.7-LICENSE.txt")
model = bundled.bundled_silero_vad_onnx_path()
with model.open("rb") as handle:
    model_hash = hashlib.file_digest(handle, "sha256").hexdigest()
font = fonts.fonts_dir() / "NotoSansCJK-Medium.ttc"
manifest = local_stt_assets.load_local_stt_asset_manifest()
print(json.dumps({
    "host_kind": layout.host_kind,
    "origins": [runtime_layout.__file__, bundled.__file__, fonts.__file__, i18n.__file__,
                local_stt_assets.__file__],
    "model": str(model),
    "model_hash": model_hash,
    "font": str(font),
    "font_exists": font.is_file(),
    "font_url": fonts.font_asset_path(fonts.FONT_FAMILY_NOTO_SANS_CJK_JP),
    "notices": notices.read_text(encoding="utf-8"),
    "license": license_path.read_text(encoding="utf-8"),
    "manifest_id": manifest.model_id,
    "locales": i18n.available_locales(),
    "title": i18n.t("app.title"),
}))
"""
    completed = subprocess.run(
        [str(layout.python_executable), "-c", script],
        cwd=tmp_path,
        env=layout.python_child_environment(),
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
    assert result["model"] == str(app_root / "puripuly_heart" / resource_paths[0])
    assert result["model_hash"] == (
        "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"
    )
    assert result["font"] == str(app_root / "puripuly_heart" / resource_paths[1])
    assert result["font_exists"] is True
    assert result["font_url"] == "/fonts/NotoSansCJK-Medium.ttc"
    assert result["notices"] == (SOURCE_ROOT / resource_paths[2]).read_text(encoding="utf-8")
    assert result["license"] == (SOURCE_ROOT / resource_paths[3]).read_text(encoding="utf-8")
    assert result["manifest_id"] == "qwen3-asr-0.6b-int8-sherpa"
    assert result["locales"] == ["en", "ko"]
    assert (
        result["title"]
        == json.loads((SOURCE_ROOT / "data" / "i18n" / "en.json").read_text(encoding="utf-8"))[
            "app.title"
        ]
    )
