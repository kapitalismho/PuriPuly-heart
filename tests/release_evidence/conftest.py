from __future__ import annotations

import importlib.util
import json
import py_compile
import shutil
from pathlib import Path

import pytest

from puripuly_heart.release_evidence.native_distribution import (
    compile_runtime,
    stage_product_metadata,
)
from tests.helpers.paths import REPO_ROOT


@pytest.fixture(scope="session")
def compiled_archive_runtime(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    tmp_path = tmp_path_factory.mktemp("native-compiled-runtime")
    root = REPO_ROOT
    layout = root / "native/windows_host/artifact-layout.json"
    app = tmp_path / "app"
    package = app / "puripuly_heart"
    (package / "ui").mkdir(parents=True)
    (package / "config").mkdir()
    (package / "data").mkdir()
    (app / "prompts").mkdir()
    for relative in ("__init__.py", "runtime_layout.py", "config/paths.py"):
        shutil.copy2(root / "src/puripuly_heart" / relative, package / relative)
    (package / "ui" / "consumer.py").write_text(
        "from dependency import VALUE\nRESULT = VALUE + 1\n", encoding="utf-8"
    )
    (package / "data" / "payload.txt").write_text("filesystem package data", encoding="utf-8")
    (app / "prompts" / "system.txt").write_text("filesystem prompt", encoding="utf-8")
    (app / "product_bootstrap.py").write_text("RESULT = 9\n", encoding="utf-8")
    for filename in ("_puripuly_native_runtime.py", "sitecustomize.py"):
        shutil.copy2(root / "native/windows_host" / filename, app / filename)
    stdlib = tmp_path / "Lib"
    (stdlib / "encodings").mkdir(parents=True)
    (stdlib / "encodings/__init__.py").write_text("", encoding="utf-8")
    (stdlib / "source_stdlib").mkdir()
    (stdlib / "source_stdlib/__init__.py").write_text("VALUE = 12\n", encoding="utf-8")
    (stdlib / "source_stdlib/payload.txt").write_text("stdlib resource", encoding="utf-8")
    sdk_source = tmp_path / "sdk_input.py"
    sdk_source.write_text("VALUE = 77\n", encoding="utf-8")
    py_compile.compile(
        str(sdk_source),
        cfile=str(stdlib / "sdk_probe.pyc"),
        dfile="trusted-sdk/sdk_probe.py",
        doraise=True,
        invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP,
    )
    sdk_source.unlink()
    dependencies = tmp_path / "site-packages"
    dependencies.mkdir()
    (dependencies / "dependency.py").write_text("VALUE = 41\n", encoding="utf-8")
    (dependencies / "hyphen-module.py").write_text("VALUE = 7\n", encoding="utf-8")
    socket_spec = importlib.util.find_spec("_socket")
    assert socket_spec is not None and socket_spec.origin is not None
    shutil.copy2(socket_spec.origin, dependencies / Path(socket_spec.origin).name)
    mixed = dependencies / "mixed"
    mixed.mkdir()
    (mixed / "__init__.py").write_text(
        "from . import _socket\nfrom .code import VALUE\n", encoding="utf-8"
    )
    (mixed / "code.py").write_text(
        "VALUE = 23\n" "def fail():\n" "    raise RuntimeError('physical source traceback')\n",
        encoding="utf-8",
    )
    (mixed / "_socket.py").write_text(
        "raise AssertionError('native shadow lost')\n", encoding="utf-8"
    )
    shutil.copy2(socket_spec.origin, mixed / Path(socket_spec.origin).name)
    (mixed / "payload.txt").write_text("mixed resource", encoding="utf-8")
    metadata = dependencies / "mixed-1.0.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text("Name: mixed\nVersion: 1.0\n", encoding="utf-8")
    (dependencies / "shared_ns/empty").mkdir(parents=True)
    (dependencies / "shared_ns/child.py").write_text("VALUE = 5\n", encoding="utf-8")
    (dependencies / "choice").mkdir()
    (dependencies / "choice/__init__.py").write_text("VALUE = 31\n", encoding="utf-8")
    (dependencies / "choice.py").write_text(
        "raise AssertionError('module shadow lost')\n", encoding="utf-8"
    )
    (dependencies / "masked").mkdir()
    (dependencies / "masked.py").write_text("VALUE = 17\n", encoding="utf-8")
    (dependencies / "masked/child.py").write_text("VALUE = 99\n", encoding="utf-8")
    flet_spec = importlib.util.find_spec("flet")
    assert flet_spec is not None and flet_spec.origin is not None
    flet_root = Path(flet_spec.origin).parent
    shutil.copytree(
        flet_root, dependencies / "flet", ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    for metadata in flet_root.parent.glob("flet-*.dist-info"):
        shutil.copytree(metadata, dependencies / metadata.name)
    stage_product_metadata(dependencies, root / "pyproject.toml")
    evidence = compile_runtime(tmp_path, layout)
    bytecode = tmp_path / "bytecode.json"
    bytecode.write_text(json.dumps(evidence), encoding="utf-8")
    return tmp_path, layout


@pytest.fixture
def staged_archive_runtime(
    tmp_path: Path, compiled_archive_runtime: tuple[Path, Path]
) -> tuple[Path, Path, dict[str, object]]:
    source, layout = compiled_archive_runtime
    shutil.copytree(source, tmp_path, dirs_exist_ok=True)
    bytecode = tmp_path / "bytecode.json"
    return layout, bytecode, json.loads(bytecode.read_text(encoding="utf-8"))
