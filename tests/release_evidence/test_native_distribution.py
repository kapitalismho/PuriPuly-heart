from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import marshal
import os
import py_compile
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
from packaging.requirements import Requirement

from puripuly_heart.release_evidence.native_distribution import (
    NativeArtifactLayout,
    bundle_runtime,
    compile_runtime,
    create_manifest,
    filter_requirements,
    finalize_soxr_wheel,
    main,
    stage_product_metadata,
    stage_sounddevice_portaudio_runtime,
    validate_dependencies,
    validate_target,
    verify_installed_soxr_record,
    verify_sounddevice_portaudio_runtime,
    verify_wheel_record,
)


def _record_digest(data: bytes) -> str:
    import base64

    return "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()


@pytest.mark.parametrize("archive_key", ["python_archive", "stdlib_archive"])
def test_native_layout_resolves_the_archive_and_requires_its_path(
    tmp_path: Path, archive_key: str
) -> None:
    root = Path(__file__).resolve().parents[2]
    source = root / "native/windows_host/artifact-layout.json"
    layout = NativeArtifactLayout.load(source)

    assert layout.resolve(tmp_path, "python_archive") == tmp_path / "app" / "python.zip"
    assert layout.resolve(tmp_path, "stdlib_archive") == tmp_path / "python314.zip"
    assert layout.resolve(tmp_path, "application_root") == tmp_path / "app"
    payload = json.loads(source.read_text(encoding="utf-8"))
    del payload[archive_key]
    incomplete = tmp_path / "layout.json"
    incomplete.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=archive_key):
        NativeArtifactLayout.load(incomplete)


@pytest.mark.parametrize(
    "change",
    [
        {"stdlib_archive": "Lib/python314.zip"},
        {"dependency_root": "app/vendor"},
        {"stdlib_root": "site-packages"},
        {"python_archive": "app/../python.zip"},
    ],
)
def test_native_layout_rejects_ambiguous_archive_ownership(
    tmp_path: Path, change: dict[str, str]
) -> None:
    root = Path(__file__).resolve().parents[2]
    source = root / "native/windows_host/artifact-layout.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload.update(change)
    altered = tmp_path / "layout.json"
    altered.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        NativeArtifactLayout.load(altered)


def test_requirement_filter_removes_viewer_and_replaces_soxr_as_whole_stanzas(
    tmp_path: Path,
) -> None:
    source = tmp_path / "requirements.txt"
    destination = tmp_path / "native.txt"
    source.write_text(
        "# generated\n"
        "flet==1.0.0 \\\n    --hash=sha256:flet\n"
        "flet-desktop==1.0.0 \\\n    --hash=sha256:viewer\n"
        "six==1.17.0 \\\n    --hash=sha256:six\n"
        "proc-tap==1.1.1 ; platform_machine == 'AMD64' \\\n    --hash=sha256:proc\n"
        "psutil==7.2.2 ; sys_platform == 'win32' \\\n    --hash=sha256:psutil\n"
        "scipy==1.18.0 ; platform_machine == 'AMD64' \\\n    --hash=sha256:scipy\n"
        "jeepney==0.9.0 ; sys_platform == 'linux' \\\n    --hash=sha256:jeepney\n"
        "soxr==1.1.0 \\\n    --hash=sha256:soxr\n"
        "    # via product\n",
        encoding="utf-8",
    )

    filter_requirements(source, destination)
    requirements = {
        requirement.name: requirement
        for line in destination.read_text(encoding="utf-8").splitlines()
        if line and not line[0].isspace() and not line.startswith("#")
        for requirement in [Requirement(line.rstrip().removesuffix("\\").strip())]
    }
    assert set(requirements) == {"flet", "six", "proc-tap", "psutil", "scipy"}
    for requirement in requirements.values():
        assert requirement.marker is None or requirement.marker.evaluate(
            {"platform_machine": "x86_64", "sys_platform": "win32"}
        )


@pytest.mark.parametrize(
    "installed",
    [
        {"numpy": "2.5.1", "unrelated": "1.0.0"},
        {"numpy": "2.5.1", "scipy": "1.17.0"},
    ],
    ids=["missing-scipy-with-same-distribution-count", "wrong-scipy-version"],
)
def test_native_dependencies_reject_incomplete_locked_closure(
    tmp_path: Path, installed: dict[str, str]
) -> None:
    requirements = tmp_path / "requirements.txt"
    requirements.write_text(
        "numpy==2.5.1\nscipy==1.18.0 ; platform_machine == 'AMD64' and sys_platform == 'win32'\n",
        encoding="utf-8",
    )
    site_packages = tmp_path / "site-packages"
    for name, version in installed.items():
        metadata = site_packages / f"{name}-{version}.dist-info"
        metadata.mkdir(parents=True)
        (metadata / "METADATA").write_text(
            f"Metadata-Version: 2.4\nName: {name}\nVersion: {version}\n", encoding="utf-8"
        )

    with pytest.raises(ValueError, match="scipy"):
        validate_dependencies(site_packages, requirements)


def test_native_dependencies_validate_windows_closure_and_custom_soxr(tmp_path: Path) -> None:
    requirements = tmp_path / "requirements.txt"
    requirements.write_text(
        "scipy==1.18.0 ; platform_machine == 'AMD64' and sys_platform == 'win32'\n"
        "soxr==1.1.0\n"
        "flet-desktop==1.0.0\n"
        "jeepney==0.9.0 ; sys_platform == 'linux'\n",
        encoding="utf-8",
    )
    installed = {"scipy": "1.18.0", "soxr": "1.1.0", "puripuly-heart": "2.8.0"}
    site_packages = tmp_path / "site-packages"
    for name, version in installed.items():
        metadata = site_packages / f"{name}-{version}.dist-info"
        metadata.mkdir(parents=True)
        (metadata / "METADATA").write_text(
            f"Metadata-Version: 2.4\nName: {name}\nVersion: {version}\n", encoding="utf-8"
        )

    report = validate_dependencies(site_packages, requirements)

    assert report["versions"] == installed


def test_finalized_soxr_wheel_owns_both_native_runtime_files(tmp_path: Path) -> None:
    source = tmp_path / "source.whl"
    destination = tmp_path / "soxr-1.1.0-cp312-abi3-win_amd64.whl"
    dll = tmp_path / "soxr.dll"
    pyd = b"native-extension"
    dll.write_bytes(b"runtime-dll")
    record_name = "soxr-1.1.0.dist-info/RECORD"
    members = {
        "soxr/__init__.py": b"",
        "soxr/soxr_ext.pyd": pyd,
        "soxr-1.1.0.dist-info/METADATA": b"Name: soxr\nVersion: 1.1.0\n",
    }
    rows = [[name, _record_digest(data), str(len(data))] for name, data in members.items()]
    rows.append([record_name, "", ""])
    with zipfile.ZipFile(source, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
        output = __import__("io").StringIO()
        writer = csv.writer(output, lineterminator="\n")
        writer.writerows(rows)
        archive.writestr(record_name, output.getvalue())

    finalize_soxr_wheel(source, dll, destination)
    verify_wheel_record(destination)
    installed = tmp_path / "site-packages"
    with zipfile.ZipFile(destination) as archive:
        archive.extractall(installed)
    installed_identities = verify_installed_soxr_record(installed)
    assert set(installed_identities) == {"soxr/soxr.dll", "soxr/soxr_ext.pyd"}

    with zipfile.ZipFile(destination) as archive:
        assert archive.read("soxr/soxr_ext.pyd") == pyd
        assert archive.read("soxr/soxr.dll") == b"runtime-dll"


def test_sounddevice_staging_excludes_asio_and_retains_standard_runtime(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "_sounddevice_data" / "portaudio-binaries"
    runtime.mkdir(parents=True)
    standard = runtime / "libportaudio64bit.dll"
    asio = runtime / "libportaudio64bit-asio.dll"
    standard.write_bytes(b"standard-portaudio")
    asio.write_bytes(b"unsupported-asio")

    result = stage_sounddevice_portaudio_runtime(tmp_path)

    assert standard.read_bytes() == b"standard-portaudio"
    assert not asio.exists()
    assert result["retained"] == {
        "_sounddevice_data/portaudio-binaries/libportaudio64bit.dll": hashlib.sha256(
            b"standard-portaudio"
        ).hexdigest()
    }
    assert result["excluded"] == {
        "_sounddevice_data/portaudio-binaries/libportaudio64bit-asio.dll": hashlib.sha256(
            b"unsupported-asio"
        ).hexdigest()
    }
    assert stage_sounddevice_portaudio_runtime(tmp_path) == {
        "retained": result["retained"],
        "excluded": {},
    }


def test_sounddevice_validation_rejects_asio_and_requires_standard_runtime(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "_sounddevice_data" / "portaudio-binaries"
    runtime.mkdir(parents=True)
    standard = runtime / "libportaudio64bit.dll"
    asio = runtime / "libportaudio64bit-asio.dll"
    standard.write_bytes(b"standard-portaudio")
    asio.write_bytes(b"unsupported-asio")

    with pytest.raises(ValueError):
        verify_sounddevice_portaudio_runtime(tmp_path)

    asio.unlink()
    standard.unlink()
    with pytest.raises(ValueError):
        verify_sounddevice_portaudio_runtime(tmp_path)


def test_product_metadata_is_non_editable_and_record_owned(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        '[project]\nname = "puripuly-heart"\nversion = "2.7.0"\ndescription = "Product"\n',
        encoding="utf-8",
    )
    site_packages = tmp_path / "site-packages"

    result = stage_product_metadata(site_packages, pyproject)

    metadata = site_packages / result["metadata"]
    assert (metadata / "INSTALLER").read_text(encoding="utf-8") == "native-experimental\n"
    assert not (metadata / "direct_url.json").exists()
    rows = list(csv.reader((metadata / "RECORD").read_text(encoding="utf-8").splitlines()))
    assert {row[0] for row in rows} == {
        f"{metadata.name}/INSTALLER",
        f"{metadata.name}/METADATA",
        f"{metadata.name}/RECORD",
        f"{metadata.name}/WHEEL",
    }


def test_runtime_imports_app_and_dependencies_without_source_reads(tmp_path: Path) -> None:
    app = tmp_path / "app"
    dependencies = tmp_path / "site-packages"
    app.mkdir()
    dependencies.mkdir()
    (app / "consumer.py").write_text(
        "from dependency import VALUE\nRESULT = VALUE + 1\n", encoding="utf-8"
    )
    (app / "product_bootstrap.py").write_text("RESULT = 9\n", encoding="utf-8")
    (dependencies / "dependency.py").write_text("VALUE = 41\n", encoding="utf-8")
    layout = Path(__file__).resolve().parents[2] / "native/windows_host/artifact-layout.json"

    compile_runtime(tmp_path, layout)

    script = (
        "import importlib.machinery, importlib.util, json, sys\n"
        f"sys.path[:0] = [{str(app)!r}, {str(dependencies)!r}]\n"
        "original_get_data = importlib.machinery.SourceFileLoader.get_data\n"
        "def reject_source_reads(self, path):\n"
        "    if path.endswith('.py'):\n"
        "        raise AssertionError('runtime source read')\n"
        "    return original_get_data(self, path)\n"
        "importlib.machinery.SourceFileLoader.get_data = reject_source_reads\n"
        "def reject_source_compilation(*args, **kwargs):\n"
        "    raise AssertionError('runtime source compilation')\n"
        "importlib.machinery.SourceFileLoader.source_to_code = reject_source_compilation\n"
        "import consumer\n"
        f"spec = importlib.util.spec_from_file_location('bootstrap', {str(app / 'product_bootstrap.pyc')!r})\n"
        "bootstrap = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(bootstrap)\n"
        "print(json.dumps([consumer.RESULT, bootstrap.RESULT]))\n"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout) == [42, 9]


def test_runtime_rebuild_replaces_changed_dependency_bytecode(tmp_path: Path) -> None:
    (tmp_path / "app").mkdir()
    dependencies = tmp_path / "site-packages"
    dependencies.mkdir()
    source = dependencies / "dependency.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    layout = Path(__file__).resolve().parents[2] / "native/windows_host/artifact-layout.json"

    compile_runtime(tmp_path, layout)
    source.write_text("VALUE = 2\n", encoding="utf-8")
    compile_runtime(tmp_path, layout)

    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            f"import sys; sys.path.insert(0, {str(dependencies)!r}); "
            "import dependency; print(dependency.VALUE)",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "2"


def test_runtime_build_rejects_uncompilable_dependency(tmp_path: Path) -> None:
    (tmp_path / "app").mkdir()
    dependencies = tmp_path / "site-packages"
    dependencies.mkdir()
    (dependencies / "dependency.py").write_text("def invalid(:\n", encoding="utf-8")
    layout = Path(__file__).resolve().parents[2] / "native/windows_host/artifact-layout.json"

    with pytest.raises(RuntimeError, match="SyntaxError"):
        compile_runtime(tmp_path, layout)


@pytest.fixture
def staged_archive_runtime(tmp_path: Path) -> tuple[Path, Path, dict[str, object]]:
    root = Path(__file__).resolve().parents[2]
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
        str(sdk_source), cfile=str(stdlib / "sdk_probe.pyc"),
        dfile="trusted-sdk/sdk_probe.py", doraise=True,
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
        "VALUE = 23\n"
        "def fail():\n"
        "    raise RuntimeError('physical source traceback')\n",
        encoding="utf-8",
    )
    (mixed / "_socket.py").write_text("raise AssertionError('native shadow lost')\n", encoding="utf-8")
    shutil.copy2(socket_spec.origin, mixed / Path(socket_spec.origin).name)
    (mixed / "payload.txt").write_text("mixed resource", encoding="utf-8")
    metadata = dependencies / "mixed-1.0.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text("Name: mixed\nVersion: 1.0\n", encoding="utf-8")
    (dependencies / "shared_ns/empty").mkdir(parents=True)
    (dependencies / "shared_ns/child.py").write_text("VALUE = 5\n", encoding="utf-8")
    (dependencies / "choice").mkdir()
    (dependencies / "choice/__init__.py").write_text("VALUE = 31\n", encoding="utf-8")
    (dependencies / "choice.py").write_text("raise AssertionError('module shadow lost')\n", encoding="utf-8")
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
    return layout, bytecode, evidence


def test_bundled_runtime_imports_code_and_flet_resources_without_recompilation(
    tmp_path: Path, staged_archive_runtime: tuple[Path, Path, dict[str, object]]
) -> None:
    layout, bytecode, _ = staged_archive_runtime
    bundled = bundle_runtime(tmp_path, layout, bytecode)
    app = tmp_path / "app"
    archive = app / "python.zip"
    dependencies = tmp_path / "site-packages"
    socket_spec = importlib.util.find_spec("_socket")
    assert socket_spec is not None and socket_spec.origin is not None
    assert (dependencies / Path(socket_spec.origin).name).read_bytes() == Path(
        socket_spec.origin
    ).read_bytes()
    assert (app / "product_bootstrap.pyc").is_file()
    assert not (app / "product_bootstrap.py").exists()
    assert list(app.rglob("*.py")) == []
    assert list(app.rglob("*.pyc")) == [app / "product_bootstrap.pyc"]
    assert not (dependencies / "flet").exists()
    assert (dependencies / "dependency.py").is_file()
    assert list(dependencies.glob("flet-*.dist-info"))
    assert list(dependencies.rglob("*.pyc")) == []
    assert list((tmp_path / "Lib").rglob("*.py")) == []
    assert list((tmp_path / "Lib").rglob("*.pyc")) == []
    assert (dependencies / "shared_ns/empty").is_dir()
    with zipfile.ZipFile(archive) as packed:
        names = packed.namelist()
        assert names == sorted(names)
        assert "puripuly_heart/ui/" in names
        assert "puripuly_heart/ui/consumer.pyc" in names
        assert "puripuly_heart/ui/consumer.py" in names
        assert "puripuly_heart/data/payload.txt" not in names
        assert not any(name.startswith("product_bootstrap.") for name in names)
        index = marshal.loads(packed.read("_native_dependencies.index"))["modules"]
        assert index["mixed"] == "mixed/__init__.py"
        assert index["choice"] == "choice/__init__.py"
        assert "mixed._socket" not in index
        assert "masked.child" not in index
        assert index["shared_ns.child"] == "shared_ns/child.py"
    stale = app / "puripuly_heart"
    (stale / "__init__.py").write_text(
        "raise AssertionError('stale app selected')\n", encoding="utf-8"
    )
    (stale / "ui" / "consumer.py").write_text(
        "raise AssertionError('stale module selected')\n", encoding="utf-8"
    )
    (dependencies / "flet").mkdir()
    (dependencies / "flet" / "__init__.py").write_text(
        "raise AssertionError('stale Flet selected')\n", encoding="utf-8"
    )
    script = (
        "import importlib.machinery, importlib.metadata, importlib.resources, importlib.util, inspect, json, os, pkgutil, sys, traceback, zipimport\n"
        f"sys.path[:0] = [{str(archive)!r}, {str(tmp_path / 'python314.zip')!r}, {str(app)!r}, {str(dependencies)!r}]\n"
        "from _puripuly_native_runtime import install\n"
        f"install({str(tmp_path)!r})\n"
        f"os.environ['PURIPULY_HEART_NATIVE_RESOURCE_ROOT'] = {str(app)!r}\n"
        "def reject_source_compilation(*args, **kwargs):\n"
        "    raise AssertionError('archive source compilation')\n"
        "zipimport._compile_source = reject_source_compilation\n"
        "original_get_data = importlib.machinery.SourceFileLoader.get_data\n"
        "def reject_source_reads(self, path):\n"
        f"    if path.endswith('.py') and (path.startswith({str(app)!r}) or path.startswith({str(dependencies)!r})):\n"
        "        raise AssertionError('loose runtime source read')\n"
        "    return original_get_data(self, path)\n"
        "importlib.machinery.SourceFileLoader.get_data = reject_source_reads\n"
        "import flet\n"
        "from puripuly_heart.ui import consumer\n"
        "import mixed, mixed.code, mixed._socket, shared_ns.child, choice, masked, sdk_probe, source_stdlib\n"
        "assert mixed.VALUE == 23 and mixed._socket.socket is not None\n"
        "assert shared_ns.child.VALUE == 5 and choice.VALUE == 31 and masked.VALUE == 17\n"
        "assert sdk_probe.VALUE == 77 and source_stdlib.VALUE == 12\n"
        "assert importlib.import_module('hyphen-module').VALUE == 7\n"
        "assert importlib.metadata.version('mixed') == '1.0'\n"
        "assert importlib.resources.files(mixed).joinpath('payload.txt').read_text() == 'mixed resource'\n"
        "assert importlib.resources.files(source_stdlib).joinpath('payload.txt').read_text() == 'stdlib resource'\n"
        f"assert mixed.__file__ == {str(dependencies / 'mixed/__init__.py')!r}\n"
        f"assert list(mixed.__path__) == [{str(dependencies / 'mixed')!r}]\n"
        f"assert mixed._socket.__file__ == {str(dependencies / 'mixed' / Path(socket_spec.origin).name)!r}\n"
        "assert {'code', '_socket'} <= {entry.name for entry in pkgutil.iter_modules(mixed.__path__)}\n"
        "importlib.machinery.SourceFileLoader.get_data = original_get_data\n"
        "assert \"raise RuntimeError('physical source traceback')\" in inspect.getsource(mixed.code.fail)\n"
        "try:\n"
        "    mixed.code.fail()\n"
        "except RuntimeError:\n"
        "    formatted = traceback.format_exc()\n"
        f"    assert {str(dependencies / 'mixed/code.py')!r} in formatted\n"
        "    assert \"raise RuntimeError('physical source traceback')\" in formatted\n"
        "from puripuly_heart.runtime_layout import current_runtime_layout\n"
        "layout = current_runtime_layout()\n"
        "icons = json.loads(importlib.resources.files('flet.controls.material').joinpath('icons.json').read_text())\n"
        "assert int(flet.Icons.ADD) == icons['ADD']\n"
        f"assert flet.__file__.startswith({str(archive)!r})\n"
        f"assert consumer.__file__.startswith({str(archive)!r})\n"
        f"spec = importlib.util.spec_from_file_location('bootstrap', {str(app / 'product_bootstrap.pyc')!r})\n"
        "bootstrap = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(bootstrap)\n"
        "print(json.dumps([consumer.RESULT, bootstrap.RESULT, "
        "layout.package_resource('data', 'payload.txt').read_text(), "
        "layout.resource('prompts', 'system.txt').read_text()]))\n"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", script], check=True, capture_output=True, text=True
    )
    assert json.loads(result.stdout) == [42, 9, "filesystem package data", "filesystem prompt"]
    assert {entry["source"] for entry in bundled["bytecode"]} == {"app/product_bootstrap.py"}


def test_bundle_cli_and_manifest_preserve_deployed_archive_and_member_evidence(
    tmp_path: Path, staged_archive_runtime: tuple[Path, Path, dict[str, object]]
) -> None:
    layout, bytecode, compiled = staged_archive_runtime
    assert (
        main(
            [
                "bundle-runtime",
                "--target-root",
                str(tmp_path),
                "--layout",
                str(layout),
                "--bytecode",
                str(bytecode),
                "--output",
                str(bytecode),
            ]
        )
        == 0
    )
    bundled = json.loads(bytecode.read_text(encoding="utf-8"))
    provenance = tmp_path / "provenance.json"
    provenance.write_text('{"build": "pinned"}', encoding="utf-8")
    manifest = create_manifest(tmp_path, layout, provenance, bytecode, tmp_path / "manifest.json")
    assert manifest["bytecode"] == bundled
    inventory = {entry["path"]: entry for entry in manifest["inventory"]}
    deployed = list(bundled["bytecode"])
    for key in ("archive", "stdlib_archive"):
        record = bundled[key]
        assert record["sha256"] == inventory[record["path"]]["sha256"]
        with zipfile.ZipFile(tmp_path / record["path"]) as archive:
            members = set(archive.namelist())
            assert {entry["path"] for entry in record["members"]} == members
            for entry in record["members"]:
                assert entry["sha256"] == hashlib.sha256(archive.read(entry["path"])).hexdigest()
                if entry["path"].endswith(".pyc"):
                    deployed.append(entry)
                    if entry["provenance"] == "build-derived":
                        source = (
                            archive.read(entry["source_member"]) if "source_member" in entry
                            else (tmp_path / entry["source"]).read_bytes()
                        )
                        assert entry["source_sha256"] == hashlib.sha256(source).hexdigest()
    original = {entry["source"]: entry for entry in compiled["bytecode"]}
    derived = [entry for entry in deployed if entry["provenance"] == "build-derived"]
    assert {entry["source"] for entry in derived} == original.keys()
    for entry in derived:
        for key in ("sha256", "source_sha256", "optimization", "invalidation_mode", "provenance"):
            assert entry[key] == original[entry["source"]][key]
    upstream = [entry for entry in deployed if entry["provenance"] == "trusted-upstream-sourceless"]
    assert len(upstream) == 1
    assert upstream[0]["path"] == "sdk_probe.pyc"
    for key in ("sha256", "input_path", "optimization", "invalidation_mode", "provenance"):
        assert upstream[0][key] == compiled["stdlib_bytecode"][0][key]
    assert upstream[0]["optimization"] == "upstream-unspecified"
    assert upstream[0]["invalidation_mode"] == "timestamp"
    assert "source" not in upstream[0] and "source_sha256" not in upstream[0]
    assert not any("/__pycache__/" in path and "/flet/" in path for path in inventory)
    assert "app/puripuly_heart/data/payload.txt" in inventory
    assert "app/prompts/system.txt" in inventory


def test_bundling_identical_inputs_produces_identical_archives(
    tmp_path: Path, staged_archive_runtime: tuple[Path, Path, dict[str, object]]
) -> None:
    layout, bytecode, _ = staged_archive_runtime
    second = tmp_path / "second"
    second.mkdir()
    shutil.copytree(tmp_path / "app", second / "app")
    shutil.copytree(tmp_path / "site-packages", second / "site-packages")
    shutil.copytree(tmp_path / "Lib", second / "Lib")
    shutil.copy2(bytecode, second / "bytecode.json")
    first = bundle_runtime(tmp_path, layout, bytecode)
    other = bundle_runtime(second, layout, second / "bytecode.json")
    assert first["archive"] == other["archive"]
    assert (tmp_path / "app/python.zip").read_bytes() == (second / "app/python.zip").read_bytes()
    assert first["stdlib_archive"] == other["stdlib_archive"]
    assert (tmp_path / "python314.zip").read_bytes() == (second / "python314.zip").read_bytes()


@pytest.mark.parametrize(
    "damage",
    [
        "missing-bytecode",
        "malformed-bytecode",
        "changed-source",
        "missing-evidence",
        "missing-icons",
        "missing-package-init",
        "invalid-archive-path",
        "existing-archive",
        "native-flet",
        "native-app",
        "native-app-data",
        "sourceless-module",
        "cache-source",
        "missing-upstream-evidence",
        "upstream-hash",
        "native-stdlib",
        "sourceless-dependency",
    ],
)
def test_bundle_failure_preserves_every_remaining_staged_input(
    tmp_path: Path, staged_archive_runtime: tuple[Path, Path, dict[str, object]], damage: str
) -> None:
    layout, bytecode, evidence = staged_archive_runtime
    if damage == "invalid-archive-path":
        payload = json.loads(layout.read_text(encoding="utf-8"))
        payload["python_archive"] = "site-packages/flet/python.zip"
        layout = tmp_path / "invalid-layout.json"
        layout.write_text(json.dumps(payload), encoding="utf-8")
    elif damage == "existing-archive":
        (tmp_path / "app/python.zip").write_bytes(b"existing archive")
    entry = next(
        item
        for item in evidence["bytecode"]
        if item["source"] == "app/puripuly_heart/ui/consumer.py"
    )
    compiled_path = tmp_path / entry["path"]
    if damage == "missing-bytecode":
        compiled_path.unlink()
    elif damage == "malformed-bytecode":
        data = compiled_path.read_bytes()[:16] + b"not a marshalled module"
        compiled_path.write_bytes(data)
        entry["sha256"] = hashlib.sha256(data).hexdigest()
    elif damage == "changed-source":
        (tmp_path / entry["source"]).write_text("RESULT = 123\n", encoding="utf-8")
    elif damage == "missing-evidence":
        evidence["bytecode"].remove(entry)
    elif damage == "missing-icons":
        (tmp_path / "site-packages/flet/controls/material/icons.json").unlink()
    elif damage == "missing-package-init":
        (tmp_path / "site-packages/flet/__init__.py").unlink()
    elif damage == "native-flet":
        (tmp_path / "site-packages/flet/unsupported.pyd").write_bytes(b"native")
    elif damage == "native-app":
        (tmp_path / "app/puripuly_heart/unsupported.pyd").write_bytes(b"native")
    elif damage == "native-app-data":
        (tmp_path / "app/puripuly_heart/data/unsupported.pyd").write_bytes(b"native")
    elif damage == "sourceless-module":
        (tmp_path / "app/puripuly_heart/orphan.pyc").write_bytes(b"orphan")
    elif damage == "cache-source":
        (tmp_path / "app/puripuly_heart/__pycache__/orphan.py").write_text(
            "RESULT = 1\n", encoding="utf-8"
        )
    elif damage == "missing-upstream-evidence":
        evidence["stdlib_bytecode"].clear()
    elif damage == "upstream-hash":
        evidence["stdlib_bytecode"][0]["sha256"] = "0" * 64
    elif damage == "native-stdlib":
        (tmp_path / "Lib/native.pyd").write_bytes(b"native")
    elif damage == "sourceless-dependency":
        (tmp_path / "site-packages/orphan.pyc").write_bytes(b"orphan")
    bytecode.write_text(json.dumps(evidence), encoding="utf-8")
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    with pytest.raises((ValueError, FileNotFoundError, FileExistsError)):
        bundle_runtime(tmp_path, layout, bytecode)
    assert {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before
    if damage == "existing-archive":
        assert (tmp_path / "app/python.zip").read_bytes() == b"existing archive"
    else:
        assert not (tmp_path / "app/python.zip").exists()
    assert not (tmp_path / "app/python.zip.tmp").exists()
    assert not (tmp_path / "python314.zip").exists()
    assert not (tmp_path / "python314.zip.tmp").exists()


@pytest.mark.parametrize(
    "damage", [
        "archive-hash", "member-hash", "missing-member", "loose-hash",
        "stdlib-hash", "upstream-provenance", "dependency-provenance", "index-provenance",
    ],
)
def test_manifest_rejects_inconsistent_deployed_bytecode_evidence(
    tmp_path: Path, staged_archive_runtime: tuple[Path, Path, dict[str, object]], damage: str
) -> None:
    layout, bytecode, _ = staged_archive_runtime
    bundled = bundle_runtime(tmp_path, layout, bytecode)
    if damage == "archive-hash":
        bundled["archive"]["sha256"] = "0" * 64
    elif damage == "member-hash":
        bundled["archive"]["members"][0]["sha256"] = "0" * 64
    elif damage == "missing-member":
        bundled["archive"]["members"].pop()
    elif damage == "stdlib-hash":
        bundled["stdlib_archive"]["sha256"] = "0" * 64
    elif damage == "upstream-provenance":
        entry = next(
            entry for entry in bundled["stdlib_archive"]["members"]
            if entry["provenance"] == "trusted-upstream-sourceless"
        )
        entry["provenance"] = "build-derived"
        entry["source_sha256"] = "0" * 64
    elif damage in {"dependency-provenance", "index-provenance"}:
        name = (
            "_native_dependencies/dependency.pyc" if damage == "dependency-provenance"
            else "_native_dependencies.index"
        )
        entry = next(entry for entry in bundled["archive"]["members"] if entry["path"] == name)
        entry["provenance"] = "build-input"
    else:
        bundled["bytecode"][0]["sha256"] = "0" * 64
    bytecode.write_text(json.dumps(bundled), encoding="utf-8")
    provenance = tmp_path / "provenance.json"
    provenance.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="evidence"):
        create_manifest(tmp_path, layout, provenance, bytecode, tmp_path / "manifest.json")


def test_validate_target_requires_the_deployed_python_archive(
    tmp_path: Path, staged_archive_runtime: tuple[Path, Path, dict[str, object]]
) -> None:
    layout_path, _, _ = staged_archive_runtime
    layout = NativeArtifactLayout.load(layout_path)
    for key in (
        "host_executable",
        "console_executable",
        "python_executable",
        "overlay_executable",
        "gpu_worker_executable",
        "openvr_dll",
    ):
        layout.resolve(tmp_path, key).write_bytes(b"placeholder")
    layout.resolve(tmp_path, "extension_dll_root").mkdir()
    with pytest.raises(FileNotFoundError, match="python.zip"):
        validate_target(tmp_path, layout_path, tmp_path / "requirements.txt", tmp_path / "vc.json")

@pytest.mark.skipif(sys.version_info[:2] != (3, 14), reason="native SDK uses CPython 3.14")
def test_stdlib_archive_supports_early_interpreter_bootstrap_without_loose_lib(
    tmp_path: Path, staged_archive_runtime: tuple[Path, Path, dict[str, object]]
) -> None:
    import encodings

    layout, bytecode, _ = staged_archive_runtime
    shutil.copytree(
        Path(encodings.__file__).parent, tmp_path / "Lib/encodings", dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    bytecode.write_text(json.dumps(compile_runtime(tmp_path, layout)), encoding="utf-8")
    bundle_runtime(tmp_path, layout, bytecode)
    shutil.rmtree(tmp_path / "Lib")
    archive = tmp_path / "python314.zip"
    executable = sys.executable
    if sys.platform == "win32":
        executable = str(tmp_path / "python.exe")
        shutil.copy2(sys._base_executable, executable)
        shutil.copy2(Path(sys.base_prefix) / "python314.dll", tmp_path / "python314.dll")
    else:
        (tmp_path / "lib").mkdir()
        shutil.copy2(archive, tmp_path / "lib/python314.zip")
        archive = tmp_path / "lib/python314.zip"
    environment = dict(os.environ)
    for name in tuple(environment):
        if name.startswith("PYTHON") or name.startswith("PURIPULY_HEART_NATIVE_"):
            del environment[name]
    environment.update(PYTHONHOME=str(tmp_path), PYTHONUTF8="1")
    script = (
        "import encodings, sys, sdk_probe\n"
        f"assert encodings.__file__.startswith({str(archive)!r})\n"
        "assert sdk_probe.VALUE == 77\n"
        "assert sys.flags.no_site == 1\n"
        "print('stdlib archive bootstrap')\n"
    )
    script_path = tmp_path / "bootstrap_probe.py"
    script_path.write_text(script, encoding="utf-8")
    result = subprocess.run(
        [executable, "-S", "-B", str(script_path)], cwd=tmp_path, env=environment,
        capture_output=True, text=True, encoding="utf-8", check=True, timeout=30,
    )
    assert result.stdout.strip() == "stdlib archive bootstrap"
    assert not (tmp_path / "Lib").exists()


@pytest.mark.parametrize(
    "damage", [
        "duplicate", "case-collision", "traversal", "native", "malformed-index",
        "index-traversal", "index-module-alias", "native-shadow", "package-shadow",
        "missing-code", "malformed-code", "stdlib-native", "stdlib-malformed",
        "file-directory-collision", "stdlib-disguised-native",
    ],
)
def test_manifest_rejects_corrupt_or_ambiguous_archives(
    tmp_path: Path, staged_archive_runtime: tuple[Path, Path, dict[str, object]], damage: str
) -> None:
    layout, bytecode, _ = staged_archive_runtime
    bundled = bundle_runtime(tmp_path, layout, bytecode)
    stdlib = damage.startswith("stdlib-")
    archive_path = tmp_path / bundled["stdlib_archive" if stdlib else "archive"]["path"]
    with zipfile.ZipFile(archive_path) as original:
        members = [(info, original.read(info)) for info in original.infolist()]
    index = next((data for info, data in members if info.filename == "_native_dependencies.index"), None)
    changes = {}
    additions = []
    omit = set()
    if damage in {"duplicate", "case-collision"}:
        name = "_native_dependencies/dependency.pyc"
        data = next(data for info, data in members if info.filename == name)
        additions.append((name if damage == "duplicate" else name.upper(), data))
    elif damage == "traversal":
        additions.append(("../outside.pyc", b"unsafe"))
    elif damage in {"native", "stdlib-native"}:
        additions.append(("native.pyd", b"native binary"))
    elif damage == "stdlib-disguised-native":
        additions.append(("payload.bin", b"MZ\0\0native binary"))
    elif damage == "file-directory-collision":
        additions.extend((("ambiguous", b"resource"), ("Ambiguous/resource.txt", b"resource")))
    elif damage == "malformed-index":
        changes["_native_dependencies.index"] = b"not marshal"
    elif damage in {"index-traversal", "index-module-alias", "native-shadow", "package-shadow"}:
        payload = marshal.loads(index)
        if damage == "index-traversal":
            payload["modules"]["dependency"] = "../dependency.py"
        elif damage == "index-module-alias":
            payload["modules"]["alias"] = "dependency.py"
        elif damage == "native-shadow":
            payload["modules"]["mixed._socket"] = "mixed/_socket.py"
        else:
            payload["modules"]["choice"] = "choice.py"
        changes["_native_dependencies.index"] = marshal.dumps(payload)
    elif damage == "missing-code":
        omit.add("_native_dependencies/dependency.pyc")
    else:
        name = "sdk_probe.pyc" if stdlib else "_native_dependencies/dependency.pyc"
        original = next(data for info, data in members if info.filename == name)
        changes[name] = original[:16] + b"not code"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for info, data in members:
            if info.filename not in omit:
                archive.writestr(info, changes.get(info.filename, data))
        for name, data in additions:
            if damage == "duplicate":
                with pytest.warns(UserWarning, match="Duplicate name"):
                    archive.writestr(name, data)
            else:
                archive.writestr(name, data)
    bytecode.write_text(json.dumps(bundled), encoding="utf-8")
    provenance = tmp_path / "provenance.json"
    provenance.write_text("{}", encoding="utf-8")
    with pytest.raises((ValueError, FileNotFoundError)):
        create_manifest(tmp_path, layout, provenance, bytecode, tmp_path / "manifest.json")


@pytest.mark.parametrize(
    "damage", ["retained-source", "loose-dependency", "loose-stdlib", "loose-stdlib-resource"]
)
def test_manifest_rejects_source_drift_and_parallel_loose_content(
    tmp_path: Path, staged_archive_runtime: tuple[Path, Path, dict[str, object]], damage: str
) -> None:
    layout, bytecode, _ = staged_archive_runtime
    bundled = bundle_runtime(tmp_path, layout, bytecode)
    if damage == "retained-source":
        (tmp_path / "site-packages/dependency.py").write_text("VALUE = 99\n", encoding="utf-8")
    else:
        relative = (
            "site-packages/stale.pyc" if damage == "loose-dependency"
            else "Lib/stale.txt" if damage == "loose-stdlib-resource" else "Lib/stale.pyc"
        )
        destination = tmp_path / relative
        destination.parent.mkdir(exist_ok=True)
        destination.write_bytes(b"stale bytecode")
    bytecode.write_text(json.dumps(bundled), encoding="utf-8")
    provenance = tmp_path / "provenance.json"
    provenance.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        create_manifest(tmp_path, layout, provenance, bytecode, tmp_path / "manifest.json")



def test_embedded_bootstrap_only_persists_bounded_uncaught_error_diagnostics(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[2]
    template = (root / "native/windows_host/python_bootstrap.py.in").read_text(encoding="utf-8")
    modules = tmp_path / "site-packages"
    modules.mkdir()
    bridge_output = tmp_path / "bridge.json"
    (modules / "certifi.py").write_text("def where(): return __file__\n", encoding="utf-8")
    (modules / "dart_bridge.py").write_text(
        "import os\n"
        "def send_bytes(port, payload):\n"
        "    open(os.environ['BRIDGE_OUTPUT'], 'wb').write(payload)\n",
        encoding="utf-8",
    )
    app = tmp_path / "app"
    app.mkdir()
    for filename in ("_puripuly_native_runtime.py", "sitecustomize.py"):
        shutil.copy2(root / "native/windows_host" / filename, app / filename)
    archive_path = app / "python.zip"

    def invoke(
        module_source: str,
        *,
        profile: Path | None = None,
    ) -> tuple[subprocess.CompletedProcess[str], dict[str, object], Path]:
        (modules / "probe.py").write_text(module_source, encoding="utf-8")
        compiled = compile_runtime(tmp_path, root / "native/windows_host/artifact-layout.json")
        index = {}
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for entry in compiled["bytecode"]:
                source = Path(entry["source"])
                if source.parts[0] == "site-packages":
                    relative = source.relative_to("site-packages").as_posix()
                    index[source.stem] = relative
                    member = "_native_dependencies/" + str(Path(relative).with_suffix(".pyc")).replace("\\", "/")
                else:
                    member = source.with_suffix(".pyc").name
                archive.write(tmp_path / entry["path"], member)
            archive.writestr("_native_dependencies.index", marshal.dumps({"version": 1, "modules": index}))
        script = (
            template.replace("{argv}", "['PuriPulyHeart']")
            .replace("{host_executable}", repr(str(tmp_path / "PuriPulyHeart.exe")))
            .replace("{module_name}", repr("probe"))
            .replace("{error_exit_code}", "255")
        )
        bridge_output.unlink(missing_ok=True)
        profile = profile or tmp_path / "profile"
        environment = {
            "PYTHONPATH": os.pathsep.join((str(archive_path), str(modules), str(root / "src"))),
            "LOCALAPPDATA": str(profile),
            "PURIPULY_HEART_NATIVE_RESOURCE_ROOT": str(app),
            "PURIPULY_HEART_NATIVE_RUNTIME_ROOT": str(tmp_path),
            "PURIPULY_HEART_NATIVE_HOST_EXECUTABLE": str(tmp_path / "PuriPulyHeart.exe"),
            "PURIPULY_HEART_NATIVE_PYTHON_EXECUTABLE": sys.executable,
            "FLET_DART_BRIDGE_EXIT_PORT": "7",
            "BRIDGE_OUTPUT": str(bridge_output),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONIOENCODING": "utf-8",
        }
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=tmp_path,
            env=environment,
            text=True,
            encoding="utf-8",
            capture_output=True,
            check=True,
        )
        payload = json.loads(bridge_output.read_text(encoding="utf-8"))
        return (
            completed,
            payload,
            profile / "puripuly-heart" / "logs" / "native-bootstrap-error.log",
        )

    normal, normal_payload, error_path = invoke("print('ordinary output')\n")
    assert normal.stdout == "ordinary output\n"
    assert normal_payload == {"code": 0, "error": ""}
    assert not error_path.exists()

    system_exit, system_exit_payload, error_path = invoke("raise SystemExit(23)\n")
    assert system_exit.stderr == ""
    assert system_exit_payload == {"code": 23, "error": ""}
    assert not error_path.exists()

    failed, failed_payload, error_path = invoke(
        "def fail():\n    raise RuntimeError('한' * 100000)\nfail()\n"
    )
    encoded = failed_payload["error"].encode("utf-8")
    assert failed_payload["code"] == 255
    assert len(encoded) <= 65536
    assert "builtins.RuntimeError" in failed_payload["error"]
    assert "probe.py" in failed_payload["error"]
    assert error_path.read_bytes() == encoded
    assert "closed file" not in failed.stderr.lower()

    blocked_profile = tmp_path / "blocked-profile"
    blocked_profile.write_text("not a directory", encoding="utf-8")
    _, blocked_payload, blocked_error_path = invoke(
        "raise LookupError('bridge survives log failure')\n",
        profile=blocked_profile,
    )
    assert blocked_payload["code"] == 255
    assert "builtins.LookupError" in blocked_payload["error"]
    assert not blocked_error_path.exists()


@pytest.fixture
def bootstrap_https_server(tmp_path: Path):
    import datetime
    import ipaddress
    import ssl
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "bootstrap-test")])
    now = datetime.datetime.now(datetime.UTC)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=1))
        .not_valid_after(now + datetime.timedelta(days=1))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .add_extension(
            x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    ca_path = tmp_path / "ca.pem"
    ca_path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    key_path = tmp_path / "key.pem"
    key_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"verified")

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(ca_path, key_path)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield ca_path, f"https://127.0.0.1:{server.server_port}/"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize(
    "policy,httpx_result,requests_result",
    [
        ("file", "verified", "SSLError"),
        ("directory", "ConnectError", "SSLError"),
        ("requests", "ConnectError", "verified"),
        ("curl", "ConnectError", "verified"),
        ("unset", "ConnectError", "SSLError"),
        ("invalid-file", "FileNotFoundError", "SSLError"),
        ("invalid-requests", "ConnectError", "OSError"),
    ],
)
def test_embedded_bootstrap_preserves_independent_ca_policy_and_tls_verification(
    tmp_path: Path,
    bootstrap_https_server,
    policy: str,
    httpx_result: str,
    requests_result: str,
) -> None:
    import certifi

    root = Path(__file__).resolve().parents[2]
    ca_path, url = bootstrap_https_server
    ca_directory = tmp_path / "empty-ca-directory"
    ca_directory.mkdir()
    policies = {
        "file": {"SSL_CERT_FILE": str(ca_path)},
        "directory": {"SSL_CERT_DIR": str(ca_directory)},
        "requests": {"REQUESTS_CA_BUNDLE": str(ca_path)},
        "curl": {"CURL_CA_BUNDLE": str(ca_path)},
        "unset": {},
        "invalid-file": {"SSL_CERT_FILE": str(tmp_path / "missing-ca.pem")},
        "invalid-requests": {"REQUESTS_CA_BUNDLE": str(tmp_path / "missing-ca.pem")},
    }
    inherited = policies[policy]
    modules = tmp_path / "startup"
    modules.mkdir()
    (modules / "dart_bridge.py").write_text(
        "import json\n"
        "def send_bytes(port, payload):\n"
        "    print(json.dumps({'bootstrap': json.loads(payload)}))\n",
        encoding="utf-8",
    )
    (modules / "_puripuly_native_runtime.py").write_text(
        "def install(root): pass\n", encoding="utf-8"
    )
    ca_keys = ("SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE")
    (modules / "probe.py").write_text(
        "import json, os\n"
        "import httpx, requests\n"
        f"result = {{'environment': {{key: os.environ.get(key) for key in {ca_keys!r}}}}}\n"
        "for name, library in [('httpx', httpx), ('requests', requests)]:\n"
        "    try:\n"
        f"        response = library.get({url!r}, timeout=5)\n"
        "        response.raise_for_status()\n"
        "        result[name] = response.text\n"
        "    except Exception as exc:\n"
        "        result[name] = type(exc).__name__\n"
        "print(json.dumps(result))\n",
        encoding="utf-8",
    )
    template = (root / "native/windows_host/python_bootstrap.py.in").read_text(encoding="utf-8")
    script = (
        template.replace("{argv}", "['PuriPulyHeart']")
        .replace("{host_executable}", repr(str(tmp_path / "PuriPulyHeart.exe")))
        .replace("{module_name}", repr("probe"))
        .replace("{error_exit_code}", "255")
    )
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in ca_keys and not key.upper().endswith("_PROXY")
    }
    environment.update(
        inherited,
        PYTHONPATH=os.pathsep.join((str(modules), str(root / "src"))),
        LOCALAPPDATA=str(tmp_path / "profile"),
        PURIPULY_HEART_NATIVE_RUNTIME_ROOT=str(tmp_path),
        FLET_DART_BRIDGE_EXIT_PORT="7",
        PYTHONDONTWRITEBYTECODE="1",
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=environment,
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=True,
    )
    result, bridge = map(json.loads, completed.stdout.splitlines())
    expected = dict.fromkeys(ca_keys)
    expected.update(inherited)
    if "SSL_CERT_FILE" not in inherited and "SSL_CERT_DIR" not in inherited:
        expected["SSL_CERT_FILE"] = certifi.where()
    if "REQUESTS_CA_BUNDLE" not in inherited and "CURL_CA_BUNDLE" not in inherited:
        expected["REQUESTS_CA_BUNDLE"] = certifi.where()
    assert result["environment"] == expected
    assert result["httpx"] == httpx_result
    assert result["requests"] == requests_result
    assert bridge == {"bootstrap": {"code": 0, "error": ""}}
