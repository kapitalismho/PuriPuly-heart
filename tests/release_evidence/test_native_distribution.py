from __future__ import annotations

import csv
import hashlib
import json
import marshal
import shutil
import subprocess
import sys
import zipfile
from importlib.metadata import Distribution
from pathlib import Path

import pytest
from packaging.requirements import Requirement

from puripuly_heart.release_evidence.native_distribution import (
    NativeArtifactLayout,
    _archive_members,
    bundle_runtime,
    compile_runtime,
    create_manifest,
    filter_requirements,
    finalize_soxr_wheel,
    main,
    stage_product_metadata,
    stage_sounddevice_portaudio_runtime,
    validate_target,
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

    with zipfile.ZipFile(destination) as archive:
        assert archive.read("soxr/soxr_ext.pyd") == pyd
        assert archive.read("soxr/soxr.dll") == b"runtime-dll"
        rows = list(csv.reader(archive.read(record_name).decode("utf-8").splitlines()))
        assert {row[0] for row in rows} == set(archive.namelist())
        for name, digest, size in rows:
            if name == record_name:
                assert (digest, size) == ("", "")
            else:
                data = archive.read(name)
                assert digest == _record_digest(data)
                assert size == str(len(data))


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
    assert result["retained"] == ["_sounddevice_data/portaudio-binaries/libportaudio64bit.dll"]
    assert result["excluded"] == ["_sounddevice_data/portaudio-binaries/libportaudio64bit-asio.dll"]
    assert stage_sounddevice_portaudio_runtime(tmp_path) == {
        "retained": result["retained"],
        "excluded": [],
    }


def test_sounddevice_staging_requires_standard_runtime(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="standard sounddevice PortAudio runtime"):
        stage_sounddevice_portaudio_runtime(tmp_path)


def test_product_metadata_is_non_editable_and_record_owned(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        '[project]\nname = "puripuly-heart"\nversion = "2.7.0"\ndescription = "Product"\n',
        encoding="utf-8",
    )
    site_packages = tmp_path / "site-packages"

    result = stage_product_metadata(site_packages, pyproject)

    assert result["name"] == "puripuly-heart"
    assert result["version"] == "2.7.0"
    metadata = site_packages / result["metadata"]
    distribution = Distribution.at(metadata)
    assert distribution.metadata["Name"] == "puripuly-heart"
    assert distribution.version == "2.7.0"
    assert distribution.metadata["Summary"] == "Product"
    assert {path.name for path in metadata.iterdir()} == {
        "METADATA",
        "WHEEL",
        "INSTALLER",
        "RECORD",
    }
    assert not list(site_packages.glob("*.egg-link"))
    assert not list(site_packages.glob("*.pth"))
    assert not (metadata / "direct_url.json").exists()
    rows = list(csv.reader((metadata / "RECORD").read_text(encoding="utf-8").splitlines()))
    assert len(rows) == len({row[0] for row in rows})
    assert {row[0] for row in rows} == {
        path.relative_to(site_packages).as_posix() for path in metadata.iterdir()
    }
    for name, digest, size in rows:
        if name == f"{metadata.name}/RECORD":
            assert (digest, size) == ("", "")
        else:
            data = (site_packages / name).read_bytes()
            assert data
            assert digest == _record_digest(data)
            assert size == str(len(data))


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


@pytest.mark.parametrize(
    "runtime_fixture",
    ["staged_archive_runtime", "staged_compact_archive_runtime"],
    ids=["full-flet", "compact-flet"],
)
def test_bundle_cli_and_manifest_preserve_required_payload(
    tmp_path: Path, request: pytest.FixtureRequest, runtime_fixture: str
) -> None:
    layout, bytecode, compiled = request.getfixturevalue(runtime_fixture)
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
    manifest = create_manifest(tmp_path, layout, tmp_path / "manifest.json")
    assert set(manifest) == {"schema", "layout_schema", "inventory"}
    inventory = {entry["path"]: entry for entry in manifest["inventory"]}
    for key in ("archive", "stdlib_archive"):
        record = bundled[key]
        with zipfile.ZipFile(tmp_path / record["path"]) as archive:
            assert {entry["path"] for entry in record["members"]} == set(archive.namelist())
        assert (
            inventory[record["path"]]["sha256"]
            == hashlib.sha256((tmp_path / record["path"]).read_bytes()).hexdigest()
        )
    assert bundled["bytecode"] == [
        entry for entry in compiled["bytecode"] if entry["source"] == "app/product_bootstrap.py"
    ]
    assert "sdk_probe.pyc" in {entry["path"] for entry in bundled["stdlib_archive"]["members"]}
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
        "native-stdlib",
        "sourceless-dependency",
    ],
)
def test_bundle_failure_preserves_every_remaining_staged_input(
    tmp_path: Path,
    staged_compact_archive_runtime: tuple[Path, Path, dict[str, object]],
    damage: str,
) -> None:
    layout, bytecode, evidence = staged_compact_archive_runtime
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
        "asr_worker_executable",
        "openvr_dll",
    ):
        layout.resolve(tmp_path, key).write_bytes(b"placeholder")
    layout.resolve(tmp_path, "extension_dll_root").mkdir()
    with pytest.raises(FileNotFoundError, match="python.zip"):
        validate_target(
            tmp_path,
            layout_path,
            tmp_path / "soxr.json",
        )


@pytest.mark.parametrize(
    "damage",
    [
        "duplicate",
        "case-collision",
        "traversal",
        "native",
        "malformed-index",
        "index-traversal",
        "index-module-alias",
        "native-shadow",
        "package-shadow",
        "missing-code",
        "malformed-code",
        "stdlib-native",
        "stdlib-malformed",
        "file-directory-collision",
        "stdlib-disguised-native",
    ],
)
def test_archive_composition_rejects_corrupt_or_ambiguous_members(
    tmp_path: Path,
    staged_compact_bundled_runtime: tuple[Path, Path, dict[str, object]],
    damage: str,
) -> None:
    layout, bytecode, bundled = staged_compact_bundled_runtime
    stdlib = damage.startswith("stdlib-")
    archive_path = tmp_path / bundled["stdlib_archive" if stdlib else "archive"]["path"]
    with zipfile.ZipFile(archive_path) as original:
        members = [(info, original.read(info)) for info in original.infolist()]
    index = next(
        (data for info, data in members if info.filename == "_native_dependencies.index"), None
    )
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
    expected_error = {
        "duplicate": "duplicate or ambiguous members",
        "case-collision": "duplicate or ambiguous members",
        "traversal": "normalized relative path",
        "native": "cannot contain native modules",
        "malformed-index": "malformed dependency index",
        "index-traversal": "normalized relative path",
        "index-module-alias": "invalid dependency index source",
        "native-shadow": "differs from filesystem import precedence",
        "package-shadow": "differs from filesystem import precedence",
        "missing-code": "dependency index lacks archived bytecode",
        "malformed-code": "malformed bytecode",
        "stdlib-native": "cannot contain native modules",
        "stdlib-malformed": "malformed bytecode",
        "file-directory-collision": "file/directory collision",
        "stdlib-disguised-native": "cannot contain native binaries",
    }[damage]
    with pytest.raises(ValueError, match=expected_error):
        _archive_members(archive_path, stdlib=stdlib, dependency_root=tmp_path / "site-packages")
