from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from puripuly_heart.release_evidence.native_distribution import prepare_python_inputs


@pytest.fixture
def python_inputs(tmp_path: Path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    version = "3.14.10"
    members = {
        "python.exe": b"patched executable",
        "python3.dll": b"patched stable ABI",
        "python314.dll": b"patched CPython core",
        "DLLs/_ssl.pyd": b"patched SSL extension",
        "DLLs/libssl-3.dll": b"patched OpenSSL",
        "DLLs/libcrypto-3.dll": b"patched crypto",
        "Lib/ssl.py": b"VALUE = 'patched stdlib'\n",
        "LICENSE.txt": b"unchanged upstream license",
    }
    package = inputs / f"python.{version}.nupkg"
    with zipfile.ZipFile(package, "w") as archive:
        for name, data in members.items():
            archive.writestr("tools/" + name, data)
        archive.writestr("tools/include/Python.h", b"not a runtime input")
        archive.writestr("tools/Lib/site-packages/pip/__init__.py", b"build-only installer")
        archive.writestr("tools/Lib/ensurepip/__init__.py", b"build-only bootstrap")
        archive.writestr("tools/Lib/venv/scripts/nt/python.exe", b"unsupported embedded launcher")
    manifest = {
        "pythons": {"3.14": {"full_version": "3.14.7", "standalone_release_date": "20260901"}},
        "dart_bridge_version": "reviewed-bridge",
    }
    (inputs / "manifest-20260908.json").write_text(json.dumps(manifest), encoding="utf-8")
    spec = {
        "versions": {
            "python": version,
            "python_build_release": "20260908",
            "python_dist_release": "20261026",
        },
    }
    spec_path = tmp_path / "upstream.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    return inputs, spec_path, members


def test_python_cache_derivation_replaces_the_whole_runtime_without_old_inputs(
    tmp_path: Path,
    python_inputs,
):
    inputs, spec, members = python_inputs
    report = prepare_python_inputs(spec, inputs, tmp_path / "cache")
    with zipfile.ZipFile(report["runtime_archive"]) as archive:
        assert set(archive.namelist()) == set(members)
        assert {name: archive.read(name) for name in archive.namelist()} == members
    manifest = json.loads(Path(report["manifest"]).read_text(encoding="utf-8"))
    assert manifest["pythons"]["3.14"] == {
        "full_version": "3.14.10",
        "standalone_release_date": "20261026",
    }
    assert manifest["dart_bridge_version"] == "reviewed-bridge"
