from __future__ import annotations

import base64
import csv
import hashlib
import json
import marshal
import zipfile
from pathlib import Path

import pytest

from puripuly_heart.release_evidence.native_distribution import (
    NativeArtifactLayout,
    bundle_runtime,
    compile_runtime,
    create_manifest,
    main,
)
from puripuly_heart.release_evidence.native_packaging_pruning import (
    ONNX_PRELOAD_DLLS,
    PRUNING_SCHEMA,
    SCIENTIFIC_TEST_ROOTS,
    SHERPA_DUPLICATE_DLLS,
    prune_runtime_dependencies,
)


def _write(root: Path, name: str, data: bytes) -> None:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _wheel(root: Path, name: str, version: str, tag: str, files: dict[str, bytes]) -> None:
    metadata = f"{name}-{version}.dist-info"
    files = {
        **files,
        f"{metadata}/METADATA": f"Name: {name.replace('_', '-')}\nVersion: {version}\n".encode(),
        f"{metadata}/WHEEL": f"Wheel-Version: 1.0\nTag: {tag}\n".encode(),
        f"{metadata}/licenses/LICENSE": b"license payload",
    }
    rows = []
    for path, data in files.items():
        _write(root, path, data)
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        rows.append([path, f"sha256={digest}", str(len(data))])
    rows.append([f"{metadata}/RECORD", "", ""])
    with (root / metadata / "RECORD").open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream, lineterminator="\n").writerows(rows)


@pytest.fixture
def staged_dependencies(tmp_path: Path) -> Path:
    root = tmp_path / "site-packages"
    for name, version in (("numpy", "2.5.1"), ("scipy", "1.18.0")):
        files = {f"{name}/__init__.py": b"VALUE = 1\n"}
        for subtree in SCIENTIFIC_TEST_ROOTS[name]:
            files[f"{subtree}/test_contract.py"] = b"raise AssertionError('excluded suite')\n"
            files[f"{subtree}/data/input.bin"] = b"test fixture"
        if name == "numpy":
            files.update(
                {
                    "numpy/testing/__init__.py": b"VALUE = 42\n",
                    "numpy/testing/_private/utils.py": b"VALUE = 42\n",
                    "numpy/testing/overrides.py": b"VALUE = 42\n",
                    "numpy/_pytesttester.py": b"VALUE = 42\n",
                    "numpy/ma/testutils.py": b"VALUE = 42\n",
                    "numpy/ma/LICENSE": b"license payload",
                    "numpy/random/LICENSE.md": b"license payload",
                    "numpy/_core/_multiarray_umath.pyd": b"native runtime",
                    "numpy/_core/tests/NOTICE.txt": b"retained test notice",
                    "numpy/_core/tests/native.pyd": b"retained native file",
                    "numpy/future/tests/test_unknown.py": b"VALUE = 42\n",
                }
            )
        else:
            files.update(
                {
                    "scipy/_lib/_testutils.py": b"VALUE = 42\n",
                    "scipy/_lib/_uarray/LICENSE": b"license payload",
                    "scipy/fft/_duccfft/LICENSE.md": b"license payload",
                    "scipy/signal/_signaltools.py": b"VALUE = 42\n",
                    "scipy/special/tests/test_hyp2f1.py": b"VALUE = 42\n",
                    "scipy/special/tests/data/input.bin": b"retained test fixture",
                    "scipy/.libs/runtime.dll": b"native runtime",
                }
            )
        _wheel(root, name, version, "cp314-cp314-win_amd64", files)
    sherpa = {}
    for dll in SHERPA_DUPLICATE_DLLS:
        sherpa[f"Scripts/{dll}"] = dll.encode()
        sherpa[f"sherpa_onnx/lib/{dll}"] = dll.encode()
    for library in ("sherpa-onnx-c-api.lib", "sherpa-onnx-cxx-api.lib"):
        sherpa[f"Scripts/{library}"] = b"retained import library"
        sherpa[f"sherpa_onnx/lib/{library}"] = b"retained import library"
    sherpa["Scripts/unrelated.dll"] = b"not selected"
    _wheel(root, "sherpa_onnx_core", "1.13.8", "py3-none-win_amd64", sherpa)
    _wheel(
        root,
        "onnxruntime",
        "1.28.0",
        "cp314-cp314-win_amd64",
        {
            **{
                f"onnxruntime/capi/{dll}": f"distinct Python ORT {dll}".encode()
                for dll in ONNX_PRELOAD_DLLS
            },
            "onnxruntime/LICENSE": b"license payload",
            "onnxruntime/ThirdPartyNotices.txt": b"notice payload",
        },
    )
    _write(root, "unrelated/tests/test_keep.py", b"VALUE = 42\n")
    _write(root, "sherpa_onnx-1.13.8.dist-info/licenses/LICENSE", b"wrapper license")
    return tmp_path


def _files(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_pruning_removes_only_qualified_files_and_rewrites_owned_records(
    staged_dependencies: Path,
) -> None:
    root = staged_dependencies
    before = _files(root)

    evidence = prune_runtime_dependencies(root)

    after = _files(root)
    removed = {row["path"]: row for row in evidence["removed_files"]}
    assert evidence["schema"] == PRUNING_SCHEMA
    assert set(before) - set(after) == removed.keys()
    assert len([path for path in removed if path.startswith("site-packages/Scripts/")]) == 3
    for path, row in removed.items():
        assert set(row) == {"path", "bytes", "sha256"}
        assert row["bytes"] == len(before[path])
        assert row["sha256"] == hashlib.sha256(before[path]).hexdigest()
    obsolete = {row["path"]: row for row in evidence["obsolete_files"]}
    assert obsolete.keys() - removed.keys() == {
        f"_runtime/local_qwen/{dll}" for dll in ONNX_PRELOAD_DLLS
    }
    for dll in ONNX_PRELOAD_DLLS:
        old = obsolete[f"_runtime/local_qwen/{dll}"]
        canonical = after[f"site-packages/onnxruntime/capi/{dll}"]
        assert old["bytes"] == len(canonical)
        assert old["sha256"] == hashlib.sha256(canonical).hexdigest()
    assert evidence["onnx_preload"]["obsolete_paths_staged"] is False
    assert not obsolete.keys() & after.keys()
    updated = {row["path"] for row in evidence["updated_records"]}
    assert {path for path in before.keys() & after.keys() if before[path] != after[path]} == updated
    assert len(updated) == 3
    assert {row["name"]: row["version"] for row in evidence["source_wheels"]} == {
        "numpy": "2.5.1",
        "scipy": "1.18.0",
        "sherpa-onnx-core": "1.13.8",
        "onnxruntime": "1.28.0",
    }
    for wheel in evidence["source_wheels"]:
        for row in wheel["source_metadata"].values():
            path = f"site-packages/{row['path']}"
            assert row["sha256"] == hashlib.sha256(before[path]).hexdigest()
    for record in updated:
        rows = list(csv.reader(after[record].decode().splitlines()))
        assert [record.removeprefix("site-packages/"), "", ""] in rows
        for path, digest, size in rows:
            assert f"site-packages/{path}" in after
            if digest:
                data = after[f"site-packages/{path}"]
                assert (
                    digest
                    == "sha256="
                    + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
                )
                assert size == str(len(data))
    for path, data in before.items():
        if path not in removed and path not in updated:
            assert after[path] == data
    assert (
        after["site-packages/sherpa_onnx/lib/onnxruntime.dll"]
        != after["site-packages/onnxruntime/capi/onnxruntime.dll"]
    )
    assert after["site-packages/numpy/testing/__init__.py"] == b"VALUE = 42\n"
    assert after["site-packages/scipy/special/tests/test_hyp2f1.py"] == b"VALUE = 42\n"
    assert {row["path"] for row in evidence["retained_test_payloads"]} == {
        "site-packages/numpy/_core/tests/native.pyd",
        "site-packages/numpy/_core/tests/NOTICE.txt",
    }


@pytest.mark.parametrize(
    "failure",
    [
        "version",
        "tag",
        "record",
        "canonical",
        "nonordinary",
        "duplicate",
        "unowned",
        "compiled",
        "archived",
        "old-preload",
    ],
)
def test_pruning_preflight_preserves_every_input_on_rejection(
    staged_dependencies: Path, failure: str
) -> None:
    root = staged_dependencies
    site = root / "site-packages"
    if failure == "version":
        _write(site, "numpy-9.0.dist-info/METADATA", b"Name: numpy\nVersion: 9.0\n")
    elif failure == "tag":
        _write(site, "scipy-1.18.0.dist-info/WHEEL", b"Tag: cp314-cp314-linux_x86_64\n")
    elif failure == "record":
        _write(site, "numpy/tests/test_contract.py", b"modified")
    elif failure == "canonical":
        (site / "sherpa_onnx/lib/onnxruntime.dll").unlink()
    elif failure == "nonordinary":
        (site / "sherpa_onnx/lib/onnxruntime.dll").unlink()
        (site / "sherpa_onnx/lib/onnxruntime.dll").mkdir()
    elif failure == "duplicate":
        rows = list(
            csv.reader((site / "sherpa_onnx_core-1.13.8.dist-info/RECORD").read_text().splitlines())
        )
        _wheel(
            site,
            "sherpa_onnx_core",
            "1.13.8",
            "py3-none-win_amd64",
            {
                row[0]: (
                    b"different"
                    if row[0] == "Scripts/onnxruntime.dll"
                    else (site / row[0]).read_bytes()
                )
                for row in rows
                if not row[0].startswith("sherpa_onnx_core-")
            },
        )
    elif failure == "unowned":
        _write(site, "scipy/signal/tests/unknown.py", b"VALUE = 42\n")
    elif failure == "compiled":
        _write(root, "app/product_bootstrap.pyc", b"compiled")
    elif failure == "archived":
        _write(root, "app/python.zip", b"archive")
    elif failure == "old-preload":
        _write(root, "_runtime/local_qwen/onnxruntime.dll", b"obsolete staging")
    before = _files(root)

    with pytest.raises((ValueError, FileNotFoundError)):
        prune_runtime_dependencies(root)

    assert _files(root) == before


def test_pruning_exclusions_flow_into_bytecode_archive_index_and_final_inventory(
    staged_dependencies: Path,
) -> None:
    root = staged_dependencies
    layout_path = Path(__file__).resolve().parents[2] / "native/windows_host/artifact-layout.json"
    for path, data in {
        "app/puripuly_heart/__init__.py": b"",
        "app/product_bootstrap.py": b"VALUE = 1\n",
        "site-packages/flet/__init__.py": b"",
        "site-packages/flet/controls/material/icons.json": b"{}",
        "Lib/runtime_probe.py": b"VALUE = 42\n",
        "Lib/encodings/__init__.py": b"",
    }.items():
        _write(root, path, data)
    evidence_path = root / "pruning.json"
    assert main(["prune-runtime", "--target-root", str(root), "--output", str(evidence_path)]) == 0
    pruning = json.loads(evidence_path.read_text(encoding="utf-8"))
    compiled = compile_runtime(root, layout_path)
    bytecode_path = root / "bytecode.json"
    bytecode_path.write_text(json.dumps(compiled), encoding="utf-8")

    bundled = bundle_runtime(root, layout_path, bytecode_path)
    manifest = create_manifest(root, layout_path, root / "manifest.json")

    excluded = {row["path"] for row in pruning["obsolete_files"]}
    assert not excluded & {row["path"] for row in manifest["inventory"]}
    assert not excluded & {row["source"] for row in compiled["bytecode"]}
    with zipfile.ZipFile(root / bundled["archive"]["path"]) as archive:
        names = set(archive.namelist())
        index = marshal.loads(archive.read("_native_dependencies.index"))["modules"]
    assert not any("test_contract" in name for name in names | set(index) | set(index.values()))
    assert "_native_dependencies/numpy/testing/__init__.pyc" in names
    assert "_native_dependencies/scipy/_lib/_testutils.pyc" in names
    assert "_native_dependencies/scipy/special/tests/test_hyp2f1.pyc" in names
    assert "_native_dependencies/unrelated/tests/test_keep.pyc" in names
    assert (
        NativeArtifactLayout.load(layout_path).resolve(root, "local_sherpa_runtime_root")
        == root / "site-packages/onnxruntime/capi"
    )


def test_layout_rejects_the_obsolete_onnx_preload_directory(tmp_path: Path) -> None:
    source = Path(__file__).resolve().parents[2] / "native/windows_host/artifact-layout.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["local_sherpa_runtime_root"] = "_runtime/local_qwen"
    candidate = tmp_path / "layout.json"
    candidate.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="canonical ONNX Runtime capi"):
        NativeArtifactLayout.load(candidate)
