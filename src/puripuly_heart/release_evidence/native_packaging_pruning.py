from __future__ import annotations

import base64
import csv
import hashlib
import io
import stat
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
from typing import Any

PRUNING_SCHEMA = "puripuly-heart/native-packaging-pruning/v1"
ONNX_CAPI_RELATIVE_DIR = PurePosixPath("site-packages/onnxruntime/capi")
ONNX_PRELOAD_DLLS = ("onnxruntime.dll", "onnxruntime_providers_shared.dll")
SHERPA_DUPLICATE_DLLS = (
    "onnxruntime.dll",
    "sherpa-onnx-c-api.dll",
    "sherpa-onnx-cxx-api.dll",
)
SCIENTIFIC_TEST_ROOTS = {
    "numpy": (
        "numpy/_core/tests",
        "numpy/_pyinstaller/tests",
        "numpy/f2py/tests",
        "numpy/fft/tests",
        "numpy/lib/tests",
        "numpy/linalg/tests",
        "numpy/ma/tests",
        "numpy/matrixlib/tests",
        "numpy/polynomial/tests",
        "numpy/random/tests",
        "numpy/testing/tests",
        "numpy/tests",
        "numpy/typing/tests",
    ),
    "scipy": (
        "scipy/_lib/tests",
        "scipy/cluster/hierarchy/tests",
        "scipy/cluster/vq/tests",
        "scipy/constants/tests",
        "scipy/datasets/tests",
        "scipy/differentiate/tests",
        "scipy/fft/_duccfft/tests",
        "scipy/fft/tests",
        "scipy/fftpack/tests",
        "scipy/integrate/_ivp/tests",
        "scipy/integrate/tests",
        "scipy/interpolate/tests",
        "scipy/io/_harwell_boeing/tests",
        "scipy/io/arff/tests",
        "scipy/io/matlab/tests",
        "scipy/io/tests",
        "scipy/linalg/tests",
        "scipy/ndimage/tests",
        "scipy/odr/tests",
        "scipy/optimize/_trustregion_constr/tests",
        "scipy/optimize/tests",
        "scipy/signal/tests",
        "scipy/sparse/csgraph/tests",
        "scipy/sparse/linalg/_dsolve/tests",
        "scipy/sparse/linalg/_eigen/arpack/tests",
        "scipy/sparse/linalg/_eigen/lobpcg/tests",
        "scipy/sparse/linalg/_eigen/tests",
        "scipy/sparse/linalg/_isolve/tests",
        "scipy/sparse/linalg/tests",
        "scipy/sparse/tests",
        "scipy/spatial/tests",
        "scipy/spatial/transform/tests",
        "scipy/stats/tests",
    ),
}
_QUALIFIED_WHEELS = {
    "numpy": ("2.5.1", "cp314-cp314-win_amd64"),
    "scipy": ("1.18.0", "cp314-cp314-win_amd64"),
    "sherpa_onnx_core": ("1.13.8", "py3-none-win_amd64"),
    "onnxruntime": ("1.28.0", "cp314-cp314-win_amd64"),
}
_NATIVE_SUFFIXES = {".dll", ".pyd", ".so", ".dylib", ".lib", ".a"}


def _ordinary_path(root: Path, path: Path, *, directory: bool = False) -> None:
    relative = path.relative_to(root)
    for current in (root, *(root / parent for parent in reversed(relative.parents)), path):
        info = current.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError(f"pruning input cannot be a reparse path: {current}")
    mode = path.lstat().st_mode
    if not (stat.S_ISDIR(mode) if directory else stat.S_ISREG(mode)):
        raise ValueError(
            f"pruning input must be an ordinary {'directory' if directory else 'file'}: {path}"
        )


def _identity(root: Path, path: Path) -> dict[str, Any]:
    _ordinary_path(root, path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def _load_wheel(site_packages: Path, name: str) -> tuple[Path, list[list[str]], dict[str, Any]]:
    version, tag = _QUALIFIED_WHEELS[name]
    metadata_dirs = sorted(site_packages.glob(f"{name}-*.dist-info"))
    expected = site_packages / f"{name}-{version}.dist-info"
    if metadata_dirs != [expected]:
        raise ValueError(f"pruning requires exactly the qualified {name} {version} wheel")
    _ordinary_path(site_packages, expected, directory=True)
    identities = {
        filename: _identity(site_packages, expected / filename)
        for filename in ("METADATA", "WHEEL", "RECORD")
    }
    metadata = BytesParser().parsebytes((expected / "METADATA").read_bytes())
    wheel = BytesParser().parsebytes((expected / "WHEEL").read_bytes())
    if (
        metadata.get("Name", "").replace("-", "_").lower() != name
        or metadata.get("Version") != version
        or wheel.get_all("Tag") != [tag]
    ):
        raise ValueError(f"unqualified wheel identity for pruning: {name}")
    with (expected / "RECORD").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))
    seen = set()
    for row in rows:
        if len(row) != 3 or not row[0] or row[0].casefold() in seen:
            raise ValueError(f"invalid wheel RECORD for pruning: {name}")
        path = PurePosixPath(row[0])
        if (
            path.is_absolute()
            or str(path) != row[0]
            or any(part in {".", ".."} for part in path.parts)
            or any(character in row[0] for character in ("\\", ":", "\0"))
        ):
            raise ValueError(f"unsafe wheel RECORD path for pruning: {row[0]}")
        seen.add(row[0].casefold())
    return (
        expected,
        rows,
        {
            "name": name.replace("_", "-"),
            "version": version,
            "wheel_tag": tag,
            "source_metadata": identities,
        },
    )


def _owned_identity(
    root: Path, site_packages: Path, relative: str, rows: dict[str, list[str]]
) -> dict[str, Any]:
    identity = _identity(root, site_packages.joinpath(*PurePosixPath(relative).parts))
    digest = "sha256=" + base64.urlsafe_b64encode(bytes.fromhex(identity["sha256"])).rstrip(
        b"="
    ).decode("ascii")
    if rows.get(relative) != [relative, digest, str(identity["bytes"])]:
        raise ValueError(f"pruning input differs from its wheel RECORD: {relative}")
    return identity


def _retained_test_payload(path: Path) -> bool:
    return path.suffix.lower() in _NATIVE_SUFFIXES or any(
        token in path.name.upper() for token in ("LICENSE", "NOTICE", "COPYING", "COPYRIGHT")
    )


def prune_runtime_dependencies(target_root: Path) -> dict[str, Any]:
    target_root = target_root.absolute()
    _ordinary_path(target_root, target_root, directory=True)
    site_packages = target_root / "site-packages"
    _ordinary_path(target_root, site_packages, directory=True)
    if any(
        path.exists()
        for path in (
            target_root / "app/python.zip",
            target_root / "app/product_bootstrap.pyc",
            target_root / "python314.zip",
        )
    ):
        raise ValueError("dependency pruning must precede runtime compilation and archiving")
    if (target_root / "_runtime/local_qwen").exists():
        raise ValueError("obsolete ONNX preload directory must not be staged")
    wheels = {name: _load_wheel(site_packages, name) for name in _QUALIFIED_WHEELS}
    ownership = {name: {row[0]: row for row in value[1]} for name, value in wheels.items()}
    removals: dict[str, dict[str, Any]] = {}
    retained = []
    for name, test_roots in SCIENTIFIC_TEST_ROOTS.items():
        rows = ownership[name]
        for relative in test_roots:
            subtree = site_packages.joinpath(*PurePosixPath(relative).parts)
            if not subtree.exists() and not subtree.is_symlink():
                continue
            _ordinary_path(target_root, subtree, directory=True)
            for path in sorted(subtree.rglob("*")):
                if path.is_dir():
                    _ordinary_path(target_root, path, directory=True)
                    continue
                identity = _owned_identity(
                    target_root, site_packages, path.relative_to(site_packages).as_posix(), rows
                )
                if _retained_test_payload(path):
                    retained.append(identity)
                else:
                    removals[identity["path"]] = identity
    duplicates = []
    for dll in SHERPA_DUPLICATE_DLLS:
        rows = ownership["sherpa_onnx_core"]
        duplicate = _owned_identity(target_root, site_packages, f"Scripts/{dll}", rows)
        canonical = _owned_identity(target_root, site_packages, f"sherpa_onnx/lib/{dll}", rows)
        if (duplicate["bytes"], duplicate["sha256"]) != (canonical["bytes"], canonical["sha256"]):
            raise ValueError(f"sherpa duplicate differs from retained canonical DLL: {dll}")
        removals[duplicate["path"]] = duplicate
        duplicates.append({"removed": duplicate["path"], "canonical": canonical})
    preload = [
        _owned_identity(
            target_root, site_packages, f"onnxruntime/capi/{dll}", ownership["onnxruntime"]
        )
        for dll in ONNX_PRELOAD_DLLS
    ]
    removed_files = sorted(removals.values(), key=lambda row: row["path"])
    obsolete_files = sorted(
        [
            *removed_files,
            *(
                {**row, "path": f"_runtime/local_qwen/{dll}"}
                for dll, row in zip(ONNX_PRELOAD_DLLS, preload, strict=True)
            ),
        ],
        key=lambda row: row["path"],
    )
    rewritten_records = {}
    for metadata_dir, rows, _ in wheels.values():
        remaining = [row for row in rows if f"site-packages/{row[0]}" not in removals]
        if remaining == rows:
            continue
        record_relative = f"{metadata_dir.name}/RECORD"
        remaining = [[row[0], "", ""] if row[0] == record_relative else row for row in remaining]
        buffer = io.StringIO(newline="")
        csv.writer(buffer, lineterminator="\n").writerows(remaining)
        rewritten_records[metadata_dir / "RECORD"] = buffer.getvalue()
    for row in removed_files:
        target_root.joinpath(*PurePosixPath(row["path"]).parts).unlink()
    for test_roots in SCIENTIFIC_TEST_ROOTS.values():
        for relative in test_roots:
            subtree = site_packages.joinpath(*PurePosixPath(relative).parts)
            if subtree.is_dir():
                directories = [subtree, *(path for path in subtree.rglob("*") if path.is_dir())]
                for directory in sorted(
                    directories, key=lambda path: len(path.parts), reverse=True
                ):
                    if not any(directory.iterdir()):
                        directory.rmdir()
    for path, content in rewritten_records.items():
        path.write_text(content, encoding="utf-8", newline="")
    for row in obsolete_files:
        if target_root.joinpath(*PurePosixPath(row["path"]).parts).exists():
            raise ValueError(f"obsolete path remains in pruned staging: {row['path']}")
    return {
        "schema": PRUNING_SCHEMA,
        "removed_files": removed_files,
        "obsolete_files": obsolete_files,
        "source_wheels": [value[2] for value in wheels.values()],
        "retained_test_payloads": retained,
        "sherpa_duplicates": duplicates,
        "onnx_preload": {
            "canonical_files": preload,
            "obsolete_paths_staged": False,
        },
        "updated_records": [_identity(target_root, path) for path in rewritten_records],
    }
