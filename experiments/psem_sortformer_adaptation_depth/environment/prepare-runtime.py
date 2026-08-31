from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import re
import shutil
from pathlib import Path

NVIMGCODEC_WHEEL = Path(
    "/usr/local/lib/python3.12/dist-packages/nvidia_nvimgcodec_cu12-0.3.0.5.dist-info/WHEEL"
)
NVIMGCODEC_BEFORE_SHA256 = "124ca2b3e0011af3bd2428db0c594788fb2f96285b8ebf167ac840082efae0e8"
NVIMGCODEC_AFTER_SHA256 = "5d826861768326f9be8fe374338189f457ddeab673762d6539b45e012014b9dc"
SYSTEM_METADATA = (
    Path("/usr/lib/python3/dist-packages/cryptography.egg-info"),
    Path("/usr/lib/python3/dist-packages/pyparsing-3.1.1.dist-info"),
    Path("/usr/lib/python3/dist-packages/six-1.16.0.egg-info"),
)
SETUPTOOLS_VENDOR_ROOT = Path("/usr/local/lib/python3.12/dist-packages/setuptools/_vendor")
SETUPTOOLS_VENDOR_METADATA_SUFFIXES = (".dist-info", ".egg-info")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def tree_identity(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    count = 0
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(child.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(child.read_bytes())
        digest.update(b"\0")
        count += 1
    return {"sha256": digest.hexdigest(), "file_count": count}


def canonical_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def installed_distribution_records() -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for distribution in importlib.metadata.distributions():
        raw_name = distribution.metadata.get("Name")
        version = distribution.version
        metadata_path = getattr(distribution, "_path", None)
        if (
            not isinstance(raw_name, str)
            or not isinstance(version, str)
            or not version
            or metadata_path is None
        ):
            raise RuntimeError("installed distribution metadata is incomplete")
        records.append(
            {
                "name": canonical_name(raw_name),
                "version": version,
                "metadata_path": str(Path(metadata_path).resolve()),
            }
        )
    return sorted(
        records,
        key=lambda row: (row["name"], row["version"], row["metadata_path"]),
    )


def installed_versions(
    records: list[dict[str, str]] | None = None,
) -> dict[str, str]:
    result: dict[str, str] = {}
    for record in records if records is not None else installed_distribution_records():
        name = record["name"]
        version = record["version"]
        if name in result and result[name] != version:
            raise RuntimeError(f"installed distribution name is ambiguous: {name}")
        result[name] = version
    return result


def protected_inventory() -> dict[str, object]:
    versions = installed_versions()
    packages = {
        name: version
        for name, version in versions.items()
        if name in {"torch", "torchvision", "triton"} or name.startswith("nvidia-")
    }
    if "torch" not in packages or not any(name.startswith("nvidia-") for name in packages):
        raise RuntimeError("NGC torch/CUDA protected package inventory is incomplete")
    import torch

    extension = Path(torch._C.__file__).resolve()
    return {
        "packages": dict(sorted(packages.items())),
        "torch_extension": {
            "path": str(extension),
            "size": extension.stat().st_size,
            "sha256": sha256_file(extension),
        },
        "torch_cuda_version": torch.version.cuda,
    }


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def repair_nvimgcodec() -> None:
    if sha256_file(NVIMGCODEC_WHEEL) != NVIMGCODEC_BEFORE_SHA256:
        raise RuntimeError("nvimgcodec WHEEL metadata does not match the pinned NGC base")
    text = NVIMGCODEC_WHEEL.read_text(encoding="utf-8")
    old = "Tag: py3-manylinux2014_x86_64"
    new = "Tag: py3-none-manylinux2014_x86_64"
    if text.count(old) != 1:
        raise RuntimeError("expected exactly one malformed nvimgcodec wheel tag")
    NVIMGCODEC_WHEEL.write_text(text.replace(old, new), encoding="utf-8")
    if sha256_file(NVIMGCODEC_WHEEL) != NVIMGCODEC_AFTER_SHA256:
        raise RuntimeError("nvimgcodec WHEEL repair produced unexpected bytes")


def snapshot(output: Path, constraint_output: Path) -> None:
    distribution_records = installed_distribution_records()
    record_payload = (
        json.dumps(distribution_records, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    value = {
        "schema_version": 1,
        "artifact_role": "ngc_base_inventory",
        "all_packages": dict(sorted(installed_versions(distribution_records).items())),
        "all_distribution_records": {
            "count": len(distribution_records),
            "records": distribution_records,
            "sha256": hashlib.sha256(record_payload).hexdigest(),
        },
        "inventory": protected_inventory(),
    }
    write_json(output, value)
    packages = value["inventory"]["packages"]
    constraints = "".join(f"{name}=={version}\n" for name, version in packages.items())
    constraint_output.write_text(constraints, encoding="utf-8")


def move_metadata(source: Path, backup_root: Path) -> dict[str, object]:
    if source.is_dir() and not backup_root.exists():
        identity = tree_identity(source)
        backup_root.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(backup_root))
        return {
            "source": str(source),
            "backup": str(backup_root),
            "action": "moved",
            **identity,
        }
    if not source.exists() and backup_root.is_dir():
        return {
            "source": str(source),
            "backup": str(backup_root),
            "action": "already_moved",
            **tree_identity(backup_root),
        }
    if not source.exists() and not backup_root.exists():
        return {
            "source": str(source),
            "backup": None,
            "action": "already_absent",
            "sha256": None,
            "file_count": 0,
        }
    raise RuntimeError(f"unexpected metadata state: {source}")


def isolate_metadata(receipt: Path) -> None:
    backup_root = Path("/opt/psem/disabled-metadata")
    moved = []
    for source in SYSTEM_METADATA:
        moved.append(move_metadata(source, backup_root / "system" / source.name))
    vendor_sources = sorted(
        source
        for source in SETUPTOOLS_VENDOR_ROOT.iterdir()
        if source.name.endswith(SETUPTOOLS_VENDOR_METADATA_SUFFIXES)
    )
    for source in vendor_sources:
        moved.append(move_metadata(source, backup_root / "setuptools-vendor" / source.name))
    write_json(
        receipt,
        {
            "schema_version": 1,
            "artifact_role": "runtime_metadata_isolation",
            "moved": moved,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("repair-nvimgcodec")
    snapshot_parser = commands.add_parser("snapshot")
    snapshot_parser.add_argument("--output", type=Path, required=True)
    snapshot_parser.add_argument("--constraint-output", type=Path, required=True)
    isolate_parser = commands.add_parser("isolate-metadata")
    isolate_parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "repair-nvimgcodec":
        repair_nvimgcodec()
    elif args.command == "snapshot":
        snapshot(args.output, args.constraint_output)
    else:
        isolate_metadata(args.receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
