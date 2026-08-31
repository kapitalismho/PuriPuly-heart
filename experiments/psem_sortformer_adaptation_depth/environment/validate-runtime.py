from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path

BASE_IMAGE = (
    "nvcr.io/nvidia/pytorch@sha256:0981807f1a51a156563e28b59dc2e7a9b5c1c7d85d1169d4965c5fd91fa38bcb"
)
NEMO_REVISION = "1a3c291b3ef0f0e11b72f789b185e1f1bda39bd6"
TORCHAUDIO_REVISION = "d8831425203385077a03c1d92cfbbe3bf2106008"
NVIMGCODEC_WHEEL = Path(
    "/usr/local/lib/python3.12/dist-packages/nvidia_nvimgcodec_cu12-0.3.0.5.dist-info/WHEEL"
)
NVIMGCODEC_AFTER_SHA256 = "5d826861768326f9be8fe374338189f457ddeab673762d6539b45e012014b9dc"
BASE_INVENTORY = Path("/opt/psem/base-protected-inventory.json")
RUNTIME_CONSTRAINTS = Path("/opt/psem/runtime-constraints.txt")
EXPECTED_RUNTIME_CONSTRAINT_COUNT = 81
NEMO_CHECKOUT = Path("/opt/nemo")
EXPECTED_PACKAGES = {
    "six": "1.16.0",
    "numpy": "1.26.4",
    "pandas": "2.2.2",
    "pyarrow": "17.0.0",
    "scipy": "1.14.1",
    "pyannote-core": "5.0.0",
    "pyannote-database": "5.1.0",
    "pyannote-metrics": "3.2.1",
    "datasets": "3.3.0",
    "fsspec": "2024.12.0",
    "dill": "0.3.8",
    "multiprocess": "0.70.16",
    "torchaudio": "2.6.0a0+d883142",
}
DISABLED_METADATA_SOURCES = (
    Path("/usr/lib/python3/dist-packages/cryptography.egg-info"),
    Path("/usr/lib/python3/dist-packages/pyparsing-3.1.1.dist-info"),
    Path("/usr/lib/python3/dist-packages/six-1.16.0.egg-info"),
    Path(
        "/usr/local/lib/python3.12/dist-packages/setuptools/_vendor/"
        "importlib_metadata-8.7.1.dist-info"
    ),
    Path(
        "/usr/local/lib/python3.12/dist-packages/setuptools/_vendor/more_itertools-10.8.0.dist-info"
    ),
    Path("/usr/local/lib/python3.12/dist-packages/setuptools/_vendor/packaging-26.0.dist-info"),
    Path("/usr/local/lib/python3.12/dist-packages/setuptools/_vendor/platformdirs-4.4.0.dist-info"),
    Path("/usr/local/lib/python3.12/dist-packages/setuptools/_vendor/wheel-0.46.3.dist-info"),
    Path("/usr/local/lib/python3.12/dist-packages/setuptools/_vendor/zipp-3.23.0.dist-info"),
    Path("/usr/local/lib/python3.12/dist-packages/setuptools/_vendor/autocommand-2.2.2.dist-info"),
    Path(
        "/usr/local/lib/python3.12/dist-packages/setuptools/_vendor/"
        "backports.tarfile-1.2.0.dist-info"
    ),
    Path(
        "/usr/local/lib/python3.12/dist-packages/setuptools/_vendor/jaraco_context-6.1.0.dist-info"
    ),
    Path(
        "/usr/local/lib/python3.12/dist-packages/setuptools/_vendor/"
        "jaraco_functools-4.4.0.dist-info"
    ),
    Path("/usr/local/lib/python3.12/dist-packages/setuptools/_vendor/jaraco.text-4.0.0.dist-info"),
    Path("/usr/local/lib/python3.12/dist-packages/setuptools/_vendor/tomli-2.4.0.dist-info"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


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
        if name in result:
            raise RuntimeError(f"duplicate installed distribution metadata: {name}")
        result[name] = record["version"]
    return result


def load_runtime_constraints() -> dict[str, str]:
    result: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        RUNTIME_CONSTRAINTS.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw_line or raw_line.startswith("#"):
            continue
        match = re.fullmatch(r"([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s]+)", raw_line)
        if match is None:
            raise RuntimeError(f"invalid runtime constraint at line {line_number}")
        name = canonical_name(match.group(1))
        if name in result:
            raise RuntimeError(f"duplicate runtime constraint: {name}")
        result[name] = match.group(2)
    if len(result) != EXPECTED_RUNTIME_CONSTRAINT_COUNT:
        raise RuntimeError("runtime constraint closure has an unexpected package count")
    return result


def inventory_difference(expected: dict[str, str], observed: dict[str, str]) -> dict[str, object]:
    return {
        "missing": sorted(set(expected) - set(observed)),
        "unexpected": sorted(set(observed) - set(expected)),
        "mismatched": {
            name: {"expected": expected[name], "observed": observed[name]}
            for name in sorted(set(expected) & set(observed))
            if expected[name] != observed[name]
        },
    }


def protected_inventory() -> dict[str, object]:
    import torch

    versions = installed_versions()
    packages = {
        name: version
        for name, version in versions.items()
        if name in {"torch", "torchvision", "triton"} or name.startswith("nvidia-")
    }
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


def assert_source_identity() -> dict[str, object]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=NEMO_CHECKOUT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=NEMO_CHECKOUT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    if head != NEMO_REVISION or dirty:
        raise RuntimeError("NeMo checkout is not the clean pinned revision")
    from nemo.collections.asr.models.sortformer_diar_models import SortformerEncLabelModel
    from nemo.collections.asr.parts.utils.asr_multispeaker_utils import get_ats_targets

    origins = {
        "SortformerEncLabelModel": str(Path(inspect.getfile(SortformerEncLabelModel)).resolve()),
        "get_ats_targets": str(Path(inspect.getfile(get_ats_targets)).resolve()),
    }
    checkout = NEMO_CHECKOUT.resolve()
    if any(not Path(path).is_relative_to(checkout) for path in origins.values()):
        raise RuntimeError("loaded NeMo symbol is outside the pinned checkout")
    return {"revision": head, "origins": origins}


def validate(mode: str, expected_image_identity: str | None) -> dict[str, object]:
    configured_identity = os.environ.get("PSEM_CONTAINER_IMAGE_IDENTITY", "").strip()
    if mode == "build" and configured_identity:
        raise RuntimeError("derived build must not claim a runnable manifest identity")
    if mode == "runtime":
        if (
            expected_image_identity is None
            or re.fullmatch(r"sha256:[0-9a-f]{64}", expected_image_identity) is None
        ):
            raise RuntimeError("runtime validation requires a derived manifest digest")
        if configured_identity != expected_image_identity:
            raise RuntimeError("runtime image identity does not match the derived manifest digest")
    base = json.loads(BASE_INVENTORY.read_text(encoding="utf-8"))
    base_packages = base.get("all_packages") if isinstance(base, dict) else None
    base_record_inventory = base.get("all_distribution_records") if isinstance(base, dict) else None
    base_records = (
        base_record_inventory.get("records") if isinstance(base_record_inventory, dict) else None
    )
    if (
        not isinstance(base, dict)
        or base.get("schema_version") != 1
        or base.get("artifact_role") != "ngc_base_inventory"
        or not isinstance(base_packages, dict)
        or any(
            not isinstance(name, str) or not isinstance(version, str)
            for name, version in base_packages.items()
        )
        or not isinstance(base_record_inventory, dict)
        or not isinstance(base_records, list)
        or base_record_inventory.get("count") != len(base_records)
        or any(
            not isinstance(record, dict)
            or not isinstance(record.get("name"), str)
            or not isinstance(record.get("version"), str)
            or not isinstance(record.get("metadata_path"), str)
            for record in base_records
        )
    ):
        raise RuntimeError("NGC base package inventory is invalid")
    base_record_payload = (
        json.dumps(base_records, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    if base_record_inventory.get("sha256") != hashlib.sha256(base_record_payload).hexdigest():
        raise RuntimeError("NGC base distribution-record inventory hash is invalid")
    base_record_versions: dict[str, str] = {}
    for record in base_records:
        name = record["name"]
        version = record["version"]
        if name in base_record_versions and base_record_versions[name] != version:
            raise RuntimeError(f"NGC base distribution name is ambiguous: {name}")
        base_record_versions[name] = version
    if dict(sorted(base_record_versions.items())) != base_packages:
        raise RuntimeError("NGC base package and distribution-record inventories differ")
    runtime_constraints = load_runtime_constraints()
    expected_inventory = dict(base_packages)
    expected_inventory.update(runtime_constraints)
    current_protected = protected_inventory()
    if base.get("inventory") != current_protected:
        raise RuntimeError("NGC torch/CUDA protected inventory drifted during image construction")
    if sha256_file(NVIMGCODEC_WHEEL) != NVIMGCODEC_AFTER_SHA256:
        raise RuntimeError("nvimgcodec WHEEL repair is absent or altered")
    if any(path.exists() for path in DISABLED_METADATA_SOURCES):
        raise RuntimeError("duplicate distribution metadata remains active")
    import torch
    import torchaudio

    if torchaudio.__version__ != "2.6.0a0+d883142":
        raise RuntimeError("torchaudio source build identity is invalid")
    if "soundfile" not in torchaudio.list_audio_backends():
        raise RuntimeError("torchaudio soundfile backend is unavailable")
    nemo = assert_source_identity()
    final_distribution_records = installed_distribution_records()
    final_inventory = dict(sorted(installed_versions(final_distribution_records).items()))
    difference = inventory_difference(expected_inventory, final_inventory)
    if any(difference.values()):
        detail = json.dumps(difference, sort_keys=True, separators=(",", ":"))
        raise RuntimeError(f"final distribution inventory drifted: {detail}")
    actual_runtime_delta = {
        name: version
        for name, version in final_inventory.items()
        if base_packages.get(name) != version
    }
    closure_difference = inventory_difference(runtime_constraints, actual_runtime_delta)
    if any(closure_difference.values()):
        detail = json.dumps(closure_difference, sort_keys=True, separators=(",", ":"))
        raise RuntimeError(f"runtime constraint closure is not exact: {detail}")
    observed = {name: final_inventory.get(name) for name in EXPECTED_PACKAGES}
    if observed != EXPECTED_PACKAGES:
        raise RuntimeError("required compatibility package inventory drifted")
    inventory_payload = (
        json.dumps(
            final_distribution_records,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    pip_check = subprocess.run(
        ["python", "-m", "pip", "check"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    accelerator: dict[str, object] = {
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
    }
    if mode == "runtime":
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("runtime must expose exactly one CUDA accelerator")
        properties = torch.cuda.get_device_properties(0)
        driver = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        if len(driver) != 1 or not driver[0].strip():
            raise RuntimeError("NVIDIA driver identity is unavailable or ambiguous")
        accelerator.update(
            {
                "device_name": properties.name,
                "device_total_memory_bytes": properties.total_memory,
                "nvidia_driver_version": driver[0].strip(),
            }
        )
    return {
        "schema_version": 1,
        "artifact_role": "issue_107_derived_runtime_validation",
        "created_at": datetime.now(UTC).isoformat(),
        "mode": mode,
        "passed": True,
        "base_image": BASE_IMAGE,
        "derived_image_identity": configured_identity or None,
        "contract_activation_ready": mode == "runtime",
        "nemo": nemo,
        "torchaudio_revision": TORCHAUDIO_REVISION,
        "package_versions": observed,
        "runtime_constraints": {
            "count": len(runtime_constraints),
            "verified_delta_count": len(actual_runtime_delta),
            "sha256": sha256_file(RUNTIME_CONSTRAINTS),
        },
        "distribution_inventory": {
            "count": len(final_distribution_records),
            "unique_name_count": len(final_inventory),
            "packages": final_inventory,
            "records": final_distribution_records,
            "sha256": hashlib.sha256(inventory_payload).hexdigest(),
        },
        "protected_inventory": current_protected,
        "accelerator": accelerator,
        "pip_check": pip_check,
        "deterministic_environment": {
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("build", "runtime"), required=True)
    parser.add_argument("--expected-image-identity")
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    value = validate(args.mode, args.expected_image_identity)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(value, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
