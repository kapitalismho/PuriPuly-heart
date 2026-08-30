from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from experiments.psem_training_strategy_gate.data.forced_alignment_reference import (
    validate_reference_checkout,
)

PACKAGE_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = PACKAGE_ROOT.parents[1]
CONTRACT_PATH = PACKAGE_ROOT / "contract.json"
CONFIG_PATH = PACKAGE_ROOT / "config.json"
DATA_SPLIT_RECEIPT_PATH = PACKAGE_ROOT / "data_split_receipt.json"
RUNTIME_CONTRACT_PATH = PACKAGE_ROOT / "runtime_contract.json"
RUNTIME_ENVIRONMENT_PATH = PACKAGE_ROOT / "runtime_environment.json"
EXPECTED_RUNTIME_ENVIRONMENT_SHA256 = (
    "ef988e8934c619619966a38cb6c3ac11a92fae678f19f4bfd2a76ee64f58c27b"
)
PINNED_CONTAINER_IMAGE_IDENTITY = (
    "sha256:0981807f1a51a156563e28b59dc2e7a9b5c1c7d85d1169d4965c5fd91fa38bcb"
)
SOURCE_MANIFEST_PATH = (
    REPOSITORY_ROOT
    / "experiments"
    / "psem_training_strategy_gate"
    / "data"
    / "v2"
    / "source_manifest.jsonl"
)
EXPECTED_CONTRACT_CANONICAL_SHA256 = (
    "6f2d8e6039c0d005668e965cdb7117c8e296b78a2bf9eabb019c8eafefd73374"
)
EXPECTED_CONFIG_CANONICAL_SHA256 = (
    "8a4722d6af4ac72b4a64a03b6df456969047b84712a01c2ed93916c10e34fbea"
)
EXPECTED_DATA_SPLIT_RECEIPT_CANONICAL_SHA256 = (
    "872879a463077416b12527359527b0b74ab609aebddb5631fa7a55afb8592287"
)
EXPECTED_RUNTIME_CONTRACT_CANONICAL_SHA256 = (
    "715fee6d3890cd85e95b573a6c8672bc9bf0b8472765bf980249721ff887e537"
)
EXPECTED_ARTIFACTS = {
    "freeze": (
        "experiments/psem_training_strategy_gate/data/v2/dataset_freeze.json",
        "bc7e63bb201c2a33a9b2d69b2364fed8f03839278098f0bd175d6833b330a41e",
    ),
    "split_manifest": (
        "experiments/psem_training_strategy_gate/data/v2/split_manifest.json",
        "dce084ca8394f70e4f7fe4c72687bbfd95998d26e9ce43e600ef2eb8a65490b4",
    ),
    "source_manifest": (
        "experiments/psem_training_strategy_gate/data/v2/source_manifest.jsonl",
        "76d5a6640ffabbc3cf91c25f5a94284f9869ad266e621ee06f48a987d5d7c6de",
    ),
    "annotation_manifest": (
        "experiments/psem_training_strategy_gate/data/v2/annotation_manifest.jsonl",
        "e2c5b917019000172b08c90d366697d3b5388c4aafbed1b23cd4b8159b6966b2",
    ),
    "topology_manifest": (
        "experiments/psem_training_strategy_gate/data/v2/topology_manifest.jsonl",
        "728c33d17d239dedf08eed9e014cd7e42f4b980c9bcb5b7826c67449f897d7cd",
    ),
    "predecessor_decision": (
        "experiments/psem_frozen_ceiling_gate/results/frozen_ceiling_1/FINAL_DECISION.md",
        "7950943a5ea05f52cea69502a4e243541e3eb1bbc856d55620800d594bd7acf1",
    ),
}
EXPECTED_ROLES = {
    "PSEM-STRATEGY-TRAIN": {"AMI": 50, "AliMeeting": 14},
    "PSEM-STRATEGY-DEV": {"AMI": 7, "AliMeeting": 3},
    "PSEM-STRATEGY-EVAL": {"AMI": 11, "AliMeeting": 8},
}
EXPECTED_SOURCE_COUNT = 93
EXPECTED_CHECKPOINT_SHA256 = "8abd32832159c6ac1148c926b7276f35ba34582c444e559dce1f1253fea42ef8"
EXPECTED_CHECKPOINT_SIZE = 471367680
EXPECTED_REFERENCE = {
    "repository": "https://github.com/nttcslab-sp/diar-forced-alignment",
    "commit": "9527b7c64846fb38316a610f32e9d3466bd6d8b7",
}


class PreflightError(RuntimeError):
    pass


@dataclass(frozen=True)
class PreflightPaths:
    checkpoint: Path | None
    corpus_root: Path | None
    reference_root: Path | None
    output_root: Path | None
    protocol_registry_root: Path | None


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def check(check_id: str, passed: bool, expected: Any, observed: Any) -> dict[str, Any]:
    return {"id": check_id, "passed": bool(passed), "expected": expected, "observed": observed}


def _path(value: Path | None, environment_name: str) -> Path | None:
    raw = value if value is not None else os.environ.get(environment_name)
    return Path(raw).expanduser().resolve() if raw else None


def resolve_paths(
    *,
    checkpoint: Path | None = None,
    corpus_root: Path | None = None,
    reference_root: Path | None = None,
    output_root: Path | None = None,
    protocol_registry_root: Path | None = None,
) -> PreflightPaths:
    return PreflightPaths(
        _path(checkpoint, "PSEM_SORTFORMER_NEMO_PATH"),
        _path(corpus_root, "PSEM_CORPUS_ROOT"),
        _path(reference_root, "PSEM_REFERENCE_ROOT"),
        _path(output_root, "PSEM_ADAPTATION_OUTPUT_ROOT"),
        _path(protocol_registry_root, "PSEM_PROTOCOL_REGISTRY_ROOT"),
    )


def _git_state() -> dict[str, Any]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {"head": head, "dirty": dirty}


def _source_rows() -> list[dict[str, Any]]:
    rows = []
    for line in SOURCE_MANIFEST_PATH.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError("source manifest row must be an object")
        rows.append(row)
    return rows


def static_checks() -> list[dict[str, Any]]:
    contract = load_json(CONTRACT_PATH)
    config = load_json(CONFIG_PATH)
    data_split_receipt = load_json(DATA_SPLIT_RECEIPT_PATH)
    runtime_contract = load_json(RUNTIME_CONTRACT_PATH)
    contract_hash = canonical_sha256(contract)
    config_hash = canonical_sha256(config)
    checks = [
        check(
            "contract.controls_exact",
            contract_hash == EXPECTED_CONTRACT_CANONICAL_SHA256,
            EXPECTED_CONTRACT_CANONICAL_SHA256,
            contract_hash,
        ),
        check(
            "config.controls_exact",
            config_hash == EXPECTED_CONFIG_CANONICAL_SHA256,
            EXPECTED_CONFIG_CANONICAL_SHA256,
            config_hash,
        ),
        check(
            "data_split_receipt.controls_exact",
            canonical_sha256(data_split_receipt) == EXPECTED_DATA_SPLIT_RECEIPT_CANONICAL_SHA256,
            EXPECTED_DATA_SPLIT_RECEIPT_CANONICAL_SHA256,
            canonical_sha256(data_split_receipt),
        ),
        check(
            "runtime_contract.controls_exact",
            canonical_sha256(runtime_contract) == EXPECTED_RUNTIME_CONTRACT_CANONICAL_SHA256,
            EXPECTED_RUNTIME_CONTRACT_CANONICAL_SHA256,
            canonical_sha256(runtime_contract),
        ),
        check(
            "runtime_environment.controls_exact",
            sha256_file(RUNTIME_ENVIRONMENT_PATH) == EXPECTED_RUNTIME_ENVIRONMENT_SHA256,
            EXPECTED_RUNTIME_ENVIRONMENT_SHA256,
            sha256_file(RUNTIME_ENVIRONMENT_PATH),
        ),
    ]
    for artifact_id, (relative_path, expected_hash) in EXPECTED_ARTIFACTS.items():
        path = REPOSITORY_ROOT / relative_path
        observed = sha256_file(path) if path.is_file() else None
        checks.append(
            check(f"artifact.{artifact_id}", observed == expected_hash, expected_hash, observed)
        )
    freeze_path = REPOSITORY_ROOT / EXPECTED_ARTIFACTS["freeze"][0]
    split_path = REPOSITORY_ROOT / EXPECTED_ARTIFACTS["split_manifest"][0]
    freeze = load_json(freeze_path)
    split = load_json(split_path)
    source_rows = _source_rows()
    source_assignments = split.get("assignments", {}).get("sources", [])
    component_assignments = split.get("assignments", {}).get("components", [])
    official_roles = freeze.get("official_roles")
    role_set = set(EXPECTED_ROLES)
    source_roles = [row.get("role") for row in source_assignments]
    component_roles = [row.get("role") for row in component_assignments]
    assignment_source_ids = [row.get("source_id") for row in source_assignments]
    manifest_source_ids = [row.get("source_id") for row in source_rows]
    split_semantics = (
        official_roles == list(EXPECTED_ROLES)
        and len(source_assignments) == EXPECTED_SOURCE_COUNT
        and len(set(assignment_source_ids)) == len(assignment_source_ids)
        and set(assignment_source_ids) == set(manifest_source_ids)
        and len(manifest_source_ids) == EXPECTED_SOURCE_COUNT
        and len(set(manifest_source_ids)) == len(manifest_source_ids)
        and set(source_roles) == role_set
        and set(component_roles) == role_set
        and all(role in role_set for role in source_roles + component_roles)
    )
    checks.append(
        check(
            "dataset.split_roles_exact",
            split_semantics,
            {
                "official_roles": list(EXPECTED_ROLES),
                "source_count": EXPECTED_SOURCE_COUNT,
                "assignment_roles": sorted(EXPECTED_ROLES),
                "source_identity_set": "exact",
            },
            {
                "official_roles": official_roles,
                "source_count": len(source_assignments),
                "source_roles": sorted(set(source_roles), key=str),
                "component_roles": sorted(set(component_roles), key=str),
                "source_identity_set_exact": set(assignment_source_ids) == set(manifest_source_ids),
            },
        )
    )
    observed_roles = {
        role: {
            corpus: sum(
                row.get("role") == role and row.get("corpus") == corpus
                for row in source_assignments
            )
            for corpus in ("AMI", "AliMeeting")
        }
        for role in EXPECTED_ROLES
    }
    checks.append(
        check(
            "dataset.role_counts", observed_roles == EXPECTED_ROLES, EXPECTED_ROLES, observed_roles
        )
    )
    checks.append(
        check(
            "dataset.identity",
            freeze.get("dataset_freeze_id") == "PSEM-STRATEGY-DATA-v2"
            and freeze.get("freeze_status") == "frozen",
            {"id": "PSEM-STRATEGY-DATA-v2", "status": "frozen"},
            {"id": freeze.get("dataset_freeze_id"), "status": freeze.get("freeze_status")},
        )
    )
    return checks


def _safe_external_output_root(path: Path | None) -> bool:
    if path is None or not path.is_absolute() or not path.is_dir():
        return False
    resolved = path.resolve()
    return (
        resolved != Path(resolved.anchor)
        and resolved != REPOSITORY_ROOT
        and not resolved.is_relative_to(REPOSITORY_ROOT)
    )


def _bound_waveform_check(corpus_root: Path | None) -> dict[str, Any]:
    failures: list[str] = []
    verified = 0
    try:
        rows = _source_rows()
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        rows = []
        failures.append(f"{type(exc).__name__}: {exc}")
    if corpus_root is None or not corpus_root.is_dir():
        failures.append("PSEM_CORPUS_ROOT is unavailable")
    else:
        root = corpus_root.resolve()
        for index, row in enumerate(rows):
            source_id = str(row.get("source_id", f"row-{index}"))
            try:
                relative = Path(row["audio_ref"])
                path = (root / relative).resolve()
                valid = (
                    not relative.is_absolute()
                    and ".." not in relative.parts
                    and path.is_relative_to(root)
                    and path.is_file()
                    and path.stat().st_size == row["waveform_size_bytes"]
                    and sha256_file(path) == row["waveform_sha256"]
                )
            except (KeyError, OSError, TypeError, ValueError):
                valid = False
            if valid:
                verified += 1
            else:
                failures.append(source_id)
    return check(
        "runtime.bound_waveforms_exact",
        len(rows) == EXPECTED_SOURCE_COUNT and verified == EXPECTED_SOURCE_COUNT and not failures,
        {"source_count": EXPECTED_SOURCE_COUNT, "byte_identity_verified": True},
        {"source_count": len(rows), "verified": verified, "failures": failures},
    )


def _reference_check(reference_root: Path | None) -> dict[str, Any]:
    try:
        observed = validate_reference_checkout(reference_root) if reference_root else None
    except Exception as exc:
        observed = {"error_type": type(exc).__name__, "error": str(exc)}
    passed = (
        isinstance(observed, dict)
        and observed.get("repository") == EXPECTED_REFERENCE["repository"]
        and observed.get("commit") == EXPECTED_REFERENCE["commit"]
    )
    return check("runtime.reference_checkout_exact", passed, EXPECTED_REFERENCE, observed)


def _runtime_environment_checks() -> list[dict[str, Any]]:
    environment = load_json(RUNTIME_ENVIRONMENT_PATH)
    compute = environment["compute"]
    container_identity = os.environ.get("PSEM_CONTAINER_IMAGE_IDENTITY")
    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if cuda_available else 0
    device_memory = (
        int(torch.cuda.get_device_properties(0).total_memory) if device_count == 1 else None
    )
    torch_version_match = re.match(r"^(\d+)\.(\d+)", torch.__version__)
    torch_version = (
        tuple(int(part) for part in torch_version_match.groups())
        if torch_version_match is not None
        else None
    )
    return [
        check("runtime.architecture", platform.machine() == "x86_64", "x86_64", platform.machine()),
        check(
            "runtime.python_version",
            sys.version_info >= (3, 10),
            ">=3.10",
            platform.python_version(),
        ),
        check(
            "runtime.pytorch_version",
            torch_version is not None and torch_version >= (2, 5),
            ">=2.5",
            torch.__version__,
        ),
        check(
            "runtime.cuda_device_count",
            cuda_available and device_count == compute["accelerator_count"],
            compute["accelerator_count"],
            device_count,
        ),
        check(
            "runtime.cuda_device_memory",
            device_memory is not None and device_memory > 0,
            {"requirement": "one CUDA accelerator with positive reported VRAM"},
            {"bytes": device_memory},
        ),
        check(
            "runtime.container_image_identity",
            container_identity == PINNED_CONTAINER_IMAGE_IDENTITY,
            PINNED_CONTAINER_IMAGE_IDENTITY,
            container_identity,
        ),
    ]


def _material_execution_policy() -> tuple[str | None, str | None]:
    try:
        runtime_contract = load_json(RUNTIME_CONTRACT_PATH)
    except (OSError, json.JSONDecodeError):
        return None, None
    if not isinstance(runtime_contract, dict):
        return None, None
    material_execution = runtime_contract.get("material_execution")
    if not isinstance(material_execution, dict):
        return None, None
    status = material_execution.get("status")
    required = material_execution.get("required_status_for_material_execution")
    return (
        status if isinstance(status, str) else None,
        required if isinstance(required, str) else None,
    )


def require_material_execution_ready() -> None:
    status, required = _material_execution_policy()
    if required != "ready" or status != required:
        observed = status if status is not None else "unavailable"
        raise PreflightError(
            f"material execution is not authorized: expected='ready', observed={observed!r}"
        )


def runtime_checks(paths: PreflightPaths) -> list[dict[str, Any]]:
    checkpoint_exists = paths.checkpoint is not None and paths.checkpoint.is_file()
    observed_hash = sha256_file(paths.checkpoint) if checkpoint_exists else None
    observed_size = paths.checkpoint.stat().st_size if checkpoint_exists else None
    execution_status, required_execution_status = _material_execution_policy()
    return [
        check(
            "runtime.material_execution_authorized",
            required_execution_status == "ready" and execution_status == "ready",
            "ready",
            execution_status,
        ),
        check("runtime.platform", platform.system() == "Linux", "Linux", platform.system()),
        check(
            "runtime.checkpoint_path",
            checkpoint_exists,
            "existing regular file",
            str(paths.checkpoint) if paths.checkpoint else None,
        ),
        check(
            "runtime.checkpoint_sha256",
            observed_hash == EXPECTED_CHECKPOINT_SHA256,
            EXPECTED_CHECKPOINT_SHA256,
            observed_hash,
        ),
        check(
            "runtime.checkpoint_size",
            observed_size == EXPECTED_CHECKPOINT_SIZE,
            EXPECTED_CHECKPOINT_SIZE,
            observed_size,
        ),
        check(
            "runtime.corpus_root",
            paths.corpus_root is not None and paths.corpus_root.is_dir(),
            "existing directory",
            str(paths.corpus_root) if paths.corpus_root else None,
        ),
        check(
            "runtime.reference_root",
            paths.reference_root is not None and paths.reference_root.is_dir(),
            "existing directory",
            str(paths.reference_root) if paths.reference_root else None,
        ),
        check(
            "runtime.output_root",
            _safe_external_output_root(paths.output_root),
            "existing absolute non-root directory outside the repository",
            str(paths.output_root) if paths.output_root else None,
        ),
        check(
            "runtime.protocol_registry_root",
            _safe_external_output_root(paths.protocol_registry_root)
            and paths.output_root is not None
            and paths.protocol_registry_root.resolve() != paths.output_root.resolve(),
            "existing absolute non-root directory outside the repository and output root",
            str(paths.protocol_registry_root) if paths.protocol_registry_root else None,
        ),
        check(
            "runtime.eval_sealed",
            os.environ.get("PSEM_ALLOW_EVAL") is None,
            "PSEM_ALLOW_EVAL absent",
            os.environ.get("PSEM_ALLOW_EVAL"),
        ),
        _bound_waveform_check(paths.corpus_root),
        _reference_check(paths.reference_root),
        *_runtime_environment_checks(),
    ]


def build_preflight(paths: PreflightPaths, *, static_only: bool = False) -> dict[str, Any]:
    git = _git_state()
    checks = static_checks()
    checks.append(check("git.worktree_clean", not git["dirty"], [], git["dirty"]))
    if not static_only:
        checks.extend(runtime_checks(paths))
    all_passed = all(row["passed"] for row in checks)
    payload = {
        "schema_version": 1,
        "artifact_role": "psem_sortformer_adaptation_preflight",
        "mode": "static" if static_only else "runtime",
        "binding": {
            "git_head": git["head"],
            "contract_canonical_sha256": canonical_sha256(load_json(CONTRACT_PATH)),
            "config_canonical_sha256": canonical_sha256(load_json(CONFIG_PATH)),
        },
        "checks": checks,
        "ready_for_runtime_audit": not static_only and all_passed,
        "static_contract_valid": all_passed
        if static_only
        else all(row["passed"] for row in checks if not row["id"].startswith("runtime.")),
    }
    return {**payload, "payload_sha256": canonical_sha256(payload)}
